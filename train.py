#!/usr/bin/env python3
"""
pokefirered_puffer -- PufferLib 4.0 training for Pokemon FireRed (native C)

Primary metric: exploration heatmap (unique tiles visited globally).
Records: heatmap progression GIF, agent view GIF.

Usage:
    python train.py                              # train with defaults (lite, 32 envs)
    python train.py --num-envs 64                # more parallelism
    python train.py --use-pixels                 # full CNN mode (slower)
    python train.py --wandb                      # enable wandb logging
    python train.py --load-model-path X.pt       # resume from checkpoint
"""

import argparse
import json
import os
import sys
import time

# Ensure local dir takes priority for imports (pfr_policy.py, binding.so)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# pokefirered-native is a sibling directory
PFRN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'pokefirered-native')
PFRN_DIR = os.path.normpath(PFRN_DIR)
if PFRN_DIR not in sys.path:
    sys.path.append(PFRN_DIR)

# Use development pufferlib (has WandbLogger, full PuffeRL API)
PUFFERLIB_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'pufferlib'))
if os.path.isdir(PUFFERLIB_DIR) and PUFFERLIB_DIR not in sys.path:
    sys.path.insert(1, PUFFERLIB_DIR)

import numpy as np
import torch
import torch.distributed  # noqa: F401  -- must import before dlopen SO copies

import pufferlib
import pufferlib.vector
import pufferlib.pytorch
from pufferlib.pufferl import PuffeRL, WandbLogger, NoLogger

import binding
from pfr_policy import PFRNPolicy, PFRNLSTM, OBS_SIZE, LITE_OBS_SIZE, NUM_ACTIONS

# Map data for heatmap
MAP_DATA_PATH = os.path.join(PFRN_DIR, 'pfr_map_data.json')

# Shared state between env and main loop (numpy arrays are by-reference)
_shared = {}


# ── Map utilities ──────────────────────────────────────────────

def load_map_data():
    with open(MAP_DATA_PATH, 'r') as f:
        data = json.load(f)
    global_shape = tuple(data['global_map_shape'])
    regions = {r['id']: r for r in data['regions'] if r['id'] >= 0}
    pad = 20
    padded_shape = (global_shape[0] + pad * 2, global_shape[1] + pad * 2)
    return regions, padded_shape, pad


def local_to_global(y, x, map_group, map_num, regions, pad):
    map_id = map_group * 256 + map_num
    region = regions.get(map_id)
    if region is None:
        return -1, -1
    gx = x + region['coordinates'][0] + pad
    gy = y + region['coordinates'][1] + pad
    return gy, gx


# ── GIF utilities ──────────────────────────────────────────────

def heatmap_to_rgb(heatmap, scale=3, pad=5):
    """Crop heatmap to visited area, colorize binary, and scale up."""
    mask = heatmap > 0
    if not mask.any():
        return np.zeros((64, 64, 3), dtype=np.uint8)
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, heatmap.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, heatmap.shape[1])
    crop = heatmap[y0:y1, x0:x1]
    # Log-scale: makes low-count tiles visible, high-count tiles brighter
    log_map = np.where(crop > 0, np.log1p(crop), 0)
    norm = (log_map / max(log_map.max(), 1) * 200 + 55).astype(np.uint8)
    norm[crop == 0] = 0
    h, w = norm.shape
    # Dark gray background (10) so unvisited areas are distinguishable
    rgb = np.full((h, w, 3), 10, dtype=np.uint8)
    rgb[crop > 0, 0] = 0   # no red for visited
    rgb[crop > 0, 1] = norm[crop > 0]  # green = log visit count
    rgb[crop > 0, 2] = 30  # slight blue tint for visited
    rgb[crop == 0] = 10  # dark gray background
    # Scale up
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def framebuf_to_rgb(argb_array):
    """(H,W,4) BGRA uint8 from capture_frame -> (H,W,3) RGB."""
    return argb_array[:, :, [2, 1, 0]].copy()


def save_gif(frames, path, duration=100):
    from PIL import Image
    if not frames:
        return
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)


def save_png(rgb, path):
    from PIL import Image
    Image.fromarray(rgb).save(path)


MAX_GIF_FRAMES = 300  # cap to avoid OOM on long runs


def thin_frames(frames):
    """Keep every other frame when list exceeds MAX_GIF_FRAMES.
    Preserves first and last frame so GIF shows full progression."""
    if len(frames) <= MAX_GIF_FRAMES:
        return frames
    thinned = frames[::2]
    if thinned[-1] is not frames[-1]:
        thinned.append(frames[-1])
    return thinned


# ── Training Environment ──────────────────────────────────────

class PFRNTraining(pufferlib.PufferEnv):
    """PufferEnv for training with built-in heatmap tracking.

    Uses PFRN C binding directly. Tracks exploration heatmap across
    all envs and episodes for monitoring.
    """

    def __init__(self, num_envs=1, frames_per_step=4, max_steps=24576,
                 log_interval=128, seed=0, use_pixels=0, **kwargs):
        import gymnasium
        self.num_agents = num_envs
        obs_size = OBS_SIZE if use_pixels else LITE_OBS_SIZE
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        super().__init__()

        self._log_interval = log_interval
        self._use_pixels = use_pixels
        os.makedirs('/tmp/pfrn_instances', exist_ok=True)
        so_path = os.path.join(PFRN_DIR, 'build', 'libpfr_game.so')

        binding.init_instances(so_path, '/tmp/pfrn_instances', num_envs)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            num_envs, seed,
            frames_per_step=frames_per_step,
            max_steps=max_steps,
            savestate_path='',
            use_pixels=use_pixels,
        )
        self._tick = 0

        # Heatmap tracking
        self._regions, self._padded_shape, self._pad = load_map_data()
        self._heatmap = np.zeros(self._padded_shape, dtype=np.float32)
        _shared['heatmap'] = self._heatmap
        _shared['padded_shape'] = self._padded_shape

    def reset(self, seed=0):
        self.rewards.fill(0)
        binding.vec_reset(self.c_envs, seed)
        self._update_heatmap()
        return self.observations, [{}]

    def step(self, actions):
        self.rewards.fill(0)
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self._tick += 1

        # Update heatmap (lightweight: 6 bytes per env)
        self._update_heatmap()

        log = {}
        if self._tick % self._log_interval == 0:
            log = binding.vec_log(self.c_envs)
            log['heatmap_tiles'] = int(np.count_nonzero(self._heatmap))

        return self.observations, self.rewards, self.terminals, self.truncations, [log]

    def _update_heatmap(self):
        for i in range(self.num_agents):
            obs = self.observations[i]
            px = int(np.frombuffer(obs[0:2], dtype=np.int16)[0])
            py = int(np.frombuffer(obs[2:4], dtype=np.int16)[0])
            mg, mn = int(obs[4]), int(obs[5])
            gy, gx = local_to_global(py, px, mg, mn,
                                     self._regions, self._pad)
            if gy >= 0 and gx >= 0:
                self._heatmap[gy, gx] += 1.0

    def render(self):
        pass

    def close(self):
        binding.vec_close(self.c_envs)
        binding.destroy_instances()


# ── Args ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='PufferLib training for Pokemon FireRed (native)')

    # Environment
    p.add_argument('--num-envs', type=int, default=32,
                   help='Parallel game instances (default: 32 for high SPS)')
    p.add_argument('--frames-per-step', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=24576)
    p.add_argument('--log-interval', type=int, default=128)
    p.add_argument('--use-pixels', action='store_true', default=False,
                   help='Enable pixel CNN (slower, larger model)')

    # Policy
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--embed-dim', type=int, default=16)

    # Training
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--total-timesteps', type=int, default=100_000_000)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--bptt-horizon', type=int, default=32)
    p.add_argument('--minibatch-size', type=int, default=None)
    p.add_argument('--max-minibatch-size', type=int, default=32768)
    p.add_argument('--update-epochs', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=2.5e-4)
    p.add_argument('--anneal-lr', type=bool, default=True)
    p.add_argument('--min-lr-ratio', type=float, default=0.0)
    p.add_argument('--gamma', type=float, default=0.998)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-coef', type=float, default=0.2)
    p.add_argument('--vf-coef', type=float, default=1.0)
    p.add_argument('--vf-clip-coef', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--optimizer', type=str, default='adam',
                   choices=['adam', 'muon'])
    p.add_argument('--adam-beta1', type=float, default=0.9)
    p.add_argument('--adam-beta2', type=float, default=0.999)
    p.add_argument('--adam-eps', type=float, default=1e-5)
    p.add_argument('--vtrace-rho-clip', type=float, default=1.0)
    p.add_argument('--vtrace-c-clip', type=float, default=1.0)
    p.add_argument('--prio-alpha', type=float, default=0.8)
    p.add_argument('--prio-beta0', type=float, default=0.2)

    # Infrastructure
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--torch-deterministic', type=bool, default=True)
    p.add_argument('--cpu-offload', type=bool, default=False)
    p.add_argument('--precision', type=str, default='float32')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--compile-mode', type=str, default='default')
    p.add_argument('--use-rnn', action='store_true', default=True)
    p.add_argument('--no-rnn', dest='use_rnn', action='store_false')
    p.add_argument('--checkpoint-interval', type=int, default=200)
    p.add_argument('--data-dir', type=str, default='experiments')

    # GIF recording
    p.add_argument('--gif-interval', type=int, default=50,
                   help='Epochs between GIF frame captures')
    p.add_argument('--heatmap-print-interval', type=int, default=10,
                   help='Epochs between heatmap progress prints')

    # Logging
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='pokefirered_puffer')
    p.add_argument('--wandb-group', type=str, default='debug')

    # Resume
    p.add_argument('--load-model-path', type=str, default=None)

    return p.parse_args()


def make_config(args):
    batch_size = args.batch_size or (args.num_envs * args.bptt_horizon)
    minibatch_size = args.minibatch_size or batch_size

    return {
        'env': 'pokefirered_puffer',
        'seed': args.seed,
        'torch_deterministic': args.torch_deterministic,
        'cpu_offload': args.cpu_offload,
        'device': args.device,
        'precision': args.precision,
        'total_timesteps': args.total_timesteps,
        'batch_size': batch_size,
        'bptt_horizon': args.bptt_horizon,
        'minibatch_size': minibatch_size,
        'max_minibatch_size': args.max_minibatch_size,
        'update_epochs': args.update_epochs,
        'use_rnn': args.use_rnn,
        'compile': args.compile,
        'compile_mode': args.compile_mode,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'anneal_lr': args.anneal_lr,
        'min_lr_ratio': args.min_lr_ratio,
        'adam_beta1': args.adam_beta1,
        'adam_beta2': args.adam_beta2,
        'adam_eps': args.adam_eps,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_coef': args.clip_coef,
        'vf_coef': args.vf_coef,
        'vf_clip_coef': args.vf_clip_coef,
        'ent_coef': args.ent_coef,
        'max_grad_norm': args.max_grad_norm,
        'vtrace_rho_clip': args.vtrace_rho_clip,
        'vtrace_c_clip': args.vtrace_c_clip,
        'prio_alpha': args.prio_alpha,
        'prio_beta0': args.prio_beta0,
        'data_dir': args.data_dir,
        'checkpoint_interval': args.checkpoint_interval,
    }


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    gif_dir = os.path.join(args.data_dir, 'gifs')
    os.makedirs(gif_dir, exist_ok=True)

    if args.compile:
        ptxas = '/usr/local/cuda/bin/ptxas'
        if os.path.exists(ptxas):
            os.environ.setdefault('TRITON_PTXAS_PATH', ptxas)

    # ── Environment ──
    use_pixels_int = 1 if args.use_pixels else 0
    vecenv = pufferlib.vector.make(
        PFRNTraining,
        env_kwargs=dict(
            num_envs=args.num_envs,
            frames_per_step=args.frames_per_step,
            max_steps=args.max_steps,
            log_interval=args.log_interval,
            seed=args.seed,
            use_pixels=use_pixels_int,
        ),
        backend=pufferlib.vector.PufferEnv,
    )

    # ── Policy ──
    policy = PFRNPolicy(
        vecenv,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
        use_pixels=args.use_pixels,
    )
    if args.use_rnn:
        policy = PFRNLSTM(vecenv, policy,
                          input_size=args.hidden_size,
                          hidden_size=args.hidden_size)
    policy = policy.to(args.device)

    param_count = sum(p.numel() for p in policy.parameters())
    mode = 'FULL (pixels+CNN)' if args.use_pixels else 'LITE (scalar+NPC+tiles)'
    print(f'Policy: {mode}, {param_count/1e6:.1f}M params')

    # ── Resume ──
    if args.load_model_path:
        state_dict = torch.load(args.load_model_path, map_location=args.device,
                                weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        print(f'Loaded checkpoint: {args.load_model_path}')

    # ── Logger ──
    logger = None
    if args.wandb:
        wandb_args = {
            'train': make_config(args),
            'wandb_project': args.wandb_project,
            'wandb_group': args.wandb_group,
            'tag': None,
            'no_model_upload': False,
        }
        logger = WandbLogger(wandb_args)

    # ── Train ──
    config = make_config(args)
    pufferl = PuffeRL(config, vecenv, policy, logger)

    print(f'Training: {args.num_envs} envs, '
          f'batch={config["batch_size"]}, '
          f'bptt={config["bptt_horizon"]}, '
          f'rnn={args.use_rnn}, '
          f'device={args.device}')
    print(f'GIF recording: every {args.gif_interval} epochs')
    print()

    # ── GIF recording state ──
    heatmap_frames = []
    view_frames = []
    last_heatmap_tiles = 0
    epoch = 0
    t0 = time.time()

    try:
        while pufferl.global_step < config['total_timesteps']:
            if config['device'] == 'cuda':
                torch.compiler.cudagraph_mark_step_begin()
            pufferl.evaluate()
            if config['device'] == 'cuda':
                torch.compiler.cudagraph_mark_step_begin()
            pufferl.train()
            epoch += 1

            # ── Heatmap progress (primary metric) ──
            heatmap = _shared.get('heatmap')
            if heatmap is not None and epoch % args.heatmap_print_interval == 0:
                tiles = int(np.count_nonzero(heatmap))
                delta = tiles - last_heatmap_tiles
                elapsed = time.time() - t0
                sps = pufferl.global_step / max(elapsed, 0.001)
                print(f'[epoch {epoch:5d} | step {pufferl.global_step:9d} | '
                      f'{sps:.0f} SPS] '
                      f'Heatmap: {tiles} tiles (+{delta})')
                last_heatmap_tiles = tiles

            # ── GIF frame capture ──
            if epoch % args.gif_interval == 0:
                if heatmap is not None:
                    heatmap_frames.append(heatmap_to_rgb(heatmap))
                    if len(heatmap_frames) > MAX_GIF_FRAMES:
                        heatmap_frames = thin_frames(heatmap_frames)
                try:
                    frame = binding.capture_frame(0)
                    view_frames.append(framebuf_to_rgb(frame))
                    if len(view_frames) > MAX_GIF_FRAMES:
                        view_frames = thin_frames(view_frames)
                except Exception:
                    pass

            # ── Save GIFs at checkpoints ──
            if epoch % config['checkpoint_interval'] == 0:
                if heatmap_frames:
                    save_gif(heatmap_frames,
                             os.path.join(gif_dir, 'heatmap.gif'), 150)
                if view_frames:
                    save_gif(view_frames,
                             os.path.join(gif_dir, 'agent_view.gif'), 50)
                if heatmap is not None:
                    save_png(heatmap_to_rgb(heatmap),
                             os.path.join(gif_dir, 'heatmap_latest.png'))

    except KeyboardInterrupt:
        print('\nTraining interrupted.')

    # ── Final save ──
    model_path = pufferl.close()
    if logger:
        logger.close(model_path, early_stop=False)

    # Save final GIFs
    if heatmap_frames:
        save_gif(heatmap_frames, os.path.join(gif_dir, 'heatmap_final.gif'), 150)
        print(f'Heatmap GIF: {gif_dir}/heatmap_final.gif '
              f'({len(heatmap_frames)} frames)')
    if view_frames:
        save_gif(view_frames, os.path.join(gif_dir, 'agent_view_final.gif'), 50)
        print(f'Agent view GIF: {gif_dir}/agent_view_final.gif '
              f'({len(view_frames)} frames)')

    heatmap = _shared.get('heatmap')
    if heatmap is not None:
        tiles = int(np.count_nonzero(heatmap))
        save_png(heatmap_to_rgb(heatmap),
                 os.path.join(gif_dir, 'heatmap_final.png'))
        print(f'Final heatmap: {tiles} unique tiles')

    print(f'Model saved: {model_path}')


if __name__ == '__main__':
    main()
