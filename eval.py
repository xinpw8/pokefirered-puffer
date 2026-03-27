#!/usr/bin/env python3
"""
eval.py -- PufferFireRed evaluation + GIF recording

Primary metric: exploration heatmap (unique tiles visited).
Generates:
  - heatmap_progression.gif: animated heatmap growth over time
  - agent_view.gif: 1st person framebuffer captures
  - heatmap_final.png: final exploration heatmap
  - eval_stats.json: per-episode stats

Usage:
    python eval.py                                   # random policy, 1 episode
    python eval.py --checkpoint experiments/X.pt     # trained policy
    python eval.py --num-episodes 5 --num-envs 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

PFRN_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'pokefirered-native'))
if PFRN_DIR not in sys.path:
    sys.path.append(PFRN_DIR)

# Use development pufferlib (matching train.py)
PUFFERLIB_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'pufferlib'))
if os.path.isdir(PUFFERLIB_DIR) and PUFFERLIB_DIR not in sys.path:
    sys.path.insert(1, PUFFERLIB_DIR)

import binding
from pfr_policy import PFRNPolicy, PFRNLSTM, OBS_SIZE, NUM_ACTIONS

# Obs layout (from pfr.h)
VISIT_START = 145
VISIT_END = 273

# Map data for global coordinate conversion
MAP_DATA_PATH = os.path.join(PFRN_DIR, 'pfr_map_data.json')


# ── Map utilities ──────────────────────────────────────────────

def load_map_data():
    """Load map region data for local-to-global coordinate conversion."""
    with open(MAP_DATA_PATH, 'r') as f:
        data = json.load(f)
    global_shape = tuple(data['global_map_shape'])
    regions = {r['id']: r for r in data['regions'] if r['id'] >= 0}
    pad = 20
    padded_shape = (global_shape[0] + pad * 2, global_shape[1] + pad * 2)
    return regions, padded_shape, pad


def local_to_global(y, x, map_group, map_num, regions, pad):
    """Convert local map coords to global heatmap coords."""
    map_id = map_group * 256 + map_num
    region = regions.get(map_id)
    if region is None:
        return -1, -1
    gx = x + region['coordinates'][0] + pad
    gy = y + region['coordinates'][1] + pad
    return gy, gx


def parse_position(obs):
    """Extract player position and map from obs bytes."""
    px = int(np.frombuffer(obs[0:2], dtype=np.int16)[0])
    py = int(np.frombuffer(obs[2:4], dtype=np.int16)[0])
    mg = int(obs[4])
    mn = int(obs[5])
    return px, py, mg, mn


def count_visited_bits(obs):
    """Count set bits in the visited tiles portion of obs."""
    visit_bytes = obs[VISIT_START:VISIT_END]
    return sum(bin(b).count('1') for b in visit_bytes)


# ── Image utilities ────────────────────────────────────────────

def heatmap_to_rgb(heatmap):
    """Convert float heatmap to RGB uint8 image (green channel)."""
    if heatmap.max() > 0:
        norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(heatmap, dtype=np.uint8)
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 1] = norm                                    # green
    rgb[..., 2] = (norm.astype(np.float32) * 0.3).astype(np.uint8)  # slight blue
    return rgb


def framebuf_to_rgb(argb_array):
    """Convert (H, W, 4) BGRA uint8 (from capture_frame) to (H, W, 3) RGB.

    On little-endian ARM, uint32 ARGB in memory is [B, G, R, A].
    """
    return argb_array[:, :, [2, 1, 0]].copy()


def save_gif(frames, path, duration=100):
    """Save list of RGB numpy arrays as animated GIF via PIL."""
    from PIL import Image
    if not frames:
        return False
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)
    return True


def save_png(rgb_array, path):
    """Save RGB numpy array as PNG."""
    from PIL import Image
    Image.fromarray(rgb_array).save(path)


# ── Policy setup ───────────────────────────────────────────────

def make_policy(args, device):
    """Create policy and optionally load checkpoint."""
    if not args.checkpoint:
        return None, None

    import torch
    import gymnasium

    class DummyEnv:
        def __init__(self):
            self.single_observation_space = gymnasium.spaces.Box(
                low=0, high=255, shape=(OBS_SIZE,), dtype=np.uint8)
            self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)

    dummy = DummyEnv()
    policy = PFRNPolicy(dummy, hidden_size=args.hidden_size,
                        embed_dim=args.embed_dim,
                        use_pixels=args.use_pixels)
    if args.use_rnn:
        policy = PFRNLSTM(dummy, policy,
                          hidden_size=args.hidden_size)

    state_dict = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # LSTM hidden state
    rnn_state = None
    if args.use_rnn:
        rnn_state = (
            torch.zeros(1, args.num_envs, args.hidden_size, device=device),
            torch.zeros(1, args.num_envs, args.hidden_size, device=device),
        )

    return policy, rnn_state


# ── Main eval loop ─────────────────────────────────────────────

def run_eval(args):
    regions, padded_shape, pad = load_map_data()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    policy, rnn_state = make_policy(args, device)

    # Init C envs
    so_path = os.path.join(PFRN_DIR, 'build', 'libpfr_game.so')
    os.makedirs('/tmp/pfrn_instances', exist_ok=True)

    num_envs = args.num_envs
    binding.init_instances(so_path, '/tmp/pfrn_instances', num_envs)

    obs_buf = np.zeros((num_envs, OBS_SIZE), dtype=np.uint8)
    act_buf = np.zeros((num_envs, 1), dtype=np.int32)
    rew_buf = np.zeros(num_envs, dtype=np.float32)
    term_buf = np.zeros(num_envs, dtype=np.uint8)
    trunc_buf = np.zeros(num_envs, dtype=np.uint8)

    c_envs = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        num_envs, args.seed,
        frames_per_step=4, max_steps=args.max_steps,
        savestate_path='',
        use_pixels=1,  # always extract pixels for eval (agent view GIF)
    )
    binding.vec_reset(c_envs, args.seed)

    # Tracking
    heatmap = np.zeros(padded_shape, dtype=np.float32)
    heatmap_frames = []
    agent_view_frames = []
    episode_stats = []

    episodes_done = 0
    env_returns = np.zeros(num_envs, dtype=np.float32)
    env_steps = np.zeros(num_envs, dtype=np.int32)

    total_steps = 0
    last_heatmap_tiles = 0

    print(f'Eval: {num_envs} envs, max_steps={args.max_steps}, '
          f'policy={"checkpoint" if policy else "random"}, '
          f'target={args.num_episodes} episodes')
    print()

    t0 = time.time()
    while episodes_done < args.num_episodes:
        # Choose action
        if policy:
            import torch
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_buf.copy()).to(device)
                if rnn_state is not None:
                    out = policy(obs_t, rnn_state)
                    logits, value, rnn_state = out[0], out[1], out[2]
                else:
                    logits, value = policy(obs_t)
                actions = torch.argmax(logits, dim=-1).cpu().numpy()
        else:
            actions = np.random.randint(0, NUM_ACTIONS, num_envs)

        act_buf[:, 0] = actions.astype(np.int32)
        rew_buf.fill(0)
        binding.vec_step(c_envs)
        total_steps += 1

        # Update heatmap from obs positions
        for i in range(num_envs):
            px, py, mg, mn = parse_position(obs_buf[i])
            gy, gx = local_to_global(py, px, mg, mn, regions, pad)
            if gy >= 0 and gx >= 0:
                heatmap[gy, gx] += 1.0
            env_returns[i] += rew_buf[i]
            env_steps[i] += 1

        # Heatmap snapshot for GIF
        if total_steps % args.heatmap_interval == 0:
            current_tiles = int(np.count_nonzero(heatmap))
            delta = current_tiles - last_heatmap_tiles
            elapsed = time.time() - t0
            sps = total_steps * num_envs / max(elapsed, 0.001)
            print(f'  [step {total_steps:6d}] '
                  f'Heatmap: {current_tiles} tiles (+{delta})  '
                  f'SPS: {sps:.0f}')
            last_heatmap_tiles = current_tiles
            heatmap_frames.append(heatmap_to_rgb(heatmap))

        # Agent view capture for GIF (env 0 only)
        if total_steps % args.view_interval == 0:
            try:
                frame = binding.capture_frame(0)
                agent_view_frames.append(framebuf_to_rgb(frame))
            except Exception:
                pass  # render might fail in headless

        # Check terminals
        for i in range(num_envs):
            if term_buf[i]:
                episodes_done += 1
                tiles = count_visited_bits(obs_buf[i])
                stat = {
                    'episode': episodes_done,
                    'tiles_visited': tiles,
                    'return': float(env_returns[i]),
                    'steps': int(env_steps[i]),
                }
                episode_stats.append(stat)
                print(f'  Episode {episodes_done}: '
                      f'tiles={tiles}, '
                      f'return={env_returns[i]:.2f}, '
                      f'steps={env_steps[i]}')

                # Reset LSTM state for this env
                if rnn_state is not None:
                    rnn_state[0][:, i, :] = 0
                    rnn_state[1][:, i, :] = 0

                env_returns[i] = 0.0
                env_steps[i] = 0

                if episodes_done >= args.num_episodes:
                    break

    # ── Final output ──
    elapsed = time.time() - t0
    final_tiles = int(np.count_nonzero(heatmap))
    print(f'\n{"=" * 60}')
    print(f'FINAL: {final_tiles} unique global tiles '
          f'across {episodes_done} episodes '
          f'({elapsed:.1f}s)')
    print(f'{"=" * 60}')

    # Save heatmap progression GIF
    if heatmap_frames:
        heatmap_frames.append(heatmap_to_rgb(heatmap))  # final frame
        gif_path = os.path.join(args.output_dir, 'heatmap_progression.gif')
        save_gif(heatmap_frames, gif_path, duration=200)
        print(f'Heatmap GIF: {gif_path} ({len(heatmap_frames)} frames)')

    # Save agent view GIF
    if agent_view_frames:
        gif_path = os.path.join(args.output_dir, 'agent_view.gif')
        save_gif(agent_view_frames, gif_path, duration=50)
        print(f'Agent view GIF: {gif_path} ({len(agent_view_frames)} frames)')

    # Save final heatmap PNG
    hm_path = os.path.join(args.output_dir, 'heatmap_final.png')
    save_png(heatmap_to_rgb(heatmap), hm_path)
    print(f'Heatmap PNG: {hm_path}')

    # Save stats
    stats_path = os.path.join(args.output_dir, 'eval_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'total_steps': total_steps,
            'num_episodes': episodes_done,
            'final_heatmap_tiles': final_tiles,
            'elapsed_seconds': elapsed,
            'episodes': episode_stats,
        }, f, indent=2)
    print(f'Stats: {stats_path}')

    binding.vec_close(c_envs)
    binding.destroy_instances()


def main():
    p = argparse.ArgumentParser(description='PFR Evaluation + GIF Recording')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to .pt model checkpoint (omit for random policy)')
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--num-episodes', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=24576)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--embed-dim', type=int, default=16)
    p.add_argument('--use-rnn', action='store_true', default=True)
    p.add_argument('--no-rnn', dest='use_rnn', action='store_false')
    p.add_argument('--use-pixels', action='store_true', default=False,
                   help='Include pixel obs in policy (default: lite mode)')
    p.add_argument('--output-dir', type=str, default='eval_output')
    p.add_argument('--heatmap-interval', type=int, default=500,
                   help='Steps between heatmap GIF snapshots')
    p.add_argument('--view-interval', type=int, default=100,
                   help='Steps between agent view GIF captures')
    args = p.parse_args()
    run_eval(args)


if __name__ == '__main__':
    main()
