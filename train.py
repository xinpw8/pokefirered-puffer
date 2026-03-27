#!/usr/bin/env python3
"""
pokefirered_puffer -- PufferLib 4.0 training for Pokemon FireRed (native C)

Uses PFRN (native vectorized PufferEnv) + PFRNPolicy/PFRNLSTM with the
real PufferLib PuffeRL training runner. No PyBoy, no train_quick.py.

Usage:
    python train.py                          # train with defaults
    python train.py --num-envs 16            # 16 parallel envs
    python train.py --wandb                  # enable wandb logging
    python train.py --device cpu             # CPU-only training
    python train.py --load-model-path X.pt   # resume from checkpoint
"""

import argparse
import os
import sys
import time

# pokefirered-native is a sibling directory -- add it to the import path
PFRN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'pokefirered-native')
PFRN_DIR = os.path.normpath(PFRN_DIR)
if PFRN_DIR not in sys.path:
    sys.path.insert(0, PFRN_DIR)

import numpy as np
import torch
# torch.distributed MUST be imported before dlopen-based SO-copy instances
# are created. It initializes internal state in libtorch that prevents the
# dlopen'd game SOs from corrupting the process during vec_step/vec_reset.
import torch.distributed  # noqa: F401

import pufferlib
import pufferlib.vector
import pufferlib.pytorch
from pufferlib.pufferl import PuffeRL, WandbLogger, NoLogger

from pfrn import PFRN, binding, OBS_SIZE, NUM_ACTIONS
from pfrn_policy import PFRNPolicy, PFRNLSTM


class PFRNTraining(pufferlib.PufferEnv):
    """Minimal PufferEnv for training that uses the PFRN C binding directly.

    Avoids the explore_map and .copy() overhead from the base PFRN class.
    The dlmopen'd SO copies conflict with pufferl's ~200 extension modules
    in the Python _update_explore_map path, so we skip all of that here.
    """

    def __init__(self, num_envs=1, frames_per_step=4, max_steps=24576,
                 log_interval=128, seed=0, **kwargs):
        import gymnasium
        self.num_agents = num_envs
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(OBS_SIZE,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        super().__init__()

        self._log_interval = log_interval
        os.makedirs('/tmp/pfrn_instances', exist_ok=True)
        so_path = os.path.join(PFRN_DIR, 'build', 'libpfr_game.so')
        tmp_dir = '/tmp/pfrn_instances'

        binding.init_instances(so_path, tmp_dir, num_envs)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            num_envs, seed,
            frames_per_step=frames_per_step,
            max_steps=max_steps,
            savestate_path='',
        )
        self._tick = 0

    def reset(self, seed=0):
        self.rewards.fill(0)
        binding.vec_reset(self.c_envs, seed)
        return self.observations, [{}]

    def step(self, actions):
        self.rewards.fill(0)
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self._tick += 1
        log = binding.vec_log(self.c_envs) if self._tick % self._log_interval == 0 else {}
        return self.observations, self.rewards, self.terminals, self.truncations, [log]

    def render(self):
        pass

    def close(self):
        binding.vec_close(self.c_envs)
        binding.destroy_instances()


def parse_args():
    p = argparse.ArgumentParser(description='PufferLib training for Pokemon FireRed (native)')

    # Environment
    p.add_argument('--num-envs', type=int, default=8,
                   help='Number of parallel game instances (vectorized in C)')
    p.add_argument('--frames-per-step', type=int, default=4,
                   help='GBA frames per RL step')
    p.add_argument('--max-steps', type=int, default=24576,
                   help='Max steps per episode before truncation')
    p.add_argument('--log-interval', type=int, default=128,
                   help='Steps between env-level stat logging')

    # Policy
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--embed-dim', type=int, default=16)
    p.add_argument('--tile-channels', type=int, default=32)
    p.add_argument('--npc-hidden', type=int, default=32)

    # Training
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--total-timesteps', type=int, default=100_000_000)
    p.add_argument('--batch-size', type=int, default=None,
                   help='Batch size (default: num_envs * bptt_horizon)')
    p.add_argument('--bptt-horizon', type=int, default=32,
                   help='BPTT unroll length')
    p.add_argument('--minibatch-size', type=int, default=None,
                   help='PPO minibatch size (default: batch_size)')
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

    # V-Trace / prioritized experience
    p.add_argument('--vtrace-rho-clip', type=float, default=1.0)
    p.add_argument('--vtrace-c-clip', type=float, default=1.0)
    p.add_argument('--prio-alpha', type=float, default=0.8)
    p.add_argument('--prio-beta0', type=float, default=0.2)

    # Infrastructure
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--torch-deterministic', type=bool, default=True)
    p.add_argument('--cpu-offload', type=bool, default=False)
    p.add_argument('--precision', type=str, default='float32')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the policy')
    p.add_argument('--compile-mode', type=str, default='default')
    p.add_argument('--use-rnn', action='store_true', default=True,
                   help='Wrap policy with LSTM (default: on)')
    p.add_argument('--no-rnn', dest='use_rnn', action='store_false')
    p.add_argument('--checkpoint-interval', type=int, default=200)
    p.add_argument('--data-dir', type=str, default='experiments')

    # Logging
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='pokefirered_puffer')
    p.add_argument('--wandb-group', type=str, default='debug')

    # Resume
    p.add_argument('--load-model-path', type=str, default=None,
                   help='Path to .pt checkpoint to resume from')

    return p.parse_args()


def make_config(args):
    """Build the config dict that PuffeRL expects."""
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

    if args.compile:
        ptxas = '/usr/local/cuda/bin/ptxas'
        if os.path.exists(ptxas):
            os.environ.setdefault('TRITON_PTXAS_PATH', ptxas)

    # -- Environment --
    vecenv = pufferlib.vector.make(
        PFRNTraining,
        env_kwargs=dict(
            num_envs=args.num_envs,
            frames_per_step=args.frames_per_step,
            max_steps=args.max_steps,
            log_interval=args.log_interval,
            seed=args.seed,
        ),
        backend=pufferlib.vector.PufferEnv,
    )

    # -- Policy --
    policy = PFRNPolicy(
        vecenv,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
        tile_channels=args.tile_channels,
        npc_hidden=args.npc_hidden,
    )
    if args.use_rnn:
        policy = PFRNLSTM(vecenv, policy,
                          input_size=args.hidden_size,
                          hidden_size=args.hidden_size)
    policy = policy.to(args.device)

    # -- Resume from checkpoint --
    if args.load_model_path:
        state_dict = torch.load(args.load_model_path, map_location=args.device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        print(f'Loaded checkpoint: {args.load_model_path}')

    # -- Logger --
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

    # -- Train --
    config = make_config(args)
    pufferl = PuffeRL(config, vecenv, policy, logger)

    print(f'Training pokefirered_puffer: {args.num_envs} envs, '
          f'batch={config["batch_size"]}, '
          f'bptt={config["bptt_horizon"]}, '
          f'rnn={args.use_rnn}, '
          f'device={args.device}')

    try:
        while pufferl.global_step < config['total_timesteps']:
            if config['device'] == 'cuda':
                torch.compiler.cudagraph_mark_step_begin()
            pufferl.evaluate()
            if config['device'] == 'cuda':
                torch.compiler.cudagraph_mark_step_begin()
            pufferl.train()
    except KeyboardInterrupt:
        print('\nTraining interrupted.')

    model_path = pufferl.close()
    if logger:
        logger.close(model_path, early_stop=False)
    print(f'Model saved: {model_path}')


if __name__ == '__main__':
    main()
