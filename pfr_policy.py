"""
pfr_policy.py -- PufferLib 4.0 policy network for PokemonFireRed

Observation layout (115,473 bytes, uint8 flat buffer):
  [0:55]          PfrScalarObs       (55 bytes)
  [55:145]        PfrNpcObs[15]      (15 NPCs x 6 bytes = 90 bytes)
  [145:273]       Visited tiles      (128 bytes = 1024 bits)
  [273:115473]    Pixels             (240x160 RGB = 115,200 bytes)

Modes:
  use_pixels=True   Full CNN pipeline (~14.6M params with LSTM)
  use_pixels=False   Lite mode: scalar+NPC+tiles only (~0.5M params with LSTM)
                    Much faster forward pass, higher SPS for exploration training.

Architecture:
  Scalar branch:  embeddings + normalize + bitfield unpack -> MLP
  NPC branch:     per-NPC embed + normalize -> MLP -> masked max-pool
  Visit branch:   128 bytes -> 1024-dim binary float -> MLP
  Pixel branch:   (3,160,240) -> CNN -> 512-dim  [only if use_pixels=True]
  Fusion:         concat all -> Linear -> LayerNorm -> ReLU -> hidden_size
  Heads:          actor (8 actions), critic (1 value)
  LSTM:           via pufferlib.models.LSTMWrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models
from pufferlib.pytorch import layer_init

# ── Observation layout constants (must match pfr.h) ──

SCALAR_SIZE = 55
NPC_SIZE_PER = 6
NPC_COUNT = 15
NPC_TOTAL = NPC_SIZE_PER * NPC_COUNT       # 90
VISIT_BYTES = 128                           # 1024 bits
PIXEL_BYTES = 240 * 160 * 3                 # 115,200

SCALAR_START = 0
SCALAR_END = SCALAR_SIZE                    # 55
NPC_START = SCALAR_END                      # 55
NPC_END = NPC_START + NPC_TOTAL             # 145
VISIT_START = NPC_END                       # 145
VISIT_END = VISIT_START + VISIT_BYTES       # 273
PIXEL_START = VISIT_END                     # 273
PIXEL_END = PIXEL_START + PIXEL_BYTES       # 115,473

OBS_SIZE = PIXEL_END                        # 115,473
NUM_ACTIONS = 8

# Embedding table sizes (upper bounds with headroom)
NUM_SPECIES = 512
NUM_MAP_GROUP = 64
NUM_MAP_NUM = 256
NUM_MAP_LAYOUT = 256
NUM_DIRECTION = 8
NUM_RUNNING = 8
NUM_TRANSITION = 8
NUM_BATTLE_OUTCOME = 8
NUM_WEATHER = 16
NUM_TYPE = 32
NUM_NPC_GFX = 256
NUM_NPC_MOVE = 32

SCREEN_H = 160
SCREEN_W = 240
MAX_STEPS = 24576


def _unpack_int16(obs, lo_idx, hi_idx):
    """Reconstruct signed int16 from two uint8 bytes (little-endian), batch-wise."""
    raw = obs[:, lo_idx].to(torch.int32) | (obs[:, hi_idx].to(torch.int32) << 8)
    return ((raw + 0x8000) % 0x10000 - 0x8000).float()


def _unpack_uint16(obs, lo_idx, hi_idx):
    """Reconstruct uint16 from two uint8 bytes (little-endian), batch-wise."""
    return (obs[:, lo_idx].to(torch.int32) | (obs[:, hi_idx].to(torch.int32) << 8)).float()


def _unpack_bits(byte_tensor, num_bits=8):
    """Unpack a byte tensor [B] into [B, num_bits] binary float vector."""
    return torch.stack([(byte_tensor >> i) & 1 for i in range(num_bits)], dim=-1).float()


class PFRNPolicy(nn.Module):
    """
    PufferLib 4.0 compatible policy for Pokemon FireRed RL.

    Args:
        use_pixels: If False (default), skip CNN branch for much faster training.
                    Lite mode uses only scalars + NPCs + visited tiles (273 bytes).
    """

    def __init__(self, env, hidden_size=256, embed_dim=16,
                 npc_hidden=32, visit_hidden=64, pixel_hidden=512,
                 use_pixels=False, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.use_pixels = use_pixels

        # ── Scalar embeddings ──
        self.species_embed = nn.Embedding(NUM_SPECIES + 1, embed_dim, padding_idx=0)
        self.map_group_embed = nn.Embedding(NUM_MAP_GROUP, embed_dim)
        self.map_num_embed = nn.Embedding(NUM_MAP_NUM, embed_dim)
        self.map_layout_embed = nn.Embedding(NUM_MAP_LAYOUT, embed_dim)
        self.direction_embed = nn.Embedding(NUM_DIRECTION, 8)
        self.running_embed = nn.Embedding(NUM_RUNNING, 4)
        self.transition_embed = nn.Embedding(NUM_TRANSITION, 4)
        self.battle_outcome_embed = nn.Embedding(NUM_BATTLE_OUTCOME, 4)
        self.weather_embed = nn.Embedding(NUM_WEATHER, 8)
        self.type_embed = nn.Embedding(NUM_TYPE, 8, padding_idx=0)

        party_per_mon = embed_dim + 1 + 1 + 8 + 8
        scalar_total = (
            2                               # player_x, player_y
            + 3 * embed_dim                 # map embeddings
            + 8                             # direction embedding
            + 8                             # avatar_flags bitfield
            + 4                             # running_state embedding
            + 4                             # transition_state embedding
            + 1                             # in_battle
            + 4                             # battle_outcome embedding
            + 6 * party_per_mon             # 6 party members
            + 8                             # badges bitfield
            + 1                             # money (normalized)
            + 8                             # weather embedding
            + 1                             # step_counter (normalized)
        )

        self.scalar_mlp = nn.Sequential(
            layer_init(nn.Linear(scalar_total, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size // 2)),
            nn.ReLU(),
        )

        # ── NPC branch ──
        npc_input_dim = 1 + 1 + embed_dim + 8 + 1 + 4
        self.npc_gfx_embed = nn.Embedding(NUM_NPC_GFX, embed_dim)
        self.npc_dir_embed = nn.Embedding(NUM_DIRECTION, 8)
        self.npc_move_embed = nn.Embedding(NUM_NPC_MOVE, 4)

        self.npc_mlp = nn.Sequential(
            layer_init(nn.Linear(npc_input_dim, npc_hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(npc_hidden, npc_hidden)),
            nn.ReLU(),
        )

        # ── Visited tiles branch ──
        self.visit_mlp = nn.Sequential(
            layer_init(nn.Linear(1024, visit_hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(visit_hidden, visit_hidden)),
            nn.ReLU(),
        )

        # ── Pixel branch (only if enabled) ──
        fusion_input = hidden_size // 2 + npc_hidden + visit_hidden
        if use_pixels:
            self.pixel_cnn = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            cnn_flat_size = 64 * 16 * 26  # after conv layers on (3, 160, 240)
            self.pixel_fc = nn.Sequential(
                layer_init(nn.Linear(cnn_flat_size, pixel_hidden)),
                nn.ReLU(),
            )
            fusion_input += pixel_hidden

        # ── Fusion ──
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(fusion_input, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # ── Output heads ──
        self.actor = layer_init(nn.Linear(hidden_size, NUM_ACTIONS), std=0.01)
        self.value_fn = layer_init(nn.Linear(hidden_size, 1), std=1)

    def encode_observations(self, obs, state=None):
        """Parse raw uint8 obs and encode into hidden feature vector.

        Args:
            obs: [B, OBS_SIZE] uint8 tensor
            state: ignored (PufferLib LSTMWrapper passes this kwarg)

        Returns:
            features: [B, hidden_size] float tensor
        """
        B = obs.shape[0]

        scalar_raw = obs[:, SCALAR_START:SCALAR_END]
        npc_raw = obs[:, NPC_START:NPC_END]
        visit_raw = obs[:, VISIT_START:VISIT_END]

        # ════ Scalar branch ════
        player_x = _unpack_int16(scalar_raw, 0, 1) / 256.0
        player_y = _unpack_int16(scalar_raw, 2, 3) / 256.0

        map_group = scalar_raw[:, 4].long().clamp(0, NUM_MAP_GROUP - 1)
        map_num = scalar_raw[:, 5].long().clamp(0, NUM_MAP_NUM - 1)
        map_layout = scalar_raw[:, 6].long().clamp(0, NUM_MAP_LAYOUT - 1)
        direction = scalar_raw[:, 7].long().clamp(0, NUM_DIRECTION - 1)
        avatar_flags_bits = _unpack_bits(scalar_raw[:, 8])
        running_state = scalar_raw[:, 9].long().clamp(0, NUM_RUNNING - 1)
        transition_state = scalar_raw[:, 10].long().clamp(0, NUM_TRANSITION - 1)
        in_battle = scalar_raw[:, 11].float().clamp(0, 1)
        battle_outcome = scalar_raw[:, 12].long().clamp(0, NUM_BATTLE_OUTCOME - 1)

        party_features = []
        for i in range(6):
            base = 13 + i * 6
            species = _unpack_uint16(scalar_raw, base, base + 1).long().clamp(0, NUM_SPECIES)
            sp_emb = self.species_embed(species)
            level = scalar_raw[:, base + 2].float() / 100.0
            hp_pct = scalar_raw[:, base + 3].float() / 255.0
            status_bits = _unpack_bits(scalar_raw[:, base + 4])
            type1 = scalar_raw[:, base + 5].long().clamp(0, NUM_TYPE - 1)
            type_emb = self.type_embed(type1)
            party_features.append(torch.cat([
                sp_emb, level.unsqueeze(1), hp_pct.unsqueeze(1),
                status_bits, type_emb,
            ], dim=1))

        party_cat = torch.cat(party_features, dim=1)
        badges_bits = _unpack_bits(scalar_raw[:, 49])
        money = _unpack_uint16(scalar_raw, 50, 51) / 65535.0
        weather = scalar_raw[:, 52].long().clamp(0, NUM_WEATHER - 1)
        step_counter = _unpack_uint16(scalar_raw, 53, 54) / float(MAX_STEPS)

        scalar_input = torch.cat([
            player_x.unsqueeze(1), player_y.unsqueeze(1),
            self.map_group_embed(map_group),
            self.map_num_embed(map_num),
            self.map_layout_embed(map_layout),
            self.direction_embed(direction),
            avatar_flags_bits,
            self.running_embed(running_state),
            self.transition_embed(transition_state),
            in_battle.unsqueeze(1),
            self.battle_outcome_embed(battle_outcome),
            party_cat,
            badges_bits,
            money.unsqueeze(1),
            self.weather_embed(weather),
            step_counter.unsqueeze(1),
        ], dim=1)

        scalar_out = self.scalar_mlp(scalar_input)

        # ════ NPC branch ════
        npc_reshaped = npc_raw.view(B, NPC_COUNT, NPC_SIZE_PER)
        npc_dx = npc_reshaped[:, :, 0].to(torch.int8).float() / 127.0
        npc_dy = npc_reshaped[:, :, 1].to(torch.int8).float() / 127.0
        npc_gfx = npc_reshaped[:, :, 2].long().clamp(0, NUM_NPC_GFX - 1)
        npc_dir = npc_reshaped[:, :, 3].long().clamp(0, NUM_DIRECTION - 1)
        npc_active = npc_reshaped[:, :, 4].float()
        npc_move = npc_reshaped[:, :, 5].long().clamp(0, NUM_NPC_MOVE - 1)

        npc_input = torch.cat([
            npc_dx.unsqueeze(2), npc_dy.unsqueeze(2),
            self.npc_gfx_embed(npc_gfx),
            self.npc_dir_embed(npc_dir),
            npc_active.unsqueeze(2),
            self.npc_move_embed(npc_move),
        ], dim=2)

        npc_features = self.npc_mlp(npc_input)
        npc_mask = npc_active.unsqueeze(2)
        npc_masked = npc_features * npc_mask + (1 - npc_mask) * (-1e9)
        npc_pooled = npc_masked.max(dim=1)[0]
        all_inactive = (npc_active.sum(dim=1) == 0).unsqueeze(1)
        npc_pooled = npc_pooled * (~all_inactive).float()

        # ════ Visited tiles branch ════
        visit_bytes = visit_raw.long()
        visit_bits = torch.stack(
            [(visit_bytes >> i) & 1 for i in range(8)], dim=2
        ).float()
        visit_flat = visit_bits.reshape(B, 1024)
        visit_out = self.visit_mlp(visit_flat)

        # ════ Pixel branch (optional) ════
        if self.use_pixels:
            pixel_raw = obs[:, PIXEL_START:PIXEL_END]
            pixels = pixel_raw.float().view(B, SCREEN_H, SCREEN_W, 3)
            pixels = pixels.permute(0, 3, 1, 2) / 255.0
            pixel_conv_out = self.pixel_cnn(pixels)
            pixel_out = self.pixel_fc(pixel_conv_out)
            fused = torch.cat([scalar_out, npc_pooled, visit_out, pixel_out], dim=1)
        else:
            fused = torch.cat([scalar_out, npc_pooled, visit_out], dim=1)

        return self.fusion(fused)

    def decode_actions(self, hidden, lookup=None):
        logits = self.actor(hidden)
        value = self.value_fn(hidden)
        return logits, value

    def forward(self, x, state=None):
        hidden = self.encode_observations(x)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)


class PFRNLSTM(pufferlib.models.LSTMWrapper):
    """LSTM wrapper for PFRNPolicy. PufferLib handles the LSTM stepping.

    Compatible with both PufferLib API variants:
      - Spark 4.0: LSTMWrapper(env, make_policy_fn, hidden_size, ...)
      - Installed:  LSTMWrapper(env, policy, input_size, hidden_size)
    """
    def __init__(self, env, policy_or_fn, hidden_size=256, **kwargs):
        import inspect
        if callable(policy_or_fn) and not isinstance(policy_or_fn, nn.Module):
            super().__init__(env, policy_or_fn, hidden_size=hidden_size, **kwargs)
        else:
            super().__init__(env, policy_or_fn, input_size=hidden_size, hidden_size=hidden_size)
