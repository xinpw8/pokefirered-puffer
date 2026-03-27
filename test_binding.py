#!/usr/bin/env python3
"""
test_binding.py -- Smoke test for PufferFireRed binding
Verifies: SO instance creation, vec_init, vec_reset, vec_step, obs integrity
"""
import sys
import os
import struct
import numpy as np

# Ensure local binding is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch.distributed must be imported before dlopen-based SO copies
import torch.distributed  # noqa: F401

import binding

# Constants from pfr.h
OBS_SIZE = 115473
NUM_ACTIONS = 8
NUM_ENVS = 2
FRAMES_PER_STEP = 4
MAX_STEPS = 24576

# Obs offsets from pfr.h
OFF_SCALAR = 0
OFF_NPC = 55
OFF_VISITED = 145
OFF_PIXELS = 273
PIXEL_SIZE = 240 * 160 * 3  # 115200

NATIVE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "pokefirered-native"))
SO_PATH = os.path.join(NATIVE_DIR, "build", "libpfr_game.so")
TMP_DIR = "/tmp/pfr_test_instances"

def parse_scalar_obs(buf):
    """Parse PfrScalarObs (55 bytes, packed little-endian)"""
    # player_x(i16) player_y(i16) map_group(u8) map_num(u8) map_layout_id(u8)
    # direction(u8) avatar_flags(u8) running_state(u8) transition_state(u8)
    # in_battle(u8) battle_outcome(u8)
    # party[6]: species(u16) level(u8) hp_pct(u8) status(u8) type1(u8) = 6 bytes each
    # badges(u8) money(u16) weather(u8) step_counter(u16)
    d = {}
    d["player_x"], d["player_y"] = struct.unpack_from("<hh", buf, 0)
    d["map_group"] = buf[4]
    d["map_num"] = buf[5]
    d["map_layout_id"] = buf[6]
    d["direction"] = buf[7]
    d["avatar_flags"] = buf[8]
    d["running_state"] = buf[9]
    d["transition_state"] = buf[10]
    d["in_battle"] = buf[11]
    d["battle_outcome"] = buf[12]

    d["party"] = []
    for i in range(6):
        off = 13 + i * 6
        species, = struct.unpack_from("<H", buf, off)
        level = buf[off + 2]
        hp_pct = buf[off + 3]
        status = buf[off + 4]
        type1 = buf[off + 5]
        d["party"].append({
            "species": species, "level": level, "hp_pct": hp_pct,
            "status": status, "type1": type1
        })

    off = 13 + 36  # = 49
    d["badges"] = buf[off]
    d["money"], = struct.unpack_from("<H", buf, off + 1)
    d["weather"] = buf[off + 3]
    d["step_counter"], = struct.unpack_from("<H", buf, off + 4)
    return d

def parse_npc_obs(buf, offset=55, max_npcs=15):
    """Parse PfrNpcObs[15] (90 bytes)"""
    npcs = []
    for i in range(max_npcs):
        off = offset + i * 6
        dx = struct.unpack_from("<b", buf, off)[0]
        dy = struct.unpack_from("<b", buf, off + 1)[0]
        gfx = buf[off + 2]
        direction = buf[off + 3]
        active = buf[off + 4]
        movement = buf[off + 5]
        npcs.append({"dx": dx, "dy": dy, "gfx_id": gfx,
                      "dir": direction, "active": active, "move": movement})
    return npcs

def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    print(f"=== PFR Binding Smoke Test ===")
    print(f"SO path: {SO_PATH}")
    print(f"OBS_SIZE={OBS_SIZE}, NUM_ACTIONS={NUM_ACTIONS}, NUM_ENVS={NUM_ENVS}")

    assert os.path.exists(SO_PATH), f"libpfr_game.so not found at {SO_PATH}"

    # 1. Create instances
    print("\n[1] Creating SO instances...")
    binding.init_instances(SO_PATH, TMP_DIR, NUM_ENVS)
    print("    OK")

    # 2. Allocate buffers
    print("[2] Allocating buffers...")
    observations = np.zeros((NUM_ENVS, OBS_SIZE), dtype=np.uint8)
    actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
    rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    terminals = np.zeros(NUM_ENVS, dtype=np.uint8)
    truncations = np.zeros(NUM_ENVS, dtype=np.uint8)
    print(f"    obs shape={observations.shape}, actions shape={actions.shape}")

    # 3. vec_init
    print("[3] Calling vec_init...")
    c_envs = binding.vec_init(
        observations, actions, rewards, terminals, truncations,
        NUM_ENVS, 42,
        frames_per_step=FRAMES_PER_STEP,
        max_steps=MAX_STEPS,
        savestate_path="",
    )
    print(f"    OK, handle={c_envs}")

    # 4. vec_reset
    print("[4] Calling vec_reset...")
    binding.vec_reset(c_envs, 42)
    print("    OK")

    # 5. Verify observations after reset
    print("\n[5] Verifying observations after reset...")
    for env_idx in range(NUM_ENVS):
        obs = observations[env_idx]
        scalar = parse_scalar_obs(obs)
        print(f"\n  --- Env {env_idx} ---")
        print(f"  Player: ({scalar["player_x"]}, {scalar["player_y"]})")
        print(f"  Map: group={scalar["map_group"]} num={scalar["map_num"]} layout={scalar["map_layout_id"]}")
        print(f"  Direction: {scalar["direction"]}  In battle: {scalar["in_battle"]}")
        print(f"  Badges: {bin(scalar["badges"])}  Money: {scalar["money"]}")
        print(f"  Step counter: {scalar["step_counter"]}")

        # Party
        for i, p in enumerate(scalar["party"]):
            if p["species"] > 0:
                print(f"  Party[{i}]: species={p["species"]} lv={p["level"]} "
                      f"hp%={p["hp_pct"]} status={p["status"]} type={p["type1"]}")

        # Verify constraints
        assert 0 <= scalar["direction"] <= 3, f"Bad direction: {scalar["direction"]}"
        assert scalar["in_battle"] in (0, 1), f"Bad in_battle: {scalar["in_battle"]}"
        assert scalar["money"] <= 65535, f"Bad money: {scalar["money"]}"

        # NPCs
        npcs = parse_npc_obs(obs)
        active_npcs = sum(1 for n in npcs if n["active"])
        print(f"  Active NPCs: {active_npcs}")

        # Visited tiles (should have at least 1 bit set after reset)
        visited = obs[OFF_VISITED:OFF_VISITED + 128]
        visit_bits = sum(bin(b).count("1") for b in visited)
        print(f"  Visited tile bits: {visit_bits}")

        # Pixels (should not be all zeros if framebuffer works)
        pixels = obs[OFF_PIXELS:OFF_PIXELS + PIXEL_SIZE]
        pixel_sum = int(np.sum(pixels.astype(np.uint64)))
        pixel_nonzero = int(np.count_nonzero(pixels))
        print(f"  Pixels: sum={pixel_sum}, nonzero={pixel_nonzero}/{PIXEL_SIZE}")
        if pixel_nonzero == 0:
            print("  WARNING: All pixels are zero! Framebuffer may not be rendering.")

    # 6. Step a few times
    print("\n[6] Stepping 10 times...")
    for step in range(10):
        actions[:] = np.random.randint(0, NUM_ACTIONS, size=(NUM_ENVS, 1)).astype(np.float32)
        binding.vec_step(c_envs)

    print("    OK, no crashes")
    print(f"    Rewards after 10 steps: {rewards}")
    print(f"    Terminals: {terminals}")

    # 7. Verify obs after stepping
    print("\n[7] Post-step obs verification...")
    for env_idx in range(NUM_ENVS):
        obs = observations[env_idx]
        scalar = parse_scalar_obs(obs)
        print(f"  Env {env_idx}: pos=({scalar["player_x"]},{scalar["player_y"]}) "
              f"dir={scalar["direction"]} step={scalar["step_counter"]}")
        pixels = obs[OFF_PIXELS:OFF_PIXELS + PIXEL_SIZE]
        print(f"    Pixels nonzero: {np.count_nonzero(pixels)}/{PIXEL_SIZE}")

    # 8. Log
    print("\n[8] Checking vec_log...")
    log = binding.vec_log(c_envs)
    print(f"    Log: {log}")

    # 9. Cleanup
    print("\n[9] Cleaning up...")
    binding.vec_close(c_envs)
    binding.destroy_instances()
    print("    OK")

    print("\n=== SMOKE TEST PASSED ===")

if __name__ == "__main__":
    main()
