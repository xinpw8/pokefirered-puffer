/*
 * pfr.h -- PufferFireRed: PufferLib ocean env for Pokemon FireRed
 *
 * Pure C environment following PufferLib 4.0 ocean conventions.
 * All game logic accessed via SO-copy instances (dlopen'd libpfr_game.so).
 * Each instance has isolated globals — safe for parallel vectorized training.
 *
 * Observation layout (flat uint8):
 *   [0..54]                    PfrScalarObs     (55 bytes)
 *   [55..144]                  PfrNpcObs[15]    (90 bytes)
 *   [145..272]                 visited_tiles    (128 bytes = 1024 bits)
 *   [273..115472]              pixels           (240*160*3 = 115200 bytes RGB)
 *   Total: 115473 bytes
 *
 * Action space: Discrete(8)
 *   0=noop 1=up 2=down 3=left 4=right 5=A 6=B 7=start
 *
 * Preprocessing (policy network, not in this file):
 *   - Categorical fields: embedding lookup (species, map, direction, etc.)
 *   - Normalized fields: divide by max (level/100, hp/255, money/65535)
 *   - Bitfields: multi-hot float vector (badges, status, avatar_flags)
 *   - Pixels: normalize to [0,1], feed through CNN
 *   - Visited tiles: binary float vector (1024 dims)
 *   - NPC dx/dy: normalize to [-1,1] (divide by 127)
 */

#ifndef PFR_H
#define PFR_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* Include from pokefirered-native/src/ (added to include path by build) */
#include "pfr_so_instance.h"

/* ================================================================
 * Observation dimensions
 * ================================================================ */

#define PFR_TILE_RADIUS       4
#define PFR_TILE_DIM          (2 * PFR_TILE_RADIUS + 1)  /* 9 */

#define PFR_MAX_PARTY         6
#define PFR_MAX_NPCS          15
#define PFR_NPC_FEATURES      6

#define PFR_NUM_BADGES        8
#define PFR_NUM_ACTIONS       8

#define PFR_FRAMES_PER_STEP   4
#define PFR_MAX_STEPS         24576

/* Screen dimensions (GBA native) */
#define PFR_SCREEN_W          240
#define PFR_SCREEN_H          160
#define PFR_SCREEN_CHANNELS   3   /* RGB */
#define PFR_PIXEL_SIZE        (PFR_SCREEN_W * PFR_SCREEN_H * PFR_SCREEN_CHANNELS)

/* Visited tiles bitfield: 1024 bits = 128 bytes */
#define PFR_VISIT_HASH_SIZE   1024
#define PFR_VISIT_BYTES       (PFR_VISIT_HASH_SIZE / 8)  /* 128 */

/* ================================================================
 * Observation structs (packed, little-endian)
 *
 * Preprocessing guide:
 *   int16  player_x/y    -> embed (categorical within map bounds)
 *   uint8  map_group     -> embed (categorical)
 *   uint8  map_num       -> embed (categorical)
 *   uint8  map_layout_id -> embed (categorical)
 *   uint8  direction     -> embed (categorical, 4 values: 0=D 1=U 2=L 3=R)
 *   uint8  avatar_flags  -> bitfield -> multi-hot float
 *   uint8  running_state -> embed (categorical)
 *   uint8  transition    -> embed (categorical)
 *   uint8  in_battle     -> binary float (0.0 or 1.0)
 *   uint8  battle_outcome-> embed (categorical)
 *   uint16 species       -> embed (categorical, 0=empty, 1-386=pokemon)
 *   uint8  level         -> normalize /100
 *   uint8  hp_pct        -> normalize /255
 *   uint8  status        -> bitfield -> multi-hot float
 *   uint8  type1         -> embed (categorical, 18 types)
 *   uint8  badges        -> bitfield -> 8-dim multi-hot float
 *   uint16 money         -> normalize /65535
 *   uint8  weather       -> embed (categorical)
 *   uint16 step_counter  -> normalize /max_steps
 * ================================================================ */

typedef struct __attribute__((packed)) {
    /* Player location (4 bytes) */
    int16_t  player_x;
    int16_t  player_y;

    /* Map identity (3 bytes) */
    uint8_t  map_group;
    uint8_t  map_num;
    uint8_t  map_layout_id;

    /* Player state (4 bytes) */
    uint8_t  player_direction;      /* embed: 0=D 1=U 2=L 3=R */
    uint8_t  player_avatar_flags;   /* bitfield -> multi-hot */
    uint8_t  player_running_state;  /* embed */
    uint8_t  player_transition_state; /* embed */

    /* Game mode flags (2 bytes) */
    uint8_t  in_battle;             /* binary 0/1 */
    uint8_t  battle_outcome;        /* embed */

    /* Party summary: 6 mons x 6 bytes = 36 bytes */
    struct __attribute__((packed)) {
        uint16_t species;           /* embed: 0=empty, 1-386=pokemon */
        uint8_t  level;             /* normalize /100 */
        uint8_t  hp_pct;            /* normalize /255 */
        uint8_t  status;            /* bitfield -> multi-hot */
        uint8_t  type1;             /* embed: 18 types */
    } party[PFR_MAX_PARTY];

    /* Badges (1 byte) */
    uint8_t  badges;                /* bitfield -> 8-dim multi-hot */

    /* Money (2 bytes, capped at 65535) */
    uint16_t money;                 /* normalize /65535 */

    /* Weather (1 byte) */
    uint8_t  weather;               /* embed */

    /* Step counter (2 bytes) */
    uint16_t step_counter;          /* normalize /max_steps */

} PfrScalarObs;
/* 4 + 3 + 4 + 2 + 36 + 1 + 2 + 1 + 2 = 55 bytes */

typedef struct __attribute__((packed)) {
    int8_t   dx;                    /* normalize /127 -> [-1,1] */
    int8_t   dy;                    /* normalize /127 -> [-1,1] */
    uint8_t  graphics_id;           /* embed (categorical sprite type) */
    uint8_t  direction;             /* embed (4 dirs) */
    uint8_t  active;                /* binary 0/1 (mask for valid NPCs) */
    uint8_t  movement_type;         /* embed (categorical) */
} PfrNpcObs;
/* 6 bytes per NPC, 15 NPCs = 90 bytes */

/* Observation byte offsets */
#define PFR_OFF_SCALAR      0
#define PFR_OFF_NPC         55
#define PFR_OFF_VISITED     145
#define PFR_OFF_PIXELS      273

#define PFR_SCALAR_OBS_SIZE sizeof(PfrScalarObs)
#define PFR_NPC_OBS_SIZE    (PFR_MAX_NPCS * sizeof(PfrNpcObs))
#define PFR_OBS_SIZE        (PFR_OFF_PIXELS + PFR_PIXEL_SIZE)  /* 115473 */

/* ================================================================
 * Exploration tracking
 * ================================================================ */

typedef struct {
    uint32_t visit_hash[PFR_VISIT_HASH_SIZE / 32];
    uint32_t visit_count;
    uint32_t prev_visit_count;

    uint8_t  prev_badges;
    uint8_t  prev_party_count;
    uint16_t prev_party_level_sum;
    uint32_t prev_money;

    uint64_t visited_maps;  /* bitfield: map-transition bonus */

    uint32_t step_count;
    float    episode_return;
} PfrRewardState;

/* ================================================================
 * Log struct (PufferLib: ALL floats, last field = n)
 * ================================================================ */

typedef struct PfrLog PfrLog;
struct PfrLog {
    float episode_return;
    float episode_length;
    float badges;
    float exploration;
    float party_level_sum;
    float n;
};

/* Reward info from game API */
typedef struct {
    int16_t  player_x;
    int16_t  player_y;
    uint8_t  map_group;
    uint8_t  map_num;
    uint8_t  badges;
    uint8_t  party_count;
    uint16_t party_level_sum;
    uint32_t money;
    uint8_t  in_battle;
} PfrRewardInfo;

/* ================================================================
 * Main env struct (PufferLib ocean pattern)
 * ================================================================ */

typedef struct PfrEnv PfrEnv;
struct PfrEnv {
    PfrLog log;                        /* MUST be first (env_binding.h) */

    unsigned char *observations;       /* PufferLib-managed buffer */
    int           *actions;            /* PufferLib-managed buffer (float32) */
    float         *rewards;            /* PufferLib-managed buffer */
    unsigned char *terminals;          /* PufferLib-managed buffer */

    PfrRewardState reward_state;
    PfrInstance    *instance;          /* game SO-copy instance */

    uint32_t max_steps;
    uint32_t frames_per_step;
    char     savestate_path[512];
    uint8_t  use_pixels;             /* skip pfr_extract_pixels when 0 */

    /* Global exploration (persists across episodes, NOT reset) */
    uint32_t global_visit_hash[PFR_VISIT_HASH_SIZE / 32];
    uint32_t global_visit_count;
};

/* ================================================================
 * GBA button defs (active-high)
 * ================================================================ */

#define PFR_BTN_A       (1 << 0)
#define PFR_BTN_B       (1 << 1)
#define PFR_BTN_START   (1 << 3)
#define PFR_BTN_RIGHT   (1 << 4)
#define PFR_BTN_LEFT    (1 << 5)
#define PFR_BTN_UP      (1 << 6)
#define PFR_BTN_DOWN    (1 << 7)

static const uint16_t sPfrActionToButtons[PFR_NUM_ACTIONS] = {
    [0] = 0,              /* noop */
    [1] = PFR_BTN_UP,
    [2] = PFR_BTN_DOWN,
    [3] = PFR_BTN_LEFT,
    [4] = PFR_BTN_RIGHT,
    [5] = PFR_BTN_A,
    [6] = PFR_BTN_B,
    [7] = 0,              /* START disabled: prevents menu trap */
};

/* ================================================================
 * Exploration hash
 * ================================================================ */

static uint32_t pfr_tile_hash(uint8_t mg, uint8_t mn, int16_t x, int16_t y) {
    uint32_t h = (uint32_t)mg * 31 + (uint32_t)mn;
    h = h * 2654435761u + (uint32_t)(uint16_t)x;
    h = h * 2654435761u + (uint32_t)(uint16_t)y;
    return h % PFR_VISIT_HASH_SIZE;
}

static bool pfr_visit_check_and_set(PfrRewardState *rs, uint8_t mg, uint8_t mn,
                                     int16_t x, int16_t y) {
    uint32_t idx = pfr_tile_hash(mg, mn, x, y);
    uint32_t word = idx / 32;
    uint32_t bit = 1u << (idx % 32);
    if (rs->visit_hash[word] & bit)
        return false;
    rs->visit_hash[word] |= bit;
    rs->visit_count++;
    return true;
}

/* Global visit tracking (persistent across episodes) */
static bool pfr_global_visit_check_and_set(PfrEnv *env, uint8_t mg, uint8_t mn,
                                            int16_t x, int16_t y) {
    uint32_t idx = pfr_tile_hash(mg, mn, x, y);
    uint32_t word = idx / 32;
    uint32_t bit = 1u << (idx % 32);
    if (env->global_visit_hash[word] & bit)
        return false;
    env->global_visit_hash[word] |= bit;
    env->global_visit_count++;
    return true;
}

/* ================================================================
 * Reward computation
 * ================================================================ */

static float pfr_compute_reward(PfrEnv *env, const PfrRewardInfo *info) {
    PfrRewardState *rs = &env->reward_state;
    float reward = 0.0f;

    /* 1. Exploration: new tiles visited */
    pfr_visit_check_and_set(rs, info->map_group, info->map_num,
                             info->player_x, info->player_y);
    uint32_t new_visits = rs->visit_count - rs->prev_visit_count;
    reward += new_visits * 0.01f;  /* per-episode: tiny revisit signal */
    rs->prev_visit_count = rs->visit_count;

    /* 1a. Global frontier bonus (never-before-visited tile) */
    if (new_visits > 0) {
        bool globally_new = pfr_global_visit_check_and_set(
            env, info->map_group, info->map_num,
            info->player_x, info->player_y);
        if (globally_new)
            reward += 50.0f;  /* dominant signal: push the frontier */
    }

    /* 1b. Map transition bonus (first visit to each map per episode) */
    {
        uint32_t map_id = (uint32_t)info->map_group * 256 + info->map_num;
        uint32_t map_bit_idx = (map_id * 7 + info->map_group) % 64;
        uint64_t map_bit = 1ULL << map_bit_idx;
        if (!(rs->visited_maps & map_bit)) {
            if (rs->visited_maps != 0)   /* skip spawn map */
                reward += 10.0f;  /* big milestone for new maps */
            rs->visited_maps |= map_bit;
        }
    }

    /* 2. Badge progression */
    uint8_t new_badges = info->badges & ~rs->prev_badges;
    if (new_badges) {
        int count = __builtin_popcount(new_badges);
        reward += count * 50.0f;
        rs->prev_badges = info->badges;
    }

    /* 3. Party level gains */
    if (info->party_level_sum > rs->prev_party_level_sum) {
        reward += (info->party_level_sum - rs->prev_party_level_sum) * 2.0f;
        rs->prev_party_level_sum = info->party_level_sum;
    }

    /* 4. New party member */
    if (info->party_count > rs->prev_party_count) {
        reward += (info->party_count - rs->prev_party_count) * 5.0f;
        rs->prev_party_count = info->party_count;
    }

    return reward;
}

/* ================================================================
 * Extract visited tiles into obs buffer
 *
 * Copies the visit_hash bitfield (1024 bits = 128 bytes) into the
 * observation buffer. Each bit = 1 tile visited. The agent learns
 * which tiles are reachable through exploration, not from raw
 * metatile behavior bytes that have no semantic meaning.
 * ================================================================ */

static void pfr_extract_visited_tiles(PfrEnv *env) {
    memcpy(env->observations + PFR_OFF_VISITED,
           env->reward_state.visit_hash,
           PFR_VISIT_BYTES);
}

/* ================================================================
 * Extract pixel observations from GBA framebuffer
 *
 * Copies 240x160 RGB pixels (115200 bytes) from the game's ARGB
 * framebuffer into the obs buffer. No downscaling.
 * ================================================================ */

static void pfr_extract_pixels(PfrEnv *env) {
    PfrInstance *inst = env->instance;

    /* Get ARGB framebuffer (4 bytes/pixel) */
    const uint32_t *fb = NULL;
    if (inst->get_framebuffer) {
        fb = inst->get_framebuffer();
    }

    unsigned char *dst = env->observations + PFR_OFF_PIXELS;
    if (fb) {
        /* Convert ARGB8888 -> RGB888 */
        for (int i = 0; i < PFR_SCREEN_W * PFR_SCREEN_H; i++) {
            uint32_t argb = fb[i];
            dst[i * 3 + 0] = (argb >> 16) & 0xFF;  /* R */
            dst[i * 3 + 1] = (argb >> 8)  & 0xFF;  /* G */
            dst[i * 3 + 2] =  argb        & 0xFF;  /* B */
        }
    } else {
        memset(dst, 0, PFR_PIXEL_SIZE);
    }
}

/* ================================================================
 * c_reset: restore savestate, snapshot baseline, extract obs
 * ================================================================ */

static void pfr_restore_episode(PfrEnv *env, bool clear_outputs) {
    PfrInstance *inst = env->instance;

    /* 1. Zero per-episode state */
    memset(&env->reward_state, 0, sizeof(PfrRewardState));

    /* 2. Restore game state */
    if (env->savestate_path[0] != '\0') {
        if (inst->load_state(env->savestate_path) != 0) {
            fprintf(stderr, "pfr: failed to load savestate: %s\n",
                    env->savestate_path);
            inst->restore_hot();
        }
    } else {
        inst->restore_hot();
    }

    /* 3. Snapshot baseline for delta reward */
    PfrRewardInfo info;
    inst->get_reward_info(&info);

    /* ASSERT: validate game state after restore */
    assert(info.party_count <= 6 &&
           "corrupt party_count after restore");

    PfrRewardState *rs = &env->reward_state;
    rs->prev_badges = info.badges;
    rs->prev_party_count = info.party_count;
    rs->prev_party_level_sum = info.party_level_sum;
    rs->prev_money = info.money;

    /* Mark starting tile visited */
    pfr_visit_check_and_set(rs, info.map_group, info.map_num,
                             info.player_x, info.player_y);
    rs->prev_visit_count = rs->visit_count;

    /* 4. Extract observations (scalars + NPCs from game, visited + pixels from us) */
    inst->extract_obs(env->observations);  /* fills [0..144] */
    pfr_extract_visited_tiles(env);         /* fills [145..272] */
    if (env->use_pixels)
        pfr_extract_pixels(env);            /* fills [273..115472] */

    if (clear_outputs) {
        env->rewards[0] = 0.0f;
        env->terminals[0] = 0;
    }
}

static void c_reset(PfrEnv *env) {
    pfr_restore_episode(env, true);
}

/* ================================================================
 * c_step: inject action, run frames, extract obs, compute reward
 *
 * Reward is RESET to 0 at the top of each step. The training code
 * (PPO) computes returns from per-step rewards, not accumulated.
 *
 * On terminal: auto-reset via pfr_restore_episode, preserving the
 * terminal=1 signal for this step (PufferLib ocean pattern).
 * ================================================================ */

static void c_step(PfrEnv *env) {
    PfrInstance *inst = env->instance;

    /* RESET reward — must be first */
    env->rewards[0] = 0.0f;

    /* ASSERT: action in valid range */
    int action = (int)env->actions[0];
    assert((action >= 0 && action < PFR_NUM_ACTIONS) ||
           (action = 0, 1));  /* clamp to noop on release builds */
    if (action < 0 || action >= PFR_NUM_ACTIONS)
        action = 0;

    /* 1. Map action to GBA buttons */
    uint16_t buttons = sPfrActionToButtons[action];

    /* AUTO-BATTLE: if in battle, force A-button every step.
     * Lite mode has no pixel obs, so agent can't see battle menus.
     * A-button selects FIGHT -> first move -> confirms text boxes. */
    if (env->observations[11]) {  /* in_battle from previous obs */
        buttons = PFR_BTN_A;
    }

    /* 2. Step game frames
     * Uses step_frames (not step_frames_fast) because the menu system
     * depends on CopyBufferedValuesToGpuRegs and ProcessDma3Requests
     * which step_frames_fast skips. DO NOT enable fast path. */
    int n = env->frames_per_step ? env->frames_per_step : PFR_FRAMES_PER_STEP;
    inst->step_frames(buttons, n);

    /* 3. Extract observations */
    inst->extract_obs(env->observations);   /* scalars + NPCs [0..144] */
    pfr_extract_visited_tiles(env);          /* visited tiles [145..272] */
    if (env->use_pixels)
        pfr_extract_pixels(env);             /* pixels [273..115472] */

    /* 4. Compute reward */
    PfrRewardInfo info;
    inst->get_reward_info(&info);

    /* ASSERT: reward info integrity */
    assert(info.party_count <= 6);
    assert(info.money <= 999999);

    float reward = pfr_compute_reward(env, &info);

    /* ASSERT: reward is finite */
    assert(isfinite(reward));

    env->rewards[0] = reward;
    env->reward_state.episode_return += reward;
    env->reward_state.step_count++;

    /* 5. Terminal check */
    bool terminal = false;
    uint32_t max_steps = env->max_steps ? env->max_steps : PFR_MAX_STEPS;
    if (env->reward_state.step_count >= max_steps)
        terminal = true;

    env->terminals[0] = terminal ? 1 : 0;

    if (terminal) {
        float final_reward = env->rewards[0];

        /* Log episode stats */
        env->log.episode_return += env->reward_state.episode_return;
        env->log.episode_length += (float)env->reward_state.step_count;
        env->log.badges += (float)__builtin_popcount(env->reward_state.prev_badges);
        env->log.exploration += (float)env->reward_state.visit_count;
        env->log.party_level_sum += (float)env->reward_state.prev_party_level_sum;
        env->log.n += 1.0f;

        /* Auto-reset, preserving terminal signal for this step */
        pfr_restore_episode(env, false);
        env->rewards[0] = final_reward;
        env->terminals[0] = 1;
    }
}

/* ================================================================
 * c_render / c_close
 * ================================================================ */

static void c_render(PfrEnv *env) {
    (void)env;
    /* Rendering handled by eval.py / play.py via capture_frame */
}

static void c_close(PfrEnv *env) {
    (void)env;
}

#endif /* PFR_H */
