/*
 * binding.c -- PufferLib 4.0 ocean binding for PufferFireRed
 *
 * Bridges pfr.h (pure C env) to Python via PufferLib env_binding.h.
 * Manages SO-copy game instances for parallel vectorized training.
 *
 * Build (from pokefirered_puffer/):
 *   cc -shared -fPIC -O2 binding.c ../pokefirered-native/src/pfr_so_instance.c \
 *      -I. -I../pokefirered-native/src -I<pufferlib>/ocean \
 *      -I<python-include> -I<numpy-include> -ldl -fopenmp \
 *      -o binding.cpython-3XX-<arch>.so
 */

#include <Python.h>
#include <dlfcn.h>

#include "pfr.h"

/* ================================================================
 * PufferLib binding macros
 *
 * These are consumed by env_binding.h to size observation/action
 * buffers and configure the vectorized environment wrapper.
 * ================================================================ */

#define OBS_SIZE   PFR_OBS_SIZE       /* 115473 bytes */
#define NUM_ATNS   1
#define ACT_SIZES  {PFR_NUM_ACTIONS}  /* {8} */
#define OBS_TYPE   UNSIGNED_CHAR
#define ACT_TYPE   FLOAT

#define Env  PfrEnv
#define Log  PfrLog

/* Forward-declare custom Python methods (before env_binding.h) */
static PyObject *py_init_instances(PyObject *self, PyObject *args);
static PyObject *py_destroy_instances(PyObject *self, PyObject *args);
static PyObject *py_capture_frame(PyObject *self, PyObject *args);
static PyObject *py_save_state(PyObject *self, PyObject *args);

#define MY_METHODS \
    {"init_instances", py_init_instances, METH_VARARGS, \
     "Create SO-copy game instances: init_instances(so_path, tmp_dir, num_envs)"}, \
    {"destroy_instances", py_destroy_instances, METH_VARARGS, \
     "Destroy all SO-copy game instances"}, \
    {"capture_frame", py_capture_frame, METH_VARARGS, \
     "Render current frame as numpy (160,240,4) uint8"}, \
    {"save_state", py_save_state, METH_VARARGS, \
     "Save game state: save_state(idx, path)"}

/* PufferLib vectorized env template — provides vec_init, vec_step,
 * vec_log, PyInit_binding, and calls our my_init/my_log hooks. */
#include "env_binding.h"

/* ================================================================
 * Global instance pool
 *
 * SO-copy instances are created once via init_instances() before
 * vec_init runs. Each call to my_init (per-env) grabs the next
 * instance from this pool via sNextEnvId counter.
 * ================================================================ */

static PfrInstance *sInstances = NULL;
static int sNumInstances = 0;
static int sNextEnvId = 0;

/* ================================================================
 * my_init -- called per env by env_binding.h vec_init
 *
 * Unpacks kwargs for env config and assigns a game instance from
 * the pre-created pool. Does NOT call c_reset — that happens when
 * the Python side calls vec_reset after vec_init.
 * ================================================================ */

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    (void)args;

    env->frames_per_step = (uint32_t)unpack(kwargs, "frames_per_step");
    if (PyErr_Occurred()) return -1;

    env->max_steps = (uint32_t)unpack(kwargs, "max_steps");
    if (PyErr_Occurred()) return -1;

    /* Savestate path (string kwarg, optional) */
    PyObject *path_obj = PyDict_GetItemString(kwargs, "savestate_path");
    if (path_obj && PyUnicode_Check(path_obj)) {
        const char *path = PyUnicode_AsUTF8(path_obj);
        if (path) {
            strncpy(env->savestate_path, path, sizeof(env->savestate_path) - 1);
            env->savestate_path[sizeof(env->savestate_path) - 1] = 0;
        }
    }

    /* Assign game instance from pre-created pool */
    int env_id = sNextEnvId++;
    if (env_id < 0 || env_id >= sNumInstances) {
        PyErr_Format(PyExc_ValueError,
                     "env_id %d out of range [0, %d). "
                     "Call init_instances() before vec_init().",
                     env_id, sNumInstances);
        return -1;
    }
    env->instance = &sInstances[env_id];

    return 0;
}

/* ================================================================
 * my_log -- called by env_binding.h vec_log
 *
 * Populates the Python dict with averaged episode stats.
 * The Log struct is all floats; env_binding.h averages across
 * envs by dividing each field by log->n before calling us.
 * ================================================================ */

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "badges", log->badges);
    assign_to_dict(dict, "exploration", log->exploration);
    assign_to_dict(dict, "party_level_sum", log->party_level_sum);
    return 0;
}

/* ================================================================
 * Custom Python methods for SO instance lifecycle
 * ================================================================ */

static PyObject *py_init_instances(PyObject *self, PyObject *args) {
    (void)self;
    const char *so_path;
    const char *tmp_dir;
    int num_envs;

    if (!PyArg_ParseTuple(args, "ssi", &so_path, &tmp_dir, &num_envs))
        return NULL;

    /* Clean up previous pool if any */
    if (sInstances) {
        pfr_instances_destroy(sInstances, sNumInstances);
        free(sInstances);
        sInstances = NULL;
        sNumInstances = 0;
    }

    sInstances = (PfrInstance *)calloc(num_envs, sizeof(PfrInstance));
    if (!sInstances) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate instance pool");
        return NULL;
    }

    if (pfr_instances_create(so_path, tmp_dir, sInstances, num_envs) != 0) {
        free(sInstances);
        sInstances = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Failed to create SO instances");
        return NULL;
    }
    sNumInstances = num_envs;
    sNextEnvId = 0;

    for (int i = 0; i < num_envs; i++) {
        fprintf(stderr, "pfr_binding: booting instance %d/%d\n", i + 1, num_envs);
        sInstances[i].boot();
    }
    fprintf(stderr, "pfr_binding: %d instances ready\n", num_envs);

    Py_RETURN_NONE;
}

static PyObject *py_destroy_instances(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    if (sInstances) {
        pfr_instances_destroy(sInstances, sNumInstances);
        free(sInstances);
        sInstances = NULL;
        sNumInstances = 0;
    }
    Py_RETURN_NONE;
}

static PyObject *py_capture_frame(PyObject *self, PyObject *args) {
    (void)self;
    int idx;
    if (!PyArg_ParseTuple(args, "i", &idx))
        return NULL;
    if (idx < 0 || idx >= sNumInstances) {
        PyErr_Format(PyExc_IndexError,
                     "Instance %d out of range [0, %d)", idx, sNumInstances);
        return NULL;
    }

    /* Render current GBA screen (no game advance) */
    if (sInstances[idx].render_current_frame)
        sInstances[idx].render_current_frame();

    /* Allocate numpy array: (160, 240, 4) uint8 ARGB */
    npy_intp dims[3] = {PFR_SCREEN_H, PFR_SCREEN_W, 4};
    PyObject *arr = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!arr) return NULL;

    uint32_t *buf = (uint32_t *)PyArray_DATA((PyArrayObject *)arr);

    /* Try direct framebuffer access first (bypasses renderer guard) */
    typedef const uint32_t *(*get_fb_fn)(void);
    get_fb_fn get_fb = (get_fb_fn)dlsym(sInstances[idx].dl_handle,
                                         "HostRendererGetFramebuffer");
    if (get_fb) {
        const uint32_t *src = get_fb();
        if (src)
            memcpy(buf, src, PFR_SCREEN_W * PFR_SCREEN_H * sizeof(uint32_t));
        else
            memset(buf, 0, PFR_SCREEN_W * PFR_SCREEN_H * sizeof(uint32_t));
    } else {
        sInstances[idx].copy_framebuffer(buf, PFR_SCREEN_W);
    }

    return arr;
}

static PyObject *py_save_state(PyObject *self, PyObject *args) {
    (void)self;
    int idx;
    const char *path;
    if (!PyArg_ParseTuple(args, "is", &idx, &path))
        return NULL;
    if (idx < 0 || idx >= sNumInstances) {
        PyErr_Format(PyExc_IndexError, "Instance %d out of range", idx);
        return NULL;
    }

    typedef int (*save_fn)(const char *);
    save_fn save = (save_fn)dlsym(sInstances[idx].dl_handle,
                                   "pfr_game_save_state");
    if (!save) {
        PyErr_SetString(PyExc_RuntimeError,
                        "pfr_game_save_state symbol not found in SO");
        return NULL;
    }
    return PyLong_FromLong(save(path));
}
