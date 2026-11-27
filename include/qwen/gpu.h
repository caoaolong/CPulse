/*
 * gpu.h - Qwen CUDA kernels + decoder forward
 *
 * Shared driver/NVRTC/context live in gpu_runtime.h. This module compiles
 * Qwen-specific kernels and runs the decoder-only forward on device.
 */

#ifndef GPU_H
#define GPU_H

#include <stddef.h>
#include <stdint.h>

#include "qwen/model.h"
#include "qwen/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load libraries, init driver, create context, compile kernels.
 * Prints diagnostics to stderr. Returns 0 on success. */
int gpu_init(void);

/* Destroy context and unload libraries. Safe to call when not inited. */
void gpu_shutdown(void);

/* 1 after a successful gpu_init, else 0. */
int gpu_ready(void);

/*
 * y = W x on the device (W row-major BF16, x/y FP32 host buffers).
 * Uploads inputs, launches, synchronizes and downloads y.
 * Returns 0 on success.
 */
int gpu_matvec(float *y, const uint16_t *W, const float *x,
               size_t rows, size_t cols);

/* ------------------------------------------------------------------ */
/* full decoder forward (weights + kv caches resident on device)      */

/*
 * Upload every weight once and allocate device-side kv caches and
 * scratch. Calls gpu_init() first if needed. Returns 0 on success;
 * on failure everything is freed and -1 comes back (caller falls back
 * to the CPU path).
 */
int gpu_forward_init(const cp_config_t *cfg, const cp_weights_t *w);

/* Release all device memory owned by the forward backend. */
void gpu_forward_free(void);

/*
 * One token at absolute position pos through all layers on the GPU.
 * Returns host logits [vocab_size] (module-owned, valid until the
 * next call) or NULL on a device failure.
 */
const float *gpu_forward_step(uint32_t tok, size_t pos);

/* Start a fresh sequence: kv write position restarts with pos=0. */
void gpu_forward_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_H */
