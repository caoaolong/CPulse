/*
 * gpu.h - CUDA driver-API + NVRTC backend for CPulse
 *
 * Runs on the stock Cygwin/GCC toolchain: no MSVC, no nvcc. The driver
 * library (nvcuda.dll, ships with the GPU driver) and the runtime
 * compiler (nvrtc64_*.dll, ships with the CUDA toolkit) are loaded with
 * dlopen at startup; kernels are compiled from embedded source strings
 * for the detected compute capability (RTX 3050 Laptop = sm_86).
 *
 * Current scope is the bring-up path: context init, NVRTC-compiled
 * BF16 matvec, per-call upload/compute/download. Persistent
 * device-resident weights and multi-op fusion come with the forward
 * pass; the kernel summation order already matches cp_matvec so parity
 * is exact up to FP32 rounding.
 *
 * Logging convention: informational bring-up lines go to stdout (stderr
 * is not wired to the console under some debuggers); errors go to stderr.
 */

#ifndef GPU_H
#define GPU_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"
#include "weights.h"

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
