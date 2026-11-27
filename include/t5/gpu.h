/*
 * t5/gpu.h - T5 CUDA kernels + Enc–Dec forward
 *
 * Shared driver/NVRTC/context live in gpu_runtime.h. This module owns
 * T5-specific FP32 kernels and the encoder / decoder-step path.
 * Public symbols are prefixed t5_gpu_* so they never collide with
 * qwen/gpu.h (gpu_*).
 */

#ifndef T5_GPU_H
#define T5_GPU_H

#include <stddef.h>
#include <stdint.h>

#include "t5/config.h"
#include "t5/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load runtime + compile T5 kernels. Returns 0 on success. */
int t5_gpu_init(void);

void t5_gpu_shutdown(void);

int t5_gpu_ready(void);

/*
 * y = W x on device (FP32 row-major W). Upload / launch / download.
 * Bring-up helper; not used by the resident forward path.
 */
int t5_gpu_matvec(float *y, const float *W, const float *x,
                  size_t rows, size_t cols);

/* ------------------------------------------------------------------ */
/* resident Enc–Dec forward                                           */

/*
 * Upload weights, allocate encoder memory + decoder caches.
 * max_src / max_tgt: capacity in tokens (0 => 512).
 * Currently requires non-gated DenseReluDense (t5-small). Weights are
 * widened to FP32 on upload when the file is BF16.
 */
int t5_gpu_forward_init(const cp_t5_config_t *cfg, const cp_t5_weights_t *w,
                        size_t max_src, size_t max_tgt);

void t5_gpu_forward_free(void);

/*
 * Full encoder over src_ids[0..n). n must be in (0, max_src].
 * Leaves memory[n, d_model] and per-layer cross K/V on device.
 */
int t5_gpu_encode(const uint32_t *src_ids, size_t n);

/* Copy encoder memory to host (n * d_model floats). Call after encode. */
int t5_gpu_encode_download(float *out);

/* Clear decoder self-attn write position; encoder memory stays. */
void t5_gpu_decode_reset(void);

/*
 * One decoder token at absolute position pos (0-based).
 * Returns host logits [vocab_size] (module-owned) or NULL on failure.
 */
const float *t5_gpu_decode_step(uint32_t tok, size_t pos);

#ifdef __cplusplus
}
#endif

#endif /* T5_GPU_H */
