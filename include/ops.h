/*
 * ops.h - reference math operators for CPulse (CPU)
 *
 * Stage 4 - forward pass building blocks. Plain FP32 kernels; weights
 * arrive as BF16 and are widened on the fly. These are the correctness
 * baseline: the GPU backend (gpu.h) must match them bit-for-bit on
 * ordering-identical loops (tolerance 1e-4 in tests/gputest.c).
 */

#ifndef OPS_H
#define OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* BF16 <-> FP32. Truncating (no rounding); exact for the values used. */
float    cp_bf16_to_fp32(uint16_t b);
uint16_t cp_fp32_to_bf16(float f);

/* y = W x (row-major, no bias). Same summation order as the GPU kernel. */
void cp_matvec(float *y, const uint16_t *W, const float *x,
               size_t rows, size_t cols);

/* y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]; w is BF16 gamma. */
void cp_rmsnorm(float *y, const float *x, const uint16_t *w,
                size_t n, float eps);

/* In-place softmax with max subtraction. */
void cp_softmax(float *x, size_t n);

/*
 * Rotary position embedding, GPT-NeoX style (half rotation), matching
 * Qwen2: for head h, pair dims (i, i+hd/2), angle = pos * theta^(-2i/hd).
 * t holds n_heads * head_dim floats, rotated in place.
 */
void cp_rope(float *t, size_t n_heads, size_t head_dim,
             size_t pos, float theta);

/* SwiGLU MLP activation: y[i] = (gate[i] * sigmoid(gate[i])) * up[i]. */
void cp_swiglu(float *y, const float *gate, const float *up, size_t n);

/* Embedding lookup: y[0..hidden) = BF16 row `tok` of E [vocab, hidden].
 * Caller must ensure tok < vocab. */
void cp_embed(float *y, const uint16_t *E, size_t hidden, uint32_t tok);

#ifdef __cplusplus
}
#endif

#endif /* OPS_H */
