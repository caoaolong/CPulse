/*
 * t5/ops.h - T5 CPU math primitives (FP32 weights)
 *
 * Matches HuggingFace T5Attention / T5LayerNorm / DenseReluDense:
 *   - LayerNorm is RMS-style (no mean subtract, weight only)
 *   - Attention scores are NOT scaled by 1/sqrt(d_kv)
 *   - Relative position bias is bucketed and added to scores
 */

#ifndef T5_OPS_H
#define T5_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CP_T5_REL_MAX_DISTANCE_DEFAULT
#define CP_T5_REL_MAX_DISTANCE_DEFAULT 128
#endif

/* y[i] = x[i] * rsqrt(mean(x^2)+eps) * w[i]  (T5LayerNorm) */
void cp_t5_layernorm(float *y, const float *x, const float *w,
                     size_t n, float eps);

/* y = W x ; W is row-major [rows, cols] (PyTorch Linear weight). */
void cp_t5_matvec(float *y, const float *W, const float *x,
                  size_t rows, size_t cols);

/*
 * Sequence gemm: for each of `seq` rows,
 *   Y[i] = W * X[i],  X:[seq,cols], Y:[seq,rows], W:[rows,cols].
 */
void cp_t5_linear_seq(float *Y, const float *W, const float *X,
                      size_t seq, size_t rows, size_t cols);

/* Embedding lookup: y[0..d) = E[tok * d ..] (FP32 rows). */
void cp_t5_embed(float *y, const float *E, size_t d_model, uint32_t tok);

void cp_t5_embed_seq(float *Y, const float *E, size_t d_model,
                     const uint32_t *toks, size_t n);

/* y = wo(relu(wi(x))) ; wi:[d_ff,d_model], wo:[d_model,d_ff]. */
void cp_t5_dense_relu_dense(float *y, const float *x,
                            const float *wi, const float *wo,
                            size_t d_model, size_t d_ff);

/*
 * HF T5Attention._relative_position_bucket.
 * relative_position = memory_position - query_position.
 * Returns bucket in [0, num_buckets).
 */
int cp_t5_relative_position_bucket(int relative_position, int bidirectional,
                                   int num_buckets, int max_distance);

/*
 * Fill position bias [n_heads, qlen, klen] from embedding
 * rel_emb [num_buckets, n_heads] (safetensors / nn.Embedding layout).
 * bidirectional=1 for encoder self-attn; 0 for decoder self-attn.
 */
void cp_t5_position_bias(float *bias, const float *rel_emb,
                         size_t qlen, size_t klen, size_t n_heads,
                         size_t num_buckets, size_t max_distance,
                         int bidirectional);

/*
 * Multi-head attention core (no QKV/O projections).
 *   q: [seq_q, n_heads, d_kv]
 *   k,v: [seq_k, n_heads, d_kv]
 *   bias: [n_heads, seq_q, seq_k] or NULL
 *   out: [seq_q, n_heads, d_kv]
 * Scores are unscaled (T5). If causal!=0, positions ki>qi are masked.
 */
void cp_t5_mha(float *out, const float *q, const float *k, const float *v,
               const float *bias, size_t seq_q, size_t seq_k,
               size_t n_heads, size_t d_kv, int causal);

/*
 * Full self-attention block body (proj + mha + out proj), no residual/norm.
 *   x,y: [seq, d_model]
 *   Wq/k/v: [inner, d_model], Wo: [d_model, inner], inner = n_heads*d_kv
 *   rel_emb: [num_buckets, n_heads] or NULL if no relative bias
 */
void cp_t5_self_attention(float *y, const float *x,
                          const float *Wq, const float *Wk, const float *Wv,
                          const float *Wo, const float *rel_emb,
                          size_t seq, size_t d_model, size_t n_heads,
                          size_t d_kv, size_t num_buckets, size_t max_distance,
                          int causal, int has_rel_bias);

/*
 * Cross-attention body (no residual/norm, no relative bias).
 *   x: decoder states [seq_q, d_model]
 *   mem: encoder memory [seq_k, d_model]
 *   y: [seq_q, d_model]
 */
void cp_t5_cross_attention(float *y, const float *x, const float *mem,
                           const float *Wq, const float *Wk, const float *Wv,
                           const float *Wo,
                           size_t seq_q, size_t seq_k, size_t d_model,
                           size_t n_heads, size_t d_kv);

#ifdef __cplusplus
}
#endif

#endif /* T5_OPS_H */
