/*
 * t5/ops.c - T5 CPU operators (see t5/ops.h)
 */

#include "t5/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void cp_t5_layernorm(float *y, const float *x, const float *w,
                     size_t n, float eps) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) y[i] = x[i] * inv * w[i];
}

void cp_t5_matvec(float *y, const float *W, const float *x,
                  size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        const float *row = W + r * cols;
        for (size_t c = 0; c < cols; c++) acc += row[c] * x[c];
        y[r] = acc;
    }
}

void cp_t5_linear_seq(float *Y, const float *W, const float *X,
                      size_t seq, size_t rows, size_t cols) {
    for (size_t i = 0; i < seq; i++)
        cp_t5_matvec(Y + i * rows, W, X + i * cols, rows, cols);
}

void cp_t5_embed(float *y, const float *E, size_t d_model, uint32_t tok) {
    memcpy(y, E + (size_t)tok * d_model, d_model * sizeof(float));
}

void cp_t5_embed_seq(float *Y, const float *E, size_t d_model,
                     const uint32_t *toks, size_t n) {
    for (size_t i = 0; i < n; i++)
        cp_t5_embed(Y + i * d_model, E, d_model, toks[i]);
}

void cp_t5_dense_relu_dense(float *y, const float *x,
                            const float *wi, const float *wo,
                            size_t d_model, size_t d_ff) {
    float *h = malloc(d_ff * sizeof *h);
    if (!h) {
        memset(y, 0, d_model * sizeof *y);
        return;
    }
    cp_t5_matvec(h, wi, x, d_ff, d_model);
    for (size_t i = 0; i < d_ff; i++)
        if (h[i] < 0.0f) h[i] = 0.0f;
    cp_t5_matvec(y, wo, h, d_model, d_ff);
    free(h);
}

int cp_t5_relative_position_bucket(int relative_position, int bidirectional,
                                   int num_buckets, int max_distance) {
    int relative_buckets = 0;
    int n = num_buckets;
    int rel = relative_position;
    if (bidirectional) {
        n /= 2;
        if (rel > 0) relative_buckets += n;
        rel = rel < 0 ? -rel : rel;
    } else {
        /* only non-positive relative positions (past keys) */
        rel = rel > 0 ? 0 : -rel;
    }

    int max_exact = n / 2;
    if (rel < max_exact) {
        relative_buckets += rel;
    } else {
        float ratio = (float)rel / (float)max_exact;
        float log_ratio = logf(ratio) / logf((float)max_distance / (float)max_exact);
        int large = max_exact + (int)(log_ratio * (float)(n - max_exact));
        if (large >= n) large = n - 1;
        if (large < 0) large = 0;
        relative_buckets += large;
    }
    return relative_buckets;
}

void cp_t5_position_bias(float *bias, const float *rel_emb,
                         size_t qlen, size_t klen, size_t n_heads,
                         size_t num_buckets, size_t max_distance,
                         int bidirectional) {
    for (size_t qi = 0; qi < qlen; qi++) {
        for (size_t ki = 0; ki < klen; ki++) {
            int rel = (int)ki - (int)qi;
            int bucket = cp_t5_relative_position_bucket(
                    rel, bidirectional, (int)num_buckets, (int)max_distance);
            const float *row = rel_emb + (size_t)bucket * n_heads;
            for (size_t h = 0; h < n_heads; h++) {
                /* bias layout: [n_heads, qlen, klen] */
                bias[(h * qlen + qi) * klen + ki] = row[h];
            }
        }
    }
}

static void t5_softmax(float *x, size_t n) {
    float m = x[0];
    for (size_t i = 1; i < n; i++)
        if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

void cp_t5_mha(float *out, const float *q, const float *k, const float *v,
               const float *bias, size_t seq_q, size_t seq_k,
               size_t n_heads, size_t d_kv, int causal) {
    if (!out || !q || !k || !v || seq_q == 0 || seq_k == 0 ||
        n_heads == 0 || d_kv == 0) {
        return;
    }

    float *score = malloc(seq_k * sizeof *score);
    if (!score) {
        memset(out, 0, seq_q * n_heads * d_kv * sizeof *out);
        return;
    }

    for (size_t h = 0; h < n_heads; h++) {
        for (size_t qi = 0; qi < seq_q; qi++) {
            const float *qh = q + (qi * n_heads + h) * d_kv;
            int all_masked = 1;
            for (size_t ki = 0; ki < seq_k; ki++) {
                if (causal && ki > qi) {
                    score[ki] = -1e9f;
                    continue;
                }
                const float *kh = k + (ki * n_heads + h) * d_kv;
                float acc = 0.0f;
                for (size_t d = 0; d < d_kv; d++) acc += qh[d] * kh[d];
                if (bias)
                    acc += bias[(h * seq_q + qi) * seq_k + ki];
                score[ki] = acc;
                all_masked = 0;
            }
            if (all_masked) {
                memset(out + (qi * n_heads + h) * d_kv, 0, d_kv * sizeof(float));
                continue;
            }
            t5_softmax(score, seq_k);
            float *oh = out + (qi * n_heads + h) * d_kv;
            for (size_t d = 0; d < d_kv; d++) {
                float acc = 0.0f;
                for (size_t ki = 0; ki < seq_k; ki++) {
                    acc += score[ki] *
                           v[(ki * n_heads + h) * d_kv + d];
                }
                oh[d] = acc;
            }
        }
    }
    free(score);
}

/* Pack linear output [seq, n_heads*d_kv] into [seq, n_heads, d_kv] — already
 * that layout if we store as seq-major then head-major contiguous. */
static void project_to_heads(float *heads, const float *X, const float *W,
                             size_t seq, size_t d_model, size_t n_heads,
                             size_t d_kv) {
    size_t inner = n_heads * d_kv;
    float *tmp = malloc(seq * inner * sizeof *tmp);
    if (!tmp) {
        memset(heads, 0, seq * inner * sizeof *heads);
        return;
    }
    cp_t5_linear_seq(tmp, W, X, seq, inner, d_model);
    /* tmp[i, h*d_kv + d] already matches heads[i, h, d] */
    memcpy(heads, tmp, seq * inner * sizeof *heads);
    free(tmp);
}

void cp_t5_self_attention(float *y, const float *x,
                          const float *Wq, const float *Wk, const float *Wv,
                          const float *Wo, const float *rel_emb,
                          size_t seq, size_t d_model, size_t n_heads,
                          size_t d_kv, size_t num_buckets, size_t max_distance,
                          int causal, int has_rel_bias) {
    size_t inner = n_heads * d_kv;
    float *q = malloc(seq * inner * sizeof *q);
    float *k = malloc(seq * inner * sizeof *k);
    float *v = malloc(seq * inner * sizeof *v);
    float *ctx = malloc(seq * inner * sizeof *ctx);
    float *bias = NULL;
    if (!q || !k || !v || !ctx) {
        free(q); free(k); free(v); free(ctx);
        memset(y, 0, seq * d_model * sizeof *y);
        return;
    }

    project_to_heads(q, x, Wq, seq, d_model, n_heads, d_kv);
    project_to_heads(k, x, Wk, seq, d_model, n_heads, d_kv);
    project_to_heads(v, x, Wv, seq, d_model, n_heads, d_kv);

    if (has_rel_bias && rel_emb) {
        bias = malloc(n_heads * seq * seq * sizeof *bias);
        if (bias) {
            cp_t5_position_bias(bias, rel_emb, seq, seq, n_heads,
                                num_buckets, max_distance, causal ? 0 : 1);
        }
    }

    cp_t5_mha(ctx, q, k, v, bias, seq, seq, n_heads, d_kv, causal);
    cp_t5_linear_seq(y, Wo, ctx, seq, d_model, inner);

    free(bias);
    free(q); free(k); free(v); free(ctx);
}

void cp_t5_cross_attention(float *y, const float *x, const float *mem,
                           const float *Wq, const float *Wk, const float *Wv,
                           const float *Wo,
                           size_t seq_q, size_t seq_k, size_t d_model,
                           size_t n_heads, size_t d_kv) {
    size_t inner = n_heads * d_kv;
    float *q = malloc(seq_q * inner * sizeof *q);
    float *k = malloc(seq_k * inner * sizeof *k);
    float *v = malloc(seq_k * inner * sizeof *v);
    float *ctx = malloc(seq_q * inner * sizeof *ctx);
    if (!q || !k || !v || !ctx) {
        free(q); free(k); free(v); free(ctx);
        memset(y, 0, seq_q * d_model * sizeof *y);
        return;
    }

    project_to_heads(q, x, Wq, seq_q, d_model, n_heads, d_kv);
    project_to_heads(k, mem, Wk, seq_k, d_model, n_heads, d_kv);
    project_to_heads(v, mem, Wv, seq_k, d_model, n_heads, d_kv);

    cp_t5_mha(ctx, q, k, v, NULL, seq_q, seq_k, n_heads, d_kv, 0);
    cp_t5_linear_seq(y, Wo, ctx, seq_q, d_model, inner);

    free(q); free(k); free(v); free(ctx);
}
