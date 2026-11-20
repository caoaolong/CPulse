/*
 * ops.c - reference math operators, CPU (see ops.h)
 */

#include "ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float cp_bf16_to_fp32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

uint16_t cp_fp32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return (uint16_t)(u >> 16); /* truncation; no rounding */
}

void cp_matvec(float *y, const uint16_t *W, const float *x,
               size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        const uint16_t *row = W + r * cols;
        for (size_t c = 0; c < cols; c++) {
            acc += cp_bf16_to_fp32(row[c]) * x[c];
        }
        y[r] = acc;
    }
}

void cp_rmsnorm(float *y, const float *x, const uint16_t *w,
                size_t n, float eps) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] * inv * cp_bf16_to_fp32(w[i]);
    }
}

void cp_softmax(float *x, size_t n) {
    float m = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > m) m = x[i];
    }
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

void cp_rope(float *t, size_t n_heads, size_t head_dim,
             size_t pos, float theta) {
    size_t half = head_dim / 2;
    for (size_t h = 0; h < n_heads; h++) {
        float *head = t + h * head_dim;
        for (size_t i = 0; i < half; i++) {
            float freq = powf(theta, -(float)(2 * i) / (float)head_dim);
            float ang = (float)pos * freq;
            float co = cosf(ang), si = sinf(ang);
            float a = head[i], b = head[i + half];
            head[i]        = a * co - b * si;
            head[i + half] = a * si + b * co;
        }
    }
}

void cp_swiglu(float *y, const float *gate, const float *up, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        y[i] = silu * up[i];
    }
}

void cp_embed(float *y, const uint16_t *E, size_t hidden, uint32_t tok) {
    const uint16_t *row = E + (size_t)tok * hidden;
    for (size_t i = 0; i < hidden; i++) {
        y[i] = cp_bf16_to_fp32(row[i]);
    }
}

void cp_embed_seq(float *Y, const uint16_t *E, size_t hidden,
                  const uint32_t *toks, size_t n) {
    for (size_t i = 0; i < n; i++) {
        cp_embed(Y + i * hidden, E, hidden, toks[i]);
    }
}

/* Double both buffers; contents preserved; cap updated on success only. */
static int kv_grow(cp_kv_t *kv, size_t ncap) {
    float *nk = malloc(ncap * kv->kv_dim * sizeof *nk);
    float *nv = malloc(ncap * kv->kv_dim * sizeof *nv);
    if (!nk || !nv) {
        free(nk);
        free(nv);
        return -1;
    }
    if (kv->len > 0) {
        memcpy(nk, kv->k, kv->len * kv->kv_dim * sizeof *nk);
        memcpy(nv, kv->v, kv->len * kv->kv_dim * sizeof *nv);
    }
    free(kv->k);
    free(kv->v);
    kv->k = nk;
    kv->v = nv;
    kv->cap = ncap;
    return 0;
}

int cp_kv_reserve(cp_kv_t *kv, size_t kv_dim, size_t n) {
    if (!kv || kv_dim == 0) return -1;
    if (kv->kv_dim == 0) kv->kv_dim = kv_dim;
    else if (kv->kv_dim != kv_dim) return -1;
    if (kv->cap >= n) return 0;
    size_t ncap = kv->cap ? kv->cap : 16;
    while (ncap < n) ncap *= 2;
    return kv_grow(kv, ncap);
}

void cp_kv_push(cp_kv_t *kv, const float *k_one, const float *v_one) {
    if (!kv || !k_one || !v_one || kv->kv_dim == 0) return;
    if (kv->len >= kv->cap && kv_grow(kv, kv->cap ? kv->cap * 2 : 16) != 0) return;
    memcpy(kv->k + kv->len * kv->kv_dim, k_one, kv->kv_dim * sizeof *k_one);
    memcpy(kv->v + kv->len * kv->kv_dim, v_one, kv->kv_dim * sizeof *v_one);
    kv->len++;
}

void cp_kv_free(cp_kv_t *kv) {
    if (!kv) return;
    free(kv->k);
    free(kv->v);
    kv->k = NULL;
    kv->v = NULL;
    kv->len = kv->cap = kv->kv_dim = 0;
}

void cp_attention(float *out, const float *q, const float *k, const float *v,
                  size_t n_heads, size_t kv_heads, size_t head_dim, size_t pos) {
    if (!out || !q || !k || !v || n_heads == 0 || kv_heads == 0 || head_dim == 0) return;
    size_t repeat = n_heads / kv_heads;
    if (repeat == 0) return;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *score = malloc((pos + 1) * sizeof *score);
    if (!score) {
        memset(out, 0, n_heads * head_dim * sizeof *out);
        return;
    }
    for (size_t h = 0; h < n_heads; h++) {
        size_t kh = h / repeat;
        const float *qh = q + h * head_dim;
        for (size_t j = 0; j <= pos; j++) {
            const float *kj = k + (j * kv_heads + kh) * head_dim;
            float acc = 0.0f;
            for (size_t d = 0; d < head_dim; d++) acc += qh[d] * kj[d];
            score[j] = acc * scale;
        }
        cp_softmax(score, pos + 1);
        float *oh = out + h * head_dim;
        for (size_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (size_t j = 0; j <= pos; j++) {
                acc += score[j] * v[(j * kv_heads + kh) * head_dim + d];
            }
            oh[d] = acc;
        }
    }
    free(score);
}
