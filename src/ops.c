/*
 * ops.c - reference math operators, CPU (see ops.h)
 */

#include "ops.h"

#include <math.h>
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
