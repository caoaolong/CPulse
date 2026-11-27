/*
 * sample.c - shared token sampling (see sample.h)
 *
 * Greedy: argmax.
 * Otherwise: temperature softmax -> optional top-k -> optional top-p ->
 * multinomial draw (xorshift64 RNG).
 */

#include "sample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef CP_SAMPLE_DEFAULT_SEED
#define CP_SAMPLE_DEFAULT_SEED 0xC0FFEE5A1ULL
#endif

struct cp_sampler {
    cp_sample_cfg_t cfg;
    int             greedy;
    uint64_t        rng;
    uint32_t        cap;
    float          *prob; /* [cap] working probs / logits */
    uint32_t       *idx;  /* [cap] indices for top-k / top-p */
};

void cp_sample_cfg_default(cp_sample_cfg_t *cfg) {
    if (!cfg) return;
    cfg->temperature = 0.0f;
    cfg->top_k = 0;
    cfg->top_p = 1.0f;
    cfg->seed = 0;
}

int cp_sample_cfg_is_greedy(const cp_sample_cfg_t *cfg) {
    if (!cfg) return 1;
    if (cfg->temperature <= 0.0f) return 1;
    if (cfg->top_k == 1) return 1;
    return 0;
}

static uint64_t rng_u64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x ? x : CP_SAMPLE_DEFAULT_SEED;
    return *s;
}

static float rng_f32(uint64_t *s) {
    /* [0, 1) */
    return (float)(rng_u64(s) >> 11) * (1.0f / 9007199254740992.0f);
}

static uint32_t argmax_f(const float *v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

/* Descending float compare for qsort on (prob, idx) via idx array. */
static float *g_sort_prob;

static int cmp_idx_desc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    float pa = g_sort_prob[ia], pb = g_sort_prob[ib];
    if (pa < pb) return 1;
    if (pa > pb) return -1;
    return (ia > ib) - (ia < ib);
}

int cp_sampler_init(cp_sampler_t **out, uint32_t vocab_size,
                    const cp_sample_cfg_t *cfg) {
    if (!out || vocab_size == 0) return -1;
    *out = NULL;

    cp_sampler_t *s = calloc(1, sizeof *s);
    if (!s) return -1;

    if (cfg)
        s->cfg = *cfg;
    else
        cp_sample_cfg_default(&s->cfg);

    s->greedy = cp_sample_cfg_is_greedy(&s->cfg);
    s->rng = s->cfg.seed ? s->cfg.seed : CP_SAMPLE_DEFAULT_SEED;
    s->cap = vocab_size;

    if (!s->greedy) {
        s->prob = malloc((size_t)vocab_size * sizeof *s->prob);
        s->idx = malloc((size_t)vocab_size * sizeof *s->idx);
        if (!s->prob || !s->idx) {
            cp_sampler_free(s);
            return -1;
        }
    }

    *out = s;
    return 0;
}

void cp_sampler_free(cp_sampler_t *s) {
    if (!s) return;
    free(s->prob);
    free(s->idx);
    free(s);
}

uint32_t cp_sampler_next(cp_sampler_t *s, const float *logits, uint32_t n) {
    if (!s || !logits || n == 0) return 0;
    if (n > s->cap) n = s->cap;

    if (s->greedy)
        return argmax_f(logits, n);

    float temp = s->cfg.temperature;
    if (temp < 1e-6f) temp = 1e-6f;

    /* max for stability */
    float m = logits[0];
    for (uint32_t i = 1; i < n; i++)
        if (logits[i] > m) m = logits[i];

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float e = expf((logits[i] - m) / temp);
        s->prob[i] = e;
        sum += e;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t i = 0; i < n; i++) s->prob[i] *= inv;

    /* Build candidate list (all indices). */
    uint32_t n_keep = n;
    for (uint32_t i = 0; i < n; i++) s->idx[i] = i;

    /* top-k: keep k largest probs */
    int top_k = s->cfg.top_k;
    if (top_k > 0 && (uint32_t)top_k < n_keep) {
        g_sort_prob = s->prob;
        qsort(s->idx, n_keep, sizeof s->idx[0], cmp_idx_desc);
        n_keep = (uint32_t)top_k;
        /* zero the dropped mass by renormalizing over keepers */
        sum = 0.0f;
        for (uint32_t i = 0; i < n_keep; i++) sum += s->prob[s->idx[i]];
        inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint32_t i = 0; i < n_keep; i++) s->prob[s->idx[i]] *= inv;
    }

    /* top-p (nucleus): among current keepers, truncate cumulative mass */
    float top_p = s->cfg.top_p;
    if (top_p > 0.0f && top_p < 1.0f && n_keep > 1) {
        g_sort_prob = s->prob;
        qsort(s->idx, n_keep, sizeof s->idx[0], cmp_idx_desc);
        float cum = 0.0f;
        uint32_t cut = n_keep;
        for (uint32_t i = 0; i < n_keep; i++) {
            cum += s->prob[s->idx[i]];
            if (cum >= top_p) {
                cut = i + 1;
                break;
            }
        }
        n_keep = cut;
        sum = 0.0f;
        for (uint32_t i = 0; i < n_keep; i++) sum += s->prob[s->idx[i]];
        inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint32_t i = 0; i < n_keep; i++) s->prob[s->idx[i]] *= inv;
    }

    /* multinomial */
    float r = rng_f32(&s->rng);
    float c = 0.0f;
    for (uint32_t i = 0; i < n_keep; i++) {
        c += s->prob[s->idx[i]];
        if (r <= c) return s->idx[i];
    }
    return s->idx[n_keep - 1];
}
