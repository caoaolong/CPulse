/*
 * sample.h - shared next-token sampling (all architectures)
 *
 * Operates only on logits[V]. Temperature / top-k / top-p / greedy are
 * model-agnostic; Qwen and T5 generate both call into this module.
 */

#ifndef SAMPLE_H
#define SAMPLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    temperature; /* <= 0 => greedy argmax */
    int      top_k;       /* <= 0 => off; 1 => greedy */
    float    top_p;       /* <= 0 or >= 1 => off (keep all) */
    uint64_t seed;        /* RNG seed when not greedy; 0 => default */
} cp_sample_cfg_t;

/* Fill cfg with greedy defaults (temp=0, top_k=0, top_p=1, seed=0). */
void cp_sample_cfg_default(cp_sample_cfg_t *cfg);

/* 1 when sampling degenerates to argmax. */
int cp_sample_cfg_is_greedy(const cp_sample_cfg_t *cfg);

/*
 * Stateful sampler: scratch buffers + RNG for one generate run.
 * vocab_size is the maximum n passed to cp_sampler_next.
 */
typedef struct cp_sampler cp_sampler_t;

int  cp_sampler_init(cp_sampler_t **out, uint32_t vocab_size,
                     const cp_sample_cfg_t *cfg);
void cp_sampler_free(cp_sampler_t *s);

/*
 * Draw one token from logits[0..n). n must be <= vocab_size from init.
 * NULL cfg on init means greedy. Never returns an out-of-range id.
 */
uint32_t cp_sampler_next(cp_sampler_t *s, const float *logits, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* SAMPLE_H */
