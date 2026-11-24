/*
 * generate.h - generation session API for CPulse
 *
 * Owns a cp_forward_t and runs prefill + greedy decode. Callers keep
 * tokenization / printing outside: pass prompt ids in, get new ids out
 * (and an optional per-token callback for streaming IO).
 *
 * Phase A1: greedy argmax only. Sampling (temp / top-k / top-p) lands in A2.
 */

#ifndef GENERATE_H
#define GENERATE_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"
#include "weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_session cp_session_t;

/* Invoked once per newly generated token, before the next forward step.
 * `pos` is the absolute position that token will occupy in the sequence. */
typedef void (*cp_gen_token_fn)(uint32_t id, size_t pos, void *userdata);

/*
 * Allocate forward state (KV caches + scratch). use_gpu mirrors
 * cp_forward_init. Returns 0 on success; on failure *out stays NULL.
 */
int  cp_session_init(cp_session_t **out, const cp_config_t *cfg,
                     const cp_weights_t *w, int use_gpu);

/* 1 when the underlying forward runs on GPU, else 0. */
int  cp_session_on_gpu(const cp_session_t *s);

/* Drop KV state for a fresh prompt. Safe on NULL. */
void cp_session_reset(cp_session_t *s);

/* Release session and forward. Safe on NULL. */
void cp_session_free(cp_session_t *s);

/*
 * Prefill `prompt[0..prompt_n)`, then greedy-decode up to `max_new` tokens.
 * Each new token is stored in out_ids (when non-NULL and within out_cap)
 * and reported via on_token (optional). Stops on eos, Qwen <|im_end|>,
 * max_new, or a forward failure.
 *
 * Returns the number of generated tokens (>= 0), or -1 on error
 * (null args, empty prompt, prefill failure).
 */
int cp_generate(cp_session_t *s,
                const uint32_t *prompt, size_t prompt_n,
                uint32_t *out_ids, size_t out_cap,
                int max_new,
                cp_gen_token_fn on_token, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* GENERATE_H */
