/*
 * t5/generate.h - T5 Enc–Dec generation session
 *
 * encode(src) -> memory; then decode from decoder_start_token_id until
 * eos / max_new. Next-token choice is shared via sample.h.
 */

#ifndef T5_GENERATE_H
#define T5_GENERATE_H

#include <stddef.h>
#include <stdint.h>

#include "sample.h"
#include "t5/config.h"
#include "t5/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_t5_session cp_t5_session_t;

typedef void (*cp_t5_gen_token_fn)(uint32_t id, size_t pos, void *userdata);

/*
 * Owns encoder + decoder scratch. max_src / max_tgt: 0 => 512.
 * Returns 0 on success; on failure *out stays NULL.
 */
int cp_t5_session_init(cp_t5_session_t **out, const cp_t5_config_t *cfg,
                       const cp_t5_weights_t *w, size_t max_src,
                       size_t max_tgt);

void cp_t5_session_free(cp_t5_session_t *s);

/*
 * Encode `src[0..src_n)`, then decode up to `max_new` tokens.
 * `sample` NULL => greedy. Generated ids (including terminating eos when
 * hit) go to out_ids when non-NULL. on_token is invoked for each emitted
 * id except eos (stored but not reported).
 *
 * Returns number of generated tokens (>= 0, includes eos if produced),
 * or -1 on error.
 */
int cp_t5_generate(cp_t5_session_t *s, const uint32_t *src, size_t src_n,
                   uint32_t *out_ids, size_t out_cap, int max_new,
                   const cp_sample_cfg_t *sample,
                   cp_t5_gen_token_fn on_token, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* T5_GENERATE_H */
