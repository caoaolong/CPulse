/*
 * t5/decoder.h - T5 decoder single-step forward (CPU) with caches
 *
 * After set_memory(encoder_out):
 *   - cross-attn K/V are precomputed per layer and stay fixed
 *   - each step appends self-attn K/V at absolute position `pos`
 *   - returns tied lm_head logits [vocab] (with HF 1/sqrt(d_model) scale)
 */

#ifndef T5_DECODER_H
#define T5_DECODER_H

#include <stddef.h>
#include <stdint.h>

#include "t5/config.h"
#include "t5/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_t5_decoder cp_t5_decoder_t;

/*
 * max_src / max_tgt: capacity (0 => 512).
 * Requires FP32 weights and non-gated DenseReluDense.
 */
int cp_t5_decoder_init(cp_t5_decoder_t **out, const cp_t5_config_t *cfg,
                       const cp_t5_weights_t *w, size_t max_src,
                       size_t max_tgt);

void cp_t5_decoder_free(cp_t5_decoder_t *d);

/*
 * Bind encoder memory [src_len, d_model] and precompute cross K/V.
 * Copies into an internal buffer; encoder may be freed afterwards.
 * Resets self-attn cache write state.
 */
int cp_t5_decoder_set_memory(cp_t5_decoder_t *d, const float *mem,
                             size_t src_len);

/* Drop self-attn cache length; cross K/V / memory stay. */
void cp_t5_decoder_reset(cp_t5_decoder_t *d);

/*
 * One decoder token at absolute position pos (0-based).
 * First call typically uses decoder_start_token_id at pos=0.
 * Returns logits [vocab_size] owned by d, or NULL on failure.
 */
const float *cp_t5_decoder_step(cp_t5_decoder_t *d, uint32_t tok, size_t pos);

#ifdef __cplusplus
}
#endif

#endif /* T5_DECODER_H */
