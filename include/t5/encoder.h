/*
 * t5/encoder.h - T5 encoder forward (CPU)
 *
 * src_ids -> shared embed -> N encoder blocks (pre-norm self-attn + FFN)
 * -> final layer norm -> memory[S, d_model].
 */

#ifndef T5_ENCODER_H
#define T5_ENCODER_H

#include <stddef.h>
#include <stdint.h>

#include "t5/config.h"
#include "t5/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_t5_encoder cp_t5_encoder_t;

/*
 * Allocate scratch for sequences up to max_seq (0 => 512).
 * Keeps pointers to cfg/w (caller must keep them alive).
 * Requires FP32 weights and non-gated DenseReluDense for now.
 */
int cp_t5_encoder_init(cp_t5_encoder_t **out, const cp_t5_config_t *cfg,
                       const cp_t5_weights_t *w, size_t max_seq);

void cp_t5_encoder_free(cp_t5_encoder_t *e);

/*
 * Run the full encoder. n must be in (0, max_seq].
 * Returns pointer to memory [n, d_model] owned by e (valid until next
 * run/free), or NULL on failure.
 */
const float *cp_t5_encoder_run(cp_t5_encoder_t *e, const uint32_t *src_ids,
                               size_t n);

size_t cp_t5_encoder_seq_len(const cp_t5_encoder_t *e);

#ifdef __cplusplus
}
#endif

#endif /* T5_ENCODER_H */
