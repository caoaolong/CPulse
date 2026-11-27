/*
 * forward.h - decoder forward pass for CPulse (CPU)
 *
 * Assembles the existing ops (embed/matvec/rmsnorm/rope/attention/
 * swiglu) into the full Qwen2 stack: per-layer pre-norm attention with
 * GQA + per-layer KV caches, pre-norm SwiGLU MLP, final RMSNorm and the
 * tied lm_head. One call runs one token at one absolute position and
 * returns the logits row for sampling.
 *
 * Lifetime: keeps pointers to cfg/w (caller must keep them alive);
 * kv caches and scratch buffers are owned here.
 */

#ifndef FORWARD_H
#define FORWARD_H

#include <stddef.h>
#include <stdint.h>

#include "qwen/model.h"
#include "qwen/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_forward cp_forward_t;

/*
 * Allocate per-layer KV caches and scratch buffers. When use_gpu != 0
 * the weights are additionally uploaded once and every step runs on
 * the GPU (kv caches and intermediates stay on device); if the GPU
 * bring-up fails the function warns and stays on the CPU path.
 * Returns 0 on success; on failure prints a diagnostic, frees
 * everything and returns -1 (*out stays NULL).
 */
int  cp_forward_init(cp_forward_t **out, const cp_config_t *cfg,
                     const cp_weights_t *w, int use_gpu);

/* 1 when steps run on the GPU, 0 for the CPU path. */
int  cp_forward_on_gpu(const cp_forward_t *f);

/* Release all caches and buffers. Safe on NULL. */
void cp_forward_free(cp_forward_t *f);

/*
 * Run token `tok` at absolute position `pos` through every layer,
 * appending its k/v to the caches. Returns the logits vector
 * (cfg->vocab_size floats, owned by f, valid until the next call).
 */
const float *cp_forward_step(cp_forward_t *f, uint32_t tok, size_t pos);

/* Tokens cached since init/reset (all layers advance together). */
size_t cp_forward_len(const cp_forward_t *f);

/* Drop all cached k/v for a fresh sequence (scratch is re-used). */
void cp_forward_reset(cp_forward_t *f);

#ifdef __cplusplus
}
#endif

#endif /* FORWARD_H */
