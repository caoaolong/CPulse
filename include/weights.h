/*
 * weights.h - bind safetensors tensors onto the model architecture
 *
 * Project : CPulse - C inference engine for Qwen2.5-0.5B
 * Stage   : 3 - weight mapping
 *
 * Takes cp_config_t (shapes) + st_file_t (flat tensor table) and resolves
 * every expected weight by name. No data is copied: the pointers alias
 * the file buffer owned by st_file_t (must outlive cp_weights_t).
 * Every tensor is checked for presence, rank, shape and dtype (BF16).
 */

#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "model.h"
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/* one decoder layer's weights (Qwen2 GQA + SwiGLU, q/k/v biases) */
typedef struct {
    const st_tensor_t *input_ln;
    const st_tensor_t *q_w, *q_b;
    const st_tensor_t *k_w, *k_b;
    const st_tensor_t *v_w, *v_b;
    const st_tensor_t *o_w;
    const st_tensor_t *post_ln;
    const st_tensor_t *gate_w, *up_w, *down_w;
} cp_layer_w_t;

typedef struct {
    cp_layer_w_t *layers; /* [n_layers] */
    uint32_t      n_layers;
    const st_tensor_t *embed_tokens;
    const st_tensor_t *final_norm;
    const st_tensor_t *lm_head; /* NULL when tie_word_embeddings */
    int           tied_head;
    size_t        n_bound;    /* tensors bound OK */
    size_t        n_expected; /* names looked up */
} cp_weights_t;

/*
 * Bind all weights. Returns 0 when every expected tensor is present
 * with the right shape/dtype; on any problem prints diagnostics to
 * stderr and returns -1 (cp_weights_t is left empty, safe to free).
 */
int cp_weights_bind(const cp_config_t *cfg, const st_file_t *f, cp_weights_t *out);

/* Traffic-light report: one line per layer + embed/norm/head. */
void cp_weights_print(const cp_weights_t *w);

/* Release the layer array (tensors themselves belong to st_file_t). */
void cp_weights_free(cp_weights_t *w);

#ifdef __cplusplus
}
#endif

#endif /* WEIGHTS_H */
