/*
 * t5_weights.h - bind T5 safetensors tensors onto encoder/decoder layers
 *
 * Zero-copy: pointers alias st_file_t buffer (must outlive cp_t5_weights_t).
 * Relative bias exists only on block 0 in the file; after bind, layers 1..N
 * alias layer 0's pointer so forward can always read lw->self_attn.rel_bias.
 * Orphan EncDecAttention.relative_attention_bias (HF unused) is not required.
 */

#ifndef T5_WEIGHTS_H
#define T5_WEIGHTS_H

#include "safetensors.h"
#include "t5/config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const st_tensor_t *q;
    const st_tensor_t *k;
    const st_tensor_t *v;
    const st_tensor_t *o;
    const st_tensor_t *rel_bias; /* SelfAttention only; may be aliased */
} cp_t5_attn_w_t;

typedef struct {
    const st_tensor_t *wi;   /* relu: DenseReluDense.wi */
    const st_tensor_t *wi_0; /* gated-gelu */
    const st_tensor_t *wi_1;
    const st_tensor_t *wo;
    int gated; /* 1 => wi_0/wi_1, 0 => wi */
} cp_t5_ff_w_t;

typedef struct {
    cp_t5_attn_w_t self_attn;
    const st_tensor_t *ln_self; /* layer.0.layer_norm */
    cp_t5_ff_w_t ff;
    const st_tensor_t *ln_ff; /* layer.1.layer_norm */
} cp_t5_enc_layer_w_t;

typedef struct {
    cp_t5_attn_w_t self_attn;
    const st_tensor_t *ln_self; /* layer.0.layer_norm */
    cp_t5_attn_w_t cross_attn;
    const st_tensor_t *ln_cross; /* layer.1.layer_norm */
    cp_t5_ff_w_t ff;
    const st_tensor_t *ln_ff; /* layer.2.layer_norm */
} cp_t5_dec_layer_w_t;

typedef struct {
    const st_tensor_t *shared; /* tied embed + lm_head */
    cp_t5_enc_layer_w_t *encoder;
    uint32_t n_encoder;
    const st_tensor_t *enc_final_ln;
    cp_t5_dec_layer_w_t *decoder;
    uint32_t n_decoder;
    const st_tensor_t *dec_final_ln;
    st_dtype_t dtype;
    int gated_ff;
    size_t n_bound;
    size_t n_expected;
} cp_t5_weights_t;

/*
 * Bind all T5 weights. Returns 0 when every expected tensor matches
 * dtype/shape; on failure prints diagnostics and leaves *out empty.
 */
int cp_t5_weights_bind(const cp_t5_config_t *cfg, const st_file_t *f,
                       cp_t5_weights_t *out);

void cp_t5_weights_print(const cp_t5_weights_t *w);

void cp_t5_weights_free(cp_t5_weights_t *w);

#ifdef __cplusplus
}
#endif

#endif /* T5_WEIGHTS_H */
