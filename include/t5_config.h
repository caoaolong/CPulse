/*
 * t5_config.h - T5 / encoder-decoder config (independent of Qwen cp_config_t)
 */

#ifndef T5_CONFIG_H
#define T5_CONFIG_H

#include "arch.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* from config.json */
    uint32_t vocab_size;
    uint32_t d_model;
    uint32_t d_ff;
    uint32_t d_kv;          /* per-head key/value dim */
    uint32_t num_heads;
    uint32_t num_layers;    /* encoder layers; decoder usually matches */
    uint32_t num_decoder_layers; /* 0 => same as num_layers */
    uint32_t relative_attention_num_buckets;
    uint32_t decoder_start_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    double   layer_norm_epsilon;
    int      is_encoder_decoder;
    char     model_type[32];
    char     torch_dtype[16];
    char     feed_forward_proj[32]; /* e.g. "relu" / "gated-gelu"; optional */

    /* derived */
    cp_arch_id_t arch;      /* must be CP_ARCH_ENCODER_DECODER */
    uint32_t     inner_dim; /* num_heads * d_kv (expect == d_model for classic T5) */
} cp_t5_config_t;

/*
 * Load and validate a T5-family config.json.
 * Only model_type in {t5, mt5, flan-t5} is accepted for now.
 * Returns 0 on success; on failure prints to stderr and returns -1.
 */
int cp_t5_config_load(const char *path, cp_t5_config_t *out);

void cp_t5_config_print(const cp_t5_config_t *c);

#ifdef __cplusplus
}
#endif

#endif /* T5_CONFIG_H */
