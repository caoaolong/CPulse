/*
 * model.h - model configuration (config.json) for CPulse
 *
 * Project : CPulse - C inference engine for Qwen2.5-0.5B
 * Stage   : 3 - config loading
 *
 * Holds the architecture hyperparameters parsed from the model's
 * config.json, plus the derived values the forward pass needs
 * (strategy id, head_dim, kv repeat factor). Validation maps model_type
 * through arch_routes.json; stub strategies fail early, implemented
 * ones get full shape / activation checks.
 */

#ifndef MODEL_H
#define MODEL_H

#include "arch.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* raw fields from config.json */
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t max_position_embeddings;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    double   rms_norm_eps;
    double   rope_theta;
    int      tie_word_embeddings;
    char     hidden_act[16];
    char     model_type[32];
    char     torch_dtype[16];

    /* derived (filled by cp_config_validate) */
    cp_arch_id_t arch;    /* from model_type via cp_arch_resolve */
    uint32_t head_dim;    /* hidden_size / num_attention_heads */
    uint32_t kv_repeat;   /* num_attention_heads / num_key_value_heads (GQA) */
} cp_config_t;

/*
 * Load config.json from disk. Returns 0 on success (including
 * validation); on failure prints a diagnostic to stderr and returns -1.
 */
int cp_config_load(const char *path, cp_config_t *out);

/* One-line-per-field dump of raw + derived values. */
void cp_config_print(const cp_config_t *c);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
