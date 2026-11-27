/*
 * bind_recipe.h - data-driven weight bind recipes from config/routes.json
 *
 * The "bind" section maps model_type -> tensor name patterns, shape
 * symbols, and logical field ids. Architecture-specific code still
 * assigns those fields onto typed weight structs.
 */

#ifndef BIND_RECIPE_H
#define BIND_RECIPE_H

#include "safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CP_BIND_MAX_DIM     4
#define CP_BIND_NAME_MAX    160
#define CP_BIND_FIELD_MAX   48
#define CP_BIND_SYM_MAX     32

typedef struct {
    char name[CP_BIND_NAME_MAX];   /* may contain {L} */
    char field[CP_BIND_FIELD_MAX];
    char shape_sym[CP_BIND_MAX_DIM][CP_BIND_SYM_MAX];
    int  ndim;
    int  only_layer;   /* -1 = all layers; else only this layer index */
    int  if_not_tied;  /* 1 => skip when tied_head */
    int  if_gated;     /* 1 => only gated FF; -1 => only non-gated; 0 always */
} cp_bind_tensor_spec_t;

typedef struct {
    char count_sym[CP_BIND_SYM_MAX];
    char prefix[CP_BIND_NAME_MAX]; /* may end with {L}. or contain {L} */
    cp_bind_tensor_spec_t *tensors;
    size_t n_tensors;
} cp_bind_layer_group_t;

typedef struct {
    char key[32];                  /* resolved recipe key, e.g. qwen2 / t5 */
    char dtype_name[16];           /* "BF16" / "F32" / empty */
    char dtype_from[CP_BIND_NAME_MAX]; /* tensor name to peek dtype; empty if unused */
    int  tied_head_from_config;    /* qwen: honor tie_word_embeddings */

    cp_bind_tensor_spec_t *globals;
    size_t n_globals;

    cp_bind_layer_group_t layers;          /* qwen single stack; unused if n_tensors=0 */
    cp_bind_layer_group_t encoder_layers;  /* t5 */
    cp_bind_layer_group_t decoder_layers;  /* t5 */
} cp_bind_recipe_t;

/* Dimension / flag context for resolving shape symbols. */
typedef struct {
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t kv_dim;
    uint32_t num_hidden_layers;
    uint32_t d_model;
    uint32_t d_ff;
    uint32_t inner_dim;
    uint32_t num_heads;
    uint32_t num_layers;
    uint32_t num_decoder_layers;
    uint32_t rel_buckets;
    int      tied_head;
    int      gated_ff;
} cp_bind_dims_t;

/* Remember routes path for subsequent load(NULL, ...) calls. */
void cp_bind_set_routes_path(const char *path);

/*
 * Load bind.<model_type> from routes JSON (follows string aliases).
 * path NULL => last cp_bind_set_routes_path / CP_DEFAULT_ROUTES.
 * Returns 0 on success.
 */
int cp_bind_recipe_load(const char *routes_path, const char *model_type,
                        cp_bind_recipe_t *out);

void cp_bind_recipe_free(cp_bind_recipe_t *r);

/* Resolve a shape symbol against dims; returns 0 on success. */
int cp_bind_resolve_sym(const cp_bind_dims_t *dims, const char *sym,
                        size_t *out);

/* Build concrete shape[] from spec; returns ndim or -1. */
int cp_bind_resolve_shape(const cp_bind_dims_t *dims,
                          const cp_bind_tensor_spec_t *spec,
                          size_t shape[CP_BIND_MAX_DIM]);

/* Expand "{L}" in pattern into buf. */
void cp_bind_expand_name(char *buf, size_t cap, const char *pattern,
                         uint32_t layer);

/* Resolve recipe dtype (fixed name or from tensor). */
st_dtype_t cp_bind_resolve_dtype(const cp_bind_recipe_t *r, const st_file_t *f);

/*
 * Look up + check one tensor. Increments n_expected; n_bound on success.
 * tag is a short prefix for diagnostics (e.g. "cp_weights").
 */
const st_tensor_t *cp_bind_expect(const st_file_t *f, const char *tag,
                                  const char *name, st_dtype_t dtype,
                                  size_t ndim, const size_t *shape,
                                  size_t *n_bound, size_t *n_expected);

/* Whether a tensor spec should run for this layer / dims. */
int cp_bind_spec_active(const cp_bind_tensor_spec_t *spec,
                        const cp_bind_dims_t *dims, int layer_index);

#ifdef __cplusplus
}
#endif

#endif /* BIND_RECIPE_H */
