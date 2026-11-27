/*
 * weights.c - Qwen weight binding via bind recipe (see weights.h)
 */

#include "qwen/weights.h"
#include "arch.h"
#include "bind_recipe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const st_tensor_t **qwen_global_slot(cp_weights_t *w, const char *field) {
    if (strcmp(field, "embed_tokens") == 0) return &w->embed_tokens;
    if (strcmp(field, "final_norm") == 0) return &w->final_norm;
    if (strcmp(field, "lm_head") == 0) return &w->lm_head;
    return NULL;
}

static const st_tensor_t **qwen_layer_slot(cp_layer_w_t *lw, const char *field) {
    if (strcmp(field, "input_ln") == 0) return &lw->input_ln;
    if (strcmp(field, "q_w") == 0) return &lw->q_w;
    if (strcmp(field, "q_b") == 0) return &lw->q_b;
    if (strcmp(field, "k_w") == 0) return &lw->k_w;
    if (strcmp(field, "k_b") == 0) return &lw->k_b;
    if (strcmp(field, "v_w") == 0) return &lw->v_w;
    if (strcmp(field, "v_b") == 0) return &lw->v_b;
    if (strcmp(field, "o_w") == 0) return &lw->o_w;
    if (strcmp(field, "post_ln") == 0) return &lw->post_ln;
    if (strcmp(field, "gate_w") == 0) return &lw->gate_w;
    if (strcmp(field, "up_w") == 0) return &lw->up_w;
    if (strcmp(field, "down_w") == 0) return &lw->down_w;
    return NULL;
}

static int bind_one(const st_file_t *f, const char *tag, st_dtype_t dt,
                    const cp_bind_dims_t *dims, const cp_bind_tensor_spec_t *spec,
                    uint32_t layer, const char *prefix,
                    const st_tensor_t **slot,
                    size_t *n_bound, size_t *n_expected) {
    if (!cp_bind_spec_active(spec, dims, (int)layer)) return 1;
    if (!slot) {
        fprintf(stderr, "%s: unknown field '%s'\n", tag, spec->field);
        return 0;
    }

    size_t shape[CP_BIND_MAX_DIM];
    int ndim = cp_bind_resolve_shape(dims, spec, shape);
    if (ndim < 0) return 0;

    char full[CP_BIND_NAME_MAX * 2];
    if (prefix && prefix[0]) {
        char leaf[CP_BIND_NAME_MAX];
        cp_bind_expand_name(leaf, sizeof leaf, spec->name, layer);
        snprintf(full, sizeof full, "%s%s", prefix, leaf);
    } else {
        cp_bind_expand_name(full, sizeof full, spec->name, layer);
    }

    *slot = cp_bind_expect(f, tag, full, dt, (size_t)ndim, shape,
                           n_bound, n_expected);
    return *slot != NULL;
}

int cp_weights_bind(const cp_config_t *cfg, const st_file_t *f, cp_weights_t *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !f) return -1;
    if (cfg->arch != CP_ARCH_DECODER_ONLY) {
        fprintf(stderr,
                "cp_weights: bind not implemented for strategy '%s' "
                "(model_type='%s')\n",
                cp_arch_name(cfg->arch), cfg->model_type);
        return -1;
    }

    cp_bind_recipe_t recipe;
    if (cp_bind_recipe_load(NULL, cfg->model_type, &recipe) != 0)
        return -1;

    cp_bind_dims_t dims;
    memset(&dims, 0, sizeof dims);
    dims.vocab_size = cfg->vocab_size;
    dims.hidden_size = cfg->hidden_size;
    dims.intermediate_size = cfg->intermediate_size;
    dims.kv_dim = cfg->head_dim * cfg->num_key_value_heads;
    dims.num_hidden_layers = cfg->num_hidden_layers;
    dims.tied_head = cfg->tie_word_embeddings ? 1 : 0;
    out->tied_head = dims.tied_head;

    st_dtype_t dt = cp_bind_resolve_dtype(&recipe, f);
    if (dt == ST_DTYPE_COUNT) {
        fprintf(stderr, "cp_weights: cannot resolve dtype from recipe\n");
        cp_bind_recipe_free(&recipe);
        return -1;
    }

    size_t n_bound = 0, n_expected = 0;
    int ok = 1;
    const char *tag = "cp_weights";

    for (size_t i = 0; i < recipe.n_globals; i++) {
        const cp_bind_tensor_spec_t *spec = &recipe.globals[i];
        const st_tensor_t **slot = qwen_global_slot(out, spec->field);
        if (!bind_one(f, tag, dt, &dims, spec, 0, NULL, slot,
                      &n_bound, &n_expected))
            ok = 0;
    }

    size_t n_layers = 0;
    if (cp_bind_resolve_sym(&dims, recipe.layers.count_sym, &n_layers) != 0) {
        cp_bind_recipe_free(&recipe);
        return -1;
    }
    out->layers = calloc(n_layers, sizeof(cp_layer_w_t));
    if (!out->layers) {
        fprintf(stderr, "cp_weights: out of memory\n");
        cp_bind_recipe_free(&recipe);
        return -1;
    }
    out->n_layers = (uint32_t)n_layers;

    for (uint32_t L = 0; L < out->n_layers; L++) {
        char prefix[CP_BIND_NAME_MAX];
        cp_bind_expand_name(prefix, sizeof prefix, recipe.layers.prefix, L);
        cp_layer_w_t *lw = &out->layers[L];
        for (size_t i = 0; i < recipe.layers.n_tensors; i++) {
            const cp_bind_tensor_spec_t *spec = &recipe.layers.tensors[i];
            const st_tensor_t **slot = qwen_layer_slot(lw, spec->field);
            if (!bind_one(f, tag, dt, &dims, spec, L, prefix, slot,
                          &n_bound, &n_expected))
                ok = 0;
        }
    }

    if (out->tied_head) {
        out->lm_head = out->embed_tokens;
        if (st_find(f, "lm_head.weight")) {
            fprintf(stderr,
                    "cp_weights: [WARN] tie_word_embeddings=1 but "
                    "lm_head.weight exists; using embed\n");
        }
    }

    out->n_bound = n_bound;
    out->n_expected = n_expected;
    cp_bind_recipe_free(&recipe);

    if (!ok) {
        cp_weights_free(out);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    return 0;
}

void cp_weights_print(const cp_weights_t *w) {
    printf("Weight binding: %zu/%zu tensors bound (%s head)\n",
           w->n_bound, w->n_expected, w->tied_head ? "tied" : "separate");
    printf("  %-46s %s\n", "model.embed_tokens.weight",
           w->embed_tokens ? "OK" : "MISS");
    for (uint32_t L = 0; L < w->n_layers; L++) {
        const cp_layer_w_t *lw = &w->layers[L];
        int layer_ok = lw->input_ln && lw->q_w && lw->q_b && lw->k_w && lw->k_b &&
                       lw->v_w && lw->v_b && lw->o_w && lw->post_ln &&
                       lw->gate_w && lw->up_w && lw->down_w;
        printf("  layer %-2u attn[q qb k kb v vb o] mlp[gate up down] ln[in post] : %s\n",
               L, layer_ok ? "OK" : "MISS");
    }
    printf("  %-46s %s\n", "model.norm.weight", w->final_norm ? "OK" : "MISS");
    printf("  %-46s %s\n", "lm_head.weight",
           w->lm_head ? (w->tied_head ? "OK (tied)" : "OK") : "MISS");
}

void cp_weights_free(cp_weights_t *w) {
    free(w->layers);
    w->layers = NULL;
    w->n_layers = 0;
}
