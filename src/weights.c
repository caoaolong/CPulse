/*
 * weights.c - weight binding (see weights.h)
 *
 * Every expected tensor is looked up by its HuggingFace name and checked
 * against the config-derived shape. Anything off fails the bind: later
 * stages must be able to trust these pointers blindly.
 */

#include "weights.h"
#include "arch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Look up one tensor and verify dtype + shape. NULL (counted) on any problem. */
static const st_tensor_t *expect(
        const st_file_t *f, const char *name,
        st_dtype_t dtype, size_t ndim, const size_t *shape,
        size_t *n_bound, size_t *n_expected) {
    (*n_expected)++;
    const st_tensor_t *t = st_find(f, name);
    if (!t) {
        fprintf(stderr, "cp_weights: [MISS] %-46s not in file\n", name);
        return NULL;
    }
    if (t->dtype != dtype) {
        fprintf(stderr, "cp_weights: [DTYPE] %-46s is %s, want %s\n",
                name, st_dtype_name(t->dtype), st_dtype_name(dtype));
        return NULL;
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "cp_weights: [RANK] %-46s ndim=%zu, want %zu\n",
                name, t->ndim, ndim);
        return NULL;
    }
    for (size_t d = 0; d < ndim; d++) {
        if (t->shape[d] != shape[d]) {
            fprintf(stderr, "cp_weights: [SHAPE] %-46s dim%zu=%zu, want %zu\n",
                    name, d, t->shape[d], shape[d]);
            return NULL;
        }
    }
    (*n_bound)++;
    return t;
}

#define EXPECT2(f, n, dt, a, b, nb, ne) \
    expect(f, n, dt, 2, ((const size_t[]){ (a), (b) }), nb, ne)
#define EXPECT1(f, n, dt, a, nb, ne) \
    expect(f, n, dt, 1, ((const size_t[]){ (a) }), nb, ne)

int cp_weights_bind(const cp_config_t *cfg, const st_file_t *f, cp_weights_t *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !f) return -1;
    if (cfg->arch != CP_ARCH_QWEN2) {
        fprintf(stderr,
                "cp_weights: bind not implemented for arch '%s' (model_type='%s')\n",
                cp_arch_name(cfg->arch), cfg->model_type);
        return -1;
    }
    out->tied_head = cfg->tie_word_embeddings ? 1 : 0;

    const st_dtype_t dt = ST_DTYPE_BF16;
    const uint32_t H = cfg->hidden_size;
    const uint32_t I = cfg->intermediate_size;
    const uint32_t V = cfg->vocab_size;
    const uint32_t kv_dim = cfg->head_dim * cfg->num_key_value_heads;

    size_t n_bound = 0, n_expected = 0;
    int ok = 1;

    out->embed_tokens = EXPECT2(f, "model.embed_tokens.weight", dt, V, H, &n_bound, &n_expected);
    ok &= out->embed_tokens != NULL;

    out->layers = calloc(cfg->num_hidden_layers, sizeof(cp_layer_w_t));
    if (!out->layers) {
        fprintf(stderr, "cp_weights: out of memory\n");
        return -1;
    }
    out->n_layers = cfg->num_hidden_layers;

    char name[128];
    for (uint32_t L = 0; L < cfg->num_hidden_layers; L++) {
        cp_layer_w_t *lw = &out->layers[L];
#define LN(suffix, field, ...) \
        do { \
            snprintf(name, sizeof(name), "model.layers.%u." suffix, L); \
            lw->field = __VA_ARGS__; \
            ok &= lw->field != NULL; \
        } while (0)
        LN("input_layernorm.weight", input_ln, EXPECT1(f, name, dt, H, &n_bound, &n_expected));
        LN("self_attn.q_proj.weight", q_w, EXPECT2(f, name, dt, H, H, &n_bound, &n_expected));
        LN("self_attn.q_proj.bias", q_b, EXPECT1(f, name, dt, H, &n_bound, &n_expected));
        LN("self_attn.k_proj.weight", k_w, EXPECT2(f, name, dt, kv_dim, H, &n_bound, &n_expected));
        LN("self_attn.k_proj.bias", k_b, EXPECT1(f, name, dt, kv_dim, &n_bound, &n_expected));
        LN("self_attn.v_proj.weight", v_w, EXPECT2(f, name, dt, kv_dim, H, &n_bound, &n_expected));
        LN("self_attn.v_proj.bias", v_b, EXPECT1(f, name, dt, kv_dim, &n_bound, &n_expected));
        LN("self_attn.o_proj.weight", o_w, EXPECT2(f, name, dt, H, H, &n_bound, &n_expected));
        LN("post_attention_layernorm.weight", post_ln, EXPECT1(f, name, dt, H, &n_bound, &n_expected));
        LN("mlp.gate_proj.weight", gate_w, EXPECT2(f, name, dt, I, H, &n_bound, &n_expected));
        LN("mlp.up_proj.weight", up_w, EXPECT2(f, name, dt, I, H, &n_bound, &n_expected));
        LN("mlp.down_proj.weight", down_w, EXPECT2(f, name, dt, H, I, &n_bound, &n_expected));
#undef LN
    }

    out->final_norm = EXPECT1(f, "model.norm.weight", dt, H, &n_bound, &n_expected);
    ok &= out->final_norm != NULL;

    if (out->tied_head) {
        /* tied: no separate tensor; alias embed so forward code is uniform */
        out->lm_head = out->embed_tokens;
        if (st_find(f, "lm_head.weight")) {
            fprintf(stderr, "cp_weights: [WARN] tie_word_embeddings=1 but lm_head.weight exists; using embed\n");
        }
    } else {
        out->lm_head = EXPECT2(f, "lm_head.weight", dt, V, H, &n_bound, &n_expected);
        ok &= out->lm_head != NULL;
    }

    out->n_bound = n_bound;
    out->n_expected = n_expected;
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
