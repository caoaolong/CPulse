/*
 * t5_weights.c - T5 weight binding via bind recipe (see t5/weights.h)
 */

#include "t5/weights.h"
#include "bind_recipe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_gated_ff(const char *feed_forward_proj) {
    return feed_forward_proj && strstr(feed_forward_proj, "gated") != NULL;
}

static const st_tensor_t **t5_global_slot(cp_t5_weights_t *w, const char *field) {
    if (strcmp(field, "shared") == 0) return &w->shared;
    if (strcmp(field, "enc_final_ln") == 0) return &w->enc_final_ln;
    if (strcmp(field, "dec_final_ln") == 0) return &w->dec_final_ln;
    return NULL;
}

static const st_tensor_t **t5_enc_slot(cp_t5_enc_layer_w_t *lw,
                                      const char *field) {
    if (strcmp(field, "enc_self_q") == 0) return &lw->self_attn.q;
    if (strcmp(field, "enc_self_k") == 0) return &lw->self_attn.k;
    if (strcmp(field, "enc_self_v") == 0) return &lw->self_attn.v;
    if (strcmp(field, "enc_self_o") == 0) return &lw->self_attn.o;
    if (strcmp(field, "enc_self_rel_bias") == 0) return &lw->self_attn.rel_bias;
    if (strcmp(field, "enc_ln_self") == 0) return &lw->ln_self;
    if (strcmp(field, "enc_ff_wi") == 0) return &lw->ff.wi;
    if (strcmp(field, "enc_ff_wi_0") == 0) return &lw->ff.wi_0;
    if (strcmp(field, "enc_ff_wi_1") == 0) return &lw->ff.wi_1;
    if (strcmp(field, "enc_ff_wo") == 0) return &lw->ff.wo;
    if (strcmp(field, "enc_ln_ff") == 0) return &lw->ln_ff;
    return NULL;
}

static const st_tensor_t **t5_dec_slot(cp_t5_dec_layer_w_t *lw,
                                      const char *field) {
    if (strcmp(field, "dec_self_q") == 0) return &lw->self_attn.q;
    if (strcmp(field, "dec_self_k") == 0) return &lw->self_attn.k;
    if (strcmp(field, "dec_self_v") == 0) return &lw->self_attn.v;
    if (strcmp(field, "dec_self_o") == 0) return &lw->self_attn.o;
    if (strcmp(field, "dec_self_rel_bias") == 0) return &lw->self_attn.rel_bias;
    if (strcmp(field, "dec_ln_self") == 0) return &lw->ln_self;
    if (strcmp(field, "dec_cross_q") == 0) return &lw->cross_attn.q;
    if (strcmp(field, "dec_cross_k") == 0) return &lw->cross_attn.k;
    if (strcmp(field, "dec_cross_v") == 0) return &lw->cross_attn.v;
    if (strcmp(field, "dec_cross_o") == 0) return &lw->cross_attn.o;
    if (strcmp(field, "dec_ln_cross") == 0) return &lw->ln_cross;
    if (strcmp(field, "dec_ff_wi") == 0) return &lw->ff.wi;
    if (strcmp(field, "dec_ff_wi_0") == 0) return &lw->ff.wi_0;
    if (strcmp(field, "dec_ff_wi_1") == 0) return &lw->ff.wi_1;
    if (strcmp(field, "dec_ff_wo") == 0) return &lw->ff.wo;
    if (strcmp(field, "dec_ln_ff") == 0) return &lw->ln_ff;
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
    char leaf[CP_BIND_NAME_MAX];
    cp_bind_expand_name(leaf, sizeof leaf, spec->name, layer);
    snprintf(full, sizeof full, "%s%s", prefix, leaf);

    *slot = cp_bind_expect(f, tag, full, dt, (size_t)ndim, shape,
                           n_bound, n_expected);
    return *slot != NULL;
}

static int bind_layer_group(const st_file_t *f, const char *tag, st_dtype_t dt,
                            const cp_bind_dims_t *dims,
                            const cp_bind_layer_group_t *group,
                            int is_encoder, cp_t5_weights_t *out,
                            size_t *n_bound, size_t *n_expected) {
    if (!group || group->n_tensors == 0) return 1;

    size_t n_layers = 0;
    if (cp_bind_resolve_sym(dims, group->count_sym, &n_layers) != 0)
        return 0;

    if (is_encoder) {
        out->encoder = calloc(n_layers, sizeof(cp_t5_enc_layer_w_t));
        if (!out->encoder) return 0;
        out->n_encoder = (uint32_t)n_layers;
    } else {
        out->decoder = calloc(n_layers, sizeof(cp_t5_dec_layer_w_t));
        if (!out->decoder) return 0;
        out->n_decoder = (uint32_t)n_layers;
    }

    int ok = 1;
    for (uint32_t L = 0; L < (uint32_t)n_layers; L++) {
        char prefix[CP_BIND_NAME_MAX];
        cp_bind_expand_name(prefix, sizeof prefix, group->prefix, L);
        for (size_t i = 0; i < group->n_tensors; i++) {
            const cp_bind_tensor_spec_t *spec = &group->tensors[i];
            const st_tensor_t **slot = is_encoder
                    ? t5_enc_slot(&out->encoder[L], spec->field)
                    : t5_dec_slot(&out->decoder[L], spec->field);
            if (!bind_one(f, tag, dt, dims, spec, L, prefix, slot,
                          n_bound, n_expected))
                ok = 0;
        }
        if (is_encoder)
            out->encoder[L].ff.gated = dims->gated_ff;
        else
            out->decoder[L].ff.gated = dims->gated_ff;
    }
    return ok;
}

int cp_t5_weights_bind(const cp_t5_config_t *cfg, const st_file_t *f,
                       cp_t5_weights_t *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !f) return -1;
    if (cfg->arch != CP_ARCH_ENCODER_DECODER) {
        fprintf(stderr,
                "cp_t5_weights: bind expects encoder-decoder, got '%s'\n",
                cp_arch_name(cfg->arch));
        return -1;
    }

    cp_bind_recipe_t recipe;
    if (cp_bind_recipe_load(NULL, cfg->model_type, &recipe) != 0)
        return -1;

    cp_bind_dims_t dims;
    memset(&dims, 0, sizeof dims);
    dims.vocab_size = cfg->vocab_size;
    dims.d_model = cfg->d_model;
    dims.d_ff = cfg->d_ff;
    dims.inner_dim = cfg->inner_dim;
    dims.num_heads = cfg->num_heads;
    dims.num_layers = cfg->num_layers;
    dims.num_decoder_layers = cfg->num_decoder_layers;
    dims.rel_buckets = cfg->relative_attention_num_buckets;
    dims.gated_ff = is_gated_ff(cfg->feed_forward_proj);
    out->gated_ff = dims.gated_ff;

    st_dtype_t dt = cp_bind_resolve_dtype(&recipe, f);
    if (dt == ST_DTYPE_COUNT) {
        fprintf(stderr,
                "cp_t5_weights: shared.weight missing or unsupported dtype "
                "(need F32/BF16/F16)\n");
        cp_bind_recipe_free(&recipe);
        return -1;
    }
    out->dtype = dt;

    size_t n_bound = 0, n_expected = 0;
    int ok = 1;
    const char *tag = "cp_t5_weights";

    for (size_t i = 0; i < recipe.n_globals; i++) {
        const cp_bind_tensor_spec_t *spec = &recipe.globals[i];
        const st_tensor_t **slot = t5_global_slot(out, spec->field);
        if (!bind_one(f, tag, dt, &dims, spec, 0, "", slot,
                      &n_bound, &n_expected))
            ok = 0;
    }

    if (!bind_layer_group(f, tag, dt, &dims, &recipe.encoder_layers, 1, out,
                          &n_bound, &n_expected))
        ok = 0;
    if (!bind_layer_group(f, tag, dt, &dims, &recipe.decoder_layers, 0, out,
                          &n_bound, &n_expected))
        ok = 0;

    /* Layers 1..N reuse block-0 relative bias (HF behaviour). */
    for (uint32_t L = 1; L < out->n_encoder; L++)
        out->encoder[L].self_attn.rel_bias =
                out->encoder[0].self_attn.rel_bias;
    for (uint32_t L = 1; L < out->n_decoder; L++)
        out->decoder[L].self_attn.rel_bias =
                out->decoder[0].self_attn.rel_bias;

    if (st_find(f, "decoder.block.0.layer.1.EncDecAttention."
                   "relative_attention_bias.weight")) {
        fprintf(stderr,
                "cp_t5_weights: [NOTE] EncDecAttention.relative_attention_bias "
                "present but unused (HF ignores it)\n");
    }

    out->n_bound = n_bound;
    out->n_expected = n_expected;
    cp_bind_recipe_free(&recipe);

    if (!ok) {
        cp_t5_weights_free(out);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    return 0;
}

void cp_t5_weights_print(const cp_t5_weights_t *w) {
    printf("T5 weight binding: %zu/%zu tensors bound (dtype=%s, ff=%s)\n",
           w->n_bound, w->n_expected, st_dtype_name(w->dtype),
           w->gated_ff ? "gated" : "relu");
    printf("  %-70s %s\n", "shared.weight", w->shared ? "OK" : "MISS");
    for (uint32_t L = 0; L < w->n_encoder; L++) {
        const cp_t5_enc_layer_w_t *lw = &w->encoder[L];
        int layer_ok = lw->self_attn.q && lw->self_attn.k && lw->self_attn.v &&
                       lw->self_attn.o && lw->ln_self && lw->ln_ff &&
                       lw->ff.wo &&
                       (w->gated_ff ? (lw->ff.wi_0 && lw->ff.wi_1)
                                    : (lw->ff.wi != NULL));
        if (L == 0) layer_ok = layer_ok && lw->self_attn.rel_bias;
        printf("  enc layer %-2u self[q k v o%s] ff ln : %s\n", L,
               L == 0 ? " rel" : "", layer_ok ? "OK" : "MISS");
    }
    printf("  %-70s %s\n", "encoder.final_layer_norm.weight",
           w->enc_final_ln ? "OK" : "MISS");
    for (uint32_t L = 0; L < w->n_decoder; L++) {
        const cp_t5_dec_layer_w_t *lw = &w->decoder[L];
        int layer_ok =
                lw->self_attn.q && lw->self_attn.k && lw->self_attn.v &&
                lw->self_attn.o && lw->ln_self && lw->cross_attn.q &&
                lw->cross_attn.k && lw->cross_attn.v && lw->cross_attn.o &&
                lw->ln_cross && lw->ln_ff && lw->ff.wo &&
                (w->gated_ff ? (lw->ff.wi_0 && lw->ff.wi_1)
                             : (lw->ff.wi != NULL));
        if (L == 0) layer_ok = layer_ok && lw->self_attn.rel_bias;
        printf("  dec layer %-2u self[q k v o%s] cross[q k v o] ff ln : %s\n",
               L, L == 0 ? " rel" : "", layer_ok ? "OK" : "MISS");
    }
    printf("  %-70s %s\n", "decoder.final_layer_norm.weight",
           w->dec_final_ln ? "OK" : "MISS");
}

void cp_t5_weights_free(cp_t5_weights_t *w) {
    if (!w) return;
    free(w->encoder);
    free(w->decoder);
    w->encoder = NULL;
    w->decoder = NULL;
    w->n_encoder = 0;
    w->n_decoder = 0;
}
