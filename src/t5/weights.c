/*
 * t5_weights.c - T5 weight binding (see t5_weights.h)
 */

#include "t5/weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_gated_ff(const char *feed_forward_proj) {
    return feed_forward_proj && strstr(feed_forward_proj, "gated") != NULL;
}

static st_dtype_t resolve_dtype(const st_file_t *f) {
    const st_tensor_t *shared = st_find(f, "shared.weight");
    if (!shared) return ST_DTYPE_COUNT;
    if (shared->dtype == ST_DTYPE_F32 || shared->dtype == ST_DTYPE_BF16 ||
        shared->dtype == ST_DTYPE_F16)
        return shared->dtype;
    return ST_DTYPE_COUNT;
}

static const st_tensor_t *expect(
        const st_file_t *f, const char *name,
        st_dtype_t dtype, size_t ndim, const size_t *shape,
        size_t *n_bound, size_t *n_expected) {
    (*n_expected)++;
    const st_tensor_t *t = st_find(f, name);
    if (!t) {
        fprintf(stderr, "cp_t5_weights: [MISS] %-70s not in file\n", name);
        return NULL;
    }
    if (t->dtype != dtype) {
        fprintf(stderr, "cp_t5_weights: [DTYPE] %-70s is %s, want %s\n",
                name, st_dtype_name(t->dtype), st_dtype_name(dtype));
        return NULL;
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "cp_t5_weights: [RANK] %-70s ndim=%zu, want %zu\n",
                name, t->ndim, ndim);
        return NULL;
    }
    for (size_t d = 0; d < ndim; d++) {
        if (t->shape[d] != shape[d]) {
            fprintf(stderr,
                    "cp_t5_weights: [SHAPE] %-70s dim%zu=%zu, want %zu\n",
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

static int bind_attn(const st_file_t *f, st_dtype_t dt,
                     const char *prefix, /* e.g. encoder.block.0.layer.0.SelfAttention */
                     uint32_t inner, uint32_t d_model,
                     int want_rel_bias, uint32_t n_buckets, uint32_t n_heads,
                     cp_t5_attn_w_t *out,
                     size_t *n_bound, size_t *n_expected) {
    char name[192];
    int ok = 1;

    snprintf(name, sizeof(name), "%s.q.weight", prefix);
    out->q = EXPECT2(f, name, dt, inner, d_model, n_bound, n_expected);
    ok &= out->q != NULL;

    snprintf(name, sizeof(name), "%s.k.weight", prefix);
    out->k = EXPECT2(f, name, dt, inner, d_model, n_bound, n_expected);
    ok &= out->k != NULL;

    snprintf(name, sizeof(name), "%s.v.weight", prefix);
    out->v = EXPECT2(f, name, dt, inner, d_model, n_bound, n_expected);
    ok &= out->v != NULL;

    snprintf(name, sizeof(name), "%s.o.weight", prefix);
    out->o = EXPECT2(f, name, dt, d_model, inner, n_bound, n_expected);
    ok &= out->o != NULL;

    out->rel_bias = NULL;
    if (want_rel_bias) {
        snprintf(name, sizeof(name), "%s.relative_attention_bias.weight",
                 prefix);
        out->rel_bias = EXPECT2(f, name, dt, n_buckets, n_heads,
                                n_bound, n_expected);
        ok &= out->rel_bias != NULL;
    }
    return ok ? 0 : -1;
}

static int bind_ff(const st_file_t *f, st_dtype_t dt,
                   const char *prefix, /* ...DenseReluDense */
                   uint32_t d_ff, uint32_t d_model, int gated,
                   cp_t5_ff_w_t *out,
                   size_t *n_bound, size_t *n_expected) {
    char name[192];
    int ok = 1;

    out->gated = gated;
    out->wi = out->wi_0 = out->wi_1 = NULL;

    if (gated) {
        snprintf(name, sizeof(name), "%s.wi_0.weight", prefix);
        out->wi_0 = EXPECT2(f, name, dt, d_ff, d_model, n_bound, n_expected);
        ok &= out->wi_0 != NULL;
        snprintf(name, sizeof(name), "%s.wi_1.weight", prefix);
        out->wi_1 = EXPECT2(f, name, dt, d_ff, d_model, n_bound, n_expected);
        ok &= out->wi_1 != NULL;
    } else {
        snprintf(name, sizeof(name), "%s.wi.weight", prefix);
        out->wi = EXPECT2(f, name, dt, d_ff, d_model, n_bound, n_expected);
        ok &= out->wi != NULL;
    }

    snprintf(name, sizeof(name), "%s.wo.weight", prefix);
    out->wo = EXPECT2(f, name, dt, d_model, d_ff, n_bound, n_expected);
    ok &= out->wo != NULL;
    return ok ? 0 : -1;
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

    st_dtype_t dt = resolve_dtype(f);
    if (dt == ST_DTYPE_COUNT) {
        fprintf(stderr,
                "cp_t5_weights: shared.weight missing or unsupported dtype "
                "(need F32/BF16/F16)\n");
        return -1;
    }
    out->dtype = dt;
    out->gated_ff = is_gated_ff(cfg->feed_forward_proj);

    const uint32_t D = cfg->d_model;
    const uint32_t I = cfg->d_ff;
    const uint32_t V = cfg->vocab_size;
    const uint32_t inner = cfg->inner_dim;
    const uint32_t H = cfg->num_heads;
    const uint32_t buckets = cfg->relative_attention_num_buckets;
    const uint32_t n_enc = cfg->num_layers;
    const uint32_t n_dec = cfg->num_decoder_layers;

    size_t n_bound = 0, n_expected = 0;
    int ok = 1;
    char name[192];
    char prefix[160];

    out->shared = EXPECT2(f, "shared.weight", dt, V, D, &n_bound, &n_expected);
    ok &= out->shared != NULL;

    out->encoder = calloc(n_enc, sizeof(cp_t5_enc_layer_w_t));
    out->decoder = calloc(n_dec, sizeof(cp_t5_dec_layer_w_t));
    if (!out->encoder || !out->decoder) {
        fprintf(stderr, "cp_t5_weights: out of memory\n");
        cp_t5_weights_free(out);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    out->n_encoder = n_enc;
    out->n_decoder = n_dec;

    for (uint32_t L = 0; L < n_enc; L++) {
        cp_t5_enc_layer_w_t *lw = &out->encoder[L];
        snprintf(prefix, sizeof(prefix),
                 "encoder.block.%u.layer.0.SelfAttention", L);
        if (bind_attn(f, dt, prefix, inner, D, L == 0, buckets, H,
                      &lw->self_attn, &n_bound, &n_expected) != 0)
            ok = 0;

        snprintf(name, sizeof(name),
                 "encoder.block.%u.layer.0.layer_norm.weight", L);
        lw->ln_self = EXPECT1(f, name, dt, D, &n_bound, &n_expected);
        ok &= lw->ln_self != NULL;

        snprintf(prefix, sizeof(prefix),
                 "encoder.block.%u.layer.1.DenseReluDense", L);
        if (bind_ff(f, dt, prefix, I, D, out->gated_ff, &lw->ff,
                    &n_bound, &n_expected) != 0)
            ok = 0;

        snprintf(name, sizeof(name),
                 "encoder.block.%u.layer.1.layer_norm.weight", L);
        lw->ln_ff = EXPECT1(f, name, dt, D, &n_bound, &n_expected);
        ok &= lw->ln_ff != NULL;
    }

    /* Layers 1..N reuse block-0 relative bias (HF behaviour). */
    for (uint32_t L = 1; L < n_enc; L++)
        out->encoder[L].self_attn.rel_bias =
                out->encoder[0].self_attn.rel_bias;

    out->enc_final_ln = EXPECT1(f, "encoder.final_layer_norm.weight", dt, D,
                                &n_bound, &n_expected);
    ok &= out->enc_final_ln != NULL;

    for (uint32_t L = 0; L < n_dec; L++) {
        cp_t5_dec_layer_w_t *lw = &out->decoder[L];
        snprintf(prefix, sizeof(prefix),
                 "decoder.block.%u.layer.0.SelfAttention", L);
        if (bind_attn(f, dt, prefix, inner, D, L == 0, buckets, H,
                      &lw->self_attn, &n_bound, &n_expected) != 0)
            ok = 0;

        snprintf(name, sizeof(name),
                 "decoder.block.%u.layer.0.layer_norm.weight", L);
        lw->ln_self = EXPECT1(f, name, dt, D, &n_bound, &n_expected);
        ok &= lw->ln_self != NULL;

        /* Cross-attn: no relative bias required (HF ignores orphan tensor). */
        snprintf(prefix, sizeof(prefix),
                 "decoder.block.%u.layer.1.EncDecAttention", L);
        if (bind_attn(f, dt, prefix, inner, D, 0, buckets, H,
                      &lw->cross_attn, &n_bound, &n_expected) != 0)
            ok = 0;

        snprintf(name, sizeof(name),
                 "decoder.block.%u.layer.1.layer_norm.weight", L);
        lw->ln_cross = EXPECT1(f, name, dt, D, &n_bound, &n_expected);
        ok &= lw->ln_cross != NULL;

        snprintf(prefix, sizeof(prefix),
                 "decoder.block.%u.layer.2.DenseReluDense", L);
        if (bind_ff(f, dt, prefix, I, D, out->gated_ff, &lw->ff,
                    &n_bound, &n_expected) != 0)
            ok = 0;

        snprintf(name, sizeof(name),
                 "decoder.block.%u.layer.2.layer_norm.weight", L);
        lw->ln_ff = EXPECT1(f, name, dt, D, &n_bound, &n_expected);
        ok &= lw->ln_ff != NULL;
    }

    for (uint32_t L = 1; L < n_dec; L++)
        out->decoder[L].self_attn.rel_bias =
                out->decoder[0].self_attn.rel_bias;

    out->dec_final_ln = EXPECT1(f, "decoder.final_layer_norm.weight", dt, D,
                                &n_bound, &n_expected);
    ok &= out->dec_final_ln != NULL;

    /* Informational: Google checkpoints sometimes ship unused cross rel-bias. */
    if (st_find(f, "decoder.block.0.layer.1.EncDecAttention."
                   "relative_attention_bias.weight")) {
        fprintf(stderr,
                "cp_t5_weights: [NOTE] EncDecAttention.relative_attention_bias "
                "present but unused (HF ignores it)\n");
    }

    out->n_bound = n_bound;
    out->n_expected = n_expected;
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
