/*
 * t5/encoder.c - T5 encoder forward (see t5/encoder.h)
 *
 * Matches HF T5Stack encoder (eval, no dropout):
 *   h = embed(src)
 *   for each block:
 *     h += self_attn(LN(h))          # bidirectional + relative bias
 *     h += DenseReluDense(LN(h))
 *   h = final_LN(h)
 */

#include "t5/encoder.h"
#include "t5/ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cp_t5_encoder {
    const cp_t5_config_t  *cfg;
    const cp_t5_weights_t *w;
    size_t max_seq;
    size_t seq_len;
    float *mem;  /* [max_seq, d_model] residual / output */
    float *xn;   /* [max_seq, d_model] normed */
    float *attn; /* [max_seq, d_model] self-attn / ff out */
    float *ff_h; /* [d_ff] single-token FFN scratch */
};

static const float *t_f32(const st_tensor_t *t) {
    return t ? (const float *)t->data : NULL;
}

static void add_seq(float *h, const float *a, size_t n) {
    for (size_t i = 0; i < n; i++) h[i] += a[i];
}

static void layernorm_seq(float *Y, const float *X, const float *w,
                          size_t seq, size_t d, float eps) {
    for (size_t i = 0; i < seq; i++)
        cp_t5_layernorm(Y + i * d, X + i * d, w, d, eps);
}

static void dense_relu_dense_seq(float *Y, const float *X, const float *wi,
                                 const float *wo, float *scratch, size_t seq,
                                 size_t d_model, size_t d_ff) {
    for (size_t i = 0; i < seq; i++) {
        cp_t5_matvec(scratch, wi, X + i * d_model, d_ff, d_model);
        for (size_t j = 0; j < d_ff; j++)
            if (scratch[j] < 0.0f) scratch[j] = 0.0f;
        cp_t5_matvec(Y + i * d_model, wo, scratch, d_model, d_ff);
    }
}

int cp_t5_encoder_init(cp_t5_encoder_t **out, const cp_t5_config_t *cfg,
                       const cp_t5_weights_t *w, size_t max_seq) {
    if (!out || !cfg || !w || !w->encoder || !w->shared || !w->enc_final_ln)
        return -1;
    if (cfg->arch != CP_ARCH_ENCODER_DECODER) {
        fprintf(stderr, "cp_t5_encoder: expected encoder-decoder\n");
        return -1;
    }
    if (w->dtype != ST_DTYPE_F32) {
        fprintf(stderr,
                "cp_t5_encoder: CPU path requires F32 weights (got %s)\n",
                st_dtype_name(w->dtype));
        return -1;
    }
    if (w->gated_ff) {
        fprintf(stderr, "cp_t5_encoder: gated-gelu FFN not implemented\n");
        return -1;
    }

    if (max_seq == 0) max_seq = 512;

    cp_t5_encoder_t *e = calloc(1, sizeof *e);
    if (!e) return -1;
    e->cfg = cfg;
    e->w = w;
    e->max_seq = max_seq;

    const size_t D = cfg->d_model;
    e->mem = malloc(max_seq * D * sizeof *e->mem);
    e->xn = malloc(max_seq * D * sizeof *e->xn);
    e->attn = malloc(max_seq * D * sizeof *e->attn);
    e->ff_h = malloc((size_t)cfg->d_ff * sizeof *e->ff_h);
    if (!e->mem || !e->xn || !e->attn || !e->ff_h) {
        cp_t5_encoder_free(e);
        return -1;
    }

    *out = e;
    return 0;
}

void cp_t5_encoder_free(cp_t5_encoder_t *e) {
    if (!e) return;
    free(e->mem);
    free(e->xn);
    free(e->attn);
    free(e->ff_h);
    free(e);
}

const float *cp_t5_encoder_run(cp_t5_encoder_t *e, const uint32_t *src_ids,
                               size_t n) {
    if (!e || !src_ids || n == 0 || n > e->max_seq) return NULL;

    const cp_t5_config_t *cfg = e->cfg;
    const cp_t5_weights_t *w = e->w;
    const size_t D = cfg->d_model;
    const size_t F = cfg->d_ff;
    const float eps = (float)cfg->layer_norm_epsilon;
    const size_t buckets = cfg->relative_attention_num_buckets;
    const size_t max_dist = CP_T5_REL_MAX_DISTANCE_DEFAULT;

    cp_t5_embed_seq(e->mem, t_f32(w->shared), D, src_ids, n);

    for (uint32_t L = 0; L < w->n_encoder; L++) {
        const cp_t5_enc_layer_w_t *lw = &w->encoder[L];
        const cp_t5_attn_w_t *sa = &lw->self_attn;

        layernorm_seq(e->xn, e->mem, t_f32(lw->ln_self), n, D, eps);
        cp_t5_self_attention(e->attn, e->xn, t_f32(sa->q), t_f32(sa->k),
                             t_f32(sa->v), t_f32(sa->o), t_f32(sa->rel_bias),
                             n, D, cfg->num_heads, cfg->d_kv, buckets,
                             max_dist, /*causal=*/0, /*has_rel_bias=*/1);
        add_seq(e->mem, e->attn, n * D);

        layernorm_seq(e->xn, e->mem, t_f32(lw->ln_ff), n, D, eps);
        dense_relu_dense_seq(e->attn, e->xn, t_f32(lw->ff.wi),
                             t_f32(lw->ff.wo), e->ff_h, n, D, F);
        add_seq(e->mem, e->attn, n * D);
    }

    layernorm_seq(e->mem, e->mem, t_f32(w->enc_final_ln), n, D, eps);
    e->seq_len = n;
    return e->mem;
}

size_t cp_t5_encoder_seq_len(const cp_t5_encoder_t *e) {
    return e ? e->seq_len : 0;
}
