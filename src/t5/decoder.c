/*
 * t5/decoder.c - T5 decoder step (see t5/decoder.h)
 *
 * HF T5Block decoder (eval, no dropout), incremental:
 *   h = embed(tok)
 *   for each layer:
 *     h += self_attn(LN(h), cache)     # causal via cache length + rel bias
 *     h += cross_attn(LN(h), memory)   # K/V precomputed
 *     h += DenseReluDense(LN(h))
 *   h = final_LN(h)
 *   logits = shared(h * d_model^{-0.5})   # tied embeddings
 */

#include "t5/decoder.h"
#include "t5/ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *kc; /* [max_tgt, inner] */
    float *vc;
    float *xk; /* [max_src, inner] cross K */
    float *xv;
} dec_layer_cache_t;

struct cp_t5_decoder {
    const cp_t5_config_t  *cfg;
    const cp_t5_weights_t *w;
    size_t max_src, max_tgt, src_len, steps;

    float *mem; /* [max_src, d_model] copy of encoder out */
    dec_layer_cache_t *lay;

    /* single-token scratch */
    float *h, *xn, *q, *ctx, *proj, *ff_h;
    float *bias;   /* [n_heads * max_tgt] */
    float *logits; /* [V] */

    size_t inner;
};

static const float *t_f32(const st_tensor_t *t) {
    return t ? (const float *)t->data : NULL;
}

int cp_t5_decoder_init(cp_t5_decoder_t **out, const cp_t5_config_t *cfg,
                       const cp_t5_weights_t *w, size_t max_src,
                       size_t max_tgt) {
    if (!out || !cfg || !w || !w->decoder || !w->shared || !w->dec_final_ln)
        return -1;
    if (cfg->arch != CP_ARCH_ENCODER_DECODER) {
        fprintf(stderr, "cp_t5_decoder: expected encoder-decoder\n");
        return -1;
    }
    if (w->dtype != ST_DTYPE_F32) {
        fprintf(stderr,
                "cp_t5_decoder: CPU path requires F32 weights (got %s)\n",
                st_dtype_name(w->dtype));
        return -1;
    }
    if (w->gated_ff) {
        fprintf(stderr, "cp_t5_decoder: gated-gelu FFN not implemented\n");
        return -1;
    }

    if (max_src == 0) max_src = 512;
    if (max_tgt == 0) max_tgt = 512;

    cp_t5_decoder_t *d = calloc(1, sizeof *d);
    if (!d) return -1;
    d->cfg = cfg;
    d->w = w;
    d->max_src = max_src;
    d->max_tgt = max_tgt;
    d->inner = (size_t)cfg->inner_dim;

    const size_t D = cfg->d_model;
    const size_t I = d->inner;
    const size_t F = cfg->d_ff;
    const size_t V = cfg->vocab_size;
    const size_t nh = cfg->num_heads;
    const uint32_t nL = w->n_decoder;

    d->mem = malloc(max_src * D * sizeof *d->mem);
    d->lay = calloc(nL, sizeof *d->lay);
    d->h = malloc(D * sizeof *d->h);
    d->xn = malloc(D * sizeof *d->xn);
    d->q = malloc(I * sizeof *d->q);
    d->ctx = malloc(I * sizeof *d->ctx);
    d->proj = malloc(D * sizeof *d->proj);
    d->ff_h = malloc(F * sizeof *d->ff_h);
    d->bias = malloc(nh * max_tgt * sizeof *d->bias);
    d->logits = malloc(V * sizeof *d->logits);
    if (!d->mem || !d->lay || !d->h || !d->xn || !d->q || !d->ctx || !d->proj ||
        !d->ff_h || !d->bias || !d->logits) {
        cp_t5_decoder_free(d);
        return -1;
    }

    for (uint32_t L = 0; L < nL; L++) {
        d->lay[L].kc = malloc(max_tgt * I * sizeof(float));
        d->lay[L].vc = malloc(max_tgt * I * sizeof(float));
        d->lay[L].xk = malloc(max_src * I * sizeof(float));
        d->lay[L].xv = malloc(max_src * I * sizeof(float));
        if (!d->lay[L].kc || !d->lay[L].vc || !d->lay[L].xk ||
            !d->lay[L].xv) {
            cp_t5_decoder_free(d);
            return -1;
        }
    }

    *out = d;
    return 0;
}

void cp_t5_decoder_free(cp_t5_decoder_t *d) {
    if (!d) return;
    if (d->lay) {
        for (uint32_t L = 0; L < d->w->n_decoder; L++) {
            free(d->lay[L].kc);
            free(d->lay[L].vc);
            free(d->lay[L].xk);
            free(d->lay[L].xv);
        }
        free(d->lay);
    }
    free(d->mem);
    free(d->h);
    free(d->xn);
    free(d->q);
    free(d->ctx);
    free(d->proj);
    free(d->ff_h);
    free(d->bias);
    free(d->logits);
    free(d);
}

int cp_t5_decoder_set_memory(cp_t5_decoder_t *d, const float *mem,
                             size_t src_len) {
    if (!d || !mem || src_len == 0 || src_len > d->max_src) return -1;

    const size_t D = d->cfg->d_model;
    const size_t I = d->inner;
    memcpy(d->mem, mem, src_len * D * sizeof(float));
    d->src_len = src_len;

    for (uint32_t L = 0; L < d->w->n_decoder; L++) {
        const cp_t5_attn_w_t *ca = &d->w->decoder[L].cross_attn;
        cp_t5_linear_seq(d->lay[L].xk, t_f32(ca->k), d->mem, src_len, I, D);
        cp_t5_linear_seq(d->lay[L].xv, t_f32(ca->v), d->mem, src_len, I, D);
    }

    cp_t5_decoder_reset(d);
    return 0;
}

void cp_t5_decoder_reset(cp_t5_decoder_t *d) {
    if (!d) return;
    d->steps = 0;
}

const float *cp_t5_decoder_step(cp_t5_decoder_t *d, uint32_t tok, size_t pos) {
    if (!d || d->src_len == 0 || pos >= d->max_tgt) return NULL;
    /* steps must be contiguous from 0 */
    if (pos != d->steps) {
        fprintf(stderr, "cp_t5_decoder: expected pos=%zu, got %zu\n", d->steps,
                pos);
        return NULL;
    }

    const cp_t5_config_t *cfg = d->cfg;
    const cp_t5_weights_t *w = d->w;
    const size_t D = cfg->d_model;
    const size_t I = d->inner;
    const size_t F = cfg->d_ff;
    const size_t V = cfg->vocab_size;
    const size_t nh = cfg->num_heads;
    const size_t dkv = cfg->d_kv;
    const float eps = (float)cfg->layer_norm_epsilon;
    const size_t buckets = cfg->relative_attention_num_buckets;
    const size_t max_dist = CP_T5_REL_MAX_DISTANCE_DEFAULT;
    const size_t klen = pos + 1;

    cp_t5_embed(d->h, t_f32(w->shared), D, tok);

    for (uint32_t L = 0; L < w->n_decoder; L++) {
        const cp_t5_dec_layer_w_t *lw = &w->decoder[L];
        dec_layer_cache_t *lc = &d->lay[L];

        /* ---- self-attn ---- */
        cp_t5_layernorm(d->xn, d->h, t_f32(lw->ln_self), D, eps);
        cp_t5_matvec(d->q, t_f32(lw->self_attn.q), d->xn, I, D);
        cp_t5_matvec(lc->kc + pos * I, t_f32(lw->self_attn.k), d->xn, I, D);
        cp_t5_matvec(lc->vc + pos * I, t_f32(lw->self_attn.v), d->xn, I, D);

        cp_t5_position_bias_query(d->bias, t_f32(lw->self_attn.rel_bias), pos,
                                  klen, nh, buckets, max_dist,
                                  /*bidirectional=*/0);
        /* cache already holds only <= pos; causal mask unused */
        cp_t5_mha(d->ctx, d->q, lc->kc, lc->vc, d->bias, 1, klen, nh, dkv, 0);
        cp_t5_matvec(d->proj, t_f32(lw->self_attn.o), d->ctx, D, I);
        for (size_t i = 0; i < D; i++) d->h[i] += d->proj[i];

        /* ---- cross-attn ---- */
        cp_t5_layernorm(d->xn, d->h, t_f32(lw->ln_cross), D, eps);
        cp_t5_matvec(d->q, t_f32(lw->cross_attn.q), d->xn, I, D);
        cp_t5_mha(d->ctx, d->q, lc->xk, lc->xv, NULL, 1, d->src_len, nh, dkv,
                  0);
        cp_t5_matvec(d->proj, t_f32(lw->cross_attn.o), d->ctx, D, I);
        for (size_t i = 0; i < D; i++) d->h[i] += d->proj[i];

        /* ---- FFN ---- */
        cp_t5_layernorm(d->xn, d->h, t_f32(lw->ln_ff), D, eps);
        cp_t5_matvec(d->ff_h, t_f32(lw->ff.wi), d->xn, F, D);
        for (size_t j = 0; j < F; j++)
            if (d->ff_h[j] < 0.0f) d->ff_h[j] = 0.0f;
        cp_t5_matvec(d->proj, t_f32(lw->ff.wo), d->ff_h, D, F);
        for (size_t i = 0; i < D; i++) d->h[i] += d->proj[i];
    }

    cp_t5_layernorm(d->xn, d->h, t_f32(w->dec_final_ln), D, eps);

    /* HF tie_word_embeddings: scale by d_model^{-0.5} before lm_head */
    {
        float scale = 1.0f / sqrtf((float)D);
        for (size_t i = 0; i < D; i++) d->xn[i] *= scale;
    }
    cp_t5_matvec(d->logits, t_f32(w->shared), d->xn, V, D);

    d->steps = pos + 1;
    return d->logits;
}
