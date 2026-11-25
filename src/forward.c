/*
 * forward.c - decoder forward pass (see forward.h)
 *
 * Qwen2 layer (pre-norm, GQA, attention biases on q/k/v, tied head):
 *
 *   h  = embed(tok)
 *   per layer:
 *     xn   = rmsnorm(h, input_ln)
 *     q,k,v = matvec(xn) + bias          (q: H, k/v: kv_dim)
 *     q,k   = rope(q, k, pos)
 *     kv_cache.push(k, v); a = attention(q, cache, pos)
 *     h   += matvec(o_w, a)
 *     xn   = rmsnorm(h, post_ln)
 *     a    = swiglu(matvec(gate_w,xn), matvec(up_w,xn))
 *     h   += matvec(down_w, a)
 *   logits = matvec(lm_head == embed_table, rmsnorm(h, final_norm))
 *
 * Summation order inside every matvec follows cp_matvec (the GPU
 * parity baseline). RoPE uses the absolute position passed by the
 * caller: prefill i-th prompt token with pos=i, decode continues.
 */

#include "forward.h"
#include "arch.h"
#include "ops.h"
#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cp_forward {
    const cp_config_t  *cfg;
    const cp_weights_t *w;
    cp_kv_t            *kv;     /* [num_hidden_layers] */
    float *h;                  /* [H] residual stream */
    float *xn;                 /* [H] normed input for projections */
    float *q;                  /* [H] */
    float *k, *v;              /* [kv_dim] */
    float *attn;               /* [H] attention head outputs */
    float *proj;               /* [H] o_proj / down_proj result */
    float *gate, *up, *act;    /* [I] */
    float *logits;             /* [V] */
    size_t steps;              /* tokens run since init/reset */
    int    use_gpu;
};

/* y[i] += bf16 bias row (safetensors data pointer) */
static void add_bias(float *y, const uint8_t *bias, size_t n) {
    const uint16_t *b = (const uint16_t *)bias;
    for (size_t i = 0; i < n; i++) y[i] += cp_bf16_to_fp32(b[i]);
}

int cp_forward_init(cp_forward_t **out, const cp_config_t *cfg,
                    const cp_weights_t *w, int use_gpu) {
    if (!out || !cfg || !w || !w->layers || cfg->num_hidden_layers == 0) {
        return -1;
    }
    if (cfg->arch != CP_ARCH_DECODER_ONLY) {
        fprintf(stderr,
                "cp_forward: not implemented for strategy '%s' "
                "(model_type='%s')\n",
                cp_arch_name(cfg->arch), cfg->model_type);
        return -1;
    }
    *out = NULL;
    cp_forward_t *f = calloc(1, sizeof *f);
    if (!f) goto fail;
    f->cfg = cfg;
    f->w = w;

    if (use_gpu) {
        if (gpu_forward_init(cfg, w) == 0) {
            f->use_gpu = 1;
        } else {
            fprintf(stderr, "cp_forward: GPU unavailable, continuing on CPU\n");
        }
    }

    const size_t H = cfg->hidden_size;
    const size_t I = cfg->intermediate_size;
    const size_t V = cfg->vocab_size;
    const size_t kv_dim = (size_t)cfg->num_key_value_heads * cfg->head_dim;

    f->kv = calloc(cfg->num_hidden_layers, sizeof *f->kv);
    f->h  = malloc(H * sizeof *f->h);
    f->xn = malloc(H * sizeof *f->xn);
    f->q  = malloc(H * sizeof *f->q);
    f->k  = malloc(kv_dim * sizeof *f->k);
    f->v  = malloc(kv_dim * sizeof *f->v);
    f->attn = malloc(H * sizeof *f->attn);
    f->proj = malloc(H * sizeof *f->proj);
    f->gate = malloc(I * sizeof *f->gate);
    f->up   = malloc(I * sizeof *f->up);
    f->act  = malloc(I * sizeof *f->act);
    f->logits = malloc(V * sizeof *f->logits);
    if (!f->kv || !f->h || !f->xn || !f->q || !f->k || !f->v ||
        !f->attn || !f->proj || !f->gate || !f->up || !f->act ||
        !f->logits) {
        goto fail;
    }
    for (uint32_t L = 0; L < cfg->num_hidden_layers; L++) {
        if (cp_kv_reserve(&f->kv[L], kv_dim, 32) != 0) goto fail;
    }
    *out = f;
    return 0;

fail:
    fprintf(stderr, "cp_forward: out of memory\n");
    cp_forward_free(f);
    return -1;
}

int cp_forward_on_gpu(const cp_forward_t *f) {
    return f ? f->use_gpu : 0;
}

void cp_forward_free(cp_forward_t *f) {
    if (!f) return;
    if (f->use_gpu) gpu_forward_free();
    if (f->kv) {
        for (uint32_t L = 0; L < f->cfg->num_hidden_layers; L++) {
            cp_kv_free(&f->kv[L]);
        }
        free(f->kv);
    }
    free(f->h);
    free(f->xn);
    free(f->q);
    free(f->k);
    free(f->v);
    free(f->attn);
    free(f->proj);
    free(f->gate);
    free(f->up);
    free(f->act);
    free(f->logits);
    free(f);
}

size_t cp_forward_len(const cp_forward_t *f) {
    return f ? f->steps : 0;
}

void cp_forward_reset(cp_forward_t *f) {
    if (!f) return;
    f->steps = 0;
    if (f->use_gpu) {
        gpu_forward_reset();
        return;
    }
    const size_t kv_dim = (size_t)f->cfg->num_key_value_heads * f->cfg->head_dim;
    for (uint32_t L = 0; L < f->cfg->num_hidden_layers; L++) {
        cp_kv_free(&f->kv[L]);
        cp_kv_reserve(&f->kv[L], kv_dim, 32);
    }
}

const float *cp_forward_step(cp_forward_t *f, uint32_t tok, size_t pos) {
    if (!f) return NULL;
    f->steps++;
    if (f->use_gpu) return gpu_forward_step(tok, pos);
    const cp_config_t *c = f->cfg;
    const cp_weights_t *w = f->w;
    const size_t H = c->hidden_size;
    const size_t I = c->intermediate_size;
    const size_t V = c->vocab_size;
    const size_t nh = c->num_attention_heads;
    const size_t nkv = c->num_key_value_heads;
    const size_t hd = c->head_dim;
    const size_t kv_dim = nkv * hd;
    const float eps = (float)c->rms_norm_eps;
    const float theta = (float)c->rope_theta;

    cp_embed(f->h, (const uint16_t *)w->embed_tokens->data, H, tok);

    for (uint32_t L = 0; L < c->num_hidden_layers; L++) {
        const cp_layer_w_t *lw = &w->layers[L];

        /* ---- attention block ---- */
        cp_rmsnorm(f->xn, f->h, (const uint16_t *)lw->input_ln->data, H, eps);
        cp_matvec(f->q, (const uint16_t *)lw->q_w->data, f->xn, H, H);
        add_bias(f->q, lw->q_b->data, H);
        cp_matvec(f->k, (const uint16_t *)lw->k_w->data, f->xn, kv_dim, H);
        add_bias(f->k, lw->k_b->data, kv_dim);
        cp_matvec(f->v, (const uint16_t *)lw->v_w->data, f->xn, kv_dim, H);
        add_bias(f->v, lw->v_b->data, kv_dim);

        cp_rope(f->q, nh, hd, pos, theta);
        cp_rope(f->k, nkv, hd, pos, theta);

        cp_kv_push(&f->kv[L], f->k, f->v);
        cp_attention(f->attn, f->q, f->kv[L].k, f->kv[L].v,
                     nh, nkv, hd, pos);
        cp_matvec(f->proj, (const uint16_t *)lw->o_w->data, f->attn, H, H);
        for (size_t i = 0; i < H; i++) f->h[i] += f->proj[i];

        /* ---- MLP block ---- */
        cp_rmsnorm(f->xn, f->h, (const uint16_t *)lw->post_ln->data, H, eps);
        cp_matvec(f->gate, (const uint16_t *)lw->gate_w->data, f->xn, I, H);
        cp_matvec(f->up, (const uint16_t *)lw->up_w->data, f->xn, I, H);
        cp_swiglu(f->act, f->gate, f->up, I);
        cp_matvec(f->proj, (const uint16_t *)lw->down_w->data, f->act, H, I);
        for (size_t i = 0; i < H; i++) f->h[i] += f->proj[i];
    }

    cp_rmsnorm(f->xn, f->h, (const uint16_t *)w->final_norm->data, H, eps);
    cp_matvec(f->logits, (const uint16_t *)w->lm_head->data, f->xn, V, H);
    return f->logits;
}
