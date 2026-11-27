/*
 * t5/gpu.c - T5 CUDA kernels + Enc–Dec forward (see t5/gpu.h)
 *
 * Driver / NVRTC / context live in gpu_runtime.c. This file owns
 * T5-specific FP32 kernels and encoder / decoder-step orchestration.
 */

#include "t5/gpu.h"
#include "gpu_runtime.h"
#include "t5/ops.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- module state ---------------- */

static cp_gpu_module_t s_mod;
static cp_gpu_fn_t s_matvec, s_linear_seq, s_embed, s_ln, s_relu, s_add;
static cp_gpu_fn_t s_ascore, s_asoft, s_av;
static int s_ready;

void t5_gpu_forward_free(void);

/* ---------------- embedded kernels (FP32, match t5/ops.c) ---------------- */

static const char *kKernelSrc =
    "extern \"C\" __global__ void matvec_f32(float *y, const float *W,\n"
    "                                       const float *x, int rows, int cols) {\n"
    "    int r = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (r >= rows) return;\n"
    "    float acc = 0.0f;\n"
    "    const float *row = W + (size_t)r * (size_t)cols;\n"
    "    for (int c = 0; c < cols; c++) acc += row[c] * x[c];\n"
    "    y[r] = acc;\n"
    "}\n"
    "extern \"C\" __global__ void linear_seq_f32(float *Y, const float *W,\n"
    "                                          const float *X, int seq,\n"
    "                                          int rows, int cols) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (idx >= seq * rows) return;\n"
    "    int i = idx / rows;\n"
    "    int r = idx - i * rows;\n"
    "    float acc = 0.0f;\n"
    "    const float *xrow = X + (size_t)i * cols;\n"
    "    const float *wrow = W + (size_t)r * cols;\n"
    "    for (int c = 0; c < cols; c++) acc += wrow[c] * xrow[c];\n"
    "    Y[idx] = acc;\n"
    "}\n"
    "extern \"C\" __global__ void embed_row_f32(float *h, const float *E,\n"
    "                                         int tok, int d) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= d) return;\n"
    "    h[i] = E[(size_t)tok * d + i];\n"
    "}\n"
    "extern \"C\" __global__ void layernorm_f32(float *y, const float *x,\n"
    "                                         const float *w, int n, float eps) {\n"
    "    float ss = 0.0f;\n"
    "    for (int i = 0; i < n; i++) ss += x[i] * x[i];\n"
    "    float inv = 1.0f / sqrtf(ss / (float)n + eps);\n"
    "    for (int i = 0; i < n; i++) y[i] = x[i] * inv * w[i];\n"
    "}\n"
    "extern \"C\" __global__ void relu_inplace(float *x, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    if (x[i] < 0.0f) x[i] = 0.0f;\n"
    "}\n"
    "extern \"C\" __global__ void add_inplace(float *h, const float *a, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    h[i] += a[i];\n"
    "}\n"
    /* scores unscaled (T5); optional bias [n_heads, seq_q, seq_k]; causal mask */
    "extern \"C\" __global__ void attn_scores_t5(float *s, const float *q,\n"
    "                                           const float *k, const float *bias,\n"
    "                                           int n_heads, int d_kv,\n"
    "                                           int seq_q, int seq_k, int causal) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_heads * seq_q * seq_k;\n"
    "    if (idx >= total) return;\n"
    "    int tmp = idx;\n"
    "    int ki = tmp % seq_k; tmp /= seq_k;\n"
    "    int qi = tmp % seq_q; tmp /= seq_q;\n"
    "    int h = tmp;\n"
    "    if (causal && ki > qi) { s[idx] = -1e9f; return; }\n"
    "    const float *qh = q + ((size_t)qi * n_heads + h) * d_kv;\n"
    "    const float *kh = k + ((size_t)ki * n_heads + h) * d_kv;\n"
    "    float acc = 0.0f;\n"
    "    for (int d = 0; d < d_kv; d++) acc += qh[d] * kh[d];\n"
    "    if (bias) acc += bias[((size_t)h * seq_q + qi) * seq_k + ki];\n"
    "    s[idx] = acc;\n"
    "}\n"
    "extern \"C\" __global__ void attn_softmax_t5(float *s, int n_heads,\n"
    "                                            int seq_q, int seq_k) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_heads * seq_q;\n"
    "    if (idx >= total) return;\n"
    "    int h = idx / seq_q;\n"
    "    int qi = idx - h * seq_q;\n"
    "    float *row = s + ((size_t)h * seq_q + qi) * seq_k;\n"
    "    float m = row[0];\n"
    "    for (int j = 1; j < seq_k; j++) if (row[j] > m) m = row[j];\n"
    "    float sum = 0.0f;\n"
    "    for (int j = 0; j < seq_k; j++) {\n"
    "        row[j] = expf(row[j] - m);\n"
    "        sum += row[j];\n"
    "    }\n"
    "    float inv = 1.0f / sum;\n"
    "    for (int j = 0; j < seq_k; j++) row[j] *= inv;\n"
    "}\n"
    "extern \"C\" __global__ void attn_av_t5(float *out, const float *s,\n"
    "                                       const float *v, int n_heads,\n"
    "                                       int d_kv, int seq_q, int seq_k) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (idx >= n_heads * seq_q * d_kv) return;\n"
    "    int tmp = idx;\n"
    "    int d = tmp % d_kv; tmp /= d_kv;\n"
    "    int qi = tmp % seq_q; tmp /= seq_q;\n"
    "    int h = tmp;\n"
    "    const float *sh = s + ((size_t)h * seq_q + qi) * seq_k;\n"
    "    float acc = 0.0f;\n"
    "    for (int ki = 0; ki < seq_k; ki++)\n"
    "        acc += sh[ki] * v[((size_t)ki * n_heads + h) * d_kv + d];\n"
    "    out[((size_t)qi * n_heads + h) * d_kv + d] = acc;\n"
    "}\n";

static int load_t5_kernels(void) {
    if (cp_gpu_rt_compile(kKernelSrc, "t5_kernels.cu", &s_mod) != 0)
        return -1;

    static const struct {
        const char *name;
        cp_gpu_fn_t *fn;
    } k_syms[] = {
        {"matvec_f32", &s_matvec},
        {"linear_seq_f32", &s_linear_seq},
        {"embed_row_f32", &s_embed},
        {"layernorm_f32", &s_ln},
        {"relu_inplace", &s_relu},
        {"add_inplace", &s_add},
        {"attn_scores_t5", &s_ascore},
        {"attn_softmax_t5", &s_asoft},
        {"attn_av_t5", &s_av},
    };
    for (size_t i = 0; i < sizeof k_syms / sizeof k_syms[0]; i++) {
        if (cp_gpu_rt_get_function(s_mod, k_syms[i].name, k_syms[i].fn) != 0)
            return -1;
    }
    return 0;
}

/* ---------------- public bring-up API ---------------- */

int t5_gpu_init(void) {
    if (s_ready) return 0;
    if (cp_gpu_rt_init() != 0) return -1;
    if (load_t5_kernels() != 0) {
        t5_gpu_shutdown();
        return -1;
    }
    s_ready = 1;
    return 0;
}

void t5_gpu_shutdown(void) {
    t5_gpu_forward_free();
    if (s_mod) {
        cp_gpu_rt_module_unload(s_mod);
        s_mod = NULL;
    }
    s_matvec = s_linear_seq = s_embed = s_ln = s_relu = s_add = NULL;
    s_ascore = s_asoft = s_av = NULL;
    s_ready = 0;
    cp_gpu_rt_shutdown();
}

int t5_gpu_ready(void) {
    return s_ready && cp_gpu_rt_ready();
}

int t5_gpu_matvec(float *y, const float *W, const float *x,
                  size_t rows, size_t cols) {
    if (!t5_gpu_ready()) {
        fprintf(stderr, "t5_gpu: not initialized\n");
        return -1;
    }
    if (rows == 0 || cols == 0 || rows > INT_MAX || cols > INT_MAX)
        return -1;

    cp_dptr_t dW = 0, dx = 0, dy = 0;
    int rc = -1;
    if (cp_gpu_rt_alloc(&dW, rows * cols * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&dx, cols * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&dy, rows * sizeof(float)) != 0)
        goto out;
    if (cp_gpu_rt_upload(dW, W, rows * cols * sizeof(float)) != 0 ||
        cp_gpu_rt_upload(dx, x, cols * sizeof(float)) != 0)
        goto out;

    int r = (int)rows, c = (int)cols;
    void *args[] = {&dy, &dW, &dx, &r, &c};
    unsigned grid = (unsigned)((rows + 255) / 256);
    if (cp_gpu_rt_launch(s_matvec, grid, 256, args) != 0) goto out;
    if (cp_gpu_rt_sync() != 0) goto out;
    if (cp_gpu_rt_download(y, dy, rows * sizeof(float)) != 0) goto out;
    rc = 0;
out:
    cp_gpu_rt_free(dW);
    cp_gpu_rt_free(dx);
    cp_gpu_rt_free(dy);
    return rc;
}

/* ==================================================================== */
/* resident forward                                                     */
/* ==================================================================== */

typedef struct {
    cp_dptr_t q, k, v, o, rel_bias; /* rel_bias may be 0 */
} g_attn_t;

typedef struct {
    g_attn_t self;
    cp_dptr_t ln_self;
    cp_dptr_t wi, wo;
    cp_dptr_t ln_ff;
} g_enc_t;

typedef struct {
    g_attn_t self;
    cp_dptr_t ln_self;
    g_attn_t cross;
    cp_dptr_t ln_cross;
    cp_dptr_t wi, wo;
    cp_dptr_t ln_ff;
    /* precomputed cross K/V after encode: [src_len, inner] */
    cp_dptr_t xk, xv;
} g_dec_t;

typedef struct {
    cp_dptr_t shared;
    g_enc_t *enc;
    cp_dptr_t enc_final_ln;
    g_dec_t *dec;
    cp_dptr_t dec_final_ln;

    /* encoder workspace */
    cp_dptr_t mem;          /* [max_src, d_model] — also residual stream */
    cp_dptr_t xn;           /* [max_src, d_model] */
    cp_dptr_t qkv;          /* [max_src, inner] scratch for q or ctx */
    cp_dptr_t kbuf, vbuf;   /* [max_src, inner] */
    cp_dptr_t proj;         /* [max_src, d_model] attn/ff out */
    cp_dptr_t ff_h;         /* [max_src, d_ff] */
    cp_dptr_t score;        /* [n_heads * max_src * max_src] */
    cp_dptr_t bias;         /* encoder / decode bias scratch */

    /* decoder single-token workspace */
    cp_dptr_t dh, dxn, dq, dctx, dproj, dff;
    cp_dptr_t dscore;       /* [n_heads * max_tgt] self or [n_heads * max_src] cross */
    cp_dptr_t dbias;        /* [n_heads * 1 * max_tgt] */
    cp_dptr_t logits;
    float *hlogits;

    /* decoder self KV cache per layer: [kv_cap, inner] */
    cp_dptr_t *kc, *vc;
    size_t kv_cap;

    uint32_t n_enc, n_dec, n_heads, d_kv, d_model, d_ff, V, inner;
    uint32_t num_buckets;
    size_t max_src, max_tgt, src_len;
    float eps;
    int max_distance;
} gfwd_t;

static gfwd_t *g_fwd;

#define GRD(n) ((unsigned)(((size_t)(n) + 255) / 256))

static int kl(cp_gpu_fn_t f, unsigned grid, unsigned block, void **args) {
    return cp_gpu_rt_launch(f, grid, block, args);
}

static void dfree(cp_dptr_t *d) {
    cp_gpu_rt_free(*d);
    *d = 0;
}

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* Widen tensor to host FP32 then upload. Supports F32 and BF16. */
static int upload_as_f32(cp_dptr_t *d, const st_tensor_t *t) {
    if (!t || !t->data) return -1;
    size_t n;
    if (t->dtype == ST_DTYPE_F32) {
        return cp_gpu_rt_upload_new(d, t->data, t->nbytes);
    }
    if (t->dtype == ST_DTYPE_BF16) {
        n = t->nbytes / sizeof(uint16_t);
        float *tmp = malloc(n * sizeof *tmp);
        if (!tmp) return -1;
        const uint16_t *src = (const uint16_t *)t->data;
        for (size_t i = 0; i < n; i++) tmp[i] = bf16_to_f32(src[i]);
        int rc = cp_gpu_rt_upload_new(d, tmp, n * sizeof(float));
        free(tmp);
        return rc;
    }
    fprintf(stderr, "t5_gpu: unsupported weight dtype for '%s'\n",
            t->name ? t->name : "?");
    return -1;
}

static int upload_attn(g_attn_t *a, const cp_t5_attn_w_t *w, int with_rel) {
    if (upload_as_f32(&a->q, w->q) != 0 ||
        upload_as_f32(&a->k, w->k) != 0 ||
        upload_as_f32(&a->v, w->v) != 0 ||
        upload_as_f32(&a->o, w->o) != 0)
        return -1;
    if (with_rel && w->rel_bias) {
        if (upload_as_f32(&a->rel_bias, w->rel_bias) != 0) return -1;
    }
    return 0;
}

static void free_attn(g_attn_t *a) {
    dfree(&a->q); dfree(&a->k); dfree(&a->v); dfree(&a->o);
    dfree(&a->rel_bias);
}

static void gfwd_release(gfwd_t *g) {
    if (!g) return;
    dfree(&g->shared);
    dfree(&g->enc_final_ln);
    dfree(&g->dec_final_ln);
    if (g->enc) {
        for (uint32_t L = 0; L < g->n_enc; L++) {
            free_attn(&g->enc[L].self);
            dfree(&g->enc[L].ln_self);
            dfree(&g->enc[L].wi); dfree(&g->enc[L].wo);
            dfree(&g->enc[L].ln_ff);
        }
        free(g->enc);
    }
    if (g->dec) {
        for (uint32_t L = 0; L < g->n_dec; L++) {
            free_attn(&g->dec[L].self);
            free_attn(&g->dec[L].cross);
            dfree(&g->dec[L].ln_self);
            dfree(&g->dec[L].ln_cross);
            dfree(&g->dec[L].wi); dfree(&g->dec[L].wo);
            dfree(&g->dec[L].ln_ff);
            dfree(&g->dec[L].xk); dfree(&g->dec[L].xv);
        }
        free(g->dec);
    }
    dfree(&g->mem); dfree(&g->xn); dfree(&g->qkv);
    dfree(&g->kbuf); dfree(&g->vbuf); dfree(&g->proj);
    dfree(&g->ff_h); dfree(&g->score); dfree(&g->bias);
    dfree(&g->dh); dfree(&g->dxn); dfree(&g->dq);
    dfree(&g->dctx); dfree(&g->dproj); dfree(&g->dff);
    dfree(&g->dscore); dfree(&g->dbias); dfree(&g->logits);
    if (g->kc) {
        for (uint32_t L = 0; L < g->n_dec; L++) dfree(&g->kc[L]);
        free(g->kc);
    }
    if (g->vc) {
        for (uint32_t L = 0; L < g->n_dec; L++) dfree(&g->vc[L]);
        free(g->vc);
    }
    free(g->hlogits);
    free(g);
}

static int gfwd_kv_grow(gfwd_t *g) {
    size_t ncap = g->kv_cap ? g->kv_cap * 2 : 64;
    if (ncap > g->max_tgt) ncap = g->max_tgt;
    if (ncap <= g->kv_cap) {
        fprintf(stderr, "t5_gpu: kv cache full (max_tgt=%zu)\n", g->max_tgt);
        return -1;
    }
    size_t bytes = ncap * g->inner * sizeof(float);
    for (uint32_t L = 0; L < g->n_dec; L++) {
        cp_dptr_t nk = 0, nv = 0;
        if (cp_gpu_rt_alloc(&nk, bytes) != 0 ||
            cp_gpu_rt_alloc(&nv, bytes) != 0) {
            cp_gpu_rt_free(nk);
            cp_gpu_rt_free(nv);
            return -1;
        }
        if (g->kc[L] && g->kv_cap) {
            cp_gpu_rt_copy_d2d(nk, g->kc[L],
                               g->kv_cap * g->inner * sizeof(float));
            cp_gpu_rt_copy_d2d(nv, g->vc[L],
                               g->kv_cap * g->inner * sizeof(float));
            cp_gpu_rt_free(g->kc[L]);
            cp_gpu_rt_free(g->vc[L]);
        }
        g->kc[L] = nk;
        g->vc[L] = nv;
    }
    g->kv_cap = ncap;
    return 0;
}

int t5_gpu_forward_init(const cp_t5_config_t *cfg, const cp_t5_weights_t *w,
                        size_t max_src, size_t max_tgt) {
    if (!t5_gpu_ready() && t5_gpu_init() != 0) return -1;
    if (g_fwd) return 0;
    if (!cfg || !w || !w->encoder || !w->decoder) return -1;
    if (cfg->arch != CP_ARCH_ENCODER_DECODER) {
        fprintf(stderr, "t5_gpu: expected encoder-decoder\n");
        return -1;
    }
    if (w->gated_ff) {
        fprintf(stderr, "t5_gpu: gated-gelu FFN not implemented yet\n");
        return -1;
    }

    if (max_src == 0) max_src = 512;
    if (max_tgt == 0) max_tgt = 512;

    gfwd_t *g = calloc(1, sizeof *g);
    if (!g) return -1;
    g->n_enc = w->n_encoder;
    g->n_dec = w->n_decoder;
    g->n_heads = cfg->num_heads;
    g->d_kv = cfg->d_kv;
    g->d_model = cfg->d_model;
    g->d_ff = cfg->d_ff;
    g->V = cfg->vocab_size;
    g->inner = cfg->inner_dim;
    g->num_buckets = cfg->relative_attention_num_buckets;
    g->max_distance = CP_T5_REL_MAX_DISTANCE_DEFAULT;
    g->eps = (float)cfg->layer_norm_epsilon;
    g->max_src = max_src;
    g->max_tgt = max_tgt;
    g->kv_cap = 64;
    if (g->kv_cap > max_tgt) g->kv_cap = max_tgt;

    g->enc = calloc(g->n_enc, sizeof *g->enc);
    g->dec = calloc(g->n_dec, sizeof *g->dec);
    g->kc = calloc(g->n_dec, sizeof *g->kc);
    g->vc = calloc(g->n_dec, sizeof *g->vc);
    g->hlogits = malloc((size_t)g->V * sizeof *g->hlogits);
    if (!g->enc || !g->dec || !g->kc || !g->vc || !g->hlogits) goto fail;

    if (upload_as_f32(&g->shared, w->shared) != 0) goto fail;
    if (upload_as_f32(&g->enc_final_ln, w->enc_final_ln) != 0) goto fail;
    if (upload_as_f32(&g->dec_final_ln, w->dec_final_ln) != 0) goto fail;

    for (uint32_t L = 0; L < g->n_enc; L++) {
        const cp_t5_enc_layer_w_t *lw = &w->encoder[L];
        g_enc_t *gl = &g->enc[L];
        if (upload_attn(&gl->self, &lw->self_attn, L == 0) != 0) goto fail;
        if (L > 0 && w->encoder[0].self_attn.rel_bias) {
            /* alias device pointer of layer 0 */
            gl->self.rel_bias = g->enc[0].self.rel_bias;
        }
        if (upload_as_f32(&gl->ln_self, lw->ln_self) != 0 ||
            upload_as_f32(&gl->wi, lw->ff.wi) != 0 ||
            upload_as_f32(&gl->wo, lw->ff.wo) != 0 ||
            upload_as_f32(&gl->ln_ff, lw->ln_ff) != 0)
            goto fail;
    }
    for (uint32_t L = 0; L < g->n_dec; L++) {
        const cp_t5_dec_layer_w_t *lw = &w->decoder[L];
        g_dec_t *gl = &g->dec[L];
        if (upload_attn(&gl->self, &lw->self_attn, L == 0) != 0) goto fail;
        if (L > 0 && w->decoder[0].self_attn.rel_bias)
            gl->self.rel_bias = g->dec[0].self.rel_bias;
        if (upload_attn(&gl->cross, &lw->cross_attn, 0) != 0) goto fail;
        if (upload_as_f32(&gl->ln_self, lw->ln_self) != 0 ||
            upload_as_f32(&gl->ln_cross, lw->ln_cross) != 0 ||
            upload_as_f32(&gl->wi, lw->ff.wi) != 0 ||
            upload_as_f32(&gl->wo, lw->ff.wo) != 0 ||
            upload_as_f32(&gl->ln_ff, lw->ln_ff) != 0)
            goto fail;
        if (cp_gpu_rt_alloc(&gl->xk, max_src * g->inner * sizeof(float)) != 0 ||
            cp_gpu_rt_alloc(&gl->xv, max_src * g->inner * sizeof(float)) != 0)
            goto fail;
    }

    size_t S = max_src, I = g->inner, D = g->d_model, F = g->d_ff;
    size_t nh = g->n_heads;
    if (cp_gpu_rt_alloc(&g->mem, S * D * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->xn, S * D * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->qkv, S * I * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->kbuf, S * I * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->vbuf, S * I * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->proj, S * D * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->ff_h, S * F * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->score, nh * S * S * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->bias, nh * S * S * sizeof(float)) != 0)
        goto fail;

    if (cp_gpu_rt_alloc(&g->dh, D * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->dxn, D * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->dq, I * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->dctx, I * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->dproj, D * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->dff, F * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->dscore,
                        nh * (max_src > max_tgt ? max_src : max_tgt) *
                                sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->dbias, nh * max_tgt * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&g->logits, (size_t)g->V * sizeof(float)) != 0)
        goto fail;

    for (uint32_t L = 0; L < g->n_dec; L++) {
        if (cp_gpu_rt_alloc(&g->kc[L], g->kv_cap * I * sizeof(float)) != 0 ||
            cp_gpu_rt_alloc(&g->vc[L], g->kv_cap * I * sizeof(float)) != 0)
            goto fail;
    }

    g_fwd = g;
    return 0;
fail:
    fprintf(stderr, "t5_gpu: forward init failed\n");
    gfwd_release(g);
    return -1;
}

void t5_gpu_forward_free(void) {
    if (!g_fwd) return;
    /* don't double-free aliased rel_bias on layers 1..N */
    if (g_fwd->enc) {
        for (uint32_t L = 1; L < g_fwd->n_enc; L++)
            g_fwd->enc[L].self.rel_bias = 0;
    }
    if (g_fwd->dec) {
        for (uint32_t L = 1; L < g_fwd->n_dec; L++)
            g_fwd->dec[L].self.rel_bias = 0;
    }
    gfwd_release(g_fwd);
    g_fwd = NULL;
}

/* ---- device helpers ---- */

static int launch_linear_seq(cp_dptr_t Y, cp_dptr_t W, cp_dptr_t X,
                             int seq, int rows, int cols) {
    void *a[] = {&Y, &W, &X, &seq, &rows, &cols};
    return kl(s_linear_seq, GRD((size_t)seq * (size_t)rows), 256, a);
}

static int launch_ln_row(cp_dptr_t y, cp_dptr_t x, cp_dptr_t w, int n,
                         float eps) {
    /* one block serial LN matching CPU loop order */
    void *a[] = {&y, &x, &w, &n, &eps};
    return kl(s_ln, 1, 1, a);
}

static int launch_ln_seq(cp_dptr_t Y, cp_dptr_t X, cp_dptr_t w, int seq,
                         int d, float eps) {
    for (int i = 0; i < seq; i++) {
        cp_dptr_t yi = Y + (cp_dptr_t)((size_t)i * (size_t)d * sizeof(float));
        cp_dptr_t xi = X + (cp_dptr_t)((size_t)i * (size_t)d * sizeof(float));
        if (launch_ln_row(yi, xi, w, d, eps) != 0) return -1;
    }
    return 0;
}

static int launch_mha(cp_dptr_t out, cp_dptr_t q, cp_dptr_t k, cp_dptr_t v,
                      cp_dptr_t bias, cp_dptr_t score,
                      int n_heads, int d_kv, int seq_q, int seq_k,
                      int causal) {
    cp_dptr_t bias_arg = bias; /* may be 0 */
    void *a[] = {&score, &q, &k, &bias_arg, &n_heads, &d_kv, &seq_q, &seq_k,
                 &causal};
    size_t nsc = (size_t)n_heads * (size_t)seq_q * (size_t)seq_k;
    if (kl(s_ascore, GRD(nsc), 256, a) != 0) return -1;
    void *b[] = {&score, &n_heads, &seq_q, &seq_k};
    if (kl(s_asoft, GRD((size_t)n_heads * (size_t)seq_q), 256, b) != 0)
        return -1;
    void *c[] = {&out, &score, &v, &n_heads, &d_kv, &seq_q, &seq_k};
    return kl(s_av, GRD((size_t)n_heads * (size_t)seq_q * (size_t)d_kv), 256,
              c);
}

static int dense_relu_dense_seq(gfwd_t *g, cp_dptr_t Y, cp_dptr_t X,
                                cp_dptr_t wi, cp_dptr_t wo, int seq) {
    int D = (int)g->d_model, F = (int)g->d_ff;
    if (launch_linear_seq(g->ff_h, wi, X, seq, F, D) != 0) return -1;
    {
        int n = seq * F;
        void *a[] = {&g->ff_h, &n};
        if (kl(s_relu, GRD((size_t)n), 256, a) != 0) return -1;
    }
    return launch_linear_seq(Y, wo, g->ff_h, seq, D, F);
}

static int dense_relu_dense_1(gfwd_t *g, cp_dptr_t y, cp_dptr_t x,
                              cp_dptr_t wi, cp_dptr_t wo) {
    int D = (int)g->d_model, F = (int)g->d_ff;
    void *a[] = {&g->dff, &wi, &x, &F, &D};
    if (kl(s_matvec, GRD((size_t)F), 256, a) != 0) return -1;
    {
        void *b[] = {&g->dff, &F};
        if (kl(s_relu, GRD((size_t)F), 256, b) != 0) return -1;
    }
    void *c[] = {&y, &wo, &g->dff, &D, &F};
    return kl(s_matvec, GRD((size_t)D), 256, c);
}

int t5_gpu_encode(const uint32_t *src_ids, size_t n) {
    gfwd_t *g = g_fwd;
    if (!g || !t5_gpu_ready() || !src_ids || n == 0 || n > g->max_src)
        return -1;

    int D = (int)g->d_model, I = (int)g->inner, S = (int)n;
    int nh = (int)g->n_heads, dkv = (int)g->d_kv;

    /* embed each token into mem[i] */
    for (size_t i = 0; i < n; i++) {
        int tok = (int)src_ids[i];
        cp_dptr_t row =
                g->mem + (cp_dptr_t)(i * (size_t)D * sizeof(float));
        void *a[] = {&row, &g->shared, &tok, &D};
        if (kl(s_embed, GRD((size_t)D), 256, a) != 0) return -1;
    }

    /* relative bias for encoder self-attn (bidirectional) — host then upload */
    float *hbias = malloc((size_t)nh * n * n * sizeof *hbias);
    float *hrel = NULL;
    if (!hbias) return -1;
    {
        /* download layer0 rel_bias [num_buckets, n_heads] */
        size_t rb_bytes = (size_t)g->num_buckets * nh * sizeof(float);
        hrel = malloc(rb_bytes);
        if (!hrel) {
            free(hbias);
            return -1;
        }
        if (cp_gpu_rt_download(hrel, g->enc[0].self.rel_bias, rb_bytes) != 0) {
            free(hbias);
            free(hrel);
            return -1;
        }
        cp_t5_position_bias(hbias, hrel, n, n, nh, g->num_buckets,
                            (size_t)g->max_distance, 1);
        if (cp_gpu_rt_upload(g->bias, hbias,
                             (size_t)nh * n * n * sizeof(float)) != 0) {
            free(hbias);
            free(hrel);
            return -1;
        }
    }
    free(hbias);
    free(hrel);

    for (uint32_t L = 0; L < g->n_enc; L++) {
        g_enc_t *lw = &g->enc[L];

        if (launch_ln_seq(g->xn, g->mem, lw->ln_self, S, D, g->eps) != 0)
            return -1;
        if (launch_linear_seq(g->qkv, lw->self.q, g->xn, S, I, D) != 0 ||
            launch_linear_seq(g->kbuf, lw->self.k, g->xn, S, I, D) != 0 ||
            launch_linear_seq(g->vbuf, lw->self.v, g->xn, S, I, D) != 0)
            return -1;
        if (launch_mha(g->qkv, g->qkv, g->kbuf, g->vbuf, g->bias, g->score,
                       nh, dkv, S, S, 0) != 0)
            return -1;
        if (launch_linear_seq(g->proj, lw->self.o, g->qkv, S, D, I) != 0)
            return -1;
        {
            int nn = S * D;
            void *a[] = {&g->mem, &g->proj, &nn};
            if (kl(s_add, GRD((size_t)nn), 256, a) != 0) return -1;
        }

        if (launch_ln_seq(g->xn, g->mem, lw->ln_ff, S, D, g->eps) != 0)
            return -1;
        if (dense_relu_dense_seq(g, g->proj, g->xn, lw->wi, lw->wo, S) != 0)
            return -1;
        {
            int nn = S * D;
            void *a[] = {&g->mem, &g->proj, &nn};
            if (kl(s_add, GRD((size_t)nn), 256, a) != 0) return -1;
        }
    }

    if (launch_ln_seq(g->mem, g->mem, g->enc_final_ln, S, D, g->eps) != 0)
        return -1;

    /* precompute cross K/V per decoder layer from memory */
    for (uint32_t L = 0; L < g->n_dec; L++) {
        g_dec_t *lw = &g->dec[L];
        if (launch_linear_seq(lw->xk, lw->cross.k, g->mem, S, I, D) != 0 ||
            launch_linear_seq(lw->xv, lw->cross.v, g->mem, S, I, D) != 0)
            return -1;
    }

    if (cp_gpu_rt_sync() != 0) return -1;
    g->src_len = n;
    return 0;
}

int t5_gpu_encode_download(float *out) {
    gfwd_t *g = g_fwd;
    if (!g || !out || g->src_len == 0) return -1;
    return cp_gpu_rt_download(out, g->mem,
                              g->src_len * g->d_model * sizeof(float));
}

void t5_gpu_decode_reset(void) {
    /* write slot is `pos`; stale cache beyond pos is unused */
}

const float *t5_gpu_decode_step(uint32_t tok, size_t pos) {
    gfwd_t *g = g_fwd;
    if (!g || !t5_gpu_ready() || g->src_len == 0) return NULL;
    if (pos >= g->max_tgt) return NULL;
    if (pos + 1 > g->kv_cap && gfwd_kv_grow(g) != 0) return NULL;

    int D = (int)g->d_model, I = (int)g->inner;
    int nh = (int)g->n_heads, dkv = (int)g->d_kv;
    int Vi = (int)g->V;
    int tok_i = (int)tok;
    int Sk = (int)(pos + 1);
    int Senc = (int)g->src_len;

    {
        void *a[] = {&g->dh, &g->shared, &tok_i, &D};
        if (kl(s_embed, GRD((size_t)D), 256, a) != 0) return NULL;
    }

    /* decoder self-attn relative bias for qlen=1, klen=pos+1 (causal) */
    float *hbias = malloc((size_t)nh * (size_t)Sk * sizeof *hbias);
    float *hrel = malloc((size_t)g->num_buckets * (size_t)nh * sizeof *hrel);
    if (!hbias || !hrel) {
        free(hbias);
        free(hrel);
        return NULL;
    }
    if (cp_gpu_rt_download(hrel, g->dec[0].self.rel_bias,
                           (size_t)g->num_buckets * nh * sizeof(float)) != 0) {
        free(hbias);
        free(hrel);
        return NULL;
    }
    cp_t5_position_bias(hbias, hrel, 1, (size_t)Sk, nh, g->num_buckets,
                        (size_t)g->max_distance, 0);
    if (cp_gpu_rt_upload(g->dbias, hbias, (size_t)nh * (size_t)Sk * sizeof(float)) !=
        0) {
        free(hbias);
        free(hrel);
        return NULL;
    }
    free(hbias);
    free(hrel);

    for (uint32_t L = 0; L < g->n_dec; L++) {
        g_dec_t *lw = &g->dec[L];

        /* self-attn */
        if (launch_ln_row(g->dxn, g->dh, lw->ln_self, D, g->eps) != 0)
            return NULL;
        {
            void *a[] = {&g->dq, &lw->self.q, &g->dxn, &I, &D};
            if (kl(s_matvec, GRD((size_t)I), 256, a) != 0) return NULL;
        }
        {
            /* project k,v for this token into cache slot */
            cp_dptr_t kslot =
                    g->kc[L] + (cp_dptr_t)(pos * (size_t)I * sizeof(float));
            cp_dptr_t vslot =
                    g->vc[L] + (cp_dptr_t)(pos * (size_t)I * sizeof(float));
            void *a[] = {&kslot, &lw->self.k, &g->dxn, &I, &D};
            if (kl(s_matvec, GRD((size_t)I), 256, a) != 0) return NULL;
            void *b[] = {&vslot, &lw->self.v, &g->dxn, &I, &D};
            if (kl(s_matvec, GRD((size_t)I), 256, b) != 0) return NULL;
        }
        /* cache already holds only positions <= pos; no causal mask */
        if (launch_mha(g->dctx, g->dq, g->kc[L], g->vc[L], g->dbias, g->dscore,
                       nh, dkv, 1, Sk, 0) != 0)
            return NULL;
        {
            void *a[] = {&g->dproj, &lw->self.o, &g->dctx, &D, &I};
            if (kl(s_matvec, GRD((size_t)D), 256, a) != 0) return NULL;
            void *b[] = {&g->dh, &g->dproj, &D};
            if (kl(s_add, GRD((size_t)D), 256, b) != 0) return NULL;
        }

        /* cross-attn */
        if (launch_ln_row(g->dxn, g->dh, lw->ln_cross, D, g->eps) != 0)
            return NULL;
        {
            void *a[] = {&g->dq, &lw->cross.q, &g->dxn, &I, &D};
            if (kl(s_matvec, GRD((size_t)I), 256, a) != 0) return NULL;
        }
        if (launch_mha(g->dctx, g->dq, lw->xk, lw->xv, 0, g->dscore, nh, dkv, 1,
                       Senc, 0) != 0)
            return NULL;
        {
            void *a[] = {&g->dproj, &lw->cross.o, &g->dctx, &D, &I};
            if (kl(s_matvec, GRD((size_t)D), 256, a) != 0) return NULL;
            void *b[] = {&g->dh, &g->dproj, &D};
            if (kl(s_add, GRD((size_t)D), 256, b) != 0) return NULL;
        }

        /* FFN */
        if (launch_ln_row(g->dxn, g->dh, lw->ln_ff, D, g->eps) != 0)
            return NULL;
        if (dense_relu_dense_1(g, g->dproj, g->dxn, lw->wi, lw->wo) != 0)
            return NULL;
        {
            void *a[] = {&g->dh, &g->dproj, &D};
            if (kl(s_add, GRD((size_t)D), 256, a) != 0) return NULL;
        }
    }

    if (launch_ln_row(g->dxn, g->dh, g->dec_final_ln, D, g->eps) != 0)
        return NULL;
    {
        void *a[] = {&g->logits, &g->shared, &g->dxn, &Vi, &D};
        if (kl(s_matvec, GRD((size_t)Vi), 256, a) != 0) return NULL;
    }
    if (cp_gpu_rt_download(g->hlogits, g->logits,
                           (size_t)g->V * sizeof(float)) != 0)
        return NULL;
    return g->hlogits;
}
