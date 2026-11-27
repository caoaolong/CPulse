/*
 * gpu.c - Qwen CUDA kernels + forward (see gpu.h)
 *
 * Driver / NVRTC / context live in gpu_runtime.c. This file only owns
 * Qwen-specific embedded kernels and the decoder forward path.
 */

#include "qwen/gpu.h"
#include "gpu_runtime.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- Qwen module state ---------------- */

static cp_gpu_module_t s_mod;
static cp_gpu_fn_t s_matvec;
static cp_gpu_fn_t s_k_embed, s_k_bias, s_k_rms, s_k_rope, s_k_ascore,
                   s_k_asoft, s_k_av, s_k_swiglu, s_k_add;
static int s_qwen_ready;

/* defined later; shutdown releases resident forward state first */
void gpu_forward_free(void);

/* ---------------- embedded kernels ---------------- */

/* All kernels in one NVRTC module. matvec_bf16 stays byte-identical to
 * the original gputest kernel (CPU-parity baseline). The rest mirror the
 * exact summation order of their cp_* CPU counterparts in ops.c. */
static const char *kKernelSrc =
    "extern \"C\" __global__ void matvec_bf16(float *y, const unsigned short *W,\n"
    "                                       const float *x, int rows, int cols) {\n"
    "    int r = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (r >= rows) return;\n"
    "    float acc = 0.0f;\n"
    "    const unsigned short *row = W + (size_t)r * (size_t)cols;\n"
    "    for (int c = 0; c < cols; c++) {\n"
    "        unsigned int u = (unsigned int)row[c] << 16;\n"
    "        acc += __uint_as_float(u) * x[c];\n"
    "    }\n"
    "    y[r] = acc;\n"
    "}\n"
    "extern \"C\" __global__ void embed_row(float *h, const unsigned short *E,\n"
    "                                       int tok, int H) {\n"
    "    int d = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (d >= H) return;\n"
    "    unsigned int u = (unsigned int)E[(size_t)tok * H + d] << 16;\n"
    "    h[d] = __uint_as_float(u);\n"
    "}\n"
    "extern \"C\" __global__ void add_bf16_bias(float *y, const unsigned short *b,\n"
    "                                           int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    unsigned int u = (unsigned int)b[i] << 16;\n"
    "    y[i] += __uint_as_float(u);\n"
    "}\n"
    "extern \"C\" __global__ void rmsnorm(float *y, const float *x,\n"
    "                                    const unsigned short *w, int n, float eps) {\n"
    "    float ss = 0.0f;\n"
    "    for (int i = 0; i < n; i++) ss += x[i] * x[i];\n"
    "    float inv = 1.0f / sqrtf(ss / (float)n + eps);\n"
    "    for (int i = 0; i < n; i++) {\n"
    "        unsigned int u = (unsigned int)w[i] << 16;\n"
    "        y[i] = x[i] * inv * __uint_as_float(u);\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void rope(float *t, int n_heads, int head_dim,\n"
    "                                int pos, float theta) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int half = head_dim / 2;\n"
    "    if (idx >= n_heads * half) return;\n"
    "    int h = idx / half;\n"
    "    int i = idx - h * half;\n"
    "    float *head = t + h * head_dim;\n"
    "    float freq = powf(theta, -(float)(2 * i) / (float)head_dim);\n"
    "    float ang = (float)pos * freq;\n"
    "    float co = cosf(ang), si = sinf(ang);\n"
    "    float a = head[i], b = head[i + half];\n"
    "    head[i] = a * co - b * si;\n"
    "    head[i + half] = a * si + b * co;\n"
    "}\n"
    "extern \"C\" __global__ void attn_scores(float *s, const float *q,\n"
    "                                        const float *k, int n_heads,\n"
    "                                        int kv_heads, int head_dim,\n"
    "                                        int pos, float scale) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_heads * (pos + 1);\n"
    "    if (idx >= total) return;\n"
    "    int h = idx / (pos + 1);\n"
    "    int j = idx - h * (pos + 1);\n"
    "    int kh = h / (n_heads / kv_heads);\n"
    "    const float *qh = q + h * head_dim;\n"
    "    const float *kj = k + ((size_t)j * kv_heads + kh) * head_dim;\n"
    "    float acc = 0.0f;\n"
    "    for (int d = 0; d < head_dim; d++) acc += qh[d] * kj[d];\n"
    "    s[idx] = acc * scale;\n"
    "}\n"
    "extern \"C\" __global__ void attn_softmax(float *s, int n_heads, int pos) {\n"
    "    int h = threadIdx.x;\n"
    "    if (h >= n_heads) return;\n"
    "    float *sh = s + (size_t)h * (pos + 1);\n"
    "    float m = sh[0];\n"
    "    for (int j = 1; j <= pos; j++) if (sh[j] > m) m = sh[j];\n"
    "    float sum = 0.0f;\n"
    "    for (int j = 0; j <= pos; j++) { sh[j] = expf(sh[j] - m); sum += sh[j]; }\n"
    "    float inv = 1.0f / sum;\n"
    "    for (int j = 0; j <= pos; j++) sh[j] *= inv;\n"
    "}\n"
    "extern \"C\" __global__ void attn_av(float *out, const float *s,\n"
    "                                    const float *v, int n_heads,\n"
    "                                    int kv_heads, int head_dim, int pos) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (idx >= n_heads * head_dim) return;\n"
    "    int h = idx / head_dim;\n"
    "    int d = idx - h * head_dim;\n"
    "    int kh = h / (n_heads / kv_heads);\n"
    "    const float *sh = s + (size_t)h * (pos + 1);\n"
    "    float acc = 0.0f;\n"
    "    for (int j = 0; j <= pos; j++)\n"
    "        acc += sh[j] * v[((size_t)j * kv_heads + kh) * head_dim + d];\n"
    "    out[idx] = acc;\n"
    "}\n"
    "extern \"C\" __global__ void swiglu(float *y, const float *gate,\n"
    "                                  const float *up, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    float g = gate[i];\n"
    "    y[i] = g / (1.0f + expf(-g)) * up[i];\n"
    "}\n"
    "extern \"C\" __global__ void add_inplace(float *h, const float *a, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    h[i] += a[i];\n"
    "}\n";

static int load_qwen_kernels(void) {
    if (cp_gpu_rt_compile(kKernelSrc, "qwen_kernels.cu", &s_mod) != 0)
        return -1;

    static const struct {
        const char *name;
        cp_gpu_fn_t *fn;
    } k_syms[] = {
        {"matvec_bf16", &s_matvec},
        {"embed_row", &s_k_embed},
        {"add_bf16_bias", &s_k_bias},
        {"rmsnorm", &s_k_rms},
        {"rope", &s_k_rope},
        {"attn_scores", &s_k_ascore},
        {"attn_softmax", &s_k_asoft},
        {"attn_av", &s_k_av},
        {"swiglu", &s_k_swiglu},
        {"add_inplace", &s_k_add},
    };
    for (size_t i = 0; i < sizeof k_syms / sizeof k_syms[0]; i++) {
        if (cp_gpu_rt_get_function(s_mod, k_syms[i].name, k_syms[i].fn) != 0)
            return -1;
    }
    return 0;
}

/* ---------------- public API ---------------- */

int gpu_init(void) {
    if (s_qwen_ready) return 0;
    if (cp_gpu_rt_init() != 0) return -1;
    if (load_qwen_kernels() != 0) {
        gpu_shutdown();
        return -1;
    }
    fflush(stdout);
    s_qwen_ready = 1;
    return 0;
}

void gpu_shutdown(void) {
    gpu_forward_free();
    if (s_mod) {
        cp_gpu_rt_module_unload(s_mod);
        s_mod = NULL;
    }
    s_matvec = s_k_embed = s_k_bias = s_k_rms = s_k_rope = NULL;
    s_k_ascore = s_k_asoft = s_k_av = s_k_swiglu = s_k_add = NULL;
    s_qwen_ready = 0;
    cp_gpu_rt_shutdown();
}

int gpu_ready(void) {
    return s_qwen_ready && cp_gpu_rt_ready();
}

int gpu_matvec(float *y, const uint16_t *W, const float *x,
               size_t rows, size_t cols) {
    if (!gpu_ready()) {
        fprintf(stderr, "gpu: not initialized (call gpu_init first)\n");
        return -1;
    }
    if (rows == 0 || cols == 0 || rows > INT_MAX || cols > INT_MAX) {
        fprintf(stderr, "gpu: bad matvec dims %zu x %zu\n", rows, cols);
        return -1;
    }

    cp_dptr_t dW = 0, dx = 0, dy = 0;
    int rc = -1;
    if (cp_gpu_rt_alloc(&dW, rows * cols * sizeof(uint16_t)) != 0 ||
        cp_gpu_rt_alloc(&dx, cols * sizeof(float)) != 0 ||
        cp_gpu_rt_alloc(&dy, rows * sizeof(float)) != 0)
        goto out;
    if (cp_gpu_rt_upload(dW, W, rows * cols * sizeof(uint16_t)) != 0 ||
        cp_gpu_rt_upload(dx, x, cols * sizeof(float)) != 0)
        goto out;

    int r = (int)rows, c = (int)cols;
    void *args[] = {&dy, &dW, &dx, &r, &c};
    unsigned int block = 256;
    unsigned int grid = (unsigned int)((rows + block - 1) / block);
    if (cp_gpu_rt_launch(s_matvec, grid, block, args) != 0) goto out;
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
/* full decoder forward on the GPU                                      */
/*                                                                      */
/* Weights are uploaded once and stay resident; every layer's k/v and   */
/* all intermediate tensors live on the device. Per token only the id   */
/* goes in and the logits row comes back. The layer structure mirrors   */
/* forward.c exactly (same op order, same summation order per kernel), */
/* so CPU vs GPU greedy runs stay in parity.                           */
/* ==================================================================== */

typedef struct {
    cp_dptr_t q_w, q_b, k_w, k_b, v_w, v_b, o_w, in_ln, post_ln, gate, up,
              down;
} glay_t;

typedef struct {
    /* device-resident weights (BF16 bytes, shapes already validated) */
    cp_dptr_t embed;    /* [V,H] - tied lm_head */
    cp_dptr_t final_ln; /* [H] */
    glay_t *lay;        /* [n_layers] */
    /* fp32 scratch */
    cp_dptr_t h, xn, q, attn, proj; /* [H] */
    cp_dptr_t kv_k, kv_v;           /* [kv_dim] */
    cp_dptr_t gbuf, ubuf, abuf;     /* [I] */
    cp_dptr_t logits;               /* [V] */
    cp_dptr_t score;                /* [n_heads * kv_cap] */
    /* kv caches: [n_layers] each, [kv_cap * kv_dim] floats */
    cp_dptr_t *kc, *vc;
    size_t kv_cap;
    /* config copies */
    uint32_t n_layers, n_heads, kv_heads, head_dim, H, I, V;
    size_t kv_dim;
    float eps, theta;
    float *hlogits; /* host-side [V] */
} gfwd_t;

static gfwd_t *g_fwd;

#define GRD(n) ((unsigned int)(((size_t)(n) + 255) / 256))

static int kl(cp_gpu_fn_t f, unsigned grid, unsigned block, void **args) {
    return cp_gpu_rt_launch(f, grid, block, args);
}

static int dev_upload(cp_dptr_t *d, const void *src, size_t bytes) {
    return cp_gpu_rt_upload_new(d, src, bytes);
}

static int dalloc(cp_dptr_t *d, size_t floats) {
    return cp_gpu_rt_alloc(d, floats * sizeof(float));
}

static void dfree(cp_dptr_t *d) {
    cp_gpu_rt_free(*d);
    *d = 0;
}

static void gfwd_release(gfwd_t *g) {
    if (!g) return;
    dfree(&g->embed);
    dfree(&g->final_ln);
    if (g->lay) {
        for (uint32_t L = 0; L < g->n_layers; L++) {
            dfree(&g->lay[L].q_w); dfree(&g->lay[L].q_b);
            dfree(&g->lay[L].k_w); dfree(&g->lay[L].k_b);
            dfree(&g->lay[L].v_w); dfree(&g->lay[L].v_b);
            dfree(&g->lay[L].o_w); dfree(&g->lay[L].in_ln);
            dfree(&g->lay[L].post_ln); dfree(&g->lay[L].gate);
            dfree(&g->lay[L].up); dfree(&g->lay[L].down);
        }
        free(g->lay);
    }
    dfree(&g->h); dfree(&g->xn); dfree(&g->q); dfree(&g->attn);
    dfree(&g->proj); dfree(&g->kv_k); dfree(&g->kv_v);
    dfree(&g->gbuf); dfree(&g->ubuf); dfree(&g->abuf);
    dfree(&g->logits); dfree(&g->score);
    if (g->kc) {
        for (uint32_t L = 0; L < g->n_layers; L++) dfree(&g->kc[L]);
        free(g->kc);
    }
    if (g->vc) {
        for (uint32_t L = 0; L < g->n_layers; L++) dfree(&g->vc[L]);
        free(g->vc);
    }
    free(g->hlogits);
    free(g);
}

/* double every layer's cache (contents preserved); score is transient */
static int gfwd_kv_grow(gfwd_t *g) {
    size_t ncap = g->kv_cap * 2;
    size_t bytes = ncap * g->kv_dim * sizeof(float);
    for (uint32_t L = 0; L < g->n_layers; L++) {
        cp_dptr_t nk = 0, nv = 0;
        if (cp_gpu_rt_alloc(&nk, bytes) != 0 ||
            cp_gpu_rt_alloc(&nv, bytes) != 0) {
            cp_gpu_rt_free(nk);
            cp_gpu_rt_free(nv);
            fprintf(stderr, "gpu: kv grow to %zu failed\n", ncap);
            return -1;
        }
        if (g->kc[L]) {
            cp_gpu_rt_copy_d2d(nk, g->kc[L],
                               g->kv_cap * g->kv_dim * sizeof(float));
            cp_gpu_rt_copy_d2d(nv, g->vc[L],
                               g->kv_cap * g->kv_dim * sizeof(float));
            cp_gpu_rt_free(g->kc[L]);
            cp_gpu_rt_free(g->vc[L]);
        }
        g->kc[L] = nk;
        g->vc[L] = nv;
    }
    cp_dptr_t ns = 0;
    if (cp_gpu_rt_alloc(&ns, (size_t)g->n_heads * ncap * sizeof(float)) != 0) {
        fprintf(stderr, "gpu: score grow failed\n");
        return -1;
    }
    dfree(&g->score);
    g->score = ns;
    g->kv_cap = ncap;
    return 0;
}

int gpu_forward_init(const cp_config_t *cfg, const cp_weights_t *w) {
    if (!gpu_ready() && gpu_init() != 0) return -1;
    if (g_fwd) return 0;
    if (!cfg || !w || !w->layers || cfg->num_hidden_layers == 0) return -1;
    if (cfg->arch != CP_ARCH_DECODER_ONLY) {
        fprintf(stderr,
                "gpu: forward not implemented for strategy '%s' "
                "(model_type='%s')\n",
                cp_arch_name(cfg->arch), cfg->model_type);
        return -1;
    }

    gfwd_t *g = calloc(1, sizeof *g);
    if (!g) return -1;
    g->n_layers = cfg->num_hidden_layers;
    g->n_heads  = cfg->num_attention_heads;
    g->kv_heads = cfg->num_key_value_heads;
    g->head_dim = cfg->head_dim;
    g->H = cfg->hidden_size;
    g->I = cfg->intermediate_size;
    g->V = cfg->vocab_size;
    g->kv_dim = (size_t)g->kv_heads * g->head_dim;
    g->eps = (float)cfg->rms_norm_eps;
    g->theta = (float)cfg->rope_theta;
    g->kv_cap = 64;

    g->lay = calloc(g->n_layers, sizeof *g->lay);
    g->kc = calloc(g->n_layers, sizeof *g->kc);
    g->vc = calloc(g->n_layers, sizeof *g->vc);
    g->hlogits = malloc((size_t)g->V * sizeof *g->hlogits);
    if (!g->lay || !g->kc || !g->vc || !g->hlogits) goto fail;

    if (dev_upload(&g->embed, w->embed_tokens->data, w->embed_tokens->nbytes) != 0)
        goto fail;
    if (dev_upload(&g->final_ln, w->final_norm->data, w->final_norm->nbytes) != 0)
        goto fail;
    for (uint32_t L = 0; L < g->n_layers; L++) {
        const cp_layer_w_t *lw = &w->layers[L];
        glay_t *gl = &g->lay[L];
        if (dev_upload(&gl->q_w, lw->q_w->data, lw->q_w->nbytes) != 0 ||
            dev_upload(&gl->q_b, lw->q_b->data, lw->q_b->nbytes) != 0 ||
            dev_upload(&gl->k_w, lw->k_w->data, lw->k_w->nbytes) != 0 ||
            dev_upload(&gl->k_b, lw->k_b->data, lw->k_b->nbytes) != 0 ||
            dev_upload(&gl->v_w, lw->v_w->data, lw->v_w->nbytes) != 0 ||
            dev_upload(&gl->v_b, lw->v_b->data, lw->v_b->nbytes) != 0 ||
            dev_upload(&gl->o_w, lw->o_w->data, lw->o_w->nbytes) != 0 ||
            dev_upload(&gl->in_ln, lw->input_ln->data, lw->input_ln->nbytes) != 0 ||
            dev_upload(&gl->post_ln, lw->post_ln->data, lw->post_ln->nbytes) != 0 ||
            dev_upload(&gl->gate, lw->gate_w->data, lw->gate_w->nbytes) != 0 ||
            dev_upload(&gl->up, lw->up_w->data, lw->up_w->nbytes) != 0 ||
            dev_upload(&gl->down, lw->down_w->data, lw->down_w->nbytes) != 0)
            goto fail;
    }
    if (dalloc(&g->h, g->H) != 0 || dalloc(&g->xn, g->H) != 0 ||
        dalloc(&g->q, g->H) != 0 || dalloc(&g->attn, g->H) != 0 ||
        dalloc(&g->proj, g->H) != 0 || dalloc(&g->kv_k, g->kv_dim) != 0 ||
        dalloc(&g->kv_v, g->kv_dim) != 0 ||
        dalloc(&g->gbuf, g->I) != 0 || dalloc(&g->ubuf, g->I) != 0 ||
        dalloc(&g->abuf, g->I) != 0 || dalloc(&g->logits, g->V) != 0 ||
        dalloc(&g->score, (size_t)g->n_heads * g->kv_cap) != 0)
        goto fail;
    for (uint32_t L = 0; L < g->n_layers; L++) {
        if (dalloc(&g->kc[L], g->kv_cap * g->kv_dim) != 0) goto fail;
        if (dalloc(&g->vc[L], g->kv_cap * g->kv_dim) != 0) goto fail;
    }
    g_fwd = g;
    return 0;

fail:
    fprintf(stderr, "gpu: forward init failed\n");
    gfwd_release(g);
    return -1;
}

void gpu_forward_free(void) {
    gfwd_release(g_fwd);
    g_fwd = NULL;
}

void gpu_forward_reset(void) {
    /* The cache write slot is derived from `pos` on every step, and
     * attention only reads slots <= pos, so stale contents never leak
     * into a new sequence. Nothing to clear; capacity and weights stay. */
}

const float *gpu_forward_step(uint32_t tok, size_t pos) {
    gfwd_t *g = g_fwd;
    if (!g || !gpu_ready()) return NULL;
    if (pos + 1 > g->kv_cap && gfwd_kv_grow(g) != 0) return NULL;

    int Hi = (int)g->H, Ii = (int)g->I, Vi = (int)g->V;
    int nh_i = (int)g->n_heads, nkv_i = (int)g->kv_heads, hd_i = (int)g->head_dim;
    int pos_i = (int)pos, tok_i = (int)tok;
    float scale = 1.0f / sqrtf((float)g->head_dim);
    int rows, cols;

    {   /* h = embed[tok] */
        void *a[] = { &g->h, &g->embed, &tok_i, &Hi };
        if (kl(s_k_embed, GRD(g->H), 256, a) != 0) return NULL;
    }

    for (uint32_t L = 0; L < g->n_layers; L++) {
        glay_t *lw = &g->lay[L];

        {   /* xn = rmsnorm(h, in_ln) */
            void *a[] = { &g->xn, &g->h, &lw->in_ln, &Hi, &g->eps };
            if (kl(s_k_rms, 1, 1, a) != 0) return NULL;
        }
        {   /* q = matvec(q_w, xn) + q_b */
            rows = Hi; cols = Hi;
            void *a[] = { &g->q, &lw->q_w, &g->xn, &rows, &cols };
            if (kl(s_matvec, GRD(Hi), 256, a) != 0) return NULL;
            void *b[] = { &g->q, &lw->q_b, &Hi };
            if (kl(s_k_bias, GRD(Hi), 256, b) != 0) return NULL;
        }
        {   /* k, v */
            rows = (int)g->kv_dim; cols = Hi;
            void *a[] = { &g->kv_k, &lw->k_w, &g->xn, &rows, &cols };
            if (kl(s_matvec, GRD(g->kv_dim), 256, a) != 0) return NULL;
            void *b[] = { &g->kv_k, &lw->k_b, &rows };
            if (kl(s_k_bias, GRD(g->kv_dim), 256, b) != 0) return NULL;
            void *c[] = { &g->kv_v, &lw->v_w, &g->xn, &rows, &cols };
            if (kl(s_matvec, GRD(g->kv_dim), 256, c) != 0) return NULL;
            void *d[] = { &g->kv_v, &lw->v_b, &rows };
            if (kl(s_k_bias, GRD(g->kv_dim), 256, d) != 0) return NULL;
        }
        {   /* rope on q (all heads) and k (kv heads only) */
            void *a[] = { &g->q, &nh_i, &hd_i, &pos_i, &g->theta };
            if (kl(s_k_rope, GRD((size_t)g->n_heads * (g->head_dim / 2)), 256, a) != 0)
                return NULL;
            void *b[] = { &g->kv_k, &nkv_i, &hd_i, &pos_i, &g->theta };
            if (kl(s_k_rope, GRD((size_t)g->kv_heads * (g->head_dim / 2)), 256, b) != 0)
                return NULL;
        }
        {   /* cache push (DtoD at the token's slot) */
            cp_dptr_t dk =
                    g->kc[L] + (cp_dptr_t)(pos * g->kv_dim * sizeof(float));
            cp_dptr_t dv =
                    g->vc[L] + (cp_dptr_t)(pos * g->kv_dim * sizeof(float));
            if (cp_gpu_rt_copy_d2d(dk, g->kv_k, g->kv_dim * sizeof(float)) !=
                        0 ||
                cp_gpu_rt_copy_d2d(dv, g->kv_v, g->kv_dim * sizeof(float)) !=
                        0) {
                fprintf(stderr, "gpu: kv push failed\n");
                return NULL;
            }
        }
        {   /* attention: scores -> softmax (in place) -> weighted v */
            size_t n = (size_t)g->n_heads * (pos + 1);
            void *a[] = { &g->score, &g->q, &g->kc[L],
                          &nh_i, &nkv_i, &hd_i, &pos_i, &scale };
            if (kl(s_k_ascore, GRD(n), 256, a) != 0) return NULL;
            void *b[] = { &g->score, &nh_i, &pos_i };
            if (kl(s_k_asoft, 1, (unsigned)g->n_heads, b) != 0) return NULL;
            void *c[] = { &g->attn, &g->score, &g->vc[L],
                          &nh_i, &nkv_i, &hd_i, &pos_i };
            if (kl(s_k_av, GRD((size_t)g->n_heads * g->head_dim), 256, c) != 0)
                return NULL;
        }
        {   /* h += matvec(o_w, attn) */
            rows = Hi; cols = Hi;
            void *a[] = { &g->proj, &lw->o_w, &g->attn, &rows, &cols };
            if (kl(s_matvec, GRD(Hi), 256, a) != 0) return NULL;
            void *b[] = { &g->h, &g->proj, &Hi };
            if (kl(s_k_add, GRD(Hi), 256, b) != 0) return NULL;
        }
        {   /* xn = rmsnorm(h, post_ln) */
            void *a[] = { &g->xn, &g->h, &lw->post_ln, &Hi, &g->eps };
            if (kl(s_k_rms, 1, 1, a) != 0) return NULL;
        }
        {   /* mlp: gbuf/ubuf = matvec, abuf = swiglu, proj = down */
            rows = Ii; cols = Hi;
            void *a[] = { &g->gbuf, &lw->gate, &g->xn, &rows, &cols };
            if (kl(s_matvec, GRD(Ii), 256, a) != 0) return NULL;
            void *b[] = { &g->ubuf, &lw->up, &g->xn, &rows, &cols };
            if (kl(s_matvec, GRD(Ii), 256, b) != 0) return NULL;
            void *c[] = { &g->abuf, &g->gbuf, &g->ubuf, &Ii };
            if (kl(s_k_swiglu, GRD(Ii), 256, c) != 0) return NULL;
            rows = Hi; cols = Ii;
            void *d[] = { &g->proj, &lw->down, &g->abuf, &rows, &cols };
            if (kl(s_matvec, GRD(Hi), 256, d) != 0) return NULL;
            void *e[] = { &g->h, &g->proj, &Hi };
            if (kl(s_k_add, GRD(Hi), 256, e) != 0) return NULL;
        }
    }

    {   /* logits = matvec(embed_table, rmsnorm(h, final_ln)) */
        void *a[] = { &g->xn, &g->h, &g->final_ln, &Hi, &g->eps };
        if (kl(s_k_rms, 1, 1, a) != 0) return NULL;
        rows = Vi; cols = Hi;
        void *b[] = { &g->logits, &g->embed, &g->xn, &rows, &cols };
        if (kl(s_matvec, GRD(Vi), 256, b) != 0) return NULL;
    }
    if (cp_gpu_rt_download(g->hlogits, g->logits,
                           (size_t)g->V * sizeof(float)) != 0) {
        fprintf(stderr, "gpu: logits download failed\n");
        return NULL;
    }
    return g->hlogits;
}
