/*
 * gpu.c - CUDA driver-API + NVRTC backend (see gpu.h)
 *
 * No toolkit headers are used: the handful of driver/NVRTC types used
 * here are redeclared (their ABI is stable), and every entry point is
 * resolved with dlsym. BF16 is handled as raw uint16_t with bit-shift
 * widening, so kernel sources need no CUDA headers either.
 */

#include "gpu.h"

#include <dlfcn.h>
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- minimal driver-API / NVRTC declarations ---------------- */

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS 0
#define CU_CTX_SCHED_BLOCKING_SYNC 4
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

typedef struct _nvrtcProgram *nvrtcProgram;
typedef int nvrtcResult;
#define NVRTC_SUCCESS 0

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*p_cuDeviceGetAttribute)(int *, int, CUdevice);
static CUresult (*p_cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxDestroy_v2)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree_v2)(CUdeviceptr);
static CUresult (*p_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemcpyDtoD_v2)(CUdeviceptr, CUdeviceptr, size_t);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleUnload)(CUmodule);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, CUstream, void **, void **);

static nvrtcResult (*p_nvrtcCreateProgram)(nvrtcProgram *, const char *, const char *,
                                           int, const char * const *, const char * const *);
static nvrtcResult (*p_nvrtcCompileProgram)(nvrtcProgram, int, const char * const *);
static nvrtcResult (*p_nvrtcGetPTXSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetPTX)(nvrtcProgram, char *);
static nvrtcResult (*p_nvrtcDestroyProgram)(nvrtcProgram *);
static const char *(*p_nvrtcGetErrorString)(nvrtcResult);
static nvrtcResult (*p_nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetProgramLog)(nvrtcProgram, char *);

/* ---------------- state ---------------- */

static void *s_hCuda;
static void *s_hNvrtc;
static CUcontext s_ctx;
static CUmodule s_mod;
static CUfunction s_matvec;
/* extra kernels for the full forward (see kKernelSrc) */
static CUfunction s_k_embed, s_k_bias, s_k_rms, s_k_rope,
                   s_k_ascore, s_k_asoft, s_k_av, s_k_swiglu, s_k_add;
static int s_ready;

/* ---------------- helpers ---------------- */

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        fprintf(stderr, "gpu: %s failed (CUresult %d)\n", #call, _r); \
        return -1; \
    } \
} while (0)

static int load_sym(void *h, const char *lib, const char *sym, void **out) {
    *out = dlsym(h, sym);
    if (!*out) {
        fprintf(stderr, "gpu: dlsym '%s' missing in %s\n", sym, lib);
        return -1;
    }
    return 0;
}

#define LOAD_CU(sym) \
    if (load_sym(s_hCuda, "nvcuda", #sym, (void **)&p_##sym) != 0) return -1
#define LOAD_NVRTC(sym) \
    if (load_sym(s_hNvrtc, "nvrtc", #sym, (void **)&p_##sym) != 0) return -1

static int str_starts(const char *s, const char *p) {
    return strncmp(s, p, strlen(p)) == 0;
}

static int str_ends(const char *s, const char *p) {
    size_t a = strlen(s), b = strlen(p);
    return a >= b && strcmp(s + a - b, p) == 0;
}

/* dlopen the first nvrtc64_*.dll (not .alt) found in dir. */
static void *try_open_in(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return NULL;
    struct dirent *e;
    void *h = NULL;
    char full[1024];
    while ((e = readdir(d)) != NULL) {
        if (!str_starts(e->d_name, "nvrtc64_") || !str_ends(e->d_name, ".dll")) continue;
        if (strstr(e->d_name, ".alt.")) continue;
        snprintf(full, sizeof full, "%s/%s", dir, e->d_name);
        h = dlopen(full, RTLD_NOW);
        if (h) {
            /* informational: stdout, so it is visible under debuggers
             * whose stderr pipe is not wired to the console. */
            fprintf(stdout, "gpu: NVRTC loaded from %s\n", full);
            break;
        }
    }
    closedir(d);
    return h;
}

/* Locate NVRTC: $CUDA_PATH first, then every toolkit under the default root. */
static void *open_nvrtc(void) {
    char dir[1024];
    const char *env = getenv("CUDA_PATH");
    if (env && *env) {
        snprintf(dir, sizeof dir, "%s/bin/x64", env);
        void *h = try_open_in(dir);
        if (h) return h;
        snprintf(dir, sizeof dir, "%s/bin", env);
        h = try_open_in(dir);
        if (h) return h;
    }
    const char *root = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA";
    DIR *d = opendir(root);
    if (!d) {
        fprintf(stderr, "gpu: CUDA toolkit root not found: %s\n", root);
        return NULL;
    }
    struct dirent *e;
    void *h = NULL;
    char sub[1024];
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        snprintf(sub, sizeof sub, "%s/%s/bin/x64", root, e->d_name);
        h = try_open_in(sub);
        if (h) break;
        snprintf(sub, sizeof sub, "%s/%s/bin", root, e->d_name);
        h = try_open_in(sub);
        if (h) break;
    }
    closedir(d);
    return h;
}

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

static int compile_kernels(int major, int minor) {
    nvrtcProgram prog = NULL;
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", major, minor);
    const char *opts[] = { arch };

    if (p_nvrtcCreateProgram(&prog, kKernelSrc, "kernels.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
        fprintf(stderr, "gpu: nvrtcCreateProgram failed\n");
        return -1;
    }
    nvrtcResult cr = p_nvrtcCompileProgram(prog, 1, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t logSize = 0;
        p_nvrtcGetProgramLogSize(prog, &logSize);
        char *log = malloc(logSize ? logSize : 1);
        if (log && logSize) p_nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "gpu: NVRTC compile failed (%s):\n%s\n",
                p_nvrtcGetErrorString(cr), log ? log : "");
        free(log);
        p_nvrtcDestroyProgram(&prog);
        return -1;
    }
    size_t ptxSize = 0;
    char *ptx = NULL;
    if (p_nvrtcGetPTXSize(prog, &ptxSize) != NVRTC_SUCCESS || ptxSize == 0) {
        fprintf(stderr, "gpu: nvrtcGetPTXSize failed\n");
        p_nvrtcDestroyProgram(&prog);
        return -1;
    }
    ptx = malloc(ptxSize);
    if (!ptx) {
        p_nvrtcDestroyProgram(&prog);
        return -1;
    }
    int rc = -1;
    if (p_nvrtcGetPTX(prog, ptx) == NVRTC_SUCCESS &&
        p_cuModuleLoadData(&s_mod, ptx) == CUDA_SUCCESS) {
        static const struct { const char *name; CUfunction *fn; } k_syms[] = {
            { "matvec_bf16",  &s_matvec },
            { "embed_row",    &s_k_embed },
            { "add_bf16_bias", &s_k_bias },
            { "rmsnorm",      &s_k_rms },
            { "rope",         &s_k_rope },
            { "attn_scores",  &s_k_ascore },
            { "attn_softmax", &s_k_asoft },
            { "attn_av",      &s_k_av },
            { "swiglu",       &s_k_swiglu },
            { "add_inplace",  &s_k_add },
        };
        rc = 0;
        for (size_t i = 0; i < sizeof k_syms / sizeof k_syms[0]; i++) {
            if (p_cuModuleGetFunction(k_syms[i].fn, s_mod, k_syms[i].name)
                    != CUDA_SUCCESS) {
                fprintf(stderr, "gpu: kernel '%s' missing from module\n",
                        k_syms[i].name);
                rc = -1;
                break;
            }
        }
    } else {
        fprintf(stderr, "gpu: PTX load failed\n");
    }
    free(ptx);
    p_nvrtcDestroyProgram(&prog);
    return rc;
}

/* ---------------- public API ---------------- */

int gpu_init(void) {
    if (s_ready) return 0;

    s_hCuda = dlopen("nvcuda.dll", RTLD_NOW);
    if (!s_hCuda) {
        fprintf(stderr, "gpu: cannot load nvcuda.dll (no NVIDIA driver?)\n");
        return -1;
    }
    LOAD_CU(cuInit);
    LOAD_CU(cuDeviceGet);
    LOAD_CU(cuDeviceGetName);
    LOAD_CU(cuDeviceGetAttribute);
    LOAD_CU(cuCtxCreate_v2);
    LOAD_CU(cuCtxDestroy_v2);
    LOAD_CU(cuCtxSynchronize);
    LOAD_CU(cuMemAlloc_v2);
    LOAD_CU(cuMemFree_v2);
    LOAD_CU(cuMemcpyHtoD_v2);
    LOAD_CU(cuMemcpyDtoH_v2);
    LOAD_CU(cuMemcpyDtoD_v2);
    LOAD_CU(cuModuleLoadData);
    LOAD_CU(cuModuleUnload);
    LOAD_CU(cuModuleGetFunction);
    LOAD_CU(cuLaunchKernel);

    s_hNvrtc = open_nvrtc();
    if (!s_hNvrtc) {
        fprintf(stderr, "gpu: NVRTC library not found\n");
        goto fail;
    }
    LOAD_NVRTC(nvrtcCreateProgram);
    LOAD_NVRTC(nvrtcCompileProgram);
    LOAD_NVRTC(nvrtcGetPTXSize);
    LOAD_NVRTC(nvrtcGetPTX);
    LOAD_NVRTC(nvrtcDestroyProgram);
    LOAD_NVRTC(nvrtcGetErrorString);
    LOAD_NVRTC(nvrtcGetProgramLogSize);
    LOAD_NVRTC(nvrtcGetProgramLog);

    CU_CHECK(p_cuInit(0));

    CUdevice dev = 0;
    CU_CHECK(p_cuDeviceGet(&dev, 0));
    char name[256] = {0};
    if (p_cuDeviceGetName(name, sizeof name, dev) == CUDA_SUCCESS) {
        /* informational: stdout (see note above); errors stay on stderr. */
        fprintf(stdout, "gpu: device 0: %s\n", name);
    }
    int major = 0, minor = 0;
    CU_CHECK(p_cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(p_cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    fprintf(stdout, "gpu: compute capability %d.%d\n", major, minor);

    CU_CHECK(p_cuCtxCreate_v2(&s_ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));

    if (compile_kernels(major, minor) != 0) goto fail;

    /* Flush now: under a debugger stdout may be a pipe (fully buffered),
     * and an early-stopped session would otherwise lose these lines. */
    fflush(stdout);

    s_ready = 1;
    return 0;

fail:
    gpu_shutdown();
    return -1;
}

void gpu_shutdown(void) {
    if (s_matvec || s_mod) {
        if (s_mod && p_cuModuleUnload) p_cuModuleUnload(s_mod);
        s_matvec = NULL;
        s_mod = NULL;
    }
    if (s_ctx && p_cuCtxDestroy_v2) {
        p_cuCtxDestroy_v2(s_ctx);
        s_ctx = NULL;
    }
    if (s_hNvrtc) {
        dlclose(s_hNvrtc);
        s_hNvrtc = NULL;
    }
    if (s_hCuda) {
        dlclose(s_hCuda);
        s_hCuda = NULL;
    }
    s_ready = 0;
}

int gpu_ready(void) {
    return s_ready;
}

int gpu_matvec(float *y, const uint16_t *W, const float *x,
               size_t rows, size_t cols) {
    if (!s_ready) {
        fprintf(stderr, "gpu: not initialized (call gpu_init first)\n");
        return -1;
    }
    if (rows == 0 || cols == 0 || rows > INT_MAX || cols > INT_MAX) {
        fprintf(stderr, "gpu: bad matvec dims %zu x %zu\n", rows, cols);
        return -1;
    }

    CUdeviceptr dW = 0, dx = 0, dy = 0;
    int rc = -1;
    if (p_cuMemAlloc_v2(&dW, rows * cols * sizeof(uint16_t)) != CUDA_SUCCESS ||
        p_cuMemAlloc_v2(&dx, cols * sizeof(float)) != CUDA_SUCCESS ||
        p_cuMemAlloc_v2(&dy, rows * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: cuMemAlloc failed\n");
        goto out;
    }
    if (p_cuMemcpyHtoD_v2(dW, W, rows * cols * sizeof(uint16_t)) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(dx, x, cols * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: HtoD copy failed\n");
        goto out;
    }

    int r = (int)rows, c = (int)cols;
    void *args[] = { &dy, &dW, &dx, &r, &c };
    unsigned int block = 256;
    unsigned int grid = (unsigned int)((rows + block - 1) / block);
    if (p_cuLaunchKernel(s_matvec, grid, 1, 1, block, 1, 1,
                         0, NULL, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: kernel launch failed\n");
        goto out;
    }
    if (p_cuCtxSynchronize() != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: synchronize failed\n");
        goto out;
    }
    if (p_cuMemcpyDtoH_v2(y, dy, rows * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: DtoH copy failed\n");
        goto out;
    }
    rc = 0;

out:
    if (dW) p_cuMemFree_v2(dW);
    if (dx) p_cuMemFree_v2(dx);
    if (dy) p_cuMemFree_v2(dy);
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
    CUdeviceptr q_w, q_b, k_w, k_b, v_w, v_b, o_w, in_ln, post_ln,
                gate, up, down;
} glay_t;

typedef struct {
    /* device-resident weights (BF16 bytes, shapes already validated) */
    CUdeviceptr embed;    /* [V,H] - tied lm_head */
    CUdeviceptr final_ln; /* [H] */
    glay_t *lay;          /* [n_layers] */
    /* fp32 scratch */
    CUdeviceptr h, xn, q, attn, proj; /* [H] */
    CUdeviceptr kv_k, kv_v;          /* [kv_dim] */
    CUdeviceptr gbuf, ubuf, abuf;    /* [I] */
    CUdeviceptr logits;              /* [V] */
    CUdeviceptr score;               /* [n_heads * kv_cap] */
    /* kv caches: [n_layers] each, [kv_cap * kv_dim] floats */
    CUdeviceptr *kc, *vc;
    size_t kv_cap;
    /* config copies */
    uint32_t n_layers, n_heads, kv_heads, head_dim, H, I, V;
    size_t kv_dim;
    float eps, theta;
    float *hlogits; /* host-side [V] */
} gfwd_t;

static gfwd_t *g_fwd;

#define GRD(n) ((unsigned int)(((size_t)(n) + 255) / 256))

static int kl(CUfunction f, unsigned grid, unsigned block, void **args) {
    if (!f) return -1;
    if (p_cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL)
            != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: kernel launch failed\n");
        return -1;
    }
    return 0;
}

static int dev_upload(CUdeviceptr *d, const void *src, size_t bytes) {
    if (p_cuMemAlloc_v2(d, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: cuMemAlloc(%zu) failed\n", bytes);
        return -1;
    }
    if (p_cuMemcpyHtoD_v2(*d, src, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: upload of %zu bytes failed\n", bytes);
        p_cuMemFree_v2(*d);
        *d = 0;
        return -1;
    }
    return 0;
}

static int dalloc(CUdeviceptr *d, size_t floats) {
    if (p_cuMemAlloc_v2(d, floats * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: scratch alloc (%zu floats) failed\n", floats);
        return -1;
    }
    return 0;
}

static void dfree(CUdeviceptr *d) {
    if (*d) p_cuMemFree_v2(*d);
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
        CUdeviceptr nk = 0, nv = 0;
        if (p_cuMemAlloc_v2(&nk, bytes) != CUDA_SUCCESS ||
            p_cuMemAlloc_v2(&nv, bytes) != CUDA_SUCCESS) {
            if (nk) p_cuMemFree_v2(nk);
            if (nv) p_cuMemFree_v2(nv);
            fprintf(stderr, "gpu: kv grow to %zu failed\n", ncap);
            return -1;
        }
        if (g->kc[L]) {
            p_cuMemcpyDtoD_v2(nk, g->kc[L], g->kv_cap * g->kv_dim * sizeof(float));
            p_cuMemcpyDtoD_v2(nv, g->vc[L], g->kv_cap * g->kv_dim * sizeof(float));
            p_cuMemFree_v2(g->kc[L]);
            p_cuMemFree_v2(g->vc[L]);
        }
        g->kc[L] = nk;
        g->vc[L] = nv;
    }
    CUdeviceptr ns = 0;
    if (p_cuMemAlloc_v2(&ns, (size_t)g->n_heads * ncap * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: score grow failed\n");
        return -1;
    }
    dfree(&g->score);
    g->score = ns;
    g->kv_cap = ncap;
    return 0;
}

int gpu_forward_init(const cp_config_t *cfg, const cp_weights_t *w) {
    if (!s_ready && gpu_init() != 0) return -1;
    if (g_fwd) return 0;
    if (!cfg || !w || !w->layers || cfg->num_hidden_layers == 0) return -1;
    if (cfg->arch != CP_ARCH_QWEN2) {
        fprintf(stderr,
                "gpu: forward not implemented for arch '%s' (model_type='%s')\n",
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
    if (!g || !s_ready) return NULL;
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
            CUdeviceptr dk = g->kc[L] + (CUdeviceptr)(pos * g->kv_dim * sizeof(float));
            CUdeviceptr dv = g->vc[L] + (CUdeviceptr)(pos * g->kv_dim * sizeof(float));
            if (p_cuMemcpyDtoD_v2(dk, g->kv_k, g->kv_dim * sizeof(float)) != CUDA_SUCCESS ||
                p_cuMemcpyDtoD_v2(dv, g->kv_v, g->kv_dim * sizeof(float)) != CUDA_SUCCESS) {
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
    if (p_cuMemcpyDtoH_v2(g->hlogits, g->logits,
                          (size_t)g->V * sizeof(float)) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu: logits download failed\n");
        return NULL;
    }
    return g->hlogits;
}
