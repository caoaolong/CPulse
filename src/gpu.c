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

static const char *kMatvecSrc =
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
    "}\n";

static int compile_matvec(int major, int minor) {
    nvrtcProgram prog = NULL;
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", major, minor);
    const char *opts[] = { arch };

    if (p_nvrtcCreateProgram(&prog, kMatvecSrc, "matvec.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
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
        p_cuModuleLoadData(&s_mod, ptx) == CUDA_SUCCESS &&
        p_cuModuleGetFunction(&s_matvec, s_mod, "matvec_bf16") == CUDA_SUCCESS) {
        rc = 0;
    } else {
        fprintf(stderr, "gpu: PTX load / function lookup failed\n");
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

    if (compile_matvec(major, minor) != 0) goto fail;

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
