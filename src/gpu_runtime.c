/*
 * gpu_runtime.c - shared CUDA driver + NVRTC shell (see gpu_runtime.h)
 */

#include "gpu_runtime.h"

#include <dlfcn.h>
#include <dirent.h>
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
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, CUstream,
                                    void **, void **);

static nvrtcResult (*p_nvrtcCreateProgram)(nvrtcProgram *, const char *,
                                           const char *, int,
                                           const char *const *,
                                           const char *const *);
static nvrtcResult (*p_nvrtcCompileProgram)(nvrtcProgram, int,
                                            const char *const *);
static nvrtcResult (*p_nvrtcGetPTXSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetPTX)(nvrtcProgram, char *);
static nvrtcResult (*p_nvrtcDestroyProgram)(nvrtcProgram *);
static const char *(*p_nvrtcGetErrorString)(nvrtcResult);
static nvrtcResult (*p_nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetProgramLog)(nvrtcProgram, char *);

static void     *s_hCuda;
static void     *s_hNvrtc;
static CUcontext s_ctx;
static int       s_ready;
static int       s_ref;
static int       s_major, s_minor;

static int load_sym(void *h, const char *lib, const char *sym, void **out) {
    *out = dlsym(h, sym);
    if (!*out) {
        fprintf(stderr, "gpu_rt: dlsym '%s' missing in %s\n", sym, lib);
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

static void *try_open_in(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return NULL;
    struct dirent *e;
    void *h = NULL;
    char full[1024];
    while ((e = readdir(d)) != NULL) {
        if (!str_starts(e->d_name, "nvrtc64_") || !str_ends(e->d_name, ".dll"))
            continue;
        if (strstr(e->d_name, ".alt.")) continue;
        snprintf(full, sizeof full, "%s/%s", dir, e->d_name);
        h = dlopen(full, RTLD_NOW);
        if (h) {
            fprintf(stdout, "gpu_rt: NVRTC loaded from %s\n", full);
            break;
        }
    }
    closedir(d);
    return h;
}

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
        fprintf(stderr, "gpu_rt: CUDA toolkit root not found: %s\n", root);
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

static void rt_teardown(void) {
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
    s_ref = 0;
    s_major = s_minor = 0;
}

int cp_gpu_rt_init(void) {
    if (s_ready) {
        s_ref++;
        return 0;
    }

    s_hCuda = dlopen("nvcuda.dll", RTLD_NOW);
    if (!s_hCuda) {
        fprintf(stderr, "gpu_rt: cannot load nvcuda.dll (no NVIDIA driver?)\n");
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
        fprintf(stderr, "gpu_rt: NVRTC library not found\n");
        rt_teardown();
        return -1;
    }
    LOAD_NVRTC(nvrtcCreateProgram);
    LOAD_NVRTC(nvrtcCompileProgram);
    LOAD_NVRTC(nvrtcGetPTXSize);
    LOAD_NVRTC(nvrtcGetPTX);
    LOAD_NVRTC(nvrtcDestroyProgram);
    LOAD_NVRTC(nvrtcGetErrorString);
    LOAD_NVRTC(nvrtcGetProgramLogSize);
    LOAD_NVRTC(nvrtcGetProgramLog);

    if (p_cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: cuInit failed\n");
        rt_teardown();
        return -1;
    }

    CUdevice dev = 0;
    if (p_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: cuDeviceGet failed\n");
        rt_teardown();
        return -1;
    }
    char name[256] = {0};
    if (p_cuDeviceGetName(name, sizeof name, dev) == CUDA_SUCCESS)
        fprintf(stdout, "gpu_rt: device 0: %s\n", name);

    if (p_cuDeviceGetAttribute(&s_major,
                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                               dev) != CUDA_SUCCESS ||
        p_cuDeviceGetAttribute(&s_minor,
                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                               dev) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: get compute capability failed\n");
        rt_teardown();
        return -1;
    }
    fprintf(stdout, "gpu_rt: compute capability %d.%d\n", s_major, s_minor);

    if (p_cuCtxCreate_v2(&s_ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev)
            != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: cuCtxCreate failed\n");
        rt_teardown();
        return -1;
    }

    fflush(stdout);
    s_ready = 1;
    s_ref = 1;
    return 0;
}

void cp_gpu_rt_shutdown(void) {
    if (!s_ready) return;
    if (s_ref > 1) {
        s_ref--;
        return;
    }
    rt_teardown();
}

int cp_gpu_rt_ready(void) {
    return s_ready;
}

int cp_gpu_rt_cc(int *major, int *minor) {
    if (!s_ready) return -1;
    if (major) *major = s_major;
    if (minor) *minor = s_minor;
    return 0;
}

int cp_gpu_rt_compile(const char *src, const char *name,
                      cp_gpu_module_t *out_mod) {
    if (!s_ready || !src || !out_mod) return -1;
    *out_mod = NULL;

    nvrtcProgram prog = NULL;
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", s_major,
             s_minor);
    const char *opts[] = {arch};

    if (p_nvrtcCreateProgram(&prog, src, name ? name : "kernels.cu", 0, NULL,
                             NULL) != NVRTC_SUCCESS) {
        fprintf(stderr, "gpu_rt: nvrtcCreateProgram failed\n");
        return -1;
    }
    nvrtcResult cr = p_nvrtcCompileProgram(prog, 1, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t logSize = 0;
        p_nvrtcGetProgramLogSize(prog, &logSize);
        char *log = malloc(logSize ? logSize : 1);
        if (log && logSize) p_nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "gpu_rt: NVRTC compile failed (%s):\n%s\n",
                p_nvrtcGetErrorString(cr), log ? log : "");
        free(log);
        p_nvrtcDestroyProgram(&prog);
        return -1;
    }

    size_t ptxSize = 0;
    if (p_nvrtcGetPTXSize(prog, &ptxSize) != NVRTC_SUCCESS || ptxSize == 0) {
        fprintf(stderr, "gpu_rt: nvrtcGetPTXSize failed\n");
        p_nvrtcDestroyProgram(&prog);
        return -1;
    }
    char *ptx = malloc(ptxSize);
    if (!ptx) {
        p_nvrtcDestroyProgram(&prog);
        return -1;
    }

    CUmodule mod = NULL;
    int rc = -1;
    if (p_nvrtcGetPTX(prog, ptx) == NVRTC_SUCCESS &&
        p_cuModuleLoadData(&mod, ptx) == CUDA_SUCCESS) {
        *out_mod = (cp_gpu_module_t)mod;
        rc = 0;
    } else {
        fprintf(stderr, "gpu_rt: PTX load failed\n");
    }
    free(ptx);
    p_nvrtcDestroyProgram(&prog);
    return rc;
}

void cp_gpu_rt_module_unload(cp_gpu_module_t mod) {
    if (mod && p_cuModuleUnload) p_cuModuleUnload((CUmodule)mod);
}

int cp_gpu_rt_get_function(cp_gpu_module_t mod, const char *fn_name,
                           cp_gpu_fn_t *out_fn) {
    if (!s_ready || !mod || !fn_name || !out_fn) return -1;
    CUfunction fn = NULL;
    if (p_cuModuleGetFunction(&fn, (CUmodule)mod, fn_name) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: kernel '%s' missing from module\n", fn_name);
        return -1;
    }
    *out_fn = (cp_gpu_fn_t)fn;
    return 0;
}

int cp_gpu_rt_alloc(cp_dptr_t *d, size_t bytes) {
    if (!s_ready || !d) return -1;
    CUdeviceptr p = 0;
    if (p_cuMemAlloc_v2(&p, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: cuMemAlloc(%zu) failed\n", bytes);
        return -1;
    }
    *d = (cp_dptr_t)p;
    return 0;
}

void cp_gpu_rt_free(cp_dptr_t d) {
    if (d && p_cuMemFree_v2) p_cuMemFree_v2((CUdeviceptr)d);
}

int cp_gpu_rt_upload(cp_dptr_t d, const void *src, size_t bytes) {
    if (!s_ready) return -1;
    if (p_cuMemcpyHtoD_v2((CUdeviceptr)d, src, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: HtoD %zu bytes failed\n", bytes);
        return -1;
    }
    return 0;
}

int cp_gpu_rt_download(void *dst, cp_dptr_t d, size_t bytes) {
    if (!s_ready) return -1;
    if (p_cuMemcpyDtoH_v2(dst, (CUdeviceptr)d, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: DtoH %zu bytes failed\n", bytes);
        return -1;
    }
    return 0;
}

int cp_gpu_rt_copy_d2d(cp_dptr_t dst, cp_dptr_t src, size_t bytes) {
    if (!s_ready) return -1;
    if (p_cuMemcpyDtoD_v2((CUdeviceptr)dst, (CUdeviceptr)src, bytes)
            != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: DtoD %zu bytes failed\n", bytes);
        return -1;
    }
    return 0;
}

int cp_gpu_rt_upload_new(cp_dptr_t *d, const void *src, size_t bytes) {
    if (cp_gpu_rt_alloc(d, bytes) != 0) return -1;
    if (cp_gpu_rt_upload(*d, src, bytes) != 0) {
        cp_gpu_rt_free(*d);
        *d = 0;
        return -1;
    }
    return 0;
}

int cp_gpu_rt_launch(cp_gpu_fn_t fn, unsigned grid, unsigned block,
                     void **args) {
    if (!s_ready || !fn) return -1;
    if (p_cuLaunchKernel((CUfunction)fn, grid, 1, 1, block, 1, 1, 0, NULL, args,
                         NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: kernel launch failed\n");
        return -1;
    }
    return 0;
}

int cp_gpu_rt_sync(void) {
    if (!s_ready) return -1;
    if (p_cuCtxSynchronize() != CUDA_SUCCESS) {
        fprintf(stderr, "gpu_rt: synchronize failed\n");
        return -1;
    }
    return 0;
}
