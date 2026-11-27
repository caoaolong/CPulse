/*
 * gpu_runtime.h - shared CUDA driver + NVRTC shell for all model backends
 *
 * Loads nvcuda/nvrtc via dlopen, owns one device context, and exposes
 * compile / mem / launch helpers. Model-specific code (qwen/gpu.c,
 * future t5/gpu.c) keeps its own kernels and forward orchestration.
 */

#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long cp_dptr_t; /* CUdeviceptr */
typedef void              *cp_gpu_module_t;
typedef void              *cp_gpu_fn_t;

/*
 * Acquire the shared runtime (refcount). First call loads libraries and
 * creates the CUDA context. Returns 0 on success.
 */
int cp_gpu_rt_init(void);

/* Release one reference; destroys context when the last user leaves. */
void cp_gpu_rt_shutdown(void);

int cp_gpu_rt_ready(void);

/* Detected compute capability (after init). */
int cp_gpu_rt_cc(int *major, int *minor);

/*
 * Compile CUDA source with NVRTC for the current CC and load as a module.
 * On success *out_mod is owned by the caller (cp_gpu_rt_module_unload).
 */
int cp_gpu_rt_compile(const char *src, const char *name,
                      cp_gpu_module_t *out_mod);

void cp_gpu_rt_module_unload(cp_gpu_module_t mod);

int cp_gpu_rt_get_function(cp_gpu_module_t mod, const char *fn_name,
                           cp_gpu_fn_t *out_fn);

int  cp_gpu_rt_alloc(cp_dptr_t *d, size_t bytes);
void cp_gpu_rt_free(cp_dptr_t d);
int  cp_gpu_rt_upload(cp_dptr_t d, const void *src, size_t bytes);
int  cp_gpu_rt_download(void *dst, cp_dptr_t d, size_t bytes);
int  cp_gpu_rt_copy_d2d(cp_dptr_t dst, cp_dptr_t src, size_t bytes);

/* Alloc + upload helper; on failure *d is 0. */
int cp_gpu_rt_upload_new(cp_dptr_t *d, const void *src, size_t bytes);

int cp_gpu_rt_launch(cp_gpu_fn_t fn, unsigned grid, unsigned block,
                     void **args);
int cp_gpu_rt_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_RUNTIME_H */
