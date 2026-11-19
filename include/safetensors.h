/*
 * safetensors.h - minimal reader for the safetensors weight format
 *
 * Project : CPulse - C inference engine for Qwen2.5-0.5B
 * Stage   : 2 - weight loading
 *
 * safetensors file layout (https://huggingface.co/docs/safetensors):
 *
 *   [0 .. 8)          : u64 little-endian, length N of the JSON header
 *   [8 .. 8+N)        : JSON header mapping tensor name -> metadata:
 *                         { "dtype": "BF16", "shape": [d0, d1, ...],
 *                           "data_offsets": [begin, end] }
 *                       (offsets are relative to the start of the data
 *                        section, i.e. to byte 8+N of the file)
 *   [8+N .. EOF)      : contiguous raw tensor data
 *
 * This module maps the whole file read-only (mmap on POSIX, plain
 * fread on native Windows), parses the JSON header with the small
 * standalone parser in json.h/json.c (no third-party dependencies),
 * and exposes a flat metadata table with direct pointers into the
 * mapped data. Pages are faulted in on first touch, so embed-only
 * flows never pay for the rest of the file.
 *
 * Design note: tensor data is zero-copy in both paths: st_tensor_t.data
 * points into st_file_t.buf and must not outlive it.
 */

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* maximum tensor rank supported by the metadata table */
#define ST_MAX_DIM 8

typedef enum {
    ST_DTYPE_F64 = 0,
    ST_DTYPE_F32,
    ST_DTYPE_F16,
    ST_DTYPE_BF16,
    ST_DTYPE_I64,
    ST_DTYPE_I32,
    ST_DTYPE_I16,
    ST_DTYPE_I8,
    ST_DTYPE_U8,
    ST_DTYPE_BOOL,
    ST_DTYPE_COUNT /* sentinel: number of known dtypes */
} st_dtype_t;

/* metadata of one tensor */
typedef struct {
    char          *name;        /* malloc'd, e.g. "model.layers.0.mlp.up_proj.weight" */
    st_dtype_t     dtype;
    size_t         ndim;
    size_t         shape[ST_MAX_DIM];
    size_t         nbytes;      /* raw data size = end - begin offset */
    size_t         data_offset; /* absolute offset of data within the file */
    const uint8_t *data;        /* points into st_file_t.buf */
} st_tensor_t;

/* a parsed safetensors file */
typedef struct {
    uint8_t     *buf;        /* entire file contents */
    size_t       buf_size;
    uint64_t     header_len; /* JSON header length (excluding 8-byte prefix) */
    st_tensor_t *tensors;    /* metadata table */
    size_t       n_tensors;
    size_t       cap;        /* allocated slots in tensors[] */
} st_file_t;

/*
 * Load a safetensors file. Reads the whole file, parses the header and
 * builds the tensor table. On failure, prints a diagnostic to stderr and
 * returns NULL (partial state is cleaned up). On success the parsed
 * header is also written to ./header.json for inspection.
 */
st_file_t *st_load(const char *path);

/* Release everything owned by the st_file_t. */
void st_free(st_file_t *f);

/* Look up a tensor by exact name; returns NULL if not found. */
const st_tensor_t *st_find(const st_file_t *f, const char *name);

/* Human-readable dtype name ("F32", "BF16", ...); "?" for unknown. */
const char *st_dtype_name(st_dtype_t dt);

/* Element size in bytes for a dtype; 0 for unknown. */
size_t st_dtype_size(st_dtype_t dt);

#ifdef __cplusplus
}
#endif

#endif /* SAFETENSORS_H */
