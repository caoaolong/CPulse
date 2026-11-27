/*
 * safetensors.c - minimal safetensors reader (see safetensors.h)
 *
 * Project : CPulse - C inference engine for Qwen2.5-0.5B
 * Stage   : 2 - weight loading
 *
 * Uses cJSON for the header and builds the tensor metadata table from
 * the parsed entries.
 */

#include "safetensors.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ------------------------------------------------------------------ */
/* dtype table                                                        */
/* ------------------------------------------------------------------ */

static const struct
{
    const char *name;
    st_dtype_t dt;
    size_t esize;
} k_dtypes[] = {
    {"F64", ST_DTYPE_F64, 8},
    {"F32", ST_DTYPE_F32, 4},
    {"F16", ST_DTYPE_F16, 2},
    {"BF16", ST_DTYPE_BF16, 2},
    {"I64", ST_DTYPE_I64, 8},
    {"I32", ST_DTYPE_I32, 4},
    {"I16", ST_DTYPE_I16, 2},
    {"I8", ST_DTYPE_I8, 1},
    {"U8", ST_DTYPE_U8, 1},
    {"BOOL", ST_DTYPE_BOOL, 1},
};

const char *st_dtype_name(st_dtype_t dt)
{
    for (size_t i = 0; i < sizeof(k_dtypes) / sizeof(k_dtypes[0]); i++)
    {
        if (k_dtypes[i].dt == dt)
            return k_dtypes[i].name;
    }
    return "?";
}

size_t st_dtype_size(st_dtype_t dt)
{
    for (size_t i = 0; i < sizeof(k_dtypes) / sizeof(k_dtypes[0]); i++)
    {
        if (k_dtypes[i].dt == dt)
            return k_dtypes[i].esize;
    }
    return 0;
}

static st_dtype_t dtype_from_string(const char *s)
{
    for (size_t i = 0; i < sizeof(k_dtypes) / sizeof(k_dtypes[0]); i++)
    {
        if (strcmp(s, k_dtypes[i].name) == 0)
            return k_dtypes[i].dt;
    }
    return ST_DTYPE_COUNT; /* unknown */
}

/* ------------------------------------------------------------------ */
/* file loading                                                       */
/* ------------------------------------------------------------------ */

static uint64_t read_le64(const uint8_t *p)
{
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

#ifdef _WIN32
static uint8_t *read_entire_file(const char *path, size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "st_load: cannot open '%s'\n", path);
        return NULL;
    }

    long long fsz;
    _fseeki64(fp, 0, SEEK_END);
    fsz = _ftelli64(fp);
    _fseeki64(fp, 0, SEEK_SET);

    if (fsz <= 0)
    {
        fprintf(stderr, "st_load: '%s' is empty or unreadable\n", path);
        fclose(fp);
        return NULL;
    }

    uint8_t *buf = malloc((size_t)fsz);
    if (!buf)
    {
        fprintf(stderr, "st_load: out of memory (%lld bytes)\n", fsz);
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)fsz, fp) != (size_t)fsz)
    {
        fprintf(stderr, "st_load: short read on '%s'\n", path);
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *out_size = (size_t)fsz;
    return buf;
}
#else /* !_WIN32 */

/*
 * Map the whole file read-only (POSIX). Header parsing and tensor
 * pointers work on the mapping directly; the OS pages data in on
 * first touch, so `--input` (embed-only) never faults in the other
 * ~700 MB. The mapping stays valid after close(fd).
 */
static uint8_t *map_entire_file(const char *path, size_t *out_size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "st_load: cannot open '%s'\n", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0)
    {
        fprintf(stderr, "st_load: '%s' is empty or unreadable\n", path);
        close(fd);
        return NULL;
    }
    size_t size = (size_t)st.st_size;
    void *p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED)
    {
        fprintf(stderr, "st_load: mmap failed on '%s'\n", path);
        return NULL;
    }
    *out_size = size;
    return (uint8_t *)p;
}

#endif /* _WIN32 */

/* Drop a buffer not yet owned by st_file_t. */
static void discard_file_buf(uint8_t *buf, size_t size)
{
#ifdef _WIN32
    (void)size;
    free(buf);
#else
    if (buf && size) munmap(buf, size);
#endif
}

/* Release the file buffer: munmap on POSIX, free on native Windows. */
static void release_file_buf(st_file_t *f)
{
    discard_file_buf(f->buf, f->buf_size);
    f->buf = NULL;
    f->buf_size = 0;
}

static int tensor_push(st_file_t *f, st_tensor_t t)
{
    if (f->n_tensors == f->cap)
    {
        size_t ncap = f->cap ? f->cap * 2 : 64;
        st_tensor_t *nt = realloc(f->tensors, ncap * sizeof(st_tensor_t));
        if (!nt)
            return 0;
        f->tensors = nt;
        f->cap = ncap;
    }
    f->tensors[f->n_tensors++] = t;
    return 1;
}

/*
 * Fill one tensor entry from a cJSON object with keys "dtype", "shape"
 * and "data_offsets". `data_start` is the absolute file offset of the
 * data section (8 + header length). Returns 0 on error.
 */
static int parse_tensor_entry(const cJSON *obj, st_tensor_t *t, size_t data_start)
{
    memset(t, 0, sizeof(*t));
    t->dtype = ST_DTYPE_COUNT;

    if (!cJSON_IsObject(obj))
        return 0;

    const cJSON *dtype = cJSON_GetObjectItemCaseSensitive(obj, "dtype");
    if (!cJSON_IsString(dtype) || !dtype->valuestring) {
        fprintf(stderr, "st_load: tensor entry missing 'dtype'\n");
        return 0;
    }
    t->dtype = dtype_from_string(dtype->valuestring);
    if (t->dtype == ST_DTYPE_COUNT) {
        fprintf(stderr, "st_load: unknown dtype '%s'\n", dtype->valuestring);
        return 0;
    }

    const cJSON *shape = cJSON_GetObjectItemCaseSensitive(obj, "shape");
    if (!cJSON_IsArray(shape)) {
        fprintf(stderr, "st_load: tensor entry missing 'shape'\n");
        return 0;
    }
    const cJSON *dim = NULL;
    cJSON_ArrayForEach(dim, shape) {
        if (!cJSON_IsNumber(dim)) {
            fprintf(stderr, "st_load: shape entry is not a number\n");
            return 0;
        }
        if (t->ndim >= ST_MAX_DIM) {
            fprintf(stderr, "st_load: tensor rank > %d unsupported\n", ST_MAX_DIM);
            return 0;
        }
        t->shape[t->ndim++] = (size_t)dim->valuedouble;
    }
    if (t->ndim == 0) {
        fprintf(stderr, "st_load: tensor entry missing 'shape'\n");
        return 0;
    }

    const cJSON *offs = cJSON_GetObjectItemCaseSensitive(obj, "data_offsets");
    if (!cJSON_IsArray(offs) || cJSON_GetArraySize(offs) < 2) {
        fprintf(stderr, "st_load: tensor entry missing 'data_offsets'\n");
        return 0;
    }
    const cJSON *b = cJSON_GetArrayItem(offs, 0);
    const cJSON *e = cJSON_GetArrayItem(offs, 1);
    if (!cJSON_IsNumber(b) || !cJSON_IsNumber(e)) {
        fprintf(stderr, "st_load: data_offsets must be numbers\n");
        return 0;
    }
    uint64_t begin = (uint64_t)b->valuedouble;
    uint64_t end = (uint64_t)e->valuedouble;
    if (end < begin) {
        fprintf(stderr, "st_load: data_offsets end < begin\n");
        return 0;
    }

    t->nbytes = (size_t)(end - begin);
    t->data_offset = data_start + (size_t)begin;
    return 1;
}

/* ------------------------------------------------------------------ */
/* load                                                             */
/* ------------------------------------------------------------------ */

st_file_t *st_load(const char *path)
{
    size_t size = 0;
#ifdef _WIN32
    uint8_t *buf = read_entire_file(path, &size);
#else
    uint8_t *buf = map_entire_file(path, &size);
#endif
    if (!buf)
        return NULL;

    if (size < 8)
    {
        fprintf(stderr, "st_load: '%s' too small for a safetensors file\n", path);
        discard_file_buf(buf, size);
        return NULL;
    }

    uint64_t hlen = read_le64(buf);
    if (hlen == 0 || hlen > size - 8 || hlen > (100u << 20))
    {
        fprintf(stderr, "st_load: implausible header length %llu\n",
                (unsigned long long)hlen);
        discard_file_buf(buf, size);
        return NULL;
    }

    st_file_t *f = calloc(1, sizeof(*f));
    if (!f)
    {
        fprintf(stderr, "st_load: out of memory\n");
        discard_file_buf(buf, size);
        return NULL;
    }
    f->buf = buf;
    f->buf_size = size;
    f->header_len = hlen;

    /* parse the JSON header */
    size_t data_start = 8 + (size_t)hlen;
    cJSON *root = cJSON_ParseWithLength((const char *)buf + 8, (size_t)hlen);
    if (!root || !cJSON_IsObject(root))
    {
        fprintf(stderr, "st_load: malformed safetensors header in '%s'\n", path);
        cJSON_Delete(root);
        st_free(f);
        return NULL;
    }

    const cJSON *entry = NULL;
    cJSON_ArrayForEach(entry, root)
    {
        if (!entry->string)
            continue;
        if (strcmp(entry->string, "__metadata__") == 0)
            continue; /* optional file-level metadata */

        st_tensor_t t;
        if (!parse_tensor_entry(entry, &t, data_start))
        {
            cJSON_Delete(root);
            st_free(f);
            return NULL;
        }
        {
            size_t nlen = strlen(entry->string);
            t.name = malloc(nlen + 1);
            if (t.name) memcpy(t.name, entry->string, nlen + 1);
        }
        if (!t.name || !tensor_push(f, t))
        {
            fprintf(stderr, "st_load: out of memory growing table\n");
            free(t.name);
            cJSON_Delete(root);
            st_free(f);
            return NULL;
        }
    }
    cJSON_Delete(root);

    /* header parsed OK: one-line summary to stderr (tiny on purpose:
     * when stderr is a pipe, e.g. under a debugger, a burst can block) */
    fprintf(stderr, "st_load: header parsed: header_len=%llu bytes, data_start=%zu, n_tensors=%zu\n",
            (unsigned long long)f->header_len, data_start, f->n_tensors);

    /* resolve absolute data pointers and validate against the file bounds */
    for (size_t i = 0; i < f->n_tensors; i++)
    {
        st_tensor_t *t = &f->tensors[i];
        if (t->data_offset + t->nbytes > size)
        {
            fprintf(stderr, "st_load: tensor '%s' extends past end of file\n",
                    t->name);
            st_free(f);
            return NULL;
        }
        t->data = buf + t->data_offset;
    }

    return f;
}

void st_free(st_file_t *f)
{
    if (!f)
        return;
    for (size_t i = 0; i < f->n_tensors; i++)
    {
        free(f->tensors[i].name);
    }
    free(f->tensors);
    release_file_buf(f);
    free(f);
}

const st_tensor_t *st_find(const st_file_t *f, const char *name)
{
    if (!f)
        return NULL;
    for (size_t i = 0; i < f->n_tensors; i++)
    {
        if (strcmp(f->tensors[i].name, name) == 0)
            return &f->tensors[i];
    }
    return NULL;
}
