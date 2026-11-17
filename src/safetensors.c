/*
 * safetensors.c - minimal safetensors reader (see safetensors.h)
 *
 * Project : CPulse - C inference engine for Qwen2.5-0.5B
 * Stage   : 2 - weight loading
 *
 * Uses the standalone parser in json.h/json.c for the header and builds
 * the tensor metadata table from the parsed entries.
 */

#include "safetensors.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static uint8_t *read_entire_file(const char *path, size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "st_load: cannot open '%s'\n", path);
        return NULL;
    }

    long long fsz;
#ifdef _WIN32
    _fseeki64(fp, 0, SEEK_END);
    fsz = _ftelli64(fp);
    _fseeki64(fp, 0, SEEK_SET);
#else
    fseek(fp, 0, SEEK_END);
    fsz = (long long)ftell(fp); /* fine for < 2 GiB; use ftello for larger */
    fseek(fp, 0, SEEK_SET);
#endif

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
 * Parse one tensor entry: an object with keys "dtype", "shape" and
 * "data_offsets" (any order). Unknown keys are skipped. `data_start` is the
 * absolute file offset of the data section (8 + header length), used to
 * convert the header's relative offsets into absolute ones.
 * Returns 0 on error.
 */
static int parse_tensor_entry(json_parser_t *j, st_tensor_t *t, size_t data_start)
{
    memset(t, 0, sizeof(*t));
    t->dtype = ST_DTYPE_COUNT; /* mark "not seen yet" */

    json_expect(j, '{');
    if (j->err)
        return 0;

    int have_offsets = 0;
    uint64_t begin = 0, end = 0;

    int c = json_peek(j);
    while (!j->err && c != '}')
    {
        char *key = json_parse_string(j, NULL);
        if (j->err)
            return 0;
        json_expect(j, ':');

        if (strcmp(key, "dtype") == 0)
        {
            char *dstr = json_parse_string(j, NULL);
            if (!j->err)
            {
                t->dtype = dtype_from_string(dstr);
                if (t->dtype == ST_DTYPE_COUNT)
                {
                    fprintf(stderr, "st_load: unknown dtype '%s'\n", dstr);
                    j->err = 1;
                }
            }
            free(dstr);
        }
        else if (strcmp(key, "shape") == 0)
        {
            json_expect(j, '[');
            while (!j->err && json_peek(j) != ']')
            {
                uint64_t d = json_parse_uint(j);
                if (j->err)
                    break;
                if (t->ndim >= ST_MAX_DIM)
                {
                    fprintf(stderr, "st_load: tensor rank > %d unsupported\n", ST_MAX_DIM);
                    j->err = 1;
                    break;
                }
                t->shape[t->ndim++] = (size_t)d;
                int sep = json_peek(j);
                if (sep == ',')
                {
                    j->p++;
                }
                else if (sep != ']')
                {
                    j->err = 1;
                }
            }
            json_expect(j, ']');
        }
        else if (strcmp(key, "data_offsets") == 0)
        {
            json_expect(j, '[');
            begin = json_parse_uint(j);
            int sep = json_peek(j);
            if (sep == ',')
                j->p++;
            end = json_parse_uint(j);
            json_expect(j, ']');
            have_offsets = 1;
        }
        else
        {
            json_skip_value(j); /* "__metadata__"-like or unknown extra field */
        }

        free(key);
        if (j->err)
            return 0;
        c = json_peek(j);
        if (c == ',')
        {
            j->p++;
        }
        else if (c != '}')
        {
            j->err = 1;
        }
    }
    json_expect(j, '}');
    if (j->err)
        return 0;

    if (t->dtype == ST_DTYPE_COUNT)
    {
        fprintf(stderr, "st_load: tensor entry missing 'dtype'\n");
        return 0;
    }
    if (t->ndim == 0)
    {
        fprintf(stderr, "st_load: tensor entry missing 'shape'\n");
        return 0;
    }
    if (!have_offsets)
    {
        fprintf(stderr, "st_load: tensor entry missing 'data_offsets'\n");
        return 0;
    }
    if (end < begin)
    {
        fprintf(stderr, "st_load: data_offsets end < begin\n");
        return 0;
    }

    t->nbytes = (size_t)(end - begin);
    t->data_offset = data_start + (size_t)begin;
    return 1;
}

/* ------------------------------------------------------------------ */
/* logging                                                            */
/* ------------------------------------------------------------------ */

/* The parsed header is always written here (relative to the process
 * working directory) so the tensor table can be inspected without
 * scrolling the console. Fixed path on purpose: no option, no plumbing. */
#define ST_HEADER_DUMP_PATH "header.json"

/* Write a JSON string literal (with quotes) for a NUL-terminated C
 * string. Escapes '"' and '\\' plus control chars; UTF-8 passes through. */
static void json_write_escaped(FILE *fp, const char *s)
{
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
    {
        if (*p == '"' || *p == '\\')
        {
            fputc('\\', fp);
            fputc(*p, fp);
        }
        else if (*p < 0x20)
        {
            fprintf(fp, "\\u%04x", *p);
        }
        else
        {
            fputc(*p, fp);
        }
    }
    fputc('"', fp);
}

/* Dump the parsed header as JSON (see ST_HEADER_DUMP_PATH), plus a
 * one-line summary to stderr. The summary is tiny on purpose: when
 * stderr is a pipe (e.g. under a debugger) a large burst can block. */
static void st_dump_header(const st_file_t *f, size_t data_start)
{
    fprintf(stderr, "st_load: header parsed: header_len=%llu bytes, data_start=%zu, n_tensors=%zu\n",
            (unsigned long long)f->header_len, data_start, f->n_tensors);

    FILE *fp = fopen(ST_HEADER_DUMP_PATH, "w");
    if (!fp)
    {
        fprintf(stderr, "st_dump_header: cannot open '%s' for writing\n",
                ST_HEADER_DUMP_PATH);
        return;
    }
    fprintf(fp, "{\n  \"header_len\": %llu,\n  \"data_start\": %zu,\n  \"n_tensors\": %zu,\n  \"tensors\": [\n",
            (unsigned long long)f->header_len, data_start, f->n_tensors);
    for (size_t i = 0; i < f->n_tensors; i++)
    {
        const st_tensor_t *t = &f->tensors[i];
        fprintf(fp, "    {\"name\": ");
        json_write_escaped(fp, t->name);
        fprintf(fp, ", \"dtype\": \"%s\", \"shape\": [", st_dtype_name(t->dtype));
        for (size_t d = 0; d < t->ndim; d++)
        {
            fprintf(fp, "%s%zu", d ? ", " : "", t->shape[d]);
        }
        fprintf(fp, "], \"data_offsets\": [%zu, %zu], \"nbytes\": %zu, \"data_offset\": %zu}%s\n",
                t->data_offset - data_start,
                t->data_offset - data_start + t->nbytes,
                t->nbytes, t->data_offset,
                i + 1 < f->n_tensors ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
}

st_file_t *st_load(const char *path)
{
    size_t size = 0;
    uint8_t *buf = read_entire_file(path, &size);
    if (!buf)
        return NULL;

    if (size < 8)
    {
        fprintf(stderr, "st_load: '%s' too small for a safetensors file\n", path);
        free(buf);
        return NULL;
    }

    uint64_t hlen = read_le64(buf);
    if (hlen == 0 || hlen > size - 8 || hlen > (100u << 20))
    {
        fprintf(stderr, "st_load: implausible header length %llu\n",
                (unsigned long long)hlen);
        free(buf);
        return NULL;
    }

    st_file_t *f = calloc(1, sizeof(*f));
    if (!f)
    {
        fprintf(stderr, "st_load: out of memory\n");
        free(buf);
        return NULL;
    }
    f->buf = buf;
    f->buf_size = size;
    f->header_len = hlen;

    /* parse the JSON header */
    size_t data_start = 8 + (size_t)hlen;
    json_parser_t j = {(const char *)buf + 8, (const char *)buf + 8 + hlen, 0};

    json_expect(&j, '{');
    if (!j.err)
    {
        int c = json_peek(&j);
        while (!j.err && c != '}')
        {
            char *name = json_parse_string(&j, NULL);
            if (j.err)
                break;
            json_expect(&j, ':');

            if (strcmp(name, "__metadata__") == 0)
            {
                json_skip_value(&j); /* optional file-level metadata: not needed */
            }
            else
            {
                st_tensor_t t;
                if (!parse_tensor_entry(&j, &t, data_start))
                {
                    free(name);
                    break;
                }
                t.name = name;
                if (!tensor_push(f, t))
                {
                    fprintf(stderr, "st_load: out of memory growing table\n");
                    free(name);
                    j.err = 1;
                    break;
                }
                name = NULL; /* ownership moved into the table */
            }
            free(name);

            c = json_peek(&j);
            if (c == ',')
            {
                j.p++;
            }
            else if (c != '}')
            {
                j.err = 1;
            }
        }
    }

    if (j.err)
    {
        fprintf(stderr, "st_load: malformed safetensors header in '%s'\n", path);
        st_free(f);
        return NULL;
    }
    json_expect(&j, '}');
    if (j.err)
    {
        fprintf(stderr, "st_load: trailing garbage after header object\n");
        st_free(f);
        return NULL;
    }

    /* header parsed OK: dump the results into the log */
    st_dump_header(f, data_start);

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
    free(f->buf);
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
