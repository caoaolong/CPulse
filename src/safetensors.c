/*
 * safetensors.c - minimal safetensors reader (see safetensors.h)
 *
 * Project : CPulse - C inference engine for Qwen2.5-0.5B
 * Stage   : 2 - weight loading
 *
 * Contains a hand-rolled recursive-descent JSON parser covering exactly
 * what a safetensors header needs: objects, arrays, strings (with escape
 * handling) and unsigned integers. Unknown JSON fields are skipped, so the
 * parser tolerates "__metadata__" and any future extra fields.
 */

#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* dtype table                                                        */
/* ------------------------------------------------------------------ */

static const struct {
    const char *name;
    st_dtype_t  dt;
    size_t      esize;
} k_dtypes[] = {
    { "F64",  ST_DTYPE_F64,  8 },
    { "F32",  ST_DTYPE_F32,  4 },
    { "F16",  ST_DTYPE_F16,  2 },
    { "BF16", ST_DTYPE_BF16, 2 },
    { "I64",  ST_DTYPE_I64,  8 },
    { "I32",  ST_DTYPE_I32,  4 },
    { "I16",  ST_DTYPE_I16,  2 },
    { "I8",   ST_DTYPE_I8,   1 },
    { "U8",   ST_DTYPE_U8,   1 },
    { "BOOL", ST_DTYPE_BOOL, 1 },
};

const char *st_dtype_name(st_dtype_t dt) {
    for (size_t i = 0; i < sizeof(k_dtypes) / sizeof(k_dtypes[0]); i++) {
        if (k_dtypes[i].dt == dt) return k_dtypes[i].name;
    }
    return "?";
}

size_t st_dtype_size(st_dtype_t dt) {
    for (size_t i = 0; i < sizeof(k_dtypes) / sizeof(k_dtypes[0]); i++) {
        if (k_dtypes[i].dt == dt) return k_dtypes[i].esize;
    }
    return 0;
}

static st_dtype_t dtype_from_string(const char *s) {
    for (size_t i = 0; i < sizeof(k_dtypes) / sizeof(k_dtypes[0]); i++) {
        if (strcmp(s, k_dtypes[i].name) == 0) return k_dtypes[i].dt;
    }
    return ST_DTYPE_COUNT; /* unknown */
}

/* ------------------------------------------------------------------ */
/* small JSON parser                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *p;   /* read cursor   */
    const char *end; /* one-past-end of the parse region */
    int         err;
} jparse_t;

static void jskip_ws(jparse_t *j) {
    while (j->p < j->end &&
           (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) {
        j->p++;
    }
}

static int jpeek(jparse_t *j) {
    jskip_ws(j);
    if (j->p >= j->end) { j->err = 1; return -1; }
    return (unsigned char)*j->p;
}

static int jnext(jparse_t *j) {
    int c = jpeek(j);
    if (!j->err && j->p < j->end) j->p++;
    return c;
}

static void jexpect(jparse_t *j, char c) {
    if (jnext(j) != c) j->err = 1;
}

/*
 * Consume the remainder of a string literal (the opening quote has already
 * been consumed). Handles backslash escapes.
 */
static void jskip_string(jparse_t *j) {
    while (!j->err) {
        if (j->p >= j->end) { j->err = 1; return; }
        char c = *j->p++;
        if (c == '\\') {
            if (j->p >= j->end) { j->err = 1; return; }
            j->p++; /* skip the escaped character */
        } else if (c == '"') {
            return;
        }
    }
}

/* Skip any JSON value: string, number, object, array, true/false/null. */
static void jskip_value(jparse_t *j) {
    int c = jnext(j);
    if (j->err) return;

    if (c == '"') {
        jskip_string(j);
        return;
    }
    if (c == '{' || c == '[') {
        int depth = 1;
        while (!j->err && depth > 0) {
            if (j->p >= j->end) { j->err = 1; return; }
            char d = *j->p++;
            if (d == '"') {
                jskip_string(j); /* strings may contain brackets/braces */
            } else if (d == '{' || d == '[') {
                depth++;
            } else if (d == '}' || d == ']') {
                depth--;
            }
        }
        return;
    }
    /* numbers / literals: run to the next delimiter */
    while (!j->err && j->p < j->end) {
        char d = *j->p;
        if (d == ',' || d == '}' || d == ']') return;
        j->p++;
    }
}

/* append one char to a growable buffer; returns 0 on OOM */
static int jstr_push(char **buf, size_t *len, size_t *cap, char c) {
    if (*len + 1 >= *cap) {
        size_t ncap = *cap * 2;
        char *nb = realloc(*buf, ncap);
        if (!nb) return 0;
        *buf = nb;
        *cap = ncap;
    }
    (*buf)[(*len)++] = c;
    return 1;
}

/*
 * Parse a JSON string into a fresh NUL-terminated malloc'd buffer (without
 * the surrounding quotes). Handles the standard escapes. \uXXXX is decoded
 * only for ASCII code points (< 0x80); tensor names and dtype tags in
 * safetensors headers are pure ASCII, so anything else is rejected as an
 * error to keep the parser honest and small.
 */
static char *jparse_string(jparse_t *j, size_t *out_len) {
    if (jpeek(j) != '"') { j->err = 1; return NULL; }
    j->p++; /* opening quote */

    size_t cap = 32, len = 0;
    char *buf = malloc(cap);
    if (!buf) { j->err = 1; return NULL; }

    while (!j->err) {
        if (j->p >= j->end) { j->err = 1; break; }
        char c = *j->p++;
        if (c == '"') break; /* closing quote */

        if (c != '\\') {
            if (!jstr_push(&buf, &len, &cap, c)) { j->err = 1; break; }
            continue;
        }

        if (j->p >= j->end) { j->err = 1; break; }
        char e = *j->p++;
        switch (e) {
        case '"':  c = '"';  break;
        case '\\': c = '\\'; break;
        case '/':  c = '/';  break;
        case 'b':  c = '\b'; break;
        case 'f':  c = '\f'; break;
        case 'n':  c = '\n'; break;
        case 'r':  c = '\r'; break;
        case 't':  c = '\t'; break;
        case 'u': {
            if (j->end - j->p < 4) { j->err = 1; break; }
            unsigned code = 0;
            for (int k = 0; k < 4; k++) {
                char h = j->p[k];
                unsigned d;
                if      (h >= '0' && h <= '9') d = (unsigned)(h - '0');
                else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                else { j->err = 1; break; }
                code = code * 16 + d;
            }
            if (j->err) break;
            j->p += 4;
            if (code < 0x80) {
                c = (char)code;
            } else {
                fprintf(stderr, "st_load: non-ASCII \\u escape in header "
                                "(safetensors names should be ASCII)\n");
                j->err = 1;
                break;
            }
            break;
        }
        default:
            j->err = 1;
            break;
        }
        if (j->err) break;
        if (!jstr_push(&buf, &len, &cap, c)) { j->err = 1; break; }
    }

    if (j->err) { free(buf); return NULL; }
    if (!jstr_push(&buf, &len, &cap, '\0')) { free(buf); j->err = 1; return NULL; }
    if (out_len) *out_len = len - 1; /* exclude NUL */
    return buf;
}

/* Parse an unsigned 64-bit integer (safetensors headers use only these). */
static uint64_t jparse_uint(jparse_t *j) {
    jskip_ws(j);
    if (j->p < j->end && *j->p == '+') j->p++;
    if (j->p < j->end && *j->p == '-') { j->err = 1; return 0; }

    uint64_t v = 0;
    int digits = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        v = v * 10 + (uint64_t)(*j->p - '0');
        j->p++;
        digits++;
    }
    if (digits == 0) j->err = 1;
    if (j->p < j->end && (*j->p == '.' || *j->p == 'e' || *j->p == 'E')) {
        j->err = 1; /* floats are not expected in tensor metadata */
    }
    return v;
}

/* ------------------------------------------------------------------ */
/* file loading                                                       */
/* ------------------------------------------------------------------ */

static uint64_t read_le64(const uint8_t *p) {
    return (uint64_t)p[0]
         | ((uint64_t)p[1] <<  8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

static uint8_t *read_entire_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
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

    if (fsz <= 0) {
        fprintf(stderr, "st_load: '%s' is empty or unreadable\n", path);
        fclose(fp);
        return NULL;
    }

    uint8_t *buf = malloc((size_t)fsz);
    if (!buf) {
        fprintf(stderr, "st_load: out of memory (%lld bytes)\n", fsz);
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)fsz, fp) != (size_t)fsz) {
        fprintf(stderr, "st_load: short read on '%s'\n", path);
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *out_size = (size_t)fsz;
    return buf;
}

static int tensor_push(st_file_t *f, st_tensor_t t) {
    if (f->n_tensors == f->cap) {
        size_t ncap = f->cap ? f->cap * 2 : 64;
        st_tensor_t *nt = realloc(f->tensors, ncap * sizeof(st_tensor_t));
        if (!nt) return 0;
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
static int parse_tensor_entry(jparse_t *j, st_tensor_t *t, size_t data_start) {
    memset(t, 0, sizeof(*t));
    t->dtype = ST_DTYPE_COUNT; /* mark "not seen yet" */

    jexpect(j, '{');
    if (j->err) return 0;

    int have_offsets = 0;
    uint64_t begin = 0, end = 0;

    int c = jpeek(j);
    while (!j->err && c != '}') {
        char *key = jparse_string(j, NULL);
        if (j->err) return 0;
        jexpect(j, ':');

        if (strcmp(key, "dtype") == 0) {
            char *dstr = jparse_string(j, NULL);
            if (!j->err) {
                t->dtype = dtype_from_string(dstr);
                if (t->dtype == ST_DTYPE_COUNT) {
                    fprintf(stderr, "st_load: unknown dtype '%s'\n", dstr);
                    j->err = 1;
                }
            }
            free(dstr);
        } else if (strcmp(key, "shape") == 0) {
            jexpect(j, '[');
            while (!j->err && jpeek(j) != ']') {
                uint64_t d = jparse_uint(j);
                if (j->err) break;
                if (t->ndim >= ST_MAX_DIM) {
                    fprintf(stderr, "st_load: tensor rank > %d unsupported\n", ST_MAX_DIM);
                    j->err = 1;
                    break;
                }
                t->shape[t->ndim++] = (size_t)d;
                int sep = jpeek(j);
                if (sep == ',') { j->p++; }
                else if (sep != ']') { j->err = 1; }
            }
            jexpect(j, ']');
        } else if (strcmp(key, "data_offsets") == 0) {
            jexpect(j, '[');
            begin = jparse_uint(j);
            int sep = jpeek(j);
            if (sep == ',') j->p++;
            end = jparse_uint(j);
            jexpect(j, ']');
            have_offsets = 1;
        } else {
            jskip_value(j); /* "__metadata__"-like or unknown extra field */
        }

        free(key);
        if (j->err) return 0;
        c = jpeek(j);
        if (c == ',') { j->p++; }
        else if (c != '}') { j->err = 1; }
    }
    jexpect(j, '}');
    if (j->err) return 0;

    if (t->dtype == ST_DTYPE_COUNT) {
        fprintf(stderr, "st_load: tensor entry missing 'dtype'\n");
        return 0;
    }
    if (t->ndim == 0) {
        fprintf(stderr, "st_load: tensor entry missing 'shape'\n");
        return 0;
    }
    if (!have_offsets) {
        fprintf(stderr, "st_load: tensor entry missing 'data_offsets'\n");
        return 0;
    }
    if (end < begin) {
        fprintf(stderr, "st_load: data_offsets end < begin\n");
        return 0;
    }

    t->nbytes = (size_t)(end - begin);
    t->data_offset = data_start + (size_t)begin;
    return 1;
}

st_file_t *st_load(const char *path) {
    size_t size = 0;
    uint8_t *buf = read_entire_file(path, &size);
    if (!buf) return NULL;

    if (size < 8) {
        fprintf(stderr, "st_load: '%s' too small for a safetensors file\n", path);
        free(buf);
        return NULL;
    }

    uint64_t hlen = read_le64(buf);
    if (hlen == 0 || hlen > size - 8 || hlen > (100u << 20)) {
        fprintf(stderr, "st_load: implausible header length %llu\n",
                (unsigned long long)hlen);
        free(buf);
        return NULL;
    }

    st_file_t *f = calloc(1, sizeof(*f));
    if (!f) {
        fprintf(stderr, "st_load: out of memory\n");
        free(buf);
        return NULL;
    }
    f->buf = buf;
    f->buf_size = size;
    f->header_len = hlen;

    /* parse the JSON header */
    size_t data_start = 8 + (size_t)hlen;
    jparse_t j = { (const char *)buf + 8, (const char *)buf + 8 + hlen, 0 };

    jexpect(&j, '{');
    if (!j.err) {
        int c = jpeek(&j);
        while (!j.err && c != '}') {
            char *name = jparse_string(&j, NULL);
            if (j.err) break;
            jexpect(&j, ':');

            if (strcmp(name, "__metadata__") == 0) {
                jskip_value(&j); /* optional file-level metadata: not needed */
            } else {
                st_tensor_t t;
                if (!parse_tensor_entry(&j, &t, data_start)) {
                    free(name);
                    break;
                }
                t.name = name;
                if (!tensor_push(f, t)) {
                    fprintf(stderr, "st_load: out of memory growing table\n");
                    free(name);
                    j.err = 1;
                    break;
                }
                name = NULL; /* ownership moved into the table */
            }
            free(name);

            c = jpeek(&j);
            if (c == ',') { j.p++; }
            else if (c != '}') { j.err = 1; }
        }
    }

    if (j.err) {
        fprintf(stderr, "st_load: malformed safetensors header in '%s'\n", path);
        st_free(f);
        return NULL;
    }
    jexpect(&j, '}');
    if (j.err) {
        fprintf(stderr, "st_load: trailing garbage after header object\n");
        st_free(f);
        return NULL;
    }

    /* resolve absolute data pointers and validate against the file bounds */
    for (size_t i = 0; i < f->n_tensors; i++) {
        st_tensor_t *t = &f->tensors[i];
        if (t->data_offset + t->nbytes > size) {
            fprintf(stderr, "st_load: tensor '%s' extends past end of file\n",
                    t->name);
            st_free(f);
            return NULL;
        }
        t->data = buf + t->data_offset;
    }

    return f;
}

void st_free(st_file_t *f) {
    if (!f) return;
    for (size_t i = 0; i < f->n_tensors; i++) {
        free(f->tensors[i].name);
    }
    free(f->tensors);
    free(f->buf);
    free(f);
}

const st_tensor_t *st_find(const st_file_t *f, const char *name) {
    if (!f) return NULL;
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    return NULL;
}
