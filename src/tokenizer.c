/*
 * tokenizer.c - Qwen2 byte-level BPE tokenizer (see tokenizer.h)
 *
 * Pipeline per input: split special tokens -> pre-tokenize with the
 * Split regex from tokenizer.json (hand-scanned, no regex engine) ->
 * ByteLevel (bytes_to_unicode per byte) -> BPE merges by rank ->
 * vocab lookup. Ranks come from merges.txt line order.
 */

#include "tokenizer.h"
#include "json.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- small hash map (open addressing, FNV-1a) ---------------- */

typedef struct {
    char    *key;   /* owned, NUL-terminated (never contains 0x01) */
    uint32_t klen;
    uint32_t val;
    uint8_t  used;
} t_ent_t;

typedef struct {
    t_ent_t *tab;
    size_t   cap;   /* power of two */
    size_t   n;
} t_map_t;

static uint64_t t_hash(const char *s, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int t_map_init(t_map_t *m, size_t want) {
    size_t cap = 16;
    while (cap < want) cap *= 2;
    m->tab = calloc(cap, sizeof *m->tab);
    if (!m->tab) return -1;
    m->cap = cap;
    m->n = 0;
    return 0;
}

static int t_map_grow(t_map_t *m) {
    t_map_t o = *m;
    m->tab = calloc(o.cap * 2, sizeof *m->tab);
    if (!m->tab) { m->tab = o.tab; return -1; }
    m->cap = o.cap * 2;
    m->n = 0;
    for (size_t i = 0; i < o.cap; i++) {
        if (!o.tab[i].used) continue;
        uint64_t h = t_hash(o.tab[i].key, o.tab[i].klen);
        size_t j = (size_t)h & (m->cap - 1);
        while (m->tab[j].used) j = (j + 1) & (m->cap - 1);
        m->tab[j] = o.tab[i];
        m->n++;
    }
    free(o.tab);
    return 0;
}

static int t_map_put(t_map_t *m, const char *k, size_t klen, uint32_t v) {
    if ((m->n + 1) * 4 > m->cap * 3 && t_map_grow(m) != 0) return -1;
    uint64_t h = t_hash(k, klen);
    size_t i = (size_t)h & (m->cap - 1);
    while (m->tab[i].used) {
        if (m->tab[i].klen == klen && memcmp(m->tab[i].key, k, klen) == 0) {
            m->tab[i].val = v;
            return 0;
        }
        i = (i + 1) & (m->cap - 1);
    }
    char *kc = malloc(klen + 1);
    if (!kc) return -1;
    memcpy(kc, k, klen);
    kc[klen] = '\0';
    m->tab[i].key = kc;
    m->tab[i].klen = (uint32_t)klen;
    m->tab[i].val = v;
    m->tab[i].used = 1;
    m->n++;
    return 0;
}

static int t_map_get(const t_map_t *m, const char *k, size_t klen,
                     uint32_t *out) {
    if (m->cap == 0) return -1;
    uint64_t h = t_hash(k, klen);
    size_t i = (size_t)h & (m->cap - 1);
    while (m->tab[i].used) {
        if (m->tab[i].klen == klen && memcmp(m->tab[i].key, k, klen) == 0) {
            *out = m->tab[i].val;
            return 0;
        }
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

static void t_map_free(t_map_t *m) {
    if (!m->tab) return;
    for (size_t i = 0; i < m->cap; i++) free(m->tab[i].key);
    free(m->tab);
    m->tab = NULL;
    m->cap = m->n = 0;
}

/* ---------------- tokenizer state ---------------- */

typedef struct {
    char    *text;  /* owned */
    size_t   len;
    uint32_t id;
} t_spec_t;

struct cp_tok {
    t_map_t   vocab;    /* token bytes -> id */
    t_map_t   merges;   /* "left\x01right" -> rank */
    char    **str;      /* id -> token bytes (aliases map/spec storage) */
    uint32_t *slen;
    size_t    n_ids;
    uint32_t  max_id;
    t_spec_t *spec;     /* special tokens, longest first */
    size_t    n_spec;
    uint32_t  b2u[256]; /* byte -> mapped code point (GPT-2) */
    uint16_t  u2b[0x144]; /* mapped code point -> byte, 0xFFFF = none */
};

/* ---------------- file helper ---------------- */

static int read_file(const char *path, char **out, size_t *out_n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "tok: cannot open '%s'\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0) { fclose(fp); return -1; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return -1; }
    if (sz > 0 && fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        fprintf(stderr, "tok: short read on '%s'\n", path);
        free(buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    buf[sz] = '\0';
    *out = buf;
    *out_n = (size_t)sz;
    return 0;
}

/* ---------------- byte mapping (GPT-2 bytes_to_unicode) ---------------- */

static void b2u_init(cp_tok_t *t) {
    int in_set[256] = {0};
    for (int b = 33; b < 127; b++) in_set[b] = 1;   /* '!'..'~' */
    for (int b = 161; b < 173; b++) in_set[b] = 1;
    for (int b = 174; b < 256; b++) in_set[b] = 1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        t->b2u[b] = in_set[b] ? (uint32_t)b : (uint32_t)(256 + n++);
    }
    for (int i = 0; i < 0x144; i++) t->u2b[i] = 0xFFFF;
    for (int b = 0; b < 256; b++) t->u2b[t->b2u[b]] = (uint16_t)b;
}

/* ---------------- vocab.json ---------------- */

static int load_vocab(cp_tok_t *t, const char *path) {
    char *buf = NULL;
    size_t n = 0;
    if (read_file(path, &buf, &n) != 0) return -1;
    if (t_map_init(&t->vocab, 262144) != 0) { free(buf); return -1; }

    json_parser_t j = { buf, buf + n, 0 };
    json_expect(&j, '{');
    while (!j.err) {
        int c = json_peek(&j);
        if (c == '}') { json_next(&j); break; }
        size_t klen = 0;
        char *key = json_parse_string(&j, &klen);
        if (j.err) { free(key); break; }
        json_expect(&j, ':');
        uint64_t id = 0;
        if (!j.err) id = json_parse_uint(&j);
        if (j.err || klen == 0 || klen > 0xFFFFFFFFu || id > 0xFFFFFFFFu) {
            j.err = 1;
            free(key);
            break;
        }
        if (t_map_put(&t->vocab, key, klen, (uint32_t)id) != 0) {
            fprintf(stderr, "tok: out of memory loading vocab\n");
            free(key);
            j.err = 1;
            break;
        }
        if (id > t->max_id) t->max_id = (uint32_t)id;
        free(key);
        c = json_peek(&j);
        if (c == ',') { json_next(&j); continue; }
        if (c == '}') { json_next(&j); break; }
        j.err = 1;
    }
    free(buf);
    if (j.err) {
        fprintf(stderr, "tok: parse error in '%s'\n", path);
        return -1;
    }
    return 0;
}

/* ---------------- merges.txt ---------------- */

static int load_merges(cp_tok_t *t, const char *path) {
    char *buf = NULL;
    size_t n = 0;
    if (read_file(path, &buf, &n) != 0) return -1;
    if (t_map_init(&t->merges, 262144) != 0) { free(buf); return -1; }

    uint32_t rank = 0;
    size_t p = 0;
    while (p < n) {
        size_t e = p;
        while (e < n && buf[e] != '\n') e++;
        size_t elen = e - p;
        if (elen > 0 && buf[e - 1] == '\r') elen--;
        /* NOTE: no blanket '#' skip: '#' is a legit token here
         * (e.g. "# X"); only a version header is skipped. */
        if (elen > 8 && memcmp(buf + p, "#version", 8) == 0) { p = e + 1; continue; }
        if (elen > 0) {
            /* split on the single ASCII space; tokens never hold 0x20
             * (space is U+0120 in this encoding, UTF-8 has no 0x20 byte) */
            size_t sp = p;
            while (sp < p + elen && buf[sp] != ' ') sp++;
            if (sp > p && sp + 1 < p + elen) {
                size_t la = sp - p, lb = p + elen - (sp + 1);
                size_t kl = la + 1 + lb;
                char *kb = malloc(kl + 1);
                if (!kb) { free(buf); return -1; }
                memcpy(kb, buf + p, la);
                kb[la] = '\x01';
                memcpy(kb + la + 1, buf + sp + 1, lb);
                kb[kl] = '\0';
                if (t_map_put(&t->merges, kb, kl, rank) != 0) {
                    free(kb);
                    free(buf);
                    return -1;
                }
                free(kb); /* put copies */
                rank++;
            }
        }
        p = e + 1;
    }
    return 0;
}

/* ---------------- specials (tokenizer_config.json) ---------------- */

static int add_spec(cp_tok_t *t, const char *s, size_t len, uint32_t id) {
    t_spec_t *ns = realloc(t->spec, (t->n_spec + 1) * sizeof *ns);
    if (!ns) return -1;
    t->spec = ns;
    char *cp = malloc(len + 1);
    if (!cp) return -1;
    memcpy(cp, s, len);
    cp[len] = '\0';
    t->spec[t->n_spec].text = cp;
    t->spec[t->n_spec].len = len;
    t->spec[t->n_spec].id = id;
    t->n_spec++;
    if (id > t->max_id) t->max_id = id;
    return 0;
}

/* core chat tokens, used when tokenizer_config.json is unavailable */
static int add_core_specials(cp_tok_t *t) {
    static const struct { const char *s; uint32_t id; } core[] = {
        { "<|endoftext|>", 151643 },
        { "<|im_start|>", 151644 },
        { "<|im_end|>", 151645 },
    };
    for (size_t i = 0; i < sizeof core / sizeof core[0]; i++) {
        if (add_spec(t, core[i].s, strlen(core[i].s), core[i].id) != 0)
            return -1;
    }
    return 0;
}

static int load_specials(cp_tok_t *t, const char *path) {
    char *buf = NULL;
    size_t n = 0;
    if (read_file(path, &buf, &n) != 0) return -1;

    json_parser_t j = { buf, buf + n, 0 };
    json_expect(&j, '{');
    while (!j.err) {
        int c = json_peek(&j);
        if (c == '}') { json_next(&j); break; }
        size_t klen = 0;
        char *key = json_parse_string(&j, &klen);
        if (j.err) { free(key); break; }
        json_expect(&j, ':');
        if (j.err) { free(key); break; }
        if (strcmp(key, "added_tokens_decoder") == 0) {
            json_expect(&j, '{');
            while (!j.err) {
                c = json_peek(&j);
                if (c == '}') { json_next(&j); break; }
                size_t ilen = 0;
                char *ids = json_parse_string(&j, &ilen);
                if (j.err) { free(ids); break; }
                json_expect(&j, ':');
                uint32_t id = (uint32_t)strtoul(ids, NULL, 10);
                free(ids);
                if (j.err) break;
                json_expect(&j, '{');
                while (!j.err) {
                    c = json_peek(&j);
                    if (c == '}') { json_next(&j); break; }
                    size_t flen = 0;
                    char *fk = json_parse_string(&j, &flen);
                    if (j.err) { free(fk); break; }
                    json_expect(&j, ':');
                    if (j.err) { free(fk); break; }
                    if (strcmp(fk, "content") == 0) {
                        size_t clen = 0;
                        char *cv = json_parse_string(&j, &clen);
                        if (j.err) { free(cv); free(fk); break; }
                        if (add_spec(t, cv, clen, id) != 0) {
                            free(cv);
                            free(fk);
                            j.err = 1;
                            break;
                        }
                        free(cv);
                    } else {
                        json_skip_value(&j);
                    }
                    free(fk);
                    if (j.err) break;
                    c = json_peek(&j);
                    if (c == ',') { json_next(&j); continue; }
                    if (c == '}') { json_next(&j); break; }
                    j.err = 1;
                }
                if (j.err) break;
                c = json_peek(&j);
                if (c == ',') { json_next(&j); continue; }
                if (c == '}') { json_next(&j); break; }
                j.err = 1;
            }
        } else {
            json_skip_value(&j);
        }
        free(key);
        if (j.err) break;
        c = json_peek(&j);
        if (c == ',') { json_next(&j); continue; }
        if (c == '}') { json_next(&j); break; }
        j.err = 1;
    }
    free(buf);
    if (j.err || t->n_spec == 0) {
        fprintf(stderr, "tok: parse error in '%s'\n", path);
        return -1;
    }
    return 0;
}

/* ---------------- public load/free ---------------- */

int cp_tok_load(cp_tok_t **out, const char *model_dir) {
    char path[1024];
    cp_tok_t *t = calloc(1, sizeof *t);
    if (!t) return -1;
    b2u_init(t);

    snprintf(path, sizeof path, "%s/vocab.json", model_dir);
    if (load_vocab(t, path) != 0) goto fail;
    snprintf(path, sizeof path, "%s/merges.txt", model_dir);
    if (load_merges(t, path) != 0) goto fail;
    snprintf(path, sizeof path, "%s/tokenizer_config.json", model_dir);
    if (load_specials(t, path) != 0) {
        fprintf(stderr, "tok: falling back to core special tokens\n");
        if (add_core_specials(t) != 0) goto fail;
    }

    /* longest special first so "<|im_start|>" beats any prefix */
    for (size_t i = 1; i < t->n_spec; i++) {
        t_spec_t k = t->spec[i];
        size_t k2 = i;
        while (k2 > 0 && t->spec[k2 - 1].len < k.len) {
            t->spec[k2] = t->spec[k2 - 1];
            k2--;
        }
        t->spec[k2] = k;
    }

    /* id -> token table (aliases vocab/spec storage; freed via those) */
    t->n_ids = (size_t)t->max_id + 1;
    t->str = calloc(t->n_ids, sizeof *t->str);
    t->slen = calloc(t->n_ids, sizeof *t->slen);
    if (!t->str || !t->slen) goto fail;
    for (size_t i = 0; i < t->vocab.cap; i++) {
        if (!t->vocab.tab[i].used) continue;
        uint32_t id = t->vocab.tab[i].val;
        if (id < t->n_ids && !t->str[id]) {
            t->str[id] = t->vocab.tab[i].key;
            t->slen[id] = t->vocab.tab[i].klen;
        }
    }
    for (size_t i = 0; i < t->n_spec; i++) {
        uint32_t id = t->spec[i].id;
        if (id < t->n_ids) {
            t->str[id] = t->spec[i].text;
            t->slen[id] = (uint32_t)t->spec[i].len;
        }
    }

    fprintf(stdout, "tok: vocab=%zu merges=%zu specials=%zu ids=%zu\n",
            t->vocab.n, t->merges.n, t->n_spec, t->n_ids);
        *out = t;
    return 0;

fail:
    cp_tok_free(t);
    if (out) *out = NULL;
    return -1;
}

void cp_tok_free(cp_tok_t *t) {
    if (!t) return;
    t_map_free(&t->vocab);
    t_map_free(&t->merges);
    for (size_t i = 0; i < t->n_spec; i++) free(t->spec[i].text);
    free(t->spec);
    free(t->str);
    free(t->slen);
    free(t);
}

/* ---------------- UTF-8 + char classes ---------------- */

enum { C_L, C_N, C_SP, C_NL, C_P };

static int u8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/* decode one unit; invalid sequences become U+FFFD consuming 1 byte */
static uint32_t u8_dec(const char *s, size_t n, int *ll) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80 || n == 1) { *ll = 1; return c < 0x80 ? c : 0xFFFD; }
    int l = u8_len(c);
    if (l < 2 || (size_t)l > n) { *ll = 1; return 0xFFFD; }
    for (int k = 1; k < l; k++) {
        if (((unsigned char)s[k] & 0xC0) != 0x80) { *ll = 1; return 0xFFFD; }
    }
    const unsigned char *u = (const unsigned char *)s;
    uint32_t cp;
    if (l == 2) {
        cp = ((uint32_t)(u[0] & 0x1F) << 6) | (u[1] & 0x3F);
        if (cp < 0x80) { *ll = 1; return 0xFFFD; }
    } else if (l == 3) {
        cp = ((uint32_t)(u[0] & 0x0F) << 12) |
             ((uint32_t)(u[1] & 0x3F) << 6) | (u[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) {
            *ll = 1;
            return 0xFFFD;
        }
    } else {
        cp = ((uint32_t)(u[0] & 0x07) << 18) |
             ((uint32_t)(u[1] & 0x3F) << 12) |
             ((uint32_t)(u[2] & 0x3F) << 6) | (u[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) { *ll = 1; return 0xFFFD; }
    }
    *ll = l;
    return cp;
}

/* approximates \p{L} / \p{N} for the pre-tokenizer regex */
static int cp_class(uint32_t cp) {
    if (cp == '\r' || cp == '\n') return C_NL;
    if (cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f') return C_SP;
    if (cp < 0x80) {
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return C_L;
        if (cp >= '0' && cp <= '9') return C_N;
        return C_P;
    }
    /* unicode spaces */
    if (cp == 0xA0 || cp == 0x1680 ||
        (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
        cp == 0x205F || cp == 0x3000) return C_SP;
    if (cp >= 0xFF10 && cp <= 0xFF19) return C_N;    /* fullwidth digits */
    if ((cp >= 0xFF21 && cp <= 0xFF3A) ||            /* fullwidth A-Z */
        (cp >= 0xFF41 && cp <= 0xFF5A)) return C_L;  /* fullwidth a-z */
    if ((cp >= 0xA1 && cp <= 0xBF) ||
        (cp >= 0x2000 && cp <= 0x206F) ||
        (cp >= 0x3000 && cp <= 0x303F) ||
        (cp >= 0xFE30 && cp <= 0xFE4F) ||
        (cp >= 0xFF00 && cp <= 0xFF65)) return C_P;
    return C_L;
}

/* ---------------- pre-tokenizer (Split regex, hand-scanned) ----------------
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
 *  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 */

static int ci_match(const char *s, size_t n, const char *lit) {
    size_t i = 0;
    while (lit[i]) {
        if (i >= n) return 0;
        char c = s[i];
        if (c >= 'A' && c <= 'Z') c += (char)32;
        if (c != lit[i]) return 0;
        i++;
    }
    return 1;
}

static size_t next_pretoken(const char *s, size_t n) {
    /* 1. contractions */
    if (n >= 2 && s[0] == '\'') {
        static const char *cs[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (size_t i = 0; i < sizeof cs / sizeof cs[0]; i++) {
            size_t L = strlen(cs[i]);
            if (n - 1 >= L && ci_match(s + 1, n - 1, cs[i])) return 1 + L;
        }
    }
    int l0 = 1, l1 = 0, k1 = -1;
    int k0 = cp_class(u8_dec(s, n, &l0));
    if ((size_t)l0 < n) {
        int ll = 1;
        k1 = cp_class(u8_dec(s + l0, n - (size_t)l0, &ll));
        l1 = ll;
    }
    /* 2. letter run, optional single non-letter prefix (no \r\n) */
    if (k0 == C_L) {
        size_t j = (size_t)l0;
        while (j < n) {
            int ll = 1;
            if (cp_class(u8_dec(s + j, n - j, &ll)) != C_L) break;
            j += (size_t)ll;
        }
        return j;
    }
    if ((k0 == C_SP || k0 == C_P) && k1 == C_L) {
        size_t j = (size_t)l0 + (size_t)l1;
        while (j < n) {
            int ll = 1;
            if (cp_class(u8_dec(s + j, n - j, &ll)) != C_L) break;
            j += (size_t)ll;
        }
        return j;
    }
    /* 3. single number */
    if (k0 == C_N) return (size_t)l0;
    /* 4. punct run, optional leading space, trailing newlines */
    if (k0 == C_P || (k0 == C_SP && k1 == C_P)) {
        size_t j = (k0 == C_SP) ? (size_t)l0 : 0;
        while (j < n) {
            int ll = 1;
            if (cp_class(u8_dec(s + j, n - j, &ll)) != C_P) break;
            j += (size_t)ll;
        }
        while (j < n) {
            int ll = 1;
            if (cp_class(u8_dec(s + j, n - j, &ll)) != C_NL) break;
            j += (size_t)ll;
        }
        return j;
    }
    /* 5. space run: "\s*[\r\n]+" wins when an NL follows, else plain spaces */
    if (k0 == C_SP) {
        size_t j = (size_t)l0, last = 0;
        while (j < n) {
            int ll = 1;
            int k = cp_class(u8_dec(s + j, n - j, &ll));
            if (k != C_SP && k != C_NL) break;
            j += (size_t)ll;
            if (k == C_NL) last = j;
        }
        return last > 0 ? last : j;
    }
    /* 6. newline run: trailing spaces after the last NL stay out */
    if (k0 == C_NL) {
        size_t j = 0, last = 0;
        while (j < n) {
            int ll = 1;
            int k = cp_class(u8_dec(s + j, n - j, &ll));
            if (k != C_SP && k != C_NL) break;
            j += (size_t)ll;
            if (k == C_NL) last = j;
        }
        return last; /* >= l0: first unit is NL */
    }
    return (size_t)l0; /* lone punct safety net */
}

/* ---------------- BPE ---------------- */

typedef struct {
    char    *p;     /* owned */
    uint32_t len;
} sym_t;

static int pair_rank(const cp_tok_t *t, const char *a, size_t la,
                     const char *b, size_t lb, uint32_t *out) {
    size_t kl = la + 1 + lb;
    char *kb = malloc(kl + 1);
    if (!kb) return -1;
    memcpy(kb, a, la);
    kb[la] = '\x01';
    memcpy(kb + la + 1, b, lb);
    kb[kl] = '\0';
    int rc = t_map_get(&t->merges, kb, kl, out);
    free(kb);
    return rc;
}

static void sym_free(sym_t *s, size_t n) {
    for (size_t i = 0; i < n; i++) free(s[i].p);
    free(s);
}

/* encode one pre-token span; appends ids, *pn updated */
static int bpe_encode(const cp_tok_t *t, const char *s, size_t n,
                      uint32_t *out, size_t *pn, size_t cap) {
    if (n == 0) return 0;
    sym_t *sym = malloc((n + 1) * sizeof *sym);
    uint32_t *rk = malloc((n + 1) * sizeof *rk);
    if (!sym || !rk) { free(sym); free(rk); return -1; }

    size_t nsym = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t cp = t->b2u[(unsigned char)s[i]];
        char tmp[4];
        int L;
        if (cp < 0x80) { tmp[0] = (char)cp; L = 1; }
        else if (cp < 0x800) {
            tmp[0] = (char)(0xC0 | (cp >> 6));
            tmp[1] = (char)(0x80 | (cp & 0x3F));
            L = 2;
        } else {
            tmp[0] = (char)(0xE0 | (cp >> 12));
            tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            tmp[2] = (char)(0x80 | (cp & 0x3F));
            L = 3;
        }
        sym[nsym].p = malloc((size_t)L + 1);
        if (!sym[nsym].p) { sym_free(sym, nsym); free(rk); return -1; }
        memcpy(sym[nsym].p, tmp, (size_t)L);
        sym[nsym].p[L] = '\0';
        sym[nsym].len = (uint32_t)L;
        nsym++;
    }
        for (size_t i = 0; i + 1 < nsym; i++) {
        if (pair_rank(t, sym[i].p, sym[i].len,
                      sym[i + 1].p, sym[i + 1].len, &rk[i]) != 0)
            rk[i] = 0xFFFFFFFFu;
    }

    while (nsym >= 2) {
        uint32_t best_rank = 0xFFFFFFFFu;
        size_t best = 0;
        for (size_t i = 0; i + 1 < nsym; i++) {
            if (rk[i] < best_rank) { best_rank = rk[i]; best = i; }
        }
        if (best_rank == 0xFFFFFFFFu) break;
        size_t nl = (size_t)sym[best].len + sym[best + 1].len;
        char *np = malloc(nl + 1);
        if (!np) { sym_free(sym, nsym); free(rk); return -1; }
        memcpy(np, sym[best].p, sym[best].len);
        memcpy(np + sym[best].len, sym[best + 1].p, sym[best + 1].len);
        np[nl] = '\0';
                free(sym[best].p);
        free(sym[best + 1].p);
        sym[best].p = np;
        sym[best].len = (uint32_t)nl;
        memmove(&sym[best + 1], &sym[best + 2],
                (nsym - best - 2) * sizeof *sym);
        nsym--;
        if (nsym >= 2) {
            if (best > 0 &&
                pair_rank(t, sym[best - 1].p, sym[best - 1].len,
                          sym[best].p, sym[best].len, &rk[best - 1]) != 0)
                rk[best - 1] = 0xFFFFFFFFu;
            if (best + 1 < nsym) {
                memmove(&rk[best + 1], &rk[best + 2],
                        (nsym - best - 2) * sizeof *rk);
                if (pair_rank(t, sym[best].p, sym[best].len,
                              sym[best + 1].p, sym[best + 1].len,
                              &rk[best]) != 0)
                    rk[best] = 0xFFFFFFFFu;
            }
        }
    }

    int rc = 0;
    for (size_t i = 0; i < nsym; i++) {
        uint32_t id;
        if (t_map_get(&t->vocab, sym[i].p, sym[i].len, &id) != 0) {
            fprintf(stderr, "tok: piece '%s' missing from vocab\n", sym[i].p);
            rc = -1;
            break;
        }
        if (*pn >= cap) { rc = -1; break; }
        out[(*pn)++] = id;
            }
    sym_free(sym, nsym);
    free(rk);
    return rc;
}

/* ---------------- public encode/decode ---------------- */

int cp_tok_lookup(const cp_tok_t *t, const char *text, uint32_t *out) {
    if (!t || !text || !out) return -1;
    size_t len = strlen(text);
    if (t_map_get(&t->vocab, text, len, out) == 0) return 0;
    for (size_t i = 0; i < t->n_spec; i++) {
        if (t->spec[i].len == len && memcmp(t->spec[i].text, text, len) == 0) {
            *out = t->spec[i].id;
            return 0;
        }
    }
    return -1;
}

int cp_tok_encode(const cp_tok_t *t, const char *text,
                  uint32_t *out, size_t cap) {
    if (!t || !text || (!out && cap > 0)) return -1;
    size_t n = strlen(text), p = 0, count = 0;
    while (p < n) {
        size_t hit = 0;
        uint32_t hid = 0;
        for (size_t i = 0; i < t->n_spec; i++) {
            size_t L = t->spec[i].len;
            if (n - p >= L && memcmp(text + p, t->spec[i].text, L) == 0) {
                hit = L;
                hid = t->spec[i].id;
                break; /* sorted longest-first */
            }
        }
        if (hit > 0) {
            if (count >= cap) return -1;
            out[count++] = hid;
            p += hit;
            continue;
        }
        size_t e = p + next_pretoken(text + p, n - p);
        if (bpe_encode(t, text + p, e - p, out, &count, cap) != 0)
            return -1;
        p = e;
    }
    return (int)count;
}

static int u8_enc(uint32_t cp, char *o) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        o[0] = (char)(0xC0 | (cp >> 6));
        o[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12));
        o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18));
    o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

int cp_tok_decode(const cp_tok_t *t, const uint32_t *ids, size_t n,
                  char *out, size_t cap) {
    if (!t || (!ids && n > 0) || !out || cap == 0) return -1;
    size_t pos = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t id = ids[i];
        if (id >= t->n_ids || !t->str[id]) {
            fprintf(stderr, "tok: unknown id %u\n", id);
            return -1;
        }
        const char *s = t->str[id];
        size_t len = t->slen[id], p = 0;
        while (p < len) {
            int ll = 1;
            uint32_t cp = u8_dec(s + p, len - p, &ll);
            char tmp[4];
            int wl;
            if (cp < 0x144 && t->u2b[cp] != 0xFFFF) {
                tmp[0] = (char)t->u2b[cp];
                wl = 1;
            } else {
                wl = u8_enc(cp, tmp);
            }
            if (pos + (size_t)wl + 1 > cap) return -1;
            memcpy(out + pos, tmp, (size_t)wl);
            pos += (size_t)wl;
            p += (size_t)ll;
        }
    }
    out[pos] = '\0';
    return (int)pos;
}
