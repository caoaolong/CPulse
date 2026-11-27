/*
 * tok/tokenizer.c - strategy resolve + facade over BPE / Unigram backends
 */

#include "tok/tokenizer.h"
#include "tok/bpe.h"
#include "tok/unigram.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char   **items;
    size_t   n;
    size_t   cap;
} str_list_t;

static str_list_t g_bpe;
static str_list_t g_unigram;
static int        g_loaded;
static char       g_path[512];

struct cp_tokenizer {
    cp_tok_strategy_t strategy;
    tok_bpe_t        *bpe;
    tok_unigram_t    *uni;
};

static void str_list_free(str_list_t *L) {
    if (!L) return;
    for (size_t i = 0; i < L->n; i++) free(L->items[i]);
    free(L->items);
    L->items = NULL;
    L->n = L->cap = 0;
}

static int str_list_push(str_list_t *L, char *s) {
    if (L->n == L->cap) {
        size_t ncap = L->cap ? L->cap * 2 : 8;
        char **ni = realloc(L->items, ncap * sizeof *ni);
        if (!ni) return -1;
        L->items = ni;
        L->cap = ncap;
    }
    L->items[L->n++] = s;
    return 0;
}

static int str_list_contains(const str_list_t *L, const char *s) {
    for (size_t i = 0; i < L->n; i++) {
        if (strcmp(L->items[i], s) == 0) return 1;
    }
    return 0;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *d = malloc(n + 1);
    if (!d) return NULL;
    memcpy(d, s, n + 1);
    return d;
}

static void routes_clear(void) {
    str_list_free(&g_bpe);
    str_list_free(&g_unigram);
    g_loaded = 0;
    g_path[0] = '\0';
}

static int fill_list_from_array(const cJSON *arr, str_list_t *L) {
    if (!cJSON_IsArray(arr)) return -1;
    const cJSON *it = NULL;
    cJSON_ArrayForEach(it, arr) {
        if (!cJSON_IsString(it) || !it->valuestring) return -1;
        char *dup = xstrdup(it->valuestring);
        if (!dup || str_list_push(L, dup) != 0) {
            free(dup);
            return -1;
        }
    }
    return 0;
}

static char *read_file(const char *path, size_t *out_n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size <= 0) { fclose(fp); return NULL; }
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[size] = '\0';
    if (out_n) *out_n = (size_t)size;
    return buf;
}

static int file_exists(const char *base, const char *name) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", base, name);
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

static int ensure_routes(void) {
    if (g_loaded) return 0;
    return cp_tok_routes_load(CP_DEFAULT_TOK_ROUTES);
}

int cp_tok_routes_load(const char *path) {
    if (!path || !path[0]) {
        fprintf(stderr, "cp_tok: empty routes path\n");
        return -1;
    }
    routes_clear();

    size_t n = 0;
    char *buf = read_file(path, &n);
    if (!buf) {
        fprintf(stderr, "cp_tok: cannot read routes '%s'\n", path);
        return -1;
    }
    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root || !cJSON_IsObject(root)) {
        fprintf(stderr, "cp_tok: parse error in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }
    /* Prefer sectioned { "tokenizer": { ... } }; fall back to flat root. */
    cJSON *tok = cJSON_GetObjectItemCaseSensitive(root, "tokenizer");
    cJSON *sec = cJSON_IsObject(tok) ? tok : root;
    if (fill_list_from_array(cJSON_GetObjectItemCaseSensitive(sec, "bpe"),
                             &g_bpe) != 0 ||
        fill_list_from_array(cJSON_GetObjectItemCaseSensitive(sec, "unigram"),
                             &g_unigram) != 0) {
        fprintf(stderr,
                "cp_tok: '%s' must define tokenizer.bpe and "
                "tokenizer.unigram as string arrays\n",
                path);
        cJSON_Delete(root);
        routes_clear();
        return -1;
    }
    cJSON_Delete(root);
    snprintf(g_path, sizeof g_path, "%s", path);
    g_loaded = 1;
    return 0;
}

const char *cp_tok_strategy_name(cp_tok_strategy_t id) {
    switch (id) {
    case CP_TOK_BPE:     return "bpe";
    case CP_TOK_UNIGRAM: return "unigram";
    default:             return "unknown";
    }
}

void cp_tok_print_routes(FILE *fp) {
    if (!fp) fp = stderr;
    if (!g_loaded) {
        fprintf(fp, "cp_tok: routes not loaded (expected %s)\n",
                CP_DEFAULT_TOK_ROUTES);
        return;
    }
    fprintf(fp, "cp_tok routes (%s):\n", g_path);
    fprintf(fp, "  bpe:");
    for (size_t i = 0; i < g_bpe.n; i++) fprintf(fp, " %s", g_bpe.items[i]);
    fprintf(fp, "\n  unigram:");
    for (size_t i = 0; i < g_unigram.n; i++)
        fprintf(fp, " %s", g_unigram.items[i]);
    fprintf(fp, "\n");
}

static cp_tok_strategy_t strategy_from_key(const char *key) {
    if (!key || !key[0] || !g_loaded) return CP_TOK_UNKNOWN;
    if (str_list_contains(&g_bpe, key)) return CP_TOK_BPE;
    if (str_list_contains(&g_unigram, key)) return CP_TOK_UNIGRAM;
    return CP_TOK_UNKNOWN;
}

static int peek_tokenizer_class(const char *model_dir, char *dst, size_t cap) {
    char path[512];
    snprintf(path, sizeof path, "%s/tokenizer_config.json", model_dir);
    size_t n = 0;
    char *buf = read_file(path, &n);
    if (!buf) return -1;
    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return -1;
    const cJSON *cls =
            cJSON_GetObjectItemCaseSensitive(root, "tokenizer_class");
    int ok = 0;
    if (cJSON_IsString(cls) && cls->valuestring) {
        snprintf(dst, cap, "%s", cls->valuestring);
        ok = 1;
    }
    cJSON_Delete(root);
    return ok ? 0 : -1;
}

/*
 * Lightweight scan for tokenizer.json "model":{..."type":"BPE"|"Unigram"...}.
 * Avoids full cJSON parse of multi-MB vocab files.
 */
static int peek_tokenizer_json_type(const char *model_dir, char *dst,
                                    size_t cap) {
    char path[512];
    snprintf(path, sizeof path, "%s/tokenizer.json", model_dir);
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    char buf[8192];
    size_t keep = 0;
    int found = 0;
    while (!found) {
        size_t n = fread(buf + keep, 1, sizeof(buf) - keep - 1, fp);
        if (n == 0) break;
        size_t total = keep + n;
        buf[total] = '\0';

        const char *model = strstr(buf, "\"model\"");
        if (model) {
            const char *type = strstr(model, "\"type\"");
            /* type should appear soon after "model" */
            if (type && (size_t)(type - model) < 400) {
                const char *q1 = strchr(type + 6, '"');
                if (q1) {
                    q1++;
                    const char *q2 = strchr(q1, '"');
                    if (q2 && (size_t)(q2 - q1) < cap) {
                        size_t len = (size_t)(q2 - q1);
                        memcpy(dst, q1, len);
                        dst[len] = '\0';
                        found = 1;
                    }
                }
            }
        }
        if (found) break;
        /* overlap window so "model"/"type" spanning chunks still match */
        keep = total > 64 ? 64 : total;
        memmove(buf, buf + total - keep, keep);
    }
    fclose(fp);
    return found ? 0 : -1;
}

cp_tok_strategy_t cp_tok_detect(const char *model_dir) {
    if (!model_dir) return CP_TOK_UNKNOWN;
    if (ensure_routes() != 0) {
        /* still allow file-layout fallback without routes */
    }

    char key[64];
    if (peek_tokenizer_json_type(model_dir, key, sizeof key) == 0) {
        cp_tok_strategy_t s = strategy_from_key(key);
        if (s != CP_TOK_UNKNOWN) return s;
        if (strcmp(key, "BPE") == 0) return CP_TOK_BPE;
        if (strcmp(key, "Unigram") == 0) return CP_TOK_UNIGRAM;
    }

    if (peek_tokenizer_class(model_dir, key, sizeof key) == 0) {
        cp_tok_strategy_t s = strategy_from_key(key);
        if (s != CP_TOK_UNKNOWN) return s;
    }

    int has_bpe_files = file_exists(model_dir, "vocab.json") &&
                        file_exists(model_dir, "merges.txt");
    if (has_bpe_files) return CP_TOK_BPE;
    if (file_exists(model_dir, "tokenizer.json") ||
        file_exists(model_dir, "spiece.model"))
        return CP_TOK_UNIGRAM;

    return CP_TOK_UNKNOWN;
}

int cp_tokenizer_load(cp_tokenizer_t **out, const char *model_dir) {
    if (!out || !model_dir) return -1;
    *out = NULL;

    if (ensure_routes() != 0) {
        fprintf(stderr,
                "cp_tok: warning: could not load %s; using heuristics\n",
                CP_DEFAULT_TOK_ROUTES);
    }

    cp_tok_strategy_t strat = cp_tok_detect(model_dir);
    if (strat == CP_TOK_UNKNOWN) {
        fprintf(stderr, "cp_tok: cannot detect tokenizer strategy in '%s'\n",
                model_dir);
        cp_tok_print_routes(stderr);
        return -1;
    }

    cp_tokenizer_t *t = calloc(1, sizeof *t);
    if (!t) return -1;
    t->strategy = strat;

    int rc = -1;
    if (strat == CP_TOK_BPE) {
        rc = tok_bpe_load(&t->bpe, model_dir);
    } else if (strat == CP_TOK_UNIGRAM) {
        rc = tok_unigram_load(&t->uni, model_dir);
    }
    if (rc != 0) {
        free(t);
        return -1;
    }
    *out = t;
    return 0;
}

void cp_tokenizer_free(cp_tokenizer_t *t) {
    if (!t) return;
    tok_bpe_free(t->bpe);
    tok_unigram_free(t->uni);
    free(t);
}

cp_tok_strategy_t cp_tokenizer_strategy(const cp_tokenizer_t *t) {
    return t ? t->strategy : CP_TOK_UNKNOWN;
}

size_t cp_tokenizer_vocab_size(const cp_tokenizer_t *t) {
    if (!t) return 0;
    if (t->strategy == CP_TOK_BPE) return tok_bpe_vocab_size(t->bpe);
    if (t->strategy == CP_TOK_UNIGRAM) return tok_unigram_vocab_size(t->uni);
    return 0;
}

uint32_t cp_tokenizer_pad_id(const cp_tokenizer_t *t) {
    if (!t || t->strategy != CP_TOK_UNIGRAM) return 0;
    return tok_unigram_pad_id(t->uni);
}

uint32_t cp_tokenizer_eos_id(const cp_tokenizer_t *t) {
    if (!t || t->strategy != CP_TOK_UNIGRAM) return 0;
    return tok_unigram_eos_id(t->uni);
}

uint32_t cp_tokenizer_unk_id(const cp_tokenizer_t *t) {
    if (!t || t->strategy != CP_TOK_UNIGRAM) return 0;
    return tok_unigram_unk_id(t->uni);
}

int cp_tokenizer_encode(const cp_tokenizer_t *t, const char *text,
                        uint32_t *out, size_t cap, int add_eos) {
    if (!t) return -1;
    if (t->strategy == CP_TOK_BPE) {
        (void)add_eos;
        return tok_bpe_encode(t->bpe, text, out, cap);
    }
    if (t->strategy == CP_TOK_UNIGRAM)
        return tok_unigram_encode(t->uni, text, out, cap, add_eos);
    return -1;
}

int cp_tokenizer_lookup(const cp_tokenizer_t *t, const char *text,
                        uint32_t *out) {
    if (!t || t->strategy != CP_TOK_BPE) return -1;
    return tok_bpe_lookup(t->bpe, text, out);
}

int cp_tokenizer_decode(const cp_tokenizer_t *t, const uint32_t *ids, size_t n,
                        char *out, size_t cap, int skip_special) {
    if (!t) return -1;
    if (t->strategy == CP_TOK_BPE) {
        (void)skip_special;
        return tok_bpe_decode(t->bpe, ids, n, out, cap);
    }
    if (t->strategy == CP_TOK_UNIGRAM)
        return tok_unigram_decode(t->uni, ids, n, out, cap, skip_special);
    return -1;
}
