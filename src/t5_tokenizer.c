/*
 * t5_tokenizer.c - T5 Unigram tokenizer (see t5_tokenizer.h)
 *
 * Shared encode path for both backends:
 *   1) space -> U+2581 (▁); optional dummy prefix ▁
 *   2) Unigram Viterbi over UTF-8 (SentencePiece EncodeOptimized style)
 *   3) optional eos append
 *
 * JSON backend reads model.vocab from tokenizer.json.
 * SPIECE backend parses SentencePiece ModelProto pieces from spiece.model.
 */

#include "t5_tokenizer.h"
#include "cJSON.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T5_META_SPACE "\xE2\x96\x81" /* U+2581 ▁ in UTF-8 */
#define T5_META_SPACE_LEN 3
#define T5_UNK_PENALTY 10.0f

/* ---------------- open-addressing string -> (id, score) ---------------- */

typedef struct {
    char    *key;
    uint32_t klen;
    uint32_t id;
    float    score;
    uint8_t  used;
} u_ent_t;

typedef struct {
    u_ent_t *tab;
    size_t   cap;
    size_t   n;
} u_map_t;

static uint64_t u_hash(const char *s, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int u_map_init(u_map_t *m, size_t want) {
    size_t cap = 16;
    while (cap < want) cap *= 2;
    m->tab = calloc(cap, sizeof *m->tab);
    if (!m->tab) return -1;
    m->cap = cap;
    m->n = 0;
    return 0;
}

static int u_map_grow(u_map_t *m) {
    u_map_t o = *m;
    m->tab = calloc(o.cap * 2, sizeof *m->tab);
    if (!m->tab) { m->tab = o.tab; return -1; }
    m->cap = o.cap * 2;
    m->n = 0;
    for (size_t i = 0; i < o.cap; i++) {
        if (!o.tab[i].used) continue;
        uint64_t h = u_hash(o.tab[i].key, o.tab[i].klen);
        size_t j = (size_t)h & (m->cap - 1);
        while (m->tab[j].used) j = (j + 1) & (m->cap - 1);
        m->tab[j] = o.tab[i];
        m->n++;
    }
    free(o.tab);
    return 0;
}

static int u_map_put(u_map_t *m, const char *k, size_t klen,
                     uint32_t id, float score) {
    if ((m->n + 1) * 4 > m->cap * 3 && u_map_grow(m) != 0) return -1;
    uint64_t h = u_hash(k, klen);
    size_t i = (size_t)h & (m->cap - 1);
    while (m->tab[i].used) {
        if (m->tab[i].klen == klen && memcmp(m->tab[i].key, k, klen) == 0) {
            m->tab[i].id = id;
            m->tab[i].score = score;
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
    m->tab[i].id = id;
    m->tab[i].score = score;
    m->tab[i].used = 1;
    m->n++;
    return 0;
}

static int u_map_get(const u_map_t *m, const char *k, size_t klen,
                     uint32_t *id, float *score) {
    if (m->cap == 0) return -1;
    uint64_t h = u_hash(k, klen);
    size_t i = (size_t)h & (m->cap - 1);
    while (m->tab[i].used) {
        if (m->tab[i].klen == klen && memcmp(m->tab[i].key, k, klen) == 0) {
            if (id) *id = m->tab[i].id;
            if (score) *score = m->tab[i].score;
            return 0;
        }
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

static void u_map_free(u_map_t *m) {
    if (!m || !m->tab) return;
    for (size_t i = 0; i < m->cap; i++) free(m->tab[i].key);
    free(m->tab);
    m->tab = NULL;
    m->cap = m->n = 0;
}

/* ---------------- tokenizer state ---------------- */

struct cp_t5_tok {
    cp_t5_tok_backend_t backend;
    u_map_t   piece_to_id; /* piece bytes -> id+score */
    char    **id_to_piece; /* owned strings; may be NULL slots */
    size_t    n_ids;
    uint32_t  pad_id;
    uint32_t  eos_id;
    uint32_t  unk_id;
    float     min_score;
    size_t    max_piece_len;
    int       add_dummy_prefix; /* 1 = prepend ▁ like SPM default */
};

/* ---------------- helpers ---------------- */

static int read_file(const char *path, char **out, size_t *out_n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "t5_tok: cannot open '%s'\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0) { fclose(fp); return -1; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return -1; }
    if (sz > 0 && fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        fprintf(stderr, "t5_tok: short read on '%s'\n", path);
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

static size_t utf8_char_len(const unsigned char *s, size_t n) {
    if (n == 0) return 0;
    unsigned char c = s[0];
    size_t need;
    if (c < 0x80) need = 1;
    else if ((c & 0xE0) == 0xC0) need = 2;
    else if ((c & 0xF0) == 0xE0) need = 3;
    else if ((c & 0xF8) == 0xF0) need = 4;
    else return 1; /* invalid: consume one byte */
    if (need > n) return 1;
    for (size_t i = 1; i < need; i++) {
        if ((s[i] & 0xC0) != 0x80) return 1;
    }
    return need;
}

static int ensure_id_slot(cp_t5_tok_t *t, uint32_t id) {
    if (id < t->n_ids) return 0;
    size_t nn = (size_t)id + 1;
    char **np = realloc(t->id_to_piece, nn * sizeof *np);
    if (!np) return -1;
    for (size_t i = t->n_ids; i < nn; i++) np[i] = NULL;
    t->id_to_piece = np;
    t->n_ids = nn;
    return 0;
}

static int add_piece(cp_t5_tok_t *t, const char *piece, size_t plen,
                     uint32_t id, float score) {
    if (plen == 0) return 0;
    if (u_map_put(&t->piece_to_id, piece, plen, id, score) != 0) return -1;
    if (ensure_id_slot(t, id) != 0) return -1;
    if (!t->id_to_piece[id]) {
        char *cp = malloc(plen + 1);
        if (!cp) return -1;
        memcpy(cp, piece, plen);
        cp[plen] = '\0';
        t->id_to_piece[id] = cp;
    }
    if (score < t->min_score) t->min_score = score;
    if (plen > t->max_piece_len) t->max_piece_len = plen;
    return 0;
}

/* ---------------- JSON backend ---------------- */

static int load_from_json(cp_t5_tok_t *t, const char *path) {
    char *buf = NULL;
    size_t n = 0;
    if (read_file(path, &buf, &n) != 0) return -1;

    cJSON *root = cJSON_ParseWithLength(buf, n);
    free(buf);
    if (!root || !cJSON_IsObject(root)) {
        fprintf(stderr, "t5_tok: parse error in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }

    const cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");
    const cJSON *vocab = model ? cJSON_GetObjectItemCaseSensitive(model, "vocab")
                               : NULL;
    if (!cJSON_IsArray(vocab)) {
        fprintf(stderr, "t5_tok: missing model.vocab array in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }

    size_t nv = (size_t)cJSON_GetArraySize(vocab);
    if (u_map_init(&t->piece_to_id, nv * 2 + 16) != 0) {
        cJSON_Delete(root);
        return -1;
    }
    t->min_score = FLT_MAX;

    uint32_t id = 0;
    const cJSON *it = NULL;
    cJSON_ArrayForEach(it, vocab) {
        if (!cJSON_IsArray(it) || cJSON_GetArraySize(it) < 2) {
            fprintf(stderr, "t5_tok: bad vocab entry at id %u\n", id);
            cJSON_Delete(root);
            return -1;
        }
        const cJSON *ps = cJSON_GetArrayItem(it, 0);
        const cJSON *sc = cJSON_GetArrayItem(it, 1);
        if (!cJSON_IsString(ps) || !ps->valuestring || !cJSON_IsNumber(sc)) {
            fprintf(stderr, "t5_tok: bad vocab entry at id %u\n", id);
            cJSON_Delete(root);
            return -1;
        }
        if (add_piece(t, ps->valuestring, strlen(ps->valuestring),
                      id, (float)sc->valuedouble) != 0) {
            fprintf(stderr, "t5_tok: out of memory loading vocab\n");
            cJSON_Delete(root);
            return -1;
        }
        id++;
    }

    /* Prefer added_tokens for pad/eos/unk when present. */
    const cJSON *added = cJSON_GetObjectItemCaseSensitive(root, "added_tokens");
    if (cJSON_IsArray(added)) {
        const cJSON *a = NULL;
        cJSON_ArrayForEach(a, added) {
            const cJSON *content = cJSON_GetObjectItemCaseSensitive(a, "content");
            const cJSON *aid = cJSON_GetObjectItemCaseSensitive(a, "id");
            if (!cJSON_IsString(content) || !content->valuestring ||
                !cJSON_IsNumber(aid))
                continue;
            const char *c = content->valuestring;
            uint32_t tid = (uint32_t)aid->valuedouble;
            if (strcmp(c, "<pad>") == 0) t->pad_id = tid;
            else if (strcmp(c, "</s>") == 0) t->eos_id = tid;
            else if (strcmp(c, "<unk>") == 0) t->unk_id = tid;
        }
    }

    const cJSON *unk = model
                           ? cJSON_GetObjectItemCaseSensitive(model, "unk_id")
                           : NULL;
    if (cJSON_IsNumber(unk)) t->unk_id = (uint32_t)unk->valuedouble;

    cJSON_Delete(root);
    if (t->n_ids == 0) {
        fprintf(stderr, "t5_tok: empty vocab in '%s'\n", path);
        return -1;
    }
    if (t->min_score == FLT_MAX) t->min_score = 0.0f;
    return 0;
}

/* ---------------- SPIECE protobuf backend ---------------- */

static int pb_read_varint(const uint8_t **pp, const uint8_t *end,
                          uint64_t *out) {
    const uint8_t *p = *pp;
    uint64_t v = 0;
    int shift = 0;
    while (p < end && shift < 64) {
        uint8_t b = *p++;
        v |= (uint64_t)(b & 0x7f) << shift;
        if ((b & 0x80) == 0) {
            *pp = p;
            *out = v;
            return 0;
        }
        shift += 7;
    }
    return -1;
}

static int pb_skip(const uint8_t **pp, const uint8_t *end, uint32_t wt) {
    const uint8_t *p = *pp;
    if (wt == 0) { /* varint */
        uint64_t v;
        if (pb_read_varint(&p, end, &v) != 0) return -1;
    } else if (wt == 1) { /* 64-bit */
        if (p + 8 > end) return -1;
        p += 8;
    } else if (wt == 2) { /* length-delimited */
        uint64_t len;
        if (pb_read_varint(&p, end, &len) != 0) return -1;
        if (p + len > end) return -1;
        p += (size_t)len;
    } else if (wt == 5) { /* 32-bit */
        if (p + 4 > end) return -1;
        p += 4;
    } else {
        return -1;
    }
    *pp = p;
    return 0;
}

static int pb_parse_piece(cp_t5_tok_t *t, const uint8_t *p, const uint8_t *end,
                          uint32_t id) {
    const char *piece = NULL;
    size_t plen = 0;
    float score = 0.0f;
    int have_score = 0;

    while (p < end) {
        uint64_t tag;
        if (pb_read_varint(&p, end, &tag) != 0) return -1;
        uint32_t field = (uint32_t)(tag >> 3);
        uint32_t wt = (uint32_t)(tag & 7);
        if (field == 1 && wt == 2) { /* piece string */
            uint64_t len;
            if (pb_read_varint(&p, end, &len) != 0) return -1;
            if (p + len > end) return -1;
            piece = (const char *)p;
            plen = (size_t)len;
            p += (size_t)len;
        } else if (field == 2 && wt == 5) { /* score float */
            if (p + 4 > end) return -1;
            memcpy(&score, p, 4);
            have_score = 1;
            p += 4;
        } else if (field == 3 && wt == 0) { /* type enum — ignore */
            uint64_t v;
            if (pb_read_varint(&p, end, &v) != 0) return -1;
            (void)v;
        } else {
            if (pb_skip(&p, end, wt) != 0) return -1;
        }
    }

    if (!piece || plen == 0) return 0; /* empty piece: skip */
    if (!have_score) score = 0.0f;
    return add_piece(t, piece, plen, id, score);
}

static int load_from_spiece(cp_t5_tok_t *t, const char *path) {
    char *buf = NULL;
    size_t n = 0;
    if (read_file(path, &buf, &n) != 0) return -1;

    /* First pass: count pieces for map sizing. */
    size_t n_pieces = 0;
    {
        const uint8_t *p = (const uint8_t *)buf;
        const uint8_t *end = p + n;
        while (p < end) {
            uint64_t tag;
            if (pb_read_varint(&p, end, &tag) != 0) break;
            uint32_t field = (uint32_t)(tag >> 3);
            uint32_t wt = (uint32_t)(tag & 7);
            if (field == 1 && wt == 2) {
                uint64_t len;
                if (pb_read_varint(&p, end, &len) != 0) break;
                if (p + len > end) break;
                p += (size_t)len;
                n_pieces++;
            } else {
                if (pb_skip(&p, end, wt) != 0) break;
            }
        }
    }

    if (n_pieces == 0) {
        fprintf(stderr, "t5_tok: no pieces in '%s'\n", path);
        free(buf);
        return -1;
    }
    if (u_map_init(&t->piece_to_id, n_pieces * 2 + 16) != 0) {
        free(buf);
        return -1;
    }
    t->min_score = FLT_MAX;

    const uint8_t *p = (const uint8_t *)buf;
    const uint8_t *end = p + n;
    uint32_t id = 0;
    while (p < end) {
        uint64_t tag;
        if (pb_read_varint(&p, end, &tag) != 0) {
            fprintf(stderr, "t5_tok: protobuf parse error in '%s'\n", path);
            free(buf);
            return -1;
        }
        uint32_t field = (uint32_t)(tag >> 3);
        uint32_t wt = (uint32_t)(tag & 7);
        if (field == 1 && wt == 2) {
            uint64_t len;
            if (pb_read_varint(&p, end, &len) != 0 || p + len > end) {
                fprintf(stderr, "t5_tok: protobuf parse error in '%s'\n", path);
                free(buf);
                return -1;
            }
            if (pb_parse_piece(t, p, p + (size_t)len, id) != 0) {
                fprintf(stderr, "t5_tok: failed to load piece %u\n", id);
                free(buf);
                return -1;
            }
            p += (size_t)len;
            id++;
        } else {
            if (pb_skip(&p, end, wt) != 0) {
                fprintf(stderr, "t5_tok: protobuf parse error in '%s'\n", path);
                free(buf);
                return -1;
            }
        }
    }
    free(buf);

    /* Conventional T5 / SPM special ids. */
    t->pad_id = 0;
    t->eos_id = 1;
    t->unk_id = 2;
    if (t->min_score == FLT_MAX) t->min_score = 0.0f;
    return 0;
}

/*
 * HF T5Tokenizer appends <extra_id_{N-1}> .. <extra_id_0> after the SPM
 * pieces (default N=100). tokenizer.json already embeds them; for the
 * spiece backend we mirror that from tokenizer_config.json when present.
 */
static int append_extra_ids(cp_t5_tok_t *t, const char *model_dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/tokenizer_config.json", model_dir);

    int extra_ids = 100;
    char *buf = NULL;
    size_t n = 0;
    if (read_file(path, &buf, &n) == 0) {
        cJSON *root = cJSON_ParseWithLength(buf, n);
        free(buf);
        if (root && cJSON_IsObject(root)) {
            const cJSON *ex =
                cJSON_GetObjectItemCaseSensitive(root, "extra_ids");
            if (cJSON_IsNumber(ex) && ex->valuedouble >= 0)
                extra_ids = (int)ex->valuedouble;
        }
        cJSON_Delete(root);
    }

    uint32_t next_id = (uint32_t)t->n_ids;
    for (int i = 0; i < extra_ids; i++) {
        char piece[32];
        /* HF order: extra_id_{N-1} first, then down to extra_id_0. */
        snprintf(piece, sizeof piece, "<extra_id_%d>", extra_ids - 1 - i);
        if (add_piece(t, piece, strlen(piece), next_id, 0.0f) != 0)
            return -1;
        next_id++;
    }
    return 0;
}

/* ---------------- encode / decode ---------------- */

/*
 * Build SPM-style normalized surface: spaces -> ▁, optional leading ▁.
 * Caller frees *out.
 */
static int spm_surface(const char *text, int add_dummy_prefix,
                       char **out, size_t *out_n) {
    size_t n = strlen(text);
    /* Worst case: every byte is space -> 3-byte ▁, plus one prefix ▁. */
    size_t cap = n * T5_META_SPACE_LEN + T5_META_SPACE_LEN + 1;
    char *buf = malloc(cap);
    if (!buf) return -1;

    size_t o = 0;
    if (add_dummy_prefix) {
        memcpy(buf + o, T5_META_SPACE, T5_META_SPACE_LEN);
        o += T5_META_SPACE_LEN;
    }
    for (size_t i = 0; i < n; i++) {
        if (text[i] == ' ') {
            memcpy(buf + o, T5_META_SPACE, T5_META_SPACE_LEN);
            o += T5_META_SPACE_LEN;
        } else {
            buf[o++] = text[i];
        }
    }
    /* If text already started with ▁ and we added dummy, SPM still adds
     * one dummy prefix only when the raw text does not begin with space
     * that becomes ▁ — matching SPM: always prepend when
     * add_dummy_prefix, even if result doubles. HF SPM uses:
     *   if add_dummy_prefix: s = ▁ + s (after space substitution in
     *   normalize). Our space->▁ then prefix matches that order when
     *   original has no leading space. Leading space becomes ▁ then
     *   another ▁ prefix — same as SPM. */
    buf[o] = '\0';
    *out = buf;
    *out_n = o;
    return 0;
}

static int unigram_encode(const cp_t5_tok_t *t, const char *s, size_t n,
                          uint32_t *out, size_t cap, size_t *out_n) {
    if (n == 0) {
        *out_n = 0;
        return 0;
    }

    float *best = malloc((n + 1) * sizeof *best);
    int *prev = malloc((n + 1) * sizeof *prev);
    uint32_t *tok_at = malloc((n + 1) * sizeof *tok_at);
    if (!best || !prev || !tok_at) {
        free(best); free(prev); free(tok_at);
        return -1;
    }

    for (size_t i = 0; i <= n; i++) {
        best[i] = -FLT_MAX;
        prev[i] = -1;
        tok_at[i] = t->unk_id;
    }
    best[0] = 0.0f;

    for (size_t pos = 0; pos < n; pos++) {
        if (best[pos] == -FLT_MAX) continue;
        int found = 0;
        size_t max_end = pos + t->max_piece_len;
        if (max_end > n) max_end = n;
        for (size_t end = pos + 1; end <= max_end; end++) {
            uint32_t id;
            float score;
            if (u_map_get(&t->piece_to_id, s + pos, end - pos, &id, &score) != 0)
                continue;
            found = 1;
            float cand = best[pos] + score;
            if (cand > best[end]) {
                best[end] = cand;
                prev[end] = (int)pos;
                tok_at[end] = id;
            }
        }
        if (!found) {
            size_t cl = utf8_char_len((const unsigned char *)s + pos, n - pos);
            size_t end = pos + cl;
            float score = t->min_score - T5_UNK_PENALTY;
            float cand = best[pos] + score;
            if (cand > best[end]) {
                best[end] = cand;
                prev[end] = (int)pos;
                tok_at[end] = t->unk_id;
            }
        }
    }

    if (best[n] == -FLT_MAX) {
        free(best); free(prev); free(tok_at);
        fprintf(stderr, "t5_tok: failed to encode (no Viterbi path)\n");
        return -1;
    }

    /* Collect then reverse. */
    size_t tmp_cap = n + 1;
    uint32_t *tmp = malloc(tmp_cap * sizeof *tmp);
    if (!tmp) {
        free(best); free(prev); free(tok_at);
        return -1;
    }
    size_t tn = 0;
    for (int cur = (int)n; cur > 0; ) {
        int p = prev[cur];
        if (p < 0) {
            free(tmp); free(best); free(prev); free(tok_at);
            fprintf(stderr, "t5_tok: broken Viterbi backtrace\n");
            return -1;
        }
        if (tn >= tmp_cap) {
            free(tmp); free(best); free(prev); free(tok_at);
            return -1;
        }
        tmp[tn++] = tok_at[cur];
        cur = p;
    }

    if (tn > cap) {
        free(tmp); free(best); free(prev); free(tok_at);
        fprintf(stderr, "t5_tok: output buffer too small (need %zu)\n", tn);
        return -1;
    }
    for (size_t i = 0; i < tn; i++) out[i] = tmp[tn - 1 - i];
    *out_n = tn;

    free(tmp); free(best); free(prev); free(tok_at);
    return 0;
}

/* ---------------- public API ---------------- */

const char *cp_t5_tok_backend_name(cp_t5_tok_backend_t b) {
    switch (b) {
    case CP_T5_TOK_JSON: return "json";
    case CP_T5_TOK_SPIECE: return "spiece";
    default: return "unknown";
    }
}

int cp_t5_tok_backend_parse(const char *s, cp_t5_tok_backend_t *out) {
    if (!s || !out) return -1;
    if (strcmp(s, "json") == 0 || strcmp(s, "tokenizer.json") == 0) {
        *out = CP_T5_TOK_JSON;
        return 0;
    }
    if (strcmp(s, "spiece") == 0 || strcmp(s, "spiece.model") == 0 ||
        strcmp(s, "spm") == 0) {
        *out = CP_T5_TOK_SPIECE;
        return 0;
    }
    return -1;
}

void cp_t5_tok_free(cp_t5_tok_t *t) {
    if (!t) return;
    u_map_free(&t->piece_to_id);
    if (t->id_to_piece) {
        for (size_t i = 0; i < t->n_ids; i++) free(t->id_to_piece[i]);
        free(t->id_to_piece);
    }
    free(t);
}

int cp_t5_tok_load(cp_t5_tok_t **out, const char *model_dir,
                   cp_t5_tok_backend_t backend) {
    if (!out || !model_dir) return -1;
    *out = NULL;

    cp_t5_tok_t *t = calloc(1, sizeof *t);
    if (!t) return -1;
    t->backend = backend;
    t->pad_id = 0;
    t->eos_id = 1;
    t->unk_id = 2;
    t->add_dummy_prefix = 1;
    t->min_score = 0.0f;

    char path[512];
    int rc;
    if (backend == CP_T5_TOK_JSON) {
        snprintf(path, sizeof path, "%s/tokenizer.json", model_dir);
        rc = load_from_json(t, path);
    } else if (backend == CP_T5_TOK_SPIECE) {
        snprintf(path, sizeof path, "%s/spiece.model", model_dir);
        rc = load_from_spiece(t, path);
        if (rc == 0) rc = append_extra_ids(t, model_dir);
    } else {
        fprintf(stderr, "t5_tok: unknown backend %d\n", (int)backend);
        rc = -1;
    }

    if (rc != 0) {
        cp_t5_tok_free(t);
        return -1;
    }
    *out = t;
    return 0;
}

cp_t5_tok_backend_t cp_t5_tok_backend(const cp_t5_tok_t *t) {
    return t ? t->backend : CP_T5_TOK_JSON;
}
uint32_t cp_t5_tok_pad_id(const cp_t5_tok_t *t) { return t ? t->pad_id : 0; }
uint32_t cp_t5_tok_eos_id(const cp_t5_tok_t *t) { return t ? t->eos_id : 1; }
uint32_t cp_t5_tok_unk_id(const cp_t5_tok_t *t) { return t ? t->unk_id : 2; }
size_t   cp_t5_tok_vocab_size(const cp_t5_tok_t *t) { return t ? t->n_ids : 0; }

int cp_t5_tok_encode(const cp_t5_tok_t *t, const char *text,
                     uint32_t *out, size_t cap, int add_eos) {
    if (!t || !text || !out) return -1;

    char *surf = NULL;
    size_t sn = 0;
    if (spm_surface(text, t->add_dummy_prefix, &surf, &sn) != 0) return -1;

    size_t n = 0;
    int rc = unigram_encode(t, surf, sn, out, cap, &n);
    free(surf);
    if (rc != 0) return -1;

    if (add_eos) {
        if (n >= cap) {
            fprintf(stderr, "t5_tok: no room for eos\n");
            return -1;
        }
        out[n++] = t->eos_id;
    }
    return (int)n;
}

int cp_t5_tok_decode(const cp_t5_tok_t *t, const uint32_t *ids, size_t n,
                     char *out, size_t cap, int skip_special) {
    if (!t || !out || cap == 0) return -1;
    if (n > 0 && !ids) return -1;

    size_t o = 0;
    out[0] = '\0';
    for (size_t i = 0; i < n; i++) {
        uint32_t id = ids[i];
        if (skip_special &&
            (id == t->pad_id || id == t->eos_id || id == t->unk_id))
            continue;
        if (id >= t->n_ids || !t->id_to_piece[id]) {
            fprintf(stderr, "t5_tok: unknown id %u\n", id);
            return -1;
        }
        const char *p = t->id_to_piece[id];
        size_t plen = strlen(p);
        /* Replace ▁ with ASCII space in the piece. */
        for (size_t j = 0; j < plen; ) {
            if (j + T5_META_SPACE_LEN <= plen &&
                memcmp(p + j, T5_META_SPACE, T5_META_SPACE_LEN) == 0) {
                if (o + 1 >= cap) return -1;
                out[o++] = ' ';
                j += T5_META_SPACE_LEN;
            } else {
                if (o + 1 >= cap) return -1;
                out[o++] = p[j++];
            }
        }
    }
    /* Strip one leading space from dummy prefix. */
    if (o > 0 && out[0] == ' ') {
        memmove(out, out + 1, o - 1);
        o--;
    }
    if (o >= cap) return -1;
    out[o] = '\0';
    return (int)o;
}
