/*
 * arch.c - load arch_routes.json and resolve model_type -> strategy
 */

#include "arch.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char   **items;
    size_t   n;
    size_t   cap;
} str_list_t;

static str_list_t g_encoder_only;
static str_list_t g_decoder_only;
static str_list_t g_encoder_decoder;
static int        g_loaded;
static char       g_path[512];

static const cp_arch_t k_encoder_only = {
    CP_ARCH_ENCODER_ONLY, "encoder-only", "encoder-only", 0
};
static const cp_arch_t k_decoder_only = {
    CP_ARCH_DECODER_ONLY, "decoder-only", "decoder-only", 1
};
/* implemented=1: strategy gate open; runtime only admits T5-family via
 * cp_t5_config_load (see command.c). Other Enc–Dec model_types still fail. */
static const cp_arch_t k_encoder_decoder = {
    CP_ARCH_ENCODER_DECODER, "encoder-decoder", "encoder-decoder", 1
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
    str_list_free(&g_encoder_only);
    str_list_free(&g_decoder_only);
    str_list_free(&g_encoder_decoder);
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

int cp_arch_routes_load(const char *path) {
    if (!path || !path[0]) {
        fprintf(stderr, "cp_arch: empty routes path\n");
        return -1;
    }

    routes_clear();

    char *buf = read_file(path, NULL);
    if (!buf) {
        fprintf(stderr, "cp_arch: cannot read routes file '%s'\n", path);
        return -1;
    }

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root || !cJSON_IsObject(root)) {
        fprintf(stderr, "cp_arch: parse error in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }

    cJSON *enc = cJSON_GetObjectItemCaseSensitive(root, "encoder-only");
    cJSON *dec = cJSON_GetObjectItemCaseSensitive(root, "decoder-only");
    cJSON *ed  = cJSON_GetObjectItemCaseSensitive(root, "encoder-decoder");
    if (fill_list_from_array(enc, &g_encoder_only) != 0 ||
        fill_list_from_array(dec, &g_decoder_only) != 0 ||
        fill_list_from_array(ed, &g_encoder_decoder) != 0) {
        fprintf(stderr,
                "cp_arch: '%s' must define encoder-only, decoder-only, "
                "and encoder-decoder as string arrays\n", path);
        cJSON_Delete(root);
        routes_clear();
        return -1;
    }
    cJSON_Delete(root);

    for (size_t i = 0; i < g_encoder_only.n; i++) {
        const char *t = g_encoder_only.items[i];
        if (str_list_contains(&g_decoder_only, t) ||
            str_list_contains(&g_encoder_decoder, t)) {
            fprintf(stderr,
                    "cp_arch: model_type '%s' appears in multiple strategies "
                    "in '%s'\n", t, path);
            routes_clear();
            return -1;
        }
    }
    for (size_t i = 0; i < g_decoder_only.n; i++) {
        const char *t = g_decoder_only.items[i];
        if (str_list_contains(&g_encoder_decoder, t)) {
            fprintf(stderr,
                    "cp_arch: model_type '%s' appears in multiple strategies "
                    "in '%s'\n", t, path);
            routes_clear();
            return -1;
        }
    }

    strncpy(g_path, path, sizeof g_path - 1);
    g_path[sizeof g_path - 1] = '\0';
    g_loaded = 1;
    return 0;
}

const cp_arch_t *cp_arch_resolve(const char *model_type) {
    if (!g_loaded) {
        fprintf(stderr, "cp_arch: routes not loaded (call cp_arch_routes_load)\n");
        return NULL;
    }
    if (!model_type || !model_type[0]) return NULL;

    if (str_list_contains(&g_decoder_only, model_type))
        return &k_decoder_only;
    if (str_list_contains(&g_encoder_only, model_type))
        return &k_encoder_only;
    if (str_list_contains(&g_encoder_decoder, model_type))
        return &k_encoder_decoder;
    return NULL;
}

const char *cp_arch_name(cp_arch_id_t id) {
    switch (id) {
    case CP_ARCH_ENCODER_ONLY:    return "encoder-only";
    case CP_ARCH_DECODER_ONLY:    return "decoder-only";
    case CP_ARCH_ENCODER_DECODER: return "encoder-decoder";
    case CP_ARCH_UNKNOWN:
    default:                      return "unknown";
    }
}

static void print_list(FILE *fp, const char *title, const str_list_t *L,
                       int implemented) {
    fprintf(fp, "  %s%s:", title, implemented ? " [implemented]" : " [stub]");
    if (L->n == 0) {
        fprintf(fp, " (empty)\n");
        return;
    }
    fprintf(fp, "\n");
    for (size_t i = 0; i < L->n; i++)
        fprintf(fp, "    - %s\n", L->items[i]);
}

void cp_arch_print_registry(FILE *fp) {
    if (!fp) fp = stderr;
    if (!g_loaded) {
        fprintf(fp, "cp_arch: routes not loaded (default file: %s)\n",
                CP_DEFAULT_ARCH_CONFIG);
        return;
    }
    fprintf(fp, "cp_arch: routes from '%s'\n", g_path);
    print_list(fp, "decoder-only", &g_decoder_only, 1);
    print_list(fp, "encoder-only", &g_encoder_only, 0);
    print_list(fp, "encoder-decoder", &g_encoder_decoder, 0);
}
