/*
 * bind_recipe.c - load / resolve bind recipes (see bind_recipe.h)
 */

#include "bind_recipe.h"
#include "arch.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_routes_path[512] = CP_DEFAULT_ROUTES;

static char *read_file(const char *path) {
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
    return buf;
}

static void copy_cstr(char *dst, size_t cap, const char *src) {
    if (!src || cap == 0) return;
    strncpy(dst, src, cap - 1);
    dst[cap - 1] = '\0';
}

void cp_bind_set_routes_path(const char *path) {
    if (!path || !path[0]) {
        copy_cstr(g_routes_path, sizeof g_routes_path, CP_DEFAULT_ROUTES);
        return;
    }
    copy_cstr(g_routes_path, sizeof g_routes_path, path);
}

static void free_tensor_specs(cp_bind_tensor_spec_t *t, size_t n) {
    (void)n;
    free(t);
}

static void free_layer_group(cp_bind_layer_group_t *g) {
    free_tensor_specs(g->tensors, g->n_tensors);
    g->tensors = NULL;
    g->n_tensors = 0;
}

void cp_bind_recipe_free(cp_bind_recipe_t *r) {
    if (!r) return;
    free_tensor_specs(r->globals, r->n_globals);
    free_layer_group(&r->layers);
    free_layer_group(&r->encoder_layers);
    free_layer_group(&r->decoder_layers);
    memset(r, 0, sizeof(*r));
}

static int parse_tensor_spec(const cJSON *obj, cp_bind_tensor_spec_t *t) {
    memset(t, 0, sizeof(*t));
    t->only_layer = -1;
    t->if_gated = 0;

    if (!cJSON_IsObject(obj)) return -1;
    const cJSON *name = cJSON_GetObjectItemCaseSensitive(obj, "name");
    const cJSON *field = cJSON_GetObjectItemCaseSensitive(obj, "field");
    const cJSON *shape = cJSON_GetObjectItemCaseSensitive(obj, "shape");
    if (!cJSON_IsString(name) || !name->valuestring ||
        !cJSON_IsString(field) || !field->valuestring ||
        !cJSON_IsArray(shape))
        return -1;

    copy_cstr(t->name, sizeof t->name, name->valuestring);
    copy_cstr(t->field, sizeof t->field, field->valuestring);

    int ndim = 0;
    const cJSON *dim = NULL;
    cJSON_ArrayForEach(dim, shape) {
        if (!cJSON_IsString(dim) || !dim->valuestring) return -1;
        if (ndim >= CP_BIND_MAX_DIM) return -1;
        copy_cstr(t->shape_sym[ndim], sizeof t->shape_sym[ndim],
                  dim->valuestring);
        ndim++;
    }
    t->ndim = ndim;

    const cJSON *only = cJSON_GetObjectItemCaseSensitive(obj, "only_layer");
    if (cJSON_IsNumber(only)) t->only_layer = (int)only->valuedouble;

    const cJSON *not_tied =
            cJSON_GetObjectItemCaseSensitive(obj, "if_not_tied");
    if (cJSON_IsTrue(not_tied)) t->if_not_tied = 1;

    const cJSON *gated = cJSON_GetObjectItemCaseSensitive(obj, "if_gated");
    if (cJSON_IsTrue(gated)) t->if_gated = 1;
    else if (cJSON_IsFalse(gated)) t->if_gated = -1;

    return 0;
}

static int parse_tensor_array(const cJSON *arr, cp_bind_tensor_spec_t **out,
                              size_t *n_out) {
    *out = NULL;
    *n_out = 0;
    if (!cJSON_IsArray(arr)) return -1;
    size_t n = (size_t)cJSON_GetArraySize(arr);
    if (n == 0) return 0;
    cp_bind_tensor_spec_t *specs = calloc(n, sizeof *specs);
    if (!specs) return -1;
    size_t i = 0;
    const cJSON *it = NULL;
    cJSON_ArrayForEach(it, arr) {
        if (parse_tensor_spec(it, &specs[i]) != 0) {
            free(specs);
            return -1;
        }
        i++;
    }
    *out = specs;
    *n_out = n;
    return 0;
}

static int parse_layer_group(const cJSON *obj, cp_bind_layer_group_t *g) {
    memset(g, 0, sizeof(*g));
    if (!obj) return 0;
    if (!cJSON_IsObject(obj)) return -1;

    const cJSON *count = cJSON_GetObjectItemCaseSensitive(obj, "count");
    const cJSON *prefix = cJSON_GetObjectItemCaseSensitive(obj, "prefix");
    const cJSON *tensors = cJSON_GetObjectItemCaseSensitive(obj, "tensors");
    if (!cJSON_IsString(count) || !count->valuestring ||
        !cJSON_IsString(prefix) || !prefix->valuestring)
        return -1;

    copy_cstr(g->count_sym, sizeof g->count_sym, count->valuestring);
    copy_cstr(g->prefix, sizeof g->prefix, prefix->valuestring);
    return parse_tensor_array(tensors, &g->tensors, &g->n_tensors);
}

static int parse_recipe_object(const cJSON *obj, const char *key,
                               cp_bind_recipe_t *out) {
    memset(out, 0, sizeof(*out));
    copy_cstr(out->key, sizeof out->key, key);

    const cJSON *dtype = cJSON_GetObjectItemCaseSensitive(obj, "dtype");
    if (cJSON_IsString(dtype) && dtype->valuestring)
        copy_cstr(out->dtype_name, sizeof out->dtype_name, dtype->valuestring);

    const cJSON *dtype_from =
            cJSON_GetObjectItemCaseSensitive(obj, "dtype_from");
    if (cJSON_IsString(dtype_from) && dtype_from->valuestring)
        copy_cstr(out->dtype_from, sizeof out->dtype_from,
                  dtype_from->valuestring);

    const cJSON *tied =
            cJSON_GetObjectItemCaseSensitive(obj, "tied_head_from_config");
    if (cJSON_IsTrue(tied)) out->tied_head_from_config = 1;

    if (parse_tensor_array(cJSON_GetObjectItemCaseSensitive(obj, "globals"),
                           &out->globals, &out->n_globals) != 0)
        return -1;
    if (parse_layer_group(cJSON_GetObjectItemCaseSensitive(obj, "layers"),
                          &out->layers) != 0)
        return -1;
    if (parse_layer_group(
                cJSON_GetObjectItemCaseSensitive(obj, "encoder_layers"),
                &out->encoder_layers) != 0)
        return -1;
    if (parse_layer_group(
                cJSON_GetObjectItemCaseSensitive(obj, "decoder_layers"),
                &out->decoder_layers) != 0)
        return -1;
    return 0;
}

int cp_bind_recipe_load(const char *routes_path, const char *model_type,
                        cp_bind_recipe_t *out) {
    memset(out, 0, sizeof(*out));
    if (!model_type || !model_type[0]) {
        fprintf(stderr, "cp_bind: empty model_type\n");
        return -1;
    }
    const char *path = (routes_path && routes_path[0]) ? routes_path
                                                       : g_routes_path;

    char *buf = read_file(path);
    if (!buf) {
        fprintf(stderr, "cp_bind: cannot read '%s'\n", path);
        return -1;
    }
    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root || !cJSON_IsObject(root)) {
        fprintf(stderr, "cp_bind: parse error in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }

    cJSON *bind = cJSON_GetObjectItemCaseSensitive(root, "bind");
    if (!cJSON_IsObject(bind)) {
        fprintf(stderr, "cp_bind: missing 'bind' section in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }

    const char *key = model_type;
    char alias_buf[32];
    /* Follow string aliases a few times (mt5 -> t5). */
    for (int hop = 0; hop < 4; hop++) {
        cJSON *node = cJSON_GetObjectItemCaseSensitive(bind, key);
        if (!node) {
            fprintf(stderr, "cp_bind: no recipe for model_type '%s' in '%s'\n",
                    model_type, path);
            cJSON_Delete(root);
            return -1;
        }
        if (cJSON_IsString(node) && node->valuestring) {
            copy_cstr(alias_buf, sizeof alias_buf, node->valuestring);
            key = alias_buf;
            continue;
        }
        if (!cJSON_IsObject(node)) {
            fprintf(stderr, "cp_bind: bind.%s must be object or alias string\n",
                    key);
            cJSON_Delete(root);
            return -1;
        }
        int rc = parse_recipe_object(node, key, out);
        cJSON_Delete(root);
        if (rc != 0) {
            fprintf(stderr, "cp_bind: invalid recipe for '%s' in '%s'\n", key,
                    path);
            cp_bind_recipe_free(out);
            return -1;
        }
        return 0;
    }

    fprintf(stderr, "cp_bind: alias loop resolving '%s'\n", model_type);
    cJSON_Delete(root);
    return -1;
}

int cp_bind_resolve_sym(const cp_bind_dims_t *dims, const char *sym,
                        size_t *out) {
    if (!dims || !sym || !out) return -1;
#define HIT(name, val) \
    if (strcmp(sym, name) == 0) { *out = (size_t)(val); return 0; }
    HIT("vocab_size", dims->vocab_size);
    HIT("hidden_size", dims->hidden_size);
    HIT("intermediate_size", dims->intermediate_size);
    HIT("kv_dim", dims->kv_dim);
    HIT("num_hidden_layers", dims->num_hidden_layers);
    HIT("d_model", dims->d_model);
    HIT("d_ff", dims->d_ff);
    HIT("inner_dim", dims->inner_dim);
    HIT("num_heads", dims->num_heads);
    HIT("num_layers", dims->num_layers);
    HIT("num_decoder_layers", dims->num_decoder_layers);
    HIT("rel_buckets", dims->rel_buckets);
#undef HIT
    fprintf(stderr, "cp_bind: unknown shape symbol '%s'\n", sym);
    return -1;
}

int cp_bind_resolve_shape(const cp_bind_dims_t *dims,
                          const cp_bind_tensor_spec_t *spec,
                          size_t shape[CP_BIND_MAX_DIM]) {
    if (!spec || spec->ndim <= 0 || spec->ndim > CP_BIND_MAX_DIM) return -1;
    for (int i = 0; i < spec->ndim; i++) {
        if (cp_bind_resolve_sym(dims, spec->shape_sym[i], &shape[i]) != 0)
            return -1;
    }
    return spec->ndim;
}

void cp_bind_expand_name(char *buf, size_t cap, const char *pattern,
                         uint32_t layer) {
    if (!buf || cap == 0) return;
    buf[0] = '\0';
    if (!pattern) return;

    char tmp[CP_BIND_NAME_MAX * 2];
    size_t o = 0;
    for (const char *p = pattern; *p && o + 1 < sizeof tmp; ) {
        if (p[0] == '{' && p[1] == 'L' && p[2] == '}') {
            int n = snprintf(tmp + o, sizeof tmp - o, "%u", layer);
            if (n < 0) break;
            o += (size_t)n;
            p += 3;
        } else {
            tmp[o++] = *p++;
        }
    }
    tmp[o < sizeof tmp ? o : sizeof tmp - 1] = '\0';
    copy_cstr(buf, cap, tmp);
}

st_dtype_t cp_bind_resolve_dtype(const cp_bind_recipe_t *r, const st_file_t *f) {
    if (!r) return ST_DTYPE_COUNT;
    if (r->dtype_from[0]) {
        const st_tensor_t *t = st_find(f, r->dtype_from);
        if (!t) return ST_DTYPE_COUNT;
        if (t->dtype == ST_DTYPE_F32 || t->dtype == ST_DTYPE_BF16 ||
            t->dtype == ST_DTYPE_F16)
            return t->dtype;
        return ST_DTYPE_COUNT;
    }
    if (strcmp(r->dtype_name, "BF16") == 0) return ST_DTYPE_BF16;
    if (strcmp(r->dtype_name, "F32") == 0) return ST_DTYPE_F32;
    if (strcmp(r->dtype_name, "F16") == 0) return ST_DTYPE_F16;
    return ST_DTYPE_COUNT;
}

const st_tensor_t *cp_bind_expect(const st_file_t *f, const char *tag,
                                  const char *name, st_dtype_t dtype,
                                  size_t ndim, const size_t *shape,
                                  size_t *n_bound, size_t *n_expected) {
    (*n_expected)++;
    const st_tensor_t *t = st_find(f, name);
    if (!t) {
        fprintf(stderr, "%s: [MISS] %-70s not in file\n", tag, name);
        return NULL;
    }
    if (t->dtype != dtype) {
        fprintf(stderr, "%s: [DTYPE] %-70s is %s, want %s\n", tag, name,
                st_dtype_name(t->dtype), st_dtype_name(dtype));
        return NULL;
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "%s: [RANK] %-70s ndim=%zu, want %zu\n", tag, name,
                t->ndim, ndim);
        return NULL;
    }
    for (size_t d = 0; d < ndim; d++) {
        if (t->shape[d] != shape[d]) {
            fprintf(stderr, "%s: [SHAPE] %-70s dim%zu=%zu, want %zu\n", tag,
                    name, d, t->shape[d], shape[d]);
            return NULL;
        }
    }
    (*n_bound)++;
    return t;
}

int cp_bind_spec_active(const cp_bind_tensor_spec_t *spec,
                        const cp_bind_dims_t *dims, int layer_index) {
    if (!spec || !dims) return 0;
    if (spec->if_not_tied && dims->tied_head) return 0;
    if (spec->if_gated == 1 && !dims->gated_ff) return 0;
    if (spec->if_gated == -1 && dims->gated_ff) return 0;
    if (spec->only_layer >= 0 && layer_index != spec->only_layer) return 0;
    return 1;
}
