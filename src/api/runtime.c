/*
 * api/runtime.c - lazy model load/cache for HTTP API
 */

#include "api/runtime.h"
#include "arch.h"
#include "bind_recipe.h"
#include "qwen/model.h"
#include "qwen/weights.h"
#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cp_api_runtime {
    char            id[256];
    char            path[1024];
    cp_config_t     cfg;
    st_file_t      *st;
    cp_weights_t    w;
    cp_tokenizer_t *tok;
    cp_session_t   *sess;
};

static void runtime_free(cp_api_runtime_t *rt) {
    if (!rt) return;
    cp_session_free(rt->sess);
    cp_tokenizer_free(rt->tok);
    cp_weights_free(&rt->w);
    st_free(rt->st);
    free(rt);
}

void cp_api_runtime_release(cp_api_ctx_t *ctx) {
    if (!ctx) return;
    runtime_free(ctx->rt);
    ctx->rt = NULL;
}

const char *cp_api_runtime_id(const cp_api_runtime_t *rt) {
    return rt ? rt->id : NULL;
}

cp_tokenizer_t *cp_api_runtime_tokenizer(cp_api_runtime_t *rt) {
    return rt ? rt->tok : NULL;
}

cp_session_t *cp_api_runtime_session(cp_api_runtime_t *rt) {
    return rt ? rt->sess : NULL;
}

const cp_config_t *cp_api_runtime_config(const cp_api_runtime_t *rt) {
    return rt ? &rt->cfg : NULL;
}

static void set_err(char *err, size_t cap, const char *msg) {
    if (err && cap) snprintf(err, cap, "%s", msg ? msg : "error");
}

int cp_api_runtime_acquire(cp_api_ctx_t *ctx, const char *id,
                           cp_api_runtime_t **out, char *err, size_t err_cap) {
    if (!ctx || !id || !out) {
        set_err(err, err_cap, "invalid arguments");
        return -1;
    }
    *out = NULL;

    if (ctx->rt && strcmp(ctx->rt->id, id) == 0) {
        *out = ctx->rt;
        return 0;
    }

    cp_api_catalog_scan(ctx);
    const char *path = cp_api_resolve_model_path(ctx, id);
    if (!path) {
        set_err(err, err_cap, "model not found in models_dir");
        return -1;
    }

    char config_path[1100];
    snprintf(config_path, sizeof config_path, "%s/config.json", path);

    char model_type[32];
    if (cp_config_peek_model_type(config_path, model_type, sizeof model_type) !=
        0) {
        set_err(err, err_cap, "failed to read model_type");
        return -1;
    }
    const cp_arch_t *arch = cp_arch_resolve(model_type);
    if (!arch || !arch->implemented) {
        set_err(err, err_cap, "model architecture not supported");
        return -1;
    }
    if (arch->id != CP_ARCH_DECODER_ONLY) {
        set_err(err, err_cap,
                "chat/completions requires a decoder-only chat model");
        return -1;
    }

    cp_api_runtime_t *rt = calloc(1, sizeof *rt);
    if (!rt) {
        set_err(err, err_cap, "out of memory");
        return -1;
    }
    snprintf(rt->id, sizeof rt->id, "%s", id);
    snprintf(rt->path, sizeof rt->path, "%s", path);

    if (cp_config_load(config_path, &rt->cfg) != 0) {
        set_err(err, err_cap, "config load failed");
        runtime_free(rt);
        return -1;
    }

    char weights_path[1100];
    snprintf(weights_path, sizeof weights_path, "%s/model.safetensors", path);
    rt->st = st_load(weights_path);
    if (!rt->st) {
        set_err(err, err_cap, "safetensors load failed");
        runtime_free(rt);
        return -1;
    }
    if (cp_weights_bind(&rt->cfg, rt->st, &rt->w) != 0) {
        set_err(err, err_cap, "weight bind failed");
        runtime_free(rt);
        return -1;
    }
    if (cp_tokenizer_load(&rt->tok, path) != 0) {
        set_err(err, err_cap, "tokenizer load failed");
        runtime_free(rt);
        return -1;
    }
    if (cp_session_init(&rt->sess, &rt->cfg, &rt->w, ctx->use_gpu) != 0) {
        set_err(err, err_cap, "session init failed");
        runtime_free(rt);
        return -1;
    }

    runtime_free(ctx->rt);
    ctx->rt = rt;
    *out = rt;
    fprintf(stdout, "api: loaded model \"%s\" from %s\n", id, path);
    fflush(stdout);
    return 0;
}
