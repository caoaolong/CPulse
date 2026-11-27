/*
 * api/ctx.c - shared API context: model catalog + JSON replies
 */

#include "api/ctx.h"
#include "server.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static int path_is_dir(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int file_exists_in(const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int looks_like_model_dir(const char *dir) {
    return file_exists_in(dir, "config.json") != 0;
}

static int catalog_add(cp_api_ctx_t *ctx, const char *id, const char *path) {
    if (ctx->n_models >= CP_API_MAX_MODELS) return -1;
    for (size_t i = 0; i < ctx->n_models; i++) {
        if (strcmp(ctx->models[i].id, id) == 0) return 0;
    }
    cp_api_model_t *e = &ctx->models[ctx->n_models];
    snprintf(e->id, sizeof e->id, "%s", id);
    snprintf(e->path, sizeof e->path, "%s", path);
    ctx->n_models++;
    return 0;
}

void cp_api_ctx_init(cp_api_ctx_t *ctx, const char *models_dir, int use_gpu) {
    memset(ctx, 0, sizeof *ctx);
    const char *root =
            (models_dir && models_dir[0]) ? models_dir : CP_DEFAULT_MODELS_DIR;
    snprintf(ctx->models_dir, sizeof ctx->models_dir, "%s", root);
    ctx->created = (long)time(NULL);
    ctx->use_gpu = use_gpu ? 1 : 0;
    cp_api_catalog_scan(ctx);
}

void cp_api_catalog_scan(cp_api_ctx_t *ctx) {
    ctx->n_models = 0;
    DIR *d = opendir(ctx->models_dir);
    if (!d) {
        fprintf(stderr, "api: cannot open models dir '%s'\n", ctx->models_dir);
        return;
    }
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char path[1024];
        snprintf(path, sizeof path, "%s/%s", ctx->models_dir, ent->d_name);
        if (!path_is_dir(path) || !looks_like_model_dir(path)) continue;
        if (catalog_add(ctx, ent->d_name, path) != 0) {
            fprintf(stderr, "api: model catalog full (max %d)\n",
                    CP_API_MAX_MODELS);
            break;
        }
    }
    closedir(d);
}

const cp_api_model_t *cp_api_find_model(const cp_api_ctx_t *ctx,
                                        const char *id) {
    if (!ctx || !id) return NULL;
    for (size_t i = 0; i < ctx->n_models; i++) {
        if (strcmp(ctx->models[i].id, id) == 0) return &ctx->models[i];
    }
    return NULL;
}

const char *cp_api_resolve_model_path(const cp_api_ctx_t *ctx, const char *id) {
    const cp_api_model_t *e = cp_api_find_model(ctx, id);
    return e ? e->path : NULL;
}

void cp_api_reply_json(struct mg_connection *c, int status, const char *json) {
    mg_http_reply(c, status,
                  "Content-Type: application/json; charset=utf-8\r\n"
                  "Access-Control-Allow-Origin: *\r\n",
                  "%s", json ? json : "{}");
}

void cp_api_reply_options(struct mg_connection *c) {
    mg_http_reply(c, 204,
                  "Access-Control-Allow-Origin: *\r\n"
                  "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                  "Access-Control-Allow-Headers: Content-Type, Authorization\r\n",
                  "");
}
