/*
 * api/ctx.h - shared HTTP API context (catalog + reply helpers)
 */

#ifndef API_CTX_H
#define API_CTX_H

#include "api/mg.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CP_API_MAX_MODELS
#define CP_API_MAX_MODELS 64
#endif

typedef struct {
    char id[256];
    char path[1024];
} cp_api_model_t;

typedef struct cp_api_runtime cp_api_runtime_t;

typedef struct {
    char              models_dir[512];
    cp_api_model_t    models[CP_API_MAX_MODELS];
    size_t            n_models;
    long              created;
    int               use_gpu;
    cp_api_runtime_t *rt; /* lazily loaded; owned */
} cp_api_ctx_t;

void cp_api_ctx_init(cp_api_ctx_t *ctx, const char *models_dir, int use_gpu);
void cp_api_catalog_scan(cp_api_ctx_t *ctx);
const char *cp_api_resolve_model_path(const cp_api_ctx_t *ctx, const char *id);
const cp_api_model_t *cp_api_find_model(const cp_api_ctx_t *ctx, const char *id);

void cp_api_reply_json(struct mg_connection *c, int status, const char *json);
void cp_api_reply_options(struct mg_connection *c);

/* Dispatch all OpenAI /v1 routes. */
void cp_api_dispatch(struct mg_connection *c, struct mg_http_message *hm,
                     cp_api_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* API_CTX_H */
