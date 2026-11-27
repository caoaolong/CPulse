/*
 * api/models.c - GET /v1/models , GET /v1/models/{id}
 */

#include "api/models.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *build_list_json(const cp_api_ctx_t *ctx) {
    cJSON *root = cJSON_CreateObject();
    cJSON *data = cJSON_CreateArray();
    if (!root || !data) {
        cJSON_Delete(root);
        cJSON_Delete(data);
        return NULL;
    }
    cJSON_AddStringToObject(root, "object", "list");
    for (size_t i = 0; i < ctx->n_models; i++) {
        cJSON *item = cJSON_CreateObject();
        if (!item) continue;
        cJSON_AddStringToObject(item, "id", ctx->models[i].id);
        cJSON_AddStringToObject(item, "object", "model");
        cJSON_AddNumberToObject(item, "created", (double)ctx->created);
        cJSON_AddStringToObject(item, "owned_by", "cpulse");
        cJSON_AddItemToArray(data, item);
    }
    cJSON_AddItemToObject(root, "data", data);
    char *s = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return s;
}

static char *build_one_json(const cp_api_ctx_t *ctx, const char *id) {
    const cp_api_model_t *e = cp_api_find_model(ctx, id);
    if (!e) return NULL;
    cJSON *item = cJSON_CreateObject();
    if (!item) return NULL;
    cJSON_AddStringToObject(item, "id", e->id);
    cJSON_AddStringToObject(item, "object", "model");
    cJSON_AddNumberToObject(item, "created", (double)ctx->created);
    cJSON_AddStringToObject(item, "owned_by", "cpulse");
    char *s = cJSON_PrintUnformatted(item);
    cJSON_Delete(item);
    return s;
}

int api_models_handle(struct mg_connection *c, struct mg_http_message *hm,
                      cp_api_ctx_t *ctx) {
    if (mg_match(hm->uri, mg_str("/v1/models"), NULL) &&
        mg_strcmp(hm->method, mg_str("GET")) == 0) {
        cp_api_catalog_scan(ctx);
        char *json = build_list_json(ctx);
        if (!json) {
            cp_api_reply_json(
                    c, 500,
                    "{\"error\":{\"message\":\"oom\",\"type\":\"server_error\"}}");
            return 1;
        }
        cp_api_reply_json(c, 200, json);
        free(json);
        return 1;
    }

    struct mg_str caps[1];
    if (mg_match(hm->uri, mg_str("/v1/models/*"), caps) &&
        mg_strcmp(hm->method, mg_str("GET")) == 0) {
        char id[256];
        size_t n = caps[0].len < sizeof id - 1 ? caps[0].len : sizeof id - 1;
        memcpy(id, caps[0].buf, n);
        id[n] = '\0';

        cp_api_catalog_scan(ctx);
        if (!cp_api_resolve_model_path(ctx, id)) {
            cp_api_reply_json(
                    c, 404,
                    "{\"error\":{\"message\":\"model not found\","
                    "\"type\":\"invalid_request_error\","
                    "\"code\":\"model_not_found\"}}");
            return 1;
        }
        char *json = build_one_json(ctx, id);
        if (!json) {
            cp_api_reply_json(
                    c, 500,
                    "{\"error\":{\"message\":\"oom\",\"type\":\"server_error\"}}");
            return 1;
        }
        cp_api_reply_json(c, 200, json);
        free(json);
        return 1;
    }

    return 0;
}
