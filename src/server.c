/*
 * server.c - OpenAI-compatible HTTP API (see server.h)
 *
 * Uses vendored Mongoose. Phase 1: GET /v1/models (+ OPTIONS CORS).
 */

/* Cygwin has no epoll; force poll/select path (also keeps IntelliSense happy). */
#define MG_TLS MG_TLS_NONE
#define MG_ENABLE_EPOLL 0
#include "mongoose.h"

#include "server.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char model_id[256];
    long created;
} server_state_t;

static void model_id_from_dir(const char *model_dir, char *out, size_t cap) {
    const char *base = model_dir && model_dir[0] ? model_dir : "model";
    size_t n = strlen(base);
    while (n > 0 && (base[n - 1] == '/' || base[n - 1] == '\\')) n--;
    size_t start = 0;
    for (size_t i = 0; i < n; i++) {
        if (base[i] == '/' || base[i] == '\\') start = i + 1;
    }
    size_t len = n - start;
    if (len == 0) {
        snprintf(out, cap, "model");
        return;
    }
    if (len >= cap) len = cap - 1;
    memcpy(out, base + start, len);
    out[len] = '\0';
}

static const char *kCors =
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

static void reply_json(struct mg_connection *c, int status, const char *json) {
    mg_http_reply(c, status,
                  "Content-Type: application/json; charset=utf-8\r\n"
                  "Access-Control-Allow-Origin: *\r\n",
                  "%s", json ? json : "{}");
}

static char *build_models_list_json(const server_state_t *st) {
    cJSON *root = cJSON_CreateObject();
    cJSON *data = cJSON_CreateArray();
    cJSON *item = cJSON_CreateObject();
    if (!root || !data || !item) {
        cJSON_Delete(root);
        cJSON_Delete(data);
        cJSON_Delete(item);
        return NULL;
    }
    cJSON_AddStringToObject(root, "object", "list");
    cJSON_AddStringToObject(item, "id", st->model_id);
    cJSON_AddStringToObject(item, "object", "model");
    cJSON_AddNumberToObject(item, "created", (double)st->created);
    cJSON_AddStringToObject(item, "owned_by", "cpulse");
    cJSON_AddItemToArray(data, item);
    cJSON_AddItemToObject(root, "data", data);
    char *s = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return s;
}

static char *build_model_json(const server_state_t *st, const char *id) {
    if (strcmp(id, st->model_id) != 0) return NULL;
    cJSON *item = cJSON_CreateObject();
    if (!item) return NULL;
    cJSON_AddStringToObject(item, "id", st->model_id);
    cJSON_AddStringToObject(item, "object", "model");
    cJSON_AddNumberToObject(item, "created", (double)st->created);
    cJSON_AddStringToObject(item, "owned_by", "cpulse");
    char *s = cJSON_PrintUnformatted(item);
    cJSON_Delete(item);
    return s;
}

static void handle_http(struct mg_connection *c, struct mg_http_message *hm,
                        server_state_t *st) {
    if (mg_strcmp(hm->method, mg_str("OPTIONS")) == 0) {
        mg_http_reply(c, 204, kCors, "");
        return;
    }

    /* GET /v1/models */
    if (mg_match(hm->uri, mg_str("/v1/models"), NULL) &&
        mg_strcmp(hm->method, mg_str("GET")) == 0) {
        char *json = build_models_list_json(st);
        if (!json) {
            reply_json(c, 500, "{\"error\":{\"message\":\"oom\",\"type\":\"server_error\"}}");
            return;
        }
        reply_json(c, 200, json);
        free(json);
        return;
    }

    /* GET /v1/models/{id} — OpenAI also exposes this */
    {
        struct mg_str caps[1];
        if (mg_match(hm->uri, mg_str("/v1/models/*"), caps) &&
            mg_strcmp(hm->method, mg_str("GET")) == 0) {
            char id[256];
            size_t n = caps[0].len < sizeof id - 1 ? caps[0].len : sizeof id - 1;
            memcpy(id, caps[0].buf, n);
            id[n] = '\0';
            char *json = build_model_json(st, id);
            if (!json) {
                reply_json(c, 404,
                           "{\"error\":{\"message\":\"model not found\","
                           "\"type\":\"invalid_request_error\",\"code\":\"model_not_found\"}}");
                return;
            }
            reply_json(c, 200, json);
            free(json);
            return;
        }
    }

    reply_json(c, 404,
               "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}");
}

static void ev_handler(struct mg_connection *c, int ev, void *ev_data) {
    server_state_t *st = (server_state_t *)c->fn_data;
    if (ev == MG_EV_HTTP_MSG) {
        handle_http(c, (struct mg_http_message *)ev_data, st);
    }
}

int cp_server_run(const cp_server_cfg_t *cfg) {
    if (!cfg) return -1;

    server_state_t st;
    memset(&st, 0, sizeof st);
    model_id_from_dir(cfg->model_dir, st.model_id, sizeof st.model_id);
    st.created = (long)time(NULL);

    const char *host = (cfg->host && cfg->host[0]) ? cfg->host : "0.0.0.0";
    int port = cfg->port > 0 ? cfg->port : CP_DEFAULT_SERVER_PORT;

    char url[128];
    snprintf(url, sizeof url, "http://%s:%d", host, port);

    struct mg_mgr mgr;
    mg_mgr_init(&mgr);
    struct mg_connection *c = mg_http_listen(&mgr, url, ev_handler, &st);
    if (!c) {
        fprintf(stderr, "server: cannot listen on %s\n", url);
        mg_mgr_free(&mgr);
        return -1;
    }

    printf("server: listening on %s\n", url);
    printf("server: model id \"%s\" (from %s)\n", st.model_id,
           cfg->model_dir ? cfg->model_dir : "(null)");
    printf("server: try  GET %s/v1/models\n", url);
    fflush(stdout);

    for (;;)
        mg_mgr_poll(&mgr, 1000);

    mg_mgr_free(&mgr);
    return 0;
}
