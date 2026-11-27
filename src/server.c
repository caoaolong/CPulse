/*
 * server.c - OpenAI-compatible HTTP listener (Mongoose)
 *
 * Endpoint handlers live under src/api/ (one file per route family).
 */

#include "server.h"
#include "api/ctx.h"

#include <stdio.h>
#include <string.h>

static void ev_handler(struct mg_connection *c, int ev, void *ev_data) {
    cp_api_ctx_t *ctx = (cp_api_ctx_t *)c->fn_data;
    if (ev == MG_EV_HTTP_MSG)
        cp_api_dispatch(c, (struct mg_http_message *)ev_data, ctx);
}

int cp_server_run(const cp_server_cfg_t *cfg) {
    if (!cfg) return -1;

    cp_api_ctx_t ctx;
    cp_api_ctx_init(&ctx, cfg->models_dir);

    const char *host = (cfg->host && cfg->host[0]) ? cfg->host : "0.0.0.0";
    int port = cfg->port > 0 ? cfg->port : CP_DEFAULT_SERVER_PORT;

    char url[128];
    snprintf(url, sizeof url, "http://%s:%d", host, port);

    struct mg_mgr mgr;
    mg_mgr_init(&mgr);
    struct mg_connection *c = mg_http_listen(&mgr, url, ev_handler, &ctx);
    if (!c) {
        fprintf(stderr, "server: cannot listen on %s\n", url);
        mg_mgr_free(&mgr);
        return -1;
    }

    printf("server: listening on %s\n", url);
    printf("server: models_dir=%s (%zu available, lazy-load on chat)\n",
           ctx.models_dir, ctx.n_models);
    for (size_t i = 0; i < ctx.n_models; i++)
        printf("  - %s\n", ctx.models[i].id);
    printf("server: try  GET %s/v1/models\n", url);
    fflush(stdout);

    for (;;)
        mg_mgr_poll(&mgr, 1000);

    mg_mgr_free(&mgr);
    return 0;
}
