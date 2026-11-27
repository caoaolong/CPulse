/*
 * api/router.c - dispatch OpenAI-compatible routes to per-endpoint handlers
 */

#include "api/chat_completions.h"
#include "api/ctx.h"
#include "api/models.h"

void cp_api_dispatch(struct mg_connection *c, struct mg_http_message *hm,
                     cp_api_ctx_t *ctx) {
    if (mg_strcmp(hm->method, mg_str("OPTIONS")) == 0) {
        cp_api_reply_options(c);
        return;
    }

    if (api_models_handle(c, hm, ctx)) return;
    if (api_chat_completions_handle(c, hm, ctx)) return;

    cp_api_reply_json(
            c, 404,
            "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}");
}
