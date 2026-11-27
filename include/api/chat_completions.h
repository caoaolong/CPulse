/*
 * api/chat_completions.h - POST /v1/chat/completions
 */

#ifndef API_CHAT_COMPLETIONS_H
#define API_CHAT_COMPLETIONS_H

#include "api/ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

int api_chat_completions_handle(struct mg_connection *c,
                                struct mg_http_message *hm, cp_api_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* API_CHAT_COMPLETIONS_H */
