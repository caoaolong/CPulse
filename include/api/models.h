/*
 * api/models.h - GET /v1/models and GET /v1/models/{id}
 */

#ifndef API_MODELS_H
#define API_MODELS_H

#include "api/ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Handle models endpoints.
 * Returns 1 if the request was consumed, 0 if URI/method did not match.
 */
int api_models_handle(struct mg_connection *c, struct mg_http_message *hm,
                      cp_api_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* API_MODELS_H */
