/*
 * api/runtime.h - lazy-loaded model bundle for HTTP chat
 *
 * Currently decoder-only (Qwen). Enc–Dec returns an error string.
 */

#ifndef API_RUNTIME_H
#define API_RUNTIME_H

#include "api/ctx.h"
#include "qwen/generate.h"
#include "tok/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_api_runtime cp_api_runtime_t;

/*
 * Ensure model `id` is loaded (path from catalog). Reuses cache if same id.
 * On success returns 0 and *out. On failure returns -1 and fills err (cap).
 */
int cp_api_runtime_acquire(cp_api_ctx_t *ctx, const char *id,
                           cp_api_runtime_t **out, char *err, size_t err_cap);

void cp_api_runtime_release(cp_api_ctx_t *ctx); /* free cached runtime */

const char *cp_api_runtime_id(const cp_api_runtime_t *rt);
cp_tokenizer_t *cp_api_runtime_tokenizer(cp_api_runtime_t *rt);
cp_session_t *cp_api_runtime_session(cp_api_runtime_t *rt);
const cp_config_t *cp_api_runtime_config(const cp_api_runtime_t *rt);

#ifdef __cplusplus
}
#endif

#endif /* API_RUNTIME_H */
