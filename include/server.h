/*
 * server.h - OpenAI-compatible HTTP API (Mongoose)
 *
 * Current surface: GET /v1/models
 * Blocking event loop; returns only on listen failure.
 */

#ifndef SERVER_H
#define SERVER_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CP_DEFAULT_SERVER_PORT
#define CP_DEFAULT_SERVER_PORT 8080
#endif

typedef struct {
    const char *model_dir; /* used to derive OpenAI model id (basename) */
    const char *host;      /* NULL => "0.0.0.0" */
    int         port;      /* <=0 => CP_DEFAULT_SERVER_PORT */
} cp_server_cfg_t;

/*
 * Run the HTTP server (never returns on success).
 * Returns non-zero if the listener could not be created.
 */
int cp_server_run(const cp_server_cfg_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* SERVER_H */
