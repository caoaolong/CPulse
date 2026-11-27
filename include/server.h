/*
 * server.h - OpenAI-compatible HTTP listener (Mongoose)
 *
 * Route handlers live in src/api/ (one file per endpoint family).
 * Serve mode does not preload weights; catalog comes from models_dir.
 */

#ifndef SERVER_H
#define SERVER_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CP_DEFAULT_SERVER_PORT
#define CP_DEFAULT_SERVER_PORT 8080
#endif

#ifndef CP_DEFAULT_MODELS_DIR
#define CP_DEFAULT_MODELS_DIR ".cache"
#endif

typedef struct {
    const char *models_dir; /* scan root; NULL => CP_DEFAULT_MODELS_DIR */
    const char *host;       /* NULL => "0.0.0.0" */
    int         port;       /* <=0 => CP_DEFAULT_SERVER_PORT */
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
