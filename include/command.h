/*
 * command.h - CLI command handlers for CPulse
 *
 * main parses argv into cp_cli_t; cp_command_run dispatches:
 *   --token  -> vocab lookup only
 *   --input  -> load model + generate
 *   (default)-> load model + summary / bind check
 */

#ifndef COMMAND_H
#define COMMAND_H

#include "sample.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CP_DEFAULT_MODEL_DIR ".cache/Qwen2.5-0.5B"
#define CP_DEFAULT_MAX_NEW   64

typedef struct {
    const char *model_dir;   /* CLI load/generate; unused by --serve catalog */
    const char *models_dir;  /* --serve scan root; NULL => CP_DEFAULT_MODELS_DIR */
    const char *routes_path; /* NULL => CP_DEFAULT_ROUTES (arch+tokenizer) */
    const char *input_text;  /* NULL unless --input */
    const char *token_text;  /* NULL unless --token (takes priority) */
    const char *view_path;   /* non-NULL => --view ("" = default path) */
    int         max_new;
    int         use_gpu;
    int         serve;       /* 1 => OpenAI-compatible HTTP server */
    int         port;        /* HTTP listen port when serve != 0 */
    cp_sample_cfg_t sample;  /* temperature / top-k / top-p / seed */
} cp_cli_t;

/* Print CLI usage to stdout. */
void cp_cmd_help(const char *prog);

/* Route to the appropriate command. Returns process exit code. */
int cp_command_run(const cp_cli_t *cli);

#ifdef __cplusplus
}
#endif

#endif /* COMMAND_H */
