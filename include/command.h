/*
 * command.h - CLI command handlers for CPulse
 *
 * main parses argv into cp_cli_t; cp_command_run dispatches:
 *   --token  -> vocab lookup only
 *   --input  -> load model + greedy generate
 *   (default)-> load model + summary / bind check
 */

#ifndef COMMAND_H
#define COMMAND_H

#ifdef __cplusplus
extern "C" {
#endif

#define CP_DEFAULT_MODEL_DIR ".cache/Qwen2.5-0.5B"
#define CP_DEFAULT_MAX_NEW   64

typedef struct {
    const char *model_dir;
    const char *arch_config; /* NULL => CP_DEFAULT_ARCH_CONFIG */
    const char *input_text;  /* NULL unless --input */
    const char *token_text;  /* NULL unless --token (takes priority) */
    /* T5 tokenizer backend: "json" (default) or "spiece"; NULL => json */
    const char *tok_backend;
    int         max_new;
    int         use_gpu;
} cp_cli_t;

/* Print CLI usage to stdout. */
void cp_cmd_help(const char *prog);

/* Route to the appropriate command. Returns process exit code. */
int cp_command_run(const cp_cli_t *cli);

#ifdef __cplusplus
}
#endif

#endif /* COMMAND_H */
