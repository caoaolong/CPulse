/*
 * main.c - Entry point of CPulse
 *
 * Parses argv into cp_cli_t and dispatches via command.h.
 * Model load / generate / token lookup live in command.c.
 *
 * Usage:
 *   ./build/cpulse.exe
 *   ./build/cpulse.exe --input <text>
 *   ./build/cpulse.exe --model <dir> --token <tok>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "command.h"

#define ENGINE_NAME    "cpulse"
#define ENGINE_VERSION "0.3.0"

int main(int argc, char **argv) {
    cp_cli_t cli = {
        .model_dir  = CP_DEFAULT_MODEL_DIR,
        .input_text = NULL,
        .token_text = NULL,
        .max_new    = CP_DEFAULT_MAX_NEW,
        .use_gpu    = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cli.model_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --input\n");
                cp_cmd_help(argv[0]);
                return 1;
            }
            cli.input_text = argv[++i];
        } else if (strcmp(argv[i], "--token") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --token\n");
                cp_cmd_help(argv[0]);
                return 1;
            }
            cli.token_text = argv[++i];
        } else if (strcmp(argv[i], "--max-new") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --max-new\n");
                cp_cmd_help(argv[0]);
                return 1;
            }
            cli.max_new = atoi(argv[++i]);
            if (cli.max_new <= 0) cli.max_new = CP_DEFAULT_MAX_NEW;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            cli.use_gpu = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            cp_cmd_help(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            cp_cmd_help(argv[0]);
            return 1;
        }
    }

    printf("%s v%s | model %s\n", ENGINE_NAME, ENGINE_VERSION, cli.model_dir);
    return cp_command_run(&cli);
}
