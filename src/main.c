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
 *   ./build/cpulse.exe --model <dir> --view [out.svg]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "command.h"

#define ENGINE_NAME    "cpulse"
#define ENGINE_VERSION "0.3.0"

/* VS Code / Cursor cppdbg on Windows may append console-redirection
 * tokens as literal argv entries when externalConsole is false. */
static int is_console_redir_token(const char *arg) {
    return strcmp(arg, "2>CON") == 0 ||
           strcmp(arg, "1>CON") == 0 ||
           strcmp(arg, "<CON") == 0;
}

int main(int argc, char **argv) {
    cp_cli_t cli = {
        .model_dir   = CP_DEFAULT_MODEL_DIR,
        .arch_config = NULL,
        .input_text  = NULL,
        .token_text  = NULL,
        .view_path   = NULL,
        .max_new     = CP_DEFAULT_MAX_NEW,
        .use_gpu     = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (is_console_redir_token(argv[i])) {
            continue;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cli.model_dir = argv[++i];
        } else if (strcmp(argv[i], "--arch-config") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --arch-config\n");
                cp_cmd_help(argv[0]);
                return 1;
            }
            cli.arch_config = argv[++i];
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
        } else if (strcmp(argv[i], "--view") == 0) {
            /* optional path: --view out.svg  or bare --view.
             * Do NOT treat cppdbg's injected "2>CON" / "1>CON" as a path. */
            if (i + 1 < argc && argv[i + 1][0] != '-' &&
                !is_console_redir_token(argv[i + 1])) {
                cli.view_path = argv[++i];
            } else {
                cli.view_path = ""; /* default: <model_name>.svg */
            }
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
