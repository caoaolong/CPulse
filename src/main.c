/*
 * main.c - Entry point of CPulse, the C LLM inference engine
 *
 * Project : CPulse - minimal LLM inference engine written from scratch in C
 * Model   : Qwen2.5-0.5B (decoder-only Transformer)
 * Stage   : 1 - project skeleton / toolchain verification
 *
 * What this stage does:
 *   - verify the build system works (gcc + Makefile)
 *   - verify the model weights are present and discoverable
 *   - prepare the ground for the next stage: safetensors loading
 *
 * Build:
 *   mingw32-make          (release, -O2)
 *   mingw32-make debug    (debug, -O0 -g3, for gdb / VS Code launch.json)
 *   mingw32-make run      (build release and run from project root)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MODEL_DIR ".cache/Qwen2.5-0.5B"
#define ENGINE_NAME    "cpulse"
#define ENGINE_VERSION "0.1.0"

static void print_banner(void) {
    printf("============================================================\n");
    printf("  %s v%s - C inference engine for Qwen2.5-0.5B\n", ENGINE_NAME, ENGINE_VERSION);
    printf("  built: %s %s, C11, gcc/MinGW-w64\n", __DATE__, __TIME__);
    printf("============================================================\n");
}

/* Check that a required model file exists; returns 0 if OK, -1 if missing */
static int check_file(const char *base, const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", base, name);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("  [MISS] %-24s (%s)\n", name, path);
        return -1;
    }
    /* read file size */
    long size = -1;
    if (fseek(fp, 0, SEEK_END) == 0) {
        size = ftell(fp);
    }
    fclose(fp);
    printf("  [ OK ] %-24s %10.1f KB\n", name, size / 1024.0);
    return 0;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [--model <model_dir>]\n", prog);
    printf("  --model <model_dir>   path to the model directory\n");
    printf("                        (default: %s)\n", DEFAULT_MODEL_DIR);
}

int main(int argc, char **argv) {
    const char *model_dir = DEFAULT_MODEL_DIR;

    /* minimal argument parsing */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    print_banner();
    printf("\n");
    printf("Model directory : %s\n", model_dir);
    printf("\n");

    /* Stage 1 gate: verify all files the engine will need next */
    printf("Checking model files...\n");
    int ok = 1;
    ok &= check_file(model_dir, "config.json") == 0;
    ok &= check_file(model_dir, "model.safetensors") == 0;
    ok &= check_file(model_dir, "tokenizer.json") == 0;
    ok &= check_file(model_dir, "tokenizer_config.json") == 0;
    ok &= check_file(model_dir, "vocab.json") == 0;
    ok &= check_file(model_dir, "merges.txt") == 0;
    printf("\n");

    if (!ok) {
        printf("Result: FAIL - some model files are missing.\n");
        printf("Hint  : run scripts/download_model.ps1 to fetch the model.\n");
        return 2;
    }

    printf("Result: OK - toolchain works, model weights found.\n");
    printf("Next  : implement safetensors header parsing + weight loading.\n");
    return 0;
}
