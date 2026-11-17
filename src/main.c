/*
 * main.c - Entry point of CPulse, the C LLM inference engine
 *
 * Project : CPulse - minimal LLM inference engine written from scratch in C
 * Model   : Qwen2.5-0.5B (decoder-only Transformer)
 * Stage   : 3 - config loading (weight mapping comes next)
 *
 * What this stage does:
 *   - verify the model weights are present and discoverable
 *   - parse the safetensors file header (hand-rolled minimal JSON parser)
 *   - print the full "tensor map": every tensor's name/shape/dtype/size,
 *     plus aggregate statistics used as the ground truth for later stages
 *
 * Usage:
 *   ./build/cpulse.exe                 summary output
 *   ./build/cpulse.exe --list          dump every tensor
 *   ./build/cpulse.exe --model <dir>   use a different model directory
 *
 * Build:
 *   mingw32-make          (release, -O2)
 *   mingw32-make debug    (debug, -O0 -g3, for gdb / VS Code launch.json)
 *   mingw32-make run      (build release and run from project root)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "safetensors.h"
#include "model.h"
#include "weights.h"

#define DEFAULT_MODEL_DIR ".cache/Qwen2.5-0.5B"
#define ENGINE_NAME       "cpulse"
#define ENGINE_VERSION    "0.3.0"

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
    long long size = -1;
#ifdef _WIN32
    _fseeki64(fp, 0, SEEK_END);
    size = _ftelli64(fp);
#else
    fseek(fp, 0, SEEK_END);
    size = (long long)ftell(fp);
#endif
    fclose(fp);
    printf("  [ OK ] %-24s %10.1f KB\n", name, size / 1024.0);
    return 0;
}

/* one-line description of a tensor for the map dump */
static void print_tensor_line(const st_tensor_t *t) {
    printf("  %-46s %-5s [", t->name, st_dtype_name(t->dtype));
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%s%zu", i ? ", " : "", t->shape[i]);
    }
    printf("] %10.1f KB\n", t->nbytes / 1024.0);
}

static int is_layer_tensor(const char *name) {
    return strncmp(name, "model.layers.", 13) == 0;
}

static void print_tensor_map(const st_file_t *f, int full_list) {
    /* ---- aggregate statistics (the ground truth for later stages) ---- */
    uint64_t total_params = 0;
    uint64_t total_bytes = 0;
    size_t   dtype_count[ST_DTYPE_COUNT] = {0};
    size_t   n_layer_tensors = 0;

    for (size_t i = 0; i < f->n_tensors; i++) {
        const st_tensor_t *t = &f->tensors[i];
        uint64_t prod = 1;
        for (size_t d = 0; d < t->ndim; d++) prod *= t->shape[d];
        total_params += prod;
        total_bytes += t->nbytes;
        if (t->dtype < ST_DTYPE_COUNT) dtype_count[t->dtype]++;
        if (is_layer_tensor(t->name)) n_layer_tensors++;
    }

    printf("Header length : %llu bytes\n", (unsigned long long)f->header_len);
    printf("Tensors       : %zu\n", f->n_tensors);
    printf("  layer-bound : %zu (model.layers.*)\n", n_layer_tensors);
    printf("  other       : %zu\n", f->n_tensors - n_layer_tensors);
    printf("Total params  : %llu (%.1f M)\n",
           (unsigned long long)total_params, total_params / 1.0e6);
    printf("Total weights : %llu bytes (%.1f MB)\n",
           (unsigned long long)total_bytes, total_bytes / (1024.0 * 1024.0));
    printf("Dtypes        : ");
    for (int d = 0; d < ST_DTYPE_COUNT; d++) {
        if (dtype_count[d] > 0) {
            printf("%s x %zu  ", st_dtype_name((st_dtype_t)d), dtype_count[d]);
        }
    }
    printf("\n");

    /* ---- non-layer tensors (embeddings, final norm, lm_head, ...) ---- */
    printf("\nNon-layer tensors:\n");
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (!is_layer_tensor(f->tensors[i].name)) {
            print_tensor_line(&f->tensors[i]);
        }
    }

    /* ---- layer 0 in full: shows the per-block structure ---- */
    printf("\nLayer 0 tensors (per-block template):\n");
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (strncmp(f->tensors[i].name, "model.layers.0.", 15) == 0) {
            print_tensor_line(&f->tensors[i]);
        }
    }

    /* ---- full dump on request ---- */
    if (full_list) {
        printf("\nAll tensors:\n");
        for (size_t i = 0; i < f->n_tensors; i++) {
            print_tensor_line(&f->tensors[i]);
        }
    }
}

/* VS Code cppdbg on Windows appends "2>CON 1>CON <CON" to the debuggee
 * command line when externalConsole is false (MIEngine console
 * redirection, see avoidWindowsConsoleRedirection in launch.json).
 * Those tokens are meant for cmd.exe but arrive as literal argv entries,
 * so ignore them to keep F5 debugging working out of the box. */
static int is_console_redir_token(const char *arg) {
    return strcmp(arg, "2>CON") == 0 ||
           strcmp(arg, "1>CON") == 0 ||
           strcmp(arg, "<CON") == 0;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [--model <model_dir>] [--list]\n", prog);
    printf("  --model <model_dir>   path to the model directory\n");
    printf("                        (default: %s)\n", DEFAULT_MODEL_DIR);
    printf("  --list                dump every tensor in the weight file\n");
}

int main(int argc, char **argv) {
    const char *model_dir = DEFAULT_MODEL_DIR;
    int full_list = 0;

    for (int i = 1; i < argc; i++) {
        if (is_console_redir_token(argv[i])) {
            continue;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--list") == 0) {
            full_list = 1;
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

    /* ---- stage 1 gate: raw files present ---- */
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

    /* ---- stage 3: config.json -> architecture hyperparameters ---- */
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    printf("Loading config: %s\n", config_path);
    cp_config_t cfg;
    if (cp_config_load(config_path, &cfg) != 0) {
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }
    printf("\n");
    cp_config_print(&cfg);

    /* ---- stage 2: parse the safetensors header, build the tensor map ---- */
    char weights_path[512];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);
    printf("\nLoading weights: %s\n", weights_path);

    st_file_t *f = st_load(weights_path);
    if (!f) {
        printf("Result: FAIL - safetensors parsing failed (see stderr).\n");
        return 4;
    }

    printf("\nTensor map:\n");
    print_tensor_map(f, full_list);

    /* ---- stage 3b: bind every tensor onto the architecture ---- */
    printf("\n");
    cp_weights_t w;
    if (cp_weights_bind(&cfg, f, &w) != 0) {
        printf("Result: FAIL - weight binding failed (see stderr).\n");
        st_free(f);
        return 5;
    }
    cp_weights_print(&w);
    cp_weights_free(&w);

    st_free(f);

    printf("\nResult: OK - config + weights parsed, tensor map built.\n");
    printf("Next  : forward-pass operators (RMSNorm, attention, MLP).\n");
    return 0;
}
