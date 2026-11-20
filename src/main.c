/*
 * main.c - Entry point of CPulse, the C LLM inference engine
 *
 * Project : CPulse - minimal LLM inference engine written from scratch in C
 * Model   : Qwen2.5-0.5B (decoder-only Transformer)
 *
 * Startup: verify model files, load config + weights, bind tensors.
 * With --input <text>: greedy-generate a raw-text continuation (this
 * checkpoint is the Qwen2.5-0.5B base model: no chat template).
 * With --gpu: run the forward pass on CUDA (falls back to cpu).
 * With --token <t>: print the vocab id of one surface-form token.
 *
 * Usage:
 *   ./build/cpulse.exe                 summary output
 *   ./build/cpulse.exe --input <text>  tokenize + embed text
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
#include "tokenizer.h"
#include "ops.h"
#include "forward.h"
#include "log.h"

#define DEFAULT_MODEL_DIR ".cache/Qwen2.5-0.5B"
#define ENGINE_NAME       "cpulse"
#define ENGINE_VERSION    "0.3.0"
#define DEFAULT_MAX_NEW    64
#define IM_END_ID        151645u

/* First-index-wins argmax over the logits row. */
static uint32_t argmax(const float *v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

/* Check that a required model file exists; silent on success,
 * one MISS line on failure. Returns 0 if OK, -1 if missing. */
static int check_file(const char *base, const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", base, name);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("  [MISS] %s\n", path);
        return -1;
    }
    fclose(fp);
    return 0;
}

/* Open the structured run log next to the binary's working directory.
 * Returns the path used, or NULL on failure. */
static const char *init_verbose_log(void) {
    cp_log_init("cpulse.log");
    if (cp_log_on()) return "cpulse.log";
    return NULL;
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
    printf("Usage: %s [--model <model_dir>] [--input <text>] [--token <token>] [--max-new <n>] [--verbose]\n", prog);
    printf("  --model <model_dir>   path to the model directory\n");
    printf("                        (default: %s)\n", DEFAULT_MODEL_DIR);
    printf("  --input <text>        greedy-generate a continuation of <text>\n");
    printf("  --token <token>       print the vocab id of one surface-form token\n");
    printf("  --max-new <n>         tokens to generate (default %d)\n", DEFAULT_MAX_NEW);
    printf("  --gpu                 run the forward pass on CUDA (falls back to cpu)\n");
    printf("  --verbose             write pipeline debug log to cpulse.log\n");
}

int main(int argc, char **argv) {
    const char *model_dir = DEFAULT_MODEL_DIR;
    const char *input_text = NULL;
    const char *token_text = NULL;
    int max_new = DEFAULT_MAX_NEW;
    int use_gpu = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (is_console_redir_token(argv[i])) {
            continue;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --input\n");
                print_usage(argv[0]);
                return 1;
            }
            input_text = argv[++i];
        } else if (strcmp(argv[i], "--token") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --token\n");
                print_usage(argv[0]);
                return 1;
            }
            token_text = argv[++i];
        } else if (strcmp(argv[i], "--max-new") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --max-new\n");
                print_usage(argv[0]);
                return 1;
            }
            max_new = atoi(argv[++i]);
            if (max_new <= 0) max_new = DEFAULT_MAX_NEW;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("%s v%s | model %s\n", ENGINE_NAME, ENGINE_VERSION, model_dir);
    if (verbose) {
        const char *logpath = init_verbose_log();
        if (!logpath) {
            fprintf(stderr, "warning: cannot open run log, continuing quiet\n");
        } else {
            atexit(cp_log_close);
            printf("verbose log: %s\n", logpath);
        }
        cp_log_begin("log");
        cp_log_u("version", 1);
        cp_log_end();
        cp_log_begin("model");
        cp_log_s("dir", model_dir, strlen(model_dir));
        cp_log_end();
    }

    /* ---- direct vocab lookup: tokenizer only, no weights loaded ---- */
    if (token_text) {
        cp_tok_t *tok = NULL;
        if (cp_tok_load(&tok, model_dir) != 0) {
            printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
            return 6;
        }
        uint32_t id = 0;
        int rc = 0;
        if (cp_tok_lookup(tok, token_text, &id) == 0) {
            printf("token \"%s\" -> id %u\n", token_text, id);
        } else {
            /* not a single vocab entry: show how it encodes as text */
            size_t cap = strlen(token_text) * 4 + 16;
            uint32_t *ids = malloc(cap * sizeof *ids);
            if (!ids) {
                printf("Result: FAIL - out of memory.\n");
                cp_tok_free(tok);
                return 6;
            }
            int n = cp_tok_encode(tok, token_text, ids, cap);
            if (n < 0) {
                printf("Result: FAIL - tokenization failed (see stderr).\n");
                free(ids);
                cp_tok_free(tok);
                return 6;
            }
            printf("token \"%s\" not in vocab as-is; as text ->", token_text);
            for (int i = 0; i < n; i++) printf(" %u", ids[i]);
            printf("\n");
            free(ids);
        }
        cp_tok_free(tok);
        return rc;
    }

    /* ---- stage 1 gate: raw files present (silent unless missing) ---- */
    int ok = 1;
    ok &= check_file(model_dir, "config.json") == 0;
    ok &= check_file(model_dir, "model.safetensors") == 0;
    ok &= check_file(model_dir, "tokenizer.json") == 0;
    ok &= check_file(model_dir, "tokenizer_config.json") == 0;
    ok &= check_file(model_dir, "vocab.json") == 0;
    ok &= check_file(model_dir, "merges.txt") == 0;
    if (!ok) {
        printf("Result: FAIL - some model files are missing.\n");
        printf("Hint  : run scripts/download_model.ps1 to fetch the model.\n");
        return 2;
    }

    /* ---- stage 3: config.json -> architecture hyperparameters ---- */
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    cp_config_t cfg;
    if (cp_config_load(config_path, &cfg) != 0) {
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }
    printf("arch: %s vocab=%u hidden=%u inter=%u layers=%u heads=%u/%u\n",
           cfg.model_type, cfg.vocab_size, cfg.hidden_size,
           cfg.intermediate_size, cfg.num_hidden_layers,
           cfg.num_attention_heads, cfg.num_key_value_heads);
    if (verbose) {
        cp_log_begin("arch");
        cp_log_s("model", cfg.model_type, strlen(cfg.model_type));
        cp_log_u("vocab", cfg.vocab_size);
        cp_log_u("hidden", cfg.hidden_size);
        cp_log_u("inter", cfg.intermediate_size);
        cp_log_u("layers", cfg.num_hidden_layers);
        cp_log_u("qheads", cfg.num_attention_heads);
        cp_log_u("kvheads", cfg.num_key_value_heads);
        cp_log_end();
    }

    /* ---- stage 2: parse the safetensors header, build the tensor map ---- */
    char weights_path[512];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);

    st_file_t *f = st_load(weights_path);
    if (!f) {
        printf("Result: FAIL - safetensors parsing failed (see stderr).\n");
        return 4;
    }

    {
        uint64_t total_params = 0, total_bytes = 0;
        for (size_t i = 0; i < f->n_tensors; i++) {
            uint64_t prod = 1;
            for (size_t d = 0; d < f->tensors[i].ndim; d++)
                prod *= f->tensors[i].shape[d];
            total_params += prod;
            total_bytes += f->tensors[i].nbytes;
        }
        printf("weights: %zu tensors, %.1fM params, %.1f MB\n",
               f->n_tensors, total_params / 1.0e6,
               total_bytes / (1024.0 * 1024.0));
        if (verbose) {
            cp_log_begin("weights");
            cp_log_u("tensors", f->n_tensors);
            cp_log_u("params", total_params);
            cp_log_u("bytes", total_bytes);
            cp_log_end();
        }
    }

    /* ---- stage 3b: bind every tensor onto the architecture ---- */
    cp_weights_t w;
    if (cp_weights_bind(&cfg, f, &w) != 0) {
        printf("Result: FAIL - weight binding failed (see stderr).\n");
        st_free(f);
        return 5;
    }
    printf("bind: %zu/%zu tensors OK%s\n",
           w.n_bound, w.n_expected, w.tied_head ? " (tied head)" : "");
    if (verbose) {
        cp_log_begin("bind");
        cp_log_u("bound", w.n_bound);
        cp_log_u("expected", w.n_expected);
        cp_log_u("tied", w.tied_head ? 1u : 0u);
        cp_log_end();
    }

    /* ---- user text -> token ids -> embedding vectors ---- */
    if (input_text) {
        printf("input: %s\n", input_text);
        if (verbose) {
            cp_log_begin("input");
            cp_log_s("text", input_text, strlen(input_text));
            cp_log_end();
        }
        cp_tok_t *tok = NULL;
        if (cp_tok_load(&tok, model_dir) != 0) {
            printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
            cp_weights_free(&w);
            st_free(f);
            return 6;
        }
        size_t cap = strlen(input_text) * 4 + 16;
        uint32_t *ids = malloc(cap * sizeof *ids);
        if (!ids) {
            printf("Result: FAIL - out of memory.\n");
            cp_tok_free(tok);
            cp_weights_free(&w);
            st_free(f);
            return 6;
        }
        int n = cp_tok_encode(tok, input_text, ids, cap);
        if (n < 0) {
            printf("Result: FAIL - tokenization failed (see stderr).\n");
            free(ids);
            cp_tok_free(tok);
            cp_weights_free(&w);
            st_free(f);
            return 6;
        }
        printf("tokens (%d):", n);
        for (int i = 0; i < n; i++) printf(" %u", ids[i]);
        printf("\n");
        if (verbose) {
            cp_log_begin("tokens");
            cp_log_u("count", (unsigned long long)n);
            cp_log_end();
        }
        /* per-token surface forms straight from the vocab */
        {
            char *piece = malloc(strlen(input_text) * 3 + 16);
            if (!piece) {
                printf("Result: FAIL - out of memory.\n");
                free(ids);
                cp_tok_free(tok);
                cp_weights_free(&w);
                st_free(f);
                return 6;
            }
            printf("pieces (%d):", n);
            for (int i = 0; i < n; i++) {
                if (cp_tok_decode(tok, &ids[i], 1, piece,
                                  strlen(input_text) * 3 + 16) < 0) {
                    printf(" [%u]<?>", ids[i]);
                    continue;
                }
                printf(" [%u]\"", ids[i]);
                for (const char *p = piece; *p; p++) {
                    if (*p == '\n') printf("\\n");
                    else if (*p == '\r') printf("\\r");
                    else if (*p == '\t') printf("\\t");
                    else if (*p == '"') printf("\\\"");
                    else putchar(*p);
                }
                putchar('"');
                if (verbose) {
                    cp_log_begin("token");
                    cp_log_u("index", (unsigned long long)i);
                    cp_log_u("id", ids[i]);
                    cp_log_s("text", piece, strlen(piece));
                    cp_log_end();
                }
            }
            printf("\n");
            free(piece);
        }
        if (n > 0) {
            /* ---- prefill + greedy decode (base model: raw continuation) ---- */
            cp_forward_t *fw = NULL;
            if (cp_forward_init(&fw, &cfg, &w, use_gpu) != 0) {
                printf("Result: FAIL - forward init failed (see stderr).\n");
                free(ids);
                cp_tok_free(tok);
                cp_weights_free(&w);
                st_free(f);
                return 7;
            }
            if (use_gpu) {
                printf("forward: %s\n",
                       cp_forward_on_gpu(fw) ? "gpu" : "cpu (gpu init failed)");
            }
            const float *logits = NULL;
            for (int i = 0; i < n; i++) {
                logits = cp_forward_step(fw, ids[i], (size_t)i);
            }
            if (!logits) {
                printf("Result: FAIL - forward prefill failed.\n");
                cp_forward_free(fw);
                free(ids);
                cp_tok_free(tok);
                cp_weights_free(&w);
                st_free(f);
                return 7;
            }
            uint32_t next = argmax(logits, cfg.vocab_size);
            size_t pos = (size_t)n;
            printf("reply: ");
            for (int g = 0; g < max_new && logits; g++) {
                if (next == cfg.eos_token_id || next == IM_END_ID) break;
                char piece[64];
                if (cp_tok_decode(tok, &next, 1, piece, sizeof piece) >= 0) {
                    printf("%s", piece);
                    fflush(stdout);
                    if (verbose) {
                        cp_log_begin("gen");
                        cp_log_u("pos", (unsigned long long)pos);
                        cp_log_u("id", next);
                        cp_log_s("text", piece, strlen(piece));
                        cp_log_end();
                    }
                }
                logits = cp_forward_step(fw, next, pos++);
                if (!logits) break;
                next = argmax(logits, cfg.vocab_size);
            }
            printf("\n");
            cp_forward_free(fw);
        }
        free(ids);
        cp_tok_free(tok);
    }
    cp_weights_free(&w);

    st_free(f);

    printf("OK\n");
    return 0;
}
