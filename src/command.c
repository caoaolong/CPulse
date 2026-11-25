/*
 * command.c - CLI commands (see command.h)
 *
 * Shared model load path is used by summary and generate. Token lookup
 * only needs the tokenizer and never touches weights.
 */

#include "command.h"
#include "arch.h"
#include "generate.h"
#include "model.h"
#include "safetensors.h"
#include "tokenizer.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- small helpers ------------------------------------------------ */

static int file_exists(const char *base, const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", base, name);
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

static int check_file(const char *base, const char *name) {
    if (file_exists(base, name)) return 0;
    printf("  [MISS] %s/%s\n", base, name);
    return -1;
}

static int check_model_files(const char *model_dir) {
    int ok = 1;
    ok &= check_file(model_dir, "config.json") == 0;
    ok &= check_file(model_dir, "model.safetensors") == 0;
    {
        int has_tok_json = file_exists(model_dir, "tokenizer.json");
        int has_bpe = file_exists(model_dir, "vocab.json") &&
                      file_exists(model_dir, "merges.txt");
        int has_spm = file_exists(model_dir, "spiece.model");
        if (!has_tok_json && !has_bpe && !has_spm) {
            printf("  [MISS] tokenizer (need tokenizer.json, or "
                   "vocab.json+merges.txt, or spiece.model)\n");
            ok = 0;
        }
    }
    if (!ok) {
        printf("Result: FAIL - some model files are missing.\n");
        printf("Hint  : run scripts/download_model.ps1 to fetch the model.\n");
        return -1;
    }
    return 0;
}

static void on_gen_token(uint32_t id, size_t pos, void *userdata) {
    cp_tok_t *tok = (cp_tok_t *)userdata;
    char piece[64];
    (void)pos;
    if (cp_tok_decode(tok, &id, 1, piece, sizeof piece) >= 0) {
        printf("%s", piece);
        fflush(stdout);
    }
}

/* Loaded model bundle shared by summary / generate. */
typedef struct {
    cp_config_t  cfg;
    st_file_t   *st;
    cp_weights_t w;
} model_bundle_t;

static void model_bundle_free(model_bundle_t *m) {
    if (!m) return;
    cp_weights_free(&m->w);
    st_free(m->st);
    m->st = NULL;
}

/*
 * Gate files, load config + safetensors, bind weights.
 * On failure prints Result line and returns process exit code (>0).
 * On success returns 0; caller must model_bundle_free().
 */
static int load_model(model_bundle_t *m, const char *model_dir) {
    memset(m, 0, sizeof(*m));

    if (check_model_files(model_dir) != 0) return 2;

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    if (cp_config_load(config_path, &m->cfg) != 0) {
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }
    printf("arch: %s (%s) vocab=%u hidden=%u inter=%u layers=%u heads=%u/%u\n",
           m->cfg.model_type, cp_arch_name(m->cfg.arch), m->cfg.vocab_size,
           m->cfg.hidden_size, m->cfg.intermediate_size, m->cfg.num_hidden_layers,
           m->cfg.num_attention_heads, m->cfg.num_key_value_heads);

    char weights_path[512];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);
    m->st = st_load(weights_path);
    if (!m->st) {
        printf("Result: FAIL - safetensors parsing failed (see stderr).\n");
        return 4;
    }

    {
        uint64_t total_params = 0, total_bytes = 0;
        for (size_t i = 0; i < m->st->n_tensors; i++) {
            uint64_t prod = 1;
            for (size_t d = 0; d < m->st->tensors[i].ndim; d++)
                prod *= m->st->tensors[i].shape[d];
            total_params += prod;
            total_bytes += m->st->tensors[i].nbytes;
        }
        printf("weights: %zu tensors, %.1fM params, %.1f MB\n",
               m->st->n_tensors, total_params / 1.0e6,
               total_bytes / (1024.0 * 1024.0));
    }

    if (cp_weights_bind(&m->cfg, m->st, &m->w) != 0) {
        printf("Result: FAIL - weight binding failed (see stderr).\n");
        st_free(m->st);
        m->st = NULL;
        return 5;
    }
    printf("bind: %zu/%zu tensors OK%s\n",
           m->w.n_bound, m->w.n_expected, m->w.tied_head ? " (tied head)" : "");
    return 0;
}

/* ---- commands ----------------------------------------------------- */

void cp_cmd_help(const char *prog) {
    printf("Usage: %s [--model <model_dir>] [--input <text>] [--token <token>] "
           "[--max-new <n>] [--gpu]\n", prog);
    printf("  --model <model_dir>   path to the model directory\n");
    printf("                        (default: %s)\n", CP_DEFAULT_MODEL_DIR);
    printf("  --input <text>        greedy-generate a continuation of <text>\n");
    printf("  --token <token>       print the vocab id of one surface-form token\n");
    printf("  --max-new <n>         tokens to generate (default %d)\n", CP_DEFAULT_MAX_NEW);
    printf("  --gpu                 run the forward pass on CUDA (falls back to cpu)\n");
}

static int cmd_token(const cp_cli_t *cli) {
    cp_tok_t *tok = NULL;
    if (cp_tok_load(&tok, cli->model_dir) != 0) {
        printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
        return 6;
    }
    uint32_t id = 0;
    int rc = 0;
    if (cp_tok_lookup(tok, cli->token_text, &id) == 0) {
        printf("token \"%s\" -> id %u\n", cli->token_text, id);
    } else {
        size_t cap = strlen(cli->token_text) * 4 + 16;
        uint32_t *ids = malloc(cap * sizeof *ids);
        if (!ids) {
            printf("Result: FAIL - out of memory.\n");
            cp_tok_free(tok);
            return 6;
        }
        int n = cp_tok_encode(tok, cli->token_text, ids, cap);
        if (n < 0) {
            printf("Result: FAIL - tokenization failed (see stderr).\n");
            free(ids);
            cp_tok_free(tok);
            return 6;
        }
        printf("token \"%s\" not in vocab as-is; as text ->", cli->token_text);
        for (int i = 0; i < n; i++) printf(" %u", ids[i]);
        printf("\n");
        free(ids);
    }
    cp_tok_free(tok);
    return rc;
}

static int cmd_summary(const cp_cli_t *cli) {
    model_bundle_t m;
    int rc = load_model(&m, cli->model_dir);
    if (rc != 0) return rc;
    model_bundle_free(&m);
    printf("OK\n");
    return 0;
}

static int cmd_generate(const cp_cli_t *cli) {
    model_bundle_t m;
    int rc = load_model(&m, cli->model_dir);
    if (rc != 0) return rc;

    printf("input: %s\n", cli->input_text);

    cp_tok_t *tok = NULL;
    if (cp_tok_load(&tok, cli->model_dir) != 0) {
        printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
        model_bundle_free(&m);
        return 6;
    }

    size_t cap = strlen(cli->input_text) * 4 + 16;
    uint32_t *ids = malloc(cap * sizeof *ids);
    if (!ids) {
        printf("Result: FAIL - out of memory.\n");
        cp_tok_free(tok);
        model_bundle_free(&m);
        return 6;
    }
    int n = cp_tok_encode(tok, cli->input_text, ids, cap);
    if (n < 0) {
        printf("Result: FAIL - tokenization failed (see stderr).\n");
        free(ids);
        cp_tok_free(tok);
        model_bundle_free(&m);
        return 6;
    }
    printf("tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" %u", ids[i]);
    printf("\n");

    {
        char *piece = malloc(strlen(cli->input_text) * 3 + 16);
        if (!piece) {
            printf("Result: FAIL - out of memory.\n");
            free(ids);
            cp_tok_free(tok);
            model_bundle_free(&m);
            return 6;
        }
        printf("pieces (%d):", n);
        for (int i = 0; i < n; i++) {
            if (cp_tok_decode(tok, &ids[i], 1, piece,
                              strlen(cli->input_text) * 3 + 16) < 0) {
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
        }
        printf("\n");
        free(piece);
    }

    if (n > 0) {
        cp_session_t *sess = NULL;
        if (cp_session_init(&sess, &m.cfg, &m.w, cli->use_gpu) != 0) {
            printf("Result: FAIL - session init failed (see stderr).\n");
            free(ids);
            cp_tok_free(tok);
            model_bundle_free(&m);
            return 7;
        }
        if (cli->use_gpu) {
            printf("forward: %s\n",
                   cp_session_on_gpu(sess) ? "gpu" : "cpu (gpu init failed)");
        }
        printf("reply: ");
        int n_gen = cp_generate(sess, ids, (size_t)n,
                                NULL, 0, cli->max_new,
                                on_gen_token, tok);
        printf("\n");
        if (n_gen < 0) {
            printf("Result: FAIL - generate failed (see stderr).\n");
            cp_session_free(sess);
            free(ids);
            cp_tok_free(tok);
            model_bundle_free(&m);
            return 7;
        }
        cp_session_free(sess);
    }

    free(ids);
    cp_tok_free(tok);
    model_bundle_free(&m);
    printf("OK\n");
    return 0;
}

int cp_command_run(const cp_cli_t *cli) {
    if (!cli || !cli->model_dir) return 1;
    if (cli->token_text) return cmd_token(cli);
    if (cli->input_text) return cmd_generate(cli);
    return cmd_summary(cli);
}
