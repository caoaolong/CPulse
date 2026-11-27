/*
 * command.c - CLI commands (see command.h)
 *
 * Shared model load path is used by summary and generate. Token lookup
 * only needs the tokenizer and never touches weights.
 */

#include "command.h"
#include "arch.h"
#include "qwen/generate.h"
#include "qwen/model.h"
#include "qwen/weights.h"
#include "view.h"
#include "safetensors.h"
#include "t5/config.h"
#include "t5/weights.h"
#include "tok/tokenizer.h"

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
    cp_tokenizer_t *tok = (cp_tokenizer_t *)userdata;
    char piece[64];
    (void)pos;
    if (cp_tokenizer_decode(tok, &id, 1, piece, sizeof piece, 0) >= 0) {
        printf("%s", piece);
        fflush(stdout);
    }
}

/* Loaded model bundle shared by summary / generate. */
typedef struct {
    int              is_t5;
    cp_config_t      cfg;     /* Qwen / decoder-only */
    cp_t5_config_t   t5_cfg;  /* T5 / encoder-decoder */
    st_file_t       *st;
    cp_weights_t     w;       /* Qwen */
    cp_t5_weights_t  t5_w;    /* T5 */
} model_bundle_t;

static void model_bundle_free(model_bundle_t *m) {
    if (!m) return;
    if (m->is_t5)
        cp_t5_weights_free(&m->t5_w);
    else
        cp_weights_free(&m->w);
    st_free(m->st);
    m->st = NULL;
}

/*
 * Gate files, load config + safetensors, bind weights.
 * On failure prints Result line and returns process exit code (>0).
 * On success returns 0; caller must model_bundle_free().
 */
static void print_st_summary(const st_file_t *st) {
    uint64_t total_params = 0, total_bytes = 0;
    for (size_t i = 0; i < st->n_tensors; i++) {
        uint64_t prod = 1;
        for (size_t d = 0; d < st->tensors[i].ndim; d++)
            prod *= st->tensors[i].shape[d];
        total_params += prod;
        total_bytes += st->tensors[i].nbytes;
    }
    printf("weights: %zu tensors, %.1fM params, %.1f MB\n",
           st->n_tensors, total_params / 1.0e6,
           total_bytes / (1024.0 * 1024.0));
}

/*
 * Encoder-decoder / T5 path: dedicated config + tokenizer smoke + weight bind.
 */
static int load_model_t5(model_bundle_t *m, const char *model_dir,
                         const char *config_path, const cp_cli_t *cli) {
    (void)cli;
    m->is_t5 = 1;
    if (cp_t5_config_load(config_path, &m->t5_cfg) != 0) {
        printf("Result: FAIL - T5 config loading failed (see stderr).\n");
        return 3;
    }
    printf("arch: %s (%s) vocab=%u d_model=%u d_ff=%u layers=%u/%u heads=%u\n",
           m->t5_cfg.model_type, cp_arch_name(m->t5_cfg.arch),
           m->t5_cfg.vocab_size, m->t5_cfg.d_model, m->t5_cfg.d_ff,
           m->t5_cfg.num_layers, m->t5_cfg.num_decoder_layers,
           m->t5_cfg.num_heads);
    cp_t5_config_print(&m->t5_cfg);

    cp_tokenizer_t *tok = NULL;
    if (cp_tokenizer_load(&tok, model_dir) != 0) {
        printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
        return 6;
    }
    {
        static const char *sample = "translate English to German: Hello";
        uint32_t ids[64];
        int n = cp_tokenizer_encode(tok, sample, ids, 64, 1);
        printf("tok[%s]: vocab=%zu sample \"%s\" ->",
               cp_tok_strategy_name(cp_tokenizer_strategy(tok)),
               cp_tokenizer_vocab_size(tok), sample);
        if (n < 0) {
            printf(" <encode failed>\n");
            cp_tokenizer_free(tok);
            printf("Result: FAIL - tokenization failed (see stderr).\n");
            return 6;
        }
        for (int i = 0; i < n; i++) printf(" %u", ids[i]);
        printf("\n");
    }
    cp_tokenizer_free(tok);

    char weights_path[512];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors",
             model_dir);
    m->st = st_load(weights_path);
    if (!m->st) {
        printf("Result: FAIL - safetensors parsing failed (see stderr).\n");
        return 4;
    }
    print_st_summary(m->st);

    if (cp_t5_weights_bind(&m->t5_cfg, m->st, &m->t5_w) != 0) {
        printf("Result: FAIL - T5 weight binding failed (see stderr).\n");
        st_free(m->st);
        m->st = NULL;
        return 5;
    }
    cp_t5_weights_print(&m->t5_w);
    printf("bind: %zu/%zu tensors OK (T5, dtype=%s)\n",
           m->t5_w.n_bound, m->t5_w.n_expected,
           st_dtype_name(m->t5_w.dtype));
    return 0;
}

static int load_model(model_bundle_t *m, const cp_cli_t *cli) {
    const char *model_dir = cli->model_dir;
    memset(m, 0, sizeof(*m));

    if (check_model_files(model_dir) != 0) return 2;

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    char model_type[32];
    if (cp_config_peek_model_type(config_path, model_type,
                                  sizeof model_type) != 0) {
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }
    const cp_arch_t *arch = cp_arch_resolve(model_type);
    if (!arch) {
        fprintf(stderr, "cp_config: unknown model_type '%s'\n", model_type);
        cp_arch_print_registry(stderr);
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }
    if (!arch->implemented) {
        fprintf(stderr,
                "cp_config: strategy '%s' recognized from model_type='%s', "
                "but not implemented yet\n",
                arch->name, model_type);
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }

    /* Strategy open for Enc–Dec, but only the T5-family loader is wired. */
    if (arch->id == CP_ARCH_ENCODER_DECODER)
        return load_model_t5(m, model_dir, config_path, cli);

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
    print_st_summary(m->st);

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
    printf("Usage: %s [--model <model_dir>] [--arch-config <path>] "
           "[--input <text>] [--token <token>] [--view [out.svg]] "
           "[--max-new <n>] [--gpu]\n", prog);
    printf("  --model <model_dir>       path to the model directory\n");
    printf("                            (default: %s)\n", CP_DEFAULT_MODEL_DIR);
    printf("  --arch-config <path>      model_type -> strategy routes JSON\n");
    printf("                            (default: %s)\n", CP_DEFAULT_ARCH_CONFIG);
    printf("  --input <text>            greedy-generate a continuation of <text>\n");
    printf("  --token <token>           print the vocab id of one surface-form token\n");
    printf("  --view [out.svg]          write model-structure SVG "
           "(default: <model_name>.svg)\n");
    printf("  --max-new <n>             tokens to generate (default %d)\n", CP_DEFAULT_MAX_NEW);
    printf("  --gpu                     run the forward pass on CUDA (falls back to cpu)\n");
}

/* Basename of model_dir -> "<name>.svg" into out (cap bytes). */
static void view_default_svg_path(const char *model_dir, char *out, size_t cap) {
    const char *base = model_dir ? model_dir : "model";
    /* skip trailing / and \\ */
    size_t n = strlen(base);
    while (n > 0 && (base[n - 1] == '/' || base[n - 1] == '\\')) n--;
    size_t start = 0;
    for (size_t i = 0; i < n; i++) {
        if (base[i] == '/' || base[i] == '\\') start = i + 1;
    }
    if (start >= n) {
        snprintf(out, cap, "model.svg");
        return;
    }
    size_t len = n - start;
    if (len + 5 > cap) len = cap > 5 ? cap - 5 : 0; /* room for ".svg\0" */
    memcpy(out, base + start, len);
    out[len] = '\0';
    snprintf(out + len, cap - len, ".svg");
}

/*
 * --view only needs config.json (no weight bind). Writes one rect per
 * block with shape annotations.
 */
static int cmd_view(const cp_cli_t *cli) {
    char default_path[256];
    view_default_svg_path(cli->model_dir, default_path, sizeof default_path);
    const char *out_path =
        (cli->view_path && cli->view_path[0]) ? cli->view_path : default_path;

    char config_path[512];
    snprintf(config_path, sizeof config_path, "%s/config.json", cli->model_dir);

    char model_type[32];
    if (cp_config_peek_model_type(config_path, model_type, sizeof model_type) != 0) {
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }
    const cp_arch_t *arch = cp_arch_resolve(model_type);
    if (!arch) {
        fprintf(stderr, "cp_config: unknown model_type '%s'\n", model_type);
        cp_arch_print_registry(stderr);
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }

    int rc = -1;
    if (arch->id == CP_ARCH_ENCODER_DECODER) {
        cp_t5_config_t t5;
        if (cp_t5_config_load(config_path, &t5) != 0) {
            printf("Result: FAIL - T5 config loading failed (see stderr).\n");
            return 3;
        }
        rc = cp_model_view_t5(&t5, out_path);
    } else if (arch->id == CP_ARCH_DECODER_ONLY) {
        cp_config_t cfg;
        if (cp_config_load(config_path, &cfg) != 0) {
            printf("Result: FAIL - config loading failed (see stderr).\n");
            return 3;
        }
        rc = cp_model_view_qwen(&cfg, out_path);
    } else {
        fprintf(stderr, "view: strategy '%s' has no diagram yet\n",
                arch->name);
        printf("Result: FAIL - model view not supported for this strategy.\n");
        return 3;
    }

    if (rc != 0) {
        printf("Result: FAIL - could not write SVG (see stderr).\n");
        return 8;
    }
    printf("view: wrote %s\n", out_path);
    printf("OK\n");
    return 0;
}

static int cmd_token(const cp_cli_t *cli) {
    cp_tokenizer_t *tok = NULL;
    if (cp_tokenizer_load(&tok, cli->model_dir) != 0) {
        printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
        return 6;
    }
    printf("tok[%s]: vocab=%zu pad=%u eos=%u unk=%u\n",
           cp_tok_strategy_name(cp_tokenizer_strategy(tok)),
           cp_tokenizer_vocab_size(tok), cp_tokenizer_pad_id(tok),
           cp_tokenizer_eos_id(tok), cp_tokenizer_unk_id(tok));

    uint32_t id = 0;
    if (cp_tokenizer_lookup(tok, cli->token_text, &id) == 0) {
        printf("token \"%s\" -> id %u\n", cli->token_text, id);
        cp_tokenizer_free(tok);
        return 0;
    }

    size_t cap = strlen(cli->token_text) * 4 + 16;
    uint32_t *ids = malloc(cap * sizeof *ids);
    if (!ids) {
        printf("Result: FAIL - out of memory.\n");
        cp_tokenizer_free(tok);
        return 6;
    }
    int add_eos = (cp_tokenizer_strategy(tok) == CP_TOK_UNIGRAM) ? 1 : 0;
    int n = cp_tokenizer_encode(tok, cli->token_text, ids, cap, add_eos);
    if (n < 0) {
        printf("Result: FAIL - tokenization failed (see stderr).\n");
        free(ids);
        cp_tokenizer_free(tok);
        return 6;
    }
    printf("encode \"%s\" ->", cli->token_text);
    for (int i = 0; i < n; i++) printf(" %u", ids[i]);
    printf("\n");

    char dec[1024];
    if (cp_tokenizer_decode(tok, ids, (size_t)n, dec, sizeof dec, 1) >= 0)
        printf("decode: \"%s\"\n", dec);

    free(ids);
    cp_tokenizer_free(tok);
    return 0;
}

static int cmd_summary(const cp_cli_t *cli) {
    model_bundle_t m;
    int rc = load_model(&m, cli);
    if (rc != 0) return rc;
    model_bundle_free(&m);
    printf("OK\n");
    return 0;
}

static int cmd_generate(const cp_cli_t *cli) {
    model_bundle_t m;
    int rc = load_model(&m, cli);
    if (rc != 0) return rc;

    if (m.is_t5) {
        printf("Result: FAIL - T5 generate not implemented yet "
               "(weight bind OK; next: ops / encoder / decoder).\n");
        model_bundle_free(&m);
        return 7;
    }

    printf("input: %s\n", cli->input_text);

    cp_tokenizer_t *tok = NULL;
    if (cp_tokenizer_load(&tok, cli->model_dir) != 0) {
        printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
        model_bundle_free(&m);
        return 6;
    }

    size_t cap = strlen(cli->input_text) * 4 + 16;
    uint32_t *ids = malloc(cap * sizeof *ids);
    if (!ids) {
        printf("Result: FAIL - out of memory.\n");
        cp_tokenizer_free(tok);
        model_bundle_free(&m);
        return 6;
    }
    int n = cp_tokenizer_encode(tok, cli->input_text, ids, cap, 0);
    if (n < 0) {
        printf("Result: FAIL - tokenization failed (see stderr).\n");
        free(ids);
        cp_tokenizer_free(tok);
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
            cp_tokenizer_free(tok);
            model_bundle_free(&m);
            return 6;
        }
        size_t piece_cap = strlen(cli->input_text) * 3 + 16;
        printf("pieces (%d):", n);
        for (int i = 0; i < n; i++) {
            if (cp_tokenizer_decode(tok, &ids[i], 1, piece, piece_cap, 0) < 0) {
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
            cp_tokenizer_free(tok);
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
            cp_tokenizer_free(tok);
            model_bundle_free(&m);
            return 7;
        }
        cp_session_free(sess);
    }

    free(ids);
    cp_tokenizer_free(tok);
    model_bundle_free(&m);
    printf("OK\n");
    return 0;
}

int cp_command_run(const cp_cli_t *cli) {
    if (!cli || !cli->model_dir) return 1;

    /* Token lookup does not need architecture routes; load them for
     * summary / generate so model_type can be classified. */
    if (!cli->token_text) {
        const char *routes = cli->arch_config ? cli->arch_config
                                              : CP_DEFAULT_ARCH_CONFIG;
        if (cp_arch_routes_load(routes) != 0) {
            printf("Result: FAIL - architecture routes loading failed "
                   "(see stderr).\n");
            return 1;
        }
    }

    if (cli->view_path) return cmd_view(cli);
    if (cli->token_text) return cmd_token(cli);
    if (cli->input_text) return cmd_generate(cli);
    return cmd_summary(cli);
}
