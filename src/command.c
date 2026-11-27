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
#include "t5/encoder.h"
#include "t5/generate.h"
#include "t5/weights.h"
#include "tok/tokenizer.h"
#include "bind_recipe.h"
#include "cJSON.h"

#include <math.h>
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

/* T5 streaming: skip special pieces (e.g. </s>) for readable reply. */
static void on_gen_token_t5(uint32_t id, size_t pos, void *userdata) {
    cp_tokenizer_t *tok = (cp_tokenizer_t *)userdata;
    char piece[64];
    (void)pos;
    if (cp_tokenizer_decode(tok, &id, 1, piece, sizeof piece,
                            /*skip_special=*/1) >= 0) {
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
 * Encoder-decoder / T5 path: dedicated config + weight bind.
 * Tokenizer is loaded later by --token / generate, not here.
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

    /* 1. 检查模型目录必要文件是否存在（非深度完整性校验） */
    if (check_model_files(model_dir) != 0) return 2;

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    /* 2. 仅窥读 model_type（完整 config 在分策略后才加载） */
    char model_type[32];
    if (cp_config_peek_model_type(config_path, model_type,
                                  sizeof model_type) != 0) {
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }

    /* 3. 经 routes.json 的 arch 节解析架构策略；未知或未实现则失败 */
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

    /* 4. 按架构分派：Enc–Dec 走 T5 专用加载（config + safetensors + bind） */
    if (arch->id == CP_ARCH_ENCODER_DECODER)
        return load_model_t5(m, model_dir, config_path, cli);

    /* 5. decoder-only：加载完整 Qwen 风格 config */
    if (cp_config_load(config_path, &m->cfg) != 0) {
        printf("Result: FAIL - config loading failed (see stderr).\n");
        return 3;
    }
    printf("arch: %s (%s) vocab=%u hidden=%u inter=%u layers=%u heads=%u/%u\n",
           m->cfg.model_type, cp_arch_name(m->cfg.arch), m->cfg.vocab_size,
           m->cfg.hidden_size, m->cfg.intermediate_size, m->cfg.num_hidden_layers,
           m->cfg.num_attention_heads, m->cfg.num_key_value_heads);

    /* 6. 映射并解析 model.safetensors（张量仍在文件缓冲里） */
    char weights_path[512];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);
    m->st = st_load(weights_path);
    if (!m->st) {
        printf("Result: FAIL - safetensors parsing failed (see stderr).\n");
        return 4;
    }
    print_st_summary(m->st);

    /* 7. 按名字把张量绑到结构体指针（零拷贝） */
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
    printf("Usage: %s [--model <model_dir>] [--routes <path>] "
           "[--input <text>] [--token <token>] [--view [out.svg]] "
           "[--max-new <n>] [--temp <f>] [--top-k <n>] [--top-p <f>] "
           "[--seed <u>] [--gpu]\n", prog);
    printf("  --model <model_dir>       path to the model directory\n");
    printf("                            (default: %s)\n", CP_DEFAULT_MODEL_DIR);
    printf("  --routes <path>           unified routes JSON (arch + tokenizer)\n");
    printf("                            (default: %s)\n", CP_DEFAULT_ROUTES);
    printf("  --input <text>            generate a continuation of <text>\n");
    printf("  --token <token>           print the vocab id of one surface-form token\n");
    printf("  --view [out.svg]          write model-structure SVG "
           "(default: <model_name>.svg)\n");
    printf("  --max-new <n>             tokens to generate (default %d)\n",
           CP_DEFAULT_MAX_NEW);
    printf("  --temp / --temperature <f>  sampling temperature "
           "(default 0 = greedy)\n");
    printf("  --top-k <n>               keep top-k logits (0 = off, 1 = greedy)\n");
    printf("  --top-p <f>               nucleus sampling (1 = off)\n");
    printf("  --seed <u>                RNG seed when not greedy\n");
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

/*
 * Compare encoder memory against gold/t5-small.json probe when src ids match.
 * Returns 0 on match / no comparable gold, -1 on numeric fail, -2 on I/O.
 */
static int t5_encoder_check_gold(const float *mem, size_t seq, size_t d_model,
                                 const uint32_t *ids, size_t n) {
    const char *path = "gold/t5-small.json";
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("encoder gold: skip (no %s)\n", path);
        return 0;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        return -2;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(fp);
        return -2;
    }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf);
        fclose(fp);
        return -2;
    }
    fclose(fp);
    buf[sz] = '\0';

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return -2;

    const cJSON *gids = cJSON_GetObjectItemCaseSensitive(root, "src_token_ids");
    if (!cJSON_IsArray(gids) || (size_t)cJSON_GetArraySize(gids) != n) {
        printf("encoder gold: skip (src id length mismatch)\n");
        cJSON_Delete(root);
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        const cJSON *it = cJSON_GetArrayItem(gids, (int)i);
        if (!cJSON_IsNumber(it) || (uint32_t)it->valuedouble != ids[i]) {
            printf("encoder gold: skip (src ids differ from gold)\n");
            cJSON_Delete(root);
            return 0;
        }
    }

    const cJSON *probe =
            cJSON_GetObjectItemCaseSensitive(root, "encoder_hidden_probe");
    const cJSON *positions =
            probe ? cJSON_GetObjectItemCaseSensitive(probe, "positions") : NULL;
    if (!cJSON_IsArray(positions)) {
        cJSON_Delete(root);
        return -2;
    }

    float max_abs = 0.0f;
    int n_cmp = 0;
    const cJSON *pos;
    cJSON_ArrayForEach(pos, positions) {
        const cJSON *jp = cJSON_GetObjectItemCaseSensitive(pos, "pos");
        const cJSON *dims = cJSON_GetObjectItemCaseSensitive(pos, "dims0_7");
        if (!cJSON_IsNumber(jp) || !cJSON_IsArray(dims)) continue;
        size_t p = (size_t)jp->valuedouble;
        if (p >= seq) continue;
        int nd = cJSON_GetArraySize(dims);
        if (nd > 8) nd = 8;
        if ((size_t)nd > d_model) nd = (int)d_model;
        for (int d = 0; d < nd; d++) {
            const cJSON *v = cJSON_GetArrayItem(dims, d);
            if (!cJSON_IsNumber(v)) continue;
            float got = mem[p * d_model + (size_t)d];
            float err = fabsf(got - (float)v->valuedouble);
            if (err > max_abs) max_abs = err;
            n_cmp++;
        }
    }
    cJSON_Delete(root);

    printf("encoder gold: compared %d values, max_abs_err=%.6g\n", n_cmp,
           max_abs);
    /* HF FP32 vs our FP32: expect ~1e-5; allow 1e-3 slack for path diffs */
    if (n_cmp == 0 || max_abs > 1e-3f) {
        fprintf(stderr, "encoder gold: FAIL max_abs_err=%.6g\n", max_abs);
        return -1;
    }
    return 0;
}

/*
 * Load gold greedy.token_ids when src ids match. Returns count, or 0 if N/A.
 */
static size_t t5_load_gold_greedy(const uint32_t *src, size_t src_n,
                                  uint32_t *out, size_t out_cap) {
    FILE *fp = fopen("gold/t5-small.json", "rb");
    if (!fp) return 0;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        return 0;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf);
        fclose(fp);
        return 0;
    }
    fclose(fp);
    buf[sz] = '\0';

    size_t n_out = 0;
    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) return 0;

    const cJSON *gids = cJSON_GetObjectItemCaseSensitive(root, "src_token_ids");
    if (!cJSON_IsArray(gids) ||
        (size_t)cJSON_GetArraySize(gids) != src_n) {
        cJSON_Delete(root);
        return 0;
    }
    for (size_t i = 0; i < src_n; i++) {
        const cJSON *it = cJSON_GetArrayItem(gids, (int)i);
        if (!cJSON_IsNumber(it) || (uint32_t)it->valuedouble != src[i]) {
            cJSON_Delete(root);
            return 0;
        }
    }
    const cJSON *greedy = cJSON_GetObjectItemCaseSensitive(root, "greedy");
    const cJSON *toks =
            greedy ? cJSON_GetObjectItemCaseSensitive(greedy, "token_ids")
                   : NULL;
    if (cJSON_IsArray(toks)) {
        int nt = cJSON_GetArraySize(toks);
        if (nt > (int)out_cap) nt = (int)out_cap;
        for (int i = 0; i < nt; i++) {
            const cJSON *it = cJSON_GetArrayItem(toks, i);
            if (!cJSON_IsNumber(it)) {
                n_out = 0;
                break;
            }
            out[i] = (uint32_t)it->valuedouble;
            n_out = (size_t)i + 1;
        }
    }
    cJSON_Delete(root);
    return n_out;
}

static int cmd_generate_t5(const cp_cli_t *cli, model_bundle_t *m) {
    printf("input: %s\n", cli->input_text);

    cp_tokenizer_t *tok = NULL;
    if (cp_tokenizer_load(&tok, cli->model_dir) != 0) {
        printf("Result: FAIL - tokenizer loading failed (see stderr).\n");
        return 6;
    }

    size_t cap = strlen(cli->input_text) * 4 + 16;
    uint32_t *ids = malloc(cap * sizeof *ids);
    if (!ids) {
        printf("Result: FAIL - out of memory.\n");
        cp_tokenizer_free(tok);
        return 6;
    }
    int n = cp_tokenizer_encode(tok, cli->input_text, ids, cap, /*add_eos=*/1);
    if (n < 0) {
        printf("Result: FAIL - tokenization failed (see stderr).\n");
        free(ids);
        cp_tokenizer_free(tok);
        return 6;
    }
    printf("tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" %u", ids[i]);
    printf("\n");

    /* Optional encoder probe (same gold file) before generate. */
    {
        cp_t5_encoder_t *enc = NULL;
        if (cp_t5_encoder_init(&enc, &m->t5_cfg, &m->t5_w, (size_t)n + 8) ==
            0) {
            const float *mem = cp_t5_encoder_run(enc, ids, (size_t)n);
            if (mem) {
                int eg = t5_encoder_check_gold(mem, (size_t)n,
                                               m->t5_cfg.d_model, ids,
                                               (size_t)n);
                if (eg == -1) {
                    printf("Result: FAIL - T5 encoder mismatch vs gold.\n");
                    cp_t5_encoder_free(enc);
                    free(ids);
                    cp_tokenizer_free(tok);
                    return 7;
                }
            }
            cp_t5_encoder_free(enc);
        }
    }

    size_t max_new = (size_t)cli->max_new;
    if (max_new == 0) max_new = 16;
    cp_t5_session_t *sess = NULL;
    if (cp_t5_session_init(&sess, &m->t5_cfg, &m->t5_w, (size_t)n + 8,
                           max_new + 8) != 0) {
        printf("Result: FAIL - T5 session init failed (see stderr).\n");
        free(ids);
        cp_tokenizer_free(tok);
        return 7;
    }

    uint32_t *out_ids = malloc(max_new * sizeof *out_ids);
    if (!out_ids) {
        printf("Result: FAIL - out of memory.\n");
        cp_t5_session_free(sess);
        free(ids);
        cp_tokenizer_free(tok);
        return 6;
    }

    if (!cp_sample_cfg_is_greedy(&cli->sample)) {
        printf("sample: temp=%.3g top_k=%d top_p=%.3g seed=%llu\n",
               cli->sample.temperature, cli->sample.top_k, cli->sample.top_p,
               (unsigned long long)cli->sample.seed);
    }

    printf("reply: ");
    int n_gen = cp_t5_generate(sess, ids, (size_t)n, out_ids, max_new,
                               (int)max_new, &cli->sample, on_gen_token_t5,
                               tok);
    printf("\n");
    if (n_gen < 0) {
        printf("Result: FAIL - T5 generate failed (see stderr).\n");
        free(out_ids);
        cp_t5_session_free(sess);
        free(ids);
        cp_tokenizer_free(tok);
        return 7;
    }

    printf("gen tokens (%d):", n_gen);
    for (int i = 0; i < n_gen; i++) printf(" %u", out_ids[i]);
    printf("\n");

    uint32_t gold_out[16];
    size_t n_gold = t5_load_gold_greedy(ids, (size_t)n, gold_out,
                                        sizeof gold_out / sizeof gold_out[0]);
    int gold_ok = 1;
    if (n_gold > 0 && cp_sample_cfg_is_greedy(&cli->sample)) {
        if ((size_t)n_gen != n_gold) {
            printf("generate gold: FAIL len got %d expect %zu\n", n_gen,
                   n_gold);
            gold_ok = 0;
        } else {
            for (size_t i = 0; i < n_gold; i++) {
                if (out_ids[i] != gold_out[i]) {
                    printf("generate gold: FAIL at %zu got %u expect %u\n", i,
                           out_ids[i], gold_out[i]);
                    gold_ok = 0;
                    break;
                }
            }
            if (gold_ok)
                printf("generate gold: OK (%zu tokens match HF greedy)\n",
                       n_gold);
        }
    } else if (n_gold > 0) {
        printf("generate gold: skip (non-greedy sampling)\n");
    }

    free(out_ids);
    cp_t5_session_free(sess);
    free(ids);
    cp_tokenizer_free(tok);

    if (!gold_ok) {
        printf("Result: FAIL - T5 greedy mismatch vs gold.\n");
        return 7;
    }
    printf("OK\n");
    return 0;
}

static int cmd_generate(const cp_cli_t *cli) {
    model_bundle_t m;
    int rc = load_model(&m, cli);
    if (rc != 0) return rc;

    if (m.is_t5) {
        rc = cmd_generate_t5(cli, &m);
        model_bundle_free(&m);
        return rc;
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
        if (!cp_sample_cfg_is_greedy(&cli->sample)) {
            printf("sample: temp=%.3g top_k=%d top_p=%.3g seed=%llu\n",
                   cli->sample.temperature, cli->sample.top_k,
                   cli->sample.top_p,
                   (unsigned long long)cli->sample.seed);
        }
        printf("reply: ");
        int n_gen = cp_generate(sess, ids, (size_t)n, NULL, 0, cli->max_new,
                                &cli->sample, on_gen_token, tok);
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

    /* Unified routes: arch + tokenizer (+ bind recipes on demand). */
    {
        const char *routes = cli->routes_path ? cli->routes_path
                                              : CP_DEFAULT_ROUTES;
        if (cp_arch_routes_load(routes) != 0) {
            printf("Result: FAIL - routes loading failed (arch section; "
                   "see stderr).\n");
            return 1;
        }
        if (cp_tok_routes_load(routes) != 0) {
            printf("Result: FAIL - routes loading failed (tokenizer section; "
                   "see stderr).\n");
            return 1;
        }
        cp_bind_set_routes_path(routes);
    }

    if (cli->view_path) return cmd_view(cli);
    if (cli->token_text) return cmd_token(cli);
    if (cli->input_text) return cmd_generate(cli);
    return cmd_summary(cli);
}
