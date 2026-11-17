/*
 * model.c - config.json loader (see model.h)
 *
 * Parses the flat config object with the standalone parser in
 * json.h/json.c. Unknown keys are skipped so extra vendor fields
 * never break loading; every required key is tracked and missing
 * keys fail the load.
 */

#include "model.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* required-key bookkeeping */
#define REQ_VOCAB      (1u << 0)
#define REQ_HIDDEN     (1u << 1)
#define REQ_INTER      (1u << 2)
#define REQ_LAYERS     (1u << 3)
#define REQ_HEADS      (1u << 4)
#define REQ_KV_HEADS   (1u << 5)
#define REQ_MAX_POS    (1u << 6)
#define REQ_NORM_EPS   (1u << 7)
#define REQ_ROPE_THETA (1u << 8)
#define REQ_ACT        (1u << 9)
#define REQ_MODEL_TYPE (1u << 10)
#define REQ_ALL        (REQ_VOCAB | REQ_HIDDEN | REQ_INTER | REQ_LAYERS | \
                        REQ_HEADS | REQ_KV_HEADS | REQ_MAX_POS | REQ_NORM_EPS | \
                        REQ_ROPE_THETA | REQ_ACT | REQ_MODEL_TYPE)

/* copy a parsed string into a fixed buffer; truncates, never overflows */
static void copy_str(char *dst, size_t cap, char *src) {
    if (!src) return;
    strncpy(dst, src, cap - 1);
    dst[cap - 1] = '\0';
    free(src);
}

static int cp_config_validate(cp_config_t *c, unsigned seen) {
    if (seen != REQ_ALL) {
        fprintf(stderr, "cp_config: missing required keys (seen=0x%x, want=0x%x)\n",
                seen, REQ_ALL);
        return -1;
    }
    if (strcmp(c->model_type, "qwen2") != 0) {
        fprintf(stderr, "cp_config: unsupported model_type '%s' (want 'qwen2')\n",
                c->model_type);
        return -1;
    }
    if (strcmp(c->hidden_act, "silu") != 0) {
        fprintf(stderr, "cp_config: unsupported hidden_act '%s' (want 'silu')\n",
                c->hidden_act);
        return -1;
    }
    if (c->hidden_size == 0 || c->hidden_size % c->num_attention_heads != 0) {
        fprintf(stderr, "cp_config: hidden_size %u not divisible by %u heads\n",
                c->hidden_size, c->num_attention_heads);
        return -1;
    }
    if (c->num_attention_heads % c->num_key_value_heads != 0) {
        fprintf(stderr, "cp_config: %u q-heads not divisible by %u kv-heads (GQA)\n",
                c->num_attention_heads, c->num_key_value_heads);
        return -1;
    }
    c->head_dim = c->hidden_size / c->num_attention_heads;
    c->kv_repeat = c->num_attention_heads / c->num_key_value_heads;
    return 0;
}

int cp_config_load(const char *path, cp_config_t *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cp_config: cannot open '%s'\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size <= 0) {
        fprintf(stderr, "cp_config: '%s' is empty\n", path);
        fclose(fp);
        return -1;
    }
    char *buf = malloc((size_t)size + 1);
    if (!buf) {
        fclose(fp);
        return -1;
    }
    if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
        fprintf(stderr, "cp_config: short read on '%s'\n", path);
        free(buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    buf[size] = '\0';

    memset(out, 0, sizeof(*out));
    /* non-zero defaults for optional keys */
    out->bos_token_id = 0;
    out->eos_token_id = 0;

    json_parser_t j = { buf, buf + size, 0 };
    json_expect(&j, '{');
    unsigned seen = 0;

    while (!j.err) {
        int c = json_peek(&j);
        if (c == '}') { json_next(&j); break; }

        size_t klen = 0;
        char *key = json_parse_string(&j, &klen);
        if (j.err) { free(key); break; }
        json_expect(&j, ':');
        if (j.err) { free(key); break; }

        if (strcmp(key, "vocab_size") == 0) {
            out->vocab_size = (uint32_t)json_parse_uint(&j); seen |= REQ_VOCAB;
        } else if (strcmp(key, "hidden_size") == 0) {
            out->hidden_size = (uint32_t)json_parse_uint(&j); seen |= REQ_HIDDEN;
        } else if (strcmp(key, "intermediate_size") == 0) {
            out->intermediate_size = (uint32_t)json_parse_uint(&j); seen |= REQ_INTER;
        } else if (strcmp(key, "num_hidden_layers") == 0) {
            out->num_hidden_layers = (uint32_t)json_parse_uint(&j); seen |= REQ_LAYERS;
        } else if (strcmp(key, "num_attention_heads") == 0) {
            out->num_attention_heads = (uint32_t)json_parse_uint(&j); seen |= REQ_HEADS;
        } else if (strcmp(key, "num_key_value_heads") == 0) {
            out->num_key_value_heads = (uint32_t)json_parse_uint(&j); seen |= REQ_KV_HEADS;
        } else if (strcmp(key, "max_position_embeddings") == 0) {
            out->max_position_embeddings = (uint32_t)json_parse_uint(&j); seen |= REQ_MAX_POS;
        } else if (strcmp(key, "bos_token_id") == 0) {
            out->bos_token_id = (uint32_t)json_parse_uint(&j);
        } else if (strcmp(key, "eos_token_id") == 0) {
            out->eos_token_id = (uint32_t)json_parse_uint(&j);
        } else if (strcmp(key, "rms_norm_eps") == 0) {
            out->rms_norm_eps = json_parse_number(&j); seen |= REQ_NORM_EPS;
        } else if (strcmp(key, "rope_theta") == 0) {
            out->rope_theta = json_parse_number(&j); seen |= REQ_ROPE_THETA;
        } else if (strcmp(key, "tie_word_embeddings") == 0) {
            out->tie_word_embeddings = json_parse_bool(&j);
        } else if (strcmp(key, "hidden_act") == 0) {
            copy_str(out->hidden_act, sizeof(out->hidden_act),
                     json_parse_string(&j, NULL)); seen |= REQ_ACT;
        } else if (strcmp(key, "model_type") == 0) {
            copy_str(out->model_type, sizeof(out->model_type),
                     json_parse_string(&j, NULL)); seen |= REQ_MODEL_TYPE;
        } else if (strcmp(key, "torch_dtype") == 0) {
            copy_str(out->torch_dtype, sizeof(out->torch_dtype),
                     json_parse_string(&j, NULL));
        } else {
            json_skip_value(&j); /* architectures, sliding_window, ... */
        }
        free(key);
        if (j.err) break;

        c = json_peek(&j);
        if (c == ',') { json_next(&j); continue; }
        if (c == '}') { json_next(&j); break; }
        j.err = 1;
    }

    free(buf);
    if (j.err) {
        fprintf(stderr, "cp_config: parse error in '%s'\n", path);
        return -1;
    }
    return cp_config_validate(out, seen);
}

void cp_config_print(const cp_config_t *c) {
    printf("Config:\n");
    printf("  model_type=%s hidden_act=%s torch_dtype=%s tie_emb=%d\n",
           c->model_type, c->hidden_act, c->torch_dtype, c->tie_word_embeddings);
    printf("  vocab=%u hidden=%u inter=%u layers=%u\n",
           c->vocab_size, c->hidden_size, c->intermediate_size, c->num_hidden_layers);
    printf("  heads q=%u kv=%u (head_dim=%u, kv_repeat=%u) max_pos=%u\n",
           c->num_attention_heads, c->num_key_value_heads,
           c->head_dim, c->kv_repeat, c->max_position_embeddings);
    printf("  rms_norm_eps=%g rope_theta=%g bos=%u eos=%u\n",
           c->rms_norm_eps, c->rope_theta, c->bos_token_id, c->eos_token_id);
}
