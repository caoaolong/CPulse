/*
 * model.c - config.json loader (see model.h)
 *
 * Parses the flat config object with cJSON. Unknown keys are ignored so
 * extra vendor fields never break loading. rope_scaling must be absent
 * or null (standard RoPE); anything else fails loudly.
 */

#include "model.h"
#include "arch.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void copy_cstr(char *dst, size_t cap, const char *src) {
    if (!src) return;
    strncpy(dst, src, cap - 1);
    dst[cap - 1] = '\0';
}

static int get_u32(const cJSON *obj, const char *key, uint32_t *out) {
    const cJSON *it = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsNumber(it)) return 0;
    *out = (uint32_t)it->valuedouble;
    return 1;
}

static int get_f64(const cJSON *obj, const char *key, double *out) {
    const cJSON *it = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsNumber(it)) return 0;
    *out = it->valuedouble;
    return 1;
}

static int get_bool(const cJSON *obj, const char *key, int *out) {
    const cJSON *it = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsBool(it)) return 0;
    *out = cJSON_IsTrue(it) ? 1 : 0;
    return 1;
}

static int get_str(const cJSON *obj, const char *key, char *dst, size_t cap) {
    const cJSON *it = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsString(it) || !it->valuestring) return 0;
    copy_cstr(dst, cap, it->valuestring);
    return 1;
}

static int cp_config_validate(cp_config_t *c, unsigned seen) {
    if (!(seen & REQ_MODEL_TYPE) || c->model_type[0] == '\0') {
        fprintf(stderr, "cp_config: missing required key 'model_type'\n");
        return -1;
    }

    const cp_arch_t *arch = cp_arch_resolve(c->model_type);
    if (!arch) {
        fprintf(stderr, "cp_config: unknown model_type '%s'\n", c->model_type);
        cp_arch_print_registry(stderr);
        return -1;
    }
    c->arch = arch->id;

    if (!arch->implemented) {
        fprintf(stderr,
                "cp_config: strategy '%s' recognized from model_type='%s', "
                "but not implemented yet\n",
                arch->name, c->model_type);
        return -1;
    }

    if (seen != REQ_ALL) {
        fprintf(stderr, "cp_config: missing required keys (seen=0x%x, want=0x%x)\n",
                seen, REQ_ALL);
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
    if (c->num_key_value_heads == 0 ||
        c->num_attention_heads % c->num_key_value_heads != 0) {
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

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root || !cJSON_IsObject(root)) {
        fprintf(stderr, "cp_config: parse error in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->bos_token_id = 0;
    out->eos_token_id = 0;
    unsigned seen = 0;

    if (get_u32(root, "vocab_size", &out->vocab_size)) seen |= REQ_VOCAB;
    if (get_u32(root, "hidden_size", &out->hidden_size)) seen |= REQ_HIDDEN;
    if (get_u32(root, "intermediate_size", &out->intermediate_size)) seen |= REQ_INTER;
    if (get_u32(root, "num_hidden_layers", &out->num_hidden_layers)) seen |= REQ_LAYERS;
    if (get_u32(root, "num_attention_heads", &out->num_attention_heads)) seen |= REQ_HEADS;
    if (get_u32(root, "num_key_value_heads", &out->num_key_value_heads)) seen |= REQ_KV_HEADS;
    if (get_u32(root, "max_position_embeddings", &out->max_position_embeddings))
        seen |= REQ_MAX_POS;
    get_u32(root, "bos_token_id", &out->bos_token_id);
    get_u32(root, "eos_token_id", &out->eos_token_id);
    if (get_f64(root, "rms_norm_eps", &out->rms_norm_eps)) seen |= REQ_NORM_EPS;
    if (get_f64(root, "rope_theta", &out->rope_theta)) seen |= REQ_ROPE_THETA;
    get_bool(root, "tie_word_embeddings", &out->tie_word_embeddings);
    if (get_str(root, "hidden_act", out->hidden_act, sizeof out->hidden_act))
        seen |= REQ_ACT;
    if (get_str(root, "model_type", out->model_type, sizeof out->model_type))
        seen |= REQ_MODEL_TYPE;
    get_str(root, "torch_dtype", out->torch_dtype, sizeof out->torch_dtype);

    {
        const cJSON *rs = cJSON_GetObjectItemCaseSensitive(root, "rope_scaling");
        if (rs && !cJSON_IsNull(rs)) {
            fprintf(stderr, "cp_config: unsupported rope_scaling "
                    "(only standard RoPE, i.e. absent or null, is supported)\n");
            cJSON_Delete(root);
            return -1;
        }
    }

    cJSON_Delete(root);
    return cp_config_validate(out, seen);
}

void cp_config_print(const cp_config_t *c) {
    printf("Config:\n");
    printf("  model_type=%s arch=%s hidden_act=%s torch_dtype=%s tie_emb=%d\n",
           c->model_type, cp_arch_name(c->arch), c->hidden_act, c->torch_dtype,
           c->tie_word_embeddings);
    printf("  vocab=%u hidden=%u inter=%u layers=%u\n",
           c->vocab_size, c->hidden_size, c->intermediate_size, c->num_hidden_layers);
    printf("  heads q=%u kv=%u (head_dim=%u, kv_repeat=%u) max_pos=%u\n",
           c->num_attention_heads, c->num_key_value_heads,
           c->head_dim, c->kv_repeat, c->max_position_embeddings);
    printf("  rms_norm_eps=%g rope_theta=%g bos=%u eos=%u\n",
           c->rms_norm_eps, c->rope_theta, c->bos_token_id, c->eos_token_id);
}
