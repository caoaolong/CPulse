/*
 * t5_config.c - T5 config.json loader (see t5_config.h)
 */

#include "t5/config.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int is_t5_family(const char *model_type) {
    return model_type &&
           (strcmp(model_type, "t5") == 0 ||
            strcmp(model_type, "mt5") == 0 ||
            strcmp(model_type, "flan-t5") == 0);
}

static int t5_validate(cp_t5_config_t *c) {
    if (!is_t5_family(c->model_type)) {
        fprintf(stderr,
                "cp_t5_config: unsupported model_type '%s' "
                "(only t5 / mt5 / flan-t5 for now)\n",
                c->model_type[0] ? c->model_type : "(empty)");
        return -1;
    }

    const cp_arch_t *arch = cp_arch_resolve(c->model_type);
    if (!arch) {
        fprintf(stderr, "cp_t5_config: unknown model_type '%s' in arch routes\n",
                c->model_type);
        cp_arch_print_registry(stderr);
        return -1;
    }
    if (arch->id != CP_ARCH_ENCODER_DECODER) {
        fprintf(stderr,
                "cp_t5_config: model_type '%s' routes to '%s', "
                "expected encoder-decoder\n",
                c->model_type, arch->name);
        return -1;
    }
    c->arch = arch->id;

    if (c->vocab_size == 0 || c->d_model == 0 || c->d_ff == 0 ||
        c->d_kv == 0 || c->num_heads == 0 || c->num_layers == 0) {
        fprintf(stderr, "cp_t5_config: missing required numeric fields "
                "(vocab_size/d_model/d_ff/d_kv/num_heads/num_layers)\n");
        return -1;
    }
    if (c->layer_norm_epsilon <= 0.0) {
        fprintf(stderr, "cp_t5_config: invalid layer_norm_epsilon %g\n",
                c->layer_norm_epsilon);
        return -1;
    }

    c->inner_dim = c->num_heads * c->d_kv;
    if (c->inner_dim != c->d_model) {
        fprintf(stderr,
                "cp_t5_config: num_heads*d_kv (%u) != d_model (%u)\n",
                c->inner_dim, c->d_model);
        return -1;
    }
    if (c->num_decoder_layers == 0)
        c->num_decoder_layers = c->num_layers;

    return 0;
}

int cp_t5_config_load(const char *path, cp_t5_config_t *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cp_t5_config: cannot open '%s'\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size <= 0) {
        fprintf(stderr, "cp_t5_config: '%s' is empty\n", path);
        fclose(fp);
        return -1;
    }
    char *buf = malloc((size_t)size + 1);
    if (!buf) {
        fclose(fp);
        return -1;
    }
    if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
        fprintf(stderr, "cp_t5_config: short read on '%s'\n", path);
        free(buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    buf[size] = '\0';

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root || !cJSON_IsObject(root)) {
        fprintf(stderr, "cp_t5_config: parse error in '%s'\n", path);
        cJSON_Delete(root);
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->decoder_start_token_id = 0;
    out->eos_token_id = 1;
    out->pad_token_id = 0;
    out->relative_attention_num_buckets = 32;
    out->layer_norm_epsilon = 1e-6;
    out->is_encoder_decoder = 1;
    copy_cstr(out->feed_forward_proj, sizeof out->feed_forward_proj, "relu");

    get_str(root, "model_type", out->model_type, sizeof out->model_type);
    get_str(root, "torch_dtype", out->torch_dtype, sizeof out->torch_dtype);
    get_str(root, "feed_forward_proj", out->feed_forward_proj,
            sizeof out->feed_forward_proj);

    get_u32(root, "vocab_size", &out->vocab_size);
    get_u32(root, "d_model", &out->d_model);
    get_u32(root, "d_ff", &out->d_ff);
    get_u32(root, "d_kv", &out->d_kv);
    get_u32(root, "num_heads", &out->num_heads);
    get_u32(root, "num_layers", &out->num_layers);
    get_u32(root, "num_decoder_layers", &out->num_decoder_layers);
    get_u32(root, "relative_attention_num_buckets",
            &out->relative_attention_num_buckets);
    get_u32(root, "decoder_start_token_id", &out->decoder_start_token_id);
    get_u32(root, "eos_token_id", &out->eos_token_id);
    get_u32(root, "pad_token_id", &out->pad_token_id);
    get_f64(root, "layer_norm_epsilon", &out->layer_norm_epsilon);
    get_bool(root, "is_encoder_decoder", &out->is_encoder_decoder);

    cJSON_Delete(root);
    return t5_validate(out);
}

void cp_t5_config_print(const cp_t5_config_t *c) {
    printf("T5 config:\n");
    printf("  model_type=%s arch=%s torch_dtype=%s ff=%s\n",
           c->model_type, cp_arch_name(c->arch), c->torch_dtype,
           c->feed_forward_proj);
    printf("  vocab=%u d_model=%u d_ff=%u d_kv=%u heads=%u\n",
           c->vocab_size, c->d_model, c->d_ff, c->d_kv, c->num_heads);
    printf("  enc_layers=%u dec_layers=%u inner_dim=%u ln_eps=%g\n",
           c->num_layers, c->num_decoder_layers, c->inner_dim,
           c->layer_norm_epsilon);
    printf("  rel_buckets=%u decoder_start=%u eos=%u pad=%u\n",
           c->relative_attention_num_buckets, c->decoder_start_token_id,
           c->eos_token_id, c->pad_token_id);
}
