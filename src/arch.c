/*
 * arch.c - model_type -> architecture registry (routing only)
 *
 * Qwen2 is the only implemented entry today. Other known types resolve
 * successfully so callers can report "recognized but not implemented"
 * instead of a generic config failure.
 */

#include "arch.h"

#include <string.h>

static const cp_arch_t k_arch_qwen2 = {
    CP_ARCH_QWEN2, "qwen2", "decoder-only", 1
};
static const cp_arch_t k_arch_llama = {
    CP_ARCH_LLAMA, "llama", "decoder-only", 0
};
static const cp_arch_t k_arch_mistral = {
    CP_ARCH_MISTRAL, "mistral", "decoder-only", 0
};
static const cp_arch_t k_arch_t5 = {
    CP_ARCH_T5, "t5", "encoder-decoder", 0
};

const cp_arch_t *cp_arch_resolve(const char *model_type) {
    if (!model_type || !model_type[0]) return NULL;

    if (strcmp(model_type, "qwen2") == 0)
        return &k_arch_qwen2;

    /* Common HF aliases that share the LLaMA-style decoder block. */
    if (strcmp(model_type, "llama") == 0 ||
        strcmp(model_type, "llama2") == 0 ||
        strcmp(model_type, "llama3") == 0 ||
        strcmp(model_type, "llama3_2") == 0 ||
        strcmp(model_type, "tinyllama") == 0)
        return &k_arch_llama;

    if (strcmp(model_type, "mistral") == 0)
        return &k_arch_mistral;

    if (strcmp(model_type, "t5") == 0 ||
        strcmp(model_type, "mt5") == 0 ||
        strcmp(model_type, "flan-t5") == 0)
        return &k_arch_t5;

    return NULL;
}

const char *cp_arch_name(cp_arch_id_t id) {
    switch (id) {
    case CP_ARCH_QWEN2:   return "qwen2";
    case CP_ARCH_LLAMA:   return "llama";
    case CP_ARCH_MISTRAL: return "mistral";
    case CP_ARCH_T5:      return "t5";
    case CP_ARCH_UNKNOWN:
    default:              return "unknown";
    }
}

void cp_arch_print_registry(FILE *fp) {
    if (!fp) fp = stderr;
    fprintf(fp, "cp_arch: known model_type values:\n");
    fprintf(fp, "  qwen2                         -> qwen2    [implemented]\n");
    fprintf(fp, "  llama / llama2 / llama3 / ...  -> llama    [stub]\n");
    fprintf(fp, "  mistral                       -> mistral  [stub]\n");
    fprintf(fp, "  t5 / mt5 / flan-t5            -> t5       [stub]\n");
}
