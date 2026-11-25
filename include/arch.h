/*
 * arch.h - architecture routing for CPulse
 *
 * Maps config.json "model_type" to a known architecture entry.
 * Only implemented arches may proceed to weight bind / forward;
 * recognized-but-stub arches fail early with a clear message.
 */

#ifndef ARCH_H
#define ARCH_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CP_ARCH_UNKNOWN = 0,
    CP_ARCH_QWEN2,
    CP_ARCH_LLAMA,
    CP_ARCH_MISTRAL,
    CP_ARCH_T5
} cp_arch_id_t;

typedef struct cp_arch {
    cp_arch_id_t id;
    const char  *name;         /* canonical id string, e.g. "qwen2" */
    const char  *family;       /* "decoder-only" / "encoder-decoder" */
    int          implemented;  /* 1 = bind+forward available */
} cp_arch_t;

/*
 * Resolve model_type (from config.json) to an architecture entry.
 * Returns NULL when the string is unknown.
 */
const cp_arch_t *cp_arch_resolve(const char *model_type);

/* Canonical name for an id; "unknown" if not recognized. */
const char *cp_arch_name(cp_arch_id_t id);

/* Print recognized model_type values and implementation status. */
void cp_arch_print_registry(FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* ARCH_H */
