/*
 * arch.h - architecture strategy routing for CPulse
 *
 * model_type is mapped to one of three strategies via the "arch" section
 * of the unified routes file (default: config/routes.json).
 * Implementation availability is decided in code: decoder-only (Qwen)
 * and encoder-decoder (T5 config path).
 */

#ifndef ARCH_H
#define ARCH_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CP_DEFAULT_ROUTES "config/routes.json"
/* Alias kept for older call sites / docs. */
#define CP_DEFAULT_ARCH_CONFIG CP_DEFAULT_ROUTES

typedef enum {
    CP_ARCH_UNKNOWN = 0,
    CP_ARCH_ENCODER_ONLY,
    CP_ARCH_DECODER_ONLY,
    CP_ARCH_ENCODER_DECODER
} cp_arch_id_t;

typedef struct cp_arch {
    cp_arch_id_t id;           /* strategy */
    const char  *name;         /* "decoder-only" / ... */
    const char  *family;       /* same as name (kept for call sites) */
    int          implemented;  /* 1 = bind+forward path exists */
} cp_arch_t;

/*
 * Load / reload the arch route table from path.
 * Expected shape (sectioned):
 *   { "arch": {
 *       "decoder-only": ["qwen2", ...],
 *       "encoder-only": [...],
 *       "encoder-decoder": [...]
 *   }, ... }
 * Returns 0 on success; on failure prints to stderr and returns -1.
 */
int cp_arch_routes_load(const char *path);

/* Resolve model_type using the loaded route table. NULL if unknown. */
const cp_arch_t *cp_arch_resolve(const char *model_type);

/* Strategy name for an id; "unknown" if not recognized. */
const char *cp_arch_name(cp_arch_id_t id);

/* Print the loaded route table (or a hint if not loaded). */
void cp_arch_print_registry(FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* ARCH_H */
