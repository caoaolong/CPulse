/*
 * tok/tokenizer.h - tokenizer strategy facade (BPE / Unigram / ...)
 *
 * Strategy is resolved from the model directory (tokenizer_config class,
 * tokenizer.json model.type, or file layout), optionally via tok_routes.json.
 */

#ifndef TOK_TOKENIZER_H
#define TOK_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CP_DEFAULT_TOK_ROUTES "tok_routes.json"

typedef enum {
    CP_TOK_UNKNOWN = 0,
    CP_TOK_BPE,
    CP_TOK_UNIGRAM
} cp_tok_strategy_t;

typedef struct cp_tokenizer cp_tokenizer_t;

/* Load / reload strategy routes (class name / model.type -> strategy). */
int cp_tok_routes_load(const char *path);

const char *cp_tok_strategy_name(cp_tok_strategy_t id);
void cp_tok_print_routes(FILE *fp);

/* Detect strategy for a model dir without loading vocab (0 / UNKNOWN on fail). */
cp_tok_strategy_t cp_tok_detect(const char *model_dir);

/*
 * Load tokenizer for model_dir. Auto-loads default tok_routes.json if needed.
 * On success *out is owned by caller (cp_tokenizer_free).
 */
int cp_tokenizer_load(cp_tokenizer_t **out, const char *model_dir);
void cp_tokenizer_free(cp_tokenizer_t *t);

cp_tok_strategy_t cp_tokenizer_strategy(const cp_tokenizer_t *t);
size_t   cp_tokenizer_vocab_size(const cp_tokenizer_t *t);
uint32_t cp_tokenizer_pad_id(const cp_tokenizer_t *t);
uint32_t cp_tokenizer_eos_id(const cp_tokenizer_t *t);
uint32_t cp_tokenizer_unk_id(const cp_tokenizer_t *t);

/*
 * Encode UTF-8 text. add_eos is honored by Unigram (T5 default); BPE ignores.
 * Returns id count, or -1 on error / overflow.
 */
int cp_tokenizer_encode(const cp_tokenizer_t *t, const char *text,
                        uint32_t *out, size_t cap, int add_eos);

/*
 * Exact surface-form lookup when the backend supports it (BPE).
 * Returns 0 and sets *out on hit, -1 on miss / unsupported.
 */
int cp_tokenizer_lookup(const cp_tokenizer_t *t, const char *text,
                        uint32_t *out);

/*
 * Decode ids to UTF-8. skip_special is honored by Unigram; BPE ignores for now.
 */
int cp_tokenizer_decode(const cp_tokenizer_t *t, const uint32_t *ids, size_t n,
                        char *out, size_t cap, int skip_special);

#ifdef __cplusplus
}
#endif

#endif /* TOK_TOKENIZER_H */
