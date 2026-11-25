/*
 * t5_tokenizer.h - T5 Unigram tokenizer (tokenizer.json / spiece.model)
 *
 * Default backend is tokenizer.json (HF fast). spiece.model is also
 * supported and should yield the same ids for ASCII inputs.
 */

#ifndef T5_TOKENIZER_H
#define T5_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CP_T5_TOK_JSON = 0,   /* default: <model_dir>/tokenizer.json */
    CP_T5_TOK_SPIECE = 1  /* <model_dir>/spiece.model */
} cp_t5_tok_backend_t;

typedef struct cp_t5_tok cp_t5_tok_t;

/*
 * Load T5 tokenizer from model_dir using the chosen backend.
 * On success *out is owned by the caller (cp_t5_tok_free); on failure
 * prints to stderr and returns -1.
 */
int cp_t5_tok_load(cp_t5_tok_t **out, const char *model_dir,
                   cp_t5_tok_backend_t backend);

void cp_t5_tok_free(cp_t5_tok_t *t);

/* Backend / special-id accessors. */
cp_t5_tok_backend_t cp_t5_tok_backend(const cp_t5_tok_t *t);
uint32_t cp_t5_tok_pad_id(const cp_t5_tok_t *t);
uint32_t cp_t5_tok_eos_id(const cp_t5_tok_t *t);
uint32_t cp_t5_tok_unk_id(const cp_t5_tok_t *t);
size_t   cp_t5_tok_vocab_size(const cp_t5_tok_t *t);

/*
 * Encode UTF-8 text to token ids (SentencePiece-style: space->▁,
 * optional dummy prefix, Unigram Viterbi). If add_eos != 0, append eos
 * (HF T5Tokenizer default). Returns id count, or -1 on error / overflow.
 */
int cp_t5_tok_encode(const cp_t5_tok_t *t, const char *text,
                     uint32_t *out, size_t cap, int add_eos);

/*
 * Decode ids to UTF-8 (▁ -> space, leading space stripped when the
 * dummy-prefix convention applies). skip_special drops pad/eos/unk.
 * Returns bytes written excluding NUL, or -1 on error / overflow.
 */
int cp_t5_tok_decode(const cp_t5_tok_t *t, const uint32_t *ids, size_t n,
                     char *out, size_t cap, int skip_special);

/* Parse "json" / "spiece" (case-insensitive). Returns 0 on success. */
int cp_t5_tok_backend_parse(const char *s, cp_t5_tok_backend_t *out);

const char *cp_t5_tok_backend_name(cp_t5_tok_backend_t b);

#ifdef __cplusplus
}
#endif

#endif /* T5_TOKENIZER_H */
