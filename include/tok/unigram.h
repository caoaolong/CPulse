/*
 * tok/unigram.h - Unigram / SentencePiece-style backend (internal)
 */

#ifndef TOK_UNIGRAM_H
#define TOK_UNIGRAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tok_unigram tok_unigram_t;

int tok_unigram_load(tok_unigram_t **out, const char *model_dir);
void tok_unigram_free(tok_unigram_t *t);

uint32_t tok_unigram_pad_id(const tok_unigram_t *t);
uint32_t tok_unigram_eos_id(const tok_unigram_t *t);
uint32_t tok_unigram_unk_id(const tok_unigram_t *t);
size_t   tok_unigram_vocab_size(const tok_unigram_t *t);

int tok_unigram_encode(const tok_unigram_t *t, const char *text,
                       uint32_t *out, size_t cap, int add_eos);
int tok_unigram_decode(const tok_unigram_t *t, const uint32_t *ids, size_t n,
                       char *out, size_t cap, int skip_special);

#ifdef __cplusplus
}
#endif

#endif /* TOK_UNIGRAM_H */
