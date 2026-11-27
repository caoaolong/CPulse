/*
 * tok/bpe.h - byte-level BPE backend (internal; use tok/tokenizer.h)
 */

#ifndef TOK_BPE_H
#define TOK_BPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tok_bpe tok_bpe_t;

int tok_bpe_load(tok_bpe_t **out, const char *model_dir);
void tok_bpe_free(tok_bpe_t *t);

int tok_bpe_encode(const tok_bpe_t *t, const char *text,
                   uint32_t *out, size_t cap);
int tok_bpe_lookup(const tok_bpe_t *t, const char *text, uint32_t *out);
int tok_bpe_decode(const tok_bpe_t *t, const uint32_t *ids, size_t n,
                   char *out, size_t cap);

size_t tok_bpe_vocab_size(const tok_bpe_t *t);

#ifdef __cplusplus
}
#endif

#endif /* TOK_BPE_H */
