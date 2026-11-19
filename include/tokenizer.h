/*
 * tokenizer.h - Qwen2 byte-level BPE tokenizer for CPulse (CPU)
 *
 * Loads vocab.json + merges.txt + tokenizer_config.json from the model
 * directory and encodes UTF-8 text to token ids, mirroring the HF
 * Qwen2Tokenizer: special-token split, GPT-2 style pre-tokenizer regex
 * (Split), ByteLevel mapping, BPE merges applied by rank. Decode inverts
 * the byte mapping for round-trip checks.
 *
 * Unicode note: \p{L} / \p{N} classes are approximated (ASCII exact;
 * non-ASCII code points: spaces/punctuation ranges -> P/SP, fullwidth
 * digits -> N, fullwidth ASCII letters -> L, everything else -> L).
 * ASCII / common English + CJK text follows HF exactly; exotic scripts
 * still encode to valid (slightly longer) id sequences.
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_tok cp_tok_t;

/*
 * Load the tokenizer from <model_dir>/vocab.json, merges.txt and
 * tokenizer_config.json. On success *out holds the tokenizer (free with
 * cp_tok_free); on failure prints a diagnostic and returns -1.
 */
int cp_tok_load(cp_tok_t **out, const char *model_dir);

/* Release everything owned by the tokenizer. Safe on NULL. */
void cp_tok_free(cp_tok_t *t);

/*
 * Encode UTF-8 text to token ids. Returns the id count, or -1 when the
 * output buffer (cap ids) is too small or a piece misses the vocab.
 * Special tokens in the text (e.g. <|im_start|>) are split out first
 * (longest match wins).
 */
int cp_tok_encode(const cp_tok_t *t, const char *text,
                  uint32_t *out, size_t cap);

/*
 * Exact lookup of one surface-form token (byte-level BPE form, e.g. "Hello").
 * Checks the vocab first, then the special tokens (<|im_start|>, ...).
 * Returns 0 and sets *out on hit, -1 on miss. For plain text that needs
 * pre-tokenizing + BPE, use cp_tok_encode instead.
 */
int cp_tok_lookup(const cp_tok_t *t, const char *text, uint32_t *out);

/*
 * Decode ids back to UTF-8 (NUL-terminated). Returns bytes written
 * excluding the NUL, or -1 on unknown id / output overflow (cap holds
 * the NUL as well).
 */
int cp_tok_decode(const cp_tok_t *t, const uint32_t *ids, size_t n,
                  char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */
