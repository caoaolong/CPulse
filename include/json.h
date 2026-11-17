/*
 * json.h - minimal JSON parser for CPulse
 *
 * Covers exactly what a safetensors header needs: objects, arrays,
 * strings (with escape handling) and unsigned integers. Unknown fields
 * can be skipped, so the parser tolerates "__metadata__" and any future
 * extra fields. Zero third-party dependencies.
 *
 * Error handling is a sticky flag: once `err` is set, every subsequent
 * operation is a no-op, so callers only need to check `err` at the end
 * of a parse sequence.
 */

#ifndef JSON_H
#define JSON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Incremental parse cursor over a fixed memory region. The fields are
 * public so callers may advance `p` directly for simple separators
 * (e.g. skipping a ',' already confirmed by json_peek). */
typedef struct {
    const char *p;   /* read cursor   */
    const char *end; /* one-past-end of the parse region */
    int         err; /* sticky error flag */
} json_parser_t;

/* Skip ASCII whitespace (space, tab, LF, CR). */
void json_skip_ws(json_parser_t *j);

/* Look at the next non-whitespace character without consuming it.
 * Returns -1 and sets err on end of input. */
int json_peek(json_parser_t *j);

/* Consume and return the next non-whitespace character. */
int json_next(json_parser_t *j);

/* Consume one specific character; sets err on mismatch. */
void json_expect(json_parser_t *j, char c);

/* Skip the remainder of a string literal (the opening quote must
 * already have been consumed). Handles backslash escapes. */
void json_skip_string(json_parser_t *j);

/* Skip any single JSON value: string, number, object, array,
 * true/false/null (nested structures included). */
void json_skip_value(json_parser_t *j);

/*
 * Parse a JSON string into a fresh NUL-terminated malloc'd buffer
 * (without the surrounding quotes). Handles the standard escapes.
 * \uXXXX is decoded only for ASCII code points (< 0x80); anything else
 * is an error. Returns NULL and sets err on failure.
 */
char *json_parse_string(json_parser_t *j, size_t *out_len);

/* Parse an unsigned 64-bit integer. Rejects signs and floats. */
uint64_t json_parse_uint(json_parser_t *j);

/* Parse a JSON number (integer, fraction and/or exponent) as double. */
double json_parse_number(json_parser_t *j);

/* Parse true/false as 1/0. Anything else sets err. */
int json_parse_bool(json_parser_t *j);

#ifdef __cplusplus
}
#endif

#endif /* JSON_H */
