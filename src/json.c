/*
 * json.c - minimal JSON parser (see json.h)
 *
 * Hand-rolled recursive-descent parser covering exactly what a
 * safetensors header needs: objects, arrays, strings (with escape
 * handling) and unsigned integers. Unknown JSON fields are skipped, so
 * the parser tolerates "__metadata__" and any future extra fields.
 */

#include "json.h"

#include <stdio.h>
#include <stdlib.h>

void json_skip_ws(json_parser_t *j) {
    while (j->p < j->end &&
           (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) {
        j->p++;
    }
}

int json_peek(json_parser_t *j) {
    json_skip_ws(j);
    if (j->p >= j->end) { j->err = 1; return -1; }
    return (unsigned char)*j->p;
}

int json_next(json_parser_t *j) {
    int c = json_peek(j);
    if (!j->err && j->p < j->end) j->p++;
    return c;
}

void json_expect(json_parser_t *j, char c) {
    if (json_next(j) != c) j->err = 1;
}

/*
 * Consume the remainder of a string literal (the opening quote has already
 * been consumed). Handles backslash escapes.
 */
void json_skip_string(json_parser_t *j) {
    while (!j->err) {
        if (j->p >= j->end) { j->err = 1; return; }
        char c = *j->p++;
        if (c == '\\') {
            if (j->p >= j->end) { j->err = 1; return; }
            j->p++; /* skip the escaped character */
        } else if (c == '"') {
            return;
        }
    }
}

/* Skip any JSON value: string, number, object, array, true/false/null. */
void json_skip_value(json_parser_t *j) {
    int c = json_next(j);
    if (j->err) return;

    if (c == '"') {
        json_skip_string(j);
        return;
    }
    if (c == '{' || c == '[') {
        int depth = 1;
        while (!j->err && depth > 0) {
            if (j->p >= j->end) { j->err = 1; return; }
            char d = *j->p++;
            if (d == '"') {
                json_skip_string(j); /* strings may contain brackets/braces */
            } else if (d == '{' || d == '[') {
                depth++;
            } else if (d == '}' || d == ']') {
                depth--;
            }
        }
        return;
    }
    /* numbers / literals: run to the next delimiter */
    while (!j->err && j->p < j->end) {
        char d = *j->p;
        if (d == ',' || d == '}' || d == ']') return;
        j->p++;
    }
}

/* append one char to a growable buffer; returns 0 on OOM */
static int json_push_char(char **buf, size_t *len, size_t *cap, char c) {
    if (*len + 1 >= *cap) {
        size_t ncap = *cap * 2;
        char *nb = realloc(*buf, ncap);
        if (!nb) return 0;
        *buf = nb;
        *cap = ncap;
    }
    (*buf)[(*len)++] = c;
    return 1;
}

/*
 * Parse a JSON string into a fresh NUL-terminated malloc'd buffer (without
 * the surrounding quotes). Handles the standard escapes. \uXXXX is decoded
 * only for ASCII code points (< 0x80); anything else is rejected as an
 * error to keep the parser honest and small.
 */
char *json_parse_string(json_parser_t *j, size_t *out_len) {
    if (json_peek(j) != '"') { j->err = 1; return NULL; }
    j->p++; /* opening quote */

    size_t cap = 32, len = 0;
    char *buf = malloc(cap);
    if (!buf) { j->err = 1; return NULL; }

    while (!j->err) {
        if (j->p >= j->end) { j->err = 1; break; }
        char c = *j->p++;
        if (c == '"') break; /* closing quote */

        if (c != '\\') {
            if (!json_push_char(&buf, &len, &cap, c)) { j->err = 1; break; }
            continue;
        }

        if (j->p >= j->end) { j->err = 1; break; }
        char e = *j->p++;
        switch (e) {
        case '"':  c = '"';  break;
        case '\\': c = '\\'; break;
        case '/':  c = '/';  break;
        case 'b':  c = '\b'; break;
        case 'f':  c = '\f'; break;
        case 'n':  c = '\n'; break;
        case 'r':  c = '\r'; break;
        case 't':  c = '\t'; break;
        case 'u': {
            if (j->end - j->p < 4) { j->err = 1; break; }
            unsigned code = 0;
            for (int k = 0; k < 4; k++) {
                char h = j->p[k];
                unsigned d;
                if      (h >= '0' && h <= '9') d = (unsigned)(h - '0');
                else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                else { j->err = 1; break; }
                code = code * 16 + d;
            }
            if (j->err) break;
            j->p += 4;
            if (code < 0x80) {
                c = (char)code;
            } else {
                fprintf(stderr, "json_parse: non-ASCII \\u escape "
                                "(names in this project should be ASCII)\n");
                j->err = 1;
                break;
            }
            break;
        }
        default:
            j->err = 1;
            break;
        }
        if (j->err) break;
        if (!json_push_char(&buf, &len, &cap, c)) { j->err = 1; break; }
    }

    if (j->err) { free(buf); return NULL; }
    if (!json_push_char(&buf, &len, &cap, '\0')) { free(buf); j->err = 1; return NULL; }
    if (out_len) *out_len = len - 1; /* exclude NUL */
    return buf;
}

/* Parse an unsigned 64-bit integer (safetensors headers use only these). */
uint64_t json_parse_uint(json_parser_t *j) {
    json_skip_ws(j);
    if (j->p < j->end && *j->p == '+') j->p++;
    if (j->p < j->end && *j->p == '-') { j->err = 1; return 0; }

    uint64_t v = 0;
    int digits = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        v = v * 10 + (uint64_t)(*j->p - '0');
        j->p++;
        digits++;
    }
    if (digits == 0) j->err = 1;
    if (j->p < j->end && (*j->p == '.' || *j->p == 'e' || *j->p == 'E')) {
        j->err = 1; /* floats are not expected in tensor metadata */
    }
    return v;
}
