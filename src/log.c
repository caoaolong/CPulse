/*
 * log.c - structured debug log (see log.h)
 */

#include "log.h"

#include <stdio.h>

static FILE *g_log;

void cp_log_init(const char *path) {
    if (g_log || !path) return;
    g_log = fopen(path, "w");
}

void cp_log_close(void) {
    if (g_log) {
        fclose(g_log);
        g_log = NULL;
    }
}

int cp_log_on(void) {
    return g_log != NULL;
}

void cp_log_begin(const char *tag) {
    if (!g_log) return;
    fputs(tag, g_log);
}

void cp_log_i(const char *key, long long v) {
    if (!g_log) return;
    fprintf(g_log, " %s=%lld", key, v);
}

void cp_log_u(const char *key, unsigned long long v) {
    if (!g_log) return;
    fprintf(g_log, " %s=%llu", key, v);
}

/* write s[0..n) quoted, escaping controls; UTF-8 passes through raw */
static void write_escaped(const char *s, size_t n) {
    fputc('"', g_log);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\n') fputs("\\n", g_log);
        else if (c == '\r') fputs("\\r", g_log);
        else if (c == '\t') fputs("\\t", g_log);
        else if (c == '"') fputs("\\\"", g_log);
        else if (c == '\\') fputs("\\\\", g_log);
        else if (c < 0x20 || c == 0x7f) fprintf(g_log, "\\x%02x", c);
        else fputc(c, g_log);
    }
    fputc('"', g_log);
}

void cp_log_s(const char *key, const char *s, size_t n) {
    if (!g_log) return;
    fprintf(g_log, " %s=", key);
    write_escaped(s, n);
}

void cp_log_h(const char *key, const uint8_t *d, size_t n) {
    if (!g_log) return;
    fprintf(g_log, " %s=", key);
    for (size_t i = 0; i < n; i++) fprintf(g_log, "%02x", d[i]);
}

void cp_log_v(const float *v, size_t n) {
    if (!g_log) return;
    fputs(" values=", g_log);
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(',', g_log);
        fprintf(g_log, "%.6g", v[i]);
    }
}

void cp_log_end(void) {
    if (!g_log) return;
    fputc('\n', g_log);
    fflush(g_log);
}
