/*
 * log.h - structured debug log for CPulse.
 *
 * Machine-readable, one record per line:
 *
 *   TAG key=value key="quoted" ...
 *
 *   integers  bare:        count=2
 *   text      double-quoted, escapes \" \\ \n \r \t and \xNN;
 *                           raw UTF-8 passes through: text="Hi!"
 *   bytes     continuous lowercase hex: hex=4869
 *   floats    comma-separated, no spaces: values=0.0169,-0.0157,...
 *
 * Record vocabulary (fields in order):
 *   log version=1
 *   model dir="..."
 *   arch model="..." vocab=.. hidden=.. inter=.. layers=.. qheads=.. kvheads=..
 *   weights tensors=.. params=.. bytes=..
 *   bind bound=.. expected=.. tied=..
 *   tok vocab=.. merges=.. specials=.. ids=..
 *   encode text="..."
 *   special id=.. pos=.. len=.. text="..."
 *   pretoken index=.. pos=.. len=.. text="..."
 *   bytes pretoken=.. len=.. hex=...
 *   symbol pretoken=.. index=.. text="..."
 *   merge pretoken=.. rank=.. left=".." right=".." merged=".."
 *   piece pretoken=.. index=.. text=".." id=..
 *   tokens count=..
 *   token index=.. id=.. text=".."
 *   embed rows=.. dim=..
 *   vector index=.. id=.. dim=.. values=..
 *
 * Logging stays completely off until cp_log_init() opens a file; every
 * call before that (or after cp_log_close()) is a cheap no-op, so call
 * sites need no `if (verbose)` guards. Each record is one line plus a
 * flush, so the log survives crashes. Not thread-safe (no threads here).
 */

#ifndef LOG_H
#define LOG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Open (truncate) the log file. A second call is ignored. */
void cp_log_init(const char *path);

/* Close the log file. Safe on NULL / already closed (atexit-friendly). */
void cp_log_close(void);

/* 1 once a file is open, else 0. */
int cp_log_on(void);

/*
 * Compose one record: cp_log_begin(tag), then any field calls in the
 * documented order, then cp_log_end(). Emits nothing until cp_log_end.
 */
void cp_log_begin(const char *tag);
void cp_log_i(const char *key, long long v);
void cp_log_u(const char *key, unsigned long long v);
void cp_log_s(const char *key, const char *s, size_t n);
void cp_log_h(const char *key, const uint8_t *d, size_t n);
void cp_log_v(const float *v, size_t n); /* writes the ` values=...` field */
void cp_log_end(void);

#ifdef __cplusplus
}
#endif

#endif /* LOG_H */
