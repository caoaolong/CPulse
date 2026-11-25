/*
 * view.h - write an SVG diagram of the model block structure
 */

#ifndef VIEW_H
#define VIEW_H

#include "model.h"
#include "t5_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Write decoder-only (Qwen-style) structure SVG. Returns 0 on success. */
int cp_model_view_qwen(const cp_config_t *c, const char *path);

/* Write T5 encoder-decoder structure SVG. Returns 0 on success. */
int cp_model_view_t5(const cp_t5_config_t *c, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* VIEW_H */
