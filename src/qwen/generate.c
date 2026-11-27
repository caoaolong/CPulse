/*
 * generate.c - generation session (see generate.h)
 *
 * Prefill walks the prompt through cp_forward_step; decode repeatedly
 * samples the latest logits until a stop id or max_new.
 */

#include "qwen/generate.h"
#include "qwen/forward.h"

#include <stdio.h>
#include <stdlib.h>

/* Qwen chat end marker; kept as a hard stop alongside cfg->eos_token_id. */
#define CP_IM_END_ID 151645u

struct cp_session {
    const cp_config_t  *cfg;
    cp_forward_t       *fw;
};

static int is_stop_id(const cp_config_t *cfg, uint32_t id) {
    return id == cfg->eos_token_id || id == CP_IM_END_ID;
}

int cp_session_init(cp_session_t **out, const cp_config_t *cfg,
                    const cp_weights_t *w, int use_gpu) {
    if (!out || !cfg || !w) return -1;
    *out = NULL;

    cp_session_t *s = calloc(1, sizeof *s);
    if (!s) {
        fprintf(stderr, "cp_session: out of memory\n");
        return -1;
    }
    s->cfg = cfg;
    if (cp_forward_init(&s->fw, cfg, w, use_gpu) != 0) {
        free(s);
        return -1;
    }
    *out = s;
    return 0;
}

int cp_session_on_gpu(const cp_session_t *s) {
    return s && s->fw ? cp_forward_on_gpu(s->fw) : 0;
}

void cp_session_reset(cp_session_t *s) {
    if (!s) return;
    cp_forward_reset(s->fw);
}

void cp_session_free(cp_session_t *s) {
    if (!s) return;
    cp_forward_free(s->fw);
    free(s);
}

int cp_generate(cp_session_t *s,
                const uint32_t *prompt, size_t prompt_n,
                uint32_t *out_ids, size_t out_cap,
                int max_new, const cp_sample_cfg_t *sample,
                cp_gen_token_fn on_token, void *userdata) {
    if (!s || !s->fw || !s->cfg || !prompt || prompt_n == 0) return -1;
    if (max_new < 0) max_new = 0;

    const cp_config_t *cfg = s->cfg;
    const float *logits = NULL;

    cp_sampler_t *smp = NULL;
    if (cp_sampler_init(&smp, cfg->vocab_size, sample) != 0) {
        fprintf(stderr, "cp_generate: sampler init failed\n");
        return -1;
    }

    for (size_t i = 0; i < prompt_n; i++) {
        logits = cp_forward_step(s->fw, prompt[i], i);
        if (!logits) {
            fprintf(stderr, "cp_generate: prefill failed at pos %zu\n", i);
            cp_sampler_free(smp);
            return -1;
        }
    }

    int n_gen = 0;
    size_t pos = prompt_n;
    uint32_t next = cp_sampler_next(smp, logits, cfg->vocab_size);

    for (int g = 0; g < max_new; g++) {
        if (is_stop_id(cfg, next)) break;

        if (out_ids && (size_t)n_gen < out_cap) {
            out_ids[n_gen] = next;
        }
        if (on_token) on_token(next, pos, userdata);
        n_gen++;

        logits = cp_forward_step(s->fw, next, pos++);
        if (!logits) break;
        next = cp_sampler_next(smp, logits, cfg->vocab_size);
    }

    cp_sampler_free(smp);
    return n_gen;
}
