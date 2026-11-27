/*
 * t5/generate.c - T5 greedy generate (see t5/generate.h)
 */

#include "t5/generate.h"
#include "t5/decoder.h"
#include "t5/encoder.h"

#include <stdio.h>
#include <stdlib.h>

struct cp_t5_session {
    const cp_t5_config_t  *cfg;
    const cp_t5_weights_t *w;
    cp_t5_encoder_t       *enc;
    cp_t5_decoder_t       *dec;
    size_t                 max_src;
    size_t                 max_tgt;
};

static uint32_t argmax(const float *v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int cp_t5_session_init(cp_t5_session_t **out, const cp_t5_config_t *cfg,
                       const cp_t5_weights_t *w, size_t max_src,
                       size_t max_tgt) {
    if (!out || !cfg || !w) return -1;
    *out = NULL;
    if (max_src == 0) max_src = 512;
    if (max_tgt == 0) max_tgt = 512;

    cp_t5_session_t *s = calloc(1, sizeof *s);
    if (!s) {
        fprintf(stderr, "cp_t5_session: out of memory\n");
        return -1;
    }
    s->cfg = cfg;
    s->w = w;
    s->max_src = max_src;
    s->max_tgt = max_tgt;

    if (cp_t5_encoder_init(&s->enc, cfg, w, max_src) != 0) {
        free(s);
        return -1;
    }
    if (cp_t5_decoder_init(&s->dec, cfg, w, max_src, max_tgt) != 0) {
        cp_t5_encoder_free(s->enc);
        free(s);
        return -1;
    }
    *out = s;
    return 0;
}

void cp_t5_session_free(cp_t5_session_t *s) {
    if (!s) return;
    cp_t5_decoder_free(s->dec);
    cp_t5_encoder_free(s->enc);
    free(s);
}

int cp_t5_generate(cp_t5_session_t *s, const uint32_t *src, size_t src_n,
                   uint32_t *out_ids, size_t out_cap, int max_new,
                   cp_t5_gen_token_fn on_token, void *userdata) {
    if (!s || !s->enc || !s->dec || !s->cfg || !src || src_n == 0) return -1;
    if (src_n > s->max_src) {
        fprintf(stderr, "cp_t5_generate: src_n=%zu exceeds max_src=%zu\n",
                src_n, s->max_src);
        return -1;
    }
    if (max_new < 0) max_new = 0;
    if ((size_t)max_new > s->max_tgt) max_new = (int)s->max_tgt;

    const float *mem = cp_t5_encoder_run(s->enc, src, src_n);
    if (!mem) {
        fprintf(stderr, "cp_t5_generate: encoder failed\n");
        return -1;
    }
    if (cp_t5_decoder_set_memory(s->dec, mem, src_n) != 0) {
        fprintf(stderr, "cp_t5_generate: set_memory failed\n");
        return -1;
    }

    const cp_t5_config_t *cfg = s->cfg;
    uint32_t tok = cfg->decoder_start_token_id;
    int n_gen = 0;

    for (int g = 0; g < max_new; g++) {
        size_t pos = (size_t)g;
        const float *logits = cp_t5_decoder_step(s->dec, tok, pos);
        if (!logits) {
            fprintf(stderr, "cp_t5_generate: decode failed at pos %zu\n", pos);
            return -1;
        }
        uint32_t next = argmax(logits, cfg->vocab_size);

        if (out_ids && (size_t)n_gen < out_cap)
            out_ids[n_gen] = next;
        n_gen++;

        if (next == cfg->eos_token_id)
            break;

        if (on_token) on_token(next, pos, userdata);
        tok = next;
    }

    return n_gen;
}
