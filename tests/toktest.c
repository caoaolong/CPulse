#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"
#include "tokenizer.h"

#define MODEL_DIR ".cache/Qwen2.5-0.5B"

static int fails = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
    else { printf("ok: %s\n", msg); } \
} while (0)

static void show_ids(const char *s, const uint32_t *ids, int n) {
    printf("  [%s] -> (%d)", s, n);
    for (int i = 0; i < n; i++) printf(" %u", ids[i]);
    printf("\n");
}

int main(void) {
    cp_tok_t *tk = NULL;
    if (cp_tok_load(&tk, MODEL_DIR) != 0) {
        fprintf(stderr, "toktest: load failed\n");
        return 1;
    }

    uint32_t ids[256];
    char buf[1024];
    int n, m;

    /* pinned single-token cases (ids read straight from vocab.json) */
    n = cp_tok_encode(tk, "Hello", ids, 256);
    show_ids("Hello", ids, n);
    CHECK(n == 1 && ids[0] == 9707, "Hello -> [9707]");

    n = cp_tok_encode(tk, "!", ids, 256);
    CHECK(n == 1 && ids[0] == 0, "! -> [0]");

    n = cp_tok_encode(tk, "", ids, 256);
    CHECK(n == 0, "empty -> []");

    /* round-trips: decode(encode(s)) == s */
    static const char *cases[] = {
        "Hello world",
        "The quick brown fox jumps.",
        "  leading spaces",
        "trailing spaces  ",
        "line1\nline2\n",
        "don't stop",
        "123 + 456 = 579",
        "Hello, 世界！",
        "中文分词测试",
        "<|im_start|>user\nHi<|im_end|>",
        "a  b\n\nc",
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        n = cp_tok_encode(tk, cases[i], ids, 256);
        if (n < 0) {
            printf("FAIL: encode '%s' error\n", cases[i]);
            fails++;
            continue;
        }
        m = cp_tok_decode(tk, (const uint32_t *)ids, (size_t)n,
                          buf, sizeof buf);
        show_ids(cases[i], ids, n);
        CHECK(m >= 0 && strcmp(buf, cases[i]) == 0, "round-trip");
    }

    /* string -> embedding glue on a synthetic table */
    {
        uint16_t Ef[8 * 4];
        for (int i = 0; i < 8 * 4; i++)
            Ef[i] = cp_fp32_to_bf16((float)i * 0.5f - 3.0f);
        uint32_t toks[3] = { 2, 5, 7 };
        float Y[3 * 4], R[4];
        int bad = 0;
        cp_embed_seq(Y, Ef, 4, toks, 3);
        for (int i = 0; i < 3 && !bad; i++) {
            cp_embed(R, Ef, 4, toks[i]);
            for (int k = 0; k < 4; k++) {
                if (Y[i * 4 + k] != R[k]) { bad = 1; break; }
            }
        }
        CHECK(!bad, "cp_embed_seq == 3x cp_embed");
    }

    cp_tok_free(tk);
    printf(fails ? "toktest: FAIL (%d)\n" : "toktest: PASS\n", fails);
    return fails != 0;
}
