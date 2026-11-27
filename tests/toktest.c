#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tok/tokenizer.h"

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
    cp_tokenizer_t *tk = NULL;
    if (cp_tokenizer_load(&tk, MODEL_DIR) != 0) {
        printf("FAIL: load tokenizer\n");
        return 1;
    }
    CHECK(cp_tokenizer_strategy(tk) == CP_TOK_BPE, "strategy is bpe");

    uint32_t ids[256];
    int n;

    n = cp_tokenizer_encode(tk, "Hello", ids, 256, 0);
    CHECK(n > 0, "encode Hello");
    show_ids("Hello", ids, n);

    n = cp_tokenizer_encode(tk, "!", ids, 256, 0);
    CHECK(n > 0, "encode !");

    n = cp_tokenizer_encode(tk, "", ids, 256, 0);
    CHECK(n == 0, "encode empty");

    const char *cases[] = {
        "Hello world",
        "你好",
        "<|im_start|>user\nHi<|im_end|>",
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        n = cp_tokenizer_encode(tk, cases[i], ids, 256, 0);
        CHECK(n > 0, cases[i]);
        show_ids(cases[i], ids, n);

        char buf[1024];
        int m = cp_tokenizer_decode(tk, (const uint32_t *)ids, (size_t)n,
                                    buf, sizeof buf, 0);
        CHECK(m >= 0, "decode");
        printf("  decode -> \"%s\"\n", buf);
    }

    cp_tokenizer_free(tk);
    printf(fails ? "DONE with failures: %d\n" : "DONE ok\n", fails);
    return fails ? 1 : 0;
}
