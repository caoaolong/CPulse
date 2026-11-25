/*
 * view.c - SVG model-structure diagram (see view.h)
 *
 * Top-level blocks stack vertically. Submodules nest as inner rectangles.
 * Multi-head attention further nests one small rect per head (grid layout).
 */

#include "view.h"
#include "arch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MV_W         640
#define MV_MARGIN_X  36
#define MV_MARGIN_Y  32
#define MV_BOX_W     (MV_W - 2 * MV_MARGIN_X)
#define MV_GAP       12
#define MV_TITLE_H   28
#define MV_LEAF_H    48
#define MV_HEAD_H    52   /* room for title + d_kv subscript */
#define MV_NEST_PAD  10
#define MV_NEST_GAP  8
#define MV_HDR_H     26
#define MV_INSET     12
#define MV_HEAD_COLS 4

typedef struct {
    FILE *fp;
    int   y;
} mv_t;

typedef enum {
    MV_LAYOUT_STACK = 0, /* children stacked vertically */
    MV_LAYOUT_GRID       /* children in a wrapping grid (attention heads) */
} mv_layout_t;

typedef struct mv_node {
    const char         *title;
    const char         *shape;
    const char         *fill;
    mv_layout_t         child_layout;
    int                 grid_cols; /* for GRID; <=0 => MV_HEAD_COLS */
    const struct mv_node *kids;
    int                 n_kids;
} mv_node_t;

static void svg_esc(FILE *fp, const char *s) {
    for (; s && *s; s++) {
        switch (*s) {
        case '&': fputs("&amp;", fp); break;
        case '<': fputs("&lt;", fp); break;
        case '>': fputs("&gt;", fp); break;
        case '"': fputs("&quot;", fp); break;
        default: fputc(*s, fp); break;
        }
    }
}

static void svg_esc_n(FILE *fp, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        switch (s[i]) {
        case '&': fputs("&amp;", fp); break;
        case '<': fputs("&lt;", fp); break;
        case '>': fputs("&gt;", fp); break;
        case '"': fputs("&quot;", fp); break;
        default: fputc(s[i], fp); break;
        }
    }
}

static void svg_sub(FILE *fp, const char *s, size_t n) {
    fputs("<tspan baseline-shift=\"sub\" font-size=\"8\">", fp);
    svg_esc_n(fp, s, n);
    fputs("</tspan>", fp);
}

/*
 * Emit shape/annotation text with SVG subscripts for common symbols:
 *   d_kv / d_model / d_ff / head_dim / q_heads / kv_heads / Wq|Wk|Wv|Wo
 */
static void svg_rich(FILE *fp, const char *s) {
    if (!s) return;
    while (*s) {
        if (s[0] == 'd' && s[1] == '_') {
            const char *p = s + 2;
            size_t n = 0;
            while (p[n] && ((p[n] >= 'a' && p[n] <= 'z') ||
                            (p[n] >= 'A' && p[n] <= 'Z') ||
                            (p[n] >= '0' && p[n] <= '9')))
                n++;
            if (n > 0) {
                fputc('d', fp);
                svg_sub(fp, p, n);
                s = p + n;
                continue;
            }
        }
        if (strncmp(s, "head_dim", 8) == 0) {
            fputs("head", fp);
            svg_sub(fp, "dim", 3);
            s += 8;
            continue;
        }
        if (strncmp(s, "q_heads", 7) == 0) {
            fputs("q", fp);
            svg_sub(fp, "heads", 5);
            s += 7;
            continue;
        }
        if (strncmp(s, "kv_heads", 8) == 0) {
            fputs("kv", fp);
            svg_sub(fp, "heads", 5);
            s += 8;
            continue;
        }
        if (s[0] == 'W' && (s[1] == 'q' || s[1] == 'k' || s[1] == 'v' ||
                            s[1] == 'o')) {
            fputc('W', fp);
            svg_sub(fp, s + 1, 1);
            s += 2;
            continue;
        }
        if (strncmp(s, "wi_0", 4) == 0 || strncmp(s, "wi_1", 4) == 0) {
            fputs("wi", fp);
            svg_sub(fp, s + 3, 1);
            s += 4;
            continue;
        }
        /* copy one character (escaped) */
        char tmp[2] = { *s, '\0' };
        svg_esc(fp, tmp);
        s++;
    }
}

static int mv_node_height(const mv_node_t *n);

static int mv_hdr_h(const mv_node_t *n) {
    return (n && n->shape && n->shape[0]) ? 40 : MV_HDR_H;
}

static int mv_stack_height(const mv_node_t *parent, const mv_node_t *kids, int n) {
    if (n <= 0) return MV_LEAF_H;
    int h = mv_hdr_h(parent) + MV_NEST_PAD;
    for (int i = 0; i < n; i++) {
        h += mv_node_height(&kids[i]);
        if (i + 1 < n) h += MV_NEST_GAP;
    }
    return h + MV_NEST_PAD;
}

static int mv_grid_height(const mv_node_t *parent, int n, int cols) {
    if (n <= 0) return MV_LEAF_H;
    if (cols <= 0) cols = MV_HEAD_COLS;
    int rows = (n + cols - 1) / cols;
    return mv_hdr_h(parent) + MV_NEST_PAD +
           rows * MV_HEAD_H + (rows > 0 ? (rows - 1) * MV_NEST_GAP : 0) +
           MV_NEST_PAD;
}

static int mv_node_height(const mv_node_t *n) {
    if (!n) return MV_LEAF_H;
    if (n->n_kids <= 0) return MV_LEAF_H;
    if (n->child_layout == MV_LAYOUT_GRID)
        return mv_grid_height(n, n->n_kids, n->grid_cols);
    return mv_stack_height(n, n->kids, n->n_kids);
}

static void mv_draw_leaf_box(FILE *fp, int x, int y, int w, int h,
                             const char *title, const char *shape,
                             const char *fill, int title_size) {
    fprintf(fp,
            "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
            "rx=\"5\" ry=\"5\" fill=\"%s\" stroke=\"#334155\" "
            "stroke-width=\"1.2\"/>\n",
            x, y, w, h, fill);
    /* Keep baseline well inside the rect (subscripts need extra descent). */
    int ty = (shape && shape[0]) ? y + 18 : y + h / 2 + 4;
    int sy = y + h - 14;
    fprintf(fp,
            "  <text x=\"%d\" y=\"%d\" "
            "font-family=\"Segoe UI,Helvetica,Arial,sans-serif\" "
            "font-size=\"%d\" font-weight=\"600\" fill=\"#0f172a\">",
            x + 8, ty, title_size);
    svg_esc(fp, title);
    fputs("</text>\n", fp);
    if (shape && shape[0]) {
        fprintf(fp,
                "  <text x=\"%d\" y=\"%d\" "
                "font-family=\"Consolas,Menlo,monospace\" "
                "font-size=\"10\" fill=\"#475569\">",
                x + 8, sy);
        svg_rich(fp, shape);
        fputs("</text>\n", fp);
    }
}

static void mv_draw_node(FILE *fp, int x, int y, int w, const mv_node_t *n);

static void mv_draw_stack_children(FILE *fp, int x, int y, int w,
                                   const mv_node_t *kids, int n) {
    int cy = y;
    for (int i = 0; i < n; i++) {
        int ch = mv_node_height(&kids[i]);
        mv_draw_node(fp, x, cy, w, &kids[i]);
        cy += ch + MV_NEST_GAP;
    }
}

static void mv_draw_grid_children(FILE *fp, int x, int y, int w,
                                  const mv_node_t *kids, int n, int cols) {
    if (cols <= 0) cols = MV_HEAD_COLS;
    int gap = MV_NEST_GAP;
    int cell_w = (w - (cols - 1) * gap) / cols;
    for (int i = 0; i < n; i++) {
        int row = i / cols;
        int col = i % cols;
        int cx = x + col * (cell_w + gap);
        int cy = y + row * (MV_HEAD_H + gap);
        mv_draw_leaf_box(fp, cx, cy, cell_w, MV_HEAD_H,
                         kids[i].title, kids[i].shape, kids[i].fill, 11);
    }
}

static void mv_draw_node(FILE *fp, int x, int y, int w, const mv_node_t *n) {
    int h = mv_node_height(n);
    if (n->n_kids <= 0) {
        mv_draw_leaf_box(fp, x, y, w, h, n->title, n->shape, n->fill, 12);
        return;
    }

    fprintf(fp,
            "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
            "rx=\"6\" ry=\"6\" fill=\"%s\" stroke=\"#334155\" "
            "stroke-width=\"1.4\"/>\n",
            x, y, w, h, n->fill ? n->fill : "#ffffff");
    fprintf(fp,
            "  <text x=\"%d\" y=\"%d\" "
            "font-family=\"Segoe UI,Helvetica,Arial,sans-serif\" "
            "font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">",
            x + 10, y + 16);
    svg_esc(fp, n->title);
    fputs("</text>\n", fp);
    if (n->shape && n->shape[0]) {
        fprintf(fp,
                "  <text x=\"%d\" y=\"%d\" "
                "font-family=\"Consolas,Menlo,monospace\" "
                "font-size=\"10\" fill=\"#475569\">",
                x + 10, y + 32);
        svg_rich(fp, n->shape);
        fputs("</text>\n", fp);
    }

    int inner_x = x + MV_INSET;
    int inner_w = w - 2 * MV_INSET;
    int inner_y = y + mv_hdr_h(n) + MV_NEST_PAD;
    if (n->child_layout == MV_LAYOUT_GRID)
        mv_draw_grid_children(fp, inner_x, inner_y, inner_w,
                              n->kids, n->n_kids, n->grid_cols);
    else
        mv_draw_stack_children(fp, inner_x, inner_y, inner_w,
                               n->kids, n->n_kids);
}

static void mv_emit(mv_t *m, const mv_node_t *n) {
    mv_draw_node(m->fp, MV_MARGIN_X, m->y, MV_BOX_W, n);
    m->y += mv_node_height(n) + MV_GAP;
}

#define MV_STACK_H   56
#define MV_STACK_OFF 5

/* Collapsed identical middle layers — stacked-card look, not expanded. */
static int mv_collapsed_stack_height(void) {
    return MV_STACK_H + 2 * MV_STACK_OFF;
}

static void mv_emit_collapsed_stack(mv_t *m, const char *title,
                                    const char *shape, const char *fill) {
    int x = MV_MARGIN_X;
    int y = m->y;
    int w = MV_BOX_W;
    int h = MV_STACK_H;
    /* back cards */
    for (int i = 2; i >= 1; i--) {
        fprintf(m->fp,
                "  <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
                "rx=\"6\" ry=\"6\" fill=\"%s\" stroke=\"#94a3b8\" "
                "stroke-width=\"1\" opacity=\"0.85\"/>\n",
                x + i * MV_STACK_OFF, y + i * MV_STACK_OFF, w, h, fill);
    }
    mv_draw_leaf_box(m->fp, x, y, w, h, title, shape, fill, 13);
    m->y += mv_collapsed_stack_height() + MV_GAP;
}

typedef struct {
    char path[512];
    char tmp[576];
    char heading[192];
} mv_meta_t;

static mv_meta_t g_mv_meta;

/*
 * Two-phase SVG: body -> path.part, then mv_end wraps with correct height.
 */
static int mv_begin(mv_t *m, const char *path, const char *heading) {
    snprintf(g_mv_meta.path, sizeof g_mv_meta.path, "%s", path);
    snprintf(g_mv_meta.tmp, sizeof g_mv_meta.tmp, "%s.part", path);
    snprintf(g_mv_meta.heading, sizeof g_mv_meta.heading, "%s",
             heading ? heading : "");
    m->fp = fopen(g_mv_meta.tmp, "wb");
    if (!m->fp) {
        fprintf(stderr, "view: cannot write '%s'\n", g_mv_meta.tmp);
        return -1;
    }
    m->y = 0;
    return 0;
}

static int mv_end(mv_t *m) {
    int content_h = m->y;
    if (content_h < 0) content_h = 0;
    fclose(m->fp);
    m->fp = NULL;

    FILE *body = fopen(g_mv_meta.tmp, "rb");
    if (!body) {
        fprintf(stderr, "view: cannot re-open '%s'\n", g_mv_meta.tmp);
        return -1;
    }
    FILE *out = fopen(g_mv_meta.path, "wb");
    if (!out) {
        fprintf(stderr, "view: cannot write '%s'\n", g_mv_meta.path);
        fclose(body);
        remove(g_mv_meta.tmp);
        return -1;
    }

    int h = MV_MARGIN_Y + MV_TITLE_H + content_h + MV_MARGIN_Y;
    fprintf(out,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" "
            "viewBox=\"0 0 %d %d\">\n"
            "  <rect width=\"100%%\" height=\"100%%\" fill=\"#f8fafc\"/>\n",
            MV_W, h, MV_W, h);
    fprintf(out,
            "  <text x=\"%d\" y=\"%d\" "
            "font-family=\"Segoe UI,Helvetica,Arial,sans-serif\" "
            "font-size=\"16\" font-weight=\"700\" fill=\"#0f172a\">",
            MV_MARGIN_X, MV_MARGIN_Y + 8);
    svg_esc(out, g_mv_meta.heading);
    fputs("</text>\n", out);
    fprintf(out, "  <g transform=\"translate(0,%d)\">\n",
            MV_MARGIN_Y + MV_TITLE_H);

    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, body)) > 0) {
        if (fwrite(buf, 1, n, out) != n) {
            fclose(body);
            fclose(out);
            remove(g_mv_meta.tmp);
            return -1;
        }
    }
    fputs("  </g>\n</svg>\n", out);
    fclose(body);
    fclose(out);
    remove(g_mv_meta.tmp);
    return 0;
}

/* Fill head_nodes[0..n_heads) with titles "Head i" and shape d_kv. */
static void mv_fill_heads(mv_node_t *head_nodes, char (*titles)[32],
                          char (*shapes)[48], uint32_t n_heads,
                          uint32_t d_kv, const char *fill) {
    for (uint32_t i = 0; i < n_heads; i++) {
        snprintf(titles[i], 32, "Head %u", i);
        snprintf(shapes[i], 48, "d_kv=%u", d_kv);
        head_nodes[i].title = titles[i];
        head_nodes[i].shape = shapes[i];
        head_nodes[i].fill = fill;
        head_nodes[i].child_layout = MV_LAYOUT_STACK;
        head_nodes[i].grid_cols = 0;
        head_nodes[i].kids = NULL;
        head_nodes[i].n_kids = 0;
    }
}

static void mv_leaf_node(mv_node_t *n, const char *title, const char *shape,
                         const char *fill) {
    n->title = title;
    n->shape = shape;
    n->fill = fill;
    n->child_layout = MV_LAYOUT_STACK;
    n->grid_cols = 0;
    n->kids = NULL;
    n->n_kids = 0;
}

/*
 * Qwen SwiGLU: gate / up / SiLU⊙ / down.
 * Writes up to 4 children into kids[]; titles/shapes need room for 4.
 */
static int mv_fill_qwen_ffn(mv_node_t *kids, char (*titles)[40],
                            char (*shapes)[80], uint32_t hidden,
                            uint32_t inter, const char *act) {
    snprintf(titles[0], 40, "gate_proj");
    snprintf(shapes[0], 80, "[%u, %u]", inter, hidden);
    snprintf(titles[1], 40, "up_proj");
    snprintf(shapes[1], 80, "[%u, %u]", inter, hidden);
    snprintf(titles[2], 40, "Act");
    snprintf(shapes[2], 80, "%s(gate) * up", act && act[0] ? act : "silu");
    snprintf(titles[3], 40, "down_proj");
    snprintf(shapes[3], 80, "[%u, %u]", hidden, inter);
    mv_leaf_node(&kids[0], titles[0], shapes[0], "#e2e8f0");
    mv_leaf_node(&kids[1], titles[1], shapes[1], "#e2e8f0");
    mv_leaf_node(&kids[2], titles[2], shapes[2], "#fef9c3");
    mv_leaf_node(&kids[3], titles[3], shapes[3], "#e2e8f0");
    return 4;
}

/*
 * T5 DenseReluDense: relu => wi / ReLU / wo;
 * gated-* => wi_0 / wi_1 / gated-act / wo.
 */
static int mv_fill_t5_ffn(mv_node_t *kids, char (*titles)[40],
                          char (*shapes)[80], uint32_t d_model,
                          uint32_t d_ff, const char *proj) {
    int gated = (proj && strstr(proj, "gated") != NULL);
    if (gated) {
        snprintf(titles[0], 40, "wi_0");
        snprintf(shapes[0], 80, "[%u, %u]", d_ff, d_model);
        snprintf(titles[1], 40, "wi_1");
        snprintf(shapes[1], 80, "[%u, %u]", d_ff, d_model);
        snprintf(titles[2], 40, "Act");
        snprintf(shapes[2], 80, "%s(wi_0) * wi_1",
                 proj[0] ? proj : "gated-gelu");
        snprintf(titles[3], 40, "wo");
        snprintf(shapes[3], 80, "[%u, %u]", d_model, d_ff);
        mv_leaf_node(&kids[0], titles[0], shapes[0], "#e0f2fe");
        mv_leaf_node(&kids[1], titles[1], shapes[1], "#e0f2fe");
        mv_leaf_node(&kids[2], titles[2], shapes[2], "#fef9c3");
        mv_leaf_node(&kids[3], titles[3], shapes[3], "#e0f2fe");
        return 4;
    }
    snprintf(titles[0], 40, "wi");
    snprintf(shapes[0], 80, "[%u, %u]", d_ff, d_model);
    snprintf(titles[1], 40, "Act");
    snprintf(shapes[1], 80, "%s", proj && proj[0] ? proj : "relu");
    snprintf(titles[2], 40, "wo");
    snprintf(shapes[2], 80, "[%u, %u]", d_model, d_ff);
    mv_leaf_node(&kids[0], titles[0], shapes[0], "#e0f2fe");
    mv_leaf_node(&kids[1], titles[1], shapes[1], "#fef9c3");
    mv_leaf_node(&kids[2], titles[2], shapes[2], "#e0f2fe");
    return 3;
}

static void mv_emit_qwen_layer(mv_t *m, const cp_config_t *c, uint32_t li) {
    char title[64];
    char s_in[96], s_attn[160], s_post[96], s_mlp[128];
    mv_node_t heads[64];
    char htitle[64][32], hshape[64][48];
    mv_node_t ffn_kids[4];
    char ft[4][40], fs[4][80];
    uint32_t nh = c->num_attention_heads;
    if (nh > 64) nh = 64;

    snprintf(title, sizeof title, "Decoder Layer %u", li);
    snprintf(s_in, sizeof s_in, "weight  [%u]  eps=%g",
             c->hidden_size, c->rms_norm_eps);
    snprintf(s_attn, sizeof s_attn,
             "GQA  q_heads=%u kv_heads=%u  Wq/Wk/Wv/Wo",
             c->num_attention_heads, c->num_key_value_heads);
    snprintf(s_post, sizeof s_post, "weight  [%u]  eps=%g",
             c->hidden_size, c->rms_norm_eps);
    snprintf(s_mlp, sizeof s_mlp, "%s  [%u] -> [%u] -> [%u]",
             c->hidden_act, c->hidden_size, c->intermediate_size,
             c->hidden_size);

    mv_fill_heads(heads, htitle, hshape, nh, c->head_dim, "#e2e8f0");
    int n_ffn = mv_fill_qwen_ffn(ffn_kids, ft, fs, c->hidden_size,
                                 c->intermediate_size, c->hidden_act);

    mv_node_t attn = {
        "Self-Attention", s_attn, "#f1f5f9",
        MV_LAYOUT_GRID, MV_HEAD_COLS, heads, (int)nh
    };
    mv_node_t ffn = {
        "MLP (SwiGLU)", s_mlp, "#f8fafc",
        MV_LAYOUT_STACK, 0, ffn_kids, n_ffn
    };
    mv_node_t kids[4] = {
        { "Input RMSNorm", s_in,   "#fef9c3", MV_LAYOUT_STACK, 0, NULL, 0 },
        attn,
        { "Post RMSNorm",  s_post, "#fef9c3", MV_LAYOUT_STACK, 0, NULL, 0 },
        ffn,
    };
    mv_node_t layer = {
        title, NULL, (li % 2) ? "#f8fafc" : "#ffffff",
        MV_LAYOUT_STACK, 0, kids, 4
    };
    mv_emit(m, &layer);
}

int cp_model_view_qwen(const cp_config_t *c, const char *path) {
    if (!c || !path) return -1;

    char heading[160];
    snprintf(heading, sizeof heading, "%s — %s",
             c->model_type, cp_arch_name(c->arch));

    mv_t m;
    if (mv_begin(&m, path, heading) != 0) return -1;

    char emb_shape[160];
    snprintf(emb_shape, sizeof emb_shape,
             "embed_tokens  [%u, %u]  max_pos=%u",
             c->vocab_size, c->hidden_size, c->max_position_embeddings);
    {
        mv_node_t emb = {
            "Embedding", emb_shape, "#dbeafe",
            MV_LAYOUT_STACK, 0, NULL, 0
        };
        mv_emit(&m, &emb);
    }

    /* First + last expanded; identical middle layers collapsed into a stack. */
    uint32_t nL = c->num_hidden_layers;
    if (nL == 0) {
        /* nothing */
    } else if (nL == 1) {
        mv_emit_qwen_layer(&m, c, 0);
    } else if (nL == 2) {
        mv_emit_qwen_layer(&m, c, 0);
        mv_emit_qwen_layer(&m, c, 1);
    } else {
        char stitle[96], sshape[128];
        uint32_t mid = nL - 2;
        snprintf(stitle, sizeof stitle,
                 "Decoder Layers 1 .. %u", nL - 2);
        snprintf(sshape, sizeof sshape,
                 "x%u identical  (same structure as Layer 0)", mid);
        mv_emit_qwen_layer(&m, c, 0);
        mv_emit_collapsed_stack(&m, stitle, sshape, "#e2e8f0");
        mv_emit_qwen_layer(&m, c, nL - 1);
    }

    {
        char shape[160];
        snprintf(shape, sizeof shape, "weight  [%u]  eps=%g  rope_theta=%g",
                 c->hidden_size, c->rms_norm_eps, c->rope_theta);
        mv_node_t n = {
            "Final RMSNorm", shape, "#fef3c7",
            MV_LAYOUT_STACK, 0, NULL, 0
        };
        mv_emit(&m, &n);
    }
    {
        char shape[160];
        if (c->tie_word_embeddings)
            snprintf(shape, sizeof shape, "tied to embed_tokens  [%u, %u]",
                     c->vocab_size, c->hidden_size);
        else
            snprintf(shape, sizeof shape, "lm_head  [%u, %u]",
                     c->vocab_size, c->hidden_size);
        mv_node_t n = {
            c->tie_word_embeddings ? "LM Head (tied)" : "LM Head",
            shape, "#dcfce7", MV_LAYOUT_STACK, 0, NULL, 0
        };
        mv_emit(&m, &n);
    }

    return mv_end(&m);
}

static void mv_emit_t5_encoder(mv_t *m, const cp_t5_config_t *c, uint32_t i) {
    char title[64], s_attn[176], s_ffn[128];
    mv_node_t heads[64];
    char htitle[64][32], hshape[64][48];
    mv_node_t ffn_kids[4];
    char ft[4][40], fs[4][80];
    uint32_t n_heads = c->num_heads;
    if (n_heads > 64) n_heads = 64;

    snprintf(title, sizeof title, "Encoder Block %u", i);
    if (i == 0)
        snprintf(s_attn, sizeof s_attn,
                 "heads=%u  Q/K/V/O [%u,%u]  + rel_bias[%u]",
                 c->num_heads, c->d_model, c->d_model,
                 c->relative_attention_num_buckets);
    else
        snprintf(s_attn, sizeof s_attn,
                 "heads=%u  Q/K/V/O [%u,%u]  (rel_bias shared)",
                 c->num_heads, c->d_model, c->d_model);
    snprintf(s_ffn, sizeof s_ffn, "DenseReluDense (%s)", c->feed_forward_proj);

    mv_fill_heads(heads, htitle, hshape, n_heads, c->d_kv, "#bae6fd");
    int n_ffn = mv_fill_t5_ffn(ffn_kids, ft, fs, c->d_model, c->d_ff,
                               c->feed_forward_proj);

    mv_node_t attn = {
        "Self-Attention", s_attn, "#e0f2fe",
        MV_LAYOUT_GRID, MV_HEAD_COLS, heads, (int)n_heads
    };
    mv_node_t ffn = {
        "FFN", s_ffn, "#f0f9ff",
        MV_LAYOUT_STACK, 0, ffn_kids, n_ffn
    };
    mv_node_t kids[2] = { attn, ffn };
    mv_node_t block = {
        title, NULL, "#f8fafc",
        MV_LAYOUT_STACK, 0, kids, 2
    };
    mv_emit(m, &block);
}

static void mv_emit_t5_decoder(mv_t *m, const cp_t5_config_t *c, uint32_t i) {
    char title[64], s_self[176], s_cross[176], s_ffn[128];
    mv_node_t heads_self[64], heads_cross[64];
    char ht1[64][32], hs1[64][48], ht2[64][32], hs2[64][48];
    mv_node_t ffn_kids[4];
    char ft[4][40], fs[4][80];
    uint32_t n_heads = c->num_heads;
    if (n_heads > 64) n_heads = 64;

    snprintf(title, sizeof title, "Decoder Block %u", i);
    if (i == 0)
        snprintf(s_self, sizeof s_self,
                 "causal  heads=%u  + rel_bias[%u]",
                 c->num_heads, c->relative_attention_num_buckets);
    else
        snprintf(s_self, sizeof s_self,
                 "causal  heads=%u  (rel_bias shared)", c->num_heads);
    snprintf(s_cross, sizeof s_cross,
             "EncDec  heads=%u  Q<-dec  K/V<-enc  [%u]",
             c->num_heads, c->d_model);
    snprintf(s_ffn, sizeof s_ffn, "DenseReluDense (%s)", c->feed_forward_proj);

    mv_fill_heads(heads_self, ht1, hs1, n_heads, c->d_kv, "#e9d5ff");
    mv_fill_heads(heads_cross, ht2, hs2, n_heads, c->d_kv, "#f5d0fe");
    int n_ffn = mv_fill_t5_ffn(ffn_kids, ft, fs, c->d_model, c->d_ff,
                               c->feed_forward_proj);

    mv_node_t self_attn = {
        "Self-Attention", s_self, "#f3e8ff",
        MV_LAYOUT_GRID, MV_HEAD_COLS, heads_self, (int)n_heads
    };
    mv_node_t cross_attn = {
        "Cross-Attention", s_cross, "#fae8ff",
        MV_LAYOUT_GRID, MV_HEAD_COLS, heads_cross, (int)n_heads
    };
    mv_node_t ffn = {
        "FFN", s_ffn, "#fdf4ff",
        MV_LAYOUT_STACK, 0, ffn_kids, n_ffn
    };
    mv_node_t kids[3] = { self_attn, cross_attn, ffn };
    mv_node_t block = {
        title, NULL, "#faf5ff",
        MV_LAYOUT_STACK, 0, kids, 3
    };
    mv_emit(m, &block);
}

static void mv_emit_repeated_blocks(mv_t *m, uint32_t n,
                                    void (*emit_one)(mv_t *, const void *,
                                                     uint32_t),
                                    const void *cfg, const char *kind,
                                    const char *fill) {
    if (n == 0) return;
    if (n == 1) {
        emit_one(m, cfg, 0);
        return;
    }
    if (n == 2) {
        emit_one(m, cfg, 0);
        emit_one(m, cfg, 1);
        return;
    }
    char stitle[96], sshape[128];
    snprintf(stitle, sizeof stitle, "%s 1 .. %u", kind, n - 2);
    snprintf(sshape, sizeof sshape,
             "x%u identical  (same structure as block 0)", n - 2);
    emit_one(m, cfg, 0);
    mv_emit_collapsed_stack(m, stitle, sshape, fill);
    emit_one(m, cfg, n - 1);
}

static void mv_emit_t5_enc_wrap(mv_t *m, const void *cfg, uint32_t i) {
    mv_emit_t5_encoder(m, (const cp_t5_config_t *)cfg, i);
}
static void mv_emit_t5_dec_wrap(mv_t *m, const void *cfg, uint32_t i) {
    mv_emit_t5_decoder(m, (const cp_t5_config_t *)cfg, i);
}

int cp_model_view_t5(const cp_t5_config_t *c, const char *path) {
    if (!c || !path) return -1;

    char heading[160];
    snprintf(heading, sizeof heading, "%s — %s",
             c->model_type, cp_arch_name(c->arch));

    mv_t m;
    if (mv_begin(&m, path, heading) != 0) return -1;

    {
        char shape[192];
        snprintf(shape, sizeof shape,
                 "shared.weight  [%u, %u]  pad=%u eos=%u start=%u",
                 c->vocab_size, c->d_model, c->pad_token_id, c->eos_token_id,
                 c->decoder_start_token_id);
        mv_node_t emb = {
            "Shared Embedding", shape, "#dbeafe",
            MV_LAYOUT_STACK, 0, NULL, 0
        };
        mv_emit(&m, &emb);
    }

    mv_emit_repeated_blocks(&m, c->num_layers, mv_emit_t5_enc_wrap, c,
                            "Encoder Blocks", "#e0f2fe");
    mv_emit_repeated_blocks(&m, c->num_decoder_layers, mv_emit_t5_dec_wrap, c,
                            "Decoder Blocks", "#fae8ff");

    {
        char s_enc[96], s_dec[96], s_lm[96];
        snprintf(s_enc, sizeof s_enc, "weight  [%u]  eps=%g",
                 c->d_model, c->layer_norm_epsilon);
        snprintf(s_dec, sizeof s_dec, "weight  [%u]  eps=%g",
                 c->d_model, c->layer_norm_epsilon);
        snprintf(s_lm, sizeof s_lm, "tied to shared  [%u, %u]",
                 c->vocab_size, c->d_model);
        mv_node_t kids[3] = {
            { "Encoder Final LN", s_enc, "#fef9c3", MV_LAYOUT_STACK, 0, NULL, 0 },
            { "Decoder Final LN", s_dec, "#fef9c3", MV_LAYOUT_STACK, 0, NULL, 0 },
            { "LM Head (tied)",   s_lm,  "#dcfce7", MV_LAYOUT_STACK, 0, NULL, 0 },
        };
        mv_node_t block = {
            "Output Head", NULL, "#ffffff",
            MV_LAYOUT_STACK, 0, kids, 3
        };
        mv_emit(&m, &block);
    }

    return mv_end(&m);
}
