/*
 * api/chat_completions.c - POST /v1/chat/completions
 *
 * Qwen ChatML prompt from messages[]; sampling via sample.h.
 * Supports stream=false JSON and stream=true SSE.
 */

#include "api/chat_completions.h"
#include "api/runtime.h"
#include "cJSON.h"
#include "sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    struct mg_connection *c;
    cp_tokenizer_t       *tok;
    int                   stream;
    char                  id[64];
    const char           *model;
    long                  created;
    int                   sent_role;
} stream_state_t;

static void reply_err(struct mg_connection *c, int status, const char *msg,
                      const char *code) {
    cJSON *root = cJSON_CreateObject();
    cJSON *err = cJSON_CreateObject();
    if (!root || !err) {
        cJSON_Delete(root);
        cJSON_Delete(err);
        cp_api_reply_json(c, status,
                          "{\"error\":{\"message\":\"oom\",\"type\":\"server_error\"}}");
        return;
    }
    cJSON_AddStringToObject(err, "message", msg ? msg : "error");
    cJSON_AddStringToObject(err, "type",
                            status >= 500 ? "server_error"
                                          : "invalid_request_error");
    if (code) cJSON_AddStringToObject(err, "code", code);
    cJSON_AddItemToObject(root, "error", err);
    char *s = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    cp_api_reply_json(c, status, s ? s : "{}");
    free(s);
}

/* Build Qwen ChatML from OpenAI messages array. */
static char *build_chatml(const cJSON *messages) {
    if (!cJSON_IsArray(messages) || cJSON_GetArraySize(messages) == 0)
        return NULL;

    size_t cap = 4096, n = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    const cJSON *msg;
    cJSON_ArrayForEach(msg, messages) {
        const cJSON *role = cJSON_GetObjectItemCaseSensitive(msg, "role");
        const cJSON *content = cJSON_GetObjectItemCaseSensitive(msg, "content");
        if (!cJSON_IsString(role) || !role->valuestring) continue;
        const char *text = "";
        if (cJSON_IsString(content) && content->valuestring)
            text = content->valuestring;
        /* ignore non-string content (multimodal) for now */

        size_t need = strlen(role->valuestring) + strlen(text) + 64;
        if (n + need + 1 > cap) {
            size_t ncap = (n + need) * 2 + 1024;
            char *nb = realloc(buf, ncap);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
            cap = ncap;
        }
        int w = snprintf(buf + n, cap - n, "<|im_start|>%s\n%s<|im_end|>\n",
                         role->valuestring, text);
        if (w < 0) {
            free(buf);
            return NULL;
        }
        n += (size_t)w;
    }

    {
        const char *tail = "<|im_start|>assistant\n";
        size_t tlen = strlen(tail);
        if (n + tlen + 1 > cap) {
            char *nb = realloc(buf, n + tlen + 1);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
        }
        memcpy(buf + n, tail, tlen + 1);
    }
    return buf;
}

static void on_token_stream(uint32_t id, size_t pos, void *userdata) {
    stream_state_t *st = (stream_state_t *)userdata;
    (void)pos;
    char piece[256];
    if (cp_tokenizer_decode(st->tok, &id, 1, piece, sizeof piece, 0) < 0)
        return;

    if (!st->sent_role) {
        cJSON *root = cJSON_CreateObject();
        cJSON *choices = cJSON_CreateArray();
        cJSON *ch = cJSON_CreateObject();
        cJSON *delta = cJSON_CreateObject();
        cJSON_AddStringToObject(root, "id", st->id);
        cJSON_AddStringToObject(root, "object", "chat.completion.chunk");
        cJSON_AddNumberToObject(root, "created", (double)st->created);
        cJSON_AddStringToObject(root, "model", st->model);
        cJSON_AddStringToObject(delta, "role", "assistant");
        cJSON_AddItemToObject(ch, "delta", delta);
        cJSON_AddNumberToObject(ch, "index", 0);
        cJSON_AddNullToObject(ch, "finish_reason");
        cJSON_AddItemToArray(choices, ch);
        cJSON_AddItemToObject(root, "choices", choices);
        char *js = cJSON_PrintUnformatted(root);
        cJSON_Delete(root);
        if (js) {
            mg_http_printf_chunk(st->c, "data: %s\r\n\r\n", js);
            free(js);
        }
        st->sent_role = 1;
    }

    cJSON *root = cJSON_CreateObject();
    cJSON *choices = cJSON_CreateArray();
    cJSON *ch = cJSON_CreateObject();
    cJSON *delta = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "id", st->id);
    cJSON_AddStringToObject(root, "object", "chat.completion.chunk");
    cJSON_AddNumberToObject(root, "created", (double)st->created);
    cJSON_AddStringToObject(root, "model", st->model);
    cJSON_AddStringToObject(delta, "content", piece);
    cJSON_AddItemToObject(ch, "delta", delta);
    cJSON_AddNumberToObject(ch, "index", 0);
    cJSON_AddNullToObject(ch, "finish_reason");
    cJSON_AddItemToArray(choices, ch);
    cJSON_AddItemToObject(root, "choices", choices);
    char *js = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (js) {
        mg_http_printf_chunk(st->c, "data: %s\r\n\r\n", js);
        free(js);
    }
}

static void send_stream_done(stream_state_t *st, const char *finish) {
    cJSON *root = cJSON_CreateObject();
    cJSON *choices = cJSON_CreateArray();
    cJSON *ch = cJSON_CreateObject();
    cJSON *delta = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "id", st->id);
    cJSON_AddStringToObject(root, "object", "chat.completion.chunk");
    cJSON_AddNumberToObject(root, "created", (double)st->created);
    cJSON_AddStringToObject(root, "model", st->model);
    cJSON_AddItemToObject(ch, "delta", delta);
    cJSON_AddNumberToObject(ch, "index", 0);
    cJSON_AddStringToObject(ch, "finish_reason", finish ? finish : "stop");
    cJSON_AddItemToArray(choices, ch);
    cJSON_AddItemToObject(root, "choices", choices);
    char *js = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (js) {
        mg_http_printf_chunk(st->c, "data: %s\r\n\r\n", js);
        free(js);
    }
    mg_http_printf_chunk(st->c, "data: [DONE]\r\n\r\n");
    mg_http_printf_chunk(st->c, "");
}

int api_chat_completions_handle(struct mg_connection *c,
                                struct mg_http_message *hm,
                                cp_api_ctx_t *ctx) {
    if (!(mg_match(hm->uri, mg_str("/v1/chat/completions"), NULL) &&
          mg_strcmp(hm->method, mg_str("POST")) == 0))
        return 0;

    cJSON *body = cJSON_ParseWithLength(hm->body.buf, hm->body.len);
    if (!body || !cJSON_IsObject(body)) {
        cJSON_Delete(body);
        reply_err(c, 400, "invalid JSON body", "invalid_request_error");
        return 1;
    }

    const cJSON *jmodel = cJSON_GetObjectItemCaseSensitive(body, "model");
    const cJSON *jmsgs = cJSON_GetObjectItemCaseSensitive(body, "messages");
    if (!cJSON_IsString(jmodel) || !jmodel->valuestring ||
        !cJSON_IsArray(jmsgs)) {
        cJSON_Delete(body);
        reply_err(c, 400, "model and messages are required", NULL);
        return 1;
    }

    int stream = 0;
    const cJSON *jstream = cJSON_GetObjectItemCaseSensitive(body, "stream");
    if (cJSON_IsTrue(jstream)) stream = 1;

    int max_tokens = 64;
    const cJSON *jmax = cJSON_GetObjectItemCaseSensitive(body, "max_tokens");
    if (cJSON_IsNumber(jmax) && jmax->valuedouble > 0)
        max_tokens = (int)jmax->valuedouble;

    cp_sample_cfg_t sample;
    cp_sample_cfg_default(&sample);
    const cJSON *jtemp = cJSON_GetObjectItemCaseSensitive(body, "temperature");
    if (cJSON_IsNumber(jtemp)) sample.temperature = (float)jtemp->valuedouble;
    const cJSON *jtop_p = cJSON_GetObjectItemCaseSensitive(body, "top_p");
    if (cJSON_IsNumber(jtop_p)) sample.top_p = (float)jtop_p->valuedouble;
    const cJSON *jtop_k = cJSON_GetObjectItemCaseSensitive(body, "top_k");
    if (cJSON_IsNumber(jtop_k)) sample.top_k = (int)jtop_k->valuedouble;
    const cJSON *jseed = cJSON_GetObjectItemCaseSensitive(body, "seed");
    if (cJSON_IsNumber(jseed) && jseed->valuedouble >= 0)
        sample.seed = (uint64_t)jseed->valuedouble;

    char *prompt = build_chatml(jmsgs);
    if (!prompt) {
        cJSON_Delete(body);
        reply_err(c, 400, "failed to build chat prompt from messages", NULL);
        return 1;
    }

    char err[256];
    cp_api_runtime_t *rt = NULL;
    if (cp_api_runtime_acquire(ctx, jmodel->valuestring, &rt, err,
                               sizeof err) != 0) {
        free(prompt);
        cJSON_Delete(body);
        reply_err(c, 404, err, "model_not_found");
        return 1;
    }

    cp_tokenizer_t *tok = cp_api_runtime_tokenizer(rt);
    cp_session_t *sess = cp_api_runtime_session(rt);
    size_t id_cap = strlen(prompt) * 4 + 64;
    uint32_t *ids = malloc(id_cap * sizeof *ids);
    if (!ids) {
        free(prompt);
        cJSON_Delete(body);
        reply_err(c, 500, "out of memory", NULL);
        return 1;
    }
    int n_prompt = cp_tokenizer_encode(tok, prompt, ids, id_cap, 0);
    free(prompt);
    if (n_prompt < 0) {
        free(ids);
        cJSON_Delete(body);
        reply_err(c, 400, "tokenize prompt failed", NULL);
        return 1;
    }

    long created = (long)time(NULL);
    char cmpl_id[64];
    snprintf(cmpl_id, sizeof cmpl_id, "chatcmpl-%ld", created);

    uint32_t *out_ids = malloc((size_t)max_tokens * sizeof *out_ids);
    if (!out_ids) {
        free(ids);
        cJSON_Delete(body);
        reply_err(c, 500, "out of memory", NULL);
        return 1;
    }

    cp_session_reset(sess);

    if (stream) {
        stream_state_t st;
        memset(&st, 0, sizeof st);
        st.c = c;
        st.tok = tok;
        st.stream = 1;
        st.model = jmodel->valuestring;
        st.created = created;
        snprintf(st.id, sizeof st.id, "%s", cmpl_id);

        mg_printf(c,
                  "HTTP/1.1 200 OK\r\n"
                  "Content-Type: text/event-stream\r\n"
                  "Cache-Control: no-cache\r\n"
                  "Access-Control-Allow-Origin: *\r\n"
                  "Transfer-Encoding: chunked\r\n\r\n");

        int n_gen = cp_generate(sess, ids, (size_t)n_prompt, out_ids,
                                (size_t)max_tokens, max_tokens, &sample,
                                on_token_stream, &st);
        if (n_gen < 0) {
            mg_http_printf_chunk(
                    c,
                    "data: {\"error\":{\"message\":\"generate failed\"}}\r\n\r\n");
            mg_http_printf_chunk(c, "data: [DONE]\r\n\r\n");
            mg_http_printf_chunk(c, "");
        } else {
            send_stream_done(&st,
                             (n_gen >= max_tokens) ? "length" : "stop");
        }
        free(out_ids);
        free(ids);
        cJSON_Delete(body);
        return 1;
    }

    int n_gen = cp_generate(sess, ids, (size_t)n_prompt, out_ids,
                            (size_t)max_tokens, max_tokens, &sample, NULL,
                            NULL);
    if (n_gen < 0) {
        free(out_ids);
        free(ids);
        cJSON_Delete(body);
        reply_err(c, 500, "generate failed", NULL);
        return 1;
    }

    size_t text_cap = (size_t)n_gen * 16 + 64;
    char *text = malloc(text_cap);
    if (!text) {
        free(out_ids);
        free(ids);
        cJSON_Delete(body);
        reply_err(c, 500, "out of memory", NULL);
        return 1;
    }
    text[0] = '\0';
    if (n_gen > 0) {
        if (cp_tokenizer_decode(tok, out_ids, (size_t)n_gen, text, text_cap,
                                0) < 0)
            text[0] = '\0';
    }

    cJSON *root = cJSON_CreateObject();
    cJSON *choices = cJSON_CreateArray();
    cJSON *ch = cJSON_CreateObject();
    cJSON *message = cJSON_CreateObject();
    cJSON *usage = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "id", cmpl_id);
    cJSON_AddStringToObject(root, "object", "chat.completion");
    cJSON_AddNumberToObject(root, "created", (double)created);
    cJSON_AddStringToObject(root, "model", jmodel->valuestring);
    cJSON_AddStringToObject(message, "role", "assistant");
    cJSON_AddStringToObject(message, "content", text);
    cJSON_AddItemToObject(ch, "message", message);
    cJSON_AddNumberToObject(ch, "index", 0);
    cJSON_AddStringToObject(ch, "finish_reason",
                            (n_gen >= max_tokens) ? "length" : "stop");
    cJSON_AddItemToArray(choices, ch);
    cJSON_AddItemToObject(root, "choices", choices);
    cJSON_AddNumberToObject(usage, "prompt_tokens", n_prompt);
    cJSON_AddNumberToObject(usage, "completion_tokens", n_gen);
    cJSON_AddNumberToObject(usage, "total_tokens", n_prompt + n_gen);
    cJSON_AddItemToObject(root, "usage", usage);

    char *js = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    cp_api_reply_json(c, 200, js ? js : "{}");
    free(js);
    free(text);
    free(out_ids);
    free(ids);
    cJSON_Delete(body);
    return 1;
}
