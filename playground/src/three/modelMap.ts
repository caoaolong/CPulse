/* modelMap.ts - 从 cpulse 生成的 header.json 构建模型结构描述。
 *
 * header.json 由 src/safetensors.c 写入 playground/public/header.json，
 * Vite 原样 serving，前端用 fetch 即可拿到。拿不到（未生成/路径不对）
 * 时回退到与 config.json 一致的内置默认值，场景照常可看。
 *
 * 粒度：每个 tensor 一个 Box（290 tensors -> 291 块，含 tied head）。
 * 列：embed | L0..L23 | norm | head；列内按数据流顺序堆叠。
 *
 * 尺寸规则（全局统一）：h=KY*sqrt(rows)，d=KY*sqrt(cols)，
 * 厚(X)=KX*cbrt(params)；场景里 Y 取 d、Z 取 h（高↔深对调过）。
 * 1D tensor（bias/norm）按 cols=1 处理，成薄片。
 *
 * 颜色规则：attn 系绿→青、mlp 系红→橙、norm 系紫（均按层号渐变），
 * 系内各 tensor 再微调明度区分；embed 深蓝 / head 品红固定。
 */

import { Color } from 'three';

export interface HeaderTensor {
    name: string;
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
    nbytes: number;
    data_offset: number;
}

export interface HeaderJson {
    header_len: number;
    data_start: number;
    n_tensors: number;
    tensors: HeaderTensor[];
}

export interface BlockSpec {
    /** 短名，如 'L3.attn.q_proj'（3D 标签拼 shape 展示） */
    name: string;
    /** 形状串，如 '[896, 896]' */
    shape: string;
    /** 数据类型，如 'BF16' */
    dtype: string;
    /** Box 颜色 */
    color: number;
    /** 所属列（0=embed, 1..N=层, N+1=norm, N+2=head） */
    col: number;
    /** 原始参数量（外包盒汇总标签用） */
    params: number;
    /** X 厚度（∝ cbrt(params)） */
    w: number;
    /** Y 高度（场景里取 d，∝ sqrt(cols)） */
    h: number;
    /** Z 深度（场景里取 h，∝ sqrt(rows)） */
    d: number;
}

/** 选中信息：右上角 HUD 只显示这三项 */
export interface SelInfo {
    title: string;
    shape: string;
    dtype: string;
}

export interface ModelSpec {
    vocabSize: number;
    hiddenSize: number;
    numLayers: number;
    intermediateSize: number;
    totalParams: number;
    blocks: BlockSpec[];
    colNames: string[];
    /** true = 来自 header.json，false = 内置回退值 */
    fromFile: boolean;
}

const C_EMBED = 0x2a4a7f;
const C_NORM_TOP = 0x6d5bd0;
const C_HEAD = 0xd946ef;

const KY: number = 0.16;
const KX: number = 6 / Math.cbrt(136134656); // embed 厚度归一到 6
const MIN_W: number = 1.0;
const MIN_HD: number = 2.5;

type Family = 'attn' | 'mlp' | 'norm';

/** 系内明度微调：同层同系各 tensor 再区分 */
const LIGHT_TWEAK: Record<string, number> = {
    q_proj: 0, q_bias: -0.12, k_proj: 0.06, k_bias: -0.06,
    v_proj: 0.06, v_bias: -0.06, o_proj: -0.04,
    gate_proj: 0, up_proj: 0.06, down_proj: -0.06,
    layernorm: 0,
};

function familyColor(family: Family, layer: number, layers: number, tweakKey: string): number {
    const t: number = layers <= 1 ? 0 : layer / (layers - 1);
    let h: number, s: number, l: number;
    if (family === 'attn') { h = 0.38 + 0.14 * t; s = 0.6; l = 0.45; }
    else if (family === 'mlp') { h = 0.1 * t; s = 0.65; l = 0.5; }
    else { h = 0.75; s = 0.5; l = 0.35 + 0.15 * t; }
    const dl: number = LIGHT_TWEAK[tweakKey] ?? 0;
    return new Color().setHSL(h, s, l).offsetHSL(0, 0, dl).getHex();
}

/** (rows, cols, params) -> (w, h, d)，带下限保证可点击 */
function sizedBox(rows: number, cols: number, params: number): { w: number; h: number; d: number } {
    return {
        w: Math.max(MIN_W, KX * Math.cbrt(params)),
        h: Math.max(MIN_HD, KY * Math.sqrt(rows)),
        d: Math.max(MIN_HD, KY * Math.sqrt(cols)),
    };
}

function findTensor(h: HeaderJson, name: string): HeaderTensor | undefined {
    return h.tensors.find((t) => t.name === name);
}

function tensorParams(t: HeaderTensor): number {
    let p = 1;
    for (const d of t.shape) p *= d;
    return p;
}

/** 层内典型模板：后缀、简称、系、代表形状（回退用）、参数（回退用） */
interface LayerTemplate {
    suffix: string;
    short: string;
    family: Family;
    rows: number;
    cols: number;
    params: number;
}

const LAYER_TEMPLATE: LayerTemplate[] = [
    { suffix: 'input_layernorm.weight', short: 'norm_in', family: 'norm', rows: 896, cols: 1, params: 896 },
    { suffix: 'self_attn.q_proj.weight', short: 'attn.q', family: 'attn', rows: 896, cols: 896, params: 802816 },
    { suffix: 'self_attn.q_proj.bias', short: 'attn.q_b', family: 'attn', rows: 896, cols: 1, params: 896 },
    { suffix: 'self_attn.k_proj.weight', short: 'attn.k', family: 'attn', rows: 128, cols: 896, params: 114688 },
    { suffix: 'self_attn.k_proj.bias', short: 'attn.k_b', family: 'attn', rows: 128, cols: 1, params: 128 },
    { suffix: 'self_attn.v_proj.weight', short: 'attn.v', family: 'attn', rows: 128, cols: 896, params: 114688 },
    { suffix: 'self_attn.v_proj.bias', short: 'attn.v_b', family: 'attn', rows: 128, cols: 1, params: 128 },
    { suffix: 'self_attn.o_proj.weight', short: 'attn.o', family: 'attn', rows: 896, cols: 896, params: 802816 },
    { suffix: 'post_attention_layernorm.weight', short: 'norm_out', family: 'norm', rows: 896, cols: 1, params: 896 },
    { suffix: 'mlp.gate_proj.weight', short: 'mlp.gate', family: 'mlp', rows: 4864, cols: 896, params: 4358144 },
    { suffix: 'mlp.up_proj.weight', short: 'mlp.up', family: 'mlp', rows: 4864, cols: 896, params: 4358144 },
    { suffix: 'mlp.down_proj.weight', short: 'mlp.down', family: 'mlp', rows: 896, cols: 4864, params: 4358144 },
];

function tweakKeyOf(suffix: string): string {
    if (suffix.endsWith('.bias')) return suffix.split('.').slice(-2).join('_').replace('_proj', '');
    if (suffix === 'input_layernorm.weight' || suffix === 'post_attention_layernorm.weight') return 'layernorm';
    const parts = suffix.split('.');
    return parts.slice(-2).join('_').replace('_weight', '');
}

function buildBlocks(
    vocab: number, hidden: number, layers: number,
    getTensor: (fullName: string, t: LayerTemplate) => { dims: number[]; params: number; dtype: string },
    getOuter: (kind: 'embed' | 'norm' | 'head') => { dims: number[]; params: number; dtype: string; present: boolean }
): { blocks: BlockSpec[]; colNames: string[] } {
    /** dims -> rows/cols（1D 按 cols=1 薄片处理）+ 形状串 */
    const shapeOf = (dims: number[]): [number, number, string] => {
        const rows: number = dims[0] ?? 0;
        const cols: number = dims.length > 1 ? dims[1] : 1;
        return [rows, cols, `[${dims.join(', ')}]`];
    };
    const blocks: BlockSpec[] = [];
    const colNames: string[] = ['embed_tokens'];
    const embed = getOuter('embed');
    {
        const [rows, cols, shape] = shapeOf(embed.dims);
        blocks.push({
            name: 'embed_tokens', shape, dtype: embed.dtype,
            color: C_EMBED, col: 0, params: embed.params,
            ...sizedBox(rows, cols, embed.params),
        });
    }
    for (let i = 0; i < layers; i++) {
        colNames.push(`L${i}`);
        for (const t of LAYER_TEMPLATE) {
            const g = getTensor(`model.layers.${i}.${t.suffix}`, t);
            const [rows, cols, shape] = shapeOf(g.dims);
            blocks.push({
                name: `L${i}.${t.short}`,
                shape, dtype: g.dtype,
                color: familyColor(t.family, i, layers, tweakKeyOf(t.suffix)),
                col: i + 1, params: g.params,
                ...sizedBox(rows, cols, g.params),
            });
        }
    }
    const norm = getOuter('norm');
    {
        const [rows, cols, shape] = shapeOf(norm.dims);
        blocks.push({
            name: 'final norm', shape, dtype: norm.dtype,
            color: C_NORM_TOP, col: layers + 1, params: norm.params,
            ...sizedBox(rows, cols, norm.params),
        });
    }
    colNames.push('final norm');
    const head = getOuter('head');
    {
        const [rows, cols, shape] = shapeOf(head.dims);
        blocks.push({
            name: head.present ? 'lm_head' : 'lm_head (tied)',
            shape: head.present ? shape : '—', dtype: head.dtype,
            color: C_HEAD, col: layers + 2, params: head.params,
            ...sizedBox(rows, cols, head.params),
        });
    }
    colNames.push('lm_head');
    return { blocks, colNames };
}

/** 内置回退值：与 .cache/Qwen2.5-0.5B/config.json / header.json 实测一致 */
export function defaultSpec(): ModelSpec {
    const vocab = 151936, hidden = 896, inter = 4864, layers = 24;
    const { blocks, colNames } = buildBlocks(
        vocab, hidden, layers,
        (_name, t) => ({
            dims: t.cols === 1 ? [t.rows] : [t.rows, t.cols],
            params: t.params, dtype: 'BF16',
        }),
        (kind) => kind === 'embed'
            ? { dims: [vocab, hidden], params: 136134656, dtype: 'BF16', present: true }
            : kind === 'norm'
                ? { dims: [hidden], params: 896, dtype: 'BF16', present: true }
                : { dims: [vocab, hidden], params: 0, dtype: 'BF16', present: false }
    );
    void inter;
    return {
        vocabSize: vocab, hiddenSize: hidden, numLayers: layers,
        intermediateSize: inter, totalParams: 494032768,
        blocks, colNames, fromFile: false,
    };
}

function specFromHeader(h: HeaderJson): ModelSpec {
    const fb = defaultSpec();

    const embedT = findTensor(h, 'model.embed_tokens.weight');
    const vocab = embedT ? embedT.shape[0] : fb.vocabSize;
    const hidden = embedT ? embedT.shape[1] : fb.hiddenSize;

    let maxLayer = -1;
    for (const t of h.tensors) {
        const m = /^model\.layers\.(\d+)\./.exec(t.name);
        if (m) maxLayer = Math.max(maxLayer, parseInt(m[1], 10));
    }
    const layers = maxLayer >= 0 ? maxLayer + 1 : fb.numLayers;

    const gateT = findTensor(h, 'model.layers.0.mlp.gate_proj.weight');
    const intermediate = gateT ? gateT.shape[0] : fb.intermediateSize;

    const normT = findTensor(h, 'model.norm.weight');
    const headT = findTensor(h, 'lm_head.weight');

    let totalParams = 0;
    for (const t of h.tensors) totalParams += tensorParams(t);

    const { blocks, colNames } = buildBlocks(
        vocab, hidden, layers,
        (name, t) => {
            const ft = findTensor(h, name);
            const dims: number[] = ft
                ? ft.shape
                : (t.cols === 1 ? [t.rows] : [t.rows, t.cols]);
            return { dims, params: ft ? tensorParams(ft) : t.params, dtype: ft ? ft.dtype : 'BF16' };
        },
        (kind) => {
            if (kind === 'embed') {
                return {
                    dims: embedT ? embedT.shape : [vocab, hidden],
                    params: embedT ? tensorParams(embedT) : 136134656,
                    dtype: embedT ? embedT.dtype : 'BF16', present: true,
                };
            }
            if (kind === 'norm') {
                return {
                    dims: normT ? normT.shape : [hidden],
                    params: normT ? tensorParams(normT) : 896,
                    dtype: normT ? normT.dtype : 'BF16', present: true,
                };
            }
            return {
                dims: headT ? headT.shape : [vocab, hidden],
                params: headT ? tensorParams(headT) : 0,
                dtype: headT ? headT.dtype : 'BF16', present: headT !== undefined,
            };
        }
    );

    return {
        vocabSize: vocab, hiddenSize: hidden, numLayers: layers,
        intermediateSize: intermediate, totalParams,
        blocks, colNames, fromFile: true,
    };
}

/** 拉取 header.json 并解析；失败返回内置默认值。 */
export async function loadModelSpec(): Promise<ModelSpec> {
    try {
        const res = await fetch(`${import.meta.env.BASE_URL}header.json`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const h = (await res.json()) as HeaderJson;
        if (!Array.isArray(h.tensors) || h.tensors.length === 0) {
            throw new Error('empty tensor list');
        }
        return specFromHeader(h);
    } catch (e) {
        console.warn('loadModelSpec: header.json unavailable, using defaults:', e);
        return defaultSpec();
    }
}
