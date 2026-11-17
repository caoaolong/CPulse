import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/* ------------------------------------------------------------------ */
/* 嵌入层规格（与 .cache/Qwen2.5-0.5B/config.json 一致）                */
/* 真实矩阵 151936 x 896 ≈ 1.36 亿参数，全部渲染不现实，               */
/* 这里均匀采样 64 x 64 个单元做示意，HUD 与坐标轴标注真实维度。        */
/* ------------------------------------------------------------------ */
export const VOCAB_SIZE: number = 151936;
export const HIDDEN_SIZE: number = 896;
const SHOW_V: number = 64;   // 采样显示的 token 行数（Z 轴）
const SHOW_H: number = 64;   // 采样显示的 hidden 维度数（X 轴）
const MAX_BAR_H: number = 7; // 柱子最大高度

/* 确定性伪随机，保证每次打开看到同一组"权重" */
function mulberry32(seed: number): () => number {
    return function (): number {
        seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
        const t: number = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        const u: number = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((u ^ (u >>> 14)) >>> 0) / 4294967296;
    };
}

function makeLabel(text: string, scale: number = 1): THREE.Sprite {
    const pad: number = 24;
    const cv: HTMLCanvasElement = document.createElement('canvas');
    const mctx: CanvasRenderingContext2D | null = cv.getContext('2d');
    if (!mctx) throw new Error('2d context unavailable');
    mctx.font = '44px Consolas, monospace';
    const w: number = Math.ceil(mctx.measureText(text).width) + pad * 2;
    cv.width = w; cv.height = 84;
    const ctx: CanvasRenderingContext2D | null = cv.getContext('2d');
    if (!ctx) throw new Error('2d context unavailable');
    ctx.font = '44px Consolas, monospace';
    ctx.fillStyle = '#dbe7ff';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, pad, 44);
    const tex: THREE.CanvasTexture = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sp: THREE.Sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
    sp.scale.set((w / 84) * 3 * scale, 3 * scale, 1);
    sp.renderOrder = 10;
    return sp;
}

/* 嵌入矩阵的 3D 柱状视图。由 React 组件持有，卸载时调用 dispose()。 */
export class EmbeddingScene {
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene = new THREE.Scene();
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;
    private mesh: THREE.InstancedMesh;
    private baseColors: Float32Array;
    private clock: THREE.Clock = new THREE.Clock();
    private tmpColor: THREE.Color = new THREE.Color();
    private rafId: number = 0;
    private hiRow: number = -1;
    private hiTimer: number = 0;
    private disposed: boolean = false;
    private onToken: (id: number) => void;
    private handleResize: () => void;

    constructor(canvas: HTMLCanvasElement, onToken: (id: number) => void) {
        this.onToken = onToken;

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(window.innerWidth, window.innerHeight);

        this.scene.background = new THREE.Color(0x0b1020);
        this.scene.fog = new THREE.Fog(0x0b1020, 140, 320);

        this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(52, 46, 52);

        this.scene.add(new THREE.HemisphereLight(0xbcd3ff, 0x1a2338, 0.9));
        const dirLight: THREE.DirectionalLight = new THREE.DirectionalLight(0xffffff, 1.6);
        dirLight.position.set(40, 70, 20);
        this.scene.add(dirLight);

        const grid: THREE.GridHelper = new THREE.GridHelper(150, 30, 0x2b3b63, 0x16223c);
        grid.position.y = -0.05;
        this.scene.add(grid);

        /* ---- 嵌入矩阵柱体 ---- */
        const rng: () => number = mulberry32(2505);
        const values: Float32Array = new Float32Array(SHOW_V * SHOW_H);
        for (let i = 0; i < values.length; i++) {
            values[i] = (rng() + rng() + rng() - 1.5) / 1.5;
        }

        const geo: THREE.BoxGeometry = new THREE.BoxGeometry(0.8, 1, 0.8);
        geo.translate(0, 0.5, 0); // 以底部为原点，方便按高度缩放
        const mat: THREE.MeshStandardMaterial = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15 });
        this.mesh = new THREE.InstancedMesh(geo, mat, SHOW_V * SHOW_H);

        const cPos: THREE.Color = new THREE.Color(0xf59e0b);
        const cNeg: THREE.Color = new THREE.Color(0x38bdf8);
        const cBase: THREE.Color = new THREE.Color(0x1e293b);
        this.baseColors = new Float32Array(SHOW_V * SHOW_H * 3);
        const m: THREE.Matrix4 = new THREE.Matrix4();
        let n: number = 0;
        for (let r = 0; r < SHOW_V; r++) {
            for (let c = 0; c < SHOW_H; c++, n++) {
                const v: number = values[n];
                const h: number = 0.25 + Math.min(1, Math.abs(v)) * MAX_BAR_H;
                m.makeScale(1, h, 1);
                m.setPosition((c - SHOW_H / 2 + 0.5), 0, (r - SHOW_V / 2 + 0.5));
                this.mesh.setMatrixAt(n, m);
                const k: number = Math.min(1, Math.abs(v) * 1.2);
                this.tmpColor.copy(cBase).lerp(v >= 0 ? cPos : cNeg, 0.25 + 0.75 * k);
                this.mesh.setColorAt(n, this.tmpColor);
                this.baseColors[n * 3] = this.tmpColor.r;
                this.baseColors[n * 3 + 1] = this.tmpColor.g;
                this.baseColors[n * 3 + 2] = this.tmpColor.b;
            }
        }
        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
        this.scene.add(this.mesh);

        /* ---- 坐标轴与标注 ---- */
        const HX: number = SHOW_H / 2, HZ: number = SHOW_V / 2;
        this.addAxis(-HX - 1, HZ + 1.5, HX + 6, HZ + 1.5, 0xf59e0b);
        this.addAxis(HX + 1.5, -HZ - 1, HX + 1.5, HZ + 6, 0x38bdf8);

        const title: THREE.Sprite = makeLabel(`embed_tokens  [${VOCAB_SIZE}, ${HIDDEN_SIZE}]`, 1.4);
        title.position.set(0, 16, 0);
        this.scene.add(title);

        const xLabel: THREE.Sprite = makeLabel(`hidden dim  H = ${HIDDEN_SIZE} (showing ${SHOW_H})`, 0.8);
        xLabel.position.set(0, 0.6, HZ + 4.5);
        this.scene.add(xLabel);

        const zLabel: THREE.Sprite = makeLabel(`vocab  V = ${VOCAB_SIZE} (showing ${SHOW_V})`, 0.8);
        zLabel.position.set(HX + 4.5, 0.6, 0);
        zLabel.center.set(0, 0.5);
        this.scene.add(zLabel);

        /* ---- 视角控制：左键旋转 / 右键平移 / 滚轮缩放 ---- */
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 2, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN
        };
        this.controls.minDistance = 10;
        this.controls.maxDistance = 400;
        this.controls.maxPolarAngle = Math.PI * 0.495;
        this.controls.update();

        this.handleResize = (): void => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        };
        window.addEventListener('resize', this.handleResize);

        this.animate();
    }

    private addAxis(x1: number, z1: number, x2: number, z2: number, color: number): void {
        const g: THREE.BufferGeometry = new THREE.BufferGeometry().setFromPoints(
            [new THREE.Vector3(x1, 0.1, z1), new THREE.Vector3(x2, 0.1, z2)]);
        this.scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color })));
    }

    private setRowColor(row: number, color: THREE.Color | null): void {
        if (!this.mesh.instanceColor) return;
        for (let c = 0; c < SHOW_H; c++) {
            const i: number = row * SHOW_H + c;
            if (color) {
                this.mesh.setColorAt(i, color);
            } else {
                this.tmpColor.setRGB(this.baseColors[i * 3], this.baseColors[i * 3 + 1], this.baseColors[i * 3 + 2]);
                this.mesh.setColorAt(i, this.tmpColor);
            }
        }
        this.mesh.instanceColor.needsUpdate = true;
    }

    private animate = (): void => {
        if (this.disposed) return;
        this.rafId = requestAnimationFrame(this.animate);
        const dt: number = this.clock.getDelta();

        this.hiTimer += dt;
        if (this.hiTimer > 1.6) {
            this.hiTimer = 0;
            if (this.hiRow >= 0) this.setRowColor(this.hiRow, null);
            this.hiRow = (this.hiRow + 7) % SHOW_V;
            this.setRowColor(this.hiRow, EmbeddingScene.HI);
            this.onToken(Math.floor((this.hiRow + 0.5) * VOCAB_SIZE / SHOW_V));
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    };

    private static HI: THREE.Color = new THREE.Color(0x4ade80);

    dispose(): void {
        this.disposed = true;
        cancelAnimationFrame(this.rafId);
        window.removeEventListener('resize', this.handleResize);
        this.controls.dispose();
        this.mesh.geometry.dispose();
        (this.mesh.material as THREE.Material).dispose();
        this.renderer.dispose();
    }
}
