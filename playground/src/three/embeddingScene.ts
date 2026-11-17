import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/* ------------------------------------------------------------------ */
/* 嵌入层规格（与 .cache/Qwen2.5-0.5B/config.json 一致）                */
/* 真实矩阵 151936 x 896 ≈ 1.36 亿参数，用单个 Box 做示意，             */
/* 长宽标注真实维度，不按比例建模。                                     */
/* ------------------------------------------------------------------ */
export const VOCAB_SIZE: number = 151936;
export const HIDDEN_SIZE: number = 896;

/* 示意 Box 的显示尺寸（与真实维度不成比例，仅保证可视） */
const BOX_W: number = 20; // X：hidden 方向
const BOX_H: number = 5;  // Y：厚度（仅示意）
const BOX_D: number = 60; // Z：vocab 方向

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

/* 嵌入矩阵的单个 Box 示意视图。由 React 组件持有，卸载时调用 dispose()。 */
export class EmbeddingScene {
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene = new THREE.Scene();
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;
    private box: THREE.Mesh;
    private cursor: THREE.Mesh;
    private clock: THREE.Clock = new THREE.Clock();
    private rafId: number = 0;
    private hiTimer: number = 0;
    private cursorT: number = 0;
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

        /* ---- 单个嵌入层 Box ---- */
        const geo: THREE.BoxGeometry = new THREE.BoxGeometry(BOX_W, BOX_H, BOX_D);
        const mat: THREE.MeshStandardMaterial = new THREE.MeshStandardMaterial({
            color: 0x2a4a7f,
            roughness: 0.4,
            metalness: 0.2,
            transparent: true,
            opacity: 0.92,
        });
        this.box = new THREE.Mesh(geo, mat);
        this.box.position.y = BOX_H / 2;
        this.scene.add(this.box);

        const edges: THREE.LineSegments = new THREE.LineSegments(
            new THREE.EdgesGeometry(geo),
            new THREE.LineBasicMaterial({ color: 0x38bdf8 })
        );
        edges.position.copy(this.box.position);
        this.scene.add(edges);

        /* 查表游标：沿 vocab(Z) 方向移动的薄片，示意 token 查表 */
        const cursorGeo: THREE.BoxGeometry = new THREE.BoxGeometry(BOX_W + 0.6, BOX_H + 0.6, 1.2);
        const cursorMat: THREE.MeshBasicMaterial = new THREE.MeshBasicMaterial({
            color: 0x4ade80,
            transparent: true,
            opacity: 0.55,
        });
        this.cursor = new THREE.Mesh(cursorGeo, cursorMat);
        this.cursor.position.y = BOX_H / 2;
        this.scene.add(this.cursor);

        /* ---- 坐标轴与宽高标注 ---- */
        const HX: number = BOX_W / 2, HZ: number = BOX_D / 2;
        this.addAxis(-HX, HZ + 1.5, HX, HZ + 1.5, 0xf59e0b);   // 宽：hidden
        this.addAxis(HX + 1.5, -HZ, HX + 1.5, HZ, 0x38bdf8);   // 长：vocab

        const title: THREE.Sprite = makeLabel(`embed_tokens  [${VOCAB_SIZE}, ${HIDDEN_SIZE}]`, 1.4);
        title.position.set(0, 16, 0);
        this.scene.add(title);

        const wLabel: THREE.Sprite = makeLabel(`宽 hidden H = ${HIDDEN_SIZE}`, 0.8);
        wLabel.position.set(0, 0.6, HZ + 4.5);
        this.scene.add(wLabel);

        const dLabel: THREE.Sprite = makeLabel(`长 vocab V = ${VOCAB_SIZE}`, 0.8);
        dLabel.position.set(HX + 8.5, 0.6, 0);
        dLabel.center.set(0, 0.5);
        this.scene.add(dLabel);

        const hLabel: THREE.Sprite = makeLabel(`厚 BF16`, 0.7);
        hLabel.position.set(-HX - 6, BOX_H / 2, -HZ);
        this.scene.add(hLabel);

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

    private animate = (): void => {
        if (this.disposed) return;
        this.rafId = requestAnimationFrame(this.animate);
        const dt: number = this.clock.getDelta();

        /* 游标匀速扫过 vocab 方向，每 1.6s 上报一次 token id */
        this.cursorT = (this.cursorT + dt * 0.12) % 1;
        const z: number = -BOX_D / 2 + this.cursorT * BOX_D;
        this.cursor.position.z = z;

        this.hiTimer += dt;
        if (this.hiTimer > 1.6) {
            this.hiTimer = 0;
            this.onToken(Math.floor(this.cursorT * VOCAB_SIZE));
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    };

    dispose(): void {
        this.disposed = true;
        cancelAnimationFrame(this.rafId);
        window.removeEventListener('resize', this.handleResize);
        this.controls.dispose();
        this.box.geometry.dispose();
        (this.box.material as THREE.Material).dispose();
        this.cursor.geometry.dispose();
        (this.cursor.material as THREE.Material).dispose();
        this.renderer.dispose();
    }
}
