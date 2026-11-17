import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { ModelSpec, SelInfo } from './modelMap';

/* ------------------------------------------------------------------ */
/* 全模型结构示意：embed + N 层 decoder + norm + lm_head，             */
/* 结构描述由 modelMap.loadModelSpec() 从 header.json 解析得到，        */
/* 全部用统一厚度的 Box 竖立起来、横向一字排开（像书架），              */
/* 仅示意结构，不按参数量比例建模。                                     */
/* ------------------------------------------------------------------ */

/* 列内堆叠间距；每块尺寸来自 spec（modelMap 按 shape 换算） */
const ROW_GAP: number = 1.0;

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

/* 全模型 tensor 级视图：列=embed|L0..|norm|head，列内按数据流堆叠。由 React 组件持有，卸载时调用 dispose()。 */
export class EmbeddingScene {
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene = new THREE.Scene();
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;
    private grid: THREE.GridHelper | null = null;
    private blocks = [] as import('./modelMap').BlockSpec[];
    private meshes: THREE.Mesh[] = [];
    private wrappers: THREE.Mesh[] = [];
    private wrapperEdges: THREE.LineSegments[] = [];
    private colHeaders: THREE.Sprite[] = [];
    private labelSprite: THREE.Sprite | null = null;
    private selMesh: number = -1; // 选中的 tensor 块下标
    private selWrap: number = -1; // 选中的层外包盒（col，即 L+1）
    private focusAnim: {
        t: number; dur: number;
        fromPos: THREE.Vector3; toPos: THREE.Vector3;
        fromTgt: THREE.Vector3; toTgt: THREE.Vector3;
    } | null = null;
    private scanning: boolean = true;
    private downX: number = 0;
    private downY: number = 0;
    private clock: THREE.Clock = new THREE.Clock();
    private rafId: number = 0;
    private hiTimer: number = 0;
    private hiIndex: number = -1;
    private disposed: boolean = false;
    private onToken: (info: SelInfo | null) => void;
    private handleResize: () => void;

    private static HI: THREE.Color = new THREE.Color(0x4ade80);

    constructor(canvas: HTMLCanvasElement, spec: ModelSpec, onToken: (info: SelInfo | null) => void) {
        this.onToken = onToken;

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(window.innerWidth, window.innerHeight);

        this.scene.background = new THREE.Color(0x0b1020);
        this.scene.fog = new THREE.Fog(0x0b1020, 350, 900);

        this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1200);

        this.scene.add(new THREE.HemisphereLight(0xbcd3ff, 0x1a2338, 0.9));
        const dirLight: THREE.DirectionalLight = new THREE.DirectionalLight(0xffffff, 1.6);
        dirLight.position.set(60, 120, 40);
        this.scene.add(dirLight);

        this.grid = new THREE.GridHelper(220, 44, 0x2b3b63, 0x16223c);
        this.grid.position.y = -0.05;
        this.scene.add(this.grid);

        /* ---- 单行串联（整行居中到原点）+ 每层外包透明 Box ---- */
        this.blocks = spec.blocks;
        const blocks = this.blocks;
        const n: number = blocks.length;

        /* 先算总行长，起点 x0 让整行以原点为中心 */
        let rowLen: number = -ROW_GAP;
        for (let i = 0; i < n; i++) rowLen += blocks[i].w + ROW_GAP;
        const x0: number = -rowLen / 2;

        let maxY: number = 0; // 最大高度（Y 取 d）
        let maxZ: number = 0; // 最大深度（Z 取 h）
        for (let i = 0; i < n; i++) {
            maxY = Math.max(maxY, blocks[i].d);
            maxZ = Math.max(maxZ, blocks[i].h);
        }
        const CY: number = maxY / 2; // 中心线高度

        /* 记录每层块的 x 范围与 Y/Z 包络，用于外包 Box */
        const layerMinX: Map<number, number> = new Map();
        const layerMaxX: Map<number, number> = new Map();
        const layerMaxY: Map<number, number> = new Map();
        const layerMaxZ: Map<number, number> = new Map();

        let cursorX: number = x0;
        for (let i = 0; i < n; i++) {
            const b = blocks[i];
            const geo: THREE.BoxGeometry = new THREE.BoxGeometry(b.w, b.d, b.h);
            const mat: THREE.MeshStandardMaterial = new THREE.MeshStandardMaterial({
                color: b.color,
                roughness: 0.45,
                metalness: 0.2,
            });
            const m: THREE.Mesh = new THREE.Mesh(geo, mat);
            const x: number = cursorX + b.w / 2;
            m.position.set(x, CY, 0);
            this.scene.add(m);
            const edges: THREE.LineSegments = new THREE.LineSegments(
                new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x8fb4ff }));
            edges.position.copy(m.position);
            this.scene.add(edges);
            this.meshes.push(m);
            cursorX += b.w + ROW_GAP;

            /* col 1..numLayers 为层内 tensor，embed/norm/head 不包 */
            if (b.col >= 1 && b.col <= spec.numLayers) {
                const L: number = b.col;
                layerMinX.set(L, Math.min(layerMinX.get(L) ?? Infinity, x - b.w / 2));
                layerMaxX.set(L, Math.max(layerMaxX.get(L) ?? -Infinity, x + b.w / 2));
                layerMaxY.set(L, Math.max(layerMaxY.get(L) ?? 0, b.d));
                layerMaxZ.set(L, Math.max(layerMaxZ.get(L) ?? 0, b.h));
            }
        }

        /* 每层外包 Box：透明 + 线框 + 层号常驻 */
        const PAD: number = 0.8;
        const WRAP_OPACITY: number = 0.32;
        for (let L = 1; L <= spec.numLayers; L++) {
            const minX: number | undefined = layerMinX.get(L);
            const maxX: number | undefined = layerMaxX.get(L);
            if (minX === undefined || maxX === undefined) continue;
            const wW: number = (maxX - minX) + PAD * 2;
            const hW: number = (layerMaxY.get(L) ?? 0) + PAD * 2;
            const dW: number = (layerMaxZ.get(L) ?? 0) + PAD * 2;
            const wgeo: THREE.BoxGeometry = new THREE.BoxGeometry(wW, hW, dW);
            const wmat: THREE.MeshStandardMaterial = new THREE.MeshStandardMaterial({
                color: 0x94a3b8,
                transparent: true,
                opacity: WRAP_OPACITY,
                roughness: 0.9,
                metalness: 0,
                depthWrite: false,
            });
            const wrap: THREE.Mesh = new THREE.Mesh(wgeo, wmat);
            wrap.position.set((minX + maxX) / 2, CY, 0);
            wrap.userData.isWrapper = true;
            wrap.userData.layer = L;
            wrap.userData.open = false;
            this.scene.add(wrap);
            this.wrappers.push(wrap);
            const wedge: THREE.LineSegments = new THREE.LineSegments(
                new THREE.EdgesGeometry(wgeo), new THREE.LineBasicMaterial({ color: 0xcbd5e1 }));
            wedge.position.copy(wrap.position);
            this.scene.add(wedge);
            this.wrapperEdges.push(wedge);

            const tag: THREE.Sprite = makeLabel(`L${L - 1}`, 0.5);
            tag.position.set((minX + maxX) / 2, CY + hW / 2 + 3, 0);
            this.scene.add(tag);
            this.colHeaders.push(tag);
        }

        /* 中心线：穿过所有块的中心点（前侧再加一条地面参考线） */
        this.addHAxis(x0 - 2, -x0 + 2, CY, 0, 0xfb7185);
        this.addHAxis(x0, -x0, 0.1, maxZ / 2 + 2, 0x38bdf8);

        const title: THREE.Sprite = makeLabel(
            `Qwen2.5-0.5B  [${spec.vocabSize}, ${spec.hiddenSize}] x${spec.numLayers} layers (${n} tensors in a row)`, 1.2);
        title.position.set(0, CY + maxY / 2 + 14, 0);
        this.scene.add(title);

        const note: THREE.Sprite = makeLabel(
            `one box per tensor, strung along X, centers aligned (H/W/D from tensor shapes)`, 0.6);
        note.position.set(0, CY + maxY / 2 + 9.5, 0);
        this.scene.add(note);

        /* 相机/网格/雾按行长自适应（行很长：291 块） */
        const fit: number = Math.max(rowLen, maxY * 2);
        const gridSize: number = Math.ceil((rowLen * 1.25) / 20) * 20;
        if (this.grid) {
            this.scene.remove(this.grid);
            this.grid.geometry.dispose();
            (this.grid.material as THREE.Material).dispose();
        }
        this.grid = new THREE.GridHelper(gridSize, gridSize / 20, 0x2b3b63, 0x16223c);
        this.grid.position.y = -0.05;
        this.scene.add(this.grid);
        this.scene.fog = new THREE.Fog(0x0b1020, fit * 1.05, fit * 2.2);

        /* ---- 视角控制：左键旋转 / 右键平移 / 滚轮缩放 ---- */
        this.camera.position.set(fit * 0.4, CY + maxY * 0.75 + 18, fit * 0.85 + 30);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, CY, 0);
        this.controls.enableDamping = false;
        this.controls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN
        };
        this.controls.minDistance = 10;
        this.controls.maxDistance = fit * 2;
        this.controls.maxPolarAngle = Math.PI * 0.495;
        this.controls.update();

        /* 点击选中：按下/抬起位移 < 6px 才算点击（避免把拖拽旋转误判） */
        this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);
        this.renderer.domElement.addEventListener('pointerup', this.onPointerUp);
        window.addEventListener('keydown', this.onKeyDown);

        this.handleResize = (): void => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        };
        window.addEventListener('resize', this.handleResize);

        this.animate();
    }

    /** 按 F 聚焦到当前选中（tensor 块或层外包盒）：动画飞过去 */
    private onKeyDown = (e: KeyboardEvent): void => {
        if (e.key !== 'f' && e.key !== 'F') return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        this.focusSelected();
    };

    private focusSelected(): void {
        let obj: THREE.Mesh | null = null;
        if (this.selMesh >= 0) obj = this.meshes[this.selMesh];
        else if (this.selWrap >= 0) obj = this.wrapOf(this.selWrap);
        if (!obj) return;
        const sphere: THREE.Sphere = new THREE.Box3().setFromObject(obj).getBoundingSphere(new THREE.Sphere());
        const dist: number = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.3;
        const dir: THREE.Vector3 = this.camera.position.clone().sub(this.controls.target);
        if (dir.lengthSq() < 1e-6) dir.set(1, 0.6, 1);
        dir.normalize();
        this.focusAnim = {
            t: 0,
            dur: 0.7,
            fromPos: this.camera.position.clone(),
            toPos: sphere.center.clone().addScaledVector(dir, dist),
            fromTgt: this.controls.target.clone(),
            toTgt: sphere.center.clone(),
        };
    }

    private static easeInOut(k: number): number {
        return k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    }

    private wrapOf(L: number): THREE.Mesh | null {
        for (const w of this.wrappers) {
            if (w.userData.layer === L) return w;
        }
        return null;
    }

    private onPointerDown = (e: PointerEvent): void => {
        this.downX = e.clientX;
        this.downY = e.clientY;
        this.focusAnim = null; // 用户一接管就停掉飞行画
    };

    private onPointerUp = (e: PointerEvent): void => {
        const dx: number = e.clientX - this.downX;
        const dy: number = e.clientY - this.downY;
        if (dx * dx + dy * dy > 36) return; // 拖拽，不算点击
        this.pick(e.clientX, e.clientY);
    };

    /** Raycast 拾取：未开的外包盒优先命中→打开（变透+让出点击）；已开则点到内部 tensor */
    private pick(clientX: number, clientY: number): void {
        const rect: DOMRect = this.renderer.domElement.getBoundingClientRect();
        const ndc: THREE.Vector2 = new THREE.Vector2(
            ((clientX - rect.left) / rect.width) * 2 - 1,
            -((clientY - rect.top) / rect.height) * 2 + 1
        );
        const ray: THREE.Raycaster = new THREE.Raycaster();
        ray.setFromCamera(ndc, this.camera);
        const closedWraps: THREE.Mesh[] = this.wrappers.filter((w) => !w.userData.open);
        const hits: THREE.Intersection<THREE.Object3D>[] =
            ray.intersectObjects([...closedWraps, ...this.meshes], false);
        if (hits.length === 0) {
            this.clearSelection();
            return;
        }
        const obj: THREE.Object3D = hits[0].object;
        if ((obj as THREE.Mesh).userData.isWrapper) {
            const L: number = (obj as THREE.Mesh).userData.layer as number;
            this.openWrapper(obj as THREE.Mesh);
            this.selectWrap(L);
            return;
        }
        const idx: number = this.meshes.indexOf(obj as THREE.Mesh);
        if (idx >= 0) this.selectMesh(idx);
    }

    /** 打开层外包盒：透明度降到近乎不可见，并从拾取目标里移除，露出内部 tensor */
    private openWrapper(wrap: THREE.Mesh): void {
        wrap.userData.open = true;
        const mat: THREE.MeshStandardMaterial = wrap.material as THREE.MeshStandardMaterial;
        mat.opacity = 0.015;
        /* 保持扫描/选中状态不动，只让路 */
    }

    private selectMesh(i: number): void {
        if (this.hiIndex >= 0) this.setHighlight(this.hiIndex, false);
        this.hiIndex = -1;
        this.clearWrapSel();
        if (this.selMesh >= 0) this.setHighlight(this.selMesh, false);
        this.selMesh = i;
        this.setHighlight(i, true);
        this.showMeshLabel(i);
        this.scanning = false;
        const b = this.blocks[i];
        this.onToken({ title: b.name, shape: b.shape, dtype: b.dtype });
    }

    /** 选中层外包盒：高亮外框 + 显示该层汇总标签 */
    private selectWrap(L: number): void {
        if (this.hiIndex >= 0) this.setHighlight(this.hiIndex, false);
        this.hiIndex = -1;
        this.clearMeshSel();
        if (this.selWrap >= 0) this.setWrapHighlight(this.selWrap, false);
        this.selWrap = L;
        this.setWrapHighlight(L, true);
        this.showWrapLabel(L);
        this.scanning = false;
        this.onToken(this.wrapSummary(L));
    }

    private clearMeshSel(): void {
        if (this.selMesh < 0) return;
        this.setHighlight(this.selMesh, false);
        this.selMesh = -1;
    }

    private clearWrapSel(): void {
        if (this.selWrap < 0) return;
        this.setWrapHighlight(this.selWrap, false);
        this.selWrap = -1;
    }

    private clearSelection(): void {
        if (this.selMesh < 0 && this.selWrap < 0) return;
        this.clearMeshSel();
        this.clearWrapSel();
        this.hideLabel();
        this.onToken(null);
        this.scanning = true;
        this.hiTimer = 0;
    }

    private wrapSummary(L: number): SelInfo {
        let n = 0, p = 0;
        const dts: string[] = [];
        for (const b of this.blocks) {
            if (b.col === L) {
                n++; p += b.params;
                if (!dts.includes(b.dtype)) dts.push(b.dtype);
            }
        }
        return {
            title: `L${L - 1}`,
            shape: `${n} tensors · ${(p / 1e6).toFixed(1)}M params`,
            dtype: dts.join(' / ') || '—',
        };
    }

    /** 选中才建标签 sprite（290 个常驻纹理太占显存），切选/取消时销毁 */
    private showMeshLabel(i: number): void {
        this.hideLabel();
        const mesh: THREE.Mesh = this.meshes[i];
        const h: number = (mesh.geometry as THREE.BoxGeometry).parameters.height;
        const b = this.blocks[i];
        const tag: THREE.Sprite = makeLabel(`${b.name}  ${b.shape}  ${b.dtype}`, 0.55);
        tag.position.set(mesh.position.x, mesh.position.y + h / 2 + 3, mesh.position.z);
        this.scene.add(tag);
        this.labelSprite = tag;
    }

    private showWrapLabel(L: number): void {
        this.hideLabel();
        const wrap: THREE.Mesh | null = this.wrapOf(L);
        if (!wrap) return;
        const h: number = (wrap.geometry as THREE.BoxGeometry).parameters.height;
        const s: SelInfo = this.wrapSummary(L);
        const tag: THREE.Sprite = makeLabel(`${s.title}  ${s.shape}  ${s.dtype}`, 0.7);
        tag.position.set(wrap.position.x, wrap.position.y + h / 2 + 3, wrap.position.z);
        this.scene.add(tag);
        this.labelSprite = tag;
    }

    private hideLabel(): void {
        if (!this.labelSprite) return;
        this.scene.remove(this.labelSprite);
        const mat = this.labelSprite.material as THREE.SpriteMaterial;
        mat.map?.dispose();
        mat.dispose();
        this.labelSprite = null;
    }

    private addHAxis(x1: number, x2: number, y: number, z: number, color: number): void {
        const g: THREE.BufferGeometry = new THREE.BufferGeometry().setFromPoints(
            [new THREE.Vector3(x1, y, z), new THREE.Vector3(x2, y, z)]);
        this.scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color })));
    }

    private setHighlight(i: number, on: boolean): void {
        const mat: THREE.MeshStandardMaterial =
            this.meshes[i].material as THREE.MeshStandardMaterial;
        mat.emissive.copy(on ? EmbeddingScene.HI : new THREE.Color(0x000000));
        mat.emissiveIntensity = on ? 0.45 : 0;
    }

    private setWrapHighlight(L: number, on: boolean): void {
        const wrap: THREE.Mesh | null = this.wrapOf(L);
        if (!wrap) return;
        const mat: THREE.MeshStandardMaterial = wrap.material as THREE.MeshStandardMaterial;
        mat.emissive.copy(on ? EmbeddingScene.HI : new THREE.Color(0x000000));
        mat.emissiveIntensity = on ? 0.5 : 0;
    }

    private animate = (): void => {
        if (this.disposed) return;
        this.rafId = requestAnimationFrame(this.animate);
        const dt: number = this.clock.getDelta();

        this.hiTimer += dt;
        if (this.scanning && this.hiTimer > 1.6) {
            this.hiTimer = 0;
            if (this.hiIndex >= 0) this.setHighlight(this.hiIndex, false);
            this.hiIndex = (this.hiIndex + 1) % this.meshes.length;
            this.setHighlight(this.hiIndex, true);
        }

        /* 聚焦飞行画：easeInOut 插值相机位姿，用户接管即停 */
        if (this.focusAnim) {
            const a = this.focusAnim;
            a.t += dt;
            const k: number = EmbeddingScene.easeInOut(Math.min(1, a.t / a.dur));
            this.camera.position.lerpVectors(a.fromPos, a.toPos, k);
            this.controls.target.lerpVectors(a.fromTgt, a.toTgt, k);
            if (a.t >= a.dur) this.focusAnim = null;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    };

    dispose(): void {
        this.disposed = true;
        cancelAnimationFrame(this.rafId);
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('keydown', this.onKeyDown);
        this.renderer.domElement.removeEventListener('pointerdown', this.onPointerDown);
        this.renderer.domElement.removeEventListener('pointerup', this.onPointerUp);
        this.controls.dispose();
        this.hideLabel();
        for (const m of this.meshes) {
            m.geometry.dispose();
            (m.material as THREE.Material).dispose();
        }
        for (const w of this.wrappers) {
            w.geometry.dispose();
            (w.material as THREE.Material).dispose();
        }
        for (const e of this.wrapperEdges) {
            e.geometry.dispose();
            (e.material as THREE.Material).dispose();
        }
        for (const s of this.colHeaders) {
            const mat = s.material as THREE.SpriteMaterial;
            mat.map?.dispose();
            mat.dispose();
        }
        this.renderer.dispose();
    }
}
