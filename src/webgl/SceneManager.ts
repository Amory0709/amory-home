import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { oceanVertexShader, oceanFragmentShader } from './OceanShader';
import { SandDrawing } from './SandDrawing';

export class SceneManager {
    private container: HTMLDivElement;
    private renderer: THREE.WebGLRenderer;
    private camera: THREE.OrthographicCamera;
    private scene: THREE.Scene;
    private material: THREE.ShaderMaterial | null = null;
    private mesh: THREE.Mesh | null = null;
    private clock: THREE.Clock;
    private animationFrameId: number | null = null;

    private virtualCamera: THREE.PerspectiveCamera;
    private controls: OrbitControls;

    private sandDrawing: SandDrawing;
    private drawingMode = false;
    private isStroking = false;
    private brushColor = '#FFFFFF';
    private brushSize = 3;

    private boundResize: () => void;
    private boundPointerDown: (e: PointerEvent) => void;
    private boundPointerMove: (e: PointerEvent) => void;
    private boundPointerUp: (e: PointerEvent) => void;

    constructor(container: HTMLDivElement) {
        this.container = container;

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
        this.camera.position.z = 1;

        this.virtualCamera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.virtualCamera.position.set(0, 1.5, -12);

        this.controls = new OrbitControls(this.virtualCamera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 0.2, 0);

        this.clock = new THREE.Clock();
        this.sandDrawing = new SandDrawing();

        this.initShader();
        this.resize();

        this.boundResize = this.resize.bind(this);
        this.boundPointerDown = this.onPointerDown.bind(this);
        this.boundPointerMove = this.onPointerMove.bind(this);
        this.boundPointerUp = this.onPointerUp.bind(this);

        window.addEventListener('resize', this.boundResize);

        const canvas = this.renderer.domElement;
        canvas.addEventListener('pointerdown', this.boundPointerDown);
        canvas.addEventListener('pointermove', this.boundPointerMove);
        canvas.addEventListener('pointerup', this.boundPointerUp);

        this.animate();
    }

    private initShader() {
        this.material = new THREE.ShaderMaterial({
            uniforms: {
                uTime: { value: 0 },
                uResolution: { value: new THREE.Vector2() },
                uCameraPos: { value: new THREE.Vector3(0, 5, -5) },
                uCameraDir: { value: new THREE.Vector3(0, -0.5, 1).normalize() },
                uCameraRight: { value: new THREE.Vector3(1, 0, 0) },
                uCameraUp: { value: new THREE.Vector3(0, 1, 0) },
                uDrawingTexture: { value: this.sandDrawing.getTexture() },
            },
            vertexShader: oceanVertexShader,
            fragmentShader: oceanFragmentShader,
            depthWrite: false,
            depthTest: false
        });

        const geometry = new THREE.PlaneGeometry(2, 2);
        this.mesh = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.mesh);
    }

    // --- Drawing mode public API ---

    public getDomElement(): HTMLElement {
        return this.renderer.domElement;
    }

    public setDrawingMode(enabled: boolean) {
        this.drawingMode = enabled;
        this.controls.enabled = !enabled;
        this.renderer.domElement.style.cursor = enabled ? 'crosshair' : '';
        if (!enabled) {
            this.isStroking = false;
        }
    }

    public startStrokeFromLongPress(clientX: number, clientY: number) {
        const pos = this.screenToGround(clientX, clientY);
        if (pos) {
            this.sandDrawing.startStroke(pos.x, pos.z, this.brushColor, this.brushSize);
            this.isStroking = true;
        }
    }

    public setBrushSize(size: number) {
        this.brushSize = size;
    }

    public setBrushColor(color: string) {
        this.brushColor = color;
    }

    public undoDrawing() {
        this.sandDrawing.undo();
    }

    public clearDrawing() {
        this.sandDrawing.clear();
    }

    // --- Pointer handlers for drawing ---

    private onPointerDown(e: PointerEvent) {
        if (!this.drawingMode) return;
        this.renderer.domElement.setPointerCapture(e.pointerId);
        const pos = this.screenToGround(e.clientX, e.clientY);
        if (pos) {
            this.sandDrawing.startStroke(pos.x, pos.z, this.brushColor, this.brushSize);
            this.isStroking = true;
        }
    }

    private onPointerMove(e: PointerEvent) {
        if (!this.drawingMode || !this.isStroking) return;
        const pos = this.screenToGround(e.clientX, e.clientY);
        if (pos) {
            this.sandDrawing.addPoint(pos.x, pos.z);
        }
    }

    private onPointerUp(e: PointerEvent) {
        if (!this.drawingMode || !this.isStroking) return;
        this.renderer.domElement.releasePointerCapture(e.pointerId);
        this.sandDrawing.endStroke();
        this.isStroking = false;
    }

    // --- Ground ray intersection (replicates GLSL GroundHeight) ---

    private groundHeight(x: number, z: number): number {
        return z * (-0.06) + Math.sin(x * 0.1) * 0.5 + Math.sin(x * 0.3) * 0.15 - 0.2;
    }

    private screenToGround(clientX: number, clientY: number): THREE.Vector3 | null {
        const rect = this.renderer.domElement.getBoundingClientRect();
        const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.virtualCamera);

        const ray = raycaster.ray;
        const p = new THREE.Vector3();
        const step = 0.4;
        const maxDist = 80;

        for (let t = 0; t < maxDist; t += step) {
            ray.at(t, p);
            if (p.y <= this.groundHeight(p.x, p.z)) {
                let lo = Math.max(0, t - step), hi = t;
                for (let i = 0; i < 8; i++) {
                    const mid = (lo + hi) / 2;
                    ray.at(mid, p);
                    if (p.y <= this.groundHeight(p.x, p.z)) hi = mid;
                    else lo = mid;
                }
                ray.at(hi, p);
                return p;
            }
        }
        return null;
    }

    // --- Render loop ---

    private updateCameraUniforms() {
        if (!this.material) return;

        this.controls.update();

        const cameraPos = this.virtualCamera.position;
        const cf = new THREE.Vector3(0, 0, -1).applyQuaternion(this.virtualCamera.quaternion).normalize();
        const cr = new THREE.Vector3(1, 0, 0).applyQuaternion(this.virtualCamera.quaternion).normalize();
        const cu = new THREE.Vector3(0, 1, 0).applyQuaternion(this.virtualCamera.quaternion).normalize();

        this.material.uniforms.uCameraPos.value.copy(cameraPos);
        this.material.uniforms.uCameraDir.value.copy(cf);
        this.material.uniforms.uCameraRight.value.copy(cr);
        this.material.uniforms.uCameraUp.value.copy(cu);
    }

    private resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.renderer.setSize(width, height);

        if (this.material) {
            this.material.uniforms.uResolution.value.set(width, height);
        }
    }

    private animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));

        const elapsedTime = this.clock.getElapsedTime();
        if (this.material) {
            this.material.uniforms.uTime.value = elapsedTime;
            this.updateCameraUniforms();
        }

        this.renderer.render(this.scene, this.camera);
    }

    public dispose() {
        window.removeEventListener('resize', this.boundResize);

        const canvas = this.renderer.domElement;
        canvas.removeEventListener('pointerdown', this.boundPointerDown);
        canvas.removeEventListener('pointermove', this.boundPointerMove);
        canvas.removeEventListener('pointerup', this.boundPointerUp);

        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
        }
        this.renderer.dispose();
        if (this.container.contains(this.renderer.domElement)) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}
