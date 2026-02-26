import * as THREE from 'three';
import { oceanVertexShader, oceanFragmentShader } from './OceanShader';

export class SceneManager {
    private container: HTMLDivElement;
    private renderer: THREE.WebGLRenderer;
    private camera: THREE.OrthographicCamera;
    private scene: THREE.Scene;
    private material: THREE.ShaderMaterial | null = null;
    private mesh: THREE.Mesh | null = null;
    private clock: THREE.Clock;
    private animationFrameId: number | null = null;

    constructor(container: HTMLDivElement) {
        this.container = container;

        // 1. Setup Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        // 2. Setup Scene & Orthographic Camera (for full screen quad)
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
        this.camera.position.z = 1;

        // 3. Clock for time
        this.clock = new THREE.Clock();

        // Initialize
        this.initShader();
        this.resize();

        // Event Listeners
        window.addEventListener('resize', this.resize.bind(this));

        // Start Loop
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

    private updateCameraUniforms() {
        if (!this.material) return;

        // In a real app, these might be driven by user input or animations
        // For now, let's look roughly down at the beach
        const cameraPos = new THREE.Vector3(10, 4, 10);
        const lookAt = new THREE.Vector3(0, 0, 0);

        const cf = new THREE.Vector3().subVectors(lookAt, cameraPos).normalize();
        const cr = new THREE.Vector3().crossVectors(cf, new THREE.Vector3(0, 1, 0)).normalize();
        const cu = new THREE.Vector3().crossVectors(cr, cf).normalize();

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
        window.removeEventListener('resize', this.resize.bind(this));
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
        }
        this.renderer.dispose();
        if (this.container.contains(this.renderer.domElement)) {
            this.container.removeChild(this.renderer.domElement);
        }
        // TODO: dispose geometries/materials properly
    }
}
