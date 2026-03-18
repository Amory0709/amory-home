import * as THREE from 'three';

interface StrokePoint {
  cx: number;
  cy: number;
}

interface Stroke {
  points: StrokePoint[];
  color: string;
  size: number;
}

const CANVAS_SIZE = 1024;

export const DRAW_BOUNDS = {
  minX: -25, maxX: 25,
  minZ: -20, maxZ: 10,
} as const;

const RANGE_X = DRAW_BOUNDS.maxX - DRAW_BOUNDS.minX;
const RANGE_Z = DRAW_BOUNDS.maxZ - DRAW_BOUNDS.minZ;

export class SandDrawing {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private texture: THREE.CanvasTexture;
  private strokes: Stroke[] = [];
  private currentStroke: Stroke | null = null;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = CANVAS_SIZE;
    this.canvas.height = CANVAS_SIZE;
    this.ctx = this.canvas.getContext('2d')!;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.flipY = false;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
  }

  private worldToCanvas(worldX: number, worldZ: number): StrokePoint | null {
    const u = (worldX - DRAW_BOUNDS.minX) / RANGE_X;
    const v = (worldZ - DRAW_BOUNDS.minZ) / RANGE_Z;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { cx: u * CANVAS_SIZE, cy: v * CANVAS_SIZE };
  }

  startStroke(worldX: number, worldZ: number, color: string, size: number) {
    const pt = this.worldToCanvas(worldX, worldZ);
    if (!pt) return;

    this.currentStroke = { points: [pt], color, size };

    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.arc(pt.cx, pt.cy, size / 2, 0, Math.PI * 2);
    this.ctx.fill();
    this.texture.needsUpdate = true;
  }

  addPoint(worldX: number, worldZ: number) {
    if (!this.currentStroke) return;
    const pt = this.worldToCanvas(worldX, worldZ);
    if (!pt) return;

    const prev = this.currentStroke.points[this.currentStroke.points.length - 1];

    this.ctx.strokeStyle = this.currentStroke.color;
    this.ctx.lineWidth = this.currentStroke.size;
    this.ctx.beginPath();
    this.ctx.moveTo(prev.cx, prev.cy);
    this.ctx.lineTo(pt.cx, pt.cy);
    this.ctx.stroke();

    this.currentStroke.points.push(pt);
    this.texture.needsUpdate = true;
  }

  endStroke() {
    if (this.currentStroke) {
      this.strokes.push(this.currentStroke);
      this.currentStroke = null;
    }
  }

  undo() {
    if (this.strokes.length === 0) return;
    this.strokes.pop();
    this.redraw();
  }

  clear() {
    this.strokes = [];
    this.currentStroke = null;
    this.ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    this.texture.needsUpdate = true;
  }

  getTexture(): THREE.CanvasTexture {
    return this.texture;
  }

  hasStrokes(): boolean {
    return this.strokes.length > 0;
  }

  private redraw() {
    this.ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    for (const stroke of this.strokes) {
      if (stroke.points.length === 0) continue;

      if (stroke.points.length === 1) {
        this.ctx.fillStyle = stroke.color;
        this.ctx.beginPath();
        this.ctx.arc(stroke.points[0].cx, stroke.points[0].cy, stroke.size / 2, 0, Math.PI * 2);
        this.ctx.fill();
      } else {
        this.ctx.strokeStyle = stroke.color;
        this.ctx.lineWidth = stroke.size;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.beginPath();
        this.ctx.moveTo(stroke.points[0].cx, stroke.points[0].cy);
        for (let i = 1; i < stroke.points.length; i++) {
          this.ctx.lineTo(stroke.points[i].cx, stroke.points[i].cy);
        }
        this.ctx.stroke();
      }
    }
    this.texture.needsUpdate = true;
  }
}
