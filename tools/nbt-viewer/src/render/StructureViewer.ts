import { mat4, vec2, vec3 } from 'gl-matrix';
import { BlockState, Structure } from 'deepslate/core';
import { StructureRenderer } from 'deepslate/render';
import type { StructureBlock, StructureModel } from '../types';
import { MinecraftResources } from './mcmeta';

type SelectionHandler = (block: StructureBlock | undefined) => void;
type DragMode = 'move' | 'rotate';

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function samePos(a: [number, number, number], b: vec3): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function createDeepslateStructure(model: StructureModel): Structure {
  return new Structure(
    model.size,
    model.palette.map(entry => new BlockState(entry.name, entry.properties)),
    model.blocks.map(block => ({
      pos: block.pos,
      state: block.state,
    })),
  );
}

export class StructureViewer {
  private readonly gl: WebGLRenderingContext;
  private readonly resourcesPromise = MinecraftResources.load();
  private renderer?: StructureRenderer;
  private structure = Structure.EMPTY;
  private model?: StructureModel;
  private selectedBlock?: vec3;
  private onSelect: SelectionHandler = () => undefined;
  private frame = 0;
  private dragMode?: DragMode;
  private dragPos?: vec2;
  private lastFrameTime = 0;
  private readonly movement = new Set<string>();
  private cPos = vec3.create();
  private cRot = vec2.fromValues(0.4, 0.6);
  private cDist = 10;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error('WebGL is unavailable.');
    this.gl = gl;

    canvas.addEventListener('contextmenu', event => event.preventDefault());
    canvas.addEventListener('pointerdown', event => this.onPointerDown(event));
    canvas.addEventListener('pointermove', event => this.onPointerMove(event));
    canvas.addEventListener('pointerup', event => this.onPointerUp(event));
    canvas.addEventListener('pointerleave', () => this.endDrag());
    canvas.addEventListener('wheel', event => this.onWheel(event), { passive: false });
    window.addEventListener('keydown', event => this.onKeyDown(event));
    window.addEventListener('keyup', event => this.onKeyUp(event));
    window.addEventListener('resize', () => this.render());

    void this.initialize();
  }

  setSelectionHandler(handler: SelectionHandler): void {
    this.onSelect = handler;
  }

  async setStructure(model: StructureModel): Promise<void> {
    this.model = model;
    this.structure = createDeepslateStructure(model);

    if (!this.renderer) {
      const resources = await this.resourcesPromise;
      this.renderer = new StructureRenderer(this.gl, this.structure, resources);
    } else {
      this.renderer.setStructure(this.structure);
    }

    vec3.copy(this.cPos, this.structure.getSize());
    vec3.scale(this.cPos, this.cPos, -0.5);
    this.cDist = Math.max(5, vec3.distance([0, 0, 0], this.cPos) * 1.7);
    this.selectedBlock = undefined;
    this.render();
  }

  private async initialize(): Promise<void> {
    const resources = await this.resourcesPromise;
    this.renderer = new StructureRenderer(this.gl, this.structure, resources);
    this.render();
  }

  private onPointerDown(event: PointerEvent): void {
    if (event.button !== 0 && event.button !== 2) return;

    event.preventDefault();
    this.canvas.setPointerCapture(event.pointerId);
    this.dragMode = event.button === 2 ? 'move' : 'rotate';
    this.dragPos = vec2.fromValues(event.clientX, event.clientY);

    if (event.button === 0) {
      this.selectBlock(event.offsetX, event.offsetY);
    }
  }

  private onPointerMove(event: PointerEvent): void {
    if (!this.dragPos) return;

    const dx = (event.clientX - this.dragPos[0]) / 100;
    const dy = (event.clientY - this.dragPos[1]) / 100;
    vec2.set(this.dragPos, event.clientX, event.clientY);

    if (this.dragMode === 'move') {
      this.moveCameraByDrag(dx * 100, dy * 100);
      this.render();
      return;
    }

    this.cRot[0] = (this.cRot[0] + dx) % (Math.PI * 2);
    this.cRot[1] = clamp(this.cRot[1] + dy, -Math.PI / 2, Math.PI / 2);
    this.render();
  }

  private onPointerUp(event: PointerEvent): void {
    this.endDrag();
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
  }

  private endDrag(): void {
    this.dragMode = undefined;
    this.dragPos = undefined;
  }

  private onWheel(event: WheelEvent): void {
    event.preventDefault();
    this.cDist = clamp(this.cDist + event.deltaY * 0.02, 2, 400);
    this.render();
  }

  private onKeyDown(event: KeyboardEvent): void {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (!this.isMovementKey(event.code)) return;

    event.preventDefault();
    this.movement.add(event.code);
    this.render();
  }

  private onKeyUp(event: KeyboardEvent): void {
    if (!this.isMovementKey(event.code)) return;

    event.preventDefault();
    this.movement.delete(event.code);
  }

  private isMovementKey(code: string): boolean {
    return ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space', 'ShiftLeft', 'ShiftRight'].includes(code);
  }

  private selectBlock(x: number, y: number): void {
    if (!this.renderer || !this.model) return;

    this.resize();
    this.clearFrame();
    this.renderer.drawColoredStructure(this.getViewMatrix());

    const color = new Uint8Array(4);
    this.gl.readPixels(x, this.canvas.height - y, 1, 1, this.gl.RGBA, this.gl.UNSIGNED_BYTE, color);

    if (color[3] === 255) {
      this.selectedBlock = vec3.fromValues(color[0], color[1], color[2]);
      this.onSelect(this.model.blocks.find(block => samePos(block.pos, this.selectedBlock!)));
    } else {
      this.selectedBlock = undefined;
      this.onSelect(undefined);
    }

    this.render();
  }

  private getViewMatrix(): mat4 {
    const viewMatrix = mat4.create();
    mat4.translate(viewMatrix, viewMatrix, [0, 0, -this.cDist]);
    mat4.rotateX(viewMatrix, viewMatrix, this.cRot[1]);
    mat4.rotateY(viewMatrix, viewMatrix, this.cRot[0]);
    mat4.translate(viewMatrix, viewMatrix, this.cPos);
    return viewMatrix;
  }

  private resize(): void {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height || !this.renderer) return;

    const pixelRatio = Math.min(window.devicePixelRatio, 2);
    const displayWidth = Math.floor(width * pixelRatio);
    const displayHeight = Math.floor(height * pixelRatio);

    if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
      this.canvas.width = displayWidth;
      this.canvas.height = displayHeight;
    }

    this.renderer.setViewport(0, 0, this.canvas.width, this.canvas.height);
  }

  private render(): void {
    if (this.frame) return;

    this.frame = requestAnimationFrame(time => {
      this.frame = 0;
      if (!this.renderer) return;

      this.updateMovement(time);
      this.resize();
      this.clearFrame();
      const viewMatrix = this.getViewMatrix();
      this.renderer.drawGrid(viewMatrix);
      this.renderer.drawStructure(viewMatrix);
      if (this.selectedBlock) {
        this.renderer.drawOutline(viewMatrix, this.selectedBlock);
      }

      if (this.movement.size > 0) {
        this.render();
      }
    });
  }

  dispose(): void {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.movement.clear();
  }

  private clearFrame(): void {
    this.gl.clearColor(0.063, 0.075, 0.086, 1);
    this.gl.clearDepth(1);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
  }

  private updateMovement(time: number): void {
    const delta = this.lastFrameTime === 0 ? 0 : Math.min(64, time - this.lastFrameTime);
    this.lastFrameTime = time;
    if (this.movement.size === 0 || delta === 0) return;

    const direction = vec3.fromValues(0, 0, 0);
    if (this.movement.has('KeyW')) direction[2] += 1;
    if (this.movement.has('KeyS')) direction[2] -= 1;
    if (this.movement.has('KeyA')) direction[0] += 1;
    if (this.movement.has('KeyD')) direction[0] -= 1;
    if (this.movement.has('Space')) direction[1] -= 1;
    if (this.movement.has('ShiftLeft') || this.movement.has('ShiftRight')) direction[1] += 1;
    if (vec3.squaredLength(direction) === 0) return;

    vec3.normalize(direction, direction);
    vec3.scale(direction, direction, delta * Math.max(0.01, this.cDist * 0.0016));
    vec3.rotateY(direction, direction, [0, 0, 0], -this.cRot[0]);
    vec3.add(this.cPos, this.cPos, direction);
  }

  private moveCameraByDrag(dx: number, dy: number): void {
    const direction = vec3.fromValues(dx, 0, dy);
    if (vec3.squaredLength(direction) === 0) return;

    vec3.scale(direction, direction, Math.max(0.01, this.cDist * 0.002));
    vec3.rotateY(direction, direction, [0, 0, 0], -this.cRot[0]);
    vec3.add(this.cPos, this.cPos, direction);
  }
}
