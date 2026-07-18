import { mat4, vec4 } from 'gl-matrix';
import { Mesh, Quad, Renderer, ShaderProgram, Vector } from 'deepslate/render';

export type ViewerBoundingBox = {
  id: string;
  label: string;
  min: [number, number, number];
  max: [number, number, number];
};

type ProjectedPoint = [number, number] | undefined;

const BOX_THICKNESS = 0.07;
const SELECTED_BOX_THICKNESS = 0.09;
const HOVERED_BOX_THICKNESS = 0.12;
const PICK_DISTANCE_PX = 10;

const BOX_EDGES: Array<[number, number]> = [
  [0, 1],
  [0, 2],
  [0, 4],
  [1, 3],
  [1, 5],
  [2, 3],
  [2, 6],
  [3, 7],
  [4, 5],
  [4, 6],
  [5, 7],
  [6, 7],
];

const VERTEX_SHADER = `
  attribute vec4 vertPos;
  attribute vec3 vertColor;

  uniform mat4 mView;
  uniform mat4 mProj;

  varying highp vec3 vColor;

  void main(void) {
    gl_Position = mProj * mView * vertPos;
    vColor = vertColor;
  }
`;

const FRAGMENT_SHADER = `
  precision highp float;
  varying highp vec3 vColor;

  void main(void) {
    gl_FragColor = vec4(vColor, 0.82);
  }
`;

export class BoundingBoxRenderer extends Renderer {
  private readonly lineShaderProgram: WebGLProgram;
  private mesh = new Mesh();
  private hoveredMesh = new Mesh();
  private selectedMesh = new Mesh();
  private boxes: ViewerBoundingBox[] = [];
  private hoveredId?: string;
  private selectedId?: string;

  constructor(gl: WebGLRenderingContext) {
    super(gl);
    this.lineShaderProgram = new ShaderProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER).getProgram();
  }

  setBoxes(boxes: ViewerBoundingBox[]): void {
    this.boxes = boxes;
    if (!boxes.some(box => box.id === this.hoveredId)) this.hoveredId = undefined;
    if (!boxes.some(box => box.id === this.selectedId)) this.selectedId = undefined;
    this.rebuildMeshes();
  }

  setHoveredId(id: string | undefined): boolean {
    if (this.hoveredId === id) return false;
    this.hoveredId = id;
    this.rebuildMeshes();
    return true;
  }

  setSelectedId(id: string | undefined): void {
    if (this.selectedId === id) return;
    this.selectedId = id;
    this.rebuildMeshes();
  }

  draw(viewMatrix: mat4): void {
    if (this.boxes.length === 0) return;

    const depthEnabled = this.gl.isEnabled(this.gl.DEPTH_TEST);
    const cullEnabled = this.gl.isEnabled(this.gl.CULL_FACE);
    const polygonOffsetEnabled = this.gl.isEnabled(this.gl.POLYGON_OFFSET_FILL);
    const depthFunc = this.gl.getParameter(this.gl.DEPTH_FUNC) as number;
    const polygonOffsetFactor = this.gl.getParameter(this.gl.POLYGON_OFFSET_FACTOR) as number;
    const polygonOffsetUnits = this.gl.getParameter(this.gl.POLYGON_OFFSET_UNITS) as number;

    this.gl.enable(this.gl.DEPTH_TEST);
    this.gl.depthFunc(this.gl.LEQUAL);
    this.gl.disable(this.gl.CULL_FACE);
    this.gl.enable(this.gl.POLYGON_OFFSET_FILL);
    this.gl.polygonOffset(-1, -1);
    this.setShader(this.lineShaderProgram);
    this.prepareDraw(viewMatrix);
    this.drawMesh(this.mesh, { pos: true, color: true });
    this.drawMesh(this.selectedMesh, { pos: true, color: true });
    this.drawMesh(this.hoveredMesh, { pos: true, color: true });

    this.gl.polygonOffset(polygonOffsetFactor, polygonOffsetUnits);
    if (!polygonOffsetEnabled) this.gl.disable(this.gl.POLYGON_OFFSET_FILL);
    if (cullEnabled) this.gl.enable(this.gl.CULL_FACE);
    this.gl.depthFunc(depthFunc);
    if (!depthEnabled) this.gl.disable(this.gl.DEPTH_TEST);
  }

  pick(x: number, y: number, viewMatrix: mat4, width: number, height: number): ViewerBoundingBox | undefined {
    if (width <= 0 || height <= 0) return undefined;

    const viewProjection = mat4.create();
    mat4.multiply(viewProjection, this.projMatrix, viewMatrix);
    let best: { box: ViewerBoundingBox; distance: number } | undefined;

    for (const box of this.boxes) {
      const corners = boxCorners(box).map(corner => this.project(corner, viewProjection, width, height));
      for (const [startIndex, endIndex] of BOX_EDGES) {
        const start = corners[startIndex];
        const end = corners[endIndex];
        if (!start || !end) continue;
        const distance = pointSegmentDistance(x, y, start, end);
        if (distance <= PICK_DISTANCE_PX && (!best || distance < best.distance)) {
          best = { box, distance };
        }
      }
    }

    return best?.box;
  }

  private rebuildMeshes(): void {
    const mesh = new Mesh();
    const hoveredMesh = new Mesh();
    const selectedMesh = new Mesh();

    this.boxes.forEach((box, index) => {
      const selected = box.id === this.selectedId;
      const hovered = box.id === this.hoveredId;
      const baseColor: [number, number, number] = selected ? [1, 0.92, 0.34] : boxColor(index);
      const target = hovered ? hoveredMesh : selected ? selectedMesh : mesh;
      const color = hovered ? brightenColor(baseColor) : baseColor;
      const thickness = hovered
        ? HOVERED_BOX_THICKNESS
        : selected
          ? SELECTED_BOX_THICKNESS
          : BOX_THICKNESS;
      addThickLineCube(target, box, color, thickness);
    });

    this.mesh = mesh.rebuild(this.gl, { pos: true, color: true });
    this.hoveredMesh = hoveredMesh.rebuild(this.gl, { pos: true, color: true });
    this.selectedMesh = selectedMesh.rebuild(this.gl, { pos: true, color: true });
  }

  private project(
    point: [number, number, number],
    viewProjection: mat4,
    width: number,
    height: number,
  ): ProjectedPoint {
    const clip = vec4.fromValues(point[0], point[1], point[2], 1);
    vec4.transformMat4(clip, clip, viewProjection);
    if (clip[3] <= 0) return undefined;

    const ndcX = clip[0] / clip[3];
    const ndcY = clip[1] / clip[3];
    return [((ndcX + 1) * width) / 2, ((1 - ndcY) * height) / 2];
  }
}

function pointSegmentDistance(x: number, y: number, start: [number, number], end: [number, number]): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(x - start[0], y - start[1]);

  const t = Math.max(0, Math.min(1, ((x - start[0]) * dx + (y - start[1]) * dy) / lengthSquared));
  return Math.hypot(x - (start[0] + t * dx), y - (start[1] + t * dy));
}

function boxColor(index: number): [number, number, number] {
  const colors: Array<[number, number, number]> = [
    [1, 0.35, 0.3],
    [0.3, 0.72, 1],
    [0.42, 0.9, 0.55],
    [0.9, 0.52, 1],
  ];
  return colors[index % colors.length];
}

function brightenColor(color: [number, number, number]): [number, number, number] {
  return color.map(channel => channel + (1 - channel) * 0.7) as [number, number, number];
}

function addThickLineCube(
  mesh: Mesh,
  box: ViewerBoundingBox,
  color: [number, number, number],
  thickness: number,
): void {
  const corners = boxCorners(box);
  const halfThickness = thickness / 2;
  for (const [startIndex, endIndex] of BOX_EDGES) {
    const start = corners[startIndex];
    const end = corners[endIndex];
    addCuboid(
      mesh,
      [
        Math.min(start[0], end[0]) - halfThickness,
        Math.min(start[1], end[1]) - halfThickness,
        Math.min(start[2], end[2]) - halfThickness,
      ],
      [
        Math.max(start[0], end[0]) + halfThickness,
        Math.max(start[1], end[1]) + halfThickness,
        Math.max(start[2], end[2]) + halfThickness,
      ],
      color,
    );
  }
}

function addCuboid(
  mesh: Mesh,
  min: [number, number, number],
  max: [number, number, number],
  color: [number, number, number],
): void {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  const p000 = new Vector(x0, y0, z0);
  const p001 = new Vector(x0, y0, z1);
  const p010 = new Vector(x0, y1, z0);
  const p011 = new Vector(x0, y1, z1);
  const p100 = new Vector(x1, y0, z0);
  const p101 = new Vector(x1, y0, z1);
  const p110 = new Vector(x1, y1, z0);
  const p111 = new Vector(x1, y1, z1);

  mesh.quads.push(
    Quad.fromPoints(p000, p100, p110, p010).setColor(color),
    Quad.fromPoints(p001, p011, p111, p101).setColor(color),
    Quad.fromPoints(p000, p001, p101, p100).setColor(color),
    Quad.fromPoints(p010, p110, p111, p011).setColor(color),
    Quad.fromPoints(p000, p010, p011, p001).setColor(color),
    Quad.fromPoints(p100, p101, p111, p110).setColor(color),
  );
}

function boxCorners(box: ViewerBoundingBox): Array<[number, number, number]> {
  const [x0, y0, z0] = box.min;
  const [x1, y1, z1] = box.max;
  return [
    [x0, y0, z0],
    [x1, y0, z0],
    [x0, y1, z0],
    [x1, y1, z0],
    [x0, y0, z1],
    [x1, y0, z1],
    [x0, y1, z1],
    [x1, y1, z1],
  ];
}
