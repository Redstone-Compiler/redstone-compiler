import { mat4, vec3, vec4 } from 'gl-matrix';
import { Mesh, Quad, Renderer, ShaderProgram, Vector } from 'deepslate/render';

export type ViewerRoute = {
  id: string;
  sourceLabel: string;
  sinkLabel: string;
  points: Array<[number, number, number]>;
  pathLength: number;
  blockCount: number;
};

type ProjectedPoint = [number, number] | undefined;

const ROUTE_THICKNESS = 0.055;
const RELATED_ROUTE_THICKNESS = 0.08;
const SELECTED_ROUTE_THICKNESS = 0.105;
const HOVERED_ROUTE_THICKNESS = 0.12;
const PICK_DISTANCE_PX = 9;

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
  uniform highp float opacity;

  void main(void) {
    gl_FragColor = vec4(vColor, opacity);
  }
`;

export class RouteRenderer extends Renderer {
  private readonly routeShaderProgram: WebGLProgram;
  private mesh = new Mesh();
  private relatedMesh = new Mesh();
  private selectedMesh = new Mesh();
  private hoveredMesh = new Mesh();
  private routes: ViewerRoute[] = [];
  private relatedIds = new Set<string>();
  private selectedId?: string;
  private hoveredId?: string;

  constructor(gl: WebGLRenderingContext) {
    super(gl);
    this.routeShaderProgram = new ShaderProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER).getProgram();
  }

  setRoutes(routes: ViewerRoute[]): void {
    this.routes = routes;
    const ids = new Set(routes.map(route => route.id));
    this.relatedIds = new Set(Array.from(this.relatedIds).filter(id => ids.has(id)));
    if (!ids.has(this.selectedId ?? '')) this.selectedId = undefined;
    if (!ids.has(this.hoveredId ?? '')) this.hoveredId = undefined;
    this.rebuildMeshes();
  }

  setRelatedIds(ids: Iterable<string>): boolean {
    const next = new Set(ids);
    if (sameStringSet(this.relatedIds, next)) return false;
    this.relatedIds = next;
    this.rebuildMeshes();
    return true;
  }

  setSelectedId(id: string | undefined): boolean {
    if (this.selectedId === id) return false;
    this.selectedId = id;
    this.rebuildMeshes();
    return true;
  }

  setHoveredId(id: string | undefined): boolean {
    if (this.hoveredId === id) return false;
    this.hoveredId = id;
    this.rebuildMeshes();
    return true;
  }

  draw(viewMatrix: mat4): void {
    if (this.routes.length === 0) return;

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
    this.gl.polygonOffset(-2, -2);
    this.setShader(this.routeShaderProgram);
    this.prepareDraw(viewMatrix);
    this.drawRouteMesh(this.mesh, 0.28);
    this.drawRouteMesh(this.relatedMesh, 0.72);
    this.drawRouteMesh(this.selectedMesh, 0.96);
    this.drawRouteMesh(this.hoveredMesh, 1);

    this.gl.polygonOffset(polygonOffsetFactor, polygonOffsetUnits);
    if (!polygonOffsetEnabled) this.gl.disable(this.gl.POLYGON_OFFSET_FILL);
    if (cullEnabled) this.gl.enable(this.gl.CULL_FACE);
    this.gl.depthFunc(depthFunc);
    if (!depthEnabled) this.gl.disable(this.gl.DEPTH_TEST);
  }

  pick(x: number, y: number, viewMatrix: mat4, width: number, height: number): ViewerRoute | undefined {
    if (width <= 0 || height <= 0) return undefined;

    const viewProjection = mat4.create();
    mat4.multiply(viewProjection, this.projMatrix, viewMatrix);
    let best: { route: ViewerRoute; distance: number } | undefined;
    for (const route of this.routes) {
      const points = route.points.map(point => this.project(point, viewProjection, width, height));
      for (let index = 1; index < points.length; index += 1) {
        const start = points[index - 1];
        const end = points[index];
        if (!start || !end) continue;
        const distance = pointSegmentDistance(x, y, start, end);
        if (distance <= PICK_DISTANCE_PX && (!best || distance < best.distance)) {
          best = { route, distance };
        }
      }
    }
    return best?.route;
  }

  private drawRouteMesh(mesh: Mesh, opacity: number): void {
    if (mesh.isEmpty()) return;
    const location = this.gl.getUniformLocation(this.routeShaderProgram, 'opacity');
    this.gl.uniform1f(location, opacity);
    this.drawMesh(mesh, { pos: true, color: true });
  }

  private rebuildMeshes(): void {
    const mesh = new Mesh();
    const relatedMesh = new Mesh();
    const selectedMesh = new Mesh();
    const hoveredMesh = new Mesh();

    for (const route of this.routes) {
      const hovered = route.id === this.hoveredId;
      const selected = route.id === this.selectedId;
      const related = this.relatedIds.has(route.id);
      const target = hovered ? hoveredMesh : selected ? selectedMesh : related ? relatedMesh : mesh;
      const color = hovered || selected ? brightenColor(routeColor(route.sourceLabel)) : routeColor(route.sourceLabel);
      const thickness = hovered
        ? HOVERED_ROUTE_THICKNESS
        : selected
          ? SELECTED_ROUTE_THICKNESS
          : related
            ? RELATED_ROUTE_THICKNESS
            : ROUTE_THICKNESS;
      addRoute(target, route.points, color, thickness);
    }

    this.mesh = mesh.rebuild(this.gl, { pos: true, color: true });
    this.relatedMesh = relatedMesh.rebuild(this.gl, { pos: true, color: true });
    this.selectedMesh = selectedMesh.rebuild(this.gl, { pos: true, color: true });
    this.hoveredMesh = hoveredMesh.rebuild(this.gl, { pos: true, color: true });
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

function addRoute(
  mesh: Mesh,
  points: Array<[number, number, number]>,
  color: [number, number, number],
  thickness: number,
): void {
  for (let index = 1; index < points.length; index += 1) {
    addSegmentPrism(mesh, points[index - 1], points[index], color, thickness);
  }
  for (const point of points) addJointCube(mesh, point, color, thickness);
}

function addSegmentPrism(
  mesh: Mesh,
  start: [number, number, number],
  end: [number, number, number],
  color: [number, number, number],
  thickness: number,
): void {
  const direction = vec3.sub(vec3.create(), end, start);
  if (vec3.squaredLength(direction) < 1e-8) return;
  vec3.normalize(direction, direction);

  const reference = Math.abs(direction[1]) < 0.9
    ? vec3.fromValues(0, 1, 0)
    : vec3.fromValues(1, 0, 0);
  const sideA = vec3.cross(vec3.create(), direction, reference);
  vec3.normalize(sideA, sideA);
  vec3.scale(sideA, sideA, thickness / 2);
  const sideB = vec3.cross(vec3.create(), direction, sideA);
  vec3.normalize(sideB, sideB);
  vec3.scale(sideB, sideB, thickness / 2);

  const startCorners = prismCorners(start, sideA, sideB);
  const endCorners = prismCorners(end, sideA, sideB);
  mesh.quads.push(
    quad(startCorners[0], startCorners[1], endCorners[1], endCorners[0], color),
    quad(startCorners[1], startCorners[2], endCorners[2], endCorners[1], color),
    quad(startCorners[2], startCorners[3], endCorners[3], endCorners[2], color),
    quad(startCorners[3], startCorners[0], endCorners[0], endCorners[3], color),
    quad(startCorners[3], startCorners[2], startCorners[1], startCorners[0], color),
    quad(endCorners[0], endCorners[1], endCorners[2], endCorners[3], color),
  );
}

function prismCorners(
  point: [number, number, number],
  sideA: vec3,
  sideB: vec3,
): Array<[number, number, number]> {
  return [
    offsetPoint(point, sideA, sideB, 1, 1),
    offsetPoint(point, sideA, sideB, -1, 1),
    offsetPoint(point, sideA, sideB, -1, -1),
    offsetPoint(point, sideA, sideB, 1, -1),
  ];
}

function offsetPoint(
  point: [number, number, number],
  sideA: vec3,
  sideB: vec3,
  scaleA: number,
  scaleB: number,
): [number, number, number] {
  return [
    point[0] + sideA[0] * scaleA + sideB[0] * scaleB,
    point[1] + sideA[1] * scaleA + sideB[1] * scaleB,
    point[2] + sideA[2] * scaleA + sideB[2] * scaleB,
  ];
}

function addJointCube(
  mesh: Mesh,
  point: [number, number, number],
  color: [number, number, number],
  thickness: number,
): void {
  const half = thickness / 2;
  const [x, y, z] = point;
  const p000: [number, number, number] = [x - half, y - half, z - half];
  const p111: [number, number, number] = [x + half, y + half, z + half];
  const [x0, y0, z0] = p000;
  const [x1, y1, z1] = p111;
  const p001: [number, number, number] = [x0, y0, z1];
  const p010: [number, number, number] = [x0, y1, z0];
  const p011: [number, number, number] = [x0, y1, z1];
  const p100: [number, number, number] = [x1, y0, z0];
  const p101: [number, number, number] = [x1, y0, z1];
  const p110: [number, number, number] = [x1, y1, z0];
  mesh.quads.push(
    quad(p000, p100, p110, p010, color),
    quad(p001, p011, p111, p101, color),
    quad(p000, p001, p101, p100, color),
    quad(p010, p110, p111, p011, color),
    quad(p000, p010, p011, p001, color),
    quad(p100, p101, p111, p110, color),
  );
}

function quad(
  p1: [number, number, number],
  p2: [number, number, number],
  p3: [number, number, number],
  p4: [number, number, number],
  color: [number, number, number],
): Quad {
  return Quad.fromPoints(toVector(p1), toVector(p2), toVector(p3), toVector(p4)).setColor(color);
}

function toVector(point: [number, number, number]): Vector {
  return new Vector(point[0], point[1], point[2]);
}

function routeColor(label: string): [number, number, number] {
  const colors: Array<[number, number, number]> = [
    [0.2, 0.78, 1],
    [1, 0.46, 0.3],
    [0.45, 0.92, 0.5],
    [0.92, 0.52, 1],
    [1, 0.78, 0.24],
    [0.35, 0.9, 0.82],
  ];
  let hash = 2166136261;
  for (const char of label) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return colors[(hash >>> 0) % colors.length];
}

function brightenColor(color: [number, number, number]): [number, number, number] {
  return color.map(channel => channel + (1 - channel) * 0.58) as [number, number, number];
}

function pointSegmentDistance(x: number, y: number, start: [number, number], end: [number, number]): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(x - start[0], y - start[1]);
  const t = Math.max(0, Math.min(1, ((x - start[0]) * dx + (y - start[1]) * dy) / lengthSquared));
  return Math.hypot(x - (start[0] + t * dx), y - (start[1] + t * dy));
}

function sameStringSet(left: Set<string>, right: Set<string>): boolean {
  return left.size === right.size && Array.from(left).every(value => right.has(value));
}
