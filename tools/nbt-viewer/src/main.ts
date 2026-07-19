import './styles.css';
import type { Viz } from '@viz-js/viz';
import { unzipSync } from 'fflate';
import { loadNbtFile, stringifyNbt } from './nbt/loadNbt';
import { toStructureModel } from './nbt/toStructure';
import { StructureViewer } from './render/StructureViewer';
import { highlightRcir } from './syntax/rcir';
import { highlightVerilog, type VerilogHighlightState } from './syntax/verilog';
import {
  NbtSimulation,
  NbtSimulationError,
  emptyWaveform,
  type GraphDotInfo,
  type SnapshotInfo,
  type TraceEntry,
  type Waveform,
  type WaveformSignal,
} from './sim/NbtSimulation';
import type { StructureBlock, StructureModel, StructurePaletteEntry } from './types';

interface DroppedFileSystemEntry {
  readonly fullPath: string;
  readonly isDirectory: boolean;
  readonly isFile: boolean;
  readonly name: string;
}

interface DroppedFileSystemFileEntry extends DroppedFileSystemEntry {
  readonly isDirectory: false;
  readonly isFile: true;
  file(successCallback: (file: File) => void, errorCallback?: (error: DOMException) => void): void;
}

interface DroppedFileSystemDirectoryEntry extends DroppedFileSystemEntry {
  readonly isDirectory: true;
  readonly isFile: false;
  createReader(): DroppedFileSystemDirectoryReader;
}

interface DroppedFileSystemDirectoryReader {
  readEntries(
    successCallback: (entries: DroppedFileSystemEntry[]) => void,
    errorCallback?: (error: DOMException) => void,
  ): void;
}

interface DroppedFileSystemFileHandle {
  readonly kind: 'file';
  readonly name: string;
  getFile(): Promise<File>;
}

interface DroppedFileSystemDirectoryHandle {
  readonly kind: 'directory';
  readonly name: string;
  values(): AsyncIterable<DroppedFileSystemHandle>;
}

type DroppedFileSystemHandle = DroppedFileSystemFileHandle | DroppedFileSystemDirectoryHandle;
type TraceAnimation = {
  timer: number;
  token: number;
};
type GraphTab = 'world' | 'logic';
type GraphWorldMode = 'raw' | 'folded';
type GraphLogicMode = 'raw' | 'simplified';
type GraphEdgeInfo = {
  element: SVGGElement;
  source: string;
  target: string;
};
type ExampleFile = {
  kind: 'nbt' | 'snapshot';
  name: string;
  path: string;
  size: number;
  outputsPath?: string;
};
type SnapshotArtifact = {
  kind: string;
  path: string;
};
type SnapshotManifest = {
  format: 'redstone-compiler.snapshot.v1';
  status: 'success' | 'failed' | 'aborted';
  top_module?: string;
  final_nbt?: string;
  artifacts: SnapshotArtifact[];
};
type SnapshotInstance = {
  instanceId?: number;
  instance: string;
  module: string;
  artifactPath: string;
  circuitPath: string;
  global_bbox: {
    min: [number, number, number];
    max: [number, number, number];
  };
  blockCount?: number;
};
type SnapshotRoute = {
  id: string;
  index: number;
  netId?: number;
  source: [number, number, number];
  sourceLabel: string;
  sink: [number, number, number];
  sinkLabel: string;
  path: Array<[number, number, number]>;
  blocks: Array<[number, number, number]>;
  pathLength: number;
  blockCount: number;
};
type SnapshotConstraint = {
  id: string;
  status: 'satisfied' | 'violated' | 'not_evaluated';
  detail: string;
  instanceIds: number[];
  netIds: number[];
};
type SnapshotDebugLocation =
  | {
    kind: 'source';
    file: string;
    start_line: number;
    start_column: number;
    end_line: number;
    end_column: number;
  }
  | { kind: 'derived'; label: string; parent: number }
  | { kind: 'fused'; parents: number[] };
type SnapshotDebugRange = {
  entity: string;
  start_line: number;
  end_line: number;
  location: number;
};
type SnapshotDebugRelation = {
  kind: 'instantiates' | 'canonicalized_to';
  from: string;
  to: string;
};
type SnapshotDebugEntity = {
  location: number;
  kind: 'module' | 'port' | 'net' | 'instance' | 'cell' | 'node' | 'other';
  parent_scope?: string;
};
type SnapshotSourceMap = {
  format: 'redstone-compiler.source-map.v1';
  locations: SnapshotDebugLocation[];
  entities: Record<string, SnapshotDebugEntity>;
  relations: SnapshotDebugRelation[];
  documents: Record<string, SnapshotDebugRange[]>;
};
type LoadedSnapshot = {
  manifest: SnapshotManifest;
  filesByPath: Map<string, File>;
  instances: SnapshotInstance[];
  routes: SnapshotRoute[];
  constraints: SnapshotConstraint[];
  interfaceJson?: string;
  sourceMap?: SnapshotSourceMap;
};

const TRACE_ANIMATION_INTERVAL_MS = 50;
const WAVEFORM_LABEL_WIDTH = 188;
const WAVEFORM_ROW_HEIGHT = 24;
const WAVEFORM_CYCLE_WIDTH = 34;
const WAVEFORM_HEADER_HEIGHT = 20;
const GRAPH_MINIMAP_INSET = 0;
const GRAPH_MINIMAP_MAX_WIDTH = 360;
const GRAPH_MINIMAP_MAX_HEIGHT = 240;
const GRAPH_MINIMAP_MIN_WIDTH = 150;
const GRAPH_MINIMAP_MIN_HEIGHT = 48;
const IR_LOCATION_HUES = [
  210, 32, 145, 275, 55, 180, 345, 105, 235, 15, 165, 300,
];

function resolveAssetPath(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, window.location.origin).href;
}

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="app-shell">
    <section id="drop-zone" class="workspace">
      <section class="viewer-panel">
        <canvas id="structure-canvas"></canvas>
        <div id="bbox-tooltip" class="bbox-tooltip hidden" role="tooltip" aria-hidden="true">
          <strong id="bbox-tooltip-title"></strong>
          <span id="bbox-tooltip-detail"></span>
        </div>
        <div class="floating-actions">
          <div class="file-actions-row">
            <label class="file-button">
              Open Snapshot
              <input id="snapshot-input" type="file" accept=".rsnap" />
            </label>
            <label class="file-button">
              Open Folder
              <input id="folder-input" type="file" multiple />
            </label>
            <label class="file-button">
              Open NBT
              <input id="file-input" type="file" accept=".nbt,.dat,.schem,.schematic,.litematic,.mcstructure" />
            </label>
          </div>
          <button id="toggle-blocks" class="file-button graph-button snapshot-box-button active" type="button" aria-pressed="true">Blocks</button>
          <button id="toggle-grid" class="file-button graph-button snapshot-box-button active" type="button" aria-pressed="true">Grid</button>
          <button id="toggle-snapshot-boxes" class="file-button graph-button snapshot-box-button hidden" type="button">Boxes</button>
          <button id="toggle-snapshot-routes" class="file-button graph-button snapshot-box-button hidden" type="button">Routes</button>
          <button id="open-graphs" class="file-button graph-button" type="button">Graphs</button>
        </div>
        <details id="switches-panel" class="floating-panel switches-panel" open>
          <summary>
            <span>Switches</span>
            <span id="switches-count">No switches</span>
          </summary>
          <div id="switches-actions" class="switches-actions hidden">
            <button id="switches-all-on" type="button">All On</button>
            <button id="switches-all-off" type="button">All Off</button>
            <label id="trace-simulation-toggle" class="trace-simulation-toggle" title="Collect trace and waveform data while switches simulate">
              <input id="trace-simulation-enabled" type="checkbox" />
              <span class="trace-simulation-label">Trace</span>
              <span class="trace-simulation-track" aria-hidden="true">
                <span class="trace-simulation-knob"></span>
              </span>
              <strong id="trace-simulation-state">Off</strong>
            </label>
          </div>
          <div id="switches-list" class="switches-list empty">Open an NBT file to control switches.</div>
        </details>
        <details id="files-panel" class="floating-panel files-panel">
          <summary>
            <span id="files-title">Files</span>
            <span id="files-count">No folder</span>
          </summary>
          <div id="files-list" class="files-list empty">Open a folder to browse NBT files.</div>
        </details>
        <aside class="floating-panel inspector-panel">
          <div class="panel-header">
            <strong>Inspector</strong>
            <button id="toggle-switch" class="panel-action hidden" type="button">Toggle</button>
          </div>
          <pre id="inspector">Select a block in the 3D view.</pre>
        </aside>
        <details id="trace-panel" class="floating-panel trace-panel">
          <summary>
            <span>Trace</span>
            <span class="trace-summary-actions">
              <button id="trace-expand" class="trace-expand-button" type="button" aria-label="Expand trace" title="Expand trace" disabled>Expand</button>
              <span id="trace-count">No events</span>
            </span>
          </summary>
          <div id="trace-content" class="trace-content">
            <div class="trace-controls">
              <button id="trace-prev" type="button" aria-label="Previous cycle">Prev</button>
              <input id="trace-cycle" type="range" min="0" max="0" value="0" />
              <button id="trace-next" type="button" aria-label="Next cycle">Next</button>
              <label class="trace-cycle-mode" title="Show actual simulator cycle numbers">
                <input id="trace-show-actual-cycles" type="checkbox" />
                <span>Actual cycles</span>
              </label>
              <span id="trace-cycle-label">cycle -</span>
            </div>
            <div id="waveform-viewer" class="waveform-viewer">
              <label class="waveform-filter-floating" title="Show only signals with value changes">
                <input id="waveform-changed-only" type="checkbox" />
                <span>Changed only</span>
              </label>
              <div id="waveform-labels" class="waveform-labels"></div>
              <div id="waveform-scroll" class="waveform-scroll">
                <canvas id="waveform-canvas"></canvas>
              </div>
            </div>
            <details class="trace-log">
              <summary>Log</summary>
              <pre id="trace-output">Trace simulation is off.</pre>
            </details>
          </div>
        </details>
        <div id="viewer-empty" class="viewer-empty">Drop an .nbt file or use Open NBT.</div>
      </section>
    </section>
    <dialog id="graph-dialog" class="graph-dialog">
      <div class="graph-dialog-surface">
        <header class="graph-dialog-header">
          <strong>Graphs</strong>
          <button id="close-graphs" class="panel-action" type="button">Close</button>
        </header>
        <div class="graph-tabs" role="tablist" aria-label="Graph views">
          <button id="graph-world-tab" class="graph-tab active" type="button" role="tab">World Graph</button>
          <button id="graph-logic-tab" class="graph-tab" type="button" role="tab">Logic Graph</button>
          <div id="graph-world-mode" class="graph-world-mode" aria-label="World graph mode">
            <button id="graph-world-raw" class="graph-mode-button active" type="button">Raw</button>
            <button id="graph-world-folded" class="graph-mode-button" type="button">Folded</button>
          </div>
          <div id="graph-logic-mode" class="graph-world-mode hidden" aria-label="Logic graph mode">
            <button id="graph-logic-raw" class="graph-mode-button active" type="button">Raw</button>
            <button id="graph-logic-simplified" class="graph-mode-button" type="button">Simplified</button>
          </div>
          <label id="graph-high-level-toggle" class="graph-tag-toggle hidden">
            <input id="graph-high-level-gates" type="checkbox" />
            <span>High-level Gates</span>
          </label>
          <label class="graph-tag-toggle">
            <input id="graph-show-tags" type="checkbox" checked />
            <span>Show Tag</span>
          </label>
          <div class="graph-zoom-controls" aria-label="Graph zoom">
            <button id="graph-zoom-out" class="graph-zoom-button" type="button" aria-label="Zoom out">-</button>
            <button id="graph-zoom-reset" class="graph-zoom-value" type="button" aria-label="Reset zoom">100%</button>
            <button id="graph-zoom-in" class="graph-zoom-button" type="button" aria-label="Zoom in">+</button>
          </div>
        </div>
        <div id="graph-status" class="graph-status">Open an NBT file to inspect graphs.</div>
        <div class="graph-viewer">
          <div id="graph-output" class="graph-output"></div>
          <div id="graph-selection-actions" class="graph-selection-actions hidden">
            <button id="open-selected-graph" class="graph-selection-button" type="button">Open Selection</button>
            <button id="open-selected-nbt" class="graph-selection-button" type="button">Open Selection As NBT</button>
          </div>
          <div id="graph-minimap" class="graph-minimap hidden" aria-hidden="true">
            <div id="graph-minimap-content" class="graph-minimap-content"></div>
            <div id="graph-minimap-viewport" class="graph-minimap-viewport"></div>
          </div>
        </div>
      </div>
    </dialog>
    <dialog id="selected-graph-dialog" class="graph-dialog">
      <div class="graph-dialog-surface">
        <header class="graph-dialog-header">
          <strong>Selected Graphs</strong>
          <button id="close-selected-graphs" class="panel-action" type="button">Close</button>
        </header>
        <div class="graph-tabs" role="tablist" aria-label="Selected graph views">
          <button id="selected-graph-world-tab" class="graph-tab active" type="button" role="tab">World Graph</button>
          <button id="selected-graph-logic-tab" class="graph-tab" type="button" role="tab">Logic Graph</button>
          <div id="selected-graph-world-mode" class="graph-world-mode" aria-label="Selected world graph mode">
            <button id="selected-graph-world-raw" class="graph-mode-button active" type="button">Raw</button>
            <button id="selected-graph-world-folded" class="graph-mode-button" type="button">Folded</button>
          </div>
          <div id="selected-graph-logic-mode" class="graph-world-mode hidden" aria-label="Selected logic graph mode">
            <button id="selected-graph-logic-raw" class="graph-mode-button active" type="button">Raw</button>
            <button id="selected-graph-logic-simplified" class="graph-mode-button" type="button">Simplified</button>
          </div>
          <label id="selected-graph-high-level-toggle" class="graph-tag-toggle hidden">
            <input id="selected-graph-high-level-gates" type="checkbox" />
            <span>High-level Gates</span>
          </label>
          <label class="graph-tag-toggle">
            <input id="selected-graph-show-tags" type="checkbox" checked />
            <span>Show Tag</span>
          </label>
          <div class="graph-zoom-controls" aria-label="Selected graph zoom">
            <button id="selected-graph-zoom-out" class="graph-zoom-button" type="button" aria-label="Zoom out">-</button>
            <button id="selected-graph-zoom-reset" class="graph-zoom-value" type="button" aria-label="Reset zoom">100%</button>
            <button id="selected-graph-zoom-in" class="graph-zoom-button" type="button" aria-label="Zoom in">+</button>
          </div>
        </div>
        <div id="selected-graph-status" class="graph-status">Select world graph nodes to open a focused graph.</div>
        <div class="graph-viewer">
          <div id="selected-graph-output" class="graph-output"></div>
        </div>
      </div>
    </dialog>
    <dialog id="selected-nbt-dialog" class="graph-dialog selected-nbt-dialog">
      <div class="graph-dialog-surface selected-nbt-dialog-surface">
        <header class="graph-dialog-header">
          <strong id="selected-nbt-title">Selected NBT</strong>
          <button id="close-selected-nbt" class="panel-action" type="button">Close</button>
        </header>
        <div id="selected-nbt-status" class="graph-status">Select world graph nodes to open focused NBT.</div>
        <div class="selected-nbt-viewer">
          <canvas id="selected-nbt-canvas"></canvas>
        </div>
      </div>
    </dialog>
    <dialog id="artifact-dialog" class="graph-dialog artifact-dialog">
      <div class="graph-dialog-surface artifact-dialog-surface">
        <header class="graph-dialog-header">
          <strong id="artifact-title">Snapshot artifact</strong>
          <div class="artifact-dialog-actions">
            <label id="ir-color-mapping-toggle" class="artifact-toggle hidden">
              <input id="ir-color-mapping" type="checkbox" checked />
              <span>Color mapping</span>
            </label>
            <button id="close-artifact" class="panel-action" type="button">Close</button>
          </div>
        </header>
        <div id="ir-comparison" class="ir-comparison hidden"></div>
        <pre id="artifact-content" class="artifact-content"></pre>
      </div>
    </dialog>
  </main>
`;

const input = document.querySelector<HTMLInputElement>('#file-input')!;
const snapshotInput = document.querySelector<HTMLInputElement>('#snapshot-input')!;
const folderInput = document.querySelector<HTMLInputElement>('#folder-input')!;
const dropZone = document.querySelector<HTMLElement>('#drop-zone')!;
const filesPanel = document.querySelector<HTMLDetailsElement>('#files-panel')!;
const filesTitle = document.querySelector<HTMLElement>('#files-title')!;
const filesList = document.querySelector<HTMLElement>('#files-list')!;
const filesCount = document.querySelector<HTMLElement>('#files-count')!;
const canvas = document.querySelector<HTMLCanvasElement>('#structure-canvas')!;
const bboxTooltip = document.querySelector<HTMLElement>('#bbox-tooltip')!;
const bboxTooltipTitle = document.querySelector<HTMLElement>('#bbox-tooltip-title')!;
const bboxTooltipDetail = document.querySelector<HTMLElement>('#bbox-tooltip-detail')!;
const viewerEmpty = document.querySelector<HTMLElement>('#viewer-empty')!;
const inspectorPanel = document.querySelector<HTMLElement>('.inspector-panel')!;
const inspector = document.querySelector<HTMLElement>('#inspector')!;
const toggleSwitchButton = document.querySelector<HTMLButtonElement>('#toggle-switch')!;
const tracePanel = document.querySelector<HTMLDetailsElement>('#trace-panel')!;
const traceExpandButton = document.querySelector<HTMLButtonElement>('#trace-expand')!;
const traceContent = document.querySelector<HTMLElement>('#trace-content')!;
const traceCount = document.querySelector<HTMLElement>('#trace-count')!;
const traceOutput = document.querySelector<HTMLElement>('#trace-output')!;
const traceCycleInput = document.querySelector<HTMLInputElement>('#trace-cycle')!;
const traceCycleLabel = document.querySelector<HTMLElement>('#trace-cycle-label')!;
const tracePrevButton = document.querySelector<HTMLButtonElement>('#trace-prev')!;
const traceNextButton = document.querySelector<HTMLButtonElement>('#trace-next')!;
const traceShowActualCyclesInput = document.querySelector<HTMLInputElement>('#trace-show-actual-cycles')!;
const waveformLabels = document.querySelector<HTMLElement>('#waveform-labels')!;
const waveformScroll = document.querySelector<HTMLElement>('#waveform-scroll')!;
const waveformCanvas = document.querySelector<HTMLCanvasElement>('#waveform-canvas')!;
const waveformChangedOnlyInput = document.querySelector<HTMLInputElement>('#waveform-changed-only')!;
const switchesPanel = document.querySelector<HTMLDetailsElement>('#switches-panel')!;
const switchesActions = document.querySelector<HTMLElement>('#switches-actions')!;
const switchesAllOnButton = document.querySelector<HTMLButtonElement>('#switches-all-on')!;
const switchesAllOffButton = document.querySelector<HTMLButtonElement>('#switches-all-off')!;
const switchesList = document.querySelector<HTMLElement>('#switches-list')!;
const switchesCount = document.querySelector<HTMLElement>('#switches-count')!;
const traceSimulationToggle = document.querySelector<HTMLLabelElement>('#trace-simulation-toggle')!;
const traceSimulationEnabledInput = document.querySelector<HTMLInputElement>('#trace-simulation-enabled')!;
const traceSimulationState = document.querySelector<HTMLElement>('#trace-simulation-state')!;
const openGraphsButton = document.querySelector<HTMLButtonElement>('#open-graphs')!;
const toggleBlocksButton = document.querySelector<HTMLButtonElement>('#toggle-blocks')!;
const toggleGridButton = document.querySelector<HTMLButtonElement>('#toggle-grid')!;
const toggleSnapshotBoxesButton = document.querySelector<HTMLButtonElement>('#toggle-snapshot-boxes')!;
const toggleSnapshotRoutesButton = document.querySelector<HTMLButtonElement>('#toggle-snapshot-routes')!;
const closeGraphsButton = document.querySelector<HTMLButtonElement>('#close-graphs')!;
const graphDialog = document.querySelector<HTMLDialogElement>('#graph-dialog')!;
const graphWorldTab = document.querySelector<HTMLButtonElement>('#graph-world-tab')!;
const graphLogicTab = document.querySelector<HTMLButtonElement>('#graph-logic-tab')!;
const graphWorldMode = document.querySelector<HTMLElement>('#graph-world-mode')!;
const graphWorldRawButton = document.querySelector<HTMLButtonElement>('#graph-world-raw')!;
const graphWorldFoldedButton = document.querySelector<HTMLButtonElement>('#graph-world-folded')!;
const graphLogicMode = document.querySelector<HTMLElement>('#graph-logic-mode')!;
const graphLogicRawButton = document.querySelector<HTMLButtonElement>('#graph-logic-raw')!;
const graphLogicSimplifiedButton = document.querySelector<HTMLButtonElement>('#graph-logic-simplified')!;
const graphHighLevelToggle = document.querySelector<HTMLElement>('#graph-high-level-toggle')!;
const graphHighLevelInput = document.querySelector<HTMLInputElement>('#graph-high-level-gates')!;
const graphShowTagsInput = document.querySelector<HTMLInputElement>('#graph-show-tags')!;
const graphZoomOutButton = document.querySelector<HTMLButtonElement>('#graph-zoom-out')!;
const graphZoomResetButton = document.querySelector<HTMLButtonElement>('#graph-zoom-reset')!;
const graphZoomInButton = document.querySelector<HTMLButtonElement>('#graph-zoom-in')!;
const graphStatus = document.querySelector<HTMLElement>('#graph-status')!;
const graphViewer = document.querySelector<HTMLElement>('.graph-viewer')!;
const graphOutput = document.querySelector<HTMLElement>('#graph-output')!;
const graphSelectionActions = document.querySelector<HTMLElement>('#graph-selection-actions')!;
const openSelectedGraphButton = document.querySelector<HTMLButtonElement>('#open-selected-graph')!;
const openSelectedNbtButton = document.querySelector<HTMLButtonElement>('#open-selected-nbt')!;
const graphMinimap = document.querySelector<HTMLElement>('#graph-minimap')!;
const graphMinimapContent = document.querySelector<HTMLElement>('#graph-minimap-content')!;
const graphMinimapViewport = document.querySelector<HTMLElement>('#graph-minimap-viewport')!;
const selectedGraphDialog = document.querySelector<HTMLDialogElement>('#selected-graph-dialog')!;
const closeSelectedGraphsButton = document.querySelector<HTMLButtonElement>('#close-selected-graphs')!;
const selectedGraphWorldTab = document.querySelector<HTMLButtonElement>('#selected-graph-world-tab')!;
const selectedGraphLogicTab = document.querySelector<HTMLButtonElement>('#selected-graph-logic-tab')!;
const selectedGraphWorldMode = document.querySelector<HTMLElement>('#selected-graph-world-mode')!;
const selectedGraphWorldRawButton = document.querySelector<HTMLButtonElement>('#selected-graph-world-raw')!;
const selectedGraphWorldFoldedButton = document.querySelector<HTMLButtonElement>('#selected-graph-world-folded')!;
const selectedGraphLogicMode = document.querySelector<HTMLElement>('#selected-graph-logic-mode')!;
const selectedGraphLogicRawButton = document.querySelector<HTMLButtonElement>('#selected-graph-logic-raw')!;
const selectedGraphLogicSimplifiedButton = document.querySelector<HTMLButtonElement>('#selected-graph-logic-simplified')!;
const selectedGraphHighLevelToggle = document.querySelector<HTMLElement>('#selected-graph-high-level-toggle')!;
const selectedGraphHighLevelInput = document.querySelector<HTMLInputElement>('#selected-graph-high-level-gates')!;
const selectedGraphShowTagsInput = document.querySelector<HTMLInputElement>('#selected-graph-show-tags')!;
const selectedGraphZoomOutButton = document.querySelector<HTMLButtonElement>('#selected-graph-zoom-out')!;
const selectedGraphZoomResetButton = document.querySelector<HTMLButtonElement>('#selected-graph-zoom-reset')!;
const selectedGraphZoomInButton = document.querySelector<HTMLButtonElement>('#selected-graph-zoom-in')!;
const selectedGraphStatus = document.querySelector<HTMLElement>('#selected-graph-status')!;
const selectedGraphOutput = document.querySelector<HTMLElement>('#selected-graph-output')!;
const selectedNbtDialog = document.querySelector<HTMLDialogElement>('#selected-nbt-dialog')!;
const selectedNbtTitle = document.querySelector<HTMLElement>('#selected-nbt-title')!;
const closeSelectedNbtButton = document.querySelector<HTMLButtonElement>('#close-selected-nbt')!;
const selectedNbtStatus = document.querySelector<HTMLElement>('#selected-nbt-status')!;
const selectedNbtCanvas = document.querySelector<HTMLCanvasElement>('#selected-nbt-canvas')!;
const artifactDialog = document.querySelector<HTMLDialogElement>('#artifact-dialog')!;
const closeArtifactButton = document.querySelector<HTMLButtonElement>('#close-artifact')!;
const artifactTitle = document.querySelector<HTMLElement>('#artifact-title')!;
const irComparison = document.querySelector<HTMLElement>('#ir-comparison')!;
const irColorMappingToggle = document.querySelector<HTMLElement>('#ir-color-mapping-toggle')!;
const irColorMappingInput = document.querySelector<HTMLInputElement>('#ir-color-mapping')!;
const artifactContent = document.querySelector<HTMLElement>('#artifact-content')!;

const viewer = new StructureViewer(canvas);
viewer.setSelectionHandler(renderSelection);
const selectedNbtViewer = new StructureViewer(selectedNbtCanvas);

let simulation: NbtSimulation | undefined;
let selectedBlock: StructureBlock | undefined;
let currentNbtBytes: Uint8Array | undefined;
let currentRoot: unknown;
let currentTrace: TraceEntry[] = [];
let traceCycles: number[] = [];
let historyTrace: TraceEntry[] = [];
let historySnapshots: SnapshotInfo[] = [];
let historyWaveform: Waveform = emptyWaveform;
let historyTraceCycles: number[] = [];
let traceCycleDisplayOffset = 0;
let traceShowActualCycles = false;
let traceSimulationEnabled = false;
let currentSnapshots: SnapshotInfo[] = [];
let currentWaveform: Waveform = emptyWaveform;
let selectedWaveformSignal: WaveformSignal | undefined;
let waveformChangedOnly = false;
let isTraceExpanded = false;
let traceBaseRoot: unknown;
let isTracePreviewActive = false;
let traceAnimation: TraceAnimation | undefined;
let traceAnimationToken = 0;
let waveformResizeFrame: number | undefined;
let graphDot: GraphDotInfo | undefined;
let currentOutputMetadataJson: string | undefined;
let graphTab: GraphTab = 'world';
let graphWorldModeValue: GraphWorldMode = 'raw';
let graphLogicModeValue: GraphLogicMode = 'raw';
let graphHighLevelLogic = false;
let graphShowTags = true;
let vizPromise: Promise<Viz> | undefined;
let graphMinimapScale = 1;
let isDraggingGraphMinimap = false;
let graphZoom = 1;
let selectedGraphNode: string | undefined;
let selectedGraphDot: GraphDotInfo | undefined;
let selectedGraphTab: GraphTab = 'world';
let selectedGraphWorldModeValue: GraphWorldMode = 'raw';
let selectedGraphLogicModeValue: GraphLogicMode = 'raw';
let selectedGraphHighLevelLogic = false;
let selectedGraphShowTags = true;
let selectedGraphZoom = 1;
let currentSnapshot: LoadedSnapshot | undefined;
let currentSnapshotPath: string | undefined;
let isolatedSnapshotBoxId: string | undefined;
let isolatedSnapshotRouteId: string | undefined;
let blocksVisible = true;
let gridVisible = true;
let snapshotBoxesVisible = true;
let snapshotRoutesVisible = true;
let pinnedIrLocations: number[] | undefined;
let pinnedIrEntities: string[] | undefined;
const irLocationColorSlots = new Map<string, number>();

folderInput.setAttribute('webkitdirectory', '');
folderInput.setAttribute('directory', '');

void loadExamples();

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (file) {
    leaveSnapshotMode();
    void openFile(file);
  }
});

snapshotInput.addEventListener('change', () => {
  const file = snapshotInput.files?.[0];
  if (!file) return;
  void openSnapshot([file]).catch(error => {
    inspector.textContent = error instanceof Error ? error.message : String(error);
  });
});

folderInput.addEventListener('change', () => {
  void openSnapshot(Array.from(folderInput.files ?? [])).catch(error => {
    inspector.textContent = error instanceof Error ? error.message : String(error);
  });
});

toggleSnapshotBoxesButton.addEventListener('click', () => {
  snapshotBoxesVisible = !snapshotBoxesVisible;
  updateSnapshotBoxes();
});

toggleBlocksButton.addEventListener('click', () => {
  blocksVisible = !blocksVisible;
  viewer.setBlocksVisible(blocksVisible);
  toggleBlocksButton.classList.toggle('active', blocksVisible);
  toggleBlocksButton.setAttribute('aria-pressed', String(blocksVisible));
});

toggleGridButton.addEventListener('click', () => {
  gridVisible = !gridVisible;
  viewer.setGridVisible(gridVisible);
  toggleGridButton.classList.toggle('active', gridVisible);
  toggleGridButton.setAttribute('aria-pressed', String(gridVisible));
});

window.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  if (pinnedIrLocations) {
    event.preventDefault();
    pinnedIrLocations = undefined;
    pinnedIrEntities = undefined;
    renderIrLocationHighlight(undefined, undefined);
    return;
  }
  if (!isolatedSnapshotBoxId && !isolatedSnapshotRouteId) return;
  event.preventDefault();
  setSnapshotRouteIsolation(undefined);
  setSnapshotBoxIsolation(undefined);
});

toggleSnapshotRoutesButton.addEventListener('click', () => {
  snapshotRoutesVisible = !snapshotRoutesVisible;
  updateSnapshotRoutes();
});

toggleSwitchButton.addEventListener('click', () => {
  void toggleSelectedSwitch().catch(error => {
    renderSimulationError(error);
  });
});

switchesAllOnButton.addEventListener('click', () => {
  void setAllSwitches(true).catch(error => {
    renderSimulationError(error);
  });
});

switchesAllOffButton.addEventListener('click', () => {
  void setAllSwitches(false).catch(error => {
    renderSimulationError(error);
  });
});

traceSimulationEnabledInput.addEventListener('change', () => {
  void setTraceSimulationEnabled(traceSimulationEnabledInput.checked).catch(error => {
    renderSimulationError(error);
  });
});

traceCycleInput.addEventListener('input', () => {
  cancelTraceAnimation();
  void renderTraceCycle(Number(traceCycleInput.value));
});

tracePrevButton.addEventListener('click', () => {
  cancelTraceAnimation();
  const axisCycles = getTraceAxisCycles();
  traceCycleInput.value = String(Math.max(0, Math.min(axisCycles.length - 1, Number(traceCycleInput.value) - 1)));
  void renderTraceCycle(Number(traceCycleInput.value));
});

traceNextButton.addEventListener('click', () => {
  cancelTraceAnimation();
  const axisCycles = getTraceAxisCycles();
  traceCycleInput.value = String(Math.min(axisCycles.length - 1, Number(traceCycleInput.value) + 1));
  void renderTraceCycle(Number(traceCycleInput.value));
});

traceShowActualCyclesInput.addEventListener('change', () => {
  cancelTraceAnimation();
  const cycle = selectedTraceCycle();
  traceShowActualCycles = traceShowActualCyclesInput.checked;
  pruneSelectedWaveformSignal();
  updateWaveformFilterControl();
  renderWaveformLabels();
  const selectedIndex = updateTraceCycleControls(findTraceAxisIndexForCycle(cycle));
  void renderTraceCycle(selectedIndex);
});

waveformCanvas.addEventListener('click', event => {
  const axisCycles = getTraceAxisCycles();
  if (axisCycles.length === 0) return;

  cancelTraceAnimation();
  const rect = waveformCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top + waveformScroll.scrollTop;
  const cycleIndex = Math.max(0, Math.min(axisCycles.length - 1, Math.floor(x / WAVEFORM_CYCLE_WIDTH)));
  const signalIndex = Math.floor((y - WAVEFORM_HEADER_HEIGHT) / WAVEFORM_ROW_HEIGHT);
  const visibleSignals = getVisibleWaveformSignals();
  if (visibleSignals[signalIndex]) {
    focusWaveformSignal(visibleSignals[signalIndex]);
  }
  traceCycleInput.value = String(cycleIndex);
  void renderTraceCycle(cycleIndex);
});

waveformScroll.addEventListener('scroll', () => {
  waveformLabels.scrollTop = waveformScroll.scrollTop;
});

waveformChangedOnlyInput.addEventListener('change', () => {
  waveformChangedOnly = waveformChangedOnlyInput.checked;
  const selectedIndex = Number(traceCycleInput.value);
  renderWaveformLabels();
  renderWaveform(selectedIndex);
  scrollWaveformToTraceIndex(selectedIndex);
});

window.addEventListener('resize', () => {
  if (waveformResizeFrame !== undefined) {
    window.cancelAnimationFrame(waveformResizeFrame);
  }

  waveformResizeFrame = window.requestAnimationFrame(() => {
    waveformResizeFrame = undefined;
    const selectedIndex = Number(traceCycleInput.value);
    renderWaveform(selectedIndex);
    scrollWaveformToTraceIndex(selectedIndex);
  });
});

traceExpandButton.addEventListener('click', event => {
  event.preventDefault();
  event.stopPropagation();
  setTraceExpanded(!isTraceExpanded);
});

openGraphsButton.addEventListener('click', () => {
  void openGraphDialog();
});

closeGraphsButton.addEventListener('click', () => {
  graphDialog.close();
});

graphDialog.addEventListener('click', event => {
  if (event.target === graphDialog) {
    graphDialog.close();
  }
});

graphWorldTab.addEventListener('click', () => {
  void setGraphTab('world');
});

graphLogicTab.addEventListener('click', () => {
  void setGraphTab('logic');
});

graphWorldRawButton.addEventListener('click', () => {
  void setGraphWorldMode('raw');
});

graphWorldFoldedButton.addEventListener('click', () => {
  void setGraphWorldMode('folded');
});

graphLogicRawButton.addEventListener('click', () => {
  void setGraphLogicMode('raw');
});

graphLogicSimplifiedButton.addEventListener('click', () => {
  void setGraphLogicMode('simplified');
});

graphHighLevelInput.addEventListener('change', () => {
  graphHighLevelLogic = graphHighLevelInput.checked;
  void renderGraphTab();
});

graphShowTagsInput.addEventListener('change', () => {
  graphShowTags = graphShowTagsInput.checked;
  void renderGraphTab();
});

graphZoomOutButton.addEventListener('click', () => {
  setGraphZoom(graphZoom - 0.25);
});

graphZoomResetButton.addEventListener('click', () => {
  setGraphZoom(1);
});

graphZoomInButton.addEventListener('click', () => {
  setGraphZoom(graphZoom + 0.25);
});

openSelectedGraphButton.addEventListener('click', () => {
  void openSelectedGraphView();
});

openSelectedNbtButton.addEventListener('click', () => {
  void openSelectedNbtView();
});

closeSelectedGraphsButton.addEventListener('click', () => {
  selectedGraphDialog.close();
});

selectedGraphDialog.addEventListener('click', event => {
  if (event.target === selectedGraphDialog) {
    selectedGraphDialog.close();
  }
});

closeSelectedNbtButton.addEventListener('click', () => {
  selectedNbtDialog.close();
});

selectedNbtDialog.addEventListener('click', event => {
  if (event.target === selectedNbtDialog) {
    selectedNbtDialog.close();
  }
});

closeArtifactButton.addEventListener('click', () => {
  artifactDialog.close();
});

irColorMappingInput.addEventListener('change', () => {
  irComparison.classList.toggle('ir-color-mapping-disabled', !irColorMappingInput.checked);
});

artifactDialog.addEventListener('click', event => {
  if (event.target === artifactDialog) artifactDialog.close();
});

selectedGraphWorldTab.addEventListener('click', () => {
  selectedGraphTab = 'world';
  void renderSelectedGraphTab();
});

selectedGraphLogicTab.addEventListener('click', () => {
  selectedGraphTab = 'logic';
  void renderSelectedGraphTab();
});

selectedGraphWorldRawButton.addEventListener('click', () => {
  selectedGraphWorldModeValue = 'raw';
  void renderSelectedGraphTab();
});

selectedGraphWorldFoldedButton.addEventListener('click', () => {
  selectedGraphWorldModeValue = 'folded';
  void renderSelectedGraphTab();
});

selectedGraphLogicRawButton.addEventListener('click', () => {
  selectedGraphLogicModeValue = 'raw';
  void renderSelectedGraphTab();
});

selectedGraphLogicSimplifiedButton.addEventListener('click', () => {
  selectedGraphLogicModeValue = 'simplified';
  void renderSelectedGraphTab();
});

selectedGraphHighLevelInput.addEventListener('change', () => {
  selectedGraphHighLevelLogic = selectedGraphHighLevelInput.checked;
  void renderSelectedGraphTab();
});

selectedGraphShowTagsInput.addEventListener('change', () => {
  selectedGraphShowTags = selectedGraphShowTagsInput.checked;
  void renderSelectedGraphTab();
});

selectedGraphZoomOutButton.addEventListener('click', () => {
  setSelectedGraphZoom(selectedGraphZoom - 0.25);
});

selectedGraphZoomResetButton.addEventListener('click', () => {
  setSelectedGraphZoom(1);
});

selectedGraphZoomInButton.addEventListener('click', () => {
  setSelectedGraphZoom(selectedGraphZoom + 0.25);
});

selectedGraphOutput.addEventListener(
  'wheel',
  event => {
    if (!event.ctrlKey) return;

    event.preventDefault();
    zoomSelectedGraphAt(event.clientX, event.clientY, selectedGraphZoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12));
  },
  { passive: false },
);

graphOutput.addEventListener('scroll', () => {
  updateGraphMinimapViewport();
});

graphOutput.addEventListener(
  'wheel',
  event => {
    if (!event.ctrlKey) return;

    event.preventDefault();
    zoomGraphAt(event.clientX, event.clientY, graphZoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12));
  },
  { passive: false },
);

graphMinimap.addEventListener('pointerdown', event => {
  if (graphMinimap.classList.contains('hidden')) return;

  isDraggingGraphMinimap = true;
  graphMinimap.setPointerCapture(event.pointerId);
  scrollGraphFromMinimap(event);
});

graphMinimap.addEventListener('pointermove', event => {
  if (!isDraggingGraphMinimap) return;

  scrollGraphFromMinimap(event);
});

graphMinimap.addEventListener('pointerup', event => {
  isDraggingGraphMinimap = false;
  graphMinimap.releasePointerCapture(event.pointerId);
});

graphMinimap.addEventListener('pointercancel', event => {
  isDraggingGraphMinimap = false;
  graphMinimap.releasePointerCapture(event.pointerId);
});

function setTraceExpanded(expanded: boolean): void {
  if (expanded && traceExpandButton.disabled) return;

  isTraceExpanded = expanded;
  if (expanded) {
    tracePanel.open = true;
  }
  tracePanel.classList.toggle('expanded', expanded);
  traceContent.classList.toggle('trace-content-expanded', expanded);
  inspectorPanel.classList.toggle('hidden-by-trace', expanded);
  traceExpandButton.textContent = expanded ? 'Collapse' : 'Expand';
  traceExpandButton.setAttribute('aria-label', expanded ? 'Collapse trace' : 'Expand trace');
  traceExpandButton.title = expanded ? 'Collapse trace' : 'Expand trace';
  const selectedIndex = Number(traceCycleInput.value);
  renderWaveform(selectedIndex);
  scrollWaveformToTraceIndex(selectedIndex);
  window.requestAnimationFrame(() => scrollWaveformToTraceIndex(selectedIndex));
}

function updateTraceExpandAvailability(hasTraceContent: boolean): void {
  traceExpandButton.disabled = !hasTraceContent;
  if (!hasTraceContent) {
    setTraceExpanded(false);
    traceExpandButton.title = 'Run a simulation before expanding trace';
    return;
  }
  traceExpandButton.title = isTraceExpanded ? 'Collapse trace' : 'Expand trace';
}

async function openGraphDialog(): Promise<void> {
  if (!graphDialog.open) graphDialog.showModal();

  if (!currentNbtBytes) {
    graphDot = undefined;
    graphOutput.replaceChildren();
    graphStatus.textContent = 'Open an NBT file before viewing graphs.';
    updateGraphTabs();
    return;
  }

  if (!graphDot) {
    graphOutput.replaceChildren();
    graphStatus.textContent = 'Generating graphs...';
    try {
      graphDot = await NbtSimulation.graphDot(currentNbtBytes, currentOutputMetadataJson);
    } catch (error) {
      graphStatus.textContent = error instanceof Error ? error.message : String(error);
      return;
    }
  }

  await renderGraphTab();
}

async function setGraphTab(nextTab: GraphTab): Promise<void> {
  graphTab = nextTab;
  updateGraphTabs();
  await renderGraphTab();
}

async function setGraphWorldMode(nextMode: GraphWorldMode): Promise<void> {
  graphWorldModeValue = nextMode;
  updateGraphTabs();
  if (graphTab === 'world') {
    await renderGraphTab();
  }
}

async function setGraphLogicMode(nextMode: GraphLogicMode): Promise<void> {
  graphLogicModeValue = nextMode;
  updateGraphTabs();
  if (graphTab === 'logic') {
    await renderGraphTab();
  }
}

async function renderGraphTab(): Promise<void> {
  updateGraphTabs();
  graphOutput.replaceChildren();
  clearGraphMinimap();
  graphSelectionActions.classList.add('hidden');
  graphViewer.classList.remove('has-selection-action');

  if (!graphDot) {
    graphStatus.textContent = currentNbtBytes ? 'Generating graphs...' : 'Open an NBT file before viewing graphs.';
    return;
  }

  graphStatus.textContent =
    graphTab === 'logic'
      ? graphLogicStatusTitle()
      : graphWorldModeValue === 'folded'
        ? 'World Graph - Folded'
        : 'World Graph - Raw';

  try {
    const dot = currentGraphDot();
    const viz = await loadViz();
    const svg = viz.renderSVGElement(dot, { engine: 'dot' });
    selectedGraphNode = undefined;
    graphOutput.append(svg);
    installGraphNodeHitAreas(svg);
    bindGraphSelection(svg);
    setGraphZoom(1);
    renderGraphMinimap(svg);
  } catch (error) {
    graphStatus.textContent = error instanceof Error ? error.message : String(error);
  }
}

function currentGraphDot(): string {
  if (!graphDot) return '';

  if (graphTab === 'logic') {
    if (graphLogicModeValue === 'simplified') {
      if (graphHighLevelLogic) {
        return graphShowTags ? graphDot.highLevelLogicDot : graphDot.highLevelLogicDotWithoutTags;
      }

      return graphShowTags ? graphDot.simplifiedLogicDot : graphDot.simplifiedLogicDotWithoutTags;
    }

    return graphShowTags ? graphDot.logicDot : graphDot.logicDotWithoutTags;
  }

  if (graphWorldModeValue === 'folded') {
    return graphShowTags ? graphDot.foldedWorldDot : graphDot.foldedWorldDotWithoutTags;
  }

  return graphShowTags ? graphDot.rawWorldDot : graphDot.rawWorldDotWithoutTags;
}

function installGraphNodeHitAreas(svg: SVGSVGElement): void {
  svg.querySelectorAll<SVGGElement>('g.node').forEach(node => {
    node.querySelector(':scope > .graph-node-hit-area')?.remove();
    const bbox = node.getBBox();
    if (!bbox) return;

    const hitArea = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    hitArea.classList.add('graph-node-hit-area');
    hitArea.setAttribute('x', String(bbox.x));
    hitArea.setAttribute('y', String(bbox.y));
    hitArea.setAttribute('width', String(bbox.width));
    hitArea.setAttribute('height', String(bbox.height));
    node.prepend(hitArea);
  });
}

function bindGraphSelection(svg: SVGSVGElement): void {
  svg.addEventListener('click', event => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    const node = target.closest<SVGGElement>('g.node');
    if (!node || !svg.contains(node)) {
      selectedGraphNode = undefined;
      applyGraphSelection(svg);
      updateGraphSelectionStatus();
      return;
    }

    const nodeId = graphElementTitle(node);
    if (!nodeId) return;
    selectedGraphNode = selectedGraphNode === nodeId ? undefined : nodeId;
    applyGraphSelection(svg);
    updateGraphSelectionStatus();
  });
}

function applyGraphSelection(svg: SVGSVGElement): void {
  const nodeElements = graphNodeElements(svg);
  const edgeElements = graphEdgeElements(svg);
  const selectedNodes = new Set<string>();
  const connectedEdges = new Set<SVGGElement>();

  if (selectedGraphNode) {
    const directionalSelection = collectDirectionalGraphSelection(selectedGraphNode, edgeElements);
    directionalSelection.nodes.forEach(nodeId => selectedNodes.add(nodeId));
    directionalSelection.edges.forEach(edge => connectedEdges.add(edge));
  }

  svg.classList.toggle('graph-has-selection', Boolean(selectedGraphNode));
  for (const [nodeId, node] of nodeElements) {
    node.classList.toggle('graph-node-selected', selectedNodes.has(nodeId));
    node.classList.toggle('graph-node-root', selectedGraphNode === nodeId);
    node.classList.toggle('graph-node-dimmed', Boolean(selectedGraphNode) && !selectedNodes.has(nodeId));
  }

  for (const edge of edgeElements) {
    edge.element.classList.toggle('graph-edge-connected', connectedEdges.has(edge.element));
    edge.element.classList.toggle('graph-edge-dimmed', Boolean(selectedGraphNode) && !connectedEdges.has(edge.element));
  }
}

function collectDirectionalGraphSelection(root: string, edges: GraphEdgeInfo[]): { nodes: Set<string>; edges: Set<SVGGElement> } {
  const nodes = new Set<string>([root]);
  const selectedEdges = new Set<SVGGElement>();

  collectGraphCone(root, edges, 'incoming', nodes, selectedEdges);
  collectGraphCone(root, edges, 'outgoing', nodes, selectedEdges);

  return { nodes, edges: selectedEdges };
}

function collectGraphCone(
  root: string,
  edges: GraphEdgeInfo[],
  direction: 'incoming' | 'outgoing',
  nodes: Set<string>,
  selectedEdges: Set<SVGGElement>,
): void {
  const visited = new Set<string>();
  const queue = [root];

  while (queue.length > 0) {
    const nodeId = queue.pop()!;
    if (!visited.add(nodeId)) continue;

    for (const edge of edges) {
      const next = direction === 'incoming' && edge.target === nodeId ? edge.source : direction === 'outgoing' && edge.source === nodeId ? edge.target : undefined;
      if (!next) continue;

      nodes.add(next);
      selectedEdges.add(edge.element);
      if (!visited.has(next)) queue.push(next);
    }
  }
}

function updateGraphSelectionStatus(): void {
  const title =
    graphTab === 'logic'
      ? graphLogicStatusTitle()
      : graphWorldModeValue === 'folded'
        ? 'World Graph - Folded'
        : 'World Graph - Raw';
  const selectedCount = selectedGraphNode ? graphOutput.querySelectorAll('svg g.node.graph-node-selected').length : 0;
  graphStatus.textContent = selectedGraphNode ? `${title} - ${selectedGraphNode} selected (${selectedCount} nodes)` : title;
  const canOpenSelection = graphTab === 'world' && selectedCount > 0;
  graphSelectionActions.classList.toggle('hidden', !canOpenSelection);
  graphViewer.classList.toggle('has-selection-action', canOpenSelection);
}

async function openSelectedGraphView(): Promise<void> {
  const sourceSvg = graphOutput.querySelector<SVGSVGElement>('svg');
  if (graphTab !== 'world' || !sourceSvg || !selectedGraphNode) return;

  const nodeIds = selectedWorldGraphNodeIds(sourceSvg);
  if (nodeIds.length === 0 || !currentNbtBytes) return;

  if (!selectedGraphDialog.open) selectedGraphDialog.showModal();
  selectedGraphOutput.replaceChildren();
  selectedGraphStatus.textContent = 'Generating selected graph...';
  selectedGraphDot = undefined;
  selectedGraphTab = 'world';
  selectedGraphWorldModeValue = graphWorldModeValue;
  selectedGraphLogicModeValue = graphLogicModeValue;
  selectedGraphHighLevelLogic = graphHighLevelLogic;
  selectedGraphShowTags = graphShowTags;
  selectedGraphHighLevelInput.checked = selectedGraphHighLevelLogic;
  selectedGraphShowTagsInput.checked = selectedGraphShowTags;

  try {
    selectedGraphDot = await NbtSimulation.selectedGraphDot(currentNbtBytes, graphWorldModeValue === 'folded', nodeIds);
    await renderSelectedGraphTab();
  } catch (error) {
    selectedGraphStatus.textContent = error instanceof Error ? error.message : String(error);
  }
}

async function openSelectedNbtView(): Promise<void> {
  const sourceSvg = graphOutput.querySelector<SVGSVGElement>('svg');
  if (graphTab !== 'world' || !sourceSvg || !selectedGraphNode) return;

  const nodeIds = selectedWorldGraphNodeIds(sourceSvg);
  if (nodeIds.length === 0 || !currentNbtBytes) return;

  selectedNbtTitle.textContent = 'Selected NBT';
  if (!selectedNbtDialog.open) selectedNbtDialog.showModal();
  selectedNbtStatus.textContent = 'Generating selected NBT...';

  try {
    const selectedRoot = await NbtSimulation.selectedNbt(currentNbtBytes, graphWorldModeValue === 'folded', nodeIds);
    const structure = toStructureModel(selectedRoot);
    if (!structure) {
      selectedNbtStatus.textContent = 'Selected graph did not produce a structure.';
      return;
    }

    await selectedNbtViewer.setStructure(structure);
    selectedNbtStatus.textContent = `Selected NBT - ${structure.blocks.length} blocks`;
  } catch (error) {
    selectedNbtStatus.textContent = error instanceof Error ? error.message : String(error);
  }
}

function selectedWorldGraphNodeIds(sourceSvg: SVGSVGElement): number[] {
  return Array.from(sourceSvg.querySelectorAll<SVGGElement>('g.node.graph-node-selected'))
    .map(node => parseGraphNodeId(graphElementTitle(node)))
    .filter((nodeId): nodeId is number => nodeId !== undefined);
}

function graphNodeElements(svg: SVGSVGElement): Map<string, SVGGElement> {
  const nodes = new Map<string, SVGGElement>();
  svg.querySelectorAll<SVGGElement>('g.node').forEach(node => {
    const nodeId = graphElementTitle(node);
    if (nodeId) nodes.set(nodeId, node);
  });
  return nodes;
}

function graphEdgeElements(svg: SVGSVGElement): GraphEdgeInfo[] {
  return Array.from(svg.querySelectorAll<SVGGElement>('g.edge')).flatMap(edge => {
    const parsed = parseGraphEdgeTitle(graphElementTitle(edge));
    return parsed ? [{ element: edge, ...parsed }] : [];
  });
}

function graphElementTitle(element: Element): string | undefined {
  return element.querySelector(':scope > title')?.textContent?.trim() || undefined;
}

function parseGraphEdgeTitle(title: string | undefined): { source: string; target: string } | undefined {
  const match = title?.match(/^(node\d+)(?::[^-]+)?->(node\d+)(?::.+)?$/);
  if (!match) return undefined;
  return { source: match[1], target: match[2] };
}

function parseGraphNodeId(title: string | undefined): number | undefined {
  const match = title?.match(/^node(\d+)$/);
  if (!match) return undefined;

  const nodeId = Number(match[1]);
  return Number.isInteger(nodeId) ? nodeId : undefined;
}

async function renderSelectedGraphTab(): Promise<void> {
  updateSelectedGraphTabs();
  selectedGraphOutput.replaceChildren();

  if (!selectedGraphDot) {
    selectedGraphStatus.textContent = 'Select world graph nodes to open a focused graph.';
    return;
  }

  selectedGraphStatus.textContent =
    selectedGraphTab === 'logic'
      ? selectedGraphLogicStatusTitle()
      : selectedGraphWorldModeValue === 'folded'
        ? 'Selected World Graph - Folded'
        : 'Selected World Graph - Raw';

  try {
    const viz = await loadViz();
    const svg = viz.renderSVGElement(currentSelectedGraphDot(), { engine: 'dot' });
    selectedGraphOutput.append(svg);
    setSelectedGraphZoom(1);
  } catch (error) {
    selectedGraphStatus.textContent = error instanceof Error ? error.message : String(error);
  }
}

function currentSelectedGraphDot(): string {
  if (!selectedGraphDot) return '';

  if (selectedGraphTab === 'logic') {
    if (selectedGraphLogicModeValue === 'simplified') {
      if (selectedGraphHighLevelLogic) {
        return selectedGraphShowTags ? selectedGraphDot.highLevelLogicDot : selectedGraphDot.highLevelLogicDotWithoutTags;
      }

      return selectedGraphShowTags ? selectedGraphDot.simplifiedLogicDot : selectedGraphDot.simplifiedLogicDotWithoutTags;
    }

    return selectedGraphShowTags ? selectedGraphDot.logicDot : selectedGraphDot.logicDotWithoutTags;
  }

  if (selectedGraphWorldModeValue === 'folded') {
    return selectedGraphShowTags ? selectedGraphDot.foldedWorldDot : selectedGraphDot.foldedWorldDotWithoutTags;
  }

  return selectedGraphShowTags ? selectedGraphDot.rawWorldDot : selectedGraphDot.rawWorldDotWithoutTags;
}

function updateSelectedGraphTabs(): void {
  selectedGraphWorldTab.classList.toggle('active', selectedGraphTab === 'world');
  selectedGraphLogicTab.classList.toggle('active', selectedGraphTab === 'logic');
  selectedGraphWorldMode.classList.toggle('hidden', selectedGraphTab !== 'world');
  selectedGraphLogicMode.classList.toggle('hidden', selectedGraphTab !== 'logic');
  selectedGraphHighLevelToggle.classList.toggle(
    'hidden',
    selectedGraphTab !== 'logic' || selectedGraphLogicModeValue !== 'simplified',
  );
  selectedGraphWorldRawButton.classList.toggle('active', selectedGraphWorldModeValue === 'raw');
  selectedGraphWorldFoldedButton.classList.toggle('active', selectedGraphWorldModeValue === 'folded');
  selectedGraphLogicRawButton.classList.toggle('active', selectedGraphLogicModeValue === 'raw');
  selectedGraphLogicSimplifiedButton.classList.toggle('active', selectedGraphLogicModeValue === 'simplified');
  selectedGraphHighLevelInput.checked = selectedGraphHighLevelLogic;
}

function graphLogicStatusTitle(): string {
  if (graphLogicModeValue === 'raw') {
    return 'Logic Graph - Raw';
  }

  return graphHighLevelLogic ? 'Logic Graph - Simplified + High-level' : 'Logic Graph - Simplified';
}

function selectedGraphLogicStatusTitle(): string {
  if (selectedGraphLogicModeValue === 'raw') {
    return 'Selected Logic Graph - Raw';
  }

  return selectedGraphHighLevelLogic ? 'Selected Logic Graph - Simplified + High-level' : 'Selected Logic Graph - Simplified';
}

async function loadViz(): Promise<Viz> {
  vizPromise ??= import('@viz-js/viz').then(module => module.instance());
  return vizPromise;
}

function setGraphZoom(nextZoom: number): void {
  graphZoom = Math.max(0.25, Math.min(3, nextZoom));
  applyGraphZoom({ refreshMinimap: true });
}

function zoomGraphAt(clientX: number, clientY: number, nextZoom: number): void {
  const previousZoom = graphZoom;
  const clampedZoom = Math.max(0.25, Math.min(3, nextZoom));
  if (clampedZoom === previousZoom) return;

  const outputRect = graphOutput.getBoundingClientRect();
  const focusX = graphOutput.scrollLeft + clientX - outputRect.left;
  const focusY = graphOutput.scrollTop + clientY - outputRect.top;
  graphZoom = clampedZoom;
  applyGraphZoom({ refreshMinimap: true });

  const ratio = clampedZoom / previousZoom;
  graphOutput.scrollLeft = focusX * ratio - (clientX - outputRect.left);
  graphOutput.scrollTop = focusY * ratio - (clientY - outputRect.top);
}

function applyGraphZoom(options: { refreshMinimap?: boolean } = {}): void {
  graphZoomResetButton.textContent = `${Math.round(graphZoom * 100)}%`;
  graphZoomOutButton.disabled = graphZoom <= 0.25;
  graphZoomInButton.disabled = graphZoom >= 3;

  const svg = graphOutput.querySelector<SVGSVGElement>('svg');
  if (svg) {
    const baseWidth = readGraphBaseSize(svg, 'width');
    const baseHeight = readGraphBaseSize(svg, 'height');
    svg.style.width = `${baseWidth * graphZoom}px`;
    svg.style.height = `${baseHeight * graphZoom}px`;
    requestAnimationFrame(() => {
      if (options.refreshMinimap) {
        renderGraphMinimap(svg);
      } else {
        updateGraphMinimapViewport();
      }
    });
    return;
  }

}

function setSelectedGraphZoom(nextZoom: number): void {
  selectedGraphZoom = Math.max(0.25, Math.min(3, nextZoom));
  applySelectedGraphZoom();
}

function zoomSelectedGraphAt(clientX: number, clientY: number, nextZoom: number): void {
  const previousZoom = selectedGraphZoom;
  const clampedZoom = Math.max(0.25, Math.min(3, nextZoom));
  if (clampedZoom === previousZoom) return;

  const outputRect = selectedGraphOutput.getBoundingClientRect();
  const focusX = selectedGraphOutput.scrollLeft + clientX - outputRect.left;
  const focusY = selectedGraphOutput.scrollTop + clientY - outputRect.top;
  selectedGraphZoom = clampedZoom;
  applySelectedGraphZoom();

  const ratio = clampedZoom / previousZoom;
  selectedGraphOutput.scrollLeft = focusX * ratio - (clientX - outputRect.left);
  selectedGraphOutput.scrollTop = focusY * ratio - (clientY - outputRect.top);
}

function applySelectedGraphZoom(): void {
  selectedGraphZoomResetButton.textContent = `${Math.round(selectedGraphZoom * 100)}%`;
  selectedGraphZoomOutButton.disabled = selectedGraphZoom <= 0.25;
  selectedGraphZoomInButton.disabled = selectedGraphZoom >= 3;

  const svg = selectedGraphOutput.querySelector<SVGSVGElement>('svg');
  if (!svg) return;

  const baseWidth = readSvgBaseSize(svg, 'width', selectedGraphZoom, selectedGraphOutput);
  const baseHeight = readSvgBaseSize(svg, 'height', selectedGraphZoom, selectedGraphOutput);
  svg.style.width = `${baseWidth * selectedGraphZoom}px`;
  svg.style.height = `${baseHeight * selectedGraphZoom}px`;
}

function readGraphBaseSize(svg: SVGSVGElement, dimension: 'width' | 'height'): number {
  return readSvgBaseSize(svg, dimension, graphZoom, graphOutput);
}

function readSvgBaseSize(svg: SVGSVGElement, dimension: 'width' | 'height', zoom: number, fallbackElement: HTMLElement): number {
  const dataKey = `base${dimension[0].toUpperCase()}${dimension.slice(1)}`;
  const cached = Number(svg.dataset[dataKey]);
  if (Number.isFinite(cached) && cached > 0) return cached;

  const rect = svg.getBoundingClientRect();
  const measured = dimension === 'width' ? rect.width : rect.height;
  const fallback = dimension === 'width' ? fallbackElement.clientWidth : fallbackElement.clientHeight;
  const value = Math.max(measured / zoom, fallback, 1);
  svg.dataset[dataKey] = String(value);
  return value;
}

function renderGraphMinimap(svg: SVGSVGElement): void {
  graphMinimapContent.replaceChildren();
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.querySelectorAll('.graph-node-hit-area').forEach(hitArea => hitArea.remove());
  graphMinimapContent.append(clone);
  graphMinimap.classList.remove('hidden');

  requestAnimationFrame(() => {
    const graphRect = currentGraphRect(svg);
    if (!graphRect) {
      clearGraphMinimap();
      return;
    }

    sizeGraphMinimap(graphRect.width, graphRect.height);

    const minimapWidth = Math.max(graphMinimapContent.clientWidth, 1);
    const minimapHeight = Math.max(graphMinimapContent.clientHeight, 1);
    graphMinimapScale = Math.min(minimapWidth / graphRect.width, minimapHeight / graphRect.height);

    const fittedWidth = graphRect.width * graphMinimapScale;
    const fittedHeight = graphRect.height * graphMinimapScale;
    graphMinimap.style.width = `${fittedWidth + GRAPH_MINIMAP_INSET * 2 + 2}px`;
    graphMinimap.style.height = `${fittedHeight + GRAPH_MINIMAP_INSET * 2 + 2}px`;
    clone.style.width = `${fittedWidth}px`;
    clone.style.height = `${fittedHeight}px`;
    updateGraphMinimapViewport();
  });
}

function updateGraphMinimapViewport(): void {
  if (graphMinimap.classList.contains('hidden')) return;
  const svg = graphOutput.querySelector<SVGSVGElement>('svg');
  const graphRect = svg ? currentGraphRect(svg) : undefined;
  if (!graphRect) return;

  const visibleLeft = Math.max(0, graphOutput.scrollLeft - graphRect.left);
  const visibleTop = Math.max(0, graphOutput.scrollTop - graphRect.top);
  const visibleRight = Math.min(graphRect.width, graphOutput.scrollLeft + graphOutput.clientWidth - graphRect.left);
  const visibleBottom = Math.min(graphRect.height, graphOutput.scrollTop + graphOutput.clientHeight - graphRect.top);
  const visibleWidth = Math.max(0, visibleRight - visibleLeft);
  const visibleHeight = Math.max(0, visibleBottom - visibleTop);

  graphMinimapViewport.style.width = `${visibleWidth * graphMinimapScale}px`;
  graphMinimapViewport.style.height = `${visibleHeight * graphMinimapScale}px`;
  graphMinimapViewport.style.transform = `translate(${visibleLeft * graphMinimapScale}px, ${
    visibleTop * graphMinimapScale
  }px)`;
}

function scrollGraphFromMinimap(event: PointerEvent): void {
  event.preventDefault();
  if (graphMinimapScale <= 0) return;
  const svg = graphOutput.querySelector<SVGSVGElement>('svg');
  const graphRect = svg ? currentGraphRect(svg) : undefined;
  if (!graphRect) return;

  const minimapContentBox = graphMinimapContent.getBoundingClientRect();
  const x = Math.max(0, Math.min(minimapContentBox.width, event.clientX - minimapContentBox.left));
  const y = Math.max(0, Math.min(minimapContentBox.height, event.clientY - minimapContentBox.top));
  graphOutput.scrollLeft = graphRect.left + x / graphMinimapScale - graphOutput.clientWidth / 2;
  graphOutput.scrollTop = graphRect.top + y / graphMinimapScale - graphOutput.clientHeight / 2;
  updateGraphMinimapViewport();
}

function clearGraphMinimap(): void {
  graphMinimap.classList.add('hidden');
  graphMinimapContent.replaceChildren();
  graphMinimapViewport.removeAttribute('style');
  graphMinimap.removeAttribute('style');
  graphMinimapScale = 1;
  isDraggingGraphMinimap = false;
}

function currentGraphRect(svg: SVGSVGElement): { left: number; top: number; width: number; height: number } | undefined {
  const width = readGraphBaseSize(svg, 'width') * graphZoom;
  const height = readGraphBaseSize(svg, 'height') * graphZoom;
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return undefined;
  }

  const outputRect = graphOutput.getBoundingClientRect();
  const svgRect = svg.getBoundingClientRect();

  return {
    left: svgRect.left - outputRect.left + graphOutput.scrollLeft,
    top: svgRect.top - outputRect.top + graphOutput.scrollTop,
    width,
    height,
  };
}

function sizeGraphMinimap(contentWidth: number, contentHeight: number): void {
  const aspect = Math.max(contentWidth / Math.max(contentHeight, 1), 0.1);
  const maxContentWidth = Math.min(GRAPH_MINIMAP_MAX_WIDTH, Math.max(GRAPH_MINIMAP_MIN_WIDTH, graphOutput.clientWidth * 0.28));
  const maxContentHeight = Math.min(
    GRAPH_MINIMAP_MAX_HEIGHT,
    Math.max(GRAPH_MINIMAP_MIN_HEIGHT, graphOutput.clientHeight * 0.28),
  );

  let minimapContentWidth = maxContentWidth;
  let minimapContentHeight = minimapContentWidth / aspect;
  if (minimapContentHeight > maxContentHeight) {
    minimapContentHeight = maxContentHeight;
    minimapContentWidth = minimapContentHeight * aspect;
  }

  graphMinimap.style.width = `${Math.max(GRAPH_MINIMAP_MIN_WIDTH, minimapContentWidth) + GRAPH_MINIMAP_INSET * 2}px`;
  graphMinimap.style.height = `${Math.max(GRAPH_MINIMAP_MIN_HEIGHT, minimapContentHeight) + GRAPH_MINIMAP_INSET * 2}px`;
}

function updateGraphTabs(): void {
  const tabs: Array<[HTMLButtonElement, GraphTab]> = [
    [graphWorldTab, 'world'],
    [graphLogicTab, 'logic'],
  ];

  for (const [button, tab] of tabs) {
    const active = tab === graphTab;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  }

  graphWorldMode.classList.toggle('hidden', graphTab !== 'world');
  graphLogicMode.classList.toggle('hidden', graphTab !== 'logic');
  graphHighLevelToggle.classList.toggle('hidden', graphTab !== 'logic' || graphLogicModeValue !== 'simplified');
  graphWorldRawButton.classList.toggle('active', graphWorldModeValue === 'raw');
  graphWorldFoldedButton.classList.toggle('active', graphWorldModeValue === 'folded');
  graphLogicRawButton.classList.toggle('active', graphLogicModeValue === 'raw');
  graphLogicSimplifiedButton.classList.toggle('active', graphLogicModeValue === 'simplified');
  graphHighLevelInput.checked = graphHighLevelLogic;
  graphShowTagsInput.checked = graphShowTags;
}

async function toggleSelectedSwitch(): Promise<void> {
  if (!selectedBlock || !currentNbtBytes) return;

  await toggleSwitchBlock(selectedBlock);
}

async function toggleSwitchBlock(block: StructureBlock): Promise<void> {
  if (!currentNbtBytes) return;

  const activeSimulation = await ensureSimulation();

  const selectedPos = block.pos;
  const baseRoot = currentRoot;
  const nextRoot = activeSimulation.toggleSwitch(block);
  if (!nextRoot) return;

  await applySimulatedRoot(activeSimulation, nextRoot, baseRoot, selectedPos);
}

async function setAllSwitches(isOn: boolean): Promise<void> {
  if (!currentNbtBytes) return;

  const structure = toStructureModel(currentRoot);
  const switches = structure?.blocks.filter(block => block.palette.name === 'minecraft:lever') ?? [];
  if (switches.length === 0) return;

  const activeSimulation = await ensureSimulation();

  const selectedPos = selectedBlock?.pos;
  const baseRoot = currentRoot;
  let nextRoot: unknown | undefined;

  for (const block of switches) {
    nextRoot = activeSimulation.setSwitch(block, isOn) ?? nextRoot;
  }

  if (!nextRoot) return;

  await applySimulatedRoot(activeSimulation, nextRoot, baseRoot, selectedPos);
}

async function ensureSimulation(): Promise<NbtSimulation> {
  if (!currentNbtBytes) throw new Error('Open an NBT file before simulating switches.');

  simulation ??= await NbtSimulation.create(currentNbtBytes, traceSimulationEnabled);
  simulation.setTraceEnabled(traceSimulationEnabled);
  return simulation;
}

async function setTraceSimulationEnabled(enabled: boolean): Promise<void> {
  traceSimulationEnabled = enabled;
  traceSimulationEnabledInput.checked = enabled;
  traceSimulationToggle.classList.toggle('active', enabled);
  traceSimulationState.textContent = enabled ? 'On' : 'Off';
  simulation?.setTraceEnabled(enabled);

  selectedWaveformSignal = undefined;
  renderTrace([], [], emptyWaveform, undefined);
  await restoreCurrentStructurePreview();
  viewer.setTraceHighlights([]);
}

async function applySimulatedRoot(
  activeSimulation: NbtSimulation,
  nextRoot: unknown,
  baseRoot: unknown,
  selectedPos: StructureBlock['pos'] | undefined,
): Promise<void> {
  currentRoot = mergeSimulatedState(currentRoot, nextRoot);
  const structure = toStructureModel(currentRoot);
  if (!structure) throw new Error('Simulator returned a structure that the viewer could not render.');

  await viewer.setStructure(structure, { preserveSelection: true, preserveView: true });
  const nextSelectedBlock = selectedPos ? structure.blocks.find(block => samePos(block.pos, selectedPos)) : undefined;
  viewer.setSelectedBlock(nextSelectedBlock);
  renderSelection(nextSelectedBlock);
  renderSwitches(structure);
  if (traceSimulationEnabled) {
    renderTrace(activeSimulation.trace(), activeSimulation.snapshots(), activeSimulation.waveform(), baseRoot, {
      animateTo: 'last',
      history: {
        trace: activeSimulation.historyTrace(),
        snapshots: activeSimulation.historySnapshots(),
        waveform: activeSimulation.historyWaveform(),
      },
    });
  } else {
    renderTrace([], [], emptyWaveform, undefined);
  }
}

dropZone.addEventListener('dragover', event => {
  event.preventDefault();
  dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));

dropZone.addEventListener('drop', event => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
  void handleDrop(event).catch(error => {
    viewerEmpty.classList.remove('hidden');
    inspector.textContent = error instanceof Error ? error.message : String(error);
  });
});

async function handleDrop(event: DragEvent): Promise<void> {
  const dropped = await collectDroppedFiles(event.dataTransfer);
  if (dropped.containsDirectory || dropped.files.length > 1) {
    if (!(await tryOpenSnapshot(dropped.files))) renderFileBrowser(dropped.files);
  } else if (dropped.files[0]) {
    leaveSnapshotMode();
    void openFile(dropped.files[0]);
  }
}

async function collectDroppedFiles(dataTransfer: DataTransfer | null): Promise<{
  containsDirectory: boolean;
  files: File[];
}> {
  if (!dataTransfer) return { containsDirectory: false, files: [] };

  const fallbackFiles = Array.from(dataTransfer.files ?? []);
  const items = Array.from(dataTransfer.items ?? []);

  try {
    const entries = items
      .map(getDroppedEntry)
      .filter((entry): entry is DroppedFileSystemEntry => Boolean(entry));

    if (entries.length === 0) {
      return { containsDirectory: false, files: fallbackFiles };
    }

    const files = await Promise.all(entries.map(entry => readDroppedEntry(entry)));
    return {
      containsDirectory: entries.some(entry => entry.isDirectory),
      files: files.flat(),
    };
  } catch (error) {
    console.warn('Falling back to dropped files after directory read failed.', error);
  }

  try {
    const handleDrop = await collectDroppedFileSystemHandles(items);
    if (handleDrop.files.length > 0) return handleDrop;
  } catch (error) {
    console.warn('File system handle drop failed.', error);
  }

  return { containsDirectory: false, files: fallbackFiles };
}

async function collectDroppedFileSystemHandles(items: DataTransferItem[]): Promise<{
  containsDirectory: boolean;
  files: File[];
}> {
  const handlePromises = items
    .map(item => {
      const itemWithHandle = item as unknown as {
        getAsFileSystemHandle?: () => Promise<DroppedFileSystemHandle | null>;
      };
      return itemWithHandle.getAsFileSystemHandle?.call(item);
    })
    .filter((promise): promise is Promise<DroppedFileSystemHandle | null> => Boolean(promise));

  if (handlePromises.length === 0) return { containsDirectory: false, files: [] };

  const handles = (await Promise.all(handlePromises)).filter(
    (handle): handle is DroppedFileSystemHandle => Boolean(handle),
  );
  const files = await Promise.all(handles.map(handle => readDroppedHandle(handle)));
  return {
    containsDirectory: handles.some(handle => handle.kind === 'directory'),
    files: files.flat(),
  };
}

async function readDroppedHandle(handle: DroppedFileSystemHandle, parentPath = ''): Promise<File[]> {
  const path = parentPath ? `${parentPath}/${handle.name}` : handle.name;

  if (handle.kind === 'file') {
    const file = await handle.getFile();
    setDroppedFilePath(file, path);
    return [file];
  }

  const files: File[] = [];
  for await (const child of handle.values()) {
    files.push(...(await readDroppedHandle(child, path)));
  }
  return files;
}

function getDroppedEntry(item: DataTransferItem): DroppedFileSystemEntry | null {
  const itemWithEntry = item as unknown as {
    webkitGetAsEntry?: () => DroppedFileSystemEntry | null;
  };
  return itemWithEntry.webkitGetAsEntry?.call(item) ?? null;
}

async function readDroppedEntry(entry: DroppedFileSystemEntry, parentPath = ''): Promise<File[]> {
  const path = parentPath ? `${parentPath}/${entry.name}` : entry.name;

  if (entry.isFile) {
    const file = await readDroppedFile(entry as DroppedFileSystemFileEntry);
    setDroppedFilePath(file, path);
    return [file];
  }

  if (!entry.isDirectory) return [];

  const children = await readDroppedDirectory(entry as DroppedFileSystemDirectoryEntry);
  const files = await Promise.all(children.map(child => readDroppedEntry(child, path)));
  return files.flat();
}

function readDroppedFile(entry: DroppedFileSystemFileEntry): Promise<File> {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

async function readDroppedDirectory(entry: DroppedFileSystemDirectoryEntry): Promise<DroppedFileSystemEntry[]> {
  const reader = entry.createReader();
  const entries: DroppedFileSystemEntry[] = [];

  while (true) {
    const batch = await new Promise<DroppedFileSystemEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (batch.length === 0) return entries;
    entries.push(...batch);
  }
}

function setDroppedFilePath(file: File, path: string): void {
  try {
    Object.defineProperty(file, 'webkitRelativePath', {
      configurable: true,
      value: path,
    });
  } catch {
    // Some browser File objects are not extensible. Falling back to file.name is fine.
  }
}

async function openSnapshot(files: File[]): Promise<void> {
  if (!(await tryOpenSnapshot(files))) {
    throw new Error('This input does not contain a redstone-compiler snapshot manifest.');
  }
}

async function tryOpenSnapshot(files: File[]): Promise<boolean> {
  const snapshot = await parseSnapshot(await expandSnapshotFiles(files));
  if (!snapshot) return false;

  currentSnapshot = snapshot;
  currentSnapshotPath = undefined;
  snapshotBoxesVisible = true;
  snapshotRoutesVisible = true;
  renderSnapshotBrowser(snapshot);
  updateSnapshotBoxes();
  updateSnapshotRoutes();

  const finalPath = snapshot.manifest.final_nbt;
  if (finalPath && snapshot.filesByPath.has(finalPath)) {
    await openSnapshotNbt(finalPath, findSnapshotEntry(finalPath));
  } else {
    viewerEmpty.classList.remove('hidden');
    inspector.textContent = `Snapshot ${snapshot.manifest.status}: no final NBT was emitted.`;
  }
  return true;
}

async function expandSnapshotFiles(files: File[]): Promise<File[]> {
  const expanded = files.filter(file => !isSnapshotArchive(file));
  for (const archive of files.filter(isSnapshotArchive)) {
    expanded.push(...await unpackSnapshotArchive(archive));
  }
  return expanded;
}

function isSnapshotArchive(file: File): boolean {
  return file.name.toLowerCase().endsWith('.rsnap');
}

async function unpackSnapshotArchive(archive: File): Promise<File[]> {
  const entries = Object.entries(unzipSync(new Uint8Array(await archive.arrayBuffer())))
    .filter(([path]) => !path.endsWith('/'))
    .sort(([left], [right]) => left.localeCompare(right));
  if (entries.length > 4096) throw new Error('Snapshot archive contains too many files.');

  let totalSize = 0;
  const files: File[] = [];
  for (const [rawPath, bytes] of entries) {
    const path = normalizePath(rawPath);
    if (!isSafeArchivePath(path)) throw new Error(`Snapshot archive contains an unsafe path: ${rawPath}`);
    totalSize += bytes.byteLength;
    if (totalSize > 256 * 1024 * 1024) throw new Error('Snapshot archive expands beyond 256 MiB.');

    const name = path.split('/').pop() ?? path;
    const file = new File([bytes], name, { type: 'application/octet-stream' });
    setDroppedFilePath(file, path);
    files.push(file);
  }
  return files;
}

function isSafeArchivePath(path: string): boolean {
  return path.length > 0
    && !path.startsWith('/')
    && !/^[a-z]:/i.test(path)
    && path.split('/').every(segment => segment.length > 0 && segment !== '.' && segment !== '..');
}

async function parseSnapshot(files: File[]): Promise<LoadedSnapshot | undefined> {
  for (const manifestFile of files.filter(file => /(^|\/)manifest\.json$/i.test(normalizePath(getDisplayPath(file))))) {
    let manifest: SnapshotManifest;
    try {
      manifest = JSON.parse(await manifestFile.text()) as SnapshotManifest;
    } catch {
      continue;
    }
    if (manifest.format !== 'redstone-compiler.snapshot.v1' || !Array.isArray(manifest.artifacts)) continue;

    const manifestPath = normalizePath(getDisplayPath(manifestFile));
    const rootPrefix = manifestPath.slice(0, -'manifest.json'.length);
    const filesByPath = new Map<string, File>();
    for (const file of files) {
      const path = normalizePath(getDisplayPath(file));
      if (!path.startsWith(rootPrefix)) continue;
      filesByPath.set(path.slice(rootPrefix.length), file);
    }

    const interfaceJson = await filesByPath.get('interface.json')?.text();
    let sourceMap: SnapshotSourceMap | undefined;
    const sourceMapFile = filesByPath.get('ir/source-map.json');
    if (sourceMapFile) {
      try {
        const parsed = JSON.parse(await sourceMapFile.text()) as SnapshotSourceMap;
        if (parsed.format === 'redstone-compiler.source-map.v1'
          && Array.isArray(parsed.locations)
          && parsed.documents && typeof parsed.documents === 'object') {
          sourceMap = parsed;
        }
      } catch (error) {
        console.warn('Skipping invalid IR source map.', error);
      }
    }
    const instances: SnapshotInstance[] = [];
    for (const artifact of manifest.artifacts) {
      if (!/^instances\/[^/]+\/instance\.json$/i.test(artifact.path)) continue;
      const file = filesByPath.get(artifact.path);
      if (!file) continue;
      try {
        const instance = parseSnapshotInstance(JSON.parse(await file.text()), artifact.path);
        if (instance) instances.push(instance);
      } catch (error) {
        console.warn(`Skipping invalid snapshot instance metadata: ${artifact.path}`, error);
      }
    }
    instances.sort((a, b) => a.instance.localeCompare(b.instance));

    const routesFile = filesByPath.get('routes/routes.json');
    let routes: SnapshotRoute[] = [];
    if (routesFile) {
      try {
        routes = parseSnapshotRoutes(JSON.parse(await routesFile.text()));
      } catch (error) {
        console.warn('Skipping invalid snapshot route metadata.', error);
      }
    }

    const reportFile = filesByPath.get('intent/report.json');
    const resolvedIntentFile = filesByPath.get('intent/resolved.json');
    let constraints: SnapshotConstraint[] = [];
    if (reportFile && resolvedIntentFile) {
      try {
        constraints = parseSnapshotConstraints(
          JSON.parse(await reportFile.text()),
          JSON.parse(await resolvedIntentFile.text()),
        );
      } catch (error) {
        console.warn('Skipping invalid physical intent metadata.', error);
      }
    }

    return { manifest, filesByPath, instances, routes, constraints, interfaceJson, sourceMap };
  }

  return undefined;
}

function parseSnapshotInstance(value: unknown, artifactPath: string): SnapshotInstance | undefined {
  const record = asRecord(value);
  const bbox = asRecord(record?.global_bbox);
  const min = readNumberTuple(bbox?.min);
  const max = readNumberTuple(bbox?.max);
  if (!record || typeof record.instance !== 'string' || typeof record.module !== 'string' || !min || !max) {
    return undefined;
  }

  const cost = asRecord(record.cost);
  const blockCount = Number(cost?.blocks);
  return {
    instanceId: Number.isInteger(Number(record.instance_id)) ? Number(record.instance_id) : undefined,
    instance: record.instance,
    module: record.module,
    artifactPath,
    circuitPath: artifactPath.replace(/instance\.json$/i, 'circuit.nbt'),
    global_bbox: { min, max },
    blockCount: Number.isFinite(blockCount) ? blockCount : undefined,
  };
}

function readNumberTuple(value: unknown): [number, number, number] | undefined {
  if (!Array.isArray(value) || value.length < 3) return undefined;
  const tuple = value.slice(0, 3).map(Number);
  return tuple.every(Number.isFinite) ? [tuple[0], tuple[1], tuple[2]] : undefined;
}

function parseSnapshotRoutes(value: unknown): SnapshotRoute[] {
  const routeValues = asRecord(value)?.routes;
  if (!Array.isArray(routeValues)) return [];

  const routes: SnapshotRoute[] = [];
  for (const value of routeValues) {
    const route = asRecord(value);
    const source = readNumberTuple(route?.source);
    const sink = readNumberTuple(route?.sink);
    const path = Array.isArray(route?.path)
      ? route.path.map(readNumberTuple).filter((point): point is [number, number, number] => Boolean(point))
      : [];
    const blocks = Array.isArray(route?.blocks)
      ? route.blocks.map(readNumberTuple).filter((point): point is [number, number, number] => Boolean(point))
      : [];
    const index = Number(route?.index);
    const pathLength = Number(route?.path_length);
    const blockCount = Number(route?.block_count);
    if (
      !route
      || !source
      || !sink
      || typeof route.source_label !== 'string'
      || typeof route.sink_label !== 'string'
      || !Number.isInteger(index)
    ) {
      continue;
    }
    routes.push({
      id: `route-${index}`,
      index,
      netId: Number.isInteger(Number(route.net_id)) ? Number(route.net_id) : undefined,
      source,
      sourceLabel: route.source_label,
      sink,
      sinkLabel: route.sink_label,
      path,
      blocks,
      pathLength: Number.isFinite(pathLength) ? pathLength : path.length,
      blockCount: Number.isFinite(blockCount) ? blockCount : 0,
    });
  }
  return routes.sort((left, right) => left.index - right.index);
}

function parseSnapshotConstraints(reportValue: unknown, resolvedValue: unknown): SnapshotConstraint[] {
  const report = Array.isArray(reportValue) ? reportValue : [];
  const resolved = asRecord(resolvedValue)?.constraints;
  const links = new Map<string, { instanceIds: number[]; netIds: number[] }>();
  if (Array.isArray(resolved)) {
    for (const value of resolved) {
      const constraint = asRecord(value);
      if (!constraint || typeof constraint.id !== 'string') continue;
      const instanceIds = ['instance', 'first', 'second']
        .map(key => Number(constraint[key]))
        .filter((id): id is number => Number.isInteger(id));
      const netIds = [Number(constraint.net)].filter((id): id is number => Number.isInteger(id));
      links.set(constraint.id, { instanceIds: [...new Set(instanceIds)], netIds });
    }
  }

  return report.flatMap(value => {
    const item = asRecord(value);
    if (
      !item
      || typeof item.id !== 'string'
      || typeof item.detail !== 'string'
      || !['satisfied', 'violated', 'not_evaluated'].includes(String(item.status))
    ) return [];
    const related = links.get(item.id) ?? { instanceIds: [], netIds: [] };
    return [{
      id: item.id,
      status: item.status as SnapshotConstraint['status'],
      detail: item.detail,
      ...related,
    }];
  });
}

function renderSnapshotBrowser(snapshot: LoadedSnapshot): void {
  filesList.replaceChildren();
  filesList.className = 'files-list snapshot-files-list';
  filesTitle.textContent = 'Snapshot';
  filesCount.textContent = `${snapshot.manifest.top_module ?? 'snapshot'} · ${snapshot.instances.length} boxes`;
  filesPanel.open = true;

  appendSnapshotSection('NBT');
  const finalPath = snapshot.manifest.final_nbt;
  if (finalPath && snapshot.filesByPath.has(finalPath)) {
    appendSnapshotNbtEntry(finalPath, finalPath, 'main');
  }

  const diagnosticPaths = ['placement-bboxes.nbt', 'routes/routes.nbt']
    .filter(path => snapshot.filesByPath.has(path));
  for (const path of diagnosticPaths) {
    const segments = path.split('/');
    appendSnapshotNbtEntry(segments[segments.length - 1] ?? path, path, 'main');
  }

  const candidateNbt = snapshot.manifest.artifacts.filter(artifact =>
    artifact.kind === 'nbt' && /^candidates\//i.test(artifact.path) && snapshot.filesByPath.has(artifact.path));
  if (candidateNbt.length > 0) {
    const body = appendSnapshotCollapsibleSection('Candidates', candidateNbt.length);
    for (const artifact of candidateNbt) {
      appendSnapshotNbtEntry(artifact.path.replace(/^candidates\//i, ''), artifact.path, 'main', body);
    }
  }

  const irArtifacts = snapshot.manifest.artifacts.filter(artifact =>
    isSnapshotIrArtifact(artifact.path) && snapshot.filesByPath.has(artifact.path));
  if (irArtifacts.length > 0) {
    appendSnapshotSection('IR');
    const button = createFileEntry('Open IR viewer', irArtifacts.map(artifact => snapshotIrLabel(artifact.path)).join(' · '));
    button.classList.add('snapshot-ir-entry');
    button.addEventListener('click', () => void openSnapshotIrViewer(irArtifacts.map(artifact => artifact.path)));
    filesList.append(button);
  }

  if (snapshot.instances.length > 0) {
    appendSnapshotSection('Instances');
    for (const instance of snapshot.instances) {
      const file = snapshot.filesByPath.get(instance.circuitPath);
      if (!file) continue;
      const button = createFileEntry(instance.instance, instance.module, file.size);
      button.dataset.snapshotPath = instance.circuitPath;
      button.dataset.snapshotInstance = instance.artifactPath;
      button.addEventListener('mouseenter', () => hoverSnapshotInstance(instance));
      button.addEventListener('mouseleave', () => hoverSnapshotInstance(undefined));
      button.addEventListener('click', () => selectSnapshotInstance(instance));
      filesList.append(button);
    }
  }

  if (snapshot.constraints.length > 0) {
    appendSnapshotSection('Constraints');
    for (const constraint of snapshot.constraints) {
      const related = constraintRelatedLabel(snapshot, constraint);
      const button = createFileEntry(constraint.id, `${constraint.status}${related ? ` · ${related}` : ''}`);
      button.classList.add('constraint-entry', `constraint-${constraint.status}`);
      button.dataset.snapshotConstraint = constraint.id;
      button.querySelector('.file-entry-size')?.classList.add('constraint-entry-status');
      button.addEventListener('click', () => focusSnapshotConstraint(constraint));
      filesList.append(button);
    }
  }

  const metadata = snapshot.manifest.artifacts.filter(
    artifact =>
      artifact.kind !== 'nbt' &&
      !isSnapshotIrArtifact(artifact.path) &&
      !/^instances\/[^/]+\/instance\.json$/i.test(artifact.path) &&
      snapshot.filesByPath.has(artifact.path),
  );
  if (metadata.length > 0) {
    const body = appendSnapshotCollapsibleSection('Others', metadata.length);
    for (const artifact of metadata) {
      const file = snapshot.filesByPath.get(artifact.path)!;
      const button = createFileEntry(artifact.path, artifact.kind, file.size);
      button.dataset.snapshotPath = artifact.path;
      button.addEventListener('click', () => void openTextArtifact(artifact.path));
      body.append(button);
    }
  }
}

function appendSnapshotSection(label: string): void {
  const heading = document.createElement('div');
  heading.className = 'snapshot-section-title';
  heading.textContent = label;
  filesList.append(heading);
}

function appendSnapshotCollapsibleSection(label: string, count: number): HTMLElement {
  const details = document.createElement('details');
  details.className = 'snapshot-collapsible';
  const summary = document.createElement('summary');
  summary.textContent = `${label} (${count})`;
  const body = document.createElement('div');
  body.className = 'snapshot-collapsible-body';
  details.append(summary, body);
  filesList.append(details);
  return body;
}

function appendSnapshotNbtEntry(
  label: string,
  path: string,
  target: 'main',
  parent: HTMLElement = filesList,
): void {
  const file = currentSnapshot?.filesByPath.get(path);
  if (!file || target !== 'main') return;
  const button = createFileEntry(label, 'NBT', file.size);
  button.dataset.snapshotPath = path;
  button.addEventListener('click', () => void openSnapshotNbt(path, button));
  parent.append(button);
}

function isSnapshotIrArtifact(path: string): boolean {
  return /\.v$/i.test(path) || /^ir\/(logical|routable)\.rcir$/i.test(path);
}

function snapshotIrLabel(path: string): string {
  if (/\.v$/i.test(path)) return 'Verilog';
  if (/logical\.rcir$/i.test(path)) return 'Logical RCIR';
  if (/routable\.rcir$/i.test(path)) return 'Routable RCIR';
  return path;
}

function createFileEntry(label: string, detail: string, size?: number): HTMLButtonElement {
  const button = document.createElement('button');
  button.className = 'file-entry';
  button.type = 'button';
  const name = document.createElement('span');
  name.className = 'file-entry-name';
  name.textContent = label;
  name.title = label;
  const metadata = document.createElement('span');
  metadata.className = 'file-entry-size snapshot-entry-detail';
  metadata.textContent = size === undefined ? detail : `${detail} · ${formatBytes(size)}`;
  button.append(name, metadata);
  return button;
}

function constraintRelatedLabel(snapshot: LoadedSnapshot, constraint: SnapshotConstraint): string {
  const instances = constraint.instanceIds
    .map(id => snapshot.instances.find(instance => instance.instanceId === id)?.instance)
    .filter((name): name is string => Boolean(name));
  const nets = constraint.netIds.map(id => {
    const route = snapshot.routes.find(candidate => candidate.netId === id);
    return route ? `net ${id}: ${route.sourceLabel}` : `net ${id}`;
  });
  return [...instances, ...nets].join(', ');
}

function focusSnapshotConstraint(constraint: SnapshotConstraint): void {
  const snapshot = currentSnapshot;
  if (!snapshot) return;

  setSnapshotRouteIsolation(undefined);
  setSnapshotBoxIsolation(undefined);
  const instances = constraint.instanceIds
    .map(id => snapshot.instances.find(instance => instance.instanceId === id))
    .filter((instance): instance is SnapshotInstance => Boolean(instance));
  const routes = snapshot.routes.filter(route =>
    route.netId !== undefined && constraint.netIds.includes(route.netId));

  if (instances.length === 1) {
    const instance = instances[0];
    setSnapshotBoxIsolation({
      id: instance.artifactPath,
      label: instance.instance,
      min: compilerPositionToNbt(instance.global_bbox.min),
      max: compilerPositionToNbt(instance.global_bbox.max).map(value => value + 1) as [number, number, number],
    });
  } else if (routes.length === 1) {
    setSnapshotRouteIsolation(routes[0].id);
  } else if (routes.length > 1) {
    viewer.setRelatedRouteIds(routes.map(route => route.id));
  }

  filesList.querySelectorAll('.file-entry.selected').forEach(entry => entry.classList.remove('selected'));
  filesList.querySelector<HTMLElement>(`[data-snapshot-constraint="${CSS.escape(constraint.id)}"]`)
    ?.classList.add('selected');
  selectedBlock = undefined;
  toggleSwitchButton.classList.add('hidden');
  inspector.textContent = [
    `${constraint.status}: ${constraint.id}`,
    constraintRelatedLabel(snapshot, constraint) || 'no linked instance or net',
    constraint.detail,
  ].join('\n');
}

async function openSnapshotNbt(path: string, selectedEntry?: Element | null): Promise<void> {
  const snapshot = currentSnapshot;
  const file = snapshot?.filesByPath.get(path);
  if (!snapshot || !file) throw new Error(`Snapshot artifact is missing: ${path}`);
  const outputMetadataJson = path === snapshot.manifest.final_nbt ? snapshot.interfaceJson : undefined;
  await openFile(file, selectedEntry, outputMetadataJson, path);
}

function snapshotBoxForInstance(instance: SnapshotInstance): {
  id: string;
  label: string;
  min: [number, number, number];
  max: [number, number, number];
} {
  return {
    id: instance.artifactPath,
    label: instance.instance,
    min: compilerPositionToNbt(instance.global_bbox.min),
    max: compilerPositionToNbt(instance.global_bbox.max).map(value => value + 1) as [number, number, number],
  };
}

function hoverSnapshotInstance(instance: SnapshotInstance | undefined): void {
  viewer.setHoveredBoundingBoxId(instance?.artifactPath);
  viewer.setRelatedRouteIds(
    instance ? relatedRouteIdsForBox(instance.artifactPath) : relatedRouteIdsForBox(isolatedSnapshotBoxId),
  );
}

function selectSnapshotInstance(instance: SnapshotInstance): void {
  setSnapshotRouteIsolation(undefined);
  setSnapshotBoxIsolation(
    isolatedSnapshotBoxId === instance.artifactPath ? undefined : snapshotBoxForInstance(instance),
  );
}

async function openTextArtifact(path: string): Promise<void> {
  const file = currentSnapshot?.filesByPath.get(path);
  if (!file) throw new Error(`Snapshot artifact is missing: ${path}`);
  irComparison.replaceChildren();
  irComparison.classList.add('hidden');
  irColorMappingToggle.classList.add('hidden');
  artifactContent.classList.remove('hidden');
  artifactTitle.textContent = path;
  await renderArtifactContent(path);
  if (!artifactDialog.open) artifactDialog.showModal();
}

async function openSnapshotIrViewer(paths: string[]): Promise<void> {
  const orderedPaths = [...paths].sort((a, b) => snapshotIrOrder(a) - snapshotIrOrder(b));
  if (orderedPaths.length === 0) return;

  artifactTitle.textContent = `${currentSnapshot?.manifest.top_module ?? 'Snapshot'} IR`;
  artifactContent.classList.add('hidden');
  irComparison.classList.remove('hidden');
  irColorMappingToggle.classList.remove('hidden');
  irComparison.classList.toggle('ir-color-mapping-disabled', !irColorMappingInput.checked);
  irComparison.replaceChildren();
  irComparison.style.removeProperty('grid-template-columns');
  pinnedIrLocations = undefined;
  pinnedIrEntities = undefined;
  irLocationColorSlots.clear();

  const sources = await Promise.all(orderedPaths.map(async path => {
    const file = currentSnapshot?.filesByPath.get(path);
    if (!file) throw new Error(`Snapshot artifact is missing: ${path}`);
    return { path, source: await file.text() };
  }));
  for (const [index, { path, source }] of sources.entries()) {
    if (index > 0) irComparison.append(createIrComparisonSplitter(index - 1));
    irComparison.append(createIrComparisonPane(path, source));
  }
  resetIrPaneWidths();
  if (!artifactDialog.open) artifactDialog.showModal();
}

function createIrComparisonSplitter(leftPaneIndex: number): HTMLElement {
  const splitter = document.createElement('div');
  splitter.className = 'ir-comparison-splitter';
  splitter.dataset.leftPaneIndex = String(leftPaneIndex);
  splitter.setAttribute('role', 'separator');
  splitter.setAttribute('aria-orientation', 'vertical');
  splitter.setAttribute('aria-label', `Resize IR panes ${leftPaneIndex + 1} and ${leftPaneIndex + 2}`);
  splitter.tabIndex = 0;

  splitter.addEventListener('pointerdown', event => beginIrPaneResize(event, splitter));
  splitter.addEventListener('dblclick', resetIrPaneWidths);
  splitter.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    resizeIrPanePair(splitter, event.key === 'ArrowLeft' ? -24 : 24);
  });
  return splitter;
}

function resetIrPaneWidths(): void {
  const paneCount = irComparison.querySelectorAll('.ir-comparison-pane').length;
  const columns = Array.from({ length: paneCount }, (_, index) => (
    index + 1 < paneCount ? ['minmax(180px, 1fr)', '7px'] : ['minmax(180px, 1fr)']
  )).flat();
  irComparison.style.gridTemplateColumns = columns.join(' ');
}

function beginIrPaneResize(event: PointerEvent, splitter: HTMLElement): void {
  if (event.button !== 0) return;
  event.preventDefault();
  const startX = event.clientX;
  const startWidths = currentIrPaneWidths();
  const leftPaneIndex = Number(splitter.dataset.leftPaneIndex);
  splitter.classList.add('dragging');
  document.body.classList.add('ir-pane-resizing');
  splitter.setPointerCapture(event.pointerId);

  const move = (moveEvent: PointerEvent) => {
    applyIrPanePairResize(startWidths, leftPaneIndex, moveEvent.clientX - startX);
  };
  const stop = () => {
    splitter.classList.remove('dragging');
    document.body.classList.remove('ir-pane-resizing');
    splitter.removeEventListener('pointermove', move);
    splitter.removeEventListener('pointerup', stop);
    splitter.removeEventListener('pointercancel', stop);
  };
  splitter.addEventListener('pointermove', move);
  splitter.addEventListener('pointerup', stop);
  splitter.addEventListener('pointercancel', stop);
}

function resizeIrPanePair(splitter: HTMLElement, delta: number): void {
  applyIrPanePairResize(currentIrPaneWidths(), Number(splitter.dataset.leftPaneIndex), delta);
}

function currentIrPaneWidths(): number[] {
  return Array.from(irComparison.querySelectorAll<HTMLElement>('.ir-comparison-pane'))
    .map(pane => pane.getBoundingClientRect().width);
}

function applyIrPanePairResize(widths: number[], leftPaneIndex: number, delta: number): void {
  const minimumWidth = 180;
  const pairWidth = widths[leftPaneIndex] + widths[leftPaneIndex + 1];
  const leftWidth = Math.min(
    pairWidth - minimumWidth,
    Math.max(minimumWidth, widths[leftPaneIndex] + delta),
  );
  const resized = [...widths];
  resized[leftPaneIndex] = leftWidth;
  resized[leftPaneIndex + 1] = pairWidth - leftWidth;
  const columns = resized.flatMap((width, index) => (
    index + 1 < resized.length ? [`${Math.round(width)}px`, '7px'] : [`${Math.round(width)}px`]
  ));
  irComparison.style.gridTemplateColumns = columns.join(' ');
}

function snapshotIrOrder(path: string): number {
  if (/\.v$/i.test(path)) return 0;
  if (/logical\.rcir$/i.test(path)) return 1;
  if (/routable\.rcir$/i.test(path)) return 2;
  return 3;
}

function createIrComparisonPane(path: string, source: string): HTMLElement {
  const pane = document.createElement('section');
  pane.className = 'ir-comparison-pane';

  const header = document.createElement('header');
  header.className = 'ir-comparison-header';
  const title = document.createElement('strong');
  title.textContent = snapshotIrLabel(path);
  const filename = document.createElement('span');
  filename.textContent = path;
  header.append(title, filename);

  const code = document.createElement('div');
  code.className = 'ir-code';
  code.classList.toggle('language-rcir', path.toLowerCase().endsWith('.rcir'));
  code.classList.toggle('language-verilog', path.toLowerCase().endsWith('.v'));
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const verilogState: VerilogHighlightState = { inBlockComment: false };
  lines.forEach((line, index) => {
    const row = document.createElement('div');
    row.className = 'ir-code-line';
    row.dataset.irPath = path;
    row.dataset.irLine = String(index + 1);
    const locationIds = sourceMapLocationsForLine(path, index + 1);
    const entityIds = sourceMapEntitiesForLine(path, index + 1);
    if (locationIds.length > 0) {
      row.classList.add('ir-code-line-linked');
      row.dataset.irLocations = locationIds.join(',');
      row.dataset.irEntities = entityIds.join(',');
      const colorGroup = sourceMapColorGroupForLine(path, index + 1);
      if (colorGroup) {
        let colorSlot = irLocationColorSlots.get(colorGroup);
        if (colorSlot === undefined) {
          colorSlot = irLocationColorSlots.size;
          irLocationColorSlots.set(colorGroup, colorSlot);
        }
        row.classList.add('ir-code-line-grouped');
        row.style.setProperty(
          '--ir-location-hue',
          String(IR_LOCATION_HUES[colorSlot % IR_LOCATION_HUES.length]),
        );
      }
      row.addEventListener('mouseenter', () => {
        if (!pinnedIrLocations) renderIrLocationHighlight(locationIds, entityIds, row);
      });
      row.addEventListener('mouseleave', () => {
        if (!pinnedIrLocations) renderIrLocationHighlight(undefined, undefined);
      });
      row.addEventListener('click', () => {
        const sameSelection = pinnedIrLocations?.length === locationIds.length
          && pinnedIrLocations.every(location => locationIds.includes(location));
        pinnedIrLocations = sameSelection ? undefined : locationIds;
        pinnedIrEntities = sameSelection ? undefined : entityIds;
        renderIrLocationHighlight(
          pinnedIrLocations,
          pinnedIrEntities,
          sameSelection ? undefined : row,
        );
        if (!sameSelection) scrollRelatedIrPanesIntoView(row);
      });
    }
    const number = document.createElement('span');
    number.className = 'ir-line-number';
    number.textContent = String(index + 1);
    const content = document.createElement('code');
    content.className = 'ir-line-content';
    if (path.toLowerCase().endsWith('.rcir')) {
      content.replaceChildren(highlightRcir(line || '\u200b'));
    } else if (path.toLowerCase().endsWith('.v')) {
      content.replaceChildren(highlightVerilog(line || '\u200b', verilogState));
    } else {
      content.textContent = line || '\u200b';
    }
    row.append(number, content);
    code.append(row);
  });

  pane.append(header, code);
  return pane;
}

function sourceMapRangesForLine(path: string, line: number): SnapshotDebugRange[] {
  return currentSnapshot?.sourceMap?.documents[path]
    ?.filter(range => range.start_line <= line && line <= range.end_line) ?? [];
}

function sourceMapLocationsForLine(path: string, line: number): number[] {
  const ranges = sourceMapRangesForLine(path, line);
  if (ranges.length === 0) return [];
  const smallestSpan = Math.min(...ranges.map(range => range.end_line - range.start_line));
  return [...new Set(ranges
    .filter(range => range.end_line - range.start_line === smallestSpan)
    .map(range => range.location))];
}

function sourceMapEntitiesForLine(path: string, line: number): string[] {
  const ranges = sourceMapRangesForLine(path, line);
  if (ranges.length === 0) return [];
  const smallestSpan = Math.min(...ranges.map(range => range.end_line - range.start_line));
  return [...new Set(ranges
    .filter(range => range.end_line - range.start_line === smallestSpan)
    .map(range => range.entity))];
}

function sourceMapRootLocations(locationId: number, visiting = new Set<number>()): number[] {
  if (visiting.has(locationId)) return [];
  const location = currentSnapshot?.sourceMap?.locations[locationId];
  if (!location || location.kind === 'source') return [locationId];

  const nextVisiting = new Set(visiting).add(locationId);
  if (location.kind === 'derived') {
    return sourceMapRootLocations(location.parent, nextVisiting);
  }
  return location.parents.flatMap(parent => sourceMapRootLocations(parent, nextVisiting));
}

function sourceMapColorGroupForLine(path: string, line: number): string | undefined {
  const ranges = sourceMapRangesForLine(path, line);
  if (ranges.length === 0) return undefined;
  const smallestSpan = Math.min(...ranges.map(range => range.end_line - range.start_line));
  const roots = [...new Set(ranges
    .filter(range => range.end_line - range.start_line === smallestSpan)
    .flatMap(range => sourceMapRootLocations(range.location)))]
    .sort((left, right) => left - right);
  return roots.length > 0 ? roots.join(',') : undefined;
}

function relatedIrLocations(selected: number[]): Set<number> {
  const locations = currentSnapshot?.sourceMap?.locations ?? [];
  const parents = new Map<number, Set<number>>();
  const children = new Map<number, Set<number>>();
  const link = (child: number, parent: number) => {
    if (!parents.has(child)) parents.set(child, new Set());
    if (!children.has(parent)) children.set(parent, new Set());
    parents.get(child)!.add(parent);
    children.get(parent)!.add(child);
  };
  locations.forEach((location, index) => {
    if (location.kind === 'derived') link(index, location.parent);
    if (location.kind === 'fused') location.parents.forEach(parent => link(index, parent));
  });
  const related = new Set(selected);
  const walk = (start: number[], edges: Map<number, Set<number>>) => {
    const visited = new Set(start);
    const queue = [...start];
    while (queue.length > 0) {
      const location = queue.shift()!;
      for (const neighbor of edges.get(location) ?? []) {
        if (visited.has(neighbor)) continue;
        visited.add(neighbor);
        related.add(neighbor);
        queue.push(neighbor);
      }
    }
  };
  // A selection sees its own lowering descendants and its provenance ancestors.
  // It deliberately does not descend again from an ancestor into sibling results.
  walk(selected, parents);
  walk(selected, children);
  return related;
}

function irRowLocations(row: HTMLElement): number[] {
  return (row.dataset.irLocations ?? '').split(',')
    .filter(Boolean)
    .map(Number)
    .filter(Number.isInteger);
}

function irRowEntities(row: HTMLElement): string[] {
  return (row.dataset.irEntities ?? '').split(',').filter(Boolean);
}

function referencedIrLocations(selectedEntities: string[]): Set<number> {
  const sourceMap = currentSnapshot?.sourceMap;
  if (!sourceMap) return new Set();
  const selected = new Set(selectedEntities);
  return new Set((sourceMap.relations ?? [])
    .filter(relation => selected.has(relation.from))
    .map(relation => sourceMap.entities?.[relation.to]?.location)
    .filter((location): location is number => Number.isInteger(location)));
}

function scopedIrLocations(exactLocations: Set<number>): Set<number> {
  const sourceMap = currentSnapshot?.sourceMap;
  if (!sourceMap) return new Set();
  const entities = Object.entries(sourceMap.entities ?? {});
  const scopes = new Set(entities
    .filter(([, entity]) => entity.kind === 'module' && exactLocations.has(entity.location))
    .map(([id]) => id));
  if (scopes.size === 0) return new Set();

  let changed = true;
  while (changed) {
    changed = false;
    for (const [id, entity] of entities) {
      if (!entity.parent_scope || !scopes.has(entity.parent_scope) || scopes.has(id)) continue;
      scopes.add(id);
      changed = true;
    }
  }

  const result = new Set<number>();
  for (const [id, entity] of entities) {
    if (!scopes.has(id)) continue;
    for (const location of relatedIrLocations([entity.location])) result.add(location);
  }
  return result;
}

function rowStartsMappedModule(row: HTMLElement): boolean {
  const path = row.dataset.irPath;
  const line = Number(row.dataset.irLine);
  if (!path || !Number.isInteger(line)) return false;
  const entities = new Set(irRowEntities(row));
  return (currentSnapshot?.sourceMap?.documents[path] ?? []).some(range =>
    entities.has(range.entity)
    && currentSnapshot?.sourceMap?.entities?.[range.entity]?.kind === 'module'
    && range.start_line === line);
}

function renderIrLocationHighlight(
  selected: number[] | undefined,
  selectedEntities: string[] | undefined,
  activeRow?: HTMLElement,
): void {
  const related = selected ? relatedIrLocations(selected) : new Set<number>();
  const scoped = selected ? scopedIrLocations(related) : new Set<number>();
  const references = selectedEntities
    ? referencedIrLocations(selectedEntities)
    : new Set<number>();
  const referencedScopes = scopedIrLocations(references);
  irComparison.querySelectorAll<HTMLElement>('.ir-code-line').forEach(row => {
    const rowLocations = irRowLocations(row);
    const locationMatches = rowLocations.some(location => related.has(location));
    const rowEntities = irRowEntities(row);
    const moduleOnly = rowEntities.length > 0 && rowEntities.every(entity =>
      currentSnapshot?.sourceMap?.entities?.[entity]?.kind === 'module');
    const matches = locationMatches
      && (!moduleOnly || rowStartsMappedModule(row) || row === activeRow);
    const scopeMatch = !matches && rowLocations.some(location => scoped.has(location));
    const referenceLocationMatch = rowLocations.some(location => references.has(location));
    const referenceMatch = referenceLocationMatch
      && (!moduleOnly || rowStartsMappedModule(row));
    const referenceScopeMatch = !referenceMatch
      && rowLocations.some(location => referencedScopes.has(location));
    row.classList.toggle('ir-location-related', matches);
    row.classList.toggle('ir-location-scope', scopeMatch);
    row.classList.toggle('ir-location-reference', referenceMatch);
    row.classList.toggle('ir-location-reference-scope', referenceScopeMatch);
    row.classList.toggle('ir-location-selected', matches && row === activeRow);
  });
}

function scrollRelatedIrPanesIntoView(activeRow: HTMLElement): void {
  for (const pane of irComparison.querySelectorAll<HTMLElement>('.ir-comparison-pane')) {
    if (pane.contains(activeRow)) continue;
    pane.querySelector<HTMLElement>('.ir-code-line.ir-location-related')
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

async function renderArtifactContent(path: string): Promise<void> {
  const file = currentSnapshot?.filesByPath.get(path);
  if (!file) throw new Error(`Snapshot artifact is missing: ${path}`);
  const text = await file.text();
  const isRcir = path.toLowerCase().endsWith('.rcir');
  const isVerilog = path.toLowerCase().endsWith('.v');
  artifactContent.classList.toggle('language-rcir', isRcir);
  artifactContent.classList.toggle('language-verilog', isVerilog);
  if (isRcir) {
    artifactContent.replaceChildren(highlightRcir(text));
    return;
  }
  if (isVerilog) {
    artifactContent.replaceChildren(highlightVerilog(text));
    return;
  }
  try {
    artifactContent.textContent = JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    artifactContent.textContent = text;
  }
}

function findSnapshotEntry(path: string): Element | null {
  return Array.from(filesList.querySelectorAll<HTMLElement>('[data-snapshot-path]')).find(
    entry => entry.dataset.snapshotPath === path,
  ) ?? null;
}

function updateSnapshotBoxes(): void {
  const snapshot = currentSnapshot;
  const available = Boolean(
    snapshot && currentSnapshotPath === snapshot.manifest.final_nbt && snapshot.instances.length > 0,
  );
  const visible = available && snapshotBoxesVisible;
  const boxes = visible
    ? snapshot!.instances.map(instance => ({
        id: instance.artifactPath,
        label: instance.instance,
        min: compilerPositionToNbt(instance.global_bbox.min),
        max: compilerPositionToNbt(instance.global_bbox.max).map(value => value + 1) as [number, number, number],
      }))
    : [];
  viewer.setBoundingBoxes(
    boxes,
    box => {
      if (!box || isolatedSnapshotBoxId === box.id) {
        setSnapshotBoxIsolation(undefined);
      } else {
        setSnapshotBoxIsolation(box);
      }
    },
    renderSnapshotBoxHover,
  );
  toggleSnapshotBoxesButton.classList.toggle('hidden', !available);
  toggleSnapshotBoxesButton.classList.toggle('active', visible);
  toggleSnapshotBoxesButton.textContent = snapshot ? `Boxes: ${snapshot.instances.length}` : 'Boxes';
  toggleSnapshotBoxesButton.setAttribute('aria-pressed', String(visible));
}

function setSnapshotBoxIsolation(box: {
  id: string;
  label: string;
  min: [number, number, number];
  max: [number, number, number];
} | undefined): void {
  isolatedSnapshotBoxId = box?.id;
  viewer.setIsolatedBoundingBox(box);
  viewer.setRelatedRouteIds(relatedRouteIdsForBox(box?.id));

  for (const entry of filesList.querySelectorAll<HTMLElement>('[data-snapshot-instance]')) {
    entry.classList.toggle('selected', entry.dataset.snapshotInstance === box?.id);
  }

  if (!box) return;
  const instance = currentSnapshot?.instances.find(candidate => candidate.artifactPath === box.id);
  if (!instance) return;
  selectedBlock = undefined;
  toggleSwitchButton.classList.add('hidden');
  inspector.textContent = [
    `isolated: ${instance.instance}`,
    `module: ${instance.module}`,
    `blocks: ${instance.blockCount ?? 'unknown'}`,
    'click again, empty space, or press Esc to restore',
  ].join('\n');
}

function relatedRouteIdsForBox(boxId: string | undefined): string[] {
  const instance = boxId
    ? currentSnapshot?.instances.find(candidate => candidate.artifactPath === boxId)
    : undefined;
  if (!instance) return [];
  const prefix = `${instance.instance}.`;
  return currentSnapshot?.routes
    .filter(route => route.sourceLabel.startsWith(prefix) || route.sinkLabel.startsWith(prefix))
    .map(route => route.id) ?? [];
}

function updateSnapshotRoutes(): void {
  const snapshot = currentSnapshot;
  const available = Boolean(
    snapshot && currentSnapshotPath === snapshot.manifest.final_nbt && snapshot.routes.length > 0,
  );
  const visible = available && snapshotRoutesVisible;
  const routes = visible
    ? snapshot!.routes
      .filter(route => !isolatedSnapshotRouteId || route.id === isolatedSnapshotRouteId)
      .map(route => ({
        id: route.id,
        sourceLabel: route.sourceLabel,
        sinkLabel: route.sinkLabel,
        points: routePoints(route).map(position => {
          const [x, y, z] = compilerPositionToNbt(position);
          return [x + 0.5, y + 0.5, z + 0.5] as [number, number, number];
        }),
        pathLength: route.pathLength,
        blockCount: route.blockCount,
      }))
    : [];
  viewer.setRoutes(
    routes,
    route => {
      if (!route) {
        if (isolatedSnapshotRouteId) setSnapshotRouteIsolation(undefined);
      } else if (isolatedSnapshotRouteId === route.id) {
        setSnapshotRouteIsolation(undefined);
      } else {
        setSnapshotRouteIsolation(route.id);
      }
    },
    renderSnapshotRouteHover,
  );
  toggleSnapshotRoutesButton.classList.toggle('hidden', !available);
  toggleSnapshotRoutesButton.classList.toggle('active', visible);
  toggleSnapshotRoutesButton.textContent = snapshot ? `Routes: ${snapshot.routes.length}` : 'Routes';
  toggleSnapshotRoutesButton.setAttribute('aria-pressed', String(visible));
}

function setSnapshotRouteIsolation(routeId: string | undefined): void {
  isolatedSnapshotRouteId = routeId;
  const route = routeId
    ? currentSnapshot?.routes.find(candidate => candidate.id === routeId)
    : undefined;
  viewer.setIsolatedBlockPositions(route ? routeBlockPositions(route) : undefined);
  updateSnapshotRoutes();
  viewer.setSelectedRouteId(routeId);
  if (routeId) {
    renderSnapshotRouteSelection(routeId);
  } else {
    renderSelection(undefined);
  }
}

function routeBlockPositions(route: SnapshotRoute): Array<[number, number, number]> {
  const compilerPositions = route.blocks.length > 0
    ? route.blocks
    : route.path.flatMap(([x, y, z]) => [[x, y, z], [x, y, z - 1]] as Array<[number, number, number]>);
  return compilerPositions.map(compilerPositionToNbt);
}

function routePoints(route: SnapshotRoute): Array<[number, number, number]> {
  const points = [route.source, ...route.path, route.sink];
  return points.filter((point, index) => index === 0 || !sameTuple(point, points[index - 1]));
}

function sameTuple(left: [number, number, number], right: [number, number, number]): boolean {
  return left[0] === right[0] && left[1] === right[1] && left[2] === right[2];
}

function renderSnapshotRouteSelection(routeId: string): void {
  const route = currentSnapshot?.routes.find(candidate => candidate.id === routeId);
  if (!route) return;
  selectedBlock = undefined;
  toggleSwitchButton.classList.add('hidden');
  inspector.textContent = [
    `route #${route.index}`,
    `source: ${route.sourceLabel}`,
    `sink: ${route.sinkLabel}`,
    `path: ${route.pathLength}`,
    `blocks: ${route.blockCount}`,
  ].join('\n');
}

function renderSnapshotRouteHover(
  route: { id: string } | undefined,
  clientX: number,
  clientY: number,
): void {
  clearSnapshotHoverUi();
  const snapshotRoute = route
    ? currentSnapshot?.routes.find(candidate => candidate.id === route.id)
    : undefined;
  if (!snapshotRoute) return;

  const instanceNames = new Set([
    routeInstanceName(snapshotRoute.sourceLabel),
    routeInstanceName(snapshotRoute.sinkLabel),
  ]);
  for (const entry of filesList.querySelectorAll<HTMLElement>('[data-snapshot-instance]')) {
    const instance = currentSnapshot?.instances.find(candidate => candidate.artifactPath === entry.dataset.snapshotInstance);
    if (instance && instanceNames.has(instance.instance)) entry.classList.add('bbox-hover');
  }

  bboxTooltipTitle.textContent = `${snapshotRoute.sourceLabel} → ${snapshotRoute.sinkLabel}`;
  bboxTooltipDetail.textContent = `route #${snapshotRoute.index} · path ${snapshotRoute.pathLength} · blocks ${snapshotRoute.blockCount}`;
  showSnapshotTooltip(clientX, clientY);
}

function routeInstanceName(label: string): string {
  const separator = label.lastIndexOf('.');
  return separator < 0 ? label : label.slice(0, separator);
}

function renderSnapshotBoxHover(
  box: { id: string } | undefined,
  clientX: number,
  clientY: number,
): void {
  clearSnapshotHoverUi();

  const instance = box
    ? currentSnapshot?.instances.find(candidate => candidate.artifactPath === box.id)
    : undefined;
  if (!instance) {
    viewer.setRelatedRouteIds(relatedRouteIdsForBox(isolatedSnapshotBoxId));
    return;
  }

  const routePrefix = `${instance.instance}.`;
  viewer.setRelatedRouteIds(
    currentSnapshot?.routes
      .filter(route => route.sourceLabel.startsWith(routePrefix) || route.sinkLabel.startsWith(routePrefix))
      .map(route => route.id) ?? [],
  );

  const entry = Array.from(filesList.querySelectorAll<HTMLElement>('[data-snapshot-instance]')).find(
    candidate => candidate.dataset.snapshotInstance === instance.artifactPath,
  );
  entry?.classList.add('bbox-hover');

  const { min, max } = instance.global_bbox;
  const size = min.map((value, index) => max[index] - value + 1);
  const modulePrefix = instance.module === instance.instance ? '' : `${instance.module} · `;
  bboxTooltipTitle.textContent = instance.instance;
  bboxTooltipDetail.textContent = `${modulePrefix}size ${size.join(' × ')} · origin (${min.join(', ')})`;
  showSnapshotTooltip(clientX, clientY);
}

function clearSnapshotHoverUi(): void {
  filesList.querySelectorAll('.file-entry.bbox-hover').forEach(entry => entry.classList.remove('bbox-hover'));
  bboxTooltip.classList.add('hidden');
  bboxTooltip.setAttribute('aria-hidden', 'true');
}

function showSnapshotTooltip(clientX: number, clientY: number): void {
  bboxTooltip.classList.remove('hidden');
  bboxTooltip.setAttribute('aria-hidden', 'false');

  const bounds = bboxTooltip.getBoundingClientRect();
  const gap = 14;
  const viewportPadding = 8;
  const left = clientX + gap + bounds.width <= window.innerWidth - viewportPadding
    ? clientX + gap
    : clientX - gap - bounds.width;
  const top = clientY + gap + bounds.height <= window.innerHeight - viewportPadding
    ? clientY + gap
    : clientY - gap - bounds.height;
  bboxTooltip.style.left = `${Math.max(viewportPadding, left)}px`;
  bboxTooltip.style.top = `${Math.max(viewportPadding, top)}px`;
}

function leaveSnapshotMode(): void {
  currentSnapshot = undefined;
  currentSnapshotPath = undefined;
  isolatedSnapshotBoxId = undefined;
  isolatedSnapshotRouteId = undefined;
  viewer.setIsolatedBoundingBox(undefined);
  viewer.setIsolatedBlockPositions(undefined);
  viewer.setSelectedRouteId(undefined);
  snapshotBoxesVisible = true;
  snapshotRoutesVisible = true;
  updateSnapshotBoxes();
  updateSnapshotRoutes();
}

function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^\.\//, '');
}

function compilerPositionToNbt(position: [number, number, number]): [number, number, number] {
  return [position[1], position[2], position[0]];
}

function renderFileBrowser(files: File[]): void {
  leaveSnapshotMode();
  const nbtFiles = files
    .filter(isSupportedFile)
    .sort((a, b) => getDisplayPath(a).localeCompare(getDisplayPath(b)));

  filesList.replaceChildren();
  filesTitle.textContent = 'Files';
  filesList.classList.toggle('empty', nbtFiles.length === 0);
  filesCount.textContent = nbtFiles.length === 0 ? 'No NBT files' : `${nbtFiles.length} files`;
  filesPanel.open = true;

  if (nbtFiles.length === 0) {
    filesList.textContent = 'No supported NBT files found.';
    return;
  }

  for (const file of nbtFiles) {
    const button = document.createElement('button');
    button.className = 'file-entry';
    button.type = 'button';
    const name = document.createElement('span');
    name.className = 'file-entry-name';
    name.textContent = getDisplayPath(file);
    const size = document.createElement('span');
    size.className = 'file-entry-size';
    size.textContent = formatBytes(file.size);
    button.append(name, size);
    button.addEventListener('click', () => void openFile(file, button));
    filesList.append(button);
  }

  void openFile(nbtFiles[0], filesList.querySelector('.file-entry'));
}

async function loadExamples(): Promise<void> {
  try {
    const response = await fetch(resolveAssetPath('examples/manifest.json'));
    if (!response.ok) throw new Error(`Failed to load examples: ${response.status}`);
    const examples = (await response.json()) as ExampleFile[];
    renderExampleBrowser(examples);
    const initialExample = examples.find(example => example.kind === 'nbt');
    if (initialExample) {
      await openExample(initialExample, findExampleEntry(initialExample.path));
    }
  } catch (error) {
    filesList.classList.add('empty');
    filesCount.textContent = 'No examples';
    filesList.textContent = error instanceof Error ? error.message : String(error);
  }
}

function renderExampleBrowser(examples: ExampleFile[]): void {
  filesList.replaceChildren();
  filesTitle.textContent = 'Files';
  filesList.classList.toggle('empty', examples.length === 0);
  filesCount.textContent = examples.length === 0 ? 'No NBT files' : `${examples.length} files`;

  if (examples.length === 0) {
    filesList.textContent = 'No supported NBT files found.';
    return;
  }

  for (const [kind, label] of [['snapshot', 'Snapshots'], ['nbt', 'NBT Examples']] as const) {
    const groupedExamples = examples.filter(example => example.kind === kind);
    if (groupedExamples.length === 0) continue;
    appendSnapshotSection(label);
    for (const example of groupedExamples) appendExampleEntry(example);
  }
}

function appendExampleEntry(example: ExampleFile): void {
  const button = document.createElement('button');
  button.className = 'file-entry';
  button.type = 'button';
  button.dataset.examplePath = example.path;
  const name = document.createElement('span');
  name.className = 'file-entry-name';
  name.textContent = example.name;
  const size = document.createElement('span');
  size.className = 'file-entry-size';
  size.textContent = formatBytes(example.size);
  button.append(name, size);
  button.addEventListener('click', () => void openExample(example, button));
  filesList.append(button);
}

function findExampleEntry(path: string): Element | null {
  return Array.from(filesList.querySelectorAll<HTMLElement>('[data-example-path]')).find(
    entry => entry.dataset.examplePath === path,
  ) ?? null;
}

async function openExample(example: ExampleFile, selectedEntry?: Element | null): Promise<void> {
  const response = await fetch(resolveAssetPath(example.path));
  if (!response.ok) throw new Error(`Failed to load ${example.path}: ${response.status}`);
  const file = new File([await response.arrayBuffer()], example.name, { type: 'application/octet-stream' });
  if (example.kind === 'snapshot') {
    await openSnapshot([file]);
    return;
  }
  const outputMetadataJson = example.outputsPath ? await loadExampleMetadata(example.outputsPath) : undefined;
  await openFile(file, selectedEntry, outputMetadataJson);
}

async function loadExampleMetadata(path: string): Promise<string | undefined> {
  const response = await fetch(resolveAssetPath(path));
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
  return response.text();
}

async function openFile(
  file: File,
  selectedEntry?: Element | null,
  outputMetadataJson?: string,
  snapshotPath?: string,
): Promise<void> {
  try {
    const parsed = await loadNbtFile(file);
    const structure = toStructureModel(parsed.root);
    simulation = undefined;
    currentNbtBytes = parsed.bytes;
    currentRoot = parsed.root;
    currentOutputMetadataJson = outputMetadataJson;
    graphDot = undefined;
    graphTab = 'world';
    graphWorldModeValue = 'raw';
    graphLogicModeValue = 'raw';
    graphHighLevelLogic = false;
    currentSnapshotPath = snapshotPath;
    isolatedSnapshotBoxId = undefined;
    isolatedSnapshotRouteId = undefined;
    viewer.setIsolatedBoundingBox(undefined);
    viewer.setIsolatedBlockPositions(undefined);
    viewer.setSelectedRouteId(undefined);

    markSelectedFile(selectedEntry);

    if (structure) {
      await viewer.setStructure(structure);
      renderSwitches(structure);
      viewerEmpty.classList.add('hidden');
      inspector.textContent = [
        `size: ${structure.size.join(' x ')}`,
        `palette: ${structure.palette.length}`,
        `blocks: ${structure.blocks.length}`,
      ].join('\n');
      renderTrace([], [], emptyWaveform, undefined);
    } else {
      viewerEmpty.classList.remove('hidden');
      inspector.textContent = 'This NBT file does not look like a Minecraft structure file.';
      renderSwitches();
      renderTrace([], [], emptyWaveform, undefined);
    }
    updateSnapshotBoxes();
    updateSnapshotRoutes();
  } catch (error) {
    simulation = undefined;
    currentNbtBytes = undefined;
    currentRoot = undefined;
    currentOutputMetadataJson = undefined;
    graphDot = undefined;
    graphTab = 'world';
    graphWorldModeValue = 'raw';
    graphLogicModeValue = 'raw';
    graphHighLevelLogic = false;
    currentSnapshotPath = snapshotPath;
    selectedBlock = undefined;
    toggleSwitchButton.classList.add('hidden');
    viewerEmpty.classList.remove('hidden');
    inspector.textContent = error instanceof Error ? error.message : String(error);
    renderSwitches();
    renderTrace([], [], emptyWaveform, undefined);
    updateSnapshotBoxes();
    updateSnapshotRoutes();
  }
}

function markSelectedFile(selectedEntry?: Element | null): void {
  filesList.querySelectorAll('.file-entry.selected').forEach(entry => {
    entry.classList.remove('selected');
  });
  selectedEntry?.classList.add('selected');
}

function renderSelection(block: StructureBlock | undefined): void {
  selectedBlock = block;
  const switchInfo = block && simulation?.getSwitch(block);
  const isLever = block?.palette.name === 'minecraft:lever';
  toggleSwitchButton.classList.toggle('hidden', !isLever);
  toggleSwitchButton.textContent = (switchInfo?.is_on ?? getLeverPowered(block)) ? 'Turn Off' : 'Turn On';

  if (!block) {
    inspector.textContent = 'No block selected.';
    return;
  }

  inspector.textContent = stringifyNbt({
    pos: block.pos,
    state: block.state,
    name: block.palette.name,
    properties: block.palette.properties,
    nbt: block.nbt,
  });
}

function renderSwitches(structure?: StructureModel): void {
  switchesList.replaceChildren();
  const switches = structure?.blocks.filter(block => block.palette.name === 'minecraft:lever') ?? [];
  switchesCount.textContent = switches.length === 0 ? 'No switches' : `${switches.length} switches`;
  switchesActions.classList.toggle('hidden', switches.length === 0);

  if (switches.length === 0) {
    switchesPanel.open = !structure;
    switchesList.className = 'switches-list empty';
    switchesList.textContent = structure ? 'No switches in this NBT file.' : 'Open an NBT file to control switches.';
    return;
  }

  switchesPanel.open = true;
  switchesList.className = 'switches-list';
  switches.forEach((block, index) => {
    const row = document.createElement('button');
    row.className = 'switch-entry';
    row.type = 'button';
    if (selectedBlock && samePos(block.pos, selectedBlock.pos)) row.classList.add('selected');

    const label = document.createElement('span');
    label.className = 'switch-entry-label';
    label.textContent = `#${index + 1}  ${block.pos.join(',')}`;

    const state = document.createElement('span');
    state.className = 'switch-entry-state';
    state.textContent = (simulation?.getSwitch(block)?.is_on ?? getLeverPowered(block)) ? 'On' : 'Off';

    row.append(label, state);
    row.addEventListener('click', () => {
      void toggleSwitchBlock(block).catch(error => {
        renderSimulationError(error);
      });
    });
    switchesList.append(row);
  });
}

function getLeverPowered(block: StructureBlock | undefined): boolean {
  return block?.palette.properties?.powered === 'true';
}

function samePos(a: [number, number, number], b: [number, number, number]): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function sameWaveformSignal(a: WaveformSignal, b: WaveformSignal): boolean {
  return samePos(a.position, b.position) && a.kind === b.kind && a.property === b.property;
}

function collectTraceCycles(trace: TraceEntry[], waveform: Waveform): number[] {
  return Array.from(new Set([...trace.map(entry => entry.cycle), ...waveform.cycles])).sort((a, b) => a - b);
}

function getActiveTrace(): TraceEntry[] {
  return traceShowActualCycles ? historyTrace : currentTrace;
}

function getActiveSnapshots(): SnapshotInfo[] {
  return traceShowActualCycles ? historySnapshots : currentSnapshots;
}

function getActiveWaveform(): Waveform {
  return traceShowActualCycles ? historyWaveform : currentWaveform;
}

function pruneSelectedWaveformSignal(): void {
  const activeWaveform = getActiveWaveform();
  if (selectedWaveformSignal && !activeWaveform.signals.some(signal => sameWaveformSignal(signal, selectedWaveformSignal!))) {
    selectedWaveformSignal = undefined;
  }
}

function signalHasWaveformChange(signal: WaveformSignal): boolean {
  if (signal.values.length <= 1) return false;

  const firstValue = signal.values[0] ?? 0;
  return signal.values.some(value => value !== firstValue);
}

function getVisibleWaveformSignals(): WaveformSignal[] {
  const activeWaveform = getActiveWaveform();
  if (!waveformChangedOnly) return activeWaveform.signals;

  return activeWaveform.signals.filter(signalHasWaveformChange);
}

function toTraceDisplayCycle(cycle: number): number {
  if (traceShowActualCycles) return cycle;

  return Math.max(1, cycle - traceCycleDisplayOffset);
}

function getTraceAxisCycles(): number[] {
  if (!traceShowActualCycles) return traceCycles;
  if (historyTraceCycles.length === 0) return [];

  const latestCycle = historyTraceCycles[historyTraceCycles.length - 1];
  return Array.from({ length: latestCycle }, (_, index) => index + 1);
}

function findTraceAxisIndexForCycle(cycle: number | undefined, axisCycles = getTraceAxisCycles()): number {
  if (cycle === undefined || axisCycles.length === 0) return -1;

  if (traceShowActualCycles && cycle >= 1 && cycle <= axisCycles[axisCycles.length - 1]) {
    return cycle - 1;
  }

  const exactIndex = axisCycles.indexOf(cycle);
  if (exactIndex >= 0) return exactIndex;

  const nextIndex = axisCycles.findIndex(axisCycle => axisCycle >= cycle);
  return nextIndex >= 0 ? nextIndex : axisCycles.length - 1;
}

function selectedTraceCycle(): number | undefined {
  const axisCycles = getTraceAxisCycles();
  return axisCycles[Number(traceCycleInput.value)];
}

function updateTraceCycleControls(selectedIndex: number): number {
  const axisCycles = getTraceAxisCycles();
  const disabled = axisCycles.length === 0;
  const safeIndex = disabled ? -1 : Math.max(0, Math.min(axisCycles.length - 1, selectedIndex));
  traceCycleInput.max = String(Math.max(0, axisCycles.length - 1));
  traceCycleInput.value = String(Math.max(0, safeIndex));
  traceCycleInput.disabled = disabled;
  tracePrevButton.disabled = disabled;
  traceNextButton.disabled = disabled;
  updateTraceCycleModeControl();
  return safeIndex;
}

function updateTraceCycleModeControl(): void {
  traceShowActualCyclesInput.checked = traceShowActualCycles;
  traceShowActualCyclesInput.disabled = !traceSimulationEnabled || (traceCycles.length === 0 && historyTraceCycles.length === 0);
}

function updateWaveformFilterControl(): void {
  const activeWaveform = getActiveWaveform();
  waveformChangedOnlyInput.checked = waveformChangedOnly;
  waveformChangedOnlyInput.disabled = !traceSimulationEnabled || activeWaveform.signals.length === 0;
}

function getCurrentWindowStartCycle(): number | undefined {
  return traceCycles[0];
}

function getCurrentWindowEndCycle(): number | undefined {
  return traceCycles[traceCycles.length - 1];
}

function getTraceAnimationStartIndex(axisCycles: number[]): number {
  if (axisCycles.length === 0) return -1;
  const startCycle = traceShowActualCycles ? getCurrentWindowStartCycle() : axisCycles[0];
  return findTraceAxisIndexForCycle(startCycle ?? axisCycles[0], axisCycles);
}

function getTraceAnimationTargetIndex(axisCycles: number[]): number {
  if (axisCycles.length === 0) return -1;
  const targetCycle = traceShowActualCycles ? getCurrentWindowEndCycle() : axisCycles[axisCycles.length - 1];
  return findTraceAxisIndexForCycle(targetCycle ?? axisCycles[axisCycles.length - 1], axisCycles);
}

function renderSimulationError(error: unknown): void {
  if (error instanceof NbtSimulationError) {
    inspector.textContent = error.message;
    renderTrace(error.trace, error.snapshots, error.waveform, currentRoot, {
      open: true,
      history: {
        trace: error.historyTrace,
        snapshots: error.historySnapshots,
        waveform: error.historyWaveform,
      },
    });
    return;
  }

  inspector.textContent = error instanceof Error ? error.message : String(error);
}

function renderTrace(
  trace: TraceEntry[],
  snapshots: SnapshotInfo[],
  waveform: Waveform,
  baseRoot: unknown,
  options: {
    animateTo?: 'last';
    history?: {
      trace: TraceEntry[];
      snapshots: SnapshotInfo[];
      waveform: Waveform;
    };
    open?: boolean;
    select?: 'first' | 'last';
  } = {},
): void {
  cancelTraceAnimation();
  currentTrace = trace;
  currentSnapshots = snapshots;
  currentWaveform = waveform;
  historyTrace = options.history?.trace ?? trace;
  historySnapshots = options.history?.snapshots ?? snapshots;
  historyWaveform = options.history?.waveform ?? waveform;
  pruneSelectedWaveformSignal();
  traceBaseRoot = baseRoot;
  traceCycles = collectTraceCycles(trace, waveform);
  historyTraceCycles = collectTraceCycles(historyTrace, historyWaveform);
  traceCycleDisplayOffset = traceCycles.length === 0 ? 0 : Math.max(0, traceCycles[0] - 1);
  traceCount.textContent =
    !traceSimulationEnabled
      ? 'Trace off'
      : trace.length === 0 && waveform.signals.length === 0
      ? 'No events'
      : `${trace.length} events / ${waveform.signals.length} signals`;
  updateTraceExpandAvailability(
    traceSimulationEnabled &&
      (trace.length > 0 || waveform.signals.length > 0 || historyTrace.length > 0 || historyWaveform.signals.length > 0),
  );
  const axisCycles = getTraceAxisCycles();
  const selectedIndex = updateTraceCycleControls(
    options.animateTo === 'last'
      ? getTraceAnimationStartIndex(axisCycles)
      : options.select === 'last'
        ? axisCycles.length - 1
        : 0,
  );
  updateWaveformFilterControl();
  renderWaveformLabels();
  void renderTraceCycle(selectedIndex);
  if (traceSimulationEnabled && (options.open || trace.length > 0)) {
    tracePanel.open = true;
  }
  if (traceSimulationEnabled && options.animateTo === 'last') {
    startTraceAnimation(getTraceAnimationTargetIndex(axisCycles));
  }
}

function cancelTraceAnimation(): void {
  if (!traceAnimation) return;

  window.clearInterval(traceAnimation.timer);
  traceAnimation = undefined;
  traceAnimationToken += 1;
}

function startTraceAnimation(targetIndex: number): void {
  if (targetIndex <= Number(traceCycleInput.value)) return;

  const token = traceAnimationToken + 1;
  traceAnimationToken = token;
  traceAnimation = {
    token,
    timer: window.setInterval(() => {
      if (!traceAnimation || traceAnimation.token !== token) return;

      const currentIndex = Number(traceCycleInput.value);
      const nextIndex = Math.min(targetIndex, currentIndex + 1);
      void renderTraceCycle(nextIndex);
      if (nextIndex >= targetIndex) {
        cancelTraceAnimation();
      }
    }, TRACE_ANIMATION_INTERVAL_MS),
  };
}

async function renderTraceCycle(index: number): Promise<void> {
  const axisCycles = getTraceAxisCycles();
  if (index < 0 || axisCycles.length === 0) {
    traceCycleLabel.textContent = 'cycle -';
    traceOutput.textContent = traceSimulationEnabled ? 'Run a simulation to inspect events.' : 'Trace simulation is off.';
    renderWaveform(-1);
    viewer.setTraceHighlights([]);
    await restoreCurrentStructurePreview();
    return;
  }

  const safeIndex = Math.max(0, Math.min(axisCycles.length - 1, index));
  const cycle = axisCycles[safeIndex];
  const activeTrace = getActiveTrace();
  const activeSnapshots = getActiveSnapshots();
  const entries = activeTrace.filter(entry => entry.cycle === cycle);
  const selectedSignalPositions: Array<[number, number, number]> = selectedWaveformSignal
    ? [rustPosToRenderPos(selectedWaveformSignal.position)]
    : [];
  const positions = uniquePositions([...entries.map(entry => rustPosToRenderPos(entry.target_position)), ...selectedSignalPositions]);
  const snapshot = activeSnapshots.find(snapshot => snapshot.cycle === cycle);
  traceCycleInput.value = String(safeIndex);
  traceCycleLabel.textContent = `cycle ${toTraceDisplayCycle(cycle)} / ${entries.length} events`;
  traceOutput.textContent = formatTrace(entries);
  renderWaveform(safeIndex);
  scrollWaveformToTraceIndex(safeIndex);
  await renderTraceSnapshot(snapshot, positions);
}

async function renderTraceSnapshot(
  snapshot: SnapshotInfo | undefined,
  highlights: Array<[number, number, number]>,
): Promise<void> {
  if (snapshot) {
    const previewBaseRoot = traceShowActualCycles ? currentRoot : traceBaseRoot;
    const previewRoot = previewBaseRoot ? mergeSimulatedState(previewBaseRoot, snapshot.root) : snapshot.root;
    const structure = toStructureModel(previewRoot);
    if (structure) {
      isTracePreviewActive = true;
      await viewer.setStructure(structure, { preserveSelection: true, preserveView: true });
    }
  }

  viewer.setTraceHighlights(highlights);
}

function renderWaveformLabels(): void {
  const activeWaveform = getActiveWaveform();
  const visibleSignals = getVisibleWaveformSignals();
  waveformLabels.replaceChildren();
  waveformLabels.style.width = `${WAVEFORM_LABEL_WIDTH}px`;
  waveformLabels.style.minWidth = `${WAVEFORM_LABEL_WIDTH}px`;

  if (activeWaveform.signals.length === 0 || visibleSignals.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'waveform-empty';
    empty.textContent = activeWaveform.signals.length === 0 ? 'No waveform' : 'No changed signals';
    waveformLabels.append(empty);
    return;
  }

  const spacer = document.createElement('div');
  spacer.className = 'waveform-label-spacer';
  spacer.style.height = `${WAVEFORM_HEADER_HEIGHT}px`;
  waveformLabels.append(spacer);

  for (const signal of visibleSignals) {
    const button = document.createElement('button');
    button.className = 'waveform-label';
    button.type = 'button';
    button.title = signal.label;
    button.classList.toggle('selected', selectedWaveformSignal ? sameWaveformSignal(signal, selectedWaveformSignal) : false);

    const name = document.createElement('span');
    name.className = 'waveform-label-name';
    name.textContent = signal.label;

    const range = document.createElement('span');
    range.className = 'waveform-label-range';
    range.textContent = signal.max_value > 1 ? `0-${signal.max_value}` : '0/1';

    button.append(name, range);
    button.addEventListener('click', () => {
      focusWaveformSignal(signal);
      renderWaveform(Number(traceCycleInput.value));
      renderWaveformLabels();
    });
    waveformLabels.append(button);
  }
}

function focusWaveformSignal(signal: WaveformSignal): void {
  selectedWaveformSignal = signal;
  const renderPos = rustPosToRenderPos(signal.position);
  viewer.setTraceHighlights([renderPos]);

  const structure = toStructureModel(currentRoot);
  const block = structure?.blocks.find(item => samePos(item.pos, renderPos));
  if (block) {
    viewer.setSelectedBlock(block);
    renderSelection(block);
  }
}

function renderWaveform(selectedTraceIndex: number): void {
  const activeWaveform = getActiveWaveform();
  const axisCycles = getTraceAxisCycles();
  const visibleSignals = getVisibleWaveformSignals();
  const width = Math.max(1, axisCycles.length * WAVEFORM_CYCLE_WIDTH);
  const height = Math.max(WAVEFORM_HEADER_HEIGHT + WAVEFORM_ROW_HEIGHT, WAVEFORM_HEADER_HEIGHT + visibleSignals.length * WAVEFORM_ROW_HEIGHT);
  const pixelRatio = window.devicePixelRatio || 1;
  waveformCanvas.style.width = `${width}px`;
  waveformCanvas.style.height = `${height}px`;
  waveformCanvas.width = Math.ceil(width * pixelRatio);
  waveformCanvas.height = Math.ceil(height * pixelRatio);

  const context = waveformCanvas.getContext('2d');
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#12161a';
  context.fillRect(0, 0, width, height);

  drawWaveformHeader(context, width, axisCycles);
  if (activeWaveform.signals.length === 0 || axisCycles.length === 0) {
    context.fillStyle = '#7f8992';
    context.font = '12px "Cascadia Mono", Consolas, monospace';
    context.fillText('No waveform data', 10, WAVEFORM_HEADER_HEIGHT + 16);
    return;
  }

  if (visibleSignals.length === 0) {
    context.fillStyle = '#7f8992';
    context.font = '12px "Cascadia Mono", Consolas, monospace';
    context.fillText('No changed signals', 10, WAVEFORM_HEADER_HEIGHT + 16);
    return;
  }

  const waveformCycleIndexByCycle = new Map(activeWaveform.cycles.map((cycle, index) => [cycle, index]));
  visibleSignals.forEach((signal, signalIndex) => {
    drawWaveformRow(context, signal, signalIndex, activeWaveform, waveformCycleIndexByCycle, axisCycles);
  });

  if (selectedTraceIndex >= 0 && selectedTraceIndex < axisCycles.length) {
    const x = selectedTraceIndex * WAVEFORM_CYCLE_WIDTH;
    context.fillStyle = 'rgb(211 50 50 / 18%)';
    context.fillRect(x, 0, WAVEFORM_CYCLE_WIDTH, height);
    context.strokeStyle = '#ff6b6b';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x + 0.5, 0);
    context.lineTo(x + 0.5, height);
    context.stroke();
  }
}

function scrollWaveformToTraceIndex(selectedTraceIndex: number): void {
  if (selectedTraceIndex < 0 || waveformScroll.clientWidth <= 0) return;

  const cycleLeft = selectedTraceIndex * WAVEFORM_CYCLE_WIDTH;
  const cycleRight = cycleLeft + WAVEFORM_CYCLE_WIDTH;
  const padding = WAVEFORM_CYCLE_WIDTH * 2;
  const visibleLeft = waveformScroll.scrollLeft;
  const visibleRight = visibleLeft + waveformScroll.clientWidth;

  if (cycleLeft < visibleLeft + padding) {
    waveformScroll.scrollLeft = Math.max(0, cycleLeft - padding);
  } else if (cycleRight > visibleRight - padding) {
    waveformScroll.scrollLeft = Math.max(0, cycleRight - waveformScroll.clientWidth + padding);
  }
}

function drawWaveformHeader(context: CanvasRenderingContext2D, width: number, axisCycles: number[]): void {
  context.fillStyle = '#161b20';
  context.fillRect(0, 0, width, WAVEFORM_HEADER_HEIGHT);
  context.fillStyle = '#9ea9b3';
  context.font = '10px "Cascadia Mono", Consolas, monospace';
  context.textBaseline = 'middle';

  axisCycles.forEach((cycle, index) => {
    const x = index * WAVEFORM_CYCLE_WIDTH;
    context.fillText(String(toTraceDisplayCycle(cycle)), x + 4, WAVEFORM_HEADER_HEIGHT / 2);
    context.strokeStyle = 'rgb(154 164 173 / 14%)';
    context.beginPath();
    context.moveTo(x + 0.5, 0);
    context.lineTo(x + 0.5, WAVEFORM_HEADER_HEIGHT);
    context.stroke();
  });
}

function drawWaveformRow(
  context: CanvasRenderingContext2D,
  signal: WaveformSignal,
  signalIndex: number,
  waveform: Waveform,
  waveformCycleIndexByCycle: Map<number, number>,
  axisCycles: number[],
): void {
  const y = WAVEFORM_HEADER_HEIGHT + signalIndex * WAVEFORM_ROW_HEIGHT;
  context.fillStyle = signalIndex % 2 === 0 ? '#101419' : '#151a20';
  context.fillRect(0, y, axisCycles.length * WAVEFORM_CYCLE_WIDTH, WAVEFORM_ROW_HEIGHT);

  if (selectedWaveformSignal && sameWaveformSignal(signal, selectedWaveformSignal)) {
    context.fillStyle = 'rgb(211 50 50 / 15%)';
    context.fillRect(0, y, axisCycles.length * WAVEFORM_CYCLE_WIDTH, WAVEFORM_ROW_HEIGHT);
  }

  context.strokeStyle = 'rgb(154 164 173 / 12%)';
  context.beginPath();
  context.moveTo(0, y + WAVEFORM_ROW_HEIGHT + 0.5);
  context.lineTo(axisCycles.length * WAVEFORM_CYCLE_WIDTH, y + WAVEFORM_ROW_HEIGHT + 0.5);
  context.stroke();

  if (signal.max_value <= 1) {
    drawDigitalSignal(context, signal, y, waveform, waveformCycleIndexByCycle, axisCycles);
  } else {
    drawStrengthSignal(context, signal, y, waveform, waveformCycleIndexByCycle, axisCycles);
  }
}

function waveformValueAtCycle(
  signal: WaveformSignal,
  cycle: number,
  waveform: Waveform,
  waveformCycleIndexByCycle: Map<number, number>,
): number {
  const exactIndex = waveformCycleIndexByCycle.get(cycle);
  if (exactIndex !== undefined) return signal.values[exactIndex] ?? 0;
  if (!traceShowActualCycles || waveform.cycles.length === 0) return 0;

  let low = 0;
  let high = waveform.cycles.length - 1;
  let nearestIndex = 0;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (waveform.cycles[mid] <= cycle) {
      nearestIndex = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return signal.values[nearestIndex] ?? 0;
}

function drawDigitalSignal(
  context: CanvasRenderingContext2D,
  signal: WaveformSignal,
  y: number,
  waveform: Waveform,
  waveformCycleIndexByCycle: Map<number, number>,
  axisCycles: number[],
): void {
  context.strokeStyle = '#f45d5d';
  context.lineWidth = 2;
  context.beginPath();

  let previousY: number | undefined;
  axisCycles.forEach((cycle, index) => {
    const x = index * WAVEFORM_CYCLE_WIDTH;
    const value = waveformValueAtCycle(signal, cycle, waveform, waveformCycleIndexByCycle);
    const signalY = y + (value > 0 ? 7 : WAVEFORM_ROW_HEIGHT - 7);

    if (previousY === undefined) {
      context.moveTo(x, signalY);
    } else if (previousY !== signalY) {
      context.lineTo(x, previousY);
      context.lineTo(x, signalY);
    } else {
      context.lineTo(x, signalY);
    }
    context.lineTo(x + WAVEFORM_CYCLE_WIDTH, signalY);
    previousY = signalY;
  });
  context.stroke();
}

function drawStrengthSignal(
  context: CanvasRenderingContext2D,
  signal: WaveformSignal,
  y: number,
  waveform: Waveform,
  waveformCycleIndexByCycle: Map<number, number>,
  axisCycles: number[],
): void {
  context.textBaseline = 'middle';
  context.font = '10px "Cascadia Mono", Consolas, monospace';

  axisCycles.forEach((cycle, index) => {
    const value = waveformValueAtCycle(signal, cycle, waveform, waveformCycleIndexByCycle);
    const normalized = Math.max(0, Math.min(1, value / Math.max(1, signal.max_value)));
    const x = index * WAVEFORM_CYCLE_WIDTH;
    const fillHeight = Math.max(2, Math.round((WAVEFORM_ROW_HEIGHT - 7) * normalized));
    const red = Math.round(96 + normalized * 159);
    const green = Math.round(58 + normalized * 82);
    context.fillStyle = value > 0 ? `rgb(${red} ${green} 72 / 82%)` : 'rgb(255 255 255 / 8%)';
    context.fillRect(x + 3, y + WAVEFORM_ROW_HEIGHT - fillHeight - 3, WAVEFORM_CYCLE_WIDTH - 6, fillHeight);
    context.fillStyle = value > 0 ? '#f7fbfd' : '#7f8992';
    context.fillText(String(value), x + 8, y + WAVEFORM_ROW_HEIGHT / 2);
  });
}

async function restoreCurrentStructurePreview(): Promise<void> {
  if (!isTracePreviewActive || !currentRoot) return;

  const structure = toStructureModel(currentRoot);
  if (structure) {
    await viewer.setStructure(structure, { preserveSelection: true, preserveView: true });
  }
  isTracePreviewActive = false;
}

function uniquePositions(positions: Array<[number, number, number]>): Array<[number, number, number]> {
  const seen = new Set<string>();
  return positions.filter(pos => {
    const key = pos.join(',');
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function rustPosToRenderPos(pos: [number, number, number]): [number, number, number] {
  return [pos[1], pos[2], pos[0]];
}

function formatTrace(trace: TraceEntry[]): string {
  const shown = trace.slice(0, 500);
  const lines = shown
    .map(entry => {
      const id = entry.event_id ?? '-';
      const pos = entry.target_position.join(',');
      return [
        `#${id}`,
        `cycle=${toTraceDisplayCycle(entry.cycle)}`,
        entry.event_type,
        `target=${pos}`,
        `dir=${entry.direction}`,
        `block=${entry.block_before}`,
        `queue=${entry.current_queue_len}/${entry.next_queue_len}`,
      ].join('  ');
    });

  if (trace.length > shown.length) {
    lines.push(`... ${trace.length - shown.length} more events in this cycle`);
  }

  return lines.join('\n');
}

function mergeSimulatedState(originalRoot: unknown, simulatedRoot: unknown): unknown {
  const original = cloneRoot(originalRoot);
  const originalRecord = asRecord(original);
  const simulatedRecord = asRecord(simulatedRoot);
  if (!originalRecord || !simulatedRecord) return simulatedRoot;

  const originalPalette = Array.isArray(originalRecord.palette) ? originalRecord.palette : undefined;
  const originalBlocks = Array.isArray(originalRecord.blocks) ? originalRecord.blocks : undefined;
  const simulatedPalette = Array.isArray(simulatedRecord.palette) ? simulatedRecord.palette : undefined;
  const simulatedBlocks = Array.isArray(simulatedRecord.blocks) ? simulatedRecord.blocks : undefined;
  if (!originalPalette || !originalBlocks || !simulatedPalette || !simulatedBlocks) return simulatedRoot;

  const simulatedByPosition = new Map<string, StructurePaletteEntry>();
  for (const rawBlock of simulatedBlocks) {
    const block = asRecord(rawBlock);
    const pos = readPosition(block);
    const state = readState(block);
    const palette = state === undefined ? undefined : readPaletteEntry(simulatedPalette[state]);
    if (pos && palette) simulatedByPosition.set(pos.join(','), palette);
  }

  const paletteCache = new Map<string, number>();

  for (const rawBlock of originalBlocks) {
    const block = asRecord(rawBlock);
    const pos = readPosition(block);
    const state = readState(block);
    if (!block || !pos || state === undefined) continue;

    const originalEntry = readPaletteEntry(originalPalette[state]);
    const simulatedEntry = simulatedByPosition.get(pos.join(','));
    const mergedEntry = mergePaletteEntry(originalEntry, simulatedEntry);
    if (!mergedEntry) continue;

    const key = `${mergedEntry.name}\0${JSON.stringify(mergedEntry.properties)}`;
    let mergedState = paletteCache.get(key);
    if (mergedState === undefined) {
      mergedState = originalPalette.length;
      paletteCache.set(key, mergedState);
      originalPalette.push({
        Name: mergedEntry.name,
        Properties: mergedEntry.properties,
      });
    }
    block.state = mergedState;
  }

  return original;
}

function mergePaletteEntry(
  original: StructurePaletteEntry | undefined,
  simulated: StructurePaletteEntry | undefined,
): StructurePaletteEntry | undefined {
  if (!original || !simulated) return undefined;

  const properties = { ...original.properties };
  switch (original.name) {
    case 'minecraft:lever':
      copyProperty(properties, simulated.properties, 'powered');
      return { name: original.name, properties };
    case 'minecraft:redstone_wire':
      copyProperty(properties, simulated.properties, 'power');
      return { name: original.name, properties };
    case 'minecraft:redstone_torch':
    case 'minecraft:redstone_wall_torch':
      copyProperty(properties, simulated.properties, 'lit');
      return { name: original.name, properties };
    case 'minecraft:repeater':
      copyProperty(properties, simulated.properties, 'powered');
      copyProperty(properties, simulated.properties, 'locked');
      return { name: original.name, properties };
    default:
      return undefined;
  }
}

function copyProperty(target: Record<string, string>, source: Record<string, string>, name: string): void {
  if (name in source) target[name] = source[name];
}

function cloneRoot(root: unknown): unknown {
  if (typeof structuredClone === 'function') return structuredClone(root);
  return JSON.parse(JSON.stringify(root)) as unknown;
}

function readPaletteEntry(value: unknown): StructurePaletteEntry | undefined {
  const entry = asRecord(value);
  if (!entry) return undefined;

  const properties = asRecord(entry.Properties);
  return {
    name: String(entry.Name ?? 'minecraft:air'),
    properties: Object.fromEntries(
      Object.entries(properties ?? {}).map(([key, propValue]) => [key, String(propValue)]),
    ),
  };
}

function readPosition(block: Record<string, unknown> | undefined): [number, number, number] | undefined {
  const pos = block?.pos;
  if (!Array.isArray(pos) || pos.length < 3) return undefined;

  const values = pos.slice(0, 3).map(Number);
  return values.every(Number.isFinite) ? [values[0], values[1], values[2]] : undefined;
}

function readState(block: Record<string, unknown> | undefined): number | undefined {
  const state = Number(block?.state);
  return Number.isInteger(state) ? state : undefined;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function isSupportedFile(file: File): boolean {
  return /\.(nbt|dat|dat_old|schem|schematic|litematic|mcstructure)$/i.test(file.name);
}

function getDisplayPath(file: Pick<File, 'name'>): string {
  return 'webkitRelativePath' in file && typeof file.webkitRelativePath === 'string' && file.webkitRelativePath
    ? file.webkitRelativePath
    : file.name;
}
