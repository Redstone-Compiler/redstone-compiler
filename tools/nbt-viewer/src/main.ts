import './styles.css';
import type { Viz } from '@viz-js/viz';
import { loadNbtFile, stringifyNbt } from './nbt/loadNbt';
import { toStructureModel } from './nbt/toStructure';
import { StructureViewer } from './render/StructureViewer';
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
type GraphEdgeInfo = {
  element: SVGGElement;
  source: string;
  target: string;
};
type ExampleFile = {
  name: string;
  path: string;
  size: number;
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

function resolveAssetPath(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, window.location.origin).href;
}

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="app-shell">
    <section id="drop-zone" class="workspace">
      <section class="viewer-panel">
        <canvas id="structure-canvas"></canvas>
        <div class="floating-actions">
          <div class="file-actions-row">
            <label class="file-button">
              Open Folder
              <input id="folder-input" type="file" multiple />
            </label>
            <label class="file-button">
              Open NBT
              <input id="file-input" type="file" accept=".nbt,.dat,.schem,.schematic,.litematic,.mcstructure" />
            </label>
          </div>
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
          </div>
          <div id="switches-list" class="switches-list empty">Open an NBT file to control switches.</div>
        </details>
        <details id="files-panel" class="floating-panel files-panel">
          <summary>
            <span>Files</span>
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
              <pre id="trace-output">Run a simulation to inspect events.</pre>
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
          <strong>Selected NBT</strong>
          <button id="close-selected-nbt" class="panel-action" type="button">Close</button>
        </header>
        <div id="selected-nbt-status" class="graph-status">Select world graph nodes to open focused NBT.</div>
        <div class="selected-nbt-viewer">
          <canvas id="selected-nbt-canvas"></canvas>
        </div>
      </div>
    </dialog>
  </main>
`;

const input = document.querySelector<HTMLInputElement>('#file-input')!;
const folderInput = document.querySelector<HTMLInputElement>('#folder-input')!;
const dropZone = document.querySelector<HTMLElement>('#drop-zone')!;
const filesPanel = document.querySelector<HTMLDetailsElement>('#files-panel')!;
const filesList = document.querySelector<HTMLElement>('#files-list')!;
const filesCount = document.querySelector<HTMLElement>('#files-count')!;
const canvas = document.querySelector<HTMLCanvasElement>('#structure-canvas')!;
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
const openGraphsButton = document.querySelector<HTMLButtonElement>('#open-graphs')!;
const closeGraphsButton = document.querySelector<HTMLButtonElement>('#close-graphs')!;
const graphDialog = document.querySelector<HTMLDialogElement>('#graph-dialog')!;
const graphWorldTab = document.querySelector<HTMLButtonElement>('#graph-world-tab')!;
const graphLogicTab = document.querySelector<HTMLButtonElement>('#graph-logic-tab')!;
const graphWorldMode = document.querySelector<HTMLElement>('#graph-world-mode')!;
const graphWorldRawButton = document.querySelector<HTMLButtonElement>('#graph-world-raw')!;
const graphWorldFoldedButton = document.querySelector<HTMLButtonElement>('#graph-world-folded')!;
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
const selectedGraphShowTagsInput = document.querySelector<HTMLInputElement>('#selected-graph-show-tags')!;
const selectedGraphZoomOutButton = document.querySelector<HTMLButtonElement>('#selected-graph-zoom-out')!;
const selectedGraphZoomResetButton = document.querySelector<HTMLButtonElement>('#selected-graph-zoom-reset')!;
const selectedGraphZoomInButton = document.querySelector<HTMLButtonElement>('#selected-graph-zoom-in')!;
const selectedGraphStatus = document.querySelector<HTMLElement>('#selected-graph-status')!;
const selectedGraphOutput = document.querySelector<HTMLElement>('#selected-graph-output')!;
const selectedNbtDialog = document.querySelector<HTMLDialogElement>('#selected-nbt-dialog')!;
const closeSelectedNbtButton = document.querySelector<HTMLButtonElement>('#close-selected-nbt')!;
const selectedNbtStatus = document.querySelector<HTMLElement>('#selected-nbt-status')!;
const selectedNbtCanvas = document.querySelector<HTMLCanvasElement>('#selected-nbt-canvas')!;

const viewer = new StructureViewer(canvas);
viewer.setSelectionHandler(renderSelection);
const selectedNbtViewer = new StructureViewer(selectedNbtCanvas);

let simulation: NbtSimulation | undefined;
let selectedBlock: StructureBlock | undefined;
let currentNbtBytes: Uint8Array | undefined;
let currentRoot: unknown;
let currentTrace: TraceEntry[] = [];
let traceCycles: number[] = [];
let currentSnapshots: SnapshotInfo[] = [];
let currentWaveform: Waveform = emptyWaveform;
let selectedWaveformSignal: WaveformSignal | undefined;
let waveformChangedOnly = false;
let isTraceExpanded = false;
let traceBaseRoot: unknown;
let isTracePreviewActive = false;
let traceAnimation: TraceAnimation | undefined;
let traceAnimationToken = 0;
let graphDot: GraphDotInfo | undefined;
let graphTab: GraphTab = 'world';
let graphWorldModeValue: GraphWorldMode = 'raw';
let graphShowTags = true;
let vizPromise: Promise<Viz> | undefined;
let graphMinimapScale = 1;
let isDraggingGraphMinimap = false;
let graphZoom = 1;
let selectedGraphNode: string | undefined;
let selectedGraphDot: GraphDotInfo | undefined;
let selectedGraphTab: GraphTab = 'world';
let selectedGraphWorldModeValue: GraphWorldMode = 'raw';
let selectedGraphShowTags = true;
let selectedGraphZoom = 1;

folderInput.setAttribute('webkitdirectory', '');
folderInput.setAttribute('directory', '');

void loadExamples();

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (file) void openFile(file);
});

folderInput.addEventListener('change', () => {
  renderFileBrowser(Array.from(folderInput.files ?? []));
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

traceCycleInput.addEventListener('input', () => {
  cancelTraceAnimation();
  void renderTraceCycle(Number(traceCycleInput.value));
});

tracePrevButton.addEventListener('click', () => {
  cancelTraceAnimation();
  traceCycleInput.value = String(Math.max(0, Number(traceCycleInput.value) - 1));
  void renderTraceCycle(Number(traceCycleInput.value));
});

traceNextButton.addEventListener('click', () => {
  cancelTraceAnimation();
  traceCycleInput.value = String(Math.min(traceCycles.length - 1, Number(traceCycleInput.value) + 1));
  void renderTraceCycle(Number(traceCycleInput.value));
});

waveformCanvas.addEventListener('click', event => {
  if (traceCycles.length === 0) return;

  cancelTraceAnimation();
  const rect = waveformCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top + waveformScroll.scrollTop;
  const cycleIndex = Math.max(0, Math.min(traceCycles.length - 1, Math.floor(x / WAVEFORM_CYCLE_WIDTH)));
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
  renderWaveformLabels();
  renderWaveform(Number(traceCycleInput.value));
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
  renderWaveform(Number(traceCycleInput.value));
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
      graphDot = await NbtSimulation.graphDot(currentNbtBytes);
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
      ? 'Logic Graph'
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
      ? 'Logic Graph'
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
  selectedGraphShowTags = graphShowTags;
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
      ? 'Selected Logic Graph'
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
  selectedGraphWorldRawButton.classList.toggle('active', selectedGraphWorldModeValue === 'raw');
  selectedGraphWorldFoldedButton.classList.toggle('active', selectedGraphWorldModeValue === 'folded');
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
  graphWorldRawButton.classList.toggle('active', graphWorldModeValue === 'raw');
  graphWorldFoldedButton.classList.toggle('active', graphWorldModeValue === 'folded');
  graphShowTagsInput.checked = graphShowTags;
}

async function toggleSelectedSwitch(): Promise<void> {
  if (!selectedBlock || !currentNbtBytes) return;

  await toggleSwitchBlock(selectedBlock);
}

async function toggleSwitchBlock(block: StructureBlock): Promise<void> {
  if (!currentNbtBytes) return;

  simulation ??= await NbtSimulation.create(currentNbtBytes);

  const selectedPos = block.pos;
  const baseRoot = currentRoot;
  const nextRoot = simulation.toggleSwitch(block);
  if (!nextRoot) return;

  await applySimulatedRoot(simulation, nextRoot, baseRoot, selectedPos);
}

async function setAllSwitches(isOn: boolean): Promise<void> {
  if (!currentNbtBytes) return;

  const structure = toStructureModel(currentRoot);
  const switches = structure?.blocks.filter(block => block.palette.name === 'minecraft:lever') ?? [];
  if (switches.length === 0) return;

  simulation ??= await NbtSimulation.create(currentNbtBytes);

  const selectedPos = selectedBlock?.pos;
  const baseRoot = currentRoot;
  let nextRoot: unknown | undefined;

  for (const block of switches) {
    nextRoot = simulation.setSwitch(block, isOn) ?? nextRoot;
  }

  if (!nextRoot) return;

  await applySimulatedRoot(simulation, nextRoot, baseRoot, selectedPos);
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
  renderTrace(activeSimulation.trace(), activeSimulation.snapshots(), activeSimulation.waveform(), baseRoot, {
    animateTo: 'last',
  });
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
    renderFileBrowser(dropped.files);
  } else if (dropped.files[0]) {
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

function renderFileBrowser(files: File[]): void {
  const nbtFiles = files
    .filter(isSupportedFile)
    .sort((a, b) => getDisplayPath(a).localeCompare(getDisplayPath(b)));

  filesList.replaceChildren();
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
    if (examples[0]) {
      await openExample(examples[0], filesList.querySelector('.file-entry'));
    }
  } catch (error) {
    filesList.classList.add('empty');
    filesCount.textContent = 'No examples';
    filesPanel.open = true;
    filesList.textContent = error instanceof Error ? error.message : String(error);
  }
}

function renderExampleBrowser(examples: ExampleFile[]): void {
  filesList.replaceChildren();
  filesList.classList.toggle('empty', examples.length === 0);
  filesCount.textContent = examples.length === 0 ? 'No NBT files' : `${examples.length} files`;
  filesPanel.open = true;

  if (examples.length === 0) {
    filesList.textContent = 'No supported NBT files found.';
    return;
  }

  for (const example of examples) {
    const button = document.createElement('button');
    button.className = 'file-entry';
    button.type = 'button';
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
}

async function openExample(example: ExampleFile, selectedEntry?: Element | null): Promise<void> {
  const response = await fetch(resolveAssetPath(example.path));
  if (!response.ok) throw new Error(`Failed to load ${example.path}: ${response.status}`);
  const file = new File([await response.arrayBuffer()], example.name, { type: 'application/octet-stream' });
  await openFile(file, selectedEntry);
}

async function openFile(file: File, selectedEntry?: Element | null): Promise<void> {
  try {
    const parsed = await loadNbtFile(file);
    const structure = toStructureModel(parsed.root);
    simulation = undefined;
    currentNbtBytes = parsed.bytes;
    currentRoot = parsed.root;
    graphDot = undefined;
    graphTab = 'world';
    graphWorldModeValue = 'raw';

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
  } catch (error) {
    simulation = undefined;
    currentNbtBytes = undefined;
    currentRoot = undefined;
    graphDot = undefined;
    graphTab = 'world';
    graphWorldModeValue = 'raw';
    selectedBlock = undefined;
    toggleSwitchButton.classList.add('hidden');
    viewerEmpty.classList.remove('hidden');
    inspector.textContent = error instanceof Error ? error.message : String(error);
    renderSwitches();
    renderTrace([], [], emptyWaveform, undefined);
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

function signalHasWaveformChange(signal: WaveformSignal): boolean {
  if (signal.values.length <= 1) return false;

  const firstValue = signal.values[0] ?? 0;
  return signal.values.some(value => value !== firstValue);
}

function getVisibleWaveformSignals(): WaveformSignal[] {
  if (!waveformChangedOnly) return currentWaveform.signals;

  return currentWaveform.signals.filter(signalHasWaveformChange);
}

function updateWaveformFilterControl(): void {
  waveformChangedOnlyInput.checked = waveformChangedOnly;
  waveformChangedOnlyInput.disabled = currentWaveform.signals.length === 0;
}

function renderSimulationError(error: unknown): void {
  if (error instanceof NbtSimulationError) {
    inspector.textContent = error.message;
    renderTrace(error.trace, error.snapshots, error.waveform, currentRoot, { open: true });
    return;
  }

  inspector.textContent = error instanceof Error ? error.message : String(error);
}

function renderTrace(
  trace: TraceEntry[],
  snapshots: SnapshotInfo[],
  waveform: Waveform,
  baseRoot: unknown,
  options: { animateTo?: 'last'; open?: boolean; select?: 'first' | 'last' } = {},
): void {
  cancelTraceAnimation();
  currentTrace = trace;
  currentSnapshots = snapshots;
  currentWaveform = waveform;
  if (selectedWaveformSignal && !waveform.signals.some(signal => sameWaveformSignal(signal, selectedWaveformSignal!))) {
    selectedWaveformSignal = undefined;
  }
  traceBaseRoot = baseRoot;
  traceCycles = Array.from(new Set([...trace.map(entry => entry.cycle), ...waveform.cycles])).sort((a, b) => a - b);
  traceCount.textContent =
    trace.length === 0 && waveform.signals.length === 0
      ? 'No events'
      : `${trace.length} events / ${waveform.signals.length} signals`;
  updateTraceExpandAvailability(trace.length > 0 || waveform.signals.length > 0);
  traceCycleInput.max = String(Math.max(0, traceCycles.length - 1));
  const selectedIndex = options.animateTo === 'last' ? 0 : options.select === 'last' ? traceCycles.length - 1 : 0;
  traceCycleInput.value = String(Math.max(0, selectedIndex));
  traceCycleInput.disabled = traceCycles.length === 0;
  tracePrevButton.disabled = traceCycles.length === 0;
  traceNextButton.disabled = traceCycles.length === 0;
  updateWaveformFilterControl();
  renderWaveformLabels();
  void renderTraceCycle(traceCycles.length === 0 ? -1 : selectedIndex);
  if (options.open || trace.length > 0) {
    tracePanel.open = true;
  }
  if (options.animateTo === 'last') {
    startTraceAnimation(traceCycles.length - 1);
  }
}

function cancelTraceAnimation(): void {
  if (!traceAnimation) return;

  window.clearInterval(traceAnimation.timer);
  traceAnimation = undefined;
  traceAnimationToken += 1;
}

function startTraceAnimation(targetIndex: number): void {
  if (targetIndex <= 0) return;

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
  if (index < 0 || traceCycles.length === 0) {
    traceCycleLabel.textContent = 'cycle -';
    traceOutput.textContent = 'Run a simulation to inspect events.';
    renderWaveform(-1);
    viewer.setTraceHighlights([]);
    await restoreCurrentStructurePreview();
    return;
  }

  const safeIndex = Math.max(0, Math.min(traceCycles.length - 1, index));
  const cycle = traceCycles[safeIndex];
  const entries = currentTrace.filter(entry => entry.cycle === cycle);
  const selectedSignalPositions: Array<[number, number, number]> = selectedWaveformSignal
    ? [rustPosToRenderPos(selectedWaveformSignal.position)]
    : [];
  const positions = uniquePositions([...entries.map(entry => rustPosToRenderPos(entry.target_position)), ...selectedSignalPositions]);
  const snapshot = currentSnapshots.find(snapshot => snapshot.cycle === cycle);
  traceCycleInput.value = String(safeIndex);
  traceCycleLabel.textContent = `cycle ${cycle} / ${entries.length} events`;
  traceOutput.textContent = formatTrace(entries);
  renderWaveform(safeIndex);
  await renderTraceSnapshot(snapshot, positions);
}

async function renderTraceSnapshot(
  snapshot: SnapshotInfo | undefined,
  highlights: Array<[number, number, number]>,
): Promise<void> {
  if (snapshot && traceBaseRoot) {
    const structure = toStructureModel(mergeSimulatedState(traceBaseRoot, snapshot.root));
    if (structure) {
      isTracePreviewActive = true;
      await viewer.setStructure(structure, { preserveSelection: true, preserveView: true });
    }
  }

  viewer.setTraceHighlights(highlights);
}

function renderWaveformLabels(): void {
  const visibleSignals = getVisibleWaveformSignals();
  waveformLabels.replaceChildren();
  waveformLabels.style.width = `${WAVEFORM_LABEL_WIDTH}px`;
  waveformLabels.style.minWidth = `${WAVEFORM_LABEL_WIDTH}px`;

  if (currentWaveform.signals.length === 0 || visibleSignals.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'waveform-empty';
    empty.textContent = currentWaveform.signals.length === 0 ? 'No waveform' : 'No changed signals';
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
  const visibleSignals = getVisibleWaveformSignals();
  const width = Math.max(1, traceCycles.length * WAVEFORM_CYCLE_WIDTH);
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

  drawWaveformHeader(context, width);
  if (currentWaveform.signals.length === 0 || traceCycles.length === 0) {
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

  const waveformCycleIndexByCycle = new Map(currentWaveform.cycles.map((cycle, index) => [cycle, index]));
  visibleSignals.forEach((signal, signalIndex) => {
    drawWaveformRow(context, signal, signalIndex, waveformCycleIndexByCycle);
  });

  if (selectedTraceIndex >= 0 && selectedTraceIndex < traceCycles.length) {
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

function drawWaveformHeader(context: CanvasRenderingContext2D, width: number): void {
  context.fillStyle = '#161b20';
  context.fillRect(0, 0, width, WAVEFORM_HEADER_HEIGHT);
  context.fillStyle = '#9ea9b3';
  context.font = '10px "Cascadia Mono", Consolas, monospace';
  context.textBaseline = 'middle';

  traceCycles.forEach((cycle, index) => {
    const x = index * WAVEFORM_CYCLE_WIDTH;
    context.fillText(String(cycle), x + 4, WAVEFORM_HEADER_HEIGHT / 2);
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
  waveformCycleIndexByCycle: Map<number, number>,
): void {
  const y = WAVEFORM_HEADER_HEIGHT + signalIndex * WAVEFORM_ROW_HEIGHT;
  context.fillStyle = signalIndex % 2 === 0 ? '#101419' : '#151a20';
  context.fillRect(0, y, traceCycles.length * WAVEFORM_CYCLE_WIDTH, WAVEFORM_ROW_HEIGHT);

  if (selectedWaveformSignal && sameWaveformSignal(signal, selectedWaveformSignal)) {
    context.fillStyle = 'rgb(211 50 50 / 15%)';
    context.fillRect(0, y, traceCycles.length * WAVEFORM_CYCLE_WIDTH, WAVEFORM_ROW_HEIGHT);
  }

  context.strokeStyle = 'rgb(154 164 173 / 12%)';
  context.beginPath();
  context.moveTo(0, y + WAVEFORM_ROW_HEIGHT + 0.5);
  context.lineTo(traceCycles.length * WAVEFORM_CYCLE_WIDTH, y + WAVEFORM_ROW_HEIGHT + 0.5);
  context.stroke();

  if (signal.max_value <= 1) {
    drawDigitalSignal(context, signal, y, waveformCycleIndexByCycle);
  } else {
    drawStrengthSignal(context, signal, y, waveformCycleIndexByCycle);
  }
}

function drawDigitalSignal(
  context: CanvasRenderingContext2D,
  signal: WaveformSignal,
  y: number,
  waveformCycleIndexByCycle: Map<number, number>,
): void {
  context.strokeStyle = '#f45d5d';
  context.lineWidth = 2;
  context.beginPath();

  let previousY: number | undefined;
  traceCycles.forEach((cycle, index) => {
    const x = index * WAVEFORM_CYCLE_WIDTH;
    const waveformIndex = waveformCycleIndexByCycle.get(cycle);
    const value = waveformIndex === undefined ? 0 : signal.values[waveformIndex] ?? 0;
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
  waveformCycleIndexByCycle: Map<number, number>,
): void {
  context.textBaseline = 'middle';
  context.font = '10px "Cascadia Mono", Consolas, monospace';

  traceCycles.forEach((cycle, index) => {
    const waveformIndex = waveformCycleIndexByCycle.get(cycle);
    const value = waveformIndex === undefined ? 0 : signal.values[waveformIndex] ?? 0;
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
        `cycle=${entry.cycle}`,
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
