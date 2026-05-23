import './styles.css';
import { loadNbtFile, stringifyNbt } from './nbt/loadNbt';
import { toStructureModel } from './nbt/toStructure';
import { StructureViewer } from './render/StructureViewer';
import { NbtSimulation, NbtSimulationError, type SnapshotInfo, type TraceEntry } from './sim/NbtSimulation';
import type { StructureBlock, StructurePaletteEntry } from './types';

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

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="app-shell">
    <section id="drop-zone" class="workspace">
      <section class="viewer-panel">
        <canvas id="structure-canvas"></canvas>
        <div class="floating-actions">
          <button id="open-folder" class="file-button" type="button">Open Folder</button>
          <button id="open-file" class="file-button" type="button">Open NBT</button>
          <input id="folder-input" class="hidden-file-input" type="file" multiple />
          <input id="file-input" class="hidden-file-input" type="file" accept=".nbt,.dat,.schem,.schematic,.litematic,.mcstructure" />
        </div>
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
            <span id="trace-count">No events</span>
          </summary>
          <div class="trace-controls">
            <button id="trace-prev" type="button" aria-label="Previous cycle">Prev</button>
            <input id="trace-cycle" type="range" min="0" max="0" value="0" />
            <button id="trace-next" type="button" aria-label="Next cycle">Next</button>
            <span id="trace-cycle-label">cycle -</span>
          </div>
          <pre id="trace-output">Run a simulation to inspect events.</pre>
        </details>
        <div id="viewer-empty" class="viewer-empty">Drop an .nbt file or use Open NBT.</div>
      </section>
    </section>
  </main>
`;

const input = document.querySelector<HTMLInputElement>('#file-input')!;
const folderInput = document.querySelector<HTMLInputElement>('#folder-input')!;
const openFileButton = document.querySelector<HTMLButtonElement>('#open-file')!;
const openFolderButton = document.querySelector<HTMLButtonElement>('#open-folder')!;
const dropZone = document.querySelector<HTMLElement>('#drop-zone')!;
const filesPanel = document.querySelector<HTMLDetailsElement>('#files-panel')!;
const filesList = document.querySelector<HTMLElement>('#files-list')!;
const filesCount = document.querySelector<HTMLElement>('#files-count')!;
const canvas = document.querySelector<HTMLCanvasElement>('#structure-canvas')!;
const viewerEmpty = document.querySelector<HTMLElement>('#viewer-empty')!;
const inspector = document.querySelector<HTMLElement>('#inspector')!;
const toggleSwitchButton = document.querySelector<HTMLButtonElement>('#toggle-switch')!;
const tracePanel = document.querySelector<HTMLDetailsElement>('#trace-panel')!;
const traceCount = document.querySelector<HTMLElement>('#trace-count')!;
const traceOutput = document.querySelector<HTMLElement>('#trace-output')!;
const traceCycleInput = document.querySelector<HTMLInputElement>('#trace-cycle')!;
const traceCycleLabel = document.querySelector<HTMLElement>('#trace-cycle-label')!;
const tracePrevButton = document.querySelector<HTMLButtonElement>('#trace-prev')!;
const traceNextButton = document.querySelector<HTMLButtonElement>('#trace-next')!;

const viewer = new StructureViewer(canvas);
viewer.setSelectionHandler(renderSelection);

let simulation: NbtSimulation | undefined;
let selectedBlock: StructureBlock | undefined;
let currentNbtBytes: Uint8Array | undefined;
let currentRoot: unknown;
let currentTrace: TraceEntry[] = [];
let traceCycles: number[] = [];
let currentSnapshots: SnapshotInfo[] = [];
let traceBaseRoot: unknown;
let isTracePreviewActive = false;

folderInput.setAttribute('webkitdirectory', '');
folderInput.setAttribute('directory', '');

openFileButton.addEventListener('click', () => {
  input.value = '';
  input.click();
});

openFolderButton.addEventListener('click', () => {
  folderInput.value = '';
  folderInput.click();
});

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

traceCycleInput.addEventListener('input', () => {
  void renderTraceCycle(Number(traceCycleInput.value));
});

tracePrevButton.addEventListener('click', () => {
  traceCycleInput.value = String(Math.max(0, Number(traceCycleInput.value) - 1));
  void renderTraceCycle(Number(traceCycleInput.value));
});

traceNextButton.addEventListener('click', () => {
  traceCycleInput.value = String(Math.min(traceCycles.length - 1, Number(traceCycleInput.value) + 1));
  void renderTraceCycle(Number(traceCycleInput.value));
});

async function toggleSelectedSwitch(): Promise<void> {
  if (!selectedBlock || !currentNbtBytes) return;

  simulation ??= await NbtSimulation.create(currentNbtBytes);

  const baseRoot = currentRoot;
  const nextRoot = simulation.toggleSwitch(selectedBlock);
  if (!nextRoot) return;

  currentRoot = mergeSimulatedState(currentRoot, nextRoot);
  const structure = toStructureModel(currentRoot);
  if (!structure) throw new Error('Simulator returned a structure that the viewer could not render.');

  await viewer.setStructure(structure);
  selectedBlock = undefined;
  renderSelection(undefined);
  renderTrace(simulation.trace(), simulation.snapshots(), baseRoot);
  inspector.textContent = [
    `updated by simulator`,
    `size: ${structure.size.join(' x ')}`,
    `palette: ${structure.palette.length}`,
    `blocks: ${structure.blocks.length}`,
  ].join('\n');
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

async function openFile(file: File, selectedEntry?: Element | null): Promise<void> {
  try {
    const parsed = await loadNbtFile(file);
    const structure = toStructureModel(parsed.root);
    simulation = undefined;
    currentNbtBytes = parsed.bytes;
    currentRoot = parsed.root;

    markSelectedFile(selectedEntry);

    if (structure) {
      await viewer.setStructure(structure);
      viewerEmpty.classList.add('hidden');
      inspector.textContent = [
        `size: ${structure.size.join(' x ')}`,
        `palette: ${structure.palette.length}`,
        `blocks: ${structure.blocks.length}`,
      ].join('\n');
      renderTrace([], [], undefined);
    } else {
      viewerEmpty.classList.remove('hidden');
      inspector.textContent = 'This NBT file does not look like a Minecraft structure file.';
      renderTrace([], [], undefined);
    }
  } catch (error) {
    simulation = undefined;
    currentNbtBytes = undefined;
    currentRoot = undefined;
    selectedBlock = undefined;
    toggleSwitchButton.classList.add('hidden');
    viewerEmpty.classList.remove('hidden');
    inspector.textContent = error instanceof Error ? error.message : String(error);
    renderTrace([], [], undefined);
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

function getLeverPowered(block: StructureBlock | undefined): boolean {
  return block?.palette.properties?.powered === 'true';
}

function renderSimulationError(error: unknown): void {
  if (error instanceof NbtSimulationError) {
    inspector.textContent = error.message;
    renderTrace(error.trace, error.snapshots, currentRoot, true);
    return;
  }

  inspector.textContent = error instanceof Error ? error.message : String(error);
}

function renderTrace(trace: TraceEntry[], snapshots: SnapshotInfo[], baseRoot: unknown, open = false): void {
  currentTrace = trace;
  currentSnapshots = snapshots;
  traceBaseRoot = baseRoot;
  traceCycles = Array.from(new Set(trace.map(entry => entry.cycle))).sort((a, b) => a - b);
  traceCount.textContent = trace.length === 0 ? 'No events' : `${trace.length} events`;
  traceCycleInput.max = String(Math.max(0, traceCycles.length - 1));
  traceCycleInput.value = '0';
  traceCycleInput.disabled = traceCycles.length === 0;
  tracePrevButton.disabled = traceCycles.length === 0;
  traceNextButton.disabled = traceCycles.length === 0;
  void renderTraceCycle(traceCycles.length === 0 ? -1 : 0);
  if (open || trace.length > 0) {
    tracePanel.open = true;
  }
}

async function renderTraceCycle(index: number): Promise<void> {
  if (index < 0 || traceCycles.length === 0) {
    traceCycleLabel.textContent = 'cycle -';
    traceOutput.textContent = 'Run a simulation to inspect events.';
    viewer.setTraceHighlights([]);
    await restoreCurrentStructurePreview();
    return;
  }

  const safeIndex = Math.max(0, Math.min(traceCycles.length - 1, index));
  const cycle = traceCycles[safeIndex];
  const entries = currentTrace.filter(entry => entry.cycle === cycle);
  const positions = uniquePositions(entries.map(entry => rustPosToRenderPos(entry.target_position)));
  const snapshot = currentSnapshots.find(snapshot => snapshot.cycle === cycle);
  traceCycleInput.value = String(safeIndex);
  traceCycleLabel.textContent = `cycle ${cycle} / ${entries.length} events`;
  traceOutput.textContent = formatTrace(entries);
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
