import './styles.css';
import { loadNbtFile, stringifyNbt } from './nbt/loadNbt';
import { toStructureModel } from './nbt/toStructure';
import { StructureViewer } from './render/StructureViewer';
import type { StructureBlock } from './types';

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
          <label class="file-button">
            Open Folder
            <input id="folder-input" type="file" multiple />
          </label>
          <label class="file-button">
            Open NBT
            <input id="file-input" type="file" accept=".nbt,.dat,.schem,.schematic,.litematic,.mcstructure" />
          </label>
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
          </div>
          <pre id="inspector">Select a block in the 3D view.</pre>
        </aside>
        <div id="viewer-empty" class="viewer-empty">Drop an .nbt file or use Open NBT.</div>
      </section>
    </section>
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
const inspector = document.querySelector<HTMLElement>('#inspector')!;

const viewer = new StructureViewer(canvas);
viewer.setSelectionHandler(renderSelection);

folderInput.setAttribute('webkitdirectory', '');
folderInput.setAttribute('directory', '');

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (file) void openFile(file);
});

folderInput.addEventListener('change', () => {
  renderFileBrowser(Array.from(folderInput.files ?? []));
});

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

    markSelectedFile(selectedEntry);

    if (structure) {
      viewer.setStructure(structure);
      viewerEmpty.classList.add('hidden');
      inspector.textContent = [
        `size: ${structure.size.join(' x ')}`,
        `palette: ${structure.palette.length}`,
        `blocks: ${structure.blocks.length}`,
      ].join('\n');
    } else {
      viewerEmpty.classList.remove('hidden');
      inspector.textContent = 'This NBT file does not look like a Minecraft structure file.';
    }
  } catch (error) {
    viewerEmpty.classList.remove('hidden');
    inspector.textContent = error instanceof Error ? error.message : String(error);
  }
}

function markSelectedFile(selectedEntry?: Element | null): void {
  filesList.querySelectorAll('.file-entry.selected').forEach(entry => {
    entry.classList.remove('selected');
  });
  selectedEntry?.classList.add('selected');
}

function renderSelection(block: StructureBlock | undefined): void {
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
