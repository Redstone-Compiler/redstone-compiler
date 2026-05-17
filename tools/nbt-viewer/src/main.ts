import './styles.css';
import { loadNbtFile, stringifyNbt } from './nbt/loadNbt';
import { toStructureModel } from './nbt/toStructure';
import { StructureViewer } from './render/StructureViewer';
import { renderTree } from './ui/TreeView';
import type { StructureBlock } from './types';

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
        <details id="tree-panel" class="floating-panel tree-panel">
          <summary>
            <span>NBT Tree</span>
            <span id="file-meta">No file loaded</span>
          </summary>
          <div id="tree-root" class="tree-root empty">Drop an .nbt file here.</div>
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
const treeRoot = document.querySelector<HTMLElement>('#tree-root')!;
const fileMeta = document.querySelector<HTMLElement>('#file-meta')!;
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
  const files = Array.from(event.dataTransfer?.files ?? []);
  if (files.length > 1) {
    renderFileBrowser(files);
  } else if (files[0]) {
    void openFile(files[0]);
  }
});

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
    button.textContent = getDisplayPath(file);
    button.addEventListener('click', () => void openFile(file, button));
    filesList.append(button);
  }

  void openFile(nbtFiles[0], filesList.querySelector('.file-entry'));
}

async function openFile(file: File, selectedEntry?: Element | null): Promise<void> {
  try {
    const parsed = await loadNbtFile(file);
    const structure = toStructureModel(parsed.root);

    treeRoot.classList.remove('empty');
    renderTree(treeRoot, parsed.root);
    fileMeta.textContent = `${getDisplayPath(file)} · ${formatBytes(parsed.byteLength)} · ${parsed.parseType}`;
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
    treeRoot.classList.add('empty');
    treeRoot.textContent = error instanceof Error ? error.message : String(error);
    fileMeta.textContent = 'Parse failed';
    viewerEmpty.classList.remove('hidden');
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
