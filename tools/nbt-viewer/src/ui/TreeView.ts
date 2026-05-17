import { stringifyNbt } from '../nbt/loadNbt';

function valueLabel(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return `Array(${value.length})`;
  if (typeof value === 'object') return `Object(${Object.keys(value as object).length})`;
  return String(value);
}

function createNode(key: string, value: unknown, depth: number): HTMLElement {
  const node = document.createElement('details');
  node.className = 'tree-node';
  node.open = depth < 2;

  const summary = document.createElement('summary');
  const keyEl = document.createElement('span');
  keyEl.className = 'tree-key';
  keyEl.textContent = key;

  const valueEl = document.createElement('span');
  valueEl.className = 'tree-value';
  valueEl.textContent = valueLabel(value);

  summary.append(keyEl, valueEl);
  node.append(summary);

  if (Array.isArray(value)) {
    value.slice(0, 500).forEach((child, index) => node.append(createTreeItem(String(index), child, depth + 1)));
    if (value.length > 500) node.append(createLeaf('...', `${value.length - 500} more items`));
  } else if (value && typeof value === 'object') {
    Object.entries(value as Record<string, unknown>).forEach(([childKey, child]) => {
      node.append(createTreeItem(childKey, child, depth + 1));
    });
  }

  return node;
}

function createLeaf(key: string, value: unknown): HTMLElement {
  const row = document.createElement('div');
  row.className = 'tree-leaf';

  const keyEl = document.createElement('span');
  keyEl.className = 'tree-key';
  keyEl.textContent = key;

  const valueEl = document.createElement('span');
  valueEl.className = 'tree-value primitive';
  valueEl.textContent = typeof value === 'string' ? value : stringifyNbt(value);

  row.append(keyEl, valueEl);
  return row;
}

function createTreeItem(key: string, value: unknown, depth: number): HTMLElement {
  if (value && typeof value === 'object') {
    return createNode(key, value, depth);
  }
  return createLeaf(key, value);
}

export function renderTree(root: HTMLElement, value: unknown): void {
  root.replaceChildren(createTreeItem('root', value, 0));
}
