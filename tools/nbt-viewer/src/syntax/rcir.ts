const KEYWORDS = new Set([
  'rcir',
  'stage',
  'target',
  'top',
  'module',
  'leaf',
  'port',
  'input',
  'output',
  'net',
  'cell',
  'in',
  'out',
  'instance',
  'bind',
  'class',
  'driver',
  'sinks',
  'node',
  'logic',
  'sequential',
  'input_ports',
  'output_ports',
  'inputs',
  'edge',
  'self',
]);

const TYPES = new Set(['bit', 'bits', 'const']);
const STAGES = new Set(['logical', 'routable']);
const BUILTINS = new Set([
  'and',
  'or',
  'xor',
  'not',
  'buffer',
  'inc',
  'register',
  'd_latch',
  'posedge',
  'negedge',
]);

const TOKEN_PATTERN = /#[^\r\n]*|"(?:\\.|[^"\\])*"|\b\d+\b|[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*|\s+|./g;

export function highlightRcir(source: string): DocumentFragment {
  const fragment = document.createDocumentFragment();

  for (const match of source.matchAll(TOKEN_PATTERN)) {
    const token = match[0];
    const kind = rcirTokenKind(token);
    if (!kind) {
      fragment.append(document.createTextNode(token));
      continue;
    }

    const span = document.createElement('span');
    span.className = `rcir-token rcir-${kind}`;
    span.textContent = token;
    fragment.append(span);
  }

  return fragment;
}

function rcirTokenKind(token: string): string | undefined {
  if (token.startsWith('#')) return 'comment';
  if (token.startsWith('"')) return 'string';
  if (/^\d+$/.test(token)) return 'number';
  if (KEYWORDS.has(token)) return 'keyword';
  if (TYPES.has(token)) return 'type';
  if (STAGES.has(token)) return 'stage';
  if (BUILTINS.has(token) || token.includes('.')) return 'operation';
  return undefined;
}
