const KEYWORDS = new Set([
  'always', 'always_comb', 'always_ff', 'always_latch', 'and', 'assign', 'automatic',
  'begin', 'buf', 'case', 'casex', 'casez', 'default', 'else', 'end', 'endcase',
  'endfunction', 'endgenerate', 'endmodule', 'endtask', 'for', 'forever', 'function',
  'generate', 'genvar', 'if', 'initial', 'inout', 'input', 'integer', 'localparam',
  'module', 'nand', 'negedge', 'nor', 'not', 'or', 'output', 'parameter', 'posedge',
  'repeat', 'signed', 'task', 'unsigned', 'while', 'xnor', 'xor',
]);

const TYPES = new Set([
  'bit', 'byte', 'int', 'logic', 'longint', 'real', 'realtime', 'reg', 'shortint',
  'shortreal', 'supply0', 'supply1', 'time', 'tri', 'tri0', 'tri1', 'wand', 'wire',
  'wor',
]);

const CONSTANTS = new Set(['true', 'false']);
const NUMBER = /^(?:(?:\d[\d_]*)?'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d[\d_]*)?)/;
const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_$]*/;
const OPERATOR = /^(?:===|!==|<<<|>>>|<=|>=|==|!=|&&|\|\||<<|>>|\+:|-:|\*\*|~\^|\^~|[+\-*/%&|^~!<>=?:])/;

export type VerilogHighlightState = { inBlockComment: boolean };

export function highlightVerilog(
  source: string,
  state: VerilogHighlightState = { inBlockComment: false },
): DocumentFragment {
  const fragment = document.createDocumentFragment();
  let offset = 0;

  while (offset < source.length) {
    if (state.inBlockComment) {
      const end = source.indexOf('*/', offset);
      const stop = end < 0 ? source.length : end + 2;
      appendToken(fragment, source.slice(offset, stop), 'comment');
      offset = stop;
      state.inBlockComment = end < 0;
      continue;
    }

    if (source.startsWith('//', offset)) {
      appendToken(fragment, source.slice(offset), 'comment');
      break;
    }
    if (source.startsWith('/*', offset)) {
      const end = source.indexOf('*/', offset + 2);
      const stop = end < 0 ? source.length : end + 2;
      appendToken(fragment, source.slice(offset, stop), 'comment');
      offset = stop;
      state.inBlockComment = end < 0;
      continue;
    }

    const ch = source[offset];
    if (/\s/.test(ch)) {
      const match = /^\s+/.exec(source.slice(offset))!;
      fragment.append(document.createTextNode(match[0]));
      offset += match[0].length;
      continue;
    }
    if (ch === '"') {
      const stop = scanString(source, offset);
      appendToken(fragment, source.slice(offset, stop), 'string');
      offset = stop;
      continue;
    }
    if (ch === '`') {
      const match = /^`[A-Za-z_][A-Za-z0-9_$]*/.exec(source.slice(offset));
      const token = match?.[0] ?? ch;
      appendToken(fragment, token, 'directive');
      offset += token.length;
      continue;
    }
    if (ch === '\\') {
      const match = /^\\\S+/.exec(source.slice(offset));
      const token = match?.[0] ?? ch;
      appendToken(fragment, token, 'identifier');
      offset += token.length;
      continue;
    }
    if (ch === '$') {
      const match = /^\$[A-Za-z_][A-Za-z0-9_$]*/.exec(source.slice(offset));
      const token = match?.[0] ?? ch;
      appendToken(fragment, token, 'system');
      offset += token.length;
      continue;
    }

    const rest = source.slice(offset);
    const number = NUMBER.exec(rest)?.[0];
    if (number) {
      appendToken(fragment, number, 'number');
      offset += number.length;
      continue;
    }
    const identifier = IDENTIFIER.exec(rest)?.[0];
    if (identifier) {
      const kind = KEYWORDS.has(identifier)
        ? 'keyword'
        : TYPES.has(identifier)
          ? 'type'
          : CONSTANTS.has(identifier)
            ? 'constant'
            : undefined;
      if (kind) appendToken(fragment, identifier, kind);
      else fragment.append(document.createTextNode(identifier));
      offset += identifier.length;
      continue;
    }
    const operator = OPERATOR.exec(rest)?.[0];
    if (operator) {
      appendToken(fragment, operator, 'operator');
      offset += operator.length;
      continue;
    }

    fragment.append(document.createTextNode(ch));
    offset += 1;
  }

  return fragment;
}

function scanString(source: string, start: number): number {
  let offset = start + 1;
  while (offset < source.length) {
    if (source[offset] === '\\') offset += 2;
    else if (source[offset++] === '"') break;
  }
  return Math.min(offset, source.length);
}

function appendToken(fragment: DocumentFragment, token: string, kind: string): void {
  const span = document.createElement('span');
  span.className = `verilog-token verilog-${kind}`;
  span.textContent = token;
  fragment.append(span);
}
