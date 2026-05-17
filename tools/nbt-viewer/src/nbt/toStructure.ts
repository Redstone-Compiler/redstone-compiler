import type { StructureBlock, StructureModel, StructurePaletteEntry } from '../types';

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function asNumberTuple(value: unknown): [number, number, number] | undefined {
  if (!Array.isArray(value) || value.length < 3) return undefined;
  const nums = value.slice(0, 3).map(Number);
  return nums.every(Number.isFinite) ? [nums[0], nums[1], nums[2]] : undefined;
}

function parsePaletteEntry(value: unknown): StructurePaletteEntry {
  const entry = asRecord(value) ?? {};
  const props = asRecord(entry.Properties) ?? {};

  return {
    name: String(entry.Name ?? 'minecraft:air'),
    properties: Object.fromEntries(
      Object.entries(props).map(([key, propValue]) => [key, String(propValue)]),
    ),
  };
}

export function toStructureModel(root: unknown): StructureModel | undefined {
  const obj = asRecord(root);
  if (!obj) return undefined;

  const size = asNumberTuple(obj.size);
  const rawPalette = Array.isArray(obj.palette) ? obj.palette : undefined;
  const rawBlocks = Array.isArray(obj.blocks) ? obj.blocks : undefined;
  if (!size || !rawPalette || !rawBlocks) return undefined;

  const palette = rawPalette.map(parsePaletteEntry);
  const blocks: StructureBlock[] = [];

  for (const rawBlock of rawBlocks) {
    const block = asRecord(rawBlock);
    if (!block) continue;

    const pos = asNumberTuple(block.pos);
    const state = Number(block.state);
    const paletteEntry = palette[state];
    if (!pos || !Number.isInteger(state) || !paletteEntry) continue;
    if (paletteEntry.name === 'minecraft:air') continue;

    blocks.push({
      pos,
      state,
      palette: paletteEntry,
      nbt: block.nbt,
    });
  }

  return { size, palette, blocks };
}
