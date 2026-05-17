export type ParsedNbt = {
  fileName: string;
  byteLength: number;
  parseType: string;
  root: unknown;
};

export type StructurePaletteEntry = {
  name: string;
  properties: Record<string, string>;
};

export type StructureBlock = {
  pos: [number, number, number];
  state: number;
  palette: StructurePaletteEntry;
  nbt?: unknown;
};

export type StructureModel = {
  size: [number, number, number];
  palette: StructurePaletteEntry[];
  blocks: StructureBlock[];
};
