import { NbtFile } from 'deepslate/nbt';
import type { ParsedNbt } from '../types';

export async function loadNbtFile(file: File): Promise<ParsedNbt> {
  const nbtFile = NbtFile.read(new Uint8Array(await file.arrayBuffer()));

  return {
    fileName: file.name,
    byteLength: file.size,
    parseType: `${nbtFile.littleEndian ? 'little' : 'big'}/${nbtFile.compression}`,
    root: nbtFile.root.toSimplifiedJson(),
  };
}

export function stringifyNbt(value: unknown): string {
  return JSON.stringify(
    value,
    (_key, item) => (typeof item === 'bigint' ? item.toString() : item),
    2,
  );
}
