import type { StructureBlock } from '../types';

type WasmModule = {
  default: (moduleOrPath?: string | URL | Request | Response | WebAssembly.Module) => Promise<unknown>;
  NbtSimulator: new (nbtBytes: Uint8Array) => WasmSimulator;
};

type WasmSimulator = {
  switches(): SwitchInfo[];
  structure(): unknown;
  toggle_switch(x: number, y: number, z: number, isOn: boolean): unknown;
};

export type SwitchInfo = {
  pos: [number, number, number];
  is_on: boolean;
};

const wasmModulePath = '/wasm/nbt-sim/nbt_sim_wasm.js';
const wasmBinaryPath = '/wasm/nbt-sim/nbt_sim_wasm_bg.wasm';

let wasmModulePromise: Promise<WasmModule> | undefined;

function loadWasmModule(): Promise<WasmModule> {
  wasmModulePromise ??= import(/* @vite-ignore */ new URL(wasmModulePath, window.location.origin).href) as Promise<WasmModule>;
  return wasmModulePromise;
}

function renderPosToRustPos(pos: [number, number, number]): [number, number, number] {
  return [pos[2], pos[0], pos[1]];
}

function samePos(a: [number, number, number], b: [number, number, number]): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

export class NbtSimulation {
  private constructor(private readonly sim: WasmSimulator) {}

  static async create(nbtBytes: Uint8Array): Promise<NbtSimulation | undefined> {
    try {
      const wasm = await loadWasmModule();
      await wasm.default(wasmBinaryPath);
      return new NbtSimulation(new wasm.NbtSimulator(nbtBytes));
    } catch (error) {
      console.warn('NBT simulator WASM is unavailable.', error);
      return undefined;
    }
  }

  structure(): unknown {
    return this.sim.structure();
  }

  getSwitch(block: StructureBlock): SwitchInfo | undefined {
    if (block.palette.name !== 'minecraft:lever') return undefined;

    const rustPos = renderPosToRustPos(block.pos);
    return this.sim.switches().find(item => samePos(item.pos, rustPos));
  }

  toggleSwitch(block: StructureBlock): unknown | undefined {
    const current = this.getSwitch(block);
    if (!current) return undefined;

    const [x, y, z] = current.pos;
    return this.sim.toggle_switch(x, y, z, !current.is_on);
  }
}
