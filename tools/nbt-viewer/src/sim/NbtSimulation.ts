import type { StructureBlock } from '../types';

type WasmModule = {
  default: (moduleOrPath?: string | URL | Request | Response | WebAssembly.Module) => Promise<unknown>;
  NbtSimulator: {
    new (nbtBytes: Uint8Array): WasmSimulator;
    trace_init(nbtBytes: Uint8Array): TraceReport;
  };
};

type WasmSimulator = {
  switches(): SwitchInfo[];
  snapshots(): SnapshotInfo[];
  structure(): unknown;
  trace(): TraceEntry[];
  toggle_switch(x: number, y: number, z: number, isOn: boolean): unknown;
};

export type SwitchInfo = {
  pos: [number, number, number];
  is_on: boolean;
};

export type TraceEntry = {
  cycle: number;
  event_id?: number;
  event_type: string;
  target_position: [number, number, number];
  direction: string;
  block_before: string;
  current_queue_len: number;
  next_queue_len: number;
};

export type TraceReport = {
  ok: boolean;
  error?: string;
  trace: TraceEntry[];
  snapshots: SnapshotInfo[];
};

export type SnapshotInfo = {
  cycle: number;
  root: unknown;
};

export class NbtSimulationError extends Error {
  constructor(message: string, readonly trace: TraceEntry[], readonly snapshots: SnapshotInfo[]) {
    super(message);
    this.name = 'NbtSimulationError';
  }
}

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

  static async create(nbtBytes: Uint8Array): Promise<NbtSimulation> {
    const wasm = await loadWasmModule();
    await wasm.default(wasmBinaryPath);

    try {
      return new NbtSimulation(new wasm.NbtSimulator(nbtBytes));
    } catch (error) {
      const report = wasm.NbtSimulator.trace_init(nbtBytes);
      throw new NbtSimulationError(report.error ?? getErrorMessage(error), report.trace, report.snapshots);
    }
  }

  structure(): unknown {
    return this.sim.structure();
  }

  trace(): TraceEntry[] {
    return this.sim.trace();
  }

  snapshots(): SnapshotInfo[] {
    return this.sim.snapshots();
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
    try {
      return this.sim.toggle_switch(x, y, z, !current.is_on);
    } catch (error) {
      throw new NbtSimulationError(getErrorMessage(error), this.trace(), this.snapshots());
    }
  }
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
