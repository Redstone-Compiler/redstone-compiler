use redstone_compiler::nbt::{NBTRoot, ToNBT};
use redstone_compiler::world::block::BlockKind;
use redstone_compiler::world::position::Position;
use redstone_compiler::world::simulator::{SimulationSnapshot, SimulationTraceEntry, Simulator};
use serde::Serialize;
use wasm_bindgen::prelude::*;

const MAX_SIMULATION_CYCLES: usize = 96;
const MAX_SIMULATION_EVENTS: usize = 10_000;
const TRACE_LIMIT: usize = 12_000;

#[derive(Serialize)]
struct SwitchInfo {
    pos: [usize; 3],
    is_on: bool,
}

#[derive(Serialize)]
struct TraceReport {
    ok: bool,
    error: Option<String>,
    trace: Vec<SimulationTraceEntry>,
    snapshots: Vec<SnapshotInfo>,
}

#[derive(Serialize)]
struct SnapshotInfo {
    cycle: usize,
    root: NBTRoot,
}

#[wasm_bindgen]
pub struct NbtSimulator {
    sim: Simulator,
    last_trace: Vec<SimulationTraceEntry>,
    last_snapshots: Vec<SnapshotInfo>,
}

#[wasm_bindgen]
impl NbtSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(nbt_bytes: &[u8]) -> Result<NbtSimulator, JsValue> {
        let nbt = NBTRoot::from_nbt_bytes(nbt_bytes).map_err(to_js_error)?;
        let world = nbt.to_world();
        let sim = Simulator::from_with_limits_and_trace(
            &world,
            MAX_SIMULATION_CYCLES,
            MAX_SIMULATION_EVENTS,
            TRACE_LIMIT,
        )
        .map_err(to_js_error)?;
        let last_trace = sim.trace().to_vec();
        let last_snapshots = snapshots_to_info(sim.snapshots());

        Ok(Self {
            sim,
            last_trace,
            last_snapshots,
        })
    }

    pub fn trace_init(nbt_bytes: &[u8]) -> Result<JsValue, JsValue> {
        let nbt = NBTRoot::from_nbt_bytes(nbt_bytes).map_err(to_js_error)?;
        let world = nbt.to_world();
        let report = match Simulator::from_with_limits_and_trace(
            &world,
            MAX_SIMULATION_CYCLES,
            MAX_SIMULATION_EVENTS,
            TRACE_LIMIT,
        ) {
            Ok(sim) => TraceReport {
                ok: true,
                error: None,
                trace: sim.trace().to_vec(),
                snapshots: snapshots_to_info(sim.snapshots()),
            },
            Err(error) => TraceReport {
                ok: false,
                error: Some(error.message().to_owned()),
                trace: error.trace().to_vec(),
                snapshots: snapshots_to_info(error.snapshots()),
            },
        };

        serde_wasm_bindgen::to_value(&report).map_err(to_js_error)
    }

    pub fn switches(&self) -> Result<JsValue, JsValue> {
        let switches = self
            .sim
            .world()
            .iter_block()
            .into_iter()
            .filter_map(|(pos, block)| {
                let BlockKind::Switch { is_on } = block.kind else {
                    return None;
                };

                Some(SwitchInfo {
                    pos: [pos.0, pos.1, pos.2],
                    is_on,
                })
            })
            .collect::<Vec<_>>();

        serde_wasm_bindgen::to_value(&switches).map_err(to_js_error)
    }

    pub fn toggle_switch(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        is_on: bool,
    ) -> Result<JsValue, JsValue> {
        self.sim.clear_trace();
        if let Err(error) = self.sim.change_state_with_limits(
            vec![(Position(x, y, z), is_on)],
            MAX_SIMULATION_CYCLES,
            MAX_SIMULATION_EVENTS,
        ) {
            self.last_trace = self.sim.trace().to_vec();
            self.last_snapshots = snapshots_to_info(self.sim.snapshots());
            return Err(to_js_error(error));
        }
        self.last_trace = self.sim.trace().to_vec();
        self.last_snapshots = snapshots_to_info(self.sim.snapshots());
        self.structure()
    }

    pub fn trace(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.last_trace).map_err(to_js_error)
    }

    pub fn snapshots(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.last_snapshots).map_err(to_js_error)
    }

    pub fn structure(&self) -> Result<JsValue, JsValue> {
        let nbt = self.sim.world().to_nbt();
        serde_wasm_bindgen::to_value(&nbt).map_err(to_js_error)
    }

    pub fn gzip_nbt(&self) -> Result<Vec<u8>, JsValue> {
        self.sim
            .world()
            .to_nbt()
            .to_gzip_bytes()
            .map_err(to_js_error)
    }
}

fn to_js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn snapshots_to_info(snapshots: &[SimulationSnapshot]) -> Vec<SnapshotInfo> {
    snapshots
        .iter()
        .map(|snapshot| SnapshotInfo {
            cycle: snapshot.cycle,
            root: snapshot.world.to_nbt(),
        })
        .collect()
}
