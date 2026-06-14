use std::collections::HashSet;

use redstone_compiler::graph::graphviz::ToGraphvizGraph;
use redstone_compiler::graph::world::{WorldGraph, WorldGraphBuilder};
use redstone_compiler::graph::{GraphNodeId, GraphNodeKind};
use redstone_compiler::nbt::{NBTRoot, ToNBT};
use redstone_compiler::output::OutputMetadata;
use redstone_compiler::transform::place_and_route::place_bound::{PlaceBound, PropagateType};
use redstone_compiler::transform::place_and_route::utils::world_to_logic_with_outputs_unoptimized;
use redstone_compiler::transform::world_to_logic::WorldToLogicTransformer;
use redstone_compiler::world::block::{Block, BlockKind, Direction};
use redstone_compiler::world::position::{DimSize, Position};
use redstone_compiler::world::simulator::{
    SimulationSnapshot, SimulationTraceEntry, SimulationWaveform, Simulator,
};
use redstone_compiler::world::{World, World3D};
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
    waveform: SimulationWaveform,
}

#[derive(Serialize)]
struct SnapshotInfo {
    cycle: usize,
    root: NBTRoot,
}

#[derive(Serialize)]
struct GraphDotInfo {
    raw_world_dot: String,
    raw_world_dot_without_tags: String,
    folded_world_dot: String,
    folded_world_dot_without_tags: String,
    logic_dot: String,
    logic_dot_without_tags: String,
}

#[wasm_bindgen]
pub struct NbtSimulator {
    sim: Simulator,
    last_trace: Vec<SimulationTraceEntry>,
    last_snapshots: Vec<SnapshotInfo>,
    last_waveform: SimulationWaveform,
    history_trace: Vec<SimulationTraceEntry>,
    history_snapshots: Vec<SnapshotInfo>,
    history_waveform: SimulationWaveform,
}

impl NbtSimulator {
    fn new_with_trace_limit(nbt_bytes: &[u8], trace_limit: usize) -> Result<NbtSimulator, JsValue> {
        let nbt = NBTRoot::from_nbt_bytes(nbt_bytes).map_err(to_js_error)?;
        let world = nbt.to_world();
        let sim = Simulator::from_preserving_torch_states_with_limits_and_trace(
            &world,
            MAX_SIMULATION_CYCLES,
            MAX_SIMULATION_EVENTS,
            trace_limit,
        )
        .map_err(to_js_error)?;
        let last_trace = sim.trace().to_vec();
        let last_snapshots = snapshots_to_info(sim.snapshots());
        let last_waveform = sim.waveform();
        let history_trace = sim.trace().to_vec();
        let history_snapshots = snapshots_to_info(sim.snapshots());
        let history_waveform = sim.waveform();

        Ok(Self {
            sim,
            last_trace,
            last_snapshots,
            last_waveform,
            history_trace,
            history_snapshots,
            history_waveform,
        })
    }

    fn clear_trace_views(&mut self) {
        self.last_trace.clear();
        self.last_snapshots.clear();
        self.last_waveform = empty_waveform();
        self.history_trace.clear();
        self.history_snapshots.clear();
        self.history_waveform = empty_waveform();
    }

    fn refresh_trace_views_from_cycle(&mut self, start_cycle: usize) {
        let trace = self.sim.trace();
        let snapshots = self.sim.snapshots();
        let trace_start = trace
            .iter()
            .position(|entry| entry.cycle >= start_cycle)
            .unwrap_or(trace.len());
        let snapshot_start = snapshots
            .iter()
            .position(|snapshot| snapshot.cycle >= start_cycle)
            .unwrap_or(snapshots.len());
        let last_snapshots = &snapshots[snapshot_start..];

        self.last_trace = trace[trace_start..].to_vec();
        self.last_snapshots = snapshots_to_info(last_snapshots);
        self.last_waveform = SimulationWaveform::from_snapshots(last_snapshots);
        self.history_trace = trace.to_vec();
        self.history_snapshots = snapshots_to_info(snapshots);
        self.history_waveform = self.sim.waveform();
    }
}

#[wasm_bindgen]
impl NbtSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(nbt_bytes: &[u8]) -> Result<NbtSimulator, JsValue> {
        Self::new_with_trace_limit(nbt_bytes, TRACE_LIMIT)
    }

    pub fn with_trace(nbt_bytes: &[u8], trace_enabled: bool) -> Result<NbtSimulator, JsValue> {
        Self::new_with_trace_limit(nbt_bytes, if trace_enabled { TRACE_LIMIT } else { 0 })
    }

    pub fn trace_init(nbt_bytes: &[u8]) -> Result<JsValue, JsValue> {
        let nbt = NBTRoot::from_nbt_bytes(nbt_bytes).map_err(to_js_error)?;
        let world = nbt.to_world();
        let report = match Simulator::from_preserving_torch_states_with_limits_and_trace(
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
                waveform: sim.waveform(),
            },
            Err(error) => TraceReport {
                ok: false,
                error: Some(error.message().to_owned()),
                trace: error.trace().to_vec(),
                snapshots: snapshots_to_info(error.snapshots()),
                waveform: error.waveform(),
            },
        };

        serde_wasm_bindgen::to_value(&report).map_err(to_js_error)
    }

    pub fn graph_dot(nbt_bytes: &[u8]) -> Result<JsValue, JsValue> {
        graph_dot_info(nbt_bytes, None)
    }

    pub fn graph_dot_with_outputs(
        nbt_bytes: &[u8],
        metadata_json: &str,
    ) -> Result<JsValue, JsValue> {
        let metadata =
            serde_json::from_str::<OutputMetadata>(metadata_json).map_err(to_js_error)?;
        graph_dot_info(nbt_bytes, Some(&metadata))
    }

    pub fn selected_graph_dot(
        nbt_bytes: &[u8],
        folded: bool,
        node_ids: JsValue,
    ) -> Result<JsValue, JsValue> {
        let selected_node_ids: Vec<GraphNodeId> =
            serde_wasm_bindgen::from_value(node_ids).map_err(to_js_error)?;
        let selected_node_ids = selected_node_ids.into_iter().collect::<HashSet<_>>();
        let nbt = NBTRoot::from_nbt_bytes(nbt_bytes).map_err(to_js_error)?;
        let world = nbt.to_world();
        let raw_world_graph = WorldGraphBuilder::new(&world).build();
        let source_world_graph = if folded {
            let transformer =
                WorldToLogicTransformer::new(raw_world_graph, true).map_err(to_js_error)?;
            transformer.world_graph().clone()
        } else {
            raw_world_graph
        };

        let selected_world_graph =
            source_world_graph.extract_subgraph_by_node_ids(&selected_node_ids);
        let transformer = WorldToLogicTransformer::new(selected_world_graph.clone(), true)
            .map_err(to_js_error)?;
        let folded_world_dot = transformer.world_graph().to_graphviz();
        let folded_world_dot_without_tags = transformer.world_graph().to_graphviz_without_tags();
        let logic_graph = transformer.transform().map_err(to_js_error)?;
        let graph_dot = GraphDotInfo {
            raw_world_dot: selected_world_graph.to_graphviz(),
            raw_world_dot_without_tags: selected_world_graph.to_graphviz_without_tags(),
            folded_world_dot,
            folded_world_dot_without_tags,
            logic_dot: logic_graph.to_graphviz(),
            logic_dot_without_tags: logic_graph.to_graphviz_without_tags(),
        };

        serde_wasm_bindgen::to_value(&graph_dot).map_err(to_js_error)
    }

    pub fn selected_nbt(
        nbt_bytes: &[u8],
        folded: bool,
        node_ids: JsValue,
    ) -> Result<JsValue, JsValue> {
        let selected_node_ids: Vec<GraphNodeId> =
            serde_wasm_bindgen::from_value(node_ids).map_err(to_js_error)?;
        let selected_node_ids = selected_node_ids.into_iter().collect::<HashSet<_>>();
        let nbt = NBTRoot::from_nbt_bytes(nbt_bytes).map_err(to_js_error)?;
        let world = nbt.to_world();
        let raw_world_graph = WorldGraphBuilder::new(&world).build();
        let source_world_graph = if folded {
            let transformer =
                WorldToLogicTransformer::new(raw_world_graph.clone(), true).map_err(to_js_error)?;
            transformer.world_graph().clone()
        } else {
            raw_world_graph.clone()
        };
        let selected_world = selected_world_from_graph(
            &world,
            &raw_world_graph,
            &source_world_graph,
            &selected_node_ids,
        );

        serde_wasm_bindgen::to_value(&selected_world.to_nbt()).map_err(to_js_error)
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
        let start_cycle = self
            .sim
            .snapshots()
            .last()
            .map_or(1, |snapshot| snapshot.cycle + 1);
        if let Err(error) = self.sim.change_state_with_limits(
            vec![(Position(x, y, z), is_on)],
            MAX_SIMULATION_CYCLES,
            MAX_SIMULATION_EVENTS,
        ) {
            self.refresh_trace_views_from_cycle(start_cycle);
            return Err(to_js_error(error));
        }
        self.refresh_trace_views_from_cycle(start_cycle);
        self.structure()
    }

    pub fn set_trace_enabled(&mut self, enabled: bool) {
        self.sim
            .set_trace_limit(if enabled { TRACE_LIMIT } else { 0 });
        self.clear_trace_views();
    }

    pub fn trace(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.last_trace).map_err(to_js_error)
    }

    pub fn snapshots(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.last_snapshots).map_err(to_js_error)
    }

    pub fn waveform(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.last_waveform).map_err(to_js_error)
    }

    pub fn history_trace(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.history_trace).map_err(to_js_error)
    }

    pub fn history_snapshots(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.history_snapshots).map_err(to_js_error)
    }

    pub fn history_waveform(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.history_waveform).map_err(to_js_error)
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

fn empty_waveform() -> SimulationWaveform {
    SimulationWaveform {
        cycles: Vec::new(),
        signals: Vec::new(),
    }
}

fn graph_dot_info(nbt_bytes: &[u8], metadata: Option<&OutputMetadata>) -> Result<JsValue, JsValue> {
    let nbt = NBTRoot::from_nbt_bytes(nbt_bytes).map_err(to_js_error)?;
    let world = nbt.to_world();
    let raw_world_graph = WorldGraphBuilder::new(&world).build();
    let transformer =
        WorldToLogicTransformer::new(raw_world_graph.clone(), true).map_err(to_js_error)?;
    let folded_world_dot = transformer.world_graph().to_graphviz();
    let folded_world_dot_without_tags = transformer.world_graph().to_graphviz_without_tags();
    let logic_graph = if let Some(metadata) = metadata {
        world_to_logic_with_outputs_unoptimized(&world, metadata).map_err(to_js_error)?
    } else {
        transformer.transform().map_err(to_js_error)?
    };
    let graph_dot = GraphDotInfo {
        raw_world_dot: raw_world_graph.to_graphviz(),
        raw_world_dot_without_tags: raw_world_graph.to_graphviz_without_tags(),
        folded_world_dot,
        folded_world_dot_without_tags,
        logic_dot: logic_graph.to_graphviz(),
        logic_dot_without_tags: logic_graph.to_graphviz_without_tags(),
    };

    serde_wasm_bindgen::to_value(&graph_dot).map_err(to_js_error)
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

fn selected_world_from_graph(
    world: &World,
    raw_world_graph: &redstone_compiler::graph::world::WorldGraph,
    source_world_graph: &redstone_compiler::graph::world::WorldGraph,
    selected_node_ids: &HashSet<GraphNodeId>,
) -> World3D {
    let mut positions = HashSet::new();
    let mut raw_node_ids = HashSet::new();
    for node_id in selected_node_ids {
        if let Some(position) = source_world_graph.positions.get(node_id) {
            positions.insert(*position);
        }

        let Some(node) = source_world_graph.graph.find_node_by_id(*node_id) else {
            continue;
        };
        if !matches!(node.kind, GraphNodeKind::Sequential(_)) && !node.tag.starts_with("Folded ") {
            raw_node_ids.insert(*node_id);
            continue;
        }
        for source_node_id in source_node_ids_from_tag(&node.tag) {
            if let Some(position) = raw_world_graph.positions.get(&source_node_id) {
                positions.insert(*position);
                raw_node_ids.insert(source_node_id);
            }
        }
    }

    add_edge_routing_blocks(world, raw_world_graph, &raw_node_ids, &mut positions);
    add_attached_blocks(world, &mut positions);
    compact_world_from_positions(world, &positions)
}

fn add_edge_routing_blocks(
    world: &World,
    raw_world_graph: &WorldGraph,
    raw_node_ids: &HashSet<GraphNodeId>,
    positions: &mut HashSet<Position>,
) {
    let world3d = World3D::from(world);
    let route_positions = raw_node_ids
        .iter()
        .flat_map(|source_node_id| {
            let Some(source_node) = raw_world_graph.graph.find_node_by_id(*source_node_id) else {
                return Vec::new();
            };
            let Some(source_position) = raw_world_graph.positions.get(source_node_id) else {
                return Vec::new();
            };
            let source_outputs = source_node
                .outputs
                .iter()
                .copied()
                .filter(|target_node_id| raw_node_ids.contains(target_node_id))
                .collect::<Vec<_>>();

            source_outputs
                .into_iter()
                .filter_map(|target_node_id| raw_world_graph.positions.get(&target_node_id))
                .flat_map(|target_position| {
                    edge_routing_positions(
                        *source_position,
                        *target_position,
                        &source_node.kind,
                        &world3d,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    positions.extend(route_positions);
}

fn edge_routing_positions(
    source_position: Position,
    target_position: Position,
    source_kind: &GraphNodeKind,
    world: &World3D,
) -> Vec<Position> {
    let GraphNodeKind::Block(source_block) = source_kind else {
        return Vec::new();
    };

    PlaceBound(PropagateType::Soft, source_position, source_block.direction)
        .propagation_bound(&source_block.kind, Some(world))
        .into_iter()
        .filter(|bound| world.size.bound_on(bound.position()))
        .filter(|bound| {
            bound
                .propagate_to(world)
                .into_iter()
                .any(|(_, position)| position == target_position)
        })
        .map(|bound| bound.position())
        .filter(|position| *position != source_position && *position != target_position)
        .filter(|position| world.size.bound_on(*position))
        .filter(|position| !world[*position].kind.is_air())
        .collect()
}

fn add_attached_blocks(world: &World, positions: &mut HashSet<Position>) {
    let world3d = World3D::from(world);
    let attached_positions = positions
        .iter()
        .filter_map(|position| {
            let block = world.blocks.iter().find_map(|(block_position, block)| {
                (block_position == position).then_some(block)
            })?;
            attached_block_position(*position, *block)
        })
        .filter(|position| world3d.size.bound_on(*position))
        .filter(|position| !world3d[*position].kind.is_air())
        .collect::<Vec<_>>();

    positions.extend(attached_positions);
}

fn attached_block_position(position: Position, block: Block) -> Option<Position> {
    match block.kind {
        BlockKind::Redstone { .. } | BlockKind::Repeater { .. } => position.down(),
        BlockKind::Switch { .. } | BlockKind::Torch { .. } => {
            attached_direction(position, block.direction)
        }
        BlockKind::Air
        | BlockKind::Cobble { .. }
        | BlockKind::RedstoneBlock
        | BlockKind::Piston { .. } => None,
    }
}

fn attached_direction(position: Position, direction: Direction) -> Option<Position> {
    if direction == Direction::None {
        return None;
    }

    position.walk(direction)
}

fn compact_world_from_positions(world: &World, positions: &HashSet<Position>) -> World3D {
    if positions.is_empty() {
        return World3D::new(DimSize(1, 1, 1));
    }

    let min_x = positions.iter().map(|position| position.0).min().unwrap();
    let min_y = positions.iter().map(|position| position.1).min().unwrap();
    let min_z = positions.iter().map(|position| position.2).min().unwrap();
    let max_x = positions.iter().map(|position| position.0).max().unwrap();
    let max_y = positions.iter().map(|position| position.1).max().unwrap();
    let max_z = positions.iter().map(|position| position.2).max().unwrap();
    let mut selected_world = World3D::new(DimSize(
        max_x - min_x + 1,
        max_y - min_y + 1,
        max_z - min_z + 1,
    ));

    for (position, block) in &world.blocks {
        if !positions.contains(position) {
            continue;
        }
        selected_world[Position(position.0 - min_x, position.1 - min_y, position.2 - min_z)] =
            *block;
    }

    selected_world
}

fn source_node_ids_from_tag(tag: &str) -> Vec<GraphNodeId> {
    let Some(start) = tag.rfind('[') else {
        return Vec::new();
    };
    let Some(end) = tag[start..].find(']') else {
        return Vec::new();
    };

    tag[start + 1..start + end]
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .collect()
}
