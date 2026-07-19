pub mod assembly;
pub mod candidate;
mod candidate_cache;
pub mod diagnostics;
mod free_3d;
pub mod heuristics;
pub mod ir;
pub mod physical_intent;
pub mod placer;
pub mod policy;
mod prepared_snapshot;
pub mod progress;
pub mod rcir;
pub mod router;
pub mod search;
pub mod topology;
pub mod visualize;

use std::time::{Duration, Instant};

use eyre::ContextCompat;
pub use prepared_snapshot::load_prepared_pnr_snapshot;
pub use rcir::{
    apply_pnr_spec, apply_routable_document, pnr_spec_from_config, routable_document_from_config,
};
use serde_json::{json, Value};

use crate::graph::module::{GraphModule, GraphModuleContext, GraphModuleDesign};
use crate::graph::GraphNodeKind;
use crate::ir::{
    LogicalDesign, PnrSpec, RoutableDesign, RoutableModule, RoutableModuleBody,
    RoutablePortDirection,
};
use crate::nbt::ToNBT;
use crate::output::{OutputEndpoint, PlacedWorld};
use crate::snapshot::{
    duration_ms, emit_json, emit_nbt, record as record_snapshot, SnapshotEvent, SnapshotProduct,
    SnapshotProductInfo,
};
use crate::transform::place_and_route::global_pnr::assembly::assemble_world;
use crate::transform::place_and_route::global_pnr::candidate::{
    generate_graph_module_candidates_with_progress_label,
    generate_routable_module_candidates_with_progress_label, CandidatePolicySet,
    UnitCandidateConfig,
};
pub use crate::transform::place_and_route::global_pnr::heuristics::GlobalHeuristicHooks;
use crate::transform::place_and_route::global_pnr::ir::{LayoutCandidate, PhysicalPortDirection};
pub use crate::transform::place_and_route::global_pnr::physical_intent::{
    ConstraintSatisfaction, PhysicalIntent, ResolvedPhysicalIntent,
};
use crate::transform::place_and_route::global_pnr::placer::{
    place_candidates_on_shelves, placement_candidates_resolved, placement_cost_breakdown_resolved,
    GlobalPlacementConfig, PlacedModule,
};
use crate::transform::place_and_route::global_pnr::policy::{
    GlobalPnrPolicies, GlobalPnrPreset, GlobalSearchBudget,
};
use crate::transform::place_and_route::global_pnr::progress::GlobalPnrProgress;
use crate::transform::place_and_route::global_pnr::router::{
    collect_topology_input_endpoints, collect_topology_output_endpoints,
    first_invalid_active_route, route_resolved_topology_with_order_from_prefix,
    GlobalRoutingConfig, NetOrderStrategy, RoutedNet,
};
use crate::transform::place_and_route::global_pnr::search::{
    layout_combinations, rank_child_candidates_with_preferred, select_layout_combination,
    ChildCandidatePool,
};
use crate::transform::place_and_route::global_pnr::topology::{
    NetId, ResolvedEndpoint, ResolvedPnrTopology,
};
use crate::transform::place_and_route::global_pnr::visualize::placement_bbox_wireframe_world;
use crate::transform::place_and_route::local_placer::{LocalPlacerConfig, NotRouteStrategy};
use crate::transform::place_and_route::sampling::SamplingPolicy;
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Clone, Debug)]
pub struct GlobalPnrConfig {
    pub candidate: CandidatePolicySet,
    pub placement: GlobalPlacementConfig,
    /// Optional cheap whole-design routing attempt evaluated before `routing`.
    /// It shares the exact same local candidates and placement.
    pub routing_probe: Option<GlobalRoutingConfig>,
    pub routing: GlobalRoutingConfig,
    pub routing_refinement: Option<GlobalRoutingConfig>,
    pub search: GlobalSearchConfig,
    pub show_progress: bool,
    pub verifier: Option<fn(&PlacedWorld) -> eyre::Result<()>>,
    /// Optional per-design physical constraints resolved against the stable
    /// typed topology. Search knobs remain separate from this design intent.
    pub physical_intent: Option<ResolvedPhysicalIntent>,
    /// Optional cross-process cache for structurally keyed local candidates.
    pub candidate_cache_dir: Option<std::path::PathBuf>,
    /// Experimental extension points. Hooks are named and recorded in
    /// snapshots; function pointers keep ownership/configuration lightweight.
    pub heuristic_hooks: GlobalHeuristicHooks,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlobalSearchConfig {
    pub budget: GlobalSearchBudget,
    pub policies: GlobalPnrPolicies,
}

/// Configuration that affects local candidate preparation.
///
/// A prepared design may be reused with different placement, routing, and
/// global search settings as long as this configuration remains unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PnrPrepareConfig {
    pub candidate: CandidatePolicySet,
    pub show_progress: bool,
    pub candidate_cache_dir: Option<std::path::PathBuf>,
}

impl From<&GlobalPnrConfig> for PnrPrepareConfig {
    fn from(config: &GlobalPnrConfig) -> Self {
        Self {
            candidate: config.candidate.clone(),
            show_progress: config.show_progress,
            candidate_cache_dir: config.candidate_cache_dir.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PnrPreparationSummary {
    pub module: String,
    pub instances: usize,
    pub unique_candidate_sets: usize,
    pub reused_candidate_sets: usize,
    /// Number of unique candidates physically retained by the prepared design.
    pub candidates: usize,
    /// Candidate references after expanding each set across its instances.
    pub candidate_references: usize,
    pub elapsed_ms: u64,
}

#[derive(Clone, Debug)]
struct PreparedCandidateSet {
    candidates: Vec<LayoutCandidate>,
}

#[derive(Clone, Debug)]
struct PreparedInstanceCandidateBinding {
    instance_name: String,
    candidate_set_index: usize,
    preferred_index: usize,
}

#[derive(Clone, Debug)]
enum PreparedPnrBody {
    Leaf {
        candidates: Vec<LayoutCandidate>,
    },
    Composite {
        candidate_sets: Vec<PreparedCandidateSet>,
        instance_bindings: Vec<PreparedInstanceCandidateBinding>,
    },
}

/// The owned boundary between local candidate generation and global PnR.
///
#[derive(Clone, Debug)]
pub struct PreparedPnrDesign {
    module_name: String,
    topology: ResolvedPnrTopology,
    prepare_config: PnrPrepareConfig,
    body: PreparedPnrBody,
    summary: PnrPreparationSummary,
    snapshot_intent: Option<ResolvedPhysicalIntent>,
    snapshot_pnr: Option<PnrSpec>,
}

impl PreparedPnrDesign {
    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    pub fn summary(&self) -> &PnrPreparationSummary {
        &self.summary
    }

    pub fn topology(&self) -> &ResolvedPnrTopology {
        &self.topology
    }

    pub fn snapshot_intent(&self) -> Option<&ResolvedPhysicalIntent> {
        self.snapshot_intent.as_ref()
    }

    pub fn apply_snapshot_config(&self, config: &mut GlobalPnrConfig) -> eyre::Result<()> {
        if let Some(spec) = &self.snapshot_pnr {
            apply_pnr_spec(spec, config)?;
        }
        config.candidate = self.prepare_config.candidate.clone();
        Ok(())
    }

    /// A deterministic, migration-only observation point for candidate and
    /// topology parity. It is deliberately not a persistent cache key.
    pub fn parity_signature(&self) -> Value {
        let candidate_sets = match &self.body {
            PreparedPnrBody::Leaf { candidates } => vec![json!({
                "instance": self.module_name,
                "preferred_index": 0,
                "candidate_hashes": candidates.iter().map(candidate_parity_hash).collect::<Vec<_>>(),
            })],
            PreparedPnrBody::Composite {
                candidate_sets,
                instance_bindings,
            } => instance_bindings
                .iter()
                .map(|binding| {
                    let set = &candidate_sets[binding.candidate_set_index];
                    json!({
                        "instance": binding.instance_name,
                        "candidate_set_index": binding.candidate_set_index,
                        "preferred_index": binding.preferred_index,
                        "candidate_hashes": set.candidates.iter().map(candidate_parity_hash).collect::<Vec<_>>(),
                    })
                })
                .collect(),
        };
        json!({
            "format": "redstone-compiler.prepared-pnr-parity.v2",
            "module": self.module_name,
            "resolved_topology_hash": debug_parity_hash(&self.topology),
            "candidate_sets": candidate_sets,
        })
    }

    fn ensure_compatible(&self, config: &GlobalPnrConfig) -> eyre::Result<()> {
        if self.prepare_config.candidate != config.candidate {
            eyre::bail!(
                "prepared PnR candidate configuration does not match; regenerate local candidates"
            );
        }
        Ok(())
    }

    fn ranked_candidate_pools(&self, limit: usize) -> Vec<ChildCandidatePool> {
        let PreparedPnrBody::Composite {
            candidate_sets,
            instance_bindings,
        } = &self.body
        else {
            return Vec::new();
        };
        instance_bindings
            .iter()
            .map(|binding| {
                let candidates = relabel_candidates(
                    &candidate_sets[binding.candidate_set_index].candidates,
                    &binding.instance_name,
                );
                rank_child_candidates_with_preferred(
                    &binding.instance_name,
                    candidates,
                    limit.max(1),
                    binding.preferred_index,
                )
            })
            .collect()
    }
}

fn debug_parity_hash(value: &impl std::fmt::Debug) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in format!("{value:?}").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn candidate_parity_hash(candidate: &LayoutCandidate) -> String {
    let mut blocks = candidate.world.iter_block();
    blocks.sort_by_key(|(position, _)| *position);
    let mut ports = candidate.ports.clone();
    ports.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.position.cmp(&right.position))
    });
    let mut occupied_cells = candidate.occupied_cells.iter().copied().collect::<Vec<_>>();
    occupied_cells.sort();
    let mut blocked_cells = candidate.blocked_cells.iter().copied().collect::<Vec<_>>();
    blocked_cells.sort();
    debug_parity_hash(&(
        candidate.bbox,
        blocks,
        ports,
        occupied_cells,
        blocked_cells,
        &candidate.cost,
    ))
}

impl Default for GlobalSearchConfig {
    fn default() -> Self {
        GlobalPnrPreset::Balanced.search_config()
    }
}

pub struct GlobalPnrResult {
    pub topology: ResolvedPnrTopology,
    pub placed_world: PlacedWorld,
    pub placement_bbox_world: World3D,
    pub selected_candidates: Vec<LayoutCandidate>,
    pub placed_modules: Vec<PlacedModule>,
    pub routed_nets: Vec<RoutedNet>,
    pub placement_cost:
        crate::transform::place_and_route::global_pnr::policy::PlacementCostBreakdown,
    pub weighted_placement_cost: usize,
    pub config_snapshot: Value,
    pub physical_intent: Option<ResolvedPhysicalIntent>,
    pub constraint_report: Vec<ConstraintSatisfaction>,
}

impl SnapshotProduct for GlobalPnrResult {
    fn emit_snapshot(&self, design_name: &str) -> eyre::Result<SnapshotProductInfo> {
        let artifact_name = snapshot_artifact_name(design_name);
        let final_nbt = format!("{artifact_name}.nbt");
        emit_nbt(&final_nbt, self.placed_world.world.to_nbt())?;
        emit_nbt("placement-bboxes.nbt", self.placement_bbox_world.to_nbt())?;
        emit_json(
            "interface.json",
            json!({
                "format": "redstone-compiler.interface.v1",
                "inputs": self.placed_world.inputs,
                "outputs": self.placed_world.outputs,
            }),
        )?;
        emit_json("pnr/config.json", self.config_snapshot.clone())?;
        if let Some(intent) = &self.physical_intent {
            emit_json("intent/resolved.json", intent)?;
            emit_json("intent/report.json", &self.constraint_report)?;
        }

        for (index, placed) in self.placed_modules.iter().enumerate() {
            let Some(candidate) = self.selected_candidates.get(placed.candidate_index) else {
                continue;
            };
            let resolved_instance = self.topology.instance_by_name(&placed.module_name);
            let resolved_definition = resolved_instance
                .and_then(|instance| self.topology.definition(instance.definition));
            let directory = format!(
                "instances/{index:02}-{}",
                snapshot_artifact_name(&placed.module_name)
            );
            emit_nbt(
                format!("{directory}/circuit.nbt"),
                normalized_candidate_world(candidate).to_nbt(),
            )?;
            let ports = candidate
                .ports
                .iter()
                .map(|port| {
                    let local = Position(
                        port.position.0 - candidate.bbox.min.0,
                        port.position.1 - candidate.bbox.min.1,
                        port.position.2 - candidate.bbox.min.2,
                    );
                    let global = Position(
                        placed.origin.0 + local.0,
                        placed.origin.1 + local.1,
                        placed.origin.2 + local.2,
                    );
                    json!({
                        "name": port.name,
                        "direction": format!("{:?}", port.direction),
                        "connection": format!("{:?}", port.connection),
                        "local_position": position_json(local),
                        "global_position": position_json(global),
                    })
                })
                .collect::<Vec<_>>();
            emit_json(
                format!("{directory}/instance.json"),
                json!({
                    "instance": placed.module_name,
                    "module": candidate.module_name,
                    "instance_id": resolved_instance.map(|instance| instance.id.0),
                    "instance_key": resolved_instance.map(|instance| &instance.key),
                    "definition_id": resolved_definition.map(|definition| definition.id.0),
                    "definition_key": resolved_definition.map(|definition| &definition.key),
                    "candidate_index": placed.candidate_index,
                    "global_origin": position_json(placed.origin),
                    "global_bbox": {
                        "min": position_json(placed.origin),
                        "max": [
                            placed.origin.0 + placed.bbox.width() - 1,
                            placed.origin.1 + placed.bbox.depth() - 1,
                            placed.origin.2 + placed.bbox.height() - 1,
                        ],
                    },
                    "local_size": [
                        candidate.bbox.width(),
                        candidate.bbox.depth(),
                        candidate.bbox.height(),
                    ],
                    "cost": {
                        "blocks": candidate.cost.block_count,
                        "bbox_volume": candidate.cost.bbox_volume,
                    },
                    "ports": ports,
                }),
            )?;
        }

        let route_descriptions = self
            .routed_nets
            .iter()
            .enumerate()
            .map(|(index, route)| {
                let resolved_net = route
                    .net_id
                    .and_then(|id| self.topology.nets.get(id.0))
                    .or_else(|| {
                        route
                            .source_label
                            .as_deref()
                            .and_then(|label| self.topology.net_by_driver_label(label))
                    });
                let resolved_sink = route.sink_endpoint.as_ref().or_else(|| {
                    resolved_net.and_then(|net| {
                        route
                            .sink_label
                            .as_deref()
                            .and_then(|label| self.topology.sink_by_label(net, label))
                    })
                });
                json!({
                    "index": index,
                    "net_id": resolved_net.map(|net| net.id.0),
                    "net_key": resolved_net.map(|net| &net.key),
                    "source_endpoint": route.source_endpoint,
                    "sink_endpoint": resolved_sink,
                    "source_label": route.source_label,
                    "sink_label": route.sink_label,
                    "source": position_json(route.source),
                    "sink": position_json(route.sink),
                    "path": route.path.iter().copied().map(position_json).collect::<Vec<_>>(),
                    "path_length": route.path.len(),
                    "block_count": route.blocks.len(),
                    "blocks": route.blocks.iter().map(|(position, _)| position_json(*position)).collect::<Vec<_>>(),
                    "required_powered_positions": route.required_powered_positions.iter().copied().map(position_json).collect::<Vec<_>>(),
                    "required_released_positions": route.required_released_positions.iter().copied().map(position_json).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        emit_json(
            "routes/routes.json",
            json!({
                "format": "redstone-compiler.routes.v1",
                "routes": route_descriptions,
            }),
        )?;
        emit_nbt(
            "routes/routes.nbt",
            routed_net_world(&self.routed_nets).to_nbt(),
        )?;

        Ok(SnapshotProductInfo {
            top_module: design_name.to_owned(),
            final_nbt,
            summary: json!({
                "world": {
                    "size": [
                        self.placed_world.world.size.0,
                        self.placed_world.world.size.1,
                        self.placed_world.world.size.2,
                    ],
                    "non_air_blocks": self.placed_world.world.iter_block().len(),
                },
                "interface": {
                    "inputs": self.placed_world.inputs.len(),
                    "outputs": self.placed_world.outputs.len(),
                },
                "topology": {
                    "definitions": self.topology.definitions.len(),
                    "instances": self.topology.instances.len(),
                    "nets": self.topology.nets.len(),
                },
                "physical_intent": {
                    "constraints": self.physical_intent.as_ref().map_or(0, |intent| intent.constraints.len()),
                    "satisfied": self.constraint_report.iter().filter(|item| matches!(item.status, physical_intent::ConstraintStatus::Satisfied)).count(),
                    "violated": self.constraint_report.iter().filter(|item| matches!(item.status, physical_intent::ConstraintStatus::Violated)).count(),
                },
                "placement": {
                    "instances": self.placed_modules.len(),
                    "volume": self.placement_cost.placement_volume,
                    "xy_footprint": self.placement_cost.xy_footprint,
                    "height": self.placement_cost.height_span,
                    "estimated_wire_length": self.placement_cost.estimated_wire_length,
                    "vertical_distance": self.placement_cost.vertical_distance,
                    "congestion": self.placement_cost.routing_congestion,
                    "weighted_total": self.weighted_placement_cost,
                },
                "routing": {
                    "routes": self.routed_nets.len(),
                    "path_length": self.routed_nets.iter().map(|route| route.path.len()).sum::<usize>(),
                    "blocks": self.routed_nets.iter().map(|route| route.blocks.len()).sum::<usize>(),
                },
            }),
        })
    }
}

fn snapshot_artifact_name(name: &str) -> String {
    let safe = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    if safe.is_empty() {
        "design".to_owned()
    } else {
        safe
    }
}

fn normalized_candidate_world(candidate: &LayoutCandidate) -> World3D {
    let mut world = World3D::new(DimSize(
        candidate.bbox.width(),
        candidate.bbox.depth(),
        candidate.bbox.height(),
    ));
    for (position, block) in candidate.world.iter_block() {
        let local = Position(
            position.0 - candidate.bbox.min.0,
            position.1 - candidate.bbox.min.1,
            position.2 - candidate.bbox.min.2,
        );
        world[local] = block;
    }
    world
}

fn routed_net_world(routes: &[RoutedNet]) -> World3D {
    let mut max = Position(0, 0, 0);
    for (position, _) in routes.iter().flat_map(|route| &route.blocks) {
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }
    let mut world = World3D::new(DimSize(max.0 + 1, max.1 + 1, max.2 + 1));
    for (position, block) in routes.iter().flat_map(|route| &route.blocks) {
        world[*position] = *block;
    }
    world
}

fn position_json(position: Position) -> Value {
    json!([position.0, position.1, position.2])
}

fn global_pnr_config_snapshot(config: &GlobalPnrConfig) -> Value {
    json!({
        "format": "redstone-compiler.pnr-config.v1",
        "effective_design_profile": pnr_spec_from_config(config),
        "runtime": {
            "show_progress": config.show_progress,
            "verifier_enabled": config.verifier.is_some(),
            "persistent_cache_enabled": config.candidate_cache_dir.is_some(),
            "experimental_hooks": config.heuristic_hooks.names(),
        }
    })
}

impl Default for GlobalPnrConfig {
    fn default() -> Self {
        Self {
            candidate: CandidatePolicySet::default(),
            placement: GlobalPlacementConfig::default(),
            routing_probe: None,
            routing: GlobalRoutingConfig::default(),
            routing_refinement: None,
            search: GlobalSearchConfig::default(),
            show_progress: true,
            verifier: None,
            physical_intent: None,
            candidate_cache_dir: None,
            heuristic_hooks: GlobalHeuristicHooks::default(),
        }
    }
}

pub fn place_and_route_module(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    Ok(place_and_route_module_with_outputs(context, module, config)?.world)
}

pub fn place_and_route_design(
    design: &GraphModuleDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    Ok(place_and_route_design_with_outputs(design, config)?.world)
}

pub fn place_and_route_design_with_outputs(
    design: &GraphModuleDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<PlacedWorld> {
    let GlobalPnrResult { placed_world, .. } =
        place_and_route_design_with_visualization(design, config)?;
    Ok(placed_world)
}

pub fn place_and_route_module_with_outputs(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
) -> eyre::Result<PlacedWorld> {
    let GlobalPnrResult { placed_world, .. } =
        place_and_route_module_with_visualization(context, module, config)?;
    Ok(placed_world)
}

pub fn place_and_route_design_with_visualization(
    design: &GraphModuleDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<GlobalPnrResult> {
    if crate::snapshot::is_active() {
        let routable = RoutableDesign::from_graph_module_design(design)?;
        let document = routable_document_from_config(&routable, config)?;
        crate::snapshot::emit_text("ir/routable.rcir", document.to_string())?;
        crate::snapshot::emit_json("ir/routable.json", &routable)?;
    }
    place_and_route_module_with_visualization(&design.context, design.top_module(), config)
}

pub fn place_and_route_routable_design_with_visualization(
    design: &RoutableDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<GlobalPnrResult> {
    design.validate()?;
    if crate::snapshot::is_active() {
        let document = routable_document_from_config(design, config)?;
        crate::snapshot::emit_text("ir/routable.rcir", document.to_string())?;
        crate::snapshot::emit_json("ir/routable.json", design)?;
    }
    let prepared = prepare_routable_design_for_global_pnr(design, &PnrPrepareConfig::from(config))?;
    run_prepared_pnr_with_visualization(&prepared, config)
}

pub fn prepare_routable_design_for_global_pnr(
    design: &RoutableDesign,
    config: &PnrPrepareConfig,
) -> eyre::Result<PreparedPnrDesign> {
    design.validate()?;
    let topology = ResolvedPnrTopology::from_routable(design)?;
    let module = design
        .module(&design.top)
        .with_context(|| format!("missing top Routable module `{}`", design.top))?;
    prepare_routable_module_with_topology(design, module, topology, config)
}

pub fn place_and_route_routable_design(
    design: &RoutableDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    Ok(
        place_and_route_routable_design_with_visualization(design, config)?
            .placed_world
            .world,
    )
}

pub fn place_and_route_logical_design_with_visualization(
    design: &LogicalDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<GlobalPnrResult> {
    design.validate()?;
    if crate::snapshot::is_active() {
        crate::snapshot::emit_text("ir/logical.rcir", design.to_string())?;
        crate::snapshot::emit_json("ir/logical.json", design)?;
    }
    place_and_route_routable_design_with_visualization(&design.lower_to_routable()?, config)
}

pub fn place_and_route_logical_design(
    design: &LogicalDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    Ok(
        place_and_route_logical_design_with_visualization(design, config)?
            .placed_world
            .world,
    )
}

pub fn place_and_route_module_with_visualization(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
) -> eyre::Result<GlobalPnrResult> {
    let prepared = prepare_module_for_global_pnr(context, module, &PnrPrepareConfig::from(config))?;
    run_prepared_pnr_with_visualization(&prepared, config)
}

pub fn prepare_design_for_global_pnr(
    design: &GraphModuleDesign,
    config: &PnrPrepareConfig,
) -> eyre::Result<PreparedPnrDesign> {
    prepare_module_for_global_pnr(&design.context, design.top_module(), config)
}

pub fn prepare_module_for_global_pnr(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &PnrPrepareConfig,
) -> eyre::Result<PreparedPnrDesign> {
    let topology = ResolvedPnrTopology::from_legacy_graph_module(context, module)?;
    prepare_module_with_topology(context, module, topology, config)
}

fn prepare_module_with_topology(
    context: &GraphModuleContext,
    module: &GraphModule,
    topology: ResolvedPnrTopology,
    config: &PnrPrepareConfig,
) -> eyre::Result<PreparedPnrDesign> {
    let started = Instant::now();
    let progress = GlobalPnrProgress::new(config.show_progress, module.name.clone());
    if module.graph.is_some() {
        progress.stage(1, 4, "generate leaf layout candidates");
        let candidate_config = config.candidate.effective_for_definition(&module.name);
        let candidate_config = candidate_config_for_child(module, &candidate_config);
        let candidates = generate_graph_module_candidates_with_progress_label(
            module,
            &candidate_config,
            config.show_progress.then_some(module.name.as_str()),
        )?;
        if candidates.is_empty() {
            eyre::bail!("graph-backed module produced no layout candidates");
        }
        let summary = PnrPreparationSummary {
            module: module.name.clone(),
            instances: 1,
            unique_candidate_sets: 1,
            reused_candidate_sets: 0,
            candidates: candidates.len(),
            candidate_references: candidates.len(),
            elapsed_ms: duration_ms(started.elapsed()),
        };
        progress.summary(format!(
            "local preparation completed: candidate_sets=1 candidates={} elapsed={:.2?}",
            candidates.len(),
            started.elapsed()
        ));
        let prepared = PreparedPnrDesign {
            module_name: module.name.clone(),
            topology,
            prepare_config: config.clone(),
            body: PreparedPnrBody::Leaf { candidates },
            summary,
            snapshot_intent: None,
            snapshot_pnr: None,
        };
        emit_prepared_pnr_snapshot(&prepared)?;
        return Ok(prepared);
    }

    progress.stage(1, 4, "generate child layout candidates");
    let (candidate_sets, instance_bindings, mut summary) =
        prepare_child_candidate_sets(context, module, config, &progress)?;
    summary.elapsed_ms = duration_ms(started.elapsed());
    let prepared = PreparedPnrDesign {
        module_name: module.name.clone(),
        topology,
        prepare_config: config.clone(),
        body: PreparedPnrBody::Composite {
            candidate_sets,
            instance_bindings,
        },
        summary,
        snapshot_intent: None,
        snapshot_pnr: None,
    };
    emit_prepared_pnr_snapshot(&prepared)?;
    Ok(prepared)
}

fn prepare_routable_module_with_topology(
    design: &RoutableDesign,
    module: &RoutableModule,
    topology: ResolvedPnrTopology,
    config: &PnrPrepareConfig,
) -> eyre::Result<PreparedPnrDesign> {
    let started = Instant::now();
    let progress = GlobalPnrProgress::new(config.show_progress, module.name.clone());
    if matches!(module.body, RoutableModuleBody::Leaf { .. }) {
        progress.stage(1, 4, "generate leaf layout candidates");
        let candidate_config = config.candidate.effective_for_definition(&module.name);
        let candidate_config = candidate_config_for_routable_child(module, &candidate_config);
        let candidates = generate_routable_module_candidates_with_progress_label(
            module,
            &candidate_config,
            config.show_progress.then_some(module.name.as_str()),
        )?;
        if candidates.is_empty() {
            eyre::bail!("Routable leaf module produced no layout candidates");
        }
        let summary = PnrPreparationSummary {
            module: module.name.clone(),
            instances: 1,
            unique_candidate_sets: 1,
            reused_candidate_sets: 0,
            candidates: candidates.len(),
            candidate_references: candidates.len(),
            elapsed_ms: duration_ms(started.elapsed()),
        };
        let prepared = PreparedPnrDesign {
            module_name: module.name.clone(),
            topology,
            prepare_config: config.clone(),
            body: PreparedPnrBody::Leaf { candidates },
            summary,
            snapshot_intent: None,
            snapshot_pnr: None,
        };
        emit_prepared_pnr_snapshot(&prepared)?;
        return Ok(prepared);
    }

    progress.stage(1, 4, "generate child layout candidates");
    let (candidate_sets, instance_bindings, mut summary) =
        prepare_routable_child_candidate_sets(design, module, config, &progress)?;
    summary.elapsed_ms = duration_ms(started.elapsed());
    let prepared = PreparedPnrDesign {
        module_name: module.name.clone(),
        topology,
        prepare_config: config.clone(),
        body: PreparedPnrBody::Composite {
            candidate_sets,
            instance_bindings,
        },
        summary,
        snapshot_intent: None,
        snapshot_pnr: None,
    };
    emit_prepared_pnr_snapshot(&prepared)?;
    Ok(prepared)
}

pub fn run_prepared_pnr(
    prepared: &PreparedPnrDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    Ok(run_prepared_pnr_with_visualization(prepared, config)?
        .placed_world
        .world)
}

pub fn run_prepared_pnr_with_visualization(
    prepared: &PreparedPnrDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<GlobalPnrResult> {
    prepared.ensure_compatible(config)?;
    if let Some(intent) = &config.physical_intent {
        intent.validate(&prepared.topology)?;
    }
    let started = Instant::now();
    let progress = GlobalPnrProgress::new(config.show_progress, prepared.module_name.clone());
    progress.summary(format!(
        "reusing prepared local candidates: candidate_sets={} stored_candidates={} instance_references={}",
        prepared.summary.unique_candidate_sets,
        prepared.summary.candidates,
        prepared.summary.candidate_references,
    ));

    if let PreparedPnrBody::Leaf { candidates } = &prepared.body {
        return run_prepared_leaf(&prepared.topology, candidates, config, &progress, started);
    }

    let candidate_pools =
        prepared.ranked_candidate_pools(config.search.budget.max_candidates_per_child);

    progress.stage(2, 4, "search child layouts, placements, and routes");
    let (candidates, placed, routed_nets) =
        search_layout_combinations(&prepared.topology, &candidate_pools, config, &progress)?;

    progress.stage(3, 4, "assemble world and collect outputs");
    let inputs = collect_topology_input_endpoints(&prepared.topology, &routed_nets);
    let outputs = collect_topology_output_endpoints(&prepared.topology, &candidates, &placed);
    let world = assemble_world(&candidates, &placed, &routed_nets)?;
    let placed_world = PlacedWorld {
        world,
        inputs,
        outputs,
    };
    let placement_bbox_world = placement_bbox_wireframe_world(&placed);
    let placement_cost = placement_cost_breakdown_resolved(
        &prepared.topology,
        &candidates,
        &placed,
        config.placement.congestion,
    )?;
    let weighted_placement_cost = placement_cost.weighted_total(config.placement.cost_weights);
    let constraint_report = config
        .physical_intent
        .as_ref()
        .map_or_else(Vec::new, |intent| {
            intent.evaluate(&prepared.topology, &candidates, &placed, &routed_nets)
        });

    progress.stage(4, 4, "complete");
    progress.summary(format!(
        "global PnR completed: outputs={} routes={} global_elapsed={:.2?} preparation_elapsed={}ms",
        placed_world.outputs.len(),
        routed_nets.len(),
        started.elapsed(),
        prepared.summary.elapsed_ms,
    ));

    Ok(GlobalPnrResult {
        topology: prepared.topology.clone(),
        placed_world,
        placement_bbox_world,
        selected_candidates: candidates,
        placed_modules: placed,
        routed_nets,
        placement_cost,
        weighted_placement_cost,
        config_snapshot: global_pnr_config_snapshot(config),
        physical_intent: config.physical_intent.clone(),
        constraint_report,
    })
}

pub fn emit_prepared_pnr_snapshot(prepared: &PreparedPnrDesign) -> eyre::Result<()> {
    let candidate_sets = match &prepared.body {
        PreparedPnrBody::Leaf { candidates } => {
            vec![(prepared.module_name.clone(), candidates.len(), 0)]
        }
        PreparedPnrBody::Composite {
            candidate_sets,
            instance_bindings,
        } => instance_bindings
            .iter()
            .map(|binding| {
                (
                    binding.instance_name.clone(),
                    candidate_sets[binding.candidate_set_index].candidates.len(),
                    binding.preferred_index,
                )
            })
            .collect(),
    };
    emit_preparation_summary(
        &prepared.summary,
        &candidate_sets,
        prepared.parity_signature(),
        &prepared.topology,
    )?;
    prepared_snapshot::emit_candidate_library(prepared)
}

fn emit_preparation_summary(
    summary: &PnrPreparationSummary,
    candidate_sets: &[(String, usize, usize)],
    parity_signature: Value,
    topology: &ResolvedPnrTopology,
) -> eyre::Result<()> {
    if !crate::snapshot::is_active() {
        return Ok(());
    }
    emit_json("pnr/topology.json", topology)?;
    emit_json(
        "pnr/preparation.json",
        json!({
            "format": "redstone-compiler.prepared-pnr.v1",
            "module": summary.module,
            "instances": summary.instances,
            "unique_candidate_sets": summary.unique_candidate_sets,
            "reused_candidate_sets": summary.reused_candidate_sets,
            "candidates": summary.candidates,
            "candidate_references": summary.candidate_references,
            "elapsed_ms": summary.elapsed_ms,
            "parity_signature": parity_signature,
            "candidate_sets": candidate_sets.iter().map(|(instance, candidates, preferred_index)| json!({
                "instance": instance,
                "candidates": candidates,
                "preferred_index": preferred_index,
            })).collect::<Vec<_>>(),
        }),
    )
}

#[derive(Clone)]
struct RankedRoutingDecision {
    routed_count: usize,
    sequence: usize,
    placement_index: usize,
    order_index: usize,
    prefix: Vec<RoutedNet>,
    semantic_feedback_nets: Vec<NetId>,
}

struct RoutingSearchProgress {
    started: Instant,
    last_reported: Instant,
    best_routed: usize,
    routing_failures: usize,
    assembly_failures: usize,
    contract_failures: usize,
    verifier_failures: usize,
}

impl RoutingSearchProgress {
    const REPORT_INTERVAL: Duration = Duration::from_secs(10);

    fn new() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            last_reported: now,
            best_routed: 0,
            routing_failures: 0,
            assembly_failures: 0,
            contract_failures: 0,
            verifier_failures: 0,
        }
    }

    fn observe_routes(&mut self, routed: usize) {
        self.best_routed = self.best_routed.max(routed);
    }

    fn maybe_report(
        &mut self,
        progress: &GlobalPnrProgress,
        attempt: usize,
        total_attempts: usize,
    ) {
        if self.last_reported.elapsed() < Self::REPORT_INTERVAL {
            return;
        }
        record_snapshot(SnapshotEvent::RoutingProgress {
            attempt,
            total_attempts,
            best_routes: self.best_routed,
            routing_failures: self.routing_failures,
            contract_failures: self.contract_failures,
            elapsed_ms: duration_ms(self.started.elapsed()),
        });
        progress.summary(format!(
            "routing search progress: attempts={attempt}/{total_attempts} best_routes={} routing_failures={} assembly_failures={} contract_failures={} verifier_failures={} elapsed={:.2?}",
            self.best_routed,
            self.routing_failures,
            self.assembly_failures,
            self.contract_failures,
            self.verifier_failures,
            self.started.elapsed()
        ));
        self.last_reported = Instant::now();
    }
}

fn reroute_last_source_group(routes: &[RoutedNet]) -> Vec<RoutedNet> {
    if let Some(net_id) = routes.iter().rev().find_map(internal_route_net_id) {
        return routes
            .iter()
            .filter(|route| route.net_id != Some(net_id))
            .cloned()
            .collect();
    }
    let Some(source_label) = routes
        .iter()
        .rev()
        .filter_map(|route| route.source_label.as_deref())
        .find(|label| label.contains('.'))
    else {
        let mut prefix = routes.to_vec();
        prefix.pop();
        return prefix;
    };
    routes
        .iter()
        .filter(|route| route.source_label.as_deref() != Some(source_label))
        .cloned()
        .collect()
}

fn internal_route_net_id(route: &RoutedNet) -> Option<NetId> {
    matches!(
        route.source_endpoint,
        Some(ResolvedEndpoint::InstancePort { .. })
    )
    .then_some(route.net_id)
    .flatten()
}

fn reroute_source_group(
    routes: &[RoutedNet],
    net_id: Option<NetId>,
    source_label: Option<&str>,
) -> Vec<RoutedNet> {
    if let Some(net_id) = net_id {
        return routes
            .iter()
            .filter(|route| route.net_id != Some(net_id))
            .cloned()
            .collect();
    }
    let Some(source_label) = source_label else {
        return reroute_last_source_group(routes);
    };
    routes
        .iter()
        .filter(|route| route.source_label.as_deref() != Some(source_label))
        .cloned()
        .collect()
}

fn reroute_untried_source_groups(
    routes: &[RoutedNet],
    previously_tried: &[NetId],
    count: usize,
) -> (Vec<RoutedNet>, Vec<NetId>) {
    let net_ids = routes
        .iter()
        .rev()
        .filter_map(internal_route_net_id)
        .filter(|net_id| !previously_tried.contains(net_id))
        .fold(Vec::<NetId>::new(), |mut selected, net_id| {
            if !selected.contains(&net_id) && selected.len() < count {
                selected.push(net_id);
            }
            selected
        });
    let net_ids = if net_ids.is_empty() {
        routes.iter().rev().filter_map(internal_route_net_id).fold(
            Vec::<NetId>::new(),
            |mut selected, net_id| {
                if !selected.contains(&net_id) && selected.len() < count {
                    selected.push(net_id);
                }
                selected
            },
        )
    } else {
        net_ids
    };
    if net_ids.is_empty() {
        return (reroute_last_source_group(routes), previously_tried.to_vec());
    }
    let prefix = routes
        .iter()
        .filter(|route| !route.net_id.is_some_and(|id| net_ids.contains(&id)))
        .cloned()
        .collect();
    let mut tried = previously_tried.to_vec();
    tried.extend(net_ids);
    (prefix, tried)
}

fn route_first_successful_placement(
    topology: &ResolvedPnrTopology,
    candidates: &[LayoutCandidate],
    placement_attempts: Vec<Vec<PlacedModule>>,
    config: &GlobalPnrConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<(Vec<PlacedModule>, Vec<RoutedNet>)> {
    let mut last_error = None;
    let order_strategies = if config.search.policies.net_order_strategies.is_empty() {
        vec![NetOrderStrategy::Criticality]
    } else {
        config.search.policies.net_order_strategies.clone()
    };
    let mut routing_configs = Vec::with_capacity(3);
    if let Some(probe) = config.routing_probe {
        routing_configs.push(probe);
    }
    if !routing_configs.contains(&config.routing) {
        routing_configs.push(config.routing);
    }
    if let Some(refinement) = config.routing_refinement {
        for round in 0..config.search.budget.max_refinement_rounds.max(1) {
            let mut refinement = refinement;
            if let crate::transform::place_and_route::global_pnr::router::GlobalRoutingStrategy::GreedyBeam {
                variant_seed,
                ..
            } = &mut refinement.strategy
            {
                *variant_seed = variant_seed.wrapping_add(round as u64 + 1);
            }
            routing_configs.push(refinement);
        }
    }
    let order_count = order_strategies.len();
    let all_decisions = (0..placement_attempts.len())
        .flat_map(|placement_index| {
            (0..order_count).map(move |order_index| RankedRoutingDecision {
                routed_count: 0,
                sequence: placement_index * order_count + order_index,
                placement_index,
                order_index,
                prefix: Vec::new(),
                semantic_feedback_nets: Vec::new(),
            })
        })
        .collect::<Vec<_>>();
    let promoted_limit = config
        .search
        .budget
        .max_detailed_routing_attempts
        .max(1)
        .min(all_decisions.len());
    let refined_limit = config
        .search
        .budget
        .max_refined_routing_attempts
        .max(1)
        .min(promoted_limit);
    let total_attempts = all_decisions.len()
        + usize::from(routing_configs.len() > 1).saturating_mul(promoted_limit)
        + routing_configs
            .len()
            .saturating_sub(2)
            .saturating_mul(refined_limit);
    let mut previous_scores = Vec::<RankedRoutingDecision>::new();
    let mut attempt_serial = 0usize;
    let mut routing_progress = RoutingSearchProgress::new();
    progress.summary(format!(
        "routing search started: placements={} net_orders={} routing_stages={} max_attempts={total_attempts}",
        placement_attempts.len(),
        order_strategies.len(),
        routing_configs.len()
    ));

    for (routing_index, routing_config) in routing_configs.iter().enumerate() {
        let decisions = if routing_index == 0 || routing_configs.len() == 1 {
            all_decisions.clone()
        } else {
            previous_scores.sort_by_key(|decision| {
                (std::cmp::Reverse(decision.routed_count), decision.sequence)
            });
            let stage_limit = if routing_index >= 2 {
                refined_limit
            } else {
                promoted_limit
            };
            previous_scores.iter().take(stage_limit).cloned().collect()
        };
        let mut next_scores = Vec::<RankedRoutingDecision>::new();

        for decision in decisions {
            attempt_serial += 1;
            let placement_index = decision.placement_index;
            let order_index = decision.order_index;
            let placed = &placement_attempts[placement_index];
            let order_strategy = order_strategies[order_index];
            progress.attempt(
                attempt_serial,
                total_attempts,
                format!(
                    "route placement {} with {order_strategy:?} / {:?}",
                    placement_index + 1,
                    routing_config.strategy
                ),
            );
            let route_started = Instant::now();
            let routed_nets = match route_resolved_topology_with_order_from_prefix(
                topology,
                config.physical_intent.as_ref(),
                &config.heuristic_hooks,
                candidates,
                placed,
                routing_config,
                order_strategy,
                progress,
                &decision.prefix,
            ) {
                Ok(routed_nets) => {
                    progress.detail(format!(
                        "routing attempt {attempt_serial} completed in {:.2?}",
                        route_started.elapsed()
                    ));
                    routed_nets
                }
                Err(failure) => {
                    routing_progress.routing_failures += 1;
                    let routed_count = failure.routed_nets.len();
                    routing_progress.observe_routes(routed_count);
                    progress.detail(format!(
                        "routing attempt {attempt_serial} exhausted after {routed_count} route(s) in {:.2?}",
                        route_started.elapsed()
                    ));
                    if routing_index + 1 < routing_configs.len() {
                        next_scores.push(RankedRoutingDecision {
                            routed_count,
                            sequence: next_scores.len(),
                            placement_index,
                            order_index,
                            prefix: failure.routed_nets.clone(),
                            semantic_feedback_nets: decision.semantic_feedback_nets.clone(),
                        });
                    }
                    save_failed_route_base_world(topology, attempt_serial, candidates, placed);
                    progress.detail(format!(
                        "placement attempt {attempt_serial} failed: {}",
                        failure.error
                    ));
                    last_error = Some(failure.error);
                    routing_progress.maybe_report(progress, attempt_serial, total_attempts);
                    continue;
                }
            };
            routing_progress.observe_routes(routed_nets.len());

            if let Some(intent) = &config.physical_intent
                && let Err(error) = intent.validate_routes(&routed_nets)
            {
                routing_progress.contract_failures += 1;
                if routing_index + 1 < routing_configs.len() {
                    next_scores.push(RankedRoutingDecision {
                        routed_count: routed_nets.len(),
                        sequence: next_scores.len(),
                        placement_index,
                        order_index,
                        prefix: Vec::new(),
                        semantic_feedback_nets: decision.semantic_feedback_nets.clone(),
                    });
                }
                progress.detail(format!(
                    "routing attempt {attempt_serial} violated physical intent: {error}"
                ));
                last_error = Some(error);
                routing_progress.maybe_report(progress, attempt_serial, total_attempts);
                continue;
            }
            if let Some((hook_name, error)) = config
                .heuristic_hooks
                .route_validators
                .iter()
                .find_map(|hook| {
                    (hook.validate)(topology, &routed_nets)
                        .err()
                        .map(|error| (hook.name, error))
                })
            {
                routing_progress.contract_failures += 1;
                let error =
                    eyre::eyre!("route validator hook `{hook_name}` rejected attempt: {error}");
                progress.detail(format!(
                    "routing attempt {attempt_serial} failed experimental validator: {error}"
                ));
                last_error = Some(error);
                routing_progress.maybe_report(progress, attempt_serial, total_attempts);
                continue;
            }

            let world = match placed_world_from_routing(topology, candidates, placed, &routed_nets)
            {
                Ok(world) => world,
                Err(error) => {
                    routing_progress.assembly_failures += 1;
                    let error = eyre::eyre!(error);
                    progress.detail(format!(
                        "placement attempt {attempt_serial} failed: {error}"
                    ));
                    last_error = Some(error);
                    routing_progress.maybe_report(progress, attempt_serial, total_attempts);
                    continue;
                }
            };
            if let Some(route) = first_invalid_active_route(&world.world, &routed_nets) {
                routing_progress.contract_failures += 1;
                let error = eyre::eyre!(
                    "assembled route from {:?} to {:?} does not satisfy its powered-position contract",
                    route.source,
                    route.sink
                );
                progress.detail(format!(
                    "placement attempt {attempt_serial} failed: {error}"
                ));
                if routing_index + 1 < routing_configs.len() {
                    next_scores.push(RankedRoutingDecision {
                        routed_count: routed_nets.len(),
                        sequence: next_scores.len(),
                        placement_index,
                        order_index,
                        prefix: reroute_source_group(
                            &routed_nets,
                            route.net_id,
                            route.source_label.as_deref(),
                        ),
                        semantic_feedback_nets: decision.semantic_feedback_nets.clone(),
                    });
                }
                last_error = Some(error);
                routing_progress.maybe_report(progress, attempt_serial, total_attempts);
                continue;
            }
            if let Some(verifier) = config.verifier {
                if let Err(error) = verifier(&world) {
                    routing_progress.verifier_failures += 1;
                    save_failed_verifier_world(
                        &topology.definitions[topology.top.0].display_name,
                        attempt_serial,
                        &world,
                        &routed_nets,
                    );
                    progress.detail(format!(
                        "placement attempt {attempt_serial} failed verifier: {error}"
                    ));
                    if routing_index + 1 < routing_configs.len() {
                        let (prefix, semantic_feedback_nets) = reroute_untried_source_groups(
                            &routed_nets,
                            &decision.semantic_feedback_nets,
                            2,
                        );
                        next_scores.push(RankedRoutingDecision {
                            routed_count: routed_nets.len(),
                            sequence: next_scores.len(),
                            placement_index,
                            order_index,
                            prefix,
                            semantic_feedback_nets,
                        });
                    }
                    last_error = Some(error);
                    routing_progress.maybe_report(progress, attempt_serial, total_attempts);
                    continue;
                }
            }
            progress.summary(format!(
                "routing search completed: selected_attempt={attempt_serial} routes={} attempts={} routing_failures={} assembly_failures={} contract_failures={} verifier_failures={} elapsed={:.2?}",
                routed_nets.len(),
                attempt_serial,
                routing_progress.routing_failures,
                routing_progress.assembly_failures,
                routing_progress.contract_failures,
                routing_progress.verifier_failures,
                routing_progress.started.elapsed()
            ));
            record_snapshot(SnapshotEvent::RoutingSummary {
                selected_attempt: attempt_serial,
                routes: routed_nets.len(),
                routing_failures: routing_progress.routing_failures,
                assembly_failures: routing_progress.assembly_failures,
                contract_failures: routing_progress.contract_failures,
                verifier_failures: routing_progress.verifier_failures,
                elapsed_ms: duration_ms(routing_progress.started.elapsed()),
            });
            return Ok((placed.clone(), routed_nets));
        }
        previous_scores = next_scores;
    }

    progress.warning(format!(
        "routing search exhausted: attempts={} routing_failures={} assembly_failures={} contract_failures={} verifier_failures={} elapsed={:.2?}",
        attempt_serial,
        routing_progress.routing_failures,
        routing_progress.assembly_failures,
        routing_progress.contract_failures,
        routing_progress.verifier_failures,
        routing_progress.started.elapsed()
    ));
    Err(last_error.unwrap_or_else(|| eyre::eyre!("no global placement attempts generated")))
}

fn placed_world_from_routing(
    topology: &ResolvedPnrTopology,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
    routed_nets: &[RoutedNet],
) -> eyre::Result<PlacedWorld> {
    Ok(PlacedWorld {
        world: assemble_world(candidates, placed, routed_nets)?,
        inputs: collect_topology_input_endpoints(topology, routed_nets),
        outputs: collect_topology_output_endpoints(topology, candidates, placed),
    })
}

fn save_failed_verifier_world(
    module_name: &str,
    attempt: usize,
    world: &PlacedWorld,
    routed_nets: &[RoutedNet],
) {
    if std::env::var_os("SAVE_FAILED_GLOBAL_PNR").is_none() {
        return;
    }

    let path = format!("test/{module_name}-failed-attempt-{attempt}.nbt");
    world.world.to_nbt().save(path);
    let metadata_path = format!("test/{module_name}-failed-attempt-{attempt}.outputs.json");
    let _ = world.metadata().save(metadata_path);
    let routes_path = format!("test/{module_name}-failed-attempt-{attempt}.routes.txt");
    let routes = routed_nets
        .iter()
        .map(|route| {
            format!(
                "source_label={:?} sink_label={:?} source={:?} sink={:?} required={:?} path={:?}\n",
                route.source_label,
                route.sink_label,
                route.source,
                route.sink,
                route.required_powered_positions,
                route.path
            )
        })
        .collect::<String>();
    let _ = std::fs::write(routes_path, routes);
}

fn save_failed_route_base_world(
    topology: &ResolvedPnrTopology,
    attempt: usize,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) {
    if std::env::var_os("SAVE_FAILED_GLOBAL_PNR").is_none() {
        return;
    }

    let Ok(world) = placed_world_from_routing(topology, candidates, placed, &[]) else {
        return;
    };
    let module_name = &topology.definitions[topology.top.0].display_name;
    let path = format!("test/{module_name}-route-failed-attempt-{attempt}.nbt");
    world.world.to_nbt().save(path);
    let metadata_path = format!("test/{module_name}-route-failed-attempt-{attempt}.outputs.json");
    let _ = world.metadata().save(metadata_path);
}

fn run_prepared_leaf(
    topology: &ResolvedPnrTopology,
    prepared_candidates: &[LayoutCandidate],
    config: &GlobalPnrConfig,
    progress: &GlobalPnrProgress,
    started: Instant,
) -> eyre::Result<GlobalPnrResult> {
    let candidate = prepared_candidates
        .iter()
        .cloned()
        .next()
        .context("graph-backed module produced no layout candidates")?;
    progress.detail(format!(
        "selected candidate for `{}`",
        candidate.module_name
    ));

    progress.stage(2, 4, "place leaf candidate");
    let placed = place_candidates_on_shelves(&[candidate.clone()], &config.placement);
    let inputs = candidate
        .ports
        .iter()
        .filter(|port| port.direction == PhysicalPortDirection::Input)
        .map(|port| OutputEndpoint::new(port.name.clone(), port.position))
        .collect();

    progress.stage(3, 4, "assemble leaf world");
    let candidates = vec![candidate];
    let world = assemble_world(&candidates, &placed, &[])?;
    let placement_bbox_world = placement_bbox_wireframe_world(&placed);
    let placement_cost = placement_cost_breakdown_resolved(
        topology,
        &candidates,
        &placed,
        config.placement.congestion,
    )?;
    let weighted_placement_cost = placement_cost.weighted_total(config.placement.cost_weights);
    let constraint_report = config
        .physical_intent
        .as_ref()
        .map_or_else(Vec::new, |intent| {
            intent.evaluate(topology, &candidates, &placed, &[])
        });

    progress.stage(4, 4, "complete");
    progress.summary(format!(
        "global PnR completed: outputs=0 routes=0 global_elapsed={:.2?}",
        started.elapsed()
    ));
    Ok(GlobalPnrResult {
        topology: topology.clone(),
        placed_world: PlacedWorld {
            world,
            inputs,
            outputs: Vec::new(),
        },
        placement_bbox_world,
        selected_candidates: candidates,
        placed_modules: placed,
        routed_nets: Vec::new(),
        placement_cost,
        weighted_placement_cost,
        config_snapshot: global_pnr_config_snapshot(config),
        physical_intent: config.physical_intent.clone(),
        constraint_report,
    })
}

// 하위 모듈마다 local placer를 실행해서 global PnR이 배치할 layout 후보를 하나씩 뽑는다.
fn prepare_routable_child_candidate_sets(
    design: &RoutableDesign,
    module: &RoutableModule,
    config: &PnrPrepareConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<(
    Vec<PreparedCandidateSet>,
    Vec<PreparedInstanceCandidateBinding>,
    PnrPreparationSummary,
)> {
    let RoutableModuleBody::Composite { instances, .. } = &module.body else {
        eyre::bail!("module `{}` is not composite", module.name);
    };
    let started = Instant::now();
    let mut bindings = Vec::new();
    let mut cache = ChildCandidateCache::default();
    let mut reused = 0usize;
    let mut candidate_references = 0usize;
    for (index, instance) in instances.iter().enumerate() {
        progress.item(
            index + 1,
            instances.len(),
            format!("generate `{}` candidate", instance.name),
        );
        let child = design.module(&instance.module).with_context(|| {
            format!(
                "instance `{}` references missing Routable module `{}`",
                instance.name, instance.module
            )
        })?;
        if !matches!(child.body, RoutableModuleBody::Leaf { .. }) {
            eyre::bail!("Routable child module `{}` is not a leaf", child.name);
        }
        let base_config = config.candidate.effective_for_definition(&child.name);
        let child_config = candidate_config_for_routable_child(child, &base_config);
        let persistent_key = routable_candidate_shape_fingerprint(child, &child_config);
        let persistent_hit = std::cell::Cell::new(false);
        let candidate_started = Instant::now();
        let (candidate_set_index, cache_hit) =
            cache.get_or_generate_index(&persistent_key, || {
                if let Some(root) = config.candidate_cache_dir.as_deref() {
                    match candidate_cache::load(root, &persistent_key, &child.name) {
                        Ok(Some(candidates)) => {
                            persistent_hit.set(true);
                            return Ok(candidates);
                        }
                        Ok(None) => {}
                        Err(error) => progress.detail(format!(
                            "ignored invalid candidate cache entry `{persistent_key}`: {error}"
                        )),
                    }
                }
                let candidates = generate_routable_module_candidates_with_progress_label(
                    child,
                    &child_config,
                    config.show_progress.then_some(instance.name.as_str()),
                )?;
                if let Some(root) = config.candidate_cache_dir.as_deref()
                    && let Err(error) = candidate_cache::store(root, &persistent_key, &candidates)
                {
                    progress.detail(format!(
                        "could not store candidate cache entry `{persistent_key}`: {error}"
                    ));
                }
                Ok(candidates)
            })?;
        let child_candidates = &cache.entries[candidate_set_index].candidates;
        if cache_hit || persistent_hit.get() {
            reused += 1;
            let source = if cache_hit { "memory" } else { "persistent" };
            progress.detail(format!("`{}` reused {source} candidates", instance.name));
        }
        progress.detail(format!(
            "`{}` produced {} candidate(s) in {:.2?}",
            instance.name,
            child_candidates.len(),
            candidate_started.elapsed()
        ));
        if child_candidates.is_empty() {
            eyre::bail!("module instance `{}` produced no candidates", instance.name);
        }
        candidate_references += child_candidates.len();
        let preferred_index = if routable_input_port_count(child) > 1 {
            child_candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| {
                    (candidate.cost.bbox_volume, candidate.cost.block_count)
                })
                .map(|(index, _)| index)
                .unwrap_or(0)
        } else {
            0
        };
        bindings.push(PreparedInstanceCandidateBinding {
            instance_name: instance.name.clone(),
            candidate_set_index,
            preferred_index,
        });
    }
    let unique_candidates = cache
        .entries
        .iter()
        .map(|entry| entry.candidates.len())
        .sum();
    let unique_candidate_sets = cache.entries.len();
    progress.summary(format!(
        "child candidates completed: instances={} unique_sets={} reused={} stored_candidates={} candidate_references={} elapsed={:.2?}",
        instances.len(), unique_candidate_sets, reused, unique_candidates, candidate_references, started.elapsed()
    ));
    record_snapshot(SnapshotEvent::CandidateSummary {
        instances: instances.len(),
        unique: unique_candidate_sets,
        reused,
        candidates: unique_candidates,
        elapsed_ms: duration_ms(started.elapsed()),
    });
    let candidate_sets = cache
        .entries
        .into_iter()
        .map(|entry| PreparedCandidateSet {
            candidates: entry.candidates,
        })
        .collect();
    Ok((
        candidate_sets,
        bindings,
        PnrPreparationSummary {
            module: module.name.clone(),
            instances: instances.len(),
            unique_candidate_sets,
            reused_candidate_sets: reused,
            candidates: unique_candidates,
            candidate_references,
            elapsed_ms: duration_ms(started.elapsed()),
        },
    ))
}

fn prepare_child_candidate_sets(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &PnrPrepareConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<(
    Vec<PreparedCandidateSet>,
    Vec<PreparedInstanceCandidateBinding>,
    PnrPreparationSummary,
)> {
    let started = Instant::now();
    let mut bindings = Vec::new();
    let mut cache = ChildCandidateCache::default();
    let mut reused = 0usize;
    let mut candidate_references = 0usize;
    for (index, instance) in module.instances.iter().enumerate() {
        progress.item(
            index + 1,
            module.instances.len(),
            format!("generate `{instance}` candidate"),
        );
        let child = &context[instance.as_str()];
        let base_config = config.candidate.effective_for_definition(&child.name);
        let child_config = candidate_config_for_child(child, &base_config);
        let persistent_key = candidate_shape_fingerprint(child, &child_config);
        let persistent_hit = std::cell::Cell::new(false);
        let candidate_started = Instant::now();
        let (candidate_set_index, cache_hit) =
            cache.get_or_generate_index(&persistent_key, || {
                if let Some(root) = config.candidate_cache_dir.as_deref() {
                    match candidate_cache::load(root, &persistent_key, &child.name) {
                        Ok(Some(candidates)) => {
                            persistent_hit.set(true);
                            return Ok(candidates);
                        }
                        Ok(None) => {}
                        Err(error) => progress.detail(format!(
                            "ignored invalid candidate cache entry `{persistent_key}`: {error}"
                        )),
                    }
                }
                let candidates = generate_graph_module_candidates_with_progress_label(
                    child,
                    &child_config,
                    config.show_progress.then_some(instance.as_str()),
                )?;
                if let Some(root) = config.candidate_cache_dir.as_deref()
                    && let Err(error) = candidate_cache::store(root, &persistent_key, &candidates)
                {
                    progress.detail(format!(
                        "could not store candidate cache entry `{persistent_key}`: {error}"
                    ));
                }
                Ok(candidates)
            })?;
        let child_candidates = &cache.entries[candidate_set_index].candidates;
        if cache_hit || persistent_hit.get() {
            reused += 1;
            let source = if cache_hit { "memory" } else { "persistent" };
            progress.detail(format!("`{instance}` reused {source} candidates"));
        }
        progress.detail(format!(
            "`{instance}` produced {} candidate(s) in {:.2?}",
            child_candidates.len(),
            candidate_started.elapsed()
        ));
        if child_candidates.is_empty() {
            return Err(eyre::eyre!(
                "module instance `{instance}` produced no candidates"
            ));
        }
        candidate_references += child_candidates.len();
        let preferred_index = if graph_module_input_port_count(child) > 1 {
            child_candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| {
                    (candidate.cost.bbox_volume, candidate.cost.block_count)
                })
                .map(|(index, _)| index)
                .unwrap_or(0)
        } else {
            0
        };
        bindings.push(PreparedInstanceCandidateBinding {
            instance_name: instance.clone(),
            candidate_set_index,
            preferred_index,
        });
    }
    let unique_candidates = cache
        .entries
        .iter()
        .map(|entry| entry.candidates.len())
        .sum();
    let unique_candidate_sets = cache.entries.len();
    progress.summary(format!(
        "child candidates completed: instances={} unique_sets={} reused={} stored_candidates={} candidate_references={} elapsed={:.2?}",
        module.instances.len(),
        unique_candidate_sets,
        reused,
        unique_candidates,
        candidate_references,
        started.elapsed()
    ));
    record_snapshot(SnapshotEvent::CandidateSummary {
        instances: module.instances.len(),
        unique: unique_candidate_sets,
        reused,
        candidates: unique_candidates,
        elapsed_ms: duration_ms(started.elapsed()),
    });
    let candidate_sets = cache
        .entries
        .into_iter()
        .map(|entry| PreparedCandidateSet {
            candidates: entry.candidates,
        })
        .collect();
    Ok((
        candidate_sets,
        bindings,
        PnrPreparationSummary {
            module: module.name.clone(),
            instances: module.instances.len(),
            unique_candidate_sets,
            reused_candidate_sets: reused,
            candidates: unique_candidates,
            candidate_references,
            elapsed_ms: duration_ms(started.elapsed()),
        },
    ))
}

#[derive(Default)]
struct ChildCandidateCache {
    entries: Vec<ChildCandidateCacheEntry>,
}

struct ChildCandidateCacheEntry {
    key: String,
    candidates: Vec<LayoutCandidate>,
}

impl ChildCandidateCache {
    fn get_or_generate_index(
        &mut self,
        key: &str,
        generate: impl FnOnce() -> eyre::Result<Vec<LayoutCandidate>>,
    ) -> eyre::Result<(usize, bool)> {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            return Ok((index, true));
        }

        let candidates = generate()?;
        let index = self.entries.len();
        self.entries.push(ChildCandidateCacheEntry {
            key: key.to_owned(),
            candidates,
        });
        Ok((index, false))
    }

    #[cfg(test)]
    fn get_or_generate(
        &mut self,
        module: &GraphModule,
        config: &UnitCandidateConfig,
        generate: impl FnOnce() -> eyre::Result<Vec<LayoutCandidate>>,
    ) -> eyre::Result<(Vec<LayoutCandidate>, bool)> {
        let key = candidate_shape_fingerprint(module, config);
        let (index, reused) = self.get_or_generate_index(&key, generate)?;
        Ok((
            relabel_candidates(&self.entries[index].candidates, &module.name),
            reused,
        ))
    }
}

fn relabel_candidates(candidates: &[LayoutCandidate], module_name: &str) -> Vec<LayoutCandidate> {
    candidates
        .iter()
        .cloned()
        .map(|mut candidate| {
            candidate.module_name = module_name.to_owned();
            candidate
        })
        .collect()
}

fn candidate_shape_fingerprint(module: &GraphModule, config: &UnitCandidateConfig) -> String {
    let nodes = module.graph.as_ref().map(|graph| {
        let mut ordered = graph.nodes.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|node| node.id);
        let canonical_ids = ordered
            .iter()
            .enumerate()
            .map(|(canonical, node)| (node.id, canonical))
            .collect::<std::collections::HashMap<_, _>>();
        ordered
            .into_iter()
            .map(|node| {
                (
                    stable_graph_node_kind_signature(&node.kind),
                    node.inputs
                        .iter()
                        .map(|input| canonical_ids[input])
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    });
    debug_parity_hash(&("local-candidate-cache-v1", nodes, &module.ports, config))
}

fn stable_graph_node_kind_signature(kind: &GraphNodeKind) -> String {
    match kind {
        GraphNodeKind::Sequential(sequential) => format!(
            "Sequential({:?},{:?},{:?})",
            sequential.sequential_type, sequential.input_ports, sequential.output_ports
        ),
        _ => kind.name(),
    }
}

fn routable_candidate_shape_fingerprint(
    module: &RoutableModule,
    config: &UnitCandidateConfig,
) -> String {
    debug_parity_hash(&(
        "routable-local-candidate-cache-v1",
        &module.ports,
        &module.body,
        config,
    ))
}

fn search_layout_combinations(
    topology: &ResolvedPnrTopology,
    pools: &[ChildCandidatePool],
    config: &GlobalPnrConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<(Vec<LayoutCandidate>, Vec<PlacedModule>, Vec<RoutedNet>)> {
    let combinations =
        layout_combinations(pools, config.search.budget.max_layout_combinations.max(1));
    let mut last_error = None;
    for (combination_index, selection) in combinations.iter().enumerate() {
        progress.detail(format!(
            "layout combination {}/{}: {:?}",
            combination_index + 1,
            combinations.len(),
            selection
        ));
        let candidates = select_layout_combination(pools, selection)
            .context("invalid child layout combination")?;
        let placement_attempts = placement_candidates_resolved(
            topology,
            config.physical_intent.as_ref(),
            &config.heuristic_hooks,
            &candidates,
            &config.placement,
            &config.search.policies.placement_heuristics,
        )?;
        progress.detail(format!(
            "layout combination {} generated {} placement attempt(s)",
            combination_index + 1,
            placement_attempts.len()
        ));
        match route_first_successful_placement(
            topology,
            &candidates,
            placement_attempts,
            config,
            progress,
        ) {
            Ok((placed, routed_nets)) => {
                let cost = placement_cost_breakdown_resolved(
                    topology,
                    &candidates,
                    &placed,
                    config.placement.congestion,
                )?;
                let weighted_total = cost.weighted_total(config.placement.cost_weights);
                progress.summary(format!(
                    "selected placement cost: volume={} xy={} height={} wire={} vertical={} congestion={} weighted_total={}",
                    cost.placement_volume,
                    cost.xy_footprint,
                    cost.height_span,
                    cost.estimated_wire_length,
                    cost.vertical_distance,
                    cost.routing_congestion,
                    weighted_total,
                ));
                record_snapshot(SnapshotEvent::PlacementCost {
                    volume: cost.placement_volume,
                    xy_footprint: cost.xy_footprint,
                    height: cost.height_span,
                    wire_length: cost.estimated_wire_length,
                    vertical_distance: cost.vertical_distance,
                    congestion: cost.routing_congestion,
                    weighted_total,
                });
                return Ok((candidates, placed, routed_nets));
            }
            Err(error) => {
                progress.detail(format!(
                    "layout combination {} failed: {error}",
                    combination_index + 1
                ));
                last_error = Some(error);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| eyre::eyre!("no child layout combinations generated")))
}

fn candidate_config_for_child(
    child: &GraphModule,
    base_config: &UnitCandidateConfig,
) -> UnitCandidateConfig {
    let mut config = base_config.clone();
    if graph_module_is_combinational(child) {
        if graph_module_input_port_count(child) > 1 {
            config.local_config = multi_input_combinational_local_config(config.local_config);
        }
        if let Some(limit) = config.combinational_sampling_limit {
            config.local_config.step_sampling_policy = SamplingPolicy::Random(limit);
            config.local_config.not_route_step_sampling_policy = SamplingPolicy::Random(limit);
            config.local_config.route_step_sampling_policy = SamplingPolicy::Random(limit);
        }
    }
    config
}

fn candidate_config_for_routable_child(
    child: &RoutableModule,
    base_config: &UnitCandidateConfig,
) -> UnitCandidateConfig {
    let mut config = base_config.clone();
    if routable_module_is_combinational(child) {
        if routable_input_port_count(child) > 1 {
            config.local_config = multi_input_combinational_local_config(config.local_config);
        }
        if let Some(limit) = config.combinational_sampling_limit {
            config.local_config.step_sampling_policy = SamplingPolicy::Random(limit);
            config.local_config.not_route_step_sampling_policy = SamplingPolicy::Random(limit);
            config.local_config.route_step_sampling_policy = SamplingPolicy::Random(limit);
        }
    }
    config
}

fn routable_module_is_combinational(module: &RoutableModule) -> bool {
    match &module.body {
        RoutableModuleBody::Leaf { nodes } => nodes
            .iter()
            .all(|node| !matches!(node.kind, crate::ir::RoutableNodeKind::Sequential { .. })),
        RoutableModuleBody::Composite { .. } => false,
    }
}

fn routable_input_port_count(module: &RoutableModule) -> usize {
    module
        .ports
        .iter()
        .filter(|port| port.direction == RoutablePortDirection::Input)
        .count()
}

fn graph_module_is_combinational(module: &GraphModule) -> bool {
    module.graph.as_ref().is_some_and(|graph| {
        graph
            .nodes
            .iter()
            .all(|node| !matches!(node.kind, GraphNodeKind::Sequential(_)))
    })
}

fn multi_input_combinational_local_config(mut config: LocalPlacerConfig) -> LocalPlacerConfig {
    config.leak_sampling = false;
    config.not_route_strategy = NotRouteStrategy::DirectAndRedstone;
    config.max_not_route_step = config.max_not_route_step.max(4);
    config.not_route_step_sampling_policy = SamplingPolicy::Random(512);
    config.max_route_step = config.max_route_step.max(4);
    config.route_step_sampling_policy = SamplingPolicy::Random(512);
    config
}

fn graph_module_input_port_count(module: &GraphModule) -> usize {
    module
        .ports
        .iter()
        .filter(|port| port.port_type.is_input())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::LogicGraph;
    use crate::graph::module::{
        GraphModule, GraphModuleContext, GraphModuleDesign, GraphModulePort, GraphModulePortTarget,
        GraphModulePortType, GraphModuleVariable,
    };
    use crate::graph::GraphNodeKind;
    use crate::nbt::{NBTRoot, ToNBT};
    use crate::snapshot::{compile_with_snapshot, SnapshotOptions};
    use crate::transform::place_and_route::global_pnr::candidate::d_latch_child_candidate_config;
    use crate::transform::place_and_route::global_pnr::policy::{
        Free3DPlacementConfig, PlacementHeuristic,
    };
    use crate::transform::place_and_route::local_placer::{
        InputPlacementStrategy, LocalPlacerConfig, NotRouteStrategy, PlacementSamplingPolicy,
        TorchPlacementStrategy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;
    use crate::transform::place_and_route::utils::world_to_logic_with_outputs;
    use crate::verilog::design::lower_design_modules;
    use crate::verilog::parser::parse_modules;
    use crate::verilog::synth::d_latch_graph_module;
    use crate::world::block::BlockKind;
    use crate::world::position::{DimSize, Position};
    use crate::world::simulator::{Simulator, MANUAL_INPUT_IDLE_CYCLES};
    use crate::world::{World, World3D};

    fn sequential_local_config() -> LocalPlacerConfig {
        LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            input_candidate_limit: None,
            step_sampling_policy: SamplingPolicy::Random(256),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: true,
            materialize_outputs: false,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectAndRedstone,
            max_not_route_step: 4,
            not_route_step_sampling_policy: SamplingPolicy::Random(256),
            max_route_step: 4,
            route_step_sampling_policy: SamplingPolicy::Random(256),
        }
    }

    #[test]
    fn child_candidate_cache_reuses_identical_module_structure() -> eyre::Result<()> {
        let first = d_latch_graph_module("q_0_master", "d", "en", "q");
        let second = d_latch_graph_module("q_1_slave", "d", "en", "q");
        let config = UnitCandidateConfig {
            dim: DimSize(3, 3, 3),
            max_candidates: 1,
            ..Default::default()
        };
        let mut cache = ChildCandidateCache::default();
        let calls = std::cell::Cell::new(0);
        let generate = |module: &GraphModule| {
            calls.set(calls.get() + 1);
            let mut world = World3D::new(DimSize(3, 3, 3));
            world[Position(1, 1, 1)] = crate::world::block::Block {
                kind: BlockKind::Cobble {
                    on_count: 0,
                    on_base_count: 0,
                },
                direction: crate::world::block::Direction::None,
            };
            Ok(vec![LayoutCandidate::from_world(
                module.name.clone(),
                world,
                Vec::new(),
            )?])
        };

        let (first_candidates, first_hit) =
            cache.get_or_generate(&first, &config, || generate(&first))?;
        let (second_candidates, second_hit) =
            cache.get_or_generate(&second, &config, || generate(&second))?;

        assert_eq!(calls.get(), 1);
        assert!(!first_hit);
        assert!(second_hit);
        assert_eq!(first_candidates[0].module_name, "q_0_master");
        assert_eq!(second_candidates[0].module_name, "q_1_slave");

        let different_config = UnitCandidateConfig {
            max_candidates: 2,
            ..config.clone()
        };
        let (_, config_hit) =
            cache.get_or_generate(&second, &different_config, || generate(&second))?;
        assert!(!config_hit);

        let mut different_port = second.clone();
        different_port.ports[0].name = "renamed".to_owned();
        let (_, port_hit) =
            cache.get_or_generate(&different_port, &config, || generate(&different_port))?;
        assert!(!port_hit);
        assert_eq!(calls.get(), 3);
        Ok(())
    }

    #[test]
    fn routable_leaf_candidate_generation_matches_graph_module_path() -> eyre::Result<()> {
        let mut module: GraphModule = LogicGraph::from_stmt("~a", "q")?.graph.into();
        module.name = "candidate_parity".to_owned();
        let design =
            GraphModuleDesign::with_top_module(GraphModuleContext::default(), module.clone());
        let routable = RoutableDesign::from_graph_module_design(&design)?;
        let routable_module = routable.module(&routable.top).unwrap();
        let config = UnitCandidateConfig {
            dim: DimSize(8, 8, 4),
            max_candidates: 2,
            ..Default::default()
        };

        let legacy = generate_graph_module_candidates_with_progress_label(&module, &config, None)?;
        let direct = generate_routable_module_candidates_with_progress_label(
            routable_module,
            &config,
            None,
        )?;

        assert!(!legacy.is_empty());
        assert_eq!(
            legacy.iter().map(candidate_parity_hash).collect::<Vec<_>>(),
            direct.iter().map(candidate_parity_hash).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn graph_backed_module_generates_world_with_global_pnr_api() -> eyre::Result<()> {
        let mut module: GraphModule = LogicGraph::from_stmt("~a", "not_a")?.graph.into();
        module.name = "not_gate".to_owned();
        let context = GraphModuleContext::default();
        let design = GraphModuleDesign::with_top_module(context, module);
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            }
            .into(),
            placement: GlobalPlacementConfig::default(),
            ..Default::default()
        };

        let world = place_and_route_design(&design, &config)?;

        assert!(!world.iter_block().is_empty());
        Ok(())
    }

    #[test]
    fn module_outputs_point_to_placed_child_ports_without_extra_routes() -> eyre::Result<()> {
        let mut context = GraphModuleContext::default();
        context.append(not_clk_module());
        let module = GraphModule {
            name: "top".to_owned(),
            instances: vec!["not_clk".to_owned()],
            ports: vec![GraphModulePort {
                name: "q".to_owned(),
                port_type: GraphModulePortType::OutputNet,
                target: GraphModulePortTarget::Module("not_clk".to_owned(), "clk_n".to_owned()),
            }],
            ..Default::default()
        };
        let design = GraphModuleDesign::with_top_module(context, module);
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            }
            .into(),
            placement: GlobalPlacementConfig::default(),
            ..Default::default()
        };

        let placed = place_and_route_design_with_outputs(&design, &config)?;

        assert_eq!(placed.outputs.len(), 1);
        assert_eq!(placed.outputs[0].name, "q");
        let output_position = placed.outputs[0].position();
        assert_ne!(placed.world[output_position].kind, BlockKind::Air);
        Ok(())
    }

    #[test]
    fn prepared_pnr_reuses_local_candidates_across_global_runs() -> eyre::Result<()> {
        let mut context = GraphModuleContext::default();
        context.append(not_clk_module());
        let module = GraphModule {
            name: "prepared_top".to_owned(),
            instances: vec!["not_clk".to_owned()],
            ports: vec![GraphModulePort {
                name: "q".to_owned(),
                port_type: GraphModulePortType::OutputNet,
                target: GraphModulePortTarget::Module("not_clk".to_owned(), "clk_n".to_owned()),
            }],
            ..Default::default()
        };
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            }
            .into(),
            placement: GlobalPlacementConfig::default(),
            show_progress: false,
            ..Default::default()
        };

        let prepared =
            prepare_module_for_global_pnr(&context, &module, &PnrPrepareConfig::from(&config))?;
        let signature = prepared.parity_signature();
        assert_eq!(prepared.summary().instances, 1);
        assert_eq!(prepared.summary().candidates, 1);

        let first = run_prepared_pnr_with_visualization(&prepared, &config)?;
        let mut replay_config = config.clone();
        replay_config.placement.spacing += 3;
        replay_config.search.budget.max_candidates_per_child = 1;
        let replay = run_prepared_pnr_with_visualization(&prepared, &replay_config)?;

        assert_eq!(prepared.parity_signature(), signature);
        assert_eq!(first.placed_world.outputs.len(), 1);
        assert_eq!(replay.placed_world.outputs.len(), 1);
        assert_eq!(
            first.placed_world.outputs[0].position(),
            replay.placed_world.outputs[0].position()
        );
        assert_eq!(
            candidate_parity_hash(&first.selected_candidates[0]),
            candidate_parity_hash(&replay.selected_candidates[0])
        );

        let mut incompatible = replay_config;
        incompatible.candidate.max_candidates += 1;
        let error = run_prepared_pnr(&prepared, &incompatible).unwrap_err();
        assert!(error.to_string().contains("regenerate local candidates"));
        Ok(())
    }

    #[test]
    fn prepared_pnr_stores_structurally_identical_candidates_once() -> eyre::Result<()> {
        let mut first: GraphModule = LogicGraph::from_stmt("~a", "y")?.graph.into();
        first.name = "first".to_owned();
        let mut second = first.clone();
        second.name = "second".to_owned();
        let mut context = GraphModuleContext::default();
        context.append(first);
        context.append(second);
        let module = GraphModule {
            name: "duplicate_top".to_owned(),
            instances: vec!["first".to_owned(), "second".to_owned()],
            ..Default::default()
        };
        let config = PnrPrepareConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            }
            .into(),
            show_progress: false,
            candidate_cache_dir: None,
        };

        let prepared = prepare_module_for_global_pnr(&context, &module, &config)?;

        assert_eq!(prepared.summary().instances, 2);
        assert_eq!(prepared.summary().unique_candidate_sets, 1);
        assert_eq!(prepared.summary().reused_candidate_sets, 1);
        assert_eq!(prepared.summary().candidates, 1);
        assert_eq!(prepared.summary().candidate_references, 2);

        let routable: RoutableDesign = r#"
            rcir 1;
            stage routable;
            target "redstone-v1";
            top "direct_duplicate_top";
            module "direct_duplicate_top" {
              port input "a";
              port output "y";
              instance "first" : "inv";
              instance "second" : "inv";
              net "a" class io driver self."a" sinks ["first"."a"];
              net "mid" class data driver "first"."y" sinks ["second"."a"];
              net "y" class io driver "second"."y" sinks [self."y"];
            }
            leaf "inv" {
              port input "a";
              port output "y";
              node 0 input "a" inputs [];
              node 1 logic not inputs [0];
              node 2 output "y" inputs [1];
            }
        "#
        .parse()?;
        let direct = prepare_routable_design_for_global_pnr(&routable, &config)?;
        assert_eq!(direct.summary().instances, 2);
        assert_eq!(direct.summary().unique_candidate_sets, 1);
        assert_eq!(direct.summary().reused_candidate_sets, 1);
        assert_eq!(direct.summary().candidates, 1);
        assert_eq!(direct.summary().candidate_references, 2);
        Ok(())
    }

    #[test]
    fn prepared_pnr_round_trips_through_snapshot_directory_and_archive() -> eyre::Result<()> {
        let output = std::path::PathBuf::from(format!(
            "target/prepared-pnr-round-trip-{}.snapshot",
            std::process::id()
        ));
        let archive = output.with_extension("rsnap");
        let _ = std::fs::remove_dir_all(&output);
        let _ = std::fs::remove_file(&archive);

        let mut module: GraphModule = LogicGraph::from_stmt("~a", "q")?.graph.into();
        module.name = "prepared_snapshot_top".to_owned();
        let design = GraphModuleDesign::with_top_module(GraphModuleContext::default(), module);
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            }
            .into(),
            show_progress: false,
            physical_intent: Some(ResolvedPhysicalIntent {
                format: physical_intent::PHYSICAL_INTENT_FORMAT.to_owned(),
                design: "prepared_snapshot_top".to_owned(),
                regions: Default::default(),
                constraints: Vec::new(),
            }),
            ..Default::default()
        };
        let prepare_config = PnrPrepareConfig::from(&config);

        let original = compile_with_snapshot(
            SnapshotOptions::new(&output, "prepared_snapshot_top"),
            || place_and_route_design_with_visualization(&design, &config),
        )?;
        let from_directory = load_prepared_pnr_snapshot(&output, &prepare_config)?;
        assert!(from_directory.snapshot_intent().is_some());
        let directory_result = run_prepared_pnr_with_visualization(&from_directory, &config)?;
        let from_archive = load_prepared_pnr_snapshot(&archive, &prepare_config)?;
        assert!(from_archive.snapshot_intent().is_some());
        let archive_result = run_prepared_pnr_with_visualization(&from_archive, &config)?;

        let expected_candidate = candidate_parity_hash(&original.selected_candidates[0]);
        assert_eq!(
            candidate_parity_hash(&directory_result.selected_candidates[0]),
            expected_candidate
        );
        assert_eq!(
            candidate_parity_hash(&archive_result.selected_candidates[0]),
            expected_candidate
        );
        assert_eq!(directory_result.placed_world.outputs.len(), 0);
        assert_eq!(archive_result.placed_world.outputs.len(), 0);

        let _ = std::fs::remove_dir_all(output);
        let _ = std::fs::remove_file(archive);
        Ok(())
    }

    #[test]
    fn layered_global_pnr_routes_connected_children_across_z_layers() -> eyre::Result<()> {
        let mut first: GraphModule = LogicGraph::from_stmt("~a", "x")?.graph.into();
        first.name = "first".to_owned();
        let mut second: GraphModule = LogicGraph::from_stmt("~x", "y")?.graph.into();
        second.name = "second".to_owned();
        let mut context = GraphModuleContext::default();
        context.append(first);
        context.append(second);
        let top = GraphModule {
            name: "layered_top".to_owned(),
            instances: vec!["first".to_owned(), "second".to_owned()],
            vars: vec![GraphModuleVariable {
                var_type: GraphModulePortType::InputNet,
                source: ("first".to_owned(), "x".to_owned()),
                target: ("second".to_owned(), "x".to_owned()),
            }],
            ports: vec![GraphModulePort {
                name: "y".to_owned(),
                port_type: GraphModulePortType::OutputNet,
                target: GraphModulePortTarget::Module("second".to_owned(), "y".to_owned()),
            }],
            ..Default::default()
        };
        let mut config = GlobalPnrPreset::Fast.config();
        config.candidate.max_candidates = 1;
        config.candidate.dim = DimSize(8, 8, 4);
        config.search.policies.placement_heuristics = vec![
            crate::transform::place_and_route::global_pnr::policy::PlacementHeuristic::Layered3D(
                crate::transform::place_and_route::global_pnr::policy::LayeredPlacementConfig {
                    layers: 2,
                    layer_spacing: 4,
                    assignment: crate::transform::place_and_route::global_pnr::policy::LayerAssignmentStrategy::Alternating,
                },
            ),
        ];

        let result = place_and_route_module_with_visualization(&context, &top, &config)?;
        let bbox_positions = result
            .placement_bbox_world
            .iter_block()
            .into_iter()
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        let min_z = bbox_positions
            .iter()
            .map(|position| position.2)
            .min()
            .unwrap();
        let max_z = bbox_positions
            .iter()
            .map(|position| position.2)
            .max()
            .unwrap();

        assert!(max_z > min_z + 4);
        assert_eq!(result.placed_world.outputs.len(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "search-heavy sequential global pnr smoke test"]
    fn d_flip_flop_module_generates_world_from_child_layout_candidates() -> eyre::Result<()> {
        init_tracing_from_env();
        let design = lower_design_modules(&parse_modules(
            r#"
            module not_clk(clk, clk_n);
              input clk;
              output clk_n;
              assign clk_n = ~clk;
            endmodule

            module d_latch(d, en, q);
              input d, en;
              output reg q;
              always @(*) begin
                if (en) begin
                  q <= d;
                end
              end
            endmodule

            module d_flip_flop(d, clk, q);
              input d, clk;
              output q;
              wire clk_n, master_q;
              not_clk inv(.clk(clk), .clk_n(clk_n));
              d_latch master(.d(d), .en(clk_n), .q(master_q));
              d_latch slave(.d(master_q), .en(clk), .q(q));
            endmodule
            "#,
        )?)?;
        let config = GlobalPnrConfig {
            candidate: d_latch_child_candidate_config(sequential_local_config()).into(),
            placement: GlobalPlacementConfig {
                spacing: 4,
                shelf_width: 64,
                max_attempts: 64,
                ..Default::default()
            },
            verifier: Some(assert_positive_edge_dff_behavior),
            ..Default::default()
        };

        let result = place_and_route_design_with_visualization(&design, &config)?;
        let placed = result.placed_world;

        assert!(!placed.world.iter_block().is_empty());
        assert_eq!(placed.outputs.len(), 1);
        assert_eq!(placed.outputs[0].name, "q");
        let nbt: NBTRoot = placed.world.to_nbt();
        nbt.save("test/d-flip-flop-global-smoke.nbt");
        result
            .placement_bbox_world
            .to_nbt()
            .save("test/d-flip-flop-global-placement-bbox.nbt");
        placed
            .metadata()
            .save("test/d-flip-flop-global-smoke.outputs.json")?;
        let logic = world_to_logic_with_outputs(&nbt.to_world(), &placed.metadata())?;
        assert!(logic
            .nodes
            .iter()
            .any(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == "q")));
        assert_positive_edge_dff_behavior(&placed)?;
        Ok(())
    }

    #[test]
    #[ignore = "search-heavy sequential global pnr smoke test"]
    fn counter_module_generates_world_from_child_layout_candidates() -> eyre::Result<()> {
        init_tracing_from_env();
        let source = r#"
            module counter(clk, q);
              input clk;
              output reg [1:0] q;
              always @(posedge clk) begin
                q <= q + 1;
              end
            endmodule
            "#;
        let logical_input = LogicalDesign::from_verilog_modules(&parse_modules(source)?)?;
        let logical_input: LogicalDesign = logical_input.to_string().parse()?;
        let sampling_limit = std::env::var("COUNTER_SAMPLING_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(32);
        let layout_limit = std::env::var("COUNTER_LAYOUT_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4);
        let mut counter_local_config = sequential_local_config();
        counter_local_config.random_seed = std::env::var("COUNTER_LOCAL_SEED")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6);
        let free_3d_heuristics = [2, 4, 6]
            .into_iter()
            .flat_map(|clearance| {
                (0..8).map(move |seed| {
                    PlacementHeuristic::Free3D(Free3DPlacementConfig {
                        seed,
                        clearance,
                        ..Free3DPlacementConfig::default()
                    })
                })
            })
            .collect();
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                max_candidates: 2,
                combinational_sampling_limit: Some(sampling_limit),
                ..d_latch_child_candidate_config(counter_local_config)
            }
            .into(),
            placement: GlobalPlacementConfig {
                spacing: 4,
                shelf_width: 64,
                max_attempts: 64,
                ..Default::default()
            },
            routing_probe: Some(GlobalRoutingConfig {
                strategy: crate::transform::place_and_route::global_pnr::router::GlobalRoutingStrategy::DirectGreedy {
                    max_steps: 128,
                },
                validation: crate::transform::place_and_route::global_pnr::router::RouteValidationMode::Deferred,
            }),
            routing: GlobalRoutingConfig {
                strategy: crate::transform::place_and_route::global_pnr::router::GlobalRoutingStrategy::GreedyBeam {
                    beam_width: 128,
                    max_expansions: 4_096,
                    variant_seed: 0,
                },
                validation: crate::transform::place_and_route::global_pnr::router::RouteValidationMode::Deferred,
            },
            routing_refinement: Some(GlobalRoutingConfig {
                strategy: crate::transform::place_and_route::global_pnr::router::GlobalRoutingStrategy::GreedyBeam {
                    beam_width: 128,
                    max_expansions: 4_096,
                    variant_seed: 0,
                },
                validation: crate::transform::place_and_route::global_pnr::router::RouteValidationMode::Deferred,
            }),
            search: GlobalSearchConfig {
                budget: GlobalSearchBudget {
                    max_candidates_per_child: 2,
                    max_layout_combinations: layout_limit,
                    max_detailed_routing_attempts: 2,
                    max_refined_routing_attempts: 2,
                    max_refinement_rounds: 8,
                },
                policies: GlobalPnrPolicies {
                    placement_heuristics: free_3d_heuristics,
                    net_order_strategies: vec![
                        NetOrderStrategy::Criticality,
                        NetOrderStrategy::HighestFanoutFirst,
                    ],
                },
            },
            verifier: Some(assert_two_bit_counter_behavior),
            ..Default::default()
        };
        let result = compile_with_snapshot(
            SnapshotOptions::new("test/counter.snapshot", "counter")
                .with_source_text("counter.v", source),
            || place_and_route_logical_design_with_visualization(&logical_input, &config),
        )?;
        let logical_source = std::fs::read_to_string("test/counter.snapshot/ir/logical.rcir")?;
        let logical: LogicalDesign = logical_source.parse()?;
        assert_eq!(logical.top, "counter");
        assert!(logical.module("counter").unwrap().cells.iter().any(|cell| {
            matches!(
                cell.kind,
                crate::ir::LogicalCellKind::Register { width: 2, .. }
            )
        }));
        let routable_source = std::fs::read_to_string("test/counter.snapshot/ir/routable.rcir")?;
        let routable: crate::ir::RoutableDocument = routable_source.parse()?;
        assert_eq!(routable.design.top, "counter");
        assert!(routable.design.modules.len() > 1);
        let design_profile = routable
            .design_bindings
            .get("counter")
            .and_then(|name| routable.design_profiles.get(name))
            .unwrap();
        assert_eq!(design_profile.search.layout_combinations, layout_limit);
        let mut embedded_config = GlobalPnrConfig::default();
        apply_routable_document(&routable, &mut embedded_config)?;
        assert_eq!(
            pnr_spec_from_config(&embedded_config),
            pnr_spec_from_config(&config)
        );
        assert!(result
            .routed_nets
            .iter()
            .all(|route| route.net_id.is_some() && route.source_endpoint.is_some()));
        assert!(result.routed_nets.iter().all(|route| {
            route
                .sink_label
                .as_deref()
                .is_some_and(|label| label.ends_with(".switch"))
                || route.sink_endpoint.is_some()
        }));
        let placed = result.placed_world;

        assert!(!placed.world.iter_block().is_empty());
        let mut output_names = placed
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>();
        output_names.sort();
        assert_eq!(output_names, vec!["q_0", "q_1"]);
        let nbt: NBTRoot = placed.world.to_nbt();
        nbt.save("test/counter-global-smoke.nbt");
        result
            .placement_bbox_world
            .to_nbt()
            .save("test/counter-global-placement-bbox.nbt");
        placed
            .metadata()
            .save("test/counter-global-smoke.outputs.json")?;
        assert_two_bit_counter_behavior(&placed)?;
        Ok(())
    }

    fn init_tracing_from_env() {
        let level = rust_log_level().unwrap_or(tracing::Level::INFO);
        let _ = tracing_subscriber::fmt().with_max_level(level).try_init();
    }

    fn rust_log_level() -> Option<tracing::Level> {
        let value = std::env::var("RUST_LOG").ok()?;
        value
            .split(',')
            .filter_map(|part| part.rsplit('=').next())
            .find_map(|level| match level.trim().to_ascii_lowercase().as_str() {
                "trace" => Some(tracing::Level::TRACE),
                "debug" => Some(tracing::Level::DEBUG),
                "info" => Some(tracing::Level::INFO),
                "warn" | "warning" => Some(tracing::Level::WARN),
                "error" => Some(tracing::Level::ERROR),
                _ => None,
            })
    }

    fn assert_positive_edge_dff_behavior(placed: &PlacedWorld) -> eyre::Result<()> {
        let data = input_endpoint_position(placed, "d")?;
        let clock = input_endpoint_position(placed, "clk")?;
        let output = placed.outputs[0].position();
        let world = World::from(&placed.world);
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        assert!(!block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(data, true)], 256, 50_000)?;
        assert!(!block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
        assert!(block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(clock, false)], 256, 50_000)?;
        assert!(block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(data, false)], 256, 50_000)?;
        assert!(block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
        assert!(!block_power(sim.world(), output));
        Ok(())
    }

    fn assert_two_bit_counter_behavior(placed: &PlacedWorld) -> eyre::Result<()> {
        let clock = input_endpoint_position(placed, "clk")?;
        let output_q0 = placed
            .outputs
            .iter()
            .find(|output| output.name == "q_0")
            .context("missing q_0 output")?
            .position();
        let output_q1 = placed
            .outputs
            .iter()
            .find(|output| output.name == "q_1")
            .context("missing q_1 output")?
            .position();
        let world = World::from(&placed.world);
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        let initial = counter_output_value(sim.world(), output_q0, output_q1);
        eyre::ensure!(
            initial == 0,
            "counter should initialize to 0: initial={initial}"
        );

        for expected in [1, 2, 3, 0, 1, 2, 3, 0] {
            sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
            let rising = counter_output_value(sim.world(), output_q0, output_q1);
            eyre::ensure!(
                rising == expected,
                "counter output mismatch on rising edge: expected={expected}, actual={rising}"
            );
            sim.advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)?;

            sim.change_state_with_limits(vec![(clock, false)], 256, 50_000)?;
            let falling = counter_output_value(sim.world(), output_q0, output_q1);
            eyre::ensure!(
                falling == expected,
                "counter output should hold on falling edge: expected={expected}, actual={falling}"
            );
            sim.advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)?;
        }
        Ok(())
    }

    fn counter_output_value(world: &World3D, q0: Position, q1: Position) -> usize {
        usize::from(block_power(world, q0)) | (usize::from(block_power(world, q1)) << 1)
    }

    fn input_endpoint_position(placed: &PlacedWorld, name: &str) -> eyre::Result<Position> {
        placed
            .inputs
            .iter()
            .find(|input| input.name == name)
            .map(|input| input.position())
            .with_context(|| format!("missing input endpoint `{name}`"))
    }

    fn block_power(world: &World3D, position: Position) -> bool {
        match world[position].kind {
            BlockKind::Redstone {
                strength, on_count, ..
            } => strength > 0 || on_count > 0,
            BlockKind::Torch { is_on }
            | BlockKind::Repeater { is_on, .. }
            | BlockKind::Switch { is_on } => is_on,
            BlockKind::Cobble {
                on_count,
                on_base_count,
            } => on_count > 0 || on_base_count > 0,
            BlockKind::RedstoneBlock => true,
            BlockKind::Air | BlockKind::Piston { .. } => false,
        }
    }

    fn not_clk_module() -> GraphModule {
        let mut module: GraphModule = LogicGraph::from_stmt("~clk", "clk_n").unwrap().graph.into();
        module.name = "not_clk".to_owned();
        module
    }
}
