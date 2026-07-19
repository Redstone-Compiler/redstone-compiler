use std::cmp::Reverse;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};

use eyre::ContextCompat;
use serde::Serialize;

use crate::graph::logic::LogicGraph;
use crate::graph::{Graph, GraphNodeKind};
use crate::ir::{
    graph_from_routable_leaf, CellFaceSpec, LocalCellContractSpec, PortAccessDirectionSpec,
    RoutableModule, RoutablePortDirection,
};
use crate::output::{OutputEndpoint, PlacedWorld};
use crate::snapshot::{emit_json, record as record_snapshot, SnapshotEvent};
use crate::transform::place_and_route::detailed_router;
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPort, PhysicalPortDirection, PortConnection,
};
use crate::transform::place_and_route::local_placer::{
    LocalPlacementFailure, LocalPlacementFailureKind, LocalPlacementStage, LocalPlacer,
    LocalPlacerConfig, LocalPlacerDebug, LocalPlacerInputConstraints, PlacementSamplingPolicy,
    PlacementSchedulePolicy, PlacementScheduler,
};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::transform::place_and_route::sampling::SamplingPolicy;
use crate::world::block::Block;
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

mod clustering;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitCandidateConfig {
    pub dim: DimSize,
    pub local_config: LocalPlacerConfig,
    pub input_constraints: LocalPlacerInputConstraints,
    pub max_candidates: usize,
    pub combinational_sampling_limit: Option<usize>,
    pub local_cell_contract: LocalCellContractSpec,
}

/// Local-candidate preparation policy before it is resolved against a typed
/// Routable definition. The default preserves the legacy single-policy API;
/// definition and port entries provide the scoped model used by RCIR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidatePolicySet {
    pub default: UnitCandidateConfig,
    pub definition_overrides: BTreeMap<String, UnitCandidateConfig>,
    pub pin_search: BTreeMap<(String, String), Vec<Position>>,
}

impl CandidatePolicySet {
    pub fn new(default: UnitCandidateConfig) -> Self {
        Self {
            default,
            definition_overrides: BTreeMap::new(),
            pin_search: BTreeMap::new(),
        }
    }

    pub fn with_definition_override(
        mut self,
        definition: impl Into<String>,
        policy: UnitCandidateConfig,
    ) -> Self {
        self.definition_overrides.insert(definition.into(), policy);
        self
    }

    pub fn with_pin_search(
        mut self,
        definition: impl Into<String>,
        port: impl Into<String>,
        positions: impl IntoIterator<Item = Position>,
    ) -> Self {
        self.pin_search.insert(
            (definition.into(), port.into()),
            positions.into_iter().collect(),
        );
        self
    }

    pub fn effective_for_definition(&self, definition: &str) -> UnitCandidateConfig {
        let mut policy = self
            .definition_overrides
            .get(definition)
            .cloned()
            .unwrap_or_else(|| self.default.clone());
        for ((owner, port), positions) in &self.pin_search {
            if owner == definition {
                policy.input_constraints = policy
                    .input_constraints
                    .with_input_positions(port.clone(), positions.iter().copied());
            }
        }
        policy
    }
}

impl Default for CandidatePolicySet {
    fn default() -> Self {
        Self::new(UnitCandidateConfig::default())
    }
}

impl From<UnitCandidateConfig> for CandidatePolicySet {
    fn from(default: UnitCandidateConfig) -> Self {
        Self::new(default)
    }
}

impl Deref for CandidatePolicySet {
    type Target = UnitCandidateConfig;

    fn deref(&self) -> &Self::Target {
        &self.default
    }
}

impl DerefMut for CandidatePolicySet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.default
    }
}

impl Default for UnitCandidateConfig {
    fn default() -> Self {
        Self {
            dim: DimSize(16, 16, 6),
            local_config: LocalPlacerConfig::default(),
            input_constraints: LocalPlacerInputConstraints::default(),
            max_candidates: 16,
            combinational_sampling_limit: None,
            local_cell_contract: LocalCellContractSpec::default(),
        }
    }
}

pub fn generate_routable_module_candidates_with_progress_label(
    module: &RoutableModule,
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
) -> eyre::Result<Vec<LayoutCandidate>> {
    generate_routable_module_candidates(
        module,
        config,
        progress_label,
        CandidateInputMode::ExternalPorts,
    )
}

pub fn generate_routable_top_leaf_candidates_with_progress_label(
    module: &RoutableModule,
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
) -> eyre::Result<Vec<LayoutCandidate>> {
    generate_routable_module_candidates(
        module,
        config,
        progress_label,
        CandidateInputMode::MaterializedSwitches,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateInputMode {
    ExternalPorts,
    MaterializedSwitches,
}

fn generate_routable_module_candidates(
    module: &RoutableModule,
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
    input_mode: CandidateInputMode,
) -> eyre::Result<Vec<LayoutCandidate>> {
    let graph = graph_from_routable_leaf(module)?;
    let ports: Vec<_> = module
        .ports
        .iter()
        .map(|port| {
            CandidatePort::new(
                &port.name,
                &port.name,
                match port.direction {
                    RoutablePortDirection::Input => PhysicalPortDirection::Input,
                    RoutablePortDirection::Output => PhysicalPortDirection::Output,
                },
            )
        })
        .collect();
    let clustered = clustering::try_generate_clustered_candidates(
        &module.name,
        &graph,
        &ports,
        config,
        progress_label,
        input_mode,
    )?;
    let mut candidates = generate_unit_candidates(
        &module.name,
        graph,
        ports,
        config,
        progress_label,
        input_mode,
    )?;
    if let Some(clustered) = clustered {
        candidates.extend(clustered);
        // Clustering is an additional search representation, not a forced
        // replacement. Keep the compact monolithic result when composition
        // overhead outweighs the benefit, while retaining clustered layouts
        // when they win on physical cost.
        candidates.sort_by_key(|candidate| {
            (
                candidate.cost.bbox_volume,
                candidate.cost.block_count,
                candidate.cost.bbox_height,
                candidate.cost.bbox_footprint,
            )
        });
        candidates.truncate(config.max_candidates.max(1));
    }
    Ok(candidates)
}

#[derive(Clone, Debug)]
struct CandidatePort {
    name: String,
    target: String,
    direction: PhysicalPortDirection,
}

impl CandidatePort {
    fn new(name: &str, target: &str, direction: PhysicalPortDirection) -> Self {
        Self {
            name: name.to_owned(),
            target: target.to_owned(),
            direction,
        }
    }
}

fn generate_unit_candidates(
    module_name: &str,
    graph: Graph,
    ports: Vec<CandidatePort>,
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
    input_mode: CandidateInputMode,
) -> eyre::Result<Vec<LayoutCandidate>> {
    // Routable leaf bodies are already technology-mapped. Candidate
    // generation must preserve that graph exactly.
    let graph = LogicGraph { graph };
    let contains_sequential = graph
        .graph
        .nodes
        .iter()
        .any(|node| matches!(node.kind, GraphNodeKind::Sequential(_)));
    let validate_truth_table = !contains_sequential;
    let mut candidates = Vec::new();
    let mut generated_count = 0usize;
    let mut truth_table_rejections = 0usize;
    let mut port_rejections = 0usize;
    let mut contract_rejections = 0usize;
    let mut schedules = if config.local_config.schedule == PlacementSchedulePolicy::Auto {
        PlacementScheduler::new(&graph).candidates()
    } else {
        vec![PlacementScheduler::new(&graph).select(config.local_config.schedule)]
    };
    // Keep auto mode performance-neutral for graphs already handled by the
    // canonical order. Heuristic schedules are recovery paths, ranked by
    // their static metrics, rather than speculative work paid on every run.
    let topological_order = graph.topological_order();
    schedules.sort_by_key(|schedule| (schedule.order != topological_order, schedule.metrics));
    let schedule_count = schedules.len();
    let mut attempted_schedules = 0usize;
    let mut adaptive_retries = 0usize;
    let mut schedule_reports = Vec::new();

    for (schedule_attempt, schedule) in schedules.into_iter().enumerate() {
        if candidates.len() >= config.max_candidates {
            break;
        }
        attempted_schedules += 1;
        if schedule_count > 1 {
            tracing::info!(
                module = module_name,
                schedule_attempt = schedule_attempt + 1,
                schedule_count,
                input_lifetime = schedule.metrics.input_lifetime,
                peak_frontier = schedule.metrics.peak_frontier,
                total_frontier = schedule.metrics.total_frontier,
                edge_lifetime = schedule.metrics.edge_lifetime,
                "trying local placement schedule"
            );
        }
        let accepted_before = candidates.len();
        let mut failures = Vec::new();
        let mut local_config = config.local_config;
        let mut adaptive_attempt = 0usize;
        let placed = loop {
            let placer = LocalPlacer::new_with_visit_order(
                graph.clone(),
                local_config,
                schedule.order.clone(),
            )?;
            let mut debug = LocalPlacerDebug::default();
            let mut placed = match input_mode {
                CandidateInputMode::ExternalPorts => placer
                    .generate_with_outputs_and_planned_inputs_debug_progress(
                        config.dim,
                        None,
                        &config.input_constraints,
                        Some(&mut debug),
                        progress_label,
                    ),
                CandidateInputMode::MaterializedSwitches => placer
                    .generate_with_outputs_and_input_constraints_debug_progress(
                        config.dim,
                        None,
                        &config.input_constraints,
                        Some(&mut debug),
                        progress_label,
                    ),
            };
            if input_mode == CandidateInputMode::ExternalPorts && placed.is_empty() {
                tracing::info!(
                    module = module_name,
                    "planned child inputs produced no candidates; retrying incremental input placement"
                );
                debug = LocalPlacerDebug::default();
                placed = placer.generate_with_outputs_and_input_constraints_debug_progress(
                    config.dim,
                    None,
                    &config.input_constraints,
                    Some(&mut debug),
                    progress_label,
                );
            }
            if !placed.is_empty() {
                break placed;
            }

            let Some(failure) = debug.failure() else {
                break placed;
            };
            failures.push(failure.clone());
            tracing::debug!(
                module = module_name,
                stage = ?failure.stage,
                failure = ?failure.kind,
                step = failure.step + 1,
                total_steps = failure.total_steps,
                node_id = failure.node_id,
                node_kind = failure.node_kind,
                input_candidates = failure.input_candidates,
                generated_candidates = failure.generated_candidates,
                route_calls = failure.route_calls,
                route_candidates = failure.route_candidates,
                "local candidate frontier exhausted"
            );

            if adaptive_attempt >= 1 {
                break placed;
            }
            let Some(escalated) = adaptive_retry_config(local_config, &failure) else {
                break placed;
            };
            adaptive_attempt += 1;
            adaptive_retries += 1;
            tracing::info!(
                module = module_name,
                stage = ?failure.stage,
                retry = adaptive_attempt,
                max_retries = 1,
                max_not_route_step = escalated.max_not_route_step,
                max_route_step = escalated.max_route_step,
                "retrying local candidate generation with a bounded stage-specific budget"
            );
            local_config = escalated;
        };
        let placed_count = placed.len();
        generated_count += placed.len();

        for placed in placed {
            if candidates.len() >= config.max_candidates {
                break;
            }
            if validate_truth_table {
                match candidate_matches_truth_table(&graph, &placed) {
                    Ok(true) => {}
                    Ok(false) => {
                        truth_table_rejections += 1;
                        continue;
                    }
                    Err(error) => {
                        tracing::debug!(
                            module = module_name,
                            error = %error,
                            "rejecting local candidate with incomplete or invalid endpoints"
                        );
                        port_rejections += 1;
                        continue;
                    }
                }
            }
            let (world, physical_ports) = candidate_layout(
                &ports,
                contains_sequential,
                &config.input_constraints,
                placed.world,
                &placed.inputs,
                &placed.outputs,
                input_mode,
            );
            if !candidate_ports_cover_module_ports(&ports, &physical_ports) {
                port_rejections += 1;
                continue;
            }
            let mut candidate =
                LayoutCandidate::from_world(module_name.to_owned(), world, physical_ports)?;
            if !candidate_satisfies_local_cell_contract(&mut candidate, &config.local_cell_contract)
            {
                contract_rejections += 1;
                continue;
            }
            candidates.push(candidate);
        }
        schedule_reports.push(LocalScheduleAttemptReport {
            attempt: schedule_attempt + 1,
            input_lifetime: schedule.metrics.input_lifetime,
            peak_frontier: schedule.metrics.peak_frontier,
            total_frontier: schedule.metrics.total_frontier,
            edge_lifetime: schedule.metrics.edge_lifetime,
            adaptive_retries: adaptive_attempt,
            generated: placed_count,
            accepted: candidates.len() - accepted_before,
            failures,
        });
    }
    tracing::info!(
        module = module_name,
        generated = generated_count,
        accepted = candidates.len(),
        attempted_schedules,
        truth_table_rejections,
        port_rejections,
        contract_rejections,
        "local candidate validation completed"
    );
    let report = LocalCandidateSearchReport {
        format: "redstone-compiler.local-candidate-search.v1",
        module: module_name,
        requested_candidates: config.max_candidates,
        schedules_available: schedule_count,
        attempted_schedules,
        adaptive_retries,
        generated: generated_count,
        accepted: candidates.len(),
        truth_table_rejections,
        port_rejections,
        attempts: schedule_reports,
        candidates: candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| LocalCandidateQuality {
                index,
                volume: candidate.cost.bbox_volume,
                footprint: candidate.cost.bbox_footprint,
                height: candidate.cost.bbox_height,
                blocks: candidate.cost.block_count,
                port_access_count: candidate.cost.port_access_points,
            })
            .collect(),
    };
    emit_json(
        format!(
            "candidates/search/{}.json",
            snapshot_file_component(module_name)
        ),
        &report,
    )?;
    record_snapshot(SnapshotEvent::LocalCandidateSearch {
        module: module_name.to_owned(),
        attempted_schedules,
        adaptive_retries,
        failures: report
            .attempts
            .iter()
            .map(|attempt| attempt.failures.len())
            .sum(),
        generated: generated_count,
        accepted: candidates.len(),
    });
    Ok(candidates)
}

fn candidate_satisfies_local_cell_contract(
    candidate: &mut LayoutCandidate,
    contract: &LocalCellContractSpec,
) -> bool {
    if contract.max_bbox.is_none() && contract.ports.is_empty() {
        return true;
    }

    // A cell package includes both occupied blocks and its declared routing
    // access points. This prevents an otherwise small world from satisfying a
    // bbox contract while exposing a pin outside that package.
    let mut min = candidate.bbox.min;
    let mut max = candidate.bbox.max;
    for position in candidate
        .ports
        .iter()
        .flat_map(|port| std::iter::once(port.position).chain(port.routing_access_positions()))
    {
        min.0 = min.0.min(position.0);
        min.1 = min.1.min(position.1);
        min.2 = min.2.min(position.2);
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }

    if let Some(limit) = contract.max_bbox {
        let extent = [max.0 - min.0 + 1, max.1 - min.1 + 1, max.2 - min.2 + 1];
        if extent
            .into_iter()
            .zip(limit)
            .any(|(actual, maximum)| actual > maximum)
        {
            return false;
        }
    }

    for (name, requirement) in &contract.ports {
        let Some(port) = candidate.ports.iter_mut().find(|port| port.name == *name) else {
            return false;
        };
        let direction_matches = match requirement.access {
            PortAccessDirectionSpec::Inward => port.direction == PhysicalPortDirection::Input,
            PortAccessDirectionSpec::Outward => port.direction == PhysicalPortDirection::Output,
            PortAccessDirectionSpec::Bidirectional => true,
        };
        if !direction_matches {
            return false;
        }
        let matching_access = port
            .routing_access_positions()
            .into_iter()
            .filter(|position| position_is_on_face(*position, min, max, requirement.face))
            .collect::<Vec<_>>();
        if matching_access.is_empty() {
            return false;
        }
        port.route_position = matching_access.first().copied();
        port.access_points = matching_access;
    }
    candidate.cost.port_access_points = candidate
        .ports
        .iter()
        .map(|port| port.routing_access_positions().len())
        .sum();
    true
}

fn position_is_on_face(
    position: Position,
    min: Position,
    max: Position,
    face: CellFaceSpec,
) -> bool {
    match face {
        CellFaceSpec::West => position.0 == min.0,
        CellFaceSpec::East => position.0 == max.0,
        CellFaceSpec::North => position.1 == min.1,
        CellFaceSpec::South => position.1 == max.1,
        CellFaceSpec::Down => position.2 == min.2,
        CellFaceSpec::Up => position.2 == max.2,
    }
}

#[derive(Debug, Serialize)]
struct LocalCandidateSearchReport<'a> {
    format: &'static str,
    module: &'a str,
    requested_candidates: usize,
    schedules_available: usize,
    attempted_schedules: usize,
    adaptive_retries: usize,
    generated: usize,
    accepted: usize,
    truth_table_rejections: usize,
    port_rejections: usize,
    attempts: Vec<LocalScheduleAttemptReport>,
    candidates: Vec<LocalCandidateQuality>,
}

#[derive(Debug, Serialize)]
struct LocalScheduleAttemptReport {
    attempt: usize,
    input_lifetime: usize,
    peak_frontier: usize,
    total_frontier: usize,
    edge_lifetime: usize,
    adaptive_retries: usize,
    generated: usize,
    accepted: usize,
    failures: Vec<LocalPlacementFailure>,
}

#[derive(Debug, Serialize)]
struct LocalCandidateQuality {
    index: usize,
    volume: usize,
    footprint: usize,
    height: usize,
    blocks: usize,
    port_access_count: usize,
}

fn snapshot_file_component(name: &str) -> String {
    if name.is_empty() {
        return "unnamed".to_owned();
    }
    let mut encoded = String::new();
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

fn adaptive_retry_config(
    config: LocalPlacerConfig,
    failure: &LocalPlacementFailure,
) -> Option<LocalPlacerConfig> {
    let mut retry = config;
    match failure.stage {
        LocalPlacementStage::NotRouting => {
            if failure.kind == LocalPlacementFailureKind::RouteDepthExhausted {
                retry.max_not_route_step = grow_route_depth(config.max_not_route_step);
            }
            if failure.kind == LocalPlacementFailureKind::RouteBeamExhausted {
                retry.not_route_step_sampling_policy =
                    grow_sampling(config.not_route_step_sampling_policy);
            }
            retry.step_sampling_policy = grow_sampling(config.step_sampling_policy);
            retry.placement_sampling_policy =
                grow_placement_sampling(config.placement_sampling_policy);
        }
        LocalPlacementStage::OrRouting => {
            if failure.kind == LocalPlacementFailureKind::RouteDepthExhausted {
                retry.max_route_step = grow_route_depth(config.max_route_step);
            }
            if failure.kind == LocalPlacementFailureKind::RouteBeamExhausted {
                retry.route_step_sampling_policy = grow_sampling(config.route_step_sampling_policy);
            }
            retry.step_sampling_policy = grow_sampling(config.step_sampling_policy);
            retry.placement_sampling_policy =
                grow_placement_sampling(config.placement_sampling_policy);
        }
        LocalPlacementStage::SequentialPlacement | LocalPlacementStage::OtherPlacement => {
            retry.step_sampling_policy = grow_sampling(config.step_sampling_policy);
            retry.placement_sampling_policy =
                grow_placement_sampling(config.placement_sampling_policy);
        }
        LocalPlacementStage::InputPlacement | LocalPlacementStage::OutputPlacement => return None,
    }
    (retry != config).then_some(retry)
}

fn grow_route_depth(depth: usize) -> usize {
    if depth == 0 {
        0
    } else {
        depth.saturating_mul(2).min(16)
    }
}

fn grow_sampling(policy: SamplingPolicy) -> SamplingPolicy {
    match policy {
        SamplingPolicy::None => SamplingPolicy::None,
        SamplingPolicy::Take(count) => SamplingPolicy::Take(count.saturating_mul(2)),
        SamplingPolicy::Random(count) => SamplingPolicy::Random(count.saturating_mul(2)),
    }
}

fn grow_placement_sampling(policy: PlacementSamplingPolicy) -> PlacementSamplingPolicy {
    match policy {
        PlacementSamplingPolicy::StepPolicy => PlacementSamplingPolicy::StepPolicy,
        PlacementSamplingPolicy::Cost {
            count,
            random_count,
            start_step,
        } => PlacementSamplingPolicy::Cost {
            count: count.saturating_mul(2),
            random_count: random_count.saturating_mul(2),
            start_step,
        },
        PlacementSamplingPolicy::Ranked {
            count,
            random_count,
            start_step,
        } => PlacementSamplingPolicy::Ranked {
            count: count.saturating_mul(2),
            random_count: random_count.saturating_mul(2),
            start_step,
        },
    }
}

fn candidate_ports_cover_module_ports(expected: &[CandidatePort], actual: &[PhysicalPort]) -> bool {
    expected
        .iter()
        .all(|expected| actual.iter().any(|port| port.name == expected.name))
}

fn candidate_matches_truth_table(
    expected: &LogicGraph,
    placed: &PlacedWorld,
) -> eyre::Result<bool> {
    let expected = expected.truth_table()?;
    let inputs = expected
        .input_names
        .iter()
        .map(|name| {
            placed
                .inputs
                .iter()
                .find(|input| input.name == *name)
                .map(|input| input.position())
                .with_context(|| format!("missing input endpoint `{name}`"))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let outputs = expected
        .output_tables
        .keys()
        .map(|name| {
            placed
                .outputs
                .iter()
                .find(|output| output.name == *name)
                .map(|output| (name.as_str(), output.position()))
                .with_context(|| format!("missing output endpoint `{name}`"))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let world = World::from(&placed.world);

    for mask in 0..(1usize << inputs.len()) {
        let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        sim.change_state_with_limits(
            inputs
                .iter()
                .enumerate()
                .map(|(index, position)| (*position, (mask & (1 << index)) != 0))
                .collect(),
            256,
            50_000,
        )?;

        for (output_name, output_position) in &outputs {
            let Some(expected_output) = expected.output_tables.get(*output_name) else {
                return Ok(false);
            };
            if sim.world()[*output_position].kind.is_powered() != expected_output[mask] {
                return Ok(false);
            }
        }
    }

    // A candidate is used as a persistent child inside a routed design, so
    // matching each truth-table row from a freshly initialized world is not
    // sufficient. Exercise both directions in one simulator as well; this
    // rejects layouts whose redstone network powers correctly from reset but
    // fails to release after an input transition.
    let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0)
        .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
    let mask_count = 1usize << inputs.len();
    for mask in (0..mask_count).chain((0..mask_count).rev()) {
        sim.change_state_with_limits(
            inputs
                .iter()
                .enumerate()
                .map(|(index, position)| (*position, (mask & (1 << index)) != 0))
                .collect(),
            256,
            50_000,
        )?;
        for (output_name, output_position) in &outputs {
            let Some(expected_output) = expected.output_tables.get(*output_name) else {
                return Ok(false);
            };
            if sim.world()[*output_position].kind.is_powered() != expected_output[mask] {
                return Ok(false);
            }
        }
        sim.advance_idle_cycles(crate::world::simulator::MANUAL_INPUT_IDLE_CYCLES)?;
    }

    Ok(true)
}

// LocalPlacer는 아직 standalone 회로를 기준으로 switch/output layout을 만든다.
// Global PnR child layout에서는 switch를 제거하고 외부 route가 물릴 수 있는
// module port metadata로 다시 노출한다.
// TODO(high-level): make LocalPlacer produce either standalone layouts with switches
// or child-module layouts with PhysicalPort metadata, instead of rewriting switches here.
fn candidate_layout(
    module_ports: &[CandidatePort],
    contains_sequential: bool,
    input_constraints: &LocalPlacerInputConstraints,
    mut world: World3D,
    inputs: &[OutputEndpoint],
    outputs: &[OutputEndpoint],
    input_mode: CandidateInputMode,
) -> (World3D, Vec<PhysicalPort>) {
    let mut ports = Vec::new();
    // Sequential child layout은 내부 feedback/state signal이 외부 route와 직접
    // 합쳐지면 back-power 때문에 latch 상태가 깨질 수 있어서 diode 연결을 요구한다.
    let needs_output_isolation = contains_sequential;
    let needs_input_isolation = contains_sequential;
    let use_direct_input_ports = !contains_sequential
        && module_ports
            .iter()
            .filter(|port| port.direction == PhysicalPortDirection::Input)
            .count()
            > 1;
    let preserve_switch_position_inputs = contains_sequential || use_direct_input_ports;
    for port in module_ports {
        match port.direction {
            PhysicalPortDirection::Input => {
                let input_name = &port.target;
                let position = inputs
                    .iter()
                    .find(|input| input.name == *input_name)
                    .map(|input| input.position())
                    .or_else(|| {
                        input_constraints
                            .positions_for_input_name(input_name)
                            .and_then(|positions| positions.into_iter().next())
                    });
                if let Some(input_position) = position {
                    if input_mode == CandidateInputMode::MaterializedSwitches {
                        ports.push(PhysicalPort {
                            name: port.name.clone(),
                            direction: PhysicalPortDirection::Input,
                            position: input_position,
                            route_position: None,
                            access_points: vec![input_position],
                            connection: PortConnection::Direct,
                        });
                        continue;
                    }
                    let Some(position) = expose_switchless_input_port(
                        &mut world,
                        input_position,
                        preserve_switch_position_inputs,
                        use_direct_input_ports,
                    ) else {
                        continue;
                    };
                    ports.push(PhysicalPort {
                        name: port.name.clone(),
                        direction: PhysicalPortDirection::Input,
                        position,
                        route_position: None,
                        access_points: vec![position],
                        connection: if needs_input_isolation || world[position].kind.is_redstone() {
                            PortConnection::InputDiode
                        } else {
                            PortConnection::Direct
                        },
                    });
                }
            }
            PhysicalPortDirection::Output => {
                let output_name = &port.target;
                if let Some(output) = outputs.iter().find(|output| output.name == *output_name) {
                    let position = output.position();
                    let access_points = expose_routeable_output_ports(&world, position);
                    let route_position = access_points[0];
                    ports.push(PhysicalPort {
                        name: port.name.clone(),
                        direction: PhysicalPortDirection::Output,
                        position,
                        route_position: Some(route_position),
                        access_points,
                        connection: if needs_output_isolation {
                            PortConnection::OutputDiode
                        } else {
                            PortConnection::Direct
                        },
                    });
                }
            }
        }
    }
    if input_mode == CandidateInputMode::ExternalPorts {
        for input in inputs {
            let _ = expose_switchless_input_port(
                &mut world,
                input.position(),
                preserve_switch_position_inputs,
                use_direct_input_ports,
            );
        }
        remove_local_input_switches(&mut world);
    }
    ports.sort_by(|a, b| a.name.cmp(&b.name));
    world.initialize_redstone_states();
    (world, ports)
}

fn remove_local_input_switches(world: &mut World3D) {
    for (position, block) in world.iter_block() {
        if block.kind.is_switch() {
            world[position] = Block::default();
        }
    }
}

// Torch/switch/repeater 같은 출력 블록은 바로 route하기 어려울 수 있으므로,
// 해당 출력이 실제로 power하는 redstone tap들을 route access point로 노출한다.
fn expose_routeable_output_ports(world: &World3D, output_position: Position) -> Vec<Position> {
    if !world.size.bound_on(output_position)
        || (!world[output_position].kind.is_torch()
            && !world[output_position].kind.is_switch()
            && !world[output_position].kind.is_repeater())
    {
        return vec![output_position];
    }

    let mut access_points = world
        .iter_block()
        .into_iter()
        .filter(|(position, block)| {
            block.kind.is_redstone()
                && detailed_router::target_powers_position(world, output_position, *position)
        })
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    access_points.sort_by_key(|position| {
        (
            output_position.manhattan_distance(position),
            position.0,
            position.1,
            position.2,
        )
    });
    access_points.dedup();
    if access_points.is_empty() {
        access_points.push(output_position);
    }
    access_points
}

fn expose_routeable_output_port(world: &World3D, output_position: Position) -> Position {
    expose_routeable_output_ports(world, output_position)[0]
}

// LocalPlacer 입력은 보통 switch로 시작하므로 global PnR child layout에서는
// switch를 제거하고, switch가 물리던 cobble 또는 redstone fanout을 input port로 노출한다.
// TODO(low-level): replace this inference with explicit input-port placement metadata
// from LocalPlacer, so this code does not need to guess from switch wiring.
fn expose_switchless_input_port(
    world: &mut World3D,
    input_position: Position,
    preserve_switch_position_input: bool,
    use_direct_input_port: bool,
) -> Option<Position> {
    if !world.size.bound_on(input_position) {
        return None;
    }
    if !world[input_position].kind.is_switch() {
        return Some(input_position);
    }

    let switch_target = input_position.walk(world[input_position].direction);
    if let Some(target) = switch_target
        .filter(|position| world.size.bound_on(*position) && world[*position].kind.is_cobble())
    {
        if use_direct_input_port {
            if let Some(port_position) = switch_powered_redstone_port(world, input_position, true) {
                world[input_position] = Block::default();
                return Some(port_position);
            }
        }
        world[input_position] = Block::default();
        return Some(target);
    }

    if preserve_switch_position_input && switch_powers_redstone(world, input_position) {
        ensure_redstone_support(world, input_position)?;
        world[input_position] = PlacedNode::new_redstone(input_position).block;
        return Some(input_position);
    }

    if let Some(port_position) =
        switch_powered_redstone_port(world, input_position, use_direct_input_port)
    {
        world[input_position] = Block::default();
        return Some(port_position);
    }

    let port_position = expose_routeable_output_port(world, input_position);
    (port_position != input_position).then(|| {
        world[input_position] = Block::default();
        port_position
    })
}

fn switch_powers_redstone(world: &World3D, input_position: Position) -> bool {
    world.iter_block().into_iter().any(|(position, block)| {
        block.kind.is_redstone()
            && detailed_router::target_powers_position(world, input_position, position)
    })
}

fn ensure_redstone_support(world: &mut World3D, position: Position) -> Option<()> {
    let support_position = position.down()?;
    if !world.size.bound_on(support_position) {
        return None;
    }
    if world[support_position].kind.is_cobble() {
        return Some(());
    }
    if !world[support_position].kind.is_air() {
        return None;
    }
    world[support_position] = PlacedNode::new_cobble(support_position).block;
    Some(())
}

fn switch_powered_redstone_port(
    world: &World3D,
    input_position: Position,
    direct_only: bool,
) -> Option<Position> {
    let direct = world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| {
            (block.kind.is_redstone()
                && detailed_router::target_powers_position(world, input_position, position))
            .then_some(position)
        })
        .collect::<Vec<_>>();
    if direct.is_empty() {
        return None;
    }

    let candidates = if direct_only {
        direct
    } else {
        redstone_network_positions(world, &direct)
    };

    candidates.into_iter().max_by_key(|position| {
        (
            downstream_consumer_count(world, *position),
            Reverse(input_position.manhattan_distance(position)),
            Reverse(position.0),
            Reverse(position.1),
            Reverse(position.2),
        )
    })
}

fn redstone_network_positions(world: &World3D, seeds: &[Position]) -> Vec<Position> {
    let redstones = world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| block.kind.is_redstone().then_some(position))
        .collect::<Vec<_>>();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    for &seed in seeds {
        if visited.insert(seed) {
            queue.push_back(seed);
        }
    }

    while let Some(position) = queue.pop_front() {
        for &next in &redstones {
            if visited.contains(&next) {
                continue;
            }
            if detailed_router::target_powers_position(world, position, next)
                || detailed_router::target_powers_position(world, next, position)
            {
                visited.insert(next);
                queue.push_back(next);
            }
        }
    }

    visited.into_iter().collect()
}

fn downstream_consumer_count(world: &World3D, source: Position) -> usize {
    world
        .iter_block()
        .into_iter()
        .filter(|(position, block)| {
            *position != source
                && !block.kind.is_redstone()
                && detailed_router::target_powers_position(world, source, *position)
        })
        .count()
}

pub fn d_latch_child_candidate_config(local_config: LocalPlacerConfig) -> UnitCandidateConfig {
    UnitCandidateConfig {
        dim: DimSize(14, 10, 6),
        local_config,
        input_constraints: LocalPlacerInputConstraints::new()
            .with_input_positions("d", [Position(0, 2, 1)])
            .with_input_positions("en", [Position(0, 6, 1)]),
        max_candidates: 1,
        combinational_sampling_limit: None,
        local_cell_contract: LocalCellContractSpec::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{BlockKind, Direction};

    fn contract_test_candidate(port_position: Position) -> LayoutCandidate {
        let mut world = World3D::new(DimSize(6, 6, 4));
        world[Position(1, 1, 1)] = PlacedNode::new_cobble(Position(1, 1, 1)).block;
        world[Position(3, 3, 2)] = PlacedNode::new_cobble(Position(3, 3, 2)).block;
        LayoutCandidate::from_world(
            "contract-test".to_owned(),
            world,
            vec![PhysicalPort {
                name: "a".to_owned(),
                direction: PhysicalPortDirection::Input,
                position: port_position,
                route_position: None,
                access_points: vec![port_position],
                connection: PortConnection::Direct,
            }],
        )
        .expect("candidate")
    }

    fn west_input_contract(max_bbox: [usize; 3]) -> LocalCellContractSpec {
        LocalCellContractSpec {
            max_bbox: Some(max_bbox),
            ports: BTreeMap::from([(
                "a".to_owned(),
                crate::ir::PortAccessSpec {
                    face: CellFaceSpec::West,
                    access: PortAccessDirectionSpec::Inward,
                },
            )]),
        }
    }

    #[test]
    fn local_cell_contract_accepts_candidate_inside_bbox_with_on_face_pin() {
        let mut candidate = contract_test_candidate(Position(1, 2, 1));

        assert!(candidate_satisfies_local_cell_contract(
            &mut candidate,
            &west_input_contract([3, 3, 2])
        ));
    }

    #[test]
    fn local_cell_contract_rejects_oversized_candidate() {
        let mut candidate = contract_test_candidate(Position(1, 2, 1));

        assert!(!candidate_satisfies_local_cell_contract(
            &mut candidate,
            &west_input_contract([2, 3, 2])
        ));
    }

    #[test]
    fn local_cell_contract_rejects_off_face_pin() {
        let mut candidate = contract_test_candidate(Position(2, 2, 1));

        assert!(!candidate_satisfies_local_cell_contract(
            &mut candidate,
            &west_input_contract([3, 3, 2])
        ));
    }

    fn failure(stage: LocalPlacementStage) -> LocalPlacementFailure {
        LocalPlacementFailure {
            stage,
            kind: LocalPlacementFailureKind::RouteDepthExhausted,
            step: 2,
            total_steps: 5,
            node_id: 3,
            node_kind: "test".to_owned(),
            input_candidates: 8,
            generated_candidates: 0,
            route_calls: 8,
            route_candidates: 0,
        }
    }

    #[test]
    fn adaptive_retry_only_grows_the_failed_or_routing_budget() {
        let config = LocalPlacerConfig {
            max_not_route_step: 3,
            not_route_step_sampling_policy: SamplingPolicy::Random(11),
            max_route_step: 4,
            route_step_sampling_policy: SamplingPolicy::Random(16),
            placement_sampling_policy: PlacementSamplingPolicy::Ranked {
                count: 32,
                random_count: 4,
                start_step: 1,
            },
            ..Default::default()
        };

        let retry = adaptive_retry_config(config, &failure(LocalPlacementStage::OrRouting))
            .expect("retry config");

        assert_eq!(retry.max_route_step, 8);
        assert_eq!(retry.route_step_sampling_policy, SamplingPolicy::Random(16));
        assert_eq!(retry.max_not_route_step, 3);
        assert_eq!(
            retry.not_route_step_sampling_policy,
            SamplingPolicy::Random(11)
        );
        assert_eq!(
            retry.placement_sampling_policy,
            PlacementSamplingPolicy::Ranked {
                count: 64,
                random_count: 8,
                start_step: 1,
            }
        );
    }

    #[test]
    fn adaptive_retry_grows_sampling_when_the_route_beam_was_pruned() {
        let config = LocalPlacerConfig {
            max_route_step: 4,
            route_step_sampling_policy: SamplingPolicy::Random(16),
            ..Default::default()
        };
        let mut failure = failure(LocalPlacementStage::OrRouting);
        failure.kind = LocalPlacementFailureKind::RouteBeamExhausted;

        let retry = adaptive_retry_config(config, &failure).expect("retry config");

        assert_eq!(retry.max_route_step, 4);
        assert_eq!(retry.route_step_sampling_policy, SamplingPolicy::Random(32));
    }

    #[test]
    fn adaptive_retry_does_not_expand_impossible_input_constraints() {
        assert!(adaptive_retry_config(
            LocalPlacerConfig::default(),
            &failure(LocalPlacementStage::InputPlacement)
        )
        .is_none());
    }

    #[test]
    fn local_candidate_search_report_serializes_failures_and_quality() {
        let report = LocalCandidateSearchReport {
            format: "redstone-compiler.local-candidate-search.v1",
            module: "adder/core",
            requested_candidates: 2,
            schedules_available: 3,
            attempted_schedules: 2,
            adaptive_retries: 1,
            generated: 4,
            accepted: 1,
            truth_table_rejections: 2,
            port_rejections: 1,
            attempts: vec![LocalScheduleAttemptReport {
                attempt: 1,
                input_lifetime: 3,
                peak_frontier: 4,
                total_frontier: 12,
                edge_lifetime: 20,
                adaptive_retries: 1,
                generated: 0,
                accepted: 0,
                failures: vec![failure(LocalPlacementStage::OrRouting)],
            }],
            candidates: vec![LocalCandidateQuality {
                index: 0,
                volume: 245,
                footprint: 49,
                height: 5,
                blocks: 65,
                port_access_count: 3,
            }],
        };

        let value = serde_json::to_value(report).expect("serialize report");
        assert_eq!(value["attempts"][0]["failures"][0]["stage"], "or_routing");
        assert_eq!(
            value["attempts"][0]["failures"][0]["kind"],
            "route_depth_exhausted"
        );
        assert_eq!(value["candidates"][0]["volume"], 245);
        assert_eq!(value["candidates"][0]["port_access_count"], 3);
        assert_eq!(snapshot_file_component("adder/core"), "adder%2Fcore");
        assert_eq!(snapshot_file_component("full_adder"), "full_adder");
    }

    #[test]
    fn candidate_pin_search_is_scoped_by_definition_and_port() {
        let policies = CandidatePolicySet::default()
            .with_pin_search("first", "d", [Position(1, 2, 3)])
            .with_pin_search("second", "d", [Position(4, 5, 1)]);

        assert_eq!(
            policies
                .effective_for_definition("first")
                .input_constraints
                .positions_for_input_name("d"),
            Some(vec![Position(1, 2, 3)])
        );
        assert_eq!(
            policies
                .effective_for_definition("second")
                .input_constraints
                .positions_for_input_name("d"),
            Some(vec![Position(4, 5, 1)])
        );
        assert_eq!(
            policies
                .effective_for_definition("third")
                .input_constraints
                .positions_for_input_name("d"),
            None
        );
    }

    #[test]
    fn switchless_direct_input_exposes_powered_redstone_instead_of_support_cobble() {
        let switch = Position(1, 1, 1);
        let support = Position(1, 1, 0);
        let input_cobble = Position(2, 1, 1);
        let input_redstone = Position(2, 1, 2);
        let mut world = World3D::new(DimSize(4, 3, 4));
        world[support] = PlacedNode::new_cobble(support).block;
        world[switch] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::East,
        };
        world[input_cobble] = PlacedNode::new_cobble(input_cobble).block;
        world[input_redstone] = PlacedNode::new_redstone(input_redstone).block;
        world.initialize_redstone_states();

        let port =
            expose_switchless_input_port(&mut world, switch, false, true).expect("input port");

        assert_eq!(port, input_redstone);
        assert!(world[switch].kind.is_air());
    }
}
