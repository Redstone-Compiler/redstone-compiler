use std::cmp::Reverse;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};

use eyre::ContextCompat;

use crate::graph::logic::LogicGraph;
use crate::graph::{Graph, GraphNodeKind};
use crate::ir::{graph_from_routable_leaf, RoutableModule, RoutablePortDirection};
use crate::output::{OutputEndpoint, PlacedWorld};
use crate::transform::place_and_route::detailed_router;
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPort, PhysicalPortDirection, PortConnection,
};
use crate::transform::place_and_route::local_placer::{
    LocalPlacer, LocalPlacerConfig, LocalPlacerInputConstraints,
};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::Block;
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitCandidateConfig {
    pub dim: DimSize,
    pub local_config: LocalPlacerConfig,
    pub input_constraints: LocalPlacerInputConstraints,
    pub max_candidates: usize,
    pub combinational_sampling_limit: Option<usize>,
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
    let ports = module
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
    generate_unit_candidates(
        &module.name,
        graph,
        ports,
        config,
        progress_label,
        input_mode,
    )
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
    let graph = LogicGraph { graph }.prepare_place()?;
    let placer = LocalPlacer::new(graph.clone(), config.local_config)?;

    let placed = placer.generate_with_outputs_and_input_constraints_progress(
        config.dim,
        None,
        &config.input_constraints,
        progress_label,
    );
    let generated_count = placed.len();

    let contains_sequential = graph
        .graph
        .nodes
        .iter()
        .any(|node| matches!(node.kind, GraphNodeKind::Sequential(_)));
    let validate_truth_table = !contains_sequential;
    let mut candidates = Vec::new();
    let mut truth_table_rejections = 0usize;
    let mut port_rejections = 0usize;
    for placed in placed {
        if candidates.len() >= config.max_candidates {
            break;
        }
        if validate_truth_table && !candidate_matches_truth_table(&graph, &placed)? {
            truth_table_rejections += 1;
            continue;
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
        candidates.push(LayoutCandidate::from_world(
            module_name.to_owned(),
            world,
            physical_ports,
        )?);
    }
    tracing::info!(
        module = module_name,
        generated = generated_count,
        accepted = candidates.len(),
        truth_table_rejections,
        port_rejections,
        "local candidate validation completed"
    );
    Ok(candidates)
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{BlockKind, Direction};

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
