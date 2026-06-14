use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};

use eyre::ContextCompat;

use crate::graph::logic::LogicGraph;
use crate::graph::module::{GraphModule, GraphModulePortTarget, GraphModulePortType};
use crate::graph::GraphNodeKind;
use crate::transform::place_and_route::detailed_router;
use crate::transform::place_and_route::global_pnr::block_kind_is_powered;
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

#[derive(Clone)]
pub struct UnitCandidateConfig {
    pub dim: DimSize,
    pub local_config: LocalPlacerConfig,
    pub input_constraints: LocalPlacerInputConstraints,
    pub max_candidates: usize,
}

impl Default for UnitCandidateConfig {
    fn default() -> Self {
        Self {
            dim: DimSize(16, 16, 6),
            local_config: LocalPlacerConfig::default(),
            input_constraints: LocalPlacerInputConstraints::default(),
            max_candidates: 16,
        }
    }
}

pub fn generate_graph_module_candidates(
    module: &GraphModule,
    config: &UnitCandidateConfig,
) -> eyre::Result<Vec<LayoutCandidate>> {
    generate_graph_module_candidates_with_progress_label(module, config, None)
}

pub fn generate_graph_module_candidates_with_progress_label(
    module: &GraphModule,
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
) -> eyre::Result<Vec<LayoutCandidate>> {
    let graph = module
        .graph
        .clone()
        .context("only graph-backed GraphModule can generate unit layout candidates")?;
    let graph = LogicGraph { graph }.prepare_place()?;
    let placer = LocalPlacer::new(graph.clone(), config.local_config)?;

    let placed = placer.generate_with_outputs_and_input_constraints_progress(
        config.dim,
        None,
        &config.input_constraints,
        progress_label,
    );

    let validate_truth_table = graph_is_combinational(module);
    let mut candidates = Vec::new();
    for placed in placed {
        if candidates.len() >= config.max_candidates {
            break;
        }
        if validate_truth_table && !candidate_matches_truth_table(&graph, &placed)? {
            continue;
        }
        let (world, ports) = switchless_candidate_layout(
            module,
            &config.input_constraints,
            placed.world,
            &placed.inputs,
            &placed.outputs,
        );
        if !candidate_ports_cover_module_ports(module, &ports) {
            continue;
        }
        candidates.push(LayoutCandidate::from_world(
            module.name.clone(),
            world,
            ports,
        )?);
    }
    Ok(candidates)
}

fn candidate_ports_cover_module_ports(module: &GraphModule, ports: &[PhysicalPort]) -> bool {
    module
        .ports
        .iter()
        .all(|module_port| ports.iter().any(|port| port.name == module_port.name))
}

fn graph_is_combinational(module: &GraphModule) -> bool {
    module.graph.as_ref().is_some_and(|graph| {
        graph
            .nodes
            .iter()
            .all(|node| !matches!(node.kind, GraphNodeKind::Sequential(_)))
    })
}

fn candidate_matches_truth_table(
    expected: &LogicGraph,
    placed: &crate::output::PlacedWorld,
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
            if block_kind_is_powered(sim.world()[*output_position].kind) != expected_output[mask] {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

// LocalPlacer는 아직 standalone 회로를 기준으로 switch/output layout을 만든다.
// Global PnR child layout에서는 switch를 제거하고 외부 route가 물릴 수 있는
// module port metadata로 다시 노출한다.
// TODO(high-level): make LocalPlacer produce either standalone layouts with switches
// or child-module layouts with PhysicalPort metadata, instead of rewriting switches here.
fn switchless_candidate_layout(
    module: &GraphModule,
    input_constraints: &LocalPlacerInputConstraints,
    mut world: World3D,
    inputs: &[crate::output::OutputEndpoint],
    outputs: &[crate::output::OutputEndpoint],
) -> (World3D, Vec<PhysicalPort>) {
    let mut ports = Vec::new();
    // Sequential child layout은 내부 feedback/state signal이 외부 route와 직접
    // 합쳐지면 back-power 때문에 latch 상태가 깨질 수 있어서 diode 연결을 요구한다.
    let contains_sequential = module_contains_sequential(module);
    let needs_output_isolation = contains_sequential;
    let needs_input_isolation = contains_sequential;
    let use_direct_input_ports = !contains_sequential && module_input_port_count(module) > 1;
    let preserve_switch_position_inputs = contains_sequential || use_direct_input_ports;
    for port in &module.ports {
        match (&port.port_type, &port.target) {
            (GraphModulePortType::InputNet, GraphModulePortTarget::Node(input_name)) => {
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
            (GraphModulePortType::OutputNet, GraphModulePortTarget::Node(output_name)) => {
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
            _ => {}
        }
    }
    for input in inputs {
        let _ = expose_switchless_input_port(
            &mut world,
            input.position(),
            preserve_switch_position_inputs,
            use_direct_input_ports,
        );
    }
    remove_local_input_switches(&mut world);
    ports.sort_by(|a, b| a.name.cmp(&b.name));
    world.initialize_redstone_states();
    (world, ports)
}

fn module_contains_sequential(module: &GraphModule) -> bool {
    module.graph.as_ref().is_some_and(|graph| {
        graph
            .nodes
            .iter()
            .any(|node| matches!(node.kind, crate::graph::GraphNodeKind::Sequential(_)))
    })
}

fn module_input_port_count(module: &GraphModule) -> usize {
    module
        .ports
        .iter()
        .filter(|port| port.port_type.is_input())
        .count()
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::BlockKind;
    use crate::world::block::Direction;

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
