use eyre::ContextCompat;
use itertools::Itertools;

use crate::graph::logic::LogicGraph;
use crate::graph::module::{GraphModule, GraphModulePortTarget, GraphModulePortType};
use crate::transform::place_and_route::detailed_router;
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPort, PhysicalPortDirection,
};
use crate::transform::place_and_route::local_placer::{
    LocalPlacer, LocalPlacerConfig, LocalPlacerInputConstraints,
};
use crate::world::block::Block;
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

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
    let graph = module
        .graph
        .clone()
        .context("only graph-backed GraphModule can generate unit layout candidates")?;
    let graph = LogicGraph { graph }.prepare_place()?;
    let placer = LocalPlacer::new(graph, config.local_config)?;

    placer
        .generate_with_outputs_and_input_constraints(config.dim, None, &config.input_constraints)
        .into_iter()
        .take(config.max_candidates)
        .map(|placed| {
            let (world, ports) = switchless_candidate_layout(
                module,
                &config.input_constraints,
                placed.world,
                &placed.inputs,
                &placed.outputs,
            );
            LayoutCandidate::from_world(module.name.clone(), world, ports)
        })
        .collect()
}

fn switchless_candidate_layout(
    module: &GraphModule,
    input_constraints: &LocalPlacerInputConstraints,
    mut world: World3D,
    inputs: &[crate::output::OutputEndpoint],
    outputs: &[crate::output::OutputEndpoint],
) -> (World3D, Vec<PhysicalPort>) {
    let mut ports = module
        .ports
        .iter()
        .filter_map(|port| match (&port.port_type, &port.target) {
            (GraphModulePortType::InputNet, GraphModulePortTarget::Node(input_name)) => inputs
                .iter()
                .find(|input| input.name == *input_name)
                .map(|input| input.position())
                .or_else(|| {
                    input_constraints
                        .positions_for_input_name(input_name)
                        .and_then(|positions| positions.into_iter().next())
                })
                .and_then(|position| {
                    expose_switchless_input_port(&mut world, position).map(|position| {
                        PhysicalPort {
                            name: port.name.clone(),
                            direction: PhysicalPortDirection::Input,
                            position,
                        }
                    })
                }),
            (GraphModulePortType::OutputNet, GraphModulePortTarget::Node(output_name)) => outputs
                .iter()
                .find(|output| output.name == *output_name)
                .map(|output| PhysicalPort {
                    name: port.name.clone(),
                    direction: PhysicalPortDirection::Output,
                    position: expose_routeable_output_port(&world, output.position()),
                }),
            _ => None,
        })
        .collect_vec();
    for input in inputs {
        let _ = expose_switchless_input_port(&mut world, input.position());
    }
    remove_local_input_switches(&mut world);
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

fn expose_routeable_output_port(world: &World3D, output_position: Position) -> Position {
    if !world.size.bound_on(output_position)
        || (!world[output_position].kind.is_torch()
            && !world[output_position].kind.is_switch()
            && !world[output_position].kind.is_repeater())
    {
        return output_position;
    }

    world
        .iter_block()
        .into_iter()
        .filter(|(position, block)| {
            block.kind.is_redstone()
                && detailed_router::target_powers_redstone(world, output_position, *position)
        })
        .map(|(position, _)| position)
        .min_by_key(|position| {
            (
                output_position.manhattan_distance(position),
                position.0,
                position.1,
                position.2,
            )
        })
        .unwrap_or(output_position)
}

fn expose_switchless_input_port(world: &mut World3D, input_position: Position) -> Option<Position> {
    if !world.size.bound_on(input_position) {
        return None;
    }
    if !world[input_position].kind.is_switch() {
        return Some(input_position);
    }

    let switch_target = input_position.walk(world[input_position].direction);
    let port_position = switch_target
        .filter(|position| world.size.bound_on(*position) && world[*position].kind.is_cobble())
        .unwrap_or_else(|| expose_routeable_output_port(world, input_position));
    if port_position == input_position {
        return None;
    }
    world[input_position] = Block::default();
    Some(port_position)
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
