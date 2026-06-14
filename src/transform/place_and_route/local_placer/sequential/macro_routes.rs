use std::collections::HashMap;

use itertools::{iproduct, Itertools};

use super::*;

// TODO: Deprecate this hardcoded sequential macro fallback once RS latch and D latch
// gate-level searched placement is reliable enough without prebuilt macro candidates.
pub(in super::super) fn generate_sequential_macro_routes(
    config: &LocalPlacerConfig,
    node: GraphNodeRef<'_>,
    sequential: &SequentialPrimitive,
    world: &World3D,
    state: &PlacementState,
) -> PlacerQueue {
    SequentialMacro::candidates(sequential)
        .into_iter()
        .flat_map(|candidate| {
            iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
                .map(|(x, y, z)| Position(x, y, z))
                .filter_map(move |anchor| place_sequential_macro(world, &candidate, anchor))
                .flat_map(|placed| {
                    route_sequential_inputs(config, node, sequential, state, &placed)
                        .into_iter()
                        .map(|world| {
                            let mut state = state.clone();
                            for (output_port, position) in &placed.output_ports {
                                state.set_port_position(node.id, output_port.to_owned(), *position);
                            }
                            let primary_output = sequential
                                .output_ports
                                .first()
                                .unwrap_or(&placed.primary_output_port);
                            let primary_position = placed.output_ports[primary_output];
                            state.set_node_position(node.id, primary_position);
                            (world, state)
                        })
                        .collect_vec()
                })
                .collect_vec()
        })
        .collect()
}

#[derive(Debug, Clone)]
pub(in super::super) struct PlacedSequentialMacro {
    world: World3D,
    input_ports: HashMap<String, Position>,
    output_ports: HashMap<String, Position>,
    primary_output_port: String,
}

pub(in super::super) fn place_sequential_macro(
    world: &World3D,
    candidate: &SequentialMacro,
    anchor: Position,
) -> Option<PlacedSequentialMacro> {
    let mut new_world = world.clone();
    for (relative, block) in candidate.world.iter_block() {
        let position = translate_position(anchor, relative)?;
        if !world.size.bound_on(position) || !world[position].kind.is_air() {
            return None;
        }
        new_world[position] = block;
    }
    new_world.initialize_redstone_states();

    Some(PlacedSequentialMacro {
        world: new_world,
        input_ports: translate_ports(anchor, &candidate.input_ports)?,
        output_ports: translate_ports(anchor, &candidate.output_ports)?,
        primary_output_port: candidate.primary_output_port.clone(),
    })
}

pub(in super::super) fn route_sequential_inputs(
    config: &LocalPlacerConfig,
    node: GraphNodeRef<'_>,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    placed: &PlacedSequentialMacro,
) -> Vec<World3D> {
    let mut worlds = vec![placed.world.clone()];
    for (input_node, input_port) in node.inputs.iter().zip(&sequential.input_ports) {
        let Some(source) = state.node_position(*input_node) else {
            return Vec::new();
        };
        let Some(&target) = placed.input_ports.get(input_port) else {
            return Vec::new();
        };

        worlds = worlds
            .into_iter()
            .flat_map(|world| {
                generate_or_routes(config, &world, source, target)
                    .routes
                    .into_iter()
                    .map(|(world, _)| world)
                    .collect_vec()
            })
            .collect_vec();
    }
    worlds
}

fn translate_ports(
    anchor: Position,
    ports: &HashMap<String, Position>,
) -> Option<HashMap<String, Position>> {
    ports
        .iter()
        .map(|(port, position)| Some((port.clone(), translate_position(anchor, *position)?)))
        .collect()
}

fn translate_position(anchor: Position, relative: Position) -> Option<Position> {
    Some(Position(
        anchor.0.checked_add(relative.0)?,
        anchor.1.checked_add(relative.1)?,
        anchor.2.checked_add(relative.2)?,
    ))
}
