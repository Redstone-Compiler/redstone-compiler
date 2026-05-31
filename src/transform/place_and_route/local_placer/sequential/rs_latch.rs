use std::collections::HashSet;

use itertools::{iproduct, Itertools};

use super::macro_routes::generate_sequential_macro_routes;
use super::*;

const RS_LATCH_PAIR_SELECTION_SAMPLE_SCOPE: u64 = 4;
const RS_LATCH_BRANCH_ROUTE_SAMPLE_SCOPE: u64 = 8;
const RS_LATCH_INPUT_BLOCK_PENALTY: usize = 1_000;
const RS_LATCH_INPUT_DISTANCE_BALANCE_WEIGHT: usize = 10;
const RS_LATCH_CLOSE_INPUT_DISTANCE: usize = 2;
const RS_LATCH_CLOSE_INPUT_PENALTY: usize = 100;
const RS_LATCH_MIN_SUPPORT_DISTANCE: usize = 2;
const RS_LATCH_MAX_SUPPORT_DISTANCE: usize = 3;
const RS_LATCH_MIN_TORCH_DISTANCE: usize = 3;
const RS_LATCH_MAX_TORCH_DISTANCE: usize = 5;
const RS_LATCH_MIN_SUPPORT_Z: usize = 1;

#[derive(Debug, Clone)]
pub(in super::super) struct RsLatchGatePlacement {
    pub(in super::super) world: World3D,
    pub(in super::super) q_torch: Position,
    pub(in super::super) q_cobble: Position,
    pub(in super::super) nq_torch: Position,
    pub(in super::super) nq_cobble: Position,
}

pub(in super::super) fn generate_rs_latch_gate_routes(
    config: &LocalPlacerConfig,
    node: GraphNodeRef<'_>,
    sequential: &SequentialPrimitive,
    world: &World3D,
    state: &PlacementState,
) -> PlacerQueue {
    let raw_pairs = generate_rs_latch_not_pairs(world);
    let pairs = select_rs_latch_not_pairs(config, node, sequential, state, raw_pairs.clone());
    let mut routed = Vec::new();
    let limit = take_sampling_limit(config.step_sampling_policy);
    for pair in pairs {
        for placed in route_rs_latch_branches(config, node, sequential, state, pair) {
            let mut state = state.clone();
            state.set_port_position(node.id, "q".to_owned(), placed.q_torch);
            state.set_port_position(node.id, "nq".to_owned(), placed.nq_torch);
            let primary_output = sequential
                .output_ports
                .first()
                .map(String::as_str)
                .unwrap_or("q");
            let primary_position = match primary_output {
                "q" => placed.q_torch,
                "nq" => placed.nq_torch,
                _ => placed.q_torch,
            };
            state.set_node_position(node.id, primary_position);
            routed.push((placed.world, state));
            if limit.is_some_and(|limit| routed.len() >= limit) {
                break;
            }
        }
        if limit.is_some_and(|limit| routed.len() >= limit) {
            break;
        }
    }

    if routed.is_empty() {
        generate_sequential_macro_routes(config, node, sequential, world, state)
    } else {
        routed
    }
}

pub(in super::super) fn select_rs_latch_not_pairs(
    config: &LocalPlacerConfig,
    node: GraphNodeRef<'_>,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    mut pairs: Vec<RsLatchGatePlacement>,
) -> Vec<RsLatchGatePlacement> {
    let Some(set_source) = rs_latch_input_source(node, sequential, state, "s") else {
        return config.step_sampling_policy.sample_with_seed(
            pairs,
            config.sampling_seed(RS_LATCH_PAIR_SELECTION_SAMPLE_SCOPE, 0),
        );
    };
    let Some(reset_source) = rs_latch_input_source(node, sequential, state, "r") else {
        return Vec::new();
    };

    pairs.sort_by_key(|placed| {
        let reset_distance = reset_source.manhattan_distance(&placed.q_cobble);
        let set_distance = set_source.manhattan_distance(&placed.nq_cobble);
        let input_block_penalty = if lies_between(reset_source, placed.q_cobble, placed.q_torch) {
            RS_LATCH_INPUT_BLOCK_PENALTY
        } else {
            0
        } + if lies_between(set_source, placed.nq_cobble, placed.nq_torch)
        {
            RS_LATCH_INPUT_BLOCK_PENALTY
        } else {
            0
        };
        let input_space_penalty = reset_distance.abs_diff(set_distance)
            * RS_LATCH_INPUT_DISTANCE_BALANCE_WEIGHT
            + usize::from(reset_distance < RS_LATCH_CLOSE_INPUT_DISTANCE)
                * RS_LATCH_CLOSE_INPUT_PENALTY
            + usize::from(set_distance < RS_LATCH_CLOSE_INPUT_DISTANCE)
                * RS_LATCH_CLOSE_INPUT_PENALTY;
        reset_distance
            + set_distance
            + placed.q_torch.manhattan_distance(&placed.nq_torch)
            + input_block_penalty
            + input_space_penalty
    });

    match config.step_sampling_policy {
        SamplingPolicy::None => pairs,
        SamplingPolicy::Take(count) | SamplingPolicy::Random(count) => {
            pairs.into_iter().take(count).collect()
        }
    }
}

fn lies_between(start: Position, end: Position, position: Position) -> bool {
    start.manhattan_distance(&position) + position.manhattan_distance(&end)
        == start.manhattan_distance(&end)
        && position != start
        && position != end
}

pub(in super::super) fn route_rs_latch_branches(
    config: &LocalPlacerConfig,
    node: GraphNodeRef<'_>,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    placed: RsLatchGatePlacement,
) -> Vec<RsLatchGatePlacement> {
    let Some(set_source) = rs_latch_input_source(node, sequential, state, "s") else {
        return route_rs_latch_feedback(config, placed);
    };
    let Some(reset_source) = rs_latch_input_source(node, sequential, state, "r") else {
        return Vec::new();
    };

    let q_torch = placed.q_torch;
    let q_cobble = placed.q_cobble;
    let nq_torch = placed.nq_torch;
    let nq_cobble = placed.nq_cobble;
    route_rs_latch_signals_to_cobble(config, placed, nq_torch, reset_source, q_torch, q_cobble)
        .into_iter()
        .flat_map(|placed| {
            route_rs_latch_signals_to_cobble(
                config, placed, q_torch, set_source, nq_torch, nq_cobble,
            )
        })
        .collect()
}

fn route_rs_latch_signals_to_cobble(
    config: &LocalPlacerConfig,
    placed: RsLatchGatePlacement,
    first_source: Position,
    second_source: Position,
    target_torch: Position,
    target_cobble: Position,
) -> Vec<RsLatchGatePlacement> {
    let worlds = generate_two_routes_to_cobble(
        config,
        &placed.world,
        first_source,
        second_source,
        target_torch,
        target_cobble,
    );
    config
        .route_step_sampling_policy
        .sample_with_seed(
            worlds,
            config.sampling_seed(
                RS_LATCH_BRANCH_ROUTE_SAMPLE_SCOPE,
                target_cobble.0 + target_cobble.1,
            ),
        )
        .into_iter()
        .map(|world| RsLatchGatePlacement { world, ..placed })
        .collect()
}

fn generate_two_routes_to_cobble(
    config: &LocalPlacerConfig,
    world: &World3D,
    first_source: Position,
    second_source: Position,
    target_torch: Position,
    target_cobble: Position,
) -> Vec<World3D> {
    let first_routes = generate_routes_to_cobble_with_paths(
        config,
        world,
        first_source,
        target_torch,
        target_cobble,
    );
    let second_routes = generate_routes_to_cobble_with_paths(
        config,
        world,
        second_source,
        target_torch,
        target_cobble,
    );

    let first_then_second = first_routes.iter().flat_map(|(first_world, first_path)| {
        let network = redstone_route_network(first_world, first_path, target_cobble);
        generate_routes_to_cobble_or_network(
            config,
            first_world,
            second_source,
            target_torch,
            target_cobble,
            &network,
        )
        .into_iter()
        .map(|(world, _)| world)
        .collect_vec()
    });

    let second_then_first = second_routes
        .iter()
        .flat_map(|(second_world, second_path)| {
            let network = redstone_route_network(second_world, second_path, target_cobble);
            generate_routes_to_cobble_or_network(
                config,
                second_world,
                first_source,
                target_torch,
                target_cobble,
                &network,
            )
            .into_iter()
            .map(|(world, _)| world)
            .collect_vec()
        });

    first_then_second.chain(second_then_first).collect()
}

fn redstone_route_network(
    world: &World3D,
    path: &[Position],
    target_cobble: Position,
) -> HashSet<Position> {
    path.iter()
        .copied()
        .filter(|&position| {
            world[position].kind.is_redstone()
                && redstone_powers_cobble(world, position, target_cobble)
        })
        .collect()
}

fn rs_latch_input_source(
    node: GraphNodeRef<'_>,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    port: &str,
) -> Option<Position> {
    node.inputs
        .iter()
        .zip(&sequential.input_ports)
        .find_map(|(input_node, input_port)| {
            (input_port == port)
                .then(|| state.node_position(*input_node))
                .flatten()
        })
}

pub(in super::super) fn generate_rs_latch_not_pairs(world: &World3D) -> Vec<RsLatchGatePlacement> {
    let cardinal = [
        Direction::East,
        Direction::West,
        Direction::South,
        Direction::North,
    ];
    let support_positions = iproduct!(
        0..world.size.0,
        0..world.size.1,
        RS_LATCH_MIN_SUPPORT_Z..world.size.2
    )
    .map(|(x, y, z)| Position(x, y, z))
    .collect_vec();
    let mut candidates = Vec::new();

    for &q_direction in &cardinal {
        for q_cobble in &support_positions {
            let Some(q_torch) = q_cobble.walk(q_direction.inverse()) else {
                continue;
            };
            if !world.size.bound_on(q_torch) {
                continue;
            }
            let q_torch_block = Block {
                kind: BlockKind::Torch { is_on: false },
                direction: q_direction,
            };
            let Some((q_world, q_torch, q_cobble)) =
                place_torch_with_cobble(world, q_torch_block, q_torch)
            else {
                continue;
            };

            for &nq_direction in &cardinal {
                for nq_cobble in &support_positions {
                    if q_cobble == *nq_cobble || q_cobble.2 != nq_cobble.2 {
                        continue;
                    }
                    let support_distance = q_cobble.manhattan_distance(nq_cobble);
                    if !(RS_LATCH_MIN_SUPPORT_DISTANCE..=RS_LATCH_MAX_SUPPORT_DISTANCE)
                        .contains(&support_distance)
                        || q_direction.inverse() != nq_direction
                    {
                        continue;
                    }
                    let Some(nq_torch) = nq_cobble.walk(nq_direction.inverse()) else {
                        continue;
                    };
                    if !q_world.size.bound_on(nq_torch)
                        || q_torch == nq_torch
                        || !(RS_LATCH_MIN_TORCH_DISTANCE..=RS_LATCH_MAX_TORCH_DISTANCE)
                            .contains(&q_torch.manhattan_distance(&nq_torch))
                    {
                        continue;
                    }
                    let nq_torch_block = Block {
                        kind: BlockKind::Torch { is_on: false },
                        direction: nq_direction,
                    };
                    let Some((world, nq_torch, nq_cobble)) =
                        place_torch_with_cobble(&q_world, nq_torch_block, nq_torch)
                    else {
                        continue;
                    };
                    candidates.push(RsLatchGatePlacement {
                        world,
                        q_torch,
                        q_cobble,
                        nq_torch,
                        nq_cobble,
                    });
                }
            }
        }
    }

    candidates
}

fn route_rs_latch_feedback(
    config: &LocalPlacerConfig,
    placed: RsLatchGatePlacement,
) -> Vec<RsLatchGatePlacement> {
    generate_routes_to_cobble(
        config,
        &placed.world,
        placed.nq_torch,
        placed.q_torch,
        placed.q_cobble,
    )
    .into_iter()
    .flat_map(|(world, _)| {
        generate_routes_to_cobble(
            config,
            &world,
            placed.q_torch,
            placed.nq_torch,
            placed.nq_cobble,
        )
        .into_iter()
        .map(|(world, _)| RsLatchGatePlacement {
            world,
            q_torch: placed.q_torch,
            q_cobble: placed.q_cobble,
            nq_torch: placed.nq_torch,
            nq_cobble: placed.nq_cobble,
        })
        .collect_vec()
    })
    .collect()
}
