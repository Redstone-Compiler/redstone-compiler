use super::*;

pub(super) fn supports_sequential_primitive(sequential: &SequentialPrimitive) -> bool {
    matches!(
        sequential.sequential_type,
        SequentialType::RsLatch | SequentialType::DLatch
    ) && (sequential.rs_latch_core().is_some()
        || !SequentialMacro::candidates(sequential).is_empty())
}

#[derive(Debug, Clone)]
pub(super) struct RsLatchGatePlacement {
    pub(super) world: World3D,
    pub(super) q_torch: Position,
    pub(super) q_cobble: Position,
    pub(super) nq_torch: Position,
    pub(super) nq_cobble: Position,
}

pub(super) const D_LATCH_SET_NODE_ID: GraphNodeId = usize::MAX - 1;
pub(super) const D_LATCH_RESET_NODE_ID: GraphNodeId = usize::MAX;
pub(super) const D_LATCH_NOT_D_NODE_ID: GraphNodeId = usize::MAX - 2;
pub(super) const D_LATCH_SET_NOT_EN_NODE_ID: GraphNodeId = usize::MAX - 3;
pub(super) const D_LATCH_SET_OR_NODE_ID: GraphNodeId = usize::MAX - 4;
pub(super) const D_LATCH_RESET_OR_NODE_ID: GraphNodeId = usize::MAX - 5;

pub(super) fn generate_d_latch_gate_routes(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    world: &World3D,
    state: &PlacementState,
) -> PlacerQueue {
    let Some(data_input) = sequential_input(node, sequential, state, "d") else {
        return Vec::new();
    };
    let Some(enable_input) = sequential_input(node, sequential, state, "en") else {
        return Vec::new();
    };
    let rs_sequential = SequentialPrimitive::new(
        SequentialType::RsLatch,
        vec!["s".to_owned(), "r".to_owned()],
        sequential.output_ports.clone(),
    );
    let rs_node = GraphNode {
        id: node.id,
        kind: GraphNodeKind::Sequential(rs_sequential.clone()),
        inputs: vec![D_LATCH_SET_NODE_ID, D_LATCH_RESET_NODE_ID],
        ..Default::default()
    };

    let queue = vec![(world.clone(), state.clone())];
    let queue = generate_not_step(config, 0, queue, data_input.node_id, D_LATCH_NOT_D_NODE_ID);
    let queue = generate_not_step(
        config,
        1,
        queue,
        enable_input.node_id,
        D_LATCH_SET_NOT_EN_NODE_ID,
    );
    let queue = generate_two_or_step(
        config,
        2,
        queue,
        (
            D_LATCH_NOT_D_NODE_ID,
            D_LATCH_SET_NOT_EN_NODE_ID,
            D_LATCH_SET_OR_NODE_ID,
        ),
        (
            data_input.node_id,
            D_LATCH_SET_NOT_EN_NODE_ID,
            D_LATCH_RESET_OR_NODE_ID,
        ),
    );
    let queue = generate_not_step(
        config,
        3,
        queue,
        D_LATCH_SET_OR_NODE_ID,
        D_LATCH_SET_NODE_ID,
    );
    let queue = generate_not_step(
        config,
        4,
        queue,
        D_LATCH_RESET_OR_NODE_ID,
        D_LATCH_RESET_NODE_ID,
    );
    let queue = queue
        .into_iter()
        .filter(|(world, state)| {
            d_latch_input_gates_behave(world, data_input.node_id, enable_input.node_id, state)
        })
        .collect_vec();
    let queue = SamplingPolicy::Random(64).sample_with_seed(queue, config.sampling_seed(7, 6));

    let mut routed = Vec::new();
    let limit = take_sampling_limit(config.step_sampling_policy);
    let rs_config = LocalPlacerConfig {
        step_sampling_policy: SamplingPolicy::Random(32),
        route_step_sampling_policy: SamplingPolicy::Random(32),
        not_route_step_sampling_policy: SamplingPolicy::Random(32),
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_route_step: 8,
        ..*config
    };
    for (world, state) in queue {
        for (world, state) in
            generate_rs_latch_gate_routes(&rs_config, &rs_node, &rs_sequential, &world, &state)
        {
            let Some(q) = state.port_position(node.id, "q") else {
                continue;
            };
            let Some(nq) = state.port_position(node.id, "nq") else {
                continue;
            };
            if std::panic::catch_unwind(|| {
                d_latch_candidate_behaves(
                    &world,
                    data_input.node_id,
                    enable_input.node_id,
                    &state,
                    q,
                    nq,
                )
            })
            .unwrap_or(false)
            {
                routed.push((world, state));
            }
        }
        if limit.is_some_and(|limit| routed.len() >= limit) {
            break;
        }
    }
    sample_d_latch_step(config, 6, routed)
}

fn d_latch_input_gates_behave(
    world: &World3D,
    data_node_id: GraphNodeId,
    enable_node_id: GraphNodeId,
    state: &PlacementState,
) -> bool {
    let Some(data) = state.node_position(data_node_id) else {
        return false;
    };
    let Some(enable) = state.node_position(enable_node_id) else {
        return false;
    };
    let Some(set) = state.node_position(D_LATCH_SET_NODE_ID) else {
        return false;
    };
    let Some(reset) = state.node_position(D_LATCH_RESET_NODE_ID) else {
        return false;
    };

    [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .all(|(data_value, enable_value)| {
            d_latch_input_gate_values(world, data, enable, set, reset, data_value, enable_value)
                .is_some_and(|(set_value, reset_value)| {
                    set_value == (data_value && enable_value)
                        && reset_value == (!data_value && enable_value)
                })
        })
}

fn d_latch_input_gate_values(
    world: &World3D,
    data: Position,
    enable: Position,
    set: Position,
    reset: Position,
    data_value: bool,
    enable_value: bool,
) -> Option<(bool, bool)> {
    let mut world = pad_world_top_for_simulation(world, 2);
    world[data].kind = BlockKind::Switch { is_on: data_value };
    world[enable].kind = BlockKind::Switch {
        is_on: enable_value,
    };
    world.initialize_redstone_states();
    let world = World::from(&world);
    let sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0).ok()?;
    let set_value = match sim.world()[set].kind {
        BlockKind::Torch { is_on } => is_on,
        _ => return None,
    };
    let reset_value = match sim.world()[reset].kind {
        BlockKind::Torch { is_on } => is_on,
        _ => return None,
    };
    Some((set_value, reset_value))
}

fn d_latch_candidate_behaves(
    world: &World3D,
    data_node_id: GraphNodeId,
    enable_node_id: GraphNodeId,
    state: &PlacementState,
    q: Position,
    nq: Position,
) -> bool {
    let Some(data) = state.node_position(data_node_id) else {
        return false;
    };
    let Some(enable) = state.node_position(enable_node_id) else {
        return false;
    };

    let mut reset_world = pad_world_top_for_simulation(world, 2);
    reset_world[enable].kind = BlockKind::Switch { is_on: true };
    reset_world[data].kind = BlockKind::Switch { is_on: false };
    reset_world.initialize_redstone_states();
    let world = World::from(&reset_world);
    let Ok(mut sim) = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0) else {
        return false;
    };

    if torch_is_on_in_world3d(sim.world(), q) || !torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, false)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(data, true)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if torch_is_on_in_world3d(sim.world(), q) || !torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, true)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if !torch_is_on_in_world3d(sim.world(), q) || torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, false)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(data, false)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if !torch_is_on_in_world3d(sim.world(), q) || torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, true)], 64, 20_000)
        .is_err()
    {
        return false;
    }

    !torch_is_on_in_world3d(sim.world(), q) && torch_is_on_in_world3d(sim.world(), nq)
}

fn torch_is_on_in_world3d(world: &World3D, position: Position) -> bool {
    matches!(world[position].kind, BlockKind::Torch { is_on: true })
}

fn pad_world_top_for_simulation(world: &World3D, extra_z: usize) -> World3D {
    let mut padded = World3D::new(DimSize(world.size.0, world.size.1, world.size.2 + extra_z));
    for (position, mut block) in world.iter_block() {
        if block.kind.is_cobble() {
            block.kind = BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            };
        }
        padded[position] = block;
    }
    padded
}

#[derive(Clone, Copy)]
struct SequentialInput {
    node_id: GraphNodeId,
}

fn sequential_input(
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    port: &str,
) -> Option<SequentialInput> {
    node.inputs
        .iter()
        .zip(&sequential.input_ports)
        .find_map(|(input_node, input_port)| {
            (input_port == port && state.node_position(*input_node).is_some()).then_some(
                SequentialInput {
                    node_id: *input_node,
                },
            )
        })
}

fn generate_not_step(
    config: &LocalPlacerConfig,
    step: usize,
    queue: PlacerQueue,
    input_node_id: GraphNodeId,
    output_node_id: GraphNodeId,
) -> PlacerQueue {
    let candidates: PlacerQueue = queue
        .into_iter()
        .flat_map(|(world, state)| {
            let source = state[&input_node_id];
            let route_config = if world[source].kind.is_switch() {
                Cow::Owned(LocalPlacerConfig {
                    not_route_strategy: NotRouteStrategy::DirectOnly,
                    ..*config
                })
            } else {
                Cow::Borrowed(config)
            };
            not_node_kind()
                .into_iter()
                .flat_map(move |kind| {
                    let state = state.clone();
                    generate_place_and_routes(&route_config, &world, source, kind)
                        .into_iter()
                        .map(move |(world, position)| {
                            let mut state = state.clone();
                            state.set_node_position(output_node_id, position);
                            (world, state)
                        })
                })
                .collect_vec()
        })
        .collect();
    sample_d_latch_step(config, step, candidates)
}

fn generate_two_or_step(
    config: &LocalPlacerConfig,
    step: usize,
    queue: PlacerQueue,
    first: (GraphNodeId, GraphNodeId, GraphNodeId),
    second: (GraphNodeId, GraphNodeId, GraphNodeId),
) -> PlacerQueue {
    let candidates = queue
        .into_iter()
        .flat_map(|(base_world, state)| {
            let first_routes =
                generate_or_routes(config, &base_world, state[&first.0], state[&first.1]).routes;
            let second_routes =
                generate_or_routes(config, &base_world, state[&second.0], state[&second.1]).routes;
            first_routes
                .into_iter()
                .cartesian_product(second_routes)
                .filter_map(
                    |((first_world, first_positions), (second_world, second_positions))| {
                        let world =
                            merge_independent_route_worlds(&base_world, first_world, second_world)?;
                        let first_position = first_positions.last().copied()?;
                        let second_position = second_positions.last().copied()?;
                        Some({
                            let mut state = state.clone();
                            state.set_node_position(first.2, first_position);
                            state.set_node_position(second.2, second_position);
                            (world, state)
                        })
                    },
                )
                .collect_vec()
        })
        .collect();
    sample_d_latch_step(config, step, candidates)
}

fn merge_independent_route_worlds(
    base: &World3D,
    first: World3D,
    second: World3D,
) -> Option<World3D> {
    let mut merged = first.clone();
    let mut first_redstone = HashSet::new();
    let mut second_redstone = HashSet::new();
    for (x, y, z) in iproduct!(0..base.size.0, 0..base.size.1, 0..base.size.2) {
        let position = Position(x, y, z);
        if first[position] != base[position] && first[position].kind.is_redstone() {
            first_redstone.insert(position);
        }
        if second[position] != base[position] && second[position].kind.is_redstone() {
            second_redstone.insert(position);
        }
        if second[position] == base[position] {
            continue;
        }
        if first[position] != base[position] {
            return None;
        }
        merged[position] = second[position];
        if merged[position].kind.is_redstone() {
            merged.update_redstone_states(position);
        }
    }
    if redstone_networks_touch(&merged, &first_redstone, &second_redstone) {
        return None;
    }
    Some(merged)
}

fn redstone_networks_touch(
    world: &World3D,
    first: &HashSet<Position>,
    second: &HashSet<Position>,
) -> bool {
    first.iter().any(|&position| {
        PlacedNode::new(position, world[position])
            .propagation_bound(Some(world))
            .into_iter()
            .any(|bound| second.contains(&bound.position()))
    }) || second.iter().any(|&position| {
        PlacedNode::new(position, world[position])
            .propagation_bound(Some(world))
            .into_iter()
            .any(|bound| first.contains(&bound.position()))
    })
}

fn sample_d_latch_step(
    config: &LocalPlacerConfig,
    step: usize,
    candidates: PlacerQueue,
) -> PlacerQueue {
    config
        .step_sampling_policy
        .sample_with_seed(candidates, config.sampling_seed(7, step))
}

fn take_sampling_limit(policy: SamplingPolicy) -> Option<usize> {
    match policy {
        SamplingPolicy::Take(count) => Some(count),
        SamplingPolicy::None | SamplingPolicy::Random(_) => None,
    }
}

pub(super) fn generate_rs_latch_gate_routes(
    config: &LocalPlacerConfig,
    node: &GraphNode,
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

pub(super) fn select_rs_latch_not_pairs(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    mut pairs: Vec<RsLatchGatePlacement>,
) -> Vec<RsLatchGatePlacement> {
    let Some(set_source) = rs_latch_input_source(node, sequential, state, "s") else {
        return config
            .step_sampling_policy
            .sample_with_seed(pairs, config.sampling_seed(4, 0));
    };
    let Some(reset_source) = rs_latch_input_source(node, sequential, state, "r") else {
        return Vec::new();
    };

    pairs.sort_by_key(|placed| {
        let reset_distance = reset_source.manhattan_distance(&placed.q_cobble);
        let set_distance = set_source.manhattan_distance(&placed.nq_cobble);
        let input_block_penalty = if lies_between(reset_source, placed.q_cobble, placed.q_torch) {
            1_000
        } else {
            0
        } + if lies_between(set_source, placed.nq_cobble, placed.nq_torch)
        {
            1_000
        } else {
            0
        };
        let input_space_penalty = (reset_distance.abs_diff(set_distance) * 10)
            + usize::from(reset_distance < 2) * 100
            + usize::from(set_distance < 2) * 100;
        let internal_interference_penalty =
            d_latch_internal_interference_penalty(state, placed) * 2_000;

        reset_distance
            + set_distance
            + placed.q_torch.manhattan_distance(&placed.nq_torch)
            + input_block_penalty
            + input_space_penalty
            + internal_interference_penalty
    });

    match config.step_sampling_policy {
        SamplingPolicy::None => pairs,
        SamplingPolicy::Take(count) | SamplingPolicy::Random(count) => {
            pairs.into_iter().take(count).collect()
        }
    }
}

fn d_latch_internal_interference_penalty(
    state: &PlacementState,
    placed: &RsLatchGatePlacement,
) -> usize {
    let protected_nodes = [
        D_LATCH_NOT_D_NODE_ID,
        D_LATCH_SET_NOT_EN_NODE_ID,
        D_LATCH_SET_OR_NODE_ID,
        D_LATCH_RESET_OR_NODE_ID,
    ];
    let latch_positions = [
        placed.q_torch,
        placed.q_cobble,
        placed.nq_torch,
        placed.nq_cobble,
    ];

    protected_nodes
        .into_iter()
        .filter_map(|node_id| state.node_position(node_id))
        .flat_map(|protected| {
            latch_positions
                .into_iter()
                .map(move |latch| 4usize.saturating_sub(protected.manhattan_distance(&latch)))
        })
        .sum()
}

fn lies_between(start: Position, end: Position, position: Position) -> bool {
    start.manhattan_distance(&position) + position.manhattan_distance(&end)
        == start.manhattan_distance(&end)
        && position != start
        && position != end
}

pub(super) fn route_rs_latch_branches(
    config: &LocalPlacerConfig,
    node: &GraphNode,
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
            config.sampling_seed(8, target_cobble.0 + target_cobble.1),
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
    node: &GraphNode,
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

pub(super) fn generate_rs_latch_not_pairs(world: &World3D) -> Vec<RsLatchGatePlacement> {
    let cardinal = [
        Direction::East,
        Direction::West,
        Direction::South,
        Direction::North,
    ];
    let support_positions = iproduct!(0..world.size.0, 0..world.size.1, 1..world.size.2)
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
                    if !(2..=3).contains(&support_distance) || q_direction.inverse() != nq_direction
                    {
                        continue;
                    }
                    let Some(nq_torch) = nq_cobble.walk(nq_direction.inverse()) else {
                        continue;
                    };
                    if !q_world.size.bound_on(nq_torch)
                        || q_torch == nq_torch
                        || !(3..=5).contains(&q_torch.manhattan_distance(&nq_torch))
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

pub(super) fn generate_sequential_macro_routes(
    config: &LocalPlacerConfig,
    node: &GraphNode,
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
pub(super) struct PlacedSequentialMacro {
    world: World3D,
    input_ports: HashMap<String, Position>,
    output_ports: HashMap<String, Position>,
    primary_output_port: String,
}

pub(super) fn place_sequential_macro(
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

pub(super) fn route_sequential_inputs(
    config: &LocalPlacerConfig,
    node: &GraphNode,
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
