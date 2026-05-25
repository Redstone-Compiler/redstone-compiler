use super::*;

const OR_ROUTE_STEP_SAMPLE_SCOPE: u64 = 3;
const NOT_ROUTE_STEP_SAMPLE_SCOPE: u64 = 5;

pub(super) fn input_node_kind() -> Vec<BlockKind> {
    vec![
        BlockKind::Switch { is_on: false },
        // BlockKind::Cobble {
        //     on_count: 0,
        //     on_base_count: 0,
        // },
        // BlockKind::Redstone {
        //     on_count: 0,
        //     state: RedstoneState::None as usize,
        //     strength: 0,
        // },
        // BlockKind::RedstoneBlock,
    ]
}

pub(super) fn not_node_kind() -> Vec<BlockKind> {
    vec![BlockKind::Torch { is_on: false }]
}

pub(super) fn place_node(world: &mut World3D, node: PlacedNode) {
    if world[node.position] == node.block {
        // TODO: Relax for no-op
        assert!(world[node.position].kind.is_cobble());
        return;
    }

    world[node.position] = node.block;
    if node.block.kind.is_redstone() {
        world.update_redstone_states(node.position);
    }
}

pub(super) fn generate_inputs(
    config: &LocalPlacerConfig,
    world: &World3D,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    let mut input_strategy = Direction::iter_direction_without_top()
        .map(|direction| Block { kind, direction })
        .collect_vec();
    if config.greedy_input_generation {
        input_strategy = input_strategy.into_iter().take(1).collect();
    }

    let place_strategy = match config.input_placement_strategy {
        InputPlacementStrategy::Boundary => iproduct!(0..1, 0..world.size.1, 0..world.size.2)
            .chain(iproduct!(0..world.size.0, 0..1, 0..world.size.2))
            .map(|(x, y, z)| Position(x, y, z))
            .unique()
            .collect_vec(),
        InputPlacementStrategy::Anywhere => {
            iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
                .map(|(x, y, z)| Position(x, y, z))
                .collect_vec()
        }
    };

    let mut generated = input_strategy
        .into_iter()
        .cartesian_product(place_strategy)
        // Place Input Node
        .flat_map(|(block, position)| {
            let placed_node = PlacedNode { position, block };
            if placed_node.has_conflict(world, &Default::default()) {
                return None;
            }

            let mut new_world = world.clone();
            place_node(&mut new_world, placed_node);
            Some((new_world, position))
        })
        .collect_vec();

    if let Some(limit) = config.input_candidate_limit {
        generated.truncate(limit);
    }
    generated
}

pub(super) fn generate_place_and_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    start: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    match kind {
        BlockKind::Torch { .. } => generate_torch_place_and_routes(config, world, start, kind),
        _ => unimplemented!(),
    }
}

pub(super) fn generate_output_routes(
    world: &World3D,
    source: Position,
) -> Vec<(World3D, Position)> {
    let source_node = PlacedNode::new(source, world[source]);
    let forbidden_cobble = torch_source_support(world, source);

    source_node
        .propagation_bound(Some(world))
        .into_iter()
        .filter(|bound| world.size.bound_on(bound.position()))
        .filter_map(|bound| place_output_redstone(world, bound, source))
        .filter(|(world, position)| {
            !route_powers_forbidden_cobble(world, &[source, *position], forbidden_cobble)
        })
        .map(|(world, position)| (source.manhattan_distance(&position), world, position))
        .sorted_by_key(|(cost, _, _)| *cost)
        .map(|(_, world, position)| (world, position))
        .collect()
}

pub(super) fn generate_torch_place_and_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    let torch_strategy =
        Direction::iter_direction_without_top().map(|direction| Block { kind, direction });

    let place_strategy = iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
        .map(|(x, y, z)| Position(x, y, z))
        // start에서 최소 한 칸 떨어진 곳에 배치한다.
        .filter(|pos| source.manhattan_distance(pos) > 1)
        .filter(|pos| {
            !config.route_torch_directly
                || matches!(
                    config.torch_placement_strategy,
                    TorchPlacementStrategy::AnywhereNonAdjacent
                )
                || source.manhattan_distance(pos) == 2
        });

    torch_strategy
        .cartesian_product(place_strategy)
        // 1. Place Torch and Cobble
        .flat_map(|(torch, torch_pos)| place_torch_with_cobble(world, torch, torch_pos))
        // 2. Route Source with Torch Place Target Position
        .flat_map(|(world, torch_pos, cobble_pos)| {
            generate_routes_to_cobble(config, &world, source, torch_pos, cobble_pos)
                .into_iter()
                .map(|(world, _)| (world, torch_pos))
                .collect_vec()
        })
        .collect()
}

fn place_output_redstone(
    world: &World3D,
    bound: PlaceBound,
    source: Position,
) -> Option<(World3D, Position)> {
    let redstone_pos = bound.position();
    let cobble_pos = redstone_pos.down()?;
    let cobble_except = if world[source].kind.is_torch() {
        vec![cobble_pos, source]
    } else {
        Vec::new()
    };
    let cobble_node = try_generate_cobble_node(world, cobble_pos, &cobble_except)?;

    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);

    let bound_back_pos = redstone_pos.walk(bound.direction())?;
    let redstone_node = PlacedNode::new_redstone(redstone_pos);
    let except = [source, bound_back_pos, redstone_pos]
        .into_iter()
        .collect::<HashSet<_>>();
    if redstone_node.has_conflict(&new_world, &except) {
        return None;
    }

    let short_except = [source, redstone_pos].into_iter().collect::<HashSet<_>>();
    if redstone_node.has_short(world, &short_except) {
        return None;
    }

    place_node(&mut new_world, redstone_node);
    new_world.update_redstone_states(source);
    if !target_powers_redstone(&new_world, source, redstone_pos) {
        return None;
    }
    if redstone_node.has_short(&new_world, &short_except) {
        return None;
    }
    if let BlockKind::Torch { .. } = world[source].kind {
        if let Some(source_cobble) = source.walk(world[source].direction) {
            if redstone_powers_cobble(&new_world, redstone_pos, source_cobble) {
                return None;
            }
        }
    }

    Some((new_world, redstone_pos))
}

pub(super) fn place_torch_with_cobble(
    world: &World3D,
    torch: Block,
    torch_pos: Position,
) -> Option<(World3D, Position, Position)> {
    let cobble_pos = torch_pos.walk(torch.direction)?;
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    let torch_node = PlacedNode::new(torch_pos, torch);
    if cobble_node.has_conflict(world, &[cobble_pos].into_iter().collect())
        || torch_node.has_conflict(world, &Default::default())
    {
        return None;
    }

    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);
    place_node(&mut new_world, torch_node);
    Some((new_world, torch_pos, cobble_pos))
}

pub(super) fn generate_routes_to_cobble(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    _torch_pos: Position,
    cobble_pos: Position,
) -> Vec<(World3D, Position)> {
    generate_routes_for_goal(
        config,
        world,
        source,
        RouteGoal::PowerCobble { cobble: cobble_pos },
    )
    .into_iter()
    .sorted_by_key(|candidate| candidate.cost)
    .map(|candidate| (candidate.world, candidate.terminal))
    .collect()
}

// torch를 연결하는 redstone routes를 생성한다.
pub(super) fn generate_routes_to_cobble_with_paths(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    _torch_pos: Position,
    cobble_pos: Position,
) -> Vec<(World3D, Vec<Position>)> {
    generate_routes_for_goal(
        config,
        world,
        source,
        RouteGoal::PowerCobble { cobble: cobble_pos },
    )
    .into_iter()
    .map(|candidate| (candidate.world, candidate.path))
    .collect()
}

pub(super) fn generate_routes_to_cobble_or_network(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    _torch_pos: Position,
    cobble_pos: Position,
    network: &HashSet<Position>,
) -> Vec<(World3D, Vec<Position>)> {
    generate_routes_for_goal(
        config,
        world,
        source,
        RouteGoal::PowerCobbleOrNetwork {
            cobble: cobble_pos,
            network,
        },
    )
    .into_iter()
    .map(|candidate| (candidate.world, candidate.path))
    .collect()
}

#[derive(Clone)]
struct RouteCandidate {
    world: World3D,
    path: Vec<Position>,
    terminal: Position,
    cost: usize,
}

#[derive(Clone, Copy)]
enum RouteGoal<'a> {
    // 특정 cobble 블록을 전원 공급 상태로 만든다.
    PowerCobble {
        cobble: Position,
    },
    // 특정 cobble을 직접 켜거나, 그 cobble을 켤 수 있는 기존 network에 합류한다.
    PowerCobbleOrNetwork {
        cobble: Position,
        network: &'a HashSet<Position>,
    },
    // 특정 위치까지 redstone path를 물리적으로 연결한다.
    ConnectPosition {
        target: Position,
    },
}

impl<'a> RouteGoal<'a> {
    fn powered_cobble(self) -> Option<Position> {
        match self {
            RouteGoal::PowerCobble { cobble } | RouteGoal::PowerCobbleOrNetwork { cobble, .. } => {
                Some(cobble)
            }
            RouteGoal::ConnectPosition { .. } => None,
        }
    }

    fn placement_target(self) -> Position {
        match self {
            RouteGoal::PowerCobble { cobble } | RouteGoal::PowerCobbleOrNetwork { cobble, .. } => {
                cobble
            }
            RouteGoal::ConnectPosition { target } => target,
        }
    }

    fn accepts_direct_bound(self, bound: PlaceBound) -> bool {
        let is_direct_signal = matches!(
            bound.propagation_type(),
            PropagateType::Hard | PropagateType::Soft
        );
        if !is_direct_signal {
            return false;
        }

        match self {
            RouteGoal::PowerCobble { cobble } => bound.position() == cobble,
            RouteGoal::PowerCobbleOrNetwork { cobble, network } => {
                bound.position() == cobble || network.contains(&bound.position())
            }
            RouteGoal::ConnectPosition { target } => bound.position() == target,
        }
    }

    fn accepts_redstone(self, world: &World3D, route: &[Position], redstone: Position) -> bool {
        match self {
            RouteGoal::PowerCobble { cobble } => redstone_powers_cobble(world, redstone, cobble),
            RouteGoal::PowerCobbleOrNetwork { cobble, network } => {
                redstone_powers_cobble(world, redstone, cobble)
                    || (redstone_connects_to_network(world, redstone, network)
                        && route_network_powers_cobble(world, route, network, cobble))
            }
            RouteGoal::ConnectPosition { target } => {
                target_powers_redstone(world, target, redstone)
            }
        }
    }

    fn allowed_short_positions(self) -> Option<&'a HashSet<Position>> {
        match self {
            RouteGoal::PowerCobbleOrNetwork { network, .. } => Some(network),
            RouteGoal::PowerCobble { .. } | RouteGoal::ConnectPosition { .. } => None,
        }
    }
}

fn generate_routes_for_goal(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    goal: RouteGoal,
) -> Vec<RouteCandidate> {
    let source_node = PlacedNode::new(source, world[source]);
    assert!(source_node.is_propagation_target());
    if let Some(cobble) = goal.powered_cobble() {
        assert!(world[cobble].kind.is_cobble());
    }
    let forbidden_cobble = torch_source_support(world, source);

    let mut route_candidates = Vec::new();
    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::DirectOnly | NotRouteStrategy::DirectAndRedstone
    ) {
        for start in source_node.propagation_bound(Some(world)) {
            if goal.accepts_direct_bound(start)
                && !route_powers_forbidden_cobble(world, &[source], forbidden_cobble)
            {
                route_candidates.push(RouteCandidate {
                    world: world.clone(),
                    path: vec![source],
                    terminal: start.position(),
                    cost: 0,
                });
            }
        }
    }

    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::RedstoneOnly | NotRouteStrategy::DirectAndRedstone
    ) && (source_node.is_diode() || source_node.block.kind.is_redstone())
    {
        let route_config = LocalPlacerConfig {
            max_route_step: config.max_not_route_step,
            route_step_sampling_policy: config.not_route_step_sampling_policy,
            ..*config
        };
        route_candidates.extend(generate_redstone_routes(&route_config, world, source, goal));
    }

    route_candidates
}

fn generate_redstone_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    goal: RouteGoal,
) -> Vec<RouteCandidate> {
    let source_node = PlacedNode::new(source, world[source]);
    let forbidden_cobble = torch_source_support(world, source);
    let mut queue = if source_node.is_diode() {
        generate_routes_to_cobble_init_states(world, source)
    } else if source_node.block.kind.is_redstone() {
        vec![(
            world.clone(),
            vec![source],
            source_node.propagation_bound(Some(world)),
        )]
    } else {
        Vec::new()
    };
    let mut candidates = Vec::new();
    let mut step = 0;

    while step < config.max_route_step && !queue.is_empty() {
        let mut next_queue = Vec::new();

        for (world, prevs, bounds) in queue {
            let prev_pos = prevs.last().copied().unwrap();
            for bound in bounds {
                if !world.size.bound_on(bound.position()) {
                    continue;
                }

                let (new_world, redstone_node) =
                    match place_redstone_for_goal(&world, bound, prev_pos, goal) {
                        PlaceRedstoneResult::Placed(new_world, redstone_node) => {
                            (new_world, redstone_node)
                        }
                        PlaceRedstoneResult::Rejected(_) => continue,
                    };

                let new_prevs = prevs
                    .iter()
                    .copied()
                    .chain([redstone_node.position])
                    .collect_vec();
                if route_powers_forbidden_cobble(&new_world, &new_prevs, forbidden_cobble) {
                    continue;
                }

                if goal.accepts_redstone(&new_world, &new_prevs, redstone_node.position) {
                    let terminal = redstone_node.position;
                    candidates.push(RouteCandidate {
                        world: new_world,
                        path: new_prevs,
                        terminal,
                        cost: prevs.len() + 1,
                    });
                } else {
                    let nexts = redstone_node.propagation_bound(Some(&new_world));
                    next_queue.push((new_world, new_prevs, nexts));
                }
            }
        }

        queue = config.route_step_sampling_policy.sample_with_seed(
            next_queue,
            config.sampling_seed(NOT_ROUTE_STEP_SAMPLE_SCOPE, step),
        );
        step += 1;
    }

    candidates
}

pub(super) fn torch_source_support(world: &World3D, source: Position) -> Option<Position> {
    matches!(world[source].kind, BlockKind::Torch { .. })
        .then(|| source.walk(world[source].direction))
        .flatten()
}

pub(super) fn route_powers_forbidden_cobble(
    world: &World3D,
    route: &[Position],
    forbidden_cobble: Option<Position>,
) -> bool {
    let Some(forbidden_cobble) = forbidden_cobble else {
        return false;
    };
    if world.size.bound_on(forbidden_cobble.up()) && world[forbidden_cobble.up()].kind.is_cobble() {
        return true;
    }
    route
        .iter()
        .copied()
        .filter(|&position| world[position].kind.is_redstone())
        .any(|position| redstone_powers_cobble(world, position, forbidden_cobble))
}

pub(super) fn redstone_powers_cobble(
    world: &World3D,
    redstone: Position,
    cobble: Position,
) -> bool {
    world[cobble].kind.is_cobble()
        && PlacedNode::new(redstone, world[redstone])
            .propagation_bound(Some(world))
            .into_iter()
            .any(|bound| bound.position() == cobble)
}

pub(super) fn redstone_connects_to_network(
    world: &World3D,
    source: Position,
    network: &HashSet<Position>,
) -> bool {
    let source_node = PlacedNode::new(source, world[source]);
    source_node
        .propagation_bound(Some(world))
        .into_iter()
        .any(|bound| network.contains(&bound.position()))
}

pub(super) fn target_powers_redstone(
    world: &World3D,
    target: Position,
    redstone: Position,
) -> bool {
    let target_node = PlacedNode::new(target, world[target]);
    target_node
        .propagation_bound(Some(world))
        .into_iter()
        .filter(|bound| bound.is_bound_on(world))
        .any(|bound| {
            bound.position() == redstone
                || bound
                    .propagate_to(world)
                    .into_iter()
                    .any(|(_, position)| position == redstone)
        })
}

pub(super) fn route_network_powers_cobble(
    world: &World3D,
    route: &[Position],
    network: &HashSet<Position>,
    cobble_pos: Position,
) -> bool {
    route
        .iter()
        .copied()
        .chain(network.iter().copied())
        .filter(|&position| world.size.bound_on(position) && world[position].kind.is_redstone())
        .any(|position| redstone_powers_cobble(world, position, cobble_pos))
}

pub(super) fn generate_routes_to_cobble_init_states(
    world: &World3D,
    source: Position,
) -> Vec<(World3D, Vec<Position>, Vec<PlaceBound>)> {
    let source_node = PlacedNode::new(source, world[source]);
    let mut states = Vec::new();

    for input_top_cobble in [false, true] {
        let mut new_world = Cow::Borrowed(world);

        if input_top_cobble {
            let Some(cobble_node) = try_generate_cobble_node(world, source.up(), &[source]) else {
                continue;
            };
            place_node(new_world.to_mut(), cobble_node);
        }

        let bounds = if input_top_cobble {
            source
                .up()
                .cardinal()
                .into_iter()
                // Cobble 위쪽에 redstone을 배치하는 케이스
                .chain(Some(source.up().up()))
                .map(|pos| PlaceBound(PropagateType::Soft, pos, pos.diff(source)))
                .collect_vec()
        } else {
            source_node.propagation_bound(Some(world))
        };

        states.push((new_world.into_owned(), vec![source], bounds));
    }

    states
}

pub(super) fn generate_or_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    from: Position,
    to: Position,
) -> RouteResult {
    let (mut queue, mut debug) = generate_or_routes_init_states(world, from, to);
    let goal = RouteGoal::ConnectPosition { target: to };
    debug.route_calls = 1;
    debug.route_samples.push((from, to));
    debug.initial_states = queue.len();

    let mut candidates = Vec::new();
    let mut step = 0;

    while step < config.max_route_step && !queue.is_empty() {
        let frontier_len = queue.len();
        let mut next_queue = vec![];
        let mut depth_debug = RouteDepthDebug {
            depth: step,
            frontier_len,
            ..Default::default()
        };

        for (world, prevs, bounds) in queue {
            let prev_pos = prevs.last().copied().unwrap();
            for bound in bounds {
                depth_debug.bounds_checked += 1;
                if !world.size.bound_on(bound.position()) {
                    debug.reject(RouteRejectReason::OutOfBounds);
                    continue;
                }

                let (new_world, redstone_node) =
                    match place_redstone_for_goal(&world, bound, prev_pos, goal) {
                        PlaceRedstoneResult::Placed(new_world, redstone_node) => {
                            (new_world, redstone_node)
                        }
                        PlaceRedstoneResult::Rejected(reason) => {
                            debug.reject(reason);
                            continue;
                        }
                    };

                let new_prevs = prevs
                    .iter()
                    .copied()
                    .chain([redstone_node.position])
                    .collect_vec();

                if goal.accepts_redstone(&new_world, &new_prevs, redstone_node.position) {
                    depth_debug.accepted_routes += 1;
                    candidates.push((new_world, new_prevs));
                } else {
                    let nexts = redstone_node.propagation_bound(Some(&new_world));
                    next_queue.push((new_world, new_prevs, nexts));
                }
            }
        }

        depth_debug.next_frontier_before_sampling = next_queue.len();
        queue = config.route_step_sampling_policy.sample_with_seed(
            next_queue,
            config.sampling_seed(OR_ROUTE_STEP_SAMPLE_SCOPE, step),
        );
        depth_debug.next_frontier_after_sampling = queue.len();
        debug.depths.push(depth_debug);
        step += 1;
    }

    debug.candidates_found = candidates.len();
    RouteResult {
        routes: candidates,
        debug,
    }
}

pub(super) struct RouteResult {
    pub(super) routes: Vec<(World3D, Vec<Position>)>,
    pub(super) debug: RouteDebug,
}

pub(super) fn generate_or_routes_init_states(
    world: &World3D,
    from: Position,
    to: Position,
) -> (Vec<(World3D, Vec<Position>, Vec<PlaceBound>)>, RouteDebug) {
    let from_node = PlacedNode::new(from, world[from]);
    let to_node = PlacedNode::new(to, world[to]);
    assert!(from_node.is_diode() && to_node.is_propagation_target());

    let mut debug = RouteDebug::default();
    let boolean_comb = [false, true];
    let mut states = Vec::new();

    for input_top_cobble in boolean_comb {
        for output_top_cobble in boolean_comb {
            let mut new_world = Cow::Borrowed(world);

            if input_top_cobble {
                let Some(cobble_node) = try_generate_cobble_node(world, from.up(), &[from]) else {
                    debug.reject(RouteRejectReason::InitInputTopCobbleConflict);
                    continue;
                };
                place_node(new_world.to_mut(), cobble_node);
            }

            if output_top_cobble {
                let Some(cobble_node) = try_generate_cobble_node(world, to.up(), &[to]) else {
                    debug.reject(RouteRejectReason::InitOutputTopCobbleConflict);
                    continue;
                };
                place_node(new_world.to_mut(), cobble_node);
            }

            let bounds = if input_top_cobble {
                from.up()
                    .cardinal()
                    .into_iter()
                    // Cobble 위쪽에 redstone을 배치하는 케이스
                    .chain(Some(from.up().up()))
                    .map(|pos| PlaceBound(PropagateType::Soft, pos, pos.diff(from)))
                    .collect_vec()
            } else {
                from_node.propagation_bound(Some(world))
            };

            states.push((new_world.into_owned(), vec![from], bounds));
        }
    }

    (states, debug)
}

pub(super) fn try_generate_cobble_node(
    world: &World3D,
    cobble_pos: Position,
    except: &[Position],
) -> Option<PlacedNode> {
    if cobble_would_stack_above_side_torch_support(world, cobble_pos) {
        return None;
    }
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    if !cobble_node.has_conflict(world, &except.iter().copied().collect()) {
        Some(cobble_node)
    } else {
        None
    }
}

pub(super) fn cobble_would_stack_above_side_torch_support(
    world: &World3D,
    cobble_pos: Position,
) -> bool {
    let Some(below) = cobble_pos.down() else {
        return false;
    };
    world.size.bound_on(below)
        && world[below].kind.is_cobble()
        && below.cardinal().into_iter().any(|position| {
            world.size.bound_on(position)
                && matches!(world[position].kind, BlockKind::Torch { .. })
                && world[position].direction == position.diff(below)
        })
}

fn place_redstone_for_goal(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    goal: RouteGoal,
) -> PlaceRedstoneResult {
    place_redstone_with_cobble_and_allowed_shorts(
        world,
        bound,
        prev,
        goal.placement_target(),
        goal.allowed_short_positions(),
    )
}

#[cfg(test)]
pub(super) fn place_redstone_with_cobble(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
) -> PlaceRedstoneResult {
    place_redstone_with_cobble_and_allowed_shorts(world, bound, prev, to, None)
}

fn place_redstone_with_cobble_and_allowed_shorts(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
    allowed_shorts: Option<&HashSet<Position>>,
) -> PlaceRedstoneResult {
    let Some(cobble_pos) = bound.position().walk(Direction::Bottom) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::NoBottomForCobble);
    };
    // 첫 번째 step에서 torch 위쪽에 cobble + redstone을 놓는 경우 예외 처리
    let cobble_except = (world[prev].kind.is_torch())
        .then_some(vec![cobble_pos, prev])
        .unwrap_or_default();
    let Some(cobble_node) = try_generate_cobble_node(world, cobble_pos, &cobble_except) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::CobbleConflict);
    };
    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);

    let bound_pos = bound.position();
    let Some(bound_back_pos) = bound_pos.walk(bound.direction()) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    };
    let redstone_node = PlacedNode::new_redstone(bound_pos);
    let mut except = [prev, bound_back_pos, bound_pos, to, to.up()]
        .into_iter()
        .collect::<HashSet<_>>();
    if let Some(allowed_shorts) = allowed_shorts {
        except.extend(allowed_shorts.iter().copied());
    }
    let mut short_except = [prev, bound_pos, to, to.up()]
        .into_iter()
        .collect::<HashSet<_>>();
    if let Some(allowed_shorts) = allowed_shorts {
        short_except.extend(allowed_shorts.iter().copied());
    }
    if redstone_node.has_conflict(&new_world, &except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    }
    if redstone_node.has_short(world, &short_except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
    }
    place_node(&mut new_world, redstone_node);
    new_world.update_redstone_states(prev);
    if !target_powers_redstone(&new_world, prev, redstone_node.position) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::DisconnectedRoute);
    }
    if redstone_node.has_short(&new_world, &short_except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
    }
    if let BlockKind::Torch { .. } = world[prev].kind {
        if let Some(source_cobble) = prev.walk(world[prev].direction) {
            if redstone_powers_cobble(&new_world, redstone_node.position, source_cobble) {
                return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
            }
        }
    }

    PlaceRedstoneResult::Placed(new_world, redstone_node)
}

pub(super) enum PlaceRedstoneResult {
    Placed(World3D, PlacedNode),
    Rejected(RouteRejectReason),
}
