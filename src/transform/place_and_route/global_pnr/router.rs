use std::collections::{HashSet, VecDeque};

use eyre::ContextCompat;

use crate::graph::module::{GraphModule, GraphModulePortTarget};
use crate::output::OutputEndpoint;
use crate::transform::place_and_route::detailed_router::{
    self, PlaceRedstoneResult, PlaceRepeaterResult,
};
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPort, PhysicalPortDirection,
};
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::transform::place_and_route::place_bound::{PlaceBound, PropagateType};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

const GLOBAL_ROUTE_PADDING: usize = 8;
const GLOBAL_ROUTE_MAX_STEPS: usize = 128;
const MAX_REDSTONE_STRENGTH: usize = 15;

#[derive(Clone, Debug)]
pub struct RoutedNet {
    pub source: Position,
    pub sink: Position,
    pub blocks: Vec<(Position, Block)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteFailure {
    Unreachable { source: Position, sink: Position },
}

pub fn route_module_variables(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
) -> eyre::Result<Vec<RoutedNet>> {
    let mut route_world = placed_candidate_world(candidates, placed_modules)?;
    let mut routes = Vec::new();
    let mut top_input_index = 0;

    for port in module.ports.iter().filter(|port| port.port_type.is_input()) {
        let sinks = resolve_port_target_positions(candidates, placed_modules, &port.target);
        if sinks.is_empty() {
            continue;
        }

        let source = external_switch_position(&route_world, top_input_index)
            .with_context(|| format!("failed to place top-level input switch `{}`", port.name))?;
        top_input_index += 1;
        let switch = input_switch_block();
        route_world[source] = switch;
        routes.push(RoutedNet {
            source,
            sink: source,
            blocks: vec![(source, switch)],
        });

        for sink in sinks {
            let (route, next_world) =
                route_point_to_point(&route_world, source, sink).map_err(|failure| {
                    eyre::eyre!(
                        "failed to route top-level input {} -> {sink:?}: {failure:?}",
                        port.name
                    )
                })?;
            route_world = next_world;
            if std::env::var_os("PRINT_GLOBAL_PNR").is_some() {
                eprintln!(
                    "route top input {}: {:?} -> {:?}",
                    port.name, source, route.sink
                );
            }
            routes.push(route);
        }
    }

    for var in &module.vars {
        let (source_port, source_candidate, source_placed) =
            resolve_port(candidates, placed_modules, &var.source.0, &var.source.1).with_context(
                || {
                    format!(
                        "source port {}.{} is not placed",
                        var.source.0, var.source.1
                    )
                },
            )?;
        let source = translate_port_route_position(source_port, source_candidate, source_placed);
        let logical_source =
            translate_candidate_position(source_port.position, source_candidate, source_placed);
        let sink = resolve_port_position(candidates, placed_modules, &var.target.0, &var.target.1)
            .with_context(|| {
                format!(
                    "target port {}.{} is not placed",
                    var.target.0, var.target.1
                )
            })?;
        let (route, next_world) =
            route_source_to_point(&route_world, source_port, logical_source, source, sink)
                .map_err(|failure| {
                    eyre::eyre!(
                        "failed to route {}.{} -> {}.{}: {failure:?}",
                        var.source.0,
                        var.source.1,
                        var.target.0,
                        var.target.1
                    )
                })?;
        route_world = next_world;
        if std::env::var_os("PRINT_GLOBAL_PNR").is_some() {
            eprintln!(
                "route var {}.{} -> {}.{}: {:?} -> {:?}",
                var.source.0, var.source.1, var.target.0, var.target.1, source, route.sink
            );
            eprintln!(
                "  blocks: {:?}",
                route
                    .blocks
                    .iter()
                    .map(|(position, block)| (*position, block.kind))
                    .collect::<Vec<_>>()
            );
        }
        routes.push(route);
    }

    Ok(routes)
}

fn route_source_to_point(
    world: &World3D,
    source_port: &PhysicalPort,
    logical_source: Position,
    route_source: Position,
    sink: Position,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    if source_port.direction == PhysicalPortDirection::Output
        && source_port.route_position.is_none()
    {
        return route_isolated_output_to_point(world, logical_source, sink);
    }

    route_point_to_point(world, route_source, sink)
}

pub fn collect_module_output_endpoints(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
) -> Vec<OutputEndpoint> {
    module
        .ports
        .iter()
        .filter(|port| port.port_type.is_output())
        .flat_map(|port| {
            resolve_observable_port_target_positions(candidates, placed_modules, &port.target)
                .into_iter()
                .map(|position| OutputEndpoint::new(port.name.clone(), position))
        })
        .collect()
}

fn resolve_observable_port_target_positions(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    target: &GraphModulePortTarget,
) -> Vec<Position> {
    match target {
        GraphModulePortTarget::Module(module_name, port_name) => {
            resolve_observable_port_position(candidates, placed_modules, module_name, port_name)
                .into_iter()
                .collect()
        }
        GraphModulePortTarget::Wire(targets) => targets
            .iter()
            .filter_map(|(module_name, port_name)| {
                resolve_observable_port_position(candidates, placed_modules, module_name, port_name)
            })
            .collect(),
        GraphModulePortTarget::Node(_) => Vec::new(),
    }
}

fn resolve_port_target_positions(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    target: &GraphModulePortTarget,
) -> Vec<Position> {
    match target {
        GraphModulePortTarget::Module(module_name, port_name) => {
            resolve_port_position(candidates, placed_modules, module_name, port_name)
                .into_iter()
                .collect()
        }
        GraphModulePortTarget::Wire(targets) => targets
            .iter()
            .filter_map(|(module_name, port_name)| {
                resolve_port_position(candidates, placed_modules, module_name, port_name)
            })
            .collect(),
        GraphModulePortTarget::Node(_) => Vec::new(),
    }
}

fn external_switch_position(world: &World3D, index: usize) -> Option<Position> {
    let max_x = world
        .iter_block()
        .into_iter()
        .map(|(position, _)| position.0)
        .max()
        .unwrap_or(0);
    let position = Position(max_x + 2, index * 3 + 1, 1);
    (world.size.bound_on(position) && world[position].kind.is_air()).then_some(position)
}

fn input_switch_block() -> Block {
    Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::West,
    }
}

fn resolve_port_position(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    module_name: &str,
    port_name: &str,
) -> Option<Position> {
    resolve_port(candidates, placed_modules, module_name, port_name).map(
        |(port, candidate, placed)| translate_candidate_position(port.position, candidate, placed),
    )
}

fn resolve_observable_port_position(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    module_name: &str,
    port_name: &str,
) -> Option<Position> {
    resolve_port(candidates, placed_modules, module_name, port_name).map(
        |(port, candidate, placed)| {
            let position = port
                .route_position
                .unwrap_or_else(|| observable_port_position(candidate, port.position));
            translate_candidate_position(position, candidate, placed)
        },
    )
}

fn observable_port_position(candidate: &LayoutCandidate, position: Position) -> Position {
    if !candidate.world.size.bound_on(position) || !candidate.world[position].kind.is_torch() {
        return position;
    }

    candidate
        .world
        .iter_block()
        .into_iter()
        .filter(|(tap, block)| {
            block.kind.is_redstone()
                && detailed_router::target_powers_redstone(&candidate.world, position, *tap)
        })
        .map(|(tap, _)| tap)
        .min_by_key(|tap| (position.manhattan_distance(tap), tap.0, tap.1, tap.2))
        .unwrap_or(position)
}

fn translate_port_route_position(
    port: &PhysicalPort,
    candidate: &LayoutCandidate,
    placed: &PlacedModule,
) -> Position {
    translate_candidate_position(
        port.route_position.unwrap_or(port.position),
        candidate,
        placed,
    )
}

fn resolve_port<'a>(
    candidates: &'a [LayoutCandidate],
    placed_modules: &'a [PlacedModule],
    module_name: &str,
    port_name: &str,
) -> Option<(
    &'a crate::transform::place_and_route::global_pnr::ir::PhysicalPort,
    &'a LayoutCandidate,
    &'a PlacedModule,
)> {
    let placed = placed_modules
        .iter()
        .find(|placed| placed.module_name == module_name)?;
    let candidate = candidates.get(placed.candidate_index)?;
    let port = candidate.ports.iter().find(|port| port.name == port_name)?;
    Some((port, candidate, placed))
}

fn placed_candidate_world(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
) -> eyre::Result<World3D> {
    let blocks = translated_candidate_blocks(candidates, placed_modules)?;
    let mut world = World3D::new(route_world_size(&blocks));
    for (position, block) in blocks {
        if !world[position].kind.is_air() {
            eyre::bail!("global route base collision at {position:?}");
        }
        world[position] = block;
    }
    world.initialize_redstone_states();
    Ok(world)
}

fn translated_candidate_blocks(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
) -> eyre::Result<Vec<(Position, Block)>> {
    let mut blocks = Vec::new();
    for placed in placed_modules {
        let candidate = candidates
            .get(placed.candidate_index)
            .with_context(|| format!("missing candidate {}", placed.candidate_index))?;
        blocks.extend(
            candidate
                .world
                .iter_block()
                .into_iter()
                .map(|(position, block)| {
                    (
                        translate_candidate_position(position, candidate, placed),
                        block,
                    )
                }),
        );
    }
    Ok(blocks)
}

fn route_world_size(blocks: &[(Position, Block)]) -> DimSize {
    let mut max = Position(0, 0, 0);
    for (position, _) in blocks {
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }
    DimSize(
        max.0 + GLOBAL_ROUTE_PADDING + 1,
        max.1 + GLOBAL_ROUTE_PADDING + 1,
        max.2 + GLOBAL_ROUTE_PADDING + 1,
    )
}

fn translate_candidate_position(
    position: Position,
    candidate: &LayoutCandidate,
    placed: &PlacedModule,
) -> Position {
    Position(
        placed.origin.0 + position.0 - candidate.bbox.min.0,
        placed.origin.1 + position.1 - candidate.bbox.min.1,
        placed.origin.2 + position.2 - candidate.bbox.min.2,
    )
}

pub fn route_point_to_point(
    world: &World3D,
    source: Position,
    sink: Position,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let goal = RouteGoal::for_sink(world, sink);
    route_point_to_point_with_bounds(world, source, sink, goal, BoundSearchMode::Propagation)
        .or_else(|_| {
            route_point_to_point_with_bounds(world, source, sink, goal, BoundSearchMode::Nearby)
        })
}

fn route_isolated_output_to_point(
    world: &World3D,
    source: Position,
    sink: Position,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let debug = std::env::var_os("PRINT_GLOBAL_PNR").is_some();
    let initial_states = isolated_output_initial_states(world, source, sink);
    if debug {
        debug_isolated_initial_states(&initial_states, source, sink);
    }
    for initial in initial_states {
        let Ok((route, routed_world)) =
            route_point_to_point_from_initial_state(world, source, sink, initial)
        else {
            continue;
        };
        if debug {
            eprintln!("isolated route source {:?} -> {:?}", source, sink);
        }
        return Ok((route, routed_world));
    }

    for tap in routeable_output_taps(world, source, sink) {
        let Ok((_, routed_world)) = route_point_to_point(world, tap, sink) else {
            continue;
        };
        if debug {
            eprintln!(
                "isolated powered-tap route source {:?} tap {:?} -> {:?}",
                source, tap, sink
            );
        }
        return Ok((
            RoutedNet {
                source,
                sink,
                blocks: added_route_blocks(world, &routed_world),
            },
            routed_world,
        ));
    }

    if debug {
        eprintln!("isolated route failed {:?} -> {:?}", source, sink);
    }
    Err(RouteFailure::Unreachable { source, sink })
}

fn routeable_output_taps(world: &World3D, source: Position, sink: Position) -> Vec<Position> {
    let direct_taps = world
        .iter_block()
        .into_iter()
        .filter(|(position, block)| {
            block.kind.is_redstone()
                && detailed_router::target_powers_redstone(world, source, *position)
        })
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    let mut taps = redstone_network_positions(world, &direct_taps);
    taps.sort_by_key(|position| {
        (
            position.manhattan_distance(&sink),
            std::cmp::Reverse(position.manhattan_distance(&source)),
            position.0,
            position.1,
            position.2,
        )
    });
    taps
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
            if detailed_router::target_powers_redstone(world, position, next)
                || detailed_router::target_powers_redstone(world, next, position)
            {
                visited.insert(next);
                queue.push_back(next);
            }
        }
    }

    visited.into_iter().collect()
}

fn debug_isolated_initial_states(
    initial_states: &[RouteSearchState],
    source: Position,
    sink: Position,
) {
    eprintln!(
        "isolated initial states source {:?} -> {:?}: {}",
        source,
        sink,
        initial_states.len()
    );
    if let Some(state) = initial_states.first() {
        let mut nearby_blocks = Vec::new();
        for (position, block) in state.world.iter_block() {
            if source.manhattan_distance(&position) <= 3 && !block.kind.is_air() {
                nearby_blocks.push((position, block.kind, block.direction));
            }
        }
        nearby_blocks.sort_by_key(|(position, _, _)| {
            (
                source.manhattan_distance(position),
                position.0,
                position.1,
                position.2,
            )
        });
        eprintln!("  nearby blocks: {:?}", nearby_blocks);
    }
    for (index, state) in initial_states.iter().enumerate() {
        let bounds = state.pending_bounds.as_deref().unwrap_or(&[]);
        let mut redstone_placed = 0usize;
        let mut repeater_placed = 0usize;
        let mut redstone_rejected = 0usize;
        let mut repeater_rejected = 0usize;
        let mut repeater_routes = Vec::new();
        let mut repeater_rejects = Vec::new();
        for bound in bounds {
            if !bound.is_bound_on(&state.world) {
                continue;
            }
            match detailed_router::place_redstone_with_cobble_and_allowed_shorts(
                &state.world,
                *bound,
                state.terminal,
                sink,
                None,
            ) {
                PlaceRedstoneResult::Placed(_, _) => redstone_placed += 1,
                PlaceRedstoneResult::Rejected(_) => redstone_rejected += 1,
            }
            for direction in Direction::iter_direction_without_top()
                .into_iter()
                .filter(|direction| direction.is_cardinal())
            {
                match detailed_router::place_repeater_with_cobble(
                    &state.world,
                    *bound,
                    state.terminal,
                    sink,
                    direction,
                    None,
                ) {
                    PlaceRepeaterResult::Placed(next_world, repeater_node) => {
                        repeater_placed += 1;
                        let routes =
                            route_point_to_point(&next_world, repeater_node.position, sink).is_ok();
                        repeater_routes.push((repeater_node.position, direction, routes));
                    }
                    PlaceRepeaterResult::Rejected(reason) => {
                        repeater_rejected += 1;
                        repeater_rejects.push((bound.position(), direction, reason));
                    }
                }
            }
        }
        eprintln!(
            "  initial #{index}: bounds={} redstone placed/rejected={}/{} repeater placed/rejected={}/{}",
            bounds.len(),
            redstone_placed,
            redstone_rejected,
            repeater_placed,
            repeater_rejected
        );
        eprintln!(
            "    bounds: {:?}",
            bounds
                .iter()
                .map(|bound| (bound.position(), bound.direction()))
                .collect::<Vec<_>>()
        );
        eprintln!("    repeater route probes: {:?}", repeater_routes);
        eprintln!("    repeater rejects: {:?}", repeater_rejects);
    }
}

fn isolated_output_initial_states(
    world: &World3D,
    source: Position,
    sink: Position,
) -> Vec<RouteSearchState> {
    let source_node = PlacedNode::new(source, world[source]);
    let mut states = Vec::new();

    states.push(RouteSearchState {
        world: world.clone(),
        terminal: source,
        route: vec![source],
        signal_strength: 2,
        pending_bounds: Some(sorted_route_bounds(
            source_node.propagation_bound(Some(world)),
            world,
            sink,
        )),
    });

    let top_cobble = source.up();
    if let Some(cobble_node) =
        detailed_router::try_generate_cobble_node(world, top_cobble, &[source])
    {
        let mut with_top_cobble = world.clone();
        detailed_router::place_node(&mut with_top_cobble, cobble_node);
        let bounds = top_cobble
            .cardinal()
            .into_iter()
            .chain([top_cobble.up()])
            .map(|candidate| PlaceBound(PropagateType::Soft, candidate, candidate.diff(source)))
            .collect::<Vec<_>>();
        states.push(RouteSearchState {
            world: with_top_cobble,
            terminal: source,
            route: vec![source],
            signal_strength: 2,
            pending_bounds: Some(sorted_route_bounds(bounds, world, sink)),
        });
    }

    states
}

fn sorted_route_bounds(
    mut bounds: Vec<PlaceBound>,
    world: &World3D,
    sink: Position,
) -> Vec<PlaceBound> {
    bounds.sort_by_key(|bound| {
        let position = bound.position();
        (
            !world.size.bound_on(position),
            world
                .size
                .bound_on(position)
                .then(|| !world[position].kind.is_air())
                .unwrap_or(true),
            position.manhattan_distance(&sink),
            position.0,
            position.1,
            position.2,
        )
    });
    bounds
}

#[derive(Clone, Copy, Debug)]
enum BoundSearchMode {
    Propagation,
    Nearby,
}

fn route_point_to_point_with_bounds(
    world: &World3D,
    source: Position,
    sink: Position,
    goal: RouteGoal,
    mode: BoundSearchMode,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    route_point_to_point_with_bounds_and_min_z(world, source, sink, goal, mode, None)
}

fn route_point_to_point_with_bounds_and_min_z(
    world: &World3D,
    source: Position,
    sink: Position,
    goal: RouteGoal,
    mode: BoundSearchMode,
    min_z: Option<usize>,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let initial_strength = initial_signal_strength(world, source);
    route_point_to_point_with_initial_queue(
        world,
        source,
        sink,
        goal,
        mode,
        min_z,
        VecDeque::from([RouteSearchState {
            world: world.clone(),
            terminal: source,
            route: vec![source],
            signal_strength: initial_strength,
            pending_bounds: None,
        }]),
    )
}

fn route_point_to_point_from_initial_state(
    world: &World3D,
    source: Position,
    sink: Position,
    initial_state: RouteSearchState,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let goal = RouteGoal::for_sink(world, sink);
    route_point_to_point_with_initial_queue(
        world,
        source,
        sink,
        goal,
        BoundSearchMode::Propagation,
        None,
        VecDeque::from([initial_state.clone()]),
    )
    .or_else(|_| {
        route_point_to_point_with_initial_queue(
            world,
            source,
            sink,
            goal,
            BoundSearchMode::Nearby,
            None,
            VecDeque::from([initial_state]),
        )
    })
}

fn route_point_to_point_with_initial_queue(
    original_world: &World3D,
    source: Position,
    sink: Position,
    goal: RouteGoal,
    mode: BoundSearchMode,
    min_z: Option<usize>,
    mut queue: VecDeque<RouteSearchState>,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let initial_visited = queue.iter().map(|state| {
        (
            state.terminal,
            state.route.len().saturating_sub(1),
            state.signal_strength,
        )
    });
    let mut visited = initial_visited.collect::<HashSet<_>>();

    while let Some(state) = queue.pop_front() {
        if !is_route_terminal(&state.world, state.terminal) {
            continue;
        }
        if goal.accepts(&state.world, state.terminal) {
            let blocks = added_route_blocks(original_world, &state.world);
            return Ok((
                RoutedNet {
                    source,
                    sink,
                    blocks,
                },
                state.world,
            ));
        }

        if state.route.len() > GLOBAL_ROUTE_MAX_STEPS {
            continue;
        }

        let terminal_node = PlacedNode::new(state.terminal, state.world[state.terminal]);
        let allowed_shorts = goal.allowed_short_positions();
        let bounds = state
            .pending_bounds
            .unwrap_or_else(|| route_bounds_for_mode(mode, &state.world, &terminal_node));
        for bound in bounds {
            if !bound.is_bound_on(&state.world) || goal.rejects_bound_position(bound.position()) {
                continue;
            }
            if min_z.is_some_and(|min_z| {
                bound.position().2 < min_z && bound.position().manhattan_distance(&source) <= 4
            }) {
                continue;
            }

            if state.signal_strength > 1 {
                match detailed_router::place_redstone_with_cobble_and_allowed_shorts(
                    &state.world,
                    bound,
                    state.terminal,
                    goal.placement_target(),
                    allowed_shorts.as_ref(),
                ) {
                    PlaceRedstoneResult::Placed(next_world, redstone_node) => {
                        let next_strength = state.signal_strength - 1;
                        if visited.insert((
                            redstone_node.position,
                            state.route.len(),
                            next_strength,
                        )) {
                            let mut next_route = state.route.clone();
                            next_route.push(redstone_node.position);
                            queue.push_back(RouteSearchState {
                                world: next_world,
                                terminal: redstone_node.position,
                                route: next_route,
                                signal_strength: next_strength,
                                pending_bounds: None,
                            });
                        }
                    }
                    PlaceRedstoneResult::Rejected(_) => {}
                }
            }

            if state.signal_strength <= 2 {
                for direction in Direction::iter_direction_without_top()
                    .into_iter()
                    .filter(|direction| direction.is_cardinal())
                {
                    match detailed_router::place_repeater_with_cobble(
                        &state.world,
                        bound,
                        state.terminal,
                        goal.placement_target(),
                        direction,
                        allowed_shorts.as_ref(),
                    ) {
                        PlaceRepeaterResult::Placed(next_world, repeater_node) => {
                            if visited.insert((
                                repeater_node.position,
                                state.route.len(),
                                MAX_REDSTONE_STRENGTH,
                            )) {
                                let mut next_route = state.route.clone();
                                next_route.push(repeater_node.position);
                                queue.push_back(RouteSearchState {
                                    world: next_world,
                                    terminal: repeater_node.position,
                                    route: next_route,
                                    signal_strength: MAX_REDSTONE_STRENGTH,
                                    pending_bounds: None,
                                });
                            }
                        }
                        PlaceRepeaterResult::Rejected(_) => {}
                    }
                }
            }
        }
    }

    Err(RouteFailure::Unreachable { source, sink })
}

#[derive(Clone)]
struct RouteSearchState {
    world: World3D,
    terminal: Position,
    route: Vec<Position>,
    signal_strength: usize,
    pending_bounds: Option<Vec<PlaceBound>>,
}

fn initial_signal_strength(world: &World3D, source: Position) -> usize {
    match world[source].kind {
        BlockKind::Redstone { strength, .. } if strength > 0 => strength,
        BlockKind::Redstone { .. } => 2,
        BlockKind::Switch { .. }
        | BlockKind::Torch { .. }
        | BlockKind::Repeater { .. }
        | BlockKind::RedstoneBlock => MAX_REDSTONE_STRENGTH,
        _ => 0,
    }
}

fn is_route_terminal(world: &World3D, position: Position) -> bool {
    world[position].kind.is_redstone()
        || world[position].kind.is_switch()
        || world[position].kind.is_torch()
        || world[position].kind.is_repeater()
        || matches!(
            world[position].kind,
            crate::world::block::BlockKind::RedstoneBlock
        )
}

#[derive(Clone, Copy)]
enum RouteGoal {
    ConnectPosition { target: Position },
    PowerCobble { cobble: Position },
    PlaceRedstone { target: Position },
}

impl RouteGoal {
    fn for_sink(world: &World3D, sink: Position) -> Self {
        if world[sink].kind.is_cobble() {
            return Self::PowerCobble { cobble: sink };
        }
        if world[sink].kind.is_air() {
            return Self::PlaceRedstone { target: sink };
        }
        Self::ConnectPosition { target: sink }
    }

    fn placement_target(self) -> Position {
        match self {
            Self::ConnectPosition { target } => target,
            Self::PowerCobble { cobble } => cobble,
            Self::PlaceRedstone { target } => target,
        }
    }

    fn rejects_bound_position(self, position: Position) -> bool {
        match self {
            Self::PlaceRedstone { .. } => false,
            _ => position == self.placement_target(),
        }
    }

    fn accepts(self, world: &World3D, redstone: Position) -> bool {
        match self {
            Self::ConnectPosition { target } => {
                detailed_router::target_powers_redstone(world, target, redstone)
            }
            Self::PowerCobble { cobble } => {
                detailed_router::redstone_powers_cobble(world, redstone, cobble)
            }
            Self::PlaceRedstone { target } => {
                redstone == target && world[redstone].kind.is_redstone()
            }
        }
    }

    fn allowed_short_positions(self) -> Option<HashSet<Position>> {
        match self {
            Self::PowerCobble { cobble } => Some(
                cobble
                    .cardinal()
                    .into_iter()
                    .chain([cobble, cobble.up()])
                    .collect(),
            ),
            Self::ConnectPosition { .. } | Self::PlaceRedstone { .. } => None,
        }
    }
}

fn route_bounds_for_mode(
    mode: BoundSearchMode,
    world: &World3D,
    terminal_node: &PlacedNode,
) -> Vec<PlaceBound> {
    match mode {
        BoundSearchMode::Propagation => terminal_node.propagation_bound(Some(world)),
        BoundSearchMode::Nearby
            if terminal_node.block.kind.is_repeater()
                || terminal_node.block.kind.is_torch()
                || terminal_node.block.kind.is_switch() =>
        {
            terminal_node.propagation_bound(Some(world))
        }
        BoundSearchMode::Nearby => nearby_route_bounds(world, terminal_node.position),
    }
}

fn nearby_route_bounds(world: &World3D, position: Position) -> Vec<PlaceBound> {
    let mut result = Vec::new();
    for next in nearby_route_positions(world, position) {
        result.push(PlaceBound(PropagateType::Soft, next, next.diff(position)));
    }
    result
}

fn nearby_route_positions(world: &World3D, position: Position) -> Vec<Position> {
    let mut result = Vec::new();
    let horizontal = [
        (position.0.checked_add(1), Some(position.1)),
        (Some(position.0), position.1.checked_add(1)),
        (position.0.checked_sub(1), Some(position.1)),
        (Some(position.0), position.1.checked_sub(1)),
    ];

    for (next_x, next_y) in horizontal {
        let (Some(next_x), Some(next_y)) = (next_x, next_y) else {
            continue;
        };
        for next_z in [position.2, position.2 + 1] {
            let next = Position(next_x, next_y, next_z);
            if world.size.bound_on(next) {
                result.push(next);
            }
        }
        if let Some(next_z) = position.2.checked_sub(1) {
            let next = Position(next_x, next_y, next_z);
            if world.size.bound_on(next) {
                result.push(next);
            }
        }
    }
    result
}

fn added_route_blocks(before: &World3D, after: &World3D) -> Vec<(Position, Block)> {
    after
        .iter_block()
        .into_iter()
        .filter(|(position, _)| before[*position].kind.is_air())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::module::{
        GraphModule, GraphModulePort, GraphModulePortTarget, GraphModulePortType,
        GraphModuleVariable,
    };
    use crate::transform::place_and_route::global_pnr::ir::{
        LayoutCandidate, PhysicalPort, PhysicalPortDirection,
    };
    use crate::transform::place_and_route::global_pnr::placer::{
        place_candidates_on_shelves, GlobalPlacementConfig,
    };
    use crate::world::block::{BlockKind, Direction};
    use crate::world::simulator::Simulator;
    use crate::world::World;

    fn candidate(
        module_name: &str,
        block_position: Position,
        port_name: &str,
        direction: PhysicalPortDirection,
    ) -> LayoutCandidate {
        let mut world = World3D::new(DimSize(2, 1, 2));
        world[block_position.down().unwrap()] = cobble_block();
        world[block_position] = redstone_block();
        world.initialize_redstone_states();
        LayoutCandidate::from_world(
            module_name.to_owned(),
            world,
            vec![PhysicalPort {
                name: port_name.to_owned(),
                direction,
                position: block_position,
                route_position: None,
            }],
        )
        .unwrap()
    }

    fn route_test_world(source: Position, sink: Position) -> World3D {
        let mut world = World3D::new(DimSize(8, 4, 3));
        world[source.down().unwrap()] = cobble_block();
        world[source] = redstone_block();
        world[sink.down().unwrap()] = cobble_block();
        world[sink] = redstone_block();
        world.initialize_redstone_states();
        world
    }

    fn route_test_world_with_size(source: Position, sink: Position, size: DimSize) -> World3D {
        let mut world = World3D::new(size);
        world[source.down().unwrap()] = cobble_block();
        world[source] = redstone_block();
        world[sink.down().unwrap()] = cobble_block();
        world[sink] = redstone_block();
        world.initialize_redstone_states();
        world
    }

    fn route_test_world_with_switch_source(
        source: Position,
        sink: Position,
        size: DimSize,
    ) -> World3D {
        let mut world = World3D::new(size);
        world[source] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[sink.down().unwrap()] = cobble_block();
        world[sink] = redstone_block();
        world.initialize_redstone_states();
        world
    }

    fn cobble_block() -> Block {
        Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        }
    }

    fn redstone_block() -> Block {
        Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        }
    }

    #[test]
    fn route_point_to_point_places_support_cobble_under_redstone() {
        let source = Position(0, 1, 1);
        let sink = Position(3, 1, 1);
        let world = route_test_world(source, sink);
        let (route, _) = route_point_to_point(&world, source, sink).unwrap();

        assert!(route.blocks.iter().any(|(_, block)| block.kind.is_cobble()));
        assert!(route
            .blocks
            .iter()
            .any(|(_, block)| block.kind.is_redstone()));
    }

    #[test]
    fn route_point_to_point_avoids_blocked_route_position() {
        let source = Position(0, 1, 1);
        let sink = Position(3, 1, 1);
        let mut world = route_test_world(source, sink);
        world[Position(1, 1, 1)] = cobble_block();
        let (route, _) = route_point_to_point(&world, source, sink).unwrap();

        assert!(!route
            .blocks
            .iter()
            .any(|(position, _)| *position == Position(1, 1, 1)));
    }

    #[test]
    fn route_point_to_point_refreshes_long_redstone_with_repeater() {
        let source = Position(0, 1, 1);
        let sink = Position(22, 1, 1);
        let world = route_test_world_with_size(source, sink, DimSize(26, 4, 3));
        let (route, _) = route_point_to_point(&world, source, sink).unwrap();

        assert!(route
            .blocks
            .iter()
            .any(|(_, block)| block.kind.is_repeater()));
    }

    #[test]
    fn route_point_to_point_does_not_assume_unpowered_redstone_has_full_strength() {
        let source = Position(0, 1, 1);
        let sink = Position(14, 1, 1);
        let world = route_test_world_with_size(source, sink, DimSize(18, 4, 3));
        let (route, _) = route_point_to_point(&world, source, sink).unwrap();

        assert!(
            route
                .blocks
                .iter()
                .any(|(_, block)| block.kind.is_repeater()),
            "unpowered redstone output ports need a repeater before a long downstream route"
        );
    }

    #[test]
    fn route_point_to_point_long_route_powers_sink_through_repeater() -> eyre::Result<()> {
        let source = Position(0, 1, 1);
        let sink = Position(22, 1, 1);
        let world = route_test_world_with_switch_source(source, sink, DimSize(26, 4, 3));
        let (_, routed_world) = route_point_to_point(&world, source, sink).unwrap();
        let world = World::from(&routed_world);
        let mut sim = Simulator::from_with_limits_and_trace(&world, 128, 20_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        sim.change_state_with_limits(vec![(source, true)], 128, 20_000)?;

        assert!(
            matches!(sim.world()[sink].kind, BlockKind::Redstone { strength, .. } if strength > 0),
            "sink redstone should be powered through the inserted repeater"
        );
        Ok(())
    }

    #[test]
    fn route_rejects_repeater_floating_above_adjacent_redstone() {
        let prev = Position(17, 7, 4);
        let repeater = Position(18, 7, 5);
        let mut world = World3D::new(DimSize(20, 9, 7));
        world[prev.down().unwrap()] = cobble_block();
        world[prev] = redstone_block();
        world.initialize_redstone_states();

        let result = detailed_router::place_repeater_with_cobble(
            &world,
            PlaceBound(PropagateType::Soft, repeater, Direction::East),
            prev,
            Position(19, 7, 5),
            Direction::East,
            None,
        );

        assert!(matches!(
            result,
            PlaceRepeaterResult::Rejected(detailed_router::RouteRejectReason::DisconnectedRoute)
        ));
    }

    #[test]
    fn route_module_variables_connects_placed_candidate_ports() -> eyre::Result<()> {
        let module = GraphModule {
            vars: vec![GraphModuleVariable {
                var_type: GraphModulePortType::InputNet,
                source: ("left".to_owned(), "out".to_owned()),
                target: ("right".to_owned(), "in".to_owned()),
            }],
            ..Default::default()
        };
        let candidates = vec![
            candidate(
                "left",
                Position(0, 0, 1),
                "out",
                PhysicalPortDirection::Output,
            ),
            candidate(
                "right",
                Position(0, 0, 1),
                "in",
                PhysicalPortDirection::Input,
            ),
        ];
        let placed = place_candidates_on_shelves(
            &candidates,
            &GlobalPlacementConfig {
                spacing: 3,
                shelf_width: 16,
            },
        );

        let routes = route_module_variables(&module, &candidates, &placed)?;

        assert_eq!(routes.len(), 1);
        assert!(!routes[0].blocks.is_empty());
        Ok(())
    }

    #[test]
    fn route_module_variables_ignores_top_level_output_ports() -> eyre::Result<()> {
        let module = GraphModule {
            ports: vec![GraphModulePort {
                name: "q".to_owned(),
                port_type: GraphModulePortType::OutputNet,
                target: GraphModulePortTarget::Module("left".to_owned(), "out".to_owned()),
            }],
            ..Default::default()
        };
        let candidates = vec![candidate(
            "left",
            Position(0, 0, 1),
            "out",
            PhysicalPortDirection::Output,
        )];
        let placed = place_candidates_on_shelves(&candidates, &GlobalPlacementConfig::default());

        let routes = route_module_variables(&module, &candidates, &placed)?;

        assert!(routes.is_empty());
        Ok(())
    }
}
