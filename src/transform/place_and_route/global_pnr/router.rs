use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

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
use crate::transform::place_and_route::global_pnr::progress::GlobalPnrProgress;
use crate::transform::place_and_route::place_bound::{PlaceBound, PropagateType};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

const GLOBAL_ROUTE_PADDING: usize = 8;
const GLOBAL_ROUTE_MAX_STEPS: usize = 128;
const GLOBAL_ROUTE_ASTAR_MAX_EXPANSIONS: usize = 2_000;
const MAX_REDSTONE_STRENGTH: usize = 15;
const FANOUT_ROUTE_SOURCE_LIMIT: usize = 8;
const FANOUT_ROUTE_TERMINAL_LIMIT: usize = 24;
const OUTPUT_ISOLATION_ESCAPE_MAX_STEPS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalRoutingStrategy {
    BreadthFirst,
    AStar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalRoutingConfig {
    pub strategy: GlobalRoutingStrategy,
}

impl Default for GlobalRoutingConfig {
    fn default() -> Self {
        Self {
            strategy: GlobalRoutingStrategy::AStar,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedPortTarget {
    position: Position,
    isolate_input: bool,
    module_contains_sequential: bool,
}

#[derive(Clone, Debug)]
pub struct RoutedNet {
    pub source: Position,
    pub sink: Position,
    pub blocks: Vec<(Position, Block)>,
    pub path: Vec<Position>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteFailure {
    Unreachable { source: Position, sink: Position },
}

pub fn route_module_variables(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    config: &GlobalRoutingConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<Vec<RoutedNet>> {
    let mut route_world = placed_candidate_world(candidates, placed_modules)?;
    let mut routes = Vec::new();
    let top_inputs = module
        .ports
        .iter()
        .filter(|port| port.port_type.is_input())
        .collect::<Vec<_>>();

    let mut top_input_index = 0;
    for (port_index, port) in top_inputs.iter().enumerate() {
        let mut sinks = resolve_port_target_positions(candidates, placed_modules, &port.target);
        if sinks.is_empty() {
            progress.item(
                port_index + 1,
                top_inputs.len(),
                format!("skip top input `{}` with no placed sinks", port.name),
            );
            continue;
        }

        let source = external_switch_position(&route_world, top_input_index, &sinks)
            .with_context(|| format!("failed to place top-level input switch `{}`", port.name))?;
        top_input_index += 1;
        sinks.sort_by_key(|sink| source.manhattan_distance(&sink.position));
        let switch = input_switch_block();
        route_world[source] = switch;
        routes.push(RoutedNet {
            source,
            sink: source,
            blocks: vec![(source, switch)],
            path: vec![source],
        });

        let mut route_sources = vec![source];
        let mut route_source_set = HashSet::from([source]);
        for (sink_index, sink) in sinks.iter().copied().enumerate() {
            progress.item(
                port_index + 1,
                top_inputs.len(),
                format!(
                    "route top input `{}` sink {}/{}",
                    port.name,
                    sink_index + 1,
                    sinks.len()
                ),
            );
            let (route, next_world) = route_to_target_from_network(
                &route_world,
                &route_sources,
                sink,
                &sinks,
                config.strategy,
            )
            .map_err(|failure| {
                eyre::eyre!(
                    "failed to route top-level input {} -> {:?}: {failure:?}",
                    port.name,
                    sink.position
                )
            })?;
            if !active_route_powers_sink(&route_world, &next_world, &route) {
                return Err(eyre::eyre!(
                    "routed top-level input {} -> {:?}, but active source {:?} does not power sink",
                    port.name,
                    sink.position,
                    route.source,
                ));
            }
            for position in route_terminal_positions(&next_world, &route.path, sink.position) {
                if route_source_set.insert(position) {
                    route_sources.push(position);
                }
            }
            prune_route_sources(
                &mut route_sources,
                &mut route_source_set,
                &sinks,
                sink_index + 1,
            );
            route_world = next_world;
            routes.push(route);
        }
    }

    let mut vars = module.vars.iter().collect::<Vec<_>>();
    vars.sort_by_key(|var| route_variable_priority(var));
    let mut var_index = 0;
    while var_index < vars.len() {
        let source_key = vars[var_index].source.clone();
        let group_start = var_index;
        while var_index < vars.len() && vars[var_index].source == source_key {
            var_index += 1;
        }
        let group_vars = &vars[group_start..var_index];

        let (source_port, source_candidate, source_placed) =
            resolve_port(candidates, placed_modules, &source_key.0, &source_key.1).with_context(
                || {
                    format!(
                        "source port {}.{} is not placed",
                        source_key.0, source_key.1
                    )
                },
            )?;
        let source = translate_port_route_position(source_port, source_candidate, source_placed);
        let logical_source =
            translate_candidate_position(source_port.position, source_candidate, source_placed);

        let mut sink_targets = Vec::new();
        for var in group_vars {
            let sinks =
                resolve_port_targets(candidates, placed_modules, &var.target.0, &var.target.1);
            if sinks.is_empty() {
                return Err(eyre::eyre!(
                    "target port {}.{} is not placed",
                    var.target.0,
                    var.target.1
                ));
            }
            for sink in sinks {
                sink_targets.push((*var, sink));
            }
        }
        sink_targets
            .sort_by_key(|(_, sink)| std::cmp::Reverse(source.manhattan_distance(&sink.position)));

        let mut route_sources = vec![source, logical_source];
        let mut route_source_set = HashSet::from([source, logical_source]);
        let all_sinks = sink_targets
            .iter()
            .map(|(_, sink)| *sink)
            .collect::<Vec<_>>();

        for (sink_index, (var, sink)) in sink_targets.iter().copied().enumerate() {
            progress.item(
                group_start + 1,
                vars.len(),
                format!(
                    "route net `{}.{}` -> `{}.{}` sink {}/{}",
                    source_key.0,
                    source_key.1,
                    var.target.0,
                    var.target.1,
                    sink_index + 1,
                    sink_targets.len()
                ),
            );
            let route_result = if sink_index == 0 {
                route_source_to_target_position(
                    &route_world,
                    source_port,
                    logical_source,
                    source,
                    sink,
                    &all_sinks,
                    config.strategy,
                )
            } else {
                route_to_target_from_network(
                    &route_world,
                    &route_sources,
                    sink,
                    &all_sinks,
                    config.strategy,
                )
            };
            let (route, next_world) = route_result.map_err(|failure| {
                eyre::eyre!(
                    "failed to route {}.{} -> {}.{} at {:?}: {failure:?}",
                    source_key.0,
                    source_key.1,
                    var.target.0,
                    var.target.1,
                    sink.position
                )
            })?;
            if !active_route_powers_sink(&route_world, &next_world, &route) {
                return Err(eyre::eyre!(
                    "routed {}.{} -> {}.{} at {:?}, but active source {:?} does not power sink",
                    source_key.0,
                    source_key.1,
                    var.target.0,
                    var.target.1,
                    sink.position,
                    route.source,
                ));
            }
            progress.detail(format!(
                "routed `{}.{}` -> `{}.{}` from {:?} to {:?} with {} block(s)",
                source_key.0,
                source_key.1,
                var.target.0,
                var.target.1,
                route.source,
                route.sink,
                route.blocks.len()
            ));
            for position in route_terminal_positions(&next_world, &route.path, sink.position) {
                if route_source_set.insert(position) {
                    route_sources.push(position);
                }
            }
            prune_route_sources(
                &mut route_sources,
                &mut route_source_set,
                &all_sinks,
                sink_index + 1,
            );
            route_world = next_world;
            routes.push(route);
        }
    }

    Ok(routes)
}

fn active_route_powers_sink(before: &World3D, after: &World3D, route: &RoutedNet) -> bool {
    if !can_validate_active_route_source(before, route.source) {
        return true;
    }

    let world = World::from(after);
    let Ok(sim) =
        Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
    else {
        return false;
    };
    !block_is_powered(sim.world(), route.source) || block_is_powered(sim.world(), route.sink)
}

fn can_validate_active_route_source(world: &World3D, position: Position) -> bool {
    matches!(
        world[position].kind,
        BlockKind::Torch { .. } | BlockKind::Repeater { .. } | BlockKind::Switch { .. }
    )
}

fn block_is_powered(world: &World3D, position: Position) -> bool {
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

fn route_variable_priority(
    var: &crate::graph::module::GraphModuleVariable,
) -> (usize, usize, &str, &str, &str, &str) {
    let is_clock_route =
        var.source.1.contains("clk") || var.target.1.contains("clk") || var.target.1 == "en";
    let target_port_priority = match (var.target.1.as_str(), is_clock_route) {
        ("d", _) => 0,
        (_, false) => 1,
        ("en", true) => 2,
        _ => 3,
    };
    let source_port_priority = if var.source.1.ends_with("_n") { 0 } else { 1 };
    (
        target_port_priority,
        source_port_priority,
        var.source.0.as_str(),
        var.source.1.as_str(),
        var.target.0.as_str(),
        var.target.1.as_str(),
    )
}

fn route_source_to_target_position(
    world: &World3D,
    source_port: &PhysicalPort,
    logical_source: Position,
    route_source: Position,
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    if source_port.direction == PhysicalPortDirection::Output && source_port.isolate_output {
        return route_isolated_output_to_target_position(
            world,
            logical_source,
            route_source,
            sink,
            same_net_sinks,
            strategy,
        );
    }
    if source_port.direction == PhysicalPortDirection::Input {
        let mut sources = redstone_network_positions(world, &[logical_source]);
        sources.push(route_source);
        return route_to_target_from_network(world, &sources, sink, same_net_sinks, strategy);
    }

    route_to_target_position(world, route_source, sink, same_net_sinks, strategy)
}

fn route_to_target_position(
    world: &World3D,
    source: Position,
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    if sink.isolate_input {
        return route_to_redstone_input_through_repeater(
            world,
            source,
            sink.position,
            same_net_sinks,
            strategy,
        );
    }

    route_point_to_point_with_strategy_and_allowed_contacts(
        world,
        source,
        sink.position,
        strategy,
        same_net_contact_positions(same_net_sinks),
    )
}

fn route_to_target_from_network(
    world: &World3D,
    sources: &[Position],
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let mut sources = sources.to_vec();
    sources.sort_by_key(|source| source.manhattan_distance(&sink.position));
    let fallback_source = sources.first().copied().unwrap_or(sink.position);

    for source in sources.into_iter().take(FANOUT_ROUTE_SOURCE_LIMIT) {
        if !world.size.bound_on(source) || !is_route_terminal(world, source) {
            continue;
        }
        if let Ok((route, next_world)) =
            route_to_target_position(world, source, sink, same_net_sinks, strategy)
        {
            if active_route_powers_sink(world, &next_world, &route) {
                return Ok((route, next_world));
            }
        }
    }

    Err(RouteFailure::Unreachable {
        source: fallback_source,
        sink: sink.position,
    })
}

fn same_net_contact_positions(sinks: &[ResolvedPortTarget]) -> Vec<Position> {
    sinks
        .iter()
        .flat_map(|sink| {
            let mut positions = vec![sink.position];
            positions.extend(sink.position.cardinal());
            positions.push(sink.position.up());
            positions
        })
        .collect()
}

fn route_terminal_positions(world: &World3D, path: &[Position], sink: Position) -> Vec<Position> {
    let mut positions = path
        .iter()
        .copied()
        .filter(|position| world.size.bound_on(*position) && is_route_terminal(world, *position))
        .collect::<Vec<_>>();
    positions.sort_by_key(|position| {
        (
            position.manhattan_distance(&sink),
            std::cmp::Reverse(position.0),
            position.1,
            position.2,
        )
    });
    positions.truncate(FANOUT_ROUTE_TERMINAL_LIMIT);
    positions
}

fn prune_route_sources(
    route_sources: &mut Vec<Position>,
    route_source_set: &mut HashSet<Position>,
    sinks: &[ResolvedPortTarget],
    next_sink_index: usize,
) {
    if route_sources.len() <= FANOUT_ROUTE_TERMINAL_LIMIT {
        return;
    }

    route_sources.sort_by_key(|source| {
        sinks
            .iter()
            .skip(next_sink_index)
            .map(|sink| source.manhattan_distance(&sink.position))
            .min()
            .unwrap_or(0)
    });
    route_sources.truncate(FANOUT_ROUTE_TERMINAL_LIMIT);

    route_source_set.clear();
    route_source_set.extend(route_sources.iter().copied());
}

fn route_isolated_output_to_target_position(
    world: &World3D,
    logical_source: Position,
    route_source: Position,
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let same_net_contacts = same_net_contact_positions(same_net_sinks);
    if sink.isolate_input && sink.module_contains_sequential {
        for (adapter_world, driver_position) in
            redstone_input_repeater_adapters(world, sink.position, &same_net_contacts)
        {
            let Ok((route, routed_world)) = route_direct_output_to_point(
                &adapter_world,
                logical_source,
                route_source,
                driver_position,
                strategy,
                same_net_contacts.clone(),
            ) else {
                continue;
            };
            let route = RoutedNet {
                source: logical_source,
                sink: sink.position,
                blocks: added_route_blocks(world, &routed_world),
                path: route.path,
            };
            if active_route_powers_sink(world, &routed_world, &route) {
                return Ok((route, routed_world));
            }
        }
    }

    route_isolated_output_to_point(
        world,
        logical_source,
        route_source,
        sink.position,
        strategy,
        same_net_contacts,
    )
}

fn route_to_redstone_input_through_repeater(
    world: &World3D,
    source: Position,
    sink: Position,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let same_net_contacts = same_net_contact_positions(same_net_sinks);
    for target in redstone_input_drive_targets(world, source, sink) {
        for (adapter_world, driver_position) in
            redstone_input_repeater_adapters(world, target, &same_net_contacts)
        {
            if target != sink
                && !detailed_router::target_powers_position(&adapter_world, target, sink)
            {
                continue;
            }
            let Ok((route, routed_world)) = route_point_to_point_with_strategy_and_allowed_contacts(
                &adapter_world,
                source,
                driver_position,
                strategy,
                same_net_contacts.clone(),
            ) else {
                continue;
            };
            let route = RoutedNet {
                source,
                sink,
                blocks: added_route_blocks(world, &routed_world),
                path: route.path,
            };
            if active_route_powers_sink(world, &routed_world, &route) {
                return Ok((route, routed_world));
            }
        }
    }

    Err(RouteFailure::Unreachable { source, sink })
}

fn route_direct_output_to_point(
    world: &World3D,
    logical_source: Position,
    route_source: Position,
    sink: Position,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: Vec<Position>,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let source_node = PlacedNode::new(route_source, world[route_source]);
    let initial = RouteSearchState {
        world: world.clone(),
        terminal: route_source,
        route: vec![route_source],
        signal_strength: 2,
        pending_bounds: Some(sorted_route_bounds(
            source_node.propagation_bound(Some(world)),
            world,
            sink,
        )),
    };
    if let Ok((route, routed_world)) = route_point_to_point_from_initial_state(
        world,
        logical_source,
        sink,
        initial,
        strategy,
        &additional_allowed_contacts,
    ) {
        if active_route_powers_sink(world, &routed_world, &route) {
            return Ok((route, routed_world));
        }
    }

    for (tap, signal_strength) in routeable_output_taps(world, route_source, sink) {
        let initial = RouteSearchState {
            world: world.clone(),
            terminal: tap,
            route: vec![tap],
            signal_strength,
            pending_bounds: None,
        };
        let Ok((route, routed_world)) = route_point_to_point_from_initial_state(
            world,
            logical_source,
            sink,
            initial,
            strategy,
            &additional_allowed_contacts,
        ) else {
            continue;
        };
        let route = RoutedNet {
            source: logical_source,
            sink,
            blocks: added_route_blocks(world, &routed_world),
            path: route.path,
        };
        if active_route_powers_sink(world, &routed_world, &route) {
            return Ok((route, routed_world));
        }
    }

    Err(RouteFailure::Unreachable {
        source: logical_source,
        sink,
    })
}

fn redstone_input_drive_targets(
    world: &World3D,
    source: Position,
    sink: Position,
) -> Vec<Position> {
    let seeds = [sink];
    let mut targets = redstone_network_positions(world, &seeds)
        .into_iter()
        .filter(|position| {
            *position == sink || detailed_router::target_powers_position(world, *position, sink)
        })
        .collect::<Vec<_>>();
    targets.sort_by_key(|position| {
        (
            position.manhattan_distance(&source),
            position.0,
            position.1,
            position.2,
        )
    });
    targets
}

fn redstone_input_repeater_adapters(
    world: &World3D,
    sink: Position,
    additional_allowed_contacts: &[Position],
) -> Vec<(World3D, Position)> {
    sink.cardinal()
        .into_iter()
        .filter_map(|repeater_position| {
            let direction = repeater_position.diff(sink).inverse();
            input_repeater_adapter_world(
                world,
                sink,
                repeater_position,
                direction,
                additional_allowed_contacts,
            )
        })
        .collect()
}

fn input_repeater_adapter_world(
    world: &World3D,
    sink: Position,
    repeater_position: Position,
    direction: Direction,
    additional_allowed_contacts: &[Position],
) -> Option<(World3D, Position)> {
    if !world.size.bound_on(repeater_position) || !world[repeater_position].kind.is_air() {
        return None;
    }
    let driver_position = repeater_position.walk(direction)?;
    if !world.size.bound_on(driver_position) || !world[driver_position].kind.is_air() {
        return None;
    }
    let repeater_support_position = repeater_position.down()?;
    let driver_support_position = driver_position.down()?;
    if !world.size.bound_on(repeater_support_position)
        || !world.size.bound_on(driver_support_position)
    {
        return None;
    }

    let mut adapter_world = world.clone();
    place_support_cobble_if_needed(&mut adapter_world, repeater_support_position)?;
    place_support_cobble_if_needed(&mut adapter_world, driver_support_position)?;

    let repeater = PlacedNode::new_repeater(repeater_position, direction);
    if repeater.has_conflict(&adapter_world, &[sink].into_iter().collect()) {
        return None;
    }
    detailed_router::place_node(&mut adapter_world, repeater);
    if adapter_touches_forbidden_existing_signal(
        world,
        &adapter_world,
        repeater_position,
        &adapter_allowed_contacts(additional_allowed_contacts, &[sink, repeater_position]),
    ) {
        return None;
    }
    let driver = PlacedNode::new_redstone(driver_position);
    if driver.has_conflict(&adapter_world, &[repeater_position].into_iter().collect()) {
        return None;
    }
    detailed_router::place_node(&mut adapter_world, driver);
    if adapter_touches_forbidden_existing_signal(
        world,
        &adapter_world,
        driver_position,
        &adapter_allowed_contacts(
            additional_allowed_contacts,
            &[sink, repeater_position, driver_position],
        ),
    ) {
        return None;
    }
    if !detailed_router::target_powers_position(&adapter_world, driver_position, repeater_position)
    {
        return None;
    }
    detailed_router::target_powers_position(&adapter_world, repeater_position, sink)
        .then_some((adapter_world, driver_position))
}

fn adapter_allowed_contacts(
    additional_allowed_contacts: &[Position],
    local_allowed_contacts: &[Position],
) -> Vec<Position> {
    let mut contacts = additional_allowed_contacts.to_vec();
    contacts.extend(local_allowed_contacts.iter().copied());
    contacts.sort();
    contacts.dedup();
    contacts
}

fn adapter_touches_forbidden_existing_signal(
    original_world: &World3D,
    adapter_world: &World3D,
    adapter_position: Position,
    allowed_contacts: &[Position],
) -> bool {
    let allowed_contacts = allowed_contacts.iter().copied().collect::<HashSet<_>>();
    original_world
        .iter_block()
        .into_iter()
        .any(|(position, block)| {
            if allowed_contacts.contains(&position) || !is_signal_terminal_block(block) {
                return false;
            }
            detailed_router::target_powers_position(adapter_world, adapter_position, position)
                || detailed_router::target_powers_position(
                    adapter_world,
                    position,
                    adapter_position,
                )
        })
}

fn place_support_cobble_if_needed(world: &mut World3D, position: Position) -> Option<()> {
    if world[position].kind.is_air() {
        let support = PlacedNode::new_cobble(position);
        if support.has_conflict(world, &HashSet::new()) {
            return None;
        }
        detailed_router::place_node(world, support);
    } else if !world[position].kind.is_cobble() {
        return None;
    }
    Some(())
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

pub fn collect_module_input_endpoints(
    module: &GraphModule,
    routed_nets: &[RoutedNet],
) -> Vec<OutputEndpoint> {
    let input_ports = module.ports.iter().filter(|port| port.port_type.is_input());
    let input_switches = routed_nets.iter().filter_map(|route| {
        (route.source == route.sink
            && route
                .blocks
                .first()
                .is_some_and(|(_, block)| block.kind.is_switch()))
        .then_some(route.source)
    });

    input_ports
        .zip(input_switches)
        .map(|(port, position)| OutputEndpoint::new(port.name.clone(), position))
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
) -> Vec<ResolvedPortTarget> {
    match target {
        GraphModulePortTarget::Module(module_name, port_name) => {
            resolve_port_targets(candidates, placed_modules, module_name, port_name)
        }
        GraphModulePortTarget::Wire(targets) => targets
            .iter()
            .flat_map(|(module_name, port_name)| {
                resolve_port_targets(candidates, placed_modules, module_name, port_name)
            })
            .collect(),
        GraphModulePortTarget::Node(_) => Vec::new(),
    }
}

fn external_switch_position(
    world: &World3D,
    index: usize,
    sinks: &[ResolvedPortTarget],
) -> Option<Position> {
    for position in external_switch_candidates_outside_layout(world, sinks) {
        if world.size.bound_on(position) && world[position].kind.is_air() {
            return Some(position);
        }
    }

    for position in external_switch_candidates_for_sinks(sinks) {
        if world.size.bound_on(position) && world[position].kind.is_air() {
            return Some(position);
        }
    }

    let max_x = world
        .iter_block()
        .into_iter()
        .map(|(position, _)| position.0)
        .max()
        .unwrap_or(0);
    let position = Position(max_x + 2, index * 3 + 1, 1);
    (world.size.bound_on(position) && world[position].kind.is_air()).then_some(position)
}

fn external_switch_candidates_outside_layout(
    world: &World3D,
    sinks: &[ResolvedPortTarget],
) -> Vec<Position> {
    let Some(center) = sink_center_position(sinks) else {
        return Vec::new();
    };
    let Some((min, max)) = occupied_bounds(world) else {
        return Vec::new();
    };

    let mut candidates = Vec::new();
    for distance in 2..=12 {
        candidates.push(Position(max.0 + distance, center.1, center.2));
        candidates.push(Position(center.0, max.1 + distance, center.2));
        if let Some(x) = min.0.checked_sub(distance) {
            candidates.push(Position(x, center.1, center.2));
        }
        if let Some(y) = min.1.checked_sub(distance) {
            candidates.push(Position(center.0, y, center.2));
        }
    }

    candidates.sort_by_key(|position| {
        let total_distance = sinks
            .iter()
            .map(|sink| position.manhattan_distance(&sink.position))
            .sum::<usize>();
        let x_outside = position.0 > max.0 || position.0 < min.0;
        (
            usize::from(!x_outside),
            total_distance,
            position.0,
            position.1,
            position.2,
        )
    });
    candidates
}

fn occupied_bounds(world: &World3D) -> Option<(Position, Position)> {
    let mut blocks = world
        .iter_block()
        .into_iter()
        .filter(|(_, block)| !block.kind.is_air())
        .map(|(position, _)| position);
    let first = blocks.next()?;
    let mut min = first;
    let mut max = first;
    for position in blocks {
        min.0 = min.0.min(position.0);
        min.1 = min.1.min(position.1);
        min.2 = min.2.min(position.2);
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }
    Some((min, max))
}

fn external_switch_candidates_for_sinks(sinks: &[ResolvedPortTarget]) -> Vec<Position> {
    let mut candidates = sinks
        .iter()
        .flat_map(|sink| external_switch_candidates_near_sink(sink.position))
        .collect::<Vec<_>>();

    if let Some(center) = sink_center_position(sinks) {
        candidates.extend(external_switch_candidates_near_sink(center));
    }

    candidates.sort_by_key(|position| {
        let max_distance = sinks
            .iter()
            .map(|sink| position.manhattan_distance(&sink.position))
            .max()
            .unwrap_or(0);
        let total_distance = sinks
            .iter()
            .map(|sink| position.manhattan_distance(&sink.position))
            .sum::<usize>();
        let edge_penalty = usize::from(position.0 < 2 || position.1 < 2) * 16;
        (
            max_distance + edge_penalty,
            total_distance + edge_penalty,
            std::cmp::Reverse(position.0),
            std::cmp::Reverse(position.1),
            position.2,
        )
    });
    candidates.dedup();
    candidates
}

fn sink_center_position(sinks: &[ResolvedPortTarget]) -> Option<Position> {
    let first = sinks.first()?;
    let mut min = first.position;
    let mut max = first.position;
    for sink in sinks {
        min.0 = min.0.min(sink.position.0);
        min.1 = min.1.min(sink.position.1);
        min.2 = min.2.min(sink.position.2);
        max.0 = max.0.max(sink.position.0);
        max.1 = max.1.max(sink.position.1);
        max.2 = max.2.max(sink.position.2);
    }
    Some(Position(
        (min.0 + max.0) / 2,
        (min.1 + max.1) / 2,
        (min.2 + max.2) / 2,
    ))
}

fn external_switch_candidates_near_sink(sink: Position) -> Vec<Position> {
    let mut candidates = Vec::new();
    for distance in 4..=8 {
        candidates.extend([
            Position(sink.0 + distance, sink.1, sink.2),
            Position(sink.0, sink.1 + distance, sink.2),
        ]);
        if let Some(x) = sink.0.checked_sub(distance) {
            candidates.push(Position(x, sink.1, sink.2));
        }
        if let Some(y) = sink.1.checked_sub(distance) {
            candidates.push(Position(sink.0, y, sink.2));
        }
    }
    candidates
}

fn input_switch_block() -> Block {
    Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::West,
    }
}

fn resolve_port_targets(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    module_name: &str,
    port_name: &str,
) -> Vec<ResolvedPortTarget> {
    let Some((port, candidate, placed)) =
        resolve_port(candidates, placed_modules, module_name, port_name)
    else {
        return Vec::new();
    };

    let position = port.position;
    vec![ResolvedPortTarget {
        position: translate_candidate_position(position, candidate, placed),
        isolate_input: port.isolate_input,
        module_contains_sequential: port.module_contains_sequential,
    }]
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
                && detailed_router::target_powers_position(&candidate.world, position, *tap)
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
    route_point_to_point_with_strategy(world, source, sink, GlobalRoutingStrategy::BreadthFirst)
}

pub fn route_point_to_point_with_strategy(
    world: &World3D,
    source: Position,
    sink: Position,
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    route_point_to_point_with_strategy_and_allowed_contacts(
        world,
        source,
        sink,
        strategy,
        Vec::new(),
    )
}

fn route_point_to_point_with_strategy_and_allowed_contacts(
    world: &World3D,
    source: Position,
    sink: Position,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: Vec<Position>,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let goal = RouteGoal::for_sink(world, sink);
    route_point_to_point_with_bounds(
        world,
        source,
        sink,
        goal,
        BoundSearchMode::Propagation,
        strategy,
        &additional_allowed_contacts,
    )
    .or_else(|_| {
        route_point_to_point_with_bounds(
            world,
            source,
            sink,
            goal,
            BoundSearchMode::Nearby,
            strategy,
            &additional_allowed_contacts,
        )
    })
}

fn route_isolated_output_to_point(
    world: &World3D,
    logical_source: Position,
    route_source: Position,
    sink: Position,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: Vec<Position>,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let initial_states = isolated_output_repeater_initial_states(
        world,
        logical_source,
        route_source,
        sink,
        &additional_allowed_contacts,
    );
    for initial in initial_states {
        let Ok((route, routed_world)) = route_point_to_point_from_initial_state(
            world,
            logical_source,
            sink,
            initial,
            strategy,
            &additional_allowed_contacts,
        ) else {
            continue;
        };
        if active_route_powers_sink(world, &routed_world, &route) {
            return Ok((route, routed_world));
        }
    }

    Err(RouteFailure::Unreachable {
        source: logical_source,
        sink,
    })
}

fn routeable_output_taps(
    world: &World3D,
    source: Position,
    sink: Position,
) -> Vec<(Position, usize)> {
    let direct_taps = world
        .iter_block()
        .into_iter()
        .filter(|(position, block)| {
            block.kind.is_redstone()
                && detailed_router::target_powers_position(world, source, *position)
        })
        .map(|(position, _)| position)
        .collect::<Vec<_>>();

    let mut taps = powered_redstone_network_taps(world, &direct_taps);
    taps.sort_by_key(|(position, strength)| {
        (
            position.manhattan_distance(&sink),
            std::cmp::Reverse(*strength),
            position.manhattan_distance(&source),
            position.0,
            position.1,
            position.2,
        )
    });
    taps
}

fn powered_redstone_network_taps(world: &World3D, seeds: &[Position]) -> Vec<(Position, usize)> {
    let redstones = world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| block.kind.is_redstone().then_some(position))
        .collect::<Vec<_>>();
    let mut strengths = HashMap::<Position, usize>::new();
    let mut queue = VecDeque::new();
    for &seed in seeds {
        if strengths
            .insert(seed, MAX_REDSTONE_STRENGTH)
            .is_none_or(|old| old < MAX_REDSTONE_STRENGTH)
        {
            queue.push_back(seed);
        }
    }

    while let Some(position) = queue.pop_front() {
        let Some(&strength) = strengths.get(&position) else {
            continue;
        };
        if strength <= 1 {
            continue;
        }

        for &next in &redstones {
            if position == next {
                continue;
            }
            if !detailed_router::target_powers_position(world, position, next)
                && !detailed_router::target_powers_position(world, next, position)
            {
                continue;
            }

            let next_strength = strength - 1;
            if strengths.get(&next).is_none_or(|old| *old < next_strength) {
                strengths.insert(next, next_strength);
                queue.push_back(next);
            }
        }
    }

    strengths.into_iter().collect()
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

fn isolated_output_repeater_initial_states(
    world: &World3D,
    logical_source: Position,
    route_source: Position,
    sink: Position,
    additional_allowed_contacts: &[Position],
) -> Vec<RouteSearchState> {
    let mut seeds = vec![(route_source, initial_signal_strength(world, route_source))];
    seeds.extend(routeable_output_taps(world, route_source, sink));
    seeds.sort_by_key(|(position, strength)| {
        (
            position.manhattan_distance(&sink),
            std::cmp::Reverse(*strength),
            position.0,
            position.1,
            position.2,
        )
    });
    seeds.dedup_by_key(|(position, _)| *position);

    let mut states = Vec::new();
    let mut output_allowed_contacts = additional_allowed_contacts.to_vec();
    output_allowed_contacts.extend(source_signal_positions(world, logical_source));
    output_allowed_contacts.extend(source_signal_positions(world, route_source));
    output_allowed_contacts.sort();
    output_allowed_contacts.dedup();

    for (seed, _) in seeds {
        if !world.size.bound_on(seed) || !is_route_terminal(world, seed) {
            continue;
        }

        for (adapter_world, driver_position) in
            output_repeater_adapters(world, seed, &output_allowed_contacts)
        {
            states.push(RouteSearchState {
                world: adapter_world,
                terminal: driver_position,
                route: vec![seed, driver_position],
                signal_strength: MAX_REDSTONE_STRENGTH,
                pending_bounds: None,
            });
        }

        states.extend(escaped_output_repeater_initial_states(
            world,
            seed,
            sink,
            &output_allowed_contacts,
        ));
    }

    states
}

fn escaped_output_repeater_initial_states(
    world: &World3D,
    seed: Position,
    sink: Position,
    output_allowed_contacts: &[Position],
) -> Vec<RouteSearchState> {
    let goal = RouteGoal::for_sink(world, sink);
    let forbidden_signal_contacts =
        route_forbidden_signal_contact_positions(world, seed, goal, output_allowed_contacts);
    let allowed_shorts = output_allowed_contacts
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut states = Vec::new();
    let mut visited = HashSet::from([seed]);
    let mut queue = VecDeque::from([RouteSearchState {
        world: world.clone(),
        terminal: seed,
        route: vec![seed],
        signal_strength: initial_signal_strength(world, seed),
        pending_bounds: None,
    }]);

    while let Some(state) = queue.pop_front() {
        let escape_steps = state.route.len().saturating_sub(1);
        if escape_steps > 0 {
            for (adapter_world, driver_position) in
                output_repeater_adapters(&state.world, state.terminal, output_allowed_contacts)
            {
                let mut route = state.route.clone();
                route.push(driver_position);
                states.push(RouteSearchState {
                    world: adapter_world,
                    terminal: driver_position,
                    route,
                    signal_strength: MAX_REDSTONE_STRENGTH,
                    pending_bounds: None,
                });
            }
        }

        if escape_steps >= OUTPUT_ISOLATION_ESCAPE_MAX_STEPS || state.signal_strength <= 1 {
            continue;
        }

        let terminal_node = PlacedNode::new(state.terminal, state.world[state.terminal]);
        for bound in
            route_bounds_for_mode(BoundSearchMode::Propagation, &state.world, &terminal_node)
        {
            if !bound.is_bound_on(&state.world) || goal.rejects_bound_position(bound.position()) {
                continue;
            }

            let PlaceRedstoneResult::Placed(next_world, redstone_node) =
                detailed_router::place_redstone_with_cobble_and_allowed_shorts(
                    &state.world,
                    bound,
                    state.terminal,
                    sink,
                    Some(&allowed_shorts),
                )
            else {
                continue;
            };

            if route_touches_forbidden_existing_signal(
                &next_world,
                redstone_node.position,
                &forbidden_signal_contacts,
            ) {
                continue;
            }

            if visited.insert(redstone_node.position) {
                let mut route = state.route.clone();
                route.push(redstone_node.position);
                queue.push_back(RouteSearchState {
                    world: next_world,
                    terminal: redstone_node.position,
                    route,
                    signal_strength: state.signal_strength - 1,
                    pending_bounds: None,
                });
            }
        }
    }

    states
}

fn output_repeater_adapters(
    world: &World3D,
    source: Position,
    additional_allowed_contacts: &[Position],
) -> Vec<(World3D, Position)> {
    source
        .cardinal()
        .into_iter()
        .filter_map(|repeater_position| {
            let direction = source.diff(repeater_position).inverse();
            output_repeater_adapter_world(
                world,
                source,
                repeater_position,
                direction,
                additional_allowed_contacts,
            )
        })
        .collect()
}

fn output_repeater_adapter_world(
    world: &World3D,
    source: Position,
    repeater_position: Position,
    direction: Direction,
    additional_allowed_contacts: &[Position],
) -> Option<(World3D, Position)> {
    if !world.size.bound_on(repeater_position) || !world[repeater_position].kind.is_air() {
        return None;
    }
    let driver_position = repeater_position.walk(direction.inverse())?;
    if !world.size.bound_on(driver_position) || !world[driver_position].kind.is_air() {
        return None;
    }
    let repeater_support_position = repeater_position.down()?;
    let driver_support_position = driver_position.down()?;
    if !world.size.bound_on(repeater_support_position)
        || !world.size.bound_on(driver_support_position)
    {
        return None;
    }

    let mut adapter_world = world.clone();
    place_support_cobble_if_needed(&mut adapter_world, repeater_support_position)?;
    place_support_cobble_if_needed(&mut adapter_world, driver_support_position)?;

    let repeater = PlacedNode::new_repeater(repeater_position, direction);
    if repeater.has_conflict(&adapter_world, &[source].into_iter().collect()) {
        return None;
    }
    detailed_router::place_node(&mut adapter_world, repeater);
    if !detailed_router::target_powers_position(&adapter_world, source, repeater_position) {
        return None;
    }
    if adapter_touches_forbidden_existing_signal(
        world,
        &adapter_world,
        repeater_position,
        &adapter_allowed_contacts(additional_allowed_contacts, &[source, repeater_position]),
    ) {
        return None;
    }

    let driver = PlacedNode::new_redstone(driver_position);
    if driver.has_conflict(&adapter_world, &[repeater_position].into_iter().collect()) {
        return None;
    }
    detailed_router::place_node(&mut adapter_world, driver);
    if !detailed_router::target_powers_position(&adapter_world, repeater_position, driver_position)
    {
        return None;
    }
    if adapter_touches_forbidden_existing_signal(
        world,
        &adapter_world,
        driver_position,
        &adapter_allowed_contacts(
            additional_allowed_contacts,
            &[source, repeater_position, driver_position],
        ),
    ) {
        return None;
    }

    Some((adapter_world, driver_position))
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
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: &[Position],
) -> Result<(RoutedNet, World3D), RouteFailure> {
    route_point_to_point_with_bounds_and_min_z(
        world,
        source,
        sink,
        goal,
        mode,
        None,
        strategy,
        additional_allowed_contacts,
    )
}

fn route_point_to_point_with_bounds_and_min_z(
    world: &World3D,
    source: Position,
    sink: Position,
    goal: RouteGoal,
    mode: BoundSearchMode,
    min_z: Option<usize>,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: &[Position],
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let initial_strength = initial_signal_strength(world, source);
    route_point_to_point_with_initial_queue(
        world,
        source,
        sink,
        goal,
        mode,
        min_z,
        vec![RouteSearchState {
            world: world.clone(),
            terminal: source,
            route: vec![source],
            signal_strength: initial_strength,
            pending_bounds: None,
        }],
        strategy,
        additional_allowed_contacts,
    )
}

fn route_point_to_point_from_initial_state(
    world: &World3D,
    source: Position,
    sink: Position,
    initial_state: RouteSearchState,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: &[Position],
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let goal = RouteGoal::for_sink(world, sink);
    route_point_to_point_with_initial_queue(
        world,
        source,
        sink,
        goal,
        BoundSearchMode::Propagation,
        None,
        vec![initial_state.clone()],
        strategy,
        additional_allowed_contacts,
    )
    .or_else(|_| {
        route_point_to_point_with_initial_queue(
            world,
            source,
            sink,
            goal,
            BoundSearchMode::Nearby,
            None,
            vec![initial_state],
            strategy,
            additional_allowed_contacts,
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
    initial_states: Vec<RouteSearchState>,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: &[Position],
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let initial_visited = initial_states
        .iter()
        .map(|state| route_visited_key(strategy, state));
    let mut visited = initial_visited.collect::<HashSet<_>>();
    let mut queue = RouteSearchQueue::new(strategy, sink, initial_states);
    let forbidden_signal_contacts = route_forbidden_signal_contact_positions(
        original_world,
        source,
        goal,
        additional_allowed_contacts,
    );
    let mut expansions = 0usize;

    while let Some(state) = queue.pop() {
        expansions += 1;
        if route_expansion_limit(strategy).is_some_and(|limit| expansions > limit) {
            break;
        }

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
                    path: state.route,
                },
                state.world,
            ));
        }

        if state.route.len() > GLOBAL_ROUTE_MAX_STEPS {
            continue;
        }

        let terminal_node = PlacedNode::new(state.terminal, state.world[state.terminal]);
        let allowed_shorts = goal.allowed_short_positions();
        let mut bounds = state
            .pending_bounds
            .unwrap_or_else(|| route_bounds_for_mode(mode, &state.world, &terminal_node));
        bounds.sort_by_key(|bound| {
            let position = bound.position();
            (
                position.manhattan_distance(&sink),
                position.0,
                position.1,
                position.2,
            )
        });
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
                        if route_touches_forbidden_existing_signal(
                            &next_world,
                            redstone_node.position,
                            &forbidden_signal_contacts,
                        ) {
                            continue;
                        }
                        let next_strength = state.signal_strength - 1;
                        if visited.insert((
                            redstone_node.position,
                            route_visited_depth(strategy, state.route.len()),
                            next_strength,
                        )) {
                            let mut next_route = state.route.clone();
                            next_route.push(redstone_node.position);
                            queue.push(RouteSearchState {
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
                            if route_touches_forbidden_existing_signal(
                                &next_world,
                                repeater_node.position,
                                &forbidden_signal_contacts,
                            ) {
                                continue;
                            }
                            if visited.insert((
                                repeater_node.position,
                                route_visited_depth(strategy, state.route.len()),
                                MAX_REDSTONE_STRENGTH,
                            )) {
                                let mut next_route = state.route.clone();
                                next_route.push(repeater_node.position);
                                queue.push(RouteSearchState {
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

fn route_forbidden_signal_contact_positions(
    world: &World3D,
    source: Position,
    goal: RouteGoal,
    additional_allowed_contacts: &[Position],
) -> Vec<Position> {
    let mut allowed = HashSet::from([source, goal.placement_target()]);
    allowed.extend(additional_allowed_contacts.iter().copied());
    allowed.extend(source_signal_positions(world, source));
    allowed.extend(goal_contact_positions(world, goal));
    world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| {
            (!allowed.contains(&position) && is_signal_terminal_block(block)).then_some(position)
        })
        .collect()
}

fn source_signal_positions(world: &World3D, source: Position) -> HashSet<Position> {
    let mut positions = HashSet::from([source]);
    if !world.size.bound_on(source) {
        return positions;
    }
    if world[source].kind.is_redstone() {
        positions.extend(redstone_network_positions(world, &[source]));
    }

    let direct_taps = world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| {
            (block.kind.is_redstone()
                && detailed_router::target_powers_position(world, source, position))
            .then_some(position)
        })
        .collect::<Vec<_>>();
    positions.extend(direct_taps.iter().copied());
    positions.extend(
        powered_redstone_network_taps(world, &direct_taps)
            .into_iter()
            .map(|(position, _)| position),
    );
    positions
}

fn goal_contact_positions(world: &World3D, goal: RouteGoal) -> HashSet<Position> {
    let target = goal.placement_target();
    let mut positions = HashSet::from([target]);
    positions.extend(
        target
            .cardinal()
            .into_iter()
            .filter(|position| world.size.bound_on(*position)),
    );
    let top = target.up();
    if world.size.bound_on(top) {
        positions.insert(top);
    }
    positions
}

fn route_touches_forbidden_existing_signal(
    routed_world: &World3D,
    route_position: Position,
    forbidden_contacts: &[Position],
) -> bool {
    forbidden_contacts.iter().copied().any(|position| {
        detailed_router::target_powers_position(routed_world, route_position, position)
            || detailed_router::target_powers_position(routed_world, position, route_position)
    })
}

fn is_signal_terminal_block(block: Block) -> bool {
    block.kind.is_redstone()
        || block.kind.is_switch()
        || block.kind.is_torch()
        || block.kind.is_repeater()
        || matches!(block.kind, BlockKind::RedstoneBlock)
}

fn route_expansion_limit(strategy: GlobalRoutingStrategy) -> Option<usize> {
    match strategy {
        GlobalRoutingStrategy::BreadthFirst => None,
        GlobalRoutingStrategy::AStar => Some(GLOBAL_ROUTE_ASTAR_MAX_EXPANSIONS),
    }
}

#[derive(Clone)]
struct RouteSearchState {
    world: World3D,
    terminal: Position,
    route: Vec<Position>,
    signal_strength: usize,
    pending_bounds: Option<Vec<PlaceBound>>,
}

fn route_visited_key(
    strategy: GlobalRoutingStrategy,
    state: &RouteSearchState,
) -> (Position, usize, usize) {
    (
        state.terminal,
        route_visited_depth(strategy, state.route.len().saturating_sub(1)),
        state.signal_strength,
    )
}

fn route_visited_depth(strategy: GlobalRoutingStrategy, route_depth: usize) -> usize {
    match strategy {
        GlobalRoutingStrategy::BreadthFirst => route_depth,
        GlobalRoutingStrategy::AStar => 0,
    }
}

enum RouteSearchQueue {
    BreadthFirst(VecDeque<RouteSearchState>),
    AStar {
        heap: BinaryHeap<AStarQueueEntry>,
        next_sequence: usize,
        sink: Position,
    },
}

impl RouteSearchQueue {
    fn new(
        strategy: GlobalRoutingStrategy,
        sink: Position,
        initial_states: Vec<RouteSearchState>,
    ) -> Self {
        match strategy {
            GlobalRoutingStrategy::BreadthFirst => Self::BreadthFirst(initial_states.into()),
            GlobalRoutingStrategy::AStar => {
                let mut queue = Self::AStar {
                    heap: BinaryHeap::new(),
                    next_sequence: 0,
                    sink,
                };
                for state in initial_states {
                    queue.push(state);
                }
                queue
            }
        }
    }

    fn pop(&mut self) -> Option<RouteSearchState> {
        match self {
            Self::BreadthFirst(queue) => queue.pop_front(),
            Self::AStar { heap, .. } => heap.pop().map(|entry| entry.state),
        }
    }

    fn push(&mut self, state: RouteSearchState) {
        match self {
            Self::BreadthFirst(queue) => queue.push_back(state),
            Self::AStar {
                heap,
                next_sequence,
                sink,
            } => {
                heap.push(AStarQueueEntry::new(state, *sink, *next_sequence));
                *next_sequence += 1;
            }
        }
    }
}

struct AStarQueueEntry {
    priority: AStarPriority,
    state: RouteSearchState,
}

impl AStarQueueEntry {
    fn new(state: RouteSearchState, sink: Position, sequence: usize) -> Self {
        Self {
            priority: AStarPriority::new(&state, sink, sequence),
            state,
        }
    }
}

impl Ord for AStarQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for AStarQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for AStarQueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for AStarQueueEntry {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AStarPriority {
    estimated_total_cost: usize,
    route_len: usize,
    manhattan_to_sink: usize,
    low_strength_penalty: usize,
    sequence: usize,
}

impl AStarPriority {
    fn new(state: &RouteSearchState, sink: Position, sequence: usize) -> Self {
        let route_len = state.route.len().saturating_sub(1);
        let manhattan_to_sink = state.terminal.manhattan_distance(&sink);
        let low_strength_penalty = usize::from(state.signal_strength <= 2) * 4;
        Self {
            estimated_total_cost: route_len + manhattan_to_sink + low_strength_penalty,
            route_len,
            manhattan_to_sink,
            low_strength_penalty,
            sequence,
        }
    }
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
                detailed_router::target_powers_position(world, target, redstone)
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
        .filter(|(position, block)| !block.kind.is_air() && before[*position] != *block)
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
                isolate_input: false,
                isolate_output: false,
                module_contains_sequential: false,
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

    fn silent_progress() -> GlobalPnrProgress {
        GlobalPnrProgress::new(false, "test")
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
    fn route_point_to_point_astar_handles_long_unpowered_redstone_route() {
        let source = Position(0, 1, 1);
        let sink = Position(44, 1, 1);
        let world = route_test_world_with_size(source, sink, DimSize(48, 4, 3));
        let (route, _) =
            route_point_to_point_with_strategy(&world, source, sink, GlobalRoutingStrategy::AStar)
                .unwrap();

        assert!(
            route
                .blocks
                .iter()
                .filter(|(_, block)| block.kind.is_repeater())
                .count()
                >= 2
        );
    }

    #[test]
    fn route_point_to_point_astar_handles_counter_carry_like_coordinates() {
        let source = Position(26, 10, 3);
        let sink = Position(70, 4, 3);
        let world = route_test_world_with_size(source, sink, DimSize(76, 16, 6));
        let (route, _) =
            route_point_to_point_with_strategy(&world, source, sink, GlobalRoutingStrategy::AStar)
                .unwrap();

        assert!(route
            .blocks
            .iter()
            .any(|(_, block)| block.kind.is_repeater()));
    }

    #[test]
    fn route_to_redstone_input_handles_counter_fanout_like_coordinates() {
        let source = Position(25, 10, 3);
        let sink = Position(41, 8, 3);
        let world = route_test_world_with_size(source, sink, DimSize(48, 16, 6));

        route_to_target_position(
            &world,
            source,
            ResolvedPortTarget {
                position: sink,
                isolate_input: true,
                module_contains_sequential: true,
            },
            &[],
            GlobalRoutingStrategy::AStar,
        )
        .unwrap();
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
    fn route_to_redstone_input_finishes_with_repeater_diode() -> eyre::Result<()> {
        let source = Position(10, 1, 1);
        let sink = Position(1, 1, 1);
        let world = route_test_world_with_switch_source(source, sink, DimSize(13, 4, 3));

        let (route, routed_world) = route_to_target_position(
            &world,
            source,
            ResolvedPortTarget {
                position: sink,
                isolate_input: true,
                module_contains_sequential: true,
            },
            &[],
            GlobalRoutingStrategy::BreadthFirst,
        )
        .unwrap();

        assert!(
            route
                .blocks
                .iter()
                .any(|(_, block)| block.kind.is_repeater()),
            "redstone input routes should end through a repeater diode"
        );
        assert!(
            sink.cardinal()
                .into_iter()
                .any(|position| routed_world.size.bound_on(position)
                    && routed_world[position].kind.is_repeater()
                    && detailed_router::target_powers_position(&routed_world, position, sink)),
            "the final repeater should power the input redstone"
        );

        let world = World::from(&routed_world);
        let mut sim = Simulator::from_with_limits_and_trace(&world, 128, 20_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        sim.change_state_with_limits(vec![(source, true)], 128, 20_000)?;

        assert!(
            matches!(sim.world()[sink].kind, BlockKind::Redstone { strength, .. } if strength > 0),
            "input redstone should be powered through the repeater diode"
        );
        Ok(())
    }

    #[test]
    fn route_to_cobble_input_finishes_with_repeater_diode() -> eyre::Result<()> {
        let source = Position(10, 1, 1);
        let sink = Position(1, 1, 1);
        let mut world = World3D::new(DimSize(13, 4, 3));
        world[source] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[sink] = cobble_block();
        world.initialize_redstone_states();

        let (route, routed_world) = route_to_target_position(
            &world,
            source,
            ResolvedPortTarget {
                position: sink,
                isolate_input: true,
                module_contains_sequential: true,
            },
            &[],
            GlobalRoutingStrategy::BreadthFirst,
        )
        .unwrap();

        assert!(
            route
                .blocks
                .iter()
                .any(|(_, block)| block.kind.is_repeater()),
            "cobble input routes should end through a repeater diode"
        );
        assert!(
            sink.cardinal()
                .into_iter()
                .any(|position| routed_world.size.bound_on(position)
                    && routed_world[position].kind.is_repeater()
                    && detailed_router::target_powers_position(&routed_world, position, sink)),
            "the final repeater should power the input cobble"
        );

        let world = World::from(&routed_world);
        let mut sim = Simulator::from_with_limits_and_trace(&world, 128, 20_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        sim.change_state_with_limits(vec![(source, true)], 128, 20_000)?;

        assert!(
            matches!(sim.world()[sink].kind, BlockKind::Cobble { on_count, .. } if on_count > 0),
            "input cobble should be powered through the repeater diode"
        );
        Ok(())
    }

    #[test]
    fn route_point_to_point_does_not_touch_existing_signal_line() {
        let source = Position(0, 1, 1);
        let sink = Position(6, 1, 1);
        let protected = Position(3, 2, 1);
        let mut world = route_test_world_with_switch_source(source, sink, DimSize(9, 5, 3));
        world[protected.down().unwrap()] = cobble_block();
        world[protected] = redstone_block();
        world.initialize_redstone_states();

        let (_, routed_world) =
            route_point_to_point_with_strategy(&world, source, sink, GlobalRoutingStrategy::AStar)
                .unwrap();

        for (position, block) in routed_world.iter_block() {
            if !block.kind.is_redstone() && !block.kind.is_repeater() {
                continue;
            }
            if position == protected || position == source || position == sink {
                continue;
            }
            assert!(
                !detailed_router::target_powers_position(&routed_world, position, protected)
                    && !detailed_router::target_powers_position(
                        &routed_world,
                        protected,
                        position
                    ),
                "route block {position:?} should not electrically touch protected line {protected:?}"
            );
        }
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
                ..Default::default()
            },
        );

        let progress = silent_progress();
        let routes = route_module_variables(
            &module,
            &candidates,
            &placed,
            &GlobalRoutingConfig::default(),
            &progress,
        )?;

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

        let progress = silent_progress();
        let routes = route_module_variables(
            &module,
            &candidates,
            &placed,
            &GlobalRoutingConfig::default(),
            &progress,
        )?;

        assert!(routes.is_empty());
        Ok(())
    }

    #[test]
    fn route_point_to_point_supports_astar_strategy() {
        let source = Position(0, 1, 1);
        let sink = Position(22, 1, 1);
        let world = route_test_world_with_size(source, sink, DimSize(26, 4, 3));
        let (route, _) =
            route_point_to_point_with_strategy(&world, source, sink, GlobalRoutingStrategy::AStar)
                .unwrap();

        assert!(route
            .blocks
            .iter()
            .any(|(_, block)| block.kind.is_repeater()));
    }
}
