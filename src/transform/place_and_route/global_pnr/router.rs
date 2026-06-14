use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use eyre::ContextCompat;

use crate::graph::module::{GraphModule, GraphModulePortTarget, GraphModuleVariable};
use crate::output::OutputEndpoint;
use crate::transform::place_and_route::detailed_router::{
    self, PlaceRedstoneResult, PlaceRepeaterResult,
};
use crate::transform::place_and_route::global_pnr::assembly::reset_dynamic_power_states;
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
const OUTPUT_ISOLATION_ESCAPE_MAX_STEPS: usize = 4;
const SIGNAL_CONTACT_SEARCH_RADIUS: usize = 3;

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
    requires_input_diode: bool,
}

#[derive(Clone)]
struct InputDiodeAdapter {
    world: World3D,
    driver: Position,
    repeater: Position,
    target: Position,
}

#[derive(Clone)]
struct ExternalInputSource {
    world: World3D,
    switch: Position,
    route_source: Position,
    blocks: Vec<(Position, Block)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoweredRouteSource {
    position: Position,
    strength: usize,
}

#[derive(Clone, Debug)]
pub struct RoutedNet {
    pub source: Position,
    pub sink: Position,
    pub blocks: Vec<(Position, Block)>,
    pub path: Vec<Position>,
    pub required_powered_positions: Vec<Position>,
    pub required_released_positions: Vec<Position>,
    pub powered_taps: Vec<(Position, usize)>,
}

impl RoutedNet {
    fn new(
        source: Position,
        sink: Position,
        blocks: Vec<(Position, Block)>,
        path: Vec<Position>,
    ) -> Self {
        let powered_taps = path
            .last()
            .copied()
            .map(|position| (position, MAX_REDSTONE_STRENGTH))
            .into_iter()
            .collect();
        Self {
            source,
            sink,
            blocks,
            path,
            required_powered_positions: vec![sink],
            required_released_positions: vec![sink],
            powered_taps,
        }
    }

    fn with_required_powered_positions(mut self, positions: Vec<Position>) -> Self {
        self.required_powered_positions = positions.clone();
        self.required_released_positions = positions;
        self
    }

    fn with_required_released_positions(mut self, positions: Vec<Position>) -> Self {
        self.required_released_positions = positions;
        self
    }

    fn with_powered_taps(mut self, taps: Vec<PoweredRouteSource>) -> Self {
        self.powered_taps = taps
            .into_iter()
            .map(|source| (source.position, source.strength))
            .collect();
        self
    }

    fn powered_route_sources(&self) -> Vec<PoweredRouteSource> {
        self.powered_taps
            .iter()
            .map(|(position, strength)| PoweredRouteSource {
                position: *position,
                strength: *strength,
            })
            .collect()
    }
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

    let mut vars = module.vars.iter().collect::<Vec<_>>();
    vars.sort_by_key(|var| route_variable_priority(var));

    route_top_input_ports(
        module,
        candidates,
        placed_modules,
        config,
        progress,
        &mut route_world,
        &mut routes,
    )?;

    route_internal_module_nets(
        &vars,
        candidates,
        placed_modules,
        config,
        progress,
        &mut route_world,
        &mut routes,
    )?;

    if let Some(route) = first_invalid_active_route(&route_world, &routes) {
        return Err(eyre::eyre!(
            "routed net from {:?} to {:?} no longer powers its sink in the final routed world",
            route.source,
            route.sink,
        ));
    }

    Ok(routes)
}

fn route_internal_module_nets(
    vars: &[&GraphModuleVariable],
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    config: &GlobalRoutingConfig,
    progress: &GlobalPnrProgress,
    route_world: &mut World3D,
    routes: &mut Vec<RoutedNet>,
) -> eyre::Result<()> {
    let grouped_vars = group_vars_by_source_ordered(vars);
    for (group_index, group_vars) in grouped_vars.iter().enumerate() {
        let source_key = group_vars[0].source.clone();

        let (source_port, source_candidate, source_placed) =
            resolve_port(candidates, placed_modules, &source_key.0, &source_key.1).with_context(
                || {
                    format!(
                        "source port {}.{} is not placed",
                        source_key.0, source_key.1
                    )
                },
            )?;
        let source_positions =
            translate_port_access_positions(source_port, source_candidate, source_placed);
        let source = source_positions.first().copied().unwrap_or_else(|| {
            translate_port_route_position(source_port, source_candidate, source_placed)
        });
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
        let all_sinks = sink_targets
            .iter()
            .map(|(_, sink)| *sink)
            .collect::<Vec<_>>();

        let mut route_sources = source_positions
            .iter()
            .copied()
            .chain([logical_source])
            .filter_map(|position| powered_route_source(route_world, position))
            .collect::<Vec<_>>();
        if route_sources.is_empty() {
            route_sources.push(PoweredRouteSource {
                position: source,
                strength: initial_signal_strength(route_world, source),
            });
        }
        let mut route_source_set = route_sources
            .iter()
            .map(|source| source.position)
            .collect::<HashSet<_>>();
        let total_sinks = sink_targets.len();
        let mut routed_sinks = 0usize;
        while !sink_targets.is_empty() {
            sort_internal_sink_targets_by_current_tree(&mut sink_targets, &route_sources);
            let mut selected_route = None;
            let mut last_error = None;

            for (sink_index, (var, sink)) in sink_targets.iter().copied().enumerate() {
                progress.item(
                    group_index + 1,
                    grouped_vars.len(),
                    format!(
                        "route net `{}.{}` -> `{}.{}` sink {}/{}",
                        source_key.0,
                        source_key.1,
                        var.target.0,
                        var.target.1,
                        routed_sinks + 1,
                        total_sinks
                    ),
                );
                let route_result = if routed_sinks == 0 {
                    route_source_to_target_from_access_points(
                        &route_world,
                        source_port,
                        logical_source,
                        &source_positions,
                        sink,
                        &all_sinks,
                        config.strategy,
                    )
                } else {
                    route_to_target_from_powered_network(
                        route_world,
                        &route_sources,
                        sink,
                        &all_sinks,
                        config.strategy,
                    )
                };
                let (route, next_world) = match route_result {
                    Ok(route) => route,
                    Err(failure) => {
                        last_error = Some(eyre::eyre!(
                            "failed to route {}.{} -> {}.{} at {:?}: {failure:?}",
                            source_key.0,
                            source_key.1,
                            var.target.0,
                            var.target.1,
                            sink.position
                        ));
                        continue;
                    }
                };
                if let Some(reason) =
                    route_power_contract_failure_reason(route_world, &next_world, &route)
                {
                    last_error = Some(eyre::eyre!(
                        "routed {}.{} -> {}.{} at {:?}, but route contract failed: {}",
                        source_key.0,
                        source_key.1,
                        var.target.0,
                        var.target.1,
                        sink.position,
                        reason,
                    ));
                    continue;
                }
                selected_route = Some((sink_index, var, sink, route, next_world));
                break;
            }

            let Some((sink_index, var, sink, route, next_world)) = selected_route else {
                return Err(last_error.unwrap_or_else(|| {
                    eyre::eyre!(
                        "failed to route {}.{} after {}/{} sink(s)",
                        source_key.0,
                        source_key.1,
                        routed_sinks,
                        total_sinks
                    )
                }));
            };

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
            for source in route_branch_sources(&next_world, &route, sink.position) {
                if route_source_set.insert(source.position) {
                    route_sources.push(source);
                }
            }
            sink_targets.remove(sink_index);
            prune_powered_route_sources(&mut route_sources, &mut route_source_set, &all_sinks, 0);
            *route_world = next_world;
            routes.push(route);
            routed_sinks += 1;
        }
    }

    Ok(())
}

fn group_vars_by_source_ordered<'a>(
    vars: &[&'a GraphModuleVariable],
) -> Vec<Vec<&'a GraphModuleVariable>> {
    let mut groups = Vec::<Vec<&GraphModuleVariable>>::new();
    for &var in vars {
        if groups
            .last()
            .and_then(|group| group.first())
            .is_some_and(|previous| previous.source == var.source)
        {
            groups.last_mut().expect("checked last group").push(var);
            continue;
        }

        groups.push(vec![var]);
    }

    groups
}

fn route_top_input_ports(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    config: &GlobalRoutingConfig,
    progress: &GlobalPnrProgress,
    route_world: &mut World3D,
    routes: &mut Vec<RoutedNet>,
) -> eyre::Result<()> {
    let top_inputs = module
        .ports
        .iter()
        .filter(|port| port.port_type.is_input())
        .collect::<Vec<_>>();

    let mut top_input_index = 0;
    for (port_index, port) in top_inputs.iter().enumerate() {
        let sinks = resolve_port_target_positions(candidates, placed_modules, &port.target);
        if sinks.is_empty() {
            progress.item(
                port_index + 1,
                top_inputs.len(),
                format!("skip top input `{}` with no placed sinks", port.name),
            );
            continue;
        }

        let input_sources = external_input_sources(route_world, top_input_index, &sinks);
        top_input_index += 1;
        if input_sources.is_empty() {
            return Err(eyre::eyre!(
                "failed to place top-level input switch `{}`",
                port.name
            ));
        }

        let mut routed = false;
        let mut last_error = None;
        for input_source in input_sources {
            let mut candidate_world = route_world.clone();
            let mut candidate_routes = routes.clone();
            match route_top_input_fanout(
                port.name.as_str(),
                port_index,
                top_inputs.len(),
                input_source,
                sinks.clone(),
                config,
                progress,
                &mut candidate_world,
                &mut candidate_routes,
            ) {
                Ok(()) => {
                    *route_world = candidate_world;
                    *routes = candidate_routes;
                    routed = true;
                    break;
                }
                Err(error) => {
                    last_error = Some(error);
                }
            }
        }

        if !routed {
            let Some(error) = last_error else {
                return Err(eyre::eyre!(
                    "failed to route top-level input `{}` with no source candidates",
                    port.name
                ));
            };
            return Err(error);
        }
    }

    Ok(())
}

fn route_top_input_fanout(
    port_name: &str,
    port_index: usize,
    total_ports: usize,
    input_source: ExternalInputSource,
    mut sinks: Vec<ResolvedPortTarget>,
    config: &GlobalRoutingConfig,
    progress: &GlobalPnrProgress,
    route_world: &mut World3D,
    routes: &mut Vec<RoutedNet>,
) -> eyre::Result<()> {
    let total_sinks = sinks.len();
    *route_world = input_source.world.clone();
    routes.push(RoutedNet::new(
        input_source.switch,
        input_source.switch,
        input_source.blocks.clone(),
        vec![input_source.route_source],
    ));

    let mut route_sources = vec![PoweredRouteSource {
        position: input_source.route_source,
        strength: MAX_REDSTONE_STRENGTH,
    }];
    let mut route_source_set = HashSet::from([input_source.route_source]);

    let all_sinks = sinks.clone();
    let mut routed_sinks = 0usize;
    while !sinks.is_empty() {
        sort_top_input_sinks_by_current_tree(&mut sinks, &route_sources);
        let mut selected_route = None;
        let mut last_error = None;

        for (sink_index, sink) in sinks.iter().copied().enumerate() {
            progress.item(
                port_index + 1,
                total_ports,
                format!(
                    "route top input `{}` sink {}/{}",
                    port_name,
                    routed_sinks + 1,
                    total_sinks
                ),
            );
            let route_result = route_top_input_to_target_from_network(
                route_world,
                input_source.switch,
                &route_sources,
                sink,
                &all_sinks,
                config.strategy,
            );
            let (mut route, next_world) = match route_result {
                Ok(route) => route,
                Err(failure) => {
                    last_error = Some(eyre::eyre!(
                        "failed to route top-level input {} -> {:?}: {failure:?}",
                        port_name,
                        sink.position
                    ));
                    continue;
                }
            };
            route.source = input_source.switch;
            if let Some(reason) =
                route_power_contract_failure_reason(route_world, &next_world, &route)
            {
                last_error = Some(eyre::eyre!(
                    "routed top-level input {} -> {:?}, but route contract failed: {}",
                    port_name,
                    sink.position,
                    reason,
                ));
                continue;
            }
            selected_route = Some((sink_index, sink, route, next_world));
            break;
        }

        let Some((sink_index, sink, route, next_world)) = selected_route else {
            return Err(last_error.unwrap_or_else(|| {
                eyre::eyre!(
                    "failed to route top-level input {} after {}/{} sink(s)",
                    port_name,
                    routed_sinks,
                    total_sinks
                )
            }));
        };

        for source in route_branch_sources(&next_world, &route, sink.position) {
            if route_source_set.insert(source.position) {
                route_sources.push(source);
            }
        }
        sinks.remove(sink_index);
        prune_powered_route_sources(&mut route_sources, &mut route_source_set, &sinks, 0);
        *route_world = next_world;
        routes.push(route);
        routed_sinks += 1;
    }

    Ok(())
}

fn sort_top_input_sinks_by_current_tree(
    sinks: &mut [ResolvedPortTarget],
    route_sources: &[PoweredRouteSource],
) {
    // Top-level fanout은 고정 순서보다 현재 route tree에서 가까운 sink를
    // 우선 시도하는 쪽이 안정적이다. 가까운 후보가 실패하면 같은 단계에서
    // 나머지 sink도 모두 시도하므로, 순서는 탐색 우선순위일 뿐이다.
    sinks.sort_by_key(|sink| {
        let nearest_source = route_sources
            .iter()
            .map(|source| source.position.manhattan_distance(&sink.position))
            .min()
            .unwrap_or(usize::MAX);
        (
            nearest_source,
            sink.position.0,
            sink.position.1,
            sink.position.2,
        )
    });
}

fn sort_internal_sink_targets_by_current_tree(
    sinks: &mut [(&GraphModuleVariable, ResolvedPortTarget)],
    route_sources: &[PoweredRouteSource],
) {
    sinks.sort_by_key(|(var, sink)| {
        let nearest_source = route_sources
            .iter()
            .map(|source| source.position.manhattan_distance(&sink.position))
            .min()
            .unwrap_or(usize::MAX);
        (
            register_next_target_bit_order(&var.target.0),
            std::cmp::Reverse(nearest_source),
            sink.position.0,
            sink.position.1,
            sink.position.2,
        )
    });
}

fn route_top_input_to_target_from_network(
    world: &World3D,
    logical_source: Position,
    sources: &[PoweredRouteSource],
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let mut sources = sources.to_vec();
    sources.sort_by_key(|source| source.position.manhattan_distance(&sink.position));
    for source in sources {
        if let Ok(route) = route_logical_source_to_target_position_with_strength(
            world,
            logical_source,
            source.position,
            source.strength,
            sink,
            same_net_sinks,
            strategy,
        ) {
            return Ok(route);
        }
    }

    Err(RouteFailure::Unreachable {
        source: logical_source,
        sink: sink.position,
    })
}

fn active_route_powers_sink(before: &World3D, after: &World3D, route: &RoutedNet) -> bool {
    if !can_validate_active_route_source(before, route.source)
        && !can_validate_active_route_source(after, route.source)
    {
        return true;
    }

    let Some(before_source_active) = route_source_power_after_settle(before, route.source, false)
    else {
        return false;
    };
    let Some((after_source_active, required_positions_powered)) =
        active_route_power_after_settle(after, route)
    else {
        return false;
    };

    if before_source_active && !after_source_active {
        return false;
    }
    if !after_source_active {
        return true;
    }

    required_positions_powered
}

fn active_route_power_after_settle(world: &World3D, route: &RoutedNet) -> Option<(bool, bool)> {
    let mut world = world.clone();
    reset_dynamic_power_states(&mut world);
    world.initialize_redstone_states();
    if matches!(world[route.source].kind, BlockKind::Switch { .. }) {
        world[route.source].kind = BlockKind::Switch { is_on: true };
    }

    let world = World::from(&world);
    let sim = Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
        .ok()?;
    Some((
        block_is_powered(sim.world(), route.source),
        route
            .required_powered_positions
            .iter()
            .all(|position| block_is_powered(sim.world(), *position)),
    ))
}

fn route_source_power_after_settle(
    world: &World3D,
    source: Position,
    force_switch_on: bool,
) -> Option<bool> {
    let mut world = world.clone();
    reset_dynamic_power_states(&mut world);
    world.initialize_redstone_states();
    if force_switch_on && matches!(world[source].kind, BlockKind::Switch { .. }) {
        world[source].kind = BlockKind::Switch { is_on: true };
    }

    let world = World::from(&world);
    let sim = Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
        .ok()?;
    Some(block_is_powered(sim.world(), source))
}

fn route_power_contract_holds(before: &World3D, after: &World3D, route: &RoutedNet) -> bool {
    route_power_contract_failure_reason(before, after, route).is_none()
}

fn route_power_contract_failure_reason(
    before: &World3D,
    after: &World3D,
    route: &RoutedNet,
) -> Option<&'static str> {
    if !active_route_powers_sink(before, after, route) {
        return Some("active source does not power all required route positions");
    }
    if !switch_route_releases_required_positions_when_off(after, route) {
        return Some("switch-off source still powers at least one required route position");
    }

    None
}

fn switch_route_releases_required_positions_when_off(world: &World3D, route: &RoutedNet) -> bool {
    if !matches!(world[route.source].kind, BlockKind::Switch { .. }) {
        return true;
    }

    let mut inactive_world = world.clone();
    reset_dynamic_power_states(&mut inactive_world);
    inactive_world[route.source].kind = BlockKind::Switch { is_on: false };
    inactive_world.initialize_redstone_states();

    let world = World::from(&inactive_world);
    let Ok(sim) =
        Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
    else {
        return false;
    };

    !route
        .required_released_positions
        .iter()
        .any(|position| block_is_powered(sim.world(), *position))
}

pub(crate) fn first_invalid_active_route<'a>(
    world: &World3D,
    routes: &'a [RoutedNet],
) -> Option<&'a RoutedNet> {
    routes
        .iter()
        .find(|route| !route_power_contract_holds(world, world, route))
}

fn can_validate_active_route_source(world: &World3D, position: Position) -> bool {
    matches!(
        world[position].kind,
        BlockKind::Redstone { .. }
            | BlockKind::Torch { .. }
            | BlockKind::Repeater { .. }
            | BlockKind::Switch { .. }
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
) -> (usize, usize, usize, &str, &str, &str, &str) {
    let is_clock_route =
        var.source.1.contains("clk") || var.target.1.contains("clk") || var.target.1 == "en";
    let target_port_priority = if is_cross_bit_next_input_route(var) {
        0
    } else if is_next_to_master_data_route(var) {
        1
    } else if is_clock_inverter_to_master_enable_route(var) {
        2
    } else if is_master_to_slave_data_route(var) {
        3
    } else if var.target.0.ends_with("_next") {
        4
    } else {
        match (var.target.1.as_str(), is_clock_route) {
            ("d", _) => 5,
            ("en", true) => 5,
            (_, false) => 6,
            _ => 7,
        }
    };
    let feedback_order = register_next_target_bit_order(&var.target.0);
    let source_port_priority = if var.source.1.ends_with("_n") { 0 } else { 1 };
    (
        target_port_priority,
        feedback_order,
        source_port_priority,
        var.source.0.as_str(),
        var.source.1.as_str(),
        var.target.0.as_str(),
        var.target.1.as_str(),
    )
}

fn is_master_to_slave_data_route(var: &crate::graph::module::GraphModuleVariable) -> bool {
    var.source.0.ends_with("_master")
        && var.source.1 == "q"
        && var.target.0.ends_with("_slave")
        && var.target.1 == "d"
}

fn is_next_to_master_data_route(var: &crate::graph::module::GraphModuleVariable) -> bool {
    var.source.0.ends_with("_next")
        && var.source.1 == "d"
        && var.target.0.ends_with("_master")
        && var.target.1 == "d"
}

fn is_clock_inverter_to_master_enable_route(
    var: &crate::graph::module::GraphModuleVariable,
) -> bool {
    var.source.0.ends_with("_clk_inv")
        && var.source.1.ends_with("_n")
        && var.target.0.ends_with("_master")
        && var.target.1 == "en"
}

fn is_cross_bit_next_input_route(var: &crate::graph::module::GraphModuleVariable) -> bool {
    let Some(source_bit) = register_module_bit(&var.source.0, "_slave") else {
        return false;
    };
    let Some(target_bit) = register_module_bit(&var.target.0, "_next") else {
        return false;
    };
    target_bit > source_bit
}

fn register_next_target_bit_order(target_module: &str) -> usize {
    register_module_bit(target_module, "_next")
        .map(|bit| usize::MAX - bit)
        .unwrap_or(usize::MAX)
}

fn register_module_bit(module: &str, suffix: &str) -> Option<usize> {
    module
        .strip_suffix(suffix)
        .and_then(|bit_name| bit_name.rsplit_once('_'))
        .and_then(|(_, bit)| bit.parse::<usize>().ok())
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
    if source_port.direction == PhysicalPortDirection::Output && source_port.requires_output_diode()
    {
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

fn route_source_to_target_from_access_points(
    world: &World3D,
    source_port: &PhysicalPort,
    logical_source: Position,
    route_sources: &[Position],
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let mut route_sources = route_sources.to_vec();
    route_sources.push(logical_source);
    route_sources.sort_by_key(|source| source.manhattan_distance(&sink.position));
    route_sources.dedup();
    let fallback_source = route_sources.first().copied().unwrap_or(logical_source);
    let mut route_sources = route_sources
        .into_iter()
        .take(FANOUT_ROUTE_SOURCE_LIMIT)
        .collect::<Vec<_>>();
    if !route_sources.contains(&logical_source) {
        route_sources.push(logical_source);
    }

    for route_source in route_sources {
        if !world.size.bound_on(route_source) || !is_route_terminal(world, route_source) {
            continue;
        }
        if let Ok((route, next_world)) = route_source_to_target_position(
            world,
            source_port,
            logical_source,
            route_source,
            sink,
            same_net_sinks,
            strategy,
        ) {
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

fn route_to_target_position(
    world: &World3D,
    source: Position,
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    if sink.requires_input_diode {
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

fn route_logical_source_to_target_position_with_strength(
    world: &World3D,
    logical_source: Position,
    route_source: Position,
    route_source_strength: usize,
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    if sink.requires_input_diode {
        return route_to_redstone_input_through_repeater_from_route_source(
            world,
            logical_source,
            route_source,
            route_source_strength,
            sink.position,
            same_net_sinks,
            strategy,
        );
    }

    let (route, routed_world) =
        route_point_to_point_with_strategy_and_allowed_contacts_and_initial_strength(
            world,
            route_source,
            sink.position,
            strategy,
            same_net_contact_positions(same_net_sinks),
            route_source_strength,
        )?;
    let powered_taps = route.powered_route_sources();
    let route = RoutedNet::new(logical_source, sink.position, route.blocks, route.path)
        .with_powered_taps(powered_taps);
    if active_route_powers_sink(world, &routed_world, &route) {
        return Ok((route, routed_world));
    }

    Err(RouteFailure::Unreachable {
        source: logical_source,
        sink: sink.position,
    })
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

fn route_to_target_from_powered_network(
    world: &World3D,
    sources: &[PoweredRouteSource],
    sink: ResolvedPortTarget,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let mut sources = sources.to_vec();
    sources.sort_by_key(|source| source.position.manhattan_distance(&sink.position));
    let fallback_source = sources
        .first()
        .map(|source| source.position)
        .unwrap_or(sink.position);

    for source in sources.into_iter().take(FANOUT_ROUTE_SOURCE_LIMIT) {
        if !world.size.bound_on(source.position) || !is_route_terminal(world, source.position) {
            continue;
        }
        if let Ok((route, next_world)) = route_logical_source_to_target_position_with_strength(
            world,
            source.position,
            source.position,
            source.strength,
            sink,
            same_net_sinks,
            strategy,
        ) {
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

fn route_branch_sources(
    world: &World3D,
    route: &RoutedNet,
    sink: Position,
) -> Vec<PoweredRouteSource> {
    let mut sources = route
        .powered_taps
        .iter()
        .copied()
        .filter_map(|(position, strength)| {
            (strength > 1 && world.size.bound_on(position) && is_route_terminal(world, position))
                .then_some(PoweredRouteSource { position, strength })
        })
        .collect::<Vec<_>>();
    sources.sort_by_key(|source| {
        (
            source.position.manhattan_distance(&sink),
            std::cmp::Reverse(source.strength),
            std::cmp::Reverse(source.position.0),
            source.position.1,
            source.position.2,
        )
    });
    sources.truncate(FANOUT_ROUTE_TERMINAL_LIMIT);
    sources
}

fn prune_powered_route_sources(
    route_sources: &mut Vec<PoweredRouteSource>,
    route_source_set: &mut HashSet<Position>,
    sinks: &[ResolvedPortTarget],
    next_sink_index: usize,
) {
    if route_sources.len() <= FANOUT_ROUTE_TERMINAL_LIMIT {
        return;
    }

    route_sources.sort_by_key(|source| {
        let nearest_remaining_sink = sinks
            .iter()
            .skip(next_sink_index)
            .map(|sink| source.position.manhattan_distance(&sink.position))
            .min()
            .unwrap_or(0);
        (
            nearest_remaining_sink,
            std::cmp::Reverse(source.strength),
            source.position.0,
            source.position.1,
            source.position.2,
        )
    });
    route_sources.truncate(FANOUT_ROUTE_TERMINAL_LIMIT);

    route_source_set.clear();
    route_source_set.extend(route_sources.iter().map(|source| source.position));
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
    if sink.requires_input_diode {
        for adapter in redstone_input_repeater_adapters(world, sink.position, &same_net_contacts) {
            let Ok((route, routed_world)) = route_direct_output_to_point(
                &adapter.world,
                logical_source,
                route_source,
                adapter.driver,
                strategy,
                same_net_contacts.clone(),
            ) else {
                continue;
            };
            let driver_route = RoutedNet::new(
                logical_source,
                adapter.driver,
                added_route_blocks(world, &routed_world),
                route.path.clone(),
            );
            if !active_route_powers_sink(world, &routed_world, &driver_route) {
                continue;
            }
            debug_assert!(detailed_router::target_powers_position(
                &routed_world,
                adapter.repeater,
                adapter.target
            ));
            let powered_taps = route.powered_route_sources();
            let route = RoutedNet::new(
                logical_source,
                adapter.target,
                added_route_blocks(world, &routed_world),
                route.path,
            )
            .with_required_powered_positions(vec![adapter.driver, adapter.repeater, adapter.target])
            .with_required_released_positions(vec![adapter.driver, adapter.repeater])
            .with_powered_taps(powered_taps);
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
    route_to_redstone_input_through_repeater_from_route_source(
        world,
        source,
        source,
        initial_signal_strength(world, source),
        sink,
        same_net_sinks,
        strategy,
    )
}

fn route_to_redstone_input_through_repeater_from_route_source(
    world: &World3D,
    logical_source: Position,
    route_source: Position,
    route_source_strength: usize,
    sink: Position,
    same_net_sinks: &[ResolvedPortTarget],
    strategy: GlobalRoutingStrategy,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let same_net_contacts = same_net_contact_positions(same_net_sinks);
    for adapter in redstone_input_repeater_adapters(world, sink, &same_net_contacts) {
        let Ok((route, routed_world)) =
            route_point_to_point_with_strategy_and_allowed_contacts_and_initial_strength(
                &adapter.world,
                route_source,
                adapter.driver,
                strategy,
                same_net_contacts.clone(),
                route_source_strength,
            )
        else {
            continue;
        };
        let driver_route = RoutedNet::new(
            logical_source,
            adapter.driver,
            added_route_blocks(world, &routed_world),
            route.path.clone(),
        );
        if !active_route_powers_sink(world, &routed_world, &driver_route) {
            continue;
        }
        debug_assert!(detailed_router::target_powers_position(
            &routed_world,
            adapter.repeater,
            adapter.target
        ));
        let powered_taps = route.powered_route_sources();
        let route = RoutedNet::new(
            logical_source,
            adapter.target,
            added_route_blocks(world, &routed_world),
            route.path,
        )
        .with_required_powered_positions(vec![adapter.driver, adapter.repeater, adapter.target])
        .with_required_released_positions(vec![adapter.driver, adapter.repeater])
        .with_powered_taps(powered_taps);
        if active_route_powers_sink(world, &routed_world, &route) {
            return Ok((route, routed_world));
        }
    }

    Err(RouteFailure::Unreachable {
        source: logical_source,
        sink,
    })
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
        powered_taps: vec![PoweredRouteSource {
            position: route_source,
            strength: 2,
        }],
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
            powered_taps: vec![PoweredRouteSource {
                position: tap,
                strength: signal_strength,
            }],
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
        let powered_taps = route.powered_route_sources();
        let route = RoutedNet::new(
            logical_source,
            sink,
            added_route_blocks(world, &routed_world),
            route.path,
        )
        .with_powered_taps(powered_taps);
        if active_route_powers_sink(world, &routed_world, &route) {
            return Ok((route, routed_world));
        }
    }

    Err(RouteFailure::Unreachable {
        source: logical_source,
        sink,
    })
}

fn redstone_input_repeater_adapters(
    world: &World3D,
    sink: Position,
    additional_allowed_contacts: &[Position],
) -> Vec<InputDiodeAdapter> {
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
) -> Option<InputDiodeAdapter> {
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
    detailed_router::target_powers_position(&adapter_world, repeater_position, sink).then_some(
        InputDiodeAdapter {
            world: adapter_world,
            driver: driver_position,
            repeater: repeater_position,
            target: sink,
        },
    )
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

fn external_input_sources(
    world: &World3D,
    index: usize,
    sinks: &[ResolvedPortTarget],
) -> Vec<ExternalInputSource> {
    let mut candidates = external_switch_candidates_outside_layout(world, sinks);
    candidates.extend(external_switch_candidates_for_sinks(sinks));
    candidates.sort_by_key(|position| external_input_candidate_cost(*position, sinks));
    candidates.dedup();

    let mut selected = Vec::new();
    selected.extend(candidates.iter().copied().take(8));
    for sink in sinks {
        let mut near_sink = candidates.clone();
        near_sink.sort_by_key(|position| {
            (
                position.manhattan_distance(&sink.position),
                external_input_candidate_cost(*position, sinks),
            )
        });
        selected.extend(near_sink.into_iter().take(4));
    }

    let max_x = world
        .iter_block()
        .into_iter()
        .map(|(position, _)| position.0)
        .max()
        .unwrap_or(0);
    let position = Position(max_x + 2, index * 3 + 1, 1);
    selected.push(position);
    selected.sort_by_key(|position| external_input_candidate_cost(*position, sinks));
    selected.dedup();

    selected
        .into_iter()
        .filter_map(|position| build_external_input_source(world, position))
        .take(24)
        .collect()
}

fn external_input_candidate_cost(
    position: Position,
    sinks: &[ResolvedPortTarget],
) -> (usize, usize, usize, usize) {
    let total_distance = sinks
        .iter()
        .map(|sink| position.manhattan_distance(&sink.position))
        .sum::<usize>();
    let max_distance = sinks
        .iter()
        .map(|sink| position.manhattan_distance(&sink.position))
        .max()
        .unwrap_or(0);
    (total_distance, max_distance, position.0, position.1)
}

fn build_external_input_source(world: &World3D, switch: Position) -> Option<ExternalInputSource> {
    let switch_block = input_switch_block();
    let route_source = switch.walk(switch_block.direction)?;
    let route_source_support = route_source.down()?;
    if !world.size.bound_on(switch)
        || !world.size.bound_on(route_source)
        || !world.size.bound_on(route_source_support)
        || !world[switch].kind.is_air()
        || !world[route_source].kind.is_air()
    {
        return None;
    }

    let mut source_world = world.clone();
    source_world[switch] = switch_block;
    place_support_cobble_if_needed(&mut source_world, route_source_support)?;

    let redstone_node = PlacedNode::new_redstone(route_source);
    if redstone_node.has_conflict(&source_world, &[switch].into_iter().collect()) {
        return None;
    }
    detailed_router::place_node(&mut source_world, redstone_node);
    if !detailed_router::target_powers_position(&source_world, switch, route_source) {
        return None;
    }

    let mut blocks = vec![(switch, switch_block)];
    if world[route_source_support].kind.is_air() {
        blocks.push((route_source_support, source_world[route_source_support]));
    }
    blocks.push((route_source, source_world[route_source]));

    Some(ExternalInputSource {
        world: source_world,
        switch,
        route_source,
        blocks,
    })
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

    let position = port.primary_route_position();
    vec![ResolvedPortTarget {
        position: translate_candidate_position(position, candidate, placed),
        requires_input_diode: port.requires_input_diode(),
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
                .routing_access_positions()
                .into_iter()
                .next()
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
    translate_candidate_position(port.primary_route_position(), candidate, placed)
}

fn translate_port_access_positions(
    port: &PhysicalPort,
    candidate: &LayoutCandidate,
    placed: &PlacedModule,
) -> Vec<Position> {
    let mut positions = port
        .routing_access_positions()
        .into_iter()
        .map(|position| translate_candidate_position(position, candidate, placed))
        .collect::<Vec<_>>();
    positions.sort();
    positions.dedup();
    positions
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
    route_point_to_point_with_strategy_and_allowed_contacts_and_initial_strength(
        world,
        source,
        sink,
        strategy,
        additional_allowed_contacts,
        initial_signal_strength(world, source),
    )
}

fn route_point_to_point_with_strategy_and_allowed_contacts_and_initial_strength(
    world: &World3D,
    source: Position,
    sink: Position,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: Vec<Position>,
    initial_strength: usize,
) -> Result<(RoutedNet, World3D), RouteFailure> {
    let goal = RouteGoal::for_sink(world, sink);
    route_point_to_point_with_bounds_and_initial_strength(
        world,
        source,
        sink,
        goal,
        BoundSearchMode::Propagation,
        None,
        strategy,
        &additional_allowed_contacts,
        initial_strength,
    )
    .or_else(|_| {
        route_point_to_point_with_bounds_and_initial_strength(
            world,
            source,
            sink,
            goal,
            BoundSearchMode::Nearby,
            None,
            strategy,
            &additional_allowed_contacts,
            initial_strength,
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
                powered_taps: vec![PoweredRouteSource {
                    position: driver_position,
                    strength: MAX_REDSTONE_STRENGTH,
                }],
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
        powered_taps: vec![PoweredRouteSource {
            position: seed,
            strength: initial_signal_strength(world, seed),
        }],
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
                let mut powered_taps = state.powered_taps.clone();
                powered_taps.push(PoweredRouteSource {
                    position: driver_position,
                    strength: MAX_REDSTONE_STRENGTH,
                });
                states.push(RouteSearchState {
                    world: adapter_world,
                    terminal: driver_position,
                    route,
                    signal_strength: MAX_REDSTONE_STRENGTH,
                    powered_taps,
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
                let next_strength = state.signal_strength - 1;
                let mut powered_taps = state.powered_taps.clone();
                powered_taps.push(PoweredRouteSource {
                    position: redstone_node.position,
                    strength: next_strength,
                });
                queue.push_back(RouteSearchState {
                    world: next_world,
                    terminal: redstone_node.position,
                    route,
                    signal_strength: next_strength,
                    powered_taps,
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
        &output_adapter_allowed_contacts(
            world,
            source,
            additional_allowed_contacts,
            &[source, repeater_position],
        ),
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
        &output_adapter_allowed_contacts(
            world,
            source,
            additional_allowed_contacts,
            &[source, repeater_position, driver_position],
        ),
    ) {
        return None;
    }

    Some((adapter_world, driver_position))
}

fn output_adapter_allowed_contacts(
    world: &World3D,
    source: Position,
    additional_allowed_contacts: &[Position],
    local_allowed_contacts: &[Position],
) -> Vec<Position> {
    let mut contacts = adapter_allowed_contacts(additional_allowed_contacts, local_allowed_contacts);
    contacts.extend(source_signal_positions(world, source));
    contacts.sort();
    contacts.dedup();
    contacts
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

fn route_point_to_point_with_bounds_and_initial_strength(
    world: &World3D,
    source: Position,
    sink: Position,
    goal: RouteGoal,
    mode: BoundSearchMode,
    min_z: Option<usize>,
    strategy: GlobalRoutingStrategy,
    additional_allowed_contacts: &[Position],
    initial_strength: usize,
) -> Result<(RoutedNet, World3D), RouteFailure> {
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
            powered_taps: vec![PoweredRouteSource {
                position: source,
                strength: initial_strength,
            }],
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
                RoutedNet::new(source, sink, blocks, state.route)
                    .with_powered_taps(state.powered_taps),
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
                            let mut powered_taps = state.powered_taps.clone();
                            powered_taps.push(PoweredRouteSource {
                                position: redstone_node.position,
                                strength: next_strength,
                            });
                            queue.push(RouteSearchState {
                                world: next_world,
                                terminal: redstone_node.position,
                                route: next_route,
                                signal_strength: next_strength,
                                powered_taps,
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
                                let mut powered_taps = state.powered_taps.clone();
                                powered_taps.push(PoweredRouteSource {
                                    position: repeater_node.position,
                                    strength: MAX_REDSTONE_STRENGTH,
                                });
                                queue.push(RouteSearchState {
                                    world: next_world,
                                    terminal: repeater_node.position,
                                    route: next_route,
                                    signal_strength: MAX_REDSTONE_STRENGTH,
                                    powered_taps,
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
        if route_position.manhattan_distance(&position) > SIGNAL_CONTACT_SEARCH_RADIUS {
            return false;
        }
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
    powered_taps: Vec<PoweredRouteSource>,
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

fn powered_route_source(world: &World3D, position: Position) -> Option<PoweredRouteSource> {
    (world.size.bound_on(position) && is_route_terminal(world, position)).then_some(
        PoweredRouteSource {
            position,
            strength: initial_signal_strength(world, position),
        },
    )
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
        LayoutCandidate, PhysicalPort, PhysicalPortDirection, PortConnection,
    };
    use crate::transform::place_and_route::global_pnr::placer::{
        place_candidates_on_shelves, GlobalPlacementConfig, PlacedModule,
    };
    use crate::world::block::{BlockKind, Direction, RedstoneState};
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
                access_points: Vec::new(),
                connection: PortConnection::Direct,
            }],
        )
        .unwrap()
    }

    #[test]
    fn resolve_port_targets_uses_exposed_route_position() {
        let mut candidates = vec![candidate(
            "right",
            Position(0, 0, 1),
            "in",
            PhysicalPortDirection::Input,
        )];
        candidates[0].ports[0].route_position = Some(Position(1, 0, 1));
        let placed = vec![PlacedModule {
            module_name: "right".to_owned(),
            candidate_index: 0,
            origin: Position(10, 20, 0),
            bbox: candidates[0].bbox,
        }];

        let targets = resolve_port_targets(&candidates, &placed, "right", "in");

        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].position, Position(11, 20, 1));
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
    fn route_result_preserves_powered_tap_strengths_for_fanout() {
        let source = Position(0, 1, 1);
        let sink = Position(6, 1, 1);
        let world = route_test_world_with_switch_source(source, sink, DimSize(10, 4, 3));
        let (route, _) = route_point_to_point(&world, source, sink).unwrap();

        assert!(
            route.powered_taps.iter().any(
                |(position, strength)| *position != source && *strength < MAX_REDSTONE_STRENGTH
            ),
            "fanout branch candidates should keep the remaining signal strength from search"
        );
        assert!(route
            .powered_taps
            .iter()
            .all(|(_, strength)| *strength <= MAX_REDSTONE_STRENGTH));
    }

    #[test]
    fn input_diode_route_preserves_powered_taps_for_later_fanout() {
        let source = Position(0, 1, 1);
        let sink = Position(12, 1, 1);
        let world = route_test_world_with_switch_source(source, sink, DimSize(16, 4, 3));

        let (route, _) = route_to_target_position(
            &world,
            source,
            ResolvedPortTarget {
                position: sink,
                requires_input_diode: true,
            },
            &[],
            GlobalRoutingStrategy::AStar,
        )
        .unwrap();

        assert!(
            route.powered_taps.len() > 1,
            "diode adapter routes must keep intermediate powered taps for fanout"
        );
    }

    #[test]
    fn output_diode_route_preserves_powered_taps_for_later_fanout() {
        let source = Position(0, 1, 1);
        let sink = Position(12, 1, 1);
        let world = route_test_world_with_size(source, sink, DimSize(16, 4, 3));
        let source_port = PhysicalPort {
            name: "q".to_owned(),
            direction: PhysicalPortDirection::Output,
            position: source,
            route_position: None,
            access_points: Vec::new(),
            connection: PortConnection::OutputDiode,
        };

        let (route, _) = route_source_to_target_position(
            &world,
            &source_port,
            source,
            source,
            ResolvedPortTarget {
                position: sink,
                requires_input_diode: true,
            },
            &[],
            GlobalRoutingStrategy::AStar,
        )
        .unwrap();

        assert!(
            route.powered_taps.len() > 1,
            "output isolation must not discard the route taps used for fanout"
        );
    }

    #[test]
    fn output_diode_route_descends_from_high_latch_tap_to_low_input() {
        let source = Position(75, 16, 5);
        let sink = Position(43, 17, 1);
        let world = route_test_world_with_size(source, sink, DimSize(84, 24, 8));
        let source_port = PhysicalPort {
            name: "q".to_owned(),
            direction: PhysicalPortDirection::Output,
            position: source,
            route_position: None,
            access_points: Vec::new(),
            connection: PortConnection::OutputDiode,
        };

        route_source_to_target_position(
            &world,
            &source_port,
            source,
            source,
            ResolvedPortTarget {
                position: sink,
                requires_input_diode: false,
            },
            &[],
            GlobalRoutingStrategy::AStar,
        )
        .unwrap();
    }

    #[test]
    fn output_diode_route_descends_to_low_input_diode() {
        let source = Position(33, 16, 5);
        let sink = Position(48, 16, 2);
        let world = route_test_world_with_size(source, sink, DimSize(60, 24, 8));
        let source_port = PhysicalPort {
            name: "q".to_owned(),
            direction: PhysicalPortDirection::Output,
            position: source,
            route_position: None,
            access_points: Vec::new(),
            connection: PortConnection::OutputDiode,
        };

        route_source_to_target_position(
            &world,
            &source_port,
            source,
            source,
            ResolvedPortTarget {
                position: sink,
                requires_input_diode: true,
            },
            &[],
            GlobalRoutingStrategy::AStar,
        )
        .unwrap();
    }

    #[test]
    fn output_access_routing_falls_back_to_logical_source() {
        let source = Position(0, 1, 1);
        let sink = Position(5, 1, 1);
        let world = route_test_world_with_size(source, sink, DimSize(8, 4, 3));
        let source_port = PhysicalPort {
            name: "q".to_owned(),
            direction: PhysicalPortDirection::Output,
            position: source,
            route_position: None,
            access_points: vec![Position(99, 99, 99)],
            connection: PortConnection::OutputDiode,
        };

        route_source_to_target_from_access_points(
            &world,
            &source_port,
            source,
            &source_port.access_points,
            ResolvedPortTarget {
                position: sink,
                requires_input_diode: false,
            },
            &[],
            GlobalRoutingStrategy::AStar,
        )
        .unwrap();
    }

    #[test]
    fn top_input_fanout_prioritizes_sinks_near_current_route_tree() {
        let mut sinks = vec![
            ResolvedPortTarget {
                position: Position(2, 0, 1),
                requires_input_diode: false,
            },
            ResolvedPortTarget {
                position: Position(8, 0, 1),
                requires_input_diode: false,
            },
            ResolvedPortTarget {
                position: Position(5, 0, 1),
                requires_input_diode: false,
            },
        ];
        let route_sources = vec![PoweredRouteSource {
            position: Position(4, 0, 1),
            strength: MAX_REDSTONE_STRENGTH,
        }];

        sort_top_input_sinks_by_current_tree(&mut sinks, &route_sources);

        assert_eq!(
            sinks.iter().map(|sink| sink.position).collect::<Vec<_>>(),
            vec![Position(5, 0, 1), Position(2, 0, 1), Position(8, 0, 1)]
        );
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
                requires_input_diode: true,
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
                requires_input_diode: true,
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
                requires_input_diode: true,
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
    fn route_rejects_repeater_when_previous_redstone_hits_the_wrong_side() {
        let prev = Position(2, 1, 1);
        let repeater = Position(2, 2, 1);
        let mut world = World3D::new(DimSize(5, 5, 3));
        world[prev.down().unwrap()] = cobble_block();
        world[prev] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: RedstoneState::North as usize,
                strength: 0,
            },
            direction: Direction::None,
        };

        let result = detailed_router::place_repeater_with_cobble(
            &world,
            PlaceBound(PropagateType::Soft, repeater, Direction::South),
            prev,
            Position(2, 3, 1),
            Direction::East,
            None,
        );

        assert!(matches!(
            result,
            PlaceRepeaterResult::Rejected(detailed_router::RouteRejectReason::DisconnectedRoute)
        ));
    }

    #[test]
    fn active_route_validation_checks_off_switch_after_turning_it_on() {
        let source = Position(0, 1, 1);
        let sink = Position(3, 1, 1);
        let before = route_test_world_with_switch_source(source, sink, DimSize(6, 4, 3));
        let mut after = before.clone();
        after[sink.down().unwrap()] = cobble_block();
        after[sink] = redstone_block();
        after.initialize_redstone_states();
        let route = RoutedNet::new(
            source,
            sink,
            vec![(sink, redstone_block())],
            vec![source, sink],
        );

        assert!(
            !active_route_powers_sink(&before, &after, &route),
            "an off top-level switch must still be validated in the active/on state"
        );
    }

    #[test]
    fn active_route_validation_rejects_route_that_turns_active_source_off() {
        let source = Position(0, 1, 1);
        let sink = Position(3, 1, 1);
        let mut before = route_test_world(source, sink);
        before[source] = Block {
            kind: BlockKind::Torch { is_on: true },
            direction: Direction::East,
        };
        let mut after = before.clone();
        after[source] = Block {
            kind: BlockKind::Torch { is_on: false },
            direction: Direction::East,
        };
        let route = RoutedNet::new(
            source,
            sink,
            vec![(sink, redstone_block())],
            vec![source, sink],
        );

        assert!(
            !active_route_powers_sink(&before, &after, &route),
            "a route that disables an already-active source must not be accepted"
        );
    }

    #[test]
    fn active_route_validation_rejects_unpowered_sink_from_active_source() {
        let source = Position(1, 1, 1);
        let sink = Position(4, 1, 1);
        let mut before = route_test_world_with_size(source, sink, DimSize(6, 3, 3));
        before[source] = Block {
            kind: BlockKind::Torch { is_on: true },
            direction: Direction::East,
        };
        before.initialize_redstone_states();
        let after = before.clone();
        let route = RoutedNet::new(source, sink, Vec::new(), vec![source, sink]);

        assert!(
            !active_route_powers_sink(&before, &after, &route),
            "an already-active source must power the routed sink"
        );
    }

    #[test]
    fn active_route_validation_can_check_redstone_sources() {
        let source = Position(1, 1, 1);
        let world = route_test_world_with_size(source, Position(4, 1, 1), DimSize(6, 3, 3));

        assert!(can_validate_active_route_source(&world, source));
    }

    #[test]
    fn active_route_validation_checks_required_powered_positions() {
        let source = Position(0, 1, 1);
        let driver = Position(1, 1, 1);
        let repeater = Position(2, 1, 1);
        let sink = Position(3, 1, 1);
        let mut before = World3D::new(DimSize(5, 3, 3));
        before[source] = Block {
            kind: BlockKind::Switch { is_on: true },
            direction: Direction::West,
        };
        before.initialize_redstone_states();
        let mut after = before.clone();
        after[sink] = Block {
            kind: BlockKind::Switch { is_on: true },
            direction: Direction::West,
        };
        after.initialize_redstone_states();
        let route = RoutedNet::new(
            source,
            sink,
            Vec::new(),
            vec![source, driver, repeater, sink],
        )
        .with_required_powered_positions(vec![driver, repeater, sink]);

        assert!(
            !active_route_powers_sink(&before, &after, &route),
            "isolated routes must validate their diode driver/repeater, not only the logical sink"
        );
    }

    #[test]
    fn route_power_contract_rejects_switch_route_that_stays_on_when_switch_is_off() {
        let source = Position(0, 1, 1);
        let sink = Position(2, 1, 1);
        let mut world = World3D::new(DimSize(4, 3, 3));
        world[source] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[sink] = Block {
            kind: BlockKind::RedstoneBlock,
            direction: Direction::None,
        };
        world.initialize_redstone_states();
        let route = RoutedNet::new(source, sink, Vec::new(), vec![source, sink]);

        assert!(
            !route_power_contract_holds(&world, &world, &route),
            "top-level switch routes must not leave required positions powered while the switch is off"
        );
    }

    #[test]
    fn switch_release_contract_can_ignore_independently_powered_diode_target() {
        let source = Position(0, 1, 1);
        let driver = Position(1, 1, 1);
        let sink = Position(2, 1, 1);
        let mut world = World3D::new(DimSize(4, 3, 3));
        world[source] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[sink] = Block {
            kind: BlockKind::RedstoneBlock,
            direction: Direction::None,
        };
        world.initialize_redstone_states();
        let route = RoutedNet::new(source, sink, Vec::new(), vec![source, driver, sink])
            .with_required_powered_positions(vec![driver, sink])
            .with_required_released_positions(vec![driver]);

        assert!(
            switch_route_releases_required_positions_when_off(&world, &route),
            "input diode routes only need route-owned driver positions to release; the child-side target may be powered independently"
        );
    }

    #[test]
    fn active_route_set_validation_rechecks_routes_against_latest_world() {
        let source = Position(1, 1, 1);
        let sink = Position(4, 1, 1);
        let mut latest_world = route_test_world_with_size(source, sink, DimSize(6, 3, 3));
        latest_world[source] = Block {
            kind: BlockKind::Torch { is_on: true },
            direction: Direction::East,
        };
        latest_world.initialize_redstone_states();
        let routes = vec![RoutedNet::new(source, sink, Vec::new(), vec![source, sink])];

        assert_eq!(
            first_invalid_active_route(&latest_world, &routes).map(|route| route.sink),
            Some(sink),
            "previously routed active nets must still power their sinks after later routing"
        );
    }

    #[test]
    fn route_priority_routes_high_bit_feedback_first() {
        let q0_feedback = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_0_next".to_owned(), "q_0".to_owned()),
        };
        let q1_feedback = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_1_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_1".to_owned()),
        };

        assert!(route_variable_priority(&q1_feedback) < route_variable_priority(&q0_feedback));
    }

    #[test]
    fn route_priority_routes_cross_bit_next_input_before_own_bit_feedback() {
        let q0_to_own_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_0_next".to_owned(), "q_0".to_owned()),
        };
        let q0_to_next_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_0".to_owned()),
        };

        assert!(route_variable_priority(&q0_to_next_bit) < route_variable_priority(&q0_to_own_bit));
    }

    #[test]
    fn route_priority_routes_cross_bit_next_input_before_next_to_master_output() {
        let fanin_to_next = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_0".to_owned()),
        };
        let next_to_master = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_1_next".to_owned(), "d".to_owned()),
            target: ("q_1_master".to_owned(), "d".to_owned()),
        };

        assert!(route_variable_priority(&fanin_to_next) < route_variable_priority(&next_to_master));
    }

    #[test]
    fn route_priority_routes_next_to_master_output_before_self_feedback() {
        let self_feedback = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_1_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_1".to_owned()),
        };
        let next_to_master = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_1_next".to_owned(), "d".to_owned()),
            target: ("q_1_master".to_owned(), "d".to_owned()),
        };

        assert!(route_variable_priority(&next_to_master) < route_variable_priority(&self_feedback));
    }

    #[test]
    fn route_priority_routes_master_to_slave_handoff_before_feedback() {
        let master_to_slave = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_master".to_owned(), "q".to_owned()),
            target: ("q_0_slave".to_owned(), "d".to_owned()),
        };
        let self_feedback = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_0_next".to_owned(), "q_0".to_owned()),
        };

        assert!(route_variable_priority(&master_to_slave) < route_variable_priority(&self_feedback));
    }

    #[test]
    fn route_priority_routes_clock_inverter_enable_before_self_feedback() {
        let clock_to_master = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_clk_inv".to_owned(), "clk_n".to_owned()),
            target: ("q_0_master".to_owned(), "en".to_owned()),
        };
        let self_feedback = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_0_next".to_owned(), "q_0".to_owned()),
        };

        assert!(route_variable_priority(&clock_to_master) < route_variable_priority(&self_feedback));
    }

    #[test]
    fn route_priority_routes_cross_bit_next_input_before_clock_enable() {
        let cross_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_0".to_owned()),
        };
        let clock_to_master = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_clk_inv".to_owned(), "clk_n".to_owned()),
            target: ("q_0_master".to_owned(), "en".to_owned()),
        };

        assert!(route_variable_priority(&cross_bit) < route_variable_priority(&clock_to_master));
    }

    #[test]
    fn route_priority_routes_cross_bit_next_input_before_master_to_slave_handoff() {
        let cross_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_0".to_owned()),
        };
        let master_to_slave = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_master".to_owned(), "q".to_owned()),
            target: ("q_0_slave".to_owned(), "d".to_owned()),
        };

        assert!(route_variable_priority(&cross_bit) < route_variable_priority(&master_to_slave));
    }

    #[test]
    fn route_priority_routes_next_to_master_data_before_clock_enable() {
        let next_to_master = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_1_next".to_owned(), "d".to_owned()),
            target: ("q_1_master".to_owned(), "d".to_owned()),
        };
        let clock_to_master = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_1_clk_inv".to_owned(), "clk_n".to_owned()),
            target: ("q_1_master".to_owned(), "en".to_owned()),
        };

        assert!(
            route_variable_priority(&next_to_master) < route_variable_priority(&clock_to_master)
        );
    }

    #[test]
    fn internal_fanout_routes_far_sink_before_near_sink() {
        let source = PoweredRouteSource {
            position: Position(0, 1, 0),
            strength: MAX_REDSTONE_STRENGTH,
        };
        let far_var = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_0".to_owned()),
        };
        let near_var = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_0_next".to_owned(), "q_0".to_owned()),
        };
        let far_sink = ResolvedPortTarget {
            position: Position(10, 1, 0),
            requires_input_diode: true,
        };
        let near_sink = ResolvedPortTarget {
            position: Position(3, 1, 0),
            requires_input_diode: true,
        };
        let mut sinks = vec![(&near_var, near_sink), (&far_var, far_sink)];

        sort_internal_sink_targets_by_current_tree(&mut sinks, &[source]);

        assert_eq!(sinks[0].1.position, far_sink.position);
        assert_eq!(sinks[1].1.position, near_sink.position);
    }

    #[test]
    fn route_var_grouping_preserves_sorted_priority_boundaries() {
        let q0_to_own_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_0_next".to_owned(), "q_0".to_owned()),
        };
        let q1_feedback = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_1_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_1".to_owned()),
        };
        let q0_to_next_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_0".to_owned()),
        };
        let vars = vec![&q0_to_own_bit, &q1_feedback, &q0_to_next_bit];

        let groups = group_vars_by_source_ordered(&vars);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0][0].target, q0_to_own_bit.target);
        assert_eq!(groups[1][0].target, q1_feedback.target);
        assert_eq!(groups[2][0].target, q0_to_next_bit.target);
    }

    #[test]
    fn route_var_grouping_keeps_adjacent_same_source_fanout_together() {
        let q0_to_own_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_0_next".to_owned(), "q_0".to_owned()),
        };
        let q0_to_next_bit = GraphModuleVariable {
            var_type: GraphModulePortType::InputNet,
            source: ("q_0_slave".to_owned(), "q".to_owned()),
            target: ("q_1_next".to_owned(), "q_0".to_owned()),
        };
        let vars = vec![&q0_to_own_bit, &q0_to_next_bit];

        let groups = group_vars_by_source_ordered(&vars);

        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0]
                .iter()
                .map(|var| var.target.clone())
                .collect::<Vec<_>>(),
            vec![q0_to_own_bit.target, q0_to_next_bit.target]
        );
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
    fn route_module_variables_routes_top_inputs_before_internal_nets() -> eyre::Result<()> {
        let module = GraphModule {
            ports: vec![GraphModulePort {
                name: "clk".to_owned(),
                port_type: GraphModulePortType::InputNet,
                target: GraphModulePortTarget::Module("clocked".to_owned(), "clk".to_owned()),
            }],
            vars: vec![GraphModuleVariable {
                var_type: GraphModulePortType::InputNet,
                source: ("left".to_owned(), "out".to_owned()),
                target: ("right".to_owned(), "in".to_owned()),
            }],
            ..Default::default()
        };
        let candidates = vec![
            candidate(
                "clocked",
                Position(0, 0, 1),
                "clk",
                PhysicalPortDirection::Input,
            ),
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

        assert!(
            routes
                .first()
                .is_some_and(|route| route.source == route.sink
                    && route
                        .blocks
                        .first()
                        .is_some_and(|(_, block)| block.kind.is_switch())),
            "top-level input switch route should reserve the global input trunk before internal nets"
        );
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
