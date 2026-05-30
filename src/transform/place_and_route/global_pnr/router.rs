use std::collections::{HashSet, VecDeque};

use eyre::ContextCompat;

use crate::graph::module::GraphModule;
use crate::transform::place_and_route::detailed_router::{self, PlaceRedstoneResult};
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::transform::place_and_route::place_bound::{PlaceBound, PropagateType};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::Block;
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

const GLOBAL_ROUTE_PADDING: usize = 8;
const GLOBAL_ROUTE_MAX_STEPS: usize = 64;

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

    for var in &module.vars {
        let source =
            resolve_port_position(candidates, placed_modules, &var.source.0, &var.source.1)
                .with_context(|| {
                    format!(
                        "source port {}.{} is not placed",
                        var.source.0, var.source.1
                    )
                })?;
        let sink = resolve_port_position(candidates, placed_modules, &var.target.0, &var.target.1)
            .with_context(|| {
                format!(
                    "target port {}.{} is not placed",
                    var.target.0, var.target.1
                )
            })?;

        let (route, next_world) =
            route_point_to_point(&route_world, source, sink).map_err(|failure| {
                eyre::eyre!(
                    "failed to route {}.{} -> {}.{}: {failure:?}",
                    var.source.0,
                    var.source.1,
                    var.target.0,
                    var.target.1
                )
            })?;
        route_world = next_world;
        routes.push(route);
    }

    Ok(routes)
}

fn resolve_port_position(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    module_name: &str,
    port_name: &str,
) -> Option<Position> {
    let placed = placed_modules
        .iter()
        .find(|placed| placed.module_name == module_name)?;
    let candidate = candidates.get(placed.candidate_index)?;
    let port = candidate.ports.iter().find(|port| port.name == port_name)?;
    Some(translate_candidate_position(
        port.position,
        candidate,
        placed,
    ))
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

#[derive(Clone, Copy)]
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
    let mut queue = VecDeque::from([(world.clone(), source, vec![source])]);
    let mut visited = HashSet::from([(source, 0)]);

    while let Some((state_world, terminal, route)) = queue.pop_front() {
        if goal.accepts(&state_world, terminal) {
            let blocks = added_route_blocks(world, &state_world);
            return Ok((
                RoutedNet {
                    source,
                    sink,
                    blocks,
                },
                state_world,
            ));
        }

        if route.len() > GLOBAL_ROUTE_MAX_STEPS {
            continue;
        }

        let terminal_node = PlacedNode::new(terminal, state_world[terminal]);
        let allowed_shorts = goal.allowed_short_positions();
        for bound in route_bounds_for_mode(mode, &state_world, &terminal_node) {
            if !bound.is_bound_on(&state_world) || bound.position() == goal.placement_target() {
                continue;
            }
            if !visited.insert((bound.position(), route.len())) {
                continue;
            }

            let PlaceRedstoneResult::Placed(next_world, redstone_node) =
                detailed_router::place_redstone_with_cobble_and_allowed_shorts(
                    &state_world,
                    bound,
                    terminal,
                    goal.placement_target(),
                    allowed_shorts.as_ref(),
                )
            else {
                continue;
            };

            let mut next_route = route.clone();
            next_route.push(redstone_node.position);
            queue.push_back((next_world, redstone_node.position, next_route));
        }
    }

    Err(RouteFailure::Unreachable { source, sink })
}

#[derive(Clone, Copy)]
enum RouteGoal {
    ConnectPosition { target: Position },
}

impl RouteGoal {
    fn for_sink(_world: &World3D, sink: Position) -> Self {
        Self::ConnectPosition { target: sink }
    }

    fn placement_target(self) -> Position {
        match self {
            Self::ConnectPosition { target } => target,
        }
    }

    fn accepts(self, world: &World3D, redstone: Position) -> bool {
        match self {
            Self::ConnectPosition { target } => {
                detailed_router::target_powers_redstone(world, target, redstone)
            }
        }
    }

    fn allowed_short_positions(self) -> Option<HashSet<Position>> {
        None
    }
}

fn route_bounds_for_mode(
    mode: BoundSearchMode,
    world: &World3D,
    terminal_node: &PlacedNode,
) -> Vec<PlaceBound> {
    match mode {
        BoundSearchMode::Propagation => terminal_node.propagation_bound(Some(world)),
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
    use crate::graph::module::{GraphModule, GraphModulePortType, GraphModuleVariable};
    use crate::transform::place_and_route::global_pnr::ir::{
        LayoutCandidate, PhysicalPort, PhysicalPortDirection,
    };
    use crate::transform::place_and_route::global_pnr::placer::{
        place_candidates_on_shelves, GlobalPlacementConfig,
    };
    use crate::world::block::{BlockKind, Direction};

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

        assert!(route
            .blocks
            .iter()
            .any(|(position, block)| *position == Position(1, 1, 0) && block.kind.is_cobble()));
        assert!(route
            .blocks
            .iter()
            .any(|(position, block)| *position == Position(1, 1, 1) && block.kind.is_redstone()));
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
}
