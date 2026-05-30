use std::collections::{HashMap, HashSet, VecDeque};

use eyre::ContextCompat;

use crate::graph::module::GraphModule;
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::Position;

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
    let blocked = translated_occupied_cells(candidates, placed_modules);
    module
        .vars
        .iter()
        .map(|var| {
            let source =
                resolve_port_position(candidates, placed_modules, &var.source.0, &var.source.1)
                    .with_context(|| {
                        format!(
                            "source port {}.{} is not placed",
                            var.source.0, var.source.1
                        )
                    })?;
            let sink =
                resolve_port_position(candidates, placed_modules, &var.target.0, &var.target.1)
                    .with_context(|| {
                        format!(
                            "target port {}.{} is not placed",
                            var.target.0, var.target.1
                        )
                    })?;

            route_point_to_point(source, sink, &blocked).map_err(|failure| {
                eyre::eyre!(
                    "failed to route {}.{} -> {}.{}: {failure:?}",
                    var.source.0,
                    var.source.1,
                    var.target.0,
                    var.target.1
                )
            })
        })
        .collect()
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

fn translated_occupied_cells(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
) -> HashSet<Position> {
    placed_modules
        .iter()
        .filter_map(|placed| {
            candidates
                .get(placed.candidate_index)
                .map(|candidate| (placed, candidate))
        })
        .flat_map(|(placed, candidate)| {
            candidate
                .occupied_cells
                .iter()
                .copied()
                .map(move |position| translate_candidate_position(position, candidate, placed))
        })
        .collect()
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
    source: Position,
    sink: Position,
    blocked: &HashSet<Position>,
) -> Result<RoutedNet, RouteFailure> {
    let mut queue = VecDeque::from([source]);
    let mut previous = HashMap::<Position, Position>::new();
    let mut visited = HashSet::from([source]);
    let bound = routing_bound(source, sink, blocked);

    while let Some(position) = queue.pop_front() {
        if position == sink {
            let path = reconstruct_path(source, sink, &previous);
            return Ok(RoutedNet {
                source,
                sink,
                blocks: route_blocks(&path),
            });
        }

        for next in neighbors(position, bound) {
            if next != sink && !route_position_available(next, blocked) {
                continue;
            }
            if !visited.insert(next) {
                continue;
            }
            previous.insert(next, position);
            queue.push_back(next);
        }
    }

    Err(RouteFailure::Unreachable { source, sink })
}

fn routing_bound(source: Position, sink: Position, blocked: &HashSet<Position>) -> Position {
    let mut max = Position(
        source.0.max(sink.0),
        source.1.max(sink.1),
        source.2.max(sink.2),
    );
    for position in blocked {
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }
    Position(max.0 + 2, max.1 + 2, max.2)
}

fn neighbors(position: Position, bound: Position) -> impl Iterator<Item = Position> {
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
        if next_x > bound.0 || next_y > bound.1 {
            continue;
        }
        result.push(Position(next_x, next_y, position.2));
        if position.2 + 1 <= bound.2 {
            result.push(Position(next_x, next_y, position.2 + 1));
        }
        if position.2 > 1 {
            result.push(Position(next_x, next_y, position.2 - 1));
        }
    }
    result.into_iter()
}

fn route_position_available(position: Position, blocked: &HashSet<Position>) -> bool {
    let Some(support) = position.down() else {
        return false;
    };
    !blocked.contains(&position) && !blocked.contains(&support)
}

fn reconstruct_path(
    source: Position,
    sink: Position,
    previous: &HashMap<Position, Position>,
) -> Vec<Position> {
    let mut path = vec![sink];
    let mut current = sink;
    while current != source {
        current = previous[&current];
        path.push(current);
    }
    path.reverse();
    path
}

fn route_blocks(path: &[Position]) -> Vec<(Position, Block)> {
    path.iter()
        .copied()
        .skip(1)
        .take(path.len().saturating_sub(2))
        .flat_map(|position| {
            let support = position.down().expect("route path should have support");
            [(support, cobble_block()), (position, redstone_block())]
        })
        .collect()
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
    use crate::world::position::DimSize;
    use crate::world::World3D;

    fn candidate(
        module_name: &str,
        block_position: Position,
        port_name: &str,
        direction: PhysicalPortDirection,
    ) -> LayoutCandidate {
        let mut world = World3D::new(DimSize(2, 1, 2));
        world[block_position] = redstone_block();
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

    #[test]
    fn route_point_to_point_places_support_cobble_under_redstone() {
        let route = route_point_to_point(Position(0, 0, 1), Position(3, 0, 1), &Default::default())
            .unwrap();

        assert!(route
            .blocks
            .iter()
            .any(|(position, block)| *position == Position(1, 0, 0) && block.kind.is_cobble()));
        assert!(route
            .blocks
            .iter()
            .any(|(position, block)| *position == Position(1, 0, 1) && block.kind.is_redstone()));
    }

    #[test]
    fn route_point_to_point_avoids_blocked_support_space() {
        let blocked = [Position(1, 0, 0)].into_iter().collect();
        let route = route_point_to_point(Position(0, 0, 1), Position(2, 0, 1), &blocked).unwrap();

        assert!(!route
            .blocks
            .iter()
            .any(|(position, _)| *position == Position(1, 0, 1)));
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
