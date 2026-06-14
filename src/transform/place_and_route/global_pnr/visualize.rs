use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

pub fn placement_bbox_wireframe_world(placed_modules: &[PlacedModule]) -> World3D {
    let mut edge_positions = Vec::new();

    for placed in placed_modules {
        let min = placed.origin;
        let max = placement_bbox_max(placed);
        edge_positions.extend(bbox_edge_positions(min, max));
    }

    let size = world_size_for_positions(&edge_positions);
    let mut world = World3D::new(size);
    for position in edge_positions {
        world[position] = cobble_block();
    }
    world
}

fn placement_bbox_max(placed: &PlacedModule) -> Position {
    Position(
        placed.origin.0 + placed.bbox.width() - 1,
        placed.origin.1 + placed.bbox.depth() - 1,
        placed.origin.2 + placed.bbox.height() - 1,
    )
}

fn bbox_edge_positions(min: Position, max: Position) -> Vec<Position> {
    let mut positions = Vec::new();

    for x in min.0..=max.0 {
        for y in min.1..=max.1 {
            for z in min.2..=max.2 {
                let boundary_axis_count = usize::from(x == min.0 || x == max.0)
                    + usize::from(y == min.1 || y == max.1)
                    + usize::from(z == min.2 || z == max.2);
                if boundary_axis_count >= 2 {
                    positions.push(Position(x, y, z));
                }
            }
        }
    }

    positions
}

fn world_size_for_positions(positions: &[Position]) -> DimSize {
    let mut max = Position(0, 0, 0);
    for position in positions {
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }
    DimSize(max.0 + 1, max.1 + 1, max.2 + 1)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::place_and_route::estimate::BoundingBox;
    use crate::world::block::BlockKind;
    use crate::world::position::Position;

    fn placed(origin: Position, bbox: BoundingBox) -> PlacedModule {
        PlacedModule {
            module_name: "child".to_owned(),
            candidate_index: 0,
            origin,
            bbox,
        }
    }

    #[test]
    fn placement_bbox_wireframe_world_draws_global_bbox_edges() {
        let world = placement_bbox_wireframe_world(&[placed(
            Position(2, 3, 1),
            BoundingBox {
                min: Position(10, 20, 5),
                max: Position(12, 22, 7),
            },
        )]);

        assert!(world[Position(2, 3, 1)].kind.is_cobble());
        assert!(world[Position(4, 5, 3)].kind.is_cobble());
        assert!(world[Position(3, 3, 1)].kind.is_cobble());
        assert!(world[Position(3, 4, 1)].kind.is_air());
        assert_eq!(world.size.0, 5);
        assert_eq!(world.size.1, 6);
        assert_eq!(world.size.2, 4);
    }

    #[test]
    fn placement_bbox_wireframe_world_keeps_multiple_module_boxes() {
        let world = placement_bbox_wireframe_world(&[
            placed(
                Position(1, 1, 0),
                BoundingBox {
                    min: Position(0, 0, 0),
                    max: Position(2, 2, 2),
                },
            ),
            placed(
                Position(8, 1, 0),
                BoundingBox {
                    min: Position(4, 5, 6),
                    max: Position(5, 6, 7),
                },
            ),
        ]);

        let cobble_count = world
            .iter_block()
            .into_iter()
            .filter(|(_, block)| matches!(block.kind, BlockKind::Cobble { .. }))
            .count();

        assert!(world[Position(1, 1, 0)].kind.is_cobble());
        assert!(world[Position(8, 1, 0)].kind.is_cobble());
        assert!(cobble_count > 20);
    }
}
