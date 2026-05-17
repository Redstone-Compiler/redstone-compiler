use std::collections::HashSet;

use crate::transform::place_and_route::place_bound::PlaceBound;
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::position::Position;
use crate::world::World3D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalPlacementError {
    OutOfBounds { position: Position },
    NoSupport { position: Position },
    CobbleConflict { position: Position },
    RedstoneConflict { position: Position },
    ShortCircuit { position: Position },
}

pub fn try_place_supported_redstone(
    world: &World3D,
    bound: PlaceBound,
    previous: Position,
    target: Position,
) -> Result<(World3D, Position), PhysicalPlacementError> {
    try_place_supported_redstone_with_exceptions(world, bound, previous, target, &HashSet::new())
}

pub fn try_place_supported_redstone_with_exceptions(
    world: &World3D,
    bound: PlaceBound,
    previous: Position,
    target: Position,
    extra_exceptions: &HashSet<Position>,
) -> Result<(World3D, Position), PhysicalPlacementError> {
    let redstone_pos = bound.position();
    if !world.size.bound_on(redstone_pos) {
        return Err(PhysicalPlacementError::OutOfBounds {
            position: redstone_pos,
        });
    }

    let Some(cobble_pos) = redstone_pos.down() else {
        return Err(PhysicalPlacementError::NoSupport {
            position: redstone_pos,
        });
    };
    if !world.size.bound_on(cobble_pos) {
        return Err(PhysicalPlacementError::OutOfBounds {
            position: cobble_pos,
        });
    }

    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    let mut cobble_exceptions = [previous, target, cobble_pos].into_iter().collect::<HashSet<_>>();
    cobble_exceptions.extend(extra_exceptions.iter().copied());
    if cobble_node.has_conflict(world, &cobble_exceptions) {
        return Err(PhysicalPlacementError::CobbleConflict {
            position: cobble_pos,
        });
    }

    let mut next_world = world.clone();
    next_world[cobble_pos] = cobble_node.block;

    let redstone_node = PlacedNode::new_redstone(redstone_pos);
    let mut redstone_exceptions = redstone_exceptions(previous, redstone_pos, bound, target);
    redstone_exceptions.extend(extra_exceptions.iter().copied());
    if redstone_node.has_conflict(&next_world, &redstone_exceptions) {
        return Err(PhysicalPlacementError::RedstoneConflict {
            position: redstone_pos,
        });
    }
    if redstone_node.has_short(world, &redstone_exceptions) {
        return Err(PhysicalPlacementError::ShortCircuit {
            position: redstone_pos,
        });
    }

    next_world[redstone_pos] = redstone_node.block;
    next_world.update_redstone_states(previous);
    next_world.update_redstone_states(redstone_pos);

    Ok((next_world, redstone_pos))
}

fn redstone_exceptions(
    previous: Position,
    redstone_pos: Position,
    bound: PlaceBound,
    target: Position,
) -> HashSet<Position> {
    let mut exceptions = [previous, redstone_pos, target, target.up()]
        .into_iter()
        .collect::<HashSet<_>>();
    if let Some(back) = redstone_pos.walk(bound.direction()) {
        exceptions.insert(back);
    }
    exceptions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::place_and_route::place_bound::{PlaceBound, PropagateType};
    use crate::world::block::{Block, BlockKind, Direction};
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

    fn switch() -> Block {
        Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::Bottom,
        }
    }

    fn redstone() -> Block {
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
    fn supported_redstone_places_cobble_and_wire() -> eyre::Result<()> {
        let mut world = World3D::new(DimSize(6, 6, 3));
        let source = Position(1, 1, 1);
        let target = Position(4, 1, 1);
        world[source] = switch();
        world[target] = switch();

        let (world, placed) = try_place_supported_redstone(
            &world,
            PlaceBound(PropagateType::Soft, Position(2, 1, 1), Direction::West),
            source,
            target,
        )
        .expect("supported redstone placement should succeed");

        assert_eq!(placed, Position(2, 1, 1));
        assert!(world[Position(2, 1, 0)].kind.is_cobble());
        assert!(world[Position(2, 1, 1)].kind.is_redstone());
        Ok(())
    }

    #[test]
    fn supported_redstone_rejects_z0_without_support_space() {
        let world = World3D::new(DimSize(6, 6, 2));

        let error = try_place_supported_redstone(
            &world,
            PlaceBound(PropagateType::Soft, Position(2, 1, 0), Direction::West),
            Position(1, 1, 0),
            Position(4, 1, 0),
        )
        .expect_err("z=0 redstone cannot have a support block below");

        assert_eq!(
            error,
            PhysicalPlacementError::NoSupport {
                position: Position(2, 1, 0)
            }
        );
    }

    #[test]
    fn supported_redstone_rejects_existing_block_conflict() {
        let mut world = World3D::new(DimSize(6, 6, 3));
        let source = Position(1, 1, 1);
        let target = Position(4, 1, 1);
        let occupied = Position(2, 1, 1);
        world[source] = switch();
        world[target] = switch();
        world[occupied] = redstone();

        let error = try_place_supported_redstone(
            &world,
            PlaceBound(PropagateType::Soft, occupied, Direction::West),
            source,
            target,
        )
        .expect_err("existing redstone is not an open placement target");

        assert_eq!(
            error,
            PhysicalPlacementError::RedstoneConflict { position: occupied }
        );
    }
}
