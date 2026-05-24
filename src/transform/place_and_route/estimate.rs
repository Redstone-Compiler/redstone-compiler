use crate::world::position::Position;
use crate::world::World3D;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoundingBox {
    pub min: Position,
    pub max: Position,
}

impl BoundingBox {
    pub fn width(self) -> usize {
        self.max.0 - self.min.0 + 1
    }

    pub fn depth(self) -> usize {
        self.max.1 - self.min.1 + 1
    }

    pub fn height(self) -> usize {
        self.max.2 - self.min.2 + 1
    }

    pub fn volume(self) -> usize {
        self.width() * self.depth() * self.height()
    }

    pub fn extent_sum(self) -> usize {
        self.width() + self.depth() + self.height()
    }

    pub fn manhattan_span(self) -> usize {
        (self.max.0 - self.min.0) + (self.max.1 - self.min.1) + (self.max.2 - self.min.2)
    }
}

pub fn bounding_box(world: &World3D) -> Option<BoundingBox> {
    bounding_box_of_positions(world.iter_block().into_iter().map(|(position, _)| position))
}

pub fn bounding_box_of_positions<I>(positions: I) -> Option<BoundingBox>
where
    I: IntoIterator<Item = Position>,
{
    let mut positions = positions.into_iter();
    let first = positions.next()?;
    let mut min = first;
    let mut max = first;

    for position in positions {
        min.0 = min.0.min(position.0);
        min.1 = min.1.min(position.1);
        min.2 = min.2.min(position.2);
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }

    Some(BoundingBox { min, max })
}

pub fn world_compact_cost(world: &World3D) -> usize {
    let block_count = world.iter_block().len();
    let Some(bounds) = bounding_box(world) else {
        return 0;
    };

    block_count * 10 + bounds.volume() + bounds.extent_sum() * 5 + bounds.height() * 20
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{Block, BlockKind, Direction};
    use crate::world::position::DimSize;

    fn cobble() -> Block {
        Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        }
    }

    #[test]
    fn bounding_box_tracks_non_air_blocks() {
        let mut world = World3D::new(DimSize(6, 6, 4));
        world[Position(1, 2, 0)] = cobble();
        world[Position(4, 3, 2)] = cobble();

        let bounds = bounding_box(&world).unwrap();

        assert_eq!(bounds.min, Position(1, 2, 0));
        assert_eq!(bounds.max, Position(4, 3, 2));
        assert_eq!(bounds.width(), 4);
        assert_eq!(bounds.depth(), 2);
        assert_eq!(bounds.height(), 3);
        assert_eq!(bounds.volume(), 24);
    }

    #[test]
    fn bounding_box_of_positions_tracks_position_extents() {
        let bounds = bounding_box_of_positions([Position(1, 2, 0), Position(4, 3, 2)]).unwrap();

        assert_eq!(bounds.min, Position(1, 2, 0));
        assert_eq!(bounds.max, Position(4, 3, 2));
        assert_eq!(bounds.width(), 4);
        assert_eq!(bounds.depth(), 2);
        assert_eq!(bounds.height(), 3);
        assert_eq!(bounds.manhattan_span(), 6);
    }

    #[test]
    fn bounding_box_of_positions_returns_none_for_empty_iterators() {
        assert_eq!(bounding_box_of_positions([]), None);
    }

    #[test]
    fn compact_cost_prefers_tighter_worlds() {
        let mut compact = World3D::new(DimSize(6, 6, 4));
        let mut spread = World3D::new(DimSize(6, 6, 4));
        compact[Position(1, 1, 0)] = cobble();
        compact[Position(2, 1, 0)] = cobble();
        spread[Position(1, 1, 0)] = cobble();
        spread[Position(5, 5, 3)] = cobble();

        assert!(world_compact_cost(&compact) < world_compact_cost(&spread));
    }
}
