use std::cmp;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use block::{Block, BlockKind, Direction, RedstoneState, RedstoneStateType};
use itertools::Itertools;
use position::{DimSize, Position};

pub mod block;
pub mod gate;
pub mod position;
pub mod simulator;

#[derive(Debug, Clone)]
pub struct World {
    pub size: DimSize,
    pub blocks: Vec<(Position, Block)>,
}

impl World {
    pub fn new(size: DimSize) -> Self {
        Self {
            size,
            blocks: Default::default(),
        }
    }
}

#[derive(Clone, Default)]
pub struct World3D {
    pub size: DimSize,
    // [z][y][x]. Clones share planes and rows. A write detaches only its
    // plane's row handles and its row's blocks; untouched planes remain shared.
    pub map: Vec<Arc<Vec<Arc<Vec<Block>>>>>,
}

impl World3D {
    pub fn new(size: DimSize) -> Self {
        Self {
            size,
            map: vec![Arc::new(vec![Arc::new(vec![Block::default(); size.0]); size.1]); size.2],
        }
    }

    pub fn iter_pos(&self) -> Vec<Position> {
        let mut result = Vec::new();
        let (z, y, x) = (self.map.len(), self.map[0].len(), self.map[0][0].len());

        for z in 0..z {
            for y in 0..y {
                for x in 0..x {
                    result.push(Position(x, y, z));
                }
            }
        }

        result
    }

    pub fn iter_block(&self) -> Vec<(Position, Block)> {
        self.iter_pos()
            .into_iter()
            .filter(|&pos| !self[pos].kind.is_air())
            .map(|pos| (pos, self[pos]))
            .collect_vec()
    }

    pub fn initialize_redstone_states(&mut self) {
        self.iter_pos()
            .iter()
            .for_each(|pos| self.update_redstone_states(*pos));
    }

    pub fn update_redstone_states(&mut self, pos: Position) {
        let BlockKind::Redstone {
            on_count, strength, ..
        } = self[pos].kind
        else {
            return;
        };

        let state = self.redstone_shape(pos, true);

        self[pos].kind = BlockKind::Redstone {
            on_count,
            state,
            strength,
        };
    }

    /// The directions dust at `pos` connects to and so points into: dust,
    /// torches, switches, and redstone blocks beside it, a repeater at either
    /// end of its axis (never a side), and dust a step up or down unless a
    /// block cuts the stair. One connection makes a line through the dust;
    /// none makes a cross. Java Edition connects dust to a repeater's output
    /// since 1.0.0, so such dust is a line along the repeater, not a cross:
    /// it does not power the blocks beside it. `repeater_outputs` false
    /// leaves those connections out, the shape this simulator used before it
    /// followed that rule (the global router still searches with it).
    pub fn redstone_shape(&self, pos: Position, repeater_outputs: bool) -> RedstoneStateType {
        let mut state = 0;

        let has_up_block = self.size.bound_on(pos.up()) && self[pos.up()].kind.is_cobble();

        pos.cardinal().iter().for_each(|&pos_src| {
            if !self.size.bound_on(pos_src) {
                return;
            }

            let flat_check = self[pos_src].kind.is_stick_to_redstone();
            let up_check = !has_up_block
                && self.size.bound_on(pos.up())
                && self[pos_src.up()].kind.is_redstone();
            let down_check = !self[pos_src].kind.is_cobble()
                && pos_src
                    .down()
                    .is_some_and(|pos| self[pos].kind.is_redstone());
            let flat_repeater_check = self[pos_src].kind.is_repeater()
                && (pos_src.walk(self[pos_src].direction) == Some(pos)
                    || (repeater_outputs
                        && pos_src.walk(self[pos_src].direction.inverse()) == Some(pos)));

            if !(flat_check || flat_repeater_check || up_check || down_check) {
                return;
            }

            state |= match pos.diff(pos_src) {
                Direction::East => RedstoneState::East,
                Direction::West => RedstoneState::West,
                Direction::South => RedstoneState::South,
                Direction::North => RedstoneState::North,
                _ => unreachable!(),
            } as usize;
        });

        if state.count_ones() == 1 {
            if state & RedstoneState::Horizontal as usize > 0 {
                state |= RedstoneState::Horizontal as usize;
            } else {
                state |= RedstoneState::Vertical as usize;
            }
        } else if state == 0 {
            state |= RedstoneState::Cardinal as usize;
        }
        state
    }

    pub fn concat(&self, other: &World3D, direction: Direction) -> Self {
        match direction {
            Direction::None => unreachable!(),
            Direction::Bottom | Direction::West | Direction::South => {
                other.concat(self, direction.inverse())
            }
            Direction::Top | Direction::North | Direction::East => {
                let mut world = Self::new(DimSize(
                    if matches!(direction, Direction::East) {
                        self.size.0 + other.size.0
                    } else {
                        cmp::max(self.size.0, other.size.0)
                    },
                    if matches!(direction, Direction::North) {
                        self.size.1 + other.size.1
                    } else {
                        cmp::max(self.size.1, other.size.1)
                    },
                    if matches!(direction, Direction::Top) {
                        self.size.2 + other.size.2
                    } else {
                        cmp::max(self.size.2, other.size.2)
                    },
                ));

                for (pos, block) in self.iter_block() {
                    world[pos] = block;
                }

                for (mut pos, block) in other.iter_block() {
                    match direction {
                        Direction::East => pos.0 += self.size.0,
                        Direction::North => pos.1 += self.size.1,
                        Direction::Top => pos.2 += self.size.2,
                        _ => (),
                    }
                    world[pos] = block;
                }

                world
            }
        }
    }

    pub fn concat_tiled(worlds: Vec<World3D>) -> Self {
        let east_chunk_len = (worlds.len() as f32).sqrt() as usize + 1;

        let mut east_worlds = worlds
            .into_iter()
            .chunks(east_chunk_len)
            .into_iter()
            .map(|chunk| chunk.collect_vec())
            .map(|mut worlds| {
                let mut world = worlds.remove(0);
                for other in worlds {
                    world = world.concat(&other, Direction::East);
                }
                world
            })
            .collect_vec();

        let mut world = east_worlds.remove(0);
        for other in east_worlds {
            world = world.concat(&other, Direction::North);
        }
        world
    }
}

impl<'a> From<&'a World> for World3D {
    fn from(value: &'a World) -> Self {
        let mut world = Self::new(value.size);
        // Sparse worlds may contain repeated coordinates: applying in order
        // preserves the previous last-entry-wins semantics, including Air.
        for (position, block) in &value.blocks {
            // Never flatten invalid coordinates: an oversized x could alias a
            // different valid row in the former linear-index lookup.
            if value.size.bound_on(*position) {
                world[*position] = *block;
            }
        }
        world
    }
}

impl<'a> From<&'a World3D> for World {
    fn from(value: &'a World3D) -> Self {
        Self {
            size: value.size,
            blocks: value.iter_block(),
        }
    }
}

impl Index<Position> for World3D {
    type Output = Block;

    fn index(&self, index: Position) -> &Self::Output {
        &self.map[index.2][index.1][index.0]
    }
}

impl IndexMut<Position> for World3D {
    fn index_mut(&mut self, index: Position) -> &mut Self::Output {
        let plane = Arc::make_mut(&mut self.map[index.2]);
        &mut Arc::make_mut(&mut plane[index.1])[index.0]
    }
}

impl Debug for World3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (height, plane) in self.map.iter().enumerate().rev() {
            writeln!(f, "h={height:?}")?;

            for row in plane.iter().rev() {
                writeln!(
                    f,
                    "  {}",
                    row.iter()
                        .map(|block| match block.kind {
                            BlockKind::Air => ".",
                            BlockKind::Cobble { .. } => "c",
                            BlockKind::Switch { .. } => "s",
                            BlockKind::Redstone { .. } => "r",
                            BlockKind::Torch { .. } => "t",
                            BlockKind::Repeater { .. } => "t",
                            BlockKind::RedstoneBlock => "b",
                            BlockKind::Piston { .. } => "p",
                        })
                        .collect::<Vec<_>>()
                        .join("")
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod storage_tests {
    use super::*;

    #[test]
    fn cloning_shares_planes_and_rows_and_detaches_only_written_regions() {
        let p = Position(1, 1, 1);
        let mut original = World3D::new(DimSize(4, 3, 2));
        original[p].kind = BlockKind::Switch { is_on: false };
        let mut branch = original.clone();
        let sibling = original.clone();
        for z in 0..2 {
            assert!(Arc::ptr_eq(&original.map[z], &branch.map[z]));
            for y in 0..3 {
                assert!(Arc::ptr_eq(&original.map[z][y], &branch.map[z][y]));
            }
        }
        branch[p].kind = BlockKind::Switch { is_on: true };
        assert!(Arc::ptr_eq(&original.map[0], &branch.map[0]));
        assert!(!Arc::ptr_eq(&original.map[1], &branch.map[1]));
        assert!(!Arc::ptr_eq(&original.map[1][1], &branch.map[1][1]));
        for z in 0..2 {
            for y in 0..3 {
                if (z, y) != (1, 1) {
                    assert!(Arc::ptr_eq(&original.map[z][y], &branch.map[z][y]));
                }
            }
        }
        assert!(!original[p].kind.is_powered());
        assert!(!sibling[p].kind.is_powered());
        assert!(branch[p].kind.is_powered());
        let detached_plane = Arc::as_ptr(&branch.map[1]);
        let detached_row = Arc::as_ptr(&branch.map[1][1]);
        branch[Position(2, 1, 1)].kind = BlockKind::RedstoneBlock;
        assert_eq!(Arc::as_ptr(&branch.map[1]), detached_plane);
        assert_eq!(Arc::as_ptr(&branch.map[1][1]), detached_row);
        assert!(original[Position(2, 1, 1)].kind.is_air());
        // Mutating the source after the fork must also preserve both branches.
        original[p].kind = BlockKind::RedstoneBlock;
        assert_eq!(sibling[p].kind, BlockKind::Switch { is_on: false });
        assert_eq!(branch[p].kind, BlockKind::Switch { is_on: true });
    }

    #[test]
    fn sparse_conversion_keeps_last_duplicates_and_ignores_invalid_coordinates() {
        let p = Position(1, 1, 1);
        let q = Position(2, 1, 1);
        let powered = Block {
            kind: BlockKind::RedstoneBlock,
            ..Default::default()
        };
        let world = World {
            size: DimSize(3, 2, 2),
            blocks: vec![
                (p, powered),
                (p, Block::default()),
                (q, Block::default()),
                (q, powered),
                (Position(3, 0, 0), powered), // must not alias (0,1,0)
                (Position(0, 2, 0), powered), // must not alias (0,0,1)
                (Position(0, 0, 2), powered),
                (Position(usize::MAX, usize::MAX, usize::MAX), powered),
            ],
        };
        let dense = World3D::from(&world);
        assert_eq!(dense.iter_block(), vec![(q, powered)]);
        assert!(dense[p].kind.is_air());
        assert!(dense[Position(0, 1, 0)].kind.is_air());
        assert!(dense[Position(0, 0, 1)].kind.is_air());
    }

    #[test]
    fn sparse_world_conversion_and_shared_empty_rows_preserve_coordinates() {
        let mut empty = World3D::new(DimSize(3, 2, 2));
        empty[Position(2, 1, 1)].kind = BlockKind::RedstoneBlock;
        assert_eq!(empty.iter_block().len(), 1);
        assert!(empty[Position(2, 0, 0)].kind.is_air());
        let sparse = World::from(&empty);
        let mut roundtrip = World3D::from(&sparse);
        assert_eq!(roundtrip.iter_block(), empty.iter_block());
        let branch = roundtrip.clone();
        roundtrip[Position(2, 1, 1)] = Block::default();
        assert!(roundtrip.iter_block().is_empty());
        assert_eq!(branch.iter_block(), empty.iter_block());
    }
}
