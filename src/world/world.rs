use std::cmp;
use std::fmt::Debug;
use std::{
    collections::BTreeMap,
    ops::{Index, IndexMut},
};

use itertools::Itertools;

use super::block::{Block, BlockKind, Direction, RedstoneState};
use super::position::{DimSize, Position, PositionIndex};

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

#[derive(Clone)]
pub struct World3D {
    pub size: DimSize,
    // z, y, z
    pub map: Vec<Vec<Vec<Block>>>,
}

impl World3D {
    pub fn new(size: DimSize) -> Self {
        Self {
            size,
            map: vec![vec![vec![Block::default(); size.0]; size.1]; size.2],
        }
    }

    pub fn iter_pos(&self) -> Vec<Position> {
        let mut result = Vec::new();
        let (z, y, x) = (self.map.len(), self.map[0].len(), self.map[0][1].len());

        for z in 0..z {
            for y in 0..y {
                for x in 0..x {
                    result.push(Position(x, y, z));
                }
            }
        }

        result
    }

    pub fn iter_block(&self) -> Vec<(Position, &Block)> {
        self.iter_pos()
            .into_iter()
            .filter(|&pos| !self[pos].kind.is_air())
            .map(|pos| (pos, &self[pos]))
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

        let mut state = 0;

        let has_up_block = self[pos.up()].kind.is_cobble();

        pos.cardinal().iter().for_each(|&pos_src| {
            let flat_check = self[pos_src].kind.is_stick_to_redstone();
            let up_check = !has_up_block && self[pos_src.up()].kind.is_redstone();
            let down_check = !self[pos_src].kind.is_cobble()
                && pos_src
                    .down()
                    .map_or(false, |pos| self[pos].kind.is_redstone());
            let flat_repeater_check = self[pos_src].kind.is_repeater()
                && pos_src
                    .walk(self[pos_src].direction)
                    .map_or(false, |walk| walk == pos);

            if !(flat_check || flat_repeater_check) && !(up_check || down_check) {
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
        }

        self[pos].kind = BlockKind::Redstone {
            on_count,
            state,
            strength,
        };
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
                    world[pos] = block.clone();
                }

                for (mut pos, block) in other.iter_block() {
                    match direction {
                        Direction::East => pos.0 += self.size.0,
                        Direction::North => pos.1 += self.size.1,
                        Direction::Top => pos.2 += self.size.2,
                        _ => (),
                    }
                    world[pos] = block.clone();
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
        let mut block_map: BTreeMap<PositionIndex, &Block> = BTreeMap::default();

        for block in &value.blocks {
            block_map.insert(block.0.index(&value.size), &block.1);
        }

        let map: Vec<Vec<Vec<Block>>> = (0..value.size.2)
            .map(|z| {
                (0..value.size.1)
                    .map(|y| {
                        (0..value.size.0)
                            .map(|x| {
                                let pos = PositionIndex(
                                    x + y * value.size.0 + z * value.size.0 * value.size.1,
                                );

                                block_map
                                    .get(&pos)
                                    .map(|block| (*block).clone())
                                    .unwrap_or(Block::default())
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self {
            size: value.size,
            map,
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
        &mut self.map[index.2][index.1][index.0]
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
