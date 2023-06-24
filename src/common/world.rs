use std::fmt::Debug;
use std::{
    collections::BTreeMap,
    ops::{Index, IndexMut},
};

use crate::{
    common::{DimSize, Position},
    graph::Graph,
};

use super::block::{Block, BlockKind};

#[derive(Debug, Clone)]
pub struct World {
    pub size: DimSize,
    pub blocks: Vec<(Position, Block)>,
}

#[derive(Clone)]
pub struct World3D {
    pub size: DimSize,
    // z, x, y
    pub map: Vec<Vec<Vec<Block>>>,
}

impl<'a> From<&'a World> for World3D {
    fn from(value: &'a World) -> Self {
        let mut block_map: BTreeMap<usize, &Block> = BTreeMap::default();

        for block in &value.blocks {
            block_map.insert(block.0.index(&value.size), &block.1);
        }

        let map: Vec<Vec<Vec<Block>>> = (0..value.size.2)
            .map(|z| {
                (0..value.size.1)
                    .map(|y| {
                        (0..value.size.0)
                            .map(|x| {
                                let pos = x + y * value.size.0 + z * value.size.0 * value.size.1;

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

impl Index<&Position> for World3D {
    type Output = Block;

    fn index(&self, index: &Position) -> &Self::Output {
        &self.map[index.2][index.1][index.0]
    }
}

impl IndexMut<&Position> for World3D {
    fn index_mut(&mut self, index: &Position) -> &mut Self::Output {
        &mut self.map[index.2][index.1][index.0]
    }
}

impl<'a> From<&'a World3D> for Graph {
    fn from(value: &'a World3D) -> Self {
        todo!()
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
                        })
                        .collect::<Vec<_>>()
                        .join("")
                )?;
            }
        }

        Ok(())
    }
}
