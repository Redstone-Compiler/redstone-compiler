use std::{collections::BTreeMap, ops::Index};

use crate::common::{DimSize, Position};

use super::block::Block;

#[derive(Debug, Clone)]
pub struct World {
    pub size: DimSize,
    pub blocks: Vec<(Position, Block)>,
}

#[derive(Debug, Clone)]
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
                (0..value.size.0)
                    .map(|x| {
                        (0..value.size.1)
                            .map(|y| {
                                // how to this?
                                // let pos = (x, y, z) * value.size;
                                let pos = x * value.size.0 + y * value.size.1 + z * value.size.2;

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
        &self.map[index.2][index.0][index.1]
    }
}
