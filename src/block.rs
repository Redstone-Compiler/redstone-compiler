use crate::{
    common::{DimSize, Position},
    world::World,
};

pub enum Direction {
    None,
    Bottom,
    Top,
    East,
    West,
    South,
    North,
}

// 블럭의 종류
#[derive(Clone, Debug, Copy)]
pub enum BlockKind {
    Cobble {
        is_on: bool,
    },
    Switch {
        is_on: bool,
    },
    Redstone {
        is_enable: bool,
        is_on: bool,
        strength: usize,
    },
    Torch {
        is_on: bool,
    },
    Repeater {
        is_on: bool,
        is_locked: bool,
    },
    RedstoneBlock,
}

// 모든 물리적 소자의 최소 단위
pub struct Block {
    pub kind: BlockKind,
    pub direction: Direction,
}

impl From<Block> for World {
    fn from(value: Block) -> Self {
        Self {
            size: DimSize(1, 1, 1),
            blocks: vec![(Position(1, 1, 1), value)],
        }
    }
}