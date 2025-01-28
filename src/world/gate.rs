use super::block::Block;
use super::position::{DimSize, Position};
use super::world::World;

// 게이트의 종류
pub enum GateKind {
    Not,
    And,
    Or,
    NAnd,
    NOr,
    Xor,
}

// Block이 모여서 만들어진 논리의 최소 단위
pub struct Gate {
    // 게이트 사이즈
    pub size: DimSize,
    // 블럭들
    pub blocks: Vec<(Position, Block)>,
    // 인풋들의 인덱스
    pub inputs: Vec<usize>,
    // 아웃풋들의 인덱스
    pub outputs: Vec<usize>,
    // 게이트 종류
    pub kind: GateKind,
}

impl From<Gate> for World {
    fn from(value: Gate) -> Self {
        Self {
            size: value.size,
            blocks: value.blocks,
        }
    }
}
