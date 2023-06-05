use crate::{
    block::Block,
    common::{DimSize, Position},
};

pub struct World {
    pub size: DimSize,
    pub blocks: Vec<(Position, Block)>,
}
