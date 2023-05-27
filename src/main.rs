pub struct Constraints {}

#[derive(Clone, Debug, Copy)]
pub enum BlockKind {
    Cobble,
    Redstone,
    Torch,
    Repeater,
    RedstoneBlock,
}

pub enum BlockState {
    None,
    Bottom,
    East,
    West,
    South,
    North,
}

pub struct Block {
    pub kind: BlockKind,
    pub state: BlockState,
}

pub struct Gate {}

pub struct LogicCircuit {}

pub struct ProcessingElement {}

fn main() {
    println!("");
}
