pub enum Direction {
    None,
    Bottom,
    Top,
    East,
    West,
    South,
    North,
}

pub enum ConstraintsKind {
    // 모든 방향에 영향을 줄 수 있음
    On,
    // 특정 방향에만 영향을 줄 수 있음 (e.g. Repeater)
    OnDirection(Direction),
}

// 다른 블럭과 겹칠 경우 영향을 줄 수 있는 부분에 대해 정의한 것
pub struct Constraints {
    // 영향을 줄 수 있는 범위, 0일 경우 아무런 영향 없음
    // 1인 경우 최대 한 칸 범위 내 블록에 영향을 미칠 수 있음
    pub range: usize,
    // 영향을 줄 수 있는 경우, 특정 방향에만 영향을 주는 지에 대한 여부
    pub direction_free: bool,
    // 영향을 주는 위치, 제약의 종류
    pub contents: Vec<(usize, Direction, ConstraintsKind)>,
}

// 블럭의 종류
#[derive(Clone, Debug, Copy)]
pub enum BlockKind {
    Cobble,
    Redstone { is_on: bool, strength: usize },
    Torch { is_on: bool },
    Repeater { is_on: bool, is_locked: bool },
    RedstoneBlock,
}

// 모든 물리적 소자의 최소 단위
pub struct Block {
    pub kind: BlockKind,
    pub direction: Direction,
}

// 사이즈
pub struct DimSize(usize, usize, usize);

// 위치
pub struct Position(usize, usize, usize);

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

// Gate가 모여서 만들어진 기능적으로 추상화된 하위레벨 회로
// 가령 Full Adder 하나는 Logic Circuit으로 표현된다.
pub struct LogicCircuit {}

// LogicCircuit들이 모여서 만들어진, 어떤 사용자 동작을 수행하는 상위레벨 회로
pub struct ProcessingElement {}

fn main() {
    println!("");
}
