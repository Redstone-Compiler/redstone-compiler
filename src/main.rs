mod block;
mod circuit;
mod common;
mod estimator;
mod gate;
mod pe;
mod simulator;
mod world;

// pub enum ConstraintsKind {
//     // 모든 방향에 영향을 줄 수 있음
//     On,
//     // 특정 방향에만 영향을 줄 수 있음 (e.g. Repeater)
//     OnDirection(Direction),
// }

// // 다른 블럭과 겹칠 경우 영향을 줄 수 있는 부분에 대해 정의한 것
// pub struct Constraints {
//     // 영향을 줄 수 있는 범위, 0일 경우 아무런 영향 없음
//     // 1인 경우 최대 한 칸 범위 내 블록에 영향을 미칠 수 있음
//     pub range: usize,
//     // 영향을 줄 수 있는 경우, 특정 방향에만 영향을 주는 지에 대한 여부
//     pub direction_free: bool,
//     // 영향을 주는 위치, 제약의 종류
//     pub contents: Vec<(usize, Direction, ConstraintsKind)>,
// }

fn main() {
    println!("");
}
