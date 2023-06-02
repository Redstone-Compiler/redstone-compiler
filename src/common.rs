// 위치
pub struct Position(usize, usize, usize);

// 사이즈
pub struct DimSize(usize, usize, usize);

pub trait Verifier {
    fn verify(&self) -> bool;
}
