// 위치
pub struct Position(pub usize, pub usize, pub usize);

// 사이즈
pub struct DimSize(pub usize, pub usize, pub usize);

pub trait Verifier {
    fn verify(&self) -> bool;
}
