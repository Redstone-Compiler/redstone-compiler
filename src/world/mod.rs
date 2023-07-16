pub mod block;
pub mod gate;
pub mod position;
pub mod simulator;
pub mod world;

pub trait Verifier {
    fn verify(&self) -> bool;
}
