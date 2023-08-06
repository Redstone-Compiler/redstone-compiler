pub trait Verify {
    fn verify(&self) -> eyre::Result<()>;
}
