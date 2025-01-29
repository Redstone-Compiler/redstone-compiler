use super::Graph;
use crate::transform::logic::LogicGraphTransformer;

pub mod builder;

#[derive(Debug, Clone, derive_more::Deref)]
pub struct LogicGraph {
    #[deref]
    pub graph: Graph,
}

impl LogicGraph {
    pub fn prepare_place(self) -> eyre::Result<Self> {
        let mut transform = LogicGraphTransformer::new(self);
        transform.decompose_xor()?;
        transform.decompose_and()?;
        transform.remove_double_neg_expression();
        Ok(transform.finish())
    }
}
