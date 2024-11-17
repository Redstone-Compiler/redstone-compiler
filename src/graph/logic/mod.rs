use std::ops::Deref;

use super::Graph;

pub mod builder;

#[derive(Debug, Clone, Deref)]
pub struct LogicGraph {
    #[deref]
    pub graph: Graph,
}
