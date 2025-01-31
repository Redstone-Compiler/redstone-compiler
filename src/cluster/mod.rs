use crate::graph::GraphNodeId;

#[derive(Debug, Clone, PartialEq)]
pub enum ClusteredType {
    Cluster(Vec<GraphNodeId>),
    Weighted(isize),
}

impl ClusteredType {
    pub fn name(&self) -> String {
        match self {
            ClusteredType::Cluster(node_ids) => format!("Cluster {node_ids:?}"),
            ClusteredType::Weighted(w) => format!("{w}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Clustered {
    pub clustered_type: ClusteredType,
}
