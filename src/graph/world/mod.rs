use std::collections::HashMap;

use crate::world::position::Position;

use super::{Graph, GraphNodeId};

#[derive(Debug, Clone)]
pub struct WorldGraph {
    pub graph: Graph,
    pub positions: HashMap<GraphNodeId, Position>,
}
