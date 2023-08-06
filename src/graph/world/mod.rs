use std::collections::{HashMap, HashSet};

use crate::{utils::Verify, world::position::Position};

use super::{Graph, GraphNodeId};

pub mod builder;

#[derive(Debug, Clone)]
pub struct WorldGraph {
    pub graph: Graph,
    pub positions: HashMap<GraphNodeId, Position>,
}

impl Verify for WorldGraph {
    fn verify(&self) -> eyre::Result<()> {
        todo!()
    }
}

// Calculate the number of cases of all possible combinations of
// graph nodes in which a block can be located
fn calculate_block_place_cases(graph: &WorldGraph) -> Vec<Vec<HashSet<GraphNodeId>>> {
    //

    todo!()
}
