use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use super::{Graph, GraphNodeId};
use crate::graph::GraphNodeKind;
use crate::utils::Verify;
use crate::world::block::BlockKind;
use crate::world::position::Position;

pub mod builder;

#[derive(Debug, Default, Clone)]
pub struct WorldGraph {
    pub graph: Graph,
    pub positions: HashMap<GraphNodeId, Position>,
    pub routings: HashSet<GraphNodeId>,
}

impl Verify for WorldGraph {
    fn verify(&self) -> eyre::Result<()> {
        todo!()
    }
}

// Calculate the number of cases of all possible combinations of
// graph nodes in which a block can be located
fn _calculate_block_place_cases(graph: &WorldGraph) -> Vec<Vec<HashSet<GraphNodeId>>> {
    // torch must be required block

    // Find torches
    let _torches = graph.graph.nodes.iter().filter(|node| matches!(node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Torch{..}))).collect_vec();

    // Can place between repeater and redstone

    Vec::new()
}

// place torch and block, route redstone
//
