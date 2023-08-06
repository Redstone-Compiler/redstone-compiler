use std::collections::HashSet;

use disjoint_set::DisjointSet;
use itertools::Itertools;

use crate::{
    graph::{world::WorldGraph, GraphNodeKind},
    world::block::BlockKind,
};

pub struct WorldGraphTransformer {
    pub graph: WorldGraph,
}

impl WorldGraphTransformer {
    pub fn new(graph: WorldGraph) -> Self {
        Self { graph }
    }

    pub fn finish(self) -> WorldGraph {
        self.graph
    }

    pub fn fold_redstone(&mut self) {
        let nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. })))
            .collect_vec();

        // clustering
        let mut cluster = DisjointSet::new();
        for node in &nodes {
            cluster.make_set(node.id);

            let prodcuers = &self.graph.graph.producers[&node.id];
            let consumers = &self.graph.graph.consumers[&node.id];

            for candidate in prodcuers.iter().chain(consumers) {
                let node = self.graph.graph.find_node_by_id(*candidate).unwrap();

                if matches!(&node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. }))
                {
                    cluster.union(node.id, *candidate).unwrap();
                }
            }
        }

        let ids: HashSet<usize> = nodes.iter().map(|node| node.id).collect();
    }
}
