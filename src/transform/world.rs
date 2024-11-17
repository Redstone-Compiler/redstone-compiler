use std::collections::{HashMap, HashSet};

use disjoint_set::DisjointSet;
use itertools::Itertools;

use crate::{
    graph::{world::WorldGraph, GraphNode, GraphNodeId, GraphNodeKind},
    world::block::{Block, BlockKind, Direction},
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
        let mut nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. })))
            .collect_vec();

        nodes.sort_by(|a, b| a.id.cmp(&b.id));

        // clustering
        let mut cluster = DisjointSet::new();

        for node in &nodes {
            cluster.make_set(node.id);
        }

        for node in &nodes {
            let producers = &self.graph.graph.producers[&node.id];
            let consumers = &self.graph.graph.consumers[&node.id];

            for candidate in producers.iter().chain(consumers) {
                let Some(candidate_node) = self.graph.graph.find_node_by_id(*candidate) else {
                    unreachable!();
                };

                if matches!(&candidate_node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. }))
                {
                    cluster.union(node.id, *candidate).unwrap();
                }
            }
        }

        let ids: HashSet<GraphNodeId> = nodes.iter().map(|node| node.id).collect();
        let mut group_inputs: HashMap<usize, HashSet<GraphNodeId>> = HashMap::new();
        let mut group_outputs: HashMap<usize, HashSet<GraphNodeId>> = HashMap::new();

        let mut group_ids: HashSet<usize> = HashSet::new();

        for id in &ids {
            // make clustered id group
            let group_id = cluster.find(*id).unwrap();
            group_ids.insert(group_id);

            // collect group input outputs
            group_inputs.entry(group_id).or_default().extend(
                self.graph.graph.producers[id]
                    .clone()
                    .into_iter()
                    .filter(|id| !ids.contains(id)),
            );
            group_outputs.entry(group_id).or_default().extend(
                self.graph.graph.consumers[id]
                    .clone()
                    .into_iter()
                    .filter(|id| !ids.contains(id)),
            );
        }

        // make clustered node
        let mut next_id = self.graph.graph.max_node_id().unwrap();

        // remove redstone node
        for id in &ids {
            self.graph.graph.remove_by_node_id_lazy(*id);
        }

        for group_id in &group_ids {
            next_id += 1;

            let node = GraphNode {
                id: next_id,
                kind: GraphNodeKind::Block(Block {
                    kind: BlockKind::Redstone {
                        on_count: 0,
                        state: 0,
                        strength: 0,
                    },
                    direction: Direction::None,
                }),
                // TODO: optimize this
                inputs: group_inputs[group_id].clone().into_iter().collect_vec(),
                outputs: group_outputs[group_id].clone().into_iter().collect_vec(),
                ..Default::default()
            };

            group_inputs[group_id].iter().for_each(|&conn| {
                self.graph
                    .graph
                    .find_node_by_id_mut(conn)
                    .unwrap()
                    .outputs
                    .push(next_id);
            });
            group_outputs[group_id].iter().for_each(|&conn| {
                self.graph
                    .graph
                    .find_node_by_id_mut(conn)
                    .unwrap()
                    .inputs
                    .push(next_id);
            });

            self.graph.routings.insert(node.id);
            self.graph.graph.nodes.push(node);
        }

        self.graph.graph.build_inputs();
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
    }

    pub fn remove_redstone(&mut self) {
        self.remove_specific_kind_of_block(|kind| matches!(&kind, BlockKind::Redstone { .. }));
    }

    pub fn remove_repeater(&mut self) {
        self.remove_specific_kind_of_block(|kind| matches!(&kind, BlockKind::Repeater { .. }));
    }

    fn remove_specific_kind_of_block<F>(&mut self, filter: F)
    where
        F: Fn(&BlockKind) -> bool,
    {
        let nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.kind, GraphNodeKind::Block(block) if filter(&block.kind)))
            .map(|node| node.id)
            .collect_vec();

        for node in nodes {
            self.graph.graph.remove_and_reconnect_by_node_id_lazy(node);
        }

        self.graph.graph.build_inputs();
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        graph::{graphviz::ToGraphvizGraph, world::builder::WorldGraphBuilder},
        nbt::NBTRoot,
    };

    use super::WorldGraphTransformer;

    #[test]
    fn unittest_fold_redstone() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/alu.nbt")?;
        let g = WorldGraphBuilder::new(&nbt.to_world()).build();

        let mut transform = WorldGraphTransformer::new(g);
        transform.remove_redstone();
        transform.remove_repeater();
        println!("{}", transform.finish().to_graphviz());

        Ok(())
    }
}
