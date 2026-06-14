mod buffers;
mod compose;
mod decompose;
mod fold_or;
mod optimize;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet, VecDeque};

use disjoint_set::DisjointSet;
use itertools::Itertools;

use crate::graph::logic::LogicGraph;
use crate::graph::{GraphNodeId, SubGraphWithGraph};

pub struct LogicGraphTransformer {
    pub graph: LogicGraph,
}

impl LogicGraphTransformer {
    pub fn new(graph: LogicGraph) -> Self {
        Self { graph }
    }

    pub fn finish(self) -> LogicGraph {
        self.graph
    }

    pub fn cluster(&self, include_ouput_node: bool) -> Vec<SubGraphWithGraph> {
        let mut tags: HashMap<GraphNodeId, HashSet<GraphNodeId>> = HashMap::new();
        let mut queue = VecDeque::from(self.graph.graph.inputs());

        // TODO: Apply memoization
        while let Some(node_id) = queue.pop_front() {
            let mut inputs: HashSet<GraphNodeId> = HashSet::new();
            let node = self.graph.graph.find_node_by_id(node_id).unwrap();

            if node.inputs.is_empty() {
                inputs.insert(node_id);
            } else {
                for input in &node.inputs {
                    if let Some(tag_inputs) = tags.get(input) {
                        inputs.extend(tag_inputs);
                    }
                }
            }

            tags.entry(node_id).or_default().extend(inputs);
            queue.extend(node.outputs.iter());
        }

        let tags: HashMap<GraphNodeId, String> = tags
            .into_iter()
            .map(|(id, inputs)| {
                let mut inputs = inputs.into_iter().collect_vec();
                inputs.sort();
                (id, inputs.into_iter().join(","))
            })
            .collect();

        // for (node_id, inputs) in &tags {
        //     self.graph.graph.find_node_by_id_mut(*node_id).unwrap().tag = inputs.clone();
        // }

        let mut uf = DisjointSet::new();
        for node_id in self.graph.graph.topological_order() {
            uf.make_set(node_id);

            let node = self.graph.graph.find_node_by_id(node_id).unwrap();
            if !node.inputs.is_empty() {
                for input in &node.inputs {
                    if tags[input] == tags[&node_id] {
                        uf.union(*input, node_id).unwrap();
                    }
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<GraphNodeId>> = HashMap::new();
        for node in &self.graph.graph.nodes {
            if !include_ouput_node && node.outputs.is_empty() {
                continue;
            }

            clusters
                .entry(uf.find(node.id).unwrap())
                .or_default()
                .push(node.id);
        }

        clusters
            .into_values()
            .map(|nodes| SubGraphWithGraph::from(&self.graph.graph, nodes))
            .collect_vec()
    }
}
