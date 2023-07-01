use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;

use crate::{common::block::Block, logic::Logic};

pub mod builder;
pub mod graphviz;

pub type GraphNodeId = usize;

#[derive(Default, Debug, Clone)]
pub enum GraphNodeKind {
    #[default]
    None,
    Input(String),
    Block(Block),
    Logic(Logic),
    Output(String),
}

impl GraphNodeKind {
    pub fn name(&self) -> String {
        match self {
            GraphNodeKind::None => format!("None"),
            GraphNodeKind::Input(input) => format!("Input {input}"),
            GraphNodeKind::Block(block) => block.kind.name(),
            GraphNodeKind::Logic(logic) => logic.logic_type.name(),
            GraphNodeKind::Output(output) => format!("Output {output}"),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct GraphNode {
    id: GraphNodeId,
    kind: GraphNodeKind,
    inputs: Vec<GraphNodeId>,
    outputs: Vec<GraphNodeId>,
}

#[derive(Default, Debug, Clone)]
pub struct Graph {
    nodes: Vec<GraphNode>,
    producers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
    consumers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
}

impl Graph {
    pub fn to_petgraph(&self) -> petgraph::Graph<(), ()> {
        self.into()
    }

    pub fn inputs(&self) -> Vec<GraphNodeId> {
        self.nodes
            .iter()
            .filter(|node| node.inputs.is_empty())
            .map(|node| node.id)
            .collect()
    }

    pub fn outputs(&self) -> Vec<GraphNodeId> {
        self.nodes
            .iter()
            .filter(|node| node.outputs.is_empty())
            .map(|node| node.id)
            .collect()
    }

    pub fn concat(&mut self, mut other: Self) {
        if let Some(id) = self.max_node_id() {
            other.rebuild_node_id_base(id + 1);
        }

        self.nodes.extend(other.nodes);
        self.producers.extend(other.producers);
        self.consumers.extend(other.consumers);
    }

    pub fn merge(&mut self, mut other: Self) {}

    pub fn replace_input_name(&mut self, from: &str, to: String) -> eyre::Result<()> {
        Ok(())
    }

    pub fn replace_output_name(&mut self, from: &str, to: String) -> eyre::Result<()> {
        Ok(())
    }

    pub fn has_cycle(&self) -> bool {
        petgraph::algo::is_cyclic_directed(&self.to_petgraph())
    }

    pub fn max_node_id(&self) -> Option<GraphNodeId> {
        self.nodes
            .iter()
            .max_by_key(|node| node.id)
            .map(|node| node.id)
    }

    fn rebuild_node_id_base(&mut self, base_index: usize) {
        for node in &mut self.nodes {
            node.id += base_index;
            for input in &mut node.inputs {
                *input += base_index;
            }
            for output in &mut node.outputs {
                *output += base_index;
            }
        }

        self.build_producers();
        self.build_consumers();
    }

    pub fn rebuild_node_ids(self) -> Self {
        let index_map = petgraph::algo::toposort(&self.to_petgraph(), None)
            .unwrap()
            .iter()
            .enumerate()
            .map(|index| (index.1.index(), index.0))
            .collect::<HashMap<_, _>>();

        let mut nodes = self
            .nodes
            .into_iter()
            .map(|node| GraphNode {
                id: index_map[&node.id],
                inputs: node.inputs.iter().map(|index| index_map[index]).collect(),
                outputs: node.outputs.iter().map(|index| index_map[index]).collect(),
                kind: node.kind,
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.id);

        let mut result = Self { nodes, ..self };
        result.build_producers();
        result.build_consumers();
        result
    }

    // outputs이 반드시 determine 되어야함
    pub fn build_inputs(&mut self) {
        let mut input_map: HashMap<usize, Vec<usize>> = HashMap::new();

        self.nodes.iter().for_each(|node| {
            node.outputs
                .iter()
                .for_each(|&id| input_map.entry(id).or_default().push(node.id));
        });

        for node in self.nodes.iter_mut() {
            node.inputs = input_map
                .get(&node.id)
                .map(|ids| ids.clone())
                .unwrap_or_default();
        }
    }

    // inputs이 반드시 determine 되어야함
    pub fn build_outputs(&mut self) {
        let mut output_map: HashMap<usize, Vec<usize>> = HashMap::new();

        self.nodes.iter().for_each(|node| {
            node.inputs
                .iter()
                .for_each(|&id| output_map.entry(id).or_default().push(node.id));
        });

        for node in self.nodes.iter_mut() {
            node.outputs = output_map
                .get(&node.id)
                .map(|ids| ids.clone())
                .unwrap_or_default();
        }
    }

    pub fn build_producers(&mut self) {
        let mut producers: HashMap<GraphNodeId, Vec<GraphNodeId>> = HashMap::new();
        self.nodes.iter().for_each(|node| {
            producers
                .entry(node.id)
                .or_default()
                .extend(node.inputs.clone());
        });
        self.producers = producers;
    }

    pub fn build_consumers(&mut self) {
        let mut consumers: HashMap<GraphNodeId, Vec<GraphNodeId>> = HashMap::new();
        self.nodes.iter().for_each(|node| {
            consumers
                .entry(node.id)
                .or_default()
                .extend(node.outputs.clone());
        });
        self.consumers = consumers;
    }
}

impl From<&Graph> for petgraph::Graph<(), ()> {
    fn from(value: &Graph) -> Self {
        petgraph::Graph::<(), ()>::from_edges(
            value
                .nodes
                .iter()
                .map(|node| {
                    node.outputs
                        .iter()
                        .map(|&id| (NodeIndex::new(node.id), NodeIndex::new(id.into())))
                        .collect::<Vec<_>>()
                })
                .flatten(),
        )
    }
}
