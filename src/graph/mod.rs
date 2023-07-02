use std::collections::{HashMap, HashSet};

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

    pub fn as_input(&self) -> &String {
        match self {
            GraphNodeKind::Input(name) => &name,
            _ => unreachable!(),
        }
    }

    pub fn as_output(&self) -> &String {
        match self {
            GraphNodeKind::Output(name) => &name,
            _ => unreachable!(),
        }
    }

    pub fn as_input_mut(&mut self) -> &mut String {
        match self {
            GraphNodeKind::Input(name) => name,
            _ => unreachable!(),
        }
    }

    pub fn as_output_mut(&mut self) -> &mut String {
        match self {
            GraphNodeKind::Output(name) => name,
            _ => unreachable!(),
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

    // src와 other의 input이 같은 것 끼리 연결하여 merge함
    pub fn merge_by_input(&mut self, mut other: Self) {
        if let Some(id) = self.max_node_id() {
            other.rebuild_node_id_base(id + 1);
        }

        let src_inputs: HashMap<String, GraphNodeId> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) => Some((name.to_owned(), node.id)),
                _ => None,
            })
            .collect();

        let replace_targets: Vec<(GraphNodeId, GraphNodeId)> = other
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) if src_inputs.contains_key(name) => {
                    Some((node.id, *src_inputs.get(name).unwrap()))
                }
                _ => None,
            })
            .collect();

        for (from, to) in replace_targets {
            let mut node = other.nodes.iter_mut().find(|node| node.id == from).unwrap();

            self.get_node_by_id(to)
                .unwrap()
                .outputs
                .extend(node.outputs.clone());
            node.id = to;
        }

        other.build_producers();
        other.build_consumers();

        self.nodes
            .extend(other.nodes.into_iter().filter(|node| match &node.kind {
                GraphNodeKind::Input(name) if src_inputs.contains_key(name) => false,
                _ => true,
            }));
        self.producers.extend(other.producers);
        self.consumers.extend(other.consumers);
    }

    // src의 output을 target의 input과 연결하여 merge함
    pub fn merge_by_outin(&mut self, mut target: Self, out_in: Vec<(&str, &str)>) {
        if let Some(id) = self.max_node_id() {
            target.rebuild_node_id_base(id + 1);
        }

        let outputs: HashSet<_> = out_in.iter().map(|(output, _)| *output).collect();
        let inputs: HashSet<_> = out_in.iter().map(|(_, input)| *input).collect();

        let src_outputs: HashMap<String, GraphNodeId> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) if outputs.contains(name.as_str()) => {
                    Some((name.to_owned(), node.inputs[0]))
                }
                _ => None,
            })
            .collect();

        let target_inputs: HashMap<String, Vec<GraphNodeId>> = target
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) if inputs.contains(name.as_str()) => {
                    Some((name.to_owned(), node.outputs.clone()))
                }
                _ => None,
            })
            .collect();

        for (output, input) in out_in {
            let out_node = src_outputs[output];
            let in_node = target_inputs[input].clone();

            self.get_node_by_id(out_node).unwrap().outputs = in_node.clone();

            for input in in_node {
                target.get_node_by_id(input).unwrap().inputs = vec![out_node];
            }
        }

        target.build_producers();
        target.build_consumers();

        self.nodes.retain(|node| match &node.kind {
            GraphNodeKind::Output(name) if src_outputs.contains_key(name.as_str()) => false,
            _ => true,
        });
        self.nodes
            .extend(target.nodes.into_iter().filter(|node| match &node.kind {
                GraphNodeKind::Input(name) if target_inputs.contains_key(name.as_str()) => false,
                _ => true,
            }));
        self.producers.extend(target.producers);
        self.consumers.extend(target.consumers);
    }

    // self의 input, output들을 target의 input과 연결함
    pub fn merge(&mut self, mut target: Self) {
        if let Some(id) = self.max_node_id() {
            target.rebuild_node_id_base(id + 1);
        }

        let target_inputs: HashSet<_> = target
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) | GraphNodeKind::Output(name) => Some(name.as_str()),
                _ => None,
            })
            .collect();

        let targets: HashSet<_> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) | GraphNodeKind::Output(name) => Some(name.to_owned()),
                _ => None,
            })
            .filter(|name| target_inputs.contains(name.as_str()))
            .collect();

        let src_inputs: HashMap<String, GraphNodeId> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) if targets.contains(name.as_str()) => {
                    Some((name.to_owned(), node.id))
                }
                _ => None,
            })
            .collect();

        let src_outputs: HashMap<String, GraphNodeId> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) if targets.contains(name.as_str()) => {
                    Some((name.to_owned(), node.inputs[0]))
                }
                _ => None,
            })
            .collect();

        let target_inputs: HashMap<String, Vec<GraphNodeId>> = target
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) if targets.contains(name.as_str()) => {
                    Some((name.to_owned(), node.outputs.clone()))
                }
                _ => None,
            })
            .collect();

        for name in &targets {
            let tar_in = target_inputs[name].clone();

            if let Some(src_in) = src_inputs.get(name) {
                self.get_node_by_id(*src_in).unwrap().outputs.extend(tar_in);
            } else if let Some(src_out) = src_outputs.get(name) {
                self.get_node_by_id(*src_out)
                    .unwrap()
                    .outputs
                    .extend(tar_in.clone());

                for input in tar_in {
                    target.get_node_by_id(input).unwrap().inputs = vec![*src_out];
                }
            }
        }

        target.build_producers();
        target.build_consumers();

        self.nodes
            .extend(target.nodes.into_iter().filter(|node| match &node.kind {
                GraphNodeKind::Input(name) if target_inputs.contains_key(name.as_str()) => false,
                _ => true,
            }));
        self.producers.extend(target.producers);
        self.consumers.extend(target.consumers);
    }

    pub fn replace_input_name(&mut self, from: &str, to: String) -> bool {
        if let Some(node) = self
            .nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == from))
        {
            *node.kind.as_input_mut() = to;
            return true;
        }

        false
    }

    pub fn replace_output_name(&mut self, from: &str, to: String) -> bool {
        if let Some(node) = self
            .nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == from))
        {
            *node.kind.as_output_mut() = to;
            return true;
        }

        false
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

    pub fn get_node_by_input_name(&mut self, input_name: &str) -> Option<&mut GraphNode> {
        self.nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == input_name))
    }

    pub fn get_node_by_output_name(&mut self, input_name: &str) -> Option<&mut GraphNode> {
        self.nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == input_name))
    }

    pub fn get_node_by_id(&mut self, node_id: GraphNodeId) -> Option<&mut GraphNode> {
        // self.nodes.binary_search_by_key(&node_id, |node| node.id)
        self.nodes.iter_mut().find(|node| node.id == node_id)
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
