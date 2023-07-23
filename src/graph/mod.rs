use std::collections::{HashMap, HashSet, VecDeque};

use itertools::Itertools;
use petgraph::stable_graph::NodeIndex;

use crate::{logic::Logic, world::block::Block};

use self::{builder::module::GraphModuleBuilder, module::GraphModule};

pub mod builder;
pub mod graphviz;
pub mod module;

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
    pub id: GraphNodeId,
    pub kind: GraphNodeKind,
    pub inputs: Vec<GraphNodeId>,
    pub outputs: Vec<GraphNodeId>,
    pub tag: String,
}

impl GraphNode {
    pub fn new(id: GraphNodeId, kind: GraphNodeKind) -> Self {
        Self {
            id,
            kind,
            ..Default::default()
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub producers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
    pub consumers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
}

impl Graph {
    pub fn to_petgraph(&self) -> petgraph::Graph<(), ()> {
        self.into()
    }

    pub fn topological_order(&self) -> Vec<GraphNodeId> {
        let nodes: HashSet<GraphNodeId> = self.nodes.iter().map(|node| node.id).collect();
        petgraph::algo::toposort(&self.to_petgraph(), None)
            .unwrap()
            .iter()
            .map(|index| index.index())
            .filter(|id| nodes.contains(id))
            .collect_vec()
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

            self.find_node_by_id_mut(to)
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
        self.build_inputs();
        self.build_producers();
        self.build_consumers();
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

            self.find_node_by_id_mut(out_node).unwrap().outputs = in_node.clone();

            for input in in_node {
                target.find_node_by_id_mut(input).unwrap().inputs = vec![out_node];
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
        self.build_inputs();
        self.build_producers();
        self.build_consumers();
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
                self.find_node_by_id_mut(*src_in)
                    .unwrap()
                    .outputs
                    .extend(tar_in);
            } else if let Some(src_out) = src_outputs.get(name) {
                self.find_node_by_id_mut(*src_out)
                    .unwrap()
                    .outputs
                    .extend(tar_in.clone());

                for input in tar_in {
                    target.find_node_by_id_mut(input).unwrap().inputs = vec![*src_out];
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
        self.build_inputs();
        self.build_producers();
        self.build_consumers();
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

    pub fn find_node_by_input_name(&mut self, input_name: &str) -> Option<&mut GraphNode> {
        self.nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == input_name))
    }

    pub fn find_node_by_output_name(&mut self, output_name: &str) -> Option<&mut GraphNode> {
        self.nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name))
    }

    pub fn find_node_by_id(&self, node_id: GraphNodeId) -> Option<&GraphNode> {
        // TODO: nodes is always sorted by id?
        self.nodes
            .binary_search_by_key(&node_id, |node| node.id)
            .map(|index| &self.nodes[index])
            .ok()
    }

    pub fn find_node_by_id_mut(&mut self, node_id: GraphNodeId) -> Option<&mut GraphNode> {
        // TODO: nodes is always sorted by id?
        self.nodes
            .binary_search_by_key(&node_id, |node| node.id)
            .map(|index| &mut self.nodes[index])
            .ok()
    }

    pub fn rebuild_node_id_base(&mut self, base_index: usize) {
        for node in &mut self.nodes {
            node.id += base_index;
            for index in itertools::chain!(&mut node.inputs, &mut node.outputs) {
                *index += base_index;
            }
        }

        self.build_producers();
        self.build_consumers();
    }

    pub fn rebuild_node_ids_deprecated(self) -> Self {
        // toposort가 기존에 없는 새로운 index를 창조함
        // ???
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
                ..Default::default()
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.id);

        let mut result = Self { nodes, ..self };
        result.build_producers();
        result.build_consumers();
        result
    }

    pub fn rebuild_node_ids(mut self) -> Self {
        self.nodes.sort_by_key(|node| match node.kind {
            GraphNodeKind::Input(_) => 0,
            GraphNodeKind::Output(_) => 1,
            _ => 2,
        });

        let indexs: HashMap<_, _> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, index))
            .collect();

        let nodes = self
            .nodes
            .into_iter()
            .map(|node| GraphNode {
                id: indexs[&node.id],
                inputs: node.inputs.iter().map(|index| indexs[index]).collect(),
                outputs: node.outputs.iter().map(|index| indexs[index]).collect(),
                kind: node.kind,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let mut result = Self { nodes, ..self };
        result.build_producers();
        result.build_consumers();
        result
    }

    pub fn remove_by_node_id_lazy(&mut self, node_id: GraphNodeId) {
        let Ok(index) = self.nodes.binary_search_by_key(&node_id, |node| node.id) else {
            return;
        };

        self.nodes.remove(index);
    }

    pub fn replace_node_id_lazy(&mut self, from: GraphNodeId, to: GraphNodeId) {
        self.nodes
            .iter_mut()
            .map(|node| itertools::chain!(&mut node.inputs, &mut node.outputs))
            .flatten()
            .filter(|node_id| **node_id == from)
            .for_each(|node_id| *node_id = to);
    }

    pub fn replace_input_node_id_lazy(&mut self, from: GraphNodeId, to: GraphNodeId) {
        self.nodes
            .iter_mut()
            .map(|node| &mut node.inputs)
            .flatten()
            .filter(|node_id| **node_id == from)
            .for_each(|node_id| *node_id = to);
    }

    pub fn replace_output_node_id_lazy(&mut self, from: GraphNodeId, to: GraphNodeId) {
        self.nodes
            .iter_mut()
            .map(|node| &mut node.outputs)
            .flatten()
            .filter(|node_id| **node_id == from)
            .for_each(|node_id| *node_id = to);
    }

    pub fn replace_target_input_node_ids(
        &mut self,
        target: GraphNodeId,
        from: GraphNodeId,
        to: Vec<GraphNodeId>,
    ) {
        let node = self.find_node_by_id_mut(target).unwrap();
        node.inputs.retain(|id| *id != from);
        node.inputs.extend(to);
    }

    pub fn replace_target_output_node_ids(
        &mut self,
        target: GraphNodeId,
        from: GraphNodeId,
        to: Vec<GraphNodeId>,
    ) {
        let node = self.find_node_by_id_mut(target).unwrap();
        node.outputs.retain(|id| *id != from);
        node.outputs.extend(to);
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

    pub fn extract_subgraph_by_node_id(&self, node_id: GraphNodeId) -> SubGraph {
        let mut nodes: HashSet<GraphNodeId> = HashSet::new();

        let mut queue: VecDeque<GraphNodeId> = VecDeque::new();
        queue.push_back(node_id);

        while let Some(node_id) = queue.pop_front() {
            if nodes.contains(&node_id) {
                continue;
            }

            nodes.insert(node_id);

            self.producers
                .get(&node_id)
                .iter()
                .for_each(|node_ids| queue.extend(node_ids.iter()));
        }

        let mut nodes = nodes.into_iter().collect_vec();
        nodes.sort();

        SubGraph { graph: self, nodes }
    }

    pub fn split_with_outputs(&self) -> Vec<SubGraph> {
        self.outputs()
            .iter()
            .map(|output| self.extract_subgraph_by_node_id(*output))
            .collect_vec()
    }

    pub fn critical_path(&self) -> Vec<GraphNodeId> {
        fn find_longest_path(graph: &petgraph::Graph<(), ()>) -> Option<Vec<NodeIndex>> {
            let topo_order = petgraph::algo::toposort(&graph, None).unwrap();

            let mut max_distances: Vec<Option<usize>> = vec![None; graph.node_count()];
            let mut max_path: Vec<Option<Vec<NodeIndex>>> = vec![None; graph.node_count()];

            for node in topo_order {
                let mut current_max_distance = None;
                let mut current_max_path = None;

                for pred in graph.neighbors_directed(node, petgraph::Direction::Incoming) {
                    let pred_distance = max_distances[pred.index()].unwrap_or(0);

                    if current_max_distance.is_none()
                        || pred_distance + 1 > current_max_distance.unwrap()
                    {
                        current_max_distance = Some(pred_distance + 1);
                        current_max_path =
                            Some(max_path[pred.index()].clone().unwrap_or_else(|| vec![pred]));
                    }
                }

                max_distances[node.index()] = current_max_distance;
                max_path[node.index()] = current_max_path.map(|mut path| {
                    path.push(node);
                    path
                });
            }

            let longest_path = max_path
                .iter()
                .max_by_key(|path| path.as_ref().map_or(0, |p| p.len()));

            longest_path.cloned().flatten()
        }

        let mut path = find_longest_path(&self.into())
            .unwrap()
            .iter()
            .map(|id| id.index())
            .collect_vec();
        path.sort();
        path
    }

    pub fn to_module(self, builder: &mut GraphModuleBuilder, name: &str) -> GraphModule {
        builder.to_graph_module(self, name)
    }

    pub fn remove_input(&mut self, input_name: &str) {
        let Some((index, _)) = self.nodes.iter().find_position(
            |node| matches!(&node.kind, GraphNodeKind::Output(name) if name == input_name),
        ) else {
            return;
        };

        self.nodes.remove(index);
        self.build_inputs();
        self.build_producers();
        self.build_consumers();
    }

    pub fn remove_output(&mut self, output_name: &str) {
        let Some((index, _)) = self.nodes.iter().find_position(
            |node| matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name),
        ) else {
            return;
        };

        self.nodes.remove(index);
        self.build_outputs();
        self.build_producers();
        self.build_consumers();
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
                        .collect_vec()
                })
                .flatten(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct SubGraph<'a> {
    pub graph: &'a Graph,
    pub nodes: Vec<GraphNodeId>,
}

impl<'a> SubGraph<'a> {
    pub fn from(graph: &'a Graph, nodes: Vec<GraphNodeId>) -> Self {
        Self { graph, nodes }
    }
}

impl<'a> From<&SubGraph<'a>> for Graph {
    fn from(value: &SubGraph) -> Self {
        let node_ids: HashSet<_> = value.nodes.iter().collect();

        let mut nodes = value
            .graph
            .nodes
            .iter()
            .filter(|node| node_ids.contains(&node.id))
            .map(|node| node.clone())
            .collect_vec();

        for node in &mut nodes {
            node.inputs.retain(|input| node_ids.contains(input));
            node.outputs.retain(|output| node_ids.contains(output));
        }

        let mut graph = Graph {
            nodes,
            ..Default::default()
        };
        graph.rebuild_node_id_base(0);

        graph
    }
}

impl<'a> From<Vec<SubGraph<'a>>> for Graph {
    fn from(value: Vec<SubGraph<'a>>) -> Self {
        if value.is_empty() {
            return Default::default();
        }

        let mut graph: Graph = (&value[0]).into();

        for subgraph in value.iter().skip(1) {
            graph.concat(subgraph.into());
        }

        graph
    }
}
