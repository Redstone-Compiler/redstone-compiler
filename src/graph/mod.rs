use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;

use itertools::Itertools;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::NodeRef;

use self::cluster::ClusteredGraph;
use self::module::builder::GraphModuleBuilder;
use self::module::GraphModule;
use crate::cluster::{Clustered, ClusteredType};
use crate::logic::Logic;
use crate::world::block::Block;

mod cluster;
pub mod graphviz;
pub mod logic;
pub mod module;
pub mod world;

pub type GraphNodeId = usize;

#[derive(Default, Debug, Clone)]
pub enum GraphNodeKind {
    #[default]
    None,
    Input(String),
    Block(Block),
    Logic(Logic),
    Output(String),
    Clustered(Clustered),
}

impl GraphNodeKind {
    pub fn name(&self) -> String {
        match self {
            GraphNodeKind::None => format!("None"),
            GraphNodeKind::Input(input) => format!("Input {input}"),
            GraphNodeKind::Block(block) => block.kind.name(),
            GraphNodeKind::Logic(logic) => logic.logic_type.name(),
            GraphNodeKind::Output(output) => format!("Output {output}"),
            GraphNodeKind::Clustered(clustered) => clustered.clustered_type.name(),
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

    pub fn is_input(&self) -> bool {
        matches!(self, GraphNodeKind::Input(_))
    }

    pub fn is_output(&self) -> bool {
        matches!(self, GraphNodeKind::Output(_))
    }

    pub fn is_logic(&self) -> bool {
        matches!(self, GraphNodeKind::Logic(_))
    }

    pub fn as_logic(&self) -> Option<&Logic> {
        match self {
            GraphNodeKind::Logic(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn as_block(&self) -> Option<&Block> {
        match self {
            GraphNodeKind::Block(inner) => Some(inner),
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

impl Display for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "N{} {:?} := {} {:?} {}",
            self.id,
            self.outputs,
            self.kind.name(),
            self.inputs,
            self.tag
        )
    }
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

    pub fn dominators(
        &self,
        target_root: GraphNodeId,
    ) -> petgraph::algo::dominators::Dominators<NodeIndex> {
        petgraph::algo::dominators::simple_fast(&self.to_petgraph(), NodeIndex::new(target_root))
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

    pub fn ids<T>(&self) -> T
    where
        T: FromIterator<GraphNodeId>,
    {
        self.nodes.iter().map(|node| node.id).collect()
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
            let node = other.nodes.iter_mut().find(|node| node.id == from).unwrap();

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

    pub fn find_and_remove_node_by_id(&mut self, node_id: GraphNodeId) -> Option<GraphNode> {
        // TODO: nodes is always sorted by id?
        self.nodes
            .binary_search_by_key(&node_id, |node| node.id)
            .map(|index| self.nodes.remove(index))
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

    // 노드 삭제하고 삭제한 노드의 Input Output끼리 연결함
    pub fn remove_and_reconnect_by_node_id_lazy(&mut self, node_id: GraphNodeId) {
        let Ok(index) = self.nodes.binary_search_by_key(&node_id, |node| node.id) else {
            return;
        };

        let node = self.nodes.remove(index);

        for input in &node.inputs {
            if let Some(input) = self.find_node_by_id_mut(*input) {
                input.outputs.extend(&node.outputs);
            }
        }

        for output in &node.outputs {
            if let Some(output) = self.find_node_by_id_mut(*output) {
                output.inputs.extend(&node.inputs);
            }
        }
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
        let mut input_map: HashMap<usize, HashSet<usize>> = HashMap::new();

        self.nodes.iter().for_each(|node| {
            node.outputs
                .iter()
                .filter(|&&id| self.find_node_by_id(id).is_some())
                .for_each(|&id| {
                    input_map.entry(id).or_default().insert(node.id);
                });
        });

        for node in self.nodes.iter_mut() {
            node.inputs = input_map
                .get(&node.id)
                .map(|ids| ids.clone().into_iter().collect_vec())
                .unwrap_or_default();
        }
    }

    // inputs이 반드시 determine 되어야함
    pub fn build_outputs(&mut self) {
        let mut output_map: HashMap<usize, HashSet<usize>> = HashMap::new();

        self.nodes.iter().for_each(|node| {
            node.inputs
                .iter()
                .filter(|&&id| self.find_node_by_id(id).is_some())
                .for_each(|&id| {
                    output_map.entry(id).or_default().insert(node.id);
                });
        });

        for node in self.nodes.iter_mut() {
            node.outputs = output_map
                .get(&node.id)
                .map(|ids| ids.clone().into_iter().collect())
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

    pub fn extract_subgraph_by_node_id(&self, node_id: GraphNodeId) -> SubGraphWithGraph {
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

        SubGraphWithGraph::from(self, nodes)
    }

    pub fn split_with_outputs(&self) -> Vec<SubGraphWithGraph> {
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

    pub fn remove_output(&mut self, output_name: &str) -> Option<GraphNodeId> {
        let Some((index, _)) = self.nodes.iter().find_position(
            |node| matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name),
        ) else {
            return None;
        };

        let id = self.nodes.remove(index).id;
        self.build_outputs();
        self.build_producers();
        self.build_consumers();

        Some(id)
    }

    pub fn verify(&self) -> eyre::Result<()> {
        if !self.nodes.windows(2).all(|w| w[0].id <= w[1].id) {
            eyre::bail!("Nodes must be sorted by id!");
        }

        let valid_ids: HashSet<GraphNodeId> = self.nodes.iter().map(|node| node.id).collect();

        for node in &self.nodes {
            if node.inputs.iter().any(|id| !valid_ids.contains(id)) {
                eyre::bail!("This is not valid graph! Invalid node id contains on inputs index!");
            }

            if node.outputs.iter().any(|id| !valid_ids.contains(id)) {
                eyre::bail!("This is not valid graph! Invalid node id contains on outputs index!");
            }
        }

        Ok(())
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
pub struct SubGraphWithGraph<'a> {
    pub graph: &'a Graph,
    pub nodes: Vec<GraphNodeId>,
    pub producers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
    pub consumers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
}

#[derive(Debug, Clone)]
pub struct SubGraph {
    pub nodes: Vec<GraphNodeId>,
    pub producers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
    pub consumers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
}

impl<'a> SubGraphWithGraph<'a> {
    pub fn from(graph: &'a Graph, nodes: Vec<GraphNodeId>) -> Self {
        let node_ids: HashSet<_> = nodes.iter().collect();
        let mut producers = HashMap::new();
        let mut consumers = HashMap::new();

        for id in &nodes {
            producers.insert(
                *id,
                graph.producers[id]
                    .iter()
                    .filter(|&id| node_ids.contains(id))
                    .copied()
                    .collect_vec(),
            );
            consumers.insert(
                *id,
                graph.consumers[id]
                    .iter()
                    .filter(|&id| node_ids.contains(id))
                    .copied()
                    .collect_vec(),
            );
        }

        Self {
            graph,
            nodes,
            producers,
            consumers,
        }
    }

    pub fn topological_order(&self) -> Vec<GraphNodeId> {
        let contains: HashSet<&usize> = self.nodes.iter().collect();
        let order = self.graph.topological_order();

        order
            .into_iter()
            .filter(|id| contains.contains(id))
            .collect_vec()
    }

    pub fn has_cycle(&self) -> bool {
        self.graph.has_cycle()
    }

    pub fn inputs(&self) -> Vec<GraphNodeId> {
        self.nodes
            .iter()
            .filter(|&id| self.producers[id].is_empty())
            .copied()
            .collect()
    }

    pub fn outputs(&self) -> Vec<GraphNodeId> {
        self.nodes
            .iter()
            .filter(|&id| self.consumers[id].is_empty())
            .copied()
            .collect()
    }

    pub fn to_subgraph(&self) -> SubGraph {
        SubGraph {
            nodes: self.nodes.clone(),
            producers: self.producers.clone(),
            consumers: self.consumers.clone(),
        }
    }
}

impl<'a> From<&SubGraphWithGraph<'a>> for Graph {
    fn from(value: &SubGraphWithGraph) -> Self {
        let node_ids: HashSet<_> = value.nodes.iter().collect();

        let mut nodes = value
            .graph
            .nodes
            .iter()
            .filter(|node| node_ids.contains(&node.id))
            .cloned()
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

impl<'a> From<Vec<SubGraphWithGraph<'a>>> for Graph {
    fn from(value: Vec<SubGraphWithGraph<'a>>) -> Self {
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

pub fn subgraphs_to_clustered_graph(graph: &Graph, subgraphs: &Vec<SubGraph>) -> ClusteredGraph {
    // GraphNodeId, Cluster Index
    let cluster_index: HashMap<GraphNodeId, usize> = subgraphs
        .iter()
        .enumerate()
        .map(|(index, sg)| sg.nodes.iter().map(|&id| (id, index)).collect_vec())
        .flatten()
        .collect();

    let mut nodes = subgraphs
        .iter()
        .enumerate()
        .map(|(index, sg)| {
            let clustered = Clustered {
                clustered_type: ClusteredType::Cluster(sg.nodes.clone()),
            };

            GraphNode {
                id: index,
                kind: GraphNodeKind::Clustered(clustered),
                ..Default::default()
            }
        })
        .collect_vec();

    let mut weighted_node = subgraphs
        .iter()
        .enumerate()
        .map(|(index, sg)| {
            let consumers = sg
                .nodes
                .iter()
                .map(|node| graph.consumers[&node.id()].clone())
                .flatten()
                .collect_vec();

            // cluster_index, weight
            let mut consumer_cluster_weights: HashMap<usize, usize> = HashMap::new();
            for consumer in consumers {
                if cluster_index[&consumer] != index {
                    *consumer_cluster_weights
                        .entry(cluster_index[&consumer])
                        .or_default() += 1;
                }
            }

            // weighed_node, consumer
            let weighted_nodes = consumer_cluster_weights
                .into_iter()
                .map(|(cluster_id, w)| {
                    let weighted = Clustered {
                        clustered_type: ClusteredType::Weighted(w as isize),
                    };

                    GraphNode {
                        kind: GraphNodeKind::Clustered(weighted),
                        outputs: vec![cluster_id],
                        ..Default::default()
                    }
                })
                .collect_vec();

            weighted_nodes
        })
        .collect_vec();

    // Put id lazy
    for (index, node) in weighted_node.iter_mut().flatten().enumerate() {
        node.id = nodes.len() + index + 1;
    }

    // Set output weighted node
    for (index, clustered_node) in nodes.iter_mut().enumerate() {
        clustered_node.outputs = weighted_node[index].iter().map(|node| node.id).collect();
    }

    let mut graph = Graph {
        nodes: vec![nodes, weighted_node.into_iter().flatten().collect_vec()].concat(),
        ..Default::default()
    };

    graph.build_inputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify().unwrap();

    ClusteredGraph { graph }
}
