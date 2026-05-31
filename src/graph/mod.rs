use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;

use indexmap::IndexMap;
use itertools::Itertools;
use module::GraphModuleBuilder;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::NodeRef;

use self::cluster::ClusteredGraph;
use self::module::GraphModule;
use crate::cluster::{Clustered, ClusteredType};
use crate::logic::Logic;
use crate::sequential::SequentialPrimitive;
use crate::world::block::Block;

pub mod analysis;
mod cluster;
pub mod graphviz;
pub mod logic;
pub mod module;
pub mod world;

pub type GraphNodeId = usize;

#[derive(Default, Debug, Clone, PartialEq)]
pub enum GraphNodeKind {
    #[default]
    None,
    Input(String),
    Block(Block),
    Logic(Logic),
    Sequential(SequentialPrimitive),
    Output(String),
    Clustered(Clustered),
}

impl GraphNodeKind {
    pub fn name(&self) -> String {
        match self {
            GraphNodeKind::None => "None".to_string(),
            GraphNodeKind::Input(input) => format!("Input {input}"),
            GraphNodeKind::Block(block) => block.kind.name(),
            GraphNodeKind::Logic(logic) => logic.logic_type.name(),
            GraphNodeKind::Sequential(sequential) => sequential.name(),
            GraphNodeKind::Output(output) => format!("Output {output}"),
            GraphNodeKind::Clustered(clustered) => clustered.clustered_type.name(),
        }
    }

    pub fn as_input(&self) -> &String {
        match self {
            GraphNodeKind::Input(name) => name,
            _ => unreachable!(),
        }
    }

    pub fn as_output(&self) -> &String {
        match self {
            GraphNodeKind::Output(name) => name,
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

    pub fn is_sequential(&self) -> bool {
        matches!(self, GraphNodeKind::Sequential(_))
    }

    pub fn as_logic(&self) -> Option<&Logic> {
        match self {
            GraphNodeKind::Logic(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn as_sequential(&self) -> Option<&SequentialPrimitive> {
        match self {
            GraphNodeKind::Sequential(inner) => Some(inner),
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
    pub kind: GraphNodeKind,
    pub inputs: Vec<GraphNodeId>,
    pub outputs: Vec<GraphNodeId>,
    pub tag: String,
}

impl Display for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} := {} {:?} {}",
            self.outputs,
            self.kind.name(),
            self.inputs,
            self.tag
        )
    }
}

impl GraphNode {
    pub fn new(kind: GraphNodeKind) -> Self {
        Self {
            kind,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy)]
pub struct GraphNodeRef<'a> {
    pub id: GraphNodeId,
    node: &'a GraphNode,
}

impl GraphNodeRef<'_> {
    pub fn new(id: GraphNodeId, node: &GraphNode) -> GraphNodeRef<'_> {
        GraphNodeRef { id, node }
    }

    pub fn clone_node(&self) -> GraphNode {
        self.node.clone()
    }
}

impl std::ops::Deref for GraphNodeRef<'_> {
    type Target = GraphNode;

    fn deref(&self) -> &Self::Target {
        self.node
    }
}

impl Display for GraphNodeRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.node, f)
    }
}

pub struct GraphNodeMut<'a> {
    pub id: GraphNodeId,
    node: &'a mut GraphNode,
}

impl std::ops::Deref for GraphNodeMut<'_> {
    type Target = GraphNode;

    fn deref(&self) -> &Self::Target {
        self.node
    }
}

impl std::ops::DerefMut for GraphNodeMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.node
    }
}

#[derive(Debug, Clone)]
pub struct GraphNodeEntry {
    pub id: GraphNodeId,
    pub node: GraphNode,
}

impl std::ops::Deref for GraphNodeEntry {
    type Target = GraphNode;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl std::ops::DerefMut for GraphNodeEntry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

#[derive(Default, Debug, Clone)]
pub struct GraphNodes {
    nodes: IndexMap<GraphNodeId, GraphNode>,
    next_node_id: GraphNodeId,
}

impl GraphNodes {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get(&self, id: GraphNodeId) -> Option<GraphNodeRef<'_>> {
        self.nodes.get(&id).map(|node| GraphNodeRef { id, node })
    }

    pub fn get_mut(&mut self, id: GraphNodeId) -> Option<GraphNodeMut<'_>> {
        self.nodes
            .get_mut(&id)
            .map(|node| GraphNodeMut { id, node })
    }

    pub fn iter(&self) -> impl Iterator<Item = GraphNodeRef<'_>> {
        self.nodes
            .iter()
            .map(|(&id, node)| GraphNodeRef { id, node })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = GraphNodeMut<'_>> {
        self.nodes
            .iter_mut()
            .map(|(&id, node)| GraphNodeMut { id, node })
    }

    pub fn retain(&mut self, f: impl FnMut(&GraphNode) -> bool) {
        let mut f = f;
        self.nodes.retain(|_, node| f(node));
    }

    pub fn remove(&mut self, index: usize) -> GraphNode {
        self.remove_entry(index).node
    }

    pub fn remove_entry(&mut self, index: usize) -> GraphNodeEntry {
        self.nodes
            .shift_remove_index(index)
            .map(|(id, node)| GraphNodeEntry { id, node })
            .expect("node index must exist")
    }

    fn insert_with_id(&mut self, id: GraphNodeId, node: GraphNode) {
        self.next_node_id = self.next_node_id.max(id + 1);
        self.nodes.insert(id, node);
    }

    fn next_node_id(&self) -> GraphNodeId {
        self.next_node_id
    }

    fn allocate_node_id(&mut self) -> GraphNodeId {
        let id = self.next_node_id();
        self.next_node_id = id + 1;
        id
    }

    fn add(&mut self, node: GraphNode) -> GraphNodeId {
        let id = self.allocate_node_id();
        self.insert_with_id(id, node);
        id
    }
}

impl From<Vec<(GraphNodeId, GraphNode)>> for GraphNodes {
    fn from(mut nodes: Vec<(GraphNodeId, GraphNode)>) -> Self {
        let next_node_id = nodes.iter().map(|(id, _)| *id).max().map_or(0, |id| id + 1);
        nodes.sort_by_key(|(id, _)| *id);
        Self {
            nodes: nodes.into_iter().collect(),
            next_node_id,
        }
    }
}

impl IntoIterator for GraphNodes {
    type Item = GraphNodeEntry;
    type IntoIter = std::vec::IntoIter<GraphNodeEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes
            .into_iter()
            .map(|(id, node)| GraphNodeEntry { id, node })
            .collect_vec()
            .into_iter()
    }
}

impl<'a> IntoIterator for &'a GraphNodes {
    type Item = GraphNodeRef<'a>;
    type IntoIter = std::vec::IntoIter<GraphNodeRef<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect_vec().into_iter()
    }
}

impl<'a> IntoIterator for &'a mut GraphNodes {
    type Item = GraphNodeMut<'a>;
    type IntoIter = std::vec::IntoIter<GraphNodeMut<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut().collect_vec().into_iter()
    }
}

#[derive(Default, Debug, Clone)]
pub struct Graph {
    pub nodes: GraphNodes,
    pub producers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
    pub consumers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
}

impl Graph {
    pub fn from_nodes_with_ids(nodes: Vec<(GraphNodeId, GraphNode)>) -> Self {
        Self {
            nodes: nodes.into(),
            ..Default::default()
        }
    }

    pub fn from_nodes(nodes: Vec<GraphNode>) -> Self {
        let nodes = nodes.into_iter().enumerate().collect_vec();
        Self::from_nodes_with_ids(nodes)
    }

    fn to_petgraph_with_node_ids(&self) -> petgraph::Graph<GraphNodeId, ()> {
        let mut graph = petgraph::Graph::<GraphNodeId, ()>::new();
        let mut id_to_index = HashMap::new();

        for node in &self.nodes {
            let index = graph.add_node(node.id);
            id_to_index.insert(node.id, index);
        }

        for node in &self.nodes {
            let Some(source) = id_to_index.get(&node.id).copied() else {
                continue;
            };
            for output in &node.outputs {
                if let Some(target) = id_to_index.get(output).copied() {
                    graph.add_edge(source, target, ());
                }
            }
        }

        graph
    }

    pub fn to_petgraph_only_edges(&self) -> petgraph::Graph<GraphNodeId, ()> {
        self.to_petgraph_with_node_ids()
    }

    pub fn to_petgraph(&self) -> petgraph::Graph<GraphNodeKind, ()> {
        let mut graph = petgraph::Graph::new();
        let mut node_to_id = HashMap::new();

        for node in &self.nodes {
            let nx = graph.add_node(node.kind.clone());
            node_to_id.insert(node.id, nx);
        }

        for (id, producers) in &self.producers {
            for producer in producers {
                graph.add_edge(node_to_id[producer], node_to_id[id], ());
            }
        }

        graph
    }

    pub fn topological_order(&self) -> Vec<GraphNodeId> {
        let graph = self.to_petgraph_only_edges();
        petgraph::algo::toposort(&graph, None)
            .unwrap()
            .iter()
            .map(|index| graph[*index])
            .collect_vec()
    }

    pub fn strongly_connected_components(&self) -> Vec<Vec<GraphNodeId>> {
        let node_ids: HashSet<GraphNodeId> = self.nodes.iter().map(|node| node.id).collect();
        let graph = self.to_petgraph_only_edges();
        let mut seen = HashSet::new();
        let mut components = petgraph::algo::kosaraju_scc(&graph)
            .into_iter()
            .filter_map(|component| {
                let mut component = component
                    .into_iter()
                    .map(|index| graph[index])
                    .collect_vec();
                component.sort();
                if component.is_empty() {
                    return None;
                }
                seen.extend(component.iter().copied());
                Some(component)
            })
            .collect_vec();
        components.extend(
            node_ids
                .difference(&seen)
                .copied()
                .map(|node_id| vec![node_id]),
        );
        for component in &mut components {
            component.sort();
        }
        components.sort();
        components
    }

    pub fn external_edges(&self, nodes: &HashSet<GraphNodeId>, incoming: bool) -> Vec<GraphNodeId> {
        let mut edges = nodes
            .iter()
            .flat_map(|node_id| {
                self.find_node_by_id(*node_id)
                    .map_or_else(Vec::new, |node| {
                        if incoming {
                            node.inputs.clone()
                        } else {
                            node.outputs.clone()
                        }
                    })
            })
            .filter(|node_id| !nodes.contains(node_id))
            .collect::<Vec<_>>();
        edges.sort();
        edges.dedup();
        edges
    }

    pub fn dominators(
        &self,
        target_root: GraphNodeId,
    ) -> petgraph::algo::dominators::Dominators<NodeIndex> {
        let graph = self.to_petgraph_only_edges();
        let target_root = graph
            .node_indices()
            .find(|index| graph[*index] == target_root)
            .expect("target root must exist in graph");
        petgraph::algo::dominators::simple_fast(&graph, target_root)
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

    pub fn concat(&mut self, other: Self) {
        self.append_graph_with_replacements(other, &HashMap::new());
        self.build_producers();
        self.build_consumers();
    }

    // src와 other의 input이 같은 것 끼리 연결하여 merge함
    pub fn merge_by_input(&mut self, other: Self) {
        let src_inputs: HashMap<String, GraphNodeId> = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) => Some((name.to_owned(), node.id)),
                _ => None,
            })
            .collect();

        let shared_inputs: Vec<(GraphNodeId, GraphNodeId, Vec<GraphNodeId>)> = other
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) if src_inputs.contains_key(name) => Some((
                    node.id,
                    *src_inputs.get(name).unwrap(),
                    node.outputs.clone(),
                )),
                _ => None,
            })
            .collect();
        let old_to_existing_ids = shared_inputs
            .iter()
            .map(|(from, to, _)| (*from, *to))
            .collect::<HashMap<_, _>>();

        let old_to_current_ids = self.append_graph_with_replacements(other, &old_to_existing_ids);

        for (_, to, outputs) in shared_inputs {
            let outputs = outputs
                .iter()
                .map(|id| old_to_current_ids.get(id).copied().unwrap_or(*id))
                .collect_vec();
            self.find_node_by_id_mut(to)
                .unwrap()
                .outputs
                .extend(outputs);
        }

        self.build_inputs();
        self.build_producers();
        self.build_consumers();
    }

    // src의 output을 target의 input과 연결하여 merge함
    pub fn merge_by_outin(&mut self, target: Self, out_in: Vec<(&str, &str)>) {
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

        let target_inputs: HashMap<String, (GraphNodeId, Vec<GraphNodeId>)> = target
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) if inputs.contains(name.as_str()) => {
                    Some((name.to_owned(), (node.id, node.outputs.clone())))
                }
                _ => None,
            })
            .collect();
        let old_to_existing_ids = out_in
            .iter()
            .map(|(output, input)| {
                let (target_input_id, _) = &target_inputs[*input];
                (*target_input_id, src_outputs[*output])
            })
            .collect::<HashMap<_, _>>();

        self.nodes.retain(|node|
            !matches!(&node.kind, GraphNodeKind::Output(name) if src_outputs.contains_key(name.as_str())));

        let old_to_current_ids = self.append_graph_with_replacements(target, &old_to_existing_ids);

        for (output, input) in out_in {
            let out_node = src_outputs[output];
            let (_, target_input_outputs) = &target_inputs[input];
            let in_node = target_input_outputs
                .iter()
                .map(|id| old_to_current_ids.get(id).copied().unwrap_or(*id))
                .collect_vec();

            self.find_node_by_id_mut(out_node).unwrap().outputs = in_node;
        }

        self.build_inputs();
        self.build_producers();
        self.build_consumers();
    }

    // self의 input, output들을 target의 input과 연결함
    pub fn merge(&mut self, target: Self) {
        let target_inputs: HashSet<_> = target
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) | GraphNodeKind::Output(name) => Some(name.to_owned()),
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
            .filter(|name| target_inputs.contains(name))
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

        let target_inputs: HashMap<String, (GraphNodeId, Vec<GraphNodeId>)> = target
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) if targets.contains(name.as_str()) => {
                    Some((name.to_owned(), (node.id, node.outputs.clone())))
                }
                _ => None,
            })
            .collect();

        let mut old_to_existing_ids = HashMap::new();
        for name in &targets {
            let (target_input_id, _) = &target_inputs[name];
            if let Some(src_in) = src_inputs.get(name) {
                old_to_existing_ids.insert(*target_input_id, *src_in);
            } else if let Some(src_out) = src_outputs.get(name) {
                old_to_existing_ids.insert(*target_input_id, *src_out);
            }
        }

        let old_to_current_ids = self.append_graph_with_replacements(target, &old_to_existing_ids);

        for name in &targets {
            let (_, target_input_outputs) = &target_inputs[name];
            let tar_in = target_input_outputs
                .iter()
                .map(|id| old_to_current_ids.get(id).copied().unwrap_or(*id))
                .collect_vec();

            if let Some(src_in) = src_inputs.get(name) {
                self.find_node_by_id_mut(*src_in)
                    .unwrap()
                    .outputs
                    .extend(tar_in);
            } else if let Some(src_out) = src_outputs.get(name) {
                self.find_node_by_id_mut(*src_out)
                    .unwrap()
                    .outputs
                    .extend(tar_in);
            }
        }

        self.build_inputs();
        self.build_producers();
        self.build_consumers();
    }

    pub fn replace_input_name(&mut self, from: &str, to: String) -> bool {
        if let Some(mut node) = self
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
        if let Some(mut node) = self
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
        petgraph::algo::is_cyclic_directed(&self.to_petgraph_only_edges())
    }

    pub fn add_node(&mut self, node: GraphNode) -> GraphNodeId {
        self.nodes.add(node)
    }

    /// Appends `imported_graph` into this graph using fresh ids for every imported node.
    /// Old ids in `old_to_existing_ids` are not inserted or overwritten; references to
    /// those imported nodes are resolved to the existing node ids in this graph.
    pub(crate) fn append_graph_with_replacements(
        &mut self,
        imported_graph: Self,
        old_to_existing_ids: &HashMap<GraphNodeId, GraphNodeId>,
    ) -> HashMap<GraphNodeId, GraphNodeId> {
        let nodes = imported_graph.nodes.into_iter().collect_vec();
        let mut old_to_current_ids = old_to_existing_ids.clone();
        let mut new_ids = Vec::new();

        for entry in nodes {
            if old_to_existing_ids.contains_key(&entry.id) {
                continue;
            }
            let id = self.add_node(entry.node);
            old_to_current_ids.insert(entry.id, id);
            new_ids.push(id);
        }

        for id in new_ids {
            if let Some(mut node) = self.find_node_by_id_mut(id) {
                for input in &mut node.inputs {
                    *input = old_to_current_ids.get(input).copied().unwrap_or(*input);
                }
                for output in &mut node.outputs {
                    *output = old_to_current_ids.get(output).copied().unwrap_or(*output);
                }
            }
        }

        old_to_current_ids
    }

    pub fn find_node_by_input_name(&mut self, input_name: &str) -> Option<GraphNodeMut<'_>> {
        self.nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == input_name))
    }

    pub fn find_node_by_output_name(&mut self, output_name: &str) -> Option<GraphNodeMut<'_>> {
        self.nodes
            .iter_mut()
            .find(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name))
    }

    pub fn find_node_by_id(&self, node_id: GraphNodeId) -> Option<GraphNodeRef<'_>> {
        self.nodes.get(node_id)
    }

    pub fn find_node_by_id_mut(&mut self, node_id: GraphNodeId) -> Option<GraphNodeMut<'_>> {
        self.nodes.get_mut(node_id)
    }

    pub fn find_and_remove_node_by_id(&mut self, node_id: GraphNodeId) -> Option<GraphNode> {
        let index = self.nodes.iter().position(|node| node.id == node_id)?;
        Some(self.nodes.remove(index))
    }

    pub fn remove_by_node_id_lazy(&mut self, node_id: GraphNodeId) {
        let Some(index) = self.nodes.iter().position(|node| node.id == node_id) else {
            return;
        };

        self.nodes.remove(index);
    }

    pub fn replace_nodes_with(
        &mut self,
        removed_nodes: &HashSet<GraphNodeId>,
        kind: GraphNodeKind,
        inputs: Vec<GraphNodeId>,
        outputs: Vec<GraphNodeId>,
        tag: String,
    ) -> GraphNodeId {
        let replacement_id = self.nodes.allocate_node_id();

        for input in &inputs {
            if let Some(mut node) = self.find_node_by_id_mut(*input) {
                node.outputs.retain(|id| !removed_nodes.contains(id));
                if !node.outputs.contains(&replacement_id) {
                    node.outputs.push(replacement_id);
                }
            }
        }
        for output in &outputs {
            if let Some(mut node) = self.find_node_by_id_mut(*output) {
                node.inputs.retain(|id| !removed_nodes.contains(id));
                if !node.inputs.contains(&replacement_id) {
                    node.inputs.push(replacement_id);
                }
            }
        }
        for node_id in removed_nodes {
            self.remove_by_node_id_lazy(*node_id);
        }

        self.nodes.insert_with_id(
            replacement_id,
            GraphNode {
                kind,
                inputs,
                outputs,
                tag,
            },
        );
        self.build_inputs();
        self.build_outputs();
        self.build_producers();
        self.build_consumers();

        replacement_id
    }

    // 노드 삭제하고 삭제한 노드의 Input Output끼리 연결함
    pub fn remove_and_reconnect_by_node_id_lazy(&mut self, node_id: GraphNodeId) {
        let Some(index) = self.nodes.iter().position(|node| node.id == node_id) else {
            return;
        };

        let node = self.nodes.remove(index);

        for input in &node.inputs {
            if let Some(mut input) = self.find_node_by_id_mut(*input) {
                input.outputs.extend(&node.outputs);
            }
        }

        for output in &node.outputs {
            if let Some(mut output) = self.find_node_by_id_mut(*output) {
                output.inputs.extend(&node.inputs);
            }
        }
    }

    pub fn replace_node_id_lazy(&mut self, from: GraphNodeId, to: GraphNodeId) {
        for mut node in self.nodes.iter_mut() {
            for node_id in &mut node.inputs {
                if *node_id == from {
                    *node_id = to;
                }
            }
            for node_id in &mut node.outputs {
                if *node_id == from {
                    *node_id = to;
                }
            }
        }
    }

    pub fn replace_input_node_id_lazy(&mut self, from: GraphNodeId, to: GraphNodeId) {
        for mut node in self.nodes.iter_mut() {
            for node_id in &mut node.inputs {
                if *node_id == from {
                    *node_id = to;
                }
            }
        }
    }

    pub fn replace_output_node_id_lazy(&mut self, from: GraphNodeId, to: GraphNodeId) {
        for mut node in self.nodes.iter_mut() {
            for node_id in &mut node.outputs {
                if *node_id == from {
                    *node_id = to;
                }
            }
        }
    }

    pub fn replace_target_input_node_ids(
        &mut self,
        target: GraphNodeId,
        from: GraphNodeId,
        to: Vec<GraphNodeId>,
    ) {
        let mut node = self.find_node_by_id_mut(target).unwrap();
        node.inputs.retain(|id| *id != from);
        node.inputs.extend(to);
    }

    pub fn replace_target_output_node_ids(
        &mut self,
        target: GraphNodeId,
        from: GraphNodeId,
        to: Vec<GraphNodeId>,
    ) {
        let mut node = self.find_node_by_id_mut(target).unwrap();
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

        for mut node in self.nodes.iter_mut() {
            node.inputs = input_map
                .get(&node.id)
                .map(|ids| ids.iter().copied().sorted().collect_vec())
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

        for mut node in self.nodes.iter_mut() {
            node.outputs = output_map
                .get(&node.id)
                .map(|ids| ids.iter().copied().sorted().collect())
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

    pub fn extract_graph_by_node_ids(&self, node_ids: &HashSet<GraphNodeId>) -> Self {
        let mut nodes = self
            .nodes
            .iter()
            .filter(|node| node_ids.contains(&node.id))
            .map(|node| (node.id, node.node.clone()))
            .collect_vec();

        for (_, node) in &mut nodes {
            node.inputs.retain(|input| node_ids.contains(input));
            node.outputs.retain(|output| node_ids.contains(output));
        }
        let mut graph = Self::from_nodes_with_ids(nodes);
        graph.build_inputs();
        graph.build_outputs();
        graph.build_producers();
        graph.build_consumers();
        graph
    }

    pub fn split_with_outputs(&self) -> Vec<SubGraphWithGraph> {
        self.outputs()
            .iter()
            .map(|output| self.extract_subgraph_by_node_id(*output))
            .collect_vec()
    }

    pub fn critical_path(&self) -> Vec<GraphNodeId> {
        fn find_longest_path(graph: &petgraph::Graph<GraphNodeId, ()>) -> Option<Vec<NodeIndex>> {
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

        let graph = self.to_petgraph_only_edges();
        let mut path = find_longest_path(&graph)
            .unwrap()
            .iter()
            .map(|id| graph[*id])
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
        let (index, _) = self.nodes.iter().find_position(
            |node| matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name),
        )?;

        let id = self.nodes.remove_entry(index).id;
        self.build_outputs();
        self.build_producers();
        self.build_consumers();

        Some(id)
    }

    pub fn verify(&self) -> eyre::Result<()> {
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

impl From<&SubGraphWithGraph<'_>> for Graph {
    fn from(value: &SubGraphWithGraph) -> Self {
        let node_ids: HashSet<_> = value.nodes.iter().collect();

        let mut nodes = value
            .graph
            .nodes
            .iter()
            .filter(|node| node_ids.contains(&node.id))
            .map(|node| (node.id, node.node.clone()))
            .collect_vec();

        for (_, node) in &mut nodes {
            node.inputs.retain(|input| node_ids.contains(input));
            node.outputs.retain(|output| node_ids.contains(output));
        }

        let mut graph = Graph::from_nodes_with_ids(nodes);
        graph.build_producers();
        graph.build_consumers();

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

pub fn subgraphs_to_clustered_graph(
    source_graph: &Graph,
    subgraphs: &[SubGraph],
) -> ClusteredGraph {
    // GraphNodeId, Cluster Index
    let cluster_index: HashMap<GraphNodeId, usize> = subgraphs
        .iter()
        .enumerate()
        .flat_map(|(index, sg)| sg.nodes.iter().map(|&id| (id, index)).collect_vec())
        .collect();

    let nodes = subgraphs
        .iter()
        .enumerate()
        .map(|(index, sg)| {
            let clustered = Clustered {
                clustered_type: ClusteredType::Cluster(sg.nodes.clone()),
            };

            (
                index,
                GraphNode {
                    kind: GraphNodeKind::Clustered(clustered),
                    ..Default::default()
                },
            )
        })
        .collect_vec();
    let mut clustered_graph = Graph::from_nodes_with_ids(nodes);

    let weighted_node = subgraphs
        .iter()
        .enumerate()
        .map(|(index, sg)| {
            let consumers = sg
                .nodes
                .iter()
                .flat_map(|node| source_graph.consumers[&node.id()].clone())
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
            consumer_cluster_weights
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
                .collect_vec()
        })
        .collect_vec();

    let mut weighted_outputs = Vec::new();
    for nodes in weighted_node {
        let mut outputs = Vec::new();
        for node in nodes {
            let id = clustered_graph.add_node(node);
            outputs.push(id);
        }
        weighted_outputs.push(outputs);
    }

    // Set output weighted node
    for (index, mut clustered_node) in clustered_graph
        .nodes
        .iter_mut()
        .take(weighted_outputs.len())
        .enumerate()
    {
        clustered_node.outputs = weighted_outputs[index].clone();
    }

    clustered_graph.build_inputs();
    clustered_graph.build_producers();
    clustered_graph.build_consumers();
    clustered_graph.verify().unwrap();

    ClusteredGraph {
        graph: clustered_graph,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequential::{SequentialPrimitive, SequentialType};

    #[test]
    fn build_inputs_and_outputs_are_sorted() {
        let mut graph = Graph::from_nodes(vec![
            GraphNode {
                outputs: vec![3],
                ..Default::default()
            },
            GraphNode {
                outputs: vec![3],
                ..Default::default()
            },
            GraphNode {
                outputs: vec![3],
                ..Default::default()
            },
            GraphNode {
                inputs: vec![2, 0, 1],
                ..Default::default()
            },
        ]);

        graph.build_inputs();
        graph.build_outputs();

        assert_eq!(graph.find_node_by_id(3).unwrap().inputs, vec![0, 1, 2]);
        assert_eq!(graph.find_node_by_id(0).unwrap().outputs, vec![3]);
        assert_eq!(graph.find_node_by_id(1).unwrap().outputs, vec![3]);
        assert_eq!(graph.find_node_by_id(2).unwrap().outputs, vec![3]);
    }

    #[test]
    fn extract_graph_by_node_ids_keeps_only_internal_edges() {
        let mut graph = Graph::from_nodes(vec![
            GraphNode {
                outputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                outputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                inputs: vec![0, 1],
                outputs: vec![3, 4],
                ..Default::default()
            },
            GraphNode {
                inputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                inputs: vec![2],
                ..Default::default()
            },
        ]);
        graph.build_producers();
        graph.build_consumers();

        let selected = HashSet::from([1, 2, 3]);
        let subgraph = graph.extract_graph_by_node_ids(&selected);

        assert_eq!(
            subgraph.nodes.iter().map(|node| node.id).collect_vec(),
            vec![1, 2, 3]
        );
        assert_eq!(subgraph.find_node_by_id(1).unwrap().outputs, vec![2]);
        assert_eq!(subgraph.find_node_by_id(2).unwrap().inputs, vec![1]);
        assert_eq!(subgraph.find_node_by_id(2).unwrap().outputs, vec![3]);
        assert_eq!(subgraph.find_node_by_id(3).unwrap().inputs, vec![2]);
        subgraph.verify().unwrap();
    }

    #[test]
    fn strongly_connected_components_include_feedback_loop() {
        let primitive = SequentialPrimitive::rs_latch();
        let components = primitive.inner_graph.strongly_connected_components();

        assert!(components
            .iter()
            .any(|component| component == &vec![2, 3, 4, 5]));
        assert!(components.iter().any(|component| component == &vec![0]));
        assert!(components.iter().any(|component| component == &vec![7]));
    }

    #[test]
    fn sequential_node_kind_is_named_and_verified() -> eyre::Result<()> {
        let mut graph = Graph::from_nodes(vec![
            GraphNode {
                kind: GraphNodeKind::Input("s".to_owned()),
                outputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                kind: GraphNodeKind::Input("r".to_owned()),
                outputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
                    SequentialType::RsLatch,
                    vec!["s".to_owned(), "r".to_owned()],
                    vec!["q".to_owned()],
                )),
                inputs: vec![0, 1],
                outputs: vec![3],
                ..Default::default()
            },
            GraphNode {
                kind: GraphNodeKind::Output("q".to_owned()),
                inputs: vec![2],
                ..Default::default()
            },
        ]);

        graph.build_inputs();
        graph.build_outputs();
        graph.verify()?;

        let latch = graph.find_node_by_id(2).unwrap();
        assert!(latch.kind.is_sequential());
        assert_eq!(latch.kind.name(), "RsLatch");
        assert_eq!(
            latch.kind.as_sequential().unwrap().output_ports,
            vec!["q".to_owned()]
        );

        Ok(())
    }

    #[test]
    fn sparse_node_ids_are_keyed_and_append_only() {
        let mut graph = Graph::from_nodes_with_ids(vec![
            (
                30,
                GraphNode {
                    inputs: vec![20],
                    ..Default::default()
                },
            ),
            (
                10,
                GraphNode {
                    outputs: vec![20],
                    ..Default::default()
                },
            ),
            (
                20,
                GraphNode {
                    inputs: vec![10],
                    outputs: vec![30],
                    ..Default::default()
                },
            ),
        ]);

        assert_eq!(
            graph.nodes.iter().map(|node| node.id).collect_vec(),
            vec![10, 20, 30]
        );
        assert_eq!(graph.topological_order(), vec![10, 20, 30]);
        assert_eq!(
            graph.strongly_connected_components(),
            vec![vec![10], vec![20], vec![30]]
        );
        assert_eq!(graph.critical_path(), vec![10, 20, 30]);
        let petgraph = graph.to_petgraph_only_edges();
        assert_eq!(petgraph.node_count(), 3);
        assert_eq!(petgraph.edge_count(), 2);

        graph.remove_by_node_id_lazy(30);
        let id = graph.add_node(GraphNode::default());

        assert_eq!(id, 31);
        assert!(graph.find_node_by_id(31).is_some());
    }

    #[test]
    fn concat_allocates_fresh_ids_and_remaps_imported_edges() {
        let mut graph = Graph::from_nodes_with_ids(vec![(
            10,
            GraphNode {
                kind: GraphNodeKind::Input("a".to_owned()),
                ..Default::default()
            },
        )]);
        let other = Graph::from_nodes_with_ids(vec![
            (
                100,
                GraphNode {
                    outputs: vec![101],
                    ..Default::default()
                },
            ),
            (
                101,
                GraphNode {
                    inputs: vec![100],
                    ..Default::default()
                },
            ),
        ]);

        graph.concat(other);

        assert_eq!(
            graph.nodes.iter().map(|node| node.id).collect_vec(),
            vec![10, 11, 12]
        );
        assert_eq!(graph.find_node_by_id(11).unwrap().outputs, vec![12]);
        assert_eq!(graph.find_node_by_id(12).unwrap().inputs, vec![11]);
    }

    #[test]
    fn merge_by_input_allocates_fresh_ids_and_reuses_shared_input() {
        let mut graph = Graph::from_nodes_with_ids(vec![(
            10,
            GraphNode {
                kind: GraphNodeKind::Input("a".to_owned()),
                ..Default::default()
            },
        )]);
        let other = Graph::from_nodes_with_ids(vec![
            (
                100,
                GraphNode {
                    kind: GraphNodeKind::Input("a".to_owned()),
                    outputs: vec![101],
                    ..Default::default()
                },
            ),
            (
                101,
                GraphNode {
                    inputs: vec![100],
                    ..Default::default()
                },
            ),
        ]);

        graph.merge_by_input(other);

        assert_eq!(
            graph.nodes.iter().map(|node| node.id).collect_vec(),
            vec![10, 11]
        );
        assert_eq!(graph.find_node_by_id(10).unwrap().outputs, vec![11]);
        assert_eq!(graph.find_node_by_id(11).unwrap().inputs, vec![10]);
    }

    #[test]
    fn merge_by_outin_preserves_imported_output_with_same_name_as_removed_output() {
        let mut graph = Graph::from_nodes_with_ids(vec![
            (
                10,
                GraphNode {
                    outputs: vec![11],
                    ..Default::default()
                },
            ),
            (
                11,
                GraphNode {
                    inputs: vec![10],
                    outputs: vec![12],
                    ..Default::default()
                },
            ),
            (
                12,
                GraphNode {
                    kind: GraphNodeKind::Output("z".to_owned()),
                    inputs: vec![11],
                    ..Default::default()
                },
            ),
        ]);
        let target = Graph::from_nodes_with_ids(vec![
            (
                100,
                GraphNode {
                    kind: GraphNodeKind::Input("i".to_owned()),
                    outputs: vec![101],
                    ..Default::default()
                },
            ),
            (
                101,
                GraphNode {
                    inputs: vec![100],
                    outputs: vec![102],
                    ..Default::default()
                },
            ),
            (
                102,
                GraphNode {
                    kind: GraphNodeKind::Output("z".to_owned()),
                    inputs: vec![101],
                    ..Default::default()
                },
            ),
        ]);

        graph.merge_by_outin(target, vec![("z", "i")]);

        let outputs = graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == "z"))
            .map(|node| node.id)
            .collect_vec();
        assert_eq!(outputs, vec![14]);
        assert_eq!(graph.find_node_by_id(11).unwrap().outputs, vec![13]);
        assert_eq!(graph.find_node_by_id(13).unwrap().inputs, vec![11]);
        assert_eq!(graph.find_node_by_id(13).unwrap().outputs, vec![14]);
        assert_eq!(graph.find_node_by_id(14).unwrap().inputs, vec![13]);
    }
}
