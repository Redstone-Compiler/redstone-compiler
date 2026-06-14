use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use super::LogicGraphTransformer;
use crate::graph::logic::LogicGraphBuilder;
use crate::graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::{Logic, LogicType};

#[derive(Clone, Default)]
struct GraphPatternMatch {
    bindings: HashMap<String, GraphNodeId>,
    logic_nodes: HashMap<GraphNodeId, GraphNodeId>,
}

impl GraphPatternMatch {
    fn bind_input(mut self, name: &str, graph_node_id: GraphNodeId) -> Option<Self> {
        if let Some(existing) = self.bindings.get(name).copied() {
            return (existing == graph_node_id).then_some(self);
        }
        self.bindings.insert(name.to_owned(), graph_node_id);
        Some(self)
    }

    fn bind_logic(
        mut self,
        pattern_node_id: GraphNodeId,
        graph_node_id: GraphNodeId,
    ) -> Option<Self> {
        if let Some(existing) = self.logic_nodes.get(&pattern_node_id).copied() {
            return (existing == graph_node_id).then_some(self);
        }
        if self.logic_nodes.values().any(|id| *id == graph_node_id) {
            return None;
        }
        self.logic_nodes.insert(pattern_node_id, graph_node_id);
        Some(self)
    }
}

impl LogicGraphTransformer {
    pub fn compose_high_level_gates(&mut self) -> eyre::Result<()> {
        self.compose_gate_pattern(("(~(x|~y))|(~(y|~x))", LogicType::Xor, "composed-xor"))?;
        self.compose_gate_pattern(("~(~x|~y)", LogicType::And, "composed-and"))?;
        self.optimize_cse()?;

        for pattern in [
            ("(~x&y)|(x&~y)", LogicType::Xor, "composed-xor"),
            ("(x&(~x|~y))|(y&(~x|~y))", LogicType::Xor, "composed-xor"),
            ("(x&(~x|~y))|~((x&y)|~y)", LogicType::Xor, "composed-xor"),
        ] {
            self.compose_gate_pattern(pattern)?;
        }
        self.compose_truth_table_gates()?;
        self.graph.graph.verify()?;

        Ok(())
    }

    fn compose_gate_pattern(
        &mut self,
        (pattern_stmt, logic_type, tag): (&str, LogicType, &str),
    ) -> eyre::Result<()> {
        let pattern_graph = Self::optimized_pattern_graph(pattern_stmt)?;
        let pattern_inputs = pattern_graph
            .inputs()
            .into_iter()
            .map(|id| {
                pattern_graph
                    .find_node_by_id(id)
                    .unwrap()
                    .kind
                    .as_input()
                    .clone()
            })
            .collect_vec();
        let pattern_root_id = pattern_graph
            .find_node_by_id(pattern_graph.outputs()[0])
            .unwrap()
            .inputs[0];
        let root_logic_type = pattern_graph
            .find_node_by_id(pattern_root_id)
            .unwrap()
            .kind
            .as_logic()
            .unwrap()
            .logic_type;
        let candidate_ids = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == root_logic_type)
            })
            .map(|node| node.id)
            .collect_vec();

        for node_id in candidate_ids {
            let Some(pattern_match) = self.match_graph_pattern(
                &pattern_graph,
                pattern_root_id,
                node_id,
                GraphPatternMatch::default(),
            ) else {
                continue;
            };

            let Some(inputs) = pattern_inputs
                .iter()
                .map(|name| pattern_match.bindings.get(name).copied())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if !inputs.iter().all_unique() {
                continue;
            }

            let Some(outputs) = self
                .graph
                .graph
                .find_node_by_id(node_id)
                .map(|node| node.outputs.clone())
            else {
                continue;
            };

            let removals = self.removable_pattern_nodes(&pattern_match.logic_nodes, node_id);
            self.graph.graph.replace_nodes_with(
                &removals,
                GraphNodeKind::Logic(Logic { logic_type }),
                inputs,
                outputs,
                tag.to_owned(),
            );
        }

        Ok(())
    }

    fn optimized_pattern_graph(pattern_stmt: &str) -> eyre::Result<Graph> {
        let pattern_graph =
            LogicGraphBuilder::new(pattern_stmt.to_owned()).build("z".to_owned())?;
        let mut transformer = LogicGraphTransformer::new(pattern_graph);
        transformer.optimize_cse()?;
        Ok(transformer.finish().graph)
    }

    fn compose_truth_table_gates(&mut self) -> eyre::Result<()> {
        let candidate_ids = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.kind, GraphNodeKind::Logic(_)) && !node.tag.starts_with("composed-")
            })
            .map(|node| node.id)
            .collect_vec();

        for node_id in candidate_ids {
            let Some((logic_type, inputs, removals)) = self.truth_table_gate_candidate(node_id)?
            else {
                continue;
            };
            let Some(outputs) = self
                .graph
                .graph
                .find_node_by_id(node_id)
                .map(|node| node.outputs.clone())
            else {
                continue;
            };

            let tag = Self::composed_gate_tag(logic_type);
            let inputs = self.binary_gate_inputs_for_replacement(logic_type, &inputs, tag);
            self.graph.graph.replace_nodes_with(
                &removals,
                GraphNodeKind::Logic(Logic { logic_type }),
                inputs,
                outputs,
                tag.to_owned(),
            );
        }

        Ok(())
    }

    fn truth_table_gate_candidate(
        &self,
        root_id: GraphNodeId,
    ) -> eyre::Result<Option<(LogicType, Vec<GraphNodeId>, HashSet<GraphNodeId>)>> {
        if self.graph.graph.find_node_by_id(root_id).is_none() {
            return Ok(None);
        }

        let mut cone_nodes = HashSet::new();
        let mut inputs = HashSet::new();
        self.collect_truth_table_cone(root_id, &mut cone_nodes, &mut inputs)?;

        let mut inputs = inputs.into_iter().collect_vec();
        inputs.sort_unstable();
        if !(2..=3).contains(&inputs.len()) {
            return Ok(None);
        }

        let table = self.evaluate_cone_truth_table(root_id, &inputs)?;
        let logic_type = if Self::is_xor_truth_table(&table, inputs.len()) {
            LogicType::Xor
        } else if inputs.len() == 2 && Self::is_and_truth_table(&table, inputs.len()) {
            LogicType::And
        } else {
            return Ok(None);
        };

        Ok(Some((
            logic_type,
            inputs,
            self.removable_cone_nodes(&cone_nodes, root_id),
        )))
    }

    fn collect_truth_table_cone(
        &self,
        node_id: GraphNodeId,
        cone_nodes: &mut HashSet<GraphNodeId>,
        inputs: &mut HashSet<GraphNodeId>,
    ) -> eyre::Result<()> {
        let Some(node) = self.graph.graph.find_node_by_id(node_id) else {
            eyre::bail!("truth-table compose references missing node {node_id}");
        };

        if !matches!(&node.kind, GraphNodeKind::Logic(_)) {
            inputs.insert(node_id);
            return Ok(());
        }

        if !cone_nodes.insert(node_id) {
            return Ok(());
        }

        for input in &node.inputs {
            self.collect_truth_table_cone(*input, cone_nodes, inputs)?;
        }

        Ok(())
    }

    fn evaluate_cone_truth_table(
        &self,
        root_id: GraphNodeId,
        inputs: &[GraphNodeId],
    ) -> eyre::Result<Vec<bool>> {
        (0..(1usize << inputs.len()))
            .map(|mask| {
                let input_values = inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| (*input, (mask & (1 << index)) != 0))
                    .collect::<HashMap<_, _>>();
                self.evaluate_cone_node(root_id, &input_values, &mut HashMap::new())
            })
            .collect()
    }

    fn evaluate_cone_node(
        &self,
        node_id: GraphNodeId,
        input_values: &HashMap<GraphNodeId, bool>,
        memo: &mut HashMap<GraphNodeId, bool>,
    ) -> eyre::Result<bool> {
        if let Some(value) = input_values.get(&node_id) {
            return Ok(*value);
        }
        if let Some(value) = memo.get(&node_id) {
            return Ok(*value);
        }

        let Some(node) = self.graph.graph.find_node_by_id(node_id) else {
            eyre::bail!("truth-table compose evaluates missing node {node_id}");
        };
        let Some(logic) = node.kind.as_logic() else {
            eyre::bail!("truth-table compose reached unbound leaf node {node_id}");
        };
        let values = node
            .inputs
            .iter()
            .map(|input| self.evaluate_cone_node(*input, input_values, memo))
            .collect::<eyre::Result<Vec<_>>>()?;
        let value = match logic.logic_type {
            LogicType::Not => !values[0],
            LogicType::And => values.iter().all(|value| *value),
            LogicType::Or => values.iter().any(|value| *value),
            LogicType::Xor => values.iter().filter(|value| **value).count() % 2 == 1,
        };

        memo.insert(node_id, value);
        Ok(value)
    }

    fn is_xor_truth_table(table: &[bool], input_count: usize) -> bool {
        table
            .iter()
            .enumerate()
            .all(|(mask, value)| *value == (mask.count_ones() as usize % 2 == 1))
            && input_count >= 2
    }

    fn is_and_truth_table(table: &[bool], input_count: usize) -> bool {
        let all_enabled = (1usize << input_count) - 1;
        table
            .iter()
            .enumerate()
            .all(|(mask, value)| *value == (mask == all_enabled))
    }

    fn composed_gate_tag(logic_type: LogicType) -> &'static str {
        match logic_type {
            LogicType::And => "composed-and",
            LogicType::Xor => "composed-xor",
            _ => unreachable!(),
        }
    }

    fn binary_gate_inputs_for_replacement(
        &mut self,
        logic_type: LogicType,
        inputs: &[GraphNodeId],
        tag: &str,
    ) -> Vec<GraphNodeId> {
        if inputs.len() <= 2 {
            return inputs.to_vec();
        }

        let mut current =
            self.add_connected_logic_node(logic_type, vec![inputs[0], inputs[1]], tag);
        for input in inputs.iter().take(inputs.len() - 1).skip(2) {
            current = self.add_connected_logic_node(logic_type, vec![current, *input], tag);
        }

        vec![current, *inputs.last().unwrap()]
    }

    fn add_connected_logic_node(
        &mut self,
        logic_type: LogicType,
        inputs: Vec<GraphNodeId>,
        tag: &str,
    ) -> GraphNodeId {
        let node_id = self.graph.graph.add_node(GraphNode {
            kind: GraphNodeKind::Logic(Logic { logic_type }),
            inputs: inputs.clone(),
            tag: tag.to_owned(),
            ..Default::default()
        });

        for input in inputs {
            if let Some(mut input_node) = self.graph.graph.find_node_by_id_mut(input) {
                if !input_node.outputs.contains(&node_id) {
                    input_node.outputs.push(node_id);
                }
            }
        }

        node_id
    }

    fn match_graph_pattern(
        &self,
        pattern_graph: &Graph,
        pattern_node_id: GraphNodeId,
        graph_node_id: GraphNodeId,
        pattern_match: GraphPatternMatch,
    ) -> Option<GraphPatternMatch> {
        let pattern_node = pattern_graph.find_node_by_id(pattern_node_id)?;
        if let GraphNodeKind::Input(name) = &pattern_node.kind {
            return pattern_match.bind_input(name, graph_node_id);
        }

        let pattern_logic = pattern_node.kind.as_logic()?;
        let graph_node = self.graph.graph.find_node_by_id(graph_node_id)?;
        let graph_logic = graph_node.kind.as_logic()?;
        let pattern_inputs = pattern_node.inputs.clone();
        let graph_inputs = graph_node.inputs.clone();

        if pattern_logic.logic_type != graph_logic.logic_type
            || pattern_inputs.len() != graph_inputs.len()
        {
            return None;
        }

        let pattern_match = pattern_match.bind_logic(pattern_node_id, graph_node_id)?;
        Self::ordered_logic_inputs(pattern_logic.logic_type, graph_inputs)
            .into_iter()
            .find_map(|graph_inputs| {
                self.match_pattern_inputs(
                    pattern_graph,
                    &pattern_inputs,
                    graph_inputs,
                    pattern_match.clone(),
                )
            })
    }

    fn ordered_logic_inputs(
        logic_type: LogicType,
        graph_inputs: Vec<GraphNodeId>,
    ) -> Vec<Vec<GraphNodeId>> {
        if matches!(logic_type, LogicType::Or | LogicType::And | LogicType::Xor)
            && graph_inputs.len() == 2
            && graph_inputs[0] != graph_inputs[1]
        {
            let reversed_inputs = vec![graph_inputs[1], graph_inputs[0]];
            return vec![graph_inputs, reversed_inputs];
        }
        vec![graph_inputs]
    }

    fn match_pattern_inputs(
        &self,
        pattern_graph: &Graph,
        pattern_inputs: &[GraphNodeId],
        graph_inputs: Vec<GraphNodeId>,
        mut pattern_match: GraphPatternMatch,
    ) -> Option<GraphPatternMatch> {
        for (pattern_input, graph_input) in pattern_inputs.iter().zip(graph_inputs) {
            pattern_match = self.match_graph_pattern(
                pattern_graph,
                *pattern_input,
                graph_input,
                pattern_match,
            )?;
        }
        Some(pattern_match)
    }

    fn removable_pattern_nodes(
        &self,
        matched_logic_nodes: &HashMap<GraphNodeId, GraphNodeId>,
        root_id: GraphNodeId,
    ) -> HashSet<GraphNodeId> {
        let matched_ids = matched_logic_nodes.values().copied().collect_vec();
        let mut removals = HashSet::from([root_id]);

        loop {
            let mut changed = false;
            for node_id in &matched_ids {
                if removals.contains(node_id) {
                    continue;
                }
                let Some(node) = self.graph.graph.find_node_by_id(*node_id) else {
                    continue;
                };
                if node.outputs.iter().all(|output| removals.contains(output)) {
                    removals.insert(*node_id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        removals
    }

    fn removable_cone_nodes(
        &self,
        cone_nodes: &HashSet<GraphNodeId>,
        root_id: GraphNodeId,
    ) -> HashSet<GraphNodeId> {
        let mut removals = HashSet::from([root_id]);

        loop {
            let mut changed = false;
            for node_id in cone_nodes {
                if removals.contains(node_id) {
                    continue;
                }
                let Some(node) = self.graph.graph.find_node_by_id(*node_id) else {
                    continue;
                };
                if node.outputs.iter().all(|output| removals.contains(output)) {
                    removals.insert(*node_id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        removals
    }
}
