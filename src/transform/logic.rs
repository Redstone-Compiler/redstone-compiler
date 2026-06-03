use std::collections::{HashMap, HashSet, VecDeque};

use disjoint_set::DisjointSet;
use itertools::Itertools;

use crate::graph::logic::{LogicGraph, LogicGraphBuilder};
use crate::graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind, SubGraphWithGraph};
use crate::logic::{Logic, LogicType};

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

    // |(a, b, c) => (a | b) | c
    pub fn decompose_binops(&mut self) {
        todo!()
    }

    // a & b => ~(a | b)
    pub fn decompose_and(&mut self) -> eyre::Result<()> {
        // check decomposable
        if self.graph.graph.nodes.iter().any(|node| match &node.kind {
            GraphNodeKind::Logic(logic) => match logic.logic_type {
                LogicType::And if node.inputs.len() != 2 => false,
                _ => false,
            },
            _ => false,
        }) {
            eyre::bail!("Cannot decompose! Run decompose binops before decomposing and!");
        }

        let target_nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.kind,
                    GraphNodeKind::Logic(logic) if matches!(logic.logic_type, LogicType::And),
                )
            })
            .map(|node| node.id)
            .collect_vec();

        let and_gate = LogicGraphBuilder::new("~(~x|~y)".to_string())
            .build("z".to_string())
            .unwrap();

        for node_id in target_nodes {
            self.replace_binops_lazy(node_id, and_gate.graph.clone())?;
        }

        // for sort input sequence by id
        self.graph.graph.build_inputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();

        Ok(())
    }

    // a ^ b => (~a & b) | (a & ~b)
    pub fn decompose_xor(&mut self) -> eyre::Result<()> {
        // check decomposable
        if self.graph.graph.nodes.iter().any(|node| match &node.kind {
            GraphNodeKind::Logic(logic) => match logic.logic_type {
                LogicType::Xor if node.inputs.len() != 2 => false,
                _ => false,
            },
            _ => false,
        }) {
            eyre::bail!("Cannot decompose! Run decompose binops before decomposing xor!");
        }

        let target_nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.kind,
                    GraphNodeKind::Logic(logic) if matches!(logic.logic_type, LogicType::Xor),
                )
            })
            .map(|node| node.id)
            .collect_vec();

        let xor_gate = LogicGraphBuilder::new("(~x&y)|(x&~y)".to_string())
            .build("z".to_string())
            .unwrap();

        for node_id in target_nodes {
            self.replace_binops_lazy(node_id, xor_gate.graph.clone())?;
        }

        // for sort input sequence by id
        self.graph.graph.build_inputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();

        Ok(())
    }

    // (a | b) | (c | d) => |(a, b, c, d)
    pub fn fusion_orops(&mut self) {
        todo!()
    }

    // (a & b) & (c & d) => &(a, b, c, d)
    pub fn fusion_andops(&mut self) {
        todo!()
    }

    // ~(~a) => a
    pub fn remove_double_neg_expression(&mut self) {
        let neg_nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.kind, GraphNodeKind::Logic(Logic { logic_type }) if *logic_type == LogicType::Not))
            .collect_vec();

        let mut remove_target_ops = Vec::new();
        let mut remove_targets = HashSet::new();

        for node in neg_nodes {
            if remove_targets.contains(&node.id) {
                continue;
            }

            if node.outputs.is_empty() {
                continue;
            }

            let output = self.graph.graph.find_node_by_id(node.outputs[0]).unwrap();

            if matches!(&output.kind, GraphNodeKind::Logic(Logic { logic_type }) if *logic_type == LogicType::Not)
            {
                remove_targets.insert(node.id);
                remove_targets.insert(output.id);
                remove_target_ops.push((node.id, output.id));
            }
        }

        for (t1, t2) in remove_target_ops {
            let t1_node = self.graph.graph.find_node_by_id(t1).unwrap();
            let t2_node = self.graph.graph.find_node_by_id(t2).unwrap();

            assert_eq!(t1_node.inputs.len(), 1);
            let input = t1_node.inputs[0];

            let outputs = t2_node.outputs.clone();

            //   in                   in
            //   |                    | \
            // not(t1)                |  not(t1)
            //   | \       =>         |    \
            //   |  \                out1   out2
            //   |   \
            // not(t2)\
            //   |     \
            //  out1   out2
            let reserve_t1 = t1_node.outputs.len() > 1;

            if reserve_t1 {
                self.graph
                    .graph
                    .find_node_by_id_mut(input)
                    .unwrap()
                    .outputs
                    .extend(outputs.clone());
            } else {
                self.graph
                    .graph
                    .replace_target_output_node_ids(input, t1, outputs.clone());
            }

            for &output in &outputs {
                self.graph
                    .graph
                    .replace_target_input_node_ids(output, t2, vec![input]);
            }

            if reserve_t1 {
                self.graph
                    .graph
                    .find_node_by_id_mut(t1)
                    .unwrap()
                    .outputs
                    .retain(|id| *id != t2);
            } else {
                self.graph.graph.remove_by_node_id_lazy(t1);
            }
            self.graph.graph.remove_by_node_id_lazy(t2);
        }

        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
    }

    // replacable only (x, y) => (z)
    fn replace_binops_lazy(&mut self, src: GraphNodeId, mut tar: Graph) -> eyre::Result<()> {
        let g = &mut self.graph.graph;

        let src_inputs = g.find_node_by_id(src).unwrap().inputs.clone();
        let src_outputs = g.find_node_by_id(src).unwrap().outputs.clone();

        let tar_inputs = tar.inputs();
        let tar_inputs_outputs = tar
            .inputs()
            .iter()
            .map(|node_id| tar.find_node_by_id(*node_id).unwrap().outputs.clone())
            .collect_vec();
        let tar_output = tar.outputs()[0];
        let tar_output_input = tar.find_node_by_id(tar.outputs()[0]).unwrap().inputs[0];
        let old_to_existing_ids = HashMap::from([
            (tar_inputs[0], src_inputs[0]),
            (tar_inputs[1], src_inputs[1]),
        ]);

        tar.remove_by_node_id_lazy(tar_output);
        let old_to_current_ids = g.append_graph_with_replacements(tar, &old_to_existing_ids);

        for index in 0..=1 {
            if let Some(mut node) = g.find_node_by_id_mut(src_inputs[index]) {
                node.outputs.extend(
                    tar_inputs_outputs[index]
                        .iter()
                        .map(|id| old_to_current_ids.get(id).copied().unwrap_or(*id)),
                );
                node.outputs.retain(|node_id| *node_id != src);
            }
        }

        let tar_output_input = old_to_current_ids
            .get(&tar_output_input)
            .copied()
            .unwrap_or(tar_output_input);

        g.replace_input_node_id_lazy(src, tar_output_input);
        g.remove_by_node_id_lazy(src);
        g.find_node_by_id_mut(tar_output_input).unwrap().outputs = src_outputs;

        Ok(())
    }

    pub fn optimize(&mut self) -> eyre::Result<()> {
        if self.graph.graph.outputs().len() != 1 {
            eyre::bail!("You must split by outputs before run optimizing!");
        }

        // optimize logic graph using quine mccluskey
        let input_nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, GraphNodeKind::Input(_)))
            .map(|node| (node.id, node.clone_node()))
            .collect_vec();
        let input_terms = input_nodes
            .iter()
            .enumerate()
            .map(|(index, (node_id, _))| (*node_id, index as u8))
            .collect::<HashMap<_, _>>();

        fn make_qmc_form(
            graph: &Graph,
            input_terms: &HashMap<GraphNodeId, u8>,
            node_id: GraphNodeId,
        ) -> eyre::Result<quine_mc_cluskey::Bool> {
            let node = graph.find_node_by_id(node_id).unwrap();
            match &node.kind {
                GraphNodeKind::Input(_) => Ok(quine_mc_cluskey::Bool::Term(input_terms[&node_id])),
                GraphNodeKind::Output(_) => make_qmc_form(graph, input_terms, node.inputs[0]),
                GraphNodeKind::Logic(logic) => Ok(match &logic.logic_type {
                    LogicType::Not => quine_mc_cluskey::Bool::Not(Box::new(make_qmc_form(
                        graph,
                        input_terms,
                        node.inputs[0],
                    )?)),
                    LogicType::And => quine_mc_cluskey::Bool::And(
                        node.inputs
                            .iter()
                            .map(|input| make_qmc_form(graph, input_terms, *input).unwrap())
                            .collect(),
                    ),
                    LogicType::Or => quine_mc_cluskey::Bool::Or(
                        node.inputs
                            .iter()
                            .map(|input| make_qmc_form(graph, input_terms, *input).unwrap())
                            .collect(),
                    ),
                    LogicType::Xor => unimplemented!(),
                }),
                _ => unreachable!(),
            }
        }

        let results = make_qmc_form(
            &self.graph.graph,
            &input_terms,
            self.graph.graph.outputs()[0],
        )?
        .simplify();

        fn make_rc_form(
            graph: &mut Graph,
            lookup: &Vec<(GraphNodeId, GraphNode)>,
            node: &quine_mc_cluskey::Bool,
        ) -> GraphNodeId {
            let node = match node {
                quine_mc_cluskey::Bool::Term(v) => GraphNode {
                    kind: GraphNodeKind::Input(lookup[*v as usize].1.kind.as_input().to_owned()),
                    ..Default::default()
                },
                quine_mc_cluskey::Bool::And(op) => GraphNode {
                    kind: GraphNodeKind::Logic(Logic {
                        logic_type: LogicType::And,
                    }),
                    inputs: op
                        .iter()
                        .map(|v| make_rc_form(graph, lookup, v))
                        .collect_vec(),
                    ..Default::default()
                },
                quine_mc_cluskey::Bool::Or(op) => GraphNode {
                    kind: GraphNodeKind::Logic(Logic {
                        logic_type: LogicType::Or,
                    }),
                    inputs: op
                        .iter()
                        .map(|v| make_rc_form(graph, lookup, v))
                        .collect_vec(),
                    ..Default::default()
                },
                quine_mc_cluskey::Bool::Not(v) => GraphNode {
                    kind: GraphNodeKind::Logic(Logic {
                        logic_type: LogicType::Not,
                    }),
                    inputs: vec![make_rc_form(graph, lookup, v)],
                    ..Default::default()
                },
                _ => unreachable!(),
            };

            if let GraphNodeKind::Input(name) = &node.kind {
                if let Some(existing) = graph.nodes.iter().find(
                    |existing| matches!(&existing.kind, GraphNodeKind::Input(other) if name == other),
                ) {
                    return existing.id;
                }
            }

            graph.add_node(node)
        }

        let mut graph = Graph::default();
        let node = make_rc_form(&mut graph, &input_nodes, &results[0]);
        graph.add_node(GraphNode {
            kind: GraphNodeKind::Output(
                self.graph
                    .graph
                    .find_node_by_id(self.graph.graph.outputs()[0])
                    .unwrap()
                    .kind
                    .as_output()
                    .to_owned(),
            ),
            inputs: vec![node],
            ..Default::default()
        });
        self.graph.graph = graph;
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();

        Ok(())
    }

    // Merge structurally equivalent logic nodes so shared sub-expressions are
    // placed once and reused by all consumers. This is intentionally structural
    // CSE only: it canonicalizes commutative input order, but does not apply
    // boolean algebra rewrites such as idempotence or absorption.
    pub fn optimize_cse(&mut self) -> eyre::Result<()> {
        let mut expressions: HashMap<GraphNodeId, String> = HashMap::new();
        let mut representatives: HashMap<String, GraphNodeId> = HashMap::new();
        let mut replacements = Vec::new();

        for node_id in self.graph.graph.topological_order() {
            let Some(node) = self.graph.graph.find_node_by_id(node_id) else {
                continue;
            };

            let expression = match &node.kind {
                GraphNodeKind::Input(name) => format!("Input({name})"),
                GraphNodeKind::Logic(logic) => {
                    let mut inputs = node
                        .inputs
                        .iter()
                        .map(|input| {
                            expressions.get(input).cloned().ok_or_else(|| {
                                eyre::eyre!("CSE expression is missing for input node {input}")
                            })
                        })
                        .collect::<eyre::Result<Vec<_>>>()?;

                    if matches!(
                        logic.logic_type,
                        LogicType::Or | LogicType::And | LogicType::Xor
                    ) {
                        inputs.sort();
                    }

                    format!("{}({})", logic.logic_type.name(), inputs.join(", "))
                }
                GraphNodeKind::Sequential(sequential) => {
                    format!("Sequential#{}({})", node.id, sequential.name())
                }
                GraphNodeKind::Output(_) => continue,
                _ => continue,
            };

            expressions.insert(node_id, expression.clone());

            if !node.kind.is_logic() {
                continue;
            }

            if let Some(&representative) = representatives.get(&expression) {
                replacements.push((node_id, representative));
            } else {
                representatives.insert(expression, node_id);
            }
        }

        for (from, to) in replacements {
            self.graph.graph.replace_input_node_id_lazy(from, to);
            self.graph.graph.remove_by_node_id_lazy(from);
        }

        self.graph.graph.build_outputs();
        self.graph.graph.build_inputs();
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
        self.graph.graph.verify()?;

        Ok(())
    }

    pub fn fold_or_chains(&mut self) -> eyre::Result<()> {
        let or_node_ids = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
            })
            .map(|node| node.id)
            .collect_vec();

        for node_id in or_node_ids {
            let Some(node) = self.graph.graph.find_node_by_id(node_id) else {
                continue;
            };
            if !matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
            {
                continue;
            }

            let mut folded_inputs = Vec::new();
            let mut removable_or_nodes = HashSet::new();

            for input in node.inputs.clone() {
                Self::collect_folded_or_inputs(
                    &self.graph.graph,
                    input,
                    node_id,
                    &mut folded_inputs,
                    &mut removable_or_nodes,
                    &mut HashSet::new(),
                )?;
            }

            folded_inputs.sort_unstable();
            folded_inputs.dedup();

            match folded_inputs.len() {
                0 => {}
                1 => {
                    let input = folded_inputs[0];
                    self.graph.graph.replace_node_id_lazy(node_id, input);
                    removable_or_nodes.insert(node_id);
                }
                _ => {
                    let inputs = self.binary_or_inputs_ending_at(&folded_inputs);
                    if let Some(mut node) = self.graph.graph.find_node_by_id_mut(node_id) {
                        node.inputs = inputs;
                    }
                }
            }

            for removable in removable_or_nodes {
                self.graph.graph.remove_by_node_id_lazy(removable);
            }

            self.graph.graph.build_outputs();
            self.graph.graph.build_inputs();
            self.graph.graph.build_outputs();
            self.graph.graph.build_producers();
            self.graph.graph.build_consumers();
        }

        self.graph.graph.verify()?;

        Ok(())
    }

    fn binary_or_inputs_ending_at(&mut self, inputs: &[GraphNodeId]) -> Vec<GraphNodeId> {
        assert!(inputs.len() >= 2);
        if inputs.len() == 2 {
            return inputs.to_vec();
        }

        let mut current = self.graph.graph.add_node(GraphNode {
            kind: GraphNodeKind::Logic(Logic {
                logic_type: LogicType::Or,
            }),
            inputs: vec![inputs[0], inputs[1]],
            tag: "folded-or".to_owned(),
            ..Default::default()
        });

        for input in inputs.iter().take(inputs.len() - 1).skip(2) {
            current = self.graph.graph.add_node(GraphNode {
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Or,
                }),
                inputs: vec![current, *input],
                tag: "folded-or".to_owned(),
                ..Default::default()
            });
        }

        vec![current, *inputs.last().unwrap()]
    }

    fn collect_folded_or_inputs(
        graph: &Graph,
        input_id: GraphNodeId,
        consumer_id: GraphNodeId,
        folded_inputs: &mut Vec<GraphNodeId>,
        removals: &mut HashSet<GraphNodeId>,
        path: &mut HashSet<GraphNodeId>,
    ) -> eyre::Result<()> {
        if !path.insert(input_id) {
            eyre::bail!("cannot fold OR chain through cyclic node {input_id}");
        }

        let Some(node) = graph.find_node_by_id(input_id) else {
            eyre::bail!("cannot fold OR chain through missing node {input_id}");
        };

        if !matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or) {
            folded_inputs.push(input_id);
            path.remove(&input_id);
            return Ok(());
        }

        let can_remove_current = node.outputs.iter().all(|output| *output == consumer_id);
        if can_remove_current {
            removals.insert(input_id);
        }

        for nested_input in &node.inputs {
            Self::collect_folded_or_inputs(
                graph,
                *nested_input,
                input_id,
                folded_inputs,
                removals,
                path,
            )?;
        }

        path.remove(&input_id);
        Ok(())
    }

    pub fn insert_buffers_for_direct_or_to_or(&mut self) -> eyre::Result<()> {
        let mut direct_edges = Vec::new();
        for node in self.graph.graph.nodes.iter() {
            if !matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
            {
                continue;
            }
            for &output_id in &node.outputs {
                let Some(output) = self.graph.graph.find_node_by_id(output_id) else {
                    continue;
                };
                if matches!(&output.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
                {
                    direct_edges.push((node.id, output.id));
                }
            }
        }

        for (from, to) in direct_edges {
            let first_not = self.graph.graph.add_node(GraphNode {
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![from],
                tag: "auto-buffer".to_owned(),
                ..Default::default()
            });
            let second_not = self.graph.graph.add_node(GraphNode {
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![first_not],
                outputs: vec![to],
                tag: "auto-buffer".to_owned(),
            });
            self.graph
                .graph
                .find_node_by_id_mut(first_not)
                .unwrap()
                .outputs = vec![second_not];

            self.graph
                .graph
                .replace_target_output_node_ids(from, to, vec![first_not]);
            self.graph
                .graph
                .replace_target_input_node_ids(to, from, vec![second_not]);
        }

        self.graph.graph.build_inputs();
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
        self.graph.graph.verify()?;

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::analysis::equivalent_expression_groups;
    use crate::graph::logic::predefined_logics;
    use crate::sequential::{SequentialPrimitive, SequentialType};

    fn has_duplicate_expression(graph: &LogicGraph, expression: &str) -> bool {
        equivalent_expression_groups(graph)
            .iter()
            .any(|group| group.expression == expression)
    }

    #[test]
    fn optimize_cse_merges_duplicate_not_nodes() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("~a|~a", "out")?;
        let mut transform = LogicGraphTransformer::new(graph);

        transform.optimize_cse()?;

        assert!(!has_duplicate_expression(&transform.graph, "Not(Input(a))"));
        transform.graph.graph.verify()?;

        Ok(())
    }

    #[test]
    fn optimize_cse_merges_commutative_or_nodes() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("(a|b)|(b|a)", "out")?;
        let mut transform = LogicGraphTransformer::new(graph);

        transform.optimize_cse()?;

        assert!(!has_duplicate_expression(
            &transform.graph,
            "Or(Input(a), Input(b))"
        ));
        transform.graph.graph.verify()?;

        Ok(())
    }

    #[test]
    fn optimize_cse_rewrites_outputs_to_representative() -> eyre::Result<()> {
        let out_x = LogicGraph::from_stmt("~a", "x")?;
        let out_y = LogicGraph::from_stmt("~a", "y")?;

        let mut graph = out_x.clone();
        graph.graph.merge(out_y.graph);

        let mut transform = LogicGraphTransformer::new(graph);
        transform.optimize_cse()?;

        let output_inputs = transform
            .graph
            .graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) if name == "x" || name == "y" => Some(node.inputs[0]),
                _ => None,
            })
            .collect_vec();

        assert_eq!(output_inputs.len(), 2);
        assert_eq!(output_inputs[0], output_inputs[1]);
        transform.graph.graph.verify()?;

        Ok(())
    }

    #[test]
    fn fold_or_chains_preserves_binary_or_fan_in() -> eyre::Result<()> {
        let mut graph = LogicGraph::from_stmt("a|b", "x")?;
        graph.graph.merge_by_outin(
            LogicGraph::from_stmt("x|c|c", "out")?.graph,
            vec![("x", "x")],
        );
        let expected_table = graph.truth_table()?;
        let mut transform = LogicGraphTransformer::new(graph);

        transform.fold_or_chains()?;

        let or_nodes = transform
            .graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
            })
            .collect_vec();
        assert_eq!(or_nodes.len(), 2);
        assert!(or_nodes.iter().all(|node| node.inputs.len() <= 2));
        assert!(or_nodes.iter().all(|node| {
            let mut inputs = node.inputs.clone();
            inputs.sort_unstable();
            inputs.dedup();
            inputs.len() == node.inputs.len()
        }));
        assert_eq!(transform.graph.truth_table()?, expected_table);
        transform.graph.graph.verify()?;

        Ok(())
    }

    #[test]
    fn prepare_place_runs_cse_for_buffered_full_adder() -> eyre::Result<()> {
        let graph = predefined_logics::buffered_full_adder_graph()?;

        assert!(!has_duplicate_expression(&graph, "Not(Input(a))"));
        assert!(!has_duplicate_expression(&graph, "Not(Input(b))"));
        assert!(!has_duplicate_expression(&graph, "Not(Input(cin))"));
        graph.graph.verify()?;

        Ok(())
    }

    #[test]
    fn prepare_place_preserves_sequential_boundaries() -> eyre::Result<()> {
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
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![2],
                outputs: vec![4],
                ..Default::default()
            },
            GraphNode {
                kind: GraphNodeKind::Output("not_q".to_owned()),
                inputs: vec![3],
                ..Default::default()
            },
        ]);
        graph.build_inputs();
        graph.build_outputs();

        let graph = LogicGraph { graph }.prepare_place()?;

        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| node.kind.is_sequential())
                .count(),
            1
        );
        assert!(graph.nodes.iter().any(|node| matches!(
            &node.kind,
            GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Not
        )));
        graph.graph.verify()?;

        Ok(())
    }
}
