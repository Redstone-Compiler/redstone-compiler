use std::collections::HashMap;

use itertools::Itertools;

use super::LogicGraphTransformer;
use crate::graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::{Logic, LogicType};

impl LogicGraphTransformer {
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
}
