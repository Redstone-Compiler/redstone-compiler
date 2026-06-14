use std::collections::HashSet;

use itertools::Itertools;

use super::LogicGraphTransformer;
use crate::graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::{Logic, LogicType};

impl LogicGraphTransformer {
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
}
