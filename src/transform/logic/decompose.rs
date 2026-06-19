use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use super::LogicGraphTransformer;
use crate::graph::logic::LogicGraphBuilder;
use crate::graph::{Graph, GraphNodeId, GraphNodeKind};
use crate::logic::{Logic, LogicType};

impl LogicGraphTransformer {
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

            let Some(output) = self.graph.graph.find_node_by_id(node.outputs[0]) else {
                continue;
            };

            if matches!(&output.kind, GraphNodeKind::Logic(Logic { logic_type }) if *logic_type == LogicType::Not)
            {
                remove_targets.insert(node.id);
                remove_targets.insert(output.id);
                remove_target_ops.push((node.id, output.id));
            }
        }

        for (t1, t2) in remove_target_ops {
            let Some(t1_node) = self.graph.graph.find_node_by_id(t1) else {
                continue;
            };
            let Some(t2_node) = self.graph.graph.find_node_by_id(t2) else {
                continue;
            };

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
}
