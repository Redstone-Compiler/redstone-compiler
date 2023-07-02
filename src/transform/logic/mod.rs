use itertools::Itertools;

use crate::{
    graph::{
        builder::logic::{LogicGraph, LogicGraphBuilder},
        Graph, GraphNode, GraphNodeId, GraphNodeKind,
    },
    logic::{Logic, LogicType},
};

pub struct LogicGraphTransformer {
    graph: LogicGraph,
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

        let and_gate = LogicGraphBuilder::new("~(x|y)".to_string())
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
    pub fn fusion_ops(&mut self) {
        todo!()
    }

    // replacable only (x, y) => (z)
    fn replace_binops_lazy(&mut self, src: GraphNodeId, mut tar: Graph) -> eyre::Result<()> {
        tar.rebuild_node_id_base(self.graph.graph.max_node_id().unwrap() + 1);

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

        for index in 0..=1 {
            if let Some(node) = g.find_node_by_id_mut(src_inputs[index]) {
                node.outputs.extend(tar_inputs_outputs[index].clone());
                node.outputs.retain(|node_id| *node_id != src);
            }
        }

        tar.replace_node_id_lazy(tar_inputs[0], src_inputs[0]);
        tar.replace_node_id_lazy(tar_inputs[1], src_inputs[1]);

        g.replace_input_node_id_lazy(src, tar_output_input);
        g.remove_by_node_id_lazy(src);

        tar.find_node_by_id_mut(tar_output_input).unwrap().outputs = src_outputs.clone();
        tar.remove_by_node_id_lazy(tar_inputs[0]);
        tar.remove_by_node_id_lazy(tar_inputs[1]);
        tar.remove_by_node_id_lazy(tar_output);

        g.nodes.extend(tar.nodes);

        Ok(())
    }

    pub fn optimize(&mut self) -> eyre::Result<()> {
        if self.graph.graph.outputs().len() != 1 {
            eyre::bail!("You must split by outputs before run optimizing!");
        }

        // optimize logic graph using quine mccluskey
        fn make_qmc_form(
            graph: &Graph,
            node_id: GraphNodeId,
        ) -> eyre::Result<quine_mc_cluskey::Bool> {
            let node = graph.find_node_by_id(node_id).unwrap();
            match &node.kind {
                GraphNodeKind::Input(_) => Ok(quine_mc_cluskey::Bool::Term(node_id as u8)),
                GraphNodeKind::Output(_) => make_qmc_form(graph, node.inputs[0]),
                GraphNodeKind::Logic(logic) => Ok(match &logic.logic_type {
                    LogicType::Not => {
                        quine_mc_cluskey::Bool::Not(Box::new(make_qmc_form(graph, node.inputs[0])?))
                    }
                    LogicType::And => quine_mc_cluskey::Bool::And(
                        node.inputs
                            .iter()
                            .map(|input| make_qmc_form(graph, *input).unwrap())
                            .collect(),
                    ),
                    LogicType::Or => quine_mc_cluskey::Bool::Or(
                        node.inputs
                            .iter()
                            .map(|input| make_qmc_form(graph, *input).unwrap())
                            .collect(),
                    ),
                    LogicType::Xor => unimplemented!(),
                }),
                _ => unreachable!(),
            }
        }

        let results = make_qmc_form(&self.graph.graph, self.graph.graph.outputs()[0])?.simplify();

        let mut nodes = Vec::new();

        fn make_rc_form(
            nodes: &mut Vec<GraphNode>,
            id: &mut usize,
            lookup: &Vec<GraphNode>,
            node: &quine_mc_cluskey::Bool,
        ) -> GraphNodeId {
            let tid = *id;
            *id += 1;

            let node = match node {
                quine_mc_cluskey::Bool::Term(v) => GraphNode {
                    id: tid,
                    kind: GraphNodeKind::Input(lookup[*v as usize].kind.as_input().to_owned()),
                    ..Default::default()
                },
                quine_mc_cluskey::Bool::And(op) => GraphNode {
                    id: tid,
                    kind: GraphNodeKind::Logic(Logic {
                        logic_type: LogicType::And,
                    }),
                    inputs: op
                        .iter()
                        .map(|v| make_rc_form(nodes, id, lookup, v))
                        .collect_vec(),
                    ..Default::default()
                },
                quine_mc_cluskey::Bool::Or(op) => GraphNode {
                    id: tid,
                    kind: GraphNodeKind::Logic(Logic {
                        logic_type: LogicType::Or,
                    }),
                    inputs: op
                        .iter()
                        .map(|v| make_rc_form(nodes, id, lookup, v))
                        .collect_vec(),
                    ..Default::default()
                },
                quine_mc_cluskey::Bool::Not(v) => GraphNode {
                    id: tid,
                    kind: GraphNodeKind::Logic(Logic {
                        logic_type: LogicType::Not,
                    }),
                    inputs: vec![make_rc_form(nodes, id, lookup, v)],
                    ..Default::default()
                },
                _ => unreachable!(),
            };

            if let GraphNodeKind::Input(name) = &node.kind {
                if let Some(node) = nodes
                    .iter()
                    .find(|node| matches!(&node.kind, GraphNodeKind::Input(other) if name == other))
                {
                    *id -= 1;
                    return node.id;
                }
            }

            nodes.push(node);
            tid
        }

        let mut id = 0;
        let node = make_rc_form(&mut nodes, &mut id, &self.graph.graph.nodes, &results[0]);
        let output_node = GraphNode {
            id: nodes.len(),
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
        };
        nodes.push(output_node);
        nodes.sort_by_key(|node| node.id);

        self.graph.graph.nodes = nodes;
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();

        Ok(())
    }
}
