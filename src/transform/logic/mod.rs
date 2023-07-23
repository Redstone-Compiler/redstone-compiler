use std::collections::{HashMap, HashSet, VecDeque};

use disjoint_set::DisjointSet;
use itertools::Itertools;

use crate::{
    graph::{
        builder::logic::{LogicGraph, LogicGraphBuilder},
        Graph, GraphNode, GraphNodeId, GraphNodeKind, SubGraph,
    },
    logic::{Logic, LogicType},
};

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

            if node.outputs.len() != 1 {
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

            let input = t1_node.inputs[0];
            let outputs = t2_node.outputs.clone();

            self.graph
                .graph
                .replace_target_output_node_ids(input, t1, outputs.clone());

            for output in outputs {
                self.graph
                    .graph
                    .replace_target_input_node_ids(output, t2, vec![input]);
            }

            self.graph.graph.remove_by_node_id_lazy(t1);
            self.graph.graph.remove_by_node_id_lazy(t2);
        }

        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
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

    pub fn optimize_cse(&mut self) -> eyre::Result<()> {
        todo!()
    }

    pub fn cluster(&self, include_ouput_node: bool) -> Vec<SubGraph> {
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
            .into_iter()
            .map(|(_, nodes)| SubGraph {
                graph: &self.graph.graph,
                nodes: nodes,
            })
            .collect_vec()
    }
}
