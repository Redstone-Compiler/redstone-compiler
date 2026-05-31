use std::collections::{HashSet, VecDeque};

use crate::graph::logic::LogicGraph;
use crate::graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::LogicType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedbackCore {
    RsLatch(RsLatchCore),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsLatchCore {
    pub set_input: GraphNodeId,
    pub reset_input: GraphNodeId,
    pub q_output: GraphNodeId,
    pub nq_output: GraphNodeId,
    pub q_not: GraphNodeId,
    pub nq_not: GraphNodeId,
    pub q_or: GraphNodeId,
    pub nq_or: GraphNodeId,
    pub feedback_scc: Vec<GraphNodeId>,
}

pub fn feedback_cores(graph: &Graph) -> Vec<FeedbackCore> {
    recognize_rs_latch_core(graph)
        .into_iter()
        .map(FeedbackCore::RsLatch)
        .collect()
}

pub fn recognize_rs_latch_core(graph: &Graph) -> Option<RsLatchCore> {
    let q_output = find_output(graph, "q")?;
    let nq_output = find_output(graph, "nq")?;
    let q_not = single_input(graph, q_output)?;
    let nq_not = single_input(graph, nq_output)?;
    ensure_logic(graph, q_not, LogicType::Not)?;
    ensure_logic(graph, nq_not, LogicType::Not)?;

    let q_or = single_input(graph, q_not)?;
    let nq_or = single_input(graph, nq_not)?;
    ensure_logic(graph, q_or, LogicType::Or)?;
    ensure_logic(graph, nq_or, LogicType::Or)?;

    let q_or_inputs = inputs(graph, q_or)?;
    let nq_or_inputs = inputs(graph, nq_or)?;
    if q_or_inputs.len() != 2 || nq_or_inputs.len() != 2 {
        return None;
    }

    let reset_input = *q_or_inputs.iter().find(|&&input| input != nq_not)?;
    let set_input = *nq_or_inputs.iter().find(|&&input| input != q_not)?;
    if !q_or_inputs.contains(&nq_not) || !nq_or_inputs.contains(&q_not) {
        return None;
    }

    let feedback_nodes = [q_or, q_not, nq_or, nq_not]
        .into_iter()
        .collect::<HashSet<_>>();
    let feedback_scc = graph
        .strongly_connected_components()
        .into_iter()
        .find(|scc| feedback_nodes.iter().all(|node| scc.contains(node)))?;

    Some(RsLatchCore {
        set_input,
        reset_input,
        q_output,
        nq_output,
        q_not,
        nq_not,
        q_or,
        nq_or,
        feedback_scc,
    })
}

// Extract the acyclic logic that drives the set/reset inputs of an RS latch core.
// Feedback SCC nodes are excluded so the returned graph can be handled by the
// existing combinational placer.
pub fn rs_latch_prefix_graph(graph: &Graph, core: &RsLatchCore) -> Option<LogicGraph> {
    let feedback_scc = core.feedback_scc.iter().copied().collect::<HashSet<_>>();
    let output_roots = [("s", core.set_input), ("r", core.reset_input)];
    let mut prefix_nodes = HashSet::new();
    let mut queue = output_roots
        .iter()
        .map(|(_, node_id)| *node_id)
        .collect::<VecDeque<_>>();

    while let Some(node_id) = queue.pop_front() {
        if feedback_scc.contains(&node_id) {
            return None;
        }
        if !prefix_nodes.insert(node_id) {
            continue;
        }

        let node = graph.find_node_by_id(node_id)?;
        for input in &node.inputs {
            queue.push_back(*input);
        }
    }

    let mut nodes = graph
        .nodes
        .iter()
        .filter(|node| prefix_nodes.contains(&node.id))
        .map(|node| {
            let mut graph_node = node.clone_node();
            graph_node
                .inputs
                .retain(|input| prefix_nodes.contains(input));
            graph_node.outputs.clear();
            (node.id, graph_node)
        })
        .collect::<Vec<_>>();

    let mut next_node_id = graph.next_node_id();
    for (output_name, source_node_id) in output_roots {
        nodes.push((
            next_node_id,
            GraphNode {
                kind: GraphNodeKind::Output(output_name.to_owned()),
                inputs: vec![source_node_id],
                ..Default::default()
            },
        ));
        next_node_id += 1;
    }
    let mut graph = Graph::from_nodes_with_ids(nodes);
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify().ok()?;

    Some(LogicGraph { graph })
}

fn find_output(graph: &Graph, output_name: &str) -> Option<GraphNodeId> {
    graph.nodes.iter().find_map(|node| {
        matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name).then_some(node.id)
    })
}

fn single_input(graph: &Graph, node_id: GraphNodeId) -> Option<GraphNodeId> {
    let node = graph.find_node_by_id(node_id)?;
    (node.inputs.len() == 1).then_some(node.inputs[0])
}

fn inputs(graph: &Graph, node_id: GraphNodeId) -> Option<Vec<GraphNodeId>> {
    Some(graph.find_node_by_id(node_id)?.inputs.clone())
}

fn ensure_logic(graph: &Graph, node_id: GraphNodeId, logic_type: LogicType) -> Option<()> {
    let node = graph.find_node_by_id(node_id)?;
    matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == logic_type)
        .then_some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequential::SequentialPrimitive;

    #[test]
    fn recognize_rs_latch_core_from_gate_level_decomposition() {
        let primitive = SequentialPrimitive::rs_latch();
        let core = recognize_rs_latch_core(&primitive.inner_graph).unwrap();

        assert_eq!(core.set_input, 0);
        assert_eq!(core.reset_input, 1);
        assert_eq!(core.q_or, 2);
        assert_eq!(core.q_not, 3);
        assert_eq!(core.nq_or, 4);
        assert_eq!(core.nq_not, 5);
        assert_eq!(core.q_output, 6);
        assert_eq!(core.nq_output, 7);
        assert_eq!(core.feedback_scc, vec![2, 3, 4, 5]);
    }

    #[test]
    fn feedback_cores_reports_rs_latch_core() {
        let primitive = SequentialPrimitive::rs_latch();
        let cores = feedback_cores(&primitive.inner_graph);

        assert!(matches!(cores.as_slice(), [FeedbackCore::RsLatch(_)]));
    }

    #[test]
    fn recognize_rs_latch_core_inside_d_latch_decomposition() {
        let primitive = SequentialPrimitive::d_latch();
        let core = recognize_rs_latch_core(&primitive.inner_graph).unwrap();

        assert_eq!(core.set_input, 2);
        assert_eq!(core.reset_input, 4);
        assert_eq!(core.q_or, 5);
        assert_eq!(core.q_not, 6);
        assert_eq!(core.nq_or, 7);
        assert_eq!(core.nq_not, 8);
        assert_eq!(core.q_output, 9);
        assert_eq!(core.nq_output, 10);
        assert_eq!(core.feedback_scc, vec![5, 6, 7, 8]);
    }

    #[test]
    fn rs_latch_prefix_graph_extracts_d_latch_input_gating() -> eyre::Result<()> {
        let primitive = SequentialPrimitive::d_latch();
        let core = recognize_rs_latch_core(&primitive.inner_graph).unwrap();
        let prefix = rs_latch_prefix_graph(&primitive.inner_graph, &core).unwrap();
        let table = prefix.truth_table()?;

        assert_eq!(table.input_names, vec!["d".to_owned(), "en".to_owned()]);
        assert_eq!(table.output_tables["s"], vec![false, false, false, true]);
        assert_eq!(table.output_tables["r"], vec![false, false, true, false]);
        assert!(!prefix.has_cycle());
        Ok(())
    }
}
