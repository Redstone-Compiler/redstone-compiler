use std::collections::HashSet;

use crate::graph::{Graph, GraphNodeId, GraphNodeKind};
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
    ensure_input(graph, set_input, "s")?;
    ensure_input(graph, reset_input, "r")?;

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

fn ensure_input(graph: &Graph, node_id: GraphNodeId, input_name: &str) -> Option<()> {
    let node = graph.find_node_by_id(node_id)?;
    matches!(&node.kind, GraphNodeKind::Input(name) if name == input_name).then_some(())
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
}
