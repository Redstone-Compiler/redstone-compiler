use super::*;
use crate::graph::analysis::equivalent_expression_groups;
use crate::graph::logic::predefined_logics;
use crate::graph::world::WorldGraphBuilder;
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::logic::{Logic, LogicType};
use crate::nbt::NBTRoot;
use crate::sequential::{SequentialPrimitive, SequentialType};
use crate::transform::world_to_logic::WorldToLogicTransformer;

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
fn compose_high_level_gates_recovers_and_from_prepared_graph() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a&b", "out")?.prepare_place()?;
    let expected_table = graph.truth_table()?;
    let mut transform = LogicGraphTransformer::new(graph);

    transform.compose_high_level_gates()?;

    assert_eq!(count_logic_type(&transform.graph, LogicType::And), 1);
    assert_eq!(transform.graph.truth_table()?, expected_table);
    transform.graph.graph.verify()?;

    Ok(())
}

#[test]
fn compose_high_level_gates_recovers_xor_from_prepared_graph() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a^b", "out")?.prepare_place()?;
    let expected_table = graph.truth_table()?;
    let mut transform = LogicGraphTransformer::new(graph);

    transform.compose_high_level_gates()?;

    assert_eq!(count_logic_type(&transform.graph, LogicType::Xor), 1);
    assert_eq!(transform.graph.truth_table()?, expected_table);
    transform.graph.graph.verify()?;

    Ok(())
}

#[test]
fn compose_high_level_gates_handles_world_logic_graph() -> eyre::Result<()> {
    let nbt = NBTRoot::load("test/alu.nbt")?;
    let raw_world_graph = WorldGraphBuilder::new(&nbt.to_world()).build();
    let transformer = WorldToLogicTransformer::new(raw_world_graph, true)?;
    let logic_graph = transformer.transform()?;
    let mut transform = LogicGraphTransformer::new(logic_graph);

    transform.remove_double_neg_expression();
    transform.optimize_cse()?;
    transform.fold_or_chains()?;
    transform.optimize_cse()?;
    transform.compose_high_level_gates()?;
    transform.optimize_cse()?;

    assert!(count_logic_type(&transform.graph, LogicType::And) > 0);
    transform.graph.graph.verify()?;

    Ok(())
}

#[test]
fn compose_high_level_gates_recovers_xor_from_world_xor_nbt() -> eyre::Result<()> {
    let nbt = NBTRoot::load("test/xor-generated.nbt")?;
    let raw_world_graph = WorldGraphBuilder::new(&nbt.to_world()).build();
    let transformer = WorldToLogicTransformer::new(raw_world_graph, true)?;
    let logic_graph = transformer.transform()?;
    let mut transform = LogicGraphTransformer::new(logic_graph);

    transform.remove_double_neg_expression();
    transform.optimize_cse()?;
    transform.fold_or_chains()?;
    transform.optimize_cse()?;
    transform.compose_high_level_gates()?;
    transform.optimize_cse()?;

    assert_eq!(count_logic_type(&transform.graph, LogicType::Xor), 1);
    transform.graph.graph.verify()?;

    Ok(())
}

#[test]
fn compose_high_level_gates_recovers_xor_from_world_half_adder_nbt() -> eyre::Result<()> {
    let nbt = NBTRoot::load("test/half-adder-generated.nbt")?;
    let raw_world_graph = WorldGraphBuilder::new(&nbt.to_world()).build();
    let transformer = WorldToLogicTransformer::new(raw_world_graph, true)?;
    let logic_graph = transformer.transform()?;
    let mut transform = LogicGraphTransformer::new(logic_graph);

    transform.remove_double_neg_expression();
    transform.optimize_cse()?;
    transform.fold_or_chains()?;
    transform.optimize_cse()?;
    transform.compose_high_level_gates()?;
    transform.optimize_cse()?;

    assert!(count_logic_type(&transform.graph, LogicType::Xor) > 0);
    assert_logic_nodes_have_input_count(&transform.graph, LogicType::Xor, 2);
    transform.graph.graph.verify()?;

    Ok(())
}

#[test]
fn compose_high_level_gates_recovers_xor_from_world_full_adder_nbt() -> eyre::Result<()> {
    let nbt = NBTRoot::load("test/full-adder.nbt")?;
    let raw_world_graph = WorldGraphBuilder::new(&nbt.to_world()).build();
    let transformer = WorldToLogicTransformer::new(raw_world_graph, true)?;
    let logic_graph = transformer.transform()?;
    let mut transform = LogicGraphTransformer::new(logic_graph);

    transform.remove_double_neg_expression();
    transform.optimize_cse()?;
    transform.fold_or_chains()?;
    transform.optimize_cse()?;
    transform.compose_high_level_gates()?;
    transform.optimize_cse()?;

    assert!(count_logic_type(&transform.graph, LogicType::Xor) > 0);
    assert_logic_nodes_have_input_count(&transform.graph, LogicType::Xor, 2);
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

fn count_logic_type(graph: &LogicGraph, logic_type: LogicType) -> usize {
    graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == logic_type)
            })
            .count()
}

fn assert_logic_nodes_have_input_count(
    graph: &LogicGraph,
    logic_type: LogicType,
    expected_input_count: usize,
) {
    for node in graph.nodes.iter().filter(
        |node| matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == logic_type),
    ) {
        assert_eq!(
            node.inputs.len(),
            expected_input_count,
            "{logic_type:?} node {} has wrong input count",
            node.id
        );
    }
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
