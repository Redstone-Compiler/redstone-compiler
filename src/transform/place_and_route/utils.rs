use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::graph::logic::LogicGraph;
use crate::graph::world::WorldGraph;
use crate::graph::{GraphNodeId, GraphNodeKind};
use crate::output::OutputMetadata;
use crate::transform::logic::LogicGraphTransformer;
use crate::transform::world_to_logic::WorldToLogicTransformer;
use crate::world::block::BlockKind;
use crate::world::{World, World3D};

pub fn world3d_to_logic(world3d: &World3D) -> eyre::Result<LogicGraph> {
    world_to_logic(&World::from(world3d))
}

pub fn world_to_logic(world: &World) -> eyre::Result<LogicGraph> {
    let world_graph = WorldGraph::from(world);
    let logic_graph = WorldToLogicTransformer::new(world_graph, true)?.transform()?;
    normalize_logic_graph(logic_graph)
}

pub fn world_to_logic_with_outputs(
    world: &World,
    metadata: &OutputMetadata,
) -> eyre::Result<LogicGraph> {
    let raw_world_graph = WorldGraph::from(world);
    let raw_position_to_node = raw_world_graph
        .positions
        .iter()
        .map(|(node_id, position)| (*position, *node_id))
        .collect::<std::collections::HashMap<_, _>>();
    let transformer = WorldToLogicTransformer::new(raw_world_graph.clone(), true)?;
    let folded_graph = transformer.world_graph().clone();
    let logic_graph = transformer.transform_preserving_node_ids()?;
    let outputs = metadata
        .outputs
        .iter()
        .map(|endpoint| {
            let position = endpoint.position();
            let Some(raw_source_id) = raw_position_to_node.get(&position).copied() else {
                eyre::bail!(
                    "missing world graph node for output {} at {:?}",
                    endpoint.name,
                    position
                );
            };
            let source_id =
                resolve_folded_output_source(raw_source_id, &raw_world_graph, &folded_graph)?;
            let source_id = if logic_graph.find_node_by_id(source_id).is_some() {
                source_id
            } else if let Some(node) = logic_graph
                .nodes
                .iter()
                .filter(|node| node.tag == format!("From #{source_id}"))
                .max_by_key(|node| node.id)
            {
                node.id
            } else {
                eyre::bail!(
                    "missing logic graph node {} for output {} at {:?}",
                    source_id,
                    endpoint.name,
                    position
                );
            };
            Ok((endpoint.name.clone(), source_id))
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    normalize_logic_graph(logic_graph.attach_outputs(outputs)?)
}

fn resolve_folded_output_source(
    raw_source_id: GraphNodeId,
    raw_graph: &WorldGraph,
    folded_graph: &WorldGraph,
) -> eyre::Result<GraphNodeId> {
    if folded_graph.graph.find_node_by_id(raw_source_id).is_some() {
        return Ok(raw_source_id);
    }

    let Some(raw_node) = raw_graph.graph.find_node_by_id(raw_source_id) else {
        eyre::bail!("missing raw output source node {raw_source_id}");
    };
    if !matches!(&raw_node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. }))
    {
        return Ok(raw_source_id);
    }

    let component = collect_redstone_component(raw_source_id, raw_graph);
    let component_inputs = component_external_edges(raw_graph, &component, true);
    let component_outputs = component_external_edges(raw_graph, &component, false);

    folded_graph
        .graph
        .nodes
        .iter()
        .find(|node| {
            matches!(&node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. }))
                && node.inputs.iter().copied().collect::<std::collections::HashSet<_>>()
                    == component_inputs
                && node.outputs.iter().copied().collect::<std::collections::HashSet<_>>()
                    == component_outputs
        })
        .map(|node| node.id)
        .ok_or_else(|| eyre::eyre!("missing folded redstone source for raw node {raw_source_id}"))
}

fn collect_redstone_component(
    root: GraphNodeId,
    graph: &WorldGraph,
) -> std::collections::HashSet<GraphNodeId> {
    let mut component = std::collections::HashSet::new();
    let mut stack = vec![root];

    while let Some(node_id) = stack.pop() {
        if !component.insert(node_id) {
            continue;
        }

        let neighbors = graph
            .graph
            .producers
            .get(&node_id)
            .into_iter()
            .flatten()
            .chain(graph.graph.consumers.get(&node_id).into_iter().flatten());
        for neighbor in neighbors {
            if graph.graph.find_node_by_id(*neighbor).is_some_and(|node| {
                matches!(&node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. }))
            }) {
                stack.push(*neighbor);
            }
        }
    }

    component
}

fn component_external_edges(
    graph: &WorldGraph,
    component: &std::collections::HashSet<GraphNodeId>,
    producers: bool,
) -> std::collections::HashSet<GraphNodeId> {
    component
        .iter()
        .flat_map(|node_id| {
            if producers {
                graph.graph.producers.get(node_id)
            } else {
                graph.graph.consumers.get(node_id)
            }
            .into_iter()
            .flatten()
            .copied()
        })
        .filter(|node_id| !component.contains(node_id))
        .collect()
}

pub fn equivalent_graph(src: &LogicGraph, tar: &LogicGraph) -> bool {
    petgraph::algo::is_isomorphic_matching(
        &src.to_petgraph(),
        &tar.to_petgraph(),
        |x, y| {
            if x.is_input() && y.is_input() {
                return true;
            }

            x == y
        },
        |_, _| true,
    )
}

pub fn equivalent_logic_with_world3d(
    expected: &LogicGraph,
    generated: &World3D,
) -> eyre::Result<bool> {
    let expected = outputs_removed(expected);
    let generated = world3d_to_logic(generated)?;
    Ok(equivalent_graph(&expected, &generated))
}

pub fn equivalent_logic_with_world3ds(
    expected: &LogicGraph,
    generates: &Vec<World3D>,
) -> eyre::Result<bool> {
    let expected = outputs_removed(expected);
    let checks = generates
        .par_iter()
        .map(|generated| {
            let logic = world3d_to_logic(generated)?;
            Ok(equivalent_graph(&expected, &logic))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(checks.into_iter().all(|x| x))
}

pub fn contains_truth_table_with_world3ds(
    expected: &LogicGraph,
    generated: &[World3D],
) -> eyre::Result<bool> {
    let expected = expected.truth_table()?;
    let checks = generated
        .par_iter()
        .map(|generated| {
            let generated = world3d_to_logic(generated)?.truth_table()?;
            Ok(generated.contains_output_tables_under_input_permutation(&expected))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(checks.into_iter().all(|x| x))
}

pub fn equivalent_logic_with_world(expected: &LogicGraph, generated: &World) -> eyre::Result<bool> {
    let expected = outputs_removed(expected);
    let generated = world_to_logic(generated)?;
    Ok(equivalent_graph(&expected, &generated))
}

fn outputs_removed(graph: &LogicGraph) -> LogicGraph {
    let mut graph = graph.clone();
    graph
        .outputs()
        .into_iter()
        .for_each(|o| graph.graph.remove_by_node_id_lazy(o));
    graph.graph.build_outputs();
    graph.graph.build_producers();
    graph
}

fn normalize_logic_graph(graph: LogicGraph) -> eyre::Result<LogicGraph> {
    let mut transform = LogicGraphTransformer::new(graph);
    transform.remove_double_neg_expression();
    transform.optimize_cse()?;
    let graph = transform.finish();
    graph.verify()?;
    Ok(graph)
}
