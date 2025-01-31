use crate::graph::logic::LogicGraph;
use crate::graph::world::WorldGraph;
use crate::transform::world_to_logic::WorldToLogicTransformer;
use crate::world::{World, World3D};

pub fn world3d_to_logic(world3d: &World3D) -> eyre::Result<LogicGraph> {
    world_to_logic(&World::from(world3d))
}

pub fn world_to_logic(world: &World) -> eyre::Result<LogicGraph> {
    let world_graph = WorldGraph::from(world);
    let logic_graph = WorldToLogicTransformer::new(world_graph)?.transform()?;
    logic_graph.verify()?;
    Ok(logic_graph)
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
