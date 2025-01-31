use std::collections::HashMap;

use super::world::WorldGraphTransformer;
use crate::graph::logic::LogicGraph;
use crate::graph::world::WorldGraph;
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::logic::{Logic, LogicType};
use crate::world::block::BlockKind;

// for testing Layout vs. Schematic
#[derive(Default)]
pub struct WorldToLogicTransformer {
    graph: WorldGraph,
}

impl WorldToLogicTransformer {
    pub fn new(graph: WorldGraph) -> eyre::Result<Self> {
        Self::verify_input(&graph)?;

        Ok(Self {
            graph: Self::optimize(graph),
        })
    }

    fn verify_input(graph: &WorldGraph) -> eyre::Result<()> {
        // verify no repeater lock
        let contains_lock_repeater = graph.graph.nodes.iter().any(|node| {
            let GraphNodeKind::Block(block) = node.kind else {
                return false;
            };

            let BlockKind::Repeater {
                lock_input1,
                lock_input2,
                ..
            } = block.kind
            else {
                return false;
            };

            lock_input1.is_some() || lock_input2.is_some()
        });

        if contains_lock_repeater {
            eyre::bail!("Cannot world to logic graph if world graph contains lock repeater block!");
        }

        Ok(())
    }

    fn optimize(graph: WorldGraph) -> WorldGraph {
        let mut transform = WorldGraphTransformer::new(graph);
        transform.fold_redstone();
        transform.remove_redstone();
        transform.remove_repeater();
        transform.finish()
    }

    pub fn transform(mut self) -> eyre::Result<LogicGraph> {
        let mut new_in_id = HashMap::new();
        let mut nodes = Vec::new();

        let mut next_new_id = self.graph.graph.max_node_id().unwrap();
        let mut input_count = 0;

        for id in self.graph.graph.topological_order() {
            let node = self.graph.graph.find_and_remove_node_by_id(id).unwrap();

            let inputs = node
                .inputs
                .iter()
                .map(|id| new_in_id.get(id).unwrap_or(id))
                .copied()
                .collect();

            let new_nodes = match node.kind.as_block().unwrap().kind {
                BlockKind::Switch { .. } => {
                    input_count += 1;

                    vec![GraphNode {
                        id: node.id,
                        kind: GraphNodeKind::Input(format!("#{}", input_count)),
                        inputs,
                        outputs: node.outputs,
                        tag: format!("From #{id}"),
                    }]
                }
                BlockKind::Redstone { .. } | BlockKind::Repeater { .. } => {
                    vec![GraphNode {
                        id: node.id,
                        kind: GraphNodeKind::Logic(Logic {
                            logic_type: LogicType::Or,
                        }),
                        inputs,
                        outputs: node.outputs,
                        tag: format!("From #{id}"),
                    }]
                }
                BlockKind::Torch { .. } => {
                    if node.inputs.len() == 1 {
                        vec![GraphNode {
                            id: node.id,
                            kind: GraphNodeKind::Logic(Logic {
                                logic_type: LogicType::Not,
                            }),
                            inputs,
                            outputs: node.outputs,
                            tag: format!("From #{id}"),
                        }]
                    } else {
                        next_new_id += 1;

                        let or_node = GraphNode {
                            id: node.id,
                            kind: GraphNodeKind::Logic(Logic {
                                logic_type: LogicType::Or,
                            }),
                            inputs,
                            outputs: vec![next_new_id],
                            tag: format!("From #{id}"),
                        };

                        let not_node = GraphNode {
                            id: next_new_id,
                            kind: GraphNodeKind::Logic(Logic {
                                logic_type: LogicType::Not,
                            }),
                            inputs: vec![or_node.id],
                            outputs: node.outputs.clone(),
                            tag: format!("From #{id}"),
                        };

                        new_in_id.insert(or_node.id, next_new_id);

                        vec![or_node, not_node]
                    }
                }
                _ => todo!(),
            };

            nodes.extend(new_nodes);
        }

        let graph = Graph {
            nodes,
            ..Default::default()
        }
        .rebuild_node_ids();

        Ok(LogicGraph { graph })
    }
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;

    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::graph::logic::LogicGraph;
    use crate::graph::subgraphs_to_clustered_graph;
    use crate::graph::world::WorldGraphBuilder;
    use crate::nbt::NBTRoot;
    use crate::transform::logic::LogicGraphTransformer;
    use crate::transform::world_to_logic::WorldToLogicTransformer;

    #[test]
    fn unittest_world_to_logic_graph() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/alu.nbt")?;

        let g = WorldGraphBuilder::new(&nbt.to_world()).build();
        g.graph.verify()?;

        let g = WorldToLogicTransformer::new(g)?.transform()?;
        g.graph.verify()?;

        let mut transform = LogicGraphTransformer::new(g);
        transform.remove_double_neg_expression();
        let sub_graphs = transform
            .cluster(true)
            .iter()
            .map(|x| x.to_subgraph())
            .collect_vec();
        let g = transform.finish();

        println!("{}", g.to_graphviz_with_clusters(&sub_graphs));

        let clustered_g = subgraphs_to_clustered_graph(&g.graph, &sub_graphs);
        println!("{}", clustered_g.to_graphviz());

        Ok(())
    }

    #[test]
    fn unittest_world_to_logic_graph_xor() -> eyre::Result<()> {
        fn buffered_xor_graph() -> eyre::Result<LogicGraph> {
            // c := (~((a&b)|~a))|(~((a&b)|~b))
            let logic_graph1 = LogicGraph::from_stmt("a&b", "c")?;
            let logic_graph2 = LogicGraph::from_stmt("(~(c|~a))|(~(c|~b))", "d")?;

            let mut fm = logic_graph1.clone();
            fm.graph.merge(logic_graph2.graph);
            fm.prepare_place()
        }

        let nbt = NBTRoot::load("test/xor-generated.nbt")?;

        let g = WorldGraphBuilder::new(&nbt.to_world()).build();
        g.graph.verify()?;

        let g = WorldToLogicTransformer::new(g)?.transform()?;
        g.graph.verify()?;

        println!("{}", g.to_graphviz());

        let mut expected = buffered_xor_graph()?;
        let outputs = expected.outputs();
        outputs
            .into_iter()
            .for_each(|o| expected.graph.remove_by_node_id_lazy(o));
        expected.graph.build_outputs();
        expected.graph.build_producers();

        println!("{}", expected.to_graphviz());

        let equivalent = petgraph::algo::is_isomorphic(&expected.to_petgraph(), &g.to_petgraph());
        assert!(equivalent);

        Ok(())
    }
}
