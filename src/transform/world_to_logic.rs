use std::collections::HashMap;

use super::world::WorldGraphTransformer;
use crate::graph::logic::LogicGraph;
use crate::graph::world::WorldGraph;
use crate::graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::{Logic, LogicType};
use crate::world::block::BlockKind;
use crate::world::position::Position;

// for testing Layout vs. Schematic
#[derive(Default)]
pub struct WorldToLogicTransformer {
    graph: WorldGraph,
}

impl WorldToLogicTransformer {
    /// Before transform, `WorldToLogicTransformer` runs some optimizations
    /// such as fold_redstone, remove_redstone, remove_repeater.
    /// However, sometimes routing information such as redstone and repeater
    /// may be needed (e.g. isomorphism), and for this, `remain_routings` is provided.
    pub fn new(graph: WorldGraph, remain_routings: bool) -> eyre::Result<Self> {
        Self::verify_input(&graph)?;

        Ok(Self {
            graph: Self::optimize(graph, remain_routings),
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

    fn optimize(graph: WorldGraph, remain_routings: bool) -> WorldGraph {
        let mut transform = WorldGraphTransformer::new(graph);
        transform.fold_redstone();
        if !remain_routings {
            transform.remove_redstone();
            transform.remove_repeater();
        }
        transform.finish()
    }

    pub fn transform(mut self) -> eyre::Result<LogicGraph> {
        self.transform_inner(true)
    }

    pub fn positions(&self) -> &HashMap<GraphNodeId, Position> {
        &self.graph.positions
    }

    pub fn world_graph(&self) -> &WorldGraph {
        &self.graph
    }

    pub fn transform_preserving_node_ids(mut self) -> eyre::Result<LogicGraph> {
        self.transform_inner(false)
    }

    fn transform_inner(&mut self, rebuild_node_ids: bool) -> eyre::Result<LogicGraph> {
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

        let mut graph = Graph {
            nodes,
            ..Default::default()
        };

        if rebuild_node_ids {
            graph = graph.rebuild_node_ids();
        } else {
            graph.nodes.sort_by_key(|node| node.id);
            graph.build_outputs();
            graph.build_producers();
            graph.build_consumers();
        }

        Ok(LogicGraph { graph })
    }
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;

    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::graph::logic::predefined_logics;
    use crate::graph::subgraphs_to_clustered_graph;
    use crate::graph::world::WorldGraphBuilder;
    use crate::nbt::NBTRoot;
    use crate::transform::logic::LogicGraphTransformer;
    use crate::transform::place_and_route::utils::equivalent_logic_with_world;
    use crate::transform::world_to_logic::WorldToLogicTransformer;
    use crate::world::block::{Block, BlockKind, Direction};
    use crate::world::position::{DimSize, Position};
    use crate::world::World;

    #[test]
    fn unittest_world_to_logic_graph() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/alu.nbt")?;

        let g = WorldGraphBuilder::new(&nbt.to_world()).build();
        g.graph.verify()?;

        let g = WorldToLogicTransformer::new(g, false)?.transform()?;
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
        let nbt = NBTRoot::load("test/xor-generated.nbt")?;
        let world = &nbt.to_world();
        let expected = predefined_logics::buffered_xor_graph()?;
        assert!(equivalent_logic_with_world(&expected, world)?);

        Ok(())
    }

    #[test]
    fn unittest_world_to_logic_graph_half_adder() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/half-adder-generated.nbt")?;
        let world = &nbt.to_world();
        let expected = predefined_logics::buffered_half_adder_graph()?;
        assert!(equivalent_logic_with_world(&expected, world)?);

        Ok(())
    }

    #[test]
    fn world_to_logic_preserves_switch_and_inverted_output_short() -> eyre::Result<()> {
        let redstone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
        let world = World {
            size: DimSize(4, 3, 2),
            blocks: vec![
                (
                    Position(0, 1, 1),
                    Block {
                        kind: BlockKind::Switch { is_on: false },
                        direction: Direction::East,
                    },
                ),
                (
                    Position(1, 1, 1),
                    Block {
                        kind: BlockKind::Cobble {
                            on_count: 0,
                            on_base_count: 0,
                        },
                        direction: Direction::None,
                    },
                ),
                (
                    Position(2, 1, 1),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::West,
                    },
                ),
                (Position(1, 1, 0), redstone),
                (Position(2, 1, 0), redstone),
            ],
        };

        let graph = WorldGraphBuilder::new(&world).build();
        let logic = WorldToLogicTransformer::new(graph, true)?.transform()?;
        let table = logic.truth_table()?;

        assert!(
            table.output_table_set().contains(&vec![true, true]),
            "shorted switch and inverted output should extract as a constant-high output, got {:?}",
            table
        );

        Ok(())
    }
}
