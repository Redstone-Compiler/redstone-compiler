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
        transform.fold_rs_latch_feedback_components();
        if !remain_routings {
            transform.remove_redstone();
            transform.remove_repeater();
        }
        transform.finish()
    }

    pub fn transform(mut self) -> eyre::Result<LogicGraph> {
        self.transform_inner()
    }

    pub fn positions(&self) -> &HashMap<GraphNodeId, Position> {
        &self.graph.positions
    }

    pub fn world_graph(&self) -> &WorldGraph {
        &self.graph
    }

    pub fn transform_preserving_node_ids(mut self) -> eyre::Result<LogicGraph> {
        self.transform_inner()
    }

    fn transform_inner(&mut self) -> eyre::Result<LogicGraph> {
        let mut graph = Graph::from_nodes_with_ids(
            self.graph
                .graph
                .nodes
                .iter()
                .map(|node| (node.id, GraphNode::default()))
                .collect(),
        );

        let mut input_count = 0;

        let mut source_replacements = HashMap::new();
        let node_ids = self
            .graph
            .graph
            .nodes
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();

        for id in node_ids {
            let node = self.graph.graph.find_node_by_id(id).unwrap().clone_node();

            let inputs = node.inputs.clone();
            let tag = transformed_tag(id, &node.tag);

            let new_node = match node.kind {
                GraphNodeKind::Sequential(sequential) => GraphNode {
                    kind: GraphNodeKind::Sequential(sequential),
                    inputs,
                    outputs: node.outputs,
                    tag,
                },
                GraphNodeKind::Block(block) => match block.kind {
                    BlockKind::Switch { .. } => {
                        input_count += 1;

                        GraphNode {
                            kind: GraphNodeKind::Input(format!("#{}", input_count)),
                            inputs,
                            outputs: node.outputs,
                            tag,
                        }
                    }
                    BlockKind::Redstone { .. } | BlockKind::Repeater { .. } => GraphNode {
                        kind: GraphNodeKind::Logic(Logic {
                            logic_type: LogicType::Or,
                        }),
                        inputs,
                        outputs: node.outputs,
                        tag,
                    },
                    BlockKind::Torch { .. } => {
                        if node.inputs.len() == 1 {
                            GraphNode {
                                kind: GraphNodeKind::Logic(Logic {
                                    logic_type: LogicType::Not,
                                }),
                                inputs,
                                outputs: node.outputs,
                                tag,
                            }
                        } else {
                            let not_node_id = graph.add_node(GraphNode {
                                kind: GraphNodeKind::Logic(Logic {
                                    logic_type: LogicType::Not,
                                }),
                                inputs: vec![id],
                                outputs: node.outputs.clone(),
                                tag: tag.clone(),
                            });

                            source_replacements.insert(id, not_node_id);

                            GraphNode {
                                kind: GraphNodeKind::Logic(Logic {
                                    logic_type: LogicType::Or,
                                }),
                                inputs,
                                outputs: vec![not_node_id],
                                tag: tag.clone(),
                            }
                        }
                    }
                    _ => todo!(),
                },
                _ => todo!(),
            };

            *graph.find_node_by_id_mut(id).unwrap() = new_node;
        }

        for mut node in graph.nodes.iter_mut() {
            if source_replacements
                .values()
                .any(|replacement| *replacement == node.id)
            {
                continue;
            }
            for input in &mut node.inputs {
                if let Some(replacement_id) = source_replacements.get(input) {
                    *input = *replacement_id;
                }
            }
        }

        graph.build_outputs();
        graph.build_producers();
        graph.build_consumers();

        Ok(LogicGraph { graph })
    }
}

fn transformed_tag(source_id: GraphNodeId, tag: &str) -> String {
    if tag.is_empty() {
        format!("From #{source_id}")
    } else {
        format!("From #{source_id}: {tag}")
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
    fn world_to_logic_extracts_rs_latch_nbt_as_sequential_node() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/rs-latch.nbt")?;
        let logic = crate::transform::place_and_route::utils::world_to_logic(&nbt.to_world())?;

        assert!(
            logic
                .nodes
                .iter()
                .any(|node| matches!(node.kind, crate::graph::GraphNodeKind::Sequential(_))),
            "expected RS latch NBT to contain a sequential logic node"
        );
        assert!(
            logic
                .nodes
                .iter()
                .any(|node| node.tag.contains("Folded RS latch feedback SCC")),
            "expected folded RS latch tag to be visible in the logic graph"
        );

        Ok(())
    }

    #[test]
    fn world_to_logic_extracts_d_latch_nbt_as_sequential_node() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/d-latch.nbt")?;
        let logic = crate::transform::place_and_route::utils::world_to_logic(&nbt.to_world())?;

        assert!(
            logic
                .nodes
                .iter()
                .any(|node| matches!(node.kind, crate::graph::GraphNodeKind::Sequential(_))),
            "expected D latch NBT to contain a sequential logic node"
        );

        Ok(())
    }

    #[test]
    fn counter_global_smoke_nbt_extracts_logic_graph_for_visualization() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/counter-global-smoke.nbt")?;
        let graph = WorldGraphBuilder::new(&nbt.to_world()).build();
        let logic = WorldToLogicTransformer::new(graph, true)?.transform()?;

        assert!(
            logic
                .graph
                .nodes
                .iter()
                .any(|node| matches!(node.kind, crate::graph::GraphNodeKind::Sequential(_))),
            "counter logic graph should expose folded sequential latch primitives"
        );

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

    #[test]
    fn world_to_logic_transform_keeps_sparse_ids_and_appends_synthetic_nodes() -> eyre::Result<()> {
        let switch = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::None,
        };
        let torch = Block {
            kind: BlockKind::Torch { is_on: true },
            direction: Direction::Bottom,
        };
        let mut graph = crate::graph::Graph::from_nodes_with_ids(vec![
            (
                10,
                crate::graph::GraphNode {
                    kind: crate::graph::GraphNodeKind::Block(switch),
                    outputs: vec![30],
                    ..Default::default()
                },
            ),
            (
                20,
                crate::graph::GraphNode {
                    kind: crate::graph::GraphNodeKind::Block(switch),
                    outputs: vec![30],
                    ..Default::default()
                },
            ),
            (
                30,
                crate::graph::GraphNode {
                    kind: crate::graph::GraphNodeKind::Block(torch),
                    inputs: vec![10, 20],
                    ..Default::default()
                },
            ),
        ]);
        graph.build_inputs();
        graph.build_outputs();
        graph.build_producers();
        graph.build_consumers();

        let world_graph = crate::graph::world::WorldGraph {
            graph,
            ..Default::default()
        };
        let logic = WorldToLogicTransformer::new(world_graph, true)?.transform()?;

        assert_eq!(
            logic.graph.nodes.iter().map(|node| node.id).collect_vec(),
            vec![10, 20, 30, 31]
        );
        assert_eq!(logic.graph.find_node_by_id(30).unwrap().outputs, vec![31]);
        assert_eq!(logic.graph.find_node_by_id(31).unwrap().inputs, vec![30]);

        Ok(())
    }
}
