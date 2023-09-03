use std::collections::HashMap;

use crate::{
    graph::{logic::LogicGraph, world::WorldGraph, Graph, GraphNode, GraphNodeKind},
    logic::{Logic, LogicType},
    world::block::BlockKind,
};

use super::world::WorldGraphTransformer;

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
            let GraphNodeKind::Block(block) = node.kind else  {
              return false;
            };

            let BlockKind::Repeater { lock_input1, lock_input2 , ..} = block.kind else {
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

        let mut node_id = self.graph.graph.max_node_id().unwrap();
        let mut input_count = 0;

        for id in self.graph.graph.topological_order() {
            let node = self.graph.graph.find_and_remove_node_by_id(id).unwrap();

            let new_nodes = match node.inputs.len() {
                0 => {
                    input_count += 1;

                    vec![GraphNode {
                        id: node.id,
                        kind: GraphNodeKind::Input(format!("#{}", input_count)),
                        inputs: vec![],
                        outputs: node.outputs,
                        tag: format!("From #{id}"),
                        ..Default::default()
                    }]
                }
                1 => vec![GraphNode {
                    id: node.id,
                    kind: GraphNodeKind::Logic(Logic {
                        logic_type: LogicType::Not,
                    }),
                    inputs: node
                        .inputs
                        .iter()
                        .map(|id| new_in_id.get(id).unwrap_or(id))
                        .copied()
                        .collect(),
                    outputs: node.outputs,
                    tag: format!("From #{id}"),
                    ..Default::default()
                }],
                _ => {
                    node_id += 1;

                    let or_node = GraphNode {
                        id: node.id,
                        kind: GraphNodeKind::Logic(Logic {
                            logic_type: LogicType::Or,
                        }),
                        inputs: node
                            .inputs
                            .iter()
                            .filter_map(|id| new_in_id.get(id))
                            .copied()
                            .collect(),
                        outputs: vec![node_id],
                        tag: format!("From #{id}"),
                        ..Default::default()
                    };

                    let not_node = GraphNode {
                        id: node_id,
                        kind: GraphNodeKind::Logic(Logic {
                            logic_type: LogicType::Not,
                        }),
                        inputs: vec![node.id],
                        outputs: node.outputs.clone(),
                        tag: format!("From #{id}"),
                        ..Default::default()
                    };

                    new_in_id.insert(node.id, node_id);

                    vec![or_node, not_node]
                }
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
    use crate::{
        graph::{graphviz::ToGraphvizGraph, world::builder::WorldGraphBuilder},
        nbt::NBTRoot,
        transform::{logic::LogicGraphTransformer, world_to_logic::WorldToLogicTransformer},
    };

    #[test]
    fn unittest_world_to_logic_graph() -> eyre::Result<()> {
        let nbt = NBTRoot::load(&"test/alu.nbt".into())?;

        let g = WorldGraphBuilder::new(&nbt.to_world()).build();
        g.graph.verify()?;

        let g = WorldToLogicTransformer::new(g)?.transform()?;
        g.graph.verify()?;

        let mut transform = LogicGraphTransformer::new(g);
        transform.remove_double_neg_expression();

        println!("{}", transform.finish().to_graphviz());

        Ok(())
    }
}
