use std::collections::{HashMap, HashSet};

use disjoint_set::DisjointSet;
use itertools::Itertools;

use crate::graph::world::WorldGraph;
use crate::graph::{GraphNode, GraphNodeId, GraphNodeKind};
use crate::sequential::SequentialPrimitive;
use crate::world::block::{Block, BlockKind, Direction};

pub struct WorldGraphTransformer {
    pub graph: WorldGraph,
}

impl WorldGraphTransformer {
    pub fn new(graph: WorldGraph) -> Self {
        Self { graph }
    }

    pub fn finish(self) -> WorldGraph {
        self.graph
    }

    pub fn fold_redstone(&mut self) {
        let mut nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. })))
            .collect_vec();

        nodes.sort_by(|a, b| a.id.cmp(&b.id));

        // clustering
        let mut cluster = DisjointSet::new();

        for node in &nodes {
            cluster.make_set(node.id);
        }

        for node in &nodes {
            let producers = &self.graph.graph.producers[&node.id];
            let consumers = &self.graph.graph.consumers[&node.id];

            for candidate in producers.iter().chain(consumers) {
                let Some(candidate_node) = self.graph.graph.find_node_by_id(*candidate) else {
                    unreachable!();
                };

                if matches!(&candidate_node.kind, GraphNodeKind::Block(block) if matches!(block.kind, BlockKind::Redstone { .. }))
                {
                    cluster.union(node.id, *candidate).unwrap();
                }
            }
        }

        let ids: HashSet<GraphNodeId> = nodes.iter().map(|node| node.id).collect();
        let mut group_inputs: HashMap<usize, HashSet<GraphNodeId>> = HashMap::new();
        let mut group_outputs: HashMap<usize, HashSet<GraphNodeId>> = HashMap::new();
        let mut group_members: HashMap<usize, Vec<GraphNodeId>> = HashMap::new();

        let mut group_ids: HashSet<usize> = HashSet::new();

        for id in &ids {
            // make clustered id group
            let group_id = cluster.find(*id).unwrap();
            group_ids.insert(group_id);
            group_members.entry(group_id).or_default().push(*id);

            // collect group input outputs
            group_inputs.entry(group_id).or_default().extend(
                self.graph.graph.producers[id]
                    .clone()
                    .into_iter()
                    .filter(|id| !ids.contains(id)),
            );
            group_outputs.entry(group_id).or_default().extend(
                self.graph.graph.consumers[id]
                    .clone()
                    .into_iter()
                    .filter(|id| !ids.contains(id)),
            );
        }

        for members in group_members.values_mut() {
            members.sort_unstable();
        }

        // remove redstone node
        for id in &ids {
            self.graph.graph.remove_by_node_id_lazy(*id);
        }

        for group_id in group_ids.iter().sorted() {
            let node = GraphNode {
                kind: GraphNodeKind::Block(Block {
                    kind: BlockKind::Redstone {
                        on_count: 0,
                        state: 0,
                        strength: 0,
                    },
                    direction: Direction::None,
                }),
                // TODO: optimize this
                inputs: group_inputs[group_id].clone().into_iter().collect_vec(),
                outputs: group_outputs[group_id].clone().into_iter().collect_vec(),
                tag: format!(
                    "Folded redstone component {:?}",
                    group_members.get(group_id).unwrap()
                ),
                ..Default::default()
            };
            let node_id = self.graph.graph.add_node(node);

            group_inputs[group_id].iter().for_each(|&conn| {
                self.graph
                    .graph
                    .find_node_by_id_mut(conn)
                    .unwrap()
                    .outputs
                    .push(node_id);
            });
            group_outputs[group_id].iter().for_each(|&conn| {
                self.graph
                    .graph
                    .find_node_by_id_mut(conn)
                    .unwrap()
                    .inputs
                    .push(node_id);
            });

            self.graph.routings.insert(node_id);
        }

        self.graph.graph.build_inputs();
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
    }

    // Collapse physical RS-latch feedback loops before converting the world graph
    // into logic. The downstream logic extraction walks nodes in topological order,
    // so a cyclic torch/redstone latch core must become one sequential node first.
    pub fn fold_rs_latch_feedback_components(&mut self) {
        loop {
            let mut folded = false;
            for component in self.graph.graph.strongly_connected_components() {
                let component_set = component.iter().copied().collect::<HashSet<_>>();
                let (fold_nodes, require_outputs, tag) =
                    if is_rs_latch_feedback_component(&self.graph, &component, &component_set) {
                        (
                            component_set,
                            false,
                            format!("Folded RS latch feedback SCC {component:?}"),
                        )
                    } else if let Some(core) = find_embedded_rs_latch_feedback_core(
                        &self.graph,
                        &component,
                        &component_set,
                    ) {
                        let mut folded_nodes = core.iter().copied().collect_vec();
                        folded_nodes.sort_unstable();
                        (
                            core,
                            true,
                            format!("Folded RS latch feedback core {folded_nodes:?}"),
                        )
                    } else {
                        continue;
                    };

                let inputs = self.graph.graph.external_edges(&fold_nodes, true);
                if inputs.len() != 2 {
                    continue;
                }
                let outputs = self.graph.graph.external_edges(&fold_nodes, false);
                if require_outputs && outputs.is_empty() {
                    continue;
                }
                self.graph.replace_nodes_with(
                    &fold_nodes,
                    GraphNodeKind::Sequential(SequentialPrimitive::rs_latch()),
                    inputs,
                    outputs,
                    tag,
                );
                folded = true;
                break;
            }
            if !folded {
                break;
            }
        }
    }

    pub fn remove_redstone(&mut self) {
        self.remove_specific_kind_of_block(
            |kind| matches!(&kind, BlockKind::Redstone { .. }),
            true,
        );
    }

    pub fn remove_repeater(&mut self) {
        self.remove_specific_kind_of_block(
            |kind| matches!(&kind, BlockKind::Repeater { .. }),
            true,
        );
    }

    fn remove_specific_kind_of_block<F>(&mut self, filter: F, except_output: bool)
    where
        F: Fn(&BlockKind) -> bool,
    {
        let nodes = self
            .graph
            .graph
            .nodes
            .iter()
            .filter(|node| !except_output || !node.outputs.is_empty())
            .filter(|node| matches!(&node.kind, GraphNodeKind::Block(block) if filter(&block.kind)))
            .map(|node| node.id)
            .collect_vec();

        for node in nodes {
            self.graph.graph.remove_and_reconnect_by_node_id_lazy(node);
        }

        self.graph.graph.build_inputs();
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
    }
}

// Recognize the narrow post-redstone-fold shape used by the current RS latch
// fixtures: two torches cross-coupled through two folded redstone routing nodes.
// Input gating around that core, such as in a D latch, stays outside the SCC and
// remains ordinary combinational logic.
fn is_rs_latch_feedback_component(
    graph: &WorldGraph,
    component: &[GraphNodeId],
    component_set: &HashSet<GraphNodeId>,
) -> bool {
    if component.len() != 4 {
        return false;
    }

    let mut torch_count = 0;
    let mut redstone_count = 0;
    for node_id in component {
        let Some(node) = graph.graph.find_node_by_id(*node_id) else {
            return false;
        };
        match &node.kind {
            GraphNodeKind::Block(block) if block.kind.is_torch() => {
                torch_count += 1;
                if !node.inputs.iter().any(|id| component_set.contains(id))
                    || !node.outputs.iter().any(|id| component_set.contains(id))
                {
                    return false;
                }
            }
            GraphNodeKind::Block(block) if block.kind.is_redstone() => {
                redstone_count += 1;
            }
            _ => return false,
        }
    }

    torch_count == 2 && redstone_count == 2
}

fn find_embedded_rs_latch_feedback_core(
    graph: &WorldGraph,
    component: &[GraphNodeId],
    component_set: &HashSet<GraphNodeId>,
) -> Option<HashSet<GraphNodeId>> {
    let torches = component
        .iter()
        .copied()
        .filter(|node_id| {
            graph.graph.find_node_by_id(*node_id).is_some_and(
                |node| matches!(&node.kind, GraphNodeKind::Block(block) if block.kind.is_torch()),
            )
        })
        .sorted()
        .collect_vec();

    for (index, left) in torches.iter().enumerate() {
        for right in torches.iter().skip(index + 1) {
            let Some(left_redstone) =
                feedback_redstone_between_torches(graph, *left, *right, component_set)
            else {
                continue;
            };
            let Some(right_redstone) =
                feedback_redstone_between_torches(graph, *right, *left, component_set)
            else {
                continue;
            };
            if left_redstone == right_redstone {
                continue;
            }

            return Some(HashSet::from([
                *left,
                *right,
                left_redstone,
                right_redstone,
            ]));
        }
    }

    None
}

fn feedback_redstone_between_torches(
    graph: &WorldGraph,
    torch_id: GraphNodeId,
    target_torch_id: GraphNodeId,
    component_set: &HashSet<GraphNodeId>,
) -> Option<GraphNodeId> {
    let torch = graph.graph.find_node_by_id(torch_id)?;
    torch
        .outputs
        .iter()
        .copied()
        .filter(|node_id| component_set.contains(node_id))
        .filter(|node_id| {
            graph
                .graph
                .find_node_by_id(*node_id)
                .is_some_and(|node| matches!(&node.kind, GraphNodeKind::Block(block) if block.kind.is_redstone()))
        })
        .find(|node_id| {
            graph
                .graph
                .find_node_by_id(*node_id)
                .is_some_and(|node| node.outputs.contains(&target_torch_id))
        })
}

#[cfg(test)]
mod tests {
    use super::WorldGraphTransformer;
    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::graph::world::WorldGraphBuilder;
    use crate::nbt::NBTRoot;

    #[test]
    fn unittest_fold_redstone() -> eyre::Result<()> {
        let nbt = NBTRoot::load("test/alu.nbt")?;
        let g = WorldGraphBuilder::new(&nbt.to_world()).build();

        let mut transform = WorldGraphTransformer::new(g);
        transform.fold_redstone();
        let folded = transform.finish();

        assert!(
            folded
                .graph
                .nodes
                .iter()
                .any(|node| node.tag.contains("Folded redstone component")),
            "expected folded redstone nodes to keep their source component tag"
        );

        let mut transform = WorldGraphTransformer::new(folded);
        transform.remove_redstone();
        transform.remove_repeater();
        if std::env::var_os("PRINT_WORLD_GRAPHS").is_some() {
            println!("{}", transform.finish().to_graphviz());
        }

        Ok(())
    }

    #[test]
    fn fold_rs_latch_feedback_components_finds_latch_cores_inside_counter_sccs() -> eyre::Result<()>
    {
        let nbt = NBTRoot::load("test/counter-global-smoke.nbt")?;
        let g = WorldGraphBuilder::new(&nbt.to_world()).build();
        let mut transform = WorldGraphTransformer::new(g);

        transform.fold_redstone();
        transform.fold_rs_latch_feedback_components();
        let folded = transform.finish();

        let sequential_nodes = folded
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, crate::graph::GraphNodeKind::Sequential(_)))
            .count();
        assert!(
            sequential_nodes >= 2,
            "expected counter SCCs to expose embedded RS latch cores, got {sequential_nodes}"
        );

        Ok(())
    }
}
