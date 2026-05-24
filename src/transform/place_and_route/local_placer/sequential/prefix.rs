use std::collections::HashMap;

use super::*;
use crate::sequential::core::rs_latch_prefix_graph;

pub(super) struct RsLatchPrefixPlan {
    graph: LogicGraph,
    input_mappings: Vec<(GraphNodeId, GraphNodeId)>,
    input_node_ids_by_port: HashMap<String, GraphNodeId>,
    input_positions_by_port: HashMap<String, Position>,
    set_source: GraphNodeId,
    reset_source: GraphNodeId,
    rs_input_nodes: RsLatchInputNodeIds,
}

impl RsLatchPrefixPlan {
    pub(super) fn build(
        node: &GraphNode,
        sequential: &SequentialPrimitive,
        state: &PlacementState,
    ) -> Option<Self> {
        let core = sequential.rs_latch_core()?;
        let graph = rs_latch_prefix_graph(&sequential.inner_graph, &core)
            .and_then(|graph| graph.prepare_place().ok())?;
        let set_source = output_source_node_id(&graph, "s")?;
        let reset_source = output_source_node_id(&graph, "r")?;

        let mut input_mappings = Vec::new();
        let mut input_node_ids_by_port = HashMap::new();
        let mut input_positions_by_port = HashMap::new();
        for prefix_input in graph.nodes.iter().filter_map(|node| match &node.kind {
            GraphNodeKind::Input(name) => Some((node.id, name.as_str())),
            _ => None,
        }) {
            let outer_input = outer_input_node_id(node, sequential, prefix_input.1)?;
            let position = state.node_position(outer_input)?;
            input_mappings.push((prefix_input.0, outer_input));
            input_node_ids_by_port.insert(prefix_input.1.to_owned(), outer_input);
            input_positions_by_port.insert(prefix_input.1.to_owned(), position);
        }

        Some(Self {
            graph,
            input_mappings,
            input_node_ids_by_port,
            input_positions_by_port,
            set_source,
            reset_source,
            rs_input_nodes: rs_latch_input_node_ids(node.id),
        })
    }

    pub(super) fn rs_node_inputs(&self) -> Vec<GraphNodeId> {
        vec![self.rs_input_nodes.set, self.rs_input_nodes.reset]
    }

    pub(super) fn input_position(&self, port: &str) -> Option<Position> {
        self.input_positions_by_port.get(port).copied()
    }

    pub(super) fn d_latch_input_node_ids(&self) -> Option<(GraphNodeId, GraphNodeId)> {
        Some((
            self.input_node_ids_by_port.get("d").copied()?,
            self.input_node_ids_by_port.get("en").copied()?,
        ))
    }

    pub(super) fn prefix_output_positions(
        &self,
        state: &PlacementState,
    ) -> Option<(Position, Position)> {
        Some((
            state.node_position(self.set_source)?,
            state.node_position(self.reset_source)?,
        ))
    }

    pub(super) fn publish_core_inputs(
        &self,
        state: &mut PlacementState,
        set_position: Position,
        reset_position: Position,
    ) {
        state.set_node_position(self.rs_input_nodes.set, set_position);
        state.set_node_position(self.rs_input_nodes.reset, reset_position);
    }

    pub(super) fn place(
        &self,
        config: &LocalPlacerConfig,
        world: &World3D,
        outer_state: &PlacementState,
    ) -> PlacerQueue {
        let mut prefix_state = PlacementState::default();
        for &(prefix_input, outer_input) in &self.input_mappings {
            let Some(position) = outer_state.node_position(outer_input) else {
                return Vec::new();
            };
            prefix_state.set_node_position(prefix_input, position);
        }

        let Ok(prefix_placer) = LocalPlacer::new(self.graph.clone(), *config) else {
            return Vec::new();
        };
        prefix_placer.generate_queue_from(vec![(world.clone(), prefix_state)], None, None)
    }

    pub(super) fn set_source(&self) -> GraphNodeId {
        self.set_source
    }

    pub(super) fn reset_source(&self) -> GraphNodeId {
        self.reset_source
    }
}

fn outer_input_node_id(
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    port: &str,
) -> Option<GraphNodeId> {
    node.inputs
        .iter()
        .zip(&sequential.input_ports)
        .find_map(|(input_node, input_port)| (input_port == port).then_some(*input_node))
}

fn output_source_node_id(graph: &LogicGraph, output_name: &str) -> Option<GraphNodeId> {
    graph.nodes.iter().find_map(|node| {
        (matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name)
            && node.inputs.len() == 1)
            .then(|| node.inputs[0])
    })
}
