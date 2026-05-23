use std::collections::HashMap;
use std::ops::Index;

use crate::graph::GraphNodeId;
use crate::world::position::Position;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum PlacementEndpoint {
    Node(GraphNodeId),
    Port(GraphNodeId, String),
}

#[derive(Debug, Clone, Default)]
pub(super) struct PlacementState {
    positions: HashMap<PlacementEndpoint, Position>,
}

impl PlacementState {
    pub(super) fn node_position(&self, node_id: GraphNodeId) -> Option<Position> {
        self.positions
            .get(&PlacementEndpoint::Node(node_id))
            .copied()
    }

    pub(super) fn node_positions(&self) -> impl Iterator<Item = Position> + '_ {
        self.positions.iter().filter_map(|(endpoint, position)| {
            matches!(endpoint, PlacementEndpoint::Node(_)).then_some(*position)
        })
    }

    #[allow(dead_code)]
    pub(super) fn port_position(&self, node_id: GraphNodeId, port: &str) -> Option<Position> {
        self.positions
            .get(&PlacementEndpoint::Port(node_id, port.to_owned()))
            .copied()
    }

    pub(super) fn set_node_position(&mut self, node_id: GraphNodeId, position: Position) {
        self.positions
            .insert(PlacementEndpoint::Node(node_id), position);
    }

    pub(super) fn set_port_position(
        &mut self,
        node_id: GraphNodeId,
        port: String,
        position: Position,
    ) {
        self.positions
            .insert(PlacementEndpoint::Port(node_id, port), position);
    }
}

impl FromIterator<(GraphNodeId, Position)> for PlacementState {
    fn from_iter<T: IntoIterator<Item = (GraphNodeId, Position)>>(iter: T) -> Self {
        let mut state = PlacementState::default();
        for (node_id, position) in iter {
            state.set_node_position(node_id, position);
        }
        state
    }
}

impl Index<&GraphNodeId> for PlacementState {
    type Output = Position;

    fn index(&self, index: &GraphNodeId) -> &Self::Output {
        &self.positions[&PlacementEndpoint::Node(*index)]
    }
}
