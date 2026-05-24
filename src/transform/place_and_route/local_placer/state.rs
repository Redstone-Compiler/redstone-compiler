use std::collections::{HashMap, HashSet};
use std::ops::Index;

use crate::graph::GraphNodeId;
use crate::world::position::Position;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum PlacementEndpoint {
    Node(GraphNodeId),
    Port(GraphNodeId, String),
}

#[derive(Debug, Clone, Default)]
pub(super) struct PlacementState {
    positions: HashMap<PlacementEndpoint, Position>,
    signal_footprints: HashMap<GraphNodeId, HashSet<Position>>,
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

    pub(super) fn retain_nodes(&mut self, node_ids: &HashSet<GraphNodeId>) {
        self.positions.retain(|endpoint, _| match endpoint {
            PlacementEndpoint::Node(node_id) | PlacementEndpoint::Port(node_id, _) => {
                node_ids.contains(node_id)
            }
        });
        self.signal_footprints
            .retain(|node_id, _| node_ids.contains(node_id));
    }

    pub(super) fn endpoint_positions(&self) -> Vec<(PlacementEndpoint, Position)> {
        let mut positions = self
            .positions
            .iter()
            .map(|(endpoint, position)| (endpoint.clone(), *position))
            .collect::<Vec<_>>();
        positions.sort();
        positions
    }

    pub(super) fn signal_footprints(&self) -> Vec<(GraphNodeId, Vec<Position>)> {
        let mut footprints = self
            .signal_footprints
            .iter()
            .map(|(node_id, positions)| {
                let mut positions = positions.iter().copied().collect::<Vec<_>>();
                positions.sort();
                (*node_id, positions)
            })
            .collect::<Vec<_>>();
        footprints.sort();
        footprints
    }

    pub(super) fn signal_positions_for_nodes(
        &self,
        node_ids: &HashSet<GraphNodeId>,
    ) -> HashSet<Position> {
        self.signal_footprints
            .iter()
            .filter(|(node_id, _)| node_ids.contains(node_id))
            .flat_map(|(_, positions)| positions.iter().copied())
            .collect()
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

    pub(super) fn set_signal_footprint(
        &mut self,
        node_id: GraphNodeId,
        positions: impl IntoIterator<Item = Position>,
    ) {
        self.signal_footprints
            .insert(node_id, positions.into_iter().collect());
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
