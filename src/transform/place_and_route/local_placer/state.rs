use std::collections::{HashMap, HashSet};
use std::ops::Index;

use crate::graph::GraphNodeId;
use crate::world::position::Position;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum SignalExpr {
    Source(Position),
    Not(Box<SignalExpr>),
    Or(Vec<SignalExpr>),
}

impl SignalExpr {
    pub(super) fn source(source: Position) -> Self {
        Self::Source(source)
    }

    pub(super) fn invert(self) -> Self {
        Self::Not(Box::new(self))
    }

    pub(super) fn or(expressions: impl IntoIterator<Item = SignalExpr>) -> Self {
        let mut expressions = expressions.into_iter().collect::<Vec<_>>();
        expressions.sort();
        expressions.dedup();
        match expressions.as_slice() {
            [] => Self::Or(Vec::new()),
            [expression] => expression.clone(),
            _ => Self::Or(expressions),
        }
    }

    pub(super) fn sources(&self) -> HashSet<Position> {
        let mut sources = HashSet::new();
        self.collect_sources(&mut sources);
        sources
    }

    fn collect_sources(&self, sources: &mut HashSet<Position>) {
        match self {
            Self::Source(source) => {
                sources.insert(*source);
            }
            Self::Not(expression) => expression.collect_sources(sources),
            Self::Or(expressions) => {
                for expression in expressions {
                    expression.collect_sources(sources);
                }
            }
        }
    }

    pub(super) fn eval(&self, source_states: &HashMap<Position, bool>) -> bool {
        match self {
            Self::Source(source) => source_states.get(source).copied().unwrap_or(false),
            Self::Not(expression) => !expression.eval(source_states),
            Self::Or(expressions) => expressions
                .iter()
                .any(|expression| expression.eval(source_states)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum PlacementEndpoint {
    Node(GraphNodeId),
    Port(GraphNodeId, String),
}

#[derive(Debug, Clone, Default)]
pub(super) struct PlacementState {
    positions: HashMap<PlacementEndpoint, Position>,
    signal_nets: HashMap<GraphNodeId, SignalNet>,
}

#[derive(Debug, Clone)]
pub(super) struct SignalNet {
    pub(super) footprint: HashSet<Position>,
    pub(super) sources: HashSet<Position>,
    pub(super) expression: SignalExpr,
    pub(super) position_sources: HashMap<Position, HashSet<Position>>,
    pub(super) position_drivers: HashMap<Position, HashSet<GraphNodeId>>,
}

impl Default for SignalNet {
    fn default() -> Self {
        Self {
            footprint: HashSet::new(),
            sources: HashSet::new(),
            expression: SignalExpr::Or(Vec::new()),
            position_sources: HashMap::new(),
            position_drivers: HashMap::new(),
        }
    }
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
        self.signal_nets
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
            .signal_nets
            .iter()
            .map(|(node_id, net)| {
                let mut positions = net.footprint.iter().copied().collect::<Vec<_>>();
                positions.sort();
                (*node_id, positions)
            })
            .collect::<Vec<_>>();
        footprints.sort();
        footprints
    }

    #[cfg(test)]
    pub(super) fn signal_positions_for_nodes(
        &self,
        node_ids: &HashSet<GraphNodeId>,
    ) -> HashSet<Position> {
        self.signal_nets
            .iter()
            .filter(|(node_id, _)| node_ids.contains(node_id))
            .flat_map(|(_, net)| net.footprint.iter().copied())
            .collect()
    }

    pub(super) fn signal_nets(&self) -> Vec<(GraphNodeId, SignalNet)> {
        let mut nets = self
            .signal_nets
            .iter()
            .map(|(node_id, net)| (*node_id, net.clone()))
            .collect::<Vec<_>>();
        nets.sort_by_key(|(node_id, _)| *node_id);
        nets
    }

    pub(super) fn signal_sources_for_nodes(
        &self,
        node_ids: &HashSet<GraphNodeId>,
    ) -> HashSet<Position> {
        self.signal_nets
            .iter()
            .filter(|(node_id, _)| node_ids.contains(node_id))
            .flat_map(|(_, net)| net.sources.iter().copied())
            .collect()
    }

    pub(super) fn signal_expression_for_nodes(
        &self,
        node_ids: &HashSet<GraphNodeId>,
    ) -> SignalExpr {
        SignalExpr::or(
            self.signal_nets
                .iter()
                .filter(|(node_id, _)| node_ids.contains(node_id))
                .map(|(_, net)| net.expression.clone()),
        )
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

    pub(super) fn set_signal_net(
        &mut self,
        node_id: GraphNodeId,
        footprint: impl IntoIterator<Item = Position>,
        sources: impl IntoIterator<Item = Position>,
    ) {
        let footprint = footprint.into_iter().collect::<HashSet<_>>();
        let sources = sources.into_iter().collect::<HashSet<_>>();
        let expression = SignalExpr::or(sources.iter().copied().map(SignalExpr::source));
        let position_sources = footprint
            .iter()
            .copied()
            .map(|position| (position, sources.clone()))
            .collect();
        let position_drivers = footprint
            .iter()
            .copied()
            .map(|position| (position, [node_id].into_iter().collect()))
            .collect();
        self.signal_nets.insert(
            node_id,
            SignalNet {
                footprint,
                sources,
                expression,
                position_sources,
                position_drivers,
            },
        );
    }

    pub(super) fn set_signal_net_with_expression(
        &mut self,
        node_id: GraphNodeId,
        footprint: impl IntoIterator<Item = Position>,
        expression: SignalExpr,
    ) {
        let footprint = footprint.into_iter().collect::<HashSet<_>>();
        let sources = expression.sources();
        let position_sources = footprint
            .iter()
            .copied()
            .map(|position| (position, sources.clone()))
            .collect();
        let position_drivers = footprint
            .iter()
            .copied()
            .map(|position| (position, [node_id].into_iter().collect()))
            .collect();
        self.signal_nets.insert(
            node_id,
            SignalNet {
                footprint,
                sources,
                expression,
                position_sources,
                position_drivers,
            },
        );
    }

    pub(super) fn set_signal_net_with_position_sources_and_contacts(
        &mut self,
        node_id: GraphNodeId,
        expression: SignalExpr,
        footprint_segments: impl IntoIterator<Item = (Position, HashSet<Position>)>,
        contact_segments: impl IntoIterator<Item = (Position, HashSet<Position>)>,
    ) {
        let mut net = SignalNet {
            expression,
            ..Default::default()
        };
        for (position, sources) in footprint_segments {
            net.footprint.insert(position);
            net.sources.extend(sources.iter().copied());
            net.position_sources.insert(position, sources);
            net.position_drivers
                .entry(position)
                .or_default()
                .insert(node_id);
        }
        for (position, sources) in contact_segments {
            net.sources.extend(sources.iter().copied());
            net.position_sources.insert(position, sources);
            net.position_drivers
                .entry(position)
                .or_default()
                .insert(node_id);
        }
        self.signal_nets.insert(node_id, net);
    }
}

impl FromIterator<(GraphNodeId, Position)> for PlacementState {
    fn from_iter<T: IntoIterator<Item = (GraphNodeId, Position)>>(iter: T) -> Self {
        let mut state = PlacementState::default();
        for (node_id, position) in iter {
            state.set_node_position(node_id, position);
            state.set_signal_net(node_id, [position], [position]);
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
