use std::{collections::HashSet, iter::repeat_with};

use petgraph::stable_graph::NodeIndex;

use crate::{
    graph::{
        world::{
            builder::{PlaceBound, PropagateType},
            WorldGraph,
        },
        GraphNodeId, GraphNodeKind, SubGraph,
    },
    world::{block::Block, position::Position, world::World3D},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PlacedNode {
    position: Position,
    block: Block,
}

impl PlacedNode {
    pub fn new(position: Position, block: Block) -> Self {
        Self { position, block }
    }

    pub fn is_propagation_target(&self) -> bool {
        self.block.kind.is_stick_to_redstone() || self.block.kind.is_repeater()
    }

    // signal을 보낼 수 있는 부분들의 위치를 반환합니다.
    pub fn propagation_bound(&self, world: Option<&World3D>) -> HashSet<PlaceBound> {
        PlaceBound(PropagateType::Soft, self.position, self.block.direction)
            .propagation_bound(&self.block.kind, world)
            .into_iter()
            .collect()
    }
}

pub struct LocalPlacer<'a> {
    graph: SubGraph<'a>,
    try_count: usize,
    max_width_size: usize,
    max_height_size: usize,
}

pub const K_MAX_LOCAL_PLACE_NODE_COUNT: usize = 25;

impl<'a> LocalPlacer<'a> {
    // you should not pass side effected sub-graph
    pub fn new(
        graph: SubGraph<'a>,
        try_count: usize,
        max_width_size: Option<usize>,
        max_height_size: Option<usize>,
    ) -> eyre::Result<Self> {
        assert!(try_count > 0);

        Self::verify(&graph);

        Ok(Self {
            graph,
            try_count,
            max_width_size: max_width_size.unwrap_or(usize::MAX),
            max_height_size: max_height_size.unwrap_or(usize::MAX),
        })
    }

    pub fn verify(graph: &SubGraph<'a>) {
        assert!(graph.nodes.len() > 0);
        assert!(graph.nodes.len() <= K_MAX_LOCAL_PLACE_NODE_COUNT);

        for node_id in &graph.nodes {
            assert!(matches!(
                graph.graph.find_node_by_id(*node_id).unwrap().kind,
                GraphNodeKind::Logic { .. }
            ));
        }
    }

    pub fn generate(&mut self) -> WorldGraph {
        let try_count = self.try_count;

        // TODO: make parallel
        repeat_with(|| self.next_place())
            .take(try_count)
            .min_by_key(|(_, c)| *c)
            .unwrap()
            .0
    }

    fn next_place(&mut self) -> (WorldGraph, usize) {
        todo!()
    }
}

pub struct LocalPlacerCostEstimator<'a> {
    graph: &'a WorldGraph,
}

impl<'a> LocalPlacerCostEstimator<'a> {
    pub fn new(graph: &'a WorldGraph) -> Self {
        Self { graph }
    }

    pub fn cost(&self) -> usize {
        let _buffer_depth = self.graph.graph.critical_path().len();

        todo!()
    }
}
