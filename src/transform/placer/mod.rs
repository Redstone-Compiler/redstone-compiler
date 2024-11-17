use std::{collections::HashSet, iter::repeat_with};

use eyre::ensure;

use crate::{
    graph::{
        logic::LogicGraph,
        world::{
            builder::{PlaceBound, PropagateType},
            WorldGraph,
        },
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

    pub fn has_conflict_with(&self, other: &Self) -> bool {
        todo!()
    }

    // pub fn
}

pub struct LocalPlacer {
    graph: LogicGraph,
}

pub const K_MAX_LOCAL_PLACE_NODE_COUNT: usize = 25;

impl LocalPlacer {
    pub fn new(graph: LogicGraph) -> eyre::Result<Self> {
        let result = Self { graph };
        result.verify()?;
        Ok(result)
    }

    fn verify(&self) -> eyre::Result<()> {
        ensure!(self.graph.nodes.len() > 0, "");
        ensure!(
            self.graph.nodes.len() <= K_MAX_LOCAL_PLACE_NODE_COUNT,
            "too large graph"
        );

        for node_id in &self.graph.nodes {
            let kind = &self.graph.find_node_by_id(node_id.id).unwrap().kind;
            ensure!(
                kind.is_input() || kind.is_output() || kind.is_logic(),
                "cannot place"
            );
            if let Some(logic) = kind.as_logic() {
                ensure!(logic.is_not() || logic.is_or(), "cannot place");
            }
        }

        Ok(())
    }

    pub fn generate(&mut self) -> World3D {
        repeat_with(|| self.next_place())
            .min_by_key(|(_, c)| *c)
            .unwrap()
            .0
    }

    fn next_place(&mut self) -> (World3D, usize) {
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

#[cfg(test)]
mod tests {

    use crate::{
        graph::{
            graphviz::ToGraphvizGraph,
            logic::{builder::LogicGraphBuilder, LogicGraph},
        },
        nbt::{NBTRoot, ToNBT},
        transform::placer::LocalPlacer,
    };

    fn build_graph_from_stmt(stmt: &str, output: &str) -> eyre::Result<LogicGraph> {
        LogicGraphBuilder::new(stmt.to_string()).build(output.to_string())
    }

    #[test]
    fn test_generate_component_and() -> eyre::Result<()> {
        let logic_graph = build_graph_from_stmt("a&b", "c")?.prepare_place()?;
        dbg!(&logic_graph);
        println!("{}", logic_graph.to_graphviz());

        let mut placer = LocalPlacer::new(logic_graph)?;
        let world3d = placer.generate();

        let nbt: NBTRoot = world3d.to_nbt();
        nbt.save("test/and-gate-new.nbt");

        Ok(())
    }
}
