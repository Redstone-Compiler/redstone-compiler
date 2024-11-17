use std::collections::{HashMap, HashSet, VecDeque};

use eyre::ensure;
use itertools::{iproduct, Itertools};

use crate::{
    graph::{
        logic::LogicGraph,
        world::{
            builder::{PlaceBound, PropagateType},
            WorldGraph,
        },
        GraphNodeId, GraphNodeKind,
    },
    logic::LogicType,
    world::{
        block::{Block, BlockKind, Direction, RedstoneState},
        position::{DimSize, Position},
        world::World3D,
    },
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

    pub fn has_conflict(&self, world: &World3D) -> bool {
        let bounds = self.propagation_bound(Some(world));
        bounds
            .into_iter()
            .filter(|bound| bound.is_bound_on(world))
            .any(|bound| !bound.propagate_to(world).is_empty())
    }
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

    pub fn generate(&mut self) -> Vec<World3D> {
        let orders = self.graph.topological_order();
        let mut queue: VecDeque<(usize, World3D, HashMap<GraphNodeId, Position>)> = VecDeque::new();
        queue.push_back((0, World3D::new(DimSize(10, 10, 10)), Default::default()));

        while let Some((order_index, world, pos)) = queue.pop_front() {
            if order_index == orders.len() {
                break;
            }

            let node_id = orders[order_index];
            for (world, place_position) in self.place_and_route_next_node(node_id, &world, &pos) {
                let mut nodes_position = pos.clone();
                nodes_position.insert(node_id, place_position);
                queue.push_back((order_index + 1, world, nodes_position));
            }
        }

        queue.into_iter().map(|(_, world, _)| world).collect()
    }

    fn place_and_route_next_node(
        &self,
        node_id: GraphNodeId,
        world: &World3D,
        positions: &HashMap<GraphNodeId, Position>,
    ) -> Vec<(World3D, Position)> {
        let node = self.graph.find_node_by_id(node_id).unwrap();

        match node.kind {
            GraphNodeKind::Input(_) => input_node_kind()
                .into_iter()
                .flat_map(|kind| generate_inputs(world, kind))
                .collect(),
            GraphNodeKind::Output(_) => output_node_kind()
                .into_iter()
                .flat_map(|kind| generate_place_and_routes(world, positions[&node.inputs[0]], kind))
                .collect(),
            GraphNodeKind::Logic(logic) => match logic.logic_type {
                LogicType::Not => not_node_kind()
                    .into_iter()
                    .flat_map(|kind| {
                        generate_place_and_routes(world, positions[&node.inputs[0]], kind)
                    })
                    .collect(),
                LogicType::Or => {
                    assert_eq!(node.inputs.len(), 2);
                    generate_routes(
                        world,
                        positions[&node.inputs[0]],
                        positions[&node.inputs[1]],
                    )
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }
}

fn input_node_kind() -> Vec<BlockKind> {
    vec![
        BlockKind::Switch { is_on: true },
        // BlockKind::Cobble {
        //     on_count: 0,
        //     on_base_count: 0,
        // },
        // BlockKind::Redstone {
        //     on_count: 0,
        //     state: RedstoneState::None as usize,
        //     strength: 0,
        // },
        // BlockKind::RedstoneBlock,
    ]
}

fn output_node_kind() -> Vec<BlockKind> {
    vec![BlockKind::Redstone {
        on_count: 0,
        state: RedstoneState::None as usize,
        strength: 0,
    }]
}

fn not_node_kind() -> Vec<BlockKind> {
    vec![BlockKind::Torch { is_on: false }]
}

fn place_new_node(world: &World3D, node: PlacedNode) -> World3D {
    let mut world = world.clone();
    world[node.position] = node.block;
    if node.block.kind.is_redstone() {
        world.update_redstone_states(node.position);
    }
    world
}

fn generate_inputs(world: &World3D, kind: BlockKind) -> Vec<(World3D, Position)> {
    // 일단 바닥에 두고 생성하기
    let block = Block {
        kind,
        direction: Direction::Bottom,
    };

    let generate_strategy = vec![
        // x == 0에서 먼저 생성
        (block, iproduct!(0..1, 0..world.size.1, 0..world.size.2)),
        // x == 0에서 못 찾으면 y == 0 에서 생성
        (block, iproduct!(0..world.size.0, 0..1, 0..world.size.2)),
    ];

    for (block, positions) in generate_strategy {
        let candidates = positions
            .filter_map(|(x, y, z)| {
                let position = Position(x, y, z);
                if !world[position].kind.is_air() {
                    return None;
                }

                let placed_node = PlacedNode { position, block };
                if placed_node.has_conflict(world) {
                    return None;
                }

                let world = place_new_node(world, placed_node);
                Some((world, position))
            })
            .collect_vec();

        if !candidates.is_empty() {
            return candidates;
        }
    }

    Vec::default()
}

fn generate_routes(world: &World3D, first: Position, second: Position) -> Vec<(World3D, Position)> {
    todo!()
}

fn generate_place_and_routes(
    world: &World3D,
    start: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    todo!()
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

        let nbt: NBTRoot = world3d[0].to_nbt();
        nbt.save("test/and-gate-new.nbt");

        Ok(())
    }
}
