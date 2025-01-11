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
        GraphNode, GraphNodeId, GraphNodeKind,
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

    pub fn new_cobble(position: Position) -> Self {
        Self {
            position,
            block: Block {
                kind: BlockKind::Cobble {
                    on_count: 0,
                    on_base_count: 0,
                },
                direction: Direction::None,
            },
        }
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
        if self.block.kind.is_cobble() {
            return !(world[self.position].kind.is_air() || world[self.position].kind.is_cobble());
        }

        if !world[self.position].kind.is_air() {
            return true;
        }

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

    pub fn generate(&mut self, finish_step: Option<usize>) -> Vec<World3D> {
        let orders = self.graph.topological_order();
        let mut queue: VecDeque<(World3D, HashMap<GraphNodeId, Position>)> = VecDeque::new();
        queue.push_back((World3D::new(DimSize(10, 10, 5)), Default::default()));

        let mut step = 0;
        while step < orders.len() && Some(step) != finish_step {
            let node_id = orders[step];
            let node = self.graph.find_node_by_id(node_id).unwrap();
            println!("current node: {node:?}");
            let prev_step_volume = queue.len();
            let mut next_queue = VecDeque::new();
            while let Some((world, pos)) = queue.pop_front() {
                for (world, place_position) in self.place_and_route_next_node(node, &world, &pos) {
                    let mut nodes_position = pos.clone();
                    nodes_position.insert(node_id, place_position);
                    next_queue.push_back((world, nodes_position));
                }
            }
            step += 1;
            queue = next_queue;
            println!("step - {step}: {prev_step_volume} -> {}", queue.len());
        }

        queue.into_iter().map(|(world, _)| world).collect()
    }

    fn place_and_route_next_node(
        &self,
        node: &GraphNode,
        world: &World3D,
        positions: &HashMap<GraphNodeId, Position>,
    ) -> Vec<(World3D, Position)> {
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

fn place_node(world: &mut World3D, node: PlacedNode) {
    world[node.position] = node.block;
    if node.block.kind.is_redstone() {
        world.update_redstone_states(node.position);
    }
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
                let placed_node = PlacedNode { position, block };
                if placed_node.has_conflict(world) {
                    return None;
                }

                let mut new_world = world.clone();
                place_node(&mut new_world, placed_node);
                Some((new_world, position))
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
    match kind {
        BlockKind::Redstone { .. } => generate_torch_place_and_routes(world, start, kind),
        _ => unimplemented!(),
    }
}

fn generate_torch_place_and_routes(
    world: &World3D,
    start: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    let torch_strategy = [
        Direction::Bottom,
        Direction::East,
        Direction::West,
        Direction::South,
        Direction::North,
    ]
    .into_iter()
    .map(|direction| Block { kind, direction })
    .collect_vec();

    // start에서 최소 두 칸 떨어진 곳에 위치시킨다.
    let generate_strategy = torch_strategy
        .into_iter()
        .cartesian_product(
            iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
                .map(|(x, y, z)| Position(x, y, z))
                .filter(|pos| start.manhattan_distance(pos) > 2),
        )
        .collect_vec();

    println!("strategy count: {}", generate_strategy.len());

    let mut candidates = Vec::new();
    for (block, position) in generate_strategy {
        let Some(cobble_position) = position.walk(block.direction) else {
            continue;
        };
        let cobble_node = PlacedNode::new_cobble(cobble_position);
        if !world.size.bound_on(cobble_position) || cobble_node.has_conflict(&world) {
            continue;
        }

        let torch_node = PlacedNode::new(position, block);
        if torch_node.has_conflict(world) {
            continue;
        }

        let mut new_world = world.clone();
        if new_world[cobble_position].kind.is_air() {
            place_node(&mut new_world, cobble_node);
        }
        place_node(&mut new_world, torch_node);
        candidates.push((new_world, position));
    }

    candidates
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
        println!("{}", logic_graph.to_graphviz());

        let mut placer = LocalPlacer::new(logic_graph)?;
        let world3d = placer.generate(Some(2));

        let nbt: NBTRoot = world3d[4].to_nbt();
        nbt.save("test/and-gate-new.nbt");

        Ok(())
    }
}
