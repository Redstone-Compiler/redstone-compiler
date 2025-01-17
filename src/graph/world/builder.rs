use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::{
    graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind},
    world::{
        block::{BlockKind, Direction},
        position::Position,
        world::{World, World3D},
    },
};

use super::WorldGraph;

#[derive(Debug, Clone)]
pub struct WorldGraphBuilder {
    world: World3D,
    queue: Vec<Position>,
    outputs: HashMap<Position, Vec<(Direction, Position)>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PropagateType {
    Soft,
    Hard,
    Torch,
    Repeater,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlaceBound(pub PropagateType, pub Position, pub Direction);

impl PlaceBound {
    pub fn propagation_type(&self) -> PropagateType {
        self.0
    }

    pub fn position(&self) -> Position {
        self.1
    }

    pub fn direction(&self) -> Direction {
        self.2
    }

    pub fn is_bound_on(&self, world: &World3D) -> bool {
        world.size.bound_on(self.position())
    }

    pub fn propagation_bound(&self, kind: &BlockKind, world: Option<&World3D>) -> Vec<PlaceBound> {
        let dir = self.direction();
        let pos = self.position();

        match kind {
            BlockKind::Switch { .. } => {
                let result = pos
                    .forwards_except(dir)
                    .into_iter()
                    .map(|pos_src| PlaceBound(PropagateType::Torch, pos_src, pos.diff(pos_src)))
                    .chain(|| -> Option<PlaceBound> {
                        let Some(pos) = pos.walk(dir) else {
                            return None;
                        };

                        Some(PlaceBound(PropagateType::Hard, pos, Direction::None))
                    }())
                    .collect_vec();

                result
            }
            BlockKind::Redstone { state, .. } => {
                let world = world.unwrap();
                let has_up_block = world[pos.up()].kind.is_cobble();

                let mut propagate_targets = Vec::new();
                propagate_targets.extend(pos.cardinal_redstone(*state));

                if !has_up_block {
                    propagate_targets.extend(
                        pos.up()
                            .cardinal_redstone(*state)
                            .into_iter()
                            .filter(|&up_cardinal| world[up_cardinal].kind.is_redstone()),
                    );
                }

                if let Some(down_pos) = pos.down() {
                    if !world[down_pos].kind.is_cobble() {
                        // Ensure redstone floors must have a block
                        unreachable!();
                    }

                    propagate_targets.push(down_pos);
                    propagate_targets.extend(
                        pos.cardinal_redstone(*state)
                            .into_iter()
                            .filter(|&pos| !world[pos].kind.is_cobble())
                            .filter_map(|pos| pos.walk(Direction::Bottom))
                            .filter(|&pos| world[pos].kind.is_redstone()),
                    );
                }

                propagate_targets
                    .iter()
                    .map(|pos_src| PlaceBound(PropagateType::Soft, *pos_src, pos_src.diff(pos)))
                    .collect_vec()
            }
            BlockKind::Torch { .. } => {
                let result = match dir {
                    Direction::Bottom => pos.cardinal(),
                    Direction::East | Direction::West | Direction::South | Direction::North => {
                        let mut positions = pos.cardinal_except(dir);
                        positions.extend(pos.down());
                        positions
                    }
                    _ => unreachable!(),
                }
                .into_iter()
                .map(|pos_src| PlaceBound(PropagateType::Torch, pos_src, pos_src.diff(pos)))
                .chain(Some(PlaceBound(
                    PropagateType::Hard,
                    pos.up(),
                    Direction::None,
                )))
                .collect_vec();

                result
            }
            BlockKind::Repeater { .. } => {
                let walk = pos.walk(dir.inverse());
                let mut result: Vec<PlaceBound> = Vec::new();

                if let Some(pos) = walk {
                    result.push(PlaceBound(PropagateType::Repeater, pos, dir));
                }

                if let Some(pos) = walk.map(|pos| pos.down()).flatten() {
                    result.push(PlaceBound(PropagateType::Soft, pos, Direction::Bottom));
                }

                result
            }
            BlockKind::RedstoneBlock => pos
                .forwards()
                .into_iter()
                .map(|pos_src| PlaceBound(PropagateType::Soft, pos_src, pos_src.diff(pos)))
                .collect_vec(),
            BlockKind::Piston { .. } => unimplemented!(),
            BlockKind::Air | BlockKind::Cobble { .. } => unreachable!(),
        }
    }

    pub fn propagate_to(&self, world: &World3D) -> Vec<(Direction, Position)> {
        let propagate_type = self.propagation_type();
        let pos = self.position();
        let dir = self.direction();
        let block = &world[pos];

        match block.kind {
            BlockKind::Air
            | BlockKind::Switch { .. }
            | BlockKind::RedstoneBlock
            | BlockKind::Torch { .. } => Vec::new(),
            // 코블에 붙어있는 레드스톤 토치, 리피터만 반응함
            BlockKind::Cobble { .. } => {
                if matches!(propagate_type, PropagateType::Torch) {
                    return Vec::new();
                }

                let mut cardinal_propagation = pos
                    .cardinal_except(dir)
                    .iter()
                    .filter(|&&pos| world.size.bound_on(pos))
                    .filter_map(|&pos_src| match world[pos_src].kind {
                        BlockKind::Torch { .. } => match propagate_type {
                            PropagateType::Torch => None,
                            PropagateType::Soft | PropagateType::Hard | PropagateType::Repeater => {
                                if pos_src.walk(world[pos_src].direction).unwrap() == pos {
                                    Some((Direction::None, pos_src))
                                } else {
                                    None
                                }
                            }
                        },
                        BlockKind::Repeater { .. } => match propagate_type {
                            PropagateType::Torch => None,
                            PropagateType::Soft | PropagateType::Hard | PropagateType::Repeater => {
                                if pos_src.walk(world[pos_src].direction).unwrap() == pos {
                                    Some((Direction::None, pos_src))
                                } else {
                                    None
                                }
                            }
                        },
                        BlockKind::Redstone { .. } => match propagate_type {
                            PropagateType::Soft | PropagateType::Torch => None,
                            PropagateType::Hard | PropagateType::Repeater => {
                                Some((Direction::None, pos_src))
                            }
                        },
                        _ => None,
                    })
                    .collect_vec();

                let up_pos = pos.up();
                if world.size.bound_on(up_pos) {
                    let up_block = &world[up_pos];
                    if (up_block.direction == Direction::Bottom
                        && matches!(up_block.kind, BlockKind::Torch { .. }))
                        || (!matches!(propagate_type, PropagateType::Soft)
                            && matches!(up_block.kind, BlockKind::Redstone { .. }))
                    {
                        cardinal_propagation.push((Direction::None, up_pos));
                    }
                }

                cardinal_propagation
            }
            BlockKind::Redstone { .. } => vec![(Direction::None, pos)],
            BlockKind::Repeater { .. } => {
                if block.direction == dir {
                    vec![(Direction::None, pos)]
                } else if block.direction.is_othogonal_plane(dir) {
                    // lock
                    match propagate_type {
                        PropagateType::Repeater => vec![(block.direction, pos)],
                        _ => vec![],
                    }
                } else {
                    vec![]
                }
            }
            BlockKind::Piston { .. } => todo!(),
        }
    }
}

impl WorldGraphBuilder {
    pub fn new(world: &World) -> Self {
        let queue = world
            .blocks
            .iter()
            .filter(|(_, block)| !block.kind.is_cobble())
            .map(|(pos, _)| pos.clone())
            .collect();
        let mut world: World3D = world.into();
        world.initialize_redstone_states();

        Self {
            world,
            queue,
            outputs: HashMap::default(),
        }
    }

    pub fn build(mut self) -> WorldGraph {
        self.visit_blocks();

        let (nodes, positions) = self.build_nodes();
        let mut graph = Graph {
            nodes,
            ..Default::default()
        };
        graph.build_inputs();

        graph.build_producers();
        graph.build_consumers();

        WorldGraph {
            graph,
            positions,
            ..Default::default()
        }
    }

    fn visit_blocks(&mut self) {
        let mut visit: HashSet<Position> = HashSet::new();

        // wiring blocks
        for pos in &self.queue {
            if visit.contains(pos) {
                continue;
            }
            visit.insert(*pos);

            let block = &self.world[*pos];
            let bound = PlaceBound(PropagateType::Soft, *pos, block.direction);
            let propagation_targets = bound.propagation_bound(&block.kind, Some(&self.world));
            let mut visits = propagation_targets
                .into_iter()
                .map(|bound| bound.propagate_to(&self.world))
                .flatten()
                .collect_vec();

            if block.kind.is_redstone() {
                // Redstone => Bottom Cobble => Up Redstone Propagation 때문에 재귀가 발생함
                visits = visits
                    .into_iter()
                    .filter(|(_, propagate_pos)| propagate_pos != pos)
                    .collect_vec();
            }

            if visits.len() > 0 {
                self.outputs.insert(*pos, visits);
            }
        }
    }

    fn build_nodes(&mut self) -> (Vec<GraphNode>, HashMap<GraphNodeId, Position>) {
        let mut graph_id: HashMap<Position, GraphNodeId> = HashMap::new();
        let mut nodes: HashMap<Position, GraphNode> = HashMap::new();

        for (index, pos) in self
            .queue
            .iter()
            .filter(|&&pos| !self.world[pos].kind.is_cobble())
            .enumerate()
        {
            graph_id.insert(*pos, index);
            nodes.insert(
                *pos,
                GraphNode::new(index, GraphNodeKind::Block(self.world[*pos])),
            );
        }

        for pos in self
            .queue
            .iter()
            .filter(|&&pos| !self.world[pos].kind.is_cobble())
        {
            let Some(outputs) = self.outputs.get(pos) else {
                continue;
            };

            let node = nodes.get_mut(pos).unwrap();
            let mut block = self.world[*pos];

            for (dir, pos) in outputs {
                let id = graph_id[&pos];

                if let BlockKind::Repeater {
                    lock_input1,
                    lock_input2,
                    ..
                } = &mut block.kind
                {
                    if block.direction.is_othogonal_plane(*dir) {
                        match (&lock_input1, &lock_input2) {
                            (None, _) => *lock_input1 = Some(id),
                            (_, None) => *lock_input2 = Some(id),
                            _ => panic!(),
                        }
                    }
                }

                node.outputs.push(id);
            }
        }

        let positions = nodes.iter().map(|(pos, node)| (node.id, *pos)).collect();
        let mut nodes = nodes.into_iter().map(|(_, node)| node).collect_vec();

        nodes.sort_by(|a, b| a.id.cmp(&b.id));

        (nodes, positions)
    }
}

impl<'a> From<&'a World> for WorldGraph {
    fn from(value: &'a World) -> Self {
        WorldGraphBuilder::new(value).build()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        graph::graphviz::ToGraphvizGraph,
        world::{
            block::{Block, BlockKind, Direction},
            position::{DimSize, Position},
            world::World,
        },
    };

    use super::WorldGraphBuilder;

    #[test]
    fn unittest_worldgraph_and_gate() {
        tracing_subscriber::fmt::init();

        let default_restone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Default::default(),
        };

        let default_cobble = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Default::default(),
        };

        let mock_world = World {
            size: DimSize(4, 6, 3),
            blocks: vec![
                (Position(0, 1, 0), default_restone.clone()),
                (Position(2, 1, 0), default_restone.clone()),
                (Position(0, 2, 0), default_cobble.clone()),
                (Position(1, 2, 0), default_cobble.clone()),
                (Position(2, 2, 0), default_cobble.clone()),
                (Position(1, 2, 1), default_restone.clone()),
                (Position(1, 4, 0), default_restone.clone()),
                (
                    Position(0, 2, 1),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::Bottom,
                    },
                ),
                (
                    Position(2, 2, 1),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::Bottom,
                    },
                ),
                (
                    Position(1, 3, 0),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::South,
                    },
                ),
                (
                    Position(0, 0, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: false },
                        direction: Direction::Bottom,
                    },
                ),
                (
                    Position(2, 0, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: false },
                        direction: Direction::Bottom,
                    },
                ),
            ],
        };

        let g = WorldGraphBuilder::new(&mock_world).build();
        println!("{}", g.to_graphviz());
    }

    #[test]
    fn unittest_worldgraph_cobble() {
        tracing_subscriber::fmt::init();

        let default_restone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Default::default(),
        };

        let default_cobble = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Default::default(),
        };

        let input_repeater = Block {
            kind: BlockKind::Repeater {
                delay: 1,
                is_on: false,
                is_locked: false,
                lock_input1: None,
                lock_input2: None,
            },
            direction: Direction::South,
        };

        let output_torch = Block {
            kind: BlockKind::Torch { is_on: false },
            direction: Direction::West,
        };

        let mock_world = World {
            size: DimSize(4, 4, 2),
            blocks: vec![
                (Position(1, 1, 0), default_cobble.clone()),
                (Position(1, 0, 0), input_repeater),
                (Position(0, 1, 0), default_restone.clone()),
                (Position(1, 2, 0), default_restone.clone()),
                (Position(2, 2, 0), default_restone.clone()),
                (Position(2, 1, 0), output_torch),
            ],
        };

        let _3d: crate::world::world::World3D = (&mock_world).into();
        println!("{:?}", _3d);
        let g = WorldGraphBuilder::new(&mock_world).build();
        println!("{}", g.to_graphviz());
    }

    #[test]
    fn unittest_worldgraph_recursive() {
        tracing_subscriber::fmt::init();

        let default_restone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Default::default(),
        };

        let default_cobble = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Default::default(),
        };

        let input_repeater = Block {
            kind: BlockKind::Repeater {
                delay: 1,
                is_on: false,
                is_locked: false,
                lock_input1: None,
                lock_input2: None,
            },
            direction: Direction::North,
        };

        let output_torch = Block {
            kind: BlockKind::Torch { is_on: false },
            direction: Direction::South,
        };

        let mock_world = World {
            size: DimSize(5, 5, 3),
            blocks: vec![
                (Position(1, 1, 0), default_cobble.clone()),
                (Position(1, 0, 1), default_cobble.clone()),
                (Position(1, 2, 1), default_cobble.clone()),
                (Position(1, 1, 1), input_repeater),
                (Position(1, 2, 0), output_torch),
                (Position(1, 0, 0), default_restone.clone()),
                (Position(1, 3, 0), default_restone.clone()),
            ],
        };

        let _3d: crate::world::world::World3D = (&mock_world).into();
        println!("{:?}", _3d);
        let g = WorldGraphBuilder::new(&mock_world).build();
        println!("{}", g.to_graphviz());
    }
}
