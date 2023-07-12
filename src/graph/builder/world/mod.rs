use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::{
    graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind},
    world::{
        block::{Block, BlockKind, Direction, RedstoneState},
        position::Position,
        world::{World, World3D},
    },
};

#[derive(Debug, Clone)]
pub struct WorldGraph {
    pub graph: Graph,
    pub positions: HashMap<GraphNodeId, Position>,
}

#[derive(Debug, Clone)]
pub struct WorldGraphBuilder {
    world: World3D,
    queue: Vec<Position>,
    outputs: HashMap<Position, Vec<(Direction, Position)>>,
}

#[derive(Clone, Debug)]
enum PropagateType {
    Soft,
    Hard,
    Torch,
    Repeater,
}

impl WorldGraphBuilder {
    pub fn new(world: &World) -> Self {
        let queue = world
            .blocks
            .iter()
            .filter(|(_, block)| !block.kind.is_cobble())
            .map(|(pos, _)| pos.clone())
            .collect();
        let world: World3D = world.into();

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
            nodes: nodes,
            ..Default::default()
        };
        graph.build_inputs();

        WorldGraph { graph, positions }
    }

    fn visit_blocks(&mut self) {
        let mut visit: HashSet<Position> = HashSet::new();

        // wiring blocks
        for pos in &self.queue {
            if visit.contains(pos) {
                continue;
            }
            visit.insert(*pos);

            let block = &self.world[&pos];
            let visits = match block.kind {
                BlockKind::Switch { .. } => self.visit_switch(pos, block),
                BlockKind::Redstone { .. } => self.visit_redstone(pos),
                BlockKind::Torch { .. } => self.visit_torch(pos, block),
                BlockKind::Repeater { .. } => self.visit_repeater(pos, block),
                BlockKind::RedstoneBlock => self.visit_redstone_block(pos),
                BlockKind::Piston { .. } => unimplemented!(),
                BlockKind::Air | BlockKind::Cobble { .. } => unreachable!(),
            };

            if visits.len() > 0 {
                self.outputs.insert(*pos, visits);
            }
        }
    }

    fn visit_switch(&self, pos: &Position, block: &Block) -> Vec<(Direction, Position)> {
        let dir = block.direction;

        pos.forwards_except(&dir)
            .into_iter()
            .map(|pos_src| self.propagate(PropagateType::Torch, pos_src, pos.diff(&pos_src)))
            .chain(|| -> Option<Vec<(Direction, Position)>> {
                let Some(pos) = pos.walk(&dir) else {
                    return None;
                };

                Some(self.propagate(PropagateType::Hard, pos, Direction::None))
            }())
            .flatten()
            .collect_vec()
    }

    fn visit_torch(&self, pos: &Position, block: &Block) -> Vec<(Direction, Position)> {
        let dir = block.direction;

        match dir {
            Direction::Bottom => pos.cardinal(),
            Direction::East | Direction::West | Direction::South | Direction::North => {
                let mut positions = pos.cardinal_except(&dir);
                positions.extend(pos.down());
                positions
            }
            _ => unreachable!(),
        }
        .into_iter()
        .map(|pos_src| vec![self.propagate(PropagateType::Torch, pos_src, pos_src.diff(&pos))])
        .flatten()
        .chain(Some(self.propagate(
            PropagateType::Hard,
            pos.up(),
            Direction::None,
        )))
        .flatten()
        .collect_vec()
    }

    fn visit_redstone(&self, pos: &Position) -> Vec<(Direction, Position)> {
        let mut state = 0;
        let has_up_block = self.world[&pos.up()].kind.is_cobble();

        pos.cardinal().iter().for_each(|pos_src| {
            let flat_check = self.world[pos_src].kind.is_stick_to_redstone();
            let up_check = !has_up_block && self.world[&pos_src.up()].kind.is_redstone();
            let down_check = !self.world[pos_src].kind.is_cobble()
                && pos_src
                    .down()
                    .map_or(false, |pos| self.world[&pos].kind.is_redstone());

            if !flat_check && !(up_check || down_check) {
                return;
            }

            state |= match pos.diff(pos_src) {
                Direction::East => RedstoneState::East,
                Direction::West => RedstoneState::West,
                Direction::South => RedstoneState::South,
                Direction::North => RedstoneState::North,
                _ => unreachable!(),
            } as usize;
        });

        if state.count_ones() == 1 {
            if state & RedstoneState::Horizontal as usize > 0 {
                state |= RedstoneState::Horizontal as usize;
            } else {
                state |= RedstoneState::Vertical as usize;
            }
        }

        let mut propagate_targets = Vec::new();
        propagate_targets.extend(pos.cardinal_redstone(state));

        if has_up_block {
            propagate_targets.extend(pos.up().cardinal_redstone(state));
        }

        if let Some(down_pos) = pos.down() {
            if !self.world[&down_pos].kind.is_cobble() {
                unimplemented!();
            }

            propagate_targets.push(down_pos);
            propagate_targets.extend(
                pos.cardinal_redstone(state)
                    .into_iter()
                    .filter(|pos| !self.world[&pos].kind.is_cobble())
                    .filter_map(|pos| pos.walk(&Direction::Bottom))
                    .filter(|pos| !self.world[&pos].kind.is_cobble()),
            );
        }

        propagate_targets
            .iter()
            .map(|pos_src| self.propagate(PropagateType::Soft, *pos_src, pos_src.diff(&pos)))
            .flatten()
            .collect_vec()
    }

    fn visit_repeater(&self, pos: &Position, block: &Block) -> Vec<(Direction, Position)> {
        let walk = pos.walk(&&block.direction.inverse());
        let mut result: Vec<(Direction, Position)> = Vec::new();

        if let Some(pos) = walk {
            result.extend(self.propagate(PropagateType::Repeater, pos, block.direction));
        }

        if let Some(pos) = walk.map(|pos| pos.down()).flatten() {
            result.extend(self.propagate(PropagateType::Soft, pos, Direction::Bottom));
        }

        result
    }

    fn visit_redstone_block(&self, pos: &Position) -> Vec<(Direction, Position)> {
        pos.forwards()
            .into_iter()
            .map(|pos_src| self.propagate(PropagateType::Soft, pos_src, pos_src.diff(&pos)))
            .flatten()
            .collect_vec()
    }

    fn propagate(
        &self,
        propagate_type: PropagateType,
        pos: Position,
        dir: Direction,
    ) -> Vec<(Direction, Position)> {
        let block = &self.world[&pos];

        match block.kind {
            BlockKind::Air
            | BlockKind::Switch { .. }
            | BlockKind::RedstoneBlock
            | BlockKind::Torch { .. } => Vec::new(),
            // 코블에 붙어있는 레드스톤 토치, 리피터만 반응함
            BlockKind::Cobble { .. } => {
                let mut cardinal_propagation = pos
                    .cardinal_except(&dir)
                    .iter()
                    .filter_map(|pos_src| match &self.world[pos_src].kind {
                        BlockKind::Torch { .. } => Some((Direction::None, *pos_src)),
                        BlockKind::Repeater { .. } => match propagate_type {
                            PropagateType::Soft => None,
                            PropagateType::Hard
                            | PropagateType::Torch
                            | PropagateType::Repeater => Some((Direction::None, *pos_src)),
                        },
                        BlockKind::Redstone { .. } => match propagate_type {
                            PropagateType::Soft => None,
                            PropagateType::Hard
                            | PropagateType::Torch
                            | PropagateType::Repeater => Some((Direction::None, *pos_src)),
                        },
                        _ => None,
                    })
                    .collect_vec();

                let up_pos = pos.up();
                let up_block = &self.world[&up_pos];
                if up_block.direction == Direction::Bottom
                    && matches!(up_block.kind, BlockKind::Torch { .. })
                {
                    cardinal_propagation.push((Direction::None, up_pos));
                }

                cardinal_propagation
            }
            BlockKind::Redstone { .. } => vec![(Direction::None, pos)],
            BlockKind::Repeater { .. } => {
                if block.direction == dir {
                    vec![(Direction::None, pos)]
                } else if block.direction.is_othogonal_plane(dir) {
                    // lock
                    vec![(dir, pos)]
                } else {
                    vec![]
                }
            }
            BlockKind::Piston { .. } => todo!(),
        }
    }

    fn build_nodes(&mut self) -> (Vec<GraphNode>, HashMap<GraphNodeId, Position>) {
        let mut graph_id: HashMap<Position, GraphNodeId> = HashMap::new();
        let mut nodes: HashMap<Position, GraphNode> = HashMap::new();

        for (index, pos) in self
            .queue
            .iter()
            .filter(|pos| !self.world[*pos].kind.is_cobble())
            .enumerate()
        {
            graph_id.insert(*pos, index);
            nodes.insert(
                *pos,
                GraphNode::new(index, GraphNodeKind::Block(self.world[pos])),
            );
        }

        for pos in self
            .queue
            .iter()
            .filter(|pos| !self.world[pos].kind.is_cobble())
        {
            let Some(outputs) = self.outputs.get(pos) else {
                continue;
            };

            let node = nodes.get_mut(pos).unwrap();
            let mut block = self.world[pos];

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
        let nodes = nodes.into_iter().map(|(_, node)| node).collect_vec();

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
        graph::graphviz::ToGraphviz,
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
}