use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use super::{Graph, GraphNodeId};
use crate::graph::{GraphNode, GraphNodeKind};
use crate::transform::place_and_route::place_bound::{PlaceBound, PropagateType};
use crate::utils::Verify;
use crate::world::block::{BlockKind, Direction};
use crate::world::position::Position;
use crate::world::world::{World, World3D};

#[derive(Debug, Default, Clone)]
pub struct WorldGraph {
    pub graph: Graph,
    pub positions: HashMap<GraphNodeId, Position>,
    pub routings: HashSet<GraphNodeId>,
}

impl Verify for WorldGraph {
    fn verify(&self) -> eyre::Result<()> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct WorldGraphBuilder {
    world: World3D,
    queue: Vec<Position>,
    outputs: HashMap<Position, Vec<(Direction, Position)>>,
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
                .filter(|bound| self.world.size.bound_on(bound.position()))
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
    use super::WorldGraphBuilder;
    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::world::block::{Block, BlockKind, Direction};
    use crate::world::position::{DimSize, Position};
    use crate::world::world::World;

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
