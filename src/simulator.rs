use std::collections::VecDeque;

use crate::common::{
    block::{BlockKind, Direction},
    world::{World, World3D},
    Position,
};

enum EventType {
    // targeting redstone, repeater
    SoftOff,
    SoftOn,
    // targeting block
    HardOn,
    HardOff,
}

struct Event {
    event_type: EventType,
    target_position: Position,
    direction: Direction,
}

pub struct Simulator {
    queue: VecDeque<Vec<Event>>,
    world: World3D,
}

impl Simulator {
    pub fn from(world: &World) -> Self {
        let mut sim = Self {
            queue: VecDeque::new(),
            world: world.into(),
        };

        sim.init(world);

        while !sim.is_empty() {
            sim.consume_events();
        }

        sim
    }

    pub fn change_state(&mut self, states: Vec<(Position, bool)>) {}

    pub fn is_empty(&mut self) -> bool {
        self.queue.is_empty()
    }

    fn init(&mut self, world: &World) {
        for (pos, block) in &world.blocks {
            match block.kind {
                BlockKind::Torch { is_on } => {
                    if !is_on {
                        continue;
                    }

                    self.init_torch_event(&block.direction, pos);
                }
                BlockKind::Switch { is_on } => {
                    if !is_on {
                        continue;
                    }

                    self.init_switch_event(&block.direction, pos);
                }
                BlockKind::RedstoneBlock => {
                    self.init_redstone_block_event(pos);
                }
                _ => (),
            };
        }
    }

    fn init_torch_event(&mut self, dir: &Direction, pos: &Position) {
        self.queue.push_back(
            match dir {
                Direction::Bottom => pos.cardinal(),
                Direction::East | Direction::West | Direction::South | Direction::North => {
                    let mut positions = pos.cardinal_except(dir);
                    positions.extend(pos.down());
                    positions
                }
                _ => unreachable!(),
            }
            .into_iter()
            .map(|pos_src| Event {
                event_type: EventType::SoftOn,
                target_position: pos_src,
                direction: pos_src.diff(pos),
            })
            .chain(Some(Event {
                event_type: EventType::HardOn,
                target_position: pos.up(),
                direction: Direction::None,
            }))
            .collect(),
        )
    }

    fn init_switch_event(&mut self, dir: &Direction, pos: &Position) {
        self.queue.push_back(
            match dir {
                Direction::Top | Direction::Bottom => pos.cardinal(),
                Direction::East | Direction::West | Direction::South | Direction::North => {
                    let mut positions = pos.cardinal_except(dir);
                    positions.extend(pos.down());
                    positions
                }
                _ => unreachable!(),
            }
            .into_iter()
            .chain(Some(pos.up()))
            .map(|pos_src| Event {
                event_type: EventType::SoftOn,
                target_position: pos_src,
                direction: pos_src.diff(pos),
            })
            .chain(Some(Event {
                event_type: EventType::HardOn,
                target_position: pos.walk(dir).unwrap(),
                direction: Direction::None,
            }))
            .collect(),
        )
    }

    fn init_redstone_block_event(&mut self, pos: &Position) {
        self.queue.push_back(
            pos.forwards()
                .into_iter()
                .map(|pos_src| Event {
                    event_type: EventType::SoftOn,
                    target_position: pos_src,
                    direction: pos_src.diff(pos),
                })
                .collect(),
        )
    }

    fn consume_events(&mut self) {}
}
