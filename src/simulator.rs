use std::collections::VecDeque;

use crate::common::{
    block::{Block, BlockKind, Direction, RedstoneState, RedstoneStateType},
    world::{World, World3D},
    Position,
};

#[derive(Clone, Debug)]
enum EventType {
    // targeting redstone, repeater
    SoftOff,
    SoftOn,
    // targeting block
    HardOn,
    HardOff,
    RedstoneOn { strength: usize },
    RedstoneOff { strength: usize },
}

#[derive(Clone, Debug)]
struct Event {
    event_type: EventType,
    target_position: Position,
    direction: Direction,
}

#[derive(Clone, Debug)]
pub struct Simulator {
    queue: VecDeque<Vec<Event>>,
    world: World3D,
}

impl Simulator {
    pub fn from(world: &World) -> eyre::Result<Self> {
        let mut sim = Self {
            queue: VecDeque::new(),
            world: world.into(),
        };

        sim.init_states(world);
        sim.init(world);

        tracing::debug!("queue: {:?}", sim.queue);

        while !sim.is_empty() {
            sim.consume_events()?;
        }

        Ok(sim)
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

    fn init_states(&mut self, world: &World) {
        for (pos, block) in &world.blocks {
            let BlockKind::Redstone {
                on_count,
                strength,
                ..
            } = block.kind else {
                continue;
            };

            let mut state = 0;

            pos.cardinal().iter().for_each(|pos_src| {
                if !matches!(
                    self.world[pos_src].kind,
                    BlockKind::Redstone { .. }
                        | BlockKind::Repeater { .. }
                        | BlockKind::Switch { .. }
                        | BlockKind::Torch { .. }
                ) {
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
                if state & (RedstoneState::East as usize | RedstoneState::West as usize) > 0 {
                    state |= RedstoneState::East as usize | RedstoneState::West as usize;
                } else {
                    state |= RedstoneState::South as usize | RedstoneState::North as usize;
                }
            }

            self.world[pos].kind = BlockKind::Redstone {
                on_count,
                state,
                strength,
            };
        }
    }

    fn init_torch_event(&mut self, dir: &Direction, pos: &Position) {
        tracing::debug!("produce torch event: {:?}, {:?}", dir, pos);

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
        tracing::debug!("produce switch event: {:?}, {:?}", dir, pos);

        self.queue.push_back(
            pos.forwards_except(dir)
                .into_iter()
                .map(|pos_src| Event {
                    event_type: EventType::SoftOn,
                    target_position: pos_src,
                    direction: pos_src.diff(pos),
                })
                .chain(|| -> Option<Event> {
                    let Some(pos) = pos.walk(dir) else {
                    return None;
                };

                    Some(Event {
                        event_type: EventType::HardOn,
                        target_position: pos,
                        direction: Direction::None,
                    })
                }())
                .collect(),
        )
    }

    fn init_redstone_block_event(&mut self, pos: &Position) {
        tracing::debug!("produce redstone block event: {:?}", pos);

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

    fn consume_events(&mut self) -> eyre::Result<()> {
        while let Some(events) = self.queue.pop_front() {
            for event in &events {
                tracing::debug!("consume event: {:?}", event);

                let mut block = self.world[&event.target_position];

                match block.kind {
                    BlockKind::Air | BlockKind::Switch { .. } | BlockKind::RedstoneBlock => (),
                    BlockKind::Cobble { .. } => {
                        self.propgate_cobble_event(&mut block, &event)?;
                    }
                    BlockKind::Redstone { .. } => {
                        self.propagate_redstone_event(&mut block, &event)?;
                    }
                    BlockKind::Torch { is_on } => todo!(),
                    BlockKind::Repeater { is_on, is_locked } => todo!(),
                }
            }
        }

        Ok(())
    }

    fn propgate_cobble_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume cobble event: {:?}", block);

        match event.event_type {
            EventType::SoftOn | EventType::HardOn => {
                let is_hard = matches!(event.event_type, EventType::HardOn);

                block.count_up(is_hard)?;

                let BlockKind::Cobble {
                    on_count,
                    on_base_count,
                    ..
                } = block.kind else {
                    unreachable!()
                };

                if (is_hard && on_base_count == 1) || on_count == 1 {
                    self.queue.push_back(
                        event
                            .target_position
                            .forwards()
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: if is_hard && on_base_count == 1 {
                                    EventType::HardOn
                                } else {
                                    EventType::SoftOn
                                },
                                target_position: pos_src,
                                direction: pos_src.diff(&event.target_position),
                            })
                            .collect(),
                    );

                    tracing::debug!("produce events: {:?}", self.queue.back());
                }
            }
            EventType::SoftOff | EventType::HardOff => {
                let is_hard = matches!(event.event_type, EventType::HardOff);

                block.count_down(is_hard)?;

                let BlockKind::Cobble {
                    on_count,
                    on_base_count,
                    ..
                } = block.kind else {
                    unreachable!()
                };

                // propagate event
                if (is_hard && on_base_count == 1) || on_count == 1 {
                    self.queue.push_back(
                        event
                            .target_position
                            .forwards()
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: if is_hard && on_base_count == 1 {
                                    EventType::HardOff
                                } else {
                                    EventType::SoftOff
                                },
                                target_position: pos_src,
                                direction: pos_src.diff(&event.target_position),
                            })
                            .collect(),
                    );

                    tracing::debug!("produce events: {:?}", self.queue.back());
                }
            }
            _ => (),
        };

        Ok(())
    }

    fn propagate_redstone_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume redstone event: {:?}", block);

        match event.event_type {
            EventType::SoftOn | EventType::HardOn => {
                let is_hard = matches!(event.event_type, EventType::HardOn);

                block.count_up(is_hard)?;

                let BlockKind::Redstone {
                    on_count,
                    state,
                    strength,
                } = block.kind else {
                    unreachable!()
                };
            }
            EventType::SoftOff | EventType::HardOff => {
                let is_hard = matches!(event.event_type, EventType::HardOff);

                block.count_down(is_hard)?;

                let BlockKind::Redstone {
                    on_count,
                    state,
                    strength,
                } = block.kind else {
                    unreachable!()
                };
            }
            EventType::RedstoneOn { strength } | EventType::RedstoneOff { strength } => todo!(),
        };

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::collections::VecDeque;

    use crate::common::DimSize;

    use super::*;

    #[test]
    pub fn unittest_simulator_init_states() {
        let default_restone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };

        let mock_world = World {
            size: DimSize(3, 4, 2),
            blocks: vec![
                (Position(1, 1, 0), default_restone.clone()),
                (Position(0, 1, 0), default_restone.clone()),
                (Position(1, 0, 0), default_restone.clone()),
                (
                    Position(1, 2, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: true },
                        direction: Direction::Top,
                    },
                ),
            ],
        };

        let mut sim = Simulator {
            queue: VecDeque::new(),
            world: (&mock_world).into(),
        };

        sim.init_states(&mock_world);

        let BlockKind::Redstone { state, .. } = sim.world.map[0][1][1].kind else {
            unreachable!();
        };

        assert_eq!(
            state,
            RedstoneState::West as usize
                | RedstoneState::South as usize
                | RedstoneState::North as usize
        );
    }

    #[test]
    fn unittest_simulator_redstone_init() {
        tracing_subscriber::fmt::init();

        let default_restone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };

        let mock_world = World {
            size: DimSize(3, 4, 2),
            blocks: vec![
                (Position(1, 1, 0), default_restone.clone()),
                (Position(0, 1, 0), default_restone.clone()),
                (Position(1, 0, 0), default_restone.clone()),
                (
                    Position(1, 2, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: true },
                        direction: Direction::Top,
                    },
                ),
            ],
        };

        let sim = Simulator::from(&mock_world).unwrap();

        let BlockKind::Redstone { on_count, ..  } = sim.world.map[0][1][1].kind else {
            unreachable!();
        };

        assert_eq!(on_count, 1);
    }
}
