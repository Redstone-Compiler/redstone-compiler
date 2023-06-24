use std::{borrow::Borrow, collections::VecDeque};

use crate::common::{
    block::{Block, BlockKind, Direction, RedstoneState},
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
    RedstoneOff,
    RepeaterOn { delay: usize },
    RepeaterOff { delay: usize },
}

impl EventType {
    fn is_hard(&self) -> bool {
        matches!(self, EventType::HardOn | EventType::HardOff)
    }

    fn is_on(&self) -> bool {
        matches!(self, EventType::SoftOn | EventType::HardOn)
    }

    fn is_redstone(&self) -> bool {
        matches!(self, EventType::RedstoneOn { .. } | EventType::RedstoneOff)
    }
}

#[derive(Clone, Debug)]
struct Event {
    event_type: EventType,
    target_position: Position,
    direction: Direction,
}

#[derive(Clone, Debug)]
pub struct Simulator {
    queue: VecDeque<VecDeque<Event>>,
    world: World3D,
    cycle: usize,
}

impl Simulator {
    pub fn from(world: &World) -> eyre::Result<Self> {
        let mut sim = Self {
            queue: VecDeque::new(),
            world: world.into(),
            cycle: 0,
        };

        sim.queue.push_back(VecDeque::new());
        sim.init_redstone_states(world);
        sim.init(world);

        tracing::debug!("queue: {:?}", sim.queue);

        sim.consume_events()?;

        Ok(sim)
    }

    pub fn change_state(&mut self, states: Vec<(Position, bool)>) {}

    pub fn run(&mut self) -> eyre::Result<usize> {
        let mut local_cycle = 0;

        while !self.queue.is_empty() {
            self.consume_events()?;
            local_cycle += 1;
            tracing::info!("simulator cycle: {local_cycle}/{}", self.cycle);
        }

        Ok(local_cycle)
    }

    pub fn is_empty(&mut self) -> bool {
        self.queue.is_empty()
    }

    fn push_event_to_current_tick(&mut self, event: Event) {
        self.queue.front_mut().unwrap().push_back(event);
    }

    fn push_event_to_next_tick(&mut self, event: Event) {
        self.queue.back_mut().unwrap().push_back(event);
    }

    fn init(&mut self, world: &World) {
        for (pos, block) in &world.blocks {
            match block.kind {
                BlockKind::Torch { is_on } if is_on => {
                    self.init_torch_event(&block.direction, pos);
                }
                BlockKind::Switch { is_on } if is_on => {
                    self.init_switch_event(&block.direction, pos);
                }
                BlockKind::RedstoneBlock => {
                    self.init_redstone_block_event(pos);
                }
                _ => (),
            };
        }
    }

    fn init_redstone_states(&mut self, world: &World) {
        for (pos, block) in &world.blocks {
            let BlockKind::Redstone {
                on_count,
                strength,
                ..
            } = block.kind else {
                continue;
            };

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

            self.world[pos].kind = BlockKind::Redstone {
                on_count,
                state,
                strength,
            };
        }
    }

    fn init_torch_event(&mut self, dir: &Direction, pos: &Position) {
        tracing::debug!("produce torch event: {:?}, {:?}", dir, pos);

        let events = match dir {
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
        }));

        self.queue[0].extend(events);
    }

    fn init_switch_event(&mut self, dir: &Direction, pos: &Position) {
        tracing::debug!("produce switch event: {:?}, {:?}", dir, pos);

        let events = pos
            .forwards_except(dir)
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
            }());

        self.queue[0].extend(events);
    }

    fn init_redstone_block_event(&mut self, pos: &Position) {
        tracing::debug!("produce redstone block event: {:?}", pos);

        let events = pos.forwards().into_iter().map(|pos_src| Event {
            event_type: EventType::SoftOn,
            target_position: pos_src,
            direction: pos_src.diff(pos),
        });

        self.queue[0].extend(events)
    }

    fn consume_events(&mut self) -> eyre::Result<()> {
        self.cycle += 1;

        self.queue.push_back(VecDeque::new());

        while let Some(event) = self.queue.front_mut().unwrap().pop_front() {
            tracing::debug!("consume event: {:?}", event);

            let mut block = self.world[&event.target_position];

            match block.kind {
                BlockKind::Air | BlockKind::Switch { .. } | BlockKind::RedstoneBlock => (),
                BlockKind::Cobble { .. } => {
                    self.propgate_cobble_event(&mut block, &event)?;
                }
                BlockKind::Redstone { .. } => {
                    // self.propagate_redstone_event(&mut block, &event)?;
                }
                BlockKind::Torch { is_on } => todo!(),
                BlockKind::Repeater { .. } => {
                    self.propgate_repeater_event(&mut block, &event)?;
                }
            }
        }

        self.queue.pop_front();

        if self.queue.back().unwrap().is_empty() {
            self.queue.pop_back();
        }

        Ok(())
    }

    fn propgate_cobble_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume cobble event: {:?}", block);

        if event.event_type.is_redstone() {
            return Ok(());
        }

        let is_hard = matches!(event.event_type, EventType::HardOn | EventType::HardOff);

        if event.event_type.is_on() {
            block.count_up(event.event_type.is_hard())?;
        } else {
            block.count_down(event.event_type.is_hard())?;
        }

        let BlockKind::Cobble {
            on_count,
            on_base_count,
            ..
        } = block.kind else {
            unreachable!()
        };

        let count_condition = if event.event_type.is_on() { 1 } else { 0 };

        if !((is_hard && on_base_count == count_condition) || on_count == count_condition) {
            return Ok(());
        }

        tracing::info!("trigger cobble event: {:?}", block);

        let events = event
            .target_position
            .forwards()
            .into_iter()
            .map(|pos_src| Event {
                event_type: if is_hard && on_base_count == count_condition {
                    if event.event_type.is_on() {
                        EventType::HardOn
                    } else {
                        EventType::HardOff
                    }
                } else {
                    if event.event_type.is_on() {
                        EventType::SoftOn
                    } else {
                        EventType::SoftOff
                    }
                },
                target_position: pos_src,
                direction: pos_src.diff(&event.target_position),
            });

        self.queue
            .back_mut()
            .unwrap()
            .extend(events.borrow().clone().collect::<VecDeque<_>>());

        tracing::debug!("produce events: {:?}", events.collect::<Vec<_>>());

        Ok(())
    }

    fn propagate_redstone_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume redstone event: {:?}", block);

        /*if !event.event_type.is_redstone() {
            if event.event_type.is_on() {
                block.count_up(event.event_type.is_hard())?;
            } else {
                block.count_down(event.event_type.is_hard())?;
            }
        } else {
            block.count_up(false)?;
        }

        let BlockKind::Redstone {
            on_count,
            state,
            strength,
        } = block.kind else {
            unreachable!()
        };

        let current_strength = strength;
        let event_strength = match event.event_type {
            EventType::RedstoneOn { strength } => strength,
            _ => unreachable!(),
        };

        let count_condition = if event.event_type.is_on() { 1 } else { 0 };

        if on_count == count_condition {
            let next_strength = match event.event_type {
                EventType::SoftOff | EventType::HardOn => 15,
                EventType::SoftOn | EventType::HardOff => 0,
                EventType::RedstoneOn { strength } => current_strength.max(strength - 1),
                EventType::RedstoneOff => todo!(),
            };

            match block.kind {
                BlockKind::Redstone {
                    ref mut strength, ..
                } => *strength = next_strength,
                _ => unreachable!(),
            };

            self.queue.extend(
                event
                    .target_position
                    .cardinal_redstone(state)
                    .into_iter()
                    .map(|pos_src| Event {
                        event_type: if event.event_type.is_on() {
                            EventType::SoftOn
                        } else {
                            EventType::SoftOff
                        },
                        target_position: pos_src,
                        direction: pos_src.diff(&event.target_position),
                    }),
            );
        }

        if !event.event_type.is_redstone() || strength != event_strength - 1 {
            self.queue.extend(
                event
                    .target_position
                    .cardinal_redstone(state)
                    .into_iter()
                    .map(|pos_src| Event {
                        event_type: if event.event_type.is_on() {
                            EventType::RedstoneOn {
                                strength: if event.event_type.is_redstone() {
                                    strength.max(event_strength - 1) - 1
                                } else {
                                    15
                                },
                            }
                        } else {
                            EventType::RedstoneOff
                        },
                        target_position: pos_src,
                        direction: Direction::None,
                    }),
            );
        }*/

        /*match event.event_type {
            EventType::SoftOn | EventType::HardOn => {
                block.count_up(event.event_type.is_hard())?;

                let BlockKind::Redstone {
                    on_count,
                    state,
                    ..
                } = block.kind else {
                    unreachable!()
                };

                if on_count == 1 {
                    block.kind.set_redstone_strength(15)?;

                    self.queue.back().unwrap().extend(
                        event
                            .target_position
                            .cardinal_redstone(state)
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: EventType::SoftOn,
                                target_position: pos_src,
                                direction: pos_src.diff(&event.target_position),
                            }),
                    );

                    self.queue.back().unwrap().extend(
                        event
                            .target_position
                            .cardinal_redstone(state)
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: EventType::RedstoneOn { strength: 15 },
                                target_position: pos_src,
                                direction: Direction::None,
                            }),
                    );
                }
            }
            EventType::SoftOff | EventType::HardOff => {
                block.count_down(event.event_type.is_hard())?;

                let BlockKind::Redstone {
                    on_count,
                    state,
                    ..
                } = block.kind else {
                    unreachable!()
                };

                if on_count == 0 {
                    block.kind.set_redstone_strength(0)?;

                    self.queue.back().unwrap().extend(
                        event
                            .target_position
                            .cardinal_redstone(state)
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: EventType::SoftOff,
                                target_position: pos_src,
                                direction: pos_src.diff(&event.target_position),
                            }),
                    );

                    self.queue.back().unwrap().extend(
                        event
                            .target_position
                            .cardinal_redstone(state)
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: EventType::RedstoneOff,
                                target_position: pos_src,
                                direction: Direction::None,
                            }),
                    );
                }
            }
            EventType::RedstoneOn { strength } => {
                block.count_up(false)?;

                let event_strength = strength;

                let BlockKind::Redstone {
                    on_count,
                    state,
                    strength,
                    ..
                } = block.kind else {
                    unreachable!()
                };

                block
                    .kind
                    .set_redstone_strength(strength.max(event_strength - 1))?;

                if on_count == 1 {
                    self.queue.back().unwrap().extend(
                        event
                            .target_position
                            .cardinal_redstone(state)
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: EventType::SoftOn,
                                target_position: pos_src,
                                direction: pos_src.diff(&event.target_position),
                            }),
                    );
                }

                if strength != event_strength - 1 {
                    self.queue.back().unwrap().extend(
                        event
                            .target_position
                            .cardinal_redstone(state)
                            .into_iter()
                            .map(|pos_src| Event {
                                event_type: EventType::RedstoneOn {
                                    strength: strength.max(event_strength - 1) - 1,
                                },
                                target_position: pos_src,
                                direction: Direction::None,
                            }),
                    );
                }
            }
            _ => todo!(),
        };*/

        Ok(())
    }

    fn propgate_repeater_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume repeater event: {:?}", block);

        let BlockKind::Repeater {
            is_on,
            is_locked,
            delay
        } = block.kind else {
            unreachable!()
        };

        let lock_signal = event.direction.is_othogonal_plane(block.direction);

        if event.direction != block.direction && !lock_signal {
            return Ok(());
        }

        if event.direction == block.direction && is_locked {
            return Ok(());
        }

        loop {
            match event.event_type {
                EventType::SoftOn | EventType::HardOn | EventType::RedstoneOn { .. } => {
                    if is_on {
                        break;
                    }

                    if lock_signal {
                        block.kind.set_repeater_lock(true)?;

                        tracing::info!("trigger repeater event: {:?}", block);

                        break;
                    }

                    self.push_event_to_next_tick(Event {
                        event_type: EventType::RepeaterOn { delay: delay },
                        target_position: event.target_position,
                        direction: event.direction,
                    });
                }
                EventType::SoftOff | EventType::HardOff | EventType::RedstoneOff => {
                    if !is_on {
                        break;
                    }

                    if lock_signal {
                        block.kind.set_repeater_lock(false)?;

                        tracing::info!("trigger repeater event: {:?}", block);

                        break;
                    }

                    self.push_event_to_next_tick(Event {
                        event_type: EventType::RepeaterOff { delay: delay },
                        target_position: event.target_position,
                        direction: event.direction,
                    });
                }
                EventType::RepeaterOn { delay } => {
                    if delay != 0 {
                        self.push_event_to_next_tick(Event {
                            event_type: EventType::RepeaterOn { delay: delay - 1 },
                            target_position: event.target_position,
                            direction: event.direction,
                        });
                        break;
                    }

                    block.kind.set_repeater_state(true)?;

                    // propagate
                    let walk = event.target_position.walk(&event.direction.inverse());

                    if let Some(pos) = walk {
                        self.push_event_to_next_tick(Event {
                            event_type: EventType::HardOn,
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });
                    }

                    if let Some(pos) = walk.map(|pos| pos.down()).flatten() {
                        self.push_event_to_next_tick(Event {
                            event_type: EventType::SoftOn,
                            target_position: pos,
                            direction: Direction::Bottom,
                        });
                    }

                    tracing::info!("trigger repeater event: {:?}", block);
                }
                EventType::RepeaterOff { delay } => {
                    if delay != 0 {
                        self.push_event_to_next_tick(Event {
                            event_type: EventType::RepeaterOff { delay: delay - 1 },
                            target_position: event.target_position,
                            direction: event.direction,
                        });
                        break;
                    }

                    block.kind.set_repeater_state(false)?;

                    let walk = event.target_position.walk(&event.direction.inverse());

                    if let Some(pos) = walk {
                        self.push_event_to_next_tick(Event {
                            event_type: EventType::HardOff,
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });
                    }

                    if let Some(pos) = walk.map(|pos| pos.down()).flatten() {
                        self.push_event_to_next_tick(Event {
                            event_type: EventType::SoftOff,
                            target_position: pos,
                            direction: Direction::Bottom,
                        });
                    }

                    tracing::info!("trigger repeater event: {:?}", block);
                }
            }

            break;
        }

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
            cycle: 0,
        };

        sim.init_redstone_states(&mock_world);

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

    #[test]
    fn unittest_simulator_cobble() {
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
            size: DimSize(3, 4, 2),
            blocks: vec![
                (Position(1, 1, 0), default_cobble.clone()),
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

        let BlockKind::Cobble { on_count, on_base_count  } = sim.world.map[0][1][1].kind else {
            unreachable!();
        };

        assert_eq!(on_count, 1);
        assert_eq!(on_base_count, 0);
    }

    #[test]
    fn unittest_simulator_repeater() {
        tracing_subscriber::fmt::init();

        let default_restone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Default::default(),
        };

        let default_repeater = Block {
            kind: BlockKind::Repeater {
                is_on: false,
                is_locked: false,
                delay: 2,
            },
            direction: Direction::South,
        };

        let mock_world = World {
            size: DimSize(3, 4, 2),
            blocks: vec![
                (Position(1, 2, 0), default_restone.clone()),
                (Position(1, 1, 0), default_repeater.clone()),
                (Position(0, 2, 0), default_restone.clone()),
                (
                    Position(1, 0, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: true },
                        direction: Direction::Top,
                    },
                ),
            ],
        };

        let mut sim = Simulator::from(&mock_world).unwrap();

        sim.run().unwrap();
    }
}
