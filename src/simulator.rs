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
    TorchOn,
    TorchOff,
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
    id: Option<usize>,
    from_id: Option<usize>,
    event_type: EventType,
    target_position: Position,
    direction: Direction,
}

#[derive(Clone, Debug)]
pub struct Simulator {
    queue: VecDeque<VecDeque<Event>>,
    world: World3D,
    cycle: usize,
    event_id_count: usize,
}

impl Simulator {
    pub fn from(world: &World) -> eyre::Result<Self> {
        let mut sim = Self {
            queue: VecDeque::new(),
            world: world.into(),
            cycle: 0,
            event_id_count: 0,
        };

        tracing::info!("Simulation target\n{:?}", sim.world);

        sim.queue.push_back(VecDeque::new());
        sim.init_redstone_states(world);
        sim.init(world);

        tracing::debug!("queue: {:?}", sim.queue);

        sim.fill_event_id();
        sim.run()?;

        Ok(sim)
    }

    fn fill_event_id(&mut self) {
        let mut event_id = self.event_id_count;
        for events in &mut self.queue {
            for event in events {
                if event.id.is_none() {
                    event.id = Some(event_id);
                    event_id += 1;
                }
            }
        }
        self.event_id_count = event_id;
    }

    pub fn change_state(&mut self, states: Vec<(Position, bool)>) -> eyre::Result<()> {
        self.queue.push_back(VecDeque::new());

        for (pos, value) in states {
            let BlockKind::Switch { is_on } = &mut self.world[&pos].kind else {
                eyre::bail!("you can change only switch state!");
            };

            if value == *is_on {
                continue;
            }

            *is_on = value;

            pos.forwards_except(&self.world[&pos].direction)
                .into_iter()
                .map(|pos_src| Event {
                    id: None,
                    from_id: None,
                    event_type: if value {
                        EventType::TorchOn
                    } else {
                        EventType::TorchOff
                    },
                    target_position: pos_src,
                    direction: pos_src.diff(&pos),
                })
                .chain(|| -> Option<Event> {
                    let Some(pos) = pos.walk(&self.world[&pos].direction) else {
                        return None;
                    };

                    Some(Event {
                        id: None,
                        from_id: None,
                        event_type: if value {
                            EventType::HardOn
                        } else {
                            EventType::HardOff
                        },
                        target_position: pos,
                        direction: Direction::None,
                    })
                }())
                .for_each(|event| self.push_event_to_current_tick(event));
        }

        if self.queue.back().unwrap().is_empty() {
            self.queue.pop_back();
        }

        self.fill_event_id();

        self.run()?;

        Ok(())
    }

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
        .map(|pos_src| {
            vec![Event {
                id: None,
                from_id: None,
                event_type: EventType::TorchOn,
                target_position: pos_src,
                direction: pos_src.diff(pos),
            }]
        })
        .flatten()
        .chain(Some(Event {
            id: None,
            from_id: None,
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
                id: None,
                from_id: None,
                event_type: EventType::TorchOn,
                target_position: pos_src,
                direction: pos_src.diff(pos),
            })
            .chain(|| -> Option<Event> {
                let Some(pos) = pos.walk(dir) else {
                    return None;
                };

                Some(Event {
                    id: None,
                    from_id: None,
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
            id: None,
            from_id: None,
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
                    self.propagate_redstone_event(&mut block, &event)?;
                }
                BlockKind::Torch { .. } => {
                    self.propgate_torch_event(&mut block, &event)?;
                }
                BlockKind::Repeater { .. } => {
                    self.propgate_repeater_event(&mut block, &event)?;
                }
            }

            self.world[&event.target_position] = block;

            self.fill_event_id();
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

        tracing::info!("trigger cobble event: {event:?}, {block:?}");

        let events = event
            .target_position
            .forwards()
            .into_iter()
            .filter(|pos| !self.world[pos].kind.is_cobble())
            .map(|pos_src| Event {
                id: None,
                from_id: event.id,
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
            })
            .collect::<Vec<_>>();

        events.into_iter().for_each(|event| {
            self.push_event_to_current_tick(event);
        });

        Ok(())
    }

    fn propagate_redstone_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume redstone event: {:?}", block);

        let BlockKind::Redstone {
            state,
            ..
        } = &mut block.kind else {
            eyre::bail!("unreachable");
        };

        let mut propagate_targets = Vec::new();

        propagate_targets.extend(event.target_position.cardinal_redstone(*state));

        let up_pos = event.target_position.up();
        if !self.world[&up_pos].kind.is_cobble() {
            propagate_targets.extend(up_pos.cardinal_redstone(*state));
        }

        if let Some(down_pos) = event.target_position.down() {
            if !self.world[&down_pos].kind.is_cobble() {
                eyre::bail!("unreachable");
            }

            propagate_targets.push(down_pos);

            propagate_targets.extend(
                event
                    .target_position
                    .cardinal_redstone(*state)
                    .into_iter()
                    .filter(|pos| !self.world[&pos].kind.is_cobble())
                    .filter_map(|pos| pos.walk(&Direction::Bottom))
                    .filter(|pos| !self.world[&pos].kind.is_cobble()),
            );
        }

        match event.event_type {
            EventType::SoftOn
            | EventType::SoftOff
            | EventType::RepeaterOn { .. }
            | EventType::RepeaterOff { .. } => {}
            EventType::TorchOn | EventType::HardOn => {
                let BlockKind::Redstone {
                    on_count,
                    strength,
                    ..
                } = &mut block.kind else {
                    eyre::bail!("unreachable");
                };

                *on_count += 1;

                if *on_count == 1 {
                    *strength = 15;

                    tracing::info!("trigger redstone event: {event:?}, {block:?}");

                    propagate_targets.into_iter().for_each(|pos| {
                        self.push_event_to_current_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::SoftOn,
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });

                        self.push_event_to_current_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::RedstoneOn { strength: 14 },
                            target_position: pos,
                            direction: Direction::None,
                        });
                    });
                }
            }
            EventType::TorchOff | EventType::HardOff => {
                let BlockKind::Redstone {
                    on_count,
                    strength,
                    ..
                } = &mut block.kind else {
                    eyre::bail!("unreachable");
                };

                *on_count -= 1;

                if *on_count == 0 {
                    *strength = 0;

                    tracing::info!("trigger redstone event: {event:?}, {block:?}");

                    propagate_targets.into_iter().for_each(|pos| {
                        self.push_event_to_current_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::SoftOff,
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });

                        self.push_event_to_current_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::RedstoneOff,
                            target_position: pos,
                            direction: Direction::None,
                        });
                    });
                }
            }
            EventType::RedstoneOn { strength } => {
                let event_strength = strength;

                let BlockKind::Redstone {
                    strength,
                    ..
                } = &mut block.kind else {
                    eyre::bail!("unreachable");
                };

                if event_strength > *strength && *strength > 0 {
                    *strength = event_strength;

                    propagate_targets.into_iter().for_each(|pos| {
                        if !self.world[&pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::SoftOn,
                                target_position: pos,
                                direction: pos.diff(&event.target_position),
                            });
                        }

                        self.push_event_to_current_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::RedstoneOn {
                                strength: *strength - 1,
                            },
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });
                    });

                    tracing::info!("trigger redstone event: {event:?}, {block:?}");
                }
            }
            EventType::RedstoneOff => {
                let BlockKind::Redstone {
                    on_count,
                    strength,
                    ..
                } = &mut block.kind else {
                    eyre::bail!("unreachable");
                };

                if *on_count == 0 {
                    *strength = 0;

                    propagate_targets.into_iter().for_each(|pos| {
                        if !self.world[&pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::SoftOff,
                                target_position: pos,
                                direction: pos.diff(&event.target_position),
                            });
                        }

                        self.push_event_to_current_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::RedstoneOff,
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });
                    });
                } else {
                    propagate_targets.into_iter().for_each(|pos| {
                        if !self.world[&pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::SoftOn,
                                target_position: pos,
                                direction: pos.diff(&event.target_position),
                            });
                        }

                        self.push_event_to_current_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::RedstoneOn { strength: 14 },
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });
                    });
                }

                tracing::info!("trigger redstone event: {event:?}, {block:?}");
            }
        };

        Ok(())
    }

    fn propgate_torch_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume torch event: {:?}", block);

        // Cobble로부터 온 이벤트가 아닌 경우
        if !self.world[&event.target_position.walk(&event.direction).unwrap()]
            .kind
            .is_cobble()
        {
            return Ok(());
        }

        // Cobble에 붙어있지 않은 경우
        if event.direction != block.direction {
            return Ok(());
        }

        let BlockKind::Torch { is_on } = &mut block.kind else {
            eyre::bail!("unreachable");
        };

        *is_on = !event.event_type.is_on();

        match block.direction {
            Direction::Bottom => event.target_position.cardinal(),
            Direction::East | Direction::West | Direction::South | Direction::North => {
                let mut positions = event.target_position.cardinal_except(&block.direction);
                positions.extend(event.target_position.down());
                positions
            }
            _ => unreachable!(),
        }
        .into_iter()
        .map(|pos_src| Event {
            id: None,
            from_id: event.id,
            event_type: if *is_on {
                EventType::TorchOn
            } else {
                EventType::TorchOff
            },
            target_position: pos_src,
            direction: pos_src.diff(&event.target_position),
        })
        .chain(Some(Event {
            id: None,
            from_id: event.id,
            event_type: if *is_on {
                EventType::HardOn
            } else {
                EventType::HardOff
            },
            target_position: event.target_position.up(),
            direction: Direction::None,
        }))
        .for_each(|event| {
            self.push_event_to_next_tick(event);
        });

        tracing::info!("trigger torch event: {event:?}, {block:?}");

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
                EventType::SoftOn
                | EventType::HardOn
                | EventType::TorchOn
                | EventType::RedstoneOn { .. } => {
                    if is_on {
                        break;
                    }

                    if lock_signal {
                        block.kind.set_repeater_lock(true)?;

                        tracing::info!("trigger repeater event: {event:?}, {block:?}");

                        break;
                    }

                    self.push_event_to_next_tick(Event {
                        id: None,
                        from_id: event.id,
                        event_type: EventType::RepeaterOn { delay: delay },
                        target_position: event.target_position,
                        direction: event.direction,
                    });
                }
                EventType::SoftOff
                | EventType::HardOff
                | EventType::TorchOff
                | EventType::RedstoneOff => {
                    if !is_on {
                        break;
                    }

                    if lock_signal {
                        block.kind.set_repeater_lock(false)?;

                        tracing::info!("trigger repeater event: {event:?}, {block:?}");

                        break;
                    }

                    self.push_event_to_next_tick(Event {
                        id: None,
                        from_id: event.id,
                        event_type: EventType::RepeaterOff { delay: delay },
                        target_position: event.target_position,
                        direction: event.direction,
                    });
                }
                EventType::RepeaterOn { delay } => {
                    if delay != 0 {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
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
                            id: None,
                            from_id: event.id,
                            event_type: EventType::HardOn,
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });
                    }

                    if let Some(pos) = walk.map(|pos| pos.down()).flatten() {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::SoftOn,
                            target_position: pos,
                            direction: Direction::Bottom,
                        });
                    }

                    tracing::info!("trigger repeater event: {event:?}, {block:?}");
                }
                EventType::RepeaterOff { delay } => {
                    if delay != 0 {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
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
                            id: None,
                            from_id: event.id,
                            event_type: EventType::HardOff,
                            target_position: pos,
                            direction: pos.diff(&event.target_position),
                        });
                    }

                    if let Some(pos) = walk.map(|pos| pos.down()).flatten() {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::SoftOff,
                            target_position: pos,
                            direction: Direction::Bottom,
                        });
                    }

                    tracing::info!("trigger repeater event: {event:?}, {block:?}");
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

        let mut sim = Simulator {
            queue: VecDeque::new(),
            world: (&mock_world).into(),
            cycle: 0,
            event_id_count: 0,
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
            size: DimSize(4, 4, 2),
            blocks: vec![
                (Position(1, 2, 0), default_restone.clone()),
                (Position(1, 1, 0), default_repeater.clone()),
                (Position(0, 2, 0), default_restone.clone()),
                (Position(2, 2, 0), default_restone.clone()),
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

        let BlockKind::Redstone {  strength , .. } = sim.world.map[0][2][2].kind else {
            unreachable!();
        };

        assert_eq!(strength, 14)
    }

    #[test]
    pub fn unittest_simulator_torch() {
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
            size: DimSize(7, 4, 2),
            blocks: vec![
                (Position(0, 1, 0), default_restone.clone()),
                (Position(0, 2, 0), default_cobble.clone()),
                (
                    Position(1, 2, 0),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::West,
                    },
                ),
                (Position(2, 2, 0), default_restone.clone()),
                (Position(3, 2, 0), default_cobble.clone()),
                (
                    Position(4, 2, 0),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::West,
                    },
                ),
                (Position(5, 2, 0), default_restone.clone()),
                (
                    Position(0, 0, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: true },
                        direction: Direction::Bottom,
                    },
                ),
            ],
        };

        let mut sim = Simulator::from(&mock_world).unwrap();

        sim.run().unwrap();

        let BlockKind::Redstone { strength, .. } = sim.world.map[0][2][2].kind else {
            unreachable!();
        };

        assert_eq!(strength, 0);

        sim.change_state(vec![(Position(0, 0, 0), false)]).unwrap();

        let BlockKind::Redstone { strength, .. } = sim.world.map[0][2][2].kind else {
            unreachable!();
        };

        assert!(strength > 0);
    }

    #[test]
    pub fn unittest_simulator_and_gate() {
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

        let mut sim = Simulator::from(&mock_world).unwrap();

        sim.run().unwrap();

        let BlockKind::Redstone { strength, .. } = sim.world.map[0][4][1].kind else {
            unreachable!();
        };

        assert_eq!(strength, 0);

        sim.change_state(vec![(Position(0, 0, 0), true), (Position(2, 0, 0), true)])
            .unwrap();

        let BlockKind::Redstone { strength, .. } = sim.world.map[0][4][1].kind else {
            unreachable!();
        };

        assert!(strength > 0);
    }
}
