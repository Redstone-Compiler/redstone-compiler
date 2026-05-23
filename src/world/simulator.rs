use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use super::block::{Block, BlockKind, Direction};
use super::position::Position;
use super::{World, World3D};

const DEFAULT_TRACE_LIMIT: usize = 0;
// Approximate Minecraft redstone torch burnout so feedback loops can settle
// instead of producing simulator events forever. These are simulator cycles,
// not exact game ticks or redstone ticks.
const TORCH_BURNOUT_WINDOW_CYCLES: usize = 60;
const TORCH_BURNOUT_TOGGLE_LIMIT: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EventKey {
    event_type: EventType,
    target_position: Position,
    direction: Direction,
}

impl EventKey {
    fn from_event(event: &Event) -> Self {
        Self {
            event_type: event.event_type.clone(),
            target_position: event.target_position,
            direction: event.direction,
        }
    }
}

impl EventType {
    fn is_hard(&self) -> bool {
        matches!(self, EventType::HardOn | EventType::HardOff)
    }

    fn is_on(&self) -> bool {
        matches!(
            self,
            EventType::SoftOn | EventType::HardOn | EventType::TorchOn
        )
    }

    fn is_redstone(&self) -> bool {
        matches!(self, EventType::RedstoneOn { .. } | EventType::RedstoneOff)
    }
}

#[derive(Clone, Debug)]
struct Event {
    id: Option<usize>,
    #[allow(dead_code)]
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
    soft_power_sources: HashSet<(Position, Position)>,
    hard_power_sources: HashSet<(Position, Position)>,
    torch_toggle_cycles: HashMap<Position, VecDeque<usize>>,
    burned_out_torches: HashSet<Position>,
    trace: Vec<SimulationTraceEntry>,
    snapshots: Vec<SimulationSnapshot>,
    trace_limit: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SimulationTraceEntry {
    pub cycle: usize,
    pub event_id: Option<usize>,
    pub from_event_id: Option<usize>,
    pub event_type: String,
    pub target_position: [usize; 3],
    pub direction: String,
    pub block_before: String,
    pub current_queue_len: usize,
    pub next_queue_len: usize,
}

#[derive(Clone, Debug)]
pub struct SimulationSnapshot {
    pub cycle: usize,
    pub world: World3D,
}

#[derive(Clone, Debug)]
pub struct SimulationTraceError {
    message: String,
    trace: Vec<SimulationTraceEntry>,
    snapshots: Vec<SimulationSnapshot>,
}

impl SimulationTraceError {
    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn trace(&self) -> &[SimulationTraceEntry] {
        &self.trace
    }

    pub fn snapshots(&self) -> &[SimulationSnapshot] {
        &self.snapshots
    }
}

impl fmt::Display for SimulationTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for SimulationTraceError {}

#[derive(Copy, Clone, Debug)]
struct SimulationLimits {
    max_cycles: Option<usize>,
    max_events: Option<usize>,
}

impl SimulationLimits {
    fn cycles(max_cycles: Option<usize>) -> Self {
        Self {
            max_cycles,
            max_events: None,
        }
    }
}

impl Simulator {
    pub fn from(world: &World) -> eyre::Result<Self> {
        Self::from_inner(world, SimulationLimits::cycles(None), DEFAULT_TRACE_LIMIT)
    }

    pub fn from_with_max_cycles(world: &World, max_cycles: usize) -> eyre::Result<Self> {
        Self::from_inner(
            world,
            SimulationLimits::cycles(Some(max_cycles)),
            DEFAULT_TRACE_LIMIT,
        )
    }

    pub fn from_with_limits_and_trace(
        world: &World,
        max_cycles: usize,
        max_events: usize,
        trace_limit: usize,
    ) -> Result<Self, SimulationTraceError> {
        let mut sim = Self::new(world, trace_limit);
        let limits = SimulationLimits {
            max_cycles: Some(max_cycles),
            max_events: Some(max_events),
        };

        tracing::info!("Simulation target\n{:?}", sim.world);

        sim.queue.push_back(VecDeque::new());
        sim.world.initialize_redstone_states();
        sim.normalize_torches_on();
        sim.init();

        tracing::debug!("queue: {:?}", sim.queue);

        sim.fill_event_id();

        if let Err(error) = sim.run_inner(limits) {
            return Err(SimulationTraceError {
                message: error.to_string(),
                trace: sim.trace,
                snapshots: sim.snapshots,
            });
        }
        sim.clear_transient_torch_burnout();

        Ok(sim)
    }

    fn from_inner(
        world: &World,
        limits: SimulationLimits,
        trace_limit: usize,
    ) -> eyre::Result<Self> {
        let mut sim = Self::new(world, trace_limit);

        tracing::info!("Simulation target\n{:?}", sim.world);

        sim.queue.push_back(VecDeque::new());
        sim.world.initialize_redstone_states();
        sim.normalize_torches_on();
        sim.init();

        tracing::debug!("queue: {:?}", sim.queue);

        sim.fill_event_id();
        sim.run_inner(limits)?;
        sim.clear_transient_torch_burnout();

        Ok(sim)
    }

    fn new(world: &World, trace_limit: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            world: world.into(),
            cycle: 0,
            event_id_count: 0,
            soft_power_sources: HashSet::new(),
            hard_power_sources: HashSet::new(),
            torch_toggle_cycles: HashMap::new(),
            burned_out_torches: HashSet::new(),
            trace: Vec::new(),
            snapshots: Vec::new(),
            trace_limit,
        }
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
        self.change_state_inner(states, SimulationLimits::cycles(None))
    }

    pub fn change_state_with_max_cycles(
        &mut self,
        states: Vec<(Position, bool)>,
        max_cycles: usize,
    ) -> eyre::Result<()> {
        self.change_state_inner(states, SimulationLimits::cycles(Some(max_cycles)))
    }

    pub fn change_state_with_limits(
        &mut self,
        states: Vec<(Position, bool)>,
        max_cycles: usize,
        max_events: usize,
    ) -> eyre::Result<()> {
        self.change_state_inner(
            states,
            SimulationLimits {
                max_cycles: Some(max_cycles),
                max_events: Some(max_events),
            },
        )
    }

    fn change_state_inner(
        &mut self,
        states: Vec<(Position, bool)>,
        limits: SimulationLimits,
    ) -> eyre::Result<()> {
        self.queue.push_back(VecDeque::new());
        let mut changed = false;

        for (pos, value) in states {
            let BlockKind::Switch { is_on } = &mut self.world[pos].kind else {
                eyre::bail!("you can change only switch state!");
            };

            if value == *is_on {
                continue;
            }

            *is_on = value;
            changed = true;

            pos.forwards_except(self.world[pos].direction)
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
                    direction: pos_src.diff(pos),
                })
                .chain(|| -> Option<Event> {
                    let pos = pos.walk(self.world[pos].direction)?;

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

        self.run_inner(limits)?;
        if changed {
            self.enqueue_torch_reevaluations();
            self.fill_event_id();
            self.run_inner(limits)?;
            self.clear_transient_torch_burnout();
        }

        Ok(())
    }

    pub fn world(&self) -> &World3D {
        &self.world
    }

    pub fn trace(&self) -> &[SimulationTraceEntry] {
        &self.trace
    }

    pub fn snapshots(&self) -> &[SimulationSnapshot] {
        &self.snapshots
    }

    pub fn clear_trace(&mut self) {
        self.trace.clear();
        self.snapshots.clear();
    }

    fn clear_transient_torch_burnout(&mut self) {
        self.torch_toggle_cycles.clear();
        self.burned_out_torches.clear();
    }

    pub fn run(&mut self) -> eyre::Result<usize> {
        self.run_inner(SimulationLimits::cycles(None))
    }

    pub fn run_with_max_cycles(&mut self, max_cycles: usize) -> eyre::Result<usize> {
        self.run_inner(SimulationLimits::cycles(Some(max_cycles)))
    }

    fn run_inner(&mut self, limits: SimulationLimits) -> eyre::Result<usize> {
        let mut local_cycle = 0;
        let mut local_events = 0;

        while !self.queue.is_empty() {
            if limits
                .max_cycles
                .is_some_and(|max_cycles| local_cycle >= max_cycles)
            {
                eyre::bail!("simulation exceeded max cycle limit ({local_cycle})");
            }

            self.consume_events(limits.max_events, &mut local_events)?;
            local_cycle += 1;
            self.record_snapshot();
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

    fn normalize_torches_on(&mut self) {
        for pos in self.world.iter_pos() {
            if matches!(self.world[pos].kind, BlockKind::Torch { .. }) {
                self.world[pos].kind = BlockKind::Torch { is_on: true };
            }
        }
    }

    fn init(&mut self) {
        for (pos, block) in self.world.iter_block() {
            match block.kind {
                BlockKind::Torch { is_on } if is_on => {
                    self.init_torch_event(block.direction, pos);
                }
                BlockKind::Switch { is_on } if is_on => {
                    self.init_switch_event(block.direction, pos);
                }
                BlockKind::RedstoneBlock => {
                    self.init_redstone_block_event(pos);
                }
                _ => (),
            };
        }
    }

    fn init_torch_event(&mut self, dir: Direction, pos: Position) {
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
        .flat_map(|pos_src| {
            vec![Event {
                id: None,
                from_id: None,
                event_type: EventType::TorchOn,
                target_position: pos_src,
                direction: pos_src.diff(pos),
            }]
        })
        .chain(Some(Event {
            id: None,
            from_id: None,
            event_type: EventType::HardOn,
            target_position: pos.up(),
            direction: Direction::None,
        }));

        self.queue[0].extend(events);
    }

    fn enqueue_torch_reevaluations(&mut self) {
        let events = self
            .world
            .iter_block()
            .into_iter()
            .filter_map(|(pos, block)| {
                if !matches!(block.kind, BlockKind::Torch { .. }) {
                    return None;
                }
                let support = pos.walk(block.direction)?;
                if !self.world.size.bound_on(support) || !self.world[support].kind.is_cobble() {
                    return None;
                }
                let BlockKind::Cobble { on_count, .. } = self.world[support].kind else {
                    return None;
                };
                Some(Event {
                    id: None,
                    from_id: None,
                    event_type: if on_count > 0 {
                        EventType::SoftOn
                    } else {
                        EventType::SoftOff
                    },
                    target_position: pos,
                    direction: block.direction,
                })
            })
            .collect::<Vec<_>>();

        if events.is_empty() {
            return;
        }
        self.queue.push_back(events.into());
    }

    fn init_switch_event(&mut self, dir: Direction, pos: Position) {
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
                let pos = pos.walk(dir)?;

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

    fn init_redstone_block_event(&mut self, pos: Position) {
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

    fn consume_events(
        &mut self,
        max_events: Option<usize>,
        local_events: &mut usize,
    ) -> eyre::Result<()> {
        self.cycle += 1;

        self.queue.push_back(VecDeque::new());
        let mut seen_events = HashSet::new();
        while let Some(event) = self.queue.front_mut().unwrap().pop_front() {
            if !self.world.size.bound_on(event.target_position) {
                continue;
            }
            if !seen_events.insert(EventKey::from_event(&event)) {
                continue;
            }

            if max_events.is_some_and(|max_events| *local_events >= max_events) {
                eyre::bail!("simulation exceeded max event limit ({local_events})");
            }
            *local_events += 1;

            tracing::debug!("consume event: {:?}", event);

            let mut block = self.world[event.target_position];
            self.record_trace(&event, &block);

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
                BlockKind::Piston { .. } => todo!(),
            }

            self.world[event.target_position] = block;

            self.fill_event_id();
        }

        self.queue.pop_front();

        if self.queue.back().unwrap().is_empty() {
            self.queue.pop_back();
        }

        Ok(())
    }

    fn record_trace(&mut self, event: &Event, block: &Block) {
        if self.trace_limit == 0 {
            return;
        }

        if self.trace.len() == self.trace_limit {
            self.trace.remove(0);
        }

        self.trace.push(SimulationTraceEntry {
            cycle: self.cycle,
            event_id: event.id,
            from_event_id: event.from_id,
            event_type: format!("{:?}", event.event_type),
            target_position: [
                event.target_position.0,
                event.target_position.1,
                event.target_position.2,
            ],
            direction: format!("{:?}", event.direction),
            block_before: format!("{:?}", block.kind),
            current_queue_len: self.queue.front().map_or(0, VecDeque::len),
            next_queue_len: self.queue.back().map_or(0, VecDeque::len),
        });
    }

    fn record_snapshot(&mut self) {
        if self.trace_limit == 0 {
            return;
        }

        self.snapshots.push(SimulationSnapshot {
            cycle: self.cycle,
            world: self.world.clone(),
        });
    }

    fn propgate_cobble_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume cobble event: {:?}", block);

        if event.event_type.is_redstone() {
            return Ok(());
        }

        let is_hard = matches!(event.event_type, EventType::HardOn | EventType::HardOff);
        let source_position = event
            .target_position
            .walk(event.direction)
            .unwrap_or(event.target_position);
        let source_key = (event.target_position, source_position);
        let power_sources = if is_hard {
            &mut self.hard_power_sources
        } else {
            &mut self.soft_power_sources
        };

        if event.event_type.is_on() {
            if !power_sources.insert(source_key) {
                return Ok(());
            }

            block.count_up(event.event_type.is_hard())?;
        } else {
            let BlockKind::Cobble {
                on_count,
                on_base_count,
            } = block.kind
            else {
                unreachable!()
            };

            if !power_sources.remove(&source_key) {
                return Ok(());
            }

            if on_count == 0 || (is_hard && on_base_count == 0) {
                return Ok(());
            }

            block.count_down(event.event_type.is_hard())?;
        }

        let BlockKind::Cobble {
            on_count,
            on_base_count,
            ..
        } = block.kind
        else {
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
            .filter(|&pos| self.world.size.bound_on(pos))
            .filter(|&pos| !self.world[pos].kind.is_cobble())
            .map(|pos_src| Event {
                id: None,
                from_id: event.id,
                event_type: if is_hard && on_base_count == count_condition {
                    if event.event_type.is_on() {
                        EventType::HardOn
                    } else {
                        EventType::HardOff
                    }
                } else if event.event_type.is_on() {
                    EventType::SoftOn
                } else {
                    EventType::SoftOff
                },
                target_position: pos_src,
                direction: pos_src.diff(event.target_position),
            })
            .collect::<Vec<_>>();

        events.into_iter().for_each(|event| {
            self.push_event_to_current_tick(event);
        });

        Ok(())
    }

    fn propagate_redstone_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume redstone event: {:?}", block);

        let BlockKind::Redstone { state, .. } = &mut block.kind else {
            eyre::bail!("unreachable");
        };

        let mut propagate_targets = Vec::new();

        propagate_targets.extend(event.target_position.cardinal_redstone(*state));

        let up_pos = event.target_position.up();
        if self.world.size.bound_on(up_pos) && !self.world[up_pos].kind.is_cobble() {
            propagate_targets.extend(up_pos.cardinal_redstone(*state).into_iter().filter(|&pos| {
                self.world.size.bound_on(pos) && self.world[pos].kind.is_redstone()
            }));
        }

        if let Some(down_pos) = event.target_position.down() {
            if !self.world[down_pos].kind.is_cobble() {
                eyre::bail!("unreachable");
            }

            propagate_targets.push(down_pos);

            propagate_targets.extend(
                event
                    .target_position
                    .cardinal_redstone(*state)
                    .into_iter()
                    .filter(|&pos| self.world.size.bound_on(pos))
                    .filter(|&pos| !self.world[pos].kind.is_cobble())
                    .filter_map(|pos| pos.walk(Direction::Bottom))
                    .filter(|&pos| {
                        self.world.size.bound_on(pos) && self.world[pos].kind.is_redstone()
                    }),
            );
        }

        propagate_targets.retain(|&pos| self.world.size.bound_on(pos));

        match event.event_type {
            EventType::SoftOn
            | EventType::SoftOff
            | EventType::RepeaterOn { .. }
            | EventType::RepeaterOff { .. } => {}
            EventType::TorchOn | EventType::HardOn => {
                let BlockKind::Redstone {
                    on_count, strength, ..
                } = &mut block.kind
                else {
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
                            direction: pos.diff(event.target_position),
                        });

                        if self.world[pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::RedstoneOn { strength: 14 },
                                target_position: pos,
                                direction: Direction::None,
                            });
                        }
                    });
                }
            }
            EventType::TorchOff | EventType::HardOff => {
                let BlockKind::Redstone {
                    on_count, strength, ..
                } = &mut block.kind
                else {
                    eyre::bail!("unreachable");
                };

                if *on_count == 0 {
                    return Ok(());
                }

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
                            direction: pos.diff(event.target_position),
                        });

                        if self.world[pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::RedstoneOff,
                                target_position: pos,
                                direction: Direction::None,
                            });
                        }
                    });
                }
            }
            EventType::RedstoneOn { strength } => {
                let event_strength = strength;

                if event.direction != Direction::None {
                    let Some(source_pos) = event.target_position.walk(event.direction) else {
                        return Ok(());
                    };
                    if let BlockKind::Redstone {
                        strength: source_strength,
                        ..
                    } = self.world[source_pos].kind
                    {
                        if source_strength <= event_strength {
                            return Ok(());
                        }
                    }
                }

                let BlockKind::Redstone { strength, .. } = &mut block.kind else {
                    eyre::bail!("unreachable");
                };

                if event_strength > 0 && event_strength > *strength {
                    *strength = event_strength;

                    propagate_targets.into_iter().for_each(|pos| {
                        if event.direction != Direction::None
                            && event.target_position.walk(event.direction) == Some(pos)
                        {
                            return;
                        }

                        if !self.world[pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::SoftOn,
                                target_position: pos,
                                direction: pos.diff(event.target_position),
                            });
                        }

                        if self.world[pos].kind.is_redstone() && *strength > 1 {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::RedstoneOn {
                                    strength: *strength - 1,
                                },
                                target_position: pos,
                                direction: pos.diff(event.target_position),
                            });
                        }
                    });

                    tracing::info!("trigger redstone event: {event:?}, {block:?}");
                }
            }
            EventType::RedstoneOff => {
                if event.direction != Direction::None {
                    let Some(source_pos) = event.target_position.walk(event.direction) else {
                        return Ok(());
                    };
                    if let BlockKind::Redstone {
                        strength: source_strength,
                        ..
                    } = self.world[source_pos].kind
                    {
                        if source_strength > 0 {
                            return Ok(());
                        }
                    }
                }

                let BlockKind::Redstone {
                    on_count, strength, ..
                } = &mut block.kind
                else {
                    eyre::bail!("unreachable");
                };

                if *strength == 0 {
                    return Ok(());
                }

                if *on_count == 0 {
                    *strength = 0;

                    propagate_targets.into_iter().for_each(|pos| {
                        if event.direction != Direction::None
                            && event.target_position.walk(event.direction) == Some(pos)
                        {
                            return;
                        }

                        if !self.world[pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::SoftOff,
                                target_position: pos,
                                direction: pos.diff(event.target_position),
                            });
                        }

                        if self.world[pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::RedstoneOff,
                                target_position: pos,
                                direction: pos.diff(event.target_position),
                            });
                        }
                    });
                } else {
                    propagate_targets.into_iter().for_each(|pos| {
                        if !self.world[pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::SoftOn,
                                target_position: pos,
                                direction: pos.diff(event.target_position),
                            });
                        }

                        if self.world[pos].kind.is_redstone() {
                            self.push_event_to_current_tick(Event {
                                id: None,
                                from_id: event.id,
                                event_type: EventType::RedstoneOn { strength: 14 },
                                target_position: pos,
                                direction: pos.diff(event.target_position),
                            });
                        }
                    });
                }

                tracing::info!("trigger redstone event: {event:?}, {block:?}");
            }
        };

        Ok(())
    }

    fn propgate_torch_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume torch event: {:?}", block);

        let Some(support_position) = event.target_position.walk(event.direction) else {
            return Ok(());
        };
        if !self.world[support_position].kind.is_cobble() {
            return Ok(());
        }

        // Cobble에 붙어있지 않은 경우
        if event.direction != block.direction {
            return Ok(());
        }

        let BlockKind::Torch { is_on } = &mut block.kind else {
            eyre::bail!("unreachable");
        };

        let next_is_on = !event.event_type.is_on();
        if *is_on == next_is_on {
            return Ok(());
        }
        if next_is_on && self.burned_out_torches.contains(&event.target_position) {
            if self.torch_is_still_burned_out(event.target_position) {
                return Ok(());
            }
            self.burned_out_torches.remove(&event.target_position);
        }
        let burned_out = self.record_torch_toggle(event.target_position);
        if burned_out {
            self.burned_out_torches.insert(event.target_position);
            if !*is_on {
                return Ok(());
            }
            *is_on = false;
        } else {
            *is_on = next_is_on;
        }

        match block.direction {
            Direction::Bottom => event.target_position.cardinal(),
            Direction::East | Direction::West | Direction::South | Direction::North => {
                let mut positions = event.target_position.cardinal_except(block.direction);
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
            direction: pos_src.diff(event.target_position),
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

    fn record_torch_toggle(&mut self, position: Position) -> bool {
        self.prune_torch_toggle_history(position);
        let history = self.torch_toggle_cycles.entry(position).or_default();
        history.push_back(self.cycle);
        history.len() >= TORCH_BURNOUT_TOGGLE_LIMIT
    }

    fn torch_is_still_burned_out(&mut self, position: Position) -> bool {
        self.prune_torch_toggle_history(position);
        self.torch_toggle_cycles
            .get(&position)
            .is_some_and(|history| history.len() >= TORCH_BURNOUT_TOGGLE_LIMIT)
    }

    fn prune_torch_toggle_history(&mut self, position: Position) {
        let Some(history) = self.torch_toggle_cycles.get_mut(&position) else {
            return;
        };
        while history
            .front()
            .is_some_and(|cycle| self.cycle.saturating_sub(*cycle) > TORCH_BURNOUT_WINDOW_CYCLES)
        {
            history.pop_front();
        }
    }

    fn propgate_repeater_event(&mut self, block: &mut Block, event: &Event) -> eyre::Result<()> {
        tracing::debug!("consume repeater event: {:?}", block);

        let BlockKind::Repeater {
            is_on,
            is_locked,
            delay,
            ..
        } = block.kind
        else {
            unreachable!()
        };

        let lock_signal = event.direction.is_othogonal_plane(block.direction);

        if event.direction != block.direction && !lock_signal {
            return Ok(());
        }

        if event.direction == block.direction && is_locked {
            return Ok(());
        }

        // TODO: Consider block off-pulse input lower by setting delay
        match event.event_type {
            EventType::SoftOn
            | EventType::HardOn
            | EventType::TorchOn
            | EventType::RedstoneOn { .. } => {
                if !is_on {
                    if lock_signal {
                        block.kind.set_repeater_lock(true)?;

                        tracing::info!("trigger repeater event: {event:?}, {block:?}");
                    } else {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::RepeaterOn { delay },
                            target_position: event.target_position,
                            direction: event.direction,
                        });
                    }
                }
            }
            EventType::SoftOff
            | EventType::HardOff
            | EventType::TorchOff
            | EventType::RedstoneOff => {
                if is_on {
                    if lock_signal {
                        block.kind.set_repeater_lock(false)?;

                        tracing::info!("trigger repeater event: {event:?}, {block:?}");
                    } else {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::RepeaterOff { delay },
                            target_position: event.target_position,
                            direction: event.direction,
                        });
                    }
                }
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
                } else {
                    block.kind.set_repeater_state(true)?;

                    // propagate
                    let walk = event.target_position.walk(event.direction.inverse());

                    if let Some(pos) = walk {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::HardOn,
                            target_position: pos,
                            direction: pos.diff(event.target_position),
                        });
                    }

                    if let Some(pos) = walk.and_then(|pos| pos.down()) {
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
                } else {
                    block.kind.set_repeater_state(false)?;

                    let walk = event.target_position.walk(event.direction.inverse());

                    if let Some(pos) = walk {
                        self.push_event_to_next_tick(Event {
                            id: None,
                            from_id: event.id,
                            event_type: EventType::HardOff,
                            target_position: pos,
                            direction: pos.diff(event.target_position),
                        });
                    }

                    if let Some(pos) = walk.and_then(|pos| pos.down()) {
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
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::nbt::NBTRoot;
    use crate::sequential::layout::SequentialMacro;
    use crate::sequential::SequentialPrimitive;
    use crate::world::block::RedstoneState;
    use crate::world::position::DimSize;

    #[test]
    pub fn unittest_simulator_init_states() {
        let _ = tracing_subscriber::fmt::try_init();

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
                (Position(1, 1, 0), default_restone),
                (Position(0, 1, 0), default_restone),
                (Position(1, 0, 0), default_restone),
                (
                    Position(1, 2, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: true },
                        direction: Direction::Top,
                    },
                ),
            ],
        };

        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.world.initialize_redstone_states();

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
        let _ = tracing_subscriber::fmt::try_init();

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
                (Position(1, 1, 0), default_restone),
                (Position(0, 1, 0), default_restone),
                (Position(1, 0, 0), default_restone),
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

        let BlockKind::Redstone { on_count, .. } = sim.world.map[0][1][1].kind else {
            unreachable!();
        };

        assert_eq!(on_count, 1);
    }

    #[test]
    fn unittest_simulator_cobble() {
        let _ = tracing_subscriber::fmt::try_init();

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
                (Position(1, 1, 0), default_cobble),
                (Position(0, 1, 0), default_restone),
                (Position(1, 0, 0), default_restone),
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

        let BlockKind::Cobble {
            on_count,
            on_base_count,
        } = sim.world.map[0][1][1].kind
        else {
            unreachable!();
        };

        assert_eq!(on_count, 1);
        assert_eq!(on_base_count, 0);
    }

    #[test]
    fn unittest_simulator_cobble_deduplicates_same_soft_power_source() -> eyre::Result<()> {
        let target = Position(1, 1, 0);
        let mock_world = World {
            size: DimSize(3, 3, 2),
            blocks: vec![(
                target,
                Block {
                    kind: BlockKind::Cobble {
                        on_count: 0,
                        on_base_count: 0,
                    },
                    direction: Direction::None,
                },
            )],
        };
        let event = Event {
            id: None,
            from_id: None,
            event_type: EventType::SoftOn,
            target_position: target,
            direction: Direction::South,
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];

        sim.propgate_cobble_event(&mut block, &event)?;
        sim.propgate_cobble_event(&mut block, &event)?;

        assert!(matches!(
            block.kind,
            BlockKind::Cobble {
                on_count: 1,
                on_base_count: 0
            }
        ));

        let event = Event {
            event_type: EventType::SoftOff,
            ..event
        };
        sim.propgate_cobble_event(&mut block, &event)?;
        sim.propgate_cobble_event(&mut block, &event)?;

        assert!(matches!(
            block.kind,
            BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0
            }
        ));

        Ok(())
    }

    #[test]
    fn unittest_simulator_cobble_event_ignores_out_of_bounds_neighbors() -> eyre::Result<()> {
        let target = Position(1, 1, 0);
        let mock_world = World {
            size: DimSize(2, 2, 1),
            blocks: vec![(
                target,
                Block {
                    kind: BlockKind::Cobble {
                        on_count: 0,
                        on_base_count: 0,
                    },
                    direction: Direction::None,
                },
            )],
        };
        let event = Event {
            id: None,
            from_id: None,
            event_type: EventType::SoftOn,
            target_position: target,
            direction: Direction::South,
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.propgate_cobble_event(&mut block, &event)
        }));

        assert!(
            result.is_ok(),
            "cobble event should not panic at world edge"
        );
        result.unwrap()?;

        Ok(())
    }

    #[test]
    fn unittest_simulator_ignores_out_of_bounds_events() -> eyre::Result<()> {
        let mock_world = World {
            size: DimSize(1, 1, 1),
            blocks: Vec::new(),
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::from([Event {
            id: None,
            from_id: None,
            event_type: EventType::SoftOn,
            target_position: Position(1, 0, 0),
            direction: Direction::East,
        }]));

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.run_with_max_cycles(1)));

        assert!(
            result.is_ok(),
            "simulator should not panic on out-of-bounds events"
        );
        result.unwrap()?;

        Ok(())
    }

    #[test]
    fn unittest_simulator_redstone_event_ignores_out_of_bounds_neighbors() -> eyre::Result<()> {
        let target = Position(1, 0, 0);
        let mock_world = World {
            size: DimSize(2, 2, 1),
            blocks: vec![(
                target,
                Block {
                    kind: BlockKind::Redstone {
                        state: Default::default(),
                        on_count: 0,
                        strength: 0,
                    },
                    direction: Direction::None,
                },
            )],
        };
        let event = Event {
            id: None,
            from_id: None,
            event_type: EventType::TorchOn,
            target_position: target,
            direction: Direction::East,
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.propagate_redstone_event(&mut block, &event)
        }));

        assert!(
            result.is_ok(),
            "redstone event should not panic at world edge"
        );
        result.unwrap()?;

        Ok(())
    }

    #[test]
    fn unittest_simulator_uninitialized_rs_latch_does_not_exceed_event_limit() {
        let primitive = SequentialPrimitive::rs_latch();
        let candidate = SequentialMacro::candidates(&primitive)
            .into_iter()
            .next()
            .unwrap();
        let world = World::from(&candidate.world);

        let result = Simulator::from_with_limits_and_trace(&world, 128, 50_000, 0);

        assert!(
            result.is_ok(),
            "uninitialized RS latch should settle or burn out instead of running forever: {:?}",
            result.err().map(|error| error.message().to_owned())
        );
    }

    #[test]
    fn unittest_simulator_full_adder_all_on_recomputes_wall_torch() -> eyre::Result<()> {
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read("test/full-adder.nbt")?)?;
        let world = nbt.to_world();
        let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        let states = world
            .blocks
            .iter()
            .filter_map(|(pos, block)| {
                matches!(block.kind, BlockKind::Switch { .. }).then_some((*pos, true))
            })
            .collect::<Vec<_>>();

        sim.change_state_with_limits(states, 256, 50_000)?;

        assert!(matches!(
            sim.world[Position(8, 2, 3)].kind,
            BlockKind::Torch { is_on: true }
        ));
        assert!(matches!(
            sim.world[Position(7, 2, 3)].kind,
            BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0
            }
        ));

        Ok(())
    }

    #[test]
    fn unittest_simulator_xor_generated_truth_table() -> eyre::Result<()> {
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read("test/xor-generated.nbt")?)?;
        let world = nbt.to_world();
        let switches = [Position(0, 6, 0), Position(0, 6, 3)];
        let output = Position(4, 7, 2);

        for mask in 0..4 {
            let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
            sim.change_state_with_limits(
                switches
                    .iter()
                    .enumerate()
                    .map(|(index, pos)| (*pos, (mask & (1 << index)) != 0))
                    .collect(),
                256,
                50_000,
            )?;

            let BlockKind::Redstone { strength, .. } = sim.world[output].kind else {
                panic!("xor output should be redstone");
            };
            assert_eq!(
                strength > 0,
                mask == 1 || mask == 2,
                "xor-generated output mismatch for mask {mask:02b}"
            );
        }

        Ok(())
    }

    #[test]
    fn unittest_simulator_full_adder_toggle_does_not_burn_out_output_torch() -> eyre::Result<()> {
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read("test/full-adder.nbt")?)?;
        let world = nbt.to_world();
        let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 50_000)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        sim.change_state_with_limits(vec![(Position(0, 5, 1), true)], 256, 50_000)?;
        sim.change_state_with_limits(vec![(Position(0, 7, 2), true)], 256, 50_000)?;
        sim.change_state_with_limits(vec![(Position(2, 0, 3), true)], 256, 50_000)?;
        sim.clear_trace();
        sim.change_state_with_limits(vec![(Position(0, 7, 2), false)], 256, 50_000)?;

        assert!(
            sim.trace().len() < 1_000,
            "full-adder final toggle should settle without long torch flicker, got {} events",
            sim.trace().len()
        );

        Ok(())
    }

    #[test]
    fn unittest_simulator_repeater() {
        let _ = tracing_subscriber::fmt::try_init();

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
                lock_input1: None,
                lock_input2: None,
            },
            direction: Direction::South,
        };

        let mock_world = World {
            size: DimSize(4, 4, 2),
            blocks: vec![
                (Position(1, 2, 0), default_restone),
                (Position(1, 1, 0), default_repeater),
                (Position(0, 2, 0), default_restone),
                (Position(2, 2, 0), default_restone),
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

        let BlockKind::Redstone { strength, .. } = sim.world.map[0][2][2].kind else {
            unreachable!();
        };

        assert_eq!(strength, 14)
    }

    #[test]
    pub fn unittest_simulator_torch() {
        let _ = tracing_subscriber::fmt::try_init();

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
                (Position(0, 1, 0), default_restone),
                (Position(0, 2, 0), default_cobble),
                (
                    Position(1, 2, 0),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::West,
                    },
                ),
                (Position(2, 2, 0), default_restone),
                (Position(3, 2, 0), default_cobble),
                (
                    Position(4, 2, 0),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::West,
                    },
                ),
                (Position(5, 2, 0), default_restone),
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
        let _ = tracing_subscriber::fmt::try_init();

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
                (Position(0, 1, 0), default_restone),
                (Position(2, 1, 0), default_restone),
                (Position(0, 2, 0), default_cobble),
                (Position(1, 2, 0), default_cobble),
                (Position(2, 2, 0), default_cobble),
                (Position(1, 2, 1), default_restone),
                (Position(1, 4, 0), default_restone),
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
