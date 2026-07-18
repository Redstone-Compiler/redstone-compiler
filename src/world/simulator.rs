use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

use super::block::{Block, BlockKind, Direction};
use super::position::Position;
use super::{World, World3D};

const DEFAULT_TRACE_LIMIT: usize = 0;
// Approximate Minecraft redstone torch burnout so feedback loops can settle
// instead of producing simulator events forever. These are simulator cycles,
// not exact game ticks or redstone ticks.
const TORCH_BURNOUT_WINDOW_CYCLES: usize = 60;
const TORCH_BURNOUT_TOGGLE_LIMIT: usize = 8;
pub const MANUAL_INPUT_IDLE_CYCLES: usize = TORCH_BURNOUT_WINDOW_CYCLES + 1;
// Torch support changes are evaluated after a small simulator delay, then
// rechecked at application time so short transient power does not force a
// stale torch state transition.
const TORCH_UPDATE_DELAY_CYCLES: usize = 1;

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

#[derive(Copy, Clone, Debug)]
struct CobblePowerInput {
    source: Position,
    hard: bool,
}

#[derive(Clone, Debug)]
pub struct Simulator {
    queue: VecDeque<VecDeque<Event>>,
    world: World3D,
    redstone_positions: Vec<Position>,
    cobble_positions: Vec<Position>,
    torch_positions: Vec<Position>,
    power_source_positions: Vec<Position>,
    redstone_inputs: Vec<Vec<Position>>,
    cobble_power_inputs: Vec<Vec<CobblePowerInput>>,
    cycle: usize,
    event_id_count: usize,
    soft_power_sources: HashSet<(Position, Position)>,
    hard_power_sources: HashSet<(Position, Position)>,
    redstone_power_sources: HashSet<(Position, Position)>,
    torch_toggle_cycles: HashMap<Position, VecDeque<usize>>,
    burned_out_torches: HashSet<Position>,
    trace: Vec<SimulationTraceEntry>,
    snapshots: Vec<SimulationSnapshot>,
    trace_limit: usize,
    profile: Option<SimulationProfile>,
}

#[derive(Clone, Debug, Default)]
pub struct SimulationProfile {
    pub fill_event_ids_time: Duration,
    pub event_processing_time: Duration,
    pub redstone_normalization_time: Duration,
    pub cobble_normalization_time: Duration,
    pub torch_reevaluation_time: Duration,
    pub fill_event_ids_calls: usize,
    pub event_batches: usize,
    pub events_processed: usize,
    pub redstone_normalization_calls: usize,
    pub redstone_relaxation_passes: usize,
    pub redstone_targets_evaluated: usize,
    pub cobble_normalization_calls: usize,
    pub cobble_targets_evaluated: usize,
    pub torch_reevaluation_calls: usize,
    pub torches_evaluated: usize,
}

impl SimulationProfile {
    pub fn measured_time(&self) -> Duration {
        self.event_processing_time
            + self.redstone_normalization_time
            + self.cobble_normalization_time
            + self.torch_reevaluation_time
    }

    #[cfg(test)]
    fn accumulate(&mut self, other: &Self) {
        self.fill_event_ids_time += other.fill_event_ids_time;
        self.event_processing_time += other.event_processing_time;
        self.redstone_normalization_time += other.redstone_normalization_time;
        self.cobble_normalization_time += other.cobble_normalization_time;
        self.torch_reevaluation_time += other.torch_reevaluation_time;
        self.fill_event_ids_calls += other.fill_event_ids_calls;
        self.event_batches += other.event_batches;
        self.events_processed += other.events_processed;
        self.redstone_normalization_calls += other.redstone_normalization_calls;
        self.redstone_relaxation_passes += other.redstone_relaxation_passes;
        self.redstone_targets_evaluated += other.redstone_targets_evaluated;
        self.cobble_normalization_calls += other.cobble_normalization_calls;
        self.cobble_targets_evaluated += other.cobble_targets_evaluated;
        self.torch_reevaluation_calls += other.torch_reevaluation_calls;
        self.torches_evaluated += other.torches_evaluated;
    }
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

#[derive(Clone, Debug, serde::Serialize, PartialEq, Eq)]
pub struct SimulationWaveform {
    pub cycles: Vec<usize>,
    pub signals: Vec<SimulationWaveformSignal>,
}

#[derive(Clone, Debug, serde::Serialize, PartialEq, Eq)]
pub struct SimulationWaveformSignal {
    pub position: [usize; 3],
    pub kind: String,
    pub property: String,
    pub label: String,
    pub max_value: usize,
    pub values: Vec<usize>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct WaveformSignalDescriptor {
    position: Position,
    kind: &'static str,
    property: &'static str,
    max_value: usize,
}

impl WaveformSignalDescriptor {
    fn sort_key(&self) -> (usize, Position, &'static str) {
        (
            waveform_signal_kind_order(self.kind),
            self.position,
            self.property,
        )
    }
}

impl SimulationWaveform {
    pub fn from_snapshots(snapshots: &[SimulationSnapshot]) -> Self {
        let mut descriptors = snapshots
            .iter()
            .flat_map(|snapshot| waveform_signal_descriptors(&snapshot.world))
            .collect::<Vec<_>>();
        descriptors.sort_by_key(WaveformSignalDescriptor::sort_key);
        descriptors.dedup();

        Self {
            cycles: snapshots.iter().map(|snapshot| snapshot.cycle).collect(),
            signals: descriptors
                .into_iter()
                .map(|descriptor| SimulationWaveformSignal {
                    position: [
                        descriptor.position.0,
                        descriptor.position.1,
                        descriptor.position.2,
                    ],
                    kind: descriptor.kind.to_owned(),
                    property: descriptor.property.to_owned(),
                    label: waveform_signal_label(descriptor),
                    max_value: descriptor.max_value,
                    values: snapshots
                        .iter()
                        .map(|snapshot| waveform_signal_value(&snapshot.world, descriptor))
                        .collect(),
                })
                .collect(),
        }
    }
}

fn waveform_signal_descriptors(world: &World3D) -> Vec<WaveformSignalDescriptor> {
    let mut descriptors = world
        .iter_block()
        .into_iter()
        .flat_map(|(position, block)| match block.kind {
            BlockKind::Switch { .. } => vec![WaveformSignalDescriptor {
                position,
                kind: "switch",
                property: "powered",
                max_value: 1,
            }],
            BlockKind::Torch { .. } => vec![WaveformSignalDescriptor {
                position,
                kind: "torch",
                property: "lit",
                max_value: 1,
            }],
            BlockKind::Repeater { .. } => vec![
                WaveformSignalDescriptor {
                    position,
                    kind: "repeater",
                    property: "powered",
                    max_value: 1,
                },
                WaveformSignalDescriptor {
                    position,
                    kind: "repeater",
                    property: "locked",
                    max_value: 1,
                },
            ],
            BlockKind::Air
            | BlockKind::Cobble { .. }
            | BlockKind::Redstone { .. }
            | BlockKind::RedstoneBlock
            | BlockKind::Piston { .. } => Vec::new(),
        })
        .collect::<Vec<_>>();
    descriptors.sort_by_key(WaveformSignalDescriptor::sort_key);
    descriptors
}

fn waveform_signal_kind_order(kind: &str) -> usize {
    match kind {
        "switch" => 0,
        "repeater" => 1,
        "torch" => 2,
        _ => 5,
    }
}

fn waveform_signal_value(world: &World3D, descriptor: WaveformSignalDescriptor) -> usize {
    if !world.size.bound_on(descriptor.position) {
        return 0;
    }

    match (
        world[descriptor.position].kind,
        descriptor.kind,
        descriptor.property,
    ) {
        (BlockKind::Cobble { on_count, .. }, "cobble", "powered") => usize::from(on_count > 0),
        (BlockKind::Switch { is_on }, "switch", "powered") => usize::from(is_on),
        (BlockKind::Redstone { strength, .. }, "redstone", "power") => strength,
        (BlockKind::Torch { is_on }, "torch", "lit") => usize::from(is_on),
        (BlockKind::Repeater { is_on, .. }, "repeater", "powered") => usize::from(is_on),
        (BlockKind::Repeater { is_locked, .. }, "repeater", "locked") => usize::from(is_locked),
        _ => 0,
    }
}

fn waveform_signal_label(descriptor: WaveformSignalDescriptor) -> String {
    format!(
        "{}.{} {},{},{}",
        descriptor.kind,
        descriptor.property,
        descriptor.position.0,
        descriptor.position.1,
        descriptor.position.2
    )
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

    pub fn waveform(&self) -> SimulationWaveform {
        SimulationWaveform::from_snapshots(&self.snapshots)
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

        tracing::trace!("Simulation target\n{:?}", sim.world);

        sim.queue.push_back(VecDeque::new());
        sim.world.initialize_redstone_states();
        sim.rebuild_connectivity_cache();
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

        Ok(sim)
    }

    pub fn from_preserving_torch_states_with_limits_and_trace(
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

        tracing::trace!("Simulation target\n{:?}", sim.world);

        sim.queue.push_back(VecDeque::new());
        sim.world.initialize_redstone_states();
        sim.rebuild_connectivity_cache();
        sim.init();
        sim.enqueue_torch_reevaluations();

        tracing::debug!("queue: {:?}", sim.queue);

        sim.fill_event_id();

        if let Err(error) = sim.run_inner(limits) {
            return Err(SimulationTraceError {
                message: error.to_string(),
                trace: sim.trace,
                snapshots: sim.snapshots,
            });
        }

        Ok(sim)
    }

    fn from_inner(
        world: &World,
        limits: SimulationLimits,
        trace_limit: usize,
    ) -> eyre::Result<Self> {
        let mut sim = Self::new(world, trace_limit);

        tracing::trace!("Simulation target\n{:?}", sim.world);

        sim.queue.push_back(VecDeque::new());
        sim.world.initialize_redstone_states();
        sim.rebuild_connectivity_cache();
        sim.normalize_torches_on();
        sim.init();

        tracing::debug!("queue: {:?}", sim.queue);

        sim.fill_event_id();
        sim.run_inner(limits)?;

        Ok(sim)
    }

    fn new(world: &World, trace_limit: usize) -> Self {
        let world = World3D::from(world);
        // Signal state changes during simulation, but block kinds do not, so
        // these position sets remain valid for the simulator's lifetime.
        let mut redstone_positions = Vec::new();
        let mut cobble_positions = Vec::new();
        let mut torch_positions = Vec::new();
        let mut power_source_positions = Vec::new();
        for (position, block) in world.iter_block() {
            if block.kind.is_redstone() {
                redstone_positions.push(position);
            }
            if block.kind.is_cobble() {
                cobble_positions.push(position);
            }
            if block.kind.is_torch() {
                torch_positions.push(position);
            }
            if matches!(
                block.kind,
                BlockKind::Torch { .. }
                    | BlockKind::Switch { .. }
                    | BlockKind::Redstone { .. }
                    | BlockKind::RedstoneBlock
                    | BlockKind::Repeater { .. }
            ) {
                power_source_positions.push(position);
            }
        }
        let volume = world.size.0 * world.size.1 * world.size.2;
        let mut sim = Self {
            queue: VecDeque::new(),
            world,
            redstone_positions,
            cobble_positions,
            torch_positions,
            power_source_positions,
            redstone_inputs: vec![Vec::new(); volume],
            cobble_power_inputs: vec![Vec::new(); volume],
            cycle: 0,
            event_id_count: 0,
            soft_power_sources: HashSet::new(),
            hard_power_sources: HashSet::new(),
            redstone_power_sources: HashSet::new(),
            torch_toggle_cycles: HashMap::new(),
            burned_out_torches: HashSet::new(),
            trace: Vec::new(),
            snapshots: Vec::new(),
            trace_limit,
            profile: None,
        };
        sim.rebuild_connectivity_cache();
        sim
    }

    fn rebuild_connectivity_cache(&mut self) {
        let volume = self.world.size.0 * self.world.size.1 * self.world.size.2;
        let mut redstone_inputs = vec![Vec::new(); volume];
        for source in self.redstone_positions.iter().copied() {
            let BlockKind::Redstone { state, .. } = self.world[source].kind else {
                continue;
            };
            for target in self.redstone_propagate_targets(source, state) {
                if !self.world.size.bound_on(target) || !self.world[target].kind.is_redstone() {
                    continue;
                }
                let inputs = &mut redstone_inputs[target.index(&self.world.size).0];
                if !inputs.contains(&source) {
                    inputs.push(source);
                }
            }
        }

        let mut cobble_power_inputs = vec![Vec::<CobblePowerInput>::new(); volume];
        for source in self.power_source_positions.iter().copied() {
            let source_block = self.world[source];
            let mut targets = Vec::new();
            match source_block.kind {
                BlockKind::Torch { .. } => {
                    let soft_targets = match source_block.direction {
                        Direction::Bottom => source.cardinal(),
                        Direction::East | Direction::West | Direction::South | Direction::North => {
                            let mut positions = source.cardinal_except(source_block.direction);
                            positions.extend(source.down());
                            positions
                        }
                        _ => Vec::new(),
                    };
                    targets.extend(soft_targets.into_iter().map(|target| (target, false)));
                    targets.push((source.up(), true));
                }
                BlockKind::Switch { .. } => {
                    targets.extend(
                        source
                            .forwards_except(source_block.direction)
                            .into_iter()
                            .map(|target| (target, false)),
                    );
                    if let Some(target) = source.walk(source_block.direction) {
                        targets.push((target, true));
                    }
                }
                BlockKind::Redstone { state, .. } => {
                    targets.extend(
                        self.redstone_propagate_targets(source, state)
                            .into_iter()
                            .map(|target| (target, false)),
                    );
                }
                BlockKind::RedstoneBlock => {
                    targets.extend(source.forwards().into_iter().map(|target| (target, false)));
                }
                BlockKind::Repeater { .. } => {
                    if let Some(target) = source.walk(source_block.direction.inverse()) {
                        targets.push((target, true));
                    }
                }
                _ => {}
            }

            for (target, hard) in targets {
                if !self.world.size.bound_on(target) || !self.world[target].kind.is_cobble() {
                    continue;
                }
                let inputs = &mut cobble_power_inputs[target.index(&self.world.size).0];
                if let Some(existing) = inputs.iter_mut().find(|input| input.source == source) {
                    existing.hard |= hard;
                } else {
                    inputs.push(CobblePowerInput { source, hard });
                }
            }
        }

        self.redstone_inputs = redstone_inputs;
        self.cobble_power_inputs = cobble_power_inputs;
    }

    pub fn set_profiling_enabled(&mut self, enabled: bool) {
        self.profile = enabled.then(SimulationProfile::default);
    }

    pub fn reset_profile(&mut self) {
        if let Some(profile) = &mut self.profile {
            *profile = SimulationProfile::default();
        }
    }

    pub fn profile(&self) -> Option<&SimulationProfile> {
        self.profile.as_ref()
    }

    fn fill_event_id(&mut self) {
        let started = self.profile.as_ref().map(|_| Instant::now());
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
        if let (Some(profile), Some(started)) = (&mut self.profile, started) {
            profile.fill_event_ids_time += started.elapsed();
            profile.fill_event_ids_calls += 1;
        }
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
        }

        Ok(())
    }

    pub fn world(&self) -> &World3D {
        &self.world
    }

    pub fn advance_idle_cycles(&mut self, cycles: usize) -> eyre::Result<()> {
        eyre::ensure!(
            self.queue.is_empty(),
            "cannot advance idle time while simulator events are pending"
        );
        self.cycle = self.cycle.saturating_add(cycles);
        Ok(())
    }

    pub fn trace(&self) -> &[SimulationTraceEntry] {
        &self.trace
    }

    pub fn snapshots(&self) -> &[SimulationSnapshot] {
        &self.snapshots
    }

    pub fn waveform(&self) -> SimulationWaveform {
        SimulationWaveform::from_snapshots(&self.snapshots)
    }

    pub fn clear_trace(&mut self) {
        self.trace.clear();
        self.snapshots.clear();
    }

    pub fn set_trace_limit(&mut self, trace_limit: usize) {
        self.trace_limit = trace_limit;
        if trace_limit == 0 {
            self.clear_trace();
        }
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

        loop {
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
                tracing::debug!("simulator cycle: {local_cycle}/{}", self.cycle);
            }

            if !self.normalize_signal_levels() {
                break;
            }
            self.enqueue_torch_reevaluations();
            self.fill_event_id();
            if self.queue.is_empty() {
                break;
            }
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

    fn schedule_event(&mut self, delay_cycles: usize, event: Event) {
        while self.queue.len() <= delay_cycles {
            self.queue.push_back(VecDeque::new());
        }
        self.queue[delay_cycles].push_back(event);
    }

    fn normalize_torches_on(&mut self) {
        for pos in self.torch_positions.iter().copied() {
            self.world[pos].kind = BlockKind::Torch { is_on: true };
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
        let started = self.profile.as_ref().map(|_| Instant::now());
        let mut torches_evaluated = 0;
        let events = self
            .torch_positions
            .iter()
            .copied()
            .filter_map(|pos| {
                let block = self.world[pos];
                torches_evaluated += 1;
                let support = pos.walk(block.direction)?;
                if !self.world.size.bound_on(support) || !self.world[support].kind.is_cobble() {
                    return None;
                }
                let support_is_powered = self.cobble_power_counts(support).0 > 0;
                Some(Event {
                    id: None,
                    from_id: None,
                    event_type: if support_is_powered {
                        EventType::SoftOn
                    } else {
                        EventType::SoftOff
                    },
                    target_position: pos,
                    direction: block.direction,
                })
            })
            .collect::<Vec<_>>();

        for event in events {
            self.schedule_event(TORCH_UPDATE_DELAY_CYCLES, event);
        }
        if let (Some(profile), Some(started)) = (&mut self.profile, started) {
            profile.torch_reevaluation_time += started.elapsed();
            profile.torch_reevaluation_calls += 1;
            profile.torches_evaluated += torches_evaluated;
        }
    }

    fn redstone_propagate_targets(&self, pos: Position, state: usize) -> Vec<Position> {
        let mut propagate_targets = Vec::new();

        propagate_targets.extend(pos.cardinal_redstone(state));

        let up_pos = pos.up();
        if self.world.size.bound_on(up_pos) && !self.world[up_pos].kind.is_cobble() {
            propagate_targets.extend(up_pos.cardinal_redstone(state).into_iter().filter(|&pos| {
                self.world.size.bound_on(pos) && self.world[pos].kind.is_redstone()
            }));
        }

        if let Some(down_pos) = pos.down() {
            if self.world[down_pos].kind.is_cobble() {
                propagate_targets.push(down_pos);

                propagate_targets.extend(
                    pos.cardinal_redstone(state)
                        .into_iter()
                        .filter(|&pos| self.world.size.bound_on(pos))
                        .filter(|&pos| !self.world[pos].kind.is_cobble())
                        .filter_map(|pos| pos.walk(Direction::Bottom))
                        .filter(|&pos| {
                            self.world.size.bound_on(pos) && self.world[pos].kind.is_redstone()
                        }),
                );
            }
        }

        propagate_targets
    }

    fn normalize_signal_levels(&mut self) -> bool {
        let redstone_changed = self.normalize_redstone_strengths();
        let cobble_changed = self.normalize_cobble_power_counts();
        redstone_changed || cobble_changed
    }

    fn normalize_redstone_strengths(&mut self) -> bool {
        let started = self.profile.as_ref().map(|_| Instant::now());
        let mut relaxation_passes = 0;
        let mut targets_evaluated = 0;
        let mut any_changed = false;
        let mut changed = true;
        while changed {
            relaxation_passes += 1;
            changed = false;
            let next_strengths = self
                .redstone_positions
                .iter()
                .copied()
                .filter_map(|pos| {
                    let block = self.world[pos];
                    let BlockKind::Redstone {
                        on_count, strength, ..
                    } = block.kind
                    else {
                        return None;
                    };
                    targets_evaluated += 1;
                    let next_strength = if on_count > 0 {
                        15
                    } else {
                        self.redstone_input_strength(pos)
                    };
                    (next_strength != strength).then_some((pos, next_strength))
                })
                .collect::<Vec<_>>();

            for (pos, next_strength) in next_strengths {
                let BlockKind::Redstone { strength, .. } = &mut self.world[pos].kind else {
                    continue;
                };
                *strength = next_strength;
                changed = true;
                any_changed = true;
            }
        }

        if let (Some(profile), Some(started)) = (&mut self.profile, started) {
            profile.redstone_normalization_time += started.elapsed();
            profile.redstone_normalization_calls += 1;
            profile.redstone_relaxation_passes += relaxation_passes;
            profile.redstone_targets_evaluated += targets_evaluated;
        }

        any_changed
    }

    fn redstone_input_strength(&self, target: Position) -> usize {
        self.redstone_inputs[target.index(&self.world.size).0]
            .iter()
            .copied()
            .filter_map(|source_pos| {
                let source_block = self.world[source_pos];
                let BlockKind::Redstone {
                    strength: source_strength,
                    ..
                } = source_block.kind
                else {
                    return None;
                };
                if source_strength <= 1 {
                    return None;
                }
                Some(source_strength - 1)
            })
            .max()
            .unwrap_or(0)
    }

    fn normalize_cobble_power_counts(&mut self) -> bool {
        let started = self.profile.as_ref().map(|_| Instant::now());
        let mut targets_evaluated = 0;
        let updates = self
            .cobble_positions
            .iter()
            .copied()
            .filter_map(|pos| {
                let block = self.world[pos];
                let BlockKind::Cobble {
                    on_count,
                    on_base_count,
                } = block.kind
                else {
                    return None;
                };
                targets_evaluated += 1;
                let (next_on_count, next_on_base_count) = self.cobble_power_counts(pos);
                (next_on_count != on_count || next_on_base_count != on_base_count).then_some((
                    pos,
                    next_on_count,
                    next_on_base_count,
                ))
            })
            .collect::<Vec<_>>();

        let changed = !updates.is_empty();
        for (pos, next_on_count, next_on_base_count) in updates {
            self.world[pos].kind = BlockKind::Cobble {
                on_count: next_on_count,
                on_base_count: next_on_base_count,
            };
        }
        if let (Some(profile), Some(started)) = (&mut self.profile, started) {
            profile.cobble_normalization_time += started.elapsed();
            profile.cobble_normalization_calls += 1;
            profile.cobble_targets_evaluated += targets_evaluated;
        }
        changed
    }

    fn cobble_power_counts(&self, target: Position) -> (usize, usize) {
        let mut sources = 0;
        let mut hard_sources = 0;
        for input in &self.cobble_power_inputs[target.index(&self.world.size).0] {
            let active = match self.world[input.source].kind {
                BlockKind::Torch { is_on }
                | BlockKind::Switch { is_on }
                | BlockKind::Repeater { is_on, .. } => is_on,
                BlockKind::Redstone { strength, .. } => strength > 0,
                BlockKind::RedstoneBlock => true,
                _ => false,
            };
            if active {
                sources += 1;
                hard_sources += usize::from(input.hard);
            }
        }
        (sources, hard_sources)
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
        let started = self.profile.as_ref().map(|_| Instant::now());
        let events_before = *local_events;
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

        if let (Some(profile), Some(started)) = (&mut self.profile, started) {
            profile.event_processing_time += started.elapsed();
            profile.event_batches += 1;
            profile.events_processed += *local_events - events_before;
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

        tracing::trace!("trigger cobble event: {event:?}, {block:?}");

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
                let source_position = event
                    .target_position
                    .walk(event.direction)
                    .unwrap_or(event.target_position);
                let source_key = (event.target_position, source_position);
                if !self.redstone_power_sources.insert(source_key) {
                    return Ok(());
                }
                let source_count = self
                    .redstone_power_sources
                    .iter()
                    .filter(|(target, _)| *target == event.target_position)
                    .count();
                let BlockKind::Redstone {
                    on_count, strength, ..
                } = &mut block.kind
                else {
                    eyre::bail!("unreachable");
                };

                *on_count = source_count;

                if *on_count == 1 {
                    *strength = 15;

                    tracing::trace!("trigger redstone event: {event:?}, {block:?}");

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
                let source_position = event
                    .target_position
                    .walk(event.direction)
                    .unwrap_or(event.target_position);
                let source_key = (event.target_position, source_position);
                if !self.redstone_power_sources.remove(&source_key) {
                    return Ok(());
                }
                let source_count = self
                    .redstone_power_sources
                    .iter()
                    .filter(|(target, _)| *target == event.target_position)
                    .count();
                let BlockKind::Redstone {
                    on_count, strength, ..
                } = &mut block.kind
                else {
                    eyre::bail!("unreachable");
                };

                *on_count = source_count;

                if *on_count == 0 {
                    *strength = 0;

                    tracing::trace!("trigger redstone event: {event:?}, {block:?}");

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

                    tracing::trace!("trigger redstone event: {event:?}, {block:?}");
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

                tracing::trace!("trigger redstone event: {event:?}, {block:?}");
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

        let support_is_powered = self.cobble_power_counts(support_position).0 > 0;
        let next_is_on = !support_is_powered;
        if *is_on == next_is_on {
            return Ok(());
        }
        if next_is_on && self.burned_out_torches.contains(&event.target_position) {
            return Ok(());
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

        tracing::trace!("trigger torch event: {event:?}, {block:?}");

        Ok(())
    }

    fn record_torch_toggle(&mut self, position: Position) -> bool {
        self.prune_torch_toggle_history(position);
        let history = self.torch_toggle_cycles.entry(position).or_default();
        history.push_back(self.cycle);
        history.len() >= TORCH_BURNOUT_TOGGLE_LIMIT
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
            is_locked, delay, ..
        } = block.kind
        else {
            unreachable!()
        };

        let lock_signal = event.direction.is_othogonal_plane(block.direction)
            && self.is_repeater_lock_source(event);

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
                if lock_signal {
                    if !is_locked {
                        block.kind.set_repeater_lock(true)?;

                        tracing::trace!("trigger repeater event: {event:?}, {block:?}");
                    }
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
            EventType::SoftOff
            | EventType::HardOff
            | EventType::TorchOff
            | EventType::RedstoneOff => {
                if lock_signal {
                    if is_locked {
                        block.kind.set_repeater_lock(false)?;

                        tracing::trace!("trigger repeater event: {event:?}, {block:?}");
                    }
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

                    tracing::trace!("trigger repeater event: {event:?}, {block:?}");
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

                    tracing::trace!("trigger repeater event: {event:?}, {block:?}");
                }
            }
        }

        Ok(())
    }

    fn is_repeater_lock_source(&self, event: &Event) -> bool {
        let Some(source_pos) = event.target_position.walk(event.direction) else {
            return false;
        };

        self.world.size.bound_on(source_pos) && self.world[source_pos].kind.is_repeater()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::nbt::NBTRoot;
    use crate::output::OutputMetadata;
    use crate::sequential::layout::SequentialMacro;
    use crate::sequential::SequentialPrimitive;
    use crate::world::block::RedstoneState;
    use crate::world::position::DimSize;

    #[test]
    fn simulator_preserves_dff_latch_torch_state_on_init() -> eyre::Result<()> {
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read("test/d-flip-flop-global-smoke.nbt")?)?;
        let world = nbt.to_world();
        let torch = Position(22, 6, 3);
        let sim = Simulator::from_preserving_torch_states_with_limits_and_trace(
            &world, 256, 50_000, 50_000,
        )
        .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        let torch_block = sim.world()[torch];
        let support = torch.walk(torch_block.direction).unwrap();

        let support_is_powered = sim.cobble_power_counts(support).0 > 0;
        assert!(!sim.burned_out_torches.contains(&torch));
        assert!(matches!(
            sim.world()[torch].kind,
            BlockKind::Torch { is_on } if is_on != support_is_powered
        ));
        Ok(())
    }

    #[test]
    fn simulator_counts_through_full_two_bit_cycle() -> eyre::Result<()> {
        let nbt_path = std::env::var("COUNTER_NBT_PATH")
            .unwrap_or_else(|_| "test/counter-global-smoke.nbt".to_owned());
        let outputs_path = std::env::var("COUNTER_OUTPUTS_PATH")
            .unwrap_or_else(|_| "test/counter-global-smoke.outputs.json".to_owned());
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read(nbt_path)?)?;
        let world = nbt.to_world();
        let clock = world
            .blocks
            .iter()
            .find_map(|(position, block)| {
                matches!(block.kind, BlockKind::Switch { .. }).then_some(*position)
            })
            .expect("counter should contain a clock switch");
        let metadata = OutputMetadata::load(outputs_path)?;
        let output_position = |name: &str| {
            metadata
                .outputs
                .iter()
                .find(|output| output.name == name)
                .unwrap_or_else(|| panic!("missing counter output `{name}`"))
                .position()
        };
        let q0 = output_position("q_0");
        let q1 = output_position("q_1");
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        let output = |sim: &Simulator| {
            let powered = |position| match sim.world()[position].kind {
                BlockKind::Redstone { strength, .. } => strength > 0,
                _ => false,
            };
            usize::from(powered(q0)) | (usize::from(powered(q1)) << 1)
        };
        assert_eq!(output(&sim), 0);
        for (edge, expected) in [1, 2, 3, 0, 1, 2, 3, 0].into_iter().enumerate() {
            sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
            assert_eq!(
                output(&sim),
                expected,
                "rising edge {} burned_out_torches={:?}",
                edge + 1,
                sim.burned_out_torches
            );
            sim.advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)?;
            sim.change_state_with_limits(vec![(clock, false)], 256, 50_000)?;
            assert_eq!(
                output(&sim),
                expected,
                "falling edge {} burned_out_torches={:?}",
                edge + 1,
                sim.burned_out_torches
            );
            sim.advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)?;
        }

        Ok(())
    }

    #[test]
    #[ignore = "manual release-mode simulator performance profile"]
    fn profile_counter_switch_latency() -> eyre::Result<()> {
        let nbt_path = std::env::var("COUNTER_NBT_PATH")
            .unwrap_or_else(|_| "test/counter-global-smoke.nbt".to_owned());
        let cycles = std::env::var("COUNTER_PROFILE_CYCLES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8);
        let profiling_enabled =
            std::env::var("COUNTER_PROFILE_ENABLED").map_or(true, |value| value != "0");
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read(nbt_path)?)?;
        let world = nbt.to_world();
        let clock = world
            .blocks
            .iter()
            .find_map(|(position, block)| {
                matches!(block.kind, BlockKind::Switch { .. }).then_some(*position)
            })
            .expect("counter should contain a clock switch");
        let volume = world.size.0 * world.size.1 * world.size.2;
        let non_air_blocks = world.blocks.len();
        let redstones = world
            .blocks
            .iter()
            .filter(|(_, block)| block.kind.is_redstone())
            .count();
        let cobbles = world
            .blocks
            .iter()
            .filter(|(_, block)| block.kind.is_cobble())
            .count();
        let torches = world
            .blocks
            .iter()
            .filter(|(_, block)| block.kind.is_torch())
            .count();

        let init_started = Instant::now();
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        let init_time = init_started.elapsed();
        sim.set_profiling_enabled(profiling_enabled);

        let mut aggregate = SimulationProfile::default();
        let mut toggle_times = Vec::with_capacity(cycles * 2);
        let mut nbt_conversion_time = Duration::ZERO;
        for edge in 0..cycles * 2 {
            let is_on = edge % 2 == 0;
            sim.reset_profile();
            let toggle_started = Instant::now();
            sim.change_state_with_limits(vec![(clock, is_on)], 256, 50_000)?;
            let toggle_time = toggle_started.elapsed();
            toggle_times.push(toggle_time);
            let profile = sim.profile().cloned().unwrap_or_default();
            aggregate.accumulate(&profile);

            let nbt_started = Instant::now();
            let _: NBTRoot = sim.world().into();
            nbt_conversion_time += nbt_started.elapsed();

            println!(
                "COUNTER_PROFILE_EDGE edge={} state={} total_ms={:.3} events={} batches={} redstone_passes={}",
                edge + 1,
                is_on,
                toggle_time.as_secs_f64() * 1_000.0,
                profile.events_processed,
                profile.event_batches,
                profile.redstone_relaxation_passes,
            );
        }

        let toggles = toggle_times.len();
        let toggle_total = toggle_times.iter().copied().sum::<Duration>();
        let toggle_min = toggle_times.iter().copied().min().unwrap_or_default();
        let toggle_max = toggle_times.iter().copied().max().unwrap_or_default();
        let measured = aggregate.measured_time();
        let unaccounted = toggle_total.saturating_sub(measured);
        let millis = |duration: Duration| duration.as_secs_f64() * 1_000.0;
        let percent = |duration: Duration| {
            if toggle_total.is_zero() {
                0.0
            } else {
                duration.as_secs_f64() * 100.0 / toggle_total.as_secs_f64()
            }
        };

        println!(
            "COUNTER_PROFILE_WORLD volume={volume} non_air={non_air_blocks} redstones={redstones} cobbles={cobbles} torches={torches}"
        );
        println!(
            "COUNTER_PROFILE_SUMMARY toggles={toggles} init_ms={:.3} total_ms={:.3} avg_ms={:.3} min_ms={:.3} max_ms={:.3} nbt_avg_ms={:.3}",
            millis(init_time),
            millis(toggle_total),
            millis(toggle_total) / toggles as f64,
            millis(toggle_min),
            millis(toggle_max),
            millis(nbt_conversion_time) / toggles as f64,
        );
        println!(
            "COUNTER_PROFILE_STAGE event_ms={:.3} event_pct={:.2} redstone_ms={:.3} redstone_pct={:.2} cobble_ms={:.3} cobble_pct={:.2} torch_ms={:.3} torch_pct={:.2} other_ms={:.3} other_pct={:.2}",
            millis(aggregate.event_processing_time),
            percent(aggregate.event_processing_time),
            millis(aggregate.redstone_normalization_time),
            percent(aggregate.redstone_normalization_time),
            millis(aggregate.cobble_normalization_time),
            percent(aggregate.cobble_normalization_time),
            millis(aggregate.torch_reevaluation_time),
            percent(aggregate.torch_reevaluation_time),
            millis(unaccounted),
            percent(unaccounted),
        );
        println!(
            "COUNTER_PROFILE_COUNTS events={} event_batches={} fill_event_ids_calls={} fill_event_ids_ms={:.3} redstone_calls={} redstone_passes={} redstone_targets={} cobble_calls={} cobble_targets={} torch_calls={} torches={}",
            aggregate.events_processed,
            aggregate.event_batches,
            aggregate.fill_event_ids_calls,
            millis(aggregate.fill_event_ids_time),
            aggregate.redstone_normalization_calls,
            aggregate.redstone_relaxation_passes,
            aggregate.redstone_targets_evaluated,
            aggregate.cobble_normalization_calls,
            aggregate.cobble_targets_evaluated,
            aggregate.torch_reevaluation_calls,
            aggregate.torches_evaluated,
        );
        Ok(())
    }

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
    fn unittest_simulator_torch_reevaluation_uses_current_support_power() -> eyre::Result<()> {
        let torch = Position(1, 1, 1);
        let support = Position(1, 1, 0);
        let mock_world = World {
            size: DimSize(3, 3, 2),
            blocks: vec![
                (
                    support,
                    Block {
                        kind: BlockKind::Cobble {
                            on_count: 0,
                            on_base_count: 0,
                        },
                        direction: Direction::None,
                    },
                ),
                (
                    torch,
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::Bottom,
                    },
                ),
            ],
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[torch];

        sim.propgate_torch_event(
            &mut block,
            &Event {
                id: None,
                from_id: None,
                event_type: EventType::SoftOn,
                target_position: torch,
                direction: Direction::Bottom,
            },
        )?;

        assert!(
            matches!(block.kind, BlockKind::Torch { is_on: true }),
            "stale powered-support reevaluation should not turn the torch off"
        );

        Ok(())
    }

    #[test]
    fn unittest_simulator_burned_out_torch_does_not_recover_during_session() -> eyre::Result<()> {
        let torch = Position(1, 1, 1);
        let support = Position(1, 1, 0);
        let mock_world = World {
            size: DimSize(3, 3, 2),
            blocks: vec![
                (
                    support,
                    Block {
                        kind: BlockKind::Cobble {
                            on_count: 0,
                            on_base_count: 0,
                        },
                        direction: Direction::None,
                    },
                ),
                (
                    torch,
                    Block {
                        kind: BlockKind::Torch { is_on: false },
                        direction: Direction::Bottom,
                    },
                ),
            ],
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        sim.cycle = TORCH_BURNOUT_WINDOW_CYCLES * 2;
        sim.burned_out_torches.insert(torch);
        sim.torch_toggle_cycles
            .insert(torch, VecDeque::from([0, 1, 2, 3, 4, 5, 6, 7]));
        let mut block = sim.world[torch];

        sim.propgate_torch_event(
            &mut block,
            &Event {
                id: None,
                from_id: None,
                event_type: EventType::SoftOff,
                target_position: torch,
                direction: Direction::Bottom,
            },
        )?;

        assert!(
            matches!(block.kind, BlockKind::Torch { is_on: false }),
            "burnout is a simulator-session stabilization state and should not recover by age"
        );

        Ok(())
    }

    #[test]
    fn simulator_idle_cycles_age_torch_toggle_history_before_burnout() -> eyre::Result<()> {
        let torch = Position(1, 1, 1);
        let mut sim = Simulator::new(
            &World {
                size: DimSize(3, 3, 2),
                blocks: Vec::new(),
            },
            DEFAULT_TRACE_LIMIT,
        );
        for _ in 0..16 {
            assert!(!sim.record_torch_toggle(torch));
            sim.advance_idle_cycles(TORCH_BURNOUT_WINDOW_CYCLES + 1)?;
        }
        assert_eq!(sim.torch_toggle_cycles[&torch].len(), 1);
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

    fn full_adder_world() -> eyre::Result<World> {
        Ok(NBTRoot::from_nbt_bytes(&std::fs::read("test/full-adder.nbt")?)?.to_world())
    }

    fn switch_positions(world: &World) -> Vec<Position> {
        let mut switches = world
            .blocks
            .iter()
            .filter_map(|(pos, block)| {
                matches!(block.kind, BlockKind::Switch { .. }).then_some(*pos)
            })
            .collect::<Vec<_>>();
        switches.sort();
        switches
    }

    fn block_is_powered(world: &World3D, pos: Position) -> bool {
        match world[pos].kind {
            BlockKind::Redstone { strength, .. } => strength > 0,
            BlockKind::Torch { is_on } | BlockKind::Switch { is_on } => is_on,
            BlockKind::Cobble { on_count, .. } => on_count > 0,
            _ => false,
        }
    }

    fn signal_snapshot(world: &World3D) -> Vec<(Position, BlockKind)> {
        let mut snapshot = world
            .iter_block()
            .into_iter()
            .filter_map(|(pos, block)| match block.kind {
                BlockKind::Switch { .. }
                | BlockKind::Torch { .. }
                | BlockKind::Redstone { .. }
                | BlockKind::Cobble { .. } => Some((pos, block.kind.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        snapshot.sort_by_key(|(pos, _)| *pos);
        snapshot
    }

    #[test]
    fn unittest_simulator_waveform_defaults_to_interactive_signal_blocks() {
        let mut first = World3D::new(DimSize(5, 1, 1));
        first[Position(0, 0, 0)] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::Top,
        };
        first[Position(1, 0, 0)] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
        first[Position(2, 0, 0)] = Block {
            kind: BlockKind::Cobble {
                on_count: 1,
                on_base_count: 0,
            },
            direction: Direction::None,
        };
        first[Position(3, 0, 0)] = Block {
            kind: BlockKind::Torch { is_on: true },
            direction: Direction::Bottom,
        };
        first[Position(4, 0, 0)] = Block {
            kind: BlockKind::Repeater {
                is_on: false,
                is_locked: false,
                delay: 1,
                lock_input1: None,
                lock_input2: None,
            },
            direction: Direction::North,
        };

        let mut second = first.clone();
        second[Position(0, 0, 0)].kind = BlockKind::Switch { is_on: true };
        second[Position(1, 0, 0)].kind = BlockKind::Redstone {
            on_count: 0,
            state: 0,
            strength: 12,
        };
        second[Position(2, 0, 0)].kind = BlockKind::Cobble {
            on_count: 0,
            on_base_count: 0,
        };
        second[Position(3, 0, 0)].kind = BlockKind::Torch { is_on: false };
        second[Position(4, 0, 0)].kind = BlockKind::Repeater {
            is_on: true,
            is_locked: true,
            delay: 1,
            lock_input1: None,
            lock_input2: None,
        };

        let waveform = SimulationWaveform::from_snapshots(&[
            SimulationSnapshot {
                cycle: 7,
                world: first,
            },
            SimulationSnapshot {
                cycle: 8,
                world: second,
            },
        ]);

        assert_eq!(waveform.cycles, vec![7, 8]);
        assert_eq!(
            waveform
                .signals
                .iter()
                .map(|signal| format!("{}.{}", signal.kind, signal.property))
                .collect::<Vec<_>>(),
            vec![
                "switch.powered",
                "repeater.locked",
                "repeater.powered",
                "torch.lit"
            ]
        );
        assert_eq!(waveform.signals[0].position, [0, 0, 0]);
        assert_eq!(waveform.signals[0].values, vec![0, 1]);
        assert_eq!(waveform.signals[1].position, [4, 0, 0]);
        assert_eq!(waveform.signals[1].values, vec![0, 1]);
        assert_eq!(waveform.signals[2].position, [4, 0, 0]);
        assert_eq!(waveform.signals[2].values, vec![0, 1]);
        assert_eq!(waveform.signals[3].position, [3, 0, 0]);
        assert_eq!(waveform.signals[3].values, vec![1, 0]);
    }

    #[test]
    fn unittest_simulator_can_reenable_trace_after_running_without_trace() -> eyre::Result<()> {
        let default_redstone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
        let world = World {
            size: DimSize(3, 1, 1),
            blocks: vec![
                (
                    Position(0, 0, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: false },
                        direction: Direction::Top,
                    },
                ),
                (Position(1, 0, 0), default_redstone),
            ],
        };
        let mut sim = Simulator::from_with_limits_and_trace(&world, 16, 100, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        sim.change_state_with_limits(vec![(Position(0, 0, 0), true)], 16, 100)?;
        assert!(sim.trace().is_empty());
        assert!(sim.snapshots().is_empty());

        sim.set_trace_limit(50_000);
        sim.change_state_with_limits(vec![(Position(0, 0, 0), false)], 16, 100)?;

        assert!(!sim.trace().is_empty());
        assert!(!sim.snapshots().is_empty());
        Ok(())
    }

    fn assert_matches_fresh_recompute(
        world: &World,
        toggles: &[(usize, bool)],
    ) -> eyre::Result<()> {
        let switches = switch_positions(world);
        let mut final_states = vec![false; switches.len()];
        let mut sequential = Simulator::from_with_limits_and_trace(world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        for (switch_index, value) in toggles {
            final_states[*switch_index] = *value;
            sequential.change_state_with_limits(
                vec![(switches[*switch_index], *value)],
                256,
                50_000,
            )?;
        }

        let mut fresh = Simulator::from_with_limits_and_trace(world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        fresh.change_state_with_limits(
            switches
                .iter()
                .zip(final_states)
                .map(|(position, value)| (*position, value))
                .collect(),
            256,
            50_000,
        )?;

        assert_eq!(
            signal_snapshot(sequential.world()),
            signal_snapshot(fresh.world()),
            "sequential switch toggles should settle to the same state as a fresh recompute"
        );

        Ok(())
    }

    #[test]
    fn unittest_simulator_full_adder_sequential_toggles_match_fresh_recompute() -> eyre::Result<()>
    {
        let world = full_adder_world()?;

        assert_matches_fresh_recompute(&world, &[(0, true), (1, true), (2, true), (1, false)])?;
        assert_matches_fresh_recompute(&world, &[(0, true), (1, true), (2, true), (0, false)])?;
        assert_matches_fresh_recompute(&world, &[(0, true), (1, true), (2, true), (2, false)])?;

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
    fn unittest_simulator_half_adder_generated_truth_table() -> eyre::Result<()> {
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read(
            "test/half-adder-generated-from-verilog.nbt",
        )?)?;
        let world = nbt.to_world();
        let switches = switch_positions(&world);
        let sum_output = Position(4, 4, 3);
        let carry_output = Position(2, 6, 1);

        assert_eq!(switches.len(), 2);
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

            let a = (mask & 0b01) != 0;
            let b = (mask & 0b10) != 0;
            assert_eq!(
                block_is_powered(sim.world(), sum_output),
                a ^ b,
                "half-adder sum mismatch for mask {mask:02b}"
            );
            assert_eq!(
                block_is_powered(sim.world(), carry_output),
                a & b,
                "half-adder carry mismatch for mask {mask:02b}"
            );
        }

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
    fn unittest_simulator_repeater_does_not_power_lower_front_redstone_directly() {
        let repeater = Position(1, 1, 2);
        let lower_front_redstone = Position(0, 1, 1);
        let mock_world = World {
            size: DimSize(3, 3, 3),
            blocks: vec![
                (repeater.down().unwrap(), test_cobble(0, 0)),
                (
                    repeater,
                    Block {
                        kind: BlockKind::Repeater {
                            is_on: true,
                            is_locked: false,
                            delay: 1,
                            lock_input1: None,
                            lock_input2: None,
                        },
                        direction: Direction::East,
                    },
                ),
                (lower_front_redstone.down().unwrap(), test_cobble(0, 0)),
                (
                    lower_front_redstone,
                    Block {
                        kind: BlockKind::Redstone {
                            on_count: 0,
                            state: 0,
                            strength: 0,
                        },
                        direction: Direction::None,
                    },
                ),
            ],
        };

        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[repeater];

        sim.propgate_repeater_event(
            &mut block,
            &Event {
                id: None,
                from_id: None,
                event_type: EventType::RepeaterOn { delay: 0 },
                target_position: repeater,
                direction: Direction::East,
            },
        )
        .unwrap();

        assert!(!sim
            .queue
            .iter()
            .flatten()
            .any(|event| event.target_position == lower_front_redstone));
    }

    fn test_repeater(is_on: bool, is_locked: bool, direction: Direction) -> Block {
        Block {
            kind: BlockKind::Repeater {
                is_on,
                is_locked,
                delay: 2,
                lock_input1: None,
                lock_input2: None,
            },
            direction,
        }
    }

    fn test_cobble(on_count: usize, on_base_count: usize) -> Block {
        Block {
            kind: BlockKind::Cobble {
                on_count,
                on_base_count,
            },
            direction: Direction::None,
        }
    }

    #[test]
    fn unittest_simulator_repeater_side_hard_power_does_not_lock() -> eyre::Result<()> {
        let target = Position(1, 1, 0);
        let source = Position(2, 1, 0);
        let mock_world = World {
            size: DimSize(3, 3, 1),
            blocks: vec![
                (target, test_repeater(false, false, Direction::North)),
                (source, test_cobble(1, 1)),
            ],
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];

        sim.propgate_repeater_event(
            &mut block,
            &Event {
                id: None,
                from_id: None,
                event_type: EventType::HardOn,
                target_position: target,
                direction: target.diff(source),
            },
        )?;

        assert!(matches!(
            block.kind,
            BlockKind::Repeater {
                is_locked: false,
                ..
            }
        ));

        Ok(())
    }

    #[test]
    fn unittest_simulator_repeater_side_repeater_power_locks() -> eyre::Result<()> {
        let target = Position(1, 1, 0);
        let source = Position(2, 1, 0);
        let mock_world = World {
            size: DimSize(3, 3, 1),
            blocks: vec![
                (target, test_repeater(true, false, Direction::North)),
                (source, test_repeater(true, false, Direction::West)),
            ],
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];

        sim.propgate_repeater_event(
            &mut block,
            &Event {
                id: None,
                from_id: None,
                event_type: EventType::HardOn,
                target_position: target,
                direction: target.diff(source),
            },
        )?;

        assert!(matches!(
            block.kind,
            BlockKind::Repeater {
                is_locked: true,
                ..
            }
        ));

        Ok(())
    }

    #[test]
    fn unittest_simulator_repeater_side_repeater_power_off_unlocks() -> eyre::Result<()> {
        let target = Position(1, 1, 0);
        let source = Position(2, 1, 0);
        let mock_world = World {
            size: DimSize(3, 3, 1),
            blocks: vec![
                (target, test_repeater(false, true, Direction::North)),
                (source, test_repeater(false, false, Direction::West)),
            ],
        };
        let mut sim = Simulator::new(&mock_world, DEFAULT_TRACE_LIMIT);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];

        sim.propgate_repeater_event(
            &mut block,
            &Event {
                id: None,
                from_id: None,
                event_type: EventType::HardOff,
                target_position: target,
                direction: target.diff(source),
            },
        )?;

        assert!(matches!(
            block.kind,
            BlockKind::Repeater {
                is_locked: false,
                ..
            }
        ));

        Ok(())
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

    #[test]
    fn redstone_direct_power_tracks_unique_sources() -> eyre::Result<()> {
        let target = Position(1, 1, 1);
        let world = World {
            size: DimSize(3, 3, 3),
            blocks: vec![
                (
                    target,
                    Block {
                        kind: BlockKind::Redstone {
                            on_count: 0,
                            state: 0,
                            strength: 0,
                        },
                        direction: Direction::None,
                    },
                ),
                (
                    Position(1, 1, 0),
                    Block {
                        kind: BlockKind::Cobble {
                            on_count: 0,
                            on_base_count: 0,
                        },
                        direction: Direction::None,
                    },
                ),
            ],
        };
        let mut sim = Simulator::new(&world, 0);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];
        let event = |event_type| Event {
            id: None,
            from_id: None,
            event_type,
            target_position: target,
            direction: Direction::West,
        };

        sim.propagate_redstone_event(&mut block, &event(EventType::TorchOn))?;
        sim.propagate_redstone_event(&mut block, &event(EventType::TorchOn))?;
        assert!(matches!(
            block.kind,
            BlockKind::Redstone {
                on_count: 1,
                strength: 15,
                ..
            }
        ));

        sim.propagate_redstone_event(&mut block, &event(EventType::TorchOff))?;
        assert!(matches!(
            block.kind,
            BlockKind::Redstone {
                on_count: 0,
                strength: 0,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn repeater_queues_off_while_on_transition_is_pending() -> eyre::Result<()> {
        let target = Position(1, 1, 1);
        let direction = Direction::West;
        let world = World {
            size: DimSize(3, 3, 3),
            blocks: vec![(
                target,
                Block {
                    kind: BlockKind::Repeater {
                        is_on: false,
                        is_locked: false,
                        delay: 1,
                        lock_input1: None,
                        lock_input2: None,
                    },
                    direction,
                },
            )],
        };
        let mut sim = Simulator::new(&world, 0);
        sim.queue.push_back(VecDeque::new());
        let mut block = sim.world[target];
        let event = |event_type| Event {
            id: None,
            from_id: None,
            event_type,
            target_position: target,
            direction,
        };

        sim.propgate_repeater_event(&mut block, &event(EventType::SoftOn))?;
        sim.propgate_repeater_event(&mut block, &event(EventType::SoftOff))?;

        let queued = sim.queue.back().expect("event queue should exist");
        assert!(queued
            .iter()
            .any(|event| matches!(event.event_type, EventType::RepeaterOn { .. })));
        assert!(queued
            .iter()
            .any(|event| matches!(event.event_type, EventType::RepeaterOff { .. })));
        Ok(())
    }
}
