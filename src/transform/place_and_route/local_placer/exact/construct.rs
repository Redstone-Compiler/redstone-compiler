//! Windowed construction of an initial layout along the Y axis.
//!
//! Solving a whole cell at once grows exponentially with the number of free
//! cells. Construction instead appends one gate at a time in a short window of
//! Y slices: the previous slices stay fixed, inputs appear when first needed,
//! and every net still needed later must be observable on the window's last
//! slice so the next window can pick it up. Each step is a small exact
//! problem, and the final step checks the whole circuit's outputs. The result
//! is long but valid; `compact` then shortens it.
//!
//! Every live net crosses every seam, so the gate order and how long outputs
//! are carried decide how crowded the cross-section gets.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use eyre::bail;

use super::encode::{CellKind, TORCH_ATTACH};
use super::layout::{ExactLayout, InputPolicy, OutputPolicy};
use super::netlist::{NetDriver, NetId, NorNetlist};
use super::{ExactLocalPlacer, ExactOutcome, ExactPlacement, ExactPlacerConfig, ExactTuning};
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

/// The order in which construction places gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOrder {
    /// Depth-first from net 0 upward (the original order).
    NetIndex,
    /// Output by output, smallest fan-in cone first, depth-first inside each
    /// cone, so an output and its private logic finish before the next one
    /// starts.
    SmallestConeFirst,
}

/// Gates in construction order.
pub(super) fn gate_order(netlist: &NorNetlist, order: GateOrder) -> Vec<NetId> {
    fn visit(netlist: &NorNetlist, net: NetId, done: &mut [bool], order: &mut Vec<NetId>) {
        if done[net] {
            return;
        }
        done[net] = true;
        for &input in &netlist.nets[net].gate_inputs {
            visit(netlist, input, done, order);
        }
        if netlist.nets[net].driver == NetDriver::Gate {
            order.push(net);
        }
    }
    let mut roots = (0..netlist.nets.len()).collect::<Vec<_>>();
    if order == GateOrder::SmallestConeFirst {
        let cone = |root: NetId| {
            let mut seen = BTreeSet::new();
            let mut stack = vec![root];
            while let Some(net) = stack.pop() {
                if seen.insert(net) {
                    stack.extend(netlist.nets[net].gate_inputs.iter().copied());
                }
            }
            seen.len()
        };
        let mut outputs = netlist
            .outputs
            .iter()
            .map(|(_, net)| *net)
            .collect::<Vec<_>>();
        outputs.sort_by_key(|&net| (cone(net), net));
        // Gates outside every output cone still come last, in index order.
        outputs.extend(roots);
        roots = outputs;
    }
    let mut done = vec![false; netlist.nets.len()];
    let mut gates = Vec::new();
    for net in roots {
        visit(netlist, net, &mut done, &mut gates);
    }
    gates
}

#[derive(Debug, Clone)]
pub struct ConstructionConfig {
    pub width: usize,
    pub height: usize,
    /// First window length tried for each gate, in Y slices.
    pub window: usize,
    /// Longest window tried before giving up on a gate.
    pub max_window: usize,
    /// Slices of the previous step that are re-solved with each new window,
    /// so the interface between steps can be reshaped.
    pub overlap: usize,
    /// When no window fits, re-solve more of the previous layout (overlap up
    /// to this) before backtracking. Off by default (= `overlap`): on the
    /// seed-2 full adder none of the wider attempts helped and each failed
    /// attempt costs a full `step_time_limit`.
    pub max_overlap: usize,
    /// On backtracking, block only the previous step's seam slice (the one
    /// the failed step had to build on) instead of its whole window. Off by
    /// default; it did not rescue the seed-2 full adder either.
    pub block_seam: bool,
    /// Construction attempts after the first, each from scratch with a new
    /// seed. A seed that paints itself into a corner can burn many minutes
    /// of failed steps, while another seed often finishes in seconds.
    pub max_restarts: usize,
    /// Time budget of one construction attempt before restarting.
    pub restart_after: Option<Duration>,
    /// Seed spacing between construction attempts.
    pub restart_seed_stride: u32,
    pub step_time_limit: Duration,
    pub workers: usize,
    pub seed: u32,
    pub rank_levels: usize,
    pub stage_levels: usize,
    pub input_policies: BTreeMap<String, InputPolicy>,
    pub output_policies: BTreeMap<String, OutputPolicy>,
    /// Times an earlier step may be re-solved after a later step fails,
    /// per attempt. Off by default: on the full adder (seeds 2 and 4) six
    /// backtracks each cost three failed steps and none let the stuck gate
    /// fit, while a restart reaches the same gate again in about 30 s.
    pub max_backtracks: usize,
    /// Simulator rejections each step's workers may hit before giving up.
    pub max_refinements: usize,
    /// Search and verification constants for every step.
    pub tuning: ExactTuning,
    pub gate_order: GateOrder,
    /// Outputs without a face policy stop crossing seams once no later gate
    /// reads them; they only need to stay observable somewhere in the box.
    /// Off, every output is carried to the last slice.
    pub early_outputs: bool,
    /// Fix the signals of frozen cells to the previous step's solution, so a
    /// step no longer re-justifies (ranks, stages, contributing sources) the
    /// whole layout built so far; those variables were over 90% of a late
    /// step's encoding. The simulator still checks the whole layout.
    pub given_frozen_signals: bool,
    /// After each step, spend up to this long minimizing the layout's block
    /// count (the model's cost) with the same frozen cells, so dead wires do
    /// not pile up. `None` keeps the first layout found. Five seconds halved
    /// the full adder's constructed blocks (about 290 to 127-164).
    pub step_optimize: Option<Duration>,
    /// Model params for every step (see `ExactPlacerConfig::model_params`).
    pub model_params: BTreeMap<String, rsdsl::IValue>,
    /// Diagnostic only: see `ExactPlacerConfig::no_fold`.
    #[doc(hidden)]
    pub no_fold: bool,
}

impl Default for ConstructionConfig {
    fn default() -> Self {
        Self {
            width: 2,
            height: 10,
            window: 3,
            max_window: 6,
            overlap: 1,
            max_overlap: 1,
            block_seam: false,
            max_restarts: 7,
            restart_after: Some(Duration::from_secs(600)),
            restart_seed_stride: 104_729,
            step_time_limit: Duration::from_secs(60),
            workers: 8,
            seed: 1,
            rank_levels: 24,
            stage_levels: 24,
            input_policies: BTreeMap::new(),
            output_policies: BTreeMap::new(),
            max_backtracks: 0,
            max_refinements: 8,
            tuning: ExactTuning::default(),
            gate_order: GateOrder::SmallestConeFirst,
            early_outputs: true,
            given_frozen_signals: true,
            step_optimize: Some(Duration::from_secs(5)),
            model_params: BTreeMap::new(),
            no_fold: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConstructionReport {
    /// `(gate net name, window length, seconds)` per accepted step of the
    /// successful attempt.
    pub steps: Vec<(String, usize, f64)>,
    pub backtracks: usize,
    /// Attempts abandoned before the successful one, with their seeds.
    pub restarts: Vec<(u32, String)>,
    /// The seed of the successful attempt.
    pub seed: u32,
    pub elapsed: Duration,
}

fn sites_in(
    dim: DimSize,
    y_range: std::ops::Range<usize>,
    policy: InputPolicy,
) -> Vec<(Position, Direction)> {
    let mut sites = Vec::new();
    for z in 0..dim.2 {
        for y in y_range.clone() {
            if policy == InputPolicy::MinYFace && y != 0 {
                continue;
            }
            for x in 0..dim.0 {
                let boundary = x == 0 || x + 1 == dim.0 || y == 0 || y + 1 == dim.1;
                if !boundary {
                    continue;
                }
                for attach in TORCH_ATTACH {
                    sites.push((Position(x, y, z), attach));
                }
            }
        }
    }
    sites
}

/// Layout state between construction steps.
#[derive(Clone)]
struct StepState {
    length: usize,
    cells: BTreeMap<Position, CellKind>,
    /// Signal function of every non-air cell in the latest solution.
    signals: BTreeMap<Position, u64>,
    placed_inputs: BTreeMap<String, (Position, Direction)>,
    available: BTreeSet<NetId>,
    result: Option<(DimSize, ExactPlacement)>,
}

impl ExactLocalPlacer {
    /// Builds the layout gate by gate. When an attempt fails or exceeds
    /// `restart_after`, it starts over with the next seed, up to
    /// `max_restarts` times.
    pub fn construct(
        &self,
        config: &ConstructionConfig,
    ) -> eyre::Result<(ExactLayout, ExactPlacement, ConstructionReport)> {
        let started = Instant::now();
        let mut restarts = Vec::new();
        let mut attempt = 0u32;
        loop {
            let seed = config
                .seed
                .wrapping_add(attempt.wrapping_mul(config.restart_seed_stride));
            let attempt_config = ConstructionConfig {
                seed,
                ..config.clone()
            };
            let deadline = config.restart_after.map(|budget| Instant::now() + budget);
            match self.construct_once(&attempt_config, deadline) {
                Ok((layout, placement, mut report)) => {
                    report.restarts = restarts;
                    report.seed = seed;
                    report.elapsed = started.elapsed();
                    return Ok((layout, placement, report));
                }
                Err(error) if (attempt as usize) < config.max_restarts => {
                    tracing::info!(seed, %error, "construction restarts with a new seed");
                    restarts.push((seed, error.to_string()));
                    attempt += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn construct_once(
        &self,
        config: &ConstructionConfig,
        deadline: Option<Instant>,
    ) -> eyre::Result<(ExactLayout, ExactPlacement, ConstructionReport)> {
        let started = Instant::now();
        let netlist = &self.netlist;
        let order = gate_order(netlist, config.gate_order);
        if order.is_empty() {
            bail!("netlist has no gates to construct");
        }
        let mut report = ConstructionReport::default();
        // states[i] is the layout before step i.
        let mut states = vec![StepState {
            length: 0,
            cells: BTreeMap::new(),
            signals: BTreeMap::new(),
            placed_inputs: BTreeMap::new(),
            available: BTreeSet::new(),
            result: None,
        }];
        let mut blocked = vec![Vec::new(); order.len()];
        let mut windows = vec![Vec::new(); order.len()];
        let mut step = 0;
        while step < order.len() {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                bail!(
                    "construction attempt with seed {} ran out of time at gate {}",
                    config.seed,
                    netlist.nets[order[step]].name
                );
            }
            match self.construct_step(
                &order,
                step,
                &states[step],
                &blocked[step],
                config,
                deadline,
            )? {
                Some((state, window_cells, entry)) => {
                    report.steps.push(entry);
                    states.truncate(step + 1);
                    states.push(state);
                    windows[step] = window_cells;
                    step += 1;
                }
                None => {
                    if step == 0 || report.backtracks >= config.max_backtracks {
                        bail!(
                            "construction could not place gate {} within {} slices",
                            netlist.nets[order[step]].name,
                            config.max_window
                        );
                    }
                    report.backtracks += 1;
                    tracing::info!(
                        gate = netlist.nets[order[step]].name,
                        "construction backtracks"
                    );
                    blocked[step].clear();
                    step -= 1;
                    let previous = windows[step].clone();
                    // The failed step could unfreeze up to `max_overlap`
                    // slices, so it built on the slice just before those.
                    let seam = states[step + 1]
                        .length
                        .checked_sub(config.max_overlap.max(config.overlap) + 1);
                    let seam_cells = previous
                        .iter()
                        .filter(|(position, _)| Some(position.1) == seam)
                        .copied()
                        .collect::<Vec<_>>();
                    if config.block_seam && !seam_cells.is_empty() {
                        blocked[step].push(seam_cells);
                    } else {
                        blocked[step].push(previous);
                    }
                }
            }
        }
        let (dim, placement) = states
            .pop()
            .and_then(|state| state.result)
            .expect("the last step produced a layout");
        report.elapsed = started.elapsed();
        Ok((
            ExactLayout::from_placement(dim, &placement),
            placement,
            report,
        ))
    }

    /// Diagnostic hook: with `EXACT_DIAGNOSE_FAILED_STEP=<path prefix>`, the
    /// first step in the process that times out is written as DIMACS and
    /// solved again for `EXACT_DIAGNOSE_SECONDS` (600). It tells a step that
    /// is merely slow from one the solver cannot decide. Construction then
    /// carries on as if the step had failed normally.
    fn diagnose_failed_step(
        &self,
        exact: &ExactPlacerConfig,
        gate: &str,
        window: usize,
    ) -> eyre::Result<()> {
        static DIAGNOSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let Ok(prefix) = std::env::var("EXACT_DIAGNOSE_FAILED_STEP") else {
            return Ok(());
        };
        if DIAGNOSED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }
        let seconds = std::env::var("EXACT_DIAGNOSE_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(600);
        let path = format!("{prefix}-{gate}-w{window}.cnf");
        self.write_dimacs(exact, &path, super::DimacsComments::Legend)?;
        let mut longer = exact.clone();
        longer.time_limit = Some(Duration::from_secs(seconds));
        let started = Instant::now();
        let (outcome, stats) = self.place(&longer)?;
        tracing::info!(
            gate,
            window,
            path,
            seconds = started.elapsed().as_secs_f64(),
            variables = stats.variables,
            clauses = stats.clauses,
            refinements = stats.refinements,
            outcome = ?outcome,
            "construction step diagnosed"
        );
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn construct_step(
        &self,
        order: &[NetId],
        step: usize,
        state: &StepState,
        blocked: &[Vec<(Position, CellKind)>],
        config: &ConstructionConfig,
        deadline: Option<Instant>,
    ) -> eyre::Result<Option<(StepState, Vec<(Position, CellKind)>, (String, usize, f64))>> {
        let netlist = &self.netlist;
        let gate = order[step];
        let is_last = step + 1 == order.len();
        // Outputs that must reach the last slice: face-bound ones, or all of
        // them without `early_outputs`.
        let carried_outputs = netlist
            .outputs
            .iter()
            .filter(|(name, _)| {
                !config.early_outputs
                    || config.output_policies.get(name) == Some(&OutputPolicy::MaxYFace)
            })
            .map(|(_, net)| *net)
            .collect::<BTreeSet<_>>();
        let consumers = |net: NetId, after: usize| {
            carried_outputs.contains(&net)
                || order[after..]
                    .iter()
                    .any(|&gate| netlist.nets[gate].gate_inputs.contains(&net))
        };
        let mut needed_inputs = netlist.nets[gate]
            .gate_inputs
            .iter()
            .filter(|&&input| matches!(netlist.nets[input].driver, NetDriver::Input(_)))
            .filter(|&&input| !state.available.contains(&input))
            .copied()
            .collect::<BTreeSet<_>>();
        if step == 0 {
            // Face-bound inputs must start at Y = 0.
            for (name, policy) in &config.input_policies {
                if *policy == InputPolicy::MinYFace {
                    if let Some(net) = netlist.input_net(name) {
                        needed_inputs.insert(net);
                    }
                }
            }
        }
        let length = state.length;
        // Cheapest re-solve first: fewest slices (overlap + window), then the
        // smaller overlap.
        let mut attempts = (config.overlap..=config.max_overlap.max(config.overlap))
            .flat_map(|overlap| {
                (config.window..=config.max_window).map(move |window| (overlap, window))
            })
            .collect::<Vec<_>>();
        attempts.sort_by_key(|&(overlap, window)| (overlap + window, overlap));
        for (overlap, window) in attempts {
            let frozen = length.saturating_sub(overlap);
            if overlap > config.overlap && frozen == length.saturating_sub(overlap - 1) {
                // Nothing more to unfreeze (the layout is shorter).
                continue;
            }
            let dim = DimSize(config.width, length + window, config.height);
            let mut exact = ExactPlacerConfig::new(dim);
            exact.workers = config.workers;
            exact.seed = config.seed;
            exact.rank_levels = config.rank_levels;
            exact.stage_levels = config.stage_levels;
            exact.no_fold = config.no_fold;
            // Stop at the attempt's deadline too, so a restart starts on time.
            let remaining =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            if remaining.is_some_and(|remaining| remaining.is_zero()) {
                return Ok(None);
            }
            exact.time_limit = Some(remaining.map_or(config.step_time_limit, |remaining| {
                remaining.min(config.step_time_limit)
            }));
            exact.max_refinements = config.max_refinements;
            exact.tuning = config.tuning.clone();
            exact.model_params = config.model_params.clone();
            exact.blocked = blocked.to_vec();
            for (position, kind) in &state.cells {
                if position.1 < frozen {
                    exact.fixed_cells.insert(*position, *kind);
                    if config.given_frozen_signals {
                        if let Some(&function) = state.signals.get(position) {
                            exact.given_signals.insert(*position, function);
                        }
                    }
                }
            }
            for x in 0..dim.0 {
                for y in 0..frozen {
                    for z in 0..dim.2 {
                        exact
                            .fixed_cells
                            .entry(Position(x, y, z))
                            .or_insert(CellKind::Air);
                    }
                }
            }
            for name in netlist.input_names() {
                let net = netlist.input_net(&name).unwrap();
                if let Some((position, attach)) = state.placed_inputs.get(&name) {
                    // Keep placed switches even inside the overlap.
                    exact
                        .fixed_cells
                        .insert(*position, CellKind::Switch(*attach));
                    exact = exact.with_input_site(name.clone(), *position, *attach);
                } else if needed_inputs.contains(&net) {
                    let policy = config
                        .input_policies
                        .get(&name)
                        .copied()
                        .unwrap_or(InputPolicy::Anywhere);
                    exact
                        .input_sites
                        .insert(name.clone(), sites_in(dim, frozen..dim.1, policy));
                } else {
                    exact.absent_inputs.insert(name.clone());
                }
            }
            let last_slice = (0..dim.2)
                .flat_map(|z| (0..dim.0).map(move |x| Position(x, dim.1 - 1, z)))
                .collect::<Vec<_>>();
            if is_last {
                for (name, policy) in &config.output_policies {
                    if *policy == OutputPolicy::MaxYFace {
                        exact = exact.with_output_sites(name.clone(), last_slice.clone());
                        exact.driving_outputs.insert(name.clone());
                    }
                }
            } else {
                let placed = state
                    .available
                    .iter()
                    .chain(needed_inputs.iter())
                    .chain(std::iter::once(&gate))
                    .copied()
                    .collect::<BTreeSet<_>>();
                let mut observations = placed
                    .iter()
                    .copied()
                    .filter(|&net| consumers(net, step + 1))
                    .map(|net| (net, last_slice.clone()))
                    .collect::<Vec<_>>();
                // Finished outputs stay observable but may sit anywhere.
                let everywhere = (0..dim.0)
                    .flat_map(|x| {
                        (0..dim.1).flat_map(move |y| (0..dim.2).map(move |z| Position(x, y, z)))
                    })
                    .collect::<Vec<_>>();
                for (_, net) in &netlist.outputs {
                    if placed.contains(net) && !consumers(*net, step + 1) {
                        observations.push((*net, everywhere.clone()));
                    }
                }
                exact.observations = Some(observations);
            }
            let step_started = Instant::now();
            let (outcome, stats) = self.place(&exact)?;
            let ExactOutcome::Placed(placement) = outcome else {
                tracing::info!(
                    gate = netlist.nets[gate].name,
                    window,
                    overlap,
                    seconds = step_started.elapsed().as_secs_f64(),
                    refinements = stats.refinements,
                    outcome = ?outcome,
                    "construction step failed"
                );
                if matches!(outcome, ExactOutcome::Unknown { .. }) {
                    self.diagnose_failed_step(&exact, &netlist.nets[gate].name, window)?;
                }
                continue;
            };
            let mut placement = placement;
            if let Some(budget) = config.step_optimize {
                // Trade the window's dead wires away while it is small: ask
                // for fewer blocks than the layout just found, minimizing.
                let remaining =
                    deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
                let mut tighter = exact.clone();
                tighter.optimize = true;
                tighter.max_blocks = Some(placement.cells.len().saturating_sub(1));
                tighter.time_limit =
                    Some(remaining.map_or(budget, |remaining| remaining.min(budget)));
                if let Ok((ExactOutcome::Placed(better), _)) = self.place(&tighter) {
                    if better.cells.len() < placement.cells.len() {
                        placement = better;
                    }
                }
            }
            let seconds = step_started.elapsed().as_secs_f64();
            tracing::info!(
                gate = netlist.nets[gate].name,
                window,
                overlap,
                seconds,
                cells = placement.cells.len(),
                "construction step"
            );
            let cells = placement.cells.iter().copied().collect::<BTreeMap<_, _>>();
            let mut placed_inputs = state.placed_inputs.clone();
            for input in &placement.placed.inputs {
                let position = input.position();
                let CellKind::Switch(attach) = cells[&position] else {
                    unreachable!("inputs are switches")
                };
                placed_inputs.insert(input.name.clone(), (position, attach));
            }
            let window_cells = (0..dim.0)
                .flat_map(|x| {
                    (frozen..dim.1).flat_map(move |y| (0..dim.2).map(move |z| Position(x, y, z)))
                })
                .map(|position| {
                    (
                        position,
                        cells.get(&position).copied().unwrap_or(CellKind::Air),
                    )
                })
                .collect();
            let mut available = state.available.clone();
            available.extend(needed_inputs.iter().copied());
            available.insert(gate);
            let next = StepState {
                length: dim.1,
                cells,
                signals: placement.signals.iter().copied().collect(),
                placed_inputs,
                available,
                result: Some((dim, *placement)),
            };
            return Ok(Some((
                next,
                window_cells,
                (netlist.nets[gate].name.clone(), window, seconds),
            )));
        }
        Ok(None)
    }
}
