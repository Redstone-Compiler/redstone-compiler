//! Windowed construction of an initial layout along the Y axis.
//!
//! Solving a whole cell at once grows exponentially with the number of free
//! cells. Construction instead appends one gate at a time in a short window of
//! Y slices: the previous slices stay fixed, inputs appear when first needed,
//! and every net still needed later must be observable on the window's last
//! slice so the next window can pick it up. Each step is a small exact
//! problem, and the final step checks the whole circuit's outputs. The result
//! is long but valid; `compact` then shortens it.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use eyre::bail;

use super::encode::{CellKind, TORCH_ATTACH};
use super::layout::{ExactLayout, InputPolicy, OutputPolicy};
use super::netlist::{NetDriver, NetId};
use super::{ExactLocalPlacer, ExactOutcome, ExactPlacement, ExactPlacerConfig, ExactTuning};
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

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
    pub step_time_limit: Duration,
    pub workers: usize,
    pub seed: u32,
    pub rank_levels: usize,
    pub stage_levels: usize,
    pub input_policies: BTreeMap<String, InputPolicy>,
    pub output_policies: BTreeMap<String, OutputPolicy>,
    /// Times an earlier step may be re-solved after a later step fails.
    pub max_backtracks: usize,
    /// Simulator rejections each step's workers may hit before giving up.
    pub max_refinements: usize,
    /// Search and verification constants for every step.
    pub tuning: ExactTuning,
    /// Diagnostic only: see `ExactPlacerConfig::legacy_encoder`.
    #[doc(hidden)]
    pub legacy_encoder: bool,
}

impl Default for ConstructionConfig {
    fn default() -> Self {
        Self {
            width: 2,
            height: 10,
            window: 3,
            max_window: 6,
            overlap: 1,
            step_time_limit: Duration::from_secs(60),
            workers: 8,
            seed: 1,
            rank_levels: 24,
            stage_levels: 24,
            input_policies: BTreeMap::new(),
            output_policies: BTreeMap::new(),
            max_backtracks: 6,
            max_refinements: 8,
            tuning: ExactTuning::default(),
            legacy_encoder: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConstructionReport {
    /// `(gate net name, window length, seconds)` per accepted step.
    pub steps: Vec<(String, usize, f64)>,
    pub backtracks: usize,
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
    placed_inputs: BTreeMap<String, (Position, Direction)>,
    available: BTreeSet<NetId>,
    result: Option<(DimSize, ExactPlacement)>,
}

impl ExactLocalPlacer {
    pub fn construct(
        &self,
        config: &ConstructionConfig,
    ) -> eyre::Result<(ExactLayout, ExactPlacement, ConstructionReport)> {
        let started = Instant::now();
        let netlist = &self.netlist;
        let order = netlist
            .topological_order()
            .into_iter()
            .filter(|&net| netlist.nets[net].driver == NetDriver::Gate)
            .collect::<Vec<_>>();
        if order.is_empty() {
            bail!("netlist has no gates to construct");
        }
        let mut report = ConstructionReport::default();
        // states[i] is the layout before step i.
        let mut states = vec![StepState {
            length: 0,
            cells: BTreeMap::new(),
            placed_inputs: BTreeMap::new(),
            available: BTreeSet::new(),
            result: None,
        }];
        let mut blocked = vec![Vec::new(); order.len()];
        let mut windows = vec![Vec::new(); order.len()];
        let mut step = 0;
        while step < order.len() {
            match self.construct_step(&order, step, &states[step], &blocked[step], config)? {
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
                    blocked[step].push(previous);
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

    #[allow(clippy::type_complexity)]
    fn construct_step(
        &self,
        order: &[NetId],
        step: usize,
        state: &StepState,
        blocked: &[Vec<(Position, CellKind)>],
        config: &ConstructionConfig,
    ) -> eyre::Result<Option<(StepState, Vec<(Position, CellKind)>, (String, usize, f64))>> {
        let netlist = &self.netlist;
        let gate = order[step];
        let is_last = step + 1 == order.len();
        let output_nets = netlist
            .outputs
            .iter()
            .map(|(_, net)| *net)
            .collect::<BTreeSet<_>>();
        let consumers = |net: NetId, after: usize| {
            output_nets.contains(&net)
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
        let frozen = length.saturating_sub(config.overlap);
        for window in config.window..=config.max_window {
            let dim = DimSize(config.width, length + window, config.height);
            let mut exact = ExactPlacerConfig::new(dim);
            exact.workers = config.workers;
            exact.seed = config.seed;
            exact.rank_levels = config.rank_levels;
            exact.stage_levels = config.stage_levels;
            exact.legacy_encoder = config.legacy_encoder;
            exact.time_limit = Some(config.step_time_limit);
            exact.max_refinements = config.max_refinements;
            exact.tuning = config.tuning.clone();
            exact.blocked = blocked.to_vec();
            for (position, kind) in &state.cells {
                if position.1 < frozen {
                    exact.fixed_cells.insert(*position, *kind);
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
                let live = state
                    .available
                    .iter()
                    .chain(needed_inputs.iter())
                    .chain(std::iter::once(&gate))
                    .copied()
                    .filter(|&net| consumers(net, step + 1))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .map(|net| (net, last_slice.clone()))
                    .collect::<Vec<_>>();
                exact.observations = Some(live);
            }
            let step_started = Instant::now();
            let (outcome, stats) = self.place(&exact)?;
            let ExactOutcome::Placed(placement) = outcome else {
                tracing::info!(
                    gate = netlist.nets[gate].name,
                    window,
                    seconds = step_started.elapsed().as_secs_f64(),
                    refinements = stats.refinements,
                    outcome = ?outcome,
                    "construction step failed"
                );
                continue;
            };
            let seconds = step_started.elapsed().as_secs_f64();
            tracing::info!(
                gate = netlist.nets[gate].name,
                window,
                seconds,
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
