//! Compaction by slice removal with exact window repair (large-neighborhood search).
//!
//! A verified layout is shortened by deleting one Y slice (or Z layer) and
//! shifting the rest back. Only the cells next to the seam are freed and
//! re-solved exactly, with every other cell fixed, so each repair is a small
//! SAT problem even when the whole box is not. An accepted repair is checked
//! by the simulator like any other exact placement.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::encode::CellKind;
use super::layout::{ExactLayout, OutputPolicy};
use super::{ExactLocalPlacer, ExactOutcome, ExactPlacement, ExactPlacerConfig, ExactTuning};
use crate::world::position::Position;

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Slices freed on each side of the seam.
    pub window_radius: usize,
    /// Wider windows tried once no seam can be repaired at the current radius.
    pub max_window_radius: usize,
    /// After shrinking, re-solve sliding windows with one block fewer.
    pub minimize_blocks: bool,
    /// Minimize each block-reduction window's cost in one solve (`optimize`)
    /// instead of asking for one block fewer per attempt (rsdsl model only).
    pub optimize_windows: bool,
    /// Slices re-solved together in each block-reduction window.
    pub reduction_window: usize,
    /// Axes the block-reduction window slides along: 1 = Y slices (the whole
    /// cross-section), 2 = Z layers (the whole length).
    pub reduction_axes: Vec<usize>,
    /// Alternate slice removal and block reduction until neither makes
    /// progress: fewer blocks often free a slice that could not be removed.
    pub repeat_rounds: bool,
    /// After a block-reduction window saves blocks, go on with the next
    /// window instead of starting the pass over at the first one, so later
    /// windows (and the Z windows after the Y ones) get their turn.
    pub continue_after_gain: bool,
    /// Fix the signals of the cells outside each window to the current
    /// layout's, so a window solve does not re-justify the rest of the box
    /// (see `ConstructionConfig::given_frozen_signals`).
    pub given_outside_signals: bool,
    /// After a slice-removal repair, spend up to this long minimizing the
    /// same window, so repairs (which only have to fit) do not add dead
    /// blocks. `None` keeps the repair as found.
    pub repair_optimize: Option<Duration>,
    /// Slice removals per round before block reduction gets a turn. A long
    /// constructed layout can otherwise spend the whole budget removing
    /// slices (the 2-bit adder did, and kept its dead blocks). `None` (the
    /// default) removes slices until none can be: with a cap of 4 the full
    /// adder ran out of time at 2x11x7 with 79 blocks instead of converging
    /// at 2x8x8 with 60.
    pub max_removals_per_round: Option<usize>,
    /// Model params for every window (see `ExactPlacerConfig::model_params`).
    pub model_params: BTreeMap<String, rsdsl::IValue>,
    /// Simulator rejections each attempt's workers may hit before giving up.
    pub max_refinements: usize,
    /// Search and verification constants for every attempt.
    pub tuning: ExactTuning,
    pub attempt_time_limit: Duration,
    pub time_limit: Option<Duration>,
    pub workers: usize,
    pub seed: u32,
    pub rank_levels: usize,
    pub stage_levels: usize,
    /// Axes to shrink: 1 = Y (length), 2 = Z (height).
    pub axes: Vec<usize>,
    pub output_policies: BTreeMap<String, OutputPolicy>,
    /// Diagnostic only: see `ExactPlacerConfig::no_fold`.
    #[doc(hidden)]
    pub no_fold: bool,
    /// Keep the layout a tile that repeats along X (see
    /// `ExactPlacerConfig::carry`).
    pub carry: Option<super::CarryTiling>,
    /// Write the layout after every accepted change to this directory, for
    /// watching compaction in the viewer (`progress.rs`).
    pub progress: Option<std::path::PathBuf>,
    /// Timing first: no change may make an output settle later (redstone
    /// ticks, `timing.rs`), and each round ends by shortening the outputs'
    /// paths window by window (`ExactPlacerConfig::timing`).
    pub timing: bool,
    /// With `timing`: the outputs whose paths to shorten, most important
    /// first (a tile's carry before its sum, say). Empty: every output,
    /// slowest first.
    pub delay_outputs: Vec<String>,
    /// Slices re-solved together in each delay window.
    pub delay_window: usize,
    /// Wider delay windows tried once a full pass at the current width
    /// shortens nothing.
    pub max_delay_window: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            window_radius: 1,
            max_window_radius: 2,
            minimize_blocks: true,
            optimize_windows: true,
            reduction_window: 3,
            reduction_axes: vec![1, 2],
            repeat_rounds: true,
            continue_after_gain: true,
            given_outside_signals: true,
            repair_optimize: Some(Duration::from_secs(5)),
            max_removals_per_round: None,
            model_params: BTreeMap::new(),
            max_refinements: 8,
            tuning: ExactTuning::default(),
            attempt_time_limit: Duration::from_secs(20),
            time_limit: None,
            workers: 8,
            seed: 1,
            rank_levels: 24,
            stage_levels: 24,
            axes: vec![1, 2],
            output_policies: BTreeMap::new(),
            no_fold: false,
            carry: None,
            progress: None,
            timing: false,
            delay_outputs: Vec::new(),
            delay_window: 3,
            max_delay_window: 5,
        }
    }
}

/// Window solves of one kind and outcome, and the time they took.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttemptTime {
    pub count: usize,
    /// Wall time of the whole attempt.
    pub wall: Duration,
    /// Grounding the model (`ExactPlacerStats::encode_time`).
    pub encode: Duration,
    /// SAT solving with simulator checks (`ExactPlacerStats::solve_time`).
    pub solve: Duration,
}

thread_local! {
    static ATTEMPTS: std::cell::RefCell<BTreeMap<(&'static str, &'static str), AttemptTime>> =
        Default::default();
}

/// Adds one window solve to the running compaction's `attempt_times`.
fn record_attempt(
    phase: &'static str,
    outcome: &'static str,
    started: Instant,
    stats: Option<&super::ExactPlacerStats>,
) {
    ATTEMPTS.with(|attempts| {
        let mut attempts = attempts.borrow_mut();
        let entry = attempts.entry((phase, outcome)).or_default();
        entry.count += 1;
        entry.wall += started.elapsed();
        if let Some(stats) = stats {
            entry.encode += stats.encode_time;
            entry.solve += stats.solve_time;
        }
    });
}

#[derive(Debug, Clone, Default)]
pub struct CompactionReport {
    /// Accepted `(axis, slice)` removals in order.
    pub removed: Vec<(usize, usize)>,
    pub attempts: usize,
    /// Accepted window re-solves that saved at least one block.
    pub block_reductions: usize,
    /// Accepted window re-solves that made an output settle sooner.
    pub delay_reductions: usize,
    /// Rounds of slice removal followed by block reduction.
    pub rounds: usize,
    /// Every window solve by `(phase, outcome)`: phases `remove` (slice
    /// removal repairs), `repair` (minimizing a repair), `reduce` (block
    /// reduction), `delay` and `delay-repair` (path shortening), `signals`;
    /// outcomes `placed`, `infeasible` (proven), `unknown` (out of time),
    /// `error` (not encodable), and for minimization `improved`,
    /// `improved-optimal`, `optimal` (proven nothing smaller).
    pub attempt_times: BTreeMap<(&'static str, &'static str), AttemptTime>,
    pub elapsed: Duration,
}

/// `y` or `z`, for progress frame labels.
fn axis_name(axis: usize) -> &'static str {
    if axis == 1 {
        "y"
    } else {
        "z"
    }
}

impl ExactLocalPlacer {
    /// Repeatedly removes the slice whose repair succeeds, emptiest first,
    /// widening the repair window when stuck, until no single-slice removal
    /// can be repaired within the limits. Then optionally trades blocks away,
    /// and with `repeat_rounds` starts over while that keeps saving blocks.
    pub fn compact(
        &self,
        mut layout: ExactLayout,
        config: &CompactionConfig,
    ) -> eyre::Result<(ExactLayout, Option<ExactPlacement>, CompactionReport)> {
        let started = Instant::now();
        let mut report = CompactionReport::default();
        let mut best = None;
        ATTEMPTS.with(|attempts| attempts.borrow_mut().clear());
        let expired = || {
            config
                .time_limit
                .is_some_and(|limit| started.elapsed() >= limit)
        };
        if config.timing {
            // A layout with feedback has no delays to keep.
            layout.delays = layout
                .timing()
                .map(|timing| timing.outputs.into_iter().collect())
                .unwrap_or_default();
        }
        if config.given_outside_signals && layout.signals.is_empty() {
            self.read_signals(&mut layout, config);
        }
        loop {
            report.rounds += 1;
            let removed = report.removed.len();
            self.remove_slices(&mut layout, &mut best, &mut report, config, &expired);
            // A capped round may have left removable slices behind.
            let capped = config
                .max_removals_per_round
                .is_some_and(|cap| report.removed.len() - removed >= cap);
            if !(config.minimize_blocks || config.timing) || expired() {
                break;
            }
            let reductions = report.block_reductions;
            if config.minimize_blocks {
                self.reduce_blocks(&mut layout, &mut best, &mut report, config, &expired);
            }
            let reduced = report.block_reductions > reductions;
            // Paths last: removal and reduction already keep every delay,
            // and a path window that cannot shorten costs a whole attempt.
            // Shortening first, a constructed full adder spent its 600 s on
            // such windows and was not compacted at all.
            let shortenings = report.delay_reductions;
            if config.timing && !expired() {
                self.reduce_delays(&mut layout, &mut best, &mut report, config, &expired);
            }
            let shortened = report.delay_reductions > shortenings;
            if !config.repeat_rounds || !(reduced || capped || shortened) || expired() {
                break;
            }
        }
        report.elapsed = started.elapsed();
        report.attempt_times =
            ATTEMPTS.with(|attempts| std::mem::take(&mut *attempts.borrow_mut()));
        Ok((layout, best, report))
    }

    fn remove_slices(
        &self,
        layout: &mut ExactLayout,
        best: &mut Option<ExactPlacement>,
        report: &mut CompactionReport,
        config: &CompactionConfig,
        expired: &impl Fn() -> bool,
    ) {
        let mut radius = config.window_radius;
        let mut removals = 0;
        'outer: loop {
            if config
                .max_removals_per_round
                .is_some_and(|cap| removals >= cap)
            {
                break;
            }
            for &axis in &config.axes {
                let length = if axis == 1 {
                    layout.dim.1
                } else {
                    layout.dim.2
                };
                let mut order = (0..length).collect::<Vec<_>>();
                order.sort_by_key(|&index| {
                    layout
                        .cells
                        .keys()
                        .filter(|position| {
                            (if axis == 1 { position.1 } else { position.2 }) == index
                        })
                        .count()
                });
                for index in order {
                    if expired() {
                        break 'outer;
                    }
                    let Some(cut) = layout.without_slice(axis, index) else {
                        continue;
                    };
                    report.attempts += 1;
                    let window = (index.saturating_sub(radius), index + radius);
                    if let Some(mut placement) =
                        self.resolve_window("remove", &cut, axis, window, None, config)
                    {
                        if let Some(budget) = config.repair_optimize {
                            // A repair only has to fit; trade its dead wires
                            // away while the window is still the one solved.
                            let repaired = ExactLayout::from_placement(cut.dim, &placement);
                            let quick = CompactionConfig {
                                attempt_time_limit: budget,
                                ..config.clone()
                            };
                            let limit = repaired.cells.len().saturating_sub(1);
                            if let (Some(better), _) = self.optimize_window(
                                "repair",
                                &repaired,
                                axis,
                                window,
                                Some(limit),
                                &quick,
                            ) {
                                placement = better;
                            }
                        }
                        tracing::info!(
                            axis,
                            index,
                            radius,
                            blocks = placement.block_count,
                            "compaction step"
                        );
                        super::progress::record_frame(
                            config.progress.as_deref(),
                            &self.name,
                            &format!("remove {}{index}", axis_name(axis)),
                            &placement,
                        );
                        *layout = ExactLayout::from_placement(cut.dim, &placement);
                        *best = Some(placement);
                        report.removed.push((axis, index));
                        removals += 1;
                        radius = config.window_radius;
                        continue 'outer;
                    }
                }
            }
            if radius < config.max_window_radius {
                radius += 1;
                continue;
            }
            break;
        }
    }

    /// Slides a `reduction_window`-slice window along each reduction axis,
    /// asking for fewer blocks, until a full pass saves nothing. With
    /// `optimize_windows` each window is minimized in one solve, and a window
    /// proven optimal stays done until the layout around it changes.
    fn reduce_blocks(
        &self,
        layout: &mut ExactLayout,
        best: &mut Option<ExactPlacement>,
        report: &mut CompactionReport,
        config: &CompactionConfig,
        expired: &impl Fn() -> bool,
    ) {
        let optimize = config.optimize_windows;
        // Block reduction keeps the box, so the windows stay the same.
        let windows = config
            .reduction_axes
            .iter()
            .flat_map(|&axis| {
                let length = if axis == 1 {
                    layout.dim.1
                } else {
                    layout.dim.2
                };
                (0..length
                    .saturating_sub(config.reduction_window.saturating_sub(1))
                    .max(1))
                    .map(move |low| (axis, low))
            })
            .collect::<Vec<_>>();
        let mut settled = std::collections::BTreeSet::new();
        let mut next = 0;
        // Windows tried since the last gain; a full pass without one ends.
        let mut idle = 0;
        while idle < windows.len() {
            if expired() {
                break;
            }
            let (axis, low) = windows[next];
            next = (next + 1) % windows.len();
            idle += 1;
            if settled.contains(&(axis, low)) {
                continue;
            }
            let limit = layout.cells.len().saturating_sub(1);
            report.attempts += 1;
            let window = (low, low + config.reduction_window);
            let (placement, optimal) = if optimize {
                self.optimize_window("reduce", layout, axis, window, Some(limit), config)
            } else {
                (
                    self.resolve_window("reduce", layout, axis, window, Some(limit), config),
                    false,
                )
            };
            if optimal {
                settled.insert((axis, low));
            }
            let Some(placement) = placement else {
                continue;
            };
            tracing::info!(
                axis,
                low,
                blocks = placement.block_count,
                optimal,
                "compaction block reduction"
            );
            super::progress::record_frame(
                config.progress.as_deref(),
                &self.name,
                &format!(
                    "reduce {}{low}-{}",
                    axis_name(axis),
                    low + config.reduction_window - 1
                ),
                &placement,
            );
            *layout = ExactLayout::from_placement(layout.dim, &placement);
            *best = Some(placement);
            report.block_reductions += 1;
            // The change alters every other window's fixed cells; this
            // window's own outside did not change, so its proof still holds.
            settled.clear();
            if optimal {
                settled.insert((axis, low));
            }
            idle = 0;
            if !config.continue_after_gain {
                next = 0;
            }
        }
    }

    /// Shortens the outputs' paths, one output at a time in priority order:
    /// slides a `delay_window`-slice window along each reduction axis, asking
    /// for the output a tick sooner and every other output no later, and
    /// widens the window after a full pass that gains nothing, until
    /// `max_delay_window` gains nothing either or the output reaches its
    /// logic depth.
    fn reduce_delays(
        &self,
        layout: &mut ExactLayout,
        best: &mut Option<ExactPlacement>,
        report: &mut CompactionReport,
        config: &CompactionConfig,
        expired: &impl Fn() -> bool,
    ) {
        let bounds = super::timing::depth_bounds(&self.netlist);
        let targets = if config.delay_outputs.is_empty() {
            let mut outputs = layout.delays.clone().into_iter().collect::<Vec<_>>();
            outputs.sort_by_key(|(name, ticks)| (std::cmp::Reverse(*ticks), name.clone()));
            outputs.into_iter().map(|(name, _)| name).collect()
        } else {
            config.delay_outputs.clone()
        };
        let windows_of = |layout: &ExactLayout, size: usize| {
            config
                .reduction_axes
                .iter()
                .flat_map(|&axis| {
                    let length = if axis == 1 {
                        layout.dim.1
                    } else {
                        layout.dim.2
                    };
                    (0..length.saturating_sub(size.saturating_sub(1)).max(1))
                        .map(move |low| (axis, low))
                })
                .collect::<Vec<_>>()
        };
        for target in targets {
            let mut size = config.delay_window;
            let mut windows = windows_of(layout, size);
            let mut next = 0;
            let mut idle = 0;
            loop {
                if expired() {
                    return;
                }
                let Some(&ticks) = layout.delays.get(&target) else {
                    break;
                };
                if bounds.get(&target).is_some_and(|&bound| ticks <= bound) {
                    break;
                }
                if idle >= windows.len() {
                    if size >= config.max_delay_window {
                        break;
                    }
                    size += 1;
                    windows = windows_of(layout, size);
                    next = 0;
                    idle = 0;
                }
                let (axis, low) = windows[next];
                next = (next + 1) % windows.len();
                idle += 1;
                let window = (low, low + size);
                // A window off the slowest path cannot shorten it.
                let on_path = critical_path(layout, &target).iter().any(|position| {
                    let value = if axis == 1 { position.1 } else { position.2 };
                    value >= window.0 && value < window.1
                });
                if !on_path {
                    continue;
                }
                let mut sooner = layout.clone();
                sooner.delays.insert(target.clone(), ticks - 1);
                report.attempts += 1;
                let Some(mut placement) =
                    self.resolve_window("delay", &sooner, axis, window, None, config)
                else {
                    continue;
                };
                if let Some(budget) = config.repair_optimize {
                    // Keep the shorter path, then trade dead blocks away.
                    let found = ExactLayout::from_placement(layout.dim, &placement);
                    let quick = CompactionConfig {
                        attempt_time_limit: budget,
                        ..config.clone()
                    };
                    let limit = found.cells.len().saturating_sub(1);
                    if let (Some(better), _) = self.optimize_window(
                        "delay-repair",
                        &found,
                        axis,
                        window,
                        Some(limit),
                        &quick,
                    ) {
                        placement = better;
                    }
                }
                let now = placement.delays.get(&target).copied().unwrap_or(ticks);
                tracing::info!(
                    axis,
                    low,
                    size,
                    output = target.as_str(),
                    ticks = now,
                    blocks = placement.block_count,
                    "compaction delay reduction"
                );
                super::progress::record_frame(
                    config.progress.as_deref(),
                    &self.name,
                    &format!(
                        "delay {target} {now} {}{low}-{}",
                        axis_name(axis),
                        low + size - 1
                    ),
                    &placement,
                );
                *layout = ExactLayout::from_placement(layout.dim, &placement);
                *best = Some(placement);
                report.delay_reductions += 1;
                idle = 0;
            }
        }
    }

    /// Minimizes the block count inside a window along `axis` (at most
    /// `limit` blocks overall). Returns the improved placement, if any, and
    /// whether the window is proven optimal (no layout with fewer blocks
    /// exists there).
    fn optimize_window(
        &self,
        phase: &'static str,
        layout: &ExactLayout,
        axis: usize,
        window: (usize, usize),
        limit: Option<usize>,
        config: &CompactionConfig,
    ) -> (Option<ExactPlacement>, bool) {
        let started = Instant::now();
        match self.window_config(layout, axis, window, limit, config) {
            Ok(mut exact) => {
                exact.optimize = true;
                match self.place(&exact) {
                    Ok((ExactOutcome::Placed(placement), stats)) => {
                        let outcome = if stats.optimal {
                            "improved-optimal"
                        } else {
                            "improved"
                        };
                        record_attempt(phase, outcome, started, Some(&stats));
                        (Some(*placement), stats.optimal)
                    }
                    // No layout with fewer blocks: the window is optimal.
                    Ok((ExactOutcome::Infeasible, stats)) => {
                        record_attempt(phase, "optimal", started, Some(&stats));
                        (None, true)
                    }
                    Ok((_, stats)) => {
                        record_attempt(phase, "unknown", started, Some(&stats));
                        (None, false)
                    }
                    Err(error) => {
                        record_attempt(phase, "error", started, None);
                        tracing::debug!(?window, %error, "compaction window rejected");
                        (None, false)
                    }
                }
            }
            Err(error) => {
                record_attempt(phase, "error", started, None);
                tracing::debug!(?window, %error, "compaction window rejected");
                (None, false)
            }
        }
    }

    /// Re-solves the slices `window.0..window.1` along `axis`, keeping every
    /// other cell. Returns `None` when no verified layout is found in time, or
    /// when the fixed cells cannot be encoded (for example, a seam that leaves
    /// a block unsupported).
    fn resolve_window(
        &self,
        phase: &'static str,
        layout: &ExactLayout,
        axis: usize,
        window: (usize, usize),
        max_blocks: Option<usize>,
        config: &CompactionConfig,
    ) -> Option<ExactPlacement> {
        self.try_resolve_window(phase, layout, axis, window, max_blocks, config)
            .unwrap_or_else(|error| {
                tracing::debug!(axis, ?window, %error, "compaction window rejected");
                None
            })
    }

    fn try_resolve_window(
        &self,
        phase: &'static str,
        cut: &ExactLayout,
        axis: usize,
        window: (usize, usize),
        max_blocks: Option<usize>,
        config: &CompactionConfig,
    ) -> eyre::Result<Option<ExactPlacement>> {
        let started = Instant::now();
        let solved = self
            .window_config(cut, axis, window, max_blocks, config)
            .and_then(|exact| self.place(&exact));
        let (outcome, stats) = match solved {
            Ok(solved) => solved,
            Err(error) => {
                record_attempt(phase, "error", started, None);
                return Err(error);
            }
        };
        let name = match &outcome {
            ExactOutcome::Placed(_) => "placed",
            ExactOutcome::Infeasible => "infeasible",
            ExactOutcome::Unknown { .. } => "unknown",
        };
        record_attempt(phase, name, started, Some(&stats));
        Ok(match outcome {
            ExactOutcome::Placed(placement) => Some(*placement),
            _ => None,
        })
    }

    /// A layout read from RCELL has no solver signals yet: solve it once with
    /// every cell fixed to read them off.
    pub(super) fn read_signals(&self, layout: &mut ExactLayout, config: &CompactionConfig) {
        match self.try_resolve_window("signals", layout, 1, (0, 0), None, config) {
            Ok(Some(placement)) => layout.signals = placement.signals.into_iter().collect(),
            Ok(None) => tracing::warn!("could not read the layout's signals"),
            Err(error) => tracing::warn!(%error, "could not read the layout's signals"),
        }
    }

    /// The placer configuration that frees `window` along `axis` and fixes
    /// every other cell of `cut`.
    pub(super) fn window_config(
        &self,
        cut: &ExactLayout,
        axis: usize,
        window: (usize, usize),
        max_blocks: Option<usize>,
        config: &CompactionConfig,
    ) -> eyre::Result<ExactPlacerConfig> {
        let dim = cut.dim;
        let coordinate = |position: Position| if axis == 1 { position.1 } else { position.2 };
        let (low, high) = window;
        let in_window = |position: Position| {
            let value = coordinate(position);
            value >= low && value < high
        };
        let mut exact = ExactPlacerConfig::new(dim);
        exact.workers = config.workers;
        exact.seed = config.seed;
        exact.rank_levels = config.rank_levels;
        exact.stage_levels = config.stage_levels;
        exact.no_fold = config.no_fold;
        exact.carry = config.carry.clone();
        exact.time_limit = Some(config.attempt_time_limit);
        exact.max_refinements = config.max_refinements;
        exact.tuning = config.tuning.clone();
        exact.model_params = config.model_params.clone();
        exact.max_blocks = max_blocks;
        if config.timing {
            // `Stage` only has to hold arrival times: the longest path in
            // the box, with room for a window's dead ends to run a little
            // longer (a cut may not be analyzable; its parent's delays still
            // bound the outputs). Fewer levels than the default also solve
            // faster (XOR 2x6x4 in 5 ticks: 3 s with 6 levels, no answer in
            // 90 s with 24).
            let longest = cut
                .timing()
                .ok()
                .and_then(|timing| timing.arrival.values().copied().max())
                .unwrap_or(0);
            let slowest = cut.delays.values().copied().max().unwrap_or(0);
            exact.timing = true;
            exact.stage_levels = longest.max(slowest) + 2;
            exact.output_delays = cut.delays.clone();
        }
        for (name, position, attach) in &cut.inputs {
            exact = exact.with_input_site(name.clone(), *position, *attach);
        }
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                for z in 0..dim.2 {
                    let position = Position(x, y, z);
                    if in_window(position) {
                        continue;
                    }
                    let kind = cut.cells.get(&position).copied().unwrap_or(CellKind::Air);
                    exact.fixed_cells.insert(position, kind);
                    if config.given_outside_signals {
                        if let Some(&function) = cut.signals.get(&position) {
                            exact.given_signals.insert(position, function);
                        }
                    }
                }
            }
        }
        for (name, policy) in &config.output_policies {
            if *policy == OutputPolicy::MaxYFace {
                let face = (0..dim.2)
                    .flat_map(|z| (0..dim.0).map(move |x| Position(x, dim.1 - 1, z)))
                    .collect::<Vec<_>>();
                exact = exact.with_output_sites(name.clone(), face);
                exact.driving_outputs.insert(name.clone());
            }
        }
        Ok(exact)
    }
}

/// The cells of the slowest path to `output` (empty without one).
fn critical_path(layout: &ExactLayout, output: &str) -> Vec<Position> {
    let Some(&(_, position)) = layout.outputs.iter().find(|(name, _)| name == output) else {
        return Vec::new();
    };
    layout
        .timing()
        .map(|timing| timing.path(position))
        .unwrap_or_default()
}

impl ExactLocalPlacer {
    /// Builds a long valid layout window by window, then compacts it.
    pub fn synthesize(
        &self,
        construction: &super::ConstructionConfig,
        compaction: &CompactionConfig,
    ) -> eyre::Result<(
        ExactLayout,
        ExactPlacement,
        super::ConstructionReport,
        CompactionReport,
    )> {
        let (layout, placement, built) = self.construct(construction)?;
        let (layout, compacted, report) = self.compact(layout, compaction)?;
        Ok((layout, compacted.unwrap_or(placement), built, report))
    }
}
