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
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompactionReport {
    /// Accepted `(axis, slice)` removals in order.
    pub removed: Vec<(usize, usize)>,
    pub attempts: usize,
    /// Accepted window re-solves that saved at least one block.
    pub block_reductions: usize,
    /// Rounds of slice removal followed by block reduction.
    pub rounds: usize,
    pub elapsed: Duration,
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
        let expired = || {
            config
                .time_limit
                .is_some_and(|limit| started.elapsed() >= limit)
        };
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
            if !config.minimize_blocks || expired() {
                break;
            }
            let reductions = report.block_reductions;
            self.reduce_blocks(&mut layout, &mut best, &mut report, config, &expired);
            let reduced = report.block_reductions > reductions;
            if !config.repeat_rounds || !(reduced || capped) || expired() {
                break;
            }
        }
        report.elapsed = started.elapsed();
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
                        self.resolve_window(&cut, axis, window, None, config)
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
                            if let (Some(better), _) =
                                self.optimize_window(&repaired, axis, window, limit, &quick)
                            {
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
                self.optimize_window(layout, axis, window, limit, config)
            } else {
                (
                    self.resolve_window(layout, axis, window, Some(limit), config),
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

    /// Minimizes the block count inside a window along `axis` (at most
    /// `limit` blocks overall). Returns the improved placement, if any, and
    /// whether the window is proven optimal (no layout with fewer blocks
    /// exists there).
    fn optimize_window(
        &self,
        layout: &ExactLayout,
        axis: usize,
        window: (usize, usize),
        limit: usize,
        config: &CompactionConfig,
    ) -> (Option<ExactPlacement>, bool) {
        match self.window_config(layout, axis, window, Some(limit), config) {
            Ok(mut exact) => {
                exact.optimize = true;
                match self.place(&exact) {
                    Ok((ExactOutcome::Placed(placement), stats)) => {
                        (Some(*placement), stats.optimal)
                    }
                    // No layout with fewer blocks: the window is optimal.
                    Ok((ExactOutcome::Infeasible, _)) => (None, true),
                    Ok(_) => (None, false),
                    Err(error) => {
                        tracing::debug!(?window, %error, "compaction window rejected");
                        (None, false)
                    }
                }
            }
            Err(error) => {
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
        layout: &ExactLayout,
        axis: usize,
        window: (usize, usize),
        max_blocks: Option<usize>,
        config: &CompactionConfig,
    ) -> Option<ExactPlacement> {
        self.try_resolve_window(layout, axis, window, max_blocks, config)
            .unwrap_or_else(|error| {
                tracing::debug!(axis, ?window, %error, "compaction window rejected");
                None
            })
    }

    fn try_resolve_window(
        &self,
        cut: &ExactLayout,
        axis: usize,
        window: (usize, usize),
        max_blocks: Option<usize>,
        config: &CompactionConfig,
    ) -> eyre::Result<Option<ExactPlacement>> {
        let exact = self.window_config(cut, axis, window, max_blocks, config)?;
        let (outcome, _) = self.place(&exact)?;
        Ok(match outcome {
            ExactOutcome::Placed(placement) => Some(*placement),
            _ => None,
        })
    }

    /// A layout read from RCELL has no solver signals yet: solve it once with
    /// every cell fixed to read them off.
    pub(super) fn read_signals(&self, layout: &mut ExactLayout, config: &CompactionConfig) {
        match self.try_resolve_window(layout, 1, (0, 0), None, config) {
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
        exact.time_limit = Some(config.attempt_time_limit);
        exact.max_refinements = config.max_refinements;
        exact.tuning = config.tuning.clone();
        exact.model_params = config.model_params.clone();
        exact.max_blocks = max_blocks;
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
