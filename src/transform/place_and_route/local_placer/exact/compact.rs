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
use super::{ExactLocalPlacer, ExactOutcome, ExactPlacement, ExactPlacerConfig};
use crate::world::position::Position;

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Slices freed on each side of the seam.
    pub window_radius: usize,
    /// Wider windows tried once no seam can be repaired at the current radius.
    pub max_window_radius: usize,
    /// After shrinking, re-solve sliding windows with one block fewer.
    pub minimize_blocks: bool,
    pub attempt_time_limit: Duration,
    pub time_limit: Option<Duration>,
    pub workers: usize,
    pub seed: u32,
    pub rank_levels: usize,
    pub stage_levels: usize,
    /// Axes to shrink: 1 = Y (length), 2 = Z (height).
    pub axes: Vec<usize>,
    pub output_policies: BTreeMap<String, OutputPolicy>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            window_radius: 1,
            max_window_radius: 2,
            minimize_blocks: true,
            attempt_time_limit: Duration::from_secs(20),
            time_limit: None,
            workers: 8,
            seed: 1,
            rank_levels: 24,
            stage_levels: 24,
            axes: vec![1, 2],
            output_policies: BTreeMap::new(),
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
    pub elapsed: Duration,
}

impl ExactLocalPlacer {
    /// Repeatedly removes the slice whose repair succeeds, emptiest first,
    /// widening the repair window when stuck, until no single-slice removal
    /// can be repaired within the limits. Then optionally trades blocks away.
    pub fn compact(
        &self,
        mut layout: ExactLayout,
        config: &CompactionConfig,
    ) -> eyre::Result<(ExactLayout, Option<ExactPlacement>, CompactionReport)> {
        let started = Instant::now();
        let mut report = CompactionReport::default();
        let mut best = None;
        let expired = || config.time_limit.is_some_and(|limit| started.elapsed() >= limit);
        let mut radius = config.window_radius;
        'outer: loop {
            for &axis in &config.axes {
                let length = if axis == 1 { layout.dim.1 } else { layout.dim.2 };
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
                    if let Some(placement) = self.resolve_window(&cut, axis, window, None, config) {
                        tracing::info!(axis, index, radius, blocks = placement.block_count, "compaction step");
                        layout = ExactLayout::from_placement(cut.dim, &placement);
                        best = Some(placement);
                        report.removed.push((axis, index));
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
        if config.minimize_blocks {
            // Slide a three-slice window along Y, asking for one block fewer.
            'shrink: loop {
                for low in 0..layout.dim.1 {
                    if expired() {
                        break 'shrink;
                    }
                    let limit = layout.cells.len().saturating_sub(1);
                    report.attempts += 1;
                    let window = (low, low + 3);
                    if let Some(placement) = self.resolve_window(&layout, 1, window, Some(limit), config) {
                        tracing::info!(low, blocks = placement.block_count, "compaction block reduction");
                        layout = ExactLayout::from_placement(layout.dim, &placement);
                        best = Some(placement);
                        report.block_reductions += 1;
                        continue 'shrink;
                    }
                }
                break;
            }
        }
        report.elapsed = started.elapsed();
        Ok((layout, best, report))
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
        exact.time_limit = Some(config.attempt_time_limit);
        exact.max_refinements = 8;
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
        let (outcome, _) = self.place(&exact)?;
        Ok(match outcome {
            ExactOutcome::Placed(placement) => Some(*placement),
            _ => None,
        })
    }
}

impl ExactLocalPlacer {
    /// Builds a long valid layout window by window, then compacts it.
    pub fn synthesize(
        &self,
        construction: &super::ConstructionConfig,
        compaction: &CompactionConfig,
    ) -> eyre::Result<(ExactLayout, ExactPlacement, super::ConstructionReport, CompactionReport)> {
        let (layout, placement, built) = self.construct(construction)?;
        let (layout, compacted, report) = self.compact(layout, compaction)?;
        Ok((layout, compacted.unwrap_or(placement), built, report))
    }
}
