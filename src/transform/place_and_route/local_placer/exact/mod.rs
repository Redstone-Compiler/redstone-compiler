//! Exact local placement with a SAT solver (CaDiCaL).
//!
//! The beam-search placer commits one route at a time, so early routes can
//! seal off later fan-out branches. This placer instead encodes every block,
//! every net label, and the simulator's power-propagation rules for the whole
//! box as one CNF formula (see `encode`). Placement and routing are solved
//! simultaneously, so a satisfying assignment is a complete layout.
//!
//! Each solution is rebuilt as a world and checked with the simulator in all
//! input cases and settled transitions. A layout that fails is excluded with a
//! blocking clause and the search resumes incrementally. Several solver
//! instances with different seeds run in parallel; the first verified layout
//! wins. An `Unsat` answer proves that no layout exists in the box under the
//! encoded rules and limits (`rank_levels`, `max_blocks`, pin sites).
//!
//! Design notes: `docs/exact_local_placer.md`.

mod acyclic;
mod cnf;
mod compact;
mod construct;
mod encode;
mod layout;
mod netlist;
mod solver;
#[cfg(test)]
mod tests;
mod verify;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub use compact::{CompactionConfig, CompactionReport};
pub use construct::{ConstructionConfig, ConstructionReport};
pub use encode::CellKind;
pub use layout::{ExactLayout, InputPolicy, OutputPolicy};
pub use netlist::{Net, NetDriver, NetId, NorNetlist};
pub use verify::ExactVerificationFailure;

use self::encode::Encoding;
use self::solver::{SatSolver, SolveResult, StopSignal};
use crate::graph::logic::LogicGraph;
use crate::output::{OutputEndpoint, PlacedWorld};
use crate::physical_cell::PhysicalCellDocument;
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

#[derive(Debug, Clone)]
pub struct ExactPlacerConfig {
    pub dim: DimSize,
    /// Candidate `(switch position, attach direction)` per input. Inputs not
    /// listed may use any boundary cell.
    pub input_sites: BTreeMap<String, Vec<(Position, Direction)>>,
    /// Candidate observation cells per output. Outputs not listed may use any cell.
    pub output_sites: BTreeMap<String, Vec<Position>>,
    /// Outputs that must drive the cell beyond the box: a torch, or a repeater
    /// whose output side is outside, on one of the output's sites.
    pub driving_outputs: std::collections::BTreeSet<String>,
    /// Replaces the netlist outputs with these nets, each observable on one of
    /// its candidate cells (used for partial layouts and interfaces).
    pub observations: Option<Vec<(NetId, Vec<Position>)>>,
    /// Inputs that have no switch in this layout (not placed yet).
    pub absent_inputs: std::collections::BTreeSet<String>,
    /// Cells whose block kind is fixed.
    pub fixed_cells: BTreeMap<Position, CellKind>,
    /// Partial layouts to exclude: at least one listed cell must differ.
    pub blocked: Vec<Vec<(Position, CellKind)>>,
    /// Upper bound on a power chain between two torch stages (local rank).
    /// Zero enforces acyclic power lazily with loop formulas instead; that is
    /// complete but converged far slower in measurements.
    pub rank_levels: usize,
    /// Upper bound on torches plus repeaters in series.
    /// Zero forbids torch feedback lazily instead.
    pub stage_levels: usize,
    /// Optional upper bound on non-air blocks.
    pub max_blocks: Option<usize>,
    /// Parallel solver instances with distinct seeds.
    pub workers: usize,
    pub seed: u32,
    pub time_limit: Option<Duration>,
    /// Maximum simulator rejections per worker before giving up.
    pub max_refinements: usize,
    /// Allow dust and repeaters that are never powered. They are harmless but
    /// useless, so synthesis leaves them out; checking hand-made cells may not.
    pub allow_unpowered_wires: bool,
    /// Diagnostic only: guard each relation's soundness clauses with a
    /// literal that tests can assume false to locate model/simulator mismatches.
    #[doc(hidden)]
    pub relax_soundness: bool,
}

impl ExactPlacerConfig {
    pub fn new(dim: DimSize) -> Self {
        Self {
            dim,
            input_sites: BTreeMap::new(),
            output_sites: BTreeMap::new(),
            driving_outputs: Default::default(),
            observations: None,
            absent_inputs: Default::default(),
            fixed_cells: BTreeMap::new(),
            blocked: Vec::new(),
            rank_levels: 24,
            stage_levels: 24,
            max_blocks: None,
            workers: 1,
            seed: 1,
            time_limit: None,
            max_refinements: 64,
            allow_unpowered_wires: false,
            relax_soundness: false,
        }
    }

    pub fn with_input_site(
        mut self,
        name: impl Into<String>,
        position: Position,
        attach: Direction,
    ) -> Self {
        self.input_sites
            .entry(name.into())
            .or_default()
            .push((position, attach));
        self
    }

    pub fn with_output_sites(
        mut self,
        name: impl Into<String>,
        positions: impl IntoIterator<Item = Position>,
    ) -> Self {
        self.output_sites
            .entry(name.into())
            .or_default()
            .extend(positions);
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExactPlacerStats {
    pub variables: i32,
    pub clauses: usize,
    pub encode_time: Duration,
    pub solve_time: Duration,
    /// Simulator rejections across all workers.
    pub refinements: usize,
    /// Lazy loop formulas and feedback cuts added across all workers.
    pub loop_formulas: usize,
    pub feedback_cuts: usize,
    pub winning_seed: Option<u32>,
    pub relations: usize,
    /// Cumulative `(section, variables, clauses)` after each encoding step.
    pub sections: Vec<(&'static str, i32, usize)>,
}

#[derive(Debug, Clone)]
pub struct ExactPlacement {
    pub placed: PlacedWorld,
    pub cells: Vec<(Position, CellKind)>,
    pub block_count: usize,
    pub rcell: PhysicalCellDocument,
}

#[derive(Debug, Clone)]
pub enum ExactOutcome {
    Placed(Box<ExactPlacement>),
    /// The solver proved that no layout satisfies the encoded constraints.
    Infeasible,
    /// Time limit reached, or every worker exhausted its refinement budget.
    Unknown { last_rejection: Option<String> },
}

pub struct ExactLocalPlacer {
    netlist: NorNetlist,
    name: String,
}

enum WorkerResult {
    Placed(Box<ExactPlacement>, u32),
    Infeasible,
    Unknown(Option<String>),
}

impl ExactLocalPlacer {
    pub fn new(graph: &LogicGraph) -> eyre::Result<Self> {
        Ok(Self {
            netlist: NorNetlist::from_logic_graph(graph)?,
            name: "exact-local-cell".to_owned(),
        })
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn netlist(&self) -> &NorNetlist {
        &self.netlist
    }

    pub fn place(&self, config: &ExactPlacerConfig) -> eyre::Result<(ExactOutcome, ExactPlacerStats)> {
        let encode_started = Instant::now();
        let encoding = Encoding::build(&self.netlist, config)?;
        let mut stats = ExactPlacerStats {
            variables: encoding.cnf.num_vars(),
            clauses: encoding.cnf.clause_count(),
            encode_time: encode_started.elapsed(),
            relations: encoding.relations.len(),
            sections: encoding.sections.clone(),
            ..Default::default()
        };
        let solve_started = Instant::now();
        let deadline = config.time_limit.map(|limit| solve_started + limit);
        let stop = AtomicBool::new(false);
        let refinements = Mutex::new(0usize);
        let lazy = Mutex::new(acyclic::LazyStats::default());
        let workers = config.workers.max(1);
        let results = std::thread::scope(|scope| {
            let handles = (0..workers)
                .map(|worker| {
                    let encoding = &encoding;
                    let stop = &stop;
                    let refinements = &refinements;
                    let lazy = &lazy;
                    scope.spawn(move || {
                        let result = self.run_worker(
                            encoding,
                            config,
                            worker,
                            stop,
                            deadline,
                            refinements,
                            lazy,
                        );
                        if !matches!(result, WorkerResult::Unknown(_)) {
                            stop.store(true, Ordering::Relaxed);
                        }
                        result
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("exact placer worker panicked"))
                .collect::<Vec<_>>()
        });
        stats.solve_time = solve_started.elapsed();
        stats.refinements = *refinements.lock().unwrap();
        let lazy = *lazy.lock().unwrap();
        stats.loop_formulas = lazy.loop_formulas;
        stats.feedback_cuts = lazy.feedback_cuts;

        let mut last_rejection = None;
        let mut infeasible = false;
        for result in results {
            match result {
                WorkerResult::Placed(placement, seed) => {
                    stats.winning_seed = Some(seed);
                    return Ok((ExactOutcome::Placed(placement), stats));
                }
                WorkerResult::Infeasible => infeasible = true,
                WorkerResult::Unknown(rejection) => {
                    last_rejection = last_rejection.or(rejection);
                }
            }
        }
        let outcome = if infeasible {
            ExactOutcome::Infeasible
        } else {
            ExactOutcome::Unknown { last_rejection }
        };
        Ok((outcome, stats))
    }

    fn run_worker(
        &self,
        encoding: &Encoding,
        config: &ExactPlacerConfig,
        worker: usize,
        stop: &AtomicBool,
        deadline: Option<Instant>,
        refinements: &Mutex<usize>,
        lazy: &Mutex<acyclic::LazyStats>,
    ) -> WorkerResult {
        let seed = config.seed.wrapping_add(worker as u32 * 7919);
        let mut solver = SatSolver::new(seed);
        // Diversify the portfolio: sparse-first (phase 0) and stable-mode variants.
        solver.set_option("phase", i32::from(worker % 2 == 1));
        solver.set_option("stabilizeonly", i32::from(worker % 4 >= 2));
        solver.add_cnf(&encoding.cnf);
        let signal = StopSignal { stop, deadline };
        let index = acyclic::LazyIndex::new(encoding);
        let mut aux = acyclic::AuxVars::new(encoding);
        let mut last_rejection = None;
        let mut rejections = 0;
        while rejections <= config.max_refinements {
            match solver.solve(&[], &signal) {
                SolveResult::Unsat => {
                    // Blocking clauses only remove simulator-rejected layouts,
                    // so Unsat after a rejection is not an infeasibility proof.
                    return if last_rejection.is_none() {
                        WorkerResult::Infeasible
                    } else {
                        WorkerResult::Unknown(last_rejection)
                    };
                }
                SolveResult::Interrupted => return WorkerResult::Unknown(last_rejection),
                SolveResult::Sat => {}
            }
            let mut worker_lazy = acyclic::LazyStats::default();
            let added = acyclic::refine(encoding, &index, &mut solver, &mut aux, &mut worker_lazy);
            if added > 0 {
                let mut total = lazy.lock().unwrap();
                total.loop_formulas += worker_lazy.loop_formulas;
                total.feedback_cuts += worker_lazy.feedback_cuts;
                continue;
            }
            let decoded = verify::decode(encoding, &solver, &self.netlist);
            let world = verify::build_world(config.dim, &decoded.kinds);
            match verify::verify(&self.netlist, &decoded, &world) {
                Ok(()) => {
                    let placement = self.placement(config.dim, &decoded, world);
                    return WorkerResult::Placed(Box::new(placement), seed);
                }
                Err(failure) => {
                    tracing::debug!(?failure, "exact placer layout rejected by simulator");
                    last_rejection = Some(format!("{} at {:?}", failure.message, failure.position));
                    *refinements.lock().unwrap() += 1;
                    rejections += 1;
                    let blocking = decoded.kind_lits.iter().map(|&lit| -lit).collect::<Vec<_>>();
                    solver.add_clause(&blocking);
                }
            }
        }
        WorkerResult::Unknown(last_rejection)
    }

    fn placement(
        &self,
        dim: DimSize,
        decoded: &verify::Decoded,
        world: crate::world::World3D,
    ) -> ExactPlacement {
        let cells = decoded
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind != CellKind::Air)
            .map(|(cell, kind)| {
                (
                    Position(
                        cell % dim.0,
                        (cell / dim.0) % dim.1,
                        cell / (dim.0 * dim.1),
                    ),
                    *kind,
                )
            })
            .collect::<Vec<_>>();
        let block_count = cells
            .iter()
            .filter(|(_, kind)| !matches!(kind, CellKind::Switch(_)))
            .count();
        let rcell = verify::to_rcell(&self.name, dim, &self.netlist, decoded);
        ExactPlacement {
            placed: PlacedWorld {
                world,
                inputs: decoded
                    .inputs
                    .iter()
                    .map(|(name, position)| OutputEndpoint::new(name.clone(), *position))
                    .collect(),
                outputs: decoded
                    .outputs
                    .iter()
                    .map(|(name, position)| OutputEndpoint::new(name.clone(), *position))
                    .collect(),
            },
            cells,
            block_count,
            rcell,
        }
    }

    /// Finds a verified layout, then repeatedly asks for one with fewer blocks
    /// until the solver proves no smaller layout exists or time runs out.
    pub fn place_minimizing_blocks(
        &self,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<(Option<ExactPlacement>, bool, ExactPlacerStats)> {
        let started = Instant::now();
        let mut best: Option<ExactPlacement> = None;
        let mut total = ExactPlacerStats::default();
        let mut proven_optimal = false;
        let mut bound = config.max_blocks;
        loop {
            let remaining = config
                .time_limit
                .map(|limit| limit.saturating_sub(started.elapsed()));
            if remaining.is_some_and(|remaining| remaining.is_zero()) {
                break;
            }
            let attempt = ExactPlacerConfig {
                max_blocks: bound,
                time_limit: remaining,
                ..config.clone()
            };
            let (outcome, stats) = self.place(&attempt)?;
            total.variables = stats.variables;
            total.clauses = stats.clauses;
            total.encode_time += stats.encode_time;
            total.solve_time += stats.solve_time;
            total.refinements += stats.refinements;
            match outcome {
                ExactOutcome::Placed(placement) => {
                    // Count every non-air block, including switches, to match the bound.
                    let used = placement.cells.len();
                    total.winning_seed = stats.winning_seed;
                    best = Some(*placement);
                    if used == 0 {
                        proven_optimal = true;
                        break;
                    }
                    bound = Some(used - 1);
                }
                ExactOutcome::Infeasible => {
                    proven_optimal = best.is_some();
                    break;
                }
                ExactOutcome::Unknown { .. } => break,
            }
        }
        Ok((best, proven_optimal, total))
    }
}
