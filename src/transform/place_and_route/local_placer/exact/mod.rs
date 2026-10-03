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
mod dimacs;
mod dsl;
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
pub use dimacs::DimacsComments;
pub use encode::CellKind;
pub use layout::{ExactLayout, InputPolicy, OutputPolicy};
pub use netlist::{Net, NetDriver, NetId, NorNetlist};
pub use verify::ExactVerificationFailure;

use self::cnf::Lit;
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
    /// Diagnostic only: encode with the hand-written `encode.rs` instead of
    /// grounding `exact_placer.rsdsl`.
    #[doc(hidden)]
    pub legacy_encoder: bool,
    /// After the first verified layout, keep asking for a cheaper one (the
    /// model's `minimize` cost, by default the number of non-air blocks)
    /// until a cheaper one is proven impossible or time runs out. The bound
    /// is an assumption on one incremental solver per worker, and workers
    /// share the best layout found so far.
    pub optimize: bool,
    /// With `optimize`, the last worker raises a proven lower bound with
    /// core-guided search (OLL) instead of lowering the best cost; optimality
    /// is proven once the two meet. With one worker the search is pure OLL.
    /// Off by default: on the cells measured so far the bound rose one unit
    /// per core with each core slower than the last, and the direct proof
    /// finished first (see docs/exact_local_placer.md).
    pub core_guided: bool,
    /// A model file to ground instead of the built-in `exact_placer.rsdsl`.
    /// It must declare the families the placer reads back (`Kind`, `Sig`,
    /// `Powered`, `Conn`, `Points`, `Hard`, `Feeds`, `Contrib`, `Rank`,
    /// `Stage`, `Observe`) and the instance facts and params it fills.
    pub model_file: Option<std::path::PathBuf>,
    /// Extra model params, for example `max_repeaters`; they override the
    /// model's defaults and the params the placer derives from this config.
    pub model_params: BTreeMap<String, rsdsl::IValue>,
    /// Search and verification constants.
    pub tuning: ExactTuning,
}

/// Search and verification constants. The defaults are the values the placer
/// was measured with; they are fields so experiments need no code edits.
#[derive(Debug, Clone)]
pub struct ExactTuning {
    /// Seed spacing between portfolio workers (`seed + worker * stride`).
    pub worker_seed_stride: u32,
    /// Simulator cycle limit for each settle during verification.
    pub sim_max_cycles: usize,
    /// Simulator event limit for each settle during verification.
    pub sim_max_events: usize,
    /// Sets of signal classes the torch lower-bound search may visit; past
    /// this it returns the depth reached, a weaker but valid bound.
    pub torch_bound_max_states: usize,
    /// Largest total objective weight; the cost counter takes one input per
    /// unit of weight.
    pub max_objective_weight: u64,
}

impl Default for ExactTuning {
    fn default() -> Self {
        Self {
            worker_seed_stride: 7919,
            sim_max_cycles: 256,
            sim_max_events: 50_000,
            torch_bound_max_states: 200_000,
            max_objective_weight: 100_000,
        }
    }
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
            legacy_encoder: false,
            optimize: false,
            core_guided: false,
            model_file: None,
            model_params: BTreeMap::new(),
            tuning: ExactTuning::default(),
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
    pub sections: Vec<(String, i32, usize)>,
    /// With `optimize`: cost of the returned layout, whether no cheaper one
    /// exists, and how many times a cheaper layout was found.
    pub cost: Option<i64>,
    pub optimal: bool,
    pub improvements: usize,
    /// The best proven lower bound on the cost (core-guided search).
    pub lower_bound: Option<i64>,
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
    Unknown {
        last_rejection: Option<String>,
    },
}

pub struct ExactLocalPlacer {
    netlist: NorNetlist,
    name: String,
}

enum WorkerResult {
    Placed(Box<ExactPlacement>, u32),
    Infeasible,
    Unknown(Option<String>),
    /// With `optimize`: the shared best layout is proven optimal.
    Optimal,
}

fn worker_seed(config: &ExactPlacerConfig, worker: usize) -> u32 {
    config
        .seed
        .wrapping_add(worker as u32 * config.tuning.worker_seed_stride)
}

/// State the portfolio workers share.
struct WorkerShared<'a> {
    stop: &'a AtomicBool,
    deadline: Option<Instant>,
    refinements: &'a Mutex<usize>,
    lazy: &'a Mutex<acyclic::LazyStats>,
    incumbent: Option<&'a Incumbent>,
}

/// The best verified layout found by any worker (with `optimize`).
struct Incumbent {
    best: Mutex<Option<(i64, Box<ExactPlacement>, u32)>>,
    cost: std::sync::atomic::AtomicI64,
    proven: AtomicBool,
    improvements: std::sync::atomic::AtomicUsize,
    /// Proven lower bound on the cost (`i64::MIN` until one is known).
    lower: std::sync::atomic::AtomicI64,
}

impl Incumbent {
    fn new() -> Self {
        Self {
            best: Mutex::new(None),
            cost: std::sync::atomic::AtomicI64::new(i64::MAX),
            proven: AtomicBool::new(false),
            improvements: std::sync::atomic::AtomicUsize::new(0),
            lower: std::sync::atomic::AtomicI64::new(i64::MIN),
        }
    }

    /// Whether the best layout meets the lower bound (and is thus optimal).
    fn closed(&self) -> bool {
        let best = self.cost.load(Ordering::Relaxed);
        let closed = best != i64::MAX && self.lower.load(Ordering::Relaxed) >= best;
        if closed {
            self.proven.store(true, Ordering::Relaxed);
        }
        closed
    }

    fn offer(&self, cost: i64, placement: ExactPlacement, seed: u32) {
        let mut best = self.best.lock().unwrap();
        if best
            .as_ref()
            .is_some_and(|(current, _, _)| *current <= cost)
        {
            return;
        }
        tracing::info!(cost, seed, "exact placer found a cheaper layout");
        *best = Some((cost, Box::new(placement), seed));
        self.cost.store(cost, Ordering::Relaxed);
        self.improvements.fetch_add(1, Ordering::Relaxed);
    }
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

    pub fn place(
        &self,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<(ExactOutcome, ExactPlacerStats)> {
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
        let incumbent = encoding.objective.as_ref().map(|_| Incumbent::new());
        let shared = WorkerShared {
            stop: &stop,
            deadline,
            refinements: &refinements,
            lazy: &lazy,
            incumbent: incumbent.as_ref(),
        };
        let workers = config.workers.max(1);
        let results = std::thread::scope(|scope| {
            let handles = (0..workers)
                .map(|worker| {
                    let encoding = &encoding;
                    let shared = &shared;
                    scope.spawn(move || {
                        let result = self.run_worker(encoding, config, worker, shared);
                        if !matches!(result, WorkerResult::Unknown(_)) {
                            shared.stop.store(true, Ordering::Relaxed);
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

        if let Some(incumbent) = incumbent {
            stats.optimal = incumbent.proven.load(Ordering::Relaxed);
            stats.improvements = incumbent.improvements.load(Ordering::Relaxed);
            let lower = incumbent.lower.load(Ordering::Relaxed);
            stats.lower_bound = (lower != i64::MIN).then_some(lower);
            if let Some((cost, placement, seed)) = incumbent.best.into_inner().unwrap() {
                stats.cost = Some(cost);
                stats.winning_seed = Some(seed);
                return Ok((ExactOutcome::Placed(placement), stats));
            }
        }
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
                WorkerResult::Optimal => {}
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
        shared: &WorkerShared,
    ) -> WorkerResult {
        if config.core_guided && worker + 1 == config.workers.max(1) && shared.incumbent.is_some() {
            return self.run_core_guided(encoding, config, worker, shared);
        }
        let seed = worker_seed(config, worker);
        let mut solver = SatSolver::new(seed);
        // Diversify the portfolio: sparse-first (phase 0) and stable-mode variants.
        solver.set_option("phase", i32::from(worker % 2 == 1));
        solver.set_option("stabilizeonly", i32::from(worker % 4 >= 2));
        solver.add_cnf(&encoding.cnf);
        let mut signal = StopSignal {
            stop: shared.stop,
            deadline: shared.deadline,
            restart: None,
        };
        let index = acyclic::LazyIndex::new(encoding);
        let mut aux = acyclic::AuxVars::new(encoding);
        let mut last_rejection = None;
        let mut rejections = 0;
        while rejections <= config.max_refinements {
            // With an objective, ask for a layout cheaper than the best so far.
            let mut assumptions = Vec::new();
            if let (Some(bound), Some(incumbent)) = (&encoding.objective, shared.incumbent) {
                if incumbent.closed() {
                    return WorkerResult::Optimal;
                }
                // Restart this solve as soon as another worker improves the bound.
                signal.restart = Some((
                    &incumbent.improvements,
                    incumbent.improvements.load(Ordering::Relaxed),
                ));
                let best = incumbent.cost.load(Ordering::Relaxed);
                if best != i64::MAX {
                    match bound.below(best) {
                        Some(lits) => assumptions = lits,
                        None => {
                            // Nothing can be cheaper than the best layout.
                            incumbent.proven.store(true, Ordering::Relaxed);
                            return WorkerResult::Optimal;
                        }
                    }
                }
            }
            match solver.solve(&assumptions, &signal) {
                SolveResult::Unsat => {
                    if let Some(incumbent) = shared.incumbent {
                        if incumbent.cost.load(Ordering::Relaxed) != i64::MAX {
                            // Blocking clauses only remove simulator-rejected
                            // layouts, so this proves no valid layout is cheaper.
                            incumbent.proven.store(true, Ordering::Relaxed);
                            return WorkerResult::Optimal;
                        }
                    }
                    // Blocking clauses only remove simulator-rejected layouts,
                    // so Unsat after a rejection is not an infeasibility proof.
                    return if last_rejection.is_none() {
                        WorkerResult::Infeasible
                    } else {
                        WorkerResult::Unknown(last_rejection)
                    };
                }
                SolveResult::Interrupted => {
                    if signal.restart.is_some() && !signal.finished() {
                        continue;
                    }
                    return WorkerResult::Unknown(last_rejection);
                }
                SolveResult::Sat => {}
            }
            let mut worker_lazy = acyclic::LazyStats::default();
            let added = acyclic::refine(encoding, &index, &mut solver, &mut aux, &mut worker_lazy);
            if added > 0 {
                let mut total = shared.lazy.lock().unwrap();
                total.loop_formulas += worker_lazy.loop_formulas;
                total.feedback_cuts += worker_lazy.feedback_cuts;
                continue;
            }
            let decoded = verify::decode(encoding, &solver, &self.netlist);
            let world = verify::build_world(config.dim, &decoded.kinds);
            match verify::verify(&self.netlist, &decoded, &world, &config.tuning) {
                Ok(()) => {
                    let cost = encoding
                        .objective
                        .as_ref()
                        .map(|bound| bound.objective.cost(|lit| solver.value(lit)));
                    let placement = self.placement(config, &decoded, world);
                    match (cost, shared.incumbent) {
                        (Some(cost), Some(incumbent)) => incumbent.offer(cost, placement, seed),
                        _ => return WorkerResult::Placed(Box::new(placement), seed),
                    }
                }
                Err(failure) => {
                    tracing::debug!(?failure, "exact placer layout rejected by simulator");
                    if let Ok(directory) = std::env::var("EXACT_DUMP_REJECTIONS") {
                        let diagnosis = verify::diagnose(
                            encoding,
                            &solver,
                            &self.netlist,
                            &decoded,
                            &world,
                            &config.tuning,
                        );
                        verify::dump_rejection(
                            &directory,
                            &self.netlist,
                            config.dim,
                            &decoded,
                            &failure,
                            &diagnosis,
                        );
                    }
                    last_rejection = Some(format!("{} at {:?}", failure.message, failure.position));
                    *shared.refinements.lock().unwrap() += 1;
                    rejections += 1;
                    let blocking = decoded
                        .kind_lits
                        .iter()
                        .map(|&lit| -lit)
                        .collect::<Vec<_>>();
                    solver.add_clause(&blocking);
                }
            }
        }
        WorkerResult::Unknown(last_rejection)
    }

    /// Core-guided lower bounding (OLL; Andres et al. 2012, as in RC2).
    /// Every cost literal is assumed false; an unsatisfiable core says at
    /// least one of its literals must hold, so the bound rises by the core's
    /// smallest weight and the core is relaxed through a totalizer (assuming
    /// "fewer than two of them", then three, ...). The first verified model
    /// that satisfies every assumption is optimal.
    fn run_core_guided(
        &self,
        encoding: &Encoding,
        config: &ExactPlacerConfig,
        worker: usize,
        shared: &WorkerShared,
    ) -> WorkerResult {
        struct Soft {
            assumption: Lit,
            weight: u64,
            /// The totalizer and output index this soft bounds, if any.
            sum: Option<(usize, usize)>,
        }
        struct Sink<'a> {
            solver: &'a mut SatSolver,
            aux: &'a mut acyclic::AuxVars,
        }
        impl rsdsl::formula::ClauseSink for Sink<'_> {
            fn fresh(&mut self) -> Lit {
                self.aux.fresh()
            }
            fn add(&mut self, clause: &[Lit]) {
                self.solver.add_clause(clause);
            }
        }

        let bound = encoding
            .objective
            .as_ref()
            .expect("optimize has an objective");
        let incumbent = shared.incumbent.expect("optimize shares an incumbent");
        let seed = worker_seed(config, worker);
        let mut solver = SatSolver::new(seed);
        solver.add_cnf(&encoding.cnf);
        let signal = StopSignal {
            stop: shared.stop,
            deadline: shared.deadline,
            restart: None,
        };
        let index = acyclic::LazyIndex::new(encoding);
        let mut aux = acyclic::AuxVars::new(encoding);
        let mut softs = bound
            .objective
            .terms
            .iter()
            .map(|&(weight, lit)| Soft {
                assumption: -lit,
                weight,
                sum: None,
            })
            .collect::<Vec<_>>();
        let mut sums: Vec<(Vec<Lit>, u64)> = Vec::new();
        let mut extended = std::collections::HashSet::new();
        let mut lower = bound.objective.offset;
        let mut last_rejection = None;
        let mut rejections = 0;
        while rejections <= config.max_refinements {
            if incumbent.closed() {
                return WorkerResult::Optimal;
            }
            let assumptions = softs
                .iter()
                .filter(|soft| soft.weight > 0)
                .map(|soft| soft.assumption)
                .collect::<Vec<_>>();
            match solver.solve(&assumptions, &signal) {
                SolveResult::Interrupted => return WorkerResult::Unknown(last_rejection),
                SolveResult::Unsat => {
                    let core = (0..softs.len())
                        .filter(|&i| softs[i].weight > 0 && solver.failed(softs[i].assumption))
                        .collect::<Vec<_>>();
                    if core.is_empty() {
                        // Unsatisfiable without assumptions.
                        return if last_rejection.is_none()
                            && incumbent.cost.load(Ordering::Relaxed) == i64::MAX
                        {
                            WorkerResult::Infeasible
                        } else {
                            WorkerResult::Unknown(last_rejection)
                        };
                    }
                    let weight = core.iter().map(|&i| softs[i].weight).min().unwrap();
                    lower += weight as i64;
                    incumbent.lower.fetch_max(lower, Ordering::Relaxed);
                    tracing::info!(lower, core = core.len(), "core-guided lower bound");
                    for &i in &core {
                        softs[i].weight -= weight;
                    }
                    // A relaxed sum whose bound is in the core may now admit
                    // one more violation, at the sum's weight.
                    for &i in &core {
                        if let Some((sum, j)) = softs[i].sum {
                            let (outputs, sum_weight) = &sums[sum];
                            if j + 1 < outputs.len() && extended.insert((sum, j)) {
                                softs.push(Soft {
                                    assumption: -outputs[j + 1],
                                    weight: *sum_weight,
                                    sum: Some((sum, j + 1)),
                                });
                            }
                        }
                    }
                    if core.len() > 1 {
                        let violated = core
                            .iter()
                            .map(|&i| -softs[i].assumption)
                            .collect::<Vec<_>>();
                        let mut sink = Sink {
                            solver: &mut solver,
                            aux: &mut aux,
                        };
                        let outputs =
                            rsdsl::formula::totalizer(&mut sink, &violated, violated.len() - 1);
                        // One violation is already paid for; allow it, charge the second.
                        softs.push(Soft {
                            assumption: -outputs[1],
                            weight,
                            sum: Some((sums.len(), 1)),
                        });
                        sums.push((outputs, weight));
                    }
                }
                SolveResult::Sat => {
                    let mut worker_lazy = acyclic::LazyStats::default();
                    let added =
                        acyclic::refine(encoding, &index, &mut solver, &mut aux, &mut worker_lazy);
                    if added > 0 {
                        let mut total = shared.lazy.lock().unwrap();
                        total.loop_formulas += worker_lazy.loop_formulas;
                        total.feedback_cuts += worker_lazy.feedback_cuts;
                        continue;
                    }
                    let decoded = verify::decode(encoding, &solver, &self.netlist);
                    let world = verify::build_world(config.dim, &decoded.kinds);
                    match verify::verify(&self.netlist, &decoded, &world, &config.tuning) {
                        Ok(()) => {
                            let cost = bound.objective.cost(|lit| solver.value(lit));
                            let placement = self.placement(config, &decoded, world);
                            incumbent.offer(cost, placement, seed);
                            // Every assumption holds, so nothing is cheaper.
                            incumbent.lower.fetch_max(cost, Ordering::Relaxed);
                            if incumbent.closed() {
                                return WorkerResult::Optimal;
                            }
                        }
                        Err(failure) => {
                            last_rejection =
                                Some(format!("{} at {:?}", failure.message, failure.position));
                            *shared.refinements.lock().unwrap() += 1;
                            rejections += 1;
                            let blocking = decoded
                                .kind_lits
                                .iter()
                                .map(|&lit| -lit)
                                .collect::<Vec<_>>();
                            solver.add_clause(&blocking);
                        }
                    }
                }
            }
        }
        WorkerResult::Unknown(last_rejection)
    }

    fn placement(
        &self,
        config: &ExactPlacerConfig,
        decoded: &verify::Decoded,
        world: crate::world::World3D,
    ) -> ExactPlacement {
        let dim = config.dim;
        let cells = decoded
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind != CellKind::Air)
            .map(|(cell, kind)| {
                (
                    Position(cell % dim.0, (cell / dim.0) % dim.1, cell / (dim.0 * dim.1)),
                    *kind,
                )
            })
            .collect::<Vec<_>>();
        let block_count = cells
            .iter()
            .filter(|(_, kind)| !matches!(kind, CellKind::Switch(_)))
            .count();
        let rcell = verify::to_rcell(&self.name, dim, &self.netlist, decoded);
        // Store settled torch states, so pasting the cell starts stable.
        let world = verify::settled_world(&world, &config.tuning).unwrap_or(world);
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
        if !config.legacy_encoder {
            // One incremental solve per worker with the block count as cost.
            let mut attempt = config.clone();
            attempt.optimize = true;
            for (name, weight) in [("block_cost", 1), ("repeater_cost", 0), ("torch_cost", 0)] {
                attempt
                    .model_params
                    .insert(name.to_owned(), rsdsl::IValue::Int(weight));
            }
            let (outcome, stats) = self.place(&attempt)?;
            let optimal = stats.optimal;
            return Ok(match outcome {
                ExactOutcome::Placed(placement) => (Some(*placement), optimal, stats),
                _ => (None, false, stats),
            });
        }
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
