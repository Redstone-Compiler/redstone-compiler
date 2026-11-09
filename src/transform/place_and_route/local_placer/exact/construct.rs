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
use super::{
    CarryTiling, ExactLocalPlacer, ExactOutcome, ExactPlacement, ExactPlacerConfig, ExactTuning,
};
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
    /// Greedily, the ready gate after which the fewest nets are still
    /// needed (live), ties to the smaller fan-in cone: every live net
    /// crosses the window's last slice, so fewer make narrower steps.
    MinLive,
}

/// Gates in construction order, each chain of OR nets built on a support
/// block placed right before the NOR it ends in (`chained_ors`).
pub(super) fn gate_order(netlist: &NorNetlist, order: GateOrder) -> Vec<NetId> {
    let gates = match order {
        GateOrder::MinLive => min_live_order(netlist),
        _ => depth_first_order(netlist, order),
    };
    keep_chains_together(netlist, gates)
}

/// OR nets built on a support block: read by exactly one net, which reads
/// no other OR net, as `NorNetlist::chain_wide_gates` makes them. The chain's
/// first OR goes on a block, each later step brings one more signal to that
/// block, and the NOR that ends the chain is a torch on it, so the OR is
/// never carried (carried on, merged dust would power its lines back and
/// need a repeater to stop that).
pub(super) fn chained_ors(netlist: &NorNetlist) -> BTreeSet<NetId> {
    let is_or = |net: NetId| netlist.nets[net].driver == NetDriver::Or;
    (0..netlist.nets.len())
        .filter(|&net| is_or(net))
        .filter(|&net| {
            let readers = (0..netlist.nets.len())
                .filter(|&reader| netlist.nets[reader].gate_inputs.contains(&net))
                .collect::<Vec<_>>();
            match readers[..] {
                [reader] => {
                    netlist.nets[reader]
                        .gate_inputs
                        .iter()
                        .filter(|&&input| is_or(input))
                        .count()
                        == 1
                }
                _ => false,
            }
        })
        .collect()
}

/// Moves each chain of `chained_ors` to just before the net it ends in, so
/// its stages are consecutive steps and the support block stays reachable.
/// Their other inputs come earlier already, so the order stays topological.
fn keep_chains_together(netlist: &NorNetlist, order: Vec<NetId>) -> Vec<NetId> {
    let chained = chained_ors(netlist);
    if chained.is_empty() {
        return order;
    }
    fn chain(netlist: &NorNetlist, chained: &BTreeSet<NetId>, net: NetId, out: &mut Vec<NetId>) {
        for &input in &netlist.nets[net].gate_inputs {
            if chained.contains(&input) && !out.contains(&input) {
                chain(netlist, chained, input, out);
                out.push(input);
            }
        }
    }
    let mut out = Vec::new();
    for net in order {
        if chained.contains(&net) {
            continue;
        }
        chain(netlist, &chained, net, &mut out);
        out.push(net);
    }
    out
}

fn depth_first_order(netlist: &NorNetlist, order: GateOrder) -> Vec<NetId> {
    fn visit(netlist: &NorNetlist, net: NetId, done: &mut [bool], order: &mut Vec<NetId>) {
        if done[net] {
            return;
        }
        done[net] = true;
        for &input in &netlist.nets[net].gate_inputs {
            visit(netlist, input, done, order);
        }
        // Torches and OR nets are each a step.
        if !matches!(netlist.nets[net].driver, NetDriver::Input(_)) {
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

/// `GateOrder::MinLive`.
fn min_live_order(netlist: &NorNetlist) -> Vec<NetId> {
    let is_input = |net: NetId| matches!(netlist.nets[net].driver, NetDriver::Input(_));
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
    let mut remaining = (0..netlist.nets.len())
        .filter(|&net| !is_input(net))
        .collect::<BTreeSet<_>>();
    let mut placed = BTreeSet::new();
    let mut order = Vec::new();
    while !remaining.is_empty() {
        let live_after = |gate: NetId| {
            let mut placed = placed.clone();
            placed.insert(gate);
            placed.extend(netlist.nets[gate].gate_inputs.iter().copied());
            placed
                .iter()
                .filter(|&&net| {
                    remaining.iter().any(|&later| {
                        later != gate && netlist.nets[later].gate_inputs.contains(&net)
                    })
                })
                .count()
        };
        let Some(next) = remaining
            .iter()
            .copied()
            .filter(|&gate| {
                netlist.nets[gate]
                    .gate_inputs
                    .iter()
                    .all(|&input| is_input(input) || placed.contains(&input))
            })
            .min_by_key(|&gate| (live_after(gate), cone(gate), gate))
        else {
            break;
        };
        placed.insert(next);
        placed.extend(netlist.nets[next].gate_inputs.iter().copied());
        remaining.remove(&next);
        order.push(next);
    }
    order
}

/// Gates per construction step, in order: one gate each, except for a
/// `CarryTiling`'s carry output.
///
/// The carry switch and the carry-out block share a column, so the carry
/// path leaves that column and must come back to it. Solving the whole way
/// in one window failed (`docs/carry_tiles.md`). Instead the gate that reads
/// the carry input gets its own step, with the switch on the window's last
/// slice; construction then keeps a corridor clear from the carry-out cell
/// along +Y (see `construct_step`); the rest of the carry path follows gate
/// by gate (an OR and the NOR the netlist realizes it with as one step); and
/// the carry output, an OR, is one last step that ORs the carry onto a block
/// and runs it back down the corridor. The carry comes as early as it can, right after the
/// gates it needs, so the corridor is short and few other nets cross the
/// seams beside it.
pub(super) fn construction_steps(
    netlist: &NorNetlist,
    order: GateOrder,
    carry: Option<&CarryTiling>,
) -> Vec<Vec<NetId>> {
    let order = gate_order(netlist, order);
    let path = carry.map_or_else(BTreeSet::new, |carry| carry_path(netlist, carry));
    let (Some(carry), Some(input)) = (
        carry,
        carry.and_then(|carry| netlist.input_net(&carry.input)),
    ) else {
        return order.into_iter().map(|gate| vec![gate]).collect();
    };
    let Some(&(_, output)) = netlist
        .outputs
        .iter()
        .find(|(name, _)| *name == carry.output)
    else {
        return order.into_iter().map(|gate| vec![gate]).collect();
    };
    if path.is_empty() {
        return order.into_iter().map(|gate| vec![gate]).collect();
    }
    // An OR is a NOT over the NOR of its terms. A block carries the OR
    // itself (and a repeater reading it passes it on), so the pair is one
    // step and neither gate needs a torch.
    let or_inner = |outer: NetId| match netlist.nets[outer].gate_inputs[..] {
        [inner]
            if path.contains(&inner)
                && netlist
                    .gates()
                    .filter(|&gate| netlist.nets[gate].gate_inputs.contains(&inner))
                    .count()
                    == 1 =>
        {
            Some(inner)
        }
        _ => None,
    };
    let mut out = vec![output];
    if let Some(inner) = or_inner(output) {
        out.insert(0, inner);
    }
    let path_order = order
        .iter()
        .copied()
        .filter(|gate| path.contains(gate) && !out.contains(gate))
        .collect::<Vec<_>>();
    let (entry, middle): (Vec<NetId>, Vec<NetId>) = path_order
        .into_iter()
        .partition(|&gate| netlist.nets[gate].gate_inputs.contains(&input));
    // Gates the carry path reads that do not depend on the carry input.
    let mut needed = BTreeSet::new();
    let mut stack = path.iter().copied().collect::<Vec<_>>();
    while let Some(net) = stack.pop() {
        for &input in &netlist.nets[net].gate_inputs {
            if netlist.nets[input].driver == NetDriver::Gate
                && !path.contains(&input)
                && needed.insert(input)
            {
                stack.push(input);
            }
        }
    }
    let mut steps = order
        .iter()
        .filter(|gate| needed.contains(gate))
        .map(|&gate| vec![gate])
        .collect::<Vec<_>>();
    if !entry.is_empty() {
        steps.push(entry);
    }
    let inners = middle
        .iter()
        .filter_map(|&gate| or_inner(gate))
        .collect::<BTreeSet<_>>();
    for &gate in &middle {
        if inners.contains(&gate) {
            continue;
        }
        steps.push(or_inner(gate).into_iter().chain([gate]).collect());
    }
    steps.push(out);
    steps.extend(
        order
            .iter()
            .filter(|gate| !path.contains(gate) && !needed.contains(gate))
            .map(|&gate| vec![gate]),
    );
    steps
}

/// Gates that depend on the carry input and feed the carry output.
fn carry_path(netlist: &NorNetlist, carry: &CarryTiling) -> BTreeSet<NetId> {
    let (Some(input), Some(&(_, output))) = (
        netlist.input_net(&carry.input),
        netlist
            .outputs
            .iter()
            .find(|(name, _)| *name == carry.output),
    ) else {
        return BTreeSet::new();
    };
    let depends = |net: NetId| {
        let mut stack = vec![net];
        let mut seen = BTreeSet::new();
        while let Some(net) = stack.pop() {
            if net == input {
                return true;
            }
            if seen.insert(net) {
                stack.extend(netlist.nets[net].gate_inputs.iter().copied());
            }
        }
        false
    };
    let mut cone = BTreeSet::new();
    let mut stack = vec![output];
    while let Some(net) = stack.pop() {
        if cone.insert(net) {
            stack.extend(netlist.nets[net].gate_inputs.iter().copied());
        }
    }
    cone.into_iter()
        .filter(|&net| netlist.nets[net].driver == NetDriver::Gate && depends(net))
        .collect()
}

#[derive(Debug, Clone)]
pub struct ConstructionConfig {
    /// Box width along X (with a `carry`, the two ghost slices included).
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
    /// Solve this many consecutive construction steps jointly before
    /// committing a seam. One retains the original gate-at-a-time path.
    pub gate_batch: usize,
    /// Outputs without a face policy stop crossing seams once no later gate
    /// reads them; they only need to stay observable somewhere in the box.
    /// Off, every output is carried to the last slice.
    pub early_outputs: bool,
    /// Fix the signals of frozen cells to the previous step's solution, so a
    /// step no longer re-justifies (ranks, stages, contributing sources) the
    /// whole layout built so far; those variables were over 90% of a late
    /// step's encoding. The simulator still checks the whole layout.
    pub given_frozen_signals: bool,
    /// Restrict each window to functions of already available/current nets,
    /// their complements, and signals present in the previous layout. This
    /// deliberately narrows synthesis; it is not solution-preserving pruning.
    pub step_vocabulary: bool,
    /// Permit intermediate ORs of each current gate's fan-in on wires, not
    /// only the complete NOR function. Bounded to fan-in <= 8.
    pub partial_or_vocabulary: bool,
    /// Continuing nets must be observed on a torch or outward repeater,
    /// rather than an arbitrary powered cell at the seam. Chained ORs keep
    /// their required solid support and are exempt.
    pub driving_seams: bool,
    /// Carry one representative per Boolean function/complement pair. The
    /// next window may regenerate the other polarity with a torch. Finished
    /// output observations and chained-OR support contracts remain intact.
    pub compress_complements: bool,
    /// Extend existing outward repeater ports at the same X/Z coordinate:
    /// a strongly powered block provides a tap, then repeaters carry forward.
    /// Requires zero overlap; other cells and new logic remain SAT choices.
    pub scaffold_seams: bool,
    /// Spaced, reusable port grid (0 keeps free seam sites). Primary inputs
    /// retain fixed slots; new nets use a nearby unused slot. With scaffolds,
    /// this leaves room to tap each rail instead of packing inaccessible ports.
    pub seam_pitch: usize,
    pub seam_primary_stride: usize,
    /// Restricted NOR-template initial solution, followed by exact compaction.
    pub gate_scaffold: bool,
    /// After each step, spend up to this long minimizing the layout's block
    /// count (the model's cost) with the same frozen cells, so dead wires do
    /// not pile up. `None` keeps the first layout found. Five seconds halved
    /// the full adder's constructed blocks (about 290 to 127-164).
    pub step_optimize: Option<Duration>,
    /// Delete redundant blocks in the accepted window before relocation
    /// optimization. Uses a fixed-layout SAT optimization with pinned ports.
    pub step_delete: Option<Duration>,
    /// Model params for every step (see `ExactPlacerConfig::model_params`).
    pub model_params: BTreeMap<String, rsdsl::IValue>,
    /// Diagnostic only: see `ExactPlacerConfig::no_fold`.
    #[doc(hidden)]
    pub no_fold: bool,
    /// Build a tile that repeats along X (see `ExactPlacerConfig::carry`).
    pub carry: Option<CarryTiling>,
    /// Time limit of the step that places the carry-out block, which has to
    /// fit the whole carry logic around one cell (`step_time_limit` if unset).
    pub carry_step_time_limit: Option<Duration>,
    /// Write the layout after every step of the successful attempt to this
    /// directory, for watching construction in the viewer (`progress.rs`).
    pub progress: Option<std::path::PathBuf>,
    /// Save the latest accepted prefix, including when a later step fails.
    pub checkpoint: Option<std::path::PathBuf>,
    /// Resume a matching netlist/order from an accepted prefix. Backtracking
    /// does not cross the loaded prefix; every subsequent step is verified.
    pub resume: Option<std::path::PathBuf>,
    /// After each step, ask for its nets sooner (`ExactPlacerConfig::timing`):
    /// each observed net at most its lower bound (the latest of its inputs'
    /// arrivals, plus a tick unless it is an OR a block can carry) plus a
    /// slack of 0, 1, 2, 4, ... ticks, each solve this long, until one
    /// succeeds or no net could gain; `step_optimize` then keeps the delays
    /// reached. `None` builds without regard to timing.
    pub step_timing: Option<Duration>,
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
            gate_batch: 1,
            early_outputs: true,
            given_frozen_signals: true,
            step_vocabulary: false,
            partial_or_vocabulary: false,
            driving_seams: false,
            compress_complements: false,
            scaffold_seams: false,
            seam_pitch: 0,
            seam_primary_stride: 1,
            gate_scaffold: false,
            step_optimize: Some(Duration::from_secs(5)),
            step_delete: None,
            model_params: BTreeMap::new(),
            no_fold: false,
            carry: None,
            carry_step_time_limit: None,
            progress: None,
            checkpoint: None,
            resume: None,
            step_timing: None,
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
    /// The gates this step placed, for progress frames.
    label: String,
    length: usize,
    cells: BTreeMap<Position, CellKind>,
    /// Signal function of every non-air cell in the latest solution.
    signals: BTreeMap<Position, u64>,
    placed_inputs: BTreeMap<String, (Position, Direction)>,
    available: BTreeSet<NetId>,
    /// The support block of each chained OR net placed (`chained_ors`).
    supports: BTreeMap<NetId, Position>,
    result: Option<(DimSize, ExactPlacement)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ConstructionCheckpoint {
    netlist: String,
    outputs: Vec<(String, NetId)>,
    order: Vec<Vec<NetId>>,
    step: usize,
    rcell: String,
    signals: Vec<([usize; 3], u64)>,
    available: BTreeSet<NetId>,
    supports: Vec<(NetId, [usize; 3])>,
}

impl ConstructionCheckpoint {
    fn save(
        placer: &ExactLocalPlacer,
        order: &[Vec<NetId>],
        step: usize,
        state: &StepState,
        path: &std::path::Path,
    ) -> eyre::Result<()> {
        let (_, placement) = state
            .result
            .as_ref()
            .expect("accepted step has a placement");
        let checkpoint = Self {
            netlist: placer.netlist.to_text(),
            outputs: placer.netlist.outputs.clone(),
            order: order.to_vec(),
            step,
            rcell: placement.rcell.to_string(),
            signals: state
                .signals
                .iter()
                .map(|(p, &f)| ([p.0, p.1, p.2], f))
                .collect(),
            available: state.available.clone(),
            supports: state
                .supports
                .iter()
                .map(|(&n, p)| (n, [p.0, p.1, p.2]))
                .collect(),
        };
        let temporary = path.with_extension("json.tmp");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&temporary, serde_json::to_vec_pretty(&checkpoint)?)?;
        std::fs::rename(temporary, path)?;
        Ok(())
    }

    fn load(
        placer: &ExactLocalPlacer,
        order: &[Vec<NetId>],
        config: &ConstructionConfig,
        path: &std::path::Path,
    ) -> eyre::Result<(usize, StepState)> {
        let checkpoint: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        eyre::ensure!(
            checkpoint.netlist == placer.netlist.to_text()
                && checkpoint.outputs == placer.netlist.outputs
                && checkpoint.order == order,
            "checkpoint netlist/order mismatch"
        );
        eyre::ensure!(
            checkpoint.step > 0 && checkpoint.step < order.len(),
            "checkpoint must be a partial construction"
        );
        let document: crate::physical_cell::PhysicalCellDocument = checkpoint.rcell.parse()?;
        let layout = ExactLayout::from_rcell(&document)?;
        eyre::ensure!(
            config.carry.is_none(),
            "resuming carry tiles is not supported"
        );
        for (name, policy) in &config.input_policies {
            if *policy == InputPolicy::MinYFace {
                eyre::ensure!(
                    layout.inputs.iter().any(|(n, p, _)| n == name && p.1 == 0),
                    "checkpoint does not satisfy input face policy for {name}"
                );
            }
        }
        eyre::ensure!(
            layout.dim.0 <= config.width && layout.dim.2 <= config.height,
            "checkpoint cross-section cannot shrink"
        );
        let mut expected = order[..checkpoint.step]
            .iter()
            .flatten()
            .copied()
            .chain(order[..checkpoint.step].iter().flatten().flat_map(|&net| {
                placer.netlist.nets[net]
                    .gate_inputs
                    .iter()
                    .copied()
                    .filter(|&n| matches!(placer.netlist.nets[n].driver, NetDriver::Input(_)))
            }))
            .collect::<BTreeSet<_>>();
        for (name, _, _) in &layout.inputs {
            expected.insert(
                placer
                    .netlist
                    .input_net(name)
                    .ok_or_else(|| eyre::eyre!("checkpoint contains unknown input {name}"))?,
            );
        }
        eyre::ensure!(
            expected == checkpoint.available,
            "checkpoint available nets mismatch"
        );
        let signals = checkpoint
            .signals
            .into_iter()
            .map(|([x, y, z], f)| (Position(x, y, z), f))
            .collect::<BTreeMap<_, _>>();
        eyre::ensure!(
            signals.len() == layout.cells.len()
                && signals.keys().all(|p| layout.cells.contains_key(p)),
            "checkpoint signal cells mismatch"
        );
        let supports = checkpoint
            .supports
            .into_iter()
            .map(|(n, [x, y, z])| (n, Position(x, y, z)))
            .collect::<BTreeMap<_, _>>();
        eyre::ensure!(
            supports.iter().all(|(&n, p)| n < placer.netlist.nets.len()
                && layout.cells.get(p) == Some(&CellKind::Solid)),
            "checkpoint support mismatch"
        );
        Ok((
            checkpoint.step,
            StepState {
                label: "resumed prefix".into(),
                length: layout.dim.1,
                cells: layout.cells,
                signals,
                placed_inputs: layout
                    .inputs
                    .into_iter()
                    .map(|(n, p, d)| (n, (p, d)))
                    .collect(),
                available: checkpoint.available,
                supports,
                result: None,
            },
        ))
    }
}

/// Bounds every output of `exact` by its delay in `placement`, with `Stage`
/// sized to the placement's longest path (see `CompactionConfig::timing`).
fn keep_delays(exact: &mut ExactPlacerConfig, placement: &ExactPlacement) {
    let cells = placement.cells.iter().copied().collect();
    let longest = super::timing::analyze(exact.dim, &cells, &[])
        .ok()
        .and_then(|timing| timing.arrival.values().copied().max())
        .unwrap_or(exact.stage_levels);
    exact.timing = true;
    exact.stage_levels = longest + 2;
    exact.output_delays = placement.delays.clone();
}

impl ExactLocalPlacer {
    /// Builds the layout gate by gate. When an attempt fails or exceeds
    /// `restart_after`, it starts over with the next seed, up to
    /// `max_restarts` times.
    pub fn construct(
        &self,
        config: &ConstructionConfig,
    ) -> eyre::Result<(ExactLayout, ExactPlacement, ConstructionReport)> {
        eyre::ensure!(
            self.netlist.state.is_empty(),
            "construction places one gate per step and cannot cut a latch's loop yet"
        );
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
        eyre::ensure!(config.gate_batch > 0, "gate batch must be positive");
        eyre::ensure!(
            config.carry.is_none() || config.gate_batch == 1,
            "batched carry-tile construction is not supported"
        );
        let order = construction_steps(netlist, config.gate_order, config.carry.as_ref());
        let order = order
            .chunks(config.gate_batch)
            .map(|steps| steps.iter().flatten().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        if order.is_empty() {
            bail!("netlist has no gates to construct");
        }
        let mut report = ConstructionReport::default();
        // states[i] is the layout before step i.
        let mut states = vec![StepState {
            label: String::new(),
            length: 0,
            cells: BTreeMap::new(),
            signals: BTreeMap::new(),
            placed_inputs: BTreeMap::new(),
            available: BTreeSet::new(),
            supports: BTreeMap::new(),
            result: None,
        }];
        let mut blocked = vec![Vec::new(); order.len()];
        let mut windows = vec![Vec::new(); order.len()];
        let mut step = 0;
        if let Some(path) = &config.resume {
            let (loaded_step, state) = ConstructionCheckpoint::load(self, &order, config, path)?;
            step = loaded_step;
            states.resize(step + 1, state.clone());
            states[step] = state;
            tracing::info!(step, path = %path.display(), "construction resumes accepted prefix");
        }
        let first_step = step;
        while step < order.len() {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                bail!(
                    "construction attempt with seed {} ran out of time at gate {}",
                    config.seed,
                    self.step_name(&order[step])
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
                    if let Some(path) = &config.checkpoint {
                        ConstructionCheckpoint::save(self, &order, step + 1, &state, path)?;
                    }
                    report.steps.push(entry);
                    states.truncate(step + 1);
                    states.push(state);
                    windows[step] = window_cells;
                    step += 1;
                }
                None => {
                    if step == first_step || report.backtracks >= config.max_backtracks {
                        bail!(
                            "construction could not place gate {} within {} slices",
                            self.step_name(&order[step]),
                            config.max_window
                        );
                    }
                    report.backtracks += 1;
                    tracing::info!(
                        gate = self.step_name(&order[step]),
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
        for state in &states[1..] {
            if let Some((_, placement)) = &state.result {
                super::progress::record_frame(
                    config.progress.as_deref(),
                    &self.name,
                    &format!("construct {}", state.label),
                    placement,
                );
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

    /// The fewest ticks each net a step observes could take: nets placed
    /// before arrive as they did on the previous step's last slice, inputs
    /// placed now at once, an OR net with its latest input, and each new
    /// gate a tick after its latest input, or with its inner gate's inputs if
    /// it is an OR over a NOR of the same step (a block carries it). Keyed by
    /// net name and by output name.
    fn step_lower_bounds(&self, gates: &[NetId], state: &StepState) -> BTreeMap<String, usize> {
        let netlist = &self.netlist;
        let previous = state
            .result
            .as_ref()
            .map(|(_, placement)| &placement.delays);
        let mut lower = BTreeMap::<NetId, usize>::new();
        for &net in &state.available {
            let ticks = previous
                .and_then(|delays| delays.get(&netlist.nets[net].name))
                .copied()
                .unwrap_or(0);
            lower.insert(net, ticks);
        }
        for &gate in gates {
            let arrival =
                |net: NetId, lower: &BTreeMap<NetId, usize>| lower.get(&net).copied().unwrap_or(0);
            let inputs = &netlist.nets[gate].gate_inputs;
            let latest = inputs
                .iter()
                .map(|&net| arrival(net, &lower))
                .max()
                .unwrap_or(0);
            if netlist.nets[gate].driver == NetDriver::Or {
                // Dust and blocks: no tick.
                lower.insert(gate, latest);
                continue;
            }
            let nor = latest + 1;
            let or = match inputs[..] {
                [inner] if gates.contains(&inner) => netlist.nets[inner]
                    .gate_inputs
                    .iter()
                    .map(|&net| arrival(net, &lower))
                    .max(),
                _ => None,
            };
            lower.insert(gate, or.map_or(nor, |or| or.min(nor)));
        }
        let mut named = lower
            .iter()
            .map(|(&net, &ticks)| (netlist.nets[net].name.clone(), ticks))
            .collect::<BTreeMap<_, _>>();
        for (name, net) in &netlist.outputs {
            if let Some(&ticks) = lower.get(net) {
                named.insert(name.clone(), ticks);
            }
        }
        named
    }

    /// Re-solves a step with its observed nets sooner (see
    /// `ConstructionConfig::step_timing`); returns the step's layout as found
    /// when no slack below its own delays succeeds.
    fn shorten_step(
        &self,
        exact: &ExactPlacerConfig,
        placement: ExactPlacement,
        lower: &BTreeMap<String, usize>,
        budget: Duration,
        deadline: Option<Instant>,
    ) -> ExactPlacement {
        let found = placement.delays.clone();
        let mut slack = 0;
        loop {
            let budgets = found
                .iter()
                .map(|(name, &ticks)| {
                    let bound = lower
                        .get(name)
                        .map_or(ticks, |&bound| ticks.min(bound + slack));
                    (name.clone(), bound)
                })
                .collect::<BTreeMap<_, _>>();
            if budgets == found {
                return placement;
            }
            let remaining =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            if remaining.is_some_and(|remaining| remaining.is_zero()) {
                return placement;
            }
            let mut sooner = exact.clone();
            keep_delays(&mut sooner, &placement);
            sooner.output_delays = budgets;
            sooner.time_limit = Some(remaining.map_or(budget, |remaining| remaining.min(budget)));
            if let Ok((ExactOutcome::Placed(better), _)) = self.place(&sooner) {
                tracing::info!(
                    slack,
                    before = ?found,
                    after = ?better.delays,
                    "construction step shortened"
                );
                return *better;
            }
            slack = if slack == 0 { 1 } else { slack * 2 };
        }
    }

    /// Diagnostic hook: with `EXACT_DIAGNOSE_INFEASIBLE`, a step proven
    /// infeasible is solved again with one requirement dropped at a time,
    /// and each outcome is logged, so the requirement that makes it
    /// impossible shows. `EXACT_DIAGNOSE_UNKNOWN` does the same for the first
    /// step that times out, which shows the requirement that makes it hard.
    /// Each variant gets `EXACT_DIAGNOSE_VARIANT_SECONDS` (20).
    fn diagnose_infeasible_step(&self, exact: &ExactPlacerConfig, gate: &str, window: usize) {
        for (name, sites) in &exact.input_sites {
            if let [(position, attach)] = sites[..] {
                let around = [
                    Direction::East,
                    Direction::West,
                    Direction::North,
                    Direction::South,
                    Direction::Top,
                    Direction::Bottom,
                ]
                .map(|direction| {
                    let neighbor = position.walk(direction).filter(|p| exact.dim.bound_on(*p));
                    (
                        direction,
                        neighbor.map(|p| exact.fixed_cells.get(&p).copied()),
                    )
                });
                tracing::info!(
                    gate,
                    window,
                    input = name.as_str(),
                    ?position,
                    ?attach,
                    dim = ?exact.dim,
                    ?around,
                    "infeasible step input"
                );
            }
        }
        let mut variants = Vec::<(String, ExactPlacerConfig)>::new();
        let mut given = exact.clone();
        given.given_signals.clear();
        variants.push(("no given signals".to_owned(), given));
        if let Some(observations) = &exact.observations {
            for (index, (net, _)) in observations.iter().enumerate() {
                let mut fewer = exact.clone();
                fewer.observations.as_mut().unwrap().remove(index);
                let name = self.netlist.nets[*net].name.clone();
                variants.push((format!("without observing {name}"), fewer));
            }
        }
        for (name, sites) in &exact.input_sites {
            if sites.len() > 1 {
                continue;
            }
            let mut free = exact.clone();
            free.input_sites.remove(name);
            variants.push((format!("input {name} anywhere"), free));
        }
        let mut unfrozen = exact.clone();
        unfrozen
            .fixed_cells
            .retain(|_, kind| !matches!(kind, CellKind::Air));
        variants.push(("frozen air free".to_owned(), unfrozen));
        let seconds = std::env::var("EXACT_DIAGNOSE_VARIANT_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);
        for (label, mut variant) in variants {
            variant.time_limit = Some(Duration::from_secs(seconds));
            let started = Instant::now();
            let outcome = self.place(&variant).map(|(outcome, _)| match outcome {
                ExactOutcome::Placed(_) => "placed",
                ExactOutcome::Infeasible => "infeasible",
                ExactOutcome::Unknown { .. } => "unknown",
            });
            tracing::info!(
                gate,
                window,
                variant = label.as_str(),
                ?outcome,
                seconds = started.elapsed().as_secs_f64(),
                "step diagnosed"
            );
        }
    }

    /// The gates of a construction step, as one name.
    fn step_name(&self, gates: &[NetId]) -> String {
        gates
            .iter()
            .map(|&gate| self.netlist.nets[gate].name.as_str())
            .collect::<Vec<_>>()
            .join("+")
    }

    #[allow(clippy::type_complexity)]
    fn construct_step(
        &self,
        order: &[Vec<NetId>],
        step: usize,
        state: &StepState,
        blocked: &[Vec<(Position, CellKind)>],
        config: &ConstructionConfig,
        deadline: Option<Instant>,
    ) -> eyre::Result<Option<(StepState, Vec<(Position, CellKind)>, (String, usize, f64))>> {
        let netlist = &self.netlist;
        let gates = &order[step];
        let gate_name = self.step_name(gates);
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
                    .flatten()
                    .any(|&gate| netlist.nets[gate].gate_inputs.contains(&net))
        };
        let mut needed_inputs = gates
            .iter()
            .flat_map(|&gate| netlist.nets[gate].gate_inputs.iter())
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
        // With polarity-compressed frontiers a non-output inversion adds no
        // information: its input's representative can regenerate either
        // polarity in a later exact window. Avoid a whole transport window.
        // Leave backtracking's window history unchanged unless it is off.
        if config.compress_complements
            && config.max_backtracks == 0
            && !is_last
            && gates.len() == 1
            && needed_inputs.is_empty()
            && state.result.is_some()
            && netlist.nets[gates[0]].driver == NetDriver::Gate
            && netlist.nets[gates[0]].gate_inputs.len() == 1
            && !netlist.outputs.iter().any(|(_, net)| *net == gates[0])
        {
            let mut next = state.clone();
            next.label = format!("{gate_name} virtual polarity");
            next.available.extend(gates.iter().copied());
            tracing::info!(gate = gate_name, "construction keeps a virtual polarity");
            return Ok(Some((next, Vec::new(), (gate_name, 0, 0.0))));
        }
        let length = state.length;
        // A NOR built in stages (`chained_ors`): the step that adds a signal
        // to a chain keeps its support block fixed and the block's slice
        // open, so the new signal can reach it.
        let chained = chained_ors(netlist);
        let support = gates
            .iter()
            .flat_map(|&gate| &netlist.nets[gate].gate_inputs)
            .filter(|input| chained.contains(input))
            .find_map(|input| state.supports.get(input))
            .copied();
        // The carry column, once the carry switch is placed, and whether
        // this step places the carry output. Until then the carry-out cell
        // is an unpowered block and a corridor from it along +Y stays clear,
        // with blocks under it, so the carry output can come back to the
        // column on a line of repeaters.
        let carry_output = config.carry.as_ref().and_then(|carry| {
            netlist
                .outputs
                .iter()
                .find(|(name, _)| *name == carry.output)
                .map(|&(_, net)| net)
        });
        let carry_out = carry_output.is_some_and(|net| gates.contains(&net));
        let corridor = config
            .carry
            .as_ref()
            .and_then(|carry| state.placed_inputs.get(&carry.input))
            .filter(|_| carry_output.is_some_and(|net| !state.available.contains(&net)))
            .map(|(position, _)| (position.1, position.2));
        let min_overlap = config.overlap;
        // Cheapest re-solve first: fewest slices (overlap + window), then the
        // smaller overlap.
        let mut attempts = (min_overlap..=config.max_overlap.max(min_overlap))
            .flat_map(|overlap| {
                (config.window..=config.max_window).map(move |window| (overlap, window))
            })
            .collect::<Vec<_>>();
        attempts.sort_by_key(|&(overlap, window)| (overlap + window, overlap));
        for (overlap, window) in attempts {
            let frozen = length.saturating_sub(overlap);
            if overlap > min_overlap && frozen == length.saturating_sub(overlap - 1) {
                // Nothing more to unfreeze (the layout is shorter).
                continue;
            }
            let frozen = support.map_or(frozen, |block| frozen.min(block.1));
            let dim = DimSize(config.width, length + window, config.height);
            let mut exact = ExactPlacerConfig::new(dim);
            exact.workers = config.workers;
            exact.seed = config.seed;
            exact.rank_levels = config.rank_levels;
            exact.stage_levels = config.stage_levels;
            exact.no_fold = config.no_fold;
            exact.carry = config.carry.clone();
            // Stop at the attempt's deadline too, so a restart starts on time.
            let remaining =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            if remaining.is_some_and(|remaining| remaining.is_zero()) {
                return Ok(None);
            }
            let step_time_limit = if carry_out {
                config
                    .carry_step_time_limit
                    .unwrap_or(config.step_time_limit)
            } else {
                config.step_time_limit
            };
            exact.time_limit =
                Some(remaining.map_or(step_time_limit, |remaining| remaining.min(step_time_limit)));
            exact.max_refinements = config.max_refinements;
            exact.tuning = config.tuning.clone();
            exact.model_params = config.model_params.clone();
            exact.blocked = blocked.to_vec();
            if config.partial_or_vocabulary {
                let values = netlist.net_values();
                exact
                    .extra_signal_functions
                    .extend(state.signals.values().copied());
                for &gate in gates {
                    let inputs = &netlist.nets[gate].gate_inputs;
                    eyre::ensure!(
                        inputs.len() <= 8,
                        "partial OR vocabulary supports fan-in <= 8"
                    );
                    let functions = inputs
                        .iter()
                        .map(|&net| {
                            values[net]
                                .iter()
                                .enumerate()
                                .fold(0u64, |mask, (i, &on)| mask | (u64::from(on) << i))
                        })
                        .collect::<Vec<_>>();
                    for subset in 1..(1usize << inputs.len()) {
                        let function = functions
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| subset & (1 << i) != 0)
                            .fold(0, |a, (_, f)| a | f);
                        exact.extra_signal_functions.insert(function);
                    }
                }
            }
            if config.step_vocabulary {
                let values = netlist.net_values();
                let full = u64::MAX >> (64 - (1usize << netlist.case_bits()));
                let mut functions = state.signals.values().copied().collect::<BTreeSet<_>>();
                functions.extend(exact.extra_signal_functions.iter().copied());
                for &net in state
                    .available
                    .iter()
                    .chain(needed_inputs.iter())
                    .chain(gates.iter())
                {
                    let function = values[net]
                        .iter()
                        .enumerate()
                        .fold(0u64, |mask, (i, &on)| mask | (u64::from(on) << i));
                    functions.insert(function);
                }
                functions.extend(functions.clone().into_iter().map(|f| !f & full));
                exact.signal_functions = Some(functions);
            }
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
            if let Some(block) = support {
                exact.fixed_cells.insert(block, CellKind::Solid);
                exact.given_signals.remove(&block);
            }
            if let Some((y0, z0)) = corridor {
                let x = dim.0 - 1;
                for y in y0..dim.1 {
                    let cell = Position(x, y, z0);
                    exact.given_signals.remove(&cell);
                    if carry_out {
                        exact.fixed_cells.remove(&cell);
                    } else if y == y0 {
                        // The carry-out block, kept unpowered: nothing may
                        // power it but the carry output.
                        exact.fixed_cells.insert(cell, CellKind::Solid);
                        exact.given_signals.insert(cell, 0);
                    } else {
                        exact.fixed_cells.insert(cell, CellKind::Air);
                        if z0 > 0 {
                            exact
                                .fixed_cells
                                .insert(Position(x, y, z0 - 1), CellKind::Solid);
                        }
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
                    let sites = match &config.carry {
                        // The carry switch: the ghost slice's last column, so
                        // the corridor starts beyond the window, and above the
                        // floor, so the corridor has blocks under it.
                        Some(carry) if carry.input == name => (1..dim.2)
                            .map(|z| (Position(0, dim.1 - 1, z), Direction::East))
                            .collect(),
                        Some(_) => sites_in(dim, frozen..dim.1, policy)
                            .into_iter()
                            .filter(|(position, _)| position.0 >= CarryTiling::GHOST)
                            .collect(),
                        None => sites_in(dim, frozen..dim.1, policy),
                    };
                    // A cell holds one switch: leave out the ones placed
                    // before (a kept switch in the overlap is a fixed cell
                    // the new input must not claim).
                    let sites = sites
                        .into_iter()
                        .filter(|(position, _)| {
                            !state
                                .placed_inputs
                                .values()
                                .any(|(placed, _)| placed == position)
                        })
                        .collect();
                    exact.input_sites.insert(name.clone(), sites);
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
                    .chain(gates.iter())
                    .copied()
                    .collect::<BTreeSet<_>>();
                let mut observations = placed
                    .iter()
                    .copied()
                    .filter(|&net| consumers(net, step + 1))
                    .map(|net| (net, last_slice.clone()))
                    .collect::<Vec<_>>();
                if config.compress_complements {
                    let before = observations.len();
                    let values = netlist.net_values();
                    let full = u64::MAX >> (64 - (1usize << netlist.case_bits()));
                    let function = |net: NetId| {
                        values[net]
                            .iter()
                            .enumerate()
                            .fold(0u64, |mask, (i, &on)| mask | (u64::from(on) << i))
                    };
                    let mut representatives = BTreeMap::<u64, (NetId, Vec<Position>)>::new();
                    let mut supports = Vec::new();
                    for (net, sites) in observations {
                        if chained.contains(&net) {
                            supports.push((net, sites));
                            continue;
                        }
                        let f = function(net);
                        let key = f.min(!f & full);
                        let priority = |net: NetId| {
                            (
                                !matches!(netlist.nets[net].driver, NetDriver::Input(_)),
                                net,
                            )
                        };
                        if representatives
                            .get(&key)
                            .is_none_or(|(old, _)| priority(net) < priority(*old))
                        {
                            representatives.insert(key, (net, sites));
                        }
                    }
                    for (net, _) in &supports {
                        let f = function(*net);
                        representatives.remove(&f.min(!f & full));
                    }
                    for (key, (net, _)) in &mut representatives {
                        let priority = |candidate: NetId| {
                            let f = function(candidate);
                            let carried = state.cells.iter().any(|(p, kind)| {
                                p.1 + 1 == length
                                    && *kind == CellKind::Repeater(Direction::South)
                                    && state.signals.get(p) == Some(&f)
                            });
                            (
                                !matches!(netlist.nets[candidate].driver, NetDriver::Input(_)),
                                !carried,
                                candidate,
                            )
                        };
                        if let Some(representative) = placed
                            .iter()
                            .copied()
                            .filter(|&candidate| {
                                let f = function(candidate);
                                f.min(!f & full) == *key
                            })
                            .min_by_key(|&candidate| priority(candidate))
                        {
                            *net = representative;
                        }
                    }
                    observations = representatives.into_values().chain(supports).collect();
                    tracing::info!(
                        gate = gate_name,
                        before,
                        after = observations.len(),
                        "construction frontier compressed"
                    );
                }
                if config.driving_seams {
                    for (net, _) in &observations {
                        if !chained.contains(net) {
                            exact
                                .driving_outputs
                                .insert(netlist.nets[*net].name.clone());
                        }
                    }
                }
                if config.scaffold_seams {
                    eyre::ensure!(
                        overlap == 0 && window >= 2,
                        "scaffolded seams need zero overlap and window >= 2"
                    );
                    let values = netlist.net_values();
                    if config.seam_pitch > 0 {
                        let pitch = config.seam_pitch;
                        eyre::ensure!(
                            pitch >= 3 || (config.gate_scaffold && pitch == 2),
                            "seam pitch must leave tap clearance"
                        );
                        let slots = (1..dim.2.saturating_sub(1))
                            .step_by(pitch)
                            .flat_map(|z| {
                                (1..dim.0.saturating_sub(1))
                                    .step_by(pitch)
                                    .map(move |x| Position(x, dim.1 - 1, z))
                            })
                            .filter(|p| !config.gate_scaffold || p.2 == 1)
                            .collect::<Vec<_>>();
                        let input_names = netlist.input_names();
                        let input_slots = (0..input_names.len())
                            .map(|i| {
                                slots
                                    .get(i * config.seam_primary_stride)
                                    .copied()
                                    .ok_or_else(|| eyre::eyre!("not enough primary slots"))
                            })
                            .collect::<eyre::Result<Vec<_>>>()?;
                        eyre::ensure!(
                            slots.len() >= observations.len().max(input_names.len()),
                            "not enough spaced seam slots"
                        );
                        let function = |net: NetId| {
                            values[net]
                                .iter()
                                .enumerate()
                                .fold(0u64, |mask, (i, &on)| mask | (u64::from(on) << i))
                        };
                        let mut used = BTreeSet::new();
                        used.extend(input_slots.iter().copied());
                        let mut assigned = BTreeMap::<u64, Position>::new();
                        for (net, sites) in &mut observations {
                            if let NetDriver::Input(name) = &netlist.nets[*net].driver {
                                let slot = input_slots
                                    [input_names.iter().position(|n| n == name).unwrap()];
                                *sites = vec![slot];
                                used.insert(slot);
                                assigned.insert(function(*net), slot);
                            }
                        }
                        // Reserve every surviving old rail before allocating
                        // new results; net order must not evict a live port.
                        for (net, sites) in &mut observations {
                            if matches!(netlist.nets[*net].driver, NetDriver::Input(_)) {
                                continue;
                            }
                            let f = function(*net);
                            if let Some(slot) = state.cells.iter().find_map(|(p, kind)| {
                                let slot = Position(p.0, dim.1 - 1, p.2);
                                (p.1 + 1 == length
                                    && *kind == CellKind::Repeater(Direction::South)
                                    && state.signals.get(p) == Some(&f)
                                    && slots.contains(&slot)
                                    && !used.contains(&slot))
                                .then_some(slot)
                            }) {
                                *sites = vec![slot];
                                used.insert(slot);
                                assigned.insert(f, slot);
                            }
                        }
                        for (net, sites) in &mut observations {
                            if matches!(netlist.nets[*net].driver, NetDriver::Input(_)) {
                                continue;
                            }
                            let f = function(*net);
                            if let Some(slot) = assigned.get(&f) {
                                *sites = vec![*slot];
                                continue;
                            }
                            let old = state.cells.iter().find_map(|(p, kind)| {
                                (p.1 + 1 == length
                                    && *kind == CellKind::Repeater(Direction::South)
                                    && state.signals.get(p) == Some(&f))
                                .then_some(Position(p.0, dim.1 - 1, p.2))
                            });
                            let parents = netlist.nets[*net]
                                .gate_inputs
                                .iter()
                                .filter_map(|parent| {
                                    let f = function(*parent);
                                    let full = u64::MAX >> (64 - (1usize << netlist.case_bits()));
                                    assigned
                                        .get(&f)
                                        .or_else(|| assigned.get(&(!f & full)))
                                        .copied()
                                })
                                .collect::<Vec<_>>();
                            let slot = old
                                .filter(|p| slots.contains(p) && !used.contains(p))
                                .or_else(|| {
                                    slots
                                        .iter()
                                        .filter(|p| !used.contains(p))
                                        .min_by_key(|p| {
                                            parents
                                                .iter()
                                                .map(|q| p.0.abs_diff(q.0) + p.2.abs_diff(q.2))
                                                .sum::<usize>()
                                        })
                                        .copied()
                                })
                                .ok_or_else(|| eyre::eyre!("spaced port allocation failed"))?;
                            *sites = vec![slot];
                            used.insert(slot);
                            assigned.insert(f, slot);
                        }
                        // The complete seam is an interface, not spare logic space.
                        for x in 0..dim.0 {
                            for z in 0..dim.2 {
                                exact
                                    .fixed_cells
                                    .insert(Position(x, dim.1 - 1, z), CellKind::Air);
                            }
                        }
                        for (_, sites) in &observations {
                            let p = sites[0];
                            exact
                                .fixed_cells
                                .insert(p, CellKind::Repeater(Direction::South));
                            exact
                                .fixed_cells
                                .insert(Position(p.0, p.1, p.2 - 1), CellKind::Solid);
                        }
                        for &input in &needed_inputs {
                            let name = &netlist.nets[input].name;
                            let slot =
                                input_slots[input_names.iter().position(|n| n == name).unwrap()];
                            let p = Position(
                                slot.0,
                                if config.gate_scaffold && length > 0 {
                                    length - 1
                                } else {
                                    length
                                },
                                slot.2,
                            );
                            eyre::ensure!(
                                config.input_policies.get(name) != Some(&InputPolicy::MinYFace)
                                    || length == 0,
                                "face input must enter at the first seam"
                            );
                            exact
                                .input_sites
                                .insert(name.clone(), vec![(p, Direction::North)]);
                            if config.gate_scaffold {
                                exact
                                    .fixed_cells
                                    .insert(p, CellKind::Switch(Direction::North));
                                exact.given_signals.remove(&p);
                                let tap_y = p.1 + 1;
                                exact
                                    .fixed_cells
                                    .insert(Position(p.0, tap_y, p.2), CellKind::Solid);
                                for y in tap_y + 1..dim.1 {
                                    exact.fixed_cells.insert(
                                        Position(p.0, y, p.2),
                                        CellKind::Repeater(Direction::South),
                                    );
                                    exact
                                        .fixed_cells
                                        .insert(Position(p.0, y, p.2 - 1), CellKind::Solid);
                                }
                            }
                        }
                    }
                    let mut scaffolded = 0;
                    // A signal consumed by this gate may retire at this step,
                    // so it has no outgoing observation. Still provide its
                    // incoming diode with a solid tap in the new window.
                    let full = u64::MAX >> (64 - (1usize << netlist.case_bits()));
                    for &input in gates
                        .iter()
                        .flat_map(|&gate| &netlist.nets[gate].gate_inputs)
                    {
                        let f = values[input]
                            .iter()
                            .enumerate()
                            .fold(0u64, |mask, (i, &on)| mask | (u64::from(on) << i));
                        for (p, kind) in &state.cells {
                            if p.1 + 1 == length
                                && *kind == CellKind::Repeater(Direction::South)
                                && state
                                    .signals
                                    .get(p)
                                    .is_some_and(|old| *old == f || *old == (!f & full))
                            {
                                exact
                                    .fixed_cells
                                    .insert(Position(p.0, length, p.2), CellKind::Solid);
                            }
                        }
                    }
                    for (net, sites) in &mut observations {
                        if chained.contains(net) {
                            continue;
                        }
                        let function = values[*net]
                            .iter()
                            .enumerate()
                            .fold(0u64, |mask, (i, &on)| mask | (u64::from(on) << i));
                        let port = state.cells.iter().find_map(|(p, kind)| {
                            (p.1 + 1 == length
                                && *kind == CellKind::Repeater(Direction::South)
                                && state.signals.get(p) == Some(&function))
                            .then_some(*p)
                        });
                        if let Some(port) = port {
                            scaffolded += 1;
                            if config.seam_pitch > 0
                                && (sites[0].0 != port.0 || sites[0].2 != port.2)
                            {
                                continue;
                            }
                            *sites = vec![Position(port.0, dim.1 - 1, port.2)];
                            for y in length..dim.1 {
                                let kind = if y == length {
                                    CellKind::Solid
                                } else {
                                    CellKind::Repeater(Direction::South)
                                };
                                exact.fixed_cells.insert(Position(port.0, y, port.2), kind);
                                if y != length {
                                    exact
                                        .fixed_cells
                                        .insert(Position(port.0, y, port.2 - 1), CellKind::Solid);
                                }
                            }
                        }
                    }
                    tracing::info!(
                        gate = gate_name,
                        scaffolded,
                        "construction extends fixed repeater rails"
                    );
                }
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
                // A chained OR sits on its support block: a block of the
                // last slice for the chain's first OR, the same block after.
                for (net, positions) in observations.iter_mut() {
                    if !chained.contains(net) {
                        continue;
                    }
                    let block = state
                        .supports
                        .get(net)
                        .copied()
                        .or(support.filter(|_| gates.contains(net)));
                    if let Some(block) = block {
                        *positions = vec![block];
                    }
                    exact
                        .solid_observations
                        .insert(netlist.nets[*net].name.clone());
                }
                exact.observations = Some(observations);
            }
            if config.gate_scaffold {
                eyre::ensure!(
                    config.scaffold_seams
                        && config.seam_pitch >= 3
                        && overlap == 0
                        && gates.len() == 1,
                    "NOR templates need a single gate, pitch >= 3 and zero-overlap scaffolded seams"
                );
                let gate = gates[0];
                let values = netlist.net_values();
                let full = u64::MAX >> (64 - (1usize << netlist.case_bits()));
                let function = |net: NetId| {
                    values[net]
                        .iter()
                        .enumerate()
                        .fold(0u64, |mask, (i, &on)| mask | (u64::from(on) << i))
                };
                let target = function(gate);
                let observed = exact
                    .observations
                    .as_ref()
                    .is_none_or(|obs| obs.iter().any(|(net, _)| function(*net) == target));
                let mut template = BTreeMap::new();
                if observed {
                    eyre::ensure!(
                        netlist.nets[gate].driver == NetDriver::Gate,
                        "NOR template does not support explicit OR stages"
                    );
                    let start_y = length + usize::from(length == 0);
                    let mut taps = Vec::new();
                    for &input in &netlist.nets[gate].gate_inputs {
                        let f = function(input);
                        let old = state.cells.iter().find_map(|(p, kind)| {
                            if p.1 + 1 == length && *kind == CellKind::Repeater(Direction::South) {
                                let old = *state.signals.get(p)?;
                                if old == f || old == (!f & full) {
                                    return Some((Position(p.0, start_y, p.2), old != f));
                                }
                            }
                            None
                        });
                        let new = needed_inputs.iter().find_map(|&net| {
                            let source = function(net);
                            if source != f && source != (!f & full) {
                                return None;
                            }
                            let site = exact.input_sites.get(&netlist.nets[net].name)?.first()?.0;
                            Some((Position(site.0, start_y, site.2), source != f))
                        });
                        taps.push(old.or(new).ok_or_else(|| {
                            eyre::eyre!("no template input tap for {}", netlist.nets[input].name)
                        })?);
                    }
                    let out = exact
                        .observations
                        .as_ref()
                        .and_then(|obs| {
                            obs.iter().find_map(|(net, sites)| {
                                (function(*net) == target
                                    && sites.len() == 1
                                    && sites[0].1 + 1 == dim.1)
                                    .then(|| sites[0].0)
                            })
                        })
                        .or_else(|| {
                            (1..dim.0.saturating_sub(2))
                                .step_by(config.seam_pitch)
                                .find(|&x| {
                                    exact
                                        .fixed_cells
                                        .get(&Position(x, dim.1 - 1, 1))
                                        .is_none_or(|k| *k == CellKind::Air)
                                })
                        })
                        .ok_or_else(|| eyre::eyre!("no output slot for NOR template"))?;
                    template = super::nor_scaffold::nor_template(&taps, out, start_y, dim)?;
                }
                // Search deletion subsets of a generic initial geometry. This
                // is not an unrestricted placement solve; compaction frees
                // spatial windows after the complete baseline is verified.
                for x in 0..dim.0 {
                    for y in length..dim.1 {
                        for z in 0..dim.2 {
                            exact
                                .fixed_cells
                                .entry(Position(x, y, z))
                                .or_insert(CellKind::Air);
                        }
                    }
                }
                for (p, kind) in template {
                    if let Some(old) = exact.fixed_cells.get(&p) {
                        if *old != CellKind::Air {
                            eyre::ensure!(
                                *old == kind,
                                "template/scaffold conflict at {p:?}: {old:?} / {kind:?}"
                            );
                            continue;
                        }
                    }
                    exact.fixed_cells.remove(&p);
                    exact.removable_cells.insert(p, kind);
                }
            }
            let step_started = Instant::now();
            if config.gate_scaffold && std::env::var_os("EXACT_TEMPLATE_DIAGNOSE").is_some() {
                let mut kinds = vec![CellKind::Air; dim.0 * dim.1 * dim.2];
                for (p, k) in exact.fixed_cells.iter().chain(exact.removable_cells.iter()) {
                    kinds[(p.2 * dim.1 + p.1) * dim.0 + p.0] = *k;
                }
                let world = super::verify::build_world(dim, &kinds);
                let mut sim =
                    crate::world::simulator::Simulator::from_settled_with_limits_and_trace(
                        &crate::world::World::from(&world),
                        4096,
                        2_000_000,
                        0,
                    )?;
                let positions = world
                    .iter_block()
                    .into_iter()
                    .map(|(p, _)| p)
                    .collect::<Vec<_>>();
                let names = netlist.input_names();
                let mut actual = BTreeMap::<Position, u64>::new();
                for case in 0..(1usize << netlist.case_bits()) {
                    sim.advance_idle_cycles(61)?;
                    let drive = names
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| !exact.absent_inputs.contains(*n))
                        .map(|(i, n)| (exact.input_sites[n][0].0, case & (1 << i) != 0))
                        .collect::<Vec<_>>();
                    sim.drive_inputs_with_limits(drive, 4096, 2_000_000)?;
                    for p in &positions {
                        if sim.world()[*p].kind.is_powered() {
                            *actual.entry(*p).or_default() |= 1 << case;
                        }
                    }
                }
                let values = netlist.net_values();
                let observations = exact.observations.clone().unwrap_or_else(|| {
                    netlist
                        .outputs
                        .iter()
                        .map(|(_, n)| (*n, positions.clone()))
                        .collect()
                });
                let mismatches = observations
                    .iter()
                    .filter_map(|(net, sites)| {
                        let target = values[*net]
                            .iter()
                            .enumerate()
                            .fold(0u64, |m, (i, &on)| m | (u64::from(on) << i));
                        (!sites
                            .iter()
                            .any(|p| actual.get(p).copied().unwrap_or(0) == target))
                        .then(|| netlist.nets[*net].name.clone())
                    })
                    .collect::<Vec<_>>();
                let frozen_mismatches = exact
                    .given_signals
                    .iter()
                    .filter(|(p, f)| actual.get(p).copied().unwrap_or(0) != **f)
                    .count();
                for (net, sites) in &observations {
                    if sites.len() == 1 {
                        tracing::info!(net=netlist.nets[*net].name,site=?sites[0],actual=actual.get(&sites[0]),"template port value");
                    }
                }
                tracing::info!(gate=gate_name,?mismatches,frozen_mismatches,functions=?actual.values().copied().collect::<BTreeSet<_>>(),"template direct simulation");
            }
            let (outcome, stats) = self.place(&exact)?;
            tracing::info!(
                gate = gate_name,
                encode_seconds = stats.encode_time.as_secs_f64(),
                solve_seconds = stats.solve_time.as_secs_f64(),
                variables = stats.variables,
                clauses = stats.clauses,
                "construction SAT model"
            );
            let ExactOutcome::Placed(placement) = outcome else {
                tracing::info!(
                    gate = gate_name,
                    window,
                    overlap,
                    seconds = step_started.elapsed().as_secs_f64(),
                    refinements = stats.refinements,
                    outcome = ?outcome,
                    "construction step failed"
                );
                if matches!(outcome, ExactOutcome::Unknown { .. }) {
                    self.diagnose_failed_step(&exact, &gate_name, window)?;
                    static DIAGNOSED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if std::env::var("EXACT_DIAGNOSE_UNKNOWN").is_ok()
                        && !DIAGNOSED.swap(true, std::sync::atomic::Ordering::SeqCst)
                    {
                        self.diagnose_infeasible_step(&exact, &gate_name, window);
                    }
                }
                if matches!(outcome, ExactOutcome::Infeasible)
                    && std::env::var("EXACT_DIAGNOSE_INFEASIBLE").is_ok()
                {
                    self.diagnose_infeasible_step(&exact, &gate_name, window);
                }
                continue;
            };
            let mut placement = placement;
            if let Some(budget) = config.step_delete {
                let mut deleting = exact.clone();
                deleting.given_signals.clear();
                let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
                deleting.time_limit = Some(remaining.map_or(budget, |r| r.min(budget)));
                let layout = ExactLayout::from_placement(dim, &placement);
                if let Ok((ExactOutcome::Placed(better), _)) =
                    self.minimize_existing(&layout, &deleting)
                {
                    if better.cells.len() < placement.cells.len() {
                        placement = better;
                    }
                }
            }
            if let Some(budget) = config.step_timing {
                let lower = self.step_lower_bounds(gates, state);
                placement =
                    Box::new(self.shorten_step(&exact, *placement, &lower, budget, deadline));
            }
            if let Some(budget) = config.step_optimize {
                // Trade the window's dead wires away while it is small: ask
                // for fewer blocks than the layout just found, minimizing.
                let remaining =
                    deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
                let mut tighter = exact.clone();
                tighter.optimize = true;
                tighter.max_blocks = Some(placement.cells.len().saturating_sub(1));
                if config.step_timing.is_some() {
                    keep_delays(&mut tighter, &placement);
                }
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
                gate = gate_name,
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
            available.extend(gates.iter().copied());
            let mut supports = state.supports.clone();
            for &gate in gates {
                if !chained.contains(&gate) {
                    continue;
                }
                if let Some(output) = placement
                    .placed
                    .outputs
                    .iter()
                    .find(|output| output.name == netlist.nets[gate].name)
                {
                    supports.insert(gate, output.position());
                }
            }
            let next = StepState {
                label: gate_name.clone(),
                length: dim.1,
                cells,
                signals: placement.signals.iter().copied().collect(),
                placed_inputs,
                available,
                supports,
                result: Some((dim, *placement)),
            };
            return Ok(Some((next, window_cells, (gate_name, window, seconds))));
        }
        Ok(None)
    }
}
