# Exact (SAT-based) local placer

`src/transform/place_and_route/local_placer/exact/` places a small NOR
netlist in a fixed box by solving placement and routing together with the
CaDiCaL SAT solver (bundled through the `cadical` crate). Every candidate is
rebuilt as a world and checked with `world::simulator` in all input cases and
settled transitions before it is returned. It complements the beam-search
local placer, whose committed early routes can seal off later fan-out
branches (see `local_full_adder_diagnostic.md`).

## Why this design (survey, 2026-10-03)

Dense redstone cells resemble transistor-level *cell layout synthesis*, not
large-scale P&R:

- Cell generators solve placement, routing, and design rules together with
  exact solvers: SMTCell (UCSD, Z3), BonnCell (Bonn/IBM; CPLEX plus CaDiCaL
  routability checks), and Cortadella et al.'s pure-SAT cell routing. BonnCell
  reports that rip-up-and-reroute fails on dense layouts with design rules.
- FPGA and ASIC routers (PathFinder in VPR/nextpnr, TritonRoute in OpenROAD)
  negotiate congestion iteratively. They are fast but incomplete, and dense
  layouts with electrical spacing rules tend to leave violations behind.
- Published Minecraft compilers (PERSHING/Dewey, MinecraftHDL, LogicLoom,
  Nucleation) use annealing plus maze or channel routing. Their full adders
  are 7–40x larger than the hand-made 2x14x10 cell. Only SAT synthesis of
  flat cells (alloc.dev "Redstone with SAT", redstone_sat with Kissat) has
  reached hand-made density.
- Solvers considered: CaDiCaL (MIT, incremental, builds here), Kissat (faster
  on some instances, not incremental), Z3/cvc5, OR-Tools CP-SAT (native
  OR-Tools install), clingo (ASP), Gurobi/CPLEX (academic licenses).

## Model (`exact_placer.rsdsl`)

The constraints live in the rsdsl model
`src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl`.
`dsl.rs` turns the netlist, vocabulary, pins, and config into an rsdsl
instance, grounds the model, and reads the variables back for decoding.
Nothing in Rust adds constraints except blocking clauses (simulator
rejections, `blocked`) and lazy loop formulas. `encode.rs` is the original
hand-written encoder of the same model, kept behind the hidden
`legacy_encoder` flag for parity checks (see `solver_dsl_design.md`,
section 11).

`ExactPlacerConfig::model_file` grounds another model file instead, and
`model_params` overrides model params (for example `max_repeaters = 0`), so
experiments need neither Rust changes nor CNF edits.

- **Blocks.** Each cell picks exactly one kind: air, solid, dust, torch (five
  attachments), repeater (four directions), or an input switch at a
  configured site. Physical support is required.
- **Signal classes.** Each non-air cell also picks a signal class. A class is
  one Boolean function of the inputs, drawn from a *vocabulary*: every net
  function of the netlist, its complement, and "unpowered".
  - Per-case power literals follow from the class.
  - Gates are not required one by one. Any composition of vocabulary
    functions is functionally correct. The solver may therefore duplicate
    gates, build inverter towers (n1 → ~n1 → n1, as in the manual cell), or
    OR signals on a shared block.
- **Power relations.** Relations mirror the simulator:
  - dust connection and pointing shape, including step-up and step-down;
  - dust weakly powering its support and the blocks it points into;
  - strong power from torches below, repeaters, and attached switches;
  - torch and repeater outputs, and repeaters reading from blocks;
  - repeater side-locking is forbidden;
  - switch soft power: the simulator lets it reach repeaters but not torches,
    so readers next to a switch are forbidden.
- **Soundness.** A powered source powers its sink. Dust and repeaters carry
  exactly their source's function; a solid may OR several sources.
- **Coverage.** Every powered non-driver has, in each case where it is on, a
  powered source with a smaller local rank and a stage that is not larger.
- **Torches.** A torch carries the complement of its support's class and sits
  on a strictly larger stage, which excludes latches and oscillators.
- **Requirements.** Every input has one switch, and every output (or
  configured observation) is visible on a dust, torch, or repeater at an
  allowed site.

Ranks and stages use order encodings (`rank_levels`, `stage_levels`). Setting
either to 0 enforces acyclicity lazily instead (`acyclic.rs`, using
answer-set-style loop formulas and feedback cuts). The lazy mode is correct
but converged far slower in measurements, so it is not the default.

**Fidelity check.** `model_accepts_manual_right_inputs_full_adder` and
`model_accepts_other_manual_full_adders` fix every block of the verified
2x14x10, 2x13x9, and 2x17x10 manual cells and require the solver to find
labels. The model therefore accepts known-good dense layouts.

### Limitations

- A single rank per cell cannot order a dust cell that carries power in
  opposite directions in different input cases. The 2x20x20 manual cell does
  this at its carry-out support (n1 and n5 meet through one dust cell), so
  that cell is rejected. Fixing it needs per-case ranks, about 8x the cost.

### Closing model–simulator gaps (2026-10-03)

Simulator rejections, not search, dominated slow construction steps: on the
full adder (seed 1) every worker of a step used up its eight rejections. Two
diagnostics find the cause: `EXACT_DUMP_REJECTIONS=<dir>` writes every
rejected layout as an RCELL whose header lists, per input case, the *root*
mismatches (cells whose simulated power differs from the model while all of
their modeled sources agree), and the ignored test `explain_rejected_block`
traces one block's events. Three causes, in order of frequency:

1. **Start-up burnout.** The simulator starts with every torch lit; settling
   from there sends glitch waves through deep NOR networks, and a torch that
   toggles eight times in 60 cycles burns out for good. A cell built in place,
   or pasted with saved torch states, never sees that transient. The verifier
   now settles with `Simulator::from_settled_with_limits_and_trace` (no burnout
   accounting during the initial settle; every later input change counts), the
   exported NBT stores the settled torch states, and generated RCELL files
   declare `start settled;` so the RCELL verifier uses the same start.
2. **Strong power is per case.** A block powers adjacent dust only while one
   of its strong sources (torch below, repeater facing it, attached switch) is
   on. The model used the block's total power whenever such a source merely
   existed. `Strong[k, c]` in `exact_placer.rsdsl` fixes the soundness rule
   for solid→dust relations (`legacy_semantics = true` restores the old rule
   for parity tests).
3. **Simulator bug: sources on one block.** Switches (and torches below a
   block) sent hard power without naming their source, so two of them on one
   block shared a bookkeeping key and turning one off unpowered the block.
   They now carry their source direction, like repeaters.

Tried and dropped: requiring stage order on every connection, not only on
contributing ones (to forbid gated feedback). The burnouts turned out to be
start-up glitches, not feedback, and the rule made search slower.

Effect: OR 1x5x3 went from 64 rejections (Unknown) to 0; full-adder
construction, seed 1, from 438 s with 258 rejections to 27 s with none, and the
generated cell passes `rcell` verification. Seed 2 still fails to place gate
`s` within four slices, now on search time alone (the legacy encoder also
fails that seed).

### Optimizing a cost (2026-10-03)

`ExactPlacerConfig::optimize` keeps searching after the first verified layout
for a cheaper one under the model's `minimize` cost (default: non-air blocks,
switches included; `model_params` `block_cost`, `repeater_cost`, `torch_cost`
change the weights). Each worker keeps one incremental CaDiCaL instance and
assumes "cost below the best so far" through a totalizer over the cost
literals; a better layout from any worker interrupts the others so they
resume with the tighter bound. The result is the best layout found, with
`stats.cost`, `stats.improvements`, and `stats.optimal` (set when a bound was
proven unsatisfiable: no valid layout is cheaper). `place_minimizing_blocks`
now uses this instead of re-encoding for every bound.

Small boxes are proven optimal in well under a second (inverter 1x4x2, NOR
1x5x2). XOR 2x6x4 drops from 44 to 30 blocks within a second and to 28 in
two minutes, without a proof: showing that no smaller layout exists is the
hard direction (`measure_optimize`). AND 2x4x3 finds its optimum (9) in
0.3 s and spends the remaining ~9 s proving that 8 is impossible.

Compaction's block-reduction phase now optimizes each 3-slice window in one
solve (`CompactionConfig::optimize_windows`) instead of asking for one block
fewer per attempt; a window proven optimal is skipped until the layout
changes. On the seed-1 full adder (same 900 s budget, same 2x14x9 box) this
reaches 153 blocks instead of 171.

Proof time, not search, limits optimality. `ExactPlacerConfig::core_guided`
adds core-guided lower bounding (OLL, the algorithm behind the RC2 MaxSAT
solver) on the last worker: every cost literal is assumed false, each
unsatisfiable core raises the bound by its smallest weight, and the core is
relaxed through a totalizer built into the running solver. It is correct
(pure OLL proves the same optima, with the bound meeting the cost) but off by
default: the bound rises one unit per core and each core takes longer than
the last. On AND 2x4x3 it reaches 8 after 7.9 s while the direct proof of 9
finishes at 9.4 s; on XOR 2x6x4 it reaches only 8 in two minutes against a
best layout of 30.

Implied structural bound (on when optimizing): every torch is a NOR gate over
vocabulary functions, and torches with different signals need supports with
different signals, so a layout needs at least as many torches, and as many
solids, as the smallest NOR network that produces the observed functions.
`dsl::min_torches` finds that number by breadth-first search over sets of
available classes (a solid can carry any OR of available classes that is
itself a class; a torch adds the complement), and the model rule
"구조적 하한" requires both counts. AND 2x4x3 needs 3 (`~a`, `~b`, their NOR),
XOR 4 within its vocabulary, the full adder 9 (computed in 0.2 ms). The AND
proof drops from about 10.8 s to 8.5 s; for plain placement the extra
constraint slowed XOR (one of six seeds timed out), so it is only set with
`optimize` (`compare_torch_lower_bound`).

Tried and left off by default: the model's `symmetry_breaking` rules (one
contributing source per dust/repeater; zero rank/stage without an incoming
contribution). They are sound but did not speed up proofs (AND 2x4x3: 10.5 s
vs 10.3 s) and were mixed for finding layouts (XOR 2x6x4, six seeds: faster
on four, one timeout), measured by `compare_symmetry_breaking`.

## Search strategies

### Monolithic solve (`ExactLocalPlacer::place`)

Solves the whole box with a parallel portfolio: seeds, phase 0 or 1, and
stable-only mode. The first verified layout wins. An `Infeasible` answer is a
proof under the encoded rules and bounds.

Measured with 8–12 workers:

| Problem | Result |
| --- | --- |
| Inverter, NOR, XOR (2x6x4) | Seconds |
| 2-input NOR in 1x2x1 | Proven infeasible |
| XNOR core (4 gates) in 2x7x6 | 14–27 s |
| XNOR core (4 gates) in 2x7x10, fixed pins | Over 120 s |
| Full adder, 2x14x10 manual pins | No result in 600 s |
| Full adder, 2x16x12 and 2x20x14 | No result in 300 s |

Difficulty grows exponentially with the number of *free* cells. Fixing the
manual cell's first Y slices and completing the rest took:

| Free cells | Time |
| --- | --- |
| 20 | 0.17 s |
| 40 | 3.2 s |
| 60 | 13 s (Kissat 19 s) |
| 80 | Over 300 s (Kissat 133 s) |

Neither swapping solvers nor switching to an ASP formulation (clingo,
prototyped) changed this trend. The clingo prototype's apparent 54-second
full adder used a latch; once torch feedback was forbidden it was not faster.

### Windowed construction and compaction (`construct.rs`, `compact.rs`)

These keep every exact solve small:

1. **Construction.** Gates are appended in topological order, each in a
   short Y window. Earlier slices stay fixed, apart from an overlap that may
   be reshaped. Inputs appear when first needed; face-bound operands start at
   Y = 0. Every net still read by a later gate must be observable on the
   window's last slice. Outputs with a face policy are carried there too;
   other outputs only need to stay observable somewhere once no later gate
   reads them (`early_outputs`). The last step checks the real outputs, for
   example the sum on the Y-max face.
2. **Compaction.** One Y slice or Z layer is removed and everything beyond it
   shifted back. Only the cells next to the seam are re-solved; all others
   stay fixed. Accepted repairs are simulator-verified, and the emptiest
   slices are tried first. When no seam can be repaired, the window radius
   grows up to `max_window_radius`.
3. **Block minimization.** Optionally, a `reduction_window`-slice window
   (3) slides along each of `reduction_axes` (Y slices, then Z layers) and
   is re-solved for fewer blocks (in one optimizing solve with
   `optimize_windows`).
4. **Rounds.** With `repeat_rounds`, slice removal and block minimization
   alternate until a round saves no block: fewer blocks often free a slice
   that could not be removed before.

Construction depends on the seed: a step can leave live signals at the seam
in a shape the next gate cannot use within its time limit. When a gate fails
in every window, construction starts over with a new seed
(`ConstructionConfig::max_restarts`, 7; seeds are `seed + k * 104729`).
`restart_after` (10 minutes) also caps each attempt, and step solves are
capped by the remaining budget. Measured on the full adder (height 10,
windows 2..4, 60-second steps, 8 workers):

| Seed | Backtracking, restart after 10 min | Restart on first failure |
|------|------------------------------------|--------------------------|
| 1    | 27 s                               | 27 s                     |
| 2    | 632 s, 1 restart                   | 261 s, 1 restart         |
| 3    | 33 s                               | 33 s                     |
| 4    | 649 s, 1 restart                   | 352 s, 1 restart         |

Without restarts, seed 2 used up its backtracks and failed.

A failed gate costs about three minutes (three windows at the step limit),
while a restart rebuilds the same prefix in about 30 seconds. Three finer
repairs are available but off by default because none rescued seeds 2 or 4:

- `max_backtracks` (0): re-solve the previous step with its layout blocked.
  Six backtracks across seeds 2 and 4 each re-placed `g22`, and gate `s`
  failed again every time.
- `max_overlap` (= `overlap`): re-solve up to that many slices of the
  previous layout before backtracking. All nine window/overlap combinations
  for gate `s` timed out.
- `block_seam`: on backtracking, block only the seam slice instead of the
  whole previous window. The re-solved seam still left `s` unplaceable.

Because every live net crosses every seam, the gate order matters.
`GateOrder::SmallestConeFirst` (the default) builds output by output,
smallest fan-in cone first, depth-first inside each cone, so an output and
its private logic finish before the next output starts. Together with
`early_outputs`, the full adder's live nets after each step drop from
`[3, 3, 3, 2, 4, 4, 4, 3, 2]` to `[3, 3, 3, 2, 4, 3, 3, 2, 0]`. The 2-bit
adder's maximum drops from 8 to 6. `GateOrder::NetIndex` is the original
depth-first order by net index. Full-adder construction with both changes
(same settings as above):

| Seed | Before | Smallest cone first, early outputs |
|------|--------|------------------------------------|
| 1    | 27 s, 300 blocks | 31 s, 278 blocks |
| 2    | 261 s, 1 restart, 306 blocks | 66 s, no restart, 297 blocks |
| 3    | 33 s, 297 blocks | 22 s, 297 blocks |
| 4    | 352 s, 1 restart, 300 blocks | 372 s, 1 restart, 279 blocks |

Seed 4 still fails at the last gate (`s`, which must drive out of the Y-max
face) on its first attempt.

**Given signals (2026-10-04).** A step fixes the kinds of the frozen slices,
but the model still encoded their signals, contributing sources, ranks and
stages, so every step re-justified the whole layout built so far. The
diagnostic hook `EXACT_DIAGNOSE_FAILED_STEP=<prefix>` (with
`EXACT_DIAGNOSE_SECONDS`, default 600) writes the first step that times out
as DIMACS and solves it again for longer. For the 2-bit adder's gate `g46`
(a 2x38x10 box, of which 60 cells are free), the CNF had 366 331 variables and
2 881 975 clauses. 92% of the named variables belonged to the 700 frozen
cells, and ten more minutes still ended in Unknown.

`ConstructionConfig::given_frozen_signals` (default on) now passes the
previous step's signals for frozen cells as `ExactPlacerConfig::
given_signals`. The model fixes their class (`given_sig`) and drops the
justification rules for them (contributing source, rank and stage order, and
the torch's stage). They still act as sources for the free cells, and the
simulator still checks the whole layout. Full-adder construction, seeds 1..4:

| Seed | Before | Given signals |
|------|--------|---------------|
| 1    | 31 s | 14 s |
| 2    | 66 s | 17 s |
| 3    | 22 s | 16 s |
| 4    | 372 s, 1 restart | 51 s, no restart |

The 2-bit adder, which failed in all eight attempts before, now constructs
on the first attempt in 146 s (2x52x10, 799 blocks). Its gate `g46` takes
3.8 s. The 4:1 mux gets further (to g58 or the four-input `out_n`, gates 16
and 19 of 20) but its first four attempts still timed out there.

`ExactLocalPlacer::synthesize` runs construction followed by compaction.
`ExactLayout::from_rcell` loads any RCELL so that existing cells can be
compacted too (`recompact_full_adder_rcell`).

### Results (2026-10-03)

The pipeline started from the `nor9` netlist with no seed layout. The
interface was the one requested for the hand-made 2x14x10 cell: both operands
on the Y-min face, the sum driving out of the Y-max face, and carry-in and
carry-out free.

**Seed 1, step by step:**

1. Construction produced a valid 2x18x10 layout with 304 blocks in 60 s. It
   used 9 steps, each with a window of 2 plus one overlap slice; the slowest
   step took 19 s.
2. Compaction at window radius 1 reached 2x14x7 with 170 blocks in 109 s,
   over 89 repair attempts with a 20 s limit.
3. Recompaction with radius escalation to 2 plus block minimization reached
   **2x13x7 with 130 blocks** in 20 minutes.
4. Saved as `test/full-adder-exact-2x13x7.rcell`.

Seeds 2 and 3 each compacted to 2x15x9 at radius 1. Results vary by seed,
and radius 1 alone can stall in a local minimum.

| Cell | Box | Volume | Blocks |
| --- | --- | ---: | ---: |
| Manual, right-hand inputs | 2x14x10 | 280 | 129 |
| Generated (seed 1), construct + compact | 2x14x7 | 196 | 170 |
| Generated (seed 1), after recompaction | 2x13x7 | 182 | 130 |
| Generated (seed 2), after recompaction | 2x15x9 | 270 | 179 |
| Manual cell + automatic compaction | 2x10x10 | 200 | 99 |

- The compactor also improves hand-made cells. Starting from
  `full-adder-right-inputs-2x14x10.rcell`, it removed four Y slices and 30
  blocks in 20 minutes. Operands stayed at the same Y-min positions and the
  sum is still an outward repeater on the Y-max face. Saved as
  `test/full-adder-right-inputs-compacted-2x10x10.rcell`.
- Every accepted step was simulator-verified. The saved cells pass the `rcell`
  binary and the physical-cell regression tests (eight fresh cases, 64
  settled transitions, no torch burnout, interface positions).
- `OutputPolicy::MaxYFace` now requires a driver on the face (a torch or an
  outward repeater), not just an observation.

Two encoding details mattered for these runs:

- Construction carries early signals such as `n1` along the whole box.
  Requiring rank to rise across repeaters made long pass-through wires need
  more than 24 rank levels. Repeaters now restart the rank and advance the
  stage instead, which also cut the exact unit tests from about 28 s to 2 s.
- Each step must leave every still-needed net observable on its last slice.
  Without the overlap slice, a step could expose a signal in a form the next
  step cannot extend, such as an outward repeater.

### Further compaction (2026-10-04)

The 153-block 2x14x9 cell was not converged. The pipeline ran slice removal
once, then block reduction, and never retried slice removal on the smaller
layout. Recompacting the saved cell (`recompact_full_adder_rcell`, 20-second
window solves, 8 workers):

1. First rerun (20 minutes): slices Y5, Z1 and Y11 came out (2x12x8), then
   block reduction reached 106 blocks. The compactor without the new options
   reached 107 in the same budget; both were still improving when time ran
   out.
2. Second rerun from 106 blocks: converged at **84 blocks** after 2021 s.
   Every gain came from Y windows. No Z window ever saved a block, but they
   only run once every Y window of a pass has failed, because each gain
   restarts the pass at the first window. The second round removed no
   further slice.

`repeat_rounds` now does these reruns inside one `compact` call. The result
is saved as `test/full-adder-exact-optimized-2x12x8.rcell` (replacing the
2x14x9 cell) and passes the same regression tests: all eight cases and every
settled input transition.

| Cell | Box | Volume | Blocks |
| --- | --- | ---: | ---: |
| Manual, right-hand inputs | 2x14x10 | 280 | 129 |
| Manual cell + automatic compaction | 2x10x10 | 200 | 99 |
| Generated, pipeline (2026-10-03) | 2x14x9 | 252 | 153 |
| Generated, recompacted twice | 2x12x8 | 192 | 84 |

### Larger circuits (2026-10-04)

`diagnose_construct_circuit` runs construction and compaction on small
circuits written as expressions (`CIRCUIT=mux2|half-adder|adder2|mux4|
full-adder`). It also prints the NOR netlist and the number of nets that must
cross the seam after each construction step: placed inputs and gates that a
later gate still reads, plus finished outputs.

The first runs exposed a netlist bug, now fixed. The expression parser builds
one n-ary node per operator chain, and `decompose_and`/`decompose_xor`
dropped every operand after the second. As a result, `a1^b1^c0` became
`a1^c0`, and the 4:1 mux lost all four data inputs, so its output was the
constant 1, which the placer proved unplaceable. `decompose_binops` now
splits such chains first (`nor_netlist_keeps_every_operand_of_long_chains`).

Results (height 10, windows 2..4, 60-second steps, 8 workers). The mux2 and
half-adder netlists use only two-operand chains, so the bug did not affect
them. The 2-bit adder also failed before the fix (22-gate netlist):

| Circuit | Gates | Max live nets | Result (width 2) |
|---------|------:|--------------:|------------------|
| mux2 | 7 | 3 | built in 9 s (2x14x10, 207 blocks); compacted to 2x7x4, 28 blocks, in 183 s |
| half adder | 7 | 3 | built in 13 s (2x14x10, 187 blocks); compacted to 2x7x7, 72 blocks, in 316 s |
| full adder (`nor9`) | 9 | 4 | built in 27–33 s, or after one restart (see above) |
| 4:1 mux | 20 | 6 | 5 attempts timed out at g31, g35 or g58 (stopped) |
| 2-bit adder | 26 | 8 (6 with smallest cone first and early outputs) | all 8 attempts failed; with the new order the best attempt finished `s0` and `c1` and placed 19 of 26 gates |

Every compacted cell passed RCELL verification. Construction grows the layout
along Y only, so every live net crosses every seam (these runs predate
`early_outputs`). A 2x10 cross-section carries the four live nets of the full
adder, but not six or more. The failing steps time out (Unknown) rather than
prove infeasibility. Width 3 made it worse for the mux: each step is larger,
and four attempts timed out at earlier gates (g24, g27, g56, g57). A greedy
gate order that minimizes live nets after each step lowers the 2-bit adder
from 8 to 6 but raises the mux from 6 to 8, so it was not adopted;
smallest-cone-first with early outputs (see above) reaches the same 6 for the
adder without hurting the mux, which has a single output. Circuits beyond
about four live nets should still be split into local cells by global
placement.

## Configuration

`ExactPlacerConfig` holds the problem (box, pins, fixed cells, observations,
`optimize`, model file and params). Constants that were measured rather than
derived live in `ExactPlacerConfig::tuning` (`ExactTuning`), with the measured
values as defaults: worker seed stride (7919), simulator cycle and event limits
per settle (256, 50 000), the torch-bound search state limit (200 000), and the
largest total objective weight (100 000). `ConstructionConfig` and
`CompactionConfig` pass a `tuning` through to every solve and also expose
their per-step `max_refinements` (8) and, for compaction, the block-reduction
`reduction_window` (3 slices). Two limits are structural, not tunable: at
most 6 inputs (functions are `u64` truth tables) and 64 signal classes for the
torch bound (class sets are `u64` masks).

`dsl.rs` sets every model param the placer derives from the config in one
place (`Prepared::params`: `rank_levels`, `stage_levels`, `max_blocks`,
`allow_unpowered_wires`, `min_torches`); `model_params` are applied after
them and override them.

## Running

```sh
cargo test --release --lib exact::
cargo test --release --lib diagnose_full_adder_construct_and_compact -- --ignored --nocapture
RECOMPACT_SOURCE=test/full-adder-right-inputs-2x14x10.rcell \
  cargo test --release --lib recompact_full_adder_rcell -- --ignored --nocapture
cargo test --release --lib diagnose_exact_full_adder -- --ignored --nocapture
CIRCUIT=mux2 cargo test --release --lib diagnose_construct_circuit -- --ignored --nocapture
# rsdsl versus the hand-written encoder
cargo test --release --lib measure_encoders -- --ignored --nocapture
ENCODER_CASE=xor SEEDS=4 cargo test --release --lib compare_encoder_portfolios -- --ignored --nocapture
PIPE_LEGACY=1 cargo test --release --lib diagnose_full_adder_construct_and_compact -- --ignored --nocapture
```

The harnesses print their knobs and write `.rcell`/`.nbt` files when given
`PIPE_WRITE` or `EXACT_FA_WRITE`. Generated RCELL files can be re-verified
with the `rcell` binary.
