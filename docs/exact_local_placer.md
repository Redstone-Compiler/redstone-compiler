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
   Y = 0. Every net still needed later must be observable on the window's
   last slice. The last step checks the real outputs, for example the sum on
   the Y-max face.
2. **Compaction.** One Y slice or Z layer is removed and everything beyond it
   shifted back. Only the cells next to the seam are re-solved; all others
   stay fixed. Accepted repairs are simulator-verified, and the emptiest
   slices are tried first. When no seam can be repaired, the window radius
   grows up to `max_window_radius`.
3. **Block minimization.** Optionally, a three-slice window slides along Y and
   is re-solved with a global bound of one block fewer.

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

## Running

```sh
cargo test --release --lib exact::
cargo test --release --lib diagnose_full_adder_construct_and_compact -- --ignored --nocapture
RECOMPACT_SOURCE=test/full-adder-right-inputs-2x14x10.rcell \
  cargo test --release --lib recompact_full_adder_rcell -- --ignored --nocapture
cargo test --release --lib diagnose_exact_full_adder -- --ignored --nocapture
# rsdsl versus the hand-written encoder
cargo test --release --lib measure_encoders -- --ignored --nocapture
ENCODER_CASE=xor SEEDS=4 cargo test --release --lib compare_encoder_portfolios -- --ignored --nocapture
PIPE_LEGACY=1 cargo test --release --lib diagnose_full_adder_construct_and_compact -- --ignored --nocapture
```

The harnesses print their knobs and write `.rcell`/`.nbt` files when given
`PIPE_WRITE` or `EXACT_FA_WRITE`. Generated RCELL files can be re-verified
with the `rcell` binary.
