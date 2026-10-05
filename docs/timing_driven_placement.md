# Timing: critical paths in redstone ticks (2026-10-05)

The exact placer can bound and shorten how long a layout takes to settle.
A torch or a repeater (the placer's repeaters have delay 1) takes one
redstone tick; dust and blocks pass a change on in the same tick. So the
delay of a path is its torches plus its repeaters. Some of those torches are
the logic. The rest are wasted: repeaters that only route, and inverter
pairs that cancel out.

Code:

- `exact/timing.rs`: static timing analysis and the logic-depth bound.
- The rules "타이밍" in `exact_placer.rsdsl`.
- `ExactPlacerConfig::{timing, output_delays}` and
  `ExactLocalPlacer::place_minimizing_delay` in `exact/mod.rs`.
- `CompactionConfig::{timing, delay_outputs, delay_window, max_delay_window}`
  in `exact/compact.rs`.
- `ConstructionConfig::step_timing` in `exact/construct.rs`.

## Static timing (`timing.rs`)

A cell fed by several sources settles only after the slowest one. A cell's
arrival is therefore the longest path to it over every power relation, as
in static timing analysis of a gate netlist. A relation counts even when no
input case drives the cell through it.

- The relations are the model's `Feeds`, mirrored in Rust, plus each torch's
  support.
- `analyze` starts every cell at tick 0, as the model's `Stage` does.
- `analyze_from` starts one input switch and gives the delay from that input
  to each output.
- `Timing::path` walks back along the slowest path.

`ExactPlacement::delays` holds the arrival at every output of every
placement. `ExactLayout::timing()` analyzes any layout, chains included.

`depth_bounds` is the matching lower bound from the logic alone: the fewest
torches between the inputs and each output.

- Dust and blocks OR any classes of the vocabulary into another class for
  free.
- A torch complements one class.
- The full adder's bound is 5 ticks for both outputs. The carry tile's
  `ncout` bound is 2: the carry-in torch, then `NOR(cin, a)`, ORed onto
  blocks.

`static_timing_matches_the_model` checks Rust against the model on three
fixtures. With every cell fixed and `timing` on, the model accepts each
fixture at exactly the analyzed delays and rejects it with any one output a
tick sooner.

## Model

With `param timing = true`, `Stage` becomes an arrival time:

- every `Feeds` relation orders its ends, strictly into a repeater;
- every torch, given cells included, comes strictly after its support;
- an observed output `o` with an `output_delay(o, d)` fact has
  `Stage <= d`.

Without timing, `Stage` only orders the contributions a layout chose to
justify its values. Given cells are included in the timing rules, so arrival
times run through the frozen slices of construction and compaction windows.
Every problem without `timing` grounds the same CNF as before
(`print_cnf_hashes`).

`Stage` only has to hold arrival times, so timing solves size `stage_levels`
to the longest path plus a little. That matters for speed. XOR in 2x6x4 with
the output in at most 5 ticks:

| `stage_levels` | Result |
| --- | --- |
| 24 | no answer in 90 s |
| 8 | 7.3 s |
| 6 | 3.3 s |

The same cell without `timing` (24 levels) took 28 s under the same load,
and budgets of 6 and 4 ticks with 24 levels took 56 s and 51 s
(`diagnose_delay_bound`).

## Using it

**One solve.** `place_minimizing_delay` finds a layout, then asks for the
slowest output a tick sooner until that is infeasible (proven) or time runs
out. An inverter carried the length of a 1x9x3 box:

- unconstrained: 4 ticks (repeaters);
- minimized: 1 tick (dust), proven
  (`minimizing_delay_shortens_the_critical_path`).

**Compaction.** With `CompactionConfig::timing`, every window solve keeps
each output's delay at most what it was, so slice removal and block
reduction never slow the layout down. Each round also ends by shortening
paths:

- one output at a time, in `delay_outputs` order (or slowest first);
- a window on the output's slowest path is re-solved with that output a tick
  sooner;
- the window widens from `delay_window` to `max_delay_window` when a full
  pass gains nothing.

`timing_compaction_keeps_and_shortens_delays` takes a NOR in a 1x9x3 box
whose layout routes through repeaters (3 ticks) to 1 tick.

Shortening comes last in a round because a path window that cannot shorten
costs a whole attempt. The slowest path of a constructed full adder runs
through nearly every window. With shortening first, its 600 s of compaction
went to such windows: nothing was removed, reduced, or shortened.

**Construction.** With `ConstructionConfig::step_timing`, each step, once
placed, is re-solved with its observed nets sooner.

- The target for each net is its lower bound plus a slack. The bound is the
  latest input arrival, plus a tick unless the net is an OR a block can
  carry.
- The slack goes 0, 1, 2, 4, ... ticks, one solve of `step_timing` each,
  until a solve succeeds.
- The step's block minimization then keeps the delays reached.

## Results

Full adder (`full_adder_graph("nor9")`, 2 wide, height 10, seed 1,
`diagnose_construct_circuit`). The critical path is the slower of `cout` and
`s`; the logic bound is 5 ticks for both. Construction is not deterministic
with parallel workers, and `step_timing` solves were 10 s each under load
from other runs.

| Run | Constructed | Compacted |
| --- | --- | --- |
| no timing, 5 workers, 1200 s | 2x18x10, 202 blocks, cout 18, s 23 (67 s) | 2x11x10, 83 blocks, cout 11, s 15 |
| `step_timing` 10 s + `timing`, 5 workers, 1200 s | 2x18x10, 244 blocks, cout 11, s 17 (252 s) | 2x14x10, 113 blocks, cout 8, s 9 |
| no timing, 8 workers, 600 s | 2x18x10, 140 blocks, cout 17, s 15 (63 s) | 2x11x10, 94 blocks, cout 11, s 14 |
| `step_timing` 10 s + `timing`, 8 workers, 600 s | 2x18x10, 264 blocks, cout 10, s 11 (171 s) | 2x13x9, 162 blocks, cout 8, s 11 (still shrinking) |
| no timing, construction only | 2x18x10, 187 blocks, cout 16, s 22 (63 s) | |
| `step_timing` 10 s, construction only | 2x18x10, 211 blocks, cout 6, s 8 (146 s) | |

With the full 1200 s, timing-first placement settles 40% sooner (9 ticks
against 15) with 36% more blocks.

None of these compactions shortened a path in its shortening phase. The
delays fell during slice removal and block reduction instead: each change
may keep or lower every delay, and each accepted layout lowers the bounds
for the next. A removed slice takes its repeaters with it (`s` went from 17
to 9).

Fixtures (bounds 5 and 5):

| Fixture | Delays |
| --- | --- |
| `full-adder-exact-optimized-2x8x8` | cout 10, s 8 |
| `full-adder-exact-2x13x7` (repaired 2026-10-05) | cout 13, s 14 |

The carry tile (bounds: `ncout` 2, `s` 5). In a chain, only the path from
the carry in matters bit after bit (`cout` of an n-bit chain is about n
times it), so the table gives that path per tile, and the chains' `cout`:

| Tile | `ncout` / `s` (all inputs) | Per-bit carry (`ncin` to `ncout`) | Chain `cout`: 4 / 8 / 16 bits |
| --- | --- | --- | --- |
| fixture `adder-carry-tile-2x16x10`, 164 blocks | 13 / 20 | 11 | 47 / 91 / 179 |
| fixture, timing compaction, shortening first, 3-slice windows, 1500 s: 2x16x9, 153 blocks | 11 / 15 | 10 | 43 / 83 / 163 |
| fixture, timing compaction, windows up to 5 slices, 2400 s: 2x15x10, 159 blocks | 11 / 18 | 11 | 47 / 91 / 179 |
| constructed with `step_timing` 10 s, no compaction: 2x27x10, 363 blocks | 14 / 21 | 14 | 62 / 118 / 230 |

Every chain passed the checks of `synthesize_carry_adder`: every case up to
4 bits; sampled cases and a 200-step random walk at 8 and 16 bits.

The model bounds an output over all inputs at once. In the tile, the
operand path to `ncout` (13 ticks) was longer than the carry path (11). The
shortening phase spent its windows on the operand path, which a chain pays
for only once. A bound on the path from one input, a second arrival time
started at that input, would aim at the carry alone. That is not built.

## Where the ticks go in the carry tile

The carry tile's per-bit carry was 11 ticks against a bound of 2. Most of the
extra time is in construction's corridor. The carry-out cell shares a column
with the carry switch, so the carry has to come back along +Y in the tile's
last X slice. Every cell of a 2-wide tile is on a seam: dust in the corridor
sits across the seam from the tile's other slice, and the seam rules forbid
it beside dust, torches, switches, and repeaters there. Construction had
filled that row with other logic, so the carry came back on repeaters and an
inverter pair (5 ticks in the fixture's corridor alone).

`step_timing` did not shorten the carry-out step: `ncout` was 19 ticks
before and after it, and 14 in the finished tile.

Keeping the row beside the corridor empty until the carry-out step would
let the carry come back on dust. That was tried with `step_timing`, and the
carry-out step then found no layout:

- seed 1: no answer in 180 s in windows of 2 and 3; window 4 hit the attempt
  deadline;
- seed 2: proven infeasible in a window of 2 (121 s); no answer in 180 s in
  a window of 3.

The row is routing space the other logic needs, so the option was removed.

## Limits

- Static timing counts every path, including false paths no input
  transition exercises, so it can overstate the real worst case. It never
  understates it.
- Compaction shortens paths only inside its windows. Once the slowest path
  runs through several windows that each need to change, it stops.
- With `step_timing`, a step solved for timing once left a layout on which
  the next step (the carry entry) was infeasible within 0.2 s for every
  window, in three seeds in a row, under heavy load. It did not happen again
  in three later runs. Construction restarts with a new seed when it does.
