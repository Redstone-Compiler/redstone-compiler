# Sequential cells with the exact placer (2026-10-10)

The exact placer builds compact combinational cells (the 2x8x8 full adder,
`exact_local_placer.md`). This adds latches and flip-flops: a netlist with
state, a model that can place a latch's loop, and a simulator check for
sequential behavior. Until now only the heuristic local placer built latches
(`test/rs-latch.nbt`, 12x9x5; `test/d-latch.nbt`, 14x10x10). The repository
also has a hand-made set-reset latch macro (8x6x3, `sequential/layout.rs`).

## What kept latches out

- `NorNetlist` rejected cycles.
- The model orders `Stage` along every contribution and puts a torch above its
  support. A torch whose output returns to its own support, which is a latch,
  was impossible.
- After every model, the lazy acyclicity check (`acyclic.rs`) cut any active
  path from a torch back to its support, even with stage levels on.
- Verification was combinational. It also started each case from every torch
  lit, and a set-reset latch started that way races its two torches until
  they burn out.

## Netlists with state

`NorNetlist::state` names the state nets: every cycle passes through one, and
a gate that reads one reads the stored value. In text form:

```text
state(q); out(q); q=NOR(r,nq); nq=NOR(s,q)
```

- Cases cover the inputs and then the stored state bits
  (`NorNetlist::case_bits`).
- In `net_values`, a state net's readers read its stored bit. The state net's
  own value is what its gate computes, which is the next value.
- `valid_cases` are the settled cases, where the next value equals the stored
  one.
- `next_state` iterates from a stored state under new inputs until the value
  holds.

`sequential_netlist` in `exact/tests.rs` holds the cells measured below.

## The model: cut each loop at its state net

A state net gets two classes. The torch carries the **next** value (`q`). The
cells past the cut carry the **stored** value (`q_stored`), and that is what
the rest of the cell reads. The fact `cut(n, m)` pairs them:

- A relation from a cell that carries `n` into dust or a repeater that carries
  `m` is the cut (`Cut[s, t]`).
- The cut alone is exempt from soundness: in the settled cases the two values
  agree and the loop is closed.
- The cut is also exempt from the stage order, because the latch's loop runs
  through it.
- Every other relation follows the combinational rules in every case of
  inputs × stored state.
- `acyclic.rs` treats the sink of an active cut as a driver, and its feedback
  search does not cross a cut.

So the layout has to compute the netlist's next value from **every** stored
value, including the cases a latch only passes through. An input cannot be
left unconnected.

The first version used only the settled cases as the case domain. The
solver then found "latches" that only hold, such as a second torch
`NOT(q)` instead of `NOR(s, q)`. Every settled case was consistent, but `s`
never reached the loop. The simulator rejected thousands of them, all with
`output q wrong after 0 -> 110` (set from 0). With the cut, the set-reset
latch had no rejection at all.

## Verification

`verify_sequential` (`exact/verify.rs`) runs in two parts:

- **Settled cases.** Each settled case is set up as it should stand: its
  switches, and every torch lit as that case has it (`start_in_case`, then
  `Simulator::from_preserving_torch_states_*`). It is left to settle, and
  every labelled cell is checked.
- **Input changes.** From every settled case, each input is flipped alone
  (`SequentialPlan`), and the outputs and torch burnout are checked against
  `next_state`. Inputs change one at a time, as levers do: releasing `s` and
  `r` together races a set-reset latch in the game too.

An exported sequential world stores its first settled case with the inputs
off (`settled_sequential_world`). RCELL has no expectations over time, so a
sequential cell's RCELL has none, and the fixtures are checked from their NBT
instead (`run_latch_fixture`). Construction still refuses state: it places
one gate per step and cannot cut a loop yet.

## Results

`explore_sequential_cells`, 8 workers unless noted, fewest blocks (levers not
counted):

| Cell | Torches | Box | Result |
| --- | --- | --- | --- |
| set-reset latch | 2 | 2x3x3 | **8 blocks, proven fewest**, 0 rejections, 2.7 s |
| set-reset latch | 2 | 2x4x3, 3x3x3, 2x4x4, 3x4x3, 2x5x4 | 8 blocks, proven fewest in each |
| D latch, set/reset gating | 6 | 2x3x4, 2x4x3, 3x3x3 | no layout (proven, 4 workers) |
| D latch, set/reset gating | 6 | 2x4x4 | 22 blocks, not proven in 600 s, 0 rejections |
| D latch as a multiplexer | 4 | 2x3x3, 2x4x3 | no layout (proven) |
| D latch as a multiplexer | 4 | 2x3x4 | **16 blocks, proven fewest**, 35 rejections, 77-111 s |
| D latch as a multiplexer | 4 | 2x4x4 | 15 blocks, not proven in 300 s |
| master-slave flip-flop | 10 | 2x6x4, 2x6x5 | nothing found in 900 s each (4 workers) |
| flip-flop from two multiplexer latches | 6 | 2x4x4, 2x5x4, 2x6x4 | nothing found in 600 s each (4 workers) |

- **The set-reset latch** is two torches on two blocks. Each block carries one
  lever, and two repeaters cross over, each from one torch into the other
  torch's block. Its 2x3x3 box is 18 cells, against the 8x6x3 hand macro
  and the 12x9x5 heuristic layout.
- **The D latch** is smaller as a multiplexer, `q = en ? d : q =
  NOR(NOR(d, ~en), NOR(q, en))`, with four torches instead of six. It
  fits 2x3x4 (24 cells) against the 14x10x10 heuristic layout. The
  simulator rejected 35 layouts on the way. They are the static hazard of the
  multiplexer form, where `q` can drop for a tick while `en` changes. The
  check over input changes catches it.
- **Flip-flops** are too large for one solve, like the full adder. They need
  construction that can place a latch's loop in one step, or two latch cells
  composed.

Fixtures, with tests that paste the NBT and pull the levers through set,
hold, and reset (`exact_rs_latch_fixture_sets_holds_and_resets`,
`exact_d_latch_fixture_follows_and_holds`):

- `test/rs-latch-exact-2x3x3.{rcell,nbt,outputs.json}`
- `test/d-latch-exact-2x3x4.{rcell,nbt,outputs.json}`

Both are in the viewer's examples.

## Next

- **Flip-flops through construction.** Place each latch, its loop with the cut,
  as one step (the state net and its readers together), then the gating
  around it.
- **Synthesis with state** (`nor_synthesis.md`). Search latch and flip-flop
  netlists by next-state function, and screen them for hazards.
- **Repeater locking.** A repeater powered from the side holds its output,
  which is Minecraft's own D latch, but the model forbids it. Allowing it as
  a primitive could shrink D latches and flip-flops a lot.
- **Sequential RCELL.** Expectations over input sequences, so RCELL can check
  latches like any other cell.

## Running

```sh
cargo test --release --lib sequential_netlists_settle_like_latches
cargo test --release --lib rs_latch_places_in_eight_blocks
cargo test --release --lib latch_fixture
SEQ_CIRCUIT=d-latch-mux SEQ_DIMS=2x3x4 SEQ_SECONDS=300 SEQ_WRITE=target/d-latch \
  cargo test --release --lib explore_sequential_cells -- --ignored --nocapture
```

`explore_sequential_cells` takes the following knobs:

- `SEQ_CIRCUIT`: `rs-latch`, `rs-latch-both`, `d-latch`, `d-latch-mux`, `dff`,
  `dff-mux`; or `SEQ_NETLIST=<text>` for any other netlist.
- `SEQ_DIMS`: boxes to try in turn; `SEQ_ALL=1` tries every one.
- `SEQ_SECONDS`, `SEQ_WORKERS`, `SEQ_OPTIMIZE=0`, `SEQ_REFINEMENTS`.
- `SEQ_FIX`: fixed cells, as hints.
- `SEQ_MODEL_FILE`: another model.
- `SEQ_WRITE`: where to write the layout.

`model_accepts_hand_rs_latch` checks that the model accepts the hand-made
latch, with levers added, and that the full pipeline verifies it.
