# Current RCELL full-adder diagnosis

Measured on 2026-09-20 from `4cb3451` using the release `rcell` binary and
`test/rcell/archive/full-adder-2x20x20-disconnected.rcell`. This is the manual 2x20x20 experiment, separate
from the automatic 2x10x10 search in `local_full_adder_diagnostic.md`.

The failed cell has since been archived. New experiments start from the verified
`test/full-adder-baseline.rcell`; see `rcell_full_adder_baseline.md`. The results
below describe the archived circuit only.

## Result

All 16 declared probes and `sum` match their expected truth tables. Only
`cout` fails: it is always on, failing four of eight cases. Input influence
analysis reports no influence from `a`, `b`, or `cin` on `cout`.
Syntax, support, orientation, occupancy, and bounds checks pass; no keepout
constraint is declared. These are fresh, settled input-case checks, not a
proof of arbitrary input transitions, timing behavior, or Minecraft fidelity.

Signatures below are in mask order 0..7, with `a` as bit 0, `b` as bit 1,
and `cin` as bit 2:

| Observation | Expected | Actual |
| --- | --- | --- |
| a_lift | 01010101 | 01010101 |
| b_lift | 00110011 | 00110011 |
| cin_lift, cin_keep, cin_final | 00001111 | 00001111 |
| n1, n1_keep, n1_final | 10001000 | 10001000 |
| n2 | 00100010 | 00100010 |
| n3 | 01000100 | 01000100 |
| n4, n4_keep, n4_final | 10011001 | 10011001 |
| n5 | 01100000 | 01100000 |
| n6 | 00000110 | 00000110 |
| n7 | 10010000 | 10010000 |
| sum | 01101001 | 01101001 |
| cout | 00010111 | 11111111 |

The intended final gate is `cout = ~(n1 | n5)`, where `n1 = ~(a | b)` and
`n5 = ~((a XNOR b) | cin)`. Evaluating this identity for all eight cases
matches majority carry. Both source signals already exist and are correct.

## Two missing carry connections

Coordinates use compiler `(x,y,z)`, with `z` as height. State JSON was collected
for all eight cases. This identifies disconnected inputs, not merely a slow
or insufficiently deep search.

### n5 transfer tower

The existing tower's support at `(0,11,14)` has no candidate power sources
and remains unpowered in all cases. The adjacent dust at `(1,11,14)` correctly
carries `n5`, but its connection mask is 12 (the Y-axis line); it does not
power the neighboring support in the X direction. Adjacency alone does not
constitute a signal connection.

Consequently the tower torch at `(0,11,15)` is always on, the upper support
at `(0,11,16)` is powered, and the next torch at `(0,11,17)` is always off.
The dust at `(0,12,17)` and repeaters at `(0,13,17)`, `(0,14,17)`, and
`(0,15,17)` therefore stay off. This branch does not carry `n5` to the final
gate.

### n1 return line

The source probe `n1_final` at `(1,19,14)` is correct. However, the final
branch's dust at `(0,18,17)` has no candidate power sources in any of the
eight dumps. Its repeater at `(0,17,17)` stays off. The source and return
line are not connected.

The final support `(0,16,17)` receives power only from the two off repeaters
at `(0,15,17)` and `(0,17,17)`. It is always unpowered, so the output torch
at `(0,16,18)` is always on. The observed failure is explained without
changing simulator semantics or the Boolean circuit.

## Recommended continuation

Preserve the current cell as the baseline. Keep the passing logic and sum
path fixed initially, and redesign the carry transport and final NOR region
around explicit `n1` and `n5` handoff points. There is no evidence yet that
the whole cell needs replacing or that 2x20x20 is infeasible.

1. Add observations at the two carry branch inputs and the final support.
   Expect `n5`, `n1`, and `n1 | n5`, respectively, so disconnection is caught
   before the final output.
2. Plan both routes together, including support blocks, dust connection
   directions, repeater orientation, headroom, and isolation from the nearby
   `n6`/`n7` sum routes. Merely adding dust beside the n5 tower is not a
   demonstrated fix.
3. Use the verified primitive patterns, make bounded changes in a separate
   candidate, and require all existing probe signatures to remain unchanged.
4. If the reserved carry region cannot accommodate those connections, move
   the final carry gate or expand the experimental bounds to establish a
   working reference before compacting it. Document that feasibility evidence
   before considering a complete floorplan redesign.
5. After all eight fresh cases pass, test input transitions with settling
   and inspect torch burnout/feedback before accepting the cell.

The inverter and all eight checked-in `*-primitive.rcell` fixtures were also
run in release mode: all passed (30 truth-table cases in total).

`cargo test --release --locked --lib world::simulator::` passed 32 tests,
with four existing ignored tests. On this checkout, compiling the test binary
first required restoring `counter.v` and `d-flip-flop.v` from their tracked
`test/*.rsnap` ZIP archives into the corresponding ignored `test/*.snapshot/`
directories: `src/ir/debug.rs` includes those files at compile time even when
only simulator tests are selected. No circuit fixture was overwritten.

## Reproduction

Run from the repository root. Use temporary NBT outputs to preserve fixtures;
the CLI exports NBT even when verification fails.

```sh
cargo build --release --locked --bin rcell
./target/release/rcell test/rcell/archive/full-adder-2x20x20-disconnected.rcell /tmp/full-adder-diagnostic.nbt
./target/release/rcell test/rcell/archive/full-adder-2x20x20-disconnected.rcell /tmp/full-adder-diagnostic.nbt \
  --case 'a=1,b=0,cin=0' --explain 0,11,14 --state-json /tmp/full-adder-case-1.json
./target/release/rcell test/rcell/archive/full-adder-2x20x20-disconnected.rcell /tmp/full-adder-diagnostic.nbt \
  --case 'a=0,b=0,cin=0' --explain 0,18,17 --state-json /tmp/full-adder-case-0.json
```

The full verification and case 0 return failure as expected. Case 1 happens
to have the correct `cout=1`, despite the disconnected circuit; its dump is
useful because the upstream `n5` signal is on.
