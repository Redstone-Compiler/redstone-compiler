# Verified 2x20x20 full adder and transferable placement rules

Measured 2026-09-20 against the repository simulator, without changing its
power, delay, initialization, or burnout semantics.

Follow-up: [the height-10 experiment](height10_full_adder.md) produces a new
2x17x10, 128-block layout using lower XNOR cores. This 2x20x20 fixture remains
the earlier verified reference and its failure-mode regressions are retained.

## Result and provenance

`test/full-adder-2x20x20.rcell` and the matching NBT now contain a working
**2x20x20** full adder: three physical input switches, 19 probes, two outputs,
and 259 blocks including 74 automatic supports. Coordinates use `z` as height.
The source uses two annotated YZ planes rather than dozens of XY sections.

The 48x22x4 baseline remains an independent, readable functional reference.
This compact result is **not** an automatic shrink of that baseline and is
not a successful run of the existing local placer. It reuses the passing
two-wide sum core from the archived failed layout, preserves its logical
signals, and replaces the failed carry transport with reserved vertical
relays and local recomputation. The archived original remains unchanged.

Validation includes:

- All eight fresh input cases, checking every probe and both outputs.
- All 64 ordered input-state pairs, checking every observation and rejecting
  any burned-out torch after the transition. Each pair has a fresh initialized
  simulator and 61 idle cycles before its second input state.
- Structural support, orientation, occupancy, and box checks during building.
- A regression showing the output burns out when only the final repeater
  delay is changed from 2 back to 1.
- Variable-height relay tests with one through six inverter pairs, checking
  every intermediate polarity and both primary-input values.

These establish a settled manual-input contract in this simulator. They do
not establish a maximum clock rate, absence of transient glitches, physical
Minecraft equivalence, or a fully routable public pin interface for composition.
The RCELL has no declared external keepout contract.

## 1. Reserve long-lived signal routes before filling the sum cone

The previous `n1` route zigzagged around the circuit and stopped short of
carry. The new route reserves `(x=1,y=19)` vertically, starting from the
powered support at `z=6`. Alternating torches and solid blocks carry `n1`
with the original polarity at `z=9,13,17`. A powered block at `z=18` exposes
the signal to the final wire at `z=19`.

The carry copy of `cin` starts with a repeater at `(0,17,11)`, a support at
`(0,18,11)`, and a wall torch at `(1,18,11)`. Its reserved vertical lane at
`(x=1,y=18)` delivers positive `cin` at `z=17`.

Transferable recipe: an attached torch inverts a powered support; its top
solid block can drive another attached torch. Two inversions preserve the
signal and add a regular vertical step. Track **signal identity, polarity,
input support, output contact, and delay** at every relay. A narrow column
is not electrically isolated merely because it fits geometrically.

For a placer, identify signals whose consumers occur much later, reserve
their relay corridor and electrical clearance, then place short-lived local
logic. Carry consumers and branch join sites should be planned together.

## 2. Local recomputation can cost less than a new fanout route

The original `n5` fanout is congested. Extending it can disturb either the
XOR cone or `cin`. Instead, reconstruct it near carry:

```text
n5 = NOR(n4, cin)
n7 = NOR(cin, n5)
carry_n5 = NOR(n7, cin) = n5
cout = NOR(n1, carry_n5)
```

This identity holds because `n5 AND cin = 0`. In general,
`NOR(NOR(A,C),C) = A AND NOT C`; recovering `A` is valid only when
`A AND C = 0` has been established. It is not an unconditional NOR rewrite.

The extra NOR uses support `(0,15,17)` and output torch `(0,15,18)`.
Its `n7` input arrives from `(0,14,17)`; the reserved `cin` relay supplies
the opposite input at `(0,16,17)`. The final output torch is `(0,18,19)`.

This layout uses ten logical NOR operations instead of the reference's nine,
plus transport inverters. Fewer Boolean gates do not necessarily mean a
smaller physical layout. Compare estimated routing/clearance cost against
duplication and relay cost when choosing a mapped logic variant.

Such rewrites belong before the placement-ready Routable boundary, or in an
explicit alternative target-mapping candidate. Do not silently rewrite the
graph inside placement; preserve source provenance and prove equivalence.

## 3. The occupied-block box is smaller than the electrical exclusion region

Several geometrically legal attempts failed electrically:

- A dust stair crossing beside the hard-powered `n7` pickup block at
  `(1,14,16)` picked up `n7`, fed it back into `cin`, and caused burnout.
- Adding a repeater at `(1,15,11)` added its support at `(1,15,10)`, blocking
  headroom used by the existing rising dust route below. `cin_keep` then
  lost all dependence on `cin` despite the new route's legal supports.
- Tapping the original `n5` pickup with an adjacent inverter and top-powered
  support fed the inverted signal back into the original dust path.

Candidate legality should therefore account for support placement, the
headroom of every rising wire edge, strong-power contacts, torch emission,
and neighboring dust connectivity. Treat these as explicit recipe footprint
and clearance data. Block overlap checks alone cannot reject these failures.

When a route changes, simulate all still-live signals, not only its endpoint.
Freeze a passing prefix and replay alternatives from the same state instead
of attributing results from unrelated search frontiers to one routing change.

## 4. Boolean correctness does not imply safe reconvergent timing

After connectivity was repaired, all upstream signatures passed, but the
final `cout` torch burned out. Its trace showed repeated support changes
while the long relay chains and the recomputed carry branch initialized.
The failure was distinct from the old disconnected, always-on output.

Changing only the final repeater at `(1,17,19)` from delay 1 to delay 2 made
all eight cases pass; delays 3 and 4 also passed the fresh-case check. Delay 2
was retained and passed the 64 settled transitions with no burned-out torch.
The negative regression records the delay-1 failure for `(a,b,cin)=(1,1,0)`.

For a placer, retain a bounded portfolio of delay choices at reconvergent
joins, then run initialization and transition verification. Do not treat
delay 2 as a universal fix or relax burnout semantics to accept a candidate.

## Next compiler work

The existing placer has not been changed by this experiment. A bounded next
implementation would expose vertical relay recipes with polarity/clearance
contracts, reserve routes for long-lived nets, and compare an equivalent
local-recomputation candidate with the original fanout candidate. Rank and
verify complete candidates with initialization and transition checks before
accepting compactness improvements. This fixture is a feasibility witness
and regression target, not evidence of a general automatic solver yet.

## Reproduction

```sh
cargo run --release --locked --bin rcell -- \
  test/full-adder-2x20x20.rcell /tmp/full-adder-compact.nbt
cargo test --release --locked --lib physical_cell::
```
