# Full-adder operand inputs on one boundary

The 2026-09-22 reference screenshot shows compiler **Y-min at the right edge**.
The user requested both operand switches there, then required sum at the
opposite end. Carry-in and carry-out placement were free.
The operand called X maps to RCELL `a`, and operand Y maps to RCELL `b`.
These signal names are distinct from the compiler coordinate axes.

`test/full-adder-right-inputs-2x14x10.rcell` implements this physical variant:

| Signal | Compiler position `(x,y,z)` | Reference view |
| --- | --- | --- |
| X / a | `(0,0,3)` | Upper switch at right boundary |
| Y / b | `(0,0,1)` | Lower switch at right boundary |
| cin | `(0,13,5)` | Other end; not grouped with operands |
| sum | `(0,13,8)` | Opposite boundary, repeater emitting toward Y-positive |
| cout | `(1,7,9)` | Output |

Both operand switches now have `y=0`. Each logical input has exactly one
physical switch; no tied duplicate contacts or simulated input aliases were
introduced. This groups the switches by boundary, not by their attachment
orientation: both attach toward X-positive, as in the original operand bank.
It does not yet specify an external wire-pin/neighbor keepout contract.

The operand B signal travels along a new bottom channel and enters its two
original NOR consumers through separate isolating repeaters. The first logic
cone is raised two levels; the second cone moves one column and one level.
Carry-in stays with the second cone. Repacking the upper carry join and using
a wall carry-output torch keeps the height at 10. A solid at `(1,9,9)` blocks an
unwanted diagonal dust connection between an XNOR branch and the carry route.

The sum boundary connection moves the sum-core torch to `(0,11,8)`, then uses
dust at `(0,12,8)` and an outward-facing repeater at `(0,13,8)`. The carry `n5`
relay moves to `(0,10,7)` to free that lane; carry-out stays at `(1,7,9)`.
This adds four blocks to the earlier 125-block input-boundary-only variant,
without enlarging its box. The `sum_core` probe verifies the source separately
from the public output repeater.

The cost of these input and output locations is a **2x14x10** box and **129 blocks**,
including 41 automatic supports, compared with 2x13x9 and 93 blocks for the
unconstrained input layout. Both fixtures remain available for comparison.
The viewer permutes the axes and displays this variant as 14x10x2; the side that
looks right depends on camera rotation, but the input boundary is fixed.

Validation: all eight fresh truth-table cases and all 64 ordered settled input
transitions pass, checking every probe and both outputs and rejecting torch
burnout. A regression also asserts the exact operand switch locations,
single-contact counts, and the sum repeater's opposite-boundary position and
outward orientation. The physical-cell release suite passes 24 tests. No
production placer or simulator behavior changed; these are simulator checks
under the existing settled manual-input contract.

```sh
cargo run --release --locked --bin rcell -- \
  test/full-adder-right-inputs-2x14x10.rcell \
  test/full-adder-right-inputs-2x14x10.nbt
cargo test --release --locked --lib physical_cell::
```

Viewer example: `?example=full-adder-right-inputs-2x14x10.nbt`.
