# Further manual compaction: 2x13x9 full adder

Measured 2026-09-22. `test/full-adder-2x13x9.rcell` and its matching NBT contain
a verified full adder with three physical switches, ten probes, and two outputs.
The nine-NOR logic decomposition from the [height-10 layout](height10_full_adder.md)
is unchanged. No simulator semantics or production placer code changed.

| Fixture | Compiler box (thickness, width, height) | Box volume | Blocks |
| --- | --- | ---: | ---: |
| Previous | 2x17x10 | 340 | 128 |
| New | 2x13x9 | 234 | 93 |

The new count includes 25 automatic supports. Volume falls 31.2% and block count
falls 27.3%. The occupied box reaches every declared extent; this is not just
a smaller declaration around unchanged geometry. Both older fixtures remain
available. This is a manual feasibility result, not a minimum-size proof or an
automatic placer result.

## Direct component reads shrink the XNOR cores

The earlier core inserted dust between a switch and the input repeater, and
between the central wall torch and the two branch repeaters. A repeater can
read the immediately adjacent switch or torch directly when its input direction
matches. Removing those intermediate wire cells and moving the branch supports
closer reduces each core from seven Y columns to five.

The first core's central NOR is at `(1,2,1)`; its two branch NOR outputs are at
`(1,0,2)` and `(1,4,2)`. Its XNOR output is at `(1,2,5)`. The second core starts
at Y=8, Z=4 and produces sum at `(1,10,8)`. The routed input adapter retains its
isolating repeaters: the earlier backfeed regression still applies. Dust
removal is conditional on component orientation and fanout, not a blanket
replacement rule.

## A solid bridge can preserve clearance that dust would consume

Lowering the carry join initially cut the descending wire between the XNOR
cores. A wire at `(0,5,7)` requires a solid support at `(0,5,6)`, but that cell
must remain air for the descent from `(1,5,6)` to `(0,5,5)`.

The passing route instead uses:

```text
n1 relay torch (0,3,7)
  -> repeater (0,4,7)
  -> hard-powered solid (0,5,7)
  -> repeater (0,6,7)
  -> final NOR support (0,7,7)
  -> cout torch (0,7,8)
```

The bridge block itself does not need a support below it. The air at `(0,5,6)`
therefore survives, and the lower XNOR wire remains connected. A regression
replaces only the bridge solid with dust: automatic support then occupies the
headroom, the source XNOR still passes, and `second_input` becomes always off.
Geometric non-overlap alone would accept this broken candidate.

For a future placer, consider `repeater -> solid -> repeater` as an alternate
transport recipe when a lower wire needs clearance. Rank the full footprint,
including automatic supports and required air, and validate the existing lower
net after insertion. It is not universally better than dust.

## Validation

`cargo test --release --locked --lib physical_cell::` passed 22 tests. The new
fixture passes all eight fresh input cases and all 64 ordered settled input
transitions, with all ten probes and both outputs checked. Transition tests
reject burned-out torches and wait 61 idle cycles between input changes. All
repeaters retain delay 1. The bridge-clearance negative regression also passes
by detecting the intended failure.

```sh
cargo run --release --locked --bin rcell -- \
  test/full-adder-2x13x9.rcell test/full-adder-2x13x9.nbt
cargo test --release --locked --lib physical_cell::
```

The viewer renders the permuted NBT axes as **13x9x2**. Open with
`?example=full-adder-2x13x9.nbt`; the sidecar metadata identifies sum and cout.
These checks establish settled behavior in the repository simulator, not a
clock-rate guarantee, glitch-free operation, Minecraft equivalence, or safe
adjacency to arbitrary other cells.
