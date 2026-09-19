# Physical Cell Lab

`rcell` is a small, text-editable laboratory for exact local redstone layouts.
It is deliberately not part of logical or Routable RCIR lowering. The tool
depends on the compiler's existing physical model, while compiler passes do not
depend on the laboratory:

```text
.rcell source
    -> PhysicalCellDocument
    -> PhysicalCellBuild
    -> World3D
    -> Simulator / NBT
```

The implementation is split by responsibility:

- `src/physical_cell/syntax.rs`: stable typed document model
- `src/physical_cell/parser.rs`: source text to document
- `src/physical_cell/emit.rs`: canonical document text
- `src/physical_cell/build.rs`: glyph expansion and physical validation
- `src/physical_cell/verify.rs`: truth-table verification through `Simulator`
- `src/bin/rcell.rs`: file, watch, verification, and NBT CLI

## Coordinate system and planes

The compiler uses `(x, y, z)`, with `z` as height. A plane names its two visible
axes and fixes the remaining axis:

```text
plane yz at x=0 { ... }  // vertical side view
plane xz at y=0 { ... }  // vertical side view
plane xy at z=0 { ... }  // horizontal floor view
```

The first named axis is the string's column axis. The second named axis is the
row coordinate. Therefore a `yz` plane uses `z=N` rows whose characters run
from `y=0` upward:

```text
plane yz at x=0 {
  z=2 "..r.";
  z=1 ".##.";
}
```

Unspecified rows and `.` cells are air. Intersecting planes may repeat the same
glyph at a position but may not assign conflicting glyphs.

### Floorplan vocabulary

Manual cells use the following terms when a layout has a preferred signal-flow
direction:

- **flow axis**: the preferred forward direction, such as `+Y`
- **input bank**: the boundary where externally controlled inputs are grouped
- **output bank**: the opposite boundary where observable outputs are grouped
- **logic field**: the space between the two banks
- **monotonic routing**: routing that does not move backward along the flow axis
  unless a local gate fold or crossover requires it

For example, a `+Y` feed-forward floorplan places its input bank at `Y-min`, its
output bank at `Y-max`, and primarily uses `Z` for isolated lanes and local
folding. These are currently design conventions rather than parser-enforced
constraints.

## Source model

Three glyphs are built in:

```text
.  air
#  solid support block
r  redstone dust
```

Blocks with essential orientation use document-local glyph definitions.
Attachment direction and signal flow are distinct concepts:

```text
glyph "T" = torch on x-;
glyph ">" = repeater toward y+ delay 1;
```

For repeaters, the stored direction identifies the side from which the repeater
reads its input; its hard-powered output is on the inverse side. A composable
torch-to-torch logic edge therefore needs an explicit hard-power element:

```text
source torch -> repeater -> solid support -> destination torch
```

A torch only weak-powers an adjacent solid block. Consequently
`source torch -> solid support -> destination torch` is not a valid cascaded
logic edge: weak power does not turn off a torch attached to that support.

Inputs materialize unpowered switches and name their attachment:

```text
input "a" at [0, 1, 1] on x+;
output "sum" at [1, 8, 6];
probe "carry_merge" at [1, 6, 4];
```

Repeating an input name declares multiple physical contacts for one logical
port. Verification drives every contact to the same value. This makes physical
fan-out explicit without pretending the contacts are independent inputs:

```text
input "a" at [0, 1, 1] on x+;
input "a" at [0, 9, 5] on x+;
```

The source never stores derived simulation state such as dust connectivity,
power strength, torch `lit`, or repeater `powered`/`locked`. `World3D` and
`Simulator` initialize those values.

Dust and repeaters may request automatic cobblestone below them:

```text
auto-support dust repeater;
```

Automatic supports are reported separately by `PhysicalCellBuild`. Torches and
switches still require their declared support block to exist because attachment
is part of their logic.

Expected combinational behavior reuses the existing logic-expression parser:

```text
expect "sum" = a ^ b ^ cin;
expect "cout" = (a & b) | (a & cin) | (b & cin);
expect "carry_merge" = (a & b) | (a & cin) | (b & cin);
```

Verification constructs a fresh simulator for every input combination, drives
the named inputs, and reads power from every named probe and output block.
Every probe and output must have one `expect` expression. Probes are internal
observations; outputs remain the cell's external contract.

The verifier prints each observation's expected and actual truth signature in
truth-table mask order. It also derives behavioral influence by toggling one
input at a time. Missing expected influence points to a disconnected dependency;
unexpected influence points to coupling or backfeed. If verification fails, the
first failing probe in declaration order (then the first failing output) is
selected automatically and rerun with a bounded causal power trace.

## CLI

Compile, verify, and export:

```powershell
cargo run --release --bin rcell -- test/inverter.rcell test/inverter.nbt
```

Print canonical source:

```powershell
cargo run --release --bin rcell -- test/inverter.rcell test/inverter.nbt --emit
```

Recompile when the source changes:

```powershell
cargo run --release --bin rcell -- test/inverter.rcell test/inverter.nbt --watch
```

Run and explain one truth-table case:

```powershell
cargo run --release --bin rcell -- test/inverter.rcell test/inverter.nbt `
  --case "a=1" `
  --explain y `
  --state-json target/inverter-debug.json
```

`--explain` accepts a probe/output name or an `x,y,z` coordinate. It walks the
settled simulator state backward through active dust, repeater, torch, and
powered-block inputs. Dust traversal follows increasing signal strength to
avoid expanding the same bidirectional wire repeatedly.

Compare the block states produced by two input cases:

```powershell
cargo run --release --bin rcell -- test/inverter.rcell test/inverter.nbt `
  --case "a=0" `
  --compare-case "a=1"
```

The state JSON contains every non-air block's final power state, direction,
candidate power sources, output labels, and the bounded simulator event trace.
This is intended both for command-line diagnosis and later viewer overlays.

Analyze a verified cell before manually compacting it:

```powershell
cargo run --release --bin rcell -- test/full-adder-2x20x20.rcell `
  test/full-adder-2x20x20.nbt `
  --compact-report `
  --compact-json target/full-adder-compact.json
```

The compact report includes the occupied bounds, block-kind counts, total
repeater delay, consecutive repeater chains, connected dust components, and
safe single mutations. A safe mutation removes one dust/repeater/torch or
replaces one repeater with dust, then reruns the complete truth table. Each
mutation is verified independently; multiple suggestions must be applied and
reverified incrementally because individually safe mutations are not
necessarily safe together.

Use `--no-verify` for an incomplete layout without `expect` statements. The NBT
is exported before verification so a failing circuit can still be inspected.

## Manual design protocol

Manual RCELL work must use the repository's simulator source, simulator tests,
and primitives verified directly against that simulator. Do not infer behavior
from memory of Minecraft mechanics. External implementations of the target
cell are not design inputs.

Before changing a circuit:

1. Write and verify the logical netlist and the eight-entry truth signature of
   every intermediate signal.
2. Select exactly one first failing probe.
3. Use a simulator trace to explain that failure in terms of coordinates and
   a causal power path.
4. If the failure cannot be explained, add instrumentation instead of changing
   the circuit.

One iteration changes one primitive or a bounded set of blocks. Regions that
already pass remain unchanged. If an earlier probe's truth signature changes,
discard the patch. Do not rewrite the entire circuit as an iteration.

After each patch:

- construct a fresh world and run all eight input combinations;
- inspect at the declared settle tick or stable state;
- check syntax/lint, support, orientation, occupancy, keepout, and bounding box;
- report every probe and final output truth signature;
- on failure, analyze only the first divergence and its causal trace.

Redesign is allowed only after reachability or influence analysis demonstrates
that the existing structure cannot realize the required dependency. Preserve
failed circuits as minimized regression cases rather than deleting them.

## Scope

Version 1 is intentionally small:

- exact planes rather than an interactive voxel editor
- cobble, dust, torch, repeater, switch inputs, and powered-block outputs
- combinational truth-table verification
- no macros, routing commands, or automatic torch attachment inference

If repeated manual experiments establish useful higher-level concepts, they can
later become physical recipes or an RCIR physical implementation section.
