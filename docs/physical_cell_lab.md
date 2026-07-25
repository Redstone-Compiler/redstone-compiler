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

Inputs materialize unpowered switches and name their attachment:

```text
input "a" at [0, 1, 1] on x+;
output "sum" at [1, 8, 6];
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
```

Verification constructs a fresh simulator for every input combination, drives
the named inputs, and reads power from the named output blocks.

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

Use `--no-verify` for an incomplete layout without `expect` statements. The NBT
is exported before verification so a failing circuit can still be inspected.

## Scope

Version 1 is intentionally small:

- exact planes rather than an interactive voxel editor
- cobble, dust, torch, repeater, switch inputs, and powered-block outputs
- combinational truth-table verification
- no macros, routing commands, or automatic torch attachment inference

If repeated manual experiments establish useful higher-level concepts, they can
later become physical recipes or an RCIR physical implementation section.
