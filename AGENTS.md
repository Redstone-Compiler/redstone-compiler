# Agent Notes

## Documentation

- Verilog RTL interface design notes: `docs/verilog_rtl_interface_design.md`
- RCIR language and lowering design: `docs/intermediate_representation_design.md`
- Physical design intent and local-cell recipes: `docs/physical_design_intent.md`
- Physical design intent and local cell recipes: `docs/physical_design_intent.md`
- PnR logging and observability: `docs/pnr_logging.md`
- Compilation snapshot artifacts: `docs/compilation_snapshots.md`
- Text-editable physical cell laboratory: `docs/physical_cell_lab.md`
- Bounded local-only full-adder diagnostic: `docs/local_full_adder_diagnostic.md`
- Manual RCELL full-adder carry diagnosis: `docs/rcell_full_adder_diagnostic.md`
- Readable, verified RCELL full-adder baseline: `docs/rcell_full_adder_baseline.md`
- Verified compact full adder and transferable placement rules: `docs/compact_full_adder_rules.md`
- Height-10 full adder and low XNOR composition rules: `docs/height10_full_adder.md`
- Further 2x13x9 compaction and carry bridge clearance: `docs/compact_full_adder_shrink.md`
- Full adder with operand switches on the same boundary: `docs/full_adder_input_boundary.md`
- Exact SAT-based local placer, survey, and construct/compact pipeline: `docs/exact_local_placer.md`
- Solver modeling DSL (rsdsl v2, crate `crates/rsdsl`): design and parity measurements `docs/solver_dsl_design.md`; grammar and language spec `docs/solver_dsl_grammar.md`; the exact placer's model is `src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl` (grounded by `exact/dsl.rs`)

When asked to create or preserve project documentation, add an appropriate file under `docs/` and link it from this file when it is useful for future agents.

## Git

When committing changes, include the intent behind the change in the commit message body.

## Building, Testing, and Running

Always use release mode for all builds, unit tests, and executions: `cargo build --release`, `cargo test --release`, and `cargo run --release`. Do not use debug builds for this project. This applies to all targets, not only local placer and place-and-route tests; the search-heavy tests are especially slow in debug mode.
