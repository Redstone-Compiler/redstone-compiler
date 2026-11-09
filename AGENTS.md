# Agent Notes

## Documentation

- 8-bit ALU and binary/BCD/7-segment compiler roadmap, module contracts, verification, and parallel ownership: `docs/alu_display_compiler_plan.md`
- ALU/display implementation, physical-port regressions and row-COW profiling: `docs/alu_display_compiler_progress.md`
- SAT Exact candidates in Global PnR, verified external interfaces, provenance and controlled whole-decoder compaction measurements: `docs/exact_global_candidates.md`
- One monolithic Exact model for the full four-input/seven-output decoder, flattened RTL graph, bounded attempts and UNKNOWN interpretation: `docs/whole_decoder_exact.md`
- Whole-decoder window construction/compaction runner, live-frontier measurements, interface contracts and failed construction checkpoints: `docs/whole_decoder_construct.md`
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
- Symbolic constraint specification of the current exact placer (physical relations, representation restrictions, optional policies, and unproved preservation obligations): `docs/exact_placer_constraint_formula.md`
- Derived pruning lemmas, counterexamples, preservation checks, and Python/Z3 iteration lab: `docs/exact_placer_derived_lemmas.md`; runner: `tools/pruning_lab/run.py`.
- Exact boundary continuation experiments, unsafe summaries, and lossless MDD compression on tiny circuits: `docs/exact_boundary_representation.md`; runner: `tools/pruning_lab/boundary.py`.
- Exact Python search iteration, fixed-pin fixtures, and measured native QF_FD AllSAT improvements: `docs/exact_search_iteration.md`; benchmark: `tools/pruning_lab/benchmark_search.py`.
- Production Rust Kind decoding and single-worker dispatch optimization, A/B measurements and CNF parity: `docs/rust_placer_optimization.md`.
- Full-adder tiles that chain into n-bit ripple-carry adders (`CarryTiling`): `docs/carry_tiles.md`
- Critical paths in redstone ticks: static timing, delay bounds, and timing-first construction and compaction: `docs/timing_driven_placement.md`
- Equivalent NOR netlists from an e-graph (egg) with exact SAT extraction, an experiment record (the code was removed for exact synthesis) that also explains why wide-NOR construction steps time out: `docs/egraph_netlists.md`
- Exact synthesis of NOR netlists that windowed construction can build (`exact::synthesize`, `nor_synthesis.rsdsl`): `docs/nor_synthesis.md`
- Sequential cells (latches, flip-flops) with the exact placer: state nets, the loop cut, sequential verification, and compact latch fixtures: `docs/sequential_exact_cells.md`
- Multi-seed combinational OFF/ON measurements of source-rank normalization: docs/combinational_source_rank_benchmark.md
- DFF cold-search profiling, canonical source-rank optimization and preservation checks: docs/dff_search_optimization.md
- Cold global DFF placement without a fixed latch, six-gate topology constraints, reproducible solver model and independent NBT stress verification: `docs/global_dff_search.md`
- Solver modeling DSL (rsdsl v2, crate `crates/rsdsl`): design and parity measurements `docs/solver_dsl_design.md`; grammar and language spec `docs/solver_dsl_grammar.md`; the exact placer's model is `src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl` (grounded by `exact/dsl.rs`)

When asked to create or preserve project documentation, add an appropriate file under `docs/` and link it from this file when it is useful for future agents.

## Git

When committing changes, include the intent behind the change in the commit message body.

## Building, Testing, and Running

Always use release mode for all builds, unit tests, and executions: `cargo build --release`, `cargo test --release`, and `cargo run --release`. Do not use debug builds for this project. This applies to all targets, not only local placer and place-and-route tests; the search-heavy tests are especially slow in debug mode.
