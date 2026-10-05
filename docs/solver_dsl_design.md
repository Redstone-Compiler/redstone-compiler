# Solver modeling DSL design (rsdsl v2)

Status: implemented. Written 2026-10-03 after the exact SAT placer
(`exact_local_placer.md`) landed in PR #57; section 11 records the
implementation and its measurements against the hand-written encoder.

## 1. Goal

The exact placer turns a logic graph into CNF in hand-written Rust
(`local_placer/exact/encode.rs`, about 1,100 lines). Changing the physics or
an encoding detail means editing that code. The goal of this design:

> Every clause the solver receives comes from a readable model file. Nobody
> edits the CaDiCaL input, or the Rust that builds it, to change constraints.

The pipeline becomes:

```
LogicGraph + placement request
        │  (Rust: netlist, signal vocabulary, pins → instance facts)
        ▼
model file (.rsdsl)  +  instance   ──ground──▶  Boolean IR  ──lower──▶  CNF
                                                                 │
                                         CaDiCaL / DIMACS / (SCIP LP) ◀┘
        ▲                                                         │
        └── Rust loops: simulator CEGAR, construct, compact ◀─────┘
```

### Requirements

The exact placer needed each of these, so the DSL must express them.

| # | Requirement | Where `encode.rs` needs it |
| --- | --- | --- |
| R1 | Variable families indexed by instance domains (cells, directions, classes, cases) | kinds, classes, per-case power |
| R2 | Variables that do not exist for some indices (guards), folded to constants | no dust at z = 0; no wall torch off the edge; inward-only repeaters |
| R3 | Choices (exactly one of several members, members carrying payloads) | block kind; signal class |
| R4 | Derived booleans (definitions) and plain constraints | dust connection and pointing, strong power, per-case power |
| R5 | Sparse relation families built from many contributions | power relations source → sink |
| R6 | Bounded integers with order encoding and `<`, `<=` | local ranks, stages |
| R7 | Cardinality with a choice of encoding | one switch per input; block limit |
| R8 | Instance tables (facts) and parameters | vocabulary functions, switch and output sites, fixed cells |
| R9 | Free decision variables per relation tuple | "contributes" parents |
| R10 | Polarity-aware encoding, so one-sided aux vars stay one-sided | per-case witnesses |
| R11 | Rule labels and readable names for every variable | explained DIMACS output |
| R12 | Incremental use: assumptions, added clauses, value lookup by name | CEGAR blocking, window repair, fixed cells |
| R13 | Optional relaxation guards per rule (which rule makes it UNSAT) | the model-fidelity diagnostics |
| R14 | Phase preference without a phase API | "air is the negated occupied variable" trick |
| R15 | Grounding and CNF no larger or slower than the hand-written encoder | measured benchmarks in `exact_local_placer.md` |

## 2. Is rsdsl v0.1 suitable?

[rsdsl](https://github.com/Redstone-Compiler/rsdsl) v0.1 is a proc-macro DSL
that lowers to a 0/1 ILP in SCIP `.lp` format. Its NOT-gate example covers a
3-cell 2D line with approximate physics and mostly forced placement.

| Aspect | v0.1 | Verdict |
| --- | --- | --- |
| `model` / `rule` / `forall` / `def` / `require` / `force` | yes | Keep the surface syntax and ideas |
| `sources` + `add` + `OR(...)` | yes; ORs every added term | Keep the idea, generalize to relations (R5) |
| `scenario` | yes | Keep, becomes an instance domain (`Case`) |
| Target | ILP (SCIP LP) through linearization | Replace with a Boolean IR and a SAT-first backend; LP becomes an optional backend |
| Domains | `Cell` (2D) plus enums fixed in the macro | Needs 3D grids, instance-supplied domains, and partial functions (R1, R8) |
| Variable existence | full cartesian product | Needs guards and constant folding (R2) |
| Choices, integers, cardinality | no; only linear sums | Needed (R3, R6, R7) |
| Acyclic power | `def DP <-> OR(sources)` lets dust rings power themselves | Must be expressible (R6, R9); this was the hardest part of the exact placer |
| Names | strings only; aux vars are `__aux_*` | Needs names, labels, and a symbol table (R11, R12) |
| Parsing | proc-macro over Rust tokens | Prefer a runtime text format (section 7) |

**Conclusion:** rsdsl is the right starting point for the language. The
core semantics and backend need redesigning around a Boolean IR rather than
ILP strings. Section 3 describes that design, called v2. The v0.1 LP
linearization code can become one lowering of the new IR.

## 3. Language

The language is specified in `solver_dsl_grammar.md`. The reference grammar
`docs/rsdsl/rsdsl.lark` is LALR(1) with no conflicts. The full exact-placer
model is `src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl`, and a sample instance is
`docs/rsdsl/inverter_1x3x2.instance.rsdsl`.

In short:

- **Compile-time world:** `grid` (with explicit axis mapping), `enum`,
  `subset`, `domain`, `fact` (extern or derived with `:=`), `param`, and `fn`.
- **Solver-level declarations:**
  - `choice` gives exactly one member per index, with member guards.
  - `var` declares free booleans.
  - `def Name[...] := formula` defines named derived booleans.
  - `relation R(...)` is a sparse family built from `R(...) |= f`
    contributions.
  - `int ... in lo..=hi` declares order-encoded integers.
- **Rules:** `rule "label" { forall/if/let/require/contribution }`.
- **Formulas:**
  - connectives `not and or xor -> <->`;
  - pattern tests `Kind[c] is Dust | Torch(_)`;
  - aggregates `any/all/count/exactly_one/at_most_one`;
  - integer comparisons.
- **Annotations:** `@prefer`, `@outside`, `@display`, `@internal`, `@label`,
  `@encoding`, `@guarded`.

## 4. Semantics and lowering

1. **Parse** to an AST with source spans; type-check compile-time against
   solver-level expressions. Guards may only use compile-time terms.
2. **Ground** against the instance:
   - Expand `forall` and evaluate guards, facts, and `match`.
   - Create choice members only where their guards hold.
   - Collect relation contributions per tuple.
   - The result is a Boolean IR DAG of `Var`, `Not`, `And`, `Or`, `Card`,
     and `IntGe` nodes.
3. **Simplify:**
   - Fold constants (including every `none`-indexed variable), flatten, and
     deduplicate.
   - Hash-cons `And` and `Or` nodes by their sorted child literals.
   - Drop relation tuples whose OR folds to `false`.
4. **Encode:**
   - Tseitin with polarity analysis (Plaisted–Greenbaum). `require` contexts
     only need one direction; top-level `def`s (unless `@internal`) get both.
   - Choices: exactly-one, pairwise up to 6 members, otherwise a sequential
     counter (same as `cnf.rs` today).
   - Integers: order encoding, with comparisons expanded like
     `Encoding::strictly_below`.
   - Cardinality: the selected encoding.
5. **Emit** CNF plus a symbol table:
   - every variable's origin (declaration, indices, member) and its display
     name;
   - the label of the rule that produced each clause;
   - polarity flips.

   Backends: in-process CaDiCaL, DIMACS with `None`, `Legend`, or `Explained`
   comments (today's `DimacsComments`), and optionally SCIP LP by linearizing
   the same IR.

Variable numbering is deterministic: declaration order, then index order. The
same model and instance always give the same CNF.

## 5. Rust interface

`crates/rsdsl` (no dependencies) implements it:

```rust
let model = rsdsl::Model::parse("exact_placer.rsdsl", SOURCE)?;   // rsdsl::Error renders rustc-style
let mut instance = rsdsl::Instance::new("full-adder");
instance
    .grid("Cell", (2, 14, 10))
    .domain("Case", (0..8).map(IValue::from))
    .domain("Class", class_names.iter().map(IValue::sym))
    .param("rank_levels", 24)
    .row("on", vec![IValue::sym("n1"), 3.into()]);        // fact rows
let program = model.ground(&instance, GroundOptions { guards: false, provenance: false })?;

program.literals();                                       // zero-terminated clauses
program.option_lit("Kind", &[IValue::cell(0, 0, 1)], "Torch", &[IValue::sym("Floor")]);
program.def_lit("Powered", &[3.into(), IValue::cell(0, 0, 1)]);   // 1 / -1 for constants
program.relation("Feeds");                                // tuples with literals
program.int_lits("Rank", &[cell]);                        // order encoding, ">= lo + i + 1"
program.guards();                                         // @guarded selectors
program.add_clause(&blocking, "label");                   // CEGAR, blocked layouts
program.write_dimacs(&mut out, Comments::Explained, &header)?;
```

`Model::ground_file` grounds against an instance file instead. The solver
binding stays in the exact placer (`solver.rs`); the program is plain CNF.

## 6. What stays in Rust

The DSL describes constraints only. These remain ordinary Rust around the
session:

- Netlist extraction and the signal vocabulary (they compute instance facts).
- Simulator verification and blocking clauses (CEGAR).
- Lazy loop formulas (`acyclic.rs`), through `add_clause`.
- Windowed construction and compaction, which change instance facts
  (`fixed`, sites, box size) and re-ground.

Grounding a 2x14x10 box should take milliseconds, so re-grounding per window
is cheap.

## 7. Runtime text versus proc-macro

v0.1 parses inside `rsdsl!{}` at compile time. v2 should parse text at run
time:

- Domains and facts come from the instance; a proc-macro gains nothing there.
- Models can be edited and re-run without recompiling, like RCELL files.
- Error messages point at model lines; explain output can quote rule text.

`include_str!` keeps the default model in the binary, and a config field can
point at another file for experiments. A thin `rsdsl!{}` wrapper over the
same parser can be added later if compile-time checking is wanted.

## 8. Coverage check against `encode.rs`

`src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl` writes the whole current model in the DSL. Every clause family has
a home:

| `encode.rs` | DSL (`exact_placer.rsdsl`) |
| --- | --- |
| `encode_kinds`: one kind per cell, existence rules, supports, one switch per input | `choice Kind` with member guards; rules "지지", "스위치" |
| `encode_fixed_cells`, `blocked` | rule "고정" (facts); blocking stays in Rust via `add_clause` |
| `encode_classes`: one class per non-air cell, unpowered, switch classes, per-case power | `choice Sig`, the signal rules, `def Powered` |
| `encode_dust_shape`: `conn`, `points` | `def Conn`, `def Points` |
| `encode_relations`: all power relations, locks, switch hazards | `relation Feeds` contributions; rules "잠김", "스위치 위험" |
| `encode_soundness` (plus relax guard) | rule "건전성" `@relaxable` |
| `encode_coverage`: contributes, ranks, stages, witnesses, parents | `var Contrib over Feeds`, `int Rank`, `int Stage`, rule "정당화" |
| `encode_torches`: inversion per case, stage order | rule "토치" |
| `encode_outputs`: sites, driving outputs | the output rule, `def Observe` |
| `encode_block_limit` | `count(...) <= max_blocks @encoding(seqcounter)` |
| Out-facing repeater pruning, z = 0 pruning | member guards plus the `outward_ok` fact |
| "Air is a negated variable" | `prefer Air` |

## 9. Validation and migration

1. **Build v2 beside the current encoder.** Put a workspace crate under
   `crates/rsdsl` and use `src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl` as the model.
2. **Check equivalence on fixed layouts.** The model-acceptance tests (manual
   2x14x10, 2x13x9, 2x17x10) must be SAT with either encoder.
3. **Check equivalence on solving.**
   - Inverter, NOR, XOR, and the infeasibility proof must agree.
   - For any solution from one encoder, assume its block kinds in the other;
     it must be SAT.
4. **Check size and speed.**
   - Compare variables and clauses per rule label.
   - Compare XOR and XNOR-core times and the completion benchmark (40/60
     free cells).
   - The DSL version must not be worse; this is R15.
5. **Switch over.** Make `ExactLocalPlacer` call the session API, then delete
   `encode.rs`, `cnf.rs`, and `dimacs.rs`. The simulator loop, construction,
   and compaction stay unchanged.

Status: done. Steps 1–4 landed first and the placer grounded the DSL by
default (section 11); step 5 followed on 2026-10-04. The hand-written
encoder, the `Cnf` builder, and the legacy DIMACS legend are gone.
`encode.rs` keeps only the shared vocabulary and geometry types, `cnf.rs` is
a plain literal container, and `dimacs.rs` writes the grounded program.
6. **Leave the heuristic placer alone.** It shares only the placement
   request. The model can validate or compact its output by fixing its blocks
   as instance facts, as today's compaction already does.

## 10. Open questions

1. Separate rsdsl repository or a crate in this workspace? A workspace crate
   is recommended while the language settles; sync it to the rsdsl repository
   later.
2. Keep the SCIP LP backend? It is cheap to keep as a second lowering, but
   has low priority.
3. Should `relation` tuples with the same key but different contributions
   stay separate, as in `encode.rs`, or be ORed into one? Resolved: they are
   ORed. Semantics are unchanged (the layout sets are equal) and the full
   adder has 60% fewer relations.
4. Expose a solver callback (IPASIR-UP) for lazy constraints later? This
   would need `cadical-sys` instead of the `cadical` crate.

## 11. Implementation and measurements (2026-10-03)

### What exists

- `crates/rsdsl` (no dependencies): lexer and recursive-descent parser
  (`parser.rs`, same language as `docs/rsdsl/rsdsl.lark`), grounder
  (`ground.rs`), formula encoder (`formula.rs`), API and DIMACS writer
  (`program.rs`), Rust-built instances (`instance.rs`).
- The exact placer grounds
  `src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl` by
  (`exact/dsl.rs`), the only encoder since 2026-10-04 (see "Removing the
  hand-written encoder" below).
- `ExactPlacerConfig::model_file` and `model_params` swap the model or set
  its params at run time. `max_repeaters` is one such param in the shipped
  model.

### Encoder decisions that turned out to matter

- **Polarity-aware Tseitin with hash-consing.** `require` contexts get one
  direction, exported `def`s both; identical junctions share one variable.
- **Distribution with a shared prefix.** `a or b or (p and q and r)` becomes
  three clauses, but when the prefix is long its disjunction is named once
  (same propagation, fewer literals).
- **Merged relation tuples.** `Feeds` ORs contributions per key, so switch
  power is one tuple per cell and neighbor instead of one per (site, input).
  This is why the DSL formula has 60% fewer relations.
- **Variable numbering by first use.** CaDiCaL decides the highest-numbered
  variables first. Integers and exported definitions are created when a
  rule first needs them, so the numbering follows the rules' dependency
  order, as the hand-written encoder's did. Creating them up front made the
  60-second XOR test time out on its fixed seed; with first-use numbering it
  passes in 0.9 s.
- **OR auxiliaries are negated AND variables**, as in `cnf.rs`, so the
  solver's default phase starts disjunctions as true.

### Parity

These tests ran while both encoders existed; they were deleted with the
hand-written encoder (see below).

- `encoders_admit_the_same_layouts` enumerates every block layout each
  encoder admits and requires equal sets: inverter 1x3x2 (144 layouts) and
  1x4x2 (1,320), NOR 1x4x2 (200), NOR 1x5x2 with at most 9 blocks (1,206),
  inverter 2x2x2 (2,596), NOR 2x3x2 with at most 5 blocks (792).
- `encoders_accept_each_others_layouts`: a layout found with one encoder is
  SAT in the other with the same pins; both prove the 1x2x1 NOR box
  infeasible.
- `model_accepts_*`: both accepted the manual 2x14x10, 2x13x9, 2x17x10 and
  the compacted 2x10x10 full adders (these tests remain, for the model only).
- rsdsl unit tests check grounded CNF against brute-force model counts
  (choices, relations, `var over`, cardinality, order-encoded integers) and
  the diagnostics (E0304 stage errors, unknown members, `none` indices).

### Size and time (`measure_encoders`, release, Apple M-series, 2026-10-03)

Measured before automatic unit folding and before the hand-written encoder
was removed; `measure_encoders` now prints only the rsdsl rows.

| Problem | Encoder | Vars | Clauses | Literals | Relations | Encode | Load into CaDiCaL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inverter 1x3x2 | legacy | 598 | 4,599 | 12,869 | 70 | 0.16 ms | 0.49 ms |
| | rsdsl | 605 | 4,404 | 12,239 | 65 | 1.5 ms | 0.40 ms |
| XOR 2x6x4 | legacy | 13,787 | 182,024 | 519,757 | 2,852 | 5.5 ms | 15 ms |
| | rsdsl | 12,243 | 115,668 | 322,585 | 1,516 | 29 ms | 10 ms |
| full adder 2x14x10 | legacy | 143,950 | 2,114,886 | 6,071,573 | 29,256 | 62 ms | 160 ms |
| | rsdsl | 112,294 | 1,087,850 | 3,035,421 | 11,788 | 298 ms | 83 ms |

The formula is half the size. Grounding is interpreted and about 5x slower
than the hand-written loops (it started at 945 ms; span-keyed resolution
caches, an Fx hasher, and allocation-free clause emission brought it to
298 ms). Encode plus load is 380 ms against 222 ms on the largest box, small
next to solve times. Compiling rules to closures is the next step if it
matters.

### Search (heavy-tailed, so read these as distributions)

- XOR 2x6x4, 4-worker portfolio, 12 base seeds, 60 s cap
  (`compare_encoder_portfolios`): both solve 9/12. Legacy median about 27 s
  (four runs under 8 s); rsdsl median about 35 s (one run under 8 s).
- Full-adder construction (`diagnose_full_adder_construct_and_compact`,
  compaction off), before first-use numbering: legacy 42 s (seed 1) and a
  failure after 23 minutes (seed 2); rsdsl 256 s and 483 s.
  After first-use numbering, rsdsl took 438 s (seed 1) and 145 s (seed 2).
  The slow steps are not search time: all eight workers use up their
  eight simulator rejections (a torch reading the wrong value in case 0, a
  simulation that does not settle) and the window grows. The encoders admit
  the same layouts, so this is a model-fidelity gap shared by both, and now
  fixable by adding rules to `exact_placer.rsdsl`.
- Closing that gap (`exact_local_placer.md`, "Closing model–simulator gaps"):
  per-case strong power as a model rule, a simulator fix, and a settled start
  for verification and export. Seed-1 construction then takes 27 s with zero
  rejections; the full pipeline compacts it to 2x14x9 (171 blocks) in 900 s.

Objectives are implemented: `minimize`/`maximize` ground to weighted cost
literals, `Program::objective_counter` adds a totalizer for incremental bounds,
and `Program::write_wcnf` exports weighted partial MaxSAT. The exact placer's
`optimize` uses them (anytime, optimal when a bound is unsatisfiable).

R15 is met for formula size and solver load, not for grounding time; search
speed is within the run-to-run variance measured so far.

### Grounding profile (2026-10-04)

Profiled with macOS `sample` on a loop of `measure_encoders` (release with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`). The call tree was folded by a
small script into inclusive and self time per function.

- Grounding is 98% of `Encoding::build_dsl`, and running rules is 92%; the
  host side (`Prepared`, read-back, `Cnf`) is under 3%. The old per-rule
  report divided by the round count twice. Per rule, the soundness rule
  ("건전성") takes about 130 ms of the 315 ms full-adder box, justification
  40 ms, and the switch-neighbor rule 39 ms.
- About 30% of self time is the allocator (malloc, free, realloc, memmove).
  By caller: formula construction (`F::junction`, `F::imp`, `F::not`) 30%,
  `Encoder::require` 10%, environment drops in `exec` 9%, order-encoded
  comparisons (`int_cmp`, from `Rank[s] < Rank[t]`) 9%, and
  `resolve_constant` in patterns 6%. The rest is the tree walk itself:
  `eval`, `eval_binary`, index and fact lookups.
- Making lookups allocation-free on hits (index keys in a reused buffer,
  `Val::Choice` carrying the resolved key index, slice-keyed def and fact
  memos, a Tseitin cache keyed by a reused sorted buffer, cached pattern
  constants, sets iterated without copying) left the time unchanged within
  noise (316-327 ms). The cost is spread over the whole interpreter;
  compiling rules remains the way to cut it. These changes keep the CNF
  byte-identical (`print_cnf_hashes` matches the previous commit). An
  earlier variant that sorted Tseitin children in place changed only the
  order of auxiliary clauses, and the XOR placement test went from about 2 s
  to 41 s, so clause order is part of what must stay fixed.

What did pay off is grounding less. Compaction windows and construction steps
fix most of the box (and give its signals), yet every rule still unfolded
over it. Required units now fold automatically (grammar section 9): rules
that use no relation run first, a required literal is fixed (a fixed choice
option fixes its siblings), later formulas fold fixed literals, and a final
unit propagation over the CNF removes what was written before. A first
version needed an explicit `@fold` annotation on the fixed-cell rules. On a
block-reduction window of the 2x8x8 full adder (`measure_window_encoding`,
48 of 128 cells free):

| | Vars | Clauses | Literals | Encode |
| --- | ---: | ---: | ---: | ---: |
| before | 58,861 | 291,745 | 809,293 | 75 ms |
| `@fold` annotation | 25,018 | 124,415 | 343,077 | 40 ms |
| automatic | 24,086 | 104,901 | 281,099 | 47 ms |

What remains is mostly the window's own constraints (justification's order
encoding, then soundness). The models are the same, so placements do not
change, but the new rule order changes clause order on every problem, and
the solver's run time is heavy-tailed in it. XOR 2x6x4 over six seeds (4
workers, 60 s cap; `compare_xor_folding`): 24.7, >60, 7.9, 32.2, 40.2,
8.3 s automatic against 1.8, >60, 25.2, 17.4, 36.6, >60 s before. With 8
workers, four seeds: 2.0, >60, 8.7, 35.5 against 1.9, 16.4, 9.0, 18.2. The
unit test happened to draw 1.8 s before and 24.7 s after with four workers,
so it now uses eight. Full-adder construction stayed within run-to-run
variance: 134-182 blocks, 63-71 s, against 127-201 blocks across earlier
runs.

### Removing the hand-written encoder (2026-10-04)

The hand-written encoder only ran behind the hidden `legacy_encoder` flag,
and every model change after the switch (per-case strong power,
`outward_ok`, given signals, unit folding) went into the rsdsl model alone.
Keeping it meant keeping the old semantics behind a `legacy_semantics`
model param, a second code path in every config (`ExactPlacerConfig`,
`ConstructionConfig`, `CompactionConfig`), and parity tests that compared
the model to an older version of itself. So it was deleted:

- `encode.rs` (1,261 → 290 lines) keeps only the shared types: block
  vocabulary, `SignalClass`, pin geometry, `Relation`, and `Encoding`.
- `cnf.rs` (185 → 40) is a literal container; the clause builder, Tseitin
  helpers, and cardinality encodings went with the encoder.
- `dimacs.rs` (291 → 62) writes the grounded program; rsdsl's provenance
  replaces the old legend and per-clause explanations.
- The `legacy_encoder` flags, `PIPE_LEGACY`, the `legacy_semantics` param,
  and the parity tests (`encoders_admit_the_same_layouts`,
  `encoders_accept_each_others_layouts`, `compare_encoders`,
  `compare_encoder_portfolios`) are gone. `measure_encoders` prints the rsdsl
  rows only.

The grounded CNF is byte-identical before and after (`print_cnf_hashes` on
XOR 2x6x4, NOR 1x5x2, and the 2x14x10 full adder). What still checks
correctness: rsdsl's brute-force model counts, the simulator verification of
every placement, `model_accepts_*` on the manual full adders, and the
verified fixtures under `test/`.

---

## Appendix A. The current exact placer model

The appendix moved to `src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl` and was updated to the
final grammar in `solver_dsl_grammar.md`. It is about 280 lines and replaces
the encoding part of `encode.rs`, `cnf.rs`, and the legend and label code in
`dimacs.rs`.
