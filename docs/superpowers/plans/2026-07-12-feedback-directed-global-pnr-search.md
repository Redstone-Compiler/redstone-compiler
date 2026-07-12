# Feedback-Directed Global PnR Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic staged global PnR search that records structured failure evidence and uses it to explore diverse counter candidates efficiently.

**Architecture:** Keep local candidate generation, Free3D placement, detailed routing, and semantic simulation as separate evaluators. Introduce structured attempt records and reports first, then replace one-child-only combination enumeration with bounded best-first Cartesian exploration, and finally add a small feedback-directed beam strategy over candidate selection, placement variants, and net ordering.

**Tech Stack:** Rust, `eyre`, existing global PnR modules, release-mode Cargo tests, NBT simulator verification.

---

## File Structure

- Create `src/transform/place_and_route/global_pnr/diagnostics.rs`: failure taxonomy, stage timing, attempt records, report aggregation.
- Create `src/transform/place_and_route/global_pnr/evaluator.rs`: staged evaluation and stage gating.
- Modify `src/transform/place_and_route/global_pnr/search.rs`: deterministic Cartesian combination frontier and candidate signatures.
- Modify `src/transform/place_and_route/global_pnr/mod.rs`: orchestrate strategy/evaluator and expose the final report.
- Modify `src/transform/place_and_route/global_pnr/router.rs`: return structured route failure context without changing routing behavior.
- Modify `src/world/simulator.rs`: preserve the two-cycle artifact regression.
- Create `test/counter-global-search-report.json` only as an ignored/generated diagnostic artifact; do not commit generated reports.

### Task 1: Structured Attempt Outcomes

**Files:**
- Create: `src/transform/place_and_route/global_pnr/diagnostics.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`

- [ ] **Step 1: Write failing unit tests for failure grouping**

Add tests that construct `RouteSearchExhausted` twice with the same net/endpoints and once with different endpoints, then assert the report contains two signatures with counts `2` and `1`. Add a timing aggregation test with two `Routing` samples of 10 ms and 20 ms and assert count `2`, total `30 ms`.

- [ ] **Step 2: Run the focused release test**

Run: `cargo test --release global_pnr::diagnostics::tests -- --nocapture`

Expected: FAIL because `diagnostics` and its types do not exist.

- [ ] **Step 3: Implement the minimal diagnostic model**

Define:

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EvaluationStage { Placement, RouteProbe, Routing, ShortVerification, FullVerification }

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum GlobalFailure {
    IllegalPlacement { reason: String },
    RouteSearchExhausted { net: String, source: Position, sink: Position },
    ForbiddenSignalContact { net: String, position: Position },
    PoweredPositionContract { net: String, source: Position, sink: Position },
    Assembly { reason: String },
    SemanticMismatch { phase: String, step: usize, expected: usize, actual: usize },
    BudgetExhausted { stage: EvaluationStage },
}

pub enum GlobalAttemptOutcome { Passed, Failed(GlobalFailure) }
pub struct StageTiming { pub stage: EvaluationStage, pub elapsed: Duration }
pub struct GlobalAttemptRecord { pub signature: String, pub timings: Vec<StageTiming>, pub outcome: GlobalAttemptOutcome }
pub struct GlobalSearchReport { pub attempts: Vec<GlobalAttemptRecord> }
```

Implement deterministic grouping and timing summaries with `BTreeMap`.

- [ ] **Step 4: Export the module and rerun tests**

Run: `cargo test --release global_pnr::diagnostics::tests -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit only `diagnostics.rs` and the module declaration with subject `Record structured global PnR outcomes` and a body explaining that it adds evidence without changing search order.

### Task 2: Staged Evaluator Without Search Changes

**Files:**
- Create: `src/transform/place_and_route/global_pnr/evaluator.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`

- [ ] **Step 1: Write evaluator stage-gating tests**

Use injected closures that append `EvaluationStage` values to a shared vector. Assert routing is not invoked after a route-probe failure, short verification is not invoked after routing failure, and full verification is invoked only after short verification succeeds.

- [ ] **Step 2: Verify the tests fail**

Run: `cargo test --release global_pnr::evaluator::tests -- --nocapture`

Expected: FAIL because `GlobalCandidateEvaluator` is undefined.

- [ ] **Step 3: Implement sequential stage gating**

Add `GlobalCandidateEvaluator::evaluate` that times each invoked stage with `Instant`, stops on the first structured failure, and returns both the optional routed solution and a complete `GlobalAttemptRecord`. Do not add concurrency or retries.

- [ ] **Step 4: Adapt current routing and verifier calls**

Move the body of `route_first_successful_placement` behind the evaluator while keeping the existing nested attempt and net-order traversal unchanged. Convert existing error branches into the nearest `GlobalFailure`; preserve their human-readable progress messages.

- [ ] **Step 5: Run focused and global tests**

Run: `cargo test --release global_pnr::evaluator::tests global_pnr::tests -- --nocapture`

Expected: evaluator tests PASS and existing global PnR behavior remains unchanged.

- [ ] **Step 6: Commit**

Commit with subject `Stage global PnR candidate evaluation` and a body stating that expensive stages are gated while ordering remains stable.

### Task 3: Two-Phase Counter Verification

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`
- Modify: `src/world/simulator.rs`

- [ ] **Step 1: Preserve the existing failing two-cycle regression**

Keep the expected sequence `[1, 2, 3, 0, 1, 2, 3, 0]` in both the generated-world verifier and saved-NBT regression. Confirm the committed NBT fails at the fifth rising edge before replacing it.

- [ ] **Step 2: Add explicit short/full verifier callbacks**

Change global config from one opaque verifier to:

```rust
pub struct GlobalVerifier {
    pub short: Option<fn(&PlacedWorld) -> Result<(), GlobalFailure>>,
    pub full: Option<fn(&PlacedWorld) -> Result<(), GlobalFailure>>,
}
```

The counter short verifier checks initialization and one complete cycle. The full verifier checks two complete cycles and returns `SemanticMismatch` with phase, edge index, expected, and actual.

- [ ] **Step 3: Run the generated counter test with a bounded diagnostic budget**

Run: `cargo test --release counter_module_generates_world_from_child_layout_candidates -- --nocapture`

Expected: either PASS with a new two-cycle-valid world or FAIL with structured stage/signature output; it must not fail with only an opaque `eyre` string.

- [ ] **Step 4: Run the saved-artifact regression**

Run: `cargo test --release simulator_counts_through_full_two_bit_cycle -- --nocapture`

Expected before a replacement artifact: FAIL at the known second-cycle transition. Preserve this as evidence; do not weaken the expected sequence.

- [ ] **Step 5: Commit verifier plumbing separately**

Commit source changes with subject `Split global semantic verification into stages`. Do not commit a newly generated NBT until it passes the full verifier.

### Task 4: Deterministic Diverse Combination Frontier

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/search.rs`

- [ ] **Step 1: Add a failing multi-child interaction test**

For three pools with two candidates each and limit eight, assert the generated frontier includes `[1, 1, 0]`, `[1, 0, 1]`, `[0, 1, 1]`, and `[1, 1, 1]`, contains no duplicates, begins with `[0, 0, 0]`, and is identical across repeated calls.

- [ ] **Step 2: Confirm current enumeration fails the test**

Run: `cargo test --release global_pnr::search::tests::layout_combinations_cover_multi_child_interactions -- --nocapture`

Expected: FAIL because the current algorithm changes only one pool from the base selection.

- [ ] **Step 3: Implement bounded best-first Cartesian traversal**

Use a `BinaryHeap<Reverse<(usize, Vec<usize>)>>` keyed first by the sum of candidate ranks and then lexicographically. Seed `[0; pool_count]`; pop one selection, emit it, and enqueue each valid neighbor formed by incrementing one dimension. Deduplicate with `HashSet<Vec<usize>>` and stop exactly at the configured limit.

- [ ] **Step 4: Run search and global tests**

Run: `cargo test --release global_pnr::search::tests global_pnr::tests -- --nocapture`

Expected: PASS, including deterministic ordering and multiple-child combinations.

- [ ] **Step 5: Commit**

Commit with subject `Explore interacting global layout candidates` and a body explaining the bounded Cartesian frontier.

### Task 5: Search Report and Failure Accounting

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/diagnostics.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`

- [ ] **Step 1: Add report serialization tests**

Derive or manually implement `serde::Serialize` for report-facing types. Serialize a report with one routing failure and one semantic mismatch; assert stable keys for `stage_counts`, `stage_millis`, `failure_counts`, and `attempts`.

- [ ] **Step 2: Add report output configuration**

Add `report_path: Option<PathBuf>` to `GlobalSearchConfig`. At the end of success or exhausted search, write pretty JSON through a temporary file followed by rename so partial reports are not mistaken for complete runs.

- [ ] **Step 3: Run diagnostic tests**

Run: `cargo test --release global_pnr::diagnostics::tests -- --nocapture`

Expected: PASS with stable JSON snapshots constructed in memory.

- [ ] **Step 4: Run a small counter diagnostic search**

Set the test-only report path to `test/counter-global-search-report.json`, run a deliberately small budget, and inspect that every attempt belongs to exactly one terminal outcome. Remove the generated report after inspection.

- [ ] **Step 5: Commit**

Commit with subject `Report global PnR search failures` and a body documenting the generated, non-versioned JSON artifact.

### Task 6: Feedback-Directed Beam Strategy

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/search.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`
- Modify: `src/transform/place_and_route/global_pnr/policy.rs`

- [ ] **Step 1: Write deterministic strategy tests**

Construct outcomes for route exhaustion, forbidden contact, and full-verifier mismatch. Assert that route exhaustion proposes alternate net order and a placement variant, forbidden contact proposes increased clearance/separation, semantic mismatch is ranked deeper than routing failure, and identical failure signatures do not enqueue duplicate candidates.

- [ ] **Step 2: Implement search candidate identity and ranking**

Define a canonical signature from child selections, placement heuristic/parameters, placement variant seed, and net-order strategy. Rank frontier entries by deepest passed stage, then semantic depth, then placement weighted cost, then signature for deterministic ties.

- [ ] **Step 3: Implement bounded mutations**

Allow only four mutation families in the first version: increment one child candidate index, choose the next configured net order, choose the next deterministic Free3D seed, and increase local clearance by one bounded step around a failed net. Enforce total-attempt and frontier-width budgets.

- [ ] **Step 4: Run strategy and global tests**

Run: `cargo test --release global_pnr::search::tests global_pnr::tests -- --nocapture`

Expected: PASS with stable candidate sequences.

- [ ] **Step 5: Commit**

Commit with subject `Guide global PnR search with failure feedback` and a body listing the four bounded mutation families.

### Task 7: Counter Benchmark and Final Verification

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`
- Potentially regenerate only after success: `test/counter-global-smoke.nbt`
- Potentially regenerate only after success: `test/counter-global-smoke.outputs.json`
- Potentially regenerate only after success: `tools/nbt-viewer/public/examples/counter-global-smoke.nbt`

- [ ] **Step 1: Run the feedback-directed counter search**

Run: `cargo test --release counter_module_generates_world_from_child_layout_candidates -- --nocapture`

Expected: PASS within the configured bounded budget and produce an artifact that passes both verifier phases. If it fails, use the report to identify the dominant structured signature; do not simply increase all limits.

- [ ] **Step 2: Verify the saved artifact independently**

Run: `cargo test --release simulator_counts_through_full_two_bit_cycle -- --nocapture`

Expected: PASS for `[1, 2, 3, 0, 1, 2, 3, 0]`.

- [ ] **Step 3: Run the complete release suite**

Run: `cargo test --release`

Expected: all newly added tests PASS. Record any pre-existing unrelated failure separately rather than weakening the new assertions.

- [ ] **Step 4: Synchronize the viewer artifact only after verification**

Copy the verified `test/counter-global-smoke.nbt` to `tools/nbt-viewer/public/examples/counter-global-smoke.nbt`, then compare SHA-256 hashes and require them to match.

- [ ] **Step 5: Commit the verified benchmark**

Commit only the verified source, metadata, and viewer artifact with subject `Generate two-cycle-valid Free3D counter` and a body recording the search budget, elapsed time, and two-cycle verification command.

## Plan Self-Review

- Spec coverage: structured outcomes, staged evaluation, two-phase verification, diverse combinations, reports, feedback mutations, deterministic budgets, and counter benchmark are each assigned to a task.
- Scope: local placer and detailed router algorithms remain unchanged; router edits expose failure context only.
- Type consistency: `EvaluationStage`, `GlobalFailure`, `GlobalAttemptRecord`, and `GlobalSearchReport` originate in Task 1 and are reused by later tasks.
- Artifact safety: no generated NBT is accepted or copied until the independent two-cycle regression passes.
