# Global PnR Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve child-layout diversity and retry global routing decisions through a bounded, observable global search state while keeping existing redstone placement and routing primitives.

**Architecture:** Add focused candidate-pool and search-state modules beside the existing global PnR code. Adapt orchestration incrementally so candidate combinations and net-order strategies are explicit and testable, then validate the complete flow with the existing release-mode counter smoke.

**Tech Stack:** Rust, petgraph-based module graphs, existing local placer and global router, Cargo release tests.

---

### Task 1: Child candidate pools

**Files:**
- Create: `src/transform/place_and_route/global_pnr/search.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`
- Test: `src/transform/place_and_route/global_pnr/search.rs`

- [ ] Add a failing unit test constructing several `LayoutCandidate` values and asserting that `rank_child_candidates(candidates, 2)` retains two deterministic, geometry-distinct candidates.
- [ ] Run `cargo test --release rank_child_candidates_keeps_bounded_geometry_diversity -- --nocapture` and verify failure because the API is missing.
- [ ] Add `ChildCandidatePool`, a deterministic ranking key, coarse port-geometry signature, and `rank_child_candidates` with the minimum implementation required by the test.
- [ ] Re-run the focused test and verify it passes.
- [ ] Add a failing test asserting `layout_combinations(&pools, 4)` contains the all-best combination plus bounded single-child substitutions in stable order.
- [ ] Run the focused combination test and verify failure.
- [ ] Implement deterministic bounded combination generation and re-run both search tests.

### Task 2: Explicit global solution state and cost

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/search.rs`
- Modify: `src/transform/place_and_route/global_pnr/placer.rs`
- Test: `src/transform/place_and_route/global_pnr/search.rs`

- [ ] Add a failing test asserting `GlobalSolutionCost` orders a shorter routed solution ahead of a larger placement with fewer arbitrary tie-break differences.
- [ ] Run `cargo test --release global_solution_cost_orders_complete_solutions -- --nocapture` and verify failure.
- [ ] Add `GlobalSolution`, `GlobalSolutionCost`, and `GlobalSolutionDiagnostics`, using named cost components and deterministic tuple ordering.
- [ ] Extract the current placement volume, wire-distance, and vertical-distance calculations into a public(crate) placement cost breakdown consumed by `GlobalSolutionCost`.
- [ ] Re-run the focused tests and existing `global_pnr::placer` tests.

### Task 3: Configurable net ordering

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/router.rs`
- Modify: `src/transform/place_and_route/global_pnr/search.rs`
- Test: `src/transform/place_and_route/global_pnr/router.rs`

- [ ] Add failing tests asserting `ordered_module_variables` produces stable `Criticality`, `LongestFirst`, `HighestFanoutFirst`, and `ReverseCriticality` orders.
- [ ] Run the four focused tests and verify failure because `NetOrderStrategy` is missing.
- [ ] Add `NetOrderStrategy` and extract variable ordering from `route_module_variables` into a strategy-aware helper.
- [ ] Add a strategy parameter to a new `route_module_variables_with_order`, preserving the existing function as the criticality-order compatibility wrapper.
- [ ] Re-run router unit tests and confirm existing routing tests remain green.

### Task 4: Search orchestration

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/search.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`
- Test: `src/transform/place_and_route/global_pnr/search.rs`

- [ ] Add a failing orchestration test using a small synthetic module where the first candidate/order is rejected by an injected attempt evaluator and a later attempt succeeds.
- [ ] Run `cargo test --release global_search_continues_after_failed_attempt -- --nocapture` and verify the test fails because no search orchestration exists.
- [ ] Add `GlobalSearchConfig` defaults and a bounded iterator over layout combinations, placement attempts, and route-order strategies.
- [ ] Replace `generate_child_candidates` with candidate-pool generation and adapt `place_and_route_module_with_visualization` to select the lowest-cost valid `GlobalSolution`.
- [ ] Preserve assembled-route contract checks, optional verifier invocation, failed-world saving, and progress reporting for every attempted state.
- [ ] Re-run focused global search, placer, and router tests.

### Task 5: Counter regression and performance evidence

**Files:**
- Modify only if required by intentional output changes: `test/counter-global-smoke.outputs.json`
- Modify only if required by intentional output changes: `tools/nbt-viewer/public/examples/manifest.json`

- [ ] Run `cargo test --release global_pnr -- --nocapture` and fix only regressions introduced by the new search orchestration.
- [ ] Run `cargo test --release counter_module_generates_world_from_child_layout_candidates -- --ignored --nocapture` and verify the two-bit counter behavior passes.
- [ ] Record total time, selected layout combination, placement attempt, route order, route count, and solution cost from the progress output.
- [ ] Run `cargo test --release` as required by repository guidance.
- [ ] Inspect `git diff --check`, `git status --short`, and generated artifact diffs; preserve unrelated pre-existing files.

