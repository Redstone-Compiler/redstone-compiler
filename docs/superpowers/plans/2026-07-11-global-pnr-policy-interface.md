# Global PnR Policy Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce configurable global-only heuristic policies, budgets, presets, and decomposed placement costs while preserving current default behavior.

**Architecture:** Define policy types in a focused module, route placement generation through enum-selected strategies, and rank attempts with a cost breakdown and weights. Keep candidate generation, local placement, detailed routing, and verification APIs intact.

**Tech Stack:** Rust, existing global PnR modules, Cargo release tests.

---

### Task 1: Global budgets and presets

**Files:**
- Create: `src/transform/place_and_route/global_pnr/policy.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`

- [ ] Write failing tests for `Fast`, `Balanced`, and `Thorough` budget ordering and default-balanced equivalence.
- [ ] Run focused tests and verify the policy API is missing.
- [ ] Implement `GlobalSearchBudget`, `GlobalPnrPolicies`, `GlobalPnrPreset`, and preset constructors.
- [ ] Re-run focused tests.

### Task 2: Placement heuristic selection

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/policy.rs`
- Modify: `src/transform/place_and_route/global_pnr/placer.rs`

- [ ] Write a failing test showing a shelf-only policy produces no grid attempts.
- [ ] Run the focused test and verify failure.
- [ ] Add `PlacementHeuristic` and dispatch existing generators through the configured ordered list.
- [ ] Re-run placer and global PnR tests.

### Task 3: Placement cost breakdown

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/policy.rs`
- Modify: `src/transform/place_and_route/global_pnr/placer.rs`

- [ ] Write failing tests for default cost compatibility and custom vertical weight ordering.
- [ ] Run focused tests and verify failure.
- [ ] Implement `PlacementCostBreakdown`, `PlacementCostWeights`, and weighted ranking.
- [ ] Re-run placer and global PnR tests.

### Task 4: End-to-end verification

**Files:**
- Modify only for intentional generated-output changes: `test/counter-global-smoke.outputs.json`

- [ ] Run formatting and focused policy/global PnR tests.
- [ ] Run the ignored release counter smoke and compare its first selected attempt with the existing fast path.
- [ ] Run `cargo test --release` and report the known unrelated simulator failure separately if it remains.
- [ ] Commit only global PnR source and documentation files.
