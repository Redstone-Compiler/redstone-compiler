# Local Candidate Structural Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate exact duplicate child-module layout candidates once per global PnR run.

**Architecture:** Compare graph-backed child modules and complete unit-candidate configuration while ignoring only module name. Cache candidate vectors in `generate_child_candidate_pools`, clone on hit, and relabel `LayoutCandidate.module_name` for the consuming instance.

**Tech Stack:** Rust, Cargo release tests, existing global/local placer APIs

---

### Task 1: Establish cache identity and failing tests

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`
- Modify: `src/transform/place_and_route/global_pnr/candidate.rs`
- Modify: `src/transform/place_and_route/local_placer/config.rs`
- Modify: `src/world/position.rs`

- [x] Derive equality for candidate configuration value types.
- [x] Add exact graph/port structural comparison that ignores only module name.
- [x] Add a generator-count regression test and verify it fails before caching.

### Task 2: Implement run-scoped candidate reuse

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`

- [x] Cache generated candidates by module structure and configuration.
- [x] Clone and relabel candidates on cache hits.
- [x] Verify graph, port, and config differences miss the cache.

### Task 3: Verify performance and behavior

**Files:**
- Modify generated counter artifacts only after a successful two-cycle Free3D run.

- [x] Run focused release tests.
- [x] Re-measure child candidate generation against the 288.6-second baseline.
- [ ] Run the two-cycle Free3D counter search and artifact regression.
- [ ] Commit intended source, documentation, and regenerated artifacts with an intent-bearing message body.
