# Full Counter Cycle Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject globally routed two-bit counters that fail after the second clock edge and regenerate a complete-cycle artifact.

**Architecture:** Extend the existing semantic verifier to cover all four counter states and falling-edge holds. Add an independent artifact regression in the world simulator tests, then regenerate the NBT through the same global PnR entry point.

**Tech Stack:** Rust, Cargo release tests, global PnR, wasm-bindgen viewer bridge

---

### Task 1: Reproduce the incomplete counter cycle

**Files:**
- Modify: `src/world/simulator.rs`

- [x] Add `simulator_counts_through_full_two_bit_cycle`, loading output positions from `counter-global-smoke.outputs.json`.
- [x] Run the focused release test and observe `actual=2, expected=3` on the old artifact.

### Task 2: Strengthen global PnR acceptance

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`

- [x] Replace the two-edge verifier with `0 -> 1 -> 2 -> 3 -> 0` rising-edge checks and falling-edge hold checks.
- [x] Run the ignored search-heavy counter test and require the stronger verifier to select a passing candidate.

### Task 3: Verify and publish the regenerated artifact

**Files:**
- Modify: `test/counter-global-smoke.nbt`
- Modify: `test/counter-global-placement-bbox.nbt`
- Modify: `test/counter-global-smoke.outputs.json`

- [x] Run the independent artifact regression against the regenerated files.
- [x] Run `cargo test --release`.
- [x] Rebuild viewer WASM, refresh examples, and verify four clock pulses.
- [x] Commit only the intended source, documentation, and generated counter artifacts with an intent-bearing message body.
