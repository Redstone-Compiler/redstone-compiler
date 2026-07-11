# Free 3D Global Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic free-3D global placement candidate that relaxes connected hard bounding boxes in continuous space and legalizes them onto the Minecraft integer grid.

**Architecture:** Define a compact policy config and dispatch variant, implement the solver in a focused module beside the existing placer, and keep the existing cost, router, and verifier boundaries intact. Build behavior test-first and preserve current Balanced defaults.

**Tech Stack:** Rust, existing `GraphModule`/`LayoutCandidate`/`PlacedModule` types, release-mode Cargo tests.

---

### Task 1: Free3D policy boundary

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/policy.rs`
- Modify: `src/transform/place_and_route/global_pnr/placer.rs`
- Test: `src/transform/place_and_route/global_pnr/policy.rs`

- [ ] Add a failing policy test asserting that Balanced excludes and Thorough includes `PlacementHeuristic::Free3D`.

```rust
#[test]
fn free_3d_is_opt_in_through_the_thorough_preset() {
    let balanced = GlobalPnrPreset::Balanced.config();
    let thorough = GlobalPnrPreset::Thorough.config();
    assert!(!balanced.search.policies.placement_heuristics.iter()
        .any(|item| matches!(item, PlacementHeuristic::Free3D(_))));
    assert!(thorough.search.policies.placement_heuristics.iter()
        .any(|item| matches!(item, PlacementHeuristic::Free3D(_))));
}
```
- [ ] Run `cargo test --release global_pnr::policy::tests::free_3d_is_opt_in_through_the_thorough_preset` and confirm the variant is missing.
- [ ] Define `Free3DPlacementConfig` with deterministic defaults for iteration count, step size, attraction, repulsion, compactness, damping, vertical scale, and clearance.
- [ ] Add `PlacementHeuristic::Free3D(Free3DPlacementConfig)` and include it only in `GlobalPnrPreset::Thorough`.

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Free3DPlacementConfig {
    pub iterations: usize,
    pub step_size: f64,
    pub attraction: f64,
    pub repulsion: f64,
    pub compactness: f64,
    pub damping: f64,
    pub vertical_scale: f64,
    pub clearance: usize,
}

pub enum PlacementHeuristic {
    // existing variants
    Free3D(Free3DPlacementConfig),
}
```
- [ ] Re-run the focused policy test and commit the policy boundary.

### Task 2: Continuous relaxation and legalization

**Files:**
- Create: `src/transform/place_and_route/global_pnr/free_3d.rs`
- Modify: `src/transform/place_and_route/global_pnr/mod.rs`
- Modify: `src/transform/place_and_route/global_pnr/placer.rs`
- Test: `src/transform/place_and_route/global_pnr/free_3d.rs`

- [ ] Add a failing test that places eight hard boxes with a Free3D-only policy and asserts multiple Z origins and no expanded AABB overlap.

```rust
#[test]
fn free_3d_placement_uses_volume_and_legalizes_boxes() {
    let placed = place_free_3d(&GraphModule::default(), &eight_test_candidates(), config())
        .expect("free 3D placement");
    assert!(placed.iter().map(|item| item.origin.2).collect::<HashSet<_>>().len() > 1);
    assert!(!has_expanded_bbox_overlap(&placed, config().clearance));
}
```
- [ ] Run the focused test and confirm no Free3D solver or dispatch exists.
- [ ] Implement deterministic cubic-lattice seeding with module centers represented as `f64` vectors.
- [ ] Implement bounded attraction, overlap repulsion, centroid compactness, damping, and vertical anisotropy updates.
- [ ] Implement integer snapping, positive-world translation, and bounded minimum-penetration AABB legalization returning `None` on failure.

```rust
pub(crate) fn place_free_3d(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    config: Free3DPlacementConfig,
) -> Option<Vec<PlacedModule>>;
```
- [ ] Dispatch `PlacementHeuristic::Free3D` from both register and general module paths without retaining unwanted flat candidates.
- [ ] Re-run the focused test and existing layered-placement tests, then commit the solver.

### Task 3: Attraction behavior and global regression

**Files:**
- Modify: `src/transform/place_and_route/global_pnr/free_3d.rs`
- Test: `src/transform/place_and_route/global_pnr/free_3d.rs`

- [ ] Add a failing test comparing the same connected modules with attraction enabled and disabled, asserting that attraction does not increase their center distance.

```rust
#[test]
fn attraction_reduces_connected_module_distance() {
    let without = place_free_3d(&connected_module(), &candidates(), config_with_attraction(0.0)).unwrap();
    let with = place_free_3d(&connected_module(), &candidates(), config_with_attraction(0.08)).unwrap();
    assert!(connected_center_distance(&with) <= connected_center_distance(&without));
}
```
- [ ] Run the focused test and confirm the assertion fails before the attraction term is wired to module nets.
- [ ] Connect `GraphModule::vars` to deterministic center-to-center spring forces while applying `vertical_scale` to Z movement.
- [ ] Run `cargo fmt --check`, `cargo test --release global_pnr`, and `git diff --check`.
- [ ] Run the ignored counter smoke with a Free3D-only policy, recording route count, output count, footprint, height, cost, and runtime before deciding whether it belongs in a preset.
- [ ] Commit only global PnR source, tests, and intentional documentation or generated artifacts.
