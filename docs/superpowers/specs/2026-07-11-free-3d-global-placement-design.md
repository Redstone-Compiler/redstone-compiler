# Free 3D Global Placement Design

## Goal

Add a global-only placement strategy that positions child-module bounding boxes freely in three dimensions instead of assigning an existing two-dimensional placement to fixed layers.

## Scope

The first implementation is a deterministic, bounded macro placer for the current small global module sets. It does not change local placement, block semantics, detailed routing, or verification. Simulated annealing and router-feedback refinement remain follow-up stages.

## Architecture

`PlacementHeuristic::Free3D` owns a `Free3DPlacementConfig` and generates placement attempts through a focused `free_3d` module. Each child layout is a hard, axis-aligned bounding box with a continuous center and velocity during relaxation.

The solver has three phases:

1. **Seed:** distribute module centers on a deterministic three-dimensional lattice sized from the module count and maximum bounding-box dimensions.
2. **Relax:** repeatedly apply net attraction, AABB-overlap repulsion, compacting force, vertical anisotropy, velocity damping, and a bounded step size.
3. **Legalize:** snap centers to integer block coordinates, translate the result into the positive world region, and iteratively remove remaining AABB overlaps along their minimum-penetration axis.

The generated placement is only a candidate. Existing placement costs rank it against other heuristics, the existing global router determines routeability, and the verifier remains the final semantic authority.

## Objective and Controls

`Free3DPlacementConfig` exposes only the controls needed by the first solver:

- `iterations`: bounded relaxation work;
- `step_size`: maximum movement per iteration;
- `attraction`: connected-module spring strength;
- `repulsion`: overlapping-box separation strength;
- `compactness`: pressure toward the placement centroid;
- `damping`: retained velocity between iterations;
- `vertical_scale`: relative cost of vertical movement;
- `clearance`: requested empty space between module bounding boxes.

All presets remain deterministic. `Balanced` keeps its current behavior. `Thorough` may add `Free3D` as an experimental candidate without removing existing layered or register-specific candidates.

## Correctness Boundary

The continuous solver is heuristic and may settle in a poor local minimum. A returned placement candidate must nevertheless satisfy:

- every child appears exactly once;
- all origins are valid nonnegative integer positions;
- expanded child bounding boxes do not overlap;
- existing global routing contracts and semantic verification remain unchanged.

If legalization cannot remove overlaps within its budget, that attempt is discarded rather than routed.

## Testing

- A Free3D-only policy generates a nonempty placement with multiple Z origins.
- Legalized placements contain every child once and have no expanded AABB overlaps.
- A connected pair is no farther apart than the same deterministic seed with attraction disabled.
- Existing global PnR release tests remain green.
- The expensive counter smoke is used after focused tests to compare footprint, height, routeability, and runtime against the recorded layered result.

## Follow-up Work

After the first solver is measurable, separate additions can provide simulated-annealing moves, route-failure heatmaps, coarse-router probes, multilevel clustering, or a voxel Poisson density solver. These are deliberately excluded from the first implementation.
