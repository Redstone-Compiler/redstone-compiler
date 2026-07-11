# Global PnR Search Redesign

## Goal

Replace the current one-way global place-then-route pipeline with an explicit search state that preserves useful child layouts and retries routing decisions, while retaining the existing local placer, detailed router, world assembly, and semantic verification code.

## Scope

The first increment contains four changes:

1. Preserve multiple layout candidates per child instance.
2. Represent a complete global attempt as a `GlobalSolution` with selected layouts, placement, routing result, cost, and diagnostics.
3. Generate a bounded, deterministic set of child-layout combinations and placement attempts.
4. Retry the same placement with multiple deterministic net orders before rejecting it.

General rip-up/reroute, negotiated congestion, module movement after routing, and external solver integration are deferred until this state model is validated.

## Existing Components Retained

- `LocalPlacer` and `LocalPlacerConfig`
- combinational truth-table candidate validation
- physical ports and routing access points
- point-to-point BFS/A* routing primitives
- assembled-world power-contract validation
- `GlobalPnrConfig::verifier`
- failed-world diagnostics
- existing counter global PnR smoke test

## Architecture

### Candidate pool

Candidate generation returns a `ChildCandidatePool` per module instance instead of selecting one layout immediately. Each pool is deterministically ranked by physical size, block count, port geometry, and a coarse geometry signature. The pool retains a configurable top-K plus geometry-diverse alternatives.

### Search state

`GlobalSolution` is the unit passed between global search phases:

```rust
pub struct GlobalSolution {
    pub selected_candidate_indices: Vec<usize>,
    pub placed_modules: Vec<PlacedModule>,
    pub routed_nets: Vec<RoutedNet>,
    pub cost: GlobalSolutionCost,
    pub diagnostics: GlobalSolutionDiagnostics,
}
```

Incomplete attempts use the same type with an empty `routed_nets` vector and diagnostics describing the last failure.

### Bounded layout combinations

The global search does not build an unbounded Cartesian product. It starts with the best-ranked layout for every child, then emits deterministic single-child substitutions and a bounded beam of cumulative substitutions. `max_layout_combinations` and `max_candidates_per_child` bound the work.

### Placement and routing

Each selected layout combination is passed to the existing placement policies. Placement attempts are ranked by a decomposed cost. For every placement, the router tries a bounded list of deterministic net orders:

- existing criticality priority
- longest estimated connection first
- highest fanout first
- reverse criticality order

The first valid route set for an attempt becomes a completed solution. Search continues until its budget is exhausted and returns the lowest-cost valid solution, rather than returning the first success globally.

## Configuration

```rust
pub struct GlobalSearchConfig {
    pub max_candidates_per_child: usize,
    pub max_layout_combinations: usize,
    pub max_solutions: usize,
    pub route_order_strategies: Vec<NetOrderStrategy>,
}
```

Defaults remain conservative so the current counter smoke stays tractable. Existing `candidate`, `placement`, and `routing` configuration remains available through `GlobalPnrConfig`.

## Cost and Diagnostics

`GlobalSolutionCost` separates:

- placement bounding volume
- estimated port-to-port wire length
- vertical distance penalty
- routed block count
- routed path length
- unrouted-net penalty

Diagnostics record the layout combination, placement index, net-order strategy, routed-net count, and final failure. Progress output reports these fields so a long search is observable.

## Correctness Boundary

Search policies are heuristic and need not be complete. A returned solution must still pass:

1. candidate truth-table validation for combinational children;
2. assembled route powered/released-position contracts;
3. the optional caller verifier;
4. existing counter behavior validation.

## Testing

- Unit tests prove top-K candidate pools retain distinct geometry.
- Unit tests prove layout combination generation is deterministic and bounded.
- Router tests prove alternative net orders are produced and attempted.
- Search tests prove a later layout or order can succeed after the first fails.
- The ignored release-mode counter smoke remains the end-to-end regression test.

## Success Criteria

- Multiple child layouts reach global placement.
- A failed routing order does not immediately discard a placement.
- Search results and diagnostics are deterministic for a fixed configuration.
- Search budget is directly configurable.
- Existing counter global PnR smoke passes without weakening its verifier.
- Existing local placer and detailed routing behavior remains unchanged.
