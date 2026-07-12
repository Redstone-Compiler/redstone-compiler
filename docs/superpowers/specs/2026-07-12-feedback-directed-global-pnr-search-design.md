# Feedback-Directed Global PnR Search Design

## Goal

Turn global PnR from a bounded enumeration that discards failure information into a measurable, staged search loop. The first benchmark is the two-bit counter, but the interfaces must remain circuit-independent and accept future placement, routing, and verification heuristics.

The initial success criterion is not merely producing an NBT. A solution must route successfully and pass two complete counter cycles. The search report must also explain how every rejected attempt failed.

## Current Failure Mode

The current pipeline generates valid local candidates, selects a small set of layout combinations, produces placement attempts, routes them, and invokes an optional verifier. This has three structural weaknesses:

1. `layout_combinations` explores the all-zero selection followed by changes to one child at a time. It does not cover interactions between alternative candidates of multiple children.
2. Placement ranking uses geometric proxies such as volume, wire length, vertical distance, and estimated congestion. These costs do not capture dynamic redstone interference.
3. Routing and verification failures become strings and are discarded. Later attempts cannot avoid the same failed net, region, or signal-contact pattern.

Increasing candidate and combination limits therefore increases runtime without proportionally increasing useful coverage.

## Architecture

Split search into four explicit interfaces:

### Search Candidate

A `GlobalSearchCandidate` identifies:

- the selected local candidate for each child;
- a placement heuristic and its parameters;
- a placement seed or mutation history;
- a net-order strategy;
- stable identifiers for reproducibility.

It contains search decisions, not a fully assembled world.

### Evaluation Pipeline

Evaluate every search candidate through increasingly expensive stages:

1. placement legality and geometric cost;
2. cheap route probe or coarse routeability estimate;
3. full detailed routing and route-contract checks;
4. short semantic verifier used as an inexpensive rejection filter;
5. full semantic verifier, including two complete counter cycles.

A stage runs only if all cheaper stages pass. The counter verifier remains a benchmark-provided callback rather than counter-specific global PnR logic.

### Structured Outcome

Replace opaque last-error handling with a `GlobalAttemptOutcome` and `GlobalFailure` taxonomy. At minimum, failures distinguish:

- illegal placement;
- route search exhaustion, including failed net and endpoints;
- forbidden signal contact or powered-position contract failure;
- assembly failure;
- semantic mismatch, including verifier phase, tick or edge, expected value, and actual value;
- evaluation budget exhaustion.

Every attempt records stage durations, placement cost breakdown, route statistics, and its final outcome. Human-readable progress messages are rendered from this structured record.

### Search Strategy

A `GlobalSearchStrategy` proposes candidates and receives completed outcomes. The first implementation uses deterministic best-first or beam search rather than simulated annealing:

- begin with diverse local-candidate selections rather than only one-child substitutions;
- retain the best candidates that reach the deepest evaluation stage;
- mutate one decision at a time: child candidate, Free3D seed or parameters, placement displacement, or net order;
- suppress exact duplicates using a canonical candidate signature;
- allocate additional work to failures that were close to success;
- preserve fixed budgets for total attempts and per-stage work.

This interface allows future annealing, evolutionary search, or learned policies without changing evaluation and diagnostics.

## Feedback Policy

The first version uses conservative, explainable feedback:

- route search exhaustion raises priority for alternate net orders, increased local clearance near the failed endpoints, or a placement mutation that shortens that net;
- forbidden signal contact penalizes reuse of the implicated spatial corridor and favors separation of the involved nets;
- semantic mismatch after successful routing ranks above route failures because it reached a deeper stage, but repeated identical mismatch signatures are deduplicated;
- budget exhaustion is not treated as evidence that a candidate is physically invalid.

Verifier feedback must not directly alter circuit semantics. It only guides which physical candidate is evaluated next.

## Candidate Diversity

Local candidate caching remains in place and is orthogonal to search. Candidate selection should explicitly preserve diversity across:

- port geometry;
- bounding-box shape;
- block count;
- sequential macro orientation or topology when available.

Combination generation should use a bounded best-first Cartesian traversal or beam expansion. It must be capable of selecting non-default candidates for multiple children simultaneously.

## Observability and Artifacts

Each run produces a compact search report containing:

- counts and time by evaluation stage;
- failure counts grouped by structured signature;
- the deepest stage reached by each candidate;
- best-so-far candidates and their costs;
- reproducible identifiers for saved failed worlds and the final solution.

Failed NBT artifacts remain opt-in. The report is always available and must not require parsing console text.

## Delivery Sequence

1. Add structured attempt outcomes and timing without changing candidate ordering.
2. Split the counter verifier into short and full phases and establish the two-cycle benchmark.
3. Add search reports and verify that current failures are classified reproducibly.
4. Replace one-child-only combination enumeration with bounded diverse Cartesian exploration.
5. Add feedback-directed beam mutations for placement and net ordering.
6. Tune Free3D only after reports show placement is the dominant remaining failure source.

Each step is independently testable and keeps the existing router and local placer behavior intact unless its evidence identifies a specific defect.

## Testing

- Unit tests cover every `GlobalFailure` classification and report aggregation.
- Combination tests prove that bounded exploration can vary multiple child selections and remains deterministic.
- Duplicate signatures are evaluated once.
- Stage gating proves that expensive verification does not run after an earlier failure.
- Existing global PnR tests continue to run with `cargo test --release`.
- The counter benchmark requires two complete cycles and records stage counts and durations.
- A fixed configuration and seed reproduce the same candidate sequence and failure signatures.

## Non-Goals

- Proving PnR soundness or completeness;
- replacing the local placer;
- implementing simulated annealing before structured evidence exists;
- encoding counter-specific behavior in the search engine;
- unbounded exhaustive enumeration.
