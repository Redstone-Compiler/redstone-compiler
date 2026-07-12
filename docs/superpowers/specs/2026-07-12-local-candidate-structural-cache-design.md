# Local Candidate Structural Cache

## Problem

Global PnR generates local placement candidates once per child instance even
when multiple child modules differ only by module name. In the two-bit counter,
the four D-latch instances consume about 242 seconds in total and the two clock
inverters consume about 14.5 seconds.

## Design

Add an in-memory cache scoped to one global PnR run. Cache entries compare the
exact graph topology, node kinds, port contract, placement dimensions, input
constraints, local placer configuration, and requested candidate count while
ignoring only `GraphModule.name`.

On a cache hit, clone the stored `LayoutCandidate` values and replace each
candidate's `module_name` with the current instance name. Exact port names stay
part of the identity, so the first implementation requires no port remapping.
This safely shares the four `d/en/q` D-latches and the two `clk/clk_n` clock
inverters, but does not merge `q_0_next` with a clock inverter.

The cache is memory-only and local to `generate_child_candidate_pools`; no
cross-run invalidation or serialization is required.

## Verification

- A unit test proves two identical modules with different names invoke the
  candidate generator once and return candidates labeled for each instance.
- A differing graph, port contract, or placer configuration must miss.
- Existing global PnR and local placer release tests continue to pass.
- Timing output from a controlled counter run confirms the candidate phase
  decreases from the 288.6-second baseline.

## Scope

Canonical graph isomorphism, normalized port-name remapping, disk caching, and
outer child-level parallelism are intentionally deferred.
