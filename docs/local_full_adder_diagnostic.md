# Bounded monolithic full-adder diagnostic

Run the local placer directly, without clustering, global packing/routing,
candidate regeneration retries, or NBT writes:

```powershell
cargo test --release --lib diagnose_monolithic_full_adder -- --ignored --nocapture
```

This ignored test is a measurement harness. **A passing test does not mean a
full-adder layout was found.** Read `LOCAL_FA result`, especially `valid`.
The fixed input is `buffered_full_adder_graph` with intermediate output
observations `c`, `i`, and `d` removed. All internal logic remains in one local
world. Its 24 nodes include three inputs and the public outputs `s` and `cout`.
Before searching, the harness checks the logical sum/carry truth signatures.
Any complete physical candidates are checked with the existing simulator-based
candidate verifier (fresh input cases and ascending/descending input transitions).
No successful physical candidate was produced in the measurements below.

## Reproducible controls

All runs use seed 42. These environment variables affect only this harness:

| Variable | Default | Allowed |
| --- | --- | --- |
| `LOCAL_FA_BEAM` | 64 | 1..512 |
| `LOCAL_FA_ROUTE_BEAM` | 8 | 1..32 |
| `LOCAL_FA_DEPTH` | 4 | 1..8 |
| `LOCAL_FA_WIDTH` | 2 | 1..10 |
| `LOCAL_FA_SIDE` | 10 | 1..20 |
| `LOCAL_FA_SECONDS` | 10 | 1..30 |
| `LOCAL_FA_SCHEDULE` | topological | topological, frontier, reconvergence |
| `LOCAL_FA_FLEXIBLE` | unset | `1` enables nonadjacent NOT placement and routed NOT inputs |
| `LOCAL_FA_NOT_SITES` | unset (unlimited) | 1..256; legal torch/support poses per parent, selected before cloning/routing |
| `LOCAL_FA_GRAPH` | buffered | buffered, nor9 |

The harness also accepts experimental schedule `defer_not`: move a NOT behind
an immediately following independent OR, while retaining a valid topological
order. This is diagnostic-only, not a new production schedule policy.

Dimensions are `(width, side, side)` in compiler `(x,y,z)` coordinates, where
`z` is height. Clear experimental environment variables before the baseline.
For a one-variable comparison:

```powershell
$env:LOCAL_FA_DEPTH = '8'
cargo test --release --lib diagnose_monolithic_full_adder -- --ignored --nocapture
Remove-Item Env:LOCAL_FA_DEPTH
```

`LocalPlacer::with_time_limit` is cooperative: it checks **between placement
steps**, not inside individual route expansions. One step can exceed the limit.
The initial planned-pin phase (not used by this harness) is outside this timer.
Interrupted partial worlds are discarded, and debug output reports
`time_limit_reached=true`, separately from an exhausted frontier. There is no
claim of a strict wall-clock deadline. Verification time is reported separately.

## Measurements, 2026-09-05

Release build, current working tree; timings exclude compilation and represent
single measurements, not statistically established performance bounds.

| Change from default | Search time | Outcome |
| --- | ---: | --- |
| None: 2x10x10, beam 64, depth 4 | 164 ms | Empty frontier at step 11/24, OR node 5 |
| Route depth 8 | 170 ms | Same OR node, step 11/24 |
| Width 4 (4x10x10) | 392 ms | Same OR node, step 11/24 |
| MinFrontier schedule | 104 ms | Same OR node, now step 7/24 |
| Flexible NOT placement/routing | 10,335 ms | Time limit after step 8/24; not a search-exhaustion result |

The failing node computes `(~a) | (a & b)` inside the first XOR cone. In the
default run it receives 38 parent states and produces no routes. Depth 8 leaves
59 parents at that node, still without a completed route. The diagnostic label
`RouteDepthExhausted` means some route frontier remained at the depth cap;
it does not prove that increasing depth will yield a route. Different depth
settings also change upstream candidate selection, so this is not a replay of
identical partial layouts.

The default run spends about 136 ms across the two later input-placement steps
(87 ms and 49 ms). They materialize 6,954 and 3,656 worlds before retaining 64.
Flexible mode makes the larger cost explicit: the first NOT expansion generates
10,073 candidates and retains 64, taking 4,012 ms for generation and 4,217 ms
including compaction/ranking. Later NOT steps similarly take seconds.

## Interpretation and next bounded experiment

A local-only failed attempt can already be subsecond with a small beam. This
does not establish fast successful synthesis or feasibility of 2x10x10 under
the current physical primitives. Expanding the full NOT placement domain is
expensive even with a bounded retained beam, because expansion precedes pruning.

The next useful work is to retain/replay the partial states immediately before
OR node 5 and inspect its fanout/access geometry, while separately bounding
NOT site/route generation before materializing thousands of worlds. Keep the
fixed graph and simulator semantics unchanged for those comparisons. Avoid
claiming an impossible layout from sampled failures or treating seed sweeps as
a diagnosis. Existing truth/crosstalk checks must remain in force.

## Follow-up: fixed-prefix replay and bounded NOT sites

The local OR expansion originally searched only from its first input toward its
second. On a frozen prefix, depth 8 / route beam 32 found zero routes in that
direction and six in the reverse direction, all passing the structural isolation
check. Production combinational OR expansion now tries the reverse direction
when the first direction yields no raw route and the second input is a legal
diode source. It still applies the existing isolation filter. This is a bounded
fallback, not two unlimited searches or a relaxation of crosstalk rules.

`reverse_or_fallback_preserves_live_full_adder_signals` contains a fixed physical
prefix, independent of candidate generation. It requires the forward search to
fail and the fallback to produce a route that preserves all five live logic
signals and the new OR value across all eight input cases in the simulator.
The generated prefix replay (after the fallback change) also found 60/60
prefixes correct before the previously failing OR.

```powershell
cargo test --release --lib replay_full_adder_first_failed_join -- --ignored --nocapture
```

To replay the next failing join with the same partial worlds in both directions:

```powershell
$env:LOCAL_REPLAY_STEP = '12'  # zero-based
$env:LOCAL_FA_DEPTH = '8'
$env:LOCAL_FA_ROUTE_BEAM = '32'
cargo test --release --lib replay_full_adder_first_failed_join -- --ignored --nocapture
Remove-Item Env:LOCAL_REPLAY_STEP, Env:LOCAL_FA_DEPTH, Env:LOCAL_FA_ROUTE_BEAM
```

This second prefix had 16 states, 13 of which passed live-signal simulation.
Neither direction found a route even at depth 16 / beam 32. This highlights why
geometric route acceptance alone is not a behavioral proof: final simulator
verification remains necessary. A separate bounded experiment,
`repair_full_adder_xor_join`, re-expanded the last NOT from 10 frozen parents
into 30 alternatives; no XOR join followed (25 ms for the repair attempt).

NOT expansion now optionally uses `LocalPlacer::with_not_site_limit`. It first
enumerates lightweight legal torch/support poses, prioritizes directly drivable
supports, then ranks by distance from the source to the support, and keeps at
most the requested count. World cloning
and route generation happen only afterward. With no limit, enumeration order
and search coverage are unchanged. A site cap is a heuristic budget and can
discard a necessary placement; it does not prove completeness or equal solution
quality. Zero is accepted by the API and yields no NOT candidates. An inverter
regression checks that a small cap still preserves working routes and that all
returned routes in that fixture invert both input values in the simulator.
Pure geometric ranking initially missed directly drivable supports in that
fixture; prioritizing direct connections fixes that omission.

Measured bounded follow-ups (all seed 42, beam 64 unless specified):

| Configuration | Search time | Outcome |
| --- | ---: | --- |
| Reverse fallback, direct NOT, depth 8 / route beam 32 | 258 ms | Reached XOR join at step 13/24 |
| Flexible NOT, 32 sites, depth 4 / route beam 8 | 596 ms | Empty at step 11/24 |
| Flexible NOT, 32 sites, depth 8 / route beam 32 | 3,349 ms | Empty at step 13/24 |
| 9-NOR graph, direct NOT, depth 8 / route beam 32 | 300 ms | Empty at step 8/23 |
| 9-NOR graph, flexible NOT, 32 sites, depth 8 / route beam 32 | 511 ms | Empty at step 8/23 |
| Reconvergence schedule, direct NOT, depth 8 / route beam 32 | 207 ms | Empty at step 8/24 |
| Beam 256, direct NOT, depth 8 / route beam 32 | 1,071 ms | Empty at step 11/24 |
| Deferred NOT schedule, direct NOT, depth 8 / route beam 32 | 242 ms | Empty at step 13/24 |

For the identical first NOT input frontier and flexible depth-4 policy, the
32-site cap reduced expansion from 10,073 to 595 candidates and step time from
4,217 to 172 ms (about 24x in these individual measurements). This is a speed /
coverage tradeoff, not an equivalent exhaustive-search optimization. After the
direct-connection priority refinement, the same first NOT still produced 595
candidates in 172 ms; the full depth-4 run took 587 ms, and the depth-8 / route
beam-32 run took 3,609 ms with the same failure stages as above.

Validation after the follow-up: `cargo test --release --lib local_ -- --skip
test_generate --skip debug_` passed 52 tests (3 diagnostic tests ignored).
The ignored replay, repair, and full-adder measurements were run explicitly.

No complete 2x10x10 full-adder was found. The next unresolved problem is joint
geometry of the two XOR branches, including access to both endpoints after
their NOTs are placed. Repeatedly widening the whole search did not solve it in
these bounded experiments. Preserve the frozen regression and simulator
semantics when investigating earlier branch placement or larger local patterns.
