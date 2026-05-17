# Local Placer Improvement Notes

## Context

The current `LocalPlacer` in `src/transform/place_and_route/local_placer.rs` places a prepared `LogicGraph` by walking nodes in topological order.

The prepared graph is expected to contain only:

- inputs
- outputs
- `Not`
- `Or`

`prepare_place()` rewrites `And` and `Xor` into `Not`/`Or` forms before placement.

The placer works by keeping a queue of candidate `(World3D, node_positions)` states. For each graph node it generates possible placements/routes, then samples the candidate queue to keep the search bounded.

## Full Adder vs Buffered Full Adder

`full_adder_graph()` describes the adder as compact logic:

```text
s = (a ^ b) ^ cin
cout = (a & b) | (s & cin)
```

After `prepare_place()`, this becomes a decomposed `Not`/`Or` network. It is logically compact, but it does not give the local placer explicit intermediate routing points such as `a ^ b` or `a & b`.

`buffered_full_adder_graph()` manually introduces intermediate signals:

```text
c = a & b
i = a ^ b
d = i & cin
s = i ^ cin
cout = (a & b) | (s & cin)
```

The XOR parts are written in an already-buffered style:

```text
i = (~(c | ~a)) | (~(c | ~b))
s = (~(d | ~i)) | (~(d | ~cin))
```

These intermediate outputs (`c`, `i`, `d`, `s`) become explicit producer/consumer connection points through `Graph::merge()`. This is a workaround for the current compiler not yet having an optimization pass that inserts buffers/intermediate nodes automatically.

## Current Test State

Relevant tests live in `src/transform/place_and_route/local_placer.rs`.

- `test_generate_component_and_shortest` finds valid AND layouts and checks equivalence.
- `test_generate_component_xor_shortest` finds valid XOR layouts and checks equivalence.
- `test_generate_component_half_adder` currently generates an NBT sample, but does not assert equivalence for all sampled worlds.
- `test_generate_component_full_adder` is ignored with `cannot route last or gate inputs`.

The immediate failure mode to investigate is that local placement reaches a late gate in the full adder, likely the final AND/OR routing area, and the candidate queue becomes empty because no route candidates are generated.

## Goal

Improve the local placer so it can find at least one valid full adder layout, then make the search reach that solution faster and more reliably.

Success criteria:

- Enable a full adder placement test that finds at least one world.
- Verify the generated world by converting it back through `world3d_to_logic()` and checking equivalence.
- Keep the search bounded enough that the test can run in a practical amount of time.
- Preserve the existing AND/XOR behavior.

## Likely Work Areas

1. Improve routing candidate generation.

   `generate_or_routes()` currently explores redstone routes locally with a max step limit and sampling. When the queue dies, it is hard to tell whether the route is physically impossible or just pruned too aggressively.

2. Add better diagnostics.

   Log or expose which graph node caused the queue to become empty, the source/target positions being routed, and how many route candidates were rejected by conflict, short-circuit, bounds, or max-step limits.

3. Make sampling more structure-aware.

   Current random sampling can discard promising partial layouts. Prefer candidates with shorter routes, lower congestion, more open space around future fanout points, or fewer occupied cells near unresolved nodes.

4. Represent buffer/fanout pressure explicitly.

   Full adder placement stresses shared intermediate signals. The placer should either insert buffer points automatically or favor placements that leave routable access around high-fanout nodes.

5. Revisit graph preparation.

   The current buffered graphs manually expose intermediate nodes. A future pass should insert these nodes automatically so the placer does not depend on hand-authored buffered logic.

## Notes For Future Changes

Keep changes incremental. First make the current ignored full adder case produce one verified solution. After that, optimize speed and quality.

Avoid changing the world/NBT model while working on the search unless a placement bug clearly comes from block semantics. Most of the immediate risk appears to be in local routing, pruning, and search ordering.
