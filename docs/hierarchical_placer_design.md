# Hierarchical Placer Design Notes

## Context

The current `LocalPlacer` places a prepared `LogicGraph` by walking nodes in topological order and expanding a queue of `(World3D, node_positions)` candidates. This works for small combinational DAGs, but it has two structural limits:

- A cyclic circuit such as an RS latch has no topological order unless the cycle is broken by an explicit state or delay boundary.
- A larger combinational unit such as a full adder can fail late because early local choices consume the physical space needed by later routes.

The desired long-term direction is to keep local placement powerful for small units such as XOR, full adders, RS latches, and flip-flops, then compose those verified layouts at a higher level.

## What EDA Tools Usually Do

Real EDA systems generally do not try to solve the whole chip as one exact placement-and-routing problem. They split the problem into stages and use different algorithms at each stage.

Common strategies:

- **Hierarchy and macros.** Repeated or complex blocks are treated as macros with known area, pins, timing, and blockages. The top-level placer reasons about macro placement and pin access instead of re-solving each macro internally.
- **Packing or clustering.** FPGA flows often pack logic into blocks before placement. ASIC flows group standard cells, preserve hierarchy, or use macro placement before detailed standard-cell placement.
- **Global placement before detailed placement.** Global placement finds approximate positions while optimizing wirelength and congestion. Detailed placement legalizes positions and resolves local conflicts.
- **Global routing before detailed routing.** Global routing estimates paths and congestion over coarse routing regions. Detailed routing then assigns concrete wires, vias, and tracks.
- **Rip-up and reroute.** Routing failures are handled by removing problematic routes, increasing congestion costs, and trying alternative paths.
- **Timing-driven cost.** Placement and routing are not optimized only for area. Timing, fanout, congestion, pin accessibility, and routability all influence cost.
- **Pareto candidate retention.** Tools avoid keeping only the smallest local solution because a slightly larger block with better pin access can improve the global result.

Relevant references:

- OpenROAD documents a staged physical design flow including global placement, detailed placement, clock/tree steps, global routing, detailed routing, and congestion-aware repair loops.
- VPR/F4PGA-style FPGA flows split the process into logic optimization, packing, placement, routing, and timing analysis.

The important lesson for this project is that exact global optimality is normally traded for staged optimization with feedback between stages.

## Optimization Position

The compiler should not aim for a single global optimum over the entire redstone circuit. That problem combines:

- block placement,
- redstone/cobble/torch/repeater legality,
- signal directionality,
- wire routing,
- timing and delay,
- fanout,
- feedback loops,
- spatial constraints.

This is too large for exact search except for tiny examples.

Instead, the project should aim for:

- **near-exact or exhaustive search inside small units,**
- **Pareto-optimal layout candidates for those units,**
- **heuristic but inspectable composition at the top level.**

This keeps the local placer valuable while preventing top-level placement from becoming an unbounded monolithic search.

## Proposed Architecture

Use a hierarchical place-and-route stack:

```text
LogicGraph
  -> sequential/cyclic normalization
  -> primitive and SCC matching
  -> unit layout candidate generation
  -> layout candidate library
  -> top-level floorplanning
  -> inter-unit routing
  -> simulation/equivalence verification
```

### Sequential And Cyclic Normalization

Cycles need explicit semantics before placement.

The graph layer should distinguish:

- **combinational edges,** which participate in ordinary logic evaluation,
- **state/delay edges,** which cross a latch, flip-flop, repeater delay, or other time boundary.

Topological ordering should operate only on the combinational graph. A graph is legal when:

- combinational edges form a DAG after state/delay edges are ignored, or
- a pure combinational SCC is recognized as a supported primitive such as an RS latch macro.

Unsupported pure combinational SCCs should fail with a clear diagnostic rather than reaching `toposort(...).unwrap()`.

### Primitive And SCC Matching

Before placement, match subgraphs into units:

- simple gates: `Not`, `Or`, decomposed `And`, decomposed `Xor`,
- arithmetic units: half adder, full adder,
- feedback units: RS latch, D latch, flip-flop,
- routing/fanout helpers: buffers, repeaters, splitters.

SCC matching should first support known patterns rather than arbitrary cyclic synthesis. For example:

- cross-coupled NOR/NAND -> RS latch primitive,
- latch plus gating -> D latch primitive,
- master/slave latch pair -> flip-flop primitive.

An arbitrary SCC placer can be added later, but known primitive matching gives a safer first path.

### Layout Candidate

The local placer should eventually return layout candidates instead of only final worlds.

Suggested structure:

```text
LayoutCandidate {
  unit_id
  world_fragment
  bbox
  ports
  occupied_cells
  blocked_cells
  cost
  timing
  source_graph_nodes
}
```

Ports should describe logical connectivity and physical access:

```text
Port {
  name
  direction: input | output | bidirectional
  signal_kind: soft | hard | torch | repeater
  positions
  preferred_route_directions
  delay
}
```

The key is that top-level routing must not inspect the internals of a unit except for obstacles and exposed ports.

### Candidate Cost

Keep multiple candidates per unit using a Pareto frontier. Do not select only the smallest layout.

Useful cost dimensions:

- bounding box volume,
- footprint area,
- block count,
- redstone count,
- max delay,
- input-to-output delay vector,
- number of exposed ports,
- port spacing,
- distance from ports to bbox boundary,
- free space around ports,
- estimated congestion around the unit,
- symmetry and rotation support,
- route failure history.

A candidate is dominated only if another candidate is no worse in every important dimension and better in at least one.

### Unit Layout Generation

The existing `LocalPlacer` can become the first implementation of unit layout generation.

For combinational DAG units:

- keep the current topological search,
- generate many candidates,
- verify candidates by converting back through `world3d_to_logic()` where possible,
- normalize candidates by rotation/translation,
- retain Pareto candidates.

For feedback units:

- use predefined macro templates first,
- add small bounded SCC search later,
- validate behavior with simulator input sequences instead of only graph isomorphism.

For example, an RS latch should be validated across sequences such as:

```text
S=0 R=0 -> holds prior state
S=1 R=0 -> sets Q
S=0 R=1 -> resets Q
S=0 R=0 -> holds updated state
```

### Top-Level Floorplanning

The top-level placer chooses one layout candidate for each unit and assigns each candidate a position and orientation.

Initial implementation can be simple:

- place units in topological order of the unit graph,
- reserve spacing around ports,
- prefer shorter estimated inter-unit routes,
- try several candidate combinations,
- backtrack when routing fails.

Later improvements:

- simulated annealing over unit positions and candidate choices,
- partition-based floorplanning,
- timing-driven candidate selection,
- congestion maps,
- critical path weighting.

### Inter-Unit Routing

Top-level routing connects unit ports.

It should be separate from unit placement and should understand:

- occupied cells from all unit fragments,
- blocked cells around sensitive redstone/torch interactions,
- source port propagation type,
- target port accepted propagation type,
- route delay,
- fanout handling.

A staged router is preferable:

1. **Global route:** choose coarse channels or route regions between unit bboxes.
2. **Detailed route:** instantiate redstone/cobble/repeater blocks.
3. **Repair:** rip up failed or congested routes and retry with higher costs.

This mirrors EDA global/detailed routing without requiring the full complexity at the start.

## Relation To Current Code

Current useful pieces:

- `Graph` already stores directed edges and has `has_cycle()`.
- `LocalPlacer` already generates concrete `World3D` candidates for small `Not`/`Or` graphs.
- `LocalPlacerDebug` already records route failure information.
- `SubGraph`, `ClusteredGraph`, and partitioning code provide a starting point for unit grouping.
- `WorldGraph` has a `routings` set, which is conceptually close to separating logic blocks from routing blocks.

Current gaps:

- `Graph::topological_order()` unwraps cycle errors.
- Edges do not encode combinational vs state/delay semantics.
- `LocalPlacer` assumes all node inputs have already been placed.
- `LocalPlacer` returns full worlds, not reusable layout candidates with ports.
- Equivalence checks are combinational and do not validate sequential behavior.
- Clustered graph nodes currently preserve grouping but do not define placement ports, bboxes, or obstacles.

## Suggested Implementation Roadmap

### Phase 1: Make DAG Assumptions Explicit

- Add clear cycle diagnostics before topological placement.
- Add a safe topological order API that returns `Result`.
- Document that current local placement supports only combinational DAGs.

This prevents accidental cyclic graphs from failing deep inside placement.

### Phase 2: Introduce LayoutCandidate

- Wrap current local placer output in a candidate structure.
- Infer basic bbox and occupied cells from `World3D`.
- Expose input/output node positions as preliminary ports.
- Add candidate cost calculation.
- Retain multiple candidates instead of immediately sampling down to final worlds.

This phase does not require changing graph semantics yet.

### Phase 3: Unit Library For Combinational Blocks

- Generate and store candidate sets for XOR, half adder, full adder, and common buffered forms.
- Keep Pareto candidates.
- Add normalization so equivalent translated/rotated candidates collapse.
- Add tests that assert at least one verified candidate exists for each unit.

This turns local placement into a reusable unit optimizer.

### Phase 4: Top-Level Candidate Composition

- Build a unit graph from matched subgraphs.
- Select candidates and place them into a larger `World3D`.
- Route between exposed ports.
- Backtrack candidate selection when routing fails.

Start with small examples such as two connected XORs or a full adder built from smaller candidates.

### Phase 5: Sequential Primitive Support

- Add primitive specs for RS latch and flip-flop.
- Add simulator-based behavioral tests.
- Support macro/template layouts first.
- Mark sequential boundaries so the combinational topological graph remains legal.

This is the first safe way to support cycles.

### Phase 6: SCC Placement

- Detect SCCs in the combinational graph.
- Match known SCC patterns to sequential primitives.
- For unmatched small SCCs, optionally run a bounded SCC layout search inside a fixed bbox.
- Fail unsupported SCCs with useful diagnostics.

Arbitrary SCC placement should be a late feature, not the foundation.

## Design Principle

The local placer should remain strong, but its responsibility should narrow:

```text
LocalPlacer = small unit optimizer
TopLevelPlacer = candidate selection + floorplan
TopLevelRouter = inter-unit physical connectivity
Verifier = combinational equivalence or sequential simulation
```

This design gives up exact global optimality, but it makes the compiler scalable, debuggable, and able to handle cyclic units through explicit semantics.

