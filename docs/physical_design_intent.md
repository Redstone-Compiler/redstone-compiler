# Physical design intent and local cell recipes

## Status

This document records the design direction for human-authored physical intent,
reusable local-cell layout recipes, and the physical representations produced by
placement and routing.

An initial, deliberately small physical subset is implemented in the trailing
`physical {}` section of Routable RCIR:

```text
physical {
  region "state" box [0, 0, 2] [63, 47, 9];
  region "clock_keepout" box [24, 0, 0] [31, 47, 9];
  require "state-master" instance "q_0_master" inside "state";
  require "master-layers" instance "q_0_master" layer 2..6;
  lock "fixed-slave" instance "q_0_slave" at [20, 10, 4];
  priority "clock-priority" net "clk" 100;
  require "clock-keepout" net "clk" avoid "clock_keepout";
}
```

The legacy `--intent design.rclayout` option remains an explicit command-line
override, but snapshots and standalone Routable inputs store the same typed
intent inside `routable.rcir`. Hard placement constraints are
applied before routing, net priority affects routing order, and hard avoid
regions reject intersecting routes. Source, resolved constraints, and a
satisfaction report are stored under `intent/` in the snapshot. Larger syntax
examples below remain exploratory and are not all accepted by the parser.

## Decision summary

- Physical constraints are not a third compileable RCIR stage. They are a
  namespaced section of a Routable RCIR document and resolve into a separate
  `PhysicalIntent` model before physical synthesis.
- The physical flow changes character from semantic lowering to constrained
  optimization. It combines a Routable design, target rules, cell recipes,
  per-design intent, and search policy to construct a physical solution.
- A reusable local-cell recipe is distinct from per-design floorplanning intent.
  It belongs to the target cell library and generates verified layout
  candidates that can be cached and reused.
- Concrete candidate sets, placed designs, global route guides, and routed
  designs are physical intermediate representations. The intent that guides
  their construction is a specification, not the solution IR itself.
- Hard requirements, soft preferences, and exact or relative locks have
  different semantics and must not be collapsed into one untyped weight.
- Three-dimensional intent should initially use axis-aligned regions, layer
  ranges, faces, corridors, and relative relations. Exact coordinates are an
  escape hatch rather than the primary authoring model.
- `compact` is a multi-objective optimization request. A smaller raw bounding
  box is not necessarily a smaller composable cell when electrical blockages,
  routing access, delay, and required clearance are included.
- Local candidate generation should retain a Pareto frontier. The globally best
  candidate may be slightly larger locally but have much better port access or
  routing behavior.
- Constraints must refer to stable Routable definitions, instances, nets, and
  public ports. They must not refer to graph indices or nodes created by
  `prepare_place()`.
- Diagnostics must distinguish invalid or contradictory intent from search
  exhaustion. A heuristic placer failing to find a solution does not prove that
  the constraints are unsatisfiable.

## The compiler has inputs and solutions, not one linear IR ladder

The coordinate-free circuit path remains a conventional lowering pipeline:

```text
Verilog
  -> LogicalDesign
  -> RoutableDesign
```

Physical synthesis consumes several inputs and searches for a solution:

```text
RoutableDesign -----------+
TargetRules --------------+
CellLayoutLibrary --------+--> ResolvedPhysicalProblem
PhysicalIntent -----------+              |
SearchPolicy -------------+              v
                                  local candidate search
                                            |
                                            v
                                  LayoutCandidateSet
                                            |
                                            v
                                      PlacedDesign
                                            |
                                            v
                                  GlobalRouteGuide
                                            |
                                            v
                                      RoutedDesign
```

`LogicalDesign` and `RoutableDesign` are compileable circuit IRs. A physical
intent file is deliberately partial and cannot define a circuit on its own. The
derived objects on the right are physical IRs because each records increasingly
committed implementation decisions.

An internal normalized representation may still be named `ConstraintSet` or
`ConstraintIr`. That is an implementation detail, just as a compiler may build
an internal IR for a query or configuration language. Public architecture and
file-format documentation should reserve “compileable IR stage” for a complete
restart point.

### Set-of-solutions interpretation

Let `S0` be every target-legal physical implementation of a Routable design.

```text
hard constraints:  S1 is a subset of S0
soft preferences:  impose a preference order on S1
PnR:               select one member of S1
```

A hard constraint refines the set of allowed implementations, but it does not
itself describe one implementation. A placement becomes concrete only when the
candidate, transform, and origin of every instance have been chosen and routes
have concrete paths.

This distinction is more prominent in hardware than in ordinary software
compilation because a single structural netlist has a very large number of
legal physical implementations. Placement may also be revised after routing
feedback, so physical compilation is not necessarily monotonic or one-way.

## Proposed artifact roles

Names and extensions remain tentative, but the responsibilities should remain
separate.

| Artifact | Role |
| --- | --- |
| `*.rcir` | Logical or Routable circuit meaning |
| target rules | Block legality, propagation, orientation, and technology capabilities |
| `*.rcell` | Reusable cell implementations, physical contracts, and local layout recipes |
| `*.rclayout` | Optional per-design floorplan and routing intent |
| candidate cache | Verified local physical candidates and their metrics |
| snapshot physical data | Selected candidates, origins, route guides, routes, and blocks |

It may be useful to share lexical conventions and parser utilities across these
formats. Sharing syntax must not merge their semantic models or validators.

## Relevant precedents

### LEF/DEF and OpenROAD

The LEF/DEF region model distinguishes hard `FENCE` regions from soft `GUIDE`
regions and assigns groups of components to regions. A fence constrains members
to the region and excludes non-members; a guide expresses a preference that may
be overridden by other physical objectives.

Reference:

- [LEF/DEF 5.8 Regions](https://coriolis.lip6.fr/doc/lefdef/lefdefref/DEFSyntax.html)

OpenROAD exposes the same kinds of intent at several physical stages:

- macro guidance regions;
- macro halos and placement blockages;
- area, outline, wirelength, guidance, and fence costs;
- pin placement on a named edge or edge interval;
- pin grouping and ordering;
- global routing capacity adjustments and generated route guides.

References:

- [OpenROAD hierarchical macro placement](https://openroad.readthedocs.io/en/latest/main/src/mpl/README.html)
- [OpenROAD pin placer](https://openroad.readthedocs.io/en/latest/main/src/ppl/README.html)
- [OpenROAD global routing](https://openroad.readthedocs.io/en/latest/main/src/grt/README.html)

The important lesson is that regions, port faces, halos, and guides are stable
domain concepts. A particular macro placer or router is an implementation of
those concepts, not part of their meaning.

### VPR floorplanning constraints

VPR assigns groups of primitives to partitions and constrains each partition
to a union of rectangular regions. In multi-layer architectures, regions carry
`layer_low` and `layer_high`. Packing and placement both honor the constraints.

Reference:

- [VPR placement constraints](https://docs.verilogtorouting.org/en/latest/vpr/placement_constraints/)

This is a strong precedent for initially modeling Redstone 3D intent as unions
of boxes and discrete layer ranges rather than a general solid-geometry or
arithmetic constraint language.

### AMD Vivado relative placement and Pblocks

Vivado relationally placed macros use relative coordinates inside a reusable
macro. The placer may choose the macro's global origin, or a separate origin
constraint may fix it. Pblocks assign hierarchical logic to device regions and
may contain routing or exclude unrelated placement.

References:

- [Vivado relative locations](https://docs.amd.com/r/2024.1-English/ug903-vivado-using-constraints/Assigning-Relative-Locations)
- [Vivado fixed RPM origin](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Assigning-a-Fixed-Location-to-an-RPM)
- [Vivado Pblock constraints](https://docs.amd.com/r/2023.1-English/ug905-vivado-hierarchical-design/Pblock-Constraints)

Vivado also explicitly separates relative placement from routing: fixing the
relative locations of logic does not guarantee use of identical routing
resources. Redstone local-cell placement and inter-cell routing should likewise
have separate contracts.

### Intel Quartus LogicLock

LogicLock regions can be nested hierarchically, reserve resources, and retain
member locations relative to a region when the region moves. Placement can be
back-annotated for reuse.

Reference:

- [Intel LogicLock region definition](https://www.intel.com/content/www/us/en/programmable/quartushelp/20.1/reference/glossary/def_logiclock_reg.htm)

This motivates exporting a successful Redstone layout as relative local locks
or reusable physical candidates rather than requiring users to copy absolute
world coordinates.

### ALIGN analog layout constraints

ALIGN applies constraints separately at each circuit hierarchy. Its vocabulary
includes ordering, alignment, enclosure, spreading, grouping, same-template
requirements, approximate port location, symmetry, and net criticality.

References:

- [ALIGN constraint reference](https://align-analoglayout.github.io/ALIGN-public/notes/const.html)
- [ALIGN system paper](https://arxiv.org/abs/2008.10682)

ALIGN is particularly relevant to local Redstone cells: it demonstrates that a
small domain vocabulary can guide hierarchical block assembly without requiring
the author to provide every coordinate.

### Relative floorplanning and constraint hierarchies

Sequence-pair floorplanning represents left/right/above/below topology without
first fixing absolute coordinates. A solver derives coordinates while
preserving the chosen relations.

- [Sequence-pair floorplanning](https://doi.org/10.1109/43.552084)

Constraint hierarchy research formalizes required constraints and preferences
at different strengths. Required constraints must hold; weaker constraints are
satisfied when they do not conflict with stronger ones.

- [Constraint Hierarchies](https://constraints.cs.washington.edu/theory/hierarchies-92.html)
- [Cassowary constraint solver report](https://constraints.cs.washington.edu/solvers/cassowary-tr.html)

The Redstone language can adopt these semantics without adopting Cassowary as
the physical solver. Redstone placement and routing contain discrete,
non-linear legality decisions that stage-specific search algorithms must handle.

Constrained floorplanning research also treats approximate relative placement
as input to a semi-automatic floorplanner and emphasizes feasibility feedback
when requested constraints cannot be met.

- [Constrained Modern Floorplanning](https://doi.org/10.1145/640000.640030)

### Hierarchical and three-dimensional physical planning

Hier-RTLMP uses logical hierarchy and dataflow to construct a multi-level
physical hierarchy before macro placement. This supports grouping by stable
logical origin instead of depending only on generated name patterns.

- [Hier-RTLMP](https://arxiv.org/abs/2304.11761)

Three-dimensional IC floorplanning commonly separates or coordinates layer
assignment with intra-layer floorplanning. It does not require users to control
every coordinate in an unconstrained continuous 3D space.

- [3D floorplanning with thermal-via planning](https://doi.org/10.1145/1123008.1123048)

For Redstone, `z` should initially behave as a discrete and comparatively
expensive layer dimension, while `x` and `y` carry most floorplanning freedom.

## Physical intent semantics

### Requirements, preferences, and locks

The public model should distinguish three concepts.

```text
require P;                    # P must hold
prefer P strength strong;    # violation is allowed and scored
lock instance i at ...;      # reuse an exact or relative solved assignment
```

Hard requirements determine legality. Soft preferences affect ranking. Locks
commit decisions and are normally generated from a known-good physical result
or written deliberately as an exact escape hatch.

Preferences should use lexicographic strength classes before numeric weights:

```text
required > strong > medium > weak
```

A numeric weight may break ties within one strength. A raw weighted sum across
unrelated metrics is difficult to interpret because volume, height, delay, and
congestion have different scales.

### Region semantics

Containment, exclusivity, and guidance should be explicit instead of hidden in
one overloaded keyword.

```text
require group state inside region logic;
reserve region logic for group state;
prefer group state inside region preferred_state strength strong;
```

- `inside` constrains the selected objects.
- `reserve` prevents unrelated objects from occupying the region.
- a preferred `inside` relation acts as a placement guide.

### Search policy is not design intent

The following affect how a solution is sought, not which physical results are
semantically valid:

- random seed;
- beam width and candidate limits;
- iteration and expansion budgets;
- shelf, grid, layered, or Free3D heuristic selection;
- tracing and progress settings.

They belong to a separate search block or compiler configuration. Reproducible
physical output requires recording them, but a constraint must not mean “run
the shelf placer.”

## Coordinate spaces and 3D geometry

### Local and global frames

A local cell recipe uses a normalized frame relative to the candidate. A
global floorplan assigns the candidate an origin and an allowed transform.

```text
local candidate coordinates
  -> normalize bbox minimum to the local origin
  -> apply legal target transform
  -> translate to the global world origin
```

Relative local offsets may be signed even if the final `World3D` representation
uses non-negative coordinates. The final assembler can translate the entire
legal solution into the non-negative world domain.

### Initial geometry vocabulary

The first version needs only a small set of shapes and relations:

- point and point set;
- axis-aligned box;
- union of non-overlapping boxes;
- `x`, `y`, or `z` slab;
- candidate face and face interval;
- routing corridor;
- keepout or reserved region;
- `inside`, `outside`, `intersects`, and `disjoint`;
- `above`, `below`, and ordering along an axis;
- alignment on a minimum, center, or maximum plane;
- distance or gap range;
- same-layer or allowed-layer range.

Arbitrary polygons, arbitrary Boolean expressions, and general geometric
algebra are not required for the first useful version.

### Legal transforms

Redstone is not invariant under arbitrary 3D rotation or reflection. Gravity,
support blocks, torch attachment, repeater direction, and vertical propagation
make many geometric transformations invalid.

The initial transform set should be conservative:

- translation;
- yaw rotation around the vertical axis when the target implementation proves
  the rotated block states are equivalent;
- no vertical flip;
- no mirror or pitch/roll rotation unless target-specific verification declares
  it safe.

## Local cell libraries and layout recipes

### A local cell has four distinct layers

1. **Implementation graph.** Stable named Routable primitives and nets that
   implement the cell behavior.
2. **Physical contract.** Public ports, access requirements, isolation,
   allowed transforms, timing, and blockage semantics.
3. **Layout recipe.** Partial spatial relations and optimization objectives
   used to generate candidates.
4. **Physical candidate or template.** Concrete blocks, routes, ports, bbox,
   obstacles, and measured costs.

Only the first and fourth are naturally called implementation IRs. The contract
and recipe are specifications consumed during candidate synthesis.

### Implementation variants

Compactness may require changing the target implementation, not merely packing
the same prepared graph into a smaller box. A target cell may therefore expose
several equivalent variants:

```text
std.xor
  -> xor.nor_network
  -> xor.buffered
  -> xor.compact_horizontal
  -> xor.low_delay
```

Each variant may have a different stable internal Routable subgraph and layout
recipe. Variant selection is target mapping or local implementation selection;
it must preserve the public operation's behavior and ports.

This avoids making nodes inserted by `prepare_place()` part of the public
contract. A user who needs internal control refers to named objects in a stable
implementation variant, not transient graph indices.

### Physical contract

A generated local candidate must publish enough information for global PnR to
avoid inspecting its internal logic:

```text
CellPhysicalContract {
  public ports and directions
  accepted connection and propagation types
  one or more routing access positions per port
  occupied cells
  electrically or physically blocked cells
  required halo or clearance
  legal transforms
  input-to-output delay information
  sequential/isolation requirements
}
```

Named internal anchors may be exposed when a recipe genuinely needs them, but
they must be deliberate target-library identities with a defined lifetime.

### Illustrative local recipe

```text
rcell 1;
target redstone-v1;

cell compact_xor implements std.xor {
  variant xor.buffered;

  ports {
    input  a prefer face west;
    input  b prefer face west;
    output y prefer face east;

    prefer order { a, b } along z;
  }

  sketch {
    require order { input_stage, merge, output_stage } along x;
    prefer align { input_stage, merge, output_stage }
      plane y-center
      strength medium;

    reserve corridor main {
      axis x;
      width 2;
    }

    prefer net result through main strength strong;
  }

  optimize pareto {
    minimize effective_volume strength strong;
    minimize height strength strong;
    maximize port_access strength medium;
    minimize max_delay strength medium;
    minimize block_count strength weak;
  }

  retain candidates 12;
}
```

The author describes a recognizable shape and interface without specifying all
block positions. An exact block template remains available as a separate
variant or lock when automation cannot match an expert layout.

## Compactness is not a bounding-box constraint

### Three separate concepts

The language and implementation must not conflate these statements:

```text
require bbox <= (8, 6, 4);       # packaging requirement
minimize effective_volume;       # quality objective
search boxes up to (16, 16, 6);  # finite search domain
```

- A bbox requirement answers whether the candidate fits a required slot.
- An objective asks the generator to prefer smaller solutions.
- A search limit bounds computation and says nothing about whether larger
  candidates would be semantically acceptable.

Using a hard bbox only to make search finite creates accidental semantics and
poor diagnostics.

### Effective size

Raw occupied bbox is not sufficient. Consider two candidates:

```text
candidate A: 5 x 5 x 3, several exposed port accesses, small halo
candidate B: 4 x 4 x 3, buried ports, large routing and isolation halo
```

Candidate B is smaller in isolation but may consume more space in a composed
design. Useful local metrics include:

- bbox width, depth, height, footprint, and volume;
- occupied block count and material-specific block counts;
- blocked-cell count and effective blockage volume;
- required electrical isolation halo;
- number and distribution of port access points;
- distance from each port access to candidate faces;
- free routing space around each port;
- estimated escape-route length and bend count;
- fanout accessibility;
- input-to-output delay vector and maximum delay;
- number of legal transforms;
- route and verification failure history.

An initial effective volume may be defined from the bounding box of:

```text
occupied cells
union blocked cells
union required port-access keepouts
```

The exact formula can evolve, so snapshots should also retain the individual
metrics rather than only one aggregate score.

### Pareto retention

Local generation must not keep only the candidate with the smallest scalar
cost. For example:

```text
candidate A: volume 80, height 3, port accesses 1
candidate B: volume 92, height 3, port accesses 5
candidate C: volume 88, height 4, shortest delay
```

None necessarily dominates the others. Global placement and routing should be
able to choose among them based on the surrounding instances and nets.

The candidate library should retain a bounded Pareto frontier, optionally with
diversity sampling among candidates with similar metrics.

### Compact-search strategy and optimality claims

A local generator may explore bounding boxes in increasing cost order:

```text
5 x 5 x 3 -> no candidate found
6 x 5 x 3 -> two candidates
6 x 6 x 3 -> eight candidates
7 x 6 x 3 -> candidate with much better port access
```

If all smaller domains were searched exhaustively, the compiler may prove a
minimum under the modeled rules. If sampling or heuristic pruning was used,
the result is only `best_found`.

Candidate metadata should record:

- search domain and budget;
- whether each smaller bound was exhausted;
- heuristic or exhaustive mode;
- best-found objective vector;
- verification status;
- whether any optimality claim is justified.

Local search can be expensive because a verified candidate library is reusable
across many designs. This makes offline or cached exhaustive search practical
for small common cells even when per-design global PnR remains heuristic.

## Per-design global physical intent

Global intent operates on stable Routable instances, groups, ports, and nets.
It selects and positions local candidates without reaching into their internal
blocks.

An illustrative document is:

```text
rclayout 1;
for design counter fingerprint "...";

region logic = box x 0..63 y 0..47 z 0..7;
region clock_lane = box x 0..63 y 20..23 z 0..2;

floorplan module counter {
  group bit0 = {
    q_0_next,
    q_0_master,
    q_0_slave
  };

  require bit0 inside logic;
  require q_0_master below q_0_slave gap 2..6;

  prefer align { bit0, bit1 }
    along x
    strength medium;
}

routing {
  prefer net clk through clock_lane strength strong;
  require net reset avoid state_keepout;
}

search {
  preset thorough;
  seed 42;
}
```

The design fingerprint prevents an old file from silently binding to a changed
Routable design. A reader must reject unresolved selectors and empty groups by
default.

### Stable selectors

The first version should prefer explicit stable identities and named groups.
Generated-name globs and regular expressions are convenient but brittle.

Later selectors may use provenance and semantic tags:

```text
instances from logical cell state ordered by source bit;
instances connected to nets of class clock;
instances with origin role slave_latch;
```

Such selectors require precise expansion semantics when one logical object maps
to multiple Routable objects. They should follow, not precede, stable Routable
IDs and provenance.

### Definition-wide and instance-specific local intent

A recipe attached to a cell definition affects every use and forms part of the
candidate-library cache key. An instance-specific preference should normally
filter or rank candidates during global PnR rather than regenerate the cell.

Explicit specialization may be added when one instance genuinely needs a
different local implementation:

```text
specialize instance critical_state using cell recipe low_delay_latch;
```

Specialization must be deliberate because it reduces reuse and makes physical
identity more complex.

## Routing intent, guides, and concrete routes

Routing has three distinct representations:

```text
RouteIntent
  -> GlobalRouteGuide
  -> DetailedRoute
```

- Route intent is user-authored and broad: prefer a corridor, avoid a region,
  constrain layers, prioritize a net, or bound delay.
- A global route guide is derived geometry assigned to a net or net branch.
- A detailed route is the concrete Redstone/cobble/repeater path and block set.

A broad user guide should affect routing cost or allowed regions. It must not be
mistaken for a complete route. Generated guides and exact paths belong in the
physical snapshot.

Potential routing vocabulary includes:

- `through` or `prefer through` a corridor;
- `avoid` or `prefer avoid` a region;
- allowed layer range;
- maximum vertical transitions;
- maximum delay or repeater count;
- route priority or criticality;
- isolation and no-contact regions;
- fanout trunk preferences.

## Internal representation and lowering

The exact Rust types may evolve, but the ownership model should resemble:

```text
PhysicalIntent {
  design_fingerprint
  named_regions
  groups
  hard_constraints
  soft_constraints
  locks
  routing_intent
}

CellLayoutLibrary {
  target
  recipes_by_definition
  exact_templates
}

ResolvedPhysicalProblem {
  routable_design
  target_rules
  resolved_cell_recipes
  resolved_physical_intent
  search_policy
}
```

Parsing is followed by explicit binding and normalization:

1. parse syntax while retaining source spans and explicit constraint IDs;
2. verify the target and design fingerprint;
3. resolve definitions, instances, ports, nets, groups, and regions;
4. type-check every constraint against its target object kind;
5. normalize syntactic sugar into a small set of typed predicates and
   objectives;
6. split constraints by consumer stage;
7. run cheap static contradiction and capacity checks;
8. generate and filter local candidates;
9. select candidates and solve global placement;
10. lower broad route intent into global guide costs or restrictions;
11. run detailed routing and verification;
12. emit a per-constraint satisfaction report.

The public language must not expose which solver data structure implements a
predicate. The same `near` relation may be used by a shelf candidate generator,
a force-directed placer, simulated annealing, or another future algorithm.

## Diagnostics and satisfaction reporting

Every constraint should have a stable ID and source span. Compilation output
should report at least:

```text
constraint c4: satisfied
constraint c7: soft violation, distance 15 exceeds preferred 12
constraint c9: no satisfying candidate found within local search budget
constraint c12: placement search exhausted
constraint c16: routing search exhausted
```

Failures fall into distinct categories:

1. invalid syntax or type;
2. stale design fingerprint or unresolved target;
3. empty or ambiguous selector;
4. statically contradictory hard constraints;
5. insufficient region capacity under a proven lower bound;
6. no local candidate found within a stated budget;
7. global placement search exhaustion;
8. routing search exhaustion;
9. completed physical result that fails behavioral or physical verification.

Only cases with a proof should be described as unsatisfiable. Heuristic search
failure is `not found within budget`.

Snapshots should record:

- every normalized constraint and objective;
- the object IDs to which it resolved;
- whether it was hard, preferred, or locked;
- satisfaction and violation magnitude;
- the stage that rejected it;
- the search budget and policy;
- the selected candidate metrics and global cost breakdown.

## Relation to the current code

The repository already contains useful starting points.

### Local input constraints

`LocalPlacerInputConstraints` maps an input name or graph node ID to an allowed
set of exact positions. It can evolve toward typed position domains:

```text
PositionDomain =
    ExactPoints
  | Region
  | Face
  | FaceInterval
  | LayerRange
```

Public files should bind by stable cell ports or implementation anchors. Graph
node IDs remain internal adapter identities.

### Unit candidate configuration

`UnitCandidateConfig` already combines a search dimension, local placer config,
input constraints, candidate limits, and sampling limits. Future code should
separate:

- semantic cell contract and recipe;
- resolved hard/soft local constraints;
- non-semantic search policy and budget.

This prevents `DimSize` from accidentally becoming a hard physical requirement
when it was intended only to bound search.

### Layout candidates

`LayoutCandidate` already owns:

- the world fragment;
- bbox;
- physical ports;
- occupied cells;
- blocked cells;
- candidate cost.

This is the correct physical boundary. It should be extended with richer
metrics, legal transforms, timing, stable Routable definition provenance, and
candidate generation/verification metadata.

`LayoutCandidateCost` currently contains only block count and bbox volume. It
should become a metric vector suitable for Pareto filtering rather than the
only scalar definition of quality.

### Global placement and routing

`PlacementCostBreakdown` already separates placement volume, XY footprint,
height span, estimated wire length, vertical distance, and congestion. Those
metrics provide natural targets for soft global objectives.

Shelf, grid, register-specific, layered, and Free3D placement heuristics are
solver policies. They should all consume the same resolved constraint model.

The router currently owns concrete route strategies and validation modes.
Broad route intent should lower to allowed regions and cost adjustments before
concrete paths are generated.

## Suggested implementation order

### Phase 1: preserve the conceptual boundary

- Keep Logical and Routable RCIR coordinate-free.
- Add stable Routable definition, instance, port, and net IDs.
- Add provenance from logical origins to Routable objects.
- Name the optional side input `PhysicalIntent`, not a new circuit stage.

### Phase 2: enrich candidate metadata without a public language

- Expand candidate metrics.
- Compute effective blockage and port-access metrics.
- Record legal transforms.
- Retain a bounded Pareto frontier.
- Record search and verification metadata in snapshots.

This validates the optimization model before freezing syntax.

### Phase 3: reusable local cell recipes

- Define stable target cell implementations and implementation variants.
- Add per-definition physical contracts.
- Add local region, face, order, alignment, gap, and corridor predicates.
- Separate result constraints from search-domain limits.
- Cache verified candidate libraries by target, implementation, recipe, and
  compiler version.

### Phase 4: per-design floorplan intent

- Add named regions and explicit instance groups.
- Add hard containment and reservation.
- Add soft guidance, relative ordering, alignment, layer, and distance
  preferences.
- Add exact and relative locks imported from snapshots.
- Emit constraint satisfaction reports.

### Phase 5: route intent and feedback

- Add route corridors, avoid regions, layer ranges, and priority.
- Emit derived global route guides.
- Feed repeated route failures into candidate and placement ranking.
- Preserve broad input intent separately from derived guides and detailed
  paths.

## Minimum useful v1

A first public physical-intent version should remain small:

- explicit design fingerprint and target;
- stable explicit selectors and named groups;
- named regions as unions of axis-aligned boxes;
- discrete `z` layer ranges;
- `require`, `prefer` with strength, and `lock`;
- containment, reservation, ordering, alignment, gap, and distance;
- local port face or face interval;
- local maximum size only when it is a real hard requirement;
- compactness as an objective over multiple exposed metrics;
- bounded Pareto candidate retention;
- conservative allowed transforms;
- per-constraint diagnostics and snapshot reporting;
- search policy in a separate block.

The first version should not include:

- arbitrary expressions or user-defined constraint functions;
- macros, includes, or procedural loops;
- arbitrary polyhedra or rotations;
- constraints on prepared graph node indices;
- silent selector fallbacks;
- one unstructured weighted cost for all metrics;
- promises of global or local optimality from heuristic search;
- a second hand-authored serialization of every physical block and route.

## Open questions

- Should reusable implementation graphs reuse Routable RCIR module syntax or
  live in a target-library-specific format?
- Should `*.rcell` and `*.rclayout` share one container header with separate
  document kinds, or only share lexer conventions?
- Which local metrics are sufficiently stable to expose in a versioned public
  objective language?
- How should effective blockage include redstone power interference that is
  state- or direction-dependent?
- Which yaw rotations can be proven valid by transforming block states, and
  which candidates must be regenerated per orientation?
- How should a logical-origin selector expand when lowering produces several
  Routable instances with different physical roles?
- When should instance-specific intent select an existing candidate versus
  specialize and regenerate a cell definition?
- Which static constraint subsets admit useful contradiction or capacity
  proofs before search?
- How should a global routing failure request new local candidates with
  different port exposure without creating an unbounded feedback loop?
- What cache identity includes target rules, recipe text, compiler version,
  verification model, and search completeness?

These questions should be answered only after stable Routable identities and
richer candidate metadata exist. The language should follow a demonstrated
physical model rather than freeze the current placer implementation.
