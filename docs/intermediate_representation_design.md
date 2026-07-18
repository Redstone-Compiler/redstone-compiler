# RCIR language and lowering design

## Status

This document is the language contract for the compiler's coordinate-free
intermediate representations. The first `rcir 2` vertical slice is implemented:
Verilog lowers to typed Logical IR, Logical IR lowers directly to Routable IR
without reconstructing a Verilog AST, both stages have deterministic text
round-trips, and compilation snapshots contain both files.

The implemented Logical subset covers the counter/register path, scalar
combinational primitives, and one-level structural hierarchy. References with
slices, the complete arithmetic/mux schemas, deeper or mixed hierarchy, stable
provenance sidecars, and the separate `*.rclayout` physical-intent input remain
future work. The experimental implementation that preceded this document is
not the specification.

The new typed grammar uses `rcir 2`. An earlier experimental `rcir 1` prototype
used a different, string-oriented syntax and must not be confused with this
contract. A compatibility reader may translate v1 into v2, but new writers emit
only v2. Readers must reject unknown major versions.

## Goals

RCIR provides two explicit restart points between Verilog and physical PnR:

```text
Verilog source
  -> elaborate and infer processes
  -> Logical IR                      ir/logical.rcir
  -> direct target/macro lowering
  -> Routable IR                     ir/routable.rcir
  -> local candidate preparation
  -> global placement and routing
  -> NBT and snapshot artifacts
```

The boundaries have different purposes:

- Logical IR is target-independent, bus-aware, and preserves intent such as
  incrementers, muxes, registers, clock edges, enables, and reset behavior.
- Routable IR is target-selected and structural. Every value is scalar and
  every cell is supported by the selected target library or macro mapper.
- Physical coordinates, bounding boxes, route paths, and Minecraft block
  states belong to snapshot/physical data, not either circuit IR.

Both stages use one small textual container grammar. They resolve into separate
typed Rust models and have separate operation schemas and validators. Sharing a
parser must not turn the in-memory IR into a collection of unvalidated strings.

## Non-negotiable lowering rule

Logical IR is the canonical input to logical-to-routable lowering:

```text
Verilog AST -> LogicalDesign -> LogicalToRoutable -> RoutableDesign
```

The lowering must not reconstruct a Verilog AST, print Verilog, invoke the
Verilog parser again, or depend on Verilog procedural semantics. Direct
`logical.rcir` input and equivalent Verilog input must enter the same
`LogicalToRoutable` implementation.

Serialization is optional during an in-memory compile:

```text
                         +-> logical.rcir snapshot
                         |
Verilog -> LogicalDesign +-> logical optimization/validation
                         |
                         +-> LogicalToRoutable -> RoutableDesign
```

If this direct lowering does not exist, Logical IR is only a debug dump and
does not justify being a compileable public stage.

## Common file grammar

The grammar below is descriptive EBNF. Exact diagnostics and escape rules are
an implementation detail, but accepted programs and stage restrictions are
part of the contract.

```text
file              := version stage target? top module+
version           := "rcir" unsigned_integer ";"
stage             := "stage" ("logical" | "routable") ";"
target            := "target" name ";"
top               := "top" name ";"

module            := "module" name "{" declaration* "}"
declaration       := port | net | cell | instance

port              := "port" direction name ":" type net_class? ";"
direction         := "input" | "output"
net               := "net" name ":" type net_class? ";"
net_class         := "class" name

cell              := "cell" name ":" operation type_args? "{"
                       cell_item*
                     "}"
cell_item         := input_binding | output_binding | attribute
input_binding     := "in" name "=" value ";"
output_binding    := "out" name "=" reference ";"
attribute         := name "=" literal ";"

instance          := "instance" name ":" name "{" binding* "}"
binding           := "bind" name "=" value ";"

type              := "bit" | "bits" "[" positive_integer "]"
type_args         := "<" literal ("," literal)* ">"
operation         := name "." name
value             := reference | constant
reference         := name
                   | name "[" unsigned_integer "]"
                   | name "[" unsigned_integer ":" unsigned_integer "]"
constant          := "const" "<" positive_integer ">" "(" unsigned_integer ")"
literal           := name | unsigned_integer | string | constant
```

Lexical rules:

- Source is UTF-8.
- `#` starts a line comment outside a string.
- A bare name starts with a letter or underscore and continues with letters,
  digits, `_`, or `-`.
- Quoted names use JSON-compatible string escaping and allow generated or
  otherwise unusual names.
- Operation names are qualified, for example `logical.inc` or `std.xor`.
- Integers are unsigned decimal in v2. Constants always carry an explicit bit
  width.
- Statements use explicit semicolons; v2 has no includes, macros, implicit
  nets, or expression-precedence grammar.

Header order is canonical. `target` is forbidden for `stage logical` and
required for `stage routable`.

## Signal and connection model

A port declaration also declares its same-named signal. Only internal signals
need a `net` declaration. Cell output bindings are drivers; cell input bindings
and output ports are consumers. Input ports are drivers.

For example:

```text
port input  a : bit;
port output y : bit;
net inverted : bit;

cell invert : logical.not<1> {
  in  value  = a;
  out result = inverted;
}

cell copy : logical.buffer<1> {
  in  value  = inverted;
  out result = y;
}
```

Every signal bit has at most one driver and any number of consumers. Fanout is
one net with multiple consumers, not duplicated point-to-point nets. Feedback
is legal only when every cycle crosses a state element.

`bits[N]` uses indices `N - 1` through `0`. A slice `value[msb:lsb]` is
inclusive and requires `msb >= lsb`. Parsing resolves names and slices into
stable typed IDs before semantic validation.

## Logical IR

Logical IR contains a normalized circuit graph, not Verilog statements.

It may contain:

- `bit` and `bits[N]` signals, references, and slices;
- explicit-width constants;
- module hierarchy and named instances;
- bus-aware combinational operations;
- state operations with explicit clock, edge, enable, and reset semantics.

It must not contain:

- `always`, blocking/nonblocking assignment, statement ordering, or implicit
  Verilog widths;
- Redstone-specific macro choices or latch decomposition;
- placement candidates, coordinates, boxes, routes, or block states.

### Initial logical operation schemas

Operation schemas define pin names, widths, and required attributes. They are
typed definitions, not arbitrary `kind` strings.

In the schema notation below, `bits[N]` means `bit` when `N = 1`; the text
writer always uses the scalar spelling `bit` for one-bit declarations.

```text
logical.buffer<N>   in value: bits[N]                 out result: bits[N]
logical.not<N>      in value: bits[N]                 out result: bits[N]
logical.and<N>      in lhs, rhs: bits[N]              out result: bits[N]
logical.or<N>       in lhs, rhs: bits[N]              out result: bits[N]
logical.xor<N>      in lhs, rhs: bits[N]              out result: bits[N]
logical.add<N>      in lhs, rhs: bits[N]              out result: bits[N]
logical.inc<N>      in value: bits[N]                 out result: bits[N]
logical.mux<N>      in select: bit, when_false,
                       when_true: bits[N]              out result: bits[N]
logical.register<N> in d: bits[N], clock: bit         out q: bits[N]
logical.d_latch<N>  in d: bits[N], enable: bit        out q: bits[N]
```

`logical.register` requires `edge = posedge` or `edge = negedge`. Optional
enable/reset pins and reset attributes will be added as one coherent schema;
they must not be encoded as ad-hoc names.

### Counter example

Input Verilog:

```verilog
module counter(clk, q);
  input clk;
  output reg [1:0] q;

  always @(posedge clk) begin
    q <= q + 1;
  end
endmodule
```

Canonical Logical RCIR:

```text
rcir 2;
stage logical;
top counter;

module counter {
  port input  clk : bit;
  port output q   : bits[2];

  net q_next : bits[2];

  cell next : logical.inc<2> {
    in  value  = q;
    out result = q_next;
  }

  cell state : logical.register<2> {
    in  d     = q_next;
    in  clock = clk;
    out q     = q;
    edge = posedge;
  }
}
```

This preserves the useful fact that the design is one two-bit incrementer and
one two-bit positive-edge register. It does not preserve the source spelling
of the `always` block.

## Routable IR

Routable IR is a scalar structural netlist accepted by a declared target.

Additional rules:

- Every port and net has type `bit`.
- Generic arithmetic and bus operations are forbidden.
- Every cell operation must appear in the selected target's capability set.
- Net classes such as `data`, `clock`, `reset`, and `external` may guide later
  physical passes but do not contain coordinates.
- Module definitions and named instances remain distinct so one verified macro
  candidate pool can be reused by many instances.

An initial `redstone-v1` target may accept operations such as `std.not`,
`std.and`, `std.or`, `std.xor`, `std.buffer`, and `std.d_latch`. A macro such as
`redstone.dff` may remain as one Routable cell only if the target declares a
local-placement implementation for it. Otherwise Logical lowering expands it
into supported latch cells.

The local preparation pass may decompose `std.xor`, insert buffers, or choose a
different implementation. Those derived nodes are not canonical Routable RCIR.

Conceptual counter lowering:

```text
logical.inc<2>
  -> q_next_0 = std.not(q_0)
  -> q_next_1 = std.xor(q_1, q_0)

logical.register<2>
  -> two target-supported state cells or state macros
```

One logical operation may therefore map to several Routable cells and nets.

## Hierarchy

Both stages share the same explicit instance syntax:

```text
instance low : counter_bit {
  bind clk = clk;
  bind d   = input_0;
  bind q   = output_0;
}
```

Bindings are checked against the referenced module's typed port list. Recursive
module-instantiation cycles are forbidden in v2. A compiler may flatten
hierarchy during lowering, but the lowering map must retain the origin of every
generated object.

## Provenance

Human-readable RCIR does not carry verbose `origin` strings on every
declaration. The compiler maintains stable IDs in memory and emits large source
and cross-stage maps separately:

```text
Verilog source span
  -> Logical cell/net ID
  -> Routable cell/net IDs
  -> physical instance/route IDs
  -> NBT block positions
```

Snapshots store this as `ir/provenance.json`. Symbolic names in RCIR are labels;
renaming a label must not change structural identity inside a running compile.

## Validation

Common validation checks:

- supported version and exactly one existing top module;
- unique declarations in each scope;
- all references, operations, modules, ports, and pins resolve;
- instance bindings exactly match the child interface;
- each signal bit has at most one driver;
- no unsupported recursive hierarchy.

Logical validation additionally checks:

- operation pin schemas, widths, slices, and explicit constant widths;
- valid state clock/edge/enable/reset combinations;
- absence of combinational cycles;
- absence of target-specific or physical declarations.

Routable validation additionally checks:

- scalar-only ports and nets;
- target capability support for every operation;
- required cell pins connected exactly once;
- driver/consumer direction and net-class consistency;
- absence of arithmetic, buses, and physical fields.

Validation must complete before expensive local candidate generation or global
PnR begins.

## Direct lowering API

The intended library boundary is:

```rust
pub fn parse_verilog(source: &str, options: FrontendOptions)
    -> Result<LogicalDesign>;

pub fn parse_rcir(source: &str)
    -> Result<CircuitIr>;

pub fn lower_to_routable(
    logical: &LogicalDesign,
    target: &TargetSpec,
    policy: &MappingPolicy,
) -> Result<(RoutableDesign, LoweringMap)>;

pub fn compile_routable(
    routable: &RoutableDesign,
    config: &GlobalPnrConfig,
) -> Result<GlobalPnrResult>;
```

`LogicalToRoutable` performs these explicit phases:

1. resolve and validate types, widths, hierarchy, and operation schemas;
2. split bus signals into deterministically named scalar signals;
3. lower arithmetic and mux operations into target-independent scalar logic;
4. select target state cells/macros according to `MappingPolicy`;
5. construct Routable modules, cells, nets, and capability references directly;
6. record a many-to-many `LoweringMap`;
7. validate the completed Routable design.

Mapping policy is separate from IR semantics. Choosing ripple carry versus
another adder, a DFF macro versus master/slave latches, or one XOR
implementation versus another must not change what `logical.add`,
`logical.register`, or `std.xor` mean.

## Canonical writer

The writer is deterministic:

1. headers use the fixed order `rcir`, `stage`, optional `target`, then `top`;
2. module and declaration order is stable;
3. generated names use deterministic counters scoped by their logical origin;
4. the writer uses bare names when legal and quoted names otherwise;
5. provenance comments do not affect semantics and are omitted canonically;
6. `parse(write(ir))` is structurally equal to `ir`;
7. repeated writes of the same IR are byte-for-byte identical.

JSON may be emitted for the Viewer, but it is a versioned DTO rather than a
direct serialization of internal Rust structs. RCIR remains the primary
human-readable, compileable format.

## Minimum acceptance tests

The first implementation is not complete until all of the following hold:

1. Logical and Routable files round-trip through the common parser/writer.
2. Invalid widths, slices, drivers, pins, operations, and hierarchy fail before
   PnR with object-specific diagnostics.
3. The counter Logical golden contains exactly one `logical.inc<2>` and one
   `logical.register<2>`.
4. Logical lowering creates scalar Routable next-state logic and supported
   state cells without constructing a Verilog AST.
5. Verilog input and its emitted `logical.rcir` produce equivalent Routable IR.
6. Emitted `routable.rcir` compiles without the original Verilog source.
7. A fixed configuration passes the existing counter behavioral verifier from
   Verilog, Logical RCIR, and Routable RCIR inputs.
8. Snapshot provenance can navigate from the logical increment/register to all
   generated Routable cells and physical objects.

## Implementation status

The first replay boundary now exists alongside the IR pipeline. A
`PreparedPnrDesign` owns the resolved typed topology and structurally
deduplicated local candidate sets. Snapshot archives persist both, so a later
run can change global placement, routing, and search settings without invoking
the local placer again.

Global routing connectivity now originates from the resolved topology. A small
`GraphModule` adapter is materialized from typed nets only because the current
physical router implementation still consumes that shape internally. Every
physical route branch carries its `NetId` and typed source/sink endpoints, so
snapshots and diagnostics no longer need to recover identity from labels.

Placement candidate ordering, Free3D attraction, layered assignment, and
wire/congestion cost now also receive connectivity reconstructed from the
resolved topology rather than from the original legacy module. Routing retry
feedback groups branches by `NetId`; labels remain presentation metadata and a
compatibility fallback only.

The physical placer/router implementations still consume a temporary
`GraphModule`-shaped adapter internally. Replacing that adapter with native
typed placement and routing plan structs is the final IR-boundary migration.

Prepared/replayed global execution no longer reads the retained legacy module
for connectivity or interface discovery. Top-level input switches and output
observation points are collected from typed top-port endpoints. The retained
module is now limited to preparation compatibility/parity metadata and the
physical adapter implementation.
