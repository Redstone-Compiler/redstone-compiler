# Verilog RTL interface design

## Purpose

The Verilog frontend preserves hardware intent long enough for target mapping
and physical design to make deliberate choices. It does not construct the
physical hierarchy used by PnR directly.

The supported pipeline is:

```text
Verilog source
  -> lexer / parser AST
  -> RTL process model
  -> synthesis cells
  -> LogicalDesign
  -> direct target mapping
  -> RoutableDesign
  -> PreparedPnrDesign
  -> global placement and routing
  -> NBT + compilation snapshot
```

`LogicGraph` remains available for small combinational expressions and as the
node-level input to local placement. It is not a circuit-wide hierarchy or net
identity model.

## Stage responsibilities

### Verilog AST and RTL

The parser owns source syntax. RTL lowering owns procedural semantics such as
clock edges, nonblocking assignments, latch inference, and next-state
expressions. Neither stage knows about Minecraft placement or routing.

### Logical IR

`LogicalDesign` is target-independent and bus-aware. It preserves:

- module definitions and typed instances;
- named nets and vector widths;
- combinational operations;
- registers, DFFs, latches, clock edges, and next-state intent;
- source/provenance locations.

Verilog input and textual Logical RCIR input converge at this boundary.

### Routable IR

Logical lowering constructs `RoutableDesign` directly. Target mapping may
bit-blast buses and expand a register into inverter, next-state, master-latch,
and slave-latch leaves. The result owns explicit definitions, instances, ports,
nets, net classes, and scalar leaf nodes.

The old intermediate graph hierarchy has been removed. In particular, the
compiler must not reconstruct another hierarchy and then convert it back into
Routable IR. This keeps instance identity, fanout, net names, and provenance in
one typed model.

### Local and global PnR

Local placement receives one Routable leaf at a time. Only its scalar node body
is adapted to `Graph`; its typed ports stay authoritative for candidate I/O.

Global PnR receives a validated `RoutableDesign`, resolves it to
`ResolvedPnrTopology`, prepares reusable candidate sets, and then performs
placement and routing. Routing branches carry typed net and endpoint IDs rather
than recovering connectivity from display names.

## Extension rules

When adding Verilog support:

1. Parse syntax without assigning physical meaning in the parser.
2. Represent procedural behavior in RTL/synthesis types.
3. Preserve buses and state intent in Logical IR.
4. Add an explicit target-mapping rule from Logical to Routable IR.
5. Validate both IR stages independently.
6. Add source-map coverage and deterministic text round-trip tests.
7. Exercise local placement only after the Routable leaf schema is stable.

New state elements or arithmetic operations should become Logical cell kinds
before they become redstone macros. New Minecraft implementations should be
selectable target mappings or candidate profiles, not new Verilog syntax.

## Current supported vertical slice

The implemented path covers small combinational modules, structural hierarchy,
D latches, positive-edge DFF expansion, and incrementing registers used by the
two-bit counter smoke test. Unsupported constructs must fail during validation
or target mapping with a stage-specific diagnostic.

See also:

- `docs/intermediate_representation_design.md`
- `docs/physical_design_intent.md`
- `docs/compilation_snapshots.md`
