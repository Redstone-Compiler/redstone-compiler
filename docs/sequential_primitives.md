# Sequential Primitive Placement

## Direction

Sequential circuits are represented as top-level `GraphNodeKind::Sequential` nodes. Each sequential node is opaque to ordinary combinational passes, but it can keep a gate-level `inner_graph` for inspection, documentation, and future pattern validation.

For an RS latch, the inner graph currently records the cyclic equations:

```text
q  = ~(r | nq)
nq = ~(s | q)
```

This keeps feedback out of the top-level placement graph while preserving the gate-level meaning of the primitive.

## Feedback Core Recognition

Sequential cells should stay decomposable instead of becoming one large atomic primitive per cell type. The current first step is `src/sequential/core.rs`, which analyzes a sequential `inner_graph` and recognizes the minimal cyclic feedback core.

For RS latch, the recognizer checks:

- a cyclic SCC containing the two OR/NOT feedback branches,
- `q` and `nq` outputs driven by NOT nodes,
- cross-coupled NOR structure,
- external `s` and `r` input nodes.

This lets larger cells remain compositions:

```text
D latch  = input gating + RS latch core
DFF      = two D latches + clock inversion
TFF/JKFF = DFF/latch core + feedback combinational logic
register = FF macros composed hierarchically
counter  = FF macros + carry/toggle logic
```

The physical placer can therefore special-case only the minimal feedback core while leaving surrounding combinational logic to normal placement/routing.

## Placement Model

The local placer now uses `PlacementState` instead of a plain `GraphNodeId -> Position` map. Existing combinational nodes still use node positions. Sequential macros can additionally register named physical ports such as:

```text
Port(rs_latch_node, "q")
Port(rs_latch_node, "nq")
```

The first implementation supports only one exposed sequential output in the top-level graph. This is intentional because graph edges do not yet encode the source output port. The RS latch macro exposes both `q` and `nq` internally, but only q-only top-level use is accepted by `LocalPlacer`.

## Current RS Latch Macro

`src/sequential/layout.rs` defines the initial `SequentialMacro` interface and one simulator-validated RS latch macro candidate. RS latch macro candidates are now gated by successful recognition of the RS latch feedback core in the primitive `inner_graph`. Tests currently validate:

- expected input and output port names,
- port positions are unique,
- port positions are in bounds and non-air,
- RS latch feedback SCC/core recognition,
- q-only sequential nodes are accepted by `LocalPlacer`,
- multi-output sequential nodes are rejected until graph edges become port-aware,
- unsupported sequential primitives such as `DLatch` are rejected,
- macro output ports are registered in placement state,
- existing source positions can be routed into macro input ports,
- the RS latch macro satisfies reset, hold, set, and reset sequences under `Simulator`.

## Remaining Work

- Make graph edges port-aware so `q` and `nq` can drive different consumers.
- Replace the first RS latch hardcoded macro with generated candidates from a constrained feedback-core synthesizer.
- Add more RS latch macro candidates with different port orientations and costs.
- Extend macro placement to orientation and mirroring.

Run placement-related tests with `cargo test --release`.
