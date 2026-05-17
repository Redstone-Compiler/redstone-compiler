# Sequential Primitive Placement

## Direction

Sequential circuits are represented as top-level `GraphNodeKind::Sequential` nodes. Each sequential node is opaque to ordinary combinational passes, but it can keep a gate-level `inner_graph` for inspection, documentation, and future pattern validation.

For an RS latch, the inner graph currently records the cyclic equations:

```text
q  = ~(s | nq)
nq = ~(r | q)
```

This keeps feedback out of the top-level placement graph while preserving the gate-level meaning of the primitive.

## Placement Model

The local placer now uses `PlacementState` instead of a plain `GraphNodeId -> Position` map. Existing combinational nodes still use node positions. Sequential macros can additionally register named physical ports such as:

```text
Port(rs_latch_node, "q")
Port(rs_latch_node, "nq")
```

The first implementation supports only one exposed sequential output in the top-level graph. This is intentional because graph edges do not yet encode the source output port. The RS latch macro exposes both `q` and `nq` internally, but only q-only top-level use is accepted by `LocalPlacer`.

## Current RS Latch Macro

`src/sequential/layout.rs` defines the initial `SequentialMacro` interface and one RS latch macro candidate. Tests currently validate:

- expected input and output port names,
- port positions are unique,
- port positions are in bounds and non-air,
- q-only sequential nodes are accepted by `LocalPlacer`,
- multi-output sequential nodes are rejected until graph edges become port-aware,
- unsupported sequential primitives such as `DLatch` are rejected,
- macro output ports are registered in placement state,
- existing source positions can be routed into macro input ports.

## Remaining Work

- Make graph edges port-aware so `q` and `nq` can drive different consumers.
- Replace or strengthen the initial RS latch macro with a simulator-validated physical latch layout.
- Add behavioral simulation tests for set/reset/hold sequences.
- Add more macro candidates with different port orientations and costs.
- Extend macro placement to orientation and mirroring.

Run placement-related tests with `cargo test --release`.
