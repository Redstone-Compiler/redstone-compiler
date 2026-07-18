# Compilation snapshots

Compilation output is a directory ending in `.snapshot`. The final NBT is kept
at the root so it is easy to find; the manifest is the machine-readable index.

```text
counter.snapshot/
|-- counter.nbt
|-- counter.v
|-- manifest.json
|-- summary.json
|-- interface.json
|-- placement-bboxes.nbt
|-- ir/
|   |-- logical.rcir
|   |-- logical.json
|   |-- routable.rcir
|   `-- routable.json
|-- instances/<index>-<name>/
|   |-- circuit.nbt
|   `-- instance.json
|-- routes/
|   |-- routes.nbt
|   `-- routes.json
`-- pnr/config.json
```

`ir/logical.rcir` preserves bus-level operations and state intent such as
`inc` and `register`. It is emitted for Verilog and logical-IR inputs.
`ir/routable.rcir` is the scalar, target-mapped IR accepted by global PnR.
The matching JSON files contain the same experimental data for tools that
prefer a structured format. Each IR artifact is emitted immediately after its
stage completes, so it remains available when later lowering, placement, or
routing fails.

`summary.json` records status, total elapsed time, selected placement and route
metrics, and typed compilation events. Failed compilation scopes still write a
summary and manifest with `status: "failed"`.

## API

Use `compile_with_snapshot` at the compilation boundary:

```rust
let result = compile_with_snapshot(
    SnapshotOptions::new("build/counter.snapshot", "counter")
        .with_source("counter.v"),
    || compile_counter(),
)?;
```

Code inside the closure can record typed data without carrying a report object:

```rust
snapshot::record(SnapshotEvent::Stage {
    module: "counter".to_owned(),
    step: 1,
    total: 4,
    name: "generate candidates".to_owned(),
});
```

An ambient run ID routes events to the correct session through a process-wide
hub. A background writer thread writes artifacts, and finalization waits for a
flush before returning. Calls outside a snapshot scope are no-ops.

The ambient run ID is thread-local. Summary events currently originate at the
sequential phase boundaries after Rayon work joins. Worker threads can inherit
the session only at the parallel boundary with `snapshot::capture()` followed
by `token.in_scope(...)`; ordinary compiler functions still carry no reporter.

## CLI compatibility

The existing positional output remains accepted. An output such as
`build/adder.nbt` now creates `build/adder.snapshot/adder.nbt`; passing a path
that already ends in `.snapshot` uses that directory directly.

Logical and Routable IR files can both be compiled through the same
snapshot-producing path. The `stage` header selects the path:

```text
redstone-compiler design.rcir build/design.snapshot
```
