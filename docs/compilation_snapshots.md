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
|   |-- routable.json
|   `-- source-map.json
|-- candidates/
|   |-- index.json
|   `-- set-<index>/
|       |-- candidate-<index>.nbt
|       `-- candidate-<index>.json
|-- instances/<index>-<name>/
|   |-- circuit.nbt
|   `-- instance.json
|-- routes/
|   |-- routes.nbt
|   `-- routes.json
|-- intent/
|   |-- <source>.rclayout
|   |-- resolved.json
|   `-- report.json
`-- pnr/
    |-- config.json
    |-- preparation.json
    `-- topology.json
```

`ir/logical.rcir` preserves bus-level operations and state intent such as
`inc` and `register`. It is emitted for Verilog and logical-IR inputs.
`ir/routable.rcir` is the self-contained scalar, target-mapped input accepted
by global PnR. It includes the circuit plus the effective source-level PnR
policy, input pin-search constraints, and physical intent. It can therefore be
compiled without the original Verilog, `.rclayout`, or ad-hoc Rust config.
The matching JSON files contain the same experimental data for tools that
prefer a structured format. Each IR artifact is emitted immediately after its
stage completes, so it remains available when later lowering, placement, or
routing fails.

`ir/source-map.json` links rendered line ranges across the original Verilog,
Logical RCIR, and Routable RCIR through source, derived, and fused locations.
Its `entities` table gives stable typed names, kinds, locations, and enclosing
scopes to those objects, while `relations` records non-provenance edges such as
an instance referencing its module definition. Keeping scope and reference
edges separate prevents navigation from treating every use of a shared
definition as the same lowering result.
It is debug-only metadata and does not affect RCIR semantics or cache keys. The
Viewer uses it for Godbolt-style hover highlighting and click-to-pin navigation
between the three panes.

`pnr/config.json` is the generated, typed expansion of the effective PnR policy
plus runtime-only metadata. It is diagnostic output; the source of truth for a
recompile is `ir/routable.rcir`.

`candidates/` contains every verified local candidate retained by the prepared
design, deduplicated by structural candidate set rather than copied once per
instance. `pnr/topology.json` assigns typed definition, instance, port, and net
IDs. `pnr/preparation.json` records candidate bindings and a migration parity
signature. Together these files make the snapshot a replayable boundary
between local preparation and global PnR.

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

A snapshot directory or its portable `.rsnap` archive can be used as input.
This reloads the verified candidate library and reruns only global placement
and routing; the local placer is not invoked:

```text
redstone-compiler build/design.rsnap build/design-rerun.snapshot
```

Candidate-affecting settings must match the preparation fingerprint. Placement,
routing, and global search settings may change without regenerating candidates.
When a snapshot contains resolved physical intent, replay reuses it unless a
new `--intent` file is supplied explicitly.

For separate fresh compilations, a persistent local-candidate cache can avoid
rerunning the local placer for an identical normalized child graph, port
contract, and candidate configuration:

```text
redstone-compiler counter.v build/counter.snapshot \
  --candidate-cache target/redstone-candidate-cache
```

The cache is opt-in. Entries are content-hash checked while loading; a corrupt
or stale entry is reported as a progress detail and regenerated instead of
failing the compilation. Global placement, routing, physical intent, and search
knobs are deliberately excluded from the key because they consume candidates
rather than change how candidates are generated.
