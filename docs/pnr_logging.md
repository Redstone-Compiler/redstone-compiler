# PnR logging

Production placement, routing, and simulation code uses `tracing`. It must not
write progress directly with `println!` or `eprintln!`; callers that do not
install a tracing subscriber remain silent.

## Levels

- `info`: phase transitions, phase timing summaries, selected solution cost,
  long-running routing heartbeats, and the final result. Routing heartbeats are
  emitted at completed-attempt boundaries after roughly ten seconds of work.
- `debug`: layout combinations, routing attempts, cache reuse, and individual
  failure reasons.
- `trace`: local placer steps, candidate counts within a step, individual nets
  and sinks, and simulator events.
- `warn`: exhausted search budgets, fallback behavior, and aggregated terminal
  search failure.
- `error`: unrecoverable failures at an application boundary.

`GlobalPnrConfig::show_progress` controls whether global PnR tracing events are
emitted. It does not print to stderr directly. Its default remains `true`.

## Running the counter smoke test

The ignored sequential smoke tests install an `info` subscriber by default:

```powershell
cargo test --release counter_module_generates_world_from_child_layout_candidates -- --ignored --nocapture
```

Set `RUST_LOG` when more or less detail is needed:

```powershell
$env:RUST_LOG = "debug"
cargo test --release counter_module_generates_world_from_child_layout_candidates -- --ignored --nocapture

$env:RUST_LOG = "trace"
cargo test --release counter_module_generates_world_from_child_layout_candidates -- --ignored --nocapture
```

Library consumers are responsible for installing and configuring their own
subscriber. The compiler CLI installs a formatted subscriber whose default
level is `info`.
