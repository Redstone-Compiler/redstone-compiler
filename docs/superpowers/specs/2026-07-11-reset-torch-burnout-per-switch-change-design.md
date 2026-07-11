# Reset Torch Burnout Per Switch Change

## Problem

The simulator currently retains `burned_out_torches` for its entire lifetime. A
counter clock transition can trigger enough internal feedback events to mark a
torch as burned out. Later clock transitions then cannot turn that torch back
on, so otherwise valid sequential circuits stop responding until the NBT is
reloaded.

## Design

Treat torch burnout as protection for one settle operation, not persistent
circuit state. Before applying a new externally requested switch-state change,
clear the torch toggle history and burned-out set. During the ensuing settle,
the existing toggle window and limit continue to stop oscillating feedback.

Initialization behavior remains unchanged. No public API or placement logic
changes are required.

## Verification

- Add a regression test using `counter-global-smoke.nbt` that drives the clock
  high, low, then high and verifies the affected wall torch changes
  `on -> off -> on`.
- Preserve a unit test proving a torch marked burned out remains off during the
  current settle operation.
- Run the targeted release tests, then `cargo test --release` as required for
  placer and place-and-route coverage.
- Rebuild the viewer WASM and reproduce the same switch sequence in the viewer.

## Scope

This change does not implement Minecraft tick-accurate burnout cooldown and
does not remove oscillation protection. It only changes the lifetime of the
existing stabilization state.
