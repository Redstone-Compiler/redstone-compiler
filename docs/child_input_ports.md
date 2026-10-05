# Child cells with inputs placed as ports (2026-10-05)

A composite design's child cells are routed by global PnR, and a child's
input port is where a global route connects. The local placer used to drive
each input with a switch. `candidate_layout` then replaced the switch with a
port, and the global router added an input diode (a repeater) in front of
that port. The truth table was checked on the layout with switches, not on
the layout that was routed.

Combinational monolithic children now get their inputs placed as ports from
the start (`CandidateInputMode::PlacedPorts`, `LocalPlacer::with_port_inputs`).
The layout checked against the truth table is the one routed.

## Why switches were the wrong stand-in

A switch powers its neighbors and is powered by nothing. A port is driven by
a route, and the route's dust is powered by its neighbors.

Issue #56 is that difference:

1. The and2 cell's switch sat beside the block above a NOT torch, a block the
   torch powers strongly. A switch there is harmless, so the cell passed its
   truth table.
2. The rewrite turned the switch site into dust.
3. That block powered the dust, and the dust fed the torch's support: a
   one-torch ring through the input.

#60 re-checks the rewritten layout. That catches the mismatch, but only after
the search has produced a layout built around it.

## The port form

Each input is two cells, with supports:

- **Terminal:** dust on an input site, on a block. This is where a route
  connects (`input_port_placements` in `local_placer/routing.rs`).
- **Repeater:** one cell over, reading the terminal. The cell's logic reads
  the input from this repeater. The placement state records the repeater as
  the input node and the terminal under the port name `terminal`.

A repeater is powered only from behind, so nothing in the cell can drive the
terminal or the route behind it. That is the same isolation the router's
input diode gave. Here it is part of the layout that is searched and checked.

`candidate_layout` keeps such a terminal as a `Direct` port with that single
access point. The router routes straight to the dust and adds no diode. Top-
level input levers are placed beside the terminal.

The simulator drives a terminal directly (`drive_inputs_with_limits` on
dust), so the truth-table check needs no switch.

Placement rules:

- Input sites are the same as for switches (box boundary, or the
  constraints).
- The repeater may point along any horizontal direction that stays in the
  box.
- Both cells need a block below, so inputs are never on the floor layer.
- Nothing already placed may power the terminal or the repeater.

## Measurements

Child cells were generated from Verilog with switch inputs (rewritten) and
with port inputs (`compare_child_input_forms`). Settings: 10x10x5 box,
greedy input generation, sampling 256, up to 8 candidates, clustering off.
Block counts include the input repeaters. With switch inputs, the router
adds about 4 blocks per input later, and those are not counted here.

| Cell | Switch inputs: candidates, min blocks, min volume, time | Port inputs |
| --- | --- | --- |
| `~a` | 8, 2, 2, 4.3 s | 8, 9, 18, 2.4 s |
| `~(a\|b)` | 8, 5, 12, 4.1 s | 8, 16, 36, 6.5 s |
| `a\|b` | 8, 7, 8, 2.4 s | 8, 13, 24, 2.3 s |
| `a&b` | 8, 19, 84, 15.9 s | 8, 31, 80, 10.0 s |
| `a^b` | 8, 46, 140, 19.8 s | 8, 50, 168, 12.1 s |
| 2:1 mux | 2, 45, 300, 19.8 s | 8, 61, 320, 11.2 s |
| counter `q_0_next` | 8, 2, 2, 3.7 s | 8, 9, 18, 2.0 s |
| counter `q_1_next` | 8, 46, 140, 16.1 s | 8, 50, 168, 9.8 s |

Port inputs search faster and find at least as many verified candidates.
Cells grow by the diode the router used to add, plus some volume.

### End-to-end composites

Three small composites of combinational children were compiled end to end
with the default global PnR settings. Each final world was simulated in
every input case (`small_combinational_composites_compile_and_compute_their_functions`):

| Composite | Switch inputs | Port inputs |
| --- | --- | --- |
| inv, then nor2 | no world: top-level input lever `b` could not be placed | compiles, 4 of 4 cases right |
| three inverters in a chain | no world: route `u0.y -> u1.a` unreachable | compiles, 2 of 2 right |
| and2, then nor2 | no world: route to a top-level input unreachable | compiles, 8 of 8 right |

With switch inputs, each route had to end in an input-diode adapter: a
repeater and a driver cell beside the port. Each top-level input lever had to
find room beside that adapter. Those cells were often occupied by the child's
own blocks.

A composite with an xor2 child found no xor2 candidate with the composite's
sampling limit (32), in either form.

The 2-bit counter and the D flip-flop end-to-end tests (ignored) fail in
global routing on master, before and after this change:

- Before this change, the counter's routes failed their contracts.
- With port inputs, every counter child gets candidates. Its `q_1_next` was
  losing all of them on master until #65. Routing then finds no path for
  `q_0_slave.q -> q_0_next.q_0`.
- That port, routed alone in the same cell, is reachable from three sides.
  So the failure is in the counter's crowded placement, not in the port
  form.

## What still rewrites switches

- **Clustered cells.** Cluster composition restores switches at the old
  switch sites and adds its own input diodes. Cluster cells keep switch
  inputs (`ExternalPorts`). Clustering re-checks its cells after
  materializing their inputs, but not the composed result after its final
  rewrite.
- **Sequential children.** They are not truth-table checked, so they keep
  switch inputs and the router's input diode.
- **Top-level leaves.** They keep their switches as the design's inputs.
  Nothing is rewritten.

The switch rewrite (`expose_switchless_input_ports`) and #60's re-check stay
for the clustered and sequential paths.

## Related gaps found on the way

- **No final-world check by default.** `GlobalPnrConfig::verifier` is `None`
  outside two ignored tests. A composite's assembled world is never simulated
  against its function. The test above does that for three combinational
  designs. Sequential designs would need a reference model.
- **Lever physics.** The simulator's switch weakly powers every adjacent
  block (`TorchOn` events to all non-attached neighbors). In the game a lever
  strongly powers only the block it is attached to, plus adjacent dust,
  repeaters, comparators, and mechanisms (minecraft.wiki, Lever). The exact
  placer's model forbids layouts that depend on this; the beam-search local
  placer does not.
- **Static power checks skip blocks.** `detailed_router::target_powers_position`
  returns false for any block source. So the router's and clustering's static
  contact checks do not see a strongly powered block feeding adjacent dust,
  which was the mechanism of issue #56.
- **Exported torch states.** A composite's exported world is not settled:
  with every input off, reading it without a settle gives wrong outputs.
  The exact placer exports settled cells (`export_world`). Global PnR does
  not.
