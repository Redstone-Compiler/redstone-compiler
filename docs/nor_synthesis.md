# Synthesizing NOR netlists construction can build

The e-graph (`egraph_netlists.md`) finds full adders with fewer gates or
less depth than `nor9`, but windowed construction cannot place them. Their
steps time out when too many nets are alive between two steps, or when a
gate's inputs branch on to later gates ("Why the steps fail"). Those two
counts separated the netlists that built from those that did not.

Exact synthesis (`exact/synthesis.rs`, model `exact/nor_synthesis.rsdsl`)
works the other way round. It builds a netlist and its construction order
together, step by step, with those counts bounded. Every netlist it returns
can be placed.

## Model

- Nets are the inputs, then the steps in construction order. Construction
  takes them in that order (`CIRCUIT_ORDER=index`).
- A step is a torch (NOR) or an OR stage, and reads one or two earlier
  nets. With at most two inputs, at most two of them can continue past the
  gate.
- An OR stage is part of a wide NOR on its support block
  (`NorNetlist::chain_wide_gates`, `construct.rs` `chained_ors`). It joins
  two signals, and only the next step reads it.
- No step is wasted:
  - each of a step's two inputs is on in some case the other is off in;
  - a torch does not read a lone OR stage (that is the NOR of the OR's two);
  - every step is read later or is an output.
- Outputs are torches.
- Between two steps at most `max_live` nets are alive. A net is alive once
  it is placed (an input once a step reads it) until its last reader.
- Outputs are at most `max_depth` torches deep, if given.
- The solve minimizes torches for a fixed number of steps;
  `synthesize_buildable_circuits` sweeps the step count.
- `SynthesisOutcome` tells a proven `Infeasible` from a time-out (`Unknown`).

## Full adder: the fewest torches

`synthesize_buildable_circuits`, 120 s per step count (300 s for the
rechecks marked):

| Bound | 9-14 steps |
| --- | --- |
| live <= 4 | 9 torches at every count, each proven |
| live <= 5, live <= 6 | 9 torches at every count, each proven |
| live <= 3 | none for 9-13 steps (proven); 14 unknown at 300 s |
| live <= 4, depth <= 5 | 9 torches only at 11 steps (cout 4, s 5); none at 9-10 steps; 10 torches at 12-13 steps |
| live <= 4, depth <= 4 | none for 9-14 steps (proven) |

- **No buildable full adder has fewer than 9 torches.** With at most 4 live
  nets this holds up to 16 steps (15 steps: 9 torches, 16 steps: 10, both
  proven), and with 5 or 6 live nets up to 14 steps.
- The bound is what excludes 8 torches, not the model. With 7 live nets and
  16 steps, synthesis finds an 8-torch netlist in 11 s (cout 3, s 5), as
  the e-graph did. The e-graph's own 8-torch netlist, chained, also takes 16
  steps and keeps 7 nets alive.
- **Depth 4/5 (cout/s) is the best possible at 9 torches**, one torch
  below `nor9`'s 5/6. It needs 11 steps, two of them OR stages. That
  netlist is the e-graph's `or_cost` 1 netlist chained, with the inputs
  renamed, found again from scratch:
  `g3=NOR(cin); g4=NOR(a); g5=NOR(a,cin); g6=NOR(g3,g4); o7=OR(g5,g6);
  g8=NOR(b,o7); g9=NOR(b,g8); cout=NOR(g5,g8); o11=OR(g6,g8);
  g12=NOR(g5,o11); s=NOR(g9,g12)`.
- Three live nets are not enough within 13 steps, and an `s` of depth 4 is
  not possible within 14.

## Placement

Every synthesized netlist built, with every step in window 2. The circuit
harness ran with 5 workers and 60 s steps, its defaults otherwise (step and
repair minimization on). First with 600 s of compaction:

| Netlist | Height | Seed | Construction | Compacted |
| --- | --- | --- | --- | --- |
| `nor9` | 10 | 1 | 64 s | 2x12x9, 146 blocks |
| `nor9` | 10 | 2 | 96 s | 2x11x10, 148 blocks |
| `nor9` | 12 | 1 | 97 s | 2x9x9, 94 blocks |
| `nor9` | 12 | 2 | 415 s, 1 restart | 2x10x10, 90 blocks |
| `nor9` | 12 | 3 | 113 s | 2x13x12, 237 blocks |
| `or_cost` 1, chained | 10 | 1 | 119 s | 2x11x9, 115 blocks |
| `or_cost` 1, chained | 10 | 2 | 110 s | 2x13x10, 116 blocks |
| `or_cost` 1, chained | 12 | 1 | 195 s | 2x11x9, 145 blocks |
| `or_cost` 1, chained | 12 | 2 | 593 s, 2 restarts | 2x10x11, 151 blocks |
| synthesized, 11 steps | 10 | 1 | 69 s | 2x12x9, 176 blocks |
| synthesized, 11 steps | 12 | 1 | 221 s | 2x13x12, 193 blocks |

At 600 s the seed moves the result more than the netlist does: `nor9` at
height 12 gave anything from 90 to 237 blocks. With 1800 s of compaction
(height 10, three seeds each), the two netlists end up close:

| Netlist | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | --- | --- | --- | --- |
| `nor9` | 2x10x8, 85 blocks, 15/13 ticks | 2x13x5, 90, 11/12 (converged) | 2x11x8, 76, 10/12 | 84 |
| `or_cost` 1, chained | 2x9x8, 87, 13/12 | 2x9x9, 80, 14/18 | 2x9x7, 72, 10/10 (converged) | 80 |

- The 30-block gap at 600 s was mostly compaction not having finished.
- The chained `or_cost` 1 netlist is about 4 blocks smaller on average, and
  gave the smallest of the six cells (2x9x7, 72 blocks, 10 ticks for both
  outputs). Three seeds cannot tell that from noise.
- The repository's smallest full adder (2x8x8, 60 blocks,
  `exact_local_placer.md`) came from the pipeline harness with inputs and
  outputs on fixed faces and 1800 s of compaction.
- What synthesis settles is the logic: 9 torches is the floor for a
  buildable full adder, and 4/5 is the best depth at that count.

## Other circuits

`synthesize_buildable_circuits` takes `SYNTH_CIRCUIT`
(`synthesis_functions`). At most 4 live nets:

| Circuit | From expressions | Hand-written | Synthesized |
| --- | --- | --- | --- |
| half adder | 7 gates | | 5 torches in 5 steps (none in 4, proven) |
| 2:1 mux | 7 gates | | 4 torches in 4 steps (none in 3, proven) |
| 4:1 mux | 20 gates, 6 live nets, never built | | 8 torches in 12 steps (proven for 12; none in 8-10, 10 torches in 11) |
| 2-bit adder | 26 gates, 6-8 live nets | 15 gates (`adder2-nor`) | 14 torches in 14-17 steps (not proven; none in 13) |

- **The 4:1 mux builds for the first time.** Its 20-gate netlist from
  expressions keeps 6 nets alive and ends in a four-input NOR. Every
  earlier attempt timed out near the end (`exact_local_placer.md`, "Larger
  circuits"). The synthesized netlist:
  `g6=NOR(a,s0); g7=NOR(s0); g8=NOR(b,g7); o9=OR(g6,g8); g10=NOR(s1,o9);
  o11=OR(s0,g10); g12=NOR(c,o11); g13=NOR(s1,g10); o14=OR(d,g10);
  g15=NOR(g7,o14); o16=OR(g12,g15); out=NOR(g13,o16)`.
  It built on the second attempt in all three runs (two seeds; 431-643 s,
  2x24x10 to 2x26x10). 1800 s of compaction with 7 workers gave 2x18x9
  with 168 blocks and 2x16x10 with 142 blocks, 17-18 ticks. Both cells
  pass RCELL verification.
- **The 2-bit adder** needs a torch less than the hand-written one, and is
  shallower. With 15 steps: c1 5, s0 4, s1 6, against `adder2-nor`'s
  7, 4, 8. Four inputs make the solves much slower: 13 steps took 257 s to
  prove infeasible, and 14-16 steps did not finish within 30 minutes.

## Next

- Carry tiles: a ripple-carry adder is as fast as its `cin` to `cout` path.
  Bound that path in synthesis, and keep the carry monotone
  (`carry_tiles.md`).
- Rank netlists by blocks only over several seeds with long compaction;
  600 s is too short for the full adder.
- Use synthesis for local cells in general: decomposed netlists waste
  torches (7 gates against 5 for a half adder, 20 against 8 for a 4:1 mux)
  and live nets.
- Prove the 2-bit adder's minimum. Symmetry breaking (the order of
  independent steps) should cut the 4-input solves.

## Running

```sh
cargo test --release --lib synthesized_
SYNTH_STEPS=9..=14 SYNTH_LIVE=4 SYNTH_DEPTH=5 SYNTH_SECONDS=120 \
  cargo test --release --lib synthesize_buildable_circuits -- --ignored --nocapture
CIRCUIT=netlist CIRCUIT_ORDER=index CIRCUIT_NETLIST='g3=NOR(cin); ...' \
  cargo test --release --lib diagnose_construct_circuit -- --ignored --nocapture
```

`CIRCUIT_NETLIST` takes `NorNetlist::to_text`'s form. A name read before it
is defined is an input, and a net nothing reads is an output.
