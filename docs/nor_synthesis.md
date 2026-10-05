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
  `synthesize_buildable_full_adders` sweeps the step count.
- `SynthesisOutcome` tells a proven `Infeasible` from a time-out (`Unknown`).

## Full adder: the fewest torches

`synthesize_buildable_full_adders`, 120 s per step count (300 s for the
rechecks marked):

| Bound | 9-14 steps |
| --- | --- |
| live <= 4 | 9 torches at every count, each proven |
| live <= 5, live <= 6 | 9 torches at every count, each proven |
| live <= 3 | none for 9-13 steps (proven); 14 unknown at 300 s |
| live <= 4, depth <= 5 | 9 torches only at 11 steps (cout 4, s 5); none at 9-10 steps; 10 torches at 12-13 steps |
| live <= 4, depth <= 4 | none for 9-14 steps (proven) |

- **No buildable full adder has fewer than 9 torches**, within 14 steps,
  even with 6 nets alive. The e-graph's 8-torch netlist, chained, takes 16
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
harness, 5 workers, 60 s steps, 600 s of plain compaction (no step or repair
minimization):

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

- The seed moves the result more than the netlist does. `nor9` at height
  12 gave anything from 90 to 237 blocks. The 11-step netlist is the
  `or_cost` 1 one in another step order and input naming, and gave 176
  blocks against 115.
- So two or three runs of 600 s cannot rank netlists by blocks. The
  repository's smallest full adder (2x8x8, 60 blocks,
  `exact_local_placer.md`) came from the full pipeline with step and repair
  minimization.
- What synthesis settles is the logic: 9 torches is the floor for a
  buildable full adder, and 4/5 is the best depth at that count.

## Next

- Carry tiles: a ripple-carry adder is as fast as its `cin` to `cout` path.
  Bound that path in synthesis, and keep the carry monotone
  (`carry_tiles.md`).
- Compare netlists under the full pipeline (step and repair minimization)
  over several seeds before ranking them by blocks.
- Synthesize the circuits whose netlists are hand-written or decomposed
  mechanically (the 2-bit adder, the 4:1 multiplexer).

## Running

```sh
cargo test --release --lib synthesized_full_adders_stay_buildable
SYNTH_STEPS=9..=14 SYNTH_LIVE=4 SYNTH_DEPTH=5 SYNTH_SECONDS=120 \
  cargo test --release --lib synthesize_buildable_full_adders -- --ignored --nocapture
CIRCUIT=netlist CIRCUIT_ORDER=index CIRCUIT_NETLIST='g3=NOR(cin); ...' \
  cargo test --release --lib diagnose_construct_circuit -- --ignored --nocapture
```

`CIRCUIT_NETLIST` takes `NorNetlist::to_text`'s form. A name read before it
is defined is an input, and a net nothing reads is an output.
