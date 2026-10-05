# Equivalent NOR netlists from an e-graph (2026-10-05)

The exact placer builds the NOR netlist it is given. Construction places one
gate per step, and the signal vocabulary is the netlist's net functions and
their complements. A hand-written netlist such as the full adder's `nor9`
therefore fixes how many torches there are and how deep the logic runs.
`exact/egraph.rs` searches the equivalent netlists with an e-graph (the `egg`
crate) and extracts the ones worth placing.

## Saturation

The language has `~`, `|`, `&`, `^`, and inputs. Only `~` (a torch: one gate,
one redstone tick) and `|` (dust and blocks: free) are built; `&` and `^`
state functions and give the rewrites something to pass through.

The rules are:

- commutativity and associativity;
- double negation, idempotence, and absorption;
- De Morgan in both directions;
- AND over OR and OR over AND, both ways;
- XOR as a sum of products and as a product of sums;
- XNOR as XOR with one input negated.

The analysis gives every e-class its truth table (up to 6 inputs). `modify`
unions any two classes with the same table as soon as the second appears.
Two forms of one function thus always meet, whichever rules produced them,
and the classes are the functions the rules have reached.

The full adder (`s = a^b^cin`, `cout = ab|a·cin|b·cin`) reaches 230 of the
256 three-input functions: 6,187 nodes, 12 iterations, 73 ms.

## Extraction

**Greedy extraction** (`Exploration::extract`) gives each class the node
whose own and inputs' gates (shared ones counted once) and depth weigh
least, until no class changes. It decides class by class, so it misses
NORs that pay off only when several gates share them. On the full adder
every weighting gave the same 13-gate netlist, against `nor9`'s 9.

**Exact extraction** (`Exploration::extract_exact`) solves the choice with
SAT, from a second rsdsl model, `exact/egraph_extract.rsdsl`:

- the outputs are needed and are torches (an OR output would become a NOR
  and a NOT in the netlist);
- a needed class picks one node, and the classes that node reads are needed
  too;
- `Depth` adds a tick per NOT and none per OR, which also rules out loops;
  `max_depth` bounds the outputs;
- the objective is the number of picked NOTs, plus `or_cost` per picked
  OR (every input of a NOR is a signal to route to its block);
- `binary` allows only NORs of two signals.

The same CaDiCaL binding as the placer lowers the bound until the minimum
is proven. On the full adder each depth bound took 0.1-0.4 s:

| Depth bound | Fewest gates (proven) | Depths cout / s | Most nets live at once |
| --- | --- | --- | --- |
| 2 | none | | |
| 3 | 9 | 2 / 3 | 7 |
| 4 and up | 8 | 3 / 4 | 6 |
| `nor9` (by hand) | 9 | 5 / 6 | 4 |

The 8-gate full adder:

```
g3 = NOR(a, b)
g4 = NOR(a, cin, g3)
g5 = NOR(b, cin, g3)
g6 = NOR(a, g3, g4, g5)
g7 = NOR(b, g3, g4, g5)
g8 = NOR(cin, g4, g5)
s = NOR(g6, g7, g8)
cout = NOR(g3, g4, g5)
```

With `or_cost` 1 or more, the cheapest netlist has 9 gates and depths 4/5,
one tick faster than `nor9`. Its widest NOR reads 3 signals, and as with
`nor9` at most 4 nets are alive at once:

```
g3 = NOR(a, b)
g4 = NOR(a)
g5 = NOR(b)
g6 = NOR(g4, g5)
g7 = NOR(cin, g3, g6)
g8 = NOR(g3, g6, g7)
g9 = NOR(cin, g7)
s = NOR(g8, g9)
cout = NOR(g3, g7)
```

With `binary` (two-input NORs only), the minimum is 9 gates at depth 5/6:
`nor9` itself, proven optimal for this e-graph. Depth 5/5 needs 12 gates,
and nothing reaches depth 4. Every gain the e-graph finds over `nor9` uses
wider NORs.

The minimum holds within this e-graph. A netlist the rules never reach can
still be smaller.

## Placement

The circuit harness places the extracted netlists:

- `CIRCUIT=egraph-full-adder` uses the fewest gates overall;
- `EGRAPH_DEPTH=<n>` uses the fewest gates within depth `n`;
- `EGRAPH_OR_COST=<w>` charges each OR node `w`;
- `EGRAPH_BINARY=1` allows only two-input NORs.

Full adder, 5 workers, 600 s of compaction, no timing:

| Netlist | Seed | Construction | Compacted |
| --- | --- | --- | --- |
| `nor9`, fan-in 2 | 1 | 64 s | 2x12x9, 146 blocks, cout 13, s 18 |
| `nor9` | 2 | 96 s | 2x11x10, 148 blocks, cout 11, s 17 |
| `or_cost` 1, 9 gates, fan-in 3 | 1 | 995 s, 3 restarts | 2x15x7, 121 blocks, cout 12, s 19 |
| `or_cost` 1 | 2 | 937 s, 4 restarts | 2x11x9, 88 blocks, cout 7, s 11 |
| 8 gates, fan-in 4 | 1 | no layout in 3 restarts | |
| depth 3, 9 gates, fan-in 4 | 1 | no layout in 5 restarts | |

- Construction is where wide NORs cost.
  - Steps with a 3- or 4-input NOR ran out of their 60 s again and again:
    `g4 = NOR(a, cin, g3)`, `cout = NOR(g4, g5, g7)`, and the `or_cost` 1
    netlist's `g7 = NOR(cin, g3, g6)`.
  - Every input of such a NOR has to reach one block within a window two
    cells wide.
  - `nor9`'s two-input NORs construct on the first try in 1-1.5 minutes.
- Once built, the `or_cost` 1 netlist gave the smallest and fastest cell
  of these runs (seed 2: 88 blocks, 11 ticks).
- Compaction varies too much from run to run (the 600 s runs did not
  converge) to rank the netlists on two seeds.

Two placement steps had failed as infeasible within 0.1 s, the first time
a new input was placed. Construction offered the new input the cell of a
switch kept from an earlier step, and the fixed switch was read as the new
input's. That is fixed: new inputs skip placed switches, and a fixed
switch belongs to the input whose only site it is. With
`EXACT_DIAGNOSE_INFEASIBLE`, an infeasible step is re-solved with one
requirement dropped at a time, which is how it was found.

## Building wide NORs in parts

Netlists can hold OR nets (`NetDriver::Or`): nets that are the OR of their
inputs, with no torch. Construction places each as a step of its own.
`ExtractOptions::split_wide` (`EGRAPH_SPLIT=1`) builds every NOR of more
than two signals from the two halves of its OR, each half an OR net when it
is wider than one signal. OR nets are shared between gates through their
e-class.

Splitting makes every torch read at most two signals, but it does not
narrow the cross-section. The OR nets are signals too, so the 8-gate full
adder still keeps 6 nets alive at once. `GateOrder::MinLive` places the
ready gate after which the fewest nets are still needed, and it finds no
order better than 6 (or than 4 for `nor9` and the `or_cost` 1 netlist). The
width comes from the netlist itself.

Same settings as above:

| Netlist | Seed | Construction | Compacted |
| --- | --- | --- | --- |
| 8 gates, split (5 OR nets) | 1 | no layout in 4 restarts | |
| 8 gates, split | 2 | 589 s, 1 restart, 2x28x10, 402 blocks | 2x23x10, 327 blocks, cout 14, s 25 |
| `or_cost` 1, split (2 OR nets) | 1 | 719 s, 3 restarts, 2x22x10, 212 blocks | 2x13x9, 138 blocks, cout 13, s 18 |

Splitting made the 8-gate netlist placeable once, but large and slow. The
`or_cost` 1 netlist came out about like `nor9` (146 blocks, 13/18 ticks on
seed 1). The physical reason: an OR is free only where it is made, on the
block a torch reads. An OR net is carried on to later steps, and dust
cannot carry it there. Dust joining two lines powers both lines back, so a
carried OR needs a block and a repeater (a diode) after the merge, which
costs cells and a tick. In `nor9` every OR feeds a torch on the spot.

## Building a wide NOR on its support block

`NorNetlist::chain_wide_gates(2)` (`EGRAPH_CHAIN=2`) turns every wide NOR
into a left-leaning chain of OR nets, its inputs first:
`NOR(x1, x2, x3, x4)` becomes `s1 = OR(x1, x2)`, `s2 = OR(s1, x3)`,
`NOR(s2, x4)`. Construction builds such a chain (`chained_ors`: OR nets
read once, by a net that reads no other OR net) on one block:

- the first OR is observed on a block of the last slice
  (`ExactPlacerConfig::solid_observations`), and the block's position is
  kept;
- each later stage and the final NOR fix that block, keep its slice open
  (the frozen boundary stops there), and observe the next OR on the same
  block;
- gate orders place a chain's stages right before the NOR it ends in.

The OR is never carried, so no repeater is needed.

Width 2, height 10, 5 workers, windows up to 4, 60 s per step:

| Netlist | Seed | Steps placed / timed out | Attempts |
| --- | --- | --- | --- |
| 8 gates, chained | 1 | 18 / 11 | 3 restarts (g5, g5, cout), stopped |
| 8 gates, chained | 2 | 13 / 12 | 2 restarts (g6_or1, g5_or1), stopped |
| depth 3, chained | 1 | 15 / 10 | 2 restarts (g8, cout_or1), stopped |

The stages do place: an OR on a block and then its torch (`g4_or1`,
`g4`) took 9-28 s each. But other steps still ran out of time, a different
one in each attempt. None was proven infeasible; every failure was a 60 s
timeout.

A box 3 wide does not rescue them. More cells per slice make every step
slower, and `nor9` itself restarted twice there (8 steps placed, 7 timed
out). The 8-gate netlist placed 4 steps and timed out 9; chained, 9 and 7.

These netlists keep 6-7 nets alive where `nor9` keeps 4, however the
gates are ordered or split, and windowed construction pays for every
crossing net in every step. Each gain the e-graph finds over `nor9` either
needs wider NORs or more nets alive at once, and at the box size that
makes `nor9` work, construction cannot place either within its limits.

## Next

- Longer steps for these netlists (the timeouts were never proofs), to
  see whether they place at all and how they compact once placed.
- Count crossing width in extraction (a bound on nets alive at once along
  the construction order), so the e-graph proposes netlists construction
  can place: with `binary` the answer was `nor9` itself.
- Constrain extraction for carry tiles: the carry needs monotone signals
  (`carry_tiles.md`).

## Running

```sh
cargo test --release --lib egraph_netlists_compute_the_full_adder
cargo test --release --lib explore_egraph_netlists -- --ignored --nocapture
CIRCUIT=egraph-full-adder CIRCUIT_COMPACT_SECONDS=600 \
  cargo test --release --lib diagnose_construct_circuit -- --ignored --nocapture
```

`explore_egraph_netlists` takes these knobs:

- saturation: `EGRAPH_ITERATIONS`, `EGRAPH_NODES`, `EGRAPH_SECONDS`;
- the sweep: `EGRAPH_MAX_DEPTH`, `EGRAPH_OR_COSTS` (comma-separated),
  `EGRAPH_BINARY=1`;
- `EGRAPH_EXTRACT_SECONDS`;
- `EGRAPH_GREEDY=1` also prints greedy extractions.

The circuit harness takes `EGRAPH_DEPTH`, `EGRAPH_OR_COST`,
`EGRAPH_BINARY`, `EGRAPH_SPLIT`, `EGRAPH_CHAIN`, and
`CIRCUIT_ORDER=min-live`.
