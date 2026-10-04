# Carry tiles: a full adder that tiles into an n-bit adder (2026-10-04)

`ExactPlacerConfig::carry` (`CarryTiling`) lays a cell out as a tile that
repeats along X. Each copy passes a carry to the next, so one full-adder tile
placed n times side by side is an n-bit ripple-carry adder: operands on the
Y-min face, sums on the Y-max face, the carry crossing each seam.

Code: the tiling rules in `exact_placer.rsdsl` (section "tiling"),
`CarryTiling` in `exact/mod.rs`, the carry steps in `construct.rs`
(`construction_steps`), and `assemble_chain` in `exact/tiling.rs`. Harnesses
in `exact/tests.rs`: `diagnose_carry_adder_tile`, `check_carry_adder_chain`,
`check_long_carry_adder_chain`, `trace_carry_adder_chain_case`, and the
probes `diagnose_carry_only_tile` and `diagnose_monotone_carry_cell`.

## How the carry crosses the seam

A 2-wide tile has its carry in and carry out in one `(y, z)` column of two
cells. A repeater cannot pass the carry: a repeater at x = 1 pointing +X reads
the cell behind it, which is the tile's own carry-in cell. A torch can. The
next tile's carry-in torch, at its x = 0, is attached across the seam to this
tile's carry-out block at x = 1. That torch is the carry's final NOR gate, so
the block carries the complement of the carry (an OR of the gate's inputs),
and the tile reads its carry in as the complement, from the block its own
carry-in torch is attached to. A torch next to a block it is not attached to
does nothing to it, so the carry-in torch and the carry-out block may sit side
by side in the same column.

## Model

The box is the tile behind two ghost slices that stand for the previous tile:

- x = 0 holds the carry switch (the netlist input `ncin`, the carry in's
  complement), attached East to the ghost block at x = 1, which stands for the
  previous tile's carry-out block. The tile's torch at x = 2 reads it.
- Ghost cells hold nothing else.
- The carry column: where the switch is, the tile's first slice holds the
  carry-in torch on the ghost block, and its last slice holds a block observed
  as the carry output (`ncout`, a `solid_output`).
- Seam isolation: the cell beyond the tile's first (last) slice is a copy of
  the cell in its last (first) slice, carrying another bit's signals, so
  nothing may interact across the seam. No dust beside dust, torches, switches,
  or repeaters on the X axis; no dust staircase across; no dust pointing at a
  block across; no torch beside dust or a repeater reading it; no repeater on
  the X axis unless the cell across is air; no strongly powered block beside
  dust, and no block read by a repeater across; no switch with anything across.
  In a 2-wide tile every cell is on the seam, so the two X slices can only
  talk through torches attached sideways: a torch powers the block above it
  and dust or repeaters beside it, never a block beside it.
- Monotone signals (below).

These rules ground only with `carry` set; every other problem keeps a
byte-identical CNF (`print_cnf_hashes`).

## Chains

`assemble_chain` places n tiles side by side. In front, a switch `cin` drives
the first carry in through a block, an inverter, and a repeater into the
block the first carry-in torch reads, so `cin` has the carry's own polarity.
Behind the last tile, a torch attached to its carry-out block is `cout`. The
`expect` lines compose the tile's netlist bit by bit. The simulator checks the
single tile in the placer; only a chain shows whether the seam rules held.

## Glitches decide the carry logic

The first tile used `nor9`'s carry, `ncout = n1 | n5` with
`n5 = NOR(n4, cin)` and `n4` the XNOR of the operands. Construction took
127 s (the carry steps in one window of 3, 30 s) and compaction reached
2x12x8 with 111 blocks in 554 s. Chains of 1 to 3 bits passed every case, but
4 bits failed 1 of 512 cases and 5 bits 7 of 2048: with `a = 0` and `b = 1` in
every bit, a torch in the last tile burned out (8 toggles in 60 cycles). `n4`
is not monotone, so the carry-out block glitches while the operands change;
the next tile's `n5` and `n7` turn the glitch into several, and the toggles
add up down the chain.

The carry now uses every prime implicant of the complement of the majority:
`ncout = NOR(a, b) | NOR(a, cin) | NOR(b, cin)` (`m1`, `m2`: two more
torches; `carry_adder_graph`). While the inputs only rise, as in every case
of the chain check (settled from all-off, then all inputs set), each term
only falls, so the block does not glitch.

That was not enough. Classes are functions, and the solver powered the
carry-out block with signals of the same steady values that went through the
XOR (an 8-bit chain still burned a sum torch out at 85 + 170 + 1: the carry-in
torch of bit 6 pulsed). The model rule "단조" now says: with every input moving
its tile's way (operands and carry in rising, so `ncin` falling), a cell whose
class only rises (falls) is powered only by cells whose classes only rise
(fall). By induction each such cell changes once. `increasing` and
`decreasing` classes are computed in `dsl.rs`; `monotone_signals = false`
turns the rule off (the `nor9`-carry comparison needs that).

## Construction

The carry switch and the carry-out block share a column, so the carry path
(gates that depend on the carry input and feed the carry output) leaves the
column and must come back to it, while construction freezes Y slices behind
it.

What did not work:

- The whole path in one step. Even the monotone carry alone (`n1`, `m1`,
  `m2` and the block, no sum) found no layout in 5 minutes in monolithic
  solves (`diagnose_carry_only_tile` in 2- and 3-wide tiles,
  `diagnose_monotone_carry_cell` as an ordinary cell in 2x6x6 to 2x8x8),
  while construction places the same four gates as an ordinary cell in
  27 s, one gate at a time.
- Two steps, the carry-in torch on the window's last slice and then the rest
  re-solving that slice (overlap 2, the column kept off the floor, 180 s to
  25 minutes per window): no layout in 2- or 3-wide tiles, with or without the
  monotone rule. Before the rule, the same step took 6.7 s because the block
  could use `n5` and needed only two feeders. Every carry term has to reach
  one cell, and in a 2-wide tile the cell's four neighbors share columns with
  the carry-in torch's four outputs.

What works (`construction_steps`, `construct_step`):

1. The carry comes right after the gates it needs (`n1`), so few other nets
   cross the seams while it is built.
2. The gate that reads the carry input (`cin`) is one step, with the switch on
   the window's last slice, at any height but the floor.
3. From then until the carry output is placed, the carry-out cell is fixed as
   an unpowered block, so nothing may power it, and a corridor from it along
   +Y in the last X slice is kept empty with blocks under it.
4. The other carry gates follow one at a time; an OR and the NOR the netlist
   realizes it with are one step. The netlist ORs the carry in two levels
   (`mm = m1 | m2`, then `ncout = n1 | mm`), so no block needs more than two
   feeders.
5. The carry output is the last carry step: it ORs `n1` and `mm` onto a block,
   and a line of repeaters in the corridor carries the result back to the
   carry-out block. The corridor and the block are the only cells re-solved
   behind the window.

## Results

| Tile | Construction | Compacted | Chains |
| --- | --- | --- | --- |
| `nor9` carry, 2 wide, height 8 | 127 s, 2x17x8, 122 blocks | 2x12x8, 111 blocks (554 s) | 1-3 bits pass; 4 bits fail 1/512, 5 bits 7/2048 (burnout) |
| monotone netlist, no monotone rule, 2 wide, height 8 | 68 s, 2x20x8, 165 blocks | 2x10x7, 80 blocks (500 s) | 1-5 bits pass every case; 8 bits fail at 85 + 170 + 1 (burnout) |
| monotone carry and rule, corridor, 2 wide, height 10 | 268 s, 2x28x10, 303 blocks | 2x17x10, 175 blocks (1200 s, still shrinking); recompacted to 2x16x10, 164 blocks (1135 s, converged) | 1-5 bits pass every case; 8 and 16 bits pass 208 cases each and a 200-step random walk |

Block counts are the tile's (the ghost block excluded). Chains are checked by
`diagnose_carry_adder_tile` (every case, 1 to 5 bits) and
`check_long_carry_adder_chain` (worst-case carry patterns, random cases, and a
random walk of input changes on one simulator, checking every output and
torch burnout after each change).

The monotone tile is about twice the size of the glitching one: two more
torches, the corridor, and a taller box.

## Fixtures

- `test/adder-carry-tile-2x16x10.rcell` (and `.nbt`): the tile, 164 blocks,
  in its 4x16x10 box with the ghost switch `ncin` and ghost block in front.
  Operands `a` at (2, 0, 7) and `b` at (3, 0, 2) on the Y-min face, `s` on the
  Y-max face at (2, 15, 5), the carry column at (y, z) = (2, 7).
- `test/adder-carry-chain4-13x16x10.rcell` (and `.nbt`, viewer example): four
  tiles, `cin` in front, `cout` behind, and each carry-out block as an output
  (`ncout0` ...). `assemble_chain` names them so each bit's `expect` line reads
  the previous carry by name; written out, the carry expression doubled with
  every bit (a 16-bit chain's RCELL was 12 MB).

`physical_cell` tests check the chain's arithmetic in all 512 cases and the
tile's 8 cases and 64 settled transitions; `exact` tests check that the tile
still satisfies the model with every cell fixed and that two copies chain.
