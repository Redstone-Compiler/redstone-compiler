# Readable full-adder experiment

Start new manual experiments from `test/full-adder-baseline.rcell`.
The old 2x20x20 cell and its NBT are preserved under `test/rcell/archive/`
as `full-adder-2x20x20-disconnected.*`. The archived cell intentionally fails
carry verification; its diagnosis is in `rcell_full_adder_diagnostic.md`.

A subsequently verified 2x20x20 implementation is available in
`test/full-adder-2x20x20.rcell`. See `compact_full_adder_rules.md` for the
compact layout's provenance, validation, and lessons for local placement.

## Geometry and logic

The new reference uses **48x22x4** compiler coordinates `(x,y,z)`, where `z`
is height. It prioritizes readable, working wiring, not the old compact bound.
There are three physical switches, one each for `a`, `b`, and `cin`; no logical
input is replicated and no intermediate signal is externally driven.

Only two heights carry routing:

- `z=1`: two repeated four-NOR cones, flowing left to right. The first computes
  `a XNOR b`; the second computes `(a XNOR b) XNOR cin`, which is `sum`.
- `z=3`: independent `n1` and `n5` carry lanes terminate at a final NOR gate.
  The two source torches power pickup blocks directly above them at `z=2`.

Dust and repeater supports are automatic. The two pickup blocks are explicit.
Glyph `T` is a wall torch attached toward `x-`. Repeater glyphs `E`, `S`, and
`N` describe output flow toward `+X`, `+Y`, and `-Y`, respectively; the RCELL
`toward` field stores the opposite, input-facing direction.

| Signal | Gate expression | Torch coordinate |
| --- | --- | --- |
| n1 | NOR(a, b) | (7,8,1) |
| n2 | NOR(a, n1) | (15,4,1) |
| n3 | NOR(b, n1) | (15,12,1) |
| xnor_ab | NOR(n2, n3) | (23,8,1) |
| n5 | NOR(xnor_ab, cin) | (31,8,1) |
| n6 | NOR(xnor_ab, n5) | (39,4,1) |
| n7 | NOR(cin, n5) | (39,12,1) |
| sum | NOR(n6, n7) | (47,8,1) |
| cout | NOR(n1, n5) | (41,18,3) |

Seven logic probes and three carry probes make every stage observable. In
particular, `carry_n1` and `carry_n5` check the transported signals, and
`carry_or` checks the final torch's support. All expectations are expressions
of the original three inputs, so a wrong intermediate value cannot become the
expected value of its consumer.

## Running and changing the experiment

```sh
cargo run --release --locked --bin rcell -- \
  test/full-adder-baseline.rcell /tmp/full-adder-baseline.nbt
cargo test --release --locked --lib physical_cell::
```

The baseline passes all eight input combinations, including all ten probes
and both outputs. In case-mask order (`a` bit 0, `b` bit 1, `cin` bit 2),
`sum=01101001` and `cout=00010111`. The exported world contains 482 blocks,
including 227 automatic supports.

Keep this as a verified reference when starting a smaller candidate. Change
one gate region or one route at a time and require every observation to pass.
Preserve the carry lanes while shrinking the repeated logic cones first;
the archived experiment demonstrates why leaving carry routing until the
end can produce a disconnected final output.

Fresh-case correctness is separate from timing or transition correctness.
The regression suite also checks all 64 ordered input-state pairs, with a
settled initial state and `MANUAL_INPUT_IDLE_CYCLES` (61 cycles) before the
next input. It checks every probe/output and rejects any torch burnout.
This is a settled manual-input contract, not a maximum clock-rate claim or
an assertion that intermediate glitches never occur.
