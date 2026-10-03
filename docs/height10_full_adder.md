# Height-10 full-adder experiment

Follow-up (2026-09-22): [further compaction](compact_full_adder_shrink.md) reduces
this to 2x13x9 and 93 blocks. This 2x17x10 fixture remains the earlier reference.

Measured 2026-09-20 with the repository simulator. The requested height limit
was 10; thickness 2 and maximum width 20 were also retained during the search.
The resulting occupied and declared box is **2x17x10** in compiler `(x,y,z)`
coordinates (`z` is height). The NBT viewer permutes this to **17x10x2**.

| Fixture | Box volume | Blocks including supports | Physical switches |
| --- | ---: | ---: | ---: |
| Previous 2x20x20 | 800 | 259 | 3 |
| New 2x17x10 | 340 | 128 | 3 |

The new fixture is `test/full-adder-2x17x10.rcell`, with matching NBT and output
metadata. It has 42 automatic supports, 10 probes, and two outputs. Height is
halved, bounding-box volume falls 57.5%, and block count falls about 50.6%.
The previous passing fixture remains available for comparison.

## Lower logic before shortening routes

This is a new manual layout, not a coordinate compression of the previous
vertical layout and not a successful run of the automatic local placer.
It implements the nine-NOR full-adder decomposition directly:

```text
n1 = NOR(a,b)
n2 = NOR(a,n1)       n3 = NOR(b,n1)
xnor = NOR(n2,n3)
n5 = NOR(xnor,cin)
n6 = NOR(xnor,n5)    n7 = NOR(cin,n5)
sum = NOR(n6,n7)
cout = NOR(n1,n5)
```

The reusable core is a low four-NOR XNOR arrangement. A wall torch sends the
first NOR result sideways along the opposite thickness plane. Two repeaters
feed that result into the outer NOR supports, which also receive their
respective original inputs. Top torches on those outer supports expose their
outputs two levels higher through solid-block pickups. Two opposing repeaters
then combine those outputs at the final NOR. With two directly attached input
switches, the first core occupies 2x7x6 including the floor supports.

This folds sibling branches horizontally instead of lifting every logic edge.
The first XNOR output is at `(1,3,5)`. A second instance is shifted along Y and
up three levels, giving sum at `(1,13,8)`. A short descending dust route connects
the instances. Mapping both full-adder stages vertically was the principal
height cost in the earlier layout.

## A switch contact is not a reusable wire input

The first successful standalone XNOR used switches that powered both a gate
support and an adjacent input branch. Connecting the second instance with wire
required a different input adapter. Simply splitting dust toward both NORs
created a feedback loop: an input repeater hard-powered the outer NOR support,
which powered the adjacent input dust back toward that same repeater. The
source XNOR remained correct but `second_input` became permanently high.

The repeater at `(0,10,4)` isolates that branch. Its direction is essential;
replacing it with dust reproduces the failure in
`low_xnor_input_branch_requires_backfeed_isolation`.

For composition, a recipe needs separate contracts for direct switch contacts
and routed inputs. Test the joined recipe, including both fanout consumers.
A standalone truth table does not establish a safe physical connection.

## Carry bypass and electrical clearance

Carry uses the original `n1` and `n5`, avoiding the extra recomputation gate
from the 2x20x20 fixture. Four relay inversions lift `n1` to `(0,3,8)`; two lift
`n5` to `(0,13,7)`. Their top routes meet at the final NOR support `(0,11,9)`;
the wall torch at `(1,11,9)` is `cout`. Every repeater uses delay 1 in this layout.

An initial attempt to tap `n1` with dust immediately above its torch failed:
the dust was also adjacent to the first XNOR's hard-powered output support.
Its rising stairs additionally touched an XNOR branch. Replacing that pickup
with attached-torch relays and moving the XNOR connection to the other plane
separated the signals. Reserve the relay's entire electrical neighborhood,
not just its occupied column.

## Validation and reproduction

All eight fresh input cases check every probe plus sum and carry against
expressions over the three original inputs. All 64 ordered input transitions
pass with 61 idle cycles between input changes, checking every observation and
rejecting any burned-out torch. The negative input-adapter regression also
passes by detecting the expected fault.

```sh
cargo run --release --locked --bin rcell -- \
  test/full-adder-2x17x10.rcell test/full-adder-2x17x10.nbt
cargo test --release --locked --lib physical_cell::
```

No simulator or production placer behavior was changed. These checks establish
settled manual-input behavior in this simulator; they do not prove minimum
size, a maximum clock rate, glitch-free outputs, Minecraft equivalence, or
electrical clearance for arbitrary neighboring cells.

The viewer link uses `?example=full-adder-2x17x10.nbt`; output metadata supplies
the `sum` and `cout` names. Rebuild the viewer's example bundle after changing
the NBT.
