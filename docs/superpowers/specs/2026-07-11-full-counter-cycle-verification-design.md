# Full Counter Cycle Verification

## Problem

The global PnR counter verifier accepted a physical world after checking only
the first two rising clock edges. The accepted artifact produced `0 -> 1 -> 2`
but failed on the next transition, leaving `q0` low and preventing the counter
from reaching `3` or wrapping to `0`.

The observed wall torch was not burned out: its support block was powered and
the simulator's burnout set was empty. It reflected the bad sequential state
created by the accepted placement.

## Design

Strengthen `assert_two_bit_counter_behavior` to verify the complete two-bit
cycle `0 -> 1 -> 2 -> 3 -> 0`. After every rising edge, also drive the clock
low and verify that the output holds. Global PnR must reject any routed world
that fails any edge in this sequence.

Add an artifact-level simulator regression that loads
`counter-global-smoke.nbt` and its output metadata, discovers the clock switch,
and independently verifies the same full cycle. This prevents a stale or
manually replaced artifact from silently weakening coverage.

## Verification

- Observe the artifact regression fail on the previously accepted NBT at the
  expected `2 -> 3` transition.
- Run the search-heavy counter global PnR test with the stronger verifier and
  generate a passing artifact.
- Run the independent artifact regression and the full release suite.
- Rebuild viewer WASM and exercise four clock pulses in the viewer.

## Scope

This change does not alter simulator torch semantics or burnout behavior. It
only strengthens semantic acceptance of global PnR results and replaces the
known-bad generated counter artifact.
