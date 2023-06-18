# redstone compiler

![](image.png)

From the left side...

- OR := A ∧ B
- NOT := ¬ A
- NOR := ¬ (A ∧ B) := NOT(OR)

## Block Optimizer

## HDL

### Layering

### Routing

## IR

## Compiler

## Simulator

Build Graph

## Test

```ps1
$env:RUST_LOG="debug"; cargo test unittest_simulator_init_states -- --nocapture
```
