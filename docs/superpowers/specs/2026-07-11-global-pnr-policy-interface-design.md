# Global PnR Policy Interface Design

## Goal

Make global PnR heuristics independently selectable, configurable, and observable without changing local placement or redstone routing semantics.

## Design

`GlobalPnrConfig` keeps the existing candidate, placement, routing, and verifier boundaries. Global search policy is split into three explicit values:

- `GlobalSearchBudget`: bounded candidate, layout-combination, and placement attempt counts.
- `GlobalPnrPolicies`: ordered placement heuristics and net-order strategies.
- `PlacementCostWeights`: objective weights used to rank placement attempts.

Placement generation is selected by `PlacementHeuristic` enum values instead of a hard-coded function sequence. Initial variants cover the existing shelf, grid, and register-specific generators. New 3D, annealing, or congestion-aware generators can be added as variants without modifying the orchestration loop.

Placement scoring returns `PlacementCostBreakdown`, containing volume, estimated wire length, and vertical-distance components. The weighted total remains deterministic and default weights preserve existing behavior.

## Presets

`GlobalPnrPreset::{Fast, Balanced, Thorough}` constructs complete global configs. Callers may override individual nested fields after selecting a preset.

## Compatibility

The balanced preset and `Default` use the current heuristic order and limits. Existing preferred child layouts remain the first search combination, and existing route contracts and verifier calls remain unchanged.

## Testing

- presets expose monotonically increasing search budgets;
- disabling a placement heuristic prevents it from generating attempts;
- cost weights change ranking while the default reproduces the previous total;
- existing global PnR tests and ignored counter smoke remain the end-to-end regressions.
