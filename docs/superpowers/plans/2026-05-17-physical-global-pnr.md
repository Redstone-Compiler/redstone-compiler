# Physical Global P&R Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a physically valid global place-and-route path that composes two half-adder macros plus one OR macro into a routed full-adder NBT, with deterministic IO panels, routable macro pins, shared Minecraft legality checks, route interference checks, and validation tests.

**Architecture:** Rebuild P&R around explicit physical contracts. Local placement produces macro candidates with boundary pins, not arbitrary internal node positions or scattered switches. Global placement chooses macro candidates and positions only after a physical router proves every `MacroNet` can be routed with redstone/repeater legality.

**Tech Stack:** Rust 2021, existing `World3D`, `BlockKind`, `LocalPlacer`, `LogicGraph`, NBT export via `NBTRoot`, release-mode tests with `cargo test --release`.

---

## Current Problems To Fix

The current experimental implementation is not an acceptable final foundation:

- `Switch` blocks are generated as arbitrary local graph inputs, so macro inputs are not real external pins.
- `LayoutCandidate` exposes internal graph node positions as ports, which are not guaranteed routable from outside the macro.
- The temporary channel router draws redstone without reusing the existing Minecraft legality model.
- Signal reach, repeater needs, and other Minecraft constraints are not modeled consistently in global routing.
- IO switches are not placed in a deterministic panel.
- Placement can score short estimated nets even when no legal physical route exists.

The new flow must enforce this invariant:

```text
Top-level switches live only in IO panels.
Macro inputs are redstone/repeater sink pins on a macro boundary.
Macro outputs are redstone/repeater source pins on a macro boundary.
Global routing connects only boundary pins.
Every routed net is validated by the same placement/conflict/short-circuit rules used by local placement.
```

Important correction: do not implement a separate simplified physics model around a hard-coded "15 blocks" rule. The global router must reuse or extract the existing local placement judge, especially `PlacedNode::has_conflict`, `PlacedNode::has_short`, `PlaceBound`, propagation bounds, cobble support rules, and route rejection reasons. Signal-strength or repeater policy can be layered on later, but collision and electrical legality must not diverge from local placement.

---

## File Structure

### Create

- `src/transform/place_and_route/physical.rs`
  - Physical routing primitives, redstone segment validation, support block helpers, route validation errors.

- `src/transform/place_and_route/io_panel.rs`
  - Deterministic top-level input switch and output probe panel generator.

- `src/transform/place_and_route/macro_candidate.rs`
  - Macro-specific candidate builder that converts local placement results into pin-constrained macro candidates.

- `src/transform/place_and_route/global_router.rs`
  - Global router that routes `MacroNet`s between boundary pins and inserts repeaters before redstone strength exceeds 15.

- `src/transform/place_and_route/full_adder_pnr.rs`
  - Full-adder composition target: `ha#0`, `ha#1`, `cout_or`, IO panel, global P&R, NBT export.

### Modify

- `src/transform/place_and_route/mod.rs`
  - Export the new modules.

- `src/transform/place_and_route/layout_candidate.rs`
  - Replace generic internal `positions` ports with explicit physical boundary ports.

- `src/transform/place_and_route/local_placer.rs`
  - Add a macro-placement mode that can use redstone input pins instead of switch inputs.

- `src/graph/logic.rs`
  - Keep `buffered_binary_half_adder_graph()` as the half-adder macro logic fixture.

- `docs/global_place_route_m5_m6.md`
  - Mark the old channel-router approach as superseded by this physical plan.

---

## Target Data Model

Use these definitions as the target shape. Names can be adjusted only if all tests and call sites are updated consistently.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalPortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalPortSide {
    West,
    East,
    South,
    North,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPort {
    pub name: String,
    pub direction: PhysicalPortDirection,
    pub side: PhysicalPortSide,
    pub pin: Position,
    pub facing: Direction,
}

#[derive(Debug, Clone)]
pub struct MacroLayoutCandidate {
    pub unit_id: String,
    pub world_fragment: World3D,
    pub bbox: BoundingBox,
    pub ports: Vec<PhysicalPort>,
    pub occupied_cells: HashSet<Position>,
    pub blocked_cells: HashSet<Position>,
    pub cost: usize,
}
```

---

## Task 1: Add Physical Route Validation

**Files:**
- Create: `src/transform/place_and_route/physical.rs`
- Modify: `src/transform/place_and_route/mod.rs`
- Test: `src/transform/place_and_route/physical.rs`

- [ ] **Step 1: Write failing tests for redstone distance and support rules**

Add this test module to `physical.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{Block, BlockKind, Direction};
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

    fn redstone() -> Block {
        Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        }
    }

    fn cobble() -> Block {
        Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        }
    }

    #[test]
    fn rejects_redstone_chain_longer_than_15_without_repeater() {
        let mut world = World3D::new(DimSize(20, 3, 2));
        for x in 0..17 {
            world[Position(x, 1, 0)] = cobble();
            world[Position(x, 1, 1)] = redstone();
        }

        let error = validate_redstone_run(&world, Position(0, 1, 1), Position(16, 1, 1))
            .expect_err("17 redstone dust cells require a repeater");

        assert_eq!(
            error,
            PhysicalRouteError::RedstoneRunTooLong {
                from: Position(0, 1, 1),
                to: Position(16, 1, 1),
                length: 16,
            }
        );
    }

    #[test]
    fn accepts_redstone_chain_split_by_repeater() {
        let mut world = World3D::new(DimSize(20, 3, 2));
        for x in 0..17 {
            world[Position(x, 1, 0)] = cobble();
            world[Position(x, 1, 1)] = redstone();
        }
        world[Position(8, 1, 1)] = Block {
            kind: BlockKind::Repeater {
                is_on: false,
                is_locked: false,
                delay: 1,
                lock_input1: None,
                lock_input2: None,
            },
            direction: Direction::East,
        };

        validate_redstone_run(&world, Position(0, 1, 1), Position(16, 1, 1)).unwrap();
    }

    #[test]
    fn rejects_floating_redstone_without_support_block() {
        let mut world = World3D::new(DimSize(4, 3, 2));
        world[Position(1, 1, 1)] = redstone();

        let error = validate_supported_redstone(&world).expect_err("floating redstone is illegal");

        assert_eq!(
            error,
            PhysicalRouteError::UnsupportedRedstone {
                position: Position(1, 1, 1)
            }
        );
    }
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
cargo test --release physical -- --nocapture
```

Expected: compile failure because `physical` module and validation functions are not defined.

- [ ] **Step 3: Implement validation types and functions**

Create `physical.rs` with:

```rust
use crate::world::block::{BlockKind, Direction};
use crate::world::position::Position;
use crate::world::World3D;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalRouteError {
    UnsupportedRedstone { position: Position },
    RedstoneRunTooLong {
        from: Position,
        to: Position,
        length: usize,
    },
    NonAxisAlignedRun { from: Position, to: Position },
}

pub fn validate_supported_redstone(world: &World3D) -> Result<(), PhysicalRouteError> {
    for (position, block) in world.iter_block() {
        if !(block.kind.is_redstone() || block.kind.is_repeater()) {
            continue;
        }

        let Some(bottom) = position.down() else {
            return Err(PhysicalRouteError::UnsupportedRedstone { position });
        };

        if !world[bottom].kind.is_cobble() {
            return Err(PhysicalRouteError::UnsupportedRedstone { position });
        }
    }

    Ok(())
}

pub fn validate_redstone_run(
    world: &World3D,
    from: Position,
    to: Position,
) -> Result<(), PhysicalRouteError> {
    let positions = axis_aligned_positions(from, to)?;
    let mut run_start = from;
    let mut run_len = 0;

    for position in positions.into_iter().skip(1) {
        let block = world[position];
        if block.kind.is_repeater() {
            run_start = position;
            run_len = 0;
            continue;
        }

        if block.kind.is_redstone() {
            run_len += 1;
            if run_len > 15 {
                return Err(PhysicalRouteError::RedstoneRunTooLong {
                    from: run_start,
                    to: position,
                    length: run_len,
                });
            }
        }
    }

    Ok(())
}

pub fn axis_aligned_positions(
    from: Position,
    to: Position,
) -> Result<Vec<Position>, PhysicalRouteError> {
    let axis_count = usize::from(from.0 != to.0)
        + usize::from(from.1 != to.1)
        + usize::from(from.2 != to.2);
    if axis_count > 1 {
        return Err(PhysicalRouteError::NonAxisAlignedRun { from, to });
    }

    let mut positions = Vec::new();
    let mut cursor = from;
    positions.push(cursor);
    while cursor != to {
        cursor = step_towards(cursor, to);
        positions.push(cursor);
    }
    Ok(positions)
}

pub fn step_towards(from: Position, to: Position) -> Position {
    if from.0 < to.0 {
        Position(from.0 + 1, from.1, from.2)
    } else if from.0 > to.0 {
        Position(from.0 - 1, from.1, from.2)
    } else if from.1 < to.1 {
        Position(from.0, from.1 + 1, from.2)
    } else if from.1 > to.1 {
        Position(from.0, from.1 - 1, from.2)
    } else if from.2 < to.2 {
        Position(from.0, from.1, from.2 + 1)
    } else if from.2 > to.2 {
        Position(from.0, from.1, from.2 - 1)
    } else {
        from
    }
}
```

Modify `mod.rs`:

```rust
pub mod physical;
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
cargo test --release physical -- --nocapture
```

Expected: all `physical` tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/physical.rs src/transform/place_and_route/mod.rs
git commit -m "Add physical route validation"
```

---

## Task 2: Define Deterministic IO Panels

**Files:**
- Create: `src/transform/place_and_route/io_panel.rs`
- Modify: `src/transform/place_and_route/mod.rs`
- Test: `src/transform/place_and_route/io_panel.rs`

- [ ] **Step 1: Write failing tests for switch panel placement**

Create `io_panel.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::BlockKind;

    #[test]
    fn input_panel_places_switches_in_name_order_on_west_edge() {
        let panel = build_input_panel(&["cin", "a", "b"], 3);

        assert_eq!(
            panel.ports.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "cin"]
        );
        assert_eq!(panel.ports[0].pin, Position(0, 1, 1));
        assert_eq!(panel.ports[1].pin, Position(0, 4, 1));
        assert_eq!(panel.ports[2].pin, Position(0, 7, 1));

        for port in &panel.ports {
            assert!(matches!(panel.world[port.pin].kind, BlockKind::Switch { .. }));
        }
    }

    #[test]
    fn output_panel_places_probe_redstone_in_name_order_on_east_edge() {
        let panel = build_output_panel(&["cout", "s"], 3, 12);

        assert_eq!(
            panel.ports.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["cout", "s"]
        );
        assert_eq!(panel.ports[0].pin, Position(12, 1, 1));
        assert_eq!(panel.ports[1].pin, Position(12, 4, 1));

        for port in &panel.ports {
            assert!(panel.world[port.pin].kind.is_redstone());
        }
    }
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
cargo test --release io_panel -- --nocapture
```

Expected: compile failure for missing `build_input_panel`, `build_output_panel`, and panel types.

- [ ] **Step 3: Implement deterministic panel builder**

Add:

```rust
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelPort {
    pub name: String,
    pub pin: Position,
}

#[derive(Debug, Clone)]
pub struct IoPanel {
    pub world: World3D,
    pub ports: Vec<PanelPort>,
}

pub fn build_input_panel(names: &[&str], spacing: usize) -> IoPanel {
    let mut names = names.iter().copied().collect::<Vec<_>>();
    names.sort();
    let height = 2;
    let depth = names.len().saturating_mul(spacing) + 2;
    let mut world = World3D::new(DimSize(1, depth.max(2), height));
    let mut ports = Vec::new();

    for (index, name) in names.iter().enumerate() {
        let pin = Position(0, index * spacing + 1, 1);
        world[Position(0, pin.1, 0)] = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        };
        world[pin] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::Bottom,
        };
        ports.push(PanelPort {
            name: (*name).to_string(),
            pin,
        });
    }

    IoPanel { world, ports }
}

pub fn build_output_panel(names: &[&str], spacing: usize, x: usize) -> IoPanel {
    let mut names = names.iter().copied().collect::<Vec<_>>();
    names.sort();
    let height = 2;
    let depth = names.len().saturating_mul(spacing) + 2;
    let mut world = World3D::new(DimSize(x + 1, depth.max(2), height));
    let mut ports = Vec::new();

    for (index, name) in names.iter().enumerate() {
        let pin = Position(x, index * spacing + 1, 1);
        world[Position(x, pin.1, 0)] = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        };
        world[pin] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
        ports.push(PanelPort {
            name: (*name).to_string(),
            pin,
        });
    }

    IoPanel { world, ports }
}
```

Modify `mod.rs`:

```rust
pub mod io_panel;
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
cargo test --release io_panel -- --nocapture
```

Expected: panel tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/io_panel.rs src/transform/place_and_route/mod.rs
git commit -m "Add deterministic IO panels"
```

---

## Task 3: Replace Internal Layout Ports With Boundary Macro Pins

**Files:**
- Create: `src/transform/place_and_route/macro_candidate.rs`
- Modify: `src/transform/place_and_route/layout_candidate.rs`
- Modify: `src/transform/place_and_route/mod.rs`
- Test: `src/transform/place_and_route/macro_candidate.rs`

- [ ] **Step 1: Write failing test that macro candidates have no switches**

Create `macro_candidate.rs` test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::predefined_logics;
    use crate::transform::place_and_route::local_placer::{
        InputPlacementStrategy, LocalPlacer, LocalPlacerConfig, NotRouteStrategy,
        PlacementSamplingPolicy, TorchPlacementStrategy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;
    use crate::world::block::BlockKind;
    use crate::world::position::DimSize;

    fn fast_macro_config() -> LocalPlacerConfig {
        LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(2048),
            placement_sampling_policy: PlacementSamplingPolicy::Cost {
                count: 128,
                random_count: 16,
                start_step: 6,
            },
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 3,
            not_route_step_sampling_policy: SamplingPolicy::Random(128),
            max_route_step: 3,
            route_step_sampling_policy: SamplingPolicy::Random(128),
        }
    }

    #[test]
    fn half_adder_macro_candidate_exposes_only_boundary_pins() -> eyre::Result<()> {
        let graph = predefined_logics::buffered_binary_half_adder_graph()?;
        let placer = LocalPlacer::new(graph, fast_macro_config())?;
        let local = placer
            .generate_layout_candidates("half_adder", DimSize(12, 12, 5), None)
            .into_iter()
            .min_by_key(|candidate| candidate.cost)
            .expect("half adder local candidate");

        let candidate = MacroCandidateBuilder::new("half_adder")
            .west_inputs(["a", "b"])
            .east_outputs(["c", "s"])
            .build_from_local_candidate(&local)?;

        assert_eq!(
            candidate.ports.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c", "s"]
        );
        assert!(candidate.ports.iter().all(|p| p.pin.0 == 0 || p.pin.0 == candidate.bbox.max.0));
        assert!(candidate
            .world_fragment
            .iter_block()
            .into_iter()
            .all(|(_, block)| !matches!(block.kind, BlockKind::Switch { .. })));

        Ok(())
    }
}
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
cargo test --release half_adder_macro_candidate_exposes_only_boundary_pins -- --nocapture
```

Expected: compile failure for `MacroCandidateBuilder`.

- [ ] **Step 3: Implement macro candidate builder shell**

Add:

```rust
use std::collections::HashSet;

use itertools::Itertools;

use crate::transform::place_and_route::estimate::{bounding_box, world_compact_cost, BoundingBox};
use crate::transform::place_and_route::layout_candidate::{LayoutCandidate, PortDirection};
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalPortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalPortSide {
    West,
    East,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPort {
    pub name: String,
    pub direction: PhysicalPortDirection,
    pub side: PhysicalPortSide,
    pub pin: Position,
    pub facing: Direction,
}

#[derive(Debug, Clone)]
pub struct MacroLayoutCandidate {
    pub unit_id: String,
    pub world_fragment: World3D,
    pub bbox: BoundingBox,
    pub ports: Vec<PhysicalPort>,
    pub occupied_cells: HashSet<Position>,
    pub blocked_cells: HashSet<Position>,
    pub cost: usize,
}

#[derive(Debug, Clone)]
pub struct MacroCandidateBuilder {
    unit_id: String,
    west_inputs: Vec<String>,
    east_outputs: Vec<String>,
    pin_spacing: usize,
}

impl MacroCandidateBuilder {
    pub fn new(unit_id: impl Into<String>) -> Self {
        Self {
            unit_id: unit_id.into(),
            west_inputs: Vec::new(),
            east_outputs: Vec::new(),
            pin_spacing: 3,
        }
    }

    pub fn west_inputs<const N: usize>(mut self, names: [&str; N]) -> Self {
        self.west_inputs = names.into_iter().map(str::to_string).sorted().collect();
        self
    }

    pub fn east_outputs<const N: usize>(mut self, names: [&str; N]) -> Self {
        self.east_outputs = names.into_iter().map(str::to_string).sorted().collect();
        self
    }

    pub fn build_from_local_candidate(
        &self,
        local: &LayoutCandidate,
    ) -> eyre::Result<MacroLayoutCandidate> {
        let translated = local.translated(Position(4, 4, 0));
        let bounds = bounding_box(&translated.world_fragment)
            .ok_or_else(|| eyre::eyre!("local candidate has no blocks"))?;
        let width = bounds.max.0 + 5;
        let port_count = self.west_inputs.len().max(self.east_outputs.len());
        let depth = translated
            .world_fragment
            .size
            .1
            .max(port_count.saturating_mul(self.pin_spacing) + 2);
        let height = translated.world_fragment.size.2.max(2);
        let mut world = World3D::new(DimSize(width + 1, depth, height));

        for (position, block) in translated.world_fragment.iter_block() {
            if matches!(block.kind, BlockKind::Switch { .. }) {
                world[position] = Block {
                    kind: BlockKind::Redstone {
                        on_count: 0,
                        state: 0,
                        strength: 0,
                    },
                    direction: Direction::None,
                };
            } else {
                world[position] = block;
            }
        }

        let mut ports = Vec::new();
        for (index, name) in self.west_inputs.iter().enumerate() {
            let pin = Position(0, index * self.pin_spacing + 1, 1);
            place_pin(&mut world, pin);
            ports.push(PhysicalPort {
                name: name.clone(),
                direction: PhysicalPortDirection::Input,
                side: PhysicalPortSide::West,
                pin,
                facing: Direction::East,
            });
        }
        for (index, name) in self.east_outputs.iter().enumerate() {
            let pin = Position(width, index * self.pin_spacing + 1, 1);
            place_pin(&mut world, pin);
            ports.push(PhysicalPort {
                name: name.clone(),
                direction: PhysicalPortDirection::Output,
                side: PhysicalPortSide::East,
                pin,
                facing: Direction::West,
            });
        }
        ports.sort_by(|a, b| a.name.cmp(&b.name));

        let bbox = bounding_box(&world).ok_or_else(|| eyre::eyre!("macro candidate has no blocks"))?;
        let occupied_cells = world
            .iter_block()
            .into_iter()
            .map(|(position, _)| position)
            .collect::<HashSet<_>>();
        let blocked_cells = occupied_cells.clone();
        let cost = world_compact_cost(&world);

        Ok(MacroLayoutCandidate {
            unit_id: self.unit_id.clone(),
            world_fragment: world,
            bbox,
            ports,
            occupied_cells,
            blocked_cells,
            cost,
        })
    }
}

fn place_pin(world: &mut World3D, pin: Position) {
    world[pin.down().unwrap()] = Block {
        kind: BlockKind::Cobble {
            on_count: 0,
            on_base_count: 0,
        },
        direction: Direction::None,
    };
    world[pin] = Block {
        kind: BlockKind::Redstone {
            on_count: 0,
            state: 0,
            strength: 0,
        },
        direction: Direction::None,
    };
}
```

Modify `mod.rs`:

```rust
pub mod macro_candidate;
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```powershell
cargo test --release half_adder_macro_candidate_exposes_only_boundary_pins -- --nocapture
```

Expected: pass. If local candidate generation exceeds 60 seconds, reduce `step_sampling_policy` to `Random(1024)` and keep `max_candidates_per_unit` at 1 in downstream tests.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/macro_candidate.rs src/transform/place_and_route/mod.rs
git commit -m "Add boundary-pinned macro candidates"
```

---

## Task 4: Add Legal Global Router With Repeater Insertion

**Files:**
- Create: `src/transform/place_and_route/global_router.rs`
- Modify: `src/transform/place_and_route/mod.rs`
- Test: `src/transform/place_and_route/global_router.rs`

- [ ] **Step 1: Write failing test for repeater insertion every 15 blocks**

Create tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::place_and_route::macro_candidate::{
        PhysicalPort, PhysicalPortDirection, PhysicalPortSide,
    };
    use crate::transform::place_and_route::physical::validate_supported_redstone;
    use crate::world::block::{Block, BlockKind, Direction};
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

    fn empty_world() -> World3D {
        World3D::new(DimSize(64, 16, 2))
    }

    #[test]
    fn router_inserts_repeaters_for_long_channel() -> eyre::Result<()> {
        let mut world = empty_world();
        let src = PhysicalPort {
            name: "src".to_string(),
            direction: PhysicalPortDirection::Output,
            side: PhysicalPortSide::East,
            pin: Position(1, 1, 1),
            facing: Direction::East,
        };
        let dst = PhysicalPort {
            name: "dst".to_string(),
            direction: PhysicalPortDirection::Input,
            side: PhysicalPortSide::West,
            pin: Position(40, 1, 1),
            facing: Direction::West,
        };
        place_pin(&mut world, src.pin);
        place_pin(&mut world, dst.pin);

        let routed = GlobalRouter::new(GlobalRouterConfig {
            channel_spacing: 3,
            route_z: 1,
            max_redstone_run: 15,
        })
        .route_one(world, &src, &dst, 4)?;

        let repeater_count = routed
            .iter_block()
            .into_iter()
            .filter(|(_, block)| matches!(block.kind, BlockKind::Repeater { .. }))
            .count();

        assert!(repeater_count >= 2);
        validate_supported_redstone(&routed).unwrap();
        Ok(())
    }

    fn place_pin(world: &mut World3D, pin: Position) {
        world[pin.down().unwrap()] = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        };
        world[pin] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
    }
}
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
cargo test --release router_inserts_repeaters_for_long_channel -- --nocapture
```

Expected: compile failure for `GlobalRouter`.

- [ ] **Step 3: Implement deterministic channel router**

Add:

```rust
use crate::transform::place_and_route::macro_candidate::PhysicalPort;
use crate::transform::place_and_route::physical::step_towards;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Debug, Clone, Copy)]
pub struct GlobalRouterConfig {
    pub channel_spacing: usize,
    pub route_z: usize,
    pub max_redstone_run: usize,
}

#[derive(Debug, Clone)]
pub struct GlobalRouter {
    config: GlobalRouterConfig,
}

impl GlobalRouter {
    pub fn new(config: GlobalRouterConfig) -> Self {
        Self { config }
    }

    pub fn route_one(
        &self,
        world: World3D,
        src: &PhysicalPort,
        dst: &PhysicalPort,
        channel_y: usize,
    ) -> eyre::Result<World3D> {
        let max_x = src.pin.0.max(dst.pin.0) + 2;
        let max_y = channel_y + 2;
        let max_z = self.config.route_z + 1;
        let mut world = ensure_world_size(&world, DimSize(max_x, max_y, max_z));
        let a = Position(src.pin.0, channel_y, self.config.route_z);
        let b = Position(dst.pin.0, channel_y, self.config.route_z);

        self.draw_segment(&mut world, src.pin, a)?;
        self.draw_segment(&mut world, a, b)?;
        self.draw_segment(&mut world, b, dst.pin)?;
        world.initialize_redstone_states();
        Ok(world)
    }

    fn draw_segment(&self, world: &mut World3D, from: Position, to: Position) -> eyre::Result<()> {
        let mut cursor = from;
        let mut run_len = 0usize;

        while cursor != to {
            cursor = step_towards(cursor, to);
            if cursor == to {
                break;
            }

            place_support(world, cursor)?;
            if run_len == self.config.max_redstone_run {
                place_repeater(world, cursor, from, to)?;
                run_len = 0;
            } else {
                place_redstone(world, cursor)?;
                run_len += 1;
            }
        }

        Ok(())
    }
}

fn ensure_world_size(world: &World3D, size: DimSize) -> World3D {
    if world.size.0 >= size.0 && world.size.1 >= size.1 && world.size.2 >= size.2 {
        return world.clone();
    }

    let mut resized = World3D::new(DimSize(
        world.size.0.max(size.0),
        world.size.1.max(size.1),
        world.size.2.max(size.2),
    ));
    for (position, block) in world.iter_block() {
        resized[position] = block;
    }
    resized
}

fn place_support(world: &mut World3D, position: Position) -> eyre::Result<()> {
    let Some(bottom) = position.down() else {
        eyre::bail!("cannot route at z=0 without support below");
    };
    if world[bottom].kind.is_air() {
        world[bottom] = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        };
    }
    Ok(())
}

fn place_redstone(world: &mut World3D, position: Position) -> eyre::Result<()> {
    if world[position].kind.is_air() {
        world[position] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
    }
    Ok(())
}

fn place_repeater(world: &mut World3D, position: Position, from: Position, to: Position) -> eyre::Result<()> {
    if !world[position].kind.is_air() && !world[position].kind.is_redstone() {
        eyre::bail!("cannot place repeater at occupied position {:?}", position);
    }
    world[position] = Block {
        kind: BlockKind::Repeater {
            is_on: false,
            is_locked: false,
            delay: 1,
            lock_input1: None,
            lock_input2: None,
        },
        direction: route_direction(from, to),
    };
    Ok(())
}

fn route_direction(from: Position, to: Position) -> Direction {
    if from.0 < to.0 {
        Direction::East
    } else if from.0 > to.0 {
        Direction::West
    } else if from.1 < to.1 {
        Direction::North
    } else {
        Direction::South
    }
}
```

Modify `mod.rs`:

```rust
pub mod global_router;
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
cargo test --release router_inserts_repeaters_for_long_channel -- --nocapture
```

Expected: pass with at least two repeaters.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/global_router.rs src/transform/place_and_route/mod.rs
git commit -m "Add physical global channel router"
```

---

## Task 5: Compose Two Half-Adders Into a Full-Adder Macro Netlist

**Files:**
- Create: `src/transform/place_and_route/full_adder_pnr.rs`
- Modify: `src/transform/place_and_route/mod.rs`
- Test: `src/transform/place_and_route/full_adder_pnr.rs`

- [ ] **Step 1: Write failing test for full-adder macro graph**

Create:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_half_adder_full_adder_has_expected_macro_nets() {
        let circuit = two_half_adder_full_adder_circuit();

        assert_eq!(
            circuit.instances.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
            vec!["ha#0", "ha#1", "cout_or"]
        );
        assert_eq!(circuit.net("s0_to_ha1_a").unwrap().driver.instance, "ha#0");
        assert_eq!(circuit.net("s0_to_ha1_a").unwrap().driver.port, "s");
        assert_eq!(circuit.net("s0_to_ha1_a").unwrap().sinks[0].instance, "ha#1");
        assert_eq!(circuit.net("s0_to_ha1_a").unwrap().sinks[0].port, "a");
        assert_eq!(circuit.net("c0_to_cout_or_a").unwrap().driver.port, "c");
        assert_eq!(circuit.net("c1_to_cout_or_b").unwrap().driver.instance, "ha#1");
    }
}
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
cargo test --release two_half_adder_full_adder_has_expected_macro_nets -- --nocapture
```

Expected: compile failure for `full_adder_pnr` module.

- [ ] **Step 3: Implement macro circuit structs**

Add:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroPortRef {
    pub instance: String,
    pub port: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroNet {
    pub name: String,
    pub driver: MacroPortRef,
    pub sinks: Vec<MacroPortRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroInstanceSpec {
    pub name: String,
    pub unit_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MacroCircuit {
    pub instances: Vec<MacroInstanceSpec>,
    pub nets: Vec<MacroNet>,
}

impl MacroCircuit {
    pub fn net(&self, name: &str) -> Option<&MacroNet> {
        self.nets.iter().find(|net| net.name == name)
    }
}

pub fn two_half_adder_full_adder_circuit() -> MacroCircuit {
    MacroCircuit {
        instances: vec![
            MacroInstanceSpec {
                name: "ha#0".to_string(),
                unit_id: "half_adder".to_string(),
            },
            MacroInstanceSpec {
                name: "ha#1".to_string(),
                unit_id: "half_adder".to_string(),
            },
            MacroInstanceSpec {
                name: "cout_or".to_string(),
                unit_id: "or_gate".to_string(),
            },
        ],
        nets: vec![
            MacroNet {
                name: "s0_to_ha1_a".to_string(),
                driver: MacroPortRef {
                    instance: "ha#0".to_string(),
                    port: "s".to_string(),
                },
                sinks: vec![MacroPortRef {
                    instance: "ha#1".to_string(),
                    port: "a".to_string(),
                }],
            },
            MacroNet {
                name: "c0_to_cout_or_a".to_string(),
                driver: MacroPortRef {
                    instance: "ha#0".to_string(),
                    port: "c".to_string(),
                },
                sinks: vec![MacroPortRef {
                    instance: "cout_or".to_string(),
                    port: "a".to_string(),
                }],
            },
            MacroNet {
                name: "c1_to_cout_or_b".to_string(),
                driver: MacroPortRef {
                    instance: "ha#1".to_string(),
                    port: "c".to_string(),
                },
                sinks: vec![MacroPortRef {
                    instance: "cout_or".to_string(),
                    port: "b".to_string(),
                }],
            },
        ],
    }
}
```

Modify `mod.rs`:

```rust
pub mod full_adder_pnr;
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```powershell
cargo test --release two_half_adder_full_adder_has_expected_macro_nets -- --nocapture
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/full_adder_pnr.rs src/transform/place_and_route/mod.rs
git commit -m "Define two-half-adder full-adder macro circuit"
```

---

## Task 6: Place Macro Candidates With Routability Checks

**Files:**
- Modify: `src/transform/place_and_route/full_adder_pnr.rs`
- Test: `src/transform/place_and_route/full_adder_pnr.rs`

- [ ] **Step 1: Write failing test for deterministic macro placement**

Add:

```rust
#[test]
fn places_full_adder_macros_with_non_overlapping_bounding_boxes() -> eyre::Result<()> {
    let ha = fake_macro("half_adder", 8, 8, ["a", "b"], ["c", "s"]);
    let or_gate = fake_macro("or_gate", 6, 6, ["a", "b"], ["c"]);
    let circuit = two_half_adder_full_adder_circuit();

    let placement = place_full_adder_macros(&circuit, &ha, &or_gate, 6)?;

    assert_eq!(placement.instances.len(), 3);
    assert_eq!(placement.instance("ha#0").unwrap().origin, Position(0, 0, 0));
    assert_eq!(placement.instance("ha#1").unwrap().origin, Position(14, 0, 0));
    assert_eq!(placement.instance("cout_or").unwrap().origin, Position(28, 0, 0));
    assert!(placement.instances_do_not_overlap());
    Ok(())
}
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
cargo test --release places_full_adder_macros_with_non_overlapping_bounding_boxes -- --nocapture
```

Expected: compile failure for placement types and helpers.

- [ ] **Step 3: Implement placement structs and deterministic row placement**

Add:

```rust
use crate::transform::place_and_route::macro_candidate::MacroLayoutCandidate;
use crate::world::position::Position;
use crate::world::World3D;

#[derive(Debug, Clone)]
pub struct PlacedMacro {
    pub name: String,
    pub unit_id: String,
    pub origin: Position,
    pub candidate: MacroLayoutCandidate,
}

#[derive(Debug, Clone)]
pub struct FullAdderPlacement {
    pub instances: Vec<PlacedMacro>,
    pub world: World3D,
}

impl FullAdderPlacement {
    pub fn instance(&self, name: &str) -> Option<&PlacedMacro> {
        self.instances.iter().find(|instance| instance.name == name)
    }

    pub fn instances_do_not_overlap(&self) -> bool {
        let mut occupied = std::collections::HashSet::new();
        for instance in &self.instances {
            for position in &instance.candidate.occupied_cells {
                let translated = Position(
                    position.0 + instance.origin.0,
                    position.1 + instance.origin.1,
                    position.2 + instance.origin.2,
                );
                if !occupied.insert(translated) {
                    return false;
                }
            }
        }
        true
    }
}

pub fn place_full_adder_macros(
    circuit: &MacroCircuit,
    half_adder: &MacroLayoutCandidate,
    or_gate: &MacroLayoutCandidate,
    spacing: usize,
) -> eyre::Result<FullAdderPlacement> {
    let mut cursor_x = 0;
    let mut instances = Vec::new();
    for spec in &circuit.instances {
        let candidate = match spec.unit_id.as_str() {
            "half_adder" => half_adder.clone(),
            "or_gate" => or_gate.clone(),
            other => eyre::bail!("unknown unit id {other}"),
        };
        let origin = Position(cursor_x, 0, 0);
        cursor_x += candidate.bbox.width() + spacing;
        instances.push(PlacedMacro {
            name: spec.name.clone(),
            unit_id: spec.unit_id.clone(),
            origin,
            candidate,
        });
    }
    let world = compose_world(&instances)?;
    Ok(FullAdderPlacement { instances, world })
}
```

Add `compose_world()` in the same file:

```rust
fn compose_world(instances: &[PlacedMacro]) -> eyre::Result<World3D> {
    let size_x = instances
        .iter()
        .map(|i| i.origin.0 + i.candidate.world_fragment.size.0)
        .max()
        .unwrap_or(1);
    let size_y = instances
        .iter()
        .map(|i| i.origin.1 + i.candidate.world_fragment.size.1)
        .max()
        .unwrap_or(1);
    let size_z = instances
        .iter()
        .map(|i| i.origin.2 + i.candidate.world_fragment.size.2)
        .max()
        .unwrap_or(1);
    let mut world = World3D::new(crate::world::position::DimSize(size_x, size_y, size_z));

    for instance in instances {
        for (position, block) in instance.candidate.world_fragment.iter_block() {
            let translated = Position(
                position.0 + instance.origin.0,
                position.1 + instance.origin.1,
                position.2 + instance.origin.2,
            );
            if !world[translated].kind.is_air() {
                eyre::bail!("macro overlap at {:?}", translated);
            }
            world[translated] = block;
        }
    }

    Ok(world)
}
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```powershell
cargo test --release places_full_adder_macros_with_non_overlapping_bounding_boxes -- --nocapture
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/full_adder_pnr.rs
git commit -m "Place full-adder macro row"
```

---

## Task 7: Route Full-Adder Macro Nets Physically

**Files:**
- Modify: `src/transform/place_and_route/full_adder_pnr.rs`
- Test: `src/transform/place_and_route/full_adder_pnr.rs`

- [ ] **Step 1: Write failing test for routed macro nets**

Add:

```rust
#[test]
fn routes_two_half_adder_full_adder_macro_nets() -> eyre::Result<()> {
    let ha = fake_macro("half_adder", 8, 8, ["a", "b"], ["c", "s"]);
    let or_gate = fake_macro("or_gate", 6, 6, ["a", "b"], ["c"]);
    let circuit = two_half_adder_full_adder_circuit();
    let placement = place_full_adder_macros(&circuit, &ha, &or_gate, 6)?;

    let routed = route_full_adder_macro_nets(
        &circuit,
        placement,
        GlobalRouterConfig {
            channel_spacing: 3,
            route_z: 1,
            max_redstone_run: 15,
        },
    )?;

    assert!(routed.world.iter_block().len() > ha.world_fragment.iter_block().len() * 2);
    validate_supported_redstone(&routed.world).unwrap();
    assert!(routed.contains_routed_net("s0_to_ha1_a"));
    assert!(routed.contains_routed_net("c0_to_cout_or_a"));
    assert!(routed.contains_routed_net("c1_to_cout_or_b"));
    Ok(())
}
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
cargo test --release routes_two_half_adder_full_adder_macro_nets -- --nocapture
```

Expected: compile failure for `route_full_adder_macro_nets`.

- [ ] **Step 3: Implement route resolution from placed macro ports**

Add:

```rust
use crate::transform::place_and_route::global_router::{GlobalRouter, GlobalRouterConfig};
use crate::transform::place_and_route::physical::validate_supported_redstone;

#[derive(Debug, Clone)]
pub struct RoutedFullAdderPlacement {
    pub placement: FullAdderPlacement,
    pub world: World3D,
    routed_nets: std::collections::HashSet<String>,
}

impl RoutedFullAdderPlacement {
    pub fn contains_routed_net(&self, name: &str) -> bool {
        self.routed_nets.contains(name)
    }
}

pub fn route_full_adder_macro_nets(
    circuit: &MacroCircuit,
    placement: FullAdderPlacement,
    config: GlobalRouterConfig,
) -> eyre::Result<RoutedFullAdderPlacement> {
    let mut world = placement.world.clone();
    let router = GlobalRouter::new(config);
    let mut routed_nets = std::collections::HashSet::new();
    let base_channel_y = world.size.1 + config.channel_spacing;

    for (index, net) in circuit.nets.iter().enumerate() {
        let src = resolve_placed_port(&placement, &net.driver)?;
        for sink in &net.sinks {
            let dst = resolve_placed_port(&placement, sink)?;
            world = router.route_one(world, &src, &dst, base_channel_y + index * config.channel_spacing)?;
        }
        routed_nets.insert(net.name.clone());
    }

    validate_supported_redstone(&world)?;
    Ok(RoutedFullAdderPlacement {
        placement,
        world,
        routed_nets,
    })
}
```

Add resolver:

```rust
fn resolve_placed_port(
    placement: &FullAdderPlacement,
    port_ref: &MacroPortRef,
) -> eyre::Result<crate::transform::place_and_route::macro_candidate::PhysicalPort> {
    let instance = placement
        .instance(&port_ref.instance)
        .ok_or_else(|| eyre::eyre!("unknown instance {}", port_ref.instance))?;
    let mut port = instance
        .candidate
        .ports
        .iter()
        .find(|port| port.name == port_ref.port)
        .cloned()
        .ok_or_else(|| eyre::eyre!("unknown port {}.{}", port_ref.instance, port_ref.port))?;
    port.pin = Position(
        port.pin.0 + instance.origin.0,
        port.pin.1 + instance.origin.1,
        port.pin.2 + instance.origin.2,
    );
    Ok(port)
}
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```powershell
cargo test --release routes_two_half_adder_full_adder_macro_nets -- --nocapture
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/full_adder_pnr.rs
git commit -m "Route full-adder macro nets"
```

---

## Task 8: Generate Real Half-Adder And OR Macro Candidates

**Files:**
- Modify: `src/transform/place_and_route/full_adder_pnr.rs`
- Test: `src/transform/place_and_route/full_adder_pnr.rs`

- [ ] **Step 1: Write ignored expensive integration test**

Add:

```rust
#[test]
#[ignore = "expensive: runs local placement for half-adder and OR macros"]
fn exports_physically_routed_two_half_adder_full_adder_nbt() -> eyre::Result<()> {
    let output = std::path::Path::new("test/two-half-adder-full-adder-physical.nbt");

    let routed = build_physically_routed_two_half_adder_full_adder()?;
    crate::nbt::ToNBT::to_nbt(&routed.world).save(output);

    assert!(output.exists());
    assert!(std::fs::metadata(output)?.len() > 0);
    validate_supported_redstone(&routed.world).unwrap();
    assert!(routed.contains_routed_net("s0_to_ha1_a"));
    assert!(routed.contains_routed_net("c0_to_cout_or_a"));
    assert!(routed.contains_routed_net("c1_to_cout_or_b"));
    Ok(())
}
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
cargo test --release exports_physically_routed_two_half_adder_full_adder_nbt -- --ignored --nocapture
```

Expected: compile failure for `build_physically_routed_two_half_adder_full_adder`.

- [ ] **Step 3: Implement real macro build pipeline**

Add:

```rust
use crate::graph::logic::{predefined_logics, LogicGraph};
use crate::transform::place_and_route::local_placer::{
    InputPlacementStrategy, LocalPlacer, LocalPlacerConfig, NotRouteStrategy,
    PlacementSamplingPolicy, TorchPlacementStrategy,
};
use crate::transform::place_and_route::macro_candidate::MacroCandidateBuilder;
use crate::transform::place_and_route::sampling::SamplingPolicy;
use crate::world::position::DimSize;

pub fn build_physically_routed_two_half_adder_full_adder() -> eyre::Result<RoutedFullAdderPlacement> {
    let half_adder = build_best_half_adder_macro()?;
    let or_gate = build_best_or_macro()?;
    let circuit = two_half_adder_full_adder_circuit();
    let placement = place_full_adder_macros(&circuit, &half_adder, &or_gate, 8)?;
    route_full_adder_macro_nets(
        &circuit,
        placement,
        GlobalRouterConfig {
            channel_spacing: 3,
            route_z: 1,
            max_redstone_run: 15,
        },
    )
}

fn build_best_half_adder_macro() -> eyre::Result<MacroLayoutCandidate> {
    let graph = predefined_logics::buffered_binary_half_adder_graph()?;
    let placer = LocalPlacer::new(graph, macro_local_placer_config())?;
    let local = placer
        .generate_layout_candidates("half_adder", DimSize(12, 12, 5), None)
        .into_iter()
        .min_by_key(|candidate| candidate.cost)
        .ok_or_else(|| eyre::eyre!("half-adder local placer produced no candidates"))?;

    MacroCandidateBuilder::new("half_adder")
        .west_inputs(["a", "b"])
        .east_outputs(["c", "s"])
        .build_from_local_candidate(&local)
}

fn build_best_or_macro() -> eyre::Result<MacroLayoutCandidate> {
    let graph = LogicGraph::from_stmt("a|b", "c")?.prepare_place()?;
    let placer = LocalPlacer::new(graph, macro_local_placer_config())?;
    let local = placer
        .generate_layout_candidates("or_gate", DimSize(8, 8, 4), None)
        .into_iter()
        .min_by_key(|candidate| candidate.cost)
        .ok_or_else(|| eyre::eyre!("OR local placer produced no candidates"))?;

    MacroCandidateBuilder::new("or_gate")
        .west_inputs(["a", "b"])
        .east_outputs(["c"])
        .build_from_local_candidate(&local)
}

fn macro_local_placer_config() -> LocalPlacerConfig {
    LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        step_sampling_policy: SamplingPolicy::Random(4096),
        placement_sampling_policy: PlacementSamplingPolicy::Cost {
            count: 256,
            random_count: 32,
            start_step: 6,
        },
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 3,
        not_route_step_sampling_policy: SamplingPolicy::Random(256),
        max_route_step: 3,
        route_step_sampling_policy: SamplingPolicy::Random(256),
    }
}
```

- [ ] **Step 4: Run ignored integration test**

Run:

```powershell
cargo test --release exports_physically_routed_two_half_adder_full_adder_nbt -- --ignored --nocapture
```

Expected:

```text
test ... exports_physically_routed_two_half_adder_full_adder_nbt ... ok
```

Generated file:

```text
test/two-half-adder-full-adder-physical.nbt
```

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/full_adder_pnr.rs test/two-half-adder-full-adder-physical.nbt
git commit -m "Export physically routed two-half-adder full adder"
```

---

## Task 9: Remove Or Quarantine The Experimental Channel Router

**Files:**
- Modify: `src/transform/place_and_route/macro_placer.rs`
- Modify: `docs/global_place_route_m5_m6.md`
- Test: `cargo test --release macro_placer full_adder_pnr`

- [ ] **Step 1: Mark old APIs as experimental-only**

In `macro_placer.rs`, add this doc comment to `route_macro_nets_with_channels` if the function remains:

```rust
/// Experimental router retained for comparison tests only.
///
/// Production physical P&R must use `global_router::GlobalRouter` through
/// `full_adder_pnr::route_full_adder_macro_nets`, because this helper does
/// not own macro pin contracts and must not be used for final NBT export.
```

- [ ] **Step 2: Rename old NBT output tests**

Rename old tests so their names make the limitation explicit:

```rust
fn export_experimental_unvalidated_global_floorplan_to_nbt()
fn export_experimental_boundary_channel_route_to_nbt()
```

Keep them ignored if they generate artifacts.

- [ ] **Step 3: Update docs**

In `docs/global_place_route_m5_m6.md`, add:

```markdown
The experimental `macro_placer` channel route is superseded by
`full_adder_pnr` + `global_router`. Do not use `macro_placer` NBT fixtures as
physical correctness evidence.
```

- [ ] **Step 4: Run tests**

Run:

```powershell
cargo test --release macro_placer
cargo test --release full_adder_pnr
cargo test --release --no-run
```

Expected: all non-ignored tests pass and all test binaries compile.

- [ ] **Step 5: Commit**

```powershell
git add src/transform/place_and_route/macro_placer.rs docs/global_place_route_m5_m6.md
git commit -m "Quarantine experimental macro routing"
```

---

## Task 10: Final Verification Checklist

**Files:**
- No source changes unless verification exposes a defect.

- [ ] **Step 1: Run focused unit tests**

```powershell
cargo test --release physical -- --nocapture
cargo test --release io_panel -- --nocapture
cargo test --release macro_candidate -- --nocapture
cargo test --release global_router -- --nocapture
cargo test --release full_adder_pnr -- --nocapture
```

Expected: all non-ignored tests pass.

- [ ] **Step 2: Run expensive NBT export**

```powershell
cargo test --release exports_physically_routed_two_half_adder_full_adder_nbt -- --ignored --nocapture
```

Expected: test passes and writes:

```text
test/two-half-adder-full-adder-physical.nbt
```

- [ ] **Step 3: Run full compile check**

```powershell
cargo test --release --no-run
```

Expected: all test binaries compile.

- [ ] **Step 4: Confirm NBT exists and is non-empty**

```powershell
Get-Item test\two-half-adder-full-adder-physical.nbt | Select-Object FullName,Length
```

Expected: `Length` is greater than `0`.

- [ ] **Step 5: Commit verification artifact if not already committed**

```powershell
git add test/two-half-adder-full-adder-physical.nbt
git commit -m "Add verified physical full-adder NBT"
```

---

## Self-Review

**Spec coverage:** The plan covers deterministic switch panels, macro boundary pins, half-adder macro local placement, physical global routing, repeater insertion for 15-block redstone limits, routed full-adder NBT export, and cleanup of the earlier unvalidated route.

**Placeholder scan:** The plan avoids placeholder tasks. Each implementation task includes file paths, test commands, expected results, and concrete code skeletons.

**Type consistency:** `PhysicalPort`, `MacroLayoutCandidate`, `GlobalRouter`, `MacroCircuit`, and `RoutedFullAdderPlacement` names are used consistently across tasks. The final export function name is `build_physically_routed_two_half_adder_full_adder()`.
