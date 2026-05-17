use std::collections::HashMap;

use crate::sequential::synthesis::synthesize_rs_latch_macros;
use crate::sequential::{SequentialPrimitive, SequentialType};
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Debug, Clone)]
pub struct SequentialMacro {
    pub primitive_type: SequentialType,
    pub world: World3D,
    pub input_ports: HashMap<String, Position>,
    pub output_ports: HashMap<String, Position>,
    pub primary_output_port: String,
    pub cost: usize,
}

impl SequentialMacro {
    pub fn candidates(primitive: &SequentialPrimitive) -> Vec<Self> {
        match primitive.sequential_type {
            SequentialType::RsLatch => primitive
                .rs_latch_core()
                .as_ref()
                .map(synthesize_rs_latch_macros)
                .unwrap_or_default(),
            SequentialType::DLatch | SequentialType::DFlipFlop => Vec::new(),
        }
    }
}

pub(super) fn canonical_rs_latch_macro() -> SequentialMacro {
    let mut world = World3D::new(DimSize(8, 6, 3));
    let r_input = Position(1, 2, 1);
    let s_input = Position(5, 2, 1);
    let q_support = Position(2, 2, 1);
    let q = Position(2, 1, 1);
    let nq_support = Position(4, 2, 1);
    let nq = Position(4, 3, 1);
    let q_feedback = [
        Position(2, 0, 1),
        Position(3, 0, 1),
        Position(4, 0, 1),
        Position(4, 1, 1),
    ];
    let nq_feedback = [
        Position(4, 4, 1),
        Position(3, 4, 1),
        Position(2, 4, 1),
        Position(2, 3, 1),
    ];

    for position in [r_input, s_input] {
        world[position.down().unwrap()] = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        };
        world[position] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
    }
    world[q_support] = Block {
        kind: BlockKind::Cobble {
            on_count: 0,
            on_base_count: 0,
        },
        direction: Direction::None,
    };
    world[nq_support] = Block {
        kind: BlockKind::Cobble {
            on_count: 0,
            on_base_count: 0,
        },
        direction: Direction::None,
    };
    world[q] = Block {
        kind: BlockKind::Torch { is_on: false },
        direction: Direction::North,
    };
    world[nq] = Block {
        kind: BlockKind::Torch { is_on: false },
        direction: Direction::South,
    };
    for position in q_feedback.into_iter().chain(nq_feedback) {
        world[position.down().unwrap()] = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Direction::None,
        };
        world[position] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
    }

    SequentialMacro {
        primitive_type: SequentialType::RsLatch,
        world,
        input_ports: [("s".to_owned(), s_input), ("r".to_owned(), r_input)]
            .into_iter()
            .collect(),
        output_ports: [("q".to_owned(), q), ("nq".to_owned(), nq)]
            .into_iter()
            .collect(),
        primary_output_port: "q".to_owned(),
        cost: 20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::simulator::Simulator;
    use crate::world::World;

    #[test]
    fn rs_latch_macro_exposes_expected_ports() {
        let primitive = SequentialPrimitive::rs_latch();
        let candidates = SequentialMacro::candidates(&primitive);

        assert!(candidates.len() > 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.primitive_type, SequentialType::RsLatch);
        assert_eq!(candidate.primary_output_port, "q");
        assert!(candidate.input_ports.contains_key("s"));
        assert!(candidate.input_ports.contains_key("r"));
        assert!(candidate.output_ports.contains_key("q"));
        assert!(candidate.output_ports.contains_key("nq"));
        assert!(!candidate.world.iter_block().is_empty());
    }

    #[test]
    fn rs_latch_macro_ports_are_unique_non_air_blocks() {
        let primitive = SequentialPrimitive::rs_latch();
        let candidate = SequentialMacro::candidates(&primitive).remove(0);
        let all_ports = candidate
            .input_ports
            .values()
            .chain(candidate.output_ports.values())
            .copied()
            .collect::<Vec<_>>();

        assert!(candidate
            .output_ports
            .contains_key(&candidate.primary_output_port));
        assert_eq!(
            all_ports
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            all_ports.len()
        );
        for position in all_ports {
            assert!(candidate.world.size.bound_on(position));
            assert!(!candidate.world[position].kind.is_air());
        }
    }

    fn torch_is_on(world: &World3D, position: Position) -> bool {
        let BlockKind::Torch { is_on } = world[position].kind else {
            panic!("expected torch at {position:?}");
        };

        is_on
    }

    fn rs_latch_macro_preview_world(candidate: &SequentialMacro) -> World3D {
        let mut world = candidate.world.clone();
        let r_driver = candidate.input_ports["r"].walk(Direction::West).unwrap();
        let s_driver = candidate.input_ports["s"].walk(Direction::East).unwrap();
        world[r_driver] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::East,
        };
        world[s_driver] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world.initialize_redstone_states();
        world
    }

    #[test]
    fn rs_latch_macro_satisfies_set_reset_hold_behavior() -> eyre::Result<()> {
        let primitive = SequentialPrimitive::rs_latch();
        let candidate = SequentialMacro::candidates(&primitive).remove(0);
        let mut candidate_world = rs_latch_macro_preview_world(&candidate);
        let s = candidate.input_ports["s"].walk(Direction::East).unwrap();
        let r = candidate.input_ports["r"].walk(Direction::West).unwrap();
        let q = candidate.output_ports["q"];
        let nq = candidate.output_ports["nq"];
        candidate_world[r].kind = BlockKind::Switch { is_on: true };
        let world = World::from(&candidate_world);
        let mut sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        assert!(!torch_is_on(sim.world(), q));
        assert!(torch_is_on(sim.world(), nq));

        sim.change_state_with_limits(vec![(r, false)], 64, 20_000)?;
        assert!(!torch_is_on(sim.world(), q));
        assert!(torch_is_on(sim.world(), nq));

        sim.change_state_with_limits(vec![(s, true)], 64, 20_000)?;
        sim.change_state_with_limits(vec![(s, false)], 64, 20_000)?;
        assert!(torch_is_on(sim.world(), q));
        assert!(!torch_is_on(sim.world(), nq));

        sim.change_state_with_limits(vec![(r, true)], 64, 20_000)?;
        sim.change_state_with_limits(vec![(r, false)], 64, 20_000)?;
        assert!(!torch_is_on(sim.world(), q));
        assert!(torch_is_on(sim.world(), nq));

        Ok(())
    }
}
