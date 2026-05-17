use std::collections::HashMap;

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
            SequentialType::RsLatch => vec![rs_latch_macro()],
            SequentialType::DLatch | SequentialType::DFlipFlop => Vec::new(),
        }
    }
}

fn rs_latch_macro() -> SequentialMacro {
    let mut world = World3D::new(DimSize(6, 5, 3));
    let s = Position(1, 1, 1);
    let r = Position(1, 3, 1);
    let q_support = Position(3, 2, 1);
    let q = Position(4, 2, 1);
    let nq_support = Position(3, 3, 1);
    let nq = Position(4, 3, 1);

    world[s] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::East,
    };
    world[r] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::East,
    };
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
        direction: Direction::West,
    };
    world[nq] = Block {
        kind: BlockKind::Torch { is_on: false },
        direction: Direction::West,
    };

    SequentialMacro {
        primitive_type: SequentialType::RsLatch,
        world,
        input_ports: [("s".to_owned(), s), ("r".to_owned(), r)]
            .into_iter()
            .collect(),
        output_ports: [("q".to_owned(), q), ("nq".to_owned(), nq)]
            .into_iter()
            .collect(),
        primary_output_port: "q".to_owned(),
        cost: 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rs_latch_macro_exposes_expected_ports() {
        let primitive = SequentialPrimitive::rs_latch();
        let candidates = SequentialMacro::candidates(&primitive);

        assert_eq!(candidates.len(), 1);
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
}
