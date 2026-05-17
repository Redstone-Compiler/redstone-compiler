use std::collections::HashMap;

use crate::sequential::core::RsLatchCore;
use crate::sequential::layout::{canonical_rs_latch_macro, SequentialMacro};
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transform {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    MirrorX,
    MirrorY,
}

#[derive(Debug, Clone)]
struct CandidateWithDrivers {
    candidate: SequentialMacro,
    preview_world: World3D,
    driver_ports: HashMap<String, Position>,
}

pub fn synthesize_rs_latch_macros(_core: &RsLatchCore) -> Vec<SequentialMacro> {
    let base = canonical_rs_latch_candidate();
    [
        Transform::Identity,
        Transform::Rotate90,
        Transform::Rotate180,
        Transform::Rotate270,
        Transform::MirrorX,
        Transform::MirrorY,
    ]
    .into_iter()
    .filter_map(|transform| Some(transform_candidate(&base, transform)))
    .filter(rs_latch_candidate_satisfies_behavior)
    .map(|candidate| candidate.candidate)
    .collect()
}

fn canonical_rs_latch_candidate() -> CandidateWithDrivers {
    let candidate = canonical_rs_latch_macro();
    let mut preview_world = candidate.world.clone();
    let r_driver = Position(0, 2, 1);
    let s_driver = Position(6, 2, 1);
    preview_world[r_driver] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::East,
    };
    preview_world[s_driver] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::West,
    };
    preview_world.initialize_redstone_states();

    CandidateWithDrivers {
        candidate,
        preview_world,
        driver_ports: [("s".to_owned(), s_driver), ("r".to_owned(), r_driver)]
            .into_iter()
            .collect(),
    }
}

fn transform_candidate(base: &CandidateWithDrivers, transform: Transform) -> CandidateWithDrivers {
    let transformed = CandidateWithDrivers {
        candidate: SequentialMacro {
            primitive_type: base.candidate.primitive_type,
            world: transform_world(&base.candidate.world, transform),
            input_ports: transform_ports(
                &base.candidate.input_ports,
                base.candidate.world.size,
                transform,
            ),
            output_ports: transform_ports(
                &base.candidate.output_ports,
                base.candidate.world.size,
                transform,
            ),
            primary_output_port: base.candidate.primary_output_port.clone(),
            cost: base.candidate.cost + transform_cost(transform),
        },
        preview_world: transform_world(&base.preview_world, transform),
        driver_ports: transform_ports(&base.driver_ports, base.candidate.world.size, transform),
    };
    pad_candidate(transformed, 1)
}

fn transform_world(world: &World3D, transform: Transform) -> World3D {
    let mut transformed = World3D::new(transform_size(world.size, transform));
    for (position, mut block) in world.iter_block() {
        block.direction = transform_direction(block.direction, transform);
        transformed[transform_position(position, world.size, transform)] = block;
    }
    transformed.initialize_redstone_states();
    transformed
}

fn transform_ports(
    ports: &HashMap<String, Position>,
    size: DimSize,
    transform: Transform,
) -> HashMap<String, Position> {
    ports
        .iter()
        .map(|(name, position)| (name.clone(), transform_position(*position, size, transform)))
        .collect()
}

fn pad_candidate(candidate: CandidateWithDrivers, padding: usize) -> CandidateWithDrivers {
    CandidateWithDrivers {
        candidate: SequentialMacro {
            primitive_type: candidate.candidate.primitive_type,
            world: pad_world(&candidate.candidate.world, padding),
            input_ports: shift_ports(&candidate.candidate.input_ports, padding),
            output_ports: shift_ports(&candidate.candidate.output_ports, padding),
            primary_output_port: candidate.candidate.primary_output_port,
            cost: candidate.candidate.cost,
        },
        preview_world: pad_world(&candidate.preview_world, padding),
        driver_ports: shift_ports(&candidate.driver_ports, padding),
    }
}

fn pad_world(world: &World3D, padding: usize) -> World3D {
    let mut padded = World3D::new(DimSize(
        world.size.0 + padding * 2,
        world.size.1 + padding * 2,
        world.size.2,
    ));
    for (position, block) in world.iter_block() {
        padded[shift_position(position, padding)] = block;
    }
    padded.initialize_redstone_states();
    padded
}

fn shift_ports(ports: &HashMap<String, Position>, padding: usize) -> HashMap<String, Position> {
    ports
        .iter()
        .map(|(name, position)| (name.clone(), shift_position(*position, padding)))
        .collect()
}

fn shift_position(position: Position, padding: usize) -> Position {
    Position(position.0 + padding, position.1 + padding, position.2)
}

fn transform_size(size: DimSize, transform: Transform) -> DimSize {
    match transform {
        Transform::Rotate90 | Transform::Rotate270 => DimSize(size.1, size.0, size.2),
        Transform::Identity | Transform::Rotate180 | Transform::MirrorX | Transform::MirrorY => {
            size
        }
    }
}

fn transform_position(position: Position, size: DimSize, transform: Transform) -> Position {
    let max_x = size.0 - 1;
    let max_y = size.1 - 1;
    match transform {
        Transform::Identity => position,
        Transform::Rotate90 => Position(max_y - position.1, position.0, position.2),
        Transform::Rotate180 => Position(max_x - position.0, max_y - position.1, position.2),
        Transform::Rotate270 => Position(position.1, max_x - position.0, position.2),
        Transform::MirrorX => Position(max_x - position.0, position.1, position.2),
        Transform::MirrorY => Position(position.0, max_y - position.1, position.2),
    }
}

fn transform_direction(direction: Direction, transform: Transform) -> Direction {
    match transform {
        Transform::Identity => direction,
        Transform::Rotate90 => match direction {
            Direction::East => Direction::North,
            Direction::North => Direction::West,
            Direction::West => Direction::South,
            Direction::South => Direction::East,
            other => other,
        },
        Transform::Rotate180 => match direction {
            Direction::East => Direction::West,
            Direction::West => Direction::East,
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            other => other,
        },
        Transform::Rotate270 => match direction {
            Direction::East => Direction::South,
            Direction::South => Direction::West,
            Direction::West => Direction::North,
            Direction::North => Direction::East,
            other => other,
        },
        Transform::MirrorX => match direction {
            Direction::East => Direction::West,
            Direction::West => Direction::East,
            other => other,
        },
        Transform::MirrorY => match direction {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            other => other,
        },
    }
}

fn transform_cost(transform: Transform) -> usize {
    match transform {
        Transform::Identity => 0,
        Transform::Rotate90 | Transform::Rotate180 | Transform::Rotate270 => 1,
        Transform::MirrorX | Transform::MirrorY => 2,
    }
}

fn rs_latch_candidate_satisfies_behavior(candidate: &CandidateWithDrivers) -> bool {
    std::panic::catch_unwind(|| rs_latch_candidate_satisfies_behavior_inner(candidate))
        .unwrap_or(false)
}

fn rs_latch_candidate_satisfies_behavior_inner(candidate: &CandidateWithDrivers) -> bool {
    let mut world = candidate.preview_world.clone();
    let s = candidate.driver_ports["s"];
    let r = candidate.driver_ports["r"];
    let q = candidate.candidate.output_ports["q"];
    let nq = candidate.candidate.output_ports["nq"];
    world[r].kind = BlockKind::Switch { is_on: true };

    let world = World::from(&world);
    let Ok(mut sim) = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
        .map_err(|error| eyre::eyre!(error.message().to_owned()))
    else {
        return false;
    };

    if torch_is_on(sim.world(), q) != Some(false) || torch_is_on(sim.world(), nq) != Some(true) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(r, false)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if torch_is_on(sim.world(), q) != Some(false) || torch_is_on(sim.world(), nq) != Some(true) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(s, true)], 64, 20_000)
        .and_then(|_| sim.change_state_with_limits(vec![(s, false)], 64, 20_000))
        .is_err()
    {
        return false;
    }
    if torch_is_on(sim.world(), q) != Some(true) || torch_is_on(sim.world(), nq) != Some(false) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(r, true)], 64, 20_000)
        .and_then(|_| sim.change_state_with_limits(vec![(r, false)], 64, 20_000))
        .is_err()
    {
        return false;
    }

    torch_is_on(sim.world(), q) == Some(false) && torch_is_on(sim.world(), nq) == Some(true)
}

fn torch_is_on(world: &World3D, position: Position) -> Option<bool> {
    let BlockKind::Torch { is_on } = world[position].kind else {
        return None;
    };
    Some(is_on)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequential::SequentialPrimitive;

    #[test]
    fn synthesize_rs_latch_macros_returns_verified_oriented_candidates() {
        let primitive = SequentialPrimitive::rs_latch();
        let core = primitive.rs_latch_core().unwrap();
        let candidates = synthesize_rs_latch_macros(&core);

        assert!(candidates.len() > 1);
        assert!(candidates.iter().all(|candidate| candidate
            .output_ports
            .contains_key(&candidate.primary_output_port)));
    }
}
