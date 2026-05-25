use std::fmt;

use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::transform::place_and_route::global_pnr::router::RoutedNet;
use crate::world::block::Block;
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Debug, PartialEq, Eq)]
pub enum AssemblyError {
    MissingCandidate { candidate_index: usize },
    Collision { position: Position },
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssemblyError::MissingCandidate { candidate_index } => {
                write!(f, "missing layout candidate at index {candidate_index}")
            }
            AssemblyError::Collision { position } => {
                write!(f, "assembly block collision at {position:?}")
            }
        }
    }
}

impl std::error::Error for AssemblyError {}

pub fn assemble_world(
    candidates: &[LayoutCandidate],
    placed_modules: &[PlacedModule],
    routed_nets: &[RoutedNet],
) -> Result<World3D, AssemblyError> {
    let mut blocks = Vec::<(Position, Block)>::new();

    for placed in placed_modules {
        let candidate =
            candidates
                .get(placed.candidate_index)
                .ok_or(AssemblyError::MissingCandidate {
                    candidate_index: placed.candidate_index,
                })?;
        for (position, block) in candidate.world.iter_block() {
            blocks.push((
                translate_candidate_position(position, candidate, placed),
                block,
            ));
        }
    }

    for route in routed_nets {
        blocks.extend(route.blocks.iter().copied());
    }

    let size = world_size_for_blocks(&blocks);
    let mut world = World3D::new(size);
    for (position, block) in blocks {
        if !world[position].kind.is_air() {
            return Err(AssemblyError::Collision { position });
        }
        world[position] = block;
    }

    Ok(world)
}

fn translate_candidate_position(
    position: Position,
    candidate: &LayoutCandidate,
    placed: &PlacedModule,
) -> Position {
    Position(
        placed.origin.0 + position.0 - candidate.bbox.min.0,
        placed.origin.1 + position.1 - candidate.bbox.min.1,
        placed.origin.2 + position.2 - candidate.bbox.min.2,
    )
}

fn world_size_for_blocks(blocks: &[(Position, Block)]) -> DimSize {
    let mut max = Position(0, 0, 0);
    for (position, _) in blocks {
        max.0 = max.0.max(position.0);
        max.1 = max.1.max(position.1);
        max.2 = max.2.max(position.2);
    }
    DimSize(max.0 + 1, max.1 + 1, max.2 + 1)
}
