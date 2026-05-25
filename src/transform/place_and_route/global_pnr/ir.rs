use std::collections::HashSet;

use eyre::ContextCompat;

use crate::transform::place_and_route::estimate::{bounding_box, BoundingBox};
use crate::world::position::Position;
use crate::world::World3D;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayoutCandidateCost {
    pub block_count: usize,
    pub bbox_volume: usize,
}

#[derive(Clone, Debug)]
pub struct LayoutCandidate {
    pub module_name: String,
    pub world: World3D,
    pub bbox: BoundingBox,
    pub occupied_cells: HashSet<Position>,
    pub blocked_cells: HashSet<Position>,
    pub cost: LayoutCandidateCost,
}

impl LayoutCandidate {
    pub fn from_world(module_name: String, world: World3D) -> eyre::Result<Self> {
        let bbox = bounding_box(&world).context("layout candidate world has no blocks")?;
        let occupied_cells = world
            .iter_block()
            .into_iter()
            .map(|(position, _)| position)
            .collect::<HashSet<_>>();
        let cost = LayoutCandidateCost {
            block_count: occupied_cells.len(),
            bbox_volume: bbox.volume(),
        };

        Ok(Self {
            module_name,
            world,
            bbox,
            occupied_cells,
            blocked_cells: HashSet::new(),
            cost,
        })
    }
}
