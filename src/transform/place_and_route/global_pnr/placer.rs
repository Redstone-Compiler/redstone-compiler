use crate::transform::place_and_route::estimate::BoundingBox;
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::world::position::Position;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalPlacementConfig {
    pub spacing: usize,
    pub shelf_width: usize,
}

impl Default for GlobalPlacementConfig {
    fn default() -> Self {
        Self {
            spacing: 2,
            shelf_width: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlacedModule {
    pub module_name: String,
    pub candidate_index: usize,
    pub origin: Position,
    pub bbox: BoundingBox,
}

pub fn place_candidates_on_shelves(
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
) -> Vec<PlacedModule> {
    // TODO: 지금은 child layout을 순서대로 shelf에 올리는 단순 휴리스틱이다.
    // module net length, routeability, port alignment, route congestion을 비용으로 넣는
    // 일반 cost model을 도입해서 특수한 사후 위치 보정을 대체해야 한다.
    let mut placed = Vec::new();
    let mut cursor_x = 0usize;
    let mut cursor_y = 0usize;
    let mut shelf_depth = 0usize;

    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let width = candidate.bbox.width();
        let depth = candidate.bbox.depth();

        if cursor_x > 0 && cursor_x + width > config.shelf_width {
            cursor_x = 0;
            cursor_y += shelf_depth + config.spacing;
            shelf_depth = 0;
        }

        placed.push(PlacedModule {
            module_name: candidate.module_name.clone(),
            candidate_index,
            origin: Position(cursor_x, cursor_y, candidate.bbox.min.2),
            bbox: candidate.bbox,
        });

        cursor_x += width + config.spacing;
        shelf_depth = shelf_depth.max(depth);
    }

    placed
}
