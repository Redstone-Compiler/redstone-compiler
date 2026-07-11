use std::collections::HashSet;

use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::world::position::Position;

#[derive(Clone, Debug)]
pub struct ChildCandidatePool {
    pub instance_name: String,
    pub candidates: Vec<LayoutCandidate>,
}

pub fn rank_child_candidates(
    instance_name: impl Into<String>,
    mut candidates: Vec<LayoutCandidate>,
    limit: usize,
) -> ChildCandidatePool {
    candidates.sort_by_key(|candidate| {
        (
            candidate.cost.bbox_volume,
            candidate.cost.block_count,
            candidate_geometry_signature(candidate),
        )
    });

    let mut selected = Vec::new();
    let mut selected_signatures = HashSet::new();
    let mut overflow = Vec::new();
    for candidate in candidates {
        let signature = candidate_geometry_signature(&candidate);
        if selected.len() < limit && selected_signatures.insert(signature) {
            selected.push(candidate);
        } else {
            overflow.push(candidate);
        }
    }
    selected.extend(overflow.into_iter().take(limit.saturating_sub(selected.len())));

    ChildCandidatePool {
        instance_name: instance_name.into(),
        candidates: selected,
    }
}

pub fn layout_combinations(pools: &[ChildCandidatePool], limit: usize) -> Vec<Vec<usize>> {
    if pools.is_empty() || limit == 0 || pools.iter().any(|pool| pool.candidates.is_empty()) {
        return Vec::new();
    }

    let base = vec![0; pools.len()];
    let mut combinations = vec![base.clone()];
    for (pool_index, pool) in pools.iter().enumerate() {
        for candidate_index in 1..pool.candidates.len() {
            if combinations.len() >= limit {
                return combinations;
            }
            let mut combination = base.clone();
            combination[pool_index] = candidate_index;
            combinations.push(combination);
        }
    }
    combinations
}

fn candidate_geometry_signature(candidate: &LayoutCandidate) -> Vec<(String, Position)> {
    let mut ports = candidate
        .ports
        .iter()
        .map(|port| (port.name.clone(), port.primary_route_position()))
        .collect::<Vec<_>>();
    ports.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    ports
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::transform::place_and_route::estimate::BoundingBox;
    use crate::transform::place_and_route::global_pnr::ir::{
        LayoutCandidate, LayoutCandidateCost, PhysicalPort, PhysicalPortDirection, PortConnection,
    };
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

    use super::{layout_combinations, rank_child_candidates, ChildCandidatePool};

    fn candidate(name: &str, volume: usize, block_count: usize, port: Position) -> LayoutCandidate {
        LayoutCandidate {
            module_name: name.to_owned(),
            world: World3D::new(DimSize(1, 1, 1)),
            bbox: BoundingBox {
                min: Position(0, 0, 0),
                max: Position(volume.saturating_sub(1), 0, 0),
            },
            ports: vec![PhysicalPort {
                name: "p".to_owned(),
                direction: PhysicalPortDirection::Output,
                position: port,
                route_position: Some(port),
                access_points: vec![port],
                connection: PortConnection::Direct,
            }],
            occupied_cells: HashSet::new(),
            blocked_cells: HashSet::new(),
            cost: LayoutCandidateCost {
                block_count,
                bbox_volume: volume,
            },
        }
    }

    #[test]
    fn rank_child_candidates_keeps_bounded_geometry_diversity() {
        let candidates = vec![
            candidate("child", 1, 1, Position(0, 0, 0)),
            candidate("child", 2, 2, Position(0, 0, 0)),
            candidate("child", 3, 3, Position(8, 0, 0)),
        ];

        let pool = rank_child_candidates("instance", candidates, 2);

        assert_eq!(pool.instance_name, "instance");
        assert_eq!(pool.candidates.len(), 2);
        assert_eq!(pool.candidates[0].cost.bbox_volume, 1);
        assert_eq!(pool.candidates[1].ports[0].position, Position(8, 0, 0));
    }

    #[test]
    fn layout_combinations_are_stable_and_bounded() {
        let pools = vec![
            ChildCandidatePool {
                instance_name: "a".to_owned(),
                candidates: vec![
                    candidate("a", 1, 1, Position(0, 0, 0)),
                    candidate("a", 2, 2, Position(1, 0, 0)),
                ],
            },
            ChildCandidatePool {
                instance_name: "b".to_owned(),
                candidates: vec![
                    candidate("b", 1, 1, Position(0, 0, 0)),
                    candidate("b", 2, 2, Position(1, 0, 0)),
                ],
            },
        ];

        assert_eq!(
            layout_combinations(&pools, 3),
            vec![vec![0, 0], vec![1, 0], vec![0, 1]]
        );
    }
}
