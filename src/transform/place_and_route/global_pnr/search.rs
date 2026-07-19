use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::transform::place_and_route::global_pnr::router::RoutedNet;
use crate::world::position::Position;

#[derive(Clone, Debug)]
pub struct ChildCandidatePool {
    pub instance_name: String,
    pub candidates: Vec<LayoutCandidate>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalSolutionCost {
    pub unrouted_nets: usize,
    pub placement_volume: usize,
    pub estimated_wire_length: usize,
    pub vertical_distance: usize,
    pub routed_path_length: usize,
    pub routed_block_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GlobalSolutionDiagnostics {
    pub layout_combination_index: usize,
    pub placement_attempt_index: usize,
    pub route_order: String,
    pub last_failure: Option<String>,
}

#[derive(Clone, Debug)]
pub struct GlobalSolution {
    pub selected_candidate_indices: Vec<usize>,
    pub placed_modules: Vec<PlacedModule>,
    pub routed_nets: Vec<RoutedNet>,
    pub cost: GlobalSolutionCost,
    pub diagnostics: GlobalSolutionDiagnostics,
}

pub fn rank_child_candidates(
    instance_name: impl Into<String>,
    mut candidates: Vec<LayoutCandidate>,
    limit: usize,
) -> ChildCandidatePool {
    sort_candidates_by_pareto_frontier(&mut candidates);

    select_ranked_candidates(instance_name, candidates, limit)
}

pub fn rank_child_candidates_with_preferred(
    instance_name: impl Into<String>,
    mut candidates: Vec<LayoutCandidate>,
    limit: usize,
    preferred_index: usize,
) -> ChildCandidatePool {
    if candidates.is_empty() {
        return ChildCandidatePool {
            instance_name: instance_name.into(),
            candidates,
        };
    }
    let preferred = candidates.remove(preferred_index.min(candidates.len() - 1));
    sort_candidates_by_pareto_frontier(&mut candidates);
    candidates.insert(0, preferred);
    select_ranked_candidates(instance_name, candidates, limit)
}

fn sort_candidates_by_pareto_frontier(candidates: &mut Vec<LayoutCandidate>) {
    let dominated = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            candidates.iter().enumerate().any(|(other_index, other)| {
                other_index != index && candidate_cost_dominates(other, candidate)
            })
        })
        .collect::<Vec<_>>();
    let mut indexed = candidates
        .drain(..)
        .enumerate()
        .collect::<Vec<(usize, LayoutCandidate)>>();
    indexed.sort_by_key(|(index, candidate)| {
        (
            dominated[*index],
            candidate.cost.bbox_volume,
            candidate.cost.block_count,
            candidate.cost.bbox_footprint,
            candidate.cost.bbox_height,
            Reverse(candidate.cost.port_access_points),
            candidate_geometry_signature(candidate),
        )
    });
    candidates.extend(indexed.into_iter().map(|(_, candidate)| candidate));
}

fn candidate_cost_dominates(left: &LayoutCandidate, right: &LayoutCandidate) -> bool {
    let left = &left.cost;
    let right = &right.cost;
    let no_worse = left.bbox_volume <= right.bbox_volume
        && left.block_count <= right.block_count
        && left.bbox_footprint <= right.bbox_footprint
        && left.bbox_height <= right.bbox_height
        && left.port_access_points >= right.port_access_points;
    let strictly_better = left.bbox_volume < right.bbox_volume
        || left.block_count < right.block_count
        || left.bbox_footprint < right.bbox_footprint
        || left.bbox_height < right.bbox_height
        || left.port_access_points > right.port_access_points;
    no_worse && strictly_better
}

fn select_ranked_candidates(
    instance_name: impl Into<String>,
    candidates: Vec<LayoutCandidate>,
    limit: usize,
) -> ChildCandidatePool {
    let mut selected = Vec::new();
    let mut selected_signatures = HashSet::new();
    let mut overflow = Vec::new();
    for (index, candidate) in candidates.into_iter().enumerate() {
        let signature = candidate_geometry_signature(&candidate);
        let geometry_is_new = selected_signatures.insert(signature);
        if selected.len() < limit && (index == 0 || geometry_is_new) {
            selected.push(candidate);
        } else {
            overflow.push(candidate);
        }
    }
    selected.extend(
        overflow
            .into_iter()
            .take(limit.saturating_sub(selected.len())),
    );

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
    let mut frontier = BinaryHeap::new();
    let mut queued = HashSet::new();
    let mut sequence = 0usize;
    frontier.push(Reverse((0usize, sequence, base.clone())));
    queued.insert(base);

    let mut combinations = Vec::with_capacity(limit);
    while let Some(Reverse((_rank_sum, _sequence, selection))) = frontier.pop() {
        combinations.push(selection.clone());
        if combinations.len() >= limit {
            break;
        }

        for (pool_index, pool) in pools.iter().enumerate() {
            if selection[pool_index] + 1 >= pool.candidates.len() {
                continue;
            }
            let mut neighbor = selection.clone();
            neighbor[pool_index] += 1;
            if queued.insert(neighbor.clone()) {
                sequence += 1;
                let rank_sum = neighbor.iter().sum();
                frontier.push(Reverse((rank_sum, sequence, neighbor)));
            }
        }
    }
    combinations
}

pub fn select_layout_combination(
    pools: &[ChildCandidatePool],
    selection: &[usize],
) -> Option<Vec<LayoutCandidate>> {
    if pools.len() != selection.len() {
        return None;
    }
    pools
        .iter()
        .zip(selection)
        .map(|(pool, &candidate_index)| pool.candidates.get(candidate_index).cloned())
        .collect()
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

    use super::{
        layout_combinations, rank_child_candidates, rank_child_candidates_with_preferred,
        select_layout_combination, ChildCandidatePool, GlobalSolutionCost,
    };
    use crate::transform::place_and_route::estimate::BoundingBox;
    use crate::transform::place_and_route::global_pnr::ir::{
        LayoutCandidate, LayoutCandidateCost, PhysicalPort, PhysicalPortDirection, PortConnection,
    };
    use crate::transform::place_and_route::global_pnr::router::{
        ordered_module_variables, NetOrderStrategy, RoutingConnection,
    };
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

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
                bbox_footprint: volume,
                bbox_height: 1,
                port_access_points: 1,
            },
        }
    }

    fn variable(source: &str, target: &str) -> RoutingConnection {
        RoutingConnection {
            source: (source.to_owned(), "q".to_owned()),
            target: (target.to_owned(), "d".to_owned()),
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
    fn rank_child_candidates_preserves_legacy_preferred_candidate_first() {
        let candidates = vec![
            candidate("child", 5, 5, Position(5, 0, 0)),
            candidate("child", 1, 1, Position(0, 0, 0)),
            candidate("child", 2, 2, Position(8, 0, 0)),
        ];

        let pool = rank_child_candidates_with_preferred("instance", candidates, 3, 0);

        assert_eq!(pool.candidates[0].cost.bbox_volume, 5);
        assert_eq!(pool.candidates.len(), 3);
    }

    #[test]
    fn rank_child_candidates_keeps_a_larger_candidate_with_more_port_access() {
        let compact = candidate("child", 1, 1, Position(0, 0, 0));
        let mut routeable = candidate("child", 2, 2, Position(1, 0, 0));
        routeable.cost.port_access_points = 3;
        routeable.ports[0].access_points =
            vec![Position(1, 0, 0), Position(1, 1, 0), Position(1, 0, 1)];

        let pool = rank_child_candidates("instance", vec![compact, routeable], 2);

        assert_eq!(pool.candidates.len(), 2);
        assert_eq!(pool.candidates[0].cost.bbox_volume, 1);
        assert_eq!(pool.candidates[1].cost.port_access_points, 3);
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

    #[test]
    fn layout_combinations_cover_multi_child_interactions() {
        let pools = (0..3)
            .map(|index| ChildCandidatePool {
                instance_name: format!("child_{index}"),
                candidates: vec![
                    candidate("child", 1, 1, Position(0, 0, 0)),
                    candidate("child", 2, 2, Position(1, 0, 0)),
                ],
            })
            .collect::<Vec<_>>();

        let combinations = layout_combinations(&pools, 8);
        assert_eq!(combinations.len(), 8);
        assert_eq!(combinations[0], vec![0, 0, 0]);
        assert!(combinations.contains(&vec![1, 1, 0]));
        assert!(combinations.contains(&vec![1, 0, 1]));
        assert!(combinations.contains(&vec![0, 1, 1]));
        assert!(combinations.contains(&vec![1, 1, 1]));
        assert_eq!(combinations, layout_combinations(&pools, 8));
        assert_eq!(combinations.iter().collect::<HashSet<_>>().len(), 8);
    }

    #[test]
    fn global_solution_cost_orders_complete_solutions() {
        let compact = GlobalSolutionCost {
            unrouted_nets: 0,
            placement_volume: 100,
            estimated_wire_length: 30,
            vertical_distance: 2,
            routed_path_length: 40,
            routed_block_count: 45,
        };
        let long_routes = GlobalSolutionCost {
            routed_path_length: 80,
            ..compact
        };
        let incomplete = GlobalSolutionCost {
            unrouted_nets: 1,
            placement_volume: 1,
            estimated_wire_length: 1,
            vertical_distance: 0,
            routed_path_length: 0,
            routed_block_count: 0,
        };

        assert!(compact < long_routes);
        assert!(long_routes < incomplete);
    }

    #[test]
    fn select_layout_combination_uses_one_candidate_from_each_pool() {
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
                    candidate("b", 3, 3, Position(0, 0, 0)),
                    candidate("b", 4, 4, Position(1, 0, 0)),
                ],
            },
        ];

        let selected = select_layout_combination(&pools, &[1, 0]).unwrap();

        assert_eq!(selected[0].cost.bbox_volume, 2);
        assert_eq!(selected[1].cost.bbox_volume, 3);
    }

    #[test]
    fn ordered_module_variables_supports_fanout_and_reverse_strategies() {
        let vars = vec![
            variable("shared", "a"),
            variable("single", "b"),
            variable("shared", "c"),
        ];

        let fanout = ordered_module_variables(&vars, NetOrderStrategy::HighestFanoutFirst);
        let reverse = ordered_module_variables(&vars, NetOrderStrategy::ReverseCriticality);

        assert_eq!(fanout[0].source.0, "shared");
        assert_eq!(fanout[1].source.0, "shared");
        assert_eq!(reverse.last().unwrap().source.0, "shared");
    }
}
