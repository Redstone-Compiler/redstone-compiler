use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::graph::module::GraphModule;
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::transform::place_and_route::global_pnr::policy::Free3DPlacementConfig;
use crate::world::position::Position;

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn component(self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    fn add_component(&mut self, axis: usize, value: f64) {
        match axis {
            0 => self.x += value,
            1 => self.y += value,
            _ => self.z += value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Body {
    center: Vec3,
    velocity: Vec3,
    size: [f64; 3],
}

pub(crate) fn place_free_3d(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    config: Free3DPlacementConfig,
) -> Option<Vec<PlacedModule>> {
    let candidate_by_name = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.module_name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let edges = module
        .vars
        .iter()
        .filter_map(|var| {
            let source = *candidate_by_name.get(var.source.0.as_str())?;
            let target = *candidate_by_name.get(var.target.0.as_str())?;
            (source != target).then_some((source, target))
        })
        .collect::<Vec<_>>();
    place_free_3d_with_edges(candidates, &edges, config)
}

pub(crate) fn place_free_3d_with_edges(
    candidates: &[LayoutCandidate],
    edges: &[(usize, usize)],
    config: Free3DPlacementConfig,
) -> Option<Vec<PlacedModule>> {
    if candidates.is_empty() {
        return None;
    }

    let mut bodies = seed_bodies(candidates, config.clearance, config.seed);
    relax(&mut bodies, edges, config);
    let mut origins = snap_origins(&bodies);
    legalize(&mut origins, candidates, config.clearance)?;
    shift_into_positive_world(&mut origins, 4);

    Some(
        candidates
            .iter()
            .enumerate()
            .map(|(candidate_index, candidate)| PlacedModule {
                module_name: candidate.module_name.clone(),
                candidate_index,
                origin: Position(
                    origins[candidate_index][0] as usize,
                    origins[candidate_index][1] as usize,
                    origins[candidate_index][2] as usize,
                ),
                bbox: candidate.bbox,
            })
            .collect(),
    )
}

fn seed_bodies(candidates: &[LayoutCandidate], clearance: usize, seed: u64) -> Vec<Body> {
    let side = (1..)
        .find(|side| side * side * side >= candidates.len())
        .unwrap_or(1);
    let stride = [
        candidates
            .iter()
            .map(|candidate| candidate.bbox.width())
            .max()
            .unwrap_or(1)
            + clearance,
        candidates
            .iter()
            .map(|candidate| candidate.bbox.depth())
            .max()
            .unwrap_or(1)
            + clearance,
        candidates
            .iter()
            .map(|candidate| candidate.bbox.height())
            .max()
            .unwrap_or(1)
            + clearance,
    ];

    let mut cells = (0..candidates.len())
        .map(|index| [index % side, (index / side) % side, index / (side * side)])
        .collect::<Vec<_>>();
    if seed != 0 {
        cells.shuffle(&mut StdRng::seed_from_u64(seed));
    }

    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let size = [
                candidate.bbox.width() as f64,
                candidate.bbox.depth() as f64,
                candidate.bbox.height() as f64,
            ];
            let cell = cells[index];
            Body {
                center: Vec3 {
                    x: (cell[0] * stride[0]) as f64 + size[0] / 2.0,
                    y: (cell[1] * stride[1]) as f64 + size[1] / 2.0,
                    z: (cell[2] * stride[2]) as f64 + size[2] / 2.0,
                },
                velocity: Vec3::default(),
                size,
            }
        })
        .collect()
}

fn relax(bodies: &mut [Body], edges: &[(usize, usize)], config: Free3DPlacementConfig) {
    for _ in 0..config.iterations {
        let count = bodies.len() as f64;
        let centroid = bodies.iter().fold(Vec3::default(), |mut sum, body| {
            sum.x += body.center.x;
            sum.y += body.center.y;
            sum.z += body.center.z;
            sum
        });
        let centroid = Vec3 {
            x: centroid.x / count,
            y: centroid.y / count,
            z: centroid.z / count,
        };
        let mut forces = bodies
            .iter()
            .map(|body| Vec3 {
                x: (centroid.x - body.center.x) * config.compactness,
                y: (centroid.y - body.center.y) * config.compactness,
                z: (centroid.z - body.center.z) * config.compactness
                    / config.vertical_scale.max(0.001),
            })
            .collect::<Vec<_>>();

        for &(source, target) in edges {
            let delta = Vec3 {
                x: bodies[target].center.x - bodies[source].center.x,
                y: bodies[target].center.y - bodies[source].center.y,
                z: bodies[target].center.z - bodies[source].center.z,
            };
            let spring = Vec3 {
                x: delta.x * config.attraction,
                y: delta.y * config.attraction,
                z: delta.z * config.attraction / config.vertical_scale.max(0.001),
            };
            forces[source].x += spring.x;
            forces[source].y += spring.y;
            forces[source].z += spring.z;
            forces[target].x -= spring.x;
            forces[target].y -= spring.y;
            forces[target].z -= spring.z;
        }

        for left_index in 0..bodies.len() {
            for right_index in left_index + 1..bodies.len() {
                let left = bodies[left_index];
                let right = bodies[right_index];
                let delta = [
                    right.center.x - left.center.x,
                    right.center.y - left.center.y,
                    right.center.z - left.center.z,
                ];
                let required = [
                    (left.size[0] + right.size[0]) / 2.0 + config.clearance as f64,
                    (left.size[1] + right.size[1]) / 2.0 + config.clearance as f64,
                    (left.size[2] + right.size[2]) / 2.0 + config.clearance as f64,
                ];
                let penetration = [
                    required[0] - delta[0].abs(),
                    required[1] - delta[1].abs(),
                    required[2] - delta[2].abs(),
                ];
                if penetration.iter().all(|value| *value > 0.0) {
                    let axis = (0..3)
                        .min_by(|&left, &right| penetration[left].total_cmp(&penetration[right]))
                        .unwrap_or(0);
                    let direction = if delta[axis] < 0.0 { -1.0 } else { 1.0 };
                    let force = direction * penetration[axis] * config.repulsion;
                    forces[left_index].add_component(axis, -force);
                    forces[right_index].add_component(axis, force);
                }
            }
        }

        for (body, force) in bodies.iter_mut().zip(forces) {
            body.velocity.x = body.velocity.x * config.damping + force.x;
            body.velocity.y = body.velocity.y * config.damping + force.y;
            body.velocity.z =
                body.velocity.z * config.damping + force.z / config.vertical_scale.max(0.001);
            for axis in 0..3 {
                let movement = body
                    .velocity
                    .component(axis)
                    .clamp(-config.step_size, config.step_size);
                body.center.add_component(axis, movement);
            }
        }
    }
}

fn snap_origins(bodies: &[Body]) -> Vec<[i64; 3]> {
    bodies
        .iter()
        .map(|body| {
            [
                (body.center.x - body.size[0] / 2.0).round() as i64,
                (body.center.y - body.size[1] / 2.0).round() as i64,
                (body.center.z - body.size[2] / 2.0).round() as i64,
            ]
        })
        .collect()
}

fn legalize(
    origins: &mut [[i64; 3]],
    candidates: &[LayoutCandidate],
    clearance: usize,
) -> Option<()> {
    let sizes = candidates
        .iter()
        .map(|candidate| {
            [
                candidate.bbox.width() as i64 + clearance as i64,
                candidate.bbox.depth() as i64 + clearance as i64,
                candidate.bbox.height() as i64 + clearance as i64,
            ]
        })
        .collect::<Vec<_>>();

    for _ in 0..origins.len().saturating_mul(origins.len()).max(1) * 8 {
        let mut collision = None;
        'pairs: for left in 0..origins.len() {
            for right in left + 1..origins.len() {
                let penetration = [0, 1, 2].map(|axis| {
                    (origins[left][axis] + sizes[left][axis])
                        .min(origins[right][axis] + sizes[right][axis])
                        - origins[left][axis].max(origins[right][axis])
                });
                if penetration.iter().all(|value| *value > 0) {
                    collision = Some((right, penetration));
                    break 'pairs;
                }
            }
        }
        let Some((right, penetration)) = collision else {
            return Some(());
        };
        let axis = (0..3).min_by_key(|&axis| penetration[axis]).unwrap_or(0);
        origins[right][axis] += penetration[axis];
    }
    None
}

fn shift_into_positive_world(origins: &mut [[i64; 3]], margin: i64) {
    for axis in 0..3 {
        let min = origins.iter().map(|origin| origin[axis]).min().unwrap_or(0);
        let shift = margin.saturating_sub(min);
        for origin in origins.iter_mut() {
            origin[axis] += shift;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::place_free_3d;
    use crate::graph::module::{GraphModule, GraphModuleVariable};
    use crate::transform::place_and_route::estimate::BoundingBox;
    use crate::transform::place_and_route::global_pnr::ir::{LayoutCandidate, LayoutCandidateCost};
    use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
    use crate::transform::place_and_route::global_pnr::policy::Free3DPlacementConfig;
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

    fn candidate(index: usize) -> LayoutCandidate {
        LayoutCandidate {
            module_name: format!("child_{index}"),
            world: World3D::new(DimSize(1, 1, 1)),
            bbox: BoundingBox {
                min: Position(0, 0, 0),
                max: Position(4, 4, 4),
            },
            ports: Vec::new(),
            occupied_cells: HashSet::new(),
            blocked_cells: HashSet::new(),
            cost: LayoutCandidateCost::default(),
        }
    }

    fn overlaps(left: &PlacedModule, right: &PlacedModule, clearance: usize) -> bool {
        let left_max = Position(
            left.origin.0 + left.bbox.width() - 1 + clearance,
            left.origin.1 + left.bbox.depth() - 1 + clearance,
            left.origin.2 + left.bbox.height() - 1 + clearance,
        );
        let right_max = Position(
            right.origin.0 + right.bbox.width() - 1 + clearance,
            right.origin.1 + right.bbox.depth() - 1 + clearance,
            right.origin.2 + right.bbox.height() - 1 + clearance,
        );
        left.origin.0 <= right_max.0
            && right.origin.0 <= left_max.0
            && left.origin.1 <= right_max.1
            && right.origin.1 <= left_max.1
            && left.origin.2 <= right_max.2
            && right.origin.2 <= left_max.2
    }

    #[test]
    fn free_3d_placement_uses_volume_and_legalizes_boxes() {
        let candidates = (0..8).map(candidate).collect::<Vec<_>>();
        let config = Free3DPlacementConfig::default();

        let placed =
            place_free_3d(&GraphModule::default(), &candidates, config).expect("free 3D placement");

        assert_eq!(placed.len(), candidates.len());
        assert!(
            placed
                .iter()
                .map(|item| item.origin.2)
                .collect::<HashSet<_>>()
                .len()
                > 1
        );
        assert!(placed.iter().enumerate().all(|(left_index, left)| {
            placed
                .iter()
                .skip(left_index + 1)
                .all(|right| !overlaps(left, right, config.clearance))
        }));
    }

    fn squared_center_distance(placed: &[PlacedModule], left: usize, right: usize) -> usize {
        let center = |item: &PlacedModule| {
            [
                item.origin.0 + item.bbox.width() / 2,
                item.origin.1 + item.bbox.depth() / 2,
                item.origin.2 + item.bbox.height() / 2,
            ]
        };
        let left = center(&placed[left]);
        let right = center(&placed[right]);
        left[0].abs_diff(right[0]).pow(2)
            + left[1].abs_diff(right[1]).pow(2)
            + left[2].abs_diff(right[2]).pow(2)
    }

    #[test]
    fn attraction_reduces_connected_module_distance() {
        let candidates = (0..4).map(candidate).collect::<Vec<_>>();
        let module = GraphModule {
            vars: vec![GraphModuleVariable {
                source: ("child_0".to_owned(), "out".to_owned()),
                target: ("child_3".to_owned(), "in".to_owned()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let base = Free3DPlacementConfig {
            compactness: 0.0,
            iterations: 40,
            ..Default::default()
        };
        let without = place_free_3d(
            &module,
            &candidates,
            Free3DPlacementConfig {
                attraction: 0.0,
                ..base
            },
        )
        .unwrap();
        let with = place_free_3d(
            &module,
            &candidates,
            Free3DPlacementConfig {
                attraction: 0.08,
                ..base
            },
        )
        .unwrap();

        assert!(squared_center_distance(&with, 0, 3) < squared_center_distance(&without, 0, 3));
    }

    #[test]
    fn placement_seed_changes_only_global_origins_and_remains_legal() {
        let candidates = (0..8).map(candidate).collect::<Vec<_>>();
        let module = GraphModule::default();
        let base = Free3DPlacementConfig {
            iterations: 0,
            ..Default::default()
        };
        let first = place_free_3d(
            &module,
            &candidates,
            Free3DPlacementConfig { seed: 1, ..base },
        )
        .unwrap();
        let second = place_free_3d(
            &module,
            &candidates,
            Free3DPlacementConfig { seed: 2, ..base },
        )
        .unwrap();

        assert_ne!(
            first.iter().map(|item| item.origin).collect::<Vec<_>>(),
            second.iter().map(|item| item.origin).collect::<Vec<_>>()
        );
        for placed in [&first, &second] {
            assert!(placed.iter().enumerate().all(|(left_index, left)| {
                placed
                    .iter()
                    .skip(left_index + 1)
                    .all(|right| !overlaps(left, right, base.clearance))
            }));
        }
    }
}
use std::collections::HashMap;
