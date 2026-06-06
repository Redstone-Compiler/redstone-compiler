use std::collections::{HashMap, HashSet};

use crate::graph::module::{GraphModule, GraphModulePortTarget};
use crate::transform::place_and_route::estimate::BoundingBox;
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::world::position::Position;

const GLOBAL_PLACEMENT_MARGIN: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalPlacementConfig {
    pub spacing: usize,
    pub shelf_width: usize,
    pub max_attempts: usize,
}

impl Default for GlobalPlacementConfig {
    fn default() -> Self {
        Self {
            spacing: 2,
            shelf_width: 64,
            max_attempts: 16,
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
    let order = (0..candidates.len()).collect::<Vec<_>>();
    place_candidates_on_shelves_in_order(candidates, &order, config)
}

pub fn placement_candidates(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
) -> Vec<Vec<PlacedModule>> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut placements = Vec::new();
    let original_order = (0..candidates.len()).collect::<Vec<_>>();
    let net_order = net_aware_candidate_order(module, candidates);

    for shelf_width in placement_shelf_width_options(config.shelf_width) {
        for spacing in placement_spacing_options(config.spacing) {
            let config = GlobalPlacementConfig {
                spacing,
                shelf_width,
                ..*config
            };
            push_unique_placement(
                &mut placements,
                place_candidates_on_shelves_in_order(candidates, &original_order, &config),
            );
            push_unique_placement(
                &mut placements,
                place_candidates_on_shelves_in_order(candidates, &net_order, &config),
            );

            for columns in grid_column_options(candidates.len()) {
                push_unique_placement(
                    &mut placements,
                    place_candidates_on_grid_in_order(
                        candidates,
                        &original_order,
                        columns,
                        &config,
                    ),
                );
                push_unique_placement(
                    &mut placements,
                    place_candidates_on_grid_in_order(candidates, &net_order, columns, &config),
                );
            }
        }
    }

    placements.sort_by_key(|placed| placement_cost(module, candidates, placed));
    placements.truncate(config.max_attempts.max(1));
    placements
}

fn placement_spacing_options(base: usize) -> Vec<usize> {
    let mut options = [base, base + 2, base + 4, base.saturating_mul(2)]
        .into_iter()
        .map(|spacing| spacing.max(1))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    options.sort();
    options
}

fn placement_shelf_width_options(base: usize) -> Vec<usize> {
    let mut options = [base, base.saturating_mul(2), 64, 96]
        .into_iter()
        .map(|width| width.max(1))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    options.sort();
    options
}

fn place_candidates_on_shelves_in_order(
    candidates: &[LayoutCandidate],
    order: &[usize],
    config: &GlobalPlacementConfig,
) -> Vec<PlacedModule> {
    let mut placed = Vec::new();
    let mut cursor_x = GLOBAL_PLACEMENT_MARGIN;
    let mut cursor_y = GLOBAL_PLACEMENT_MARGIN;
    let mut shelf_depth = 0usize;

    for &candidate_index in order {
        let candidate = &candidates[candidate_index];
        let width = candidate.bbox.width();
        let depth = candidate.bbox.depth();

        if cursor_x > GLOBAL_PLACEMENT_MARGIN && cursor_x + width > config.shelf_width {
            cursor_x = GLOBAL_PLACEMENT_MARGIN;
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

fn place_candidates_on_grid_in_order(
    candidates: &[LayoutCandidate],
    order: &[usize],
    columns: usize,
    config: &GlobalPlacementConfig,
) -> Vec<PlacedModule> {
    let columns = columns.max(1);
    let mut column_widths = vec![0usize; columns];
    let mut row_depths = Vec::<usize>::new();

    for (slot, &candidate_index) in order.iter().enumerate() {
        let candidate = &candidates[candidate_index];
        let column = slot % columns;
        let row = slot / columns;
        if row_depths.len() <= row {
            row_depths.push(0);
        }
        column_widths[column] = column_widths[column].max(candidate.bbox.width());
        row_depths[row] = row_depths[row].max(candidate.bbox.depth());
    }

    let mut column_offsets = vec![GLOBAL_PLACEMENT_MARGIN; columns];
    for column in 1..columns {
        column_offsets[column] =
            column_offsets[column - 1] + column_widths[column - 1] + config.spacing;
    }

    let mut row_offsets = vec![GLOBAL_PLACEMENT_MARGIN; row_depths.len()];
    for row in 1..row_depths.len() {
        row_offsets[row] = row_offsets[row - 1] + row_depths[row - 1] + config.spacing;
    }

    order
        .iter()
        .enumerate()
        .map(|(slot, &candidate_index)| {
            let candidate = &candidates[candidate_index];
            let column = slot % columns;
            let row = slot / columns;
            PlacedModule {
                module_name: candidate.module_name.clone(),
                candidate_index,
                origin: Position(
                    column_offsets[column],
                    row_offsets[row],
                    candidate.bbox.min.2,
                ),
                bbox: candidate.bbox,
            }
        })
        .collect()
}

fn grid_column_options(count: usize) -> Vec<usize> {
    let square = (count as f64).sqrt().ceil() as usize;
    let mut options = [2, square, square.saturating_add(1)]
        .into_iter()
        .filter(|columns| *columns > 1 && *columns < count)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    options.sort();
    options
}

fn push_unique_placement(placements: &mut Vec<Vec<PlacedModule>>, placed: Vec<PlacedModule>) {
    let signature = placement_signature(&placed);
    if placements
        .iter()
        .any(|existing| placement_signature(existing) == signature)
    {
        return;
    }
    placements.push(placed);
}

fn placement_signature(placed: &[PlacedModule]) -> Vec<(usize, Position)> {
    let mut signature = placed
        .iter()
        .map(|placed| (placed.candidate_index, placed.origin))
        .collect::<Vec<_>>();
    signature.sort();
    signature
}

fn net_aware_candidate_order(module: &GraphModule, candidates: &[LayoutCandidate]) -> Vec<usize> {
    let module_to_candidate = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.module_name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let edges = module_edges(module, &module_to_candidate);
    if edges.is_empty() {
        return (0..candidates.len()).collect();
    }

    let mut degree = vec![0usize; candidates.len()];
    for &(left, right) in &edges {
        degree[left] += 1;
        degree[right] += 1;
    }

    let mut order = Vec::new();
    let mut remaining = (0..candidates.len()).collect::<HashSet<_>>();
    let first = (0..candidates.len())
        .max_by_key(|index| (degree[*index], std::cmp::Reverse(*index)))
        .unwrap_or(0);
    order.push(first);
    remaining.remove(&first);

    while !remaining.is_empty() {
        let next = remaining
            .iter()
            .copied()
            .max_by_key(|index| {
                let links_to_placed = edges
                    .iter()
                    .filter(|(left, right)| {
                        (*left == *index && order.contains(right))
                            || (*right == *index && order.contains(left))
                    })
                    .count();
                (links_to_placed, degree[*index], std::cmp::Reverse(*index))
            })
            .unwrap();
        order.push(next);
        remaining.remove(&next);
    }

    order
}

fn module_edges(
    module: &GraphModule,
    module_to_candidate: &HashMap<&str, usize>,
) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for var in &module.vars {
        let Some(&source) = module_to_candidate.get(var.source.0.as_str()) else {
            continue;
        };
        let Some(&target) = module_to_candidate.get(var.target.0.as_str()) else {
            continue;
        };
        if source != target {
            edges.push((source, target));
        }
    }

    for port in &module.ports {
        let targets = target_modules(&port.target);
        for (left_index, left) in targets.iter().enumerate() {
            for right in targets.iter().skip(left_index + 1) {
                let Some(&left) = module_to_candidate.get(left.as_str()) else {
                    continue;
                };
                let Some(&right) = module_to_candidate.get(right.as_str()) else {
                    continue;
                };
                if left != right {
                    edges.push((left, right));
                }
            }
        }
    }

    edges
}

fn target_modules(target: &GraphModulePortTarget) -> Vec<String> {
    match target {
        GraphModulePortTarget::Module(module, _) => vec![module.clone()],
        GraphModulePortTarget::Wire(targets) => targets
            .iter()
            .map(|(module, _)| module.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect(),
        GraphModulePortTarget::Node(_) => Vec::new(),
    }
}

fn placement_cost(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) -> usize {
    let placed_by_module = placed
        .iter()
        .map(|placed| (placed.module_name.as_str(), placed))
        .collect::<HashMap<_, _>>();
    let mut cost = placement_bbox_cost(placed);

    for var in &module.vars {
        let Some(source) = placed_by_module.get(var.source.0.as_str()) else {
            continue;
        };
        let Some(target) = placed_by_module.get(var.target.0.as_str()) else {
            continue;
        };
        let source_candidate = &candidates[source.candidate_index];
        let target_candidate = &candidates[target.candidate_index];
        let Some(source_port) = source_candidate
            .ports
            .iter()
            .find(|port| port.name == var.source.1)
        else {
            continue;
        };
        let Some(target_port) = target_candidate
            .ports
            .iter()
            .find(|port| port.name == var.target.1)
        else {
            continue;
        };
        let source_position =
            translate_candidate_position(source_port.position, source_candidate, source);
        let target_position =
            translate_candidate_position(target_port.position, target_candidate, target);
        cost += source_position.manhattan_distance(&target_position) * 8;
        cost += source_position.2.abs_diff(target_position.2) * 16;
    }

    cost
}

fn placement_bbox_cost(placed: &[PlacedModule]) -> usize {
    let Some(first) = placed.first() else {
        return 0;
    };
    let mut min = first.origin;
    let mut max = first.origin;
    for placed in placed {
        min.0 = min.0.min(placed.origin.0);
        min.1 = min.1.min(placed.origin.1);
        min.2 = min.2.min(placed.origin.2);
        max.0 = max.0.max(placed.origin.0 + placed.bbox.width());
        max.1 = max.1.max(placed.origin.1 + placed.bbox.depth());
        max.2 = max.2.max(placed.origin.2 + placed.bbox.height());
    }
    (max.0 - min.0 + 1) * (max.1 - min.1 + 1) * (max.2 - min.2 + 1)
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
