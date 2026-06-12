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

    if is_register_bit_module_set(candidates) {
        for spacing in placement_spacing_options(config.spacing) {
            let config = GlobalPlacementConfig { spacing, ..*config };
            if let Some(placed) = place_register_bit_grid(candidates, &config) {
                push_unique_placement(&mut placements, placed);
            }
            if let Some(placed) = place_register_bit_slices(candidates, &config) {
                push_unique_placement(&mut placements, placed);
            }
        }
        placements.truncate(config.max_attempts.max(1));
        return placements;
    }

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

fn is_register_bit_module_set(candidates: &[LayoutCandidate]) -> bool {
    !candidates.is_empty()
        && candidates
            .iter()
            .all(|candidate| register_bit_module_role(&candidate.module_name).is_some())
}

fn place_register_bit_slices(
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
) -> Option<Vec<PlacedModule>> {
    let mut slices = HashMap::<usize, RegisterBitSlice>::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let Some((bit, role)) = register_bit_module_role(&candidate.module_name) else {
            return None;
        };
        let slice = slices.entry(bit).or_default();
        match role {
            RegisterBitRole::Clock => {
                slice.clock = choose_register_bit_candidate(candidates, slice.clock, index, role)
            }
            RegisterBitRole::Next => {
                slice.next = choose_register_bit_candidate(candidates, slice.next, index, role)
            }
            RegisterBitRole::Master => {
                slice.master = choose_register_bit_candidate(candidates, slice.master, index, role)
            }
            RegisterBitRole::Slave => {
                slice.slave = choose_register_bit_candidate(candidates, slice.slave, index, role)
            }
        }
    }

    let mut bits = slices.into_iter().collect::<Vec<_>>();
    bits.sort_by_key(|(bit, _)| *bit);
    if bits.is_empty()
        || bits.iter().any(|(_, slice)| {
            slice.clock.is_none()
                || slice.next.is_none()
                || slice.master.is_none()
                || slice.slave.is_none()
        })
    {
        return None;
    }

    let mut placed = Vec::new();
    let mut cursor_x = GLOBAL_PLACEMENT_MARGIN;
    let cursor_y = GLOBAL_PLACEMENT_MARGIN;

    for (_, slice) in bits {
        for candidate_index in [slice.master?, slice.slave?, slice.next?, slice.clock?] {
            push_placed_candidate(&mut placed, candidates, candidate_index, cursor_x, cursor_y);
            cursor_x += candidates[candidate_index].bbox.width() + config.spacing;
        }
    }

    Some(placed)
}

fn place_register_bit_grid(
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
) -> Option<Vec<PlacedModule>> {
    let mut slices = HashMap::<usize, RegisterBitSlice>::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let Some((bit, role)) = register_bit_module_role(&candidate.module_name) else {
            return None;
        };
        let slice = slices.entry(bit).or_default();
        match role {
            RegisterBitRole::Clock => {
                slice.clock = choose_register_bit_candidate(candidates, slice.clock, index, role)
            }
            RegisterBitRole::Next => {
                slice.next = choose_register_bit_candidate(candidates, slice.next, index, role)
            }
            RegisterBitRole::Master => {
                slice.master = choose_register_bit_candidate(candidates, slice.master, index, role)
            }
            RegisterBitRole::Slave => {
                slice.slave = choose_register_bit_candidate(candidates, slice.slave, index, role)
            }
        }
    }

    let mut bits = slices.into_iter().collect::<Vec<_>>();
    bits.sort_by_key(|(bit, _)| *bit);
    if bits.is_empty()
        || bits.iter().any(|(_, slice)| {
            slice.clock.is_none()
                || slice.next.is_none()
                || slice.master.is_none()
                || slice.slave.is_none()
        })
    {
        return None;
    }

    let master_width = bits
        .iter()
        .map(|(_, slice)| candidates[slice.master.unwrap()].bbox.width())
        .max()
        .unwrap_or(0);
    let slave_width = bits
        .iter()
        .map(|(_, slice)| candidates[slice.slave.unwrap()].bbox.width())
        .max()
        .unwrap_or(0);
    let master_x = GLOBAL_PLACEMENT_MARGIN;
    let slave_x = master_x + master_width + config.spacing;
    let next_x = slave_x + slave_width + config.spacing;

    let mut placed = Vec::new();
    let mut cursor_y = GLOBAL_PLACEMENT_MARGIN;
    for (bit, slice) in bits {
        let clock = slice.clock?;
        let next = slice.next?;
        let master = slice.master?;
        let slave = slice.slave?;

        let master_y = cursor_y;
        let master_q_y = translated_port_y(candidates, master, master_y, "q", true)?;
        let slave_y = align_port_y(candidates, slave, "d", false, master_q_y)?;
        let slave_q_y = translated_port_y(candidates, slave, slave_y, "q", true)?;
        let next_feedback_port = format!("q_{bit}");
        let next_y = align_port_y(candidates, next, &next_feedback_port, true, slave_q_y)?;

        let data_row = [
            (master, master_x, master_y),
            (slave, slave_x, slave_y),
            (next, next_x, next_y),
        ];
        let data_bottom = data_row
            .iter()
            .map(|(index, _, y)| y + candidates[*index].bbox.depth())
            .max()
            .unwrap_or(cursor_y);
        let clock_y = data_bottom + config.spacing;
        let master_en_x = translated_port_x(candidates, master, master_x, "en", false)?;
        let clock_x = align_port_x(
            candidates,
            clock,
            "clk_n",
            true,
            master_en_x.saturating_sub(config.spacing),
        )?;

        let row = [
            (master, master_x, master_y),
            (slave, slave_x, slave_y),
            (next, next_x, next_y),
            (clock, clock_x, clock_y),
        ];
        let Some(row_shift) = row
            .iter()
            .map(|(index, _, y)| y.saturating_sub(candidates[*index].bbox.min.1))
            .min()
        else {
            return None;
        };
        let row_shift = GLOBAL_PLACEMENT_MARGIN.saturating_sub(row_shift);

        for (candidate_index, x, y) in row {
            push_placed_candidate(&mut placed, candidates, candidate_index, x, y + row_shift);
        }

        let row_bottom = [
            (next, next_y + row_shift),
            (slave, slave_y + row_shift),
            (master, master_y + row_shift),
            (clock, clock_y + row_shift),
        ]
        .into_iter()
        .map(|(index, y)| y + candidates[index].bbox.depth())
        .max()
        .unwrap_or(cursor_y);
        cursor_y = row_bottom + config.spacing;
    }

    Some(placed)
}

fn translated_port_y(
    candidates: &[LayoutCandidate],
    candidate_index: usize,
    origin_y: usize,
    port_name: &str,
    use_route_position: bool,
) -> Option<usize> {
    let candidate = &candidates[candidate_index];
    let position = candidate_port_position(candidate, port_name, use_route_position)?;
    Some(origin_y + position.1 - candidate.bbox.min.1)
}

fn align_port_y(
    candidates: &[LayoutCandidate],
    candidate_index: usize,
    port_name: &str,
    use_route_position: bool,
    target_y: usize,
) -> Option<usize> {
    let candidate = &candidates[candidate_index];
    let position = candidate_port_position(candidate, port_name, use_route_position)?;
    Some(target_y + candidate.bbox.min.1 - position.1)
}

fn translated_port_x(
    candidates: &[LayoutCandidate],
    candidate_index: usize,
    origin_x: usize,
    port_name: &str,
    use_route_position: bool,
) -> Option<usize> {
    let candidate = &candidates[candidate_index];
    let position = candidate_port_position(candidate, port_name, use_route_position)?;
    Some(origin_x + position.0 - candidate.bbox.min.0)
}

fn align_port_x(
    candidates: &[LayoutCandidate],
    candidate_index: usize,
    port_name: &str,
    use_route_position: bool,
    target_x: usize,
) -> Option<usize> {
    let candidate = &candidates[candidate_index];
    let position = candidate_port_position(candidate, port_name, use_route_position)?;
    Some((target_x + candidate.bbox.min.0).saturating_sub(position.0))
}

fn candidate_port_position(
    candidate: &LayoutCandidate,
    port_name: &str,
    use_route_position: bool,
) -> Option<Position> {
    let port = candidate.ports.iter().find(|port| port.name == port_name)?;
    Some(if use_route_position {
        port.route_position.unwrap_or(port.position)
    } else {
        port.position
    })
}

fn push_placed_candidate(
    placed: &mut Vec<PlacedModule>,
    candidates: &[LayoutCandidate],
    candidate_index: usize,
    x: usize,
    y: usize,
) {
    let candidate = &candidates[candidate_index];
    placed.push(PlacedModule {
        module_name: candidate.module_name.clone(),
        candidate_index,
        origin: Position(x, y, candidate.bbox.min.2),
        bbox: candidate.bbox,
    });
}

#[derive(Default)]
struct RegisterBitSlice {
    clock: Option<usize>,
    next: Option<usize>,
    master: Option<usize>,
    slave: Option<usize>,
}

#[derive(Clone, Copy)]
enum RegisterBitRole {
    Clock,
    Next,
    Master,
    Slave,
}

fn choose_register_bit_candidate(
    candidates: &[LayoutCandidate],
    current: Option<usize>,
    candidate_index: usize,
    role: RegisterBitRole,
) -> Option<usize> {
    let Some(current) = current else {
        return Some(candidate_index);
    };

    (register_bit_candidate_score(&candidates[candidate_index], role)
        > register_bit_candidate_score(&candidates[current], role))
    .then_some(candidate_index)
    .or(Some(current))
}

fn register_bit_candidate_score(candidate: &LayoutCandidate, role: RegisterBitRole) -> usize {
    let preferred_ports: &[&str] = match role {
        RegisterBitRole::Clock => &["clk", "clk_n"],
        RegisterBitRole::Next => &["d"],
        RegisterBitRole::Master => &["d", "en", "q"],
        RegisterBitRole::Slave => &["d", "en", "q"],
    };

    candidate
        .ports
        .iter()
        .filter(|port| preferred_ports.contains(&port.name.as_str()))
        .map(|port| port.route_position.unwrap_or(port.position).2 + port.position.2)
        .sum()
}

fn register_bit_module_role(name: &str) -> Option<(usize, RegisterBitRole)> {
    let (bit_name, role) = if let Some(bit_name) = name.strip_suffix("_clk_inv") {
        (bit_name, RegisterBitRole::Clock)
    } else if let Some(bit_name) = name.strip_suffix("_next") {
        (bit_name, RegisterBitRole::Next)
    } else if let Some(bit_name) = name.strip_suffix("_master") {
        (bit_name, RegisterBitRole::Master)
    } else if let Some(bit_name) = name.strip_suffix("_slave") {
        (bit_name, RegisterBitRole::Slave)
    } else {
        return None;
    };

    let (_, bit) = bit_name.rsplit_once('_')?;
    bit.parse().ok().map(|bit| (bit, role))
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
    let mut options = [2, 4, square, square.saturating_add(1)]
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
        let source_position = translate_candidate_position(
            source_port.route_position.unwrap_or(source_port.position),
            source_candidate,
            source,
        );
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
