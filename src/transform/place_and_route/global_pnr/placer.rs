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
            if let Some(placed) = place_register_bit_carry_chain(candidates, &config) {
                push_unique_placement(&mut placements, placed);
            }
            if let Some(placed) = place_register_bit_carry_aligned_slices(candidates, &config) {
                push_unique_placement(&mut placements, placed);
            }
            if let Some(placed) = place_register_bit_grid(candidates, &config) {
                push_unique_placement(&mut placements, placed);
            }
            if let Some(placed) = place_register_bit_triangles(candidates, &config) {
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

fn place_register_bit_carry_chain(
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
) -> Option<Vec<PlacedModule>> {
    let mut slices = register_bit_slices(candidates)?;
    if slices.is_empty() {
        return None;
    }

    let mut placed = Vec::new();
    let mut cursor_x = GLOBAL_PLACEMENT_MARGIN;
    let base_y = GLOBAL_PLACEMENT_MARGIN;
    let mut previous_slave_q_y = None;

    for (bit, slice) in slices.drain(..) {
        let clock = slice.clock?;
        let next = slice.next?;
        let master = slice.master?;
        let slave = slice.slave?;

        let next_x = cursor_x;
        let clock_x = next_x + candidates[next].bbox.width() + config.spacing;
        let master_x = clock_x + candidates[clock].bbox.width() + config.spacing;
        let slave_x = master_x + candidates[master].bbox.width() + config.spacing;

        let master_y = base_y;
        let master_en_y = translated_port_y(candidates, master, master_y, "en", false)?;
        let clock_y = align_port_y(candidates, clock, "clk_n", true, master_en_y)?;

        let master_q_y = translated_port_y(candidates, master, master_y, "q", true)?;
        let slave_y = align_port_y(candidates, slave, "d", false, master_q_y)?;
        let slave_q_y = translated_port_y(candidates, slave, slave_y, "q", true)?;

        let carry_input = if bit == 0 {
            format!("q_{bit}")
        } else {
            format!("q_{}", bit - 1)
        };
        let next_target_y = previous_slave_q_y.unwrap_or(slave_q_y);
        let next_y = align_port_y(candidates, next, &carry_input, true, next_target_y)?;

        let row = [
            (next, next_x, next_y),
            (clock, clock_x, clock_y),
            (master, master_x, master_y),
            (slave, slave_x, slave_y),
        ];
        let row_top = row
            .iter()
            .map(|(index, _, y)| y.saturating_sub(candidates[*index].bbox.min.1))
            .min()?;
        let row_shift = GLOBAL_PLACEMENT_MARGIN.saturating_sub(row_top);

        for (candidate_index, x, y) in row {
            push_placed_candidate(&mut placed, candidates, candidate_index, x, y + row_shift);
        }

        previous_slave_q_y = Some(slave_q_y + row_shift);
        cursor_x = slave_x + candidates[slave].bbox.width() + config.spacing;
    }

    Some(placed)
}

fn place_register_bit_carry_aligned_slices(
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
) -> Option<Vec<PlacedModule>> {
    let mut slices = register_bit_slices(candidates)?;
    if slices.is_empty() {
        return None;
    }

    let mut placed = Vec::new();
    let mut cursor_x = GLOBAL_PLACEMENT_MARGIN;
    let base_y = GLOBAL_PLACEMENT_MARGIN;
    let mut previous_slave_q_y = None;

    for (bit, slice) in slices.drain(..) {
        let clock = slice.clock?;
        let next = slice.next?;
        let master = slice.master?;
        let slave = slice.slave?;

        let next_x = cursor_x;
        let clock_x = next_x + candidates[next].bbox.width() + config.spacing;
        let master_x = clock_x + candidates[clock].bbox.width() + config.spacing;
        let slave_x = master_x + candidates[master].bbox.width() + config.spacing;

        let carry_input = if bit == 0 {
            format!("q_{bit}")
        } else {
            format!("q_{}", bit - 1)
        };
        let next_y = if let Some(previous_slave_q_y) = previous_slave_q_y {
            align_port_y(candidates, next, &carry_input, true, previous_slave_q_y)?
        } else {
            base_y
        };

        let next_d_y = translated_port_y(candidates, next, next_y, "d", true)?;
        let master_y = align_port_y(candidates, master, "d", false, next_d_y)?;

        let master_en_y = translated_port_y(candidates, master, master_y, "en", false)?;
        let clock_y = align_port_y(candidates, clock, "clk_n", true, master_en_y)?;

        let master_q_y = translated_port_y(candidates, master, master_y, "q", true)?;
        let slave_y = align_port_y(candidates, slave, "d", false, master_q_y)?;
        let slave_q_y = translated_port_y(candidates, slave, slave_y, "q", true)?;

        let row = [
            (next, next_x, next_y),
            (clock, clock_x, clock_y),
            (master, master_x, master_y),
            (slave, slave_x, slave_y),
        ];
        let row_top = row
            .iter()
            .map(|(index, _, y)| y.saturating_sub(candidates[*index].bbox.min.1))
            .min()?;
        let row_shift = GLOBAL_PLACEMENT_MARGIN.saturating_sub(row_top);

        for (candidate_index, x, y) in row {
            push_placed_candidate(&mut placed, candidates, candidate_index, x, y + row_shift);
        }

        previous_slave_q_y = Some(slave_q_y + row_shift);
        cursor_x = slave_x + candidates[slave].bbox.width() + config.spacing;
    }

    Some(placed)
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
            RegisterBitRole::Clock => slice.clock = Some(index),
            RegisterBitRole::Next => slice.next = Some(index),
            RegisterBitRole::Master => slice.master = Some(index),
            RegisterBitRole::Slave => slice.slave = Some(index),
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
        for candidate_index in [slice.next?, slice.clock?, slice.master?, slice.slave?] {
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
            RegisterBitRole::Clock => slice.clock = Some(index),
            RegisterBitRole::Next => slice.next = Some(index),
            RegisterBitRole::Master => slice.master = Some(index),
            RegisterBitRole::Slave => slice.slave = Some(index),
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

    let next_width = bits
        .iter()
        .map(|(_, slice)| candidates[slice.next.unwrap()].bbox.width())
        .max()
        .unwrap_or(0);
    let master_width = bits
        .iter()
        .map(|(_, slice)| candidates[slice.master.unwrap()].bbox.width())
        .max()
        .unwrap_or(0);
    let next_x = GLOBAL_PLACEMENT_MARGIN;
    let master_x = next_x + next_width + config.spacing;
    let slave_x = master_x + master_width + config.spacing;
    let clock_x = master_x;

    let mut placed = Vec::new();
    let mut cursor_y = GLOBAL_PLACEMENT_MARGIN;
    for (_, slice) in bits {
        let clock = slice.clock?;
        let next = slice.next?;
        let master = slice.master?;
        let slave = slice.slave?;

        let next_y = cursor_y;
        let next_d_y = translated_port_y(candidates, next, next_y, "d", true)?;
        let master_y = align_port_y(candidates, master, "d", false, next_d_y)?;

        let master_q_y = translated_port_y(candidates, master, master_y, "q", true)?;
        let slave_y = align_port_y(candidates, slave, "d", false, master_q_y)?;

        let clock_y = master_y + candidates[master].bbox.depth() + config.spacing;

        let row = [
            (next, next_x, next_y),
            (master, master_x, master_y),
            (slave, slave_x, slave_y),
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
            (master, master_y + row_shift),
            (clock, clock_y + row_shift),
            (slave, slave_y + row_shift),
        ]
        .into_iter()
        .map(|(index, y)| y + candidates[index].bbox.depth())
        .max()
        .unwrap_or(cursor_y);
        cursor_y = row_bottom + config.spacing;
    }

    Some(placed)
}

fn place_register_bit_triangles(
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
            RegisterBitRole::Clock => slice.clock = Some(index),
            RegisterBitRole::Next => slice.next = Some(index),
            RegisterBitRole::Master => slice.master = Some(index),
            RegisterBitRole::Slave => slice.slave = Some(index),
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
    let master_x = GLOBAL_PLACEMENT_MARGIN;
    let slave_x = master_x + master_width + config.spacing;
    let clock_x = master_x;

    let mut placed = Vec::new();
    let mut cursor_y = GLOBAL_PLACEMENT_MARGIN;
    for (_, slice) in bits {
        let clock = slice.clock?;
        let next = slice.next?;
        let master = slice.master?;
        let slave = slice.slave?;

        let bit_width = slave_x + candidates[slave].bbox.width() - master_x;
        let next_x = master_x + bit_width.saturating_sub(candidates[next].bbox.width()) / 2;
        let next_y = cursor_y;
        let master_y = next_y + candidates[next].bbox.depth() + config.spacing;

        let master_q_y = translated_port_y(candidates, master, master_y, "q", true)?;
        let slave_y = align_port_y(candidates, slave, "d", false, master_q_y)?;

        let clock_y = master_y + candidates[master].bbox.depth() + config.spacing;

        let row = [
            (next, next_x, next_y),
            (master, master_x, master_y),
            (slave, slave_x, slave_y),
            (clock, clock_x, clock_y),
        ];
        let Some(row_top) = row
            .iter()
            .map(|(index, _, y)| y.saturating_sub(candidates[*index].bbox.min.1))
            .min()
        else {
            return None;
        };
        let row_shift = cursor_y.saturating_sub(row_top);

        for (candidate_index, x, y) in row {
            push_placed_candidate(&mut placed, candidates, candidate_index, x, y + row_shift);
        }

        let row_bottom = [
            (next, next_y + row_shift),
            (master, master_y + row_shift),
            (slave, slave_y + row_shift),
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

fn candidate_port_position(
    candidate: &LayoutCandidate,
    port_name: &str,
    use_route_position: bool,
) -> Option<Position> {
    let port = candidate.ports.iter().find(|port| port.name == port_name)?;
    Some(if use_route_position {
        port.primary_route_position()
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

fn register_bit_slices(candidates: &[LayoutCandidate]) -> Option<Vec<(usize, RegisterBitSlice)>> {
    let mut slices = HashMap::<usize, RegisterBitSlice>::new();

    for (index, candidate) in candidates.iter().enumerate() {
        let Some((bit, role)) = register_bit_module_role(&candidate.module_name) else {
            return None;
        };
        let slice = slices.entry(bit).or_default();
        match role {
            RegisterBitRole::Clock => slice.clock = Some(index),
            RegisterBitRole::Next => slice.next = Some(index),
            RegisterBitRole::Master => slice.master = Some(index),
            RegisterBitRole::Slave => slice.slave = Some(index),
        }
    }

    let mut slices = slices.into_iter().collect::<Vec<_>>();
    slices.sort_by_key(|(bit, _)| *bit);
    if slices.iter().any(|(_, slice)| {
        slice.clock.is_none()
            || slice.next.is_none()
            || slice.master.is_none()
            || slice.slave.is_none()
    }) {
        return None;
    }
    Some(slices)
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
        let source_positions = source_port
            .routing_access_positions()
            .into_iter()
            .map(|position| translate_candidate_position(position, source_candidate, source));
        let target_position = translate_candidate_position(
            target_port.primary_route_position(),
            target_candidate,
            target,
        );
        let Some(source_position) =
            source_positions.min_by_key(|position| position.manhattan_distance(&target_position))
        else {
            continue;
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::place_and_route::global_pnr::ir::{
        LayoutCandidateCost, PhysicalPort, PhysicalPortDirection, PortConnection,
    };
    use crate::world::position::DimSize;
    use crate::world::World3D;

    fn test_candidate(module_name: &str, ports: &[(&str, Position)]) -> LayoutCandidate {
        LayoutCandidate {
            module_name: module_name.to_owned(),
            world: World3D::new(DimSize(1, 1, 1)),
            bbox: BoundingBox {
                min: Position(0, 0, 0),
                max: Position(5, 12, 2),
            },
            ports: ports
                .iter()
                .map(|(name, position)| PhysicalPort {
                    name: (*name).to_owned(),
                    direction: PhysicalPortDirection::Output,
                    position: *position,
                    route_position: Some(*position),
                    access_points: vec![*position],
                    connection: PortConnection::Direct,
                })
                .collect(),
            occupied_cells: HashSet::new(),
            blocked_cells: HashSet::new(),
            cost: LayoutCandidateCost::default(),
        }
    }

    fn placed_by_name<'a>(placed: &'a [PlacedModule], name: &str) -> &'a PlacedModule {
        placed
            .iter()
            .find(|placed| placed.module_name == name)
            .expect("placed module")
    }

    fn placed_port_y(
        candidates: &[LayoutCandidate],
        placed: &[PlacedModule],
        module_name: &str,
        port_name: &str,
    ) -> usize {
        let placed_module = placed_by_name(placed, module_name);
        translated_port_y(
            candidates,
            placed_module.candidate_index,
            placed_module.origin.1,
            port_name,
            true,
        )
        .expect("port y")
    }

    #[test]
    fn carry_aligned_slices_align_next_data_and_cross_bit_carry_ports() {
        let candidates = vec![
            test_candidate("q_0_clk_inv", &[("clk_n", Position(1, 3, 1))]),
            test_candidate(
                "q_0_next",
                &[("q_0", Position(0, 2, 1)), ("d", Position(5, 6, 1))],
            ),
            test_candidate(
                "q_0_master",
                &[
                    ("d", Position(0, 6, 1)),
                    ("en", Position(0, 3, 1)),
                    ("q", Position(5, 8, 1)),
                ],
            ),
            test_candidate(
                "q_0_slave",
                &[("d", Position(0, 8, 1)), ("q", Position(5, 10, 1))],
            ),
            test_candidate("q_1_clk_inv", &[("clk_n", Position(1, 3, 1))]),
            test_candidate(
                "q_1_next",
                &[
                    ("q_0", Position(0, 4, 1)),
                    ("q_1", Position(0, 2, 1)),
                    ("d", Position(5, 7, 1)),
                ],
            ),
            test_candidate(
                "q_1_master",
                &[
                    ("d", Position(0, 7, 1)),
                    ("en", Position(0, 3, 1)),
                    ("q", Position(5, 8, 1)),
                ],
            ),
            test_candidate(
                "q_1_slave",
                &[("d", Position(0, 8, 1)), ("q", Position(5, 10, 1))],
            ),
        ];

        let placed = place_register_bit_carry_aligned_slices(
            &candidates,
            &GlobalPlacementConfig {
                spacing: 4,
                ..Default::default()
            },
        )
        .expect("placement");

        assert_eq!(
            placed_port_y(&candidates, &placed, "q_1_next", "q_0"),
            placed_port_y(&candidates, &placed, "q_0_slave", "q")
        );
        assert_eq!(
            placed_port_y(&candidates, &placed, "q_1_next", "d"),
            placed_port_y(&candidates, &placed, "q_1_master", "d")
        );
    }
}
