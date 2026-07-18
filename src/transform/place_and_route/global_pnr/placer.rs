use std::collections::{HashMap, HashSet};

use eyre::{ContextCompat, WrapErr};

use crate::graph::module::{GraphModule, GraphModulePortTarget};
use crate::transform::place_and_route::estimate::BoundingBox;
use crate::transform::place_and_route::global_pnr::free_3d::{
    place_free_3d, place_free_3d_with_edges,
};
use crate::transform::place_and_route::global_pnr::heuristics::{
    GlobalHeuristicHooks, PlacementHeuristicContext,
};
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::physical_intent::{
    ResolvedPhysicalConstraint, ResolvedPhysicalIntent,
};
use crate::transform::place_and_route::global_pnr::policy::{
    LayerAssignmentStrategy, LayeredPlacementConfig, PlacementCostBreakdown, PlacementCostWeights,
    PlacementHeuristic, RoutingCongestionConfig,
};
use crate::transform::place_and_route::global_pnr::topology::{
    ResolvedEndpoint, ResolvedPnrTopology,
};
use crate::world::position::Position;

const GLOBAL_PLACEMENT_MARGIN: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalPlacementConfig {
    pub spacing: usize,
    pub shelf_width: usize,
    pub max_attempts: usize,
    pub cost_weights: PlacementCostWeights,
    pub congestion: RoutingCongestionConfig,
}

impl Default for GlobalPlacementConfig {
    fn default() -> Self {
        Self {
            spacing: 2,
            shelf_width: 64,
            max_attempts: 16,
            cost_weights: PlacementCostWeights::default(),
            congestion: RoutingCongestionConfig::default(),
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

#[derive(Clone, Debug)]
struct PlacementConnection {
    source_instance: String,
    source_port: String,
    target_instance: String,
    target_port: String,
}

#[derive(Clone, Debug, Default)]
struct PlacementConnectivity {
    connections: Vec<PlacementConnection>,
    proximity_edges: Vec<(String, String)>,
}

impl PlacementConnectivity {
    fn from_legacy(module: &GraphModule) -> Self {
        let connections = module
            .vars
            .iter()
            .map(|var| PlacementConnection {
                source_instance: var.source.0.clone(),
                source_port: var.source.1.clone(),
                target_instance: var.target.0.clone(),
                target_port: var.target.1.clone(),
            })
            .collect::<Vec<_>>();
        let mut proximity_edges = connections
            .iter()
            .map(|connection| {
                (
                    connection.source_instance.clone(),
                    connection.target_instance.clone(),
                )
            })
            .collect::<Vec<_>>();
        for port in &module.ports {
            let targets = target_modules(&port.target);
            for (index, left) in targets.iter().enumerate() {
                for right in targets.iter().skip(index + 1) {
                    proximity_edges.push((left.clone(), right.clone()));
                }
            }
        }
        Self {
            connections,
            proximity_edges,
        }
    }

    fn from_resolved(topology: &ResolvedPnrTopology) -> eyre::Result<Self> {
        let mut connections = Vec::new();
        let mut proximity_edges = Vec::new();
        for net in &topology.nets {
            let endpoints = std::iter::once(&net.driver)
                .chain(net.sinks.iter())
                .filter_map(|endpoint| match endpoint {
                    ResolvedEndpoint::InstancePort { instance, port } => Some((*instance, *port)),
                    ResolvedEndpoint::TopPort { .. } => None,
                })
                .collect::<Vec<_>>();
            for (index, (left, _)) in endpoints.iter().enumerate() {
                for (right, _) in endpoints.iter().skip(index + 1) {
                    let left = topology
                        .instances
                        .get(left.0)
                        .context("resolved placement edge has an unknown instance")?;
                    let right = topology
                        .instances
                        .get(right.0)
                        .context("resolved placement edge has an unknown instance")?;
                    if left.id != right.id {
                        proximity_edges
                            .push((left.display_name.clone(), right.display_name.clone()));
                    }
                }
            }
            let ResolvedEndpoint::InstancePort {
                instance: source_instance,
                port: source_port,
            } = net.driver
            else {
                continue;
            };
            let source_instance = topology
                .instances
                .get(source_instance.0)
                .context("resolved placement net has an unknown driver instance")?;
            let source_port = topology
                .port(source_port)
                .context("resolved placement net has an unknown driver port")?;
            for sink in &net.sinks {
                let ResolvedEndpoint::InstancePort { instance, port } = sink else {
                    continue;
                };
                let target_instance = topology
                    .instances
                    .get(instance.0)
                    .context("resolved placement net has an unknown sink instance")?;
                let target_port = topology
                    .port(*port)
                    .context("resolved placement net has an unknown sink port")?;
                connections.push(PlacementConnection {
                    source_instance: source_instance.display_name.clone(),
                    source_port: source_port.name.clone(),
                    target_instance: target_instance.display_name.clone(),
                    target_port: target_port.name.clone(),
                });
            }
        }
        proximity_edges.sort();
        proximity_edges.dedup();
        Ok(Self {
            connections,
            proximity_edges,
        })
    }

    fn candidate_edges(&self, candidates: &[LayoutCandidate]) -> Vec<(usize, usize)> {
        let by_name = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.module_name.as_str(), index))
            .collect::<HashMap<_, _>>();
        self.proximity_edges
            .iter()
            .filter_map(|(left, right)| {
                let left = *by_name.get(left.as_str())?;
                let right = *by_name.get(right.as_str())?;
                (left != right).then_some((left, right))
            })
            .collect()
    }
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
    heuristics: &[PlacementHeuristic],
) -> Vec<Vec<PlacedModule>> {
    let connectivity = PlacementConnectivity::from_legacy(module);
    placement_candidates_with_connectivity(
        &connectivity,
        Some(module),
        candidates,
        config,
        heuristics,
    )
}

fn placement_candidates_with_connectivity(
    connectivity: &PlacementConnectivity,
    legacy_module: Option<&GraphModule>,
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
    heuristics: &[PlacementHeuristic],
) -> Vec<Vec<PlacedModule>> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut placements = Vec::new();
    for heuristic in heuristics {
        if let PlacementHeuristic::Free3D(free_3d) = heuristic {
            let edges = connectivity.candidate_edges(candidates);
            let placed = legacy_module.map_or_else(
                || place_free_3d_with_edges(candidates, &edges, *free_3d),
                |module| place_free_3d(module, candidates, *free_3d),
            );
            if let Some(placed) = placed {
                push_unique_placement(&mut placements, placed);
            }
        }
    }
    let original_order = (0..candidates.len()).collect::<Vec<_>>();
    let net_order = net_aware_candidate_order(connectivity, candidates);

    if is_register_bit_module_set(candidates) {
        let layered_configs = heuristics
            .iter()
            .filter_map(|heuristic| match heuristic {
                PlacementHeuristic::Layered3D(config) => Some(*config),
                _ => None,
            })
            .collect::<Vec<_>>();
        let has_register_seed = heuristics.iter().any(|heuristic| {
            matches!(
                heuristic,
                PlacementHeuristic::RegisterCarryChain
                    | PlacementHeuristic::RegisterCarryAlignedSlices
                    | PlacementHeuristic::RegisterGrid
                    | PlacementHeuristic::RegisterTriangles
                    | PlacementHeuristic::RegisterSlices
            )
        });
        for spacing in placement_spacing_options(config.spacing) {
            let config = GlobalPlacementConfig { spacing, ..*config };
            for heuristic in heuristics {
                let placed = match heuristic {
                    PlacementHeuristic::RegisterCarryChain => {
                        place_register_bit_carry_chain(candidates, &config)
                    }
                    PlacementHeuristic::RegisterCarryAlignedSlices => {
                        place_register_bit_carry_aligned_slices(candidates, &config)
                    }
                    PlacementHeuristic::RegisterGrid => {
                        place_register_bit_grid(candidates, &config)
                    }
                    PlacementHeuristic::RegisterTriangles => {
                        place_register_bit_triangles(candidates, &config)
                    }
                    PlacementHeuristic::RegisterSlices => {
                        place_register_bit_slices(candidates, &config)
                    }
                    PlacementHeuristic::Shelf
                    | PlacementHeuristic::Grid
                    | PlacementHeuristic::Layered3D(_)
                    | PlacementHeuristic::Free3D(_) => None,
                };
                if let Some(placed) = placed {
                    push_unique_placement(&mut placements, placed.clone());
                    for layered in &layered_configs {
                        push_unique_placement(
                            &mut placements,
                            apply_layered_placement(
                                connectivity,
                                candidates,
                                placed.clone(),
                                *layered,
                            ),
                        );
                    }
                }
            }
            if !has_register_seed {
                let seeds = [
                    place_register_bit_carry_chain(candidates, &config),
                    place_register_bit_carry_aligned_slices(candidates, &config),
                    place_register_bit_grid(candidates, &config),
                    place_register_bit_triangles(candidates, &config),
                    place_register_bit_slices(candidates, &config),
                ];
                for placed in seeds.into_iter().flatten() {
                    for layered in &layered_configs {
                        push_unique_placement(
                            &mut placements,
                            apply_layered_placement(
                                connectivity,
                                candidates,
                                placed.clone(),
                                *layered,
                            ),
                        );
                    }
                }
            }
        }
        placements.sort_by_key(|placed| {
            placement_cost_breakdown_connectivity(
                connectivity,
                candidates,
                placed,
                config.congestion,
            )
            .weighted_total(config.cost_weights)
        });
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
            if heuristics.contains(&PlacementHeuristic::Shelf) {
                push_unique_placement(
                    &mut placements,
                    place_candidates_on_shelves_in_order(candidates, &original_order, &config),
                );
                push_unique_placement(
                    &mut placements,
                    place_candidates_on_shelves_in_order(candidates, &net_order, &config),
                );
            }
            if heuristics.contains(&PlacementHeuristic::Grid) {
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
            for layered in heuristics.iter().filter_map(|heuristic| match heuristic {
                PlacementHeuristic::Layered3D(config) => Some(*config),
                _ => None,
            }) {
                let seed = place_candidates_on_shelves_in_order(candidates, &net_order, &config);
                push_unique_placement(
                    &mut placements,
                    apply_layered_placement(connectivity, candidates, seed, layered),
                );
            }
        }
    }

    placements.sort_by_key(|placed| {
        placement_cost_breakdown_connectivity(connectivity, candidates, placed, config.congestion)
            .weighted_total(config.cost_weights)
    });
    placements.truncate(config.max_attempts.max(1));
    placements
}

pub fn placement_candidates_resolved(
    topology: &ResolvedPnrTopology,
    intent: Option<&ResolvedPhysicalIntent>,
    hooks: &GlobalHeuristicHooks,
    candidates: &[LayoutCandidate],
    config: &GlobalPlacementConfig,
    heuristics: &[PlacementHeuristic],
) -> eyre::Result<Vec<Vec<PlacedModule>>> {
    let connectivity = PlacementConnectivity::from_resolved(topology)?;
    let mut placements =
        placement_candidates_with_connectivity(&connectivity, None, candidates, config, heuristics);
    let context = PlacementHeuristicContext {
        topology,
        candidates,
        intent,
    };
    for hook in &hooks.placement_transforms {
        (hook.apply)(&context, &mut placements)
            .with_context(|| format!("placement transform hook `{}` failed", hook.name))?;
    }

    let mut constrained = Vec::new();
    let mut last_error = None;
    for mut placed in placements {
        let result = if let Some(intent) = intent {
            apply_placement_intent(topology, intent, candidates, &mut placed)
        } else {
            validate_no_placement_overlap(&placed)
        };
        match result {
            Ok(()) => push_unique_placement(&mut constrained, placed),
            Err(error) => last_error = Some(error),
        }
    }
    if constrained.is_empty() {
        return Err(last_error.unwrap_or_else(|| {
            eyre::eyre!("physical placement constraints eliminated every placement attempt")
        }));
    }
    constrained.sort_by_key(|placed| {
        let preference = intent.map_or(Default::default(), |intent| {
            intent.placement_preference_cost(topology, candidates, placed)
        });
        let hook_cost = hooks
            .placement_cost_terms
            .iter()
            .map(|hook| (hook.evaluate)(&context, placed))
            .sum::<usize>();
        (
            preference.strong,
            preference.medium,
            preference.weak,
            hook_cost,
            placement_cost_breakdown_connectivity(
                &connectivity,
                candidates,
                placed,
                config.congestion,
            )
            .weighted_total(config.cost_weights),
        )
    });
    constrained.truncate(config.max_attempts.max(1));
    Ok(constrained)
}

fn apply_placement_intent(
    topology: &ResolvedPnrTopology,
    intent: &ResolvedPhysicalIntent,
    candidates: &[LayoutCandidate],
    placed: &mut [PlacedModule],
) -> eyre::Result<()> {
    let fixed_instances = intent
        .constraints
        .iter()
        .filter_map(|constraint| match constraint {
            ResolvedPhysicalConstraint::FixedOrigin { instance, .. } => Some(*instance),
            _ => None,
        })
        .collect::<HashSet<_>>();

    // Exact locks are applied first. Other hard constraints may move only
    // unlocked instances and must validate locked origins as written.
    for constraint in &intent.constraints {
        let ResolvedPhysicalConstraint::FixedOrigin {
            id,
            instance,
            origin,
        } = constraint
        else {
            continue;
        };
        let item = placed_instance_mut(topology, placed, *instance, id)?;
        item.origin = Position(origin[0], origin[1], origin[2]);
    }

    for constraint in &intent.constraints {
        let (instance_id, id) = match constraint {
            ResolvedPhysicalConstraint::Inside { instance, id, .. }
            | ResolvedPhysicalConstraint::LayerRange { instance, id, .. } => (*instance, id),
            _ => continue,
        };
        let item = placed_instance_mut(topology, placed, instance_id, id)?;
        let candidate = candidates
            .get(item.candidate_index)
            .with_context(|| format!("constraint `{id}` references a missing candidate"))?;
        let size = [
            candidate.bbox.width(),
            candidate.bbox.depth(),
            candidate.bbox.height(),
        ];
        let locked = fixed_instances.contains(&instance_id);

        match constraint {
            ResolvedPhysicalConstraint::Inside { region, .. } => {
                let region = intent
                    .regions
                    .get(region)
                    .with_context(|| format!("constraint `{id}` has an unknown region"))?;
                let max_origin = [
                    maximum_origin(region.max[0], size[0], id)?,
                    maximum_origin(region.max[1], size[1], id)?,
                    maximum_origin(region.max[2], size[2], id)?,
                ];
                if !locked {
                    item.origin.0 = item.origin.0.clamp(region.min[0], max_origin[0]);
                    item.origin.1 = item.origin.1.clamp(region.min[1], max_origin[1]);
                    item.origin.2 = item.origin.2.clamp(region.min[2], max_origin[2]);
                }
            }
            ResolvedPhysicalConstraint::LayerRange { min, max, .. } => {
                let max_origin = maximum_origin(*max, size[2], id)?;
                if !locked {
                    item.origin.2 = item.origin.2.clamp(*min, max_origin);
                }
            }
            _ => unreachable!(),
        }
    }

    validate_placement_intent(topology, intent, candidates, placed)?;
    validate_no_placement_overlap(placed)
}

fn placed_instance_mut<'a>(
    topology: &ResolvedPnrTopology,
    placed: &'a mut [PlacedModule],
    instance: crate::transform::place_and_route::global_pnr::topology::InstanceId,
    constraint_id: &str,
) -> eyre::Result<&'a mut PlacedModule> {
    let instance = topology
        .instances
        .get(instance.0)
        .with_context(|| format!("constraint `{constraint_id}` has an unknown instance id"))?;
    placed
        .iter_mut()
        .find(|placed| placed.module_name == instance.display_name)
        .with_context(|| {
            format!(
                "constraint `{constraint_id}` targets unplaced instance `{}`",
                instance.display_name
            )
        })
}

fn maximum_origin(region_max: usize, size: usize, id: &str) -> eyre::Result<usize> {
    region_max
        .checked_add(1)
        .and_then(|extent| extent.checked_sub(size))
        .with_context(|| format!("constraint `{id}` region is smaller than the selected candidate"))
}

fn validate_placement_intent(
    topology: &ResolvedPnrTopology,
    intent: &ResolvedPhysicalIntent,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) -> eyre::Result<()> {
    for constraint in &intent.constraints {
        let (instance_id, id) = match constraint {
            ResolvedPhysicalConstraint::Inside { instance, id, .. }
            | ResolvedPhysicalConstraint::LayerRange { instance, id, .. }
            | ResolvedPhysicalConstraint::FixedOrigin { instance, id, .. } => (*instance, id),
            _ => continue,
        };
        let instance = &topology.instances[instance_id.0];
        let item = placed
            .iter()
            .find(|placed| placed.module_name == instance.display_name)
            .with_context(|| format!("constraint `{id}` targets an unplaced instance"))?;
        let candidate = &candidates[item.candidate_index];
        let min = [item.origin.0, item.origin.1, item.origin.2];
        let max = [
            item.origin.0 + candidate.bbox.width() - 1,
            item.origin.1 + candidate.bbox.depth() - 1,
            item.origin.2 + candidate.bbox.height() - 1,
        ];
        let satisfied = match constraint {
            ResolvedPhysicalConstraint::Inside { region, .. } => {
                intent.regions[region].contains_box(min, max)
            }
            ResolvedPhysicalConstraint::LayerRange { min, max, .. } => {
                item.origin.2 >= *min && item.origin.2 + candidate.bbox.height() - 1 <= *max
            }
            ResolvedPhysicalConstraint::FixedOrigin { origin, .. } => min == *origin,
            _ => true,
        };
        if !satisfied {
            eyre::bail!(
                "physical constraint `{id}` is violated by instance `{}` at {:?}",
                instance.display_name,
                item.origin
            );
        }
    }
    Ok(())
}

fn validate_no_placement_overlap(placed: &[PlacedModule]) -> eyre::Result<()> {
    for (index, left) in placed.iter().enumerate() {
        for right in placed.iter().skip(index + 1) {
            let overlaps = left.origin.0 < right.origin.0 + right.bbox.width()
                && right.origin.0 < left.origin.0 + left.bbox.width()
                && left.origin.1 < right.origin.1 + right.bbox.depth()
                && right.origin.1 < left.origin.1 + left.bbox.depth()
                && left.origin.2 < right.origin.2 + right.bbox.height()
                && right.origin.2 < left.origin.2 + left.bbox.height();
            if overlaps {
                eyre::bail!(
                    "physical constraints overlap instances `{}` and `{}`",
                    left.module_name,
                    right.module_name
                );
            }
        }
    }
    Ok(())
}

fn apply_layered_placement(
    connectivity: &PlacementConnectivity,
    candidates: &[LayoutCandidate],
    mut placed: Vec<PlacedModule>,
    config: LayeredPlacementConfig,
) -> Vec<PlacedModule> {
    let layer_count = config.layers.max(1);
    let layer_stride = candidates
        .iter()
        .map(|candidate| candidate.bbox.height())
        .max()
        .unwrap_or(1)
        .saturating_add(config.layer_spacing);
    let assignments = match config.assignment {
        LayerAssignmentStrategy::Alternating => {
            (0..placed.len()).map(|index| index % layer_count).collect()
        }
        LayerAssignmentStrategy::NetAware => {
            net_aware_layer_assignments(connectivity, &placed, layer_count)
        }
    };
    for (placed, layer) in placed.iter_mut().zip(assignments) {
        placed.origin.2 = placed.bbox.min.2 + layer * layer_stride;
    }
    placed
}

fn net_aware_layer_assignments(
    connectivity: &PlacementConnectivity,
    placed: &[PlacedModule],
    layer_count: usize,
) -> Vec<usize> {
    let target_load = placed.len().div_ceil(layer_count.max(1));
    let mut assignments = Vec::with_capacity(placed.len());
    let mut assigned_by_module = HashMap::<&str, usize>::new();
    let mut layer_loads = vec![0usize; layer_count.max(1)];

    for item in placed {
        let connected_layers = connectivity
            .proximity_edges
            .iter()
            .filter_map(|(left, right)| {
                if left == &item.module_name {
                    assigned_by_module.get(right.as_str()).copied()
                } else if right == &item.module_name {
                    assigned_by_module.get(left.as_str()).copied()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let layer = (0..layer_loads.len())
            .min_by_key(|&layer| {
                let cross_layer_edges = connected_layers
                    .iter()
                    .filter(|&&connected| connected != layer)
                    .count();
                let overflow = usize::from(layer_loads[layer] >= target_load);
                (overflow, cross_layer_edges, layer_loads[layer], layer)
            })
            .unwrap_or(0);
        assignments.push(layer);
        assigned_by_module.insert(item.module_name.as_str(), layer);
        layer_loads[layer] += 1;
    }
    assignments
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

fn net_aware_candidate_order(
    connectivity: &PlacementConnectivity,
    candidates: &[LayoutCandidate],
) -> Vec<usize> {
    let module_to_candidate = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.module_name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let edges = connectivity
        .proximity_edges
        .iter()
        .filter_map(|(left, right)| {
            let left = *module_to_candidate.get(left.as_str())?;
            let right = *module_to_candidate.get(right.as_str())?;
            (left != right).then_some((left, right))
        })
        .collect::<Vec<_>>();
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

#[cfg(test)]
pub(crate) fn placement_cost_breakdown(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
    congestion_config: RoutingCongestionConfig,
) -> PlacementCostBreakdown {
    placement_cost_breakdown_connectivity(
        &PlacementConnectivity::from_legacy(module),
        candidates,
        placed,
        congestion_config,
    )
}

fn placement_cost_breakdown_connectivity(
    connectivity: &PlacementConnectivity,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
    congestion_config: RoutingCongestionConfig,
) -> PlacementCostBreakdown {
    let placed_by_module = placed
        .iter()
        .map(|placed| (placed.module_name.as_str(), placed))
        .collect::<HashMap<_, _>>();
    let (placement_volume, xy_footprint, height_span) = placement_bbox_metrics(placed);
    let mut cost = PlacementCostBreakdown {
        placement_volume,
        xy_footprint,
        height_span,
        ..PlacementCostBreakdown::default()
    };

    let mut net_regions = Vec::new();
    for connection in &connectivity.connections {
        let Some(source) = placed_by_module.get(connection.source_instance.as_str()) else {
            continue;
        };
        let Some(target) = placed_by_module.get(connection.target_instance.as_str()) else {
            continue;
        };
        let source_candidate = &candidates[source.candidate_index];
        let target_candidate = &candidates[target.candidate_index];
        let Some(source_port) = source_candidate
            .ports
            .iter()
            .find(|port| port.name == connection.source_port)
        else {
            continue;
        };
        let Some(target_port) = target_candidate
            .ports
            .iter()
            .find(|port| port.name == connection.target_port)
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
        cost.estimated_wire_length += source_position.manhattan_distance(&target_position);
        cost.vertical_distance += source_position.2.abs_diff(target_position.2);
        net_regions.push((source_position, target_position));
    }
    cost.routing_congestion = estimate_routing_congestion(&net_regions, congestion_config);

    cost
}

pub(crate) fn placement_cost_breakdown_resolved(
    topology: &ResolvedPnrTopology,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
    congestion_config: RoutingCongestionConfig,
) -> eyre::Result<PlacementCostBreakdown> {
    let connectivity = PlacementConnectivity::from_resolved(topology)?;
    Ok(placement_cost_breakdown_connectivity(
        &connectivity,
        candidates,
        placed,
        congestion_config,
    ))
}

fn estimate_routing_congestion(
    net_regions: &[(Position, Position)],
    config: RoutingCongestionConfig,
) -> usize {
    let bin_xy = config.bin_size_xy.max(1);
    let bin_z = config.bin_size_z.max(1);
    let mut demand = HashMap::<(usize, usize, usize), usize>::new();

    for &(source, target) in net_regions {
        let min = Position(
            source.0.min(target.0) / bin_xy,
            source.1.min(target.1) / bin_xy,
            source.2.min(target.2) / bin_z,
        );
        let max = Position(
            source.0.max(target.0) / bin_xy,
            source.1.max(target.1) / bin_xy,
            source.2.max(target.2) / bin_z,
        );
        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                for z in min.2..=max.2 {
                    *demand.entry((x, y, z)).or_default() += 1;
                }
            }
        }
    }

    demand.values().fold(0usize, |total, &count| {
        total.saturating_add(count.saturating_mul(count.saturating_sub(1)) / 2)
    })
}

fn placement_bbox_metrics(placed: &[PlacedModule]) -> (usize, usize, usize) {
    let Some(first) = placed.first() else {
        return (0, 0, 0);
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
    let width = max.0 - min.0 + 1;
    let depth = max.1 - min.1 + 1;
    let height = max.2 - min.2 + 1;
    (width * depth * height, width * depth, height)
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
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::transform::place_and_route::global_pnr::heuristics::PlacementTransformHook;
    use crate::transform::place_and_route::global_pnr::ir::{
        LayoutCandidateCost, PhysicalPort, PhysicalPortDirection, PortConnection,
    };
    use crate::transform::place_and_route::global_pnr::physical_intent::{
        IntentRegion, ResolvedPhysicalConstraint, ResolvedPhysicalIntent,
    };
    use crate::transform::place_and_route::global_pnr::policy::{
        Free3DPlacementConfig, LayerAssignmentStrategy, LayeredPlacementConfig, PlacementHeuristic,
        RoutingCongestionConfig,
    };
    use crate::transform::place_and_route::global_pnr::topology::{
        DefinitionId, DefinitionKey, InstanceId, InstanceKey, ResolvedDefinition,
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

    static PLACEMENT_TRANSFORM_CALLED: AtomicBool = AtomicBool::new(false);

    fn observe_placement_transform(
        _context: &PlacementHeuristicContext<'_>,
        _placements: &mut Vec<Vec<PlacedModule>>,
    ) -> eyre::Result<()> {
        PLACEMENT_TRANSFORM_CALLED.store(true, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn resolved_placement_applies_fixed_origin_and_region_constraints() -> eyre::Result<()> {
        PLACEMENT_TRANSFORM_CALLED.store(false, Ordering::SeqCst);
        let candidates = vec![test_candidate("child", &[])];
        let topology = ResolvedPnrTopology {
            top: DefinitionId(0),
            definitions: vec![
                ResolvedDefinition {
                    id: DefinitionId(0),
                    key: DefinitionKey("top".to_owned()),
                    display_name: "top".to_owned(),
                    ports: Vec::new(),
                    is_leaf: false,
                },
                ResolvedDefinition {
                    id: DefinitionId(1),
                    key: DefinitionKey("child".to_owned()),
                    display_name: "child".to_owned(),
                    ports: Vec::new(),
                    is_leaf: true,
                },
            ],
            ports: Vec::new(),
            instances: vec![
                crate::transform::place_and_route::global_pnr::topology::ResolvedInstance {
                    id: InstanceId(0),
                    key: InstanceKey("top/child".to_owned()),
                    display_name: "child".to_owned(),
                    definition: DefinitionId(1),
                },
            ],
            nets: Vec::new(),
        };
        let intent = ResolvedPhysicalIntent {
            format: "test".to_owned(),
            design: "top".to_owned(),
            regions: [(
                "logic".to_owned(),
                IntentRegion {
                    min: [10, 10, 2],
                    max: [30, 30, 8],
                },
            )]
            .into_iter()
            .collect(),
            constraints: vec![
                ResolvedPhysicalConstraint::Inside {
                    id: "inside".to_owned(),
                    instance: InstanceId(0),
                    region: "logic".to_owned(),
                },
                ResolvedPhysicalConstraint::LayerRange {
                    id: "layers".to_owned(),
                    instance: InstanceId(0),
                    min: 2,
                    max: 6,
                },
                ResolvedPhysicalConstraint::FixedOrigin {
                    id: "lock".to_owned(),
                    instance: InstanceId(0),
                    origin: [12, 14, 3],
                },
            ],
        };

        let hooks = GlobalHeuristicHooks {
            placement_transforms: vec![PlacementTransformHook {
                name: "test-observer",
                apply: observe_placement_transform,
            }],
            ..Default::default()
        };
        let placements = placement_candidates_resolved(
            &topology,
            Some(&intent),
            &hooks,
            &candidates,
            &GlobalPlacementConfig::default(),
            &[PlacementHeuristic::Shelf],
        )?;

        assert!(!placements.is_empty());
        assert!(PLACEMENT_TRANSFORM_CALLED.load(Ordering::SeqCst));
        assert_eq!(placements[0][0].origin, Position(12, 14, 3));
        Ok(())
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

    #[test]
    fn placement_policy_can_disable_grid_attempts() {
        let candidates = (0..5)
            .map(|index| test_candidate(&format!("child_{index}"), &[]))
            .collect::<Vec<_>>();
        let module = GraphModule::default();
        let config = GlobalPlacementConfig {
            max_attempts: 128,
            ..Default::default()
        };

        let shelf_only =
            placement_candidates(&module, &candidates, &config, &[PlacementHeuristic::Shelf]);
        let shelf_and_grid = placement_candidates(
            &module,
            &candidates,
            &config,
            &[PlacementHeuristic::Shelf, PlacementHeuristic::Grid],
        );

        assert!(!shelf_only.is_empty());
        assert!(shelf_only.len() < shelf_and_grid.len());
    }

    #[test]
    fn layered_placement_heuristic_assigns_modules_to_multiple_z_layers() {
        let candidates = (0..4)
            .map(|index| test_candidate(&format!("child_{index}"), &[]))
            .collect::<Vec<_>>();
        let placements = placement_candidates(
            &GraphModule::default(),
            &candidates,
            &GlobalPlacementConfig::default(),
            &[PlacementHeuristic::Layered3D(LayeredPlacementConfig {
                layers: 2,
                layer_spacing: 4,
                assignment: LayerAssignmentStrategy::Alternating,
            })],
        );

        assert!(!placements.is_empty());
        let z_origins = placements[0]
            .iter()
            .map(|placed| placed.origin.2)
            .collect::<HashSet<_>>();
        assert_eq!(z_origins.len(), 2);
        assert!(z_origins.iter().copied().max().unwrap() >= 7);
    }

    #[test]
    fn layered_placement_can_wrap_register_specific_heuristics() {
        let candidates = vec![
            test_candidate("q_0_clk_inv", &[("clk_n", Position(1, 3, 1))]),
            test_candidate("q_0_next", &[("q_0", Position(0, 2, 1))]),
            test_candidate("q_0_master", &[("q", Position(5, 8, 1))]),
            test_candidate("q_0_slave", &[("q", Position(5, 10, 1))]),
        ];
        let placements = placement_candidates(
            &GraphModule::default(),
            &candidates,
            &GlobalPlacementConfig::default(),
            &[
                PlacementHeuristic::RegisterSlices,
                PlacementHeuristic::Layered3D(LayeredPlacementConfig {
                    layers: 2,
                    layer_spacing: 4,
                    assignment: LayerAssignmentStrategy::Alternating,
                }),
            ],
        );

        assert!(placements.iter().any(|placed| {
            placed
                .iter()
                .map(|module| module.origin.2)
                .collect::<HashSet<_>>()
                .len()
                > 1
        }));
    }

    #[test]
    fn layered_placement_can_run_alone_for_register_modules() {
        let candidates = vec![
            test_candidate("q_0_clk_inv", &[("clk_n", Position(1, 3, 1))]),
            test_candidate("q_0_next", &[("q_0", Position(0, 2, 1))]),
            test_candidate("q_0_master", &[("q", Position(5, 8, 1))]),
            test_candidate("q_0_slave", &[("q", Position(5, 10, 1))]),
        ];
        let placements = placement_candidates(
            &GraphModule::default(),
            &candidates,
            &GlobalPlacementConfig::default(),
            &[PlacementHeuristic::Layered3D(LayeredPlacementConfig {
                layers: 2,
                layer_spacing: 4,
                assignment: LayerAssignmentStrategy::Alternating,
            })],
        );

        assert!(!placements.is_empty());
        assert!(placements.iter().all(|placed| {
            placed
                .iter()
                .map(|module| module.origin.2)
                .collect::<HashSet<_>>()
                .len()
                > 1
        }));
    }

    #[test]
    fn register_placements_rank_free_3d_with_shared_cost_weights() {
        let candidates = vec![
            test_candidate("q_0_clk_inv", &[("clk_n", Position(1, 3, 1))]),
            test_candidate("q_0_next", &[("q_0", Position(0, 2, 1))]),
            test_candidate("q_0_master", &[("q", Position(5, 8, 1))]),
            test_candidate("q_0_slave", &[("q", Position(5, 10, 1))]),
            test_candidate("q_1_clk_inv", &[("clk_n", Position(1, 3, 1))]),
            test_candidate("q_1_next", &[("q_1", Position(0, 2, 1))]),
            test_candidate("q_1_master", &[("q", Position(5, 8, 1))]),
            test_candidate("q_1_slave", &[("q", Position(5, 10, 1))]),
        ];
        let module = GraphModule::default();
        let config = GlobalPlacementConfig {
            cost_weights: PlacementCostWeights {
                placement_volume: 0,
                xy_footprint: 0,
                height_span: 1,
                estimated_wire_length: 0,
                vertical_distance: 0,
                routing_congestion: 0,
            },
            ..Default::default()
        };
        let placements = placement_candidates(
            &module,
            &candidates,
            &config,
            &[
                PlacementHeuristic::RegisterSlices,
                PlacementHeuristic::Free3D(Free3DPlacementConfig::default()),
            ],
        );
        let costs = placements
            .iter()
            .map(|placed| {
                placement_cost_breakdown(&module, &candidates, placed, config.congestion)
                    .weighted_total(config.cost_weights)
            })
            .collect::<Vec<_>>();

        assert!(costs.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn routing_congestion_penalizes_overlapping_net_regions() {
        let config = RoutingCongestionConfig {
            bin_size_xy: 4,
            bin_size_z: 2,
        };
        let overlapping = estimate_routing_congestion(
            &[
                (Position(0, 0, 0), Position(12, 0, 0)),
                (Position(0, 1, 0), Position(12, 1, 0)),
            ],
            config,
        );
        let separated = estimate_routing_congestion(
            &[
                (Position(0, 0, 0), Position(12, 0, 0)),
                (Position(0, 8, 0), Position(12, 8, 0)),
            ],
            config,
        );

        assert!(overlapping > separated);
        assert_eq!(separated, 0);
    }
}
