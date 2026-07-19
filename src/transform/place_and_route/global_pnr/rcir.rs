use std::collections::{BTreeMap, BTreeSet};

use eyre::{ContextCompat, WrapErr};

use crate::ir::{
    CandidateSpec, CongestionSpec, Free3dSweepSpec, InputPlacementSpec, LayerAssignmentSpec,
    LocalPlacerSpec, NetOrderSpec, NotRouteSpec, ObjectiveSpec, PhysicalConstraintSpec,
    PhysicalRegionSpec, PhysicalSpec, PlacementHeuristicSpec, PlacementSamplingSpec, PlacementSpec,
    PnrSpec, PortRef, PreferenceSpec, RoutableDesign, RoutableDocument, RoutableModuleBody,
    RouteStageSpec, RouteStrategySpec, RouteValidationSpec, RoutingSpec, SamplingSpec, SearchSpec,
    TorchPlacementSpec,
};
use crate::transform::place_and_route::global_pnr::candidate::{
    CandidatePolicySet, UnitCandidateConfig,
};
use crate::transform::place_and_route::global_pnr::physical_intent::{
    IntentRegion, PhysicalConstraint, PhysicalIntent, PreferenceStrength,
    ResolvedPhysicalConstraint, PHYSICAL_INTENT_FORMAT,
};
use crate::transform::place_and_route::global_pnr::policy::{
    Free3DPlacementConfig, GlobalPnrPolicies, GlobalSearchBudget, LayerAssignmentStrategy,
    LayeredPlacementConfig, PlacementCostWeights, PlacementHeuristic, RoutingCongestionConfig,
};
use crate::transform::place_and_route::global_pnr::router::{
    GlobalRoutingConfig, GlobalRoutingStrategy, NetOrderStrategy, RouteValidationMode,
};
use crate::transform::place_and_route::global_pnr::topology::{
    InstanceId, NetId, ResolvedPnrTopology,
};
use crate::transform::place_and_route::global_pnr::{GlobalPnrConfig, GlobalSearchConfig};
use crate::transform::place_and_route::local_placer::{
    InputPlacementStrategy, LocalPlacerConfig, LocalPlacerInputConstraints, NotRouteStrategy,
    PlacementSamplingPolicy, TorchPlacementStrategy,
};
use crate::transform::place_and_route::sampling::SamplingPolicy;
use crate::world::position::{DimSize, Position};

pub fn routable_document_from_config(
    design: &RoutableDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<RoutableDocument> {
    let topology = ResolvedPnrTopology::from_routable(design)?;
    let mut pin_search = BTreeMap::new();
    let mut candidate_profiles = BTreeMap::new();
    let mut candidate_bindings = BTreeMap::new();
    let mut interned_profiles = Vec::<(String, CandidateSpec)>::new();

    for module in &design.modules {
        let RoutableModuleBody::Leaf { nodes } = &module.body else {
            continue;
        };
        let policy = config
            .candidate
            .definition_overrides
            .get(&module.name)
            .unwrap_or(&config.candidate.default);
        let candidate_spec = candidate_spec_from_policy(policy);
        let profile = interned_profiles
            .iter()
            .find(|(_, existing)| *existing == candidate_spec)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| {
                let base = if interned_profiles.is_empty() {
                    format!("{}-cell-search", design.top)
                } else {
                    format!("{}-cell-search", module.name)
                };
                let mut name = base.clone();
                let mut suffix = 2;
                while candidate_profiles.contains_key(&name) {
                    name = format!("{base}-{suffix}");
                    suffix += 1;
                }
                candidate_profiles.insert(name.clone(), candidate_spec.clone());
                interned_profiles.push((name.clone(), candidate_spec));
                name
            });
        candidate_bindings.insert(module.name.clone(), profile);
        for (port, positions) in policy.input_constraints.input_positions() {
            if module.ports.iter().any(|candidate| {
                candidate.name == port
                    && candidate.direction == crate::ir::RoutablePortDirection::Input
            }) {
                pin_search.insert(
                    PortRef {
                        definition: module.name.clone(),
                        port: port.to_owned(),
                    },
                    positions.iter().map(position_array).collect(),
                );
            }
        }
        for (node_id, positions) in policy.input_constraints.node_positions() {
            let Some(crate::ir::RoutableNodeKind::Input { name }) = nodes
                .iter()
                .find(|node| node.id == node_id)
                .map(|node| &node.kind)
            else {
                continue;
            };
            pin_search.insert(
                PortRef {
                    definition: module.name.clone(),
                    port: name.clone(),
                },
                positions.iter().map(position_array).collect(),
            );
        }
    }
    for ((definition, port), positions) in &config.candidate.pin_search {
        pin_search.insert(
            PortRef {
                definition: definition.clone(),
                port: port.clone(),
            },
            positions.iter().map(position_array).collect(),
        );
    }

    let design_profile_name = format!("{}-design", design.top);
    let mut design_profiles = BTreeMap::new();
    design_profiles.insert(design_profile_name.clone(), pnr_spec_from_config(config));
    let mut design_bindings = BTreeMap::new();
    design_bindings.insert(design.top.clone(), design_profile_name);

    Ok(RoutableDocument {
        design: design.clone(),
        candidate_profiles,
        design_profiles,
        candidate_bindings,
        design_bindings,
        pin_search,
        physical: config
            .physical_intent
            .as_ref()
            .map(|intent| physical_spec_from_resolved(intent, &topology))
            .transpose()?,
    })
}

pub fn apply_routable_document(
    document: &RoutableDocument,
    config: &mut GlobalPnrConfig,
) -> eyre::Result<()> {
    document.design.validate()?;
    let design_profile_name = document
        .design_bindings
        .get(&document.design.top)
        .with_context(|| {
            format!(
                "top module `{}` has no @pnr.design profile",
                document.design.top
            )
        })?;
    let design_profile = document
        .design_profiles
        .get(design_profile_name)
        .with_context(|| format!("missing pnr.design profile `{design_profile_name}`"))?;
    apply_pnr_spec(design_profile, config)?;

    let mut resolved_candidates = Vec::<(String, UnitCandidateConfig)>::new();
    for module in &document.design.modules {
        if !matches!(module.body, RoutableModuleBody::Leaf { .. }) {
            continue;
        }
        let profile_name = document
            .candidate_bindings
            .get(&module.name)
            .with_context(|| format!("leaf `{}` has no @pnr.candidate profile", module.name))?;
        let profile = document
            .candidate_profiles
            .get(profile_name)
            .with_context(|| format!("missing pnr.candidate profile `{profile_name}`"))?;
        resolved_candidates.push((module.name.clone(), candidate_policy_from_spec(profile)));
    }
    let (_, default_candidate) = resolved_candidates
        .first()
        .context("routable document contains no leaf candidate profile bindings")?;
    let default_candidate = default_candidate.clone();
    let mut candidate = CandidatePolicySet::new(default_candidate.clone());
    for (definition, policy) in resolved_candidates {
        if policy != default_candidate {
            candidate.definition_overrides.insert(definition, policy);
        }
    }
    config.candidate = candidate;

    for (port_ref, positions) in &document.pin_search {
        let module = document
            .design
            .module(&port_ref.definition)
            .with_context(|| {
                format!(
                    "unknown @pnr.pin_search definition `{}`",
                    port_ref.definition
                )
            })?;
        if !matches!(module.body, RoutableModuleBody::Leaf { .. }) {
            eyre::bail!(
                "@pnr.pin_search definition `{}` is not a leaf",
                port_ref.definition
            );
        }
        let port = module
            .ports
            .iter()
            .find(|port| port.name == port_ref.port)
            .with_context(|| {
                format!(
                    "unknown @pnr.pin_search port `{}.{}`",
                    port_ref.definition, port_ref.port
                )
            })?;
        if port.direction != crate::ir::RoutablePortDirection::Input {
            eyre::bail!(
                "@pnr.pin_search port `{}.{}` is not an input",
                port_ref.definition,
                port_ref.port
            );
        }
        config.candidate.pin_search.insert(
            (port_ref.definition.clone(), port_ref.port.clone()),
            positions.iter().copied().map(array_position).collect(),
        );
    }

    if let Some(physical) = &document.physical {
        let topology = ResolvedPnrTopology::from_routable(&document.design)?;
        config.physical_intent =
            Some(physical_intent_from_spec(physical, &document.design.top).bind(&topology)?);
    }
    Ok(())
}

pub fn pnr_spec_from_config(config: &GlobalPnrConfig) -> PnrSpec {
    PnrSpec {
        placement: PlacementSpec {
            initial_spacing: config.placement.spacing,
            shelf_width: config.placement.shelf_width,
            max_attempts: config.placement.max_attempts,
            heuristics: placement_specs(&config.search.policies.placement_heuristics),
            congestion: CongestionSpec {
                bin_size_xy: config.placement.congestion.bin_size_xy,
                bin_size_z: config.placement.congestion.bin_size_z,
            },
            objective: objective_spec(config.placement.cost_weights),
        },
        routing: RoutingSpec {
            probe: config.routing_probe.map(route_stage_spec),
            primary: route_stage_spec(config.routing),
            refinement: config.routing_refinement.map(route_stage_spec),
            net_order: config
                .search
                .policies
                .net_order_strategies
                .iter()
                .copied()
                .map(net_order_spec)
                .collect(),
        },
        search: SearchSpec {
            candidates_per_child: config.search.budget.max_candidates_per_child,
            layout_combinations: config.search.budget.max_layout_combinations,
            detailed_routing_attempts: config.search.budget.max_detailed_routing_attempts,
            refined_routing_attempts: config.search.budget.max_refined_routing_attempts,
            refinement_rounds: config.search.budget.max_refinement_rounds,
        },
    }
}

pub fn apply_pnr_spec(spec: &PnrSpec, config: &mut GlobalPnrConfig) -> eyre::Result<()> {
    config.placement.spacing = spec.placement.initial_spacing;
    config.placement.shelf_width = spec.placement.shelf_width;
    config.placement.max_attempts = spec.placement.max_attempts;
    config.placement.congestion = RoutingCongestionConfig {
        bin_size_xy: spec.placement.congestion.bin_size_xy,
        bin_size_z: spec.placement.congestion.bin_size_z,
    };
    config.placement.cost_weights = placement_weights(spec.placement.objective);
    config.routing_probe = spec.routing.probe.map(route_stage_config);
    config.routing = route_stage_config(spec.routing.primary);
    config.routing_refinement = spec.routing.refinement.map(route_stage_config);
    config.search = GlobalSearchConfig {
        budget: GlobalSearchBudget {
            max_candidates_per_child: spec.search.candidates_per_child,
            max_layout_combinations: spec.search.layout_combinations,
            max_detailed_routing_attempts: spec.search.detailed_routing_attempts,
            max_refined_routing_attempts: spec.search.refined_routing_attempts,
            max_refinement_rounds: spec.search.refinement_rounds,
        },
        policies: GlobalPnrPolicies {
            placement_heuristics: expand_placement_specs(&spec.placement.heuristics)?,
            net_order_strategies: spec
                .routing
                .net_order
                .iter()
                .copied()
                .map(net_order_strategy)
                .collect(),
        },
    };
    Ok(())
}

fn candidate_spec_from_policy(policy: &UnitCandidateConfig) -> CandidateSpec {
    CandidateSpec {
        search_box: [policy.dim.0, policy.dim.1, policy.dim.2],
        retain: policy.max_candidates,
        combinational_samples: policy.combinational_sampling_limit,
        local_placer: local_spec(policy.local_config),
    }
}

fn candidate_policy_from_spec(spec: &CandidateSpec) -> UnitCandidateConfig {
    UnitCandidateConfig {
        dim: DimSize(spec.search_box[0], spec.search_box[1], spec.search_box[2]),
        local_config: local_config(spec.local_placer.clone()),
        input_constraints: LocalPlacerInputConstraints::default(),
        max_candidates: spec.retain,
        combinational_sampling_limit: spec.combinational_samples,
    }
}

fn local_spec(config: LocalPlacerConfig) -> LocalPlacerSpec {
    LocalPlacerSpec {
        random_seed: config.random_seed,
        greedy_input_generation: config.greedy_input_generation,
        input_placement: match config.input_placement_strategy {
            InputPlacementStrategy::Boundary => InputPlacementSpec::Boundary,
            InputPlacementStrategy::Anywhere => InputPlacementSpec::Anywhere,
        },
        input_candidate_limit: config.input_candidate_limit,
        step_sampling: sampling_spec(config.step_sampling_policy),
        placement_sampling: placement_sampling_spec(config.placement_sampling_policy),
        leak_sampling: config.leak_sampling,
        route_torch_directly: config.route_torch_directly,
        materialize_outputs: config.materialize_outputs,
        torch_placement: match config.torch_placement_strategy {
            TorchPlacementStrategy::DirectOnly => TorchPlacementSpec::DirectOnly,
            TorchPlacementStrategy::AnywhereNonAdjacent => TorchPlacementSpec::AnywhereNonAdjacent,
        },
        not_route_strategy: match config.not_route_strategy {
            NotRouteStrategy::DirectOnly => NotRouteSpec::DirectOnly,
            NotRouteStrategy::RedstoneOnly => NotRouteSpec::RedstoneOnly,
            NotRouteStrategy::DirectAndRedstone => NotRouteSpec::DirectAndRedstone,
        },
        max_not_route_step: config.max_not_route_step,
        not_route_step_sampling: sampling_spec(config.not_route_step_sampling_policy),
        max_route_step: config.max_route_step,
        route_step_sampling: sampling_spec(config.route_step_sampling_policy),
    }
}

fn local_config(spec: LocalPlacerSpec) -> LocalPlacerConfig {
    LocalPlacerConfig {
        random_seed: spec.random_seed,
        greedy_input_generation: spec.greedy_input_generation,
        input_placement_strategy: match spec.input_placement {
            InputPlacementSpec::Boundary => InputPlacementStrategy::Boundary,
            InputPlacementSpec::Anywhere => InputPlacementStrategy::Anywhere,
        },
        input_candidate_limit: spec.input_candidate_limit,
        step_sampling_policy: sampling_policy(spec.step_sampling),
        placement_sampling_policy: placement_sampling_policy(spec.placement_sampling),
        leak_sampling: spec.leak_sampling,
        route_torch_directly: spec.route_torch_directly,
        materialize_outputs: spec.materialize_outputs,
        torch_placement_strategy: match spec.torch_placement {
            TorchPlacementSpec::DirectOnly => TorchPlacementStrategy::DirectOnly,
            TorchPlacementSpec::AnywhereNonAdjacent => TorchPlacementStrategy::AnywhereNonAdjacent,
        },
        not_route_strategy: match spec.not_route_strategy {
            NotRouteSpec::DirectOnly => NotRouteStrategy::DirectOnly,
            NotRouteSpec::RedstoneOnly => NotRouteStrategy::RedstoneOnly,
            NotRouteSpec::DirectAndRedstone => NotRouteStrategy::DirectAndRedstone,
        },
        max_not_route_step: spec.max_not_route_step,
        not_route_step_sampling_policy: sampling_policy(spec.not_route_step_sampling),
        max_route_step: spec.max_route_step,
        route_step_sampling_policy: sampling_policy(spec.route_step_sampling),
    }
}

fn sampling_spec(policy: SamplingPolicy) -> SamplingSpec {
    match policy {
        SamplingPolicy::None => SamplingSpec::None,
        SamplingPolicy::Take(n) => SamplingSpec::Take(n),
        SamplingPolicy::Random(n) => SamplingSpec::Random(n),
    }
}
fn sampling_policy(spec: SamplingSpec) -> SamplingPolicy {
    match spec {
        SamplingSpec::None => SamplingPolicy::None,
        SamplingSpec::Take(n) => SamplingPolicy::Take(n),
        SamplingSpec::Random(n) => SamplingPolicy::Random(n),
    }
}
fn placement_sampling_spec(policy: PlacementSamplingPolicy) -> PlacementSamplingSpec {
    match policy {
        PlacementSamplingPolicy::StepPolicy => PlacementSamplingSpec::StepPolicy,
        PlacementSamplingPolicy::Cost {
            count,
            random_count,
            start_step,
        } => PlacementSamplingSpec::Cost {
            count,
            random_count,
            start_step,
        },
        PlacementSamplingPolicy::Ranked {
            count,
            random_count,
            start_step,
        } => PlacementSamplingSpec::Ranked {
            count,
            random_count,
            start_step,
        },
    }
}
fn placement_sampling_policy(spec: PlacementSamplingSpec) -> PlacementSamplingPolicy {
    match spec {
        PlacementSamplingSpec::StepPolicy => PlacementSamplingPolicy::StepPolicy,
        PlacementSamplingSpec::Cost {
            count,
            random_count,
            start_step,
        } => PlacementSamplingPolicy::Cost {
            count,
            random_count,
            start_step,
        },
        PlacementSamplingSpec::Ranked {
            count,
            random_count,
            start_step,
        } => PlacementSamplingPolicy::Ranked {
            count,
            random_count,
            start_step,
        },
    }
}

fn placement_specs(heuristics: &[PlacementHeuristic]) -> Vec<PlacementHeuristicSpec> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < heuristics.len() {
        if let PlacementHeuristic::Free3D(first) = heuristics[index] {
            let mut seeds = BTreeSet::new();
            let mut clearances = BTreeSet::new();
            let mut end = index;
            while let Some(PlacementHeuristic::Free3D(next)) = heuristics.get(end).copied() {
                if !same_free3d_family(first, next) {
                    break;
                }
                seeds.insert(next.seed);
                clearances.insert(next.clearance);
                end += 1;
            }
            let seeds = seeds.into_iter().collect::<Vec<_>>();
            let clearances = clearances.into_iter().collect::<Vec<_>>();
            let complete_cartesian = end - index == seeds.len() * clearances.len()
                && clearances.iter().all(|clearance| {
                    seeds.iter().all(|seed| {
                        heuristics[index..end].iter().any(|heuristic| {
                            matches!(heuristic, PlacementHeuristic::Free3D(value) if value.seed == *seed && value.clearance == *clearance)
                        })
                    })
                });
            let emit = |value: Free3DPlacementConfig, seeds, clearances| {
                PlacementHeuristicSpec::Free3dSweep(Free3dSweepSpec {
                    seeds,
                    clearances,
                    iterations: value.iterations,
                    step_size: value.step_size,
                    attraction: value.attraction,
                    repulsion: value.repulsion,
                    compactness: value.compactness,
                    damping: value.damping,
                    vertical_scale: value.vertical_scale,
                })
            };
            if complete_cartesian {
                result.push(emit(first, seeds, clearances));
            } else {
                for heuristic in &heuristics[index..end] {
                    let PlacementHeuristic::Free3D(value) = heuristic else {
                        unreachable!()
                    };
                    result.push(emit(*value, vec![value.seed], vec![value.clearance]));
                }
            }
            index = end;
            continue;
        }
        result.push(match heuristics[index] {
            PlacementHeuristic::Shelf => PlacementHeuristicSpec::Shelf,
            PlacementHeuristic::Grid => PlacementHeuristicSpec::Grid,
            PlacementHeuristic::RegisterCarryChain => PlacementHeuristicSpec::RegisterCarryChain,
            PlacementHeuristic::RegisterCarryAlignedSlices => {
                PlacementHeuristicSpec::RegisterCarryAlignedSlices
            }
            PlacementHeuristic::RegisterGrid => PlacementHeuristicSpec::RegisterGrid,
            PlacementHeuristic::RegisterTriangles => PlacementHeuristicSpec::RegisterTriangles,
            PlacementHeuristic::RegisterSlices => PlacementHeuristicSpec::RegisterSlices,
            PlacementHeuristic::Layered3D(value) => PlacementHeuristicSpec::Layered3d {
                layers: value.layers,
                layer_spacing: value.layer_spacing,
                assignment: match value.assignment {
                    LayerAssignmentStrategy::Alternating => LayerAssignmentSpec::Alternating,
                    LayerAssignmentStrategy::NetAware => LayerAssignmentSpec::NetAware,
                },
            },
            PlacementHeuristic::Free3D(_) => unreachable!(),
        });
        index += 1;
    }
    result
}

fn same_free3d_family(a: Free3DPlacementConfig, b: Free3DPlacementConfig) -> bool {
    a.iterations == b.iterations
        && a.step_size == b.step_size
        && a.attraction == b.attraction
        && a.repulsion == b.repulsion
        && a.compactness == b.compactness
        && a.damping == b.damping
        && a.vertical_scale == b.vertical_scale
}

fn expand_placement_specs(
    specs: &[PlacementHeuristicSpec],
) -> eyre::Result<Vec<PlacementHeuristic>> {
    let mut result = Vec::new();
    for spec in specs {
        match spec {
            PlacementHeuristicSpec::Shelf => result.push(PlacementHeuristic::Shelf),
            PlacementHeuristicSpec::Grid => result.push(PlacementHeuristic::Grid),
            PlacementHeuristicSpec::RegisterCarryChain => {
                result.push(PlacementHeuristic::RegisterCarryChain)
            }
            PlacementHeuristicSpec::RegisterCarryAlignedSlices => {
                result.push(PlacementHeuristic::RegisterCarryAlignedSlices)
            }
            PlacementHeuristicSpec::RegisterGrid => result.push(PlacementHeuristic::RegisterGrid),
            PlacementHeuristicSpec::RegisterTriangles => {
                result.push(PlacementHeuristic::RegisterTriangles)
            }
            PlacementHeuristicSpec::RegisterSlices => {
                result.push(PlacementHeuristic::RegisterSlices)
            }
            PlacementHeuristicSpec::Layered3d {
                layers,
                layer_spacing,
                assignment,
            } => result.push(PlacementHeuristic::Layered3D(LayeredPlacementConfig {
                layers: *layers,
                layer_spacing: *layer_spacing,
                assignment: match assignment {
                    LayerAssignmentSpec::Alternating => LayerAssignmentStrategy::Alternating,
                    LayerAssignmentSpec::NetAware => LayerAssignmentStrategy::NetAware,
                },
            })),
            PlacementHeuristicSpec::Free3dSweep(sweep) => {
                if sweep.seeds.is_empty() || sweep.clearances.is_empty() {
                    eyre::bail!("free3d sweep must contain seeds and clearances");
                }
                for clearance in &sweep.clearances {
                    for seed in &sweep.seeds {
                        result.push(PlacementHeuristic::Free3D(Free3DPlacementConfig {
                            seed: *seed,
                            clearance: *clearance,
                            iterations: sweep.iterations,
                            step_size: sweep.step_size,
                            attraction: sweep.attraction,
                            repulsion: sweep.repulsion,
                            compactness: sweep.compactness,
                            damping: sweep.damping,
                            vertical_scale: sweep.vertical_scale,
                        }));
                    }
                }
            }
        }
    }
    if result.is_empty() {
        eyre::bail!("PnR requires at least one placement heuristic");
    }
    Ok(result)
}

fn objective_spec(w: PlacementCostWeights) -> ObjectiveSpec {
    ObjectiveSpec {
        placement_volume: w.placement_volume,
        xy_footprint: w.xy_footprint,
        height_span: w.height_span,
        estimated_wire_length: w.estimated_wire_length,
        vertical_distance: w.vertical_distance,
        routing_congestion: w.routing_congestion,
    }
}
fn placement_weights(w: ObjectiveSpec) -> PlacementCostWeights {
    PlacementCostWeights {
        placement_volume: w.placement_volume,
        xy_footprint: w.xy_footprint,
        height_span: w.height_span,
        estimated_wire_length: w.estimated_wire_length,
        vertical_distance: w.vertical_distance,
        routing_congestion: w.routing_congestion,
    }
}

fn route_stage_spec(config: GlobalRoutingConfig) -> RouteStageSpec {
    RouteStageSpec {
        strategy: match config.strategy {
            GlobalRoutingStrategy::BreadthFirst => RouteStrategySpec::BreadthFirst,
            GlobalRoutingStrategy::AStar => RouteStrategySpec::AStar,
            GlobalRoutingStrategy::DirectGreedy { max_steps } => {
                RouteStrategySpec::DirectGreedy { max_steps }
            }
            GlobalRoutingStrategy::GreedyBeam {
                beam_width,
                max_expansions,
                variant_seed,
            } => RouteStrategySpec::GreedyBeam {
                width: beam_width,
                max_expansions,
                variant_seed,
            },
        },
        validation: match config.validation {
            RouteValidationMode::Incremental => RouteValidationSpec::Incremental,
            RouteValidationMode::Deferred => RouteValidationSpec::Deferred,
        },
    }
}
fn route_stage_config(spec: RouteStageSpec) -> GlobalRoutingConfig {
    GlobalRoutingConfig {
        strategy: match spec.strategy {
            RouteStrategySpec::BreadthFirst => GlobalRoutingStrategy::BreadthFirst,
            RouteStrategySpec::AStar => GlobalRoutingStrategy::AStar,
            RouteStrategySpec::DirectGreedy { max_steps } => {
                GlobalRoutingStrategy::DirectGreedy { max_steps }
            }
            RouteStrategySpec::GreedyBeam {
                width,
                max_expansions,
                variant_seed,
            } => GlobalRoutingStrategy::GreedyBeam {
                beam_width: width,
                max_expansions,
                variant_seed,
            },
        },
        validation: match spec.validation {
            RouteValidationSpec::Incremental => RouteValidationMode::Incremental,
            RouteValidationSpec::Deferred => RouteValidationMode::Deferred,
        },
    }
}
fn net_order_spec(value: NetOrderStrategy) -> NetOrderSpec {
    match value {
        NetOrderStrategy::Criticality => NetOrderSpec::Criticality,
        NetOrderStrategy::HighestFanoutFirst => NetOrderSpec::HighestFanoutFirst,
        NetOrderStrategy::ReverseCriticality => NetOrderSpec::ReverseCriticality,
    }
}
fn net_order_strategy(value: NetOrderSpec) -> NetOrderStrategy {
    match value {
        NetOrderSpec::Criticality => NetOrderStrategy::Criticality,
        NetOrderSpec::HighestFanoutFirst => NetOrderStrategy::HighestFanoutFirst,
        NetOrderSpec::ReverseCriticality => NetOrderStrategy::ReverseCriticality,
    }
}

fn physical_intent_from_spec(spec: &PhysicalSpec, design: &str) -> PhysicalIntent {
    PhysicalIntent {
        format: PHYSICAL_INTENT_FORMAT.to_owned(),
        design: design.to_owned(),
        regions: spec
            .regions
            .iter()
            .map(|(name, region)| {
                (
                    name.clone(),
                    IntentRegion {
                        min: region.min,
                        max: region.max,
                    },
                )
            })
            .collect(),
        constraints: spec
            .constraints
            .iter()
            .map(|constraint| match constraint {
                PhysicalConstraintSpec::Inside {
                    id,
                    instance,
                    region,
                } => PhysicalConstraint::Inside {
                    id: id.clone(),
                    instance: instance.clone(),
                    region: region.clone(),
                },
                PhysicalConstraintSpec::LayerRange {
                    id,
                    instance,
                    min,
                    max,
                } => PhysicalConstraint::LayerRange {
                    id: id.clone(),
                    instance: instance.clone(),
                    min: *min,
                    max: *max,
                },
                PhysicalConstraintSpec::FixedOrigin {
                    id,
                    instance,
                    origin,
                } => PhysicalConstraint::FixedOrigin {
                    id: id.clone(),
                    instance: instance.clone(),
                    origin: *origin,
                },
                PhysicalConstraintSpec::NetPriority { id, net, priority } => {
                    PhysicalConstraint::NetPriority {
                        id: id.clone(),
                        net: net.clone(),
                        priority: *priority,
                    }
                }
                PhysicalConstraintSpec::NetAvoid { id, net, region } => {
                    PhysicalConstraint::NetAvoid {
                        id: id.clone(),
                        net: net.clone(),
                        region: region.clone(),
                    }
                }
                PhysicalConstraintSpec::PreferInside {
                    id,
                    instance,
                    region,
                    strength,
                } => PhysicalConstraint::PreferInside {
                    id: id.clone(),
                    instance: instance.clone(),
                    region: region.clone(),
                    strength: preference_strength(*strength),
                },
                PhysicalConstraintSpec::SameLayer {
                    id,
                    first,
                    second,
                    strength,
                } => PhysicalConstraint::SameLayer {
                    id: id.clone(),
                    first: first.clone(),
                    second: second.clone(),
                    strength: preference_strength(*strength),
                },
            })
            .collect(),
    }
}

fn physical_spec_from_resolved(
    intent: &crate::transform::place_and_route::global_pnr::ResolvedPhysicalIntent,
    topology: &ResolvedPnrTopology,
) -> eyre::Result<PhysicalSpec> {
    let instance_name = |id: InstanceId| {
        topology
            .instances
            .get(id.0)
            .map(|v| v.display_name.clone())
            .context("resolved physical intent references an unknown instance")
    };
    let net_name = |id: NetId| {
        topology
            .nets
            .get(id.0)
            .map(|v| v.display_name.clone())
            .context("resolved physical intent references an unknown net")
    };
    let constraints = intent
        .constraints
        .iter()
        .map(|constraint| {
            Ok(match constraint {
                ResolvedPhysicalConstraint::Inside {
                    id,
                    instance,
                    region,
                } => PhysicalConstraintSpec::Inside {
                    id: id.clone(),
                    instance: instance_name(*instance)?,
                    region: region.clone(),
                },
                ResolvedPhysicalConstraint::LayerRange {
                    id,
                    instance,
                    min,
                    max,
                } => PhysicalConstraintSpec::LayerRange {
                    id: id.clone(),
                    instance: instance_name(*instance)?,
                    min: *min,
                    max: *max,
                },
                ResolvedPhysicalConstraint::FixedOrigin {
                    id,
                    instance,
                    origin,
                } => PhysicalConstraintSpec::FixedOrigin {
                    id: id.clone(),
                    instance: instance_name(*instance)?,
                    origin: *origin,
                },
                ResolvedPhysicalConstraint::NetPriority { id, net, priority } => {
                    PhysicalConstraintSpec::NetPriority {
                        id: id.clone(),
                        net: net_name(*net)?,
                        priority: *priority,
                    }
                }
                ResolvedPhysicalConstraint::NetAvoid { id, net, region } => {
                    PhysicalConstraintSpec::NetAvoid {
                        id: id.clone(),
                        net: net_name(*net)?,
                        region: region.clone(),
                    }
                }
                ResolvedPhysicalConstraint::PreferInside {
                    id,
                    instance,
                    region,
                    strength,
                } => PhysicalConstraintSpec::PreferInside {
                    id: id.clone(),
                    instance: instance_name(*instance)?,
                    region: region.clone(),
                    strength: preference_spec(*strength),
                },
                ResolvedPhysicalConstraint::SameLayer {
                    id,
                    first,
                    second,
                    strength,
                } => PhysicalConstraintSpec::SameLayer {
                    id: id.clone(),
                    first: instance_name(*first)?,
                    second: instance_name(*second)?,
                    strength: preference_spec(*strength),
                },
            })
        })
        .collect::<eyre::Result<Vec<_>>>()
        .wrap_err("could not serialize resolved physical intent")?;
    Ok(PhysicalSpec {
        regions: intent
            .regions
            .iter()
            .map(|(name, region)| {
                (
                    name.clone(),
                    PhysicalRegionSpec {
                        min: region.min,
                        max: region.max,
                    },
                )
            })
            .collect(),
        constraints,
    })
}

fn preference_strength(value: PreferenceSpec) -> PreferenceStrength {
    match value {
        PreferenceSpec::Weak => PreferenceStrength::Weak,
        PreferenceSpec::Medium => PreferenceStrength::Medium,
        PreferenceSpec::Strong => PreferenceStrength::Strong,
    }
}
fn preference_spec(value: PreferenceStrength) -> PreferenceSpec {
    match value {
        PreferenceStrength::Weak => PreferenceSpec::Weak,
        PreferenceStrength::Medium => PreferenceSpec::Medium,
        PreferenceStrength::Strong => PreferenceSpec::Strong,
    }
}
fn position_array(position: &Position) -> [usize; 3] {
    [position.0, position.1, position.2]
}
fn array_position(position: [usize; 3]) -> Position {
    Position(position[0], position[1], position[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        RoutableModule, RoutableModuleBody, RoutableNode, RoutableNodeKind, RoutablePort,
        RoutablePortDirection, ROUTABLE_IR_TARGET, ROUTABLE_IR_VERSION,
    };
    use crate::transform::place_and_route::global_pnr::policy::{
        Free3DPlacementConfig, PlacementHeuristic,
    };

    #[test]
    fn config_round_trips_through_routable_document() -> eyre::Result<()> {
        let design = RoutableDesign {
            version: ROUTABLE_IR_VERSION,
            target: ROUTABLE_IR_TARGET.to_owned(),
            top: "leaf".to_owned(),
            modules: vec![RoutableModule {
                name: "leaf".to_owned(),
                ports: vec![RoutablePort {
                    name: "a".to_owned(),
                    direction: RoutablePortDirection::Input,
                }],
                body: RoutableModuleBody::Leaf {
                    nodes: vec![RoutableNode {
                        id: 0,
                        kind: RoutableNodeKind::Input {
                            name: "a".to_owned(),
                        },
                        inputs: Vec::new(),
                        tag: String::new(),
                    }],
                },
            }],
        };
        let mut original = GlobalPnrConfig::default();
        original.candidate =
            original
                .candidate
                .clone()
                .with_pin_search("leaf", "a", [Position(1, 2, 3)]);
        let mut leaf_policy = original.candidate.default.clone();
        leaf_policy.max_candidates = 3;
        original
            .candidate
            .definition_overrides
            .insert("leaf".to_owned(), leaf_policy);
        original.search.policies.placement_heuristics = [2, 4]
            .into_iter()
            .flat_map(|clearance| {
                (0..2).map(move |seed| {
                    PlacementHeuristic::Free3D(Free3DPlacementConfig {
                        seed,
                        clearance,
                        ..Default::default()
                    })
                })
            })
            .collect();
        let document = routable_document_from_config(&design, &original)?;
        let text = document.to_string();
        assert!(text.contains("profile pnr.candidate \"leaf-cell-search\""));
        assert!(text.contains("profile pnr.design \"leaf-design\""));
        assert!(text.contains("@pnr.candidate(profile = \"leaf-cell-search\")"));
        assert!(text.contains("@pnr.design(profile = \"leaf-design\")"));
        assert!(!text.contains("candidate-defaults"));
        let parsed: RoutableDocument = text.parse()?;
        let mut restored = GlobalPnrConfig::default();
        apply_routable_document(&parsed, &mut restored)?;
        assert_eq!(
            restored.candidate.effective_for_definition("leaf"),
            original.candidate.effective_for_definition("leaf")
        );
        assert_eq!(
            pnr_spec_from_config(&restored),
            pnr_spec_from_config(&original)
        );
        Ok(())
    }

    #[test]
    fn standalone_routable_requires_explicit_profiles() {
        let document = RoutableDocument::circuit_only(RoutableDesign {
            version: ROUTABLE_IR_VERSION,
            target: ROUTABLE_IR_TARGET.to_owned(),
            top: "leaf".to_owned(),
            modules: vec![RoutableModule {
                name: "leaf".to_owned(),
                ports: Vec::new(),
                body: RoutableModuleBody::Leaf { nodes: Vec::new() },
            }],
        });
        let error = apply_routable_document(&document, &mut GlobalPnrConfig::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no @pnr.design profile"));
    }
}
