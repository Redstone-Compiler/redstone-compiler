use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use eyre::ContextCompat;

use super::{
    candidate_layout, candidate_matches_truth_table, candidate_ports_cover_module_ports,
    candidate_satisfies_local_cell_contract, generate_unit_candidates, CandidateInputMode,
    CandidatePort, UnitCandidateConfig,
};
use crate::graph::logic::LogicGraph;
use crate::graph::{Graph, GraphNode, GraphNodeId, GraphNodeKind};
use crate::ir::{
    Endpoint, NetClass, RoutableDesign, RoutableInstance, RoutableModule, RoutableModuleBody,
    RoutableNet, RoutableNode, RoutableNodeKind, RoutablePort, RoutablePortDirection,
    ROUTABLE_IR_TARGET, ROUTABLE_IR_VERSION,
};
use crate::logic::LogicType;
use crate::output::PlacedWorld;
use crate::transform::place_and_route::global_pnr::assembly::assemble_world;
use crate::transform::place_and_route::global_pnr::combinational_macro::CombinationalMacroLibrary;
use crate::transform::place_and_route::global_pnr::heuristics::GlobalHeuristicHooks;
use crate::transform::place_and_route::global_pnr::ir::{LayoutCandidate, PhysicalPortDirection};
use crate::transform::place_and_route::global_pnr::placer::{
    place_candidates_on_shelves, GlobalPlacementConfig,
};
use crate::transform::place_and_route::global_pnr::progress::GlobalPnrProgress;
use crate::transform::place_and_route::global_pnr::router::{
    collect_topology_input_endpoints, collect_topology_output_endpoints,
    route_resolved_topology_with_order_from_prefix, GlobalRoutingConfig, GlobalRoutingStrategy,
    NetOrderStrategy, RouteValidationMode,
};
use crate::transform::place_and_route::global_pnr::topology::ResolvedPnrTopology;

const CLUSTER_TRIGGER_LOGIC_NODES: usize = 8;
const CLUSTER_MAX_LOGIC_NODES: usize = 4;

#[derive(Clone, Debug)]
struct ClusterBoundaryPort {
    name: String,
    direction: PhysicalPortDirection,
    signal: GraphNodeId,
}

#[derive(Clone, Debug)]
struct CombinationalCluster {
    name: String,
    original_nodes: Vec<GraphNodeId>,
    graph: Graph,
    ports: Vec<ClusterBoundaryPort>,
}

#[derive(Clone, Debug)]
struct CombinationalClusterPlan {
    original: Graph,
    clusters: Vec<CombinationalCluster>,
    synthetic: RoutableDesign,
}

pub(super) fn try_generate_clustered_candidates(
    module_name: &str,
    graph: &Graph,
    ports: &[CandidatePort],
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
    input_mode: CandidateInputMode,
) -> eyre::Result<Option<Vec<LayoutCandidate>>> {
    let logic_nodes = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.kind, GraphNodeKind::Logic(_)))
        .count();
    if logic_nodes < CLUSTER_TRIGGER_LOGIC_NODES {
        return Ok(None);
    }
    let plan =
        match CombinationalClusterPlan::build(module_name, graph, ports, CLUSTER_MAX_LOGIC_NODES) {
            Ok(Some(plan)) => plan,
            Ok(None) => return Ok(None),
            Err(error) => {
                tracing::debug!(
                    module = module_name,
                    error = %error,
                    "combinational clustering plan failed; falling back to monolithic placement"
                );
                return Ok(None);
            }
        };

    match compose_candidate(
        &plan,
        module_name,
        ports,
        config,
        progress_label,
        input_mode,
    ) {
        Ok(Some(candidate)) => {
            tracing::info!(
                module = module_name,
                clusters = plan.clusters.len(),
                bbox_volume = candidate.cost.bbox_volume,
                blocks = candidate.cost.block_count,
                "clustered candidate composition succeeded"
            );
            Ok(Some(vec![candidate]))
        }
        Ok(None) => {
            tracing::info!(
                module = module_name,
                clusters = plan.clusters.len(),
                "clustered candidate composition exhausted; falling back to monolithic placement"
            );
            Ok(None)
        }
        Err(error) => {
            tracing::debug!(
                module = module_name,
                error = %error,
                "clustered candidate composition failed; falling back to monolithic placement"
            );
            Ok(None)
        }
    }
}

impl CombinationalClusterPlan {
    fn build(
        module_name: &str,
        graph: &Graph,
        ports: &[CandidatePort],
        max_logic_nodes: usize,
    ) -> eyre::Result<Option<Self>> {
        if max_logic_nodes == 0 {
            eyre::bail!("cluster size must be positive");
        }
        let mut original = graph.clone();
        original.build_outputs();
        original.build_producers();
        original.build_consumers();
        original.verify()?;
        if original.has_cycle()
            || original.nodes.iter().any(|node| {
                !matches!(
                    node.kind,
                    GraphNodeKind::Input(_) | GraphNodeKind::Output(_) | GraphNodeKind::Logic(_)
                )
            })
        {
            return Ok(None);
        }

        let logic_order = original
            .topological_order()
            .into_iter()
            .filter(|node_id| {
                original
                    .find_node_by_id(*node_id)
                    .is_some_and(|node| matches!(node.kind, GraphNodeKind::Logic(_)))
            })
            .collect::<Vec<_>>();
        if logic_order.len() <= max_logic_nodes {
            return Ok(None);
        }

        // Lowering provenance tags preserve useful semantic cones even after
        // XOR/half-adder logic has been technology-mapped to NOT/OR nodes.
        // Prefer those cones over arbitrary fixed-size cuts: this minimizes
        // duplicated top inputs and exposes reusable half-adder macros.
        let mut tagged_chunks = Vec::<Vec<GraphNodeId>>::new();
        let mut tagged_index = HashMap::<String, usize>::new();
        let all_tagged = logic_order.iter().all(|node_id| {
            original
                .find_node_by_id(*node_id)
                .is_some_and(|node| !node.tag.is_empty())
        });
        if all_tagged {
            for node_id in &logic_order {
                let tag = original.find_node_by_id(*node_id).unwrap().tag.clone();
                let index = if let Some(index) = tagged_index.get(&tag) {
                    *index
                } else {
                    let index = tagged_chunks.len();
                    tagged_chunks.push(Vec::new());
                    tagged_index.insert(tag, index);
                    index
                };
                tagged_chunks[index].push(*node_id);
            }
        }
        let chunks = if tagged_chunks.len() > 1
            && tagged_chunks
                .iter()
                .all(|chunk| chunk.len() <= max_logic_nodes.saturating_mul(3))
        {
            tagged_chunks
        } else {
            logic_order
                .chunks(max_logic_nodes)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>()
        };
        let cluster_for_node = chunks
            .iter()
            .enumerate()
            .flat_map(|(cluster, nodes)| nodes.iter().map(move |node| (*node, cluster)))
            .collect::<HashMap<_, _>>();
        let mut clusters = Vec::with_capacity(chunks.len());
        for (index, original_nodes) in chunks.into_iter().enumerate() {
            clusters.push(extract_cluster(
                module_name,
                index,
                &original,
                original_nodes,
                &cluster_for_node,
            )?);
        }

        let synthetic =
            synthetic_design(module_name, &original, ports, &clusters, &cluster_for_node)?;
        synthetic.validate()?;
        Ok(Some(Self {
            original,
            clusters,
            synthetic,
        }))
    }
}

fn extract_cluster(
    module_name: &str,
    index: usize,
    graph: &Graph,
    original_nodes: Vec<GraphNodeId>,
    cluster_for_node: &HashMap<GraphNodeId, usize>,
) -> eyre::Result<CombinationalCluster> {
    let own = original_nodes.iter().copied().collect::<HashSet<_>>();
    let incoming = original_nodes
        .iter()
        .flat_map(|node_id| graph.find_node_by_id(*node_id).into_iter())
        .flat_map(|node| node.inputs.clone())
        .filter(|input| !own.contains(input))
        .collect::<BTreeSet<_>>();
    let outgoing = original_nodes
        .iter()
        .copied()
        .filter(|node_id| {
            graph.find_node_by_id(*node_id).is_some_and(|node| {
                node.outputs.iter().any(|output| {
                    !own.contains(output)
                        && (cluster_for_node.contains_key(output)
                            || graph.find_node_by_id(*output).is_some_and(|candidate| {
                                matches!(candidate.kind, GraphNodeKind::Output(_))
                            }))
                })
            })
        })
        .collect::<BTreeSet<_>>();
    if outgoing.is_empty() {
        eyre::bail!("cluster {index} has no observable boundary output");
    }

    let mut nodes = Vec::<GraphNode>::new();
    let mut remap = HashMap::<GraphNodeId, GraphNodeId>::new();
    let mut ports = Vec::new();
    for signal in incoming {
        let name = signal_port_name(signal);
        let id = nodes.len();
        nodes.push(GraphNode {
            kind: GraphNodeKind::Input(name.clone()),
            ..Default::default()
        });
        remap.insert(signal, id);
        ports.push(ClusterBoundaryPort {
            name,
            direction: PhysicalPortDirection::Input,
            signal,
        });
    }

    for original_id in &original_nodes {
        let original = graph
            .find_node_by_id(*original_id)
            .with_context(|| format!("missing original cluster node {original_id}"))?;
        let id = nodes.len();
        let inputs = original
            .inputs
            .iter()
            .map(|input| {
                remap
                    .get(input)
                    .copied()
                    .with_context(|| format!("cluster node {original_id} lost input {input}"))
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        nodes.push(GraphNode {
            kind: original.kind.clone(),
            inputs,
            tag: original.tag.clone(),
            ..Default::default()
        });
        remap.insert(*original_id, id);
    }

    for signal in outgoing {
        let name = signal_port_name(signal);
        nodes.push(GraphNode {
            kind: GraphNodeKind::Output(name.clone()),
            inputs: vec![remap[&signal]],
            ..Default::default()
        });
        ports.push(ClusterBoundaryPort {
            name,
            direction: PhysicalPortDirection::Output,
            signal,
        });
    }

    let mut graph = Graph::from_nodes(nodes);
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify()?;
    Ok(CombinationalCluster {
        name: format!("{module_name}.__cluster_{index}"),
        original_nodes,
        graph,
        ports,
    })
}

fn synthetic_design(
    module_name: &str,
    graph: &Graph,
    ports: &[CandidatePort],
    clusters: &[CombinationalCluster],
    cluster_for_node: &HashMap<GraphNodeId, usize>,
) -> eyre::Result<RoutableDesign> {
    let input_by_name = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.kind {
            GraphNodeKind::Input(name) => Some((name.clone(), node.id)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let output_source_by_name = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.kind {
            GraphNodeKind::Output(name) => {
                node.inputs.first().map(|source| (name.clone(), *source))
            }
            _ => None,
        })
        .collect::<HashMap<_, _>>();

    let mut top_ports = Vec::new();
    let mut input_port_by_signal = HashMap::<GraphNodeId, String>::new();
    let mut output_ports_by_signal = BTreeMap::<GraphNodeId, Vec<String>>::new();
    for port in ports {
        let direction = match port.direction {
            PhysicalPortDirection::Input => RoutablePortDirection::Input,
            PhysicalPortDirection::Output => RoutablePortDirection::Output,
        };
        top_ports.push(RoutablePort {
            name: port.name.clone(),
            direction,
        });
        match port.direction {
            PhysicalPortDirection::Input => {
                let signal = *input_by_name
                    .get(port.target.as_str())
                    .with_context(|| format!("missing graph input `{}`", port.target))?;
                input_port_by_signal.insert(signal, port.name.clone());
            }
            PhysicalPortDirection::Output => {
                let signal = *output_source_by_name
                    .get(port.target.as_str())
                    .with_context(|| format!("missing graph output `{}`", port.target))?;
                output_ports_by_signal
                    .entry(signal)
                    .or_default()
                    .push(port.name.clone());
            }
        }
    }

    let mut signal_sinks = BTreeMap::<GraphNodeId, Vec<Endpoint>>::new();
    let mut signal_driver = BTreeMap::<GraphNodeId, Endpoint>::new();
    for (signal, port) in &input_port_by_signal {
        signal_driver.insert(*signal, Endpoint::SelfPort { port: port.clone() });
    }
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for port in &cluster.ports {
            let endpoint = Endpoint::InstancePort {
                instance: cluster.name.clone(),
                port: port.name.clone(),
            };
            match port.direction {
                PhysicalPortDirection::Input => {
                    signal_sinks.entry(port.signal).or_default().push(endpoint)
                }
                PhysicalPortDirection::Output => {
                    if signal_driver.insert(port.signal, endpoint).is_some() {
                        eyre::bail!(
                            "signal {} is driven by more than one cluster (latest {cluster_index})",
                            port.signal
                        );
                    }
                }
            }
        }
    }
    for (signal, output_ports) in output_ports_by_signal {
        for port in output_ports {
            signal_sinks
                .entry(signal)
                .or_default()
                .push(Endpoint::SelfPort { port });
        }
    }

    let mut nets = Vec::new();
    for (signal, mut sinks) in signal_sinks {
        sinks.sort();
        sinks.dedup();
        let driver = signal_driver
            .remove(&signal)
            .with_context(|| format!("cluster boundary signal {signal} has no driver"))?;
        let is_io = matches!(driver, Endpoint::SelfPort { .. })
            || sinks
                .iter()
                .any(|sink| matches!(sink, Endpoint::SelfPort { .. }));
        nets.push(RoutableNet {
            name: signal_net_name(signal),
            class: if is_io { NetClass::Io } else { NetClass::Data },
            driver,
            sinks,
            origin: None,
        });
    }

    let top_name = format!("{module_name}.__clustered");
    let mut modules = clusters
        .iter()
        .map(cluster_routable_module)
        .collect::<eyre::Result<Vec<_>>>()?;
    modules.push(RoutableModule {
        name: top_name.clone(),
        ports: top_ports,
        body: RoutableModuleBody::Composite {
            instances: clusters
                .iter()
                .map(|cluster| RoutableInstance {
                    name: cluster.name.clone(),
                    module: cluster.name.clone(),
                    origin: None,
                })
                .collect(),
            nets,
        },
    });
    let design = RoutableDesign {
        version: ROUTABLE_IR_VERSION,
        target: ROUTABLE_IR_TARGET.to_owned(),
        top: top_name,
        modules,
        debug: Default::default(),
    };

    // Every original logic edge must either remain within one cluster or be
    // represented by exactly one explicit boundary net.
    for node in graph
        .nodes
        .iter()
        .filter(|node| matches!(node.kind, GraphNodeKind::Logic(_)))
    {
        for input in &node.inputs {
            if cluster_for_node.get(input) == cluster_for_node.get(&node.id) {
                continue;
            }
            let target_cluster = cluster_for_node[&node.id];
            let boundary_name = signal_port_name(*input);
            if !clusters[target_cluster].ports.iter().any(|port| {
                port.direction == PhysicalPortDirection::Input && port.name == boundary_name
            }) {
                eyre::bail!(
                    "cross-cluster edge {input} -> {} has no boundary port",
                    node.id
                );
            }
        }
    }
    Ok(design)
}

fn cluster_routable_module(cluster: &CombinationalCluster) -> eyre::Result<RoutableModule> {
    let ports = cluster
        .ports
        .iter()
        .map(|port| RoutablePort {
            name: port.name.clone(),
            direction: match port.direction {
                PhysicalPortDirection::Input => RoutablePortDirection::Input,
                PhysicalPortDirection::Output => RoutablePortDirection::Output,
            },
        })
        .collect();
    let nodes = cluster
        .graph
        .nodes
        .iter()
        .map(|node| {
            let kind = match &node.kind {
                GraphNodeKind::Input(name) => RoutableNodeKind::Input { name: name.clone() },
                GraphNodeKind::Output(name) => RoutableNodeKind::Output { name: name.clone() },
                GraphNodeKind::Logic(logic) => match logic.logic_type {
                    LogicType::Not => RoutableNodeKind::Not,
                    LogicType::And => eyre::bail!("unmapped AND in clustered Routable leaf"),
                    LogicType::Or => RoutableNodeKind::Or,
                    LogicType::Xor => eyre::bail!("unmapped XOR in clustered Routable leaf"),
                },
                _ => eyre::bail!("cluster contains a non-combinational node"),
            };
            Ok(RoutableNode {
                id: node.id,
                kind,
                inputs: node.inputs.clone(),
                tag: node.tag.clone(),
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(RoutableModule {
        name: cluster.name.clone(),
        ports,
        body: RoutableModuleBody::Leaf { nodes },
    })
}

fn compose_candidate(
    plan: &CombinationalClusterPlan,
    module_name: &str,
    module_ports: &[CandidatePort],
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
    input_mode: CandidateInputMode,
) -> eyre::Result<Option<LayoutCandidate>> {
    debug_assert_eq!(
        plan.clusters
            .iter()
            .map(|cluster| cluster.original_nodes.len())
            .sum::<usize>(),
        plan.original
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, GraphNodeKind::Logic(_)))
            .count()
    );
    let mut macro_library = CombinationalMacroLibrary::default();
    let mut candidate_sets = Vec::with_capacity(plan.clusters.len());
    for (cluster_index, cluster) in plan.clusters.iter().enumerate() {
        let mut cluster_config = config.clone();
        // Keep a small geometry portfolio. The most compact local result is
        // not necessarily externally routable once several cluster boundary
        // pins share a face.
        cluster_config.max_candidates = 2;
        cluster_config.input_constraints = Default::default();
        cluster_config.local_cell_contract = Default::default();
        // Cluster boundaries are real composition pins, not merely observable
        // aliases of an internal node. Materializing every boundary output
        // gives the composer a stable endpoint even when a cluster has
        // multiple outgoing signals.
        cluster_config.local_config.materialize_outputs = true;
        let candidate_ports = cluster
            .ports
            .iter()
            .map(|port| CandidatePort::new(&port.name, &port.name, port.direction.clone()))
            .collect::<Vec<_>>();
        let macro_ports = candidate_ports
            .iter()
            .map(|port| (port.name.clone(), port.direction.clone()))
            .collect::<Vec<_>>();
        let (generated, _) = macro_library.resolve_graph_or_generate(
            &cluster.name,
            &cluster.graph,
            &macro_ports,
            &cluster_config,
            || {
                generate_unit_candidates(
                    &cluster.name,
                    cluster.graph.clone(),
                    candidate_ports,
                    &cluster_config,
                    progress_label,
                    CandidateInputMode::ExternalPorts,
                )
            },
        )?;
        if generated.is_empty() {
            tracing::info!(
                module = module_name,
                cluster = cluster_index,
                cluster_name = cluster.name,
                "cluster produced no verified physical candidate"
            );
            return Ok(None);
        }
        candidate_sets.push(generated);
    }

    let topology = ResolvedPnrTopology::from_routable(&plan.synthetic)?;
    let progress = GlobalPnrProgress::new(false, module_name);
    let hooks = GlobalHeuristicHooks::default();
    for mut candidates in cluster_candidate_combinations(&candidate_sets) {
        // External input switches are allocated at the route world's maximum
        // X boundary. The topological first clusters consume those inputs, so
        // place them at that side instead of forcing every input across the
        // entire shelf.
        if plan.clusters.len() >= 3 {
            candidates.reverse();
        }
        for spacing in [4usize, 8] {
            let mut placed = place_candidates_on_shelves(
                &candidates,
                &GlobalPlacementConfig {
                    spacing,
                    shelf_width: 96,
                    max_attempts: 1,
                    ..Default::default()
                },
            );
            // Local candidates may legitimately expose a boundary pin on their
            // minimum Z face. Give the composer routing space below those pins;
            // the generic shelf placer otherwise puts them directly on the route
            // world's floor, where no supporting redstone path can approach.
            for module in &mut placed {
                module.origin.2 = module.origin.2.max(4);
            }
            for strategy in [
                GlobalRoutingStrategy::DirectGreedy { max_steps: 128 },
                GlobalRoutingStrategy::GreedyBeam {
                    beam_width: 64,
                    max_expansions: 2_048,
                    variant_seed: 0,
                },
            ] {
                let routing = GlobalRoutingConfig {
                    strategy,
                    validation: RouteValidationMode::Deferred,
                };
                let routes = match route_resolved_topology_with_order_from_prefix(
                    &topology,
                    None,
                    &hooks,
                    &candidates,
                    &placed,
                    &routing,
                    NetOrderStrategy::Criticality,
                    &progress,
                    &[],
                ) {
                    Ok(routes) => routes,
                    Err(error) => {
                        tracing::debug!(
                            module = module_name,
                            spacing,
                            strategy = ?strategy,
                            routed_nets = error.routed_nets.len(),
                            error = %error.error,
                            "cluster composition routing attempt failed"
                        );
                        continue;
                    }
                };
                let placed_world = PlacedWorld {
                    world: assemble_world(&candidates, &placed, &routes)?,
                    inputs: collect_topology_input_endpoints(&topology, &routes),
                    outputs: collect_topology_output_endpoints(&topology, &candidates, &placed),
                };
                if !candidate_matches_truth_table(
                    &LogicGraph {
                        graph: plan.original.clone(),
                    },
                    &placed_world,
                )? {
                    continue;
                }
                let (world, physical_ports) = candidate_layout(
                    module_ports,
                    false,
                    &config.input_constraints,
                    placed_world.world,
                    &placed_world.inputs,
                    &placed_world.outputs,
                    input_mode,
                );
                if !candidate_ports_cover_module_ports(module_ports, &physical_ports) {
                    continue;
                }
                let mut candidate =
                    LayoutCandidate::from_world(module_name.to_owned(), world, physical_ports)?;
                if candidate_satisfies_local_cell_contract(
                    &mut candidate,
                    &config.local_cell_contract,
                ) {
                    return Ok(Some(candidate));
                }
            }
        }
    }
    Ok(None)
}

/// A bounded portfolio: the all-best combination plus one alternative at a
/// time. This captures the common case where a single cluster needs a more
/// accessible pin geometry without exploding into a Cartesian product.
fn cluster_candidate_combinations(
    candidate_sets: &[Vec<LayoutCandidate>],
) -> Vec<Vec<LayoutCandidate>> {
    if candidate_sets.iter().any(Vec::is_empty) {
        return Vec::new();
    }
    let best = candidate_sets
        .iter()
        .map(|set| set[0].clone())
        .collect::<Vec<_>>();
    let mut combinations = vec![best.clone()];
    for (set_index, set) in candidate_sets.iter().enumerate() {
        for alternative in set.iter().skip(1) {
            let mut combination = best.clone();
            combination[set_index] = alternative.clone();
            combinations.push(combination);
        }
    }
    combinations
}

fn signal_port_name(signal: GraphNodeId) -> String {
    format!("signal_{signal}")
}

fn signal_net_name(signal: GraphNodeId) -> String {
    format!("signal_{signal}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::predefined_logics;
    use crate::transform::place_and_route::local_placer::{
        NotRouteStrategy, PlacementSamplingPolicy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;

    fn graph_ports(graph: &Graph) -> Vec<CandidatePort> {
        graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) => {
                    Some(CandidatePort::new(name, name, PhysicalPortDirection::Input))
                }
                GraphNodeKind::Output(name) => Some(CandidatePort::new(
                    name,
                    name,
                    PhysicalPortDirection::Output,
                )),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn full_adder_plan_partitions_every_logic_node_with_explicit_boundaries() -> eyre::Result<()> {
        let graph = predefined_logics::buffered_full_adder_graph()?
            .prepare_place()?
            .graph;
        let ports = graph_ports(&graph);
        let plan = CombinationalClusterPlan::build("full_adder", &graph, &ports, 4)?
            .context("full adder should require more than one cluster")?;

        assert!(plan.clusters.len() > 1);
        let planned = plan
            .clusters
            .iter()
            .flat_map(|cluster| cluster.original_nodes.iter().copied())
            .collect::<BTreeSet<_>>();
        let expected = graph
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, GraphNodeKind::Logic(_)))
            .map(|node| node.id)
            .collect::<BTreeSet<_>>();
        assert_eq!(planned, expected);
        assert!(plan.clusters.iter().all(|cluster| {
            cluster
                .ports
                .iter()
                .any(|port| port.direction == PhysicalPortDirection::Output)
        }));
        plan.synthetic.validate()?;
        Ok(())
    }

    #[test]
    fn two_stage_dag_has_an_end_to_end_cluster_composition_path() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("~(a|b)", "y")?.prepare_place()?.graph;
        let ports = graph_ports(&graph);
        let plan = CombinationalClusterPlan::build("nor", &graph, &ports, 1)?
            .context("NOR should split at one logic node per cluster")?;
        assert_eq!(plan.clusters.len(), 2);
        // Physical composition is exercised by `compose_candidate`; failure is
        // allowed to fall back in production, but this small DAG must route.
        let mut config = UnitCandidateConfig {
            dim: crate::world::position::DimSize(8, 8, 4),
            max_candidates: 1,
            ..Default::default()
        };
        config.local_config.greedy_input_generation = true;
        config.local_config.input_candidate_limit = Some(8);
        config.local_config.step_sampling_policy = SamplingPolicy::Random(32);
        config.local_config.placement_sampling_policy = PlacementSamplingPolicy::StepPolicy;
        config.local_config.not_route_strategy = NotRouteStrategy::DirectAndRedstone;
        config.local_config.max_not_route_step = 3;
        config.local_config.not_route_step_sampling_policy = SamplingPolicy::Random(32);
        config.local_config.max_route_step = 3;
        config.local_config.route_step_sampling_policy = SamplingPolicy::Random(32);
        let candidate = compose_candidate(
            &plan,
            "nor",
            &ports,
            &config,
            None,
            CandidateInputMode::MaterializedSwitches,
        )?
        .context("clustered top leaf should produce a materialized candidate")?;
        let input_ports = candidate
            .ports
            .iter()
            .filter(|port| port.direction == PhysicalPortDirection::Input)
            .collect::<Vec<_>>();
        assert_eq!(input_ports.len(), 2);
        assert!(input_ports
            .iter()
            .all(|port| candidate.world[port.position].kind.is_switch()));
        Ok(())
    }
}
