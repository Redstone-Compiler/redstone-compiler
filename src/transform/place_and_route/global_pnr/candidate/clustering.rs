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
    ClusteringSpec, Endpoint, NetClass, RoutableDesign, RoutableInstance, RoutableModule,
    RoutableModuleBody, RoutableNet, RoutableNode, RoutableNodeKind, RoutablePort,
    RoutablePortDirection, ROUTABLE_IR_TARGET, ROUTABLE_IR_VERSION,
};
use crate::logic::LogicType;
use crate::output::{OutputEndpoint, PlacedWorld};
use crate::transform::place_and_route::global_pnr::assembly::assemble_world;
use crate::transform::place_and_route::global_pnr::combinational_macro::CombinationalMacroLibrary;
use crate::transform::place_and_route::global_pnr::heuristics::GlobalHeuristicHooks;
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPortDirection, PortConnection,
};
use crate::transform::place_and_route::global_pnr::placer::{
    place_candidates_on_shelves, GlobalPlacementConfig, PlacedModule,
};
use crate::transform::place_and_route::global_pnr::progress::GlobalPnrProgress;
use crate::transform::place_and_route::global_pnr::router::{
    collect_topology_input_endpoints, collect_topology_output_endpoints, materialize_input_diode,
    redstone_network_positions, route_point_to_point_with_strategy_and_allowed_contacts,
    route_resolved_topology_with_order_from_prefix, GlobalRoutingConfig, GlobalRoutingStrategy,
    NetOrderStrategy, RouteValidationMode,
};
use crate::transform::place_and_route::global_pnr::topology::{
    ResolvedEndpoint, ResolvedPnrTopology,
};
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

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

fn candidate_matches_truth_table_with_switch_inputs(
    graph: &Graph,
    candidate: &LayoutCandidate,
) -> bool {
    let mut verification_world = candidate.world.clone();
    let mut inputs = Vec::new();
    for port in candidate
        .ports
        .iter()
        .filter(|port| port.direction == PhysicalPortDirection::Input)
    {
        let access_points = port.routing_access_positions();
        let restored_switch = (access_points.len() > 1)
            .then_some(port.route_position)
            .flatten()
            .filter(|position| {
                verification_world.size.bound_on(*position)
                    && verification_world[*position].kind.is_air()
            })
            .and_then(|switch| {
                access_points
                    .iter()
                    .copied()
                    .find(|target| switch.manhattan_distance(target) == 1)
                    .map(|target| (switch, target))
            });
        if let Some((switch, target)) = restored_switch {
            verification_world[switch] = Block {
                kind: BlockKind::Switch { is_on: false },
                direction: switch.diff(target),
            };
            inputs.push(OutputEndpoint::new(port.name.clone(), switch));
        } else {
            inputs.extend(
                access_points
                    .into_iter()
                    .map(|position| OutputEndpoint::new(port.name.clone(), position)),
            );
        }
    }
    let placed = PlacedWorld {
        world: verification_world,
        inputs,
        outputs: candidate
            .ports
            .iter()
            .filter(|port| port.direction == PhysicalPortDirection::Output)
            .map(|port| OutputEndpoint::new(port.name.clone(), port.primary_route_position()))
            .collect(),
    };
    candidate_matches_truth_table(
        &LogicGraph {
            graph: graph.clone(),
        },
        &placed,
    )
    .unwrap_or(false)
}

pub(super) fn try_generate_clustered_candidates(
    module_name: &str,
    graph: &Graph,
    ports: &[CandidatePort],
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
    input_mode: CandidateInputMode,
) -> eyre::Result<Option<Vec<LayoutCandidate>>> {
    let clustering = &config.clustering;
    if !clustering.enabled {
        return Ok(None);
    }
    if graph
        .nodes
        .iter()
        .any(|node| matches!(node.kind, GraphNodeKind::Sequential(_)))
    {
        return Ok(None);
    }
    let logic_nodes = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.kind, GraphNodeKind::Logic(_)))
        .count();
    if logic_nodes < clustering.trigger_logic_nodes {
        return Ok(None);
    }
    let plan = match CombinationalClusterPlan::build(module_name, graph, ports, clustering) {
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
        clustering: &ClusteringSpec,
    ) -> eyre::Result<Option<Self>> {
        let max_logic_nodes = clustering.max_logic_nodes;
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
        let all_tagged = clustering.prefer_provenance
            && logic_order.iter().all(|node_id| {
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
        let mut chunks = if tagged_chunks.len() > 1
            && tagged_chunks
                .iter()
                .all(|chunk| chunk.len() <= clustering.max_tagged_logic_nodes)
        {
            tagged_chunks
        } else {
            logic_order
                .chunks(max_logic_nodes)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>()
        };
        if all_tagged && chunks.len() > 1 {
            let mut consumed = vec![false; chunks.len()];
            for small_index in 0..chunks.len() {
                if chunks[small_index].len() >= clustering.max_logic_nodes {
                    continue;
                }
                let small_nodes = chunks[small_index].clone();
                let upstream_chunks = (0..chunks.len())
                    .filter(|&candidate_index| candidate_index != small_index)
                    .filter(|&candidate_index| {
                        let candidate_nodes = chunks[candidate_index]
                            .iter()
                            .copied()
                            .collect::<HashSet<_>>();
                        small_nodes
                            .iter()
                            .filter_map(|node| original.find_node_by_id(*node))
                            .flat_map(|node| node.inputs.clone())
                            .any(|input| candidate_nodes.contains(&input))
                    })
                    .count();
                // A small multi-input join (for example the carry OR joining
                // two half adders) is a useful physical cluster in its own
                // right. Arbitrarily absorbing it into either producer makes
                // the other dependency a long backward cut and prevents the
                // two producer macros from being composed symmetrically.
                if upstream_chunks > 1 {
                    continue;
                }
                let target = (0..chunks.len())
                    .filter(|&candidate_index| {
                        candidate_index != small_index
                            && !consumed[candidate_index]
                            && chunks[candidate_index].len() + small_nodes.len()
                                <= clustering.max_tagged_logic_nodes
                    })
                    .map(|candidate_index| {
                        let candidate_nodes = chunks[candidate_index]
                            .iter()
                            .copied()
                            .collect::<HashSet<_>>();
                        let dependency_edges = small_nodes
                            .iter()
                            .filter_map(|node| original.find_node_by_id(*node))
                            .flat_map(|node| node.inputs.clone())
                            .filter(|input| candidate_nodes.contains(input))
                            .count();
                        // Prefer the downstream chunk when the small node has
                        // an equal number of inputs from multiple chunks.
                        // Merging into an upstream chunk would create a
                        // backward boundary net (and often a cluster cycle);
                        // merging into the already-dependent chunk keeps the
                        // cluster graph feed-forward.
                        let incoming_boundary_edges = chunks[candidate_index]
                            .iter()
                            .filter_map(|node| original.find_node_by_id(*node))
                            .flat_map(|node| node.inputs.clone())
                            .filter(|input| {
                                !candidate_nodes.contains(input)
                                    && !small_nodes.contains(input)
                                    && original.find_node_by_id(*input).is_some_and(|node| {
                                        matches!(node.kind, GraphNodeKind::Logic(_))
                                    })
                            })
                            .count();
                        (dependency_edges, incoming_boundary_edges, candidate_index)
                    })
                    .filter(|(dependency_edges, _, _)| *dependency_edges > 0)
                    .max();
                if let Some((_, _, target_index)) = target {
                    chunks[target_index].extend(small_nodes);
                    consumed[small_index] = true;
                }
            }
            chunks = chunks
                .into_iter()
                .enumerate()
                .filter_map(|(index, chunk)| (!consumed[index]).then_some(chunk))
                .collect();
        }
        // Avoid exposing a multi-sink electrical fanout at a physical cluster
        // boundary when one consumer can be absorbed by the producer cluster.
        // Prefer cheap OR nodes over torch-backed NOT nodes. Any additional
        // inputs must be top-level inputs or already owned by the producer,
        // otherwise the relocation could introduce a backward cluster edge.
        loop {
            let owner = chunks
                .iter()
                .enumerate()
                .flat_map(|(cluster, nodes)| nodes.iter().map(move |node| (*node, cluster)))
                .collect::<HashMap<_, _>>();
            let relocation = logic_order.iter().find_map(|signal| {
                let producer_cluster = *owner.get(signal)?;
                let producer = original.find_node_by_id(*signal)?;
                let mut consumers_by_cluster = BTreeMap::<usize, Vec<GraphNodeId>>::new();
                for consumer in &producer.outputs {
                    let Some(consumer_cluster) = owner.get(consumer).copied() else {
                        continue;
                    };
                    if consumer_cluster != producer_cluster {
                        consumers_by_cluster
                            .entry(consumer_cluster)
                            .or_default()
                            .push(*consumer);
                    }
                }
                consumers_by_cluster
                    .into_iter()
                    .filter(|(_, consumers)| consumers.len() > 1)
                    .find_map(|(consumer_cluster, consumers)| {
                        if chunks[producer_cluster].len() >= clustering.max_tagged_logic_nodes {
                            return None;
                        }
                        consumers
                            .into_iter()
                            .filter_map(|consumer| {
                                let node = original.find_node_by_id(consumer)?;
                                let GraphNodeKind::Logic(logic) = &node.kind else {
                                    return None;
                                };
                                let inputs_are_feed_forward = node.inputs.iter().all(|input| {
                                    *input == *signal
                                        || owner
                                            .get(input)
                                            .is_none_or(|cluster| *cluster == producer_cluster)
                                });
                                inputs_are_feed_forward.then_some((
                                    match logic.logic_type {
                                        LogicType::Or => 0,
                                        LogicType::Not => 1,
                                        _ => 2,
                                    },
                                    consumer,
                                ))
                            })
                            .min()
                            .map(|(_, consumer)| (producer_cluster, consumer_cluster, consumer))
                    })
            });
            let Some((producer_cluster, consumer_cluster, consumer)) = relocation else {
                break;
            };
            chunks[consumer_cluster].retain(|node| *node != consumer);
            chunks[producer_cluster].push(consumer);
        }
        // Prefer cutting immediately before an inverter instead of exporting
        // the inverter's torch-backed output. A NOT output generally needs an
        // extra repeater diode to become a safe routing terminal, while its
        // input is commonly an OR/redstone network that can be routed directly.
        // Move the inverter to its sole downstream consumer cluster when both
        // sides still satisfy the configured cluster-size contract.
        loop {
            let owner = chunks
                .iter()
                .enumerate()
                .flat_map(|(cluster, nodes)| nodes.iter().map(move |node| (*node, cluster)))
                .collect::<HashMap<_, _>>();
            let relocation = logic_order.iter().find_map(|node_id| {
                let producer_cluster = *owner.get(node_id)?;
                let node = original.find_node_by_id(*node_id)?;
                let GraphNodeKind::Logic(logic) = &node.kind else {
                    return None;
                };
                if logic.logic_type != LogicType::Not || chunks[producer_cluster].len() <= 1 {
                    return None;
                }
                let consumer_clusters = node
                    .outputs
                    .iter()
                    .filter_map(|output| owner.get(output).copied())
                    .filter(|cluster| *cluster != producer_cluster)
                    .collect::<BTreeSet<_>>();
                let consumer_cluster = consumer_clusters.iter().copied().next()?;
                let has_local_logic_consumer = node
                    .outputs
                    .iter()
                    .any(|output| owner.get(output) == Some(&producer_cluster));
                (consumer_clusters.len() == 1
                    && !has_local_logic_consumer
                    && chunks[consumer_cluster].len() < clustering.max_tagged_logic_nodes)
                    .then_some((producer_cluster, consumer_cluster, *node_id))
            });
            let Some((producer_cluster, consumer_cluster, node_id)) = relocation else {
                break;
            };
            chunks[producer_cluster].retain(|node| *node != node_id);
            chunks[consumer_cluster].push(node_id);
        }
        // If an inverter feeds logic on both sides of a cut, moving it would
        // break the local consumer but exporting its torch-backed result makes
        // a poor physical pin. Duplicate that one-node cone in the downstream
        // cluster and export the inverter input instead. This is the same
        // bounded logic replication used by physical synthesis to trade one
        // cheap gate for a substantially easier cut.
        if chunks.len() == 2 {
            loop {
                let owner = chunks
                    .iter()
                    .enumerate()
                    .flat_map(|(cluster, nodes)| nodes.iter().map(move |node| (*node, cluster)))
                    .collect::<HashMap<_, _>>();
                let duplication = logic_order.iter().find_map(|node_id| {
                    let producer_cluster = *owner.get(node_id)?;
                    let node = original.find_node_by_id(*node_id)?;
                    let GraphNodeKind::Logic(logic) = &node.kind else {
                        return None;
                    };
                    if logic.logic_type != LogicType::Not {
                        return None;
                    }
                    let local_consumers = node
                        .outputs
                        .iter()
                        .filter(|output| owner.get(output) == Some(&producer_cluster))
                        .copied()
                        .collect::<Vec<_>>();
                    let mut external_consumers = BTreeMap::<usize, Vec<GraphNodeId>>::new();
                    for output in &node.outputs {
                        if let Some(cluster) = owner.get(output).copied()
                            && cluster != producer_cluster
                        {
                            external_consumers.entry(cluster).or_default().push(*output);
                        }
                    }
                    let (consumer_cluster, consumers) = external_consumers.into_iter().next()?;
                    (!local_consumers.is_empty()
                        && consumers.len() > 0
                        && node.outputs.iter().all(|output| {
                            owner.get(output).is_none_or(|cluster| {
                                *cluster == producer_cluster || *cluster == consumer_cluster
                            })
                        })
                        && chunks[consumer_cluster].len() < clustering.max_tagged_logic_nodes)
                        .then_some((
                            producer_cluster,
                            consumer_cluster,
                            *node_id,
                            node.kind.clone(),
                            node.inputs.clone(),
                            node.tag.clone(),
                            consumers,
                        ))
                });
                let Some((
                    _producer_cluster,
                    consumer_cluster,
                    original_inverter,
                    kind,
                    inputs,
                    tag,
                    consumers,
                )) = duplication
                else {
                    break;
                };
                let duplicate = original.add_node(GraphNode {
                    kind,
                    inputs,
                    outputs: Vec::new(),
                    tag,
                });
                for consumer in consumers {
                    if let Some(mut node) = original.nodes.get_mut(consumer) {
                        for input in &mut node.inputs {
                            if *input == original_inverter {
                                *input = duplicate;
                            }
                        }
                    }
                }
                chunks[consumer_cluster].push(duplicate);
                original.build_outputs();
                original.build_producers();
                original.build_consumers();
            }
        }
        original.verify()?;
        let topological_rank = original
            .topological_order()
            .into_iter()
            .enumerate()
            .map(|(rank, node)| (node, rank))
            .collect::<HashMap<_, _>>();
        for chunk in &mut chunks {
            chunk.sort_by_key(|node| topological_rank.get(node).copied().unwrap_or(usize::MAX));
        }
        let cluster_for_node = chunks
            .iter()
            .enumerate()
            .flat_map(|(cluster, nodes)| nodes.iter().map(move |node| (*node, cluster)))
            .collect::<HashMap<_, _>>();
        tracing::debug!(module = module_name, clusters = ?chunks, "selected combinational cluster cuts");
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
        let input_id = nodes.len();
        nodes.push(GraphNode {
            kind: GraphNodeKind::Input(name.clone()),
            ..Default::default()
        });
        remap.insert(signal, input_id);
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
        cluster_config.max_candidates = config.clustering.candidates_per_cluster.max(1);
        cluster_config.input_constraints = Default::default();
        if cluster_config.dim.0 <= 2 {
            cluster_config.local_config.greedy_input_generation = false;
            cluster_config.local_config.input_candidate_limit = None;
        }
        if cluster_config.dim.0 <= 2 {
            let input_ports = cluster
                .ports
                .iter()
                .filter(|port| port.direction == PhysicalPortDirection::Input)
                .collect::<Vec<_>>();
            let routeable_input_positions =
                clustered_input_position_groups(cluster_config.dim, &cluster.graph, &input_ports);
            for (port, positions) in input_ports.into_iter().zip(routeable_input_positions) {
                cluster_config.input_constraints =
                    std::mem::take(&mut cluster_config.input_constraints)
                        .with_input_positions(&port.name, positions);
            }
        }
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
        let mut seed_candidate_sets = Vec::new();
        for seed_variant in 0..config.clustering.candidate_seed_variants.max(1) {
            let mut variant_config = cluster_config.clone();
            variant_config.local_config.random_seed = variant_config
                .local_config
                .random_seed
                .wrapping_add(seed_variant as u64);
            let generate = || {
                generate_unit_candidates(
                    &cluster.name,
                    cluster.graph.clone(),
                    candidate_ports.clone(),
                    &variant_config,
                    progress_label,
                    CandidateInputMode::ExternalPorts,
                )
            };
            let variant_candidates = if config.clustering.reuse_macros {
                macro_library
                    .resolve_graph_or_generate(
                        &cluster.name,
                        &cluster.graph,
                        &macro_ports,
                        &variant_config,
                        generate,
                    )?
                    .0
            } else {
                generate()?
            };
            seed_candidate_sets.push(variant_candidates);
        }
        let mut generated = Vec::new();
        let longest_seed_set = seed_candidate_sets
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or_default();
        for candidate_index in 0..longest_seed_set {
            for seed_candidates in &seed_candidate_sets {
                if let Some(candidate) = seed_candidates.get(candidate_index) {
                    generated.push(candidate.clone());
                }
            }
        }
        let mut generated = if config.local_cell_contract.max_bbox.is_some() {
            horizontal_mirror_variants(&generated)?
        } else {
            generated
        };
        // Materialize every independently named input boundary inside the
        // candidate. The packer must see the driver and repeater footprint;
        // adding them later in the global router can produce an apparently
        // compact placement that has no legal ingress or that backfeeds a
        // sibling input.
        let generated_before_materialization = generated.len();
        let mut materialization_failures = BTreeMap::<String, usize>::new();
        let mut isolation_rejections = 0usize;
        let mut bbox_rejections = 0usize;
        let mut post_input_truth_rejections = 0usize;
        let mut post_materialization_truth_rejections = 0usize;
        let top_input_ports = cluster
            .ports
            .iter()
            .filter(|port| port.direction == PhysicalPortDirection::Input)
            .filter(|port| {
                plan.original
                    .find_node_by_id(port.signal)
                    .is_some_and(|node| matches!(node.kind, GraphNodeKind::Input(_)))
            })
            .map(|port| port.name.clone())
            .collect::<HashSet<_>>();
        let cluster_logic_nodes = cluster
            .original_nodes
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let intercluster_outputs = cluster
            .ports
            .iter()
            .filter(|port| port.direction == PhysicalPortDirection::Output)
            .filter(|port| {
                plan.original
                    .find_node_by_id(port.signal)
                    .is_some_and(|node| {
                        node.outputs.iter().copied().any(|consumer| {
                            !cluster_logic_nodes.contains(&consumer)
                                && plan.original.find_node_by_id(consumer).is_some_and(|node| {
                                    matches!(node.kind, GraphNodeKind::Logic(_))
                                })
                        })
                    })
            })
            .map(|port| port.name.clone())
            .collect::<HashSet<_>>();
        generated = generated
            .into_iter()
            .filter_map(|mut candidate| {
                candidate = pad_candidate_y(candidate, 2).ok()?;
                let mut input_drivers = Vec::new();
                let mut port_indices = (0..candidate.ports.len()).collect::<Vec<_>>();
                port_indices.sort_by_key(|port_index| {
                    let port = &candidate.ports[*port_index];
                    if port.direction != PhysicalPortDirection::Input {
                        2
                    } else if top_input_ports.contains(&port.name) {
                        0
                    } else {
                        1
                    }
                });
                for port_index in port_indices {
                    if candidate.ports[port_index].direction
                        != PhysicalPortDirection::Input
                    {
                        candidate.ports[port_index].connection = PortConnection::Direct;
                        continue;
                    }
                    if top_input_ports.contains(&candidate.ports[port_index].name) {
                        let switch = candidate.ports[port_index].route_position?;
                        let target = candidate.ports[port_index]
                            .routing_access_positions()
                            .into_iter()
                            .find(|target| switch.manhattan_distance(target) == 1)?;
                        if !candidate.world.size.bound_on(switch)
                            || !candidate.world[switch].kind.is_air()
                        {
                            return None;
                        }
                        candidate.world[switch] = Block {
                            kind: BlockKind::Switch { is_on: false },
                            direction: switch.diff(target),
                        };
                        let port = &mut candidate.ports[port_index];
                        port.position = switch;
                        port.route_position = Some(switch);
                        port.access_points = vec![switch];
                        port.connection = PortConnection::Direct;
                        continue;
                    }
                    let preferred_repeater = candidate.ports[port_index].route_position;
                    let allowed_contacts =
                        candidate.ports[port_index].routing_access_positions();
                    if allowed_contacts.len() > 1 {
                        let fanout = allowed_contacts.iter().copied().find_map(|primary| {
                            let (base_world, driver) = materialize_input_diode(
                                &candidate.world,
                                primary,
                                preferred_repeater,
                                1,
                                &allowed_contacts,
                                &input_drivers,
                            )?;
                            let strategies = std::iter::once(
                                GlobalRoutingStrategy::DirectGreedy {
                                    max_steps: config.clustering.direct_max_steps,
                                },
                            )
                            .chain((0..8).map(|variant_seed| {
                                GlobalRoutingStrategy::GreedyBeam {
                                    beam_width: config.clustering.beam_width,
                                    max_expansions: config.clustering.beam_max_expansions,
                                    variant_seed,
                                }
                            }));
                            strategies.into_iter().find_map(|strategy| {
                                let mut world = base_world.clone();
                                for contact in allowed_contacts
                                    .iter()
                                    .copied()
                                    .filter(|contact| *contact != primary)
                                {
                                    if crate::transform::place_and_route::detailed_router::target_powers_position(
                                        &world,
                                        driver,
                                        contact,
                                    ) {
                                        continue;
                                    }
                                    let Ok((_, routed_world)) =
                                        route_point_to_point_with_strategy_and_allowed_contacts(
                                            &world,
                                            driver,
                                            contact,
                                            strategy,
                                            allowed_contacts.clone(),
                                        )
                                    else {
                                        return None;
                                    };
                                    world = routed_world;
                                }
                                existing_switches_do_not_power_driver(&world, driver)
                                    .then_some((world, driver))
                            })
                        });
                        let Some((world, driver)) = fanout else {
                            *materialization_failures
                                .entry(candidate.ports[port_index].name.clone())
                                .or_default() += 1;
                            return None;
                        };
                        candidate.world = world;
                        let port = &mut candidate.ports[port_index];
                        port.position = driver;
                        port.route_position = Some(driver);
                        port.access_points = vec![driver];
                        port.connection = PortConnection::Direct;
                        input_drivers.push(driver);
                        continue;
                    }
                    let mut diode_targets = allowed_contacts.clone();
                    diode_targets.extend(redstone_network_positions(
                        &candidate.world,
                        &allowed_contacts,
                    ));
                    diode_targets.sort_unstable();
                    diode_targets.dedup();
                    let Some((world, driver)) = diode_targets.into_iter().find_map(|sink| {
                            materialize_input_diode(
                                &candidate.world,
                                sink,
                                preferred_repeater,
                                1,
                                &allowed_contacts,
                                &input_drivers,
                            )
                        })
                    else {
                        *materialization_failures
                            .entry(candidate.ports[port_index].name.clone())
                            .or_default() += 1;
                        return None;
                    };
                    candidate.world = world;
                    let port = &mut candidate.ports[port_index];
                    port.position = driver;
                    port.route_position = Some(driver);
                    port.access_points = vec![driver];
                    port.connection = PortConnection::Direct;
                    input_drivers.push(driver);
                }
                let has_intercluster_input = candidate.ports.iter().any(|port| {
                    port.direction == PhysicalPortDirection::Input
                        && !top_input_ports.contains(&port.name)
                });
                if has_intercluster_input
                    && !candidate_input_drivers_are_isolated(&candidate.world, &candidate.ports)
                {
                    isolation_rejections += 1;
                    return None;
                }
                let input_materialized = LayoutCandidate::from_world(
                    candidate.module_name.clone(),
                    candidate.world.clone(),
                    candidate.ports.clone(),
                )
                .ok()?;
                if !candidate_matches_truth_table_with_switch_inputs(
                    &cluster.graph,
                    &input_materialized,
                ) {
                    post_input_truth_rejections += 1;
                    return None;
                }
                for port_index in (0..candidate.ports.len()).rev() {
                    if candidate.ports[port_index].direction
                        != PhysicalPortDirection::Output
                        || !intercluster_outputs
                            .contains(&candidate.ports[port_index].name)
                    {
                        continue;
                    }
                    let route_sources =
                        candidate.ports[port_index].routing_access_positions();
                    if route_sources.is_empty() {
                        return None;
                    }
                }
                let mut materialized = LayoutCandidate::from_world(
                    candidate.module_name,
                    candidate.world,
                    candidate.ports,
                )
                .ok()?;
                if !candidate_matches_truth_table_with_switch_inputs(
                    &cluster.graph,
                    &materialized,
                ) {
                    post_materialization_truth_rejections += 1;
                    return None;
                }
                if let Some(limit) = config.local_cell_contract.max_bbox
                    && (materialized.bbox.width() > limit[0]
                        || materialized.bbox.depth() > limit[1]
                        || materialized.bbox.height() > limit[2])
                {
                    if bbox_rejections == 0 {
                        tracing::debug!(
                            actual = ?[
                                materialized.bbox.width(),
                                materialized.bbox.depth(),
                                materialized.bbox.height(),
                            ],
                            ?limit,
                            min = ?materialized.bbox.min,
                            max = ?materialized.bbox.max,
                            "materialized cluster candidate exceeds bbox contract"
                        );
                    }
                    bbox_rejections += 1;
                    return None;
                }
                materialized.blocked_cells = candidate.blocked_cells;
                Some(materialized)
            })
            .collect();
        tracing::debug!(
            module = module_name,
            cluster = cluster_index,
            generated_before_materialization,
            materialized = generated.len(),
            input_failures = ?materialization_failures,
            isolation_rejections,
            bbox_rejections,
            post_input_truth_rejections,
            post_materialization_truth_rejections,
            "cluster input adapter materialization summary"
        );
        if generated.is_empty() {
            tracing::info!(
                module = module_name,
                cluster = cluster_index,
                cluster_name = cluster.name,
                "cluster produced no verified physical candidate"
            );
            return Ok(None);
        }
        tracing::trace!(
            module = module_name,
            cluster = cluster_index,
            candidates = ?generated
                .iter()
                .take(16)
                .enumerate()
                .map(|(candidate_index, candidate)| (
                    candidate_index,
                    candidate
                        .ports
                        .iter()
                        .map(|port| (
                            port.name.as_str(),
                            port.direction.clone(),
                            port.primary_route_position(),
                            candidate.world[port.primary_route_position()].kind,
                            port.routing_access_positions()
                                .into_iter()
                                .map(|position| (position, candidate.world[position].kind))
                                .collect::<Vec<_>>(),
                        ))
                        .collect::<Vec<_>>(),
                ))
                .collect::<Vec<_>>(),
            "cluster candidate port geometries"
        );
        candidate_sets.push(generated);
    }

    let topology = ResolvedPnrTopology::from_routable(&plan.synthetic)?;
    let progress = GlobalPnrProgress::new(false, module_name);
    let hooks = GlobalHeuristicHooks::default();
    let mut routing_strategies = Vec::new();
    if config.clustering.direct_max_steps > 0 {
        routing_strategies.push(GlobalRoutingStrategy::DirectGreedy {
            max_steps: config.clustering.direct_max_steps,
        });
    }
    if config.clustering.beam_width > 0 && config.clustering.beam_max_expansions > 0 {
        routing_strategies.extend(
            (0..8).map(|variant_seed| GlobalRoutingStrategy::GreedyBeam {
                beam_width: config.clustering.beam_width,
                max_expansions: config.clustering.beam_max_expansions,
                variant_seed,
            }),
        );
    }
    let net_orders = [
        NetOrderStrategy::Criticality,
        NetOrderStrategy::HighestFanoutFirst,
        NetOrderStrategy::ReverseCriticality,
    ];
    let mut combination_attempts = 0usize;
    let mut placement_attempts = 0usize;
    let mut routing_attempts = 0usize;
    let mut routing_successes = 0usize;
    let mut verification_rejections = 0usize;
    let mut intercluster_coupling_rejections = 0usize;
    let mut port_rejections = 0usize;
    let mut contract_rejections = 0usize;
    let mut diagnostic_emitted = false;
    for mut candidates in cluster_candidate_combinations(
        &candidate_sets,
        config.clustering.max_alternative_combinations,
    ) {
        combination_attempts += 1;
        // External input switches are allocated at the route world's maximum
        // X boundary. The topological first clusters consume those inputs, so
        // place them at that side instead of forcing every input across the
        // entire shelf.
        if config.clustering.input_boundary_bias && plan.clusters.len() >= 3 {
            candidates.reverse();
        }
        let mut placement_trials = config
            .local_cell_contract
            .max_bbox
            .map(|limit| {
                place_candidates_in_bbox(
                    &candidates,
                    limit[0],
                    limit[1],
                    limit[2],
                    config.clustering.routing_floor_margin,
                    4096,
                )
            })
            .unwrap_or_default();
        placement_trials.sort_by_key(|placed| {
            let connection_cost = estimated_cluster_connection_cost(&topology, &candidates, placed);
            (
                connection_cost,
                placed
                    .iter()
                    .map(|module| (module.origin.0, module.origin.1, module.origin.2))
                    .collect::<Vec<_>>(),
            )
        });
        tracing::debug!(
            module = module_name,
            net_distance_ranges = ?topology
                .nets
                .iter()
                .filter(|net| matches!(net.driver, ResolvedEndpoint::InstancePort { .. }))
                .map(|net| {
                    let distances = placement_trials
                        .iter()
                        .filter_map(|placed| {
                            estimated_cluster_net_length(&topology, net, &candidates, placed)
                        })
                        .collect::<Vec<_>>();
                    (
                        net.display_name.clone(),
                        distances.iter().min().copied(),
                        distances.iter().max().copied(),
                    )
                })
                .collect::<Vec<_>>(),
            "ranked packed cluster layouts by estimated wire length"
        );
        placement_trials.truncate(512);
        if config.local_cell_contract.max_bbox.is_none() {
            placement_trials.extend(config.clustering.placement_spacings.iter().map(|&spacing| {
                let mut placed = place_candidates_on_shelves(
                    &candidates,
                    &GlobalPlacementConfig {
                        spacing,
                        shelf_width: config.clustering.shelf_width,
                        max_attempts: 1,
                        ..Default::default()
                    },
                );
                // Local candidates may legitimately expose a boundary pin on their
                // minimum Z face. Give the composer routing space below those pins;
                // the generic shelf placer otherwise puts them directly on the route
                // world's floor, where no supporting redstone path can approach.
                for module in &mut placed {
                    module.origin.2 = module.origin.2.max(config.clustering.routing_floor_margin);
                }
                placed
            }));
        }
        for placed in placement_trials {
            placement_attempts += 1;
            if !candidate_signals_are_isolated(&candidates, &placed)? {
                intercluster_coupling_rejections += 1;
                continue;
            }
            for (&strategy, &net_order) in routing_strategies
                .iter()
                .flat_map(|strategy| net_orders.iter().map(move |order| (strategy, order)))
            {
                routing_attempts += 1;
                let routing = GlobalRoutingConfig {
                    strategy,
                    // Cluster boundary nets can be logically correlated after
                    // cut refinement (for example x and OR(x, cin)). Validating
                    // each partially routed net as an independent stimulus can
                    // reject a composition that is correct once every boundary
                    // is connected. The completed cluster world is exhaustively
                    // truth-table checked below.
                    validation: RouteValidationMode::Deferred,
                    top_inputs_last: true,
                    defer_feedback_cycles: true,
                };
                let routes = match route_resolved_topology_with_order_from_prefix(
                    &topology,
                    None,
                    &hooks,
                    &candidates,
                    &placed,
                    &routing,
                    net_order,
                    &progress,
                    &[],
                ) {
                    Ok(routes) => routes,
                    Err(error) => {
                        tracing::debug!(
                            module = module_name,
                            strategy = ?strategy,
                            net_order = ?net_order,
                            routed_nets = error.routed_nets.len(),
                            error = %error.error,
                            "cluster composition routing attempt failed"
                        );
                        continue;
                    }
                };
                routing_successes += 1;
                let world = assemble_world(&candidates, &placed, &routes)?;
                let placed_world = PlacedWorld {
                    world,
                    inputs: collect_topology_input_endpoints(&topology, &routes),
                    outputs: collect_topology_output_endpoints(&topology, &candidates, &placed),
                };
                match candidate_matches_truth_table(
                    &LogicGraph {
                        graph: plan.original.clone(),
                    },
                    &placed_world,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        verification_rejections += 1;
                        if !diagnostic_emitted {
                            trace_clustered_failure_state(
                                &placed_world,
                                &routes,
                                &candidates,
                                &placed,
                            );
                            tracing::debug!(
                                module = module_name,
                                placed = ?placed
                                    .iter()
                                    .map(|module| (
                                        module.module_name.as_str(),
                                        module.candidate_index,
                                        module.origin,
                                    ))
                                    .collect::<Vec<_>>(),
                                routes = ?routes
                                    .iter()
                                    .map(|route| (
                                        route.source_label.as_deref(),
                                        route.sink_label.as_deref(),
                                        route.source,
                                        route.sink,
                                        route.path.clone(),
                                    ))
                                    .collect::<Vec<_>>(),
                                inputs = ?placed_world
                                    .inputs
                                    .iter()
                                    .map(|endpoint| (endpoint.name.as_str(), endpoint.position()))
                                    .collect::<Vec<_>>(),
                                outputs = ?placed_world
                                    .outputs
                                    .iter()
                                    .map(|endpoint| (endpoint.name.as_str(), endpoint.position()))
                                    .collect::<Vec<_>>(),
                                "first clustered candidate truth-table rejection"
                            );
                            diagnostic_emitted = true;
                        }
                        continue;
                    }
                    Err(error) => {
                        // A densely routed composition can oscillate or exceed
                        // the bounded simulator budget. That rejects this
                        // physical candidate; it must not abort the remaining
                        // placement and candidate portfolio.
                        tracing::debug!(
                            module = module_name,
                            strategy = ?strategy,
                            net_order = ?net_order,
                            error = %error,
                            "clustered candidate verification failed"
                        );
                        verification_rejections += 1;
                        continue;
                    }
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
                    port_rejections += 1;
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
                contract_rejections += 1;
            }
        }
    }
    tracing::info!(
        module = module_name,
        combinations = combination_attempts,
        placements = placement_attempts,
        routing_attempts,
        routing_successes,
        intercluster_coupling_rejections,
        verification_rejections,
        port_rejections,
        contract_rejections,
        "clustered candidate composition summary"
    );
    Ok(None)
}

fn candidate_input_drivers_are_isolated(
    world: &World3D,
    ports: &[crate::transform::place_and_route::global_pnr::ir::PhysicalPort],
) -> bool {
    let inputs = ports
        .iter()
        .filter(|port| port.direction == PhysicalPortDirection::Input)
        .map(|port| port.position)
        .collect::<Vec<_>>();
    for left in 0..inputs.len() {
        for right in left + 1..inputs.len() {
            if crate::transform::place_and_route::detailed_router::target_powers_position(
                world,
                inputs[left],
                inputs[right],
            ) || crate::transform::place_and_route::detailed_router::target_powers_position(
                world,
                inputs[right],
                inputs[left],
            ) {
                return false;
            }
        }
    }
    true
}

fn existing_switches_do_not_power_driver(world: &World3D, driver: Position) -> bool {
    let switches = world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| block.kind.is_switch().then_some(position))
        .collect::<Vec<_>>();
    let mut inactive = world.clone();
    for switch in &switches {
        inactive[*switch].kind = BlockKind::Switch { is_on: false };
    }
    for switch in switches {
        let simulation_world = World::from(&inactive);
        let Ok(mut sim) = Simulator::from_with_limits_and_trace(&simulation_world, 256, 50_000, 0)
        else {
            return false;
        };
        if sim.world()[driver].kind.is_powered()
            || sim
                .change_state_with_limits(vec![(switch, true)], 256, 50_000)
                .is_err()
            || sim.world()[driver].kind.is_powered()
        {
            return false;
        }
    }
    true
}

fn pad_candidate_y(candidate: LayoutCandidate, padding: usize) -> eyre::Result<LayoutCandidate> {
    if padding == 0 {
        return Ok(candidate);
    }
    let mut world = World3D::new(DimSize(
        candidate.world.size.0,
        candidate.world.size.1 + padding * 2,
        candidate.world.size.2,
    ));
    let translate = |position: Position| Position(position.0, position.1 + padding, position.2);
    for (position, block) in candidate.world.iter_block() {
        world[translate(position)] = block;
    }
    let ports = candidate
        .ports
        .into_iter()
        .map(|mut port| {
            port.position = translate(port.position);
            port.route_position = port.route_position.map(translate);
            port.access_points = port.access_points.into_iter().map(translate).collect();
            port
        })
        .collect();
    let mut padded = LayoutCandidate::from_world(candidate.module_name, world, ports)?;
    padded.blocked_cells = candidate.blocked_cells.into_iter().map(translate).collect();
    Ok(padded)
}

fn cluster_input_positions(dim: crate::world::position::DimSize) -> Vec<Position> {
    if dim.0 == 0 || dim.1 == 0 || dim.2 <= 1 {
        return Vec::new();
    }
    let mut positions = Vec::new();
    if dim.0 <= 2 {
        // Cluster composition may need driver -> repeater -> cell at inputs
        // and source -> repeater -> route at outputs. Reserve two Y rows on
        // each side so those adapters can fit without forcing an otherwise
        // compact width-constrained candidate beyond its bbox contract.
        let (min_y, max_y) = if dim.1 >= 6 {
            (2, dim.1 - 3)
        } else {
            (1.min(dim.1.saturating_sub(1)), dim.1.saturating_sub(2))
        };
        for x in 0..dim.0 {
            // Floor repeaters need a support cell and some vertical clearance;
            // avoid asking the LocalPlacer for switch sites on either vertical
            // extreme when those sites will become physical input adapters.
            for z in 2..dim.2.saturating_sub(1) {
                positions.push(Position(x, min_y, z));
                if max_y != min_y {
                    positions.push(Position(x, max_y, z));
                }
            }
        }
        return positions;
    }
    for y in 0..dim.1 {
        for z in 1..dim.2 {
            positions.push(Position(0, y, z));
            if dim.0 > 1 {
                positions.push(Position(dim.0 - 1, y, z));
            }
        }
    }
    positions
}

fn clustered_input_position_groups(
    dim: DimSize,
    graph: &Graph,
    input_ports: &[&ClusterBoundaryPort],
) -> Vec<Vec<Position>> {
    if input_ports.is_empty() {
        return Vec::new();
    }
    let positions = cluster_input_positions(dim);
    let mut anchors = positions
        .iter()
        .map(|position| (position.1, position.2))
        .collect::<Vec<_>>();
    anchors.sort_unstable();
    anchors.dedup();
    if anchors.is_empty() {
        return vec![Vec::new(); input_ports.len()];
    }

    let signatures = input_ports
        .iter()
        .map(|port| reachable_output_signature(graph, &port.name))
        .collect::<Vec<_>>();
    let mut unique_signatures = Vec::<BTreeSet<String>>::new();
    for signature in &signatures {
        if !unique_signatures.contains(signature) {
            unique_signatures.push(signature.clone());
        }
    }

    let mut selected = vec![anchors[0]];
    while selected.len() < unique_signatures.len() && selected.len() < anchors.len() {
        let next = anchors
            .iter()
            .copied()
            .filter(|anchor| !selected.contains(anchor))
            .max_by_key(|anchor| {
                (
                    selected
                        .iter()
                        .map(|selected| {
                            anchor.0.abs_diff(selected.0) + anchor.1.abs_diff(selected.1)
                        })
                        .min()
                        .unwrap_or(0),
                    anchor.0,
                    anchor.1,
                )
            })
            .unwrap();
        selected.push(next);
    }

    signatures
        .iter()
        .enumerate()
        .map(|(input_index, signature)| {
            let signature_index = unique_signatures
                .iter()
                .position(|candidate| candidate == signature)
                .unwrap_or(0);
            let base_anchor = selected[signature_index % selected.len()];
            let signature_occurrence = signatures[..input_index]
                .iter()
                .filter(|previous| *previous == signature)
                .count();
            let alternate_y = anchors
                .iter()
                .map(|anchor| anchor.0)
                .max_by_key(|y| y.abs_diff(base_anchor.0))
                .unwrap_or(base_anchor.0);
            let is_cluster_boundary = input_ports[input_index].name.starts_with("signal_");
            let desired_anchor = if is_cluster_boundary {
                (
                    if signature_occurrence % 2 == 0 {
                        base_anchor.0
                    } else {
                        alternate_y
                    },
                    base_anchor
                        .1
                        .saturating_add((signature_occurrence / 2) * 4)
                        .min(dim.2.saturating_sub(2)),
                )
            } else {
                (
                    base_anchor.0,
                    base_anchor
                        .1
                        .saturating_add(signature_occurrence * 4)
                        .min(dim.2.saturating_sub(2)),
                )
            };
            let anchor = anchors
                .iter()
                .copied()
                .min_by_key(|anchor| {
                    anchor.0.abs_diff(desired_anchor.0) + anchor.1.abs_diff(desired_anchor.1)
                })
                .unwrap_or(base_anchor);
            positions
                .iter()
                .copied()
                .filter(|position| {
                    position.1.abs_diff(anchor.0) + position.2.abs_diff(anchor.1) <= 1
                })
                .collect()
        })
        .collect()
}

fn reachable_output_signature(graph: &Graph, input_name: &str) -> BTreeSet<String> {
    let Some(input) = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == input_name))
    else {
        return BTreeSet::new();
    };
    let mut outputs = BTreeSet::new();
    let mut stack = input.outputs.clone();
    let mut visited = HashSet::new();
    while let Some(node_id) = stack.pop() {
        if !visited.insert(node_id) {
            continue;
        }
        let Some(node) = graph.find_node_by_id(node_id) else {
            continue;
        };
        if let GraphNodeKind::Output(name) = &node.kind {
            outputs.insert(name.clone());
        } else {
            stack.extend(node.outputs.iter().copied());
        }
    }
    outputs
}

fn horizontal_mirror_variants(
    candidates: &[LayoutCandidate],
) -> eyre::Result<Vec<LayoutCandidate>> {
    let mut variants = Vec::with_capacity(candidates.len() * 4);
    variants.extend(candidates.iter().cloned());
    for (mirror_x, mirror_y) in [(true, false), (false, true), (true, true)] {
        for candidate in candidates {
            let transform_position = |position: Position| {
                Position(
                    if mirror_x {
                        candidate.world.size.0 - 1 - position.0
                    } else {
                        position.0
                    },
                    if mirror_y {
                        candidate.world.size.1 - 1 - position.1
                    } else {
                        position.1
                    },
                    position.2,
                )
            };
            let transform_direction = |direction: Direction| match direction {
                Direction::East if mirror_x => Direction::West,
                Direction::West if mirror_x => Direction::East,
                Direction::South if mirror_y => Direction::North,
                Direction::North if mirror_y => Direction::South,
                direction => direction,
            };

            let mut world = World3D::new(candidate.world.size);
            for (position, mut block) in candidate.world.iter_block() {
                block.direction = transform_direction(block.direction);
                world[transform_position(position)] = block;
            }
            world.initialize_redstone_states();
            let ports = candidate
                .ports
                .iter()
                .map(|port| {
                    let mut port = port.clone();
                    port.position = transform_position(port.position);
                    port.route_position = port.route_position.map(transform_position);
                    port.access_points = port
                        .access_points
                        .into_iter()
                        .map(transform_position)
                        .collect();
                    port
                })
                .collect();
            let mut variant =
                LayoutCandidate::from_world(candidate.module_name.clone(), world, ports)?;
            variant.blocked_cells = candidate
                .blocked_cells
                .iter()
                .copied()
                .map(transform_position)
                .collect();
            variants.push(variant);
        }
    }
    Ok(variants)
}

fn place_candidates_in_bbox(
    candidates: &[LayoutCandidate],
    width_limit: usize,
    depth_limit: usize,
    height_limit: usize,
    floor_margin: usize,
    max_layouts: usize,
) -> Vec<Vec<PlacedModule>> {
    if max_layouts == 0
        || candidates
            .iter()
            .any(|candidate| candidate.bbox.width() > width_limit)
    {
        return Vec::new();
    }

    let mut order = (0..candidates.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| {
        let candidate = &candidates[index];
        std::cmp::Reverse((
            candidate.occupied_cells.len() + candidate.blocked_cells.len(),
            candidate.bbox.volume(),
        ))
    });
    let reserved_cells = candidates
        .iter()
        .map(|candidate| {
            // Empty routing terminals are connection sites rather than solid
            // package cells. Keeping them out of collision occupancy lets a
            // producer terminal and its consumer terminal coincide, which is
            // the compact equivalent of a zero-length inter-cluster route.
            let mut cells = candidate
                .world
                .iter_block()
                .into_iter()
                .map(|(position, block)| (position, block.kind.is_cobble()))
                .collect::<BTreeMap<_, _>>();
            for position in candidate.ports.iter().flat_map(|port| {
                std::iter::once(port.position).chain(port.routing_access_positions())
            }) {
                cells.insert(position, false);
            }
            for &position in &candidate.blocked_cells {
                cells.entry(position).or_insert(false);
            }
            let mut cells = cells
                .into_iter()
                .map(|(position, shareable_cobble)| {
                    (
                        position.0.checked_sub(candidate.bbox.min.0),
                        position.1.checked_sub(candidate.bbox.min.1),
                        position.2.checked_sub(candidate.bbox.min.2),
                        shareable_cobble,
                    )
                })
                .filter_map(|(x, y, z, shareable_cobble)| Some((x?, y?, z?, shareable_cobble)))
                .collect::<Vec<_>>();
            cells.sort_unstable();
            cells.dedup();
            cells
        })
        .collect::<Vec<_>>();
    fn search(
        candidates: &[LayoutCandidate],
        reserved_cells: &[Vec<(usize, usize, usize, bool)>],
        order: &[usize],
        cursor: usize,
        depth_limit: usize,
        height_limit: usize,
        width_limit: usize,
        occupied: &mut [Option<bool>],
        positions: &mut [Option<(usize, usize, usize)>],
        layouts: &mut Vec<Vec<(usize, usize, usize)>>,
        max_layouts: usize,
        scan_variant: usize,
    ) {
        if layouts.len() >= max_layouts {
            return;
        }
        if cursor == order.len() {
            layouts.push(positions.iter().map(|position| position.unwrap()).collect());
            return;
        }

        let candidate_index = order[cursor];
        let width = candidates[candidate_index].bbox.width();
        let depth = candidates[candidate_index].bbox.depth();
        let height = candidates[candidate_index].bbox.height();
        if width > width_limit || depth > depth_limit || height > height_limit {
            return;
        }
        let mut zs = (0..=height_limit - height).collect::<Vec<_>>();
        let mut ys = (0..=depth_limit - depth).collect::<Vec<_>>();
        let mut xs = (0..=width_limit - width).collect::<Vec<_>>();
        if scan_variant & 1 != 0 {
            zs.reverse();
        }
        if scan_variant & 2 != 0 {
            ys.reverse();
        }
        if scan_variant & 4 != 0 {
            xs.reverse();
        }
        for z in zs {
            for &y in &ys {
                for &x in &xs {
                    let cells = reserved_cells[candidate_index]
                        .iter()
                        .filter_map(|&(local_x, local_y, local_z, shareable_cobble)| {
                            let world_x = x + local_x;
                            let world_y = y + local_y;
                            let world_z = z + local_z;
                            (world_x < width_limit
                                && world_y < depth_limit
                                && world_z < height_limit)
                                .then_some((
                                    (world_z * depth_limit + world_y) * width_limit + world_x,
                                    shareable_cobble,
                                ))
                        })
                        .collect::<Vec<_>>();
                    if cells.len() != reserved_cells[candidate_index].len() {
                        continue;
                    }
                    if cells.iter().any(|&(cell, shareable_cobble)| {
                        occupied[cell]
                            .is_some_and(|existing_cobble| !(existing_cobble && shareable_cobble))
                    }) {
                        continue;
                    }
                    let newly_occupied = cells
                        .iter()
                        .filter_map(|&(cell, shareable_cobble)| {
                            if occupied[cell].is_none() {
                                occupied[cell] = Some(shareable_cobble);
                                Some(cell)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    positions[candidate_index] = Some((x, y, z));
                    search(
                        candidates,
                        reserved_cells,
                        order,
                        cursor + 1,
                        depth_limit,
                        height_limit,
                        width_limit,
                        occupied,
                        positions,
                        layouts,
                        max_layouts,
                        scan_variant,
                    );
                    positions[candidate_index] = None;
                    for cell in newly_occupied {
                        occupied[cell] = None;
                    }
                }
            }
        }
    }

    let per_variant = max_layouts.div_ceil(8);
    let mut packed = Vec::new();
    for scan_variant in 0..8 {
        let mut occupied = vec![
            None;
            width_limit
                .saturating_mul(depth_limit)
                .saturating_mul(height_limit)
        ];
        let mut positions = vec![None; candidates.len()];
        let mut variant_layouts = Vec::new();
        search(
            candidates,
            &reserved_cells,
            &order,
            0,
            depth_limit,
            height_limit,
            width_limit,
            &mut occupied,
            &mut positions,
            &mut variant_layouts,
            per_variant,
            scan_variant,
        );
        packed.extend(variant_layouts);
    }
    packed.sort();
    packed.dedup();
    packed.truncate(max_layouts);
    tracing::debug!(
        dimensions = ?candidates
            .iter()
            .map(|candidate| (
                candidate.bbox.width(),
                candidate.bbox.depth(),
                candidate.bbox.height()
            ))
            .collect::<Vec<_>>(),
        width_limit,
        depth_limit,
        height_limit,
        layouts = packed.len(),
        "cluster cell packing completed"
    );
    let mut layouts = Vec::with_capacity(packed.len());
    for positions in packed {
        layouts.push(
            candidates
                .iter()
                .enumerate()
                .map(|(candidate_index, candidate)| {
                    let (x, y, z) = positions[candidate_index];
                    PlacedModule {
                        module_name: candidate.module_name.clone(),
                        candidate_index,
                        origin: crate::world::position::Position(4 + x, 4 + y, floor_margin + z),
                        bbox: candidate.bbox,
                    }
                })
                .collect(),
        );
    }
    layouts
}

fn support_cannot_relay_power(
    world: &World3D,
    support: Position,
    driving_signal: Position,
) -> bool {
    world.iter_block().into_iter().all(|(position, block)| {
        position == support
            || position == driving_signal
            || (!block.kind.is_redstone()
                && !block.kind.is_torch()
                && !block.kind.is_repeater()
                && !block.kind.is_switch()
                && !matches!(block.kind, crate::world::block::BlockKind::RedstoneBlock))
            || !crate::transform::place_and_route::detailed_router::target_powers_position(
                world, support, position,
            )
    })
}

fn candidate_signals_are_isolated(
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) -> eyre::Result<bool> {
    let world = assemble_world(candidates, placed, &[])?;
    let mut blocks = Vec::new();
    let mut boundary_ports = HashMap::<Position, Vec<(String, PhysicalPortDirection)>>::new();
    for (owner, module) in placed.iter().enumerate() {
        let candidate = candidates
            .get(module.candidate_index)
            .with_context(|| format!("missing clustered candidate {}", module.candidate_index))?;
        blocks.extend(
            candidate
                .world
                .iter_block()
                .into_iter()
                .map(|(position, block)| {
                    (
                        owner,
                        Position(
                            module.origin.0 + position.0 - candidate.bbox.min.0,
                            module.origin.1 + position.1 - candidate.bbox.min.1,
                            module.origin.2 + position.2 - candidate.bbox.min.2,
                        ),
                        block,
                    )
                }),
        );
        for port in &candidate.ports {
            for local in port.routing_access_positions() {
                let position = Position(
                    module.origin.0 + local.0 - candidate.bbox.min.0,
                    module.origin.1 + local.1 - candidate.bbox.min.1,
                    module.origin.2 + local.2 - candidate.bbox.min.2,
                );
                boundary_ports
                    .entry(position)
                    .or_default()
                    .push((port.name.clone(), port.direction.clone()));
            }
        }
    }

    for left in 0..blocks.len() {
        for right in left + 1..blocks.len() {
            let (left_owner, left_position, left_block) = blocks[left];
            let (right_owner, right_position, right_block) = blocks[right];
            if left_owner == right_owner || left_position == right_position {
                continue;
            }
            let intended_boundary_contact = boundary_ports
                .get(&left_position)
                .into_iter()
                .flatten()
                .any(|(left_name, left_direction)| {
                    boundary_ports
                        .get(&right_position)
                        .into_iter()
                        .flatten()
                        .any(|(right_name, right_direction)| {
                            left_name == right_name && left_direction != right_direction
                        })
                });
            if intended_boundary_contact {
                continue;
            }
            let left_is_signal = left_block.kind.is_redstone()
                || left_block.kind.is_torch()
                || left_block.kind.is_repeater()
                || left_block.kind.is_switch()
                || matches!(
                    left_block.kind,
                    crate::world::block::BlockKind::RedstoneBlock
                );
            let right_is_signal = right_block.kind.is_redstone()
                || right_block.kind.is_torch()
                || right_block.kind.is_repeater()
                || right_block.kind.is_switch()
                || matches!(
                    right_block.kind,
                    crate::world::block::BlockKind::RedstoneBlock
                );
            let left_powers_right = left_is_signal
                && crate::transform::place_and_route::detailed_router::target_powers_position(
                    &world,
                    left_position,
                    right_position,
                );
            let right_powers_left = right_is_signal
                && crate::transform::place_and_route::detailed_router::target_powers_position(
                    &world,
                    right_position,
                    left_position,
                );
            let left_edge_is_harmless_support_power = left_powers_right
                && right_block.kind.is_cobble()
                && support_cannot_relay_power(&world, right_position, left_position);
            let right_edge_is_harmless_support_power = right_powers_left
                && left_block.kind.is_cobble()
                && support_cannot_relay_power(&world, left_position, right_position);
            if (left_powers_right && !left_edge_is_harmless_support_power)
                || (right_powers_left && !right_edge_is_harmless_support_power)
            {
                static LOGGED_COUPLING: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !LOGGED_COUPLING.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::debug!(
                        left_owner,
                        ?left_position,
                        left_kind = ?left_block.kind,
                        left_ports = ?boundary_ports.get(&left_position),
                        right_owner,
                        ?right_position,
                        right_kind = ?right_block.kind,
                        right_ports = ?boundary_ports.get(&right_position),
                        "rejecting unintended inter-cluster power edge"
                    );
                }
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn trace_clustered_failure_state(
    placed_world: &PlacedWorld,
    routes: &[super::super::router::RoutedNet],
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) {
    let world = World::from(&placed_world.world);
    let Ok(mut sim) = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0) else {
        return;
    };
    let changes = placed_world
        .inputs
        .iter()
        .map(|endpoint| (endpoint.position(), endpoint.name == "a"))
        .collect::<Vec<_>>();
    if sim.change_state_with_limits(changes, 256, 50_000).is_err() {
        return;
    }
    let top_input_routes = routes
        .iter()
        .filter(|route| {
            route
                .source_label
                .as_deref()
                .is_some_and(|label| !label.contains('.'))
        })
        .cloned()
        .collect::<Vec<_>>();
    let base_cluster_port_states = assemble_world(candidates, placed, &top_input_routes)
        .ok()
        .and_then(|world| {
            let world = World::from(&world);
            let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0).ok()?;
            let changes = placed_world
                .inputs
                .iter()
                .map(|endpoint| (endpoint.position(), endpoint.name == "a"))
                .collect::<Vec<_>>();
            sim.change_state_with_limits(changes, 256, 50_000).ok()?;
            Some(cluster_port_power_states(&sim, candidates, placed))
        });
    tracing::debug!(
        route_states = ?routes
            .iter()
            .map(|route| (
                route.source_label.as_deref(),
                route.sink_label.as_deref(),
                route.source,
                sim.world()[route.source].kind.is_powered(),
                route.sink,
                sim.world()[route.sink].kind.is_powered(),
                route
                    .required_powered_positions
                    .iter()
                    .map(|position| (*position, sim.world()[*position].kind.is_powered()))
                    .collect::<Vec<_>>(),
            ))
            .collect::<Vec<_>>(),
        output_states = ?placed_world
            .outputs
            .iter()
            .map(|endpoint| (
                endpoint.name.as_str(),
                endpoint.position(),
                sim.world()[endpoint.position()].kind.is_powered(),
            ))
            .collect::<Vec<_>>(),
        cluster_port_states = ?cluster_port_power_states(&sim, candidates, placed),
        base_cluster_port_states = ?base_cluster_port_states,
        "clustered failure state for a=1, b=0, cin=0"
    );
}

fn cluster_port_power_states<'a>(
    sim: &Simulator,
    candidates: &'a [LayoutCandidate],
    placed: &'a [PlacedModule],
) -> Vec<(
    &'a str,
    Vec<(
        &'a str,
        PhysicalPortDirection,
        Position,
        crate::world::block::BlockKind,
        bool,
    )>,
)> {
    placed
        .iter()
        .filter_map(|module| {
            let candidate = candidates.get(module.candidate_index)?;
            Some((
                module.module_name.as_str(),
                candidate
                    .ports
                    .iter()
                    .map(|port| {
                        let local = port.primary_route_position();
                        let position = Position(
                            module.origin.0 + local.0 - candidate.bbox.min.0,
                            module.origin.1 + local.1 - candidate.bbox.min.1,
                            module.origin.2 + local.2 - candidate.bbox.min.2,
                        );
                        (
                            port.name.as_str(),
                            port.direction.clone(),
                            position,
                            sim.world()[position].kind,
                            sim.world()[position].kind.is_powered(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ))
        })
        .collect()
}

fn estimated_cluster_connection_cost(
    topology: &ResolvedPnrTopology,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) -> (usize, usize, usize) {
    let world = assemble_world(candidates, placed, &[]).ok();
    let mut disconnected = 0usize;
    let mut longest = 0usize;
    let mut total = 0usize;
    for net in &topology.nets {
        let Some(source) = cluster_endpoint_position(topology, &net.driver, candidates, placed)
        else {
            continue;
        };
        for sink in &net.sinks {
            let Some(sink) = cluster_endpoint_position(topology, sink, candidates, placed) else {
                continue;
            };
            let distance = source.manhattan_distance(&sink);
            total = total.saturating_add(distance);
            longest = longest.max(distance);
            let directly_connected = source == sink
                || world.as_ref().is_some_and(|world| {
                    world.size.bound_on(source)
                    && world.size.bound_on(sink)
                    && crate::transform::place_and_route::detailed_router::target_powers_position(
                        world, source, sink,
                    )
                });
            if !directly_connected {
                disconnected += 1;
            }
        }
    }
    (disconnected, longest, total)
}

fn estimated_cluster_net_length(
    topology: &ResolvedPnrTopology,
    net: &crate::transform::place_and_route::global_pnr::topology::ResolvedNet,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) -> Option<usize> {
    let source = cluster_endpoint_position(topology, &net.driver, candidates, placed)?;
    Some(
        net.sinks
            .iter()
            .filter_map(|sink| cluster_endpoint_position(topology, sink, candidates, placed))
            .map(|sink| source.manhattan_distance(&sink))
            .sum(),
    )
}

fn cluster_endpoint_position(
    topology: &ResolvedPnrTopology,
    endpoint: &ResolvedEndpoint,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
) -> Option<crate::world::position::Position> {
    let ResolvedEndpoint::InstancePort { instance, port } = endpoint else {
        return None;
    };
    let instance = topology.instances.get(instance.0)?;
    let port_name = &topology.port(*port)?.name;
    let placed = placed
        .iter()
        .find(|module| module.module_name == instance.display_name)?;
    let candidate = candidates.get(placed.candidate_index)?;
    let port = candidate
        .ports
        .iter()
        .find(|port| port.name == *port_name)?;
    let position = port.primary_route_position();
    Some(crate::world::position::Position(
        placed.origin.0 + position.0 - candidate.bbox.min.0,
        placed.origin.1 + position.1 - candidate.bbox.min.1,
        placed.origin.2 + position.2 - candidate.bbox.min.2,
    ))
}

/// Enumerate a bounded Cartesian portfolio in increasing total candidate rank.
///
/// Trying only one changed cluster at a time misses compositions where two
/// adjacent macros both need a different pin geometry. Ranking by the sum of
/// each selected candidate's index preserves the cheap all-best and
/// single-substitution trials first, while still reaching those coordinated
/// substitutions without materializing the full Cartesian product.
fn cluster_candidate_combinations(
    candidate_sets: &[Vec<LayoutCandidate>],
    max_combinations: usize,
) -> Vec<Vec<LayoutCandidate>> {
    if max_combinations == 0 || candidate_sets.iter().any(Vec::is_empty) {
        return Vec::new();
    }

    fn enumerate_rank_sum(
        candidate_sets: &[Vec<LayoutCandidate>],
        set_index: usize,
        remaining_rank: usize,
        selected: &mut Vec<usize>,
        combinations: &mut Vec<Vec<LayoutCandidate>>,
        max_combinations: usize,
    ) {
        if combinations.len() >= max_combinations {
            return;
        }
        if set_index == candidate_sets.len() {
            if remaining_rank == 0 {
                combinations.push(
                    selected
                        .iter()
                        .enumerate()
                        .map(|(index, &candidate_index)| {
                            candidate_sets[index][candidate_index].clone()
                        })
                        .collect(),
                );
            }
            return;
        }

        let max_rank = remaining_rank.min(candidate_sets[set_index].len() - 1);
        for rank in 0..=max_rank {
            selected.push(rank);
            enumerate_rank_sum(
                candidate_sets,
                set_index + 1,
                remaining_rank - rank,
                selected,
                combinations,
                max_combinations,
            );
            selected.pop();
            if combinations.len() >= max_combinations {
                return;
            }
        }
    }

    let maximum_rank_sum = candidate_sets.iter().map(|set| set.len() - 1).sum();
    let mut combinations = Vec::with_capacity(max_combinations);
    let mut selected = Vec::with_capacity(candidate_sets.len());
    for rank_sum in 0..=maximum_rank_sum {
        enumerate_rank_sum(
            candidate_sets,
            0,
            rank_sum,
            &mut selected,
            &mut combinations,
            max_combinations,
        );
        if combinations.len() >= max_combinations {
            break;
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
    use crate::transform::place_and_route::estimate::BoundingBox;
    use crate::transform::place_and_route::global_pnr::ir::LayoutCandidateCost;
    use crate::transform::place_and_route::local_placer::{
        NotRouteStrategy, PlacementSamplingPolicy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

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

    fn named_candidate(name: &str) -> LayoutCandidate {
        LayoutCandidate {
            module_name: name.to_owned(),
            world: World3D::new(DimSize(1, 1, 1)),
            bbox: BoundingBox {
                min: Position(0, 0, 0),
                max: Position(0, 0, 0),
            },
            ports: Vec::new(),
            occupied_cells: HashSet::new(),
            blocked_cells: HashSet::new(),
            cost: LayoutCandidateCost::default(),
        }
    }

    #[test]
    fn cluster_candidate_combinations_include_coordinated_alternatives() {
        let candidate_sets = vec![
            vec![named_candidate("a0"), named_candidate("a1")],
            vec![named_candidate("b0"), named_candidate("b1")],
            vec![named_candidate("c0"), named_candidate("c1")],
        ];

        let combinations = cluster_candidate_combinations(&candidate_sets, 8);
        let names = combinations
            .iter()
            .map(|combination| {
                combination
                    .iter()
                    .map(|candidate| candidate.module_name.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(names.len(), 8);
        assert_eq!(names[0], vec!["a0", "b0", "c0"]);
        assert!(names.contains(&vec!["a1", "b1", "c0"]));
        assert!(names.contains(&vec!["a1", "b1", "c1"]));
    }

    #[test]
    fn full_adder_plan_partitions_every_logic_node_with_explicit_boundaries() -> eyre::Result<()> {
        let graph = predefined_logics::buffered_full_adder_graph()?
            .prepare_place()?
            .graph;
        let ports = graph_ports(&graph);
        let plan = CombinationalClusterPlan::build(
            "full_adder",
            &graph,
            &ports,
            &ClusteringSpec::default(),
        )?
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
        let plan = CombinationalClusterPlan::build(
            "nor",
            &graph,
            &ports,
            &ClusteringSpec {
                max_logic_nodes: 1,
                ..Default::default()
            },
        )?
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

    #[test]
    fn disabled_clustering_skips_large_combinational_leaf() -> eyre::Result<()> {
        let graph = predefined_logics::buffered_full_adder_graph()?
            .prepare_place()?
            .graph;
        let ports = graph_ports(&graph);
        let config = UnitCandidateConfig {
            clustering: ClusteringSpec {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(try_generate_clustered_candidates(
            "full_adder",
            &graph,
            &ports,
            &config,
            None,
            CandidateInputMode::ExternalPorts,
        )?
        .is_none());
        Ok(())
    }
}
