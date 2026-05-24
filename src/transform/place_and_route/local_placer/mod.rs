use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use eyre::ensure;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use itertools::{iproduct, Itertools};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::place_bound::PlaceBound;
use super::placed_node::PlacedNode;
use super::sampling::SamplingPolicy;
use crate::graph::logic::LogicGraph;
use crate::graph::{GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::LogicType;
use crate::sequential::layout::SequentialMacro;
use crate::sequential::{SequentialPrimitive, SequentialType};
use crate::transform::place_and_route::estimate::{bounding_box_of_positions, world_compact_cost};
use crate::transform::place_and_route::place_bound::PropagateType;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::World3D;

mod config;
mod debug;
mod isolation;
mod state;

pub use config::{
    InputPlacementStrategy, LocalPlacerConfig, NotRouteStrategy, PlacementSamplingPolicy,
    TorchPlacementStrategy, K_MAX_LOCAL_PLACE_NODE_COUNT,
};
pub use debug::{LocalPlacerDebug, RouteDebug, RouteDepthDebug, RouteRejectReason, StepDebug};
use isolation::RouteIsolation;
use state::PlacementState;

mod routing;
use routing::*;
mod sequential;
use sequential::{
    generate_d_latch_gate_routes, generate_rs_latch_gate_routes, generate_sequential_macro_routes,
    supports_sequential_primitive,
};
#[cfg(test)]
use sequential::{
    generate_rs_latch_not_pairs, place_sequential_macro, route_rs_latch_branches,
    route_sequential_inputs, select_rs_latch_not_pairs, D_LATCH_NOT_D_NODE_ID,
    D_LATCH_RESET_NODE_ID, D_LATCH_RESET_OR_NODE_ID, D_LATCH_SET_NODE_ID,
    D_LATCH_SET_NOT_EN_NODE_ID, D_LATCH_SET_OR_NODE_ID,
};

pub struct LocalPlacer {
    graph: LogicGraph,
    config: LocalPlacerConfig,
    visit_orders: Vec<GraphNodeId>,
    cost_join_pairs_by_step: Vec<Vec<FutureJoinPair>>,
}

type PlacerQueue = Vec<(World3D, PlacementState)>;
impl LocalPlacer {
    pub fn new(graph: LogicGraph, config: LocalPlacerConfig) -> eyre::Result<Self> {
        let visit_orders = graph.topological_order();
        let cost_join_pairs_by_step = build_cost_join_pairs_by_step(&graph, &visit_orders);
        let result = Self {
            graph,
            config,
            visit_orders,
            cost_join_pairs_by_step,
        };
        result.verify()?;
        Ok(result)
    }

    fn verify(&self) -> eyre::Result<()> {
        ensure!(!self.graph.nodes.is_empty(), "");
        ensure!(
            self.graph.nodes.len() <= K_MAX_LOCAL_PLACE_NODE_COUNT,
            "too large graph"
        );

        for node_id in &self.graph.nodes {
            let kind = &self.graph.find_node_by_id(node_id.id).unwrap().kind;
            ensure!(
                kind.is_input() || kind.is_output() || kind.is_logic() || kind.is_sequential(),
                "cannot place"
            );
            if let Some(logic) = kind.as_logic() {
                ensure!(logic.is_not() || logic.is_or(), "cannot place");
            }
            if let Some(sequential) = kind.as_sequential() {
                ensure!(
                    supports_sequential_primitive(sequential),
                    "sequential primitive placement is not implemented"
                );
                ensure!(
                    sequential.output_ports.len() == 1,
                    "local placer currently supports only one exposed sequential output"
                );
            }
        }

        Ok(())
    }

    #[allow(clippy::vec_init_then_push)]
    pub fn generate(&self, dim: DimSize, finish_step: Option<usize>) -> Vec<World3D> {
        self.generate_inner(dim, finish_step, None)
    }

    pub fn generate_with_debug(
        &self,
        dim: DimSize,
        finish_step: Option<usize>,
        debug: &mut LocalPlacerDebug,
    ) -> Vec<World3D> {
        self.generate_inner(dim, finish_step, Some(debug))
    }

    fn generate_inner(
        &self,
        dim: DimSize,
        finish_step: Option<usize>,
        debug: Option<&mut LocalPlacerDebug>,
    ) -> Vec<World3D> {
        self.generate_queue(dim, finish_step, debug)
            .into_iter()
            .map(|(world, _)| world)
            .collect()
    }

    fn generate_queue(
        &self,
        dim: DimSize,
        finish_step: Option<usize>,
        mut debug: Option<&mut LocalPlacerDebug>,
    ) -> PlacerQueue {
        tracing::info!("generate starts");

        let mut queue = PlacerQueue::new();
        queue.push((World3D::new(dim), Default::default()));

        let mut step = 0;
        while step < self.visit_orders.len() && Some(step) != finish_step {
            let prev_len = queue.len();
            let result = self.do_step(step, queue);
            let next_len = result.queue.len();

            let compacted = self.compact_queue_after_step(step, result.queue);
            let compacted_len = compacted.len();
            queue = self.sample(step, compacted);
            let sampled_len = queue.len();
            if let Some(debug) = debug.as_deref_mut() {
                let mut step_debug = result.debug;
                step_debug.sampled_len = sampled_len;
                debug.steps.push(step_debug);
            }

            step += 1;
            tracing::info!(
                "from {prev_len} -> generated {next_len} -> compacted {compacted_len} -> sampled {sampled_len}"
            );
        }

        tracing::info!("generate complete");
        queue
    }

    fn do_step(&self, step: usize, queue: PlacerQueue) -> StepResult {
        let node = self.graph.find_node_by_id(self.visit_orders[step]).unwrap();
        tracing::info!("[{}/{}] {node}", step + 1, self.visit_orders.len());

        let input_positions = queue
            .first()
            .map(|(_, state)| {
                node.inputs
                    .iter()
                    .filter_map(|id| state.node_position(*id).map(|pos| (*id, pos)))
                    .collect_vec()
            })
            .unwrap_or_default();
        let input_queue_len = queue.len();

        let generated = queue
            .into_par_iter()
            .panic_fuse()
            .progress_with_style(progress_style(step + 1, self.visit_orders.len()))
            .map(|(world, state)| {
                let generation = self.generate_place_and_route(node, world, &state);
                (generation.items, generation.route_debug)
            })
            .collect::<Vec<_>>();

        let mut next_queue = Vec::new();
        let mut route_debug = RouteDebug::default();
        let mut has_route_debug = false;
        for (items, debug) in generated {
            next_queue.extend(items);
            if let Some(route) = debug {
                has_route_debug = true;
                route_debug.merge(route);
            }
        }

        let step_debug = StepDebug {
            step,
            total_steps: self.visit_orders.len(),
            node_id: node.id,
            node_kind: format!("{:?}", node.kind),
            input_node_ids: node.inputs.clone(),
            input_positions,
            input_queue_len,
            generated_len: next_queue.len(),
            sampled_len: 0,
            route_debug: has_route_debug.then_some(route_debug),
        };

        StepResult {
            queue: next_queue,
            debug: step_debug,
        }
    }

    fn generate_place_and_route(
        &self,
        node: &GraphNode,
        world: World3D,
        state: &PlacementState,
    ) -> PlacementGeneration {
        let mut route_debug = None;
        let items = match node.kind {
            GraphNodeKind::Input(_) => input_node_kind()
                .into_iter()
                .flat_map(|kind| generate_inputs(&self.config, &world, kind))
                .map(|(world, position)| {
                    let mut state = state.clone();
                    state.set_node_position(node.id, position);
                    state.set_signal_net(node.id, [position], [position]);
                    (world, state)
                })
                .collect(),
            GraphNodeKind::Output(_) => vec![(world.clone(), state.clone())],
            GraphNodeKind::Logic(logic) => match logic.logic_type {
                LogicType::Not => not_node_kind()
                    .into_iter()
                    .flat_map(|kind| {
                        generate_place_and_routes(
                            &self.config,
                            &world,
                            state[&node.inputs[0]],
                            kind,
                        )
                    })
                    .map(|(world, position)| {
                        let mut state = state.clone();
                        state.set_node_position(node.id, position);
                        let input_node_ids = [node.inputs[0]].into_iter().collect();
                        let sources = state.signal_sources_for_nodes(&input_node_ids);
                        let footprint = [Some(position), position.walk(world[position].direction)]
                            .into_iter()
                            .flatten();
                        state.set_signal_net(node.id, footprint, sources);
                        (world, state)
                    })
                    .collect(),
                LogicType::Or => {
                    assert_eq!(node.inputs.len(), 2);
                    let input_a = state[&node.inputs[0]];
                    let input_b = state[&node.inputs[1]];
                    let input_node_ids = node.inputs.iter().copied().collect::<HashSet<_>>();
                    let input_sources = state.signal_sources_for_nodes(&input_node_ids);
                    let sealed_output_source_ids =
                        self.graph.externally_observable_output_source_ids();
                    let mut protected_positions =
                        state.signal_positions_for_nodes(&sealed_output_source_ids);
                    protected_positions.extend(state.endpoint_positions().into_iter().filter_map(
                        |(endpoint, position)| match endpoint {
                            state::PlacementEndpoint::Node(node_id)
                                if sealed_output_source_ids.contains(&node_id) =>
                            {
                                Some(position)
                            }
                            _ => None,
                        },
                    ));
                    let isolation =
                        RouteIsolation::new(&world, [input_a, input_b], protected_positions);
                    let signal_nets = state.signal_nets();
                    let result = generate_or_routes_with_signal_nets(
                        &self.config,
                        &world,
                        input_a,
                        input_b,
                        &signal_nets,
                    );
                    route_debug = Some(result.debug);
                    result
                        .routes
                        .into_iter()
                        .flat_map(|route| {
                            let _shadow_accepts_sources =
                                route.impact.accepts_sources(&input_sources);
                            // Keep the OR tap on the terminal redstone where both inputs
                            // have joined. Source or mid-route taps can see only one input.
                            let positions = Some(route.footprint.terminal)
                                .filter(|position| {
                                    route.world[*position].kind.is_redstone()
                                        && isolation.accepts_or_route(&route.world, &route.path)
                                })
                                .into_iter()
                                .collect_vec();
                            positions
                                .into_iter()
                                .map(|position| {
                                    let mut state = state.clone();
                                    state.set_node_position(node.id, position);
                                    let footprint =
                                        [Some(position), position.down()].into_iter().flatten();
                                    state.set_signal_net(node.id, footprint, input_sources.clone());
                                    (route.world.clone(), state)
                                })
                                .collect_vec()
                        })
                        .collect()
                }
                _ => unreachable!(),
            },
            GraphNodeKind::Sequential(ref sequential) => match sequential.sequential_type {
                SequentialType::RsLatch if sequential.rs_latch_core().is_some() => {
                    generate_rs_latch_gate_routes(&self.config, node, sequential, &world, state)
                }
                SequentialType::DLatch if sequential.rs_latch_core().is_some() => {
                    generate_d_latch_gate_routes(&self.config, node, sequential, &world, state)
                }
                _ => {
                    generate_sequential_macro_routes(&self.config, node, sequential, &world, state)
                }
            },
            _ => unreachable!(),
        };

        PlacementGeneration { items, route_debug }
    }

    fn sample(&self, step: usize, queue: PlacerQueue) -> PlacerQueue {
        match self.config.placement_sampling_policy {
            PlacementSamplingPolicy::StepPolicy => {
                if self.config.leak_sampling && queue.len() > 10_000 {
                    // TODO: deallocate on other thread
                    let leak = Box::leak(Box::new(queue));
                    self.config
                        .step_sampling_policy
                        .sample_with_taking_seed(leak, self.config.sampling_seed(1, step))
                } else {
                    self.config
                        .step_sampling_policy
                        .sample_with_seed(queue, self.config.sampling_seed(1, step))
                }
            }
            PlacementSamplingPolicy::Cost {
                count,
                random_count,
                start_step,
            } => {
                if step < start_step {
                    self.config
                        .step_sampling_policy
                        .sample_with_seed(queue, self.config.sampling_seed(1, step))
                } else {
                    self.sample_by_cost(step, queue, count, random_count)
                }
            }
            PlacementSamplingPolicy::Ranked {
                count,
                random_count,
                start_step,
            } => {
                if step < start_step {
                    self.config
                        .step_sampling_policy
                        .sample_with_seed(queue, self.config.sampling_seed(1, step))
                } else {
                    self.sample_by_ranked(step, queue, count, random_count)
                }
            }
        }
    }

    // OR 라우트는 같은 블록 배치에서도 route 위의 여러 tap 위치를
    // PlacementState에 남길 수 있다. 이후 실제 배치될 노드가 그 위치에서
    // 다시 라우팅해야 할 때만 tap 차이를 보존하고, 아니면 같은 후보로 본다.
    fn compact_queue_after_step(&self, step: usize, queue: PlacerQueue) -> PlacerQueue {
        let mut queue = queue;
        let live_node_ids = self
            .visit_orders
            .iter()
            .skip(step + 1)
            .filter_map(|node_id| self.graph.find_node_by_id(*node_id))
            .filter(|node| !matches!(node.kind, GraphNodeKind::Output(_)))
            .flat_map(|node| node.inputs.iter().copied())
            .chain(self.graph.externally_observable_output_source_ids())
            .collect::<HashSet<_>>();

        for (_, state) in &mut queue {
            state.retain_nodes(&live_node_ids);
        }

        queue
            .into_iter()
            .unique_by(|(world, state)| {
                (
                    world.iter_block(),
                    state.endpoint_positions(),
                    state.signal_footprints(),
                )
            })
            .collect()
    }

    fn sample_by_cost(
        &self,
        step: usize,
        queue: PlacerQueue,
        count: usize,
        random_count: usize,
    ) -> PlacerQueue {
        if queue.len() <= count + random_count {
            return queue;
        }

        let mut scored = queue
            .into_iter()
            .map(|item| (self.placement_cost(step, &item.0, &item.1), item))
            .collect_vec();

        let best_count = count.min(scored.len());
        if best_count < scored.len() {
            scored.select_nth_unstable_by_key(best_count, |(cost, _)| *cost);
        }

        let rest = scored.split_off(best_count);
        let mut best = scored.into_iter().map(|(_, item)| item).collect_vec();
        let rest = rest.into_iter().map(|(_, item)| item).collect_vec();
        best.extend(
            SamplingPolicy::Random(random_count)
                .sample_with_seed(rest, self.config.sampling_seed(2, step)),
        );
        best
    }

    // 확장된 placement 후보들을 beam search처럼 줄인다. cost sampling과 같은
    // heuristic cost를 쓰되, beam 일부는 기하적으로 다른 후보를 위해 남긴다.
    // 이렇게 해야 compact하지만 거의 같은 후보들만 살아남아 후반 join/fanout
    // route를 막는 상황을 줄일 수 있다.
    fn sample_by_ranked(
        &self,
        step: usize,
        queue: PlacerQueue,
        count: usize,
        random_count: usize,
    ) -> PlacerQueue {
        if queue.len() <= count + random_count {
            return queue;
        }

        // 확장된 모든 후보에 점수를 매긴다. `index`는 deterministic tie-breaker라서,
        // cost가 같은 후보들은 기존 queue 순서를 유지한다.
        let mut scored = queue
            .into_iter()
            .enumerate()
            .map(|(index, item)| ScoredPlacement {
                cost: self.placement_cost(step, &item.0, &item.1),
                // diversity 보존용 coarse geometry bucket이다. 완전한 placement
                // identity가 아니므로 semantic deduplication 용도로 쓰면 안 된다.
                signature: bounding_box_of_positions(item.1.node_positions())
                    .map(|bounds| bounds.manhattan_span() / 4)
                    .unwrap_or_default(),
                index,
                item,
            })
            .collect_vec();
        scored.sort_unstable_by_key(|item| (item.cost, item.index));

        // beam 대부분은 cost가 낮은 후보로 채우고, 25%는 diversity bucket용으로
        // 남긴다. 25% 비율은 heuristic이므로, 컴포넌트별 튜닝이 필요해지면
        // config knob으로 드러내는 것이 좋다.
        let ranked_count = count.min(scored.len());
        let diversity_count = if ranked_count > 1 {
            (ranked_count / 4).max(1)
        } else {
            0
        };
        let best_count = ranked_count.saturating_sub(diversity_count);

        let rest = scored.split_off(best_count);
        let mut selected = scored;
        let mut selected_signatures = selected
            .iter()
            .map(|item| item.signature)
            .collect::<HashSet<_>>();
        let mut overflow = Vec::new();

        // 남겨둔 beam slot에는 아직 선택되지 않은 geometry signature별 최저 cost
        // 후보를 먼저 넣고, 부족하면 일반 cost 순서 후보로 채운다.
        for item in rest {
            if selected.len() < ranked_count && selected_signatures.insert(item.signature) {
                selected.push(item);
            } else {
                overflow.push(item);
            }
        }

        // distinct signature 개수가 남겨둔 slot보다 적으면, 남은 beam은 overflow에서
        // cost 순서대로 채운다.
        let mut overflow = overflow.into_iter();
        while selected.len() < ranked_count {
            let Some(item) = overflow.next() else {
                break;
            };
            selected.push(item);
        }

        // deterministic beam 선택 뒤에 작은 stochastic tail을 붙인다. diversity
        // 보존과는 별개로, deterministic ranking이 나쁜 방향으로 굳었을 때 빠져나갈
        // 기회를 남기는 용도다.
        let mut best = selected.into_iter().map(|item| item.item).collect_vec();
        let random_pool = overflow.map(|item| item.item).collect_vec();
        best.extend(
            SamplingPolicy::Random(random_count)
                .sample_with_seed(random_pool, self.config.sampling_seed(2, step)),
        );
        best
    }

    fn placement_cost(&self, step: usize, world: &World3D, state: &PlacementState) -> usize {
        let current_node_id = self.visit_orders[step];
        let mut cost = world_compact_cost(world);

        if let Some(position) = state.node_position(current_node_id) {
            cost += local_density(world, position) * 3;
        }

        for pair in &self.cost_join_pairs_by_step[step] {
            let (Some(a), Some(b)) = (state.node_position(pair.a), state.node_position(pair.b))
            else {
                continue;
            };
            cost += a.manhattan_distance(&b) * pair.weight * 8;
        }

        cost
    }
}

struct ScoredPlacement {
    cost: usize,
    signature: usize,
    index: usize,
    item: (World3D, PlacementState),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FutureJoinPair {
    a: GraphNodeId,
    b: GraphNodeId,
    weight: usize,
}

fn build_cost_join_pairs_by_step(
    graph: &LogicGraph,
    visit_orders: &[GraphNodeId],
) -> Vec<Vec<FutureJoinPair>> {
    let mut order_index = HashMap::new();
    for (index, node_id) in visit_orders.iter().copied().enumerate() {
        order_index.insert(node_id, index);
    }

    visit_orders
        .iter()
        .enumerate()
        .map(|(step, _)| {
            let remaining_fanouts = remaining_fanout_counts_by_node(graph, &order_index, step);
            let mut pair_weights = HashMap::<(GraphNodeId, GraphNodeId), usize>::new();

            for node in graph
                .nodes
                .iter()
                .filter(|node| order_index[&node.id] > step)
            {
                for (a, b) in node
                    .inputs
                    .iter()
                    .copied()
                    .filter(|input| order_index.get(input).is_some_and(|index| *index <= step))
                    .tuple_combinations()
                {
                    let weight = 1
                        + remaining_fanouts
                            .get(&a)
                            .copied()
                            .unwrap_or_default()
                            .saturating_sub(1)
                        + remaining_fanouts
                            .get(&b)
                            .copied()
                            .unwrap_or_default()
                            .saturating_sub(1);
                    *pair_weights.entry((a, b)).or_default() += weight;
                }
            }

            pair_weights
                .into_iter()
                .map(|((a, b), weight)| FutureJoinPair { a, b, weight })
                .sorted_by_key(|pair| (pair.a, pair.b))
                .collect_vec()
        })
        .collect_vec()
}

fn remaining_fanout_counts_by_node(
    graph: &LogicGraph,
    order_index: &HashMap<GraphNodeId, usize>,
    step: usize,
) -> HashMap<GraphNodeId, usize> {
    let mut counts = HashMap::new();
    for node in graph
        .nodes
        .iter()
        .filter(|node| order_index[&node.id] > step)
    {
        for input in &node.inputs {
            if order_index.get(input).is_some_and(|index| *index <= step) {
                *counts.entry(*input).or_default() += 1;
            }
        }
    }
    counts
}

fn local_density(world: &World3D, position: Position) -> usize {
    position
        .forwards()
        .into_iter()
        .filter(|&pos| world.size.bound_on(pos))
        .filter(|&pos| !world[pos].kind.is_air())
        .count()
}

struct StepResult {
    queue: PlacerQueue,
    debug: StepDebug,
}

struct PlacementGeneration {
    items: PlacerQueue,
    route_debug: Option<RouteDebug>,
}

fn progress_style(step: usize, len: usize) -> ProgressStyle {
    ProgressStyle::with_template(
        &format!("{{spinner:.green}} [{{eta_precise}}] [{step}/{len}] [{{bar:40.cyan/blue}}] {{pos:>7}}/{{len:7}} {{msg}}"),
    )
    .unwrap()
    .progress_chars("#>-")
}

#[cfg(test)]
#[path = "tests.rs"]
mod unit_tests;

#[cfg(test)]
mod component_tests;
