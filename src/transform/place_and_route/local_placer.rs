use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ops::Index;

use eyre::ensure;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use itertools::{iproduct, Itertools};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::place_bound::PlaceBound;
use super::placed_node::PlacedNode;
use super::sampling::SamplingPolicy;
use crate::graph::logic::LogicGraph;
use crate::graph::world::WorldGraph;
use crate::graph::{GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::LogicType;
use crate::sequential::layout::SequentialMacro;
use crate::sequential::{SequentialPrimitive, SequentialType};
use crate::transform::place_and_route::estimate::world_compact_cost;
use crate::transform::place_and_route::place_bound::PropagateType;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

pub struct LocalPlacer {
    graph: LogicGraph,
    config: LocalPlacerConfig,
    visit_orders: Vec<GraphNodeId>,
    cost_join_pairs_by_step: Vec<Vec<(GraphNodeId, GraphNodeId)>>,
}

#[derive(Copy, Clone)]
pub struct LocalPlacerConfig {
    pub random_seed: u64,
    pub greedy_input_generation: bool,
    pub input_placement_strategy: InputPlacementStrategy,
    pub step_sampling_policy: SamplingPolicy,
    pub placement_sampling_policy: PlacementSamplingPolicy,
    // dealloc 시간을 줄이기 위해 generation들을 leak 시킨다
    pub leak_sampling: bool,
    // torch place시 input과 direct로 연결되도록 강제한다.
    pub route_torch_directly: bool,
    pub torch_placement_strategy: TorchPlacementStrategy,
    pub not_route_strategy: NotRouteStrategy,
    pub max_not_route_step: usize,
    pub not_route_step_sampling_policy: SamplingPolicy,
    // 최대 routing 거리를 지정한다.
    pub max_route_step: usize,
    pub route_step_sampling_policy: SamplingPolicy,
}

impl Default for LocalPlacerConfig {
    fn default() -> Self {
        Self {
            random_seed: 42,
            greedy_input_generation: false,
            input_placement_strategy: InputPlacementStrategy::default(),
            step_sampling_policy: SamplingPolicy::default(),
            placement_sampling_policy: PlacementSamplingPolicy::default(),
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::default(),
            not_route_strategy: NotRouteStrategy::default(),
            max_not_route_step: 0,
            not_route_step_sampling_policy: SamplingPolicy::default(),
            max_route_step: 0,
            route_step_sampling_policy: SamplingPolicy::default(),
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum InputPlacementStrategy {
    #[default]
    Boundary,
    Anywhere,
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum TorchPlacementStrategy {
    #[default]
    DirectOnly,
    AnywhereNonAdjacent,
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum NotRouteStrategy {
    #[default]
    DirectOnly,
    RedstoneOnly,
    DirectAndRedstone,
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum PlacementSamplingPolicy {
    #[default]
    StepPolicy,
    Cost {
        count: usize,
        random_count: usize,
        start_step: usize,
    },
}

impl LocalPlacerConfig {
    pub fn exhaustive(max_route_step: usize) -> Self {
        Self {
            random_seed: 42,
            greedy_input_generation: false,
            input_placement_strategy: InputPlacementStrategy::Anywhere,
            step_sampling_policy: SamplingPolicy::None,
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: false,
            torch_placement_strategy: TorchPlacementStrategy::AnywhereNonAdjacent,
            not_route_strategy: NotRouteStrategy::DirectAndRedstone,
            max_not_route_step: max_route_step,
            not_route_step_sampling_policy: SamplingPolicy::None,
            max_route_step,
            route_step_sampling_policy: SamplingPolicy::None,
        }
    }

    pub fn cost_sampling(
        count: usize,
        random_count: usize,
        start_step: usize,
    ) -> PlacementSamplingPolicy {
        PlacementSamplingPolicy::Cost {
            count,
            random_count,
            start_step,
        }
    }

    fn sampling_seed(self, scope: u64, step: usize) -> u64 {
        self.random_seed
            ^ scope.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (step as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
    }
}

pub const K_MAX_LOCAL_PLACE_NODE_COUNT: usize = 40;

type PlacerQueue = Vec<(World3D, PlacementState)>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PlacementEndpoint {
    Node(GraphNodeId),
    Port(GraphNodeId, String),
}

#[derive(Debug, Clone, Default)]
struct PlacementState {
    positions: HashMap<PlacementEndpoint, Position>,
}

impl PlacementState {
    fn node_position(&self, node_id: GraphNodeId) -> Option<Position> {
        self.positions
            .get(&PlacementEndpoint::Node(node_id))
            .copied()
    }

    #[allow(dead_code)]
    fn port_position(&self, node_id: GraphNodeId, port: &str) -> Option<Position> {
        self.positions
            .get(&PlacementEndpoint::Port(node_id, port.to_owned()))
            .copied()
    }

    fn set_node_position(&mut self, node_id: GraphNodeId, position: Position) {
        self.positions
            .insert(PlacementEndpoint::Node(node_id), position);
    }

    fn set_port_position(&mut self, node_id: GraphNodeId, port: String, position: Position) {
        self.positions
            .insert(PlacementEndpoint::Port(node_id, port), position);
    }
}

impl FromIterator<(GraphNodeId, Position)> for PlacementState {
    fn from_iter<T: IntoIterator<Item = (GraphNodeId, Position)>>(iter: T) -> Self {
        let mut state = PlacementState::default();
        for (node_id, position) in iter {
            state.set_node_position(node_id, position);
        }
        state
    }
}

impl Index<&GraphNodeId> for PlacementState {
    type Output = Position;

    fn index(&self, index: &GraphNodeId) -> &Self::Output {
        &self.positions[&PlacementEndpoint::Node(*index)]
    }
}

#[derive(Debug, Default)]
pub struct LocalPlacerDebug {
    pub steps: Vec<StepDebug>,
}

impl LocalPlacerDebug {
    pub fn print_summary(&self) {
        for step in &self.steps {
            println!(
                "[{}/{}] node={} kind={} inputs={:?} queue={} generated={} sampled={}",
                step.step + 1,
                step.total_steps,
                step.node_id,
                step.node_kind,
                step.input_node_ids,
                step.input_queue_len,
                step.generated_len,
                step.sampled_len,
            );
            if let Some(route) = &step.route_debug {
                println!(
                    "  routes={} candidates={} initial_states={} samples={:?}",
                    route.route_calls,
                    route.candidates_found,
                    route.initial_states,
                    route.route_samples
                );
                println!("  rejected={:?}", route.rejected);
                for depth in &route.depths {
                    println!(
                        "  depth {}: frontier={} bounds={} accepted={} next_pre_sample={} next_post_sample={}",
                        depth.depth,
                        depth.frontier_len,
                        depth.bounds_checked,
                        depth.accepted_routes,
                        depth.next_frontier_before_sampling,
                        depth.next_frontier_after_sampling,
                    );
                }
            }
        }
    }

    pub fn first_empty_step(&self) -> Option<&StepDebug> {
        self.steps
            .iter()
            .find(|step| step.input_queue_len > 0 && step.generated_len == 0)
    }
}

#[derive(Debug, Default)]
pub struct StepDebug {
    pub step: usize,
    pub total_steps: usize,
    pub node_id: GraphNodeId,
    pub node_kind: String,
    pub input_node_ids: Vec<GraphNodeId>,
    pub input_positions: Vec<(GraphNodeId, Position)>,
    pub input_queue_len: usize,
    pub generated_len: usize,
    pub sampled_len: usize,
    pub route_debug: Option<RouteDebug>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteRejectReason {
    InitInputTopCobbleConflict,
    InitOutputTopCobbleConflict,
    OutOfBounds,
    NoBottomForCobble,
    CobbleConflict,
    RedstoneConflict,
    ShortCircuit,
}

#[derive(Debug, Default, Clone)]
pub struct RouteDebug {
    pub route_calls: usize,
    pub route_samples: Vec<(Position, Position)>,
    pub initial_states: usize,
    pub candidates_found: usize,
    pub rejected: HashMap<RouteRejectReason, usize>,
    pub depths: Vec<RouteDepthDebug>,
}

impl RouteDebug {
    fn merge(&mut self, other: RouteDebug) {
        self.route_calls += other.route_calls;
        self.initial_states += other.initial_states;
        self.candidates_found += other.candidates_found;
        if self.route_samples.len() < 8 {
            let remaining = 8 - self.route_samples.len();
            self.route_samples
                .extend(other.route_samples.into_iter().take(remaining));
        }
        for (reason, count) in other.rejected {
            *self.rejected.entry(reason).or_default() += count;
        }
        if self.depths.len() < other.depths.len() {
            self.depths
                .resize_with(other.depths.len(), RouteDepthDebug::default);
        }
        for (depth, other_depth) in other.depths.into_iter().enumerate() {
            self.depths[depth].depth = depth;
            self.depths[depth].frontier_len += other_depth.frontier_len;
            self.depths[depth].bounds_checked += other_depth.bounds_checked;
            self.depths[depth].accepted_routes += other_depth.accepted_routes;
            self.depths[depth].next_frontier_before_sampling +=
                other_depth.next_frontier_before_sampling;
            self.depths[depth].next_frontier_after_sampling +=
                other_depth.next_frontier_after_sampling;
        }
    }

    fn reject(&mut self, reason: RouteRejectReason) {
        *self.rejected.entry(reason).or_default() += 1;
    }
}

#[derive(Debug, Default, Clone)]
pub struct RouteDepthDebug {
    pub depth: usize,
    pub frontier_len: usize,
    pub bounds_checked: usize,
    pub accepted_routes: usize,
    pub next_frontier_before_sampling: usize,
    pub next_frontier_after_sampling: usize,
}

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

            queue = self.sample(step, result.queue);
            let sampled_len = queue.len();
            if let Some(debug) = debug.as_deref_mut() {
                let mut step_debug = result.debug;
                step_debug.sampled_len = sampled_len;
                debug.steps.push(step_debug);
            }

            step += 1;
            tracing::info!("from {prev_len} -> generated {next_len} -> sampled {sampled_len}");
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
                    (world, state)
                })
                .collect(),
            GraphNodeKind::Output(_) => {
                let position = state[&node.inputs[0]];
                let mut state = state.clone();
                state.set_node_position(node.id, position);
                vec![(world.clone(), state)]
            }
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
                        (world, state)
                    })
                    .collect(),
                LogicType::Or => {
                    assert_eq!(node.inputs.len(), 2);
                    let result = generate_or_routes(
                        &self.config,
                        &world,
                        state[&node.inputs[0]],
                        state[&node.inputs[1]],
                    );
                    route_debug = Some(result.debug);
                    result
                        .routes
                        .into_iter()
                        .flat_map(|(world, positions)| {
                            positions
                                .into_iter()
                                .map(|position| {
                                    let mut state = state.clone();
                                    state.set_node_position(node.id, position);
                                    (world.clone(), state)
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
        }
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

    fn placement_cost(&self, step: usize, world: &World3D, state: &PlacementState) -> usize {
        let current_node_id = self.visit_orders[step];
        let mut cost = world_compact_cost(world);

        if let Some(position) = state.node_position(current_node_id) {
            cost += local_density(world, position) * 3;
        }

        for (a, b) in &self.cost_join_pairs_by_step[step] {
            let (Some(a), Some(b)) = (state.node_position(*a), state.node_position(*b)) else {
                continue;
            };
            cost += a.manhattan_distance(&b) * 8;
        }

        cost
    }
}

fn build_cost_join_pairs_by_step(
    graph: &LogicGraph,
    visit_orders: &[GraphNodeId],
) -> Vec<Vec<(GraphNodeId, GraphNodeId)>> {
    let mut order_index = HashMap::new();
    for (index, node_id) in visit_orders.iter().copied().enumerate() {
        order_index.insert(node_id, index);
    }

    visit_orders
        .iter()
        .enumerate()
        .map(|(step, _)| {
            graph
                .nodes
                .iter()
                .filter(|node| order_index[&node.id] > step)
                .flat_map(|node| {
                    node.inputs
                        .iter()
                        .copied()
                        .filter(|input| order_index.get(input).is_some_and(|index| *index <= step))
                        .tuple_combinations()
                        .collect_vec()
                })
                .collect_vec()
        })
        .collect_vec()
}

fn local_density(world: &World3D, position: Position) -> usize {
    position
        .forwards()
        .into_iter()
        .filter(|&pos| world.size.bound_on(pos))
        .filter(|&pos| !world[pos].kind.is_air())
        .count()
}

fn supports_sequential_primitive(sequential: &SequentialPrimitive) -> bool {
    matches!(
        sequential.sequential_type,
        SequentialType::RsLatch | SequentialType::DLatch
    ) && (sequential.rs_latch_core().is_some()
        || !SequentialMacro::candidates(sequential).is_empty())
}

#[derive(Debug, Clone)]
struct RsLatchGatePlacement {
    world: World3D,
    q_torch: Position,
    q_cobble: Position,
    nq_torch: Position,
    nq_cobble: Position,
}

const D_LATCH_SET_NODE_ID: GraphNodeId = usize::MAX - 1;
const D_LATCH_RESET_NODE_ID: GraphNodeId = usize::MAX;
const D_LATCH_NOT_D_NODE_ID: GraphNodeId = usize::MAX - 2;
const D_LATCH_SET_NOT_EN_NODE_ID: GraphNodeId = usize::MAX - 3;
const D_LATCH_SET_OR_NODE_ID: GraphNodeId = usize::MAX - 4;
const D_LATCH_RESET_OR_NODE_ID: GraphNodeId = usize::MAX - 5;

fn generate_d_latch_gate_routes(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    world: &World3D,
    state: &PlacementState,
) -> PlacerQueue {
    let Some(data_input) = sequential_input(node, sequential, state, "d") else {
        return Vec::new();
    };
    let Some(enable_input) = sequential_input(node, sequential, state, "en") else {
        return Vec::new();
    };
    let rs_sequential = SequentialPrimitive::new(
        SequentialType::RsLatch,
        vec!["s".to_owned(), "r".to_owned()],
        sequential.output_ports.clone(),
    );
    let rs_node = GraphNode {
        id: node.id,
        kind: GraphNodeKind::Sequential(rs_sequential.clone()),
        inputs: vec![D_LATCH_SET_NODE_ID, D_LATCH_RESET_NODE_ID],
        ..Default::default()
    };

    let queue = vec![(world.clone(), state.clone())];
    let queue = generate_not_step(config, 0, queue, data_input.node_id, D_LATCH_NOT_D_NODE_ID);
    let queue = generate_not_step(
        config,
        1,
        queue,
        enable_input.node_id,
        D_LATCH_SET_NOT_EN_NODE_ID,
    );
    let queue = generate_two_or_step(
        config,
        2,
        queue,
        (
            D_LATCH_NOT_D_NODE_ID,
            D_LATCH_SET_NOT_EN_NODE_ID,
            D_LATCH_SET_OR_NODE_ID,
        ),
        (
            data_input.node_id,
            D_LATCH_SET_NOT_EN_NODE_ID,
            D_LATCH_RESET_OR_NODE_ID,
        ),
    );
    let queue = generate_not_step(
        config,
        3,
        queue,
        D_LATCH_SET_OR_NODE_ID,
        D_LATCH_SET_NODE_ID,
    );
    let queue = generate_not_step(
        config,
        4,
        queue,
        D_LATCH_RESET_OR_NODE_ID,
        D_LATCH_RESET_NODE_ID,
    );
    let queue = queue
        .into_iter()
        .filter(|(world, state)| {
            d_latch_input_gates_behave(world, data_input.node_id, enable_input.node_id, state)
        })
        .collect_vec();
    let queue = SamplingPolicy::Random(64).sample_with_seed(queue, config.sampling_seed(7, 6));

    let mut routed = Vec::new();
    let limit = take_sampling_limit(config.step_sampling_policy);
    let rs_config = LocalPlacerConfig {
        step_sampling_policy: SamplingPolicy::Random(32),
        route_step_sampling_policy: SamplingPolicy::Random(32),
        not_route_step_sampling_policy: SamplingPolicy::Random(32),
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_route_step: 8,
        ..*config
    };
    for (world, state) in queue {
        for (world, state) in
            generate_rs_latch_gate_routes(&rs_config, &rs_node, &rs_sequential, &world, &state)
        {
            let Some(q) = state.port_position(node.id, "q") else {
                continue;
            };
            let Some(nq) = state.port_position(node.id, "nq") else {
                continue;
            };
            if std::panic::catch_unwind(|| {
                d_latch_candidate_behaves(
                    &world,
                    data_input.node_id,
                    enable_input.node_id,
                    &state,
                    q,
                    nq,
                )
            })
            .unwrap_or(false)
            {
                routed.push((world, state));
            }
        }
        if limit.is_some_and(|limit| routed.len() >= limit) {
            break;
        }
    }
    sample_d_latch_step(config, 6, routed)
}

fn d_latch_input_gates_behave(
    world: &World3D,
    data_node_id: GraphNodeId,
    enable_node_id: GraphNodeId,
    state: &PlacementState,
) -> bool {
    let Some(data) = state.node_position(data_node_id) else {
        return false;
    };
    let Some(enable) = state.node_position(enable_node_id) else {
        return false;
    };
    let Some(set) = state.node_position(D_LATCH_SET_NODE_ID) else {
        return false;
    };
    let Some(reset) = state.node_position(D_LATCH_RESET_NODE_ID) else {
        return false;
    };

    [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .all(|(data_value, enable_value)| {
            d_latch_input_gate_values(world, data, enable, set, reset, data_value, enable_value)
                .is_some_and(|(set_value, reset_value)| {
                    set_value == (data_value && enable_value)
                        && reset_value == (!data_value && enable_value)
                })
        })
}

fn d_latch_input_gate_values(
    world: &World3D,
    data: Position,
    enable: Position,
    set: Position,
    reset: Position,
    data_value: bool,
    enable_value: bool,
) -> Option<(bool, bool)> {
    let mut world = pad_world_top_for_simulation(world, 2);
    world[data].kind = BlockKind::Switch { is_on: data_value };
    world[enable].kind = BlockKind::Switch {
        is_on: enable_value,
    };
    world.initialize_redstone_states();
    let world = World::from(&world);
    let sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0).ok()?;
    let set_value = match sim.world()[set].kind {
        BlockKind::Torch { is_on } => is_on,
        _ => return None,
    };
    let reset_value = match sim.world()[reset].kind {
        BlockKind::Torch { is_on } => is_on,
        _ => return None,
    };
    Some((set_value, reset_value))
}

fn d_latch_candidate_behaves(
    world: &World3D,
    data_node_id: GraphNodeId,
    enable_node_id: GraphNodeId,
    state: &PlacementState,
    q: Position,
    nq: Position,
) -> bool {
    let Some(data) = state.node_position(data_node_id) else {
        return false;
    };
    let Some(enable) = state.node_position(enable_node_id) else {
        return false;
    };

    let mut reset_world = pad_world_top_for_simulation(world, 2);
    reset_world[enable].kind = BlockKind::Switch { is_on: true };
    reset_world[data].kind = BlockKind::Switch { is_on: false };
    reset_world.initialize_redstone_states();
    let world = World::from(&reset_world);
    let Ok(mut sim) = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0) else {
        return false;
    };

    if torch_is_on_in_world3d(sim.world(), q) || !torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, false)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(data, true)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if torch_is_on_in_world3d(sim.world(), q) || !torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, true)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if !torch_is_on_in_world3d(sim.world(), q) || torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, false)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(data, false)], 64, 20_000)
        .is_err()
    {
        return false;
    }
    if !torch_is_on_in_world3d(sim.world(), q) || torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if sim
        .change_state_with_limits(vec![(enable, true)], 64, 20_000)
        .is_err()
    {
        return false;
    }

    !torch_is_on_in_world3d(sim.world(), q) && torch_is_on_in_world3d(sim.world(), nq)
}

fn torch_is_on_in_world3d(world: &World3D, position: Position) -> bool {
    matches!(world[position].kind, BlockKind::Torch { is_on: true })
}

fn pad_world_top_for_simulation(world: &World3D, extra_z: usize) -> World3D {
    let mut padded = World3D::new(DimSize(world.size.0, world.size.1, world.size.2 + extra_z));
    for (position, mut block) in world.iter_block() {
        if block.kind.is_cobble() {
            block.kind = BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            };
        }
        padded[position] = block;
    }
    padded
}

#[derive(Clone, Copy)]
struct SequentialInput {
    node_id: GraphNodeId,
}

fn sequential_input(
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    port: &str,
) -> Option<SequentialInput> {
    node.inputs
        .iter()
        .zip(&sequential.input_ports)
        .find_map(|(input_node, input_port)| {
            (input_port == port && state.node_position(*input_node).is_some()).then_some(
                SequentialInput {
                    node_id: *input_node,
                },
            )
        })
}

fn generate_not_step(
    config: &LocalPlacerConfig,
    step: usize,
    queue: PlacerQueue,
    input_node_id: GraphNodeId,
    output_node_id: GraphNodeId,
) -> PlacerQueue {
    let candidates: PlacerQueue = queue
        .into_iter()
        .flat_map(|(world, state)| {
            let source = state[&input_node_id];
            let route_config = if world[source].kind.is_switch() {
                Cow::Owned(LocalPlacerConfig {
                    not_route_strategy: NotRouteStrategy::DirectOnly,
                    ..*config
                })
            } else {
                Cow::Borrowed(config)
            };
            not_node_kind()
                .into_iter()
                .flat_map(move |kind| {
                    let state = state.clone();
                    generate_place_and_routes(&route_config, &world, source, kind)
                        .into_iter()
                        .map(move |(world, position)| {
                            let mut state = state.clone();
                            state.set_node_position(output_node_id, position);
                            (world, state)
                        })
                })
                .collect_vec()
        })
        .collect();
    sample_d_latch_step(config, step, candidates)
}

fn generate_two_or_step(
    config: &LocalPlacerConfig,
    step: usize,
    queue: PlacerQueue,
    first: (GraphNodeId, GraphNodeId, GraphNodeId),
    second: (GraphNodeId, GraphNodeId, GraphNodeId),
) -> PlacerQueue {
    let candidates = queue
        .into_iter()
        .flat_map(|(base_world, state)| {
            let first_routes =
                generate_or_routes(config, &base_world, state[&first.0], state[&first.1]).routes;
            let second_routes =
                generate_or_routes(config, &base_world, state[&second.0], state[&second.1]).routes;
            first_routes
                .into_iter()
                .cartesian_product(second_routes)
                .filter_map(
                    |((first_world, first_positions), (second_world, second_positions))| {
                        let world =
                            merge_independent_route_worlds(&base_world, first_world, second_world)?;
                        let first_position = first_positions.last().copied()?;
                        let second_position = second_positions.last().copied()?;
                        Some({
                            let mut state = state.clone();
                            state.set_node_position(first.2, first_position);
                            state.set_node_position(second.2, second_position);
                            (world, state)
                        })
                    },
                )
                .collect_vec()
        })
        .collect();
    sample_d_latch_step(config, step, candidates)
}

fn merge_independent_route_worlds(
    base: &World3D,
    first: World3D,
    second: World3D,
) -> Option<World3D> {
    let mut merged = first.clone();
    let mut first_redstone = HashSet::new();
    let mut second_redstone = HashSet::new();
    for (x, y, z) in iproduct!(0..base.size.0, 0..base.size.1, 0..base.size.2) {
        let position = Position(x, y, z);
        if first[position] != base[position] && first[position].kind.is_redstone() {
            first_redstone.insert(position);
        }
        if second[position] != base[position] && second[position].kind.is_redstone() {
            second_redstone.insert(position);
        }
        if second[position] == base[position] {
            continue;
        }
        if first[position] != base[position] {
            return None;
        }
        merged[position] = second[position];
        if merged[position].kind.is_redstone() {
            merged.update_redstone_states(position);
        }
    }
    if redstone_networks_touch(&merged, &first_redstone, &second_redstone) {
        return None;
    }
    Some(merged)
}

fn redstone_networks_touch(
    world: &World3D,
    first: &HashSet<Position>,
    second: &HashSet<Position>,
) -> bool {
    first.iter().any(|&position| {
        PlacedNode::new(position, world[position])
            .propagation_bound(Some(world))
            .into_iter()
            .any(|bound| second.contains(&bound.position()))
    }) || second.iter().any(|&position| {
        PlacedNode::new(position, world[position])
            .propagation_bound(Some(world))
            .into_iter()
            .any(|bound| first.contains(&bound.position()))
    })
}

fn sample_d_latch_step(
    config: &LocalPlacerConfig,
    step: usize,
    candidates: PlacerQueue,
) -> PlacerQueue {
    config
        .step_sampling_policy
        .sample_with_seed(candidates, config.sampling_seed(7, step))
}

fn take_sampling_limit(policy: SamplingPolicy) -> Option<usize> {
    match policy {
        SamplingPolicy::Take(count) => Some(count),
        SamplingPolicy::None | SamplingPolicy::Random(_) => None,
    }
}

fn generate_rs_latch_gate_routes(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    world: &World3D,
    state: &PlacementState,
) -> PlacerQueue {
    let raw_pairs = generate_rs_latch_not_pairs(world);
    let pairs = select_rs_latch_not_pairs(config, node, sequential, state, raw_pairs.clone());
    let mut routed = Vec::new();
    let limit = take_sampling_limit(config.step_sampling_policy);
    for pair in pairs {
        for placed in route_rs_latch_branches(config, node, sequential, state, pair) {
            let mut state = state.clone();
            state.set_port_position(node.id, "q".to_owned(), placed.q_torch);
            state.set_port_position(node.id, "nq".to_owned(), placed.nq_torch);
            let primary_output = sequential
                .output_ports
                .first()
                .map(String::as_str)
                .unwrap_or("q");
            let primary_position = match primary_output {
                "q" => placed.q_torch,
                "nq" => placed.nq_torch,
                _ => placed.q_torch,
            };
            state.set_node_position(node.id, primary_position);
            routed.push((placed.world, state));
            if limit.is_some_and(|limit| routed.len() >= limit) {
                break;
            }
        }
        if limit.is_some_and(|limit| routed.len() >= limit) {
            break;
        }
    }

    if routed.is_empty() {
        generate_sequential_macro_routes(config, node, sequential, world, state)
    } else {
        routed
    }
}

fn select_rs_latch_not_pairs(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    mut pairs: Vec<RsLatchGatePlacement>,
) -> Vec<RsLatchGatePlacement> {
    let Some(set_source) = rs_latch_input_source(node, sequential, state, "s") else {
        return config
            .step_sampling_policy
            .sample_with_seed(pairs, config.sampling_seed(4, 0));
    };
    let Some(reset_source) = rs_latch_input_source(node, sequential, state, "r") else {
        return Vec::new();
    };

    pairs.sort_by_key(|placed| {
        let reset_distance = reset_source.manhattan_distance(&placed.q_cobble);
        let set_distance = set_source.manhattan_distance(&placed.nq_cobble);
        let input_block_penalty = if lies_between(reset_source, placed.q_cobble, placed.q_torch) {
            1_000
        } else {
            0
        } + if lies_between(set_source, placed.nq_cobble, placed.nq_torch)
        {
            1_000
        } else {
            0
        };
        let input_space_penalty = (reset_distance.abs_diff(set_distance) * 10)
            + usize::from(reset_distance < 2) * 100
            + usize::from(set_distance < 2) * 100;
        let internal_interference_penalty =
            d_latch_internal_interference_penalty(state, placed) * 2_000;

        reset_distance
            + set_distance
            + placed.q_torch.manhattan_distance(&placed.nq_torch)
            + input_block_penalty
            + input_space_penalty
            + internal_interference_penalty
    });

    match config.step_sampling_policy {
        SamplingPolicy::None => pairs,
        SamplingPolicy::Take(count) | SamplingPolicy::Random(count) => {
            pairs.into_iter().take(count).collect()
        }
    }
}

fn d_latch_internal_interference_penalty(
    state: &PlacementState,
    placed: &RsLatchGatePlacement,
) -> usize {
    let protected_nodes = [
        D_LATCH_NOT_D_NODE_ID,
        D_LATCH_SET_NOT_EN_NODE_ID,
        D_LATCH_SET_OR_NODE_ID,
        D_LATCH_RESET_OR_NODE_ID,
    ];
    let latch_positions = [
        placed.q_torch,
        placed.q_cobble,
        placed.nq_torch,
        placed.nq_cobble,
    ];

    protected_nodes
        .into_iter()
        .filter_map(|node_id| state.node_position(node_id))
        .flat_map(|protected| {
            latch_positions
                .into_iter()
                .map(move |latch| 4usize.saturating_sub(protected.manhattan_distance(&latch)))
        })
        .sum()
}

fn lies_between(start: Position, end: Position, position: Position) -> bool {
    start.manhattan_distance(&position) + position.manhattan_distance(&end)
        == start.manhattan_distance(&end)
        && position != start
        && position != end
}

fn route_rs_latch_branches(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    placed: RsLatchGatePlacement,
) -> Vec<RsLatchGatePlacement> {
    let Some(set_source) = rs_latch_input_source(node, sequential, state, "s") else {
        return route_rs_latch_feedback(config, placed);
    };
    let Some(reset_source) = rs_latch_input_source(node, sequential, state, "r") else {
        return Vec::new();
    };

    let q_torch = placed.q_torch;
    let q_cobble = placed.q_cobble;
    let nq_torch = placed.nq_torch;
    let nq_cobble = placed.nq_cobble;
    route_rs_latch_signals_to_cobble(config, placed, nq_torch, reset_source, q_torch, q_cobble)
        .into_iter()
        .flat_map(|placed| {
            route_rs_latch_signals_to_cobble(
                config, placed, q_torch, set_source, nq_torch, nq_cobble,
            )
        })
        .collect()
}

fn route_rs_latch_signals_to_cobble(
    config: &LocalPlacerConfig,
    placed: RsLatchGatePlacement,
    first_source: Position,
    second_source: Position,
    target_torch: Position,
    target_cobble: Position,
) -> Vec<RsLatchGatePlacement> {
    let worlds = generate_two_routes_to_cobble(
        config,
        &placed.world,
        first_source,
        second_source,
        target_torch,
        target_cobble,
    );
    config
        .route_step_sampling_policy
        .sample_with_seed(
            worlds,
            config.sampling_seed(8, target_cobble.0 + target_cobble.1),
        )
        .into_iter()
        .map(|world| RsLatchGatePlacement { world, ..placed })
        .collect()
}

fn generate_two_routes_to_cobble(
    config: &LocalPlacerConfig,
    world: &World3D,
    first_source: Position,
    second_source: Position,
    target_torch: Position,
    target_cobble: Position,
) -> Vec<World3D> {
    let first_routes = generate_routes_to_cobble_with_paths(
        config,
        world,
        first_source,
        target_torch,
        target_cobble,
    );
    let second_routes = generate_routes_to_cobble_with_paths(
        config,
        world,
        second_source,
        target_torch,
        target_cobble,
    );

    let first_then_second = first_routes.iter().flat_map(|(first_world, first_path)| {
        let network = redstone_route_network(first_world, first_path, target_cobble);
        generate_routes_to_cobble_or_network(
            config,
            first_world,
            second_source,
            target_torch,
            target_cobble,
            &network,
        )
        .into_iter()
        .map(|(world, _)| world)
        .collect_vec()
    });

    let second_then_first = second_routes
        .iter()
        .flat_map(|(second_world, second_path)| {
            let network = redstone_route_network(second_world, second_path, target_cobble);
            generate_routes_to_cobble_or_network(
                config,
                second_world,
                first_source,
                target_torch,
                target_cobble,
                &network,
            )
            .into_iter()
            .map(|(world, _)| world)
            .collect_vec()
        });

    first_then_second.chain(second_then_first).collect()
}

fn redstone_route_network(
    world: &World3D,
    path: &[Position],
    target_cobble: Position,
) -> HashSet<Position> {
    path.iter()
        .copied()
        .filter(|&position| {
            world[position].kind.is_redstone()
                && redstone_powers_cobble(world, position, target_cobble)
        })
        .collect()
}

fn rs_latch_input_source(
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    port: &str,
) -> Option<Position> {
    node.inputs
        .iter()
        .zip(&sequential.input_ports)
        .find_map(|(input_node, input_port)| {
            (input_port == port)
                .then(|| state.node_position(*input_node))
                .flatten()
        })
}

fn generate_rs_latch_not_pairs(world: &World3D) -> Vec<RsLatchGatePlacement> {
    let cardinal = [
        Direction::East,
        Direction::West,
        Direction::South,
        Direction::North,
    ];
    let support_positions = iproduct!(0..world.size.0, 0..world.size.1, 1..world.size.2)
        .map(|(x, y, z)| Position(x, y, z))
        .collect_vec();
    let mut candidates = Vec::new();

    for &q_direction in &cardinal {
        for q_cobble in &support_positions {
            let Some(q_torch) = q_cobble.walk(q_direction.inverse()) else {
                continue;
            };
            if !world.size.bound_on(q_torch) {
                continue;
            }
            let q_torch_block = Block {
                kind: BlockKind::Torch { is_on: false },
                direction: q_direction,
            };
            let Some((q_world, q_torch, q_cobble)) =
                place_torch_with_cobble(world, q_torch_block, q_torch)
            else {
                continue;
            };

            for &nq_direction in &cardinal {
                for nq_cobble in &support_positions {
                    if q_cobble == *nq_cobble || q_cobble.2 != nq_cobble.2 {
                        continue;
                    }
                    let support_distance = q_cobble.manhattan_distance(nq_cobble);
                    if !(2..=3).contains(&support_distance) || q_direction.inverse() != nq_direction
                    {
                        continue;
                    }
                    let Some(nq_torch) = nq_cobble.walk(nq_direction.inverse()) else {
                        continue;
                    };
                    if !q_world.size.bound_on(nq_torch)
                        || q_torch == nq_torch
                        || !(3..=5).contains(&q_torch.manhattan_distance(&nq_torch))
                    {
                        continue;
                    }
                    let nq_torch_block = Block {
                        kind: BlockKind::Torch { is_on: false },
                        direction: nq_direction,
                    };
                    let Some((world, nq_torch, nq_cobble)) =
                        place_torch_with_cobble(&q_world, nq_torch_block, nq_torch)
                    else {
                        continue;
                    };
                    candidates.push(RsLatchGatePlacement {
                        world,
                        q_torch,
                        q_cobble,
                        nq_torch,
                        nq_cobble,
                    });
                }
            }
        }
    }

    candidates
}

fn route_rs_latch_feedback(
    config: &LocalPlacerConfig,
    placed: RsLatchGatePlacement,
) -> Vec<RsLatchGatePlacement> {
    generate_routes_to_cobble(
        config,
        &placed.world,
        placed.nq_torch,
        placed.q_torch,
        placed.q_cobble,
    )
    .into_iter()
    .flat_map(|(world, _)| {
        generate_routes_to_cobble(
            config,
            &world,
            placed.q_torch,
            placed.nq_torch,
            placed.nq_cobble,
        )
        .into_iter()
        .map(|(world, _)| RsLatchGatePlacement {
            world,
            q_torch: placed.q_torch,
            q_cobble: placed.q_cobble,
            nq_torch: placed.nq_torch,
            nq_cobble: placed.nq_cobble,
        })
        .collect_vec()
    })
    .collect()
}

fn generate_sequential_macro_routes(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    world: &World3D,
    state: &PlacementState,
) -> PlacerQueue {
    SequentialMacro::candidates(sequential)
        .into_iter()
        .flat_map(|candidate| {
            iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
                .map(|(x, y, z)| Position(x, y, z))
                .filter_map(move |anchor| place_sequential_macro(world, &candidate, anchor))
                .flat_map(|placed| {
                    route_sequential_inputs(config, node, sequential, state, &placed)
                        .into_iter()
                        .map(|world| {
                            let mut state = state.clone();
                            for (output_port, position) in &placed.output_ports {
                                state.set_port_position(node.id, output_port.to_owned(), *position);
                            }
                            let primary_output = sequential
                                .output_ports
                                .first()
                                .unwrap_or(&placed.primary_output_port);
                            let primary_position = placed.output_ports[primary_output];
                            state.set_node_position(node.id, primary_position);
                            (world, state)
                        })
                        .collect_vec()
                })
                .collect_vec()
        })
        .collect()
}

#[derive(Debug, Clone)]
struct PlacedSequentialMacro {
    world: World3D,
    input_ports: HashMap<String, Position>,
    output_ports: HashMap<String, Position>,
    primary_output_port: String,
}

fn place_sequential_macro(
    world: &World3D,
    candidate: &SequentialMacro,
    anchor: Position,
) -> Option<PlacedSequentialMacro> {
    let mut new_world = world.clone();
    for (relative, block) in candidate.world.iter_block() {
        let position = translate_position(anchor, relative)?;
        if !world.size.bound_on(position) || !world[position].kind.is_air() {
            return None;
        }
        new_world[position] = block;
    }
    new_world.initialize_redstone_states();

    Some(PlacedSequentialMacro {
        world: new_world,
        input_ports: translate_ports(anchor, &candidate.input_ports)?,
        output_ports: translate_ports(anchor, &candidate.output_ports)?,
        primary_output_port: candidate.primary_output_port.clone(),
    })
}

fn route_sequential_inputs(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    state: &PlacementState,
    placed: &PlacedSequentialMacro,
) -> Vec<World3D> {
    let mut worlds = vec![placed.world.clone()];
    for (input_node, input_port) in node.inputs.iter().zip(&sequential.input_ports) {
        let Some(source) = state.node_position(*input_node) else {
            return Vec::new();
        };
        let Some(&target) = placed.input_ports.get(input_port) else {
            return Vec::new();
        };

        worlds = worlds
            .into_iter()
            .flat_map(|world| {
                generate_or_routes(config, &world, source, target)
                    .routes
                    .into_iter()
                    .map(|(world, _)| world)
                    .collect_vec()
            })
            .collect_vec();
    }
    worlds
}

fn translate_ports(
    anchor: Position,
    ports: &HashMap<String, Position>,
) -> Option<HashMap<String, Position>> {
    ports
        .iter()
        .map(|(port, position)| Some((port.clone(), translate_position(anchor, *position)?)))
        .collect()
}

fn translate_position(anchor: Position, relative: Position) -> Option<Position> {
    Some(Position(
        anchor.0.checked_add(relative.0)?,
        anchor.1.checked_add(relative.1)?,
        anchor.2.checked_add(relative.2)?,
    ))
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

fn input_node_kind() -> Vec<BlockKind> {
    vec![
        BlockKind::Switch { is_on: false },
        // BlockKind::Cobble {
        //     on_count: 0,
        //     on_base_count: 0,
        // },
        // BlockKind::Redstone {
        //     on_count: 0,
        //     state: RedstoneState::None as usize,
        //     strength: 0,
        // },
        // BlockKind::RedstoneBlock,
    ]
}

fn not_node_kind() -> Vec<BlockKind> {
    vec![BlockKind::Torch { is_on: false }]
}

fn place_node(world: &mut World3D, node: PlacedNode) {
    if world[node.position] == node.block {
        // TODO: Relax for no-op
        assert!(world[node.position].kind.is_cobble());
        return;
    }

    world[node.position] = node.block;
    if node.block.kind.is_redstone() {
        world.update_redstone_states(node.position);
    }
}

fn generate_inputs(
    config: &LocalPlacerConfig,
    world: &World3D,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    let mut input_strategy = Direction::iter_direction_without_top()
        .map(|direction| Block { kind, direction })
        .collect_vec();
    if config.greedy_input_generation {
        input_strategy = input_strategy.into_iter().take(1).collect();
    }

    let place_strategy = match config.input_placement_strategy {
        InputPlacementStrategy::Boundary => iproduct!(0..1, 0..world.size.1, 0..world.size.2)
            .chain(iproduct!(0..world.size.0, 0..1, 0..world.size.2))
            .map(|(x, y, z)| Position(x, y, z))
            .collect_vec(),
        InputPlacementStrategy::Anywhere => {
            iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
                .map(|(x, y, z)| Position(x, y, z))
                .collect_vec()
        }
    };

    input_strategy
        .into_iter()
        .cartesian_product(place_strategy)
        // Place Input Node
        .flat_map(|(block, position)| {
            let placed_node = PlacedNode { position, block };
            if placed_node.has_conflict(world, &Default::default()) {
                return None;
            }

            let mut new_world = world.clone();
            place_node(&mut new_world, placed_node);
            Some((new_world, position))
        })
        .collect_vec()
}

fn generate_place_and_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    start: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    match kind {
        BlockKind::Torch { .. } => generate_torch_place_and_routes(config, world, start, kind),
        _ => unimplemented!(),
    }
}

fn generate_torch_place_and_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    let torch_strategy =
        Direction::iter_direction_without_top().map(|direction| Block { kind, direction });

    let place_strategy = iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
        .map(|(x, y, z)| Position(x, y, z))
        // start에서 최소 두 칸 떨어진 곳에 위치시킨다.
        .filter(|pos| source.manhattan_distance(pos) > 1)
        .filter(|pos| {
            !config.route_torch_directly
                || matches!(
                    config.torch_placement_strategy,
                    TorchPlacementStrategy::AnywhereNonAdjacent
                )
                || source.manhattan_distance(pos) == 2
        });

    torch_strategy
        .cartesian_product(place_strategy)
        // 1. Place Torch and Cobble
        .flat_map(|(torch, torch_pos)| place_torch_with_cobble(world, torch, torch_pos))
        // 2. Route Source with Torch Place Target Position
        .flat_map(|(world, torch_pos, cobble_pos)| {
            generate_routes_to_cobble(config, &world, source, torch_pos, cobble_pos)
                .into_iter()
                .map(|(world, _)| (world, torch_pos))
                .collect_vec()
        })
        .collect()
}

fn place_torch_with_cobble(
    world: &World3D,
    torch: Block,
    torch_pos: Position,
) -> Option<(World3D, Position, Position)> {
    let cobble_pos = torch_pos.walk(torch.direction)?;
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    let torch_node = PlacedNode::new(torch_pos, torch);
    if cobble_node.has_conflict(world, &[cobble_pos].into_iter().collect())
        || torch_node.has_conflict(world, &Default::default())
    {
        return None;
    }

    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);
    place_node(&mut new_world, torch_node);
    Some((new_world, torch_pos, cobble_pos))
}

fn generate_routes_to_cobble(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    _torch_pos: Position,
    cobble_pos: Position,
) -> Vec<(World3D, Position)> {
    let source_node = PlacedNode::new(source, world[source]);
    assert!(source_node.is_propagation_target() && world[cobble_pos].kind.is_cobble());
    let forbidden_cobble = torch_source_support(world, source);

    // (world, pos, cost)
    let mut route_candidates = Vec::new();
    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::DirectOnly | NotRouteStrategy::DirectAndRedstone
    ) {
        for start in source_node.propagation_bound(Some(world)) {
            let directly_connected = start.position() == cobble_pos
                && matches!(
                    start.propagation_type(),
                    // TODO: 조건 삭제 가능한지 확인
                    // Hard: Switch -> Torch 연결
                    // Soft: Redstone -> Torch 연결
                    PropagateType::Hard | PropagateType::Soft
                );

            if directly_connected
                && !route_powers_forbidden_cobble(world, &[source], forbidden_cobble)
            {
                route_candidates.push((world.clone(), start.position(), 0usize));
                continue;
            }
        }
    }

    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::RedstoneOnly | NotRouteStrategy::DirectAndRedstone
    ) && (source_node.is_diode() || source_node.block.kind.is_redstone())
    {
        let route_config = LocalPlacerConfig {
            max_route_step: config.max_not_route_step,
            route_step_sampling_policy: config.not_route_step_sampling_policy,
            ..*config
        };
        route_candidates.extend(
            generate_redstone_routes_to_cobble(&route_config, world, source, cobble_pos)
                .into_iter()
                .map(|(world, positions)| {
                    (world, positions.last().copied().unwrap(), positions.len())
                }),
        );
    }

    route_candidates
        .into_iter()
        .sorted_by_key(|(_, _, cost)| *cost)
        .map(|(world, pos, _)| (world, pos))
        .collect()
}

// 두 torch를 연결하는 redstone routes를 생성한다.
fn generate_routes_to_cobble_with_paths(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    _torch_pos: Position,
    cobble_pos: Position,
) -> Vec<(World3D, Vec<Position>)> {
    let source_node = PlacedNode::new(source, world[source]);
    assert!(source_node.is_propagation_target() && world[cobble_pos].kind.is_cobble());
    let forbidden_cobble = torch_source_support(world, source);

    let mut route_candidates = Vec::new();
    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::DirectOnly | NotRouteStrategy::DirectAndRedstone
    ) {
        for start in source_node.propagation_bound(Some(world)) {
            let directly_connected = start.position() == cobble_pos
                && matches!(
                    start.propagation_type(),
                    PropagateType::Hard | PropagateType::Soft
                );

            if directly_connected
                && !route_powers_forbidden_cobble(world, &[source], forbidden_cobble)
            {
                route_candidates.push((world.clone(), vec![source]));
            }
        }
    }

    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::RedstoneOnly | NotRouteStrategy::DirectAndRedstone
    ) && (source_node.is_diode() || source_node.block.kind.is_redstone())
    {
        let route_config = LocalPlacerConfig {
            max_route_step: config.max_not_route_step,
            route_step_sampling_policy: config.not_route_step_sampling_policy,
            ..*config
        };
        route_candidates.extend(generate_redstone_routes_to_cobble(
            &route_config,
            world,
            source,
            cobble_pos,
        ));
    }

    route_candidates
}

fn generate_routes_to_cobble_or_network(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    _torch_pos: Position,
    cobble_pos: Position,
    network: &HashSet<Position>,
) -> Vec<(World3D, Vec<Position>)> {
    let source_node = PlacedNode::new(source, world[source]);
    assert!(source_node.is_propagation_target() && world[cobble_pos].kind.is_cobble());
    let forbidden_cobble = torch_source_support(world, source);

    let mut route_candidates = Vec::new();
    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::DirectOnly | NotRouteStrategy::DirectAndRedstone
    ) {
        for start in source_node.propagation_bound(Some(world)) {
            let directly_connected = start.position() == cobble_pos
                && matches!(
                    start.propagation_type(),
                    PropagateType::Hard | PropagateType::Soft
                );

            if (directly_connected || network.contains(&start.position()))
                && !route_powers_forbidden_cobble(world, &[source], forbidden_cobble)
            {
                route_candidates.push((world.clone(), vec![source]));
            }
        }
    }

    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::RedstoneOnly | NotRouteStrategy::DirectAndRedstone
    ) && (source_node.is_diode() || source_node.block.kind.is_redstone())
    {
        let route_config = LocalPlacerConfig {
            max_route_step: config.max_not_route_step,
            route_step_sampling_policy: config.not_route_step_sampling_policy,
            ..*config
        };
        route_candidates.extend(generate_redstone_routes_to_cobble_or_network(
            &route_config,
            world,
            source,
            cobble_pos,
            network,
        ));
    }

    route_candidates
}

fn generate_redstone_routes_to_cobble(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    cobble_pos: Position,
) -> Vec<(World3D, Vec<Position>)> {
    let source_node = PlacedNode::new(source, world[source]);
    let forbidden_cobble = torch_source_support(world, source);
    let mut queue = if source_node.is_diode() {
        generate_routes_to_cobble_init_states(world, source)
    } else if source_node.block.kind.is_redstone() {
        vec![(
            world.clone(),
            vec![source],
            source_node.propagation_bound(Some(world)),
        )]
    } else {
        Vec::new()
    };
    let mut candidates = Vec::new();
    let mut step = 0;

    while step < config.max_route_step && !queue.is_empty() {
        let mut next_queue = Vec::new();

        for (world, prevs, bounds) in queue {
            let prev_pos = prevs.last().copied().unwrap();
            for bound in bounds {
                if !world.size.bound_on(bound.position()) {
                    continue;
                }

                let (new_world, redstone_node) =
                    match place_redstone_with_cobble(&world, bound, prev_pos, cobble_pos) {
                        PlaceRedstoneResult::Placed(new_world, redstone_node) => {
                            (new_world, redstone_node)
                        }
                        PlaceRedstoneResult::Rejected(_) => continue,
                    };

                let new_prevs = prevs
                    .iter()
                    .copied()
                    .chain([redstone_node.position])
                    .collect_vec();
                if route_powers_forbidden_cobble(&new_world, &new_prevs, forbidden_cobble) {
                    continue;
                }

                if redstone_powers_cobble(&new_world, redstone_node.position, cobble_pos) {
                    candidates.push((new_world, new_prevs));
                } else {
                    let nexts = redstone_node.propagation_bound(Some(&new_world));
                    next_queue.push((new_world, new_prevs, nexts));
                }
            }
        }

        queue = config
            .route_step_sampling_policy
            .sample_with_seed(next_queue, config.sampling_seed(5, step));
        step += 1;
    }

    candidates
}

fn generate_redstone_routes_to_cobble_or_network(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    cobble_pos: Position,
    network: &HashSet<Position>,
) -> Vec<(World3D, Vec<Position>)> {
    let source_node = PlacedNode::new(source, world[source]);
    let forbidden_cobble = torch_source_support(world, source);
    let mut queue = if source_node.is_diode() {
        generate_routes_to_cobble_init_states(world, source)
    } else if source_node.block.kind.is_redstone() {
        vec![(
            world.clone(),
            vec![source],
            source_node.propagation_bound(Some(world)),
        )]
    } else {
        Vec::new()
    };
    let mut candidates = Vec::new();
    let mut step = 0;

    while step < config.max_route_step && !queue.is_empty() {
        let mut next_queue = Vec::new();

        for (world, prevs, bounds) in queue {
            let prev_pos = prevs.last().copied().unwrap();
            for bound in bounds {
                if !world.size.bound_on(bound.position()) {
                    continue;
                }

                let (new_world, redstone_node) = match place_redstone_with_cobble_allowing_shorts(
                    &world, bound, prev_pos, cobble_pos, network,
                ) {
                    PlaceRedstoneResult::Placed(new_world, redstone_node) => {
                        (new_world, redstone_node)
                    }
                    PlaceRedstoneResult::Rejected(_) => continue,
                };

                let new_prevs = prevs
                    .iter()
                    .copied()
                    .chain([redstone_node.position])
                    .collect_vec();
                if route_powers_forbidden_cobble(&new_world, &new_prevs, forbidden_cobble) {
                    continue;
                }

                let powers_cobble =
                    redstone_powers_cobble(&new_world, redstone_node.position, cobble_pos);
                let connects_to_network =
                    redstone_connects_to_network(&new_world, redstone_node.position, network)
                        && route_network_powers_cobble(&new_world, &new_prevs, network, cobble_pos);
                if powers_cobble || connects_to_network {
                    candidates.push((new_world, new_prevs));
                } else {
                    let nexts = redstone_node.propagation_bound(Some(&new_world));
                    next_queue.push((new_world, new_prevs, nexts));
                }
            }
        }

        queue = config
            .route_step_sampling_policy
            .sample_with_seed(next_queue, config.sampling_seed(5, step));
        step += 1;
    }

    candidates
}

fn torch_source_support(world: &World3D, source: Position) -> Option<Position> {
    matches!(world[source].kind, BlockKind::Torch { .. })
        .then(|| source.walk(world[source].direction))
        .flatten()
}

fn route_powers_forbidden_cobble(
    world: &World3D,
    route: &[Position],
    forbidden_cobble: Option<Position>,
) -> bool {
    let Some(forbidden_cobble) = forbidden_cobble else {
        return false;
    };
    if world.size.bound_on(forbidden_cobble.up()) && world[forbidden_cobble.up()].kind.is_cobble() {
        return true;
    }
    route
        .iter()
        .copied()
        .filter(|&position| world[position].kind.is_redstone())
        .any(|position| redstone_powers_cobble(world, position, forbidden_cobble))
}

fn redstone_powers_cobble(world: &World3D, redstone: Position, cobble: Position) -> bool {
    world[cobble].kind.is_cobble()
        && PlacedNode::new(redstone, world[redstone])
            .propagation_bound(Some(world))
            .into_iter()
            .any(|bound| bound.position() == cobble)
}

fn redstone_connects_to_network(
    world: &World3D,
    source: Position,
    network: &HashSet<Position>,
) -> bool {
    let source_node = PlacedNode::new(source, world[source]);
    source_node
        .propagation_bound(Some(world))
        .into_iter()
        .any(|bound| network.contains(&bound.position()))
}

fn route_network_powers_cobble(
    world: &World3D,
    route: &[Position],
    network: &HashSet<Position>,
    cobble_pos: Position,
) -> bool {
    route
        .iter()
        .copied()
        .chain(network.iter().copied())
        .filter(|&position| world.size.bound_on(position) && world[position].kind.is_redstone())
        .any(|position| redstone_powers_cobble(world, position, cobble_pos))
}

fn generate_routes_to_cobble_init_states(
    world: &World3D,
    source: Position,
) -> Vec<(World3D, Vec<Position>, Vec<PlaceBound>)> {
    let source_node = PlacedNode::new(source, world[source]);
    let mut states = Vec::new();

    for input_top_cobble in [false, true] {
        let mut new_world = Cow::Borrowed(world);

        if input_top_cobble {
            let Some(cobble_node) = try_generate_cobble_node(world, source.up(), &[source]) else {
                continue;
            };
            place_node(new_world.to_mut(), cobble_node);
        }

        let bounds = if input_top_cobble {
            source
                .up()
                .cardinal()
                .into_iter()
                // Cobble 위쪽에 redstone을 배치하는 케이스
                .chain(Some(source.up().up()))
                .map(|pos| PlaceBound(PropagateType::Soft, pos, pos.diff(source)))
                .collect_vec()
        } else {
            source_node.propagation_bound(Some(world))
        };

        states.push((new_world.into_owned(), vec![source], bounds));
    }

    states
}

fn generate_or_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    from: Position,
    to: Position,
) -> RouteResult {
    let (mut queue, mut debug) = generate_or_routes_init_states(world, from, to);
    debug.route_calls = 1;
    debug.route_samples.push((from, to));
    debug.initial_states = queue.len();

    let mut candidates = Vec::new();
    let mut step = 0;

    while step < config.max_route_step && !queue.is_empty() {
        let frontier_len = queue.len();
        let mut next_queue = vec![];
        let mut depth_debug = RouteDepthDebug {
            depth: step,
            frontier_len,
            ..Default::default()
        };

        for (world, prevs, bounds) in queue {
            let prev_pos = prevs.last().copied().unwrap();
            for bound in bounds {
                depth_debug.bounds_checked += 1;
                if !world.size.bound_on(bound.position()) {
                    debug.reject(RouteRejectReason::OutOfBounds);
                    continue;
                }

                let (new_world, redstone_node) =
                    match place_redstone_with_cobble(&world, bound, prev_pos, to) {
                        PlaceRedstoneResult::Placed(new_world, redstone_node) => {
                            (new_world, redstone_node)
                        }
                        PlaceRedstoneResult::Rejected(reason) => {
                            debug.reject(reason);
                            continue;
                        }
                    };

                let new_prevs = prevs
                    .iter()
                    .copied()
                    .chain([redstone_node.position])
                    .collect();

                if redstone_node.has_connection_with(&new_world, to) {
                    depth_debug.accepted_routes += 1;
                    candidates.push((new_world, new_prevs));
                } else {
                    let nexts = redstone_node.propagation_bound(Some(&new_world));
                    next_queue.push((new_world, new_prevs, nexts));
                }
            }
        }

        depth_debug.next_frontier_before_sampling = next_queue.len();
        queue = config
            .route_step_sampling_policy
            .sample_with_seed(next_queue, config.sampling_seed(3, step));
        depth_debug.next_frontier_after_sampling = queue.len();
        debug.depths.push(depth_debug);
        step += 1;
    }

    debug.candidates_found = candidates.len();
    RouteResult {
        routes: candidates,
        debug,
    }
}

struct RouteResult {
    routes: Vec<(World3D, Vec<Position>)>,
    debug: RouteDebug,
}

fn generate_or_routes_init_states(
    world: &World3D,
    from: Position,
    to: Position,
) -> (Vec<(World3D, Vec<Position>, Vec<PlaceBound>)>, RouteDebug) {
    let from_node = PlacedNode::new(from, world[from]);
    let to_node = PlacedNode::new(to, world[to]);
    assert!(from_node.is_diode() && to_node.is_propagation_target());

    let mut debug = RouteDebug::default();
    let boolean_comb = [false, true];
    let mut states = Vec::new();

    for input_top_cobble in boolean_comb {
        for output_top_cobble in boolean_comb {
            let mut new_world = Cow::Borrowed(world);

            if input_top_cobble {
                let Some(cobble_node) = try_generate_cobble_node(world, from.up(), &[from]) else {
                    debug.reject(RouteRejectReason::InitInputTopCobbleConflict);
                    continue;
                };
                place_node(new_world.to_mut(), cobble_node);
            }

            if output_top_cobble {
                let Some(cobble_node) = try_generate_cobble_node(world, to.up(), &[to]) else {
                    debug.reject(RouteRejectReason::InitOutputTopCobbleConflict);
                    continue;
                };
                place_node(new_world.to_mut(), cobble_node);
            }

            let bounds = if input_top_cobble {
                from.up()
                    .cardinal()
                    .into_iter()
                    // Cobble 위쪽에 redstone을 배치하는 케이스
                    .chain(Some(from.up().up()))
                    .map(|pos| PlaceBound(PropagateType::Soft, pos, pos.diff(from)))
                    .collect_vec()
            } else {
                from_node.propagation_bound(Some(world))
            };

            states.push((new_world.into_owned(), vec![from], bounds));
        }
    }

    (states, debug)
}

fn try_generate_cobble_node(
    world: &World3D,
    cobble_pos: Position,
    except: &[Position],
) -> Option<PlacedNode> {
    if cobble_would_stack_above_side_torch_support(world, cobble_pos) {
        return None;
    }
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    if !cobble_node.has_conflict(world, &except.iter().copied().collect()) {
        Some(cobble_node)
    } else {
        None
    }
}

fn cobble_would_stack_above_side_torch_support(world: &World3D, cobble_pos: Position) -> bool {
    let Some(below) = cobble_pos.down() else {
        return false;
    };
    world.size.bound_on(below)
        && world[below].kind.is_cobble()
        && below.cardinal().into_iter().any(|position| {
            world.size.bound_on(position)
                && matches!(world[position].kind, BlockKind::Torch { .. })
                && world[position].direction == position.diff(below)
        })
}

fn place_redstone_with_cobble(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
) -> PlaceRedstoneResult {
    let Some(cobble_pos) = bound.position().walk(Direction::Bottom) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::NoBottomForCobble);
    };
    // 첫 번째 step에서 torch 위쪽에 cobble + redstone이 놓인 경우 예외처리
    let cobble_except = (world[prev].kind.is_torch())
        .then_some(vec![cobble_pos, prev])
        .unwrap_or_default();
    let Some(cobble_node) = try_generate_cobble_node(world, cobble_pos, &cobble_except) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::CobbleConflict);
    };
    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);

    let bound_pos = bound.position();
    let Some(bound_back_pos) = bound_pos.walk(bound.direction()) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    };
    let redstone_node = PlacedNode::new_redstone(bound_pos);
    let except = [prev, bound_back_pos, bound_pos, to, to.up()]
        .into_iter()
        .collect();
    if redstone_node.has_conflict(&new_world, &except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    }
    if redstone_node.has_short(world, &except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
    }
    place_node(&mut new_world, redstone_node);
    if let BlockKind::Torch { .. } = world[prev].kind {
        if let Some(source_cobble) = prev.walk(world[prev].direction) {
            if redstone_powers_cobble(&new_world, redstone_node.position, source_cobble) {
                return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
            }
        }
    }
    new_world.update_redstone_states(prev);

    PlaceRedstoneResult::Placed(new_world, redstone_node)
}

fn place_redstone_with_cobble_allowing_shorts(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
    allowed_shorts: &HashSet<Position>,
) -> PlaceRedstoneResult {
    let Some(cobble_pos) = bound.position().walk(Direction::Bottom) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::NoBottomForCobble);
    };
    let cobble_except = (world[prev].kind.is_torch())
        .then_some(vec![cobble_pos, prev])
        .unwrap_or_default();
    let Some(cobble_node) = try_generate_cobble_node(world, cobble_pos, &cobble_except) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::CobbleConflict);
    };
    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);

    let bound_pos = bound.position();
    let Some(bound_back_pos) = bound_pos.walk(bound.direction()) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    };
    let redstone_node = PlacedNode::new_redstone(bound_pos);
    let mut except = [prev, bound_back_pos, bound_pos, to, to.up()]
        .into_iter()
        .collect::<HashSet<_>>();
    except.extend(allowed_shorts.iter().copied());
    if redstone_node.has_conflict(&new_world, &except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    }
    if redstone_node.has_short(world, &except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
    }
    place_node(&mut new_world, redstone_node);
    if let BlockKind::Torch { .. } = world[prev].kind {
        if let Some(source_cobble) = prev.walk(world[prev].direction) {
            if redstone_powers_cobble(&new_world, redstone_node.position, source_cobble) {
                return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
            }
        }
    }
    new_world.update_redstone_states(prev);

    PlaceRedstoneResult::Placed(new_world, redstone_node)
}

enum PlaceRedstoneResult {
    Placed(World3D, PlacedNode),
    Rejected(RouteRejectReason),
}

pub struct LocalPlacerCostEstimator<'a> {
    graph: &'a WorldGraph,
}

impl<'a> LocalPlacerCostEstimator<'a> {
    pub fn new(graph: &'a WorldGraph) -> Self {
        Self { graph }
    }

    pub fn cost(&self) -> usize {
        let _buffer_depth = self.graph.graph.critical_path().len();

        todo!()
    }
}

#[cfg(test)]
#[path = "local_placer/tests.rs"]
mod unit_tests;

#[cfg(test)]
mod tests {

    use crate::graph::analysis::equivalent_expression_groups;
    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::graph::logic::predefined_logics;
    use crate::graph::{GraphNode, GraphNodeKind};
    use crate::nbt::{NBTRoot, ToNBT};
    use crate::sequential::{SequentialPrimitive, SequentialType};
    use crate::transform::place_and_route::estimate::world_compact_cost;
    use crate::transform::place_and_route::local_placer::{
        generate_d_latch_gate_routes, generate_rs_latch_not_pairs, route_rs_latch_branches,
        select_rs_latch_not_pairs, InputPlacementStrategy, LocalPlacer, LocalPlacerConfig,
        LocalPlacerDebug, NotRouteStrategy, PlacementSamplingPolicy, SamplingPolicy,
        TorchPlacementStrategy, D_LATCH_NOT_D_NODE_ID, D_LATCH_RESET_NODE_ID,
        D_LATCH_RESET_OR_NODE_ID, D_LATCH_SET_NODE_ID, D_LATCH_SET_NOT_EN_NODE_ID,
        D_LATCH_SET_OR_NODE_ID,
    };
    use crate::transform::place_and_route::utils::{
        equivalent_logic_with_world3d, equivalent_logic_with_world3ds, world3d_to_logic,
    };
    use crate::world::block::{Block, BlockKind, Direction};
    use crate::world::position::{DimSize, Position};
    use crate::world::simulator::Simulator;
    use crate::world::{World, World3D};

    #[test]
    fn test_generate_component_and_shortest() -> eyre::Result<()> {
        let logic_graph = predefined_logics::and_graph()?;
        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::None,
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: false,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 1,
            not_route_step_sampling_policy: SamplingPolicy::None,
            max_route_step: 1,
            route_step_sampling_policy: SamplingPolicy::Random(100),
        };
        let placer = LocalPlacer::new(logic_graph.clone(), config)?;
        let worlds = placer.generate(DimSize(10, 10, 5), None);
        assert!(!worlds.is_empty());
        assert!(equivalent_logic_with_world3ds(&logic_graph, &worlds)?);
        Ok(())
    }

    fn save_worlds_to_nbt(worlds: Vec<World3D>, path: &str) -> eyre::Result<()> {
        let concated_world = World3D::concat_tiled(worlds);
        let nbt: NBTRoot = concated_world.to_nbt();
        nbt.save(path);
        Ok(())
    }

    fn torch_is_on(world: &World3D, position: Position) -> bool {
        let BlockKind::Torch { is_on } = world[position].kind else {
            panic!("expected torch at {position:?}");
        };

        is_on
    }

    fn assert_rs_latch_behavior(
        world: &World3D,
        s: Position,
        r: Position,
        q: Position,
        nq: Position,
    ) -> eyre::Result<()> {
        let mut reset_world = world.clone();
        reset_world[r].kind = BlockKind::Switch { is_on: true };
        reset_world.initialize_redstone_states();
        let world = World::from(&reset_world);
        let mut sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        eyre::ensure!(!torch_is_on(sim.world(), q), "reset should turn q off");
        eyre::ensure!(torch_is_on(sim.world(), nq), "reset should turn nq on");

        sim.change_state_with_limits(vec![(r, false)], 64, 20_000)?;
        eyre::ensure!(!torch_is_on(sim.world(), q), "hold reset should keep q off");
        eyre::ensure!(torch_is_on(sim.world(), nq), "hold reset should keep nq on");

        sim.change_state_with_limits(vec![(s, true)], 64, 20_000)?;
        sim.change_state_with_limits(vec![(s, false)], 64, 20_000)?;
        eyre::ensure!(torch_is_on(sim.world(), q), "set should turn q on");
        eyre::ensure!(!torch_is_on(sim.world(), nq), "set should turn nq off");

        sim.change_state_with_limits(vec![(r, true)], 64, 20_000)?;
        sim.change_state_with_limits(vec![(r, false)], 64, 20_000)?;
        eyre::ensure!(
            !torch_is_on(sim.world(), q),
            "reset again should turn q off"
        );
        eyre::ensure!(
            torch_is_on(sim.world(), nq),
            "reset again should turn nq on"
        );

        Ok(())
    }

    fn assert_d_latch_behavior(
        world: &World3D,
        d: Position,
        en: Position,
        q: Position,
        nq: Position,
    ) -> eyre::Result<()> {
        let mut reset_world = pad_world_top(world, 2);
        reset_world[en].kind = BlockKind::Switch { is_on: true };
        reset_world[d].kind = BlockKind::Switch { is_on: false };
        reset_world.initialize_redstone_states();
        let world = World::from(&reset_world);
        let mut sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        eyre::ensure!(!torch_is_on(sim.world(), q), "reset should turn q off");
        eyre::ensure!(torch_is_on(sim.world(), nq), "reset should turn nq on");

        sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
        sim.change_state_with_limits(vec![(d, true)], 64, 20_000)?;
        eyre::ensure!(
            !torch_is_on(sim.world(), q),
            "disabled latch should hold q off"
        );
        eyre::ensure!(
            torch_is_on(sim.world(), nq),
            "disabled latch should hold nq on"
        );

        sim.change_state_with_limits(vec![(en, true)], 64, 20_000)?;
        eyre::ensure!(
            torch_is_on(sim.world(), q),
            "enabled data high should set q"
        );
        eyre::ensure!(
            !torch_is_on(sim.world(), nq),
            "enabled data high should clear nq"
        );

        sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
        sim.change_state_with_limits(vec![(d, false)], 64, 20_000)?;
        eyre::ensure!(
            torch_is_on(sim.world(), q),
            "disabled latch should hold q on"
        );
        eyre::ensure!(
            !torch_is_on(sim.world(), nq),
            "disabled latch should hold nq off"
        );

        sim.change_state_with_limits(vec![(en, true)], 64, 20_000)?;
        eyre::ensure!(
            !torch_is_on(sim.world(), q),
            "enabled data low should reset q"
        );
        eyre::ensure!(
            torch_is_on(sim.world(), nq),
            "enabled data low should set nq"
        );

        Ok(())
    }

    fn trace_d_latch_behavior(
        world: &World3D,
        d: Position,
        en: Position,
        signals: &[(&str, Position)],
    ) -> eyre::Result<()> {
        let mut reset_world = pad_world_top(world, 2);
        reset_world[en].kind = BlockKind::Switch { is_on: true };
        reset_world[d].kind = BlockKind::Switch { is_on: false };
        reset_world.initialize_redstone_states();
        let world = World::from(&reset_world);
        let mut sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        eprintln!("  trace reset: {}", signal_summary(sim.world(), signals));
        eprintln!(
            "  reset support neighbors: {}",
            neighborhood_summary(sim.world(), signals, "reset_support")
        );
        eprintln!(
            "  q support neighbors: {}",
            neighborhood_summary(sim.world(), signals, "q_support")
        );
        eprintln!(
            "  nq support neighbors: {}",
            neighborhood_summary(sim.world(), signals, "nq_support")
        );
        sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
        eprintln!("  trace disabled: {}", signal_summary(sim.world(), signals));
        sim.change_state_with_limits(vec![(d, true)], 64, 20_000)?;
        eprintln!(
            "  trace d high hold: {}",
            signal_summary(sim.world(), signals)
        );
        sim.change_state_with_limits(vec![(en, true)], 64, 20_000)?;
        eprintln!(
            "  trace d high enabled: {}",
            signal_summary(sim.world(), signals)
        );
        sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
        eprintln!(
            "  trace hold q on: {}",
            signal_summary(sim.world(), signals)
        );
        sim.change_state_with_limits(vec![(d, false)], 64, 20_000)?;
        eprintln!(
            "  trace hold q on d low: {}",
            signal_summary(sim.world(), signals)
        );

        Ok(())
    }

    fn signal_summary(world: &World3D, signals: &[(&str, Position)]) -> String {
        signals
            .iter()
            .map(|(name, position)| {
                let value = match world[*position].kind {
                    BlockKind::Switch { is_on } => format!("switch={is_on}"),
                    BlockKind::Torch { is_on } => format!("torch={is_on}"),
                    BlockKind::Redstone { strength, .. } => format!("redstone={strength}"),
                    ref kind => format!("{kind:?}"),
                };
                format!(
                    "{name}@{position:?}/{:?}:{value}",
                    world[*position].direction
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn neighborhood_summary(world: &World3D, signals: &[(&str, Position)], name: &str) -> String {
        let Some((_, center)) = signals.iter().find(|(signal_name, _)| *signal_name == name) else {
            return String::new();
        };
        center
            .forwards()
            .into_iter()
            .filter(|&position| world.size.bound_on(position))
            .map(|position| {
                format!(
                    "{position:?}/{:?}:{:?}",
                    world[position].direction, world[position].kind
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn pad_world_top(world: &World3D, extra_z: usize) -> World3D {
        let mut padded = World3D::new(DimSize(world.size.0, world.size.1, world.size.2 + extra_z));
        for (position, mut block) in world.iter_block() {
            if block.kind.is_cobble() {
                block.kind = BlockKind::Cobble {
                    on_count: 0,
                    on_base_count: 0,
                };
            }
            padded[position] = block;
        }
        padded
    }

    #[test]
    fn test_generate_component_rs_latch() -> eyre::Result<()> {
        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::None,
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectAndRedstone,
            max_not_route_step: 4,
            not_route_step_sampling_policy: SamplingPolicy::Take(4),
            max_route_step: 8,
            route_step_sampling_policy: SamplingPolicy::Take(4),
        };
        let s = Position(6, 2, 1);
        let r = Position(0, 2, 1);
        let mut world = World3D::new(DimSize(12, 9, 5));
        world[s] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[r] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::East,
        };
        let sequential = SequentialPrimitive::new(
            SequentialType::RsLatch,
            vec!["s".to_owned(), "r".to_owned()],
            vec!["q".to_owned()],
        );
        let node = GraphNode {
            id: 2,
            kind: GraphNodeKind::Sequential(sequential.clone()),
            inputs: vec![0, 1],
            ..Default::default()
        };
        let state = [(0, s), (1, r)].into_iter().collect();

        let pairs = select_rs_latch_not_pairs(
            &config,
            &node,
            &sequential,
            &state,
            generate_rs_latch_not_pairs(&world),
        );
        let generated = pairs
            .into_iter()
            .flat_map(|placed| route_rs_latch_branches(&config, &node, &sequential, &state, placed))
            .collect::<Vec<_>>();
        assert!(!generated.is_empty());
        let valid = generated.into_iter().find_map(|placed| {
            assert_rs_latch_behavior(&placed.world, s, r, placed.q_torch, placed.nq_torch)
                .ok()
                .map(|_| placed.world)
        });
        let world = valid
            .expect("expected at least one generated RS latch to pass set/reset/hold simulation");

        save_worlds_to_nbt(vec![world], "test/rs-latch.nbt")?;

        Ok(())
    }

    #[test]
    fn test_generate_component_d_latch() -> eyre::Result<()> {
        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(256),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectAndRedstone,
            max_not_route_step: 4,
            not_route_step_sampling_policy: SamplingPolicy::Random(256),
            max_route_step: 4,
            route_step_sampling_policy: SamplingPolicy::Random(256),
        };
        let d = Position(0, 2, 1);
        let en = Position(0, 6, 1);
        let mut world = World3D::new(DimSize(14, 10, 6));
        world[d] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::East,
        };
        world[en] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::East,
        };
        let sequential = SequentialPrimitive::new(
            SequentialType::DLatch,
            vec!["d".to_owned(), "en".to_owned()],
            vec!["q".to_owned()],
        );
        let node = GraphNode {
            id: 2,
            kind: GraphNodeKind::Sequential(sequential.clone()),
            inputs: vec![0, 1],
            ..Default::default()
        };
        let state = [(0, d), (1, en)].into_iter().collect();

        let generated = generate_d_latch_gate_routes(&config, &node, &sequential, &world, &state);
        assert!(!generated.is_empty());
        let mut checked = 0usize;
        let valid = generated.iter().find_map(|(world, state)| {
            let q = state.port_position(2, "q")?;
            let nq = state.port_position(2, "nq")?;
            checked += 1;
            match std::panic::catch_unwind(|| assert_d_latch_behavior(&world, d, en, q, nq)) {
                Ok(Ok(())) => Some(pad_world_top(&world, 2)),
                Ok(Err(error)) => {
                    if checked <= 3 {
                        eprintln!("candidate {checked} failed: {error}; q={q:?} nq={nq:?}");
                        let q_support = q.walk(world[q].direction).unwrap();
                        let nq_support = nq.walk(world[nq].direction).unwrap();
                        let _ = trace_d_latch_behavior(
                            world,
                            d,
                            en,
                            &[
                                ("d", d),
                                ("en", en),
                                ("not_d", state[&D_LATCH_NOT_D_NODE_ID]),
                                ("set_not_en", state[&D_LATCH_SET_NOT_EN_NODE_ID]),
                                ("set_or", state[&D_LATCH_SET_OR_NODE_ID]),
                                ("reset_or", state[&D_LATCH_RESET_OR_NODE_ID]),
                                ("set", state[&D_LATCH_SET_NODE_ID]),
                                ("reset", state[&D_LATCH_RESET_NODE_ID]),
                                ("q_support", q_support),
                                ("nq_support", nq_support),
                                ("q", q),
                                ("nq", nq),
                            ],
                        );
                    }
                    None
                }
                Err(_) => None,
            }
        });
        let world = valid.unwrap_or_else(|| {
            panic!(
                "expected at least one valid D latch candidate, checked {checked} of {} generated",
                generated.len()
            )
        });

        save_worlds_to_nbt(vec![world], "test/d-latch.nbt")?;

        Ok(())
    }

    #[test]
    fn test_generate_component_xor_simple() -> eyre::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: false,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(100),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 5,
            not_route_step_sampling_policy: SamplingPolicy::Random(100),
            max_route_step: 5,
            route_step_sampling_policy: SamplingPolicy::Random(100),
        };
        let logic_graph = predefined_logics::xor_graph()?;
        let placer = LocalPlacer::new(logic_graph, config)?;
        let worlds = placer.generate(DimSize(10, 10, 5), None);

        let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
        let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
        println!("{}", sample_logic.to_graphviz());

        save_worlds_to_nbt(sampled_worlds, "test/xor-gate-simple.nbt")?;

        Ok(())
    }

    #[test]
    fn test_generate_component_xor_complex() -> eyre::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();
        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: false,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(1000),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: true,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 3,
            not_route_step_sampling_policy: SamplingPolicy::Random(1000),
            max_route_step: 3,
            route_step_sampling_policy: SamplingPolicy::Random(1000),
        };

        let xor_graph = predefined_logics::buffered_xor_graph()?;
        let placer = LocalPlacer::new(xor_graph, config)?;
        let worlds = placer.generate(DimSize(10, 10, 5), None);

        let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
        let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
        println!("{}", sample_logic.to_graphviz());

        save_worlds_to_nbt(sampled_worlds, "test/xor-gate-complex.nbt")?;

        Ok(())
    }

    #[test]
    fn test_generate_component_xor_shortest() -> eyre::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();
        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(1)
        //     .build_global()
        //     .unwrap();

        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(10000),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: true,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 1,
            not_route_step_sampling_policy: SamplingPolicy::Random(1000),
            max_route_step: 1,
            route_step_sampling_policy: SamplingPolicy::Random(1000),
        };

        let xor_graph = predefined_logics::buffered_xor_graph()?;
        let placer = LocalPlacer::new(xor_graph.clone(), config)?;
        let worlds = placer.generate(DimSize(10, 10, 5), None);

        let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
        let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
        println!("{}", sample_logic.to_graphviz());

        for sample in &sampled_worlds {
            // For debug
            // if !equivalent_logic_with_world3d(&xor_graph, sample)? {
            //     let sample_logic = world3d_to_logic(&sample)?.prepare_place()?;
            //     println!("{}", sample_logic.to_graphviz());
            // }
            assert!(equivalent_logic_with_world3d(&xor_graph, sample)?);
        }

        save_worlds_to_nbt(sampled_worlds, "test/xor-gate-shortest.nbt")?;

        Ok(())
    }

    #[test]
    fn test_generate_component_half_adder() -> eyre::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(10000),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 3,
            not_route_step_sampling_policy: SamplingPolicy::Random(100),
            max_route_step: 3,
            route_step_sampling_policy: SamplingPolicy::Random(100),
        };

        let fa_graph = predefined_logics::buffered_half_adder_graph()?;
        println!("{}", fa_graph.to_graphviz());
        let placer = LocalPlacer::new(fa_graph, config)?;
        let worlds = placer.generate(DimSize(10, 10, 5), None);

        let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
        let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
        println!("{}", sample_logic.to_graphviz());

        save_worlds_to_nbt(sampled_worlds, "test/half-adder.nbt")?;

        Ok(())
    }

    #[test]
    #[ignore = "cannot route last or gate inputs"]
    fn test_generate_component_full_adder() -> eyre::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(10000),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 8,
            not_route_step_sampling_policy: SamplingPolicy::Random(100),
            max_route_step: 4,
            route_step_sampling_policy: SamplingPolicy::Random(100),
        };

        let fa_graph = predefined_logics::buffered_full_adder_graph()?;
        println!("{}", fa_graph.to_graphviz());
        let placer = LocalPlacer::new(fa_graph, config)?;
        let worlds = placer.generate(DimSize(10, 10, 5), None);

        let sampled_worlds = SamplingPolicy::Random(1).sample(worlds);
        let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
        println!("{}", sample_logic.to_graphviz());

        save_worlds_to_nbt(sampled_worlds, "test/full-adder.nbt")?;

        Ok(())
    }

    #[test]
    #[ignore = "debug-only: compare full adder local placer cost sampling"]
    fn debug_full_adder_with_cost_sampling() -> eyre::Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let config = LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(10000),
            placement_sampling_policy: PlacementSamplingPolicy::Cost {
                count: 9000,
                random_count: 1000,
                start_step: 28,
            },
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 8,
            not_route_step_sampling_policy: SamplingPolicy::Random(100),
            max_route_step: 8,
            route_step_sampling_policy: SamplingPolicy::Random(100),
        };

        let fa_graph = predefined_logics::buffered_full_adder_graph()?;
        for group in equivalent_expression_groups(&fa_graph).into_iter().take(12) {
            println!(
                "duplicate expression: {} nodes={:?}",
                group.expression, group.node_ids
            );
        }
        let placer = LocalPlacer::new(fa_graph, config)?;
        let mut debug = LocalPlacerDebug::default();
        let worlds = placer.generate_with_debug(DimSize(10, 10, 5), None, &mut debug);

        debug.print_summary();
        if let Some(step) = debug.first_empty_step() {
            println!(
                "first empty step: step={} node={} kind={} inputs={:?} input_positions={:?}",
                step.step + 1,
                step.node_id,
                step.node_kind,
                step.input_node_ids,
                step.input_positions,
            );
        }
        println!("worlds generated: {}", worlds.len());
        if let Some(world) = worlds.into_iter().min_by_key(world_compact_cost) {
            save_worlds_to_nbt(vec![world], "test/full-adder.nbt")?;
        }

        Ok(())
    }
}
