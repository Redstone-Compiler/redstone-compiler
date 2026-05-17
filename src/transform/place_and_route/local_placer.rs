use std::borrow::Cow;
use std::collections::HashMap;
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
use crate::sequential::SequentialPrimitive;
use crate::transform::place_and_route::estimate::world_compact_cost;
use crate::transform::place_and_route::place_bound::PropagateType;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

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
            route_torch_directly: false,
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
                    !SequentialMacro::candidates(sequential).is_empty(),
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
        mut debug: Option<&mut LocalPlacerDebug>,
    ) -> Vec<World3D> {
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
        queue.into_iter().map(|(world, _)| world).collect()
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
            GraphNodeKind::Sequential(ref sequential) => {
                generate_sequential_macro_routes(&self.config, node, sequential, &world, state)
            }
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
                            for output_port in &sequential.output_ports {
                                if let Some(position) = placed.output_ports.get(output_port) {
                                    state.set_port_position(
                                        node.id,
                                        output_port.to_owned(),
                                        *position,
                                    );
                                }
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
    torch_pos: Position,
    cobble_pos: Position,
) -> Vec<(World3D, Position)> {
    let source_node = PlacedNode::new(source, world[source]);
    assert!(source_node.is_propagation_target() && world[cobble_pos].kind.is_cobble());

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

            if directly_connected {
                route_candidates.push((world.clone(), start.position(), 0usize));
                continue;
            }
        }
    }

    if matches!(
        config.not_route_strategy,
        NotRouteStrategy::RedstoneOnly | NotRouteStrategy::DirectAndRedstone
    ) && source_node.is_diode()
    {
        let route_config = LocalPlacerConfig {
            max_route_step: config.max_not_route_step,
            route_step_sampling_policy: config.not_route_step_sampling_policy,
            ..*config
        };
        route_candidates.extend(
            generate_or_routes(&route_config, world, source, torch_pos)
                .routes
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
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    if !cobble_node.has_conflict(world, &except.iter().copied().collect()) {
        Some(cobble_node)
    } else {
        None
    }
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
    use crate::graph::logic::{predefined_logics, LogicGraph};
    use crate::graph::{Graph, GraphNode, GraphNodeKind};
    use crate::nbt::{NBTRoot, ToNBT};
    use crate::sequential::{SequentialPrimitive, SequentialType};
    use crate::transform::place_and_route::estimate::world_compact_cost;
    use crate::transform::place_and_route::local_placer::{
        InputPlacementStrategy, LocalPlacer, LocalPlacerConfig, LocalPlacerDebug, NotRouteStrategy,
        PlacementSamplingPolicy, SamplingPolicy, TorchPlacementStrategy,
    };
    use crate::transform::place_and_route::utils::{
        equivalent_logic_with_world3d, equivalent_logic_with_world3ds, world3d_to_logic,
    };
    use crate::world::block::BlockKind;
    use crate::world::position::DimSize;
    use crate::world::World3D;

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
            route_torch_directly: true,
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
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 1,
            not_route_step_sampling_policy: SamplingPolicy::None,
            max_route_step: 1,
            route_step_sampling_policy: SamplingPolicy::None,
        };
        let mut graph = Graph {
            nodes: vec![
                GraphNode {
                    id: 0,
                    kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
                        SequentialType::RsLatch,
                        Vec::new(),
                        vec!["q".to_owned()],
                    )),
                    ..Default::default()
                },
                GraphNode {
                    id: 1,
                    kind: GraphNodeKind::Output("q".to_owned()),
                    inputs: vec![0],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        graph.build_outputs();
        graph.build_inputs();
        graph.build_producers();
        graph.build_consumers();

        let placer = LocalPlacer::new(LogicGraph { graph }, config)?;
        let worlds = placer.generate(DimSize(8, 6, 3), None);
        assert!(!worlds.is_empty());
        assert!(worlds.iter().any(|world| world.iter_block().len() >= 20));
        assert!(worlds.iter().any(|world| {
            world
                .iter_block()
                .into_iter()
                .any(|(_, block)| matches!(block.kind, BlockKind::Torch { .. }))
        }));

        save_worlds_to_nbt(vec![worlds[0].clone()], "test/rs-latch.nbt")?;

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
            max_route_step: 8,
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
