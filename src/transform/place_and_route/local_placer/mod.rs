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
use crate::transform::place_and_route::estimate::world_compact_cost;
use crate::transform::place_and_route::place_bound::PropagateType;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

mod config;
mod debug;
mod state;

pub use config::{
    InputPlacementStrategy, LocalPlacerConfig, NotRouteStrategy, PlacementSamplingPolicy,
    TorchPlacementStrategy, K_MAX_LOCAL_PLACE_NODE_COUNT,
};
pub use debug::{LocalPlacerDebug, RouteDebug, RouteDepthDebug, RouteRejectReason, StepDebug};
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
    cost_join_pairs_by_step: Vec<Vec<(GraphNodeId, GraphNodeId)>>,
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
