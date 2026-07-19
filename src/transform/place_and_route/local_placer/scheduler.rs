use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap, HashSet};

use super::PlacementSchedulePolicy;
use crate::graph::logic::LogicGraph;
use crate::graph::GraphNodeId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlacementScheduleMetrics {
    pub input_lifetime: usize,
    pub peak_frontier: usize,
    pub total_frontier: usize,
    pub edge_lifetime: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementSchedule {
    pub order: Vec<GraphNodeId>,
    pub metrics: PlacementScheduleMetrics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScheduleHeuristic {
    MinFrontier,
    Reconvergence,
    CriticalPath,
}

pub struct PlacementScheduler<'a> {
    graph: &'a LogicGraph,
}

impl<'a> PlacementScheduler<'a> {
    pub fn new(graph: &'a LogicGraph) -> Self {
        Self { graph }
    }

    pub fn select(&self, policy: PlacementSchedulePolicy) -> PlacementSchedule {
        match policy {
            PlacementSchedulePolicy::Topological => self.from_order(self.graph.topological_order()),
            PlacementSchedulePolicy::MinFrontier => {
                self.from_order(self.list_schedule(ScheduleHeuristic::MinFrontier))
            }
            PlacementSchedulePolicy::Reconvergence => {
                self.from_order(self.list_schedule(ScheduleHeuristic::Reconvergence))
            }
            // A single auto-scheduled placer keeps the canonical order. The
            // global candidate generator can safely try the other schedules
            // as fallbacks after this known-compatible baseline fails.
            PlacementSchedulePolicy::Auto => self.from_order(self.graph.topological_order()),
        }
    }

    pub fn candidates(&self) -> Vec<PlacementSchedule> {
        let orders = [
            self.graph.topological_order(),
            self.list_schedule(ScheduleHeuristic::MinFrontier),
            self.list_schedule(ScheduleHeuristic::Reconvergence),
            self.list_schedule(ScheduleHeuristic::CriticalPath),
        ];
        let mut seen = HashSet::new();
        orders
            .into_iter()
            .filter(|order| seen.insert(order.clone()))
            .map(|order| self.from_order(order))
            .collect()
    }

    fn from_order(&self, order: Vec<GraphNodeId>) -> PlacementSchedule {
        PlacementSchedule {
            metrics: schedule_metrics(self.graph, &order),
            order,
        }
    }

    fn list_schedule(&self, heuristic: ScheduleHeuristic) -> Vec<GraphNodeId> {
        let node_ids = self
            .graph
            .nodes
            .iter()
            .map(|node| node.id)
            .collect::<HashSet<_>>();
        let mut remaining_inputs = self
            .graph
            .nodes
            .iter()
            .map(|node| {
                (
                    node.id,
                    node.inputs
                        .iter()
                        .filter(|input| node_ids.contains(input))
                        .count(),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut ready = remaining_inputs
            .iter()
            .filter_map(|(node_id, count)| (*count == 0).then_some(*node_id))
            .collect::<BTreeSet<_>>();
        let critical_depth = critical_depths(self.graph);
        let mut scheduled = HashSet::new();
        let mut order = Vec::with_capacity(node_ids.len());

        while !ready.is_empty() {
            let node_id = ready
                .iter()
                .copied()
                .min_by_key(|candidate| {
                    self.priority(heuristic, *candidate, &scheduled, &critical_depth)
                })
                .unwrap();
            ready.remove(&node_id);
            scheduled.insert(node_id);
            order.push(node_id);

            let node = self.graph.find_node_by_id(node_id).unwrap();
            for consumer in &node.outputs {
                let Some(count) = remaining_inputs.get_mut(consumer) else {
                    continue;
                };
                *count = count.saturating_sub(1);
                if *count == 0 && !scheduled.contains(consumer) {
                    ready.insert(*consumer);
                }
            }
        }

        assert_eq!(
            order.len(),
            node_ids.len(),
            "local placement graph must be acyclic"
        );
        order
    }

    fn priority(
        &self,
        heuristic: ScheduleHeuristic,
        candidate: GraphNodeId,
        scheduled: &HashSet<GraphNodeId>,
        critical_depth: &HashMap<GraphNodeId, usize>,
    ) -> (usize, usize, usize, Reverse<usize>, GraphNodeId) {
        let mut after = scheduled.clone();
        after.insert(candidate);
        let frontier = frontier_size(self.graph, &after);
        let node = self.graph.find_node_by_id(candidate).unwrap();
        let closes = node
            .inputs
            .iter()
            .filter(|input| {
                self.graph.find_node_by_id(**input).is_some_and(|producer| {
                    producer
                        .outputs
                        .iter()
                        .all(|consumer| after.contains(consumer))
                })
            })
            .count();
        let unlocks_reconvergence = node
            .outputs
            .iter()
            .filter_map(|consumer| self.graph.find_node_by_id(*consumer))
            .filter(|consumer| {
                consumer.inputs.len() > 1
                    && consumer.inputs.iter().all(|input| after.contains(input))
            })
            .count();
        let depth = critical_depth.get(&candidate).copied().unwrap_or_default();

        match heuristic {
            ScheduleHeuristic::MinFrontier => (
                frontier,
                usize::MAX - closes,
                usize::MAX - unlocks_reconvergence,
                Reverse(depth),
                candidate,
            ),
            ScheduleHeuristic::Reconvergence => (
                usize::MAX - unlocks_reconvergence,
                frontier,
                usize::MAX - closes,
                Reverse(depth),
                candidate,
            ),
            ScheduleHeuristic::CriticalPath => (
                usize::MAX - depth,
                frontier,
                usize::MAX - unlocks_reconvergence,
                Reverse(closes),
                candidate,
            ),
        }
    }
}

fn frontier_size(graph: &LogicGraph, scheduled: &HashSet<GraphNodeId>) -> usize {
    scheduled
        .iter()
        .filter(|node_id| {
            graph.find_node_by_id(**node_id).is_some_and(|node| {
                node.outputs
                    .iter()
                    .any(|consumer| !scheduled.contains(consumer))
            })
        })
        .count()
}

fn critical_depths(graph: &LogicGraph) -> HashMap<GraphNodeId, usize> {
    let mut depths = HashMap::new();
    for node_id in graph.topological_order().into_iter().rev() {
        let depth = graph
            .find_node_by_id(node_id)
            .unwrap()
            .outputs
            .iter()
            .filter_map(|consumer| depths.get(consumer))
            .copied()
            .max()
            .unwrap_or(0)
            + 1;
        depths.insert(node_id, depth);
    }
    depths
}

fn schedule_metrics(graph: &LogicGraph, order: &[GraphNodeId]) -> PlacementScheduleMetrics {
    let positions = order
        .iter()
        .enumerate()
        .map(|(index, node_id)| (*node_id, index))
        .collect::<HashMap<_, _>>();
    let mut scheduled = HashSet::new();
    let mut peak_frontier = 0;
    let mut total_frontier = 0;
    for node_id in order {
        scheduled.insert(*node_id);
        let frontier = frontier_size(graph, &scheduled);
        peak_frontier = peak_frontier.max(frontier);
        total_frontier += frontier;
    }
    let edge_lifetime = graph
        .nodes
        .iter()
        .map(|node| {
            node.outputs
                .iter()
                .filter_map(|consumer| {
                    Some(positions.get(consumer)?.saturating_sub(positions[&node.id]))
                })
                .sum::<usize>()
        })
        .sum();
    let input_lifetime = graph
        .nodes
        .iter()
        .filter(|node| node.kind.is_input())
        .filter_map(|node| {
            let input_position = positions[&node.id];
            node.outputs
                .iter()
                .filter_map(|consumer| positions.get(consumer))
                .min()
                .map(|consumer_position| consumer_position.saturating_sub(input_position))
        })
        .sum();
    PlacementScheduleMetrics {
        input_lifetime,
        peak_frontier,
        total_frontier,
        edge_lifetime,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::predefined_logics;

    #[test]
    fn full_adder_scheduler_generates_distinct_legal_orders() -> eyre::Result<()> {
        let graph = predefined_logics::full_adder_graph()?;
        let scheduler = PlacementScheduler::new(&graph);
        let candidates = scheduler.candidates();

        assert!(candidates.len() >= 2);
        for schedule in &candidates {
            let position = schedule
                .order
                .iter()
                .enumerate()
                .map(|(index, node_id)| (*node_id, index))
                .collect::<HashMap<_, _>>();
            for node in &graph.nodes {
                for input in &node.inputs {
                    assert!(position[input] < position[&node.id]);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn auto_schedule_uses_the_safe_topological_baseline() -> eyre::Result<()> {
        let graph = predefined_logics::full_adder_graph()?;
        let scheduler = PlacementScheduler::new(&graph);
        let auto = scheduler.select(PlacementSchedulePolicy::Auto);

        assert_eq!(auto.order, graph.topological_order());
        Ok(())
    }
}
