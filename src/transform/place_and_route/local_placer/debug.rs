use std::collections::HashMap;

use serde::Serialize;

use crate::graph::GraphNodeId;
pub use crate::transform::place_and_route::detailed_router::RouteRejectReason;
use crate::world::position::Position;

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

    /// Returns the first stage that exhausted the placement frontier.
    ///
    /// `sampled_len` is used instead of only `generated_len`: a bounded
    /// sampling policy can also intentionally reduce a non-empty expansion to
    /// an empty frontier.
    pub fn failure(&self) -> Option<LocalPlacementFailure> {
        self.steps
            .iter()
            .find(|step| step.sampled_len == 0)
            .map(|step| {
                let kind = if step.input_queue_len == 0 {
                    LocalPlacementFailureKind::InitialFrontierEmpty
                } else if step.generated_len > 0 {
                    LocalPlacementFailureKind::PlacementBeamExhausted
                } else if let Some(route) = &step.route_debug {
                    if route
                        .depths
                        .last()
                        .is_some_and(|depth| depth.next_frontier_after_sampling > 0)
                    {
                        LocalPlacementFailureKind::RouteDepthExhausted
                    } else if route.depths.iter().any(|depth| {
                        depth.next_frontier_before_sampling > depth.next_frontier_after_sampling
                    }) {
                        LocalPlacementFailureKind::RouteBeamExhausted
                    } else {
                        LocalPlacementFailureKind::NoLegalRoute
                    }
                } else {
                    LocalPlacementFailureKind::NoLegalPlacement
                };
                LocalPlacementFailure {
                    stage: step.stage,
                    kind,
                    step: step.step,
                    total_steps: step.total_steps,
                    node_id: step.node_id,
                    node_kind: step.node_kind.clone(),
                    input_candidates: step.input_queue_len,
                    generated_candidates: step.generated_len,
                    route_calls: step
                        .route_debug
                        .as_ref()
                        .map_or(0, |debug| debug.route_calls),
                    route_candidates: step
                        .route_debug
                        .as_ref()
                        .map_or(0, |debug| debug.candidates_found),
                }
            })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPlacementStage {
    InputPlacement,
    NotRouting,
    OrRouting,
    SequentialPlacement,
    OutputPlacement,
    OtherPlacement,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPlacementFailureKind {
    InitialFrontierEmpty,
    PlacementBeamExhausted,
    RouteDepthExhausted,
    RouteBeamExhausted,
    NoLegalRoute,
    NoLegalPlacement,
}

impl Default for LocalPlacementStage {
    fn default() -> Self {
        Self::OtherPlacement
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalPlacementFailure {
    pub stage: LocalPlacementStage,
    pub kind: LocalPlacementFailureKind,
    pub step: usize,
    pub total_steps: usize,
    pub node_id: GraphNodeId,
    pub node_kind: String,
    pub input_candidates: usize,
    pub generated_candidates: usize,
    pub route_calls: usize,
    pub route_candidates: usize,
}

#[derive(Debug, Default)]
pub struct StepDebug {
    pub step: usize,
    pub total_steps: usize,
    pub node_id: GraphNodeId,
    pub node_kind: String,
    pub stage: LocalPlacementStage,
    pub input_node_ids: Vec<GraphNodeId>,
    pub input_positions: Vec<(GraphNodeId, Position)>,
    pub input_queue_len: usize,
    pub generated_len: usize,
    pub sampled_len: usize,
    pub route_debug: Option<RouteDebug>,
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
    pub(super) fn merge(&mut self, other: RouteDebug) {
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

    pub(super) fn reject(&mut self, reason: RouteRejectReason) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_reports_the_stage_that_first_exhausted_the_frontier() {
        let debug = LocalPlacerDebug {
            steps: vec![
                StepDebug {
                    sampled_len: 4,
                    ..Default::default()
                },
                StepDebug {
                    step: 1,
                    total_steps: 3,
                    node_id: 7,
                    node_kind: "Logic(Or)".to_owned(),
                    stage: LocalPlacementStage::OrRouting,
                    input_queue_len: 4,
                    generated_len: 0,
                    sampled_len: 0,
                    route_debug: Some(RouteDebug {
                        route_calls: 4,
                        candidates_found: 0,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        };

        let failure = debug.failure().expect("failure");
        assert_eq!(failure.stage, LocalPlacementStage::OrRouting);
        assert_eq!(failure.kind, LocalPlacementFailureKind::NoLegalRoute);
        assert_eq!(failure.step, 1);
        assert_eq!(failure.input_candidates, 4);
        assert_eq!(failure.route_calls, 4);
    }
}
