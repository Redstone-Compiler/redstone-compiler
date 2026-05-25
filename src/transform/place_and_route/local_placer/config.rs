use std::collections::HashMap;

use crate::graph::GraphNodeId;
use crate::transform::place_and_route::sampling::SamplingPolicy;
use crate::world::position::Position;

#[derive(Copy, Clone)]
pub struct LocalPlacerConfig {
    pub random_seed: u64,
    pub greedy_input_generation: bool,
    pub input_placement_strategy: InputPlacementStrategy,
    pub input_candidate_limit: Option<usize>,
    pub step_sampling_policy: SamplingPolicy,
    pub placement_sampling_policy: PlacementSamplingPolicy,
    // dealloc 시간을 줄이기 위해 generation을 leak 시킨다.
    pub leak_sampling: bool,
    // torch placement를 input과 direct로 연결하도록 강제한다.
    pub route_torch_directly: bool,
    pub materialize_outputs: bool,
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
            input_candidate_limit: None,
            step_sampling_policy: SamplingPolicy::default(),
            placement_sampling_policy: PlacementSamplingPolicy::default(),
            leak_sampling: false,
            route_torch_directly: true,
            materialize_outputs: false,
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
    /// 입력을 search 영역의 boundary에만 배치한다.
    #[default]
    Boundary,
    /// 입력을 search 영역 내부 어디에나 배치할 수 있다.
    Anywhere,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct LocalPlacerInputConstraints {
    positions_by_node_id: HashMap<GraphNodeId, Vec<Position>>,
    positions_by_input_name: HashMap<String, Vec<Position>>,
}

impl LocalPlacerInputConstraints {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_node_positions(
        mut self,
        node_id: GraphNodeId,
        positions: impl IntoIterator<Item = Position>,
    ) -> Self {
        self.positions_by_node_id
            .insert(node_id, positions.into_iter().collect());
        self
    }

    pub fn with_input_positions(
        mut self,
        input_name: impl Into<String>,
        positions: impl IntoIterator<Item = Position>,
    ) -> Self {
        self.positions_by_input_name
            .insert(input_name.into(), positions.into_iter().collect());
        self
    }

    pub(super) fn positions_for(
        &self,
        node_id: GraphNodeId,
        input_name: &str,
    ) -> Option<Vec<Position>> {
        self.positions_by_node_id
            .get(&node_id)
            .or_else(|| self.positions_by_input_name.get(input_name))
            .cloned()
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum TorchPlacementStrategy {
    /// Torch를 입력과 직접 연결되는 위치에만 배치한다.
    #[default]
    DirectOnly,
    /// Torch를 입력과 인접하지 않은 위치까지 확장해서 배치한다.
    AnywhereNonAdjacent,
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum NotRouteStrategy {
    /// NOT gate 입력을 torch에 직접 연결되는 형태로만 route한다.
    #[default]
    DirectOnly,
    /// NOT gate 입력을 redstone route로만 연결한다.
    RedstoneOnly,
    /// 직접 연결과 redstone route를 모두 후보로 탐색한다.
    DirectAndRedstone,
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum PlacementSamplingPolicy {
    /// 각 step에서 `step_sampling_policy`만 적용한다.
    #[default]
    StepPolicy,
    /// Heuristic cost 상위 후보와 random tail을 남긴다.
    Cost {
        count: usize,
        random_count: usize,
        start_step: usize,
    },
    /// Cost 상위 후보에 더해 geometry diversity가 다른 beam 후보를 보존한다.
    Ranked {
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
            input_candidate_limit: None,
            step_sampling_policy: SamplingPolicy::None,
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: false,
            materialize_outputs: false,
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

    pub fn ranked_sampling(
        count: usize,
        random_count: usize,
        start_step: usize,
    ) -> PlacementSamplingPolicy {
        PlacementSamplingPolicy::Ranked {
            count,
            random_count,
            start_step,
        }
    }

    pub(super) fn sampling_seed(self, scope: u64, step: usize) -> u64 {
        self.random_seed
            ^ scope.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (step as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
    }
}

pub const K_MAX_LOCAL_PLACE_NODE_COUNT: usize = 40;
