use crate::transform::place_and_route::sampling::SamplingPolicy;

#[derive(Copy, Clone)]
pub struct LocalPlacerConfig {
    pub random_seed: u64,
    pub greedy_input_generation: bool,
    pub input_placement_strategy: InputPlacementStrategy,
    pub step_sampling_policy: SamplingPolicy,
    pub placement_sampling_policy: PlacementSamplingPolicy,
    // dealloc 시간을 줄이기 위해 generation을 leak 시킨다.
    pub leak_sampling: bool,
    // torch placement를 input과 direct로 연결하도록 강제한다.
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

    pub(super) fn sampling_seed(self, scope: u64, step: usize) -> u64 {
        self.random_seed
            ^ scope.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (step as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
    }
}

pub const K_MAX_LOCAL_PLACE_NODE_COUNT: usize = 40;
