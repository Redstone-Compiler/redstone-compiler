use crate::transform::place_and_route::global_pnr::candidate::UnitCandidateConfig;
use crate::transform::place_and_route::global_pnr::placer::GlobalPlacementConfig;
use crate::transform::place_and_route::global_pnr::router::{
    GlobalRoutingConfig, NetOrderStrategy,
};
use crate::transform::place_and_route::global_pnr::{GlobalPnrConfig, GlobalSearchConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalPnrPreset {
    Fast,
    Balanced,
    Thorough,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalSearchBudget {
    pub max_candidates_per_child: usize,
    pub max_layout_combinations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementHeuristic {
    Shelf,
    Grid,
    RegisterCarryChain,
    RegisterCarryAlignedSlices,
    RegisterGrid,
    RegisterTriangles,
    RegisterSlices,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalPnrPolicies {
    pub placement_heuristics: Vec<PlacementHeuristic>,
    pub net_order_strategies: Vec<NetOrderStrategy>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementCostWeights {
    pub placement_volume: usize,
    pub estimated_wire_length: usize,
    pub vertical_distance: usize,
}

impl Default for PlacementCostWeights {
    fn default() -> Self {
        Self {
            placement_volume: 1,
            estimated_wire_length: 8,
            vertical_distance: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlacementCostBreakdown {
    pub placement_volume: usize,
    pub estimated_wire_length: usize,
    pub vertical_distance: usize,
}

impl PlacementCostBreakdown {
    pub fn weighted_total(self, weights: PlacementCostWeights) -> usize {
        self.placement_volume
            .saturating_mul(weights.placement_volume)
            .saturating_add(
                self.estimated_wire_length
                    .saturating_mul(weights.estimated_wire_length),
            )
            .saturating_add(
                self.vertical_distance
                    .saturating_mul(weights.vertical_distance),
            )
    }
}

impl GlobalPnrPolicies {
    pub fn balanced() -> Self {
        Self {
            placement_heuristics: vec![
                PlacementHeuristic::Shelf,
                PlacementHeuristic::Grid,
                PlacementHeuristic::RegisterCarryChain,
                PlacementHeuristic::RegisterCarryAlignedSlices,
                PlacementHeuristic::RegisterGrid,
                PlacementHeuristic::RegisterTriangles,
                PlacementHeuristic::RegisterSlices,
            ],
            net_order_strategies: vec![
                NetOrderStrategy::Criticality,
                NetOrderStrategy::HighestFanoutFirst,
                NetOrderStrategy::ReverseCriticality,
            ],
        }
    }
}

impl GlobalPnrPreset {
    pub fn search_config(self) -> GlobalSearchConfig {
        let budget = match self {
            Self::Fast => GlobalSearchBudget {
                max_candidates_per_child: 2,
                max_layout_combinations: 4,
            },
            Self::Balanced => GlobalSearchBudget {
                max_candidates_per_child: 4,
                max_layout_combinations: 16,
            },
            Self::Thorough => GlobalSearchBudget {
                max_candidates_per_child: 8,
                max_layout_combinations: 64,
            },
        };
        let mut policies = GlobalPnrPolicies::balanced();
        if self == Self::Fast {
            policies.net_order_strategies.truncate(1);
        }
        GlobalSearchConfig { budget, policies }
    }

    pub fn config(self) -> GlobalPnrConfig {
        GlobalPnrConfig {
            candidate: UnitCandidateConfig::default(),
            placement: GlobalPlacementConfig::default(),
            routing: GlobalRoutingConfig::default(),
            search: self.search_config(),
            show_progress: true,
            verifier: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GlobalPnrPreset, PlacementCostBreakdown, PlacementCostWeights};
    use crate::transform::place_and_route::global_pnr::GlobalPnrConfig;

    #[test]
    fn presets_increase_global_search_budget_monotonically() {
        let fast = GlobalPnrPreset::Fast.config();
        let balanced = GlobalPnrPreset::Balanced.config();
        let thorough = GlobalPnrPreset::Thorough.config();

        assert!(
            fast.search.budget.max_candidates_per_child
                <= balanced.search.budget.max_candidates_per_child
        );
        assert!(
            fast.search.budget.max_layout_combinations
                <= balanced.search.budget.max_layout_combinations
        );
        assert!(
            balanced.search.budget.max_candidates_per_child
                <= thorough.search.budget.max_candidates_per_child
        );
        assert!(
            balanced.search.budget.max_layout_combinations
                <= thorough.search.budget.max_layout_combinations
        );
    }

    #[test]
    fn default_global_config_uses_balanced_policy() {
        let default = GlobalPnrConfig::default();
        let balanced = GlobalPnrPreset::Balanced.config();

        assert_eq!(default.search, balanced.search);
    }

    #[test]
    fn placement_cost_breakdown_uses_adjustable_weights() {
        let cost = PlacementCostBreakdown {
            placement_volume: 100,
            estimated_wire_length: 10,
            vertical_distance: 2,
        };

        assert_eq!(cost.weighted_total(PlacementCostWeights::default()), 212);
        assert_eq!(
            cost.weighted_total(PlacementCostWeights {
                vertical_distance: 100,
                ..PlacementCostWeights::default()
            }),
            380
        );
    }
}
