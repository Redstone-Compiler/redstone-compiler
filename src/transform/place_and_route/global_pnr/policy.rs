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
    Layered3D(LayeredPlacementConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayeredPlacementConfig {
    pub layers: usize,
    pub layer_spacing: usize,
    pub assignment: LayerAssignmentStrategy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutingCongestionConfig {
    pub bin_size_xy: usize,
    pub bin_size_z: usize,
}

impl Default for RoutingCongestionConfig {
    fn default() -> Self {
        Self {
            bin_size_xy: 8,
            bin_size_z: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerAssignmentStrategy {
    Alternating,
    NetAware,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalPnrPolicies {
    pub placement_heuristics: Vec<PlacementHeuristic>,
    pub net_order_strategies: Vec<NetOrderStrategy>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementCostWeights {
    pub placement_volume: usize,
    pub xy_footprint: usize,
    pub height_span: usize,
    pub estimated_wire_length: usize,
    pub vertical_distance: usize,
    pub routing_congestion: usize,
}

impl Default for PlacementCostWeights {
    fn default() -> Self {
        Self {
            placement_volume: 1,
            xy_footprint: 0,
            height_span: 0,
            estimated_wire_length: 8,
            vertical_distance: 16,
            routing_congestion: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlacementCostBreakdown {
    pub placement_volume: usize,
    pub xy_footprint: usize,
    pub height_span: usize,
    pub estimated_wire_length: usize,
    pub vertical_distance: usize,
    pub routing_congestion: usize,
}

impl PlacementCostBreakdown {
    pub fn weighted_total(self, weights: PlacementCostWeights) -> usize {
        self.placement_volume
            .saturating_mul(weights.placement_volume)
            .saturating_add(self.xy_footprint.saturating_mul(weights.xy_footprint))
            .saturating_add(self.height_span.saturating_mul(weights.height_span))
            .saturating_add(
                self.estimated_wire_length
                    .saturating_mul(weights.estimated_wire_length),
            )
            .saturating_add(
                self.vertical_distance
                    .saturating_mul(weights.vertical_distance),
            )
            .saturating_add(
                self.routing_congestion
                    .saturating_mul(weights.routing_congestion),
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
        } else if self == Self::Thorough {
            policies
                .placement_heuristics
                .push(PlacementHeuristic::Layered3D(LayeredPlacementConfig {
                    layers: 2,
                    layer_spacing: 4,
                    assignment: LayerAssignmentStrategy::NetAware,
                }));
        }
        GlobalSearchConfig { budget, policies }
    }

    pub fn config(self) -> GlobalPnrConfig {
        let mut config = GlobalPnrConfig {
            candidate: UnitCandidateConfig::default(),
            placement: GlobalPlacementConfig::default(),
            routing: GlobalRoutingConfig::default(),
            search: self.search_config(),
            show_progress: true,
            verifier: None,
        };
        if self == Self::Thorough {
            config.placement.cost_weights.routing_congestion = 4;
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GlobalPnrPreset, PlacementCostBreakdown, PlacementCostWeights, PlacementHeuristic,
    };
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
    fn layered_3d_is_opt_in_through_the_thorough_preset() {
        let balanced = GlobalPnrPreset::Balanced.config();
        let thorough = GlobalPnrPreset::Thorough.config();

        assert!(!balanced
            .search
            .policies
            .placement_heuristics
            .iter()
            .any(|heuristic| matches!(heuristic, PlacementHeuristic::Layered3D(_))));
        assert!(thorough
            .search
            .policies
            .placement_heuristics
            .iter()
            .any(|heuristic| matches!(heuristic, PlacementHeuristic::Layered3D(_))));
    }

    #[test]
    fn routing_congestion_cost_is_opt_in_through_the_thorough_preset() {
        let balanced = GlobalPnrPreset::Balanced.config();
        let thorough = GlobalPnrPreset::Thorough.config();

        assert_eq!(balanced.placement.cost_weights.routing_congestion, 0);
        assert!(thorough.placement.cost_weights.routing_congestion > 0);
    }

    #[test]
    fn placement_cost_breakdown_uses_adjustable_weights() {
        let cost = PlacementCostBreakdown {
            placement_volume: 100,
            xy_footprint: 40,
            height_span: 8,
            estimated_wire_length: 10,
            vertical_distance: 2,
            routing_congestion: 7,
        };

        assert_eq!(cost.weighted_total(PlacementCostWeights::default()), 212);
        assert_eq!(
            cost.weighted_total(PlacementCostWeights {
                vertical_distance: 100,
                ..PlacementCostWeights::default()
            }),
            380
        );
        assert_eq!(
            cost.weighted_total(PlacementCostWeights {
                placement_volume: 0,
                xy_footprint: 1,
                height_span: 10,
                estimated_wire_length: 0,
                vertical_distance: 0,
                routing_congestion: 11,
            }),
            197
        );
    }
}
