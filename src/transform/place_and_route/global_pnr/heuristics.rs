use std::fmt;

use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::physical_intent::ResolvedPhysicalIntent;
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::transform::place_and_route::global_pnr::router::RoutedNet;
use crate::transform::place_and_route::global_pnr::topology::{NetId, ResolvedPnrTopology};

pub struct PlacementHeuristicContext<'a> {
    pub topology: &'a ResolvedPnrTopology,
    pub candidates: &'a [LayoutCandidate],
    pub intent: Option<&'a ResolvedPhysicalIntent>,
}

pub type PlacementTransformFn =
    for<'a> fn(&PlacementHeuristicContext<'a>, &mut Vec<Vec<PlacedModule>>) -> eyre::Result<()>;
pub type PlacementCostFn = for<'a> fn(&PlacementHeuristicContext<'a>, &[PlacedModule]) -> usize;
pub type NetPriorityFn = fn(&ResolvedPnrTopology, NetId, Option<&ResolvedPhysicalIntent>) -> usize;
pub type RouteValidatorFn = fn(&ResolvedPnrTopology, &[RoutedNet]) -> eyre::Result<()>;

#[derive(Clone, Copy)]
pub struct PlacementTransformHook {
    pub name: &'static str,
    pub apply: PlacementTransformFn,
}

#[derive(Clone, Copy)]
pub struct PlacementCostHook {
    pub name: &'static str,
    pub evaluate: PlacementCostFn,
}

#[derive(Clone, Copy)]
pub struct NetPriorityHook {
    pub name: &'static str,
    pub evaluate: NetPriorityFn,
}

#[derive(Clone, Copy)]
pub struct RouteValidatorHook {
    pub name: &'static str,
    pub validate: RouteValidatorFn,
}

#[derive(Clone, Default)]
pub struct GlobalHeuristicHooks {
    pub placement_transforms: Vec<PlacementTransformHook>,
    pub placement_cost_terms: Vec<PlacementCostHook>,
    pub net_priority_terms: Vec<NetPriorityHook>,
    pub route_validators: Vec<RouteValidatorHook>,
}

impl GlobalHeuristicHooks {
    pub fn is_empty(&self) -> bool {
        self.placement_transforms.is_empty()
            && self.placement_cost_terms.is_empty()
            && self.net_priority_terms.is_empty()
            && self.route_validators.is_empty()
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.placement_transforms
            .iter()
            .map(|hook| hook.name)
            .chain(self.placement_cost_terms.iter().map(|hook| hook.name))
            .chain(self.net_priority_terms.iter().map(|hook| hook.name))
            .chain(self.route_validators.iter().map(|hook| hook.name))
            .collect()
    }
}

impl fmt::Debug for GlobalHeuristicHooks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GlobalHeuristicHooks")
            .field(
                "placement_transforms",
                &self
                    .placement_transforms
                    .iter()
                    .map(|hook| hook.name)
                    .collect::<Vec<_>>(),
            )
            .field(
                "placement_cost_terms",
                &self
                    .placement_cost_terms
                    .iter()
                    .map(|hook| hook.name)
                    .collect::<Vec<_>>(),
            )
            .field(
                "net_priority_terms",
                &self
                    .net_priority_terms
                    .iter()
                    .map(|hook| hook.name)
                    .collect::<Vec<_>>(),
            )
            .field(
                "route_validators",
                &self
                    .route_validators
                    .iter()
                    .map(|hook| hook.name)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}
