use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::RoutableDesign;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoutableDocument {
    pub design: RoutableDesign,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub candidate_profiles: BTreeMap<String, CandidateSpec>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub design_profiles: BTreeMap<String, PnrSpec>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub candidate_bindings: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub design_bindings: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub pin_search: BTreeMap<PortRef, Vec<[usize; 3]>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical: Option<PhysicalSpec>,
}

impl RoutableDocument {
    pub fn circuit_only(design: RoutableDesign) -> Self {
        Self {
            design,
            candidate_profiles: BTreeMap::new(),
            design_profiles: BTreeMap::new(),
            candidate_bindings: BTreeMap::new(),
            design_bindings: BTreeMap::new(),
            pin_search: BTreeMap::new(),
            physical: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PortRef {
    pub definition: String,
    pub port: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PnrSpec {
    pub placement: PlacementSpec,
    pub routing: RoutingSpec,
    pub search: SearchSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSpec {
    pub search_box: [usize; 3],
    pub retain: usize,
    pub combinational_samples: Option<usize>,
    pub local_placer: LocalPlacerSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalPlacerSpec {
    pub random_seed: u64,
    pub greedy_input_generation: bool,
    pub input_placement: InputPlacementSpec,
    pub input_candidate_limit: Option<usize>,
    pub step_sampling: SamplingSpec,
    pub placement_sampling: PlacementSamplingSpec,
    pub leak_sampling: bool,
    pub route_torch_directly: bool,
    pub materialize_outputs: bool,
    pub torch_placement: TorchPlacementSpec,
    pub not_route_strategy: NotRouteSpec,
    pub max_not_route_step: usize,
    pub not_route_step_sampling: SamplingSpec,
    pub max_route_step: usize,
    pub route_step_sampling: SamplingSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputPlacementSpec {
    Boundary,
    Anywhere,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "count", rename_all = "snake_case")]
pub enum SamplingSpec {
    None,
    Take(usize),
    Random(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlacementSamplingSpec {
    StepPolicy,
    Cost {
        count: usize,
        random_count: usize,
        start_step: usize,
    },
    Ranked {
        count: usize,
        random_count: usize,
        start_step: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TorchPlacementSpec {
    DirectOnly,
    AnywhereNonAdjacent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotRouteSpec {
    DirectOnly,
    RedstoneOnly,
    DirectAndRedstone,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementSpec {
    pub initial_spacing: usize,
    pub shelf_width: usize,
    pub max_attempts: usize,
    pub heuristics: Vec<PlacementHeuristicSpec>,
    pub congestion: CongestionSpec,
    pub objective: ObjectiveSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlacementHeuristicSpec {
    Shelf,
    Grid,
    RegisterCarryChain,
    RegisterCarryAlignedSlices,
    RegisterGrid,
    RegisterTriangles,
    RegisterSlices,
    Layered3d {
        layers: usize,
        layer_spacing: usize,
        assignment: LayerAssignmentSpec,
    },
    Free3dSweep(Free3dSweepSpec),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Free3dSweepSpec {
    pub seeds: Vec<u64>,
    pub clearances: Vec<usize>,
    pub iterations: usize,
    pub step_size: f64,
    pub attraction: f64,
    pub repulsion: f64,
    pub compactness: f64,
    pub damping: f64,
    pub vertical_scale: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerAssignmentSpec {
    Alternating,
    NetAware,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CongestionSpec {
    pub bin_size_xy: usize,
    pub bin_size_z: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    pub placement_volume: usize,
    pub xy_footprint: usize,
    pub height_span: usize,
    pub estimated_wire_length: usize,
    pub vertical_distance: usize,
    pub routing_congestion: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingSpec {
    pub probe: Option<RouteStageSpec>,
    pub primary: RouteStageSpec,
    pub refinement: Option<RouteStageSpec>,
    pub net_order: Vec<NetOrderSpec>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteStageSpec {
    pub strategy: RouteStrategySpec,
    pub validation: RouteValidationSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RouteStrategySpec {
    BreadthFirst,
    AStar,
    DirectGreedy {
        max_steps: usize,
    },
    GreedyBeam {
        width: usize,
        max_expansions: usize,
        variant_seed: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteValidationSpec {
    Incremental,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetOrderSpec {
    Criticality,
    HighestFanoutFirst,
    ReverseCriticality,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchSpec {
    pub candidates_per_child: usize,
    pub layout_combinations: usize,
    pub detailed_routing_attempts: usize,
    pub refined_routing_attempts: usize,
    pub refinement_rounds: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalSpec {
    pub regions: BTreeMap<String, PhysicalRegionSpec>,
    pub constraints: Vec<PhysicalConstraintSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalRegionSpec {
    pub min: [usize; 3],
    pub max: [usize; 3],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhysicalConstraintSpec {
    Inside {
        id: String,
        instance: String,
        region: String,
    },
    LayerRange {
        id: String,
        instance: String,
        min: usize,
        max: usize,
    },
    FixedOrigin {
        id: String,
        instance: String,
        origin: [usize; 3],
    },
    NetPriority {
        id: String,
        net: String,
        priority: usize,
    },
    NetAvoid {
        id: String,
        net: String,
        region: String,
    },
    PreferInside {
        id: String,
        instance: String,
        region: String,
        strength: PreferenceSpec,
    },
    SameLayer {
        id: String,
        first: String,
        second: String,
        strength: PreferenceSpec,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceSpec {
    Weak,
    Medium,
    Strong,
}
