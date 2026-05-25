use eyre::ContextCompat;
use itertools::Itertools;

use crate::graph::logic::LogicGraph;
use crate::graph::module::{GraphModule, GraphModulePortTarget, GraphModulePortType};
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPort, PhysicalPortDirection,
};
use crate::transform::place_and_route::local_placer::{
    LocalPlacer, LocalPlacerConfig, LocalPlacerInputConstraints,
};
use crate::world::position::{DimSize, Position};

#[derive(Clone)]
pub struct UnitCandidateConfig {
    pub dim: DimSize,
    pub local_config: LocalPlacerConfig,
    pub input_constraints: LocalPlacerInputConstraints,
    pub max_candidates: usize,
}

impl Default for UnitCandidateConfig {
    fn default() -> Self {
        Self {
            dim: DimSize(16, 16, 6),
            local_config: LocalPlacerConfig::default(),
            input_constraints: LocalPlacerInputConstraints::default(),
            max_candidates: 16,
        }
    }
}

pub fn generate_graph_module_candidates(
    module: &GraphModule,
    config: &UnitCandidateConfig,
) -> eyre::Result<Vec<LayoutCandidate>> {
    let graph = module
        .graph
        .clone()
        .context("only graph-backed GraphModule can generate unit layout candidates")?;
    let graph = LogicGraph { graph }.prepare_place()?;
    let placer = LocalPlacer::new(graph, config.local_config)?;

    placer
        .generate_with_outputs_and_input_constraints(config.dim, None, &config.input_constraints)
        .into_iter()
        .take(config.max_candidates)
        .map(|placed| {
            let ports = candidate_ports(module, &config.input_constraints, &placed.outputs);
            LayoutCandidate::from_world(module.name.clone(), placed.world, ports)
        })
        .collect()
}

fn candidate_ports(
    module: &GraphModule,
    input_constraints: &LocalPlacerInputConstraints,
    outputs: &[crate::output::OutputEndpoint],
) -> Vec<PhysicalPort> {
    let mut ports = module
        .ports
        .iter()
        .filter_map(|port| match (&port.port_type, &port.target) {
            (GraphModulePortType::InputNet, GraphModulePortTarget::Node(input_name)) => {
                input_constraints
                    .positions_for_input_name(input_name)
                    .and_then(|positions| positions.into_iter().next())
                    .map(|position| PhysicalPort {
                        name: port.name.clone(),
                        direction: PhysicalPortDirection::Input,
                        position,
                    })
            }
            (GraphModulePortType::OutputNet, GraphModulePortTarget::Node(output_name)) => outputs
                .iter()
                .find(|output| output.name == *output_name)
                .map(|output| PhysicalPort {
                    name: port.name.clone(),
                    direction: PhysicalPortDirection::Output,
                    position: output.position(),
                }),
            _ => None,
        })
        .collect_vec();
    ports.sort_by(|a, b| a.name.cmp(&b.name));
    ports
}

pub fn d_latch_child_candidate_config(local_config: LocalPlacerConfig) -> UnitCandidateConfig {
    UnitCandidateConfig {
        dim: DimSize(14, 10, 6),
        local_config,
        input_constraints: LocalPlacerInputConstraints::new()
            .with_input_positions("d", [Position(0, 2, 1)])
            .with_input_positions("en", [Position(0, 6, 1)]),
        max_candidates: 1,
    }
}
