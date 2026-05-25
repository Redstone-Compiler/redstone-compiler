pub mod assembly;
pub mod candidate;
pub mod ir;
pub mod placer;
pub mod router;

use eyre::ContextCompat;

use crate::graph::module::{GraphModule, GraphModuleContext};
use crate::transform::place_and_route::global_pnr::assembly::assemble_world;
use crate::transform::place_and_route::global_pnr::candidate::{
    generate_graph_module_candidates, UnitCandidateConfig,
};
use crate::transform::place_and_route::global_pnr::placer::{
    place_candidates_on_shelves, GlobalPlacementConfig,
};
use crate::transform::place_and_route::global_pnr::router::route_module_variables;
use crate::world::World3D;

#[derive(Clone, Default)]
pub struct GlobalPnrConfig {
    pub candidate: UnitCandidateConfig,
    pub placement: GlobalPlacementConfig,
}

pub fn place_and_route_module(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    if module.graph.is_some() {
        let candidates = generate_graph_module_candidates(module, &config.candidate)?;
        let candidate = candidates
            .into_iter()
            .next()
            .context("graph-backed module produced no layout candidates")?;
        let placed = place_candidates_on_shelves(&[candidate.clone()], &config.placement);
        return assemble_world(&[candidate], &placed, &[]).map_err(Into::into);
    }

    let mut candidates = Vec::new();
    for instance in &module.instances {
        let child = &context[instance.as_str()];
        let mut child_candidates = generate_graph_module_candidates(child, &config.candidate)?;
        let candidate = child_candidates
            .drain(..)
            .next()
            .with_context(|| format!("module instance `{instance}` produced no candidates"))?;
        candidates.push(candidate);
    }

    let placed = place_candidates_on_shelves(&candidates, &config.placement);
    let routed_nets = route_module_variables(module, &candidates, &placed)?;
    assemble_world(&candidates, &placed, &routed_nets).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::LogicGraph;
    use crate::graph::module::{
        GraphModule, GraphModuleContext, GraphModulePort, GraphModulePortTarget,
        GraphModulePortType, GraphModuleVariable,
    };
    use crate::graph::{Graph, GraphNode, GraphNodeKind};
    use crate::nbt::{NBTRoot, ToNBT};
    use crate::sequential::{SequentialPrimitive, SequentialType};
    use crate::transform::place_and_route::global_pnr::candidate::d_latch_child_candidate_config;
    use crate::transform::place_and_route::local_placer::{
        InputPlacementStrategy, LocalPlacerConfig, NotRouteStrategy, PlacementSamplingPolicy,
        TorchPlacementStrategy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;
    use crate::world::position::DimSize;

    fn sequential_local_config() -> LocalPlacerConfig {
        LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            input_candidate_limit: None,
            step_sampling_policy: SamplingPolicy::Random(256),
            placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
            leak_sampling: false,
            route_torch_directly: true,
            materialize_outputs: false,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectAndRedstone,
            max_not_route_step: 4,
            not_route_step_sampling_policy: SamplingPolicy::Random(256),
            max_route_step: 4,
            route_step_sampling_policy: SamplingPolicy::Random(256),
        }
    }

    #[test]
    fn graph_backed_module_generates_world_with_global_pnr_api() -> eyre::Result<()> {
        let mut module: GraphModule = LogicGraph::from_stmt("~a", "not_a")?.graph.into();
        module.name = "not_gate".to_owned();
        let context = GraphModuleContext::default();
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            },
            placement: GlobalPlacementConfig::default(),
        };

        let world = place_and_route_module(&context, &module, &config)?;

        assert!(!world.iter_block().is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "search-heavy sequential global pnr smoke test"]
    fn d_flip_flop_module_generates_world_from_child_layout_candidates() -> eyre::Result<()> {
        let mut context = GraphModuleContext::default();
        context.append(not_clk_module());
        context.append(d_latch_module("master"));
        context.append(d_latch_module("slave"));
        let module = d_flip_flop_module();
        context.append(module.clone());
        let config = GlobalPnrConfig {
            candidate: d_latch_child_candidate_config(sequential_local_config()),
            placement: GlobalPlacementConfig {
                spacing: 4,
                shelf_width: 64,
            },
        };

        let world = place_and_route_module(&context, &module, &config)?;

        assert!(!world.iter_block().is_empty());
        let nbt: NBTRoot = world.to_nbt();
        nbt.save("test/d-flip-flop-global-smoke.nbt");
        Ok(())
    }

    fn d_flip_flop_module() -> GraphModule {
        GraphModule {
            name: "d_flip_flop".to_owned(),
            graph: None,
            instances: vec![
                "not_clk".to_owned(),
                "master".to_owned(),
                "slave".to_owned(),
            ],
            vars: vec![
                GraphModuleVariable {
                    var_type: GraphModulePortType::InputNet,
                    source: ("not_clk".to_owned(), "clk_n".to_owned()),
                    target: ("master".to_owned(), "en".to_owned()),
                },
                GraphModuleVariable {
                    var_type: GraphModulePortType::InputNet,
                    source: ("master".to_owned(), "q".to_owned()),
                    target: ("slave".to_owned(), "d".to_owned()),
                },
            ],
            ports: vec![
                GraphModulePort {
                    name: "d".to_owned(),
                    port_type: GraphModulePortType::InputNet,
                    target: GraphModulePortTarget::Module("master".to_owned(), "d".to_owned()),
                },
                GraphModulePort {
                    name: "clk".to_owned(),
                    port_type: GraphModulePortType::InputNet,
                    target: GraphModulePortTarget::Wire(vec![
                        ("not_clk".to_owned(), "clk".to_owned()),
                        ("slave".to_owned(), "en".to_owned()),
                    ]),
                },
                GraphModulePort {
                    name: "q".to_owned(),
                    port_type: GraphModulePortType::OutputNet,
                    target: GraphModulePortTarget::Module("slave".to_owned(), "q".to_owned()),
                },
            ],
        }
    }

    fn not_clk_module() -> GraphModule {
        let mut graph = Graph {
            nodes: vec![
                GraphNode {
                    id: 0,
                    kind: GraphNodeKind::Input("clk".to_owned()),
                    ..Default::default()
                },
                GraphNode {
                    id: 1,
                    kind: GraphNodeKind::Logic(crate::logic::Logic {
                        logic_type: crate::logic::LogicType::Not,
                    }),
                    inputs: vec![0],
                    ..Default::default()
                },
                GraphNode {
                    id: 2,
                    kind: GraphNodeKind::Output("clk_n".to_owned()),
                    inputs: vec![1],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        graph.build_outputs();
        graph.build_producers();
        graph.build_consumers();
        graph.verify().unwrap();
        let mut module: GraphModule = graph.into();
        module.name = "not_clk".to_owned();
        module
    }

    fn d_latch_module(name: &str) -> GraphModule {
        let mut graph = Graph {
            nodes: vec![
                GraphNode {
                    id: 0,
                    kind: GraphNodeKind::Input("d".to_owned()),
                    ..Default::default()
                },
                GraphNode {
                    id: 1,
                    kind: GraphNodeKind::Input("en".to_owned()),
                    ..Default::default()
                },
                GraphNode {
                    id: 2,
                    kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
                        SequentialType::DLatch,
                        vec!["d".to_owned(), "en".to_owned()],
                        vec!["q".to_owned()],
                    )),
                    inputs: vec![0, 1],
                    ..Default::default()
                },
                GraphNode {
                    id: 3,
                    kind: GraphNodeKind::Output("q".to_owned()),
                    inputs: vec![2],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        graph.build_outputs();
        graph.build_producers();
        graph.build_consumers();
        graph.verify().unwrap();
        let mut module: GraphModule = graph.into();
        module.name = name.to_owned();
        module
    }
}
