pub mod assembly;
pub mod candidate;
pub mod ir;
pub mod placer;
pub mod router;

use eyre::ContextCompat;

use crate::graph::module::{GraphModule, GraphModuleContext};
use crate::output::PlacedWorld;
use crate::transform::place_and_route::global_pnr::assembly::assemble_world;
use crate::transform::place_and_route::global_pnr::candidate::{
    generate_graph_module_candidates, UnitCandidateConfig,
};
use crate::transform::place_and_route::global_pnr::placer::{
    place_candidates_on_shelves, GlobalPlacementConfig,
};
use crate::transform::place_and_route::global_pnr::router::{
    collect_module_output_endpoints, route_module_variables,
};
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
    Ok(place_and_route_module_with_outputs(context, module, config)?.world)
}

pub fn place_and_route_module_with_outputs(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
) -> eyre::Result<PlacedWorld> {
    if module.graph.is_some() {
        let candidates = generate_graph_module_candidates(module, &config.candidate)?;
        let candidate = candidates
            .into_iter()
            .next()
            .context("graph-backed module produced no layout candidates")?;
        let placed = place_candidates_on_shelves(&[candidate.clone()], &config.placement);
        let world = assemble_world(&[candidate], &placed, &[])?;
        return Ok(PlacedWorld {
            world,
            inputs: Vec::new(),
            outputs: Vec::new(),
        });
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

    let placed = align_variable_targets_on_y(
        module,
        &candidates,
        place_candidates_on_shelves(&candidates, &config.placement),
    );
    if std::env::var_os("PRINT_GLOBAL_PNR").is_some() {
        for (candidate, placed_module) in candidates.iter().zip(&placed) {
            eprintln!(
                "candidate {} origin {:?} bbox {:?}",
                candidate.module_name, placed_module.origin, candidate.bbox
            );
            for port in &candidate.ports {
                eprintln!(
                    "  port {} {:?} {:?} {:?} route={:?} isolate={}",
                    port.name,
                    port.direction,
                    port.position,
                    candidate.world[port.position].kind,
                    port.route_position,
                    port.isolate_input
                );
            }
        }
    }
    let routed_nets = route_module_variables(module, &candidates, &placed)?;
    let outputs = collect_module_output_endpoints(module, &candidates, &placed);
    let world = assemble_world(&candidates, &placed, &routed_nets)?;
    Ok(PlacedWorld {
        world,
        inputs: Vec::new(),
        outputs,
    })
}

fn align_variable_targets_on_y(
    module: &GraphModule,
    candidates: &[crate::transform::place_and_route::global_pnr::ir::LayoutCandidate],
    mut placed: Vec<crate::transform::place_and_route::global_pnr::placer::PlacedModule>,
) -> Vec<crate::transform::place_and_route::global_pnr::placer::PlacedModule> {
    for var in &module.vars {
        let Some(source_index) = placed
            .iter()
            .position(|placed_module| placed_module.module_name == var.source.0)
        else {
            continue;
        };
        let Some(target_index) = placed
            .iter()
            .position(|placed_module| placed_module.module_name == var.target.0)
        else {
            continue;
        };
        let Some(source_port) = candidates[source_index]
            .ports
            .iter()
            .find(|port| port.name == var.source.1)
        else {
            continue;
        };
        let Some(target_port) = candidates[target_index]
            .ports
            .iter()
            .find(|port| port.name == var.target.1)
        else {
            continue;
        };

        let source_y = placed[source_index].origin.1 + source_port.position.1
            - candidates[source_index].bbox.min.1;
        let Some(target_origin_y) = source_y
            .checked_add(candidates[target_index].bbox.min.1)
            .and_then(|y| y.checked_sub(target_port.position.1))
        else {
            continue;
        };
        placed[target_index].origin.1 = target_origin_y;

        let source_z = placed[source_index].origin.2 + source_port.position.2
            - candidates[source_index].bbox.min.2;
        let Some(target_origin_z) = source_z
            .checked_add(candidates[target_index].bbox.min.2)
            .and_then(|z| z.checked_sub(target_port.position.2))
        else {
            continue;
        };
        placed[target_index].origin.2 = target_origin_z;
    }

    placed
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
    use crate::transform::place_and_route::utils::world_to_logic_with_outputs;
    use crate::world::block::BlockKind;
    use crate::world::position::{DimSize, Position};
    use crate::world::simulator::Simulator;
    use crate::world::World;

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
    fn module_outputs_point_to_placed_child_ports_without_extra_routes() -> eyre::Result<()> {
        let mut context = GraphModuleContext::default();
        context.append(not_clk_module());
        let module = GraphModule {
            name: "top".to_owned(),
            instances: vec!["not_clk".to_owned()],
            ports: vec![GraphModulePort {
                name: "q".to_owned(),
                port_type: GraphModulePortType::OutputNet,
                target: GraphModulePortTarget::Module("not_clk".to_owned(), "clk_n".to_owned()),
            }],
            ..Default::default()
        };
        context.append(module.clone());
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            },
            placement: GlobalPlacementConfig::default(),
        };

        let placed = place_and_route_module_with_outputs(&context, &module, &config)?;

        assert_eq!(placed.outputs.len(), 1);
        assert_eq!(placed.outputs[0].name, "q");
        let output_position = placed.outputs[0].position();
        assert_ne!(
            placed.world[output_position].kind,
            crate::world::block::BlockKind::Air
        );
        Ok(())
    }

    #[test]
    #[ignore = "search-heavy sequential global pnr smoke test"]
    fn d_flip_flop_module_generates_world_from_child_layout_candidates() -> eyre::Result<()> {
        init_tracing_from_env();
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

        let placed = place_and_route_module_with_outputs(&context, &module, &config)?;

        assert!(!placed.world.iter_block().is_empty());
        assert_eq!(placed.outputs.len(), 1);
        assert_eq!(placed.outputs[0].name, "q");
        let nbt: NBTRoot = placed.world.to_nbt();
        nbt.save("test/d-flip-flop-global-smoke.nbt");
        placed
            .metadata()
            .save("test/d-flip-flop-global-smoke.outputs.json")?;
        let logic = world_to_logic_with_outputs(&nbt.to_world(), &placed.metadata())?;
        assert!(logic
            .nodes
            .iter()
            .any(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == "q")));
        assert_positive_edge_dff_behavior(&placed)?;
        Ok(())
    }

    fn init_tracing_from_env() {
        let Some(level) = rust_log_level() else {
            return;
        };
        let _ = tracing_subscriber::fmt().with_max_level(level).try_init();
    }

    fn rust_log_level() -> Option<tracing::Level> {
        let value = std::env::var("RUST_LOG").ok()?;
        value
            .split(',')
            .filter_map(|part| part.rsplit('=').next())
            .find_map(|level| match level.trim().to_ascii_lowercase().as_str() {
                "trace" => Some(tracing::Level::TRACE),
                "debug" => Some(tracing::Level::DEBUG),
                "info" => Some(tracing::Level::INFO),
                "warn" | "warning" => Some(tracing::Level::WARN),
                "error" => Some(tracing::Level::ERROR),
                _ => None,
            })
    }

    fn assert_positive_edge_dff_behavior(placed: &PlacedWorld) -> eyre::Result<()> {
        let mut switches = placed
            .world
            .iter_block()
            .into_iter()
            .filter_map(|(position, block)| block.kind.is_switch().then_some(position))
            .collect::<Vec<_>>();
        switches.sort();
        eyre::ensure!(switches.len() >= 2, "expected d and clk switches");
        let data = switches[0];
        let clock = switches[1];
        let output = placed.outputs[0].position();
        let world = World::from(&placed.world);
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        assert!(!block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(data, true)], 256, 50_000)?;
        assert!(!block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
        assert!(block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(clock, false)], 256, 50_000)?;
        assert!(block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(data, false)], 256, 50_000)?;
        assert!(block_power(sim.world(), output));
        sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
        assert!(!block_power(sim.world(), output));
        Ok(())
    }

    fn block_power(world: &crate::world::World3D, position: Position) -> bool {
        match world[position].kind {
            BlockKind::Redstone {
                strength, on_count, ..
            } => strength > 0 || on_count > 0,
            BlockKind::Torch { is_on }
            | BlockKind::Repeater { is_on, .. }
            | BlockKind::Switch { is_on } => is_on,
            BlockKind::Cobble {
                on_count,
                on_base_count,
            } => on_count > 0 || on_base_count > 0,
            BlockKind::RedstoneBlock => true,
            BlockKind::Air | BlockKind::Piston { .. } => false,
        }
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
