pub mod assembly;
pub mod candidate;
pub mod ir;
pub mod placer;
pub mod progress;
pub mod router;

use eyre::ContextCompat;

use crate::graph::module::{GraphModule, GraphModuleContext, GraphModuleDesign};
use crate::output::PlacedWorld;
use crate::transform::place_and_route::global_pnr::assembly::assemble_world;
use crate::transform::place_and_route::global_pnr::candidate::{
    generate_graph_module_candidates_with_progress_label, UnitCandidateConfig,
};
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::{
    place_candidates_on_shelves, GlobalPlacementConfig,
};
use crate::transform::place_and_route::global_pnr::progress::GlobalPnrProgress;
use crate::transform::place_and_route::global_pnr::router::{
    collect_module_output_endpoints, route_module_variables,
};
use crate::world::World3D;

#[derive(Clone)]
pub struct GlobalPnrConfig {
    pub candidate: UnitCandidateConfig,
    pub placement: GlobalPlacementConfig,
    pub show_progress: bool,
}

impl Default for GlobalPnrConfig {
    fn default() -> Self {
        Self {
            candidate: UnitCandidateConfig::default(),
            placement: GlobalPlacementConfig::default(),
            show_progress: true,
        }
    }
}

pub fn place_and_route_module(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    Ok(place_and_route_module_with_outputs(context, module, config)?.world)
}

pub fn place_and_route_design(
    design: &GraphModuleDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<World3D> {
    Ok(place_and_route_design_with_outputs(design, config)?.world)
}

pub fn place_and_route_design_with_outputs(
    design: &GraphModuleDesign,
    config: &GlobalPnrConfig,
) -> eyre::Result<PlacedWorld> {
    place_and_route_module_with_outputs(&design.context, design.top_module(), config)
}

pub fn place_and_route_module_with_outputs(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
) -> eyre::Result<PlacedWorld> {
    let progress = GlobalPnrProgress::new(config.show_progress, module.name.clone());
    if module.graph.is_some() {
        return place_graph_backed_module(module, config, &progress);
    }

    progress.stage(1, 5, "generate child layout candidates");
    let candidates = generate_child_candidates(context, module, config, &progress)?;

    progress.stage(2, 5, "place child candidates");
    let placed = place_candidates_on_shelves(&candidates, &config.placement);

    progress.stage(3, 5, "route module ports and variables");
    let routed_nets = route_module_variables(module, &candidates, &placed, &progress)?;

    progress.stage(4, 5, "assemble world and collect outputs");
    let outputs = collect_module_output_endpoints(module, &candidates, &placed);
    let world = assemble_world(&candidates, &placed, &routed_nets)?;
    let placed_world = PlacedWorld {
        world,
        inputs: Vec::new(),
        outputs,
    };

    progress.stage(5, 5, "complete");
    progress.detail(format!(
        "outputs={} routes={}",
        placed_world.outputs.len(),
        routed_nets.len()
    ));

    Ok(placed_world)
}

fn place_graph_backed_module(
    module: &GraphModule,
    config: &GlobalPnrConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<PlacedWorld> {
    progress.stage(1, 4, "generate leaf layout candidates");
    let candidates = generate_graph_module_candidates_with_progress_label(
        module,
        &config.candidate,
        config.show_progress.then_some(module.name.as_str()),
    )?;
    let candidate = candidates
        .into_iter()
        .next()
        .context("graph-backed module produced no layout candidates")?;
    progress.detail(format!(
        "selected candidate for `{}`",
        candidate.module_name
    ));

    progress.stage(2, 4, "place leaf candidate");
    let placed = place_candidates_on_shelves(&[candidate.clone()], &config.placement);

    progress.stage(3, 4, "assemble leaf world");
    let world = assemble_world(&[candidate], &placed, &[])?;

    progress.stage(4, 4, "complete");
    Ok(PlacedWorld {
        world,
        inputs: Vec::new(),
        outputs: Vec::new(),
    })
}

// 하위 모듈마다 local placer를 실행해서 global PnR이 배치할 layout 후보를 하나씩 뽑는다.
fn generate_child_candidates(
    context: &GraphModuleContext,
    module: &GraphModule,
    config: &GlobalPnrConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<Vec<LayoutCandidate>> {
    let mut candidates = Vec::new();
    for (index, instance) in module.instances.iter().enumerate() {
        progress.item(
            index + 1,
            module.instances.len(),
            format!("generate `{instance}` candidate"),
        );
        let child = &context[instance.as_str()];
        let mut child_candidates = generate_graph_module_candidates_with_progress_label(
            child,
            &config.candidate,
            config.show_progress.then_some(instance.as_str()),
        )?;
        progress.detail(format!(
            "`{instance}` produced {} candidate(s)",
            child_candidates.len()
        ));
        let candidate = child_candidates
            .drain(..)
            .next()
            .with_context(|| format!("module instance `{instance}` produced no candidates"))?;
        candidates.push(candidate);
    }
    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::LogicGraph;
    use crate::graph::module::{
        GraphModule, GraphModuleContext, GraphModuleDesign, GraphModulePort, GraphModulePortTarget,
        GraphModulePortType,
    };
    use crate::graph::GraphNodeKind;
    use crate::nbt::{NBTRoot, ToNBT};
    use crate::transform::place_and_route::global_pnr::candidate::d_latch_child_candidate_config;
    use crate::transform::place_and_route::local_placer::{
        InputPlacementStrategy, LocalPlacerConfig, NotRouteStrategy, PlacementSamplingPolicy,
        TorchPlacementStrategy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;
    use crate::transform::place_and_route::utils::world_to_logic_with_outputs;
    use crate::verilog::design::lower_design_modules;
    use crate::verilog::parser::parse_modules;
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
        let design = GraphModuleDesign::with_top_module(context, module);
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            },
            placement: GlobalPlacementConfig::default(),
            ..Default::default()
        };

        let world = place_and_route_design(&design, &config)?;

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
        let design = GraphModuleDesign::with_top_module(context, module);
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                dim: DimSize(8, 8, 4),
                max_candidates: 1,
                ..Default::default()
            },
            placement: GlobalPlacementConfig::default(),
            ..Default::default()
        };

        let placed = place_and_route_design_with_outputs(&design, &config)?;

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
        let design = lower_design_modules(&parse_modules(
            r#"
            module not_clk(clk, clk_n);
              input clk;
              output clk_n;
              assign clk_n = ~clk;
            endmodule

            module d_latch(d, en, q);
              input d, en;
              output reg q;
              always @(*) begin
                if (en) begin
                  q <= d;
                end
              end
            endmodule

            module d_flip_flop(d, clk, q);
              input d, clk;
              output q;
              wire clk_n, master_q;
              not_clk inv(.clk(clk), .clk_n(clk_n));
              d_latch master(.d(d), .en(clk_n), .q(master_q));
              d_latch slave(.d(master_q), .en(clk), .q(q));
            endmodule
            "#,
        )?)?;
        let config = GlobalPnrConfig {
            candidate: d_latch_child_candidate_config(sequential_local_config()),
            placement: GlobalPlacementConfig {
                spacing: 4,
                shelf_width: 64,
            },
            ..Default::default()
        };

        let placed = place_and_route_design_with_outputs(&design, &config)?;

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

    fn not_clk_module() -> GraphModule {
        let mut module: GraphModule = LogicGraph::from_stmt("~clk", "clk_n").unwrap().graph.into();
        module.name = "not_clk".to_owned();
        module
    }
}
