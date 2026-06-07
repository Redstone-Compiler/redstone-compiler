pub mod assembly;
pub mod candidate;
pub mod ir;
pub mod placer;
pub mod progress;
pub mod router;

use eyre::ContextCompat;

use crate::graph::module::{GraphModule, GraphModuleContext, GraphModuleDesign};
use crate::graph::GraphNodeKind;
use crate::output::PlacedWorld;
use crate::transform::place_and_route::global_pnr::assembly::assemble_world;
use crate::transform::place_and_route::global_pnr::candidate::{
    generate_graph_module_candidates_with_progress_label, UnitCandidateConfig,
};
use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::{
    place_candidates_on_shelves, placement_candidates, GlobalPlacementConfig, PlacedModule,
};
use crate::transform::place_and_route::global_pnr::progress::GlobalPnrProgress;
use crate::transform::place_and_route::global_pnr::router::{
    collect_module_input_endpoints, collect_module_output_endpoints, route_module_variables,
    GlobalRoutingConfig, RoutedNet,
};
use crate::transform::place_and_route::local_placer::{LocalPlacerConfig, NotRouteStrategy};
use crate::transform::place_and_route::sampling::SamplingPolicy;
use crate::world::block::BlockKind;
use crate::world::World3D;

#[derive(Clone)]
pub struct GlobalPnrConfig {
    pub candidate: UnitCandidateConfig,
    pub placement: GlobalPlacementConfig,
    pub routing: GlobalRoutingConfig,
    pub show_progress: bool,
}

impl Default for GlobalPnrConfig {
    fn default() -> Self {
        Self {
            candidate: UnitCandidateConfig::default(),
            placement: GlobalPlacementConfig::default(),
            routing: GlobalRoutingConfig::default(),
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
    let placement_attempts = placement_candidates(module, &candidates, &config.placement);
    progress.detail(format!(
        "generated {} placement attempt(s)",
        placement_attempts.len()
    ));

    progress.stage(3, 5, "route module ports and variables");
    let (placed, routed_nets) = route_first_successful_placement(
        module,
        &candidates,
        placement_attempts,
        config,
        &progress,
    )?;

    progress.stage(4, 5, "assemble world and collect outputs");
    let inputs = collect_module_input_endpoints(module, &routed_nets);
    let outputs = collect_module_output_endpoints(module, &candidates, &placed);
    let world = assemble_world(&candidates, &placed, &routed_nets)?;
    let placed_world = PlacedWorld {
        world,
        inputs,
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

fn route_first_successful_placement(
    module: &GraphModule,
    candidates: &[LayoutCandidate],
    placement_attempts: Vec<Vec<PlacedModule>>,
    config: &GlobalPnrConfig,
    progress: &GlobalPnrProgress,
) -> eyre::Result<(Vec<PlacedModule>, Vec<RoutedNet>)> {
    let mut last_error = None;
    let total_attempts = placement_attempts.len();

    for (attempt_index, placed) in placement_attempts.into_iter().enumerate() {
        progress.item(attempt_index + 1, total_attempts, "route placement attempt");
        match route_module_variables(module, candidates, &placed, &config.routing, progress) {
            Ok(routed_nets) => {
                progress.detail(format!(
                    "selected placement attempt {} with {} route(s)",
                    attempt_index + 1,
                    routed_nets.len()
                ));
                return Ok((placed, routed_nets));
            }
            Err(error) => {
                progress.detail(format!(
                    "placement attempt {} failed: {error}",
                    attempt_index + 1
                ));
                last_error = Some(error);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| eyre::eyre!("no global placement attempts generated")))
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
    let inputs = candidate
        .ports
        .iter()
        .filter(|port| {
            port.direction
                == crate::transform::place_and_route::global_pnr::ir::PhysicalPortDirection::Input
        })
        .map(|port| crate::output::OutputEndpoint::new(port.name.clone(), port.position))
        .collect();

    progress.stage(3, 4, "assemble leaf world");
    let world = assemble_world(&[candidate], &placed, &[])?;

    progress.stage(4, 4, "complete");
    Ok(PlacedWorld {
        world,
        inputs,
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
        let child_config = candidate_config_for_child(child, &config.candidate);
        let mut child_candidates = generate_graph_module_candidates_with_progress_label(
            child,
            &child_config,
            config.show_progress.then_some(instance.as_str()),
        )?;
        progress.detail(format!(
            "`{instance}` produced {} candidate(s)",
            child_candidates.len()
        ));
        let candidate = if graph_module_input_port_count(child) > 1 {
            child_candidates
                .drain(..)
                .min_by_key(child_candidate_selection_cost)
        } else {
            child_candidates.drain(..).next()
        }
        .with_context(|| format!("module instance `{instance}` produced no candidates"))?;
        candidates.push(candidate);
    }
    Ok(candidates)
}

fn candidate_config_for_child(
    child: &GraphModule,
    base_config: &UnitCandidateConfig,
) -> UnitCandidateConfig {
    let mut config = base_config.clone();
    if graph_module_input_port_count(child) > 1 && graph_module_is_combinational(child) {
        config.local_config = multi_input_combinational_local_config(config.local_config);
    }
    config
}

fn graph_module_is_combinational(module: &GraphModule) -> bool {
    module.graph.as_ref().is_some_and(|graph| {
        graph
            .nodes
            .iter()
            .all(|node| !matches!(node.kind, GraphNodeKind::Sequential(_)))
    })
}

fn multi_input_combinational_local_config(mut config: LocalPlacerConfig) -> LocalPlacerConfig {
    config.leak_sampling = false;
    config.not_route_strategy = NotRouteStrategy::DirectAndRedstone;
    config.max_not_route_step = config.max_not_route_step.max(4);
    config.not_route_step_sampling_policy = SamplingPolicy::Random(512);
    config.max_route_step = config.max_route_step.max(4);
    config.route_step_sampling_policy = SamplingPolicy::Random(512);
    config
}

fn graph_module_input_port_count(module: &GraphModule) -> usize {
    module
        .ports
        .iter()
        .filter(|port| port.port_type.is_input())
        .count()
}

fn child_candidate_selection_cost(candidate: &LayoutCandidate) -> (usize, usize, usize) {
    (
        input_port_initial_power_count(candidate),
        candidate.cost.bbox_volume,
        candidate.cost.block_count,
    )
}

fn input_port_initial_power_count(candidate: &LayoutCandidate) -> usize {
    candidate
        .ports
        .iter()
        .filter(|port| {
            matches!(
                port.direction,
                crate::transform::place_and_route::global_pnr::ir::PhysicalPortDirection::Input
            )
        })
        .filter(|port| {
            candidate.world.size.bound_on(port.position)
                && block_is_powered(candidate.world[port.position].kind)
        })
        .count()
}

fn block_is_powered(kind: BlockKind) -> bool {
    match kind {
        BlockKind::Cobble {
            on_count,
            on_base_count,
        } => on_count > 0 || on_base_count > 0,
        BlockKind::Switch { is_on } => is_on,
        BlockKind::Redstone { strength, .. } => strength > 0,
        BlockKind::Torch { is_on } => is_on,
        BlockKind::Repeater { is_on, .. } => is_on,
        BlockKind::RedstoneBlock => true,
        BlockKind::Piston { is_on, .. } => is_on,
        BlockKind::Air => false,
    }
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
                spacing: 2,
                shelf_width: 24,
                ..Default::default()
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

    #[test]
    #[ignore = "search-heavy sequential global pnr smoke test"]
    fn counter_module_generates_world_from_child_layout_candidates() -> eyre::Result<()> {
        init_tracing_from_env();
        let design = lower_design_modules(&parse_modules(
            r#"
            module counter(clk, q);
              input clk;
              output reg [1:0] q;
              always @(posedge clk) begin
                q <= q + 1;
              end
            endmodule
            "#,
        )?)?;
        let config = GlobalPnrConfig {
            candidate: UnitCandidateConfig {
                max_candidates: 4,
                ..d_latch_child_candidate_config(sequential_local_config())
            },
            placement: GlobalPlacementConfig {
                spacing: 4,
                shelf_width: 64,
                max_attempts: 64,
                ..Default::default()
            },
            ..Default::default()
        };

        let placed = place_and_route_design_with_outputs(&design, &config)?;

        assert!(!placed.world.iter_block().is_empty());
        let mut output_names = placed
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>();
        output_names.sort();
        assert_eq!(output_names, vec!["q_0", "q_1"]);
        let nbt: NBTRoot = placed.world.to_nbt();
        nbt.save("test/counter-global-smoke.nbt");
        placed
            .metadata()
            .save("test/counter-global-smoke.outputs.json")?;
        assert_two_bit_counter_behavior(&placed)?;
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
        let data = input_endpoint_position(placed, "d")?;
        let clock = input_endpoint_position(placed, "clk")?;
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

    fn assert_two_bit_counter_behavior(placed: &PlacedWorld) -> eyre::Result<()> {
        let clock = input_endpoint_position(placed, "clk")?;
        let output_q0 = placed
            .outputs
            .iter()
            .find(|output| output.name == "q_0")
            .context("missing q_0 output")?
            .position();
        let output_q1 = placed
            .outputs
            .iter()
            .find(|output| output.name == "q_1")
            .context("missing q_1 output")?
            .position();
        let world = World::from(&placed.world);
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

        let initial = counter_output_value(sim.world(), output_q0, output_q1);
        sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
        let first_rise = counter_output_value(sim.world(), output_q0, output_q1);
        eyre::ensure!(
            first_rise == 1,
            "counter output should become 1 on first rising edge: initial={initial}, first_rise={first_rise}"
        );

        sim.change_state_with_limits(vec![(clock, false)], 256, 50_000)?;
        let first_fall = counter_output_value(sim.world(), output_q0, output_q1);
        eyre::ensure!(
            first_fall == first_rise,
            "counter output should hold on falling edge: first_rise={first_rise}, first_fall={first_fall}"
        );

        sim.change_state_with_limits(vec![(clock, true)], 256, 50_000)?;
        let second_rise = counter_output_value(sim.world(), output_q0, output_q1);
        eyre::ensure!(
            second_rise == 2,
            "counter output should become 2 on second rising edge: initial={initial}, first_rise={first_rise}, first_fall={first_fall}, second_rise={second_rise}"
        );
        Ok(())
    }

    fn counter_output_value(world: &crate::world::World3D, q0: Position, q1: Position) -> usize {
        usize::from(block_power(world, q0)) | (usize::from(block_power(world, q1)) << 1)
    }

    fn input_endpoint_position(placed: &PlacedWorld, name: &str) -> eyre::Result<Position> {
        placed
            .inputs
            .iter()
            .find(|input| input.name == name)
            .map(|input| input.position())
            .with_context(|| format!("missing input endpoint `{name}`"))
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
