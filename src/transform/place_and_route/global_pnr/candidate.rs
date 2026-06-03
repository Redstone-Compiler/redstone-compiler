use eyre::ContextCompat;

use crate::graph::logic::LogicGraph;
use crate::graph::module::{GraphModule, GraphModulePortTarget, GraphModulePortType};
use crate::transform::place_and_route::detailed_router;
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPort, PhysicalPortDirection,
};
use crate::transform::place_and_route::local_placer::{
    LocalPlacer, LocalPlacerConfig, LocalPlacerInputConstraints,
};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::Block;
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

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
    generate_graph_module_candidates_with_progress_label(module, config, None)
}

pub fn generate_graph_module_candidates_with_progress_label(
    module: &GraphModule,
    config: &UnitCandidateConfig,
    progress_label: Option<&str>,
) -> eyre::Result<Vec<LayoutCandidate>> {
    let graph = module
        .graph
        .clone()
        .context("only graph-backed GraphModule can generate unit layout candidates")?;
    let graph = LogicGraph { graph }.prepare_place()?;
    let placer = LocalPlacer::new(graph, config.local_config)?;

    let placed = placer.generate_with_outputs_and_input_constraints_progress(
        config.dim,
        None,
        &config.input_constraints,
        progress_label,
    );

    placed
        .into_iter()
        .take(config.max_candidates)
        .map(|placed| {
            let (world, ports) = switchless_candidate_layout(
                module,
                &config.input_constraints,
                placed.world,
                &placed.inputs,
                &placed.outputs,
            );
            LayoutCandidate::from_world(module.name.clone(), world, ports)
        })
        .collect()
}

// Local placer媛 留뚮뱺 ?낅┰ ?ㅽ뻾??switch/output layout??global PnR??child layout?쇰줈 諛붽씔??
// ?낅젰 switch???쒓굅?댁꽌 ?몃? route媛 臾쇰┫ port濡??몄텧?섍퀬, 異쒕젰 ?꾩튂??module port metadata濡?蹂댁〈?쒕떎.
// TODO(high-level): make LocalPlacer produce either standalone layouts with switches
// or child-module layouts with PhysicalPort metadata, instead of rewriting switches here.
fn switchless_candidate_layout(
    module: &GraphModule,
    input_constraints: &LocalPlacerInputConstraints,
    mut world: World3D,
    inputs: &[crate::output::OutputEndpoint],
    outputs: &[crate::output::OutputEndpoint],
) -> (World3D, Vec<PhysicalPort>) {
    let mut ports = Vec::new();
    // Sequential child layout? ?대? feedback/state signal???몃? route? 吏곸젒 ?욎씠硫?    // back-power??latch state ?ㅼ뿼???앷만 ???덉쑝誘濡?port ?곌껐??蹂댁닔?곸쑝濡?寃⑸━?쒕떎.
    let needs_output_isolation = module_contains_sequential(module);
    let needs_input_isolation = module_contains_sequential(module);
    for port in &module.ports {
        match (&port.port_type, &port.target) {
            (GraphModulePortType::InputNet, GraphModulePortTarget::Node(input_name)) => {
                let position = inputs
                    .iter()
                    .find(|input| input.name == *input_name)
                    .map(|input| input.position())
                    .or_else(|| {
                        input_constraints
                            .positions_for_input_name(input_name)
                            .and_then(|positions| positions.into_iter().next())
                    });
                if let Some(input_position) = position {
                    let Some(position) = expose_switchless_input_port(&mut world, input_position)
                    else {
                        continue;
                    };
                    ports.push(PhysicalPort {
                        name: port.name.clone(),
                        direction: PhysicalPortDirection::Input,
                        position,
                        route_position: None,
                        isolate_input: needs_input_isolation && world[position].kind.is_redstone(),
                    });
                }
            }
            (GraphModulePortType::OutputNet, GraphModulePortTarget::Node(output_name)) => {
                if let Some(output) = outputs.iter().find(|output| output.name == *output_name) {
                    let position = output.position();
                    let route_position = (!needs_output_isolation)
                        .then(|| expose_routeable_output_port(&world, position));
                    ports.push(PhysicalPort {
                        name: port.name.clone(),
                        direction: PhysicalPortDirection::Output,
                        position,
                        route_position,
                        isolate_input: false,
                    });
                }
            }
            _ => {}
        }
    }
    for input in inputs {
        let _ = expose_switchless_input_port(&mut world, input.position());
    }
    remove_local_input_switches(&mut world);
    ports.sort_by(|a, b| a.name.cmp(&b.name));
    world.initialize_redstone_states();
    (world, ports)
}

fn module_contains_sequential(module: &GraphModule) -> bool {
    module.graph.as_ref().is_some_and(|graph| {
        graph
            .nodes
            .iter()
            .any(|node| matches!(node.kind, crate::graph::GraphNodeKind::Sequential(_)))
    })
}

fn remove_local_input_switches(world: &mut World3D) {
    for (position, block) in world.iter_block() {
        if block.kind.is_switch() {
            world[position] = Block::default();
        }
    }
}

// Torch/switch/repeater 異쒕젰 ?먯껜蹂대떎, 洹?異쒕젰???ㅼ젣濡??꾩썝??怨듦툒?섎뒗 redstone tap???덉쑝硫?// 洹?tap???몃? route ?쒖옉?먯쑝濡??몄텧?쒕떎. 議고빀?뚮줈 異쒕젰? ?대젃寃?湲곗〈 異쒕젰留앹뿉 route瑜?遺숈씤??
fn expose_routeable_output_port(world: &World3D, output_position: Position) -> Position {
    if !world.size.bound_on(output_position)
        || (!world[output_position].kind.is_torch()
            && !world[output_position].kind.is_switch()
            && !world[output_position].kind.is_repeater())
    {
        return output_position;
    }

    world
        .iter_block()
        .into_iter()
        .filter(|(position, block)| {
            block.kind.is_redstone()
                && detailed_router::target_powers_position(world, output_position, *position)
        })
        .map(|(position, _)| position)
        .min_by_key(|position| {
            (
                output_position.manhattan_distance(position),
                position.0,
                position.1,
                position.2,
            )
        })
        .unwrap_or(output_position)
}

// Local placer ?낅젰? 蹂댄넻 switch濡?留뚮뱾?댁졇 ?덉쑝誘濡?global PnR child layout?먯꽌??switch瑜??쒓굅?쒕떎.
// switch媛 cobble??耳쒕뒗 援ъ“硫?cobble??port濡? redstone fanout??耳쒕뒗 援ъ“硫?switch ?먮━瑜?redstone port濡?諛붽씔??
// TODO(low-level): replace this inference with explicit input-port placement metadata
// from LocalPlacer, so this code does not need to guess from switch wiring.
fn expose_switchless_input_port(world: &mut World3D, input_position: Position) -> Option<Position> {
    if !world.size.bound_on(input_position) {
        return None;
    }
    if !world[input_position].kind.is_switch() {
        return Some(input_position);
    }

    let switch_target = input_position.walk(world[input_position].direction);
    if let Some(target) = switch_target
        .filter(|position| world.size.bound_on(*position) && world[*position].kind.is_cobble())
    {
        world[input_position] = Block::default();
        return Some(target);
    }

    if switch_powers_redstone(world, input_position) {
        ensure_redstone_support(world, input_position)?;
        world[input_position] = PlacedNode::new_redstone(input_position).block;
        return Some(input_position);
    }

    let port_position = expose_routeable_output_port(world, input_position);
    (port_position != input_position).then(|| {
        world[input_position] = Block::default();
        port_position
    })
}

fn switch_powers_redstone(world: &World3D, input_position: Position) -> bool {
    world.iter_block().into_iter().any(|(position, block)| {
        block.kind.is_redstone()
            && detailed_router::target_powers_position(world, input_position, position)
    })
}

fn ensure_redstone_support(world: &mut World3D, position: Position) -> Option<()> {
    let support_position = position.down()?;
    if !world.size.bound_on(support_position) {
        return None;
    }
    if world[support_position].kind.is_cobble() {
        return Some(());
    }
    if !world[support_position].kind.is_air() {
        return None;
    }
    world[support_position] = PlacedNode::new_cobble(support_position).block;
    Some(())
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
