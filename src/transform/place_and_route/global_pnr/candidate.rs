use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};

use eyre::ContextCompat;

use crate::graph::logic::LogicGraph;
use crate::graph::module::{GraphModule, GraphModulePortTarget, GraphModulePortType};
use crate::graph::GraphNodeKind;
use crate::output::OutputEndpoint;
use crate::transform::place_and_route::detailed_router;
use crate::transform::place_and_route::global_pnr::ir::{
    LayoutCandidate, PhysicalPort, PhysicalPortDirection, PhysicalPortRouteAdapter,
};
use crate::transform::place_and_route::local_placer::{
    LocalPlacer, LocalPlacerConfig, LocalPlacerInputConstraints,
};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

#[derive(Clone)]
pub struct UnitCandidateConfig {
    pub dim: DimSize,
    pub local_config: LocalPlacerConfig,
    pub input_constraints: LocalPlacerInputConstraints,
    pub max_candidates: usize,
}

const CHILD_LAYOUT_PORT_PADDING: usize = 2;

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
    let placer = LocalPlacer::new(graph.clone(), config.local_config)?;

    let placed = placer.generate_with_outputs_and_input_constraints_progress(
        config.dim,
        None,
        &config.input_constraints,
        progress_label,
    );

    let validate_truth_table = graph_is_combinational(module);
    let mut candidates = Vec::new();
    for placed in placed {
        if candidates.len() >= config.max_candidates {
            break;
        }
        if validate_truth_table && !candidate_matches_truth_table(&graph, &placed)? {
            continue;
        }
        let (world, inputs, outputs) = padded_child_layout(
            placed.world,
            &placed.inputs,
            &placed.outputs,
            CHILD_LAYOUT_PORT_PADDING,
        );
        let (world, ports) = switchless_candidate_layout(
            module,
            &config.input_constraints,
            world,
            &inputs,
            &outputs,
        );
        if !candidate_ports_cover_module_ports(module, &ports) {
            continue;
        }
        candidates.push(LayoutCandidate::from_world(
            module.name.clone(),
            world,
            ports,
        )?);
    }
    Ok(candidates)
}

fn candidate_ports_cover_module_ports(module: &GraphModule, ports: &[PhysicalPort]) -> bool {
    module
        .ports
        .iter()
        .all(|module_port| ports.iter().any(|port| port.name == module_port.name))
}

fn padded_child_layout(
    world: World3D,
    inputs: &[OutputEndpoint],
    outputs: &[OutputEndpoint],
    padding: usize,
) -> (World3D, Vec<OutputEndpoint>, Vec<OutputEndpoint>) {
    if padding == 0 {
        return (world, inputs.to_vec(), outputs.to_vec());
    }

    let mut padded = World3D::new(DimSize(
        world.size.0 + padding * 2,
        world.size.1 + padding * 2,
        world.size.2,
    ));
    for (position, block) in world.iter_block() {
        padded[shift_position_xy(position, padding)] = block;
    }
    padded.initialize_redstone_states();

    let inputs = inputs
        .iter()
        .map(|input| {
            OutputEndpoint::new(
                input.name.clone(),
                shift_position_xy(input.position(), padding),
            )
        })
        .collect();
    let outputs = outputs
        .iter()
        .map(|output| {
            OutputEndpoint::new(
                output.name.clone(),
                shift_position_xy(output.position(), padding),
            )
        })
        .collect();

    (padded, inputs, outputs)
}

fn shift_position_xy(position: Position, offset: usize) -> Position {
    Position(position.0 + offset, position.1 + offset, position.2)
}

fn graph_is_combinational(module: &GraphModule) -> bool {
    module.graph.as_ref().is_some_and(|graph| {
        graph
            .nodes
            .iter()
            .all(|node| !matches!(node.kind, GraphNodeKind::Sequential(_)))
    })
}

fn candidate_matches_truth_table(
    expected: &LogicGraph,
    placed: &crate::output::PlacedWorld,
) -> eyre::Result<bool> {
    let expected = expected.truth_table()?;
    let inputs = expected
        .input_names
        .iter()
        .map(|name| {
            placed
                .inputs
                .iter()
                .find(|input| input.name == *name)
                .map(|input| input.position())
                .with_context(|| format!("missing input endpoint `{name}`"))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let outputs = expected
        .output_tables
        .keys()
        .map(|name| {
            placed
                .outputs
                .iter()
                .find(|output| output.name == *name)
                .map(|output| (name.as_str(), output.position()))
                .with_context(|| format!("missing output endpoint `{name}`"))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let world = World::from(&placed.world);

    for mask in 0..(1usize << inputs.len()) {
        let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        sim.change_state_with_limits(
            inputs
                .iter()
                .enumerate()
                .map(|(index, position)| (*position, (mask & (1 << index)) != 0))
                .collect(),
            256,
            50_000,
        )?;

        for (output_name, output_position) in &outputs {
            let Some(expected_output) = expected.output_tables.get(*output_name) else {
                return Ok(false);
            };
            if block_is_powered(sim.world()[*output_position].kind) != expected_output[mask] {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

// Local placer媛 留뚮뱺 ?낅┰ ?ㅽ뻾??switch/output layout??global PnR??child layout?쇰줈 諛붽씔??
// ?낅젰 switch???쒓굅?댁꽌 ?몃? route媛 臾쇰┫ port濡??몄텧?섍퀬, 異쒕젰 ?꾩튂??module port metadata濡?蹂댁〈?쒕떎.
// TODO(high-level): make LocalPlacer produce either standalone layouts with switches
// or child-module layouts with PhysicalPort metadata, instead of rewriting switches here.
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
    let contains_sequential = module_contains_sequential(module);
    let use_direct_input_ports = module_input_port_count(module) > 1;
    let preserve_switch_position_inputs =
        module_contains_sequential(module) || use_direct_input_ports;
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
                    let Some(position) = expose_switchless_input_port(
                        &mut world,
                        input_position,
                        preserve_switch_position_inputs,
                        use_direct_input_ports,
                    ) else {
                        continue;
                    };
                    ports.push(PhysicalPort {
                        name: port.name.clone(),
                        direction: PhysicalPortDirection::Input,
                        position,
                        route_position: None,
                        isolated_route_position: None,
                        isolated_route_blocks: Vec::new(),
                        isolated_route_options: Vec::new(),
                        isolate_input: needs_input_isolation || world[position].kind.is_redstone(),
                        isolate_output: false,
                        module_contains_sequential: contains_sequential,
                    });
                }
            }
            (GraphModulePortType::OutputNet, GraphModulePortTarget::Node(output_name)) => {
                if let Some(output) = outputs.iter().find(|output| output.name == *output_name) {
                    let position = output.position();
                    let route_position = Some(expose_routeable_output_port(&world, position));
                    let isolated_route_options = if needs_output_isolation {
                        isolated_output_port_adapters(&world, position)
                    } else {
                        Vec::new()
                    };
                    let (isolated_route_position, isolated_route_blocks) = isolated_route_options
                        .first()
                        .map(|adapter| (Some(adapter.position), adapter.blocks.clone()))
                        .unwrap_or((None, Vec::new()));
                    ports.push(PhysicalPort {
                        name: port.name.clone(),
                        direction: PhysicalPortDirection::Output,
                        position,
                        route_position,
                        isolated_route_position,
                        isolated_route_blocks,
                        isolated_route_options,
                        isolate_input: false,
                        isolate_output: needs_output_isolation,
                        module_contains_sequential: contains_sequential,
                    });
                }
            }
            _ => {}
        }
    }
    for input in inputs {
        let _ = expose_switchless_input_port(
            &mut world,
            input.position(),
            preserve_switch_position_inputs,
            use_direct_input_ports,
        );
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

fn module_input_port_count(module: &GraphModule) -> usize {
    module
        .ports
        .iter()
        .filter(|port| port.port_type.is_input())
        .count()
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
fn expose_switchless_input_port(
    world: &mut World3D,
    input_position: Position,
    preserve_switch_position_input: bool,
    use_direct_input_port: bool,
) -> Option<Position> {
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

    if preserve_switch_position_input && switch_powers_redstone(world, input_position) {
        ensure_redstone_support(world, input_position)?;
        world[input_position] = PlacedNode::new_redstone(input_position).block;
        return Some(input_position);
    }

    if let Some(port_position) =
        switch_powered_redstone_port(world, input_position, use_direct_input_port)
    {
        world[input_position] = Block::default();
        return Some(port_position);
    }

    let port_position = expose_routeable_output_port(world, input_position);
    (port_position != input_position).then(|| {
        world[input_position] = Block::default();
        port_position
    })
}

fn isolated_output_port_adapters(
    world: &World3D,
    output_position: Position,
) -> Vec<PhysicalPortRouteAdapter> {
    let taps = routeable_output_redstone_taps(world, output_position);
    let mut candidates = Vec::new();
    for tap in taps {
        for repeater_position in tap.cardinal() {
            let direction = tap.diff(repeater_position).inverse();
            if let Some(candidate) =
                isolated_output_adapter_world(world, tap, repeater_position, direction)
            {
                candidates.push(candidate);
            }
        }
    }

    candidates
        .into_iter()
        .map(
            |(adapter_world, repeater_position)| PhysicalPortRouteAdapter {
                position: repeater_position,
                blocks: added_blocks(world, &adapter_world),
            },
        )
        .collect()
}

fn added_blocks(before: &World3D, after: &World3D) -> Vec<(Position, Block)> {
    after
        .iter_block()
        .into_iter()
        .filter(|(position, block)| !block.kind.is_air() && before[*position] != *block)
        .collect()
}

fn routeable_output_redstone_taps(world: &World3D, output_position: Position) -> Vec<Position> {
    let direct = if world[output_position].kind.is_redstone() {
        vec![output_position]
    } else {
        world
            .iter_block()
            .into_iter()
            .filter_map(|(position, block)| {
                (block.kind.is_redstone()
                    && detailed_router::target_powers_position(world, output_position, position))
                .then_some(position)
            })
            .collect::<Vec<_>>()
    };

    let mut taps = redstone_network_positions(world, &direct);
    taps.sort_by_key(|position| {
        (
            output_position.manhattan_distance(position),
            position.0,
            position.1,
            position.2,
        )
    });
    taps
}

fn isolated_output_adapter_world(
    world: &World3D,
    tap: Position,
    repeater_position: Position,
    direction: Direction,
) -> Option<(World3D, Position)> {
    if !world.size.bound_on(repeater_position) || !world[repeater_position].kind.is_air() {
        return None;
    }

    let mut adapter_world = world.clone();
    let support_position = repeater_position.down()?;
    place_support_cobble_if_needed(&mut adapter_world, support_position)?;

    let repeater = PlacedNode::new_repeater(repeater_position, direction);
    detailed_router::place_node(&mut adapter_world, repeater);
    if !detailed_router::target_powers_position(&adapter_world, tap, repeater_position) {
        return None;
    }

    Some((adapter_world, repeater_position))
}

fn place_support_cobble_if_needed(world: &mut World3D, position: Position) -> Option<()> {
    if world[position].kind.is_cobble() {
        return Some(());
    }
    if !world[position].kind.is_air() {
        return None;
    }

    world[position] = PlacedNode::new_cobble(position).block;
    Some(())
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

fn switch_powered_redstone_port(
    world: &World3D,
    input_position: Position,
    direct_only: bool,
) -> Option<Position> {
    let direct = world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| {
            (block.kind.is_redstone()
                && detailed_router::target_powers_position(world, input_position, position))
            .then_some(position)
        })
        .collect::<Vec<_>>();
    if direct.is_empty() {
        return None;
    }

    let candidates = if direct_only {
        direct
    } else {
        redstone_network_positions(world, &direct)
    };

    candidates.into_iter().max_by_key(|position| {
        (
            downstream_consumer_count(world, *position),
            Reverse(input_position.manhattan_distance(position)),
            Reverse(position.0),
            Reverse(position.1),
            Reverse(position.2),
        )
    })
}

fn redstone_network_positions(world: &World3D, seeds: &[Position]) -> Vec<Position> {
    let redstones = world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| block.kind.is_redstone().then_some(position))
        .collect::<Vec<_>>();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    for &seed in seeds {
        if visited.insert(seed) {
            queue.push_back(seed);
        }
    }

    while let Some(position) = queue.pop_front() {
        for &next in &redstones {
            if visited.contains(&next) {
                continue;
            }
            if detailed_router::target_powers_position(world, position, next)
                || detailed_router::target_powers_position(world, next, position)
            {
                visited.insert(next);
                queue.push_back(next);
            }
        }
    }

    visited.into_iter().collect()
}

fn downstream_consumer_count(world: &World3D, source: Position) -> usize {
    world
        .iter_block()
        .into_iter()
        .filter(|(position, block)| {
            *position != source
                && !block.kind.is_redstone()
                && detailed_router::target_powers_position(world, source, *position)
        })
        .count()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::place_and_route::local_placer::{
        InputPlacementStrategy, NotRouteStrategy, PlacementSamplingPolicy, TorchPlacementStrategy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;
    use crate::verilog::synth::d_latch_graph_module;

    fn redstone_block() -> Block {
        Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        }
    }

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
    fn expose_isolated_output_port_adds_repeater_diode_inside_candidate() {
        let mut world = World3D::new(DimSize(5, 3, 3));
        let tap = Position(1, 1, 1);
        world[tap.down().unwrap()] = PlacedNode::new_cobble(tap.down().unwrap()).block;
        world[tap] = redstone_block();
        world.initialize_redstone_states();

        let adapter = isolated_output_port_adapters(&world, tap)
            .into_iter()
            .next()
            .unwrap();

        assert!(adapter
            .blocks
            .iter()
            .any(|(position, block)| *position == adapter.position && block.kind.is_repeater()));
    }

    #[test]
    fn d_latch_candidate_keeps_direct_q_tap_and_exposes_isolated_repeater_port() -> eyre::Result<()>
    {
        let module = d_latch_graph_module("d_latch", "d", "en", "q");
        let candidates = generate_graph_module_candidates(
            &module,
            &d_latch_child_candidate_config(sequential_local_config()),
        )?;

        let q = candidates[0]
            .ports
            .iter()
            .find(|port| port.name == "q")
            .unwrap();
        let route_position = q.route_position.unwrap();
        let isolated_route_position = q.isolated_route_position.unwrap();

        assert!(q.isolate_output);
        assert!(candidates[0].world[route_position].kind.is_redstone());
        assert!(q
            .isolated_route_blocks
            .iter()
            .any(|(position, block)| *position == isolated_route_position
                && block.kind.is_repeater()));
        assert!(!q.isolated_route_options.is_empty());
        Ok(())
    }
}
