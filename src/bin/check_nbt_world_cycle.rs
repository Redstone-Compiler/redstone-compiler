use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use petgraph::algo::kosaraju_scc;
use redstone_compiler::graph::logic::predefined_logics;
use redstone_compiler::graph::world::WorldGraphBuilder;
use redstone_compiler::graph::GraphNodeId;
use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::transform::place_and_route::utils::world_to_logic;
use redstone_compiler::transform::world::WorldGraphTransformer;
use redstone_compiler::world::block::BlockKind;
use redstone_compiler::world::position::Position;
use redstone_compiler::world::simulator::{SimulationTraceEntry, Simulator};

fn main() -> eyre::Result<()> {
    let paths = std::env::args()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    eyre::ensure!(
        !paths.is_empty(),
        "usage: check_nbt_world_cycle <file.nbt>..."
    );

    for path in paths {
        let bytes = std::fs::read(&path)?;
        let nbt = NBTRoot::from_nbt_bytes(&bytes)?;
        let world = nbt.to_world();
        let world_graph = WorldGraphBuilder::new(&world).build();
        let graph = &world_graph.graph;
        let has_cycle = graph.has_cycle();

        println!("{}", path.display());
        println!(
            "  raw nodes: {}, edges: {}, has_cycle: {}",
            graph.nodes.len(),
            graph
                .nodes
                .iter()
                .map(|node| node.outputs.len())
                .sum::<usize>(),
            has_cycle
        );

        let mut folded = WorldGraphTransformer::new(world_graph.clone());
        folded.fold_redstone();
        let folded = folded.finish();
        println!(
            "  folded-redstone nodes: {}, edges: {}, has_cycle: {}",
            folded.graph.nodes.len(),
            folded
                .graph
                .nodes
                .iter()
                .map(|node| node.outputs.len())
                .sum::<usize>(),
            folded.graph.has_cycle()
        );

        let node_kind_by_id = graph
            .nodes
            .iter()
            .map(|node| (node.id, node.kind.name()))
            .collect::<HashMap<GraphNodeId, String>>();

        let petgraph = graph.to_petgraph_only_edges();
        let cyclic_components = kosaraju_scc(&petgraph)
            .into_iter()
            .map(|component| {
                component
                    .into_iter()
                    .map(|node| petgraph[node])
                    .filter(|id| world_graph.positions.contains_key(id))
                    .collect::<Vec<_>>()
            })
            .filter(|component| component.len() > 1)
            .collect::<Vec<_>>();

        println!("  cyclic_components: {}", cyclic_components.len());
        for (index, component) in cyclic_components.iter().take(12).enumerate() {
            println!("    component #{index}: {} nodes", component.len());
            for id in component.iter().take(24) {
                let pos = world_graph.positions[id];
                let kind = node_kind_by_id
                    .get(id)
                    .map(String::as_str)
                    .unwrap_or("unknown");
                println!("      {id}: {kind} @ ({}, {}, {})", pos.0, pos.1, pos.2);
            }
            if component.len() > 24 {
                println!("      ... {} more", component.len() - 24);
            }
        }

        if let Ok(center) = std::env::var("PRINT_NEAR") {
            let parts = center
                .split(',')
                .map(str::trim)
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()?;
            eyre::ensure!(parts.len() == 3, "PRINT_NEAR must be x,y,z, got `{center}`");
            print_nearby_blocks(&world, Position(parts[0], parts[1], parts[2]), 2);
        }

        if std::env::var_os("PRINT_UNSUPPORTED").is_some() {
            print_unsupported_signal_blocks(&world);
        }

        if std::env::var_os("PRINT_REPEATERS").is_some() {
            print_repeaters(&world);
        }

        if let Ok(center) = std::env::var("PRINT_EDGES_NEAR") {
            let parts = center
                .split(',')
                .map(str::trim)
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()?;
            eyre::ensure!(
                parts.len() == 3,
                "PRINT_EDGES_NEAR must be x,y,z, got `{center}`"
            );
            print_edges_near(
                graph,
                &world_graph.positions,
                &node_kind_by_id,
                Position(parts[0], parts[1], parts[2]),
                4,
            );
        }

        if let Ok(target) = std::env::var("PRINT_TARGET_STATES") {
            let parts = target
                .split(',')
                .map(str::trim)
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()?;
            eyre::ensure!(
                parts.len() == 3,
                "PRINT_TARGET_STATES must be x,y,z, got `{target}`"
            );
            print_target_states(&world, Position(parts[0], parts[1], parts[2]))?;
        }

        if let Ok(target) = std::env::var("PRINT_TARGET_SEQUENCE") {
            let parts = target
                .split(',')
                .map(str::trim)
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()?;
            eyre::ensure!(
                parts.len() == 3,
                "PRINT_TARGET_SEQUENCE must be x,y,z, got `{target}`"
            );
            print_target_sequence(&world, Position(parts[0], parts[1], parts[2]))?;
        }

        if std::env::var_os("PRINT_DFF_ACTIVITY").is_some() {
            print_dff_activity(&world)?;
        }

        if std::env::var_os("STRESS_DFF_SIM").is_some() {
            stress_dff_simulation(&world)?;
        }

        if std::env::var_os("PRINT_DFF_WATCH").is_some() {
            print_dff_watch_sequence(&world)?;
        }

        if std::env::var_os("PRINT_LOGIC").is_some() {
            let logic = world_to_logic(&world)?;
            println!("  logic graph:");
            for node in &logic.graph.nodes {
                println!(
                    "    {}: {} pos={:?} inputs={:?} outputs={:?}",
                    node.id,
                    node.kind.name(),
                    world_graph.positions.get(&node.id),
                    node.inputs,
                    node.outputs
                );
            }

            let expected = match path.file_name().and_then(|name| name.to_str()) {
                Some("xor-generated.nbt") => Some(predefined_logics::buffered_xor_graph()?),
                Some("half-adder-generated.nbt") => {
                    Some(predefined_logics::buffered_half_adder_graph()?)
                }
                _ => None,
            };
            if let Some(expected) = expected {
                println!("  expected graph:");
                for node in &expected.graph.nodes {
                    println!(
                        "    {}: {} inputs={:?} outputs={:?}",
                        node.id,
                        node.kind.name(),
                        node.inputs,
                        node.outputs
                    );
                }
            }
        }

        if std::env::var_os("PRINT_TRUTH").is_some() {
            let leaves = graph
                .nodes
                .iter()
                .filter(|node| node.outputs.is_empty())
                .filter_map(|node| world_graph.positions.get(&node.id).copied())
                .collect::<Vec<_>>();
            let switches = world
                .blocks
                .iter()
                .filter_map(|(pos, block)| {
                    if matches!(block.kind, BlockKind::Switch { .. }) {
                        Some(*pos)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            println!("  switches: {switches:?}");
            println!("  leaves: {leaves:?}");
            let masks = if let Ok(mask) = std::env::var("PRINT_MASK") {
                vec![usize::from_str_radix(&mask, 2).or_else(|_| mask.parse())?]
            } else {
                (0..(1usize << switches.len())).collect::<Vec<_>>()
            };
            for mask in masks {
                let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 50_000)
                    .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
                let states = switches
                    .iter()
                    .enumerate()
                    .map(|(index, pos)| (*pos, (mask & (1 << index)) != 0))
                    .collect::<Vec<(Position, bool)>>();
                if let Err(error) = sim.change_state_with_limits(states.clone(), 256, 50_000) {
                    println!(
                        "  mask={mask:0width$b} states={states:?}",
                        width = switches.len()
                    );
                    print_trace_context(sim.trace());
                    return Err(error);
                }
                let world = sim.world();
                println!(
                    "  mask={mask:0width$b} states={states:?}",
                    width = switches.len()
                );
                for pos in &leaves {
                    println!("    leaf {:?}: {:?}", pos, world[*pos].kind);
                }
                for (pos, block) in world.iter_block() {
                    match block.kind {
                        BlockKind::Switch { is_on } => {
                            println!("    switch {:?} dir={:?}: {is_on}", pos, block.direction);
                        }
                        BlockKind::Torch { is_on } => {
                            println!("    torch {:?} dir={:?}: {is_on}", pos, block.direction);
                        }
                        BlockKind::Redstone { strength, .. } => {
                            println!(
                                "    redstone {:?} dir={:?}: {strength}",
                                pos, block.direction
                            );
                        }
                        _ => {}
                    }
                }
                if mask == (1usize << switches.len()) - 1 {
                    println!("  trace around xor output:");
                    print_trace_for_positions(
                        sim.trace(),
                        &[
                            Position(2, 1, 2),
                            Position(3, 1, 2),
                            Position(3, 0, 2),
                            Position(3, 2, 2),
                        ],
                    );
                }
            }

            if switches.len() >= 2 {
                let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 50_000)
                    .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
                println!("  sequential 00 -> 01 -> 11");
                sim.change_state_with_limits(vec![(switches[0], true)], 256, 50_000)?;
                print_signal_state(sim.world());
                sim.change_state_with_limits(vec![(switches[1], true)], 256, 50_000)?;
                print_signal_state(sim.world());

                let mut sim = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 50_000)
                    .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
                println!("  sequential 00 -> 10 -> 11");
                sim.change_state_with_limits(vec![(switches[1], true)], 256, 50_000)?;
                print_signal_state(sim.world());
                sim.change_state_with_limits(vec![(switches[0], true)], 256, 50_000)?;
                print_signal_state(sim.world());
            }
        }

        if std::env::var_os("PRINT_COMPONENTS").is_some() {
            print_physical_components(graph, &world_graph.positions, &node_kind_by_id);
        }
    }

    Ok(())
}

fn print_nearby_blocks(world: &redstone_compiler::world::World, center: Position, radius: usize) {
    println!("  nearby blocks around {:?}:", center);
    for (pos, block) in &world.blocks {
        let dx = pos.0.abs_diff(center.0);
        let dy = pos.1.abs_diff(center.1);
        let dz = pos.2.abs_diff(center.2);
        if dx <= radius && dy <= radius && dz <= radius {
            println!("    {:?}: {:?} dir={:?}", pos, block.kind, block.direction);
        }
    }
}

fn print_unsupported_signal_blocks(world: &redstone_compiler::world::World) {
    let blocks = world
        .blocks
        .iter()
        .filter(|(pos, block)| {
            matches!(
                block.kind,
                BlockKind::Redstone { .. } | BlockKind::Repeater { .. }
            ) && pos.down().is_none_or(|bottom| {
                !world
                    .blocks
                    .iter()
                    .any(|(p, b)| *p == bottom && matches!(b.kind, BlockKind::Cobble { .. }))
            })
        })
        .collect::<Vec<_>>();
    println!("  unsupported signal blocks: {}", blocks.len());
    for (pos, block) in blocks.iter().take(40) {
        println!("    {:?}: {:?} dir={:?}", pos, block.kind, block.direction);
    }
}

fn print_repeaters(world: &redstone_compiler::world::World) {
    println!("  repeaters:");
    for (pos, block) in &world.blocks {
        if matches!(block.kind, BlockKind::Repeater { .. }) {
            println!("    {:?}: {:?} dir={:?}", pos, block.kind, block.direction);
        }
    }
}

fn print_edges_near(
    graph: &redstone_compiler::graph::Graph,
    positions: &HashMap<GraphNodeId, Position>,
    node_kind_by_id: &HashMap<GraphNodeId, String>,
    center: Position,
    radius: usize,
) {
    println!("  raw edges near {:?}:", center);
    let nearby_ids = positions
        .iter()
        .filter_map(|(id, pos)| {
            (pos.0.abs_diff(center.0) <= radius
                && pos.1.abs_diff(center.1) <= radius
                && pos.2.abs_diff(center.2) <= radius)
                .then_some(*id)
        })
        .collect::<HashSet<_>>();
    for node in graph
        .nodes
        .iter()
        .filter(|node| nearby_ids.contains(&node.id))
    {
        let pos = positions[&node.id];
        let kind = node_kind_by_id
            .get(&node.id)
            .map(String::as_str)
            .unwrap_or("unknown");
        println!("    {}: {} @ {:?}", node.id, kind, pos);
        for input in &node.inputs {
            if let Some(input_pos) = positions.get(input) {
                let input_kind = node_kind_by_id
                    .get(input)
                    .map(String::as_str)
                    .unwrap_or("unknown");
                println!("      <- {}: {} @ {:?}", input, input_kind, input_pos);
            }
        }
        for output in &node.outputs {
            if let Some(output_pos) = positions.get(output) {
                let output_kind = node_kind_by_id
                    .get(output)
                    .map(String::as_str)
                    .unwrap_or("unknown");
                println!("      -> {}: {} @ {:?}", output, output_kind, output_pos);
            }
        }
    }
}

fn print_target_states(
    world: &redstone_compiler::world::World,
    target: Position,
) -> eyre::Result<()> {
    let switches = world
        .blocks
        .iter()
        .filter_map(|(pos, block)| matches!(block.kind, BlockKind::Switch { .. }).then_some(*pos))
        .collect::<Vec<_>>();
    println!("  target states for {:?}, switches: {:?}", target, switches);
    for mask in 0..(1usize << switches.len()) {
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        let states = switches
            .iter()
            .enumerate()
            .map(|(index, pos)| (*pos, (mask & (1 << index)) != 0))
            .collect::<Vec<_>>();
        sim.change_state_with_limits(states, 256, 50_000)?;
        println!(
            "    mask={mask:0width$b} {:?}",
            sim.world()[target].kind,
            width = switches.len()
        );
    }
    Ok(())
}

fn print_target_sequence(
    world: &redstone_compiler::world::World,
    target: Position,
) -> eyre::Result<()> {
    let switches = world
        .blocks
        .iter()
        .filter_map(|(pos, block)| matches!(block.kind, BlockKind::Switch { .. }).then_some(*pos))
        .collect::<Vec<_>>();
    eyre::ensure!(switches.len() >= 2, "expected at least d and clk switches");
    let d = switches[0];
    let clk = switches[1];
    let mut sim =
        Simulator::from_preserving_torch_states_with_limits_and_trace(world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
    println!(
        "  target sequence for {:?}, d={:?}, clk={:?}",
        target, d, clk
    );
    println!("    init {:?}", sim.world()[target].kind);
    for (name, pos, value) in [
        ("d=1", d, true),
        ("clk=1", clk, true),
        ("clk=0", clk, false),
        ("d=0", d, false),
        ("clk=1", clk, true),
    ] {
        sim.change_state_with_limits(vec![(pos, value)], 256, 50_000)?;
        println!("    {name:<5} {:?}", sim.world()[target].kind);
    }
    Ok(())
}

fn print_dff_activity(world: &redstone_compiler::world::World) -> eyre::Result<()> {
    let switches = world
        .blocks
        .iter()
        .filter_map(|(pos, block)| matches!(block.kind, BlockKind::Switch { .. }).then_some(*pos))
        .collect::<Vec<_>>();
    eyre::ensure!(switches.len() >= 2, "expected d and clk switches");
    let d = switches[0];
    let clk = switches[1];
    let q = Position(22, 6, 5);
    let redstones = world
        .blocks
        .iter()
        .filter_map(|(pos, block)| matches!(block.kind, BlockKind::Redstone { .. }).then_some(*pos))
        .collect::<Vec<_>>();
    let mut ever_on = HashSet::<Position>::new();
    let mut sim =
        Simulator::from_preserving_torch_states_with_limits_and_trace(world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
    record_on_redstones(sim.world(), &redstones, &mut ever_on);

    let randomish_steps = [
        ("#1 on", d, true),
        ("#2 on", clk, true),
        ("#1 off", d, false),
        ("#2 off", clk, false),
        ("#2 on", clk, true),
        ("#1 on", d, true),
        ("#2 off", clk, false),
        ("#1 off", d, false),
        ("#1 on", d, true),
        ("#2 on", clk, true),
    ];
    println!(
        "  dff randomish activity: #1={:?}, #2={:?}, q={:?}",
        d, clk, q
    );
    println!("    init q={}", block_power(sim.world(), q));
    for (name, pos, value) in randomish_steps {
        sim.change_state_with_limits(vec![(pos, value)], 256, 50_000)?;
        record_on_redstones(sim.world(), &redstones, &mut ever_on);
        println!("    {name:<7} q={}", block_power(sim.world(), q));
    }

    let never_on = redstones
        .iter()
        .copied()
        .filter(|pos| !ever_on.contains(pos))
        .collect::<Vec<_>>();
    println!(
        "    redstones never powered during randomish sequence: {} / {}",
        never_on.len(),
        redstones.len()
    );
    for pos in never_on.iter().take(80) {
        println!("      never_on {:?}", pos);
    }

    print_dff_edge_checks(world, "assume #1=d #2=clk", d, clk, q)?;
    print_dff_edge_checks(world, "assume #1=clk #2=d", clk, d, q)?;

    Ok(())
}

fn print_dff_edge_checks(
    world: &redstone_compiler::world::World,
    label: &str,
    d: Position,
    clk: Position,
    q: Position,
) -> eyre::Result<()> {
    println!("  dff edge checks {label}: d={:?}, clk={:?}", d, clk);
    println!("    rising:");
    for data in [false, true, false] {
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        sim.change_state_with_limits(vec![(clk, false)], 256, 50_000)?;
        sim.change_state_with_limits(vec![(d, data)], 256, 50_000)?;
        let before = block_power(sim.world(), q);
        sim.change_state_with_limits(vec![(clk, true)], 256, 50_000)?;
        let after = block_power(sim.world(), q);
        println!("      d={data:<5} q_before_edge={before:<5} q_after_rising={after}");
    }
    println!("    falling:");
    for data in [false, true, false] {
        let mut sim =
            Simulator::from_preserving_torch_states_with_limits_and_trace(world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
        sim.change_state_with_limits(vec![(clk, true)], 256, 50_000)?;
        sim.change_state_with_limits(vec![(d, data)], 256, 50_000)?;
        let before = block_power(sim.world(), q);
        sim.change_state_with_limits(vec![(clk, false)], 256, 50_000)?;
        let after = block_power(sim.world(), q);
        println!("      d={data:<5} q_before_edge={before:<5} q_after_falling={after}");
    }
    Ok(())
}

fn stress_dff_simulation(world: &redstone_compiler::world::World) -> eyre::Result<()> {
    let switches = world
        .blocks
        .iter()
        .filter_map(|(pos, block)| matches!(block.kind, BlockKind::Switch { .. }).then_some(*pos))
        .collect::<Vec<_>>();
    eyre::ensure!(switches.len() >= 2, "expected d and clk switches");
    let output = Position(22, 6, 5);
    let iterations = std::env::var("STRESS_DFF_SIM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(500);

    for (label, data, clock) in [
        ("#1=data #2=clock", switches[0], switches[1]),
        ("#1=clock #2=data", switches[1], switches[0]),
    ] {
        let mut signatures = HashMap::<Vec<bool>, usize>::new();
        let mut edge_failures = Vec::new();
        let mut runtime_failures = Vec::new();

        for iteration in 0..iterations {
            match run_dff_stress_sequence(world, data, clock, output) {
                Ok(result) => {
                    *signatures.entry(result.timeline.clone()).or_default() += 1;
                    if !result.edge_expectations_hold {
                        edge_failures.push((iteration, result.timeline));
                    }
                }
                Err(error) => {
                    runtime_failures.push((iteration, error.to_string()));
                }
            }
        }

        println!("  stress {label}: iterations={iterations}");
        println!("    runtime failures: {}", runtime_failures.len());
        for (iteration, error) in runtime_failures.iter().take(8) {
            println!("      iter {iteration}: {error}");
        }
        println!("    distinct timelines: {}", signatures.len());
        for (timeline, count) in &signatures {
            println!("      {count}x {:?}", timeline);
        }
        println!("    edge expectation failures: {}", edge_failures.len());
        for (iteration, timeline) in edge_failures.iter().take(8) {
            println!("      iter {iteration}: {:?}", timeline);
        }
    }

    Ok(())
}

fn print_dff_watch_sequence(world: &redstone_compiler::world::World) -> eyre::Result<()> {
    let d = Position(28, 1, 1);
    let clk = Position(30, 4, 1);
    let watched = [
        ("#1/d_switch", d),
        ("#2/clk_switch", clk),
        ("not_clk.in", Position(1, 0, 4)),
        ("not_clk.out", Position(2, 1, 3)),
        ("master.d", Position(7, 0, 1)),
        ("master.en", Position(8, 5, 1)),
        ("master.q", Position(10, 6, 5)),
        ("slave.d", Position(19, 0, 1)),
        ("slave.d.stub", Position(18, 1, 2)),
        ("slave.en", Position(20, 5, 1)),
        ("slave.q", Position(22, 6, 5)),
    ];
    let mut sim =
        Simulator::from_preserving_torch_states_with_limits_and_trace(world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

    println!("  dff watch sequence:");
    print_watch_row("init", sim.world(), &watched);
    for (name, pos, value) in [
        ("d=1", d, true),
        ("clk=1", clk, true),
        ("clk=0", clk, false),
        ("d=0", d, false),
        ("clk=1", clk, true),
        ("clk=0", clk, false),
        ("d=1", d, true),
        ("clk=1", clk, true),
    ] {
        sim.change_state_with_limits(vec![(pos, value)], 256, 50_000)?;
        print_watch_row(name, sim.world(), &watched);
    }
    Ok(())
}

fn print_watch_row(
    name: &str,
    world: &redstone_compiler::world::World3D,
    watched: &[(&str, Position)],
) {
    println!("    {name}:");
    for (label, pos) in watched {
        println!(
            "      {label:<14} {:<5} {:?}",
            block_power(world, *pos),
            world[*pos].kind
        );
    }
}

struct DffStressResult {
    timeline: Vec<bool>,
    edge_expectations_hold: bool,
}

fn run_dff_stress_sequence(
    world: &redstone_compiler::world::World,
    data: Position,
    clock: Position,
    output: Position,
) -> eyre::Result<DffStressResult> {
    let mut sim =
        Simulator::from_preserving_torch_states_with_limits_and_trace(world, 256, 50_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

    let mut timeline = Vec::new();
    timeline.push(block_power(sim.world(), output));
    for (pos, value) in [
        (data, true),
        (clock, true),
        (clock, false),
        (data, false),
        (clock, true),
        (clock, false),
        (data, true),
        (clock, true),
        (data, false),
        (clock, false),
    ] {
        sim.change_state_with_limits(vec![(pos, value)], 256, 50_000)?;
        timeline.push(block_power(sim.world(), output));
    }

    // Positive-edge DFF expectation for the sequence above:
    // initial q=false, then q captures data on each clock false->true transition.
    let edge_expectations_hold = timeline[2] == true && timeline[5] == false && timeline[8] == true;

    Ok(DffStressResult {
        timeline,
        edge_expectations_hold,
    })
}

fn record_on_redstones(
    world: &redstone_compiler::world::World3D,
    redstones: &[Position],
    ever_on: &mut HashSet<Position>,
) {
    for pos in redstones {
        if block_power(world, *pos) {
            ever_on.insert(*pos);
        }
    }
}

fn block_power(world: &redstone_compiler::world::World3D, pos: Position) -> bool {
    match world[pos].kind {
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
        _ => false,
    }
}

fn print_physical_components(
    graph: &redstone_compiler::graph::Graph,
    positions: &HashMap<GraphNodeId, Position>,
    node_kind_by_id: &HashMap<GraphNodeId, String>,
) {
    let mut neighbors = HashMap::<GraphNodeId, Vec<GraphNodeId>>::new();
    for node in &graph.nodes {
        for &output in &node.outputs {
            neighbors.entry(node.id).or_default().push(output);
            neighbors.entry(output).or_default().push(node.id);
        }
    }

    let mut visited = HashSet::new();
    let mut components = Vec::new();
    for node in &graph.nodes {
        if !visited.insert(node.id) {
            continue;
        }
        let mut queue = VecDeque::from([node.id]);
        let mut component = Vec::new();
        while let Some(id) = queue.pop_front() {
            component.push(id);
            for &next in neighbors.get(&id).into_iter().flatten() {
                if visited.insert(next) {
                    queue.push_back(next);
                }
            }
        }
        component.sort();
        components.push(component);
    }
    components.sort_by_key(|component| std::cmp::Reverse(component.len()));

    println!("  physical_components: {}", components.len());
    for (index, component) in components.iter().take(20).enumerate() {
        let has_switch = component
            .iter()
            .any(|id| node_kind_by_id.get(id).is_some_and(|kind| kind == "Switch"));
        let has_torch = component
            .iter()
            .any(|id| node_kind_by_id.get(id).is_some_and(|kind| kind == "Torch"));
        let has_repeater = component.iter().any(|id| {
            node_kind_by_id
                .get(id)
                .is_some_and(|kind| kind == "Repeater")
        });
        let redstone_count = component
            .iter()
            .filter(|id| {
                node_kind_by_id
                    .get(id)
                    .is_some_and(|kind| kind == "Redstone")
            })
            .count();

        println!(
            "    component #{index}: nodes={} redstone={} switch={} torch={} repeater={}",
            component.len(),
            redstone_count,
            has_switch,
            has_torch,
            has_repeater
        );
        for id in component.iter().take(20) {
            let Some(pos) = positions.get(id) else {
                continue;
            };
            let kind = node_kind_by_id
                .get(id)
                .map(String::as_str)
                .unwrap_or("unknown");
            println!("      {id}: {kind} @ ({}, {}, {})", pos.0, pos.1, pos.2);
        }
        if component.len() > 20 {
            println!("      ... {} more", component.len() - 20);
        }
    }
}

fn print_trace_for_positions(trace: &[SimulationTraceEntry], positions: &[Position]) {
    for entry in trace.iter().filter(|entry| {
        positions.iter().any(|pos| {
            entry.target_position[0] == pos.0
                && entry.target_position[1] == pos.1
                && entry.target_position[2] == pos.2
        })
    }) {
        println!(
            "    #{} cycle={} {} target={},{},{} dir={} block={}",
            entry.event_id.unwrap_or_default(),
            entry.cycle,
            entry.event_type,
            entry.target_position[0],
            entry.target_position[1],
            entry.target_position[2],
            entry.direction,
            entry.block_before
        );
    }
}

fn print_trace_context(trace: &[SimulationTraceEntry]) {
    let Some(last) = trace.last() else {
        println!("    trace: empty");
        return;
    };

    let cycle = last.cycle;
    let target = last.target_position;

    println!(
        "    trace context: cycle={cycle}, last target=({}, {}, {}), {} events total",
        target[0],
        target[1],
        target[2],
        trace.len()
    );

    let cycle_events = trace
        .iter()
        .filter(|entry| entry.cycle == cycle)
        .collect::<Vec<_>>();
    println!("    cycle {cycle}: {} events", cycle_events.len());

    println!("    target history in cycle {cycle}:");
    for entry in cycle_events
        .iter()
        .filter(|entry| entry.target_position == target)
    {
        print_trace_entry(entry);
    }

    println!("    ancestry:");
    let mut current = last;
    print_trace_entry(current);
    while let Some(from_id) = current.from_event_id {
        let Some(parent) = trace.iter().find(|entry| entry.event_id == Some(from_id)) else {
            println!("      missing parent #{from_id}");
            break;
        };
        current = parent;
        print_trace_entry(current);
    }

    println!("    recent cycle events:");
    for entry in cycle_events.iter().rev().take(80).rev() {
        print_trace_entry(entry);
    }
}

fn print_trace_entry(entry: &SimulationTraceEntry) {
    println!(
        "      #{:<5?} <- #{:<5?} cycle={:<3} {:<24} target={},{},{} dir={:<6} block={}",
        entry.event_id,
        entry.from_event_id,
        entry.cycle,
        entry.event_type,
        entry.target_position[0],
        entry.target_position[1],
        entry.target_position[2],
        entry.direction,
        entry.block_before
    );
}

fn print_signal_state(world: &redstone_compiler::world::World3D) {
    for (pos, block) in world.iter_block() {
        match block.kind {
            BlockKind::Switch { is_on } => {
                println!("    switch {:?} dir={:?}: {is_on}", pos, block.direction);
            }
            BlockKind::Torch { is_on } => {
                println!("    torch {:?} dir={:?}: {is_on}", pos, block.direction);
            }
            BlockKind::Redstone { strength, .. } => {
                println!(
                    "    redstone {:?} dir={:?}: {strength}",
                    pos, block.direction
                );
            }
            _ => {}
        }
    }
}
