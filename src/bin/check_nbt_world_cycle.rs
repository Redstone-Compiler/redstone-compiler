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
