use std::path::PathBuf;

use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::world::block::BlockKind;
use redstone_compiler::world::simulator::{SimulationTraceEntry, Simulator};

const MAX_CYCLES: usize = 128;
const MAX_EVENTS: usize = 20_000;
const TRACE_LIMIT: usize = 20_000;

fn main() -> eyre::Result<()> {
    if std::env::var_os("REPRO_TOGGLE_TRACE").is_some() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .without_time()
            .init();
    }

    let paths = std::env::args()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    eyre::ensure!(!paths.is_empty(), "usage: repro_toggle <file.nbt>...");

    for path in paths {
        let bytes = std::fs::read(&path)?;
        let nbt = NBTRoot::from_nbt_bytes(&bytes)?;
        let world = nbt.to_world();
        let switches = world
            .blocks
            .iter()
            .filter_map(|(pos, block)| {
                let BlockKind::Switch { is_on } = block.kind else {
                    return None;
                };
                Some((*pos, is_on))
            })
            .collect::<Vec<_>>();

        println!("{}: {} switches", path.display(), switches.len());

        for (pos, is_on) in switches {
            let mut sim = match Simulator::from_with_limits_and_trace(
                &world,
                MAX_CYCLES,
                MAX_EVENTS,
                TRACE_LIMIT,
            ) {
                Ok(sim) => sim,
                Err(error) => {
                    println!("  init fail before toggling {:?}: {error}", pos);
                    print_trace_context(error.trace());
                    continue;
                }
            };
            let next = !is_on;
            sim.clear_trace();
            match sim.change_state_with_limits(vec![(pos, next)], MAX_CYCLES, MAX_EVENTS) {
                Ok(()) => {
                    println!("  ok {:?}: {} -> {}", pos, is_on, next);
                }
                Err(error) => {
                    println!("  fail {:?}: {} -> {}: {error}", pos, is_on, next);
                    print_trace_context(sim.trace());
                }
            }
        }
    }

    Ok(())
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
