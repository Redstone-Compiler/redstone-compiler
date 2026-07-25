use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::physical_cell::{
    PhysicalCellBuild, PhysicalCellCaseSimulation, PhysicalCellDocument,
};
use redstone_compiler::world::block::BlockKind;
use redstone_compiler::world::position::Position;
use redstone_compiler::world::simulator::Simulator;
use serde_json::{json, Value};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "rcell", about = "Compile and verify a physical redstone cell")]
struct Options {
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// Print the canonical source after parsing.
    #[structopt(long)]
    emit: bool,

    /// Recompile whenever the input file changes.
    #[structopt(long)]
    watch: bool,

    /// Build and export without running `expect` statements.
    #[structopt(long)]
    no_verify: bool,

    /// Run one truth-table case, for example `a=1,b=0,cin=1`.
    #[structopt(long, value_name = "ASSIGNMENTS")]
    case: Option<String>,

    /// Compare the selected case with a second set of assignments.
    #[structopt(long, value_name = "ASSIGNMENTS", requires = "case")]
    compare_case: Option<String>,

    /// Explain the final power state of an output name or `x,y,z` coordinate.
    #[structopt(long, value_name = "OUTPUT_OR_POSITION", requires = "case")]
    explain: Option<String>,

    /// Write every non-air block's final state and candidate power sources as JSON.
    #[structopt(long, parse(from_os_str), value_name = "PATH", requires = "case")]
    state_json: Option<PathBuf>,
}

fn main() -> eyre::Result<()> {
    let options = Options::from_args();
    if options.watch {
        watch(&options)
    } else {
        compile(&options)
    }
}

fn compile(options: &Options) -> eyre::Result<()> {
    let source = std::fs::read_to_string(&options.input)?;
    let document: PhysicalCellDocument = source.parse()?;
    let build = document.build()?;
    let output = options
        .output
        .clone()
        .unwrap_or_else(|| options.input.with_extension("nbt"));
    NBTRoot::from(&build.world).save(&output);

    if options.emit {
        print!("{document}");
    }
    if let Some(case) = &options.case {
        let first = document.simulate_case(&build, parse_case(case)?, 10_000)?;
        print_case(&first);
        if let Some(path) = &options.state_json {
            write_state_json(path, &build, &first)?;
            println!("wrote state diagnostics to {}", path.display());
        }
        if let Some(target) = &options.explain {
            let (label, position) = resolve_target(target, &build)?;
            print_explanation(&label, position, &first.simulator);
        }
        if let Some(second) = &options.compare_case {
            let second = document.simulate_case(&build, parse_case(second)?, 0)?;
            print_case_diff(&first, &second);
        }
        if first.actual != first.expected {
            eyre::bail!("selected truth-table case failed");
        }
    } else if !options.no_verify {
        let verification = document.verify(&build)?;
        if !verification.failures.is_empty() {
            for failure in &verification.failures {
                eprintln!(
                    "FAIL inputs={:?} expected={:?} actual={:?}",
                    failure.inputs, failure.expected, failure.actual
                );
            }
            eyre::bail!(
                "{} of {} truth-table cases failed",
                verification.failures.len(),
                verification.cases
            );
        }
        println!("verified {} truth-table cases", verification.cases);
    }
    println!(
        "exported {} blocks ({} automatic supports) to {}",
        build.world.iter_block().len(),
        build.auto_supports.len(),
        output.display()
    );
    Ok(())
}

fn parse_case(text: &str) -> eyre::Result<BTreeMap<String, bool>> {
    let mut values = BTreeMap::new();
    for assignment in text
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (name, value) = assignment
            .split_once('=')
            .ok_or_else(|| eyre::eyre!("case assignment `{assignment}` must use name=value"))?;
        let value = match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => true,
            "0" | "false" | "off" => false,
            other => eyre::bail!("case value `{other}` must be 0/1, false/true, or off/on"),
        };
        eyre::ensure!(
            values.insert(name.trim().to_owned(), value).is_none(),
            "duplicate case input `{}`",
            name.trim()
        );
    }
    eyre::ensure!(
        !values.is_empty(),
        "case must contain at least one assignment"
    );
    Ok(values)
}

fn print_case(case: &PhysicalCellCaseSimulation) {
    let status = if case.actual == case.expected {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "{status} inputs={:?} expected={:?} actual={:?}",
        case.inputs, case.expected, case.actual
    );
}

fn resolve_target(target: &str, build: &PhysicalCellBuild) -> eyre::Result<(String, Position)> {
    if let Some(position) = build.outputs.get(target) {
        return Ok((target.to_owned(), *position));
    }
    let coordinates = target
        .split(',')
        .map(str::trim)
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>();
    if let Ok(coordinates) = coordinates {
        if coordinates.len() == 3 {
            let position = Position(coordinates[0], coordinates[1], coordinates[2]);
            eyre::ensure!(
                build.world.size.bound_on(position),
                "explain coordinate {position:?} is outside the cell"
            );
            return Ok((format!("{position:?}"), position));
        }
    }
    eyre::bail!(
        "unknown output `{target}`; available outputs are {:?}, or use x,y,z",
        build.outputs.keys().collect::<Vec<_>>()
    )
}

fn print_explanation(label: &str, position: Position, simulator: &Simulator) {
    println!("power explanation for {label} at {position:?}:");
    let mut visited = BTreeSet::new();
    print_power_node(simulator, position, 0, &mut visited);
}

fn print_power_node(
    simulator: &Simulator,
    position: Position,
    depth: usize,
    visited: &mut BTreeSet<Position>,
) {
    let block = simulator.world()[position];
    let indent = "  ".repeat(depth);
    println!(
        "{indent}- {position:?} kind={} powered={} state={}",
        block.kind.name(),
        block.kind.is_powered(),
        block_state_summary(block.kind)
    );
    if depth >= 16 || !visited.insert(position) {
        if depth < 16 {
            println!("{indent}  (cycle)");
        }
        return;
    }
    let sources =
        select_explanation_sources(block.kind, simulator.diagnostic_power_sources(position));
    if sources.is_empty() {
        println!("{indent}  no candidate power sources");
        return;
    }
    for source in sources {
        println!(
            "{indent}  <- {} {:?} kind={} active={} strength={} hard={}",
            source.relation,
            source.position,
            source.kind,
            source.active,
            source.strength,
            source.hard
        );
        if source.active
            && !source.relation.ends_with("-side")
            && source.relation != "switch-support"
        {
            print_power_node(
                simulator,
                Position(source.position[0], source.position[1], source.position[2]),
                depth + 1,
                visited,
            );
        }
    }
}

fn select_explanation_sources(
    target: BlockKind,
    sources: Vec<redstone_compiler::world::simulator::SimulationPowerSource>,
) -> Vec<redstone_compiler::world::simulator::SimulationPowerSource> {
    let BlockKind::Redstone {
        strength: target_strength,
        ..
    } = target
    else {
        return sources;
    };
    if target_strength == 0 {
        return sources;
    }
    let direct = sources
        .iter()
        .filter(|source| source.active && source.relation == "direct-power")
        .cloned()
        .collect::<Vec<_>>();
    if !direct.is_empty() {
        return direct;
    }
    let strongest = sources
        .iter()
        .filter(|source| {
            source.active && source.relation == "dust-neighbor" && source.strength > target_strength
        })
        .map(|source| source.strength)
        .max();
    if let Some(strength) = strongest {
        sources
            .into_iter()
            .filter(|source| source.relation == "dust-neighbor" && source.strength == strength)
            .collect()
    } else {
        sources
    }
}

fn write_state_json(
    path: &Path,
    build: &PhysicalCellBuild,
    case: &PhysicalCellCaseSimulation,
) -> eyre::Result<()> {
    let output_names = build.outputs.iter().fold(
        BTreeMap::<Position, Vec<&str>>::new(),
        |mut names, (name, position)| {
            names.entry(*position).or_default().push(name);
            names
        },
    );
    let blocks = case
        .simulator
        .world()
        .iter_block()
        .into_iter()
        .map(|(position, block)| {
            json!({
                "position": [position.0, position.1, position.2],
                "kind": block.kind.name(),
                "powered": block.kind.is_powered(),
                "direction": format!("{:?}", block.direction),
                "state": block_state_json(block.kind),
                "outputs": output_names.get(&position).cloned().unwrap_or_default(),
                "power_sources": case.simulator.diagnostic_power_sources(position),
            })
        })
        .collect::<Vec<_>>();
    let value = json!({
        "inputs": case.inputs,
        "expected": case.expected,
        "actual": case.actual,
        "blocks": blocks,
        "trace": case.simulator.trace(),
    });
    std::fs::write(path, serde_json::to_vec_pretty(&value)?)?;
    Ok(())
}

fn block_state_json(kind: BlockKind) -> Value {
    match kind {
        BlockKind::Air => json!({}),
        BlockKind::Cobble {
            on_count,
            on_base_count,
        } => json!({"power_count": on_count, "hard_power_count": on_base_count}),
        BlockKind::Switch { is_on } => json!({"on": is_on}),
        BlockKind::Redstone {
            on_count,
            state,
            strength,
        } => json!({
            "direct_power_count": on_count,
            "connections": state,
            "power": strength,
        }),
        BlockKind::Torch { is_on } => json!({"lit": is_on}),
        BlockKind::Repeater {
            is_on,
            is_locked,
            delay,
            ..
        } => json!({"powered": is_on, "locked": is_locked, "delay": delay}),
        BlockKind::RedstoneBlock => json!({"powered": true}),
        BlockKind::Piston { is_on, .. } => json!({"powered": is_on}),
    }
}

fn block_state_summary(kind: BlockKind) -> String {
    match kind {
        BlockKind::Redstone { strength, .. } => format!("power={strength}"),
        BlockKind::Cobble {
            on_count,
            on_base_count,
        } => format!("sources={on_count} hard_sources={on_base_count}"),
        BlockKind::Torch { is_on } => format!("lit={is_on}"),
        BlockKind::Repeater {
            is_on,
            is_locked,
            delay,
            ..
        } => format!("powered={is_on} locked={is_locked} delay={delay}"),
        BlockKind::Switch { is_on } => format!("on={is_on}"),
        _ => format!("{kind:?}"),
    }
}

fn print_case_diff(first: &PhysicalCellCaseSimulation, second: &PhysicalCellCaseSimulation) {
    println!("state diff {:?} -> {:?}:", first.inputs, second.inputs);
    let mut changes = 0usize;
    for (position, first_block) in first.simulator.world().iter_block() {
        let second_block = second.simulator.world()[position];
        if first_block.kind != second_block.kind {
            changes += 1;
            println!(
                "  {position:?}: {} -> {}",
                block_state_summary(first_block.kind),
                block_state_summary(second_block.kind)
            );
        }
    }
    println!("{changes} block states changed");
}

fn watch(options: &Options) -> eyre::Result<()> {
    let mut modified = None;
    loop {
        let next = last_modified(&options.input)?;
        if modified != Some(next) {
            modified = Some(next);
            if let Err(error) = compile(options) {
                eprintln!("{error:?}");
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn last_modified(path: &Path) -> eyre::Result<SystemTime> {
    Ok(std::fs::metadata(path)?.modified()?)
}
