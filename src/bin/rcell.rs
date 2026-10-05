use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::physical_cell::{
    PhysicalCellBuild, PhysicalCellCaseSimulation, PhysicalCellCompactReport, PhysicalCellDocument,
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

    /// Analyze physical cost, long chains, and individually safe simplifications.
    #[structopt(long)]
    compact_report: bool,

    /// Write the compact-analysis report as JSON.
    #[structopt(long, parse(from_os_str), value_name = "PATH")]
    compact_json: Option<PathBuf>,
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
    NBTRoot::from(&document.export_world(&build)).save(&output);

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
        print_verification(&document, &verification);
        if !verification.failures.is_empty() {
            if let Some(divergence) = &verification.first_divergence {
                eprintln!(
                    "first divergence: {} at {:?}, case {} inputs={:?}, expected={} actual={}",
                    divergence.observation,
                    divergence.position,
                    divergence.case_index,
                    divergence.inputs,
                    u8::from(divergence.expected),
                    u8::from(divergence.actual)
                );
                let traced = document.simulate_case(&build, divergence.inputs.clone(), 10_000)?;
                print_explanation(
                    &divergence.observation,
                    divergence.position,
                    &traced.simulator,
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
    if options.compact_report || options.compact_json.is_some() {
        let report = document.compact_report(&build)?;
        if options.compact_report {
            print_compact_report(&report);
        }
        if let Some(path) = &options.compact_json {
            std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
            println!("wrote compact analysis to {}", path.display());
        }
    }
    println!(
        "exported {} blocks ({} automatic supports) to {}",
        build.world.iter_block().len(),
        build.auto_supports.len(),
        output.display()
    );
    Ok(())
}

fn print_compact_report(report: &PhysicalCellCompactReport) {
    println!("compact analysis:");
    println!(
        "  contract: {}x{}x{}",
        report.contract_size[0], report.contract_size[1], report.contract_size[2]
    );
    if let Some(bounds) = &report.occupied_bounds {
        println!(
            "  occupied: {}x{}x{} min={:?} max={:?} volume={}",
            bounds.size[0],
            bounds.size[1],
            bounds.size[2],
            bounds.min,
            bounds.max,
            report.occupied_volume
        );
    }
    println!(
        "  blocks: total={} cobble={} dust={} repeaters={} torches={} switches={}",
        report.block_counts.values().sum::<usize>(),
        report.block_counts.get("Cobble").copied().unwrap_or(0),
        report.block_counts.get("Redstone").copied().unwrap_or(0),
        report.block_counts.get("Repeater").copied().unwrap_or(0),
        report.block_counts.get("Torch").copied().unwrap_or(0),
        report.block_counts.get("Switch").copied().unwrap_or(0),
    );
    println!(
        "  repeater delay: total={} chains={}",
        report.repeater_delay_total,
        report.repeater_chains.len()
    );
    for chain in &report.repeater_chains {
        println!(
            "    chain repeaters={} delay={} direction={} from={:?} to={:?}",
            chain.repeaters,
            chain.total_delay,
            chain.direction,
            chain.positions.first().expect("chain is non-empty"),
            chain.positions.last().expect("chain is non-empty")
        );
    }
    println!("  dust components: {}", report.dust_components.len());
    for component in report.dust_components.iter().take(8) {
        println!(
            "    blocks={} size={}x{}x{} min={:?} max={:?}",
            component.blocks,
            component.bounds.size[0],
            component.bounds.size[1],
            component.bounds.size[2],
            component.bounds.min,
            component.bounds.max
        );
    }
    println!(
        "  individually safe mutations: {}",
        report.safe_mutations.len()
    );
    for mutation in &report.safe_mutations {
        println!(
            "    {} {:?}: {} -> {}",
            mutation.mutation, mutation.position, mutation.original, mutation.replacement
        );
    }
    if report.safe_mutations.len() > 1 {
        println!("    note: each mutation passed independently; combinations must be reverified");
    }
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

fn print_verification(
    document: &PhysicalCellDocument,
    verification: &redstone_compiler::physical_cell::PhysicalCellVerification,
) {
    println!(
        "truth signatures (case bit order: {:?}, mask 0..{}):",
        verification.input_names,
        verification.cases.saturating_sub(1)
    );
    for (kind, name) in document
        .probes
        .iter()
        .map(|probe| ("probe", probe.name.as_str()))
        .chain(
            document
                .outputs
                .iter()
                .map(|output| ("output", output.name.as_str())),
        )
    {
        let signature = &verification.signatures[name];
        let status = if signature.expected == signature.actual {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "  {kind} {name}: expected={} actual={} influence expected={:?} actual={:?} {status}",
            signature_bits(&signature.expected),
            signature_bits(&signature.actual),
            signature.expected_influence,
            signature.actual_influence,
        );
    }
    println!(
        "physical checks: syntax/support/orientation/occupancy/bbox PASS; keepout not declared"
    );
}

fn signature_bits(values: &[bool]) -> String {
    values
        .iter()
        .map(|value| if *value { '1' } else { '0' })
        .collect()
}

fn resolve_target(target: &str, build: &PhysicalCellBuild) -> eyre::Result<(String, Position)> {
    if let Some(position) = build.observation_position(target) {
        return Ok((target.to_owned(), position));
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
        "unknown observation `{target}`; available probes/outputs are {:?}, or use x,y,z",
        build
            .observations()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    )
}

fn print_explanation(label: &str, position: Position, simulator: &Simulator) {
    println!("power explanation for {label} at {position:?}:");
    if simulator.is_torch_burned_out(position) {
        println!("diagnosis: observation torch burned out after repeated toggles");
    }
    let mut visited = BTreeSet::new();
    print_power_node(simulator, position, 0, &mut visited);
    let active_sources = visited
        .iter()
        .flat_map(|position| simulator.diagnostic_power_sources(*position))
        .filter(|source| source.active)
        .count();
    println!(
        "backward slice: nodes={} active_sources={}",
        visited.len(),
        active_sources
    );
    if !simulator.world()[position].kind.is_powered() && active_sources == 0 {
        println!("diagnosis: unpowered observation component has no active source");
    }
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
    if depth >= 16 || visited.len() >= 64 || !visited.insert(position) {
        if depth < 16 {
            println!("{indent}  (cycle or diagnostic node limit)");
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
        if (source.active || !block.kind.is_powered())
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
    let probe_names = build.probes.iter().fold(
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
                "torch_burned_out": case.simulator.is_torch_burned_out(position),
                "outputs": output_names.get(&position).cloned().unwrap_or_default(),
                "probes": probe_names.get(&position).cloned().unwrap_or_default(),
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
