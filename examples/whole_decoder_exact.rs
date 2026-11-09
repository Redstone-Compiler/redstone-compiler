//! One simultaneous exact SAT model for the whole four-input seven-output decoder.
//! No physical cells, global routing, or window decomposition.
//! Usage: whole_decoder_exact FRESH_DIRECTORY XxYxZ SOLVER_SECONDS
//! WHOLE_WORKERS defaults to 1; WHOLE_PIN_CONTRACT=faces restricts input/output faces.
//! Solver seconds exclude encoding; orchestrators should enforce a wall timeout.
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eyre::{ensure, ContextCompat};
use redstone_compiler::ir::LogicalDesign;
use redstone_compiler::nbt::{NBTRoot, ToNBT};
use redstone_compiler::output::PlacedWorld;
use redstone_compiler::transform::place_and_route::local_placer::exact::{
    ExactLocalPlacer, ExactOutcome, ExactPlacerConfig, NetDriver,
};
use redstone_compiler::world::position::DimSize;
use redstone_compiler::world::simulator::Simulator;
const MASKS: [u8; 16] = [
    0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f, 0, 0, 0, 0, 0, 0,
];
fn index(name: &str, prefix: &str) -> eyre::Result<usize> {
    Ok(name
        .strip_prefix(prefix)
        .and_then(|s| s.strip_prefix('_'))
        .context("unexpected endpoint name")?
        .parse()?)
}
fn check(sim: &mut Simulator, placed: &PlacedWorld, value: usize) -> eyre::Result<()> {
    sim.advance_idle_cycles(61)?;
    let inputs = placed
        .inputs
        .iter()
        .map(|p| Ok((p.position(), value & (1 << index(&p.name, "digit")?) != 0)))
        .collect::<eyre::Result<Vec<_>>>()?;
    sim.drive_inputs_with_limits(inputs, 4096, 2_000_000)?;
    for output in &placed.outputs {
        let bit = index(&output.name, "seg")?;
        ensure!(
            sim.world()[output.position()].kind.is_powered() == (MASKS[value] & (1 << bit) != 0),
            "input {value}, output {} mismatch",
            output.name
        );
    }
    ensure!(
        !sim.world()
            .iter_block()
            .iter()
            .any(|(p, _)| sim.is_torch_burned_out(*p)),
        "torch burnout"
    );
    Ok(())
}
fn main() -> eyre::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let directory = PathBuf::from(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("target/alu-display/whole-decoder-exact"),
    );
    let dims = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("3x8x4")
        .split('x')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    ensure!(dims.len() == 3, "expected XxYxZ");
    let seconds = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("30")
        .parse::<u64>()?;
    std::fs::create_dir_all(&directory)?;
    ensure!(
        !directory.join("attempt.json").exists(),
        "use a fresh attempt directory"
    );
    let source = format!(
        "{}\n{}",
        include_str!("../test/alu_display/bcd_to_7seg_minimized.v"),
        include_str!("../test/alu_display/bcd_digit_to_7seg.v")
    );
    let logical = LogicalDesign::from_verilog_source(&source)?;
    std::fs::write(directory.join("source.v"), source)?;
    std::fs::write(directory.join("logical.rcir"), logical.to_string())?;
    let graph = logical.lower_flat_logic_graph()?;
    let truth = graph.truth_table()?;
    ensure!(
        truth.input_names.len() == 4 && truth.output_tables.len() == 7,
        "expected four inputs/seven outputs"
    );
    for mask in 0..16 {
        let mut digit = 0;
        for (i, name) in truth.input_names.iter().enumerate() {
            if mask & (1 << i) != 0 {
                digit |= 1 << index(name, "digit")?;
            }
        }
        for (name, values) in &truth.output_tables {
            ensure!(
                values[mask] == (MASKS[digit] & (1 << index(name, "seg")?) != 0),
                "flattened graph truth mismatch"
            );
        }
    }
    let placer = ExactLocalPlacer::new(&graph)?;
    std::fs::write(directory.join("graph.txt"), format!("{graph:#?}"))?;
    std::fs::write(
        directory.join("nor.txt"),
        format!("{:#?}", placer.netlist()),
    )?;
    let mut config = ExactPlacerConfig::new(DimSize(dims[0], dims[1], dims[2]));
    config.time_limit = Some(Duration::from_secs(seconds));
    config.workers = std::env::var("WHOLE_WORKERS")
        .unwrap_or("1".into())
        .parse()?;
    ensure!(config.workers > 0, "workers must be positive");
    config.seed = 1;
    config.optimize = false;
    config.driving_outputs = truth.output_tables.keys().cloned().collect();
    let pin_contract = std::env::var("WHOLE_PIN_CONTRACT").unwrap_or("free".into());
    ensure!(
        pin_contract == "free" || pin_contract == "faces",
        "expected free or faces pin contract"
    );
    if pin_contract == "faces" {
        use redstone_compiler::world::block::Direction;
        use redstone_compiler::world::position::Position;
        ensure!(
            dims[0] >= 5 && dims[2] >= 5,
            "faces contract needs X,Z >= 5"
        );
        for (bit, (x, z)) in [(1, 1), (3, 1), (1, 3), (3, 3)].into_iter().enumerate() {
            config.input_sites.insert(
                format!("digit_{bit}"),
                vec![(Position(x, 0, z), Direction::North)],
            );
        }
        let output_y = dims[1] - 1;
        let sites = (0..dims[0])
            .flat_map(|x| (0..dims[2]).map(move |z| Position(x, output_y, z)))
            .collect::<Vec<_>>();
        for name in truth.output_tables.keys() {
            config.output_sites.insert(name.clone(), sites.clone());
        }
    }

    let functions = placer.netlist().net_values();
    let unique_functions = functions
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .len();
    let start = Instant::now();
    let mut report = serde_json::json!({"mode":"whole-model-exact","pin_contract":pin_contract,"inputs":4,"outputs":7,"logical_static_cases":16,"net_functions":functions,"unique_net_functions":unique_functions,"nor_nets":placer.netlist().nets.len(),"nor_gates":placer.netlist().nets.iter().filter(|n|n.driver==NetDriver::Gate).count(),"dim":dims,"time_limit_seconds":seconds,"workers":config.workers,"seed":1,"optimize":false,"status":"running"});
    report["input_sites"] = serde_json::to_value(config.input_sites.iter().map(|(name, sites)| (name, sites.iter().map(|(p,d)| serde_json::json!({"position":[p.0,p.1,p.2],"attach":format!("{d:?}")})).collect::<Vec<_>>())).collect::<std::collections::BTreeMap<_,_>>())?;
    report["output_sites"] = serde_json::to_value(
        config
            .output_sites
            .iter()
            .map(|(name, sites)| {
                (
                    name,
                    sites.iter().map(|p| [p.0, p.1, p.2]).collect::<Vec<_>>(),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>(),
    )?;
    std::fs::write(
        directory.join("attempt.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("START dim={dims:?} inputs=4 outputs=7 NOR={} unique_functions={unique_functions} solver_limit={seconds}s", placer.netlist().gates().count());
    let (outcome, stats) = placer.place(&config)?;
    report["wall_ms"] = (start.elapsed().as_millis() as u64).into();
    report["encode_ms"] = (stats.encode_time.as_millis() as u64).into();
    report["solve_ms"] = (stats.solve_time.as_millis() as u64).into();
    report["variables"] = stats.variables.into();
    report["clauses"] = stats.clauses.into();
    report["refinements"] = stats.refinements.into();
    match outcome {
        ExactOutcome::Unknown { last_rejection } => {
            report["status"] = "unknown".into();
            report["last_rejection"] = last_rejection.into();
        }
        ExactOutcome::Infeasible => report["status"] = "infeasible".into(),
        ExactOutcome::Placed(placement) => {
            let placed = &placement.placed;
            placed.world.to_nbt().save(directory.join("decoder.nbt"));
            std::fs::write(
                directory.join("interface.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"inputs":placed.inputs,"outputs":placed.outputs}),
                )?,
            )?;
            let world =
                NBTRoot::from_nbt_bytes(&std::fs::read(directory.join("decoder.nbt"))?)?.to_world();
            let verified = (|| -> eyre::Result<()> {
                ensure!(
                    placed.inputs.len() == 4 && placed.outputs.len() == 7,
                    "unexpected physical endpoints"
                );
                for value in 0..16 {
                    let mut sim =
                        Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
                    check(&mut sim, placed, value)?;
                }
                let mut sim =
                    Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
                for from in 0..16 {
                    for to in 0..16 {
                        check(&mut sim, placed, from)?;
                        check(&mut sim, placed, to)?;
                    }
                }
                Ok(())
            })();
            report["blocks"] = placed.world.iter_block().len().into();
            match verified {
                Ok(()) => {
                    report["status"] = "verified".into();
                    report["static_cases"] = 16.into();
                    report["ordered_transitions"] = 256.into();
                    report["nbt_roundtrip"] = true.into();
                }
                Err(err) => {
                    report["status"] = "verification-failed".into();
                    report["error"] = format!("{err:#}").into();
                }
            }
        }
    }
    std::fs::write(
        directory.join("attempt.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let mut concise = report.clone();
    concise.as_object_mut().unwrap().remove("net_functions");
    println!("RESULT {concise}");
    Ok(())
}
