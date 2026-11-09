//! Whole decoder construction followed by separately saved/verified compaction.
//! Uses one flattened RTL graph; no cell mapper or global router.
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eyre::{ensure, ContextCompat};
use redstone_compiler::ir::LogicalDesign;
use redstone_compiler::nbt::{NBTRoot, ToNBT};
use redstone_compiler::output::PlacedWorld;
use redstone_compiler::transform::place_and_route::local_placer::exact::{
    CompactionConfig, ConstructionConfig, ExactLocalPlacer, GateOrder,
};
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

fn save_verify(directory: &Path, placed: &PlacedWorld) -> eyre::Result<serde_json::Value> {
    std::fs::create_dir_all(directory)?;
    placed.world.to_nbt().save(directory.join("decoder.nbt"));
    std::fs::write(
        directory.join("interface.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"inputs":placed.inputs,"outputs":placed.outputs}),
        )?,
    )?;
    let world = NBTRoot::from_nbt_bytes(&std::fs::read(directory.join("decoder.nbt"))?)?.to_world();
    ensure!(
        placed.inputs.len() == 4 && placed.outputs.len() == 7,
        "unexpected physical interface"
    );
    for value in 0..16 {
        let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
        check(&mut sim, placed, value)?;
    }
    let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
    for from in 0..16 {
        for to in 0..16 {
            check(&mut sim, placed, from)?;
            check(&mut sim, placed, to)?;
        }
    }
    let size = placed.world.size;
    let result = serde_json::json!({"status":"verified","static_cases":16,"ordered_transitions":256,"transition_drives":512,"nbt_roundtrip":true,"blocks":placed.world.iter_block().len(),"dim":[size.0,size.1,size.2]});
    std::fs::write(
        directory.join("verification.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}
fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .try_init()
        .ok();
    let directory = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("target/alu-display/whole-decoder-construct-4x6".into()),
    );
    std::fs::create_dir_all(&directory)?;
    ensure!(
        !directory.join("attempt.json").exists(),
        "use a fresh directory"
    );
    let source = format!(
        "{}\n{}",
        include_str!("../test/alu_display/bcd_to_7seg_minimized.v"),
        include_str!("../test/alu_display/bcd_digit_to_7seg.v")
    );
    let logical = LogicalDesign::from_verilog_source(&source)?;
    let graph = logical.lower_flat_logic_graph()?;
    let truth = graph.truth_table()?;
    ensure!(
        truth.input_names == ["digit_0", "digit_1", "digit_2", "digit_3"]
            && truth.output_tables.len() == 7,
        "unexpected graph interface"
    );
    for bit in 0..7 {
        for value in 0..16 {
            ensure!(
                truth.output_tables[&format!("seg_{bit}")][value]
                    == (MASKS[value] & (1 << bit) != 0),
                "logical oracle mismatch"
            );
        }
    }
    let original = ExactLocalPlacer::new(&graph)?;
    let max_fanin = std::env::var("CONSTRUCT_MAX_FANIN")
        .unwrap_or("0".into())
        .parse::<usize>()?;
    let placer = if max_fanin == 0 {
        original
    } else {
        ExactLocalPlacer::from_netlist(original.netlist().chain_wide_gates(max_fanin)?)
    };
    let net_values = placer.netlist().net_values();
    let input_names = placer.netlist().input_names();
    ensure!(
        input_names.len() == 4 && placer.netlist().outputs.len() == 7,
        "unexpected chained interface"
    );
    for mask in 0..16 {
        let mut digit = 0;
        for (i, name) in input_names.iter().enumerate() {
            if mask & (1 << i) != 0 {
                digit |= 1 << index(name, "digit")?;
            }
        }
        for (name, net) in &placer.netlist().outputs {
            ensure!(
                net_values[*net][mask] == (MASKS[digit] & (1 << index(name, "seg")?) != 0),
                "chained netlist oracle mismatch"
            );
        }
    }
    let nor_gates = placer
        .netlist()
        .nets
        .iter()
        .filter(|n| {
            matches!(
                n.driver,
                redstone_compiler::transform::place_and_route::local_placer::exact::NetDriver::Gate
            )
        })
        .count();
    let or_nets = placer
        .netlist()
        .nets
        .iter()
        .filter(|n| {
            matches!(
                n.driver,
                redstone_compiler::transform::place_and_route::local_placer::exact::NetDriver::Or
            )
        })
        .count();

    std::fs::write(directory.join("source.v"), source)?;
    std::fs::write(directory.join("logical.rcir"), logical.to_string())?;
    std::fs::write(directory.join("nor.txt"), placer.netlist().to_text())?;
    let mut config = ConstructionConfig {
        width: 4,
        height: 6,
        gate_order: GateOrder::MinLive,
        early_outputs: true,
        step_time_limit: Duration::from_secs(10),
        restart_after: Some(Duration::from_secs(120)),
        max_restarts: 0,
        workers: 1,
        seed: 1,
        step_optimize: None,
        progress: Some(directory.join("construction-progress")),
        ..Default::default()
    };
    let number = |name: &str, fallback: u64| -> eyre::Result<u64> {
        Ok(std::env::var(name)
            .ok()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(fallback))
    };
    config.max_restarts = number("CONSTRUCT_RESTARTS", 0)? as usize;
    config.overlap = number("CONSTRUCT_OVERLAP", 1)? as usize;
    config.max_overlap = config.overlap;
    config.width = number("CONSTRUCT_WIDTH", 4)? as usize;
    config.height = number("CONSTRUCT_HEIGHT", 6)? as usize;
    config.workers = number("CONSTRUCT_WORKERS", 1)? as usize;
    ensure!(
        config.width > 0 && config.height > 0 && config.workers > 0,
        "invalid dimensions/workers"
    );
    config.step_time_limit = Duration::from_secs(number("CONSTRUCT_STEP_SECONDS", 10)?);
    config.restart_after = Some(Duration::from_secs(number(
        "CONSTRUCT_BUDGET_SECONDS",
        120,
    )?));
    let optimize_ms = number("CONSTRUCT_STEP_OPT_MS", 0)?;
    config.step_optimize = (optimize_ms > 0).then(|| Duration::from_millis(optimize_ms));
    config.gate_order = match std::env::var("CONSTRUCT_ORDER")
        .as_deref()
        .unwrap_or("min-live")
    {
        "min-live" => GateOrder::MinLive,
        "smallest-cone" => GateOrder::SmallestConeFirst,
        "net-index" => GateOrder::NetIndex,
        value => eyre::bail!("unknown construction order {value}"),
    };
    let output_mode = std::env::var("CONSTRUCT_OUTPUT_MODE").unwrap_or("anywhere".into());
    ensure!(
        output_mode == "anywhere" || output_mode == "max-y",
        "unknown output mode"
    );
    if output_mode == "max-y" {
        for name in truth.output_tables.keys() {
            config.output_policies.insert(name.clone(), redstone_compiler::transform::place_and_route::local_placer::exact::OutputPolicy::MaxYFace);
        }
    }
    std::fs::write(
        directory.join("construction-config.txt"),
        format!("{config:#?}"),
    )?;
    let mut report = serde_json::json!({"mode":"whole-decoder-construct-compact","output_contract":if output_mode == "anywhere" {"anywhere-observable"} else {"max-y-driving"},"status":"constructing","logical_static_cases":16,"inputs":4,"outputs":7,"nor_gates":nor_gates,"or_nets":or_nets,"construction_steps":nor_gates+or_nets,"max_fanin":max_fanin,"netlist_static_cases":16,"overlap":config.overlap,"width":config.width,"height":config.height,"window":config.window,"max_window":config.max_window,"gate_order":format!("{:?}",config.gate_order),"early_outputs":true,"workers":config.workers,"seed":1,"step_seconds":config.step_time_limit.as_secs(),"construction_budget_seconds":config.restart_after.unwrap().as_secs(),"max_restarts":config.max_restarts,"restart_seed_stride":config.restart_seed_stride,"step_optimize_ms":optimize_ms});
    let write_report = |report: &serde_json::Value| -> eyre::Result<()> {
        std::fs::write(
            directory.join("attempt.json"),
            serde_json::to_vec_pretty(report)?,
        )?;
        Ok(())
    };
    write_report(&report)?;
    let started = Instant::now();
    let (layout, baseline, construction) = match placer.construct(&config) {
        Ok(value) => value,
        Err(error) => {
            report["status"] = "construction-failed".into();
            report["error"] = format!("{error:#}").into();
            report["construct_ms"] = (started.elapsed().as_millis() as u64).into();
            write_report(&report)?;
            println!("RESULT {report}");
            return Ok(());
        }
    };
    report["construct_ms"] = (started.elapsed().as_millis() as u64).into();
    std::fs::write(
        directory.join("construction-report.txt"),
        format!("{construction:#?}"),
    )?;
    match save_verify(&directory.join("constructed"), &baseline.placed) {
        Ok(result) => report["constructed"] = result,
        Err(error) => {
            report["status"] = "construction-verification-failed".into();
            report["error"] = format!("{error:#}").into();
            write_report(&report)?;
            return Ok(());
        }
    }
    report["status"] = "compacting".into();
    write_report(&report)?;
    let compact = CompactionConfig {
        output_policies: config.output_policies.clone(),
        time_limit: Some(Duration::from_secs(60)),
        attempt_time_limit: Duration::from_secs(5),
        workers: config.workers,
        seed: 1,
        progress: Some(directory.join("compaction-progress")),
        ..Default::default()
    };
    std::fs::write(
        directory.join("compaction-config.txt"),
        format!("{compact:#?}"),
    )?;
    let start = Instant::now();
    match placer.compact(layout, &compact) {
        Ok((_layout, best, details)) => {
            std::fs::write(
                directory.join("compaction-report.txt"),
                format!("{details:#?}"),
            )?;
            let selected = best.as_ref().unwrap_or(&baseline);
            match save_verify(&directory.join("compacted"), &selected.placed) {
                Ok(result) => {
                    report["compacted"] = result;
                    report["status"] = "verified".into();
                    report["compaction_produced_new_placement"] = best.is_some().into();
                }
                Err(error) => {
                    report["status"] = "compaction-verification-failed".into();
                    report["error"] = format!("{error:#}").into();
                }
            }
        }
        Err(error) => {
            report["status"] = "compaction-failed-baseline-preserved".into();
            report["error"] = format!("{error:#}").into();
        }
    }
    report["compact_ms"] = (start.elapsed().as_millis() as u64).into();
    write_report(&report)?;
    println!("RESULT {report}");
    Ok(())
}
