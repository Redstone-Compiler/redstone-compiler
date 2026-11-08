//! End-to-end decoder experiment. Artifacts stay outside tracked fixtures.
//! cargo run --release --example decoder_nbt -- target/alu-display/decoder
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use eyre::{ensure, ContextCompat};
use redstone_compiler::ir::LogicalDesign;
use redstone_compiler::nbt::{NBTRoot, ToNBT};
use redstone_compiler::transform::place_and_route::global_pnr::candidate::UnitCandidateConfig;
use redstone_compiler::transform::place_and_route::global_pnr::{
    place_and_route_routable_design_with_visualization, GlobalPnrConfig,
};
use redstone_compiler::transform::place_and_route::local_placer::{
    LocalPlacerConfig, PlacementSchedulePolicy,
};
use redstone_compiler::transform::place_and_route::sampling::SamplingPolicy;
use redstone_compiler::world::position::DimSize;
use redstone_compiler::world::simulator::Simulator;
use tracing_subscriber::prelude::*;

const MASKS: [u8; 16] = [
    0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f, 0, 0, 0, 0, 0, 0,
];
static DIAGNOSTIC_DIR: OnceLock<PathBuf> = OnceLock::new();
static FAILURE_INDEX: AtomicUsize = AtomicUsize::new(0);

fn bit_index(name: &str, bus: &str) -> Option<usize> {
    name.strip_prefix(bus)?
        .trim_matches(|c| c == '_' || c == '[' || c == ']')
        .parse()
        .ok()
}

fn verify_static(placed: &redstone_compiler::output::PlacedWorld) -> eyre::Result<()> {
    use redstone_compiler::world::World;
    ensure!(
        (placed.inputs.len() == 4 || placed.inputs.len() == 5) && placed.outputs.len() == 7,
        "decoder interface mismatch"
    );
    let world = World::from(&placed.world);
    for value in 0usize..(1usize << placed.inputs.len()) {
        let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
        let inputs = placed
            .inputs
            .iter()
            .map(|input| {
                let enabled = if input.name == "blank" {
                    value & 16 != 0
                } else {
                    let bit = bit_index(&input.name, "digit").context("unknown input")?;
                    ensure!(bit < 4, "digit index outside decoder");
                    value & (1 << bit) != 0
                };
                Ok((input.position(), enabled))
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        sim.drive_inputs_with_limits(inputs, 4096, 2_000_000)?;
        let expected = if value & 16 != 0 {
            0
        } else {
            MASKS[value & 15]
        };
        for output in &placed.outputs {
            let bit = bit_index(&output.name, "seg").context("unknown output")?;
            if bit < 7
                && sim.world()[output.position()].kind.is_powered() != (expected & (1 << bit) != 0)
            {
                if let Some(dir) = DIAGNOSTIC_DIR.get() {
                    let dir = dir.join(format!(
                        "failed-layout-{}",
                        FAILURE_INDEX.fetch_add(1, Ordering::Relaxed)
                    ));
                    std::fs::create_dir_all(&dir)?;
                    placed.world.to_nbt().save(dir.join("decoder.nbt"));
                    placed.metadata().save(dir.join("decoder.outputs.json"))?;
                    std::fs::write(
                        dir.join("interface.json"),
                        serde_json::to_string_pretty(
                            &serde_json::json!({"inputs":placed.inputs,"outputs":placed.outputs}),
                        )?,
                    )?;
                    std::fs::write(
                        dir.join("failure.json"),
                        serde_json::to_string_pretty(
                            &serde_json::json!({"input":value,"output":output.name,"expected_mask":expected,"observed":placed.outputs.iter().map(|o|(&o.name,sim.world()[o.position()].kind.is_powered())).collect::<Vec<_>>()}),
                        )?,
                    )?;
                }
            }
            ensure!(
                bit < 7
                    && sim.world()[output.position()].kind.is_powered()
                        == (expected & (1 << bit) != 0),
                "static decoder mismatch input={value}, output={}",
                output.name
            );
        }
    }
    Ok(())
}

fn verify_candidate(placed: &redstone_compiler::output::PlacedWorld) -> eyre::Result<()> {
    verify_static(placed)?;
    let world = redstone_compiler::world::World::from(&placed.world);
    let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
    let states = 1usize << placed.inputs.len();
    // Reject a layout with transition hazards before selecting it. The exported
    // NBT is tested again independently after serialization below.
    for from in 0..states {
        for to in 0..states {
            for value in [from, to] {
                sim.advance_idle_cycles(61)?;
                let inputs = placed
                    .inputs
                    .iter()
                    .map(|input| {
                        let bit = if input.name == "blank" {
                            4
                        } else {
                            bit_index(&input.name, "digit").context("unknown input")?
                        };
                        Ok((input.position(), value & (1 << bit) != 0))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;
                sim.drive_inputs_with_limits(inputs, 4096, 2_000_000)?;
                let expected = if value & 16 != 0 {
                    0
                } else {
                    MASKS[value & 15]
                };
                let correct = placed.outputs.iter().all(|output| {
                    bit_index(&output.name, "seg").is_some_and(|bit| {
                        bit < 7
                            && sim.world()[output.position()].kind.is_powered()
                                == (expected & (1 << bit) != 0)
                    })
                });
                let burned = sim
                    .world()
                    .iter_block()
                    .iter()
                    .any(|(p, _)| sim.is_torch_burned_out(*p));
                if !correct || burned {
                    if let Some(dir) = DIAGNOSTIC_DIR.get() {
                        let dir = dir.join(format!(
                            "failed-layout-{}",
                            FAILURE_INDEX.fetch_add(1, Ordering::Relaxed)
                        ));
                        std::fs::create_dir_all(&dir)?;
                        placed.world.to_nbt().save(dir.join("decoder.nbt"));
                        std::fs::write(
                            dir.join("interface.json"),
                            serde_json::to_string_pretty(
                                &serde_json::json!({"inputs":placed.inputs,"outputs":placed.outputs}),
                            )?,
                        )?;
                        std::fs::write(
                            dir.join("failure.json"),
                            serde_json::to_string_pretty(
                                &serde_json::json!({"from":from,"to":to,"driving":value,"correct":correct,"burned_out":burned}),
                            )?,
                        )?;
                    }
                    eyre::bail!("decoder transition {from}->{to} while driving {value}: correct={correct}, burnout={burned}");
                }
            }
        }
    }
    Ok(())
}

fn main() -> eyre::Result<()> {
    let debug = std::env::var_os("PNR_DEBUG").is_some();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::DEBUG)
        .finish()
        .with(tracing_subscriber::filter::filter_fn(move |meta| {
            *meta.level() <= tracing::Level::INFO
                || (debug && meta.target() == "redstone_compiler::global_pnr")
        }));
    tracing::subscriber::set_global_default(subscriber).ok();
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("target/alu-display/decoder".into()),
    );
    std::fs::create_dir_all(&dir)?;
    let _ = DIAGNOSTIC_DIR.set(dir.join("diagnostics"));
    let _ = std::fs::remove_file(dir.join("verification.json"));
    let _ = std::fs::remove_file(dir.join("failure.json"));
    let source_path = std::env::args()
        .nth(2)
        .unwrap_or("test/alu_display/bcd_to_7seg.v".into());
    let mut source = std::fs::read_to_string(&source_path)?;
    for extra in std::env::args().skip(3) {
        source.push('\n');
        source.push_str(&std::fs::read_to_string(extra)?);
    }
    std::fs::write(dir.join("source.v"), &source)?;
    let logical = LogicalDesign::from_verilog_source_named(&source, "bcd_to_7seg.v")?;
    std::fs::write(dir.join("logical.rcir"), logical.to_string())?;
    let gates = std::env::var("BIT_GATES").unwrap_or("1".into()).parse()?;
    let boundary_inputs = std::env::var("BIT_INPUTS").unwrap_or("2".into()).parse()?;
    let routable = if std::env::var_os("BIT_LITERAL").is_some() {
        logical.lower_to_routable_with_literal_associative_cells(8)?
    } else if std::env::var_os("BIT_EXPANDED").is_some() {
        logical.lower_to_routable_with_expanded_associative_cells(8)?
    } else if std::env::var_os("BIT_NARY").is_some() {
        logical.lower_to_routable_with_associative_cells(8)?
    } else if std::env::var_os("BIT_BALANCED").is_some() {
        logical.lower_to_routable_with_balanced_associative_trees(gates, boundary_inputs)?
    } else {
        logical.lower_to_routable_with_bit_mapping_limits(gates, boundary_inputs)?
    };
    std::fs::write(dir.join("routable.rcir"), routable.to_string())?;
    let mut config = GlobalPnrConfig {
        candidate: UnitCandidateConfig {
            nary_stagger_inputs: std::env::var_os("NARY_STAGGER").is_some(),
            verified_nary_cells: std::env::var_os("BIT_NARY").is_some()
                || std::env::var_os("BIT_EXPANDED").is_some()
                || std::env::var_os("BIT_LITERAL").is_some(),
            reserve_combinational_io_diodes: true,
            dim: DimSize(8, 8, 4),
            max_candidates: 4,
            local_config: LocalPlacerConfig {
                greedy_input_generation: true,
                input_candidate_limit: Some(16),
                schedule: PlacementSchedulePolicy::Auto,
                placement_sampling_policy: LocalPlacerConfig::ranked_sampling(64, 0, 0),
                max_route_step: 4,
                route_step_sampling_policy: SamplingPolicy::Random(32),
                not_route_step_sampling_policy: SamplingPolicy::Random(8),
                ..Default::default()
            },
            ..Default::default()
        }
        .into(),
        candidate_cache_dir: Some(dir.join("candidate-cache")),
        ..Default::default()
    };
    config.search =
        redstone_compiler::transform::place_and_route::global_pnr::policy::GlobalPnrPreset::Fast
            .search_config();
    config.placement.max_attempts = 4;
    config.placement.spacing = std::env::var("PNR_SPACING").unwrap_or("4".into()).parse()?;
    if std::env::var_os("PNR_LAYERED").is_some() {
        use redstone_compiler::transform::place_and_route::global_pnr::policy::{
            LayerAssignmentStrategy, LayeredPlacementConfig, PlacementHeuristic,
        };
        config.search.policies.placement_heuristics =
            vec![PlacementHeuristic::Layered3D(LayeredPlacementConfig {
                layers: 2,
                layer_spacing: 6,
                assignment: LayerAssignmentStrategy::NetAware,
            })];
    }
    config.search.policies.net_order_strategies = vec![
        redstone_compiler::transform::place_and_route::global_pnr::router::NetOrderStrategy::Criticality,
        redstone_compiler::transform::place_and_route::global_pnr::router::NetOrderStrategy::HighestFanoutFirst,
        redstone_compiler::transform::place_and_route::global_pnr::router::NetOrderStrategy::ReverseCriticality,
    ];
    if std::env::var_os("PNR_PIN_ACCESS").is_some() {
        config.search.policies.net_order_strategies = vec![redstone_compiler::transform::place_and_route::global_pnr::router::NetOrderStrategy::PinAccess];
    }
    config.search.budget.max_layout_combinations = 1;
    config.verifier = Some(verify_candidate);
    if let Ok(value) = std::env::var("PNR_MAX_STEPS") {
        config.routing.max_route_steps = value.parse()?;
        ensure!(
            config.routing.max_route_steps > 0,
            "route step budget must be positive"
        );
    }
    if let Ok(value) = std::env::var("PNR_ASTAR_EXPANSIONS") {
        let max_expansions = value.parse()?;
        ensure!(max_expansions > 0, "A* expansion budget must be positive");
        config.routing.strategy = redstone_compiler::transform::place_and_route::global_pnr::router::GlobalRoutingStrategy::AStarBudgeted { max_expansions };
    }
    if std::env::var_os("PNR_GREEDY").is_some() {
        config.routing.strategy = redstone_compiler::transform::place_and_route::global_pnr::router::GlobalRoutingStrategy::GreedyBeam { beam_width: 16, max_expansions: 2000, variant_seed: 0 };
    }
    if std::env::var_os("PNR_SHELVES").is_some() {
        config.search.policies.placement_heuristics = vec![redstone_compiler::transform::place_and_route::global_pnr::policy::PlacementHeuristic::Shelf];
    }
    let start = Instant::now();
    let result = place_and_route_routable_design_with_visualization(&routable, &config)?;
    let elapsed = start.elapsed();
    std::fs::write(
        dir.join("pnr-config.json"),
        serde_json::to_string_pretty(&result.config_snapshot)?,
    )?;
    let placed = &result.placed_world;
    placed.world.to_nbt().save(dir.join("decoder.nbt"));
    placed.metadata().save(dir.join("decoder.outputs.json"))?;
    std::fs::write(
        dir.join("interface.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"inputs":placed.inputs,"outputs":placed.outputs}),
        )?,
    )?;
    // Read back the exported NBT, rather than testing only the pre-export world.
    let world = NBTRoot::from_nbt_bytes(&std::fs::read(dir.join("decoder.nbt"))?)?.to_world();
    ensure!(
        (placed.inputs.len() == 4 || placed.inputs.len() == 5) && placed.outputs.len() == 7,
        "unexpected interface"
    );
    let mut seen = [false; 7];
    for output in &placed.outputs {
        let bit = bit_index(&output.name, "seg").context("unknown segment endpoint")?;
        ensure!(bit < 7 && !seen[bit], "duplicate/invalid segment endpoint");
        seen[bit] = true;
    }
    let mut simulator = Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
    let mut cases = 0;
    // All 32 initial assignments, followed by all 1024 ordered transitions.
    let states = 1usize << placed.inputs.len();
    for from in 0..states {
        for to in 0..states {
            for value in [from, to] {
                simulator.advance_idle_cycles(61)?;
                let inputs = placed
                    .inputs
                    .iter()
                    .map(|input| {
                        let powered = if input.name == "blank" {
                            value & 16 != 0
                        } else {
                            let bit = bit_index(&input.name, "digit")
                                .context("unknown input endpoint")?;
                            ensure!(bit < 4, "invalid digit endpoint");
                            value & (1 << bit) != 0
                        };
                        Ok((input.position(), powered))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;
                simulator.drive_inputs_with_limits(inputs, 4096, 2_000_000)?;
                let expected = if value & 16 != 0 {
                    0
                } else {
                    MASKS[value & 15]
                };
                for output in &placed.outputs {
                    let bit = bit_index(&output.name, "seg").unwrap();
                    if simulator.world()[output.position()].kind.is_powered()
                        != (expected & (1 << bit) != 0)
                    {
                        std::fs::write(
                            dir.join("failure.json"),
                            serde_json::to_string_pretty(&serde_json::json!({
                                "from":from,"to":to,"driving":value,"driven_case":cases,
                                "output":output.name,"expected_mask":expected,
                                "observed":placed.outputs.iter().map(|o|(&o.name,simulator.world()[o.position()].kind.is_powered())).collect::<Vec<_>>(),
                                "burned_out":simulator.world().iter_block().iter().filter_map(|(p,_)|simulator.is_torch_burned_out(*p).then_some([p.0,p.1,p.2])).collect::<Vec<_>>()
                            }))?,
                        )?;
                    }
                    ensure!(
                        simulator.world()[output.position()].kind.is_powered()
                            == (expected & (1 << bit) != 0),
                        "decoder mismatch: input={value}, output={}",
                        output.name
                    );
                }
                ensure!(
                    !simulator
                        .world()
                        .iter_block()
                        .iter()
                        .any(|(pos, _)| simulator.is_torch_burned_out(*pos)),
                    "torch burnout"
                );
                cases += 1;
            }
        }
    }
    let summary = serde_json::json!({
        "compile_ms":elapsed.as_millis(),"blocks":world.blocks.len(),
        "static_cases":states,"driven_cases":cases,
        "ordered_transitions":states * states,"nbt_roundtrip":true,
        "blank_input":placed.inputs.iter().any(|p|p.name=="blank"),
        "invalid_bcd_off":true,"torch_burnout_checked":true
    });
    std::fs::write(
        dir.join("verification.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
