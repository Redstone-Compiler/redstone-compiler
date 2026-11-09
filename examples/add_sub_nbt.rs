//! Bounded Verilog -> NBT arithmetic experiment, independent integer oracle.
//! cargo run --release --example add_sub_nbt -- 2
//! Widths 1/2/4 are exhaustive statically; width 8 is explicitly sampled.
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use eyre::{ensure, ContextCompat, WrapErr};
use redstone_compiler::ir::LogicalDesign;
use redstone_compiler::nbt::{NBTRoot, ToNBT};
use redstone_compiler::output::{OutputEndpoint, PlacedWorld};
use redstone_compiler::transform::place_and_route::global_pnr::candidate::UnitCandidateConfig;
use redstone_compiler::transform::place_and_route::global_pnr::policy::GlobalPnrPreset;
use redstone_compiler::transform::place_and_route::global_pnr::router::{
    GlobalRoutingStrategy, NetOrderStrategy,
};
use redstone_compiler::transform::place_and_route::global_pnr::{
    place_and_route_routable_design_with_visualization, GlobalPnrConfig,
};
use redstone_compiler::transform::place_and_route::local_placer::{
    LocalPlacerConfig, PlacementSchedulePolicy,
};
use redstone_compiler::transform::place_and_route::sampling::SamplingPolicy;
use redstone_compiler::world::position::DimSize;
use redstone_compiler::world::simulator::Simulator;
use redstone_compiler::world::{World, World3D};
use tracing_subscriber::prelude::*;

const MAX_CYCLES: usize = 4096;
const MAX_EVENTS: usize = 2_000_000;
const SEED: u64 = 0xadd8_2026_1108_0001;

fn bit_index(name: &str, bus: &str, width: usize) -> Option<usize> {
    if width == 1 && name == bus {
        return Some(0);
    }
    name.strip_prefix(bus)?
        .trim_matches(|c| c == '_' || c == '[' || c == ']')
        .parse()
        .ok()
}

fn directed_cases(width: usize) -> Vec<(usize, usize)> {
    let mask = (1 << width) - 1;
    let mut cases = vec![
        (0, 0),
        (mask, 0),
        (0, mask),
        (mask, 1),
        (1, mask),
        (mask, mask),
        (mask >> 1, 1),
        (0x55 & mask, 0xaa & mask),
        (0xaa & mask, 0x55 & mask),
    ];
    for bit in 1..=width {
        let carry = (1 << bit) - 1;
        cases.extend([(carry, 1), (1, carry), (carry, mask ^ carry)]);
    }
    cases
}

fn operand_cases(width: usize) -> Vec<(usize, usize)> {
    if width <= 4 {
        return (0..1 << width)
            .flat_map(|a| (0..1 << width).map(move |b| (a, b)))
            .collect();
    }
    let mut cases = directed_cases(width);
    let mut state = SEED;
    for _ in 0..256 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        cases.push(((state & 255) as usize, ((state >> 8) & 255) as usize));
    }
    cases
}

fn static_cases(width: usize) -> Vec<(usize, usize, bool)> {
    operand_cases(width)
        .into_iter()
        .flat_map(|(a, b)| [(a, b, false), (a, b, true)])
        .collect()
}

fn validate_interface(placed: &PlacedWorld, width: usize) -> eyre::Result<()> {
    ensure!(
        placed.inputs.len() == 2 * width + 1 && placed.outputs.len() == width + 1,
        "add/sub interface width mismatch"
    );
    let mut inputs = HashSet::new();
    for input in &placed.inputs {
        let endpoint = if input.name == "subtract" {
            ("subtract", 0)
        } else {
            ["a", "b"]
                .into_iter()
                .find_map(|bus| bit_index(&input.name, bus, width).map(|bit| (bus, bit)))
                .context("unknown operand input")?
        };
        ensure!(
            (endpoint.0 == "subtract" || endpoint.1 < width) && inputs.insert(endpoint),
            "duplicate or invalid input"
        );
    }
    let mut outputs = HashSet::new();
    for output in &placed.outputs {
        let bit = if output.name == "c" {
            width
        } else {
            bit_index(&output.name, "y", width).context("unknown result output")?
        };
        ensure!(
            bit <= width && outputs.insert(bit),
            "duplicate or invalid output"
        );
    }
    ensure!(
        placed.outputs.iter().any(|output| output.name == "c"),
        "missing carry/no-borrow output"
    );
    Ok(())
}

fn drive_and_check(
    sim: &mut Simulator,
    placed: &PlacedWorld,
    width: usize,
    a: usize,
    b: usize,
    subtract: bool,
) -> eyre::Result<()> {
    let inputs = placed
        .inputs
        .iter()
        .map(|input| {
            let value = if input.name == "subtract" {
                subtract
            } else if let Some(bit) = bit_index(&input.name, "a", width) {
                a & (1 << bit) != 0
            } else {
                let bit = bit_index(&input.name, "b", width).context("unknown operand input")?;
                b & (1 << bit) != 0
            };
            Ok((input.position(), value))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    sim.drive_inputs_with_limits(inputs, MAX_CYCLES, MAX_EVENTS)?;
    let mask = (1usize << width) - 1;
    let expected = if subtract {
        a.wrapping_sub(b) & mask
    } else {
        (a + b) & mask
    };
    let carry = if subtract { a >= b } else { a + b > mask };
    for output in &placed.outputs {
        let expected_bit = if output.name == "c" {
            carry
        } else {
            let bit = bit_index(&output.name, "y", width).context("unknown result output")?;
            expected & (1 << bit) != 0
        };
        ensure!(sim.world()[output.position()].kind.is_powered()==expected_bit,"add/sub {width}-bit mismatch: a={a}, b={b}, subtract={subtract}, result={expected}, carry/no-borrow={carry}, output={}",output.name);
    }
    let burned = sim
        .world()
        .iter_block()
        .iter()
        .filter_map(|(pos, _)| sim.is_torch_burned_out(*pos).then_some(*pos))
        .collect::<Vec<_>>();
    ensure!(burned.is_empty(), "add/sub torch burnout at {burned:?}");
    Ok(())
}

fn verify_static_world(
    world: &World,
    placed: &PlacedWorld,
    width: usize,
    cases: &[(usize, usize, bool)],
) -> eyre::Result<()> {
    for &(a, b, subtract) in cases {
        let mut sim =
            Simulator::from_settled_with_limits_and_trace(world, MAX_CYCLES, MAX_EVENTS, 0)?;
        drive_and_check(&mut sim, placed, width, a, b, subtract)
            .wrap_err_with(|| format!("static case a={a}, b={b}, subtract={subtract}"))?;
    }
    Ok(())
}

fn candidate_verifier(placed: &PlacedWorld) -> eyre::Result<()> {
    let width = (placed.inputs.len() - 1) / 2;
    ensure!(
        [1, 2, 4, 8].contains(&width),
        "unsupported add/sub interface"
    );
    validate_interface(placed, width)?;
    // Reject bad assembled layouts inside PnR so it can try another layout.
    // NBT roundtrip and sequential-transition checks remain independent.
    let world = World::from(&placed.world);
    let cases = static_cases(width);
    verify_static_world(&world, placed, width, &cases)?;
    let mut sim = Simulator::from_settled_with_limits_and_trace(&world, MAX_CYCLES, MAX_EVENTS, 0)?;
    if width <= 2 {
        for &(a, b, subtract) in &cases {
            for &(next_a, next_b, next_subtract) in &cases {
                sim.advance_idle_cycles(61)?;
                drive_and_check(&mut sim, placed, width, a, b, subtract)?;
                sim.advance_idle_cycles(61)?;
                drive_and_check(&mut sim,placed,width,next_a,next_b,next_subtract).wrap_err_with(||format!("assembled ordered transition ({a},{b},{subtract})->({next_a},{next_b},{next_subtract})"))?;
            }
        }
    } else {
        for (index, &(a, b, subtract)) in cases.iter().enumerate() {
            sim.advance_idle_cycles(61)?;
            drive_and_check(&mut sim, placed, width, a, b, subtract)
                .wrap_err_with(|| format!("assembled continuous walk case {}", index + 1))?;
        }
    }
    Ok(())
}

fn run(width: usize, dir: &Path) -> eyre::Result<()> {
    let started = Instant::now();
    std::fs::write(
        dir.join("status.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"stage":"compile","bits":width}))?,
    )?;
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("test/alu_display/add_sub{width}.v")),
    )?;
    std::fs::write(dir.join("add_sub.v"), &source)?;
    let logical = LogicalDesign::from_verilog_source_named(&source, "add_sub.v")?;
    std::fs::write(dir.join("logical.rcir"), logical.to_string())?;
    let routable = logical.lower_to_routable_with_associative_cells(8)?;
    std::fs::write(dir.join("routable.rcir"), routable.to_string())?;
    let dim_text = std::env::var("ADD_SUB_DIM").unwrap_or("10x10x5".into());
    let dims = dim_text
        .split('x')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    ensure!(
        dims.len() == 3 && dims.iter().all(|dim| *dim > 0),
        "ADD_SUB_DIM must be positive XxYxZ"
    );
    let beam = std::env::var("ADD_SUB_BEAM")
        .unwrap_or("256".into())
        .parse::<usize>()?;
    ensure!(beam > 0, "ADD_SUB_BEAM must be positive");
    let cache_dir = std::env::var_os("ADD_SUB_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("candidate-cache"));
    let cache_entries_before = std::fs::read_dir(&cache_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.path().join("index.json").is_file())
                .count()
        })
        .unwrap_or(0);
    let mut config = GlobalPnrConfig {
        candidate: UnitCandidateConfig {
            verified_nary_cells: true,
            reserve_combinational_io_diodes: true,
            dim: DimSize(dims[0], dims[1], dims[2]),
            max_candidates: 4,
            local_config: LocalPlacerConfig {
                greedy_input_generation: true,
                input_candidate_limit: Some(16),
                schedule: PlacementSchedulePolicy::Auto,
                placement_sampling_policy: LocalPlacerConfig::ranked_sampling(beam, 0, 0),
                max_route_step: 4,
                route_step_sampling_policy: SamplingPolicy::Random(32),
                not_route_step_sampling_policy: SamplingPolicy::Random(8),
                ..Default::default()
            },
            ..Default::default()
        }
        .into(),
        candidate_cache_dir: Some(cache_dir.clone()),
        ..Default::default()
    };
    config.search = GlobalPnrPreset::Fast.search_config();
    config.search.budget.max_layout_combinations = 1;
    config.search.policies.net_order_strategies = vec![
        NetOrderStrategy::Criticality,
        NetOrderStrategy::HighestFanoutFirst,
        NetOrderStrategy::ReverseCriticality,
    ];
    config.placement.max_attempts = 4;
    config.placement.spacing = 4;
    config.routing.strategy = GlobalRoutingStrategy::AStarBudgeted {
        max_expansions: 20_000,
    };
    config.verifier = Some(candidate_verifier);
    std::fs::write(
        dir.join("experiment.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"cache_dir":cache_dir,"cache_entries_before":cache_entries_before,"candidate_dim":dims,"beam":beam,"bits":width,"operation":"shared_add_sub_with_no_borrow","candidate_verifier":"static_and_exhaustive_ordered_up_to_2bit_else_bounded_walk","mapping":"associative_cells_8","verified_nary_cells":true,"reserve_io_diodes":true,"astar_expansions":20000,"placement_attempts":4,"layout_combinations":1,"preset":"fast","sim_max_cycles":MAX_CYCLES,"sim_max_events":MAX_EVENTS,"seed":SEED,"random_operand_pairs":if width==8 {256}else{0},"random_cases":if width==8 {512}else{0}}),
        )?,
    )?;
    let result = place_and_route_routable_design_with_visualization(&routable, &config)?;
    let compile_ms = started.elapsed().as_millis();
    let placed = &result.placed_world;
    validate_interface(placed, width)?;
    placed.world.to_nbt().save(dir.join("add_sub.nbt"));
    placed.metadata().save(dir.join("add_sub.outputs.json"))?;
    std::fs::write(
        dir.join("interface.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"inputs":placed.inputs,"outputs":placed.outputs}),
        )?,
    )?;
    verify_saved(width, dir, Some(compile_ms), false, None, started)
}

fn fingerprint(bytes: &[u8]) -> String {
    let value = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    format!("fnv1a64:{value:016x}")
}

#[derive(serde::Deserialize)]
struct Interface {
    inputs: Vec<OutputEndpoint>,
    outputs: Vec<OutputEndpoint>,
}

fn archive_evidence(dir: &Path) -> eyre::Result<Option<String>> {
    let names = ["verification.json", "failure.json", "status.json"];
    if !names.iter().any(|name| dir.join(name).is_file()) {
        return Ok(None);
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let relative = format!("diagnostics/prior-{stamp}");
    let archive = dir.join(&relative);
    std::fs::create_dir_all(&archive)?;
    for name in names {
        let path = dir.join(name);
        if path.is_file() {
            std::fs::copy(&path, archive.join(name))?;
        }
    }
    Ok(Some(relative))
}

fn verify_saved(
    width: usize,
    dir: &Path,
    compile_ms: Option<u128>,
    recheck: bool,
    prior_evidence: Option<String>,
    started: Instant,
) -> eyre::Result<()> {
    let verification_started = Instant::now();
    let nbt_bytes = std::fs::read(dir.join("add_sub.nbt"))?;
    let interface_bytes = std::fs::read(dir.join("interface.json"))?;
    let world = NBTRoot::from_nbt_bytes(&nbt_bytes)?.to_world();
    let interface: Interface = serde_json::from_slice(&interface_bytes)?;
    let placed = PlacedWorld {
        world: World3D::from(&world),
        inputs: interface.inputs,
        outputs: interface.outputs,
    };
    validate_interface(&placed, width)?;
    std::fs::write(
        dir.join("status.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"stage":"verify_nbt","bits":width,"operation":"shared_add_sub_with_no_borrow","compile_ms":compile_ms}),
        )?,
    )?;
    let cases = static_cases(width);
    verify_static_world(&world, &placed, width, &cases)?;
    let mut sim = Simulator::from_settled_with_limits_and_trace(&world, MAX_CYCLES, MAX_EVENTS, 0)?;
    std::fs::write(
        dir.join("status.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"stage":"verify_transitions","bits":width,"static_cases_passed":cases.len(),"compile_ms":compile_ms}),
        )?,
    )?;
    let mut transitions = 0;
    if width <= 2 {
        for &(a, b, subtract) in &cases {
            for &(next_a, next_b, next_subtract) in &cases {
                sim.advance_idle_cycles(61)?;
                drive_and_check(&mut sim, &placed, width, a, b, subtract)?;
                sim.advance_idle_cycles(61)?;
                drive_and_check(&mut sim, &placed, width, next_a, next_b, next_subtract).wrap_err_with(|| {
                    format!("ordered transition ({a},{b},{subtract})->({next_a},{next_b},{next_subtract})")
                })?;
                transitions += 1;
            }
        }
    } else {
        for &(a, b, subtract) in &cases {
            sim.advance_idle_cycles(61)?;
            drive_and_check(&mut sim, &placed, width, a, b, subtract).wrap_err_with(|| {
                format!(
                    "continuous walk case {}: a={a}, b={b}, subtract={subtract}",
                    transitions + 1
                )
            })?;
            transitions += 1;
        }
    }
    ensure!(
        std::fs::read(dir.join("add_sub.nbt"))? == nbt_bytes,
        "NBT changed during verification"
    );
    let summary = serde_json::json!({"mode":if recheck {"verify-only"}else{"compile-and-verify"},"verification_after_simfix":recheck,"simulator_normalization":"fixpoint-on-off-notification","prior_evidence":prior_evidence,"nbt_fingerprint":fingerprint(&nbt_bytes),"interface_fingerprint":fingerprint(&interface_bytes),"nbt_unchanged":true,"verification_ms":verification_started.elapsed().as_millis(),"bits":width,"operation":"shared_add_sub_with_no_borrow","compile_ms":compile_ms,"total_ms":started.elapsed().as_millis(),"blocks":world.blocks.len(),"nbt_roundtrip":true,"static_cases":cases.len(),"static_exhaustive":width<=4,"ordered_transitions_exhaustive":width<=2,"ordered_transition_pairs":if width<=2 {transitions}else{0},"continuous_walk_cases":if width>2 {transitions}else{0},"random_seed":if width==8 {Some(SEED)}else{None},"random_operand_pairs":if width==8 {256}else{0},"random_cases":if width==8 {512}else{0},"eight_bit_scope":"directed carry/borrow boundaries plus 256 seeded random operand pairs, both operations; not exhaustive proof"});
    std::fs::write(
        dir.join("verification.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    std::fs::write(dir.join("status.json"), b"{\"stage\":\"verified\"}")?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
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
    let mut args = std::env::args().skip(1);
    let first = args.next().unwrap_or("2".into());
    let recheck = first == "--verify-only";
    let width = if recheck {
        args.next().context("--verify-only requires width")?
    } else {
        first
    }
    .parse::<usize>()?;
    ensure!(
        [1, 2, 4, 8].contains(&width),
        "expected width 1, 2, 4, or 8"
    );
    let dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("target/alu-display/add-sub-{width}")));
    ensure!(args.next().is_none(), "unexpected extra argument");
    if recheck {
        ensure!(
            dir.join("add_sub.nbt").is_file() && dir.join("interface.json").is_file(),
            "verify-only requires saved add_sub.nbt and interface.json"
        );
    } else {
        std::fs::create_dir_all(&dir)?;
    }
    let compile_ms = ["verification.json", "status.json"]
        .into_iter()
        .find_map(|name| {
            std::fs::read(dir.join(name))
                .ok()
                .and_then(|data| serde_json::from_slice::<serde_json::Value>(&data).ok())
                .and_then(|value| {
                    value
                        .get("compile_ms")
                        .and_then(|value| value.as_u64())
                        .map(u128::from)
                })
        });
    let prior_evidence = archive_evidence(&dir)?;
    for file in ["verification.json", "failure.json"] {
        let path = dir.join(file);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }
    let started = Instant::now();
    let result = if recheck {
        verify_saved(
            width,
            &dir,
            compile_ms,
            true,
            prior_evidence.clone(),
            started,
        )
    } else {
        run(width, &dir)
    };
    if let Err(error) = result {
        std::fs::write(
            dir.join("failure.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"bits":width,"mode":if recheck {"verify-only"}else{"compile-and-verify"},"prior_evidence":prior_evidence,"elapsed_ms":started.elapsed().as_millis(),"error":format!("{error:?}")}),
            )?,
        )?;
        return Err(error);
    }
    Ok(())
}
