//! Isolate shared input routing into 2, 4, or 10 four-input minterms.
//! cargo run --release --example minterm_fanout_probe -- target/alu-display/minterm-2 2
use std::path::PathBuf;
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

fn bit_index(name: &str, bus: &str) -> Option<usize> {
    name.strip_prefix(bus)?
        .trim_matches(|c| c == '_' || c == '[' || c == ']')
        .parse()
        .ok()
}

fn verify_static(placed: &redstone_compiler::output::PlacedWorld) -> eyre::Result<()> {
    use redstone_compiler::world::World;
    ensure!(placed.inputs.len() == 4, "expected 4 digit inputs");
    ensure!(
        [2, 4, 10].contains(&placed.outputs.len())
            || (std::env::var_os("MINTERM_OR").is_some() && placed.outputs.len() == 3),
        "unexpected minterm output count"
    );
    let mut seen = vec![false; placed.outputs.len()];
    for output in &placed.outputs {
        let index = bit_index(&output.name, "out").context("unknown output")?;
        ensure!(
            index < seen.len() && !seen[index],
            "invalid or duplicate minterm output"
        );
        seen[index] = true;
    }
    let world = World::from(&placed.world);
    for value in 0usize..16 {
        let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
        let inputs = placed
            .inputs
            .iter()
            .map(|p| {
                let bit = bit_index(&p.name, "digit").context("unknown input")?;
                ensure!(bit < 4, "invalid input");
                Ok((p.position(), value & (1 << bit) != 0))
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        sim.drive_inputs_with_limits(inputs, 4096, 2_000_000)?;
        for p in &placed.outputs {
            let index = bit_index(&p.name, "out").context("unknown output")?;
            ensure!(
                sim.world()[p.position()].kind.is_powered()
                    == (if std::env::var_os("MINTERM_OR").is_some() && index == 2 {
                        value < 2
                    } else {
                        value == index
                    }),
                "minterm mismatch value={value} output={}",
                p.name
            );
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
            .unwrap_or("target/alu-display/minterms".into()),
    );
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(dir.join("verification.json"));
    let count: usize = std::env::args().nth(2).unwrap_or("2".into()).parse()?;
    ensure!([2, 4, 10].contains(&count), "count must be 2, 4, or 10");
    let mut source = format!(
        "module minterms(digit, out);\ninput [3:0] digit;\noutput [{}:0] out;\n",
        count - 1
    );
    for index in 0..count {
        source.push_str(&format!(
            "wire d{index};\nassign d{index} = digit == 4'd{index};\n"
        ));
    }
    source.push_str(&format!(
        "assign out = {{{}}};\n",
        (0..count)
            .rev()
            .map(|i| format!("d{i}"))
            .collect::<Vec<_>>()
            .join(",")
    ));
    if std::env::var_os("MINTERM_OR").is_some() {
        ensure!(count == 2, "MINTERM_OR requires count 2");
        source = source
            .replace("output [1:0] out;", "output [2:0] out;")
            .replace("assign out = {d1,d0};", "assign out = {d0 | d1,d1,d0};");
    }
    source.push_str("endmodule\n");
    std::fs::write(dir.join("source.v"), &source)?;
    let logical = LogicalDesign::from_verilog_source_named(&source, "minterms.v")?;
    std::fs::write(dir.join("logical.rcir"), logical.to_string())?;
    let routable = logical.lower_to_routable_with_literal_associative_cells(8)?;
    std::fs::write(dir.join("routable.rcir"), routable.to_string())?;
    let mut config = GlobalPnrConfig {
        candidate: UnitCandidateConfig {
            verified_nary_cells: true,
            nary_stagger_inputs: std::env::var_os("NARY_STAGGER").is_some(),
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
    config.placement.max_attempts = 1;
    config.routing.max_route_steps = 256;
    config.routing.strategy = redstone_compiler::transform::place_and_route::global_pnr::router::GlobalRoutingStrategy::AStarBudgeted { max_expansions: 20_000 };
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
    config.search.policies.net_order_strategies = vec![redstone_compiler::transform::place_and_route::global_pnr::router::NetOrderStrategy::Criticality];
    if std::env::var_os("PIN_ACCESS").is_some() {
        config.search.policies.net_order_strategies = vec![redstone_compiler::transform::place_and_route::global_pnr::router::NetOrderStrategy::PinAccess];
    }
    config.search.budget.max_layout_combinations = 1;
    config.verifier = Some(verify_static);
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
    placed.world.to_nbt().save(dir.join("minterms.nbt"));
    placed.metadata().save(dir.join("minterms.outputs.json"))?;
    std::fs::write(
        dir.join("interface.json"),
        serde_json::to_string_pretty(
            &serde_json::json!({"inputs":placed.inputs,"outputs":placed.outputs}),
        )?,
    )?;
    let world = NBTRoot::from_nbt_bytes(&std::fs::read(dir.join("minterms.nbt"))?)?.to_world();
    let mut reloaded = placed.clone();
    reloaded.world = redstone_compiler::world::World3D::from(&world);
    verify_static(&reloaded)?;
    let summary = serde_json::json!({"outputs":placed.outputs.len(),"compile_ms":elapsed.as_millis(),"static_states":16,"nbt_roundtrip":true});
    std::fs::write(
        dir.join("verification.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
