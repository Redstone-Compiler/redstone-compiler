use std::path::PathBuf;

use mimalloc::MiMalloc;
use redstone_compiler::ir::{CircuitIr, LogicalDesign};
use redstone_compiler::snapshot::{compile_with_snapshot, SnapshotOptions};
use redstone_compiler::transform::place_and_route::global_pnr::topology::ResolvedPnrTopology;
use redstone_compiler::transform::place_and_route::global_pnr::{
    emit_prepared_pnr_snapshot, load_prepared_pnr_snapshot,
    place_and_route_logical_design_with_visualization,
    place_and_route_routable_design_with_visualization, run_prepared_pnr_with_visualization,
    GlobalPnrConfig, PhysicalIntent, PnrPrepareConfig,
};
use structopt::StructOpt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
pub struct CompilerOption {
    #[structopt(parse(from_os_str))]
    pub input: PathBuf,

    #[structopt(parse(from_os_str))]
    pub output: Option<PathBuf>,

    /// Optional per-design floorplan and routing intent.
    #[structopt(long, parse(from_os_str))]
    pub intent: Option<PathBuf>,

    /// Reuse structurally identical local-placement candidates across runs.
    #[structopt(long, parse(from_os_str))]
    pub candidate_cache: Option<PathBuf>,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let opt = CompilerOption::from_args();

    match opt.input.extension().and_then(|ext| ext.to_str()) {
        Some("rcir") => compile_rcir_input(opt),
        Some("rsnap" | "snapshot") => replay_snapshot_input(opt),
        Some("v") => compile_verilog_input(opt),
        _ => eyre::bail!("unsupported input file extension: {:?}", opt.input),
    }
}

fn replay_snapshot_input(opt: CompilerOption) -> eyre::Result<()> {
    let mut base_config = GlobalPnrConfig::default();
    base_config.candidate_cache_dir = opt.candidate_cache.clone();
    let prepare_config = PnrPrepareConfig::from(&base_config);
    let Some(output) = opt.output else {
        let prepared = load_prepared_pnr_snapshot(&opt.input, &prepare_config)?;
        let (explicit_intent, _) =
            bind_physical_intent(opt.intent.as_deref(), prepared.topology())?;
        let active_intent = explicit_intent
            .as_ref()
            .or_else(|| prepared.snapshot_intent());
        println!(
            "loaded prepared PnR: module={} instances={} candidate_sets={} candidates={} constraints={}",
            prepared.module_name(),
            prepared.summary().instances,
            prepared.summary().unique_candidate_sets,
            prepared.summary().candidates,
            active_intent.map_or(0, |intent| intent.constraints.len()),
        );
        return Ok(());
    };

    let (snapshot_dir, snapshot_archive, options) =
        snapshot_options_without_source(&opt.input, &output);
    let prepared = load_prepared_pnr_snapshot(&opt.input, &prepare_config)?;
    let (physical_intent, intent_source) =
        bind_physical_intent(opt.intent.as_deref(), prepared.topology())?;
    let mut config = base_config;
    config.physical_intent = physical_intent.or_else(|| prepared.snapshot_intent().cloned());
    compile_with_snapshot(options, || {
        emit_intent_source(intent_source.as_ref())?;
        emit_prepared_pnr_snapshot(&prepared)?;
        run_prepared_pnr_with_visualization(&prepared, &config)
    })?;
    println!(
        "replayed global PnR without local placement: path={}",
        snapshot_dir.display()
    );
    println!(
        "exported snapshot archive: path={}",
        snapshot_archive.display()
    );
    Ok(())
}

fn compile_verilog_input(opt: CompilerOption) -> eyre::Result<()> {
    let source = std::fs::read_to_string(&opt.input)?;
    let logical = LogicalDesign::from_verilog_source(&source)?;
    let Some(output) = opt.output else {
        let cells = logical
            .modules
            .iter()
            .map(|module| module.cells.len())
            .sum::<usize>();
        let instances = logical
            .modules
            .iter()
            .map(|module| module.instances.len())
            .sum::<usize>();
        println!(
            "loaded Verilog as logical IR: top={} modules={} cells={} instances={}",
            logical.top,
            logical.modules.len(),
            cells,
            instances
        );
        return Ok(());
    };

    let (snapshot_dir, snapshot_archive, options) = snapshot_options(&opt.input, &output);
    let routable = logical.lower_to_routable()?;
    let topology = ResolvedPnrTopology::from_routable(&routable)?;
    let (physical_intent, intent_source) = bind_physical_intent(opt.intent.as_deref(), &topology)?;
    let mut config = GlobalPnrConfig::default();
    config.physical_intent = physical_intent;
    config.candidate_cache_dir = opt.candidate_cache.clone();
    compile_with_snapshot(options, || {
        emit_intent_source(intent_source.as_ref())?;
        place_and_route_logical_design_with_visualization(&logical, &config)
    })?;

    println!("exported Verilog snapshot: path={}", snapshot_dir.display());
    println!(
        "exported snapshot archive: path={}",
        snapshot_archive.display()
    );

    Ok(())
}

fn compile_rcir_input(opt: CompilerOption) -> eyre::Result<()> {
    let source = std::fs::read_to_string(&opt.input)?;
    let ir: CircuitIr = source.parse()?;
    let Some(output) = opt.output else {
        match &ir {
            CircuitIr::Logical(design) => println!(
                "loaded logical IR: top={} modules={}",
                design.top,
                design.modules.len()
            ),
            CircuitIr::Routable(design) => println!(
                "loaded routable IR: top={} modules={} target={}",
                design.top,
                design.modules.len(),
                design.target
            ),
        }
        return Ok(());
    };

    let (snapshot_dir, snapshot_archive, options) = snapshot_options(&opt.input, &output);
    let routable = match &ir {
        CircuitIr::Logical(design) => design.lower_to_routable()?,
        CircuitIr::Routable(design) => design.clone(),
    };
    let topology = ResolvedPnrTopology::from_routable(&routable)?;
    let (physical_intent, intent_source) = bind_physical_intent(opt.intent.as_deref(), &topology)?;
    let mut config = GlobalPnrConfig::default();
    config.physical_intent = physical_intent;
    config.candidate_cache_dir = opt.candidate_cache.clone();
    match &ir {
        CircuitIr::Logical(design) => compile_with_snapshot(options, || {
            emit_intent_source(intent_source.as_ref())?;
            place_and_route_logical_design_with_visualization(design, &config)
        })?,
        CircuitIr::Routable(design) => compile_with_snapshot(options, || {
            emit_intent_source(intent_source.as_ref())?;
            place_and_route_routable_design_with_visualization(design, &config)
        })?,
    };

    println!("exported rcir snapshot: path={}", snapshot_dir.display());
    println!(
        "exported snapshot archive: path={}",
        snapshot_archive.display()
    );
    Ok(())
}

fn snapshot_options(
    input: &std::path::Path,
    output: &std::path::Path,
) -> (PathBuf, PathBuf, SnapshotOptions) {
    let snapshot_dir = snapshot_output_dir(output);
    let snapshot_archive = snapshot_dir.with_extension("rsnap");
    let design_name = output
        .file_stem()
        .or_else(|| input.file_stem())
        .and_then(|name| name.to_str())
        .unwrap_or("design")
        .to_owned();
    let options = SnapshotOptions::new(&snapshot_dir, design_name).with_source(input);
    (snapshot_dir, snapshot_archive, options)
}

fn bind_physical_intent(
    path: Option<&std::path::Path>,
    topology: &ResolvedPnrTopology,
) -> eyre::Result<(
    Option<redstone_compiler::transform::place_and_route::global_pnr::ResolvedPhysicalIntent>,
    Option<(String, String)>,
)> {
    let Some(path) = path else {
        return Ok((None, None));
    };
    let source = std::fs::read_to_string(path)?;
    let intent: PhysicalIntent = source.parse()?;
    let resolved = intent.bind(topology)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("design.rclayout")
        .to_owned();
    Ok((Some(resolved), Some((file_name, source))))
}

fn emit_intent_source(source: Option<&(String, String)>) -> eyre::Result<()> {
    if let Some((file_name, source)) = source {
        redstone_compiler::snapshot::emit_text(format!("intent/{file_name}"), source.clone())?;
    }
    Ok(())
}

fn snapshot_options_without_source(
    input: &std::path::Path,
    output: &std::path::Path,
) -> (PathBuf, PathBuf, SnapshotOptions) {
    let snapshot_dir = snapshot_output_dir(output);
    let snapshot_archive = snapshot_dir.with_extension("rsnap");
    let design_name = output
        .file_stem()
        .or_else(|| input.file_stem())
        .and_then(|name| name.to_str())
        .unwrap_or("design")
        .to_owned();
    let options = SnapshotOptions::new(&snapshot_dir, design_name);
    (snapshot_dir, snapshot_archive, options)
}

fn snapshot_output_dir(output: &std::path::Path) -> PathBuf {
    if output.extension().and_then(|extension| extension.to_str()) == Some("snapshot") {
        output.to_owned()
    } else {
        output.with_extension("snapshot")
    }
}
