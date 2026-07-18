use std::path::PathBuf;

use mimalloc::MiMalloc;
use redstone_compiler::ir::{CircuitIr, LogicalDesign};
use redstone_compiler::snapshot::{compile_with_snapshot, SnapshotOptions};
use redstone_compiler::transform::place_and_route::global_pnr::{
    place_and_route_logical_design_with_visualization,
    place_and_route_routable_design_with_visualization, GlobalPnrConfig,
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
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let opt = CompilerOption::from_args();

    match opt.input.extension().and_then(|ext| ext.to_str()) {
        Some("rcir") => compile_rcir_input(opt),
        Some("v") => compile_verilog_input(opt),
        _ => eyre::bail!("unsupported input file extension: {:?}", opt.input),
    }
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
    compile_with_snapshot(options, || {
        place_and_route_logical_design_with_visualization(&logical, &GlobalPnrConfig::default())
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
    match &ir {
        CircuitIr::Logical(design) => compile_with_snapshot(options, || {
            place_and_route_logical_design_with_visualization(design, &GlobalPnrConfig::default())
        })?,
        CircuitIr::Routable(design) => compile_with_snapshot(options, || {
            place_and_route_routable_design_with_visualization(design, &GlobalPnrConfig::default())
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

fn snapshot_output_dir(output: &std::path::Path) -> PathBuf {
    if output.extension().and_then(|extension| extension.to_str()) == Some("snapshot") {
        output.to_owned()
    } else {
        output.with_extension("snapshot")
    }
}
