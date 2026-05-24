use std::path::PathBuf;

use mimalloc::MiMalloc;
use redstone_compiler::verilog;
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

    if opt.input.extension().and_then(|ext| ext.to_str()) == Some("v") {
        let graph = verilog::load_logic_graph(&opt.input)?;
        let prepared = graph.prepare_place()?;
        println!(
            "loaded Verilog graph: nodes={} inputs={} outputs={}",
            prepared.nodes.len(),
            prepared.inputs().len(),
            prepared.outputs().len()
        );
        return Ok(());
    }

    eyre::bail!("unsupported input file extension: {:?}", opt.input)
}
