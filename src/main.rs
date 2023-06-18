pub mod circuit;
pub mod common;
pub mod estimator;
pub mod graph;
pub mod logic;
pub mod option;
pub mod simulator;
pub mod verilog;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
pub struct CompilerOption {
    #[structopt(parse(from_os_str))]
    pub input: PathBuf,

    #[structopt(parse(from_os_str))]
    pub output: Option<PathBuf>,
}

impl CompilerOption {
    pub fn run(self) -> eyre::Result<()> {
        Ok(())
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let opt = CompilerOption::from_args();
    // println!("{opt:?}");
}
