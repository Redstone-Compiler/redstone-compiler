use std::path::{Path, PathBuf};
use std::time::SystemTime;

use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::physical_cell::PhysicalCellDocument;
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
    NBTRoot::from(&build.world).save(&output);

    if options.emit {
        print!("{document}");
    }
    if !options.no_verify {
        let verification = document.verify(&build)?;
        if !verification.failures.is_empty() {
            for failure in &verification.failures {
                eprintln!(
                    "FAIL inputs={:?} expected={:?} actual={:?}",
                    failure.inputs, failure.expected, failure.actual
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
    println!(
        "exported {} blocks ({} automatic supports) to {}",
        build.world.iter_block().len(),
        build.auto_supports.len(),
        output.display()
    );
    Ok(())
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
