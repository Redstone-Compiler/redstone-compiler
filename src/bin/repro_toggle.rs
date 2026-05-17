use std::path::PathBuf;

use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::world::block::BlockKind;
use redstone_compiler::world::simulator::Simulator;

fn main() -> eyre::Result<()> {
    if std::env::var_os("REPRO_TOGGLE_TRACE").is_some() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .without_time()
            .init();
    }

    let paths = std::env::args()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    eyre::ensure!(!paths.is_empty(), "usage: repro_toggle <file.nbt>...");

    for path in paths {
        let bytes = std::fs::read(&path)?;
        let nbt = NBTRoot::from_nbt_bytes(&bytes)?;
        let world = nbt.to_world();
        let switches = world
            .blocks
            .iter()
            .filter_map(|(pos, block)| {
                let BlockKind::Switch { is_on } = block.kind else {
                    return None;
                };
                Some((*pos, is_on))
            })
            .collect::<Vec<_>>();

        println!("{}: {} switches", path.display(), switches.len());

        for (pos, is_on) in switches {
            let mut sim = match Simulator::from(&world) {
                Ok(sim) => sim,
                Err(error) => {
                    println!("  init fail before toggling {:?}: {error}", pos);
                    continue;
                }
            };
            let next = !is_on;
            match sim.change_state(vec![(pos, next)]) {
                Ok(()) => {
                    println!("  ok {:?}: {} -> {}", pos, is_on, next);
                }
                Err(error) => {
                    println!("  fail {:?}: {} -> {}: {error}", pos, is_on, next);
                }
            }
        }
    }

    Ok(())
}
