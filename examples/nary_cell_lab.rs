//! Export and verify the same n-ary recipes used by the opt-in production dispatcher.
//! cargo run --release --example nary_cell_lab
use std::path::Path;
use std::time::Instant;

use eyre::ContextCompat;
use redstone_compiler::graph::logic::LogicGraph;
use redstone_compiler::nbt::ToNBT;
use redstone_compiler::output::OutputEndpoint;
use redstone_compiler::transform::place_and_route::global_pnr::candidate::nary::build_verified_candidate;
use redstone_compiler::transform::place_and_route::global_pnr::ir::PhysicalPortDirection;
fn main() -> eyre::Result<()> {
    let directory = Path::new("target/nary-cell-lab");
    std::fs::create_dir_all(directory)?;
    let start = Instant::now();
    for count in 2..=8 {
        let names = (0..count).map(|i| format!("a{i}")).collect::<Vec<_>>();
        for (function, expression) in [
            ("or", names.join("|")),
            ("nor", format!("~({})", names.join("|"))),
            ("and", names.join("&")),
        ] {
            let candidate =
                build_verified_candidate(&LogicGraph::from_stmt(&expression, "y")?, "lab")?
                    .context("unrecognized signature")?;
            let stem = format!("{function}-{count}");
            std::fs::write(
                directory.join(format!("{stem}.nbt")),
                fastnbt::to_bytes(&candidate.world.to_nbt())?,
            )?;
            let endpoints = |direction| {
                candidate
                    .ports
                    .iter()
                    .filter(|p| p.direction == direction)
                    .map(|p| OutputEndpoint::new(p.name.clone(), p.position))
                    .collect::<Vec<_>>()
            };
            std::fs::write(
                directory.join(format!("{stem}.interface.json")),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"format":"redstone-compiler.outputs.v1","inputs":endpoints(PhysicalPortDirection::Input),"outputs":endpoints(PhysicalPortDirection::Output)}),
                )?,
            )?;
            println!(
                "PASS {function}{count}: static={} transitions={} blocks={} size={:?}",
                1 << count,
                68 + 3 * count,
                candidate.world.iter_block().len(),
                candidate.world.size
            );
        }
    }
    println!("completed in {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}
