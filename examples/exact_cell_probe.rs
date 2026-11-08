//! Bounded direct exact synthesis versus verified literal recipes.
//! cargo run --release --example exact_cell_probe -- 5
use std::time::{Duration, Instant};

use redstone_compiler::graph::logic::LogicGraph;
use redstone_compiler::transform::place_and_route::global_pnr::candidate::nary::build_verified_candidate;
use redstone_compiler::transform::place_and_route::local_placer::exact::{
    ExactLocalPlacer, ExactOutcome, ExactPlacerConfig, NetDriver,
};
use redstone_compiler::world::position::DimSize;
fn main() -> eyre::Result<()> {
    let seconds = std::env::args()
        .nth(1)
        .unwrap_or("5".into())
        .parse::<u64>()?;
    for (name, expression) in [
        ("or2", "a|b"),
        ("mixed_or2", "a|~b"),
        ("or3", "a|b|c"),
        ("mixed_or3", "a|~b|c"),
        ("or4", "a|b|c|d"),
        ("mixed_and4", "~(a|~b|c|~d)"),
    ] {
        // Keep OR chains: exact folds them into NOR inputs. prepare_place()
        // inserts OR-to-OR buffers intended for the route-based placer.
        let graph = LogicGraph::from_stmt(expression, "y")?;
        let recipe = build_verified_candidate(&graph, name)?.unwrap();
        let placer = ExactLocalPlacer::new(&graph)?;
        let gates = placer
            .netlist()
            .nets
            .iter()
            .filter(|n| n.driver == NetDriver::Gate)
            .count();
        let mut config = ExactPlacerConfig::new(DimSize(3, 5, 3));
        config.time_limit = Some(Duration::from_secs(seconds));
        config.driving_outputs.insert("y".into());
        config.workers = 1;
        config.optimize = std::env::var_os("EXACT_OPTIMIZE").is_some();
        let start = Instant::now();
        let (outcome, stats) = placer.place(&config)?;
        let result = match outcome {
            ExactOutcome::Placed(p) => format!(
                "placed logic_blocks_excluding_switches={} world_blocks={} dim={:?}",
                p.block_count,
                p.placed.world.iter_block().len(),
                p.placed.world.size
            ),
            ExactOutcome::Infeasible => "infeasible".into(),
            ExactOutcome::Unknown { last_rejection } => {
                format!("unknown rejection={last_rejection:?}")
            }
        };
        println!("{name}: NOR={gates} recipe_blocks={} recipe_dim={:?} exact={result} wall={:.3}s encode={:.3}s solve={:.3}s clauses={}", recipe.world.iter_block().len(), recipe.world.size, start.elapsed().as_secs_f64(), stats.encode_time.as_secs_f64(), stats.solve_time.as_secs_f64(), stats.clauses);
    }
    Ok(())
}
