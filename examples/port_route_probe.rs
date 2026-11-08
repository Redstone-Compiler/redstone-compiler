//! Minimal global input-routing probe; no NBT editing or export.
//! cargo run --release --example port_route_probe -- single
//! cargo run --release --example port_route_probe -- fanout
use std::path::PathBuf;
use std::time::Instant;

use redstone_compiler::ir::LogicalDesign;
use redstone_compiler::transform::place_and_route::global_pnr::candidate::UnitCandidateConfig;
use redstone_compiler::transform::place_and_route::global_pnr::{
    place_and_route_routable_design_with_visualization, GlobalPnrConfig,
};
use redstone_compiler::transform::place_and_route::local_placer::{
    LocalPlacerConfig, PlacementSchedulePolicy,
};
use redstone_compiler::transform::place_and_route::sampling::SamplingPolicy;
use redstone_compiler::world::position::DimSize;
use tracing_subscriber::prelude::*;
fn main() -> eyre::Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::DEBUG)
        .finish()
        .with(tracing_subscriber::filter::filter_fn(|meta| {
            *meta.level() <= tracing::Level::INFO
                || meta.target() == "redstone_compiler::global_pnr"
        }));
    tracing::subscriber::set_global_default(subscriber).ok();
    let mode = std::env::args().nth(1).unwrap_or_else(|| "single".into());
    let source = match mode.as_str() {
        "single" => "module probe(a,y); input a; output y; assign y=~a; endmodule",
        "fanout" => {
            "module probe(a,b,y,z); input a,b; output y,z; assign y=~a; assign z=a&b; endmodule"
        }
        "chain" => {
            "module probe(a,b,y); input a,b; output y; wire n; assign n=~a; assign y=n|b; endmodule"
        }
        _ => eyre::bail!("expected single, fanout, or chain"),
    };
    let logical = LogicalDesign::from_verilog_source(source)?;
    let routable = logical.lower_to_routable_with_bit_mapping_limits(1, 2)?;
    let cache = PathBuf::from(
        std::env::args()
            .nth(2)
            .unwrap_or_else(|| "target/alu-display/port-route-cache".into()),
    );
    let mut config = GlobalPnrConfig {
        candidate: UnitCandidateConfig {
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
        candidate_cache_dir: Some(cache),
        ..Default::default()
    };
    config.search =
        redstone_compiler::transform::place_and_route::global_pnr::policy::GlobalPnrPreset::Fast
            .search_config();
    config.placement.max_attempts = 4;
    let start = Instant::now();
    match place_and_route_routable_design_with_visualization(&routable, &config) {
        Ok(result) => println!(
            "probe={mode} success elapsed={:?} inputs={} outputs={}",
            start.elapsed(),
            result.placed_world.inputs.len(),
            result.placed_world.outputs.len()
        ),
        Err(error) => {
            eprintln!(
                "probe={mode} FAILED elapsed={:?}: {error:?}",
                start.elapsed()
            );
            return Err(error);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use redstone_compiler::transform::place_and_route::detailed_router;
    use redstone_compiler::transform::place_and_route::placed_node::PlacedNode;
    use redstone_compiler::world::block::{Block, BlockKind, Direction};
    use redstone_compiler::world::position::Position;
    use redstone_compiler::world::World3D;

    use super::*;

    #[test]
    fn overhead_wire_can_cage_an_unrouted_dust_port() {
        use redstone_compiler::transform::place_and_route::global_pnr::router::{
            route_point_to_point_with_strategy, GlobalRoutingStrategy,
        };
        let mut world = World3D::new(DimSize(7, 5, 5));
        for x in 0..7 {
            for y in 0..5 {
                let p = Position(x, y, 0);
                world[p] = PlacedNode::new_cobble(p).block;
            }
        }
        let source = Position(2, 2, 1);
        let sink = Position(4, 2, 2);
        for p in [
            Position(2, 1, 1),
            Position(2, 1, 2),
            Position(2, 3, 1),
            Position(2, 3, 2),
            Position(3, 2, 1),
            Position(4, 2, 1),
        ] {
            world[p] = PlacedNode::new_cobble(p).block;
        }
        world[Position(1, 2, 1)] =
            PlacedNode::new_repeater(Position(1, 2, 1), Direction::West).block;
        world[source] = PlacedNode::new_redstone(source).block;
        world[sink] = PlacedNode::new_redstone(sink).block;
        world.initialize_redstone_states();
        assert!(
            route_point_to_point_with_strategy(&world, source, sink, GlobalRoutingStrategy::AStar)
                .is_ok(),
            "unroofed port must have an uphill exit"
        );
        let roof = source.up();
        let support = detailed_router::try_generate_cobble_node(&world, roof, &[])
            .expect("existing rules permit roofing isolated dust");
        detailed_router::place_node(&mut world, support);
        let overhead = PlacedNode::new_redstone(roof.up());
        assert!(!overhead.has_conflict(&world, &Default::default()));
        assert!(!overhead.has_short(&world, &Default::default()));
        detailed_router::place_node(&mut world, overhead);
        assert!(
            route_point_to_point_with_strategy(&world, source, sink, GlobalRoutingStrategy::AStar)
                .is_err(),
            "unrelated overhead wire consumes the future port escape"
        );
    }

    #[test]
    fn recorded_compact_output_needs_first_hop_repeater() {
        // Failed source (43,4,2) neighborhood, translated by -40 on X.
        let mut world = World3D::new(DimSize(12, 12, 6));
        for (x, y, z) in [
            (2, 4, 0),
            (2, 3, 1),
            (4, 3, 1),
            (3, 4, 1),
            (5, 4, 1),
            (1, 3, 2),
            (3, 3, 2),
            (2, 4, 2),
            (6, 4, 2),
            (1, 5, 2),
            (2, 5, 2),
        ] {
            world[Position(x, y, z)] = Block {
                kind: BlockKind::Cobble {
                    on_count: 0,
                    on_base_count: 0,
                },
                direction: Direction::None,
            };
        }
        for (x, y, z, state) in [
            (1, 4, 1, 3),
            (3, 6, 1, 3),
            (2, 3, 2, 11),
            (4, 3, 2, 3),
            (5, 3, 2, 10),
            (5, 4, 2, 5),
            (3, 3, 3, 3),
            (2, 4, 3, 12),
            (2, 5, 3, 6),
        ] {
            world[Position(x, y, z)] = Block {
                kind: BlockKind::Redstone {
                    on_count: 0,
                    state,
                    strength: 0,
                },
                direction: Direction::None,
            };
        }
        world[Position(2, 4, 1)] =
            PlacedNode::new_repeater(Position(2, 4, 1), Direction::West).block;
        let source = Position(3, 4, 2);
        let sink = Position(8, 4, 2);
        world[source] = Block {
            kind: BlockKind::Torch { is_on: false },
            direction: Direction::Bottom,
        };
        let mut dust = Vec::new();
        let mut repeaters = Vec::new();
        for bound in PlacedNode::new(source, world[source]).propagation_bound(Some(&world)) {
            if let detailed_router::PlaceRedstoneResult::Placed(_, node) =
                detailed_router::place_redstone_with_cobble(&world, bound, source, sink)
            {
                dust.push(node.position);
            }
            for direction in [
                Direction::West,
                Direction::East,
                Direction::North,
                Direction::South,
            ] {
                if let detailed_router::PlaceRepeaterResult::Placed(_, node) =
                    detailed_router::place_repeater_with_cobble(
                        &world, bound, source, sink, direction, None,
                    )
                {
                    repeaters.push((node.position, direction));
                }
            }
        }
        eprintln!("dust={dust:?}, repeaters={repeaters:?}");
        assert!(dust.is_empty());
        assert!(
            !repeaters.is_empty(),
            "a legal directional escape exists before signal attenuation"
        );
    }
}
