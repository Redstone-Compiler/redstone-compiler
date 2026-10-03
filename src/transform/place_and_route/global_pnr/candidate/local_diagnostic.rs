//! A fixed full-adder experiment; no clustering, cache, packing, or retry loop.
use std::time::{Duration, Instant};

use super::*;
use crate::graph::logic::predefined_logics;
use crate::transform::place_and_route::local_placer::{NotRouteStrategy, TorchPlacementStrategy};

fn knob(name: &str, default: usize, max: usize) -> eyre::Result<usize> {
    let value = std::env::var(name).map_or(Ok(default), |value| value.parse::<usize>())?;
    eyre::ensure!(value > 0 && value <= max, "{name} must be in 1..={max}");
    Ok(value)
}

#[test]
#[ignore = "bounded local-only full-adder experiment; run explicitly with --nocapture"]
fn diagnose_monolithic_full_adder() -> eyre::Result<()> {
    let beam = knob("LOCAL_FA_BEAM", 64, 512)?;
    let depth = knob("LOCAL_FA_DEPTH", 4, 16)?;
    let route_beam = knob("LOCAL_FA_ROUTE_BEAM", 8, 32)?;
    let width = knob("LOCAL_FA_WIDTH", 2, 10)?;
    let side = knob("LOCAL_FA_SIDE", 10, 20)?;
    let height = knob("LOCAL_FA_HEIGHT", side, 20)?;
    let seconds = knob("LOCAL_FA_SECONDS", 10, 30)?;
    let schedule = match std::env::var("LOCAL_FA_SCHEDULE").as_deref() {
        Ok("frontier") => PlacementSchedulePolicy::MinFrontier,
        Ok("reconvergence") => PlacementSchedulePolicy::Reconvergence,
        Ok("topological") | Ok("defer_not") | Err(_) => PlacementSchedulePolicy::Topological,
        Ok(other) => eyre::bail!("unknown schedule: {other}"),
    };
    let flexible = std::env::var("LOCAL_FA_FLEXIBLE").as_deref() == Ok("1");
    let variant = std::env::var("LOCAL_FA_GRAPH").unwrap_or_else(|_| "buffered".to_owned());
    let mut graph = match variant.as_str() {
        "buffered" => predefined_logics::buffered_full_adder_graph()?,
        "nor9" | "nor10" => {
            let mut assignments = vec![
                ("n1", "~(a|b)"),
                ("n2", "~(a|n1)"),
                ("n3", "~(b|n1)"),
                ("n4", "~(n2|n3)"),
                ("n5", "~(n4|cin)"),
                ("n6", "~(n4|n5)"),
                ("n7", "~(cin|n5)"),
                ("s", "~(n6|n7)"),
            ];
            if variant == "nor10" {
                // Match the compact manual cell's local carry recomputation.
                assignments.push(("carry_n5", "~(n7|cin)"));
                assignments.push(("cout", "~(n1|carry_n5)"));
            } else {
                assignments.push(("cout", "~(n1|n5)"));
            }
            LogicGraph::from_assignments(
                assignments
                    .into_iter()
                    .map(|(name, expr)| (name.to_owned(), expr.to_owned())),
            )?
            .prepare_place()?
        }
        other => eyre::bail!("unknown fixed graph: {other}"),
    };
    // The predefined graph exposes intermediate observations as outputs too.
    // Keep only the full-adder's public outputs; its internal logic is unchanged.
    for name in [
        "c", "i", "d", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "carry_n5",
    ] {
        graph.graph.remove_output(name);
    }
    let table = graph.truth_table()?;
    let sum = (0..8usize)
        .map(|mask| mask.count_ones() % 2 == 1)
        .collect::<Vec<_>>();
    let carry = (0..8usize)
        .map(|mask| mask.count_ones() >= 2)
        .collect::<Vec<_>>();
    eyre::ensure!(
        table.input_names == ["a", "b", "cin"]
            && table.output_tables.len() == 2
            && table.output_table_set().contains(&sum)
            && table.output_table_set().contains(&carry),
        "fixed graph is not a full adder: {table:?}"
    );
    let expected = graph.clone();
    let config = LocalPlacerConfig {
        random_seed: 42,
        schedule,
        greedy_input_generation: true,
        step_sampling_policy: SamplingPolicy::Random(beam),
        placement_sampling_policy: LocalPlacerConfig::ranked_sampling(beam - beam / 8, beam / 8, 0),
        max_route_step: depth,
        route_step_sampling_policy: SamplingPolicy::Random(route_beam),
        max_not_route_step: if flexible { depth } else { 0 },
        not_route_step_sampling_policy: SamplingPolicy::Random(route_beam),
        route_torch_directly: !flexible,
        torch_placement_strategy: if flexible {
            TorchPlacementStrategy::AnywhereNonAdjacent
        } else {
            TorchPlacementStrategy::DirectOnly
        },
        not_route_strategy: if flexible {
            NotRouteStrategy::DirectAndRedstone
        } else {
            NotRouteStrategy::DirectOnly
        },
        ..Default::default()
    };
    let dim = DimSize(width, side, height);
    let pins = std::env::var("LOCAL_FA_PINS").unwrap_or_else(|_| "free".to_owned());
    let input_constraints = match pins.as_str() {
        "free" => LocalPlacerInputConstraints::default(),
        "manual" => {
            eyre::ensure!(dim == DimSize(2, 14, 10), "manual pins require 2x14x10");
            LocalPlacerInputConstraints::new()
                .with_input_positions("a", [Position(0, 0, 3)])
                .with_input_positions("b", [Position(0, 0, 1)])
                .with_input_positions("cin", [Position(0, 13, 5)])
        }
        other => eyre::bail!("unknown pin constraint: {other}"),
    };
    let joint = std::env::var("LOCAL_FA_JOINT").as_deref() == Ok("1");
    println!("LOCAL_FA config graph={variant} dim={dim:?} pins={pins} nodes={} beam={beam} route_beam={route_beam} depth={depth} schedule={schedule:?} flexible={flexible} joint={joint} seed=42 step_boundary_limit_s={seconds}", graph.nodes.len());
    let defer_not = std::env::var("LOCAL_FA_SCHEDULE").as_deref() == Ok("defer_not");
    let mut order = PlacementScheduler::new(&graph).select(schedule).order;
    if defer_not {
        // Experimental local schedule: let an independent OR route use free
        // space before committing the previous branch's terminal torch.
        for index in 1..order.len() {
            let previous = graph.find_node_by_id(order[index - 1]).unwrap();
            let current = graph.find_node_by_id(order[index]).unwrap();
            if matches!(&previous.kind, GraphNodeKind::Logic(logic) if logic.logic_type == crate::logic::LogicType::Not)
                && matches!(&current.kind, GraphNodeKind::Logic(logic) if logic.logic_type == crate::logic::LogicType::Or)
                && !current.inputs.contains(&previous.id)
            {
                order.swap(index - 1, index);
            }
        }
        println!("LOCAL_FA defer_not_order={order:?}");
    }
    let mut placer = LocalPlacer::new_with_visit_order(graph, config, order)?
        .with_time_limit(Duration::from_secs(seconds as u64));
    if joint {
        placer = placer.with_joint_ready_or_routes();
    }
    if std::env::var_os("LOCAL_FA_NOT_SITES").is_some() {
        let sites = knob("LOCAL_FA_NOT_SITES", 32, 256)?;
        println!("LOCAL_FA not_site_limit={sites}");
        placer = placer.with_not_site_limit(sites);
    }
    let mut debug = LocalPlacerDebug::default();
    let started = Instant::now();
    let candidates = placer.generate_with_outputs_and_input_constraints_debug_progress(
        dim,
        None,
        &input_constraints,
        Some(&mut debug),
        None,
    );
    let search_ms = started.elapsed().as_millis();
    debug.print_summary();
    println!(
        "LOCAL_FA failure={:?} time_limit_reached={}",
        debug.failure(),
        debug.time_limit_reached
    );
    let verify_started = Instant::now();
    let mut valid = 0;
    let mut invalid = 0;
    let mut errors = 0;
    for candidate in &candidates {
        match candidate_matches_truth_table(&expected, candidate) {
            Ok(true) => valid += 1,
            Ok(false) => invalid += 1,
            Err(error) => {
                errors += 1;
                if errors == 1 {
                    println!("LOCAL_FA first_verification_error={error:#}");
                }
            }
        }
    }
    println!("LOCAL_FA result search_ms={search_ms} verification_ms={} generated={} valid={valid} invalid={invalid} errors={errors}", verify_started.elapsed().as_millis(), candidates.len());
    // This is a measurement harness. A passing test does NOT assert a layout was found.
    Ok(())
}
