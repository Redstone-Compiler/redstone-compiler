use std::collections::BTreeSet;
use std::time::Duration;

use super::*;
use crate::graph::logic::LogicGraph;

fn graph(assignments: &[(&str, &str)]) -> LogicGraph {
    LogicGraph::from_assignments(
        assignments
            .iter()
            .map(|(name, expr)| (name.to_string(), expr.to_string())),
    )
    .unwrap()
    .prepare_place()
    .unwrap()
}

pub(super) fn full_adder_graph(variant: &str) -> LogicGraph {
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
        assignments.push(("carry_n5", "~(n7|cin)"));
        assignments.push(("cout", "~(n1|carry_n5)"));
    } else {
        assignments.push(("cout", "~(n1|n5)"));
    }
    let mut graph = LogicGraph::from_assignments(
        assignments
            .into_iter()
            .map(|(name, expr)| (name.to_owned(), expr.to_owned())),
    )
    .unwrap()
    .prepare_place()
    .unwrap();
    for name in ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "carry_n5"] {
        graph.graph.remove_output(name);
    }
    graph
}

fn expect_placed(placer: &ExactLocalPlacer, config: &ExactPlacerConfig) -> ExactPlacement {
    let (outcome, stats) = placer.place(config).unwrap();
    match outcome {
        ExactOutcome::Placed(placement) => *placement,
        other => panic!("expected a placement, got {other:?} ({stats:?})"),
    }
}

#[test]
fn nor_netlist_folds_or_nodes_into_gate_inputs() {
    let netlist = NorNetlist::from_logic_graph(&full_adder_graph("nor10")).unwrap();
    assert_eq!(netlist.input_names(), ["a", "b", "cin"]);
    assert_eq!(netlist.gates().count(), 10);
    assert!(netlist
        .gates()
        .all(|gate| netlist.nets[gate].gate_inputs.len() == 2));
    let values = netlist.net_values();
    let outputs = netlist.outputs.iter().cloned().collect::<BTreeMap<_, _>>();
    for case in 0..8usize {
        let ones = case.count_ones();
        assert_eq!(values[outputs["s"]][case], ones % 2 == 1);
        assert_eq!(values[outputs["cout"]][case], ones >= 2);
    }
}

/// The raw expression, the prepared graph, and the NOR netlist must compute
/// the same function, including operator chains with three operands.
#[test]
fn nor_netlist_keeps_every_operand_of_long_chains() {
    let cases: [(&str, fn(&BTreeMap<&str, bool>) -> bool); 4] = [
        ("a&b&c", |v| v["a"] && v["b"] && v["c"]),
        ("a|b|c", |v| v["a"] || v["b"] || v["c"]),
        ("a^b^c", |v| v["a"] ^ v["b"] ^ v["c"]),
        ("(~b&~c&a)|(b&c&a)", |v| v["a"] && (v["b"] == v["c"])),
    ];
    for (expr, expected) in cases {
        let raw = LogicGraph::from_assignments([("out".to_owned(), expr.to_owned())]).unwrap();
        let prepared = raw.clone().prepare_place().unwrap();
        let netlist = NorNetlist::from_logic_graph(&prepared).unwrap();
        let names = netlist.input_names();
        let values = netlist.net_values();
        let out = netlist
            .outputs
            .iter()
            .find(|(name, _)| name == "out")
            .unwrap()
            .1;
        let tables = [
            ("raw", raw.truth_table().unwrap()),
            ("prepared", prepared.truth_table().unwrap()),
        ];
        for case in 0..1usize << names.len() {
            let assignment = names
                .iter()
                .enumerate()
                .map(|(bit, name)| (name.as_str(), case & (1 << bit) != 0))
                .collect::<BTreeMap<_, _>>();
            let want = expected(&assignment);
            for (stage, table) in &tables {
                let index = table
                    .input_names
                    .iter()
                    .enumerate()
                    .fold(0, |index, (bit, name)| {
                        index | (usize::from(assignment[name.as_str()]) << bit)
                    });
                assert_eq!(
                    table.output_tables["out"][index], want,
                    "{expr}: {stage}, case {assignment:?}"
                );
            }
            assert_eq!(
                values[out][case], want,
                "{expr}: netlist, case {assignment:?}"
            );
        }
    }
}

/// Every order places every gate once, after its inputs; smallest-cone-first
/// finishes the small output before starting the larger one's private logic.
#[test]
fn gate_orders_are_topological_and_finish_small_cones_first() {
    let netlist =
        NorNetlist::from_logic_graph(&graph(&[("big", "a^b^c"), ("small", "~(a|b)")])).unwrap();
    let gates = netlist.gates().collect::<BTreeSet<_>>();
    for order in [
        GateOrder::NetIndex,
        GateOrder::SmallestConeFirst,
        GateOrder::MinLive,
    ] {
        let placed = construct::gate_order(&netlist, order);
        assert_eq!(placed.iter().copied().collect::<BTreeSet<_>>(), gates);
        assert_eq!(placed.len(), gates.len());
        for (index, &gate) in placed.iter().enumerate() {
            for input in &netlist.nets[gate].gate_inputs {
                if gates.contains(input) {
                    assert!(placed[..index].contains(input), "{order:?}");
                }
            }
        }
    }
    let placed = construct::gate_order(&netlist, GateOrder::SmallestConeFirst);
    let output = |name: &str| netlist.outputs.iter().find(|(n, _)| n == name).unwrap().1;
    // `small` is a single NOR gate, so it comes first.
    assert_eq!(placed[0], output("small"));
    assert_eq!(placed.last(), Some(&output("big")));
}

#[test]
fn exact_placer_builds_a_verified_inverter() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~a")])).unwrap();
    let mut config = ExactPlacerConfig::new(DimSize(1, 3, 2));
    config.time_limit = Some(Duration::from_secs(20));
    let placement = expect_placed(&placer, &config);
    assert_eq!(placement.placed.inputs.len(), 1);
    assert_eq!(placement.placed.outputs.len(), 1);
}

#[test]
fn exact_placer_builds_a_verified_nor_gate() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~(a|b)")])).unwrap();
    let mut config = ExactPlacerConfig::new(DimSize(1, 4, 2));
    config.time_limit = Some(Duration::from_secs(20));
    let placement = expect_placed(&placer, &config);
    let document = placement.rcell.to_string();
    let reparsed: crate::physical_cell::PhysicalCellDocument = document.parse().unwrap();
    let build = reparsed.build().unwrap();
    let verification = reparsed.verify(&build).unwrap();
    assert!(verification.failures.is_empty(), "{document}");
}

#[test]
fn exact_placer_proves_a_too_small_box_infeasible() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~(a|b)")])).unwrap();
    let config = ExactPlacerConfig::new(DimSize(1, 2, 1));
    let (outcome, _) = placer.place(&config).unwrap();
    assert!(matches!(outcome, ExactOutcome::Infeasible), "{outcome:?}");
}

#[test]
fn exact_placer_builds_a_verified_xor() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "a^b")])).unwrap();
    let mut config = ExactPlacerConfig::new(DimSize(2, 6, 4));
    // Solve times are heavy-tailed (1-60 s across seeds, see
    // compare_xor_folding and compare_xor_models); eight workers make a slow
    // draw less likely. Connecting dust to repeater outputs moved seed 1 from
    // 2 s to 36 s alone, past 60 s with the whole suite running.
    config.workers = 8;
    config.time_limit = Some(Duration::from_secs(120));
    expect_placed(&placer, &config);
}

#[test]
fn exact_placer_builds_verified_and_and_or_gates() {
    for expr in ["a&b", "a|b"] {
        let placer = ExactLocalPlacer::new(&graph(&[("out", expr)])).unwrap();
        let mut config = ExactPlacerConfig::new(DimSize(2, 4, 3));
        config.workers = 4;
        config.time_limit = Some(Duration::from_secs(60));
        expect_placed(&placer, &config);
    }
}

/// Seconds of per-step minimization; `0` turns it off, unset keeps the
/// default.
fn step_optimize_from_env(name: &str) -> Option<Duration> {
    match std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    {
        Some(0) => None,
        Some(seconds) => Some(Duration::from_secs(seconds)),
        None => ConstructionConfig::default().step_optimize,
    }
}

/// Seconds of minimization after each slice-removal repair; `0` turns it
/// off, unset keeps the default.
fn repair_optimize_from_env(name: &str) -> Option<Duration> {
    match std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    {
        Some(0) => None,
        Some(seconds) => Some(Duration::from_secs(seconds)),
        None => CompactionConfig::default().repair_optimize,
    }
}

/// Slice removals per compaction round; `0` means no cap, unset keeps the
/// default.
fn removals_per_round_from_env(name: &str) -> Option<usize> {
    match std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        Some(0) => None,
        Some(cap) => Some(cap),
        None => CompactionConfig::default().max_removals_per_round,
    }
}

/// `name=value,...` model params; values are `true`, `false` or integers.
fn model_params_from_env(name: &str) -> BTreeMap<String, rsdsl::IValue> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair
                .split_once('=')
                .expect("model params look like name=value");
            let value = match value {
                "true" => rsdsl::IValue::Bool(true),
                "false" => rsdsl::IValue::Bool(false),
                number => rsdsl::IValue::Int(number.parse().expect("integer model param")),
            };
            (key.to_owned(), value)
        })
        .collect()
}

/// `index` selects `GateOrder::NetIndex`; anything else the default.
fn gate_order_from_env(name: &str) -> GateOrder {
    match std::env::var(name).as_deref() {
        Ok("index") => GateOrder::NetIndex,
        Ok("min-live") => GateOrder::MinLive,
        _ => GateOrder::SmallestConeFirst,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Measurement harness for the full adder. A passing run does not by itself
/// mean a layout was found: read the `EXACT_FA result` line.
///
/// Knobs: `EXACT_FA_DIM=2x14x10`, `EXACT_FA_GRAPH=nor9|nor10`,
/// `EXACT_FA_PINS=manual|free`, `EXACT_FA_SUM_FACE=1` (sum on the y-max face),
/// `EXACT_FA_WORKERS`, `EXACT_FA_SECONDS`, `EXACT_FA_RANKS`,
/// `EXACT_FA_MINIMIZE=1`, `EXACT_FA_WRITE=<path prefix>` (writes .rcell/.nbt).
#[test]
#[ignore = "exact full-adder measurement; run explicitly with --nocapture"]
fn diagnose_exact_full_adder() -> eyre::Result<()> {
    let dim_text = std::env::var("EXACT_FA_DIM").unwrap_or_else(|_| "2x14x10".to_owned());
    let parts = dim_text
        .split('x')
        .map(|part| part.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    eyre::ensure!(parts.len() == 3, "EXACT_FA_DIM must look like 2x14x10");
    let dim = DimSize(parts[0], parts[1], parts[2]);
    let variant = std::env::var("EXACT_FA_GRAPH").unwrap_or_else(|_| "nor9".to_owned());
    let pins = std::env::var("EXACT_FA_PINS").unwrap_or_else(|_| "manual".to_owned());
    let placer = ExactLocalPlacer::new(&full_adder_graph(&variant))?
        .with_name(format!("exact-full-adder-{}x{}x{}", dim.0, dim.1, dim.2));
    let mut config = ExactPlacerConfig::new(dim);
    config.workers = env_usize("EXACT_FA_WORKERS", 8);
    config.rank_levels = env_usize("EXACT_FA_RANKS", 32);
    config.time_limit = Some(Duration::from_secs(
        env_usize("EXACT_FA_SECONDS", 600) as u64
    ));
    if pins == "manual" {
        eyre::ensure!(
            dim.0 >= 2 && dim.1 >= 14 && dim.2 >= 6,
            "manual pins need 2x14x6 or larger"
        );
        config = config
            .with_input_site("a", Position(0, 0, 3), Direction::East)
            .with_input_site("b", Position(0, 0, 1), Direction::East)
            .with_input_site("cin", Position(0, dim.1 - 1, 5), Direction::East);
    }
    if std::env::var("EXACT_FA_SUM_FACE").as_deref() == Ok("1") {
        let face = (0..dim.2)
            .flat_map(|z| (0..dim.0).map(move |x| Position(x, dim.1 - 1, z)))
            .collect::<Vec<_>>();
        config = config.with_output_sites("s", face);
    }
    println!(
        "EXACT_FA config graph={variant} dim={dim:?} pins={pins} workers={} ranks={} gates={}",
        config.workers,
        config.rank_levels,
        placer.netlist().gates().count()
    );
    let placement = if std::env::var("EXACT_FA_MINIMIZE").as_deref() == Ok("1") {
        let (best, optimal, stats) = placer.place_minimizing_blocks(&config)?;
        println!("EXACT_FA stats {stats:?} proven_optimal={optimal}");
        best
    } else {
        let (outcome, stats) = placer.place(&config)?;
        println!("EXACT_FA stats {stats:?}");
        match outcome {
            ExactOutcome::Placed(placement) => Some(*placement),
            other => {
                println!("EXACT_FA result outcome={other:?}");
                None
            }
        }
    };
    if let Some(placement) = placement {
        println!(
            "EXACT_FA result placed blocks={} inputs={:?} outputs={:?}",
            placement.block_count, placement.placed.inputs, placement.placed.outputs
        );
        println!("{}", placement.rcell);
        if let Ok(prefix) = std::env::var("EXACT_FA_WRITE") {
            std::fs::write(format!("{prefix}.rcell"), placement.rcell.to_string())?;
            crate::nbt::NBTRoot::from(&placement.placed.world).save(format!("{prefix}.nbt"));
            println!("EXACT_FA wrote {prefix}.rcell and {prefix}.nbt");
        }
    }
    Ok(())
}

/// Prints the truth vector of every torch in the manual 2x14x10 cell, to see
/// which logic functions a dense hand layout actually uses.
#[test]
#[ignore = "analysis of the manual cell; run explicitly with --nocapture"]
fn analyze_manual_full_adder_torches() -> eyre::Result<()> {
    let source = include_str!("../../../../../test/full-adder-right-inputs-2x14x10.rcell");
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let build = document.build()?;
    let netlist = NorNetlist::from_logic_graph(&full_adder_graph("nor10"))?;
    let values = netlist.net_values();
    let names = ["a", "b", "cin"];
    let mut vectors = BTreeMap::<crate::world::position::Position, Vec<bool>>::new();
    for case in 0..8usize {
        let inputs = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.to_string(), case & (1 << index) != 0))
            .collect();
        let simulation = document.simulate_case(&build, inputs, 0)?;
        for (position, block) in simulation.simulator.world().iter_block() {
            if block.kind.is_torch() {
                vectors
                    .entry(position)
                    .or_default()
                    .push(block.kind.is_powered());
            }
        }
    }
    for (position, vector) in vectors {
        let bits = vector
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect::<String>();
        let matches = (0..netlist.nets.len())
            .filter(|&net| values[net] == vector)
            .map(|net| netlist.nets[net].name.clone())
            .collect::<Vec<_>>();
        let complement = (0..netlist.nets.len())
            .filter(|&net| values[net].iter().zip(&vector).all(|(a, b)| a != b))
            .map(|net| format!("~{}", netlist.nets[net].name))
            .collect::<Vec<_>>();
        println!("torch {position:?} {bits} nets={matches:?} complements={complement:?}");
    }
    Ok(())
}

fn manual_kind(block: &crate::world::block::Block) -> CellKind {
    use crate::world::block::BlockKind;
    match block.kind {
        BlockKind::Air => CellKind::Air,
        BlockKind::Cobble { .. } => CellKind::Solid,
        BlockKind::Redstone { .. } => CellKind::Dust,
        BlockKind::Torch { .. } => CellKind::Torch(block.direction),
        BlockKind::Repeater { .. } => CellKind::Repeater(block.direction),
        BlockKind::Switch { .. } => CellKind::Switch(block.direction),
        other => panic!("unexpected block {other:?}"),
    }
}

/// Fixes every block of a verified hand-made cell and asks the solver to
/// complete the class, rank, and stage labels. If this is unsatisfiable, the
/// model rejects a known-good layout; the failed assumptions name the cells.
fn assert_model_accepts_rcell(
    source: &str,
    variant: &str,
    rank_levels: usize,
    stage_levels: usize,
) {
    assert_encoder_accepts_rcell(source, variant, rank_levels, stage_levels);
}

fn assert_encoder_accepts_rcell(
    source: &str,
    variant: &str,
    rank_levels: usize,
    stage_levels: usize,
) {
    use super::encode::{Encoding, TORCH_ATTACH};
    use super::solver::{SatSolver, SolveResult, StopSignal};
    let document: crate::physical_cell::PhysicalCellDocument = source.parse().unwrap();
    let build = document.build().unwrap();
    let netlist = NorNetlist::from_logic_graph(&full_adder_graph(variant)).unwrap();
    let mut config = ExactPlacerConfig::new(build.world.size);
    config.rank_levels = rank_levels;
    config.stage_levels = stage_levels;
    config.allow_unpowered_wires = true;
    for input in &document.inputs {
        config = config.with_input_site(
            input.name.clone(),
            input.position,
            build.world[input.position].direction,
        );
    }
    for output in &document.outputs {
        let name = if output.name == "sum" {
            "s"
        } else {
            output.name.as_str()
        };
        config = config.with_output_sites(name, [output.position]);
        // An output repeater that points out of the box drives the neighbor.
        if let CellKind::Repeater(direction) = manual_kind(&build.world[output.position]) {
            let out = output.position.walk(direction.inverse());
            if out.is_none_or(|out| !config.dim.bound_on(out)) {
                config.driving_outputs.insert(name.to_owned());
            }
        }
    }
    let encoding = Encoding::build(&netlist, &config).unwrap();
    let mut assumptions = Vec::new();
    let mut described = Vec::new();
    for cell in 0..encoding.geometry.len() {
        let position = encoding.geometry.position(cell);
        let kind = manual_kind(&build.world[position]);
        let lit = match kind {
            CellKind::Air => encoding.air[cell],
            CellKind::Solid => encoding.solid[cell],
            CellKind::Dust => encoding.dust[cell],
            CellKind::Torch(attach) => {
                let index = TORCH_ATTACH.iter().position(|&d| d == attach).unwrap();
                encoding.torch[cell][index]
            }
            CellKind::Repeater(direction) => {
                let index = super::encode::CARDINALS
                    .iter()
                    .position(|&d| d == direction)
                    .unwrap();
                encoding.repeater[cell][index]
            }
            CellKind::Switch(attach) => {
                encoding
                    .switches
                    .iter()
                    .find(|site| site.cell == cell && site.attach == attach)
                    .unwrap()
                    .lit
            }
        };
        assert!(
            !encoding.cnf.is_false(lit),
            "{position:?} {kind:?} is not encodable"
        );
        assumptions.push(lit);
        described.push((lit, position, kind));
    }
    let mut solver = SatSolver::new(1);
    solver.add_cnf(&encoding.cnf);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let signal = StopSignal {
        stop: &stop,
        deadline: Some(std::time::Instant::now() + Duration::from_secs(120)),
        restart: None,
    };
    let result = solver.solve(&assumptions, &signal);
    if result == SolveResult::Unsat {
        let core = described
            .iter()
            .filter(|(lit, _, _)| solver.failed(*lit))
            .map(|(_, position, kind)| format!("{position:?}={kind:?}"))
            .collect::<Vec<_>>();
        // Localize further: fix blocks, then assume the simulator's power
        // values and report which (cell, case) values the model refuses.
        let mut solver = SatSolver::new(1);
        solver.add_cnf(&encoding.cnf);
        for (lit, _, _) in &described {
            solver.add_clause(&[*lit]);
        }
        let names = ["a", "b", "cin"];
        let mut value_assumptions = Vec::new();
        for case in 0..8usize {
            let inputs = names
                .iter()
                .enumerate()
                .map(|(index, name)| (name.to_string(), case & (1 << index) != 0))
                .collect();
            let simulation = document.simulate_case(&build, inputs, 0).unwrap();
            for cell in 0..encoding.geometry.len() {
                let position = encoding.geometry.position(cell);
                let block = simulation.simulator.world()[position];
                if block.kind.is_air() || block.kind.is_switch() {
                    continue;
                }
                let powered = block.kind.is_powered();
                let lit = encoding.values[cell][case];
                value_assumptions.push((if powered { lit } else { -lit }, position, case, powered));
            }
        }
        let lits = value_assumptions
            .iter()
            .map(|(lit, ..)| *lit)
            .collect::<Vec<_>>();
        let second = solver.solve(&lits, &signal);
        // Third pass: relax soundness per relation to find the disagreeing rule.
        let mut relaxed_config = config.clone();
        relaxed_config.relax_soundness = true;
        let relaxed = Encoding::build(&netlist, &relaxed_config).unwrap();
        let mut relaxed_solver = SatSolver::new(1);
        relaxed_solver.add_cnf(&relaxed.cnf);
        for (lit, _, _) in &described {
            relaxed_solver.add_clause(&[*lit]);
        }
        for (lit, ..) in &value_assumptions {
            relaxed_solver.add_clause(&[*lit]);
        }
        let guards = relaxed.relaxations.clone();
        let coverage_guards = relaxed
            .coverage_relaxations
            .iter()
            .map(|(_, guard)| -*guard)
            .collect::<Vec<_>>();
        let mut all_guards = guards.iter().map(|guard| -*guard).collect::<Vec<_>>();
        all_guards.extend(coverage_guards.iter().copied());
        let third = relaxed_solver.solve(&all_guards, &signal);
        if third == SolveResult::Unsat {
            let uncovered = relaxed
                .coverage_relaxations
                .iter()
                .filter(|(_, guard)| relaxed_solver.failed(-*guard))
                .map(|(cell, _)| format!("{:?}", relaxed.geometry.position(*cell)))
                .collect::<Vec<_>>();
            eprintln!("coverage core: {uncovered:?}");
        }
        let relation_core = if third == SolveResult::Unsat {
            relaxed
                .relations
                .iter()
                .zip(&guards)
                .filter(|(_, guard)| relaxed_solver.failed(-**guard))
                .map(|(relation, _)| {
                    format!(
                        "{:?}->{:?} {:?}->{:?}",
                        relaxed.geometry.position(relation.source),
                        relaxed.geometry.position(relation.sink),
                        relation.source_kind,
                        relation.sink_kind
                    )
                })
                .collect::<Vec<_>>()
        } else {
            vec![format!("relaxed solve is {third:?}")]
        };
        eprintln!("relation core: {relation_core:?}");
        // Cells whose simulated function is outside the class vocabulary.
        let mut functions = BTreeMap::<crate::world::position::Position, u64>::new();
        for (_, position, case, powered) in &value_assumptions {
            *functions.entry(*position).or_default() |= u64::from(*powered) << case;
        }
        let outside = functions
            .iter()
            .filter(|(_, function)| {
                **function != 0
                    && !encoding
                        .classes
                        .iter()
                        .any(|class| class.function == **function)
            })
            .map(|(position, function)| format!("{position:?}={function:08b}"))
            .collect::<Vec<_>>();
        eprintln!("functions outside vocabulary: {outside:?}");
        let value_core = if second == SolveResult::Unsat {
            value_assumptions
                .iter()
                .filter(|(lit, ..)| solver.failed(*lit))
                .map(|(_, position, case, powered)| format!("{position:?}@{case}={powered}"))
                .collect::<Vec<_>>()
        } else {
            vec![format!("value assumptions are {second:?}")]
        };
        panic!("model rejects the manual layout; core cells: {core:?}; value core: {value_core:?}");
    }
    assert_eq!(result, SolveResult::Sat);
}

#[test]
fn model_accepts_manual_right_inputs_full_adder() {
    assert_model_accepts_rcell(
        include_str!("../../../../../test/full-adder-right-inputs-2x14x10.rcell"),
        "nor10",
        24,
        24,
    );
}

#[test]
fn model_accepts_other_manual_full_adders() {
    for source in [
        include_str!("../../../../../test/full-adder-2x13x9.rcell"),
        include_str!("../../../../../test/full-adder-2x17x10.rcell"),
    ] {
        assert_model_accepts_rcell(source, "nor9", 24, 24);
    }
    // A layout the exact placer compacted with the since-removed hand-written
    // encoder. (The generated 2x13x7 cell is not checked here: it has a
    // repeater pointing out of the box, legal only with the whole max-Y face
    // as sum sites.)
    assert_model_accepts_rcell(
        include_str!("../../../../../test/full-adder-right-inputs-compacted-2x10x10.rcell"),
        "nor10",
        24,
        24,
    );
    // `full-adder-2x20x20.rcell` is intentionally absent: its carry-out support
    // ORs n1 and n5 through one dust cell that carries power in opposite
    // directions depending on the input case. A single rank per cell cannot
    // order that bridge (see docs/exact_local_placer.md, "Limitations").
}

#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn measure_exact_xor_modes() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "a^b")])).unwrap();
    for (ranks, stages) in [(24, 12), (0, 12), (24, 0), (0, 0)] {
        let mut config = ExactPlacerConfig::new(DimSize(2, 6, 4));
        config.workers = 4;
        config.rank_levels = ranks;
        config.stage_levels = stages;
        config.time_limit = Some(Duration::from_secs(60));
        let (outcome, stats) = placer.place(&config).unwrap();
        println!(
            "XOR ranks={ranks} stages={stages} placed={} solve={:?} loops={} feedback={} rejections={} vars={} clauses={}",
            matches!(outcome, ExactOutcome::Placed(_)),
            stats.solve_time,
            stats.loop_formulas,
            stats.feedback_cuts,
            stats.refinements,
            stats.variables,
            stats.clauses
        );
    }
}

/// How hard is routing alone? Fix the manual cell's torches (position, side,
/// and optionally signal class) and let the solver choose everything else.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn measure_routing_with_manual_torches() {
    use super::encode::{Encoding, TORCH_ATTACH};
    use super::solver::{SatSolver, SolveResult, StopSignal};
    let source = include_str!("../../../../../test/full-adder-right-inputs-2x14x10.rcell");
    let document: crate::physical_cell::PhysicalCellDocument = source.parse().unwrap();
    let build = document.build().unwrap();
    let netlist = NorNetlist::from_logic_graph(&full_adder_graph("nor10")).unwrap();
    let mut config = ExactPlacerConfig::new(build.world.size);
    for input in &document.inputs {
        config = config.with_input_site(input.name.clone(), input.position, Direction::East);
    }
    let encoding = Encoding::build(&netlist, &config).unwrap();
    let fix_classes = std::env::var("FIX_CLASSES").as_deref() == Ok("1");
    let mut assumptions = Vec::new();
    let names = ["a", "b", "cin"];
    let mut functions = BTreeMap::<crate::world::position::Position, u64>::new();
    for case in 0..8usize {
        let inputs = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.to_string(), case & (1 << index) != 0))
            .collect();
        let simulation = document.simulate_case(&build, inputs, 0).unwrap();
        for (position, block) in simulation.simulator.world().iter_block() {
            if block.kind.is_torch() {
                *functions.entry(position).or_default() |=
                    u64::from(block.kind.is_powered()) << case;
            }
        }
    }
    for (position, function) in &functions {
        let cell = encoding.geometry.index(*position);
        let attach = build.world[*position].direction;
        let slot = TORCH_ATTACH.iter().position(|&d| d == attach).unwrap();
        assumptions.push(encoding.torch[cell][slot]);
        if fix_classes {
            let class = encoding
                .classes
                .iter()
                .position(|class| class.function == *function)
                .unwrap();
            assumptions.push(encoding.class_lits[cell][class]);
        }
    }
    // No other torches anywhere.
    for cell in 0..encoding.geometry.len() {
        if !functions.contains_key(&encoding.geometry.position(cell)) {
            for &torch in &encoding.torch[cell] {
                if !encoding.cnf.is_false(torch) {
                    assumptions.push(-torch);
                }
            }
        }
    }
    let workers = 8;
    let started = std::time::Instant::now();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let results = std::thread::scope(|scope| {
        (0..workers)
            .map(|worker| {
                let encoding = &encoding;
                let assumptions = &assumptions;
                let stop = &stop;
                scope.spawn(move || {
                    let mut solver = SatSolver::new(1 + worker as u32 * 7919);
                    solver.set_option("phase", i32::from(worker % 2 == 1));
                    solver.add_cnf(&encoding.cnf);
                    let signal = StopSignal {
                        stop,
                        deadline: Some(std::time::Instant::now() + Duration::from_secs(300)),
                        restart: None,
                    };
                    let result = solver.solve(assumptions, &signal);
                    if result != SolveResult::Interrupted {
                        stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    result
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    println!(
        "ROUTING fix_classes={fix_classes} torches={} results={results:?} elapsed={:?}",
        functions.len(),
        started.elapsed()
    );
}

#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn measure_xnor_core_stage() {
    let mut core = LogicGraph::from_assignments(
        [
            ("n1", "~(a|b)"),
            ("n2", "~(a|n1)"),
            ("n3", "~(b|n1)"),
            ("n4", "~(n2|n3)"),
        ]
        .into_iter()
        .map(|(name, expr)| (name.to_owned(), expr.to_owned())),
    )
    .unwrap()
    .prepare_place()
    .unwrap();
    core.graph.remove_output("n2");
    core.graph.remove_output("n3");
    let placer = ExactLocalPlacer::new(&core).unwrap();
    let length = env_usize("CORE_LENGTH", 7);
    let height = env_usize("CORE_HEIGHT", 10);
    let dim = DimSize(2, length, height);
    let face = (0..height)
        .flat_map(|z| (0..2).map(move |x| Position(x, length - 1, z)))
        .collect::<Vec<_>>();
    let mut config = ExactPlacerConfig::new(dim);
    if std::env::var("CORE_PINS").as_deref() != Ok("free") {
        config = config
            .with_input_site("a", Position(0, 0, 3), Direction::East)
            .with_input_site("b", Position(0, 0, 1), Direction::East);
    }
    if std::env::var("CORE_FACE").as_deref() != Ok("0") {
        config = config
            .with_output_sites("n4", face.clone())
            .with_output_sites("n1", face);
    }
    config.workers = env_usize("CORE_WORKERS", 8);
    config.rank_levels = env_usize("CORE_RANKS", 24);
    config.stage_levels = env_usize("CORE_STAGES", 12);
    config.seed = env_usize("CORE_SEED", 1) as u32;
    config.time_limit = Some(Duration::from_secs(env_usize("CORE_SECONDS", 120) as u64));
    let (outcome, stats) = placer.place(&config).unwrap();
    println!(
        "CORE length={length} height={height} placed={} solve={:?} vars={} clauses={}",
        matches!(outcome, ExactOutcome::Placed(_)),
        stats.solve_time,
        stats.variables,
        stats.clauses
    );
    if let ExactOutcome::Placed(placement) = outcome {
        println!("{}", placement.rcell);
    }
}

/// Stage-two difficulty in the ideal case: keep the manual cell's blocks with
/// `y < SPLIT_Y` and let the solver complete the rest of the box.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn measure_completion_after_manual_prefix() {
    use super::encode::{Encoding, CARDINALS, TORCH_ATTACH};
    use super::solver::{SatSolver, SolveResult, StopSignal};
    let source = include_str!("../../../../../test/full-adder-right-inputs-2x14x10.rcell");
    let document: crate::physical_cell::PhysicalCellDocument = source.parse().unwrap();
    let build = document.build().unwrap();
    let netlist = NorNetlist::from_logic_graph(&full_adder_graph("nor10")).unwrap();
    let split = env_usize("SPLIT_Y", 7);
    let mut config = ExactPlacerConfig::new(build.world.size);
    for input in &document.inputs {
        config = config.with_input_site(input.name.clone(), input.position, Direction::East);
    }
    if std::env::var("SUM_FACE").as_deref() == Ok("1") {
        let face = (0..10)
            .flat_map(|z| (0..2).map(move |x| Position(x, 13, z)))
            .collect::<Vec<_>>();
        config = config.with_output_sites("s", face);
    }
    let encoding = Encoding::build(&netlist, &config).unwrap();
    let mut assumptions = Vec::new();
    for cell in 0..encoding.geometry.len() {
        let position = encoding.geometry.position(cell);
        if position.1 >= split {
            continue;
        }
        let lit = match manual_kind(&build.world[position]) {
            CellKind::Air => encoding.air[cell],
            CellKind::Solid => encoding.solid[cell],
            CellKind::Dust => encoding.dust[cell],
            CellKind::Torch(attach) => {
                encoding.torch[cell][TORCH_ATTACH.iter().position(|&d| d == attach).unwrap()]
            }
            CellKind::Repeater(direction) => {
                encoding.repeater[cell][CARDINALS.iter().position(|&d| d == direction).unwrap()]
            }
            CellKind::Switch(attach) => {
                encoding
                    .switches
                    .iter()
                    .find(|site| site.cell == cell && site.attach == attach)
                    .unwrap()
                    .lit
            }
        };
        assumptions.push(lit);
    }
    if let Ok(path) = std::env::var("DUMP") {
        write_dimacs(&encoding, &assumptions, &path);
        println!("COMPLETION dumped {path}");
        return;
    }
    let workers = env_usize("WORKERS", 8);
    let seconds = env_usize("SECONDS", 300) as u64;
    let started = std::time::Instant::now();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let results = std::thread::scope(|scope| {
        (0..workers)
            .map(|worker| {
                let encoding = &encoding;
                let assumptions = &assumptions;
                let stop = &stop;
                scope.spawn(move || {
                    let mut solver = SatSolver::new(1 + worker as u32 * 7919);
                    solver.set_option("phase", i32::from(worker % 2 == 1));
                    solver.add_cnf(&encoding.cnf);
                    let signal = StopSignal {
                        stop,
                        deadline: Some(std::time::Instant::now() + Duration::from_secs(seconds)),
                        restart: None,
                    };
                    let result = solver.solve(assumptions, &signal);
                    if result != SolveResult::Interrupted {
                        stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    result
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    println!(
        "COMPLETION split_y={split} fixed={} results={results:?} elapsed={:?}",
        assumptions.len(),
        started.elapsed()
    );
}

fn write_dimacs(encoding: &super::encode::Encoding, units: &[i32], path: &str) {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let clauses = encoding.cnf.clause_count() + units.len();
    writeln!(out, "p cnf {} {}", encoding.cnf.num_vars(), clauses).unwrap();
    let mut line = Vec::new();
    for &lit in encoding.cnf.literals() {
        if lit == 0 {
            line.push("0".to_owned());
            writeln!(out, "{}", line.join(" ")).unwrap();
            line.clear();
        } else {
            line.push(lit.to_string());
        }
    }
    for unit in units {
        writeln!(out, "{unit} 0").unwrap();
    }
}

/// Writes the whole 2x14x10 manual-pin problem as DIMACS for external solvers.
#[test]
#[ignore = "export; run explicitly"]
fn export_full_adder_dimacs() {
    use super::encode::Encoding;
    let netlist = NorNetlist::from_logic_graph(&full_adder_graph("nor10")).unwrap();
    let config = ExactPlacerConfig::new(DimSize(2, 14, 10))
        .with_input_site("a", Position(0, 0, 3), Direction::East)
        .with_input_site("b", Position(0, 0, 1), Direction::East)
        .with_input_site("cin", Position(0, 13, 5), Direction::East);
    let encoding = Encoding::build(&netlist, &config).unwrap();
    write_dimacs(&encoding, &[], &std::env::var("DUMP").unwrap());
}

#[test]
fn compaction_shrinks_a_loose_nor_and_keeps_it_verified() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~(a|b)")])).unwrap();
    let mut config = ExactPlacerConfig::new(DimSize(2, 6, 4));
    config.workers = 4;
    config.time_limit = Some(Duration::from_secs(60));
    let loose = expect_placed(&placer, &config);
    let layout = ExactLayout::from_placement(config.dim, &loose);
    let compaction = CompactionConfig {
        workers: 4,
        attempt_time_limit: Duration::from_secs(10),
        time_limit: Some(Duration::from_secs(60)),
        ..Default::default()
    };
    let (compacted, placement, report) = placer.compact(layout, &compaction).unwrap();
    println!(
        "compaction {:?} -> {:?} {report:?}",
        config.dim, compacted.dim
    );
    let placement = placement.expect("at least one slice can be removed from a loose box");
    assert!(compacted.dim.1 * compacted.dim.2 < config.dim.1 * config.dim.2);
    let document = placement.rcell.to_string();
    let reparsed: crate::physical_cell::PhysicalCellDocument = document.parse().unwrap();
    let build = reparsed.build().unwrap();
    assert!(
        reparsed.verify(&build).unwrap().failures.is_empty(),
        "{document}"
    );
}

/// Construct-then-compact pipeline for the full adder with the requested
/// interface: both operands on the Y-min face, sum on the Y-max face.
///
/// Knobs: `PIPE_HEIGHT`, `PIPE_WINDOW`, `PIPE_STEP_SECONDS`,
/// `PIPE_COMPACT_SECONDS`, `PIPE_WRITE=<path prefix>`.
#[test]
#[ignore = "full-adder pipeline measurement; run explicitly with --nocapture"]
fn diagnose_full_adder_construct_and_compact() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let placer = ExactLocalPlacer::new(&full_adder_graph("nor9"))?.with_name("exact-full-adder");
    let input_policies = [
        ("a".to_owned(), InputPolicy::MinYFace),
        ("b".to_owned(), InputPolicy::MinYFace),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    let output_policies = [("s".to_owned(), OutputPolicy::MaxYFace)]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let construction = ConstructionConfig {
        height: env_usize("PIPE_HEIGHT", 10),
        window: env_usize("PIPE_WINDOW", 2),
        max_window: env_usize("PIPE_MAX_WINDOW", 4),
        step_time_limit: Duration::from_secs(env_usize("PIPE_STEP_SECONDS", 60) as u64),
        workers: env_usize("PIPE_WORKERS", 8),
        seed: env_usize("PIPE_SEED", 1) as u32,
        rank_levels: env_usize("PIPE_RANKS", 24),
        input_policies,
        output_policies: output_policies.clone(),
        max_overlap: env_usize("PIPE_MAX_OVERLAP", 1),
        block_seam: std::env::var("PIPE_BLOCK_SEAM").as_deref() == Ok("1"),
        max_restarts: env_usize("PIPE_RESTARTS", 7),
        max_backtracks: env_usize("PIPE_BACKTRACKS", 0),
        gate_order: gate_order_from_env("PIPE_ORDER"),
        early_outputs: std::env::var("PIPE_EARLY_OUTPUTS").as_deref() != Ok("0"),
        given_frozen_signals: std::env::var("PIPE_GIVEN").as_deref() != Ok("0"),
        step_optimize: step_optimize_from_env("PIPE_STEP_OPTIMIZE"),
        no_fold: std::env::var("PIPE_FOLD").as_deref() == Ok("0"),
        model_params: model_params_from_env("PIPE_MODEL_PARAMS"),
        ..Default::default()
    };
    let (layout, placement, report) = placer.construct(&construction)?;
    println!(
        "PIPE constructed dim={:?} blocks={} seed={} restarts={:?} steps={:?} elapsed={:?}",
        layout.dim,
        placement.block_count,
        report.seed,
        report
            .restarts
            .iter()
            .map(|(seed, _)| *seed)
            .collect::<Vec<_>>(),
        report.steps,
        report.elapsed
    );
    if let Ok(prefix) = std::env::var("PIPE_WRITE") {
        std::fs::write(format!("{prefix}-loose.rcell"), placement.rcell.to_string())?;
    }
    let compaction = CompactionConfig {
        workers: env_usize("PIPE_WORKERS", 8),
        attempt_time_limit: Duration::from_secs(env_usize("PIPE_ATTEMPT_SECONDS", 20) as u64),
        seed: env_usize("PIPE_SEED", 1) as u32,
        rank_levels: env_usize("PIPE_RANKS", 24),
        window_radius: env_usize("PIPE_RADIUS", 1),
        time_limit: Some(Duration::from_secs(
            env_usize("PIPE_COMPACT_SECONDS", 1800) as u64
        )),
        output_policies,
        repair_optimize: repair_optimize_from_env("PIPE_REPAIR_OPTIMIZE"),
        max_removals_per_round: removals_per_round_from_env("PIPE_REMOVALS_PER_ROUND"),
        ..Default::default()
    };
    let (compacted, best, report) = placer.compact(layout, &compaction)?;
    println!(
        "PIPE compacted dim={:?} removed={:?} attempts={} elapsed={:?}",
        compacted.dim, report.removed, report.attempts, report.elapsed
    );
    let final_placement = best.unwrap_or(placement);
    println!(
        "PIPE result blocks={} inputs={:?} outputs={:?}",
        final_placement.block_count, final_placement.placed.inputs, final_placement.placed.outputs
    );
    println!("{}", final_placement.rcell);
    if let Ok(prefix) = std::env::var("PIPE_WRITE") {
        std::fs::write(format!("{prefix}.rcell"), final_placement.rcell.to_string())?;
        crate::nbt::NBTRoot::from(&final_placement.placed.world).save(format!("{prefix}.nbt"));
    }
    Ok(())
}

/// Compacts an existing full-adder RCELL (generated or hand-made) further.
/// Knobs: `RECOMPACT_SOURCE=<rcell path>`, `RECOMPACT_WRITE=<path prefix>`,
/// `RECOMPACT_CIRCUIT=<name>` (another `circuit_graph` circuit instead of the
/// full adder), `RECOMPACT_SECONDS`, `RECOMPACT_WORKERS`, `RECOMPACT_RADIUS`,
/// `RECOMPACT_ATTEMPT_SECONDS`, `RECOMPACT_CONTINUE=0` (restart each
/// block-reduction pass after a gain), `RECOMPACT_ROUNDS=0`,
/// `RECOMPACT_REDUCTION_AXES=1` (Y windows only), `RECOMPACT_GIVEN=0` (no
/// given signals outside the window).
#[test]
#[ignore = "full-adder recompaction measurement; run explicitly with --nocapture"]
fn recompact_full_adder_rcell() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let path = std::env::var("RECOMPACT_SOURCE")?;
    let document: crate::physical_cell::PhysicalCellDocument =
        std::fs::read_to_string(&path)?.parse()?;
    let mut layout = ExactLayout::from_rcell(&document)?;
    // Another circuit's cell keeps its own output names and has no face policy.
    let circuit = std::env::var("RECOMPACT_CIRCUIT").ok();
    if circuit.is_none() {
        for (name, _) in layout.outputs.iter_mut() {
            if name == "sum" {
                *name = "s".to_owned();
            }
        }
    }
    let graph = match &circuit {
        Some(circuit) => circuit_graph(circuit)?,
        None => full_adder_graph("nor9"),
    };
    let placer = ExactLocalPlacer::new(&graph)?.with_name(format!(
        "exact-{}",
        circuit.as_deref().unwrap_or("full-adder")
    ));
    let compaction = CompactionConfig {
        workers: env_usize("RECOMPACT_WORKERS", 8),
        window_radius: 1,
        max_window_radius: env_usize("RECOMPACT_RADIUS", 2),
        attempt_time_limit: Duration::from_secs(env_usize("RECOMPACT_ATTEMPT_SECONDS", 20) as u64),
        time_limit: Some(Duration::from_secs(
            env_usize("RECOMPACT_SECONDS", 1200) as u64
        )),
        output_policies: match circuit {
            Some(_) => BTreeMap::new(),
            None => [("s".to_owned(), OutputPolicy::MaxYFace)]
                .into_iter()
                .collect(),
        },
        continue_after_gain: std::env::var("RECOMPACT_CONTINUE").as_deref() != Ok("0"),
        given_outside_signals: std::env::var("RECOMPACT_GIVEN").as_deref() != Ok("0"),
        repair_optimize: repair_optimize_from_env("RECOMPACT_REPAIR_OPTIMIZE"),
        max_removals_per_round: removals_per_round_from_env("RECOMPACT_REMOVALS_PER_ROUND"),
        repeat_rounds: std::env::var("RECOMPACT_ROUNDS").as_deref() != Ok("0"),
        reduction_axes: match std::env::var("RECOMPACT_REDUCTION_AXES").as_deref() {
            Ok("1") => vec![1],
            _ => vec![1, 2],
        },
        ..Default::default()
    };
    println!(
        "RECOMPACT start dim={:?} cells={}",
        layout.dim,
        layout.cells.len()
    );
    let (compacted, best, report) = placer.compact(layout, &compaction)?;
    println!(
        "RECOMPACT done dim={:?} cells={} removed={:?} reductions={} attempts={} elapsed={:?}",
        compacted.dim,
        compacted.cells.len(),
        report.removed,
        report.block_reductions,
        report.attempts,
        report.elapsed
    );
    if let (Some(placement), Ok(prefix)) = (best, std::env::var("RECOMPACT_WRITE")) {
        std::fs::write(format!("{prefix}.rcell"), placement.rcell.to_string())?;
        crate::nbt::NBTRoot::from(&placement.placed.world).save(format!("{prefix}.nbt"));
        println!(
            "RECOMPACT wrote {prefix}.rcell blocks={}",
            placement.block_count
        );
    }
    Ok(())
}

/// Writes the CaDiCaL input for a tiny inverter as DIMACS.
/// `DUMP=<path>`, `COMMENTS=none|legend|explained` (default `explained`).
#[test]
#[ignore = "export; run explicitly with DUMP=<path>"]
fn export_inverter_dimacs() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~a")]))
        .unwrap()
        .with_name("inverter");
    let config = ExactPlacerConfig::new(DimSize(1, 3, 2));
    let comments = match std::env::var("COMMENTS").as_deref() {
        Ok("none") => DimacsComments::None,
        Ok("legend") => DimacsComments::Legend,
        _ => DimacsComments::Explained,
    };
    placer
        .write_dimacs(&config, std::env::var("DUMP").unwrap(), comments)
        .unwrap();
}

#[test]
fn dimacs_comment_levels_only_add_comment_lines() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~a")])).unwrap();
    let config = ExactPlacerConfig::new(DimSize(1, 3, 2));
    let directory = std::env::temp_dir().join(format!("exact-dimacs-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut formulas = Vec::new();
    for (level, comments) in [
        ("none", DimacsComments::None),
        ("legend", DimacsComments::Legend),
        ("explained", DimacsComments::Explained),
    ] {
        let path = directory.join(format!("{level}.cnf"));
        placer.write_dimacs(&config, &path, comments).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let comment_lines = text.lines().filter(|line| line.starts_with('c')).count();
        let formula = text
            .lines()
            .filter(|line| !line.starts_with('c'))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        formulas.push((comments, comment_lines, formula));
    }
    std::fs::remove_dir_all(&directory).unwrap();
    assert_eq!(formulas[0].1, 0);
    assert!(formulas[1].1 > 0 && formulas[2].1 > formulas[1].1);
    assert!(formulas[0].2[0].starts_with("p cnf "));
    assert!(formulas
        .iter()
        .all(|(_, _, formula)| *formula == formulas[0].2));
}

/// Encode time of one compaction window of the 2x8x8 full adder (Y slices
/// 3..6 free, the rest fixed with given signals, optimizing), the shape of
/// every block-reduction attempt:
/// `cargo test --release --lib measure_window_encoding -- --ignored --nocapture`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn measure_window_encoding() {
    let source = include_str!("../../../../../test/full-adder-exact-optimized-2x8x8.rcell");
    let document: crate::physical_cell::PhysicalCellDocument = source.parse().unwrap();
    let mut layout = ExactLayout::from_rcell(&document).unwrap();
    for (name, _) in layout.outputs.iter_mut() {
        if name == "sum" {
            *name = "s".to_owned();
        }
    }
    let placer = ExactLocalPlacer::new(&full_adder_graph("nor9")).unwrap();
    let compaction = CompactionConfig {
        output_policies: [("s".to_owned(), OutputPolicy::MaxYFace)]
            .into_iter()
            .collect(),
        given_outside_signals: std::env::var("ENCODE_GIVEN").as_deref() != Ok("0"),
        ..Default::default()
    };
    if compaction.given_outside_signals {
        placer.read_signals(&mut layout, &compaction);
        assert!(!layout.signals.is_empty());
    }
    let limit = layout.cells.len() - 1;
    let mut config = placer
        .window_config(&layout, 1, (3, 6), Some(limit), &compaction)
        .unwrap();
    config.optimize = true;
    let rounds = env_usize("ENCODE_ROUNDS", 20) as u32;
    let started = std::time::Instant::now();
    let mut encoding = None;
    for _ in 0..rounds {
        encoding = Some(Encoding::build(placer.netlist(), &config).unwrap());
    }
    let elapsed = started.elapsed() / rounds;
    let encoding = encoding.unwrap();
    if let (Some(program), true) = (&encoding.program, std::env::var("ENCODE_RULES").is_ok()) {
        let mut stats = program.rule_stats();
        stats.sort_by_key(|(_, clauses, _)| std::cmp::Reverse(*clauses));
        for (rule, clauses, time) in stats {
            println!("  RULE {time:>8.2?} {clauses:>8} {rule}");
        }
    }
    println!(
        "ENCODE window 2x8x8 y=3..6 given={} vars={} clauses={} literals={} time={elapsed:?}",
        compaction.given_outside_signals,
        encoding.cnf.num_vars(),
        encoding.cnf.clause_count(),
        encoding.cnf.literals().len() - encoding.cnf.clause_count(),
    );
}

/// Optimizes block-reduction windows of the 2x8x8 full adder with and
/// without folding fixed cells, each for `WINDOW_SECONDS` (5), and prints the
/// cost reached: does a smaller CNF change how far a short optimize gets?
/// `cargo test --release --lib compare_window_folding -- --ignored --nocapture`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn compare_window_folding() {
    // `WINDOW_SOURCE=<rcell>` should be a loose layout (a construction
    // result): windows of an optimized cell have nothing left to gain.
    let source = match std::env::var("WINDOW_SOURCE") {
        Ok(path) => std::fs::read_to_string(path).unwrap(),
        Err(_) => {
            include_str!("../../../../../test/full-adder-exact-optimized-2x8x8.rcell").to_owned()
        }
    };
    let document: crate::physical_cell::PhysicalCellDocument = source.parse().unwrap();
    let mut layout = ExactLayout::from_rcell(&document).unwrap();
    for (name, _) in layout.outputs.iter_mut() {
        if name == "sum" {
            *name = "s".to_owned();
        }
    }
    let placer = ExactLocalPlacer::new(&full_adder_graph("nor9")).unwrap();
    let compaction = CompactionConfig {
        output_policies: [("s".to_owned(), OutputPolicy::MaxYFace)]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    placer.read_signals(&mut layout, &compaction);
    let seconds = env_usize("WINDOW_SECONDS", 5) as u64;
    let step = env_usize("WINDOW_STEP", 1);
    for fold in [true, false] {
        let mut costs = Vec::new();
        for low in (0..layout.dim.1.saturating_sub(2)).step_by(step) {
            let mut config = placer
                .window_config(&layout, 1, (low, low + 3), None, &compaction)
                .unwrap();
            config.optimize = true;
            config.time_limit = Some(Duration::from_secs(seconds));
            config.no_fold = !fold;
            let started = std::time::Instant::now();
            let (outcome, stats) = placer.place(&config).unwrap();
            let solved = matches!(outcome, ExactOutcome::Placed(_));
            costs.push(format!(
                "y{low}:{}{}@{:.1}s",
                stats.cost.map_or("-".to_owned(), |c| c.to_string()),
                if stats.optimal {
                    "*"
                } else if solved {
                    ""
                } else {
                    "?"
                },
                started.elapsed().as_secs_f64()
            ));
        }
        println!("FOLD fold={fold} {}", costs.join(" "));
    }
}

/// Placement time of XOR 2x6x4 (4 workers, 60 s) over `FOLD_SEEDS` (6) base
/// seeds, with automatic folding and with the original grounding order: the
/// order change alone shifts each run, so compare the distributions.
/// `cargo test --release --lib compare_xor_folding -- --ignored --nocapture`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn compare_xor_folding() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "a^b")])).unwrap();
    let seeds = env_usize("FOLD_SEEDS", 6) as u32;
    for fold in [true, false] {
        let mut times = Vec::new();
        for seed in 1..=seeds {
            let mut config = ExactPlacerConfig::new(DimSize(2, 6, 4));
            config.workers = env_usize("FOLD_WORKERS", 4);
            config.seed = seed;
            config.time_limit = Some(Duration::from_secs(60));
            config.no_fold = !fold;
            let started = std::time::Instant::now();
            let (outcome, _) = placer.place(&config).unwrap();
            let ok = matches!(outcome, ExactOutcome::Placed(_));
            times.push(format!(
                "{}{:.1}",
                if ok { "" } else { "!" },
                started.elapsed().as_secs_f64()
            ));
        }
        println!("XORFOLD fold={fold} {}", times.join(" "));
    }
}

/// Prints a hash of the CNF for a few fixed problems, to check that a
/// grounder change keeps the solver's input identical:
/// `cargo test --release --lib print_cnf_hashes -- --ignored --nocapture`.
#[test]
#[ignore = "diagnostic; run explicitly with --nocapture"]
fn print_cnf_hashes() {
    use std::hash::{Hash, Hasher};
    for (name, graph, dim) in [
        ("xor 2x6x4", graph(&[("out", "a^b")]), DimSize(2, 6, 4)),
        ("nor 1x5x2", graph(&[("out", "~(a|b)")]), DimSize(1, 5, 2)),
        (
            "full adder 2x14x10",
            full_adder_graph("nor9"),
            DimSize(2, 14, 10),
        ),
    ] {
        let netlist = NorNetlist::from_logic_graph(&graph).unwrap();
        let encoding = Encoding::build(&netlist, &ExactPlacerConfig::new(dim)).unwrap();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        encoding.cnf.literals().hash(&mut hasher);
        println!(
            "CNFHASH {name} {:016x} vars={}",
            hasher.finish(),
            encoding.cnf.num_vars()
        );
    }
}

/// Formula sizes and encode times of the grounded model:
/// `cargo test --release --lib measure_encoders -- --ignored --nocapture`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn measure_encoders() {
    let cases = [
        ("inverter 1x3x2", graph(&[("out", "~a")]), DimSize(1, 3, 2)),
        ("xor 2x6x4", graph(&[("out", "a^b")]), DimSize(2, 6, 4)),
        (
            "full adder 2x14x10",
            full_adder_graph("nor9"),
            DimSize(2, 14, 10),
        ),
    ];
    let only = std::env::var("ENCODE_CASE").ok();
    for (name, graph, dim) in cases {
        if only.as_deref().is_some_and(|only| !name.starts_with(only)) {
            continue;
        }
        let netlist = NorNetlist::from_logic_graph(&graph).unwrap();
        let config = ExactPlacerConfig::new(dim);
        let rounds = env_usize("ENCODE_ROUNDS", 5) as u32;
        let started = std::time::Instant::now();
        let mut encoding = None;
        for _ in 0..rounds {
            encoding = Some(Encoding::build(&netlist, &config).unwrap());
        }
        let elapsed = started.elapsed() / rounds;
        let encoding = encoding.unwrap();
        let started = std::time::Instant::now();
        let mut solver = SatSolver::new(1);
        solver.add_cnf(&encoding.cnf);
        let load = started.elapsed();
        drop(solver);
        if let Some(program) = &encoding.program {
            if std::env::var("ENCODE_RULES").is_ok() {
                // Stats belong to the last encoding only, not to all rounds.
                for (rule, clauses, time) in program.rule_stats() {
                    println!("  RULE {time:>8.2?} {clauses:>8} {rule}");
                }
            }
        }
        println!(
            "ENCODE {name} vars={} clauses={} literals={} relations={} time={elapsed:?} load={load:?}",
            encoding.cnf.num_vars(),
            encoding.cnf.clause_count(),
            encoding.cnf.literals().len() - encoding.cnf.clause_count(),
            encoding.relations.len(),
        );
    }
}

/// Model params and replacement model files change the constraints without
/// touching Rust or the CNF.
#[test]
fn model_params_and_files_change_the_constraints() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~(a|b)")])).unwrap();
    let mut config = ExactPlacerConfig::new(DimSize(1, 4, 2));
    config
        .model_params
        .insert("max_repeaters".to_owned(), rsdsl::IValue::Int(0));
    let placement = expect_placed(&placer, &config);
    assert!(placement
        .cells
        .iter()
        .all(|(_, kind)| !matches!(kind, CellKind::Repeater(_))));

    config
        .model_params
        .insert("max_repaeters".to_owned(), rsdsl::IValue::Int(0));
    let error = placer.place(&config).unwrap_err().to_string();
    assert!(error.contains("max_repaeters"), "{error}");
    config.model_params.clear();

    let directory = std::env::temp_dir().join(format!("exact-model-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("variant.rsdsl");
    let variant = format!(
        "{}\nrule \"이 실험에서는 아무 배치도 허용하지 않음\" {{ require false; }}\n",
        include_str!("exact_placer.rsdsl")
    );
    std::fs::write(&path, variant).unwrap();
    config.model_file = Some(path);
    let (outcome, _) = placer.place(&config).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(matches!(outcome, ExactOutcome::Infeasible), "{outcome:?}");
}

/// Solves a small problem with one worker and dumps every simulator
/// rejection: `EXACT_DUMP_REJECTIONS=<dir> REJECT_CASE=or|nor|xor`.
#[test]
#[ignore = "diagnostic; run explicitly with --nocapture"]
fn dump_rejections() {
    let (graph, dim) = match std::env::var("REJECT_CASE").as_deref() {
        Ok("nor") => (graph(&[("out", "~(a|b)")]), DimSize(1, 4, 2)),
        Ok("xor") => (graph(&[("out", "a^b")]), DimSize(2, 6, 4)),
        Ok("and") => (graph(&[("out", "a&b")]), DimSize(1, 6, 3)),
        _ => (graph(&[("out", "a|b")]), DimSize(1, 5, 3)),
    };
    let placer = ExactLocalPlacer::new(&graph).unwrap();
    let mut config = ExactPlacerConfig::new(dim);
    config.workers = 1;
    config.seed = env_usize("REJECT_SEED", 1) as u32;
    config.max_refinements = env_usize("REJECT_LIMIT", 64);
    config.time_limit = Some(Duration::from_secs(env_usize("SECONDS", 120) as u64));
    let started = std::time::Instant::now();
    let (outcome, stats) = placer.place(&config).unwrap();
    let label = match outcome {
        ExactOutcome::Placed(_) => "placed".to_owned(),
        ExactOutcome::Infeasible => "infeasible".to_owned(),
        ExactOutcome::Unknown { last_rejection } => format!("unknown ({last_rejection:?})"),
    };
    println!(
        "REJECT {label} refinements={} elapsed={:?}",
        stats.refinements,
        started.elapsed()
    );
}

/// Explains one block of a dumped rejection: `EXPLAIN_RCELL=<path>
/// EXPLAIN_AT=x,y,z EXPLAIN_CASE=a=1,b=0,...` prints its final state, whether
/// it burned out, and every trace event that targeted it or its neighbors.
#[test]
#[ignore = "diagnostic; run explicitly with --nocapture"]
fn explain_rejected_block() {
    use crate::world::simulator::Simulator;
    let source = std::fs::read_to_string(std::env::var("EXPLAIN_RCELL").unwrap()).unwrap();
    let source = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("expect"))
        .collect::<Vec<_>>()
        .join("\n");
    let document: crate::physical_cell::PhysicalCellDocument = source.parse().unwrap();
    let build = document.build().unwrap();
    let at = std::env::var("EXPLAIN_AT").unwrap();
    let coords = at
        .split(',')
        .map(|v| v.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let target = crate::world::position::Position(coords[0], coords[1], coords[2]);
    let assignments = std::env::var("EXPLAIN_CASE").unwrap_or_default();
    let mut inputs = Vec::new();
    for assignment in assignments.split(',').filter(|a| !a.is_empty()) {
        let (name, value) = assignment.split_once('=').unwrap();
        let input = document
            .inputs
            .iter()
            .find(|input| input.name == name)
            .unwrap();
        inputs.push((input.position, value == "1"));
    }
    let world = crate::world::World::from(&build.world);
    let mut simulator = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 200_000)
        .map_err(|error| error.message().to_owned())
        .unwrap();
    simulator
        .drive_inputs_with_limits(inputs, 256, 50_000)
        .unwrap();
    println!(
        "EXPLAIN {at}: {:?} burned_out={}",
        simulator.world()[target],
        simulator.is_torch_burned_out(target)
    );
    let mut near = vec![target];
    near.extend(
        target
            .forwards()
            .into_iter()
            .filter(|p| world.size.bound_on(*p)),
    );
    if let Ok(extra) = std::env::var("EXPLAIN_ALSO") {
        for item in extra.split(';') {
            let c = item
                .split(',')
                .map(|v| v.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            near.push(crate::world::position::Position(c[0], c[1], c[2]));
        }
    }
    let limit = env_usize("EXPLAIN_CYCLES", 40);
    for entry in simulator.trace() {
        let position = crate::world::position::Position(
            entry.target_position[0],
            entry.target_position[1],
            entry.target_position[2],
        );
        if near.contains(&position) && entry.cycle <= limit {
            println!(
                "  c{:>4} {:?} {} dir={} before={}",
                entry.cycle,
                entry.target_position,
                entry.event_type,
                entry.direction,
                entry.block_before
            );
        }
    }
}

/// `optimize` returns a layout whose cost is proven minimal: one block fewer
/// is infeasible, and a plain placement is never cheaper.
#[test]
fn optimize_finds_and_proves_the_cheapest_layout() {
    for (graph, dim) in [
        (graph(&[("out", "~a")]), DimSize(1, 4, 2)),
        (graph(&[("out", "~(a|b)")]), DimSize(1, 5, 2)),
    ] {
        let placer = ExactLocalPlacer::new(&graph).unwrap();
        let mut config = ExactPlacerConfig::new(dim);
        config.workers = 2;
        config.time_limit = Some(Duration::from_secs(60));
        let plain = expect_placed(&placer, &config);
        config.optimize = true;
        let (outcome, stats) = placer.place(&config).unwrap();
        let ExactOutcome::Placed(best) = outcome else {
            panic!("expected a placement, got {outcome:?}");
        };
        // The default cost counts non-air blocks, switches included.
        let blocks = best.cells.len() as i64;
        assert_eq!(stats.cost, Some(blocks), "{dim:?}");
        assert!(stats.optimal, "{dim:?}: {stats:?}");
        assert!(blocks <= plain.cells.len() as i64);
        let mut tighter = config.clone();
        tighter.optimize = false;
        tighter.max_blocks = Some(best.cells.len() - 1);
        let (outcome, _) = placer.place(&tighter).unwrap();
        assert!(
            matches!(outcome, ExactOutcome::Infeasible),
            "{dim:?}: {outcome:?}"
        );
    }
}

/// Cost weights come from model params, so preferences change without code.
#[test]
fn optimize_uses_the_model_cost_weights() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~(a|b)")])).unwrap();
    let mut config = ExactPlacerConfig::new(DimSize(1, 5, 2));
    config.workers = 2;
    config.time_limit = Some(Duration::from_secs(60));
    config.optimize = true;
    config
        .model_params
        .insert("torch_cost".to_owned(), rsdsl::IValue::Int(10));
    let (outcome, stats) = placer.place(&config).unwrap();
    let ExactOutcome::Placed(best) = outcome else {
        panic!("expected a placement, got {outcome:?}");
    };
    let torches = best
        .cells
        .iter()
        .filter(|(_, kind)| matches!(kind, CellKind::Torch(_)))
        .count() as i64;
    assert_eq!(stats.cost, Some(best.cells.len() as i64 + 10 * torches));
    assert!(stats.optimal);
    assert_eq!(torches, 1, "a NOR needs exactly one torch");
}

/// Prints how far `optimize` gets: `OPT_CASE=xor|nor SECONDS=120 WORKERS=8`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn measure_optimize() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let (graph, dim) = match std::env::var("OPT_CASE").as_deref() {
        Ok("nor") => (graph(&[("out", "~(a|b)")]), DimSize(1, 5, 2)),
        Ok("and") => (graph(&[("out", "a&b")]), DimSize(2, 4, 3)),
        Ok("xor-small") => (graph(&[("out", "a^b")]), DimSize(2, 5, 3)),
        Ok("nor-wide") => (graph(&[("out", "~(a|b)")]), DimSize(2, 4, 3)),
        _ => (graph(&[("out", "a^b")]), DimSize(2, 6, 4)),
    };
    let placer = ExactLocalPlacer::new(&graph).unwrap();
    let mut config = ExactPlacerConfig::new(dim);
    config.workers = env_usize("WORKERS", 8);
    config.time_limit = Some(Duration::from_secs(env_usize("SECONDS", 120) as u64));
    config.optimize = true;
    config.core_guided = std::env::var("OPT_CORES").as_deref() == Ok("1");
    if let Ok(torches) = std::env::var("OPT_MIN_TORCHES") {
        config.model_params.insert(
            "min_torches".to_owned(),
            rsdsl::IValue::Int(torches.parse().unwrap()),
        );
    }
    if std::env::var("OPT_SYMMETRY").as_deref() == Ok("0") {
        config
            .model_params
            .insert("symmetry_breaking".to_owned(), rsdsl::IValue::Bool(false));
    }
    let started = std::time::Instant::now();
    let (outcome, stats) = placer.place(&config).unwrap();
    println!(
        "OPT {:?} cost={:?} lower={:?} optimal={} improvements={} rejections={} elapsed={:?}",
        dim,
        stats.cost,
        stats.lower_bound,
        stats.optimal,
        stats.improvements,
        stats.refinements,
        started.elapsed()
    );
    if let ExactOutcome::Placed(placement) = outcome {
        println!("{}", placement.rcell);
    }
}

/// A/B of the model's symmetry-breaking rules: time to a first layout (XOR
/// 2x6x4, several seeds) and time to an optimality proof (small boxes):
/// `SEEDS=6 SECONDS=60 PROOF_SECONDS=120`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn compare_symmetry_breaking() {
    let seeds = env_usize("SEEDS", 6) as u32;
    let seconds = env_usize("SECONDS", 60) as u64;
    let proof_seconds = env_usize("PROOF_SECONDS", 120) as u64;
    let with = |config: &mut ExactPlacerConfig, on: bool| {
        config
            .model_params
            .insert("symmetry_breaking".to_owned(), rsdsl::IValue::Bool(on));
    };
    if std::env::var("SKIP_FEASIBLE").is_err() {
        let placer = ExactLocalPlacer::new(&graph(&[("out", "a^b")])).unwrap();
        for on in [false, true] {
            let mut times = Vec::new();
            for seed in 0..seeds {
                let mut config = ExactPlacerConfig::new(DimSize(2, 6, 4));
                config.workers = 4;
                config.seed = 1 + seed * 104_729;
                config.time_limit = Some(Duration::from_secs(seconds));
                with(&mut config, on);
                let started = std::time::Instant::now();
                let (outcome, _) = placer.place(&config).unwrap();
                let label = if matches!(outcome, ExactOutcome::Placed(_)) {
                    "ok"
                } else {
                    "--"
                };
                times.push(format!("{label}{:.1}", started.elapsed().as_secs_f64()));
            }
            println!("SYM feasible xor symmetry={on} {times:?}");
        }
    }
    for (name, assignments, dim) in [
        ("nor-1x5x2", vec![("out", "~(a|b)")], DimSize(1, 5, 2)),
        ("nor-2x4x2", vec![("out", "~(a|b)")], DimSize(2, 4, 2)),
        ("or-2x4x3", vec![("out", "a|b")], DimSize(2, 4, 3)),
        ("and-2x4x3", vec![("out", "a&b")], DimSize(2, 4, 3)),
    ] {
        let placer = ExactLocalPlacer::new(&graph(&assignments)).unwrap();
        for on in [false, true] {
            let mut config = ExactPlacerConfig::new(dim);
            config.workers = 4;
            config.optimize = true;
            config.time_limit = Some(Duration::from_secs(proof_seconds));
            with(&mut config, on);
            let started = std::time::Instant::now();
            let (_, stats) = placer.place(&config).unwrap();
            println!(
                "SYM proof {name} symmetry={on} cost={:?} optimal={} elapsed={:.1}s",
                stats.cost,
                stats.optimal,
                started.elapsed().as_secs_f64()
            );
        }
    }
}

/// Pure core-guided search (one worker running OLL) proves the same optimum,
/// and its lower bound meets the cost.
#[test]
fn core_guided_search_proves_the_optimum() {
    for (graph, dim, expected) in [
        (graph(&[("out", "~a")]), DimSize(1, 4, 2), None),
        (graph(&[("out", "~(a|b)")]), DimSize(1, 5, 2), Some(4)),
    ] {
        let placer = ExactLocalPlacer::new(&graph).unwrap();
        let mut config = ExactPlacerConfig::new(dim);
        config.workers = 1;
        config.optimize = true;
        config.time_limit = Some(Duration::from_secs(60));
        let (_, direct) = placer.place(&config).unwrap();
        config.core_guided = true;
        let (outcome, stats) = placer.place(&config).unwrap();
        assert!(matches!(outcome, ExactOutcome::Placed(_)), "{outcome:?}");
        assert!(stats.optimal, "{dim:?}: {stats:?}");
        assert_eq!(stats.cost, direct.cost, "{dim:?}");
        assert_eq!(stats.lower_bound, stats.cost, "{dim:?}");
        if let Some(expected) = expected {
            assert_eq!(stats.cost, Some(expected));
        }
    }
}

/// The torch lower bound counts NOR gates over the signal vocabulary.
#[test]
fn min_torches_counts_nor_gates() {
    let bound = |assignments: &[(&str, &str)]| {
        let netlist = NorNetlist::from_logic_graph(&graph(assignments)).unwrap();
        let classes = super::encode::vocabulary(&netlist);
        let values = netlist.net_values();
        let function = |net: NetId| {
            values[net]
                .iter()
                .enumerate()
                .fold(0u64, |mask, (case, &on)| mask | (u64::from(on) << case))
        };
        let class_of = |f: u64| classes.iter().position(|c| c.function == f).unwrap();
        let inputs = netlist
            .input_names()
            .iter()
            .map(|name| class_of(function(netlist.input_net(name).unwrap())))
            .collect::<Vec<_>>();
        let targets = netlist
            .outputs
            .iter()
            .map(|(_, net)| class_of(function(*net)))
            .collect::<Vec<_>>();
        super::dsl::min_torches(&classes, &inputs, &targets, 200_000)
    };
    assert_eq!(bound(&[("out", "~a")]), 1);
    assert_eq!(bound(&[("out", "~(a|b)")]), 1);
    assert_eq!(bound(&[("out", "a|b")]), 0);
    assert_eq!(bound(&[("out", "a&b")]), 3);
    let xor = bound(&[("out", "a^b")]);
    println!("MIN_TORCHES xor={xor} full_adder={}", {
        let netlist = NorNetlist::from_logic_graph(&full_adder_graph("nor9")).unwrap();
        let classes = super::encode::vocabulary(&netlist);
        let values = netlist.net_values();
        let function = |net: NetId| {
            values[net]
                .iter()
                .enumerate()
                .fold(0u64, |mask, (case, &on)| mask | (u64::from(on) << case))
        };
        let class_of = |f: u64| classes.iter().position(|c| c.function == f).unwrap();
        let inputs = netlist
            .input_names()
            .iter()
            .map(|name| class_of(function(netlist.input_net(name).unwrap())))
            .collect::<Vec<_>>();
        let targets = netlist
            .outputs
            .iter()
            .map(|(_, net)| class_of(function(*net)))
            .collect::<Vec<_>>();
        let started = std::time::Instant::now();
        let n = super::dsl::min_torches(&classes, &inputs, &targets, 200_000);
        format!("{n} ({:?})", started.elapsed())
    });
    assert!(xor >= 3);
}

/// A/B of the implied torch lower bound (`min_torches`, computed by default)
/// against `min_torches = 0`: first-layout time for XOR 2x6x4 over seeds, and
/// the optimality proof for AND 2x4x3. `SEEDS=6 SECONDS=60`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn compare_torch_lower_bound() {
    let seeds = env_usize("SEEDS", 6) as u32;
    let seconds = env_usize("SECONDS", 60) as u64;
    let off = |config: &mut ExactPlacerConfig, bound: bool| {
        if !bound {
            config
                .model_params
                .insert("min_torches".to_owned(), rsdsl::IValue::Int(0));
        }
    };
    let xor = ExactLocalPlacer::new(&graph(&[("out", "a^b")])).unwrap();
    for bound in [false, true] {
        let mut times = Vec::new();
        for seed in 0..seeds {
            let mut config = ExactPlacerConfig::new(DimSize(2, 6, 4));
            config.workers = 4;
            config.seed = 1 + seed * 104_729;
            config.time_limit = Some(Duration::from_secs(seconds));
            off(&mut config, bound);
            let started = std::time::Instant::now();
            let (outcome, _) = xor.place(&config).unwrap();
            let label = if matches!(outcome, ExactOutcome::Placed(_)) {
                "ok"
            } else {
                "--"
            };
            times.push(format!("{label}{:.1}", started.elapsed().as_secs_f64()));
        }
        println!("BOUND feasible xor bound={bound} {times:?}");
    }
    let and = ExactLocalPlacer::new(&graph(&[("out", "a&b")])).unwrap();
    for bound in [false, true] {
        for rep in 0..2u32 {
            let mut config = ExactPlacerConfig::new(DimSize(2, 4, 3));
            config.workers = 4;
            config.seed = 1 + rep;
            config.optimize = true;
            config.time_limit = Some(Duration::from_secs(120));
            off(&mut config, bound);
            let started = std::time::Instant::now();
            let (_, stats) = and.place(&config).unwrap();
            println!(
                "BOUND proof and bound={bound} rep={rep} cost={:?} optimal={} elapsed={:.1}s",
                stats.cost,
                stats.optimal,
                started.elapsed().as_secs_f64()
            );
        }
    }
}

/// Nets that must cross the seam after each construction step: inputs and
/// gates already placed that a later gate still reads, plus the finished
/// outputs in `kept` (carried to the last slice).
fn live_after_each_step(
    netlist: &NorNetlist,
    order: &[NetId],
    kept: &BTreeSet<NetId>,
) -> Vec<usize> {
    let outputs = kept;
    let mut placed = BTreeSet::new();
    (0..order.len())
        .map(|step| {
            placed.insert(order[step]);
            placed.extend(netlist.nets[order[step]].gate_inputs.iter().copied());
            placed
                .iter()
                .filter(|&&net| {
                    outputs.contains(&net)
                        || order[step + 1..]
                            .iter()
                            .any(|&gate| netlist.nets[gate].gate_inputs.contains(&net))
                })
                .count()
        })
        .collect()
}

/// The small circuits the construction harnesses measure, by name.
fn circuit_graph(circuit: &str) -> eyre::Result<LogicGraph> {
    let (assignments, internal): (Vec<(&str, &str)>, Vec<&str>) = match circuit {
        "half-adder" => (vec![("sum", "a^b"), ("carry", "a&b")], vec![]),
        "adder2" => (
            vec![
                ("s0", "a0^b0"),
                ("c0", "a0&b0"),
                ("s1", "a1^b1^c0"),
                ("c1", "(a1&b1)|(c0&(a1^b1))"),
            ],
            vec!["c0"],
        ),
        // The same 2-bit ripple-carry adder written as NOR gates by hand: a
        // half adder for bit 0 (XNOR from four NORs, then s0 and c0) and the
        // nor9 full adder for bit 1 with c0 as its carry in.
        "adder2-nor" => (
            vec![
                ("h1", "~(a0|b0)"),
                ("h2", "~(a0|h1)"),
                ("h3", "~(b0|h1)"),
                ("h4", "~(h2|h3)"),
                ("s0", "~h4"),
                ("c0", "~(s0|h1)"),
                ("n1", "~(a1|b1)"),
                ("n2", "~(a1|n1)"),
                ("n3", "~(b1|n1)"),
                ("n4", "~(n2|n3)"),
                ("n5", "~(n4|c0)"),
                ("n6", "~(n4|n5)"),
                ("n7", "~(c0|n5)"),
                ("s1", "~(n6|n7)"),
                ("c1", "~(n1|n5)"),
            ],
            vec![
                "h1", "h2", "h3", "h4", "c0", "n1", "n2", "n3", "n4", "n5", "n6", "n7",
            ],
        ),
        "mux4" => (
            vec![("out", "(~s1&~s0&a)|(~s1&s0&b)|(s1&~s0&c)|(s1&s0&d)")],
            vec![],
        ),
        _ => (vec![("out", "(a&~s)|(b&s)")], vec![]),
    };
    if circuit == "full-adder" {
        return Ok(full_adder_graph("nor9"));
    }
    let mut graph = LogicGraph::from_assignments(
        assignments
            .iter()
            .map(|(name, expr)| (name.to_string(), expr.to_string())),
    )?
    .prepare_place()?;
    for name in internal {
        graph.graph.remove_output(name);
    }
    Ok(graph)
}

/// Runs `run` inside a compilation snapshot when the environment variable
/// `variable` names a `.snapshot` directory (its `.rsnap` archive is written
/// beside it), so the placer's frames and the final layout can be played
/// back in the viewer.
fn with_snapshot(
    variable: &str,
    design: &str,
    run: impl FnOnce() -> eyre::Result<crate::output::PlacedWorld>,
) -> eyre::Result<crate::output::PlacedWorld> {
    match std::env::var(variable) {
        Ok(directory) => crate::snapshot::compile_with_snapshot(
            crate::snapshot::SnapshotOptions::new(directory, design),
            run,
        ),
        Err(_) => run(),
    }
}

/// One line per kind of compaction window solve: how many, and where the
/// time went.
fn print_attempt_times(prefix: &str, report: &CompactionReport) {
    for ((phase, outcome), time) in &report.attempt_times {
        println!(
            "{prefix} attempts {phase:<12} {outcome:<16} count={:>4} wall={:>7.1}s encode={:>6.1}s solve={:>7.1}s",
            time.count,
            time.wall.as_secs_f64(),
            time.encode.as_secs_f64(),
            time.solve.as_secs_f64()
        );
    }
}

/// Construction plus compaction for small circuits:
/// `CIRCUIT=mux2|half-adder|adder2|adder2-nor|mux4|full-adder|egraph-full-adder CIRCUIT_WIDTH=2
/// CIRCUIT_HEIGHT=10 CIRCUIT_MAX_WINDOW=4 CIRCUIT_STEP_SECONDS=60 CIRCUIT_RESTART_SECONDS=600
/// CIRCUIT_COMPACT_SECONDS=300
/// CIRCUIT_SEED=1 CIRCUIT_WORKERS=8 CIRCUIT_WRITE=<prefix>`; `CIRCUIT_NETLIST_ONLY=1` stops
/// after printing the NOR netlist and its live-net counts;
/// `CIRCUIT_PROGRESS=<directory>` records every accepted step as a frame for
/// the viewer (`?frames=<directory>/frames.json`); `CIRCUIT_SNAPSHOT=<dir>.snapshot`
/// runs inside a compilation snapshot, whose `.rsnap` holds the frames;
/// `CIRCUIT_TIMING=1` compacts timing first (`CompactionConfig::timing`) and
/// `CIRCUIT_STEP_TIMING=<seconds>` builds it so (`ConstructionConfig::step_timing`).
#[test]
#[ignore = "circuit pipeline measurement; run explicitly with --nocapture"]
fn diagnose_construct_circuit() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let circuit = std::env::var("CIRCUIT").unwrap_or_else(|_| "mux2".to_owned());
    // `egraph-full-adder`: the cheapest netlist the e-graph holds
    // (`Exploration::extract_exact`) within `EGRAPH_DEPTH` (any), with
    // `EGRAPH_OR_COST` (0) per OR node, only 2-input NORs with
    // `EGRAPH_BINARY=1`, wide NORs built from OR nets with `EGRAPH_SPLIT=1`,
    // and NORs of more than `EGRAPH_CHAIN` signals built in stages on one
    // support block (`NorNetlist::chain_wide_gates`).
    let placer = match circuit.as_str() {
        "egraph-full-adder" => {
            let (inputs, outputs) = full_adder_functions();
            let exploration = Exploration::new(&inputs, &outputs, Limits::default())?;
            let options = ExtractOptions {
                max_depth: std::env::var("EGRAPH_DEPTH")
                    .ok()
                    .map(|depth| depth.parse::<usize>())
                    .transpose()?,
                or_cost: env_usize("EGRAPH_OR_COST", 0),
                binary: std::env::var("EGRAPH_BINARY").as_deref() == Ok("1"),
                split_wide: std::env::var("EGRAPH_SPLIT").as_deref() == Ok("1"),
                ..Default::default()
            };
            let extraction = exploration
                .extract_exact(&options)?
                .ok_or_else(|| eyre::eyre!("no netlist for {options:?}"))?;
            match std::env::var("EGRAPH_CHAIN") {
                Ok(fan_in) => ExactLocalPlacer::from_netlist(
                    extraction.netlist.chain_wide_gates(fan_in.parse()?)?,
                ),
                Err(_) => ExactLocalPlacer::from_netlist(extraction.netlist),
            }
        }
        _ => ExactLocalPlacer::new(&circuit_graph(&circuit)?)?,
    }
    .with_name(format!("exact-{circuit}"));
    let netlist = placer.netlist();
    if circuit.starts_with("adder2") {
        // Both adder netlists must add: {c1, s1, s0} = a1a0 + b1b0.
        let names = netlist.input_names();
        let values = netlist.net_values();
        let output = |name: &str| netlist.outputs.iter().find(|(n, _)| n == name).unwrap().1;
        for case in 0..1usize << names.len() {
            let bit = |name: &str| {
                usize::from(case & (1 << names.iter().position(|n| n == name).unwrap()) != 0)
            };
            let sum = bit("a0") + 2 * bit("a1") + bit("b0") + 2 * bit("b1");
            for (name, shift) in [("s0", 0), ("s1", 1), ("c1", 2)] {
                eyre::ensure!(
                    values[output(name)][case] == (sum >> shift & 1 == 1),
                    "{circuit}: {name} is wrong in case {case}"
                );
            }
        }
    }
    println!(
        "CIRCUIT {circuit} inputs={:?} outputs={:?} gates={}",
        netlist.input_names(),
        netlist
            .outputs
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>(),
        netlist.gates().count()
    );
    for (net, driver) in netlist.nets.iter().enumerate().map(|(i, n)| (i, &n.driver)) {
        let kind = match driver {
            NetDriver::Input(_) => continue,
            NetDriver::Gate => "NOR",
            NetDriver::Or => "OR",
        };
        let inputs = netlist.nets[net]
            .gate_inputs
            .iter()
            .map(|&input| netlist.nets[input].name.as_str())
            .collect::<Vec<_>>();
        println!("CIRCUIT gate {} = {kind}{inputs:?}", netlist.nets[net].name);
    }
    let all_outputs = netlist
        .outputs
        .iter()
        .map(|(_, net)| *net)
        .collect::<BTreeSet<_>>();
    for order in [
        GateOrder::NetIndex,
        GateOrder::SmallestConeFirst,
        GateOrder::MinLive,
    ] {
        let gates = construct::gate_order(netlist, order);
        for (label, kept) in [("kept", &all_outputs), ("early", &BTreeSet::new())] {
            let live = live_after_each_step(netlist, &gates, kept);
            println!(
                "CIRCUIT live nets {order:?} outputs {label} {live:?} max={}",
                live.iter().max().unwrap_or(&0)
            );
        }
    }
    if std::env::var("CIRCUIT_NETLIST_ONLY").as_deref() == Ok("1") {
        return Ok(());
    }
    let seed = env_usize("CIRCUIT_SEED", 1) as u32;
    let construction = ConstructionConfig {
        width: env_usize("CIRCUIT_WIDTH", 2),
        height: env_usize("CIRCUIT_HEIGHT", 10),
        step_time_limit: Duration::from_secs(env_usize("CIRCUIT_STEP_SECONDS", 60) as u64),
        restart_after: Some(Duration::from_secs(
            env_usize("CIRCUIT_RESTART_SECONDS", 600) as u64,
        )),
        window: 2,
        max_window: env_usize("CIRCUIT_MAX_WINDOW", 4),
        seed,
        gate_order: gate_order_from_env("CIRCUIT_ORDER"),
        early_outputs: std::env::var("CIRCUIT_EARLY_OUTPUTS").as_deref() != Ok("0"),
        given_frozen_signals: std::env::var("CIRCUIT_GIVEN").as_deref() != Ok("0"),
        step_optimize: step_optimize_from_env("CIRCUIT_STEP_OPTIMIZE"),
        model_params: model_params_from_env("CIRCUIT_MODEL_PARAMS"),
        progress: std::env::var("CIRCUIT_PROGRESS").ok().map(Into::into),
        step_timing: std::env::var("CIRCUIT_STEP_TIMING").ok().map(|seconds| {
            Duration::from_secs(seconds.parse().expect("CIRCUIT_STEP_TIMING seconds"))
        }),
        workers: env_usize("CIRCUIT_WORKERS", 8),
        ..Default::default()
    };
    with_snapshot("CIRCUIT_SNAPSHOT", &circuit, || {
        let (layout, placement, report) = placer.construct(&construction)?;
        println!(
            "CIRCUIT constructed dim={:?} blocks={} delays={:?} seed={} restarts={:?} elapsed={:?}",
            layout.dim,
            placement.block_count,
            placement.delays,
            report.seed,
            report
                .restarts
                .iter()
                .map(|(seed, _)| *seed)
                .collect::<Vec<_>>(),
            report.elapsed
        );
        let compaction = CompactionConfig {
            seed,
            time_limit: Some(Duration::from_secs(
                env_usize("CIRCUIT_COMPACT_SECONDS", 300) as u64,
            )),
            max_removals_per_round: removals_per_round_from_env("CIRCUIT_REMOVALS_PER_ROUND"),
            repair_optimize: repair_optimize_from_env("CIRCUIT_REPAIR_OPTIMIZE"),
            progress: std::env::var("CIRCUIT_PROGRESS").ok().map(Into::into),
            timing: std::env::var("CIRCUIT_TIMING").as_deref() == Ok("1"),
            workers: env_usize("CIRCUIT_WORKERS", 8),
            ..Default::default()
        };
        let (compacted, best, report) = placer.compact(layout, &compaction)?;
        let result = best.unwrap_or(placement);
        let document = result.rcell.to_string();
        let reparsed: crate::physical_cell::PhysicalCellDocument = document.parse()?;
        let build = reparsed.build()?;
        let verification = reparsed.verify(&build)?;
        println!(
            "CIRCUIT compacted dim={:?} blocks={} delays={:?} bounds={:?} removed={:?} delay_reductions={} elapsed={:?} rcell_failures={}",
            compacted.dim,
            result.block_count,
            result.delays,
            depth_bounds(placer.netlist()),
            report.removed,
            report.delay_reductions,
            report.elapsed,
            verification.failures.len()
        );
        print_attempt_times("CIRCUIT", &report);
        if let Ok(prefix) = std::env::var("CIRCUIT_WRITE") {
            std::fs::write(format!("{prefix}.rcell"), document)?;
            crate::nbt::NBTRoot::from(&result.placed.world).save(format!("{prefix}.nbt"));
        }
        assert!(verification.failures.is_empty());
        Ok(result.placed.clone())
    })?;
    Ok(())
}

/// A tile that only passes its carry on: solved once with ghost slices,
/// then chained, the last torch must follow `cin` through every tile.
#[test]
fn carry_tile_chains_its_carry() {
    let mut logic = graph(&[("cin", "~ncin"), ("ncout", "~cin")]);
    logic.graph.remove_output("cin");
    let placer = ExactLocalPlacer::new(&logic).unwrap();
    let carry = CarryTiling {
        input: "ncin".to_owned(),
        output: "ncout".to_owned(),
    };
    let mut config = ExactPlacerConfig::new(DimSize(4, 2, 4));
    config.carry = Some(carry.clone());
    config.workers = 4;
    config.time_limit = Some(Duration::from_secs(60));
    let placement = expect_placed(&placer, &config);
    let tile = ExactLayout::from_placement(config.dim, &placement);
    println!("{}", placement.rcell);
    let chain = assemble_chain(placer.netlist(), &carry, &tile, 3, "carry-chain").unwrap();
    println!("{chain}");
    let build = chain.build().unwrap();
    let verification = chain.verify(&build).unwrap();
    assert_eq!(verification.cases, 2);
    assert!(
        verification.failures.is_empty(),
        "{:?}",
        verification.failures
    );
}

/// The full adder as a carry tile. The carry in arrives inverted (`ncin`,
/// read by the carry-in torch) and the carry out leaves inverted on a block
/// (`ncout`), whose torch is the next tile's carry-in torch.
///
/// The sum is `nor9`'s. The carry is not: `nor9`'s `cout = NOR(n1, n5)`
/// goes through the XNOR `n4`, which is not monotone, so when the operands
/// change the block glitches, the next tile's sum and carry turn the glitch
/// into several, and a 4-bit chain burned a torch out. `ncout = NOR(a, b) |
/// NOR(a, cin) | NOR(b, cin)` (every prime implicant) only falls while the
/// inputs only rise, and passes a glitch on as one glitch at most.
/// `xor_carry` restores `nor9`'s carry for comparison.
pub(super) fn carry_adder_graph(xor_carry: bool) -> LogicGraph {
    let mut assignments = vec![
        ("cin", "~ncin"),
        ("n1", "~(a|b)"),
        ("n2", "~(a|n1)"),
        ("n3", "~(b|n1)"),
        ("n4", "~(n2|n3)"),
        ("n5", "~(n4|cin)"),
        ("n6", "~(n4|n5)"),
        ("n7", "~(cin|n5)"),
        ("s", "~(n6|n7)"),
    ];
    if xor_carry {
        assignments.push(("ncout", "n1|n5"));
    } else {
        assignments.push(("m1", "~(a|cin)"));
        assignments.push(("m2", "~(b|cin)"));
        // `m1 | m2` as a net of its own (an OR is a NOT over a NOR), so
        // construction ORs two terms at a time onto a block.
        assignments.push(("mm", "~(~(m1|m2))"));
        assignments.push(("ncout", "n1|mm"));
    }
    let mut logic = graph(&assignments);
    for name in [
        "cin", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "m1", "m2", "mm",
    ] {
        logic.graph.remove_output(name);
    }
    logic
}

#[test]
fn carry_adder_netlist_keeps_the_carry_terms_apart() {
    let netlist = NorNetlist::from_logic_graph(&carry_adder_graph(false)).unwrap();
    println!("{:?}", netlist.summary());
    let ncout = netlist
        .outputs
        .iter()
        .find(|(name, _)| name == "ncout")
        .unwrap()
        .1;
    // ncout = NOT(NOR(n1, mm)), mm = NOT(NOR(m1, m2)).
    let [inner] = netlist.nets[ncout].gate_inputs[..] else {
        panic!("ncout is a NOT");
    };
    assert_eq!(netlist.nets[inner].gate_inputs.len(), 2);
}

/// The carry interface of `carry_adder_graph`.
fn adder_carry() -> CarryTiling {
    CarryTiling {
        input: "ncin".to_owned(),
        output: "ncout".to_owned(),
    }
}

/// Writes a cell to `directory` as `<name>.rcell`, its settled world as
/// `<name>.nbt`, and its interface metadata (inputs and outputs, for the
/// viewer) as `<name>.outputs.json`.
fn write_cell(
    directory: &std::path::Path,
    name: &str,
    document: &crate::physical_cell::PhysicalCellDocument,
    build: &crate::physical_cell::PhysicalCellBuild,
) -> eyre::Result<()> {
    std::fs::create_dir_all(directory)?;
    let path = directory.join(name);
    std::fs::write(path.with_extension("rcell"), document.to_string())?;
    crate::nbt::NBTRoot::from(&document.export_world(build)).save(path.with_extension("nbt"));
    std::fs::write(
        path.with_extension("outputs.json"),
        serde_json::to_string_pretty(&document.interface_json())?,
    )?;
    println!("TILE wrote {}.{{rcell,nbt,outputs.json}}", path.display());
    Ok(())
}

/// File names of the kept cells: the tile by its own size (without the ghost
/// slices), a chain by its bits and box.
fn tile_file_name(size: DimSize) -> String {
    format!(
        "adder-carry-tile-{}x{}x{}",
        size.0 - CarryTiling::GHOST,
        size.1,
        size.2
    )
}

fn chain_file_name(bits: usize, size: DimSize) -> String {
    format!("adder-carry-chain{bits}-{}x{}x{}", size.0, size.1, size.2)
}

/// One run from the netlist to the files: builds the full-adder carry tile
/// (`docs/carry_tiles.md`), compacts it, chains it, checks every chain of
/// 1..=`TILE_BITS` bits in every case and the `TILE_LONG_BITS` chains with
/// sampled cases and a random input walk, and writes the tile and the
/// `TILE_WRITE_BITS` chains to `TILE_OUT` (`.rcell`, settled `.nbt`,
/// `.outputs.json`). With the defaults it takes about an hour, most of it
/// compaction (construction alone gives a valid but large tile in minutes).
///
/// Knobs: `TILE_OUT` (`target/carry-adder`), `TILE_BITS` (4),
/// `TILE_LONG_BITS` (`8,16`; empty for none), `TILE_WRITE_BITS` (`4`),
/// `TILE_SAMPLES` (200), `TILE_WALK` (200), `TILE_COMPACT_SECONDS` (2400),
/// `TILE_TIMING=1` (timing-first compaction, `CompactionConfig::timing`)
/// with `TILE_DELAY_OUTPUTS` (`ncout,s`), `TILE_DELAY_WINDOW` (3) and
/// `TILE_MAX_DELAY_WINDOW` (5), `TILE_STEP_TIMING=<seconds>`
/// (`ConstructionConfig::step_timing`),
/// `TILE_SOURCE=<tile .rcell>` (compact and check that tile instead of
/// constructing one), `TILE_PROGRESS=<directory>` (every accepted step as a
/// frame for the viewer), `TILE_SNAPSHOT=<dir>.snapshot` (run inside a
/// compilation snapshot, whose `.rsnap` holds the frames); construction: `TILE_WIDTH` (2), `TILE_HEIGHT` (10),
/// `TILE_WINDOW`, `TILE_MAX_WINDOW`, `TILE_STEP_SECONDS`,
/// `TILE_CARRY_SECONDS` (180, the carry-out step), `TILE_RESTART_SECONDS`
/// (600), `TILE_RESTARTS` (7), `TILE_SEED`, `TILE_WORKERS`; experiments:
/// `TILE_CARRY=xor` (`nor9`'s carry), `TILE_NETLIST=carry` (the carry
/// alone), `TILE_MODEL_PARAMS`.
#[test]
#[ignore = "builds the carry tile and its chains; run explicitly with --nocapture"]
fn synthesize_carry_adder() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let xor_carry = std::env::var("TILE_CARRY").as_deref() == Ok("xor");
    // `TILE_NETLIST=carry`: the monotone carry alone, no sum.
    let carry_only = std::env::var("TILE_NETLIST").as_deref() == Ok("carry");
    let logic = if carry_only {
        let mut logic = graph(&[
            ("cin", "~ncin"),
            ("n1", "~(a|b)"),
            ("m1", "~(a|cin)"),
            ("m2", "~(b|cin)"),
            ("ncout", "n1|m1|m2"),
        ]);
        for name in ["cin", "n1", "m1", "m2"] {
            logic.graph.remove_output(name);
        }
        logic
    } else {
        carry_adder_graph(xor_carry)
    };
    let placer = ExactLocalPlacer::new(&logic)?.with_name("exact-adder-carry-tile");
    let carry = adder_carry();
    let out = std::path::PathBuf::from(
        std::env::var("TILE_OUT").unwrap_or_else(|_| "target/carry-adder".to_owned()),
    );
    let width = env_usize("TILE_WIDTH", 2);
    let input_policies = [
        ("a".to_owned(), InputPolicy::MinYFace),
        ("b".to_owned(), InputPolicy::MinYFace),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    let output_policies = [("s".to_owned(), OutputPolicy::MaxYFace)]
        .into_iter()
        .filter(|_| !carry_only)
        .collect::<BTreeMap<_, _>>();
    // `nor9`'s carry is not monotone; without this its tile cannot be placed.
    let mut model_params = if xor_carry {
        [("monotone_signals".to_owned(), rsdsl::IValue::Bool(false))]
            .into_iter()
            .collect()
    } else {
        BTreeMap::new()
    };
    model_params.extend(model_params_from_env("TILE_MODEL_PARAMS"));
    let construction = ConstructionConfig {
        width: CarryTiling::GHOST + width,
        model_params: model_params.clone(),
        height: env_usize("TILE_HEIGHT", 10),
        window: env_usize("TILE_WINDOW", 2),
        max_window: env_usize("TILE_MAX_WINDOW", 5),
        step_time_limit: Duration::from_secs(env_usize("TILE_STEP_SECONDS", 60) as u64),
        carry_step_time_limit: Some(Duration::from_secs(
            env_usize("TILE_CARRY_SECONDS", 180) as u64
        )),
        workers: env_usize("TILE_WORKERS", 8),
        seed: env_usize("TILE_SEED", 1) as u32,
        max_restarts: env_usize("TILE_RESTARTS", 7),
        restart_after: Some(Duration::from_secs(
            env_usize("TILE_RESTART_SECONDS", 600) as u64
        )),
        input_policies,
        output_policies: output_policies.clone(),
        carry: Some(carry.clone()),
        progress: std::env::var("TILE_PROGRESS").ok().map(Into::into),
        step_timing: std::env::var("TILE_STEP_TIMING")
            .ok()
            .map(|seconds| Duration::from_secs(seconds.parse().expect("TILE_STEP_TIMING seconds"))),
        ..Default::default()
    };
    let compaction = CompactionConfig {
        progress: std::env::var("TILE_PROGRESS").ok().map(Into::into),
        workers: env_usize("TILE_WORKERS", 8),
        seed: env_usize("TILE_SEED", 1) as u32,
        time_limit: Some(Duration::from_secs(
            env_usize("TILE_COMPACT_SECONDS", 2400) as u64
        )),
        output_policies,
        carry: Some(carry.clone()),
        model_params,
        timing: std::env::var("TILE_TIMING").as_deref() == Ok("1"),
        delay_outputs: std::env::var("TILE_DELAY_OUTPUTS")
            .unwrap_or_else(|_| "ncout,s".to_owned())
            .split(',')
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect(),
        delay_window: env_usize("TILE_DELAY_WINDOW", 3),
        max_delay_window: env_usize("TILE_MAX_DELAY_WINDOW", 5),
        ..Default::default()
    };
    with_snapshot("TILE_SNAPSHOT", "adder-carry-tile", || {
        let started = std::time::Instant::now();
        let (layout, placement) = if let Ok(source) = std::env::var("TILE_SOURCE") {
            // Compact an earlier tile further instead of constructing one.
            let document: crate::physical_cell::PhysicalCellDocument =
                std::fs::read_to_string(source)?.parse()?;
            let layout = ExactLayout::from_rcell(&document)?;
            let exact = placer.window_config(&layout, 1, (0, 0), None, &compaction)?;
            let ExactOutcome::Placed(placement) = placer.place(&exact)?.0 else {
                eyre::bail!("the source tile does not verify under the model");
            };
            (layout, *placement)
        } else {
            let (layout, placement, report) = placer.construct(&construction)?;
            println!(
                "TILE constructed dim={:?} blocks={} seed={} restarts={} steps={:?} elapsed={:?}",
                layout.dim,
                placement.block_count,
                report.seed,
                report.restarts.len(),
                report.steps,
                report.elapsed
            );
            (layout, placement)
        };
        println!(
            "TILE start dim={:?} blocks={} delays={:?}",
            layout.dim,
            placement.block_count,
            layout.timing()?.outputs
        );
        let (compacted, best, report) = placer.compact(layout, &compaction)?;
        print_attempt_times("TILE", &report);
        let tile = best.unwrap_or(placement);
        println!(
            "TILE compacted dim={:?} blocks={} delays={:?} removed={:?} reductions={} delay_reductions={} elapsed={:?}",
            compacted.dim,
            tile.block_count,
            compacted.timing()?.outputs,
            report.removed,
            report.block_reductions,
            report.delay_reductions,
            report.elapsed
        );
        println!("{}", tile.rcell);
        let tile_build = tile.rcell.build()?;
        write_cell(
            &out,
            &tile_file_name(tile.rcell.size),
            &tile.rcell,
            &tile_build,
        )?;

        let bit_list = |name: &str, default: &str| {
            std::env::var(name)
                .unwrap_or_else(|_| default.to_owned())
                .split(',')
                .filter(|bits| !bits.is_empty())
                .map(|bits| bits.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
        };
        let write_bits = bit_list("TILE_WRITE_BITS", "4")?;
        let long_bits = if carry_only {
            Vec::new()
        } else {
            bit_list("TILE_LONG_BITS", "8,16")?
        };
        let exhaustive_bits = env_usize("TILE_BITS", 4);
        let mut failures = Vec::new();
        let all_bits = (1..=exhaustive_bits)
            .chain(long_bits.iter().copied())
            .chain(write_bits.iter().copied())
            .collect::<BTreeSet<_>>();
        for bits in all_bits {
            let chain = assemble_chain(
                placer.netlist(),
                &carry,
                &compacted,
                bits,
                &format!("exact-adder-carry-chain{bits}"),
            )?;
            let build = chain.build()?;
            let timing = ExactLayout::from_rcell(&chain)?.timing()?;
            println!(
                "TILE chain bits={bits} ticks cout={:?} slowest={:?}",
                timing.output("cout"),
                timing.critical()
            );
            if bits <= exhaustive_bits {
                let verification = chain.verify(&build)?;
                println!(
                    "TILE chain bits={bits} dim={:?} cases={} failures={}",
                    chain.size,
                    verification.cases,
                    verification.failures.len()
                );
                if let Some(failure) = verification.failures.first() {
                    failures.push(format!("{bits} bits: {:?}", failure.inputs));
                }
            }
            if long_bits.contains(&bits) {
                for failure in sample_adder_chain(
                    &build,
                    bits,
                    env_usize("TILE_SAMPLES", 200),
                    env_usize("TILE_WALK", 200),
                    env_usize("TILE_SEED", 1) as u64,
                )? {
                    failures.push(format!("{bits} bits: {failure}"));
                }
            }
            if write_bits.contains(&bits) {
                write_cell(&out, &chain_file_name(bits, chain.size), &chain, &build)?;
            }
        }
        println!(
            "TILE done in {:?}: tile {:?} with {} blocks, {} failures",
            started.elapsed(),
            compacted.dim,
            tile.block_count,
            failures.len()
        );
        for failure in failures.iter().take(5) {
            println!("  {failure}");
        }
        eyre::ensure!(failures.is_empty(), "a chain fails");
        Ok(tile.placed.clone())
    })?;
    Ok(())
}

/// Chains a full-adder tile RCELL (from `synthesize_carry_adder`) for
/// 1..=`TILE_BITS` bits and reports each chain's failing cases:
/// `TILE_SOURCE=<tile .rcell>`, `TILE_BITS` (6), `TILE_CARRY=xor` (a tile
/// built with `nor9`'s carry), `TILE_OUT=<directory>` (writes the chains).
#[test]
#[ignore = "carry chain check of a tile file; run explicitly with --nocapture"]
fn check_carry_adder_chain() -> eyre::Result<()> {
    let source = std::fs::read_to_string(std::env::var("TILE_SOURCE")?)?;
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let tile = ExactLayout::from_rcell(&document)?;
    let xor_carry = std::env::var("TILE_CARRY").as_deref() == Ok("xor");
    let placer = ExactLocalPlacer::new(&carry_adder_graph(xor_carry))?;
    for bits in 1..=env_usize("TILE_BITS", 6) {
        let chain = assemble_chain(
            placer.netlist(),
            &adder_carry(),
            &tile,
            bits,
            "exact-adder-chain",
        )?;
        let build = chain.build()?;
        let verification = chain.verify(&build)?;
        println!(
            "CHAIN bits={bits} cases={} failures={} first={:?}",
            verification.cases,
            verification.failures.len(),
            verification.failures.first().map(|failure| &failure.inputs)
        );
        if let Ok(directory) = std::env::var("TILE_OUT") {
            write_cell(
                std::path::Path::new(&directory),
                &chain_file_name(bits, chain.size),
                &chain,
                &build,
            )?;
        }
    }
    Ok(())
}

/// Places only the monotone carry of `carry_adder_graph` (`n1`, `m1`, `m2`
/// and the carry-out block) as a tile, in one solve, to tell a hard carry
/// step from an impossible one: `CARRY_DIM` (4x5x5, ghost slices included),
/// `CARRY_SECONDS` (120), `CARRY_WORKERS` (8), `CARRY_MODEL_PARAMS`
/// (for example `seam_isolation=false`).
#[test]
#[ignore = "carry tile feasibility probe; run explicitly with --nocapture"]
fn diagnose_carry_only_tile() -> eyre::Result<()> {
    let mut logic = graph(&[
        ("cin", "~ncin"),
        ("n1", "~(a|b)"),
        ("m1", "~(a|cin)"),
        ("m2", "~(b|cin)"),
        ("ncout", "n1|m1|m2"),
    ]);
    for name in ["cin", "n1", "m1", "m2"] {
        logic.graph.remove_output(name);
    }
    let placer = ExactLocalPlacer::new(&logic)?.with_name("carry-only-tile");
    let dim_text = std::env::var("CARRY_DIM").unwrap_or_else(|_| "4x5x5".to_owned());
    let parts = dim_text
        .split('x')
        .map(|part| part.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let mut config = ExactPlacerConfig::new(DimSize(parts[0], parts[1], parts[2]));
    config.carry = Some(CarryTiling {
        input: "ncin".to_owned(),
        output: "ncout".to_owned(),
    });
    config.workers = env_usize("CARRY_WORKERS", 8);
    config.time_limit = Some(Duration::from_secs(env_usize("CARRY_SECONDS", 120) as u64));
    config.model_params = model_params_from_env("CARRY_MODEL_PARAMS");
    let started = std::time::Instant::now();
    let (outcome, stats) = placer.place(&config)?;
    match outcome {
        ExactOutcome::Placed(placement) => {
            println!(
                "CARRY placed blocks={} in {:?}",
                placement.block_count,
                started.elapsed()
            );
            println!("{}", placement.rcell);
        }
        other => println!(
            "CARRY {other:?} in {:?} vars={} clauses={}",
            started.elapsed(),
            stats.variables,
            stats.clauses
        ),
    }
    Ok(())
}

/// The monotone carry as an ordinary cell (switches `a`, `b`, `cin`, output
/// `cout` on a torch), to compare with `diagnose_carry_only_tile`:
/// `CARRY_DIM` (2x6x6), `CARRY_SECONDS` (120), `CARRY_WORKERS` (8);
/// `CARRY_CONSTRUCT=1` builds it gate by gate instead (`CARRY_HEIGHT`, 6).
#[test]
#[ignore = "carry cell probe; run explicitly with --nocapture"]
fn diagnose_monotone_carry_cell() -> eyre::Result<()> {
    let mut logic = graph(&[
        ("n1", "~(a|b)"),
        ("m1", "~(a|cin)"),
        ("m2", "~(b|cin)"),
        ("cout", "~(n1|m1|m2)"),
    ]);
    for name in ["n1", "m1", "m2"] {
        logic.graph.remove_output(name);
    }
    let placer = ExactLocalPlacer::new(&logic)?.with_name("monotone-carry");
    if std::env::var("CARRY_CONSTRUCT").as_deref() == Ok("1") {
        let (layout, placement, report) = placer.construct(&ConstructionConfig {
            height: env_usize("CARRY_HEIGHT", 6),
            ..Default::default()
        })?;
        println!(
            "CARRY constructed dim={:?} blocks={} steps={:?}",
            layout.dim, placement.block_count, report.steps
        );
        println!("{}", placement.rcell);
        return Ok(());
    }
    let dim_text = std::env::var("CARRY_DIM").unwrap_or_else(|_| "2x6x6".to_owned());
    let parts = dim_text
        .split('x')
        .map(|part| part.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let mut config = ExactPlacerConfig::new(DimSize(parts[0], parts[1], parts[2]));
    config.workers = env_usize("CARRY_WORKERS", 8);
    config.time_limit = Some(Duration::from_secs(env_usize("CARRY_SECONDS", 120) as u64));
    let started = std::time::Instant::now();
    let (outcome, stats) = placer.place(&config)?;
    match outcome {
        ExactOutcome::Placed(placement) => {
            println!(
                "CARRY placed blocks={} in {:?}",
                placement.block_count,
                started.elapsed()
            );
            println!("{}", placement.rcell);
        }
        other => println!(
            "CARRY {other:?} in {:?} vars={} clauses={}",
            started.elapsed(),
            stats.variables,
            stats.clauses
        ),
    }
    Ok(())
}

/// What output `name` of an n-bit ripple-carry adder chain shows for
/// `a + b + cin`: `s{i}` the sum bits, `ncout{i}` the complement of the carry
/// out of bit `i`, `cout` the last carry.
fn expected_chain_output(name: &str, bits: usize, a: u64, b: u64, cin: bool) -> Option<bool> {
    let sum = a + b + u64::from(cin);
    if name == "cout" {
        return Some(sum >> bits & 1 == 1);
    }
    if let Some(bit) = name.strip_prefix("ncout") {
        let bit = bit.parse::<usize>().ok()?;
        let low = (1u64 << (bit + 1)) - 1;
        let carry = ((a & low) + (b & low) + u64::from(cin)) >> (bit + 1) & 1 == 1;
        return Some(!carry);
    }
    let bit = name.strip_prefix('s')?.parse::<usize>().ok()?;
    Some(sum >> bit & 1 == 1)
}

/// Checks an assembled full-adder chain against `a + b + cin` without
/// enumerating every case: worst-case carry patterns (full propagation,
/// generation at every bit, alternation) and `samples` random cases, each
/// from a settled start, then a random walk of `walk` input changes on one
/// simulator (every output after each change, no burned-out torch).
/// Returns the failures.
fn sample_adder_chain(
    build: &crate::physical_cell::PhysicalCellBuild,
    bits: usize,
    samples: usize,
    walk: usize,
    seed: u64,
) -> eyre::Result<Vec<String>> {
    use crate::world::simulator::{Simulator, MANUAL_INPUT_IDLE_CYCLES};
    use crate::world::World;

    let mask = (1u64 << bits) - 1;
    let assignment = |a: u64, b: u64, cin: bool| {
        let mut inputs = BTreeMap::new();
        for bit in 0..bits {
            inputs.insert(format!("a{bit}"), a >> bit & 1 == 1);
            inputs.insert(format!("b{bit}"), b >> bit & 1 == 1);
        }
        inputs.insert("cin".to_owned(), cin);
        inputs
    };
    let check = |simulator: &Simulator, a: u64, b: u64, cin: bool| -> Result<(), String> {
        for (name, position) in build.observations() {
            let expected = expected_chain_output(name, bits, a, b, cin)
                .ok_or_else(|| format!("output {name} is not an adder output"))?;
            if simulator.world()[position].kind.is_powered() != expected {
                return Err(format!("{name} wrong for {a} + {b} + {}", u8::from(cin)));
            }
        }
        for (position, block) in simulator.world().iter_block() {
            if block.kind.is_torch() && simulator.is_torch_burned_out(position) {
                return Err(format!(
                    "torch {position:?} burned out at {a} + {b} + {}",
                    u8::from(cin)
                ));
            }
        }
        Ok(())
    };
    let drive = |simulator: &mut Simulator, inputs: &BTreeMap<String, bool>| -> eyre::Result<()> {
        let contacts = inputs
            .iter()
            .flat_map(|(name, &value)| {
                build.input_contacts[name]
                    .iter()
                    .map(move |position| (*position, value))
            })
            .collect();
        simulator.drive_inputs_with_limits(contacts, 4096, 1_000_000)?;
        Ok(())
    };
    let world = World::from(&build.world);
    let settled = || {
        Simulator::from_settled_with_limits_and_trace(&world, 4096, 1_000_000, 0)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))
    };
    let mut seed = seed | 1;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let alternating = 0x5555_5555_5555_5555 & mask;
    let mut cases = vec![
        (0, mask, false),
        (0, mask, true),
        (mask, mask, false),
        (mask, mask, true),
        (mask, 0, true),
        (1, mask, false),
        (alternating, !alternating & mask, true),
        (alternating, alternating, false),
    ];
    for _ in 0..samples {
        cases.push((random() & mask, random() & mask, random() & 1 == 1));
    }
    let mut failures = Vec::new();
    for &(a, b, cin) in &cases {
        let mut simulator = settled()?;
        drive(&mut simulator, &assignment(a, b, cin))?;
        if let Err(failure) = check(&simulator, a, b, cin) {
            failures.push(failure);
        }
    }
    let mut simulator = settled()?;
    for _ in 0..walk {
        let (a, b, cin) = (random() & mask, random() & mask, random() & 1 == 1);
        simulator.advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)?;
        drive(&mut simulator, &assignment(a, b, cin))?;
        if let Err(failure) = check(&simulator, a, b, cin) {
            failures.push(format!("walk: {failure}"));
            simulator = settled()?;
        }
    }
    println!(
        "CHAIN bits={bits} sampled cases={} walk={walk} failures={}",
        cases.len(),
        failures.len()
    );
    Ok(failures)
}

/// Checks a long chain of a full-adder tile file with `sample_adder_chain`:
/// `TILE_SOURCE=<tile .rcell>`, `TILE_BITS` (8), `TILE_SAMPLES` (200),
/// `TILE_WALK` (200), `TILE_SEED` (1), `TILE_OUT=<directory>` (writes the chain).
#[test]
#[ignore = "long carry chain check of a tile file; run explicitly with --nocapture"]
fn check_long_carry_adder_chain() -> eyre::Result<()> {
    let source = std::fs::read_to_string(std::env::var("TILE_SOURCE")?)?;
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let tile = ExactLayout::from_rcell(&document)?;
    let placer = ExactLocalPlacer::new(&carry_adder_graph(false))?;
    let bits = env_usize("TILE_BITS", 8);
    let chain = assemble_chain(
        placer.netlist(),
        &adder_carry(),
        &tile,
        bits,
        "exact-adder-chain",
    )?;
    let build = chain.build()?;
    if let Ok(directory) = std::env::var("TILE_OUT") {
        write_cell(
            std::path::Path::new(&directory),
            &chain_file_name(bits, chain.size),
            &chain,
            &build,
        )?;
    }
    let failures = sample_adder_chain(
        &build,
        bits,
        env_usize("TILE_SAMPLES", 200),
        env_usize("TILE_WALK", 200),
        env_usize("TILE_SEED", 1) as u64,
    )?;
    for failure in failures.iter().take(5) {
        println!("  {failure}");
    }
    eyre::ensure!(failures.is_empty(), "the chain fails");
    Ok(())
}

/// Prints the torch toggles in one case of a chained full-adder tile:
/// `TILE_SOURCE`, `TILE_BITS` (8), `TILE_CASE=a,b,cin` (85,170,1),
/// `TILE_TRACE_X` (only torches at this X; all by default).
#[test]
#[ignore = "carry chain trace; run explicitly with --nocapture"]
fn trace_carry_adder_chain_case() -> eyre::Result<()> {
    use crate::world::simulator::Simulator;
    use crate::world::World;

    let source = std::fs::read_to_string(std::env::var("TILE_SOURCE")?)?;
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let tile = ExactLayout::from_rcell(&document)?;
    let placer = ExactLocalPlacer::new(&carry_adder_graph(false))?;
    let carry = CarryTiling {
        input: "ncin".to_owned(),
        output: "ncout".to_owned(),
    };
    let bits = env_usize("TILE_BITS", 8);
    let chain = assemble_chain(placer.netlist(), &carry, &tile, bits, "exact-adder-chain")?;
    let build = chain.build()?;
    let case = std::env::var("TILE_CASE").unwrap_or_else(|_| "85,170,1".to_owned());
    let values = case
        .split(',')
        .map(|value| value.parse::<u64>())
        .collect::<Result<Vec<_>, _>>()?;
    let mut simulator = Simulator::from_settled_with_limits_and_trace(
        &World::from(&build.world),
        4096,
        1_000_000,
        0,
    )
    .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
    simulator.set_trace_limit(1_000_000);
    let mut contacts = Vec::new();
    for bit in 0..bits {
        for (name, value) in [("a", values[0]), ("b", values[1])] {
            for position in &build.input_contacts[&format!("{name}{bit}")] {
                contacts.push((*position, value >> bit & 1 == 1));
            }
        }
    }
    for position in &build.input_contacts["cin"] {
        contacts.push((*position, values[2] == 1));
    }
    simulator.drive_inputs_with_limits(contacts, 4096, 1_000_000)?;
    let only_x = std::env::var("TILE_TRACE_X")
        .ok()
        .and_then(|x| x.parse::<usize>().ok());
    // A torch's state as each event reaching it found it; a change between
    // two such events is a toggle.
    let mut toggles = BTreeMap::<[usize; 3], Vec<(usize, String)>>::new();
    for entry in simulator.trace() {
        if !entry.block_before.contains("Torch") {
            continue;
        }
        let states = toggles.entry(entry.target_position).or_default();
        if states
            .last()
            .is_none_or(|(_, before)| *before != entry.block_before)
        {
            states.push((entry.cycle, entry.block_before.clone()));
        }
    }
    for states in toggles.values_mut() {
        states.remove(0);
    }
    for (position, events) in &toggles {
        if only_x.is_some_and(|x| position[0] != x) {
            continue;
        }
        let burned = simulator.is_torch_burned_out(Position(position[0], position[1], position[2]));
        println!(
            "TORCH {position:?} toggles={} burned={burned} {:?}",
            events.len(),
            events
                .iter()
                .map(|(cycle, state)| format!(
                    "{cycle}:{}",
                    if state.contains("true") { "on" } else { "off" }
                ))
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

/// The kept full-adder carry tile still satisfies the model with every cell
/// fixed, and two copies chain into a 2-bit adder.
#[test]
fn carry_adder_tile_fixture_satisfies_the_model_and_chains() -> eyre::Result<()> {
    let source = include_str!("../../../../../test/adder-carry-tile-2x16x10.rcell");
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let tile = ExactLayout::from_rcell(&document)?;
    let placer = ExactLocalPlacer::new(&carry_adder_graph(false))?;
    let carry = CarryTiling {
        input: "ncin".to_owned(),
        output: "ncout".to_owned(),
    };
    let compaction = CompactionConfig {
        workers: 1,
        carry: Some(carry.clone()),
        output_policies: [("s".to_owned(), OutputPolicy::MaxYFace)]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let exact = placer.window_config(&tile, 1, (0, 0), None, &compaction)?;
    assert!(matches!(placer.place(&exact)?.0, ExactOutcome::Placed(_)));
    let chain = assemble_chain(placer.netlist(), &carry, &tile, 2, "chain")?;
    let build = chain.build()?;
    let verification = chain.verify(&build)?;
    assert_eq!(verification.cases, 32);
    assert!(verification.failures.is_empty());
    Ok(())
}

/// A fixture read for timing tests: the placer, the layout with the
/// netlist's output names, and the compaction settings that fix it.
fn timing_fixture(name: &str) -> eyre::Result<(ExactLocalPlacer, ExactLayout, CompactionConfig)> {
    let (source, graph, carry) = match name {
        "full-adder-2x8x8" => (
            include_str!("../../../../../test/full-adder-exact-optimized-2x8x8.rcell"),
            full_adder_graph("nor9"),
            None,
        ),
        "full-adder-2x13x7" => (
            include_str!("../../../../../test/full-adder-exact-2x13x7.rcell"),
            full_adder_graph("nor9"),
            None,
        ),
        "carry-tile-2x16x10" => (
            include_str!("../../../../../test/adder-carry-tile-2x16x10.rcell"),
            carry_adder_graph(false),
            Some(adder_carry()),
        ),
        other => eyre::bail!("no timing fixture `{other}`"),
    };
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let mut layout = ExactLayout::from_rcell(&document)?;
    for (output, _) in layout.outputs.iter_mut() {
        if output == "sum" {
            *output = "s".to_owned();
        }
    }
    let placer = ExactLocalPlacer::new(&graph)?;
    let compaction = CompactionConfig {
        workers: 1,
        carry,
        output_policies: [("s".to_owned(), OutputPolicy::MaxYFace)]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    Ok((placer, layout, compaction))
}

/// The model with `timing` accepts each fixture with every output's delay
/// at the static timing analysis' value and rejects it with any one output a
/// tick sooner, so `timing.rs` mirrors the model's power relations.
#[test]
fn static_timing_matches_the_model() -> eyre::Result<()> {
    for name in [
        "full-adder-2x8x8",
        "full-adder-2x13x7",
        "carry-tile-2x16x10",
    ] {
        let (placer, layout, compaction) = timing_fixture(name)?;
        let timing = analyze_timing(layout.dim, &layout.cells, &layout.outputs)?;
        assert_eq!(timing.outputs.len(), layout.outputs.len());
        let longest = timing.arrival.values().copied().max().unwrap_or(0);
        let mut exact = placer.window_config(&layout, 1, (0, 0), None, &compaction)?;
        exact.timing = true;
        exact.stage_levels = longest + 1;
        exact.output_delays = timing.outputs.iter().cloned().collect();
        let (outcome, _) = placer.place(&exact)?;
        let ExactOutcome::Placed(placement) = outcome else {
            panic!("{name}: the model rejects the layout at its own delays {timing:?}");
        };
        assert_eq!(placement.delays, exact.output_delays, "{name}");
        for (output, ticks) in &timing.outputs {
            let mut sooner = exact.clone();
            sooner.output_delays.insert(output.clone(), ticks - 1);
            let (outcome, _) = placer.place(&sooner)?;
            assert!(
                matches!(outcome, ExactOutcome::Infeasible),
                "{name}: `{output}` in {} ticks should be infeasible",
                ticks - 1
            );
        }
    }
    Ok(())
}

/// Delays of the fixtures, their logic-depth bounds, and each slowest path:
/// `cargo test --release --lib report_fixture_timing -- --ignored --nocapture`.
#[test]
#[ignore = "report; run explicitly with --nocapture"]
fn report_fixture_timing() -> eyre::Result<()> {
    for name in [
        "full-adder-2x8x8",
        "full-adder-2x13x7",
        "carry-tile-2x16x10",
    ] {
        let (placer, layout, _) = timing_fixture(name)?;
        let timing = analyze_timing(layout.dim, &layout.cells, &layout.outputs)?;
        println!(
            "TIMING {name} outputs={:?} bounds={:?}",
            timing.outputs,
            depth_bounds(placer.netlist())
        );
        let show = |timing: &Timing, label: &str| {
            let Some((output, ticks)) = timing.critical() else {
                return;
            };
            let position = layout
                .outputs
                .iter()
                .find(|(name, _)| name == output)
                .unwrap()
                .1;
            let path = timing
                .path(position)
                .into_iter()
                .map(|position| {
                    format!(
                        "{:?}@({},{},{})={}",
                        layout
                            .cells
                            .get(&position)
                            .copied()
                            .unwrap_or(CellKind::Air),
                        position.0,
                        position.1,
                        position.2,
                        timing.arrival[&position]
                    )
                })
                .collect::<Vec<_>>();
            println!("  {label} {output} {ticks}: {}", path.join(" -> "));
        };
        show(&timing, "critical");
        for (input, position, _) in &layout.inputs {
            let from = analyze_timing_from(layout.dim, &layout.cells, &layout.outputs, *position)?;
            println!("  from {input}: {:?}", from.outputs);
            if std::env::var("TIMING_PATHS").is_ok() {
                show(&from, &format!("from {input}:"));
            }
        }
    }
    let chain: crate::physical_cell::PhysicalCellDocument =
        include_str!("../../../../../test/adder-carry-chain4-13x16x10.rcell").parse()?;
    let timing = ExactLayout::from_rcell(&chain)?.timing()?;
    println!(
        "TIMING carry-chain4 cout={:?} slowest={:?}",
        timing.output("cout"),
        timing.critical()
    );
    let (placer, tile, _) = timing_fixture("carry-tile-2x16x10")?;
    for bits in [8, 16] {
        let chain = assemble_chain(placer.netlist(), &adder_carry(), &tile, bits, "chain")?;
        let timing = ExactLayout::from_rcell(&chain)?.timing()?;
        println!(
            "TIMING carry-chain{bits} cout={:?} slowest={:?}",
            timing.output("cout"),
            timing.critical()
        );
    }
    Ok(())
}

/// Minimizing the critical path of a one-solve cell ends at the logic depth
/// or with a proof that the box allows nothing sooner, and never ends slower
/// than an unconstrained layout.
#[test]
fn minimizing_delay_shortens_the_critical_path() -> eyre::Result<()> {
    // XOR is slower to prove: `diagnose_delay_bound` measures it. The
    // inverter must carry its signal the length of the box.
    for (graph, dim) in [
        (graph(&[("out", "~a")]), DimSize(1, 9, 3)),
        (graph(&[("out", "~(~(a|b)|c)")]), DimSize(2, 6, 3)),
    ] {
        let placer = ExactLocalPlacer::new(&graph)?;
        let mut config = ExactPlacerConfig::new(dim);
        config.time_limit = Some(Duration::from_secs(60));
        if dim.0 == 1 {
            config = config
                .with_input_site("a", Position(0, 0, 1), Direction::Bottom)
                .with_output_sites("out", (0..dim.2).map(|z| Position(0, dim.1 - 1, z)));
        }
        let first = expect_placed(&placer, &config);
        let (placement, proven) = placer.place_minimizing_delay(&config)?;
        let placement = placement.expect("a layout");
        let slowest = |delays: &BTreeMap<String, usize>| delays.values().copied().max().unwrap();
        let bound = depth_bounds(placer.netlist()).into_values().max().unwrap();
        println!(
            "DELAY {dim:?} unconstrained={:?} minimized={:?} bound={bound} proven={proven}",
            first.delays, placement.delays
        );
        assert!(proven);
        assert!(slowest(&placement.delays) <= slowest(&first.delays));
        // Both boxes have room for the logic depth.
        assert_eq!(slowest(&placement.delays), bound);
    }
    Ok(())
}

/// One solve of `DELAY_EXPR` (`a^b`) in `DELAY_DIM` (`2,6,4`) with every
/// output at most `DELAY_TICKS` (6) redstone ticks, for `DELAY_SECONDS` (120)
/// on `DELAY_WORKERS` (4): how hard is a delay bound for the solver?
/// `cargo test --release --lib diagnose_delay_bound -- --ignored --nocapture`.
#[test]
#[ignore = "diagnostic; run explicitly with --nocapture"]
fn diagnose_delay_bound() -> eyre::Result<()> {
    let expression = std::env::var("DELAY_EXPR").unwrap_or_else(|_| "a^b".to_owned());
    let dim = std::env::var("DELAY_DIM").unwrap_or_else(|_| "2,6,4".to_owned());
    let dim = dim
        .split(',')
        .map(|value| value.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let placer = ExactLocalPlacer::new(&graph(&[("out", expression.as_str())]))?;
    let mut config = ExactPlacerConfig::new(DimSize(dim[0], dim[1], dim[2]));
    config.workers = env_usize("DELAY_WORKERS", 4);
    config.time_limit = Some(Duration::from_secs(env_usize("DELAY_SECONDS", 120) as u64));
    config.timing = std::env::var("DELAY_TIMING").as_deref() != Ok("0");
    config.stage_levels = env_usize("DELAY_STAGES", config.stage_levels);
    if config.timing {
        config.output_delays = [("out".to_owned(), env_usize("DELAY_TICKS", 6))]
            .into_iter()
            .collect();
    }
    let started = std::time::Instant::now();
    let (outcome, stats) = placer.place(&config)?;
    let result = match &outcome {
        ExactOutcome::Placed(placement) => format!("placed {:?}", placement.delays),
        ExactOutcome::Infeasible => "infeasible".to_owned(),
        ExactOutcome::Unknown { .. } => "unknown".to_owned(),
    };
    println!(
        "DELAYBOUND {expression} {dim:?} ticks={:?} stages={} {result} vars={} clauses={} elapsed={:?}",
        config.output_delays.get("out"),
        config.stage_levels,
        stats.variables,
        stats.clauses,
        started.elapsed()
    );
    if let ExactOutcome::Placed(placement) = outcome {
        println!("{}", placement.rcell);
    }
    Ok(())
}

/// One timing-first compaction window of `WINDOW_SOURCE` (a full adder
/// `.rcell`), Y slices `WINDOW_LOW..WINDOW_LOW+3`, printing the outcome:
/// `cargo test --release --lib diagnose_timing_window -- --ignored --nocapture`.
#[test]
#[ignore = "diagnostic; run explicitly with --nocapture"]
fn diagnose_timing_window() -> eyre::Result<()> {
    let source = std::fs::read_to_string(std::env::var("WINDOW_SOURCE")?)?;
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let mut layout = ExactLayout::from_rcell(&document)?;
    let placer = ExactLocalPlacer::new(&full_adder_graph("nor9"))?;
    let compaction = CompactionConfig {
        timing: std::env::var("WINDOW_TIMING").as_deref() != Ok("0"),
        ..Default::default()
    };
    layout.delays = layout.timing()?.outputs.into_iter().collect();
    if std::env::var("WINDOW_GIVEN").as_deref() == Ok("1") {
        placer.read_signals(&mut layout, &CompactionConfig::default());
    }
    println!(
        "WINDOW delays={:?} outputs={:?} signals={}",
        layout.delays,
        layout.outputs,
        layout.signals.len()
    );
    let low = env_usize("WINDOW_LOW", 3);
    let size = env_usize("WINDOW_SIZE", 3);
    let exact = placer.window_config(&layout, 1, (low, low + size), None, &compaction)?;
    println!(
        "WINDOW timing={} stage_levels={} budgets={:?}",
        exact.timing, exact.stage_levels, exact.output_delays
    );
    let (outcome, stats) = placer.place(&exact)?;
    match outcome {
        ExactOutcome::Placed(placement) => println!("WINDOW placed {:?}", placement.delays),
        other => println!("WINDOW {other:?} {:?}", stats.solve_time),
    }
    Ok(())
}

/// Timing-first compaction never lets an output settle later, and shortens
/// a path that runs through repeaters when a window can.
#[test]
fn timing_compaction_keeps_and_shortens_delays() -> eyre::Result<()> {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~(a|b)")]))?;
    let dim = DimSize(1, 9, 3);
    let mut config = ExactPlacerConfig::new(dim)
        .with_input_site("a", Position(0, 0, 1), Direction::Bottom)
        .with_input_site("b", Position(0, 8, 1), Direction::Bottom)
        .with_output_sites("out", (0..dim.2).map(|z| Position(0, 4, z)));
    config.workers = 2;
    // A slow start: the first seed whose layout routes through repeaters.
    let mut start = None;
    for seed in 1..20 {
        config.seed = seed;
        let placement = expect_placed(&placer, &config);
        if placement.delays["out"] >= 3 {
            start = Some(placement);
            break;
        }
    }
    let start = start.expect("a layout with repeaters on the path");
    let layout = ExactLayout::from_placement(dim, &start);
    let compaction = CompactionConfig {
        workers: 2,
        timing: true,
        axes: Vec::new(),
        minimize_blocks: false,
        time_limit: Some(Duration::from_secs(60)),
        ..Default::default()
    };
    let (compacted, best, report) = placer.compact(layout, &compaction)?;
    let delays = compacted.timing()?.outputs;
    println!(
        "TIMINGCOMPACT start={:?} end={delays:?} reductions={}",
        start.delays, report.delay_reductions
    );
    assert!(best.is_some());
    assert!(delays[0].1 < start.delays["out"]);
    assert_eq!(delays[0].1, 1, "dust both ways into the torch's block");
    Ok(())
}

/// `source` with its `plane` blocks replaced by those of `solved` (same box
/// and glyph legend), keeping its name, comments, pins, and `expect` lines.
fn splice_planes(source: &str, solved: &str) -> eyre::Result<String> {
    let planes = |text: &str| {
        let mut blocks = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find("  plane ") {
            let end = start + rest[start..].find("\n  }\n").expect("closed plane") + 5;
            blocks.push(rest[start..end].to_owned());
            rest = &rest[end..];
        }
        blocks
    };
    let glyphs = |text: &str| {
        text.lines()
            .filter(|line| line.trim_start().starts_with("glyph "))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    eyre::ensure!(glyphs(source) == glyphs(solved), "the glyph legends differ");
    let (old, new) = (planes(source), planes(solved));
    eyre::ensure!(old.len() == new.len(), "the plane counts differ");
    let mut text = source.to_owned();
    for (old, new) in old.iter().zip(&new) {
        text = text.replacen(old.as_str(), new, 1);
    }
    Ok(text)
}

/// Repairs a full-adder `.rcell` that no longer verifies (after a physics
/// fix, say) by re-solving Y windows of `REPAIR_SOURCE`, nearest
/// `REPAIR_CENTER` (the middle) first and from 3 to 6 slices wide, keeping
/// every other cell, the pins, and the box. The first window layout that
/// passes every case and every settled transition, spliced into the source
/// file (`splice_planes`), is written to `REPAIR_OUT`:
/// `cargo test --release --lib repair_full_adder_rcell -- --ignored --nocapture`.
#[test]
#[ignore = "repair tool; run explicitly with --nocapture"]
fn repair_full_adder_rcell() -> eyre::Result<()> {
    let source_path = std::env::var("REPAIR_SOURCE")?;
    let source = std::fs::read_to_string(&source_path)?;
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let mut layout = ExactLayout::from_rcell(&document)?;
    for (output, _) in layout.outputs.iter_mut() {
        if output == "sum" {
            *output = "s".to_owned();
        }
    }
    let placer = ExactLocalPlacer::new(&full_adder_graph("nor9"))?;
    let compaction = CompactionConfig {
        workers: env_usize("REPAIR_WORKERS", 8),
        attempt_time_limit: Duration::from_secs(env_usize("REPAIR_SECONDS", 60) as u64),
        output_policies: [("s".to_owned(), OutputPolicy::MaxYFace)]
            .into_iter()
            .collect(),
        given_outside_signals: false,
        ..Default::default()
    };
    let center = env_usize("REPAIR_CENTER", layout.dim.1 / 2);
    let mut windows = (3..=6usize.min(layout.dim.1))
        .flat_map(|size| (0..=layout.dim.1 - size).map(move |low| (size, low)))
        .collect::<Vec<_>>();
    windows.sort_by_key(|&(size, low)| (size, (low + size / 2).abs_diff(center)));
    for (size, low) in windows {
        let mut exact = placer.window_config(&layout, 1, (low, low + size), None, &compaction)?;
        // A layout that passes every case may still burn a torch out while
        // the inputs change; block it and ask the window for another.
        for _ in 0..env_usize("REPAIR_TRIES", 8) {
            let started = std::time::Instant::now();
            let (outcome, _) = placer.place(&exact)?;
            let ExactOutcome::Placed(placement) = outcome else {
                println!(
                    "REPAIR window y{low}..{} {} in {:?}",
                    low + size,
                    if matches!(outcome, ExactOutcome::Infeasible) {
                        "infeasible"
                    } else {
                        "unknown"
                    },
                    started.elapsed()
                );
                break;
            };
            let text = splice_planes(&source, &placement.rcell.to_string())?;
            let repaired: crate::physical_cell::PhysicalCellDocument = text.parse()?;
            let build = repaired.build()?;
            assert!(repaired.verify(&build)?.failures.is_empty());
            if let Some(failure) = repaired.settled_transition_failure(&build)? {
                println!("REPAIR window y{low}..{} placed but {failure}", low + size);
                exact.blocked.push(
                    placement
                        .cells
                        .iter()
                        .copied()
                        .filter(|(position, _)| position.1 >= low && position.1 < low + size)
                        .collect(),
                );
                continue;
            }
            println!(
                "REPAIR window y{low}..{} blocks {} -> {} delays {:?}",
                low + size,
                layout.block_count(),
                placement.block_count,
                placement.delays
            );
            std::fs::write(std::env::var("REPAIR_OUT")?, text)?;
            return Ok(());
        }
    }
    eyre::bail!("no window repaired {source_path}")
}

/// Placement time of XOR 2x6x4 (8 workers, 60 s, as in
/// `exact_placer_builds_a_verified_xor`) over `XOR_SEEDS` (6) seeds with the
/// built-in model and, given `XOR_MODEL=<path>`, with that model too:
/// `cargo test --release --lib compare_xor_models -- --ignored --nocapture`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn compare_xor_models() {
    let placer = ExactLocalPlacer::new(&graph(&[("out", "a^b")])).unwrap();
    let seeds = env_usize("XOR_SEEDS", 6) as u32;
    let models = [None]
        .into_iter()
        .chain(std::env::var("XOR_MODEL").ok().map(Some))
        .collect::<Vec<_>>();
    for model in models {
        let mut times = Vec::new();
        for seed in 1..=seeds {
            let mut config = ExactPlacerConfig::new(DimSize(2, 6, 4));
            config.workers = 8;
            config.seed = seed;
            config.time_limit = Some(Duration::from_secs(60));
            config.model_file = model.clone().map(Into::into);
            let started = std::time::Instant::now();
            let (outcome, _) = placer.place(&config).unwrap();
            let ok = matches!(outcome, ExactOutcome::Placed(_));
            times.push(format!(
                "{}{:.1}",
                if ok { "" } else { "!" },
                started.elapsed().as_secs_f64()
            ));
        }
        println!(
            "XORMODEL {} {}",
            model.as_deref().unwrap_or("built-in"),
            times.join(" ")
        );
    }
}

/// Delays of any `.rcell` (`TIMING_SOURCE`), and the slowest path to each
/// output with its torches and repeaters counted:
/// `cargo test --release --lib report_rcell_timing -- --ignored --nocapture`.
#[test]
#[ignore = "report; run explicitly with --nocapture"]
fn report_rcell_timing() -> eyre::Result<()> {
    let source = std::fs::read_to_string(std::env::var("TIMING_SOURCE")?)?;
    let document: crate::physical_cell::PhysicalCellDocument = source.parse()?;
    let layout = ExactLayout::from_rcell(&document)?;
    let timing = layout.timing()?;
    let count = |predicate: fn(&CellKind) -> bool| {
        layout.cells.values().filter(|kind| predicate(kind)).count()
    };
    println!(
        "RCELLTIMING blocks={} torches={} repeaters={} dust={} outputs={:?}",
        layout.block_count(),
        count(|kind| matches!(kind, CellKind::Torch(_))),
        count(|kind| matches!(kind, CellKind::Repeater(_))),
        count(|kind| matches!(kind, CellKind::Dust)),
        timing.outputs
    );
    for (output, position) in &layout.outputs {
        let path = timing.path(*position);
        let kinds = path
            .iter()
            .map(|position| layout.cells.get(position).copied().unwrap_or(CellKind::Air))
            .collect::<Vec<_>>();
        println!(
            "RCELLTIMING path {output}: {} torches, {} repeaters, {} cells: {}",
            kinds
                .iter()
                .filter(|kind| matches!(kind, CellKind::Torch(_)))
                .count(),
            kinds
                .iter()
                .filter(|kind| matches!(kind, CellKind::Repeater(_)))
                .count(),
            kinds.len(),
            kinds
                .iter()
                .map(|kind| match kind {
                    CellKind::Torch(_) => "T",
                    CellKind::Repeater(_) => "R",
                    CellKind::Dust => "d",
                    CellKind::Solid => "b",
                    CellKind::Switch(_) => "S",
                    CellKind::Air => ".",
                })
                .collect::<Vec<_>>()
                .join("")
        );
    }
    Ok(())
}

/// The full adder as functions for the e-graph: `(inputs, outputs)`.
fn full_adder_functions() -> (Vec<&'static str>, Vec<(&'static str, &'static str)>) {
    (
        vec!["a", "b", "cin"],
        vec![("s", "a^b^cin"), ("cout", "(a&b)|(a&cin)|(b&cin)")],
    )
}

/// Output functions of a netlist by name, as truth-table rows.
fn output_functions(netlist: &NorNetlist) -> BTreeMap<String, Vec<bool>> {
    let values = netlist.net_values();
    netlist
        .outputs
        .iter()
        .map(|(name, net)| (name.clone(), values[*net].clone()))
        .collect()
}

/// Netlists extracted from the saturated full adder compute the full adder.
/// Exact extraction needs fewer gates than the hand-written `nor9` (8, proven
/// for this e-graph, against 9) and, at the same 9 gates, half its depth;
/// greedy extraction misses the shared NORs and needs more.
#[test]
fn egraph_netlists_compute_the_full_adder() -> eyre::Result<()> {
    let (inputs, outputs) = full_adder_functions();
    let exploration = Exploration::new(&inputs, &outputs, Limits::default())?;
    let nor9 = NorNetlist::from_logic_graph(&full_adder_graph("nor9"))?;
    let expected = output_functions(&nor9);
    let greedy = exploration.extract(Weights {
        gates: 1.0,
        depth: 0.0,
    })?;
    assert_eq!(output_functions(&greedy), expected);
    let fewest = exploration
        .extract_exact(&ExtractOptions::default())?
        .expect("a netlist");
    assert_eq!(output_functions(&fewest.netlist), expected);
    assert!(fewest.proven);
    assert_eq!(fewest.netlist.gates().count(), fewest.cost);
    assert!(fewest.cost < nor9.gates().count());
    let within = |depth| ExtractOptions {
        max_depth: Some(depth),
        ..Default::default()
    };
    let shallow = exploration
        .extract_exact(&within(3))?
        .expect("a netlist within 3 gates");
    assert_eq!(output_functions(&shallow.netlist), expected);
    assert!(netlist_depths(&shallow.netlist)
        .values()
        .all(|&depth| depth <= 3));
    assert!(exploration.extract_exact(&within(2))?.is_none());
    // Split, the 8 gates read at most two signals each, OR nets included.
    let split = exploration
        .extract_exact(&ExtractOptions {
            split_wide: true,
            ..Default::default()
        })?
        .expect("a netlist");
    assert_eq!(output_functions(&split.netlist), expected);
    assert_eq!(split.netlist.gates().count(), fewest.cost);
    assert!(split
        .netlist
        .nets
        .iter()
        .all(|net| net.gate_inputs.len() <= 2));
    assert!(split
        .netlist
        .nets
        .iter()
        .any(|net| net.driver == NetDriver::Or));
    Ok(())
}

/// Chaining wide NORs keeps every function, leaves no NOR wider than two
/// signals, and construction places each chain's stages consecutively,
/// right before the NOR they end in.
#[test]
fn chained_wide_gates_keep_functions_and_stay_together() -> eyre::Result<()> {
    let (inputs, outputs) = full_adder_functions();
    let exploration = Exploration::new(&inputs, &outputs, Limits::default())?;
    let wide = exploration
        .extract_exact(&ExtractOptions::default())?
        .expect("a netlist")
        .netlist;
    assert!(wide
        .gates()
        .any(|gate| wide.nets[gate].gate_inputs.len() > 2));
    let chained = wide.chain_wide_gates(2)?;
    assert_eq!(output_functions(&chained), output_functions(&wide));
    assert_eq!(chained.gates().count(), wide.gates().count());
    assert!(chained.nets.iter().all(|net| net.gate_inputs.len() <= 2));
    let ors = construct::chained_ors(&chained);
    assert_eq!(
        ors.len(),
        chained
            .nets
            .iter()
            .filter(|net| net.driver == NetDriver::Or)
            .count()
    );
    for order in [GateOrder::SmallestConeFirst, GateOrder::MinLive] {
        let placed = construct::gate_order(&chained, order);
        for (index, &net) in placed.iter().enumerate() {
            for input in &chained.nets[net].gate_inputs {
                if !matches!(chained.nets[*input].driver, NetDriver::Input(_)) {
                    assert!(
                        placed[..index].contains(input),
                        "{order:?}: not topological"
                    );
                }
            }
            if ors.contains(&net) {
                let reader = placed[index + 1];
                assert!(
                    chained.nets[reader].gate_inputs.contains(&net),
                    "{order:?}: chain split"
                );
            }
        }
    }
    Ok(())
}

/// Saturates `EGRAPH_CIRCUIT` (`full-adder`) and prints the e-graph's size
/// and, per weighting, the extracted netlist's gates, depths, and live nets
/// beside `nor9`'s:
/// `cargo test --release --lib explore_egraph_netlists -- --ignored --nocapture`.
#[test]
#[ignore = "experiment; run explicitly with --nocapture"]
fn explore_egraph_netlists() -> eyre::Result<()> {
    let (inputs, outputs) = full_adder_functions();
    let limits = Limits {
        iterations: env_usize("EGRAPH_ITERATIONS", 12),
        nodes: env_usize("EGRAPH_NODES", 50_000),
        time: Duration::from_secs(env_usize("EGRAPH_SECONDS", 20) as u64),
    };
    let started = std::time::Instant::now();
    let exploration = Exploration::new(&inputs, &outputs, limits)?;
    println!(
        "EGRAPH classes={} nodes={} iterations={} stop={:?} in {:?}",
        exploration.classes(),
        exploration.nodes(),
        exploration.iterations,
        exploration.stop_reason,
        started.elapsed()
    );
    let describe = |label: &str, netlist: &NorNetlist| {
        let min_live = live_after_each_step(
            netlist,
            &construct::gate_order(netlist, GateOrder::MinLive),
            &BTreeSet::new(),
        );
        println!(
            "EGRAPH {label}: min-live order live_max={} live={min_live:?}",
            min_live.iter().max().unwrap_or(&0)
        );
        let gates = construct::gate_order(netlist, GateOrder::SmallestConeFirst);
        let live = live_after_each_step(netlist, &gates, &BTreeSet::new());
        let fan_in = netlist
            .gates()
            .map(|gate| netlist.nets[gate].gate_inputs.len())
            .max()
            .unwrap_or(0);
        let ors = netlist
            .nets
            .iter()
            .filter(|net| net.driver == NetDriver::Or)
            .count();
        println!(
            "EGRAPH {label}: gates={} or_nets={ors} depths={:?} max_fan_in={fan_in} live_max={} live={live:?}",
            netlist.gates().count(),
            netlist_depths(netlist),
            live.iter().max().unwrap_or(&0)
        );
        for net in gates {
            let inputs = netlist.nets[net]
                .gate_inputs
                .iter()
                .map(|&input| netlist.nets[input].name.as_str())
                .collect::<Vec<_>>();
            let kind = if netlist.nets[net].driver == NetDriver::Or {
                "OR"
            } else {
                "NOR"
            };
            println!("EGRAPH     {} = {kind}{inputs:?}", netlist.nets[net].name);
        }
    };
    describe(
        "nor9",
        &NorNetlist::from_logic_graph(&full_adder_graph("nor9"))?,
    );
    if std::env::var("EGRAPH_GREEDY").is_ok() {
        for (gates, depth) in [(1.0, 0.0), (1.0, 1.0), (0.01, 1.0)] {
            let netlist = exploration.extract(Weights { gates, depth })?;
            describe(&format!("greedy gates*{gates}+depth*{depth}"), &netlist);
        }
    }
    // Exact: the cheapest netlist within each depth, shallowest first, for
    // each OR cost (`EGRAPH_OR_COSTS`, `0`).
    let seconds = Duration::from_secs(env_usize("EGRAPH_EXTRACT_SECONDS", 60) as u64);
    let or_costs = std::env::var("EGRAPH_OR_COSTS")
        .unwrap_or_else(|_| "0".to_owned())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    for or_cost in or_costs {
        for depth in (2..=env_usize("EGRAPH_MAX_DEPTH", 8))
            .map(Some)
            .chain([None])
        {
            let started = std::time::Instant::now();
            let options = ExtractOptions {
                max_depth: depth,
                or_cost,
                binary: std::env::var("EGRAPH_BINARY").as_deref() == Ok("1"),
                split_wide: std::env::var("EGRAPH_SPLIT").as_deref() == Ok("1"),
                time_limit: seconds,
            };
            match exploration.extract_exact(&options)? {
                Some(extraction) => describe(
                    &format!(
                        "exact or_cost={or_cost} depth<={} cost={} proven={} in {:.1?}",
                        depth.map_or("any".to_owned(), |depth| depth.to_string()),
                        extraction.cost,
                        extraction.proven,
                        started.elapsed()
                    ),
                    &extraction.netlist,
                ),
                None => println!(
                    "EGRAPH exact or_cost={or_cost} depth<={depth:?}: none in {:.1?}",
                    started.elapsed()
                ),
            }
        }
    }
    Ok(())
}
