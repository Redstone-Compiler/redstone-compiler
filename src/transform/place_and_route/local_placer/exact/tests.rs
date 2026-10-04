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

/// Both orders place every gate once, after its inputs; smallest-cone-first
/// finishes the small output before starting the larger one's private logic.
#[test]
fn gate_orders_are_topological_and_finish_small_cones_first() {
    let netlist =
        NorNetlist::from_logic_graph(&graph(&[("big", "a^b^c"), ("small", "~(a|b)")])).unwrap();
    let gates = netlist.gates().collect::<BTreeSet<_>>();
    for order in [GateOrder::NetIndex, GateOrder::SmallestConeFirst] {
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
    config.workers = 4;
    config.time_limit = Some(Duration::from_secs(60));
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
    for legacy in [false, true] {
        assert_encoder_accepts_rcell(source, variant, rank_levels, stage_levels, legacy);
    }
}

fn assert_encoder_accepts_rcell(
    source: &str,
    variant: &str,
    rank_levels: usize,
    stage_levels: usize,
    legacy: bool,
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
    config.legacy_encoder = legacy;
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
    // A layout the exact placer compacted with the legacy encoder. (The
    // generated 2x13x7 cell is not checked here: it has a repeater pointing out
    // of the box, legal only with the whole max-Y face as sum sites.)
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
        legacy_encoder: std::env::var("PIPE_LEGACY").as_deref() == Ok("1"),
        max_overlap: env_usize("PIPE_MAX_OVERLAP", 1),
        block_seam: std::env::var("PIPE_BLOCK_SEAM").as_deref() == Ok("1"),
        max_restarts: env_usize("PIPE_RESTARTS", 7),
        max_backtracks: env_usize("PIPE_BACKTRACKS", 0),
        gate_order: gate_order_from_env("PIPE_ORDER"),
        early_outputs: std::env::var("PIPE_EARLY_OUTPUTS").as_deref() != Ok("0"),
        given_frozen_signals: std::env::var("PIPE_GIVEN").as_deref() != Ok("0"),
        step_optimize: step_optimize_from_env("PIPE_STEP_OPTIMIZE"),
        tuning: ExactTuning {
            fold_fixed_cells: std::env::var("PIPE_FOLD").as_deref() != Ok("0"),
            ..Default::default()
        },
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
        legacy_encoder: std::env::var("PIPE_LEGACY").as_deref() == Ok("1"),
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

/// Solves the same problem from both encoders with several seeds and prints
/// formula sizes and solve times: `ENCODER_CASE=xor|nor|inverter SEEDS=4`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn compare_encoders() {
    let (graph, dim) = match std::env::var("ENCODER_CASE").as_deref() {
        Ok("nor") => (graph(&[("out", "~(a|b)")]), DimSize(1, 4, 2)),
        Ok("inverter") => (graph(&[("out", "~a")]), DimSize(1, 3, 2)),
        _ => (graph(&[("out", "a^b")]), DimSize(2, 6, 4)),
    };
    let netlist = NorNetlist::from_logic_graph(&graph).unwrap();
    let seeds = env_usize("SEEDS", 4) as u32;
    let seconds = env_usize("SECONDS", 120) as u64;
    for legacy in [true, false] {
        let mut config = ExactPlacerConfig::new(dim);
        config.legacy_encoder = legacy;
        let started = std::time::Instant::now();
        let encoding = Encoding::build(&netlist, &config).unwrap();
        let encode_time = started.elapsed();
        let mut times = Vec::new();
        for seed in 0..seeds {
            let mut solver = SatSolver::new(1 + seed * 7919);
            solver.add_cnf(&encoding.cnf);
            let stop = std::sync::atomic::AtomicBool::new(false);
            let signal = StopSignal {
                stop: &stop,
                deadline: Some(std::time::Instant::now() + Duration::from_secs(seconds)),
                restart: None,
            };
            let started = std::time::Instant::now();
            let result = solver.solve(&[], &signal);
            times.push(format!(
                "{result:?}@{:.2}s",
                started.elapsed().as_secs_f64()
            ));
        }
        println!(
            "ENCODER legacy={legacy} vars={} clauses={} relations={} encode={encode_time:?} solves={times:?}",
            encoding.cnf.num_vars(),
            encoding.cnf.clause_count(),
            encoding.relations.len(),
        );
    }
}

/// Runs the 4-worker `place` portfolio from both encoders over several base
/// seeds: `ENCODER_CASE=xor SEEDS=4 SECONDS=120`.
#[test]
#[ignore = "measurement; run explicitly with --nocapture"]
fn compare_encoder_portfolios() {
    let (graph, dim) = match std::env::var("ENCODER_CASE").as_deref() {
        Ok("nor") => (graph(&[("out", "~(a|b)")]), DimSize(1, 4, 2)),
        _ => (graph(&[("out", "a^b")]), DimSize(2, 6, 4)),
    };
    let placer = ExactLocalPlacer::new(&graph).unwrap();
    let seeds = env_usize("SEEDS", 4) as u32;
    let seconds = env_usize("SECONDS", 120) as u64;
    for legacy in [true, false] {
        let mut times = Vec::new();
        for seed in 0..seeds {
            let mut config = ExactPlacerConfig::new(dim);
            config.legacy_encoder = legacy;
            config.workers = 4;
            config.seed = 1 + seed * 104_729;
            config.time_limit = Some(Duration::from_secs(seconds));
            let started = std::time::Instant::now();
            let (outcome, stats) = placer.place(&config).unwrap();
            let worker = stats
                .winning_seed
                .map(|seed| (seed.wrapping_sub(config.seed) / 7919).to_string())
                .unwrap_or_default();
            let label = match outcome {
                ExactOutcome::Placed(_) => "placed",
                ExactOutcome::Infeasible => "infeasible",
                ExactOutcome::Unknown { .. } => "unknown",
            };
            times.push(format!(
                "{label}@{:.1}s/w{worker}",
                started.elapsed().as_secs_f64()
            ));
            println!(
                "PORTFOLIO legacy={legacy} seed={seed} {}",
                times.last().unwrap()
            );
        }
        println!("PORTFOLIO legacy={legacy} all={times:?}");
    }
}

/// Each encoder accepts the layouts the other one finds (with the same pins
/// and observation cells), and both agree on an infeasible box.
#[test]
fn encoders_accept_each_others_layouts() {
    let cases = [
        (graph(&[("out", "~a")]), DimSize(1, 3, 2)),
        (graph(&[("out", "~(a|b)")]), DimSize(1, 4, 2)),
        (graph(&[("out", "~(~a)")]), DimSize(1, 5, 3)),
    ];
    for (graph, dim) in cases {
        let placer = ExactLocalPlacer::new(&graph).unwrap();
        for legacy in [true, false] {
            let mut config = ExactPlacerConfig::new(dim);
            config.legacy_encoder = legacy;
            config
                .model_params
                .insert("legacy_semantics".to_owned(), rsdsl::IValue::Bool(true));
            config.time_limit = Some(Duration::from_secs(60));
            let placement = expect_placed(&placer, &config);
            let mut other = config.clone();
            other.legacy_encoder = !legacy;
            let position_of = |[x, y, z]: [usize; 3]| crate::world::position::Position(x, y, z);
            for endpoint in &placement.placed.inputs {
                let position = position_of(endpoint.position);
                let kind = placement
                    .cells
                    .iter()
                    .find(|(p, _)| *p == position)
                    .map(|(_, kind)| *kind)
                    .unwrap();
                let CellKind::Switch(attach) = kind else {
                    panic!("input {} is not a switch: {kind:?}", endpoint.name);
                };
                other
                    .input_sites
                    .insert(endpoint.name.clone(), vec![(position, attach)]);
            }
            for endpoint in &placement.placed.outputs {
                other
                    .output_sites
                    .insert(endpoint.name.clone(), vec![position_of(endpoint.position)]);
            }
            let encoding = Encoding::build(placer.netlist(), &other).unwrap();
            let assumptions = placement
                .cells
                .iter()
                .map(|&(position, kind)| {
                    encoding
                        .kind_lit(encoding.geometry.index(position), kind)
                        .unwrap_or_else(|| panic!("{position:?} cannot hold {kind:?}"))
                })
                .collect::<Vec<_>>();
            let mut solver = SatSolver::new(1);
            solver.add_cnf(&encoding.cnf);
            let stop = std::sync::atomic::AtomicBool::new(false);
            let signal = StopSignal {
                stop: &stop,
                deadline: Some(std::time::Instant::now() + Duration::from_secs(60)),
                restart: None,
            };
            assert_eq!(
                solver.solve(&assumptions, &signal),
                SolveResult::Sat,
                "legacy={} rejects the layout found with legacy={legacy} in {dim:?}",
                !legacy
            );
        }
    }
    let placer = ExactLocalPlacer::new(&graph(&[("out", "~(a|b)")])).unwrap();
    for legacy in [true, false] {
        let mut config = ExactPlacerConfig::new(DimSize(1, 2, 1));
        config.legacy_encoder = legacy;
        let (outcome, _) = placer.place(&config).unwrap();
        assert!(
            matches!(outcome, ExactOutcome::Infeasible),
            "legacy={legacy}: {outcome:?}"
        );
    }
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
            config.tuning.fold_fixed_cells = fold;
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

/// Formula sizes and encode times of both encoders:
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
    let encoders = match std::env::var("ENCODE_ONLY").as_deref() {
        Ok("dsl") => vec![false],
        Ok("legacy") => vec![true],
        _ => vec![true, false],
    };
    for (name, graph, dim) in cases {
        if only.as_deref().is_some_and(|only| !name.starts_with(only)) {
            continue;
        }
        let netlist = NorNetlist::from_logic_graph(&graph).unwrap();
        for &legacy in &encoders {
            let mut config = ExactPlacerConfig::new(dim);
            config.legacy_encoder = legacy;
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
                "ENCODE {name} legacy={legacy} vars={} clauses={} literals={} relations={} time={elapsed:?} load={load:?}",
                encoding.cnf.num_vars(),
                encoding.cnf.clause_count(),
                encoding.cnf.literals().len() - encoding.cnf.clause_count(),
                encoding.relations.len(),
            );
        }
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

/// Every block layout one encoder admits (switches named with their input),
/// by incremental enumeration with blocking clauses over the kind literals.
fn admitted_layouts(
    encoding: &Encoding,
    netlist: &NorNetlist,
    limit: usize,
) -> Option<std::collections::BTreeSet<Vec<String>>> {
    use super::encode::{CARDINALS, TORCH_ATTACH};
    let geometry = encoding.geometry;
    let mut options = Vec::new();
    for cell in 0..geometry.len() {
        let mut kinds = vec![CellKind::Air, CellKind::Solid, CellKind::Dust];
        kinds.extend(TORCH_ATTACH.map(CellKind::Torch));
        kinds.extend(CARDINALS.map(CellKind::Repeater));
        let mut cell_options = kinds
            .into_iter()
            .filter_map(|kind| {
                encoding
                    .kind_lit(cell, kind)
                    .map(|lit| (format!("{kind:?}"), lit))
            })
            .collect::<Vec<_>>();
        for site in encoding.switches.iter().filter(|site| site.cell == cell) {
            let label = format!("Switch({:?})/{}", site.attach, netlist.nets[site.net].name);
            cell_options.push((label, site.lit));
        }
        options.push(cell_options);
    }
    let mut solver = SatSolver::new(1);
    solver.add_cnf(&encoding.cnf);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let signal = StopSignal {
        stop: &stop,
        deadline: None,
        restart: None,
    };
    let mut layouts = std::collections::BTreeSet::new();
    while solver.solve(&[], &signal) == SolveResult::Sat {
        let chosen = options
            .iter()
            .map(|cell| {
                let true_options = cell
                    .iter()
                    .filter(|(_, lit)| solver.value(*lit))
                    .collect::<Vec<_>>();
                assert_eq!(true_options.len(), 1, "{cell:?}");
                true_options[0].clone()
            })
            .collect::<Vec<_>>();
        solver.add_clause(&chosen.iter().map(|(_, lit)| -lit).collect::<Vec<_>>());
        layouts.insert(chosen.into_iter().map(|(label, _)| label).collect());
        if layouts.len() > limit {
            return None;
        }
    }
    Some(layouts)
}

/// Both encoders admit exactly the same block layouts on small boxes.
#[test]
fn encoders_admit_the_same_layouts() {
    let cases = [
        (graph(&[("out", "~a")]), DimSize(1, 3, 2), None),
        (graph(&[("out", "~a")]), DimSize(1, 4, 2), None),
        (graph(&[("out", "~(a|b)")]), DimSize(1, 4, 2), None),
        (graph(&[("out", "~(a|b)")]), DimSize(1, 5, 2), Some(9)),
        (graph(&[("out", "~a")]), DimSize(2, 2, 2), None),
        (graph(&[("out", "~(a|b)")]), DimSize(2, 3, 2), Some(5)),
    ];
    for (graph, dim, max_blocks) in cases {
        let netlist = NorNetlist::from_logic_graph(&graph).unwrap();
        let mut sets = Vec::new();
        for legacy in [true, false] {
            let mut config = ExactPlacerConfig::new(dim);
            config.legacy_encoder = legacy;
            config
                .model_params
                .insert("legacy_semantics".to_owned(), rsdsl::IValue::Bool(true));
            // Bound the count on larger boxes.
            config.max_blocks = max_blocks;
            let encoding = Encoding::build(&netlist, &config).unwrap();
            let layouts = admitted_layouts(&encoding, &netlist, 20_000)
                .unwrap_or_else(|| panic!("too many layouts in {dim:?}"));
            sets.push(layouts);
        }
        assert!(!sets[0].is_empty(), "{dim:?}");
        assert_eq!(
            sets[0].len(),
            sets[1].len(),
            "layout counts differ in {dim:?}"
        );
        assert_eq!(sets[0], sets[1], "layouts differ in {dim:?}");
        println!(
            "LAYOUTS {dim:?} max_blocks={max_blocks:?}: {}",
            sets[0].len()
        );
    }
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

/// Construction plus compaction for small circuits:
/// `CIRCUIT=mux2|half-adder|adder2|adder2-nor|mux4|full-adder CIRCUIT_WIDTH=2
/// CIRCUIT_HEIGHT=10 CIRCUIT_STEP_SECONDS=60 CIRCUIT_COMPACT_SECONDS=300
/// CIRCUIT_SEED=1 CIRCUIT_WRITE=<prefix>`; `CIRCUIT_NETLIST_ONLY=1` stops
/// after printing the NOR netlist and its live-net counts.
#[test]
#[ignore = "circuit pipeline measurement; run explicitly with --nocapture"]
fn diagnose_construct_circuit() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let circuit = std::env::var("CIRCUIT").unwrap_or_else(|_| "mux2".to_owned());
    let graph = circuit_graph(&circuit)?;
    let placer = ExactLocalPlacer::new(&graph)?.with_name(format!("exact-{circuit}"));
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
    for gate in netlist.gates() {
        let inputs = netlist.nets[gate]
            .gate_inputs
            .iter()
            .map(|&input| netlist.nets[input].name.as_str())
            .collect::<Vec<_>>();
        println!("CIRCUIT gate {} = NOR{inputs:?}", netlist.nets[gate].name);
    }
    let all_outputs = netlist
        .outputs
        .iter()
        .map(|(_, net)| *net)
        .collect::<BTreeSet<_>>();
    for order in [GateOrder::NetIndex, GateOrder::SmallestConeFirst] {
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
        window: 2,
        max_window: 4,
        seed,
        gate_order: gate_order_from_env("CIRCUIT_ORDER"),
        early_outputs: std::env::var("CIRCUIT_EARLY_OUTPUTS").as_deref() != Ok("0"),
        given_frozen_signals: std::env::var("CIRCUIT_GIVEN").as_deref() != Ok("0"),
        step_optimize: step_optimize_from_env("CIRCUIT_STEP_OPTIMIZE"),
        model_params: model_params_from_env("CIRCUIT_MODEL_PARAMS"),
        ..Default::default()
    };
    let (layout, placement, report) = placer.construct(&construction)?;
    println!(
        "CIRCUIT constructed dim={:?} blocks={} seed={} restarts={:?} elapsed={:?}",
        layout.dim,
        placement.block_count,
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
        ..Default::default()
    };
    let (compacted, best, report) = placer.compact(layout, &compaction)?;
    let result = best.unwrap_or(placement);
    let document = result.rcell.to_string();
    let reparsed: crate::physical_cell::PhysicalCellDocument = document.parse()?;
    let build = reparsed.build()?;
    let verification = reparsed.verify(&build)?;
    println!(
        "CIRCUIT compacted dim={:?} blocks={} removed={:?} elapsed={:?} rcell_failures={}",
        compacted.dim,
        result.block_count,
        report.removed,
        report.elapsed,
        verification.failures.len()
    );
    if let Ok(prefix) = std::env::var("CIRCUIT_WRITE") {
        std::fs::write(format!("{prefix}.rcell"), document)?;
        crate::nbt::NBTRoot::from(&result.placed.world).save(format!("{prefix}.nbt"));
    }
    assert!(verification.failures.is_empty());
    Ok(())
}
