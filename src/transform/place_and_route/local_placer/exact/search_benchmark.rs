//! Opt-in release benchmark using the production CaDiCaL wrapper and Python fixtures.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::cnf::Cnf;
use super::solver::{SatSolver, SolveResult, StopSignal};

#[test]
#[ignore = "release production encode/place timing and CNF regression snapshots"]
fn benchmark_production_pipeline() {
    use super::encode::Encoding;
    use super::{ExactLocalPlacer, ExactOutcome, ExactPlacerConfig};
    use crate::graph::logic::LogicGraph;
    use crate::world::position::DimSize;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let label = std::env::var("RUST_PIPELINE_LABEL").unwrap_or_else(|_| "baseline".into());
    let out = root.join("target/rust-pipeline").join(label);
    std::fs::create_dir_all(&out).unwrap();
    let mut rows = Vec::new();
    for (name, expr, dim, place) in [
        ("inverter", "~a", DimSize(1, 3, 2), true),
        ("nor", "~(a|b)", DimSize(1, 4, 2), true),
        ("nor_large", "~(a|b)", DimSize(2, 5, 3), true),
        ("xor", "a^b", DimSize(2, 6, 4), false),
    ] {
        let graph = LogicGraph::from_assignments([("out".to_owned(), expr.to_owned())])
            .unwrap()
            .prepare_place()
            .unwrap();
        let placer = ExactLocalPlacer::new(&graph).unwrap();
        let mut config = ExactPlacerConfig::new(dim);
        config.workers = 1;
        config.time_limit = Some(Duration::from_secs(10));
        for repeat in 0..7 {
            let started = Instant::now();
            let encoding = Encoding::build(placer.netlist(), &config).unwrap();
            let encode_s = started.elapsed().as_secs_f64();
            if repeat == 0 {
                let bytes: Vec<u8> = encoding
                    .cnf
                    .literals()
                    .iter()
                    .flat_map(|n| n.to_le_bytes())
                    .collect();
                std::fs::write(out.join(format!("{name}.cnf.bin")), bytes).unwrap();
            }
            let mut row = serde_json::json!({"case":name,"repeat":repeat,"encode_s":encode_s,
                "variables":encoding.cnf.num_vars(),"clauses":encoding.cnf.clause_count()});
            if place {
                let start = Instant::now();
                let (result, stats) = placer.place(&config).unwrap();
                row["place_s"] = start.elapsed().as_secs_f64().into();
                assert!(
                    matches!(result, ExactOutcome::Placed(_)),
                    "{name}: {result:?}"
                );
                row["place_encode_s"] = stats.encode_time.as_secs_f64().into();
                row["place_solve_s"] = stats.solve_time.as_secs_f64().into();
            }
            println!("{row}");
            rows.push(row);
        }
    }
    std::fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
}

struct Problem {
    cnf: Cnf,
    options: Vec<Vec<i32>>,
    expected: BTreeSet<Vec<usize>>,
    input_s: f64,
}

fn read_problem(root: &Path, name: &str) -> Problem {
    let started = Instant::now();
    let prefix = root.join("target/boundary-lab").join(name);
    let source = std::fs::read_to_string(prefix.with_extension("cnf")).unwrap();
    let mut vars = 0;
    let mut clauses = 0;
    let mut literals = Vec::new();
    for line in source.lines() {
        if line.starts_with('p') {
            let fields: Vec<_> = line.split_whitespace().collect();
            vars = fields[2].parse().unwrap();
            clauses = fields[3].parse().unwrap();
        } else if !line.starts_with('c') {
            literals.extend(line.split_whitespace().map(|s| s.parse::<i32>().unwrap()));
        }
    }
    assert_eq!(literals.iter().filter(|&&v| v == 0).count(), clauses);
    let mut choices = BTreeMap::<(usize, usize, usize), Vec<(String, i32)>>::new();
    for line in std::fs::read_to_string(prefix.with_extension("tsv"))
        .unwrap()
        .lines()
    {
        let f: Vec<_> = line.split('\t').collect();
        choices
            .entry((
                f[1].parse().unwrap(),
                f[0].parse().unwrap(),
                f[2].parse().unwrap(),
            ))
            .or_default()
            .push((f[3].to_owned(), f[4].parse().unwrap()));
    }
    let options = choices
        .values()
        .map(|opts| opts.iter().map(|(_, lit)| *lit).collect())
        .collect();
    let input_s = started.elapsed().as_secs_f64();
    // Reference reading and coordinate/label validation are outside timed work.
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            root.join("tools/pruning_lab/boundary_fixtures")
                .join(format!("{name}.layouts.json")),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, ((y, x, z), opts)) in choices.into_iter().enumerate() {
        assert_eq!(fixture["cells"][i], serde_json::json!([x, y, z]));
        assert_eq!(
            fixture["options"][i],
            serde_json::json!(opts.iter().map(|(label, _)| label).collect::<Vec<_>>())
        );
    }
    let expected = serde_json::from_value::<Vec<Vec<usize>>>(fixture["layouts"].clone())
        .unwrap()
        .into_iter()
        .collect();
    Problem {
        cnf: Cnf::from_literals(vars, literals, clauses),
        options,
        expected,
        input_s,
    }
}

struct Search<'a> {
    solver: SatSolver,
    signal: StopSignal<'a>,
    options: &'a [Vec<i32>],
    started: Instant,
    first_s: Option<f64>,
    checks: usize,
    solve_s: f64,
    decode_s: f64,
}

impl Search<'_> {
    fn check(&mut self, assumptions: &[i32]) -> bool {
        assert!(!self.signal.finished(), "benchmark deadline exceeded");
        self.checks += 1;
        let start = Instant::now();
        let status = self.solver.solve(assumptions, &self.signal);
        self.solve_s += start.elapsed().as_secs_f64();
        match status {
            SolveResult::Sat => {
                self.first_s
                    .get_or_insert_with(|| self.started.elapsed().as_secs_f64());
                true
            }
            SolveResult::Unsat => false,
            SolveResult::Interrupted => panic!("incomplete search, never certify partial results"),
        }
    }

    fn decode(&mut self) -> Vec<usize> {
        let start = Instant::now();
        let result = self
            .options
            .iter()
            .map(|opts| {
                opts.iter()
                    .position(|&lit| self.solver.value(lit))
                    .expect("exactly one Kind")
            })
            .collect();
        self.decode_s += start.elapsed().as_secs_f64();
        result
    }

    fn flat(&mut self) -> Vec<Vec<usize>> {
        let mut layouts = Vec::new();
        while self.check(&[]) {
            let selected = self.decode();
            let block: Vec<_> = self
                .options
                .iter()
                .zip(&selected)
                .map(|(opts, &i)| -opts[i])
                .collect();
            self.solver.add_clause(&block);
            layouts.push(selected);
        }
        layouts
    }

    fn partition(
        &mut self,
        depth: usize,
        assumptions: &mut Vec<i32>,
        mut witness: Vec<usize>,
        layouts: &mut Vec<Vec<usize>>,
    ) {
        if depth == self.options.len() {
            layouts.push(witness);
            return;
        }
        let saved = assumptions.len();
        loop {
            let lit = self.options[depth][witness[depth]];
            assumptions.push(lit);
            self.partition(depth + 1, assumptions, witness, layouts);
            assumptions.pop();
            // The sibling excludes the completed choice. Its remaining domain is disjoint.
            assumptions.push(-lit);
            if !self.check(assumptions) {
                break;
            }
            witness = self.decode();
        }
        assumptions.truncate(saved);
    }
}

#[test]
#[ignore = "explicit release benchmark; generate boundary fixtures first"]
fn benchmark_projected_search() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let names = [
        "inv3",
        "inv4",
        "inv5",
        "nor4",
        "nor5",
        "inv_fixed6",
        "inv_fixed7",
        "nor_fixed7",
        "inv_fixed8",
        "inv_long9",
        "nor_fixed10",
        "inv_long10",
        "inv_wide8",
    ];
    let configs = [
        ("flat_phase0", 0, 0, false),
        ("flat_phase1", 1, 0, false),
        ("flat_stable0", 0, 1, false),
        ("flat_stable1", 1, 1, false),
        ("partition_phase0", 0, 0, true),
    ];
    let seeds: usize = std::env::var("RUST_SEARCH_SEEDS")
        .unwrap_or_else(|_| "5".into())
        .parse()
        .unwrap();
    let mut records = Vec::new();
    let out = root.join("target/rust-search");
    std::fs::create_dir_all(&out).unwrap();
    for name in names {
        for seed in 1..=seeds {
            for offset in 0..configs.len() {
                let (mode, phase, stable, partition) = configs[(offset + seed - 1) % configs.len()];
                let problem = read_problem(root, name);
                let started = Instant::now();
                let stop = AtomicBool::new(false);
                let mut solver = SatSolver::new(seed as u32);
                solver.set_option("phase", phase);
                solver.set_option("stabilizeonly", stable);
                solver.add_cnf(&problem.cnf);
                let setup_s = started.elapsed().as_secs_f64();
                let mut search = Search {
                    solver,
                    options: &problem.options,
                    signal: StopSignal {
                        stop: &stop,
                        deadline: Some(started + Duration::from_secs(30)),
                        restart: None,
                    },
                    started,
                    first_s: None,
                    checks: 0,
                    solve_s: 0.0,
                    decode_s: 0.0,
                };
                let layouts = if partition {
                    let mut layouts = Vec::new();
                    if search.check(&[]) {
                        let witness = search.decode();
                        search.partition(0, &mut Vec::new(), witness, &mut layouts);
                    }
                    layouts
                } else {
                    search.flat()
                };
                let total_s = problem.input_s + started.elapsed().as_secs_f64();
                let count = layouts.len();
                let actual: BTreeSet<_> = layouts.into_iter().collect();
                assert_eq!(actual.len(), count, "duplicate layouts");
                assert_eq!(actual, problem.expected, "{name}/{mode}/{seed}");
                let row = serde_json::json!({"instance":name,"mode":mode,"seed":seed,
                    "total_s":total_s,"input_s":problem.input_s,"setup_s":setup_s,
                    "first_sat_s":search.first_s.map(|t| t + problem.input_s),"solve_s":search.solve_s,"decode_s":search.decode_s,
                    "checks":search.checks,"model_count":count,"exact_set_verified":true});
                println!("{row}");
                records.push(row);
                std::fs::write(out.join("results.json"), serde_json::to_vec_pretty(&serde_json::json!({
                    "scope":"production Rust CaDiCaL wrapper; CNF/Kind file loading, solver setup, enumeration and decoding; reference loading/validation and grounding excluded", "records":records
                })).unwrap()).unwrap();
            }
        }
    }
}

fn decode_kinds_reference(
    encoding: &super::encode::Encoding,
    solver: &SatSolver,
) -> (Vec<super::encode::CellKind>, Vec<i32>) {
    use super::encode::{CellKind, CARDINALS, TORCH_ATTACH};
    let geometry = encoding.geometry;
    let mut kinds = Vec::with_capacity(geometry.len());
    let mut kind_lits = Vec::new();
    for cell in 0..geometry.len() {
        let mut chosen = (CellKind::Air, encoding.air[cell]);
        let mut options = vec![
            (CellKind::Solid, encoding.solid[cell]),
            (CellKind::Dust, encoding.dust[cell]),
        ];
        for (index, attach) in TORCH_ATTACH.into_iter().enumerate() {
            options.push((CellKind::Torch(attach), encoding.torch[cell][index]));
        }
        for (index, direction) in CARDINALS.into_iter().enumerate() {
            options.push((
                CellKind::Repeater(direction),
                encoding.repeater[cell][index],
            ));
        }
        for site in encoding.switches.iter().filter(|site| site.cell == cell) {
            options.push((CellKind::Switch(site.attach), site.lit));
        }
        for (kind, lit) in options {
            if !encoding.cnf.is_false(lit) && solver.value(lit) {
                chosen = (kind, lit);
            }
        }
        kinds.push(chosen.0);
        kind_lits.push(chosen.1);
    }
    (kinds, kind_lits)
}

#[test]
#[ignore = "release decoder A/B comparison over actual production encodings"]
fn benchmark_kind_decoder() {
    use super::encode::Encoding;
    use super::verify::decode_kinds;
    use super::{ExactLocalPlacer, ExactPlacerConfig};
    use crate::graph::logic::LogicGraph;
    use crate::world::position::DimSize;
    let mut rows = Vec::new();
    for (name, expr, dim) in [
        ("inverter", "~a", DimSize(1, 3, 2)),
        ("nor", "~(a|b)", DimSize(1, 4, 2)),
        ("nor_large", "~(a|b)", DimSize(2, 5, 3)),
    ] {
        let graph = LogicGraph::from_assignments([("out".into(), expr.into())])
            .unwrap()
            .prepare_place()
            .unwrap();
        let placer = ExactLocalPlacer::new(&graph).unwrap();
        let config = ExactPlacerConfig::new(dim);
        let encoding = Encoding::build(placer.netlist(), &config).unwrap();
        let mut solver = SatSolver::new(1);
        solver.add_cnf(&encoding.cnf);
        let stop = AtomicBool::new(false);
        let signal = StopSignal {
            stop: &stop,
            deadline: Some(Instant::now() + Duration::from_secs(10)),
            restart: None,
        };
        let mut models = 0;
        while models < 30 && solver.solve(&[], &signal) == SolveResult::Sat {
            let expected = decode_kinds_reference(&encoding, &solver);
            assert_eq!(expected, decode_kinds(&encoding, &solver));
            for offset in 0..2 {
                let optimized = (models + offset) % 2 == 0;
                let start = Instant::now();
                for _ in 0..2000 {
                    std::hint::black_box(if optimized {
                        decode_kinds(&encoding, &solver)
                    } else {
                        decode_kinds_reference(&encoding, &solver)
                    });
                }
                rows.push(
                    serde_json::json!({"case":name,"model":models,"optimized":optimized,
                    "iterations":2000,"total_s":start.elapsed().as_secs_f64()}),
                );
            }
            solver.add_clause(&expected.1.iter().map(|l| -l).collect::<Vec<_>>());
            models += 1;
        }
        assert!(models > 0);
    }
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/rust-pipeline/decoder.json");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(out, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
}

#[test]
#[ignore = "interleaved release A/B for production worker dispatch, including simulation"]
fn benchmark_worker_dispatch() {
    use super::{ExactLocalPlacer, ExactOutcome, ExactPlacerConfig};
    use crate::graph::logic::LogicGraph;
    use crate::world::position::DimSize;
    let mut rows = Vec::new();
    for (name, expr, dim) in [
        ("inverter", "~a", DimSize(1, 3, 2)),
        ("nor", "~(a|b)", DimSize(1, 4, 2)),
        ("nor_large", "~(a|b)", DimSize(2, 5, 3)),
    ] {
        let graph = LogicGraph::from_assignments([("out".into(), expr.into())])
            .unwrap()
            .prepare_place()
            .unwrap();
        let placer = ExactLocalPlacer::new(&graph).unwrap();
        let mut config = ExactPlacerConfig::new(dim);
        config.workers = 1;
        config.time_limit = Some(Duration::from_secs(10));
        for repeat in 0..30 {
            let mut expected = None;
            for offset in 0..2 {
                let inline = (repeat + offset) % 2 == 0;
                let start = Instant::now();
                let (outcome, stats) = placer.place_with_worker_dispatch(&config, inline).unwrap();
                let elapsed = start.elapsed().as_secs_f64();
                let ExactOutcome::Placed(placement) = outcome else {
                    panic!("expected verified placement")
                };
                let document = placement.rcell.to_string();
                if let Some(old) = &expected {
                    assert_eq!(old, &document);
                } else {
                    expected = Some(document);
                }
                rows.push(
                    serde_json::json!({"case":name,"repeat":repeat,"inline":inline,
                    "total_s":elapsed,"encode_s":stats.encode_time.as_secs_f64(),
                    "solve_s":stats.solve_time.as_secs_f64(),"refinements":stats.refinements}),
                );
            }
        }
    }
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/rust-pipeline/dispatch.json");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(out, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
}
