//! Exact synthesis of NOR netlists that windowed construction can build
//! (`nor_synthesis.rsdsl`).
//!
//! The e-graph (`egraph.rs`) finds netlists with fewer gates or less depth,
//! but construction cannot place most of them: between two steps they keep
//! more nets alive than a box two cells wide carries. Synthesis builds the
//! netlist and its construction order together, step by step, with the nets
//! alive between steps bounded, so everything it returns passes that screen.
//! Wide NORs come out as chains of OR stages on a support block, the form
//! construction builds best (`NorNetlist::chain_wide_gates`).

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use eyre::{ensure, eyre};
use rsdsl::{GroundOptions, IValue, Instance, Model};

use super::cnf::Cnf;
use super::egraph::truth_table;
use super::netlist::{Net, NetDriver, NorNetlist};
use super::solver::{SatSolver, SolveResult, StopSignal};

const SYNTHESIS_SOURCE: &str = include_str!("nor_synthesis.rsdsl");

fn synthesis_model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        Model::parse("nor_synthesis.rsdsl", SYNTHESIS_SOURCE)
            .unwrap_or_else(|error| panic!("{error}"))
    })
}

/// What `synthesize` looks for.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisOptions {
    /// Steps: torches plus OR stages.
    pub steps: usize,
    /// Nets alive between two steps, at most.
    pub max_live: usize,
    /// Torches on the slowest path to an output, at most.
    pub max_depth: Option<usize>,
    pub time_limit: Duration,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            steps: 11,
            max_live: 4,
            max_depth: None,
            time_limit: Duration::from_secs(60),
        }
    }
}

/// A netlist found by `synthesize`, its nets in construction order.
#[derive(Debug, Clone)]
pub struct Synthesis {
    pub netlist: NorNetlist,
    pub torches: usize,
    /// No netlist of this many steps within the bounds has fewer torches.
    pub proven: bool,
}

/// What `synthesize` concluded.
#[derive(Debug, Clone)]
pub enum SynthesisOutcome {
    Found(Synthesis),
    /// Proven: no netlist of this many steps fits the bounds.
    Infeasible,
    /// `time_limit` ran out before a netlist was found.
    Unknown,
}

/// The netlist of `options.steps` steps computing `outputs` (expressions over
/// `inputs`, as in `Exploration::new`) with the fewest torches, within the
/// live and depth bounds. Construction should take its nets in index order
/// (`GateOrder::NetIndex`), the order the bound holds for.
pub fn synthesize(
    inputs: &[&str],
    outputs: &[(&str, &str)],
    options: &SynthesisOptions,
) -> eyre::Result<SynthesisOutcome> {
    ensure!(inputs.len() <= 6, "synthesis takes at most 6 inputs");
    let cases = 1usize << inputs.len();
    let targets = outputs
        .iter()
        .map(|&(name, expression)| Ok((name, truth_table(inputs, expression)?)))
        .collect::<eyre::Result<Vec<_>>>()?;
    let nets = inputs.len() + options.steps;
    let mut instance = Instance::new("synthesis");
    instance
        .domain("Net", (0..nets).map(IValue::from))
        .domain("Case", (0..cases).map(IValue::from))
        .domain("Output", targets.iter().map(|(name, _)| IValue::sym(*name)))
        .domain("Tick", (0..=options.steps).map(IValue::from))
        .param("max_live", options.max_live)
        .param("levels", options.steps);
    if let Some(depth) = options.max_depth {
        instance.param("max_depth", depth);
    }
    for fact in ["input", "step", "on", "target"] {
        instance.fact(fact);
    }
    for net in 0..nets {
        let fact = if net < inputs.len() { "input" } else { "step" };
        instance.row(fact, vec![net.into()]);
    }
    for (index, _) in inputs.iter().enumerate() {
        for case in (0..cases).filter(|case| case >> index & 1 == 1) {
            instance.row("on", vec![index.into(), case.into()]);
        }
    }
    for (name, table) in &targets {
        for case in (0..cases).filter(|case| table >> case & 1 == 1) {
            instance.row("target", vec![IValue::sym(*name), case.into()]);
        }
    }
    let ground = GroundOptions {
        guards: false,
        provenance: false,
        positive_or_aux: false,
        no_fold: false,
    };
    let mut program = synthesis_model()
        .ground(&instance, ground)
        .map_err(|error| eyre!("grounding nor_synthesis failed:\n{error}"))?;
    let objective = program
        .objective()
        .ok_or_else(|| eyre!("nor_synthesis has no objective"))?;
    let at_least = program.objective_counter(objective.total_weight());
    let lit = |program: &rsdsl::Program, var: &str, key: &[usize]| {
        let key = key
            .iter()
            .map(|&value| value.into())
            .collect::<Vec<IValue>>();
        program
            .var_lit(var, &key)
            .ok_or_else(|| eyre!("nor_synthesis has no {var}{key:?}"))
    };
    let steps = inputs.len()..nets;
    let nor = steps
        .clone()
        .map(|t| lit(&program, "Nor", &[t]))
        .collect::<eyre::Result<Vec<_>>>()?;
    let mut reads = Vec::new();
    for t in steps.clone() {
        for j in 0..t {
            reads.push((j, t, lit(&program, "Read", &[j, t])?));
        }
    }
    let mut chosen_outputs = Vec::new();
    for (name, _) in &targets {
        for t in steps.clone() {
            let key = [IValue::sym(*name), t.into()];
            let out = program
                .var_lit("Out", &key)
                .ok_or_else(|| eyre!("nor_synthesis has no Out[{name}, {t}]"))?;
            chosen_outputs.push((*name, t, out));
        }
    }
    let cnf = Cnf::from_literals(
        program.num_vars(),
        program.literals().to_vec(),
        program.clause_count(),
    );
    let mut solver = SatSolver::new(1);
    solver.add_cnf(&cnf);
    let stop = AtomicBool::new(false);
    let signal = StopSignal {
        stop: &stop,
        deadline: Some(Instant::now() + options.time_limit),
        restart: None,
    };
    let mut best = None::<(i64, NorNetlist)>;
    let mut proven = false;
    let mut infeasible = false;
    loop {
        let assumptions = match &best {
            None => Vec::new(),
            Some((cost, _)) => {
                let bound = cost - 1 - objective.offset;
                if bound < 0 {
                    proven = true;
                    break;
                }
                at_least
                    .get(bound as usize)
                    .map(|&lit| vec![-lit])
                    .unwrap_or_default()
            }
        };
        match solver.solve(&assumptions, &signal) {
            SolveResult::Sat => {}
            SolveResult::Unsat => {
                proven = best.is_some();
                infeasible = best.is_none();
                break;
            }
            SolveResult::Interrupted => break,
        }
        let cost = objective.cost(|lit| solver.value(lit));
        let mut netlist = NorNetlist {
            nets: inputs
                .iter()
                .enumerate()
                .map(|(index, &name)| Net {
                    name: name.to_owned(),
                    node_id: index,
                    driver: NetDriver::Input(name.to_owned()),
                    gate_inputs: Vec::new(),
                })
                .collect(),
            outputs: Vec::new(),
        };
        for (index, t) in steps.clone().enumerate() {
            let torch = solver.value(nor[index]);
            netlist.nets.push(Net {
                name: format!("{}{t}", if torch { "g" } else { "o" }),
                node_id: t,
                driver: if torch {
                    NetDriver::Gate
                } else {
                    NetDriver::Or
                },
                gate_inputs: reads
                    .iter()
                    .filter(|&&(_, reader, lit)| reader == t && solver.value(lit))
                    .map(|&(j, _, _)| j)
                    .collect(),
            });
        }
        for &(name, t, out) in &chosen_outputs {
            if solver.value(out) {
                netlist.nets[t].name = name.to_owned();
                netlist.outputs.push((name.to_owned(), t));
            }
        }
        netlist.outputs.sort();
        check_outputs(&netlist, inputs, &targets)?;
        best = Some((cost, netlist));
    }
    Ok(match best {
        Some((cost, netlist)) => SynthesisOutcome::Found(Synthesis {
            netlist,
            torches: cost as usize,
            proven,
        }),
        None if infeasible => SynthesisOutcome::Infeasible,
        None => SynthesisOutcome::Unknown,
    })
}

/// The netlist computes every target (bit `k` of a table is case `k`, bit
/// `i` of `k` is `inputs[i]`).
fn check_outputs(
    netlist: &NorNetlist,
    inputs: &[&str],
    targets: &[(&str, u64)],
) -> eyre::Result<()> {
    let values = netlist.net_values();
    // `net_values` numbers cases by sorted input names.
    let names = netlist.input_names();
    let outputs = netlist.outputs.iter().cloned().collect::<BTreeMap<_, _>>();
    for (name, table) in targets {
        let net = outputs[*name];
        for case in 0..1usize << inputs.len() {
            let sorted = inputs
                .iter()
                .enumerate()
                .filter(|(index, _)| case >> index & 1 == 1)
                .map(|(_, input)| 1 << names.iter().position(|n| n == input).unwrap())
                .sum::<usize>();
            ensure!(
                values[net][sorted] == (table >> case & 1 == 1),
                "synthesized {name} is wrong in case {case}"
            );
        }
    }
    Ok(())
}
