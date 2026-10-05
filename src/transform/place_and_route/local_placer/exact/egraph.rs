//! Equivalent NOR netlists from an e-graph (egg).
//!
//! A hand-written netlist (`nor9`, say) fixes which gates the placer builds.
//! Equality saturation explores other expressions of the same functions: De
//! Morgan, double negation, commutativity and associativity, distribution and
//! factoring, absorption, and XOR expansions. Every e-class carries its truth
//! table, and classes with the same table are merged as soon as they appear,
//! so two forms of one function always meet, whichever rules produced them.
//!
//! Extraction then picks NOR netlists in the placer's terms. A NOT is a torch
//! (one gate and one redstone tick), an OR is free (dust and blocks), and a
//! shared subterm is built once. `Weights` trade gates against depth, so a few
//! weightings give a small set of distinct netlists to place.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use egg::{
    define_language, rewrite as rw, Analysis, DidMerge, EGraph, Id, RecExpr, Rewrite, Runner,
    StopReason, Symbol,
};
use eyre::{bail, ensure, eyre};
use rsdsl::{GroundOptions, IValue, Instance, Model, Objective};

use super::cnf::{Cnf, Lit};
use super::netlist::{Net, NetDriver, NetId, NorNetlist};
use super::solver::{SatSolver, SolveResult, StopSignal};

const EXTRACT_SOURCE: &str = include_str!("egraph_extract.rsdsl");

fn extract_model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        Model::parse("egraph_extract.rsdsl", EXTRACT_SOURCE)
            .unwrap_or_else(|error| panic!("{error}"))
    })
}

/// What `Exploration::extract_exact` minimizes, and within what.
#[derive(Debug, Clone, Copy)]
pub struct ExtractOptions {
    /// Gates on the slowest path to an output, at most.
    pub max_depth: Option<usize>,
    /// Weight of each OR node (one more input on a NOR) against a gate's 1.
    pub or_cost: usize,
    /// Only NORs of at most two signals.
    pub binary: bool,
    /// Build a NOR of more than two signals from OR nets: the two halves of
    /// its OR, each an OR net if wider than one signal, so construction
    /// places the merges as steps of their own before the torch.
    pub split_wide: bool,
    pub time_limit: Duration,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            or_cost: 0,
            binary: false,
            split_wide: false,
            time_limit: Duration::from_secs(60),
        }
    }
}

/// A netlist picked by `Exploration::extract_exact`.
#[derive(Debug, Clone)]
pub struct Extraction {
    pub netlist: NorNetlist,
    /// The minimized cost: gates, plus `or_cost` per OR node.
    pub cost: usize,
    /// No netlist in the e-graph within the depth costs less.
    pub proven: bool,
}

/// The extraction model on a solver, with what reads its answers back.
struct GroundedExtraction {
    nodes: Vec<(Id, Bool)>,
    picks: Vec<Lit>,
    objective: Objective,
    /// `-at_least[b]` requires cost `<= offset + b` (`objective_counter`).
    at_least: Vec<Lit>,
    solver: SatSolver,
}

impl GroundedExtraction {
    /// Assumptions for a cost of at most `cost`; `None` if none is that low.
    fn at_most(&self, cost: i64) -> Option<Vec<Lit>> {
        let bound = cost - self.objective.offset;
        if bound < 0 {
            return None;
        }
        Some(
            self.at_least
                .get(bound as usize)
                .map(|&lit| vec![-lit])
                .unwrap_or_default(),
        )
    }

    fn cost(&self) -> i64 {
        self.objective.cost(|lit| self.solver.value(lit))
    }

    /// The picked nodes by class.
    fn chosen(&self) -> HashMap<Id, Choice> {
        self.nodes
            .iter()
            .zip(&self.picks)
            .filter(|(_, &lit)| self.solver.value(lit))
            .map(|((class, node), _)| {
                let choice = Choice {
                    node: node.clone(),
                    gates: BTreeSet::new(),
                    depth: 0,
                };
                (*class, choice)
            })
            .collect()
    }
}

define_language! {
    /// Boolean expressions over named inputs. Only `~` and `|` are built: a
    /// NOT is a torch, an OR is dust and blocks. `&` and `^` state functions
    /// and give the rewrites something to go through.
    pub enum Bool {
        "~" = Not(Id),
        "|" = Or([Id; 2]),
        "&" = And([Id; 2]),
        "^" = Xor([Id; 2]),
        Var(Symbol),
    }
}

/// The truth table of every e-class: bit `k` is the value in input case `k`,
/// where bit `i` of `k` is input `i`. Classes with the same table are merged.
#[derive(Debug, Clone, Default)]
pub struct TruthTable {
    inputs: Vec<Symbol>,
    full: u64,
    by_table: HashMap<u64, Id>,
}

impl TruthTable {
    fn new(inputs: &[&str]) -> eyre::Result<Self> {
        ensure!(inputs.len() <= 6, "truth tables hold at most 6 inputs");
        let cases = 1u32 << inputs.len();
        Ok(Self {
            inputs: inputs.iter().map(|&name| Symbol::from(name)).collect(),
            full: if cases == 64 {
                u64::MAX
            } else {
                (1u64 << cases) - 1
            },
            by_table: HashMap::new(),
        })
    }

    fn input(&self, symbol: Symbol) -> u64 {
        let index = self
            .inputs
            .iter()
            .position(|&input| input == symbol)
            .unwrap_or_else(|| panic!("unknown input {symbol}"));
        (0..self.full.count_ones())
            .filter(|case| case & (1 << index) != 0)
            .fold(0, |table, case| table | (1 << case))
    }
}

impl Analysis<Bool> for TruthTable {
    type Data = u64;

    fn make(egraph: &EGraph<Bool, Self>, enode: &Bool) -> u64 {
        let table = |id: &Id| egraph[*id].data;
        let analysis = &egraph.analysis;
        match enode {
            Bool::Not(a) => !table(a) & analysis.full,
            Bool::Or([a, b]) => table(a) | table(b),
            Bool::And([a, b]) => table(a) & table(b),
            Bool::Xor([a, b]) => table(a) ^ table(b),
            Bool::Var(symbol) => analysis.input(*symbol),
        }
    }

    fn merge(&mut self, a: &mut u64, b: u64) -> DidMerge {
        assert_eq!(*a, b, "merged e-classes compute different functions");
        DidMerge(false, false)
    }

    fn modify(egraph: &mut EGraph<Bool, Self>, id: Id) {
        let table = egraph[id].data;
        match egraph.analysis.by_table.get(&table) {
            Some(&other) => {
                egraph.union(other, id);
            }
            None => {
                egraph.analysis.by_table.insert(table, id);
            }
        }
    }
}

fn rules() -> Vec<Rewrite<Bool, TruthTable>> {
    let mut rules = vec![
        rw!("or-comm"; "(| ?a ?b)" => "(| ?b ?a)"),
        rw!("and-comm"; "(& ?a ?b)" => "(& ?b ?a)"),
        rw!("xor-comm"; "(^ ?a ?b)" => "(^ ?b ?a)"),
        rw!("or-assoc"; "(| ?a (| ?b ?c))" => "(| (| ?a ?b) ?c)"),
        rw!("and-assoc"; "(& ?a (& ?b ?c))" => "(& (& ?a ?b) ?c)"),
        rw!("xor-assoc"; "(^ ?a (^ ?b ?c))" => "(^ (^ ?a ?b) ?c)"),
        rw!("not-not"; "(~ (~ ?a))" => "?a"),
        rw!("or-absorb"; "(| ?a (& ?a ?b))" => "?a"),
        rw!("and-absorb"; "(& ?a (| ?a ?b))" => "?a"),
        rw!("or-idem"; "(| ?a ?a)" => "?a"),
        rw!("and-idem"; "(& ?a ?a)" => "?a"),
    ];
    rules.extend(rw!("and-de-morgan"; "(& ?a ?b)" <=> "(~ (| (~ ?a) (~ ?b)))"));
    rules.extend(rw!("or-de-morgan"; "(~ (| ?a ?b))" <=> "(& (~ ?a) (~ ?b))"));
    rules.extend(rw!("and-over-or"; "(& ?a (| ?b ?c))" <=> "(| (& ?a ?b) (& ?a ?c))"));
    rules.extend(rw!("or-over-and"; "(| ?a (& ?b ?c))" <=> "(& (| ?a ?b) (| ?a ?c))"));
    rules.extend(rw!("xor-sop"; "(^ ?a ?b)" <=> "(| (& ?a (~ ?b)) (& (~ ?a) ?b))"));
    rules.extend(rw!("xor-pos"; "(^ ?a ?b)" <=> "(& (| ?a ?b) (~ (& ?a ?b)))"));
    rules.extend(rw!("xnor"; "(~ (^ ?a ?b))" <=> "(^ (~ ?a) ?b)"));
    rules
}

/// Limits of one saturation run.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub iterations: usize,
    pub nodes: usize,
    pub time: std::time::Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            iterations: 12,
            nodes: 50_000,
            time: std::time::Duration::from_secs(20),
        }
    }
}

/// A saturated e-graph and its output classes.
pub struct Exploration {
    egraph: EGraph<Bool, TruthTable>,
    inputs: Vec<String>,
    outputs: Vec<(String, Id)>,
    pub iterations: usize,
    pub stop_reason: Option<StopReason>,
}

/// How to weigh an extracted netlist: torches and ticks on its slowest path.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub gates: f64,
    pub depth: f64,
}

#[derive(Debug, Clone)]
struct Choice {
    node: Bool,
    gates: BTreeSet<Id>,
    depth: usize,
}

impl Exploration {
    /// Saturates `outputs` (infix expressions over `inputs` with `~ & ^ |`)
    /// within `limits`.
    pub fn new(inputs: &[&str], outputs: &[(&str, &str)], limits: Limits) -> eyre::Result<Self> {
        let mut egraph = EGraph::new(TruthTable::new(inputs)?);
        let mut roots = Vec::new();
        for &(name, expression) in outputs {
            let expression = parse(expression)?;
            roots.push((name.to_owned(), egraph.add_expr(&expression)));
        }
        let runner = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(limits.iterations)
            .with_node_limit(limits.nodes)
            .with_time_limit(limits.time)
            .run(&rules());
        let iterations = runner.iterations.len();
        let stop_reason = runner.stop_reason.clone();
        let egraph = runner.egraph;
        let outputs = roots
            .into_iter()
            .map(|(name, id)| (name, egraph.find(id)))
            .collect();
        Ok(Self {
            egraph,
            inputs: inputs.iter().map(|&name| name.to_owned()).collect(),
            outputs,
            iterations,
            stop_reason,
        })
    }

    pub fn classes(&self) -> usize {
        self.egraph.number_of_classes()
    }

    pub fn nodes(&self) -> usize {
        self.egraph.total_number_of_nodes()
    }

    /// The netlist with the least weighted cost, greedily: each class takes
    /// the NOT, OR, or input node whose own and inputs' gates (shared ones
    /// counted once) and depth weigh least, until no class changes.
    pub fn extract(&self, weights: Weights) -> eyre::Result<NorNetlist> {
        let egraph = &self.egraph;
        let score = |choice: &Choice| {
            weights.gates * choice.gates.len() as f64 + weights.depth * choice.depth as f64
        };
        let mut best = HashMap::<Id, Choice>::new();
        for _ in 0..256 {
            let mut changed = false;
            for class in egraph.classes() {
                for node in &class.nodes {
                    let Some(candidate) = self.candidate(class.id, node, &best) else {
                        continue;
                    };
                    let better = match best.get(&class.id) {
                        None => true,
                        Some(current) => {
                            let (new, old) = (score(&candidate), score(current));
                            new < old - 1e-9
                                || ((new - old).abs() <= 1e-9
                                    && (candidate.gates.len(), candidate.depth)
                                        < (current.gates.len(), current.depth))
                        }
                    };
                    if better {
                        best.insert(class.id, candidate);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        self.netlist(&best, false)
    }

    /// The netlist with the fewest NOT gates (plus `or_cost` per OR node)
    /// whose outputs are at most `max_depth` gates deep, picked exactly by
    /// SAT over the e-graph (`egraph_extract.rsdsl`): one node per needed
    /// class, shared classes built once. Lowers the bound until it is proven
    /// or `time_limit` runs out; `None` when no netlist of the e-graph fits.
    pub fn extract_exact(&self, options: &ExtractOptions) -> eyre::Result<Option<Extraction>> {
        let mut grounded = self.ground_extraction(options)?;
        let stop = AtomicBool::new(false);
        let signal = StopSignal {
            stop: &stop,
            deadline: Some(Instant::now() + options.time_limit),
            restart: None,
        };
        let mut best = None::<(i64, HashMap<Id, Choice>)>;
        let mut proven = false;
        loop {
            let assumptions = match &best {
                None => Vec::new(),
                Some((cost, _)) => match grounded.at_most(cost - 1) {
                    Some(assumptions) => assumptions,
                    None => {
                        proven = true;
                        break;
                    }
                },
            };
            match grounded.solver.solve(&assumptions, &signal) {
                SolveResult::Sat => best = Some((grounded.cost(), grounded.chosen())),
                SolveResult::Unsat => {
                    proven = best.is_some();
                    break;
                }
                SolveResult::Interrupted => break,
            }
        }
        let Some((cost, chosen)) = best else {
            return Ok(None);
        };
        Ok(Some(Extraction {
            netlist: self.netlist(&chosen, options.split_wide)?,
            cost: cost as usize,
            proven,
        }))
    }

    /// `egraph_extract.rsdsl` grounded over this e-graph, on a solver.
    fn ground_extraction(&self, options: &ExtractOptions) -> eyre::Result<GroundedExtraction> {
        let ExtractOptions {
            max_depth,
            or_cost,
            binary,
            ..
        } = *options;
        let egraph = &self.egraph;
        let classes = egraph.classes().map(|class| class.id).collect::<Vec<_>>();
        let index = classes
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect::<HashMap<_, _>>();
        let mut nodes = Vec::<(Id, Bool)>::new();
        let mut rows = Vec::new();
        for class in egraph.classes() {
            let own = class.id;
            for node in &class.nodes {
                let children = match node {
                    Bool::Var(_) => Vec::new(),
                    Bool::Not(a) => vec![egraph.find(*a)],
                    Bool::Or([a, b]) => vec![egraph.find(*a), egraph.find(*b)],
                    Bool::And(_) | Bool::Xor(_) => continue,
                };
                if children.contains(&own) {
                    continue;
                }
                let n = nodes.len();
                nodes.push((own, node.clone()));
                rows.push(("node_class", vec![n.into(), index[&own].into()]));
                match node {
                    Bool::Not(_) => rows.push(("gate", vec![n.into()])),
                    Bool::Or(_) => rows.push(("or_node", vec![n.into()])),
                    _ => {}
                }
                for child in children.into_iter().collect::<BTreeSet<_>>() {
                    rows.push((
                        "edge",
                        vec![n.into(), index[&own].into(), index[&child].into()],
                    ));
                }
            }
        }
        let levels = max_depth.unwrap_or(24);
        let mut instance = Instance::new("egraph");
        instance
            .domain("Class", (0..classes.len()).map(IValue::from))
            .domain("Node", (0..nodes.len()).map(IValue::from))
            .domain("Tick", (0..=levels).map(IValue::from))
            .param("levels", levels)
            .param("or_cost", or_cost)
            .param("binary", binary);
        if let Some(depth) = max_depth {
            instance.param("max_depth", depth);
        }
        for fact in ["node_class", "edge", "gate", "or_node", "root"] {
            instance.fact(fact);
        }
        for (fact, row) in rows {
            instance.row(fact, row);
        }
        for (_, root) in &self.outputs {
            instance.row("root", vec![index[&egraph.find(*root)].into()]);
        }
        let ground = GroundOptions {
            guards: false,
            provenance: false,
            positive_or_aux: false,
            no_fold: false,
        };
        let mut program = extract_model()
            .ground(&instance, ground)
            .map_err(|error| eyre!("grounding egraph_extract failed:\n{error}"))?;
        let objective = program
            .objective()
            .ok_or_else(|| eyre!("egraph_extract has no objective"))?;
        let at_least = program.objective_counter(objective.total_weight());
        let picks = (0..nodes.len())
            .map(|n| program.var_lit("Pick", &[n.into()]))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| eyre!("egraph_extract has no Pick for every node"))?;
        let cnf = Cnf::from_literals(
            program.num_vars(),
            program.literals().to_vec(),
            program.clause_count(),
        );
        let mut solver = SatSolver::new(1);
        solver.add_cnf(&cnf);
        Ok(GroundedExtraction {
            nodes,
            picks,
            objective,
            at_least,
            solver,
        })
    }

    fn candidate(&self, class: Id, node: &Bool, best: &HashMap<Id, Choice>) -> Option<Choice> {
        let child = |id: &Id| best.get(&self.egraph.find(*id));
        match node {
            Bool::Var(_) => Some(Choice {
                node: node.clone(),
                gates: BTreeSet::new(),
                depth: 0,
            }),
            Bool::Or([a, b]) => {
                let (a, b) = (child(a)?, child(b)?);
                Some(Choice {
                    node: node.clone(),
                    gates: a.gates.union(&b.gates).copied().collect(),
                    depth: a.depth.max(b.depth),
                })
            }
            Bool::Not(a) => {
                let a = child(a)?;
                // A NOT of its own class (through rewrites) builds nothing.
                if a.gates.contains(&class) {
                    return None;
                }
                let mut gates = a.gates.clone();
                gates.insert(class);
                Some(Choice {
                    node: node.clone(),
                    gates,
                    depth: a.depth + 1,
                })
            }
            Bool::And(_) | Bool::Xor(_) => None,
        }
    }

    fn netlist(&self, best: &HashMap<Id, Choice>, split: bool) -> eyre::Result<NorNetlist> {
        let mut builder = Builder {
            exploration: self,
            best,
            split,
            nets: self
                .inputs
                .iter()
                .enumerate()
                .map(|(index, name)| Net {
                    name: name.clone(),
                    node_id: index,
                    driver: NetDriver::Input(name.clone()),
                    gate_inputs: Vec::new(),
                })
                .collect(),
            net_of_class: HashMap::new(),
        };
        let mut outputs = Vec::new();
        for (name, root) in &self.outputs {
            let (root, node) = builder.choice(*root)?;
            let output = match node {
                Bool::Var(_) => bail!("output `{name}` is an input"),
                Bool::Not(_) => builder.net(root, 0)?,
                // An OR output needs a driver: a NOT over the NOR of its terms.
                _ => {
                    let inputs = builder.sources(root, 0)?;
                    let nor = builder.push(NetDriver::Gate, inputs);
                    builder.nets[nor].name = format!("{name}_n");
                    builder.push(NetDriver::Gate, BTreeSet::from([nor]))
                }
            };
            outputs.push((name.clone(), output));
        }
        let mut nets = builder.nets;
        for (name, output) in &outputs {
            if nets[*output].driver == NetDriver::Gate {
                nets[*output].name = name.clone();
            }
        }
        outputs.sort();
        let netlist = NorNetlist { nets, outputs };
        netlist.check_acyclic()?;
        Ok(netlist)
    }
}

/// Turns one chosen node per class into nets.
struct Builder<'a> {
    exploration: &'a Exploration,
    best: &'a HashMap<Id, Choice>,
    /// NORs of more than two signals read OR nets instead (see `ExtractOptions::split_wide`).
    split: bool,
    nets: Vec<Net>,
    net_of_class: HashMap<Id, NetId>,
}

impl Builder<'_> {
    fn choice(&self, class: Id) -> eyre::Result<(Id, Bool)> {
        let class = self.exploration.egraph.find(class);
        let choice = self
            .best
            .get(&class)
            .ok_or_else(|| eyre!("class {class} has no buildable form"))?;
        Ok((class, choice.node.clone()))
    }

    fn push(&mut self, driver: NetDriver, inputs: BTreeSet<NetId>) -> NetId {
        let index = self.nets.len();
        let prefix = if driver == NetDriver::Or { "o" } else { "g" };
        self.nets.push(Net {
            name: format!("{prefix}{index}"),
            node_id: index,
            driver,
            gate_inputs: inputs.into_iter().collect(),
        });
        index
    }

    /// The nets whose OR a class is (one net unless it chose an OR).
    fn sources(&mut self, class: Id, depth: usize) -> eyre::Result<BTreeSet<NetId>> {
        ensure!(depth < 256, "extraction recursed too deep");
        let (class, node) = self.choice(class)?;
        match node {
            Bool::Or([a, b]) => {
                let mut set = self.sources(a, depth + 1)?;
                set.extend(self.sources(b, depth + 1)?);
                Ok(set)
            }
            _ => Ok(BTreeSet::from([self.net(class, depth + 1)?])),
        }
    }

    /// The net of a class that chose an input or a NOT.
    fn net(&mut self, class: Id, depth: usize) -> eyre::Result<NetId> {
        ensure!(depth < 256, "extraction recursed too deep");
        if let Some(&net) = self.net_of_class.get(&class) {
            return Ok(net);
        }
        let net = match self.choice(class)?.1 {
            Bool::Var(symbol) => self
                .nets
                .iter()
                .position(|net| net.driver == NetDriver::Input(symbol.to_string()))
                .ok_or_else(|| eyre!("unknown input {symbol}"))?,
            Bool::Not(a) => {
                let inputs = self.gate_inputs(a, depth + 1)?;
                self.push(NetDriver::Gate, inputs)
            }
            other => bail!("class {class} chose {other} as a net"),
        };
        self.net_of_class.insert(class, net);
        Ok(net)
    }

    /// What a NOT of `class` reads: the leaves of its OR, or, with `split`
    /// and more than two leaves, the OR's two halves, each an OR net if it
    /// is an OR itself.
    fn gate_inputs(&mut self, class: Id, depth: usize) -> eyre::Result<BTreeSet<NetId>> {
        let leaves = self.sources(class, depth)?;
        if !self.split || leaves.len() <= 2 {
            return Ok(leaves);
        }
        match self.choice(class)?.1 {
            Bool::Or([a, b]) => Ok(BTreeSet::from([
                self.term(a, depth + 1)?,
                self.term(b, depth + 1)?,
            ])),
            _ => Ok(leaves),
        }
    }

    /// The net of a class, an OR net if it chose an OR.
    fn term(&mut self, class: Id, depth: usize) -> eyre::Result<NetId> {
        ensure!(depth < 256, "extraction recursed too deep");
        let (class, node) = self.choice(class)?;
        let Bool::Or([a, b]) = node else {
            return self.net(class, depth + 1);
        };
        if let Some(&net) = self.net_of_class.get(&class) {
            return Ok(net);
        }
        let inputs = BTreeSet::from([self.term(a, depth + 1)?, self.term(b, depth + 1)?]);
        let net = self.push(NetDriver::Or, inputs);
        self.net_of_class.insert(class, net);
        Ok(net)
    }
}

/// The truth table of `expression` over `inputs`: bit `k` is its value in
/// case `k`, where bit `i` of `k` is `inputs[i]`.
pub fn truth_table(inputs: &[&str], expression: &str) -> eyre::Result<u64> {
    let mut egraph = EGraph::new(TruthTable::new(inputs)?);
    let id = egraph.add_expr(&parse(expression)?);
    Ok(egraph[id].data)
}

/// Gates on the slowest path to each output (OR nets free, NOT one).
pub fn netlist_depths(netlist: &NorNetlist) -> BTreeMap<String, usize> {
    let mut depth = vec![0usize; netlist.nets.len()];
    for net in netlist.topological_order() {
        let latest = netlist.nets[net]
            .gate_inputs
            .iter()
            .map(|&input| depth[input])
            .max()
            .unwrap_or(0);
        depth[net] = match netlist.nets[net].driver {
            NetDriver::Input(_) => 0,
            NetDriver::Gate => latest + 1,
            NetDriver::Or => latest,
        };
    }
    netlist
        .outputs
        .iter()
        .map(|(name, net)| (name.clone(), depth[*net]))
        .collect()
}

/// The gates as functions of the inputs: two netlists with the same gate
/// functions and gate inputs build the same circuit.
pub fn signature(netlist: &NorNetlist) -> BTreeSet<(Vec<bool>, Vec<Vec<bool>>)> {
    let values = netlist.net_values();
    netlist
        .gates()
        .map(|gate| {
            let mut inputs = netlist.nets[gate]
                .gate_inputs
                .iter()
                .map(|&input| values[input].clone())
                .collect::<Vec<_>>();
            inputs.sort();
            (values[gate].clone(), inputs)
        })
        .collect()
}

/// Parses `~ & ^ |` (tightest first) over names and parentheses.
fn parse(text: &str) -> eyre::Result<RecExpr<Bool>> {
    struct Parser<'a> {
        chars: std::iter::Peekable<std::str::Chars<'a>>,
        expression: RecExpr<Bool>,
    }
    impl Parser<'_> {
        fn skip(&mut self) {
            while self.chars.peek().is_some_and(|c| c.is_whitespace()) {
                self.chars.next();
            }
        }
        fn binary(&mut self, level: usize) -> eyre::Result<Id> {
            const OPERATORS: [char; 3] = ['|', '^', '&'];
            if level == OPERATORS.len() {
                return self.unary();
            }
            let mut left = self.binary(level + 1)?;
            loop {
                self.skip();
                if self.chars.peek() != Some(&OPERATORS[level]) {
                    return Ok(left);
                }
                self.chars.next();
                let right = self.binary(level + 1)?;
                let node = match OPERATORS[level] {
                    '|' => Bool::Or([left, right]),
                    '^' => Bool::Xor([left, right]),
                    _ => Bool::And([left, right]),
                };
                left = self.expression.add(node);
            }
        }
        fn unary(&mut self) -> eyre::Result<Id> {
            self.skip();
            match self.chars.peek() {
                Some('~') => {
                    self.chars.next();
                    let inner = self.unary()?;
                    Ok(self.expression.add(Bool::Not(inner)))
                }
                Some('(') => {
                    self.chars.next();
                    let inner = self.binary(0)?;
                    self.skip();
                    ensure!(self.chars.next() == Some(')'), "missing `)`");
                    Ok(inner)
                }
                Some(c) if c.is_alphanumeric() || *c == '_' => {
                    let mut name = String::new();
                    while let Some(&c) = self.chars.peek() {
                        if !(c.is_alphanumeric() || c == '_') {
                            break;
                        }
                        name.push(c);
                        self.chars.next();
                    }
                    Ok(self.expression.add(Bool::Var(Symbol::from(name))))
                }
                other => bail!("unexpected {other:?}"),
            }
        }
    }
    let mut parser = Parser {
        chars: text.chars().peekable(),
        expression: RecExpr::default(),
    };
    parser.binary(0)?;
    parser.skip();
    ensure!(parser.chars.next().is_none(), "trailing text in `{text}`");
    Ok(parser.expression)
}
