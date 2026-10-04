//! Builds the exact placer's CNF by grounding `exact_placer.rsdsl`.
//!
//! Rust only prepares the compile-time world (box size, signal vocabulary,
//! pin sites, fixed cells) as an rsdsl instance; every constraint lives in
//! the model file. The grounded program is then read back into the same
//! `Encoding` fields that the hand-written encoder (`encode.rs`) fills, so
//! decoding, verification, lazy acyclicity, and the search pipeline are
//! shared.

use std::collections::HashSet;
use std::sync::OnceLock;

use eyre::{bail, ensure, eyre};
use rsdsl::{GroundOptions, IValue, Instance, Model};

use super::cnf::Cnf;
use super::encode::{
    default_input_sites, vocabulary, CellKind, Encoding, Geometry, ObjectiveBound, OutputSite,
    Relation, SignalClass, SinkKind, SourceKind, SwitchSite, CARDINALS, TORCH_ATTACH,
};
use super::netlist::{NetId, NorNetlist};
use super::ExactPlacerConfig;
use crate::world::block::Direction;
use crate::world::position::Position;

const SOURCE: &str = include_str!("exact_placer.rsdsl");

/// Labels in `exact_placer.rsdsl` whose `@guarded` selectors back
/// `relax_soundness`.
const SOUNDNESS_RULE: &str =
    "건전성: 원천이 켜지면 받는 쪽도 켜짐 (가루·리피터는 원천과 똑같은 신호)";
const COVERAGE_RULE: &str = "켜진 것은 켜진 원천이 있어야 함";

/// Signal functions are `u64` truth tables over input cases (2^6 = 64).
const MAX_INPUTS: usize = 6;
/// The torch lower-bound search tracks sets of classes as `u64` masks.
const MAX_BOUND_CLASSES: usize = 64;

pub(super) fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        Model::parse("exact_placer.rsdsl", SOURCE).unwrap_or_else(|error| panic!("{error}"))
    })
}

fn attach_name(attach: Direction) -> &'static str {
    match attach {
        Direction::Bottom => "Floor",
        Direction::East => "East",
        Direction::West => "West",
        Direction::North => "North",
        Direction::South => "South",
        other => unreachable!("no attachment toward {other:?}"),
    }
}

fn direction_name(direction: Direction) -> &'static str {
    match direction {
        Direction::East => "East",
        Direction::West => "West",
        Direction::North => "North",
        Direction::South => "South",
        Direction::Top => "Up",
        Direction::Bottom => "Down",
        Direction::None => unreachable!("no direction"),
    }
}

fn cell_value(geometry: &Geometry, cell: usize) -> IValue {
    let Position(x, y, z) = geometry.position(cell);
    IValue::cell(x, y, z)
}

/// Distinct symbol names in order, keeping the originals where possible.
fn unique_names<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .enumerate()
        .map(|(index, name)| {
            let mut candidate = name.to_owned();
            if !seen.insert(candidate.clone()) {
                candidate = format!("{name}#{index}");
                seen.insert(candidate.clone());
            }
            candidate
        })
        .collect()
}

fn net_function(values: &[Vec<bool>], net: NetId) -> u64 {
    values[net]
        .iter()
        .enumerate()
        .fold(0u64, |mask, (case, &on)| mask | (u64::from(on) << case))
}

/// A signal that must be observable somewhere.
struct Observation {
    name: String,
    /// Its symbol in the model's `Output` domain.
    symbol: String,
    function: u64,
    class: usize,
    /// Candidate observation cells.
    cells: Vec<usize>,
}

/// The compile-time world of one placement problem, computed and validated
/// in Rust before it becomes an rsdsl instance.
struct Prepared {
    geometry: Geometry,
    classes: Vec<SignalClass>,
    class_names: Vec<String>,
    cases: usize,
    /// Switch candidates of the present inputs as `(cell, attach, net)`.
    sites: Vec<(usize, Direction, NetId)>,
    present_classes: Vec<usize>,
    observations: Vec<Observation>,
}

impl Prepared {
    fn new(netlist: &NorNetlist, config: &ExactPlacerConfig) -> eyre::Result<Self> {
        let geometry = Geometry { dim: config.dim };
        ensure!(geometry.len() > 0, "empty placement box");
        let input_count = netlist.input_names().len();
        ensure!(
            input_count <= MAX_INPUTS,
            "exact placer supports at most {MAX_INPUTS} inputs"
        );
        let classes = vocabulary(netlist);
        let class_names = unique_names(classes.iter().map(|class| class.name.as_str()));
        let mut prepared = Self {
            geometry,
            classes,
            class_names,
            cases: 1 << input_count,
            sites: Vec::new(),
            present_classes: Vec::new(),
            observations: Vec::new(),
        };
        prepared.collect_sites(netlist, config)?;
        prepared.collect_observations(netlist, config)?;
        Ok(prepared)
    }

    fn class_of_net(&self, net: NetId) -> usize {
        self.classes
            .iter()
            .position(|class| class.input == Some(net))
            .expect("every input has a class")
    }

    /// Switch sites per present input, as the hand-written encoder chooses them.
    fn collect_sites(
        &mut self,
        netlist: &NorNetlist,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<()> {
        let geometry = self.geometry;
        for name in netlist.input_names() {
            if config.absent_inputs.contains(&name) {
                continue;
            }
            let net = netlist.input_net(&name).unwrap();
            self.present_classes.push(self.class_of_net(net));
            let candidates = match config.input_sites.get(&name) {
                Some(sites) => sites.clone(),
                None => default_input_sites(config.dim),
            };
            ensure!(
                !candidates.is_empty(),
                "input `{name}` has no candidate site"
            );
            let mut legal = false;
            for (position, attach) in candidates {
                ensure!(
                    config.dim.bound_on(position),
                    "input `{name}` site {position:?} is outside the box"
                );
                let cell = geometry.index(position);
                if geometry.step(cell, attach).is_none() {
                    continue;
                }
                legal = true;
                if !self.sites.contains(&(cell, attach, net)) {
                    self.sites.push((cell, attach, net));
                }
            }
            if !legal {
                bail!("input `{name}` has no legal switch site");
            }
        }
        Ok(())
    }

    /// Observed signals (the outputs, or `config.observations`) and where
    /// each may be observed.
    fn collect_observations(
        &mut self,
        netlist: &NorNetlist,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<()> {
        let values = netlist.net_values();
        let requested = match &config.observations {
            Some(observations) => observations
                .iter()
                .map(|(net, sites)| (netlist.nets[*net].name.clone(), *net, Some(sites.clone())))
                .collect::<Vec<_>>(),
            None => netlist
                .outputs
                .iter()
                .map(|(name, net)| (name.clone(), *net, config.output_sites.get(name).cloned()))
                .collect(),
        };
        let symbols = unique_names(requested.iter().map(|(name, ..)| name.as_str()));
        for ((name, net, positions), symbol) in requested.into_iter().zip(symbols) {
            let function = net_function(&values, net);
            let Some(class) = self
                .classes
                .iter()
                .position(|class| class.function == function)
            else {
                bail!("output `{name}` is constant; the exact placer needs a driven signal");
            };
            let cells = match positions {
                Some(positions) => positions
                    .iter()
                    .map(|position| {
                        ensure!(
                            config.dim.bound_on(*position),
                            "output `{name}` site {position:?} is outside the box"
                        );
                        Ok(self.geometry.index(*position))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?,
                None => (0..self.geometry.len()).collect(),
            };
            self.observations.push(Observation {
                name,
                symbol,
                function,
                class,
                cells,
            });
        }
        Ok(())
    }

    /// Every model param the placer sets. `config.model_params` are applied
    /// after these, so they override both these and the model's defaults.
    fn params(&self, config: &ExactPlacerConfig) -> Vec<(&'static str, IValue)> {
        // The implied torch bound speeds up optimality proofs (AND 2x4x3:
        // about 20% faster) but slowed finding a first layout in
        // measurements, so it is only set when optimizing.
        let min_torches = if config.optimize {
            let targets = self
                .observations
                .iter()
                .map(|o| o.class)
                .collect::<Vec<_>>();
            min_torches(
                &self.classes,
                &self.present_classes,
                &targets,
                config.tuning.torch_bound_max_states,
            )
        } else {
            0
        };
        vec![
            ("rank_levels", config.rank_levels.into()),
            ("stage_levels", config.stage_levels.into()),
            (
                "max_blocks",
                config.max_blocks.map_or(IValue::None, IValue::from),
            ),
            ("allow_unpowered_wires", config.allow_unpowered_wires.into()),
            ("min_torches", min_torches.into()),
        ]
    }

    /// The rsdsl instance: grid, domains, facts, and params.
    fn instance(&self, config: &ExactPlacerConfig) -> eyre::Result<Instance> {
        let geometry = &self.geometry;
        let names = &self.class_names;
        let dim = config.dim;
        let mut instance = Instance::new("exact");
        instance
            .grid("Cell", (dim.0, dim.1, dim.2))
            .domain("Case", (0..self.cases).map(IValue::from))
            .domain("Class", names.iter().map(IValue::sym))
            .domain(
                "Output",
                self.observations.iter().map(|o| IValue::sym(&o.symbol)),
            );
        for fact in [
            "on",
            "unpowered",
            "input_class",
            "present",
            "switch_site",
            "output_site",
            "output_class",
            "driving",
            "fixed",
            "given_sig",
        ] {
            instance.fact(fact);
        }
        for (&position, &function) in &config.given_signals {
            ensure!(
                config.fixed_cells.contains_key(&position),
                "given signal at {position:?} needs a fixed cell"
            );
            let Some(class) = self
                .classes
                .iter()
                .position(|class| class.function == function)
            else {
                bail!("given signal at {position:?} is not in the vocabulary");
            };
            instance.row(
                "given_sig",
                vec![
                    cell_value(geometry, geometry.index(position)),
                    IValue::sym(&names[class]),
                ],
            );
        }
        for (class, name) in self.classes.iter().zip(names).skip(1) {
            for case in 0..self.cases {
                if class.value(case) {
                    instance.row("on", vec![IValue::sym(name), case.into()]);
                }
            }
        }
        instance.row("unpowered", vec![IValue::sym(&names[0])]);
        for (class, name) in self.classes.iter().zip(names) {
            if class.input.is_some() {
                instance.row("input_class", vec![IValue::sym(name)]);
            }
        }
        for &class in &self.present_classes {
            instance.row("present", vec![IValue::sym(&names[class])]);
        }
        for &(cell, attach, net) in &self.sites {
            instance.row(
                "switch_site",
                vec![
                    cell_value(geometry, cell),
                    IValue::sym(attach_name(attach)),
                    IValue::sym(&names[self.class_of_net(net)]),
                ],
            );
        }
        for observation in &self.observations {
            let symbol = IValue::sym(&observation.symbol);
            for &cell in &observation.cells {
                instance.row(
                    "output_site",
                    vec![symbol.clone(), cell_value(geometry, cell)],
                );
            }
            instance.row(
                "output_class",
                vec![symbol.clone(), IValue::sym(&names[observation.class])],
            );
            if config.driving_outputs.contains(&observation.name) {
                instance.row("driving", vec![symbol]);
            }
        }
        for (&position, &kind) in &config.fixed_cells {
            ensure!(
                dim.bound_on(position),
                "fixed cell {position:?} is outside the box"
            );
            let cell = geometry.index(position);
            let member = match kind {
                CellKind::Air => IValue::unit("Air"),
                CellKind::Solid => IValue::unit("Solid"),
                CellKind::Dust => IValue::unit("Dust"),
                CellKind::Torch(attach) => {
                    IValue::member("Torch", vec![IValue::sym(attach_name(attach))])
                }
                CellKind::Repeater(direction) => {
                    IValue::member("Repeater", vec![IValue::sym(direction_name(direction))])
                }
                CellKind::Switch(attach) => {
                    let Some(&(_, _, net)) = self
                        .sites
                        .iter()
                        .find(|&&(c, a, _)| c == cell && a == attach)
                    else {
                        bail!("fixed cell {position:?} cannot hold {kind:?}");
                    };
                    IValue::member(
                        "Switch",
                        vec![
                            IValue::sym(attach_name(attach)),
                            IValue::sym(&names[self.class_of_net(net)]),
                        ],
                    )
                }
            };
            instance.row("fixed", vec![cell_value(geometry, cell), member]);
        }
        for (name, value) in self.params(config) {
            instance.param(name, value);
        }
        for (name, value) in &config.model_params {
            instance.param(name, value.clone());
        }
        Ok(instance)
    }

    /// Reads the grounded families back into the encoder's layout.
    fn read_back(
        &self,
        program: &rsdsl::Program,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<Encoding> {
        let geometry = self.geometry;
        let kind = |cell: usize, member: &str, payload: &[IValue]| {
            program
                .option_lit("Kind", &[cell_value(&geometry, cell)], member, payload)
                .unwrap_or(-1)
        };
        let def = |name: &str, key: &[IValue]| {
            program
                .def_lit(name, key)
                .unwrap_or_else(|| panic!("model has no `{name}` at {key:?}"))
        };
        let mut encoding = Encoding {
            geometry,
            cnf: Cnf::new(),
            classes: self.classes.clone(),
            cases: self.cases,
            air: Vec::new(),
            solid: Vec::new(),
            dust: Vec::new(),
            torch: Vec::new(),
            repeater: Vec::new(),
            is_torch: Vec::new(),
            is_repeater: Vec::new(),
            is_switch: Vec::new(),
            switches: Vec::new(),
            class_lits: Vec::new(),
            values: Vec::new(),
            conn: Vec::new(),
            points: Vec::new(),
            hard: Vec::new(),
            relations: Vec::new(),
            ranks: Vec::new(),
            stages: Vec::new(),
            output_sites: Vec::new(),
            block_lits: Vec::new(),
            sections: Vec::new(),
            relaxations: Vec::new(),
            coverage_relaxations: Vec::new(),
            observed: self
                .observations
                .iter()
                .map(|o| (o.name.clone(), o.function))
                .collect(),
            program: None,
            objective: None,
        };
        for cell in 0..geometry.len() {
            let at = cell_value(&geometry, cell);
            let here = std::slice::from_ref(&at);
            let air = kind(cell, "Air", &[]);
            encoding.air.push(air);
            encoding.block_lits.push(-air);
            encoding.solid.push(kind(cell, "Solid", &[]));
            encoding.dust.push(kind(cell, "Dust", &[]));
            encoding.torch.push(
                TORCH_ATTACH.map(|attach| kind(cell, "Torch", &[IValue::sym(attach_name(attach))])),
            );
            encoding.repeater.push(CARDINALS.map(|direction| {
                kind(cell, "Repeater", &[IValue::sym(direction_name(direction))])
            }));
            encoding.class_lits.push(
                self.class_names
                    .iter()
                    .map(|name| {
                        program
                            .option_lit("Sig", here, "Carry", &[IValue::sym(name)])
                            .expect("every class is a signal option")
                    })
                    .collect(),
            );
            encoding.values.push(
                (0..self.cases)
                    .map(|case| def("Powered", &[case.into(), at.clone()]))
                    .collect(),
            );
            let per_direction = |name: &str| {
                CARDINALS.map(|direction| {
                    def(name, &[at.clone(), IValue::sym(direction_name(direction))])
                })
            };
            encoding.conn.push(per_direction("Conn"));
            encoding.points.push(per_direction("Points"));
            encoding.hard.push(def("Hard", here));
            if config.rank_levels > 0 {
                encoding.ranks.push(program.int_lits("Rank", here).unwrap());
            }
            if config.stage_levels > 0 {
                encoding
                    .stages
                    .push(program.int_lits("Stage", here).unwrap());
            }
        }
        for &(cell, attach, net) in &self.sites {
            let class = &self.class_names[self.class_of_net(net)];
            let lit = kind(
                cell,
                "Switch",
                &[IValue::sym(attach_name(attach)), IValue::sym(class)],
            );
            encoding.switches.push(SwitchSite {
                cell,
                attach,
                net,
                lit,
            });
        }

        let cell_index = |value: &IValue| match *value {
            IValue::Cell(x, y, z) => geometry.index(Position(x as usize, y as usize, z as usize)),
            ref other => unreachable!("expected a cell, found {other:?}"),
        };
        for (tuple, lit) in program.relation("Feeds").expect("model declares Feeds") {
            let [source, sink, IValue::Sym(from), IValue::Sym(to)] = tuple.as_slice() else {
                unreachable!("Feeds(src, dst, from, to)");
            };
            let source_kind = match from.as_str() {
                "FromDust" => SourceKind::Dust,
                "FromTorch" => SourceKind::Torch,
                "FromRepeater" => SourceKind::Repeater,
                "FromSolid" => SourceKind::Solid,
                "FromSwitch" => SourceKind::Switch,
                other => unreachable!("unknown source kind {other}"),
            };
            let sink_kind = match to.as_str() {
                "DustSink" => SinkKind::Dust,
                "RepeaterSink" => SinkKind::Repeater,
                "SolidSink" => SinkKind::Solid,
                other => unreachable!("unknown sink kind {other}"),
            };
            encoding.relations.push(Relation {
                source: cell_index(source),
                sink: cell_index(sink),
                source_kind,
                sink_kind,
                lit,
            });
        }

        for observation in &self.observations {
            let before = encoding.output_sites.len();
            for &cell in &observation.cells {
                let lit = def(
                    "Observe",
                    &[
                        IValue::sym(&observation.symbol),
                        cell_value(&geometry, cell),
                    ],
                );
                if lit != -1 {
                    encoding.output_sites.push(OutputSite {
                        name: observation.name.clone(),
                        cell,
                        lit,
                    });
                }
            }
            ensure!(
                encoding.output_sites.len() > before,
                "output `{}` has no legal site",
                observation.name
            );
        }

        for guard in program.guards() {
            if guard.rule == SOUNDNESS_RULE {
                encoding.relaxations.push(guard.lit);
            } else if guard.rule == COVERAGE_RULE {
                encoding
                    .coverage_relaxations
                    .push((cell_index(&guard.key[0].1), guard.lit));
            }
        }
        if config.relax_soundness {
            ensure!(
                encoding.relaxations.len() == encoding.relations.len(),
                "one soundness guard per relation"
            );
        }
        Ok(encoding)
    }
}

/// Grounds the configured model (the built-in one unless `model_file` is set).
fn ground(
    config: &ExactPlacerConfig,
    instance: &Instance,
    provenance: bool,
) -> eyre::Result<rsdsl::Program> {
    let options = GroundOptions {
        guards: config.relax_soundness,
        provenance,
        positive_or_aux: false,
    };
    let custom;
    let model = match &config.model_file {
        Some(path) => {
            let source = std::fs::read_to_string(path)
                .map_err(|error| eyre!("cannot read model {}: {error}", path.display()))?;
            custom = Model::parse(&path.display().to_string(), &source)
                .map_err(|error| eyre!("{error}"))?;
            &custom
        }
        None => model(),
    };
    model
        .ground(instance, options)
        .map_err(|error| eyre!("grounding {} failed:\n{error}", model.name()))
}

impl Encoding {
    pub(super) fn build_dsl(
        netlist: &NorNetlist,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<Self> {
        Self::build_dsl_with(netlist, config, false)
    }

    /// `provenance` records which rule instance emitted each clause, for
    /// explained DIMACS exports.
    pub(super) fn build_dsl_with(
        netlist: &NorNetlist,
        config: &ExactPlacerConfig,
        provenance: bool,
    ) -> eyre::Result<Self> {
        let prepared = Prepared::new(netlist, config)?;
        let instance = prepared.instance(config)?;
        let mut program = ground(config, &instance, provenance)?;
        let mut encoding = prepared.read_back(&program, config)?;

        if config.optimize {
            let objective = program
                .objective()
                .ok_or_else(|| eyre!("optimize needs a `minimize` objective in the model"))?;
            let total = objective.total_weight();
            ensure!(
                total <= config.tuning.max_objective_weight,
                "objective weights sum to {total}, above tuning.max_objective_weight ({})",
                config.tuning.max_objective_weight
            );
            let at_least = program.objective_counter(total);
            encoding.objective = Some(ObjectiveBound {
                objective,
                at_least,
            });
        }

        // The fixed-cell rule silently fails on impossible kinds; report them.
        let geometry = prepared.geometry;
        for (&position, &kind) in &config.fixed_cells {
            if encoding.kind_lit(geometry.index(position), kind).is_none() {
                bail!("fixed cell {position:?} cannot hold {kind:?}");
            }
        }
        for layout in &config.blocked {
            let mut clause = Vec::new();
            let mut excluded = false;
            for &(position, kind) in layout {
                if !config.dim.bound_on(position) {
                    continue;
                }
                match encoding.kind_lit(geometry.index(position), kind) {
                    Some(lit) => clause.push(-lit),
                    // The cell cannot hold that kind here, so the layout is
                    // already excluded.
                    None => {
                        excluded = true;
                        break;
                    }
                }
            }
            if !excluded {
                program.add_clause(&clause, "이전에 거부된 배치는 다시 쓰지 않음");
            }
        }

        let mut total = 0;
        encoding.sections = program
            .rule_clause_counts()
            .into_iter()
            .map(|(rule, clauses)| {
                total += clauses;
                (rule, program.num_vars(), total)
            })
            .collect();
        encoding.cnf = Cnf::from_literals(
            program.num_vars(),
            program.literals().to_vec(),
            program.clause_count(),
        );
        encoding.program = Some(Box::new(program));
        Ok(encoding)
    }
}

/// A lower bound on the torches any layout needs to produce `targets` from
/// the `inputs` classes. Every cell carries a vocabulary class, so a solid
/// carries an OR of available classes only when that OR is itself a class,
/// and a torch carries the complement of its support's class: a torch is a
/// NOR gate over vocabulary functions. Breadth-first search over sets of
/// available classes finds the fewest torches; if it would visit more than
/// `max_states` sets, the depth reached so far is returned (still a bound).
pub(super) fn min_torches(
    classes: &[SignalClass],
    inputs: &[usize],
    targets: &[usize],
    max_states: usize,
) -> usize {
    if classes.len() > MAX_BOUND_CLASSES {
        return 0;
    }
    let goal = targets.iter().fold(0u64, |set, &class| set | 1 << class);
    // Add every class that is the OR of available classes below it.
    let close = |mut set: u64| loop {
        let mut grown = set;
        for (index, class) in classes.iter().enumerate().skip(1) {
            if grown & 1 << index != 0 {
                continue;
            }
            let union = classes
                .iter()
                .enumerate()
                .filter(|(other, c)| set & 1 << other != 0 && c.function & !class.function == 0)
                .fold(0u64, |union, (_, c)| union | c.function);
            if union == class.function {
                grown |= 1 << index;
            }
        }
        if grown == set {
            return set;
        }
        set = grown;
    };
    let start = close(inputs.iter().fold(1u64, |set, &class| set | 1 << class));
    if start & goal == goal {
        return 0;
    }
    let mut seen = std::collections::HashSet::from([start]);
    let mut layer = vec![start];
    for depth in 1.. {
        let mut next = Vec::new();
        for &set in &layer {
            for (index, class) in classes.iter().enumerate().skip(1) {
                let Some(complement) = class.complement else {
                    continue;
                };
                if set & 1 << index == 0 || set & 1 << complement != 0 {
                    continue;
                }
                let grown = close(set | 1 << complement);
                if grown & goal == goal {
                    return depth;
                }
                if seen.insert(grown) {
                    next.push(grown);
                }
            }
            if seen.len() > max_states {
                return depth;
            }
        }
        if next.is_empty() {
            return depth;
        }
        layer = next;
    }
    unreachable!()
}
