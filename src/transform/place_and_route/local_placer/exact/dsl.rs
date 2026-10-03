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
    default_input_sites, vocabulary, CellKind, Encoding, Geometry, OutputSite, Relation, SinkKind,
    SourceKind, SwitchSite, CARDINALS, TORCH_ATTACH,
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
        let geometry = Geometry { dim: config.dim };
        ensure!(geometry.len() > 0, "empty placement box");
        let input_count = netlist.input_names().len();
        ensure!(input_count <= 6, "exact placer supports at most 6 inputs");
        let classes = vocabulary(netlist);
        let cases = 1usize << input_count;
        let class_names = unique_names(classes.iter().map(|class| class.name.as_str()));
        let class_of_net = |net: NetId| {
            classes
                .iter()
                .position(|class| class.input == Some(net))
                .expect("every input has a class")
        };

        let mut instance = Instance::new("exact");
        let dim = config.dim;
        instance
            .grid("Cell", (dim.0, dim.1, dim.2))
            .domain("Case", (0..cases).map(IValue::from))
            .domain("Class", class_names.iter().map(IValue::sym))
            .param("rank_levels", config.rank_levels)
            .param("stage_levels", config.stage_levels)
            .param(
                "max_blocks",
                config.max_blocks.map_or(IValue::None, IValue::from),
            )
            .param("allow_unpowered_wires", config.allow_unpowered_wires);
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
        ] {
            instance.fact(fact);
        }
        for (class, name) in classes.iter().zip(&class_names).skip(1) {
            for case in 0..cases {
                if class.value(case) {
                    instance.row("on", vec![IValue::sym(name), case.into()]);
                }
            }
        }
        instance.row("unpowered", vec![IValue::sym(&class_names[0])]);
        for (class, name) in classes.iter().zip(&class_names) {
            if class.input.is_some() {
                instance.row("input_class", vec![IValue::sym(name)]);
            }
        }

        // Switch sites per present input, as the hand-written encoder chooses them.
        let mut sites = Vec::<(usize, Direction, NetId)>::new();
        for name in netlist.input_names() {
            if config.absent_inputs.contains(&name) {
                continue;
            }
            let net = netlist.input_net(&name).unwrap();
            instance.row(
                "present",
                vec![IValue::sym(&class_names[class_of_net(net)])],
            );
            let candidates = match config.input_sites.get(&name) {
                Some(sites) => sites.clone(),
                None => default_input_sites(dim),
            };
            ensure!(
                !candidates.is_empty(),
                "input `{name}` has no candidate site"
            );
            let mut legal = false;
            for (position, attach) in candidates {
                ensure!(
                    dim.bound_on(position),
                    "input `{name}` site {position:?} is outside the box"
                );
                let cell = geometry.index(position);
                if geometry.step(cell, attach).is_none() {
                    continue;
                }
                legal = true;
                if !sites.contains(&(cell, attach, net)) {
                    sites.push((cell, attach, net));
                }
            }
            if !legal {
                bail!("input `{name}` has no legal switch site");
            }
        }
        for &(cell, attach, net) in &sites {
            instance.row(
                "switch_site",
                vec![
                    cell_value(&geometry, cell),
                    IValue::sym(attach_name(attach)),
                    IValue::sym(&class_names[class_of_net(net)]),
                ],
            );
        }

        // Observed signals and where they may be observed.
        let values = netlist.net_values();
        let observations = match &config.observations {
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
        let output_names = unique_names(observations.iter().map(|(name, ..)| name.as_str()));
        instance.domain("Output", output_names.iter().map(IValue::sym));
        let mut observed = Vec::new();
        let mut observation_cells = Vec::new();
        for ((name, net, positions), symbol) in observations.iter().zip(&output_names) {
            let function = net_function(&values, *net);
            let Some(class) = classes.iter().position(|class| class.function == function) else {
                bail!("output `{name}` is constant; the exact placer needs a driven signal");
            };
            observed.push((name.clone(), function));
            let cells = match positions {
                Some(positions) => positions
                    .iter()
                    .map(|position| {
                        ensure!(
                            dim.bound_on(*position),
                            "output `{name}` site {position:?} is outside the box"
                        );
                        Ok(geometry.index(*position))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?,
                None => (0..geometry.len()).collect(),
            };
            for &cell in &cells {
                instance.row(
                    "output_site",
                    vec![IValue::sym(symbol), cell_value(&geometry, cell)],
                );
            }
            instance.row(
                "output_class",
                vec![IValue::sym(symbol), IValue::sym(&class_names[class])],
            );
            if config.driving_outputs.contains(name) {
                instance.row("driving", vec![IValue::sym(symbol)]);
            }
            observation_cells.push((name.clone(), symbol.clone(), cells));
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
                    let Some(&(_, _, net)) =
                        sites.iter().find(|&&(c, a, _)| c == cell && a == attach)
                    else {
                        bail!("fixed cell {position:?} cannot hold {kind:?}");
                    };
                    IValue::member(
                        "Switch",
                        vec![
                            IValue::sym(attach_name(attach)),
                            IValue::sym(&class_names[class_of_net(net)]),
                        ],
                    )
                }
            };
            instance.row("fixed", vec![cell_value(&geometry, cell), member]);
        }

        for (name, value) in &config.model_params {
            instance.param(name, value.clone());
        }

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
        let mut program = model
            .ground(&instance, options)
            .map_err(|error| eyre!("grounding {} failed:\n{error}", model.name()))?;

        // Read the grounded families back into the encoder's layout.
        let kind = |program: &rsdsl::Program, cell: usize, member: &str, payload: &[IValue]| {
            program
                .option_lit("Kind", &[cell_value(&geometry, cell)], member, payload)
                .unwrap_or(-1)
        };
        let def = |program: &rsdsl::Program, name: &str, key: &[IValue]| {
            program
                .def_lit(name, key)
                .unwrap_or_else(|| panic!("model has no `{name}` at {key:?}"))
        };
        let mut encoding = Self {
            geometry,
            cnf: Cnf::new(),
            classes: classes.clone(),
            cases,
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
            observed,
            program: None,
        };
        for cell in 0..geometry.len() {
            let at = cell_value(&geometry, cell);
            let air = kind(&program, cell, "Air", &[]);
            encoding.air.push(air);
            encoding.block_lits.push(-air);
            encoding.solid.push(kind(&program, cell, "Solid", &[]));
            encoding.dust.push(kind(&program, cell, "Dust", &[]));
            encoding.torch.push(
                TORCH_ATTACH.map(|attach| {
                    kind(&program, cell, "Torch", &[IValue::sym(attach_name(attach))])
                }),
            );
            encoding.repeater.push(CARDINALS.map(|direction| {
                kind(
                    &program,
                    cell,
                    "Repeater",
                    &[IValue::sym(direction_name(direction))],
                )
            }));
            encoding.class_lits.push(
                class_names
                    .iter()
                    .map(|name| {
                        program
                            .option_lit(
                                "Sig",
                                std::slice::from_ref(&at),
                                "Carry",
                                &[IValue::sym(name)],
                            )
                            .expect("every class is a signal option")
                    })
                    .collect(),
            );
            encoding.values.push(
                (0..cases)
                    .map(|case| def(&program, "Powered", &[case.into(), at.clone()]))
                    .collect(),
            );
            let per_direction = |name: &str| {
                CARDINALS.map(|direction| {
                    def(
                        &program,
                        name,
                        &[at.clone(), IValue::sym(direction_name(direction))],
                    )
                })
            };
            encoding.conn.push(per_direction("Conn"));
            encoding.points.push(per_direction("Points"));
            encoding
                .hard
                .push(def(&program, "Hard", std::slice::from_ref(&at)));
            if config.rank_levels > 0 {
                encoding
                    .ranks
                    .push(program.int_lits("Rank", std::slice::from_ref(&at)).unwrap());
            }
            if config.stage_levels > 0 {
                encoding.stages.push(
                    program
                        .int_lits("Stage", std::slice::from_ref(&at))
                        .unwrap(),
                );
            }
        }
        for &(cell, attach, net) in &sites {
            let class = &class_names[class_of_net(net)];
            let lit = kind(
                &program,
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

        for (name, symbol, cells) in &observation_cells {
            let before = encoding.output_sites.len();
            for &cell in cells {
                let lit = def(
                    &program,
                    "Observe",
                    &[IValue::sym(symbol), cell_value(&geometry, cell)],
                );
                if lit != -1 {
                    encoding.output_sites.push(OutputSite {
                        name: name.clone(),
                        cell,
                        lit,
                    });
                }
            }
            ensure!(
                encoding.output_sites.len() > before,
                "output `{name}` has no legal site"
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

        // The fixed-cell rule silently fails on impossible kinds; report them.
        for (&position, &kind) in &config.fixed_cells {
            if encoding.kind_lit(geometry.index(position), kind).is_none() {
                bail!("fixed cell {position:?} cannot hold {kind:?}");
            }
        }
        for layout in &config.blocked {
            let mut clause = Vec::new();
            let mut excluded = false;
            for &(position, kind) in layout {
                if !dim.bound_on(position) {
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
