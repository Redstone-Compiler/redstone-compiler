//! CNF model of a redstone cell over a fixed box.
//!
//! Every cell picks one block kind, and every non-air cell picks a signal
//! class. A class is a Boolean function of the cell inputs taken from a
//! vocabulary: the netlist's net functions and their complements (class 0 is
//! "unpowered"). Per-case power literals `val(cell, case)` follow the class.
//! A structural power relation `x -> y` says that, given the chosen blocks,
//! the simulator delivers power from `x` to `y`. The model requires:
//!
//! * soundness: if `x` is powered then `y` is; dust and repeaters carry exactly
//!   their source's function, while a solid block may OR several sources;
//! * coverage: a powered element that is not a driver has, in every case where
//!   it is on, a powered source with a smaller local rank and a stage that is
//!   not larger, so power is traced back to a driver without loops;
//! * torches and repeaters: a torch carries the complement of its support and
//!   a repeater its input; both sit on a strictly larger stage than what feeds
//!   them (repeaters restart the local rank), so latches, clocks, and repeater
//!   memory loops are impossible;
//! * every input has one switch and every output is observed somewhere.
//!
//! Netlist gates are not required individually. Any composition of vocabulary
//! functions is functionally correct by construction, so the solver may
//! duplicate gates, build inverter towers, or OR signals on a shared block.
//!
//! The relations mirror `world::simulator` (dust shape and step connections,
//! weak/strong block power, torch and repeater outputs). Behavior the model
//! cannot express precisely is forbidden instead, and every solution is still
//! checked with the simulator.

use std::collections::BTreeMap;

use eyre::{bail, ensure};

use super::cnf::{Cnf, Lit};
use super::netlist::{NetDriver, NetId, NorNetlist};
use super::ExactPlacerConfig;
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

pub(super) const CARDINALS: [Direction; 4] = [
    Direction::East,
    Direction::West,
    Direction::North,
    Direction::South,
];
pub(super) const TORCH_ATTACH: [Direction; 5] = [
    Direction::Bottom,
    Direction::East,
    Direction::West,
    Direction::North,
    Direction::South,
];
const ALL_DIRECTIONS: [Direction; 6] = [
    Direction::East,
    Direction::West,
    Direction::North,
    Direction::South,
    Direction::Top,
    Direction::Bottom,
];

fn cardinal_index(direction: Direction) -> usize {
    CARDINALS
        .iter()
        .position(|&d| d == direction)
        .expect("cardinal direction")
}

fn perpendicular(direction: Direction) -> [Direction; 2] {
    match direction {
        Direction::East | Direction::West => [Direction::North, Direction::South],
        _ => [Direction::East, Direction::West],
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CellKind {
    Air,
    Solid,
    Dust,
    /// Torch attached toward the given direction (`Bottom` stands on a block).
    Torch(Direction),
    /// Repeater whose input is at `walk(direction)`; output at the opposite side.
    Repeater(Direction),
    /// Input switch attached toward the given direction.
    Switch(Direction),
}

/// A signal class: one Boolean function of the inputs.
#[derive(Debug, Clone)]
pub(super) struct SignalClass {
    pub(super) name: String,
    /// Bit `k` is the value in input case `k`.
    pub(super) function: u64,
    pub(super) complement: Option<usize>,
    pub(super) input: Option<NetId>,
}

impl SignalClass {
    pub(super) fn value(&self, case: usize) -> bool {
        self.function & (1 << case) != 0
    }
}

/// Unpowered, every net function, and every complement, without duplicates.
pub(super) fn vocabulary(netlist: &NorNetlist) -> Vec<SignalClass> {
    let values = netlist.net_values();
    let cases = 1usize << netlist.input_names().len();
    let full = if cases >= 64 {
        u64::MAX
    } else {
        (1u64 << cases) - 1
    };
    let mask = |vector: &[bool]| {
        vector
            .iter()
            .enumerate()
            .fold(0u64, |mask, (case, &on)| mask | (u64::from(on) << case))
    };
    let mut classes = vec![SignalClass {
        name: "unpowered".to_owned(),
        function: 0,
        complement: None,
        input: None,
    }];
    let add = |classes: &mut Vec<SignalClass>, name: String, function: u64| {
        if function == 0 || function == full {
            return;
        }
        if classes.iter().any(|class| class.function == function) {
            return;
        }
        classes.push(SignalClass {
            name,
            function,
            complement: None,
            input: None,
        });
    };
    for (net, vector) in values.iter().enumerate() {
        add(&mut classes, netlist.nets[net].name.clone(), mask(vector));
    }
    for (net, vector) in values.iter().enumerate() {
        add(
            &mut classes,
            format!("~{}", netlist.nets[net].name),
            !mask(vector) & full,
        );
    }
    for index in 1..classes.len() {
        let complement = !classes[index].function & full;
        classes[index].complement = classes
            .iter()
            .position(|class| class.function == complement);
    }
    for (net, vector) in values.iter().enumerate() {
        if matches!(netlist.nets[net].driver, NetDriver::Input(_)) {
            let function = mask(vector);
            if let Some(class) = classes.iter_mut().find(|class| class.function == function) {
                class.input = Some(net);
            }
        }
    }
    classes
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum SinkKind {
    Dust,
    Repeater,
    Solid,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum SourceKind {
    Dust,
    Torch,
    Repeater,
    Solid,
    Switch,
}

#[derive(Debug, Clone)]
pub(super) struct Relation {
    pub(super) source: usize,
    pub(super) sink: usize,
    pub(super) source_kind: SourceKind,
    pub(super) sink_kind: SinkKind,
    pub(super) lit: Lit,
}

#[derive(Debug, Clone)]
pub(super) struct SwitchSite {
    pub(super) cell: usize,
    pub(super) attach: Direction,
    pub(super) net: NetId,
    pub(super) lit: Lit,
}

#[derive(Debug, Clone)]
pub(super) struct OutputSite {
    pub(super) name: String,
    pub(super) cell: usize,
    pub(super) lit: Lit,
}

#[derive(Debug, Copy, Clone)]
pub(super) struct Geometry {
    pub(super) dim: DimSize,
}

impl Geometry {
    pub(super) fn len(&self) -> usize {
        self.dim.0 * self.dim.1 * self.dim.2
    }

    pub(super) fn index(&self, position: Position) -> usize {
        position.index(&self.dim).0
    }

    pub(super) fn position(&self, index: usize) -> Position {
        let x = index % self.dim.0;
        let y = (index / self.dim.0) % self.dim.1;
        let z = index / (self.dim.0 * self.dim.1);
        Position(x, y, z)
    }

    pub(super) fn step(&self, index: usize, direction: Direction) -> Option<usize> {
        let position = self.position(index).walk(direction)?;
        self.dim.bound_on(position).then(|| self.index(position))
    }
}

pub(super) struct Encoding {
    pub(super) geometry: Geometry,
    pub(super) cnf: Cnf,
    pub(super) classes: Vec<SignalClass>,
    pub(super) cases: usize,
    pub(super) air: Vec<Lit>,
    pub(super) solid: Vec<Lit>,
    pub(super) dust: Vec<Lit>,
    pub(super) torch: Vec<[Lit; 5]>,
    pub(super) repeater: Vec<[Lit; 4]>,
    /// Per-cell "any torch/repeater/switch" literals (legacy encoder only).
    pub(super) is_torch: Vec<Lit>,
    pub(super) is_repeater: Vec<Lit>,
    pub(super) is_switch: Vec<Lit>,
    pub(super) switches: Vec<SwitchSite>,
    pub(super) class_lits: Vec<Vec<Lit>>,
    /// `values[cell][case]`: the cell is powered in that input case.
    pub(super) values: Vec<Vec<Lit>>,
    pub(super) conn: Vec<[Lit; 4]>,
    pub(super) points: Vec<[Lit; 4]>,
    pub(super) hard: Vec<Lit>,
    pub(super) relations: Vec<Relation>,
    pub(super) ranks: Vec<Vec<Lit>>,
    pub(super) stages: Vec<Vec<Lit>>,
    pub(super) output_sites: Vec<OutputSite>,
    pub(super) block_lits: Vec<Lit>,
    pub(super) sections: Vec<(String, i32, usize)>,
    /// Per-relation soundness guards (only with `relax_soundness`).
    pub(super) relaxations: Vec<Lit>,
    /// Per-cell coverage guards (only with `relax_soundness`).
    pub(super) coverage_relaxations: Vec<(usize, Lit)>,
    /// Required observations as `(name, function)`.
    pub(super) observed: Vec<(String, u64)>,
    /// The grounded rsdsl program behind `cnf` (absent for the legacy encoder).
    pub(super) program: Option<Box<rsdsl::Program>>,
    /// The model's cost and its incremental bound (with `optimize`).
    pub(super) objective: Option<ObjectiveBound>,
}

/// `at_least[j]` is implied by `cost - offset >= j + 1`; assuming
/// `-at_least[b]` asks for a layout of cost at most `offset + b`.
pub(super) struct ObjectiveBound {
    pub(super) objective: rsdsl::Objective,
    pub(super) at_least: Vec<Lit>,
}

impl ObjectiveBound {
    /// Assumptions for "cost below `best`", or `None` when nothing cheaper
    /// is possible.
    pub(super) fn below(&self, best: i64) -> Option<Vec<Lit>> {
        let bound = best - 1 - self.objective.offset;
        if bound < 0 {
            return None;
        }
        Some(match self.at_least.get(bound as usize) {
            Some(&lit) => vec![-lit],
            None => Vec::new(),
        })
    }
}

impl Encoding {
    pub(super) fn build(netlist: &NorNetlist, config: &ExactPlacerConfig) -> eyre::Result<Self> {
        if config.legacy_encoder {
            ensure!(
                !config.optimize,
                "the legacy encoder has no objective; optimize needs the rsdsl model"
            );
            Self::build_legacy(netlist, config)
        } else {
            Self::build_dsl(netlist, config)
        }
    }

    /// The hand-written encoding that `exact_placer.rsdsl` replaces; kept to
    /// check that both produce equivalent formulas.
    fn build_legacy(netlist: &NorNetlist, config: &ExactPlacerConfig) -> eyre::Result<Self> {
        let geometry = Geometry { dim: config.dim };
        ensure!(geometry.len() > 0, "empty placement box");
        let input_count = netlist.input_names().len();
        ensure!(input_count <= 6, "exact placer supports at most 6 inputs");
        let mut encoding = Self {
            geometry,
            cnf: Cnf::new(),
            classes: vocabulary(netlist),
            cases: 1 << input_count,
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
            observed: Vec::new(),
            program: None,
            objective: None,
        };
        let mut sections = Vec::new();
        let mark = |name: &'static str, cnf: &Cnf, sections: &mut Vec<_>| {
            sections.push((name.to_owned(), cnf.num_vars(), cnf.clause_count()));
        };
        encoding.encode_kinds(netlist, config)?;
        encoding.encode_fixed_cells(config)?;
        mark("kinds", &encoding.cnf, &mut sections);
        encoding.encode_classes(config.allow_unpowered_wires);
        mark("classes", &encoding.cnf, &mut sections);
        encoding.encode_dust_shape();
        mark("dust_shape", &encoding.cnf, &mut sections);
        encoding.encode_relations();
        mark("relations", &encoding.cnf, &mut sections);
        encoding.encode_soundness(config.relax_soundness);
        mark("soundness", &encoding.cnf, &mut sections);
        encoding.encode_coverage(
            config.rank_levels,
            config.stage_levels,
            config.relax_soundness,
        );
        mark("coverage", &encoding.cnf, &mut sections);
        encoding.encode_torches(config.stage_levels);
        mark("torches", &encoding.cnf, &mut sections);
        encoding.encode_outputs(netlist, config)?;
        mark("outputs", &encoding.cnf, &mut sections);
        if let Some(max_blocks) = config.max_blocks {
            encoding.encode_block_limit(max_blocks);
            mark("block_limit", &encoding.cnf, &mut sections);
        }
        encoding.sections = sections;
        Ok(encoding)
    }

    fn class_of_function(&self, function: u64) -> Option<usize> {
        self.classes
            .iter()
            .position(|class| class.function == function)
    }

    fn input_class(&self, net: NetId) -> usize {
        self.classes
            .iter()
            .position(|class| class.input == Some(net))
            .expect("every input has a class")
    }

    fn encode_kinds(
        &mut self,
        netlist: &NorNetlist,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<()> {
        let geometry = self.geometry;
        let fals = self.cnf.fals();
        let mut sites_by_cell = BTreeMap::<usize, Vec<(Direction, NetId)>>::new();
        for name in netlist.input_names() {
            if config.absent_inputs.contains(&name) {
                continue;
            }
            let net = netlist.input_net(&name).unwrap();
            let sites = match config.input_sites.get(&name) {
                Some(sites) => sites.clone(),
                None => default_input_sites(geometry.dim),
            };
            ensure!(!sites.is_empty(), "input `{name}` has no candidate site");
            for (position, attach) in sites {
                ensure!(
                    geometry.dim.bound_on(position),
                    "input `{name}` site {position:?} is outside the box"
                );
                let cell = geometry.index(position);
                if geometry.step(cell, attach).is_none() {
                    continue;
                }
                sites_by_cell.entry(cell).or_default().push((attach, net));
            }
        }

        for cell in 0..geometry.len() {
            let position = geometry.position(cell);
            let raised = position.2 >= 1;
            // Air is the negation of an "occupied" variable so that a false
            // initial phase starts the search from an empty box.
            let air = -self.cnf.new_var();
            let solid = self.cnf.new_var();
            let dust = if raised { self.cnf.new_var() } else { fals };
            let mut torch = [fals; 5];
            for (index, attach) in TORCH_ATTACH.into_iter().enumerate() {
                let supported = if attach == Direction::Bottom {
                    raised
                } else {
                    geometry.step(cell, attach).is_some()
                };
                if supported {
                    torch[index] = self.cnf.new_var();
                }
            }
            let mut repeater = [fals; 4];
            if raised {
                for (slot, direction) in repeater.iter_mut().zip(CARDINALS) {
                    // A repeater must read from inside the box. One that emits
                    // out of the box is only useful as an output observation.
                    let reads_inside = geometry.step(cell, direction).is_some();
                    let emits_inside = geometry.step(cell, direction.inverse()).is_some();
                    if reads_inside && (emits_inside || may_observe_output(config, position)) {
                        *slot = self.cnf.new_var();
                    }
                }
            }
            let mut kind_lits = vec![air, solid, dust];
            kind_lits.extend(torch);
            kind_lits.extend(repeater);
            let mut switch_lits = Vec::new();
            for &(attach, net) in sites_by_cell.get(&cell).into_iter().flatten() {
                let lit = self.cnf.new_var();
                switch_lits.push(lit);
                kind_lits.push(lit);
                self.switches.push(SwitchSite {
                    cell,
                    attach,
                    net,
                    lit,
                });
            }
            let kind_lits = kind_lits
                .into_iter()
                .filter(|&lit| !self.cnf.is_false(lit))
                .collect::<Vec<_>>();
            self.cnf
                .rule("칸마다 블록 종류는 정확히 하나 (공기/블록/가루/토치/리피터/스위치)");
            self.cnf.exactly_one(&kind_lits);

            self.air.push(air);
            self.solid.push(solid);
            self.dust.push(dust);
            self.torch.push(torch);
            self.repeater.push(repeater);
            self.cnf
                .rule("보조: 이 칸이 (어느 방향이든) 토치/리피터/스위치인가");
            let is_torch = self.cnf.or(&torch);
            let is_repeater = self.cnf.or(&repeater);
            let is_switch = self.cnf.or(&switch_lits);
            self.is_torch.push(is_torch);
            self.is_repeater.push(is_repeater);
            self.is_switch.push(is_switch);
            self.block_lits.push(-air);
        }

        // Physical support.
        for cell in 0..geometry.len() {
            if let Some(below) = geometry.step(cell, Direction::Bottom) {
                self.cnf
                    .rule("가루·리피터·바닥 토치는 아래 칸이 블록이어야 함");
                let below_solid = self.solid[below];
                let dust = self.dust[cell];
                self.cnf.implies(&[dust], &[below_solid]);
                for &repeater in &self.repeater[cell].clone() {
                    self.cnf.implies(&[repeater], &[below_solid]);
                }
                let torch = self.torch[cell][0];
                self.cnf.implies(&[torch], &[below_solid]);
            }
            for (index, attach) in TORCH_ATTACH.into_iter().enumerate().skip(1) {
                if let Some(support) = geometry.step(cell, attach) {
                    self.cnf.rule("벽 토치는 붙은 쪽 칸이 블록이어야 함");
                    let torch = self.torch[cell][index];
                    let support_solid = self.solid[support];
                    self.cnf.implies(&[torch], &[support_solid]);
                }
            }
        }
        for site in self.switches.clone() {
            let support = geometry.step(site.cell, site.attach).unwrap();
            self.cnf.rule("스위치는 붙은 쪽 칸이 블록이어야 함");
            let support_solid = self.solid[support];
            self.cnf.implies(&[site.lit], &[support_solid]);
        }

        self.cnf.rule("입력마다 스위치는 정확히 하나");
        // Each present input has exactly one switch.
        for name in netlist.input_names() {
            if config.absent_inputs.contains(&name) {
                continue;
            }
            let net = netlist.input_net(&name).unwrap();
            let lits = self
                .switches
                .iter()
                .filter(|site| site.net == net)
                .map(|site| site.lit)
                .collect::<Vec<_>>();
            if lits.is_empty() {
                bail!("input `{name}` has no legal switch site");
            }
            self.cnf.exactly_one(&lits);
        }
        Ok(())
    }

    fn encode_classes(&mut self, allow_unpowered_wires: bool) {
        let class_count = self.classes.len();
        for cell in 0..self.geometry.len() {
            let lits = (0..class_count)
                .map(|_| self.cnf.new_var())
                .collect::<Vec<_>>();
            let mut one_hot = vec![self.air[cell]];
            one_hot.extend(lits.iter().copied());
            self.cnf
                .rule("차 있는 칸은 신호를 정확히 하나 실음 (공기는 신호 없음)");
            self.cnf.exactly_one(&one_hot);

            let solid = self.solid[cell];
            let is_switch = self.is_switch[cell];
            self.cnf
                .rule("'꺼짐' 신호는 블록만 (허용 시 죽은 가루/리피터도)");
            // A torch is never unpowered because its support is never constant.
            if allow_unpowered_wires {
                self.cnf.implies(
                    &[lits[0]],
                    &[solid, self.dust[cell], self.is_repeater[cell]],
                );
            } else {
                self.cnf.implies(&[lits[0]], &[solid]);
            }
            for (class, &lit) in lits.iter().enumerate().skip(1) {
                self.cnf.rule("스위치는 입력 신호만 실을 수 있음");
                if self.classes[class].input.is_none() {
                    self.cnf.implies(&[lit, is_switch], &[]);
                }
            }
            self.cnf.rule("토치는 반드시 꺼짐이 아닌 신호를 실음");
            let mut clause = vec![-self.is_torch[cell]];
            clause.extend(lits[1..].iter().copied());
            self.cnf.clause(&clause);
            self.cnf.rule("경우 k에 켜짐 ⇔ 실은 신호가 경우 k에 1");
            let values = (0..self.cases)
                .map(|case| {
                    let on = (1..class_count)
                        .filter(|&class| self.classes[class].value(case))
                        .map(|class| lits[class])
                        .collect::<Vec<_>>();
                    self.cnf.or(&on)
                })
                .collect::<Vec<_>>();
            self.class_lits.push(lits);
            self.values.push(values);
        }
        self.cnf.rule("스위치는 자기 입력 신호를 실음");
        for site in self.switches.clone() {
            let class = self.input_class(site.net);
            let lit = self.class_lits[site.cell][class];
            self.cnf.implies(&[site.lit], &[lit]);
        }
    }

    /// Dust connection and pointing directions, matching
    /// `World3D::update_redstone_states` and `cardinal_redstone`.
    fn encode_dust_shape(&mut self) {
        self.cnf
            .rule("가루 연결: 이웃이 가루/토치/스위치/뒤에서 읽는 리피터이거나 계단으로 이어짐");
        let geometry = self.geometry;
        let fals = self.cnf.fals();
        for cell in 0..geometry.len() {
            let mut conn = [fals; 4];
            let dust = self.dust[cell];
            if !self.cnf.is_false(dust) {
                for (index, direction) in CARDINALS.into_iter().enumerate() {
                    let Some(neighbor) = geometry.step(cell, direction) else {
                        continue;
                    };
                    let mut reasons = vec![
                        self.dust[neighbor],
                        self.is_torch[neighbor],
                        self.is_switch[neighbor],
                        self.repeater[neighbor][cardinal_index(direction.inverse())],
                    ];
                    if let (Some(above), Some(neighbor_above)) = (
                        geometry.step(cell, Direction::Top),
                        geometry.step(neighbor, Direction::Top),
                    ) {
                        let step_up = self
                            .cnf
                            .and(&[-self.solid[above], self.dust[neighbor_above]]);
                        reasons.push(step_up);
                    }
                    if let Some(neighbor_below) = geometry.step(neighbor, Direction::Bottom) {
                        let step_down = self
                            .cnf
                            .and(&[-self.solid[neighbor], self.dust[neighbor_below]]);
                        reasons.push(step_down);
                    }
                    let any = self.cnf.or(&reasons);
                    conn[index] = self.cnf.and(&[dust, any]);
                }
            }
            self.conn.push(conn);
        }
        self.cnf
            .rule("가루가 가리키는 방향: 연결된 쪽, 또는 옆 연결이 없으면 직선/십자");
        for cell in 0..geometry.len() {
            let mut points = [fals; 4];
            let dust = self.dust[cell];
            if !self.cnf.is_false(dust) {
                for (index, direction) in CARDINALS.into_iter().enumerate() {
                    let [first, second] = perpendicular(direction);
                    let first = self.conn[cell][cardinal_index(first)];
                    let second = self.conn[cell][cardinal_index(second)];
                    let no_side = self.cnf.and(&[-first, -second]);
                    let reason = self.cnf.or(&[self.conn[cell][index], no_side]);
                    points[index] = self.cnf.and(&[dust, reason]);
                }
            }
            self.points.push(points);
        }
    }

    fn relate(
        &mut self,
        source: usize,
        sink: usize,
        source_kind: SourceKind,
        sink_kind: SinkKind,
        lit: Lit,
    ) {
        if self.cnf.is_false(lit) {
            return;
        }
        self.relations.push(Relation {
            source,
            sink,
            source_kind,
            sink_kind,
            lit,
        });
    }

    /// Torches attached to `support`, as `(torch cell, torch literal)`.
    fn attached_torches(&self, support: usize) -> Vec<(usize, Lit)> {
        let mut torches = Vec::new();
        if let Some(above) = self.geometry.step(support, Direction::Top) {
            torches.push((above, self.torch[above][0]));
        }
        for (index, attach) in TORCH_ATTACH.into_iter().enumerate().skip(1) {
            if let Some(cell) = self.geometry.step(support, attach.inverse()) {
                torches.push((cell, self.torch[cell][index]));
            }
        }
        torches
            .into_iter()
            .filter(|(_, lit)| !self.cnf.is_false(*lit))
            .collect()
    }

    /// Repeaters whose input side is `cell`, as `(repeater cell, literal)`.
    fn reading_repeaters(&self, cell: usize) -> Vec<(usize, Lit)> {
        CARDINALS
            .into_iter()
            .filter_map(|direction| {
                let reader = self.geometry.step(cell, direction)?;
                let lit = self.repeater[reader][cardinal_index(direction.inverse())];
                (!self.cnf.is_false(lit)).then_some((reader, lit))
            })
            .collect()
    }

    fn encode_relations(&mut self) {
        let geometry = self.geometry;
        for cell in 0..geometry.len() {
            let dust = self.dust[cell];
            if !self.cnf.is_false(dust) {
                // Dust to dust: flat neighbors and step connections are symmetric.
                self.cnf.rule("전원 관계: 가루 ↔ 옆 가루");
                for direction in [Direction::East, Direction::North] {
                    if let Some(other) = geometry.step(cell, direction) {
                        let edge = self.cnf.and(&[dust, self.dust[other]]);
                        self.relate(cell, other, SourceKind::Dust, SinkKind::Dust, edge);
                        self.relate(other, cell, SourceKind::Dust, SinkKind::Dust, edge);
                    }
                }
                self.cnf
                    .rule("전원 관계: 가루 ↔ 계단 가루 (아래 가루 위 칸이 블록이 아닐 때)");
                if let Some(above) = geometry.step(cell, Direction::Top) {
                    for direction in CARDINALS {
                        let Some(other) = geometry
                            .step(cell, direction)
                            .and_then(|side| geometry.step(side, Direction::Top))
                        else {
                            continue;
                        };
                        let edge = self.cnf.and(&[dust, self.dust[other], -self.solid[above]]);
                        self.relate(cell, other, SourceKind::Dust, SinkKind::Dust, edge);
                        self.relate(other, cell, SourceKind::Dust, SinkKind::Dust, edge);
                    }
                }
                self.cnf
                    .rule("전원 관계: 가루 → 가리키는 블록(약전원), 가루 → 뒤에서 읽는 리피터");
                // Dust weakly powers its support and the blocks it points into.
                let below = geometry.step(cell, Direction::Bottom).unwrap();
                self.relate(cell, below, SourceKind::Dust, SinkKind::Solid, dust);
                for (index, direction) in CARDINALS.into_iter().enumerate() {
                    let Some(side) = geometry.step(cell, direction) else {
                        continue;
                    };
                    let lit = self.cnf.and(&[self.points[cell][index], self.solid[side]]);
                    self.relate(cell, side, SourceKind::Dust, SinkKind::Solid, lit);
                    let reader = self.repeater[side][cardinal_index(direction.inverse())];
                    let lit = self.cnf.and(&[dust, reader]);
                    self.relate(cell, side, SourceKind::Dust, SinkKind::Repeater, lit);
                }
            }

            self.cnf
                .rule("전원 관계: 토치 → 주변 가루·리피터, 위 블록(강전원)");
            let is_torch = self.is_torch[cell];
            if !self.cnf.is_false(is_torch) {
                for direction in ALL_DIRECTIONS {
                    let Some(target) = geometry.step(cell, direction) else {
                        continue;
                    };
                    if direction == Direction::Top {
                        let lit = self.cnf.and(&[is_torch, self.solid[target]]);
                        self.relate(cell, target, SourceKind::Torch, SinkKind::Solid, lit);
                        continue;
                    }
                    // The torch's own support is solid, so it can never be dust here.
                    let lit = self.cnf.and(&[is_torch, self.dust[target]]);
                    self.relate(cell, target, SourceKind::Torch, SinkKind::Dust, lit);
                    if direction != Direction::Bottom {
                        let reader = self.repeater[target][cardinal_index(direction.inverse())];
                        let lit = self.cnf.and(&[is_torch, reader]);
                        self.relate(cell, target, SourceKind::Torch, SinkKind::Repeater, lit);
                    }
                }
            }

            for (index, direction) in CARDINALS.into_iter().enumerate() {
                let repeater = self.repeater[cell][index];
                if self.cnf.is_false(repeater) {
                    continue;
                }
                self.cnf
                    .rule("전원 관계: 리피터 → 앞쪽 블록/가루/같은 방향 리피터");
                let Some(output) = geometry.step(cell, direction.inverse()) else {
                    continue;
                };
                let lit = self.cnf.and(&[repeater, self.solid[output]]);
                self.relate(cell, output, SourceKind::Repeater, SinkKind::Solid, lit);
                let lit = self.cnf.and(&[repeater, self.dust[output]]);
                self.relate(cell, output, SourceKind::Repeater, SinkKind::Dust, lit);
                let lit = self.cnf.and(&[repeater, self.repeater[output][index]]);
                self.relate(cell, output, SourceKind::Repeater, SinkKind::Repeater, lit);
                self.cnf
                    .rule("리피터가 다른 리피터 옆구리를 치면 잠김 → 금지");
                // A repeater feeding another repeater's side locks it.
                for side in perpendicular(direction) {
                    let locked = self.repeater[output][cardinal_index(side)];
                    self.cnf.implies(&[repeater, locked], &[]);
                }
            }

            // Solid blocks: any power reaches repeaters reading them; strong
            // power also reaches all adjacent dust.
            self.cnf.rule(
                "전원 관계: 블록 → 읽는 리피터; 강전원 블록(아래 토치/리피터/스위치) → 주변 가루",
            );
            let solid = self.solid[cell];
            for (reader, lit) in self.reading_repeaters(cell) {
                let lit = self.cnf.and(&[solid, lit]);
                self.relate(cell, reader, SourceKind::Solid, SinkKind::Repeater, lit);
            }
            let mut strong_sources = Vec::new();
            if let Some(below) = geometry.step(cell, Direction::Bottom) {
                strong_sources.push(self.is_torch[below]);
            }
            for (index, direction) in CARDINALS.into_iter().enumerate() {
                if let Some(source) = geometry.step(cell, direction) {
                    strong_sources.push(self.repeater[source][index]);
                }
            }
            for site in &self.switches {
                if geometry.step(site.cell, site.attach) == Some(cell) {
                    strong_sources.push(site.lit);
                }
            }
            let any_strong = self.cnf.or(&strong_sources);
            let hard = self.cnf.and(&[solid, any_strong]);
            self.hard.push(hard);
            for direction in ALL_DIRECTIONS {
                if let Some(target) = geometry.step(cell, direction) {
                    let lit = self.cnf.and(&[hard, self.dust[target]]);
                    self.relate(cell, target, SourceKind::Solid, SinkKind::Dust, lit);
                }
            }
        }

        for site in self.switches.clone() {
            let support = geometry.step(site.cell, site.attach).unwrap();
            self.cnf
                .rule("전원 관계: 스위치 → 붙은 블록(강전원), 주변 가루·리피터");
            self.relate(
                site.cell,
                support,
                SourceKind::Switch,
                SinkKind::Solid,
                site.lit,
            );
            for direction in ALL_DIRECTIONS {
                let Some(target) = geometry.step(site.cell, direction) else {
                    continue;
                };
                if target == support {
                    continue;
                }
                let lit = self.cnf.and(&[site.lit, self.dust[target]]);
                self.relate(site.cell, target, SourceKind::Switch, SinkKind::Dust, lit);
                if CARDINALS.contains(&direction) {
                    let reader = self.repeater[target][cardinal_index(direction.inverse())];
                    let lit = self.cnf.and(&[site.lit, reader]);
                    self.relate(
                        site.cell,
                        target,
                        SourceKind::Switch,
                        SinkKind::Repeater,
                        lit,
                    );
                }
                self.cnf.rule(
                    "스위치 옆 블록의 약전원은 토치는 무시하고 리피터는 받음 → 그런 배치 금지",
                );
                // The simulator records this soft power but torches ignore it
                // while repeaters do not. Keep both away from such blocks.
                for (_, torch) in self.attached_torches(target) {
                    self.cnf.implies(&[site.lit, torch], &[]);
                }
                for (_, reader) in self.reading_repeaters(target) {
                    let target_solid = self.solid[target];
                    self.cnf.implies(&[site.lit, target_solid, reader], &[]);
                }
            }
        }
    }

    /// If a source is powered, its sink is. Dust and repeaters must also not be
    /// powered without their source, i.e. they carry exactly its function.
    fn encode_soundness(&mut self, relax: bool) {
        self.cnf
            .rule("건전성: 원천이 켜지면 받는 쪽도 켜짐 (가루·리피터는 원천과 똑같은 신호)");
        for relation in self.relations.clone() {
            let guard = if relax {
                let guard = self.cnf.new_var();
                self.relaxations.push(guard);
                vec![relation.lit, -guard]
            } else {
                vec![relation.lit]
            };
            for case in 0..self.cases {
                let source = self.values[relation.source][case];
                let sink = self.values[relation.sink][case];
                let mut premise = guard.clone();
                premise.push(source);
                self.cnf.implies(&premise, &[sink]);
                if relation.sink_kind != SinkKind::Solid {
                    let mut premise = guard.clone();
                    premise.push(sink);
                    self.cnf.implies(&premise, &[source]);
                }
            }
        }
    }

    fn order_levels(&mut self, levels: usize) -> Vec<Vec<Lit>> {
        self.cnf.rule("순위/단계 순서 인코딩: '≥ k+1'이면 '≥ k'");
        (0..self.geometry.len())
            .map(|_| {
                let lits = (0..levels).map(|_| self.cnf.new_var()).collect::<Vec<_>>();
                for window in lits.windows(2) {
                    self.cnf.implies(&[window[1]], &[window[0]]);
                }
                lits
            })
            .collect()
    }

    /// `premise -> order(lower) < order(upper)` in order encoding
    /// (level `k` means "at least `k + 1`").
    fn strictly_below(&mut self, premise: Lit, lower: &[Lit], upper: &[Lit]) {
        self.cnf.implies(&[premise], &[upper[0]]);
        for level in 0..lower.len() {
            if level + 1 < upper.len() {
                self.cnf
                    .implies(&[premise, lower[level]], &[upper[level + 1]]);
            } else {
                self.cnf.implies(&[premise, lower[level]], &[]);
            }
        }
    }

    fn encode_coverage(&mut self, rank_levels: usize, stage_levels: usize, relax: bool) {
        let geometry = self.geometry;
        // Zero levels selects lazy acyclicity: loops are excluded by nogoods
        // added after each model (see `acyclic`), not by order encodings.
        if rank_levels > 0 {
            self.ranks = self.order_levels(rank_levels);
        }
        if stage_levels > 0 {
            self.stages = self.order_levels(stage_levels);
        }
        let mut parents = vec![Vec::new(); geometry.len()];
        let mut witnesses = vec![vec![Vec::new(); self.cases]; geometry.len()];
        for relation in self.relations.clone() {
            self.cnf.rule("정당화: 기여 관계는 실제로 있고, 순위는 오르고 단계는 내려가지 않음 (리피터에서 순위 초기화·단계 상승)");
            let contributes = self.cnf.new_var();
            self.cnf.implies(&[contributes], &[relation.lit]);
            // Repeaters restart the local rank and advance the stage instead,
            // so long repeated wires need few rank levels while loops through
            // repeaters (memory) stay impossible.
            let into_repeater = relation.sink_kind == SinkKind::Repeater;
            if rank_levels > 0 && !into_repeater {
                let lower = self.ranks[relation.source].clone();
                let upper = self.ranks[relation.sink].clone();
                self.strictly_below(contributes, &lower, &upper);
            }
            if stage_levels > 0 {
                let lower = self.stages[relation.source].clone();
                let upper = self.stages[relation.sink].clone();
                if into_repeater {
                    self.strictly_below(contributes, &lower, &upper);
                } else {
                    for level in 0..stage_levels {
                        self.cnf
                            .implies(&[contributes, lower[level]], &[upper[level]]);
                    }
                }
            }
            if relation.sink_kind == SinkKind::Solid {
                self.cnf.rule("정당화: 경우 k의 증인은 그 경우에 켜진 원천");
                for case in 0..self.cases {
                    let witness = self.cnf.new_var();
                    let source = self.values[relation.source][case];
                    self.cnf.implies(&[witness], &[contributes]);
                    self.cnf.implies(&[witness], &[source]);
                    witnesses[relation.sink][case].push(witness);
                }
            } else {
                // Dust and repeaters carry their source's exact function, so one
                // contributing source covers every case.
                let source_unpowered = self.class_lits[relation.source][0];
                self.cnf.implies(&[contributes, source_unpowered], &[]);
                parents[relation.sink].push(contributes);
            }
        }
        for cell in 0..geometry.len() {
            let guard = if relax {
                let guard = self.cnf.new_var();
                self.coverage_relaxations.push((cell, guard));
                Some(guard)
            } else {
                None
            };
            self.cnf
                .rule("켜진 가루·리피터는 기여하는 원천이 있어야 함");
            for element in [self.dust[cell], self.is_repeater[cell]] {
                let mut clause = vec![-element, self.class_lits[cell][0]];
                clause.extend(parents[cell].iter().copied());
                clause.extend(guard);
                self.cnf.clause(&clause);
            }
            self.cnf.rule("켜진 블록은 그 경우에 켜진 원천이 있어야 함");
            for case in 0..self.cases {
                let mut clause = vec![-self.values[cell][case], -self.solid[cell]];
                clause.extend(witnesses[cell][case].iter().copied());
                clause.extend(guard);
                self.cnf.clause(&clause);
            }
        }
    }

    /// A torch inverts its support and sits on a strictly earlier stage.
    fn encode_torches(&mut self, stage_levels: usize) {
        let geometry = self.geometry;
        for cell in 0..geometry.len() {
            for (index, attach) in TORCH_ATTACH.into_iter().enumerate() {
                let torch = self.torch[cell][index];
                if self.cnf.is_false(torch) {
                    continue;
                }
                let support = geometry.step(cell, attach).unwrap();
                self.cnf.rule("토치는 받침 블록의 반대 (경우마다)");
                for case in 0..self.cases {
                    let output = self.values[cell][case];
                    let input = self.values[support][case];
                    self.cnf.implies(&[torch, output], &[-input]);
                    self.cnf.implies(&[torch, -output], &[input]);
                }
                self.cnf.rule("토치는 받침보다 높은 단계 (되먹임 금지)");
                if stage_levels > 0 {
                    let lower = self.stages[support].clone();
                    let upper = self.stages[cell].clone();
                    self.strictly_below(torch, &lower, &upper);
                }
            }
        }
    }

    pub(super) fn kind_lit(&self, cell: usize, kind: CellKind) -> Option<Lit> {
        let lit = match kind {
            CellKind::Air => self.air[cell],
            CellKind::Solid => self.solid[cell],
            CellKind::Dust => self.dust[cell],
            CellKind::Torch(attach) => {
                self.torch[cell][TORCH_ATTACH.iter().position(|&d| d == attach)?]
            }
            CellKind::Repeater(direction) => {
                self.repeater[cell][CARDINALS.iter().position(|&d| d == direction)?]
            }
            CellKind::Switch(attach) => {
                self.switches
                    .iter()
                    .find(|site| site.cell == cell && site.attach == attach)?
                    .lit
            }
        };
        (!self.cnf.is_false(lit)).then_some(lit)
    }

    fn encode_fixed_cells(&mut self, config: &ExactPlacerConfig) -> eyre::Result<()> {
        self.cnf.rule("고정된 칸");
        for (&position, &kind) in &config.fixed_cells {
            ensure!(
                self.geometry.dim.bound_on(position),
                "fixed cell {position:?} is outside the box"
            );
            let cell = self.geometry.index(position);
            let Some(lit) = self.kind_lit(cell, kind) else {
                bail!("fixed cell {position:?} cannot hold {kind:?}");
            };
            self.cnf.clause(&[lit]);
        }
        self.cnf.rule("이전에 거부된 배치는 다시 쓰지 않음");
        for layout in &config.blocked {
            let mut clause = Vec::new();
            for &(position, kind) in layout {
                if !self.geometry.dim.bound_on(position) {
                    continue;
                }
                match self.kind_lit(self.geometry.index(position), kind) {
                    Some(lit) => clause.push(-lit),
                    // The cell cannot hold that kind here, so the layout is
                    // already excluded.
                    None => {
                        clause.clear();
                        clause.push(self.cnf.tru());
                        break;
                    }
                }
            }
            self.cnf.clause(&clause);
        }
        Ok(())
    }

    fn net_function(values: &[Vec<bool>], net: NetId) -> u64 {
        values[net]
            .iter()
            .enumerate()
            .fold(0u64, |mask, (case, &on)| mask | (u64::from(on) << case))
    }

    fn encode_outputs(
        &mut self,
        netlist: &NorNetlist,
        config: &ExactPlacerConfig,
    ) -> eyre::Result<()> {
        let geometry = self.geometry;
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
        for (name, net, sites) in &observations {
            let function = Self::net_function(&values, *net);
            let Some(class) = self.class_of_function(function) else {
                bail!("output `{name}` is constant; the exact placer needs a driven signal");
            };
            self.observed.push((name.clone(), function));
            let cells = match sites {
                Some(positions) => positions
                    .iter()
                    .map(|position| {
                        ensure!(
                            geometry.dim.bound_on(*position),
                            "output `{name}` site {position:?} is outside the box"
                        );
                        Ok(geometry.index(*position))
                    })
                    .collect::<eyre::Result<Vec<_>>>()?,
                None => (0..geometry.len()).collect(),
            };
            self.cnf
                .rule("출력: 허용된 위치의 가루/리피터/토치가 그 신호를 실어야 함");
            let driving = config.driving_outputs.contains(name);
            let mut lits = Vec::new();
            for cell in cells {
                let element = if driving {
                    let mut drivers = vec![self.is_torch[cell]];
                    for (index, direction) in CARDINALS.into_iter().enumerate() {
                        if geometry.step(cell, direction.inverse()).is_none() {
                            drivers.push(self.repeater[cell][index]);
                        }
                    }
                    self.cnf.or(&drivers)
                } else {
                    self.cnf
                        .or(&[self.dust[cell], self.is_repeater[cell], self.is_torch[cell]])
                };
                let lit = self.cnf.and(&[element, self.class_lits[cell][class]]);
                if self.cnf.is_false(lit) {
                    continue;
                }
                lits.push(lit);
                self.output_sites.push(OutputSite {
                    name: name.clone(),
                    cell,
                    lit,
                });
            }
            ensure!(!lits.is_empty(), "output `{name}` has no legal site");
            self.cnf.clause(&lits);
        }
        Ok(())
    }

    /// Sequential-counter bound on the number of non-air cells.
    fn encode_block_limit(&mut self, max_blocks: usize) {
        let lits = self.block_lits.clone();
        if max_blocks >= lits.len() {
            return;
        }
        if max_blocks == 0 {
            for lit in lits {
                self.cnf.clause(&[-lit]);
            }
            return;
        }
        self.cnf.rule("블록 개수 상한 (순차 카운터)");
        // counters[i][j]: at least j + 1 of the first i + 1 literals are true.
        let mut previous: Vec<Lit> = Vec::new();
        for (index, &lit) in lits.iter().enumerate() {
            let width = max_blocks.min(index + 1);
            let current = (0..width).map(|_| self.cnf.new_var()).collect::<Vec<_>>();
            self.cnf.implies(&[lit], &[current[0]]);
            for j in 0..width {
                if j < previous.len() {
                    self.cnf.implies(&[previous[j]], &[current[j]]);
                }
                if j > 0 && j - 1 < previous.len() {
                    self.cnf.implies(&[lit, previous[j - 1]], &[current[j]]);
                }
            }
            if previous.len() == max_blocks {
                self.cnf.implies(&[lit, previous[max_blocks - 1]], &[]);
            }
            previous = current;
        }
    }
}

fn may_observe_output(config: &ExactPlacerConfig, position: Position) -> bool {
    if matches!(
        config.fixed_cells.get(&position),
        Some(CellKind::Repeater(_))
    ) {
        return true;
    }
    match &config.observations {
        Some(observations) => observations
            .iter()
            .any(|(_, sites)| sites.contains(&position)),
        // Outputs without explicit sites may be observed anywhere.
        None => {
            config.output_sites.is_empty()
                || config
                    .output_sites
                    .values()
                    .any(|sites| sites.contains(&position))
        }
    }
}

pub(super) fn default_input_sites(dim: DimSize) -> Vec<(Position, Direction)> {
    let mut sites = Vec::new();
    for z in 0..dim.2 {
        for y in 0..dim.1 {
            for x in 0..dim.0 {
                let boundary = x == 0 || y == 0 || x + 1 == dim.0 || y + 1 == dim.1;
                if !boundary {
                    continue;
                }
                for attach in TORCH_ATTACH {
                    sites.push((Position(x, y, z), attach));
                }
            }
        }
    }
    sites
}
