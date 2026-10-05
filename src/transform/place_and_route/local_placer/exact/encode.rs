//! Shared encoding types for the exact placer.
//!
//! The constraint model itself is `exact_placer.rsdsl`, grounded by `dsl.rs`.
//! This module holds what the rest of the placer reads back from a grounded
//! model: the cell kinds, the signal vocabulary (every netlist net function
//! and its complement; class 0 is "unpowered"), the literal tables per cell,
//! the power relations, and the objective bound. Solutions are always checked
//! with the simulator.
//!
//! A hand-written encoder of the same model lived here until 2026-10-04; it
//! served as an independent cross-check while the rsdsl model was new.

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
    /// The grounded rsdsl program behind `cnf`.
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
        Self::build_dsl(netlist, config)
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
