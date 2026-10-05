//! Static timing: how many redstone ticks after an input changes each cell
//! of a layout can still change.
//!
//! A torch or a repeater (the placer's repeaters have delay 1) takes one
//! redstone tick; dust and blocks pass a change on in the same tick. A cell
//! fed by several sources settles only after the slowest, so its arrival is
//! the longest path to it over every power relation, whether or not some
//! input case drives the cell through it: the worst case, as in static timing
//! analysis of a gate netlist. The relations are the model's `Feeds` (mirrored
//! here from `exact_placer.rsdsl`) plus each torch's support, and the model
//! with `timing` bounds the same quantity with `Stage`, so `analyze` gives
//! exactly the least `output_delay` a layout satisfies.
//!
//! `depth_bounds` is the matching lower bound from the logic alone: the
//! fewest torches between the inputs and each output.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use eyre::bail;

use super::encode::{vocabulary, CellKind, TORCH_ATTACH};
use super::netlist::NorNetlist;
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

const CARDINALS: [Direction; 4] = [
    Direction::East,
    Direction::West,
    Direction::North,
    Direction::South,
];
const DIRECTIONS: [Direction; 6] = [
    Direction::East,
    Direction::West,
    Direction::North,
    Direction::South,
    Direction::Top,
    Direction::Bottom,
];

/// Arrival times of a layout's cells, in redstone ticks.
#[derive(Debug, Clone, Default)]
pub struct Timing {
    /// The latest arrival at every cell that has one.
    pub arrival: BTreeMap<Position, usize>,
    /// The cell each arrival comes through (absent at a path's start).
    previous: BTreeMap<Position, Position>,
    /// Arrival at each output that has one, in the given order.
    pub outputs: Vec<(String, usize)>,
}

impl Timing {
    pub fn output(&self, name: &str) -> Option<usize> {
        self.outputs
            .iter()
            .find(|(output, _)| output == name)
            .map(|&(_, ticks)| ticks)
    }

    /// The slowest output (the first of several as slow).
    pub fn critical(&self) -> Option<(&str, usize)> {
        self.outputs.iter().fold(
            None,
            |slowest: Option<(&str, usize)>, (name, ticks)| match slowest {
                Some((_, best)) if best >= *ticks => slowest,
                _ => Some((name.as_str(), *ticks)),
            },
        )
    }

    /// The path that sets `position`'s arrival, from where it starts.
    pub fn path(&self, position: Position) -> Vec<Position> {
        let mut path = vec![position];
        while let Some(&previous) = self.previous.get(path.last().unwrap()) {
            path.push(previous);
        }
        path.reverse();
        path
    }
}

/// Arrival times with every cell starting at tick 0, as the model's `Stage`
/// does: inputs, and cells nothing powers (an unpowered torch support, say).
pub fn analyze(
    dim: DimSize,
    cells: &BTreeMap<Position, CellKind>,
    outputs: &[(String, Position)],
) -> eyre::Result<Timing> {
    propagate(dim, cells, outputs, cells.keys().copied().collect())
}

/// Arrival times of the changes of one input switch: only cells it reaches
/// get one, so `output` reads the delay from that input to each output.
pub fn analyze_from(
    dim: DimSize,
    cells: &BTreeMap<Position, CellKind>,
    outputs: &[(String, Position)],
    input: Position,
) -> eyre::Result<Timing> {
    propagate(dim, cells, outputs, vec![input])
}

fn propagate(
    dim: DimSize,
    cells: &BTreeMap<Position, CellKind>,
    outputs: &[(String, Position)],
    starts: Vec<Position>,
) -> eyre::Result<Timing> {
    let fanout = fanout(dim, cells);
    let delay = |position: &Position| match cells.get(position) {
        Some(CellKind::Torch(_) | CellKind::Repeater(_)) => 1,
        _ => 0,
    };
    // Without feedback through a torch or repeater, no path takes more
    // ticks than there are torches and repeaters.
    let bound = cells.keys().map(delay).sum::<usize>();
    let mut timing = Timing::default();
    let mut queue = VecDeque::new();
    for start in starts {
        timing.arrival.insert(start, 0);
        queue.push_back(start);
    }
    while let Some(source) = queue.pop_front() {
        let ticks = timing.arrival[&source];
        for &target in fanout.get(&source).into_iter().flatten() {
            let candidate = ticks + delay(&target);
            if timing
                .arrival
                .get(&target)
                .is_some_and(|&current| current >= candidate)
            {
                continue;
            }
            if candidate > bound {
                bail!("a torch or repeater at {target:?} feeds back into its own input");
            }
            timing.arrival.insert(target, candidate);
            timing.previous.insert(target, source);
            queue.push_back(target);
        }
    }
    timing.outputs = outputs
        .iter()
        .filter_map(|(name, position)| {
            timing
                .arrival
                .get(position)
                .map(|&ticks| (name.clone(), ticks))
        })
        .collect();
    Ok(timing)
}

/// Every cell each cell can power: the model's `Feeds` relations, and each
/// torch's support toward the torch.
fn fanout(dim: DimSize, cells: &BTreeMap<Position, CellKind>) -> BTreeMap<Position, Vec<Position>> {
    let step = |position: Option<Position>, direction: Direction| {
        position
            .and_then(|position| position.walk(direction))
            .filter(|position| dim.bound_on(*position))
    };
    // Outside the box is air, as in the model.
    let kind = |position: Option<Position>| {
        position
            .and_then(|position| cells.get(&position).copied())
            .unwrap_or(CellKind::Air)
    };
    let perpendicular = |direction: Direction| match direction {
        Direction::East | Direction::West => [Direction::North, Direction::South],
        _ => [Direction::East, Direction::West],
    };
    let stick = |position: Option<Position>| {
        matches!(
            kind(position),
            CellKind::Dust | CellKind::Torch(_) | CellKind::Switch(_)
        )
    };
    let conn = |c: Position, d: Direction| {
        let c = Some(c);
        kind(c) == CellKind::Dust
            && (stick(step(c, d))
                || kind(step(c, d)) == CellKind::Repeater(d.inverse())
                || (kind(step(c, Direction::Top)) != CellKind::Solid
                    && kind(step(step(c, d), Direction::Top)) == CellKind::Dust)
                || (kind(step(c, d)) != CellKind::Solid
                    && kind(step(step(c, d), Direction::Bottom)) == CellKind::Dust))
    };
    let points = |c: Position, d: Direction| {
        kind(Some(c)) == CellKind::Dust
            && (conn(c, d) || perpendicular(d).iter().all(|&p| !conn(c, p)))
    };
    let hard = |c: Position| {
        let c = Some(c);
        kind(c) == CellKind::Solid
            && (matches!(kind(step(c, Direction::Bottom)), CellKind::Torch(_))
                || CARDINALS
                    .iter()
                    .any(|&d| kind(step(c, d)) == CellKind::Repeater(d))
                || TORCH_ATTACH
                    .iter()
                    .any(|&a| kind(step(c, a.inverse())) == CellKind::Switch(a)))
    };

    let mut fanout = BTreeMap::<Position, Vec<Position>>::new();
    let mut feed = |source: Position, target: Option<Position>| {
        if let Some(target) = target {
            let targets = fanout.entry(source).or_default();
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
    };
    for (&c, &k) in cells {
        let here = Some(c);
        match k {
            CellKind::Air => {}
            CellKind::Dust => {
                for d in CARDINALS {
                    let next = step(here, d);
                    if kind(next) == CellKind::Dust {
                        feed(c, next);
                    }
                    let up = step(next, Direction::Top);
                    if kind(up) == CellKind::Dust
                        && kind(step(here, Direction::Top)) != CellKind::Solid
                    {
                        feed(c, up);
                        feed(up.unwrap(), here);
                    }
                    if points(c, d) && kind(next) == CellKind::Solid {
                        feed(c, next);
                    }
                    if kind(next) == CellKind::Repeater(d.inverse()) {
                        feed(c, next);
                    }
                }
                feed(c, step(here, Direction::Bottom));
            }
            CellKind::Torch(attach) => {
                if let Some(support) = step(here, attach) {
                    feed(support, here);
                }
                for d in DIRECTIONS {
                    let next = step(here, d);
                    if d == Direction::Top {
                        if kind(next) == CellKind::Solid {
                            feed(c, next);
                        }
                    } else if kind(next) == CellKind::Dust {
                        feed(c, next);
                    }
                    if CARDINALS.contains(&d) && kind(next) == CellKind::Repeater(d.inverse()) {
                        feed(c, next);
                    }
                }
            }
            CellKind::Repeater(d) => {
                let out = step(here, d.inverse());
                if matches!(kind(out), CellKind::Solid | CellKind::Dust)
                    || kind(out) == CellKind::Repeater(d)
                {
                    feed(c, out);
                }
            }
            CellKind::Solid => {
                for d in CARDINALS {
                    let next = step(here, d);
                    if kind(next) == CellKind::Repeater(d.inverse()) {
                        feed(c, next);
                    }
                }
                if hard(c) {
                    for d in DIRECTIONS {
                        let next = step(here, d);
                        if kind(next) == CellKind::Dust {
                            feed(c, next);
                        }
                    }
                }
            }
            CellKind::Switch(attach) => {
                feed(c, step(here, attach));
                for d in DIRECTIONS {
                    if d == attach {
                        continue;
                    }
                    let next = step(here, d);
                    if kind(next) == CellKind::Dust {
                        feed(c, next);
                    }
                    if CARDINALS.contains(&d) && kind(next) == CellKind::Repeater(d.inverse()) {
                        feed(c, next);
                    }
                }
            }
        }
    }
    fanout
}

/// The fewest torches between the inputs and each netlist output, given the
/// signal vocabulary the placer uses: dust and blocks OR classes for free
/// (into another class of the vocabulary), and a torch complements one. No
/// layout of the netlist has an output earlier than this.
pub fn depth_bounds(netlist: &NorNetlist) -> BTreeMap<String, usize> {
    let classes = vocabulary(netlist);
    let mut depth = vec![None::<usize>; classes.len()];
    let mut reached = BTreeSet::new();
    for (index, class) in classes.iter().enumerate() {
        if class.input.is_some() || index == 0 {
            reached.insert(index);
            depth[index] = Some(0);
        }
    }
    let close = |reached: &mut BTreeSet<usize>, depth: &mut [Option<usize>], level: usize| {
        for (index, class) in classes.iter().enumerate() {
            if reached.contains(&index) {
                continue;
            }
            let union = reached
                .iter()
                .map(|&other| classes[other].function)
                .filter(|&function| function & !class.function == 0)
                .fold(0, |union, function| union | function);
            if union == class.function {
                reached.insert(index);
                depth[index] = Some(level);
            }
        }
    };
    close(&mut reached, &mut depth, 0);
    let mut level = 0;
    loop {
        level += 1;
        let complements = reached
            .iter()
            .filter_map(|&index| classes[index].complement)
            .filter(|index| !reached.contains(index))
            .collect::<Vec<_>>();
        if complements.is_empty() {
            break;
        }
        for index in complements {
            reached.insert(index);
            depth[index] = Some(level);
        }
        close(&mut reached, &mut depth, level);
    }
    let values = netlist.net_values();
    let function = |net: usize| {
        values[net]
            .iter()
            .enumerate()
            .fold(0u64, |mask, (case, &on)| mask | (u64::from(on) << case))
    };
    netlist
        .outputs
        .iter()
        .filter_map(|(name, net)| {
            let function = function(*net);
            classes
                .iter()
                .position(|class| class.function == function)
                .and_then(|index| depth[index])
                .map(|ticks| (name.clone(), ticks))
        })
        .collect()
}
