//! Decoding solver models into worlds and checking them with the simulator.

use std::collections::BTreeMap;

use super::encode::{CellKind, Encoding, CARDINALS, TORCH_ATTACH};
use super::netlist::{NetDriver, NorNetlist};
use super::solver::SatSolver;
use crate::physical_cell::{
    AutoSupport, AxisDirection, CellExpectation, CellGlyph, CellInput, CellOutput, CellPlane,
    PhysicalCellDocument, PlaneAxes,
};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::{Simulator, MANUAL_INPUT_IDLE_CYCLES};
use crate::world::{World, World3D};

#[derive(Debug, Clone)]
pub(super) struct Decoded {
    pub(super) kinds: Vec<CellKind>,
    /// Signal function (bit per input case) of every powered cell.
    pub(super) functions: Vec<Option<u64>>,
    pub(super) class_names: Vec<Option<String>>,
    pub(super) inputs: Vec<(String, Position)>,
    pub(super) outputs: Vec<(String, Position)>,
    /// Function (bit per case) required at each observation name.
    pub(super) observed: Vec<(String, u64)>,
    /// Literals that describe the block choice; their negation blocks this layout.
    pub(super) kind_lits: Vec<i32>,
}

pub(super) fn decode(encoding: &Encoding, solver: &SatSolver, netlist: &NorNetlist) -> Decoded {
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
            options.push((CellKind::Repeater(direction), encoding.repeater[cell][index]));
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
    let classes = (0..geometry.len())
        .map(|cell| {
            encoding.class_lits[cell]
                .iter()
                .position(|&lit| solver.value(lit))
        })
        .collect::<Vec<_>>();
    let functions = classes
        .iter()
        .map(|class| class.map(|class| encoding.classes[class].function))
        .collect();
    let class_names = classes
        .iter()
        .map(|class| class.map(|class| encoding.classes[class].name.clone()))
        .collect();
    let inputs = encoding
        .switches
        .iter()
        .filter(|site| solver.value(site.lit))
        .map(|site| {
            let NetDriver::Input(name) = &netlist.nets[site.net].driver else {
                unreachable!("switches drive input nets")
            };
            (name.clone(), geometry.position(site.cell))
        })
        .collect();
    let mut outputs = BTreeMap::new();
    for site in &encoding.output_sites {
        if solver.value(site.lit) {
            outputs
                .entry(site.name.clone())
                .or_insert(geometry.position(site.cell));
        }
    }
    Decoded {
        kinds,
        functions,
        class_names,
        inputs,
        outputs: outputs.into_iter().collect(),
        observed: encoding.observed.clone(),
        kind_lits,
    }
}

pub(super) fn build_world(dim: DimSize, kinds: &[CellKind]) -> World3D {
    let mut world = World3D::new(dim);
    for (index, kind) in kinds.iter().enumerate() {
        let x = index % dim.0;
        let y = (index / dim.0) % dim.1;
        let z = index / (dim.0 * dim.1);
        let position = Position(x, y, z);
        world[position] = match *kind {
            CellKind::Air => continue,
            CellKind::Solid => PlacedNode::new_cobble(position).block,
            CellKind::Dust => PlacedNode::new_redstone(position).block,
            CellKind::Torch(attach) => Block {
                kind: BlockKind::Torch { is_on: true },
                direction: attach,
            },
            CellKind::Repeater(direction) => PlacedNode::new_repeater(position, direction).block,
            CellKind::Switch(attach) => Block {
                kind: BlockKind::Switch { is_on: false },
                direction: attach,
            },
        };
    }
    world.initialize_redstone_states();
    world
}

#[derive(Debug, Clone)]
pub struct ExactVerificationFailure {
    pub message: String,
    pub position: Option<Position>,
}

fn case_inputs(
    netlist: &NorNetlist,
    inputs: &[(String, Position)],
    case: usize,
) -> Vec<(Position, bool)> {
    let names = netlist.input_names();
    inputs
        .iter()
        .map(|(name, position)| {
            let index = names.iter().position(|n| n == name).unwrap();
            (*position, case & (1 << index) != 0)
        })
        .collect()
}

fn settle_case(
    world: &World,
    netlist: &NorNetlist,
    inputs: &[(String, Position)],
    case: usize,
) -> Result<Simulator, ExactVerificationFailure> {
    let mut simulator = Simulator::from_with_limits_and_trace(world, 256, 50_000, 0).map_err(|error| {
        ExactVerificationFailure {
            message: format!("initial simulation failed: {}", error.message()),
            position: None,
        }
    })?;
    simulator
        .drive_inputs_with_limits(case_inputs(netlist, inputs, case), 256, 50_000)
        .map_err(|error| ExactVerificationFailure {
            message: format!("case {case} did not settle: {error}"),
            position: None,
        })?;
    Ok(simulator)
}

/// Checks every net-labelled element in every input case, then every settled
/// input transition for the public outputs and torch burnout.
pub(super) fn verify(
    netlist: &NorNetlist,
    decoded: &Decoded,
    world: &World3D,
) -> Result<(), ExactVerificationFailure> {
    let case_count = 1usize << netlist.input_names().len();
    let world = World::from(world);
    let dim = world.size;
    let observed = decoded
        .observed
        .iter()
        .cloned()
        .collect::<BTreeMap<_, _>>();

    for case in 0..case_count {
        let simulator = settle_case(&world, netlist, &decoded.inputs, case)?;
        for (cell, function) in decoded.functions.iter().enumerate() {
            let Some(function) = function else {
                continue;
            };
            if !matches!(
                decoded.kinds[cell],
                CellKind::Dust | CellKind::Torch(_) | CellKind::Repeater(_)
            ) {
                continue;
            }
            let position = Position(
                cell % dim.0,
                (cell / dim.0) % dim.1,
                cell / (dim.0 * dim.1),
            );
            let actual = simulator.world()[position].kind.is_powered();
            let expected = function & (1 << case) != 0;
            if actual != expected {
                return Err(ExactVerificationFailure {
                    message: format!(
                        "{:?} carrying {} is {} in case {case}, expected {}",
                        decoded.kinds[cell],
                        decoded.class_names[cell].as_deref().unwrap_or("?"),
                        actual,
                        expected
                    ),
                    position: Some(position),
                });
            }
        }
    }

    for from in 0..case_count {
        for to in 0..case_count {
            let mut simulator = settle_case(&world, netlist, &decoded.inputs, from)?;
            simulator
                .advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)
                .map_err(|error| ExactVerificationFailure {
                    message: format!("idle after case {from} failed: {error}"),
                    position: None,
                })?;
            simulator
                .drive_inputs_with_limits(case_inputs(netlist, &decoded.inputs, to), 256, 50_000)
                .map_err(|error| ExactVerificationFailure {
                    message: format!("transition {from}->{to} did not settle: {error}"),
                    position: None,
                })?;
            for (name, position) in &decoded.outputs {
                let actual = simulator.world()[*position].kind.is_powered();
                if actual != (observed[name] & (1 << to) != 0) {
                    return Err(ExactVerificationFailure {
                        message: format!("output {name} wrong after transition {from}->{to}"),
                        position: Some(*position),
                    });
                }
            }
            for (position, block) in simulator.world().iter_block() {
                if block.kind.is_torch() && simulator.is_torch_burned_out(position) {
                    return Err(ExactVerificationFailure {
                        message: format!("torch burned out after transition {from}->{to}"),
                        position: Some(position),
                    });
                }
            }
        }
    }
    Ok(())
}

fn axis(direction: Direction) -> AxisDirection {
    match direction {
        Direction::East => AxisDirection::XPositive,
        Direction::West => AxisDirection::XNegative,
        Direction::North => AxisDirection::YPositive,
        Direction::South => AxisDirection::YNegative,
        Direction::Top => AxisDirection::ZPositive,
        Direction::Bottom | Direction::None => AxisDirection::ZNegative,
    }
}

/// Boolean expression of a net over the inputs, in RCELL `expect` syntax.
pub(super) fn net_expression(netlist: &NorNetlist, net: usize) -> String {
    match &netlist.nets[net].driver {
        NetDriver::Input(name) => name.clone(),
        NetDriver::Gate => format!(
            "~({})",
            netlist.nets[net]
                .gate_inputs
                .iter()
                .map(|&input| net_expression(netlist, input))
                .collect::<Vec<_>>()
                .join("|")
        ),
    }
}

/// Renders the layout as an editable RCELL document (one `yz` plane per x).
pub(super) fn to_rcell(
    name: &str,
    dim: DimSize,
    netlist: &NorNetlist,
    decoded: &Decoded,
) -> PhysicalCellDocument {
    let mut glyphs = BTreeMap::new();
    let mut glyph_of = BTreeMap::new();
    let mut next = ['T', 'U', 'V', 'W', 'X', 'Y', 'P', 'Q', 'R', 'S', 'A', 'B', 'C', 'D'].into_iter();
    let mut planes = Vec::new();
    for x in 0..dim.0 {
        let mut rows = BTreeMap::new();
        for z in 0..dim.2 {
            let mut row = String::new();
            for y in 0..dim.1 {
                let cell = x + dim.0 * (y + dim.1 * z);
                let glyph = match decoded.kinds[cell] {
                    CellKind::Air | CellKind::Switch(_) => '.',
                    CellKind::Solid => '#',
                    CellKind::Dust => 'r',
                    kind @ (CellKind::Torch(_) | CellKind::Repeater(_)) => {
                        *glyph_of.entry(kind).or_insert_with(|| {
                            let glyph = next.next().expect("enough glyph letters");
                            let spec = match kind {
                                CellKind::Torch(attach) => CellGlyph::Torch {
                                    support: axis(attach),
                                },
                                CellKind::Repeater(direction) => CellGlyph::Repeater {
                                    toward: axis(direction),
                                    delay: 1,
                                },
                                _ => unreachable!(),
                            };
                            glyphs.insert(glyph, spec);
                            glyph
                        })
                    }
                };
                row.push(glyph);
            }
            if row.chars().any(|glyph| glyph != '.') {
                rows.insert(z, row);
            }
        }
        planes.push(CellPlane {
            axes: PlaneAxes::Yz,
            fixed: x,
            rows,
        });
    }
    let inputs = decoded
        .inputs
        .iter()
        .map(|(input, position)| {
            let cell = position.0 + dim.0 * (position.1 + dim.1 * position.2);
            let CellKind::Switch(attach) = decoded.kinds[cell] else {
                unreachable!("input sites hold switches")
            };
            CellInput {
                name: input.clone(),
                position: *position,
                support: axis(attach),
            }
        })
        .collect();
    let outputs = decoded
        .outputs
        .iter()
        .map(|(output, position)| CellOutput {
            name: output.clone(),
            position: *position,
        })
        .collect();
    let expectations = netlist
        .outputs
        .iter()
        .map(|(output, net)| CellExpectation {
            output: output.clone(),
            expression: net_expression(netlist, *net),
        })
        .collect();
    PhysicalCellDocument {
        name: name.to_owned(),
        size: dim,
        auto_support: AutoSupport::default(),
        glyphs,
        inputs,
        probes: Vec::new(),
        outputs,
        planes,
        expectations,
    }
}
