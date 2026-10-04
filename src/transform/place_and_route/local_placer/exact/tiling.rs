//! Chains of `CarryTiling` tiles: a tile solved once, repeated along X.
//!
//! The solver checks one tile with ghost slices in front (see
//! `CarryTiling`). A chain is the real test: the tiles sit side by side,
//! each carry-in torch attached to the previous tile's carry-out block, so
//! the seam rules either held or the chain miscomputes.

use std::collections::BTreeMap;

use eyre::{ensure, eyre};

use super::encode::CellKind;
use super::layout::ExactLayout;
use super::netlist::NorNetlist;
use super::verify::{net_expression_with, rcell_document};
use super::CarryTiling;
use crate::physical_cell::PhysicalCellDocument;
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

/// Slices in front of the first tile that drive its carry in.
const DRIVER: usize = 4;

/// Assembles `bits` copies of `tile` (a layout solved with `carry`) into one
/// RCELL document with `expect` lines for every output.
///
/// Tile `i`'s inputs and outputs get the suffix `i` (`a` becomes `a0`, ...).
/// In front, a switch `cin` drives the first carry in through an inverter
/// and a repeater into the block the first carry-in torch reads, so `cin`
/// has the carry's own polarity. Behind the last tile, a torch attached to
/// its carry-out block is the output `cout`.
pub fn assemble_chain(
    netlist: &NorNetlist,
    carry: &CarryTiling,
    tile: &ExactLayout,
    bits: usize,
    name: &str,
) -> eyre::Result<PhysicalCellDocument> {
    ensure!(bits >= 1, "a chain needs at least one tile");
    let ghost = CarryTiling::GHOST;
    ensure!(
        tile.dim.0 > ghost + 1,
        "the tile is narrower than its ghost slices"
    );
    let width = tile.dim.0 - ghost;
    let &(_, Position(_, y0, z0), _) = tile
        .inputs
        .iter()
        .find(|(input, ..)| *input == carry.input)
        .ok_or_else(|| eyre!("the tile has no `{}` switch", carry.input))?;
    let carry_out = netlist
        .outputs
        .iter()
        .find(|(output, _)| *output == carry.output)
        .map(|&(_, net)| net)
        .ok_or_else(|| eyre!("`{}` is not a netlist output", carry.output))?;

    // The driver's repeater stands on a block below the carry row.
    let dz = usize::from(z0 == 0);
    let row = z0 + dz;
    let dim = DimSize(
        DRIVER + width * bits + 1,
        tile.dim.1,
        (tile.dim.2 + dz).max(row + 2),
    );
    let mut cells = BTreeMap::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    // cin -> block -> torch (~cin) -> repeater -> block: the first tile's
    // carry-in torch reads ~cin and so carries cin.
    cells.insert(Position(0, y0, row), CellKind::Solid);
    cells.insert(
        Position(0, y0, row + 1),
        CellKind::Switch(Direction::Bottom),
    );
    inputs.push((
        "cin".to_owned(),
        Position(0, y0, row + 1),
        Direction::Bottom,
    ));
    cells.insert(Position(1, y0, row), CellKind::Torch(Direction::West));
    cells.insert(Position(2, y0, row), CellKind::Repeater(Direction::West));
    cells.insert(Position(2, y0, row - 1), CellKind::Solid);
    cells.insert(Position(3, y0, row), CellKind::Solid);

    let mut expectations = Vec::new();
    // The value of the block each tile's carry-in torch reads.
    let mut carry_in = "~cin".to_owned();
    for bit in 0..bits {
        let x0 = DRIVER + width * bit;
        let shift =
            |position: Position| Position(x0 + position.0 - ghost, position.1, position.2 + dz);
        for (&position, &kind) in &tile.cells {
            if position.0 >= ghost {
                cells.insert(shift(position), kind);
            }
        }
        for (input, position, attach) in &tile.inputs {
            if *input != carry.input {
                inputs.push((format!("{input}{bit}"), shift(*position), *attach));
            }
        }
        for (output, position) in &tile.outputs {
            if *output != carry.output {
                outputs.push((format!("{output}{bit}"), shift(*position)));
            }
        }
        let rename = |input: &str| {
            if input == carry.input {
                format!("({carry_in})")
            } else {
                format!("{input}{bit}")
            }
        };
        for &(ref output, net) in &netlist.outputs {
            if *output != carry.output {
                expectations.push((
                    format!("{output}{bit}"),
                    net_expression_with(netlist, net, &rename),
                ));
            }
        }
        carry_in = net_expression_with(netlist, carry_out, &rename);
    }
    let end = Position(DRIVER + width * bits, y0, row);
    cells.insert(end, CellKind::Torch(Direction::West));
    outputs.push(("cout".to_owned(), end));
    expectations.push(("cout".to_owned(), format!("~({carry_in})")));
    Ok(rcell_document(
        name,
        dim,
        &cells,
        &inputs,
        &outputs,
        expectations,
    ))
}
