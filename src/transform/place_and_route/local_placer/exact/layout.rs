//! A placed exact layout that windowed construction and compaction rewrite.

use std::collections::BTreeMap;

use super::encode::CellKind;
use super::ExactPlacement;
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

/// Where an observed output may sit after the layout is rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputPolicy {
    Anywhere,
    /// Drives out through the face with the largest Y coordinate: a torch or
    /// an outward repeater there, so a neighbor cell can be attached.
    MaxYFace,
}

/// Where an input switch may sit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputPolicy {
    /// Any cell of the box boundary (every cell of a 2-wide box).
    Anywhere,
    /// On the face with Y = 0.
    MinYFace,
}

#[derive(Debug, Clone)]
pub struct ExactLayout {
    pub dim: DimSize,
    /// Non-air cells, including input switches.
    pub cells: BTreeMap<Position, CellKind>,
    pub inputs: Vec<(String, Position, Direction)>,
    pub outputs: Vec<(String, Position)>,
}

impl ExactLayout {
    pub fn from_placement(dim: DimSize, placement: &ExactPlacement) -> Self {
        let cells = placement.cells.iter().copied().collect::<BTreeMap<_, _>>();
        let inputs = placement
            .placed
            .inputs
            .iter()
            .map(|input| {
                let position = input.position();
                let CellKind::Switch(attach) = cells[&position] else {
                    unreachable!("inputs are switches")
                };
                (input.name.clone(), position, attach)
            })
            .collect();
        let outputs = placement
            .placed
            .outputs
            .iter()
            .map(|output| (output.name.clone(), output.position()))
            .collect();
        Self {
            dim,
            cells,
            inputs,
            outputs,
        }
    }

    /// Reads a built RCELL document. Outputs keep their RCELL names; rename
    /// them to the netlist's output names before compacting if they differ.
    pub fn from_rcell(document: &crate::physical_cell::PhysicalCellDocument) -> eyre::Result<Self> {
        use crate::world::block::BlockKind;
        let build = document.build()?;
        let mut cells = BTreeMap::new();
        for (position, block) in build.world.iter_block() {
            let kind = match block.kind {
                BlockKind::Cobble { .. } => CellKind::Solid,
                BlockKind::Redstone { .. } => CellKind::Dust,
                BlockKind::Torch { .. } => CellKind::Torch(block.direction),
                BlockKind::Repeater { .. } => CellKind::Repeater(block.direction),
                BlockKind::Switch { .. } => CellKind::Switch(block.direction),
                other => eyre::bail!("unsupported block {other:?} at {position:?}"),
            };
            cells.insert(position, kind);
        }
        let inputs = build
            .inputs
            .iter()
            .map(|(name, position)| (name.clone(), *position, build.world[*position].direction))
            .collect();
        let outputs = build
            .outputs
            .iter()
            .map(|(name, position)| (name.clone(), *position))
            .collect();
        Ok(Self {
            dim: document.size,
            cells,
            inputs,
            outputs,
        })
    }

    pub fn block_count(&self) -> usize {
        self.cells
            .values()
            .filter(|kind| !matches!(kind, CellKind::Switch(_)))
            .count()
    }

    /// Removes slice `index` along Y (`axis == 1`) or Z (`axis == 2`) and
    /// shifts everything beyond it back by one. Returns `None` when an input
    /// switch sits in the removed slice.
    pub fn without_slice(&self, axis: usize, index: usize) -> Option<Self> {
        let coordinate = |position: Position| match axis {
            1 => position.1,
            2 => position.2,
            _ => unreachable!("only Y and Z slices are removed"),
        };
        let shift = |position: Position| {
            let mut shifted = position;
            match axis {
                1 => shifted.1 -= usize::from(position.1 > index),
                _ => shifted.2 -= usize::from(position.2 > index),
            }
            shifted
        };
        if self
            .inputs
            .iter()
            .any(|(_, position, _)| coordinate(*position) == index)
        {
            return None;
        }
        let mut dim = self.dim;
        match axis {
            1 => dim.1 -= 1,
            _ => dim.2 -= 1,
        }
        let cells = self
            .cells
            .iter()
            .filter(|(position, _)| coordinate(**position) != index)
            .map(|(position, kind)| (shift(*position), *kind))
            .collect();
        let inputs = self
            .inputs
            .iter()
            .map(|(name, position, attach)| (name.clone(), shift(*position), *attach))
            .collect();
        let outputs = self
            .outputs
            .iter()
            .filter(|(_, position)| coordinate(*position) != index)
            .map(|(name, position)| (name.clone(), shift(*position)))
            .collect();
        Some(Self {
            dim,
            cells,
            inputs,
            outputs,
        })
    }
}
