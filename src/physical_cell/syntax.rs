use std::collections::BTreeMap;

use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AxisDirection {
    XNegative,
    XPositive,
    YNegative,
    YPositive,
    ZNegative,
    ZPositive,
}

impl AxisDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::XNegative => "x-",
            Self::XPositive => "x+",
            Self::YNegative => "y-",
            Self::YPositive => "y+",
            Self::ZNegative => "z-",
            Self::ZPositive => "z+",
        }
    }

    pub(crate) fn direction(self) -> Direction {
        match self {
            Self::XNegative => Direction::West,
            Self::XPositive => Direction::East,
            Self::YNegative => Direction::South,
            Self::YPositive => Direction::North,
            Self::ZNegative => Direction::Bottom,
            Self::ZPositive => Direction::Top,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutoSupport {
    pub dust: bool,
    pub repeater: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CellGlyph {
    Torch { support: AxisDirection },
    Repeater { toward: AxisDirection, delay: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneAxes {
    Xy,
    Xz,
    Yz,
}

impl PlaneAxes {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Xy => "xy",
            Self::Xz => "xz",
            Self::Yz => "yz",
        }
    }

    pub fn fixed_axis(self) -> char {
        match self {
            Self::Xy => 'z',
            Self::Xz => 'y',
            Self::Yz => 'x',
        }
    }

    pub fn row_axis(self) -> char {
        match self {
            Self::Xy => 'y',
            Self::Xz | Self::Yz => 'z',
        }
    }

    pub(crate) fn column_len(self, size: DimSize) -> usize {
        match self {
            Self::Xy | Self::Xz => size.0,
            Self::Yz => size.1,
        }
    }

    pub(crate) fn row_len(self, size: DimSize) -> usize {
        match self {
            Self::Xy => size.1,
            Self::Xz | Self::Yz => size.2,
        }
    }

    pub(crate) fn fixed_len(self, size: DimSize) -> usize {
        match self {
            Self::Xy => size.2,
            Self::Xz => size.1,
            Self::Yz => size.0,
        }
    }

    pub(crate) fn position(self, fixed: usize, row: usize, column: usize) -> Position {
        match self {
            Self::Xy => Position(column, row, fixed),
            Self::Xz => Position(column, fixed, row),
            Self::Yz => Position(fixed, column, row),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellPlane {
    pub axes: PlaneAxes,
    pub fixed: usize,
    pub rows: BTreeMap<usize, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellInput {
    pub name: String,
    pub position: Position,
    pub support: AxisDirection,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellOutput {
    pub name: String,
    pub position: Position,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellExpectation {
    pub output: String,
    pub expression: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalCellDocument {
    pub name: String,
    pub size: DimSize,
    pub auto_support: AutoSupport,
    pub glyphs: BTreeMap<char, CellGlyph>,
    pub inputs: Vec<CellInput>,
    pub outputs: Vec<CellOutput>,
    pub planes: Vec<CellPlane>,
    pub expectations: Vec<CellExpectation>,
}

impl PhysicalCellDocument {
    pub fn build(&self) -> eyre::Result<super::PhysicalCellBuild> {
        super::PhysicalCellBuild::from_document(self)
    }
}
