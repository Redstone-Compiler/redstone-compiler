//! Small, text-editable physical redstone cells.
//!
//! This module is intentionally downstream of the compiler's world model:
//! it parses an exact local-cell description, expands only derived physical
//! details, and then delegates connectivity and behavior to `World3D` and
//! `Simulator`.

mod build;
mod emit;
mod parser;
mod syntax;
mod verify;

pub use build::PhysicalCellBuild;
pub use syntax::{
    AutoSupport, AxisDirection, CellExpectation, CellGlyph, CellInput, CellOutput, CellPlane,
    PhysicalCellDocument, PlaneAxes,
};
pub use verify::{PhysicalCellCaseFailure, PhysicalCellCaseSimulation, PhysicalCellVerification};
