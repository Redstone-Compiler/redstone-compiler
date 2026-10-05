//! rsdsl v2: a constraint modeling language for the exact redstone placer.
//!
//! A model file declares a compile-time world (grids, domains, facts,
//! params), solver-level families (choices, variables, definitions,
//! relations, bounded integers), and labeled rules. Grounding it against an
//! instance yields CNF plus a symbol table that names every variable and
//! records which rule produced each clause.
//!
//! Language reference: docs/solver_dsl_grammar.md in the repository root.

pub mod ast;
pub mod diag;
pub mod formula;
mod fxhash;
mod ground;
pub mod instance;
pub mod lexer;
pub mod parser;
mod program;
pub mod value;

#[cfg(test)]
mod tests;

pub use diag::{Diagnostic, Error, SourceMap};
pub use formula::Lit;
pub use ground::{GroundOptions, Program};
pub use instance::{IValue, Instance};
pub use program::{Comments, Guard, Model, Objective};
