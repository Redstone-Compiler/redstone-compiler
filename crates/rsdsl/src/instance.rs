//! Instances built from Rust: the compile-time world a model is grounded in.

use std::collections::HashMap;

/// A compile-time value written without knowing the model's declarations.
/// Names are resolved against the expected type during grounding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IValue {
    None,
    Bool(bool),
    Int(i64),
    /// A domain symbol or an enum variant.
    Sym(String),
    Cell(i64, i64, i64),
    /// A choice member with its payload.
    Member(String, Vec<IValue>),
    Tuple(Vec<IValue>),
}

impl IValue {
    pub fn sym(name: impl Into<String>) -> Self {
        IValue::Sym(name.into())
    }

    pub fn cell(x: usize, y: usize, z: usize) -> Self {
        IValue::Cell(x as i64, y as i64, z as i64)
    }

    pub fn member(name: impl Into<String>, payload: Vec<IValue>) -> Self {
        IValue::Member(name.into(), payload)
    }

    pub fn unit(name: impl Into<String>) -> Self {
        IValue::Member(name.into(), Vec::new())
    }
}

impl From<i64> for IValue {
    fn from(value: i64) -> Self {
        IValue::Int(value)
    }
}

impl From<usize> for IValue {
    fn from(value: usize) -> Self {
        IValue::Int(value as i64)
    }
}

impl From<bool> for IValue {
    fn from(value: bool) -> Self {
        IValue::Bool(value)
    }
}

impl From<&str> for IValue {
    fn from(value: &str) -> Self {
        IValue::Sym(value.to_owned())
    }
}

/// Grid sizes, domain values, params, and extern facts.
#[derive(Debug, Clone, Default)]
pub struct Instance {
    pub name: String,
    pub grids: HashMap<String, [i64; 3]>,
    pub domains: HashMap<String, Vec<IValue>>,
    pub params: HashMap<String, IValue>,
    pub facts: HashMap<String, Vec<Vec<IValue>>>,
}

impl Instance {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    pub fn grid(&mut self, name: &str, dims: (usize, usize, usize)) -> &mut Self {
        self.grids.insert(
            name.to_owned(),
            [dims.0 as i64, dims.1 as i64, dims.2 as i64],
        );
        self
    }

    pub fn domain(&mut self, name: &str, values: impl IntoIterator<Item = IValue>) -> &mut Self {
        self.domains
            .insert(name.to_owned(), values.into_iter().collect());
        self
    }

    pub fn param(&mut self, name: &str, value: impl Into<IValue>) -> &mut Self {
        self.params.insert(name.to_owned(), value.into());
        self
    }

    /// Declares `name` with no rows yet, so an empty fact is still given.
    pub fn fact(&mut self, name: &str) -> &mut Self {
        self.facts.entry(name.to_owned()).or_default();
        self
    }

    pub fn row(&mut self, name: &str, row: Vec<IValue>) -> &mut Self {
        self.facts.entry(name.to_owned()).or_default().push(row);
        self
    }
}
