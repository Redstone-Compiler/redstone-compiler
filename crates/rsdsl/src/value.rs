//! Compile-time values and types.

use std::fmt;
use std::sync::Arc;

/// A compile-time value. Identifiers refer to declarations in the world.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Str(Arc<str>),
    /// Element `index` of symbol domain `domain`.
    Sym(u32, u32),
    /// Variant `variant` of enum `enum_id` (subsets use their base enum).
    Variant(u32, u32),
    /// Cell with linear index `index` of grid `grid`.
    Cell(u32, u32),
    /// Member `member` of choice `choice` with its payload.
    Member(u32, u32, Arc<[Value]>),
    Tuple(Arc<[Value]>),
    Set(Arc<[Value]>),
    None,
}

impl Value {
    pub fn is_none(&self) -> bool {
        matches!(self, Value::None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Bool,
    Int,
    Str,
    Domain(u32),
    Enum(u32),
    Subset(u32),
    Grid(u32),
    Choice(u32),
    Tuple(Vec<Type>),
    Set(Box<Type>),
    Option(Box<Type>),
}

impl Type {
    /// The type of the values inside an option, or the type itself.
    pub fn unwrap_option(&self) -> &Type {
        match self {
            Type::Option(inner) => inner,
            other => other,
        }
    }
}

/// Names used when rendering values and types.
pub trait Names {
    fn enum_name(&self, id: u32) -> &str;
    fn variant_name(&self, enum_id: u32, variant: u32) -> &str;
    fn subset_name(&self, id: u32) -> &str;
    fn domain_name(&self, id: u32) -> &str;
    fn symbol_name(&self, domain: u32, index: u32) -> &str;
    fn grid_name(&self, id: u32) -> &str;
    fn cell_coords(&self, grid: u32, index: u32) -> [i64; 3];
    fn choice_name(&self, id: u32) -> &str;
    fn member_name(&self, choice: u32, member: u32) -> &str;
}

pub struct Show<'a, N: Names + ?Sized>(pub &'a Value, pub &'a N);

impl<N: Names + ?Sized> fmt::Display for Show<'_, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = self.1;
        match self.0 {
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Sym(d, i) => f.write_str(names.symbol_name(*d, *i)),
            Value::Variant(e, v) => f.write_str(names.variant_name(*e, *v)),
            Value::Cell(g, i) => {
                let [x, y, z] = names.cell_coords(*g, *i);
                write!(f, "({x},{y},{z})")
            }
            Value::Member(c, m, payload) => {
                f.write_str(names.member_name(*c, *m))?;
                if !payload.is_empty() {
                    f.write_str("(")?;
                    for (i, value) in payload.iter().enumerate() {
                        if i > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{}", Show(value, names))?;
                    }
                    f.write_str(")")?;
                }
                Ok(())
            }
            Value::Tuple(items) | Value::Set(items) => {
                let (open, close) = if matches!(self.0, Value::Tuple(_)) {
                    ("(", ")")
                } else {
                    ("[", "]")
                };
                f.write_str(open)?;
                for (i, value) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", Show(value, names))?;
                }
                f.write_str(close)
            }
            Value::None => f.write_str("none"),
        }
    }
}

pub struct ShowType<'a, N: Names + ?Sized>(pub &'a Type, pub &'a N);

impl<N: Names + ?Sized> fmt::Display for ShowType<'_, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = self.1;
        match self.0 {
            Type::Bool => f.write_str("bool"),
            Type::Int => f.write_str("int"),
            Type::Str => f.write_str("string"),
            Type::Domain(d) => f.write_str(names.domain_name(*d)),
            Type::Enum(e) => f.write_str(names.enum_name(*e)),
            Type::Subset(s) => f.write_str(names.subset_name(*s)),
            Type::Grid(g) => f.write_str(names.grid_name(*g)),
            Type::Choice(c) => f.write_str(names.choice_name(*c)),
            Type::Tuple(items) => {
                f.write_str("(")?;
                for (i, t) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", ShowType(t, names))?;
                }
                f.write_str(")")
            }
            Type::Set(inner) => write!(f, "set<{}>", ShowType(inner, names)),
            Type::Option(inner) => write!(f, "{}?", ShowType(inner, names)),
        }
    }
}
