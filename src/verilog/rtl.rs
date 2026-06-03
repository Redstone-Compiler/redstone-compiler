use std::collections::HashMap;

use eyre::ContextCompat;

use crate::verilog::ast::{
    AlwaysSensitivity as AstAlwaysSensitivity, AlwaysStmt as AstAlwaysStmt,
    Assignment as AstAssignment, Expr as AstExpr, PortDirection, VerilogModule,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RtlSignalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RtlSignalRef {
    pub signal: RtlSignalId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlSignal {
    pub id: RtlSignalId,
    pub name: String,
    pub width: usize,
    pub kind: RtlSignalKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtlSignalKind {
    Input,
    Output,
    RegOutput,
    Wire,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlPort {
    pub name: String,
    pub signal: RtlSignalRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlAssign {
    pub output: RtlSignalRef,
    pub expr: RtlExpr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlProcess {
    pub sensitivity: RtlSensitivity,
    pub statements: Vec<RtlStmt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtlSensitivity {
    Combinational,
    Posedge(RtlSignalRef),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtlStmt {
    Assign {
        kind: RtlAssignKind,
        lhs: RtlSignalRef,
        rhs: RtlExpr,
    },
    If {
        condition: RtlExpr,
        then_branch: Vec<RtlStmt>,
        else_branch: Vec<RtlStmt>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtlAssignKind {
    Blocking,
    NonBlocking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtlExpr {
    Signal(RtlSignalRef),
    Const {
        value: usize,
        width: usize,
    },
    Not(Box<RtlExpr>),
    Add(Box<RtlExpr>, Box<RtlExpr>),
    Mux {
        select: Box<RtlExpr>,
        when_true: Box<RtlExpr>,
        when_false: Box<RtlExpr>,
    },
    And(Box<RtlExpr>, Box<RtlExpr>),
    Or(Box<RtlExpr>, Box<RtlExpr>),
    Xor(Box<RtlExpr>, Box<RtlExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlModule {
    pub name: String,
    pub signals: Vec<RtlSignal>,
    pub ports: Vec<RtlPort>,
    pub continuous_assigns: Vec<RtlAssign>,
    pub processes: Vec<RtlProcess>,
    signal_index: HashMap<String, RtlSignalId>,
}

impl RtlModule {
    pub fn signal_ref(&self, name: &str) -> eyre::Result<RtlSignalRef> {
        self.signal_index
            .get(name)
            .copied()
            .map(|signal| RtlSignalRef { signal })
            .with_context(|| format!("unknown RTL signal `{name}`"))
    }

    pub fn signal_expr(&self, name: &str) -> eyre::Result<RtlExpr> {
        Ok(RtlExpr::Signal(self.signal_ref(name)?))
    }

    pub fn signal_name(&self, signal: RtlSignalRef) -> eyre::Result<&str> {
        self.signals
            .get(signal.signal.0)
            .map(|signal| signal.name.as_str())
            .with_context(|| format!("unknown RTL signal id {}", signal.signal.0))
    }

    pub fn signal_width(&self, signal: RtlSignalRef) -> eyre::Result<usize> {
        self.signals
            .get(signal.signal.0)
            .map(|signal| signal.width)
            .with_context(|| format!("unknown RTL signal id {}", signal.signal.0))
    }
}

pub fn lower_rtl_module(module: &VerilogModule) -> eyre::Result<RtlModule> {
    let mut signals = Vec::new();
    let mut signal_index = HashMap::new();

    for declaration in &module.declarations {
        let kind = rtl_signal_kind(declaration.direction)?;
        let width = declaration
            .range
            .as_ref()
            .map(|range| range.msb.abs_diff(range.lsb) + 1)
            .unwrap_or(1);
        for name in &declaration.names {
            if signal_index.contains_key(name) {
                eyre::bail!("duplicate RTL signal `{name}` in module `{}`", module.name);
            }
            let id = RtlSignalId(signals.len());
            signal_index.insert(name.clone(), id);
            signals.push(RtlSignal {
                id,
                name: name.clone(),
                width,
                kind,
            });
        }
    }

    let ports = module
        .ports
        .iter()
        .map(|name| {
            let signal = *signal_index
                .get(name)
                .with_context(|| format!("module port `{name}` has no declaration"))?;
            Ok(RtlPort {
                name: name.clone(),
                signal: RtlSignalRef { signal },
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    let continuous_assigns = module
        .assignments
        .iter()
        .map(|assignment| lower_continuous_assign(assignment, &signal_index))
        .collect::<eyre::Result<Vec<_>>>()?;

    let processes = module
        .always_blocks
        .iter()
        .map(|always| {
            let sensitivity = match &always.sensitivity {
                AstAlwaysSensitivity::Any => RtlSensitivity::Combinational,
                AstAlwaysSensitivity::Posedge(clock) => {
                    RtlSensitivity::Posedge(signal_ref(clock, &signal_index)?)
                }
            };
            Ok(RtlProcess {
                sensitivity,
                statements: vec![lower_always_stmt(&always.body, &signal_index)?],
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    Ok(RtlModule {
        name: module.name.clone(),
        signals,
        ports,
        continuous_assigns,
        processes,
        signal_index,
    })
}

fn rtl_signal_kind(direction: Option<PortDirection>) -> eyre::Result<RtlSignalKind> {
    match direction {
        Some(PortDirection::Input) => Ok(RtlSignalKind::Input),
        Some(PortDirection::Output) => Ok(RtlSignalKind::Output),
        Some(PortDirection::OutputReg) => Ok(RtlSignalKind::RegOutput),
        Some(PortDirection::Wire) => Ok(RtlSignalKind::Wire),
        None => eyre::bail!("RTL declaration is missing a direction"),
    }
}

fn lower_continuous_assign(
    assignment: &AstAssignment,
    signal_index: &HashMap<String, RtlSignalId>,
) -> eyre::Result<RtlAssign> {
    Ok(RtlAssign {
        output: signal_ref(&assignment.output, signal_index)?,
        expr: lower_expr(&assignment.expr, signal_index)?,
    })
}

fn lower_always_stmt(
    stmt: &AstAlwaysStmt,
    signal_index: &HashMap<String, RtlSignalId>,
) -> eyre::Result<RtlStmt> {
    match stmt {
        AstAlwaysStmt::If {
            condition,
            then_branch,
        } => Ok(RtlStmt::If {
            condition: signal_expr(condition, signal_index)?,
            then_branch: vec![lower_always_stmt(then_branch, signal_index)?],
            else_branch: Vec::new(),
        }),
        AstAlwaysStmt::NonBlockingAssign { output, data } => Ok(RtlStmt::Assign {
            kind: RtlAssignKind::NonBlocking,
            lhs: signal_ref(output, signal_index)?,
            rhs: lower_expr(data, signal_index)?,
        }),
    }
}

fn lower_expr(
    expr: &AstExpr,
    signal_index: &HashMap<String, RtlSignalId>,
) -> eyre::Result<RtlExpr> {
    match expr {
        AstExpr::Ident(name) => signal_expr(name, signal_index),
        AstExpr::Number(value) => Ok(RtlExpr::Const {
            value: *value,
            width: 1,
        }),
        AstExpr::Not(expr) => Ok(RtlExpr::Not(Box::new(lower_expr(expr, signal_index)?))),
        AstExpr::Binary { op, left, right } => {
            let left = Box::new(lower_expr(left, signal_index)?);
            let right = Box::new(lower_expr(right, signal_index)?);
            Ok(match op {
                crate::verilog::ast::BinaryOp::Add => RtlExpr::Add(left, right),
                crate::verilog::ast::BinaryOp::And => RtlExpr::And(left, right),
                crate::verilog::ast::BinaryOp::Xor => RtlExpr::Xor(left, right),
                crate::verilog::ast::BinaryOp::Or => RtlExpr::Or(left, right),
            })
        }
    }
}

fn signal_expr(name: &str, signal_index: &HashMap<String, RtlSignalId>) -> eyre::Result<RtlExpr> {
    Ok(RtlExpr::Signal(signal_ref(name, signal_index)?))
}

fn signal_ref(
    name: &str,
    signal_index: &HashMap<String, RtlSignalId>,
) -> eyre::Result<RtlSignalRef> {
    signal_index
        .get(name)
        .copied()
        .map(|signal| RtlSignalRef { signal })
        .with_context(|| format!("unknown RTL signal `{name}`"))
}
