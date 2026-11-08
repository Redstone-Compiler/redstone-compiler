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
        value: u128,
        width: usize,
    },
    Select {
        value: Box<RtlExpr>,
        msb: usize,
        lsb: usize,
    },
    Concat(Vec<RtlExpr>),
    Resize {
        value: Box<RtlExpr>,
        width: usize,
    },
    Equal(Box<RtlExpr>, Box<RtlExpr>),
    GreaterEqual(Box<RtlExpr>, Box<RtlExpr>),
    Sub(Box<RtlExpr>, Box<RtlExpr>),
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
    /// Elaborate an unsigned input-port expression in the declared child-port width.
    /// Output bindings must be checked separately and restricted to named signals.
    pub fn elaborate_expr(&self, expr: &AstExpr, expected_width: usize) -> eyre::Result<RtlExpr> {
        eyre::ensure!(
            (1..=128).contains(&expected_width),
            "port width must be in 1..=128"
        );
        let value = elaborate(
            expr,
            Some(expected_width),
            &self.signal_index,
            &self.signals,
        )?;
        resize(value, expected_width, &self.signals)
    }

    pub fn expr_width(&self, expr: &RtlExpr) -> eyre::Result<usize> {
        rtl_expr_width(expr, &self.signals)
    }

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
        let width = if let Some(range) = &declaration.range {
            eyre::ensure!(
                range.lsb == 0 && range.msb < 128,
                "only unsigned packed [N-1:0] vectors of width 1..=128 are supported"
            );
            range.msb + 1
        } else {
            1
        };
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
        .map(|assignment| lower_continuous_assign(assignment, &signal_index, &signals))
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
                statements: vec![lower_always_stmt(&always.body, &signal_index, &signals)?],
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
    signals: &[RtlSignal],
) -> eyre::Result<RtlAssign> {
    Ok(RtlAssign {
        output: signal_ref(&assignment.output, signal_index)?,
        expr: lower_assignment_expr(
            &assignment.expr,
            signal_ref(&assignment.output, signal_index)?,
            signal_index,
            signals,
        )?,
    })
}

fn lower_always_stmt(
    stmt: &AstAlwaysStmt,
    signal_index: &HashMap<String, RtlSignalId>,
    signals: &[RtlSignal],
) -> eyre::Result<RtlStmt> {
    match stmt {
        AstAlwaysStmt::If {
            condition,
            then_branch,
        } => Ok(RtlStmt::If {
            condition: signal_expr(condition, signal_index)?,
            then_branch: vec![lower_always_stmt(then_branch, signal_index, signals)?],
            else_branch: Vec::new(),
        }),
        AstAlwaysStmt::NonBlockingAssign { output, data } => Ok(RtlStmt::Assign {
            kind: RtlAssignKind::NonBlocking,
            lhs: signal_ref(output, signal_index)?,
            rhs: lower_assignment_expr(
                data,
                signal_ref(output, signal_index)?,
                signal_index,
                signals,
            )?,
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
            value: *value as u128,
            width: constant_width(*value),
        }),
        AstExpr::Not(expr) => Ok(RtlExpr::Not(Box::new(lower_expr(expr, signal_index)?))),
        AstExpr::Binary { op, left, right } => {
            let left = Box::new(lower_expr(left, signal_index)?);
            let right = Box::new(lower_expr(right, signal_index)?);
            Ok(match op {
                crate::verilog::ast::BinaryOp::Equal => RtlExpr::Equal(left, right),
                crate::verilog::ast::BinaryOp::GreaterEqual => RtlExpr::GreaterEqual(left, right),
                crate::verilog::ast::BinaryOp::Sub => RtlExpr::Sub(left, right),
                crate::verilog::ast::BinaryOp::Add => RtlExpr::Add(left, right),
                crate::verilog::ast::BinaryOp::And => RtlExpr::And(left, right),
                crate::verilog::ast::BinaryOp::Xor => RtlExpr::Xor(left, right),
                crate::verilog::ast::BinaryOp::Or => RtlExpr::Or(left, right),
            })
        }
        _ => eyre::bail!("extended expression requires typed elaboration"),
    }
}

// Legacy scalar/counter syntax retains its historical unsized-number behavior.
// New packed-expression syntax requires explicitly sized constants.
fn uses_typed_syntax(expr: &AstExpr) -> bool {
    match expr {
        AstExpr::Ident(_) | AstExpr::Number(_) => false,
        AstExpr::Not(value) => uses_typed_syntax(value),
        AstExpr::Binary { op, left, right } => {
            matches!(
                op,
                crate::verilog::ast::BinaryOp::Equal
                    | crate::verilog::ast::BinaryOp::GreaterEqual
                    | crate::verilog::ast::BinaryOp::Sub
            ) || uses_typed_syntax(left)
                || uses_typed_syntax(right)
        }
        _ => true,
    }
}

fn has_unsized_literal(expr: &AstExpr) -> bool {
    match expr {
        AstExpr::Number(_) => true,
        AstExpr::Ident(_) | AstExpr::SizedNumber { .. } => false,
        AstExpr::Not(value) | AstExpr::Select { value, .. } => has_unsized_literal(value),
        AstExpr::Binary { left, right, .. } => {
            has_unsized_literal(left) || has_unsized_literal(right)
        }
        AstExpr::Concat(parts) => parts.iter().any(has_unsized_literal),
        AstExpr::Mux {
            select,
            when_true,
            when_false,
        } => {
            has_unsized_literal(select)
                || has_unsized_literal(when_true)
                || has_unsized_literal(when_false)
        }
    }
}

fn lower_assignment_expr(
    expr: &AstExpr,
    output: RtlSignalRef,
    index: &HashMap<String, RtlSignalId>,
    signals: &[RtlSignal],
) -> eyre::Result<RtlExpr> {
    if has_unsized_literal(expr) && !uses_typed_syntax(expr) {
        return lower_expr(expr, index);
    }
    let target = signals[output.signal.0].width;
    let value = elaborate(expr, Some(target), index, signals)?;
    resize(value, target, signals)
}

fn rtl_expr_width(expr: &RtlExpr, signals: &[RtlSignal]) -> eyre::Result<usize> {
    Ok(match expr {
        RtlExpr::Signal(signal) => {
            signals
                .get(signal.signal.0)
                .ok_or_else(|| eyre::eyre!("unknown RTL signal"))?
                .width
        }
        RtlExpr::Const { width, .. } | RtlExpr::Resize { width, .. } => *width,
        RtlExpr::Select { msb, lsb, .. } => msb - lsb + 1,
        RtlExpr::Concat(parts) => parts
            .iter()
            .map(|p| rtl_expr_width(p, signals))
            .collect::<eyre::Result<Vec<_>>>()?
            .iter()
            .sum(),
        RtlExpr::Equal(..) | RtlExpr::GreaterEqual(..) => 1,
        RtlExpr::Not(value) => rtl_expr_width(value, signals)?,
        RtlExpr::Sub(a, b)
        | RtlExpr::Add(a, b)
        | RtlExpr::And(a, b)
        | RtlExpr::Or(a, b)
        | RtlExpr::Xor(a, b) => rtl_expr_width(a, signals)?.max(rtl_expr_width(b, signals)?),
        RtlExpr::Mux {
            when_true,
            when_false,
            ..
        } => rtl_expr_width(when_true, signals)?.max(rtl_expr_width(when_false, signals)?),
    })
}

fn resize(value: RtlExpr, width: usize, signals: &[RtlSignal]) -> eyre::Result<RtlExpr> {
    if rtl_expr_width(&value, signals)? == width {
        Ok(value)
    } else {
        Ok(RtlExpr::Resize {
            value: Box::new(value),
            width,
        })
    }
}

fn natural_width(
    expr: &AstExpr,
    index: &HashMap<String, RtlSignalId>,
    signals: &[RtlSignal],
) -> eyre::Result<usize> {
    let width = match expr {
        AstExpr::Ident(name) => signals[signal_ref(name, index)?.signal.0].width,
        AstExpr::Number(_) => eyre::bail!("unsized constants are unsupported in packed expressions; use an explicitly sized literal"),
        AstExpr::SizedNumber { width, .. } => *width,
        AstExpr::Select { value, msb, lsb } => {
            let width = natural_width(value, index, signals)?;
            eyre::ensure!(lsb <= msb && *msb < width, "static select [{msb}:{lsb}] is outside width {width} or ascending");
            msb - lsb + 1
        }
        AstExpr::Concat(parts) => {
            eyre::ensure!(!parts.is_empty(), "empty concatenation");
            parts.iter().map(|p| natural_width(p, index, signals)).collect::<eyre::Result<Vec<_>>>()?.iter().sum()
        }
        AstExpr::Not(value) => natural_width(value, index, signals)?,
        AstExpr::Binary { op, left, right } => {
            let width = natural_width(left, index, signals)?.max(natural_width(right, index, signals)?);
            if matches!(op, crate::verilog::ast::BinaryOp::Equal | crate::verilog::ast::BinaryOp::GreaterEqual) { 1 } else { width }
        }
        AstExpr::Mux { select, when_true, when_false } => {
            natural_width(select, index, signals)?;
            natural_width(when_true, index, signals)?.max(natural_width(when_false, index, signals)?)
        }
    };
    eyre::ensure!(
        (1..=128).contains(&width),
        "packed expression width must be in 1..=128, got {width}"
    );
    Ok(width)
}

fn elaborate(
    expr: &AstExpr,
    context: Option<usize>,
    index: &HashMap<String, RtlSignalId>,
    signals: &[RtlSignal],
) -> eyre::Result<RtlExpr> {
    use crate::verilog::ast::BinaryOp;
    let natural = natural_width(expr, index, signals)?;
    let width = natural.max(context.unwrap_or(0));
    let value = match expr {
        AstExpr::Ident(name) => signal_expr(name, index)?,
        AstExpr::SizedNumber { value, width } => RtlExpr::Const {
            value: *value,
            width: *width,
        },
        AstExpr::Number(_) => unreachable!("natural_width rejects unsized literals"),
        AstExpr::Select { value, msb, lsb } => RtlExpr::Select {
            value: Box::new(elaborate(value, None, index, signals)?),
            msb: *msb,
            lsb: *lsb,
        },
        AstExpr::Concat(parts) => RtlExpr::Concat(
            parts
                .iter()
                .map(|p| elaborate(p, None, index, signals))
                .collect::<eyre::Result<_>>()?,
        ),
        AstExpr::Not(value) => RtlExpr::Not(Box::new(resize(
            elaborate(value, Some(width), index, signals)?,
            width,
            signals,
        )?)),
        AstExpr::Binary { op, left, right } => {
            let operand_width = if matches!(op, BinaryOp::Equal | BinaryOp::GreaterEqual) {
                natural_width(left, index, signals)?.max(natural_width(right, index, signals)?)
            } else {
                width
            };
            let left = Box::new(resize(
                elaborate(left, Some(operand_width), index, signals)?,
                operand_width,
                signals,
            )?);
            let right = Box::new(resize(
                elaborate(right, Some(operand_width), index, signals)?,
                operand_width,
                signals,
            )?);
            match op {
                BinaryOp::Add => RtlExpr::Add(left, right),
                BinaryOp::And => RtlExpr::And(left, right),
                BinaryOp::Or => RtlExpr::Or(left, right),
                BinaryOp::Xor => RtlExpr::Xor(left, right),
                BinaryOp::Equal => RtlExpr::Equal(left, right),
                BinaryOp::GreaterEqual => RtlExpr::GreaterEqual(left, right),
                BinaryOp::Sub => RtlExpr::Sub(left, right),
            }
        }
        AstExpr::Mux {
            select,
            when_true,
            when_false,
        } => {
            let mut select = elaborate(select, None, index, signals)?;
            let select_width = rtl_expr_width(&select, signals)?;
            if select_width != 1 {
                select = RtlExpr::Not(Box::new(RtlExpr::Equal(
                    Box::new(select),
                    Box::new(RtlExpr::Const {
                        value: 0,
                        width: select_width,
                    }),
                )));
            }
            RtlExpr::Mux {
                select: Box::new(select),
                when_true: Box::new(resize(
                    elaborate(when_true, Some(width), index, signals)?,
                    width,
                    signals,
                )?),
                when_false: Box::new(resize(
                    elaborate(when_false, Some(width), index, signals)?,
                    width,
                    signals,
                )?),
            }
        }
    };
    Ok(value)
}

fn constant_width(value: usize) -> usize {
    (usize::BITS - value.leading_zeros()).max(1) as usize
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verilog::parser::parse_module;

    #[test]
    fn input_port_expression_uses_child_width_context() -> eyre::Result<()> {
        let ast = parse_module("module m(a); input [7:0] a; child u(.x(a + 8'd1)); endmodule")?;
        let rtl = lower_rtl_module(&ast)?;
        let expr = rtl.elaborate_expr(&ast.instances[0].connections[0].1, 9)?;
        let RtlExpr::Add(left, right) = &expr else {
            panic!("expected add")
        };
        assert_eq!(rtl.expr_width(left)?, 9);
        assert_eq!(rtl.expr_width(right)?, 9);
        assert_eq!(rtl.expr_width(&expr)?, 9);
        Ok(())
    }

    #[test]
    fn packed_add_extends_before_arithmetic_and_truncates_after() -> eyre::Result<()> {
        let rtl = lower("module m(a,b,y,z); input [7:0] a,b; output [8:0] y; output [3:0] z; assign y = a + {1'b0,b}; assign z = a + 8'd1; endmodule")?;
        assert_eq!(rtl.expr_width(&rtl.continuous_assigns[0].expr)?, 9);
        let RtlExpr::Add(left, right) = &rtl.continuous_assigns[0].expr else {
            panic!("expected add")
        };
        assert_eq!(rtl.expr_width(left)?, 9);
        assert_eq!(rtl.expr_width(right)?, 9);
        let RtlExpr::Resize { value, width: 4 } = &rtl.continuous_assigns[1].expr else {
            panic!("expected output truncation")
        };
        assert_eq!(rtl.expr_width(value)?, 8);
        Ok(())
    }

    #[test]
    fn packed_select_concat_and_multibit_condition_have_explicit_widths() -> eyre::Result<()> {
        let rtl = lower("module m(a,y); input [7:0] a; output [4:0] y; assign y = a ? {1'b1,a[7:4]} : 5'd0; endmodule")?;
        let RtlExpr::Mux {
            select,
            when_true,
            when_false,
        } = &rtl.continuous_assigns[0].expr
        else {
            panic!("expected mux")
        };
        assert_eq!(rtl.expr_width(select)?, 1);
        assert_eq!(rtl.expr_width(when_true)?, 5);
        assert_eq!(rtl.expr_width(when_false)?, 5);
        assert!(matches!(select.as_ref(), RtlExpr::Not(_)));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_packed_semantics() {
        for source in [
            "module m(a,y); input [0:7] a; output y; assign y = a[0]; endmodule",
            "module m(a,y); input [7:4] a; output y; assign y = a[4]; endmodule",
            "module m(a,y); input [7:0] a; output y; assign y = a[8]; endmodule",
            "module m(a,y); input [7:0] a; output y; assign y = a[0:2]; endmodule",
            "module m(a,y); input [7:0] a; output y; assign y = a == 0; endmodule",
        ] {
            assert!(lower(source).is_err(), "unexpectedly accepted {source}");
        }
    }

    #[test]
    fn lowers_latch_and_clocked_process_shapes() -> eyre::Result<()> {
        let rtl = lower(d_latch_source())?;
        let process = single_process(&rtl);
        assert_eq!(process.sensitivity, RtlSensitivity::Combinational);
        assert_eq!(
            process.statements.as_slice(),
            [RtlStmt::If {
                condition: rtl.signal_expr("en")?,
                then_branch: vec![assign_stmt(&rtl, "q", rtl.signal_expr("d")?)?],
                else_branch: Vec::new(),
            }]
        );

        let rtl = lower(dff_source())?;
        let process = single_process(&rtl);
        assert_eq!(
            process.sensitivity,
            RtlSensitivity::Posedge(rtl.signal_ref("clk")?)
        );
        assert_eq!(
            process.statements.as_slice(),
            [assign_stmt(&rtl, "q", rtl.signal_expr("d")?)?]
        );

        Ok(())
    }

    #[test]
    fn lowers_clocked_next_state_expressions() -> eyre::Result<()> {
        let rtl = lower(toggle_source())?;
        assert_eq!(
            single_process(&rtl).statements.as_slice(),
            [assign_stmt(
                &rtl,
                "q",
                RtlExpr::Not(Box::new(rtl.signal_expr("q")?))
            )?]
        );

        let rtl = lower(counter_source())?;
        assert_eq!(
            single_process(&rtl).statements.as_slice(),
            [assign_stmt(
                &rtl,
                "q",
                RtlExpr::Add(
                    Box::new(rtl.signal_expr("q")?),
                    Box::new(RtlExpr::Const { value: 1, width: 1 }),
                ),
            )?]
        );

        Ok(())
    }

    fn lower(source: &str) -> eyre::Result<RtlModule> {
        lower_rtl_module(&parse_module(source)?)
    }

    fn single_process(rtl: &RtlModule) -> &RtlProcess {
        rtl.processes.first().expect("expected one RTL process")
    }

    fn assign_stmt(rtl: &RtlModule, lhs: &str, rhs: RtlExpr) -> eyre::Result<RtlStmt> {
        Ok(RtlStmt::Assign {
            kind: RtlAssignKind::NonBlocking,
            lhs: rtl.signal_ref(lhs)?,
            rhs,
        })
    }

    fn d_latch_source() -> &'static str {
        r#"
        module d_latch(d, en, q);
          input d, en;
          output reg q;
          always @(*) begin
            if (en) begin
              q <= d;
            end
          end
        endmodule
        "#
    }

    fn dff_source() -> &'static str {
        r#"
        module dff(clk, d, q);
          input clk, d;
          output reg q;
          always @(posedge clk) begin
            q <= d;
          end
        endmodule
        "#
    }

    fn toggle_source() -> &'static str {
        r#"
        module toggle(clk, q);
          input clk;
          output reg q;
          always @(posedge clk) begin
            q <= ~q;
          end
        endmodule
        "#
    }

    fn counter_source() -> &'static str {
        r#"
        module counter(clk, q);
          input clk;
          output reg [3:0] q;
          always @(posedge clk) begin
            q <= q + 1;
          end
        endmodule
        "#
    }
}
