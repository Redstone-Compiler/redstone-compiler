use std::collections::{HashMap, HashSet};

use eyre::ContextCompat;

use super::logical::{
    ClockEdge, LogicalBinding, LogicalCell, LogicalCellKind, LogicalDesign, LogicalInput,
    LogicalInstance, LogicalModule, LogicalNet, LogicalOutput, LogicalPort, LogicalPortDirection,
    LogicalValue, LOGICAL_IR_VERSION,
};
use super::RoutableDesign;
use crate::verilog::ast::{
    AlwaysBlock, AlwaysSensitivity, AlwaysStmt, Assignment, BinaryOp, Declaration, Expr, Instance,
    PortDirection, Range, VerilogModule,
};
use crate::verilog::rtl::{lower_rtl_module, RtlExpr, RtlModule, RtlSignalKind, RtlSignalRef};
use crate::verilog::synth::{synthesize_module, SynthCell};

impl LogicalDesign {
    pub fn from_verilog_source(source: &str) -> eyre::Result<Self> {
        let modules = crate::verilog::parser::parse_modules(source)?;
        Self::from_verilog_modules(&modules)
    }

    pub fn from_verilog_modules(modules: &[VerilogModule]) -> eyre::Result<Self> {
        let Some(top) = modules.last() else {
            eyre::bail!("expected at least one Verilog module");
        };
        let design = Self {
            version: LOGICAL_IR_VERSION,
            top: top.name.clone(),
            modules: modules
                .iter()
                .map(logical_module_from_verilog)
                .collect::<eyre::Result<Vec<_>>>()?,
        };
        design.validate()?;
        Ok(design)
    }

    pub fn lower_to_routable(&self) -> eyre::Result<RoutableDesign> {
        super::logical_lowering::lower_logical_to_routable(self)
    }

    pub fn to_verilog_modules(&self) -> eyre::Result<Vec<VerilogModule>> {
        self.validate()?;
        self.modules.iter().map(logical_module_to_verilog).collect()
    }
}

fn logical_module_from_verilog(module: &VerilogModule) -> eyre::Result<LogicalModule> {
    let rtl = lower_rtl_module(module)?;
    let synth = synthesize_module(&rtl)?;
    if !module.instances.is_empty()
        && (!rtl.continuous_assigns.is_empty() || !synth.cells.is_empty())
    {
        eyre::bail!(
            "logical RCIR does not support mixing instances and local cells in module `{}`",
            module.name
        );
    }

    let nets = rtl
        .signals
        .iter()
        .map(|signal| LogicalNet {
            name: signal.name.clone(),
            width: signal.width,
            origin: Some(format!("verilog.signal.{}", signal.name)),
        })
        .collect::<Vec<_>>();
    let ports = rtl
        .ports
        .iter()
        .map(|port| {
            let signal = rtl
                .signals
                .get(port.signal.signal.0)
                .with_context(|| format!("unknown RTL signal id {}", port.signal.signal.0))?;
            let direction = match signal.kind {
                RtlSignalKind::Input => LogicalPortDirection::Input,
                RtlSignalKind::Output | RtlSignalKind::RegOutput => LogicalPortDirection::Output,
                RtlSignalKind::Wire => {
                    eyre::bail!(
                        "module port `{}` is declared as an internal wire",
                        port.name
                    )
                }
            };
            Ok(LogicalPort {
                name: port.name.clone(),
                direction,
                net: signal.name.clone(),
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let instances = module
        .instances
        .iter()
        .map(|instance| LogicalInstance {
            name: instance.instance_name.clone(),
            module: instance.module_name.clone(),
            bindings: instance
                .connections
                .iter()
                .map(|(port, net)| LogicalBinding {
                    port: port.clone(),
                    net: net.clone(),
                })
                .collect(),
            origin: Some(format!("verilog.instance.{}", instance.instance_name)),
        })
        .collect();

    let mut builder = LogicalModuleBuilder::new(&rtl, nets);
    for assignment in &rtl.continuous_assigns {
        let output = rtl.signal_name(assignment.output)?.to_owned();
        let width = rtl.signal_width(assignment.output)?;
        builder.emit_expr_to(&assignment.expr, &output, width, "assign")?;
    }
    for (index, cell) in synth.cells.iter().enumerate() {
        builder.emit_synth_cell(cell, index)?;
    }

    Ok(LogicalModule {
        name: module.name.clone(),
        nets: builder.nets,
        ports,
        cells: builder.cells,
        instances,
    })
}

struct LogicalModuleBuilder<'a> {
    rtl: &'a RtlModule,
    nets: Vec<LogicalNet>,
    cells: Vec<LogicalCell>,
}

impl<'a> LogicalModuleBuilder<'a> {
    fn new(rtl: &'a RtlModule, nets: Vec<LogicalNet>) -> Self {
        Self {
            rtl,
            nets,
            cells: Vec::new(),
        }
    }

    fn emit_synth_cell(&mut self, cell: &SynthCell, index: usize) -> eyre::Result<()> {
        match cell {
            SynthCell::DLatch {
                output: output_ref,
                data,
                enable,
            } => {
                let output_name = self.rtl.signal_name(*output_ref)?.to_owned();
                let width = self.rtl.signal_width(*output_ref)?;
                let cell_name = self.fresh_cell_name(&output_name, "state");
                self.cells.push(LogicalCell {
                    name: cell_name,
                    kind: LogicalCellKind::DLatch { width },
                    inputs: vec![
                        input("d", self.signal_value(*data)?),
                        input("enable", self.signal_value(*enable)?),
                    ],
                    outputs: vec![output("q", output_name)],
                    origin: Some(format!("verilog.process.{index}")),
                });
            }
            SynthCell::Dff {
                output: output_ref,
                data,
                clock,
            }
            | SynthCell::Register {
                output: output_ref,
                data,
                clock,
            } => {
                let output_name = self.rtl.signal_name(*output_ref)?.to_owned();
                let width = self.rtl.signal_width(*output_ref)?;
                let next_role = format!("{output_name}_next");
                let data = self.emit_expr_value(data, width, &next_role)?;
                let kind = match cell {
                    SynthCell::Dff { .. } => LogicalCellKind::Dff {
                        edge: ClockEdge::Posedge,
                    },
                    SynthCell::Register { .. } => LogicalCellKind::Register {
                        width,
                        edge: ClockEdge::Posedge,
                    },
                    SynthCell::DLatch { .. } => unreachable!(),
                };
                let cell_name = self.fresh_cell_name(&output_name, "state");
                self.cells.push(LogicalCell {
                    name: cell_name,
                    kind,
                    inputs: vec![input("d", data), input("clock", self.signal_value(*clock)?)],
                    outputs: vec![output("q", output_name)],
                    origin: Some(format!("verilog.process.{index}")),
                });
            }
        }
        Ok(())
    }

    fn emit_expr_value(
        &mut self,
        expr: &RtlExpr,
        expected_width: usize,
        role: &str,
    ) -> eyre::Result<LogicalValue> {
        match expr {
            RtlExpr::Signal(signal) => self.signal_value(*signal),
            RtlExpr::Const { value, width } => Ok(LogicalValue::Constant {
                value: *value as u128,
                width: *width,
            }),
            _ => {
                let net = self.fresh_net(role, expected_width);
                self.emit_expr_to(expr, &net, expected_width, role)?;
                Ok(LogicalValue::Net { net })
            }
        }
    }

    fn emit_expr_to(
        &mut self,
        expr: &RtlExpr,
        output_net: &str,
        expected_width: usize,
        role: &str,
    ) -> eyre::Result<()> {
        let (kind, inputs) = match expr {
            RtlExpr::Signal(signal) => (
                LogicalCellKind::Buffer,
                vec![input("value", self.signal_value(*signal)?)],
            ),
            RtlExpr::Const { value, width } => (
                LogicalCellKind::Buffer,
                vec![input(
                    "value",
                    LogicalValue::Constant {
                        value: *value as u128,
                        width: *width,
                    },
                )],
            ),
            RtlExpr::Not(value) => (
                LogicalCellKind::Not,
                vec![input(
                    "value",
                    self.emit_expr_value(value, expected_width, role)?,
                )],
            ),
            RtlExpr::Add(left, right) if is_one(right) => (
                LogicalCellKind::Inc,
                vec![input(
                    "value",
                    self.emit_expr_value(left, expected_width, role)?,
                )],
            ),
            RtlExpr::Add(left, right) if is_one(left) => (
                LogicalCellKind::Inc,
                vec![input(
                    "value",
                    self.emit_expr_value(right, expected_width, role)?,
                )],
            ),
            RtlExpr::Add(left, right) => (
                LogicalCellKind::Add,
                self.binary_inputs(left, right, expected_width, role)?,
            ),
            RtlExpr::And(left, right) => (
                LogicalCellKind::And,
                self.binary_inputs(left, right, expected_width, role)?,
            ),
            RtlExpr::Or(left, right) => (
                LogicalCellKind::Or,
                self.binary_inputs(left, right, expected_width, role)?,
            ),
            RtlExpr::Xor(left, right) => (
                LogicalCellKind::Xor,
                self.binary_inputs(left, right, expected_width, role)?,
            ),
            RtlExpr::Mux {
                select,
                when_true,
                when_false,
            } => (
                LogicalCellKind::Mux,
                vec![
                    input("select", self.emit_expr_value(select, 1, role)?),
                    input(
                        "when_true",
                        self.emit_expr_value(when_true, expected_width, role)?,
                    ),
                    input(
                        "when_false",
                        self.emit_expr_value(when_false, expected_width, role)?,
                    ),
                ],
            ),
        };
        let cell_name = self.fresh_cell_name(output_net, role);
        self.cells.push(LogicalCell {
            name: cell_name,
            kind,
            inputs,
            outputs: vec![output("result", output_net.to_owned())],
            origin: Some(format!("verilog.expression.{role}")),
        });
        Ok(())
    }

    fn binary_inputs(
        &mut self,
        left: &RtlExpr,
        right: &RtlExpr,
        width: usize,
        role: &str,
    ) -> eyre::Result<Vec<LogicalInput>> {
        Ok(vec![
            input("lhs", self.emit_expr_value(left, width, role)?),
            input("rhs", self.emit_expr_value(right, width, role)?),
        ])
    }

    fn signal_value(&self, signal: RtlSignalRef) -> eyre::Result<LogicalValue> {
        Ok(LogicalValue::Net {
            net: self.rtl.signal_name(signal)?.to_owned(),
        })
    }

    fn fresh_net(&mut self, role: &str, width: usize) -> String {
        for suffix in 0.. {
            let name = if suffix == 0 {
                role.to_owned()
            } else {
                format!("{role}_{suffix}")
            };
            if self.nets.iter().all(|net| net.name != name) {
                self.nets.push(LogicalNet {
                    name: name.clone(),
                    width,
                    origin: Some(format!("generated.{role}")),
                });
                return name;
            }
        }
        unreachable!("infinite generated-net suffix search exhausted")
    }

    fn fresh_cell_name(&mut self, output: &str, role: &str) -> String {
        let base = if role.ends_with("_next") {
            "next"
        } else if role == "assign" {
            output
        } else {
            role
        };
        for suffix in 0.. {
            let name = if suffix == 0 {
                base.to_owned()
            } else {
                format!("{base}_{suffix}")
            };
            if self.cells.iter().all(|cell| cell.name != name) {
                return name;
            }
        }
        unreachable!("infinite generated-cell suffix search exhausted")
    }
}

fn logical_module_to_verilog(module: &LogicalModule) -> eyre::Result<VerilogModule> {
    if !module.instances.is_empty() && !module.cells.is_empty() {
        eyre::bail!(
            "redstone-v1 lowering does not support mixed cells and instances in module `{}`",
            module.name
        );
    }
    for constant in module.cells.iter().flat_map(|cell| {
        cell.inputs.iter().filter_map(|input| match input.value {
            LogicalValue::Constant { value, .. } => Some(value),
            LogicalValue::Net { .. } => None,
        })
    }) {
        if constant > usize::MAX as u128 {
            eyre::bail!("constant {constant} exceeds the current redstone-v1 mapper integer range");
        }
    }
    let sequential_outputs = module
        .cells
        .iter()
        .filter(|cell| cell.kind.is_sequential())
        .flat_map(|cell| cell.outputs.iter().map(|output| output.net.as_str()))
        .collect::<HashSet<_>>();
    let port_by_net = module
        .ports
        .iter()
        .map(|port| (port.net.as_str(), port))
        .collect::<HashMap<_, _>>();
    let declarations = module
        .nets
        .iter()
        .map(|net| {
            let direction = match port_by_net.get(net.name.as_str()).copied() {
                Some(port) if port.direction == LogicalPortDirection::Input => PortDirection::Input,
                Some(_) if sequential_outputs.contains(net.name.as_str()) => {
                    PortDirection::OutputReg
                }
                Some(_) => PortDirection::Output,
                None => PortDirection::Wire,
            };
            Declaration {
                direction: Some(direction),
                range: (net.width > 1).then_some(Range {
                    msb: net.width - 1,
                    lsb: 0,
                }),
                names: vec![net.name.clone()],
            }
        })
        .collect();

    let sequential_data_nets = module
        .cells
        .iter()
        .filter(|cell| cell.kind.is_sequential())
        .filter_map(|cell| match cell.input_value("d").ok()? {
            LogicalValue::Net { net } => Some(net.as_str()),
            LogicalValue::Constant { .. } => None,
        })
        .collect::<HashSet<_>>();
    let mut assignments = Vec::new();
    for cell in module
        .cells
        .iter()
        .filter(|cell| !cell.kind.is_sequential())
    {
        if matches!(cell.kind, LogicalCellKind::Mux) {
            let output = cell.output("result")?;
            if sequential_data_nets.contains(output.as_str()) {
                continue;
            }
            eyre::bail!("combinational mux lowering is not implemented for redstone-v1");
        }
        assignments.push(Assignment {
            output: cell.output("result")?.clone(),
            expr: immediate_cell_expr(cell)?,
        });
    }

    let drivers = module
        .cells
        .iter()
        .filter(|cell| !cell.kind.is_sequential())
        .map(|cell| Ok((cell.output("result")?.as_str(), cell)))
        .collect::<eyre::Result<HashMap<_, _>>>()?;
    let always_blocks = module
        .cells
        .iter()
        .filter(|cell| cell.kind.is_sequential())
        .map(|cell| sequential_cell_to_always(cell, &drivers))
        .collect::<eyre::Result<Vec<_>>>()?;

    Ok(VerilogModule {
        name: module.name.clone(),
        ports: module.ports.iter().map(|port| port.name.clone()).collect(),
        declarations,
        assignments,
        always_blocks,
        instances: module
            .instances
            .iter()
            .map(|instance| Instance {
                module_name: instance.module.clone(),
                instance_name: instance.name.clone(),
                connections: instance
                    .bindings
                    .iter()
                    .map(|binding| (binding.port.clone(), binding.net.clone()))
                    .collect(),
            })
            .collect(),
    })
}

fn immediate_cell_expr(cell: &LogicalCell) -> eyre::Result<Expr> {
    let value = |pin| cell.input_value(pin).map(value_expr);
    Ok(match cell.kind {
        LogicalCellKind::Buffer => value("value")?,
        LogicalCellKind::Not => Expr::Not(Box::new(value("value")?)),
        LogicalCellKind::And => binary(BinaryOp::And, value("lhs")?, value("rhs")?),
        LogicalCellKind::Or => binary(BinaryOp::Or, value("lhs")?, value("rhs")?),
        LogicalCellKind::Xor => binary(BinaryOp::Xor, value("lhs")?, value("rhs")?),
        LogicalCellKind::Add => binary(BinaryOp::Add, value("lhs")?, value("rhs")?),
        LogicalCellKind::Inc => binary(BinaryOp::Add, value("value")?, Expr::Number(1)),
        LogicalCellKind::Mux => eyre::bail!("mux is not an immediate Verilog expression"),
        LogicalCellKind::DLatch { .. }
        | LogicalCellKind::Dff { .. }
        | LogicalCellKind::Register { .. } => {
            eyre::bail!("sequential cell is not a continuous expression")
        }
    })
}

fn sequential_cell_to_always(
    cell: &LogicalCell,
    drivers: &HashMap<&str, &LogicalCell>,
) -> eyre::Result<AlwaysBlock> {
    let output = cell.output("q")?.clone();
    match &cell.kind {
        LogicalCellKind::DLatch { .. } => {
            let enable = net_value_name(cell.input_value("enable")?, "latch enable")?;
            Ok(AlwaysBlock {
                sensitivity: AlwaysSensitivity::Any,
                body: AlwaysStmt::If {
                    condition: enable.to_owned(),
                    then_branch: Box::new(AlwaysStmt::NonBlockingAssign {
                        output,
                        data: expand_value(cell.input_value("d")?, drivers, &mut Vec::new())?,
                    }),
                },
            })
        }
        LogicalCellKind::Dff { edge } | LogicalCellKind::Register { edge, .. } => {
            if *edge != ClockEdge::Posedge {
                eyre::bail!("redstone-v1 currently supports only posedge state cells");
            }
            let clock = net_value_name(cell.input_value("clock")?, "clock")?;
            let data = cell.input_value("d")?;
            let body = enabled_assignment(data, &output, drivers)?;
            Ok(AlwaysBlock {
                sensitivity: AlwaysSensitivity::Posedge(clock.to_owned()),
                body,
            })
        }
        _ => eyre::bail!("cell `{}` is not sequential", cell.name),
    }
}

fn enabled_assignment(
    data: &LogicalValue,
    output: &str,
    drivers: &HashMap<&str, &LogicalCell>,
) -> eyre::Result<AlwaysStmt> {
    if let LogicalValue::Net { net } = data {
        if let Some(mux) = drivers.get(net.as_str()).copied() {
            if matches!(mux.kind, LogicalCellKind::Mux)
                && matches!(mux.input_value("when_false")?, LogicalValue::Net { net } if net == output)
            {
                let select =
                    net_value_name(mux.input_value("select")?, "clocked enable")?.to_owned();
                return Ok(AlwaysStmt::If {
                    condition: select,
                    then_branch: Box::new(AlwaysStmt::NonBlockingAssign {
                        output: output.to_owned(),
                        data: expand_value(
                            mux.input_value("when_true")?,
                            drivers,
                            &mut Vec::new(),
                        )?,
                    }),
                });
            }
        }
    }
    Ok(AlwaysStmt::NonBlockingAssign {
        output: output.to_owned(),
        data: expand_value(data, drivers, &mut Vec::new())?,
    })
}

fn expand_value(
    value: &LogicalValue,
    drivers: &HashMap<&str, &LogicalCell>,
    active: &mut Vec<String>,
) -> eyre::Result<Expr> {
    let LogicalValue::Net { net } = value else {
        return Ok(value_expr(value));
    };
    let Some(cell) = drivers.get(net.as_str()).copied() else {
        return Ok(Expr::Ident(net.clone()));
    };
    if active.contains(net) {
        eyre::bail!("cycle while expanding logical net `{net}`");
    }
    active.push(net.clone());
    let expand =
        |pin: &str, active: &mut Vec<String>| expand_value(cell.input_value(pin)?, drivers, active);
    let expr = match cell.kind {
        LogicalCellKind::Buffer => expand("value", active)?,
        LogicalCellKind::Not => Expr::Not(Box::new(expand("value", active)?)),
        LogicalCellKind::And => binary(
            BinaryOp::And,
            expand("lhs", active)?,
            expand("rhs", active)?,
        ),
        LogicalCellKind::Or => binary(BinaryOp::Or, expand("lhs", active)?, expand("rhs", active)?),
        LogicalCellKind::Xor => binary(
            BinaryOp::Xor,
            expand("lhs", active)?,
            expand("rhs", active)?,
        ),
        LogicalCellKind::Add => binary(
            BinaryOp::Add,
            expand("lhs", active)?,
            expand("rhs", active)?,
        ),
        LogicalCellKind::Inc => binary(BinaryOp::Add, expand("value", active)?, Expr::Number(1)),
        LogicalCellKind::Mux => {
            eyre::bail!("generic mux expression cannot be represented by the current Verilog AST")
        }
        _ => eyre::bail!("sequential cell cannot drive a combinational expression"),
    };
    active.pop();
    Ok(expr)
}

fn value_expr(value: &LogicalValue) -> Expr {
    match value {
        LogicalValue::Net { net } => Expr::Ident(net.clone()),
        LogicalValue::Constant { value, .. } => Expr::Number(*value as usize),
    }
}

fn net_value_name<'a>(value: &'a LogicalValue, role: &str) -> eyre::Result<&'a str> {
    match value {
        LogicalValue::Net { net } => Ok(net),
        LogicalValue::Constant { .. } => eyre::bail!("{role} must be a net"),
    }
}

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn is_one(expr: &RtlExpr) -> bool {
    matches!(expr, RtlExpr::Const { value: 1, .. })
}

fn input(pin: &str, value: LogicalValue) -> LogicalInput {
    LogicalInput {
        pin: pin.to_owned(),
        value,
    }
}

fn output(pin: &str, net: String) -> LogicalOutput {
    LogicalOutput {
        pin: pin.to_owned(),
        net,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COUNTER: &str = r#"
        module counter(clk, q);
          input clk;
          output reg [1:0] q;
          always @(posedge clk) begin
            q <= q + 1;
          end
        endmodule
    "#;

    #[test]
    fn counter_logical_ir_preserves_increment_and_register() -> eyre::Result<()> {
        let logical = LogicalDesign::from_verilog_source(COUNTER)?;
        let module = logical.module("counter").unwrap();

        assert!(module
            .cells
            .iter()
            .any(|cell| matches!(cell.kind, LogicalCellKind::Inc)));
        assert!(module
            .cells
            .iter()
            .any(|cell| matches!(cell.kind, LogicalCellKind::Register { width: 2, .. })));
        assert!(module
            .nets
            .iter()
            .any(|net| net.name == "q" && net.width == 2));
        Ok(())
    }

    #[test]
    fn counter_logical_ir_lowers_to_routable() -> eyre::Result<()> {
        let logical = LogicalDesign::from_verilog_source(COUNTER)?;
        let routable = logical.lower_to_routable()?;

        assert_eq!(routable.top, "counter");
        assert!(routable.module("counter").is_some_and(|module| matches!(
            module.body,
            super::super::RoutableModuleBody::Composite { .. }
        )));
        Ok(())
    }

    #[test]
    fn combinational_logical_ir_lowers_to_routable_leaf() -> eyre::Result<()> {
        let logical = LogicalDesign::from_verilog_source(
            r#"
            module half_adder(a, b, s, c);
              input a, b;
              output s, c;
              assign s = a ^ b;
              assign c = a & b;
            endmodule
            "#,
        )?;
        let routable = logical.lower_to_routable()?;

        assert_eq!(routable.top, "half_adder");
        assert_eq!(routable.modules.len(), 1);
        Ok(())
    }

    #[test]
    fn structural_hierarchy_survives_logical_ir() -> eyre::Result<()> {
        let logical = LogicalDesign::from_verilog_source(
            r#"
            module inv(a, y);
              input a;
              output y;
              assign y = ~a;
            endmodule

            module top(a, y);
              input a;
              output y;
              inv u0(.a(a), .y(y));
            endmodule
            "#,
        )?;
        let text = logical.to_string();
        let reparsed: LogicalDesign = text.parse()?;
        let top = reparsed.module("top").unwrap();

        assert_eq!(top.instances.len(), 1);
        assert_eq!(top.instances[0].module, "inv");
        assert_eq!(top.instances[0].bindings.len(), 2);
        assert_eq!(reparsed.lower_to_routable()?.top, "top");
        Ok(())
    }
}
