use crate::graph::module::GraphModule;
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::sequential::{SequentialPrimitive, SequentialType};
use crate::verilog::rtl::{
    RtlAssignKind, RtlExpr, RtlModule, RtlSensitivity, RtlSignalRef, RtlStmt,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthNetlist {
    pub cells: Vec<SynthCell>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynthCell {
    DLatch {
        output: RtlSignalRef,
        data: RtlSignalRef,
        enable: RtlSignalRef,
    },
    Dff {
        output: RtlSignalRef,
        data: RtlExpr,
        clock: RtlSignalRef,
    },
    Register {
        output: RtlSignalRef,
        data: RtlExpr,
        clock: RtlSignalRef,
    },
}

pub fn synthesize_module(module: &RtlModule) -> eyre::Result<SynthNetlist> {
    let mut cells = Vec::new();
    for process in &module.processes {
        cells.push(synthesize_process(module, process)?);
    }
    Ok(SynthNetlist { cells })
}

pub fn graph_module_from_single_synth_cell(
    rtl: &RtlModule,
    netlist: &SynthNetlist,
    instance_name: &str,
) -> eyre::Result<Option<GraphModule>> {
    let [cell] = netlist.cells.as_slice() else {
        return Ok(None);
    };

    match cell {
        SynthCell::DLatch {
            output,
            data,
            enable,
        } => Ok(Some(d_latch_graph_module(
            instance_name,
            rtl.signal_name(*data)?,
            rtl.signal_name(*enable)?,
            rtl.signal_name(*output)?,
        ))),
        SynthCell::Dff { .. } | SynthCell::Register { .. } => Ok(None),
    }
}

fn synthesize_process(
    module: &RtlModule,
    process: &crate::verilog::rtl::RtlProcess,
) -> eyre::Result<SynthCell> {
    match process.sensitivity {
        RtlSensitivity::Combinational => synthesize_latch_process(process),
        RtlSensitivity::Posedge(clock) => synthesize_clocked_process(module, process, clock),
    }
}

fn synthesize_latch_process(process: &crate::verilog::rtl::RtlProcess) -> eyre::Result<SynthCell> {
    let [stmt] = process.statements.as_slice() else {
        eyre::bail!("expected a single RTL process statement");
    };
    let RtlStmt::If {
        condition,
        then_branch,
        else_branch,
    } = stmt
    else {
        eyre::bail!("unsupported combinational process shape");
    };
    if !else_branch.is_empty() {
        eyre::bail!("if/else procedural lowering is not implemented yet");
    }
    let [then_stmt] = then_branch.as_slice() else {
        eyre::bail!("expected a single if-then statement");
    };
    let RtlStmt::Assign { kind, lhs, rhs } = then_stmt else {
        eyre::bail!("expected assignment inside if branch");
    };
    if *kind != RtlAssignKind::NonBlocking {
        eyre::bail!("only nonblocking latch assignments are supported");
    }

    let RtlExpr::Signal(enable) = condition else {
        eyre::bail!("latch enable must be a signal");
    };
    let RtlExpr::Signal(data) = rhs else {
        eyre::bail!("latch data must be a signal");
    };

    Ok(SynthCell::DLatch {
        output: *lhs,
        data: *data,
        enable: *enable,
    })
}

fn synthesize_clocked_process(
    module: &RtlModule,
    process: &crate::verilog::rtl::RtlProcess,
    clock: RtlSignalRef,
) -> eyre::Result<SynthCell> {
    let [stmt] = process.statements.as_slice() else {
        eyre::bail!("expected a single clocked RTL process statement");
    };
    let (output, data) = clocked_next_value(stmt)?;

    if module.signal_width(output)? == 1 {
        return Ok(SynthCell::Dff {
            output,
            data,
            clock,
        });
    }

    Ok(SynthCell::Register {
        output,
        data,
        clock,
    })
}

fn clocked_next_value(stmt: &RtlStmt) -> eyre::Result<(RtlSignalRef, RtlExpr)> {
    match stmt {
        RtlStmt::Assign { kind, lhs, rhs } => {
            ensure_nonblocking(*kind)?;
            Ok((*lhs, rhs.clone()))
        }
        RtlStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if !else_branch.is_empty() {
                eyre::bail!("clocked if/else procedural lowering is not implemented yet");
            }
            let [then_stmt] = then_branch.as_slice() else {
                eyre::bail!("expected a single clocked if-then statement");
            };
            let RtlStmt::Assign { kind, lhs, rhs } = then_stmt else {
                eyre::bail!("expected assignment inside clocked if branch");
            };
            ensure_nonblocking(*kind)?;
            Ok((
                *lhs,
                RtlExpr::Mux {
                    select: Box::new(condition.clone()),
                    when_true: Box::new(rhs.clone()),
                    when_false: Box::new(RtlExpr::Signal(*lhs)),
                },
            ))
        }
    }
}

fn ensure_nonblocking(kind: RtlAssignKind) -> eyre::Result<()> {
    if kind != RtlAssignKind::NonBlocking {
        eyre::bail!("only nonblocking sequential assignments are supported");
    }
    Ok(())
}

fn d_latch_graph_module(name: &str, data: &str, enable: &str, output: &str) -> GraphModule {
    let mut graph = Graph::from_nodes(vec![
        GraphNode {
            kind: GraphNodeKind::Input(data.to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Input(enable.to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
                SequentialType::DLatch,
                vec![data.to_owned(), enable.to_owned()],
                vec![output.to_owned()],
            )),
            inputs: vec![0, 1],
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Output(output.to_owned()),
            inputs: vec![2],
            ..Default::default()
        },
    ]);
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify().unwrap();

    let mut module: GraphModule = graph.into();
    module.name = name.to_owned();
    module
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verilog::parser::parse_module;
    use crate::verilog::rtl::lower_rtl_module;

    #[test]
    fn synthesizes_latch_and_dff_cells() -> eyre::Result<()> {
        let (rtl, netlist) = synthesize(d_latch_source())?;
        assert_eq!(
            netlist.cells,
            vec![SynthCell::DLatch {
                output: rtl.signal_ref("q")?,
                data: rtl.signal_ref("d")?,
                enable: rtl.signal_ref("en")?,
            }]
        );

        let (rtl, netlist) = synthesize(dff_source())?;
        assert_eq!(
            netlist.cells,
            vec![SynthCell::Dff {
                output: rtl.signal_ref("q")?,
                data: rtl.signal_expr("d")?,
                clock: rtl.signal_ref("clk")?,
            }]
        );

        Ok(())
    }

    #[test]
    fn synthesizes_clocked_next_state_cells() -> eyre::Result<()> {
        let (rtl, netlist) = synthesize(enabled_dff_source())?;
        assert_eq!(
            netlist.cells,
            vec![SynthCell::Dff {
                output: rtl.signal_ref("q")?,
                data: RtlExpr::Mux {
                    select: Box::new(rtl.signal_expr("en")?),
                    when_true: Box::new(rtl.signal_expr("d")?),
                    when_false: Box::new(rtl.signal_expr("q")?),
                },
                clock: rtl.signal_ref("clk")?,
            }]
        );

        let (rtl, netlist) = synthesize(toggle_source())?;
        assert_eq!(
            netlist.cells,
            vec![SynthCell::Dff {
                output: rtl.signal_ref("q")?,
                data: RtlExpr::Not(Box::new(rtl.signal_expr("q")?)),
                clock: rtl.signal_ref("clk")?,
            }]
        );

        let (rtl, netlist) = synthesize(counter_source())?;
        assert_eq!(
            netlist.cells,
            vec![SynthCell::Register {
                output: rtl.signal_ref("q")?,
                data: RtlExpr::Add(
                    Box::new(rtl.signal_expr("q")?),
                    Box::new(RtlExpr::Const { value: 1, width: 1 }),
                ),
                clock: rtl.signal_ref("clk")?,
            }]
        );

        Ok(())
    }

    fn synthesize(source: &str) -> eyre::Result<(RtlModule, SynthNetlist)> {
        let rtl = lower_rtl_module(&parse_module(source)?)?;
        let netlist = synthesize_module(&rtl)?;
        Ok((rtl, netlist))
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

    fn enabled_dff_source() -> &'static str {
        r#"
        module dff_en(clk, en, d, q);
          input clk, en, d;
          output reg q;
          always @(posedge clk) begin
            if (en) begin
              q <= d;
            end
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
