use std::collections::{HashMap, HashSet};

use eyre::{ContextCompat, WrapErr};

use super::logical::{
    ClockEdge, LogicalCell, LogicalCellKind, LogicalDesign, LogicalModule, LogicalPortDirection,
    LogicalValue,
};
use super::RoutableDesign;
use crate::graph::logic::LogicGraph;
use crate::graph::module::{
    GraphModule, GraphModuleContext, GraphModuleDesign, GraphModulePort, GraphModulePortTarget,
    GraphModulePortType, GraphModuleVariable,
};
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::logic::{Logic, LogicType};
use crate::sequential::{SequentialPrimitive, SequentialType};

/// Lowers the typed logical netlist without reconstructing Verilog syntax.
///
/// `GraphModuleDesign` remains a legacy adapter boundary for the current PnR
/// implementation. The semantic lowering itself is driven exclusively by
/// `LogicalDesign`; direct RCIR input and Verilog input therefore take the same
/// path from this point onward.
pub(crate) fn lower_logical_to_routable(design: &LogicalDesign) -> eyre::Result<RoutableDesign> {
    design.validate()?;
    let graph_design = lower_logical_design(design)
        .wrap_err("logical IR uses a construct not yet supported by redstone-v1 mapping")?;
    RoutableDesign::from_graph_module_design(&graph_design)
}

fn lower_logical_design(design: &LogicalDesign) -> eyre::Result<GraphModuleDesign> {
    let definitions = design
        .modules
        .iter()
        .map(|module| (module.name.as_str(), module))
        .collect::<HashMap<_, _>>();
    let top = definitions
        .get(design.top.as_str())
        .copied()
        .with_context(|| format!("unknown logical top module `{}`", design.top))?;

    if top.instances.is_empty() {
        if let Some(state_design) = lower_state_design(top)? {
            return Ok(state_design);
        }
        return Ok(GraphModuleDesign::with_top_module(
            GraphModuleContext::default(),
            graph_backed_module(top, &top.name)?,
        ));
    }

    if !top.cells.is_empty() {
        eyre::bail!(
            "mixed logical cells and instances are not supported in module `{}` yet",
            top.name
        );
    }

    let mut context = GraphModuleContext::default();
    for instance in &top.instances {
        let definition = definitions
            .get(instance.module.as_str())
            .copied()
            .with_context(|| {
                format!(
                    "instance `{}` references unknown logical module `{}`",
                    instance.name, instance.module
                )
            })?;
        if !definition.instances.is_empty() {
            eyre::bail!(
                "nested logical hierarchy is not supported for child `{}` yet",
                instance.name
            );
        }
        context.append(graph_backed_module(definition, &instance.name)?);
    }

    Ok(GraphModuleDesign::with_top_module(
        context,
        hierarchical_module(top, &definitions)?,
    ))
}

fn lower_state_design(module: &LogicalModule) -> eyre::Result<Option<GraphModuleDesign>> {
    let sequential = module
        .cells
        .iter()
        .filter(|cell| cell.kind.is_sequential())
        .collect::<Vec<_>>();
    let [state] = sequential.as_slice() else {
        return Ok(None);
    };

    let (width, edge) = match state.kind {
        LogicalCellKind::Dff { edge } => (1, edge),
        LogicalCellKind::Register { width, edge } => (width, edge),
        LogicalCellKind::DLatch { .. } => return Ok(None),
        _ => unreachable!(),
    };
    if edge != ClockEdge::Posedge {
        eyre::bail!("redstone-v1 currently supports only posedge registers");
    }

    let output = state.output("q")?;
    let clock = net_value(state.input_value("clock")?, "register clock")?;
    let data = net_value(state.input_value("d")?, "register data")?;
    let driver = module.cells.iter().find(|cell| {
        !cell.kind.is_sequential() && cell.output("result").ok().map(String::as_str) == Some(data)
    });

    if width > 1 {
        let driver = driver.with_context(|| {
            format!(
                "register `{}` data net `{data}` has no logical driver",
                state.name
            )
        })?;
        if !matches!(driver.kind, LogicalCellKind::Inc)
            || net_value(driver.input_value("value")?, "increment input")? != output
        {
            eyre::bail!("redstone-v1 currently supports only `register <= register + 1`");
        }
        return register_increment_design(&module.name, output, clock, width).map(Some);
    }

    let next_expr = match driver {
        Some(driver) if matches!(driver.kind, LogicalCellKind::Inc) => {
            let input = net_value(driver.input_value("value")?, "increment input")?;
            if input != output {
                eyre::bail!("scalar increment must feed back from its register output");
            }
            format!("~{output}")
        }
        Some(driver) if matches!(driver.kind, LogicalCellKind::Not) => {
            format!("~{}", net_value(driver.input_value("value")?, "not input")?)
        }
        Some(driver) if matches!(driver.kind, LogicalCellKind::Buffer) => {
            net_value(driver.input_value("value")?, "buffer input")?.to_owned()
        }
        Some(_) => eyre::bail!("unsupported scalar register data operation"),
        None => data.to_owned(),
    };
    scalar_dff_design(&module.name, output, clock, &next_expr).map(Some)
}

fn scalar_dff_design(
    module_name: &str,
    output: &str,
    clock: &str,
    next_expr: &str,
) -> eyre::Result<GraphModuleDesign> {
    let clock_inverter = format!("{output}_clk_inv");
    let next = format!("{output}_next");
    let master = format!("{output}_master");
    let slave = format!("{output}_slave");

    let mut context = GraphModuleContext::default();
    context.append(not_clock_module(&clock_inverter)?);
    context.append(combinational_output_module(&next, next_expr, "d")?);
    context.append(d_latch_graph_module(&master));
    context.append(d_latch_graph_module(&slave));

    let mut vars = vec![
        module_var((&next, "d"), (&master, "d")),
        module_var((&master, "q"), (&slave, "d")),
        module_var((&clock_inverter, "clk_n"), (&master, "en")),
    ];
    if next_expr.contains(output) {
        vars.push(module_var((&slave, "q"), (&next, output)));
    }

    Ok(GraphModuleDesign::with_top_module(
        context,
        GraphModule {
            name: module_name.to_owned(),
            graph: None,
            instances: vec![clock_inverter.clone(), next.clone(), master, slave.clone()],
            vars,
            ports: vec![
                GraphModulePort {
                    name: clock.to_owned(),
                    port_type: GraphModulePortType::InputNet,
                    target: GraphModulePortTarget::Wire(vec![
                        (clock_inverter, "clk".to_owned()),
                        (slave.clone(), "en".to_owned()),
                    ]),
                },
                GraphModulePort {
                    name: output.to_owned(),
                    port_type: GraphModulePortType::OutputNet,
                    target: GraphModulePortTarget::Module(slave, "q".to_owned()),
                },
            ],
        },
    ))
}

fn register_increment_design(
    module_name: &str,
    output: &str,
    clock: &str,
    width: usize,
) -> eyre::Result<GraphModuleDesign> {
    let mut context = GraphModuleContext::default();
    let mut instances = Vec::new();
    let mut vars = Vec::new();
    let mut ports = Vec::new();

    for carry_bit in 2..width {
        let carry = carry_module_name(output, carry_bit);
        context.append(combinational_output_module(
            &carry,
            &carry_expr(output, carry_bit),
            &carry_signal_name(carry_bit),
        )?);
        instances.push(carry.clone());
        for input in carry_inputs(carry_bit) {
            connect_increment_input(&mut vars, output, input, &carry);
        }
    }

    for bit in 0..width {
        let bit_name = bit_signal_name(output, bit);
        let clock_inverter = format!("{bit_name}_clk_inv");
        let next = format!("{bit_name}_next");
        let master = format!("{bit_name}_master");
        let slave = format!("{bit_name}_slave");

        context.append(not_clock_module(&clock_inverter)?);
        context.append(next_bit_module(&next, output, bit)?);
        context.append(d_latch_graph_module(&master));
        context.append(d_latch_graph_module(&slave));
        instances.extend([
            clock_inverter.clone(),
            next.clone(),
            master.clone(),
            slave.clone(),
        ]);

        vars.push(module_var((&next, "d"), (&master, "d")));
        vars.push(module_var((&master, "q"), (&slave, "d")));
        vars.push(module_var((&clock_inverter, "clk_n"), (&master, "en")));
        for input in next_bit_inputs(bit) {
            connect_increment_input(&mut vars, output, input, &next);
        }
        ports.push(GraphModulePort {
            name: bit_name,
            port_type: GraphModulePortType::OutputNet,
            target: GraphModulePortTarget::Module(slave, "q".to_owned()),
        });
    }

    ports.insert(
        0,
        GraphModulePort {
            name: clock.to_owned(),
            port_type: GraphModulePortType::InputNet,
            target: GraphModulePortTarget::Wire(
                (0..width)
                    .flat_map(|bit| {
                        let bit_name = bit_signal_name(output, bit);
                        [
                            (format!("{bit_name}_clk_inv"), "clk".to_owned()),
                            (format!("{bit_name}_slave"), "en".to_owned()),
                        ]
                    })
                    .collect(),
            ),
        },
    );

    Ok(GraphModuleDesign::with_top_module(
        context,
        GraphModule {
            name: module_name.to_owned(),
            graph: None,
            instances,
            vars,
            ports,
        },
    ))
}

fn graph_backed_module(module: &LogicalModule, name: &str) -> eyre::Result<GraphModule> {
    if !module.instances.is_empty() {
        eyre::bail!("logical child module `{}` is not a leaf", module.name);
    }
    if module.nets.iter().any(|net| net.width != 1) {
        eyre::bail!(
            "generic logical leaf lowering currently requires scalar nets in module `{}`",
            module.name
        );
    }

    let mut nodes = Vec::<GraphNode>::new();
    let mut producers = HashMap::<String, usize>::new();
    for port in module
        .ports
        .iter()
        .filter(|port| port.direction == LogicalPortDirection::Input)
    {
        let id = nodes.len();
        nodes.push(GraphNode {
            kind: GraphNodeKind::Input(port.name.clone()),
            ..Default::default()
        });
        producers.insert(port.net.clone(), id);
    }

    let mut pending = module.cells.iter().collect::<Vec<_>>();
    while !pending.is_empty() {
        let before = pending.len();
        let mut unresolved = Vec::new();
        for cell in pending {
            if !emit_graph_cell(cell, &mut nodes, &mut producers)? {
                unresolved.push(cell);
            }
        }
        pending = unresolved;
        if pending.len() == before {
            let names = pending
                .iter()
                .map(|cell| cell.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            eyre::bail!("unresolved or cyclic logical leaf cells: {names}");
        }
    }

    for port in module
        .ports
        .iter()
        .filter(|port| port.direction == LogicalPortDirection::Output)
    {
        let input = producers
            .get(&port.net)
            .copied()
            .with_context(|| format!("output port `{}` has no logical producer", port.name))?;
        nodes.push(GraphNode {
            kind: GraphNodeKind::Output(port.name.clone()),
            inputs: vec![input],
            ..Default::default()
        });
    }

    let mut graph = Graph::from_nodes(nodes);
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify()?;
    let mut graph_module: GraphModule = graph.into();
    graph_module.name = name.to_owned();
    for port in &mut graph_module.ports {
        let logical = module
            .ports
            .iter()
            .find(|candidate| candidate.name == port.name)
            .with_context(|| format!("unknown logical port `{}`", port.name))?;
        port.port_type = match logical.direction {
            LogicalPortDirection::Input => GraphModulePortType::InputNet,
            LogicalPortDirection::Output => GraphModulePortType::OutputNet,
        };
    }
    Ok(graph_module)
}

fn emit_graph_cell(
    cell: &LogicalCell,
    nodes: &mut Vec<GraphNode>,
    producers: &mut HashMap<String, usize>,
) -> eyre::Result<bool> {
    let input_node = |pin: &str| -> eyre::Result<Option<usize>> {
        match cell.input_value(pin)? {
            LogicalValue::Net { net } => Ok(producers.get(net).copied()),
            LogicalValue::Constant { .. } => {
                eyre::bail!("logical constants are not supported by scalar leaf lowering yet")
            }
        }
    };
    let output = cell
        .outputs
        .first()
        .with_context(|| format!("cell `{}` has no output", cell.name))?
        .net
        .clone();

    if matches!(cell.kind, LogicalCellKind::Buffer) {
        let Some(source) = input_node("value")? else {
            return Ok(false);
        };
        producers.insert(output, source);
        return Ok(true);
    }

    let (kind, pins): (GraphNodeKind, &[&str]) = match &cell.kind {
        LogicalCellKind::Not => (
            GraphNodeKind::Logic(Logic {
                logic_type: LogicType::Not,
            }),
            &["value"],
        ),
        LogicalCellKind::And => (
            GraphNodeKind::Logic(Logic {
                logic_type: LogicType::And,
            }),
            &["lhs", "rhs"],
        ),
        LogicalCellKind::Or => (
            GraphNodeKind::Logic(Logic {
                logic_type: LogicType::Or,
            }),
            &["lhs", "rhs"],
        ),
        LogicalCellKind::Xor => (
            GraphNodeKind::Logic(Logic {
                logic_type: LogicType::Xor,
            }),
            &["lhs", "rhs"],
        ),
        LogicalCellKind::DLatch { .. } => (
            GraphNodeKind::Sequential(SequentialPrimitive::new(
                SequentialType::DLatch,
                vec!["d".to_owned(), "en".to_owned()],
                vec!["q".to_owned()],
            )),
            &["d", "enable"],
        ),
        LogicalCellKind::Add
        | LogicalCellKind::Inc
        | LogicalCellKind::Mux
        | LogicalCellKind::Dff { .. }
        | LogicalCellKind::Register { .. } => {
            eyre::bail!(
                "logical cell `{}` requires target mapping before scalar leaf lowering",
                cell.name
            )
        }
        LogicalCellKind::Buffer => unreachable!(),
    };
    let mut inputs = Vec::with_capacity(pins.len());
    for pin in pins {
        let Some(node) = input_node(pin)? else {
            return Ok(false);
        };
        inputs.push(node);
    }
    let id = nodes.len();
    nodes.push(GraphNode {
        kind,
        inputs,
        tag: cell.name.clone(),
        ..Default::default()
    });
    producers.insert(output, id);
    Ok(true)
}

fn hierarchical_module(
    module: &LogicalModule,
    definitions: &HashMap<&str, &LogicalModule>,
) -> eyre::Result<GraphModule> {
    if module.nets.iter().any(|net| net.width != 1) {
        eyre::bail!("hierarchical routable lowering currently requires scalar nets");
    }
    let mut sources = HashMap::<String, Vec<(String, String)>>::new();
    let mut sinks = HashMap::<String, Vec<(String, String)>>::new();
    for instance in &module.instances {
        let definition = definitions
            .get(instance.module.as_str())
            .copied()
            .with_context(|| format!("unknown logical module `{}`", instance.module))?;
        for binding in &instance.bindings {
            let port = definition
                .ports
                .iter()
                .find(|port| port.name == binding.port)
                .with_context(|| {
                    format!(
                        "instance `{}.{}` has no matching port",
                        instance.name, binding.port
                    )
                })?;
            let endpoint = (instance.name.clone(), binding.port.clone());
            match port.direction {
                LogicalPortDirection::Input => {
                    sinks.entry(binding.net.clone()).or_default().push(endpoint)
                }
                LogicalPortDirection::Output => sources
                    .entry(binding.net.clone())
                    .or_default()
                    .push(endpoint),
            }
        }
    }

    let top_ports = module
        .ports
        .iter()
        .map(|port| port.net.as_str())
        .collect::<HashSet<_>>();
    let mut vars = Vec::new();
    for net in &module.nets {
        if top_ports.contains(net.name.as_str()) {
            continue;
        }
        let Some(source) = single_endpoint(sources.get(&net.name)) else {
            continue;
        };
        for target in sinks.get(&net.name).into_iter().flatten() {
            vars.push(GraphModuleVariable {
                var_type: GraphModulePortType::InputNet,
                source: source.clone(),
                target: target.clone(),
            });
        }
    }

    let ports = module
        .ports
        .iter()
        .map(|port| GraphModulePort {
            name: port.name.clone(),
            port_type: match port.direction {
                LogicalPortDirection::Input => GraphModulePortType::InputNet,
                LogicalPortDirection::Output => GraphModulePortType::OutputNet,
            },
            target: match port.direction {
                LogicalPortDirection::Input => target_from_endpoints(sinks.get(&port.net)),
                LogicalPortDirection::Output => target_from_endpoints(sources.get(&port.net)),
            },
        })
        .collect();

    Ok(GraphModule {
        name: module.name.clone(),
        graph: None,
        instances: module
            .instances
            .iter()
            .map(|instance| instance.name.clone())
            .collect(),
        vars,
        ports,
    })
}

fn combinational_output_module(name: &str, expr: &str, output: &str) -> eyre::Result<GraphModule> {
    let mut module: GraphModule = LogicGraph::from_stmt(expr, output)?.graph.into();
    module.name = name.to_owned();
    Ok(module)
}

fn next_bit_module(name: &str, signal: &str, bit: usize) -> eyre::Result<GraphModule> {
    let bit_name = bit_signal_name(signal, bit);
    if bit == 0 {
        return combinational_output_module(name, &format!("~{bit_name}"), "d");
    }
    let rhs = if bit == 1 {
        bit_signal_name(signal, 0)
    } else {
        carry_signal_name(bit)
    };
    buffered_xor_output_module(name, &bit_name, &rhs, "d")
}

fn buffered_xor_output_module(
    name: &str,
    left: &str,
    right: &str,
    output: &str,
) -> eyre::Result<GraphModule> {
    let product = format!("{output}_and");
    let mut graph = LogicGraph::from_stmt(&format!("{left}&{right}"), &product)?;
    graph.graph.merge(
        LogicGraph::from_stmt(
            &format!("(~({product}|~{left}))|(~({product}|~{right}))"),
            output,
        )?
        .graph,
    );
    graph.graph.remove_output(&product);
    let mut module: GraphModule = graph.graph.into();
    module.name = name.to_owned();
    Ok(module)
}

fn not_clock_module(name: &str) -> eyre::Result<GraphModule> {
    combinational_output_module(name, "~clk", "clk_n")
}

fn d_latch_graph_module(name: &str) -> GraphModule {
    let mut graph = Graph::from_nodes(vec![
        GraphNode {
            kind: GraphNodeKind::Input("d".to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Input("en".to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
                SequentialType::DLatch,
                vec!["d".to_owned(), "en".to_owned()],
                vec!["q".to_owned()],
            )),
            inputs: vec![0, 1],
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Output("q".to_owned()),
            inputs: vec![2],
            ..Default::default()
        },
    ]);
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph
        .verify()
        .expect("built-in D latch graph must be valid");
    let mut module: GraphModule = graph.into();
    module.name = name.to_owned();
    module
}

#[derive(Clone, Copy)]
enum IncrementInput {
    Bit(usize),
    Carry(usize),
}

fn next_bit_inputs(bit: usize) -> Vec<IncrementInput> {
    match bit {
        0 => vec![IncrementInput::Bit(0)],
        1 => vec![IncrementInput::Bit(1), IncrementInput::Bit(0)],
        _ => vec![IncrementInput::Bit(bit), IncrementInput::Carry(bit)],
    }
}

fn carry_inputs(bit: usize) -> Vec<IncrementInput> {
    if bit == 2 {
        vec![IncrementInput::Bit(1), IncrementInput::Bit(0)]
    } else {
        vec![IncrementInput::Bit(bit - 1), IncrementInput::Carry(bit - 1)]
    }
}

fn connect_increment_input(
    vars: &mut Vec<GraphModuleVariable>,
    signal: &str,
    input: IncrementInput,
    target: &str,
) {
    let source = match input {
        IncrementInput::Bit(bit) => (format!("{}_{}_slave", signal, bit), "q".to_owned()),
        IncrementInput::Carry(bit) => (carry_module_name(signal, bit), carry_signal_name(bit)),
    };
    let target_port = match input {
        IncrementInput::Bit(bit) => bit_signal_name(signal, bit),
        IncrementInput::Carry(bit) => carry_signal_name(bit),
    };
    vars.push(GraphModuleVariable {
        var_type: GraphModulePortType::InputNet,
        source,
        target: (target.to_owned(), target_port),
    });
}

fn carry_expr(signal: &str, bit: usize) -> String {
    if bit == 2 {
        format!(
            "{}&{}",
            bit_signal_name(signal, 1),
            bit_signal_name(signal, 0)
        )
    } else {
        format!(
            "{}&{}",
            bit_signal_name(signal, bit - 1),
            carry_signal_name(bit - 1)
        )
    }
}

fn bit_signal_name(signal: &str, bit: usize) -> String {
    format!("{signal}_{bit}")
}

fn carry_signal_name(bit: usize) -> String {
    format!("carry_{bit}")
}

fn carry_module_name(signal: &str, bit: usize) -> String {
    format!("{signal}_carry_{bit}")
}

fn module_var(source: (&str, &str), target: (&str, &str)) -> GraphModuleVariable {
    GraphModuleVariable {
        var_type: GraphModulePortType::InputNet,
        source: (source.0.to_owned(), source.1.to_owned()),
        target: (target.0.to_owned(), target.1.to_owned()),
    }
}

fn target_from_endpoints(endpoints: Option<&Vec<(String, String)>>) -> GraphModulePortTarget {
    match endpoints {
        Some(endpoints) if endpoints.len() == 1 => {
            GraphModulePortTarget::Module(endpoints[0].0.clone(), endpoints[0].1.clone())
        }
        Some(endpoints) => GraphModulePortTarget::Wire(endpoints.clone()),
        None => GraphModulePortTarget::Wire(Vec::new()),
    }
}

fn single_endpoint(endpoints: Option<&Vec<(String, String)>>) -> Option<(String, String)> {
    let endpoints = endpoints?;
    (endpoints.len() == 1).then(|| endpoints[0].clone())
}

fn net_value<'a>(value: &'a LogicalValue, role: &str) -> eyre::Result<&'a str> {
    match value {
        LogicalValue::Net { net } => Ok(net),
        LogicalValue::Constant { .. } => eyre::bail!("{role} must be a net"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_lowering_does_not_require_verilog_reconstruction() -> eyre::Result<()> {
        let logical = LogicalDesign::from_verilog_source(
            r#"
            module counter(clk, q);
              input clk;
              output reg [1:0] q;
              always @(posedge clk) begin
                q <= q + 1;
              end
            endmodule
            "#,
        )?;

        let routable = lower_logical_to_routable(&logical)?;
        let top = routable.module("counter").context("missing counter top")?;
        let super::super::routable::RoutableModuleBody::Composite { instances, .. } = &top.body
        else {
            panic!("counter must lower to a composite routable module");
        };
        assert!(instances
            .iter()
            .any(|instance| instance.name == "q_0_master"));
        assert!(instances
            .iter()
            .any(|instance| instance.name == "q_1_slave"));
        Ok(())
    }
}
