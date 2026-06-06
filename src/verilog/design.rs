use std::collections::{HashMap, HashSet};

use eyre::ContextCompat;

use crate::graph::logic::LogicGraph;
use crate::graph::module::{
    GraphModule, GraphModuleContext, GraphModuleDesign, GraphModulePort, GraphModulePortTarget,
    GraphModulePortType, GraphModuleVariable,
};
use crate::verilog::ast::{PortDirection, VerilogModule};
use crate::verilog::rtl::{lower_rtl_module, RtlExpr, RtlModule, RtlSignalRef};
use crate::verilog::synth::{
    d_latch_graph_module, graph_module_from_single_synth_cell, synthesize_module, SynthCell,
};

pub fn lower_design_modules(modules: &[VerilogModule]) -> eyre::Result<GraphModuleDesign> {
    let Some(top_module) = modules.last() else {
        eyre::bail!("expected at least one Verilog module");
    };
    lower_design_with_top(modules, &top_module.name)
}

pub fn lower_design_with_top(
    modules: &[VerilogModule],
    top_name: &str,
) -> eyre::Result<GraphModuleDesign> {
    let definitions = modules
        .iter()
        .map(|module| (module.name.as_str(), module))
        .collect::<HashMap<_, _>>();
    let top = definitions
        .get(top_name)
        .copied()
        .with_context(|| format!("unknown top Verilog module `{top_name}`"))?;

    if top.instances.is_empty() {
        if let Some(design) = synthesized_design(top)? {
            return Ok(design);
        }

        return Ok(GraphModuleDesign::with_top_module(
            GraphModuleContext::default(),
            graph_backed_module(&definitions, top, &top.name)?,
        ));
    }

    let mut context = GraphModuleContext::default();
    for instance in &top.instances {
        context.append(graph_backed_module(
            &definitions,
            module_definition(&definitions, &instance.module_name)?,
            &instance.instance_name,
        )?);
    }

    Ok(GraphModuleDesign::with_top_module(
        context,
        hierarchical_module(top, &definitions)?,
    ))
}

fn module_definition<'a>(
    definitions: &'a HashMap<&str, &'a VerilogModule>,
    module_name: &str,
) -> eyre::Result<&'a VerilogModule> {
    definitions
        .get(module_name)
        .copied()
        .with_context(|| format!("unknown Verilog module `{module_name}`"))
}

fn graph_backed_module(
    _definitions: &HashMap<&str, &VerilogModule>,
    module: &VerilogModule,
    instance_name: &str,
) -> eyre::Result<GraphModule> {
    let rtl = lower_rtl_module(module)?;
    let netlist = synthesize_module(&rtl)?;
    if let Some(module) = graph_module_from_single_synth_cell(&rtl, &netlist, instance_name)? {
        return Ok(module);
    }
    if !netlist.cells.is_empty() {
        eyre::bail!(
            "synthesized Verilog module `{}` has no GraphModule adapter yet",
            module.name
        );
    }

    if !module.instances.is_empty() {
        eyre::bail!(
            "nested hierarchical Verilog module `{}` is not supported as a leaf instance yet",
            module.name
        );
    }

    let graph = crate::verilog::lower::lower_module(module)?;
    let mut graph_module: GraphModule = graph.graph.into();
    for port in &mut graph_module.ports {
        port.port_type = graph_module_port_type(port_direction(module, &port.name)?);
    }
    graph_module.name = instance_name.to_owned();
    Ok(graph_module)
}

fn synthesized_design(module: &VerilogModule) -> eyre::Result<Option<GraphModuleDesign>> {
    let rtl = lower_rtl_module(module)?;
    let netlist = synthesize_module(&rtl)?;
    let [cell] = netlist.cells.as_slice() else {
        return Ok(None);
    };

    match cell {
        SynthCell::Register {
            output,
            data,
            clock,
        } => register_design(&rtl, *output, data, *clock).map(Some),
        SynthCell::Dff {
            output,
            data,
            clock,
        } => dff_design(&rtl, *output, data, *clock).map(Some),
        SynthCell::DLatch { .. } => Ok(None),
    }
}

fn dff_design(
    rtl: &RtlModule,
    output: RtlSignalRef,
    data: &RtlExpr,
    clock: RtlSignalRef,
) -> eyre::Result<GraphModuleDesign> {
    let output_name = rtl.signal_name(output)?;
    let clock_name = rtl.signal_name(clock)?;
    let next_expr = dff_next_expr(rtl, output, data)?;
    let clock_inverter_name = format!("{output_name}_clk_inv");
    let next_name = format!("{output_name}_next");
    let master_name = format!("{output_name}_master");
    let slave_name = format!("{output_name}_slave");

    let mut context = GraphModuleContext::default();
    context.append(not_clock_module(&clock_inverter_name)?);
    context.append(combinational_output_module(&next_name, &next_expr, "d")?);
    context.append(d_latch_graph_module(&master_name, "d", "en", "q"));
    context.append(d_latch_graph_module(&slave_name, "d", "en", "q"));

    Ok(GraphModuleDesign::with_top_module(
        context,
        GraphModule {
            name: rtl.name.clone(),
            graph: None,
            instances: vec![
                clock_inverter_name.clone(),
                next_name.clone(),
                master_name.clone(),
                slave_name.clone(),
            ],
            vars: vec![
                module_var((&next_name, "d"), (&master_name, "d")),
                module_var((&master_name, "q"), (&slave_name, "d")),
                module_var((&clock_inverter_name, "clk_n"), (&master_name, "en")),
                module_var((&clock_inverter_name, "clk"), (&slave_name, "en")),
                module_var((&slave_name, "q"), (&next_name, output_name)),
            ],
            ports: vec![
                GraphModulePort {
                    name: clock_name.to_owned(),
                    port_type: GraphModulePortType::InputNet,
                    target: GraphModulePortTarget::Module(clock_inverter_name, "clk".to_owned()),
                },
                GraphModulePort {
                    name: output_name.to_owned(),
                    port_type: GraphModulePortType::OutputNet,
                    target: GraphModulePortTarget::Module(slave_name, "q".to_owned()),
                },
            ],
        },
    ))
}

fn dff_next_expr(rtl: &RtlModule, output: RtlSignalRef, data: &RtlExpr) -> eyre::Result<String> {
    let output_name = rtl.signal_name(output)?;
    if is_increment_by_one(output, data) {
        return Ok(format!("~{output_name}"));
    }
    match data {
        RtlExpr::Signal(signal) => Ok(rtl.signal_name(*signal)?.to_owned()),
        RtlExpr::Not(expr) => {
            let RtlExpr::Signal(signal) = expr.as_ref() else {
                eyre::bail!("only signal negation DFF data is supported for GraphModule lowering");
            };
            Ok(format!("~{}", rtl.signal_name(*signal)?))
        }
        _ => eyre::bail!("unsupported DFF data expression for GraphModule lowering"),
    }
}

fn register_design(
    rtl: &RtlModule,
    output: RtlSignalRef,
    data: &RtlExpr,
    clock: RtlSignalRef,
) -> eyre::Result<GraphModuleDesign> {
    let output_name = rtl.signal_name(output)?;
    let clock_name = rtl.signal_name(clock)?;
    let width = rtl.signal_width(output)?;
    ensure_register_increment_expr(output, data)?;

    let mut context = GraphModuleContext::default();
    let mut instances = Vec::new();
    let mut vars = Vec::new();
    let mut ports = Vec::new();
    let clock_inverter_name = format!("{output_name}_clk_inv");

    context.append(not_clock_module(&clock_inverter_name)?);
    instances.push(clock_inverter_name.clone());

    for carry_bit in 2..width {
        let carry_name = carry_module_name(output_name, carry_bit);
        context.append(combinational_output_module(
            &carry_name,
            &carry_expr(output_name, carry_bit),
            &carry_signal_name(carry_bit),
        )?);
        instances.push(carry_name.clone());

        for input in carry_inputs(carry_bit) {
            connect_increment_input(&mut vars, output_name, input, &carry_name);
        }
    }

    for bit in 0..width {
        let bit_name = bit_signal_name(output_name, bit);
        let next_name = format!("{bit_name}_next");
        let master_name = format!("{bit_name}_master");
        let slave_name = format!("{bit_name}_slave");

        context.append(combinational_output_module(
            &next_name,
            &next_bit_expr(output_name, bit),
            "d",
        )?);
        context.append(d_latch_graph_module(&master_name, "d", "en", "q"));
        context.append(d_latch_graph_module(&slave_name, "d", "en", "q"));

        instances.extend([next_name.clone(), master_name.clone(), slave_name.clone()]);

        vars.push(module_var((&next_name, "d"), (&master_name, "d")));
        vars.push(module_var((&master_name, "q"), (&slave_name, "d")));
        vars.push(module_var(
            (&clock_inverter_name, "clk_n"),
            (&master_name, "en"),
        ));
        vars.push(module_var(
            (&clock_inverter_name, "clk"),
            (&slave_name, "en"),
        ));

        for input in next_bit_inputs(bit) {
            connect_increment_input(&mut vars, output_name, input, &next_name);
        }

        ports.push(GraphModulePort {
            name: bit_name,
            port_type: GraphModulePortType::OutputNet,
            target: GraphModulePortTarget::Module(slave_name, "q".to_owned()),
        });
    }

    ports.insert(
        0,
        GraphModulePort {
            name: clock_name.to_owned(),
            port_type: GraphModulePortType::InputNet,
            target: GraphModulePortTarget::Module(clock_inverter_name, "clk".to_owned()),
        },
    );

    Ok(GraphModuleDesign::with_top_module(
        context,
        GraphModule {
            name: rtl.name.clone(),
            graph: None,
            instances,
            vars,
            ports,
        },
    ))
}

fn ensure_register_increment_expr(output: RtlSignalRef, data: &RtlExpr) -> eyre::Result<()> {
    if !is_increment_by_one(output, data) {
        eyre::bail!("only `reg <= reg + 1` is supported for GraphModule lowering");
    }

    Ok(())
}

fn is_increment_by_one(output: RtlSignalRef, data: &RtlExpr) -> bool {
    let RtlExpr::Add(left, right) = data else {
        return false;
    };

    let signal = match (left.as_ref(), right.as_ref()) {
        (RtlExpr::Signal(signal), RtlExpr::Const { value: 1, .. })
        | (RtlExpr::Const { value: 1, .. }, RtlExpr::Signal(signal)) => *signal,
        _ => return false,
    };
    signal == output
}

fn next_bit_expr(signal: &str, bit: usize) -> String {
    let bit_name = bit_signal_name(signal, bit);
    if bit == 0 {
        return format!("~{bit_name}");
    }
    if bit == 1 {
        return format!("{bit_name}^{}", bit_signal_name(signal, 0));
    }

    format!("{bit_name}^{}", carry_signal_name(bit))
}

fn next_bit_inputs(bit: usize) -> Vec<IncrementInput> {
    match bit {
        0 => vec![IncrementInput::Bit(0)],
        1 => vec![IncrementInput::Bit(1), IncrementInput::Bit(0)],
        _ => vec![IncrementInput::Bit(bit), IncrementInput::Carry(bit)],
    }
}

fn carry_expr(signal: &str, carry_bit: usize) -> String {
    if carry_bit == 2 {
        return format!(
            "{}&{}",
            bit_signal_name(signal, 1),
            bit_signal_name(signal, 0)
        );
    }

    format!(
        "{}&{}",
        bit_signal_name(signal, carry_bit - 1),
        carry_signal_name(carry_bit - 1)
    )
}

fn carry_inputs(carry_bit: usize) -> Vec<IncrementInput> {
    if carry_bit == 2 {
        return vec![IncrementInput::Bit(1), IncrementInput::Bit(0)];
    }

    vec![
        IncrementInput::Bit(carry_bit - 1),
        IncrementInput::Carry(carry_bit - 1),
    ]
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

#[derive(Clone, Copy)]
enum IncrementInput {
    Bit(usize),
    Carry(usize),
}

fn connect_increment_input(
    vars: &mut Vec<GraphModuleVariable>,
    signal: &str,
    input: IncrementInput,
    target_module: &str,
) {
    match input {
        IncrementInput::Bit(bit) => vars.push(module_var(
            (&format!("{}_{}_slave", signal, bit), "q"),
            (target_module, &bit_signal_name(signal, bit)),
        )),
        IncrementInput::Carry(bit) => vars.push(module_var(
            (&carry_module_name(signal, bit), &carry_signal_name(bit)),
            (target_module, &carry_signal_name(bit)),
        )),
    }
}

fn combinational_output_module(name: &str, expr: &str, output: &str) -> eyre::Result<GraphModule> {
    let mut module: GraphModule = LogicGraph::from_stmt(expr, output)?.graph.into();
    module.name = name.to_owned();
    Ok(module)
}

fn not_clock_module(name: &str) -> eyre::Result<GraphModule> {
    combinational_output_module(name, "~clk", "clk_n")
}

fn module_var(source: (&str, &str), target: (&str, &str)) -> GraphModuleVariable {
    GraphModuleVariable {
        var_type: GraphModulePortType::InputNet,
        source: (source.0.to_owned(), source.1.to_owned()),
        target: (target.0.to_owned(), target.1.to_owned()),
    }
}

fn hierarchical_module(
    module: &VerilogModule,
    definitions: &HashMap<&str, &VerilogModule>,
) -> eyre::Result<GraphModule> {
    let endpoints = connection_endpoints(module, definitions)?;
    let top_port_names = declared_names(module, |direction| direction == PortDirection::Input)
        .into_iter()
        .chain(declared_names(module, is_output_direction))
        .collect::<HashSet<_>>();

    let mut ports = Vec::new();
    for name in declared_names(module, |direction| direction == PortDirection::Input) {
        ports.push(GraphModulePort {
            name: name.clone(),
            port_type: GraphModulePortType::InputNet,
            target: target_from_endpoints(endpoints.sinks.get(&name)),
        });
    }
    for name in declared_names(module, is_output_direction) {
        ports.push(GraphModulePort {
            name: name.clone(),
            port_type: GraphModulePortType::OutputNet,
            target: target_from_endpoints(endpoints.sources.get(&name)),
        });
    }

    let mut vars = Vec::new();
    for signal in endpoints.signal_names() {
        if top_port_names.contains(&signal) {
            continue;
        }
        let Some(source) = single_endpoint(endpoints.sources.get(&signal)) else {
            continue;
        };
        for target in endpoints.sinks.get(&signal).into_iter().flatten() {
            vars.push(GraphModuleVariable {
                var_type: GraphModulePortType::InputNet,
                source: source.clone(),
                target: target.clone(),
            });
        }
    }

    Ok(GraphModule {
        name: module.name.clone(),
        graph: None,
        instances: module
            .instances
            .iter()
            .map(|instance| instance.instance_name.clone())
            .collect(),
        vars,
        ports,
    })
}

#[derive(Default)]
struct ConnectionEndpoints {
    sources: HashMap<String, Vec<(String, String)>>,
    sinks: HashMap<String, Vec<(String, String)>>,
}

impl ConnectionEndpoints {
    fn signal_names(&self) -> HashSet<String> {
        self.sources
            .keys()
            .chain(self.sinks.keys())
            .cloned()
            .collect()
    }
}

fn connection_endpoints(
    module: &VerilogModule,
    definitions: &HashMap<&str, &VerilogModule>,
) -> eyre::Result<ConnectionEndpoints> {
    let mut endpoints = ConnectionEndpoints::default();
    for instance in &module.instances {
        let definition = module_definition(definitions, &instance.module_name)?;
        for (port_name, signal_name) in &instance.connections {
            let endpoint = (instance.instance_name.clone(), port_name.clone());
            match port_direction(definition, port_name)? {
                PortDirection::Input => endpoints
                    .sinks
                    .entry(signal_name.clone())
                    .or_default()
                    .push(endpoint),
                PortDirection::Output | PortDirection::OutputReg => endpoints
                    .sources
                    .entry(signal_name.clone())
                    .or_default()
                    .push(endpoint),
                PortDirection::Wire => {}
            }
        }
    }
    Ok(endpoints)
}

fn target_from_endpoints(endpoints: Option<&Vec<(String, String)>>) -> GraphModulePortTarget {
    match endpoints {
        Some(endpoints) if endpoints.len() == 1 => {
            let (module, port) = &endpoints[0];
            GraphModulePortTarget::Module(module.clone(), port.clone())
        }
        Some(endpoints) => GraphModulePortTarget::Wire(endpoints.clone()),
        None => GraphModulePortTarget::Wire(Vec::new()),
    }
}

fn single_endpoint(endpoints: Option<&Vec<(String, String)>>) -> Option<(String, String)> {
    let endpoints = endpoints?;
    (endpoints.len() == 1).then(|| endpoints[0].clone())
}

fn declared_names(
    module: &VerilogModule,
    matches_direction: impl Fn(PortDirection) -> bool,
) -> Vec<String> {
    module
        .declarations
        .iter()
        .filter(|declaration| declaration.direction.is_some_and(&matches_direction))
        .flat_map(|declaration| declaration.names.iter().cloned())
        .collect()
}

fn port_direction(module: &VerilogModule, port_name: &str) -> eyre::Result<PortDirection> {
    module
        .declarations
        .iter()
        .find(|declaration| declaration.names.iter().any(|name| name == port_name))
        .and_then(|declaration| declaration.direction)
        .with_context(|| {
            format!(
                "module `{}` has no declared port `{port_name}`",
                module.name
            )
        })
}

fn graph_module_port_type(direction: PortDirection) -> GraphModulePortType {
    match direction {
        PortDirection::Input => GraphModulePortType::InputNet,
        PortDirection::Output | PortDirection::OutputReg => GraphModulePortType::OutputNet,
        PortDirection::Wire => GraphModulePortType::InputNet,
    }
}

fn is_output_direction(direction: PortDirection) -> bool {
    matches!(direction, PortDirection::Output | PortDirection::OutputReg)
}

#[cfg(test)]
mod tests {
    use super::lower_design_modules;
    use crate::graph::module::GraphModulePortTarget;
    use crate::verilog::parser::parse_modules;

    #[test]
    fn lowers_structural_d_flip_flop_to_graph_module_design() -> eyre::Result<()> {
        let modules = parse_modules(
            r#"
            module not_clk(clk, clk_n);
              input clk;
              output clk_n;
              assign clk_n = ~clk;
            endmodule

            module d_latch(d, en, q);
              input d, en;
              output reg q;
              always @(*) begin
                if (en) begin
                  q <= d;
                end
              end
            endmodule

            module d_flip_flop(d, clk, q);
              input d, clk;
              output q;
              wire clk_n, master_q;
              not_clk inv(.clk(clk), .clk_n(clk_n));
              d_latch master(.d(d), .en(clk_n), .q(master_q));
              d_latch slave(.d(master_q), .en(clk), .q(q));
            endmodule
            "#,
        )?;

        let design = lower_design_modules(&modules)?;
        let top = design.top_module();

        assert_eq!(top.name, "d_flip_flop");
        assert_eq!(top.instances, vec!["inv", "master", "slave"]);
        assert_eq!(top.vars.len(), 2);
        assert!(matches!(
            &top.port_by_name("clk")?.target,
            GraphModulePortTarget::Wire(targets)
                if targets == &vec![
                    ("inv".to_owned(), "clk".to_owned()),
                    ("slave".to_owned(), "en".to_owned())
                ]
        ));
        assert!(matches!(
            &top.port_by_name("q")?.target,
            GraphModulePortTarget::Module(module, port)
                if module == "slave" && port == "q"
        ));

        Ok(())
    }

    #[test]
    fn lowers_register_increment_to_bit_dff_design() -> eyre::Result<()> {
        let modules = parse_modules(
            r#"
            module counter(clk, q);
              input clk;
              output reg [3:0] q;
              always @(posedge clk) begin
                q <= q + 1;
              end
            endmodule
            "#,
        )?;

        let design = lower_design_modules(&modules)?;
        let top = design.top_module();

        assert_eq!(top.name, "counter");
        assert_eq!(top.instances.len(), 15);
        assert!(matches!(
            &top.port_by_name("clk")?.target,
            GraphModulePortTarget::Module(module, port)
                if module == "q_clk_inv" && port == "clk"
        ));
        for bit in 0..4 {
            assert!(matches!(
                &top.port_by_name(&format!("q_{bit}"))?.target,
                GraphModulePortTarget::Module(module, port)
                    if module == &format!("q_{bit}_slave") && port == "q"
            ));
        }

        Ok(())
    }

    #[test]
    fn lowers_scalar_increment_to_dff_design() -> eyre::Result<()> {
        let modules = parse_modules(
            r#"
            module counter(clk, q);
              input clk;
              output reg q;
              always @(posedge clk) begin
                q <= q + 1;
              end
            endmodule
            "#,
        )?;

        let design = lower_design_modules(&modules)?;
        let top = design.top_module();

        assert_eq!(top.name, "counter");
        assert_eq!(
            top.instances,
            vec!["q_clk_inv", "q_next", "q_master", "q_slave"]
        );
        assert!(matches!(
            &top.port_by_name("clk")?.target,
            GraphModulePortTarget::Module(module, port)
                if module == "q_clk_inv" && port == "clk"
        ));
        assert!(matches!(
            &top.port_by_name("q")?.target,
            GraphModulePortTarget::Module(module, port)
                if module == "q_slave" && port == "q"
        ));

        Ok(())
    }
}
