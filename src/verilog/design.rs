use std::collections::{HashMap, HashSet};

use eyre::ContextCompat;

use crate::graph::module::{
    GraphModule, GraphModuleContext, GraphModuleDesign, GraphModulePort, GraphModulePortTarget,
    GraphModulePortType, GraphModuleVariable,
};
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::sequential::{SequentialPrimitive, SequentialType};
use crate::verilog::ast::{AlwaysSensitivity, AlwaysStmt, PortDirection, VerilogModule};

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
    if let Some(latch) = recognized_d_latch(module) {
        return Ok(d_latch_graph_module(
            instance_name,
            &latch.data,
            &latch.enable,
            &latch.output,
        ));
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

struct RecognizedDLatch {
    data: String,
    enable: String,
    output: String,
}

fn recognized_d_latch(module: &VerilogModule) -> Option<RecognizedDLatch> {
    if !module.assignments.is_empty() || !module.instances.is_empty() {
        return None;
    }
    let [always] = module.always_blocks.as_slice() else {
        return None;
    };
    if always.sensitivity != AlwaysSensitivity::Any {
        return None;
    }
    let AlwaysStmt::If {
        condition,
        then_branch,
    } = &always.body
    else {
        return None;
    };
    let AlwaysStmt::NonBlockingAssign { output, data } = then_branch.as_ref() else {
        return None;
    };

    Some(RecognizedDLatch {
        data: data.clone(),
        enable: condition.clone(),
        output: output.clone(),
    })
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
}
