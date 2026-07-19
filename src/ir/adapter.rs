use std::collections::{BTreeMap, HashSet};

use eyre::ContextCompat;

use super::routable::{
    Endpoint, NetClass, RoutableDesign, RoutableInstance, RoutableModule, RoutableModuleBody,
    RoutableNet, RoutableNode, RoutableNodeKind, RoutablePort, RoutablePortDirection,
    RoutableSequentialPrimitive, ROUTABLE_IR_TARGET, ROUTABLE_IR_VERSION,
};
use crate::graph::module::{
    GraphModule, GraphModuleContext, GraphModuleDesign, GraphModulePort, GraphModulePortTarget,
    GraphModulePortType, GraphModuleVariable,
};
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::logic::{Logic, LogicType};
use crate::sequential::{SequentialPrimitive, SequentialType};

impl RoutableDesign {
    pub fn from_graph_module_design(design: &GraphModuleDesign) -> eyre::Result<Self> {
        let mut converted = design
            .context
            .modules()
            .map(routable_module_from_graph_module)
            .collect::<eyre::Result<Vec<_>>>()?;
        converted.sort_by(|left, right| left.name.cmp(&right.name));

        let top_index = converted
            .iter()
            .position(|module| module.name == design.top)
            .with_context(|| format!("missing top module `{}`", design.top))?;
        let mut top = converted.remove(top_index);
        let mut definitions = Vec::<RoutableModule>::new();
        let mut definition_names = BTreeMap::<String, String>::new();
        for module in converted {
            let canonical = if matches!(module.body, RoutableModuleBody::Leaf { .. }) {
                definitions
                    .iter()
                    .find(|candidate| {
                        candidate.ports == module.ports && candidate.body == module.body
                    })
                    .map(|candidate| candidate.name.clone())
            } else {
                None
            };
            if let Some(canonical) = canonical {
                definition_names.insert(module.name, canonical);
            } else {
                definition_names.insert(module.name.clone(), module.name.clone());
                definitions.push(module);
            }
        }

        for module in definitions.iter_mut().chain(std::iter::once(&mut top)) {
            if let RoutableModuleBody::Composite { instances, .. } = &mut module.body {
                for instance in instances {
                    instance.module = definition_names
                        .get(&instance.module)
                        .cloned()
                        .unwrap_or_else(|| instance.module.clone());
                }
            }
        }
        definitions.push(top);
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        let ir = Self {
            version: ROUTABLE_IR_VERSION,
            target: ROUTABLE_IR_TARGET.to_owned(),
            top: design.top.clone(),
            modules: definitions,
            debug: Default::default(),
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn to_graph_module_design(&self) -> eyre::Result<GraphModuleDesign> {
        self.validate()?;
        let top = self
            .module(&self.top)
            .with_context(|| format!("unknown top routable module `{}`", self.top))?;
        match &top.body {
            RoutableModuleBody::Leaf { .. } => Ok(GraphModuleDesign::with_top_module(
                GraphModuleContext::default(),
                graph_module_from_leaf(top, &top.name)?,
            )),
            RoutableModuleBody::Composite { instances, nets } => {
                let mut context = GraphModuleContext::default();
                for instance in instances {
                    let definition = self.module(&instance.module).with_context(|| {
                        format!(
                            "instance `{}` references missing module `{}`",
                            instance.name, instance.module
                        )
                    })?;
                    if !matches!(definition.body, RoutableModuleBody::Leaf { .. }) {
                        eyre::bail!(
                            "legacy PnR adapter supports only leaf children; instance `{}` uses composite module `{}`",
                            instance.name,
                            instance.module
                        );
                    }
                    context.append(graph_module_from_leaf(definition, &instance.name)?);
                }
                let top_module = graph_module_from_composite(top, instances, nets)?;
                Ok(GraphModuleDesign::with_top_module(context, top_module))
            }
        }
    }
}

fn routable_module_from_graph_module(module: &GraphModule) -> eyre::Result<RoutableModule> {
    let ports = module
        .ports
        .iter()
        .map(|port| {
            Ok(RoutablePort {
                name: port.name.clone(),
                direction: routable_port_direction(port.port_type)?,
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let body = if let Some(graph) = &module.graph {
        RoutableModuleBody::Leaf {
            nodes: graph
                .nodes
                .iter()
                .map(|node| {
                    Ok(RoutableNode {
                        id: node.id,
                        kind: routable_node_kind(&node.kind)?,
                        inputs: node.inputs.clone(),
                        tag: node.tag.clone(),
                    })
                })
                .collect::<eyre::Result<Vec<_>>>()?,
        }
    } else {
        let instances = module
            .instances
            .iter()
            .map(|name| RoutableInstance {
                name: name.clone(),
                module: name.clone(),
                origin: None,
            })
            .collect::<Vec<_>>();
        let nets = routable_nets_from_graph_module(module)?;
        RoutableModuleBody::Composite { instances, nets }
    };
    Ok(RoutableModule {
        name: module.name.clone(),
        ports,
        body,
    })
}

fn routable_nets_from_graph_module(module: &GraphModule) -> eyre::Result<Vec<RoutableNet>> {
    let mut connections = BTreeMap::<Endpoint, Vec<Endpoint>>::new();
    for variable in &module.vars {
        connections
            .entry(Endpoint::InstancePort {
                instance: variable.source.0.clone(),
                port: variable.source.1.clone(),
            })
            .or_default()
            .push(Endpoint::InstancePort {
                instance: variable.target.0.clone(),
                port: variable.target.1.clone(),
            });
    }

    for port in &module.ports {
        let direction = routable_port_direction(port.port_type)?;
        match (direction, &port.target) {
            (RoutablePortDirection::Input, GraphModulePortTarget::Module(instance, target)) => {
                connections
                    .entry(Endpoint::SelfPort {
                        port: port.name.clone(),
                    })
                    .or_default()
                    .push(Endpoint::InstancePort {
                        instance: instance.clone(),
                        port: target.clone(),
                    });
            }
            (RoutablePortDirection::Input, GraphModulePortTarget::Wire(targets)) => {
                connections
                    .entry(Endpoint::SelfPort {
                        port: port.name.clone(),
                    })
                    .or_default()
                    .extend(
                        targets
                            .iter()
                            .map(|(instance, target)| Endpoint::InstancePort {
                                instance: instance.clone(),
                                port: target.clone(),
                            }),
                    );
            }
            (RoutablePortDirection::Output, GraphModulePortTarget::Module(instance, source)) => {
                connections
                    .entry(Endpoint::InstancePort {
                        instance: instance.clone(),
                        port: source.clone(),
                    })
                    .or_default()
                    .push(Endpoint::SelfPort {
                        port: port.name.clone(),
                    });
            }
            (_, GraphModulePortTarget::Node(_)) => {
                eyre::bail!(
                    "composite GraphModule `{}` port `{}` targets a leaf node",
                    module.name,
                    port.name
                );
            }
            (RoutablePortDirection::Output, GraphModulePortTarget::Wire(_)) => {
                eyre::bail!(
                    "output port `{}` uses ambiguous GraphModule wire target",
                    port.name
                );
            }
        }
    }

    let mut used_names = HashSet::new();
    Ok(connections
        .into_iter()
        .map(|(driver, mut sinks)| {
            sinks.sort();
            sinks.dedup();
            let base_name = preferred_net_name(&driver, &sinks);
            let name = unique_name(base_name, &mut used_names);
            RoutableNet {
                class: classify_net(&name, &driver, &sinks),
                name,
                driver,
                sinks,
                origin: None,
            }
        })
        .collect())
}

fn graph_module_from_leaf(
    module: &RoutableModule,
    instance_name: &str,
) -> eyre::Result<GraphModule> {
    let graph = graph_from_routable_leaf(module)?;

    Ok(GraphModule {
        name: instance_name.to_owned(),
        graph: Some(graph),
        instances: Vec::new(),
        vars: Vec::new(),
        ports: module
            .ports
            .iter()
            .map(|port| GraphModulePort {
                name: port.name.clone(),
                port_type: graph_port_type(port.direction),
                target: GraphModulePortTarget::Node(port.name.clone()),
            })
            .collect(),
    })
}

/// Lowers a Routable leaf to the logic graph consumed by the local placer.
///
/// This deliberately stops below `GraphModule`: Routable ports and hierarchy
/// remain the source of truth while the node graph is adapted only at the
/// local-placement boundary.
pub(crate) fn graph_from_routable_leaf(module: &RoutableModule) -> eyre::Result<Graph> {
    let RoutableModuleBody::Leaf { nodes } = &module.body else {
        eyre::bail!("module `{}` is not a leaf", module.name);
    };
    let graph_nodes = nodes
        .iter()
        .map(|node| {
            Ok((
                node.id,
                GraphNode {
                    kind: graph_node_kind(&node.kind)?,
                    inputs: node.inputs.clone(),
                    tag: node.tag.clone(),
                    ..Default::default()
                },
            ))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let mut graph = Graph::from_nodes_with_ids(graph_nodes);
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify()?;
    Ok(graph)
}

fn graph_module_from_composite(
    module: &RoutableModule,
    instances: &[RoutableInstance],
    nets: &[RoutableNet],
) -> eyre::Result<GraphModule> {
    let instance_names = instances
        .iter()
        .map(|instance| instance.name.clone())
        .collect::<Vec<_>>();
    let mut vars = Vec::new();
    for net in nets {
        let Endpoint::InstancePort {
            instance: source_instance,
            port: source_port,
        } = &net.driver
        else {
            continue;
        };
        for sink in &net.sinks {
            if let Endpoint::InstancePort {
                instance: target_instance,
                port: target_port,
            } = sink
            {
                vars.push(GraphModuleVariable {
                    var_type: GraphModulePortType::InputNet,
                    source: (source_instance.clone(), source_port.clone()),
                    target: (target_instance.clone(), target_port.clone()),
                });
            }
        }
    }

    let ports = module
        .ports
        .iter()
        .map(|port| {
            let endpoint = Endpoint::SelfPort {
                port: port.name.clone(),
            };
            let target = match port.direction {
                RoutablePortDirection::Input => {
                    let net = nets
                        .iter()
                        .find(|net| net.driver == endpoint)
                        .with_context(|| format!("missing input net for port `{}`", port.name))?;
                    let targets = net
                        .sinks
                        .iter()
                        .filter_map(|sink| match sink {
                            Endpoint::InstancePort { instance, port } => {
                                Some((instance.clone(), port.clone()))
                            }
                            Endpoint::SelfPort { .. } => None,
                        })
                        .collect::<Vec<_>>();
                    match targets.as_slice() {
                        [] => eyre::bail!("input port `{}` has no instance sinks", port.name),
                        [(instance, target)] => {
                            GraphModulePortTarget::Module(instance.clone(), target.clone())
                        }
                        _ => GraphModulePortTarget::Wire(targets),
                    }
                }
                RoutablePortDirection::Output => {
                    let net = nets
                        .iter()
                        .find(|net| net.sinks.contains(&endpoint))
                        .with_context(|| format!("missing output net for port `{}`", port.name))?;
                    let Endpoint::InstancePort { instance, port } = &net.driver else {
                        eyre::bail!("output port `{}` is not driven by an instance", port.name);
                    };
                    GraphModulePortTarget::Module(instance.clone(), port.clone())
                }
            };
            Ok(GraphModulePort {
                name: port.name.clone(),
                port_type: graph_port_type(port.direction),
                target,
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    Ok(GraphModule {
        name: module.name.clone(),
        graph: None,
        instances: instance_names,
        vars,
        ports,
    })
}

fn routable_node_kind(kind: &GraphNodeKind) -> eyre::Result<RoutableNodeKind> {
    match kind {
        GraphNodeKind::Input(name) => Ok(RoutableNodeKind::Input { name: name.clone() }),
        GraphNodeKind::Output(name) => Ok(RoutableNodeKind::Output { name: name.clone() }),
        GraphNodeKind::Logic(logic) => Ok(match logic.logic_type {
            LogicType::Not => RoutableNodeKind::Not,
            LogicType::And => RoutableNodeKind::And,
            LogicType::Or => RoutableNodeKind::Or,
            LogicType::Xor => RoutableNodeKind::Xor,
        }),
        GraphNodeKind::Sequential(sequential) => Ok(RoutableNodeKind::Sequential {
            primitive: match sequential.sequential_type {
                SequentialType::RsLatch => RoutableSequentialPrimitive::RsLatch,
                SequentialType::DLatch => RoutableSequentialPrimitive::DLatch,
            },
            input_ports: sequential.input_ports.clone(),
            output_ports: sequential.output_ports.clone(),
        }),
        GraphNodeKind::None => eyre::bail!("routable leaf contains unresolved node"),
        GraphNodeKind::Block(_) => eyre::bail!("routable leaf contains physical block node"),
        GraphNodeKind::Clustered(_) => eyre::bail!("routable leaf contains clustered node"),
    }
}

fn graph_node_kind(kind: &RoutableNodeKind) -> eyre::Result<GraphNodeKind> {
    Ok(match kind {
        RoutableNodeKind::Input { name } => GraphNodeKind::Input(name.clone()),
        RoutableNodeKind::Output { name } => GraphNodeKind::Output(name.clone()),
        RoutableNodeKind::Not => GraphNodeKind::Logic(Logic {
            logic_type: LogicType::Not,
        }),
        RoutableNodeKind::And => GraphNodeKind::Logic(Logic {
            logic_type: LogicType::And,
        }),
        RoutableNodeKind::Or => GraphNodeKind::Logic(Logic {
            logic_type: LogicType::Or,
        }),
        RoutableNodeKind::Xor => GraphNodeKind::Logic(Logic {
            logic_type: LogicType::Xor,
        }),
        RoutableNodeKind::Sequential {
            primitive,
            input_ports,
            output_ports,
        } => GraphNodeKind::Sequential(SequentialPrimitive::new(
            match primitive {
                RoutableSequentialPrimitive::RsLatch => SequentialType::RsLatch,
                RoutableSequentialPrimitive::DLatch => SequentialType::DLatch,
            },
            input_ports.clone(),
            output_ports.clone(),
        )),
    })
}

fn routable_port_direction(port_type: GraphModulePortType) -> eyre::Result<RoutablePortDirection> {
    if port_type.is_input() && !port_type.is_output() {
        return Ok(RoutablePortDirection::Input);
    }
    if port_type.is_output() && !port_type.is_input() {
        return Ok(RoutablePortDirection::Output);
    }
    eyre::bail!("inout GraphModule ports are not supported by routable RCIR")
}

fn graph_port_type(direction: RoutablePortDirection) -> GraphModulePortType {
    match direction {
        RoutablePortDirection::Input => GraphModulePortType::InputNet,
        RoutablePortDirection::Output => GraphModulePortType::OutputNet,
    }
}

fn preferred_net_name(driver: &Endpoint, sinks: &[Endpoint]) -> String {
    if let Endpoint::SelfPort { port } = driver {
        return port.clone();
    }
    if let Some(Endpoint::SelfPort { port }) = sinks
        .iter()
        .find(|endpoint| matches!(endpoint, Endpoint::SelfPort { .. }))
    {
        return port.clone();
    }
    match driver {
        Endpoint::SelfPort { port } => port.clone(),
        Endpoint::InstancePort { instance, port } => format!("{instance}_{port}"),
    }
}

fn unique_name(base: String, used: &mut HashSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    for suffix in 2.. {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!()
}

fn classify_net(name: &str, driver: &Endpoint, sinks: &[Endpoint]) -> NetClass {
    let lower = name.to_ascii_lowercase();
    if lower.contains("clk") || lower.contains("clock") {
        NetClass::Clock
    } else if lower.contains("rst") || lower.contains("reset") {
        NetClass::Reset
    } else if matches!(driver, Endpoint::SelfPort { .. })
        || sinks
            .iter()
            .any(|sink| matches!(sink, Endpoint::SelfPort { .. }))
    {
        NetClass::Io
    } else {
        NetClass::Data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_design_round_trips_through_routable_ir() -> eyre::Result<()> {
        let source = r#"
            module counter(clk, q);
              input clk;
              output reg [1:0] q;
              always @(posedge clk) begin
                q <= q + 1;
              end
            endmodule
        "#;
        let modules = crate::verilog::parser::parse_modules(source)?;
        let design = crate::verilog::design::lower_design_modules(&modules)?;
        let ir = RoutableDesign::from_graph_module_design(&design)?;
        let text = ir.to_string();
        let parsed: RoutableDesign = text.parse()?;
        let restored = parsed.to_graph_module_design()?;

        assert_eq!(parsed.top, "counter");
        assert!(parsed.modules.len() < design.context.modules().count());
        assert_eq!(
            restored.top_module().instances.len(),
            design.top_module().instances.len()
        );
        assert_eq!(
            restored.top_module().ports.len(),
            design.top_module().ports.len()
        );
        assert!(text.contains("instance \"q_0_master\""));
        assert!(text.contains("class clock"));
        let top = parsed.module("counter").unwrap();
        let RoutableModuleBody::Composite { instances, .. } = &top.body else {
            panic!("counter should be composite");
        };
        let latch_definitions =
            ["q_0_master", "q_0_slave", "q_1_master", "q_1_slave"].map(|name| {
                instances
                    .iter()
                    .find(|instance| instance.name == name)
                    .unwrap()
                    .module
                    .as_str()
            });
        assert!(latch_definitions
            .iter()
            .all(|definition| *definition == latch_definitions[0]));
        Ok(())
    }
}
