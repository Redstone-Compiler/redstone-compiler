use std::collections::{BTreeMap, HashMap, HashSet};

use eyre::ContextCompat;
use serde::{Deserialize, Serialize};

use crate::graph::module::{
    GraphModule, GraphModuleContext, GraphModulePortTarget, GraphModulePortType,
};
use crate::ir::{Endpoint, NetClass, RoutableDesign, RoutableModuleBody, RoutablePortDirection};

macro_rules! numeric_id {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub usize);
    };
}

numeric_id!(DefinitionId);
numeric_id!(InstanceId);
numeric_id!(PortId);
numeric_id!(NetId);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DefinitionKey(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InstanceKey(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NetKey(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPnrTopology {
    pub top: DefinitionId,
    pub definitions: Vec<ResolvedDefinition>,
    pub ports: Vec<ResolvedPort>,
    pub instances: Vec<ResolvedInstance>,
    pub nets: Vec<ResolvedNet>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDefinition {
    pub id: DefinitionId,
    pub key: DefinitionKey,
    pub display_name: String,
    pub ports: Vec<PortId>,
    pub is_leaf: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPort {
    pub id: PortId,
    pub definition: DefinitionId,
    pub name: String,
    pub direction: RoutablePortDirection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedInstance {
    pub id: InstanceId,
    pub key: InstanceKey,
    pub display_name: String,
    pub definition: DefinitionId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedNet {
    pub id: NetId,
    pub key: NetKey,
    pub display_name: String,
    pub class: NetClass,
    pub driver: ResolvedEndpoint,
    pub sinks: Vec<ResolvedEndpoint>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedEndpoint {
    TopPort { port: PortId },
    InstancePort { instance: InstanceId, port: PortId },
}

impl ResolvedPnrTopology {
    pub fn definition(&self, id: DefinitionId) -> Option<&ResolvedDefinition> {
        self.definitions
            .get(id.0)
            .filter(|definition| definition.id == id)
    }

    pub fn port(&self, id: PortId) -> Option<&ResolvedPort> {
        self.ports.get(id.0).filter(|port| port.id == id)
    }

    pub fn instance_by_name(&self, name: &str) -> Option<&ResolvedInstance> {
        self.instances
            .iter()
            .find(|instance| instance.display_name == name)
    }

    pub fn endpoint_label(&self, endpoint: &ResolvedEndpoint) -> Option<String> {
        match endpoint {
            ResolvedEndpoint::TopPort { port } => Some(self.port(*port)?.name.clone()),
            ResolvedEndpoint::InstancePort { instance, port } => Some(format!(
                "{}.{}",
                self.instances.get(instance.0)?.display_name,
                self.port(*port)?.name
            )),
        }
    }

    pub fn net_by_driver_label(&self, label: &str) -> Option<&ResolvedNet> {
        self.nets
            .iter()
            .find(|net| self.endpoint_label(&net.driver).as_deref() == Some(label))
    }

    pub fn sink_by_label<'a>(
        &'a self,
        net: &'a ResolvedNet,
        label: &str,
    ) -> Option<&'a ResolvedEndpoint> {
        net.sinks
            .iter()
            .find(|sink| self.endpoint_label(sink).as_deref() == Some(label))
    }

    pub fn from_routable(design: &RoutableDesign) -> eyre::Result<Self> {
        design.validate()?;

        let mut modules = design.modules.iter().collect::<Vec<_>>();
        modules.sort_by(|left, right| left.name.cmp(&right.name));
        let definition_ids = modules
            .iter()
            .enumerate()
            .map(|(index, module)| (module.name.as_str(), DefinitionId(index)))
            .collect::<HashMap<_, _>>();
        let top = *definition_ids
            .get(design.top.as_str())
            .with_context(|| format!("unknown Routable top module `{}`", design.top))?;

        let mut ports = Vec::new();
        let mut port_ids = HashMap::<(DefinitionId, &str), PortId>::new();
        let mut definitions = Vec::with_capacity(modules.len());
        for module in &modules {
            let definition = definition_ids[module.name.as_str()];
            let mut module_ports = module.ports.iter().collect::<Vec<_>>();
            module_ports.sort_by(|left, right| left.name.cmp(&right.name));
            let mut definition_ports = Vec::with_capacity(module_ports.len());
            for port in module_ports {
                let id = PortId(ports.len());
                ports.push(ResolvedPort {
                    id,
                    definition,
                    name: port.name.clone(),
                    direction: port.direction,
                });
                port_ids.insert((definition, port.name.as_str()), id);
                definition_ports.push(id);
            }
            definitions.push(ResolvedDefinition {
                id: definition,
                key: DefinitionKey(module.name.clone()),
                display_name: module.name.clone(),
                ports: definition_ports,
                is_leaf: matches!(module.body, RoutableModuleBody::Leaf { .. }),
            });
        }

        let top_module = design
            .module(&design.top)
            .with_context(|| format!("unknown Routable top module `{}`", design.top))?;
        let RoutableModuleBody::Composite {
            instances: source_instances,
            nets: source_nets,
        } = &top_module.body
        else {
            return Ok(Self {
                top,
                definitions,
                ports,
                instances: Vec::new(),
                nets: Vec::new(),
            });
        };

        let mut sorted_instances = source_instances.iter().collect::<Vec<_>>();
        sorted_instances.sort_by(|left, right| left.name.cmp(&right.name));
        let mut instance_ids = HashMap::<&str, InstanceId>::new();
        let mut instances = Vec::with_capacity(sorted_instances.len());
        for instance in sorted_instances {
            let definition = *definition_ids
                .get(instance.module.as_str())
                .with_context(|| {
                    format!(
                        "instance `{}` references unknown definition `{}`",
                        instance.name, instance.module
                    )
                })?;
            if !definitions[definition.0].is_leaf {
                eyre::bail!(
                    "current global PnR supports only leaf children; instance `{}` uses composite definition `{}`",
                    instance.name,
                    instance.module
                );
            }
            let id = InstanceId(instances.len());
            instance_ids.insert(instance.name.as_str(), id);
            instances.push(ResolvedInstance {
                id,
                key: InstanceKey(format!("{}/{}", design.top, instance.name)),
                display_name: instance.name.clone(),
                definition,
            });
        }

        let mut sorted_nets = source_nets.iter().collect::<Vec<_>>();
        sorted_nets.sort_by(|left, right| left.name.cmp(&right.name));
        let nets = sorted_nets
            .into_iter()
            .enumerate()
            .map(|(index, net)| {
                let mut sinks = net
                    .sinks
                    .iter()
                    .map(|endpoint| {
                        resolve_endpoint(endpoint, top, &instances, &instance_ids, &port_ids)
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;
                sinks.sort();
                Ok(ResolvedNet {
                    id: NetId(index),
                    key: NetKey(format!("{}/net/{}", design.top, net.name)),
                    display_name: net.name.clone(),
                    class: net.class,
                    driver: resolve_endpoint(
                        &net.driver,
                        top,
                        &instances,
                        &instance_ids,
                        &port_ids,
                    )?,
                    sinks,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        Ok(Self {
            top,
            definitions,
            ports,
            instances,
            nets,
        })
    }

    /// Resolve the legacy graph shape without strengthening its historical
    /// validation rules. Unused child ports remain legal on this compatibility
    /// path, while native Routable input continues to use strict validation.
    pub fn from_legacy_graph_module(
        context: &GraphModuleContext,
        top_module: &GraphModule,
    ) -> eyre::Result<Self> {
        let mut module_names = top_module.instances.clone();
        module_names.push(top_module.name.clone());
        module_names.sort();
        module_names.dedup();
        let definition_ids = module_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), DefinitionId(index)))
            .collect::<HashMap<_, _>>();
        let top = definition_ids[top_module.name.as_str()];

        let mut ports = Vec::new();
        let mut port_ids = HashMap::<(DefinitionId, &str), PortId>::new();
        let mut definitions = Vec::new();
        for module_name in &module_names {
            let module = if module_name == &top_module.name {
                top_module
            } else {
                context
                    .get(module_name)
                    .with_context(|| format!("missing legacy child module `{module_name}`"))?
            };
            let definition = definition_ids[module_name.as_str()];
            let mut module_ports = module.ports.iter().collect::<Vec<_>>();
            module_ports.sort_by(|left, right| left.name.cmp(&right.name));
            let mut definition_ports = Vec::new();
            for port in module_ports {
                let id = PortId(ports.len());
                ports.push(ResolvedPort {
                    id,
                    definition,
                    name: port.name.clone(),
                    direction: legacy_port_direction(port.port_type)?,
                });
                port_ids.insert((definition, port.name.as_str()), id);
                definition_ports.push(id);
            }
            definitions.push(ResolvedDefinition {
                id: definition,
                key: DefinitionKey(module.name.clone()),
                display_name: module.name.clone(),
                ports: definition_ports,
                is_leaf: module.graph.is_some(),
            });
        }

        let mut instance_names = top_module.instances.iter().collect::<Vec<_>>();
        instance_names.sort();
        let mut instance_ids = HashMap::<&str, InstanceId>::new();
        let mut instances = Vec::new();
        for name in instance_names {
            let definition = definition_ids[name.as_str()];
            let id = InstanceId(instances.len());
            instance_ids.insert(name.as_str(), id);
            instances.push(ResolvedInstance {
                id,
                key: InstanceKey(format!("{}/{}", top_module.name, name)),
                display_name: name.clone(),
                definition,
            });
        }

        let instance_endpoint = |instance: &str, port: &str| -> eyre::Result<ResolvedEndpoint> {
            let instance_id = *instance_ids
                .get(instance)
                .with_context(|| format!("unknown legacy instance `{instance}`"))?;
            let definition = instances[instance_id.0].definition;
            Ok(ResolvedEndpoint::InstancePort {
                instance: instance_id,
                port: *port_ids
                    .get(&(definition, port))
                    .with_context(|| format!("unknown legacy port `{instance}.{port}`"))?,
            })
        };
        let top_endpoint = |port: &str| -> eyre::Result<ResolvedEndpoint> {
            Ok(ResolvedEndpoint::TopPort {
                port: *port_ids
                    .get(&(top, port))
                    .with_context(|| format!("unknown legacy top port `{port}`"))?,
            })
        };

        let mut connections = BTreeMap::<ResolvedEndpoint, Vec<ResolvedEndpoint>>::new();
        for variable in &top_module.vars {
            connections
                .entry(instance_endpoint(&variable.source.0, &variable.source.1)?)
                .or_default()
                .push(instance_endpoint(&variable.target.0, &variable.target.1)?);
        }
        for port in &top_module.ports {
            let direction = legacy_port_direction(port.port_type)?;
            match (direction, &port.target) {
                (RoutablePortDirection::Input, GraphModulePortTarget::Module(instance, target)) => {
                    connections
                        .entry(top_endpoint(&port.name)?)
                        .or_default()
                        .push(instance_endpoint(instance, target)?);
                }
                (RoutablePortDirection::Input, GraphModulePortTarget::Wire(targets)) => {
                    let source = top_endpoint(&port.name)?;
                    for (instance, target) in targets {
                        connections
                            .entry(source.clone())
                            .or_default()
                            .push(instance_endpoint(instance, target)?);
                    }
                }
                (
                    RoutablePortDirection::Output,
                    GraphModulePortTarget::Module(instance, source),
                ) => {
                    connections
                        .entry(instance_endpoint(instance, source)?)
                        .or_default()
                        .push(top_endpoint(&port.name)?);
                }
                (_, GraphModulePortTarget::Node(_)) => {}
                (RoutablePortDirection::Output, GraphModulePortTarget::Wire(_)) => {
                    eyre::bail!(
                        "legacy output port `{}` has an ambiguous wire target",
                        port.name
                    )
                }
            }
        }

        let port_names = ports
            .iter()
            .map(|port| (port.id, port.name.as_str()))
            .collect::<HashMap<_, _>>();
        let instance_names = instances
            .iter()
            .map(|instance| (instance.id, instance.display_name.as_str()))
            .collect::<HashMap<_, _>>();
        let mut used_names = HashSet::new();
        let nets = connections
            .into_iter()
            .enumerate()
            .map(|(index, (driver, mut sinks))| {
                sinks.sort();
                sinks.dedup();
                let base_name = endpoint_name(&driver, &port_names, &instance_names);
                let display_name = unique_net_name(base_name, &mut used_names);
                ResolvedNet {
                    id: NetId(index),
                    key: NetKey(format!("{}/net/{display_name}", top_module.name)),
                    class: classify_legacy_net(&display_name, &driver),
                    display_name,
                    driver,
                    sinks,
                }
            })
            .collect();

        Ok(Self {
            top,
            definitions,
            ports,
            instances,
            nets,
        })
    }
}

fn legacy_port_direction(port_type: GraphModulePortType) -> eyre::Result<RoutablePortDirection> {
    match (port_type.is_input(), port_type.is_output()) {
        (true, false) => Ok(RoutablePortDirection::Input),
        (false, true) => Ok(RoutablePortDirection::Output),
        _ => eyre::bail!("legacy inout or directionless ports are not supported"),
    }
}

fn endpoint_name(
    endpoint: &ResolvedEndpoint,
    ports: &HashMap<PortId, &str>,
    instances: &HashMap<InstanceId, &str>,
) -> String {
    match endpoint {
        ResolvedEndpoint::TopPort { port } => ports[port].to_owned(),
        ResolvedEndpoint::InstancePort { instance, port } => {
            format!("{}_{}", instances[instance], ports[port])
        }
    }
}

fn unique_net_name(base: String, used: &mut HashSet<String>) -> String {
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

fn classify_legacy_net(name: &str, driver: &ResolvedEndpoint) -> NetClass {
    let lower = name.to_ascii_lowercase();
    if lower.contains("clk") || lower.contains("clock") {
        NetClass::Clock
    } else if lower.contains("rst") || lower.contains("reset") {
        NetClass::Reset
    } else if matches!(driver, ResolvedEndpoint::TopPort { .. }) {
        NetClass::Io
    } else {
        NetClass::Data
    }
}

fn resolve_endpoint(
    endpoint: &Endpoint,
    top: DefinitionId,
    instances: &[ResolvedInstance],
    instance_ids: &HashMap<&str, InstanceId>,
    port_ids: &HashMap<(DefinitionId, &str), PortId>,
) -> eyre::Result<ResolvedEndpoint> {
    let resolved = match endpoint {
        Endpoint::SelfPort { port } => ResolvedEndpoint::TopPort {
            port: *port_ids
                .get(&(top, port.as_str()))
                .with_context(|| format!("unknown top port `{port}`"))?,
        },
        Endpoint::InstancePort { instance, port } => {
            let instance_id = *instance_ids
                .get(instance.as_str())
                .with_context(|| format!("unknown instance `{instance}`"))?;
            let definition = instances
                .get(instance_id.0)
                .with_context(|| format!("missing resolved instance {instance_id:?}"))?
                .definition;
            ResolvedEndpoint::InstancePort {
                instance: instance_id,
                port: *port_ids
                    .get(&(definition, port.as_str()))
                    .with_context(|| format!("unknown port `{port}` on instance `{instance}`"))?,
            }
        }
    };
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::LogicalDesign;

    #[test]
    fn counter_topology_has_stable_typed_instance_and_net_ids() -> eyre::Result<()> {
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
        let topology = ResolvedPnrTopology::from_routable(&logical.lower_to_routable()?)?;

        assert_eq!(topology.instances.len(), 8);
        assert_eq!(topology.nets.len(), 9);
        assert!(topology
            .instances
            .iter()
            .any(|instance| instance.key.0 == "counter/q_0_master"));
        assert!(topology.nets.iter().all(|net| !net.sinks.is_empty()));
        assert!(topology.nets.iter().any(|net| net.sinks.len() > 1));
        assert_eq!(
            topology,
            ResolvedPnrTopology::from_routable(&logical.lower_to_routable()?)?
        );
        Ok(())
    }
}
