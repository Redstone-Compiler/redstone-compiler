use std::collections::{HashMap, HashSet};

use eyre::{Context, ContextCompat};
use serde::{Deserialize, Serialize};

pub const ROUTABLE_IR_VERSION: u32 = 1;
pub const ROUTABLE_IR_TARGET: &str = "redstone-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableDesign {
    pub version: u32,
    pub target: String,
    pub top: String,
    pub modules: Vec<RoutableModule>,
    #[serde(skip)]
    pub debug: super::debug::IrDebugInfo,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableModule {
    pub name: String,
    pub ports: Vec<RoutablePort>,
    pub body: RoutableModuleBody,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoutableModuleBody {
    Leaf {
        nodes: Vec<RoutableNode>,
    },
    Composite {
        instances: Vec<RoutableInstance>,
        nets: Vec<RoutableNet>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutablePort {
    pub name: String,
    pub direction: RoutablePortDirection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutablePortDirection {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableInstance {
    pub name: String,
    pub module: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableNet {
    pub name: String,
    pub class: NetClass,
    pub driver: Endpoint,
    pub sinks: Vec<Endpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetClass {
    #[default]
    Data,
    Clock,
    Reset,
    Io,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Endpoint {
    SelfPort { port: String },
    InstancePort { instance: String, port: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableNode {
    pub id: usize,
    pub kind: RoutableNodeKind,
    pub inputs: Vec<usize>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tag: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoutableNodeKind {
    Input {
        name: String,
    },
    Output {
        name: String,
    },
    Not,
    And,
    Or,
    Xor,
    Sequential {
        primitive: RoutableSequentialPrimitive,
        input_ports: Vec<String>,
        output_ports: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutableSequentialPrimitive {
    RsLatch,
    DLatch,
}

impl RoutableDesign {
    pub fn module(&self, name: &str) -> Option<&RoutableModule> {
        self.modules.iter().find(|module| module.name == name)
    }

    pub fn validate(&self) -> eyre::Result<()> {
        if self.version != ROUTABLE_IR_VERSION {
            eyre::bail!(
                "unsupported routable IR version {}; expected {}",
                self.version,
                ROUTABLE_IR_VERSION
            );
        }
        if self.target != ROUTABLE_IR_TARGET {
            eyre::bail!(
                "unsupported routable IR target `{}`; expected `{}`",
                self.target,
                ROUTABLE_IR_TARGET
            );
        }
        ensure_unique(
            self.modules.iter().map(|module| module.name.as_str()),
            "module",
        )?;
        self.module(&self.top)
            .with_context(|| format!("unknown top module `{}`", self.top))?;

        let modules = self
            .modules
            .iter()
            .map(|module| (module.name.as_str(), module))
            .collect::<HashMap<_, _>>();
        for module in &self.modules {
            module
                .validate(&modules)
                .wrap_err_with(|| format!("invalid routable module `{}`", module.name))?;
        }
        self.validate_hierarchy(&modules)?;
        Ok(())
    }

    fn validate_hierarchy(&self, modules: &HashMap<&str, &RoutableModule>) -> eyre::Result<()> {
        fn visit<'a>(
            name: &'a str,
            modules: &HashMap<&'a str, &'a RoutableModule>,
            active: &mut Vec<&'a str>,
            done: &mut HashSet<&'a str>,
        ) -> eyre::Result<()> {
            if done.contains(name) {
                return Ok(());
            }
            if let Some(index) = active.iter().position(|candidate| *candidate == name) {
                let mut cycle = active[index..].to_vec();
                cycle.push(name);
                eyre::bail!("recursive routable hierarchy: {}", cycle.join(" -> "));
            }
            active.push(name);
            let module = modules
                .get(name)
                .copied()
                .with_context(|| format!("unknown module `{name}`"))?;
            if let RoutableModuleBody::Composite { instances, .. } = &module.body {
                for instance in instances {
                    visit(&instance.module, modules, active, done)?;
                }
            }
            active.pop();
            done.insert(name);
            Ok(())
        }

        visit(&self.top, modules, &mut Vec::new(), &mut HashSet::new())
    }
}

impl RoutableModule {
    fn validate(&self, modules: &HashMap<&str, &RoutableModule>) -> eyre::Result<()> {
        if self.name.is_empty() {
            eyre::bail!("module name must not be empty");
        }
        ensure_unique(self.ports.iter().map(|port| port.name.as_str()), "port")?;
        match &self.body {
            RoutableModuleBody::Leaf { nodes } => self.validate_leaf(nodes),
            RoutableModuleBody::Composite { instances, nets } => {
                self.validate_composite(instances, nets, modules)
            }
        }
    }

    fn validate_leaf(&self, nodes: &[RoutableNode]) -> eyre::Result<()> {
        ensure_unique(nodes.iter().map(|node| node.id), "node id")?;
        let ids = nodes.iter().map(|node| node.id).collect::<HashSet<_>>();
        for node in nodes {
            for input in &node.inputs {
                if !ids.contains(input) {
                    eyre::bail!("node {} references missing input node {}", node.id, input);
                }
            }
            validate_node_arity(node)?;
        }

        let input_nodes = nodes
            .iter()
            .filter_map(|node| match &node.kind {
                RoutableNodeKind::Input { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<HashSet<_>>();
        ensure_unique(
            nodes.iter().filter_map(|node| match &node.kind {
                RoutableNodeKind::Input { name } => Some(name.as_str()),
                _ => None,
            }),
            "input node name",
        )?;
        let output_nodes = nodes
            .iter()
            .filter_map(|node| match &node.kind {
                RoutableNodeKind::Output { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<HashSet<_>>();
        ensure_unique(
            nodes.iter().filter_map(|node| match &node.kind {
                RoutableNodeKind::Output { name } => Some(name.as_str()),
                _ => None,
            }),
            "output node name",
        )?;
        for port in &self.ports {
            let exists = match port.direction {
                RoutablePortDirection::Input => input_nodes.contains(port.name.as_str()),
                RoutablePortDirection::Output => output_nodes.contains(port.name.as_str()),
            };
            if !exists {
                eyre::bail!(
                    "{} port `{}` has no matching leaf graph node",
                    direction_name(port.direction),
                    port.name
                );
            }
        }
        for name in input_nodes {
            if !self
                .ports
                .iter()
                .any(|port| port.name == name && port.direction == RoutablePortDirection::Input)
            {
                eyre::bail!("input node `{name}` has no matching module port");
            }
        }
        for name in output_nodes {
            if !self
                .ports
                .iter()
                .any(|port| port.name == name && port.direction == RoutablePortDirection::Output)
            {
                eyre::bail!("output node `{name}` has no matching module port");
            }
        }
        Ok(())
    }

    fn validate_composite(
        &self,
        instances: &[RoutableInstance],
        nets: &[RoutableNet],
        modules: &HashMap<&str, &RoutableModule>,
    ) -> eyre::Result<()> {
        ensure_unique(
            instances.iter().map(|instance| instance.name.as_str()),
            "instance",
        )?;
        ensure_unique(nets.iter().map(|net| net.name.as_str()), "net")?;
        let instance_modules = instances
            .iter()
            .map(|instance| {
                let module = modules
                    .get(instance.module.as_str())
                    .copied()
                    .with_context(|| {
                        format!(
                            "instance `{}` references unknown module `{}`",
                            instance.name, instance.module
                        )
                    })?;
                Ok((instance.name.as_str(), module))
            })
            .collect::<eyre::Result<HashMap<_, _>>>()?;

        let mut driven_ports = HashSet::new();
        let mut consumed_ports = HashSet::new();
        for net in nets {
            if net.name.is_empty() {
                eyre::bail!("net name must not be empty");
            }
            validate_endpoint(&net.driver, true, &self.ports, &instance_modules)
                .wrap_err_with(|| format!("invalid driver of net `{}`", net.name))?;
            if !driven_ports.insert(net.driver.clone()) {
                eyre::bail!("endpoint {} drives more than one net", net.driver.display());
            }
            if net.sinks.is_empty() {
                eyre::bail!("net `{}` has no sinks", net.name);
            }
            let mut local_sinks = HashSet::new();
            for sink in &net.sinks {
                validate_endpoint(sink, false, &self.ports, &instance_modules)
                    .wrap_err_with(|| format!("invalid sink of net `{}`", net.name))?;
                if !local_sinks.insert(sink.clone()) {
                    eyre::bail!("net `{}` repeats sink {}", net.name, sink.display());
                }
                if !consumed_ports.insert(sink.clone()) {
                    eyre::bail!("endpoint {} is driven by more than one net", sink.display());
                }
            }
        }

        for port in &self.ports {
            let endpoint = Endpoint::SelfPort {
                port: port.name.clone(),
            };
            let connected = match port.direction {
                RoutablePortDirection::Input => driven_ports.contains(&endpoint),
                RoutablePortDirection::Output => consumed_ports.contains(&endpoint),
            };
            if !connected {
                eyre::bail!("top-level port `{}` is not connected", port.name);
            }
        }
        for instance in instances {
            let definition = instance_modules[instance.name.as_str()];
            for port in &definition.ports {
                let endpoint = Endpoint::InstancePort {
                    instance: instance.name.clone(),
                    port: port.name.clone(),
                };
                let connected = match port.direction {
                    RoutablePortDirection::Input => consumed_ports.contains(&endpoint),
                    RoutablePortDirection::Output => driven_ports.contains(&endpoint),
                };
                if !connected {
                    eyre::bail!(
                        "instance port `{}.{}` is not connected",
                        instance.name,
                        port.name
                    );
                }
            }
        }
        Ok(())
    }
}

impl Endpoint {
    pub fn display(&self) -> String {
        match self {
            Endpoint::SelfPort { port } => format!("self.{port}"),
            Endpoint::InstancePort { instance, port } => format!("{instance}.{port}"),
        }
    }
}

fn validate_endpoint(
    endpoint: &Endpoint,
    driver: bool,
    ports: &[RoutablePort],
    instances: &HashMap<&str, &RoutableModule>,
) -> eyre::Result<()> {
    let (owner, port) = match endpoint {
        Endpoint::SelfPort { port } => {
            (None, ports.iter().find(|candidate| candidate.name == *port))
        }
        Endpoint::InstancePort { instance, port } => {
            let module = instances
                .get(instance.as_str())
                .copied()
                .with_context(|| format!("unknown instance `{instance}`"))?;
            (
                Some(instance.as_str()),
                module
                    .ports
                    .iter()
                    .find(|candidate| candidate.name == *port),
            )
        }
    };
    let port = port.with_context(|| format!("unknown endpoint `{}`", endpoint.display()))?;
    let expected = match (owner.is_none(), driver) {
        (true, true) | (false, false) => RoutablePortDirection::Input,
        (true, false) | (false, true) => RoutablePortDirection::Output,
    };
    if port.direction != expected {
        eyre::bail!(
            "endpoint `{}` has direction {}, expected {}",
            endpoint.display(),
            direction_name(port.direction),
            direction_name(expected)
        );
    }
    Ok(())
}

fn validate_node_arity(node: &RoutableNode) -> eyre::Result<()> {
    let valid = match &node.kind {
        RoutableNodeKind::Input { .. } => node.inputs.is_empty(),
        RoutableNodeKind::Output { .. } | RoutableNodeKind::Not => node.inputs.len() == 1,
        RoutableNodeKind::And | RoutableNodeKind::Or | RoutableNodeKind::Xor => {
            node.inputs.len() >= 2
        }
        RoutableNodeKind::Sequential { input_ports, .. } => node.inputs.len() == input_ports.len(),
    };
    if !valid {
        eyre::bail!("node {} has invalid input arity", node.id);
    }
    Ok(())
}

fn direction_name(direction: RoutablePortDirection) -> &'static str {
    match direction {
        RoutablePortDirection::Input => "input",
        RoutablePortDirection::Output => "output",
    }
}

fn ensure_unique<T>(values: impl IntoIterator<Item = T>, kind: &str) -> eyre::Result<()>
where
    T: Eq + std::hash::Hash + std::fmt::Debug,
{
    let mut seen = HashSet::new();
    for value in values {
        if seen.contains(&value) {
            eyre::bail!("duplicate {kind} {value:?}");
        }
        seen.insert(value);
    }
    Ok(())
}
