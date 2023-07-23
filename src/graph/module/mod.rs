use std::collections::HashMap;

use eyre::ContextCompat;
use itertools::Itertools;

use super::Graph;

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum GraphModulePortType {
    #[default]
    InputNet,
    OutputNet,
    InputReg,
    OutputReg,
    InOut,
}

impl GraphModulePortType {
    pub fn is_input(&self) -> bool {
        matches!(
            self,
            GraphModulePortType::InOut
                | GraphModulePortType::InputNet
                | GraphModulePortType::InputReg
        )
    }

    pub fn is_output(&self) -> bool {
        matches!(
            self,
            GraphModulePortType::InOut
                | GraphModulePortType::OutputNet
                | GraphModulePortType::OutputReg
        )
    }
}

#[derive(Clone, Debug)]
pub enum GraphModulePortTarget {
    Node(String),
    // (module name, port name)
    Module(String, String),
    Wire(Vec<(String, String)>),
}

impl Default for GraphModulePortTarget {
    fn default() -> Self {
        Self::Node("".to_string())
    }
}

#[derive(Default, Clone, Debug)]
pub struct GraphModulePort {
    pub name: String,
    pub port_type: GraphModulePortType,
    pub target: GraphModulePortTarget,
}

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum GraphModuleVariableType {
    #[default]
    Wire,
    Reg,
}

#[derive(Default, Clone, Debug)]
pub struct GraphModuleVariable {
    pub var_type: GraphModulePortType,
    // (source module, source port)
    pub source: (String, String),
    // (target module, target port)
    pub target: (String, String),
}

#[derive(Default, Clone, Debug)]
pub struct GraphModule {
    pub name: String,
    pub graph: Option<Graph>,
    pub instances: Vec<String>,
    // inner ports connection info
    pub vars: Vec<GraphModuleVariable>,
    // outer ports
    pub ports: Vec<GraphModulePort>,
}

impl GraphModule {
    pub fn port_by_name(&self, name: &str) -> eyre::Result<&GraphModulePort> {
        // TODO: makes this fast
        self.ports
            .iter()
            .find(|port| port.name == name)
            .context("Port not found!")
    }

    pub fn wrap(&self, name: &str) -> eyre::Result<GraphModule> {
        if self.graph.is_none() {
            eyre::bail!("You can wrap only graph module!");
        }

        Ok(GraphModule {
            name: name.to_string(),
            instances: vec![self.name.clone()],
            ports: self
                .ports
                .iter()
                .map(|port| GraphModulePort {
                    name: port.name.to_string(),
                    port_type: port.port_type,
                    target: GraphModulePortTarget::Module(
                        self.name.to_string(),
                        port.name.to_string(),
                    ),
                })
                .collect(),
            ..Default::default()
        })
    }
}

impl From<&Graph> for GraphModule {
    fn from(value: &Graph) -> Self {
        value.clone().into()
    }
}

impl From<Graph> for GraphModule {
    fn from(value: Graph) -> Self {
        Self {
            name: "DefaultGraphModule".to_string(),
            ports: value
                .inputs()
                .iter()
                .map(|input| {
                    let input_name = value.find_node_by_id(*input).unwrap().kind.as_input();
                    GraphModulePort {
                        port_type: GraphModulePortType::InputNet,
                        target: GraphModulePortTarget::Node(input_name.clone()),
                        name: input_name.clone(),
                        ..Default::default()
                    }
                })
                .chain(value.outputs().iter().map(|output| {
                    let output_name = value
                        .find_node_by_id(*output)
                        .unwrap()
                        .kind
                        .as_output()
                        .clone();
                    GraphModulePort {
                        port_type: GraphModulePortType::OutputNet,
                        target: GraphModulePortTarget::Node(output_name.clone()),
                        name: output_name,
                        ..Default::default()
                    }
                }))
                .collect_vec(),
            graph: Some(value),
            ..Default::default()
        }
    }
}

impl From<&GraphModule> for Graph {
    fn from(value: &GraphModule) -> Self {
        todo!()
    }
}

#[derive(Default, Clone, Debug)]
pub struct GraphModuleContext {
    modules: Vec<GraphModule>,
    module_index: HashMap<String, usize>,
}

impl GraphModuleContext {
    pub fn append(&mut self, module: GraphModule) {
        *self.module_index.entry(module.name.clone()).or_default() = self.modules.len() - 1;
        self.modules.push(module);
    }

    pub fn extend<I>(&mut self, modules: I)
    where
        I: Iterator<Item = GraphModule>,
    {
        modules.for_each(|module| self.append(module));
    }
}
