use std::collections::{HashMap, HashSet};
use std::ops::Index;

use eyre::ContextCompat;
use itertools::Itertools;

use super::{Graph, GraphNodeId};
use crate::graph::GraphNodeKind;

pub mod builder;

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
        let input_ports = value
            .inputs()
            .iter()
            .map(|input| {
                let input_name = value.find_node_by_id(*input).unwrap().kind.as_input();
                GraphModulePort {
                    port_type: GraphModulePortType::InputNet,
                    target: GraphModulePortTarget::Node(input_name.clone()),
                    name: input_name.clone(),
                }
            })
            .collect_vec();
        let output_ports = value
            .outputs()
            .iter()
            .map(|output| {
                let output_name = value.find_node_by_id(*output).unwrap().kind.as_output();
                GraphModulePort {
                    port_type: GraphModulePortType::OutputNet,
                    target: GraphModulePortTarget::Node(output_name.clone()),
                    name: output_name.clone(),
                }
            })
            .collect_vec();

        Self {
            name: "DefaultGraphModule".to_string(),
            ports: input_ports.into_iter().chain(output_ports).collect_vec(),
            graph: Some(value),
            ..Default::default()
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct GraphModuleContext {
    modules: Vec<GraphModule>,
    module_index: HashMap<String, usize>,
}

impl GraphModuleContext {
    pub fn append(&mut self, module: GraphModule) {
        *self.module_index.entry(module.name.clone()).or_default() = self.modules.len();
        self.modules.push(module);
    }

    pub fn extend<I>(&mut self, modules: I)
    where
        I: Iterator<Item = GraphModule>,
    {
        modules.for_each(|module| self.append(module));
    }
}

impl Index<&str> for GraphModuleContext {
    type Output = GraphModule;

    fn index(&self, index: &str) -> &Self::Output {
        &self.modules[self.module_index[index]]
    }
}

#[derive(Clone, Debug)]
pub struct GraphWithSubGraphs(pub Graph, pub Vec<Vec<GraphNodeId>>);

impl From<(&GraphModuleContext, &GraphModule)> for GraphWithSubGraphs {
    fn from(value: (&GraphModuleContext, &GraphModule)) -> Self {
        let context = value.0;
        let module = value.1;

        if let Some(graph) = &module.graph {
            return GraphWithSubGraphs(graph.clone(), Vec::new());
        }

        // (module, (port, port))
        let mut module_port_name: HashMap<String, HashMap<String, String>> = HashMap::new();
        for port in &module.ports {
            match &port.target {
                GraphModulePortTarget::Node(_) => unreachable!(),
                GraphModulePortTarget::Module(instance_name, port_name) => {
                    module_port_name
                        .entry(instance_name.clone())
                        .or_default()
                        .insert(port.name.clone(), port_name.clone());
                }
                GraphModulePortTarget::Wire(wire) => {
                    for (instance_name, port_name) in wire {
                        module_port_name
                            .entry(instance_name.clone())
                            .or_default()
                            .insert(port.name.clone(), port_name.clone());
                    }
                }
            }
        }

        for (index, var) in module.vars.iter().enumerate() {
            module_port_name
                .entry(var.source.0.clone())
                .or_default()
                .insert(var.source.1.clone(), format!("tmp#{index}"));
            module_port_name
                .entry(var.target.0.clone())
                .or_default()
                .insert(var.target.1.clone(), format!("tmp#{index}"));
        }

        let mut graph_ids: Vec<HashSet<GraphNodeId>> = vec![HashSet::new()];
        let mut graph = module
            .instances
            .iter()
            .map(|instance| {
                let graph: GraphWithSubGraphs = (context, &context[instance]).into();
                let mut graph = graph.0;
                graph
                    .nodes
                    .iter_mut()
                    .for_each(|node| match &mut node.kind {
                        GraphNodeKind::Input(name) | GraphNodeKind::Output(name) => {
                            if let Some(port_name) = module_port_name
                                .get_mut(instance)
                                .map(|ports| ports.get_mut(name))
                                .flatten()
                            {
                                *name = port_name.clone();
                            }
                        }
                        _ => (),
                    });
                graph
            })
            .reduce(|mut p, g| {
                graph_ids.push(p.ids());
                p.merge(g);
                p
            })
            .unwrap();
        graph_ids.push(graph.ids());

        let mut removed_id: HashSet<GraphNodeId> = HashSet::new();
        for (index, _) in module.vars.iter().enumerate() {
            let id = graph
                .remove_output(format!("tmp#{index}").as_str())
                .unwrap();
            removed_id.insert(id);
        }

        let subgraph = graph_ids
            .windows(2)
            .map(|w| {
                w[1].difference(&w[0])
                    .into_iter()
                    .filter(|p| !removed_id.contains(p))
                    .map(|p| *p)
                    .collect_vec()
            })
            .collect_vec();

        GraphWithSubGraphs(graph, subgraph)
    }
}
