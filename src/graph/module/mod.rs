use itertools::Itertools;

use super::{Graph, GraphNodeId};

pub type GraphModuleId = usize;
pub type GraphModulePortId = usize;

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum GraphModulePortType {
    #[default]
    InputNet,
    OutputNet,
    InputReg,
    OutputReg,
    InOut,
}

#[derive(Clone, Debug)]
pub enum GraphModulePortTarget {
    Node(GraphNodeId),
    Module(GraphModuleId, GraphModulePortId),
}

impl Default for GraphModulePortTarget {
    fn default() -> Self {
        Self::Node(0)
    }
}

#[derive(Default, Clone, Debug)]
pub struct GraphModulePort {
    pub id: GraphModulePortId,
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
    pub source: GraphModulePortId,
    pub targe: GraphModulePortId,
}

#[derive(Default, Clone, Debug)]
enum GraphModuleState {
    #[default]
    UnInitialized,
    Initialized,
}

#[derive(Default, Clone, Debug)]
pub struct GraphModule {
    state: GraphModuleState,
    pub id: GraphModuleId,
    pub graph: Option<Graph>,
    pub instances: Vec<Box<GraphModule>>,
    pub vars: Vec<GraphModuleVariable>,
    pub ports: Vec<GraphModulePort>,
}

impl GraphModule {
    pub fn from_instances(instances: Vec<Box<GraphModule>>) -> Self {
        Self {
            instances,
            ..Default::default()
        }
    }

    pub fn numbering_ports(&mut self, base: usize) -> usize {
        self.ports
            .iter_mut()
            .enumerate()
            .for_each(|(index, port)| port.id = base + index);

        self.state = GraphModuleState::Initialized;

        base + self.ports.len()
    }

    pub fn check_init(&self) -> eyre::Result<()> {
        match self.state {
            GraphModuleState::UnInitialized => {
                eyre::bail!("You cannot use unitialized graph module!")
            }
            GraphModuleState::Initialized => Ok(()),
        }
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
            id: 0,
            ports: value
                .inputs()
                .iter()
                .map(|input| GraphModulePort {
                    port_type: GraphModulePortType::InputNet,
                    target: GraphModulePortTarget::Node(*input),
                    ..Default::default()
                })
                .chain(value.outputs().iter().map(|output| GraphModulePort {
                    port_type: GraphModulePortType::OutputNet,
                    target: GraphModulePortTarget::Node(*output),
                    ..Default::default()
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
