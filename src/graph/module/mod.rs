use itertools::Itertools;

use super::{Graph, GraphNodeId};

pub type GraphModuleId = usize;
pub type GraphModulePortId = usize;

#[derive(Default, PartialEq, Eq, Clone, Copy)]
pub enum GraphModulePortType {
    #[default]
    InputNet,
    OutputNet,
    InputReg,
    OutputReg,
    InOut,
}

#[derive(Default, Clone)]
pub struct GraphModulePort {
    pub id: GraphModulePortId,
    pub port_type: GraphModulePortType,
    pub target: (GraphModuleId, GraphNodeId),
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
pub enum GraphModuleVariableType {
    #[default]
    Wire,
    Reg,
}

#[derive(Default, Clone)]
pub struct GraphModuleVariable {
    pub var_type: GraphModulePortType,
    pub source: GraphModulePortId,
    pub targe: GraphModulePortId,
}

#[derive(Default, Clone)]
enum GraphModuleState {
    #[default]
    UnInitialized,
    Initialized,
}

#[derive(Default, Clone)]
pub struct GraphModule {
    state: GraphModuleState,
    pub id: GraphModuleId,
    pub graph: Option<Graph>,
    pub instances: Vec<Box<GraphModule>>,
    pub vars: Vec<GraphModuleVariable>,
    pub ports: Vec<GraphModulePort>,
}

impl GraphModule {
    pub fn numbering_ports(&mut self, base: usize) -> usize {
        self.ports
            .iter_mut()
            .enumerate()
            .for_each(|(index, port)| port.id = base + index);

        self.state = GraphModuleState::Initialized;

        base + self.ports.len()
    }

    fn check_init(&self) -> eyre::Result<()> {
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
                    target: (0, *input),
                    ..Default::default()
                })
                .chain(value.outputs().iter().map(|output| GraphModulePort {
                    port_type: GraphModulePortType::OutputNet,
                    target: (0, *output),
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
