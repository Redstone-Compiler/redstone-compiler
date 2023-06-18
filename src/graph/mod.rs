use std::collections::HashMap;

use crate::{common::block::Block, logic::Logic};

pub type GraphNodeId = usize;

#[derive(Default, Debug, Clone)]
pub enum GraphNodeOption {
    #[default]
    None,
    Input,
    Block {
        inputs: Vec<GraphNodeId>,
        outputs: Vec<GraphNodeId>,
        block: Block,
    },
    Logic {
        inputs: Vec<GraphNodeId>,
        outputs: Vec<GraphNodeId>,
        logic: Logic,
    },
    Output,
}

#[derive(Default, Debug, Clone)]
pub struct GraphNode {
    id: GraphNodeId,
    option: GraphNodeOption,
}

pub struct Graph {
    nodes: Vec<GraphNodeOption>,
    producers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
    consumers: HashMap<GraphNodeId, Vec<GraphNodeId>>,
}
