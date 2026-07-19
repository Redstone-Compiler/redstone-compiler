use super::{RoutableModule, RoutableModuleBody, RoutableNodeKind, RoutableSequentialPrimitive};
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::logic::{Logic, LogicType};
use crate::sequential::{SequentialPrimitive, SequentialType};

/// Adapts a typed Routable leaf to the node graph consumed by local placement.
/// Hierarchy, instances, nets, and ports remain owned by Routable IR.
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
                    kind: graph_node_kind(&node.kind),
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

fn graph_node_kind(kind: &RoutableNodeKind) -> GraphNodeKind {
    match kind {
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
    }
}
