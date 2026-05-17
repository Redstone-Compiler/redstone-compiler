use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::logic::{Logic, LogicType};

pub mod core;
pub mod layout;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SequentialType {
    RsLatch,
    DLatch,
    DFlipFlop,
}

impl SequentialType {
    pub fn name(&self) -> String {
        match self {
            SequentialType::RsLatch => "RsLatch".to_owned(),
            SequentialType::DLatch => "DLatch".to_owned(),
            SequentialType::DFlipFlop => "DFlipFlop".to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SequentialPrimitive {
    pub sequential_type: SequentialType,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    pub inner_graph: Box<Graph>,
}

impl PartialEq for SequentialPrimitive {
    fn eq(&self, other: &Self) -> bool {
        self.sequential_type == other.sequential_type
            && self.input_ports == other.input_ports
            && self.output_ports == other.output_ports
    }
}

impl Eq for SequentialPrimitive {}

impl Default for SequentialPrimitive {
    fn default() -> Self {
        Self {
            sequential_type: SequentialType::RsLatch,
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            inner_graph: Box::default(),
        }
    }
}

impl SequentialPrimitive {
    pub fn new(
        sequential_type: SequentialType,
        input_ports: Vec<String>,
        output_ports: Vec<String>,
    ) -> Self {
        Self {
            sequential_type,
            input_ports,
            output_ports,
            inner_graph: Box::new(default_inner_graph(sequential_type)),
        }
    }

    pub fn rs_latch() -> Self {
        Self::new(
            SequentialType::RsLatch,
            vec!["s".to_owned(), "r".to_owned()],
            vec!["q".to_owned(), "nq".to_owned()],
        )
    }

    pub fn d_latch() -> Self {
        Self::new(
            SequentialType::DLatch,
            vec!["d".to_owned(), "en".to_owned()],
            vec!["q".to_owned(), "nq".to_owned()],
        )
    }

    pub fn name(&self) -> String {
        self.sequential_type.name()
    }

    pub fn feedback_cores(&self) -> Vec<core::FeedbackCore> {
        core::feedback_cores(&self.inner_graph)
    }

    pub fn rs_latch_core(&self) -> Option<core::RsLatchCore> {
        core::recognize_rs_latch_core(&self.inner_graph)
    }
}

fn default_inner_graph(sequential_type: SequentialType) -> Graph {
    match sequential_type {
        SequentialType::RsLatch => rs_latch_inner_graph(),
        SequentialType::DLatch => d_latch_inner_graph(),
        SequentialType::DFlipFlop => Graph::default(),
    }
}

fn rs_latch_inner_graph() -> Graph {
    // q = ~(r | nq), nq = ~(s | q)
    let mut graph = Graph {
        nodes: vec![
            GraphNode {
                id: 0,
                kind: GraphNodeKind::Input("s".to_owned()),
                ..Default::default()
            },
            GraphNode {
                id: 1,
                kind: GraphNodeKind::Input("r".to_owned()),
                ..Default::default()
            },
            GraphNode {
                id: 2,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Or,
                }),
                inputs: vec![1, 5],
                ..Default::default()
            },
            GraphNode {
                id: 3,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                id: 4,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Or,
                }),
                inputs: vec![0, 3],
                ..Default::default()
            },
            GraphNode {
                id: 5,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![4],
                ..Default::default()
            },
            GraphNode {
                id: 6,
                kind: GraphNodeKind::Output("q".to_owned()),
                inputs: vec![3],
                ..Default::default()
            },
            GraphNode {
                id: 7,
                kind: GraphNodeKind::Output("nq".to_owned()),
                inputs: vec![5],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify().unwrap();
    graph
}

fn d_latch_inner_graph() -> Graph {
    // s = d & en, r = ~d & en, q = ~(r | nq), nq = ~(s | q)
    let mut graph = Graph {
        nodes: vec![
            GraphNode {
                id: 0,
                kind: GraphNodeKind::Input("d".to_owned()),
                ..Default::default()
            },
            GraphNode {
                id: 1,
                kind: GraphNodeKind::Input("en".to_owned()),
                ..Default::default()
            },
            GraphNode {
                id: 2,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::And,
                }),
                inputs: vec![0, 1],
                ..Default::default()
            },
            GraphNode {
                id: 3,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![0],
                ..Default::default()
            },
            GraphNode {
                id: 4,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::And,
                }),
                inputs: vec![3, 1],
                ..Default::default()
            },
            GraphNode {
                id: 5,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Or,
                }),
                inputs: vec![4, 8],
                ..Default::default()
            },
            GraphNode {
                id: 6,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![5],
                ..Default::default()
            },
            GraphNode {
                id: 7,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Or,
                }),
                inputs: vec![2, 6],
                ..Default::default()
            },
            GraphNode {
                id: 8,
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![7],
                ..Default::default()
            },
            GraphNode {
                id: 9,
                kind: GraphNodeKind::Output("q".to_owned()),
                inputs: vec![6],
                ..Default::default()
            },
            GraphNode {
                id: 10,
                kind: GraphNodeKind::Output("nq".to_owned()),
                inputs: vec![8],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify().unwrap();
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rs_latch_contains_cyclic_gate_level_inner_graph() {
        let primitive = SequentialPrimitive::rs_latch();

        assert_eq!(primitive.input_ports, vec!["s".to_owned(), "r".to_owned()]);
        assert_eq!(
            primitive.output_ports,
            vec!["q".to_owned(), "nq".to_owned()]
        );
        assert!(primitive.inner_graph.has_cycle());
        assert!(primitive
            .inner_graph
            .nodes
            .iter()
            .any(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == "q")));
        assert!(primitive
            .inner_graph
            .nodes
            .iter()
            .any(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == "nq")));
        primitive.inner_graph.verify().unwrap();
    }

    #[test]
    fn sequential_equality_ignores_inner_graph_identity() {
        let left = SequentialPrimitive::rs_latch();
        let right = SequentialPrimitive::new(
            SequentialType::RsLatch,
            vec!["s".to_owned(), "r".to_owned()],
            vec!["q".to_owned(), "nq".to_owned()],
        );

        assert_eq!(left, right);
    }

    #[test]
    fn sequential_primitive_reports_feedback_core() {
        let primitive = SequentialPrimitive::rs_latch();

        assert!(matches!(
            primitive.feedback_cores().as_slice(),
            [core::FeedbackCore::RsLatch(_)]
        ));
        assert!(primitive.rs_latch_core().is_some());
    }

    #[test]
    fn d_latch_contains_input_gating_around_rs_latch_core() {
        let primitive = SequentialPrimitive::d_latch();

        assert_eq!(primitive.input_ports, vec!["d".to_owned(), "en".to_owned()]);
        assert_eq!(
            primitive.output_ports,
            vec!["q".to_owned(), "nq".to_owned()]
        );
        assert!(primitive.inner_graph.has_cycle());
        assert!(primitive.rs_latch_core().is_some());
        primitive.inner_graph.verify().unwrap();
    }
}
