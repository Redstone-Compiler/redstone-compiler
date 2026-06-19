use super::LogicGraphTransformer;
use crate::graph::{GraphNode, GraphNodeKind};
use crate::logic::{Logic, LogicType};

impl LogicGraphTransformer {
    pub fn insert_buffers_for_direct_or_to_or(&mut self) -> eyre::Result<()> {
        let mut direct_edges = Vec::new();
        for node in self.graph.graph.nodes.iter() {
            if !matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
            {
                continue;
            }
            for &output_id in &node.outputs {
                let Some(output) = self.graph.graph.find_node_by_id(output_id) else {
                    continue;
                };
                if matches!(&output.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
                {
                    direct_edges.push((node.id, output.id));
                }
            }
        }

        for (from, to) in direct_edges {
            let first_not = self.graph.graph.add_node(GraphNode {
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![from],
                tag: "auto-buffer".to_owned(),
                ..Default::default()
            });
            let second_not = self.graph.graph.add_node(GraphNode {
                kind: GraphNodeKind::Logic(Logic {
                    logic_type: LogicType::Not,
                }),
                inputs: vec![first_not],
                outputs: vec![to],
                tag: "auto-buffer".to_owned(),
            });
            self.graph
                .graph
                .find_node_by_id_mut(first_not)
                .unwrap()
                .outputs = vec![second_not];

            self.graph
                .graph
                .replace_target_output_node_ids(from, to, vec![first_not]);
            self.graph
                .graph
                .replace_target_input_node_ids(to, from, vec![second_not]);
        }

        self.graph.graph.build_inputs();
        self.graph.graph.build_outputs();
        self.graph.graph.build_producers();
        self.graph.graph.build_consumers();
        self.graph.graph.verify()?;

        Ok(())
    }
}
