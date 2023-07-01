use super::{builder::logic::LogicGraph, Graph, GraphNode};

pub struct GraphvizBuilder<'a> {
    graph: &'a Graph,
}

impl<'a> GraphvizBuilder<'a> {
    pub fn new(g: &'a Graph) -> Self {
        Self { graph: g }
    }

    pub fn build(&self, graph_name: &str) -> String {
        format!(
            r#"
digraph {graph_name} {{
    graph [label="Line edges", splines=line, nodesep=0.8]
    node [shape=box]
{}
{}
}}
        "#,
            self.print_nodes(),
            self.print_edges()
        )
    }

    fn print_nodes(&self) -> String {
        self.graph
            .nodes
            .iter()
            .map(|node| Self::print_node(node))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn print_node(node: &GraphNode) -> String {
        format!("    node{} [label=\"{}\"]", node.id, node.kind.name())
    }

    fn print_edges(&self) -> String {
        self.graph
            .nodes
            .iter()
            .map(|node| {
                format!(
                    "    node{}->{{{}}}",
                    node.id,
                    node.outputs
                        .iter()
                        .map(|id| format!("node{id}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub trait ToGraphviz {
    fn to_graphviz(&self) -> String;
}

impl ToGraphviz for Graph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::new(self).build("DefaultGraph")
    }
}

impl ToGraphviz for LogicGraph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::new(&self.graph).build("LogicGraph")
    }
}
