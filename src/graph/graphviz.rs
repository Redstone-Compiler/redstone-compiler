use itertools::Itertools;

use super::{builder::logic::LogicGraph, Graph, GraphNode, GraphNodeId, SubGraph};

pub struct GraphvizBuilder<'a> {
    graph: &'a Graph,
    clusters: Option<Vec<(String, Vec<GraphNodeId>)>>,
}

impl<'a> GraphvizBuilder<'a> {
    pub fn new(g: &'a Graph) -> Self {
        Self {
            graph: g,
            clusters: None,
        }
    }

    pub fn with_cluster(&mut self, clusters: Vec<(String, Vec<GraphNodeId>)>) -> &mut Self {
        self.clusters = Some(clusters);
        self
    }

    pub fn build(&self, graph_name: &str) -> String {
        format!(
            r#"
digraph {graph_name} {{
    rankdir=LR
    graph [label="Line edges", splines=line, nodesep=0.8]
    node [shape=record]
{}
{}
{}
}}
        "#,
            self.print_nodes(),
            self.print_edges(),
            self.print_cluster(),
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
        if node.tag.is_empty() {
            format!("    node{} [label=\"{}\"]", node.id, node.kind.name())
        } else {
            format!(
                "    node{} [label=\"{} | {}\"]",
                node.id,
                node.kind.name(),
                node.tag
            )
        }
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

    fn print_cluster(&self) -> String {
        let Some(clusters) = &self.clusters else {
            return String::new();
        };

        clusters
            .into_iter()
            .enumerate()
            .map(|(index, (name, members))| {
                format!(
                    "    subgraph cluster_{index} {{ label=\"{name}\" {} }}",
                    members
                        .iter()
                        .map(|id| format!("node{id}"))
                        .collect_vec()
                        .join(" ")
                )
            })
            .collect_vec()
            .join("\n")
    }
}

pub trait ToGraphviz {
    fn to_graphviz(&self) -> String;

    fn to_graphviz_with_clusters(&self, clusters: &Vec<SubGraph>) -> String;
}

impl ToGraphviz for Graph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::new(self).build("DefaultGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &Vec<SubGraph>) -> String {
        GraphvizBuilder::new(self)
            .with_cluster(
                clusters
                    .iter()
                    .enumerate()
                    .map(|(index, g)| (format!("Cluster {}", index), g.nodes.clone()))
                    .collect_vec(),
            )
            .build("DefaultGraph")
    }
}

impl ToGraphviz for LogicGraph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::new(&self.graph).build("LogicGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &Vec<SubGraph>) -> String {
        GraphvizBuilder::new(&self.graph)
            .with_cluster(
                clusters
                    .iter()
                    .enumerate()
                    .map(|(index, g)| (format!("Cluster {}", index), g.nodes.clone()))
                    .collect_vec(),
            )
            .build("LogicGraph")
    }
}
