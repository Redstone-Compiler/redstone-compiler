use std::collections::HashMap;

use itertools::Itertools;

use super::{
    builder::{logic::LogicGraph, world::WorldGraph},
    Graph, GraphNode, GraphNodeId, SubGraph,
};

pub struct GraphvizBuilder<'a> {
    graph: Option<&'a Graph>,
    clusters: Option<Vec<(String, Vec<GraphNodeId>)>>,
    show_node_id: bool,
    table_style: bool,
    named_inputs: Option<HashMap<(GraphNodeId, GraphNodeId), String>>,
    subname: Option<HashMap<GraphNodeId, String>>,
}

impl<'a> GraphvizBuilder<'a> {
    pub fn new() -> Self {
        Self {
            graph: None,
            clusters: None,
            show_node_id: false,
            table_style: false,
            named_inputs: None,
            subname: None,
        }
    }

    pub fn with_graph(&mut self, graph: &'a Graph) -> &mut Self {
        self.graph = Some(graph);
        self
    }

    pub fn with_cluster(&mut self, clusters: Vec<(String, Vec<GraphNodeId>)>) -> &mut Self {
        self.clusters = Some(clusters);
        self
    }

    pub fn with_special_inputs(
        &mut self,
        // <source, <target, name>>
        inputs: HashMap<(GraphNodeId, GraphNodeId), String>,
    ) -> &mut Self {
        self.named_inputs = Some(inputs);
        self
    }

    pub fn with_colors(&mut self, colors: HashMap<GraphNodeId, String>) -> &mut Self {
        self
    }

    pub fn with_table(&mut self) -> &mut Self {
        self.table_style = true;
        self
    }

    pub fn with_show_node_id(&mut self) -> &mut Self {
        self.show_node_id = true;
        self
    }

    pub fn with_subname(&mut self, subname: HashMap<GraphNodeId, String>) -> &mut Self {
        self.subname = Some(subname);
        self
    }

    pub fn build(&self, graph_name: &str) -> String {
        format!(
            r#"
digraph {graph_name} {{
    rankdir={}
    graph [label="Line edges", nodesep={}]
    node [shape={}]
{}
{}
{}
}}
        "#,
            if self.table_style { "TB" } else { "LR" },
            if self.table_style { "auto" } else { "0.8" },
            if self.table_style {
                "plaintext"
            } else {
                "record"
            },
            self.print_nodes(),
            self.print_edges(),
            self.print_cluster(),
        )
    }

    fn print_nodes(&self) -> String {
        if let Some(graph) = self.graph {
            graph
                .nodes
                .iter()
                .map(|node| self.print_node(node))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            todo!()
        }
    }

    fn print_node(&self, node: &GraphNode) -> String {
        let name = if self.show_node_id {
            format!("{} #{}", node.kind.name(), node.id)
        } else {
            node.kind.name()
        };

        let name = if let Some(subname) = self
            .subname
            .as_ref()
            .and_then(|subname| subname.get(&node.id))
        {
            if self.table_style {
                format!(
                    r#"<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                    <TR><TD ALIGN="LEFT">{name}</TD></TR>
                    <TR><TD ALIGN="LEFT">{subname}</TD></TR>
                </TABLE>"#
                )
            } else {
                format!("{name} | {subname}")
            }
        } else {
            name
        };

        let inputs = if self.table_style && node.inputs.len() > 0 {
            format!(
                r#"<TR>
                {}
            </TR>"#,
                node.inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| {
                        format!(
                            r#"<TD PORT="in{}_{}">{}</TD>"#,
                            input,
                            node.id,
                            self.named_inputs
                                .as_ref()
                                .and_then(|map| {
                                    map.get(&(4, node.id))
                                        .and_then(|name| Some(name.to_owned()))
                                })
                                .unwrap_or(format!("Input {}", index + 1))
                        )
                    })
                    .join("")
            )
        } else {
            "".to_string()
        };

        if !self.table_style {
            if node.tag.is_empty() {
                format!("    node{} [label=\"{}\"]", node.id, name)
            } else {
                format!("    node{} [label=\"{} | {}\"]", node.id, name, node.tag)
            }
        } else {
            format!(
                r#"    node{} [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            {}
            <TR><TD COLSPAN="{}" PORT="out">
                {}
            </TD></TR>
        </TABLE>
    >]"#,
                node.id,
                inputs,
                node.inputs.len(),
                name
            )
        }
    }

    fn print_edges(&self) -> String {
        if let Some(graph) = self.graph {
            if self.table_style {
                graph
                    .nodes
                    .iter()
                    .map(|node| {
                        node.inputs.iter().map(|input| {
                            format!(
                                "    node{}:out->node{}:{}\n",
                                input,
                                node.id,
                                format!("in{}_{}", input, node.id),
                            )
                        })
                    })
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("")
            } else {
                graph
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
        } else {
            todo!()
        }
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
        GraphvizBuilder::new()
            .with_graph(self)
            .build("DefaultGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &Vec<SubGraph>) -> String {
        GraphvizBuilder::new()
            .with_graph(self)
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
        GraphvizBuilder::new()
            .with_graph(&self.graph)
            .build("LogicGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &Vec<SubGraph>) -> String {
        GraphvizBuilder::new()
            .with_graph(&self.graph)
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

impl ToGraphviz for WorldGraph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::new()
            .with_graph(&self.graph)
            .with_table()
            .with_show_node_id()
            .with_subname(
                self.positions
                    .iter()
                    .map(|(id, pos)| (*id, format!("{pos:?}")))
                    .collect(),
            )
            .build("WorldGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &Vec<SubGraph>) -> String {
        GraphvizBuilder::new()
            .with_graph(&self.graph)
            .with_cluster(
                clusters
                    .iter()
                    .enumerate()
                    .map(|(index, g)| (format!("Cluster {}", index), g.nodes.clone()))
                    .collect_vec(),
            )
            .build("WorldGraph")
    }
}
