use std::collections::HashMap;

use itertools::Itertools;

use super::cluster::ClusteredGraph;
use super::logic::LogicGraph;
use super::module::{GraphModule, GraphWithSubGraphs};
use super::world::WorldGraph;
use super::{Graph, GraphNode, GraphNodeId, SubGraph};
use crate::graph::module::GraphModulePortTarget;

#[derive(Default)]
pub struct GraphvizBuilder<'a> {
    graph: Option<&'a Graph>,
    module: Option<&'a GraphModule>,
    depth: Option<usize>,
    clusters: Option<Vec<(String, Vec<GraphNodeId>)>>,
    show_node_id: bool,
    table_style: bool,
    named_inputs: Option<HashMap<(GraphNodeId, GraphNodeId), String>>,
    subname: Option<HashMap<GraphNodeId, String>>,
}

impl<'a> GraphvizBuilder<'a> {
    pub fn with_graph(&mut self, graph: &'a Graph) -> &mut Self {
        self.graph = Some(graph);
        self
    }

    pub fn with_module(&mut self, module: &'a GraphModule) -> &mut Self {
        self.module = Some(module);
        self
    }

    pub fn with_depth(&mut self, depth: usize) -> &mut Self {
        self.depth = Some(depth);
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

    pub fn with_colors(&mut self, _colors: HashMap<GraphNodeId, String>) -> &mut Self {
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
            if self.table_style || self.module.is_some() {
                "TB"
            } else {
                "LR"
            },
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
        } else if let Some(module) = self.module {
            self.print_module(module)
        } else {
            unreachable!()
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

        let inputs = if self.table_style && !node.inputs.is_empty() {
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
                                    map.get(&(4, node.id)).map(|name| name.to_owned())
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

    fn print_module(&self, module: &GraphModule) -> String {
        let inputs = module
            .ports
            .iter()
            .filter(|port| port.port_type.is_input())
            .map(|port| {
                let name = match &port.target {
                    GraphModulePortTarget::Node(_) => port.name.clone(),
                    GraphModulePortTarget::Module(_, _) => port.name.clone(),
                    GraphModulePortTarget::Wire(wire) => {
                        format!("{}[{}..0]", port.name, wire.len())
                    }
                };

                format!("<{}> {}", port.name.to_lowercase().replace(" ", "_"), name)
            })
            .join("|");
        let outputs = module
            .ports
            .iter()
            .filter(|port| port.port_type.is_output())
            .map(|port| {
                let name = match &port.target {
                    GraphModulePortTarget::Node(_) => port.name.clone(),
                    GraphModulePortTarget::Module(_, _) => port.name.clone(),
                    GraphModulePortTarget::Wire(wire) => {
                        format!("{}[{}..0]", port.name, wire.len())
                    }
                };

                format!("<{}> {}", port.name.to_lowercase().replace(" ", "_"), name)
            })
            .join("|");

        format!(
            "    module1 [label=\"{{{}}}|{}|{{{}}}\"]",
            inputs, module.name, outputs
        )
    }

    fn print_edges(&self) -> String {
        if let Some(graph) = self.graph {
            if self.table_style {
                graph
                    .nodes
                    .iter()
                    .flat_map(|node| {
                        node.inputs.iter().map(|input| {
                            let in_suffix = format!("in{}_{}", input, node.id);
                            format!("    node{}:out->node{}:{}\n", input, node.id, in_suffix)
                        })
                    })
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
            "".to_string()
        }
    }

    fn print_cluster(&self) -> String {
        let Some(clusters) = &self.clusters else {
            return String::new();
        };

        clusters
            .iter()
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

pub trait ToGraphvizGraph {
    fn to_graphviz(&self) -> String;

    fn to_graphviz_with_clusters(&self, clusters: &[SubGraph]) -> String;
}

impl ToGraphvizGraph for Graph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::default()
            .with_graph(self)
            .build("DefaultGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &[SubGraph]) -> String {
        GraphvizBuilder::default()
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

impl ToGraphvizGraph for LogicGraph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::default()
            .with_graph(&self.graph)
            .build("LogicGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &[SubGraph]) -> String {
        GraphvizBuilder::default()
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

impl ToGraphvizGraph for WorldGraph {
    fn to_graphviz(&self) -> String {
        let mut subnames = HashMap::new();
        subnames.extend(
            self.positions
                .iter()
                .map(|(id, pos)| (*id, format!("{pos:?}"))),
        );
        subnames.extend(self.routings.iter().map(|id| (*id, "Routings".to_string())));

        GraphvizBuilder::default()
            .with_graph(&self.graph)
            .with_table()
            .with_show_node_id()
            .with_subname(subnames)
            .build("WorldGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &[SubGraph]) -> String {
        GraphvizBuilder::default()
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

impl ToGraphvizGraph for GraphWithSubGraphs {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::default()
            .with_graph(&self.0)
            .with_cluster(
                self.1
                    .iter()
                    .enumerate()
                    .map(|(index, g)| (format!("Cluster {}", index), g.clone()))
                    .collect_vec(),
            )
            .build("LogicGraph")
    }

    fn to_graphviz_with_clusters(&self, clusters: &[SubGraph]) -> String {
        GraphvizBuilder::default()
            .with_graph(&self.0)
            .with_cluster(
                self.1
                    .iter()
                    .enumerate()
                    .map(|(index, g)| (format!("Cluster {}", index), g.clone()))
                    .chain(
                        clusters
                            .iter()
                            .enumerate()
                            .map(|(index, g)| (format!("Cluster {}", index), g.nodes.clone())),
                    )
                    .collect_vec(),
            )
            .build("LogicGraph")
    }
}

impl ToGraphvizGraph for ClusteredGraph {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::default()
            .with_graph(&self.graph)
            .build("ClusteredGraph")
    }

    fn to_graphviz_with_clusters(&self, _: &[SubGraph]) -> String {
        unimplemented!()
    }
}

pub trait ToGraphvizModule {
    fn to_graphviz(&self) -> String;
}

impl ToGraphvizModule for GraphModule {
    fn to_graphviz(&self) -> String {
        GraphvizBuilder::default()
            .with_module(self)
            .build("GraphModule")
    }
}
