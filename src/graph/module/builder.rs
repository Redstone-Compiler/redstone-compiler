use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::graph::module::{
    GraphModule, GraphModuleContext, GraphModulePort, GraphModulePortTarget, GraphModulePortType,
    GraphModuleVariable,
};
use crate::graph::Graph;

#[derive(Debug)]
pub struct GraphModuleBuilder {
    context: GraphModuleContext,
}

impl GraphModuleBuilder {
    pub fn new() -> Self {
        Self {
            context: GraphModuleContext::default(),
        }
    }

    pub fn finish(self) -> GraphModuleContext {
        self.context
    }

    pub fn to_graph_module(&mut self, graph: Graph, name: &str) -> GraphModule {
        let mut module: GraphModule = graph.into();
        module.name = format!("{name}#inner");
        let wrap = module.wrap(name).unwrap();
        self.context.append(module);
        self.context.append(wrap.clone());
        wrap
    }

    pub fn generate_parallel(
        &mut self,
        name: &str,
        gm: GraphModule,
        conn: &[(String, String)],
        count: usize,
        wiring: bool,
    ) -> GraphModule {
        let modules = (0..count)
            .map(|index| {
                let mut gm = gm.clone();
                gm.name = format!("{}#{index}", gm.name);
                gm
            })
            .collect_vec();

        let var_outputs: HashSet<String> = conn.iter().map(|(src, _)| src.to_string()).collect();
        let var_inputs: HashSet<String> = conn.iter().map(|(_, tar)| tar.to_string()).collect();
        let ports = if !wiring {
            Self::build_ports_seperately(&gm, count, var_inputs, var_outputs)
        } else {
            Self::build_ports_wired(&gm, count, var_inputs, var_outputs)
        };

        let instances = modules.iter().map(|mo| mo.name.clone()).collect_vec();
        let vars = (0..count - 1)
            .flat_map(|index| {
                conn.iter()
                    .map(|(src, tar)| GraphModuleVariable {
                        var_type: GraphModulePortType::InputNet,
                        source: (format!("{}#{index}", gm.name), src.to_owned()),
                        target: (format!("{}#{}", gm.name, index + 1), tar.to_owned()),
                    })
                    .collect_vec()
            })
            .collect_vec();

        self.context.extend(modules.into_iter());

        GraphModule {
            name: name.to_string(),
            graph: None,
            instances,
            vars,
            ports,
        }
    }

    fn build_ports_seperately(
        gm: &GraphModule,
        count: usize,
        var_inputs: HashSet<String>,
        var_outputs: HashSet<String>,
    ) -> Vec<GraphModulePort> {
        let mut ports = vec![];

        for index in 0..count {
            let filter_ports = |port: &&GraphModulePort| {
                let port_name = &port.name;
                match index {
                    0 => port.port_type.is_input() || !var_outputs.contains(port_name),
                    i if i == count - 1 => {
                        port.port_type.is_output() || !var_inputs.contains(port_name)
                    }
                    _ => !var_inputs.contains(port_name) && !var_outputs.contains(port_name),
                }
            };

            ports.extend(gm.ports.iter().filter(&filter_ports).map(|port| {
                let target = GraphModulePortTarget::Module(
                    format!("{}#{}", gm.name, index),
                    port.name.to_string(),
                );
                let name = match index {
                    0 => format!("{}#0", port.name),
                    i if i == count - 1 => format!("{}#{}", port.name, count - 1),
                    _ => format!("{}#{}", port.name, index),
                };
                GraphModulePort {
                    name,
                    port_type: port.port_type,
                    target,
                }
            }));
        }

        ports
    }

    fn build_ports_wired(
        gm: &GraphModule,
        count: usize,
        var_inputs: HashSet<String>,
        var_outputs: HashSet<String>,
    ) -> Vec<GraphModulePort> {
        let mut port_types: HashMap<String, GraphModulePortType> = HashMap::default();
        let mut instance_ports: HashMap<String, Vec<(String, String)>> = HashMap::default();

        for index in 0..count {
            let filter_ports = |port: &&GraphModulePort| {
                let port_name = &port.name;
                match index {
                    0 => port.port_type.is_input() || !var_outputs.contains(port_name),
                    i if i == count - 1 => {
                        port.port_type.is_output() || !var_inputs.contains(port_name)
                    }
                    _ => !var_inputs.contains(port_name) && !var_outputs.contains(port_name),
                }
            };

            let target = format!("{}#{}", gm.name, index);
            gm.ports.iter().filter(&filter_ports).for_each(|port| {
                port_types
                    .entry(port.name.clone())
                    .or_insert(port.port_type);

                let name = match index {
                    0 => format!("{}#0", port.name),
                    i if i == count - 1 => format!("{}#{}", port.name, count - 1),
                    _ => format!("{}#{}", port.name, index),
                };

                instance_ports
                    .entry(port.name.clone())
                    .or_default()
                    .push((target.clone(), name));
            });
        }

        instance_ports
            .into_iter()
            .map(|(key, value)| GraphModulePort {
                port_type: port_types.remove(&key).unwrap(),
                name: key,
                target: GraphModulePortTarget::Wire(value),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::GraphModuleBuilder;
    use crate::graph::graphviz::{ToGraphvizGraph, ToGraphvizModule};
    use crate::graph::logic::builder::LogicGraphBuilder;
    use crate::graph::logic::LogicGraph;
    use crate::graph::module::{GraphModule, GraphWithSubGraphs};

    fn build_graph_from_stmt(stmt: &str, output: &str) -> eyre::Result<LogicGraph> {
        LogicGraphBuilder::new(stmt.to_string()).build(output.to_string())
    }

    fn get_full_adder_graph() -> eyre::Result<LogicGraph> {
        let out_s = build_graph_from_stmt("(a^b)^cin", "s")?;
        let out_cout = build_graph_from_stmt("(a&b)|(s&cin)", "cout")?;

        let mut fa = out_s.clone();
        fa.graph.merge(out_cout.graph);

        Ok(fa)
    }

    #[test]
    fn unittest_module_full_adder_parallel() -> eyre::Result<()> {
        let mut builder = GraphModuleBuilder::new();

        let fa = get_full_adder_graph()?;
        let gm: GraphModule = fa.graph.to_module(&mut builder, "full_adder");

        let gm = builder.generate_parallel(
            "full_adder32",
            gm,
            vec![("cout".to_string(), "cin".to_string())].as_slice(),
            4,
            true,
        );
        println!("{}", gm.to_graphviz());

        let context = builder.finish();
        let graph: GraphWithSubGraphs = (&context, &gm).into();

        // let mut transform = LogicGraphTransformer::new(LogicGraph { graph: graph.0 });
        // transform.decompose_xor()?;
        // transform.decompose_and()?;
        // let graph = transform.finish();
        println!("{}", graph.to_graphviz());

        Ok(())
    }
}
