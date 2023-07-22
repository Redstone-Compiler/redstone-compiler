use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::graph::{
    module::{
        GraphModule, GraphModuleContext, GraphModulePort, GraphModulePortTarget,
        GraphModulePortType, GraphModuleVariable,
    },
    Graph,
};

#[derive(Debug, Default)]
pub struct GraphModuleBuilder {
    context: GraphModuleContext,
}

impl GraphModuleBuilder {
    pub fn append(
        &mut self,
        module: GraphModule,
        port_to: Vec<(String, String)>,
        port_from: Vec<(String, String)>,
    ) -> eyre::Result<&mut Self> {
        Ok(self)
    }

    pub fn generate_parallel(
        name: &str,
        gm: GraphModule,
        conn: &[(String, String)],
        count: usize,
        wire: bool,
    ) -> GraphModule {
        let modules = (0..count)
            .map(|index| {
                let mut gm = gm.clone();
                gm.name = format!("{}#{index}", gm.name);
                gm
            })
            .collect_vec();

        let var_inputs: HashSet<String> = conn.iter().map(|(src, _)| src.to_string()).collect();
        let var_outputs: HashSet<String> = conn.iter().map(|(_, tar)| tar.to_string()).collect();

        let ports = if !wire {
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
        } else {
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
        };

        GraphModule {
            name: name.to_string(),
            graph: None,
            instances: modules.iter().map(|mo| mo.name.clone()).collect_vec(),
            vars: (0..count - 1)
                .flat_map(|index| {
                    conn.iter()
                        .map(|(src, tar)| GraphModuleVariable {
                            var_type: GraphModulePortType::InputNet,
                            source: (format!("{}#{index}", gm.name), src.to_owned()),
                            target: (format!("{}#{}", gm.name, index + 1), tar.to_owned()),
                        })
                        .collect_vec()
                })
                .collect_vec(),
            ports,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{
        builder::logic::{LogicGraph, LogicGraphBuilder},
        module::GraphModule,
    };

    use super::GraphModuleBuilder;

    fn build_graph_from_stmt(stmt: &str, output: &str) -> eyre::Result<LogicGraph> {
        LogicGraphBuilder::new(stmt.to_string()).build(output.to_string())
    }

    #[test]
    fn unittest_module_full_adder_parallel() -> eyre::Result<()> {
        let out_s = build_graph_from_stmt("(a^b)^cin", "s")?;
        let out_cout = build_graph_from_stmt("(a&b)|(s&cin)", "cout")?;

        let mut fa = out_s.clone();
        fa.graph.merge(out_cout.graph);

        let gm: GraphModule = fa.graph.to_module("full_adder");
        println!("{:?}", gm);

        let gm = GraphModuleBuilder::generate_parallel(
            "full_adder32",
            gm,
            vec![("cin".to_string(), "cout".to_string())].as_slice(),
            32,
            true,
        );
        println!("{:#?}", gm.ports);

        Ok(())
    }
}
