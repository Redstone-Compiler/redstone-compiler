use itertools::Itertools;

use crate::graph::{
    module::{GraphModule, GraphModuleId, GraphModulePortId},
    Graph, GraphNodeId,
};

#[derive(Debug, Default)]
pub struct GraphModuleBuilder {
    port_id_iter: GraphModulePortId,
    module_id_iter: GraphModuleId,
}

impl GraphModuleBuilder {
    pub fn new(base_module_id: GraphModuleId) -> Self {
        Self {
            module_id_iter: base_module_id,
            port_id_iter: 0,
        }
    }

    pub fn append(
        &mut self,
        module: GraphModule,
        port_to: Vec<(GraphModuleId, String)>,
        port_from: Vec<(GraphModuleId, String)>,
    ) -> eyre::Result<&mut Self> {
        module.check_init()?;

        Ok(self)
    }

    pub fn generate_parallel(graph: Graph, count: usize) -> GraphModule {
        let inputs = graph
            .inputs()
            .iter()
            .map(|input| graph.find_node_by_id(*input).unwrap().kind.as_input())
            .collect_vec();
        let outputs = graph
            .inputs()
            .iter()
            .map(|output| graph.find_node_by_id(*output).unwrap().kind.as_input())
            .collect_vec();

        let mut instance: GraphModule = graph.into();
        let mut result = GraphModule::from_instances(
            (0..count).map(|_| Box::new(instance.clone())).collect_vec(),
        );

        result.numbering_ports(0);

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::builder::logic::{LogicGraph, LogicGraphBuilder};

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

        GraphModuleBuilder::generate_parallel(fa.graph, 32);

        Ok(())
    }
}
