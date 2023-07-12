use itertools::Itertools;

use crate::graph::{module::GraphModule, Graph};

#[derive(Debug, Default)]
pub struct GraphModuleBuilder {
    module: GraphModule,
}

impl GraphModuleBuilder {
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

    #[test]
    fn unittest_module_full_adder_parallel() {}
}
