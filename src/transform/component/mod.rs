// Generate general World3D Component from LogicGraph

use crate::{
    graph::logic::LogicGraph,
    world::{position::DimSize, world::World3D},
};

// struct IOTemplate {}

struct Template {
    dim: DimSize,
}

// pub fn generate_template(graph: &LogicGraph, dim: DimSize) -> Vec<IOTemplate> {
//     todo!()
// }

// 가능한 모든 경우의 수를 생성하는 것을 목표로한다.
pub fn generate(graph: &LogicGraph) -> World3D {
    let inputs = graph.inputs();
    todo!()
}

#[cfg(test)]
mod tests {

    use crate::{
        graph::{
            graphviz::ToGraphvizGraph,
            logic::{builder::LogicGraphBuilder, LogicGraph},
        },
        nbt::{NBTRoot, ToNBT},
        transform::{component::generate, logic::LogicGraphTransformer},
    };

    fn build_graph_from_stmt(stmt: &str, output: &str) -> eyre::Result<LogicGraph> {
        LogicGraphBuilder::new(stmt.to_string()).build(output.to_string())
    }

    #[test]
    fn test_generate_component_and() -> eyre::Result<()> {
        let logic_graph = build_graph_from_stmt("a&b", "c")?;

        let mut transform = LogicGraphTransformer::new(logic_graph);
        transform.decompose_and()?;
        transform.remove_double_neg_expression();
        let logic_graph = transform.finish();
        println!("{}", logic_graph.to_graphviz());

        let world3d = generate(&logic_graph);

        let nbt: NBTRoot = world3d.to_nbt();
        nbt.save("test/and-gate-new.nbt");

        Ok(())
    }
}
