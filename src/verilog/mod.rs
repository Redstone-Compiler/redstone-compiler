pub mod ast;
pub mod design;
pub mod lexer;
pub mod lower;
pub mod parser;

use std::fs;
use std::path::Path;

use crate::graph::logic::LogicGraph;
use crate::graph::module::GraphModuleDesign;

pub fn load_logic_graph(path: impl AsRef<Path>) -> eyre::Result<LogicGraph> {
    let source = fs::read_to_string(path)?;
    let modules = parser::parse_modules(&source)?;
    lower::lower_modules(&modules)
}

pub fn load_graph_module_design(path: impl AsRef<Path>) -> eyre::Result<GraphModuleDesign> {
    let source = fs::read_to_string(path)?;
    let modules = parser::parse_modules(&source)?;
    design::lower_design_modules(&modules)
}

#[cfg(test)]
mod tests {
    use super::load_logic_graph;
    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::graph::logic::LogicGraph;
    use crate::graph::GraphNodeKind;
    use crate::logic::LogicType;
    use crate::nbt::NBTRoot;
    use crate::output::OutputMetadata;
    use crate::transform::place_and_route::utils::{world_to_logic, world_to_logic_with_outputs};

    #[test]
    fn load_logic_graph_reads_verilog_file() -> eyre::Result<()> {
        let graph = load_logic_graph("test/half-adder.v")?;
        let table = graph.truth_table()?;

        assert_eq!(table.input_names, vec!["a", "b"]);
        assert_eq!(table.output_tables["s"], vec![false, true, true, false]);
        assert_eq!(table.output_tables["c"], vec![false, false, false, true]);

        Ok(())
    }

    #[test]
    fn half_adder_verilog_raw_graph_has_named_logic_outputs() -> eyre::Result<()> {
        let graph = load_logic_graph("test/half-adder.v")?;
        maybe_dump_graphviz("raw Verilog LogicGraph", &graph);

        assert_eq!(
            output_source_logic_types(&graph),
            vec![
                ("c".to_owned(), LogicType::And),
                ("s".to_owned(), LogicType::Xor)
            ]
        );
        assert_eq!(output_source_input_names(&graph, "s"), vec!["a", "b"]);
        assert_eq!(output_source_input_names(&graph, "c"), vec!["a", "b"]);

        Ok(())
    }

    #[test]
    fn half_adder_prepare_place_preserves_named_outputs() -> eyre::Result<()> {
        let graph = load_logic_graph("test/half-adder.v")?;
        let prepared = graph.prepare_place()?;
        maybe_dump_graphviz("prepared Verilog LogicGraph", &prepared);

        assert_eq!(output_names(&prepared), vec!["c", "s"]);
        let observable_sources = prepared.externally_observable_output_source_ids();
        assert_eq!(observable_sources.len(), 2);
        for source_id in output_source_ids(&prepared) {
            assert!(observable_sources.contains(&source_id));
        }

        let table = prepared.truth_table()?;
        assert_eq!(table.input_names, vec!["a", "b"]);
        assert_eq!(table.output_tables["s"], vec![false, true, true, false]);
        assert_eq!(table.output_tables["c"], vec![false, false, false, true]);

        Ok(())
    }

    #[test]
    fn half_adder_generated_nbt_matches_verilog_functions_without_named_outputs() -> eyre::Result<()>
    {
        let expected = load_logic_graph("test/half-adder.v")?;
        let nbt = NBTRoot::load("test/half-adder-generated-from-verilog.nbt")?;
        let generated = world_to_logic(&nbt.to_world())?;
        maybe_dump_graphviz("NBT roundtrip LogicGraph", &generated);

        assert!(generated
            .truth_table()?
            .contains_output_tables_under_input_permutation(&expected.truth_table()?));
        assert!(output_names(&generated).is_empty());

        Ok(())
    }

    #[test]
    fn half_adder_generated_nbt_restores_outputs_with_metadata() -> eyre::Result<()> {
        let expected = load_logic_graph("test/half-adder.v")?;
        let nbt = NBTRoot::load("test/half-adder-generated-from-verilog.nbt")?;
        let metadata = OutputMetadata::load("test/half-adder-generated-from-verilog.outputs.json")?;
        let generated = world_to_logic_with_outputs(&nbt.to_world(), &metadata)?;

        assert_eq!(output_names(&generated), vec!["c", "s"]);
        assert!(generated
            .externally_observable_truth_table()?
            .contains_output_tables_under_input_permutation(
                &expected.externally_observable_truth_table()?
            ));

        Ok(())
    }

    fn maybe_dump_graphviz(name: &str, graph: &LogicGraph) {
        if std::env::var_os("PRINT_VERILOG_GRAPHS").is_some() {
            eprintln!("--- {name} ---\n{}", graph.to_graphviz());
        }
    }

    fn output_names(graph: &LogicGraph) -> Vec<String> {
        let mut names = graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) => Some(name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    fn output_source_ids(graph: &LogicGraph) -> Vec<usize> {
        let mut ids = graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(_) => Some(node.inputs[0]),
                _ => None,
            })
            .collect::<Vec<_>>();
        ids.sort();
        ids
    }

    fn output_source_logic_types(graph: &LogicGraph) -> Vec<(String, LogicType)> {
        let mut outputs = graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) => {
                    let source = graph.find_node_by_id(node.inputs[0]).unwrap();
                    match &source.kind {
                        GraphNodeKind::Logic(logic) => Some((name.clone(), logic.logic_type)),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        outputs.sort_by(|(a, _), (b, _)| a.cmp(b));
        outputs
    }

    fn output_source_input_names(graph: &LogicGraph, output_name: &str) -> Vec<String> {
        let output = graph
            .nodes
            .iter()
            .find(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == output_name))
            .unwrap();
        let source = graph.find_node_by_id(output.inputs[0]).unwrap();
        let mut input_names = source
            .inputs
            .iter()
            .map(|input_id| graph.find_node_by_id(*input_id).unwrap())
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) => Some(name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        input_names.sort();
        input_names
    }
}
