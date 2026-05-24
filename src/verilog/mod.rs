pub mod ast;
pub mod lexer;
pub mod lower;
pub mod parser;

use std::fs;
use std::path::Path;

use crate::graph::logic::LogicGraph;

pub fn load_logic_graph(path: impl AsRef<Path>) -> eyre::Result<LogicGraph> {
    let source = fs::read_to_string(path)?;
    let module = parser::parse_module(&source)?;
    lower::lower_module(&module)
}

#[cfg(test)]
mod tests {
    use super::load_logic_graph;

    #[test]
    fn load_logic_graph_reads_verilog_file() -> eyre::Result<()> {
        let graph = load_logic_graph("test/half-adder.v")?;
        let table = graph.truth_table()?;

        assert_eq!(table.input_names, vec!["a", "b"]);
        assert_eq!(table.output_tables["s"], vec![false, true, true, false]);
        assert_eq!(table.output_tables["c"], vec![false, false, false, true]);

        Ok(())
    }
}
