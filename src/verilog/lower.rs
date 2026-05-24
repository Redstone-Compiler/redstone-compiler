use crate::graph::logic::LogicGraph;
use crate::verilog::ast::VerilogModule;

pub fn lower_module(module: &VerilogModule) -> eyre::Result<LogicGraph> {
    if module.assignments.is_empty() {
        eyre::bail!("module `{}` has no continuous assignments", module.name);
    }

    LogicGraph::from_assignments(
        module
            .assignments
            .iter()
            .map(|assign| (assign.output.clone(), assign.expr.to_logic_stmt())),
    )
}

#[cfg(test)]
mod tests {
    use super::lower_module;
    use crate::verilog::parser::parse_module;

    #[test]
    fn lowers_half_adder_to_truth_table() -> eyre::Result<()> {
        let module = parse_module(
            r#"
            module half_adder(a, b, s, c);
              input a, b;
              output s, c;
              assign s = a ^ b;
              assign c = a & b;
            endmodule
            "#,
        )?;

        let graph = lower_module(&module)?;
        let table = graph.truth_table()?;

        assert_eq!(table.input_names, vec!["a", "b"]);
        assert_eq!(table.output_tables["s"], vec![false, true, true, false]);
        assert_eq!(table.output_tables["c"], vec![false, false, false, true]);

        Ok(())
    }

    #[test]
    fn lowers_full_adder_with_intermediate_output_use() -> eyre::Result<()> {
        let module = parse_module(
            r#"
            module full_adder(a, b, cin, sum, cout);
              input a, b, cin;
              output sum, cout;
              wire s;
              assign s = a ^ b;
              assign sum = s ^ cin;
              assign cout = (a & b) | (s & cin);
            endmodule
            "#,
        )?;

        let graph = lower_module(&module)?;
        let table = graph.truth_table()?;
        let sum = (0..8)
            .map(|mask: usize| mask.count_ones() % 2 == 1)
            .collect::<Vec<_>>();
        let carry = (0..8)
            .map(|mask: usize| mask.count_ones() >= 2)
            .collect::<Vec<_>>();

        assert_eq!(table.output_tables["sum"], sum);
        assert_eq!(table.output_tables["cout"], carry);

        Ok(())
    }
}
