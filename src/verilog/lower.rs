use std::collections::HashMap;

use crate::graph::logic::LogicGraph;
use crate::verilog::ast::{Expr, VerilogModule};

pub fn lower_module(module: &VerilogModule) -> eyre::Result<LogicGraph> {
    let mut context = HashMap::new();
    context.insert(module.name.clone(), module);
    lower_module_with_context(&context, module)
}

pub fn lower_modules(modules: &[VerilogModule]) -> eyre::Result<LogicGraph> {
    let Some(top_module) = modules.last() else {
        eyre::bail!("expected at least one Verilog module");
    };
    let context = modules
        .iter()
        .map(|module| (module.name.clone(), module))
        .collect::<HashMap<_, _>>();

    lower_module_with_context(&context, top_module)
}

fn lower_module_with_context(
    context: &HashMap<String, &VerilogModule>,
    module: &VerilogModule,
) -> eyre::Result<LogicGraph> {
    let assignments = collect_assignments(context, module, None, &HashMap::new())?;
    if assignments.is_empty() {
        eyre::bail!("module `{}` has no continuous assignments", module.name);
    }

    LogicGraph::from_assignments(assignments)
}

fn collect_assignments(
    context: &HashMap<String, &VerilogModule>,
    module: &VerilogModule,
    instance_prefix: Option<&str>,
    substitutions: &HashMap<String, String>,
) -> eyre::Result<Vec<(String, String)>> {
    let mut assignments = Vec::new();

    for instance in &module.instances {
        let child = context.get(&instance.module_name).ok_or_else(|| {
            eyre::eyre!(
                "unknown Verilog module `{}` for instance `{}`",
                instance.module_name,
                instance.instance_name
            )
        })?;
        let child_prefix = scoped_name(instance_prefix, &instance.instance_name);
        let child_substitutions = instance
            .connections
            .iter()
            .map(|(port, signal)| {
                (
                    port.clone(),
                    rewrite_signal(signal, instance_prefix, substitutions),
                )
            })
            .collect::<HashMap<_, _>>();
        assignments.extend(collect_assignments(
            context,
            child,
            Some(&child_prefix),
            &child_substitutions,
        )?);
    }

    for assignment in &module.assignments {
        let output = rewrite_signal(&assignment.output, instance_prefix, substitutions);
        let expr = rewrite_expr(&assignment.expr, instance_prefix, substitutions);
        assignments.push((output, expr.to_logic_stmt()));
    }

    Ok(assignments)
}

fn rewrite_expr(
    expr: &Expr,
    instance_prefix: Option<&str>,
    substitutions: &HashMap<String, String>,
) -> Expr {
    match expr {
        Expr::Ident(name) => Expr::Ident(rewrite_signal(name, instance_prefix, substitutions)),
        Expr::Not(expr) => Expr::Not(Box::new(rewrite_expr(expr, instance_prefix, substitutions))),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(rewrite_expr(left, instance_prefix, substitutions)),
            right: Box::new(rewrite_expr(right, instance_prefix, substitutions)),
        },
    }
}

fn rewrite_signal(
    name: &str,
    instance_prefix: Option<&str>,
    substitutions: &HashMap<String, String>,
) -> String {
    if let Some(replacement) = substitutions.get(name) {
        return replacement.clone();
    }

    instance_prefix
        .map(|prefix| scoped_name(Some(prefix), name))
        .unwrap_or_else(|| name.to_owned())
}

fn scoped_name(prefix: Option<&str>, name: &str) -> String {
    prefix
        .map(|prefix| format!("{prefix}__{name}"))
        .unwrap_or_else(|| name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::{lower_module, lower_modules};
    use crate::verilog::parser::{parse_module, parse_modules};

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

    #[test]
    fn lowers_vector_bit_selects_by_flattening_names() -> eyre::Result<()> {
        let module = parse_module(
            r#"
            module bit_xor(a, y);
              input [1:0] a;
              output y;
              assign y = a[0] ^ a[1];
            endmodule
            "#,
        )?;

        let graph = lower_module(&module)?;
        let table = graph.truth_table()?;

        assert_eq!(table.input_names, vec!["a_0", "a_1"]);
        assert_eq!(table.output_tables["y"], vec![false, true, true, false]);

        Ok(())
    }

    #[test]
    fn lowers_named_module_instances_by_inlining() -> eyre::Result<()> {
        let modules = parse_modules(
            r#"
            module half_adder(a, b, s, c);
              input a, b;
              output s, c;
              assign s = a ^ b;
              assign c = a & b;
            endmodule

            module two_half_adders(a, b, cin, sum, carry0, carry1);
              input a, b, cin;
              output sum, carry0, carry1;
              wire s0;
              half_adder ha0(.a(a), .b(b), .s(s0), .c(carry0));
              half_adder ha1(.a(s0), .b(cin), .s(sum), .c(carry1));
            endmodule
            "#,
        )?;

        let graph = lower_modules(&modules)?;
        let table = graph.truth_table()?;
        let sum = (0..8)
            .map(|mask: usize| mask.count_ones() % 2 == 1)
            .collect::<Vec<_>>();

        assert_eq!(table.output_tables["sum"], sum);

        Ok(())
    }
}
