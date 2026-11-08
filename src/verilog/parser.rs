use crate::verilog::ast::{
    AlwaysBlock, AlwaysSensitivity, AlwaysStmt, Assignment, BinaryOp, Declaration, Expr, Instance,
    PortDirection, Range, VerilogModule,
};
use crate::verilog::lexer::{lex, Token};

pub fn parse_module(source: &str) -> eyre::Result<VerilogModule> {
    let mut modules = parse_modules(source)?;
    if modules.len() != 1 {
        eyre::bail!("expected exactly one module, got {}", modules.len());
    }
    Ok(modules.remove(0))
}

pub fn parse_modules(source: &str) -> eyre::Result<Vec<VerilogModule>> {
    Parser::new(lex(source)?).parse_modules()
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, index: 0 }
    }

    fn parse_modules(&mut self) -> eyre::Result<Vec<VerilogModule>> {
        let mut modules = Vec::new();
        while self.peek().is_some() {
            modules.push(self.parse_one_module()?);
        }
        Ok(modules)
    }

    fn parse_one_module(&mut self) -> eyre::Result<VerilogModule> {
        self.expect(Token::Module)?;
        let name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let ports = self.parse_ident_list()?;
        self.expect(Token::RParen)?;
        self.expect(Token::Semi)?;

        let mut declarations = Vec::new();
        let mut assignments = Vec::new();
        let mut always_blocks = Vec::new();
        let mut instances = Vec::new();
        while !self.consume(&Token::EndModule) {
            match self.peek() {
                Some(Token::Input) | Some(Token::Output) | Some(Token::Wire) => {
                    declarations.push(self.parse_declaration()?);
                }
                Some(Token::Assign) => assignments.push(self.parse_assignment()?),
                Some(Token::Ident(_)) => instances.push(self.parse_instance()?),
                Some(Token::Always) => always_blocks.push(self.parse_always_block()?),
                Some(token) => eyre::bail!("unsupported Verilog token in module body: {token:?}"),
                None => eyre::bail!("expected endmodule"),
            }
        }

        Ok(VerilogModule {
            name,
            ports,
            declarations,
            assignments,
            always_blocks,
            instances,
        })
    }

    fn parse_declaration(&mut self) -> eyre::Result<Declaration> {
        let direction = match self.next() {
            Some(Token::Input) => PortDirection::Input,
            Some(Token::Output) => {
                if self.consume(&Token::Reg) {
                    PortDirection::OutputReg
                } else {
                    PortDirection::Output
                }
            }
            Some(Token::Wire) => PortDirection::Wire,
            Some(token) => eyre::bail!("expected declaration direction, got {token:?}"),
            None => eyre::bail!("expected declaration direction"),
        };
        let range = self.parse_optional_range()?;
        let names = self.parse_ident_list()?;
        self.expect(Token::Semi)?;

        Ok(Declaration {
            direction: Some(direction),
            range,
            names,
        })
    }

    fn parse_assignment(&mut self) -> eyre::Result<Assignment> {
        self.expect(Token::Assign)?;
        let output = self.expect_ident()?;
        self.expect(Token::Eq)?;
        let expr = self.parse_expr()?;
        self.expect(Token::Semi)?;

        Ok(Assignment { output, expr })
    }

    fn parse_always_block(&mut self) -> eyre::Result<AlwaysBlock> {
        self.expect(Token::Always)?;
        let sensitivity = self.parse_always_sensitivity()?;
        let body = self.parse_always_stmt()?;

        Ok(AlwaysBlock { sensitivity, body })
    }

    fn parse_always_sensitivity(&mut self) -> eyre::Result<AlwaysSensitivity> {
        self.expect(Token::At)?;
        self.expect(Token::LParen)?;
        if self.consume(&Token::Star) {
            self.expect(Token::RParen)?;
            return Ok(AlwaysSensitivity::Any);
        }

        if self.consume(&Token::Posedge) {
            let clock = self.parse_signal_name()?;
            self.expect(Token::RParen)?;
            return Ok(AlwaysSensitivity::Posedge(clock));
        }

        let got = self.peek().cloned();
        eyre::bail!("unsupported always sensitivity: {got:?}")
    }

    fn parse_always_stmt(&mut self) -> eyre::Result<AlwaysStmt> {
        self.expect(Token::Begin)?;
        let stmt = self.parse_always_stmt_inner()?;
        self.expect(Token::End)?;
        Ok(stmt)
    }

    fn parse_always_stmt_inner(&mut self) -> eyre::Result<AlwaysStmt> {
        // TODO: Extend this to parse a real Verilog procedural statement list.
        // For now, the design lower only consumes a single `if (...) begin ... end`
        // latch pattern or a single nonblocking assignment.
        if self.peek() == Some(&Token::If) {
            return self.parse_always_if_stmt();
        }

        self.parse_nonblocking_assignment_stmt()
    }

    fn parse_always_if_stmt(&mut self) -> eyre::Result<AlwaysStmt> {
        self.expect(Token::If)?;
        self.expect(Token::LParen)?;
        let condition = self.parse_signal_name()?;
        self.expect(Token::RParen)?;
        self.expect(Token::Begin)?;
        let then_branch = self.parse_always_stmt_inner()?;
        self.expect(Token::End)?;

        Ok(AlwaysStmt::If {
            condition,
            then_branch: Box::new(then_branch),
        })
    }

    fn parse_nonblocking_assignment_stmt(&mut self) -> eyre::Result<AlwaysStmt> {
        let output = self.parse_signal_name()?;
        self.expect(Token::Le)?;
        let data = self.parse_expr()?;
        self.expect(Token::Semi)?;
        Ok(AlwaysStmt::NonBlockingAssign { output, data })
    }

    fn parse_instance(&mut self) -> eyre::Result<Instance> {
        let module_name = self.expect_ident()?;
        let instance_name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let connections = self.parse_named_connections()?;
        self.expect(Token::RParen)?;
        self.expect(Token::Semi)?;

        Ok(Instance {
            module_name,
            instance_name,
            connections,
        })
    }

    fn parse_named_connections(&mut self) -> eyre::Result<Vec<(String, Expr)>> {
        let mut connections = vec![self.parse_named_connection()?];
        while self.consume(&Token::Comma) {
            connections.push(self.parse_named_connection()?);
        }
        Ok(connections)
    }

    fn parse_named_connection(&mut self) -> eyre::Result<(String, Expr)> {
        if !self.consume(&Token::Dot) {
            eyre::bail!("unsupported Verilog construct: positional instance ports");
        }
        let port = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let signal = self.parse_expr()?;
        self.expect(Token::RParen)?;

        Ok((port, signal))
    }

    fn parse_expr(&mut self) -> eyre::Result<Expr> {
        let select = self.parse_or()?;
        if self.consume(&Token::Question) {
            let when_true = self.parse_expr()?;
            self.expect(Token::Colon)?;
            let when_false = self.parse_expr()?;
            Ok(Expr::Mux {
                select: Box::new(select),
                when_true: Box::new(when_true),
                when_false: Box::new(when_false),
            })
        } else {
            Ok(select)
        }
    }

    fn parse_or(&mut self) -> eyre::Result<Expr> {
        let mut expr = self.parse_xor()?;
        while self.consume(&Token::Or) {
            let rhs = self.parse_xor()?;
            expr = Expr::Binary {
                op: BinaryOp::Or,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_xor(&mut self) -> eyre::Result<Expr> {
        let mut expr = self.parse_and()?;
        while self.consume(&Token::Xor) {
            let rhs = self.parse_and()?;
            expr = Expr::Binary {
                op: BinaryOp::Xor,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> eyre::Result<Expr> {
        let mut expr = self.parse_equal()?;
        while self.consume(&Token::And) {
            let rhs = self.parse_equal()?;
            expr = Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_equal(&mut self) -> eyre::Result<Expr> {
        let mut expr = self.parse_compare()?;
        while self.consume(&Token::Equal) {
            let right = self.parse_compare()?;
            expr = Expr::Binary {
                op: BinaryOp::Equal,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_compare(&mut self) -> eyre::Result<Expr> {
        let mut expr = self.parse_add()?;
        while self.consume(&Token::GreaterEqual) {
            let right = self.parse_add()?;
            expr = Expr::Binary {
                op: BinaryOp::GreaterEqual,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_add(&mut self) -> eyre::Result<Expr> {
        let mut expr = self.parse_unary()?;
        loop {
            let op = if self.consume(&Token::Plus) {
                BinaryOp::Add
            } else if self.consume(&Token::Minus) {
                BinaryOp::Sub
            } else {
                break;
            };
            let rhs = self.parse_unary()?;
            expr = Expr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> eyre::Result<Expr> {
        if self.consume(&Token::Not) {
            return Ok(Expr::Not(Box::new(self.parse_unary()?)));
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> eyre::Result<Expr> {
        match self.next() {
            Some(Token::Ident(name)) => {
                let mut value = Expr::Ident(name);
                if self.consume(&Token::LBracket) {
                    let msb = self.expect_number()?;
                    let lsb = if self.consume(&Token::Colon) {
                        self.expect_number()?
                    } else {
                        msb
                    };
                    self.expect(Token::RBracket)?;
                    value = Expr::Select {
                        value: Box::new(value),
                        msb,
                        lsb,
                    };
                }
                Ok(value)
            }
            Some(Token::Number(value)) => Ok(Expr::Number(value)),
            Some(Token::SizedNumber { value, width }) => Ok(Expr::SizedNumber { value, width }),
            Some(Token::LBrace) => {
                let mut parts = vec![self.parse_expr()?];
                while self.consume(&Token::Comma) {
                    parts.push(self.parse_expr()?);
                }
                self.expect(Token::RBrace)?;
                Ok(Expr::Concat(parts))
            }
            Some(Token::LParen) => {
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Some(token) => eyre::bail!("expected expression, got {token:?}"),
            None => eyre::bail!("expected expression"),
        }
    }

    fn parse_ident_list(&mut self) -> eyre::Result<Vec<String>> {
        let mut names = vec![self.expect_ident()?];
        while self.consume(&Token::Comma) {
            names.push(self.expect_ident()?);
        }
        Ok(names)
    }

    fn parse_signal_name(&mut self) -> eyre::Result<String> {
        let name = self.expect_ident()?;
        eyre::ensure!(self.peek() != Some(&Token::LBracket),
            "static selects are currently supported only in expressions, not procedural destinations, conditions, or clocks");
        Ok(name)
    }

    fn parse_optional_range(&mut self) -> eyre::Result<Option<Range>> {
        if !self.consume(&Token::LBracket) {
            return Ok(None);
        }

        let msb = self.expect_number()?;
        self.expect(Token::Colon)?;
        let lsb = self.expect_number()?;
        self.expect(Token::RBracket)?;

        Ok(Some(Range { msb, lsb }))
    }

    fn expect_ident(&mut self) -> eyre::Result<String> {
        match self.next() {
            Some(Token::Ident(name)) => Ok(name),
            Some(token) => eyre::bail!("expected identifier, got {token:?}"),
            None => eyre::bail!("expected identifier"),
        }
    }

    fn expect_number(&mut self) -> eyre::Result<usize> {
        match self.next() {
            Some(Token::Number(value)) => Ok(value),
            Some(token) => eyre::bail!("expected number, got {token:?}"),
            None => eyre::bail!("expected number"),
        }
    }

    fn expect(&mut self, expected: Token) -> eyre::Result<()> {
        let got = self.next();
        if got == Some(expected.clone()) {
            return Ok(());
        }

        eyre::bail!("expected {expected:?}, got {got:?}")
    }

    fn consume(&mut self, expected: &Token) -> bool {
        if self.peek() == Some(expected) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.index).cloned()?;
        self.index += 1;
        Some(token)
    }
}

#[cfg(test)]
mod tests {
    use super::parse_module;
    use crate::verilog::ast::{AlwaysSensitivity, AlwaysStmt, BinaryOp, Expr, PortDirection};

    #[test]
    fn rejects_procedural_select_instead_of_aliasing_another_signal() {
        assert!(parse_module("module m(clk,a,a_0,q); input clk; input [1:0] a; input a_0; output reg q; always @(posedge clk) begin if (a[0]) begin q <= a_0; end end endmodule").is_err());
    }

    #[test]
    fn named_input_port_bindings_preserve_expressions() -> eyre::Result<()> {
        let module = parse_module("module m(a,y); input [7:0] a; output [3:0] y; child u(.x({1'b0,a[6:4]}), .y(y)); endmodule")?;
        assert!(
            matches!(&module.instances[0].connections[0].1, Expr::Concat(parts) if parts.len() == 2)
        );
        assert!(matches!(&module.instances[0].connections[1].1, Expr::Ident(name) if name == "y"));
        Ok(())
    }

    #[test]
    fn parses_right_associative_mux_and_real_bus_select() -> eyre::Result<()> {
        let module = parse_module("module m(a,y); input [3:0] a; output [6:0] y; assign y = a == 4'd0 ? 7'h3f : a[0] ? 7'h06 : 7'h00; endmodule")?;
        let Expr::Mux {
            select, when_false, ..
        } = &module.assignments[0].expr
        else {
            panic!("expected mux")
        };
        assert!(matches!(
            select.as_ref(),
            Expr::Binary {
                op: BinaryOp::Equal,
                ..
            }
        ));
        let Expr::Mux { select, .. } = when_false.as_ref() else {
            panic!("expected right associative mux")
        };
        assert!(matches!(
            select.as_ref(),
            Expr::Select { msb: 0, lsb: 0, .. }
        ));
        Ok(())
    }

    #[test]
    fn parses_half_adder_module() -> eyre::Result<()> {
        let module = parse_module(
            r#"
            module half_adder(a, b, s, c);
              input a, b;
              output s, c;
              wire tmp;
              assign s = a ^ b;
              assign c = a & b;
            endmodule
            "#,
        )?;

        assert_eq!(module.name, "half_adder");
        assert_eq!(module.ports, vec!["a", "b", "s", "c"]);
        assert_eq!(
            module.declarations[0].direction.as_ref(),
            Some(&PortDirection::Input)
        );
        assert_eq!(
            module.declarations[1].direction.as_ref(),
            Some(&PortDirection::Output)
        );
        assert_eq!(
            module.declarations[2].direction.as_ref(),
            Some(&PortDirection::Wire)
        );
        assert_eq!(module.assignments.len(), 2);
        assert_eq!(module.assignments[0].output, "s");
        assert_eq!(
            module.assignments[0].expr,
            Expr::Binary {
                op: BinaryOp::Xor,
                left: Box::new(Expr::Ident("a".to_owned())),
                right: Box::new(Expr::Ident("b".to_owned())),
            }
        );

        Ok(())
    }

    #[test]
    fn parses_verilog_operator_precedence() -> eyre::Result<()> {
        let module = parse_module(
            r#"
            module precedence(a, b, c, y);
              input a, b, c;
              output y;
              assign y = a ^ b & c;
            endmodule
            "#,
        )?;

        assert_eq!(module.assignments[0].expr.to_logic_stmt(), "(a^(b&c))");

        Ok(())
    }

    #[test]
    fn parses_d_latch_always_subset() -> eyre::Result<()> {
        let module = parse_module(
            r#"
            module d_latch(d, en, q);
              input d, en;
              output reg q;
              always @(*) begin
                if (en) begin
                  q <= d;
                end
              end
            endmodule
            "#,
        )?;

        assert_eq!(
            module.declarations[1].direction,
            Some(PortDirection::OutputReg)
        );
        assert_eq!(module.always_blocks.len(), 1);
        assert_eq!(module.always_blocks[0].sensitivity, AlwaysSensitivity::Any);
        assert_eq!(
            module.always_blocks[0].body,
            AlwaysStmt::If {
                condition: "en".to_owned(),
                then_branch: Box::new(AlwaysStmt::NonBlockingAssign {
                    output: "q".to_owned(),
                    data: Expr::Ident("d".to_owned())
                })
            }
        );

        Ok(())
    }

    #[test]
    fn parses_vector_declarations() -> eyre::Result<()> {
        let module = parse_module(
            r#"
            module vectors(a, y);
              input [3:0] a;
              output y;
              assign y = a[0];
            endmodule
            "#,
        )?;

        let range = module.declarations[0]
            .range
            .as_ref()
            .expect("expected parsed vector range");
        assert_eq!(range.msb, 3);
        assert_eq!(range.lsb, 0);
        assert_eq!(module.assignments[0].expr.to_logic_stmt(), "a_0");

        Ok(())
    }
}
