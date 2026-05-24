# Verilog Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, testable Verilog frontend that lowers easy `assign`-based combinational modules into the existing `LogicGraph` pipeline before adding broader RTL features.

**Architecture:** Start with no external Verilog dependency: parse a deliberately small Verilog subset into a local AST, render expressions into fully parenthesized logic strings, and reuse `LogicGraph::from_stmt` plus a new multi-assignment helper. Keep parser, AST, and lowering in separate files so a real parser such as `sv-parser` can replace only the source parsing layer later.

**Tech Stack:** Rust 2021, existing `LogicGraph`, `Graph::merge`, `eyre`, `cargo test --release` for local placer and place-and-route tests.

---

## Scope

Build this in increasing order of value:

1. Multi-assignment combinational graph helper.
2. Verilog-friendly identifier support in the existing expression parser.
3. Tiny Verilog AST and source parser for scalar `module`, `input`, `output`, `wire`, and `assign`.
4. Verilog expression lowering for `~`, `&`, `^`, `|`, and parentheses with Verilog precedence.
5. CLI parse-and-prepare integration.
6. Small-gate NBT export path only after graph lowering is stable.
7. Bus flattening and module instances as follow-up work.

This plan intentionally excludes `always`, procedural assignments, parameters, generate blocks, signed arithmetic, memories, delays, tasks/functions, tri-state, and full SystemVerilog syntax.

## File Structure

- Modify `src/graph/logic.rs`: add a small public helper for merging multiple output assignments into one `LogicGraph`; extend identifier tokenization enough for common Verilog scalar names.
- Replace `src/verilog/mod.rs`: expose the frontend API and keep the old commented `sv_parser` stub out of the active path.
- Create `src/verilog/ast.rs`: define `VerilogModule`, declarations, assignments, and expression AST.
- Create `src/verilog/lexer.rs`: tokenize the supported source subset and strip comments.
- Create `src/verilog/parser.rs`: parse the supported subset into `VerilogModule`.
- Create `src/verilog/lower.rs`: convert `VerilogModule` into `LogicGraph`.
- Modify `src/main.rs`: call the frontend for `.v` files and report parse/prepare success before adding NBT output.
- Optional parser replacement path: modify `Cargo.toml` to add `sv-parser` only when the handwritten subset is not enough.

---

### Task 1: Add Multi-Assignment LogicGraph Helper

**Files:**
- Modify: `src/graph/logic.rs`

- [ ] **Step 1: Write the failing test**

Add this test inside `#[cfg(test)] mod tests` in `src/graph/logic.rs`.

```rust
#[test]
fn from_assignments_builds_half_adder() -> eyre::Result<()> {
    let graph = LogicGraph::from_assignments([
        ("s".to_owned(), "a^b".to_owned()),
        ("c".to_owned(), "a&b".to_owned()),
    ])?;
    let table = graph.truth_table()?;

    assert_eq!(table.input_names, vec!["a", "b"]);
    assert_eq!(table.output_tables["s"], vec![false, true, true, false]);
    assert_eq!(table.output_tables["c"], vec![false, false, false, true]);

    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test --release graph::logic::tests::from_assignments_builds_half_adder
```

Expected: fail to compile because `LogicGraph::from_assignments` does not exist.

- [ ] **Step 3: Implement the helper**

Add this method to `impl LogicGraph` in `src/graph/logic.rs`.

```rust
pub fn from_assignments<I>(assignments: I) -> eyre::Result<LogicGraph>
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut graphs = assignments
        .into_iter()
        .map(|(output, expr)| LogicGraph::from_stmt(&expr, &output))
        .collect::<eyre::Result<Vec<_>>>()?
        .into_iter();

    let Some(mut graph) = graphs.next() else {
        eyre::bail!("expected at least one logic assignment");
    };

    for next in graphs {
        graph.graph.merge(next.graph);
    }

    Ok(graph)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
cargo test --release graph::logic::tests::from_assignments_builds_half_adder
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/graph/logic.rs
git commit -m "Add multi-assignment logic graph helper"
```

---

### Task 2: Accept Common Verilog Identifiers In Logic Expressions

**Files:**
- Modify: `src/graph/logic.rs`

- [ ] **Step 1: Write the failing test**

Add this test inside `#[cfg(test)] mod tests` in `src/graph/logic.rs`.

```rust
#[test]
fn logic_parser_accepts_verilog_style_identifiers() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("A_0&carry_in", "SUM_0")?;
    let table = graph.truth_table()?;

    assert_eq!(table.input_names, vec!["A_0", "carry_in"]);
    assert_eq!(table.output_tables["SUM_0"], vec![false, false, false, true]);

    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test --release graph::logic::tests::logic_parser_accepts_verilog_style_identifiers
```

Expected: panic or fail while lexing `A_0`.

- [ ] **Step 3: Extend identifier scanning**

Replace the identifier branch in `LogicGraphBuilder::next` with helper predicates.

```rust
fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()
}
```

Then change the match arm from the lowercase-only branch to:

```rust
ch if is_ident_start(ch) => {
    let mut result = String::new();

    while self.stmt.len() != next_ptr
        && is_ident_continue(self.stmt.chars().nth(next_ptr).unwrap())
    {
        result.push(self.stmt.chars().nth(next_ptr).unwrap());
        next_ptr = self.next_ptr();
    }

    self.ptr -= 1;

    LogicStringTokenType::Ident(result)
}
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
cargo test --release graph::logic::tests
```

Expected: both pass.

- [ ] **Step 5: Commit**

```powershell
git add src/graph/logic.rs
git commit -m "Accept Verilog-style logic identifiers"
```

---

### Task 3: Define Minimal Verilog AST

**Files:**
- Create: `src/verilog/ast.rs`
- Modify: `src/verilog/mod.rs`

- [ ] **Step 1: Write the AST file**

Create `src/verilog/ast.rs`.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerilogModule {
    pub name: String,
    pub ports: Vec<String>,
    pub declarations: Vec<Declaration>,
    pub assignments: Vec<Assignment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    pub direction: Option<PortDirection>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortDirection {
    Input,
    Output,
    Wire,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub output: String,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Ident(String),
    Not(Box<Expr>),
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    And,
    Xor,
    Or,
}

impl Expr {
    pub fn to_logic_stmt(&self) -> String {
        match self {
            Expr::Ident(name) => name.clone(),
            Expr::Not(expr) => format!("~({})", expr.to_logic_stmt()),
            Expr::Binary { op, left, right } => {
                let op = match op {
                    BinaryOp::And => "&",
                    BinaryOp::Xor => "^",
                    BinaryOp::Or => "|",
                };
                format!("({}{}{})", left.to_logic_stmt(), op, right.to_logic_stmt())
            }
        }
    }
}
```

- [ ] **Step 2: Expose the module**

Replace `src/verilog/mod.rs` with:

```rust
pub mod ast;
```

- [ ] **Step 3: Add a unit test for expression rendering**

Add this to `src/verilog/ast.rs`.

```rust
#[cfg(test)]
mod tests {
    use super::{BinaryOp, Expr};

    #[test]
    fn expr_renders_fully_parenthesized_logic_stmt() {
        let expr = Expr::Binary {
            op: BinaryOp::Xor,
            left: Box::new(Expr::Ident("a".to_owned())),
            right: Box::new(Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(Expr::Ident("b".to_owned())),
                right: Box::new(Expr::Ident("c".to_owned())),
            }),
        };

        assert_eq!(expr.to_logic_stmt(), "(a^(b&c))");
    }
}
```

- [ ] **Step 4: Run AST test**

Run:

```powershell
cargo test --release verilog::ast::tests::expr_renders_fully_parenthesized_logic_stmt
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/verilog/mod.rs src/verilog/ast.rs
git commit -m "Add minimal Verilog AST"
```

---

### Task 4: Add Lexer For Supported Verilog Subset

**Files:**
- Create: `src/verilog/lexer.rs`
- Modify: `src/verilog/mod.rs`

- [ ] **Step 1: Write failing lexer test**

Create `src/verilog/lexer.rs` with token definitions and this test first.

```rust
#[cfg(test)]
mod tests {
    use super::{lex, Token};

    #[test]
    fn lexes_combinational_module_subset() -> eyre::Result<()> {
        let tokens = lex(
            r#"
            module half_adder(a, b, s, c);
              input a, b;
              output s, c;
              assign s = a ^ b;
              assign c = a & b;
            endmodule
            "#,
        )?;

        assert_eq!(
            tokens,
            vec![
                Token::Module,
                Token::Ident("half_adder".to_owned()),
                Token::LParen,
                Token::Ident("a".to_owned()),
                Token::Comma,
                Token::Ident("b".to_owned()),
                Token::Comma,
                Token::Ident("s".to_owned()),
                Token::Comma,
                Token::Ident("c".to_owned()),
                Token::RParen,
                Token::Semi,
                Token::Input,
                Token::Ident("a".to_owned()),
                Token::Comma,
                Token::Ident("b".to_owned()),
                Token::Semi,
                Token::Output,
                Token::Ident("s".to_owned()),
                Token::Comma,
                Token::Ident("c".to_owned()),
                Token::Semi,
                Token::Assign,
                Token::Ident("s".to_owned()),
                Token::Eq,
                Token::Ident("a".to_owned()),
                Token::Xor,
                Token::Ident("b".to_owned()),
                Token::Semi,
                Token::Assign,
                Token::Ident("c".to_owned()),
                Token::Eq,
                Token::Ident("a".to_owned()),
                Token::And,
                Token::Ident("b".to_owned()),
                Token::Semi,
                Token::EndModule,
            ]
        );

        Ok(())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test --release verilog::lexer::tests::lexes_combinational_module_subset
```

Expected: fail because `lexer` is not exported or `lex` is not implemented.

- [ ] **Step 3: Implement tokenization**

Add this implementation to `src/verilog/lexer.rs`.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Module,
    EndModule,
    Input,
    Output,
    Wire,
    Assign,
    Ident(String),
    LParen,
    RParen,
    Comma,
    Semi,
    Eq,
    Not,
    And,
    Xor,
    Or,
}

pub fn lex(source: &str) -> eyre::Result<Vec<Token>> {
    let chars = strip_comments(source).chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() {
            index += 1;
            continue;
        }

        match ch {
            '(' => tokens.push(Token::LParen),
            ')' => tokens.push(Token::RParen),
            ',' => tokens.push(Token::Comma),
            ';' => tokens.push(Token::Semi),
            '=' => tokens.push(Token::Eq),
            '~' => tokens.push(Token::Not),
            '&' => tokens.push(Token::And),
            '^' => tokens.push(Token::Xor),
            '|' => tokens.push(Token::Or),
            ch if is_ident_start(ch) => {
                let start = index;
                index += 1;
                while index < chars.len() && is_ident_continue(chars[index]) {
                    index += 1;
                }
                let text = chars[start..index].iter().collect::<String>();
                tokens.push(match text.as_str() {
                    "module" => Token::Module,
                    "endmodule" => Token::EndModule,
                    "input" => Token::Input,
                    "output" => Token::Output,
                    "wire" => Token::Wire,
                    "assign" => Token::Assign,
                    _ => Token::Ident(text),
                });
                continue;
            }
            _ => eyre::bail!("unsupported Verilog character `{ch}` at byte-like index {index}"),
        }

        index += 1;
    }

    Ok(tokens)
}

fn strip_comments(source: &str) -> String {
    let mut result = String::new();
    let mut chars = source.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\n' {
                    result.push('\n');
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()
}
```

- [ ] **Step 4: Export lexer**

Update `src/verilog/mod.rs`.

```rust
pub mod ast;
pub mod lexer;
```

- [ ] **Step 5: Run lexer test**

Run:

```powershell
cargo test --release verilog::lexer::tests::lexes_combinational_module_subset
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/verilog/mod.rs src/verilog/lexer.rs
git commit -m "Add minimal Verilog lexer"
```

---

### Task 5: Parse Scalar Module Declarations And Assignments

**Files:**
- Create: `src/verilog/parser.rs`
- Modify: `src/verilog/mod.rs`

- [ ] **Step 1: Write parser tests**

Create `src/verilog/parser.rs` and add these tests first.

```rust
#[cfg(test)]
mod tests {
    use super::parse_module;
    use crate::verilog::ast::{BinaryOp, Expr, PortDirection};

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
        assert_eq!(module.declarations[0].direction, Some(PortDirection::Input));
        assert_eq!(module.declarations[1].direction, Some(PortDirection::Output));
        assert_eq!(module.declarations[2].direction, Some(PortDirection::Wire));
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
}
```

- [ ] **Step 2: Run parser tests to verify they fail**

Run:

```powershell
cargo test --release verilog::parser::tests
```

Expected: fail because parser is not implemented.

- [ ] **Step 3: Implement parser skeleton**

Use `crate::verilog::lexer::lex` and a cursor over `Vec<Token>`. The public entry point should be:

```rust
pub fn parse_module(source: &str) -> eyre::Result<VerilogModule> {
    Parser::new(lex(source)?).parse_module()
}
```

Define:

```rust
struct Parser {
    tokens: Vec<Token>,
    index: usize,
}
```

Add methods:

```rust
impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, index: 0 }
    }

    fn parse_module(&mut self) -> eyre::Result<VerilogModule> {
        self.expect(Token::Module)?;
        let name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let ports = self.parse_ident_list()?;
        self.expect(Token::RParen)?;
        self.expect(Token::Semi)?;

        let mut declarations = Vec::new();
        let mut assignments = Vec::new();
        while !self.consume(&Token::EndModule) {
            match self.peek() {
                Some(Token::Input) | Some(Token::Output) | Some(Token::Wire) => {
                    declarations.push(self.parse_declaration()?);
                }
                Some(Token::Assign) => assignments.push(self.parse_assignment()?),
                Some(token) => eyre::bail!("unsupported Verilog token in module body: {token:?}"),
                None => eyre::bail!("expected endmodule"),
            }
        }

        Ok(VerilogModule {
            name,
            ports,
            declarations,
            assignments,
        })
    }
}
```

- [ ] **Step 4: Implement declarations, assignments, and expressions**

Expression precedence must be:

```text
primary / unary ~
&
^
|
```

Implement methods with these signatures:

```rust
fn parse_declaration(&mut self) -> eyre::Result<Declaration>;
fn parse_assignment(&mut self) -> eyre::Result<Assignment>;
fn parse_expr(&mut self) -> eyre::Result<Expr>;
fn parse_or(&mut self) -> eyre::Result<Expr>;
fn parse_xor(&mut self) -> eyre::Result<Expr>;
fn parse_and(&mut self) -> eyre::Result<Expr>;
fn parse_unary(&mut self) -> eyre::Result<Expr>;
fn parse_primary(&mut self) -> eyre::Result<Expr>;
```

Use left-associative binary construction:

```rust
expr = Expr::Binary {
    op: BinaryOp::And,
    left: Box::new(expr),
    right: Box::new(rhs),
};
```

- [ ] **Step 5: Export parser**

Update `src/verilog/mod.rs`.

```rust
pub mod ast;
pub mod lexer;
pub mod parser;
```

- [ ] **Step 6: Run parser tests**

Run:

```powershell
cargo test --release verilog::parser::tests
```

Expected: pass.

- [ ] **Step 7: Commit**

```powershell
git add src/verilog/mod.rs src/verilog/parser.rs
git commit -m "Parse minimal combinational Verilog modules"
```

---

### Task 6: Lower Parsed Verilog To LogicGraph

**Files:**
- Create: `src/verilog/lower.rs`
- Modify: `src/verilog/mod.rs`

- [ ] **Step 1: Write lowering tests**

Create `src/verilog/lower.rs` and add these tests first.

```rust
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
              assign sum = (a ^ b) ^ cin;
              assign cout = (a & b) | (sum & cin);
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
```

- [ ] **Step 2: Run lowering tests to verify they fail**

Run:

```powershell
cargo test --release verilog::lower::tests
```

Expected: fail because `lower_module` is not implemented or not exported.

- [ ] **Step 3: Implement lower_module**

Add:

```rust
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
```

- [ ] **Step 4: Export lower and public helpers**

Update `src/verilog/mod.rs`.

```rust
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
```

- [ ] **Step 5: Run lowering tests**

Run:

```powershell
cargo test --release verilog::lower::tests
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/verilog/mod.rs src/verilog/lower.rs
git commit -m "Lower minimal Verilog to logic graph"
```

---

### Task 7: Add Unsupported-Construct Diagnostics

**Files:**
- Modify: `src/verilog/lexer.rs`
- Modify: `src/verilog/parser.rs`

- [ ] **Step 1: Write tests for clear failures**

Add these tests to `src/verilog/parser.rs`.

```rust
#[test]
fn rejects_always_blocks_with_clear_message() {
    let error = parse_module(
        r#"
        module bad(clk, q);
          input clk;
          output q;
          always @(posedge clk) q = ~q;
        endmodule
        "#,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unsupported"));
}

#[test]
fn rejects_vector_declarations_with_clear_message() {
    let error = parse_module(
        r#"
        module bad(a, y);
          input [3:0] a;
          output y;
          assign y = a;
        endmodule
        "#,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unsupported"));
}
```

- [ ] **Step 2: Run tests to see current failure behavior**

Run:

```powershell
cargo test --release verilog::parser::tests
```

Expected: fail if the lexer reports a confusing character error or parser misses the unsupported construct.

- [ ] **Step 3: Add keyword tokens for common rejected constructs**

In `src/verilog/lexer.rs`, add token variants:

```rust
Always,
At,
LBracket,
RBracket,
Colon,
```

Map:

```rust
"always" => Token::Always,
"posedge" => Token::Ident("posedge".to_owned()),
```

And character tokens:

```rust
'@' => tokens.push(Token::At),
'[' => tokens.push(Token::LBracket),
']' => tokens.push(Token::RBracket),
':' => tokens.push(Token::Colon),
```

- [ ] **Step 4: Reject unsupported constructs in parser**

In module-body parsing, add explicit branches:

```rust
Some(Token::Always) => eyre::bail!("unsupported Verilog construct: always block"),
Some(Token::LBracket) => eyre::bail!("unsupported Verilog construct: vector declaration"),
```

In `parse_declaration`, reject a bracket immediately after the direction:

```rust
if matches!(self.peek(), Some(Token::LBracket)) {
    eyre::bail!("unsupported Verilog construct: vector declaration");
}
```

- [ ] **Step 5: Run diagnostics tests**

Run:

```powershell
cargo test --release verilog::parser::tests
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/verilog/lexer.rs src/verilog/parser.rs
git commit -m "Report unsupported Verilog constructs clearly"
```

---

### Task 8: Wire Verilog Graph Loading Into CLI

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Write or run a manual fixture**

Use existing `test/alu.v` only after fixing it to valid continuous assignments in a separate task. For this task, use a temporary valid file:

```verilog
module half_adder(a, b, s, c);
  input a, b;
  output s, c;
  assign s = a ^ b;
  assign c = a & b;
endmodule
```

- [ ] **Step 2: Implement parse-and-prepare path**

Update `main`:

```rust
fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let opt = CompilerOption::from_args();

    if opt.input.extension().and_then(|ext| ext.to_str()) == Some("v") {
        let graph = redstone_compiler::verilog::load_logic_graph(&opt.input)?;
        let prepared = graph.prepare_place()?;
        println!(
            "loaded Verilog graph: nodes={} inputs={} outputs={}",
            prepared.nodes.len(),
            prepared.inputs().len(),
            prepared.outputs().len()
        );
        return Ok(());
    }

    eyre::bail!("unsupported input file extension: {:?}", opt.input);
}
```

If `main.rs` cannot refer to the library as `redstone_compiler` in this package layout, use:

```rust
use redstone_compiler::verilog;
```

and call:

```rust
let graph = verilog::load_logic_graph(&opt.input)?;
```

- [ ] **Step 3: Run CLI against the temporary fixture**

Run:

```powershell
cargo run --release -- test/half-adder-verilog-smoke.v
```

Expected output contains:

```text
loaded Verilog graph:
```

- [ ] **Step 4: Add a checked-in smoke fixture**

Create `test/half-adder.v`:

```verilog
module half_adder(a, b, s, c);
  input a, b;
  output s, c;

  assign s = a ^ b;
  assign c = a & b;
endmodule
```

Run:

```powershell
cargo run --release -- test/half-adder.v
```

Expected: same parse-and-prepare success output.

- [ ] **Step 5: Commit**

```powershell
git add src/main.rs test/half-adder.v
git commit -m "Load minimal Verilog from CLI"
```

---

### Task 9: Add Tiny-Gate NBT Export From Verilog

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Keep export limited to tiny graphs**

Do not try full adder NBT export in this task. Start with `and` or `half_adder`; `full_adder` placement is already a separate local placer quality problem.

- [ ] **Step 2: Add export path when output is provided**

In `main.rs`, after `prepare_place()`, if `opt.output` is `Some(path)`, run local placement and save the first sampled world:

```rust
use redstone_compiler::nbt::ToNBT;
use redstone_compiler::transform::place_and_route::local_placer::{
    InputPlacementStrategy, LocalPlacer, LocalPlacerConfig, NotRouteStrategy,
    PlacementSamplingPolicy, SamplingPolicy, TorchPlacementStrategy,
};
use redstone_compiler::world::position::DimSize;
```

Use a conservative config copied from the existing AND/half-adder component tests:

```rust
let config = LocalPlacerConfig {
    random_seed: 42,
    greedy_input_generation: true,
    input_placement_strategy: InputPlacementStrategy::Boundary,
    input_candidate_limit: None,
    step_sampling_policy: SamplingPolicy::Random(10000),
    placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
    leak_sampling: false,
    route_torch_directly: true,
    torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
    not_route_strategy: NotRouteStrategy::DirectOnly,
    max_not_route_step: 3,
    not_route_step_sampling_policy: SamplingPolicy::Random(100),
    max_route_step: 3,
    route_step_sampling_policy: SamplingPolicy::Random(100),
};
let placer = LocalPlacer::new(prepared.clone(), config)?;
let worlds = placer.generate(DimSize(10, 10, 5), None);
let Some(world) = worlds.into_iter().next() else {
    eyre::bail!("placement produced no worlds");
};
world.to_nbt().save(output);
```

- [ ] **Step 3: Run export smoke test**

Run:

```powershell
cargo run --release -- test/half-adder.v test/half-adder-generated-from-verilog.nbt
```

Expected: command exits successfully and creates `test/half-adder-generated-from-verilog.nbt`.

- [ ] **Step 4: Verify generated NBT can be read**

Run:

```powershell
cargo run --release --bin check_nbt_world_cycle -- test/half-adder-generated-from-verilog.nbt
```

Expected: command exits successfully. If equivalence checking for this filename is not recognized by the binary, success here only proves the NBT is readable; graph equivalence should be added as a separate test.

- [ ] **Step 5: Commit**

```powershell
git add src/main.rs test/half-adder-generated-from-verilog.nbt
git commit -m "Export tiny Verilog circuits to NBT"
```

---

### Task 10: Add Bus Flattening As The First Real Extension

**Files:**
- Modify: `src/verilog/ast.rs`
- Modify: `src/verilog/lexer.rs`
- Modify: `src/verilog/parser.rs`
- Modify: `src/verilog/lower.rs`

- [ ] **Step 1: Add test for vector declarations and bit selects**

Add parser/lowering test:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cargo test --release verilog::lower::tests::lowers_vector_bit_selects_by_flattening_names
```

Expected: fail because vectors are rejected.

- [ ] **Step 3: Extend AST declarations**

Change `Declaration`:

```rust
pub struct Declaration {
    pub direction: Option<PortDirection>,
    pub range: Option<Range>,
    pub names: Vec<String>,
}

pub struct Range {
    pub msb: usize,
    pub lsb: usize,
}
```

- [ ] **Step 4: Parse ranges and bit selects**

Support only numeric `[msb:lsb]` declarations and single-bit `name[index]` references. Lower bit select names to `format!("{name}_{index}")`.

- [ ] **Step 5: Run vector test**

Run:

```powershell
cargo test --release verilog::lower::tests::lowers_vector_bit_selects_by_flattening_names
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/verilog/ast.rs src/verilog/lexer.rs src/verilog/parser.rs src/verilog/lower.rs
git commit -m "Flatten simple Verilog bit selects"
```

---

### Task 11: Add Named Module Instance Lowering

**Files:**
- Modify: `src/verilog/ast.rs`
- Modify: `src/verilog/parser.rs`
- Modify: `src/verilog/lower.rs`
- Consider: `src/graph/module.rs`

- [ ] **Step 1: Add test for two half adders composed structurally**

Use a source with two modules:

```verilog
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
```

The lowering test should assert the `sum` truth table matches `a ^ b ^ cin`.

- [ ] **Step 2: Parse multiple modules**

Add:

```rust
pub fn parse_modules(source: &str) -> eyre::Result<Vec<VerilogModule>>;
```

Keep `parse_module` as a wrapper that requires exactly one module.

- [ ] **Step 3: Parse named instances only**

Add AST:

```rust
pub struct Instance {
    pub module_name: String,
    pub instance_name: String,
    pub connections: Vec<(String, String)>,
}
```

Reject positional instances with `unsupported Verilog construct: positional instance ports`.

- [ ] **Step 4: Lower instances by inlining**

For each instance, clone the child module assignments and rename local signals with `instance_name__signal`. Apply named port substitutions before calling `LogicGraph::from_assignments`.

- [ ] **Step 5: Run instance test**

Run:

```powershell
cargo test --release verilog::lower::tests::lowers_named_module_instances_by_inlining
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/verilog/ast.rs src/verilog/parser.rs src/verilog/lower.rs
git commit -m "Inline simple named Verilog module instances"
```

---

## Verification Commands

Run focused tests after each task. At the end of Tasks 1 through 8, run:

```powershell
cargo test --release graph::logic::tests
cargo test --release verilog::
```

Before claiming the frontend is ready for local placement, run:

```powershell
cargo test --release
```

This project explicitly prefers `cargo test --release` because local placer and place-and-route tests are too slow in debug builds.

## Implementation Notes

- Keep Verilog lowering combinational until sequential semantics are explicit.
- Render parsed expressions as fully parenthesized strings before passing them to `LogicGraph::from_stmt`; this avoids relying on the current expression parser's precedence.
- Treat clear unsupported diagnostics as a feature. A narrow frontend that fails clearly is better than silently accepting RTL it cannot lower correctly.
- Do not use full adder NBT export as the first success criterion. Graph lowering for full adder is appropriate early; physical placement for full adder depends on ongoing local placer quality work.
- Add `sv-parser` only when the handcrafted subset becomes the bottleneck. The AST and lowering split in this plan makes that replacement local to parsing.
