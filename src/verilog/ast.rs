#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerilogModule {
    pub name: String,
    pub ports: Vec<String>,
    pub declarations: Vec<Declaration>,
    pub assignments: Vec<Assignment>,
    pub always_blocks: Vec<AlwaysBlock>,
    pub instances: Vec<Instance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    pub direction: Option<PortDirection>,
    pub range: Option<Range>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Range {
    pub msb: usize,
    pub lsb: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortDirection {
    Input,
    Output,
    OutputReg,
    Wire,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub output: String,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instance {
    pub module_name: String,
    pub instance_name: String,
    pub connections: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlwaysBlock {
    pub sensitivity: AlwaysSensitivity,
    pub body: AlwaysStmt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlwaysSensitivity {
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlwaysStmt {
    If {
        condition: String,
        then_branch: Box<AlwaysStmt>,
    },
    NonBlockingAssign {
        output: String,
        data: String,
    },
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
