//! Syntax tree of rsdsl v2.

use crate::diag::Span;

/// A binder name, shared cheaply with the grounder's environment.
pub type Name = std::sync::Arc<str>;

#[derive(Debug, Clone)]
pub struct File {
    pub header: Header,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Header {
    Model { name: String },
    Instance { name: String, of: String },
}

#[derive(Debug, Clone)]
pub struct Annotation {
    pub name: String,
    pub args: Vec<AnnArg>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum AnnArg {
    Named(String, Expr),
    Pos(Expr),
}

#[derive(Debug, Clone)]
pub struct Item {
    pub annotations: Vec<Annotation>,
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ItemKind {
    Include(String),
    Grid {
        name: String,
        axes: [String; 3],
        dirs: String,
        directions: Vec<(String, i8, usize)>,
    },
    GridValue {
        name: String,
        dims: [i64; 3],
    },
    Enum {
        name: String,
        variants: Vec<String>,
    },
    Subset {
        name: String,
        of: String,
        values: Expr,
    },
    Domain {
        name: String,
        ty: Option<TypeExpr>,
        value: Option<Expr>,
    },
    Fact {
        name: String,
        params: Vec<Param>,
        derived: Option<Expr>,
    },
    FactValue {
        name: String,
        value: Expr,
    },
    Param {
        name: String,
        ty: TypeExpr,
        default: Option<Expr>,
    },
    ParamValue {
        name: String,
        value: Expr,
    },
    Fn {
        name: String,
        params: Vec<Param>,
        ret: TypeExpr,
        body: Expr,
    },
    Choice {
        name: String,
        index: Vec<Binder>,
        members: Vec<Member>,
    },
    Var {
        name: String,
        index: VarIndex,
    },
    Def {
        name: String,
        index: Vec<Binder>,
        body: Expr,
    },
    Relation {
        name: String,
        params: Vec<Param>,
    },
    Int {
        name: String,
        index: Vec<Binder>,
        range: Range,
    },
    Rule {
        label: String,
        body: Vec<Stmt>,
    },
    Objective {
        minimize: bool,
        expr: Expr,
    },
}

#[derive(Debug, Clone)]
pub enum VarIndex {
    Binders(Vec<Binder>),
    Over(String),
}

#[derive(Debug, Clone)]
pub struct Member {
    pub name: String,
    pub params: Vec<Param>,
    pub guard: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: Option<String>,
    pub ty: TypeExpr,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Named(String, Span),
    Int,
    Bool,
    Set(Box<TypeExpr>),
    Option(Box<TypeExpr>),
    Tuple(Vec<TypeExpr>),
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub annotations: Vec<Annotation>,
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Require(Expr),
    Forall {
        binders: Vec<Binder>,
        guard: Option<Expr>,
        body: Vec<Stmt>,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        otherwise: Vec<Stmt>,
    },
    Let {
        name: Name,
        value: Expr,
    },
    Contribute {
        relation: String,
        args: Vec<Expr>,
        value: Expr,
    },
}

#[derive(Debug, Clone)]
pub struct Binder {
    pub kind: BinderKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum BinderKind {
    /// `x: T`
    Typed { name: Name, ty: TypeExpr },
    /// `x in S` or `x in lo..hi`
    In { name: Name, source: InSource },
    /// `(a, _, b) in R`
    Tuple {
        names: Vec<Option<Name>>,
        relation: String,
    },
}

#[derive(Debug, Clone)]
pub enum InSource {
    Range(Range),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Range {
    pub lo: Box<Expr>,
    pub hi: Box<Expr>,
    pub inclusive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Iff,
    Imp,
    Or,
    Xor,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregator {
    Any,
    All,
    Count,
    ExactlyOne,
    AtMostOne,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Int(i64),
    Str(String),
    Bool(bool),
    None,
    Name(String),
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    ListComp {
        expr: Box<Expr>,
        binders: Vec<Binder>,
        guard: Option<Box<Expr>>,
    },
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Is(Box<Expr>, Vec<PatAlt>),
    Index(Box<Expr>, Vec<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Field(Box<Expr>, String),
    Aggregate {
        kind: Aggregator,
        expr: Box<Expr>,
        binders: Vec<Binder>,
        guard: Option<Box<Expr>>,
    },
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<(Vec<PatAlt>, Expr)>,
    },
}

#[derive(Debug, Clone)]
pub enum PatAlt {
    /// A member or variant name, or a const value of the scrutinee's type.
    Name(String, Span),
    /// `Member(args...)`
    Ctor(String, Vec<PatArg>, Span),
    Wild,
}

#[derive(Debug, Clone)]
pub enum PatArg {
    Wild,
    Expr(Expr),
}
