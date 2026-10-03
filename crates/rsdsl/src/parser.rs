//! Recursive-descent parser following docs/rsdsl/rsdsl.lark.

use crate::ast::*;
use crate::diag::{Diagnostic, Result, Span};
use crate::lexer::{lex, Tok, Token};

pub fn parse(file: u32, text: &str) -> Result<File> {
    let tokens = lex(file, text)?;
    let mut parser = Parser { tokens, pos: 0 };
    parser.file()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, offset: usize) -> &Tok {
        let index = (self.pos + offset).min(self.tokens.len() - 1);
        &self.tokens[index].tok
    }

    fn span(&self) -> Span {
        self.tokens[self.pos].span
    }

    fn prev_span(&self) -> Span {
        self.tokens[self.pos.saturating_sub(1)].span
    }

    fn bump(&mut self) -> Token {
        let token = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        token
    }

    fn is_p(&self, p: &str) -> bool {
        matches!(self.peek(), Tok::P(q) if *q == p)
    }

    fn is_kw(&self, k: &str) -> bool {
        matches!(self.peek(), Tok::Kw(q) if *q == k)
    }

    fn eat_p(&mut self, p: &str) -> bool {
        if self.is_p(p) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, k: &str) -> bool {
        if self.is_kw(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn unexpected(&self, expected: &str) -> Diagnostic {
        let found = match self.peek() {
            Tok::Ident(name) => format!("identifier `{name}`"),
            Tok::Int(value) => format!("integer `{value}`"),
            Tok::Str(_) => "string".to_owned(),
            Tok::Kw(k) => format!("`{k}`"),
            Tok::P(p) => format!("`{p}`"),
            Tok::Hole => "`_`".to_owned(),
            Tok::Eof => "end of file".to_owned(),
        };
        Diagnostic::error(
            "E0100",
            format!("expected {expected}, found {found}"),
            self.span(),
        )
    }

    fn expect_p(&mut self, p: &str) -> Result<Span> {
        if self.is_p(p) {
            Ok(self.bump().span)
        } else {
            Err(self.unexpected(&format!("`{p}`")))
        }
    }

    fn expect_kw(&mut self, k: &str) -> Result<Span> {
        if self.is_kw(k) {
            Ok(self.bump().span)
        } else {
            Err(self.unexpected(&format!("`{k}`")))
        }
    }

    fn ident(&mut self) -> Result<(String, Span)> {
        match self.peek().clone() {
            Tok::Ident(name) => {
                let span = self.bump().span;
                Ok((name, span))
            }
            Tok::Kw(k) => Err(self
                .unexpected("a name")
                .with_help(format!("`{k}` is a reserved word"))),
            _ => Err(self.unexpected("a name")),
        }
    }

    fn int(&mut self) -> Result<i64> {
        match self.peek().clone() {
            Tok::Int(value) => {
                self.bump();
                Ok(value)
            }
            _ => Err(self.unexpected("an integer")),
        }
    }

    fn string(&mut self) -> Result<String> {
        match self.peek().clone() {
            Tok::Str(value) => {
                self.bump();
                Ok(value)
            }
            _ => Err(self.unexpected("a string")),
        }
    }

    // ----- file and items ---------------------------------------------------

    fn file(&mut self) -> Result<File> {
        self.expect_kw("rsdsl")?;
        let version_span = self.span();
        let version = self.int()?;
        if version != 2 {
            return Err(Diagnostic::error(
                "E0101",
                format!("unsupported rsdsl version {version}; this compiler reads version 2"),
                version_span,
            ));
        }
        self.expect_p(";")?;
        let header = if self.eat_kw("model") {
            let (name, _) = self.ident()?;
            Header::Model { name }
        } else if self.eat_kw("instance") {
            let (name, _) = self.ident()?;
            self.expect_kw("of")?;
            let (of, _) = self.ident()?;
            Header::Instance { name, of }
        } else {
            return Err(self.unexpected("`model` or `instance`"));
        };
        self.expect_p(";")?;
        let mut items = Vec::new();
        while *self.peek() != Tok::Eof {
            items.push(self.item()?);
        }
        Ok(File { header, items })
    }

    fn annotations(&mut self) -> Result<Vec<Annotation>> {
        let mut annotations = Vec::new();
        while self.is_p("@") {
            let start = self.bump().span;
            let (name, _) = self.ident()?;
            let mut args = Vec::new();
            if self.eat_p("(") {
                if !self.is_p(")") {
                    loop {
                        let named = matches!(self.peek(), Tok::Ident(_))
                            && matches!(self.peek_at(1), Tok::P("="));
                        if named {
                            let (key, _) = self.ident()?;
                            self.expect_p("=")?;
                            args.push(AnnArg::Named(key, self.expr()?));
                        } else {
                            args.push(AnnArg::Pos(self.expr()?));
                        }
                        if !self.eat_p(",") {
                            break;
                        }
                    }
                }
                self.expect_p(")")?;
            }
            annotations.push(Annotation {
                name,
                args,
                span: start.to(self.prev_span()),
            });
        }
        Ok(annotations)
    }

    fn item(&mut self) -> Result<Item> {
        let annotations = self.annotations()?;
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::Kw("include") => {
                self.bump();
                let path = self.string()?;
                self.expect_p(";")?;
                ItemKind::Include(path)
            }
            Tok::Kw("grid") => self.grid()?,
            Tok::Kw("enum") => {
                self.bump();
                let (name, _) = self.ident()?;
                self.expect_p("{")?;
                let mut variants = Vec::new();
                while !self.is_p("}") {
                    variants.push(self.ident()?.0);
                    if !self.eat_p(",") {
                        break;
                    }
                }
                self.expect_p("}")?;
                ItemKind::Enum { name, variants }
            }
            Tok::Kw("subset") => {
                self.bump();
                let (name, _) = self.ident()?;
                self.expect_kw("of")?;
                let (of, _) = self.ident()?;
                self.expect_p("=")?;
                let values = self.expr()?;
                self.expect_p(";")?;
                ItemKind::Subset { name, of, values }
            }
            Tok::Kw("domain") => {
                self.bump();
                let (name, _) = self.ident()?;
                let ty = if self.eat_p(":") {
                    Some(self.ty()?)
                } else {
                    None
                };
                let value = if self.eat_p("=") {
                    Some(self.expr()?)
                } else {
                    None
                };
                self.expect_p(";")?;
                ItemKind::Domain { name, ty, value }
            }
            Tok::Kw("fact") => {
                self.bump();
                let (name, _) = self.ident()?;
                if self.eat_p("=") {
                    let value = self.expr()?;
                    self.expect_p(";")?;
                    ItemKind::FactValue { name, value }
                } else {
                    self.expect_p("(")?;
                    let params = self.params(")")?;
                    self.expect_p(")")?;
                    let derived = if self.eat_p(":=") {
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    self.expect_p(";")?;
                    ItemKind::Fact {
                        name,
                        params,
                        derived,
                    }
                }
            }
            Tok::Kw("param") => {
                self.bump();
                let (name, _) = self.ident()?;
                if self.eat_p("=") {
                    let value = self.expr()?;
                    self.expect_p(";")?;
                    ItemKind::ParamValue { name, value }
                } else {
                    self.expect_p(":")?;
                    let ty = self.ty()?;
                    let default = if self.eat_p("=") {
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    self.expect_p(";")?;
                    ItemKind::Param { name, ty, default }
                }
            }
            Tok::Kw("fn") => {
                self.bump();
                let (name, _) = self.ident()?;
                self.expect_p("(")?;
                let params = self.params(")")?;
                self.expect_p(")")?;
                self.expect_p("->")?;
                let ret = self.ty()?;
                self.expect_p("=")?;
                let body = self.expr()?;
                self.expect_p(";")?;
                ItemKind::Fn {
                    name,
                    params,
                    ret,
                    body,
                }
            }
            Tok::Kw("choice") => {
                self.bump();
                let (name, _) = self.ident()?;
                let index = self.index()?;
                self.expect_p("{")?;
                let mut members = Vec::new();
                while !self.is_p("}") {
                    let (member, member_span) = self.ident()?;
                    let params = if self.eat_p("(") {
                        let params = self.params(")")?;
                        self.expect_p(")")?;
                        params
                    } else {
                        Vec::new()
                    };
                    let guard = if self.eat_kw("if") {
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    members.push(Member {
                        name: member,
                        params,
                        guard,
                        span: member_span.to(self.prev_span()),
                    });
                    if !self.eat_p(",") {
                        break;
                    }
                }
                self.expect_p("}")?;
                ItemKind::Choice {
                    name,
                    index,
                    members,
                }
            }
            Tok::Kw("var") => {
                self.bump();
                let (name, _) = self.ident()?;
                let index = if self.eat_kw("over") {
                    VarIndex::Over(self.ident()?.0)
                } else {
                    VarIndex::Binders(self.index()?)
                };
                self.expect_p(";")?;
                ItemKind::Var { name, index }
            }
            Tok::Kw("def") => {
                self.bump();
                let (name, _) = self.ident()?;
                let index = if self.is_p("[") {
                    self.index()?
                } else {
                    Vec::new()
                };
                self.expect_p(":=")?;
                let body = self.expr()?;
                self.expect_p(";")?;
                ItemKind::Def { name, index, body }
            }
            Tok::Kw("relation") => {
                self.bump();
                let (name, _) = self.ident()?;
                self.expect_p("(")?;
                let params = self.params(")")?;
                self.expect_p(")")?;
                self.expect_p(";")?;
                ItemKind::Relation { name, params }
            }
            Tok::Kw("int") => {
                self.bump();
                let (name, _) = self.ident()?;
                let index = self.index()?;
                self.expect_kw("in")?;
                let lo = self.sum()?;
                let range = self.range_tail(lo)?;
                self.expect_p(";")?;
                ItemKind::Int { name, index, range }
            }
            Tok::Kw("rule") => {
                self.bump();
                let label = self.string()?;
                let body = self.block()?;
                ItemKind::Rule { label, body }
            }
            Tok::Kw(k @ ("minimize" | "maximize")) => {
                self.bump();
                let expr = self.expr()?;
                self.expect_p(";")?;
                ItemKind::Objective {
                    minimize: k == "minimize",
                    expr,
                }
            }
            _ => return Err(self.unexpected("a declaration")),
        };
        Ok(Item {
            annotations,
            kind,
            span: start.to(self.prev_span()),
        })
    }

    fn grid(&mut self) -> Result<ItemKind> {
        self.expect_kw("grid")?;
        let (name, _) = self.ident()?;
        if self.eat_p("=") {
            self.expect_p("(")?;
            let x = self.int()?;
            self.expect_p(",")?;
            let y = self.int()?;
            self.expect_p(",")?;
            let z = self.int()?;
            self.expect_p(")")?;
            self.expect_p(";")?;
            return Ok(ItemKind::GridValue {
                name,
                dims: [x, y, z],
            });
        }
        self.expect_p("(")?;
        let a = self.ident()?.0;
        self.expect_p(",")?;
        let b = self.ident()?.0;
        self.expect_p(",")?;
        let c = self.ident()?.0;
        self.expect_p(")")?;
        self.expect_kw("dirs")?;
        let (dirs, _) = self.ident()?;
        self.expect_p("{")?;
        let axes = [a, b, c];
        let mut directions = Vec::new();
        while !self.is_p("}") {
            let (direction, _) = self.ident()?;
            self.expect_p("=")?;
            let sign = if self.eat_p("+") {
                1
            } else if self.eat_p("-") {
                -1
            } else {
                return Err(self.unexpected("`+` or `-`"));
            };
            let (axis, axis_span) = self.ident()?;
            let Some(axis_index) = axes.iter().position(|a| *a == axis) else {
                return Err(Diagnostic::error(
                    "E0201",
                    format!("unknown axis `{axis}`; the grid declares {axes:?}"),
                    axis_span,
                ));
            };
            directions.push((direction, sign, axis_index));
            if !self.eat_p(",") {
                break;
            }
        }
        self.expect_p("}")?;
        self.eat_p(";");
        Ok(ItemKind::Grid {
            name,
            axes,
            dirs,
            directions,
        })
    }

    fn params(&mut self, close: &str) -> Result<Vec<Param>> {
        let mut params = Vec::new();
        if self.is_p(close) {
            return Ok(params);
        }
        loop {
            let named =
                matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek_at(1), Tok::P(":"));
            let name = if named {
                let (name, _) = self.ident()?;
                self.expect_p(":")?;
                Some(name)
            } else {
                None
            };
            params.push(Param {
                name,
                ty: self.ty()?,
            });
            if !self.eat_p(",") {
                break;
            }
        }
        Ok(params)
    }

    fn ty(&mut self) -> Result<TypeExpr> {
        let mut ty = if self.eat_kw("int") {
            TypeExpr::Int
        } else if self.eat_kw("bool") {
            TypeExpr::Bool
        } else if self.eat_kw("set") {
            self.expect_p("<")?;
            let inner = self.ty()?;
            self.expect_p(">")?;
            TypeExpr::Set(Box::new(inner))
        } else if self.eat_p("(") {
            let mut types = vec![self.ty()?];
            while self.eat_p(",") {
                types.push(self.ty()?);
            }
            self.expect_p(")")?;
            TypeExpr::Tuple(types)
        } else {
            let (name, span) = self.ident()?;
            TypeExpr::Named(name, span)
        };
        while self.eat_p("?") {
            ty = TypeExpr::Option(Box::new(ty));
        }
        Ok(ty)
    }

    fn index(&mut self) -> Result<Vec<Binder>> {
        self.expect_p("[")?;
        let binders = self.binders()?;
        self.expect_p("]")?;
        Ok(binders)
    }

    // ----- statements -------------------------------------------------------

    fn block(&mut self) -> Result<Vec<Stmt>> {
        self.expect_p("{")?;
        let mut stmts = Vec::new();
        while !self.is_p("}") {
            stmts.push(self.stmt()?);
        }
        self.expect_p("}")?;
        Ok(stmts)
    }

    fn stmt(&mut self) -> Result<Stmt> {
        let annotations = self.annotations()?;
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::Kw("require") => {
                self.bump();
                let expr = self.expr()?;
                self.expect_p(";")?;
                StmtKind::Require(expr)
            }
            Tok::Kw("forall") => {
                self.bump();
                let binders = self.binders()?;
                let guard = if self.eat_kw("where") {
                    Some(self.expr()?)
                } else {
                    None
                };
                let body = self.block()?;
                StmtKind::Forall {
                    binders,
                    guard,
                    body,
                }
            }
            Tok::Kw("if") => return self.if_stmt(annotations),
            Tok::Kw("let") => {
                self.bump();
                let (name, _) = self.ident()?;
                self.expect_p("=")?;
                let value = self.expr()?;
                self.expect_p(";")?;
                StmtKind::Let {
                    name: name.into(),
                    value,
                }
            }
            Tok::Ident(relation) => {
                self.bump();
                self.expect_p("(")?;
                let args = self.args(")")?;
                self.expect_p(")")?;
                self.expect_p("|=")?;
                let value = self.expr()?;
                self.expect_p(";")?;
                StmtKind::Contribute {
                    relation,
                    args,
                    value,
                }
            }
            _ => return Err(self.unexpected("a statement")),
        };
        Ok(Stmt {
            annotations,
            kind,
            span: start.to(self.prev_span()),
        })
    }

    fn if_stmt(&mut self, annotations: Vec<Annotation>) -> Result<Stmt> {
        let start = self.expect_kw("if")?;
        let cond = self.expr()?;
        let then = self.block()?;
        let otherwise = if self.eat_kw("else") {
            if self.is_kw("if") {
                vec![self.if_stmt(Vec::new())?]
            } else {
                self.block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt {
            annotations,
            kind: StmtKind::If {
                cond,
                then,
                otherwise,
            },
            span: start.to(self.prev_span()),
        })
    }

    fn binders(&mut self) -> Result<Vec<Binder>> {
        let mut binders = vec![self.binder()?];
        while self.eat_p(",") {
            binders.push(self.binder()?);
        }
        Ok(binders)
    }

    fn binder(&mut self) -> Result<Binder> {
        let start = self.span();
        let kind = if self.eat_p("(") {
            let mut names = Vec::new();
            loop {
                if matches!(self.peek(), Tok::Hole) {
                    self.bump();
                    names.push(None);
                } else {
                    names.push(Some(self.ident()?.0.into()));
                }
                if !self.eat_p(",") {
                    break;
                }
            }
            self.expect_p(")")?;
            self.expect_kw("in")?;
            let (relation, _) = self.ident()?;
            BinderKind::Tuple { names, relation }
        } else {
            let name: crate::ast::Name = self.ident()?.0.into();
            if self.eat_p(":") {
                BinderKind::Typed {
                    name,
                    ty: self.ty()?,
                }
            } else if self.eat_kw("in") {
                let first = self.expr()?;
                let source = if self.is_p("..") || self.is_p("..=") {
                    InSource::Range(self.range_tail(first)?)
                } else {
                    InSource::Expr(first)
                };
                BinderKind::In { name, source }
            } else {
                return Err(self.unexpected("`:` or `in` after a binder name"));
            }
        };
        Ok(Binder {
            kind,
            span: start.to(self.prev_span()),
        })
    }

    fn range_tail(&mut self, lo: Expr) -> Result<Range> {
        let inclusive = if self.eat_p("..=") {
            true
        } else {
            self.expect_p("..")?;
            false
        };
        let hi = self.sum()?;
        Ok(Range {
            lo: Box::new(lo),
            hi: Box::new(hi),
            inclusive,
        })
    }

    // ----- expressions ------------------------------------------------------

    fn mk(&self, kind: ExprKind, start: Span) -> Expr {
        Expr {
            kind,
            span: start.to(self.prev_span()),
        }
    }

    fn binary(&self, op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        let span = lhs.span.to(rhs.span);
        Expr {
            kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
            span,
        }
    }

    pub fn expr(&mut self) -> Result<Expr> {
        let lhs = self.imp()?;
        if self.eat_p("<->") {
            let rhs = self.imp()?;
            if self.is_p("<->") {
                return Err(Diagnostic::error(
                    "E0102",
                    "`<->` does not chain; add parentheses",
                    self.span(),
                ));
            }
            return Ok(self.binary(BinOp::Iff, lhs, rhs));
        }
        Ok(lhs)
    }

    fn imp(&mut self) -> Result<Expr> {
        let lhs = self.or()?;
        if self.eat_p("->") {
            let rhs = self.imp()?;
            return Ok(self.binary(BinOp::Imp, lhs, rhs));
        }
        Ok(lhs)
    }

    fn or(&mut self) -> Result<Expr> {
        let mut lhs = self.xor()?;
        while self.eat_kw("or") {
            let rhs = self.xor()?;
            lhs = self.binary(BinOp::Or, lhs, rhs);
        }
        Ok(lhs)
    }

    fn xor(&mut self) -> Result<Expr> {
        let mut lhs = self.and()?;
        while self.eat_kw("xor") {
            let rhs = self.and()?;
            lhs = self.binary(BinOp::Xor, lhs, rhs);
        }
        Ok(lhs)
    }

    fn and(&mut self) -> Result<Expr> {
        let mut lhs = self.not()?;
        while self.eat_kw("and") {
            let rhs = self.not()?;
            lhs = self.binary(BinOp::And, lhs, rhs);
        }
        Ok(lhs)
    }

    fn not(&mut self) -> Result<Expr> {
        let start = self.span();
        if self.eat_kw("not") {
            let inner = self.not()?;
            return Ok(self.mk(ExprKind::Not(Box::new(inner)), start));
        }
        self.cmp()
    }

    fn cmp(&mut self) -> Result<Expr> {
        let lhs = self.sum()?;
        let op = match self.peek() {
            Tok::P("==") => Some(BinOp::Eq),
            Tok::P("!=") => Some(BinOp::Ne),
            Tok::P("<") => Some(BinOp::Lt),
            Tok::P("<=") => Some(BinOp::Le),
            Tok::P(">") => Some(BinOp::Gt),
            Tok::P(">=") => Some(BinOp::Ge),
            Tok::Kw("in") => Some(BinOp::In),
            _ => None,
        };
        let result = if let Some(op) = op {
            self.bump();
            let rhs = self.sum()?;
            self.binary(op, lhs, rhs)
        } else if self.eat_kw("is") {
            let start = lhs.span;
            let pattern = self.pattern()?;
            self.mk(ExprKind::Is(Box::new(lhs), pattern), start)
        } else {
            return Ok(lhs);
        };
        if matches!(
            self.peek(),
            Tok::P("==" | "!=" | "<" | "<=" | ">" | ">=") | Tok::Kw("in" | "is")
        ) {
            return Err(Diagnostic::error(
                "E0103",
                "comparisons do not chain; combine them with `and`",
                self.span(),
            ));
        }
        Ok(result)
    }

    fn sum(&mut self) -> Result<Expr> {
        let mut lhs = self.term()?;
        loop {
            let op = if self.is_p("+") {
                BinOp::Add
            } else if self.is_p("-") {
                BinOp::Sub
            } else {
                break;
            };
            self.bump();
            let rhs = self.term()?;
            lhs = self.binary(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let op = if self.is_p("*") {
                BinOp::Mul
            } else if self.is_p("/") {
                BinOp::Div
            } else if self.is_p("%") {
                BinOp::Mod
            } else {
                break;
            };
            self.bump();
            let rhs = self.unary()?;
            lhs = self.binary(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr> {
        let start = self.span();
        if self.eat_p("-") {
            let inner = self.unary()?;
            return Ok(self.mk(ExprKind::Neg(Box::new(inner)), start));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr> {
        let start = self.span();
        let mut expr = self.primary()?;
        loop {
            if self.eat_p("[") {
                let args = self.args("]")?;
                self.expect_p("]")?;
                expr = self.mk(ExprKind::Index(Box::new(expr), args), start);
            } else if self.eat_p("(") {
                let args = self.args(")")?;
                self.expect_p(")")?;
                expr = self.mk(ExprKind::Call(Box::new(expr), args), start);
            } else if self.eat_p(".") {
                let (field, _) = self.ident()?;
                expr = self.mk(ExprKind::Field(Box::new(expr), field), start);
            } else {
                return Ok(expr);
            }
        }
    }

    fn args(&mut self, close: &str) -> Result<Vec<Expr>> {
        let mut args = Vec::new();
        if self.is_p(close) {
            return Ok(args);
        }
        loop {
            args.push(self.expr()?);
            if !self.eat_p(",") {
                break;
            }
        }
        Ok(args)
    }

    fn primary(&mut self) -> Result<Expr> {
        let start = self.span();
        let kind = match self.peek().clone() {
            Tok::Int(value) => {
                self.bump();
                ExprKind::Int(value)
            }
            Tok::Str(value) => {
                self.bump();
                ExprKind::Str(value)
            }
            Tok::Kw("true") => {
                self.bump();
                ExprKind::Bool(true)
            }
            Tok::Kw("false") => {
                self.bump();
                ExprKind::Bool(false)
            }
            Tok::Kw("none") => {
                self.bump();
                ExprKind::None
            }
            Tok::Ident(name) => {
                self.bump();
                ExprKind::Name(name)
            }
            Tok::P("(") => {
                self.bump();
                let first = self.expr()?;
                if self.eat_p(",") {
                    let mut items = vec![first];
                    items.extend(self.args(")")?);
                    self.expect_p(")")?;
                    ExprKind::Tuple(items)
                } else {
                    self.expect_p(")")?;
                    return Ok(Expr {
                        kind: first.kind,
                        span: start.to(self.prev_span()),
                    });
                }
            }
            Tok::P("[") => {
                self.bump();
                if self.eat_p("]") {
                    ExprKind::List(Vec::new())
                } else {
                    let first = self.expr()?;
                    if self.eat_kw("for") {
                        let binders = self.binders()?;
                        let guard = if self.eat_kw("where") {
                            Some(Box::new(self.expr()?))
                        } else {
                            None
                        };
                        self.expect_p("]")?;
                        ExprKind::ListComp {
                            expr: Box::new(first),
                            binders,
                            guard,
                        }
                    } else {
                        let mut items = vec![first];
                        if self.eat_p(",") {
                            items.extend(self.args("]")?);
                        }
                        self.expect_p("]")?;
                        ExprKind::List(items)
                    }
                }
            }
            Tok::Kw(k @ ("any" | "all" | "count" | "exactly_one" | "at_most_one")) => {
                self.bump();
                let kind = match k {
                    "any" => Aggregator::Any,
                    "all" => Aggregator::All,
                    "count" => Aggregator::Count,
                    "exactly_one" => Aggregator::ExactlyOne,
                    _ => Aggregator::AtMostOne,
                };
                self.expect_p("(")?;
                let expr = self.expr()?;
                self.expect_kw("for")?;
                let binders = self.binders()?;
                let guard = if self.eat_kw("where") {
                    Some(Box::new(self.expr()?))
                } else {
                    None
                };
                self.expect_p(")")?;
                ExprKind::Aggregate {
                    kind,
                    expr: Box::new(expr),
                    binders,
                    guard,
                }
            }
            Tok::Kw("if") => {
                self.bump();
                let cond = self.expr()?;
                self.expect_p("{")?;
                let then = self.expr()?;
                self.expect_p("}")?;
                self.expect_kw("else")?;
                let otherwise = if self.is_kw("if") {
                    self.primary()?
                } else {
                    self.expect_p("{")?;
                    let otherwise = self.expr()?;
                    self.expect_p("}")?;
                    otherwise
                };
                ExprKind::If {
                    cond: Box::new(cond),
                    then: Box::new(then),
                    otherwise: Box::new(otherwise),
                }
            }
            Tok::Kw("match") => {
                self.bump();
                let scrutinee = self.expr()?;
                self.expect_p("{")?;
                let mut arms = Vec::new();
                while !self.is_p("}") {
                    let pattern = self.pattern()?;
                    self.expect_p("=>")?;
                    let value = self.expr()?;
                    arms.push((pattern, value));
                    if !self.eat_p(",") {
                        break;
                    }
                }
                self.expect_p("}")?;
                ExprKind::Match {
                    scrutinee: Box::new(scrutinee),
                    arms,
                }
            }
            _ => return Err(self.unexpected("an expression")),
        };
        Ok(self.mk(kind, start))
    }

    fn pattern(&mut self) -> Result<Vec<PatAlt>> {
        let mut alts = vec![self.pat_alt()?];
        while self.eat_p("|") {
            alts.push(self.pat_alt()?);
        }
        Ok(alts)
    }

    fn pat_alt(&mut self) -> Result<PatAlt> {
        if matches!(self.peek(), Tok::Hole) {
            self.bump();
            return Ok(PatAlt::Wild);
        }
        let (name, span) = self.ident()?;
        if self.eat_p("(") {
            let mut args = Vec::new();
            loop {
                if matches!(self.peek(), Tok::Hole) {
                    self.bump();
                    args.push(PatArg::Wild);
                } else {
                    args.push(PatArg::Expr(self.expr()?));
                }
                if !self.eat_p(",") {
                    break;
                }
            }
            self.expect_p(")")?;
            return Ok(PatAlt::Ctor(name, args, span.to(self.prev_span())));
        }
        Ok(PatAlt::Name(name, span))
    }
}
