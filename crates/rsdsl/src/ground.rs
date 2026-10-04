//! Grounding: evaluates a model against an instance into CNF.
//!
//! Compile-time expressions evaluate to `Value`s; solver-time expressions to
//! formulas (`F`). Rules run twice: first to collect relation contributions,
//! then (after every relation is frozen) to emit constraints.

use std::sync::Arc;

use crate::ast::*;
use crate::diag::{Diagnostic, Result, Span};
use crate::formula::{Card, CardOp, Encoder, Lit, Pol, VarOrigin, F};
use crate::fxhash::{FxMap as HashMap, FxSet as HashSet};
use crate::instance::{IValue, Instance};
use crate::value::{Names, Show, ShowType, Type, Value};

// ----- declarations -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decl {
    Enum(u32),
    Subset(u32),
    Domain(u32),
    Grid(u32),
    Fact(u32),
    Param(u32),
    Fn(u32),
    Choice(u32),
    Var(u32),
    Def(u32),
    Relation(u32),
    Int(u32),
}

pub(crate) struct EnumDef {
    pub name: String,
    pub variants: Vec<String>,
}

pub(crate) struct SubsetDef {
    pub name: String,
    pub base: u32,
    pub members: Vec<u32>,
}

pub(crate) struct DomainDef {
    pub name: String,
    pub int: bool,
    pub values: Vec<Value>,
    pub symbols: Vec<String>,
}

pub(crate) struct GridDef {
    pub name: String,
    pub axes: [String; 3],
    pub dims: [i64; 3],
    pub dirs: u32,
    /// Per direction variant: (axis, sign).
    pub steps: Vec<(usize, i8)>,
}

impl GridDef {
    fn len(&self) -> u32 {
        (self.dims[0] * self.dims[1] * self.dims[2]) as u32
    }
    pub(crate) fn coords(&self, index: u32) -> [i64; 3] {
        let i = index as i64;
        let x = i % self.dims[0];
        let y = (i / self.dims[0]) % self.dims[1];
        let z = i / (self.dims[0] * self.dims[1]);
        [x, y, z]
    }
    fn index(&self, c: [i64; 3]) -> Option<u32> {
        if (0..3).all(|a| c[a] >= 0 && c[a] < self.dims[a]) {
            Some((c[0] + self.dims[0] * (c[1] + self.dims[1] * c[2])) as u32)
        } else {
            None
        }
    }
}

enum FactKind {
    Extern {
        tuples: Vec<Vec<Value>>,
        set: HashSet<Vec<Value>>,
    },
    Derived {
        body: Arc<Expr>,
        memo: HashMap<Vec<Value>, bool>,
    },
}

struct FactDef {
    name: String,
    params: Arc<[(Option<Name>, Type)]>,
    kind: FactKind,
}

struct ParamDef {
    value: Value,
}

struct FnDef {
    params: Vec<(Name, Type)>,
    ret: Type,
    body: Arc<Expr>,
}

#[derive(Clone)]
pub(crate) struct Dim {
    pub name: Name,
    pub ty: Type,
}

pub(crate) struct MemberDef {
    pub name: String,
    pub params: Vec<(Name, Type)>,
    guard: Option<Arc<Expr>>,
}

pub(crate) struct OptionEntry {
    pub member: u32,
    pub payload: Arc<[Value]>,
    pub lit: Lit,
}

pub(crate) struct ChoiceDef {
    pub name: String,
    pub index: Arc<[Dim]>,
    pub members: Vec<MemberDef>,
    prefer: Option<u32>,
    outside: Option<u32>,
    pub display: Option<String>,
    pub keys: Vec<Vec<Value>>,
    pub lookup: HashMap<Vec<Value>, u32>,
    pub options: Vec<Vec<OptionEntry>>,
}

pub(crate) enum VarIndexKind {
    Dims(Arc<[Dim]>),
    Over(u32),
}

pub(crate) struct VarDef {
    pub name: String,
    pub index: VarIndexKind,
    pub display: Option<String>,
    outside: bool,
    pub keys: Vec<Vec<Value>>,
    pub lookup: HashMap<Vec<Value>, Lit>,
}

pub(crate) struct DefDef {
    pub name: String,
    pub index: Arc<[Dim]>,
    body: Arc<Expr>,
    internal: bool,
    outside: bool,
    pub display: Option<String>,
    pub keys: Vec<Vec<Value>>,
    pub memo: HashMap<Vec<Value>, F>,
    in_progress: HashSet<Vec<Value>>,
}

pub(crate) struct RelDef {
    pub name: String,
    pub params: Arc<[(String, Type)]>,
    pub display: Option<String>,
    contribs: HashMap<Vec<Value>, Vec<F>>,
    contrib_order: Vec<Vec<Value>>,
    pub frozen: bool,
    pub tuples: Vec<(Vec<Value>, Lit)>,
    pub lookup: HashMap<Vec<Value>, u32>,
    field_index: HashMap<(usize, Value), Vec<u32>>,
}

pub(crate) struct IntDef {
    pub name: String,
    pub index: Arc<[Dim]>,
    pub lo: i64,
    pub hi: i64,
    pub display: Option<String>,
    outside: Option<i64>,
    grounded: bool,
    pub keys: Vec<Vec<Value>>,
    pub lookup: HashMap<Vec<Value>, Arc<[Lit]>>,
}

pub(crate) struct RuleDef {
    pub label: String,
    body: Arc<Vec<Stmt>>,
    guarded: Option<Vec<String>>,
    /// `@fold`: its requires are unit facts, applied before every other rule
    /// so that they fold to constants there.
    fold: bool,
    has_contrib: bool,
    has_require: bool,
    span: Span,
    pub clauses: usize,
    /// Time spent running the rule in both passes.
    pub time: std::time::Duration,
}

/// What a named solver variable stands for.
#[derive(Debug, Clone)]
pub(crate) enum NameRef {
    Option { choice: u32, key: u32, option: u32 },
    Occupied { choice: u32, key: u32 },
    Var { var: u32, key: u32 },
    Def { def: u32, key: u32 },
    Rel { rel: u32, tuple: u32 },
    IntGe { int: u32, key: u32, value: i64 },
}

pub(crate) struct GuardDef {
    pub rule: String,
    pub names: Vec<String>,
    pub key: Vec<Value>,
    pub lit: Lit,
}

/// Grounding options.
#[derive(Debug, Clone, Default)]
pub struct GroundOptions {
    /// Emit `@guarded` selectors (otherwise those rules are unguarded).
    pub guards: bool,
    /// Record the binder valuation of every clause (for explained output).
    pub provenance: bool,
    /// Encode OR auxiliaries as positive variables instead of negated AND
    /// variables (changes only the solver's initial phases).
    pub positive_or_aux: bool,
    /// Run `@fold` rules as ordinary rules, without fixing their literals
    /// first (same models, larger CNF).
    pub no_fold: bool,
}

#[derive(Clone)]
enum Val {
    C(Value),
    F(F),
    I(IntTerm),
    /// `Family[key]` of a choice, awaiting `is`: the key's index, or `None`
    /// when the key is outside the domain.
    Choice(u32, Option<u32>),
    /// A type name used as a set (`d in Dir4`) or as a qualifier.
    Type(Type),
}

#[derive(Clone)]
enum IntTerm {
    Const(i64),
    Var {
        lits: Arc<[Lit]>,
        lo: i64,
        offset: i64,
    },
    Count {
        items: Vec<F>,
        offset: i64,
    },
    /// `constant + Σ weight · [formula]`, for weighted sums (objectives).
    Linear {
        terms: Vec<(i64, F)>,
        constant: i64,
    },
}

/// A resolved pattern alternative over one choice.
enum Alt {
    Any,
    Value(Value),
    /// Member index and payload values (`None` for `_`).
    Member(u32, Vec<Option<Value>>),
}

impl Alt {
    fn matches(&self, choice: u32, member: u32, payload: &[Value]) -> bool {
        match self {
            Alt::Any => true,
            Alt::Value(Value::Member(c, m, p)) => *c == choice && *m == member && **p == *payload,
            Alt::Value(_) => false,
            Alt::Member(m, args) => {
                *m == member
                    && args
                        .iter()
                        .zip(payload)
                        .all(|(arg, actual)| arg.as_ref().map_or(true, |a| a == actual))
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Collect,
    Emit,
}

type Env = Vec<(Name, Val)>;

/// Binder names are short, so compare inline rather than through memcmp.
#[inline]
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).all(|(x, y)| x == y)
}

pub struct Program {
    pub(crate) names: HashMap<String, Decl>,
    pub(crate) enums: Vec<EnumDef>,
    pub(crate) subsets: Vec<SubsetDef>,
    pub(crate) domains: Vec<DomainDef>,
    pub(crate) grids: Vec<GridDef>,
    facts: Vec<FactDef>,
    params: Vec<ParamDef>,
    fns: Vec<Arc<FnDef>>,
    pub(crate) choices: Vec<ChoiceDef>,
    pub(crate) vars: Vec<VarDef>,
    pub(crate) defs: Vec<DefDef>,
    pub(crate) relations: Vec<RelDef>,
    pub(crate) ints: Vec<IntDef>,
    pub(crate) rules: Vec<RuleDef>,
    variants: HashMap<String, Vec<(u32, u32)>>,
    members: HashMap<String, Vec<(u32, u32)>>,
    symbols: HashMap<String, Vec<(u32, u32)>>,
    pub(crate) encoder: Encoder,
    pub(crate) name_refs: Vec<NameRef>,
    /// Extra names for literals that alias an existing variable.
    pub(crate) aliases: HashMap<Lit, Vec<u32>>,
    /// `(label, valuation)` per provenance id.
    pub(crate) origins: Vec<(String, String)>,
    pub(crate) guards: Vec<GuardDef>,
    guard_lookup: HashMap<(usize, Vec<Value>), Lit>,
    options: GroundOptions,
    mode: Mode,
    /// Running `@fold` rules: requires record their literals as fixed.
    folding: bool,
    /// Reusable buffer for family indices and fact arguments.
    key_scratch: Vec<Value>,
    current_rule: Option<usize>,
    pub warnings: Vec<Diagnostic>,
    pub(crate) sources: crate::diag::SourceMap,
    type_values: HashMap<Type, Arc<[Value]>>,
    /// `minimize`/`maximize` items, grounded after the rules.
    objectives: Vec<(bool, Expr, Span)>,
    /// The summed objective as `(weight, literal)` costs plus a constant:
    /// cost = constant + Σ weight · [literal]. Weights are positive.
    pub(crate) objective: Option<(Vec<(u64, Lit)>, i64)>,
    /// Grounding-time caches keyed by source span: an AST node always
    /// resolves the same way, so names and declarations are looked up once.
    name_cache: HashMap<(Span, Option<Type>), Val>,
    decl_cache: HashMap<Span, Option<Decl>>,
    member_cache: HashMap<(Span, u32), u32>,
}

fn err(code: &'static str, message: impl Into<String>, span: Span) -> Diagnostic {
    Diagnostic::error(code, message, span)
}

impl Program {
    pub(crate) fn ground(
        file: &File,
        instance: &Instance,
        instance_file: Option<&File>,
        options: GroundOptions,
    ) -> Result<Program> {
        let mut p = Program {
            names: HashMap::default(),
            enums: Vec::new(),
            subsets: Vec::new(),
            domains: Vec::new(),
            grids: Vec::new(),
            facts: Vec::new(),
            params: Vec::new(),
            fns: Vec::new(),
            choices: Vec::new(),
            vars: Vec::new(),
            defs: Vec::new(),
            relations: Vec::new(),
            ints: Vec::new(),
            rules: Vec::new(),
            variants: HashMap::default(),
            members: HashMap::default(),
            symbols: HashMap::default(),
            encoder: Encoder::new(),
            name_refs: Vec::new(),
            aliases: HashMap::default(),
            origins: vec![("상수".to_owned(), String::new())],
            guards: Vec::new(),
            guard_lookup: HashMap::default(),
            options,
            mode: Mode::Collect,
            folding: false,
            key_scratch: Vec::new(),
            current_rule: None,
            warnings: Vec::new(),
            sources: Default::default(),
            type_values: HashMap::default(),
            name_cache: HashMap::default(),
            decl_cache: HashMap::default(),
            member_cache: HashMap::default(),
            objectives: Vec::new(),
            objective: None,
        };
        p.encoder.or_as_negated_and = !p.options.positive_or_aux;
        p.declare(file, instance, instance_file)?;
        p.check_instance_names(instance)?;
        p.ground_families(file)?;
        p.run_rules()?;
        // The caches refer to AST nodes that are only borrowed while grounding.
        p.name_cache = HashMap::default();
        p.decl_cache = HashMap::default();
        p.member_cache = HashMap::default();
        Ok(p)
    }

    fn add_name(&mut self, name: &str, decl: Decl, span: Span) -> Result<()> {
        if self.names.insert(name.to_owned(), decl).is_some() {
            return Err(err("E0200", format!("`{name}` is declared twice"), span));
        }
        Ok(())
    }

    // ----- declaration phase ---------------------------------------------------

    fn declare(
        &mut self,
        file: &File,
        instance: &Instance,
        instance_file: Option<&File>,
    ) -> Result<()> {
        // Pass A: names of enums, subsets, domains, grids, choices.
        for item in &file.items {
            match &item.kind {
                ItemKind::Enum { name, variants } => {
                    let id = self.enums.len() as u32;
                    self.add_name(name, Decl::Enum(id), item.span)?;
                    for (v, variant) in variants.iter().enumerate() {
                        self.variants
                            .entry(variant.clone())
                            .or_default()
                            .push((id, v as u32));
                    }
                    self.enums.push(EnumDef {
                        name: name.clone(),
                        variants: variants.clone(),
                    });
                }
                ItemKind::Grid {
                    name,
                    axes,
                    dirs,
                    directions,
                } => {
                    let enum_id = self.enums.len() as u32;
                    self.add_name(dirs, Decl::Enum(enum_id), item.span)?;
                    for (v, (variant, _, _)) in directions.iter().enumerate() {
                        self.variants
                            .entry(variant.clone())
                            .or_default()
                            .push((enum_id, v as u32));
                    }
                    self.enums.push(EnumDef {
                        name: dirs.clone(),
                        variants: directions.iter().map(|(v, _, _)| v.clone()).collect(),
                    });
                    let grid_id = self.grids.len() as u32;
                    self.add_name(name, Decl::Grid(grid_id), item.span)?;
                    let dims = instance_grid(name, instance, instance_file).ok_or_else(|| {
                        err(
                            "E0601",
                            format!("the instance does not give the size of grid `{name}`"),
                            item.span,
                        )
                    })?;
                    if dims.iter().any(|&d| d <= 0) {
                        return Err(err(
                            "E0602",
                            format!("grid `{name}` has an empty dimension"),
                            item.span,
                        ));
                    }
                    self.grids.push(GridDef {
                        name: name.clone(),
                        axes: axes.clone(),
                        dims,
                        dirs: enum_id,
                        steps: directions
                            .iter()
                            .map(|(_, sign, axis)| (*axis, *sign))
                            .collect(),
                    });
                }
                ItemKind::Domain { name, ty, .. } => {
                    let id = self.domains.len() as u32;
                    self.add_name(name, Decl::Domain(id), item.span)?;
                    let int = matches!(ty, Some(TypeExpr::Int));
                    self.domains.push(DomainDef {
                        name: name.clone(),
                        int,
                        values: Vec::new(),
                        symbols: Vec::new(),
                    });
                }
                ItemKind::Choice { name, .. } => {
                    let id = self.choices.len() as u32;
                    self.add_name(name, Decl::Choice(id), item.span)?;
                    self.choices.push(ChoiceDef {
                        name: name.clone(),
                        index: Arc::from(Vec::new()),
                        members: Vec::new(),
                        prefer: None,
                        outside: None,
                        display: None,
                        keys: Vec::new(),
                        lookup: HashMap::default(),
                        options: Vec::new(),
                    });
                }
                _ => {}
            }
        }
        // Subsets need enums.
        for item in &file.items {
            if let ItemKind::Subset { name, of, values } = &item.kind {
                let base = match self.names.get(of) {
                    Some(Decl::Enum(e)) => *e,
                    _ => return Err(err("E0202", format!("`{of}` is not an enum"), item.span)),
                };
                let ExprKind::List(items) = &values.kind else {
                    return Err(err(
                        "E0203",
                        "a subset lists its variants in brackets",
                        values.span,
                    ));
                };
                let mut members = Vec::new();
                for v in items {
                    let ExprKind::Name(n) = &v.kind else {
                        return Err(err("E0203", "expected a variant name", v.span));
                    };
                    let Some(index) = self.enums[base as usize]
                        .variants
                        .iter()
                        .position(|x| x == n)
                    else {
                        return Err(err(
                            "E0204",
                            format!("`{n}` is not a variant of `{of}`"),
                            v.span,
                        ));
                    };
                    members.push(index as u32);
                }
                let id = self.subsets.len() as u32;
                self.add_name(name, Decl::Subset(id), item.span)?;
                self.subsets.push(SubsetDef {
                    name: name.clone(),
                    base,
                    members,
                });
            }
        }
        // Domain values from the instance.
        for d in 0..self.domains.len() {
            let name = self.domains[d].name.clone();
            let int = self.domains[d].int;
            let values = instance_domain(&name, instance, instance_file)?;
            let Some(values) = values else {
                let span = file
                    .items
                    .iter()
                    .find(|i| matches!(&i.kind, ItemKind::Domain { name: n, .. } if *n == name))
                    .map(|i| i.span)
                    .unwrap_or_default();
                return Err(err(
                    "E0601",
                    format!("the instance does not give the values of domain `{name}`"),
                    span,
                ));
            };
            for v in values {
                match (v, int) {
                    (DomainLiteral::Int(i), true) => self.domains[d].values.push(Value::Int(i)),
                    (DomainLiteral::Sym(s), false) => {
                        let index = self.domains[d].symbols.len() as u32;
                        self.symbols
                            .entry(s.clone())
                            .or_default()
                            .push((d as u32, index));
                        self.domains[d].symbols.push(s);
                        self.domains[d].values.push(Value::Sym(d as u32, index));
                    }
                    (DomainLiteral::Int(_), false) => {
                        return Err(err_nospan(format!(
                            "domain `{name}` holds symbols, not integers"
                        )))
                    }
                    (DomainLiteral::Sym(s), true) => {
                        return Err(err_nospan(format!(
                            "domain `{name}` holds integers, not `{s}`"
                        )))
                    }
                }
            }
        }
        // Choice index and member types (choices may be referenced as types).
        for item in &file.items {
            if let ItemKind::Choice {
                name,
                index,
                members,
            } = &item.kind
            {
                let id = match self.names[name] {
                    Decl::Choice(id) => id,
                    _ => unreachable!(),
                };
                let dims = self.dims(index)?;
                let mut member_defs = Vec::new();
                for (m, member) in members.iter().enumerate() {
                    let mut params = Vec::new();
                    for (i, param) in member.params.iter().enumerate() {
                        let ty = self.resolve_type(&param.ty)?;
                        params.push((
                            param.name.clone().unwrap_or_else(|| format!("_{i}")).into(),
                            ty,
                        ));
                    }
                    self.members
                        .entry(member.name.clone())
                        .or_default()
                        .push((id, m as u32));
                    member_defs.push(MemberDef {
                        name: member.name.clone(),
                        params,
                        guard: member.guard.clone().map(Arc::new),
                    });
                }
                let choice = &mut self.choices[id as usize];
                choice.index = dims.into();
                choice.members = member_defs;
                for annotation in &item.annotations {
                    match annotation.name.as_str() {
                        "prefer" | "outside" => {
                            let Some(AnnArg::Pos(Expr {
                                kind: ExprKind::Name(member),
                                span,
                            })) = annotation.args.first()
                            else {
                                return Err(err(
                                    "E0702",
                                    format!("`@{}` takes a member name", annotation.name),
                                    annotation.span,
                                ));
                            };
                            let Some(m) = self.choices[id as usize]
                                .members
                                .iter()
                                .position(|x| x.name == *member)
                            else {
                                return Err(err(
                                    "E0703",
                                    format!("`{member}` is not a member of `{name}`"),
                                    *span,
                                ));
                            };
                            if !self.choices[id as usize].members[m].params.is_empty() {
                                return Err(err(
                                    "E0704",
                                    "only members without payload can be preferred or outside",
                                    *span,
                                ));
                            }
                            if annotation.name == "prefer" {
                                self.choices[id as usize].prefer = Some(m as u32);
                            } else {
                                self.choices[id as usize].outside = Some(m as u32);
                            }
                        }
                        "display" => {
                            self.choices[id as usize].display = Some(display_arg(annotation)?)
                        }
                        "label" => {}
                        other => return Err(unknown_annotation(other, annotation.span, "choice")),
                    }
                }
            }
        }
        // Params, facts, functions.
        for item in &file.items {
            match &item.kind {
                ItemKind::Param { name, ty, default } => {
                    let ty = self.resolve_type(ty)?;
                    let id = self.params.len() as u32;
                    self.add_name(name, Decl::Param(id), item.span)?;
                    self.params.push(ParamDef { value: Value::None });
                    let value = if let Some(v) = instance.params.get(name) {
                        self.ivalue(v, &ty, item.span)?
                    } else if let Some(expr) = instance_file.and_then(|f| find_param_value(f, name))
                    {
                        self.const_eval(expr, &mut Vec::new(), Some(&ty))?
                    } else if let Some(default) = default {
                        self.const_eval(default, &mut Vec::new(), Some(&ty))?
                    } else {
                        return Err(err(
                            "E0601",
                            format!("the instance does not set param `{name}`"),
                            item.span,
                        ));
                    };
                    if !self.conforms(&value, &ty) {
                        return Err(err(
                            "E0400",
                            format!("param `{name}` expects {}", ShowType(&ty, self)),
                            item.span,
                        ));
                    }
                    self.params[id as usize].value = value;
                }
                ItemKind::Fn {
                    name,
                    params,
                    ret,
                    body,
                } => {
                    let mut ps = Vec::new();
                    for p in params {
                        let Some(pname) = &p.name else {
                            return Err(err("E0205", "function parameters need names", item.span));
                        };
                        ps.push((pname.as_str().into(), self.resolve_type(&p.ty)?));
                    }
                    let ret = self.resolve_type(ret)?;
                    let id = self.fns.len() as u32;
                    self.add_name(name, Decl::Fn(id), item.span)?;
                    self.fns.push(Arc::new(FnDef {
                        params: ps,
                        ret,
                        body: Arc::new(body.clone()),
                    }));
                }
                ItemKind::Fact { name, params, .. } => {
                    let mut ps = Vec::new();
                    for p in params {
                        ps.push((p.name.as_deref().map(Name::from), self.resolve_type(&p.ty)?));
                    }
                    let id = self.facts.len() as u32;
                    self.add_name(name, Decl::Fact(id), item.span)?;
                    self.facts.push(FactDef {
                        name: name.clone(),
                        params: ps.into(),
                        kind: FactKind::Extern {
                            tuples: Vec::new(),
                            set: HashSet::default(),
                        },
                    });
                }
                _ => {}
            }
        }
        // Fact contents: extern tuples or derived bodies.
        for item in &file.items {
            if let ItemKind::Fact { name, derived, .. } = &item.kind {
                let id = match self.names[name] {
                    Decl::Fact(id) => id as usize,
                    _ => unreachable!(),
                };
                if let Some(body) = derived {
                    if self.facts[id].params.iter().any(|(n, _)| n.is_none()) {
                        return Err(err(
                            "E0206",
                            "a derived fact names its parameters",
                            item.span,
                        ));
                    }
                    self.facts[id].kind = FactKind::Derived {
                        body: Arc::new(body.clone()),
                        memo: HashMap::default(),
                    };
                    continue;
                }
                let types: Vec<Type> = self.facts[id]
                    .params
                    .iter()
                    .map(|(_, t)| t.clone())
                    .collect();
                let tuple_type = Type::Tuple(types.clone());
                let mut tuples = Vec::new();
                if let Some(rows) = instance.facts.get(name) {
                    for row in rows {
                        if row.len() != types.len() {
                            return Err(err(
                                "E0601",
                                format!("fact `{name}` takes {} values per row", types.len()),
                                item.span,
                            ));
                        }
                        let mut tuple = Vec::new();
                        for (v, t) in row.iter().zip(&types) {
                            tuple.push(self.ivalue(v, t, item.span)?);
                        }
                        tuples.push(tuple);
                    }
                } else if let Some(expr) = instance_file.and_then(|f| find_fact_value(f, name)) {
                    let set_type = Type::Set(Box::new(if types.len() == 1 {
                        types[0].clone()
                    } else {
                        tuple_type.clone()
                    }));
                    let value = self.const_eval(expr, &mut Vec::new(), Some(&set_type))?;
                    let Value::Set(items) = value else {
                        return Err(err(
                            "E0400",
                            format!("fact `{name}` expects a list"),
                            expr.span,
                        ));
                    };
                    for v in items.iter() {
                        let tuple = match (v, types.len()) {
                            (Value::Tuple(items), n) if n > 1 => items.to_vec(),
                            (other, 1) => vec![other.clone()],
                            _ => {
                                return Err(err(
                                    "E0400",
                                    format!("fact `{name}` rows have {} values", types.len()),
                                    expr.span,
                                ))
                            }
                        };
                        for (value, ty) in tuple.iter().zip(&types) {
                            if !self.conforms(value, ty) {
                                return Err(err(
                                    "E0400",
                                    format!("fact `{name}` expects {}", ShowType(ty, self)),
                                    expr.span,
                                ));
                            }
                        }
                        tuples.push(tuple);
                    }
                } else {
                    return Err(err(
                        "E0601",
                        format!("the instance does not give fact `{name}`"),
                        item.span,
                    ));
                }
                let set = tuples.iter().cloned().collect();
                self.facts[id].kind = FactKind::Extern { tuples, set };
            }
        }
        Ok(())
    }

    /// Instance entries the model does not declare are mistakes (typically a
    /// misspelled param), not something to ignore.
    fn check_instance_names(&self, instance: &Instance) -> Result<()> {
        let unknown = |kind: &str, name: &str| {
            Diagnostic::error_nospan(
                "E0603",
                format!("the instance sets {kind} `{name}`, which the model does not declare"),
            )
        };
        for name in instance.params.keys() {
            if !matches!(self.names.get(name), Some(Decl::Param(_))) {
                return Err(unknown("param", name));
            }
        }
        for name in instance.facts.keys() {
            if !matches!(self.names.get(name), Some(Decl::Fact(_))) {
                return Err(unknown("fact", name));
            }
        }
        for name in instance.domains.keys() {
            if !matches!(self.names.get(name), Some(Decl::Domain(_))) {
                return Err(unknown("domain", name));
            }
        }
        for name in instance.grids.keys() {
            if !matches!(self.names.get(name), Some(Decl::Grid(_))) {
                return Err(unknown("grid", name));
            }
        }
        Ok(())
    }

    fn dims(&mut self, binders: &[Binder]) -> Result<Vec<Dim>> {
        binders
            .iter()
            .map(|b| match &b.kind {
                BinderKind::Typed { name, ty } => Ok(Dim {
                    name: name.clone(),
                    ty: self.resolve_type(ty)?,
                }),
                _ => Err(err(
                    "E0207",
                    "family indices are written `name: Type`",
                    b.span,
                )),
            })
            .collect()
    }

    fn resolve_type(&self, ty: &TypeExpr) -> Result<Type> {
        Ok(match ty {
            TypeExpr::Int => Type::Int,
            TypeExpr::Bool => Type::Bool,
            TypeExpr::Set(inner) => Type::Set(Box::new(self.resolve_type(inner)?)),
            TypeExpr::Option(inner) => Type::Option(Box::new(self.resolve_type(inner)?)),
            TypeExpr::Tuple(items) => Type::Tuple(
                items
                    .iter()
                    .map(|t| self.resolve_type(t))
                    .collect::<Result<_>>()?,
            ),
            TypeExpr::Named(name, span) => match self.names.get(name) {
                Some(Decl::Enum(e)) => Type::Enum(*e),
                Some(Decl::Subset(s)) => Type::Subset(*s),
                Some(Decl::Domain(d)) => Type::Domain(*d),
                Some(Decl::Grid(g)) => Type::Grid(*g),
                Some(Decl::Choice(c)) => Type::Choice(*c),
                _ => return Err(err("E0208", format!("`{name}` is not a type"), *span)),
            },
        })
    }

    pub(crate) fn conforms(&self, value: &Value, ty: &Type) -> bool {
        match (value, ty) {
            (Value::None, Type::Option(_)) => true,
            (v, Type::Option(inner)) => self.conforms(v, inner),
            (Value::Bool(_), Type::Bool) => true,
            (Value::Int(_), Type::Int) => true,
            (Value::Int(i), Type::Domain(d)) => {
                self.domains[*d as usize].int
                    && self.domains[*d as usize].values.contains(&Value::Int(*i))
            }
            (Value::Sym(d, _), Type::Domain(e)) => d == e,
            (Value::Variant(e, _), Type::Enum(f)) => e == f,
            (Value::Variant(e, v), Type::Subset(s)) => {
                let subset = &self.subsets[*s as usize];
                subset.base == *e && subset.members.contains(v)
            }
            (Value::Cell(g, _), Type::Grid(h)) => g == h,
            (Value::Member(c, _, _), Type::Choice(d)) => c == d,
            (Value::Tuple(items), Type::Tuple(types)) => {
                items.len() == types.len()
                    && items.iter().zip(types).all(|(v, t)| self.conforms(v, t))
            }
            (Value::Set(items), Type::Set(inner)) => items.iter().all(|v| self.conforms(v, inner)),
            _ => false,
        }
    }

    pub(crate) fn ivalue(&self, v: &IValue, ty: &Type, span: Span) -> Result<Value> {
        let bad = || {
            err(
                "E0400",
                format!("instance value does not fit {}", ShowType(ty, self)),
                span,
            )
        };
        Ok(match (v, ty.unwrap_option()) {
            (IValue::None, _) if matches!(ty, Type::Option(_)) => Value::None,
            (IValue::Bool(b), Type::Bool) => Value::Bool(*b),
            (IValue::Int(i), Type::Int) => Value::Int(*i),
            (IValue::Int(i), Type::Domain(d)) if self.domains[*d as usize].int => Value::Int(*i),
            (IValue::Sym(s), Type::Domain(d)) => {
                let Some(index) = self.domains[*d as usize]
                    .symbols
                    .iter()
                    .position(|x| x == s)
                else {
                    return Err(err(
                        "E0401",
                        format!(
                            "`{s}` is not in domain `{}`",
                            self.domains[*d as usize].name
                        ),
                        span,
                    ));
                };
                Value::Sym(*d, index as u32)
            }
            (IValue::Sym(s), Type::Enum(e)) => {
                let Some(index) = self.enums[*e as usize].variants.iter().position(|x| x == s)
                else {
                    return Err(bad());
                };
                Value::Variant(*e, index as u32)
            }
            (IValue::Sym(s), Type::Subset(sub)) => {
                let subset = &self.subsets[*sub as usize];
                let Some(index) = self.enums[subset.base as usize]
                    .variants
                    .iter()
                    .position(|x| x == s)
                else {
                    return Err(bad());
                };
                Value::Variant(subset.base, index as u32)
            }
            (IValue::Cell(x, y, z), Type::Grid(g)) => {
                let Some(index) = self.grids[*g as usize].index([*x, *y, *z]) else {
                    return Err(err(
                        "E0402",
                        format!("cell ({x},{y},{z}) is outside the grid"),
                        span,
                    ));
                };
                Value::Cell(*g, index)
            }
            (IValue::Member(name, payload), Type::Choice(c)) => {
                let choice = &self.choices[*c as usize];
                let Some(m) = choice.members.iter().position(|x| x.name == *name) else {
                    return Err(err(
                        "E0403",
                        format!("`{name}` is not a member of `{}`", choice.name),
                        span,
                    ));
                };
                let params = choice.members[m].params.clone();
                if params.len() != payload.len() {
                    return Err(bad());
                }
                let mut values = Vec::new();
                for (p, (_, t)) in payload.iter().zip(&params) {
                    values.push(self.ivalue(p, t, span)?);
                }
                Value::Member(*c, m as u32, values.into())
            }
            (IValue::Tuple(items), Type::Tuple(types)) if items.len() == types.len() => {
                Value::Tuple(
                    items
                        .iter()
                        .zip(types)
                        .map(|(v, t)| self.ivalue(v, t, span))
                        .collect::<Result<Vec<_>>>()?
                        .into(),
                )
            }
            _ => return Err(bad()),
        })
    }

    // ----- solver families -------------------------------------------------------

    fn ground_families(&mut self, file: &File) -> Result<()> {
        // Register solver-level names first so rules can reference any order.
        for item in &file.items {
            match &item.kind {
                ItemKind::Var { name, index } => {
                    let id = self.vars.len() as u32;
                    self.add_name(name, Decl::Var(id), item.span)?;
                    let index = match index {
                        VarIndex::Binders(b) => VarIndexKind::Dims(self.dims(b)?.into()),
                        VarIndex::Over(rel) => VarIndexKind::Over(u32::MAX - rel.len() as u32), // resolved below
                    };
                    let mut display = None;
                    let mut outside = false;
                    for a in &item.annotations {
                        match a.name.as_str() {
                            "display" => display = Some(display_arg(a)?),
                            "outside" => outside = bool_arg(a)?,
                            "prefer" | "label" => {}
                            other => return Err(unknown_annotation(other, a.span, "var")),
                        }
                    }
                    self.vars.push(VarDef {
                        name: name.clone(),
                        index,
                        display,
                        outside,
                        keys: Vec::new(),
                        lookup: HashMap::default(),
                    });
                }
                ItemKind::Def { name, index, body } => {
                    let id = self.defs.len() as u32;
                    self.add_name(name, Decl::Def(id), item.span)?;
                    let dims = self.dims(index)?;
                    let mut display = None;
                    let mut internal = false;
                    let mut outside = false;
                    for a in &item.annotations {
                        match a.name.as_str() {
                            "display" => display = Some(display_arg(a)?),
                            "internal" => internal = true,
                            "outside" => outside = bool_arg(a)?,
                            "label" => {}
                            other => return Err(unknown_annotation(other, a.span, "def")),
                        }
                    }
                    self.defs.push(DefDef {
                        name: name.clone(),
                        index: dims.into(),
                        body: Arc::new(body.clone()),
                        internal,
                        outside,
                        display,
                        keys: Vec::new(),
                        memo: HashMap::default(),
                        in_progress: HashSet::default(),
                    });
                }
                ItemKind::Relation { name, params } => {
                    let id = self.relations.len() as u32;
                    self.add_name(name, Decl::Relation(id), item.span)?;
                    let mut ps = Vec::new();
                    for (i, p) in params.iter().enumerate() {
                        ps.push((
                            p.name.clone().unwrap_or_else(|| format!("_{i}")),
                            self.resolve_type(&p.ty)?,
                        ));
                    }
                    let mut display = None;
                    for a in &item.annotations {
                        match a.name.as_str() {
                            "display" => display = Some(display_arg(a)?),
                            "label" => {}
                            other => return Err(unknown_annotation(other, a.span, "relation")),
                        }
                    }
                    self.relations.push(RelDef {
                        name: name.clone(),
                        params: ps.into(),
                        display,
                        contribs: HashMap::default(),
                        contrib_order: Vec::new(),
                        frozen: false,
                        tuples: Vec::new(),
                        lookup: HashMap::default(),
                        field_index: HashMap::default(),
                    });
                }
                ItemKind::Int { name, index, range } => {
                    let id = self.ints.len() as u32;
                    self.add_name(name, Decl::Int(id), item.span)?;
                    let dims = self.dims(index)?;
                    let lo = self.const_int(&range.lo)?;
                    let mut hi = self.const_int(&range.hi)?;
                    if !range.inclusive {
                        hi -= 1;
                    }
                    if hi < lo {
                        return Err(err("E0300", "empty integer range", item.span));
                    }
                    let mut display = None;
                    let mut outside = None;
                    for a in &item.annotations {
                        match a.name.as_str() {
                            "display" => display = Some(display_arg(a)?),
                            "encoding" => {
                                let encoding = name_arg(a)?;
                                if encoding != "order" {
                                    return Err(err(
                                        "E0705",
                                        format!(
                                            "integers support `@encoding(order)`, not `{encoding}`"
                                        ),
                                        a.span,
                                    ));
                                }
                            }
                            "outside" => {
                                let Some(AnnArg::Pos(e)) = a.args.first() else {
                                    return Err(err("E0702", "`@outside` takes a value", a.span));
                                };
                                outside = Some(self.const_int(e)?);
                            }
                            "label" => {}
                            other => return Err(unknown_annotation(other, a.span, "int")),
                        }
                    }
                    self.ints.push(IntDef {
                        name: name.clone(),
                        index: dims.into(),
                        lo,
                        hi,
                        display,
                        outside,
                        grounded: false,
                        keys: Vec::new(),
                        lookup: HashMap::default(),
                    });
                }
                ItemKind::Rule { label, body } => {
                    let mut guarded = None;
                    let mut fold = false;
                    for a in &item.annotations {
                        match a.name.as_str() {
                            "guarded" => {
                                let mut per = Vec::new();
                                for arg in &a.args {
                                    let AnnArg::Named(key, value) = arg else {
                                        return Err(err(
                                            "E0702",
                                            "write `@guarded(per = binders)`",
                                            a.span,
                                        ));
                                    };
                                    if key != "per" {
                                        return Err(err(
                                            "E0702",
                                            format!("unknown `@guarded` argument `{key}`"),
                                            a.span,
                                        ));
                                    }
                                    match &value.kind {
                                        ExprKind::Name(n) => per.push(n.clone()),
                                        ExprKind::Tuple(items) => {
                                            for i in items {
                                                let ExprKind::Name(n) = &i.kind else {
                                                    return Err(err(
                                                        "E0702",
                                                        "`per` lists binder names",
                                                        i.span,
                                                    ));
                                                };
                                                per.push(n.clone());
                                            }
                                        }
                                        _ => {
                                            return Err(err(
                                                "E0702",
                                                "`per` lists binder names",
                                                value.span,
                                            ))
                                        }
                                    }
                                }
                                guarded = Some(per);
                            }
                            "fold" => fold = true,
                            "label" => {}
                            other => return Err(unknown_annotation(other, a.span, "rule")),
                        }
                    }
                    let has_contrib =
                        contains(body, |s| matches!(s.kind, StmtKind::Contribute { .. }));
                    if fold && has_contrib {
                        return Err(err(
                            "E0706",
                            "a `@fold` rule cannot contribute to relations",
                            item.span,
                        ));
                    }
                    self.rules.push(RuleDef {
                        label: label.clone(),
                        body: Arc::new(body.clone()),
                        guarded,
                        fold,
                        has_contrib,
                        has_require: contains(body, |s| matches!(s.kind, StmtKind::Require(_))),
                        span: item.span,
                        clauses: 0,
                        time: std::time::Duration::ZERO,
                    });
                }
                ItemKind::Include(path) => {
                    return Err(err(
                        "E0800",
                        format!("`include \"{path}\"` is not supported yet"),
                        item.span,
                    )
                    .with_help("inline the included declarations"));
                }
                ItemKind::Objective { minimize, expr } => {
                    self.objectives.push((*minimize, expr.clone(), item.span));
                }
                _ => {}
            }
        }
        // Resolve `var V over R`.
        for item in &file.items {
            if let ItemKind::Var {
                name,
                index: VarIndex::Over(rel),
            } = &item.kind
            {
                let Some(Decl::Relation(r)) = self.names.get(rel) else {
                    return Err(err(
                        "E0209",
                        format!("`{rel}` is not a relation"),
                        item.span,
                    ));
                };
                let Decl::Var(v) = self.names[name] else {
                    unreachable!()
                };
                self.vars[v as usize].index = VarIndexKind::Over(*r);
            }
        }

        // Choices, index variables, integers, in declaration order.
        for item in &file.items {
            match &item.kind {
                ItemKind::Choice { name, .. } => {
                    let Decl::Choice(id) = self.names[name] else {
                        unreachable!()
                    };
                    self.ground_choice(id, item.span)?;
                }
                ItemKind::Var { name, .. } => {
                    let Decl::Var(id) = self.names[name] else {
                        unreachable!()
                    };
                    if let VarIndexKind::Dims(dims) = &self.vars[id as usize].index {
                        let dims = dims.clone();
                        for key in self.product(&dims)? {
                            let k = self.vars[id as usize].keys.len() as u32;
                            let lit = self.named_var(NameRef::Var { var: id, key: k });
                            self.vars[id as usize].keys.push(key.clone());
                            self.vars[id as usize].lookup.insert(key, lit);
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Creates an integer family's order-encoding variables. Families are
    /// created on first use, so variables are numbered in the order the rules
    /// need them (CaDiCaL's initial decision order follows numbering).
    fn ground_int(&mut self, id: u32) -> Result<()> {
        if self.ints[id as usize].grounded {
            return Ok(());
        }
        self.ints[id as usize].grounded = true;
        let (dims, lo, hi, name) = {
            let def = &self.ints[id as usize];
            (def.index.clone(), def.lo, def.hi, def.name.clone())
        };
        let saved = self.encoder.origin;
        self.encoder.origin = self.new_origin(format!("정의: {name}"), String::new());
        for key in self.product(&dims)? {
            let k = self.ints[id as usize].keys.len() as u32;
            let mut lits = Vec::new();
            for v in lo + 1..=hi {
                let lit = self.named_var(NameRef::IntGe {
                    int: id,
                    key: k,
                    value: v,
                });
                if let Some(&prev) = lits.last() {
                    self.encoder.clause(&[-lit, prev]);
                }
                lits.push(lit);
            }
            self.ints[id as usize].keys.push(key.clone());
            self.ints[id as usize].lookup.insert(key, lits.into());
        }
        self.encoder.origin = saved;
        Ok(())
    }

    fn ground_choice(&mut self, id: u32, span: Span) -> Result<()> {
        let dims = self.choices[id as usize].index.clone();
        let name = self.choices[id as usize].name.clone();
        let origin = self.new_origin(format!("선택: {name} 는 정확히 하나"), String::new());
        for key in self.product(&dims)? {
            let k = self.choices[id as usize].keys.len() as u32;
            let mut env: Env = dims
                .iter()
                .zip(&key)
                .map(|(d, v)| (d.name.clone(), Val::C(v.clone())))
                .collect();
            let mut options = Vec::new();
            let member_count = self.choices[id as usize].members.len();
            for m in 0..member_count {
                let (params, guard) = {
                    let member = &self.choices[id as usize].members[m];
                    (member.params.clone(), member.guard.clone())
                };
                let payload_dims: Vec<Dim> = params
                    .iter()
                    .map(|(n, t)| Dim {
                        name: n.clone(),
                        ty: t.clone(),
                    })
                    .collect();
                for payload in self.product(&payload_dims)? {
                    let base = env.len();
                    for (d, v) in payload_dims.iter().zip(&payload) {
                        env.push((d.name.clone(), Val::C(v.clone())));
                    }
                    let exists = match &guard {
                        Some(g) => self.const_bool(g, &mut env)?,
                        None => true,
                    };
                    env.truncate(base);
                    if !exists {
                        continue;
                    }
                    let option = options.len() as u32;
                    let preferred = self.choices[id as usize].prefer == Some(m as u32);
                    let lit = if preferred {
                        let var = self.named_var(NameRef::Occupied { choice: id, key: k });
                        -var
                    } else {
                        let var = self.named_var(NameRef::Option {
                            choice: id,
                            key: k,
                            option,
                        });
                        var
                    };
                    options.push(OptionEntry {
                        member: m as u32,
                        payload: payload.into(),
                        lit,
                    });
                }
            }
            if options.is_empty() {
                return Err(err(
                    "E0310",
                    format!("choice `{name}` has no possible member at some index"),
                    span,
                ));
            }
            let lits: Vec<Lit> = options.iter().map(|o| o.lit).collect();
            self.encoder.origin = origin;
            self.encoder.clause(&lits);
            self.encoder.at_most(&lits, 1, false);
            self.encoder.origin = 0;
            let choice = &mut self.choices[id as usize];
            choice.keys.push(key.clone());
            choice.lookup.insert(key, k);
            choice.options.push(options);
        }
        Ok(())
    }

    /// A fresh variable that stands for `name`.
    fn named_var(&mut self, name: NameRef) -> Lit {
        let index = self.name_refs.len() as u32;
        self.name_refs.push(name);
        self.encoder.new_var(VarOrigin::Named(index))
    }

    /// Names an existing literal: an auxiliary takes the name, anything else
    /// (a choice option, a negated variable) records an alias.
    fn name_lit(&mut self, lit: Lit, name: NameRef) {
        let index = self.name_refs.len() as u32;
        self.name_refs.push(name);
        let var = lit.unsigned_abs() as usize;
        if lit > 0 && matches!(self.encoder.var_origin[var], VarOrigin::Aux(_)) {
            self.encoder.var_origin[var] = VarOrigin::Named(index);
        } else {
            self.aliases.entry(lit).or_default().push(index);
        }
    }

    fn new_origin(&mut self, label: String, valuation: String) -> u32 {
        self.origins.push((label, valuation));
        (self.origins.len() - 1) as u32
    }

    fn product(&mut self, dims: &[Dim]) -> Result<Vec<Vec<Value>>> {
        let mut out = vec![Vec::new()];
        for dim in dims {
            let values = self.values_of(&dim.ty)?;
            let mut next = Vec::with_capacity(out.len() * values.len());
            for prefix in &out {
                for v in values.iter() {
                    let mut key = prefix.clone();
                    key.push(v.clone());
                    next.push(key);
                }
            }
            out = next;
        }
        Ok(out)
    }

    fn values_of(&mut self, ty: &Type) -> Result<Arc<[Value]>> {
        if let Some(v) = self.type_values.get(ty) {
            return Ok(v.clone());
        }
        let values: Vec<Value> = match ty {
            Type::Bool => vec![Value::Bool(false), Value::Bool(true)],
            Type::Domain(d) => self.domains[*d as usize].values.clone(),
            Type::Enum(e) => (0..self.enums[*e as usize].variants.len() as u32)
                .map(|v| Value::Variant(*e, v))
                .collect(),
            Type::Subset(s) => {
                let subset = &self.subsets[*s as usize];
                subset
                    .members
                    .iter()
                    .map(|&v| Value::Variant(subset.base, v))
                    .collect()
            }
            Type::Grid(g) => (0..self.grids[*g as usize].len())
                .map(|i| Value::Cell(*g, i))
                .collect(),
            Type::Choice(c) => {
                let mut out = Vec::new();
                let members = self.choices[*c as usize].members.len();
                for m in 0..members {
                    let dims: Vec<Dim> = self.choices[*c as usize].members[m]
                        .params
                        .iter()
                        .map(|(n, t)| Dim {
                            name: n.clone(),
                            ty: t.clone(),
                        })
                        .collect();
                    for payload in self.product(&dims)? {
                        out.push(Value::Member(*c, m as u32, payload.into()));
                    }
                }
                out
            }
            Type::Tuple(types) => {
                let dims: Vec<Dim> = types
                    .iter()
                    .map(|t| Dim {
                        name: Name::from(""),
                        ty: t.clone(),
                    })
                    .collect();
                self.product(&dims)?
                    .into_iter()
                    .map(|k| Value::Tuple(k.into()))
                    .collect()
            }
            other => {
                return Err(err_nospan(format!(
                    "cannot enumerate values of type {}",
                    ShowType(other, self)
                )))
            }
        };
        let values: Arc<[Value]> = values.into();
        self.type_values.insert(ty.clone(), values.clone());
        Ok(values)
    }

    // ----- rules -------------------------------------------------------------

    fn run_rules(&mut self) -> Result<()> {
        // Pass 0: `@fold` rules fix literals before anything uses them.
        if self.options.no_fold {
            for rule in &mut self.rules {
                rule.fold = false;
            }
        }
        if self.rules.iter().any(|r| r.fold) {
            self.mode = Mode::Emit;
            self.folding = true;
            for r in 0..self.rules.len() {
                if self.rules[r].fold {
                    let before = self.encoder.clause_count;
                    self.run_rule(r)?;
                    self.rules[r].clauses = self.encoder.clause_count - before;
                }
            }
            self.folding = false;
            self.fix_choice_siblings();
        }
        // Pass 1: collect relation contributions.
        self.mode = Mode::Collect;
        for r in 0..self.rules.len() {
            if self.rules[r].has_contrib {
                self.run_rule(r)?;
            }
        }
        // Freeze relations.
        for rel in 0..self.relations.len() {
            self.freeze_relation(rel as u32)?;
        }
        // Variables over relations.
        for v in 0..self.vars.len() {
            if let VarIndexKind::Over(rel) = self.vars[v].index {
                let tuples: Vec<Vec<Value>> = self.relations[rel as usize]
                    .tuples
                    .iter()
                    .map(|(t, _)| t.clone())
                    .collect();
                for tuple in tuples {
                    let k = self.vars[v].keys.len() as u32;
                    let lit = self.named_var(NameRef::Var {
                        var: v as u32,
                        key: k,
                    });
                    self.vars[v].keys.push(tuple.clone());
                    self.vars[v].lookup.insert(tuple, lit);
                }
            }
        }
        // Pass 2: emit constraints.
        self.mode = Mode::Emit;
        for r in 0..self.rules.len() {
            if self.rules[r].has_require && !self.rules[r].fold {
                let before = self.encoder.clause_count;
                self.run_rule(r)?;
                self.rules[r].clauses = self.encoder.clause_count - before;
                if self.rules[r].clauses == 0 {
                    self.warnings.push(Diagnostic::warning(
                        "W0101",
                        format!("rule \"{}\" produced no clauses", self.rules[r].label),
                        self.rules[r].span,
                    ));
                }
            }
        }
        // Exported definitions and integers the rules never touched, so the
        // host can look up every index; the rest were created on first use.
        for d in 0..self.defs.len() {
            if self.defs[d].internal {
                continue;
            }
            let dims = self.defs[d].index.clone();
            for key in self.product(&dims)? {
                self.get_def(d as u32, &key, Span::default())?;
            }
        }
        for i in 0..self.ints.len() {
            self.ground_int(i as u32)?;
        }
        self.ground_objectives()?;
        self.encoder.origin = 0;
        self.encoder.guard = None;
        Ok(())
    }

    /// A choice takes exactly one option, so an option fixed true fixes the
    /// others of the same key false.
    fn fix_choice_siblings(&mut self) {
        for choice in &self.choices {
            for options in &choice.options {
                if options
                    .iter()
                    .any(|o| self.encoder.value(o.lit) == Some(true))
                {
                    for o in options {
                        if self.encoder.value(o.lit).is_none() {
                            self.encoder.fix(-o.lit);
                        }
                    }
                }
            }
        }
    }

    /// Sums every `minimize` (and negated `maximize`) item into positive
    /// weighted literals; terms are encoded exactly (both directions), so
    /// the cost of a model can be read from those literals.
    fn ground_objectives(&mut self) -> Result<()> {
        if self.objectives.is_empty() {
            return Ok(());
        }
        let objectives = std::mem::take(&mut self.objectives);
        self.encoder.origin = self.new_origin("목적함수".to_owned(), String::new());
        let mut weights = HashMap::<Lit, i64>::default();
        let mut order = Vec::new();
        let mut constant = 0i64;
        for (minimize, expr, span) in objectives {
            let value = self.eval(&expr, &mut Vec::new(), Some(&Type::Int))?;
            let term = match value {
                Val::I(term) => term,
                Val::C(Value::Int(k)) => IntTerm::Const(k),
                _ => {
                    return Err(err(
                        "E0442",
                        "an objective must be an integer expression",
                        span,
                    ))
                }
            };
            let (terms, offset) = into_linear(if minimize { term } else { scale(term, -1) });
            constant += offset;
            for (weight, formula) in terms {
                let (weight, formula) = match formula {
                    F::Const(true) => {
                        constant += weight;
                        continue;
                    }
                    F::Const(false) => continue,
                    f if weight < 0 => {
                        // w·[f] = w + |w|·[not f]
                        constant += weight;
                        (-weight, f.not())
                    }
                    f => (weight, f),
                };
                let lit = self.encoder.lit_of(&formula, Pol::Both);
                match lit {
                    1 => constant += weight,
                    -1 => {}
                    _ => {
                        if !weights.contains_key(&lit) {
                            order.push(lit);
                        }
                        *weights.entry(lit).or_default() += weight;
                    }
                }
            }
        }
        let terms = order
            .into_iter()
            .map(|lit| (weights[&lit] as u64, lit))
            .filter(|(weight, _)| *weight > 0)
            .collect();
        self.objective = Some((terms, constant));
        Ok(())
    }

    fn run_rule(&mut self, r: usize) -> Result<()> {
        let started = std::time::Instant::now();
        let result = self.run_rule_body(r);
        self.rules[r].time += started.elapsed();
        result
    }

    fn run_rule_body(&mut self, r: usize) -> Result<()> {
        self.current_rule = Some(r);
        let label = self.rules[r].label.clone();
        let origin = self.new_origin(label, String::new());
        self.encoder.origin = origin;
        let body = self.rules[r].body.clone();
        let mut env = Vec::new();
        let result = self.exec(&body, &mut env);
        self.current_rule = None;
        self.encoder.origin = 0;
        self.encoder.guard = None;
        result
    }

    fn exec(&mut self, stmts: &[Stmt], env: &mut Env) -> Result<()> {
        let base = env.len();
        for stmt in stmts {
            self.exec_stmt(stmt, env)?;
        }
        env.truncate(base);
        Ok(())
    }

    fn exec_stmt(&mut self, stmt: &Stmt, env: &mut Env) -> Result<()> {
        match &stmt.kind {
            StmtKind::Require(expr) => {
                if self.mode != Mode::Emit {
                    return Ok(());
                }
                let mut encoding = None;
                for a in &stmt.annotations {
                    match a.name.as_str() {
                        "encoding" => encoding = Some(name_arg(a)?),
                        other => return Err(unknown_annotation(other, a.span, "require")),
                    }
                }
                let value = self.eval(expr, env, Some(&Type::Bool))?;
                let mut f = self.to_formula(value, expr.span)?;
                if let Some(enc) = encoding {
                    if !matches!(enc.as_str(), "pairwise" | "seqcounter" | "auto") {
                        return Err(err(
                            "E0705",
                            format!("unsupported cardinality encoding `{enc}`"),
                            stmt.span,
                        ));
                    }
                    set_card_encoding(&mut f, &enc);
                }
                self.set_guard(env, stmt.span)?;
                if self.options.provenance {
                    let valuation = self.valuation(env);
                    let label = self.rules[self.current_rule.unwrap()].label.clone();
                    self.encoder.origin = self.new_origin(label, valuation);
                }
                if self.folding {
                    let units = match &f {
                        // A contradiction is emitted as the empty clause below.
                        F::Const(_) => Vec::new(),
                        F::Lit(l) => vec![*l],
                        F::And(items) if items.iter().all(|i| matches!(i, F::Lit(_))) => items
                            .iter()
                            .map(|i| match i {
                                F::Lit(l) => *l,
                                _ => unreachable!(),
                            })
                            .collect(),
                        _ => {
                            return Err(err(
                                "E0706",
                                "a `@fold` rule may only require literals",
                                expr.span,
                            ))
                        }
                    };
                    for lit in units {
                        self.encoder.fix(lit);
                    }
                }
                self.encoder.require(f);
                Ok(())
            }
            StmtKind::Forall {
                binders,
                guard,
                body,
            } => {
                if self.mode == Mode::Collect {
                    let over_relation = binders
                        .iter()
                        .any(|b| matches!(&b.kind, BinderKind::Tuple { relation, .. } if matches!(self.names.get(relation), Some(Decl::Relation(_)))));
                    let body_contributes =
                        contains(body, |s| matches!(s.kind, StmtKind::Contribute { .. }));
                    if !body_contributes {
                        return Ok(());
                    }
                    if over_relation {
                        return Err(err(
                            "E0503",
                            "a rule cannot iterate over a relation and contribute to relations in the same loop",
                            stmt.span,
                        ));
                    }
                } else if !contains(body, |s| matches!(s.kind, StmtKind::Require(_))) {
                    return Ok(());
                }
                self.each_binding(binders, guard.as_ref(), env, &mut |p, env| {
                    p.exec(body, env)
                })
            }
            StmtKind::If {
                cond,
                then,
                otherwise,
            } => {
                if self.const_bool(cond, env)? {
                    self.exec(then, env)
                } else {
                    self.exec(otherwise, env)
                }
            }
            StmtKind::Let { name, value } => {
                let v = self.eval(value, env, None)?;
                env.push((name.clone(), v));
                Ok(())
            }
            StmtKind::Contribute {
                relation,
                args,
                value,
            } => {
                if self.mode != Mode::Collect {
                    return Ok(());
                }
                let Some(Decl::Relation(rel)) = self.names.get(relation).copied() else {
                    return Err(err(
                        "E0209",
                        format!("`{relation}` is not a relation"),
                        stmt.span,
                    ));
                };
                let params = self.relations[rel as usize].params.clone();
                if params.len() != args.len() {
                    return Err(err(
                        "E0410",
                        format!("`{relation}` takes {} fields", params.len()),
                        stmt.span,
                    ));
                }
                // The value first: most contributions fold to false and need
                // no key.
                let v = self.eval(value, env, Some(&Type::Bool))?;
                let f = self.to_formula(v, value.span)?;
                if f == F::Const(false) {
                    return Ok(());
                }
                let mut key = Vec::with_capacity(args.len());
                for (arg, (_, ty)) in args.iter().zip(params.iter()) {
                    let v = self.const_eval(arg, env, Some(ty))?;
                    if v.is_none() {
                        return Ok(());
                    }
                    if !self.conforms(&v, ty) {
                        return Err(err(
                            "E0400",
                            format!("expected {}", ShowType(ty, self)),
                            arg.span,
                        ));
                    }
                    key.push(v);
                }
                let def = &mut self.relations[rel as usize];
                if !def.contribs.contains_key(&key) {
                    def.contrib_order.push(key.clone());
                }
                def.contribs.entry(key).or_default().push(f);
                Ok(())
            }
        }
    }

    fn set_guard(&mut self, env: &Env, span: Span) -> Result<()> {
        let r = self.current_rule.unwrap();
        let Some(per) = self.rules[r].guarded.clone() else {
            self.encoder.guard = None;
            return Ok(());
        };
        if !self.options.guards {
            self.encoder.guard = None;
            return Ok(());
        }
        let mut key = Vec::new();
        for name in &per {
            let Some((_, Val::C(v))) = env.iter().rev().find(|(n, _)| same(n, name)) else {
                return Err(err(
                    "E0706",
                    format!("`@guarded(per = ...)` names `{name}`, which is not bound here"),
                    span,
                ));
            };
            key.push(v.clone());
        }
        let lit = if let Some(&lit) = self.guard_lookup.get(&(r, key.clone())) {
            lit
        } else {
            let index = self.guards.len() as u32;
            let lit = self.encoder.new_var(VarOrigin::Guard(index));
            self.guards.push(GuardDef {
                rule: self.rules[r].label.clone(),
                names: per.clone(),
                key: key.clone(),
                lit,
            });
            self.guard_lookup.insert((r, key), lit);
            lit
        };
        self.encoder.guard = Some(lit);
        Ok(())
    }

    fn valuation(&self, env: &Env) -> String {
        env.iter()
            .filter_map(|(n, v)| match v {
                Val::C(value) => Some(format!("{n}={}", Show(value, self))),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn freeze_relation(&mut self, rel: u32) -> Result<()> {
        let order = std::mem::take(&mut self.relations[rel as usize].contrib_order);
        let mut contribs = std::mem::take(&mut self.relations[rel as usize].contribs);
        let name = self.relations[rel as usize].name.clone();
        let origin = self.new_origin(format!("관계: {name}"), String::new());
        self.encoder.origin = origin;
        for key in order {
            let parts = contribs.remove(&key).unwrap_or_default();
            let f = F::or(parts);
            let lit = match &f {
                F::Const(false) => continue,
                F::Const(true) => 1,
                _ => self.encoder.lit_of(&f, Pol::Both),
            };
            let index = self.relations[rel as usize].tuples.len() as u32;
            if lit != 1 {
                self.name_lit(lit, NameRef::Rel { rel, tuple: index });
            }
            let def = &mut self.relations[rel as usize];
            for (field, value) in key.iter().enumerate() {
                def.field_index
                    .entry((field, value.clone()))
                    .or_default()
                    .push(index);
            }
            def.lookup.insert(key.clone(), index);
            def.tuples.push((key, lit));
        }
        self.encoder.origin = 0;
        self.relations[rel as usize].frozen = true;
        Ok(())
    }

    fn get_def(&mut self, d: u32, key: &[Value], span: Span) -> Result<F> {
        if let Some(f) = self.defs[d as usize].memo.get(key) {
            return Ok(f.clone());
        }
        let key = key.to_vec();
        if !self.defs[d as usize].in_progress.insert(key.clone()) {
            return Err(err(
                "E0501",
                format!(
                    "`{}` is defined in terms of itself",
                    self.defs[d as usize].name
                ),
                span,
            ));
        }
        let dims = self.defs[d as usize].index.clone();
        let mut env: Env = dims
            .iter()
            .zip(&key)
            .map(|(dim, v)| (dim.name.clone(), Val::C(v.clone())))
            .collect();
        let body = self.defs[d as usize].body.clone();
        let saved_origin = self.encoder.origin;
        let saved_guard = self.encoder.guard.take();
        let origin = self.new_origin(
            format!("정의: {}", self.defs[d as usize].name),
            String::new(),
        );
        self.encoder.origin = origin;
        let value = self.eval(&body, &mut env, Some(&Type::Bool))?;
        let f = self.to_formula(value, body.span)?;
        let result = if self.defs[d as usize].internal {
            f
        } else {
            match f {
                F::Const(b) => F::Const(b),
                other => {
                    let lit = self.encoder.lit_of(&other, Pol::Both);
                    let k = self.defs[d as usize].keys.len() as u32;
                    self.defs[d as usize].keys.push(key.clone());
                    self.name_lit(lit, NameRef::Def { def: d, key: k });
                    F::Lit(lit)
                }
            }
        };
        self.encoder.origin = saved_origin;
        self.encoder.guard = saved_guard;
        let def = &mut self.defs[d as usize];
        def.in_progress.remove(&key);
        def.memo.insert(key, result.clone());
        Ok(result)
    }

    // ----- binder iteration ------------------------------------------------------

    fn each_binding(
        &mut self,
        binders: &[Binder],
        guard: Option<&Expr>,
        env: &mut Env,
        f: &mut dyn FnMut(&mut Program, &mut Env) -> Result<()>,
    ) -> Result<()> {
        self.bind_from(binders, 0, guard, env, f)
    }

    fn bind_from(
        &mut self,
        binders: &[Binder],
        index: usize,
        guard: Option<&Expr>,
        env: &mut Env,
        f: &mut dyn FnMut(&mut Program, &mut Env) -> Result<()>,
    ) -> Result<()> {
        if index == binders.len() {
            if let Some(g) = guard {
                if !self.const_bool(g, env)? {
                    return Ok(());
                }
            }
            return f(self, env);
        }
        let binder = &binders[index];
        match &binder.kind {
            BinderKind::Typed { name, ty } => {
                let ty = self.resolve_type(ty)?;
                let values = self.values_of(&ty).map_err(|d| d.with_span(binder.span))?;
                for v in values.iter() {
                    env.push((name.clone(), Val::C(v.clone())));
                    let r = self.bind_from(binders, index + 1, guard, env, f);
                    env.pop();
                    r?;
                }
                Ok(())
            }
            BinderKind::In { name, source } => {
                // Sets are shared slices; iterate them without copying.
                let values: Arc<[Value]> = match source {
                    InSource::Range(range) => {
                        let lo = self.const_int_env(&range.lo, env)?;
                        let hi = self.const_int_env(&range.hi, env)?;
                        let hi = if range.inclusive { hi } else { hi - 1 };
                        (lo..=hi).map(Value::Int).collect()
                    }
                    InSource::Expr(e) => match self.eval(e, env, None)? {
                        Val::C(Value::Set(items)) => items,
                        Val::Type(t) => self.values_of(&t)?,
                        _ => return Err(err("E0411", "expected a set to iterate over", e.span)),
                    },
                };
                for v in values.iter() {
                    env.push((name.clone(), Val::C(v.clone())));
                    let r = self.bind_from(binders, index + 1, guard, env, f);
                    env.pop();
                    r?;
                }
                Ok(())
            }
            BinderKind::Tuple { names, relation } => {
                let candidates = self.tuple_candidates(relation, names, guard, env, binder.span)?;
                for tuple in candidates {
                    if tuple.len() != names.len() {
                        return Err(err(
                            "E0410",
                            format!("`{relation}` has {} fields", tuple.len()),
                            binder.span,
                        ));
                    }
                    let base = env.len();
                    for (n, v) in names.iter().zip(tuple.iter()) {
                        if let Some(n) = n {
                            env.push((n.clone(), Val::C(v.clone())));
                        }
                    }
                    let r = self.bind_from(binders, index + 1, guard, env, f);
                    env.truncate(base);
                    r?;
                }
                Ok(())
            }
        }
    }

    /// Tuples of a relation or extern fact, narrowed by `field == expr`
    /// conjuncts of the guard when the expression is already known.
    fn tuple_candidates(
        &mut self,
        source: &str,
        names: &[Option<Name>],
        guard: Option<&Expr>,
        env: &mut Env,
        span: Span,
    ) -> Result<Vec<Arc<[Value]>>> {
        match self.names.get(source).copied() {
            Some(Decl::Relation(rel)) => {
                if !self.relations[rel as usize].frozen {
                    return Err(err(
                        "E0503",
                        format!(
                            "relation `{source}` is used before all contributions are collected"
                        ),
                        span,
                    ));
                }
                let mut filter: Option<(usize, Value)> = None;
                if let Some(g) = guard {
                    let mut conjuncts = Vec::new();
                    flatten_and(g, &mut conjuncts);
                    for c in conjuncts {
                        let ExprKind::Binary(BinOp::Eq, l, r) = &c.kind else {
                            continue;
                        };
                        for (field_side, other) in [(l, r), (r, l)] {
                            let ExprKind::Name(n) = &field_side.kind else {
                                continue;
                            };
                            let Some(field) =
                                names.iter().position(|x| x.as_deref() == Some(n.as_str()))
                            else {
                                continue;
                            };
                            if mentions_any(other, names) {
                                continue;
                            }
                            let ty = self.relations[rel as usize].params[field].1.clone();
                            if let Ok(Val::C(v)) = self.eval(other, env, Some(&ty)) {
                                filter = Some((field, v));
                                break;
                            }
                        }
                        if filter.is_some() {
                            break;
                        }
                    }
                }
                let def = &self.relations[rel as usize];
                Ok(match filter {
                    Some(key) => def
                        .field_index
                        .get(&key)
                        .map(|idx| {
                            idx.iter()
                                .map(|&i| def.tuples[i as usize].0.clone().into())
                                .collect()
                        })
                        .unwrap_or_default(),
                    None => def.tuples.iter().map(|(t, _)| t.clone().into()).collect(),
                })
            }
            Some(Decl::Fact(fact)) => match &self.facts[fact as usize].kind {
                FactKind::Extern { tuples, .. } => {
                    Ok(tuples.iter().map(|t| t.clone().into()).collect())
                }
                FactKind::Derived { .. } => {
                    let types: Vec<Dim> = self.facts[fact as usize]
                        .params
                        .iter()
                        .map(|(n, t)| Dim {
                            name: n.clone().unwrap_or_else(|| Name::from("")),
                            ty: t.clone(),
                        })
                        .collect();
                    let mut out = Vec::new();
                    for key in self.product(&types)? {
                        if self.fact_holds(fact, &key, span)? {
                            out.push(key.into());
                        }
                    }
                    Ok(out)
                }
            },
            _ => Err(err(
                "E0209",
                format!("`{source}` is not a relation or fact"),
                span,
            )),
        }
    }

    fn fact_holds(&mut self, fact: u32, args: &[Value], span: Span) -> Result<bool> {
        if args.iter().any(Value::is_none) {
            return Ok(false);
        }
        match &self.facts[fact as usize].kind {
            FactKind::Extern { set, .. } => Ok(set.contains(args)),
            FactKind::Derived { body, memo } => {
                if let Some(&b) = memo.get(args) {
                    return Ok(b);
                }
                let body = body.clone();
                let mut env: Env = self.facts[fact as usize]
                    .params
                    .iter()
                    .zip(args)
                    .map(|((n, _), v)| (n.clone().unwrap(), Val::C(v.clone())))
                    .collect();
                let b = self.const_bool(&body, &mut env).map_err(|d| {
                    d.with_note(format!(
                        "while evaluating fact `{}`",
                        self.facts[fact as usize].name
                    ))
                })?;
                let _ = span;
                if let FactKind::Derived { memo, .. } = &mut self.facts[fact as usize].kind {
                    memo.insert(args.to_vec(), b);
                }
                Ok(b)
            }
        }
    }

    // ----- evaluation ----------------------------------------------------------

    fn const_eval(&mut self, expr: &Expr, env: &mut Env, expect: Option<&Type>) -> Result<Value> {
        match self.eval(expr, env, expect)? {
            Val::C(v) => Ok(v),
            Val::Type(_) => Err(err("E0412", "expected a value, found a type", expr.span)),
            _ => Err(err("E0304", "this must be known at grounding time", expr.span)
                .with_help("solver variables cannot appear in guards, indices, or arguments of facts and functions")),
        }
    }

    fn const_bool(&mut self, expr: &Expr, env: &mut Env) -> Result<bool> {
        match self.const_eval(expr, env, Some(&Type::Bool))? {
            Value::Bool(b) => Ok(b),
            other => Err(err(
                "E0400",
                format!("expected a bool, found `{}`", Show(&other, self)),
                expr.span,
            )),
        }
    }

    fn const_int(&mut self, expr: &Expr) -> Result<i64> {
        self.const_int_env(expr, &mut Vec::new())
    }

    fn const_int_env(&mut self, expr: &Expr, env: &mut Env) -> Result<i64> {
        match self.const_eval(expr, env, Some(&Type::Int))? {
            Value::Int(i) => Ok(i),
            Value::None => Err(
                err("E0412", "this optional value is `none` here", expr.span)
                    .with_help("guard the use with `has(...)`"),
            ),
            other => Err(err(
                "E0400",
                format!("expected an integer, found `{}`", Show(&other, self)),
                expr.span,
            )),
        }
    }

    fn to_formula(&self, v: Val, span: Span) -> Result<F> {
        match v {
            Val::F(f) => Ok(f),
            Val::C(Value::Bool(b)) => Ok(F::Const(b)),
            Val::C(other) => Err(err(
                "E0400",
                format!("expected a formula, found `{}`", Show(&other, self)),
                span,
            )),
            Val::I(_) => Err(
                err("E0400", "expected a formula, found an integer term", span)
                    .with_help("compare it, e.g. `count(...) <= 3`"),
            ),
            Val::Choice(..) => Err(err("E0400", "a choice must be tested with `is`", span)
                .with_help("write `Kind[c] is Member`")),
            Val::Type(_) => Err(err("E0400", "expected a formula, found a type", span)),
        }
    }

    fn eval(&mut self, expr: &Expr, env: &mut Env, expect: Option<&Type>) -> Result<Val> {
        let span = expr.span;
        match &expr.kind {
            ExprKind::Int(i) => Ok(Val::C(Value::Int(*i))),
            ExprKind::Str(s) => Ok(Val::C(Value::Str(s.as_str().into()))),
            ExprKind::Bool(b) => Ok(Val::C(Value::Bool(*b))),
            ExprKind::None => Ok(Val::C(Value::None)),
            ExprKind::Name(name) => self.eval_name(name, env, expect, span),
            ExprKind::Tuple(items) => {
                let types = match expect.map(Type::unwrap_option) {
                    Some(Type::Tuple(t)) if t.len() == items.len() => Some(t.clone()),
                    _ => None,
                };
                let mut out = Vec::new();
                for (i, item) in items.iter().enumerate() {
                    let t = types.as_ref().map(|t| &t[i]);
                    out.push(self.const_eval(item, env, t)?);
                }
                Ok(Val::C(Value::Tuple(out.into())))
            }
            ExprKind::List(items) => {
                let inner = match expect.map(Type::unwrap_option) {
                    Some(Type::Set(t)) => Some((**t).clone()),
                    _ => None,
                };
                let mut out = Vec::new();
                for item in items {
                    out.push(self.const_eval(item, env, inner.as_ref())?);
                }
                Ok(Val::C(Value::Set(out.into())))
            }
            ExprKind::ListComp {
                expr: inner,
                binders,
                guard,
            } => {
                let elem = match expect.map(Type::unwrap_option) {
                    Some(Type::Set(t)) => Some((**t).clone()),
                    _ => None,
                };
                let mut out = Vec::new();
                self.each_binding(binders, guard.as_deref(), env, &mut |p, env| {
                    out.push(p.const_eval(inner, env, elem.as_ref())?);
                    Ok(())
                })?;
                Ok(Val::C(Value::Set(out.into())))
            }
            ExprKind::Not(inner) => match self.eval(inner, env, Some(&Type::Bool))? {
                Val::C(Value::Bool(b)) => Ok(Val::C(Value::Bool(!b))),
                other => Ok(Val::F(self.to_formula(other, inner.span)?.not())),
            },
            ExprKind::Neg(inner) => match self.eval(inner, env, Some(&Type::Int))? {
                Val::C(Value::Int(i)) => Ok(Val::C(Value::Int(-i))),
                _ => Err(err("E0400", "only integers can be negated", inner.span)),
            },
            ExprKind::Binary(op, l, r) => self.eval_binary(*op, l, r, env, expect, span),
            ExprKind::Is(lhs, pattern) => self.eval_is(lhs, pattern, env, span),
            ExprKind::Index(base, args) => self.eval_index(base, args, env, span),
            ExprKind::Call(base, args) => self.eval_call(base, args, env, expect, span),
            ExprKind::Field(base, field) => self.eval_field(base, field, env, span),
            ExprKind::Aggregate {
                kind,
                expr: inner,
                binders,
                guard,
            } => {
                let mut items: Vec<Val> = Vec::new();
                let expect_item = Type::Bool;
                self.each_binding(binders, guard.as_deref(), env, &mut |p, env| {
                    items.push(p.eval(inner, env, Some(&expect_item))?);
                    Ok(())
                })?;
                let mut formulas = Vec::with_capacity(items.len());
                for item in items {
                    formulas.push(self.to_formula(item, inner.span)?);
                }
                Ok(match kind {
                    Aggregator::Any => const_or_formula(F::or(formulas)),
                    Aggregator::All => const_or_formula(F::and(formulas)),
                    Aggregator::Count => {
                        if formulas.iter().all(|f| matches!(f, F::Const(_))) {
                            Val::I(IntTerm::Const(
                                formulas.iter().filter(|f| **f == F::Const(true)).count() as i64,
                            ))
                        } else {
                            Val::I(IntTerm::Count {
                                items: formulas,
                                offset: 0,
                            })
                        }
                    }
                    Aggregator::ExactlyOne => const_or_formula(F::card(Card {
                        items: formulas,
                        op: CardOp::Eq,
                        k: 1,
                        encoding: None,
                    })),
                    Aggregator::AtMostOne => const_or_formula(F::card(Card {
                        items: formulas,
                        op: CardOp::Le,
                        k: 1,
                        encoding: None,
                    })),
                })
            }
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => {
                if self.const_bool(cond, env)? {
                    self.eval(then, env, expect)
                } else {
                    self.eval(otherwise, env, expect)
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                let v = self.const_eval(scrutinee, env, None)?;
                for (pattern, value) in arms {
                    if self.const_matches(&v, pattern, env)? {
                        return self.eval(value, env, expect);
                    }
                }
                Err(err(
                    "E0420",
                    format!("no `match` arm covers `{}`", Show(&v, self)),
                    span,
                ))
            }
        }
    }

    fn eval_name(
        &mut self,
        name: &str,
        env: &mut Env,
        expect: Option<&Type>,
        span: Span,
    ) -> Result<Val> {
        if let Some((_, v)) = env.iter().rev().find(|(n, _)| same(n, name)) {
            return Ok(v.clone());
        }
        let key = (span, expect.cloned());
        if let Some(v) = self.name_cache.get(&key) {
            return Ok(v.clone());
        }
        let constant = match self.names.get(name).copied() {
            Some(Decl::Param(p)) => Some(Val::C(self.params[p as usize].value.clone())),
            Some(Decl::Enum(e)) => Some(Val::Type(Type::Enum(e))),
            Some(Decl::Subset(s)) => Some(Val::Type(Type::Subset(s))),
            Some(Decl::Domain(d)) => Some(Val::Type(Type::Domain(d))),
            Some(Decl::Grid(g)) => Some(Val::Type(Type::Grid(g))),
            Some(Decl::Choice(c)) => Some(Val::Type(Type::Choice(c))),
            Some(_) => None,
            None => Some(Val::C(self.resolve_constant(name, expect, span)?)),
        };
        if let Some(v) = constant {
            self.name_cache.insert(key, v.clone());
            return Ok(v);
        }
        match self.names.get(name).copied() {
            Some(Decl::Def(d)) if self.defs[d as usize].index.is_empty() => {
                return Ok(Val::F(self.get_def(d, &[], span)?));
            }
            Some(Decl::Fact(f)) if self.facts[f as usize].params.is_empty() => {
                return Ok(Val::C(Value::Bool(self.fact_holds(f, &[], span)?)));
            }
            _ => Err(err("E0413", format!("`{name}` needs arguments"), span)),
        }
    }

    /// The declaration a family or call name refers to.
    fn decl_at(&mut self, name: &str, span: Span) -> Option<Decl> {
        if let Some(decl) = self.decl_cache.get(&span) {
            return *decl;
        }
        let decl = self.names.get(name).copied();
        self.decl_cache.insert(span, decl);
        decl
    }

    fn member_at(&mut self, choice: u32, name: &str, span: Span) -> Option<u32> {
        if let Some(&m) = self.member_cache.get(&(span, choice)) {
            return Some(m);
        }
        let m = self.choices[choice as usize]
            .members
            .iter()
            .position(|x| x.name == name)? as u32;
        self.member_cache.insert((span, choice), m);
        Some(m)
    }

    /// Variants, choice members without payload, and domain symbols.
    fn resolve_constant(&self, name: &str, expect: Option<&Type>, span: Span) -> Result<Value> {
        let mut candidates: Vec<Value> = Vec::new();
        for &(e, v) in self.variants.get(name).into_iter().flatten() {
            candidates.push(Value::Variant(e, v));
        }
        for &(c, m) in self.members.get(name).into_iter().flatten() {
            if self.choices[c as usize].members[m as usize]
                .params
                .is_empty()
            {
                candidates.push(Value::Member(c, m, Arc::from(Vec::new())));
            }
        }
        for &(d, i) in self.symbols.get(name).into_iter().flatten() {
            candidates.push(Value::Sym(d, i));
        }
        if candidates.is_empty() {
            return Err(err("E0210", format!("`{name}` is not defined"), span));
        }
        if let Some(expect) = expect {
            let fitting: Vec<&Value> = candidates
                .iter()
                .filter(|v| self.conforms(v, expect))
                .collect();
            if fitting.len() == 1 {
                return Ok(fitting[0].clone());
            }
            // A subset value is spelled with its base enum's variant.
            if fitting.is_empty() {
                if let Type::Subset(s) = expect.unwrap_option() {
                    let base = self.subsets[*s as usize].base;
                    if let Some(v) = candidates
                        .iter()
                        .find(|v| matches!(v, Value::Variant(e, _) if *e == base))
                    {
                        return Ok(v.clone());
                    }
                }
            }
        }
        if candidates.len() == 1 {
            return Ok(candidates.pop().unwrap());
        }
        let options = candidates
            .iter()
            .map(|v| self.qualified(v))
            .collect::<Vec<_>>()
            .join("`, `");
        Err(err("E0211", format!("`{name}` is ambiguous here"), span)
            .with_note(format!("candidates: `{options}`"))
            .with_help("qualify it, for example `Dir6.East`"))
    }

    fn qualified(&self, v: &Value) -> String {
        match v {
            Value::Variant(e, x) => format!(
                "{}.{}",
                self.enums[*e as usize].name, self.enums[*e as usize].variants[*x as usize]
            ),
            Value::Member(c, m, _) => format!(
                "{}.{}",
                self.choices[*c as usize].name, self.choices[*c as usize].members[*m as usize].name
            ),
            Value::Sym(d, i) => format!(
                "{}.{}",
                self.domains[*d as usize].name, self.domains[*d as usize].symbols[*i as usize]
            ),
            other => format!("{}", Show(other, self)),
        }
    }

    fn type_of(&self, v: &Value) -> Option<Type> {
        Some(match v {
            Value::Bool(_) => Type::Bool,
            Value::Int(_) => Type::Int,
            Value::Sym(d, _) => Type::Domain(*d),
            Value::Variant(e, _) => Type::Enum(*e),
            Value::Cell(g, _) => Type::Grid(*g),
            Value::Member(c, _, _) => Type::Choice(*c),
            _ => return None,
        })
    }

    fn eval_binary(
        &mut self,
        op: BinOp,
        l: &Expr,
        r: &Expr,
        env: &mut Env,
        _expect: Option<&Type>,
        span: Span,
    ) -> Result<Val> {
        match op {
            BinOp::And | BinOp::Or | BinOp::Imp | BinOp::Iff | BinOp::Xor => {
                let lv = self.eval(l, env, Some(&Type::Bool))?;
                // Short-circuit on known left operands, so guards like
                // `has(x) and f(x)` never evaluate the right side needlessly.
                if let Val::C(Value::Bool(b)) = lv {
                    match (op, b) {
                        (BinOp::And, false) => return Ok(Val::C(Value::Bool(false))),
                        (BinOp::Or, true) | (BinOp::Imp, false) => {
                            return Ok(Val::C(Value::Bool(true)))
                        }
                        _ => {}
                    }
                }
                let rv = self.eval(r, env, Some(&Type::Bool))?;
                if let (Val::C(Value::Bool(a)), Val::C(Value::Bool(b))) = (&lv, &rv) {
                    let (a, b) = (*a, *b);
                    return Ok(Val::C(Value::Bool(match op {
                        BinOp::And => a && b,
                        BinOp::Or => a || b,
                        BinOp::Imp => !a || b,
                        BinOp::Iff => a == b,
                        _ => a != b,
                    })));
                }
                let a = self.to_formula(lv, l.span)?;
                let b = self.to_formula(rv, r.span)?;
                Ok(const_or_formula(match op {
                    BinOp::And => F::and(vec![a, b]),
                    BinOp::Or => F::or(vec![a, b]),
                    BinOp::Imp => F::imp(a, b),
                    BinOp::Iff => F::iff(a, b),
                    _ => F::xor(a, b),
                }))
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                let lv = self.eval(l, env, Some(&Type::Int))?;
                let rv = self.eval(r, env, Some(&Type::Int))?;
                match (lv, rv) {
                    (Val::C(Value::Int(a)), Val::C(Value::Int(b))) => {
                        Ok(Val::C(Value::Int(match op {
                            BinOp::Add => a + b,
                            BinOp::Sub => a - b,
                            BinOp::Mul => a * b,
                            BinOp::Div | BinOp::Mod if b == 0 => {
                                return Err(err("E0430", "division by zero", span))
                            }
                            BinOp::Div => a.div_euclid(b),
                            _ => a.rem_euclid(b),
                        })))
                    }
                    (Val::I(t), Val::C(Value::Int(k))) if matches!(op, BinOp::Add | BinOp::Sub) => {
                        Ok(Val::I(shift(t, if op == BinOp::Add { k } else { -k })))
                    }
                    (Val::C(Value::Int(k)), Val::I(t)) if op == BinOp::Add => {
                        Ok(Val::I(shift(t, k)))
                    }
                    (Val::C(Value::Int(k)), Val::I(t)) if op == BinOp::Sub => {
                        Ok(Val::I(add(IntTerm::Const(k), t, -1)))
                    }
                    (Val::I(a), Val::I(b)) if matches!(op, BinOp::Add | BinOp::Sub) => {
                        Ok(Val::I(add(a, b, if op == BinOp::Add { 1 } else { -1 })))
                    }
                    (Val::I(t), Val::C(Value::Int(k))) | (Val::C(Value::Int(k)), Val::I(t))
                        if op == BinOp::Mul =>
                    {
                        Ok(Val::I(scale(t, k)))
                    }
                    _ => Err(err(
                        "E0441",
                        "solver integers support +, -, and multiplication by a constant",
                        span,
                    )),
                }
            }
            BinOp::In => {
                let lv = self.const_eval(l, env, None)?;
                let ty = self.type_of(&lv);
                match self.eval(r, env, ty.map(|t| Type::Set(Box::new(t))).as_ref())? {
                    Val::C(Value::Set(items)) => Ok(Val::C(Value::Bool(items.contains(&lv)))),
                    Val::Type(t) => Ok(Val::C(Value::Bool(self.conforms(&lv, &t)))),
                    _ => Err(err("E0411", "`in` expects a set or a type", r.span)),
                }
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                let lv = self.eval(l, env, None)?;
                let expect_r = match &lv {
                    Val::C(v) => self.type_of(v),
                    Val::I(_) => Some(Type::Int),
                    _ => None,
                };
                let rv = self.eval(r, env, expect_r.as_ref())?;
                // Re-resolve the left side with the right side's type, for
                // `East == d` where the left name alone is ambiguous.
                self.compare(op, lv, rv, span)
            }
        }
    }

    fn compare(&mut self, op: BinOp, lv: Val, rv: Val, span: Span) -> Result<Val> {
        match (lv, rv) {
            (Val::C(a), Val::C(b)) => {
                if a.is_none() || b.is_none() {
                    if matches!(op, BinOp::Eq | BinOp::Ne) {
                        return Err(err("E0421", "`== none` is not allowed", span)
                            .with_help("use `has(x)`"));
                    }
                    return Err(err("E0412", "this optional value is `none` here", span)
                        .with_help("guard the use with `has(...)`"));
                }
                let result = match op {
                    BinOp::Eq => a == b,
                    BinOp::Ne => a != b,
                    _ => {
                        let (Value::Int(x), Value::Int(y)) = (&a, &b) else {
                            return Err(err("E0400", "ordering comparisons need integers", span));
                        };
                        match op {
                            BinOp::Lt => x < y,
                            BinOp::Le => x <= y,
                            BinOp::Gt => x > y,
                            _ => x >= y,
                        }
                    }
                };
                Ok(Val::C(Value::Bool(result)))
            }
            (Val::I(a), Val::I(b)) => Ok(const_or_formula(self.int_cmp(op, a, b, span)?)),
            (Val::I(a), Val::C(Value::Int(k))) => Ok(const_or_formula(self.int_cmp(
                op,
                a,
                IntTerm::Const(k),
                span,
            )?)),
            (Val::C(Value::Int(k)), Val::I(b)) => Ok(const_or_formula(self.int_cmp(
                op,
                IntTerm::Const(k),
                b,
                span,
            )?)),
            (Val::I(_), Val::C(Value::None)) | (Val::C(Value::None), Val::I(_)) => {
                Err(err("E0412", "this optional value is `none` here", span)
                    .with_help("guard the use with `has(...)`"))
            }
            _ => Err(err("E0400", "these operands cannot be compared", span)
                .with_help("formulas are compared with `<->`")),
        }
    }

    /// Order-encoded comparisons; `a op b`.
    fn int_cmp(&mut self, op: BinOp, a: IntTerm, b: IntTerm, span: Span) -> Result<F> {
        use IntTerm::*;
        match op {
            BinOp::Gt => return self.int_cmp(BinOp::Lt, b, a, span),
            BinOp::Ge => return self.int_cmp(BinOp::Le, b, a, span),
            BinOp::Ne => return Ok(self.int_cmp(BinOp::Eq, a, b, span)?.not()),
            BinOp::Lt => return self.int_cmp(BinOp::Le, shift(a, 1), b, span),
            _ => {}
        }
        match (a, b) {
            (Const(x), Const(y)) => Ok(F::Const(if op == BinOp::Eq { x == y } else { x <= y })),
            (Count { items, offset }, Const(k)) => Ok(F::card(Card {
                items,
                op: if op == BinOp::Eq {
                    CardOp::Eq
                } else {
                    CardOp::Le
                },
                k: k - offset,
                encoding: None,
            })),
            (Const(k), Count { items, offset }) => {
                if op == BinOp::Eq {
                    return Ok(F::card(Card {
                        items,
                        op: CardOp::Eq,
                        k: k - offset,
                        encoding: None,
                    }));
                }
                Ok(F::card(Card {
                    items,
                    op: CardOp::Ge,
                    k: k - offset,
                    encoding: None,
                }))
            }
            (Var { lits, lo, offset }, Const(k)) => {
                // lo + n + offset <= k  <=>  not ge(k - offset + 1)
                let ge = |v: i64| ge_lit(&lits, lo, v);
                let le = F::Lit(-ge(k - offset + 1)).simplify_const();
                if op == BinOp::Eq {
                    Ok(F::and(vec![F::Lit(ge(k - offset)).simplify_const(), le]))
                } else {
                    Ok(le)
                }
            }
            (Const(k), Var { lits, lo, offset }) => {
                let ge = |v: i64| ge_lit(&lits, lo, v);
                let ge_k = F::Lit(ge(k - offset)).simplify_const();
                if op == BinOp::Eq {
                    Ok(F::and(vec![
                        ge_k,
                        F::Lit(-ge(k - offset + 1)).simplify_const(),
                    ]))
                } else {
                    Ok(ge_k)
                }
            }
            (
                Var {
                    lits: la,
                    lo: lo_a,
                    offset: oa,
                },
                Var {
                    lits: lb,
                    lo: lo_b,
                    offset: ob,
                },
            ) => {
                // a + oa <= b + ob  <=>  for all v: a >= v -> b >= v + oa - ob
                let d = oa - ob;
                let hi_a = lo_a + la.len() as i64;
                let mut parts = Vec::new();
                for v in lo_a..=hi_a {
                    let premise = ge_lit(&la, lo_a, v);
                    let conclusion = ge_lit(&lb, lo_b, v + d);
                    parts.push(F::or(vec![
                        F::Lit(-premise).simplify_const(),
                        F::Lit(conclusion).simplify_const(),
                    ]));
                }
                let le = F::and(parts);
                if op == BinOp::Eq {
                    let back = self.int_cmp(
                        BinOp::Le,
                        Var {
                            lits: lb,
                            lo: lo_b,
                            offset: ob,
                        },
                        Var {
                            lits: la,
                            lo: lo_a,
                            offset: oa,
                        },
                        span,
                    )?;
                    Ok(F::and(vec![le, back]))
                } else {
                    Ok(le)
                }
            }
            _ => Err(err(
                "E0441",
                "this integer comparison is not supported",
                span,
            )
            .with_help("counts and order-encoded integers compare with each other and with constants; weighted sums can only be minimized or maximized")),
        }
    }

    fn eval_is(
        &mut self,
        lhs: &Expr,
        pattern: &[PatAlt],
        env: &mut Env,
        span: Span,
    ) -> Result<Val> {
        match self.eval(lhs, env, None)? {
            Val::Choice(c, k) => {
                let Some(k) = k else {
                    let Some(outside) = self.choices[c as usize].outside else {
                        return Err(err(
                            "E0421",
                            format!(
                                "`{}[...]` may be outside its domain",
                                self.choices[c as usize].name
                            ),
                            span,
                        )
                        .with_help(format!(
                            "declare `@outside(Member)` on `choice {}`, or guard with `has(...)`",
                            self.choices[c as usize].name
                        )));
                    };
                    let v = Value::Member(c, outside, Arc::from(Vec::new()));
                    return Ok(Val::C(Value::Bool(self.const_matches(&v, pattern, env)?)));
                };
                let alts = self.resolve_alts(c, pattern, env)?;
                let encoder = &self.encoder;
                let lits = self.choices[c as usize].options[k as usize]
                    .iter()
                    .filter(|o| alts.iter().any(|alt| alt.matches(c, o.member, &o.payload)))
                    .map(|o| encoder.lit_f(o.lit))
                    .collect();
                Ok(const_or_formula(F::or(lits)))
            }
            Val::C(v) => Ok(Val::C(Value::Bool(self.const_matches(&v, pattern, env)?))),
            _ => Err(err("E0400", "`is` tests a choice or a value", lhs.span)),
        }
    }

    /// Resolves a pattern against the members of choice `c` once, so that
    /// matching each option is a comparison.
    fn resolve_alts(&mut self, c: u32, pattern: &[PatAlt], env: &mut Env) -> Result<Vec<Alt>> {
        let mut alts = Vec::with_capacity(pattern.len());
        for alt in pattern {
            alts.push(match alt {
                PatAlt::Wild => Alt::Any,
                PatAlt::Name(name, span) => {
                    if let Some((_, Val::C(bound))) = env.iter().rev().find(|(n, _)| same(n, name))
                    {
                        Alt::Value(bound.clone())
                    } else {
                        let Some(m) = self.member_at(c, name, *span) else {
                            return Err(err(
                                "E0451",
                                format!(
                                    "`{name}` is not a member of `{}`",
                                    self.choices[c as usize].name
                                ),
                                *span,
                            ));
                        };
                        if !self.choices[c as usize].members[m as usize]
                            .params
                            .is_empty()
                        {
                            return Err(err(
                                "E0450",
                                format!("`{name}` carries a payload; write `{name}(_)`"),
                                *span,
                            ));
                        }
                        Alt::Member(m, Vec::new())
                    }
                }
                PatAlt::Ctor(name, args, span) => {
                    let Some(m) = self.member_at(c, name, *span) else {
                        return Err(err(
                            "E0451",
                            format!(
                                "`{name}` is not a member of `{}`",
                                self.choices[c as usize].name
                            ),
                            *span,
                        ));
                    };
                    let arity = self.choices[c as usize].members[m as usize].params.len();
                    if arity != args.len() {
                        return Err(err(
                            "E0452",
                            format!("`{name}` takes {arity} payload values"),
                            *span,
                        ));
                    }
                    let mut values = Vec::with_capacity(args.len());
                    for (index, arg) in args.iter().enumerate() {
                        values.push(match arg {
                            PatArg::Wild => None,
                            PatArg::Expr(e) => {
                                let ty = self.choices[c as usize].members[m as usize].params[index]
                                    .1
                                    .clone();
                                Some(self.const_eval(e, env, Some(&ty))?)
                            }
                        });
                    }
                    Alt::Member(m, values)
                }
            });
        }
        Ok(alts)
    }

    fn const_matches(&mut self, v: &Value, pattern: &[PatAlt], env: &mut Env) -> Result<bool> {
        for alt in pattern {
            let hit = match alt {
                PatAlt::Wild => true,
                PatAlt::Name(name, span) => {
                    if let Some((_, Val::C(bound))) = env.iter().rev().find(|(n, _)| same(n, name))
                    {
                        bound == v
                    } else {
                        match v {
                            Value::Member(c, m, payload) => {
                                let member = &self.choices[*c as usize].members[*m as usize];
                                if member.name == *name {
                                    if !payload.is_empty() {
                                        return Err(err(
                                            "E0450",
                                            format!(
                                                "`{name}` carries a payload; write `{name}(_)`"
                                            ),
                                            *span,
                                        ));
                                    }
                                    true
                                } else {
                                    if !self.choices[*c as usize]
                                        .members
                                        .iter()
                                        .any(|x| x.name == *name)
                                    {
                                        return Err(err(
                                            "E0451",
                                            format!(
                                                "`{name}` is not a member of `{}`",
                                                self.choices[*c as usize].name
                                            ),
                                            *span,
                                        ));
                                    }
                                    false
                                }
                            }
                            _ => {
                                let ty = self.type_of(v);
                                // Patterns resolve the same way every time;
                                // share the name cache with `eval_name`.
                                let key = (*span, ty);
                                let resolved = match self.name_cache.get(&key) {
                                    Some(Val::C(value)) => value.clone(),
                                    _ => {
                                        let value =
                                            self.resolve_constant(name, key.1.as_ref(), *span)?;
                                        self.name_cache.insert(key, Val::C(value.clone()));
                                        value
                                    }
                                };
                                &resolved == v
                            }
                        }
                    }
                }
                PatAlt::Ctor(name, args, span) => match v {
                    Value::Member(c, m, payload) => {
                        let members = &self.choices[*c as usize].members;
                        let Some(target) = members.iter().position(|x| x.name == *name) else {
                            return Err(err(
                                "E0451",
                                format!(
                                    "`{name}` is not a member of `{}`",
                                    self.choices[*c as usize].name
                                ),
                                *span,
                            ));
                        };
                        if members[target].params.len() != args.len() {
                            return Err(err(
                                "E0452",
                                format!(
                                    "`{name}` takes {} payload values",
                                    members[target].params.len()
                                ),
                                *span,
                            ));
                        }
                        if target as u32 != *m {
                            false
                        } else {
                            let types: Vec<Type> = members[target]
                                .params
                                .iter()
                                .map(|(_, t)| t.clone())
                                .collect();
                            let mut all = true;
                            for ((arg, ty), actual) in args.iter().zip(&types).zip(payload.iter()) {
                                if let PatArg::Expr(e) = arg {
                                    let expected = self.const_eval(e, env, Some(ty))?;
                                    if expected != *actual {
                                        all = false;
                                        break;
                                    }
                                }
                            }
                            all
                        }
                    }
                    _ => {
                        return Err(err(
                            "E0452",
                            "constructor patterns apply to choice members",
                            *span,
                        ))
                    }
                },
            };
            if hit {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn eval_index(&mut self, base: &Expr, args: &[Expr], env: &mut Env, span: Span) -> Result<Val> {
        let ExprKind::Name(name) = &base.kind else {
            return Err(err("E0414", "only families can be indexed", base.span));
        };
        let decl = self.decl_at(name, base.span);
        let dims: Arc<[Dim]> = match decl {
            Some(Decl::Choice(c)) => self.choices[c as usize].index.clone(),
            Some(Decl::Var(v)) => match &self.vars[v as usize].index {
                VarIndexKind::Dims(d) => d.clone(),
                VarIndexKind::Over(_) => {
                    return Err(err(
                        "E0415",
                        format!("`{name}` is indexed by relation tuples; write `{name}(...)`"),
                        span,
                    ))
                }
            },
            Some(Decl::Def(d)) => self.defs[d as usize].index.clone(),
            Some(Decl::Int(i)) => self.ints[i as usize].index.clone(),
            _ => {
                return Err(err(
                    "E0414",
                    format!("`{name}` is not an indexed family"),
                    base.span,
                ))
            }
        };
        if dims.len() != args.len() {
            return Err(err(
                "E0410",
                format!("`{name}` takes {} indices", dims.len()),
                span,
            ));
        }
        // Reuse one buffer for the key; lookups that hit allocate nothing.
        let mut key = std::mem::take(&mut self.key_scratch);
        key.clear();
        let result = self.index_with(decl.unwrap(), &dims, args, env, span, &mut key, name);
        self.key_scratch = key;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn index_with(
        &mut self,
        decl: Decl,
        dims: &[Dim],
        args: &[Expr],
        env: &mut Env,
        span: Span,
        key: &mut Vec<Value>,
        name: &str,
    ) -> Result<Val> {
        for (arg, dim) in args.iter().zip(dims.iter()) {
            let v = self.const_eval(arg, env, Some(&dim.ty))?;
            if !v.is_none() && !self.conforms(&v, &dim.ty) {
                return Err(err(
                    "E0400",
                    format!("index expects {}", ShowType(&dim.ty, self)),
                    arg.span,
                ));
            }
            key.push(v);
        }
        let outside = key.iter().any(Value::is_none);
        match decl {
            Decl::Choice(c) => {
                if outside {
                    return Ok(Val::Choice(c, None));
                }
                let Some(&k) = self.choices[c as usize].lookup.get(key.as_slice()) else {
                    return Err(err("E0400", "index is outside the choice's domain", span));
                };
                Ok(Val::Choice(c, Some(k)))
            }
            Decl::Var(v) => {
                if outside {
                    return Ok(Val::C(Value::Bool(self.vars[v as usize].outside)));
                }
                Ok(const_or_formula(
                    self.encoder.lit_f(self.vars[v as usize].lookup[key.as_slice()]),
                ))
            }
            Decl::Def(d) => {
                if outside {
                    return Ok(Val::C(Value::Bool(self.defs[d as usize].outside)));
                }
                Ok(const_or_formula(self.get_def(d, key, span)?))
            }
            Decl::Int(i) => {
                self.ground_int(i)?;
                let def = &self.ints[i as usize];
                if outside {
                    return match def.outside {
                        Some(v) => Ok(Val::I(IntTerm::Const(v))),
                        None => Err(err(
                            "E0421",
                            format!("`{name}[...]` may be outside its domain"),
                            span,
                        )
                        .with_help(format!(
                            "declare `@outside(v)` on `int {name}`, or guard with `has(...)`"
                        ))),
                    };
                }
                Ok(Val::I(IntTerm::Var {
                    lits: def.lookup[key.as_slice()].clone(),
                    lo: def.lo,
                    offset: 0,
                }))
            }
            _ => unreachable!(),
        }
    }

    fn eval_call(
        &mut self,
        base: &Expr,
        args: &[Expr],
        env: &mut Env,
        expect: Option<&Type>,
        span: Span,
    ) -> Result<Val> {
        let ExprKind::Name(name) = &base.kind else {
            return Err(err("E0416", "only names can be called", base.span));
        };
        match name.as_str() {
            "has" => {
                if args.len() != 1 {
                    return Err(err("E0410", "`has` takes one argument", span));
                }
                let v = self.const_eval(&args[0], env, None)?;
                return Ok(Val::C(Value::Bool(!v.is_none())));
            }
            "step" if !self.names.contains_key("step") => {
                if args.len() != 2 {
                    return Err(err("E0410", "`step` takes a cell and a direction", span));
                }
                let cell = self.const_eval(&args[0], env, None)?;
                let Value::Cell(g, index) = cell else {
                    if cell.is_none() {
                        return Ok(Val::C(Value::None));
                    }
                    return Err(err("E0400", "`step` expects a cell", args[0].span));
                };
                let dirs = self.grids[g as usize].dirs;
                let d = self.const_eval(&args[1], env, Some(&Type::Enum(dirs)))?;
                let Value::Variant(e, v) = d else {
                    return Err(err("E0400", "`step` expects a direction", args[1].span));
                };
                if e != dirs {
                    return Err(err(
                        "E0400",
                        format!("`step` expects a `{}`", self.enums[dirs as usize].name),
                        args[1].span,
                    ));
                }
                let grid = &self.grids[g as usize];
                let (axis, sign) = grid.steps[v as usize];
                let mut c = grid.coords(index);
                c[axis] += sign as i64;
                return Ok(Val::C(match grid.index(c) {
                    Some(i) => Value::Cell(g, i),
                    None => Value::None,
                }));
            }
            "opposite" if !self.names.contains_key("opposite") => {
                if args.len() != 1 {
                    return Err(err("E0410", "`opposite` takes a direction", span));
                }
                let d = self.const_eval(&args[0], env, None)?;
                let Value::Variant(e, v) = d else {
                    return Err(err("E0400", "`opposite` expects a direction", args[0].span));
                };
                let Some(grid) = self.grids.iter().find(|g| g.dirs == e) else {
                    return Err(err(
                        "E0400",
                        "`opposite` expects a grid direction",
                        args[0].span,
                    ));
                };
                let (axis, sign) = grid.steps[v as usize];
                let Some(o) = grid
                    .steps
                    .iter()
                    .position(|&(a, s)| a == axis && s == -sign)
                else {
                    return Err(err("E0400", "this direction has no opposite", args[0].span));
                };
                return Ok(Val::C(Value::Variant(e, o as u32)));
            }
            _ => {}
        }
        match self.decl_at(name, base.span) {
            Some(Decl::Fn(f)) => {
                let def = self.fns[f as usize].clone();
                if def.params.len() != args.len() {
                    return Err(err(
                        "E0410",
                        format!("`{name}` takes {} arguments", def.params.len()),
                        span,
                    ));
                }
                let mut local: Env = Vec::with_capacity(args.len());
                for (arg, (pname, ty)) in args.iter().zip(&def.params) {
                    let v = self.const_eval(arg, env, Some(ty))?;
                    if !self.conforms(&v, ty) {
                        return Err(err(
                            "E0400",
                            format!("`{name}` expects {}", ShowType(ty, self)),
                            arg.span,
                        ));
                    }
                    local.push((pname.clone(), Val::C(v)));
                }
                let v = self.const_eval(&def.body, &mut local, Some(&def.ret))?;
                Ok(Val::C(v))
            }
            Some(Decl::Fact(fact)) => {
                let params = self.facts[fact as usize].params.clone();
                if params.len() != args.len() {
                    return Err(err(
                        "E0410",
                        format!("`{name}` takes {} arguments", params.len()),
                        span,
                    ));
                }
                // Reuse one buffer for the arguments; memo hits allocate nothing.
                let mut values = std::mem::take(&mut self.key_scratch);
                values.clear();
                for (arg, (_, ty)) in args.iter().zip(params.iter()) {
                    match self.const_eval(arg, env, Some(ty)) {
                        Ok(v) => values.push(v),
                        Err(e) => {
                            self.key_scratch = values;
                            return Err(e);
                        }
                    }
                }
                let holds = self.fact_holds(fact, &values, span);
                self.key_scratch = values;
                Ok(Val::C(Value::Bool(holds?)))
            }
            Some(Decl::Relation(rel)) => {
                if !self.relations[rel as usize].frozen {
                    return Err(err("E0503", format!("relation `{name}` is used before all contributions are collected"), span)
                        .with_help("relation atoms can only be used in constraints, not in contributions or definitions they feed"));
                }
                let key = self.relation_key(rel, args, env, span)?;
                let Some(key) = key else {
                    return Ok(Val::C(Value::Bool(false)));
                };
                let def = &self.relations[rel as usize];
                Ok(match def.lookup.get(&key) {
                    Some(&i) => const_or_formula(F::Lit(def.tuples[i as usize].1).simplify_const()),
                    None => Val::C(Value::Bool(false)),
                })
            }
            Some(Decl::Var(v)) => {
                let VarIndexKind::Over(rel) = self.vars[v as usize].index else {
                    return Err(err(
                        "E0415",
                        format!("`{name}` is indexed with brackets: `{name}[...]`"),
                        span,
                    ));
                };
                let key = self.relation_key(rel, args, env, span)?;
                let Some(key) = key else {
                    return Ok(Val::C(Value::Bool(false)));
                };
                Ok(match self.vars[v as usize].lookup.get(&key) {
                    Some(&lit) => Val::F(F::Lit(lit)),
                    None => Val::C(Value::Bool(false)),
                })
            }
            Some(Decl::Grid(g)) => {
                if args.len() != 3 {
                    return Err(err("E0410", "a cell literal takes three coordinates", span));
                }
                let mut c = [0i64; 3];
                for (i, arg) in args.iter().enumerate() {
                    c[i] = self.const_int_env(arg, env)?;
                }
                match self.grids[g as usize].index(c) {
                    Some(i) => Ok(Val::C(Value::Cell(g, i))),
                    None => Err(err(
                        "E0402",
                        format!("cell ({},{},{}) is outside the grid", c[0], c[1], c[2]),
                        span,
                    )),
                }
            }
            Some(_) => Err(err(
                "E0416",
                format!("`{name}` cannot be called"),
                base.span,
            )),
            None => {
                // Choice member constructor: `Torch(East)`.
                let candidates: Vec<(u32, u32)> =
                    self.members.get(name).cloned().unwrap_or_default();
                let candidate = match (candidates.len(), expect.map(Type::unwrap_option)) {
                    (0, _) => {
                        return Err(err("E0210", format!("`{name}` is not defined"), base.span))
                    }
                    (1, _) => candidates[0],
                    (_, Some(Type::Choice(c))) => *candidates
                        .iter()
                        .find(|(x, _)| x == c)
                        .ok_or_else(|| err("E0211", format!("`{name}` is ambiguous"), span))?,
                    _ => return Err(err("E0211", format!("`{name}` is ambiguous"), span)),
                };
                let (c, m) = candidate;
                let params = self.choices[c as usize].members[m as usize].params.clone();
                if params.len() != args.len() {
                    return Err(err(
                        "E0452",
                        format!("`{name}` takes {} payload values", params.len()),
                        span,
                    ));
                }
                let mut payload = Vec::new();
                for (arg, (_, ty)) in args.iter().zip(params.iter()) {
                    payload.push(self.const_eval(arg, env, Some(ty))?);
                }
                Ok(Val::C(Value::Member(c, m, payload.into())))
            }
        }
    }

    fn relation_key(
        &mut self,
        rel: u32,
        args: &[Expr],
        env: &mut Env,
        span: Span,
    ) -> Result<Option<Vec<Value>>> {
        let params = self.relations[rel as usize].params.clone();
        if params.len() != args.len() {
            return Err(err(
                "E0410",
                format!(
                    "`{}` takes {} fields",
                    self.relations[rel as usize].name,
                    params.len()
                ),
                span,
            ));
        }
        let mut key = Vec::new();
        for (arg, (_, ty)) in args.iter().zip(params.iter()) {
            let v = self.const_eval(arg, env, Some(ty))?;
            if v.is_none() {
                return Ok(None);
            }
            key.push(v);
        }
        Ok(Some(key))
    }

    fn eval_field(&mut self, base: &Expr, field: &str, env: &mut Env, span: Span) -> Result<Val> {
        if let ExprKind::Name(name) = &base.kind {
            if !env.iter().any(|(n, _)| same(n, name)) {
                match self.names.get(name).copied() {
                    Some(Decl::Enum(e)) => {
                        let Some(v) = self.enums[e as usize]
                            .variants
                            .iter()
                            .position(|x| x == field)
                        else {
                            return Err(err(
                                "E0204",
                                format!("`{field}` is not a variant of `{name}`"),
                                span,
                            ));
                        };
                        return Ok(Val::C(Value::Variant(e, v as u32)));
                    }
                    Some(Decl::Subset(s)) => {
                        let base_enum = self.subsets[s as usize].base;
                        let Some(v) = self.enums[base_enum as usize]
                            .variants
                            .iter()
                            .position(|x| x == field)
                        else {
                            return Err(err(
                                "E0204",
                                format!("`{field}` is not a variant of `{name}`"),
                                span,
                            ));
                        };
                        return Ok(Val::C(Value::Variant(base_enum, v as u32)));
                    }
                    Some(Decl::Choice(c)) => {
                        let Some(m) = self.choices[c as usize]
                            .members
                            .iter()
                            .position(|x| x.name == field)
                        else {
                            return Err(err(
                                "E0451",
                                format!("`{field}` is not a member of `{name}`"),
                                span,
                            ));
                        };
                        return Ok(Val::C(Value::Member(c, m as u32, Arc::from(Vec::new()))));
                    }
                    Some(Decl::Domain(d)) => {
                        let Some(i) = self.domains[d as usize]
                            .symbols
                            .iter()
                            .position(|x| x == field)
                        else {
                            return Err(err(
                                "E0401",
                                format!("`{field}` is not in domain `{name}`"),
                                span,
                            ));
                        };
                        return Ok(Val::C(Value::Sym(d, i as u32)));
                    }
                    _ => {}
                }
            }
        }
        match self.const_eval(base, env, None)? {
            Value::Cell(g, index) => {
                let grid = &self.grids[g as usize];
                let Some(axis) = grid.axes.iter().position(|a| a == field) else {
                    return Err(err(
                        "E0417",
                        format!("cells have fields {:?}", grid.axes),
                        span,
                    ));
                };
                Ok(Val::C(Value::Int(grid.coords(index)[axis])))
            }
            _ => Err(err("E0417", format!("no field `{field}` here"), span)),
        }
    }
}

// ----- helpers -------------------------------------------------------------------

fn err_nospan(message: String) -> Diagnostic {
    Diagnostic::error_nospan("E0400", message)
}

trait WithSpan {
    fn with_span(self, span: Span) -> Self;
}

impl WithSpan for Diagnostic {
    fn with_span(mut self, span: Span) -> Self {
        if self.span.is_none() {
            self.span = Some(span);
        }
        self
    }
}

fn const_or_formula(f: F) -> Val {
    match f {
        F::Const(b) => Val::C(Value::Bool(b)),
        other => Val::F(other),
    }
}

fn shift(t: IntTerm, k: i64) -> IntTerm {
    match t {
        IntTerm::Const(c) => IntTerm::Const(c + k),
        IntTerm::Var { lits, lo, offset } => IntTerm::Var {
            lits,
            lo,
            offset: offset + k,
        },
        IntTerm::Count { items, offset } => IntTerm::Count {
            items,
            offset: offset + k,
        },
        IntTerm::Linear { terms, constant } => IntTerm::Linear {
            terms,
            constant: constant + k,
        },
    }
}

/// An integer term as `(Σ weight · [formula], constant)`.
fn into_linear(t: IntTerm) -> (Vec<(i64, F)>, i64) {
    match t {
        IntTerm::Const(c) => (Vec::new(), c),
        IntTerm::Var { lits, lo, offset } => {
            (lits.iter().map(|&l| (1, F::Lit(l))).collect(), lo + offset)
        }
        IntTerm::Count { items, offset } => (items.into_iter().map(|f| (1, f)).collect(), offset),
        IntTerm::Linear { terms, constant } => (terms, constant),
    }
}

/// The simplest term for a weighted sum: unit weights stay a count, so they
/// can still be compared with cardinality encodings.
fn from_linear(terms: Vec<(i64, F)>, constant: i64) -> IntTerm {
    let terms = terms
        .into_iter()
        .filter(|(w, f)| *w != 0 && *f != F::Const(false))
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return IntTerm::Const(constant);
    }
    if terms.iter().all(|(w, _)| *w == 1) {
        return IntTerm::Count {
            items: terms.into_iter().map(|(_, f)| f).collect(),
            offset: constant,
        };
    }
    IntTerm::Linear { terms, constant }
}

fn scale(t: IntTerm, k: i64) -> IntTerm {
    let (terms, constant) = into_linear(t);
    from_linear(
        terms.into_iter().map(|(w, f)| (w * k, f)).collect(),
        constant * k,
    )
}

fn add(a: IntTerm, b: IntTerm, sign: i64) -> IntTerm {
    let (mut terms, constant) = into_linear(a);
    let (other, other_constant) = into_linear(b);
    terms.extend(other.into_iter().map(|(w, f)| (w * sign, f)));
    from_linear(terms, constant + sign * other_constant)
}

/// Literal for `x >= v` of an order-encoded integer with base `lo`.
fn ge_lit(lits: &[Lit], lo: i64, v: i64) -> Lit {
    if v <= lo {
        1
    } else if v > lo + lits.len() as i64 {
        -1
    } else {
        lits[(v - lo - 1) as usize]
    }
}

impl F {
    fn simplify_const(self) -> F {
        match self {
            F::Lit(1) => F::Const(true),
            F::Lit(-1) => F::Const(false),
            other => other,
        }
    }
}

fn set_card_encoding(f: &mut F, encoding: &str) {
    match f {
        F::Card(card) => card.encoding = Some(encoding.to_owned()),
        F::And(items) | F::Or(items) => items
            .iter_mut()
            .for_each(|i| set_card_encoding(i, encoding)),
        _ => {}
    }
}

fn contains(stmts: &[Stmt], pred: fn(&Stmt) -> bool) -> bool {
    stmts.iter().any(|s| {
        pred(s)
            || match &s.kind {
                StmtKind::Forall { body, .. } => contains(body, pred),
                StmtKind::If {
                    then, otherwise, ..
                } => contains(then, pred) || contains(otherwise, pred),
                _ => false,
            }
    })
}

fn flatten_and<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let ExprKind::Binary(BinOp::And, l, r) = &e.kind {
        flatten_and(l, out);
        flatten_and(r, out);
    } else {
        out.push(e);
    }
}

fn mentions_any(e: &Expr, names: &[Option<Name>]) -> bool {
    let mut found = false;
    visit_names(e, &mut |n| {
        if names.iter().any(|x| x.as_deref() == Some(n)) {
            found = true;
        }
    });
    found
}

fn visit_names(e: &Expr, f: &mut dyn FnMut(&str)) {
    match &e.kind {
        ExprKind::Name(n) => f(n),
        ExprKind::Tuple(items) | ExprKind::List(items) => {
            items.iter().for_each(|i| visit_names(i, f))
        }
        ExprKind::Not(x) | ExprKind::Neg(x) => visit_names(x, f),
        ExprKind::Binary(_, l, r) => {
            visit_names(l, f);
            visit_names(r, f);
        }
        ExprKind::Is(x, _) | ExprKind::Field(x, _) => visit_names(x, f),
        ExprKind::Index(b, args) | ExprKind::Call(b, args) => {
            visit_names(b, f);
            args.iter().for_each(|a| visit_names(a, f));
        }
        ExprKind::If {
            cond,
            then,
            otherwise,
        } => {
            visit_names(cond, f);
            visit_names(then, f);
            visit_names(otherwise, f);
        }
        ExprKind::Match { scrutinee, arms } => {
            visit_names(scrutinee, f);
            arms.iter().for_each(|(_, v)| visit_names(v, f));
        }
        ExprKind::Aggregate { expr, guard, .. } | ExprKind::ListComp { expr, guard, .. } => {
            visit_names(expr, f);
            if let Some(g) = guard {
                visit_names(g, f);
            }
        }
        _ => {}
    }
}

fn display_arg(a: &Annotation) -> Result<String> {
    match a.args.first() {
        Some(AnnArg::Pos(Expr {
            kind: ExprKind::Str(s),
            ..
        })) => Ok(s.clone()),
        _ => Err(err("E0702", "`@display` takes a string", a.span)),
    }
}

fn name_arg(a: &Annotation) -> Result<String> {
    match a.args.first() {
        Some(AnnArg::Pos(Expr {
            kind: ExprKind::Name(s),
            ..
        })) => Ok(s.clone()),
        _ => Err(err("E0702", format!("`@{}` takes a name", a.name), a.span)),
    }
}

fn bool_arg(a: &Annotation) -> Result<bool> {
    match a.args.first() {
        Some(AnnArg::Pos(Expr {
            kind: ExprKind::Bool(b),
            ..
        })) => Ok(*b),
        _ => Err(err(
            "E0702",
            format!("`@{}` takes true or false", a.name),
            a.span,
        )),
    }
}

fn unknown_annotation(name: &str, span: Span, target: &str) -> Diagnostic {
    err(
        "E0701",
        format!("`@{name}` does not apply to a {target}"),
        span,
    )
    .with_help("annotations: @display @prefer @outside @internal @label @encoding @guarded")
}

enum DomainLiteral {
    Int(i64),
    Sym(String),
}

fn instance_grid(name: &str, instance: &Instance, file: Option<&File>) -> Option<[i64; 3]> {
    if let Some(d) = instance.grids.get(name) {
        return Some(*d);
    }
    file?.items.iter().find_map(|i| match &i.kind {
        ItemKind::GridValue { name: n, dims } if n == name => Some(*dims),
        _ => None,
    })
}

fn instance_domain(
    name: &str,
    instance: &Instance,
    file: Option<&File>,
) -> Result<Option<Vec<DomainLiteral>>> {
    if let Some(values) = instance.domains.get(name) {
        let mut out = Vec::new();
        for v in values {
            out.push(match v {
                IValue::Int(i) => DomainLiteral::Int(*i),
                IValue::Sym(s) => DomainLiteral::Sym(s.clone()),
                _ => {
                    return Err(err_nospan(format!(
                        "domain `{name}` values are symbols or integers"
                    )))
                }
            });
        }
        return Ok(Some(out));
    }
    let Some(file) = file else { return Ok(None) };
    for item in &file.items {
        if let ItemKind::Domain {
            name: n,
            value: Some(value),
            ..
        } = &item.kind
        {
            if n != name {
                continue;
            }
            let ExprKind::List(items) = &value.kind else {
                return Err(err(
                    "E0203",
                    "domain values are listed in brackets",
                    value.span,
                ));
            };
            let mut out = Vec::new();
            for v in items {
                out.push(match &v.kind {
                    ExprKind::Int(i) => DomainLiteral::Int(*i),
                    ExprKind::Name(s) => DomainLiteral::Sym(s.clone()),
                    _ => return Err(err("E0203", "domain values are names or integers", v.span)),
                });
            }
            return Ok(Some(out));
        }
    }
    Ok(None)
}

fn find_param_value<'a>(file: &'a File, name: &str) -> Option<&'a Expr> {
    file.items.iter().find_map(|i| match &i.kind {
        ItemKind::ParamValue { name: n, value } if n == name => Some(value),
        _ => None,
    })
}

fn find_fact_value<'a>(file: &'a File, name: &str) -> Option<&'a Expr> {
    file.items.iter().find_map(|i| match &i.kind {
        ItemKind::FactValue { name: n, value } if n == name => Some(value),
        _ => None,
    })
}

impl Names for Program {
    fn enum_name(&self, id: u32) -> &str {
        &self.enums[id as usize].name
    }
    fn variant_name(&self, enum_id: u32, variant: u32) -> &str {
        &self.enums[enum_id as usize].variants[variant as usize]
    }
    fn subset_name(&self, id: u32) -> &str {
        &self.subsets[id as usize].name
    }
    fn domain_name(&self, id: u32) -> &str {
        &self.domains[id as usize].name
    }
    fn symbol_name(&self, domain: u32, index: u32) -> &str {
        &self.domains[domain as usize].symbols[index as usize]
    }
    fn grid_name(&self, id: u32) -> &str {
        &self.grids[id as usize].name
    }
    fn cell_coords(&self, grid: u32, index: u32) -> [i64; 3] {
        self.grids[grid as usize].coords(index)
    }
    fn choice_name(&self, id: u32) -> &str {
        &self.choices[id as usize].name
    }
    fn member_name(&self, choice: u32, member: u32) -> &str {
        &self.choices[choice as usize].members[member as usize].name
    }
}
