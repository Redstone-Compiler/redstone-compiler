//! Public API: parsing a model, grounding it, and reading the result back.

use std::collections::HashMap;
use std::io::{self, Write};

use crate::ast::{File, Header};
use crate::diag::{Diagnostic, Error, SourceMap, Span};
use crate::formula::{Lit, VarOrigin, F};
use crate::ground::{Dim, GroundOptions, NameRef, Program, VarIndexKind};
use crate::instance::{IValue, Instance};
use crate::parser;
use crate::value::{Names, Show, Value};

/// A parsed model, ready to be grounded against instances.
#[derive(Debug, Clone)]
pub struct Model {
    file: File,
    name: String,
    sources: SourceMap,
}

impl Model {
    pub fn parse(file_name: &str, text: &str) -> Result<Model, Error> {
        let mut sources = SourceMap::default();
        let id = sources.add(file_name, text);
        let file = parser::parse(id, text).map_err(|d| Error {
            diagnostics: vec![d],
            sources: sources.clone(),
        })?;
        let name = match &file.header {
            Header::Model { name } => name.clone(),
            Header::Instance { .. } => {
                return Err(Error {
                    diagnostics: vec![Diagnostic::error_nospan(
                        "E0102",
                        format!("`{file_name}` is an instance file, not a model"),
                    )],
                    sources,
                })
            }
        };
        Ok(Model {
            file,
            name,
            sources,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Grounds against an instance built in Rust.
    pub fn ground(&self, instance: &Instance, options: GroundOptions) -> Result<Program, Error> {
        let mut program =
            Program::ground(&self.file, instance, None, options).map_err(|d| Error {
                diagnostics: vec![d],
                sources: self.sources.clone(),
            })?;
        program.sources = self.sources.clone();
        Ok(program)
    }

    /// Grounds against an instance file (`instance NAME of MODEL;`).
    pub fn ground_file(
        &self,
        file_name: &str,
        text: &str,
        options: GroundOptions,
    ) -> Result<Program, Error> {
        let mut sources = self.sources.clone();
        let id = sources.add(file_name, text);
        let fail = |d: Diagnostic, sources: &SourceMap| Error {
            diagnostics: vec![d],
            sources: sources.clone(),
        };
        let file = parser::parse(id, text).map_err(|d| fail(d, &sources))?;
        match &file.header {
            Header::Instance { of, .. } if *of == self.name => {}
            Header::Instance { of, .. } => {
                return Err(fail(
                    Diagnostic::error_nospan(
                        "E0600",
                        format!("instance is for model `{of}`, not `{}`", self.name),
                    ),
                    &sources,
                ))
            }
            Header::Model { .. } => {
                return Err(fail(
                    Diagnostic::error_nospan(
                        "E0600",
                        format!("`{file_name}` is a model, not an instance"),
                    ),
                    &sources,
                ))
            }
        }
        let empty = Instance::default();
        let mut program = Program::ground(&self.file, &empty, Some(&file), options)
            .map_err(|d| fail(d, &sources))?;
        program.sources = sources;
        Ok(program)
    }
}

/// A `@guarded` selector: assuming `-lit` relaxes that rule instance.
#[derive(Debug, Clone)]
pub struct Guard {
    pub rule: String,
    pub key: Vec<(String, IValue)>,
    pub lit: Lit,
}

/// How much explanation `write_dimacs` adds as comment lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Comments {
    /// Plain DIMACS, exactly as the solver receives it.
    #[default]
    None,
    /// A legend for the variables that have a direct meaning.
    Legend,
    /// The legend, a header per rule instance, and a sentence per clause.
    Explained,
}

impl Program {
    pub fn num_vars(&self) -> i32 {
        self.encoder.num_vars
    }

    pub fn clause_count(&self) -> usize {
        self.encoder.clause_count
    }

    /// Zero-terminated clauses, flattened.
    pub fn literals(&self) -> &[Lit] {
        &self.encoder.literals
    }

    /// Adds a clause from outside the model (for example a blocked layout).
    pub fn add_clause(&mut self, lits: &[Lit], label: &str) {
        let origin = self.origins.len() as u32;
        self.origins.push((label.to_owned(), String::new()));
        self.encoder.origin = origin;
        self.encoder.clause(lits);
        self.encoder.origin = 0;
    }

    pub fn warnings(&self) -> &[Diagnostic] {
        &self.warnings
    }

    pub fn render_warnings(&self) -> String {
        self.warnings
            .iter()
            .map(|w| w.render(&self.sources))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// `(rule, clauses, grounding time)` in declaration order.
    pub fn rule_stats(&self) -> Vec<(String, usize, std::time::Duration)> {
        self.rules
            .iter()
            .map(|rule| (rule.label.clone(), rule.clauses, rule.time))
            .collect()
    }

    /// Clauses emitted by each rule, in declaration order.
    pub fn rule_clause_counts(&self) -> Vec<(String, usize)> {
        self.rules
            .iter()
            .map(|rule| (rule.label.clone(), rule.clauses))
            .collect()
    }

    fn key_of(&self, dims: &[Dim], key: &[IValue]) -> Option<Vec<Value>> {
        if dims.len() != key.len() {
            return None;
        }
        dims.iter()
            .zip(key)
            .map(|(dim, v)| self.ivalue(v, &dim.ty, Span::default()).ok())
            .collect()
    }

    fn decl_id(&self, name: &str) -> Option<crate::ground::Decl> {
        self.names.get(name).copied()
    }

    /// The literal of `choice[key] is member(payload)`; `None` when that
    /// option does not exist at `key`.
    pub fn option_lit(
        &self,
        choice: &str,
        key: &[IValue],
        member: &str,
        payload: &[IValue],
    ) -> Option<Lit> {
        let crate::ground::Decl::Choice(c) = self.decl_id(choice)? else {
            return None;
        };
        let def = &self.choices[c as usize];
        let key = self.key_of(&def.index, key)?;
        let k = *def.lookup.get(&key)?;
        let m = def.members.iter().position(|x| x.name == member)?;
        let params: Vec<Dim> = def.members[m]
            .params
            .iter()
            .map(|(n, t)| Dim {
                name: n.clone(),
                ty: t.clone(),
            })
            .collect();
        let payload = self.key_of(&params, payload)?;
        def.options[k as usize]
            .iter()
            .find(|o| o.member == m as u32 && *o.payload == *payload)
            .map(|o| o.lit)
    }

    /// Every option of `choice[key]` as `(member, literal)`.
    pub fn choice_options(&self, choice: &str, key: &[IValue]) -> Option<Vec<(IValue, Lit)>> {
        let crate::ground::Decl::Choice(c) = self.decl_id(choice)? else {
            return None;
        };
        let def = &self.choices[c as usize];
        let key = self.key_of(&def.index, key)?;
        let k = *def.lookup.get(&key)?;
        Some(
            def.options[k as usize]
                .iter()
                .map(|o| {
                    (
                        self.to_ivalue(&Value::Member(c, o.member, o.payload.clone())),
                        o.lit,
                    )
                })
                .collect(),
        )
    }

    /// Literal of an exported definition; constants are the true literal
    /// `1` or its negation.
    pub fn def_lit(&self, def: &str, key: &[IValue]) -> Option<Lit> {
        let crate::ground::Decl::Def(d) = self.decl_id(def)? else {
            return None;
        };
        let def = &self.defs[d as usize];
        let key = self.key_of(&def.index, key)?;
        match def.memo.get(&key)? {
            F::Const(true) => Some(1),
            F::Const(false) => Some(-1),
            F::Lit(l) => Some(*l),
            _ => None,
        }
    }

    /// Literal of a variable; `var V over R` takes the relation tuple.
    pub fn var_lit(&self, var: &str, key: &[IValue]) -> Option<Lit> {
        let crate::ground::Decl::Var(v) = self.decl_id(var)? else {
            return None;
        };
        let def = &self.vars[v as usize];
        let dims: Vec<Dim> = match &def.index {
            VarIndexKind::Dims(dims) => dims.to_vec(),
            VarIndexKind::Over(rel) => self.relations[*rel as usize]
                .params
                .iter()
                .map(|(n, t)| Dim {
                    name: n.as_str().into(),
                    ty: t.clone(),
                })
                .collect(),
        };
        let key = self.key_of(&dims, key)?;
        def.lookup.get(&key).copied()
    }

    /// Order-encoding literals of an integer: element `i` means
    /// `value >= lo + i + 1`.
    pub fn int_lits(&self, int: &str, key: &[IValue]) -> Option<Vec<Lit>> {
        let crate::ground::Decl::Int(i) = self.decl_id(int)? else {
            return None;
        };
        let def = &self.ints[i as usize];
        let key = self.key_of(&def.index, key)?;
        def.lookup.get(&key).map(|lits| lits.to_vec())
    }

    /// The tuples of a relation with their literals.
    pub fn relation(&self, name: &str) -> Option<Vec<(Vec<IValue>, Lit)>> {
        let crate::ground::Decl::Relation(r) = self.decl_id(name)? else {
            return None;
        };
        Some(
            self.relations[r as usize]
                .tuples
                .iter()
                .map(|(tuple, lit)| (tuple.iter().map(|v| self.to_ivalue(v)).collect(), *lit))
                .collect(),
        )
    }

    pub fn guards(&self) -> Vec<Guard> {
        self.guards
            .iter()
            .map(|g| Guard {
                rule: g.rule.clone(),
                key: g
                    .names
                    .iter()
                    .cloned()
                    .zip(g.key.iter().map(|v| self.to_ivalue(v)))
                    .collect(),
                lit: g.lit,
            })
            .collect()
    }

    pub fn to_ivalue(&self, v: &Value) -> IValue {
        match v {
            Value::Bool(b) => IValue::Bool(*b),
            Value::Int(i) => IValue::Int(*i),
            Value::Str(s) => IValue::Sym(s.to_string()),
            Value::Sym(d, i) => IValue::Sym(self.symbol_name(*d, *i).to_owned()),
            Value::Variant(e, x) => IValue::Sym(self.variant_name(*e, *x).to_owned()),
            Value::Cell(g, i) => {
                let [x, y, z] = self.cell_coords(*g, *i);
                IValue::Cell(x, y, z)
            }
            Value::Member(c, m, payload) => IValue::Member(
                self.member_name(*c, *m).to_owned(),
                payload.iter().map(|p| self.to_ivalue(p)).collect(),
            ),
            Value::Tuple(items) | Value::Set(items) => {
                IValue::Tuple(items.iter().map(|p| self.to_ivalue(p)).collect())
            }
            Value::None => IValue::None,
        }
    }

    // ----- names ---------------------------------------------------------------

    fn render(&self, template: Option<&str>, family: &str, names: &[(&str, String)]) -> String {
        match template {
            Some(template) => {
                let mut out = template.to_owned();
                for (name, value) in names {
                    out = out.replace(&format!("{{{name}}}"), value);
                }
                out
            }
            None => {
                let args = names
                    .iter()
                    .filter(|(n, _)| *n != "member")
                    .map(|(_, v)| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                match names.iter().find(|(n, _)| *n == "member") {
                    Some((_, member)) => format!("{family}[{args}] is {member}"),
                    None => format!("{family}[{args}]"),
                }
            }
        }
    }

    fn bind_names<'a>(
        &self,
        dims: impl Iterator<Item = &'a str>,
        key: &[Value],
    ) -> Vec<(&'a str, String)> {
        dims.zip(key)
            .map(|(n, v)| (n, Show(v, self).to_string()))
            .collect()
    }

    fn name_text(&self, name: &NameRef) -> String {
        match *name {
            NameRef::Option { choice, key, .. } | NameRef::Occupied { choice, key } => {
                let def = &self.choices[choice as usize];
                let key_values = &def.keys[key as usize];
                let mut names = self.bind_names(def.index.iter().map(|d| &*d.name), key_values);
                let member = match *name {
                    NameRef::Option { option, .. } => {
                        let o = &def.options[key as usize][option as usize];
                        Show(&Value::Member(choice, o.member, o.payload.clone()), self).to_string()
                    }
                    _ => {
                        let preferred = def.options[key as usize]
                            .iter()
                            .find(|o| o.lit < 0)
                            .map(|o| self.member_name(choice, o.member).to_owned())
                            .unwrap_or_default();
                        format!("≠{preferred}")
                    }
                };
                names.push(("member", member));
                self.render(def.display.as_deref(), &def.name, &names)
            }
            NameRef::Var { var, key } => {
                let def = &self.vars[var as usize];
                let key_values = &def.keys[key as usize];
                let names = match &def.index {
                    VarIndexKind::Dims(dims) => {
                        self.bind_names(dims.iter().map(|d| &*d.name), key_values)
                    }
                    VarIndexKind::Over(rel) => self.bind_names(
                        self.relations[*rel as usize]
                            .params
                            .iter()
                            .map(|(n, _)| n.as_str()),
                        key_values,
                    ),
                };
                self.render(def.display.as_deref(), &def.name, &names)
            }
            NameRef::Def { def, key } => {
                let def = &self.defs[def as usize];
                let names =
                    self.bind_names(def.index.iter().map(|d| &*d.name), &def.keys[key as usize]);
                self.render(def.display.as_deref(), &def.name, &names)
            }
            NameRef::Rel { rel, tuple } => {
                let def = &self.relations[rel as usize];
                let names = self.bind_names(
                    def.params.iter().map(|(n, _)| n.as_str()),
                    &def.tuples[tuple as usize].0,
                );
                self.render(def.display.as_deref(), &def.name, &names)
            }
            NameRef::IntGe { int, key, value } => {
                let def = &self.ints[int as usize];
                let names =
                    self.bind_names(def.index.iter().map(|d| &*d.name), &def.keys[key as usize]);
                format!(
                    "{} ≥ {value}",
                    self.render(def.display.as_deref(), &def.name, &names)
                )
            }
        }
    }

    /// What variable `var` stands for.
    pub fn var_name(&self, var: i32) -> String {
        match &self.encoder.var_origin[var as usize] {
            VarOrigin::True => "상수 참".to_owned(),
            VarOrigin::Named(index) => self.name_text(&self.name_refs[*index as usize]),
            VarOrigin::Aux(_) => format!("보조#{var}"),
            VarOrigin::Guard(index) => {
                let g = &self.guards[*index as usize];
                let key = g
                    .names
                    .iter()
                    .zip(&g.key)
                    .map(|(n, v)| format!("{n}={}", Show(v, self)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("가드[{}; {key}]", g.rule)
            }
        }
    }

    /// Names of a literal: aliases of exactly that literal first.
    fn lit_texts(&self, aliases: &HashMap<Lit, Vec<String>>, lit: Lit) -> Option<String> {
        if let Some(texts) = aliases.get(&lit) {
            return Some(texts.join(" = "));
        }
        if lit > 0 && !matches!(self.encoder.var_origin[lit as usize], VarOrigin::Aux(_)) {
            return Some(self.var_name(lit));
        }
        None
    }

    /// Writes the CNF as DIMACS with the requested comments. `header`
    /// lines are written as comments first (ignored with `Comments::None`).
    pub fn write_dimacs(
        &self,
        out: &mut dyn Write,
        comments: Comments,
        header: &[String],
    ) -> io::Result<()> {
        let alias_texts: HashMap<Lit, Vec<String>> = if comments == Comments::None {
            HashMap::new()
        } else {
            self.aliases
                .iter()
                .map(|(&lit, names)| {
                    (
                        lit,
                        names
                            .iter()
                            .map(|&n| self.name_text(&self.name_refs[n as usize]))
                            .collect(),
                    )
                })
                .collect()
        };
        if comments != Comments::None {
            for line in header {
                writeln!(out, "c {line}")?;
            }
            writeln!(
                out,
                "c Lines starting with `c` are comments; solvers ignore them."
            )?;
            writeln!(
                out,
                "c A clause lists literals ending in 0 and requires at least one to hold;"
            )?;
            writeln!(out, "c -N means \"variable N is false\".")?;
            writeln!(out, "c")?;
            writeln!(out, "c Variable legend (unlisted variables are auxiliary):")?;
            for var in 1..=self.encoder.num_vars {
                let mut texts = Vec::new();
                if !matches!(self.encoder.var_origin[var as usize], VarOrigin::Aux(_)) {
                    texts.push(self.var_name(var));
                }
                if let Some(names) = alias_texts.get(&var) {
                    texts.extend(names.iter().cloned());
                }
                if let Some(names) = alias_texts.get(&-var) {
                    texts.extend(names.iter().map(|n| format!("¬({n})")));
                }
                if !texts.is_empty() {
                    writeln!(out, "c   {var} = {}", texts.join(" = "))?;
                }
            }
            writeln!(out, "c")?;
        }
        writeln!(
            out,
            "p cnf {} {}",
            self.encoder.num_vars, self.encoder.clause_count
        )?;
        let explained = comments == Comments::Explained;
        let mut last_origin = u32::MAX;
        let mut clause = Vec::new();
        let mut index = 0usize;
        for &lit in &self.encoder.literals {
            if lit != 0 {
                clause.push(lit);
                continue;
            }
            if explained {
                let origin = self.encoder.clause_origin[index];
                if origin != last_origin {
                    let (label, valuation) = &self.origins[origin as usize];
                    writeln!(out, "c")?;
                    if valuation.is_empty() {
                        writeln!(out, "c [규칙] {label}")?;
                    } else {
                        writeln!(out, "c [규칙] {label} ({valuation})")?;
                    }
                    last_origin = origin;
                }
            }
            let numbers = clause
                .iter()
                .map(|l: &Lit| l.to_string())
                .collect::<Vec<_>>();
            writeln!(out, "{} 0", numbers.join(" "))?;
            if explained {
                writeln!(out, "c   {}", self.explain(&alias_texts, &clause))?;
            }
            clause.clear();
            index += 1;
        }
        Ok(())
    }

    /// Reads a clause back as "if all premises hold, a conclusion does".
    fn explain(&self, aliases: &HashMap<Lit, Vec<String>>, clause: &[Lit]) -> String {
        let mut premises = Vec::new();
        let mut conclusions = Vec::new();
        for &lit in clause {
            if lit == 1 {
                conclusions.push("상수 참".to_owned());
            } else if lit == -1 {
                premises.push("상수 참".to_owned());
            } else if let Some(text) = self.lit_texts(aliases, lit) {
                conclusions.push(text);
            } else if let Some(text) = self.lit_texts(aliases, -lit) {
                premises.push(text);
            } else if lit > 0 {
                conclusions.push(format!("보조#{lit}"));
            } else {
                premises.push(format!("보조#{}", -lit));
            }
        }
        match (premises.is_empty(), conclusions.len()) {
            (true, 1) => format!("항상: {}", conclusions[0]),
            (true, _) => format!("하나 이상 참: {}", conclusions.join(" | ")),
            (false, 0) if premises.len() == 1 => format!("항상 거짓: {}", premises[0]),
            (false, 0) => format!("동시에 참일 수 없음: {}", premises.join(" & ")),
            (false, _) => format!(
                "만약 {} 이면 → {}",
                premises.join(" 그리고 "),
                conclusions.join(" 또는 ")
            ),
        }
    }
}
