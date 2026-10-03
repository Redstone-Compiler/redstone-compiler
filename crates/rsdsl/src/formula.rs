//! Formulas in negation normal form and their polarity-aware CNF encoding.

use crate::fxhash::FxMap;

pub type Lit = i32;

/// A solver-time formula. Negation is pushed to the literals.
#[derive(Debug, Clone, PartialEq)]
pub enum F {
    Const(bool),
    Lit(Lit),
    And(Vec<F>),
    Or(Vec<F>),
    Card(Box<Card>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardOp {
    Le,
    Ge,
    Eq,
}

/// `count(items) op k`.
#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    pub items: Vec<F>,
    pub op: CardOp,
    pub k: i64,
    pub encoding: Option<String>,
}

impl F {
    pub fn and(items: Vec<F>) -> F {
        F::junction(items, true)
    }

    pub fn or(items: Vec<F>) -> F {
        F::junction(items, false)
    }

    /// Flattens, folds constants, sorts literals first (by variable), and
    /// removes duplicates; a complementary pair decides the junction. The
    /// input vector is reused when nothing needs flattening.
    fn junction(mut items: Vec<F>, is_and: bool) -> F {
        let (unit, zero) = (F::Const(is_and), F::Const(!is_and));
        let (unit_lit, zero_lit) = if is_and { (1, -1) } else { (-1, 1) };
        let nested = |f: &F| match f {
            F::And(_) => is_and,
            F::Or(_) => !is_and,
            F::Const(_) => true,
            F::Lit(l) => *l == 1 || *l == -1,
            F::Card(_) => false,
        };
        if items.iter().any(nested) {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    F::Const(b) if b == is_and => {}
                    F::Const(_) => return zero,
                    F::Lit(l) if l == unit_lit => {}
                    F::Lit(l) if l == zero_lit => return zero,
                    F::And(inner) if is_and => out.extend(inner),
                    F::Or(inner) if !is_and => out.extend(inner),
                    other => out.push(other),
                }
            }
            items = out;
        }
        let key = |f: &F| match f {
            F::Lit(l) => (0u8, l.unsigned_abs(), *l),
            _ => (1u8, 0, 0),
        };
        if items.len() > 1 {
            items.sort_by(|a, b| key(a).cmp(&key(b)));
            items.dedup_by(|a, b| matches!((a, b), (F::Lit(x), F::Lit(y)) if x == y));
            let complementary = items
                .windows(2)
                .any(|w| matches!((&w[0], &w[1]), (F::Lit(x), F::Lit(y)) if *x == -*y));
            if complementary {
                return zero;
            }
        }
        match items.len() {
            0 => unit,
            1 => items.pop().unwrap(),
            _ if is_and => F::And(items),
            _ => F::Or(items),
        }
    }

    pub fn not(self) -> F {
        match self {
            F::Const(b) => F::Const(!b),
            F::Lit(l) => F::Lit(-l),
            F::And(items) => F::or(items.into_iter().map(F::not).collect()),
            F::Or(items) => F::and(items.into_iter().map(F::not).collect()),
            F::Card(card) => {
                let Card {
                    items,
                    op,
                    k,
                    encoding,
                } = *card;
                let make = |op, k| {
                    F::card(Card {
                        items: items.clone(),
                        op,
                        k,
                        encoding: encoding.clone(),
                    })
                };
                match op {
                    CardOp::Le => make(CardOp::Ge, k + 1),
                    CardOp::Ge => make(CardOp::Le, k - 1),
                    CardOp::Eq => F::or(vec![make(CardOp::Le, k - 1), make(CardOp::Ge, k + 1)]),
                }
            }
        }
    }

    pub fn imp(a: F, b: F) -> F {
        F::or(vec![a.not(), b])
    }

    pub fn iff(a: F, b: F) -> F {
        if let (F::Const(x), other) | (other, F::Const(x)) = (&a, &b) {
            return if *x {
                other.clone()
            } else {
                other.clone().not()
            };
        }
        F::and(vec![F::imp(a.clone(), b.clone()), F::imp(b, a)])
    }

    pub fn xor(a: F, b: F) -> F {
        F::iff(a, b).not()
    }

    /// Folds trivial cardinalities to plain formulas.
    pub fn card(card: Card) -> F {
        let n = card.items.len() as i64;
        let all = || F::and(card.items.clone());
        let none = || F::and(card.items.iter().cloned().map(F::not).collect());
        match card.op {
            CardOp::Le if card.k < 0 => F::Const(false),
            CardOp::Le if card.k >= n => F::Const(true),
            CardOp::Le if card.k == 0 => none(),
            CardOp::Ge if card.k <= 0 => F::Const(true),
            CardOp::Ge if card.k > n => F::Const(false),
            CardOp::Ge if card.k == n => all(),
            CardOp::Ge if card.k == 1 => F::or(card.items.clone()),
            CardOp::Eq if card.k < 0 || card.k > n => F::Const(false),
            CardOp::Eq if card.k == 0 => none(),
            CardOp::Eq if card.k == n => all(),
            _ => F::Card(Box::new(card)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pol {
    Pos,
    Neg,
    Both,
}

impl Pol {
    fn pos(self) -> bool {
        matches!(self, Pol::Pos | Pol::Both)
    }
    fn neg(self) -> bool {
        matches!(self, Pol::Neg | Pol::Both)
    }
}

struct Aux {
    lit: Lit,
    pos: bool,
    neg: bool,
}

/// What a variable stands for; resolved to text by the program.
#[derive(Debug, Clone)]
pub enum VarOrigin {
    True,
    /// Created for a declaration-level object (choice option, def, ...).
    Named(u32),
    /// Tseitin or counter auxiliary, created under provenance `origin`.
    Aux(u32),
    /// Guard selector.
    Guard(u32),
}

/// CNF under construction.
pub struct Encoder {
    pub num_vars: i32,
    pub literals: Vec<Lit>,
    pub clause_count: usize,
    /// Provenance id of each clause.
    pub clause_origin: Vec<u32>,
    /// Provenance id for clauses emitted now.
    pub origin: u32,
    /// Selector appended (negated) to clauses from `require`.
    pub guard: Option<Lit>,
    pub var_origin: Vec<VarOrigin>,
    cache: FxMap<(bool, Vec<Lit>), Aux>,
    /// Reused buffer for normalizing clauses.
    scratch: Vec<Lit>,
    /// Represent OR auxiliaries as negated AND variables (the default).
    pub or_as_negated_and: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        let mut encoder = Self {
            num_vars: 0,
            literals: Vec::new(),
            clause_count: 0,
            clause_origin: Vec::new(),
            origin: 0,
            guard: None,
            // Index 0 is unused so that `var_origin[var]` works directly.
            var_origin: vec![VarOrigin::True],
            cache: FxMap::default(),
            scratch: Vec::new(),
            or_as_negated_and: true,
        };
        let t = encoder.new_var(VarOrigin::True);
        debug_assert_eq!(t, 1);
        encoder.raw_clause(&[1]);
        encoder
    }

    pub fn tru(&self) -> Lit {
        1
    }

    pub fn new_var(&mut self, origin: VarOrigin) -> Lit {
        self.num_vars += 1;
        self.var_origin.push(origin);
        self.num_vars
    }

    fn raw_clause(&mut self, lits: &[Lit]) {
        self.literals.extend_from_slice(lits);
        self.literals.push(0);
        self.clause_count += 1;
        self.clause_origin.push(self.origin);
    }

    /// Emits a clause, dropping false literals and skipping satisfied ones.
    pub fn clause(&mut self, lits: &[Lit]) {
        self.clause_with(lits, None);
    }

    fn clause_with(&mut self, lits: &[Lit], extra: Option<Lit>) {
        if lits.contains(&1) || extra == Some(1) {
            return;
        }
        let mut out = std::mem::take(&mut self.scratch);
        out.clear();
        out.extend(lits.iter().copied().chain(extra).filter(|&l| l != -1));
        out.sort_unstable_by_key(|l| (l.abs(), *l));
        out.dedup();
        if !out.windows(2).any(|w| w[0] == -w[1]) {
            if out.is_empty() {
                out.push(-1);
            }
            self.raw_clause(&out);
        }
        self.scratch = out;
    }

    fn guarded_clause(&mut self, lits: &[Lit]) {
        let guard = self.guard.map(|g| -g);
        self.clause_with(lits, guard);
    }

    /// A literal equivalent to `f` in the directions `pol` requires.
    pub fn lit_of(&mut self, f: &F, pol: Pol) -> Lit {
        match f {
            F::Const(true) => 1,
            F::Const(false) => -1,
            F::Lit(l) => *l,
            F::And(items) | F::Or(items) => {
                let is_and = matches!(f, F::And(_));
                let (unit, zero) = if is_and { (1, -1) } else { (-1, 1) };
                let mut child_lits: Vec<Lit> = Vec::with_capacity(items.len());
                for child in items {
                    let lit = self.lit_of(child, pol);
                    if lit == zero {
                        return zero;
                    }
                    if lit != unit && !child_lits.contains(&lit) {
                        if child_lits.contains(&-lit) {
                            return zero;
                        }
                        child_lits.push(lit);
                    }
                }
                match child_lits.len() {
                    0 => return unit,
                    1 => return child_lits[0],
                    _ => {}
                }
                let mut sorted = child_lits.clone();
                sorted.sort_unstable();
                let key = (is_and, sorted);
                if !self.cache.contains_key(&key) {
                    let var = self.new_var(VarOrigin::Aux(self.origin));
                    // A disjunction is the negation of a conjunction variable,
                    // so the solver's default false phase starts every
                    // auxiliary junction as "true" for ORs and "false" for
                    // ANDs, matching the hand-written encoder.
                    let lit = if is_and || !self.or_as_negated_and {
                        var
                    } else {
                        -var
                    };
                    self.cache.insert(
                        key.clone(),
                        Aux {
                            lit,
                            pos: false,
                            neg: false,
                        },
                    );
                }
                let entry = self.cache.get_mut(&key).unwrap();
                let aux = entry.lit;
                let need_pos = pol.pos() && !entry.pos;
                let need_neg = pol.neg() && !entry.neg;
                entry.pos |= pol.pos();
                entry.neg |= pol.neg();
                // aux -> f
                if need_pos {
                    if is_and {
                        for &c in &child_lits {
                            self.clause(&[-aux, c]);
                        }
                    } else {
                        let mut clause = vec![-aux];
                        clause.extend(&child_lits);
                        self.clause(&clause);
                    }
                }
                // f -> aux
                if need_neg {
                    if is_and {
                        let mut clause: Vec<Lit> = child_lits.iter().map(|c| -c).collect();
                        clause.push(aux);
                        self.clause(&clause);
                    } else {
                        for &c in &child_lits {
                            self.clause(&[aux, -c]);
                        }
                    }
                }
                aux
            }
            F::Card(card) => {
                // Reified cardinality: an auxiliary literal equivalent to the
                // constraint, through a sequential counter with outputs.
                self.reified_card(card, pol)
            }
        }
    }

    /// Asserts `f`, distributing small conjunctions so that implications
    /// such as `p -> (a and b)` become `-p a`, `-p b` without auxiliaries.
    pub fn require(&mut self, f: F) {
        match f {
            F::Const(true) => {}
            F::Const(false) => self.guarded_clause(&[]),
            F::Lit(l) => self.guarded_clause(&[l]),
            F::And(items) => {
                for item in items {
                    self.require(item);
                }
            }
            F::Card(card) => self.card(*card),
            F::Or(mut items) => {
                // Junctions keep their literals first.
                let split = items
                    .iter()
                    .position(|f| !matches!(f, F::Lit(_)))
                    .unwrap_or(items.len());
                let compound = items.split_off(split);
                let mut lits: Vec<Lit> = items
                    .iter()
                    .map(|f| match f {
                        F::Lit(l) => *l,
                        _ => unreachable!(),
                    })
                    .collect();
                if compound.is_empty() {
                    self.guarded_clause(&lits);
                    return;
                }
                // Distribute one conjunction whose parts are literals or
                // flat disjunctions; the rest become auxiliary literals.
                let mut compound = compound;
                let distribute = compound.iter().position(|c| match c {
                    F::And(parts) => {
                        parts.len() <= 64
                            && parts.iter().all(|p| match p {
                                F::Lit(_) => true,
                                F::Or(inner) => inner.iter().all(|q| matches!(q, F::Lit(_))),
                                _ => false,
                            })
                    }
                    _ => false,
                });
                let chosen = distribute.map(|i| compound.swap_remove(i));
                for other in &compound {
                    let l = self.lit_of(other, Pol::Pos);
                    lits.push(l);
                }
                match chosen {
                    Some(F::And(parts)) => {
                        // Distributing copies `lits` into every part; naming
                        // their disjunction once is smaller when both are
                        // long, and propagates the same.
                        let shared = (parts.len() - 1) * lits.len().saturating_sub(1) > 2;
                        if shared {
                            let or = F::or(lits.iter().map(|&l| F::Lit(l)).collect());
                            lits = vec![self.lit_of(&or, Pol::Pos)];
                        }
                        let prefix = lits.len();
                        for part in parts {
                            match part {
                                F::Lit(l) => lits.push(l),
                                F::Or(inner) => lits.extend(inner.iter().map(|q| match q {
                                    F::Lit(l) => *l,
                                    _ => unreachable!("checked above"),
                                })),
                                _ => unreachable!("checked above"),
                            }
                            self.guarded_clause(&lits);
                            lits.truncate(prefix);
                        }
                    }
                    _ => self.guarded_clause(&lits),
                }
            }
        }
    }

    /// Top-level cardinality constraint.
    fn card(&mut self, card: Card) {
        let lits: Vec<Lit> = card
            .items
            .iter()
            .map(|item| self.lit_of(item, Pol::Both))
            .collect();
        let pairwise = card.encoding.as_deref() == Some("pairwise");
        match card.op {
            CardOp::Le => self.at_most(&lits, card.k, pairwise),
            CardOp::Ge => {
                if card.k == 1 {
                    self.guarded_clause(&lits);
                } else {
                    let negated: Vec<Lit> = lits.iter().map(|l| -l).collect();
                    self.at_most(&negated, lits.len() as i64 - card.k, pairwise);
                }
            }
            CardOp::Eq => {
                if card.k == 1 {
                    self.guarded_clause(&lits);
                    self.at_most(&lits, 1, pairwise);
                } else {
                    self.at_most(&lits, card.k, pairwise);
                    let negated: Vec<Lit> = lits.iter().map(|l| -l).collect();
                    self.at_most(&negated, lits.len() as i64 - card.k, pairwise);
                }
            }
        }
    }

    /// At most `k` of `lits`: pairwise for k = 1 and few literals, otherwise
    /// a sequential counter (Sinz 2005).
    pub fn at_most(&mut self, lits: &[Lit], k: i64, force_pairwise: bool) {
        let lits: Vec<Lit> = lits.iter().copied().filter(|&l| l != -1).collect();
        let forced = lits.iter().filter(|&&l| l == 1).count() as i64;
        let lits: Vec<Lit> = lits.into_iter().filter(|&l| l != 1).collect();
        let k = k - forced;
        if k < 0 {
            self.guarded_clause(&[]);
            return;
        }
        let n = lits.len();
        if k as usize >= n {
            return;
        }
        if k == 0 {
            for &l in &lits {
                self.guarded_clause(&[-l]);
            }
            return;
        }
        if k == 1 && (n <= 6 || force_pairwise) {
            for i in 0..n {
                for j in i + 1..n {
                    self.guarded_clause(&[-lits[i], -lits[j]]);
                }
            }
            return;
        }
        let k = k as usize;
        let origin = self.origin;
        if k == 1 {
            let prefix: Vec<Lit> = (0..n - 1)
                .map(|_| self.new_var(VarOrigin::Aux(origin)))
                .collect();
            self.guarded_clause(&[-lits[0], prefix[0]]);
            for i in 1..n - 1 {
                self.guarded_clause(&[-lits[i], prefix[i]]);
                self.guarded_clause(&[-prefix[i - 1], prefix[i]]);
                self.guarded_clause(&[-lits[i], -prefix[i - 1]]);
            }
            self.guarded_clause(&[-lits[n - 1], -prefix[n - 2]]);
            return;
        }
        // counters[j]: at least j + 1 of the literals seen so far are true.
        let mut previous: Vec<Lit> = Vec::new();
        for (index, &lit) in lits.iter().enumerate() {
            let width = k.min(index + 1);
            let current: Vec<Lit> = (0..width)
                .map(|_| self.new_var(VarOrigin::Aux(origin)))
                .collect();
            self.guarded_clause(&[-lit, current[0]]);
            for j in 0..width {
                if j < previous.len() {
                    self.guarded_clause(&[-previous[j], current[j]]);
                }
                if j > 0 && j - 1 < previous.len() {
                    self.guarded_clause(&[-lit, -previous[j - 1], current[j]]);
                }
            }
            if previous.len() == k {
                self.guarded_clause(&[-lit, -previous[k - 1]]);
            }
            previous = current;
        }
    }

    /// Totalizer (Bailleux & Boufkhad 2003) over `inputs`, truncated at
    /// `cap + 1`: `outputs[j]` is implied by "at least j + 1 inputs are true".
    /// Only that direction is encoded, which is what upper bounds need:
    /// assuming `-outputs[b]` allows at most `b` true inputs. Bounds stay
    /// assumptions, so one solver can tighten them incrementally.
    pub fn totalizer(&mut self, inputs: &[Lit], cap: usize) -> Vec<Lit> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let limit = cap + 1;
        let origin = self.origin;
        let mut level: Vec<Vec<Lit>> = inputs.iter().map(|&l| vec![l]).collect();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut pairs = level.into_iter();
            while let Some(left) = pairs.next() {
                let Some(right) = pairs.next() else {
                    next.push(left);
                    break;
                };
                let width = (left.len() + right.len()).min(limit);
                let outputs: Vec<Lit> = (0..width)
                    .map(|_| self.new_var(VarOrigin::Aux(origin)))
                    .collect();
                for i in 0..=left.len() {
                    for j in 0..=right.len() {
                        if i + j == 0 {
                            continue;
                        }
                        let target = outputs[(i + j).min(width) - 1];
                        let mut clause = vec![target];
                        if i > 0 {
                            clause.push(-left[i - 1]);
                        }
                        if j > 0 {
                            clause.push(-right[j - 1]);
                        }
                        self.clause(&clause);
                    }
                }
                next.push(outputs);
            }
            level = next;
        }
        level.pop().unwrap()
    }

    /// Totalizer-style reification: outputs `ge[j]` mean "at least j + 1".
    fn reified_card(&mut self, card: &Card, pol: Pol) -> Lit {
        let lits: Vec<Lit> = card
            .items
            .iter()
            .map(|item| self.lit_of(item, Pol::Both))
            .collect();
        let n = lits.len();
        // Unary counter with full equivalence: ge[i][j] <-> at least j+1 of
        // the first i+1 literals.
        let mut previous: Vec<Lit> = Vec::new();
        for (i, &lit) in lits.iter().enumerate() {
            let width = i + 1;
            let mut current = Vec::with_capacity(width);
            for j in 0..width {
                let keep = previous.get(j).copied().unwrap_or(-1);
                let step = if j == 0 {
                    1
                } else {
                    previous.get(j - 1).copied().unwrap_or(-1)
                };
                // current[j] <-> keep or (lit and step)
                let with_lit = self.lit_of(&F::and(vec![F::Lit(lit), F::Lit(step)]), Pol::Both);
                let value = self.lit_of(&F::or(vec![F::Lit(keep), F::Lit(with_lit)]), Pol::Both);
                current.push(value);
            }
            previous = current;
        }
        let ge = |j: i64| -> Lit {
            if j <= 0 {
                1
            } else if j as usize > n {
                -1
            } else {
                previous[j as usize - 1]
            }
        };
        let f = match card.op {
            CardOp::Le => F::Lit(-ge(card.k + 1)),
            CardOp::Ge => F::Lit(ge(card.k)),
            CardOp::Eq => F::and(vec![F::Lit(ge(card.k)), F::Lit(-ge(card.k + 1))]),
        };
        self.lit_of(&f, pol)
    }
}
