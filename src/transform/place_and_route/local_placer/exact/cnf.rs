//! Minimal CNF builder: DIMACS-style signed literals and small Tseitin helpers.

pub(super) type Lit = i32;

#[derive(Debug, Clone)]
pub(super) struct Cnf {
    num_vars: i32,
    /// Zero-terminated clauses, flattened.
    literals: Vec<Lit>,
    clause_count: usize,
    true_lit: Lit,
    /// `(first clause index, rule)`: which modeling rule emitted each run of
    /// clauses, so exported formulas can be explained.
    rules: Vec<(usize, &'static str)>,
}

impl Cnf {
    pub(super) fn new() -> Self {
        let mut cnf = Self {
            num_vars: 0,
            literals: Vec::new(),
            clause_count: 0,
            true_lit: 0,
            rules: vec![(0, "상수: 변수 1은 항상 참")],
        };
        let true_lit = cnf.new_var();
        cnf.literals.extend([true_lit, 0]);
        cnf.clause_count = 1;
        cnf.true_lit = true_lit;
        cnf
    }

    /// Wraps an already built formula whose variable 1 is constant true.
    pub(super) fn from_literals(num_vars: i32, literals: Vec<Lit>, clause_count: usize) -> Self {
        debug_assert_eq!(literals.get(..2), Some(&[1, 0][..]));
        Self {
            num_vars,
            literals,
            clause_count,
            true_lit: 1,
            rules: vec![(0, "rsdsl 모델 (규칙별 설명은 rsdsl DIMACS 출력 참고)")],
        }
    }

    pub(super) fn new_var(&mut self) -> Lit {
        self.num_vars += 1;
        self.num_vars
    }

    pub(super) fn num_vars(&self) -> i32 {
        self.num_vars
    }

    pub(super) fn clause_count(&self) -> usize {
        self.clause_count
    }

    pub(super) fn literals(&self) -> &[Lit] {
        &self.literals
    }

    /// Labels the clauses added from now on with `rule`.
    pub(super) fn rule(&mut self, rule: &'static str) {
        if self.rules.last().is_some_and(|(_, last)| *last == rule) {
            return;
        }
        self.rules.push((self.clause_count, rule));
    }

    pub(super) fn rules(&self) -> &[(usize, &'static str)] {
        &self.rules
    }

    pub(super) fn tru(&self) -> Lit {
        self.true_lit
    }

    pub(super) fn fals(&self) -> Lit {
        -self.true_lit
    }

    pub(super) fn is_false(&self, lit: Lit) -> bool {
        lit == -self.true_lit
    }

    pub(super) fn is_true(&self, lit: Lit) -> bool {
        lit == self.true_lit
    }

    /// Adds a clause, dropping constant-false literals and skipping clauses
    /// that already contain a constant-true literal.
    pub(super) fn clause(&mut self, lits: &[Lit]) {
        if lits.iter().any(|&lit| lit == self.true_lit) {
            return;
        }
        let start = self.literals.len();
        for &lit in lits {
            debug_assert!(lit != 0);
            if lit == -self.true_lit {
                continue;
            }
            self.literals.push(lit);
        }
        if self.literals.len() == start {
            // An empty clause makes the formula unsatisfiable; keep it explicit.
            self.literals.extend([self.true_lit, 0, -self.true_lit]);
            self.clause_count += 1;
        }
        self.literals.push(0);
        self.clause_count += 1;
    }

    pub(super) fn implies(&mut self, premise: &[Lit], conclusion: &[Lit]) {
        let clause = premise
            .iter()
            .map(|&lit| -lit)
            .chain(conclusion.iter().copied())
            .collect::<Vec<_>>();
        self.clause(&clause);
    }

    /// Returns a literal equivalent to the conjunction of `lits`.
    pub(super) fn and(&mut self, lits: &[Lit]) -> Lit {
        if lits.iter().any(|&lit| self.is_false(lit)) {
            return self.fals();
        }
        let lits = lits
            .iter()
            .copied()
            .filter(|&lit| !self.is_true(lit))
            .collect::<Vec<_>>();
        match lits.as_slice() {
            [] => self.tru(),
            [single] => *single,
            _ => {
                let out = self.new_var();
                for &lit in &lits {
                    self.clause(&[-out, lit]);
                }
                let mut clause = lits.iter().map(|&lit| -lit).collect::<Vec<_>>();
                clause.push(out);
                self.clause(&clause);
                out
            }
        }
    }

    /// Returns a literal equivalent to the disjunction of `lits`.
    pub(super) fn or(&mut self, lits: &[Lit]) -> Lit {
        let negated = lits.iter().map(|&lit| -lit).collect::<Vec<_>>();
        -self.and(&negated)
    }

    pub(super) fn at_most_one(&mut self, lits: &[Lit]) {
        let lits = lits
            .iter()
            .copied()
            .filter(|&lit| !self.is_false(lit))
            .collect::<Vec<_>>();
        if lits.len() <= 6 {
            for i in 0..lits.len() {
                for j in i + 1..lits.len() {
                    self.clause(&[-lits[i], -lits[j]]);
                }
            }
            return;
        }
        // Sequential counter (Sinz 2005).
        let prefix = (0..lits.len() - 1)
            .map(|_| self.new_var())
            .collect::<Vec<_>>();
        self.clause(&[-lits[0], prefix[0]]);
        for i in 1..lits.len() - 1 {
            self.clause(&[-lits[i], prefix[i]]);
            self.clause(&[-prefix[i - 1], prefix[i]]);
            self.clause(&[-lits[i], -prefix[i - 1]]);
        }
        self.clause(&[-lits[lits.len() - 1], -prefix[lits.len() - 2]]);
    }

    pub(super) fn exactly_one(&mut self, lits: &[Lit]) {
        self.clause(lits);
        self.at_most_one(lits);
    }
}
