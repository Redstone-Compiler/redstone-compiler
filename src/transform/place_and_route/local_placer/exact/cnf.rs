//! The grounded formula handed to CaDiCaL: DIMACS-style signed literals.

pub(super) type Lit = i32;

#[derive(Debug, Clone)]
pub(super) struct Cnf {
    num_vars: i32,
    /// Zero-terminated clauses, flattened.
    literals: Vec<Lit>,
    clause_count: usize,
    true_lit: Lit,
}

impl Cnf {
    /// Wraps an already built formula whose variable 1 is constant true.
    pub(super) fn from_literals(num_vars: i32, literals: Vec<Lit>, clause_count: usize) -> Self {
        Self {
            num_vars,
            literals,
            clause_count,
            true_lit: 1,
        }
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

    pub(super) fn is_false(&self, lit: Lit) -> bool {
        lit == -self.true_lit
    }
}
