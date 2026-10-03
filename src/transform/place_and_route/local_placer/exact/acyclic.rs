//! Lazy acyclicity: nogoods for unfounded power and torch feedback.
//!
//! In lazy mode the CNF only requires every powered element to have *some*
//! powered source, so the solver may propose wires that power each other in a
//! loop with no driver, or torches whose output reaches their own support.
//! After each model this module computes what is actually reachable from the
//! drivers and returns clauses that exclude exactly the offending structures:
//!
//! * unfounded sets: cells claimed powered in a case but unreachable from any
//!   driver get a loop formula ("if any of them is powered, some relation from
//!   outside the set must deliver power"), as in answer-set solving;
//! * feedback: a cycle of active relations through a torch's support and the
//!   torch itself is forbidden as a whole.
//!
//! Both clauses hold in every acyclic, physically consistent layout, so they
//! never remove a valid solution.

use std::collections::{HashMap, VecDeque};

use super::cnf::Lit;
use super::encode::{Encoding, TORCH_ATTACH};
use super::solver::SatSolver;

/// Allocates auxiliary variables beyond the encoded formula.
pub(super) struct AuxVars {
    next: Lit,
    supports: HashMap<(usize, usize), Lit>,
}

impl AuxVars {
    pub(super) fn new(encoding: &Encoding) -> Self {
        Self {
            next: encoding.cnf.num_vars() + 1,
            supports: HashMap::new(),
        }
    }

    /// Literal implying "relation `index` delivers power in `case`".
    fn support(
        &mut self,
        encoding: &Encoding,
        solver: &mut SatSolver,
        index: usize,
        case: usize,
    ) -> Lit {
        if let Some(&lit) = self.supports.get(&(index, case)) {
            return lit;
        }
        let lit = self.next;
        self.next += 1;
        let relation = &encoding.relations[index];
        solver.add_clause(&[-lit, relation.lit]);
        solver.add_clause(&[-lit, encoding.values[relation.source][case]]);
        self.supports.insert((index, case), lit);
        lit
    }

    fn fresh(&mut self) -> Lit {
        let lit = self.next;
        self.next += 1;
        lit
    }
}

pub(super) struct LazyIndex {
    incoming: Vec<Vec<usize>>,
    outgoing: Vec<Vec<usize>>,
}

impl LazyIndex {
    pub(super) fn new(encoding: &Encoding) -> Self {
        let mut incoming = vec![Vec::new(); encoding.geometry.len()];
        let mut outgoing = vec![Vec::new(); encoding.geometry.len()];
        for (index, relation) in encoding.relations.iter().enumerate() {
            incoming[relation.sink].push(index);
            outgoing[relation.source].push(index);
        }
        Self { incoming, outgoing }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub(super) struct LazyStats {
    pub(super) loop_formulas: usize,
    pub(super) feedback_cuts: usize,
}

/// Returns the number of clauses added; zero means the model is acyclic and
/// every powered cell is reachable from a driver in every case.
pub(super) fn refine(
    encoding: &Encoding,
    index: &LazyIndex,
    solver: &mut SatSolver,
    aux: &mut AuxVars,
    stats: &mut LazyStats,
) -> usize {
    let cells = encoding.geometry.len();
    let active = encoding
        .relations
        .iter()
        .map(|relation| solver.value(relation.lit))
        .collect::<Vec<_>>();
    let torch_of = (0..cells)
        .map(|cell| {
            TORCH_ATTACH
                .into_iter()
                .enumerate()
                .find(|(slot, _)| {
                    let lit = encoding.torch[cell][*slot];
                    !encoding.cnf.is_false(lit) && solver.value(lit)
                })
                .map(|(slot, attach)| {
                    (
                        encoding.torch[cell][slot],
                        encoding.geometry.step(cell, attach).unwrap(),
                    )
                })
        })
        .collect::<Vec<_>>();
    let is_switch = (0..cells)
        .map(|cell| solver.value(encoding.is_switch[cell]))
        .collect::<Vec<_>>();
    // Read the whole model before adding clauses: adding invalidates it.
    let powered_by_case = (0..encoding.cases)
        .map(|case| {
            (0..cells)
                .map(|cell| solver.value(encoding.values[cell][case]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut added = 0;
    for (case, powered) in powered_by_case.iter().enumerate() {
        // Least fixed point from drivers along active relations.
        let mut reached = vec![false; cells];
        let mut queue = VecDeque::new();
        for cell in 0..cells {
            if powered[cell] && (torch_of[cell].is_some() || is_switch[cell]) {
                reached[cell] = true;
                queue.push_back(cell);
            }
        }
        while let Some(cell) = queue.pop_front() {
            for &relation in &index.outgoing[cell] {
                if !active[relation] {
                    continue;
                }
                let sink = encoding.relations[relation].sink;
                if !reached[sink] {
                    reached[sink] = true;
                    queue.push_back(sink);
                }
            }
        }
        let unfounded = (0..cells)
            .filter(|&cell| powered[cell] && !reached[cell])
            .collect::<Vec<_>>();
        if unfounded.is_empty() {
            continue;
        }
        // One loop formula per connected group of unfounded cells.
        let mut group_of = vec![usize::MAX; cells];
        let mut groups = Vec::<Vec<usize>>::new();
        for &start in &unfounded {
            if group_of[start] != usize::MAX {
                continue;
            }
            let id = groups.len();
            let mut members = vec![start];
            group_of[start] = id;
            let mut cursor = 0;
            while cursor < members.len() {
                let cell = members[cursor];
                cursor += 1;
                let neighbors = index.outgoing[cell]
                    .iter()
                    .map(|&r| encoding.relations[r].sink)
                    .chain(index.incoming[cell].iter().map(|&r| encoding.relations[r].source))
                    .collect::<Vec<_>>();
                for neighbor in neighbors {
                    if powered[neighbor] && !reached[neighbor] && group_of[neighbor] == usize::MAX {
                        group_of[neighbor] = id;
                        members.push(neighbor);
                    }
                }
            }
            groups.push(members);
        }
        for (id, members) in groups.iter().enumerate() {
            let mut external = Vec::new();
            for &cell in members {
                for &relation in &index.incoming[cell] {
                    if group_of[encoding.relations[relation].source] != id {
                        external.push(aux.support(encoding, solver, relation, case));
                    }
                }
            }
            let any_support = aux.fresh();
            let mut clause = vec![-any_support];
            clause.extend(external.iter().copied());
            solver.add_clause(&clause);
            for &cell in members {
                solver.add_clause(&[-encoding.values[cell][case], any_support]);
            }
            stats.loop_formulas += 1;
            added += 1;
        }
    }
    if added > 0 {
        return added;
    }

    // Feedback: a path of active relations from a torch back to its support.
    for (torch_cell, torch) in torch_of.iter().enumerate() {
        let Some((torch_lit, support)) = *torch else {
            continue;
        };
        let mut previous = vec![usize::MAX; cells];
        let mut via = vec![usize::MAX; cells];
        let mut queue = VecDeque::from([torch_cell]);
        previous[torch_cell] = torch_cell;
        let mut found = false;
        'search: while let Some(cell) = queue.pop_front() {
            let mut steps = index.outgoing[cell]
                .iter()
                .filter(|&&relation| active[relation])
                .map(|&relation| (encoding.relations[relation].sink, Some(relation)))
                .collect::<Vec<_>>();
            // Passing through another torch: its support powers it.
            for (other, other_torch) in torch_of.iter().enumerate() {
                if let Some((_, other_support)) = other_torch {
                    if *other_support == cell {
                        steps.push((other, None));
                    }
                }
            }
            for (next, relation) in steps {
                if previous[next] != usize::MAX {
                    continue;
                }
                previous[next] = cell;
                via[next] = relation.unwrap_or(usize::MAX);
                if next == support {
                    found = true;
                    break 'search;
                }
                queue.push_back(next);
            }
        }
        if !found {
            continue;
        }
        let mut clause = vec![-torch_lit];
        let mut cell = support;
        while cell != torch_cell {
            if via[cell] == usize::MAX {
                // Entered `cell` as a torch from its support.
                let (lit, _) = torch_of[cell].unwrap();
                clause.push(-lit);
            } else {
                clause.push(-encoding.relations[via[cell]].lit);
            }
            cell = previous[cell];
        }
        solver.add_clause(&clause);
        stats.feedback_cuts += 1;
        added += 1;
    }
    added
}
