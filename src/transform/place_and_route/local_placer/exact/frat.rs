//! Reads a CaDiCaL FRAT proof (`SatSolver::trace_frat`) back: which original
//! clauses a refutation rests on.
//!
//! Lines are `o <id> <lits> 0` for an original clause (also those added
//! between solves), `a <id> <lits> 0 l <ids> 0` for a derived clause and the
//! clauses it was derived from, and `d`/`f` for deletions and the clauses
//! left at the end.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use eyre::{bail, eyre};

use super::cnf::Lit;

/// The original clauses the proof's empty clause is derived from, following
/// each derived clause back to the clauses it cites.
pub(super) fn refutation_originals(path: &Path) -> eyre::Result<Vec<Vec<Lit>>> {
    let file = std::fs::File::open(path)
        .map_err(|error| eyre!("cannot read the proof {}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    // Ranges into one flat buffer each, to keep large proofs compact.
    let mut original_lits = Vec::new();
    let mut originals: HashMap<u64, (usize, usize)> = HashMap::new();
    let mut hints = Vec::new();
    let mut derived: HashMap<u64, (usize, usize)> = HashMap::new();
    let mut empty = None;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let mut words = line.split_ascii_whitespace();
        let (Some(tag), Some(id)) = (words.next(), words.next()) else {
            continue;
        };
        let id: u64 = id.parse()?;
        match tag {
            "o" => {
                let start = original_lits.len();
                for word in words.by_ref() {
                    let lit: Lit = word.parse()?;
                    if lit == 0 {
                        break;
                    }
                    original_lits.push(lit);
                }
                originals.insert(id, (start, original_lits.len()));
            }
            "a" => {
                let mut size = 0;
                for word in words.by_ref() {
                    if word == "0" {
                        break;
                    }
                    size += 1;
                }
                if words.next() != Some("l") {
                    bail!("derived clause {id} has no antecedents; trace with frat=1");
                }
                let start = hints.len();
                for word in words.by_ref() {
                    let hint: i64 = word.parse()?;
                    if hint == 0 {
                        break;
                    }
                    hints.push(hint.unsigned_abs());
                }
                derived.insert(id, (start, hints.len()));
                if size == 0 {
                    empty = Some(id);
                }
            }
            _ => {}
        }
    }
    let empty = empty
        .or_else(|| {
            originals
                .iter()
                .find(|(_, &(start, end))| start == end)
                .map(|(&id, _)| id)
        })
        .ok_or_else(|| eyre!("the proof derives no empty clause"))?;
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![empty];
    let mut used = Vec::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(&(start, end)) = derived.get(&id) {
            stack.extend(&hints[start..end]);
        } else if let Some(&(start, end)) = originals.get(&id) {
            used.push(original_lits[start..end].to_vec());
        } else {
            bail!("the proof cites clause {id}, which it never introduced");
        }
    }
    Ok(used)
}
