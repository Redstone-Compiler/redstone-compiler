//! DIMACS export with explanations, to show what the SAT solver receives.
//!
//! The solver only sees numbered variables and clauses. Optionally the same
//! formula gets comment lines (`c ...`, ignored by solvers): a legend for the
//! named variables, a header for each modeling rule, and every clause read
//! back as a sentence. rsdsl writes them from the grounded model.

use std::path::Path;

use super::encode::Encoding;
use super::{ExactLocalPlacer, ExactPlacerConfig};

/// How much explanation `write_dimacs` adds as comment lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DimacsComments {
    /// Plain DIMACS, exactly as the solver receives it.
    #[default]
    None,
    /// A legend for the variables that have a direct meaning.
    Legend,
    /// The legend, a header per modeling rule, and a sentence per clause.
    Explained,
}

impl ExactLocalPlacer {
    /// Writes the CNF that `place` would hand to CaDiCaL as DIMACS, with the
    /// requested amount of explanation in comment lines.
    pub fn write_dimacs(
        &self,
        config: &ExactPlacerConfig,
        path: impl AsRef<Path>,
        comments: DimacsComments,
    ) -> eyre::Result<()> {
        let encoding =
            Encoding::build_dsl_with(&self.netlist, config, comments == DimacsComments::Explained)?;
        let program = encoding
            .program
            .as_ref()
            .expect("rsdsl encodings keep their program");
        let header = [
            format!(
                "CaDiCaL input (DIMACS CNF) for `{}` in a {:?} box.",
                self.name, config.dim
            ),
            format!(
                "Input cases: case k sets input i to bit i of k, inputs = {:?} ({} cases).",
                self.netlist.input_names(),
                encoding.cases
            ),
            "Model: exact_placer.rsdsl; [규칙] headers name its rules.".to_owned(),
        ];
        let comments = match comments {
            DimacsComments::None => rsdsl::Comments::None,
            DimacsComments::Legend => rsdsl::Comments::Legend,
            DimacsComments::Explained => rsdsl::Comments::Explained,
        };
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        program.write_dimacs(&mut out, comments, &header)?;
        std::io::Write::flush(&mut out)?;
        Ok(())
    }
}
