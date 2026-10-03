//! DIMACS export with explanations, to show what the SAT solver receives.
//!
//! The solver only sees numbered variables and clauses. Optionally the same
//! formula gets comment lines (`c ...`, ignored by solvers): a legend for the
//! named variables, a header for each modeling rule, and every clause read
//! back as a sentence ("if A and B then C or D").

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use super::cnf::Lit;
use super::encode::{Encoding, SinkKind, SourceKind, CARDINALS, TORCH_ATTACH};
use super::netlist::NetDriver;
use super::{ExactLocalPlacer, ExactPlacerConfig};
use crate::world::block::Direction;

fn axis(direction: Direction) -> &'static str {
    match direction {
        Direction::East => "x+",
        Direction::West => "x-",
        Direction::North => "y+",
        Direction::South => "y-",
        Direction::Top => "z+",
        Direction::Bottom => "z-",
        Direction::None => "none",
    }
}

/// Names of the variables that have a direct meaning, by variable number.
fn variable_names(encoding: &Encoding, netlist: &super::NorNetlist) -> HashMap<Lit, Vec<String>> {
    let mut names = HashMap::<Lit, Vec<String>>::new();
    let mut name = |lit: Lit, text: String| {
        if !encoding.cnf.is_false(lit) && !encoding.cnf.is_true(lit) {
            names.entry(lit.abs()).or_default().push(text);
        }
    };
    for cell in 0..encoding.geometry.len() {
        let p = encoding.geometry.position(cell);
        let at = format!("({},{},{})", p.0, p.1, p.2);
        name(-encoding.air[cell], format!("{at} 차 있음"));
        name(encoding.solid[cell], format!("{at} 블록"));
        name(encoding.dust[cell], format!("{at} 가루"));
        for (index, attach) in TORCH_ATTACH.into_iter().enumerate() {
            name(
                encoding.torch[cell][index],
                format!("{at} 토치({}에 붙음)", axis(attach)),
            );
        }
        for (index, direction) in CARDINALS.into_iter().enumerate() {
            name(
                encoding.repeater[cell][index],
                format!("{at} 리피터({}에서 입력)", axis(direction)),
            );
        }
        for site in encoding.switches.iter().filter(|site| site.cell == cell) {
            let input = match &netlist.nets[site.net].driver {
                NetDriver::Input(input) => input.as_str(),
                NetDriver::Gate => "?",
            };
            name(
                site.lit,
                format!("{at} {input} 스위치({}에 붙음)", axis(site.attach)),
            );
        }
        for (class, &lit) in encoding.class_lits[cell].iter().enumerate() {
            name(lit, format!("{at} 신호={}", encoding.classes[class].name));
        }
        for (case, &lit) in encoding.values[cell].iter().enumerate() {
            name(lit, format!("{at} 경우{case}에 켜짐"));
        }
        for (index, direction) in CARDINALS.into_iter().enumerate() {
            name(
                encoding.conn[cell][index],
                format!("{at} 가루가 {}쪽과 연결", axis(direction)),
            );
            name(
                encoding.points[cell][index],
                format!("{at} 가루가 {}쪽을 가리킴", axis(direction)),
            );
        }
        name(encoding.hard[cell], format!("{at} 강전원 받는 블록"));
        for (level, &lit) in encoding.ranks.get(cell).into_iter().flatten().enumerate() {
            name(lit, format!("{at} 순위 ≥ {}", level + 1));
        }
        for (level, &lit) in encoding.stages.get(cell).into_iter().flatten().enumerate() {
            name(lit, format!("{at} 단계 ≥ {}", level + 1));
        }
    }
    let kind = |source: SourceKind| match source {
        SourceKind::Dust => "가루",
        SourceKind::Torch => "토치",
        SourceKind::Repeater => "리피터",
        SourceKind::Solid => "블록",
        SourceKind::Switch => "스위치",
    };
    let sink = |sink: SinkKind| match sink {
        SinkKind::Dust => "가루",
        SinkKind::Repeater => "리피터",
        SinkKind::Solid => "블록",
    };
    for relation in &encoding.relations {
        let from = encoding.geometry.position(relation.source);
        let to = encoding.geometry.position(relation.sink);
        name(
            relation.lit,
            format!(
                "전원관계[({},{},{}) {} → ({},{},{}) {}]",
                from.0,
                from.1,
                from.2,
                kind(relation.source_kind),
                to.0,
                to.1,
                to.2,
                sink(relation.sink_kind)
            ),
        );
    }
    for site in &encoding.output_sites {
        let p = encoding.geometry.position(site.cell);
        name(
            site.lit,
            format!("출력 {}을 ({},{},{})에서 관측", site.name, p.0, p.1, p.2),
        );
    }
    names
}

fn describe(names: &HashMap<Lit, Vec<String>>, true_lit: Lit, lit: Lit) -> String {
    if lit.abs() == true_lit {
        return "상수 참".to_owned();
    }
    match names.get(&lit.abs()) {
        Some(texts) => texts.join(" = "),
        None => format!("보조#{}", lit.abs()),
    }
}

/// Reads a clause back as "if all premises hold, at least one conclusion does".
fn explain(names: &HashMap<Lit, Vec<String>>, true_lit: Lit, clause: &[Lit]) -> String {
    let premises = clause
        .iter()
        .filter(|&&lit| lit < 0)
        .map(|&lit| describe(names, true_lit, lit))
        .collect::<Vec<_>>();
    let conclusions = clause
        .iter()
        .filter(|&&lit| lit > 0)
        .map(|&lit| describe(names, true_lit, lit))
        .collect::<Vec<_>>();
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
        if !config.legacy_encoder {
            let encoding = Encoding::build_dsl_with(
                &self.netlist,
                config,
                comments == DimacsComments::Explained,
            )?;
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
            return Ok(());
        }
        let encoding = Encoding::build(&self.netlist, config)?;
        let names = if comments == DimacsComments::None {
            HashMap::new()
        } else {
            variable_names(&encoding, &self.netlist)
        };
        let true_lit = encoding.cnf.tru();
        let mut text = String::new();
        if comments != DimacsComments::None {
            let cases = encoding.cases;
            let inputs = self.netlist.input_names();
            writeln!(
                text,
                "c CaDiCaL input (DIMACS CNF) for `{}` in a {:?} box.",
                self.name, config.dim
            )?;
            writeln!(
                text,
                "c Lines starting with `c` are comments; solvers ignore them."
            )?;
            writeln!(
                text,
                "c A clause line lists literals ending in 0 and requires at least one to hold;"
            )?;
            writeln!(text, "c -N means \"variable N is false\".")?;
            writeln!(
                text,
                "c Input cases: case k sets input i to bit i of k, inputs = {inputs:?} ({cases} cases)."
            )?;
            writeln!(text, "c")?;
            writeln!(
                text,
                "c Variable legend (unlisted variables are auxiliary):"
            )?;
            let mut named = names.iter().collect::<Vec<_>>();
            named.sort();
            for (var, texts) in named {
                writeln!(text, "c   {var} = {}", texts.join(" = "))?;
            }
            writeln!(text, "c")?;
        }
        writeln!(
            text,
            "p cnf {} {}",
            encoding.cnf.num_vars(),
            encoding.cnf.clause_count()
        )?;
        let explained = comments == DimacsComments::Explained;
        let rules = encoding.cnf.rules();
        let mut next_rule = 0;
        let mut clause = Vec::new();
        let mut index = 0;
        for &lit in encoding.cnf.literals() {
            if lit != 0 {
                clause.push(lit);
                continue;
            }
            while explained && next_rule < rules.len() && rules[next_rule].0 <= index {
                writeln!(text, "c")?;
                writeln!(text, "c [규칙] {}", rules[next_rule].1)?;
                next_rule += 1;
            }
            let numbers = clause.iter().map(|lit| lit.to_string()).collect::<Vec<_>>();
            writeln!(text, "{} 0", numbers.join(" "))?;
            if explained {
                writeln!(text, "c   {}", explain(&names, true_lit, &clause))?;
            }
            clause.clear();
            index += 1;
        }
        std::fs::write(path, text)?;
        Ok(())
    }
}
