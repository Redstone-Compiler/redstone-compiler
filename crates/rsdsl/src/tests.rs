use crate::parser::parse;

const EXACT_PLACER: &str =
    include_str!("../../../src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl");
const INVERTER: &str = include_str!("../../../docs/rsdsl/inverter_1x3x2.instance.rsdsl");

#[test]
fn parses_the_reference_examples() {
    let model = parse(0, EXACT_PLACER).unwrap_or_else(|d| panic!("{d}"));
    assert!(model.items.len() > 50);
    let instance = parse(1, INVERTER).unwrap_or_else(|d| panic!("{d}"));
    assert!(matches!(
        instance.header,
        crate::ast::Header::Instance { .. }
    ));
}

#[test]
fn rejects_chained_comparisons_and_equivalences() {
    for source in ["a < b < c", "a <-> b <-> c"] {
        let text = format!("rsdsl 2; model m; rule \"x\" {{ require {source}; }}");
        assert!(parse(0, &text).is_err(), "{source}");
    }
}

use std::collections::BTreeSet;

use crate::{GroundOptions, IValue, Lit, Model, Program};

const TINY: &str = r#"
rsdsl 2;
model tiny;

grid Cell(x, y, z) dirs Dir6 {
    East = +x, West = -x, North = +y, South = -y, Up = +z, Down = -z,
}
enum Color { Red, Green }

@prefer(Empty) @outside(Empty)
choice Paint[c: Cell] { Empty, Dot(k: Color) }

def Painted[c: Cell] := Paint[c] is Dot(_);

relation Adj(a: Cell, b: Cell);
var Use over Adj;

rule "neighbors differ" {
    forall c: Cell, k: Color {
        require Paint[c] is Dot(k) -> not Paint[step(c, East)] is Dot(k);
    }
}

rule "adjacent painted pairs" {
    forall c: Cell {
        Adj(c, step(c, East)) |= Painted[c] and Painted[step(c, East)];
    }
}

rule "some pair is used" {
    forall (a, b) in Adj {
        require Use(a, b) -> Adj(a, b);
    }
    require any(Use(a, b) for (a, b) in Adj);
}

rule "at most two painted" {
    require count(Painted[c] for c: Cell) <= 2;
}
"#;

fn clauses(program: &Program) -> Vec<Vec<Lit>> {
    program
        .literals()
        .split(|&lit| lit == 0)
        .filter(|clause| !clause.is_empty())
        .map(<[Lit]>::to_vec)
        .collect()
}

/// Every satisfying assignment, projected onto `project`.
fn projected_models(program: &Program, project: &[Lit]) -> BTreeSet<Vec<bool>> {
    let n = program.num_vars() as usize;
    assert!(n <= 24, "{n} variables are too many to enumerate");
    let clauses = clauses(program);
    let mut out = BTreeSet::new();
    for mask in 0u64..(1 << n) {
        let value = |lit: Lit| (mask >> (lit.unsigned_abs() - 1) & 1 == 1) == (lit > 0);
        if clauses
            .iter()
            .all(|clause| clause.iter().any(|&lit| value(lit)))
        {
            out.insert(project.iter().map(|&lit| value(lit)).collect());
        }
    }
    out
}

#[test]
fn grounded_cnf_has_exactly_the_intended_models() {
    let model = Model::parse("tiny.rsdsl", TINY).unwrap_or_else(|e| panic!("{e}"));
    let mut instance = crate::Instance::new("three");
    instance.grid("Cell", (3, 1, 1));
    let program = model
        .ground(&instance, GroundOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let members = [
        ("Empty", vec![]),
        ("Dot", vec![IValue::sym("Red")]),
        ("Dot", vec![IValue::sym("Green")]),
    ];
    let mut project = Vec::new();
    for x in 0..3 {
        for (member, payload) in &members {
            project.push(
                program
                    .option_lit("Paint", &[IValue::cell(x, 0, 0)], member, payload)
                    .unwrap(),
            );
        }
    }
    let actual = projected_models(&program, &project);

    // Direct enumeration: 0 = empty, 1 = red, 2 = green.
    let mut expected = BTreeSet::new();
    for code in 0..27usize {
        let paint = [code % 3, code / 3 % 3, code / 9];
        let differ = (0..2).all(|i| paint[i] == 0 || paint[i] != paint[i + 1]);
        let pairs = (0..2)
            .filter(|&i| paint[i] != 0 && paint[i + 1] != 0)
            .count();
        let painted = paint.iter().filter(|&&p| p != 0).count();
        if differ && pairs >= 1 && painted <= 2 {
            expected.insert(
                paint
                    .iter()
                    .flat_map(|&p| (0..3).map(move |m| m == p))
                    .collect::<Vec<_>>(),
            );
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn integer_order_encoding_counts_strict_chains() {
    let source = r#"
        rsdsl 2;
        model chain;
        grid Cell(x, y, z) dirs Dir6 {
            East = +x, West = -x, North = +y, South = -y, Up = +z, Down = -z,
        }
        param top: int = 3;
        int Level[c: Cell] in 0..=top;
        rule "rise" {
            forall c: Cell where has(step(c, East)) {
                require Level[c] < Level[step(c, East)];
            }
        }
        rule "end" {
            require Level[Cell(2, 0, 0)] <= 2 or Level[Cell(0, 0, 0)] == 1;
        }
    "#;
    let model = Model::parse("chain.rsdsl", source).unwrap_or_else(|e| panic!("{e}"));
    let mut instance = crate::Instance::new("three");
    instance.grid("Cell", (3, 1, 1));
    let program = model
        .ground(&instance, GroundOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let mut project = Vec::new();
    for x in 0..3 {
        project.extend(program.int_lits("Level", &[IValue::cell(x, 0, 0)]).unwrap());
    }
    let actual = projected_models(&program, &project);
    let mut expected = BTreeSet::new();
    for code in 0..64usize {
        let level = [code % 4, code / 4 % 4, code / 16];
        if level[0] < level[1] && level[1] < level[2] && (level[2] <= 2 || level[0] == 1) {
            expected.insert(
                level
                    .iter()
                    .flat_map(|&v| (1..=3).map(move |t| v >= t))
                    .collect::<Vec<_>>(),
            );
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(expected.len(), 2); // (0,1,2) and (1,2,3)
}

#[test]
fn reports_stage_and_name_errors() {
    let header = "rsdsl 2; model m; grid Cell(x, y, z) dirs Dir6 { East = +x, West = -x, North = +y, South = -y, Up = +z, Down = -z, } choice K[c: Cell] { A, B }";
    let cases = [
        (
            "rule \"r\" { forall c: Cell where K[c] is A { require K[c] is B; } }",
            "E0304",
        ),
        (
            "rule \"r\" { forall c: Cell { require K[c] is Nope; } }",
            "E0451",
        ),
        ("rule \"r\" { require undefined_name; }", "E0210"),
        (
            "rule \"r\" { forall c: Cell { require K[step(c, East)] is A; } }",
            "E0421",
        ),
    ];
    for (rule, code) in cases {
        let text = format!("{header} {rule}");
        let model = Model::parse("m.rsdsl", &text).unwrap_or_else(|e| panic!("{e}"));
        let mut instance = crate::Instance::new("i");
        instance.grid("Cell", (2, 1, 1));
        let error = match model.ground(&instance, GroundOptions::default()) {
            Ok(_) => panic!("{rule} should fail with {code}"),
            Err(error) => error,
        };
        assert_eq!(error.diagnostics[0].code, code, "{}", error.render());
        assert!(
            error.render().contains("--> m.rsdsl:1:"),
            "{}",
            error.render()
        );
    }
}

#[test]
fn grounds_the_reference_instance() {
    let model = Model::parse("exact_placer.rsdsl", EXACT_PLACER).unwrap_or_else(|e| panic!("{e}"));
    let program = model
        .ground_file("inverter.rsdsl", INVERTER, GroundOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    let floor = IValue::cell(0, 0, 0);
    assert!(program
        .option_lit("Kind", std::slice::from_ref(&floor), "Solid", &[])
        .is_some());
    assert!(program
        .option_lit("Kind", std::slice::from_ref(&floor), "Dust", &[])
        .is_none());
    assert!(program.relation("Feeds").unwrap().len() > 10);
    let mut text = Vec::new();
    program
        .write_dimacs(&mut text, crate::Comments::Legend, &[])
        .unwrap();
    let text = String::from_utf8(text).unwrap();
    assert!(text.contains("(0,0,1) Dust"), "{text}");
}
