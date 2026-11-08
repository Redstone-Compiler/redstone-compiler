//! Export CNF and its exact Kind-choice literals for Python boundary experiments.
//! Usage: boundary_export MODEL INSTANCE PREFIX X Y Z
use std::fs::File;
use std::io::{BufWriter, Write};

use rsdsl::{Comments, GroundOptions, IValue, Model};
fn label(v: &IValue) -> String {
    match v {
        IValue::Member(n, args) => {
            if args.is_empty() {
                n.clone()
            } else {
                format!(
                    "{}({})",
                    n,
                    args.iter().map(label).collect::<Vec<_>>().join(",")
                )
            }
        }
        IValue::Sym(n) => n.clone(),
        _ => panic!("unexpected choice value: {v:?}"),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: boundary_export MODEL INSTANCE PREFIX X Y Z".into());
    }
    let model = Model::parse(&a[1], &std::fs::read_to_string(&a[1])?)?;
    let program = model.ground_file(
        &a[2],
        &std::fs::read_to_string(&a[2])?,
        GroundOptions::default(),
    )?;
    let mut cnf = BufWriter::new(File::create(format!("{}.cnf", a[3]))?);
    program.write_dimacs(&mut cnf, Comments::None, &[])?;
    let mut symbols = BufWriter::new(File::create(format!("{}.tsv", a[3]))?);
    let (nx, ny, nz): (usize, usize, usize) = (a[4].parse()?, a[5].parse()?, a[6].parse()?);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                for (value, lit) in program
                    .choice_options("Kind", &[IValue::cell(x, y, z)])
                    .ok_or("missing Kind")?
                {
                    writeln!(symbols, "{x}\t{y}\t{z}\t{}\t{lit}", label(&value))?;
                }
            }
        }
    }
    Ok(())
}
