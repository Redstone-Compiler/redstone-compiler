//! Grounds a model against an instance file and writes DIMACS.
//!
//! `cargo run --release -p rsdsl --example ground -- MODEL INSTANCE OUT [none|legend|explained]`

use std::io::BufWriter;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: ground MODEL INSTANCE OUT [none|legend|explained]");
        std::process::exit(2);
    }
    let comments = match args.get(4).map(String::as_str) {
        Some("legend") => rsdsl::Comments::Legend,
        Some("explained") => rsdsl::Comments::Explained,
        _ => rsdsl::Comments::None,
    };
    let model_text = std::fs::read_to_string(&args[1]).expect("read model");
    let instance_text = std::fs::read_to_string(&args[2]).expect("read instance");
    let started = std::time::Instant::now();
    let model = rsdsl::Model::parse(&args[1], &model_text).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let options = rsdsl::GroundOptions {
        provenance: comments == rsdsl::Comments::Explained,
        ..Default::default()
    };
    let program = model
        .ground_file(&args[2], &instance_text, options)
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    let elapsed = started.elapsed();
    if !program.warnings().is_empty() {
        eprintln!("{}", program.render_warnings());
    }
    eprintln!(
        "{} vars, {} clauses in {:.1} ms",
        program.num_vars(),
        program.clause_count(),
        elapsed.as_secs_f64() * 1e3
    );
    for (rule, clauses) in program.rule_clause_counts() {
        eprintln!("  {clauses:>7}  {rule}");
    }
    let file = std::fs::File::create(&args[3]).expect("create output");
    let mut out = BufWriter::new(file);
    program
        .write_dimacs(
            &mut out,
            comments,
            &[format!("{} grounded with {}", args[2], args[1])],
        )
        .expect("write");
}
