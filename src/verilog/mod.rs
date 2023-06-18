use std::{collections::HashMap, path::PathBuf};

use sv_parser::{parse_sv, SyntaxTree};

pub fn load(path: &PathBuf) -> eyre::Result<SyntaxTree> {
    let defines = HashMap::new();
    let includes: Vec<PathBuf> = Vec::new();

    let result = parse_sv(path, &defines, &includes, false, false);
    let Ok((syntax_tree, _)) = result else {
        eyre::bail!("System-verilog input parse err!");
    };

    Ok(syntax_tree)
}
