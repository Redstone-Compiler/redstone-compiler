use std::fs;
use std::path::Path;
use std::process::Command;

use eyre::{Context, Result};

use crate::solver::types::Solution;

/// SCIP 실행 및 솔루션 파싱
/// 
/// SCIP 솔버를 실행하여 LP 파일을 해결하고 솔루션 파일을 생성합니다.

/// SCIP 솔버 실행
/// 
/// SCIP를 실행하여 LP 파일을 읽고 최적화한 후 솔루션을 파일로 저장합니다.
pub fn run_scip(scip_path: &str, lp_path: &Path, sol_path: &Path) -> Result<()> {
    let lpq = format!("\"{}\"", lp_path.display());
    let solq = format!("\"{}\"", sol_path.display());
    let cmd = format!("read {} optimize write solution {} quit", lpq, solq);

    let status = Command::new(scip_path)
        .arg("-c")
        .arg(cmd)
        .status()
        .with_context(|| format!("failed to run scip: {}", scip_path))?;

    if !status.success() {
        eyre::bail!("SCIP failed: {:?}", status);
    }
    Ok(())
}

/// SCIP 솔루션 파일 파싱
/// 
/// SCIP가 생성한 솔루션 파일을 읽어서 변수 값들을 파싱합니다.
pub fn parse_scip_sol(path: &Path) -> Result<Solution> {
    let txt = fs::read_to_string(path).with_context(|| format!("read sol: {}", path.display()))?;
    let mut vals = std::collections::HashMap::new();
    for raw in txt.lines() {
        let raw = raw.trim();
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        if raw.starts_with("solution status")
            || raw.starts_with("no solution")
            || raw.starts_with("objective value")
        {
            continue;
        }
        let main = raw.split('(').next().unwrap().trim();
        let mut it = main.split_whitespace();
        let Some(name) = it.next() else { continue };
        let Some(val_s) = it.next() else { continue };
        if let Ok(v) = val_s.parse::<f64>() {
            vals.insert(name.to_string(), v);
        }
    }
    Ok(Solution { vals })
}

