use std::collections::HashSet;

use eyre::Result;

use crate::solver::and_macro::AndMacro;
use crate::solver::types::{Cell, Dir, Net, in_bounds};

/// 두 셀 사이의 방향 계산
pub fn dir_between(a: Cell, b: Cell) -> Option<Dir> {
    if a.x + 1 == b.x && a.z == b.z {
        return Some(Dir::E);
    }
    if a.x == b.x + 1 && a.z == b.z {
        return Some(Dir::W);
    }
    if a.x == b.x && a.z + 1 == b.z {
        return Some(Dir::S);
    }
    if a.x == b.x && a.z == b.z + 1 {
        return Some(Dir::N);
    }
    None
}

/// 솔루션에서 경로 재구성
/// 
/// SCIP 솔루션의 흐름 변수(f)를 기반으로 시작점에서 목표점까지의 경로를 재구성합니다.
pub fn reconstruct_path(
    net: Net,
    start: Cell,
    goal: Cell,
    xs: usize,
    zs: usize,
    sol: &crate::solver::types::Solution,
) -> Result<Vec<Cell>> {
    let mut path = vec![start];
    let mut cur = start;

    let cap = xs * zs + 10;
    for _ in 0..cap {
        if cur == goal {
            return Ok(path);
        }
        let mut next: Option<Cell> = None;
        for d in Dir::all() {
            let name = format!("f_{}_{}_{}_{}", net.name(), cur.x, cur.z, d.idx());
            if sol.get_bin(&name) {
                let (dx, dz) = d.delta();
                let nx = cur.x as i32 + dx;
                let nz = cur.z as i32 + dz;
                if !in_bounds(nx, nz, xs, zs) {
                    eyre::bail!("path goes out of bounds from {:?} via {:?}", cur, d);
                }
                next = Some(Cell {
                    x: nx as usize,
                    z: nz as usize,
                });
                break;
            }
        }
        let Some(nxt) = next else {
            eyre::bail!("no outgoing edge for {:?} on net {:?}", cur, net);
        };
        path.push(nxt);
        cur = nxt;
    }

    eyre::bail!("path reconstruction exceeded cap; likely cycle or missing edges")
}

/// 경로를 "타일"로 배치 (더스트 + 자동 리피터)
/// 
/// 경로의 각 셀에 레드스톤 더스트를 배치하고,
/// 긴 직선 구간에는 자동으로 리피터를 삽입합니다.
pub fn place_path_tiles(
    path: &[Cell],
    ax: usize,
    az: usize,
    terminals: &HashSet<Cell>,
    max_dust_run: usize,
    ox: i32,
    oy: i32,
    oz: i32,
    vox: &mut std::collections::HashMap<(i32, i32, i32), String>,
    repeater_marks: &mut HashSet<Cell>,
) -> Result<()> {
    if path.is_empty() {
        return Ok(());
    }

    // 1) 모든 경로 셀에 더스트 배치 (매크로 토치 셀은 제외)
    for &c in path {
        if AndMacro::is_torch_cell(c, ax, az) {
            continue;
        }
        let base = (ox + c.x as i32, oy, oz + c.z as i32);
        vox.entry(base)
            .or_insert_with(|| "minecraft:stone".to_string());
        vox.insert(
            (ox + c.x as i32, oy + 1, oz + c.z as i32),
            "minecraft:redstone_wire".to_string(),
        );
    }

    // 2) 긴 직선 구간에 자동 리피터 삽입
    if max_dust_run == 0 || path.len() < 3 {
        return Ok(());
    }

    let mut run_len: usize = 0;
    for i in 1..(path.len() - 1) {
        let prev = path[i - 1];
        let cur = path[i];
        let next = path[i + 1];

        if terminals.contains(&cur) {
            run_len = 0;
            continue;
        }
        // 매크로 점유 셀 내부에는 리피터를 배치하지 않음
        if AndMacro::is_internal_occupied_cell(cur, ax, az) {
            run_len = 0;
            continue;
        }

        let d1 = dir_between(prev, cur).ok_or_else(|| eyre::eyre!("bad path segment"))?;
        let d2 = dir_between(cur, next).ok_or_else(|| eyre::eyre!("bad path segment"))?;

        if d1 == d2 {
            run_len += 1;
            if run_len >= max_dust_run {
                let base = (ox + cur.x as i32, oy, oz + cur.z as i32);
                vox.entry(base)
                    .or_insert_with(|| "minecraft:stone".to_string());
                vox.insert(
                    (ox + cur.x as i32, oy + 1, oz + cur.z as i32),
                    format!("minecraft:repeater[facing={}]", d2.facing()),
                );
                repeater_marks.insert(cur);
                run_len = 0;
            }
        } else {
            run_len = 0;
        }
    }

    Ok(())
}

