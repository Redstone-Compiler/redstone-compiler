//! Place and Route 솔버 모듈
//!
//! 이 모듈은 선형 계획법(LP)을 사용하여 마인크래프트 레드스톤 회로의
//! 배치 및 라우팅을 최적화합니다.

mod and_macro;
mod constraints;
mod lp;
mod new;
mod new2;
mod path;
mod placement;
mod scip;
mod types;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use eyre::Result;

use self::and_macro::AndMacro;
use self::constraints::add_all_constraints;
use self::lp::Lp;
use self::path::{place_path_tiles, reconstruct_path};
use self::placement::placements_to_world3d;
use self::scip::{parse_scip_sol, run_scip};
use self::types::{in_bounds, v_anchor, v_f, v_w, Args, Cell, Dir, Net, Placement};
use crate::nbt::NBTRoot;

/// Place and Route 솔버 실행
///
/// 주어진 인자에 따라 LP 모델을 생성하고, SCIP로 해결한 후,
/// 결과를 마인크래프트 블록 배치로 변환하여 NBT 파일로 저장합니다.
pub fn run_solver(args: Args) -> Result<()> {
    let out_dir = PathBuf::from(&args.out_dir);
    fs::create_dir_all(&out_dir)?;

    let xs = args.x;
    let zs = args.z;
    if xs < 9 || zs < 7 {
        eyre::bail!("데모 AND(매크로+라우팅) 안정적으로 돌리려면 x>=9, z>=7 추천");
    }

    let midz = zs / 2;

    // 입력 터미널 (왼쪽 가장자리)
    let a_src = Cell {
        x: 0,
        z: midz.saturating_sub(1),
    };
    let b_src = Cell {
        x: 0,
        z: (midz + 1).min(zs - 1),
    };

    // 출력 터미널 (오른쪽 가장자리)
    let y_sink = Cell { x: xs - 1, z: midz };

    // AND 앵커 도메인 계산
    let (ax_min, ax_max, az_min, az_max) = AndMacro::anchor_domain(xs, zs);

    let mut lp = Lp::new();

    // 변수: 앵커들
    let mut anchors = Vec::new();
    for ax in ax_min..=ax_max {
        for az in az_min..=az_max {
            let (a_sink, b_sink, y_src) = AndMacro::pins(ax, az);

            // 터미널과 충돌 방지
            for &t in &[a_src, b_src, y_sink] {
                if a_sink == t || b_sink == t || y_src == t {
                    continue;
                }
            }
            if a_sink == b_sink || a_sink == y_src || b_sink == y_src {
                continue;
            }

            // 내부 셀이 터미널과 겹치지 않도록 확인
            let mut bad = false;
            for &(dx, dz) in AndMacro::internal_occupied() {
                let x = ax as i32 + dx;
                let z = az as i32 + dz;
                if !in_bounds(x, z, xs, zs) {
                    bad = true;
                    break;
                }
                let c = Cell {
                    x: x as usize,
                    z: z as usize,
                };
                if c == a_src || c == b_src || c == y_sink {
                    bad = true;
                    break;
                }
            }
            if bad {
                continue;
            }

            let v = lp.bin(v_anchor(ax, az));
            anchors.push(((ax, az), v));
        }
    }
    if anchors.is_empty() {
        eyre::bail!("유효한 AND anchor 후보가 하나도 없습니다");
    }

    // 변수: w (점유) + f (흐름)
    let w_index =
        |net: Net, c: Cell, xs: usize, zs: usize| -> usize { net.idx() * xs * zs + c.z * xs + c.x };
    let f_index = |net: Net, c: Cell, d: Dir, xs: usize, zs: usize| -> usize {
        net.idx() * xs * zs * 4 + (c.z * xs + c.x) * 4 + d.idx()
    };

    let mut w_var = vec![usize::MAX; Net::all().len() * xs * zs];
    let mut f_var = vec![usize::MAX; Net::all().len() * xs * zs * 4];

    for net in Net::all() {
        for z in 0..zs {
            for x in 0..xs {
                let c = Cell { x, z };
                let wi = lp.bin(v_w(net, c));
                w_var[w_index(net, c, xs, zs)] = wi;

                for d in Dir::all() {
                    let (dx, dz) = d.delta();
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;
                    if in_bounds(nx, nz, xs, zs) {
                        let fi = lp.bin(v_f(net, c, d));
                        f_var[f_index(net, c, d, xs, zs)] = fi;
                    }
                }
            }
        }
    }

    // 모든 제약 조건 추가
    add_all_constraints(
        &mut lp, &args, xs, zs, &anchors, &w_var, &f_var, a_src, b_src, y_sink,
    );

    // LP 파일 저장 및 SCIP 실행
    let lp_path = out_dir.join("pnr_and_tiles.lp");
    let sol_path = out_dir.join("pnr_and_tiles.sol");
    lp.write_lp(&lp_path)?;
    eprintln!("Wrote LP: {}", lp_path.display());

    if args.run_scip.unwrap_or(false) {
        run_scip(&args.scip, &lp_path, &sol_path)?;
        eprintln!("Wrote SOL: {}", sol_path.display());
    } else if !sol_path.exists() {
        eyre::bail!(
            "SOL 파일이 없습니다: {} (또는 --run-scip true)",
            sol_path.display()
        );
    }

    // 솔루션 파싱
    let sol = parse_scip_sol(&sol_path)?;
    if !sol.has_any() {
        eyre::bail!(
            "SCIP가 infeasible/no solution. LP 확인: {}",
            lp_path.display()
        );
    }

    // 선택된 앵커 찾기
    let mut chosen = None;
    for ((ax, az), _) in &anchors {
        if sol.get_bin(&v_anchor(*ax, *az)) {
            chosen = Some((*ax, *az));
            break;
        }
    }
    let (ax, az) = chosen.ok_or_else(|| eyre::eyre!("no AND anchor chosen"))?;

    let (a_sink, b_sink, y_src) = AndMacro::pins(ax, az);

    // 경로 재구성 (A/B/Y)
    let a_path = reconstruct_path(Net::A, a_src, a_sink, xs, zs, &sol)?;
    let b_path = reconstruct_path(Net::B, b_src, b_sink, xs, zs, &sol)?;
    let y_path = reconstruct_path(Net::Y, y_src, y_sink, xs, zs, &sol)?;

    // ---- 마인크래프트 블록 배치 (3D) ----
    let ox = args.origin_x;
    let oy = args.origin_y;
    let oz = args.origin_z;

    let mut vox: HashMap<(i32, i32, i32), String> = HashMap::new();

    // 출력 램프 베이스 + 위에 더스트
    vox.insert(
        (ox + y_sink.x as i32, oy, oz + y_sink.z as i32),
        "minecraft:redstone_lamp".to_string(),
    );
    vox.insert(
        (ox + y_sink.x as i32, oy + 1, oz + y_sink.z as i32),
        "minecraft:redstone_wire".to_string(),
    );

    // A/B 터미널 지지 블록 + 더스트
    for &src in &[a_src, b_src] {
        vox.entry((ox + src.x as i32, oy, oz + src.z as i32))
            .or_insert_with(|| "minecraft:stone".to_string());
        vox.insert(
            (ox + src.x as i32, oy + 1, oz + src.z as i32),
            "minecraft:redstone_wire".to_string(),
        );
    }

    // A/B용 레버를 베이스 블록의 서쪽에 배치
    for &src in &[a_src, b_src] {
        vox.insert(
            (ox - 1, oy, oz + src.z as i32),
            "minecraft:lever[face=wall,facing=east]".to_string(),
        );
    }

    // AND 매크로 블록 배치
    AndMacro::emit_blocks(ax, az, ox, oy, oz, &mut vox);

    // 터미널 집합 (리피터 교체 방지)
    let mut terminals: HashSet<Cell> = HashSet::new();
    terminals.insert(a_src);
    terminals.insert(b_src);
    terminals.insert(y_sink);
    terminals.insert(a_sink);
    terminals.insert(b_sink);
    terminals.insert(y_src);

    // 경로를 타일로 배치 (더스트 + 리피터)
    let mut repeater_marks: HashSet<Cell> = HashSet::new();
    place_path_tiles(
        &a_path,
        ax,
        az,
        &terminals,
        args.max_dust_run,
        ox,
        oy,
        oz,
        &mut vox,
        &mut repeater_marks,
    )?;
    place_path_tiles(
        &b_path,
        ax,
        az,
        &terminals,
        args.max_dust_run,
        ox,
        oy,
        oz,
        &mut vox,
        &mut repeater_marks,
    )?;
    place_path_tiles(
        &y_path,
        ax,
        az,
        &terminals,
        args.max_dust_run,
        ox,
        oy,
        oz,
        &mut vox,
        &mut repeater_marks,
    )?;

    // ---- 결과 출력 ----
    let mut placements: Vec<Placement> = vox
        .into_iter()
        .map(|((x, y, z), block)| Placement { x, y, z, block })
        .collect();
    placements.sort_by_key(|p| (p.y, p.z, p.x));

    eprintln!("Anchor chosen: AND at (ax={}, az={})", ax, az);
    eprintln!(
        "Pins: A_sink={:?} B_sink={:?} Y_src={:?} Y_sink={:?}",
        a_sink, b_sink, y_src, y_sink
    );
    eprintln!(
        "Paths: A(len={}) B(len={}) Y(len={})",
        a_path.len(),
        b_path.len(),
        y_path.len()
    );

    // /setblock 명령 출력
    let emit = args.emit_commands.unwrap_or(true) && !args.preview;
    if emit {
        for p in &placements {
            println!("/setblock {} {} {} {}", p.x, p.y, p.z, p.block);
        }
    }

    // ASCII 프리뷰 (--preview 또는 emit_commands=false일 때 항상)
    if args.preview || !args.emit_commands.unwrap_or(true) {
        let mut occ: HashMap<Cell, char> = HashMap::new();
        for &c in &a_path {
            occ.insert(c, Net::A.ch());
        }
        for &c in &b_path {
            occ.insert(c, Net::B.ch());
        }
        for &c in &y_path {
            occ.insert(c, Net::Y.ch());
        }

        println!("y=origin_y+1 layer preview:");
        for z in 0..zs {
            let mut row = String::new();
            for x in 0..xs {
                let c = Cell { x, z };
                let ch = if c == a_src {
                    'A'
                } else if c == b_src {
                    'B'
                } else if c == y_sink {
                    'Y'
                } else if AndMacro::is_torch_cell(c, ax, az) {
                    '&'
                } else if repeater_marks.contains(&c) {
                    'R'
                } else if let Some(&m) = occ.get(&c) {
                    m
                } else if AndMacro::is_internal_occupied_cell(c, ax, az) {
                    'm'
                } else {
                    '.'
                };
                row.push(ch);
            }
            println!("{}", row);
        }
    }

    // Placement를 World3D로 변환하고 NBT로 저장
    let world3d = placements_to_world3d(&placements)?;
    let nbt: NBTRoot = (&world3d).into();
    let nbt_path = out_dir.join("pnr_and_3d.nbt");
    nbt.save(&nbt_path);
    eprintln!("Saved NBT: {}", nbt_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver() -> Result<()> {
        let args = Args {
            x: 9,
            z: 7,
            origin_x: 0,
            origin_y: 64,
            origin_z: 0,
            scip: "scip".to_string(),
            run_scip: Some(true),
            emit_commands: Some(true),
            out_dir: ".".to_string(),
            max_dust_run: 12,
            macro_keepout: 0,
            net_keepout: 0,
            preview: false,
        };
        run_solver(args)
    }
}
