use crate::solver::and_macro::AndMacro;
use crate::solver::lp::{Lp, Sense};
use crate::solver::types::{Args, Cell, Dir, Net, in_bounds};

/// LP 모델에 모든 제약 조건 추가
pub fn add_all_constraints(
    lp: &mut Lp,
    args: &Args,
    xs: usize,
    zs: usize,
    anchors: &[((usize, usize), usize)],
    w_var: &[usize],
    f_var: &[usize],
    a_src: Cell,
    b_src: Cell,
    y_sink: Cell,
) {
    // 변수 인덱스 계산 함수들
    let w_index = |net: Net, c: Cell, xs: usize, zs: usize| -> usize {
        net.idx() * xs * zs + c.z * xs + c.x
    };
    let f_index = |net: Net, c: Cell, d: Dir, xs: usize, zs: usize| -> usize {
        net.idx() * xs * zs * 4 + (c.z * xs + c.x) * 4 + d.idx()
    };

    // 1) 정확히 하나의 앵커만 선택
    {
        let terms = anchors.iter().map(|(_, v)| (1.0, *v)).collect();
        lp.add_c("one_and".to_string(), terms, Sense::Eq, 1.0);
    }

    // 2) 매크로 내부 차단: w(net,cell) + anchor <= 1
    add_macro_internal_blocking(lp, anchors, xs, zs, w_var, w_index);

    // 2b) 민감한 블록 주변 keepout (의도된 핀만 전원 공급 가능하도록)
    add_sensitive_block_keepout(lp, anchors, xs, zs, w_var, w_index);

    // 2c) 선택적: 매크로 keepout (반경)
    if args.macro_keepout > 0 {
        add_macro_keepout(lp, anchors, args.macro_keepout, xs, zs, w_var, w_index);
    }

    // 3) 터미널 소유권: A/B 소스와 Y 싱크에서 w=1 강제
    add_terminal_ownership(lp, a_src, b_src, y_sink, xs, zs, w_var, w_index);

    // 4) 매크로 핀 소유권: w >= anchor (anchor - w <= 0)
    add_macro_pin_ownership(lp, anchors, xs, zs, w_var, w_index);

    // 5) 흐름을 w에 연결 (엣지는 양쪽 끝점 셀이 모두 사용됨을 의미)
    add_flow_linkage(lp, xs, zs, w_var, f_var, w_index, f_index);

    // 6) 차수 제한: 분기 없음 (out<=1, in<=1)
    add_degree_limits(lp, xs, zs, f_var, f_index);

    // 7) 앵커 의존적 소스/싱크를 포함한 흐름 보존
    add_flow_conservation(lp, anchors, a_src, b_src, y_sink, xs, zs, f_var, f_index);

    // 8) 같은 셀에서 네트워크 간 겹침 없음
    add_no_overlap(lp, xs, zs, w_var, w_index);

    // 8b) 선택적: 인접성에 의한 네트워크 keepout (더스트 자동 연결 단락 방지)
    if args.net_keepout > 0 {
        add_net_keepout(lp, xs, zs, w_var, w_index);
    }

    // 목적 함수: 사용된 라우팅 셀 총합 최소화
    add_objective(lp, xs, zs, w_var, w_index);
}

/// 매크로 내부 차단 제약 조건 추가
fn add_macro_internal_blocking(
    lp: &mut Lp,
    anchors: &[((usize, usize), usize)],
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    for ((ax, az), av) in anchors {
        for &(dx, dz) in AndMacro::internal_occupied() {
            let x = *ax as i32 + dx;
            let z = *az as i32 + dz;
            if !in_bounds(x, z, xs, zs) {
                continue;
            }
            let c = Cell {
                x: x as usize,
                z: z as usize,
            };
            for net in Net::all() {
                let w = w_var[w_index(net, c, xs, zs)];
                lp.add_c(
                    format!("blk_{}_{}_{}_{}_{}", net.name(), ax, az, dx, dz),
                    vec![(1.0, w), (1.0, *av)],
                    Sense::Le,
                    1.0,
                );
            }
        }
    }
}

/// 민감한 블록 주변 keepout 제약 조건 추가
/// 
/// 민감한 블록은 (ax,az)와 (ax,az+2)의 인버터 마운트 블록입니다.
/// 다른 네트워크의 임의의 더스트가 인접하면 토치가 잘못 뒤집힐 수 있습니다.
fn add_sensitive_block_keepout(
    lp: &mut Lp,
    anchors: &[((usize, usize), usize)],
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    for ((ax, az), av) in anchors {
        let inv_a_blk = Cell { x: *ax, z: *az };
        let inv_b_blk = Cell { x: *ax, z: az + 2 };
        let sensitive = [
            (inv_a_blk, Net::A, Cell { x: ax - 1, z: *az }),
            (
                inv_b_blk,
                Net::B,
                Cell {
                    x: ax - 1,
                    z: az + 2,
                },
            ),
        ];

        for (blk, allowed_net, allowed_pin) in sensitive {
            for d in Dir::all() {
                let (dx, dz) = d.delta();
                let nx = blk.x as i32 + dx;
                let nz = blk.z as i32 + dz;
                if !in_bounds(nx, nz, xs, zs) {
                    continue;
                }
                let n = Cell {
                    x: nx as usize,
                    z: nz as usize,
                };
                for net in Net::all() {
                    // 지정된 네트워크만 지정된 핀 셀에 허용
                    if net == allowed_net && n == allowed_pin {
                        continue;
                    }
                    let w = w_var[w_index(net, n, xs, zs)];
                    lp.add_c(
                        format!("keepout_{}_{}_{}_{}_{}", net.name(), ax, az, blk.x, blk.z),
                        vec![(1.0, w), (1.0, *av)],
                        Sense::Le,
                        1.0,
                    );
                }
            }
        }
    }
}

/// 매크로 keepout 제약 조건 추가 (반경 기반)
fn add_macro_keepout(
    lp: &mut Lp,
    anchors: &[((usize, usize), usize)],
    radius: usize,
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    let r = radius as i32;
    for ((ax, az), av) in anchors {
        let (a_sink, b_sink, y_src) = AndMacro::pins(*ax, *az);

        for &(dx, dz) in AndMacro::internal_occupied() {
            let cx = *ax as i32 + dx;
            let cz = *az as i32 + dz;
            for oz_ in -r..=r {
                for ox_ in -r..=r {
                    let nx = cx + ox_;
                    let nz = cz + oz_;
                    if !in_bounds(nx, nz, xs, zs) {
                        continue;
                    }
                    let c = Cell {
                        x: nx as usize,
                        z: nz as usize,
                    };

                    // 핀(including y_src)은 차단하지 않음
                    if c == a_sink || c == b_sink || c == y_src {
                        continue;
                    }

                    for net in Net::all() {
                        let w = w_var[w_index(net, c, xs, zs)];
                        lp.add_c(
                            format!("mkeep_{}_{}_{}_{}_{}", net.name(), ax, az, nx, nz),
                            vec![(1.0, w), (1.0, *av)],
                            Sense::Le,
                            1.0,
                        );
                    }
                }
            }
        }
    }
}

/// 터미널 소유권 제약 조건 추가
fn add_terminal_ownership(
    lp: &mut Lp,
    a_src: Cell,
    b_src: Cell,
    y_sink: Cell,
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    let wa = w_var[w_index(Net::A, a_src, xs, zs)];
    lp.add_c("A_src".to_string(), vec![(1.0, wa)], Sense::Eq, 1.0);

    let wb = w_var[w_index(Net::B, b_src, xs, zs)];
    lp.add_c("B_src".to_string(), vec![(1.0, wb)], Sense::Eq, 1.0);

    let wy = w_var[w_index(Net::Y, y_sink, xs, zs)];
    lp.add_c("Y_sink".to_string(), vec![(1.0, wy)], Sense::Eq, 1.0);
}

/// 매크로 핀 소유권 제약 조건 추가
fn add_macro_pin_ownership(
    lp: &mut Lp,
    anchors: &[((usize, usize), usize)],
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    for ((ax, az), av) in anchors {
        let (a_sink, b_sink, y_src) = AndMacro::pins(*ax, *az);

        let wa = w_var[w_index(Net::A, a_sink, xs, zs)];
        lp.add_c(
            format!("pin_A_{}_{}", ax, az),
            vec![(1.0, *av), (-1.0, wa)],
            Sense::Le,
            0.0,
        );

        let wb = w_var[w_index(Net::B, b_sink, xs, zs)];
        lp.add_c(
            format!("pin_B_{}_{}", ax, az),
            vec![(1.0, *av), (-1.0, wb)],
            Sense::Le,
            0.0,
        );

        let wy = w_var[w_index(Net::Y, y_src, xs, zs)];
        lp.add_c(
            format!("pin_Y_{}_{}", ax, az),
            vec![(1.0, *av), (-1.0, wy)],
            Sense::Le,
            0.0,
        );
    }
}

/// 흐름-점유 연결 제약 조건 추가
fn add_flow_linkage(
    lp: &mut Lp,
    xs: usize,
    zs: usize,
    w_var: &[usize],
    f_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
    f_index: impl Fn(Net, Cell, Dir, usize, usize) -> usize,
) {
    for net in Net::all() {
        for z in 0..zs {
            for x in 0..xs {
                let c = Cell { x, z };
                let w = w_var[w_index(net, c, xs, zs)];

                for d in Dir::all() {
                    let fi = f_var[f_index(net, c, d, xs, zs)];
                    if fi == usize::MAX {
                        continue;
                    }
                    lp.add_c(
                        format!("link1_{}_{}_{}_{}", net.name(), x, z, d.idx()),
                        vec![(1.0, fi), (-1.0, w)],
                        Sense::Le,
                        0.0,
                    );

                    let (dx, dz) = d.delta();
                    let nx = (x as i32 + dx) as usize;
                    let nz = (z as i32 + dz) as usize;
                    let n = Cell { x: nx, z: nz };
                    let wn = w_var[w_index(net, n, xs, zs)];
                    lp.add_c(
                        format!("link2_{}_{}_{}_{}", net.name(), x, z, d.idx()),
                        vec![(1.0, fi), (-1.0, wn)],
                        Sense::Le,
                        0.0,
                    );
                }
            }
        }
    }
}

/// 차수 제한 제약 조건 추가
fn add_degree_limits(
    lp: &mut Lp,
    xs: usize,
    zs: usize,
    f_var: &[usize],
    f_index: impl Fn(Net, Cell, Dir, usize, usize) -> usize,
) {
    for net in Net::all() {
        for z in 0..zs {
            for x in 0..xs {
                let c = Cell { x, z };
                let mut out_terms = Vec::new();
                let mut in_terms = Vec::new();

                for d in Dir::all() {
                    let fo = f_var[f_index(net, c, d, xs, zs)];
                    if fo != usize::MAX {
                        out_terms.push((1.0, fo));
                    }
                    let (dx, dz) = d.delta();
                    let px = x as i32 + dx;
                    let pz = z as i32 + dz;
                    if in_bounds(px, pz, xs, zs) {
                        let p = Cell {
                            x: px as usize,
                            z: pz as usize,
                        };
                        let fi = f_var[f_index(net, p, d.opposite(), xs, zs)];
                        if fi != usize::MAX {
                            in_terms.push((1.0, fi));
                        }
                    }
                }

                lp.add_c(
                    format!("out1_{}_{}_{}", net.name(), x, z),
                    out_terms,
                    Sense::Le,
                    1.0,
                );
                lp.add_c(
                    format!("in1_{}_{}_{}", net.name(), x, z),
                    in_terms,
                    Sense::Le,
                    1.0,
                );
            }
        }
    }
}

/// 흐름 보존 제약 조건 추가
fn add_flow_conservation(
    lp: &mut Lp,
    anchors: &[((usize, usize), usize)],
    a_src: Cell,
    b_src: Cell,
    y_sink: Cell,
    xs: usize,
    zs: usize,
    f_var: &[usize],
    f_index: impl Fn(Net, Cell, Dir, usize, usize) -> usize,
) {
    for net in Net::all() {
        for z in 0..zs {
            for x in 0..xs {
                let c = Cell { x, z };
                let mut terms: Vec<(f64, usize)> = Vec::new();

                // 출력
                for d in Dir::all() {
                    let fo = f_var[f_index(net, c, d, xs, zs)];
                    if fo != usize::MAX {
                        terms.push((1.0, fo));
                    }
                }
                // 입력
                for d in Dir::all() {
                    let (dx, dz) = d.delta();
                    let px = x as i32 + dx;
                    let pz = z as i32 + dz;
                    if !in_bounds(px, pz, xs, zs) {
                        continue;
                    }
                    let p = Cell {
                        x: px as usize,
                        z: pz as usize,
                    };
                    let fi = f_var[f_index(net, p, d.opposite(), xs, zs)];
                    if fi != usize::MAX {
                        terms.push((-1.0, fi));
                    }
                }

                let mut rhs = 0.0;
                match net {
                    Net::A => {
                        if c == a_src {
                            rhs = 1.0;
                        }
                        for ((ax, az), av) in anchors {
                            let (a_sink, _, _) = AndMacro::pins(*ax, *az);
                            if a_sink == c {
                                terms.push((1.0, *av)); // out-in + anchor = 0  => out-in = -anchor
                            }
                        }
                    }
                    Net::B => {
                        if c == b_src {
                            rhs = 1.0;
                        }
                        for ((ax, az), av) in anchors {
                            let (_, b_sink, _) = AndMacro::pins(*ax, *az);
                            if b_sink == c {
                                terms.push((1.0, *av));
                            }
                        }
                    }
                    Net::Y => {
                        if c == y_sink {
                            rhs = -1.0;
                        }
                        for ((ax, az), av) in anchors {
                            let (_, _, y_src) = AndMacro::pins(*ax, *az);
                            if y_src == c {
                                terms.push((-1.0, *av)); // out-in - anchor = 0 => out-in = anchor
                            }
                        }
                    }
                }

                lp.add_c(
                    format!("flow_{}_{}_{}", net.name(), x, z),
                    terms,
                    Sense::Eq,
                    rhs,
                );
            }
        }
    }
}

/// 네트워크 간 겹침 방지 제약 조건 추가
fn add_no_overlap(
    lp: &mut Lp,
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    for z in 0..zs {
        for x in 0..xs {
            let c = Cell { x, z };
            let wa = w_var[w_index(Net::A, c, xs, zs)];
            let wb = w_var[w_index(Net::B, c, xs, zs)];
            let wy = w_var[w_index(Net::Y, c, xs, zs)];
            lp.add_c(
                format!("no_overlap_{}_{}", x, z),
                vec![(1.0, wa), (1.0, wb), (1.0, wy)],
                Sense::Le,
                1.0,
            );
        }
    }
}

/// 네트워크 keepout 제약 조건 추가 (인접성 기반)
fn add_net_keepout(
    lp: &mut Lp,
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    for z in 0..zs {
        for x in 0..xs {
            let c = Cell { x, z };
            for d in Dir::all() {
                let (dx, dz) = d.delta();
                let nx = x as i32 + dx;
                let nz = z as i32 + dz;
                if !in_bounds(nx, nz, xs, zs) {
                    continue;
                }
                let n = Cell {
                    x: nx as usize,
                    z: nz as usize,
                };

                let nets = Net::all();
                for i in 0..nets.len() {
                    for j in (i + 1)..nets.len() {
                        let ni = nets[i];
                        let nj = nets[j];
                        let wi = w_var[w_index(ni, c, xs, zs)];
                        let wj = w_var[w_index(nj, n, xs, zs)];
                        lp.add_c(
                            format!("nk_{}_{}_{}_{}", ni.name(), nj.name(), x, z),
                            vec![(1.0, wi), (1.0, wj)],
                            Sense::Le,
                            1.0,
                        );
                        let wi2 = w_var[w_index(ni, n, xs, zs)];
                        let wj2 = w_var[w_index(nj, c, xs, zs)];
                        lp.add_c(
                            format!("nk2_{}_{}_{}_{}", ni.name(), nj.name(), x, z),
                            vec![(1.0, wi2), (1.0, wj2)],
                            Sense::Le,
                            1.0,
                        );
                    }
                }
            }
        }
    }
}

/// 목적 함수 추가: 사용된 라우팅 셀 총합 최소화
fn add_objective(
    lp: &mut Lp,
    xs: usize,
    zs: usize,
    w_var: &[usize],
    w_index: impl Fn(Net, Cell, usize, usize) -> usize,
) {
    for net in Net::all() {
        for z in 0..zs {
            for x in 0..xs {
                let c = Cell { x, z };
                let w = w_var[w_index(net, c, xs, zs)];
                lp.add_obj(1.0, w);
            }
        }
    }
}

