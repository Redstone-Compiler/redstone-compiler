use std::collections::HashMap;

use crate::solver::types::Cell;

/// AND 게이트 매크로 정의
/// 
/// 벽 토치 기반 AND 게이트를 구현합니다.
/// 라우팅 평면(y = origin_y + 1)에 배치되며,
/// 내부적으로 NOT(A)와 NOT(B)를 계산하여 AND 결과를 출력합니다.
pub struct AndMacro;

impl AndMacro {
    /// AND 매크로가 배치될 수 있는 앵커 도메인 계산
    /// 
    /// 매크로는 (ax, az) 위치에 앵커되며, 다음 공간이 필요합니다:
    /// - ax-1 >= 0 (A/B 핀 공간)
    /// - ax+5 < xs (Y 소스 핀 공간)
    /// - az+2 < zs (B 핀 공간)
    /// 
    /// 반환값: (ax_min, ax_max, az_min, az_max)
    pub fn anchor_domain(xs: usize, zs: usize) -> (usize, usize, usize, usize) {
        let ax_min = 1usize;
        let ax_max = xs.saturating_sub(6);
        let az_min = 0usize;
        let az_max = zs.saturating_sub(3);
        (ax_min, ax_max, az_min, az_max)
    }

    /// 매크로 내부 점유 셀 목록 (라우팅 평면 기준, 앵커 기준 상대 좌표)
    /// 
    /// 레이아웃 (라우팅 평면):
    /// ```
    /// z=az    :  A_pin(ax-1)  [BLK](0,0)  [WALL_TORCH](1,0)  BUS_DUST(2,0)
    /// z=az+1  :                               BUS_DUST(2,1) [BLK](3,1) [WALL_TORCH](4,1)  Y_pin(5,1)
    /// z=az+2  :  B_pin(ax-1)  [BLK](0,2)  [WALL_TORCH](1,2)  BUS_DUST(2,2)
    /// ```
    /// 
    /// 참고: 핀(A_pin/B_pin/Y_pin)은 라우터가 볼 수 있는 더스트 셀이므로 여기서는 차단하지 않습니다.
    pub fn internal_occupied() -> &'static [(i32, i32)] {
        &[
            // 라우팅 평면의 고체 블록 (토치 마운트)
            (0, 0),
            (0, 2),
            (3, 1),
            // 벽 토치 (자신의 셀을 점유)
            (1, 0),
            (1, 2),
            (4, 1),
            // 내부 버스 더스트
            (2, 0),
            (2, 1),
            (2, 2),
        ]
    }

    /// 매크로 핀 위치 (라우터가 볼 수 있는 더스트):
    /// - A sink = (ax-1, az)
    /// - B sink = (ax-1, az+2)
    /// - Y source = (ax+5, az+1)
    pub fn pins(ax: usize, az: usize) -> (Cell, Cell, Cell) {
        let a_sink = Cell { x: ax - 1, z: az };
        let b_sink = Cell {
            x: ax - 1,
            z: az + 2,
        };
        let y_src = Cell {
            x: ax + 5,
            z: az + 1,
        };
        (a_sink, b_sink, y_src)
    }

    /// 특정 셀이 토치 셀인지 확인 (벽 토치가 있는 셀)
    pub fn is_torch_cell(c: Cell, ax: usize, az: usize) -> bool {
        (c.x == ax + 1 && c.z == az)
            || (c.x == ax + 1 && c.z == az + 2)
            || (c.x == ax + 4 && c.z == az + 1)
    }

    /// 특정 셀이 매크로 내부 점유 셀인지 확인
    pub fn is_internal_occupied_cell(c: Cell, ax: usize, az: usize) -> bool {
        let rx = c.x as i32 - ax as i32;
        let rz = c.z as i32 - az as i32;
        Self::internal_occupied()
            .iter()
            .any(|&(dx, dz)| dx == rx && dz == rz)
    }

    /// AND 매크로 블록들을 3D 공간에 배치
    /// 
    /// 매크로는 다음과 같이 구성됩니다:
    /// - 3개의 고체 블록 (인버터 A, 인버터 B, 출력 블록)을 라우팅 평면(oy+1)에 배치
    /// - 각 고체 블록 아래에 지지 블록(oy) 배치
    /// - 벽 토치를 고체 블록의 동쪽 면에 부착
    /// - 내부 버스 더스트를 x=ax+2 위치에 배치
    pub fn emit_blocks(
        ax: usize,
        az: usize,
        ox: i32,
        oy: i32,
        oz: i32,
        vox: &mut HashMap<(i32, i32, i32), String>,
    ) {
        // ---- AND 매크로 (고정, 벽 토치 기반) ----
        // 라우팅 평면(oy+1)에 3개의 고체 블록 배치: 두 개의 인버터 블록 + 하나의 출력 블록
        // 각각은 oy에 지지 블록이 필요합니다.
        for &(bx, bz) in &[
            (ax as i32 + 0, az as i32 + 0), // 인버터 A 블록 (라우팅 평면 고체)
            (ax as i32 + 0, az as i32 + 2), // 인버터 B 블록
            (ax as i32 + 3, az as i32 + 1), // 출력 블록
        ] {
            // oy에 지지 블록
            let sup = (ox + bx, oy, oz + bz);
            if !vox.contains_key(&sup) {
                vox.insert(sup, "minecraft:stone".to_string());
            }
            // oy+1에 고체 블록 (마운트)
            vox.insert((ox + bx, oy + 1, oz + bz), "minecraft:stone".to_string());
        }

        // 벽 토치 배치 (라우팅 평면 oy+1에 위치)
        // - 인버터: 인버터 블록의 동쪽 면에 부착 => 토치 셀은 +1x, facing=east
        // - 출력: 출력 블록의 동쪽 면에 부착 => 토치 셀은 +1x (ax+4), facing=east
        for &(tx, tz) in &[
            (ax as i32 + 1, az as i32 + 0), // NOT(A) 토치
            (ax as i32 + 1, az as i32 + 2), // NOT(B) 토치
            (ax as i32 + 4, az as i32 + 1), // 출력 토치 (AND)
        ] {
            vox.insert(
                (ox + tx, oy + 1, oz + tz),
                "minecraft:redstone_wall_torch[facing=east]".to_string(),
            );
        }

        // 내부 버스 더스트를 x=ax+2에 배치하여 NOT(A)와 NOT(B)를 출력 블록(ax+3,az+1)에 연결
        for &(dx, dz) in &[
            (ax as i32 + 2, az as i32 + 0),
            (ax as i32 + 2, az as i32 + 1),
            (ax as i32 + 2, az as i32 + 2),
        ] {
            // oy에 지지 블록
            let base = (ox + dx, oy, oz + dz);
            if !vox.contains_key(&base) {
                vox.insert(base, "minecraft:stone".to_string());
            }
            vox.insert(
                (ox + dx, oy + 1, oz + dz),
                "minecraft:redstone_wire".to_string(),
            );
        }
    }
}

