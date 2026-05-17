use std::collections::HashMap;
use structopt::StructOpt;

/// 네트워크 타입: A, B, Y 세 개의 신호 네트워크
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Net {
    A,
    B,
    Y,
}

impl Net {
    /// 모든 네트워크 목록
    pub fn all() -> [Net; 3] {
        [Net::A, Net::B, Net::Y]
    }

    /// 네트워크를 인덱스로 변환 (0=A, 1=B, 2=Y)
    pub fn idx(self) -> usize {
        match self {
            Net::A => 0,
            Net::B => 1,
            Net::Y => 2,
        }
    }

    /// 네트워크 이름 문자열
    pub fn name(self) -> &'static str {
        match self {
            Net::A => "A",
            Net::B => "B",
            Net::Y => "Y",
        }
    }

    /// 네트워크를 문자로 표현 (ASCII 프리뷰용)
    pub fn ch(self) -> char {
        match self {
            Net::A => 'a',
            Net::B => 'b',
            Net::Y => 'y',
        }
    }
}

/// 방향: 북(N), 동(E), 남(S), 서(W)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dir {
    N,
    E,
    S,
    W,
}

impl Dir {
    /// 모든 방향 목록
    pub fn all() -> [Dir; 4] {
        [Dir::N, Dir::E, Dir::S, Dir::W]
    }

    /// 방향을 인덱스로 변환 (0=N, 1=E, 2=S, 3=W)
    pub fn idx(self) -> usize {
        match self {
            Dir::N => 0,
            Dir::E => 1,
            Dir::S => 2,
            Dir::W => 3,
        }
    }

    /// 방향에 따른 좌표 델타 (x, z)
    pub fn delta(self) -> (i32, i32) {
        match self {
            Dir::N => (0, -1),
            Dir::E => (1, 0),
            Dir::S => (0, 1),
            Dir::W => (-1, 0),
        }
    }

    /// 반대 방향 반환
    pub fn opposite(self) -> Dir {
        match self {
            Dir::N => Dir::S,
            Dir::E => Dir::W,
            Dir::S => Dir::N,
            Dir::W => Dir::E,
        }
    }

    /// 마인크래프트 facing 속성 문자열
    pub fn facing(self) -> &'static str {
        match self {
            Dir::N => "north",
            Dir::E => "east",
            Dir::S => "south",
            Dir::W => "west",
        }
    }
}

/// 2D 그리드 셀 좌표 (라우팅 평면)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cell {
    pub x: usize,
    pub z: usize,
}

/// 변수 이름 생성 함수들
/// w_{net}_{x}_{z}: 네트워크가 셀을 점유하는지 여부
pub fn v_w(net: Net, c: Cell) -> String {
    format!("w_{}_{}_{}", net.name(), c.x, c.z)
}

/// f_{net}_{x}_{z}_{dir}: 네트워크가 셀에서 특정 방향으로 흐르는지 여부
pub fn v_f(net: Net, c: Cell, d: Dir) -> String {
    format!("f_{}_{}_{}_{}", net.name(), c.x, c.z, d.idx())
}

/// and_{ax}_{az}: AND 매크로가 특정 앵커 위치에 배치되는지 여부
pub fn v_anchor(ax: usize, az: usize) -> String {
    format!("and_{}_{}", ax, az)
}

/// 경계 체크: 좌표가 그리드 범위 내에 있는지 확인
pub fn in_bounds(x: i32, z: i32, xs: usize, zs: usize) -> bool {
    x >= 0 && z >= 0 && (x as usize) < xs && (z as usize) < zs
}

/// SCIP 솔버의 해결책을 담는 구조체
#[derive(Debug)]
pub struct Solution {
    pub vals: HashMap<String, f64>,
}

impl Solution {
    /// 이진 변수의 값을 불리언으로 반환 (0.5 초과면 true)
    pub fn get_bin(&self, name: &str) -> bool {
        self.vals.get(name).copied().unwrap_or(0.0) > 0.5
    }

    /// 솔루션이 비어있지 않은지 확인
    pub fn has_any(&self) -> bool {
        !self.vals.is_empty()
    }
}

/// 마인크래프트 블록 배치 정보
#[derive(Clone, Debug)]
pub struct Placement {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub block: String,
}

/// 명령줄 인자 구조체
#[derive(StructOpt, Debug)]
#[structopt(name = "solver", about = "Place and route solver")]
pub struct Args {
    /// 그리드 X 크기 (마인크래프트 x)
    #[structopt(long, default_value = "9")]
    pub x: usize,

    /// 그리드 Z 크기 (마인크래프트 z)
    #[structopt(long, default_value = "7")]
    pub z: usize,

    /// setblock 명령의 원점 X (기본 레이어 y)
    #[structopt(long, default_value = "0")]
    pub origin_x: i32,

    #[structopt(long, default_value = "64")]
    pub origin_y: i32,

    #[structopt(long, default_value = "0")]
    pub origin_z: i32,

    /// SCIP 실행 파일 경로 (예: "D:\\SCIP\\bin\\scip.exe")
    #[structopt(long, default_value = "scip")]
    pub scip: String,

    /// SCIP를 실행하여 해결할지 여부
    #[structopt(long)]
    pub run_scip: Option<bool>,

    /// /setblock 명령 출력 여부
    #[structopt(long)]
    pub emit_commands: Option<bool>,

    /// 출력 디렉토리
    #[structopt(long, default_value = ".")]
    pub out_dir: String,

    /// 긴 직선 레드스톤 더스트 구간에 자동으로 리피터 삽입 (0이면 비활성화)
    #[structopt(long, default_value = "12")]
    pub max_dust_run: usize,

    /// 매크로 내부 주변 keepout (0이면 비활성화)
    #[structopt(long, default_value = "0")]
    pub macro_keepout: usize,

    /// 다른 네트워크 간 인접 keepout (0이면 비활성화, 더스트 자동 연결 방지)
    #[structopt(long, default_value = "0")]
    pub net_keepout: usize,

    /// ASCII 프리뷰 출력 (NBT는 여전히 작성됨)
    #[structopt(long)]
    pub preview: bool,
}

