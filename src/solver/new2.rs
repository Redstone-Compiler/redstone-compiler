use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use eyre::{bail, Context, Result};

// ============================================================
// 0) Grid / Dir
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cell {
    pub x: usize,
    pub z: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    North,
    East,
    South,
    West,
}

impl Dir {
    pub const ALL: [Dir; 4] = [Dir::North, Dir::East, Dir::South, Dir::West];

    pub fn dxz(self) -> (i32, i32) {
        match self {
            Dir::North => (0, -1),
            Dir::East => (1, 0),
            Dir::South => (0, 1),
            Dir::West => (-1, 0),
        }
    }

    pub fn opp(self) -> Dir {
        match self {
            Dir::North => Dir::South,
            Dir::East => Dir::West,
            Dir::South => Dir::North,
            Dir::West => Dir::East,
        }
    }

    pub fn mc_facing(self) -> &'static str {
        match self {
            Dir::North => "north",
            Dir::East => "east",
            Dir::South => "south",
            Dir::West => "west",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Grid {
    pub xs: usize,
    pub zs: usize,
}

impl Grid {
    pub fn new(xs: usize, zs: usize) -> Self {
        Self { xs, zs }
    }

    pub fn in_bounds(&self, x: i32, z: i32) -> bool {
        x >= 0 && z >= 0 && (x as usize) < self.xs && (z as usize) < self.zs
    }

    pub fn neighbor(&self, c: Cell, dir: Dir) -> Option<Cell> {
        let (dx, dz) = dir.dxz();
        let nx = c.x as i32 + dx;
        let nz = c.z as i32 + dz;
        if self.in_bounds(nx, nz) {
            Some(Cell {
                x: nx as usize,
                z: nz as usize,
            })
        } else {
            None
        }
    }

    pub fn iter_cells(&self) -> impl Iterator<Item = Cell> + '_ {
        (0..self.xs).flat_map(move |x| (0..self.zs).map(move |z| Cell { x, z }))
    }

    /// combinational / no-feedback helper: total order index (acyclic flow: low -> high)
    pub fn order(&self, c: Cell) -> usize {
        c.x * self.zs + c.z
    }
}

// ============================================================
// 1) LP builder (binary-only is enough for now)
// ============================================================

#[derive(Clone, Copy, Debug)]
pub enum Sense {
    Le,
    Ge,
    Eq,
}

#[derive(Clone, Debug)]
pub struct Constraint {
    pub name: String,
    pub terms: Vec<(f64, usize)>,
    pub sense: Sense,
    pub rhs: f64,
}

#[derive(Default)]
pub struct Lp {
    var_names: Vec<String>,
    var_index: HashMap<String, usize>,
    is_binary: Vec<bool>,
    objective: Vec<(f64, usize)>,
    constraints: Vec<Constraint>,
    minimize: bool,
}

impl Lp {
    pub fn new() -> Self {
        Self {
            minimize: true,
            ..Default::default()
        }
    }

    pub fn bin(&mut self, name: impl Into<String>) -> usize {
        let name = name.into();
        if let Some(&i) = self.var_index.get(&name) {
            return i;
        }
        let i = self.var_names.len();
        self.var_names.push(name.clone());
        self.var_index.insert(name, i);
        self.is_binary.push(true);
        i
    }

    pub fn add_obj(&mut self, coef: f64, var: usize) {
        self.objective.push((coef, var));
    }

    pub fn add_c(
        &mut self,
        name: impl Into<String>,
        terms: Vec<(f64, usize)>,
        sense: Sense,
        rhs: f64,
    ) {
        self.constraints.push(Constraint {
            name: name.into(),
            terms,
            sense,
            rhs,
        });
    }

    pub fn eq(&mut self, name: impl Into<String>, var: usize, rhs: f64) {
        self.add_c(name, vec![(1.0, var)], Sense::Eq, rhs);
    }

    pub fn write_lp(&self, path: &Path) -> Result<()> {
        let mut s = String::new();

        if self.minimize {
            s.push_str("Minimize\n obj: ");
        } else {
            s.push_str("Maximize\n obj: ");
        }

        if self.objective.is_empty() {
            s.push_str("0\n");
        } else {
            s.push_str(&format_linexpr(&self.objective, &self.var_names));
            s.push('\n');
        }

        s.push_str("Subject To\n");
        for c in &self.constraints {
            s.push(' ');
            s.push_str(&c.name);
            s.push_str(": ");
            s.push_str(&format_linexpr(&c.terms, &self.var_names));
            match c.sense {
                Sense::Le => s.push_str(" <= "),
                Sense::Ge => s.push_str(" >= "),
                Sense::Eq => s.push_str(" = "),
            }
            s.push_str(&format!("{}", c.rhs));
            s.push('\n');
        }

        s.push_str("Bounds\n");
        for (i, name) in self.var_names.iter().enumerate() {
            if self.is_binary[i] {
                s.push_str(&format!(" 0 <= {} <= 1\n", name));
            } else {
                s.push_str(&format!(" {} free\n", name));
            }
        }

        s.push_str("Binaries\n");
        for (i, name) in self.var_names.iter().enumerate() {
            if self.is_binary[i] {
                s.push(' ');
                s.push_str(name);
                s.push('\n');
            }
        }
        s.push_str("End\n");

        fs::write(path, s)?;
        Ok(())
    }

    pub fn var_name(&self, v: usize) -> &str {
        &self.var_names[v]
    }
}

fn format_linexpr(terms: &[(f64, usize)], var_names: &[String]) -> String {
    let mut out = String::new();
    let mut first = true;
    for &(coef, vid) in terms {
        if coef == 0.0 {
            continue;
        }
        if first {
            first = false;
            out.push_str(&format_term(coef, &var_names[vid], true));
        } else {
            out.push_str(&format_term(coef, &var_names[vid], false));
        }
    }
    if out.is_empty() {
        out.push('0');
    }
    out
}

fn format_term(coef: f64, name: &str, first: bool) -> String {
    let sign = if coef < 0.0 { "-" } else { "+" };
    let abs = coef.abs();

    if first {
        if coef < 0.0 {
            if abs == 1.0 {
                format!("- {}", name)
            } else {
                format!("- {} {}", abs, name)
            }
        } else if abs == 1.0 {
            format!("{}", name)
        } else {
            format!("{} {}", abs, name)
        }
    } else if abs == 1.0 {
        format!(" {} {}", sign, name)
    } else {
        format!(" {} {} {}", sign, abs, name)
    }
}

// ============================================================
// 2) SCIP integration
// ============================================================

#[derive(Debug, Clone)]
pub struct Solution {
    pub vals: HashMap<String, f64>,
}

impl Solution {
    pub fn get_bin(&self, name: &str) -> bool {
        self.vals.get(name).copied().unwrap_or(0.0) > 0.5
    }
}

fn run_scip(scip_path: &str, lp_path: &Path, sol_path: &Path) -> Result<()> {
    let lpq = format!("\"{}\"", lp_path.display());
    let solq = format!("\"{}\"", sol_path.display());
    let cmd = format!("read {} optimize write solution {} quit", lpq, solq);

    let status = Command::new(scip_path)
        .arg("-c")
        .arg(cmd)
        .status()
        .with_context(|| format!("failed to run scip: {}", scip_path))?;

    if !status.success() {
        bail!("SCIP failed: {:?}", status);
    }
    Ok(())
}

fn parse_scip_sol(path: &Path) -> Result<Solution> {
    let txt = fs::read_to_string(path).with_context(|| format!("read sol: {}", path.display()))?;
    let mut vals = HashMap::new();
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

// ============================================================
// 3) Variable naming helpers
// ============================================================

fn nm_cell(prefix: &str, c: Cell) -> String {
    format!("{}_{}_{}", prefix, c.x, c.z)
}
fn nm_cell_dir(prefix: &str, c: Cell, d: Dir) -> String {
    format!("{}_{}_{}_{}", prefix, c.x, c.z, d.mc_facing())
}
fn nm_sc(prefix: &str, s: usize, c: Cell) -> String {
    format!("{}_{}_{}_{}", prefix, s, c.x, c.z)
}
fn nm_sc_dir(prefix: &str, s: usize, c: Cell, d: Dir) -> String {
    format!("{}_{}_{}_{}_{}", prefix, s, c.x, c.z, d.mc_facing())
}

// ============================================================
// 4) Problem Spec (TruthTable 기반: NOT/AND/XOR)
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalKind {
    /// top dust on a block: (requires S=1 and D1=1), observed as DP1
    TopDust,
    /// ground dust on floor (air at y=0): (requires S=0 and D0=1), observed as DP0
    GroundDust,
}

#[derive(Clone, Debug)]
pub struct Port {
    pub name: String,
    pub at: Cell,
    pub kind: SignalKind,
}

#[derive(Clone, Debug)]
pub struct GateSpec {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    /// scenarios: each row is (input bits, output bits)
    pub table: Vec<(Vec<u8>, Vec<u8>)>,
}

#[derive(Clone, Copy, Debug)]
pub enum GateKind {
    Not,
    And,
    Xor,
}

impl GateSpec {
    pub fn from_kind(kind: GateKind, grid: &Grid) -> Self {
        let midz = grid.zs / 2;

        match kind {
            GateKind::Not => {
                let in0 = Port {
                    name: "A".into(),
                    at: Cell { x: 0, z: midz },
                    kind: SignalKind::TopDust,
                };
                let out = Port {
                    name: "Y".into(),
                    at: Cell {
                        x: grid.xs - 1,
                        z: midz,
                    },
                    kind: SignalKind::GroundDust,
                };

                let table = vec![(vec![0], vec![1]), (vec![1], vec![0])];
                Self {
                    inputs: vec![in0],
                    outputs: vec![out],
                    table,
                }
            }
            GateKind::And => {
                let a = Port {
                    name: "A".into(),
                    at: Cell {
                        x: 0,
                        z: midz.saturating_sub(1),
                    },
                    kind: SignalKind::TopDust,
                };
                let b = Port {
                    name: "B".into(),
                    at: Cell { x: 0, z: midz + 1 },
                    kind: SignalKind::TopDust,
                };
                let y = Port {
                    name: "Y".into(),
                    at: Cell {
                        x: grid.xs - 1,
                        z: midz,
                    },
                    kind: SignalKind::GroundDust,
                };
                let mut table = Vec::new();
                for &ai in &[0u8, 1u8] {
                    for &bi in &[0u8, 1u8] {
                        let yi = (ai & bi) as u8;
                        table.push((vec![ai, bi], vec![yi]));
                    }
                }
                Self {
                    inputs: vec![a, b],
                    outputs: vec![y],
                    table,
                }
            }
            GateKind::Xor => {
                let a = Port {
                    name: "A".into(),
                    at: Cell {
                        x: 0,
                        z: midz.saturating_sub(1),
                    },
                    kind: SignalKind::TopDust,
                };
                let b = Port {
                    name: "B".into(),
                    at: Cell { x: 0, z: midz + 1 },
                    kind: SignalKind::TopDust,
                };
                let y = Port {
                    name: "Y".into(),
                    at: Cell {
                        x: grid.xs - 1,
                        z: midz,
                    },
                    kind: SignalKind::GroundDust,
                };
                let mut table = Vec::new();
                for &ai in &[0u8, 1u8] {
                    for &bi in &[0u8, 1u8] {
                        let yi = (ai ^ bi) as u8;
                        table.push((vec![ai, bi], vec![yi]));
                    }
                }
                Self {
                    inputs: vec![a, b],
                    outputs: vec![y],
                    table,
                }
            }
        }
    }

    pub fn scenario_count(&self) -> usize {
        self.table.len()
    }
}

// ============================================================
// 5) Vars (typed handles)
// ============================================================

#[derive(Clone)]
pub struct Vars {
    // y=0 occupancy: S (block), D0 (ground dust), wall torch TW0, repeater R0
    pub s: Vec<Vec<usize>>,
    pub d0: Vec<Vec<usize>>,
    pub tw0: Vec<Vec<[usize; 4]>>, // per dir
    pub r0: Vec<Vec<[usize; 4]>>,  // per dir

    // y=1 occupancy: D1(top dust), T1(standing torch)
    pub d1: Vec<Vec<usize>>,
    pub t1: Vec<Vec<usize>>,

    // Dust shape (placement): Conn and (Cross/AxisNS/AxisEW) for each layer
    pub conn0: Vec<Vec<[usize; 4]>>,
    pub conn1: Vec<Vec<[usize; 4]>>,
    pub cross0: Vec<Vec<usize>>,
    pub axis_ns0: Vec<Vec<usize>>,
    pub axis_ew0: Vec<Vec<usize>>,
    pub cross1: Vec<Vec<usize>>,
    pub axis_ns1: Vec<Vec<usize>>,
    pub axis_ew1: Vec<Vec<usize>>,

    // State per scenario
    pub bp: Vec<Vec<Vec<Vec<usize>>>>,        // [s][x][z]
    pub dp0: Vec<Vec<Vec<Vec<usize>>>>,       // [s][x][z]
    pub dp1: Vec<Vec<Vec<Vec<usize>>>>,       // [s][x][z]
    pub to1: Vec<Vec<Vec<Vec<usize>>>>,       // [s][x][z]
    pub tow0: Vec<Vec<Vec<Vec<[usize; 4]>>>>, // [s][x][z][dir]
    pub ro0: Vec<Vec<Vec<Vec<[usize; 4]>>>>,  // [s][x][z][dir]

    // For debug checks
    pub out_dp0_vars: Vec<usize>, // DP0 var ids at output per scenario (assumes one output)
}

impl Vars {
    pub fn alloc(lp: &mut Lp, grid: &Grid, sc: usize, out_port: &Port) -> Self {
        let xs = grid.xs;
        let zs = grid.zs;

        let mut s = vec![vec![0usize; zs]; xs];
        let mut d0 = vec![vec![0usize; zs]; xs];
        let mut d1 = vec![vec![0usize; zs]; xs];
        let mut t1 = vec![vec![0usize; zs]; xs];
        let mut tw0 = vec![vec![[0usize; 4]; zs]; xs];
        let mut r0 = vec![vec![[0usize; 4]; zs]; xs];

        let mut conn0 = vec![vec![[0usize; 4]; zs]; xs];
        let mut conn1 = vec![vec![[0usize; 4]; zs]; xs];

        let mut cross0 = vec![vec![0usize; zs]; xs];
        let mut axis_ns0 = vec![vec![0usize; zs]; xs];
        let mut axis_ew0 = vec![vec![0usize; zs]; xs];

        let mut cross1 = vec![vec![0usize; zs]; xs];
        let mut axis_ns1 = vec![vec![0usize; zs]; xs];
        let mut axis_ew1 = vec![vec![0usize; zs]; xs];

        for c in grid.iter_cells() {
            s[c.x][c.z] = lp.bin(nm_cell("S", c));
            d0[c.x][c.z] = lp.bin(nm_cell("D0", c));
            d1[c.x][c.z] = lp.bin(nm_cell("D1", c));
            t1[c.x][c.z] = lp.bin(nm_cell("T1", c));

            cross0[c.x][c.z] = lp.bin(nm_cell("Cross0", c));
            axis_ns0[c.x][c.z] = lp.bin(nm_cell("AxisNS0", c));
            axis_ew0[c.x][c.z] = lp.bin(nm_cell("AxisEW0", c));

            cross1[c.x][c.z] = lp.bin(nm_cell("Cross1", c));
            axis_ns1[c.x][c.z] = lp.bin(nm_cell("AxisNS1", c));
            axis_ew1[c.x][c.z] = lp.bin(nm_cell("AxisEW1", c));

            for (i, d) in Dir::ALL.iter().enumerate() {
                tw0[c.x][c.z][i] = lp.bin(nm_cell_dir("TW0", c, *d));
                r0[c.x][c.z][i] = lp.bin(nm_cell_dir("R0", c, *d));
                conn0[c.x][c.z][i] = lp.bin(nm_cell_dir("Conn0", c, *d));
                conn1[c.x][c.z][i] = lp.bin(nm_cell_dir("Conn1", c, *d));
            }
        }

        let mut bp = vec![vec![vec![vec![0usize; zs]; xs]; 1]; sc];
        let mut dp0 = vec![vec![vec![vec![0usize; zs]; xs]; 1]; sc];
        let mut dp1 = vec![vec![vec![vec![0usize; zs]; xs]; 1]; sc];
        let mut to1 = vec![vec![vec![vec![0usize; zs]; xs]; 1]; sc];
        let mut tow0 = vec![vec![vec![vec![[0usize; 4]; zs]; xs]; 1]; sc];
        let mut ro0 = vec![vec![vec![vec![[0usize; 4]; zs]; xs]; 1]; sc];

        for si in 0..sc {
            for c in grid.iter_cells() {
                bp[si][0][c.x][c.z] = lp.bin(nm_sc("BP", si, c));
                dp0[si][0][c.x][c.z] = lp.bin(nm_sc("DP0", si, c));
                dp1[si][0][c.x][c.z] = lp.bin(nm_sc("DP1", si, c));
                to1[si][0][c.x][c.z] = lp.bin(nm_sc("TO1", si, c));

                for (i, d) in Dir::ALL.iter().enumerate() {
                    tow0[si][0][c.x][c.z][i] = lp.bin(nm_sc_dir("TOW0", si, c, *d));
                    ro0[si][0][c.x][c.z][i] = lp.bin(nm_sc_dir("RO0", si, c, *d));
                }
            }
        }

        // output var ids for test assertions (assume single output, GroundDust)
        let mut out_dp0_vars = Vec::new();
        for si in 0..sc {
            let out = out_port.at;
            out_dp0_vars.push(dp0[si][0][out.x][out.z]);
        }

        Self {
            s,
            d0,
            tw0,
            r0,
            d1,
            t1,
            conn0,
            conn1,
            cross0,
            axis_ns0,
            axis_ew0,
            cross1,
            axis_ns1,
            axis_ew1,
            bp,
            dp0,
            dp1,
            to1,
            tow0,
            ro0,
            out_dp0_vars,
        }
    }
}

// ============================================================
// 6) RuleSetMinimal (네가 합의한 “최소 명세” 중심)
// ============================================================

pub struct RuleSetMinimal {
    /// no-feedback helper: only allow power to flow from lower order -> higher order
    pub acyclic_by_order: bool,
}

impl Default for RuleSetMinimal {
    fn default() -> Self {
        Self {
            acyclic_by_order: true,
        }
    }
}

impl RuleSetMinimal {
    pub fn add_all(&self, lp: &mut Lp, grid: &Grid, spec: &GateSpec, vars: &Vars) -> Result<()> {
        self.add_placement(lp, grid, vars)?;
        self.add_dust_shape(lp, grid, vars)?;
        self.add_state_domains(lp, grid, spec, vars)?;
        self.add_block_power(lp, grid, spec, vars)?;
        self.add_torch_rules(lp, grid, spec, vars)?;
        self.add_repeater_rules(lp, grid, spec, vars)?;
        self.add_dust_power(lp, grid, spec, vars)?;
        Ok(())
    }

    fn add_placement(&self, lp: &mut Lp, grid: &Grid, v: &Vars) -> Result<()> {
        for c in grid.iter_cells() {
            let s = v.s[c.x][c.z];
            let d0 = v.d0[c.x][c.z];
            let d1 = v.d1[c.x][c.z];
            let t1 = v.t1[c.x][c.z];

            // y=0 occupancy: S + D0 + sum(TW0) + sum(R0) <= 1
            let mut occ0 = vec![(1.0, s), (1.0, d0)];
            for i in 0..4 {
                occ0.push((1.0, v.tw0[c.x][c.z][i]));
                occ0.push((1.0, v.r0[c.x][c.z][i]));
            }
            lp.add_c(format!("occ0_{}_{}", c.x, c.z), occ0, Sense::Le, 1.0);

            // y=1 occupancy: D1 + T1 <= 1
            lp.add_c(
                format!("occ1_{}_{}", c.x, c.z),
                vec![(1.0, d1), (1.0, t1)],
                Sense::Le,
                1.0,
            );

            // top components need block
            lp.add_c(
                format!("d1_needs_s_{}_{}", c.x, c.z),
                vec![(1.0, d1), (-1.0, s)],
                Sense::Le,
                0.0,
            );
            lp.add_c(
                format!("t1_needs_s_{}_{}", c.x, c.z),
                vec![(1.0, t1), (-1.0, s)],
                Sense::Le,
                0.0,
            );

            // wall torch support: TW0[c,dir] <= S[c + opp(dir)]
            for (i, dir) in Dir::ALL.iter().enumerate() {
                let tw = v.tw0[c.x][c.z][i];
                if let Some(att) = grid.neighbor(c, dir.opp()) {
                    lp.add_c(
                        format!("tw_support_{}_{}_{}", c.x, c.z, dir.mc_facing()),
                        vec![(1.0, tw), (-1.0, v.s[att.x][att.z])],
                        Sense::Le,
                        0.0,
                    );
                } else {
                    // out of bounds -> must be 0
                    lp.eq(
                        format!("tw_oob0_{}_{}_{}", c.x, c.z, dir.mc_facing()),
                        tw,
                        0.0,
                    );
                }
            }

            // repeater support (we require air cell and assume floor exists) - no extra constraint besides occupancy.
        }
        Ok(())
    }

    /// Dust shape:
    /// - "default 4-way (Cross) if no candidate"
    /// - "if NS candidate exists => AxisNS forced"
    /// - else if EW candidate exists => AxisEW forced
    /// Conn derived:
    ///   ConnN = Cross + AxisNS ; ConnS = Cross + AxisNS ; ConnE = Cross + AxisEW ; ConnW = Cross + AxisEW
    fn add_dust_shape(&self, lp: &mut Lp, grid: &Grid, v: &Vars) -> Result<()> {
        self.add_dust_shape_layer(lp, grid, v, true)?;
        self.add_dust_shape_layer(lp, grid, v, false)?;
        Ok(())
    }

    fn add_dust_shape_layer(&self, lp: &mut Lp, grid: &Grid, v: &Vars, ground: bool) -> Result<()> {
        let (d, cross, axis_ns, axis_ew, conn) = if ground {
            (&v.d0, &v.cross0, &v.axis_ns0, &v.axis_ew0, &v.conn0)
        } else {
            (&v.d1, &v.cross1, &v.axis_ns1, &v.axis_ew1, &v.conn1)
        };

        for c in grid.iter_cells() {
            let d_here = d[c.x][c.z];
            let cross_here = cross[c.x][c.z];
            let ns_here = axis_ns[c.x][c.z];
            let ew_here = axis_ew[c.x][c.z];

            // one-hot: Cross + NS + EW = D
            lp.add_c(
                format!(
                    "shape_onehot_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![
                    (1.0, cross_here),
                    (1.0, ns_here),
                    (1.0, ew_here),
                    (-1.0, d_here),
                ],
                Sense::Eq,
                0.0,
            );
            // each <= D
            lp.add_c(
                format!(
                    "shape_cross_le_d_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, cross_here), (-1.0, d_here)],
                Sense::Le,
                0.0,
            );
            lp.add_c(
                format!(
                    "shape_ns_le_d_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, ns_here), (-1.0, d_here)],
                Sense::Le,
                0.0,
            );
            lp.add_c(
                format!(
                    "shape_ew_le_d_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, ew_here), (-1.0, d_here)],
                Sense::Le,
                0.0,
            );

            // Candidate vars per dir
            let mut cand = [0usize; 4];
            for (i, dir) in Dir::ALL.iter().enumerate() {
                cand[i] = lp.bin(format!(
                    "Cand{}_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z,
                    dir.mc_facing()
                ));
                if let Some(n) = grid.neighbor(c, *dir) {
                    // Cand >= (neighbor dust on either layer) OR (neighbor standing torch) OR (neighbor wall torch any dir) OR (neighbor repeater any dir)
                    // lower bounds:
                    lp.add_c(
                        format!(
                            "cand_ge_d0_{}_{}_{}_{}",
                            if ground { "0" } else { "1" },
                            c.x,
                            c.z,
                            dir.mc_facing()
                        ),
                        vec![(1.0, cand[i]), (-1.0, v.d0[n.x][n.z])],
                        Sense::Ge,
                        0.0,
                    );
                    lp.add_c(
                        format!(
                            "cand_ge_d1_{}_{}_{}_{}",
                            if ground { "0" } else { "1" },
                            c.x,
                            c.z,
                            dir.mc_facing()
                        ),
                        vec![(1.0, cand[i]), (-1.0, v.d1[n.x][n.z])],
                        Sense::Ge,
                        0.0,
                    );
                    lp.add_c(
                        format!(
                            "cand_ge_t1_{}_{}_{}_{}",
                            if ground { "0" } else { "1" },
                            c.x,
                            c.z,
                            dir.mc_facing()
                        ),
                        vec![(1.0, cand[i]), (-1.0, v.t1[n.x][n.z])],
                        Sense::Ge,
                        0.0,
                    );
                    for j in 0..4 {
                        lp.add_c(
                            format!(
                                "cand_ge_tw0_{}_{}_{}_{}_{}",
                                if ground { "0" } else { "1" },
                                c.x,
                                c.z,
                                dir.mc_facing(),
                                j
                            ),
                            vec![(1.0, cand[i]), (-1.0, v.tw0[n.x][n.z][j])],
                            Sense::Ge,
                            0.0,
                        );
                        lp.add_c(
                            format!(
                                "cand_ge_r0_{}_{}_{}_{}_{}",
                                if ground { "0" } else { "1" },
                                c.x,
                                c.z,
                                dir.mc_facing(),
                                j
                            ),
                            vec![(1.0, cand[i]), (-1.0, v.r0[n.x][n.z][j])],
                            Sense::Ge,
                            0.0,
                        );
                    }

                    // upper: Cand <= sum(all those)
                    let mut up = vec![
                        (1.0, cand[i]),
                        (-1.0, v.d0[n.x][n.z]),
                        (-1.0, v.d1[n.x][n.z]),
                        (-1.0, v.t1[n.x][n.z]),
                    ];
                    for j in 0..4 {
                        up.push((-1.0, v.tw0[n.x][n.z][j]));
                        up.push((-1.0, v.r0[n.x][n.z][j]));
                    }
                    lp.add_c(
                        format!(
                            "cand_up_{}_{}_{}_{}",
                            if ground { "0" } else { "1" },
                            c.x,
                            c.z,
                            dir.mc_facing()
                        ),
                        up,
                        Sense::Le,
                        0.0,
                    );
                } else {
                    lp.eq(
                        format!(
                            "cand_oob_{}_{}_{}_{}",
                            if ground { "0" } else { "1" },
                            c.x,
                            c.z,
                            dir.mc_facing()
                        ),
                        cand[i],
                        0.0,
                    );
                }
            }

            // CandNS = OR(CandN, CandS), CandEW = OR(CandE, CandW)
            let cand_ns = lp.bin(format!(
                "CandNS{}_{}_{}",
                if ground { "0" } else { "1" },
                c.x,
                c.z
            ));
            let cand_ew = lp.bin(format!(
                "CandEW{}_{}_{}",
                if ground { "0" } else { "1" },
                c.x,
                c.z
            ));

            // OR equivalence:
            // cand_ns >= candN, candS ; cand_ns <= candN + candS
            lp.add_c(
                format!(
                    "candns_ge_n_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, cand_ns), (-1.0, cand[0])],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "candns_ge_s_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, cand_ns), (-1.0, cand[2])],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "candns_up_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, cand_ns), (-1.0, cand[0]), (-1.0, cand[2])],
                Sense::Le,
                0.0,
            );

            lp.add_c(
                format!(
                    "candew_ge_e_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, cand_ew), (-1.0, cand[1])],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "candew_ge_w_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, cand_ew), (-1.0, cand[3])],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "candew_up_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, cand_ew), (-1.0, cand[1]), (-1.0, cand[3])],
                Sense::Le,
                0.0,
            );

            // Priority forcing:
            // AxisNS = CandNS
            lp.add_c(
                format!(
                    "ns_eq_candns_lo_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, ns_here), (-1.0, cand_ns)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "ns_eq_candns_up_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, ns_here), (-1.0, cand_ns)],
                Sense::Le,
                0.0,
            );

            // AxisEW = CandEW AND (NOT CandNS)
            // ew <= cand_ew
            lp.add_c(
                format!(
                    "ew_le_candew_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, ew_here), (-1.0, cand_ew)],
                Sense::Le,
                0.0,
            );
            // ew + cand_ns <= 1
            lp.add_c(
                format!(
                    "ew_plus_candns_le1_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, ew_here), (1.0, cand_ns)],
                Sense::Le,
                1.0,
            );
            // ew >= cand_ew - cand_ns  (ew - cand_ew + cand_ns >= 0)
            lp.add_c(
                format!(
                    "ew_ge_candew_minus_candns_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, ew_here), (-1.0, cand_ew), (1.0, cand_ns)],
                Sense::Ge,
                0.0,
            );

            // Conn derivation:
            // N/S: Cross or NS
            // E/W: Cross or EW
            let conn_n = conn[c.x][c.z][0];
            let conn_e = conn[c.x][c.z][1];
            let conn_s = conn[c.x][c.z][2];
            let conn_w = conn[c.x][c.z][3];

            // Conn <= D
            for (i, dir) in Dir::ALL.iter().enumerate() {
                lp.add_c(
                    format!(
                        "conn_le_d_{}_{}_{}_{}",
                        if ground { "0" } else { "1" },
                        c.x,
                        c.z,
                        dir.mc_facing()
                    ),
                    vec![(1.0, conn[c.x][c.z][i]), (-1.0, d_here)],
                    Sense::Le,
                    0.0,
                );
            }

            // N: >= cross, >= ns, <= cross + ns
            lp.add_c(
                format!(
                    "connn_ge_cross_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_n), (-1.0, cross_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "connn_ge_ns_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_n), (-1.0, ns_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "connn_up_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_n), (-1.0, cross_here), (-1.0, ns_here)],
                Sense::Le,
                0.0,
            );

            // S
            lp.add_c(
                format!(
                    "conns_ge_cross_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_s), (-1.0, cross_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "conns_ge_ns_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_s), (-1.0, ns_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "conns_up_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_s), (-1.0, cross_here), (-1.0, ns_here)],
                Sense::Le,
                0.0,
            );

            // E
            lp.add_c(
                format!(
                    "conne_ge_cross_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_e), (-1.0, cross_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "conne_ge_ew_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_e), (-1.0, ew_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "conne_up_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_e), (-1.0, cross_here), (-1.0, ew_here)],
                Sense::Le,
                0.0,
            );

            // W
            lp.add_c(
                format!(
                    "connw_ge_cross_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_w), (-1.0, cross_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "connw_ge_ew_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_w), (-1.0, ew_here)],
                Sense::Ge,
                0.0,
            );
            lp.add_c(
                format!(
                    "connw_up_{}_{}_{}",
                    if ground { "0" } else { "1" },
                    c.x,
                    c.z
                ),
                vec![(1.0, conn_w), (-1.0, cross_here), (-1.0, ew_here)],
                Sense::Le,
                0.0,
            );
        }

        Ok(())
    }

    fn add_state_domains(&self, lp: &mut Lp, grid: &Grid, spec: &GateSpec, v: &Vars) -> Result<()> {
        let sc = spec.scenario_count();
        for si in 0..sc {
            for c in grid.iter_cells() {
                // bp <= S
                lp.add_c(
                    format!("dom_bp_{}_{}_{}", si, c.x, c.z),
                    vec![(1.0, v.bp[si][0][c.x][c.z]), (-1.0, v.s[c.x][c.z])],
                    Sense::Le,
                    0.0,
                );
                // dp0 <= D0
                lp.add_c(
                    format!("dom_dp0_{}_{}_{}", si, c.x, c.z),
                    vec![(1.0, v.dp0[si][0][c.x][c.z]), (-1.0, v.d0[c.x][c.z])],
                    Sense::Le,
                    0.0,
                );
                // dp1 <= D1
                lp.add_c(
                    format!("dom_dp1_{}_{}_{}", si, c.x, c.z),
                    vec![(1.0, v.dp1[si][0][c.x][c.z]), (-1.0, v.d1[c.x][c.z])],
                    Sense::Le,
                    0.0,
                );
                // to1 <= T1
                lp.add_c(
                    format!("dom_to1_{}_{}_{}", si, c.x, c.z),
                    vec![(1.0, v.to1[si][0][c.x][c.z]), (-1.0, v.t1[c.x][c.z])],
                    Sense::Le,
                    0.0,
                );
                for (i, dir) in Dir::ALL.iter().enumerate() {
                    lp.add_c(
                        format!("dom_tow0_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                        vec![
                            (1.0, v.tow0[si][0][c.x][c.z][i]),
                            (-1.0, v.tw0[c.x][c.z][i]),
                        ],
                        Sense::Le,
                        0.0,
                    );
                    lp.add_c(
                        format!("dom_ro0_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                        vec![(1.0, v.ro0[si][0][c.x][c.z][i]), (-1.0, v.r0[c.x][c.z][i])],
                        Sense::Le,
                        0.0,
                    );
                }
            }
        }
        Ok(())
    }

    /// Minimal (strong) simplification:
    ///   BP = DP1   (block is powered iff its top dust is on)
    fn add_block_power(&self, lp: &mut Lp, grid: &Grid, spec: &GateSpec, v: &Vars) -> Result<()> {
        let sc = spec.scenario_count();
        for si in 0..sc {
            for c in grid.iter_cells() {
                let bp = v.bp[si][0][c.x][c.z];
                let dp1 = v.dp1[si][0][c.x][c.z];
                // bp - dp1 = 0
                lp.add_c(
                    format!("bp_eq_dp1_{}_{}_{}", si, c.x, c.z),
                    vec![(1.0, bp), (-1.0, dp1)],
                    Sense::Eq,
                    0.0,
                );
            }
        }
        Ok(())
    }

    /// Torch rules:
    /// standing: TO1 = T1 ∧ ¬BP
    /// wall:     TOW0 = TW0 ∧ ¬BP(attached)
    /// (and attached block is not powered by the torch -> we enforce that by *not* including it as a source in BP)
    fn add_torch_rules(&self, lp: &mut Lp, grid: &Grid, spec: &GateSpec, v: &Vars) -> Result<()> {
        let sc = spec.scenario_count();
        for si in 0..sc {
            for c in grid.iter_cells() {
                let to1 = v.to1[si][0][c.x][c.z];
                let t1 = v.t1[c.x][c.z];
                let bp = v.bp[si][0][c.x][c.z];

                // to1 <= t1 is domain
                // to1 + bp <= 1
                lp.add_c(
                    format!("to1_off_if_bp_{}_{}_{}", si, c.x, c.z),
                    vec![(1.0, to1), (1.0, bp)],
                    Sense::Le,
                    1.0,
                );
                // to1 >= t1 - bp  -> to1 - t1 + bp >= 0
                lp.add_c(
                    format!("to1_on_if_t1_and_not_bp_{}_{}_{}", si, c.x, c.z),
                    vec![(1.0, to1), (-1.0, t1), (1.0, bp)],
                    Sense::Ge,
                    0.0,
                );

                // wall torches
                for (i, dir) in Dir::ALL.iter().enumerate() {
                    let tow = v.tow0[si][0][c.x][c.z][i];
                    let tw = v.tw0[c.x][c.z][i];
                    if let Some(att) = grid.neighbor(c, dir.opp()) {
                        let bp_att = v.bp[si][0][att.x][att.z];
                        // tow + bp_att <= 1
                        lp.add_c(
                            format!(
                                "tow_off_if_bpatt_{}_{}_{}_{}",
                                si,
                                c.x,
                                c.z,
                                dir.mc_facing()
                            ),
                            vec![(1.0, tow), (1.0, bp_att)],
                            Sense::Le,
                            1.0,
                        );
                        // tow >= tw - bp_att  -> tow - tw + bp_att >= 0
                        lp.add_c(
                            format!(
                                "tow_on_if_tw_and_not_bpatt_{}_{}_{}_{}",
                                si,
                                c.x,
                                c.z,
                                dir.mc_facing()
                            ),
                            vec![(1.0, tow), (-1.0, tw), (1.0, bp_att)],
                            Sense::Ge,
                            0.0,
                        );
                    } else {
                        // already forced tw=0, tow<=tw -> tow must be 0, but keep explicit:
                        lp.eq(
                            format!("tow_oob0_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                            tow,
                            0.0,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Repeater (combinational simplification):
    /// - input is DP0(back) (only)
    /// - output is RO0 = R0 ∧ DP0(back)
    fn add_repeater_rules(
        &self,
        lp: &mut Lp,
        grid: &Grid,
        spec: &GateSpec,
        v: &Vars,
    ) -> Result<()> {
        let sc = spec.scenario_count();
        for si in 0..sc {
            for c in grid.iter_cells() {
                for (i, dir) in Dir::ALL.iter().enumerate() {
                    let r = v.r0[c.x][c.z][i];
                    let ro = v.ro0[si][0][c.x][c.z][i];

                    // back cell (input)
                    let back = match grid.neighbor(c, dir.opp()) {
                        Some(b) => b,
                        None => {
                            lp.eq(
                                format!(
                                    "rep_back_oob_ro0_{}_{}_{}_{}",
                                    si,
                                    c.x,
                                    c.z,
                                    dir.mc_facing()
                                ),
                                ro,
                                0.0,
                            );
                            continue;
                        }
                    };
                    let dp_back = v.dp0[si][0][back.x][back.z];

                    // ro = r ∧ dp_back
                    // ro <= r (domain already)
                    // ro <= dp_back
                    lp.add_c(
                        format!("ro_le_dpback_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                        vec![(1.0, ro), (-1.0, dp_back)],
                        Sense::Le,
                        0.0,
                    );
                    // ro >= r + dp_back - 1   => ro - r - dp_back >= -1
                    lp.add_c(
                        format!("ro_ge_and_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                        vec![(1.0, ro), (-1.0, r), (-1.0, dp_back)],
                        Sense::Ge,
                        -1.0,
                    );
                }
            }
        }
        Ok(())
    }

    /// Dust power (boolean):
    /// DP = D ∧ OR(incoming sources)
    ///
    /// sources include:
    /// - incoming from neighbor dust along Conn (same layer), but only from lower-order neighbors if acyclic_by_order
    /// - wall torch output from self/adjacent cells
    /// - standing torch output from same/adjacent cells
    /// - repeater output from back cell that points to this cell
    ///
    /// For input ports: we "pin" DP directly and skip equation at that cell.
    fn add_dust_power(&self, lp: &mut Lp, grid: &Grid, spec: &GateSpec, v: &Vars) -> Result<()> {
        let sc = spec.scenario_count();

        // Build pinned DP sets (only TopDust inputs for now)
        let mut pinned_top: HashMap<Cell, Vec<u8>> = HashMap::new();
        for (si, (in_bits, _out_bits)) in spec.table.iter().enumerate() {
            for (pi, p) in spec.inputs.iter().enumerate() {
                if matches!(p.kind, SignalKind::TopDust) {
                    pinned_top.entry(p.at).or_default().resize(sc, 0);
                    pinned_top.get_mut(&p.at).unwrap()[si] = in_bits[pi];
                }
            }
        }

        for si in 0..sc {
            // pin inputs
            for p in &spec.inputs {
                match p.kind {
                    SignalKind::TopDust => {
                        let bit = spec.table[si].0
                            [spec.inputs.iter().position(|pp| pp.name == p.name).unwrap()];
                        lp.eq(
                            format!("pin_in_top_{}_{}_{}", si, p.at.x, p.at.z),
                            v.dp1[si][0][p.at.x][p.at.z],
                            bit as f64,
                        );
                    }
                    SignalKind::GroundDust => {
                        let bit = spec.table[si].0
                            [spec.inputs.iter().position(|pp| pp.name == p.name).unwrap()];
                        lp.eq(
                            format!("pin_in_ground_{}_{}_{}", si, p.at.x, p.at.z),
                            v.dp0[si][0][p.at.x][p.at.z],
                            bit as f64,
                        );
                    }
                }
            }

            // DP0 equations
            for c in grid.iter_cells() {
                let dp = v.dp0[si][0][c.x][c.z];
                let d = v.d0[c.x][c.z];

                // if this is pinned ground input, skip equation (already fixed)
                let is_pinned_ground = spec
                    .inputs
                    .iter()
                    .any(|p| p.kind == SignalKind::GroundDust && p.at == c);
                if is_pinned_ground {
                    continue;
                }

                let mut sources: Vec<usize> = Vec::new();

                // incoming from neighbor ground dust (Conn both sides), oriented by order if enabled
                for (i, dir) in Dir::ALL.iter().enumerate() {
                    if let Some(n) = grid.neighbor(c, *dir) {
                        if self.acyclic_by_order && grid.order(n) >= grid.order(c) {
                            continue;
                        }
                        let pow =
                            lp.bin(format!("Pow00_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()));
                        let dpn = v.dp0[si][0][n.x][n.z];
                        let c_from_n = v.conn0[n.x][n.z][dir.opp() as usize];
                        let n_from_c = v.conn0[c.x][c.z][i];

                        // pow = dpn ∧ Conn(n->c) ∧ Conn(c->n)
                        // pow <= dpn
                        lp.add_c(
                            format!("pow00_le_dpn_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                            vec![(1.0, pow), (-1.0, dpn)],
                            Sense::Le,
                            0.0,
                        );
                        // pow <= Conn(n->c)
                        lp.add_c(
                            format!(
                                "pow00_le_conn_n2c_{}_{}_{}_{}",
                                si,
                                c.x,
                                c.z,
                                dir.mc_facing()
                            ),
                            vec![(1.0, pow), (-1.0, c_from_n)],
                            Sense::Le,
                            0.0,
                        );
                        // pow <= Conn(c->n)
                        lp.add_c(
                            format!(
                                "pow00_le_conn_c2n_{}_{}_{}_{}",
                                si,
                                c.x,
                                c.z,
                                dir.mc_facing()
                            ),
                            vec![(1.0, pow), (-1.0, n_from_c)],
                            Sense::Le,
                            0.0,
                        );
                        // pow >= dpn + Conn(n->c) + Conn(c->n) - 2
                        lp.add_c(
                            format!("pow00_ge_and_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                            vec![(1.0, pow), (-1.0, dpn), (-1.0, c_from_n), (-1.0, n_from_c)],
                            Sense::Ge,
                            -2.0,
                        );

                        sources.push(pow);
                    }
                }

                // wall torches at self / neighbors
                for src in neighbors_plus_self(grid, c) {
                    for j in 0..4 {
                        sources.push(v.tow0[si][0][src.x][src.z][j]);
                    }
                }

                // standing torches at self / neighbors (coarse but useful)
                for src in neighbors_plus_self(grid, c) {
                    sources.push(v.to1[si][0][src.x][src.z]);
                }

                // repeater outputs that point into this cell
                for (i, dir) in Dir::ALL.iter().enumerate() {
                    // if there is repeater at back cell facing dir, front is c
                    if let Some(back) = grid.neighbor(c, dir.opp()) {
                        sources.push(v.ro0[si][0][back.x][back.z][i]);
                    }
                }

                add_or_equiv(
                    lp,
                    format!("dp0_or_{}_{}_{}", si, c.x, c.z),
                    dp,
                    d,
                    &sources,
                );
            }

            // DP1 equations
            for c in grid.iter_cells() {
                let dp = v.dp1[si][0][c.x][c.z];
                let d = v.d1[c.x][c.z];

                // pinned top input -> skip equation
                let is_pinned_top = spec
                    .inputs
                    .iter()
                    .any(|p| p.kind == SignalKind::TopDust && p.at == c);
                if is_pinned_top {
                    continue;
                }

                let mut sources: Vec<usize> = Vec::new();

                // incoming from neighbor top dust (Conn both sides), oriented by order if enabled
                for (i, dir) in Dir::ALL.iter().enumerate() {
                    if let Some(n) = grid.neighbor(c, *dir) {
                        if self.acyclic_by_order && grid.order(n) >= grid.order(c) {
                            continue;
                        }
                        let pow =
                            lp.bin(format!("Pow11_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()));
                        let dpn = v.dp1[si][0][n.x][n.z];
                        let c_from_n = v.conn1[n.x][n.z][dir.opp() as usize];
                        let n_from_c = v.conn1[c.x][c.z][i];

                        // pow = dpn ∧ Conn(n->c) ∧ Conn(c->n)
                        lp.add_c(
                            format!("pow11_le_dpn_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                            vec![(1.0, pow), (-1.0, dpn)],
                            Sense::Le,
                            0.0,
                        );
                        lp.add_c(
                            format!(
                                "pow11_le_conn_n2c_{}_{}_{}_{}",
                                si,
                                c.x,
                                c.z,
                                dir.mc_facing()
                            ),
                            vec![(1.0, pow), (-1.0, c_from_n)],
                            Sense::Le,
                            0.0,
                        );
                        lp.add_c(
                            format!(
                                "pow11_le_conn_c2n_{}_{}_{}_{}",
                                si,
                                c.x,
                                c.z,
                                dir.mc_facing()
                            ),
                            vec![(1.0, pow), (-1.0, n_from_c)],
                            Sense::Le,
                            0.0,
                        );
                        lp.add_c(
                            format!("pow11_ge_and_{}_{}_{}_{}", si, c.x, c.z, dir.mc_facing()),
                            vec![(1.0, pow), (-1.0, dpn), (-1.0, c_from_n), (-1.0, n_from_c)],
                            Sense::Ge,
                            -2.0,
                        );

                        sources.push(pow);
                    }
                }

                // wall / standing torches around (coarse)
                for src in neighbors_plus_self(grid, c) {
                    for j in 0..4 {
                        sources.push(v.tow0[si][0][src.x][src.z][j]);
                    }
                }
                for src in neighbors_plus_self(grid, c) {
                    sources.push(v.to1[si][0][src.x][src.z]);
                }

                add_or_equiv(
                    lp,
                    format!("dp1_or_{}_{}_{}", si, c.x, c.z),
                    dp,
                    d,
                    &sources,
                );
            }
        }

        Ok(())
    }
}

fn neighbors_plus_self(grid: &Grid, c: Cell) -> Vec<Cell> {
    let mut out = vec![c];
    for d in Dir::ALL {
        if let Some(n) = grid.neighbor(c, d) {
            out.push(n);
        }
    }
    out
}

// Helper: map Dir index consistently with Dir::ALL order
trait DirIndex {
    fn as_usize(self) -> usize;
}
impl DirIndex for Dir {
    fn as_usize(self) -> usize {
        match self {
            Dir::North => 0,
            Dir::East => 1,
            Dir::South => 2,
            Dir::West => 3,
        }
    }
}
fn usize_dir(d: Dir) -> usize {
    d.as_usize()
}
trait OppIdx {
    fn opp_idx(self) -> usize;
}
impl OppIdx for Dir {
    fn opp_idx(self) -> usize {
        self.opp().as_usize()
    }
}
// small helper to keep earlier expression readable
trait AsUsizeDir {
    fn as_usize_dir(self) -> usize;
}
impl AsUsizeDir for Dir {
    fn as_usize_dir(self) -> usize {
        self.as_usize()
    }
}

// DP = D ∧ OR(sources)
fn add_or_equiv(lp: &mut Lp, name: String, y: usize, gate: usize, sources: &[usize]) {
    // y <= gate (domain also provides, but keep)
    lp.add_c(
        format!("{}_y_le_gate", name),
        vec![(1.0, y), (-1.0, gate)],
        Sense::Le,
        0.0,
    );

    // y <= sum(sources)
    let mut up = vec![(1.0, y)];
    for &sv in sources {
        up.push((-1.0, sv));
    }
    lp.add_c(format!("{}_y_le_sum", name), up, Sense::Le, 0.0);

    // y >= source + gate - 1  -> y - source - gate >= -1
    for (i, &sv) in sources.iter().enumerate() {
        lp.add_c(
            format!("{}_y_ge_src{}_and_gate", name, i),
            vec![(1.0, y), (-1.0, sv), (-1.0, gate)],
            Sense::Ge,
            -1.0,
        );
    }
}

// ============================================================
// 7) Synthesis (build LP + solve + emit placements)
// ============================================================

#[derive(Clone, Debug)]
pub struct Placement {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub block: String,
}

#[derive(Clone, Debug)]
pub struct SynthResult {
    pub placements: Vec<Placement>,
    pub solution: Solution,
    pub lp_path: PathBuf,
    pub sol_path: PathBuf,
}

pub fn synthesize_truth_table_gate(
    grid: Grid,
    spec: GateSpec,
    scip_path: &str,
    out_dir: impl AsRef<Path>,
    origin_x: i32,
    origin_y: i32,
    origin_z: i32,
) -> Result<SynthResult> {
    if spec.outputs.len() != 1 {
        bail!("this demo code assumes single-output gate");
    }
    let out_port = spec.outputs[0].clone();

    let mut lp = Lp::new();
    let vars = Vars::alloc(&mut lp, &grid, spec.scenario_count(), &out_port);
    let rules = RuleSetMinimal::default();

    // Force port placements:
    // - inputs are TopDust => require S=1 and D1=1 at input cells
    for p in &spec.inputs {
        match p.kind {
            SignalKind::TopDust => {
                let c = p.at;
                lp.eq(format!("force_S_in_{}_{}", c.x, c.z), vars.s[c.x][c.z], 1.0);
                lp.eq(
                    format!("force_D1_in_{}_{}", c.x, c.z),
                    vars.d1[c.x][c.z],
                    1.0,
                );
            }
            SignalKind::GroundDust => {
                let c = p.at;
                lp.eq(
                    format!("force_S0_in_{}_{}", c.x, c.z),
                    vars.s[c.x][c.z],
                    0.0,
                );
                lp.eq(
                    format!("force_D0_in_{}_{}", c.x, c.z),
                    vars.d0[c.x][c.z],
                    1.0,
                );
            }
        }
    }

    // - output is GroundDust => require S=0, D0=1 at output
    {
        let c = out_port.at;
        lp.eq(
            format!("force_S_out_{}_{}", c.x, c.z),
            vars.s[c.x][c.z],
            0.0,
        );
        lp.eq(
            format!("force_D0_out_{}_{}", c.x, c.z),
            vars.d0[c.x][c.z],
            1.0,
        );
    }

    // Add physics constraints
    rules.add_all(&mut lp, &grid, &spec, &vars)?;

    // Truth table constraints (output only; inputs are pinned in dust-power step)
    for (si, (_in_bits, out_bits)) in spec.table.iter().enumerate() {
        let want = out_bits[0] as f64;
        let out = out_port.at;
        lp.eq(
            format!("truth_out_{}_{}_{}", si, out.x, out.z),
            vars.dp0[si][0][out.x][out.z],
            want,
        );
    }

    // Objective: small circuit
    for c in grid.iter_cells() {
        lp.add_obj(0.2, vars.s[c.x][c.z]);
        lp.add_obj(1.0, vars.d0[c.x][c.z]);
        lp.add_obj(1.0, vars.d1[c.x][c.z]);
        lp.add_obj(5.0, vars.t1[c.x][c.z]);
        for i in 0..4 {
            lp.add_obj(5.0, vars.tw0[c.x][c.z][i]);
            lp.add_obj(5.0, vars.r0[c.x][c.z][i]);
        }
    }

    // Write / solve
    let out_dir = out_dir.as_ref().to_path_buf();
    fs::create_dir_all(&out_dir)?;

    let lp_path = out_dir.join("gate.lp");
    let sol_path = out_dir.join("gate.sol");
    lp.write_lp(&lp_path)?;

    run_scip(scip_path, &lp_path, &sol_path)?;
    let sol = parse_scip_sol(&sol_path)?;

    // Emit placements
    let mut placements = Vec::new();

    // Blocks / dust / torches / repeaters
    for c in grid.iter_cells() {
        let wx = origin_x + c.x as i32;
        let wz = origin_z + c.z as i32;

        if sol.get_bin(lp.var_name(vars.s[c.x][c.z])) {
            placements.push(Placement {
                x: wx,
                y: origin_y,
                z: wz,
                block: "minecraft:stone".to_string(),
            });
        }
        if sol.get_bin(lp.var_name(vars.d0[c.x][c.z])) {
            placements.push(Placement {
                x: wx,
                y: origin_y,
                z: wz,
                block: "minecraft:redstone_wire".to_string(),
            });
        }
        if sol.get_bin(lp.var_name(vars.d1[c.x][c.z])) {
            placements.push(Placement {
                x: wx,
                y: origin_y + 1,
                z: wz,
                block: "minecraft:redstone_wire".to_string(),
            });
        }
        if sol.get_bin(lp.var_name(vars.t1[c.x][c.z])) {
            placements.push(Placement {
                x: wx,
                y: origin_y + 1,
                z: wz,
                block: "minecraft:redstone_torch".to_string(),
            });
        }
        for (i, dir) in Dir::ALL.iter().enumerate() {
            if sol.get_bin(lp.var_name(vars.tw0[c.x][c.z][i])) {
                placements.push(Placement {
                    x: wx,
                    y: origin_y,
                    z: wz,
                    block: format!("minecraft:redstone_wall_torch[facing={}]", dir.mc_facing()),
                });
            }
            if sol.get_bin(lp.var_name(vars.r0[c.x][c.z][i])) {
                placements.push(Placement {
                    x: wx,
                    y: origin_y,
                    z: wz,
                    block: format!("minecraft:repeater[facing={}]", dir.mc_facing()),
                });
            }
        }
    }

    // Add levers for inputs (visual convenience; not part of ILP)
    // Place lever on west side of input block (x-1), facing east. (You can customize later.)
    for p in &spec.inputs {
        if matches!(p.kind, SignalKind::TopDust) {
            let c = p.at;
            let lx = origin_x + c.x as i32 - 1;
            let lz = origin_z + c.z as i32;
            placements.push(Placement {
                x: lx,
                y: origin_y,
                z: lz,
                block: "minecraft:lever[face=wall,facing=east]".to_string(),
            });
        }
    }

    placements.sort_by_key(|p| (p.y, p.z, p.x));

    Ok(SynthResult {
        placements,
        solution: sol,
        lp_path,
        sol_path,
    })
}

// ============================================================
// 8) Tests: NOT / AND / XOR synthesis (prints /setblock)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn scip_path() -> String {
        std::env::var("SCIP_PATH").unwrap_or_else(|_| "scip".to_string())
    }

    fn print_setblock(result: &SynthResult) {
        println!("# LP  : {}", result.lp_path.display());
        println!("# SOL : {}", result.sol_path.display());
        for p in &result.placements {
            println!("/setblock {} {} {} {}", p.x, p.y, p.z, p.block);
        }
    }

    #[test]
    fn synth_not_gate() -> Result<()> {
        let grid = Grid::new(9, 7);
        let spec = GateSpec::from_kind(GateKind::Not, &grid);

        println!("spec: {:?}", spec);

        let out_dir = std::env::temp_dir().join("redstone_ilp_not");
        let r = synthesize_truth_table_gate(grid, spec.clone(), &scip_path(), &out_dir, 0, 0, 0)?;

        // sanity: ensure output truth table satisfied (by reading solution vars)
        for (si, (_inb, outb)) in spec.table.iter().enumerate() {
            let out_var_name = format!(
                "DP0_{}_{}_{}",
                si, spec.outputs[0].at.x, spec.outputs[0].at.z
            );
            assert_eq!(r.solution.get_bin(&out_var_name), outb[0] == 1);
        }

        print_setblock(&r);
        Ok(())
    }

    #[test]
    fn synth_and_gate() -> Result<()> {
        let grid = Grid::new(11, 9);
        let spec = GateSpec::from_kind(GateKind::And, &grid);

        let out_dir = std::env::temp_dir().join("redstone_ilp_and");
        let r = synthesize_truth_table_gate(grid, spec.clone(), &scip_path(), &out_dir, 0, 0, 0)?;

        for (si, (_inb, outb)) in spec.table.iter().enumerate() {
            let out_var_name = format!(
                "DP0_{}_{}_{}",
                si, spec.outputs[0].at.x, spec.outputs[0].at.z
            );
            assert_eq!(r.solution.get_bin(&out_var_name), outb[0] == 1);
        }

        print_setblock(&r);
        Ok(())
    }

    #[test]
    fn synth_xor_gate() -> Result<()> {
        let grid = Grid::new(13, 9);
        let spec = GateSpec::from_kind(GateKind::Xor, &grid);

        let out_dir = std::env::temp_dir().join("redstone_ilp_xor");
        let r = synthesize_truth_table_gate(grid, spec.clone(), &scip_path(), &out_dir, 0, 0, 0)?;

        for (si, (_inb, outb)) in spec.table.iter().enumerate() {
            let out_var_name = format!(
                "DP0_{}_{}_{}",
                si, spec.outputs[0].at.x, spec.outputs[0].at.z
            );
            assert_eq!(r.solution.get_bin(&out_var_name), outb[0] == 1);
        }

        print_setblock(&r);
        Ok(())
    }
}
