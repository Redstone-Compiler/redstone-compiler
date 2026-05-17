use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use eyre::Context;
use structopt::StructOpt;

use crate::nbt::NBTRoot;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(StructOpt, Debug)]
#[structopt(
    name = "solver",
    about = "NOT synthesis ILP (combinational/no time, wall-torch + top/ground dust)"
)]
struct Args {
    /// Grid X size (minecraft x)
    #[structopt(long, default_value = "9")]
    x: usize,
    /// Grid Z size (minecraft z)
    #[structopt(long, default_value = "7")]
    z: usize,

    /// Origin for setblock commands (this is the base y)
    #[structopt(long, default_value = "0")]
    origin_x: i32,
    #[structopt(long, default_value = "64")]
    origin_y: i32,
    #[structopt(long, default_value = "0")]
    origin_z: i32,

    /// Path to scip executable
    #[structopt(long, default_value = "scip")]
    scip: String,

    /// Run scip to solve
    #[structopt(long)]
    run_scip: Option<bool>,

    /// Emit /setblock commands
    #[structopt(long)]
    emit_commands: Option<bool>,

    /// Output directory
    #[structopt(long, default_value = ".")]
    out_dir: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Cell {
    x: usize,
    z: usize,
}

fn in_bounds(x: i32, z: i32, xs: usize, zs: usize) -> bool {
    x >= 0 && z >= 0 && (x as usize) < xs && (z as usize) < zs
}

fn neighbors4(c: Cell, xs: usize, zs: usize) -> Vec<Cell> {
    let mut out = Vec::new();
    for (dx, dz) in [(0, -1), (1, 0), (0, 1), (-1, 0)] {
        let nx = c.x as i32 + dx;
        let nz = c.z as i32 + dz;
        if in_bounds(nx, nz, xs, zs) {
            out.push(Cell {
                x: nx as usize,
                z: nz as usize,
            });
        }
    }
    out
}

/// fixed DAG order (acyclic): only allow influence from smaller ord -> bigger ord
fn ord(c: Cell, zs: usize) -> usize {
    c.x * zs + c.z
}

fn incoming_neighbors(c: Cell, xs: usize, zs: usize) -> Vec<Cell> {
    let oc = ord(c, zs);
    neighbors4(c, xs, zs)
        .into_iter()
        .filter(|n| ord(*n, zs) < oc)
        .collect()
}

// -------------------- LP core --------------------
#[derive(Clone, Debug)]
struct Constraint {
    name: String,
    terms: Vec<(f64, usize)>,
    sense: Sense,
    rhs: f64,
}

#[derive(Clone, Copy, Debug)]
enum Sense {
    Le,
    Ge,
    Eq,
}

#[derive(Default)]
struct Lp {
    var_names: Vec<String>,
    var_index: HashMap<String, usize>,
    is_binary: Vec<bool>,
    objective: Vec<(f64, usize)>,
    constraints: Vec<Constraint>,
    minimize: bool,
}

impl Lp {
    fn new() -> Self {
        Self {
            minimize: true,
            ..Default::default()
        }
    }

    fn bin(&mut self, name: String) -> usize {
        if let Some(&i) = self.var_index.get(&name) {
            return i;
        }
        let i = self.var_names.len();
        self.var_names.push(name.clone());
        self.var_index.insert(name, i);
        self.is_binary.push(true);
        i
    }

    fn add_obj(&mut self, coef: f64, var: usize) {
        self.objective.push((coef, var));
    }

    fn add_c(&mut self, name: String, terms: Vec<(f64, usize)>, sense: Sense, rhs: f64) {
        self.constraints.push(Constraint {
            name,
            terms,
            sense,
            rhs,
        });
    }

    fn write_lp(&self, path: &Path) -> eyre::Result<()> {
        let mut s = String::new();
        s.push_str(if self.minimize {
            "Minimize\n obj: "
        } else {
            "Maximize\n obj: "
        });

        if self.objective.is_empty() {
            s.push_str("0\n");
        } else {
            s.push_str(&format_linexpr(&self.objective, &self.var_names));
            s.push('\n');
        }

        s.push_str("Subject To\n");
        for c in &self.constraints {
            s.push_str(" ");
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
                s.push_str(" ");
                s.push_str(name);
                s.push('\n');
            }
        }
        s.push_str("End\n");
        fs::write(path, s)?;
        Ok(())
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

#[derive(Debug)]
struct Solution {
    vals: HashMap<String, f64>,
}
impl Solution {
    fn get_bin(&self, name: &str) -> bool {
        self.vals.get(name).copied().unwrap_or(0.0) > 0.5
    }
    fn has_any(&self) -> bool {
        !self.vals.is_empty()
    }
}

fn parse_scip_sol(path: &Path) -> eyre::Result<Solution> {
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

fn run_scip(scip_path: &str, lp_path: &Path, sol_path: &Path) -> eyre::Result<()> {
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

// -------------------- Variable naming --------------------
// Base block at y=oy
fn v_s(c: Cell) -> String {
    format!("S_{}_{}", c.x, c.z)
}

// Dust placement:
// - D1: top dust at y=oy+1 (requires S under)
// - D0: ground dust at y=oy (assume solid floor at y=oy-1)
fn v_d0(c: Cell) -> String {
    format!("D0_{}_{}", c.x, c.z)
}
fn v_d1(c: Cell) -> String {
    format!("D1_{}_{}", c.x, c.z)
}

// States per scenario s in {0,1}:
fn v_dp0(s: usize, c: Cell) -> String {
    format!("DP0_{}_{}_{}", s, c.x, c.z)
}
fn v_dp1(s: usize, c: Cell) -> String {
    format!("DP1_{}_{}_{}", s, c.x, c.z)
}
fn v_bp(s: usize, c: Cell) -> String {
    format!("BP_{}_{}_{}", s, c.x, c.z)
}

// Wall torch placement/state.
// Torch is a block at y=oy in an "air cell" (no S, no D0).
fn v_tw(c: Cell, dir: usize) -> String {
    format!("TW_{}_{}_{}", c.x, c.z, dir)
}
fn v_to(s: usize, c: Cell, dir: usize) -> String {
    format!("TO_{}_{}_{}_{}", s, c.x, c.z, dir)
}

// -------------------- wall torch dirs --------------------
// IMPORTANT: minecraft wall torch "facing" == direction of supporting block (the wall it's attached to).
#[derive(Clone, Copy)]
struct TorchDir {
    dx: i32,
    dz: i32,
    facing: &'static str,
}
const TD: [TorchDir; 4] = [
    TorchDir {
        dx: 0,
        dz: -1,
        facing: "north",
    }, // support north
    TorchDir {
        dx: 1,
        dz: 0,
        facing: "east",
    }, // support east
    TorchDir {
        dx: 0,
        dz: 1,
        facing: "south",
    }, // support south
    TorchDir {
        dx: -1,
        dz: 0,
        facing: "west",
    }, // support west
];

fn support_cell(torch: Cell, dir: usize, xs: usize, zs: usize) -> Option<Cell> {
    let d = TD[dir];
    let sx = torch.x as i32 + d.dx;
    let sz = torch.z as i32 + d.dz;
    if in_bounds(sx, sz, xs, zs) {
        Some(Cell {
            x: sx as usize,
            z: sz as usize,
        })
    } else {
        None
    }
}

// -------------------- Minecraft placements --------------------
#[derive(Clone, Debug)]
struct Placement {
    x: i32,
    y: i32,
    z: i32,
    block: String,
}

fn parse_block_string(block_str: &str) -> eyre::Result<(BlockKind, Direction)> {
    let (name, props_str) = if let Some(bracket_pos) = block_str.find('[') {
        (
            &block_str[..bracket_pos],
            Some(&block_str[bracket_pos + 1..block_str.len() - 1]),
        )
    } else {
        (block_str, None)
    };

    let mut facing = None;
    let mut face = None;
    if let Some(props) = props_str {
        for prop in props.split(',') {
            let parts: Vec<&str> = prop.split('=').collect();
            if parts.len() == 2 {
                match parts[0].trim() {
                    "facing" => facing = Some(parts[1].trim()),
                    "face" => face = Some(parts[1].trim()),
                    _ => {}
                }
            }
        }
    }

    let mc_facing_to_dir = |f: &str| match f {
        "east" => Direction::North,
        "west" => Direction::South,
        "south" => Direction::East,
        "north" => Direction::West,
        _ => Direction::None,
    };

    let (kind, dir) = match name {
        "minecraft:air" => (BlockKind::Air, Direction::None),
        "minecraft:stone" | "minecraft:cobblestone" => (
            BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            Direction::None,
        ),
        "minecraft:redstone_lamp" => (
            BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            Direction::None,
        ),
        "minecraft:redstone_wire" => (
            BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            Direction::None,
        ),
        "minecraft:redstone_wall_torch" => {
            let dir = if let Some(facing_val) = facing {
                mc_facing_to_dir(facing_val)
            } else {
                Direction::None
            };
            (BlockKind::Torch { is_on: false }, dir)
        }
        "minecraft:lever" => {
            let dir = if let Some(face_val) = face {
                if face_val == "wall" {
                    if let Some(facing_val) = facing {
                        mc_facing_to_dir(facing_val)
                    } else {
                        Direction::None
                    }
                } else {
                    Direction::None
                }
            } else {
                Direction::None
            };
            (BlockKind::Switch { is_on: false }, dir)
        }
        _ => eyre::bail!("Unknown block type: {}", name),
    };

    Ok((kind, dir))
}

fn placements_to_world3d(placements: &[Placement]) -> eyre::Result<World3D> {
    if placements.is_empty() {
        return Ok(World3D::new(DimSize(1, 1, 1)));
    }

    let min_x = placements.iter().map(|p| p.x).min().unwrap();
    let max_x = placements.iter().map(|p| p.x).max().unwrap();
    let min_y = placements.iter().map(|p| p.y).min().unwrap();
    let max_y = placements.iter().map(|p| p.y).max().unwrap();
    let min_z = placements.iter().map(|p| p.z).min().unwrap();
    let max_z = placements.iter().map(|p| p.z).max().unwrap();

    let size_x = (max_z - min_z + 1) as usize; // internal x <- mc z
    let size_y = (max_x - min_x + 1) as usize; // internal y <- mc x
    let size_z = (max_y - min_y + 1) as usize; // internal z <- mc y

    let mut world = World3D::new(DimSize(size_x, size_y, size_z));

    for p in placements {
        let rel_x = (p.z - min_z) as usize;
        let rel_y = (p.x - min_x) as usize;
        let rel_z = (p.y - min_y) as usize;

        let (kind, dir) = parse_block_string(&p.block)?;
        let pos = Position(rel_x, rel_y, rel_z);
        world[pos] = Block {
            kind,
            direction: dir,
        };
    }

    Ok(world)
}

// -------------------- NOT synthesis ILP (no time, combinational) --------------------
//
// Key constraints that fix your observed wrong solutions:
//
// 1) Wall torch faces/support mapping fixed: facing == support direction.
// 2) Wall torch requires D1 on its support block: TW <= D1[support]  (forces "1 1 2 wire").
// 3) DP1 is ONLY driven from IN (pinned) and propagates via incoming DP1 (no BP->DP1 shortcut).
// 4) BP is ONLY powered by its own top dust: BP == DP1 & S  (no neighbor-dust cheating).
//
fn run_solver(args: Args) -> eyre::Result<()> {
    let out_dir = PathBuf::from(&args.out_dir);
    fs::create_dir_all(&out_dir)?;

    let xs = args.x;
    let zs = args.z;
    // if xs < 5 || zs < 5 {
    //     eyre::bail!("NOT 합성 데모는 x,z 최소 5 이상 권장");
    // }

    let midz = zs / 2;
    let in_port = Cell { x: 0, z: midz };
    let out_port = Cell { x: xs - 1, z: midz };

    let mut lp = Lp::new();

    // placement vars
    let mut S = vec![vec![0usize; zs]; xs]; // block at y=oy
    let mut D0 = vec![vec![0usize; zs]; xs]; // ground dust at y=oy
    let mut D1 = vec![vec![0usize; zs]; xs]; // top dust at y=oy+1

    for x in 0..xs {
        for z in 0..zs {
            let c = Cell { x, z };
            S[x][z] = lp.bin(v_s(c));
            D0[x][z] = lp.bin(v_d0(c));
            D1[x][z] = lp.bin(v_d1(c));

            // top dust needs block under it
            lp.add_c(
                format!("d1_needs_block_{}_{}", x, z),
                vec![(1.0, D1[x][z]), (-1.0, S[x][z])],
                Sense::Le,
                0.0,
            );

            // ground dust can't overlap block at same cell
            lp.add_c(
                format!("d0_no_block_samecell_{}_{}", x, z),
                vec![(1.0, D0[x][z]), (1.0, S[x][z])],
                Sense::Le,
                1.0,
            );
        }
    }

    // wall torch placement vars: torch at y=oy in air cell (no S, no D0)
    let mut TW: Vec<Vec<Vec<Option<usize>>>> = vec![vec![vec![None; 4]; zs]; xs];
    for x in 0..xs {
        for z in 0..zs {
            let tc = Cell { x, z };
            let ot = ord(tc, zs);

            for dir in 0..4 {
                let Some(sc) = support_cell(tc, dir, xs, zs) else {
                    continue;
                };
                // DAG restriction: support must be "earlier" than torch
                if ord(sc, zs) >= ot {
                    continue;
                }

                let tw = lp.bin(v_tw(tc, dir));
                TW[x][z][dir] = Some(tw);

                // torch cell is air at y=oy: cannot place block or ground dust
                lp.add_c(
                    format!("tw_air_noS_{}_{}_{}", x, z, dir),
                    vec![(1.0, S[x][z]), (1.0, tw)],
                    Sense::Le,
                    1.0,
                );
                lp.add_c(
                    format!("tw_air_noD0_{}_{}_{}", x, z, dir),
                    vec![(1.0, D0[x][z]), (1.0, tw)],
                    Sense::Le,
                    1.0,
                );

                // must attach to an existing support block
                lp.add_c(
                    format!("tw_support_block_exists_{}_{}_{}", x, z, dir),
                    vec![(1.0, tw), (-1.0, S[sc.x][sc.z])],
                    Sense::Le,
                    0.0,
                );

                // **critical**: to be meaningfully controllable, support must have top dust
                // (forces "1 1 2 wire" kind of solutions)
                lp.add_c(
                    format!("tw_requires_d1_on_support_{}_{}_{}", x, z, dir),
                    vec![(1.0, tw), (-1.0, D1[sc.x][sc.z])],
                    Sense::Le,
                    0.0,
                );
            }

            // at most one wall torch per cell
            let mut sum_terms = Vec::new();
            for dir in 0..4 {
                if let Some(tw) = TW[x][z][dir] {
                    sum_terms.push((1.0, tw));
                }
            }
            if !sum_terms.is_empty() {
                lp.add_c(format!("one_tw_{}_{}", x, z), sum_terms, Sense::Le, 1.0);
            }
        }
    }

    // Force terminals
    // IN: must be a block + top dust
    lp.add_c(
        "force_S_in".to_string(),
        vec![(1.0, S[in_port.x][in_port.z])],
        Sense::Eq,
        1.0,
    );
    lp.add_c(
        "force_D1_in".to_string(),
        vec![(1.0, D1[in_port.x][in_port.z])],
        Sense::Eq,
        1.0,
    );

    // OUT: ground dust (output wire at y=oy)
    lp.add_c(
        "force_D0_out".to_string(),
        vec![(1.0, D0[out_port.x][out_port.z])],
        Sense::Eq,
        1.0,
    );

    // state vars
    let mut DP0 = vec![vec![vec![0usize; zs]; xs]; 2];
    let mut DP1 = vec![vec![vec![0usize; zs]; xs]; 2];
    let mut BP = vec![vec![vec![0usize; zs]; xs]; 2];

    for s in 0..=1 {
        for x in 0..xs {
            for z in 0..zs {
                let c = Cell { x, z };
                DP0[s][x][z] = lp.bin(v_dp0(s, c));
                DP1[s][x][z] = lp.bin(v_dp1(s, c));
                BP[s][x][z] = lp.bin(v_bp(s, c));

                // domains
                lp.add_c(
                    format!("dp0_le_d0_{}_{}_{}", s, x, z),
                    vec![(1.0, DP0[s][x][z]), (-1.0, D0[x][z])],
                    Sense::Le,
                    0.0,
                );
                lp.add_c(
                    format!("dp1_le_d1_{}_{}_{}", s, x, z),
                    vec![(1.0, DP1[s][x][z]), (-1.0, D1[x][z])],
                    Sense::Le,
                    0.0,
                );
                lp.add_c(
                    format!("bp_le_s_{}_{}_{}", s, x, z),
                    vec![(1.0, BP[s][x][z]), (-1.0, S[x][z])],
                    Sense::Le,
                    0.0,
                );
            }
        }
    }

    // Pin input top-dust directly (acts as the input signal)
    lp.add_c(
        "pin_dp1_in_s0".to_string(),
        vec![(1.0, DP1[0][in_port.x][in_port.z])],
        Sense::Eq,
        0.0,
    );
    lp.add_c(
        "pin_dp1_in_s1".to_string(),
        vec![(1.0, DP1[1][in_port.x][in_port.z])],
        Sense::Eq,
        1.0,
    );

    // Define BP == DP1 & S (no neighbor cheating)
    for s in 0..=1 {
        for x in 0..xs {
            for z in 0..zs {
                // BP <= DP1
                lp.add_c(
                    format!("bp_le_dp1_{}_{}_{}", s, x, z),
                    vec![(1.0, BP[s][x][z]), (-1.0, DP1[s][x][z])],
                    Sense::Le,
                    0.0,
                );
                // BP <= S already exists via bp_le_s
                // BP >= DP1 + S - 1
                lp.add_c(
                    format!("bp_ge_dp1_plus_s_minus1_{}_{}_{}", s, x, z),
                    vec![(1.0, BP[s][x][z]), (-1.0, DP1[s][x][z]), (-1.0, S[x][z])],
                    Sense::Ge,
                    -1.0,
                );
            }
        }
    }

    // Torch state vars: TO = TW & !BP[support]
    let mut TO: Vec<Vec<Vec<Vec<Option<usize>>>>> = vec![vec![vec![vec![None; 4]; zs]; xs]; 2];
    for s in 0..=1 {
        for x in 0..xs {
            for z in 0..zs {
                let tc = Cell { x, z };
                for dir in 0..4 {
                    let Some(tw) = TW[x][z][dir] else { continue };
                    let sc = support_cell(tc, dir, xs, zs).unwrap();
                    let to = lp.bin(v_to(s, tc, dir));
                    TO[s][x][z][dir] = Some(to);

                    // to <= tw
                    lp.add_c(
                        format!("to_le_tw_{}_{}_{}_{}", s, x, z, dir),
                        vec![(1.0, to), (-1.0, tw)],
                        Sense::Le,
                        0.0,
                    );
                    // to + BP_support <= 1
                    lp.add_c(
                        format!("to_off_if_support_on_{}_{}_{}_{}", s, x, z, dir),
                        vec![(1.0, to), (1.0, BP[s][sc.x][sc.z])],
                        Sense::Le,
                        1.0,
                    );
                    // to >= tw - BP_support   <=> to - tw + BP_support >= 0
                    lp.add_c(
                        format!("to_ge_tw_minus_support_{}_{}_{}_{}", s, x, z, dir),
                        vec![(1.0, to), (-1.0, tw), (1.0, BP[s][sc.x][sc.z])],
                        Sense::Ge,
                        0.0,
                    );
                }
            }
        }
    }

    // DP1 propagation (top dust): ONLY from incoming DP1, no BP source.
    // DP1[c] = D1[c] & OR( DP1[incoming] )
    for s in 0..=1 {
        for x in 0..xs {
            for z in 0..zs {
                let c = Cell { x, z };
                if c == in_port {
                    continue; // pinned; avoid empty-source upper bound forcing 0
                }

                let dp = DP1[s][x][z];
                let mut src: Vec<usize> = Vec::new();
                for n in incoming_neighbors(c, xs, zs) {
                    src.push(DP1[s][n.x][n.z]);
                }

                // upper: DP1 <= sum(src)
                let mut up_terms = vec![(1.0, dp)];
                for &sv in &src {
                    up_terms.push((-1.0, sv));
                }
                lp.add_c(
                    format!("dp1_up_{}_{}_{}", s, x, z),
                    up_terms,
                    Sense::Le,
                    0.0,
                );

                // lower: DP1 >= source + D1 - 1
                for (i, &sv) in src.iter().enumerate() {
                    lp.add_c(
                        format!("dp1_lo_{}_{}_{}_{}", s, x, z, i),
                        vec![(1.0, dp), (-1.0, sv), (-1.0, D1[x][z])],
                        Sense::Ge,
                        -1.0,
                    );
                }
            }
        }
    }

    // DP0 propagation (ground dust): DP0 = D0 & OR( incoming DP0, incoming TO(nei) )
    for s in 0..=1 {
        for x in 0..xs {
            for z in 0..zs {
                let c = Cell { x, z };
                let dp = DP0[s][x][z];
                let oc = ord(c, zs);

                let mut src: Vec<usize> = Vec::new();

                // incoming ground dust
                for n in incoming_neighbors(c, xs, zs) {
                    src.push(DP0[s][n.x][n.z]);
                }

                // incoming torches from earlier neighbors
                for u in neighbors4(c, xs, zs) {
                    if ord(u, zs) >= oc {
                        continue;
                    }
                    for dir in 0..4 {
                        let Some(to) = TO[s][u.x][u.z][dir] else {
                            continue;
                        };
                        let sup = support_cell(u, dir, xs, zs).unwrap();
                        if sup == c {
                            continue; // torch doesn't power its own support
                        }
                        src.push(to);
                    }
                }

                // upper: DP0 <= sum(src)
                let mut up_terms = vec![(1.0, dp)];
                for &sv in &src {
                    up_terms.push((-1.0, sv));
                }
                lp.add_c(
                    format!("dp0_up_{}_{}_{}", s, x, z),
                    up_terms,
                    Sense::Le,
                    0.0,
                );

                // lower: DP0 >= source + D0 - 1
                for (i, &sv) in src.iter().enumerate() {
                    lp.add_c(
                        format!("dp0_lo_{}_{}_{}_{}", s, x, z, i),
                        vec![(1.0, dp), (-1.0, sv), (-1.0, D0[x][z])],
                        Sense::Ge,
                        -1.0,
                    );
                }
            }
        }
    }

    // Truth table at OUT (ground dust)
    lp.add_c(
        "out_when_in0".to_string(),
        vec![(1.0, DP0[0][out_port.x][out_port.z])],
        Sense::Eq,
        1.0,
    );
    lp.add_c(
        "out_when_in1".to_string(),
        vec![(1.0, DP0[1][out_port.x][out_port.z])],
        Sense::Eq,
        0.0,
    );

    // Objective (tune weights as you like)
    for x in 0..xs {
        for z in 0..zs {
            lp.add_obj(0.2, S[x][z]);
            lp.add_obj(1.0, D0[x][z]);
            lp.add_obj(1.0, D1[x][z]);
            for dir in 0..4 {
                if let Some(tw) = TW[x][z][dir] {
                    lp.add_obj(5.0, tw);
                }
            }
        }
    }

    let lp_path = out_dir.join("not_no_time_walltorch_fixed.lp");
    let sol_path = out_dir.join("not_no_time_walltorch_fixed.sol");
    lp.write_lp(&lp_path)?;
    eprintln!("Wrote LP: {}", lp_path.display());

    if !args.run_scip.unwrap_or(false) {
        eyre::bail!("run_scip=false라서 해를 못 만듭니다. --run-scip true로 실행하세요.");
    }
    run_scip(&args.scip, &lp_path, &sol_path)?;
    eprintln!("Wrote SOL: {}", sol_path.display());

    let sol = parse_scip_sol(&sol_path)?;
    if !sol.has_any() {
        eyre::bail!("SCIP infeasible/no solution: {}", lp_path.display());
    }

    // Emit blocks
    let ox = args.origin_x;
    let oy = args.origin_y;
    let oz = args.origin_z;

    let mut placements: Vec<Placement> = Vec::new();

    // Base blocks S at y=oy
    for x in 0..xs {
        for z in 0..zs {
            let c = Cell { x, z };
            if sol.get_bin(&v_s(c)) {
                placements.push(Placement {
                    x: ox + x as i32,
                    y: oy,
                    z: oz + z as i32,
                    block: "minecraft:stone".to_string(),
                });
            }
        }
    }

    // Ground dust D0 at y=oy
    for x in 0..xs {
        for z in 0..zs {
            let c = Cell { x, z };
            if sol.get_bin(&v_d0(c)) {
                placements.push(Placement {
                    x: ox + x as i32,
                    y: oy,
                    z: oz + z as i32,
                    block: "minecraft:redstone_wire".to_string(),
                });
            }
        }
    }

    // Top dust D1 at y=oy+1
    for x in 0..xs {
        for z in 0..zs {
            let c = Cell { x, z };
            if sol.get_bin(&v_d1(c)) {
                placements.push(Placement {
                    x: ox + x as i32,
                    y: oy + 1,
                    z: oz + z as i32,
                    block: "minecraft:redstone_wire".to_string(),
                });
            }
        }
    }

    // Wall torches at y=oy
    for x in 0..xs {
        for z in 0..zs {
            let tc = Cell { x, z };
            for dir in 0..4 {
                if TW[x][z][dir].is_none() {
                    continue;
                }
                if sol.get_bin(&v_tw(tc, dir)) {
                    let facing = TD[dir].facing;
                    placements.push(Placement {
                        x: ox + x as i32,
                        y: oy,
                        z: oz + z as i32,
                        block: format!("minecraft:redstone_wall_torch[facing={}]", facing),
                    });
                }
            }
        }
    }

    // Lever at west of input base block
    placements.push(Placement {
        x: ox - 1,
        y: oy,
        z: oz + in_port.z as i32,
        block: "minecraft:lever[face=wall,facing=east]".to_string(),
    });

    placements.sort_by_key(|p| (p.y, p.z, p.x));

    eprintln!("IN  = {:?}", in_port);
    eprintln!("OUT = {:?}", out_port);
    eprintln!("NOTE: D0 assumes solid floor exists at y=oy-1.");

    if args.emit_commands.unwrap_or(true) {
        for p in &placements {
            println!("/setblock {} {} {} {}", p.x, p.y, p.z, p.block);
        }
    }

    // Save NBT
    let world3d = placements_to_world3d(&placements)?;
    let nbt: NBTRoot = (&world3d).into();
    let nbt_path = out_dir.join("not_no_time_walltorch_fixed.nbt");
    nbt.save(&nbt_path);
    eprintln!("Saved NBT: {}", nbt_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_solver() -> eyre::Result<()> {
        let args = Args {
            x: 3,
            z: 1,
            origin_x: 0,
            origin_y: 0,
            origin_z: 0,
            scip: "scip".to_string(),
            run_scip: Some(true),
            emit_commands: Some(true),
            out_dir: ".".to_string(),
        };
        run_solver(args)
    }
}
