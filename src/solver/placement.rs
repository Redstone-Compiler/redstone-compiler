use eyre::Result;

use crate::solver::types::Placement;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

/// 블록 문자열을 파싱하여 BlockKind와 Direction으로 변환
/// 
/// 마인크래프트 블록 문자열(예: "minecraft:redstone_wall_torch[facing=east]")을
/// 파싱하여 내부 BlockKind와 Direction으로 변환합니다.
/// 
/// world3d 좌표계는 (x, y, z) = (mc_z, mc_x, mc_y)이므로,
/// 방향도 이에 맞춰 마인크래프트 방향을 내부 Direction으로 변환합니다.
pub fn parse_block_string(block_str: &str) -> Result<(BlockKind, Direction)> {
    // world3d 좌표계는 (x, y, z) = (mc_z, mc_x, mc_y).
    // 방향도 이에 맞춰 mc 방향을 내부 Direction으로 변환한다.
    let mc_facing_to_dir = |f: &str| match f {
        "east" => Direction::North,  // mc +x -> internal +y
        "west" => Direction::South,   // mc -x -> internal -y
        "south" => Direction::East,    // mc +z -> internal +x
        "north" => Direction::West,   // mc -z -> internal -x
        _ => Direction::None,
    };

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
        "minecraft:redstone_torch" => (BlockKind::Torch { is_on: false }, Direction::Bottom),
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
        s if s.starts_with("minecraft:repeater") => {
            let dir = if let Some(facing_val) = facing {
                mc_facing_to_dir(facing_val)
            } else {
                Direction::None
            };
            (
                BlockKind::Repeater {
                    is_on: false,
                    is_locked: false,
                    delay: 1,
                    lock_input1: None,
                    lock_input2: None,
                },
                dir,
            )
        }
        _ => eyre::bail!("Unknown block type: {}", name),
    };

    Ok((kind, dir))
}

/// Placement 목록을 World3D로 변환
/// 
/// 마인크래프트 좌표계의 Placement들을 내부 World3D 좌표계로 변환합니다.
/// 
/// 좌표 변환:
/// - world3d DimSize(x, y, z) = (mc_z, mc_x, mc_y)
/// - MC (x,y,z) -> world3d (x,y,z) = (mc_z, mc_x, mc_y)
pub fn placements_to_world3d(placements: &[Placement]) -> Result<World3D> {
    if placements.is_empty() {
        return Ok(World3D::new(DimSize(1, 1, 1)));
    }

    let min_x = placements.iter().map(|p| p.x).min().unwrap();
    let max_x = placements.iter().map(|p| p.x).max().unwrap();
    let min_y = placements.iter().map(|p| p.y).min().unwrap();
    let max_y = placements.iter().map(|p| p.y).max().unwrap();
    let min_z = placements.iter().map(|p| p.z).min().unwrap();
    let max_z = placements.iter().map(|p| p.z).max().unwrap();

    // world3d DimSize(x, y, z) = (mc_z, mc_x, mc_y)
    let size_x = (max_z - min_z + 1) as usize;
    let size_y = (max_x - min_x + 1) as usize;
    let size_z = (max_y - min_y + 1) as usize;

    let mut world = World3D::new(DimSize(size_x, size_y, size_z));

    for p in placements {
        // MC (x,y,z) -> world3d (x,y,z) = (mc_z, mc_x, mc_y)
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

