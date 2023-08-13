use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::world::{
    block::{Block, BlockKind, Direction, RedstoneState},
    position::{DimSize, Position},
    world::{World, World3D},
};

#[derive(Serialize, Deserialize, Debug)]
pub struct NBTRoot {
    #[serde(rename = "size")]
    // (y, z, x)
    size: (i32, i32, i32),

    blocks: Vec<NBTBlock>,
    palette: Vec<NBTPalette>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NBTBlock {
    state: i32,
    // (y, z, x)
    pos: (i32, i32, i32),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NBTPalette {
    #[serde(rename = "Name")]
    name: String,

    #[serde(rename = "Properties")]
    properties: Option<NBTPaletteProperty>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct NBTPaletteProperty {
    facing: Option<String>,
    // torch
    lit: Option<String>,
    face: Option<String>,

    // redstone
    power: Option<String>,
    east: Option<String>,
    west: Option<String>,
    south: Option<String>,
    north: Option<String>,

    // repeater
    delay: Option<String>,
    locked: Option<String>,

    // etc
    powered: Option<String>,
    conditional: Option<String>,
    waterlogged: Option<String>,
}

pub trait ToNBT {
    fn to_nbt(&self) -> NBTRoot;
}

impl ToNBT for World3D {
    fn to_nbt(&self) -> NBTRoot {
        self.into()
    }
}

impl From<&World3D> for NBTRoot {
    fn from(value: &World3D) -> Self {
        world3d_to_nbt(value)
    }
}

impl NBTRoot {
    pub fn load(path: &PathBuf) -> eyre::Result<NBTRoot> {
        let file = File::open(path).unwrap();
        let mut decoder = GzDecoder::new(file);

        let mut bytes = vec![];
        decoder.read_to_end(&mut bytes).unwrap();

        Ok(fastnbt::from_bytes(&bytes).unwrap())
    }

    pub fn save(&self, path: &PathBuf) {
        let new_bytes = fastnbt::to_bytes(&self).unwrap();
        let outfile = File::create(path).unwrap();
        let mut encoder = GzEncoder::new(outfile, Compression::best());
        encoder.write_all(&new_bytes).unwrap();
    }

    pub fn to_world(&self) -> World {
        nbt_to_world(self)
    }
}

fn nbt_block_name(block: &Block) -> (String, String, Option<NBTPaletteProperty>) {
    let (palette_name, specify_name, property) = match block.kind {
        BlockKind::Air => ("air", "air".to_owned(), None),
        BlockKind::Cobble { .. } => ("stone_bricks", "stone_bricks".to_owned(), None),
        BlockKind::Switch { is_on } => (
            "lever",
            "lever".to_owned(),
            Some(NBTPaletteProperty {
                face: match block.direction {
                    Direction::Bottom => Some("floor".to_owned()),
                    Direction::Top => Some("ceiling".to_owned()),
                    Direction::East | Direction::West | Direction::South | Direction::North => {
                        Some("wall".to_owned())
                    }
                    _ => unreachable!(),
                },
                facing: match block.direction {
                    Direction::Bottom | Direction::Top => None,
                    Direction::East => Some("south".to_owned()),
                    Direction::West => Some("north".to_owned()),
                    Direction::South => Some("west".to_owned()),
                    Direction::North => Some("east".to_owned()),
                    _ => unreachable!(),
                },
                powered: is_on.then(|| "true".to_owned()),
                ..Default::default()
            }),
        ),
        BlockKind::Redstone {
            state, strength, ..
        } => (
            "redstone_wire",
            format!("redstone_wire_{}_{}", strength, state),
            Some(NBTPaletteProperty {
                power: Some(strength.to_string()),
                east: ((state & RedstoneState::South as usize) > 0).then(|| "side".to_owned()),
                west: ((state & RedstoneState::North as usize) > 0).then(|| "side".to_owned()),
                south: ((state & RedstoneState::West as usize) > 0).then(|| "side".to_owned()),
                north: ((state & RedstoneState::East as usize) > 0).then(|| "side".to_owned()),
                ..Default::default()
            }),
        ),
        BlockKind::Torch { .. } => {
            if matches!(block.direction, Direction::Bottom) {
                ("redstone_torch", "redstone_torch".to_owned(), None)
            } else {
                let facing = match block.direction {
                    Direction::East => "south",
                    Direction::West => "north",
                    Direction::South => "west",
                    Direction::North => "east",
                    _ => unreachable!(),
                }
                .to_owned();

                (
                    "redstone_wall_torch",
                    format!("redstone_wall_torch_{facing}"),
                    Some(NBTPaletteProperty {
                        facing: Some(facing),
                        ..Default::default()
                    }),
                )
            }
        }
        BlockKind::Repeater {
            is_on,
            is_locked,
            delay,
            ..
        } => {
            let facing = match block.direction {
                Direction::East => "north",
                Direction::West => "south",
                Direction::South => "east",
                Direction::North => "west",
                _ => unreachable!(),
            }
            .to_owned();

            (
                "repeater",
                "repeater".to_owned(),
                Some(NBTPaletteProperty {
                    facing: Some(facing),
                    delay: Some(delay.to_string()),
                    locked: Some(is_locked.to_string()),
                    powered: Some(is_on.to_string()),
                    ..Default::default()
                }),
            )
        }
        BlockKind::RedstoneBlock => ("redstone_block", "redstone_block".to_owned(), None),
        BlockKind::Piston { .. } => todo!(),
    };

    (format!("minecraft:{palette_name}"), specify_name, property)
}

fn world3d_to_nbt(world: &World3D) -> NBTRoot {
    let mut palette: Vec<NBTPalette> = Vec::new();
    let mut palette_index: HashMap<String, usize> = HashMap::new();

    // make palette
    for (_, block) in world.iter_block() {
        let (name, specify_name, properties) = nbt_block_name(&block);

        if !palette_index.contains_key(&specify_name) {
            palette.push(NBTPalette { name, properties });
            palette_index.insert(specify_name, palette.len() - 1);
        }
    }

    // make block
    let mut blocks: Vec<NBTBlock> = Vec::new();
    for (pos, block) in world.iter_block() {
        let (_, specify_name, _) = nbt_block_name(&block);

        blocks.push(NBTBlock {
            state: palette_index[&specify_name] as i32,
            pos: (pos.2 as i32, pos.0 as i32, pos.1 as i32),
        });
    }

    NBTRoot {
        size: (
            world.size.2 as i32,
            world.size.0 as i32,
            world.size.1 as i32,
        ),
        blocks,
        palette,
    }
}

fn nbt_palette_to_block(palette: &NBTPalette) -> (BlockKind, Direction) {
    match &palette.name[..] {
        "minecraft:air" => (BlockKind::Air, Direction::None),
        "minecraft:stone_bricks"
        | "minecraft:mossy_cobblestone"
        | "minecraft:cobblestone"
        | "minecraft:oak_planks" => (
            BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            Direction::None,
        ),
        "minecraft:lever" => (
            BlockKind::Switch {
                is_on: palette
                    .properties
                    .as_ref()
                    .map(|p| p.powered.as_ref().map(|p| p.eq("true").then(|| 0)))
                    .flatten()
                    .is_some(),
            },
            if let Some(face) = &palette.properties.as_ref().unwrap().face {
                match &face[..] {
                    "floor" => Direction::Bottom,
                    "ceiling" => Direction::Top,
                    "wall" => {
                        if let Some(facing) = &palette.properties.as_ref().unwrap().facing {
                            match &facing[..] {
                                "none" => Direction::None,
                                "east" => Direction::South,
                                "west" => Direction::North,
                                "south" => Direction::West,
                                "north" => Direction::East,
                                _ => unreachable!(),
                            }
                        } else {
                            unimplemented!()
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                Direction::None
            },
        ),
        "minecraft:redstone_wire" => {
            let mut state = 0;

            if let Some(properties) = &palette.properties {
                if let Some(east) = &properties.east {
                    if east != "none" {
                        state |= RedstoneState::South as usize;
                    }
                }

                if let Some(west) = &properties.west {
                    if west != "none" {
                        state |= RedstoneState::North as usize;
                    }
                }

                if let Some(south) = &properties.south {
                    if south != "none" {
                        state |= RedstoneState::West as usize;
                    }
                }

                if let Some(north) = &properties.north {
                    if north != "none" {
                        state |= RedstoneState::East as usize;
                    }
                }
            }

            (
                BlockKind::Redstone {
                    on_count: 0,
                    strength: if let Some(power) =
                        palette.properties.as_ref().unwrap().power.as_ref()
                    {
                        power.parse().unwrap()
                    } else {
                        0
                    },
                    state,
                },
                Direction::None,
            )
        }
        "minecraft:redstone_torch" => (BlockKind::Torch { is_on: false }, Direction::Bottom),
        "minecraft:redstone_wall_torch" => (
            BlockKind::Torch { is_on: false },
            match &palette
                .properties
                .as_ref()
                .unwrap()
                .facing
                .as_ref()
                .unwrap()[..]
            {
                "none" => Direction::None,
                "east" => Direction::South,
                "west" => Direction::North,
                "south" => Direction::West,
                "north" => Direction::East,
                _ => unreachable!(),
            },
        ),
        "minecraft:repeater" => (
            BlockKind::Repeater {
                is_on: false,
                is_locked: false,
                delay: palette
                    .properties
                    .as_ref()
                    .unwrap()
                    .delay
                    .as_ref()
                    .unwrap()
                    .parse()
                    .unwrap(),
                lock_input1: None,
                lock_input2: None,
            },
            match &palette
                .properties
                .as_ref()
                .unwrap()
                .facing
                .as_ref()
                .unwrap()[..]
            {
                "none" => Direction::None,
                "east" => Direction::North,
                "west" => Direction::South,
                "south" => Direction::East,
                "north" => Direction::West,
                _ => unreachable!(),
            },
        ),
        "minecraft:redstone_block" => (BlockKind::RedstoneBlock, Direction::None),
        remain => {
            eprintln!("{remain}");
            unimplemented!()
        }
    }
}

fn nbt_to_world(nbt: &NBTRoot) -> World {
    let palette = nbt
        .palette
        .iter()
        .map(|palette| nbt_palette_to_block(palette))
        .collect_vec();

    let mut blocks = Vec::new();
    for block in &nbt.blocks {
        if matches!(palette[block.state as usize].0, BlockKind::Air) {
            continue;
        }

        blocks.push((
            Position(
                block.pos.2 as usize,
                block.pos.0 as usize,
                block.pos.1 as usize,
            ),
            Block {
                kind: palette[block.state as usize].0,
                direction: palette[block.state as usize].1,
            },
        ));
    }

    World {
        size: DimSize(
            nbt.size.2 as usize + 1,
            nbt.size.0 as usize + 1,
            nbt.size.1 as usize + 1,
        ),
        blocks,
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use fastnbt::stream::{Parser, Value};
    use flate2::{read::GzDecoder, write::GzEncoder, Compression};

    use crate::{
        graph::{graphviz::ToGraphvizGraph, world::builder::WorldGraphBuilder},
        world::{
            block::Direction,
            position::{DimSize, Position},
            world::World,
        },
    };

    use super::*;

    #[test]
    fn unittest_show_nbt_structure() -> eyre::Result<()> {
        let file = std::fs::File::open(&"test/alu.nbt").unwrap();
        let decoder = GzDecoder::new(file);

        let mut parser = Parser::new(decoder);
        let mut indent = 0;

        loop {
            match parser.next() {
                Err(e) => {
                    println!("{:?}", e);
                    break;
                }
                Ok(value) => {
                    match value {
                        Value::CompoundEnd => indent -= 4,
                        Value::ListEnd => indent -= 4,
                        _ => {}
                    }

                    println!("{:indent$}{:?}", "", value, indent = indent);

                    match value {
                        Value::Compound(_) => indent += 4,
                        Value::List(_, _, _) => indent += 4,
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn unittest_nbt_file_load_and_save() -> eyre::Result<()> {
        let file = std::fs::File::open(&"test/test.nbt").unwrap();
        let mut decoder = GzDecoder::new(file);

        let mut bytes = vec![];
        decoder.read_to_end(&mut bytes).unwrap();

        let mut leveldat: NBTRoot = fastnbt::from_bytes(&bytes).unwrap();
        leveldat.palette.push(NBTPalette {
            name: "minecraft:redstone_wall_torch".to_owned(),
            properties: Some(NBTPaletteProperty {
                facing: Some("west".to_owned()),
                face: Some("wall".to_owned()),
                lit: Some("false".to_owned()),
                ..Default::default()
            }),
        });

        let y = leveldat.palette.len() - 1;
        leveldat.blocks.iter_mut().for_each(|x| x.state = y as i32);

        let new_bytes = fastnbt::to_bytes(&leveldat).unwrap();
        let outfile = std::fs::File::create("test/level.nbt").unwrap();
        let mut encoder = GzEncoder::new(outfile, Compression::best());
        encoder.write_all(&new_bytes).unwrap();

        Ok(())
    }

    #[test]
    fn unittest_export_and_gate_to_nbt() {
        let default_restone = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Default::default(),
        };

        let default_cobble = Block {
            kind: BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            },
            direction: Default::default(),
        };

        let mock_world = World {
            size: DimSize(4, 6, 3),
            blocks: vec![
                (Position(0, 1, 0), default_restone.clone()),
                (Position(2, 1, 0), default_restone.clone()),
                (Position(0, 2, 0), default_cobble.clone()),
                (Position(1, 2, 0), default_cobble.clone()),
                (Position(2, 2, 0), default_cobble.clone()),
                (Position(1, 2, 1), default_restone.clone()),
                (Position(1, 4, 0), default_restone.clone()),
                (
                    Position(0, 2, 1),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::Bottom,
                    },
                ),
                (
                    Position(2, 2, 1),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::Bottom,
                    },
                ),
                (
                    Position(1, 3, 0),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::South,
                    },
                ),
                (
                    Position(0, 0, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: false },
                        direction: Direction::Bottom,
                    },
                ),
                (
                    Position(2, 0, 0),
                    Block {
                        kind: BlockKind::Switch { is_on: false },
                        direction: Direction::Bottom,
                    },
                ),
            ],
        };

        let mut world3d: World3D = (&mock_world).into();
        world3d.initialize_redstone_states();

        let nbt: NBTRoot = world3d.to_nbt();
        let new_bytes = fastnbt::to_bytes(&nbt).unwrap();
        let outfile = std::fs::File::create("test/and-gate.nbt").unwrap();
        let mut encoder = GzEncoder::new(outfile, Compression::best());
        encoder.write_all(&new_bytes).unwrap();
    }

    #[test]
    fn unittest_import_nbt_as_world() -> eyre::Result<()> {
        let nbt = NBTRoot::load(&"test/alu.nbt".into())?;
        nbt.save(&"test/alu-export.nbt".into());
        let g = WorldGraphBuilder::new(&nbt.to_world()).build();
        println!("{}", g.to_graphviz());

        Ok(())
    }
}
