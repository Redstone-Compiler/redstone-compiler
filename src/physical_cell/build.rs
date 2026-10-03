use std::collections::{BTreeMap, BTreeSet, HashSet};

use eyre::{bail, ensure, ContextCompat};

use super::{CellGlyph, PhysicalCellDocument};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{Block, BlockKind};
use crate::world::position::Position;
use crate::world::World3D;

#[derive(Clone)]
pub struct PhysicalCellBuild {
    pub world: World3D,
    pub inputs: BTreeMap<String, Position>,
    pub input_contacts: BTreeMap<String, Vec<Position>>,
    pub probes: BTreeMap<String, Position>,
    pub outputs: BTreeMap<String, Position>,
    pub auto_supports: BTreeSet<Position>,
}

impl PhysicalCellBuild {
    pub fn observation_position(&self, name: &str) -> Option<Position> {
        self.probes
            .get(name)
            .or_else(|| self.outputs.get(name))
            .copied()
    }

    pub fn observations(&self) -> impl Iterator<Item = (&str, Position)> {
        self.probes
            .iter()
            .chain(self.outputs.iter())
            .map(|(name, position)| (name.as_str(), *position))
    }

    pub(crate) fn from_document(document: &PhysicalCellDocument) -> eyre::Result<Self> {
        ensure!(
            document.size.0 > 0 && document.size.1 > 0 && document.size.2 > 0,
            "cell dimensions must be positive"
        );
        let mut cells = BTreeMap::<Position, char>::new();
        for plane in &document.planes {
            ensure!(
                plane.fixed < plane.axes.fixed_len(document.size),
                "plane {} fixes {}={} outside the cell",
                plane.axes.as_str(),
                plane.axes.fixed_axis(),
                plane.fixed
            );
            for (row, text) in &plane.rows {
                ensure!(
                    *row < plane.axes.row_len(document.size),
                    "plane {} row {}={} is outside the cell",
                    plane.axes.as_str(),
                    plane.axes.row_axis(),
                    row
                );
                let glyphs = text.chars().collect::<Vec<_>>();
                ensure!(
                    glyphs.len() == plane.axes.column_len(document.size),
                    "plane {} {}={} has {} cells; expected {}",
                    plane.axes.as_str(),
                    plane.axes.row_axis(),
                    row,
                    glyphs.len(),
                    plane.axes.column_len(document.size)
                );
                for (column, glyph) in glyphs.into_iter().enumerate() {
                    if glyph == '.' {
                        continue;
                    }
                    ensure!(
                        matches!(glyph, '#' | 'r') || document.glyphs.contains_key(&glyph),
                        "undefined glyph `{glyph}`"
                    );
                    let position = plane.axes.position(plane.fixed, *row, column);
                    if let Some(existing) = cells.insert(position, glyph) {
                        ensure!(
                            existing == glyph,
                            "conflicting glyphs `{existing}` and `{glyph}` at {position:?}"
                        );
                    }
                }
            }
        }

        let mut world = World3D::new(document.size);
        for (position, glyph) in &cells {
            world[*position] = match glyph {
                '#' => PlacedNode::new_cobble(*position).block,
                'r' => PlacedNode::new_redstone(*position).block,
                glyph => match &document.glyphs[glyph] {
                    CellGlyph::Torch { support } => Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: support.direction(),
                    },
                    CellGlyph::Repeater { toward, delay } => Block {
                        kind: BlockKind::Repeater {
                            is_on: false,
                            is_locked: false,
                            delay: *delay,
                            lock_input1: None,
                            lock_input2: None,
                        },
                        direction: toward.direction(),
                    },
                },
            };
        }

        let mut inputs = BTreeMap::new();
        let mut input_contacts = BTreeMap::<String, Vec<Position>>::new();
        let mut input_positions = HashSet::new();
        for input in &document.inputs {
            ensure!(
                document.size.bound_on(input.position),
                "input `{}` is outside the cell",
                input.name
            );
            inputs.entry(input.name.clone()).or_insert(input.position);
            input_contacts
                .entry(input.name.clone())
                .or_default()
                .push(input.position);
            ensure!(
                input_positions.insert(input.position),
                "multiple inputs occupy {:?}",
                input.position
            );
            ensure!(
                world[input.position].kind.is_air(),
                "input `{}` overlaps a block at {:?}",
                input.name,
                input.position
            );
            world[input.position] = Block {
                kind: BlockKind::Switch { is_on: false },
                direction: input.support.direction(),
            };
        }

        let mut auto_supports = BTreeSet::new();
        let support_targets = world
            .iter_block()
            .into_iter()
            .filter_map(|(position, block)| {
                let enabled = (block.kind.is_redstone() && document.auto_support.dust)
                    || (block.kind.is_repeater() && document.auto_support.repeater);
                enabled.then_some(position)
            })
            .collect::<Vec<_>>();
        for position in support_targets {
            let below = position
                .down()
                .with_context(|| format!("cannot support block at floor position {position:?}"))?;
            match world[below].kind {
                kind if kind.is_air() => {
                    world[below] = PlacedNode::new_cobble(below).block;
                    auto_supports.insert(below);
                }
                kind if kind.is_cobble() => {}
                _ => bail!("support below {position:?} conflicts at {below:?}"),
            }
        }

        for (position, block) in world.iter_block() {
            if block.kind.is_torch() || block.kind.is_switch() {
                let support = position.walk(block.direction).with_context(|| {
                    format!("attachment at {position:?} points outside the cell")
                })?;
                ensure!(
                    document.size.bound_on(support) && world[support].kind.is_cobble(),
                    "attachment at {position:?} requires solid support at {support:?}"
                );
            } else if block.kind.is_redstone() || block.kind.is_repeater() {
                let below = position.down().with_context(|| {
                    format!("signal block at {position:?} has no floor support")
                })?;
                ensure!(
                    world[below].kind.is_cobble(),
                    "signal block at {position:?} requires support at {below:?}"
                );
            }
        }

        let mut outputs = BTreeMap::new();
        for output in &document.outputs {
            ensure!(
                document.size.bound_on(output.position),
                "output `{}` is outside the cell",
                output.name
            );
            ensure!(
                outputs
                    .insert(output.name.clone(), output.position)
                    .is_none(),
                "duplicate output `{}`",
                output.name
            );
            ensure!(
                !world[output.position].kind.is_air(),
                "output `{}` points to air at {:?}",
                output.name,
                output.position
            );
        }

        let mut probes = BTreeMap::new();
        for probe in &document.probes {
            ensure!(
                document.size.bound_on(probe.position),
                "probe `{}` is outside the cell",
                probe.name
            );
            ensure!(
                probes.insert(probe.name.clone(), probe.position).is_none(),
                "duplicate probe `{}`",
                probe.name
            );
            ensure!(
                !outputs.contains_key(&probe.name),
                "observation `{}` is declared as both an output and a probe",
                probe.name
            );
            ensure!(
                !world[probe.position].kind.is_air(),
                "probe `{}` points to air at {:?}",
                probe.name,
                probe.position
            );
        }

        world.initialize_redstone_states();
        Ok(Self {
            world,
            inputs,
            input_contacts,
            probes,
            outputs,
            auto_supports,
        })
    }
}
