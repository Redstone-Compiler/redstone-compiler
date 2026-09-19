use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;

use super::{PhysicalCellBuild, PhysicalCellDocument};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{BlockKind, Direction};
use crate::world::position::Position;
use crate::world::simulator::Simulator;
use crate::world::World;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PhysicalCellCompactReport {
    pub contract_size: [usize; 3],
    pub occupied_bounds: Option<PhysicalCellBounds>,
    pub occupied_volume: usize,
    pub block_counts: BTreeMap<String, usize>,
    pub repeater_delay_total: usize,
    pub repeater_chains: Vec<PhysicalCellRepeaterChain>,
    pub dust_components: Vec<PhysicalCellDustComponent>,
    pub safe_mutations: Vec<PhysicalCellSafeMutation>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PhysicalCellBounds {
    pub min: [usize; 3],
    pub max: [usize; 3],
    pub size: [usize; 3],
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PhysicalCellRepeaterChain {
    pub positions: Vec<[usize; 3]>,
    pub direction: String,
    pub repeaters: usize,
    pub total_delay: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PhysicalCellDustComponent {
    pub blocks: usize,
    pub bounds: PhysicalCellBounds,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PhysicalCellSafeMutation {
    pub position: [usize; 3],
    pub mutation: String,
    pub original: String,
    pub replacement: String,
}

impl PhysicalCellDocument {
    pub fn compact_report(
        &self,
        build: &PhysicalCellBuild,
    ) -> eyre::Result<PhysicalCellCompactReport> {
        let occupied = build.world.iter_block();
        let occupied_bounds = bounds(occupied.iter().map(|(position, _)| *position));
        let occupied_volume = occupied_bounds
            .as_ref()
            .map_or(0, |bounds| bounds.size.iter().product());
        let mut block_counts = BTreeMap::new();
        let mut repeater_delay_total = 0;
        for (_, block) in &occupied {
            *block_counts.entry(block.kind.name()).or_insert(0) += 1;
            if let BlockKind::Repeater { delay, .. } = block.kind {
                repeater_delay_total += delay;
            }
        }

        Ok(PhysicalCellCompactReport {
            contract_size: [self.size.0, self.size.1, self.size.2],
            occupied_bounds,
            occupied_volume,
            block_counts,
            repeater_delay_total,
            repeater_chains: repeater_chains(build),
            dust_components: dust_components(build)?,
            safe_mutations: safe_mutations(self, build),
        })
    }
}

fn repeater_chains(build: &PhysicalCellBuild) -> Vec<PhysicalCellRepeaterChain> {
    let repeaters = build
        .world
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| match block.kind {
            BlockKind::Repeater { delay, .. } => Some((position, (block.direction, delay))),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let has_predecessor = repeaters
        .iter()
        .filter_map(|(position, (direction, _))| {
            position
                .walk(direction.inverse())
                .filter(|next| {
                    repeaters
                        .get(next)
                        .is_some_and(|(next_direction, _)| next_direction == direction)
                })
                .map(|next| (next, *position))
        })
        .map(|(next, _)| next)
        .collect::<BTreeSet<_>>();

    let mut chains = Vec::new();
    for (start, (direction, _)) in &repeaters {
        if has_predecessor.contains(start) {
            continue;
        }
        let mut positions = Vec::new();
        let mut total_delay = 0;
        let mut cursor = *start;
        while let Some((next_direction, delay)) = repeaters.get(&cursor) {
            if next_direction != direction {
                break;
            }
            positions.push([cursor.0, cursor.1, cursor.2]);
            total_delay += delay;
            let Some(next) = cursor.walk(direction.inverse()) else {
                break;
            };
            cursor = next;
        }
        if positions.len() >= 2 {
            chains.push(PhysicalCellRepeaterChain {
                repeaters: positions.len(),
                positions,
                direction: format!("{direction:?}"),
                total_delay,
            });
        }
    }
    chains.sort_by_key(|chain| {
        (
            std::cmp::Reverse(chain.repeaters),
            chain.positions.first().copied(),
        )
    });
    chains
}

fn dust_components(build: &PhysicalCellBuild) -> eyre::Result<Vec<PhysicalCellDustComponent>> {
    let world = World::from(&build.world);
    let simulator = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0)
        .map_err(|error| eyre::eyre!(error.message().to_owned()))?;
    let dust = simulator
        .world()
        .iter_block()
        .into_iter()
        .filter_map(|(position, block)| block.kind.is_redstone().then_some(position))
        .collect::<BTreeSet<_>>();
    let mut neighbors = BTreeMap::<Position, BTreeSet<Position>>::new();
    for position in &dust {
        for source in simulator
            .diagnostic_power_sources(*position)
            .into_iter()
            .filter(|source| source.relation == "dust-neighbor")
        {
            let source = Position(source.position[0], source.position[1], source.position[2]);
            neighbors.entry(*position).or_default().insert(source);
            neighbors.entry(source).or_default().insert(*position);
        }
    }

    let mut unseen = dust;
    let mut components = Vec::new();
    while let Some(start) = unseen.iter().next().copied() {
        let mut queue = VecDeque::from([start]);
        let mut component = BTreeSet::new();
        unseen.remove(&start);
        while let Some(position) = queue.pop_front() {
            component.insert(position);
            for next in neighbors.get(&position).into_iter().flatten() {
                if unseen.remove(next) {
                    queue.push_back(*next);
                }
            }
        }
        if component.len() >= 2 {
            components.push(PhysicalCellDustComponent {
                blocks: component.len(),
                bounds: bounds(component.into_iter()).expect("component is non-empty"),
            });
        }
    }
    components.sort_by_key(|component| std::cmp::Reverse(component.blocks));
    Ok(components)
}

fn safe_mutations(
    document: &PhysicalCellDocument,
    build: &PhysicalCellBuild,
) -> Vec<PhysicalCellSafeMutation> {
    let outputs = build.outputs.values().copied().collect::<BTreeSet<_>>();
    let mut mutations = Vec::new();
    for (position, block) in build.world.iter_block() {
        if outputs.contains(&position) {
            continue;
        }
        if matches!(
            block.kind,
            BlockKind::Redstone { .. } | BlockKind::Repeater { .. } | BlockKind::Torch { .. }
        ) && mutation_passes(document, build, position, BlockKind::Air)
        {
            mutations.push(PhysicalCellSafeMutation {
                position: [position.0, position.1, position.2],
                mutation: "remove".to_owned(),
                original: block.kind.name(),
                replacement: "Air".to_owned(),
            });
        }
        if block.kind.is_repeater()
            && mutation_passes(
                document,
                build,
                position,
                PlacedNode::new_redstone(position).block.kind,
            )
        {
            mutations.push(PhysicalCellSafeMutation {
                position: [position.0, position.1, position.2],
                mutation: "replace".to_owned(),
                original: block.kind.name(),
                replacement: "Redstone".to_owned(),
            });
        }
    }
    mutations.sort_by_key(|mutation| (mutation.position, mutation.mutation.clone()));
    mutations
}

fn mutation_passes(
    document: &PhysicalCellDocument,
    build: &PhysicalCellBuild,
    position: Position,
    replacement: BlockKind,
) -> bool {
    let mut mutated = build.clone();
    mutated.world[position].kind = replacement;
    mutated.world[position].direction = Direction::None;
    mutated.world.initialize_redstone_states();
    document
        .verify(&mutated)
        .is_ok_and(|verification| verification.failures.is_empty())
}

fn bounds(positions: impl IntoIterator<Item = Position>) -> Option<PhysicalCellBounds> {
    let mut positions = positions.into_iter();
    let first = positions.next()?;
    let mut min = [first.0, first.1, first.2];
    let mut max = min;
    for position in positions {
        min[0] = min[0].min(position.0);
        min[1] = min[1].min(position.1);
        min[2] = min[2].min(position.2);
        max[0] = max[0].max(position.0);
        max[1] = max[1].max(position.1);
        max[2] = max[2].max(position.2);
    }
    Some(PhysicalCellBounds {
        min,
        max,
        size: [
            max[0] - min[0] + 1,
            max[1] - min[1] + 1,
            max[2] - min[2] + 1,
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_report_finds_repeater_chains_and_safe_replacements() -> eyre::Result<()> {
        let document: PhysicalCellDocument = r#"
            rcell 1;
            cell "buffer-chain" size [1, 5, 2] {
              glyph "P" = repeater toward y- delay 1;
              input "a" at [0, 0, 1] on y+;
              output "y" at [0, 3, 1];
              plane yz at x=0 {
                z=0 "..##.";
                z=1 ".#PP.";
              }
              expect "y" = a;
            }
        "#
        .parse()?;
        let build = document.build()?;
        let report = document.compact_report(&build)?;
        assert_eq!(report.contract_size, [1, 5, 2]);
        assert_eq!(report.block_counts["Repeater"], 2);
        assert_eq!(report.repeater_chains.len(), 1);
        assert_eq!(report.repeater_chains[0].repeaters, 2);
        assert!(report.safe_mutations.iter().any(|mutation| {
            mutation.position == [0, 2, 1]
                && mutation.mutation == "replace"
                && mutation.replacement == "Redstone"
        }));
        Ok(())
    }
}
