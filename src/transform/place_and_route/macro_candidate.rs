use std::collections::HashSet;

use crate::graph::logic::LogicGraph;
use crate::graph::GraphNodeKind;
use crate::transform::place_and_route::estimate::{bounding_box, world_compact_cost, BoundingBox};
use crate::transform::place_and_route::local_placer::LocalPlacement;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalPortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalPortSide {
    West,
    East,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPort {
    pub name: String,
    pub direction: PhysicalPortDirection,
    pub side: PhysicalPortSide,
    pub pin: Position,
    pub facing: Direction,
}

#[derive(Debug, Clone)]
pub struct MacroLayoutCandidate {
    pub unit_id: String,
    pub world_fragment: World3D,
    pub bbox: BoundingBox,
    pub ports: Vec<PhysicalPort>,
    pub occupied_cells: HashSet<Position>,
    pub blocked_cells: HashSet<Position>,
    pub cost: usize,
}

#[derive(Debug, Clone)]
pub struct MacroCandidateBuilder {
    unit_id: String,
    west_inputs: Vec<String>,
    east_outputs: Vec<String>,
    pin_spacing: usize,
}

impl MacroCandidateBuilder {
    pub fn new(unit_id: impl Into<String>) -> Self {
        Self {
            unit_id: unit_id.into(),
            west_inputs: Vec::new(),
            east_outputs: Vec::new(),
            pin_spacing: 3,
        }
    }

    pub fn west_inputs<const N: usize>(mut self, names: [&str; N]) -> Self {
        self.west_inputs = sorted(names);
        self
    }

    pub fn east_outputs<const N: usize>(mut self, names: [&str; N]) -> Self {
        self.east_outputs = sorted(names);
        self
    }

    pub fn build(
        &self,
        graph: &LogicGraph,
        local: &LocalPlacement,
    ) -> eyre::Result<MacroLayoutCandidate> {
        let translated = translate_local_without_switches(local, Position(4, 4, 0));
        let local_bounds = bounding_box(&translated)
            .ok_or_else(|| eyre::eyre!("local placement has no blocks"))?;
        let port_count = self.west_inputs.len().max(self.east_outputs.len());
        let east_x = local_bounds.max.0 + 4;
        let size = DimSize(
            east_x + 1,
            translated.size.1.max(port_count * self.pin_spacing + 2),
            translated.size.2.max(2),
        );
        let mut world = World3D::new(size);

        for (position, block) in translated.iter_block() {
            world[position] = block;
        }

        let mut ports = Vec::new();
        for (index, name) in self.west_inputs.iter().enumerate() {
            ensure_graph_has_port(graph, name, PhysicalPortDirection::Input)?;
            let pin = Position(0, index * self.pin_spacing + 1, 1);
            place_pin(&mut world, pin);
            ports.push(PhysicalPort {
                name: name.clone(),
                direction: PhysicalPortDirection::Input,
                side: PhysicalPortSide::West,
                pin,
                facing: Direction::East,
            });
        }

        for (index, name) in self.east_outputs.iter().enumerate() {
            ensure_graph_has_port(graph, name, PhysicalPortDirection::Output)?;
            let pin = Position(east_x, index * self.pin_spacing + 1, 1);
            place_pin(&mut world, pin);
            ports.push(PhysicalPort {
                name: name.clone(),
                direction: PhysicalPortDirection::Output,
                side: PhysicalPortSide::East,
                pin,
                facing: Direction::West,
            });
        }
        ports.sort_by(|a, b| a.name.cmp(&b.name));

        let bbox =
            bounding_box(&world).ok_or_else(|| eyre::eyre!("macro candidate has no blocks"))?;
        let occupied_cells = world
            .iter_block()
            .into_iter()
            .map(|(position, _)| position)
            .collect::<HashSet<_>>();
        let blocked_cells = occupied_cells.clone();
        let cost = world_compact_cost(&world);

        Ok(MacroLayoutCandidate {
            unit_id: self.unit_id.clone(),
            world_fragment: world,
            bbox,
            ports,
            occupied_cells,
            blocked_cells,
            cost,
        })
    }
}

fn sorted<const N: usize>(names: [&str; N]) -> Vec<String> {
    let mut names = names.into_iter().map(str::to_string).collect::<Vec<_>>();
    names.sort();
    names
}

fn translate_local_without_switches(local: &LocalPlacement, offset: Position) -> World3D {
    let mut world = World3D::new(DimSize(
        local.world.size.0 + offset.0,
        local.world.size.1 + offset.1,
        local.world.size.2 + offset.2,
    ));

    for (position, block) in local.world.iter_block() {
        let position = Position(
            position.0 + offset.0,
            position.1 + offset.1,
            position.2 + offset.2,
        );
        world[position] = if matches!(block.kind, BlockKind::Switch { .. }) {
            redstone()
        } else {
            block
        };
    }

    world
}

fn ensure_graph_has_port(
    graph: &LogicGraph,
    name: &str,
    direction: PhysicalPortDirection,
) -> eyre::Result<()> {
    let found = graph.nodes.iter().any(|node| match (&node.kind, direction) {
        (GraphNodeKind::Input(input), PhysicalPortDirection::Input) => input == name,
        (GraphNodeKind::Output(output), PhysicalPortDirection::Output) => output == name,
        _ => false,
    });
    if !found {
        eyre::bail!("graph does not expose {:?} port {}", direction, name);
    }
    Ok(())
}

fn place_pin(world: &mut World3D, pin: Position) {
    world[pin.down().expect("macro pins are placed above support")] = Block {
        kind: BlockKind::Cobble {
            on_count: 0,
            on_base_count: 0,
        },
        direction: Direction::None,
    };
    world[pin] = redstone();
}

fn redstone() -> Block {
    Block {
        kind: BlockKind::Redstone {
            on_count: 0,
            state: 0,
            strength: 0,
        },
        direction: Direction::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::predefined_logics;
    use crate::transform::place_and_route::local_placer::{
        InputPlacementStrategy, LocalPlacer, LocalPlacerConfig, NotRouteStrategy,
        PlacementSamplingPolicy, TorchPlacementStrategy,
    };
    use crate::transform::place_and_route::sampling::SamplingPolicy;
    use crate::world::position::DimSize;

    #[test]
    fn half_adder_macro_candidate_exposes_only_boundary_pins() -> eyre::Result<()> {
        let graph = predefined_logics::buffered_binary_half_adder_graph()?;
        let placer = LocalPlacer::new(graph.clone(), fast_macro_config())?;
        let local = placer
            .generate_placements(DimSize(12, 12, 5), None)
            .into_iter()
            .min_by_key(|placement| world_compact_cost(&placement.world))
            .expect("half adder local candidate");

        let candidate = MacroCandidateBuilder::new("half_adder")
            .west_inputs(["a", "b"])
            .east_outputs(["c", "s"])
            .build(&graph, &local)?;

        assert_eq!(
            candidate
                .ports
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c", "s"]
        );
        assert!(
            candidate
                .ports
                .iter()
                .all(|p| p.pin.0 == 0 || p.pin.0 == candidate.bbox.max.0)
        );
        assert!(candidate
            .world_fragment
            .iter_block()
            .into_iter()
            .all(|(_, block)| !matches!(block.kind, BlockKind::Switch { .. })));

        Ok(())
    }

    fn fast_macro_config() -> LocalPlacerConfig {
        LocalPlacerConfig {
            random_seed: 42,
            greedy_input_generation: true,
            input_placement_strategy: InputPlacementStrategy::Boundary,
            step_sampling_policy: SamplingPolicy::Random(2048),
            placement_sampling_policy: PlacementSamplingPolicy::Cost {
                count: 128,
                random_count: 16,
                start_step: 6,
            },
            leak_sampling: false,
            route_torch_directly: true,
            torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
            not_route_strategy: NotRouteStrategy::DirectOnly,
            max_not_route_step: 3,
            not_route_step_sampling_policy: SamplingPolicy::Random(128),
            max_route_step: 3,
            route_step_sampling_policy: SamplingPolicy::Random(128),
        }
    }
}
