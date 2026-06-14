use std::collections::HashSet;

use eyre::ContextCompat;

use crate::transform::place_and_route::estimate::{bounding_box, BoundingBox};
use crate::world::position::Position;
use crate::world::World3D;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalPortDirection {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortConnection {
    Direct,
    InputDiode,
    OutputDiode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalPort {
    pub name: String,
    pub direction: PhysicalPortDirection,
    pub position: Position,
    pub route_position: Option<Position>,
    pub access_points: Vec<Position>,
    pub connection: PortConnection,
}

impl PhysicalPort {
    pub fn routing_access_positions(&self) -> Vec<Position> {
        if self.access_points.is_empty() {
            return vec![self.route_position.unwrap_or(self.position)];
        }
        self.access_points.clone()
    }

    pub fn primary_route_position(&self) -> Position {
        self.routing_access_positions()
            .into_iter()
            .next()
            .unwrap_or(self.position)
    }

    pub fn requires_input_diode(&self) -> bool {
        self.connection == PortConnection::InputDiode
    }

    pub fn requires_output_diode(&self) -> bool {
        self.connection == PortConnection::OutputDiode
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayoutCandidateCost {
    pub block_count: usize,
    pub bbox_volume: usize,
}

#[derive(Clone, Debug)]
pub struct LayoutCandidate {
    pub module_name: String,
    pub world: World3D,
    pub bbox: BoundingBox,
    pub ports: Vec<PhysicalPort>,
    pub occupied_cells: HashSet<Position>,
    pub blocked_cells: HashSet<Position>,
    pub cost: LayoutCandidateCost,
}

impl LayoutCandidate {
    pub fn from_world(
        module_name: String,
        world: World3D,
        ports: Vec<PhysicalPort>,
    ) -> eyre::Result<Self> {
        let bbox = bounding_box(&world).context("layout candidate world has no blocks")?;
        let occupied_cells = world
            .iter_block()
            .into_iter()
            .map(|(position, _)| position)
            .collect::<HashSet<_>>();
        let cost = LayoutCandidateCost {
            block_count: occupied_cells.len(),
            bbox_volume: bbox.volume(),
        };

        Ok(Self {
            module_name,
            world,
            bbox,
            ports,
            occupied_cells,
            blocked_cells: HashSet::new(),
            cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_port(connection: PortConnection, access_points: Vec<Position>) -> PhysicalPort {
        PhysicalPort {
            name: "p".to_owned(),
            direction: PhysicalPortDirection::Input,
            position: Position(1, 2, 3),
            route_position: Some(Position(4, 5, 6)),
            access_points,
            connection,
        }
    }

    #[test]
    fn physical_port_prefers_explicit_access_points() {
        let port = test_port(PortConnection::Direct, vec![Position(7, 8, 9)]);

        assert_eq!(port.routing_access_positions(), vec![Position(7, 8, 9)]);
    }

    #[test]
    fn physical_port_falls_back_to_route_position_as_access_point() {
        let port = test_port(PortConnection::Direct, Vec::new());

        assert_eq!(port.routing_access_positions(), vec![Position(4, 5, 6)]);
    }

    #[test]
    fn physical_port_connection_describes_diode_requirement() {
        let input = test_port(PortConnection::InputDiode, Vec::new());
        let output = test_port(PortConnection::OutputDiode, Vec::new());

        assert!(input.requires_input_diode());
        assert!(!input.requires_output_diode());
        assert!(!output.requires_input_diode());
        assert!(output.requires_output_diode());
    }
}
