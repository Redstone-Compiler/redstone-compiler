use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelPort {
    pub name: String,
    pub pin: Position,
}

#[derive(Debug, Clone)]
pub struct IoPanel {
    pub world: World3D,
    pub ports: Vec<PanelPort>,
}

pub fn build_input_panel(names: &[&str], spacing: usize) -> IoPanel {
    let names = sorted_names(names);
    let mut world = World3D::new(panel_size(1, names.len(), spacing));
    let mut ports = Vec::new();

    for (index, name) in names.into_iter().enumerate() {
        let pin = Position(0, index * spacing + 1, 1);
        place_cobble_support(&mut world, pin);
        world[pin] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::Bottom,
        };
        ports.push(PanelPort { name, pin });
    }

    IoPanel { world, ports }
}

pub fn build_output_panel(names: &[&str], spacing: usize, x: usize) -> IoPanel {
    let names = sorted_names(names);
    let mut world = World3D::new(panel_size(x + 1, names.len(), spacing));
    let mut ports = Vec::new();

    for (index, name) in names.into_iter().enumerate() {
        let pin = Position(x, index * spacing + 1, 1);
        place_cobble_support(&mut world, pin);
        world[pin] = Block {
            kind: BlockKind::Redstone {
                on_count: 0,
                state: 0,
                strength: 0,
            },
            direction: Direction::None,
        };
        ports.push(PanelPort { name, pin });
    }

    IoPanel { world, ports }
}

fn sorted_names(names: &[&str]) -> Vec<String> {
    let mut names = names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    names.sort();
    names
}

fn panel_size(width: usize, port_count: usize, spacing: usize) -> DimSize {
    DimSize(width, port_count.saturating_mul(spacing) + 2, 2)
}

fn place_cobble_support(world: &mut World3D, pin: Position) {
    let support = pin
        .down()
        .expect("panel pins are placed at z=1 and must have support below");
    world[support] = Block {
        kind: BlockKind::Cobble {
            on_count: 0,
            on_base_count: 0,
        },
        direction: Direction::None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_panel_places_switches_in_name_order_on_west_edge() {
        let panel = build_input_panel(&["cin", "a", "b"], 3);

        assert_eq!(
            panel
                .ports
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "cin"]
        );
        assert_eq!(panel.ports[0].pin, Position(0, 1, 1));
        assert_eq!(panel.ports[1].pin, Position(0, 4, 1));
        assert_eq!(panel.ports[2].pin, Position(0, 7, 1));

        for port in &panel.ports {
            assert!(matches!(panel.world[port.pin].kind, BlockKind::Switch { .. }));
            assert!(panel.world[port.pin.down().unwrap()].kind.is_cobble());
        }
    }

    #[test]
    fn output_panel_places_probe_redstone_in_name_order_on_east_edge() {
        let panel = build_output_panel(&["cout", "s"], 3, 12);

        assert_eq!(
            panel
                .ports
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec!["cout", "s"]
        );
        assert_eq!(panel.ports[0].pin, Position(12, 1, 1));
        assert_eq!(panel.ports[1].pin, Position(12, 4, 1));

        for port in &panel.ports {
            assert!(panel.world[port.pin].kind.is_redstone());
            assert!(panel.world[port.pin.down().unwrap()].kind.is_cobble());
        }
    }
}
