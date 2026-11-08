//! Preserve the hard-powered attachment of exact solver input switches.
use super::*;

/// A lever may power both its attachment block and adjacent dust. The generic
/// conversion sorts those contacts by coordinates; its first dust contact is
/// not an equivalent hard-power replacement for the attachment. Keep all
/// contacts for isolation checks, but ask the input diode to drive the solid.
/// Shared attachments are allowed here: final external drivers must still be
/// distinct and pass the complete post-adaptation truth/backfeed checks.
pub(super) fn restore_attached_input_targets(
    candidate: &mut LayoutCandidate,
    original: &PlacedWorld,
    module_ports: &[CandidatePort],
) -> usize {
    let mut changed = 0;
    for port in candidate
        .ports
        .iter_mut()
        .filter(|p| p.direction == PhysicalPortDirection::Input)
    {
        let Some(spec) = module_ports.iter().find(|p| p.name == port.name) else {
            continue;
        };
        let Some(input) = original.inputs.iter().find(|p| p.name == spec.target) else {
            continue;
        };
        let switch = input.position();
        let block = original.world[switch];
        if !block.kind.is_switch() {
            continue;
        }
        let Some(target) = switch
            .walk(block.direction)
            .filter(|p| original.world.size.bound_on(*p) && original.world[*p].kind.is_cobble())
        else {
            continue;
        };
        changed += usize::from(port.position != target);
        port.position = target;
        port.route_position = Some(switch);
        port.access_points.retain(|p| *p != target);
        port.access_points.insert(0, target);
        port.connection = PortConnection::InputDiode;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::Direction;
    #[test]
    fn coordinate_sorted_dust_does_not_replace_hard_powered_attachment() -> eyre::Result<()> {
        let mut world = World3D::new(DimSize(6, 5, 3));
        let switch = Position(2, 2, 1);
        let support = Position(3, 2, 1);
        let dust = Position(1, 2, 1);
        world[support] = PlacedNode::new_cobble(support).block;
        world[dust.down().unwrap()] = PlacedNode::new_cobble(dust.down().unwrap()).block;
        world[dust] = PlacedNode::new_redstone(dust).block;
        world[switch] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::East,
        };
        let placed = PlacedWorld {
            world: world.clone(),
            inputs: vec![OutputEndpoint::new("solver_name".into(), switch)],
            outputs: vec![],
        };
        let ports = vec![PhysicalPort {
            name: "public_name".into(),
            direction: PhysicalPortDirection::Input,
            position: dust,
            route_position: Some(switch),
            access_points: vec![dust, support],
            connection: PortConnection::InputDiode,
        }];
        let mut candidate = LayoutCandidate::from_world("test".into(), world, ports)?;
        let module_ports = [CandidatePort::new(
            "public_name",
            "solver_name",
            PhysicalPortDirection::Input,
        )];
        assert_eq!(
            restore_attached_input_targets(&mut candidate, &placed, &module_ports),
            1
        );
        assert_eq!(candidate.ports[0].position, support);
        assert_eq!(candidate.ports[0].route_position, Some(switch));
        assert_eq!(candidate.ports[0].access_points, vec![support, dust]);
        Ok(())
    }
}
