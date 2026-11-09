use super::*;
use crate::transform::place_and_route::global_pnr::router::{
    materialize_input_diode, materialize_output_diode,
};

/// Keep routing support blocks off the public dust drivers. This changes only
/// clearance metadata, and is called only for the opt-in prepared recipes.
pub(super) fn reserve_driver_headroom(candidate: &mut LayoutCandidate) {
    for port in &candidate.ports {
        if !matches!(
            port.connection,
            PortConnection::Direct | PortConnection::RestoredDirect
        ) || !candidate.world[port.position].kind.is_redstone()
        {
            continue;
        }
        let above = port.position.up();
        if candidate.world.size.bound_on(above) && candidate.world[above].kind.is_air() {
            candidate.blocked_cells.insert(above);
        }
    }
    candidate.include_reserved_cells_in_bounds();
}

/// Reserve the approach geometry before unrelated nets can occupy it.
/// This is used only for the opt-in combinational external-port policy.
pub(super) fn reserve(candidate: LayoutCandidate) -> eyre::Result<Option<LayoutCandidate>> {
    let shift = |p: Position| Position(p.0 + 3, p.1 + 3, p.2 + 1);
    let mut world = World3D::new(DimSize(
        candidate.world.size.0 + 6,
        candidate.world.size.1 + 6,
        candidate.world.size.2 + 2,
    ));
    for (position, block) in candidate.world.iter_block() {
        world[shift(position)] = block;
    }
    let mut ports = candidate.ports;
    for port in &mut ports {
        port.position = shift(port.position);
        port.route_position = port.route_position.map(shift);
        port.access_points = port.access_points.iter().copied().map(shift).collect();
    }
    let unreserved_world =
        tracing::enabled!(target: "redstone_compiler::global_pnr", tracing::Level::DEBUG)
            .then(|| world.clone());
    // Outputs get first choice: their escape needs a free first hop from the
    // internal source; later global nets must not be allowed to consume it.
    for port in &mut ports {
        if port.direction != PhysicalPortDirection::Output {
            continue;
        }
        let Some((next_world, driver)) =
            materialize_output_diode(&world, port.position, &port.routing_access_positions())
        else {
            tracing::debug!(target: "redstone_compiler::global_pnr", module=%candidate.module_name, port=%port.name, position=?port.position, contacts=?port.routing_access_positions(), stage="output", "IO diode reservation rejected candidate");
            return Ok(None);
        };
        world = next_world;
        port.position = driver;
        port.route_position = Some(driver);
        port.access_points = vec![driver];
        port.connection = PortConnection::Direct;
    }
    let mut drivers = Vec::new();
    for port in &mut ports {
        if port.direction != PhysicalPortDirection::Input || !port.requires_input_diode() {
            continue;
        }
        let contacts = port.routing_access_positions();
        // The logical terminal is authoritative. route_position may be the
        // former (now empty) switch site, useful only as preferred orientation.
        let Some((next_world, driver)) = materialize_input_diode(
            &world,
            port.position,
            port.route_position,
            1,
            &contacts,
            &drivers,
        ) else {
            let independently_materializable = unreserved_world.as_ref().map(|original| {
                materialize_input_diode(
                    original,
                    port.position,
                    port.route_position,
                    1,
                    &contacts,
                    &[],
                )
                .is_some()
            });
            tracing::debug!(target: "redstone_compiler::global_pnr", module=%candidate.module_name, port=%port.name, position=?port.position, preferred=?port.route_position, contacts=?contacts, previous_inputs=drivers.len(), ?independently_materializable, stage="input", "IO diode reservation rejected candidate");
            return Ok(None);
        };
        world = next_world;
        drivers.push(driver);
        port.position = driver;
        port.route_position = Some(driver);
        port.access_points = vec![driver];
        port.connection = PortConnection::Direct;
    }
    world.initialize_redstone_states();
    let mut reserved = LayoutCandidate::from_world(candidate.module_name, world, ports)?;
    reserved.blocked_cells = candidate.blocked_cells.into_iter().map(shift).collect();
    if certify_restored_outputs(&reserved).unwrap_or(false) {
        for port in &mut reserved.ports {
            if port.direction == PhysicalPortDirection::Output {
                port.connection = PortConnection::RestoredDirect;
            }
        }
    }
    reserve_driver_headroom(&mut reserved);
    Ok(Some(reserved))
}

/// Certification is bounded independently of local synthesis size. Larger
/// cells retain conservative Direct ports rather than an unverified promise.
fn certify_restored_outputs(candidate: &LayoutCandidate) -> eyre::Result<bool> {
    let inputs = candidate
        .ports
        .iter()
        .filter(|p| p.direction == PhysicalPortDirection::Input)
        .collect::<Vec<_>>();
    if inputs.len() > 8 {
        return Ok(false);
    }
    let world = crate::world::World::from(&candidate.world);
    for mask in 0..(1usize << inputs.len()) {
        let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 1024, 200_000, 0)?;
        sim.drive_inputs_with_limits(
            inputs
                .iter()
                .enumerate()
                .map(|(bit, p)| (p.position, mask & (1 << bit) != 0))
                .collect(),
            1024,
            200_000,
        )?;
        for output in candidate
            .ports
            .iter()
            .filter(|p| p.direction == PhysicalPortDirection::Output)
        {
            let block = sim.world()[output.position].kind;
            if block.is_powered() && !matches!(block, BlockKind::Redstone { strength: 15, .. }) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attenuated_output_cannot_claim_restored_drive() -> eyre::Result<()> {
        let mut world = World3D::new(DimSize(5, 3, 3));
        for x in 1..=3 {
            let p = Position(x, 1, 1);
            world[p.down().unwrap()] = PlacedNode::new_cobble(p.down().unwrap()).block;
            world[p] = PlacedNode::new_redstone(p).block;
        }
        world.initialize_redstone_states();
        let ports = [
            (PhysicalPortDirection::Input, Position(1, 1, 1)),
            (PhysicalPortDirection::Output, Position(3, 1, 1)),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (direction, position))| PhysicalPort {
            name: format!("pin{i}"),
            direction,
            position,
            route_position: Some(position),
            access_points: vec![position],
            connection: PortConnection::Direct,
        })
        .collect();
        let candidate = LayoutCandidate::from_world("attenuated".into(), world, ports)?;
        assert!(!certify_restored_outputs(&candidate)?);
        Ok(())
    }
    #[test]
    fn driver_headroom_is_air_metadata_and_preserves_prior_reservations() -> eyre::Result<()> {
        let mut world = World3D::new(DimSize(5, 5, 4));
        let input = Position(2, 2, 1);
        let output = Position(3, 2, 1);
        for p in [input, output] {
            world[p.down().unwrap()] = PlacedNode::new_cobble(p.down().unwrap()).block;
            world[p] = PlacedNode::new_redstone(p).block;
        }
        world[output.up()] = PlacedNode::new_cobble(output.up()).block;
        let before = world.iter_block();
        let ports = [
            (input, PhysicalPortDirection::Input),
            (output, PhysicalPortDirection::Output),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (position, direction))| PhysicalPort {
            name: format!("p{i}"),
            direction,
            position,
            route_position: Some(position),
            access_points: vec![position],
            connection: PortConnection::Direct,
        })
        .collect();
        let mut candidate = LayoutCandidate::from_world("test".into(), world, ports)?;
        let prior = Position(0, 0, 3);
        candidate.blocked_cells.insert(prior);
        reserve_driver_headroom(&mut candidate);
        assert_eq!(candidate.world.iter_block(), before);
        assert_eq!(
            candidate.blocked_cells,
            [prior, input.up()].into_iter().collect()
        );
        assert!(!candidate.occupied_cells.contains(&input.up()));
        assert_eq!(candidate.bbox.max.2, 3);
        assert_eq!(candidate.cost.bbox_height, 4);
        assert_eq!(candidate.cost.block_count, before.len());
        Ok(())
    }
}
