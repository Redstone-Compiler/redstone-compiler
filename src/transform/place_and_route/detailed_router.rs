use std::collections::HashSet;

use crate::transform::place_and_route::place_bound::PlaceBound;
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{BlockKind, Direction};
use crate::world::position::Position;
use crate::world::World3D;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteRejectReason {
    InitInputTopCobbleConflict,
    InitOutputTopCobbleConflict,
    OutOfBounds,
    NoBottomForCobble,
    CobbleConflict,
    RedstoneConflict,
    DisconnectedRoute,
    ShortCircuit,
}

pub enum PlaceRedstoneResult {
    Placed(World3D, PlacedNode),
    Rejected(RouteRejectReason),
}

pub enum PlaceRepeaterResult {
    Placed(World3D, PlacedNode),
    Rejected(RouteRejectReason),
}

pub fn place_node(world: &mut World3D, node: PlacedNode) {
    if world[node.position] == node.block {
        assert!(world[node.position].kind.is_cobble());
        return;
    }

    world[node.position] = node.block;
    if node.block.kind.is_redstone() {
        world.update_redstone_states(node.position);
    }
}

/// Keep local candidate enumeration stable; global routing may reuse fixed floors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SupportReusePolicy {
    #[default]
    FreshOnly,
    ReuseExisting,
}

pub fn try_generate_cobble_node(
    world: &World3D,
    cobble_pos: Position,
    except: &[Position],
) -> Option<PlacedNode> {
    try_generate_cobble_node_with_policy(world, cobble_pos, except, SupportReusePolicy::FreshOnly)
}

pub fn try_generate_cobble_node_with_policy(
    world: &World3D,
    cobble_pos: Position,
    except: &[Position],
    support_policy: SupportReusePolicy,
) -> Option<PlacedNode> {
    if !world.size.bound_on(cobble_pos) {
        return None;
    }
    if support_policy == SupportReusePolicy::ReuseExisting && world[cobble_pos].kind.is_cobble() {
        // Reusing a support changes neither geometry nor its current power.
        // New dust/repeater placement still checks conflicts and shorts below.
        return Some(PlacedNode::new(cobble_pos, world[cobble_pos]));
    }
    if cobble_would_stack_above_side_torch_support(world, cobble_pos) {
        return None;
    }
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    if !cobble_node.has_conflict(world, &except.iter().copied().collect()) {
        Some(cobble_node)
    } else {
        None
    }
}

pub fn cobble_would_stack_above_side_torch_support(world: &World3D, cobble_pos: Position) -> bool {
    let Some(below) = cobble_pos.down() else {
        return false;
    };
    world.size.bound_on(below)
        && world[below].kind.is_cobble()
        && below.cardinal().into_iter().any(|position| {
            world.size.bound_on(position)
                && matches!(world[position].kind, BlockKind::Torch { .. })
                && world[position].direction == position.diff(below)
        })
}

pub fn place_redstone_with_cobble(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
) -> PlaceRedstoneResult {
    place_redstone_with_cobble_and_allowed_shorts(world, bound, prev, to, None)
}

pub fn place_redstone_with_cobble_and_allowed_shorts(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
    allowed_shorts: Option<&HashSet<Position>>,
) -> PlaceRedstoneResult {
    place_redstone_with_cobble_and_policy(
        world,
        bound,
        prev,
        to,
        allowed_shorts,
        SupportReusePolicy::FreshOnly,
    )
}

pub fn place_redstone_with_cobble_and_policy(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
    allowed_shorts: Option<&HashSet<Position>>,
    support_policy: SupportReusePolicy,
) -> PlaceRedstoneResult {
    let Some(cobble_pos) = bound.position().walk(Direction::Bottom) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::NoBottomForCobble);
    };
    let cobble_except = (world[prev].kind.is_torch())
        .then_some(vec![cobble_pos, prev])
        .unwrap_or_default();
    let Some(cobble_node) =
        try_generate_cobble_node_with_policy(world, cobble_pos, &cobble_except, support_policy)
    else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::CobbleConflict);
    };
    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);

    let bound_pos = bound.position();
    let Some(bound_back_pos) = bound_pos.walk(bound.direction()) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    };
    let redstone_node = PlacedNode::new_redstone(bound_pos);
    let mut except = [prev, bound_back_pos, bound_pos, to, to.up()]
        .into_iter()
        .collect::<HashSet<_>>();
    if let Some(allowed_shorts) = allowed_shorts {
        except.extend(allowed_shorts.iter().copied());
    }
    let mut short_except = [prev, bound_pos, to, to.up()]
        .into_iter()
        .collect::<HashSet<_>>();
    if let Some(allowed_shorts) = allowed_shorts {
        short_except.extend(allowed_shorts.iter().copied());
    }
    if redstone_node.has_conflict(&new_world, &except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::RedstoneConflict);
    }
    if redstone_node.has_short(world, &short_except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
    }
    place_node(&mut new_world, redstone_node);
    new_world.update_redstone_states(prev);
    if !target_powers_position(&new_world, prev, redstone_node.position) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::DisconnectedRoute);
    }
    if redstone_node.has_short(&new_world, &short_except) {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
    }
    if let BlockKind::Torch { .. } = world[prev].kind {
        if let Some(source_cobble) = prev.walk(world[prev].direction) {
            if redstone_powers_cobble(&new_world, redstone_node.position, source_cobble) {
                return PlaceRedstoneResult::Rejected(RouteRejectReason::ShortCircuit);
            }
        }
    }

    PlaceRedstoneResult::Placed(new_world, redstone_node)
}

pub fn place_repeater_with_cobble(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
    direction: Direction,
    allowed_shorts: Option<&HashSet<Position>>,
) -> PlaceRepeaterResult {
    place_repeater_with_cobble_and_policy(
        world,
        bound,
        prev,
        to,
        direction,
        allowed_shorts,
        SupportReusePolicy::FreshOnly,
    )
}

pub fn place_repeater_with_cobble_and_policy(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
    direction: Direction,
    allowed_shorts: Option<&HashSet<Position>>,
    support_policy: SupportReusePolicy,
) -> PlaceRepeaterResult {
    let Some(cobble_pos) = bound.position().walk(Direction::Bottom) else {
        return PlaceRepeaterResult::Rejected(RouteRejectReason::NoBottomForCobble);
    };
    let Some(cobble_node) =
        try_generate_cobble_node_with_policy(world, cobble_pos, &[], support_policy)
    else {
        return PlaceRepeaterResult::Rejected(RouteRejectReason::CobbleConflict);
    };
    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);

    let repeater_node = PlacedNode::new_repeater(bound.position(), direction);
    let mut except = [prev, bound.position(), to, to.up()]
        .into_iter()
        .collect::<HashSet<_>>();
    if let Some(allowed_shorts) = allowed_shorts {
        except.extend(allowed_shorts.iter().copied());
    }
    if repeater_node.has_conflict(&new_world, &except) {
        return PlaceRepeaterResult::Rejected(RouteRejectReason::RedstoneConflict);
    }
    place_node(&mut new_world, repeater_node);
    // A new repeater changes adjacent dust's connection shape just as a new
    // dust piece does. Test connectivity against that resulting shape.
    new_world.update_redstone_states(prev);
    if !target_powers_position(&new_world, prev, repeater_node.position) {
        return PlaceRepeaterResult::Rejected(RouteRejectReason::DisconnectedRoute);
    }

    PlaceRepeaterResult::Placed(new_world, repeater_node)
}

pub fn redstone_powers_cobble(world: &World3D, redstone: Position, cobble: Position) -> bool {
    if !world[redstone].kind.is_redstone() {
        return false;
    }
    world[cobble].kind.is_cobble()
        && PlacedNode::new(redstone, world[redstone])
            .propagation_bound(Some(world))
            .into_iter()
            .any(|bound| bound.position() == cobble)
}

pub fn target_powers_position(world: &World3D, target: Position, position: Position) -> bool {
    if world[target].kind.is_air() || world[target].kind.is_cobble() {
        return false;
    }
    let target_node = PlacedNode::new(target, world[target]);
    target_node
        .propagation_bound(Some(world))
        .into_iter()
        .filter(|bound| bound.is_bound_on(world))
        .any(|bound| {
            let propagates_to_position = bound
                .propagate_to(world)
                .into_iter()
                .any(|(_, propagated)| propagated == position);

            if world[position].kind.is_repeater() {
                return propagates_to_position;
            }

            bound.position() == position || propagates_to_position
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::place_and_route::global_pnr::router::{
        route_point_to_point_with_strategy, GlobalRoutingStrategy,
    };
    use crate::transform::place_and_route::place_bound::PropagateType;
    use crate::world::block::Block;
    use crate::world::position::DimSize;
    use crate::world::simulator::Simulator;
    use crate::world::World;

    #[test]
    fn reuse_does_not_apply_new_block_stack_rule_but_new_support_still_does() {
        let mut world = World3D::new(DimSize(5, 5, 5));
        let below = Position(2, 2, 1);
        let support = below.up();
        let torch = Position(3, 2, 1);
        world[below] = PlacedNode::new_cobble(below).block;
        world[support] = PlacedNode::new_cobble(support).block;
        world[torch] = Block {
            kind: BlockKind::Torch { is_on: true },
            direction: torch.diff(below),
        };
        assert!(cobble_would_stack_above_side_torch_support(&world, support));
        assert!(try_generate_cobble_node_with_policy(
            &world,
            support,
            &[],
            SupportReusePolicy::ReuseExisting
        )
        .is_some());
        world[support] = Block::default();
        assert!(world[support].kind.is_air());
        assert!(try_generate_cobble_node_with_policy(
            &world,
            support,
            &[],
            SupportReusePolicy::ReuseExisting
        )
        .is_none());
    }

    #[test]
    fn existing_support_is_reused_without_resetting_power() {
        let mut world = World3D::new(DimSize(5, 5, 4));
        let support = Position(2, 2, 1);
        world[support] = Block {
            kind: BlockKind::Cobble {
                on_count: 3,
                on_base_count: 2,
            },
            direction: Direction::None,
        };
        let existing = world[support];
        assert!(
            try_generate_cobble_node(&world, support, &[]).is_none(),
            "local placement's default must keep its original support-enumeration policy"
        );
        let node = try_generate_cobble_node_with_policy(
            &world,
            support,
            &[],
            SupportReusePolicy::ReuseExisting,
        )
        .unwrap();
        assert_eq!(node.block, existing);
        place_node(&mut world, node);
        assert_eq!(world[support], existing);
        world[support] = PlacedNode::new_redstone(support).block;
        assert!(try_generate_cobble_node_with_policy(
            &world,
            support,
            &[],
            SupportReusePolicy::ReuseExisting
        )
        .is_none());
    }

    #[test]
    fn perpendicular_repeater_refreshes_predecessor_dust_shape() -> eyre::Result<()> {
        let mut world = World3D::new(DimSize(5, 4, 3));
        for x in 0..5 {
            for y in 0..4 {
                let p = Position(x, y, 0);
                world[p] = PlacedNode::new_cobble(p).block;
            }
        }
        let switch = Position(1, 2, 1);
        let source = Position(2, 2, 1);
        let repeater = Position(2, 1, 1);
        let sink = Position(2, 0, 1);
        world[switch] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[source] = PlacedNode::new_redstone(source).block;
        world[sink] = PlacedNode::new_redstone(sink).block;
        world.initialize_redstone_states();
        let old_shape = world[source];
        let routed = match place_repeater_with_cobble_and_policy(
            &world,
            PlaceBound(PropagateType::Soft, repeater, Direction::South),
            source,
            sink,
            Direction::North,
            None,
            SupportReusePolicy::ReuseExisting,
        ) {
            PlaceRepeaterResult::Placed(world, _) => world,
            PlaceRepeaterResult::Rejected(reason) => {
                eyre::bail!("legal elbow rejected: {reason:?}")
            }
        };
        assert_ne!(
            routed[source], old_shape,
            "straight dust becomes an elbow toward the newly attached repeater"
        );
        assert!(target_powers_position(&routed, source, repeater));
        let mut simulator =
            Simulator::from_settled_with_limits_and_trace(&World::from(&routed), 128, 10_000, 0)?;
        for on in [false, true, false, true, false] {
            simulator.drive_inputs_with_limits(vec![(switch, on)], 128, 10_000)?;
            assert_eq!(simulator.world()[sink].kind.is_powered(), on);
        }
        Ok(())
    }

    #[test]
    fn routes_across_existing_cobble_floor_and_switches_on_and_off() -> eyre::Result<()> {
        let mut world = World3D::new(DimSize(26, 5, 4));
        for x in 0..26 {
            for y in 0..5 {
                let p = Position(x, y, 0);
                world[p] = PlacedNode::new_cobble(p).block;
            }
        }
        let source = Position(0, 2, 1);
        let sink = Position(22, 2, 1);
        world[source] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[sink] = PlacedNode::new_redstone(sink).block;
        world.initialize_redstone_states();
        let (route, routed) =
            route_point_to_point_with_strategy(&world, source, sink, GlobalRoutingStrategy::AStar)
                .map_err(|error| eyre::eyre!("floor route failed: {error:?}"))?;
        assert!(route.path.len() > 1);
        assert!(
            route
                .blocks
                .iter()
                .any(|(_, block)| block.kind.is_repeater()),
            "the long floor route must reuse supports for both dust and repeaters"
        );
        for x in 0..26 {
            for y in 0..5 {
                let p = Position(x, y, 0);
                assert_eq!(routed[p], world[p]);
            }
        }
        let converted = World::from(&routed);
        let mut simulator =
            Simulator::from_settled_with_limits_and_trace(&converted, 256, 50_000, 0)?;
        for powered in [true, false, true, false] {
            simulator.drive_inputs_with_limits(vec![(source, powered)], 256, 50_000)?;
            assert_eq!(simulator.world()[sink].kind.is_powered(), powered);
        }
        Ok(())
    }

    #[test]
    fn reusing_support_does_not_allow_foreign_torch_backfeed() {
        let mut world = World3D::new(DimSize(6, 5, 4));
        let source = Position(1, 2, 1);
        let wire = Position(2, 2, 1);
        let foreign_support = Position(3, 2, 1);
        let foreign_torch = Position(3, 2, 2);
        for p in [
            source.down().unwrap(),
            wire.down().unwrap(),
            foreign_support,
        ] {
            world[p] = PlacedNode::new_cobble(p).block;
        }
        world[source] = Block {
            kind: BlockKind::Switch { is_on: false },
            direction: Direction::West,
        };
        world[foreign_torch] = Block {
            kind: BlockKind::Torch { is_on: true },
            direction: Direction::Bottom,
        };
        let bound = PlaceBound(PropagateType::Soft, wire, wire.diff(source));
        assert!(matches!(
            place_redstone_with_cobble_and_policy(
                &world,
                bound,
                source,
                Position(5, 2, 1),
                None,
                SupportReusePolicy::ReuseExisting
            ),
            PlaceRedstoneResult::Rejected(
                RouteRejectReason::RedstoneConflict | RouteRejectReason::ShortCircuit
            )
        ));
    }
}
