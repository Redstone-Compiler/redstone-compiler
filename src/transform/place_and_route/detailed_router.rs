use std::collections::HashSet;

use crate::transform::place_and_route::place_bound::PlaceBound;
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{Block, BlockKind, Direction};
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

pub fn try_generate_cobble_node(
    world: &World3D,
    cobble_pos: Position,
    except: &[Position],
) -> Option<PlacedNode> {
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
    let Some(cobble_pos) = bound.position().walk(Direction::Bottom) else {
        return PlaceRedstoneResult::Rejected(RouteRejectReason::NoBottomForCobble);
    };
    let cobble_except = (world[prev].kind.is_torch())
        .then_some(vec![cobble_pos, prev])
        .unwrap_or_default();
    let Some(cobble_node) = try_generate_cobble_node(world, cobble_pos, &cobble_except) else {
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
    if !target_powers_redstone(&new_world, prev, redstone_node.position) {
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
    let Some(cobble_pos) = bound.position().walk(Direction::Bottom) else {
        return PlaceRepeaterResult::Rejected(RouteRejectReason::NoBottomForCobble);
    };
    let Some(cobble_node) = try_generate_cobble_node(world, cobble_pos, &[]) else {
        return PlaceRepeaterResult::Rejected(RouteRejectReason::CobbleConflict);
    };
    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);

    let repeater_node = PlacedNode {
        position: bound.position(),
        block: Block {
            kind: BlockKind::Repeater {
                is_on: false,
                is_locked: false,
                delay: 1,
                lock_input1: None,
                lock_input2: None,
            },
            direction,
        },
    };
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

pub fn target_powers_redstone(world: &World3D, target: Position, redstone: Position) -> bool {
    if world[target].kind.is_air() || world[target].kind.is_cobble() {
        return false;
    }
    let target_node = PlacedNode::new(target, world[target]);
    target_node
        .propagation_bound(Some(world))
        .into_iter()
        .filter(|bound| bound.is_bound_on(world))
        .any(|bound| {
            bound.position() == redstone
                || bound
                    .propagate_to(world)
                    .into_iter()
                    .any(|(_, position)| position == redstone)
        })
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
            bound.position() == position
                || bound
                    .propagate_to(world)
                    .into_iter()
                    .any(|(_, propagated)| propagated == position)
        })
}
