use super::*;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::World3D;

fn empty_world() -> World3D {
    World3D::new(DimSize(6, 6, 4))
}

fn switch(direction: Direction) -> Block {
    Block {
        kind: BlockKind::Switch { is_on: false },
        direction,
    }
}

fn torch(direction: Direction) -> Block {
    Block {
        kind: BlockKind::Torch { is_on: false },
        direction,
    }
}

fn config(max_route_step: usize) -> LocalPlacerConfig {
    LocalPlacerConfig {
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        step_sampling_policy: SamplingPolicy::None,
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_route_step,
        route_step_sampling_policy: SamplingPolicy::None,
    }
}

#[test]
fn generate_inputs_uses_greedy_boundary_switch_placements() {
    let world = World3D::new(DimSize(3, 3, 2));
    let generated = generate_inputs(&config(1), &world, BlockKind::Switch { is_on: false });

    assert_eq!(generated.len(), 12);
    for (world, position) in generated {
        assert!(position.0 == 0 || position.1 == 0);
        assert!(world[position].kind.is_switch());
        assert_eq!(world[position].direction, Direction::East);
    }
}

#[test]
fn generate_inputs_can_search_anywhere() {
    let world = World3D::new(DimSize(3, 3, 2));
    let mut config = config(1);
    config.input_placement_strategy = InputPlacementStrategy::Anywhere;

    let generated = generate_inputs(&config, &world, BlockKind::Switch { is_on: false });

    assert_eq!(generated.len(), 18);
}

#[test]
fn place_torch_with_cobble_places_support_block() {
    let world = empty_world();
    let torch_pos = Position(2, 1, 1);

    let (world, placed_torch_pos, cobble_pos) =
        place_torch_with_cobble(&world, torch(Direction::West), torch_pos).unwrap();

    assert_eq!(placed_torch_pos, torch_pos);
    assert_eq!(cobble_pos, Position(1, 1, 1));
    assert!(world[torch_pos].kind.is_torch());
    assert!(world[cobble_pos].kind.is_cobble());
}

#[test]
fn generate_routes_to_cobble_accepts_direct_hard_connection() {
    let mut world = empty_world();
    let source = Position(1, 1, 1);
    let cobble_pos = Position(2, 1, 1);
    place_node(&mut world, PlacedNode::new(source, switch(Direction::East)));
    place_node(&mut world, PlacedNode::new_cobble(cobble_pos));

    let routes =
        generate_routes_to_cobble(&config(1), &world, source, Position(3, 1, 1), cobble_pos);

    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].1, cobble_pos);
}

#[test]
fn generate_routes_to_cobble_can_use_redstone_route() {
    let mut world = empty_world();
    let source = Position(1, 1, 1);
    let torch_pos = Position(3, 1, 1);
    let cobble_pos = Position(3, 1, 0);
    let mut config = config(1);
    config.not_route_strategy = NotRouteStrategy::RedstoneOnly;
    place_node(
        &mut world,
        PlacedNode::new(source, torch(Direction::Bottom)),
    );
    place_node(&mut world, PlacedNode::new_cobble(cobble_pos));
    place_node(
        &mut world,
        PlacedNode::new(torch_pos, torch(Direction::Bottom)),
    );

    let routes = generate_routes_to_cobble(&config, &world, source, torch_pos, cobble_pos);

    assert!(routes.iter().any(|(_, pos)| *pos == Position(2, 1, 1)));
}

#[test]
fn exhaustive_config_disables_search_reduction_knobs() {
    let config = LocalPlacerConfig::exhaustive(12);

    assert!(!config.greedy_input_generation);
    assert_eq!(
        config.input_placement_strategy,
        InputPlacementStrategy::Anywhere
    );
    assert_eq!(config.step_sampling_policy, SamplingPolicy::None);
    assert!(!config.route_torch_directly);
    assert_eq!(
        config.torch_placement_strategy,
        TorchPlacementStrategy::AnywhereNonAdjacent
    );
    assert_eq!(
        config.not_route_strategy,
        NotRouteStrategy::DirectAndRedstone
    );
    assert_eq!(config.max_route_step, 12);
    assert_eq!(config.route_step_sampling_policy, SamplingPolicy::None);
}

#[test]
fn generate_or_routes_init_states_includes_top_cobble_variants() {
    let mut world = empty_world();
    let from = Position(1, 1, 1);
    let to = Position(4, 1, 1);
    place_node(&mut world, PlacedNode::new(from, torch(Direction::Bottom)));
    place_node(&mut world, PlacedNode::new(to, torch(Direction::Bottom)));

    let (states, debug) = generate_or_routes_init_states(&world, from, to);

    assert_eq!(states.len(), 4);
    assert!(debug.rejected.is_empty());
    assert_eq!(
        states
            .iter()
            .filter(|(world, _, _)| world[from.up()].kind.is_cobble())
            .count(),
        2
    );
    assert_eq!(
        states
            .iter()
            .filter(|(world, _, _)| world[to.up()].kind.is_cobble())
            .count(),
        2
    );
}

#[test]
fn place_redstone_with_cobble_places_redstone_and_support() {
    let mut world = empty_world();
    let prev = Position(1, 1, 1);
    let to = Position(4, 1, 1);
    let redstone_pos = Position(2, 1, 1);
    place_node(&mut world, PlacedNode::new(prev, torch(Direction::Bottom)));
    place_node(&mut world, PlacedNode::new(to, torch(Direction::Bottom)));

    let result = place_redstone_with_cobble(
        &world,
        PlaceBound(PropagateType::Torch, redstone_pos, Direction::West),
        prev,
        to,
    );

    let PlaceRedstoneResult::Placed(world, redstone_node) = result else {
        panic!("expected redstone placement to succeed");
    };
    assert_eq!(redstone_node.position, redstone_pos);
    assert!(world[redstone_pos].kind.is_redstone());
    assert!(world[Position(2, 1, 0)].kind.is_cobble());
}

#[test]
fn has_connection_with_requires_world_after_cobble_placement() {
    let mut world = empty_world();
    let to = Position(2, 2, 1);
    let redstone_pos = Position(3, 2, 2);
    place_node(&mut world, PlacedNode::new(to, torch(Direction::Bottom)));

    let mut new_world = world.clone();
    place_node(&mut new_world, PlacedNode::new_cobble(to.up()));
    let redstone_node = PlacedNode::new_redstone(redstone_pos);

    assert!(!redstone_node.has_connection_with(&world, to));
    assert!(redstone_node.has_connection_with(&new_world, to));
}

#[test]
fn generate_or_routes_finds_adjacent_torch_route() {
    let mut world = empty_world();
    let from = Position(1, 1, 1);
    let to = Position(3, 1, 1);
    place_node(&mut world, PlacedNode::new(from, torch(Direction::Bottom)));
    place_node(&mut world, PlacedNode::new(to, torch(Direction::Bottom)));

    let result = generate_or_routes(&config(1), &world, from, to);

    assert!(!result.routes.is_empty());
    assert!(result.debug.candidates_found > 0);
    assert_eq!(result.debug.route_calls, 1);
}
