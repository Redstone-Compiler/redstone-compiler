use super::*;
use crate::graph::logic::LogicGraph;
use crate::graph::{Graph, GraphNode, GraphNodeKind};
use crate::sequential::layout::SequentialMacro;
use crate::sequential::{SequentialPrimitive, SequentialType};
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
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        step_sampling_policy: SamplingPolicy::None,
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: max_route_step,
        not_route_step_sampling_policy: SamplingPolicy::None,
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
fn placement_cost_penalizes_spread_future_join_inputs() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a|b", "c")?.prepare_place()?;
    let placer = LocalPlacer::new(graph.clone(), config(1))?;
    let input_a = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == "a"))
        .unwrap()
        .id;
    let input_b = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == "b"))
        .unwrap()
        .id;
    let step = placer
        .visit_orders
        .iter()
        .position(|id| *id == input_a)
        .unwrap()
        .max(
            placer
                .visit_orders
                .iter()
                .position(|id| *id == input_b)
                .unwrap(),
        );
    let world = empty_world();

    let near = [(input_a, Position(1, 1, 1)), (input_b, Position(2, 1, 1))]
        .into_iter()
        .collect();
    let far = [(input_a, Position(1, 1, 1)), (input_b, Position(5, 5, 1))]
        .into_iter()
        .collect();

    assert!(placer.placement_cost(step, &world, &far) > placer.placement_cost(step, &world, &near));

    Ok(())
}

#[test]
fn cost_sampling_keeps_lower_cost_candidates() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a|b", "c")?.prepare_place()?;
    let placer = LocalPlacer::new(graph.clone(), config(1))?;
    let input_a = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == "a"))
        .unwrap()
        .id;
    let input_b = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == "b"))
        .unwrap()
        .id;
    let step = placer
        .visit_orders
        .iter()
        .position(|id| *id == input_a)
        .unwrap()
        .max(
            placer
                .visit_orders
                .iter()
                .position(|id| *id == input_b)
                .unwrap(),
        );
    let world = empty_world();
    let near = [(input_a, Position(1, 1, 1)), (input_b, Position(2, 1, 1))]
        .into_iter()
        .collect();
    let far = [(input_a, Position(1, 1, 1)), (input_b, Position(5, 5, 1))]
        .into_iter()
        .collect();

    let sampled = placer.sample_by_cost(step, vec![(world.clone(), far), (world, near)], 1, 0);

    assert_eq!(sampled.len(), 1);
    assert_eq!(sampled[0].1[&input_b], Position(2, 1, 1));

    Ok(())
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
fn generate_routes_to_cobble_skips_redstone_route_for_non_diode_source() {
    let mut world = empty_world();
    let source = Position(1, 1, 1);
    let torch_pos = Position(3, 1, 1);
    let cobble_pos = Position(3, 1, 0);
    let mut config = config(1);
    config.not_route_strategy = NotRouteStrategy::DirectAndRedstone;
    place_node(&mut world, PlacedNode::new_cobble(Position(1, 1, 0)));
    place_node(&mut world, PlacedNode::new_redstone(source));
    place_node(&mut world, PlacedNode::new_cobble(cobble_pos));
    place_node(
        &mut world,
        PlacedNode::new(torch_pos, torch(Direction::Bottom)),
    );

    let routes = generate_routes_to_cobble(&config, &world, source, torch_pos, cobble_pos);

    assert!(routes.is_empty());
}

#[test]
fn exhaustive_config_disables_search_reduction_knobs() {
    let config = LocalPlacerConfig::exhaustive(12);

    assert_eq!(config.random_seed, 42);
    assert!(!config.greedy_input_generation);
    assert_eq!(
        config.input_placement_strategy,
        InputPlacementStrategy::Anywhere
    );
    assert_eq!(config.step_sampling_policy, SamplingPolicy::None);
    assert_eq!(
        config.placement_sampling_policy,
        PlacementSamplingPolicy::StepPolicy
    );
    assert!(!config.route_torch_directly);
    assert_eq!(
        config.torch_placement_strategy,
        TorchPlacementStrategy::AnywhereNonAdjacent
    );
    assert_eq!(
        config.not_route_strategy,
        NotRouteStrategy::DirectAndRedstone
    );
    assert_eq!(config.max_not_route_step, 12);
    assert_eq!(config.not_route_step_sampling_policy, SamplingPolicy::None);
    assert_eq!(config.max_route_step, 12);
    assert_eq!(config.route_step_sampling_policy, SamplingPolicy::None);
}

#[test]
fn random_sampling_uses_explicit_seed() {
    let src = (0..32).collect::<Vec<_>>();

    let first = SamplingPolicy::Random(8).sample_with_seed(src.clone(), 7);
    let second = SamplingPolicy::Random(8).sample_with_seed(src.clone(), 7);
    let different_seed = SamplingPolicy::Random(8).sample_with_seed(src, 8);

    assert_eq!(first, second);
    assert_ne!(first, different_seed);
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

#[test]
fn local_placer_accepts_q_only_sequential_primitives_with_macro_candidates() {
    let mut graph = Graph {
        nodes: vec![
            GraphNode {
                id: 0,
                kind: GraphNodeKind::Input("s".to_owned()),
                outputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                id: 1,
                kind: GraphNodeKind::Input("r".to_owned()),
                outputs: vec![2],
                ..Default::default()
            },
            GraphNode {
                id: 2,
                kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
                    SequentialType::RsLatch,
                    vec!["s".to_owned(), "r".to_owned()],
                    vec!["q".to_owned()],
                )),
                inputs: vec![0, 1],
                outputs: vec![3],
                ..Default::default()
            },
            GraphNode {
                id: 3,
                kind: GraphNodeKind::Output("q".to_owned()),
                inputs: vec![2],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    graph.build_inputs();
    graph.build_outputs();

    let placer = LocalPlacer::new(LogicGraph { graph }, config(1));

    assert!(placer.is_ok());
}

#[test]
fn local_placer_rejects_multi_output_sequential_primitives_until_edges_are_port_aware() {
    let mut graph = Graph {
        nodes: vec![GraphNode {
            id: 0,
            kind: GraphNodeKind::Sequential(SequentialPrimitive::rs_latch()),
            outputs: vec![1],
            ..Default::default()
        }],
        ..Default::default()
    };
    graph.build_inputs();
    graph.build_outputs();

    let err = match LocalPlacer::new(LogicGraph { graph }, config(1)) {
        Ok(_) => panic!("expected multi-output sequential primitive to be rejected"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("only one exposed sequential output"));
}

#[test]
fn local_placer_rejects_sequential_primitives_without_macro_candidates() {
    let mut graph = Graph {
        nodes: vec![GraphNode {
            id: 0,
            kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
                SequentialType::DLatch,
                Vec::new(),
                vec!["q".to_owned()],
            )),
            outputs: vec![1],
            ..Default::default()
        }],
        ..Default::default()
    };
    graph.build_inputs();
    graph.build_outputs();

    let err = match LocalPlacer::new(LogicGraph { graph }, config(1)) {
        Ok(_) => panic!("expected unsupported sequential primitive to be rejected"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("sequential primitive placement is not implemented"));
}

#[test]
fn sequential_macro_generation_registers_output_port_positions() {
    let node = GraphNode {
        id: 7,
        kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
            SequentialType::RsLatch,
            Vec::new(),
            vec!["q".to_owned()],
        )),
        ..Default::default()
    };
    let GraphNodeKind::Sequential(sequential) = &node.kind else {
        panic!("expected sequential node");
    };

    let generated = generate_sequential_macro_routes(
        &config(1),
        &node,
        sequential,
        &World3D::new(DimSize(8, 8, 4)),
        &PlacementState::default(),
    );

    assert!(!generated.is_empty());
    assert!(generated[0].1.node_position(node.id).is_some());
    assert!(generated[0].1.port_position(node.id, "q").is_some());
}

#[test]
fn sequential_macro_routes_input_ports_from_existing_sources() {
    let mut world = World3D::new(DimSize(8, 8, 4));
    let source_s = Position(0, 0, 1);
    let source_r = Position(0, 4, 1);
    place_node(
        &mut world,
        PlacedNode::new(source_s, switch(Direction::East)),
    );
    place_node(
        &mut world,
        PlacedNode::new(source_r, switch(Direction::East)),
    );
    place_node(&mut world, PlacedNode::new_cobble(Position(1, 0, 0)));
    place_node(&mut world, PlacedNode::new_cobble(Position(1, 4, 0)));

    let sequential = SequentialPrimitive::new(
        SequentialType::RsLatch,
        vec!["s".to_owned(), "r".to_owned()],
        vec!["q".to_owned()],
    );
    let node = GraphNode {
        id: 2,
        kind: GraphNodeKind::Sequential(sequential.clone()),
        inputs: vec![0, 1],
        ..Default::default()
    };
    let candidate = SequentialMacro::candidates(&sequential).remove(0);
    let placed = place_sequential_macro(&world, &candidate, Position(0, 0, 0)).unwrap();
    let state = [(0, source_s), (1, source_r)].into_iter().collect();

    let routed = route_sequential_inputs(&config(1), &node, &sequential, &state, &placed);

    assert!(!routed.is_empty());
}
