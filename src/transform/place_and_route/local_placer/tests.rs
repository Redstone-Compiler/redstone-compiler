use super::*;
use crate::graph::logic::{predefined_logics, LogicGraph};
use crate::graph::{Graph, GraphNode, GraphNodeKind, GraphNodeRef};
use crate::logic::LogicType;
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
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::None,
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: true,
        materialize_outputs: false,
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
    let generated = generate_inputs(&config(1), &world, BlockKind::Switch { is_on: false }, None);

    assert_eq!(generated.len(), 10);
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

    let generated = generate_inputs(&config, &world, BlockKind::Switch { is_on: false }, None);

    assert_eq!(generated.len(), 18);
}

#[test]
fn generate_inputs_limits_candidates_per_world() {
    let world = World3D::new(DimSize(3, 3, 2));
    let mut config = config(1);
    config.input_candidate_limit = Some(4);

    let generated = generate_inputs(&config, &world, BlockKind::Switch { is_on: false }, None);

    assert_eq!(generated.len(), 4);
}

#[test]
fn generate_inputs_uses_explicit_position_region() {
    let world = World3D::new(DimSize(6, 6, 4));
    let positions = [Position(3, 3, 1), Position(4, 3, 1)];

    let generated = generate_inputs(
        &config(1),
        &world,
        BlockKind::Switch { is_on: false },
        Some(&positions),
    );

    assert_eq!(generated.len(), 2);
    assert_eq!(
        generated
            .iter()
            .map(|(_, position)| *position)
            .collect_vec(),
        positions
    );
}

#[test]
fn local_placer_limits_input_search_to_named_constraints() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a", "out")?;
    let placer = LocalPlacer::new(graph, config(1))?;
    let input_position = Position(3, 3, 1);
    let constraints =
        LocalPlacerInputConstraints::new().with_input_positions("a", [input_position]);

    let generated = placer.generate_with_input_constraints(DimSize(6, 6, 4), None, &constraints);

    assert_eq!(generated.len(), 1);
    assert!(generated[0][input_position].kind.is_switch());
    Ok(())
}

#[test]
fn local_placer_accepts_chained_or_after_buffer_insertion() -> eyre::Result<()> {
    let mut graph = LogicGraph::from_stmt("a|b", "x")?;
    graph.graph.merge(LogicGraph::from_stmt("x|c", "y")?.graph);
    let graph = graph.prepare_place()?;

    let _ = LocalPlacer::new(graph, config(4))?;

    Ok(())
}

#[test]
fn local_placer_accepts_unbuffered_full_adder_after_buffer_insertion() -> eyre::Result<()> {
    let graph = predefined_logics::full_adder_graph()?;

    let _ = LocalPlacer::new(graph, config(4))?;

    Ok(())
}

#[test]
fn local_placer_seeded_queue_reuses_preplaced_input_nodes() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a", "out")?;
    let placer = LocalPlacer::new(graph.clone(), config(1))?;
    let input_id = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == "a"))
        .unwrap()
        .id;
    let input_position = Position(2, 2, 1);
    let mut world = empty_world();
    world[input_position] = switch(Direction::East);
    let state = [(input_id, input_position)].into_iter().collect();

    let generated = placer.generate_queue_from(vec![(world, state)], None, None);

    assert_eq!(generated.len(), 1);
    assert_eq!(generated[0].1.node_position(input_id), Some(input_position));
    assert!(generated[0].0[input_position].kind.is_switch());
    Ok(())
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
fn compact_queue_keeps_positions_needed_by_future_logic() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a|b", "c")?.prepare_place()?;
    let placer = LocalPlacer::new(graph.clone(), config(1))?;
    let input_a = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Input(name) if name == "a"))
        .unwrap()
        .id;
    let or_id = graph
        .nodes
        .iter()
        .find(|node| {
            matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
        })
        .unwrap()
        .id;
    let or_step = placer
        .visit_orders
        .iter()
        .position(|id| *id == or_id)
        .unwrap();
    let world = empty_world();
    let queue = vec![
        (
            world.clone(),
            [(input_a, Position(1, 1, 1))].into_iter().collect(),
        ),
        (world, [(input_a, Position(2, 1, 1))].into_iter().collect()),
    ];

    let compacted = placer.compact_queue_after_step(or_step - 1, queue);

    assert_eq!(compacted.len(), 2);
    Ok(())
}

#[test]
fn compact_queue_keeps_positions_needed_by_external_outputs() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a|b", "c")?.prepare_place()?;
    let placer = LocalPlacer::new(graph.clone(), config(1))?;
    let or_id = graph
        .nodes
        .iter()
        .find(|node| {
            matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
        })
        .unwrap()
        .id;
    let or_step = placer
        .visit_orders
        .iter()
        .position(|id| *id == or_id)
        .unwrap();
    let world = empty_world();
    let queue = vec![
        (
            world.clone(),
            [(or_id, Position(1, 1, 1))].into_iter().collect(),
        ),
        (world, [(or_id, Position(2, 1, 1))].into_iter().collect()),
    ];

    let compacted = placer.compact_queue_after_step(or_step, queue);

    assert_eq!(compacted.len(), 2);
    Ok(())
}

#[test]
fn output_step_routes_visible_redstone_endpoint_from_or_source() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("a|b", "c")?.prepare_place()?;
    let mut config = config(1);
    config.materialize_outputs = true;
    let placer = LocalPlacer::new(graph.clone(), config)?;
    let or_id = graph
        .nodes
        .iter()
        .find(|node| {
            matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
        })
        .unwrap()
        .id;
    let output_id = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == "c"))
        .unwrap()
        .id;
    let output_step = placer
        .visit_orders
        .iter()
        .position(|id| *id == output_id)
        .unwrap();
    let source = Position(2, 2, 1);
    let mut world = empty_world();
    place_node(&mut world, PlacedNode::new_cobble(Position(2, 2, 0)));
    place_node(&mut world, PlacedNode::new_redstone(source));
    let state = [(or_id, source)].into_iter().collect();

    let result = placer.do_step(output_step, vec![(world, state)], None);

    assert!(!result.queue.is_empty());
    assert!(result.queue.iter().any(|(world, state)| {
        state
            .node_position(output_id)
            .is_some_and(|position| world[position].kind.is_redstone())
    }));
    Ok(())
}

#[test]
fn output_step_routes_visible_redstone_endpoint_from_torch_source() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("~a", "out")?.prepare_place()?;
    let mut config = config(1);
    config.materialize_outputs = true;
    let placer = LocalPlacer::new(graph.clone(), config)?;
    let not_id = graph
        .nodes
        .iter()
        .find(|node| {
            matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Not)
        })
        .unwrap()
        .id;
    let output_id = graph
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, GraphNodeKind::Output(name) if name == "out"))
        .unwrap()
        .id;
    let output_step = placer
        .visit_orders
        .iter()
        .position(|id| *id == output_id)
        .unwrap();
    let source = Position(2, 2, 1);
    let mut world = empty_world();
    place_node(&mut world, PlacedNode::new_cobble(Position(2, 2, 0)));
    place_node(
        &mut world,
        PlacedNode::new(source, torch(Direction::Bottom)),
    );
    let state = [(not_id, source)].into_iter().collect();

    let result = placer.do_step(output_step, vec![(world, state)], None);

    assert!(!result.queue.is_empty());
    assert!(result.queue.iter().any(|(world, state)| {
        state
            .node_position(output_id)
            .is_some_and(|position| world[position].kind.is_redstone())
    }));
    Ok(())
}

#[test]
fn generate_with_outputs_reports_materialized_output_positions() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("~a", "out")?.prepare_place()?;
    let mut config = config(1);
    config.materialize_outputs = true;
    let placer = LocalPlacer::new(graph, config)?;

    let placed = placer.generate_with_outputs(DimSize(5, 5, 3), None);

    assert!(!placed.is_empty());
    assert!(placed.iter().any(|placed| {
        placed.outputs.iter().any(|endpoint| {
            endpoint.name == "out" && placed.world[endpoint.position()].kind.is_redstone()
        })
    }));
    Ok(())
}

#[test]
fn generate_with_outputs_reports_input_positions_after_queue_compaction() -> eyre::Result<()> {
    let graph = LogicGraph::from_stmt("~a", "out")?.prepare_place()?;
    let placer = LocalPlacer::new(graph, config(1))?;

    let placed = placer.generate_with_outputs(DimSize(5, 5, 3), None);

    assert!(!placed.is_empty());
    assert!(placed.iter().any(|placed| {
        placed.inputs.iter().any(|endpoint| {
            endpoint.name == "a" && placed.world[endpoint.position()].kind.is_switch()
        })
    }));
    Ok(())
}

#[test]
fn future_join_cost_weights_pairs_with_remaining_fanout() {
    let mut graph = Graph::from_nodes(vec![
        GraphNode {
            kind: GraphNodeKind::Input("a".to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Input("b".to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Input("c".to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Input("d".to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Input("e".to_owned()),
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Logic(crate::logic::Logic {
                logic_type: LogicType::Or,
            }),
            inputs: vec![0, 1],
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Logic(crate::logic::Logic {
                logic_type: LogicType::Or,
            }),
            inputs: vec![0, 2],
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Logic(crate::logic::Logic {
                logic_type: LogicType::Or,
            }),
            inputs: vec![3, 4],
            ..Default::default()
        },
    ]);
    graph.build_outputs();
    let graph = LogicGraph { graph };
    let visit_orders = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let step_after_inputs = 4;

    let pairs = build_cost_join_pairs_by_step(&graph, &visit_orders);
    let pairs = &pairs[step_after_inputs];
    let fanout_pair = pairs
        .iter()
        .find(|pair| pair.a == 0 && pair.b == 1)
        .unwrap();
    let simple_pair = pairs
        .iter()
        .find(|pair| pair.a == 3 && pair.b == 4)
        .unwrap();

    assert_eq!(fanout_pair.weight, 2);
    assert_eq!(simple_pair.weight, 1);
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
fn ranked_sampling_keeps_lower_cost_candidates() -> eyre::Result<()> {
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
    let ranked_config = LocalPlacerConfig {
        placement_sampling_policy: PlacementSamplingPolicy::Ranked {
            count: 1,
            random_count: 0,
            start_step: 0,
        },
        ..placer.config
    };

    let ranked = LocalPlacer::new(graph, ranked_config)?;
    let sampled = ranked.sample(step, vec![(world.clone(), far), (world, near)]);

    assert_eq!(sampled.len(), 1);
    assert_eq!(sampled[0].1[&input_b], Position(2, 1, 1));

    Ok(())
}

#[test]
fn ranked_sampling_preserves_diverse_geometry_candidates() -> eyre::Result<()> {
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
    let compact_a = [(input_a, Position(1, 1, 1)), (input_b, Position(2, 1, 1))]
        .into_iter()
        .collect();
    let compact_b = [(input_a, Position(1, 1, 1)), (input_b, Position(1, 2, 1))]
        .into_iter()
        .collect();
    let spread = [(input_a, Position(1, 1, 1)), (input_b, Position(6, 6, 1))]
        .into_iter()
        .collect();
    let ranked_config = LocalPlacerConfig {
        placement_sampling_policy: LocalPlacerConfig::ranked_sampling(2, 0, 0),
        ..placer.config
    };

    let ranked = LocalPlacer::new(graph, ranked_config)?;
    let sampled = ranked.sample(
        step,
        vec![
            (world.clone(), compact_a),
            (world.clone(), compact_b),
            (world, spread),
        ],
    );
    let sampled_b_positions = sampled
        .iter()
        .map(|(_, state)| state[&input_b])
        .collect_vec();

    assert_eq!(sampled.len(), 2);
    assert!(sampled_b_positions.contains(&Position(6, 6, 1)));

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
    let torch_pos = Position(4, 1, 1);
    let cobble_pos = Position(3, 1, 1);
    let mut config = config(1);
    config.not_route_strategy = NotRouteStrategy::RedstoneOnly;
    place_node(
        &mut world,
        PlacedNode::new(source, torch(Direction::Bottom)),
    );
    place_node(&mut world, PlacedNode::new_cobble(cobble_pos));

    let routes = generate_routes_to_cobble(&config, &world, source, torch_pos, cobble_pos);

    assert!(routes.iter().any(|(world, pos)| {
        world[*pos].kind.is_redstone()
            && PlacedNode::new(*pos, world[*pos]).has_connection_with(world, cobble_pos)
    }));
}

#[test]
fn generate_routes_to_cobble_can_extend_redstone_source() {
    let mut world = empty_world();
    let source = Position(1, 1, 1);
    let torch_pos = Position(4, 1, 1);
    let cobble_pos = Position(3, 1, 1);
    let mut config = config(1);
    config.not_route_strategy = NotRouteStrategy::DirectAndRedstone;
    place_node(&mut world, PlacedNode::new_cobble(Position(1, 1, 0)));
    place_node(&mut world, PlacedNode::new_redstone(source));
    place_node(&mut world, PlacedNode::new_cobble(cobble_pos));

    let routes = generate_routes_to_cobble(&config, &world, source, torch_pos, cobble_pos);

    assert!(routes.iter().any(|(world, pos)| {
        world[*pos].kind.is_redstone()
            && PlacedNode::new(*pos, world[*pos]).has_connection_with(world, cobble_pos)
    }));
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
fn redstone_below_switch_powered_cobble_is_short() {
    let mut world = empty_world();
    let switch_pos = Position(0, 2, 2);
    let cobble_pos = Position(1, 2, 2);
    let redstone_pos = Position(1, 2, 1);

    place_node(
        &mut world,
        PlacedNode::new(switch_pos, switch(Direction::East)),
    );
    place_node(&mut world, PlacedNode::new_cobble(cobble_pos));

    let redstone_node = PlacedNode::new_redstone(redstone_pos);

    assert!(redstone_node.has_short(&world, &Default::default()));
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
    let mut graph = Graph::from_nodes(vec![
        GraphNode {
            kind: GraphNodeKind::Input("s".to_owned()),
            outputs: vec![2],
            ..Default::default()
        },
        GraphNode {
            kind: GraphNodeKind::Input("r".to_owned()),
            outputs: vec![2],
            ..Default::default()
        },
        GraphNode {
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
            kind: GraphNodeKind::Output("q".to_owned()),
            inputs: vec![2],
            ..Default::default()
        },
    ]);
    graph.build_inputs();
    graph.build_outputs();

    let placer = LocalPlacer::new(LogicGraph { graph }, config(1));

    assert!(placer.is_ok());
}

#[test]
fn local_placer_rejects_multi_output_sequential_primitives_until_edges_are_port_aware() {
    let mut graph = Graph::from_nodes(vec![GraphNode {
        kind: GraphNodeKind::Sequential(SequentialPrimitive::rs_latch()),
        outputs: vec![1],
        ..Default::default()
    }]);
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
    let mut graph = Graph::from_nodes(vec![GraphNode {
        kind: GraphNodeKind::Sequential(SequentialPrimitive::new(
            SequentialType::DFlipFlop,
            Vec::new(),
            vec!["q".to_owned()],
        )),
        outputs: vec![1],
        ..Default::default()
    }]);
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
    let node_id = 7;
    let node = GraphNode {
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
    let node_ref = GraphNodeRef::new(node_id, &node);

    let generated = generate_sequential_macro_routes(
        &config(1),
        node_ref,
        sequential,
        &World3D::new(DimSize(8, 8, 4)),
        &PlacementState::default(),
    );

    assert!(!generated.is_empty());
    assert!(generated[0].1.node_position(node_id).is_some());
    assert!(generated[0].1.port_position(node_id, "q").is_some());
}

#[test]
fn sequential_macro_routes_input_ports_from_existing_sources() {
    let mut world = World3D::new(DimSize(12, 9, 4));
    let source_s = Position(11, 3, 1);
    let source_r = Position(0, 3, 1);
    place_node(
        &mut world,
        PlacedNode::new(source_s, switch(Direction::West)),
    );
    place_node(
        &mut world,
        PlacedNode::new(source_r, switch(Direction::East)),
    );

    let sequential = SequentialPrimitive::new(
        SequentialType::RsLatch,
        vec!["s".to_owned(), "r".to_owned()],
        vec!["q".to_owned()],
    );
    let node = GraphNode {
        kind: GraphNodeKind::Sequential(sequential.clone()),
        inputs: vec![0, 1],
        ..Default::default()
    };
    let node_ref = GraphNodeRef::new(2, &node);
    let candidate = SequentialMacro::candidates(&sequential).remove(0);
    let placed = place_sequential_macro(&world, &candidate, Position(2, 1, 0)).unwrap();
    let state = [(0, source_s), (1, source_r)].into_iter().collect();

    let routed = route_sequential_inputs(&config(4), node_ref, &sequential, &state, &placed);

    assert!(!routed.is_empty());
}
