use crate::graph::analysis::equivalent_expression_groups;
use crate::graph::graphviz::ToGraphvizGraph;
use crate::graph::logic::predefined_logics;
use crate::graph::{GraphNode, GraphNodeKind};
use crate::nbt::{NBTRoot, ToNBT};
use crate::sequential::{SequentialPrimitive, SequentialType};
use crate::transform::place_and_route::estimate::world_compact_cost;
use crate::transform::place_and_route::local_placer::{
    generate_d_latch_gate_routes, generate_rs_latch_not_pairs, route_rs_latch_branches,
    select_rs_latch_not_pairs, InputPlacementStrategy, LocalPlacer, LocalPlacerConfig,
    LocalPlacerDebug, NotRouteStrategy, PlacementSamplingPolicy, SamplingPolicy,
    TorchPlacementStrategy, D_LATCH_INPUT_GATING_NODES,
};
use crate::transform::place_and_route::utils::{
    contains_truth_table_with_world3ds, equivalent_logic_with_world3d,
    equivalent_logic_with_world3ds, world3d_to_logic,
};
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::Simulator;
use crate::world::{World, World3D};

#[test]
fn test_generate_component_and_shortest() -> eyre::Result<()> {
    let logic_graph = predefined_logics::and_graph()?;
    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::None,
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: false,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 1,
        not_route_step_sampling_policy: SamplingPolicy::None,
        max_route_step: 1,
        route_step_sampling_policy: SamplingPolicy::Random(100),
    };
    let placer = LocalPlacer::new(logic_graph.clone(), config)?;
    let worlds = placer.generate(DimSize(10, 10, 5), None);
    assert!(!worlds.is_empty());
    assert!(equivalent_logic_with_world3ds(&logic_graph, &worlds)?);
    Ok(())
}

fn save_worlds_to_nbt(worlds: Vec<World3D>, path: &str) -> eyre::Result<()> {
    let concated_world = World3D::concat_tiled(worlds);
    let nbt: NBTRoot = concated_world.to_nbt();
    nbt.save(path);
    Ok(())
}

fn torch_is_on(world: &World3D, position: Position) -> bool {
    let BlockKind::Torch { is_on } = world[position].kind else {
        panic!("expected torch at {position:?}");
    };

    is_on
}

fn assert_rs_latch_behavior(
    world: &World3D,
    s: Position,
    r: Position,
    q: Position,
    nq: Position,
) -> eyre::Result<()> {
    let mut reset_world = world.clone();
    reset_world[r].kind = BlockKind::Switch { is_on: true };
    reset_world.initialize_redstone_states();
    let world = World::from(&reset_world);
    let mut sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
        .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

    eyre::ensure!(!torch_is_on(sim.world(), q), "reset should turn q off");
    eyre::ensure!(torch_is_on(sim.world(), nq), "reset should turn nq on");

    sim.change_state_with_limits(vec![(r, false)], 64, 20_000)?;
    eyre::ensure!(!torch_is_on(sim.world(), q), "hold reset should keep q off");
    eyre::ensure!(torch_is_on(sim.world(), nq), "hold reset should keep nq on");

    sim.change_state_with_limits(vec![(s, true)], 64, 20_000)?;
    sim.change_state_with_limits(vec![(s, false)], 64, 20_000)?;
    eyre::ensure!(torch_is_on(sim.world(), q), "set should turn q on");
    eyre::ensure!(!torch_is_on(sim.world(), nq), "set should turn nq off");

    sim.change_state_with_limits(vec![(r, true)], 64, 20_000)?;
    sim.change_state_with_limits(vec![(r, false)], 64, 20_000)?;
    eyre::ensure!(
        !torch_is_on(sim.world(), q),
        "reset again should turn q off"
    );
    eyre::ensure!(
        torch_is_on(sim.world(), nq),
        "reset again should turn nq on"
    );

    Ok(())
}

fn assert_d_latch_behavior(
    world: &World3D,
    d: Position,
    en: Position,
    q: Position,
    nq: Position,
) -> eyre::Result<()> {
    let mut reset_world = pad_world_top(world, 2);
    reset_world[en].kind = BlockKind::Switch { is_on: true };
    reset_world[d].kind = BlockKind::Switch { is_on: false };
    reset_world.initialize_redstone_states();
    let world = World::from(&reset_world);
    let mut sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
        .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

    eyre::ensure!(!torch_is_on(sim.world(), q), "reset should turn q off");
    eyre::ensure!(torch_is_on(sim.world(), nq), "reset should turn nq on");

    sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
    sim.change_state_with_limits(vec![(d, true)], 64, 20_000)?;
    eyre::ensure!(
        !torch_is_on(sim.world(), q),
        "disabled latch should hold q off"
    );
    eyre::ensure!(
        torch_is_on(sim.world(), nq),
        "disabled latch should hold nq on"
    );

    sim.change_state_with_limits(vec![(en, true)], 64, 20_000)?;
    eyre::ensure!(
        torch_is_on(sim.world(), q),
        "enabled data high should set q"
    );
    eyre::ensure!(
        !torch_is_on(sim.world(), nq),
        "enabled data high should clear nq"
    );

    sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
    sim.change_state_with_limits(vec![(d, false)], 64, 20_000)?;
    eyre::ensure!(
        torch_is_on(sim.world(), q),
        "disabled latch should hold q on"
    );
    eyre::ensure!(
        !torch_is_on(sim.world(), nq),
        "disabled latch should hold nq off"
    );

    sim.change_state_with_limits(vec![(en, true)], 64, 20_000)?;
    eyre::ensure!(
        !torch_is_on(sim.world(), q),
        "enabled data low should reset q"
    );
    eyre::ensure!(
        torch_is_on(sim.world(), nq),
        "enabled data low should set nq"
    );

    Ok(())
}

fn trace_d_latch_behavior(
    world: &World3D,
    d: Position,
    en: Position,
    signals: &[(&str, Position)],
) -> eyre::Result<()> {
    let mut reset_world = pad_world_top(world, 2);
    reset_world[en].kind = BlockKind::Switch { is_on: true };
    reset_world[d].kind = BlockKind::Switch { is_on: false };
    reset_world.initialize_redstone_states();
    let world = World::from(&reset_world);
    let mut sim = Simulator::from_with_limits_and_trace(&world, 64, 20_000, 0)
        .map_err(|error| eyre::eyre!(error.message().to_owned()))?;

    eprintln!("  trace reset: {}", signal_summary(sim.world(), signals));
    eprintln!(
        "  reset support neighbors: {}",
        neighborhood_summary(sim.world(), signals, "reset_support")
    );
    eprintln!(
        "  q support neighbors: {}",
        neighborhood_summary(sim.world(), signals, "q_support")
    );
    eprintln!(
        "  nq support neighbors: {}",
        neighborhood_summary(sim.world(), signals, "nq_support")
    );
    sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
    eprintln!("  trace disabled: {}", signal_summary(sim.world(), signals));
    sim.change_state_with_limits(vec![(d, true)], 64, 20_000)?;
    eprintln!(
        "  trace d high hold: {}",
        signal_summary(sim.world(), signals)
    );
    sim.change_state_with_limits(vec![(en, true)], 64, 20_000)?;
    eprintln!(
        "  trace d high enabled: {}",
        signal_summary(sim.world(), signals)
    );
    sim.change_state_with_limits(vec![(en, false)], 64, 20_000)?;
    eprintln!(
        "  trace hold q on: {}",
        signal_summary(sim.world(), signals)
    );
    sim.change_state_with_limits(vec![(d, false)], 64, 20_000)?;
    eprintln!(
        "  trace hold q on d low: {}",
        signal_summary(sim.world(), signals)
    );

    Ok(())
}

fn signal_summary(world: &World3D, signals: &[(&str, Position)]) -> String {
    signals
        .iter()
        .map(|(name, position)| {
            let value = match world[*position].kind {
                BlockKind::Switch { is_on } => format!("switch={is_on}"),
                BlockKind::Torch { is_on } => format!("torch={is_on}"),
                BlockKind::Redstone { strength, .. } => format!("redstone={strength}"),
                ref kind => format!("{kind:?}"),
            };
            format!(
                "{name}@{position:?}/{:?}:{value}",
                world[*position].direction
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn neighborhood_summary(world: &World3D, signals: &[(&str, Position)], name: &str) -> String {
    let Some((_, center)) = signals.iter().find(|(signal_name, _)| *signal_name == name) else {
        return String::new();
    };
    center
        .forwards()
        .into_iter()
        .filter(|&position| world.size.bound_on(position))
        .map(|position| {
            format!(
                "{position:?}/{:?}:{:?}",
                world[position].direction, world[position].kind
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn pad_world_top(world: &World3D, extra_z: usize) -> World3D {
    let mut padded = World3D::new(DimSize(world.size.0, world.size.1, world.size.2 + extra_z));
    for (position, mut block) in world.iter_block() {
        if block.kind.is_cobble() {
            block.kind = BlockKind::Cobble {
                on_count: 0,
                on_base_count: 0,
            };
        }
        padded[position] = block;
    }
    padded
}

#[test]
fn test_generate_component_rs_latch() -> eyre::Result<()> {
    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::None,
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectAndRedstone,
        max_not_route_step: 4,
        not_route_step_sampling_policy: SamplingPolicy::Take(4),
        max_route_step: 8,
        route_step_sampling_policy: SamplingPolicy::Take(4),
    };
    let s = Position(6, 2, 1);
    let r = Position(0, 2, 1);
    let mut world = World3D::new(DimSize(12, 9, 5));
    world[s] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::West,
    };
    world[r] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::East,
    };
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
    let state = [(0, s), (1, r)].into_iter().collect();

    let pairs = select_rs_latch_not_pairs(
        &config,
        &node,
        &sequential,
        &state,
        generate_rs_latch_not_pairs(&world),
    );
    let generated = pairs
        .into_iter()
        .flat_map(|placed| route_rs_latch_branches(&config, &node, &sequential, &state, placed))
        .collect::<Vec<_>>();
    assert!(!generated.is_empty());
    let valid = generated.into_iter().find_map(|placed| {
        assert_rs_latch_behavior(&placed.world, s, r, placed.q_torch, placed.nq_torch)
            .ok()
            .map(|_| placed.world)
    });
    let world =
        valid.expect("expected at least one generated RS latch to pass set/reset/hold simulation");

    save_worlds_to_nbt(vec![world], "test/rs-latch.nbt")?;

    Ok(())
}

#[test]
fn test_generate_component_d_latch() -> eyre::Result<()> {
    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::Random(256),
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectAndRedstone,
        max_not_route_step: 4,
        not_route_step_sampling_policy: SamplingPolicy::Random(256),
        max_route_step: 4,
        route_step_sampling_policy: SamplingPolicy::Random(256),
    };
    let d = Position(0, 2, 1);
    let en = Position(0, 6, 1);
    let mut world = World3D::new(DimSize(14, 10, 6));
    world[d] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::East,
    };
    world[en] = Block {
        kind: BlockKind::Switch { is_on: false },
        direction: Direction::East,
    };
    let sequential = SequentialPrimitive::new(
        SequentialType::DLatch,
        vec!["d".to_owned(), "en".to_owned()],
        vec!["q".to_owned()],
    );
    let node = GraphNode {
        id: 2,
        kind: GraphNodeKind::Sequential(sequential.clone()),
        inputs: vec![0, 1],
        ..Default::default()
    };
    let state = [(0, d), (1, en)].into_iter().collect();

    let generated = generate_d_latch_gate_routes(&config, &node, &sequential, &world, &state);
    assert!(!generated.is_empty());
    let mut checked = 0usize;
    let valid = generated.iter().find_map(|(world, state)| {
        let q = state.port_position(2, "q")?;
        let nq = state.port_position(2, "nq")?;
        checked += 1;
        match std::panic::catch_unwind(|| assert_d_latch_behavior(&world, d, en, q, nq)) {
            Ok(Ok(())) => Some(pad_world_top(&world, 2)),
            Ok(Err(error)) => {
                if checked <= 3 {
                    eprintln!("candidate {checked} failed: {error}; q={q:?} nq={nq:?}");
                    let q_support = q.walk(world[q].direction).unwrap();
                    let nq_support = nq.walk(world[nq].direction).unwrap();
                    let nodes = D_LATCH_INPUT_GATING_NODES;
                    let _ = trace_d_latch_behavior(
                        world,
                        d,
                        en,
                        &[
                            ("d", d),
                            ("en", en),
                            ("not_d", state[&nodes.not_d]),
                            ("not_en", state[&nodes.not_en]),
                            ("set_or", state[&nodes.set_or]),
                            ("reset_or", state[&nodes.reset_or]),
                            ("set", state[&nodes.set]),
                            ("reset", state[&nodes.reset]),
                            ("q_support", q_support),
                            ("nq_support", nq_support),
                            ("q", q),
                            ("nq", nq),
                        ],
                    );
                }
                None
            }
            Err(_) => None,
        }
    });
    let world = valid.unwrap_or_else(|| {
        panic!(
            "expected at least one valid D latch candidate, checked {checked} of {} generated",
            generated.len()
        )
    });

    save_worlds_to_nbt(vec![world], "test/d-latch.nbt")?;

    Ok(())
}

#[test]
fn test_generate_component_xor_simple() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: false,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::Random(100),
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 5,
        not_route_step_sampling_policy: SamplingPolicy::Random(100),
        max_route_step: 5,
        route_step_sampling_policy: SamplingPolicy::Random(100),
    };
    let logic_graph = predefined_logics::xor_graph()?;
    let placer = LocalPlacer::new(logic_graph, config)?;
    let worlds = placer.generate(DimSize(10, 10, 5), None);

    let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
    let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
    println!("{}", sample_logic.to_graphviz());

    save_worlds_to_nbt(sampled_worlds, "test/xor-gate-simple.nbt")?;

    Ok(())
}

#[test]
fn test_generate_component_xor_complex() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: false,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::Random(1000),
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: true,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 3,
        not_route_step_sampling_policy: SamplingPolicy::Random(1000),
        max_route_step: 3,
        route_step_sampling_policy: SamplingPolicy::Random(1000),
    };

    let xor_graph = predefined_logics::buffered_xor_graph()?;
    let placer = LocalPlacer::new(xor_graph, config)?;
    let worlds = placer.generate(DimSize(10, 10, 5), None);

    let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
    let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
    println!("{}", sample_logic.to_graphviz());

    save_worlds_to_nbt(sampled_worlds, "test/xor-gate-complex.nbt")?;

    Ok(())
}

#[test]
fn test_generate_component_xor_shortest() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .unwrap();

    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::Random(10000),
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: true,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 1,
        not_route_step_sampling_policy: SamplingPolicy::Random(1000),
        max_route_step: 1,
        route_step_sampling_policy: SamplingPolicy::Random(1000),
    };

    let xor_graph = predefined_logics::buffered_xor_graph()?;
    let placer = LocalPlacer::new(xor_graph.clone(), config)?;
    let worlds = placer.generate(DimSize(10, 10, 5), None);

    let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
    let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
    println!("{}", sample_logic.to_graphviz());

    for sample in &sampled_worlds {
        // For debug
        // if !equivalent_logic_with_world3d(&xor_graph, sample)? {
        //     let sample_logic = world3d_to_logic(&sample)?.prepare_place()?;
        //     println!("{}", sample_logic.to_graphviz());
        // }
        assert!(equivalent_logic_with_world3d(&xor_graph, sample)?);
    }

    save_worlds_to_nbt(sampled_worlds, "test/xor-gate-shortest.nbt")?;

    Ok(())
}

#[test]
fn test_generate_component_half_adder() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::Random(10000),
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 3,
        not_route_step_sampling_policy: SamplingPolicy::Random(100),
        max_route_step: 3,
        route_step_sampling_policy: SamplingPolicy::Random(100),
    };

    let fa_graph = predefined_logics::buffered_half_adder_graph()?;
    println!("{}", fa_graph.to_graphviz());
    let placer = LocalPlacer::new(fa_graph, config)?;
    let worlds = placer.generate(DimSize(10, 10, 5), None);

    let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);
    let sample_logic = world3d_to_logic(&sampled_worlds[0])?.prepare_place()?;
    println!("{}", sample_logic.to_graphviz());

    save_worlds_to_nbt(sampled_worlds, "test/half-adder.nbt")?;

    Ok(())
}

#[test]
fn test_generate_component_full_adder() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: Some(25),
        step_sampling_policy: SamplingPolicy::None,
        placement_sampling_policy: LocalPlacerConfig::ranked_sampling(2000, 500, 0),
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 0,
        not_route_step_sampling_policy: SamplingPolicy::Random(10),
        max_route_step: 4,
        route_step_sampling_policy: SamplingPolicy::Random(10),
    };

    let fa_graph = predefined_logics::buffered_full_adder_graph()?;
    let expected_graph = predefined_logics::full_adder_graph()?;
    println!("{}", fa_graph.to_graphviz());
    let placer = LocalPlacer::new(fa_graph.clone(), config)?;
    let worlds = placer.generate(DimSize(10, 10, 5), None);

    let generated_count = worlds.len();
    let mut valid_worlds = Vec::new();
    for world in worlds {
        if contains_truth_table_with_world3ds(&expected_graph, std::slice::from_ref(&world))? {
            valid_worlds.push(world);
        }
    }
    println!(
        "full adder candidates: generated={generated_count}, valid={}",
        valid_worlds.len()
    );
    eyre::ensure!(
        !valid_worlds.is_empty(),
        "expected at least one generated full adder candidate to match the truth table"
    );

    let world = valid_worlds
        .into_iter()
        .min_by_key(world_compact_cost)
        .unwrap();
    let sample_logic = world3d_to_logic(&world)?.prepare_place()?;
    println!("{}", sample_logic.to_graphviz());

    save_worlds_to_nbt(vec![world], "test/full-adder.nbt")?;

    Ok(())
}

#[test]
#[ignore = "debug-only: compare full adder local placer cost sampling"]
fn debug_full_adder_with_cost_sampling() -> eyre::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::Random(10000),
        placement_sampling_policy: PlacementSamplingPolicy::Cost {
            count: 9000,
            random_count: 1000,
            start_step: 28,
        },
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 8,
        not_route_step_sampling_policy: SamplingPolicy::Random(100),
        max_route_step: 8,
        route_step_sampling_policy: SamplingPolicy::Random(100),
    };

    let fa_graph = predefined_logics::buffered_full_adder_graph()?;
    for group in equivalent_expression_groups(&fa_graph).into_iter().take(12) {
        println!(
            "duplicate expression: {} nodes={:?}",
            group.expression, group.node_ids
        );
    }
    let placer = LocalPlacer::new(fa_graph, config)?;
    let mut debug = LocalPlacerDebug::default();
    let worlds = placer.generate_with_debug(DimSize(10, 10, 5), None, &mut debug);

    debug.print_summary();
    if let Some(step) = debug.first_empty_step() {
        println!(
            "first empty step: step={} node={} kind={} inputs={:?} input_positions={:?}",
            step.step + 1,
            step.node_id,
            step.node_kind,
            step.input_node_ids,
            step.input_positions,
        );
    }
    println!("worlds generated: {}", worlds.len());
    if let Some(world) = worlds.into_iter().min_by_key(world_compact_cost) {
        save_worlds_to_nbt(vec![world], "test/full-adder.nbt")?;
    }

    Ok(())
}
