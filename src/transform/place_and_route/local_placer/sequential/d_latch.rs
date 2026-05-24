use itertools::Itertools;

use super::prefix::RsLatchPrefixPlan;
use super::rs_latch::generate_rs_latch_gate_routes;
use super::*;
use crate::world::simulator::Simulator;
use crate::world::World;

const D_LATCH_SAMPLE_SCOPE: u64 = 7;
const D_LATCH_OUTPUT_SAMPLE_STEP: usize = 6;
const D_LATCH_PREFIX_SAMPLE_COUNT: usize = 64;
const RS_CORE_SEARCH_SAMPLE_COUNT: usize = 32;
const RS_CORE_SEARCH_MAX_ROUTE_STEP: usize = 8;
const D_LATCH_SIM_EXTRA_Z: usize = 2;
const D_LATCH_SIM_MAX_CYCLES: usize = 64;
const D_LATCH_SIM_MAX_EVENTS: usize = 20_000;
const D_LATCH_SIM_TRACE_LIMIT: usize = 0;

pub(in super::super) fn generate_d_latch_gate_routes(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    sequential: &SequentialPrimitive,
    world: &World3D,
    state: &PlacementState,
) -> PlacerQueue {
    let Some(prefix_plan) = RsLatchPrefixPlan::build(node, sequential, state) else {
        return Vec::new();
    };
    let Some((data_input, enable_input)) = prefix_plan.d_latch_input_node_ids() else {
        return Vec::new();
    };

    let rs_sequential = SequentialPrimitive::new(
        SequentialType::RsLatch,
        vec!["s".to_owned(), "r".to_owned()],
        sequential.output_ports.clone(),
    );
    let rs_node = GraphNode {
        id: node.id,
        kind: GraphNodeKind::Sequential(rs_sequential.clone()),
        inputs: prefix_plan.rs_node_inputs(),
        ..Default::default()
    };

    let queue = prefix_plan.place(config, world, state);
    let queue = retain_valid_d_latch_prefix(config, queue, &prefix_plan);
    let routed = route_d_latch_core(
        config,
        node,
        state,
        &prefix_plan,
        data_input,
        enable_input,
        &rs_node,
        &rs_sequential,
        queue,
    );

    config.step_sampling_policy.sample_with_seed(
        routed,
        config.sampling_seed(D_LATCH_SAMPLE_SCOPE, D_LATCH_OUTPUT_SAMPLE_STEP),
    )
}

fn route_d_latch_core(
    config: &LocalPlacerConfig,
    node: &GraphNode,
    outer_state: &PlacementState,
    prefix_plan: &RsLatchPrefixPlan,
    data_input: GraphNodeId,
    enable_input: GraphNodeId,
    rs_node: &GraphNode,
    rs_sequential: &SequentialPrimitive,
    queue: PlacerQueue,
) -> PlacerQueue {
    let mut routed = Vec::new();
    let limit = take_sampling_limit(config.step_sampling_policy);
    let rs_config = LocalPlacerConfig {
        step_sampling_policy: SamplingPolicy::Random(RS_CORE_SEARCH_SAMPLE_COUNT),
        route_step_sampling_policy: SamplingPolicy::Random(RS_CORE_SEARCH_SAMPLE_COUNT),
        not_route_step_sampling_policy: SamplingPolicy::Random(RS_CORE_SEARCH_SAMPLE_COUNT),
        max_not_route_step: RS_CORE_SEARCH_MAX_ROUTE_STEP,
        max_route_step: RS_CORE_SEARCH_MAX_ROUTE_STEP,
        ..*config
    };

    for (world, prefix_state) in queue {
        let Some((set_position, reset_position)) =
            prefix_plan.prefix_output_positions(&prefix_state)
        else {
            continue;
        };
        let mut state = outer_state.clone();
        prefix_plan.publish_core_inputs(&mut state, set_position, reset_position);
        for (world, state) in
            generate_rs_latch_gate_routes(&rs_config, rs_node, rs_sequential, &world, &state)
        {
            let Some(q) = state.port_position(node.id, "q") else {
                continue;
            };
            let Some(nq) = state.port_position(node.id, "nq") else {
                continue;
            };
            if std::panic::catch_unwind(|| {
                d_latch_candidate_behaves(&world, data_input, enable_input, &state, q, nq)
            })
            .unwrap_or(false)
            {
                routed.push((world, state));
                break;
            }
        }
        if limit.is_some_and(|limit| routed.len() >= limit) {
            break;
        }
        if !routed.is_empty() {
            break;
        }
    }

    routed
}

fn retain_valid_d_latch_prefix(
    config: &LocalPlacerConfig,
    queue: PlacerQueue,
    prefix_plan: &RsLatchPrefixPlan,
) -> PlacerQueue {
    let Some(data) = prefix_plan.input_position("d") else {
        return Vec::new();
    };
    let Some(enable) = prefix_plan.input_position("en") else {
        return Vec::new();
    };
    let queue = queue
        .into_iter()
        .filter(|(world, state)| {
            d_latch_prefix_behaves(
                world,
                data,
                enable,
                prefix_plan.set_source(),
                prefix_plan.reset_source(),
                state,
            )
        })
        .collect_vec();
    SamplingPolicy::Random(D_LATCH_PREFIX_SAMPLE_COUNT).sample_with_seed(
        queue,
        config.sampling_seed(D_LATCH_SAMPLE_SCOPE, D_LATCH_OUTPUT_SAMPLE_STEP),
    )
}

fn d_latch_prefix_behaves(
    world: &World3D,
    data: Position,
    enable: Position,
    set_node_id: GraphNodeId,
    reset_node_id: GraphNodeId,
    state: &PlacementState,
) -> bool {
    let Some(set) = state.node_position(set_node_id) else {
        return false;
    };
    let Some(reset) = state.node_position(reset_node_id) else {
        return false;
    };

    [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .all(|(data_value, enable_value)| {
            d_latch_input_gate_values(world, data, enable, set, reset, data_value, enable_value)
                .is_some_and(|(set_value, reset_value)| {
                    set_value == (data_value && enable_value)
                        && reset_value == (!data_value && enable_value)
                })
        })
}

fn d_latch_input_gate_values(
    world: &World3D,
    data: Position,
    enable: Position,
    set: Position,
    reset: Position,
    data_value: bool,
    enable_value: bool,
) -> Option<(bool, bool)> {
    let mut world = pad_world_top_for_simulation(world, D_LATCH_SIM_EXTRA_Z);
    world[data].kind = BlockKind::Switch { is_on: data_value };
    world[enable].kind = BlockKind::Switch {
        is_on: enable_value,
    };
    world.initialize_redstone_states();
    let world = World::from(&world);
    let sim = run_d_latch_simulation(&world)?;
    let set_value = match sim.world()[set].kind {
        BlockKind::Torch { is_on } => is_on,
        _ => return None,
    };
    let reset_value = match sim.world()[reset].kind {
        BlockKind::Torch { is_on } => is_on,
        _ => return None,
    };
    Some((set_value, reset_value))
}

fn d_latch_candidate_behaves(
    world: &World3D,
    data_node_id: GraphNodeId,
    enable_node_id: GraphNodeId,
    state: &PlacementState,
    q: Position,
    nq: Position,
) -> bool {
    let Some(data) = state.node_position(data_node_id) else {
        return false;
    };
    let Some(enable) = state.node_position(enable_node_id) else {
        return false;
    };

    let mut reset_world = pad_world_top_for_simulation(world, D_LATCH_SIM_EXTRA_Z);
    reset_world[enable].kind = BlockKind::Switch { is_on: true };
    reset_world[data].kind = BlockKind::Switch { is_on: false };
    reset_world.initialize_redstone_states();
    let world = World::from(&reset_world);
    let Some(mut sim) = run_d_latch_simulation(&world) else {
        return false;
    };

    if torch_is_on_in_world3d(sim.world(), q) || !torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if !change_d_latch_switch(&mut sim, enable, false) {
        return false;
    }
    if !change_d_latch_switch(&mut sim, data, true) {
        return false;
    }
    if torch_is_on_in_world3d(sim.world(), q) || !torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if !change_d_latch_switch(&mut sim, enable, true) {
        return false;
    }
    if !torch_is_on_in_world3d(sim.world(), q) || torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if !change_d_latch_switch(&mut sim, enable, false) {
        return false;
    }
    if !change_d_latch_switch(&mut sim, data, false) {
        return false;
    }
    if !torch_is_on_in_world3d(sim.world(), q) || torch_is_on_in_world3d(sim.world(), nq) {
        return false;
    }
    if !change_d_latch_switch(&mut sim, enable, true) {
        return false;
    }

    !torch_is_on_in_world3d(sim.world(), q) && torch_is_on_in_world3d(sim.world(), nq)
}

fn run_d_latch_simulation(world: &World) -> Option<Simulator> {
    Simulator::from_with_limits_and_trace(
        world,
        D_LATCH_SIM_MAX_CYCLES,
        D_LATCH_SIM_MAX_EVENTS,
        D_LATCH_SIM_TRACE_LIMIT,
    )
    .ok()
}

fn change_d_latch_switch(sim: &mut Simulator, position: Position, value: bool) -> bool {
    sim.change_state_with_limits(
        vec![(position, value)],
        D_LATCH_SIM_MAX_CYCLES,
        D_LATCH_SIM_MAX_EVENTS,
    )
    .is_ok()
}

fn torch_is_on_in_world3d(world: &World3D, position: Position) -> bool {
    matches!(world[position].kind, BlockKind::Torch { is_on: true })
}

fn pad_world_top_for_simulation(world: &World3D, extra_z: usize) -> World3D {
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
