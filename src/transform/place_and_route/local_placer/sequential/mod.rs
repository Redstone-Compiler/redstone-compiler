use super::*;

mod d_latch;
mod macro_routes;
mod prefix;
mod rs_latch;
mod scenario;

pub(super) use d_latch::generate_d_latch_gate_routes;
pub(super) use macro_routes::generate_sequential_macro_routes;
#[cfg(test)]
pub(super) use macro_routes::{place_sequential_macro, route_sequential_inputs};
pub(super) use rs_latch::generate_rs_latch_gate_routes;
#[cfg(test)]
pub(super) use rs_latch::{
    generate_rs_latch_not_pairs, route_rs_latch_branches, select_rs_latch_not_pairs,
};

pub(super) fn supports_sequential_primitive(sequential: &SequentialPrimitive) -> bool {
    matches!(
        sequential.sequential_type,
        SequentialType::RsLatch | SequentialType::DLatch
    ) && (sequential.rs_latch_core().is_some()
        || !SequentialMacro::candidates(sequential).is_empty())
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RsLatchInputNodeIds {
    pub(super) set: GraphNodeId,
    pub(super) reset: GraphNodeId,
}

const SEQUENTIAL_SYNTHETIC_NODE_ID_STRIDE: usize = 4;

// Prefix placement runs in its own graph, then publishes only the set/reset
// endpoints back into the outer placement state for the RS latch core router.
pub(super) fn rs_latch_input_node_ids(node_id: GraphNodeId) -> RsLatchInputNodeIds {
    let base = usize::MAX - node_id * SEQUENTIAL_SYNTHETIC_NODE_ID_STRIDE;
    RsLatchInputNodeIds {
        set: base - 1,
        reset: base - 2,
    }
}

fn take_sampling_limit(policy: SamplingPolicy) -> Option<usize> {
    match policy {
        SamplingPolicy::Take(count) => Some(count),
        SamplingPolicy::None | SamplingPolicy::Random(_) => None,
    }
}
