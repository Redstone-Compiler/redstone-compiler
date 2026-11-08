//! Bounded local-candidate diagnostic for the decoder's three-input AND cone.
use redstone_compiler::ir::{
    LogicalDesign, RoutableModule, RoutableModuleBody, RoutableNode, RoutableNodeKind,
};
use redstone_compiler::transform::place_and_route::global_pnr::candidate::{
    generate_routable_module_candidates_with_progress_label, UnitCandidateConfig,
};
use redstone_compiler::transform::place_and_route::local_placer::{
    LocalPlacerConfig, PlacementSchedulePolicy,
};
use redstone_compiler::transform::place_and_route::sampling::SamplingPolicy;
use redstone_compiler::world::position::DimSize;

fn evaluate(module: &RoutableModule, mask: usize) -> bool {
    let RoutableModuleBody::Leaf { nodes } = &module.body else {
        panic!()
    };
    fn value(id: usize, nodes: &[RoutableNode], mask: usize) -> bool {
        let node = nodes.iter().find(|node| node.id == id).unwrap();
        match &node.kind {
            RoutableNodeKind::Input { name } => {
                mask & (1 << name.strip_prefix('a').unwrap().parse::<usize>().unwrap()) != 0
            }
            RoutableNodeKind::Not => !value(node.inputs[0], nodes, mask),
            RoutableNodeKind::Or => node.inputs.iter().any(|id| value(*id, nodes, mask)),
            RoutableNodeKind::Output { .. } => value(node.inputs[0], nodes, mask),
            _ => panic!(),
        }
    }
    value(
        nodes
            .iter()
            .find(|node| matches!(node.kind, RoutableNodeKind::Output { .. }))
            .unwrap()
            .id,
        nodes,
        mask,
    )
}
fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::try_init().ok();
    let logical =
        LogicalDesign::from_verilog_source(include_str!("../test/alu_display/bcd_to_7seg.v"))?;
    let mapped = logical.lower_to_routable_with_bit_mapping_limits(4, 3)?;
    let original = mapped.module("bcd_to_7seg_bit_19").unwrap().clone();
    let mut flat = original.clone();
    flat.name = "flat_and3".into();
    let mut nodes = Vec::new();
    for i in 0..3 {
        nodes.push(RoutableNode {
            id: i,
            kind: RoutableNodeKind::Input {
                name: format!("a{i}"),
            },
            inputs: vec![],
            tag: String::new(),
        });
    }
    for i in 0..3 {
        nodes.push(RoutableNode {
            id: i + 3,
            kind: RoutableNodeKind::Not,
            inputs: vec![i],
            tag: String::new(),
        });
    }
    nodes.push(RoutableNode {
        id: 6,
        kind: RoutableNodeKind::Or,
        inputs: vec![3, 4, 5],
        tag: String::new(),
    });
    nodes.push(RoutableNode {
        id: 7,
        kind: RoutableNodeKind::Not,
        inputs: vec![6],
        tag: String::new(),
    });
    nodes.push(RoutableNode {
        id: 8,
        kind: RoutableNodeKind::Output { name: "y".into() },
        inputs: vec![7],
        tag: String::new(),
    });
    flat.body = RoutableModuleBody::Leaf { nodes };
    for mask in 0..8 {
        eyre::ensure!(
            evaluate(&original, mask) == evaluate(&flat, mask),
            "parity mismatch"
        );
    }
    let config = UnitCandidateConfig {
        dim: DimSize(8, 8, 4),
        max_candidates: 4,
        local_config: LocalPlacerConfig {
            greedy_input_generation: true,
            input_candidate_limit: Some(16),
            schedule: PlacementSchedulePolicy::Auto,
            placement_sampling_policy: LocalPlacerConfig::ranked_sampling(64, 0, 0),
            max_route_step: 4,
            route_step_sampling_policy: SamplingPolicy::Random(32),
            not_route_step_sampling_policy: SamplingPolicy::Random(8),
            ..Default::default()
        },
        ..Default::default()
    };
    let selection = std::env::args().nth(1).unwrap_or("twoinputs".into());
    if selection == "twoinputs" {
        let path = std::env::args()
            .nth(2)
            .unwrap_or("test/alu_display/bcd_to_7seg.v".into());
        let logical = LogicalDesign::from_verilog_source(&std::fs::read_to_string(path)?)?;
        let mapped = logical.lower_to_routable_with_bit_mapping_limits(4, 2)?;
        std::fs::create_dir_all("target/alu-display")?;
        std::fs::write(
            "target/alu-display/decoder-twoinputs.rcir",
            mapped.to_string(),
        )?;
        if let RoutableModuleBody::Composite { instances, .. } =
            &mapped.module(&mapped.top).unwrap().body
        {
            println!(
                "instances={} definitions={}",
                instances.len(),
                mapped.modules.len() - 1
            );
        }
        for module in &mapped.modules {
            if let RoutableModuleBody::Leaf { nodes } = &module.body {
                let started = std::time::Instant::now();
                let result = generate_routable_module_candidates_with_progress_label(
                    module,
                    &config,
                    Some(&module.name),
                );
                println!(
                    "name={} nodes={} elapsed={:?} result={:?}",
                    module.name,
                    nodes.len(),
                    started.elapsed(),
                    result.as_ref().map(|v| v.len())
                );
            }
        }
        return Ok(());
    }
    eyre::ensure!(
        selection == "original",
        "flat AND3 passes all eight Boolean cases, but its 3-input OR is unsupported by the binary local placer; use original or twoinputs"
    );
    let candidate = original;
    let started = std::time::Instant::now();
    let result = generate_routable_module_candidates_with_progress_label(
        &candidate,
        &config,
        Some(&candidate.name),
    );
    println!(
        "name={} elapsed={:?} result={:?}",
        candidate.name,
        started.elapsed(),
        result.as_ref().map(|v| v.len())
    );
    Ok(())
}
