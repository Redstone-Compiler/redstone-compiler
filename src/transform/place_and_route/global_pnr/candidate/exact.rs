//! Bounded SAT synthesis followed by the same external-port adaptation as local search.
use std::time::{Duration, Instant};

use super::*;
use crate::transform::logic::LogicGraphTransformer;
use crate::transform::place_and_route::estimate::bounding_box;
use crate::transform::place_and_route::global_pnr::ir::ExactCandidateProvenance;
use crate::transform::place_and_route::local_placer::exact::{
    ExactLocalPlacer, ExactOutcome, ExactPlacerConfig,
};

pub(super) fn generate(
    module_name: &str,
    graph: &Graph,
    ports: &[CandidatePort],
    config: &UnitCandidateConfig,
    input_mode: CandidateInputMode,
) -> eyre::Result<Vec<LayoutCandidate>> {
    let Some(policy) = &config.exact else {
        return Ok(Vec::new());
    };
    eyre::ensure!(
        policy.search_box.iter().all(|v| *v > 0)
            && policy.time_limit_ms > 0
            && (1..=6).contains(&policy.max_inputs)
            && policy.max_logic_nodes > 0,
        "invalid Exact candidate bounds"
    );
    if config.max_candidates == 0
        || config.input_constraints != LocalPlacerInputConstraints::default()
        || graph
            .nodes
            .iter()
            .any(|n| matches!(n.kind, GraphNodeKind::Sequential(_)))
        || ports
            .iter()
            .filter(|p| p.direction == PhysicalPortDirection::Input)
            .count()
            > policy.max_inputs
        || graph
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, GraphNodeKind::Logic(_)))
            .count()
            > policy.max_logic_nodes
    {
        return Ok(Vec::new());
    }
    let original = LogicGraph {
        graph: graph.clone(),
    };
    // Mapping inserts double-NOT physical buffers. They are identities, not
    // required gates in exact synthesis; keep the truth contract as the oracle.
    let mut transform = LogicGraphTransformer::new(original.clone());
    transform.decompose_binops();
    transform.decompose_xor()?;
    transform.decompose_and()?;
    transform.remove_double_neg_expression();
    transform.optimize_cse()?;
    let normalized = transform.finish();
    let solver = match ExactLocalPlacer::new(&normalized) {
        Ok(solver) => solver.with_name(module_name),
        Err(error) => {
            tracing::debug!(module=module_name, %error, "Exact unsupported; retaining normal candidates");
            return Ok(Vec::new());
        }
    };
    let [x, y, z] = policy.search_box;
    let mut exact = ExactPlacerConfig::new(DimSize(x, y, z));
    exact.time_limit = Some(Duration::from_millis(policy.time_limit_ms));
    exact.workers = 1;
    exact.seed = 1;
    exact.driving_outputs = ports
        .iter()
        .filter(|p| p.direction == PhysicalPortDirection::Output)
        .map(|p| p.target.clone())
        .collect();
    let start = Instant::now();
    let (placement, optimal) = if policy.optimize {
        let (placement, optimal, _) = solver.place_minimizing_blocks(&exact)?;
        (placement, optimal)
    } else {
        let (outcome, stats) = solver.place(&exact)?;
        (
            match outcome {
                ExactOutcome::Placed(placement) => Some(*placement),
                _ => None,
            },
            stats.optimal,
        )
    };
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let Some(placement) = placement else {
        tracing::info!(
            module = module_name,
            elapsed_ms,
            "Exact produced no candidate within bounds; using fallback"
        );
        return Ok(Vec::new());
    };
    let bbox = bounding_box(&placement.placed.world).context("Exact returned empty placement")?;
    let provenance = ExactCandidateProvenance {
        search_box: policy.search_box,
        core_bbox: [bbox.width(), bbox.depth(), bbox.height()],
        core_blocks: placement.placed.world.iter_block().len(),
        core_logic_blocks: placement.block_count,
        elapsed_ms,
        optimal,
    };
    if !candidate_matches_truth_table(&original, &placement.placed)? {
        tracing::info!(
            module = module_name,
            "Exact standalone verification rejected"
        );
        return Ok(Vec::new());
    }
    let placed = placement.placed;
    let standalone = placed.clone();
    let (world, physical_ports) = candidate_layout(
        ports,
        false,
        &config.input_constraints,
        placed.world,
        &placed.inputs,
        &placed.outputs,
        input_mode,
    );
    if !missing_candidate_ports(ports, &physical_ports).is_empty() {
        tracing::info!(module = module_name, "Exact endpoint conversion rejected");
        return Ok(Vec::new());
    }
    let mut candidate = LayoutCandidate::from_world(module_name.to_owned(), world, physical_ports)?;
    if input_mode == CandidateInputMode::ExternalPorts {
        super::exact_adapter::restore_attached_input_targets(&mut candidate, &standalone, ports);
        // External Exact candidates require a validated, independently drivable
        // interface even if other recipe backends use their own port adapters.
        let Some(reserved) = io_diodes::reserve(candidate)? else {
            tracing::info!(module = module_name, "Exact IO diode adaptation rejected");
            return Ok(Vec::new());
        };
        candidate = reserved;
        if !super::exact_validation::verify_external_candidate(&original, &candidate)? {
            tracing::info!(module = module_name, "Exact external verification rejected");
            return Ok(Vec::new());
        }
    }
    if !candidate_ports_cover_module_ports(ports, &candidate.ports) {
        tracing::info!(
            module = module_name,
            "Exact adapted inputs are missing or aliased"
        );
        return Ok(Vec::new());
    }
    if !candidate_satisfies_local_cell_contract(&mut candidate, &config.local_cell_contract) {
        tracing::info!(
            module = module_name,
            "Exact adapted geometry violates local cell contract"
        );
        return Ok(Vec::new());
    }
    candidate.exact_core = Some(provenance);
    tracing::info!(
        module = module_name,
        elapsed_ms,
        core_blocks = candidate.exact_core.as_ref().unwrap().core_blocks,
        adapted_blocks = candidate.cost.block_count,
        adapted_volume = candidate.cost.bbox_volume,
        "accepted SAT Exact candidate"
    );
    Ok(vec![candidate])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sat_candidate_preserves_external_truth_and_provenance() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("a|b", "y")?;
        let ports = vec![
            CandidatePort::new("a", "a", PhysicalPortDirection::Input),
            CandidatePort::new("b", "b", PhysicalPortDirection::Input),
            CandidatePort::new("y", "y", PhysicalPortDirection::Output),
        ];
        let config = UnitCandidateConfig {
            exact: Some(ExactCandidateConfig::default()),
            ..Default::default()
        };
        let candidates = generate(
            "sat_or",
            &graph.graph,
            &ports,
            &config,
            CandidateInputMode::ExternalPorts,
        )?;
        assert!(
            !candidates.is_empty(),
            "bounded SAT OR must survive external adaptation"
        );
        let candidate = &candidates[0];
        assert!(candidate.exact_core.is_some());
        assert!(candidate.exact_core.as_ref().unwrap().core_blocks < 20);
        assert!(super::super::exact_validation::verify_external_candidate(
            &graph, candidate
        )?);
        let cache_dir = std::env::temp_dir().join(format!(
            "redstone-exact-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        crate::transform::place_and_route::global_pnr::candidate_cache::store(
            &cache_dir,
            "exact",
            &candidates,
        )?;
        let restored = crate::transform::place_and_route::global_pnr::candidate_cache::load(
            &cache_dir, "exact", "sat_or",
        )?
        .unwrap();
        assert_eq!(restored[0].exact_core, candidate.exact_core);
        assert!(super::super::exact_validation::verify_external_candidate(
            &graph,
            &restored[0]
        )?);
        std::fs::remove_dir_all(cache_dir)?;

        let mut disabled = config.clone();
        disabled.exact = None;
        assert!(generate(
            "off",
            &graph.graph,
            &ports,
            &disabled,
            CandidateInputMode::ExternalPorts
        )?
        .is_empty());
        Ok(())
    }
    #[test]
    fn sat_mixed_polarity_candidate_cache_preserves_truth_and_provenance() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("a|~b", "y")?;
        let ports = vec![
            CandidatePort::new("a", "a", PhysicalPortDirection::Input),
            CandidatePort::new("b", "b", PhysicalPortDirection::Input),
            CandidatePort::new("y", "y", PhysicalPortDirection::Output),
        ];
        let config = UnitCandidateConfig {
            exact: Some(ExactCandidateConfig {
                time_limit_ms: 10_000,
                ..Default::default()
            }),
            ..Default::default()
        };
        let candidates = generate(
            "sat_mixed",
            &graph.graph,
            &ports,
            &config,
            CandidateInputMode::ExternalPorts,
        )?;
        assert!(
            !candidates.is_empty(),
            "bounded SAT OR must survive external adaptation"
        );
        let candidate = &candidates[0];
        assert!(candidate.exact_core.is_some());
        assert!(super::super::exact_validation::verify_external_candidate(
            &graph, candidate
        )?);
        let cache_dir = std::env::temp_dir().join(format!(
            "redstone-exact-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        crate::transform::place_and_route::global_pnr::candidate_cache::store(
            &cache_dir,
            "exact",
            &candidates,
        )?;
        let restored = crate::transform::place_and_route::global_pnr::candidate_cache::load(
            &cache_dir,
            "exact",
            "sat_mixed",
        )?
        .unwrap();
        assert_eq!(restored[0].exact_core, candidate.exact_core);
        assert!(super::super::exact_validation::verify_external_candidate(
            &graph,
            &restored[0]
        )?);
        std::fs::remove_dir_all(cache_dir)?;

        let mut disabled = config.clone();
        disabled.exact = None;
        assert!(generate(
            "off",
            &graph.graph,
            &ports,
            &disabled,
            CandidateInputMode::ExternalPorts
        )?
        .is_empty());
        Ok(())
    }
}
