//! Independent simulation gate for compact exact cells after boundary conversion.
use super::*;
use crate::nbt::{NBTRoot, ToNBT};
use crate::world::simulator::MANUAL_INPUT_IDLE_CYCLES;

/// Validate the exported interface rather than the solver's switch-bearing body.
/// A rejected candidate may fall back to the existing verified recipe.
pub(super) fn verify_external_candidate(
    graph: &LogicGraph,
    candidate: &LayoutCandidate,
) -> eyre::Result<bool> {
    let truth = graph.truth_table()?;
    let count = truth.input_names.len();
    if count > 8 {
        return Ok(false);
    }
    let inputs = truth
        .input_names
        .iter()
        .map(|name| {
            candidate
                .ports
                .iter()
                .find(|p| p.name == *name && p.direction == PhysicalPortDirection::Input)
        })
        .collect::<Option<Vec<_>>>();
    let Some(inputs) = inputs else {
        return Ok(false);
    };
    let outputs = truth
        .output_tables
        .iter()
        .map(|(name, values)| {
            candidate
                .ports
                .iter()
                .find(|p| p.name == *name && p.direction == PhysicalPortDirection::Output)
                .map(|port| (port, values))
        })
        .collect::<Option<Vec<_>>>();
    let Some(outputs) = outputs else {
        return Ok(false);
    };
    let world = NBTRoot::from_nbt_bytes(&fastnbt::to_bytes(&candidate.world.to_nbt())?)?.to_world();
    let fresh = || Simulator::from_settled_with_limits_and_trace(&world, 1024, 200_000, 0);
    let drive = |sim: &mut Simulator, mask: usize| -> eyre::Result<bool> {
        sim.advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)?;
        sim.drive_inputs_with_limits(
            inputs
                .iter()
                .enumerate()
                .map(|(bit, p)| (p.position, mask & (1 << bit) != 0))
                .collect(),
            1024,
            200_000,
        )?;
        for (bit, input) in inputs.iter().enumerate() {
            if sim.world()[input.position].kind.is_powered() != (mask & (1 << bit) != 0) {
                return Ok(false);
            }
        }
        for (output, values) in &outputs {
            let block = sim.world()[output.position].kind;
            if block.is_powered() != values[mask] {
                return Ok(false);
            }
            if values[mask]
                && output.connection == PortConnection::RestoredDirect
                && !matches!(block, BlockKind::Redstone { strength: 15, .. })
            {
                return Ok(false);
            }
        }
        Ok(sim
            .world()
            .iter_block()
            .iter()
            .all(|(p, _)| !sim.is_torch_burned_out(*p)))
    };
    let states = 1 << count;
    for mask in 0..states {
        if !drive(&mut fresh()?, mask)? {
            return Ok(false);
        }
    }
    let mut sim = fresh()?;
    if count <= 4 {
        for from in 0..states {
            for to in 0..states {
                if !drive(&mut sim, from)? || !drive(&mut sim, to)? {
                    return Ok(false);
                }
            }
        }
    } else {
        for mask in (0..states).chain((0..states).rev()) {
            if !drive(&mut sim, mask)? {
                return Ok(false);
            }
        }
    }
    // An unrelated route touching an output must not feed the input boundary.
    for (output, _) in &outputs {
        let mut sim = fresh()?;
        if !drive(&mut sim, 0)? {
            return Ok(false);
        }
        sim.drive_inputs_with_limits(vec![(output.position, true)], 1024, 200_000)?;
        if inputs
            .iter()
            .any(|p| sim.world()[p.position].kind.is_powered())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verified_recipe_export_passes_but_a_shorted_input_is_rejected() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("a&~b", "y")?;
        let candidate = super::super::nary::build_verified_candidate(&graph, "arbitrary")?.unwrap();
        assert!(verify_external_candidate(&graph, &candidate)?);
        // Moving a restored-dust label onto the preceding torch preserves
        // Boolean values but violates the electrical strength/type contract.
        let mut false_promise = candidate.clone();
        let out = false_promise
            .ports
            .iter_mut()
            .find(|p| p.name == "y")
            .unwrap();
        out.position = Position(out.position.0 - 2, out.position.1, out.position.2);
        out.route_position = Some(out.position);
        out.access_points = vec![out.position];
        assert!(!verify_external_candidate(&graph, &false_promise)?);
        let mut shorted = candidate.clone();
        let a = shorted
            .ports
            .iter()
            .find(|p| p.name == "a")
            .unwrap()
            .position;
        let b = shorted.ports.iter_mut().find(|p| p.name == "b").unwrap();
        b.position = a;
        b.route_position = Some(a);
        b.access_points = vec![a];
        assert!(!verify_external_candidate(&graph, &shorted)?);
        Ok(())
    }
}
