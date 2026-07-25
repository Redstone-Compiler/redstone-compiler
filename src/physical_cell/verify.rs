use std::collections::{BTreeMap, BTreeSet};

use eyre::{ensure, Context};

use super::{PhysicalCellBuild, PhysicalCellDocument};
use crate::graph::logic::LogicGraph;
use crate::world::simulator::Simulator;
use crate::world::World;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalCellCaseFailure {
    pub inputs: BTreeMap<String, bool>,
    pub expected: BTreeMap<String, bool>,
    pub actual: BTreeMap<String, bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalCellVerification {
    pub cases: usize,
    pub failures: Vec<PhysicalCellCaseFailure>,
}

impl PhysicalCellDocument {
    pub fn verify(&self, build: &PhysicalCellBuild) -> eyre::Result<PhysicalCellVerification> {
        ensure!(
            !self.expectations.is_empty(),
            "cell has no `expect` statements"
        );
        let graph = LogicGraph::from_assignments(
            self.expectations
                .iter()
                .map(|expectation| (expectation.output.clone(), expectation.expression.clone())),
        )?;
        let truth = graph.truth_table()?;
        let expected_inputs = truth.input_names.iter().cloned().collect::<BTreeSet<_>>();
        let declared_inputs = build.inputs.keys().cloned().collect::<BTreeSet<_>>();
        ensure!(
            expected_inputs == declared_inputs,
            "expectations use inputs {:?}, but the cell declares {:?}",
            expected_inputs,
            declared_inputs
        );
        let expected_outputs = truth.output_tables.keys().cloned().collect::<BTreeSet<_>>();
        let declared_outputs = build.outputs.keys().cloned().collect::<BTreeSet<_>>();
        ensure!(
            expected_outputs == declared_outputs,
            "expectations define outputs {:?}, but the cell declares {:?}",
            expected_outputs,
            declared_outputs
        );

        let mut failures = Vec::new();
        let cases = 1usize << truth.input_names.len();
        for mask in 0..cases {
            let inputs = truth
                .input_names
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), mask & (1 << index) != 0))
                .collect::<BTreeMap<_, _>>();
            let world = World::from(&build.world);
            let mut simulator = Simulator::from_with_limits_and_trace(&world, 256, 50_000, 0)
                .map_err(|error| eyre::eyre!(error.message().to_owned()))
                .context("failed to initialize physical cell simulation")?;
            simulator.drive_inputs_with_limits(
                inputs
                    .iter()
                    .map(|(name, value)| (build.inputs[name], *value))
                    .collect(),
                256,
                50_000,
            )?;

            let expected = truth
                .output_tables
                .iter()
                .map(|(name, values)| (name.clone(), values[mask]))
                .collect::<BTreeMap<_, _>>();
            let actual = build
                .outputs
                .iter()
                .map(|(name, position)| {
                    (name.clone(), simulator.world()[*position].kind.is_powered())
                })
                .collect::<BTreeMap<_, _>>();
            if actual != expected {
                failures.push(PhysicalCellCaseFailure {
                    inputs,
                    expected,
                    actual,
                });
            }
        }
        Ok(PhysicalCellVerification { cases, failures })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{BlockKind, Direction};
    use crate::world::position::Position;

    const INVERTER: &str = include_str!("../../test/inverter.rcell");

    #[test]
    fn physical_cell_round_trips_canonical_text() -> eyre::Result<()> {
        let first: PhysicalCellDocument = INVERTER.parse()?;
        let text = first.to_string();
        let second: PhysicalCellDocument = text.parse()?;
        assert_eq!(first, second);
        assert!(text.contains("plane yz at x=1"));
        Ok(())
    }

    #[test]
    fn physical_cell_builds_directional_blocks_and_verifies_logic() -> eyre::Result<()> {
        let document: PhysicalCellDocument = INVERTER.parse()?;
        let build = document.build()?;
        assert!(matches!(
            build.world[Position(0, 1, 1)].kind,
            BlockKind::Switch { .. }
        ));
        assert_eq!(build.world[Position(0, 1, 1)].direction, Direction::East);
        assert!(matches!(
            build.world[Position(1, 2, 1)].kind,
            BlockKind::Torch { .. }
        ));
        assert_eq!(build.world[Position(1, 2, 1)].direction, Direction::South);
        let verification = document.verify(&build)?;
        assert_eq!(verification.cases, 2);
        assert!(verification.failures.is_empty());
        Ok(())
    }

    #[test]
    fn auto_support_expands_dust_but_keeps_it_out_of_source_text() -> eyre::Result<()> {
        let source = r#"
            rcell 1;
            cell "wire" size [1, 2, 2] {
              auto-support dust;
              plane yz at x=0 {
                z=1 "rr";
              }
            }
        "#;
        let document: PhysicalCellDocument = source.parse()?;
        let build = document.build()?;
        assert_eq!(
            build.auto_supports,
            BTreeSet::from([Position(0, 0, 0), Position(0, 1, 0)])
        );
        assert_eq!(document.to_string().matches('#').count(), 0);
        Ok(())
    }
}
