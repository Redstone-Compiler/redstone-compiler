use std::collections::{BTreeMap, BTreeSet};

use eyre::{ensure, Context};

use super::{PhysicalCellBuild, PhysicalCellDocument};
use crate::graph::logic::LogicGraph;
use crate::world::position::Position;
use crate::world::simulator::Simulator;
use crate::world::World;

#[derive(Clone, Debug)]
pub struct PhysicalCellCaseSimulation {
    pub inputs: BTreeMap<String, bool>,
    pub expected: BTreeMap<String, bool>,
    pub actual: BTreeMap<String, bool>,
    pub simulator: Simulator,
}

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
    pub input_names: Vec<String>,
    pub signatures: BTreeMap<String, PhysicalCellTruthSignature>,
    pub first_divergence: Option<PhysicalCellDivergence>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalCellTruthSignature {
    pub expected: Vec<bool>,
    pub actual: Vec<bool>,
    pub expected_influence: BTreeSet<String>,
    pub actual_influence: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalCellDivergence {
    pub observation: String,
    pub position: Position,
    pub case_index: usize,
    pub inputs: BTreeMap<String, bool>,
    pub expected: bool,
    pub actual: bool,
}

impl PhysicalCellDocument {
    fn verification_truth(
        &self,
        build: &PhysicalCellBuild,
    ) -> eyre::Result<crate::graph::logic::LogicTruthTable> {
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
        let declared_outputs = build
            .observations()
            .map(|(name, _)| name.to_owned())
            .collect::<BTreeSet<_>>();
        ensure!(
            expected_outputs == declared_outputs,
            "expectations define observations {:?}, but the cell declares {:?}",
            expected_outputs,
            declared_outputs
        );
        Ok(truth)
    }

    pub fn simulate_case(
        &self,
        build: &PhysicalCellBuild,
        inputs: BTreeMap<String, bool>,
        trace_limit: usize,
    ) -> eyre::Result<PhysicalCellCaseSimulation> {
        let truth = self.verification_truth(build)?;
        let supplied_inputs = inputs.keys().cloned().collect::<BTreeSet<_>>();
        let expected_inputs = truth.input_names.iter().cloned().collect::<BTreeSet<_>>();
        ensure!(
            supplied_inputs == expected_inputs,
            "case supplies inputs {:?}, but the cell expects {:?}",
            supplied_inputs,
            expected_inputs
        );
        let mask = truth
            .input_names
            .iter()
            .enumerate()
            .fold(0usize, |mask, (index, name)| {
                mask | (usize::from(inputs[name]) << index)
            });
        let expected = truth
            .output_tables
            .iter()
            .map(|(name, values)| (name.clone(), values[mask]))
            .collect::<BTreeMap<_, _>>();

        let world = World::from(&build.world);
        let mut simulator = Simulator::from_with_limits_and_trace(&world, 256, 50_000, trace_limit)
            .map_err(|error| eyre::eyre!(error.message().to_owned()))
            .context("failed to initialize physical cell simulation")?;
        simulator.drive_inputs_with_limits(
            inputs
                .iter()
                .flat_map(|(name, value)| {
                    build.input_contacts[name]
                        .iter()
                        .map(move |position| (*position, *value))
                })
                .collect(),
            256,
            50_000,
        )?;
        let actual = build
            .observations()
            .map(|(name, position)| {
                (
                    name.to_owned(),
                    simulator.world()[position].kind.is_powered(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        Ok(PhysicalCellCaseSimulation {
            inputs,
            expected,
            actual,
            simulator,
        })
    }

    pub fn verify(&self, build: &PhysicalCellBuild) -> eyre::Result<PhysicalCellVerification> {
        let truth = self.verification_truth(build)?;

        let mut failures = Vec::new();
        let cases = 1usize << truth.input_names.len();
        let mut case_inputs = Vec::with_capacity(cases);
        let mut expected_values = truth
            .output_tables
            .keys()
            .map(|name| (name.clone(), Vec::with_capacity(cases)))
            .collect::<BTreeMap<_, _>>();
        let mut actual_values = expected_values.clone();
        for mask in 0..cases {
            let inputs = truth
                .input_names
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), mask & (1 << index) != 0))
                .collect::<BTreeMap<_, _>>();
            let case = self.simulate_case(build, inputs.clone(), 0)?;
            let expected = case.expected;
            let actual = case.actual;
            case_inputs.push(inputs.clone());
            for (name, value) in &expected {
                expected_values.get_mut(name).unwrap().push(*value);
            }
            for (name, value) in &actual {
                actual_values.get_mut(name).unwrap().push(*value);
            }
            if actual != expected {
                failures.push(PhysicalCellCaseFailure {
                    inputs,
                    expected,
                    actual,
                });
            }
        }
        let signatures = expected_values
            .into_iter()
            .map(|(name, expected)| {
                let actual = actual_values.remove(&name).unwrap();
                let expected_influence = influences(&truth.input_names, &expected);
                let actual_influence = influences(&truth.input_names, &actual);
                (
                    name,
                    PhysicalCellTruthSignature {
                        expected,
                        actual,
                        expected_influence,
                        actual_influence,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut observation_order = self
            .probes
            .iter()
            .map(|probe| probe.name.as_str())
            .chain(self.outputs.iter().map(|output| output.name.as_str()));
        let first_divergence = observation_order.find_map(|name| {
            let signature = &signatures[name];
            signature
                .expected
                .iter()
                .zip(&signature.actual)
                .position(|(expected, actual)| expected != actual)
                .map(|case_index| PhysicalCellDivergence {
                    observation: name.to_owned(),
                    position: build.observation_position(name).unwrap(),
                    case_index,
                    inputs: case_inputs[case_index].clone(),
                    expected: signature.expected[case_index],
                    actual: signature.actual[case_index],
                })
        });

        Ok(PhysicalCellVerification {
            cases,
            failures,
            input_names: truth.input_names,
            signatures,
            first_divergence,
        })
    }
}

fn influences(input_names: &[String], values: &[bool]) -> BTreeSet<String> {
    input_names
        .iter()
        .enumerate()
        .filter_map(|(input_index, name)| {
            let bit = 1usize << input_index;
            (0..values.len())
                .filter(|mask| mask & bit == 0)
                .any(|mask| values[mask] != values[mask | bit])
                .then(|| name.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{BlockKind, Direction};
    use crate::world::position::{DimSize, Position};

    const INVERTER: &str = include_str!("../../test/inverter.rcell");
    const NOR_CASCADE: &str = include_str!("../../test/nor-cascade-primitive.rcell");
    const FULL_ADDER_2X20X20: &str = include_str!("../../test/full-adder-2x20x20.rcell");

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
    fn physical_cell_reports_probe_signatures_in_declaration_order() -> eyre::Result<()> {
        let source = INVERTER.replace(
            "output \"y\" at [1, 2, 1];",
            "output \"y\" at [1, 2, 1];\n  probe \"tap\" at [1, 2, 1];",
        );
        let source = source.replace(
            "expect \"y\" = ~a;",
            "expect \"y\" = ~a;\n  expect \"tap\" = ~a;",
        );
        let document: PhysicalCellDocument = source.parse()?;
        let emitted = document.to_string();
        assert!(emitted.contains("probe \"tap\" at [1, 2, 1];"));

        let build = document.build()?;
        assert_eq!(build.observation_position("tap"), Some(Position(1, 2, 1)));
        let verification = document.verify(&build)?;
        assert!(verification.failures.is_empty());
        assert_eq!(verification.signatures["tap"].expected, vec![true, false]);
        assert_eq!(verification.signatures["tap"].actual, vec![true, false]);
        assert_eq!(
            verification.signatures["tap"].actual_influence,
            BTreeSet::from(["a".to_owned()])
        );
        assert_eq!(verification.first_divergence, None);
        Ok(())
    }

    #[test]
    fn repeated_input_declarations_drive_all_physical_contacts() -> eyre::Result<()> {
        let source = r#"
            rcell 1;
            cell "two-contact-input" size [2, 3, 2] {
              glyph "T" = torch on y-;
              input "a" at [0, 1, 1] on x+;
              input "a" at [1, 0, 1] on y+;
              output "y" at [1, 2, 1];
              plane yz at x=1 {
                z=1 ".#T";
              }
              expect "y" = ~a;
            }
        "#;
        let document: PhysicalCellDocument = source.parse()?;
        let build = document.build()?;
        assert_eq!(build.inputs.len(), 1);
        assert_eq!(build.input_contacts["a"].len(), 2);
        let verification = document.verify(&build)?;
        assert_eq!(verification.cases, 2);
        assert!(verification.failures.is_empty());
        Ok(())
    }

    #[test]
    fn physical_cell_selects_the_first_failing_probe() -> eyre::Result<()> {
        let source = INVERTER.replace(
            "output \"y\" at [1, 2, 1];",
            "output \"y\" at [1, 2, 1];\n  probe \"tap\" at [1, 2, 1];",
        );
        let source = source.replace(
            "expect \"y\" = ~a;",
            "expect \"y\" = ~a;\n  expect \"tap\" = a;",
        );
        let document: PhysicalCellDocument = source.parse()?;
        let build = document.build()?;
        let verification = document.verify(&build)?;
        let divergence = verification.first_divergence.unwrap();
        assert_eq!(divergence.observation, "tap");
        assert_eq!(divergence.case_index, 0);
        assert_eq!(divergence.inputs, BTreeMap::from([("a".to_owned(), false)]));
        assert!(!divergence.expected);
        assert!(divergence.actual);
        Ok(())
    }

    #[test]
    fn physical_cell_simulates_one_named_case_with_diagnostics() -> eyre::Result<()> {
        let document: PhysicalCellDocument = INVERTER.parse()?;
        let build = document.build()?;
        let case = document.simulate_case(&build, BTreeMap::from([("a".to_owned(), true)]), 128)?;
        assert_eq!(case.expected, BTreeMap::from([("y".to_owned(), false)]));
        assert_eq!(case.actual, case.expected);
        assert!(!case.simulator.trace().is_empty());
        let output = build.outputs["y"];
        let sources = case.simulator.diagnostic_power_sources(output);
        assert!(sources
            .iter()
            .any(|source| source.relation == "torch-support"));
        Ok(())
    }

    #[test]
    fn physical_cell_cascades_torches_through_a_hard_power_edge() -> eyre::Result<()> {
        let document: PhysicalCellDocument = NOR_CASCADE.parse()?;
        let build = document.build()?;
        let verification = document.verify(&build)?;
        assert_eq!(verification.cases, 4);
        assert!(verification.failures.is_empty());

        let case = document.simulate_case(
            &build,
            BTreeMap::from([("a".to_owned(), false), ("b".to_owned(), false)]),
            128,
        )?;
        let support = Position(1, 4, 1);
        assert!(case
            .simulator
            .diagnostic_power_sources(support)
            .iter()
            .any(|source| source.hard && source.relation == "powers-block"));
        Ok(())
    }

    #[test]
    fn full_adder_fits_2x20x20_with_three_switches_and_passes_all_cases() -> eyre::Result<()> {
        let document: PhysicalCellDocument = FULL_ADDER_2X20X20.parse()?;
        assert_eq!(document.size, DimSize(2, 20, 20));
        assert_eq!(
            document
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["a", "b", "cin"])
        );

        let build = document.build()?;
        assert_eq!(
            build
                .world
                .iter_block()
                .into_iter()
                .filter(|(_, block)| matches!(block.kind, BlockKind::Switch { .. }))
                .count(),
            3
        );

        let verification = document.verify(&build)?;
        assert_eq!(verification.cases, 8);
        assert!(verification.failures.is_empty());
        assert_eq!(
            verification.signatures["sum"].actual,
            vec![false, true, true, false, true, false, false, true]
        );
        assert_eq!(
            verification.signatures["cout"].actual,
            vec![false, false, false, true, false, true, true, true]
        );
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
