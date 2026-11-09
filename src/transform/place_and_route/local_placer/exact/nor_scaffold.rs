//! Deterministic starting geometry for a NOR cell, not an exact optimum.
//!
//! The upper repeater/dust line restores power between inputs and merges them.
//! It needs the exact model's opt-in case-ranked dust power semantics. Callers must provide actual strong
//! power at the tap solids and independently verify the complete resulting cell.
use std::collections::{BTreeMap, BTreeSet};

use eyre::ensure;

use super::CellKind;
use crate::world::block::Direction;
use crate::world::position::{DimSize, Position};

/// Build a bounded NOR scaffold from strong-solid taps on z=1, y=start_y.
/// A true polarity flag means that the NOR consumes the complement of that tap.
/// The returned map includes the tap solids but does not generate their drivers.
/// The final repeater at `(output_x, dim.1 - 1, 1)` points out of the box.
///
/// This is only a candidate: it does not certify simulation, isolation, timing,
/// or completeness of a later deletion-only exact search over these positions.
pub(super) fn nor_template(
    taps: &[(Position, bool)],
    output_x: usize,
    start_y: usize,
    dim: DimSize,
) -> eyre::Result<BTreeMap<Position, CellKind>> {
    ensure!(!taps.is_empty(), "NOR scaffold needs at least one tap");
    ensure!(dim.2 >= 7, "NOR scaffold needs height >= 7");
    ensure!(output_x < dim.0, "output x is outside the box");
    let last_start = start_y
        .checked_add(8)
        .ok_or_else(|| eyre::eyre!("start y overflow"))?;
    ensure!(
        last_start < dim.1,
        "NOR scaffold needs y=start_y+8 inside the box"
    );
    let mut tap_x = BTreeSet::new();
    for &(position, _) in taps {
        ensure!(
            position.1 == start_y && position.2 == 1,
            "tap {position:?} must lie on y=start_y, z=1"
        );
        let end = position
            .0
            .checked_add(1)
            .ok_or_else(|| eyre::eyre!("tap x overflow"))?;
        ensure!(
            end < dim.0,
            "tap {position:?} needs one column to its right"
        );
        ensure!(tap_x.insert(position.0), "duplicate tap x={}", position.0);
    }
    let ordered = tap_x.iter().copied().collect::<Vec<_>>();
    ensure!(
        ordered.windows(2).all(|pair| pair[1] - pair[0] >= 3),
        "tap spacing must be at least three columns"
    );

    let mut cells = BTreeMap::new();
    fn put(
        cells: &mut BTreeMap<Position, CellKind>,
        dim: DimSize,
        position: Position,
        kind: CellKind,
    ) -> eyre::Result<()> {
        ensure!(
            dim.bound_on(position),
            "scaffold block {kind:?} at {position:?} is outside {dim:?}"
        );
        if let Some(previous) = cells.get(&position) {
            ensure!(
                *previous == kind,
                "scaffold collision at {position:?}: {previous:?} versus {kind:?}"
            );
        } else {
            cells.insert(position, kind);
        }
        Ok(())
    }
    fn dust(
        cells: &mut BTreeMap<Position, CellKind>,
        dim: DimSize,
        x: usize,
        y: usize,
        z: usize,
    ) -> eyre::Result<()> {
        ensure!(z > 0, "dust needs a support below it");
        put(cells, dim, Position(x, y, z - 1), CellKind::Solid)?;
        put(cells, dim, Position(x, y, z), CellKind::Dust)
    }

    let mut merge_x = vec![output_x];
    let mut entry_x = BTreeSet::new();
    for &(tap, inverted) in taps {
        let x1 = tap.0 + 1;
        put(&mut cells, dim, tap, CellKind::Solid)?;
        put(
            &mut cells,
            dim,
            Position(x1, start_y, 1),
            CellKind::Torch(Direction::West),
        )?;
        put(&mut cells, dim, Position(x1, start_y, 2), CellKind::Solid)?;
        let x = x1;
        let mut z = if inverted {
            3
        } else {
            put(
                &mut cells,
                dim,
                Position(x1, start_y, 3),
                CellKind::Torch(Direction::Bottom),
            )?;
            put(&mut cells, dim, Position(x1, start_y, 4), CellKind::Solid)?;
            5
        };
        let mut y = start_y;
        dust(&mut cells, dim, x, y, z)?;
        while z < 5 {
            y += 1;
            z += 1;
            dust(&mut cells, dim, x, y, z)?;
        }
        while y < start_y + 2 {
            y += 1;
            dust(&mut cells, dim, x, y, z)?;
        }
        put(
            &mut cells,
            dim,
            Position(x, start_y + 3, 4),
            CellKind::Solid,
        )?;
        put(
            &mut cells,
            dim,
            Position(x, start_y + 3, 5),
            CellKind::Repeater(Direction::South),
        )?;
        merge_x.push(x);
        entry_x.insert(x);
    }
    // Include the output column even when it lies beyond all input columns.
    let left = *merge_x.iter().min().unwrap();
    let right = *merge_x.iter().max().unwrap();
    for x in left..=right {
        put(
            &mut cells,
            dim,
            Position(x, start_y + 4, 4),
            CellKind::Solid,
        )?;
        // Sources feed actual dust entries, never a repeater side. All other
        // bus positions restore power toward the output, including adjacent
        // repeaters in a sparse pitch-three layout.
        let kind = if x == output_x || entry_x.contains(&x) {
            CellKind::Dust
        } else if x < output_x {
            CellKind::Repeater(Direction::West)
        } else {
            CellKind::Repeater(Direction::East)
        };
        put(&mut cells, dim, Position(x, start_y + 4, 5), kind)?;
    }
    put(
        &mut cells,
        dim,
        Position(output_x, start_y + 5, 4),
        CellKind::Torch(Direction::South),
    )?;
    dust(&mut cells, dim, output_x, start_y + 5, 3)?;
    dust(&mut cells, dim, output_x, start_y + 6, 2)?;
    dust(&mut cells, dim, output_x, start_y + 7, 1)?;
    for y in start_y + 8..dim.1 {
        put(&mut cells, dim, Position(output_x, y, 0), CellKind::Solid)?;
        put(
            &mut cells,
            dim,
            Position(output_x, y, 1),
            CellKind::Repeater(Direction::South),
        )?;
    }
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::transform::place_and_route::local_placer::exact::{
        ExactLocalPlacer, ExactOutcome, ExactPlacerConfig, NorNetlist,
    };
    use crate::world::simulator::Simulator;
    use crate::world::World;

    #[test]
    fn two_negative_taps_are_and_in_simulator_and_exact_model() -> eyre::Result<()> {
        check_two_negative_scaffold(false)
    }

    #[test]
    fn pitch_three_negative_taps_preserve_intermediate_pass_rail() -> eyre::Result<()> {
        check_two_negative_scaffold(true)
    }

    fn check_two_negative_scaffold(pass_rail: bool) -> eyre::Result<()> {
        let dim = DimSize(24, 11, 7);
        let output = Position(7, 10, 1);
        let mut inputs = vec![
            Position(9, 0, 1),
            Position(if pass_rail { 15 } else { 13 }, 0, 1),
        ];
        if pass_rail {
            inputs.push(Position(12, 0, 1));
        }
        let netlist = NorNetlist::from_text(if pass_rail {
            "na=NOR(a); nb=NOR(b); out=NOR(na,nb); pass=OR(c)"
        } else {
            "na=NOR(a); nb=NOR(b); out=NOR(na,nb)"
        })?;
        let placer = ExactLocalPlacer::from_netlist(netlist);
        let mut config = ExactPlacerConfig::new(dim)
            .with_input_site("a", inputs[0], Direction::North)
            .with_input_site("b", inputs[1], Direction::North)
            .with_output_sites("out", [output]);
        if pass_rail {
            config = config
                .with_input_site("c", inputs[2], Direction::North)
                .with_output_sites("pass", [Position(12, 10, 1)]);
        }
        for x in 0..dim.0 {
            for y in 0..dim.1 {
                for z in 0..dim.2 {
                    config.fixed_cells.insert(Position(x, y, z), CellKind::Air);
                }
            }
        }
        config.fixed_cells.extend(nor_template(
            &[
                (Position(inputs[0].0, 1, 1), true),
                (Position(inputs[1].0, 1, 1), true),
            ],
            7,
            1,
            dim,
        )?);
        if pass_rail {
            config
                .fixed_cells
                .insert(Position(12, 1, 1), CellKind::Solid);
            for y in 2..dim.1 {
                config
                    .fixed_cells
                    .insert(Position(12, y, 0), CellKind::Solid);
                config
                    .fixed_cells
                    .insert(Position(12, y, 1), CellKind::Repeater(Direction::South));
            }
            config.driving_outputs.insert("pass".into());
        }
        for &input in &inputs {
            config
                .fixed_cells
                .insert(input, CellKind::Switch(Direction::North));
        }
        let mut kinds = vec![CellKind::Air; dim.0 * dim.1 * dim.2];
        for (&p, &kind) in &config.fixed_cells {
            kinds[p.0 + dim.0 * (p.1 + dim.1 * p.2)] = kind;
        }
        let physical = super::super::verify::build_world(dim, &kinds);
        let world = World::from(&physical);
        let cases = 1usize << inputs.len();
        for from in 0..cases {
            for to in 0..cases {
                let mut sim =
                    Simulator::from_settled_with_limits_and_trace(&world, 4096, 2_000_000, 0)?;
                for value in [from, to] {
                    sim.advance_idle_cycles(61)?;
                    sim.drive_inputs_with_limits(
                        inputs
                            .iter()
                            .enumerate()
                            .map(|(i, &p)| (p, value & (1 << i) != 0))
                            .collect(),
                        4096,
                        2_000_000,
                    )?;
                    assert_eq!(
                        sim.world()[output].kind.is_powered(),
                        value & 3 == 3,
                        "physical AND mismatch: from={from},to={to},driving={value}"
                    );
                    if pass_rail {
                        assert_eq!(
                            sim.world()[Position(12, 10, 1)].kind.is_powered(),
                            value & 4 != 0,
                            "pass rail contaminated at {from}->{to}, driving {value}"
                        );
                    }
                    assert!(
                        !sim.world()
                            .iter_block()
                            .iter()
                            .any(|(p, _)| sim.is_torch_burned_out(*p)),
                        "physical torch burnout"
                    );
                }
            }
        }
        eprintln!(
            "two-negative scaffold pass_rail={pass_rail} physical static={cases}/ordered={} PASS",
            cases * cases
        );
        config.driving_outputs.insert("out".into());
        for (name, value) in [
            ("case_rank_levels", rsdsl::IValue::Int(8)),
            ("symmetry_breaking", rsdsl::IValue::Bool(true)),
            ("canonical_source_ranks", rsdsl::IValue::Bool(true)),
        ] {
            config.model_params.insert(name.into(), value);
        }
        config.stage_levels = 48;
        config.time_limit = Some(Duration::from_secs(20));
        let (outcome, stats) = placer.place(&config)?;
        eprintln!(
            "two-negative scaffold exact placed={} encode={:?} solve={:?} refinements={}",
            matches!(outcome, ExactOutcome::Placed(_)),
            stats.encode_time,
            stats.solve_time,
            stats.refinements
        );
        assert!(
            matches!(outcome, ExactOutcome::Placed(_)),
            "exact fixed scaffold failed: {outcome:?}"
        );
        assert_eq!(stats.refinements, 0);
        Ok(())
    }
}
