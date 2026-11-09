//! Truth-recognized, verified deterministic n-ary cells. No module-name dispatch.
use eyre::ensure;

use super::{LayoutCandidate, PhysicalPort, PhysicalPortDirection, PortConnection};
use crate::graph::logic::LogicGraph;
use crate::graph::GraphNodeKind;
use crate::nbt::{NBTRoot, ToNBT};
use crate::output::{OutputEndpoint, PlacedWorld};
use crate::transform::place_and_route::placed_node::PlacedNode;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::simulator::{Simulator, MANUAL_INPUT_IDLE_CYCLES};
use crate::world::{World, World3D};

/// The one exceptional assignment and its output value. Each input whose
/// pattern bit is one is inverted before the OR bus. The bus is therefore
/// zero exactly at `pattern`; an output inverter implements equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Function {
    pattern: usize,
    equal: bool,
}

fn expected(function: Function, _count: usize, mask: usize) -> bool {
    (mask == function.pattern) == function.equal
}

// Recognition uses the entire truth signature, never a module/name hint.
fn recognize(graph: &LogicGraph) -> eyre::Result<Option<(Function, Vec<String>, String)>> {
    // Reject unsupported sizes before truth-table construction can explode.
    let count = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.kind, GraphNodeKind::Input(_)))
        .count();
    if !(2..=8).contains(&count)
        || graph
            .nodes
            .iter()
            .any(|node| matches!(node.kind, GraphNodeKind::Sequential(_)))
    {
        return Ok(None);
    }
    let truth = graph.truth_table()?;
    if !(2..=8).contains(&truth.input_names.len()) || truth.output_tables.len() != 1 {
        return Ok(None);
    }
    let (output, values) = truth.output_tables.iter().next().unwrap();
    let true_patterns = values
        .iter()
        .enumerate()
        .filter_map(|(mask, &v)| v.then_some(mask))
        .collect::<Vec<_>>();
    let false_patterns = values
        .iter()
        .enumerate()
        .filter_map(|(mask, &v)| (!v).then_some(mask))
        .collect::<Vec<_>>();
    let function = if true_patterns.len() == 1 {
        Some(Function {
            pattern: true_patterns[0],
            equal: true,
        })
    } else if false_patterns.len() == 1 {
        Some(Function {
            pattern: false_patterns[0],
            equal: false,
        })
    } else {
        None
    };
    Ok(function.map(|function| (function, truth.input_names, output.clone())))
}

fn put(world: &mut World3D, node: PlacedNode) {
    world[node.position.down().unwrap()] =
        PlacedNode::new_cobble(node.position.down().unwrap()).block;
    world[node.position] = node.block;
}

fn recipe(function: Function, names: &[String], output: &str, pitch: usize) -> PlacedWorld {
    debug_assert!(pitch >= 3 && pitch % 3 == 0);
    let last = 2 + pitch * (names.len() - 1);
    let mut world = World3D::new(DimSize(last + 7, 8, 3));
    let mut inputs = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let x = 2 + pitch * index;
        let input = Position(x, 1, 1);
        inputs.push(OutputEndpoint::new(name.clone(), input));
        put(&mut world, PlacedNode::new_redstone(input));
        put(
            &mut world,
            PlacedNode::new_repeater(Position(x, 2, 1), Direction::South),
        );
        if function.pattern & (1 << index) != 0 {
            put(&mut world, PlacedNode::new_cobble(Position(x, 3, 1)));
            put(
                &mut world,
                PlacedNode::new(
                    Position(x, 4, 1),
                    Block {
                        kind: BlockKind::Torch { is_on: true },
                        direction: Direction::South,
                    },
                ),
            );
            put(
                &mut world,
                PlacedNode::new_repeater(Position(x, 5, 1), Direction::South),
            );
        } else {
            for y in 3..=5 {
                put(&mut world, PlacedNode::new_redstone(Position(x, y, 1)));
            }
        }
    }
    // Each stage joins one input and forwards accumulated OR to the right.
    // A repeater between stages prevents strength exhaustion and backfeed.
    for x in 2..=last {
        let p = Position(x, 6, 1);
        if x > 2 && (x - 2) % 3 == 2 {
            put(&mut world, PlacedNode::new_repeater(p, Direction::West));
        } else {
            put(&mut world, PlacedNode::new_redstone(p));
        }
    }
    put(
        &mut world,
        PlacedNode::new_repeater(Position(last + 1, 6, 1), Direction::West),
    );
    let endpoint = if !function.equal {
        let p = Position(last + 2, 6, 1);
        put(&mut world, PlacedNode::new_redstone(p));
        p
    } else {
        put(&mut world, PlacedNode::new_cobble(Position(last + 2, 6, 1)));
        put(
            &mut world,
            PlacedNode::new(
                Position(last + 3, 6, 1),
                Block {
                    kind: BlockKind::Torch { is_on: true },
                    direction: Direction::West,
                },
            ),
        );
        put(
            &mut world,
            PlacedNode::new_repeater(Position(last + 4, 6, 1), Direction::West),
        );
        let p = Position(last + 5, 6, 1);
        put(&mut world, PlacedNode::new_redstone(p));
        p
    };
    world.initialize_redstone_states();
    PlacedWorld {
        world,
        inputs,
        outputs: vec![OutputEndpoint::new(output.to_owned(), endpoint)],
    }
}

fn stagger_inputs(cell: PlacedWorld) -> PlacedWorld {
    let padding = 3 * (cell.inputs.len() - 1);
    let shift = |p: Position| Position(p.0, p.1 + padding, p.2);
    let mut world = World3D::new(DimSize(
        cell.world.size.0,
        cell.world.size.1 + padding,
        cell.world.size.2,
    ));
    for (position, block) in cell.world.iter_block() {
        world[shift(position)] = block;
    }
    let inputs = cell
        .inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let original = shift(input.position());
            let tip = Position(original.0, original.1 - 3 * index, original.2);
            for y in tip.1..original.1 {
                let p = Position(tip.0, y, tip.2);
                put(
                    &mut world,
                    if (y - tip.1) % 2 == 1 {
                        PlacedNode::new_repeater(p, Direction::South)
                    } else {
                        PlacedNode::new_redstone(p)
                    },
                );
            }
            OutputEndpoint::new(input.name.clone(), tip)
        })
        .collect();
    let outputs = cell
        .outputs
        .iter()
        .map(|p| OutputEndpoint::new(p.name.clone(), shift(p.position())))
        .collect();
    world.initialize_redstone_states();
    PlacedWorld {
        world,
        inputs,
        outputs,
    }
}

fn drive(
    sim: &mut Simulator,
    cell: &PlacedWorld,
    function: Function,
    mask: usize,
) -> eyre::Result<()> {
    sim.advance_idle_cycles(MANUAL_INPUT_IDLE_CYCLES)?;
    sim.drive_inputs_with_limits(
        cell.inputs
            .iter()
            .enumerate()
            .map(|(bit, p)| (p.position(), mask & (1 << bit) != 0))
            .collect(),
        1024,
        200_000,
    )?;
    ensure!(
        sim.world()[cell.outputs[0].position()].kind.is_powered()
            == expected(function, cell.inputs.len(), mask),
        "output mismatch {function:?}/{} mask={mask}",
        cell.inputs.len()
    );
    if expected(function, cell.inputs.len(), mask) {
        ensure!(
            matches!(
                sim.world()[cell.outputs[0].position()].kind,
                BlockKind::Redstone { strength: 15, .. }
            ),
            "restored output did not supply strength 15: {function:?} mask={mask}"
        );
    }
    for (bit, input) in cell.inputs.iter().enumerate() {
        ensure!(
            sim.world()[input.position()].kind.is_powered() == (mask & (1 << bit) != 0),
            "input backfeed {function:?}/{} mask={mask} bit={bit}",
            cell.inputs.len()
        );
    }
    ensure!(
        sim.world()
            .iter_block()
            .iter()
            .all(|(p, _)| !sim.is_torch_burned_out(*p)),
        "torch burnout"
    );
    Ok(())
}

/// Build a 2..8-input single-minterm/maxterm cell from its complete truth table.
/// Every returned
/// cell passed NBT roundtrip, all assignments and bounded transition checks.
pub fn build_verified_candidate(
    graph: &LogicGraph,
    module_name: &str,
) -> eyre::Result<Option<LayoutCandidate>> {
    build_verified_candidate_with_stagger(graph, module_name, false)
}

/// Optional deterministic staggered input boundary; all verification is retained.
pub fn build_verified_candidate_with_stagger(
    graph: &LogicGraph,
    module_name: &str,
    stagger: bool,
) -> eyre::Result<Option<LayoutCandidate>> {
    let Some((function, inputs, output)) = recognize(graph)? else {
        return Ok(None);
    };
    let count = inputs.len();
    // Wider staggered boundaries leave a three-block corridor between input
    // columns. Restore bus strength every third block regardless of pitch.
    let cell = recipe(function, &inputs, &output, if stagger { 6 } else { 3 });
    let cell = if stagger { stagger_inputs(cell) } else { cell };
    let bytes = fastnbt::to_bytes(&cell.world.to_nbt())?;
    let world: World = NBTRoot::from_nbt_bytes(&bytes)?.to_world();
    for mask in 0..(1 << count) {
        let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 1024, 200_000, 0)?;
        drive(&mut sim, &cell, function, mask)?;
    }
    let mut sim = Simulator::from_settled_with_limits_and_trace(&world, 1024, 200_000, 0)?;
    let full = (1 << count) - 1;
    let mut sequence = vec![0, full, 0, full, 0];
    for bit in 0..count {
        sequence.extend([1 << bit, full ^ (1 << bit), 0]);
        // Exercise both edges of the exceptional assignment, even when the
        // bounded random walk never happens to reach this mixed-polarity mask.
        sequence.extend([
            function.pattern,
            function.pattern ^ (1 << bit),
            function.pattern,
        ]);
    }
    let mut seed = 42u32;
    for _ in 0..64 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        sequence.push(seed as usize & full);
    }
    for mask in sequence {
        drive(&mut sim, &cell, function, mask)?;
    }
    let ports = cell
        .inputs
        .iter()
        .map(|p| (p, PhysicalPortDirection::Input))
        .chain(
            cell.outputs
                .iter()
                .map(|p| (p, PhysicalPortDirection::Output)),
        )
        .map(|(p, direction)| PhysicalPort {
            name: p.name.clone(),
            direction: direction.clone(),
            position: p.position(),
            route_position: Some(p.position()),
            access_points: vec![p.position()],
            connection: if direction == PhysicalPortDirection::Output {
                PortConnection::RestoredDirect
            } else {
                PortConnection::Direct
            },
        })
        .collect();
    let mut candidate = LayoutCandidate::from_world(module_name.to_owned(), cell.world, ports)?;
    super::io_diodes::reserve_driver_headroom(&mut candidate);
    Ok(Some(candidate))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staggered_patterns_preserve_truth_transitions_and_weak_inputs() -> eyre::Result<()> {
        for (count, patterns) in [
            (4, (0..16).collect::<Vec<_>>()),
            (8, vec![0, 0x55, 0xaa, 0xff]),
        ] {
            for pattern in patterns {
                let terms = (0..count)
                    .map(|bit| {
                        if pattern & (1 << bit) != 0 {
                            format!("port_{bit}")
                        } else {
                            format!("~port_{bit}")
                        }
                    })
                    .collect::<Vec<_>>();
                for equal in [true, false] {
                    let expr = terms.join("&");
                    let expr = if equal { expr } else { format!("~({expr})") };
                    let graph = LogicGraph::from_stmt(&expr, "result")?;
                    let candidate =
                        build_verified_candidate_with_stagger(&graph, "arbitrary", true)?.unwrap();
                    let inputs = candidate
                        .ports
                        .iter()
                        .filter(|p| p.direction == PhysicalPortDirection::Input)
                        .collect::<Vec<_>>();
                    for (i, p) in inputs.iter().enumerate() {
                        assert_eq!(p.position.1, 1 + 3 * (count - 1 - i));
                        assert_eq!(p.position.0, 2 + 6 * i);
                        assert!(candidate.blocked_cells.contains(&p.position.up()));
                        assert!(candidate.world[p.position.up()].kind.is_air());
                    }
                }
            }
        }
        // Feed each tip through fourteen extra dust steps: it receives strength
        // one, so its immediately following repeater must restore the signal.
        let function = Function {
            pattern: 5,
            equal: true,
        };
        let names = (0..4).map(|i| format!("weak_{i}")).collect::<Vec<_>>();
        let cell = stagger_inputs(recipe(function, &names, "out", 6));
        let shift = |p: Position| Position(p.0, p.1 + 14, p.2);
        let mut world = World3D::new(DimSize(
            cell.world.size.0,
            cell.world.size.1 + 14,
            cell.world.size.2,
        ));
        for (p, b) in cell.world.iter_block() {
            world[shift(p)] = b;
        }
        let tips = cell
            .inputs
            .iter()
            .map(|p| shift(p.position()))
            .collect::<Vec<_>>();
        let inputs = cell
            .inputs
            .iter()
            .map(|p| {
                let tip = shift(p.position());
                let source = Position(tip.0, tip.1 - 14, tip.2);
                for y in source.1..tip.1 {
                    put(
                        &mut world,
                        PlacedNode::new_redstone(Position(tip.0, y, tip.2)),
                    );
                }
                OutputEndpoint::new(p.name.clone(), source)
            })
            .collect();
        world.initialize_redstone_states();
        let weak = PlacedWorld {
            world,
            inputs,
            outputs: cell
                .outputs
                .iter()
                .map(|p| OutputEndpoint::new(p.name.clone(), shift(p.position())))
                .collect(),
        };
        let mut sim = Simulator::from_settled_with_limits_and_trace(
            &World::from(&weak.world),
            1024,
            200_000,
            0,
        )?;
        for mask in (0..16).chain((0..16).rev()) {
            drive(&mut sim, &weak, function, mask)?;
            for (bit, tip) in tips.iter().enumerate() {
                let BlockKind::Redstone { strength, .. } = sim.world()[*tip].kind else {
                    panic!("tip not dust")
                };
                assert_eq!(strength, usize::from(mask & (1 << bit) != 0));
            }
        }
        Ok(())
    }

    #[test]
    fn literal_patterns_verify_all_four_input_masks_and_wide_examples() -> eyre::Result<()> {
        for (count, patterns) in [
            (4, (0..16).collect::<Vec<_>>()),
            (8, vec![0, 1, 0x55, 0xaa, 0x81, 0xff]),
        ] {
            for pattern in patterns {
                let terms = (0..count)
                    .map(|bit| {
                        let name = format!("arbitrary_operand_{bit}");
                        if pattern & (1 << bit) != 0 {
                            name
                        } else {
                            format!("~{name}")
                        }
                    })
                    .collect::<Vec<_>>();
                for equal in [true, false] {
                    let conjunction = terms.join("&");
                    let expression = if equal {
                        conjunction
                    } else {
                        format!("~({conjunction})")
                    };
                    let graph = LogicGraph::from_stmt(&expression, "unrelated_output")?;
                    let recognized = recognize(&graph)?.unwrap();
                    assert_eq!(recognized.0, Function { pattern, equal });
                    // The factory checks every static input and bounded walking
                    // transitions after NBT roundtrip, including input isolation.
                    let candidate = build_verified_candidate(&graph, "no_name_hint")?.unwrap();
                    assert_eq!(candidate.ports.len(), count + 1);
                    assert_eq!(candidate.module_name, "no_name_hint");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn dispatcher_uses_recipe_with_named_ports_and_checks_bbox() -> eyre::Result<()> {
        use super::super::{
            candidate_satisfies_local_cell_contract,
            generate_routable_module_candidates_with_progress_label, UnitCandidateConfig,
        };
        use crate::ir::{
            LocalCellContractSpec, LogicalDesign, RoutableModuleBody, RoutablePortDirection,
        };
        let logical = LogicalDesign::from_verilog_source("module arbitrary(left,middle,right,result); input left,middle,right; output result; assign result=(left&middle)&right; endmodule")?;
        let routable = logical.lower_to_routable_with_associative_cells(8)?;
        let module = routable
            .modules
            .iter()
            .find(|m| {
                matches!(m.body, RoutableModuleBody::Leaf { .. })
                    && m.ports
                        .iter()
                        .filter(|p| p.direction == RoutablePortDirection::Input)
                        .count()
                        == 3
            })
            .unwrap();
        let config = UnitCandidateConfig {
            verified_nary_cells: true,
            reserve_combinational_io_diodes: true,
            ..Default::default()
        };
        let candidates =
            generate_routable_module_candidates_with_progress_label(module, &config, None)?;
        assert_eq!(candidates.len(), 1);
        let mut candidate = candidates[0].clone();
        // Recipe pins are already isolated; generic I/O reservation must not add another stage.
        assert_eq!(candidate.world.iter_block().len(), 54);
        assert!(candidate.ports.iter().all(|p| p.connection
            == if p.direction == PhysicalPortDirection::Output {
                PortConnection::RestoredDirect
            } else {
                PortConnection::Direct
            }));
        assert!(module
            .ports
            .iter()
            .all(|port| candidate.ports.iter().any(|p| p.name == port.name)));
        let contract = LocalCellContractSpec {
            max_bbox: Some([1, 1, 1]),
            ..Default::default()
        };
        assert!(!candidate_satisfies_local_cell_contract(
            &mut candidate,
            &contract
        ));
        Ok(())
    }

    #[test]
    fn dispatcher_two_input_recipes_cover_every_transition() -> eyre::Result<()> {
        use super::super::{
            generate_routable_module_candidates_with_progress_label, UnitCandidateConfig,
        };
        use crate::ir::{LogicalDesign, RoutableModuleBody, RoutablePortDirection};
        for (operator, function, block_count) in [
            (
                "&",
                Function {
                    pattern: 3,
                    equal: true,
                },
                38,
            ),
            (
                "|",
                Function {
                    pattern: 0,
                    equal: false,
                },
                32,
            ),
        ] {
            let source = format!("module arbitrary(left,right,result); input left,right; output result; assign result=left{operator}right; endmodule");
            let routable = LogicalDesign::from_verilog_source(&source)?
                .lower_to_routable_with_associative_cells(8)?;
            let module = routable
                .modules
                .iter()
                .find(|m| {
                    matches!(m.body, RoutableModuleBody::Leaf { .. })
                        && m.ports
                            .iter()
                            .filter(|p| p.direction == RoutablePortDirection::Input)
                            .count()
                            == 2
                })
                .unwrap();
            let config = UnitCandidateConfig {
                verified_nary_cells: true,
                reserve_combinational_io_diodes: true,
                ..Default::default()
            };
            let candidates =
                generate_routable_module_candidates_with_progress_label(module, &config, None)?;
            assert_eq!(candidates.len(), 1);
            let candidate = &candidates[0];
            assert_eq!(candidate.world.iter_block().len(), block_count);
            let cell = PlacedWorld {
                world: candidate.world.clone(),
                inputs: candidate
                    .ports
                    .iter()
                    .filter(|p| p.direction == PhysicalPortDirection::Input)
                    .map(|p| OutputEndpoint::new(p.name.clone(), p.position))
                    .collect(),
                outputs: candidate
                    .ports
                    .iter()
                    .filter(|p| p.direction == PhysicalPortDirection::Output)
                    .map(|p| OutputEndpoint::new(p.name.clone(), p.position))
                    .collect(),
            };
            for from in 0..4 {
                for to in 0..4 {
                    let mut sim = Simulator::from_settled_with_limits_and_trace(
                        &World::from(&cell.world),
                        1024,
                        200_000,
                        0,
                    )?;
                    drive(&mut sim, &cell, function, from)?;
                    drive(&mut sim, &cell, function, to)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn recipes_verify_all_arities_functions_and_arbitrary_names() -> eyre::Result<()> {
        for count in 2..=8 {
            let names = (0..count)
                .map(|i| format!("operand_{i}"))
                .collect::<Vec<_>>();
            for expression in [
                names.join("|"),
                format!("~({})", names.join("|")),
                names.join("&"),
            ] {
                let graph = LogicGraph::from_stmt(&expression, "renamed_result")?;
                let candidate = build_verified_candidate(&graph, "arbitrary_definition")?.unwrap();
                assert_eq!(candidate.module_name, "arbitrary_definition");
                assert_eq!(
                    candidate
                        .ports
                        .iter()
                        .filter(|p| p.direction == PhysicalPortDirection::Input)
                        .count(),
                    count
                );
                assert!(candidate.ports.iter().all(|p| p.connection
                    == if p.direction == PhysicalPortDirection::Output {
                        PortConnection::RestoredDirect
                    } else {
                        PortConnection::Direct
                    }));
                for port in &candidate.ports {
                    assert!(candidate.blocked_cells.contains(&port.position.up()));
                    assert!(candidate.world[port.position.up()].kind.is_air());
                }
                assert_eq!(candidate.cost.bbox_height, 3);

                assert!(candidate
                    .ports
                    .iter()
                    .any(|p| p.name == "renamed_result"
                        && p.direction == PhysicalPortDirection::Output));
            }
        }
        for unsupported in ["a^b", "(a&b)|(c&d)", "~a", "a|b|c|d|e|f|g|h|i"] {
            assert!(build_verified_candidate(
                &LogicGraph::from_stmt(unsupported, "y")?,
                "not_magic"
            )?
            .is_none());
        }
        Ok(())
    }
}
