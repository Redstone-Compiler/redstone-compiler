use std::collections::BTreeMap;

use super::candidate::{
    generate_routable_module_candidates_with_progress_label, UnitCandidateConfig,
};
use super::ir::{LayoutCandidate, PhysicalPortDirection};
use crate::graph::logic::LogicGraph;
use crate::graph::Graph;
use crate::ir::{graph_from_routable_leaf, RoutableModule};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CombinationalMacroKind {
    Xor,
    HalfAdder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MacroCandidateSource {
    GeneratedAndVerified,
    ReusedVerifiedMacro,
    MonolithicFallback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MacroBinding {
    kind: CombinationalMacroKind,
    inputs: Vec<String>,
    outputs: BTreeMap<MacroOutputRole, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MacroOutputRole {
    Sum,
    Carry,
}

#[derive(Clone)]
struct MacroEntry {
    binding: MacroBinding,
    config: UnitCandidateConfig,
    candidates: Vec<LayoutCandidate>,
}

/// Per-compilation library of semantically verified combinational layouts.
///
/// The first recognized leaf is generated through the normal monolithic path,
/// including its exhaustive truth-table check. Later semantically equivalent
/// leaves reuse that physical result after deterministic port relabeling.
/// Unrecognized graphs keep using the monolithic generator.
#[derive(Default)]
pub(super) struct CombinationalMacroLibrary {
    entries: Vec<MacroEntry>,
}

impl CombinationalMacroLibrary {
    /// Cluster-composer seam. `generate_verified` must use the ordinary local
    /// candidate path, which truth-table checks combinational candidates before
    /// returning them.
    pub(super) fn resolve_graph_or_generate(
        &mut self,
        name: &str,
        graph: &Graph,
        ports: &[(String, PhysicalPortDirection)],
        config: &UnitCandidateConfig,
        generate_verified: impl FnOnce() -> eyre::Result<Vec<LayoutCandidate>>,
    ) -> eyre::Result<(Vec<LayoutCandidate>, MacroCandidateSource)> {
        let binding = recognize_macro_graph(graph)?;
        if let Some(binding) = &binding {
            validate_boundary_ports(binding, ports)?;
        }
        self.resolve_recognized(name, binding, config, generate_verified)
    }

    pub(super) fn candidates_for(
        &mut self,
        module: &RoutableModule,
        config: &UnitCandidateConfig,
        progress_label: Option<&str>,
    ) -> eyre::Result<(Vec<LayoutCandidate>, MacroCandidateSource)> {
        self.resolve_with(module, config, || {
            generate_routable_module_candidates_with_progress_label(module, config, progress_label)
        })
    }

    fn resolve_with(
        &mut self,
        module: &RoutableModule,
        config: &UnitCandidateConfig,
        generate_verified: impl FnOnce() -> eyre::Result<Vec<LayoutCandidate>>,
    ) -> eyre::Result<(Vec<LayoutCandidate>, MacroCandidateSource)> {
        let binding = recognize_macro(module)?;
        self.resolve_recognized(&module.name, binding, config, generate_verified)
    }

    fn resolve_recognized(
        &mut self,
        module_name: &str,
        binding: Option<MacroBinding>,
        config: &UnitCandidateConfig,
        generate_verified: impl FnOnce() -> eyre::Result<Vec<LayoutCandidate>>,
    ) -> eyre::Result<(Vec<LayoutCandidate>, MacroCandidateSource)> {
        let Some(binding) = binding else {
            return Ok((
                generate_verified()?,
                MacroCandidateSource::MonolithicFallback,
            ));
        };
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.binding.kind == binding.kind && entry.config == *config)
        {
            return Ok((
                relabel_candidates(&entry.candidates, &entry.binding, module_name, &binding)?,
                MacroCandidateSource::ReusedVerifiedMacro,
            ));
        }

        // This callback is deliberately the existing local candidate path:
        // candidates enter the library only after its truth-table verifier has
        // accepted every combinational layout.
        let candidates = generate_verified()?;
        if !candidates.is_empty() {
            self.entries.push(MacroEntry {
                binding,
                config: config.clone(),
                candidates: candidates.clone(),
            });
        }
        Ok((candidates, MacroCandidateSource::GeneratedAndVerified))
    }
}

fn recognize_macro(module: &RoutableModule) -> eyre::Result<Option<MacroBinding>> {
    recognize_macro_graph(&graph_from_routable_leaf(module)?)
}

fn recognize_macro_graph(graph: &Graph) -> eyre::Result<Option<MacroBinding>> {
    let graph = LogicGraph {
        graph: graph.clone(),
    };
    let table = graph.truth_table()?;
    if table.input_names.len() != 2 {
        return Ok(None);
    }
    let xor = [false, true, true, false];
    let and = [false, false, false, true];
    let mut outputs = BTreeMap::new();
    for (name, values) in &table.output_tables {
        let role = if values.as_slice() == xor {
            MacroOutputRole::Sum
        } else if values.as_slice() == and {
            MacroOutputRole::Carry
        } else {
            return Ok(None);
        };
        if outputs.insert(role, name.clone()).is_some() {
            return Ok(None);
        }
    }
    let kind = match outputs.keys().copied().collect::<Vec<_>>().as_slice() {
        [MacroOutputRole::Sum] => CombinationalMacroKind::Xor,
        [MacroOutputRole::Sum, MacroOutputRole::Carry]
        | [MacroOutputRole::Carry, MacroOutputRole::Sum] => CombinationalMacroKind::HalfAdder,
        _ => return Ok(None),
    };
    Ok(Some(MacroBinding {
        kind,
        inputs: table.input_names,
        outputs,
    }))
}

fn validate_boundary_ports(
    binding: &MacroBinding,
    ports: &[(String, PhysicalPortDirection)],
) -> eyre::Result<()> {
    for input in &binding.inputs {
        if !ports
            .iter()
            .any(|(name, direction)| name == input && *direction == PhysicalPortDirection::Input)
        {
            eyre::bail!("macro cluster is missing input boundary port `{input}`");
        }
    }
    for output in binding.outputs.values() {
        if !ports
            .iter()
            .any(|(name, direction)| name == output && *direction == PhysicalPortDirection::Output)
        {
            eyre::bail!("macro cluster is missing output boundary port `{output}`");
        }
    }
    Ok(())
}

fn relabel_candidates(
    candidates: &[LayoutCandidate],
    old: &MacroBinding,
    module_name: &str,
    new: &MacroBinding,
) -> eyre::Result<Vec<LayoutCandidate>> {
    let input_names = old
        .inputs
        .iter()
        .cloned()
        .zip(new.inputs.iter().cloned())
        .collect::<BTreeMap<_, _>>();
    let output_names =
        old.outputs
            .iter()
            .map(|(role, old_name)| {
                Ok((
                    old_name.clone(),
                    new.outputs.get(role).cloned().ok_or_else(|| {
                        eyre::eyre!("macro output role disappeared during relabel")
                    })?,
                ))
            })
            .collect::<eyre::Result<BTreeMap<_, _>>>()?;
    candidates
        .iter()
        .cloned()
        .map(|mut candidate| {
            candidate.module_name = module_name.to_owned();
            for port in &mut candidate.ports {
                let names = match port.direction {
                    PhysicalPortDirection::Input => &input_names,
                    PhysicalPortDirection::Output => &output_names,
                };
                port.name = names.get(&port.name).cloned().ok_or_else(|| {
                    eyre::eyre!(
                        "verified macro candidate has unexpected port `{}`",
                        port.name
                    )
                })?;
            }
            candidate
                .ports
                .sort_by(|left, right| left.name.cmp(&right.name));
            Ok(candidate)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::{predefined_logics, LogicGraph};
    use crate::world::block::{Block, BlockKind};
    use crate::world::position::{DimSize, Position};
    use crate::world::World3D;

    fn test_candidate(module: &str, inputs: [&str; 2], output: &str) -> LayoutCandidate {
        let mut world = World3D::new(DimSize(1, 1, 1));
        world[Position(0, 0, 0)] = Block {
            kind: BlockKind::RedstoneBlock,
            ..Default::default()
        };
        let ports = inputs
            .into_iter()
            .map(|name| super::super::ir::PhysicalPort {
                name: name.to_owned(),
                direction: PhysicalPortDirection::Input,
                position: Position(0, 0, 0),
                route_position: None,
                access_points: vec![Position(0, 0, 0)],
                connection: super::super::ir::PortConnection::Direct,
            })
            .chain(std::iter::once(super::super::ir::PhysicalPort {
                name: output.to_owned(),
                direction: PhysicalPortDirection::Output,
                position: Position(0, 0, 0),
                route_position: None,
                access_points: vec![Position(0, 0, 0)],
                connection: super::super::ir::PortConnection::Direct,
            }))
            .collect();
        LayoutCandidate::from_world(module.to_owned(), world, ports).expect("candidate")
    }

    #[test]
    fn truth_tables_recognize_xor_and_half_adder_macros() -> eyre::Result<()> {
        let xor = LogicGraph::from_stmt("a^b", "sum")?.prepare_place()?;
        let half_adder = predefined_logics::half_adder_graph()?;

        assert_eq!(
            recognize_macro_graph(&xor.graph)?.map(|binding| binding.kind),
            Some(CombinationalMacroKind::Xor)
        );
        assert_eq!(
            recognize_macro_graph(&half_adder.graph)?.map(|binding| binding.kind),
            Some(CombinationalMacroKind::HalfAdder)
        );
        Ok(())
    }

    #[test]
    fn verified_xor_macro_is_reused_with_boundary_port_relabeling() -> eyre::Result<()> {
        let first = LogicGraph::from_stmt("a^b", "sum")?.prepare_place()?;
        let second = LogicGraph::from_stmt("left^right", "result")?.prepare_place()?;
        let config = UnitCandidateConfig::default();
        let mut library = CombinationalMacroLibrary::default();
        let first_ports = vec![
            ("a".to_owned(), PhysicalPortDirection::Input),
            ("b".to_owned(), PhysicalPortDirection::Input),
            ("sum".to_owned(), PhysicalPortDirection::Output),
        ];
        let (_, first_source) = library.resolve_graph_or_generate(
            "xor-first",
            &first.graph,
            &first_ports,
            &config,
            || Ok(vec![test_candidate("xor-first", ["a", "b"], "sum")]),
        )?;
        assert_eq!(first_source, MacroCandidateSource::GeneratedAndVerified);

        let second_ports = vec![
            ("left".to_owned(), PhysicalPortDirection::Input),
            ("right".to_owned(), PhysicalPortDirection::Input),
            ("result".to_owned(), PhysicalPortDirection::Output),
        ];
        let (reused, second_source) = library.resolve_graph_or_generate(
            "xor-second",
            &second.graph,
            &second_ports,
            &config,
            || panic!("equivalent XOR must reuse the verified macro"),
        )?;

        assert_eq!(second_source, MacroCandidateSource::ReusedVerifiedMacro);
        assert_eq!(reused[0].module_name, "xor-second");
        assert_eq!(
            reused[0]
                .ports
                .iter()
                .map(|port| port.name.as_str())
                .collect::<Vec<_>>(),
            vec!["left", "result", "right"]
        );
        Ok(())
    }

    #[test]
    fn unknown_graph_uses_monolithic_fallback() -> eyre::Result<()> {
        let and = LogicGraph::from_stmt("a&b", "out")?.prepare_place()?;
        let mut library = CombinationalMacroLibrary::default();
        let (_, source) = library.resolve_graph_or_generate(
            "and",
            &and.graph,
            &[
                ("a".to_owned(), PhysicalPortDirection::Input),
                ("b".to_owned(), PhysicalPortDirection::Input),
                ("out".to_owned(), PhysicalPortDirection::Output),
            ],
            &UnitCandidateConfig::default(),
            || Ok(vec![test_candidate("and", ["a", "b"], "out")]),
        )?;
        assert_eq!(source, MacroCandidateSource::MonolithicFallback);
        Ok(())
    }
}
