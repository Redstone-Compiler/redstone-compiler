use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub type DebugLocationId = usize;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IrDebugInfo {
    pub locations: Vec<DebugLocation>,
    pub entities: BTreeMap<String, DebugLocationId>,
    pub relations: Vec<DebugRelation>,
}

impl PartialEq for IrDebugInfo {
    fn eq(&self, _other: &Self) -> bool {
        // Debug locations do not participate in circuit identity.
        true
    }
}

impl Eq for IrDebugInfo {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DebugLocation {
    Source {
        file: String,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
    },
    Derived {
        label: String,
        parent: DebugLocationId,
    },
    Fused {
        parents: Vec<DebugLocationId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrSourceMap {
    pub format: String,
    pub locations: Vec<DebugLocation>,
    pub entities: BTreeMap<String, DebugEntity>,
    pub relations: Vec<DebugRelation>,
    pub documents: BTreeMap<String, Vec<DebugRange>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugEntity {
    pub location: DebugLocationId,
    pub kind: DebugEntityKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_scope: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugEntityKind {
    Module,
    Port,
    Net,
    Instance,
    Cell,
    Node,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugRelation {
    pub kind: DebugRelationKind,
    pub from: String,
    pub to: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugRelationKind {
    Instantiates,
    CanonicalizedTo,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugRange {
    pub entity: String,
    pub start_line: usize,
    pub end_line: usize,
    pub location: DebugLocationId,
}

impl IrDebugInfo {
    pub fn source(
        &mut self,
        file: impl Into<String>,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
    ) -> DebugLocationId {
        self.intern(DebugLocation::Source {
            file: file.into(),
            start_line,
            start_column,
            end_line,
            end_column,
        })
    }

    pub fn derived(
        &mut self,
        label: impl Into<String>,
        parent: DebugLocationId,
    ) -> DebugLocationId {
        self.intern(DebugLocation::Derived {
            label: label.into(),
            parent,
        })
    }

    pub fn bind(&mut self, entity: impl Into<String>, location: DebugLocationId) {
        self.entities.insert(entity.into(), location);
    }

    pub fn get(&self, entity: &str) -> Option<DebugLocationId> {
        self.entities.get(entity).copied()
    }

    pub fn relate(
        &mut self,
        kind: DebugRelationKind,
        from: impl Into<String>,
        to: impl Into<String>,
    ) {
        let relation = DebugRelation {
            kind,
            from: from.into(),
            to: to.into(),
        };
        if !self.relations.contains(&relation) {
            self.relations.push(relation);
        }
    }

    fn intern(&mut self, location: DebugLocation) -> DebugLocationId {
        if let Some(index) = self
            .locations
            .iter()
            .position(|candidate| candidate == &location)
        {
            return index;
        }
        let id = self.locations.len();
        self.locations.push(location);
        id
    }

    pub fn source_map(
        &self,
        generated_documents: impl IntoIterator<Item = (String, Vec<DebugRange>)>,
    ) -> IrSourceMap {
        let mut documents = generated_documents.into_iter().collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        let mut covered_sources = BTreeSet::new();
        for (entity, location) in &self.entities {
            let Some(source) = direct_source_parent(&self.locations, *location) else {
                continue;
            };
            let DebugLocation::Source {
                file,
                start_line,
                end_line,
                ..
            } = &self.locations[source]
            else {
                unreachable!()
            };
            covered_sources.insert(source);
            if seen.insert((file.clone(), *start_line, *end_line, entity.clone())) {
                documents.entry(file.clone()).or_default().push(DebugRange {
                    entity: entity.clone(),
                    start_line: *start_line,
                    end_line: *end_line,
                    location: source,
                });
            }
        }
        for (location, candidate) in self.locations.iter().enumerate() {
            let DebugLocation::Source {
                file,
                start_line,
                end_line,
                ..
            } = candidate
            else {
                continue;
            };
            if covered_sources.contains(&location) {
                continue;
            }
            if seen.insert((
                file.clone(),
                *start_line,
                *end_line,
                format!("source/{location}"),
            )) {
                documents.entry(file.clone()).or_default().push(DebugRange {
                    entity: format!("source/{location}"),
                    start_line: *start_line,
                    end_line: *end_line,
                    location,
                });
            }
        }
        for ranges in documents.values_mut() {
            ranges.sort_by_key(|range| (range.start_line, range.end_line, range.entity.clone()));
        }
        IrSourceMap {
            format: "redstone-compiler.source-map.v1".to_owned(),
            locations: self.locations.clone(),
            entities: self
                .entities
                .iter()
                .map(|(entity, location)| (entity.clone(), debug_entity(entity, *location)))
                .collect(),
            relations: self.relations.clone(),
            documents,
        }
    }
}

fn direct_source_parent(
    locations: &[DebugLocation],
    location: DebugLocationId,
) -> Option<DebugLocationId> {
    let DebugLocation::Derived { parent, .. } = locations.get(location)? else {
        return None;
    };
    matches!(locations.get(*parent), Some(DebugLocation::Source { .. })).then_some(*parent)
}

fn debug_entity(entity: &str, location: DebugLocationId) -> DebugEntity {
    let parts = entity.split('/').collect::<Vec<_>>();
    let kind = match parts.get(2).copied() {
        Some("module") => DebugEntityKind::Module,
        Some("port") => DebugEntityKind::Port,
        Some("net") => DebugEntityKind::Net,
        Some("instance") => DebugEntityKind::Instance,
        Some("cell") => DebugEntityKind::Cell,
        Some("node") => DebugEntityKind::Node,
        _ => DebugEntityKind::Other,
    };
    let parent_scope = (kind != DebugEntityKind::Module && parts.len() >= 4)
        .then(|| format!("{}/{}/module/{}", parts[0], parts[1], parts[1]));
    DebugEntity {
        location,
        kind,
        parent_scope,
    }
}

pub fn logical_entity(module: &str, kind: &str, name: &str) -> String {
    format!("logical/{module}/{kind}/{name}")
}

pub fn routable_entity(module: &str, kind: &str, name: &str) -> String {
    format!("routable/{module}/{kind}/{name}")
}

pub fn build_source_map(
    logical: &super::LogicalDesign,
    logical_text: &str,
    routable: &super::RoutableDesign,
    routable_text: &str,
) -> IrSourceMap {
    routable.debug.source_map([
        (
            "ir/logical.rcir".to_owned(),
            logical_ranges(logical, logical_text),
        ),
        (
            "ir/routable.rcir".to_owned(),
            routable_ranges(routable, routable_text),
        ),
    ])
}

fn logical_ranges(design: &super::LogicalDesign, text: &str) -> Vec<DebugRange> {
    let lines = text.lines().collect::<Vec<_>>();
    let mut ranges = Vec::new();
    for module in &design.modules {
        let Some(module_start) =
            find_line(&lines, &format!("module {} {{", rcir_name(&module.name)))
        else {
            continue;
        };
        let module_end = find_block_end(&lines, module_start, "}");
        push_range(
            design,
            &mut ranges,
            logical_entity(&module.name, "module", &module.name),
            module_start,
            module_end,
        );
        for net in &module.nets {
            if let Some(line) = find_line_in(
                &lines,
                module_start,
                module_end,
                &format!("net {} :", rcir_name(&net.name)),
            )
            .or_else(|| {
                module
                    .ports
                    .iter()
                    .find(|port| port.net == net.name)
                    .and_then(|port| {
                        find_line_in(
                            &lines,
                            module_start,
                            module_end,
                            &format!(
                                "port {} {} :",
                                match port.direction {
                                    super::LogicalPortDirection::Input => "input",
                                    super::LogicalPortDirection::Output => "output",
                                },
                                rcir_name(&port.name)
                            ),
                        )
                    })
            }) {
                push_range(
                    design,
                    &mut ranges,
                    logical_entity(&module.name, "net", &net.name),
                    line,
                    line,
                );
            }
        }
        for instance in &module.instances {
            let needle = format!("instance {} :", rcir_name(&instance.name));
            if let Some(start) = find_line_in(&lines, module_start, module_end, &needle) {
                push_range(
                    design,
                    &mut ranges,
                    logical_entity(&module.name, "instance", &instance.name),
                    start,
                    find_block_end(&lines, start, "  }"),
                );
            }
        }
        for cell in &module.cells {
            let needle = format!("cell {} :", rcir_name(&cell.name));
            if let Some(start) = find_line_in(&lines, module_start, module_end, &needle) {
                push_range(
                    design,
                    &mut ranges,
                    logical_entity(&module.name, "cell", &cell.name),
                    start,
                    find_block_end(&lines, start, "  }"),
                );
            }
        }
    }
    ranges
}

fn routable_ranges(design: &super::RoutableDesign, text: &str) -> Vec<DebugRange> {
    use super::routable::RoutableModuleBody;

    let lines = text.lines().collect::<Vec<_>>();
    let mut ranges = Vec::new();
    for module in &design.modules {
        let header = match module.body {
            RoutableModuleBody::Leaf { .. } => "leaf",
            RoutableModuleBody::Composite { .. } => "module",
        };
        let Some(module_start) = find_line(
            &lines,
            &format!("{header} {} {{", rcir_quoted(&module.name)),
        ) else {
            continue;
        };
        let module_end = find_block_end(&lines, module_start, "}");
        push_range(
            design,
            &mut ranges,
            routable_entity(&module.name, "module", &module.name),
            module_start,
            module_end,
        );
        for port in &module.ports {
            let needle = format!(
                "port {} {}",
                match port.direction {
                    super::routable::RoutablePortDirection::Input => "input",
                    super::routable::RoutablePortDirection::Output => "output",
                },
                rcir_quoted(&port.name)
            );
            if let Some(line) = find_line_in(&lines, module_start, module_end, &needle) {
                push_range(
                    design,
                    &mut ranges,
                    routable_entity(&module.name, "port", &port.name),
                    line,
                    line,
                );
            }
        }
        match &module.body {
            RoutableModuleBody::Composite { instances, nets } => {
                for instance in instances {
                    let needle = format!("instance {} :", rcir_quoted(&instance.name));
                    if let Some(line) = find_line_in(&lines, module_start, module_end, &needle) {
                        push_range(
                            design,
                            &mut ranges,
                            routable_entity(&module.name, "instance", &instance.name),
                            line,
                            line,
                        );
                    }
                }
                for net in nets {
                    let needle = format!("net {} class", rcir_quoted(&net.name));
                    if let Some(line) = find_line_in(&lines, module_start, module_end, &needle) {
                        push_range(
                            design,
                            &mut ranges,
                            routable_entity(&module.name, "net", &net.name),
                            line,
                            line,
                        );
                    }
                }
            }
            RoutableModuleBody::Leaf { nodes } => {
                for node in nodes {
                    let needle = format!("node {} ", node.id);
                    if let Some(line) = find_line_in(&lines, module_start, module_end, &needle) {
                        push_range(
                            design,
                            &mut ranges,
                            routable_entity(&module.name, "node", &node.id.to_string()),
                            line,
                            line,
                        );
                    }
                }
            }
        }
    }
    ranges
}

trait DebugDesign {
    fn debug(&self) -> &IrDebugInfo;
}

impl DebugDesign for super::LogicalDesign {
    fn debug(&self) -> &IrDebugInfo {
        &self.debug
    }
}

impl DebugDesign for super::RoutableDesign {
    fn debug(&self) -> &IrDebugInfo {
        &self.debug
    }
}

fn push_range(
    design: &impl DebugDesign,
    ranges: &mut Vec<DebugRange>,
    entity: String,
    start: usize,
    end: usize,
) {
    if let Some(location) = design.debug().get(&entity) {
        ranges.push(DebugRange {
            entity,
            start_line: start + 1,
            end_line: end + 1,
            location,
        });
    }
}

fn find_line(lines: &[&str], needle: &str) -> Option<usize> {
    lines.iter().position(|line| line.trim() == needle)
}

fn find_line_in(lines: &[&str], start: usize, end: usize, needle: &str) -> Option<usize> {
    (start..=end).find(|line| lines[*line].trim_start().starts_with(needle))
}

fn find_block_end(lines: &[&str], start: usize, closing: &str) -> usize {
    (start + 1..lines.len())
        .find(|line| lines[*line] == closing)
        .unwrap_or(start)
}

fn rcir_name(value: &str) -> String {
    let mut chars = value.chars();
    let bare = matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-');
    if bare {
        value.to_owned()
    } else {
        rcir_quoted(value)
    }
}

fn rcir_quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ancestors(map: &IrSourceMap, location: DebugLocationId) -> BTreeSet<DebugLocationId> {
        fn visit(map: &IrSourceMap, location: DebugLocationId, result: &mut BTreeSet<usize>) {
            if !result.insert(location) {
                return;
            }
            match &map.locations[location] {
                DebugLocation::Source { .. } => {}
                DebugLocation::Derived { parent, .. } => visit(map, *parent, result),
                DebugLocation::Fused { parents } => {
                    for parent in parents {
                        visit(map, *parent, result);
                    }
                }
            }
        }
        let mut result = BTreeSet::new();
        visit(map, location, &mut result);
        result
    }

    #[test]
    fn counter_source_map_links_all_three_documents() -> eyre::Result<()> {
        let source = include_str!("../../test/counter.snapshot/counter.v");
        let logical = super::super::LogicalDesign::from_verilog_source_named(source, "counter.v")?;
        let routable = logical.lower_to_routable()?;
        let map = build_source_map(
            &logical,
            &logical.to_string(),
            &routable,
            &routable.to_string(),
        );

        assert!(map
            .documents
            .get("counter.v")
            .is_some_and(|ranges| !ranges.is_empty()));
        assert!(map
            .documents
            .get("ir/logical.rcir")
            .is_some_and(|ranges| ranges.iter().any(|range| range.entity.contains("/cell/"))));
        assert!(map.documents.get("ir/routable.rcir").is_some_and(|ranges| {
            ranges
                .iter()
                .any(|range| range.entity.contains("/instance/q_0_master"))
        }));
        Ok(())
    }

    #[test]
    fn hierarchical_leaf_nodes_keep_definition_level_provenance() -> eyre::Result<()> {
        let source = include_str!("../../test/d-flip-flop.snapshot/d-flip-flop.v");
        let logical =
            super::super::LogicalDesign::from_verilog_source_named(source, "d-flip-flop.v")?;
        let routable = logical.lower_to_routable()?;
        let map = build_source_map(
            &logical,
            &logical.to_string(),
            &routable,
            &routable.to_string(),
        );

        let inv_not = map.entities[&routable_entity("inv", "node", "1")].location;
        let logical_not = map.entities[&logical_entity("not_clk", "cell", "clk_n")].location;
        assert!(ancestors(&map, inv_not).contains(&logical_not));

        let latch = map.entities[&routable_entity("master", "node", "2")].location;
        let logical_latch = map.entities[&logical_entity("d_latch", "cell", "state")].location;
        assert!(ancestors(&map, latch).contains(&logical_latch));
        assert!(!ancestors(&map, inv_not).contains(&logical_latch));

        assert!(map.documents["ir/routable.rcir"]
            .iter()
            .any(|range| range.entity == routable_entity("d_flip_flop", "net", "clk_n")));
        let latch_scope = logical_entity("d_latch", "module", "d_latch");
        assert_eq!(
            map.entities[&logical_entity("d_latch", "cell", "state")].parent_scope,
            Some(latch_scope.clone())
        );
        assert!(map.documents["d-flip-flop.v"]
            .iter()
            .any(|range| range.entity == latch_scope
                && range.start_line == 8
                && range.end_line == 16));
        Ok(())
    }

    #[test]
    fn source_map_separates_instantiation_references_from_provenance() -> eyre::Result<()> {
        let source = include_str!("../../test/d-flip-flop.snapshot/d-flip-flop.v");
        let logical =
            super::super::LogicalDesign::from_verilog_source_named(source, "d-flip-flop.v")?;
        let routable = logical.lower_to_routable()?;
        let map = build_source_map(
            &logical,
            &logical.to_string(),
            &routable,
            &routable.to_string(),
        );

        let expected = [
            DebugRelation {
                kind: DebugRelationKind::Instantiates,
                from: logical_entity("d_flip_flop", "instance", "inv"),
                to: logical_entity("not_clk", "module", "not_clk"),
            },
            DebugRelation {
                kind: DebugRelationKind::Instantiates,
                from: routable_entity("d_flip_flop", "instance", "inv"),
                to: routable_entity("inv", "module", "inv"),
            },
            DebugRelation {
                kind: DebugRelationKind::Instantiates,
                from: routable_entity("d_flip_flop", "instance", "slave"),
                to: routable_entity("master", "module", "master"),
            },
        ];
        for relation in expected {
            assert!(map.relations.contains(&relation), "missing {relation:?}");
        }
        Ok(())
    }
}
