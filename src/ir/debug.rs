use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub type DebugLocationId = usize;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IrDebugInfo {
    pub locations: Vec<DebugLocation>,
    pub entities: BTreeMap<String, DebugLocationId>,
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
    pub documents: BTreeMap<String, Vec<DebugRange>>,
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
            if seen.insert((file.clone(), *start_line, *end_line, location)) {
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
            documents,
        }
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
}
