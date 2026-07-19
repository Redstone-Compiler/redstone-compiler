use std::fmt;
use std::str::FromStr;

use super::routable::{
    Endpoint, NetClass, RoutableDesign, RoutableInstance, RoutableModule, RoutableModuleBody,
    RoutableNet, RoutableNode, RoutableNodeKind, RoutablePort, RoutablePortDirection,
    RoutableSequentialPrimitive,
};
use super::syntax::{tokenize, Token};
use super::{
    CandidateSpec, CongestionSpec, Free3dSweepSpec, InputPlacementSpec, LayerAssignmentSpec,
    LocalPlacerSpec, NetOrderSpec, NotRouteSpec, ObjectiveSpec, PhysicalConstraintSpec,
    PhysicalRegionSpec, PhysicalSpec, PlacementHeuristicSpec, PlacementSamplingSpec,
    PlacementScheduleSpec, PlacementSpec, PnrSpec, PortRef, PreferenceSpec, RoutableDocument,
    RouteStageSpec, RouteStrategySpec, RouteValidationSpec, RoutingSpec, SamplingSpec, SearchSpec,
    TorchPlacementSpec,
};

impl fmt::Display for RoutableDesign {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_routable_document(output, &RoutableDocument::circuit_only(self.clone()))
    }
}

impl fmt::Display for RoutableDocument {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_routable_document(output, self)
    }
}

fn write_routable_document(
    output: &mut fmt::Formatter<'_>,
    document: &RoutableDocument,
) -> fmt::Result {
    let design = &document.design;
    writeln!(output, "rcir {};", design.version)?;
    writeln!(output, "stage routable;")?;
    writeln!(output, "target {};", quoted(&design.target))?;
    writeln!(output, "top {};", quoted(&design.top))?;

    for (name, candidate) in &document.candidate_profiles {
        writeln!(output)?;
        writeln!(output, "profile pnr.candidate {} {{", quoted(name))?;
        write_candidate_body(output, candidate, "  ")?;
        writeln!(output, "}}")?;
    }
    for (name, pnr) in &document.design_profiles {
        writeln!(output)?;
        writeln!(output, "profile pnr.design {} {{", quoted(name))?;
        write_design_profile_body(output, pnr)?;
        writeln!(output, "}}")?;
    }

    let mut modules = design.modules.iter().collect::<Vec<_>>();
    modules.sort_by(|left, right| left.name.cmp(&right.name));
    for module in modules {
        writeln!(output)?;
        match &module.body {
            RoutableModuleBody::Leaf { nodes } => {
                if let Some(profile) = document.design_bindings.get(&module.name) {
                    write_profile_binding(output, "design", profile)?;
                }
                if let Some(profile) = document.candidate_bindings.get(&module.name) {
                    write_profile_binding(output, "candidate", profile)?;
                }
                write_leaf(output, module, nodes, &document.pin_search)?
            }
            RoutableModuleBody::Composite { instances, nets } => {
                if let Some(profile) = document.design_bindings.get(&module.name) {
                    write_profile_binding(output, "design", profile)?;
                }
                write_composite(output, module, instances, nets)?
            }
        }
    }
    if let Some(physical) = &document.physical {
        writeln!(output)?;
        write_physical(output, physical)?;
    }
    Ok(())
}

fn write_design_profile_body(output: &mut fmt::Formatter<'_>, pnr: &PnrSpec) -> fmt::Result {
    let placement = &pnr.placement;
    writeln!(output, "  placement {{")?;
    writeln!(output, "    initial-spacing {};", placement.initial_spacing)?;
    writeln!(output, "    shelf-width {};", placement.shelf_width)?;
    writeln!(output, "    max-attempts {};", placement.max_attempts)?;
    for heuristic in &placement.heuristics {
        write_placement_heuristic(output, heuristic)?;
    }
    writeln!(output, "    congestion {{")?;
    writeln!(
        output,
        "      bin-size-xy {};",
        placement.congestion.bin_size_xy
    )?;
    writeln!(
        output,
        "      bin-size-z {};",
        placement.congestion.bin_size_z
    )?;
    writeln!(output, "    }}")?;
    writeln!(output, "    objective {{")?;
    let objective = placement.objective;
    writeln!(
        output,
        "      placement-volume {};",
        objective.placement_volume
    )?;
    writeln!(output, "      xy-footprint {};", objective.xy_footprint)?;
    writeln!(output, "      height-span {};", objective.height_span)?;
    writeln!(
        output,
        "      estimated-wire-length {};",
        objective.estimated_wire_length
    )?;
    writeln!(
        output,
        "      vertical-distance {};",
        objective.vertical_distance
    )?;
    writeln!(
        output,
        "      routing-congestion {};",
        objective.routing_congestion
    )?;
    writeln!(output, "    }}")?;
    writeln!(output, "  }}")?;

    writeln!(output)?;
    writeln!(output, "  routing {{")?;
    if let Some(probe) = pnr.routing.probe {
        write_route_stage(output, "probe", probe)?;
    }
    write_route_stage(output, "primary", pnr.routing.primary)?;
    if let Some(refinement) = pnr.routing.refinement {
        write_route_stage(output, "refinement", refinement)?;
    }
    write!(output, "    net-order [")?;
    for (index, order) in pnr.routing.net_order.iter().enumerate() {
        if index > 0 {
            write!(output, ", ")?;
        }
        write!(
            output,
            "{}",
            match order {
                NetOrderSpec::Criticality => "criticality",
                NetOrderSpec::HighestFanoutFirst => "highest-fanout-first",
                NetOrderSpec::ReverseCriticality => "reverse-criticality",
            }
        )?;
    }
    writeln!(output, "];")?;
    writeln!(output, "  }}")?;

    let search = pnr.search;
    writeln!(output)?;
    writeln!(output, "  search {{")?;
    writeln!(
        output,
        "    candidates-per-child {};",
        search.candidates_per_child
    )?;
    writeln!(
        output,
        "    layout-combinations {};",
        search.layout_combinations
    )?;
    writeln!(
        output,
        "    detailed-routing-attempts {};",
        search.detailed_routing_attempts
    )?;
    writeln!(
        output,
        "    refined-routing-attempts {};",
        search.refined_routing_attempts
    )?;
    writeln!(
        output,
        "    refinement-rounds {};",
        search.refinement_rounds
    )?;
    writeln!(output, "  }}")
}

fn write_profile_binding(
    output: &mut fmt::Formatter<'_>,
    kind: &str,
    profile: &str,
) -> fmt::Result {
    writeln!(output, "@pnr.{kind}(profile = {})", quoted(profile))
}

fn write_placement_heuristic(
    output: &mut fmt::Formatter<'_>,
    heuristic: &PlacementHeuristicSpec,
) -> fmt::Result {
    let name = match heuristic {
        PlacementHeuristicSpec::Shelf => Some("shelf"),
        PlacementHeuristicSpec::Grid => Some("grid"),
        PlacementHeuristicSpec::RegisterCarryChain => Some("register-carry-chain"),
        PlacementHeuristicSpec::RegisterCarryAlignedSlices => Some("register-carry-aligned-slices"),
        PlacementHeuristicSpec::RegisterGrid => Some("register-grid"),
        PlacementHeuristicSpec::RegisterTriangles => Some("register-triangles"),
        PlacementHeuristicSpec::RegisterSlices => Some("register-slices"),
        _ => None,
    };
    if let Some(name) = name {
        return writeln!(output, "    heuristic {name};");
    }
    match heuristic {
        PlacementHeuristicSpec::Layered3d {
            layers,
            layer_spacing,
            assignment,
        } => {
            writeln!(output, "    heuristic layered3d {{")?;
            writeln!(output, "      layers {layers};")?;
            writeln!(output, "      layer-spacing {layer_spacing};")?;
            writeln!(
                output,
                "      assignment {};",
                match assignment {
                    LayerAssignmentSpec::Alternating => "alternating",
                    LayerAssignmentSpec::NetAware => "net-aware",
                }
            )?;
            writeln!(output, "    }}")
        }
        PlacementHeuristicSpec::Free3dSweep(config) => {
            writeln!(output, "    heuristic free3d {{")?;
            write!(output, "      seeds [")?;
            write_u64_list(output, &config.seeds)?;
            writeln!(output, "];")?;
            write!(output, "      clearances [")?;
            write_usize_list(output, &config.clearances)?;
            writeln!(output, "];")?;
            writeln!(output, "      iterations {};", config.iterations)?;
            writeln!(output, "      step-size {};", config.step_size)?;
            writeln!(output, "      attraction {};", config.attraction)?;
            writeln!(output, "      repulsion {};", config.repulsion)?;
            writeln!(output, "      compactness {};", config.compactness)?;
            writeln!(output, "      damping {};", config.damping)?;
            writeln!(output, "      vertical-scale {};", config.vertical_scale)?;
            writeln!(output, "    }}")
        }
        _ => unreachable!(),
    }
}

fn write_candidate_body(
    output: &mut fmt::Formatter<'_>,
    candidate: &CandidateSpec,
    indent: &str,
) -> fmt::Result {
    writeln!(
        output,
        "{indent}search-box [{}, {}, {}];",
        candidate.search_box[0], candidate.search_box[1], candidate.search_box[2]
    )?;
    writeln!(output, "{indent}retain {};", candidate.retain)?;
    match candidate.combinational_samples {
        Some(value) => writeln!(output, "{indent}combinational-samples {value};")?,
        None => writeln!(output, "{indent}combinational-samples none;")?,
    }
    writeln!(output, "{indent}local-placer {{")?;
    write_local_placer_body(output, &candidate.local_placer, &format!("{indent}  "))?;
    writeln!(output, "{indent}}}")
}

fn write_local_placer_body(
    output: &mut fmt::Formatter<'_>,
    local: &LocalPlacerSpec,
    indent: &str,
) -> fmt::Result {
    writeln!(output, "{indent}random-seed {};", local.random_seed)?;
    writeln!(
        output,
        "{indent}schedule {};",
        match local.schedule {
            PlacementScheduleSpec::Topological => "topological",
            PlacementScheduleSpec::MinFrontier => "min-frontier",
            PlacementScheduleSpec::Reconvergence => "reconvergence",
            PlacementScheduleSpec::Auto => "auto",
        }
    )?;
    writeln!(
        output,
        "{indent}greedy-input-generation {};",
        local.greedy_input_generation
    )?;
    writeln!(
        output,
        "{indent}input-placement {};",
        match local.input_placement {
            InputPlacementSpec::Boundary => "boundary",
            InputPlacementSpec::Anywhere => "anywhere",
        }
    )?;
    match local.input_candidate_limit {
        Some(value) => writeln!(output, "{indent}input-candidate-limit {value};")?,
        None => writeln!(output, "{indent}input-candidate-limit none;")?,
    }
    writeln!(
        output,
        "{indent}step-sampling {};",
        sampling_text(local.step_sampling)
    )?;
    writeln!(
        output,
        "{indent}placement-sampling {};",
        placement_sampling_text(local.placement_sampling)
    )?;
    writeln!(output, "{indent}leak-sampling {};", local.leak_sampling)?;
    writeln!(
        output,
        "{indent}route-torch-directly {};",
        local.route_torch_directly
    )?;
    writeln!(
        output,
        "{indent}materialize-outputs {};",
        local.materialize_outputs
    )?;
    writeln!(
        output,
        "{indent}torch-placement {};",
        match local.torch_placement {
            TorchPlacementSpec::DirectOnly => "direct-only",
            TorchPlacementSpec::AnywhereNonAdjacent => "anywhere-non-adjacent",
        }
    )?;
    writeln!(
        output,
        "{indent}not-route-strategy {};",
        match local.not_route_strategy {
            NotRouteSpec::DirectOnly => "direct-only",
            NotRouteSpec::RedstoneOnly => "redstone-only",
            NotRouteSpec::DirectAndRedstone => "direct-and-redstone",
        }
    )?;
    writeln!(
        output,
        "{indent}max-not-route-step {};",
        local.max_not_route_step
    )?;
    writeln!(
        output,
        "{indent}not-route-step-sampling {};",
        sampling_text(local.not_route_step_sampling)
    )?;
    writeln!(output, "{indent}max-route-step {};", local.max_route_step)?;
    writeln!(
        output,
        "{indent}route-step-sampling {};",
        sampling_text(local.route_step_sampling)
    )
}

fn write_route_stage(
    output: &mut fmt::Formatter<'_>,
    name: &str,
    stage: RouteStageSpec,
) -> fmt::Result {
    writeln!(output, "    {name} {{")?;
    match stage.strategy {
        RouteStrategySpec::BreadthFirst => writeln!(output, "      strategy breadth-first;")?,
        RouteStrategySpec::AStar => writeln!(output, "      strategy astar;")?,
        RouteStrategySpec::DirectGreedy { max_steps } => {
            writeln!(output, "      strategy direct-greedy;")?;
            writeln!(output, "      max-steps {max_steps};")?;
        }
        RouteStrategySpec::GreedyBeam {
            width,
            max_expansions,
            variant_seed,
        } => {
            writeln!(output, "      strategy greedy-beam;")?;
            writeln!(output, "      width {width};")?;
            writeln!(output, "      max-expansions {max_expansions};")?;
            writeln!(output, "      variant-seed {variant_seed};")?;
        }
    }
    writeln!(
        output,
        "      validation {};",
        match stage.validation {
            RouteValidationSpec::Incremental => "incremental",
            RouteValidationSpec::Deferred => "deferred",
        }
    )?;
    writeln!(output, "    }}")
}

fn write_physical(output: &mut fmt::Formatter<'_>, physical: &PhysicalSpec) -> fmt::Result {
    writeln!(output, "physical {{")?;
    for (name, region) in &physical.regions {
        writeln!(
            output,
            "  region {} box [{}, {}, {}] [{}, {}, {}];",
            quoted(name),
            region.min[0],
            region.min[1],
            region.min[2],
            region.max[0],
            region.max[1],
            region.max[2]
        )?;
    }
    for constraint in &physical.constraints {
        match constraint {
            PhysicalConstraintSpec::Inside {
                id,
                instance,
                region,
            } => writeln!(
                output,
                "  require {} instance {} inside {};",
                quoted(id),
                quoted(instance),
                quoted(region)
            )?,
            PhysicalConstraintSpec::LayerRange {
                id,
                instance,
                min,
                max,
            } => writeln!(
                output,
                "  require {} instance {} layer {min}..{max};",
                quoted(id),
                quoted(instance)
            )?,
            PhysicalConstraintSpec::FixedOrigin {
                id,
                instance,
                origin,
            } => writeln!(
                output,
                "  lock {} instance {} at [{}, {}, {}];",
                quoted(id),
                quoted(instance),
                origin[0],
                origin[1],
                origin[2]
            )?,
            PhysicalConstraintSpec::NetPriority { id, net, priority } => writeln!(
                output,
                "  priority {} net {} {priority};",
                quoted(id),
                quoted(net)
            )?,
            PhysicalConstraintSpec::NetAvoid { id, net, region } => writeln!(
                output,
                "  require {} net {} avoid {};",
                quoted(id),
                quoted(net),
                quoted(region)
            )?,
            PhysicalConstraintSpec::PreferInside {
                id,
                instance,
                region,
                strength,
            } => writeln!(
                output,
                "  prefer {} instance {} inside {} strength {};",
                quoted(id),
                quoted(instance),
                quoted(region),
                preference_text(*strength)
            )?,
            PhysicalConstraintSpec::SameLayer {
                id,
                first,
                second,
                strength,
            } => writeln!(
                output,
                "  prefer {} instances {} {} same-layer strength {};",
                quoted(id),
                quoted(first),
                quoted(second),
                preference_text(*strength)
            )?,
        }
    }
    writeln!(output, "}}")
}

fn sampling_text(sampling: SamplingSpec) -> String {
    match sampling {
        SamplingSpec::None => "none".to_owned(),
        SamplingSpec::Take(count) => format!("take({count})"),
        SamplingSpec::Random(count) => format!("random({count})"),
    }
}

fn placement_sampling_text(sampling: PlacementSamplingSpec) -> String {
    match sampling {
        PlacementSamplingSpec::StepPolicy => "step-policy".to_owned(),
        PlacementSamplingSpec::Cost {
            count,
            random_count,
            start_step,
        } => format!("cost({count}, {random_count}, {start_step})"),
        PlacementSamplingSpec::Ranked {
            count,
            random_count,
            start_step,
        } => format!("ranked({count}, {random_count}, {start_step})"),
    }
}

fn preference_text(preference: PreferenceSpec) -> &'static str {
    match preference {
        PreferenceSpec::Weak => "weak",
        PreferenceSpec::Medium => "medium",
        PreferenceSpec::Strong => "strong",
    }
}

fn write_u64_list(output: &mut fmt::Formatter<'_>, values: &[u64]) -> fmt::Result {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(output, ", ")?;
        }
        write!(output, "{value}")?;
    }
    Ok(())
}

fn write_usize_list(output: &mut fmt::Formatter<'_>, values: &[usize]) -> fmt::Result {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(output, ", ")?;
        }
        write!(output, "{value}")?;
    }
    Ok(())
}

impl FromStr for RoutableDesign {
    type Err = eyre::Report;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Ok(source.parse::<RoutableDocument>()?.design)
    }
}

impl FromStr for RoutableDocument {
    type Err = eyre::Report;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let tokens = tokenize(source)?;
        let mut parser = Parser::new(tokens);
        parser.expect_keyword("rcir")?;
        let version = parser.expect_number()?;
        parser.expect_symbol(';')?;
        parser.expect_keyword("stage")?;
        parser.expect_keyword("routable")?;
        parser.expect_symbol(';')?;
        parser.expect_keyword("target")?;
        let target = parser.expect_string()?;
        parser.expect_symbol(';')?;
        parser.expect_keyword("top")?;
        let top = parser.expect_string()?;
        parser.expect_symbol(';')?;

        let mut candidate_profiles = std::collections::BTreeMap::new();
        let mut design_profiles = std::collections::BTreeMap::new();
        while parser.consume_keyword("profile") {
            parser.expect_keyword("pnr")?;
            parser.expect_symbol('.')?;
            let kind = parser.expect_word()?;
            let name = parser.expect_string()?;
            match kind.as_str() {
                "candidate" => {
                    let profile = parser.parse_candidate_spec()?;
                    if candidate_profiles.insert(name.clone(), profile).is_some() {
                        eyre::bail!("duplicate pnr.candidate profile `{name}`");
                    }
                }
                "design" => {
                    let profile = parser.parse_design_profile()?;
                    if design_profiles.insert(name.clone(), profile).is_some() {
                        eyre::bail!("duplicate pnr.design profile `{name}`");
                    }
                }
                _ => eyre::bail!("unknown PnR profile kind `pnr.{kind}`"),
            }
        }
        let mut modules = Vec::new();
        let mut pin_search = std::collections::BTreeMap::new();
        let mut candidate_bindings = std::collections::BTreeMap::new();
        let mut design_bindings = std::collections::BTreeMap::new();
        while !parser.is_done() && !parser.peek_keyword("physical") {
            let mut bindings = Vec::new();
            while parser.consume_symbol('@') {
                bindings.push(parser.parse_definition_profile_binding()?);
            }
            if parser.consume_keyword("leaf") {
                let module = parser.parse_leaf(&mut pin_search)?;
                for (kind, profile) in bindings {
                    match kind.as_str() {
                        "candidate" => {
                            if candidate_bindings
                                .insert(module.name.clone(), profile)
                                .is_some()
                            {
                                eyre::bail!("duplicate @pnr.candidate on leaf `{}`", module.name);
                            }
                        }
                        "design" => {
                            if design_bindings
                                .insert(module.name.clone(), profile)
                                .is_some()
                            {
                                eyre::bail!("duplicate @pnr.design on leaf `{}`", module.name);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                modules.push(module);
            } else if parser.consume_keyword("module") {
                let module = parser.parse_composite()?;
                for (kind, profile) in bindings {
                    if kind != "design" {
                        eyre::bail!(
                            "module `{}` requires @pnr.design, not @pnr.{kind}",
                            module.name
                        );
                    }
                    if design_bindings
                        .insert(module.name.clone(), profile)
                        .is_some()
                    {
                        eyre::bail!("duplicate @pnr.design on module `{}`", module.name);
                    }
                }
                modules.push(module);
            } else {
                return Err(
                    parser.unexpected("definition annotation, `leaf`, `module`, or `physical`")
                );
            }
        }

        let physical = if parser.consume_keyword("physical") {
            Some(parser.parse_physical()?)
        } else {
            None
        };
        if !parser.is_done() {
            return Err(parser.unexpected("end of RCIR document"));
        }

        let design = RoutableDesign {
            version,
            target,
            top,
            modules,
            debug: Default::default(),
        };
        design.validate()?;
        for (definition, profile) in &candidate_bindings {
            if !candidate_profiles.contains_key(profile) {
                eyre::bail!(
                    "leaf `{definition}` references unknown pnr.candidate profile `{profile}`"
                );
            }
        }
        for (definition, profile) in &design_bindings {
            if !design_profiles.contains_key(profile) {
                eyre::bail!(
                    "module `{definition}` references unknown pnr.design profile `{profile}`"
                );
            }
        }
        Ok(RoutableDocument {
            design,
            candidate_profiles,
            design_profiles,
            candidate_bindings,
            design_bindings,
            pin_search,
            physical,
        })
    }
}

fn write_leaf(
    output: &mut fmt::Formatter<'_>,
    module: &RoutableModule,
    nodes: &[RoutableNode],
    pin_search: &std::collections::BTreeMap<PortRef, Vec<[usize; 3]>>,
) -> fmt::Result {
    writeln!(output, "leaf {} {{", quoted(&module.name))?;
    write_ports(output, &module.name, &module.ports, pin_search)?;
    let mut nodes = nodes.iter().collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.id);
    for node in nodes {
        write!(output, "  node {} ", node.id)?;
        match &node.kind {
            RoutableNodeKind::Input { name } => write!(output, "input {}", quoted(name))?,
            RoutableNodeKind::Output { name } => write!(output, "output {}", quoted(name))?,
            RoutableNodeKind::Not => write!(output, "logic not")?,
            RoutableNodeKind::Or => write!(output, "logic or")?,
            RoutableNodeKind::Sequential {
                primitive,
                input_ports,
                output_ports,
            } => {
                write!(
                    output,
                    "sequential {} input_ports ",
                    sequential_name(*primitive)
                )?;
                write_string_list(output, input_ports)?;
                write!(output, " output_ports ")?;
                write_string_list(output, output_ports)?;
            }
        }
        write!(output, " inputs ")?;
        write_number_list(output, &node.inputs)?;
        if !node.tag.is_empty() {
            write!(output, " tag {}", quoted(&node.tag))?;
        }
        writeln!(output, ";")?;
    }
    writeln!(output, "}}")
}

fn write_composite(
    output: &mut fmt::Formatter<'_>,
    module: &RoutableModule,
    instances: &[RoutableInstance],
    nets: &[RoutableNet],
) -> fmt::Result {
    writeln!(output, "module {} {{", quoted(&module.name))?;
    write_ports(output, &module.name, &module.ports, &Default::default())?;

    let mut instances = instances.iter().collect::<Vec<_>>();
    instances.sort_by(|left, right| left.name.cmp(&right.name));
    for instance in instances {
        write!(
            output,
            "  instance {} : {}",
            quoted(&instance.name),
            quoted(&instance.module)
        )?;
        if let Some(origin) = &instance.origin {
            write!(output, " origin {}", quoted(origin))?;
        }
        writeln!(output, ";")?;
    }

    let mut nets = nets.iter().collect::<Vec<_>>();
    nets.sort_by(|left, right| left.name.cmp(&right.name));
    for net in nets {
        write!(
            output,
            "  net {} class {} driver ",
            quoted(&net.name),
            net_class_name(net.class)
        )?;
        write_endpoint(output, &net.driver)?;
        write!(output, " sinks [")?;
        let mut sinks = net.sinks.iter().collect::<Vec<_>>();
        sinks.sort();
        for (index, sink) in sinks.into_iter().enumerate() {
            if index > 0 {
                write!(output, ", ")?;
            }
            write_endpoint(output, sink)?;
        }
        write!(output, "]")?;
        if let Some(origin) = &net.origin {
            write!(output, " origin {}", quoted(origin))?;
        }
        writeln!(output, ";")?;
    }
    writeln!(output, "}}")
}

fn write_ports(
    output: &mut fmt::Formatter<'_>,
    definition: &str,
    ports: &[RoutablePort],
    pin_search: &std::collections::BTreeMap<PortRef, Vec<[usize; 3]>>,
) -> fmt::Result {
    let mut ports = ports.iter().collect::<Vec<_>>();
    ports.sort_by(|left, right| left.name.cmp(&right.name));
    for port in ports {
        let key = PortRef {
            definition: definition.to_owned(),
            port: port.name.clone(),
        };
        if let Some(positions) = pin_search.get(&key) {
            write!(output, "  @pnr.pin_search(positions = [")?;
            for (index, position) in positions.iter().enumerate() {
                if index > 0 {
                    write!(output, ", ")?;
                }
                write!(
                    output,
                    "[{}, {}, {}]",
                    position[0], position[1], position[2]
                )?;
            }
            writeln!(output, "])")?;
        }
        writeln!(
            output,
            "  port {} {};",
            direction_name(port.direction),
            quoted(&port.name)
        )?;
    }
    Ok(())
}

fn write_endpoint(output: &mut fmt::Formatter<'_>, endpoint: &Endpoint) -> fmt::Result {
    match endpoint {
        Endpoint::SelfPort { port } => write!(output, "self.{}", quoted(port)),
        Endpoint::InstancePort { instance, port } => {
            write!(output, "{}.{}", quoted(instance), quoted(port))
        }
    }
}

fn write_string_list(output: &mut fmt::Formatter<'_>, values: &[String]) -> fmt::Result {
    write!(output, "[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(output, ", ")?;
        }
        write!(output, "{}", quoted(value))?;
    }
    write!(output, "]")
}

fn write_number_list(output: &mut fmt::Formatter<'_>, values: &[usize]) -> fmt::Result {
    write!(output, "[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(output, ", ")?;
        }
        write!(output, "{value}")?;
    }
    write!(output, "]")
}

fn quoted(value: &str) -> String {
    let mut result = String::with_capacity(value.len() + 2);
    result.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            ch => result.push(ch),
        }
    }
    result.push('"');
    result
}

fn direction_name(direction: RoutablePortDirection) -> &'static str {
    match direction {
        RoutablePortDirection::Input => "input",
        RoutablePortDirection::Output => "output",
    }
}

fn net_class_name(class: NetClass) -> &'static str {
    match class {
        NetClass::Data => "data",
        NetClass::Clock => "clock",
        NetClass::Reset => "reset",
        NetClass::Io => "io",
    }
}

fn sequential_name(primitive: RoutableSequentialPrimitive) -> &'static str {
    match primitive {
        RoutableSequentialPrimitive::RsLatch => "rs_latch",
        RoutableSequentialPrimitive::DLatch => "d_latch",
    }
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    fn is_done(&self) -> bool {
        self.position == self.tokens.len()
    }

    fn parse_design_profile(&mut self) -> eyre::Result<PnrSpec> {
        self.expect_symbol('{')?;
        self.expect_keyword("placement")?;
        let placement = self.parse_placement_spec()?;
        self.expect_keyword("routing")?;
        let routing = self.parse_routing_spec()?;
        self.expect_keyword("search")?;
        let search = self.parse_search_spec()?;
        self.expect_symbol('}')?;
        Ok(PnrSpec {
            placement,
            routing,
            search,
        })
    }

    fn parse_definition_profile_binding(&mut self) -> eyre::Result<(String, String)> {
        self.expect_keyword("pnr")?;
        self.expect_symbol('.')?;
        let kind = self.expect_word()?;
        if kind != "candidate" && kind != "design" {
            eyre::bail!("unknown definition annotation `@pnr.{kind}`");
        }
        self.expect_symbol('(')?;
        self.expect_keyword("profile")?;
        self.expect_symbol('=')?;
        let profile = self.expect_string()?;
        self.expect_symbol(')')?;
        Ok((kind, profile))
    }

    fn parse_candidate_spec(&mut self) -> eyre::Result<CandidateSpec> {
        self.expect_symbol('{')?;
        self.expect_keyword("search-box")?;
        let search_box = self.parse_position()?;
        self.expect_symbol(';')?;
        self.expect_keyword("retain")?;
        let retain = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("combinational-samples")?;
        let combinational_samples = self.parse_optional_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("local-placer")?;
        let local_placer = self.parse_local_placer_spec()?;
        self.expect_symbol('}')?;
        Ok(CandidateSpec {
            search_box,
            retain,
            combinational_samples,
            local_placer,
        })
    }

    fn parse_local_placer_spec(&mut self) -> eyre::Result<LocalPlacerSpec> {
        self.expect_symbol('{')?;
        self.expect_keyword("random-seed")?;
        let random_seed = self.expect_u64()?;
        self.expect_symbol(';')?;
        self.expect_keyword("schedule")?;
        let schedule = match self.expect_word()?.as_str() {
            "topological" => PlacementScheduleSpec::Topological,
            "min-frontier" => PlacementScheduleSpec::MinFrontier,
            "reconvergence" => PlacementScheduleSpec::Reconvergence,
            "auto" => PlacementScheduleSpec::Auto,
            value => eyre::bail!("unknown placement schedule `{value}`"),
        };
        self.expect_symbol(';')?;
        self.expect_keyword("greedy-input-generation")?;
        let greedy_input_generation = self.expect_bool()?;
        self.expect_symbol(';')?;
        self.expect_keyword("input-placement")?;
        let input_placement = match self.expect_word()?.as_str() {
            "boundary" => InputPlacementSpec::Boundary,
            "anywhere" => InputPlacementSpec::Anywhere,
            value => eyre::bail!("unknown input placement `{value}`"),
        };
        self.expect_symbol(';')?;
        self.expect_keyword("input-candidate-limit")?;
        let input_candidate_limit = self.parse_optional_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("step-sampling")?;
        let step_sampling = self.parse_sampling()?;
        self.expect_symbol(';')?;
        self.expect_keyword("placement-sampling")?;
        let placement_sampling = self.parse_placement_sampling()?;
        self.expect_symbol(';')?;
        self.expect_keyword("leak-sampling")?;
        let leak_sampling = self.expect_bool()?;
        self.expect_symbol(';')?;
        self.expect_keyword("route-torch-directly")?;
        let route_torch_directly = self.expect_bool()?;
        self.expect_symbol(';')?;
        self.expect_keyword("materialize-outputs")?;
        let materialize_outputs = self.expect_bool()?;
        self.expect_symbol(';')?;
        self.expect_keyword("torch-placement")?;
        let torch_placement = match self.expect_word()?.as_str() {
            "direct-only" => TorchPlacementSpec::DirectOnly,
            "anywhere-non-adjacent" => TorchPlacementSpec::AnywhereNonAdjacent,
            value => eyre::bail!("unknown torch placement `{value}`"),
        };
        self.expect_symbol(';')?;
        self.expect_keyword("not-route-strategy")?;
        let not_route_strategy = match self.expect_word()?.as_str() {
            "direct-only" => NotRouteSpec::DirectOnly,
            "redstone-only" => NotRouteSpec::RedstoneOnly,
            "direct-and-redstone" => NotRouteSpec::DirectAndRedstone,
            value => eyre::bail!("unknown NOT-route strategy `{value}`"),
        };
        self.expect_symbol(';')?;
        self.expect_keyword("max-not-route-step")?;
        let max_not_route_step = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("not-route-step-sampling")?;
        let not_route_step_sampling = self.parse_sampling()?;
        self.expect_symbol(';')?;
        self.expect_keyword("max-route-step")?;
        let max_route_step = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("route-step-sampling")?;
        let route_step_sampling = self.parse_sampling()?;
        self.expect_symbol(';')?;
        self.expect_symbol('}')?;
        Ok(LocalPlacerSpec {
            random_seed,
            schedule,
            greedy_input_generation,
            input_placement,
            input_candidate_limit,
            step_sampling,
            placement_sampling,
            leak_sampling,
            route_torch_directly,
            materialize_outputs,
            torch_placement,
            not_route_strategy,
            max_not_route_step,
            not_route_step_sampling,
            max_route_step,
            route_step_sampling,
        })
    }

    fn parse_placement_spec(&mut self) -> eyre::Result<PlacementSpec> {
        self.expect_symbol('{')?;
        self.expect_keyword("initial-spacing")?;
        let initial_spacing = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("shelf-width")?;
        let shelf_width = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("max-attempts")?;
        let max_attempts = self.expect_usize()?;
        self.expect_symbol(';')?;
        let mut heuristics = Vec::new();
        while self.consume_keyword("heuristic") {
            heuristics.push(self.parse_placement_heuristic()?);
        }
        if heuristics.is_empty() {
            eyre::bail!("PnR placement requires at least one heuristic");
        }
        self.expect_keyword("congestion")?;
        self.expect_symbol('{')?;
        self.expect_keyword("bin-size-xy")?;
        let bin_size_xy = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("bin-size-z")?;
        let bin_size_z = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_symbol('}')?;
        self.expect_keyword("objective")?;
        self.expect_symbol('{')?;
        self.expect_keyword("placement-volume")?;
        let placement_volume = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("xy-footprint")?;
        let xy_footprint = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("height-span")?;
        let height_span = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("estimated-wire-length")?;
        let estimated_wire_length = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("vertical-distance")?;
        let vertical_distance = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("routing-congestion")?;
        let routing_congestion = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_symbol('}')?;
        self.expect_symbol('}')?;
        Ok(PlacementSpec {
            initial_spacing,
            shelf_width,
            max_attempts,
            heuristics,
            congestion: CongestionSpec {
                bin_size_xy,
                bin_size_z,
            },
            objective: ObjectiveSpec {
                placement_volume,
                xy_footprint,
                height_span,
                estimated_wire_length,
                vertical_distance,
                routing_congestion,
            },
        })
    }

    fn parse_placement_heuristic(&mut self) -> eyre::Result<PlacementHeuristicSpec> {
        let kind = self.expect_word()?;
        let simple = match kind.as_str() {
            "shelf" => Some(PlacementHeuristicSpec::Shelf),
            "grid" => Some(PlacementHeuristicSpec::Grid),
            "register-carry-chain" => Some(PlacementHeuristicSpec::RegisterCarryChain),
            "register-carry-aligned-slices" => {
                Some(PlacementHeuristicSpec::RegisterCarryAlignedSlices)
            }
            "register-grid" => Some(PlacementHeuristicSpec::RegisterGrid),
            "register-triangles" => Some(PlacementHeuristicSpec::RegisterTriangles),
            "register-slices" => Some(PlacementHeuristicSpec::RegisterSlices),
            _ => None,
        };
        if let Some(heuristic) = simple {
            self.expect_symbol(';')?;
            return Ok(heuristic);
        }
        match kind.as_str() {
            "layered3d" => {
                self.expect_symbol('{')?;
                self.expect_keyword("layers")?;
                let layers = self.expect_usize()?;
                self.expect_symbol(';')?;
                self.expect_keyword("layer-spacing")?;
                let layer_spacing = self.expect_usize()?;
                self.expect_symbol(';')?;
                self.expect_keyword("assignment")?;
                let assignment = match self.expect_word()?.as_str() {
                    "alternating" => LayerAssignmentSpec::Alternating,
                    "net-aware" => LayerAssignmentSpec::NetAware,
                    value => eyre::bail!("unknown layer assignment `{value}`"),
                };
                self.expect_symbol(';')?;
                self.expect_symbol('}')?;
                Ok(PlacementHeuristicSpec::Layered3d {
                    layers,
                    layer_spacing,
                    assignment,
                })
            }
            "free3d" => {
                self.expect_symbol('{')?;
                self.expect_keyword("seeds")?;
                let seeds = self.parse_u64_list()?;
                self.expect_symbol(';')?;
                self.expect_keyword("clearances")?;
                let clearances = self.parse_usize_list()?;
                self.expect_symbol(';')?;
                self.expect_keyword("iterations")?;
                let iterations = self.expect_usize()?;
                self.expect_symbol(';')?;
                self.expect_keyword("step-size")?;
                let step_size = self.expect_f64()?;
                self.expect_symbol(';')?;
                self.expect_keyword("attraction")?;
                let attraction = self.expect_f64()?;
                self.expect_symbol(';')?;
                self.expect_keyword("repulsion")?;
                let repulsion = self.expect_f64()?;
                self.expect_symbol(';')?;
                self.expect_keyword("compactness")?;
                let compactness = self.expect_f64()?;
                self.expect_symbol(';')?;
                self.expect_keyword("damping")?;
                let damping = self.expect_f64()?;
                self.expect_symbol(';')?;
                self.expect_keyword("vertical-scale")?;
                let vertical_scale = self.expect_f64()?;
                self.expect_symbol(';')?;
                self.expect_symbol('}')?;
                if seeds.is_empty() || clearances.is_empty() {
                    eyre::bail!("free3d seeds and clearances must not be empty");
                }
                Ok(PlacementHeuristicSpec::Free3dSweep(Free3dSweepSpec {
                    seeds,
                    clearances,
                    iterations,
                    step_size,
                    attraction,
                    repulsion,
                    compactness,
                    damping,
                    vertical_scale,
                }))
            }
            _ => eyre::bail!("unknown placement heuristic `{kind}`"),
        }
    }

    fn parse_routing_spec(&mut self) -> eyre::Result<RoutingSpec> {
        self.expect_symbol('{')?;
        let probe = if self.consume_keyword("probe") {
            Some(self.parse_route_stage()?)
        } else {
            None
        };
        self.expect_keyword("primary")?;
        let primary = self.parse_route_stage()?;
        let refinement = if self.consume_keyword("refinement") {
            Some(self.parse_route_stage()?)
        } else {
            None
        };
        self.expect_keyword("net-order")?;
        self.expect_symbol('[')?;
        let mut net_order = Vec::new();
        if !self.consume_symbol(']') {
            loop {
                net_order.push(match self.expect_word()?.as_str() {
                    "criticality" => NetOrderSpec::Criticality,
                    "highest-fanout-first" => NetOrderSpec::HighestFanoutFirst,
                    "reverse-criticality" => NetOrderSpec::ReverseCriticality,
                    value => eyre::bail!("unknown net order `{value}`"),
                });
                if self.consume_symbol(']') {
                    break;
                }
                self.expect_symbol(',')?;
            }
        }
        self.expect_symbol(';')?;
        self.expect_symbol('}')?;
        if net_order.is_empty() {
            eyre::bail!("PnR routing requires at least one net order");
        }
        Ok(RoutingSpec {
            probe,
            primary,
            refinement,
            net_order,
        })
    }

    fn parse_route_stage(&mut self) -> eyre::Result<RouteStageSpec> {
        self.expect_symbol('{')?;
        self.expect_keyword("strategy")?;
        let kind = self.expect_word()?;
        self.expect_symbol(';')?;
        let strategy = match kind.as_str() {
            "breadth-first" => RouteStrategySpec::BreadthFirst,
            "astar" => RouteStrategySpec::AStar,
            "direct-greedy" => {
                self.expect_keyword("max-steps")?;
                let max_steps = self.expect_usize()?;
                self.expect_symbol(';')?;
                RouteStrategySpec::DirectGreedy { max_steps }
            }
            "greedy-beam" => {
                self.expect_keyword("width")?;
                let width = self.expect_usize()?;
                self.expect_symbol(';')?;
                self.expect_keyword("max-expansions")?;
                let max_expansions = self.expect_usize()?;
                self.expect_symbol(';')?;
                self.expect_keyword("variant-seed")?;
                let variant_seed = self.expect_u64()?;
                self.expect_symbol(';')?;
                RouteStrategySpec::GreedyBeam {
                    width,
                    max_expansions,
                    variant_seed,
                }
            }
            _ => eyre::bail!("unknown route strategy `{kind}`"),
        };
        self.expect_keyword("validation")?;
        let validation = match self.expect_word()?.as_str() {
            "incremental" => RouteValidationSpec::Incremental,
            "deferred" => RouteValidationSpec::Deferred,
            value => eyre::bail!("unknown route validation mode `{value}`"),
        };
        self.expect_symbol(';')?;
        self.expect_symbol('}')?;
        Ok(RouteStageSpec {
            strategy,
            validation,
        })
    }

    fn parse_search_spec(&mut self) -> eyre::Result<SearchSpec> {
        self.expect_symbol('{')?;
        self.expect_keyword("candidates-per-child")?;
        let candidates_per_child = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("layout-combinations")?;
        let layout_combinations = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("detailed-routing-attempts")?;
        let detailed_routing_attempts = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("refined-routing-attempts")?;
        let refined_routing_attempts = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_keyword("refinement-rounds")?;
        let refinement_rounds = self.expect_usize()?;
        self.expect_symbol(';')?;
        self.expect_symbol('}')?;
        Ok(SearchSpec {
            candidates_per_child,
            layout_combinations,
            detailed_routing_attempts,
            refined_routing_attempts,
            refinement_rounds,
        })
    }

    fn parse_physical(&mut self) -> eyre::Result<PhysicalSpec> {
        self.expect_symbol('{')?;
        let mut regions = std::collections::BTreeMap::new();
        let mut constraints = Vec::new();
        while !self.consume_symbol('}') {
            if self.consume_keyword("region") {
                let name = self.expect_string()?;
                self.expect_keyword("box")?;
                let min = self.parse_position()?;
                let max = self.parse_position()?;
                self.expect_symbol(';')?;
                if regions
                    .insert(name.clone(), PhysicalRegionSpec { min, max })
                    .is_some()
                {
                    eyre::bail!("duplicate physical region `{name}`");
                }
                continue;
            }
            if self.consume_keyword("require") {
                let id = self.expect_string()?;
                if self.consume_keyword("instance") {
                    let instance = self.expect_string()?;
                    if self.consume_keyword("inside") {
                        let region = self.expect_string()?;
                        self.expect_symbol(';')?;
                        constraints.push(PhysicalConstraintSpec::Inside {
                            id,
                            instance,
                            region,
                        });
                    } else if self.consume_keyword("layer") {
                        let min = self.expect_usize()?;
                        self.expect_symbol('.')?;
                        self.expect_symbol('.')?;
                        let max = self.expect_usize()?;
                        self.expect_symbol(';')?;
                        constraints.push(PhysicalConstraintSpec::LayerRange {
                            id,
                            instance,
                            min,
                            max,
                        });
                    } else {
                        return Err(self.unexpected("`inside` or `layer`"));
                    }
                } else if self.consume_keyword("net") {
                    let net = self.expect_string()?;
                    self.expect_keyword("avoid")?;
                    let region = self.expect_string()?;
                    self.expect_symbol(';')?;
                    constraints.push(PhysicalConstraintSpec::NetAvoid { id, net, region });
                } else {
                    return Err(self.unexpected("`instance` or `net`"));
                }
                continue;
            }
            if self.consume_keyword("lock") {
                let id = self.expect_string()?;
                self.expect_keyword("instance")?;
                let instance = self.expect_string()?;
                self.expect_keyword("at")?;
                let origin = self.parse_position()?;
                self.expect_symbol(';')?;
                constraints.push(PhysicalConstraintSpec::FixedOrigin {
                    id,
                    instance,
                    origin,
                });
                continue;
            }
            if self.consume_keyword("priority") {
                let id = self.expect_string()?;
                self.expect_keyword("net")?;
                let net = self.expect_string()?;
                let priority = self.expect_usize()?;
                self.expect_symbol(';')?;
                constraints.push(PhysicalConstraintSpec::NetPriority { id, net, priority });
                continue;
            }
            if self.consume_keyword("prefer") {
                let id = self.expect_string()?;
                if self.consume_keyword("instance") {
                    let instance = self.expect_string()?;
                    self.expect_keyword("inside")?;
                    let region = self.expect_string()?;
                    self.expect_keyword("strength")?;
                    let strength = self.parse_preference()?;
                    self.expect_symbol(';')?;
                    constraints.push(PhysicalConstraintSpec::PreferInside {
                        id,
                        instance,
                        region,
                        strength,
                    });
                } else if self.consume_keyword("instances") {
                    let first = self.expect_string()?;
                    let second = self.expect_string()?;
                    self.expect_keyword("same-layer")?;
                    self.expect_keyword("strength")?;
                    let strength = self.parse_preference()?;
                    self.expect_symbol(';')?;
                    constraints.push(PhysicalConstraintSpec::SameLayer {
                        id,
                        first,
                        second,
                        strength,
                    });
                } else {
                    return Err(self.unexpected("`instance` or `instances`"));
                }
                continue;
            }
            return Err(self.unexpected("physical declaration"));
        }
        Ok(PhysicalSpec {
            regions,
            constraints,
        })
    }

    fn parse_leaf(
        &mut self,
        pin_search: &mut std::collections::BTreeMap<PortRef, Vec<[usize; 3]>>,
    ) -> eyre::Result<RoutableModule> {
        let name = self.expect_string()?;
        self.expect_symbol('{')?;
        let mut ports = Vec::new();
        let mut nodes = Vec::new();
        while !self.consume_symbol('}') {
            let annotation = if self.consume_symbol('@') {
                Some(self.parse_port_annotation()?)
            } else {
                None
            };
            if self.consume_keyword("port") {
                let port = self.parse_port()?;
                if let Some(positions) = annotation {
                    if port.direction != RoutablePortDirection::Input {
                        eyre::bail!(
                            "@pnr.pin_search may annotate only input ports; `{}.{}` is an output",
                            name,
                            port.name
                        );
                    }
                    let key = PortRef {
                        definition: name.clone(),
                        port: port.name.clone(),
                    };
                    if pin_search.insert(key, positions).is_some() {
                        eyre::bail!("duplicate @pnr.pin_search on `{}.{}`", name, port.name);
                    }
                }
                ports.push(port);
            } else if self.consume_keyword("node") {
                if annotation.is_some() {
                    eyre::bail!("@pnr.pin_search may annotate only input ports");
                }
                nodes.push(self.parse_node()?);
            } else {
                return Err(self.unexpected("port annotation, `port`, `node`, or `}`"));
            }
        }
        Ok(RoutableModule {
            name,
            ports,
            body: RoutableModuleBody::Leaf { nodes },
        })
    }

    fn parse_port_annotation(&mut self) -> eyre::Result<Vec<[usize; 3]>> {
        self.expect_keyword("pnr")?;
        self.expect_symbol('.')?;
        let annotation = self.expect_word()?;
        if annotation != "pin_search" {
            eyre::bail!("unknown PnR annotation `@pnr.{annotation}`");
        }
        self.expect_symbol('(')?;
        self.expect_keyword("positions")?;
        self.expect_symbol('=')?;
        let positions = self.parse_position_list()?;
        if positions.is_empty() {
            eyre::bail!("@pnr.pin_search requires at least one position");
        }
        self.expect_symbol(')')?;
        Ok(positions)
    }

    fn parse_composite(&mut self) -> eyre::Result<RoutableModule> {
        let name = self.expect_string()?;
        self.expect_symbol('{')?;
        let mut ports = Vec::new();
        let mut instances = Vec::new();
        let mut nets = Vec::new();
        while !self.consume_symbol('}') {
            if self.consume_keyword("port") {
                ports.push(self.parse_port()?);
            } else if self.consume_keyword("instance") {
                instances.push(self.parse_instance()?);
            } else if self.consume_keyword("net") {
                nets.push(self.parse_net()?);
            } else {
                return Err(self.unexpected("`port`, `instance`, `net`, or `}`"));
            }
        }
        Ok(RoutableModule {
            name,
            ports,
            body: RoutableModuleBody::Composite { instances, nets },
        })
    }

    fn parse_port(&mut self) -> eyre::Result<RoutablePort> {
        let direction = if self.consume_keyword("input") {
            RoutablePortDirection::Input
        } else if self.consume_keyword("output") {
            RoutablePortDirection::Output
        } else {
            return Err(self.unexpected("`input` or `output`"));
        };
        let name = self.expect_string()?;
        self.expect_symbol(';')?;
        Ok(RoutablePort { name, direction })
    }

    fn parse_node(&mut self) -> eyre::Result<RoutableNode> {
        let id = self.expect_number()? as usize;
        let kind = if self.consume_keyword("input") {
            RoutableNodeKind::Input {
                name: self.expect_string()?,
            }
        } else if self.consume_keyword("output") {
            RoutableNodeKind::Output {
                name: self.expect_string()?,
            }
        } else if self.consume_keyword("logic") {
            match self.expect_word()?.as_str() {
                "not" => RoutableNodeKind::Not,
                "or" => RoutableNodeKind::Or,
                kind => eyre::bail!("unknown routable logic node `{kind}`"),
            }
        } else if self.consume_keyword("sequential") {
            let primitive = match self.expect_word()?.as_str() {
                "rs_latch" => RoutableSequentialPrimitive::RsLatch,
                "d_latch" => RoutableSequentialPrimitive::DLatch,
                kind => eyre::bail!("unknown routable sequential primitive `{kind}`"),
            };
            self.expect_keyword("input_ports")?;
            let input_ports = self.parse_string_list()?;
            self.expect_keyword("output_ports")?;
            let output_ports = self.parse_string_list()?;
            RoutableNodeKind::Sequential {
                primitive,
                input_ports,
                output_ports,
            }
        } else {
            return Err(self.unexpected("routable node kind"));
        };
        self.expect_keyword("inputs")?;
        let inputs = self.parse_number_list()?;
        let tag = if self.consume_keyword("tag") {
            self.expect_string()?
        } else {
            String::new()
        };
        self.expect_symbol(';')?;
        Ok(RoutableNode {
            id,
            kind,
            inputs,
            tag,
        })
    }

    fn parse_instance(&mut self) -> eyre::Result<RoutableInstance> {
        let name = self.expect_string()?;
        self.expect_symbol(':')?;
        let module = self.expect_string()?;
        let origin = if self.consume_keyword("origin") {
            Some(self.expect_string()?)
        } else {
            None
        };
        self.expect_symbol(';')?;
        Ok(RoutableInstance {
            name,
            module,
            origin,
        })
    }

    fn parse_net(&mut self) -> eyre::Result<RoutableNet> {
        let name = self.expect_string()?;
        self.expect_keyword("class")?;
        let class = match self.expect_word()?.as_str() {
            "data" => NetClass::Data,
            "clock" => NetClass::Clock,
            "reset" => NetClass::Reset,
            "io" => NetClass::Io,
            class => eyre::bail!("unknown routable net class `{class}`"),
        };
        self.expect_keyword("driver")?;
        let driver = self.parse_endpoint()?;
        self.expect_keyword("sinks")?;
        self.expect_symbol('[')?;
        let mut sinks = Vec::new();
        if !self.consume_symbol(']') {
            loop {
                sinks.push(self.parse_endpoint()?);
                if self.consume_symbol(']') {
                    break;
                }
                self.expect_symbol(',')?;
            }
        }
        let origin = if self.consume_keyword("origin") {
            Some(self.expect_string()?)
        } else {
            None
        };
        self.expect_symbol(';')?;
        Ok(RoutableNet {
            name,
            class,
            driver,
            sinks,
            origin,
        })
    }

    fn parse_endpoint(&mut self) -> eyre::Result<Endpoint> {
        if self.consume_keyword("self") {
            self.expect_symbol('.')?;
            return Ok(Endpoint::SelfPort {
                port: self.expect_string()?,
            });
        }
        let instance = self.expect_string()?;
        self.expect_symbol('.')?;
        Ok(Endpoint::InstancePort {
            instance,
            port: self.expect_string()?,
        })
    }

    fn parse_string_list(&mut self) -> eyre::Result<Vec<String>> {
        self.expect_symbol('[')?;
        let mut values = Vec::new();
        if self.consume_symbol(']') {
            return Ok(values);
        }
        loop {
            values.push(self.expect_string()?);
            if self.consume_symbol(']') {
                return Ok(values);
            }
            self.expect_symbol(',')?;
        }
    }

    fn parse_number_list(&mut self) -> eyre::Result<Vec<usize>> {
        self.expect_symbol('[')?;
        let mut values = Vec::new();
        if self.consume_symbol(']') {
            return Ok(values);
        }
        loop {
            values.push(self.expect_number()? as usize);
            if self.consume_symbol(']') {
                return Ok(values);
            }
            self.expect_symbol(',')?;
        }
    }

    fn peek_keyword(&self, expected: &str) -> bool {
        matches!(self.tokens.get(self.position), Some(Token::Word(word)) if word == expected)
    }

    fn expect_usize(&mut self) -> eyre::Result<usize> {
        self.expect_integer("usize")
    }

    fn expect_u64(&mut self) -> eyre::Result<u64> {
        self.expect_integer("u64")
    }

    fn expect_integer<T>(&mut self, name: &str) -> eyre::Result<T>
    where
        T: std::str::FromStr,
    {
        match self.tokens.get(self.position).cloned() {
            Some(Token::Number(value)) => {
                let value = value.parse::<T>().map_err(|_| self.unexpected(name))?;
                self.position += 1;
                Ok(value)
            }
            _ => Err(self.unexpected(name)),
        }
    }

    fn expect_bool(&mut self) -> eyre::Result<bool> {
        if self.consume_keyword("true") {
            Ok(true)
        } else if self.consume_keyword("false") {
            Ok(false)
        } else {
            Err(self.unexpected("`true` or `false`"))
        }
    }

    fn expect_f64(&mut self) -> eyre::Result<f64> {
        let integer = match self.tokens.get(self.position).cloned() {
            Some(Token::Number(value)) => {
                self.position += 1;
                value
            }
            _ => return Err(self.unexpected("floating-point number")),
        };
        let text = if self.consume_symbol('.') {
            let fraction = match self.tokens.get(self.position).cloned() {
                Some(Token::Number(value)) => {
                    self.position += 1;
                    value
                }
                _ => return Err(self.unexpected("fractional digits")),
            };
            format!("{integer}.{fraction}")
        } else {
            integer
        };
        text.parse::<f64>()
            .map_err(|_| self.unexpected("floating-point number"))
    }

    fn parse_optional_usize(&mut self) -> eyre::Result<Option<usize>> {
        if self.consume_keyword("none") {
            Ok(None)
        } else {
            Ok(Some(self.expect_usize()?))
        }
    }

    fn parse_sampling(&mut self) -> eyre::Result<SamplingSpec> {
        let kind = self.expect_word()?;
        if kind == "none" {
            return Ok(SamplingSpec::None);
        }
        self.expect_symbol('(')?;
        let count = self.expect_usize()?;
        self.expect_symbol(')')?;
        match kind.as_str() {
            "take" => Ok(SamplingSpec::Take(count)),
            "random" => Ok(SamplingSpec::Random(count)),
            _ => eyre::bail!("unknown sampling mode `{kind}`"),
        }
    }

    fn parse_placement_sampling(&mut self) -> eyre::Result<PlacementSamplingSpec> {
        let kind = self.expect_word()?;
        if kind == "step-policy" {
            return Ok(PlacementSamplingSpec::StepPolicy);
        }
        self.expect_symbol('(')?;
        let count = self.expect_usize()?;
        self.expect_symbol(',')?;
        let random_count = self.expect_usize()?;
        self.expect_symbol(',')?;
        let start_step = self.expect_usize()?;
        self.expect_symbol(')')?;
        match kind.as_str() {
            "cost" => Ok(PlacementSamplingSpec::Cost {
                count,
                random_count,
                start_step,
            }),
            "ranked" => Ok(PlacementSamplingSpec::Ranked {
                count,
                random_count,
                start_step,
            }),
            _ => eyre::bail!("unknown placement sampling mode `{kind}`"),
        }
    }

    fn parse_position(&mut self) -> eyre::Result<[usize; 3]> {
        self.expect_symbol('[')?;
        let x = self.expect_usize()?;
        self.expect_symbol(',')?;
        let y = self.expect_usize()?;
        self.expect_symbol(',')?;
        let z = self.expect_usize()?;
        self.expect_symbol(']')?;
        Ok([x, y, z])
    }

    fn parse_position_list(&mut self) -> eyre::Result<Vec<[usize; 3]>> {
        self.expect_symbol('[')?;
        let mut values = Vec::new();
        if self.consume_symbol(']') {
            return Ok(values);
        }
        loop {
            values.push(self.parse_position()?);
            if self.consume_symbol(']') {
                return Ok(values);
            }
            self.expect_symbol(',')?;
        }
    }

    fn parse_u64_list(&mut self) -> eyre::Result<Vec<u64>> {
        self.expect_symbol('[')?;
        let mut values = Vec::new();
        if self.consume_symbol(']') {
            return Ok(values);
        }
        loop {
            values.push(self.expect_u64()?);
            if self.consume_symbol(']') {
                return Ok(values);
            }
            self.expect_symbol(',')?;
        }
    }

    fn parse_usize_list(&mut self) -> eyre::Result<Vec<usize>> {
        self.expect_symbol('[')?;
        let mut values = Vec::new();
        if self.consume_symbol(']') {
            return Ok(values);
        }
        loop {
            values.push(self.expect_usize()?);
            if self.consume_symbol(']') {
                return Ok(values);
            }
            self.expect_symbol(',')?;
        }
    }

    fn parse_preference(&mut self) -> eyre::Result<PreferenceSpec> {
        match self.expect_word()?.as_str() {
            "weak" => Ok(PreferenceSpec::Weak),
            "medium" => Ok(PreferenceSpec::Medium),
            "strong" => Ok(PreferenceSpec::Strong),
            value => eyre::bail!("unknown preference strength `{value}`"),
        }
    }

    fn expect_keyword(&mut self, expected: &str) -> eyre::Result<()> {
        if self.consume_keyword(expected) {
            Ok(())
        } else {
            Err(self.unexpected(&format!("`{expected}`")))
        }
    }

    fn consume_keyword(&mut self, expected: &str) -> bool {
        if matches!(self.tokens.get(self.position), Some(Token::Word(word)) if word == expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn expect_word(&mut self) -> eyre::Result<String> {
        match self.tokens.get(self.position).cloned() {
            Some(Token::Word(value)) => {
                self.position += 1;
                Ok(value)
            }
            _ => Err(self.unexpected("word")),
        }
    }

    fn expect_string(&mut self) -> eyre::Result<String> {
        match self.tokens.get(self.position).cloned() {
            Some(Token::String(value)) => {
                self.position += 1;
                Ok(value)
            }
            _ => Err(self.unexpected("quoted string")),
        }
    }

    fn expect_number(&mut self) -> eyre::Result<u32> {
        match self.tokens.get(self.position).cloned() {
            Some(Token::Number(value)) => {
                let value = value
                    .parse::<u32>()
                    .map_err(|_| self.unexpected("32-bit number"))?;
                self.position += 1;
                Ok(value)
            }
            _ => Err(self.unexpected("number")),
        }
    }

    fn expect_symbol(&mut self, expected: char) -> eyre::Result<()> {
        if self.consume_symbol(expected) {
            Ok(())
        } else {
            Err(self.unexpected(&format!("`{expected}`")))
        }
    }

    fn consume_symbol(&mut self, expected: char) -> bool {
        if self.tokens.get(self.position) == Some(&Token::Symbol(expected)) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn unexpected(&self, expected: &str) -> eyre::Report {
        eyre::eyre!(
            "expected {expected} at token {}, found {:?}",
            self.position,
            self.tokens.get(self.position)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ROUTABLE_IR_TARGET, ROUTABLE_IR_VERSION};

    #[test]
    fn routable_text_round_trips_deterministically() -> eyre::Result<()> {
        let design = sample_design();
        design.validate()?;
        let first = design.to_string();
        let reparsed: RoutableDesign = first.parse()?;
        let second = reparsed.to_string();

        assert_eq!(reparsed, design);
        assert_eq!(second, first);
        assert!(first.contains("stage routable;"));
        assert!(first.contains("driver self.\"a\""));
        Ok(())
    }

    #[test]
    fn routable_text_rejects_wrong_endpoint_direction() {
        let mut design = sample_design();
        let RoutableModuleBody::Composite { nets, .. } = &mut design.modules[1].body else {
            unreachable!();
        };
        nets[0].driver = Endpoint::SelfPort {
            port: "y".to_owned(),
        };

        assert!(format!("{:#}", design.validate().unwrap_err()).contains("invalid driver"));
    }

    #[test]
    fn routable_text_rejects_unmapped_logic_nodes() {
        let source = sample_design()
            .to_string()
            .replace("logic not", "logic xor");

        let error = source.parse::<RoutableDesign>().unwrap_err();
        assert!(format!("{error:#}").contains("unknown routable logic node `xor`"));
    }

    fn sample_design() -> RoutableDesign {
        RoutableDesign {
            version: ROUTABLE_IR_VERSION,
            target: ROUTABLE_IR_TARGET.to_owned(),
            top: "top".to_owned(),
            debug: Default::default(),
            modules: vec![
                RoutableModule {
                    name: "inv".to_owned(),
                    ports: vec![
                        RoutablePort {
                            name: "a".to_owned(),
                            direction: RoutablePortDirection::Input,
                        },
                        RoutablePort {
                            name: "y".to_owned(),
                            direction: RoutablePortDirection::Output,
                        },
                    ],
                    body: RoutableModuleBody::Leaf {
                        nodes: vec![
                            RoutableNode {
                                id: 0,
                                kind: RoutableNodeKind::Input {
                                    name: "a".to_owned(),
                                },
                                inputs: vec![],
                                tag: String::new(),
                            },
                            RoutableNode {
                                id: 1,
                                kind: RoutableNodeKind::Not,
                                inputs: vec![0],
                                tag: "generated".to_owned(),
                            },
                            RoutableNode {
                                id: 2,
                                kind: RoutableNodeKind::Output {
                                    name: "y".to_owned(),
                                },
                                inputs: vec![1],
                                tag: String::new(),
                            },
                        ],
                    },
                },
                RoutableModule {
                    name: "top".to_owned(),
                    ports: vec![
                        RoutablePort {
                            name: "a".to_owned(),
                            direction: RoutablePortDirection::Input,
                        },
                        RoutablePort {
                            name: "y".to_owned(),
                            direction: RoutablePortDirection::Output,
                        },
                    ],
                    body: RoutableModuleBody::Composite {
                        instances: vec![RoutableInstance {
                            name: "u0".to_owned(),
                            module: "inv".to_owned(),
                            origin: Some("logical.cell.0".to_owned()),
                        }],
                        nets: vec![
                            RoutableNet {
                                name: "a".to_owned(),
                                class: NetClass::Io,
                                driver: Endpoint::SelfPort {
                                    port: "a".to_owned(),
                                },
                                sinks: vec![Endpoint::InstancePort {
                                    instance: "u0".to_owned(),
                                    port: "a".to_owned(),
                                }],
                                origin: None,
                            },
                            RoutableNet {
                                name: "y".to_owned(),
                                class: NetClass::Io,
                                driver: Endpoint::InstancePort {
                                    instance: "u0".to_owned(),
                                    port: "y".to_owned(),
                                },
                                sinks: vec![Endpoint::SelfPort {
                                    port: "y".to_owned(),
                                }],
                                origin: None,
                            },
                        ],
                    },
                },
            ],
        }
    }
}
