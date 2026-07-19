use std::collections::{BTreeMap, HashSet};
use std::str::FromStr;

use eyre::{Context, ContextCompat};
use serde::{Deserialize, Serialize};

use crate::transform::place_and_route::global_pnr::ir::LayoutCandidate;
use crate::transform::place_and_route::global_pnr::placer::PlacedModule;
use crate::transform::place_and_route::global_pnr::router::RoutedNet;
use crate::transform::place_and_route::global_pnr::topology::{
    InstanceId, NetId, ResolvedPnrTopology,
};

pub const PHYSICAL_INTENT_FORMAT: &str = "redstone-compiler.physical-intent.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalIntent {
    pub format: String,
    pub design: String,
    pub regions: BTreeMap<String, IntentRegion>,
    pub constraints: Vec<PhysicalConstraint>,
}

impl PhysicalIntent {
    pub fn bind(&self, topology: &ResolvedPnrTopology) -> eyre::Result<ResolvedPhysicalIntent> {
        let top = topology
            .definition(topology.top)
            .context("resolved topology is missing its top definition")?;
        if self.design != top.display_name && self.design != top.key.0 {
            eyre::bail!(
                "physical intent targets design `{}`, but the resolved top is `{}`",
                self.design,
                top.display_name
            );
        }

        let mut ids = HashSet::new();
        let constraints = self
            .constraints
            .iter()
            .map(|constraint| {
                if !ids.insert(constraint.id().to_owned()) {
                    eyre::bail!("duplicate physical constraint id `{}`", constraint.id());
                }
                self.bind_constraint(topology, constraint)
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        let resolved = ResolvedPhysicalIntent {
            format: PHYSICAL_INTENT_FORMAT.to_owned(),
            design: self.design.clone(),
            regions: self.regions.clone(),
            constraints,
        };
        resolved.validate(topology)?;
        Ok(resolved)
    }

    fn bind_constraint(
        &self,
        topology: &ResolvedPnrTopology,
        constraint: &PhysicalConstraint,
    ) -> eyre::Result<ResolvedPhysicalConstraint> {
        let resolve_instance = |name: &str| {
            topology
                .instances
                .iter()
                .find(|instance| instance.display_name == name || instance.key.0 == name)
                .map(|instance| instance.id)
                .with_context(|| format!("unknown physical-intent instance `{name}`"))
        };
        let resolve_net = |name: &str| {
            topology
                .nets
                .iter()
                .find(|net| net.display_name == name || net.key.0 == name)
                .map(|net| net.id)
                .with_context(|| format!("unknown physical-intent net `{name}`"))
        };

        Ok(match constraint {
            PhysicalConstraint::Inside {
                id,
                instance,
                region,
            } => {
                if !self.regions.contains_key(region) {
                    eyre::bail!("constraint `{id}` references unknown region `{region}`");
                }
                ResolvedPhysicalConstraint::Inside {
                    id: id.clone(),
                    instance: resolve_instance(instance)?,
                    region: region.clone(),
                }
            }
            PhysicalConstraint::LayerRange {
                id,
                instance,
                min,
                max,
            } => ResolvedPhysicalConstraint::LayerRange {
                id: id.clone(),
                instance: resolve_instance(instance)?,
                min: *min,
                max: *max,
            },
            PhysicalConstraint::FixedOrigin {
                id,
                instance,
                origin,
            } => ResolvedPhysicalConstraint::FixedOrigin {
                id: id.clone(),
                instance: resolve_instance(instance)?,
                origin: *origin,
            },
            PhysicalConstraint::NetPriority { id, net, priority } => {
                ResolvedPhysicalConstraint::NetPriority {
                    id: id.clone(),
                    net: resolve_net(net)?,
                    priority: *priority,
                }
            }
            PhysicalConstraint::NetAvoid { id, net, region } => {
                if !self.regions.contains_key(region) {
                    eyre::bail!("constraint `{id}` references unknown region `{region}`");
                }
                ResolvedPhysicalConstraint::NetAvoid {
                    id: id.clone(),
                    net: resolve_net(net)?,
                    region: region.clone(),
                }
            }
            PhysicalConstraint::PreferInside {
                id,
                instance,
                region,
                strength,
            } => {
                if !self.regions.contains_key(region) {
                    eyre::bail!("constraint `{id}` references unknown region `{region}`");
                }
                ResolvedPhysicalConstraint::PreferInside {
                    id: id.clone(),
                    instance: resolve_instance(instance)?,
                    region: region.clone(),
                    strength: *strength,
                }
            }
            PhysicalConstraint::SameLayer {
                id,
                first,
                second,
                strength,
            } => ResolvedPhysicalConstraint::SameLayer {
                id: id.clone(),
                first: resolve_instance(first)?,
                second: resolve_instance(second)?,
                strength: *strength,
            },
        })
    }
}

impl FromStr for PhysicalIntent {
    type Err = eyre::Report;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        parse_physical_intent(source)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntentRegion {
    pub min: [usize; 3],
    pub max: [usize; 3],
}

impl IntentRegion {
    pub fn contains_box(&self, min: [usize; 3], max: [usize; 3]) -> bool {
        (0..3).all(|axis| min[axis] >= self.min[axis] && max[axis] <= self.max[axis])
    }

    pub fn contains_point(&self, point: [usize; 3]) -> bool {
        (0..3).all(|axis| point[axis] >= self.min[axis] && point[axis] <= self.max[axis])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhysicalConstraint {
    Inside {
        id: String,
        instance: String,
        region: String,
    },
    LayerRange {
        id: String,
        instance: String,
        min: usize,
        max: usize,
    },
    FixedOrigin {
        id: String,
        instance: String,
        origin: [usize; 3],
    },
    NetPriority {
        id: String,
        net: String,
        priority: usize,
    },
    NetAvoid {
        id: String,
        net: String,
        region: String,
    },
    PreferInside {
        id: String,
        instance: String,
        region: String,
        strength: PreferenceStrength,
    },
    SameLayer {
        id: String,
        first: String,
        second: String,
        strength: PreferenceStrength,
    },
}

impl PhysicalConstraint {
    pub fn id(&self) -> &str {
        match self {
            Self::Inside { id, .. }
            | Self::LayerRange { id, .. }
            | Self::FixedOrigin { id, .. }
            | Self::NetPriority { id, .. }
            | Self::NetAvoid { id, .. }
            | Self::PreferInside { id, .. }
            | Self::SameLayer { id, .. } => id,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPhysicalIntent {
    pub format: String,
    pub design: String,
    pub regions: BTreeMap<String, IntentRegion>,
    pub constraints: Vec<ResolvedPhysicalConstraint>,
}

impl ResolvedPhysicalIntent {
    pub fn validate(&self, topology: &ResolvedPnrTopology) -> eyre::Result<()> {
        let top = topology
            .definition(topology.top)
            .context("resolved topology is missing its top definition")?;
        if self.design != top.display_name && self.design != top.key.0 {
            eyre::bail!(
                "resolved physical intent targets `{}`, not `{}`",
                self.design,
                top.display_name
            );
        }
        for (name, region) in &self.regions {
            if (0..3).any(|axis| region.min[axis] > region.max[axis]) {
                eyre::bail!("physical region `{name}` has an inverted axis range");
            }
        }

        let mut fixed = BTreeMap::<InstanceId, ([usize; 3], &str)>::new();
        for constraint in &self.constraints {
            match constraint {
                ResolvedPhysicalConstraint::Inside {
                    id,
                    instance,
                    region,
                } => {
                    topology
                        .instances
                        .get(instance.0)
                        .with_context(|| format!("constraint `{id}` has an unknown instance"))?;
                    if !self.regions.contains_key(region) {
                        eyre::bail!("constraint `{id}` has an unknown region `{region}`");
                    }
                }
                ResolvedPhysicalConstraint::LayerRange {
                    id,
                    instance,
                    min,
                    max,
                } => {
                    topology
                        .instances
                        .get(instance.0)
                        .with_context(|| format!("constraint `{id}` has an unknown instance"))?;
                    if min > max {
                        eyre::bail!("constraint `{id}` has an inverted layer range");
                    }
                }
                ResolvedPhysicalConstraint::FixedOrigin {
                    id,
                    instance,
                    origin,
                } => {
                    topology
                        .instances
                        .get(instance.0)
                        .with_context(|| format!("constraint `{id}` has an unknown instance"))?;
                    if let Some((previous, previous_id)) = fixed.insert(*instance, (*origin, id))
                        && previous != *origin
                    {
                        eyre::bail!(
                            "constraints `{previous_id}` and `{id}` lock the same instance to different origins"
                        );
                    }
                }
                ResolvedPhysicalConstraint::NetPriority { id, net, .. }
                | ResolvedPhysicalConstraint::NetAvoid { id, net, .. } => {
                    topology
                        .nets
                        .get(net.0)
                        .with_context(|| format!("constraint `{id}` has an unknown net"))?;
                    if let ResolvedPhysicalConstraint::NetAvoid { region, .. } = constraint
                        && !self.regions.contains_key(region)
                    {
                        eyre::bail!("constraint `{id}` has an unknown region `{region}`");
                    }
                }
                ResolvedPhysicalConstraint::PreferInside {
                    id,
                    instance,
                    region,
                    ..
                } => {
                    topology
                        .instances
                        .get(instance.0)
                        .with_context(|| format!("constraint `{id}` has an unknown instance"))?;
                    if !self.regions.contains_key(region) {
                        eyre::bail!("constraint `{id}` has an unknown region `{region}`");
                    }
                }
                ResolvedPhysicalConstraint::SameLayer {
                    id, first, second, ..
                } => {
                    topology.instances.get(first.0).with_context(|| {
                        format!("constraint `{id}` has an unknown first instance")
                    })?;
                    topology.instances.get(second.0).with_context(|| {
                        format!("constraint `{id}` has an unknown second instance")
                    })?;
                    if first == second {
                        eyre::bail!("constraint `{id}` compares an instance with itself");
                    }
                }
            }
        }
        Ok(())
    }

    pub fn constraints_for_instance(
        &self,
        instance: InstanceId,
    ) -> impl Iterator<Item = &ResolvedPhysicalConstraint> {
        self.constraints
            .iter()
            .filter(move |constraint| match constraint {
                ResolvedPhysicalConstraint::Inside { instance: id, .. }
                | ResolvedPhysicalConstraint::LayerRange { instance: id, .. }
                | ResolvedPhysicalConstraint::FixedOrigin { instance: id, .. }
                | ResolvedPhysicalConstraint::PreferInside { instance: id, .. } => *id == instance,
                ResolvedPhysicalConstraint::SameLayer { first, second, .. } => {
                    *first == instance || *second == instance
                }
                _ => false,
            })
    }

    pub fn net_priority(&self, net: NetId) -> usize {
        self.constraints
            .iter()
            .filter_map(|constraint| match constraint {
                ResolvedPhysicalConstraint::NetPriority {
                    net: id, priority, ..
                } if *id == net => Some(*priority),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    pub fn placement_preference_cost(
        &self,
        topology: &ResolvedPnrTopology,
        candidates: &[LayoutCandidate],
        placed: &[PlacedModule],
    ) -> PreferenceCost {
        let mut cost = PreferenceCost::default();
        for constraint in &self.constraints {
            match constraint {
                ResolvedPhysicalConstraint::PreferInside {
                    instance,
                    region,
                    strength,
                    ..
                } => {
                    let Some((min, max)) = placed_bounds(topology, candidates, placed, *instance)
                    else {
                        cost.add(*strength, usize::MAX / 4);
                        continue;
                    };
                    let Some(region) = self.regions.get(region) else {
                        cost.add(*strength, usize::MAX / 4);
                        continue;
                    };
                    let distance = (0..3)
                        .map(|axis| {
                            region.min[axis].saturating_sub(min[axis])
                                + max[axis].saturating_sub(region.max[axis])
                        })
                        .sum();
                    cost.add(*strength, distance);
                }
                ResolvedPhysicalConstraint::SameLayer {
                    first,
                    second,
                    strength,
                    ..
                } => {
                    let first =
                        placed_bounds(topology, candidates, placed, *first).map(|(min, _)| min[2]);
                    let second =
                        placed_bounds(topology, candidates, placed, *second).map(|(min, _)| min[2]);
                    cost.add(
                        *strength,
                        first
                            .zip(second)
                            .map_or(usize::MAX / 4, |(left, right)| left.abs_diff(right)),
                    );
                }
                _ => {}
            }
        }
        cost
    }

    pub fn evaluate(
        &self,
        topology: &ResolvedPnrTopology,
        candidates: &[LayoutCandidate],
        placed: &[PlacedModule],
        routes: &[RoutedNet],
    ) -> Vec<ConstraintSatisfaction> {
        self.constraints
            .iter()
            .map(|constraint| {
                let (satisfied, detail) = match constraint {
                    ResolvedPhysicalConstraint::Inside {
                        instance, region, ..
                    } => {
                        let bounds = placed_bounds(topology, candidates, placed, *instance);
                        match (self.regions.get(region), bounds) {
                            (Some(region), Some((min, max))) => (
                                region.contains_box(min, max),
                                format!(
                                    "instance {:?} bbox {min:?}..{max:?} inside `{region:?}`",
                                    instance
                                ),
                            ),
                            _ => (false, "instance or region was not available".to_owned()),
                        }
                    }
                    ResolvedPhysicalConstraint::LayerRange {
                        instance, min, max, ..
                    } => match placed_bounds(topology, candidates, placed, *instance) {
                        Some((placed_min, placed_max)) => (
                            placed_min[2] >= *min && placed_max[2] <= *max,
                            format!(
                                "instance {:?} occupies z={}..{}; allowed={min}..{max}",
                                instance, placed_min[2], placed_max[2]
                            ),
                        ),
                        None => (false, "instance was not placed".to_owned()),
                    },
                    ResolvedPhysicalConstraint::FixedOrigin {
                        instance, origin, ..
                    } => match placed_bounds(topology, candidates, placed, *instance) {
                        Some((placed_min, _)) => (
                            placed_min == *origin,
                            format!(
                                "instance {:?} origin={placed_min:?}; required={origin:?}",
                                instance
                            ),
                        ),
                        None => (false, "instance was not placed".to_owned()),
                    },
                    ResolvedPhysicalConstraint::NetPriority { net, priority, .. } => (
                        true,
                        format!("net {:?} routed with priority {priority}", net),
                    ),
                    ResolvedPhysicalConstraint::NetAvoid { net, region, .. } => {
                        let Some(region) = self.regions.get(region) else {
                            return ConstraintSatisfaction {
                                id: constraint.id().to_owned(),
                                status: ConstraintStatus::Violated,
                                detail: "avoid region was not available".to_owned(),
                            };
                        };
                        let intersections = routes
                            .iter()
                            .filter(|route| route.net_id == Some(*net))
                            .flat_map(|route| route.path.iter().copied())
                            .filter(|position| {
                                region.contains_point([position.0, position.1, position.2])
                            })
                            .count();
                        (
                            intersections == 0,
                            format!(
                                "net {:?} has {intersections} path point(s) in avoid region",
                                net
                            ),
                        )
                    }
                    ResolvedPhysicalConstraint::PreferInside {
                        instance,
                        region,
                        strength,
                        ..
                    } => {
                        let value = placed_bounds(topology, candidates, placed, *instance)
                            .zip(self.regions.get(region))
                            .map_or(usize::MAX, |((min, max), region)| {
                                (0..3)
                                    .map(|axis| {
                                        region.min[axis].saturating_sub(min[axis])
                                            + max[axis].saturating_sub(region.max[axis])
                                    })
                                    .sum()
                            });
                        (
                            value == 0,
                            format!(
                                "instance {:?} preferred inside `{region}` at {strength:?} strength; aggregate penalty={value}",
                                instance
                            ),
                        )
                    }
                    ResolvedPhysicalConstraint::SameLayer {
                        first,
                        second,
                        strength,
                        ..
                    } => {
                        let first_z = placed_bounds(topology, candidates, placed, *first)
                            .map(|(min, _)| min[2]);
                        let second_z = placed_bounds(topology, candidates, placed, *second)
                            .map(|(min, _)| min[2]);
                        let distance = first_z
                            .zip(second_z)
                            .map_or(usize::MAX, |(left, right)| left.abs_diff(right));
                        (
                            distance == 0,
                            format!(
                                "instances {:?} and {:?} preferred same layer at {strength:?} strength; z distance={distance}",
                                first, second
                            ),
                        )
                    }
                };
                ConstraintSatisfaction {
                    id: constraint.id().to_owned(),
                    status: if satisfied {
                        ConstraintStatus::Satisfied
                    } else {
                        ConstraintStatus::Violated
                    },
                    detail,
                }
            })
            .collect()
    }

    pub fn validate_routes(&self, routes: &[RoutedNet]) -> eyre::Result<()> {
        for constraint in &self.constraints {
            let ResolvedPhysicalConstraint::NetAvoid {
                id, net, region, ..
            } = constraint
            else {
                continue;
            };
            let region = self
                .regions
                .get(region)
                .with_context(|| format!("constraint `{id}` has an unknown avoid region"))?;
            if let Some(position) = routes
                .iter()
                .filter(|route| route.net_id == Some(*net))
                .flat_map(|route| route.path.iter().copied())
                .find(|position| region.contains_point([position.0, position.1, position.2]))
            {
                eyre::bail!(
                    "routing violates physical constraint `{id}`: net {:?} enters avoid region at {:?}",
                    net,
                    position
                );
            }
        }
        Ok(())
    }
}

fn placed_bounds(
    topology: &ResolvedPnrTopology,
    candidates: &[LayoutCandidate],
    placed: &[PlacedModule],
    instance: InstanceId,
) -> Option<([usize; 3], [usize; 3])> {
    let instance = topology.instances.get(instance.0)?;
    let placed = placed
        .iter()
        .find(|placed| placed.module_name == instance.display_name)?;
    let candidate = candidates.get(placed.candidate_index)?;
    let min = [placed.origin.0, placed.origin.1, placed.origin.2];
    let max = [
        placed.origin.0 + candidate.bbox.width() - 1,
        placed.origin.1 + candidate.bbox.depth() - 1,
        placed.origin.2 + candidate.bbox.height() - 1,
    ];
    Some((min, max))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedPhysicalConstraint {
    Inside {
        id: String,
        instance: InstanceId,
        region: String,
    },
    LayerRange {
        id: String,
        instance: InstanceId,
        min: usize,
        max: usize,
    },
    FixedOrigin {
        id: String,
        instance: InstanceId,
        origin: [usize; 3],
    },
    NetPriority {
        id: String,
        net: NetId,
        priority: usize,
    },
    NetAvoid {
        id: String,
        net: NetId,
        region: String,
    },
    PreferInside {
        id: String,
        instance: InstanceId,
        region: String,
        strength: PreferenceStrength,
    },
    SameLayer {
        id: String,
        first: InstanceId,
        second: InstanceId,
        strength: PreferenceStrength,
    },
}

impl ResolvedPhysicalConstraint {
    pub fn id(&self) -> &str {
        match self {
            Self::Inside { id, .. }
            | Self::LayerRange { id, .. }
            | Self::FixedOrigin { id, .. }
            | Self::NetPriority { id, .. }
            | Self::NetAvoid { id, .. }
            | Self::PreferInside { id, .. }
            | Self::SameLayer { id, .. } => id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceStrength {
    Strong,
    Medium,
    Weak,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PreferenceCost {
    pub strong: usize,
    pub medium: usize,
    pub weak: usize,
}

impl PreferenceCost {
    fn add(&mut self, strength: PreferenceStrength, value: usize) {
        match strength {
            PreferenceStrength::Strong => self.strong = self.strong.saturating_add(value),
            PreferenceStrength::Medium => self.medium = self.medium.saturating_add(value),
            PreferenceStrength::Weak => self.weak = self.weak.saturating_add(value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstraintSatisfaction {
    pub id: String,
    pub status: ConstraintStatus,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintStatus {
    Satisfied,
    Violated,
    NotEvaluated,
}

fn parse_physical_intent(source: &str) -> eyre::Result<PhysicalIntent> {
    let source = source
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n");
    let statements = source
        .split(';')
        .map(str::trim)
        .filter(|statement| !statement.is_empty())
        .collect::<Vec<_>>();
    let mut design = None;
    let mut regions = BTreeMap::new();
    let mut constraints = Vec::new();
    let mut next_id = 1usize;

    for statement in statements {
        let tokens = statement.split_whitespace().collect::<Vec<_>>();
        match tokens.as_slice() {
            ["rclayout", "1"] => {}
            ["for", "design", name] => design = Some((*name).to_owned()),
            ["region", name, "=", "box", "x", x, "y", y, "z", z] => {
                let (x0, x1) = parse_range(x).wrap_err("invalid x region range")?;
                let (y0, y1) = parse_range(y).wrap_err("invalid y region range")?;
                let (z0, z1) = parse_range(z).wrap_err("invalid z region range")?;
                if regions
                    .insert(
                        (*name).to_owned(),
                        IntentRegion {
                            min: [x0, y0, z0],
                            max: [x1, y1, z1],
                        },
                    )
                    .is_some()
                {
                    eyre::bail!("duplicate physical-intent region `{name}`");
                }
            }
            ["require", "instance", instance, "inside", region] => {
                constraints.push(PhysicalConstraint::Inside {
                    id: generated_id(&mut next_id),
                    instance: (*instance).to_owned(),
                    region: (*region).to_owned(),
                });
            }
            ["require", "instance", instance, "layer", range] => {
                let (min, max) = parse_range(range).wrap_err("invalid layer range")?;
                constraints.push(PhysicalConstraint::LayerRange {
                    id: generated_id(&mut next_id),
                    instance: (*instance).to_owned(),
                    min,
                    max,
                });
            }
            ["lock", "instance", instance, "at", x, y, z] => {
                constraints.push(PhysicalConstraint::FixedOrigin {
                    id: generated_id(&mut next_id),
                    instance: (*instance).to_owned(),
                    origin: [parse_usize(x)?, parse_usize(y)?, parse_usize(z)?],
                });
            }
            ["priority", "net", net, priority] => {
                constraints.push(PhysicalConstraint::NetPriority {
                    id: generated_id(&mut next_id),
                    net: (*net).to_owned(),
                    priority: parse_usize(priority)?,
                });
            }
            ["require", "net", net, "avoid", region] => {
                constraints.push(PhysicalConstraint::NetAvoid {
                    id: generated_id(&mut next_id),
                    net: (*net).to_owned(),
                    region: (*region).to_owned(),
                });
            }
            ["prefer", "instance", instance, "inside", region, "strength", strength] => {
                constraints.push(PhysicalConstraint::PreferInside {
                    id: generated_id(&mut next_id),
                    instance: (*instance).to_owned(),
                    region: (*region).to_owned(),
                    strength: parse_strength(strength)?,
                });
            }
            ["prefer", "instances", first, second, "same_layer", "strength", strength] => {
                constraints.push(PhysicalConstraint::SameLayer {
                    id: generated_id(&mut next_id),
                    first: (*first).to_owned(),
                    second: (*second).to_owned(),
                    strength: parse_strength(strength)?,
                });
            }
            _ => eyre::bail!("unsupported physical-intent statement `{statement}`"),
        }
    }

    Ok(PhysicalIntent {
        format: PHYSICAL_INTENT_FORMAT.to_owned(),
        design: design.context("physical intent is missing `for design <name>;`")?,
        regions,
        constraints,
    })
}

fn parse_range(value: &str) -> eyre::Result<(usize, usize)> {
    let (min, max) = value
        .split_once("..")
        .with_context(|| format!("expected inclusive range, got `{value}`"))?;
    let min = parse_usize(min)?;
    let max = parse_usize(max)?;
    if min > max {
        eyre::bail!("range minimum {min} exceeds maximum {max}");
    }
    Ok((min, max))
}

fn parse_usize(value: &str) -> eyre::Result<usize> {
    value
        .parse()
        .with_context(|| format!("expected non-negative integer, got `{value}`"))
}

fn parse_strength(value: &str) -> eyre::Result<PreferenceStrength> {
    match value {
        "strong" => Ok(PreferenceStrength::Strong),
        "medium" => Ok(PreferenceStrength::Medium),
        "weak" => Ok(PreferenceStrength::Weak),
        _ => eyre::bail!("unknown preference strength `{value}`"),
    }
}

fn generated_id(next_id: &mut usize) -> String {
    let id = format!("c{}", *next_id);
    *next_id += 1;
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::LogicalDesign;

    #[test]
    fn parses_and_binds_counter_floorplan_intent() -> eyre::Result<()> {
        let intent: PhysicalIntent = r#"
            rclayout 1;
            for design counter;
            region state = box x 0..63 y 0..47 z 2..9;
            require instance q_0_master inside state;
            require instance q_0_master layer 2..6;
            lock instance q_0_slave at 20 10 4;
            priority net clk 100;
            prefer instance q_0_master inside state strength strong;
            prefer instances q_0_master q_0_slave same_layer strength medium;
        "#
        .parse()?;
        let logical = LogicalDesign::from_verilog_source(
            r#"
                module counter(clk, q);
                  input clk;
                  output reg [1:0] q;
                  always @(posedge clk) begin
                    q <= q + 1;
                  end
                endmodule
            "#,
        )?;
        let topology = ResolvedPnrTopology::from_routable(&logical.lower_to_routable()?)?;
        let resolved = intent.bind(&topology)?;
        let clock_net = topology
            .nets
            .iter()
            .find(|net| net.display_name == "clk")
            .context("counter clock net")?;

        assert_eq!(resolved.constraints.len(), 6);
        assert_eq!(resolved.net_priority(clock_net.id), 100);
        assert!(resolved.constraints.iter().any(|constraint| matches!(
            constraint,
            ResolvedPhysicalConstraint::FixedOrigin { origin, .. } if *origin == [20, 10, 4]
        )));
        Ok(())
    }

    #[test]
    fn binding_rejects_unknown_instances() -> eyre::Result<()> {
        let intent: PhysicalIntent = r#"
            rclayout 1;
            for design top;
            region logic = box x 0..8 y 0..8 z 0..4;
            require instance missing inside logic;
        "#
        .parse()?;
        let topology = ResolvedPnrTopology {
            top: crate::transform::place_and_route::global_pnr::topology::DefinitionId(0),
            definitions: vec![
                crate::transform::place_and_route::global_pnr::topology::ResolvedDefinition {
                    id: crate::transform::place_and_route::global_pnr::topology::DefinitionId(0),
                    key: crate::transform::place_and_route::global_pnr::topology::DefinitionKey(
                        "top".to_owned(),
                    ),
                    display_name: "top".to_owned(),
                    ports: Vec::new(),
                    is_leaf: false,
                },
            ],
            ports: Vec::new(),
            instances: Vec::new(),
            nets: Vec::new(),
        };

        let error = intent.bind(&topology).unwrap_err();
        assert!(format!("{error:#}").contains("unknown physical-intent instance `missing`"));
        Ok(())
    }
}
