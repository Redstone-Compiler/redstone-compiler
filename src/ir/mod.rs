pub mod debug;
mod leaf_graph;
pub(crate) use leaf_graph::graph_from_routable_leaf;
pub mod logical;
mod logical_adapter;
mod logical_lowering;
mod logical_text;
pub mod pnr;
pub mod routable;
mod syntax;
mod text;

use std::fmt;
use std::str::FromStr;

pub use debug::{DebugLocation, DebugLocationId, DebugRange, IrDebugInfo, IrSourceMap};
pub use logical::{
    ClockEdge, LogicalBinding, LogicalCell, LogicalCellKind, LogicalDesign, LogicalInput,
    LogicalInstance, LogicalModule, LogicalNet, LogicalOutput, LogicalPort, LogicalPortDirection,
    LogicalValue, LOGICAL_IR_VERSION,
};
pub use pnr::{
    CandidateSpec, CongestionSpec, Free3dSweepSpec, InputPlacementSpec, LayerAssignmentSpec,
    LocalPlacerSpec, NetOrderSpec, NotRouteSpec, ObjectiveSpec, PhysicalConstraintSpec,
    PhysicalRegionSpec, PhysicalSpec, PlacementHeuristicSpec, PlacementSamplingSpec,
    PlacementScheduleSpec, PlacementSpec, PnrSpec, PortRef, PreferenceSpec, RoutableDocument,
    RouteStageSpec, RouteStrategySpec, RouteValidationSpec, RoutingSpec, SamplingSpec, SearchSpec,
    TorchPlacementSpec,
};
pub use routable::{
    Endpoint, NetClass, RoutableDesign, RoutableInstance, RoutableModule, RoutableModuleBody,
    RoutableNet, RoutableNode, RoutableNodeKind, RoutablePort, RoutablePortDirection,
    RoutableSequentialPrimitive, ROUTABLE_IR_TARGET, ROUTABLE_IR_VERSION,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CircuitIr {
    Logical(LogicalDesign),
    Routable(RoutableDesign),
}

/// A source RCIR file, including stage-specific sidecar dialects that do not
/// belong to the circuit graph itself.
#[derive(Clone, Debug, PartialEq)]
pub enum RcirDocument {
    Logical(LogicalDesign),
    Routable(RoutableDocument),
}

impl FromStr for RcirDocument {
    type Err = eyre::Report;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        match declared_stage(source)?.as_str() {
            "logical" => Ok(Self::Logical(source.parse()?)),
            "routable" => Ok(Self::Routable(source.parse()?)),
            stage => eyre::bail!("unsupported rcir stage `{stage}`"),
        }
    }
}

pub fn parse_rcir_document(source: &str) -> eyre::Result<RcirDocument> {
    source.parse()
}

impl CircuitIr {
    pub fn top(&self) -> &str {
        match self {
            Self::Logical(design) => &design.top,
            Self::Routable(design) => &design.top,
        }
    }

    pub fn validate(&self) -> eyre::Result<()> {
        match self {
            Self::Logical(design) => design.validate(),
            Self::Routable(design) => design.validate(),
        }
    }
}

impl fmt::Display for CircuitIr {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Logical(design) => design.fmt(output),
            Self::Routable(design) => design.fmt(output),
        }
    }
}

impl FromStr for CircuitIr {
    type Err = eyre::Report;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        match declared_stage(source)?.as_str() {
            "logical" => Ok(Self::Logical(source.parse()?)),
            "routable" => Ok(Self::Routable(source.parse()?)),
            stage => eyre::bail!("unsupported rcir stage `{stage}`"),
        }
    }
}

pub fn parse_rcir(source: &str) -> eyre::Result<CircuitIr> {
    source.parse()
}

fn declared_stage(source: &str) -> eyre::Result<String> {
    for declaration in source.split(';') {
        let declaration = declaration
            .lines()
            .map(|line| line.split('#').next().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(" ");
        let mut words = declaration.split_whitespace();
        if words.next() == Some("stage") {
            return words
                .next()
                .map(str::to_owned)
                .ok_or_else(|| eyre::eyre!("rcir stage declaration has no value"));
        }
    }
    eyre::bail!("rcir input has no stage declaration")
}

#[cfg(test)]
mod circuit_tests {
    use super::*;

    #[test]
    fn circuit_ir_dispatches_from_stage_header() -> eyre::Result<()> {
        let logical = LogicalDesign::from_verilog_source(
            "module inv(a, y); input a; output y; assign y = ~a; endmodule",
        )?;
        let text = logical.to_string();

        assert!(matches!(parse_rcir(&text)?, CircuitIr::Logical(_)));
        assert!(matches!(
            parse_rcir(&logical.lower_to_routable()?.to_string())?,
            CircuitIr::Routable(_)
        ));
        Ok(())
    }
}
