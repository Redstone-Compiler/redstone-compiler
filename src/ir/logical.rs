use std::collections::{HashMap, HashSet};

use eyre::{ContextCompat, WrapErr};
use serde::{Deserialize, Serialize};

pub const LOGICAL_IR_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalDesign {
    pub version: u32,
    pub top: String,
    pub modules: Vec<LogicalModule>,
    #[serde(skip)]
    pub debug: super::debug::IrDebugInfo,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalModule {
    pub name: String,
    pub nets: Vec<LogicalNet>,
    pub ports: Vec<LogicalPort>,
    pub cells: Vec<LogicalCell>,
    pub instances: Vec<LogicalInstance>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalNet {
    pub name: String,
    pub width: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalPort {
    pub name: String,
    pub direction: LogicalPortDirection,
    pub net: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalPortDirection {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalCell {
    pub name: String,
    pub kind: LogicalCellKind,
    pub inputs: Vec<LogicalInput>,
    pub outputs: Vec<LogicalOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalInput {
    pub pin: String,
    pub value: LogicalValue,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalOutput {
    pub pin: String,
    pub net: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogicalValue {
    Net { net: String },
    Constant { value: u128, width: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogicalCellKind {
    Buffer,
    Not,
    And,
    Or,
    Xor,
    Add,
    Inc,
    Mux,
    DLatch { width: usize },
    Dff { edge: ClockEdge },
    Register { width: usize, edge: ClockEdge },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockEdge {
    Posedge,
    Negedge,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalInstance {
    pub name: String,
    pub module: String,
    pub bindings: Vec<LogicalBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalBinding {
    pub port: String,
    pub net: String,
}

impl LogicalDesign {
    pub fn module(&self, name: &str) -> Option<&LogicalModule> {
        self.modules.iter().find(|module| module.name == name)
    }

    pub fn validate(&self) -> eyre::Result<()> {
        if self.version != LOGICAL_IR_VERSION {
            eyre::bail!(
                "unsupported logical IR version {}; expected {}",
                self.version,
                LOGICAL_IR_VERSION
            );
        }
        ensure_unique(
            self.modules.iter().map(|module| module.name.as_str()),
            "module",
        )?;
        self.module(&self.top)
            .with_context(|| format!("unknown logical top module `{}`", self.top))?;

        let modules = self
            .modules
            .iter()
            .map(|module| (module.name.as_str(), module))
            .collect::<HashMap<_, _>>();
        for module in &self.modules {
            module
                .validate(&modules)
                .wrap_err_with(|| format!("invalid logical module `{}`", module.name))?;
        }
        validate_hierarchy(&self.top, &modules)
    }
}

impl LogicalModule {
    fn validate(&self, modules: &HashMap<&str, &LogicalModule>) -> eyre::Result<()> {
        if self.name.is_empty() {
            eyre::bail!("module name must not be empty");
        }
        ensure_unique(self.nets.iter().map(|net| net.name.as_str()), "net")?;
        ensure_unique(self.ports.iter().map(|port| port.name.as_str()), "port")?;
        ensure_unique(self.cells.iter().map(|cell| cell.name.as_str()), "cell")?;
        ensure_unique(
            self.instances.iter().map(|instance| instance.name.as_str()),
            "instance",
        )?;

        let nets = self
            .nets
            .iter()
            .map(|net| {
                if net.name.is_empty() {
                    eyre::bail!("net name must not be empty");
                }
                if net.width == 0 {
                    eyre::bail!("net `{}` has zero width", net.name);
                }
                Ok((net.name.as_str(), net))
            })
            .collect::<eyre::Result<HashMap<_, _>>>()?;

        let mut drivers = HashMap::<&str, String>::new();
        let mut used = HashSet::<&str>::new();
        for port in &self.ports {
            let net = nets.get(port.net.as_str()).with_context(|| {
                format!("port `{}` references unknown net `{}`", port.name, port.net)
            })?;
            if port.name.is_empty() {
                eyre::bail!("port name must not be empty");
            }
            if port.name != port.net {
                eyre::bail!(
                    "logical RCIR requires port `{}` to use a same-named net, found `{}`",
                    port.name,
                    port.net
                );
            }
            match port.direction {
                LogicalPortDirection::Input => add_driver(
                    &mut drivers,
                    net.name.as_str(),
                    format!("input port `{}`", port.name),
                )?,
                LogicalPortDirection::Output => {
                    used.insert(net.name.as_str());
                }
            }
        }

        for cell in &self.cells {
            cell.validate(&nets)?;
            for input in &cell.inputs {
                if let LogicalValue::Net { net } = &input.value {
                    used.insert(net.as_str());
                }
            }
            for output in &cell.outputs {
                add_driver(
                    &mut drivers,
                    output.net.as_str(),
                    format!("cell `{}.{}`", cell.name, output.pin),
                )?;
            }
        }

        for instance in &self.instances {
            let definition = modules
                .get(instance.module.as_str())
                .copied()
                .with_context(|| {
                    format!(
                        "instance `{}` references unknown module `{}`",
                        instance.name, instance.module
                    )
                })?;
            ensure_unique(
                instance
                    .bindings
                    .iter()
                    .map(|binding| binding.port.as_str()),
                "instance binding",
            )?;
            if instance.bindings.len() != definition.ports.len() {
                eyre::bail!(
                    "instance `{}` binds {} ports, expected {}",
                    instance.name,
                    instance.bindings.len(),
                    definition.ports.len()
                );
            }
            for port in &definition.ports {
                let binding = instance
                    .bindings
                    .iter()
                    .find(|binding| binding.port == port.name)
                    .with_context(|| {
                        format!(
                            "instance `{}` is missing port `{}`",
                            instance.name, port.name
                        )
                    })?;
                let parent_net = nets.get(binding.net.as_str()).with_context(|| {
                    format!(
                        "instance `{}.{}` references unknown net `{}`",
                        instance.name, port.name, binding.net
                    )
                })?;
                let child_net = definition.net_for_port(port)?;
                if parent_net.width != child_net.width {
                    eyre::bail!(
                        "instance `{}.{}` width {} does not match net `{}` width {}",
                        instance.name,
                        port.name,
                        child_net.width,
                        parent_net.name,
                        parent_net.width
                    );
                }
                match port.direction {
                    LogicalPortDirection::Input => {
                        used.insert(parent_net.name.as_str());
                    }
                    LogicalPortDirection::Output => add_driver(
                        &mut drivers,
                        parent_net.name.as_str(),
                        format!("instance `{}.{}`", instance.name, port.name),
                    )?,
                }
            }
        }

        for net in used {
            if !drivers.contains_key(net) {
                eyre::bail!("used net `{net}` has no driver");
            }
        }
        self.validate_combinational_cycles()?;
        Ok(())
    }

    fn net_for_port(&self, port: &LogicalPort) -> eyre::Result<&LogicalNet> {
        self.nets
            .iter()
            .find(|net| net.name == port.net)
            .with_context(|| format!("port `{}` references unknown net `{}`", port.name, port.net))
    }

    fn validate_combinational_cycles(&self) -> eyre::Result<()> {
        let mut edges = HashMap::<&str, Vec<&str>>::new();
        for cell in &self.cells {
            if cell.kind.is_sequential() {
                continue;
            }
            let inputs = cell.inputs.iter().filter_map(|input| match &input.value {
                LogicalValue::Net { net } => Some(net.as_str()),
                LogicalValue::Constant { .. } => None,
            });
            for input in inputs {
                edges
                    .entry(input)
                    .or_default()
                    .extend(cell.outputs.iter().map(|output| output.net.as_str()));
            }
        }

        fn visit<'a>(
            net: &'a str,
            edges: &HashMap<&'a str, Vec<&'a str>>,
            active: &mut Vec<&'a str>,
            done: &mut HashSet<&'a str>,
        ) -> eyre::Result<()> {
            if done.contains(net) {
                return Ok(());
            }
            if let Some(index) = active.iter().position(|candidate| *candidate == net) {
                let mut cycle = active[index..].to_vec();
                cycle.push(net);
                eyre::bail!("combinational cycle: {}", cycle.join(" -> "));
            }
            active.push(net);
            for next in edges.get(net).into_iter().flatten() {
                visit(next, edges, active, done)?;
            }
            active.pop();
            done.insert(net);
            Ok(())
        }

        let mut done = HashSet::new();
        for net in edges.keys().copied() {
            visit(net, &edges, &mut Vec::new(), &mut done)?;
        }
        Ok(())
    }
}

impl LogicalCell {
    fn validate(&self, nets: &HashMap<&str, &LogicalNet>) -> eyre::Result<()> {
        if self.name.is_empty() {
            eyre::bail!("cell name must not be empty");
        }
        ensure_unique(
            self.inputs.iter().map(|input| input.pin.as_str()),
            "input pin",
        )?;
        ensure_unique(
            self.outputs.iter().map(|output| output.pin.as_str()),
            "output pin",
        )?;
        let (input_pins, output_pins) = self.kind.pin_names();
        ensure_pin_set(
            self.inputs.iter().map(|input| input.pin.as_str()),
            input_pins,
            "input",
        )?;
        ensure_pin_set(
            self.outputs.iter().map(|output| output.pin.as_str()),
            output_pins,
            "output",
        )?;
        for input in &self.inputs {
            validate_value(&input.value, nets)?;
        }
        for output in &self.outputs {
            nets.get(output.net.as_str()).with_context(|| {
                format!(
                    "cell `{}.{}` references unknown output net `{}`",
                    self.name, output.pin, output.net
                )
            })?;
        }

        let output_width = nets[self
            .output("result")
            .or_else(|_| self.output("q"))?
            .as_str()]
        .width;
        match &self.kind {
            LogicalCellKind::Buffer | LogicalCellKind::Not | LogicalCellKind::Inc => {
                ensure_data_width(
                    self.input_value(self.kind.primary_input_pin())?,
                    output_width,
                    nets,
                )?;
            }
            LogicalCellKind::And
            | LogicalCellKind::Or
            | LogicalCellKind::Xor
            | LogicalCellKind::Add => {
                ensure_data_width(self.input_value("lhs")?, output_width, nets)?;
                ensure_data_width(self.input_value("rhs")?, output_width, nets)?;
            }
            LogicalCellKind::Mux => {
                ensure_exact_width(self.input_value("select")?, 1, nets, "mux select")?;
                ensure_data_width(self.input_value("when_true")?, output_width, nets)?;
                ensure_data_width(self.input_value("when_false")?, output_width, nets)?;
            }
            LogicalCellKind::DLatch { width } => {
                ensure_declared_width(*width, output_width, &self.name)?;
                ensure_data_width(self.input_value("d")?, output_width, nets)?;
                ensure_exact_width(self.input_value("enable")?, 1, nets, "latch enable")?;
            }
            LogicalCellKind::Dff { .. } => {
                ensure_declared_width(1, output_width, &self.name)?;
                ensure_data_width(self.input_value("d")?, output_width, nets)?;
                ensure_exact_width(self.input_value("clock")?, 1, nets, "DFF clock")?;
            }
            LogicalCellKind::Register { width, .. } => {
                ensure_declared_width(*width, output_width, &self.name)?;
                ensure_data_width(self.input_value("d")?, output_width, nets)?;
                ensure_exact_width(self.input_value("clock")?, 1, nets, "register clock")?;
            }
        }
        Ok(())
    }

    pub fn input_value(&self, pin: &str) -> eyre::Result<&LogicalValue> {
        self.inputs
            .iter()
            .find(|input| input.pin == pin)
            .map(|input| &input.value)
            .with_context(|| format!("cell `{}` has no input pin `{pin}`", self.name))
    }

    pub fn output(&self, pin: &str) -> eyre::Result<&String> {
        self.outputs
            .iter()
            .find(|output| output.pin == pin)
            .map(|output| &output.net)
            .with_context(|| format!("cell `{}` has no output pin `{pin}`", self.name))
    }
}

impl LogicalCellKind {
    pub fn is_sequential(&self) -> bool {
        matches!(
            self,
            Self::DLatch { .. } | Self::Dff { .. } | Self::Register { .. }
        )
    }

    fn primary_input_pin(&self) -> &'static str {
        match self {
            Self::Buffer | Self::Not => "value",
            Self::Inc => "value",
            _ => unreachable!(),
        }
    }

    fn pin_names(&self) -> (&'static [&'static str], &'static [&'static str]) {
        match self {
            Self::Buffer | Self::Not => (&["value"], &["result"]),
            Self::And | Self::Or | Self::Xor | Self::Add => (&["lhs", "rhs"], &["result"]),
            Self::Inc => (&["value"], &["result"]),
            Self::Mux => (&["select", "when_true", "when_false"], &["result"]),
            Self::DLatch { .. } => (&["d", "enable"], &["q"]),
            Self::Dff { .. } | Self::Register { .. } => (&["d", "clock"], &["q"]),
        }
    }
}

fn validate_value(value: &LogicalValue, nets: &HashMap<&str, &LogicalNet>) -> eyre::Result<()> {
    match value {
        LogicalValue::Net { net } => {
            nets.get(net.as_str())
                .with_context(|| format!("unknown logical net `{net}`"))?;
        }
        LogicalValue::Constant { value, width } => {
            if *width == 0 {
                eyre::bail!("constant has zero width");
            }
            if *width < 128 && *value >= (1u128 << *width) {
                eyre::bail!("constant value {value} does not fit in {width} bits");
            }
        }
    }
    Ok(())
}

fn value_width(value: &LogicalValue, nets: &HashMap<&str, &LogicalNet>) -> eyre::Result<usize> {
    match value {
        LogicalValue::Net { net } => Ok(nets
            .get(net.as_str())
            .with_context(|| format!("unknown logical net `{net}`"))?
            .width),
        LogicalValue::Constant { width, .. } => Ok(*width),
    }
}

fn ensure_data_width(
    value: &LogicalValue,
    output_width: usize,
    nets: &HashMap<&str, &LogicalNet>,
) -> eyre::Result<()> {
    let width = value_width(value, nets)?;
    let compatible = match value {
        LogicalValue::Net { .. } => width == output_width,
        LogicalValue::Constant { .. } => width <= output_width,
    };
    if !compatible {
        eyre::bail!("input width {width} does not match output width {output_width}");
    }
    Ok(())
}

fn ensure_exact_width(
    value: &LogicalValue,
    expected: usize,
    nets: &HashMap<&str, &LogicalNet>,
    role: &str,
) -> eyre::Result<()> {
    let actual = value_width(value, nets)?;
    if actual != expected {
        eyre::bail!("{role} has width {actual}, expected {expected}");
    }
    Ok(())
}

fn ensure_declared_width(declared: usize, actual: usize, cell: &str) -> eyre::Result<()> {
    if declared == 0 || declared != actual {
        eyre::bail!("cell `{cell}` declares width {declared}, output has width {actual}");
    }
    Ok(())
}

fn ensure_pin_set<'a>(
    actual: impl IntoIterator<Item = &'a str>,
    expected: &[&str],
    kind: &str,
) -> eyre::Result<()> {
    let actual = actual.into_iter().collect::<HashSet<_>>();
    let expected = expected.iter().copied().collect::<HashSet<_>>();
    if actual != expected {
        eyre::bail!("{kind} pins are {:?}, expected {:?}", actual, expected);
    }
    Ok(())
}

fn add_driver<'a>(
    drivers: &mut HashMap<&'a str, String>,
    net: &'a str,
    driver: String,
) -> eyre::Result<()> {
    if let Some(previous) = drivers.insert(net, driver.clone()) {
        eyre::bail!("net `{net}` has multiple drivers: {previous} and {driver}");
    }
    Ok(())
}

fn validate_hierarchy<'a>(
    top: &'a str,
    modules: &HashMap<&'a str, &'a LogicalModule>,
) -> eyre::Result<()> {
    fn visit<'a>(
        name: &'a str,
        modules: &HashMap<&'a str, &'a LogicalModule>,
        active: &mut Vec<&'a str>,
        done: &mut HashSet<&'a str>,
    ) -> eyre::Result<()> {
        if done.contains(name) {
            return Ok(());
        }
        if let Some(index) = active.iter().position(|candidate| *candidate == name) {
            let mut cycle = active[index..].to_vec();
            cycle.push(name);
            eyre::bail!("recursive logical hierarchy: {}", cycle.join(" -> "));
        }
        active.push(name);
        let module = modules
            .get(name)
            .copied()
            .with_context(|| format!("unknown logical module `{name}`"))?;
        for instance in &module.instances {
            visit(&instance.module, modules, active, done)?;
        }
        active.pop();
        done.insert(name);
        Ok(())
    }

    visit(top, modules, &mut Vec::new(), &mut HashSet::new())
}

fn ensure_unique<T>(values: impl IntoIterator<Item = T>, kind: &str) -> eyre::Result<()>
where
    T: Eq + std::hash::Hash + std::fmt::Debug,
{
    let mut seen = HashSet::new();
    for value in values {
        if seen.contains(&value) {
            eyre::bail!("duplicate {kind} {value:?}");
        }
        seen.insert(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_rejects_combinational_cycle() -> eyre::Result<()> {
        let mut design = LogicalDesign::from_verilog_source(
            "module inv(a, y); input a; output y; assign y = ~a; endmodule",
        )?;
        let cell = &mut design.modules[0].cells[0];
        cell.inputs[0].value = LogicalValue::Net {
            net: "y".to_owned(),
        };

        let error = format!("{:#}", design.validate().unwrap_err());
        assert!(error.contains("combinational cycle"));
        Ok(())
    }

    #[test]
    fn validator_rejects_register_width_mismatch() -> eyre::Result<()> {
        let mut design = LogicalDesign::from_verilog_source(
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
        let state = design.modules[0]
            .cells
            .iter_mut()
            .find(|cell| matches!(cell.kind, LogicalCellKind::Register { .. }))
            .unwrap();
        state.kind = LogicalCellKind::Register {
            width: 3,
            edge: ClockEdge::Posedge,
        };

        let error = format!("{:#}", design.validate().unwrap_err());
        assert!(error.contains("declares width 3"));
        Ok(())
    }
}
