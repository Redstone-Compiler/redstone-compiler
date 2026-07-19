use std::fmt;
use std::str::FromStr;

use eyre::ContextCompat;

use super::logical::{
    ClockEdge, LogicalBinding, LogicalCell, LogicalCellKind, LogicalDesign, LogicalInput,
    LogicalInstance, LogicalModule, LogicalNet, LogicalOutput, LogicalPort, LogicalPortDirection,
    LogicalValue,
};
use super::syntax::{tokenize, Token};

impl fmt::Display for LogicalDesign {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(output, "rcir {};", self.version)?;
        writeln!(output, "stage logical;")?;
        writeln!(output, "top {};", name(&self.top))?;

        let mut modules = self.modules.iter().collect::<Vec<_>>();
        modules.sort_by(|left, right| left.name.cmp(&right.name));
        for module in modules {
            writeln!(output)?;
            write_module(output, module)?;
        }
        Ok(())
    }
}

impl FromStr for LogicalDesign {
    type Err = eyre::Report;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let mut parser = Parser::new(tokenize(source)?);
        parser.expect_keyword("rcir")?;
        let version = parser.expect_u32()?;
        parser.expect_symbol(';')?;
        parser.expect_keyword("stage")?;
        parser.expect_keyword("logical")?;
        parser.expect_symbol(';')?;
        parser.expect_keyword("top")?;
        let top = parser.expect_name()?;
        parser.expect_symbol(';')?;

        let mut modules = Vec::new();
        while !parser.is_done() {
            parser.expect_keyword("module")?;
            modules.push(parser.parse_module()?);
        }
        let design = Self {
            version,
            top,
            modules,
            debug: Default::default(),
        };
        design.validate()?;
        Ok(design)
    }
}

fn write_module(output: &mut fmt::Formatter<'_>, module: &LogicalModule) -> fmt::Result {
    writeln!(output, "module {} {{", name(&module.name))?;

    let net_widths = module
        .nets
        .iter()
        .map(|net| (net.name.as_str(), net.width))
        .collect::<std::collections::HashMap<_, _>>();
    let mut ports = module.ports.iter().collect::<Vec<_>>();
    ports.sort_by(|left, right| left.name.cmp(&right.name));
    for port in &ports {
        let width = net_widths[port.net.as_str()];
        writeln!(
            output,
            "  port {} {} : {};",
            direction_name(port.direction),
            name(&port.name),
            type_name(width)
        )?;
    }

    let port_nets = module
        .ports
        .iter()
        .map(|port| port.net.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut nets = module
        .nets
        .iter()
        .filter(|net| !port_nets.contains(net.name.as_str()))
        .collect::<Vec<_>>();
    nets.sort_by(|left, right| left.name.cmp(&right.name));
    if !nets.is_empty() && !ports.is_empty() {
        writeln!(output)?;
    }
    for net in nets {
        writeln!(
            output,
            "  net {} : {};",
            name(&net.name),
            type_name(net.width)
        )?;
    }

    let mut instances = module.instances.iter().collect::<Vec<_>>();
    instances.sort_by(|left, right| left.name.cmp(&right.name));
    for instance in instances {
        writeln!(
            output,
            "\n  instance {} : {} {{",
            name(&instance.name),
            name(&instance.module)
        )?;
        let mut bindings = instance.bindings.iter().collect::<Vec<_>>();
        bindings.sort_by(|left, right| left.port.cmp(&right.port));
        for binding in bindings {
            writeln!(
                output,
                "    bind {} = {};",
                name(&binding.port),
                name(&binding.net)
            )?;
        }
        writeln!(output, "  }}")?;
    }

    let mut cells = module.cells.iter().collect::<Vec<_>>();
    cells.sort_by(|left, right| left.name.cmp(&right.name));
    for cell in cells {
        let width = cell_width(cell, &net_widths);
        writeln!(
            output,
            "\n  cell {} : logical.{}<{}> {{",
            name(&cell.name),
            operation_name(&cell.kind),
            width
        )?;

        let mut inputs = cell.inputs.iter().collect::<Vec<_>>();
        inputs.sort_by_key(|input| pin_rank(&cell.kind, &input.pin));
        for input in inputs {
            write!(output, "    in  {} = ", name(&input.pin))?;
            write_value(output, &input.value)?;
            writeln!(output, ";")?;
        }

        let mut outputs = cell.outputs.iter().collect::<Vec<_>>();
        outputs.sort_by_key(|output| output.pin.as_str());
        for pin in outputs {
            writeln!(output, "    out {} = {};", name(&pin.pin), name(&pin.net))?;
        }

        match &cell.kind {
            LogicalCellKind::Dff { edge } | LogicalCellKind::Register { edge, .. } => {
                writeln!(output, "    edge = {};", edge_name(*edge))?;
            }
            _ => {}
        }
        writeln!(output, "  }}")?;
    }
    writeln!(output, "}}")
}

fn write_value(output: &mut fmt::Formatter<'_>, value: &LogicalValue) -> fmt::Result {
    match value {
        LogicalValue::Net { net } => write!(output, "{}", name(net)),
        LogicalValue::Constant { value, width } => write!(output, "const<{width}>({value})"),
    }
}

fn cell_width(cell: &LogicalCell, net_widths: &std::collections::HashMap<&str, usize>) -> usize {
    match cell.kind {
        LogicalCellKind::DLatch { width } | LogicalCellKind::Register { width, .. } => width,
        LogicalCellKind::Dff { .. } => 1,
        _ => cell
            .outputs
            .first()
            .and_then(|output| net_widths.get(output.net.as_str()).copied())
            .unwrap_or(1),
    }
}

fn pin_rank(kind: &LogicalCellKind, pin: &str) -> usize {
    let order: &[&str] = match kind {
        LogicalCellKind::Buffer | LogicalCellKind::Not | LogicalCellKind::Inc => &["value"],
        LogicalCellKind::And
        | LogicalCellKind::Or
        | LogicalCellKind::Xor
        | LogicalCellKind::Add => &["lhs", "rhs"],
        LogicalCellKind::Mux => &["select", "when_false", "when_true"],
        LogicalCellKind::DLatch { .. } => &["d", "enable"],
        LogicalCellKind::Dff { .. } | LogicalCellKind::Register { .. } => &["d", "clock"],
    };
    order
        .iter()
        .position(|expected| *expected == pin)
        .unwrap_or(usize::MAX)
}

fn name(value: &str) -> String {
    if is_bare_name(value) {
        return value.to_owned();
    }
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

fn is_bare_name(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn type_name(width: usize) -> String {
    if width == 1 {
        "bit".to_owned()
    } else {
        format!("bits[{width}]")
    }
}

fn direction_name(direction: LogicalPortDirection) -> &'static str {
    match direction {
        LogicalPortDirection::Input => "input",
        LogicalPortDirection::Output => "output",
    }
}

fn edge_name(edge: ClockEdge) -> &'static str {
    match edge {
        ClockEdge::Posedge => "posedge",
        ClockEdge::Negedge => "negedge",
    }
}

fn operation_name(kind: &LogicalCellKind) -> &'static str {
    match kind {
        LogicalCellKind::Buffer => "buffer",
        LogicalCellKind::Not => "not",
        LogicalCellKind::And => "and",
        LogicalCellKind::Or => "or",
        LogicalCellKind::Xor => "xor",
        LogicalCellKind::Add => "add",
        LogicalCellKind::Inc => "inc",
        LogicalCellKind::Mux => "mux",
        LogicalCellKind::DLatch { .. } => "d_latch",
        LogicalCellKind::Dff { .. } => "dff",
        LogicalCellKind::Register { .. } => "register",
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

    fn parse_module(&mut self) -> eyre::Result<LogicalModule> {
        let module_name = self.expect_name()?;
        self.expect_symbol('{')?;
        let mut nets = Vec::<LogicalNet>::new();
        let mut ports = Vec::new();
        let mut cells = Vec::new();
        let mut instances = Vec::new();
        while !self.consume_symbol('}') {
            if self.consume_keyword("port") {
                let (port, width) = self.parse_port()?;
                add_or_check_net(&mut nets, &port.net, width)?;
                ports.push(port);
            } else if self.consume_keyword("net") {
                let net = self.parse_net()?;
                add_or_check_net(&mut nets, &net.name, net.width)?;
            } else if self.consume_keyword("cell") {
                cells.push(self.parse_cell()?);
            } else if self.consume_keyword("instance") {
                instances.push(self.parse_instance()?);
            } else {
                return Err(self.unexpected("`port`, `net`, `cell`, `instance`, or `}`"));
            }
        }
        Ok(LogicalModule {
            name: module_name,
            nets,
            ports,
            cells,
            instances,
        })
    }

    fn parse_port(&mut self) -> eyre::Result<(LogicalPort, usize)> {
        let direction = if self.consume_keyword("input") {
            LogicalPortDirection::Input
        } else if self.consume_keyword("output") {
            LogicalPortDirection::Output
        } else {
            return Err(self.unexpected("`input` or `output`"));
        };
        let port_name = self.expect_name()?;
        self.expect_symbol(':')?;
        let width = self.parse_type()?;
        self.expect_symbol(';')?;
        Ok((
            LogicalPort {
                name: port_name.clone(),
                direction,
                net: port_name,
            },
            width,
        ))
    }

    fn parse_net(&mut self) -> eyre::Result<LogicalNet> {
        let net_name = self.expect_name()?;
        self.expect_symbol(':')?;
        let width = self.parse_type()?;
        self.expect_symbol(';')?;
        Ok(LogicalNet {
            name: net_name,
            width,
            origin: None,
        })
    }

    fn parse_type(&mut self) -> eyre::Result<usize> {
        if self.consume_keyword("bit") {
            return Ok(1);
        }
        self.expect_keyword("bits")?;
        self.expect_symbol('[')?;
        let width = self.expect_usize()?;
        self.expect_symbol(']')?;
        if width == 0 {
            eyre::bail!("bits[0] is not a valid RCIR type");
        }
        Ok(width)
    }

    fn parse_instance(&mut self) -> eyre::Result<LogicalInstance> {
        let instance_name = self.expect_name()?;
        self.expect_symbol(':')?;
        let module = self.expect_name()?;
        self.expect_symbol('{')?;
        let mut bindings = Vec::new();
        while !self.consume_symbol('}') {
            self.expect_keyword("bind")?;
            let port = self.expect_name()?;
            self.expect_symbol('=')?;
            let net = self.expect_name()?;
            self.expect_symbol(';')?;
            bindings.push(LogicalBinding { port, net });
        }
        Ok(LogicalInstance {
            name: instance_name,
            module,
            bindings,
            origin: None,
        })
    }

    fn parse_cell(&mut self) -> eyre::Result<LogicalCell> {
        let cell_name = self.expect_name()?;
        self.expect_symbol(':')?;
        self.expect_keyword("logical")?;
        self.expect_symbol('.')?;
        let operation = self.expect_word()?;
        self.expect_symbol('<')?;
        let width = self.expect_usize()?;
        self.expect_symbol('>')?;
        if width == 0 {
            eyre::bail!("logical operation `{operation}` has zero width");
        }
        self.expect_symbol('{')?;

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut edge = None;
        while !self.consume_symbol('}') {
            if self.consume_keyword("in") {
                let pin = self.expect_name()?;
                self.expect_symbol('=')?;
                let value = self.parse_value()?;
                self.expect_symbol(';')?;
                inputs.push(LogicalInput { pin, value });
            } else if self.consume_keyword("out") {
                let pin = self.expect_name()?;
                self.expect_symbol('=')?;
                let net = self.expect_name()?;
                self.expect_symbol(';')?;
                outputs.push(LogicalOutput { pin, net });
            } else if self.consume_keyword("edge") {
                self.expect_symbol('=')?;
                edge = Some(self.parse_edge()?);
                self.expect_symbol(';')?;
            } else {
                return Err(self.unexpected("`in`, `out`, `edge`, or `}`"));
            }
        }

        let kind = match operation.as_str() {
            "buffer" => LogicalCellKind::Buffer,
            "not" => LogicalCellKind::Not,
            "and" => LogicalCellKind::And,
            "or" => LogicalCellKind::Or,
            "xor" => LogicalCellKind::Xor,
            "add" => LogicalCellKind::Add,
            "inc" => LogicalCellKind::Inc,
            "mux" => LogicalCellKind::Mux,
            "d_latch" => LogicalCellKind::DLatch { width },
            "dff" => {
                if width != 1 {
                    eyre::bail!("logical.dff must have width 1, found {width}");
                }
                LogicalCellKind::Dff {
                    edge: edge.context("logical.dff requires an edge attribute")?,
                }
            }
            "register" => LogicalCellKind::Register {
                width,
                edge: edge.context("logical.register requires an edge attribute")?,
            },
            other => eyre::bail!("unknown logical operation `logical.{other}`"),
        };
        if edge.is_some()
            && !matches!(
                kind,
                LogicalCellKind::Dff { .. } | LogicalCellKind::Register { .. }
            )
        {
            eyre::bail!("logical.{operation} does not accept an edge attribute");
        }
        Ok(LogicalCell {
            name: cell_name,
            kind,
            inputs,
            outputs,
            origin: None,
        })
    }

    fn parse_value(&mut self) -> eyre::Result<LogicalValue> {
        if self.consume_keyword("const") {
            self.expect_symbol('<')?;
            let width = self.expect_usize()?;
            self.expect_symbol('>')?;
            self.expect_symbol('(')?;
            let value = self.expect_number()?;
            self.expect_symbol(')')?;
            return Ok(LogicalValue::Constant { value, width });
        }
        Ok(LogicalValue::Net {
            net: self.expect_name()?,
        })
    }

    fn parse_edge(&mut self) -> eyre::Result<ClockEdge> {
        match self.expect_word()?.as_str() {
            "posedge" => Ok(ClockEdge::Posedge),
            "negedge" => Ok(ClockEdge::Negedge),
            edge => eyre::bail!("unknown clock edge `{edge}`"),
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

    fn expect_name(&mut self) -> eyre::Result<String> {
        match self.tokens.get(self.position).cloned() {
            Some(Token::Word(value)) | Some(Token::String(value)) => {
                self.position += 1;
                Ok(value)
            }
            _ => Err(self.unexpected("identifier or quoted name")),
        }
    }

    fn expect_number(&mut self) -> eyre::Result<u128> {
        match self.tokens.get(self.position).cloned() {
            Some(Token::Number(value)) => {
                self.position += 1;
                value.parse().map_err(|_| self.unexpected("number"))
            }
            _ => Err(self.unexpected("number")),
        }
    }

    fn expect_u32(&mut self) -> eyre::Result<u32> {
        self.expect_number()?
            .try_into()
            .map_err(|_| self.unexpected("32-bit number"))
    }

    fn expect_usize(&mut self) -> eyre::Result<usize> {
        self.expect_number()?
            .try_into()
            .map_err(|_| self.unexpected("platform-sized number"))
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

fn add_or_check_net(nets: &mut Vec<LogicalNet>, name: &str, width: usize) -> eyre::Result<()> {
    if let Some(existing) = nets.iter().find(|net| net.name == name) {
        if existing.width != width {
            eyre::bail!(
                "logical net `{name}` is declared with widths {} and {width}",
                existing.width
            );
        }
        return Ok(());
    }
    nets.push(LogicalNet {
        name: name.to_owned(),
        width,
        origin: None,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_text_round_trips_deterministically() -> eyre::Result<()> {
        let design = LogicalDesign::from_verilog_source(
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
        let first = design.to_string();
        let expected = r#"rcir 1;
stage logical;
top counter;

module counter {
  port input clk : bit;
  port output q : bits[2];

  net q_next : bits[2];

  cell next : logical.inc<2> {
    in  value = q;
    out result = q_next;
  }

  cell state : logical.register<2> {
    in  d = q_next;
    in  clock = clk;
    out q = q;
    edge = posedge;
  }
}
"#;
        let reparsed: LogicalDesign = first.parse()?;
        let second = reparsed.to_string();
        let reparsed_again: LogicalDesign = second.parse()?;

        assert_eq!(reparsed_again, reparsed);
        assert_eq!(second, first);
        assert_eq!(first, expected);
        assert!(first.contains("rcir 1;"));
        assert!(first.contains("cell next : logical.inc<2>"));
        assert!(first.contains("cell state : logical.register<2>"));
        assert!(!first.contains("origin"));
        assert_eq!(reparsed.lower_to_routable()?.top, "counter");
        Ok(())
    }

    #[test]
    fn parses_hand_written_typed_counter() -> eyre::Result<()> {
        let source = r#"
            rcir 1;
            stage logical;
            top counter;

            module counter {
              port input clk : bit;
              port output q : bits[2];
              net q_next : bits[2];

              cell next : logical.inc<2> {
                in value = q;
                out result = q_next;
              }
              cell state : logical.register<2> {
                in d = q_next;
                in clock = clk;
                out q = q;
                edge = posedge;
              }
            }
        "#;
        let logical: LogicalDesign = source.parse()?;
        assert_eq!(logical.to_string().parse::<LogicalDesign>()?, logical);
        assert_eq!(logical.lower_to_routable()?.top, "counter");
        Ok(())
    }
}
