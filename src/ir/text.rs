use std::fmt;
use std::str::FromStr;

use super::routable::{
    Endpoint, NetClass, RoutableDesign, RoutableInstance, RoutableModule, RoutableModuleBody,
    RoutableNet, RoutableNode, RoutableNodeKind, RoutablePort, RoutablePortDirection,
    RoutableSequentialPrimitive,
};
use super::syntax::{tokenize, Token};

impl fmt::Display for RoutableDesign {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(output, "rcir {};", self.version)?;
        writeln!(output, "stage routable;")?;
        writeln!(output, "target {};", quoted(&self.target))?;
        writeln!(output, "top {};", quoted(&self.top))?;

        let mut modules = self.modules.iter().collect::<Vec<_>>();
        modules.sort_by(|left, right| left.name.cmp(&right.name));
        for module in modules {
            writeln!(output)?;
            match &module.body {
                RoutableModuleBody::Leaf { nodes } => write_leaf(output, module, nodes)?,
                RoutableModuleBody::Composite { instances, nets } => {
                    write_composite(output, module, instances, nets)?
                }
            }
        }
        Ok(())
    }
}

impl FromStr for RoutableDesign {
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

        let mut modules = Vec::new();
        while !parser.is_done() {
            if parser.consume_keyword("leaf") {
                modules.push(parser.parse_leaf()?);
            } else if parser.consume_keyword("module") {
                modules.push(parser.parse_composite()?);
            } else {
                return Err(parser.unexpected("`leaf` or `module`"));
            }
        }

        let design = Self {
            version,
            target,
            top,
            modules,
        };
        design.validate()?;
        Ok(design)
    }
}

fn write_leaf(
    output: &mut fmt::Formatter<'_>,
    module: &RoutableModule,
    nodes: &[RoutableNode],
) -> fmt::Result {
    writeln!(output, "leaf {} {{", quoted(&module.name))?;
    write_ports(output, &module.ports)?;
    let mut nodes = nodes.iter().collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.id);
    for node in nodes {
        write!(output, "  node {} ", node.id)?;
        match &node.kind {
            RoutableNodeKind::Input { name } => write!(output, "input {}", quoted(name))?,
            RoutableNodeKind::Output { name } => write!(output, "output {}", quoted(name))?,
            RoutableNodeKind::Not => write!(output, "logic not")?,
            RoutableNodeKind::And => write!(output, "logic and")?,
            RoutableNodeKind::Or => write!(output, "logic or")?,
            RoutableNodeKind::Xor => write!(output, "logic xor")?,
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
    write_ports(output, &module.ports)?;

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

fn write_ports(output: &mut fmt::Formatter<'_>, ports: &[RoutablePort]) -> fmt::Result {
    let mut ports = ports.iter().collect::<Vec<_>>();
    ports.sort_by(|left, right| left.name.cmp(&right.name));
    for port in ports {
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

    fn parse_leaf(&mut self) -> eyre::Result<RoutableModule> {
        let name = self.expect_string()?;
        self.expect_symbol('{')?;
        let mut ports = Vec::new();
        let mut nodes = Vec::new();
        while !self.consume_symbol('}') {
            if self.consume_keyword("port") {
                ports.push(self.parse_port()?);
            } else if self.consume_keyword("node") {
                nodes.push(self.parse_node()?);
            } else {
                return Err(self.unexpected("`port`, `node`, or `}`"));
            }
        }
        Ok(RoutableModule {
            name,
            ports,
            body: RoutableModuleBody::Leaf { nodes },
        })
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
                "and" => RoutableNodeKind::And,
                "or" => RoutableNodeKind::Or,
                "xor" => RoutableNodeKind::Xor,
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
                let value = u32::try_from(value).map_err(|_| self.unexpected("32-bit number"))?;
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

    fn sample_design() -> RoutableDesign {
        RoutableDesign {
            version: ROUTABLE_IR_VERSION,
            target: ROUTABLE_IR_TARGET.to_owned(),
            top: "top".to_owned(),
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
