//! NOR netlist view of a logic graph for the exact placer.
//!
//! Every physical net is driven either by an input switch or by one or more
//! redstone torches. A torch computes NOR over the signals that power its
//! support block, so OR nodes are folded into the input set of the NOT that
//! consumes them.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use eyre::{bail, ensure, ContextCompat};

use crate::graph::logic::LogicGraph;
use crate::graph::{GraphNodeId, GraphNodeKind};
use crate::logic::LogicType;

pub type NetId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetDriver {
    Input(String),
    /// A torch: NOR of its inputs.
    Gate,
    /// Dust and blocks: OR of its inputs, no torch. Netlists from a logic
    /// graph fold ORs into the gates that read them; a wide NOR can instead
    /// read OR nets, which construction places as steps of their own.
    Or,
}

#[derive(Debug, Clone)]
pub struct Net {
    pub name: String,
    pub node_id: GraphNodeId,
    pub driver: NetDriver,
    /// Nets whose OR powers this gate's torch support (or that an OR net
    /// joins). Empty for inputs.
    pub gate_inputs: Vec<NetId>,
}

#[derive(Debug, Clone)]
pub struct NorNetlist {
    pub nets: Vec<Net>,
    pub outputs: Vec<(String, NetId)>,
    /// The state nets of a sequential cell (a latch's output, say): every
    /// cycle of the netlist passes through one, and a gate that reads one
    /// reads the stored state. Each becomes a case bit after the inputs
    /// (`case_bits`), and the cases are the settled states (`valid_cases`).
    /// Empty for combinational logic.
    pub state: Vec<NetId>,
}

impl NorNetlist {
    pub fn from_logic_graph(graph: &LogicGraph) -> eyre::Result<Self> {
        let mut net_of_node = HashMap::<GraphNodeId, NetId>::new();
        let mut nets = Vec::new();

        let mut node_ids = graph.nodes.iter().map(|node| node.id).collect::<Vec<_>>();
        node_ids.sort();
        for &node_id in &node_ids {
            let node = graph.find_node_by_id(node_id).unwrap();
            let driver = match &node.kind {
                GraphNodeKind::Input(name) => NetDriver::Input(name.clone()),
                GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Not => {
                    NetDriver::Gate
                }
                GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or => continue,
                GraphNodeKind::Output(_) => continue,
                other => bail!(
                    "exact placer supports NOT/OR logic only, found {}",
                    other.name()
                ),
            };
            let name = match &driver {
                NetDriver::Input(name) => name.clone(),
                NetDriver::Gate | NetDriver::Or => format!("g{node_id}"),
            };
            net_of_node.insert(node_id, nets.len());
            nets.push(Net {
                name,
                node_id,
                driver,
                gate_inputs: Vec::new(),
            });
        }

        // Expand each NOT input through OR nodes into the set of driving nets.
        fn collect_sources(
            graph: &LogicGraph,
            net_of_node: &HashMap<GraphNodeId, NetId>,
            node_id: GraphNodeId,
            sources: &mut BTreeSet<NetId>,
            depth: usize,
        ) -> eyre::Result<()> {
            ensure!(depth < 64, "OR nesting is too deep");
            if let Some(&net) = net_of_node.get(&node_id) {
                sources.insert(net);
                return Ok(());
            }
            let node = graph
                .find_node_by_id(node_id)
                .with_context(|| format!("missing node {node_id}"))?;
            match &node.kind {
                GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or => {
                    for &input in &node.inputs {
                        collect_sources(graph, net_of_node, input, sources, depth + 1)?;
                    }
                    Ok(())
                }
                other => bail!("unsupported NOR input {}", other.name()),
            }
        }

        for net in nets.iter_mut() {
            if net.driver != NetDriver::Gate {
                continue;
            }
            let node = graph.find_node_by_id(net.node_id).unwrap();
            ensure!(
                node.inputs.len() == 1,
                "NOT node {} must have one input",
                net.node_id
            );
            let mut sources = BTreeSet::new();
            collect_sources(graph, &net_of_node, node.inputs[0], &mut sources, 0)?;
            ensure!(!sources.is_empty(), "gate {} has no inputs", net.node_id);
            net.gate_inputs = sources.into_iter().collect();
        }

        let mut outputs = Vec::new();
        for &node_id in &node_ids {
            let node = graph.find_node_by_id(node_id).unwrap();
            if let GraphNodeKind::Output(name) = &node.kind {
                let source = node.inputs[0];
                let net = match net_of_node.get(&source) {
                    Some(&net) => net,
                    None => {
                        // An OR output needs a driver: realize it as NOT(NOR(...)).
                        let mut sources = BTreeSet::new();
                        collect_sources(graph, &net_of_node, source, &mut sources, 0)?;
                        let nor = nets.len();
                        nets.push(Net {
                            name: format!("{name}_n"),
                            node_id: source,
                            driver: NetDriver::Gate,
                            gate_inputs: sources.into_iter().collect(),
                        });
                        let buffered = nets.len();
                        nets.push(Net {
                            name: format!("g{source}"),
                            node_id: source,
                            driver: NetDriver::Gate,
                            gate_inputs: vec![nor],
                        });
                        net_of_node.insert(source, buffered);
                        buffered
                    }
                };
                outputs.push((name.clone(), net));
            }
        }
        outputs.sort();
        for (name, net) in &outputs {
            if nets[*net].driver == NetDriver::Gate && nets[*net].name.starts_with('g') {
                nets[*net].name = name.clone();
            }
        }
        ensure!(
            !outputs.is_empty(),
            "exact placer needs at least one output"
        );

        let netlist = Self {
            nets,
            outputs,
            state: Vec::new(),
        };
        netlist.check_acyclic()?;
        Ok(netlist)
    }

    /// No cycle except through a state net (whose readers read the stored
    /// state, so their edges from it are cut).
    pub(super) fn check_acyclic(&self) -> eyre::Result<()> {
        let mut state = vec![0u8; self.nets.len()];
        fn visit(netlist: &NorNetlist, net: NetId, state: &mut [u8]) -> eyre::Result<()> {
            match state[net] {
                1 => bail!("combinational loop through net {}", netlist.nets[net].name),
                2 => return Ok(()),
                _ => {}
            }
            state[net] = 1;
            for &input in &netlist.nets[net].gate_inputs {
                if !netlist.state.contains(&input) {
                    visit(netlist, input, state)?;
                }
            }
            state[net] = 2;
            Ok(())
        }
        for net in 0..self.nets.len() {
            visit(self, net, &mut state)?;
        }
        Ok(())
    }

    /// The same functions with every NOR of more than `max_fan_in` signals
    /// (at least 2) built on a chain of OR nets: `NOR(x1, x2, x3, x4)`
    /// becomes `s1 = OR(x1, x2)`, `s2 = OR(s1, x3)`, `NOR(s2, x4)`, its
    /// inputs taken in topological order (inputs first). Construction builds
    /// such a chain on one support block, a signal more per step
    /// (`construct.rs`, support blocks).
    pub fn chain_wide_gates(&self, max_fan_in: usize) -> eyre::Result<Self> {
        ensure!(max_fan_in >= 2, "a chain needs NORs of two signals");
        let rank = self
            .topological_order()
            .into_iter()
            .enumerate()
            .map(|(rank, net)| (net, rank))
            .collect::<HashMap<_, _>>();
        let mut netlist = self.clone();
        for gate in self.gates().collect::<Vec<_>>() {
            let mut inputs = self.nets[gate].gate_inputs.clone();
            if inputs.len() <= max_fan_in {
                continue;
            }
            let is_input = |net: NetId| matches!(self.nets[net].driver, NetDriver::Input(_));
            inputs.sort_by_key(|&input| (!is_input(input), rank[&input]));
            let last = inputs.pop().unwrap();
            let mut chain = inputs[0];
            for (index, &input) in inputs.iter().enumerate().skip(1) {
                netlist.nets.push(Net {
                    name: format!("{}_or{index}", self.nets[gate].name),
                    node_id: self.nets[gate].node_id,
                    driver: NetDriver::Or,
                    gate_inputs: vec![chain, input],
                });
                chain = netlist.nets.len() - 1;
            }
            netlist.nets[gate].gate_inputs = vec![chain, last];
        }
        netlist.check_acyclic()?;
        Ok(netlist)
    }

    /// One line per netlist: `g3=NOR(a,b); g4=NOR(a,g3); ...`, gates and OR
    /// nets in net order (`from_text` reads it back). A sequential netlist
    /// starts with `state(q,...)` and `out(...)` items.
    pub fn to_text(&self) -> String {
        let names = |nets: &mut dyn Iterator<Item = NetId>| {
            nets.map(|net| self.nets[net].name.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut items = Vec::new();
        if !self.state.is_empty() {
            items.push(format!("state({})", names(&mut self.state.iter().copied())));
            items.push(format!(
                "out({})",
                names(&mut self.outputs.iter().map(|(_, net)| *net))
            ));
        }
        items.extend(self.nets.iter().filter_map(|net| {
            let kind = match net.driver {
                NetDriver::Input(_) => return None,
                NetDriver::Gate => "NOR",
                NetDriver::Or => "OR",
            };
            let inputs = names(&mut net.gate_inputs.iter().copied());
            Some(format!("{}={kind}({inputs})", net.name))
        }));
        items.join("; ")
    }

    /// Reads `to_text`'s form. A name that is never defined is an input.
    /// `state(q,...)` names the state nets, which may be read before they
    /// are defined; every cycle must pass through one. The outputs are the
    /// nets in `out(...)`, or else the nets no other net reads, each under
    /// its own name.
    pub fn from_text(text: &str) -> eyre::Result<Self> {
        let list = |inner: &str| {
            inner
                .split(',')
                .map(|name| name.trim().to_owned())
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>()
        };
        let mut defined = Vec::<(String, NetDriver, Vec<String>)>::new();
        let mut state_names = Vec::new();
        let mut out_names = None;
        for item in text
            .split(';')
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            if let Some(inner) = item
                .strip_prefix("state(")
                .and_then(|r| r.strip_suffix(')'))
            {
                state_names.extend(list(inner));
                continue;
            }
            if let Some(inner) = item.strip_prefix("out(").and_then(|r| r.strip_suffix(')')) {
                out_names = Some(list(inner));
                continue;
            }
            let (name, rest) = item
                .split_once('=')
                .ok_or_else(|| eyre::eyre!("expected `name=NOR(...)`: {item}"))?;
            let rest = rest.trim();
            let (driver, inputs) = if let Some(inputs) = rest.strip_prefix("NOR(") {
                (NetDriver::Gate, inputs)
            } else if let Some(inputs) = rest.strip_prefix("OR(") {
                (NetDriver::Or, inputs)
            } else {
                eyre::bail!("expected NOR(...) or OR(...): {item}");
            };
            let inputs = inputs
                .strip_suffix(')')
                .ok_or_else(|| eyre::eyre!("missing `)`: {item}"))?;
            defined.push((name.trim().to_owned(), driver, list(inputs)));
        }
        let is_defined = |name: &str| defined.iter().any(|(defined, _, _)| defined == name);
        let inputs = defined
            .iter()
            .flat_map(|(_, _, reads)| reads)
            .filter(|read| !is_defined(read))
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut nets = inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| Net {
                node_id: index,
                driver: NetDriver::Input(input.clone()),
                name: input,
                gate_inputs: Vec::new(),
            })
            .collect::<Vec<_>>();
        for (name, driver, _) in &defined {
            nets.push(Net {
                name: name.clone(),
                node_id: nets.len(),
                driver: driver.clone(),
                gate_inputs: Vec::new(),
            });
        }
        let net_of = |nets: &[Net], name: &str| {
            nets.iter()
                .position(|net| net.name == name)
                .ok_or_else(|| eyre::eyre!("unknown net `{name}`"))
        };
        for (name, _, reads) in &defined {
            let net = net_of(&nets, name)?;
            nets[net].gate_inputs = reads
                .iter()
                .map(|read| net_of(&nets, read))
                .collect::<eyre::Result<_>>()?;
        }
        let state = state_names
            .iter()
            .map(|name| net_of(&nets, name))
            .collect::<eyre::Result<Vec<_>>>()?;
        let mut outputs = match out_names {
            Some(names) => names
                .iter()
                .map(|name| Ok((name.clone(), net_of(&nets, name)?)))
                .collect::<eyre::Result<Vec<_>>>()?,
            None => (0..nets.len())
                .filter(|&net| !matches!(nets[net].driver, NetDriver::Input(_)))
                .filter(|&net| !nets.iter().any(|other| other.gate_inputs.contains(&net)))
                .map(|net| (nets[net].name.clone(), net))
                .collect(),
        };
        outputs.sort();
        let netlist = Self {
            nets,
            outputs,
            state,
        };
        netlist.check_acyclic()?;
        Ok(netlist)
    }

    /// Torches on the slowest path to each output (OR nets add none).
    pub fn output_depths(&self) -> BTreeMap<String, usize> {
        let mut depth = vec![0usize; self.nets.len()];
        for net in self.topological_order() {
            let latest = self.nets[net]
                .gate_inputs
                .iter()
                .map(|&input| depth[input])
                .max()
                .unwrap_or(0);
            depth[net] = match self.nets[net].driver {
                NetDriver::Input(_) => 0,
                NetDriver::Gate => latest + 1,
                NetDriver::Or => latest,
            };
        }
        self.outputs
            .iter()
            .map(|(name, net)| (name.clone(), depth[*net]))
            .collect()
    }

    pub fn input_names(&self) -> Vec<String> {
        let mut names = self
            .nets
            .iter()
            .filter_map(|net| match &net.driver {
                NetDriver::Input(name) => Some(name.clone()),
                NetDriver::Gate | NetDriver::Or => None,
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn input_net(&self, name: &str) -> Option<NetId> {
        self.nets
            .iter()
            .position(|net| net.driver == NetDriver::Input(name.to_owned()))
    }

    pub fn gates(&self) -> impl Iterator<Item = NetId> + '_ {
        (0..self.nets.len()).filter(|&net| self.nets[net].driver == NetDriver::Gate)
    }

    /// Distinct multi-net input sets; their OR is what a shared torch support carries.
    pub fn input_sets(&self) -> Vec<Vec<NetId>> {
        self.gates()
            .map(|gate| self.nets[gate].gate_inputs.clone())
            .filter(|inputs| inputs.len() >= 2)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Value of every net for each input assignment. Bit `i` of a case index is
    /// the value of `input_names()[i]`.
    /// Bits of a case: the inputs (sorted by name), then the state nets.
    pub fn case_bits(&self) -> usize {
        self.input_names().len() + self.state.len()
    }

    /// Value of every net in every case (`case_bits`): bit `i` of a case is
    /// `input_names()[i]`, and bit `inputs + j` the stored value of
    /// `state[j]`, which its readers read. A state net's own value is what
    /// its gate computes; in the cases outside `valid_cases` it differs from
    /// the stored value.
    pub fn net_values(&self) -> Vec<Vec<bool>> {
        let input_names = self.input_names();
        let case_count = 1usize << self.case_bits();
        let mut values = vec![vec![false; case_count]; self.nets.len()];
        let order = self.topological_order();
        let read = |values: &[Vec<bool>], input: NetId, case: usize| match self
            .state
            .iter()
            .position(|&state| state == input)
        {
            Some(index) => case & (1 << (input_names.len() + index)) != 0,
            None => values[input][case],
        };
        for case in 0..case_count {
            for &net in &order {
                values[net][case] = match &self.nets[net].driver {
                    NetDriver::Input(name) => {
                        let index = input_names.iter().position(|n| n == name).unwrap();
                        case & (1 << index) != 0
                    }
                    NetDriver::Gate => !self.nets[net]
                        .gate_inputs
                        .iter()
                        .any(|&input| read(&values, input, case)),
                    NetDriver::Or => self.nets[net]
                        .gate_inputs
                        .iter()
                        .any(|&input| read(&values, input, case)),
                };
            }
        }
        values
    }

    /// The settled cases as a mask over `case_bits` cases: those whose
    /// stored state is what the state nets compute from it. Every case of a
    /// combinational netlist.
    pub fn valid_cases(&self) -> u64 {
        let case_count = 1usize << self.case_bits();
        let inputs = self.input_names().len();
        let values = self.net_values();
        (0..case_count)
            .filter(|&case| {
                self.state
                    .iter()
                    .enumerate()
                    .all(|(index, &net)| values[net][case] == (case & (1 << (inputs + index)) != 0))
            })
            .fold(0u64, |mask, case| mask | 1 << case)
    }

    /// The state a sequential netlist settles in when `inputs` (a case's
    /// input bits) are applied with `state` (its state bits, shifted down)
    /// stored: the state nets are recomputed from the stored state until
    /// they hold. `None` if that never settles.
    pub fn next_state(&self, inputs: usize, state: usize) -> Option<usize> {
        let shift = self.input_names().len();
        let values = self.net_values();
        let mut state = state;
        for _ in 0..=(1usize << self.state.len()) {
            let case = inputs | state << shift;
            let next = self
                .state
                .iter()
                .enumerate()
                .filter(|&(_, &net)| values[net][case])
                .fold(0usize, |next, (index, _)| next | 1 << index);
            if next == state {
                return Some(state);
            }
            state = next;
        }
        None
    }

    pub fn topological_order(&self) -> Vec<NetId> {
        let mut order = Vec::new();
        let mut done = vec![false; self.nets.len()];
        fn visit(netlist: &NorNetlist, net: NetId, done: &mut [bool], order: &mut Vec<NetId>) {
            if done[net] {
                return;
            }
            done[net] = true;
            for &input in &netlist.nets[net].gate_inputs {
                if !netlist.state.contains(&input) {
                    visit(netlist, input, done, order);
                }
            }
            order.push(net);
        }
        for net in 0..self.nets.len() {
            visit(self, net, &mut done, &mut order);
        }
        order
    }

    pub fn summary(&self) -> BTreeMap<String, Vec<String>> {
        self.gates()
            .map(|gate| {
                (
                    self.nets[gate].name.clone(),
                    self.nets[gate]
                        .gate_inputs
                        .iter()
                        .map(|&input| self.nets[input].name.clone())
                        .collect(),
                )
            })
            .collect()
    }
}
