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
                other => bail!("exact placer supports NOT/OR logic only, found {}", other.name()),
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
            ensure!(node.inputs.len() == 1, "NOT node {} must have one input", net.node_id);
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
        ensure!(!outputs.is_empty(), "exact placer needs at least one output");

        let netlist = Self { nets, outputs };
        netlist.check_acyclic()?;
        Ok(netlist)
    }

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
                visit(netlist, input, state)?;
            }
            state[net] = 2;
            Ok(())
        }
        for net in 0..self.nets.len() {
            visit(self, net, &mut state)?;
        }
        Ok(())
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
    pub fn net_values(&self) -> Vec<Vec<bool>> {
        let input_names = self.input_names();
        let case_count = 1usize << input_names.len();
        let mut values = vec![vec![false; case_count]; self.nets.len()];
        let order = self.topological_order();
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
                        .any(|&input| values[input][case]),
                    NetDriver::Or => self.nets[net]
                        .gate_inputs
                        .iter()
                        .any(|&input| values[input][case]),
                };
            }
        }
        values
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
                visit(netlist, input, done, order);
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
