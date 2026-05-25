use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use super::Graph;
use crate::graph::{GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::{Logic, LogicType};
use crate::transform::logic::LogicGraphTransformer;

#[derive(Debug, Clone, derive_more::Deref)]
pub struct LogicGraph {
    #[deref]
    pub graph: Graph,
}

impl LogicGraph {
    pub fn from_stmt(stmt: &str, output: &str) -> eyre::Result<LogicGraph> {
        LogicGraphBuilder::new(stmt.to_string()).build(output.to_string())
    }

    pub fn from_assignments<I>(assignments: I) -> eyre::Result<LogicGraph>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let mut graphs = assignments
            .into_iter()
            .map(|(output, expr)| LogicGraph::from_stmt(&expr, &output))
            .collect::<eyre::Result<Vec<_>>>()?
            .into_iter();

        let Some(mut graph) = graphs.next() else {
            eyre::bail!("expected at least one logic assignment");
        };

        for next in graphs {
            graph.graph.merge(next.graph);
        }

        Ok(graph)
    }

    pub fn prepare_place(self) -> eyre::Result<Self> {
        let mut transform = LogicGraphTransformer::new(self);
        transform.decompose_xor()?;
        transform.decompose_and()?;
        transform.remove_double_neg_expression();
        transform.optimize_cse()?;
        transform.insert_buffers_for_direct_or_to_or()?;
        Ok(transform.finish())
    }

    pub fn truth_table(&self) -> eyre::Result<LogicTruthTable> {
        LogicTruthTable::from_graph(self)
    }

    pub fn externally_observable_output_source_ids(&self) -> HashSet<GraphNodeId> {
        self.nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(_) => {
                    let producer = self.find_node_by_id(node.inputs[0])?;
                    let has_internal_consumers = producer.outputs.iter().any(|output| {
                        self.find_node_by_id(*output)
                            .is_some_and(|node| !matches!(node.kind, GraphNodeKind::Output(_)))
                    });
                    (!has_internal_consumers).then_some(producer.id)
                }
                _ => None,
            })
            .collect()
    }

    pub fn named_outputs(&self) -> Vec<(String, GraphNodeId)> {
        let mut outputs = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) => Some((name.clone(), node.inputs[0])),
                _ => None,
            })
            .collect::<Vec<_>>();
        outputs.sort_by(|(a, _), (b, _)| a.cmp(b));
        outputs
    }

    pub fn terminal_sources(&self) -> Vec<GraphNodeId> {
        self.outputs()
            .into_iter()
            .filter(|node_id| {
                self.find_node_by_id(*node_id)
                    .is_some_and(|node| !matches!(node.kind, GraphNodeKind::Output(_)))
            })
            .sorted()
            .collect()
    }

    pub fn attach_outputs<I>(mut self, outputs: I) -> eyre::Result<Self>
    where
        I: IntoIterator<Item = (String, GraphNodeId)>,
    {
        let mut next_id = self.graph.max_node_id().map_or(0, |id| id + 1);
        for (name, source_id) in outputs {
            if self.find_node_by_id(source_id).is_none() {
                eyre::bail!("cannot attach output {name}: missing source node {source_id}");
            }

            self.graph.nodes.push(GraphNode {
                id: next_id,
                kind: GraphNodeKind::Output(name),
                inputs: vec![source_id],
                ..Default::default()
            });
            next_id += 1;
        }

        self.graph.nodes.sort_by_key(|node| node.id);
        self.graph.build_outputs();
        self.graph.build_producers();
        self.graph.build_consumers();
        self.graph.verify()?;
        Ok(self)
    }

    pub fn attach_anonymous_outputs(self) -> eyre::Result<Self> {
        let outputs = self
            .terminal_sources()
            .into_iter()
            .enumerate()
            .map(|(index, source_id)| (format!("#{index}"), source_id))
            .collect::<Vec<_>>();
        self.attach_outputs(outputs)
    }

    pub fn externally_observable_truth_table(&self) -> eyre::Result<LogicTruthTable> {
        let table = self.truth_table()?;
        let output_source_ids = self.externally_observable_output_source_ids();
        let mut output_names = self
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) => output_source_ids
                    .contains(&node.inputs[0])
                    .then_some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        if output_names.is_empty() {
            return Ok(table);
        }

        output_names.sort();
        table.select_outputs(&output_names)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicTruthTable {
    pub input_names: Vec<String>,
    pub output_tables: HashMap<String, Vec<bool>>,
}

impl LogicTruthTable {
    fn from_graph(graph: &LogicGraph) -> eyre::Result<Self> {
        let mut inputs = graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Input(name) => Some((node.id, name.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        inputs.sort_by(|(_, a), (_, b)| a.cmp(b));

        let mut outputs = graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                GraphNodeKind::Output(name) => Some((node.id, name.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        if outputs.is_empty() {
            outputs = graph
                .outputs()
                .into_iter()
                .map(|node_id| (node_id, format!("#{node_id}")))
                .collect();
        }
        outputs.sort_by(|(_, a), (_, b)| a.cmp(b));

        let mut output_tables = outputs
            .iter()
            .map(|(_, name)| (name.clone(), Vec::new()))
            .collect::<HashMap<_, _>>();

        for mask in 0..(1usize << inputs.len()) {
            let mut values = HashMap::<GraphNodeId, bool>::new();
            for (index, (input_id, _)) in inputs.iter().enumerate() {
                values.insert(*input_id, (mask & (1 << index)) != 0);
            }

            for node_id in graph.topological_order() {
                if values.contains_key(&node_id) {
                    continue;
                }

                let node = graph.find_node_by_id(node_id).unwrap();
                let value = match &node.kind {
                    GraphNodeKind::Input(_) => continue,
                    GraphNodeKind::Logic(logic) => match logic.logic_type {
                        LogicType::Not => !values[&node.inputs[0]],
                        LogicType::And => node.inputs.iter().all(|input| values[input]),
                        LogicType::Or => node.inputs.iter().any(|input| values[input]),
                        LogicType::Xor => {
                            node.inputs.iter().filter(|input| values[input]).count() % 2 == 1
                        }
                    },
                    GraphNodeKind::Output(_) => values[&node.inputs[0]],
                    _ => eyre::bail!("unsupported node kind in truth table: {:?}", node.kind),
                };
                values.insert(node_id, value);
            }

            for (output_id, output_name) in &outputs {
                output_tables
                    .get_mut(output_name)
                    .unwrap()
                    .push(values[output_id]);
            }
        }

        Ok(Self {
            input_names: inputs.into_iter().map(|(_, name)| name).collect(),
            output_tables,
        })
    }

    pub fn output_table_set(&self) -> std::collections::HashSet<Vec<bool>> {
        self.output_tables.values().cloned().collect()
    }

    pub fn select_outputs(&self, output_names: &[&str]) -> eyre::Result<Self> {
        let mut output_tables = HashMap::new();
        for output_name in output_names {
            let Some(output_table) = self.output_tables.get(*output_name) else {
                eyre::bail!("missing output truth table: {output_name}");
            };
            output_tables.insert((*output_name).to_owned(), output_table.clone());
        }

        Ok(Self {
            input_names: self.input_names.clone(),
            output_tables,
        })
    }

    pub fn contains_output_tables(&self, expected: &LogicTruthTable) -> bool {
        self.input_names.len() == expected.input_names.len()
            && self
                .output_table_set()
                .is_superset(&expected.output_table_set())
    }

    pub fn contains_output_tables_under_input_permutation(
        &self,
        expected: &LogicTruthTable,
    ) -> bool {
        if self.input_names.len() != expected.input_names.len() {
            return false;
        }

        let input_count = self.input_names.len();
        let actual = self.output_table_set();
        (0..input_count)
            .permutations(input_count)
            .any(|permutation| {
                actual.is_superset(&permuted_output_table_set(expected, &permutation))
            })
    }
}

fn permuted_output_table_set(
    table: &LogicTruthTable,
    generated_to_expected_input: &[usize],
) -> HashSet<Vec<bool>> {
    table
        .output_tables
        .values()
        .map(|output_table| {
            (0..output_table.len())
                .map(|generated_mask| {
                    let expected_mask = generated_to_expected_input.iter().enumerate().fold(
                        0usize,
                        |mask, (generated_index, expected_index)| {
                            if generated_mask & (1 << generated_index) != 0 {
                                mask | (1 << expected_index)
                            } else {
                                mask
                            }
                        },
                    );
                    output_table[expected_mask]
                })
                .collect()
        })
        .collect()
}

#[derive(Default)]
pub struct LogicGraphBuilder {
    stmt: String,
    node_id: usize,
    ptr: usize,
    nodes: Vec<GraphNode>,
    inputs: HashMap<String, GraphNodeId>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LogicStringTokenType {
    Ident(String),
    And,
    Or,
    Xor,
    Not,
    ParStart,
    ParEnd,
    Eof,
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()
}

impl LogicGraphBuilder {
    pub fn new(stmt: String) -> Self {
        LogicGraphBuilder {
            stmt,
            ..Default::default()
        }
    }

    pub fn build(mut self, output_name: String) -> eyre::Result<LogicGraph> {
        self.do_parse(output_name);

        let mut graph = Graph {
            nodes: self.nodes.clone(),
            ..Default::default()
        };
        graph.build_outputs();

        Ok(LogicGraph { graph })
    }

    fn next_id(&mut self) -> usize {
        let id = self.node_id;
        self.node_id += 1;
        id
    }

    fn next_ptr(&mut self) -> usize {
        let ptr = self.ptr;
        self.ptr += 1;
        ptr
    }

    fn new_node(&mut self, kind: GraphNodeKind, inputs: Vec<GraphNodeId>) -> GraphNodeId {
        let node = GraphNode {
            id: self.next_id(),
            kind,
            inputs,
            ..Default::default()
        };
        self.nodes.push(node);
        self.nodes.last().unwrap().clone().id
    }

    fn new_input_node(&mut self, name: String) -> GraphNodeId {
        if self.inputs.contains_key(&name) {
            return self.inputs[&name];
        }

        let id = self.new_node(GraphNodeKind::Input(name.clone()), vec![]);
        self.inputs.insert(name, id);
        id
    }

    fn new_output_node(&mut self, name: String, input: GraphNodeId) -> GraphNodeId {
        self.new_node(GraphNodeKind::Output(name), vec![input])
    }

    fn new_logic_node(&mut self, logic_type: LogicType, inputs: Vec<GraphNodeId>) -> GraphNodeId {
        self.new_node(GraphNodeKind::Logic(Logic { logic_type }), inputs)
    }

    fn skip_ws(&mut self) {
        let mut ptr = self.ptr;

        while self.stmt.len() != ptr && matches!(self.stmt.chars().nth(ptr).unwrap(), ' ' | '\n') {
            ptr = self.next_ptr();
        }
    }

    fn next(&mut self) -> LogicStringTokenType {
        self.skip_ws();

        if self.ptr == self.stmt.len() {
            return LogicStringTokenType::Eof;
        }

        let mut next_ptr = self.next_ptr();

        match self.stmt.chars().nth(next_ptr).unwrap() {
            '&' => LogicStringTokenType::And,
            '^' => LogicStringTokenType::Xor,
            '|' => LogicStringTokenType::Or,
            '(' => LogicStringTokenType::ParStart,
            ')' => LogicStringTokenType::ParEnd,
            '~' => LogicStringTokenType::Not,
            ch if is_ident_start(ch) => {
                let mut result = String::new();

                while self.stmt.len() != next_ptr
                    && is_ident_continue(self.stmt.chars().nth(next_ptr).unwrap())
                {
                    result.push(self.stmt.chars().nth(next_ptr).unwrap());
                    next_ptr = self.next_ptr();
                }

                self.ptr -= 1;

                LogicStringTokenType::Ident(result)
            }
            _ => unimplemented!(),
        }
    }

    fn lookup(&mut self) -> LogicStringTokenType {
        let cur_ptr = self.ptr;
        let lookup = self.next();
        self.ptr = cur_ptr;
        lookup
    }

    fn do_parse(&mut self, output_name: String) {
        let node = self.parse_or();
        self.new_output_node(output_name, node);
    }

    fn parse_or(&mut self) -> GraphNodeId {
        let mut outputs = Vec::new();

        let parse_and = self.parse_and();
        outputs.push(parse_and);
        while let LogicStringTokenType::Or = self.lookup() {
            self.next();
            outputs.push(self.parse_and());
        }

        if outputs.len() == 1 {
            return parse_and;
        }

        self.new_logic_node(LogicType::Or, outputs)
    }

    fn parse_and(&mut self) -> GraphNodeId {
        let mut outputs = Vec::new();

        let parse_xor = self.parse_xor();
        outputs.push(parse_xor);
        while let LogicStringTokenType::And = self.lookup() {
            self.next();
            outputs.push(self.parse_xor());
        }

        if outputs.len() == 1 {
            return parse_xor;
        }

        self.new_logic_node(LogicType::And, outputs)
    }

    fn parse_xor(&mut self) -> GraphNodeId {
        let mut outputs = Vec::new();

        let parse_par = self.parse_par();
        outputs.push(parse_par);
        while let LogicStringTokenType::Xor = self.lookup() {
            self.next();
            outputs.push(self.parse_par());
        }

        if outputs.len() == 1 {
            return parse_par;
        }

        self.new_logic_node(LogicType::Xor, outputs)
    }

    fn parse_par(&mut self) -> GraphNodeId {
        match self.lookup() {
            LogicStringTokenType::Ident(ident) => {
                self.next();
                self.new_input_node(ident)
            }
            LogicStringTokenType::Not => {
                self.next();

                let node = match self.lookup() {
                    LogicStringTokenType::Ident(ident) => {
                        self.next();
                        self.new_input_node(ident)
                    }
                    LogicStringTokenType::ParStart => self.parse_or(),
                    _ => panic!(),
                };
                self.new_logic_node(LogicType::Not, vec![node])
            }
            LogicStringTokenType::ParStart => {
                self.next();
                let node = self.parse_or();

                if self.next() != LogicStringTokenType::ParEnd {
                    panic!();
                }

                node
            }
            _ => panic!("{:?}", self.lookup()),
        }
    }
}

pub mod predefined_logics {
    use super::LogicGraph;

    pub fn and_graph() -> eyre::Result<LogicGraph> {
        LogicGraph::from_stmt("a&b", "c")?.prepare_place()
    }

    pub fn xor_graph() -> eyre::Result<LogicGraph> {
        LogicGraph::from_stmt("a^b", "c")?.prepare_place()
    }

    pub fn buffered_xor_graph() -> eyre::Result<LogicGraph> {
        // c := (~((a&b)|~a))|(~((a&b)|~b))
        let logic_graph1 = LogicGraph::from_stmt("a&b", "c")?;
        let logic_graph2 = LogicGraph::from_stmt("(~(c|~a))|(~(c|~b))", "d")?;

        let mut fm = logic_graph1.clone();
        fm.graph.merge(logic_graph2.graph);
        fm.prepare_place()
    }

    pub fn half_adder_graph() -> eyre::Result<LogicGraph> {
        let s = LogicGraph::from_stmt("a^b", "s")?;
        let c = LogicGraph::from_stmt("a&b", "c")?;

        let mut ha = s.clone();
        ha.graph.merge(c.graph);
        ha.prepare_place()
    }

    pub fn buffered_half_adder_graph() -> eyre::Result<LogicGraph> {
        let and_0 = LogicGraph::from_stmt("a&b", "c")?;
        let xor_o = LogicGraph::from_stmt("(~(c|~a))|(~(c|~b))", "i")?;
        let and_1 = LogicGraph::from_stmt("i&cin", "d")?;
        let out_s = LogicGraph::from_stmt("(~(d|~i))|(~(d|~cin))", "s")?;

        let mut ha = and_0.clone();
        ha.graph.merge(xor_o.graph);
        ha.graph.merge(and_1.graph);
        ha.graph.merge(out_s.graph);
        ha.prepare_place()
    }

    pub fn full_adder_graph() -> eyre::Result<LogicGraph> {
        let out_s = LogicGraph::from_stmt("(a^b)^cin", "s")?;
        let out_cout = LogicGraph::from_stmt("(a&b)|((a^b)&cin)", "cout")?;

        let mut fa = out_s.clone();
        fa.graph.merge(out_cout.graph);
        fa.prepare_place()
    }

    pub fn buffered_full_adder_graph() -> eyre::Result<LogicGraph> {
        let and_0 = LogicGraph::from_stmt("a&b", "c")?;
        let xor_o = LogicGraph::from_stmt("(~(c|~a))|(~(c|~b))", "i")?;
        let and_1 = LogicGraph::from_stmt("i&cin", "d")?;
        let out_s = LogicGraph::from_stmt("(~(d|~i))|(~(d|~cin))", "s")?;

        let out_cout = LogicGraph::from_stmt("c|d", "cout")?;

        let mut fa = and_0.clone();
        fa.graph.merge(xor_o.graph);
        fa.graph.merge(and_1.graph);
        fa.graph.merge(out_s.graph);
        fa.graph.merge(out_cout.graph);
        fa.prepare_place()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::graph::logic::LogicGraph;
    use crate::graph::{Graph, GraphNodeKind};
    use crate::logic::LogicType;
    use crate::transform::logic::LogicGraphTransformer;

    #[test]
    fn truth_table_reports_xor_output() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("a^b", "s")?;
        let table = graph.truth_table()?;

        assert_eq!(table.input_names, vec!["a", "b"]);
        assert_eq!(table.output_tables["s"], vec![false, true, true, false]);

        Ok(())
    }

    #[test]
    fn truth_table_can_compare_full_adder_outputs_by_function() -> eyre::Result<()> {
        let graph = super::predefined_logics::buffered_full_adder_graph()?;
        let table = graph.truth_table()?;
        let sum = (0..8)
            .map(|mask: usize| mask.count_ones() % 2 == 1)
            .collect::<Vec<_>>();
        let carry = (0..8)
            .map(|mask: usize| mask.count_ones() >= 2)
            .collect::<Vec<_>>();

        assert!(table.output_table_set().contains(&sum));
        assert!(table.output_table_set().contains(&carry));

        Ok(())
    }

    #[test]
    fn truth_table_can_compare_unbuffered_full_adder_outputs_by_function() -> eyre::Result<()> {
        let graph = super::predefined_logics::full_adder_graph()?;
        let table = graph.truth_table()?;
        let sum = (0..8)
            .map(|mask: usize| mask.count_ones() % 2 == 1)
            .collect::<Vec<_>>();
        let carry = (0..8)
            .map(|mask: usize| mask.count_ones() >= 2)
            .collect::<Vec<_>>();

        assert!(table.output_table_set().contains(&sum));
        assert!(table.output_table_set().contains(&carry));

        Ok(())
    }

    #[test]
    fn prepare_place_inserts_buffers_between_direct_or_nodes() -> eyre::Result<()> {
        let mut graph = LogicGraph::from_stmt("a|b", "x")?;
        graph.graph.merge(LogicGraph::from_stmt("x|c", "y")?.graph);
        let graph = graph.prepare_place()?;

        let has_direct_or_to_or = graph.nodes.iter().any(|node| {
            matches!(&node.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
                && node.outputs.iter().any(|output_id| {
                    graph.find_node_by_id(*output_id).is_some_and(|output| {
                        matches!(&output.kind, GraphNodeKind::Logic(logic) if logic.logic_type == LogicType::Or)
                    })
                })
        });
        let auto_buffer_count = graph
            .nodes
            .iter()
            .filter(|node| node.tag == "auto-buffer")
            .count();

        assert!(!has_direct_or_to_or);
        assert_eq!(auto_buffer_count, 2);

        Ok(())
    }

    #[test]
    fn truth_table_can_check_required_output_tables() -> eyre::Result<()> {
        let graph = super::predefined_logics::buffered_xor_graph()?;
        let required = LogicGraph::from_stmt("a^b", "s")?;

        assert!(graph
            .truth_table()?
            .contains_output_tables(&required.truth_table()?));

        Ok(())
    }

    #[test]
    fn from_assignments_builds_half_adder() -> eyre::Result<()> {
        let graph = LogicGraph::from_assignments([
            ("s".to_owned(), "a^b".to_owned()),
            ("c".to_owned(), "a&b".to_owned()),
        ])?;
        let table = graph.truth_table()?;

        assert_eq!(table.input_names, vec!["a", "b"]);
        assert_eq!(table.output_tables["s"], vec![false, true, true, false]);
        assert_eq!(table.output_tables["c"], vec![false, false, false, true]);

        Ok(())
    }

    #[test]
    fn logic_parser_accepts_verilog_style_identifiers() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("A_0&carry_in", "SUM_0")?;
        let table = graph.truth_table()?;

        assert_eq!(table.input_names, vec!["A_0", "carry_in"]);
        assert_eq!(
            table.output_tables["SUM_0"],
            vec![false, false, false, true]
        );

        Ok(())
    }

    #[test]
    fn attach_anonymous_outputs_names_terminal_sources() -> eyre::Result<()> {
        let mut graph = LogicGraph::from_stmt("a&b", "out")?;
        graph.graph.remove_output("out");
        let graph = graph.attach_anonymous_outputs()?;

        assert_eq!(graph.named_outputs().len(), 1);
        assert_eq!(graph.named_outputs()[0].0, "#0");
        assert_eq!(graph.terminal_sources().len(), 0);
        assert_eq!(
            graph.truth_table()?.output_tables["#0"],
            vec![false, false, false, true]
        );

        Ok(())
    }

    #[test]
    fn unittest_logicgraph_full_adder() -> eyre::Result<()> {
        // s = (a ^ b) ^ cin;
        // cout = (a & b) | (s & cin);
        let logic_graph1 = LogicGraph::from_stmt("(a1^b1)^cin1", "s1")?;
        let logic_graph2 = LogicGraph::from_stmt("(a1&b1)|(s1&cin1)", "cout1")?;

        let mut fa1 = logic_graph1.clone();
        fa1.graph.merge(logic_graph2.graph);

        let logic_graph3 = LogicGraph::from_stmt("(a2^b2)^cout1", "s2")?;
        let logic_graph4 = LogicGraph::from_stmt("(a2&b2)|(s2&cout1)", "cout2")?;

        let mut fa2 = logic_graph3.clone();
        fa2.graph.merge(logic_graph4.graph);

        let mut fa = fa1.clone();
        fa.graph.merge(fa2.graph);

        let mut transform = LogicGraphTransformer::new(fa);
        transform.decompose_xor()?;
        transform.decompose_and()?;
        transform.remove_double_neg_expression();
        let clusters = transform.cluster(false);
        let clusters = clusters.iter().map(|x| x.to_subgraph()).collect_vec();
        println!("{}", transform.graph.to_graphviz_with_clusters(&clusters));

        let finish = transform.finish();
        println!("{:?}", finish.graph.critical_path());
        println!("{:?}", finish.graph.clone().topological_order());

        let g: Graph = finish.graph.split_with_outputs().into();
        println!("{}", g.to_graphviz());

        let splits = finish.graph.split_with_outputs();

        let mut graph: Graph = (&finish.graph.split_with_outputs()[0]).into();
        graph = graph.rebuild_node_ids();
        println!("{}", graph.to_graphviz());

        let mut transform = LogicGraphTransformer::new(LogicGraph { graph });
        transform.optimize()?;

        let mut finish = transform.finish();
        println!("{}", finish.to_graphviz());

        #[allow(clippy::needless_range_loop)]
        for index in 1..2 {
            let mut graph: Graph = (&splits[index]).into();
            graph = graph.rebuild_node_ids();
            println!("{}", graph.to_graphviz());

            let mut transform = LogicGraphTransformer::new(LogicGraph { graph });
            transform.optimize()?;

            let finish2 = transform.finish();
            println!("{}", finish2.to_graphviz());
            finish.graph.concat(finish2.graph);
        }

        println!("{}", finish.to_graphviz());

        Ok(())
    }
}
