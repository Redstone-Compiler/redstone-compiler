use std::collections::HashMap;

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

    pub fn prepare_place(self) -> eyre::Result<Self> {
        let mut transform = LogicGraphTransformer::new(self);
        transform.decompose_xor()?;
        transform.decompose_and()?;
        transform.remove_double_neg_expression();
        Ok(transform.finish())
    }
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
            'a'..='z' => {
                let mut result = String::new();

                while self.stmt.len() != next_ptr
                    && matches!(self.stmt.chars().nth(next_ptr).unwrap(), 'a'..='z' | '0'..='9')
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
        let out_cout = LogicGraph::from_stmt("(a&b)|(s&cin)", "cout")?;

        let mut fa = out_s.clone();
        fa.graph.merge(out_cout.graph);
        fa.prepare_place()
    }

    pub fn buffered_full_adder_graph() -> eyre::Result<LogicGraph> {
        let and_0 = LogicGraph::from_stmt("a&b", "c")?;
        let xor_o = LogicGraph::from_stmt("(~(c|~a))|(~(c|~b))", "i")?;
        let and_1 = LogicGraph::from_stmt("i&cin", "d")?;
        let out_s = LogicGraph::from_stmt("(~(d|~i))|(~(d|~cin))", "s")?;

        let out_cout = LogicGraph::from_stmt("(a&b)|(s&cin)", "cout")?;

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
    use crate::graph::Graph;
    use crate::transform::logic::LogicGraphTransformer;

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
