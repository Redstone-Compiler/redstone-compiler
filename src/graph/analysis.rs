use std::collections::HashMap;

use itertools::Itertools;

use crate::graph::logic::LogicGraph;
use crate::graph::{Graph, GraphNodeId, GraphNodeKind};
use crate::logic::LogicType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquivalentExpressionGroup {
    pub expression: String,
    pub node_ids: Vec<GraphNodeId>,
}

pub fn equivalent_expression_groups(graph: &LogicGraph) -> Vec<EquivalentExpressionGroup> {
    let mut memo = HashMap::new();
    let mut groups: HashMap<String, Vec<GraphNodeId>> = HashMap::new();

    for node in &graph.nodes {
        if !node.kind.is_logic() {
            continue;
        }

        if let Some(expression) = canonical_expression(&graph.graph, node.id, &mut memo) {
            groups.entry(expression).or_default().push(node.id);
        }
    }

    groups
        .into_iter()
        .filter_map(|(expression, mut node_ids)| {
            if node_ids.len() <= 1 {
                return None;
            }
            node_ids.sort();
            Some(EquivalentExpressionGroup {
                expression,
                node_ids,
            })
        })
        .sorted_by(|a, b| {
            b.node_ids
                .len()
                .cmp(&a.node_ids.len())
                .then_with(|| a.expression.cmp(&b.expression))
        })
        .collect()
}

fn canonical_expression(
    graph: &Graph,
    node_id: GraphNodeId,
    memo: &mut HashMap<GraphNodeId, String>,
) -> Option<String> {
    if let Some(expression) = memo.get(&node_id) {
        return Some(expression.clone());
    }

    let node = graph.find_node_by_id(node_id)?;
    let expression = match &node.kind {
        GraphNodeKind::Input(name) => format!("Input({name})"),
        GraphNodeKind::Logic(logic) => {
            let mut inputs = node
                .inputs
                .iter()
                .filter_map(|input| canonical_expression(graph, *input, memo))
                .collect_vec();

            if matches!(
                logic.logic_type,
                LogicType::Or | LogicType::And | LogicType::Xor
            ) {
                inputs.sort();
            }

            format!("{}({})", logic.logic_type.name(), inputs.join(", "))
        }
        GraphNodeKind::Output(name) => {
            if node.inputs.len() == 1 {
                canonical_expression(graph, node.inputs[0], memo)?
            } else {
                format!("Output({name})")
            }
        }
        _ => return None,
    };

    memo.insert(node_id, expression.clone());
    Some(expression)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::logic::{predefined_logics, LogicGraph};

    #[test]
    fn equivalent_expression_groups_reports_duplicate_logic_nodes() -> eyre::Result<()> {
        let graph = LogicGraph::from_stmt("~a|~a", "out")?;

        let groups = equivalent_expression_groups(&graph);

        assert!(groups
            .iter()
            .any(|group| { group.expression == "Not(Input(a))" && group.node_ids.len() == 2 }));

        Ok(())
    }

    #[test]
    fn buffered_full_adder_has_visible_duplicate_not_expressions() -> eyre::Result<()> {
        let graph = predefined_logics::buffered_full_adder_graph()?;

        let groups = equivalent_expression_groups(&graph);

        assert!(groups
            .iter()
            .any(|group| group.expression == "Not(Input(a))"));
        assert!(groups
            .iter()
            .any(|group| group.expression == "Not(Input(b))"));
        assert!(groups
            .iter()
            .any(|group| group.expression == "Not(Input(cin))"));

        Ok(())
    }
}
