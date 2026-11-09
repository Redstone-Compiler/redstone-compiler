//! Unsigned bit mapping with structural sharing and bounded primitive leaves.
//! Select/concat/resize are wiring; constants fold before physical generation.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Bit {
    Constant(bool),
    Input(String),
    Not(usize),
    And(usize, usize),
    Or(usize, usize),
    Xor(usize, usize),
}

#[derive(Clone, Default)]
struct Builder {
    bits: Vec<Bit>,
    intern: HashMap<Bit, usize>,
}

fn bit_inputs(bit: &Bit) -> Vec<usize> {
    match *bit {
        Bit::Not(a) => vec![a],
        Bit::And(a, b) | Bit::Or(a, b) | Bit::Xor(a, b) => vec![a, b],
        _ => vec![],
    }
}

fn bit_depth(id: usize, bits: &[Bit], cache: &mut HashMap<usize, usize>) -> usize {
    if let Some(&depth) = cache.get(&id) {
        return depth;
    }
    let inputs = bit_inputs(&bits[id]);
    let depth = if inputs.is_empty() {
        0
    } else {
        1 + inputs
            .into_iter()
            .map(|input| bit_depth(input, bits, cache))
            .max()
            .unwrap()
    };
    cache.insert(id, depth);
    depth
}

fn live_bits(bits: &[Bit], outputs: &[(String, usize)]) -> HashSet<usize> {
    let mut live = HashSet::new();
    let mut stack = outputs.iter().map(|(_, id)| *id).collect::<Vec<_>>();
    while let Some(id) = stack.pop() {
        if live.insert(id) {
            stack.extend(bit_inputs(&bits[id]));
        }
    }
    live
}

impl Builder {
    fn bit(&mut self, bit: Bit) -> usize {
        if let Some(&id) = self.intern.get(&bit) {
            return id;
        }
        let id = self.bits.len();
        self.bits.push(bit.clone());
        self.intern.insert(bit, id);
        id
    }
    fn constant(&mut self, value: bool) -> usize {
        self.bit(Bit::Constant(value))
    }
    fn not(&mut self, a: usize) -> usize {
        match self.bits[a] {
            Bit::Constant(v) => self.constant(!v),
            Bit::Not(v) => v,
            _ => self.bit(Bit::Not(a)),
        }
    }
    fn binary(&mut self, op: &str, mut a: usize, mut b: usize) -> usize {
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        if a == b {
            return if op == "xor" { self.constant(false) } else { a };
        }
        if matches!(self.bits[a], Bit::Not(v) if v == b)
            || matches!(self.bits[b], Bit::Not(v) if v == a)
        {
            return self.constant(op != "and");
        }
        for (x, y) in [(a, b), (b, a)] {
            if let Bit::Constant(v) = self.bits[x] {
                return match (op, v) {
                    ("and", false) | ("or", true) => x,
                    ("and", true) | ("or", false) | ("xor", false) => y,
                    ("xor", true) => self.not(y),
                    _ => unreachable!(),
                };
            }
        }
        self.bit(match op {
            "and" => Bit::And(a, b),
            "or" => Bit::Or(a, b),
            "xor" => Bit::Xor(a, b),
            _ => unreachable!(),
        })
    }
    /// Reassociate only fanout-free AND/OR cones. Shared intermediate signals
    /// remain boundaries, so balancing cannot duplicate their logic.
    fn balanced(&self, outputs: &[(String, usize)]) -> (Self, Vec<(String, usize)>) {
        let live = live_bits(&self.bits, outputs);
        let mut fanout = vec![0; self.bits.len()];
        for &id in &live {
            for input in bit_inputs(&self.bits[id]) {
                fanout[input] += 1;
            }
        }
        for (_, id) in outputs {
            fanout[*id] += 1;
        }
        let mut rebuilt = Self::default();
        // Preserve unused interface inputs as possible constant-source seeds.
        for bit in &self.bits {
            if matches!(bit, Bit::Input(_)) {
                rebuilt.bit(bit.clone());
            }
        }
        fn rebuild(
            id: usize,
            old: &Builder,
            fanout: &[usize],
            new: &mut Builder,
            memo: &mut HashMap<usize, usize>,
            depths: &mut HashMap<usize, usize>,
        ) -> usize {
            if let Some(&mapped) = memo.get(&id) {
                return mapped;
            }
            let mapped = match old.bits[id].clone() {
                Bit::And(_, _) | Bit::Or(_, _) => {
                    let is_and = matches!(old.bits[id], Bit::And(..));
                    fn collect(
                        id: usize,
                        root: bool,
                        is_and: bool,
                        old: &Builder,
                        fanout: &[usize],
                        operands: &mut Vec<usize>,
                    ) {
                        let same = if is_and {
                            matches!(old.bits[id], Bit::And(..))
                        } else {
                            matches!(old.bits[id], Bit::Or(..))
                        };
                        if same && (root || fanout[id] == 1) {
                            for input in bit_inputs(&old.bits[id]) {
                                collect(input, false, is_and, old, fanout, operands);
                            }
                        } else {
                            operands.push(id);
                        }
                    }
                    let mut operands = Vec::new();
                    collect(id, true, is_and, old, fanout, &mut operands);
                    let mut queue = std::collections::BinaryHeap::new();
                    for input in operands {
                        let input = rebuild(input, old, fanout, new, memo, depths);
                        queue.push(std::cmp::Reverse((
                            bit_depth(input, &new.bits, depths),
                            input,
                        )));
                    }
                    while queue.len() > 1 {
                        let std::cmp::Reverse((_, a)) = queue.pop().unwrap();
                        let std::cmp::Reverse((_, b)) = queue.pop().unwrap();
                        let combined = new.binary(if is_and { "and" } else { "or" }, a, b);
                        queue.push(std::cmp::Reverse((
                            bit_depth(combined, &new.bits, depths),
                            combined,
                        )));
                    }
                    queue.pop().unwrap().0 .1
                }
                Bit::Not(a) => {
                    let a = rebuild(a, old, fanout, new, memo, depths);
                    new.not(a)
                }
                Bit::Xor(a, b) => {
                    let a = rebuild(a, old, fanout, new, memo, depths);
                    let b = rebuild(b, old, fanout, new, memo, depths);
                    new.binary("xor", a, b)
                }
                other => new.bit(other),
            };
            memo.insert(id, mapped);
            mapped
        }
        let mut memo = HashMap::new();
        let mut depths = HashMap::new();
        let mapped = outputs
            .iter()
            .map(|(name, id)| {
                (
                    name.clone(),
                    rebuild(*id, self, &fanout, &mut rebuilt, &mut memo, &mut depths),
                )
            })
            .collect::<Vec<_>>();
        let gate_count = |bits: &[Bit], live: &HashSet<usize>| {
            live.iter()
                .filter(|&&id| !bit_inputs(&bits[id]).is_empty())
                .count()
        };
        let mut original_depths = HashMap::new();
        let no_deeper = outputs.iter().zip(&mapped).all(|((_, old), (_, new))| {
            bit_depth(*new, &rebuilt.bits, &mut depths)
                <= bit_depth(*old, &self.bits, &mut original_depths)
        });
        if no_deeper
            && gate_count(&rebuilt.bits, &live_bits(&rebuilt.bits, &mapped))
                <= gate_count(&self.bits, &live)
        {
            (rebuilt, mapped)
        } else {
            (self.clone(), outputs.to_vec())
        }
    }

    fn value(
        &mut self,
        value: &LogicalValue,
        nets: &HashMap<String, Vec<usize>>,
    ) -> Option<Vec<usize>> {
        match value {
            LogicalValue::Net { net } => nets.get(net).cloned(),
            LogicalValue::Constant { value, width } => Some(
                (0..*width)
                    .map(|i| self.constant(i < 128 && (value >> i) & 1 != 0))
                    .collect(),
            ),
        }
    }
    fn resize(&mut self, mut bits: Vec<usize>, width: usize) -> Vec<usize> {
        bits.resize(width, self.constant(false));
        bits.truncate(width);
        bits
    }
    fn cell(
        &mut self,
        cell: &LogicalCell,
        inputs: &[Vec<usize>],
        width: usize,
    ) -> eyre::Result<Vec<usize>> {
        let get = |pin: &str| {
            inputs[cell
                .inputs
                .iter()
                .position(|input| input.pin == pin)
                .unwrap()]
            .clone()
        };
        Ok(match &cell.kind {
            LogicalCellKind::Select { msb, lsb } => get("value")[*lsb..=*msb].to_vec(),
            LogicalCellKind::Concat => {
                let mut result = get("rhs");
                result.extend(get("lhs"));
                result
            }
            LogicalCellKind::Resize { .. } | LogicalCellKind::Buffer => {
                self.resize(get("value"), width)
            }
            LogicalCellKind::Not => {
                let value = self.resize(get("value"), width);
                value.into_iter().map(|a| self.not(a)).collect()
            }
            LogicalCellKind::And
            | LogicalCellKind::Or
            | LogicalCellKind::Xor
            | LogicalCellKind::Add
            | LogicalCellKind::Sub
            | LogicalCellKind::Inc => {
                let a = self.resize(
                    get(if cell.kind == LogicalCellKind::Inc {
                        "value"
                    } else {
                        "lhs"
                    }),
                    width,
                );
                let b = if cell.kind == LogicalCellKind::Inc {
                    let mut bits = vec![self.constant(false); width];
                    bits[0] = self.constant(true);
                    bits
                } else {
                    self.resize(get("rhs"), width)
                };
                let subtract = cell.kind == LogicalCellKind::Sub;
                let b = if subtract {
                    b.into_iter().map(|bit| self.not(bit)).collect()
                } else {
                    b
                };
                let mut carry = self.constant(subtract);
                a.into_iter()
                    .zip(b)
                    .map(|(a, b)| match cell.kind {
                        LogicalCellKind::And => self.binary("and", a, b),
                        LogicalCellKind::Or => self.binary("or", a, b),
                        LogicalCellKind::Xor => self.binary("xor", a, b),
                        _ => {
                            let p = self.binary("xor", a, b);
                            let sum = self.binary("xor", p, carry);
                            let generate = self.binary("and", a, b);
                            let propagate = self.binary("and", p, carry);
                            carry = self.binary("or", generate, propagate);
                            sum
                        }
                    })
                    .collect()
            }
            LogicalCellKind::GreaterEqual => {
                let mut greater_equal = self.constant(true);
                for (a, b) in get("lhs").into_iter().zip(get("rhs")) {
                    let not_b = self.not(b);
                    let greater = self.binary("and", a, not_b);
                    let different = self.binary("xor", a, b);
                    let equal = self.not(different);
                    let prior = self.binary("and", equal, greater_equal);
                    greater_equal = self.binary("or", greater, prior);
                }
                vec![greater_equal]
            }
            LogicalCellKind::Equal => {
                let mut same = self.constant(true);
                for (a, b) in get("lhs").into_iter().zip(get("rhs")) {
                    let different = self.binary("xor", a, b);
                    let equal = self.not(different);
                    same = self.binary("and", same, equal);
                }
                vec![same]
            }
            LogicalCellKind::Mux => {
                let select = get("select")[0];
                let yes = self.resize(get("when_true"), width);
                let no = self.resize(get("when_false"), width);
                let inverse = self.not(select);
                yes.into_iter()
                    .zip(no)
                    .map(|(yes, no)| {
                        if yes == no {
                            return yes;
                        }
                        let a = self.binary("and", select, yes);
                        let b = self.binary("and", inverse, no);
                        self.binary("or", a, b)
                    })
                    .collect()
            }
            _ => eyre::bail!(
                "generic bit mapping does not support sequential cell `{}`",
                cell.name
            ),
        })
    }
    fn module(
        &mut self,
        design: &LogicalDesign,
        module: &LogicalModule,
        mut nets: HashMap<String, Vec<usize>>,
    ) -> eyre::Result<HashMap<String, Vec<usize>>> {
        // Flatten hierarchy before scheduling cells. Requiring every child
        // input first invents dependencies between independent child outputs.
        fn flatten(
            design: &LogicalDesign,
            module: &LogicalModule,
            names: &HashMap<String, String>,
            used: &mut HashSet<String>,
            serial: &mut usize,
            cells: &mut Vec<(LogicalCell, usize)>,
        ) {
            for cell in &module.cells {
                let mut renamed = cell.clone();
                for input in &mut renamed.inputs {
                    if let LogicalValue::Net { net } = &mut input.value {
                        *net = names[net].clone();
                    }
                }
                for output in &mut renamed.outputs {
                    output.net = names[&output.net].clone();
                }
                let width = module
                    .nets
                    .iter()
                    .find(|net| net.name == cell.outputs[0].net)
                    .unwrap()
                    .width;
                cells.push((renamed, width));
            }
            for instance in &module.instances {
                let child = design.module(&instance.module).unwrap();
                let mut child_names = HashMap::new();
                for binding in &instance.bindings {
                    let port = child
                        .ports
                        .iter()
                        .find(|port| port.name == binding.port)
                        .unwrap();
                    child_names.insert(port.net.clone(), names[&binding.net].clone());
                }
                for net in &child.nets {
                    if !child_names.contains_key(&net.name) {
                        let name = loop {
                            let name = format!("__flat_{}", *serial);
                            *serial += 1;
                            if used.insert(name.clone()) {
                                break name;
                            }
                        };
                        child_names.insert(net.name.clone(), name);
                    }
                }
                flatten(design, child, &child_names, used, serial, cells);
            }
        }
        let names = module
            .nets
            .iter()
            .map(|net| (net.name.clone(), net.name.clone()))
            .collect::<HashMap<_, _>>();
        let mut used = names.values().cloned().collect::<HashSet<_>>();
        let mut cells = Vec::new();
        flatten(design, module, &names, &mut used, &mut 0, &mut cells);
        while !cells.is_empty() {
            let before = cells.len();
            let mut remaining = Vec::new();
            for (cell, width) in cells {
                let Some(inputs) = cell
                    .inputs
                    .iter()
                    .map(|input| self.value(&input.value, &nets))
                    .collect::<Option<Vec<_>>>()
                else {
                    remaining.push((cell, width));
                    continue;
                };
                let output = cell.output("result")?;
                let value = self.cell(&cell, &inputs, width)?;
                nets.insert(output.clone(), value);
            }
            cells = remaining;
            if cells.len() == before {
                eyre::bail!(
                    "combinational cycle or unavailable driver in `{}`",
                    module.name
                );
            }
        }
        Ok(nets)
    }
}

pub(super) fn lower(design: &LogicalDesign) -> eyre::Result<RoutableDesign> {
    lower_with_limits(design, 1, 2)
}

pub(super) fn lower_with_limits(
    design: &LogicalDesign,
    max_gates: usize,
    max_inputs: usize,
) -> eyre::Result<RoutableDesign> {
    lower_with_balancing(design, max_gates, max_inputs, false)
}

pub(super) fn lower_with_balancing(
    design: &LogicalDesign,
    max_gates: usize,
    max_inputs: usize,
    balance: bool,
) -> eyre::Result<RoutableDesign> {
    lower_options(design, max_gates, max_inputs, balance, None, false, false)
}

pub(super) fn lower_with_associative_cells(
    design: &LogicalDesign,
    max_arity: usize,
) -> eyre::Result<RoutableDesign> {
    eyre::ensure!(
        (2..=8).contains(&max_arity),
        "associative cells support arity 2 through 8"
    );
    lower_options(design, 1, 2, false, Some(max_arity), false, false)
}

pub(super) fn lower_with_expanded_associative_cells(
    design: &LogicalDesign,
    max_arity: usize,
) -> eyre::Result<RoutableDesign> {
    let baseline = lower_with_associative_cells(design, max_arity)?;
    let expanded = lower_options(design, 1, 2, false, Some(max_arity), true, false)?;
    let count = |design: &RoutableDesign| match &design.module(&design.top).unwrap().body {
        RoutableModuleBody::Composite { instances, .. } => instances.len(),
        _ => 0,
    };
    // This is a cell-count tradeoff, not a raw Boolean-gate optimization.
    // Shared logic may be duplicated inside recipes only if fewer physical
    // instances are required. The ordinary shared mapping remains the fallback.
    Ok(if count(&expanded) < count(&baseline) {
        expanded
    } else {
        baseline
    })
}

pub(super) fn lower_with_literal_associative_cells(
    design: &LogicalDesign,
    max_arity: usize,
) -> eyre::Result<RoutableDesign> {
    let baseline = lower_with_expanded_associative_cells(design, max_arity)?;
    let literal = lower_options(design, 1, 2, false, Some(max_arity), true, true)?;
    let count = |design: &RoutableDesign| match &design.module(&design.top).unwrap().body {
        RoutableModuleBody::Composite { instances, .. } => instances.len(),
        _ => 0,
    };
    Ok(if count(&literal) < count(&baseline) {
        literal
    } else {
        baseline
    })
}

fn build_flat_bits(
    design: &LogicalDesign,
) -> eyre::Result<(Builder, Vec<(String, usize)>, Vec<RoutablePort>)> {
    let top = design.module(&design.top).unwrap();
    // Preserve ordinary bus_0 naming, but reserve all declared port names
    // first. Escaped generated names cannot alias a scalar input or output.
    let mut used_names = top
        .ports
        .iter()
        .map(|port| port.name.clone())
        .collect::<HashSet<_>>();
    let mut port_names = HashMap::<String, Vec<String>>::new();
    for port in &top.ports {
        let width = top
            .nets
            .iter()
            .find(|net| net.name == port.net)
            .unwrap()
            .width;
        let names = if width == 1 {
            vec![port.name.clone()]
        } else {
            (0..width)
                .map(|bit| {
                    let ordinary = bit_signal_name(&port.name, bit);
                    if used_names.insert(ordinary.clone()) {
                        return ordinary;
                    }
                    let mut escaped = format!("{}[{bit}]", port.name);
                    while !used_names.insert(escaped.clone()) {
                        escaped.insert(0, '$');
                    }
                    escaped
                })
                .collect()
        };
        port_names.insert(port.name.clone(), names);
    }
    let mut builder = Builder::default();
    let mut inputs = HashMap::new();
    for port in top
        .ports
        .iter()
        .filter(|p| p.direction == LogicalPortDirection::Input)
    {
        let width = top.nets.iter().find(|n| n.name == port.net).unwrap().width;
        let bits = (0..width)
            .map(|i| builder.bit(Bit::Input(port_names[&port.name][i].clone())))
            .collect();
        inputs.insert(port.net.clone(), bits);
    }
    let values = builder.module(design, top, inputs)?;
    let ports: Vec<RoutablePort> = top
        .ports
        .iter()
        .flat_map(|port| {
            port_names[&port.name].iter().map(|name| RoutablePort {
                name: name.clone(),
                direction: match port.direction {
                    LogicalPortDirection::Input => RoutablePortDirection::Input,
                    LogicalPortDirection::Output => RoutablePortDirection::Output,
                },
            })
        })
        .collect();
    let mut outputs = Vec::new();
    for port in top
        .ports
        .iter()
        .filter(|p| p.direction == LogicalPortDirection::Output)
    {
        let bits = &values[&port.net];
        for (i, &bit) in bits.iter().enumerate() {
            outputs.push((port_names[&port.name][i].clone(), bit));
        }
    }
    Ok((builder, outputs, ports))
}

pub(super) fn lower_flat_logic_graph(design: &LogicalDesign) -> eyre::Result<LogicGraph> {
    design.validate()?;
    eyre::ensure!(
        !design
            .modules
            .iter()
            .any(|m| m.cells.iter().any(|c| c.kind.is_sequential())),
        "flat logic graph only supports combinational designs"
    );
    let (builder, outputs, _) = build_flat_bits(design)?;
    let live = live_bits(&builder.bits, &outputs);
    let mut nodes = Vec::new();
    for (id, bit) in builder.bits.iter().enumerate() {
        if !live.contains(&id) && !matches!(bit, Bit::Input(_)) {
            continue;
        }
        let kind = match bit {
            Bit::Input(name) => GraphNodeKind::Input(name.clone()),
            Bit::Not(_) => GraphNodeKind::Logic(Logic {
                logic_type: LogicType::Not,
            }),
            Bit::And(..) => GraphNodeKind::Logic(Logic {
                logic_type: LogicType::And,
            }),
            Bit::Or(..) => GraphNodeKind::Logic(Logic {
                logic_type: LogicType::Or,
            }),
            Bit::Xor(..) => GraphNodeKind::Logic(Logic {
                logic_type: LogicType::Xor,
            }),
            Bit::Constant(_) => {
                eyre::bail!("flat logic graph cannot represent a live constant output")
            }
        };
        nodes.push((
            id,
            GraphNode {
                kind,
                inputs: bit_inputs(bit),
                ..Default::default()
            },
        ));
    }
    for (index, (name, input)) in outputs.into_iter().enumerate() {
        nodes.push((
            builder.bits.len() + index,
            GraphNode {
                kind: GraphNodeKind::Output(name),
                inputs: vec![input],
                ..Default::default()
            },
        ));
    }
    let mut graph = Graph::from_nodes_with_ids(nodes);
    graph.build_outputs();
    graph.build_producers();
    graph.build_consumers();
    graph.verify()?;
    let mut transform = crate::transform::logic::LogicGraphTransformer::new(LogicGraph { graph });
    transform.decompose_xor()?;
    transform.decompose_and()?;
    transform.remove_double_neg_expression();
    transform.optimize_cse()?;
    Ok(transform.finish())
}

fn lower_options(
    design: &LogicalDesign,
    max_gates: usize,
    max_inputs: usize,
    balance: bool,
    associative_arity: Option<usize>,
    expand_shared: bool,
    absorb_literals: bool,
) -> eyre::Result<RoutableDesign> {
    eyre::ensure!(
        max_gates >= 1 && max_inputs >= 2,
        "bit mapping needs at least one gate and two input ports"
    );
    let (mut builder, mut outputs, ports) = build_flat_bits(design)?;
    let top = design.module(&design.top).unwrap();
    if balance {
        (builder, outputs) = builder.balanced(&outputs);
    }
    let mut live = HashSet::new();
    fn visit(id: usize, bits: &[Bit], live: &mut HashSet<usize>) {
        if !live.insert(id) {
            return;
        }
        match bits[id] {
            Bit::Not(a) => visit(a, bits, live),
            Bit::And(a, b) | Bit::Or(a, b) | Bit::Xor(a, b) => {
                visit(a, bits, live);
                visit(b, bits, live);
            }
            _ => {}
        }
    }
    for (_, bit) in &outputs {
        visit(*bit, &builder.bits, &mut live);
    }
    fn dependencies(bit: &Bit) -> Vec<usize> {
        match *bit {
            Bit::Not(a) => vec![a],
            Bit::And(a, b) | Bit::Or(a, b) | Bit::Xor(a, b) => vec![a, b],
            _ => vec![],
        }
    }
    let mut fanout = vec![0; builder.bits.len()];
    for &id in &live {
        for input in dependencies(&builder.bits[id]) {
            fanout[input] += 1;
        }
    }
    for (_, id) in &outputs {
        fanout[*id] += 1;
    }
    let mut absorbed = HashSet::new();
    let mut groups = HashMap::new();
    // Fanout-free cones only: a shared decode signal remains one routed driver.
    // Every merge checks both gate and unique boundary-input budgets.
    for root in (0..builder.bits.len()).rev() {
        if !live.contains(&root)
            || (!expand_shared && absorbed.contains(&root))
            || matches!(builder.bits[root], Bit::Input(_) | Bit::Constant(_))
        {
            continue;
        }
        let mut included = HashSet::from([root]);
        let gate_limit = associative_arity.map_or(max_gates, |arity| {
            if matches!(builder.bits[root], Bit::And(..) | Bit::Or(..)) {
                arity - 1
            } else {
                1
            }
        });
        let input_limit = associative_arity.unwrap_or(max_inputs);
        let boundary = |included: &HashSet<usize>| {
            included
                .iter()
                .flat_map(|id| dependencies(&builder.bits[*id]))
                .filter(|id| !included.contains(id))
                .collect::<std::collections::BTreeSet<_>>()
        };
        loop {
            let mut merged = false;
            for candidate in boundary(&included).into_iter().rev() {
                if (!expand_shared && fanout[candidate] != 1)
                    || matches!(builder.bits[candidate], Bit::Input(_) | Bit::Constant(_))
                {
                    continue;
                }
                // A NOT's operand is an opaque literal terminal. Never
                // absorb an AND/OR hidden underneath that inversion.
                if absorb_literals
                    && included.iter().any(
                        |id| matches!(builder.bits[*id],Bit::Not(operand) if operand==candidate),
                    )
                {
                    continue;
                }
                let same_operation = matches!(
                    (&builder.bits[root], &builder.bits[candidate]),
                    (Bit::And(..), Bit::And(..)) | (Bit::Or(..), Bit::Or(..))
                );
                let literal_not = absorb_literals
                    && matches!(builder.bits[root], Bit::And(..) | Bit::Or(..))
                    && matches!(builder.bits[candidate], Bit::Not(..));
                if associative_arity.is_some() && !same_operation && !literal_not {
                    continue;
                }
                let mut proposed = included.clone();
                proposed.insert(candidate);
                let operations = proposed
                    .iter()
                    .filter(|id| {
                        !absorb_literals
                            || !matches!(builder.bits[**id], Bit::Not(..))
                            || **id == root
                    })
                    .count();
                if operations > gate_limit {
                    continue;
                }
                // Contradictory literals would become a constant function,
                // not the single-exception cube/clause the recipe promises.
                fn collect_literal_polarities(
                    id: usize,
                    bits: &[Bit],
                    included: &HashSet<usize>,
                    negative: bool,
                    polarities: &mut HashMap<usize, bool>,
                ) -> bool {
                    if !included.contains(&id) {
                        return polarities
                            .insert(id, negative)
                            .is_none_or(|prior| prior == negative);
                    }
                    match bits[id] {
                        Bit::And(a, b) | Bit::Or(a, b) => {
                            collect_literal_polarities(a, bits, included, negative, polarities)
                                && collect_literal_polarities(
                                    b, bits, included, negative, polarities,
                                )
                        }
                        Bit::Not(a) => {
                            collect_literal_polarities(a, bits, included, !negative, polarities)
                        }
                        _ => false,
                    }
                }
                if absorb_literals
                    && matches!(builder.bits[root], Bit::And(..) | Bit::Or(..))
                    && !collect_literal_polarities(
                        root,
                        &builder.bits,
                        &proposed,
                        false,
                        &mut HashMap::new(),
                    )
                {
                    continue;
                }
                if boundary(&proposed).len() <= input_limit {
                    included = proposed;
                    if !expand_shared {
                        absorbed.insert(candidate);
                    }
                    merged = true;
                    break;
                }
            }
            if !merged {
                break;
            }
        }
        groups.insert(root, included);
    }
    if expand_shared {
        // Select complete cells from the outputs backwards. A shared prefix
        // expanded into every consumer must not remain as an unused instance.
        let mut needed = HashSet::new();
        let mut pending = outputs.iter().map(|(_, id)| *id).collect::<Vec<_>>();
        while let Some(root) = pending.pop() {
            if !needed.insert(root) {
                continue;
            }
            if let Some(included) = groups.get(&root) {
                pending.extend(
                    included
                        .iter()
                        .flat_map(|id| dependencies(&builder.bits[*id]))
                        .filter(|id| !included.contains(id)),
                );
            }
        }
        live = needed;
    }
    let mut modules = Vec::new();
    let mut instances = Vec::new();
    let mut connections = NetConnections::default();
    let mut endpoints = HashMap::new();
    for (id, bit) in builder.bits.iter().enumerate() {
        if !live.contains(&id) || absorbed.contains(&id) {
            continue;
        }
        if let Bit::Input(name) = bit {
            endpoints.insert(id, self_port(name));
            continue;
        }
        let name = format!("{}_bit_{id}", top.name);
        let (expr, args): (String, Vec<(String, usize)>) = if let Bit::Constant(value) = bit {
            let seed = builder
                .bits
                .iter()
                .position(|bit| matches!(bit, Bit::Input(_)))
                .ok_or_else(|| {
                    eyre::eyre!("input-free constant design requires a constant-source target cell")
                })?;
            if let Bit::Input(input) = &builder.bits[seed] {
                endpoints.insert(seed, self_port(input));
            }
            (
                (if *value { "a|~a" } else { "a&~a" }).to_owned(),
                vec![("a".to_owned(), seed)],
            )
        } else {
            let included = &groups[&id];
            let boundary = included
                .iter()
                .flat_map(|id| dependencies(&builder.bits[*id]))
                .filter(|id| !included.contains(id))
                .collect::<std::collections::BTreeSet<_>>();
            let args = boundary
                .into_iter()
                .enumerate()
                .map(|(index, id)| (format!("a{index}"), id))
                .collect::<Vec<_>>();
            fn expression(
                id: usize,
                bits: &[Bit],
                included: &HashSet<usize>,
                args: &[(String, usize)],
            ) -> String {
                if !included.contains(&id) {
                    return args
                        .iter()
                        .find(|(_, input)| *input == id)
                        .unwrap()
                        .0
                        .clone();
                }
                let expr = |id| expression(id, bits, included, args);
                match bits[id] {
                    Bit::Not(a) => format!("~({})", expr(a)),
                    Bit::And(a, b) => format!("({}&{})", expr(a), expr(b)),
                    Bit::Or(a, b) => format!("({}|{})", expr(a), expr(b)),
                    Bit::Xor(a, b) => format!("({}^{})", expr(a), expr(b)),
                    _ => unreachable!(),
                }
            }
            (expression(id, &builder.bits, included, &args), args)
        };
        modules.push(combinational_output_module(&name, &expr, "y")?);
        for (pin, source) in args {
            connections.connect(
                &format!("bit_{source}"),
                NetClass::Data,
                endpoints[&source].clone(),
                instance_port(&name, &pin),
            );
        }
        endpoints.insert(id, instance_port(&name, "y"));
        instances.push(name);
    }
    // Materialize pass-through outputs as a buffer leaf: the physical output
    // collector observes instance outputs, not top-input aliases.
    for (index, (name, id)) in outputs.into_iter().enumerate() {
        let mut driver = endpoints[&id].clone();
        if matches!(driver, Endpoint::SelfPort { .. }) {
            let instance = format!("{}_output_{index}", top.name);
            modules.push(RoutableModule {
                name: instance.clone(),
                ports: vec![
                    RoutablePort {
                        name: "a".into(),
                        direction: RoutablePortDirection::Input,
                    },
                    RoutablePort {
                        name: "y".into(),
                        direction: RoutablePortDirection::Output,
                    },
                ],
                body: RoutableModuleBody::Leaf {
                    nodes: vec![
                        RoutableNode {
                            id: 0,
                            kind: RoutableNodeKind::Input { name: "a".into() },
                            inputs: vec![],
                            tag: String::new(),
                        },
                        RoutableNode {
                            id: 1,
                            kind: RoutableNodeKind::Not,
                            inputs: vec![0],
                            tag: String::new(),
                        },
                        RoutableNode {
                            id: 2,
                            kind: RoutableNodeKind::Not,
                            inputs: vec![1],
                            tag: String::new(),
                        },
                        RoutableNode {
                            id: 3,
                            kind: RoutableNodeKind::Output { name: "y".into() },
                            inputs: vec![2],
                            tag: String::new(),
                        },
                    ],
                },
            });
            connections.connect(
                &format!("bit_{id}"),
                NetClass::Io,
                driver,
                instance_port(&instance, "a"),
            );
            driver = instance_port(&instance, "y");
            instances.push(instance);
        }
        connections.connect(&name, NetClass::Io, driver, self_port(&name));
    }
    modules.push(composite_module(
        &top.name,
        ports,
        instances,
        connections.finish(),
    ));
    finish_design(&top.name, modules)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_decoder_flat_graph_preserves_all_seven_outputs() -> eyre::Result<()> {
        let source = format!(
            "{}\n{}",
            include_str!("../../../test/alu_display/bcd_to_7seg_minimized.v"),
            include_str!("../../../test/alu_display/bcd_digit_to_7seg.v")
        );
        let design = LogicalDesign::from_verilog_source(&source)?;
        let graph = design.lower_flat_logic_graph()?;
        let truth = graph.truth_table()?;
        assert_eq!(
            truth.input_names,
            vec!["digit_0", "digit_1", "digit_2", "digit_3"]
        );
        assert_eq!(truth.output_tables.len(), 7);
        let masks = [
            0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f, 0, 0, 0, 0, 0, 0,
        ];
        for bit in 0..7 {
            for value in 0..16 {
                assert_eq!(
                    truth.output_tables[&format!("seg_{bit}")][value],
                    masks[value] & (1 << bit) != 0
                );
            }
        }
        assert!(graph.nodes.iter().all(|node| matches!(&node.kind, GraphNodeKind::Input(_) | GraphNodeKind::Output(_)) || matches!(&node.kind, GraphNodeKind::Logic(logic) if matches!(logic.logic_type, LogicType::Not | LogicType::Or))));
        Ok(())
    }

    fn simulate(design: &RoutableDesign, input: &HashMap<String, bool>) -> HashMap<String, bool> {
        let top = design.module(&design.top).unwrap();
        let RoutableModuleBody::Composite { instances, nets } = &top.body else {
            panic!("expected bit-mapped composite")
        };
        let mut values = input
            .iter()
            .map(|(name, &value)| (self_port(name), value))
            .collect::<HashMap<_, _>>();
        let mut remaining = instances.iter().collect::<Vec<_>>();
        while !remaining.is_empty() {
            for net in nets {
                if let Some(value) = values.get(&net.driver).copied() {
                    for sink in &net.sinks {
                        values.insert(sink.clone(), value);
                    }
                }
            }
            let before = remaining.len();
            remaining.retain(|instance| {
                let leaf = design.module(&instance.module).unwrap();
                let RoutableModuleBody::Leaf { nodes } = &leaf.body else {
                    panic!("nested physical module")
                };
                if leaf
                    .ports
                    .iter()
                    .filter(|p| p.direction == RoutablePortDirection::Input)
                    .any(|p| !values.contains_key(&instance_port(&instance.name, &p.name)))
                {
                    return true;
                }
                fn evaluate(
                    id: usize,
                    nodes: &[RoutableNode],
                    instance: &str,
                    values: &HashMap<Endpoint, bool>,
                ) -> bool {
                    let node = nodes.iter().find(|node| node.id == id).unwrap();
                    match &node.kind {
                        RoutableNodeKind::Input { name } => values[&instance_port(instance, name)],
                        RoutableNodeKind::Output { .. } => {
                            evaluate(node.inputs[0], nodes, instance, values)
                        }
                        RoutableNodeKind::Not => !evaluate(node.inputs[0], nodes, instance, values),
                        RoutableNodeKind::Or => node
                            .inputs
                            .iter()
                            .any(|id| evaluate(*id, nodes, instance, values)),
                        _ => panic!("unexpected sequential node"),
                    }
                }
                for node in nodes {
                    if let RoutableNodeKind::Output { name } = &node.kind {
                        let value = evaluate(node.id, nodes, &instance.name, &values);
                        values.insert(instance_port(&instance.name, name), value);
                    }
                }
                false
            });
            assert!(remaining.len() < before, "stuck physical evaluator");
        }
        for net in nets {
            if let Some(value) = values.get(&net.driver).copied() {
                for sink in &net.sinks {
                    values.insert(sink.clone(), value);
                }
            }
        }
        top.ports
            .iter()
            .filter(|p| p.direction == RoutablePortDirection::Output)
            .map(|p| (p.name.clone(), values[&self_port(&p.name)]))
            .collect()
    }

    fn bits(input: &mut HashMap<String, bool>, name: &str, width: usize, value: usize) {
        for i in 0..width {
            input.insert(
                if width == 1 {
                    name.to_owned()
                } else {
                    format!("{name}_{i}")
                },
                value & (1 << i) != 0,
            );
        }
    }
    fn number(output: &HashMap<String, bool>, name: &str, width: usize) -> usize {
        (0..width)
            .map(|i| {
                usize::from(
                    output[&if width == 1 {
                        name.to_owned()
                    } else {
                        format!("{name}_{i}")
                    }],
                ) << i
            })
            .sum()
    }
    fn compile(source: &str) -> eyre::Result<RoutableDesign> {
        let logical = LogicalDesign::from_verilog_source(source)?;
        // Every new operation must survive the editable RCIR text boundary.
        let reparsed: LogicalDesign = logical.to_string().parse()?;
        assert_eq!(logical.to_string(), reparsed.to_string());
        reparsed.lower_to_routable()
    }

    #[test]
    fn decoder_vectors_survive_typed_and_bit_mapping() -> eyre::Result<()> {
        let source = include_str!("../../../test/alu_display/bcd_to_7seg.v");
        let logical = LogicalDesign::from_verilog_source(source)?;
        let design = logical.lower_to_routable_with_bit_mapping_limits(4, 3)?;
        let baseline = lower_with_limits(&LogicalDesign::from_verilog_source(source)?, 1, 2)?;
        let count = |design: &RoutableDesign| match &design.module(&design.top).unwrap().body {
            RoutableModuleBody::Composite { instances, .. } => instances.len(),
            _ => 0,
        };
        assert!(count(&design) < count(&baseline));
        eprintln!(
            "decoder instances: grouped={}, primitive={}",
            count(&design),
            count(&baseline)
        );
        for module in &design.modules {
            if let RoutableModuleBody::Leaf { .. } = &module.body {
                assert!(
                    module
                        .ports
                        .iter()
                        .filter(|p| p.direction == RoutablePortDirection::Input)
                        .count()
                        <= 3
                );
            }
        }
        let masks = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f];
        for digit in 0..16 {
            for blank in 0..2 {
                let mut input = HashMap::new();
                bits(&mut input, "digit", 4, digit);
                bits(&mut input, "blank", 1, blank);
                let output = simulate(&design, &input);
                assert_eq!(output, simulate(&baseline, &input));
                assert_eq!(
                    number(&output, "seg", 7),
                    if blank == 0 && digit < 10 {
                        masks[digit]
                    } else {
                        0
                    },
                    "digit={digit},blank={blank}"
                );
            }
        }
        Ok(())
    }

    fn primitive_stats(design: &RoutableDesign) -> (usize, usize) {
        let RoutableModuleBody::Composite { instances, nets } =
            &design.module(&design.top).unwrap().body
        else {
            panic!()
        };
        fn depth(instance: &str, nets: &[RoutableNet], memo: &mut HashMap<String, usize>) -> usize {
            if let Some(&depth) = memo.get(instance) {
                return depth;
            }
            let parents=nets.iter().filter(|net| net.sinks.iter().any(|sink|matches!(sink,Endpoint::InstancePort{instance:target,..} if target==instance))).map(|net|match &net.driver {Endpoint::SelfPort{..}=>0,Endpoint::InstancePort{instance,..}=>depth(instance,nets,memo)}).max().unwrap_or(0);
            memo.insert(instance.to_owned(), parents + 1);
            parents + 1
        }
        let mut memo = HashMap::new();
        (
            instances.len(),
            instances
                .iter()
                .map(|instance| depth(&instance.name, nets, &mut memo))
                .max()
                .unwrap_or(0),
        )
    }

    #[test]
    fn associative_balancing_preserves_decoder_vectors_and_sharing() -> eyre::Result<()> {
        for (name, source) in [
            (
                "ternary",
                include_str!("../../../test/alu_display/bcd_to_7seg.v"),
            ),
            (
                "sop",
                include_str!("../../../test/alu_display/bcd_to_7seg_sop.v"),
            ),
        ] {
            let logical = LogicalDesign::from_verilog_source(source)?;
            let baseline = logical.lower_to_routable()?;
            let balanced = logical.lower_to_routable_with_balanced_associative_trees(1, 2)?;
            let before = primitive_stats(&baseline);
            let after = primitive_stats(&balanced);
            eprintln!(
                "balanced {name}: instances {} -> {}, primitive stages {} -> {}",
                before.0, after.0, before.1, after.1
            );
            assert!(after.0 <= before.0);
            assert!(after.1 <= before.1);
            for mask in 0..32 {
                let mut input = HashMap::new();
                bits(&mut input, "digit", 4, mask & 15);
                bits(&mut input, "blank", 1, mask >> 4);
                assert_eq!(
                    simulate(&baseline, &input),
                    simulate(&balanced, &input),
                    "{name}, input={mask}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn associative_balancing_shortens_reductions_without_duplicating_shared_logic(
    ) -> eyre::Result<()> {
        for (name, source) in [
            (
                "reductions",
                "module top(a,all,zero); input [7:0] a; output all,zero; assign all=a[0]&a[1]&a[2]&a[3]&a[4]&a[5]&a[6]&a[7]; assign zero=~(a[0]|a[1]|a[2]|a[3]|a[4]|a[5]|a[6]|a[7]); endmodule",
            ),
            (
                "shared",
                "module top(a,y,z); input [7:0] a; output y,z; wire p; assign p=a[0]|a[1]|a[2]|a[3]; assign y=p|a[4]|a[5]|a[6]|a[7]; assign z=p^a[7]; endmodule",
            ),
        ] {
            let logical = LogicalDesign::from_verilog_source(source)?;
            let baseline = logical.lower_to_routable()?;
            let balanced = logical.lower_to_routable_with_balanced_associative_trees(1, 2)?;
            let before = primitive_stats(&baseline);
            let after = primitive_stats(&balanced);
            eprintln!(
                "balanced {name}: instances {} -> {}, primitive stages {} -> {}",
                before.0, after.0, before.1, after.1
            );
            assert!(after.0 <= before.0);
            assert!(after.1 < before.1);
            if name == "reductions" {
                assert_eq!(before, (15, 8));
                assert_eq!(after, (15, 4));
            }
            for mask in 0..256 {
                let mut input = HashMap::new();
                bits(&mut input, "a", 8, mask);
                let actual = simulate(&balanced, &input);
                assert_eq!(actual, simulate(&baseline, &input));
                if name == "reductions" {
                    assert_eq!(actual["all"], mask == 255);
                    assert_eq!(actual["zero"], mask == 0);
                } else {
                    assert_eq!(actual["y"], mask != 0);
                    assert_eq!(actual["z"], (mask & 15 != 0) ^ (mask & 128 != 0));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn associative_cells_are_pure_bounded_and_or_and_preserve_decoder() -> eyre::Result<()> {
        for (name, source) in [
            (
                "ternary",
                include_str!("../../../test/alu_display/bcd_to_7seg.v"),
            ),
            (
                "sop",
                include_str!("../../../test/alu_display/bcd_to_7seg_sop.v"),
            ),
        ] {
            let logical = LogicalDesign::from_verilog_source(source)?;
            let primitive = logical.lower_to_routable()?;
            let mapped = logical.lower_to_routable_with_associative_cells(8)?;
            eprintln!(
                "associative cells {name}: {:?} -> {:?} (instances, stages)",
                primitive_stats(&primitive),
                primitive_stats(&mapped)
            );
            for leaf in mapped
                .modules
                .iter()
                .filter(|module| matches!(module.body, RoutableModuleBody::Leaf { .. }))
            {
                let table = LogicGraph {
                    graph: crate::ir::graph_from_routable_leaf(leaf)?,
                }
                .truth_table()?;
                let n = table.input_names.len();
                assert!((1..=8).contains(&n));
                assert_eq!(table.output_tables.len(), 1);
                let actual = table.output_tables.values().next().unwrap();
                let and = (0..1usize << n)
                    .map(|mask| mask == (1 << n) - 1)
                    .collect::<Vec<_>>();
                let or = (0..1usize << n).map(|mask| mask != 0).collect::<Vec<_>>();
                let not = n == 1 && actual == &vec![true, false];
                let xor = n == 2 && actual == &vec![false, true, true, false];
                assert!(
                    actual == &and || actual == &or || not || xor,
                    "mixed-function leaf {}",
                    leaf.name
                );
            }
            for mask in 0..32 {
                let mut input = HashMap::new();
                bits(&mut input, "digit", 4, mask & 15);
                bits(&mut input, "blank", 1, mask >> 4);
                assert_eq!(
                    simulate(&mapped, &input),
                    simulate(&primitive, &input),
                    "{name} input={mask}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn associative_cells_group_eight_operands_but_not_not_or_xor() -> eyre::Result<()> {
        let logical = LogicalDesign::from_verilog_source(
            "module top(a,any,all,inverted,parity); input [7:0] a; output any,all,inverted,parity; assign any=a[0]|a[1]|a[2]|a[3]|a[4]|a[5]|a[6]|a[7]; assign all=a[0]&a[1]&a[2]&a[3]&a[4]&a[5]&a[6]&a[7]; assign inverted=~a[0]; assign parity=a[0]^a[1]; endmodule",
        )?;
        let mapped = logical.lower_to_routable_with_associative_cells(8)?;
        assert_eq!(primitive_stats(&mapped).0, 4);
        assert!(logical.lower_to_routable_with_associative_cells(9).is_err());
        for mask in 0..256 {
            let mut input = HashMap::new();
            bits(&mut input, "a", 8, mask);
            let actual = simulate(&mapped, &input);
            assert_eq!(actual["any"], mask != 0);
            assert_eq!(actual["all"], mask == 255);
            assert_eq!(actual["inverted"], mask & 1 == 0);
            assert_eq!(actual["parity"], ((mask & 1) ^ ((mask >> 1) & 1)) != 0);
        }
        Ok(())
    }

    #[test]
    fn expanded_associative_cells_reduce_instances_with_bounded_pure_leaves() -> eyre::Result<()> {
        for (name, source) in [
            (
                "ternary",
                include_str!("../../../test/alu_display/bcd_to_7seg.v"),
            ),
            (
                "sop",
                include_str!("../../../test/alu_display/bcd_to_7seg_sop.v"),
            ),
        ] {
            let logical = LogicalDesign::from_verilog_source(source)?;
            let shared = logical.lower_to_routable_with_associative_cells(8)?;
            let expanded = logical.lower_to_routable_with_expanded_associative_cells(8)?;
            let primitive = logical.lower_to_routable()?;
            assert!(primitive_stats(&expanded).0 <= primitive_stats(&shared).0);
            let raw_gates = |design: &RoutableDesign| {
                let RoutableModuleBody::Composite { instances, .. } =
                    &design.module(&design.top).unwrap().body
                else {
                    panic!()
                };
                instances
                    .iter()
                    .map(
                        |instance| match &design.module(&instance.module).unwrap().body {
                            RoutableModuleBody::Leaf { nodes } => nodes
                                .iter()
                                .filter(|node| {
                                    matches!(
                                        node.kind,
                                        RoutableNodeKind::Not | RoutableNodeKind::Or
                                    )
                                })
                                .count(),
                            _ => 0,
                        },
                    )
                    .sum::<usize>()
            };
            eprintln!(
                "expanded {name}: shared {:?}, expanded {:?}, binary NOR nodes {} -> {}",
                primitive_stats(&shared),
                primitive_stats(&expanded),
                raw_gates(&shared),
                raw_gates(&expanded)
            );
            for leaf in expanded
                .modules
                .iter()
                .filter(|module| matches!(module.body, RoutableModuleBody::Leaf { .. }))
            {
                let table = LogicGraph {
                    graph: crate::ir::graph_from_routable_leaf(leaf)?,
                }
                .truth_table()?;
                let n = table.input_names.len();
                assert!((1..=8).contains(&n));
                assert_eq!(table.output_tables.len(), 1);
                let actual = table.output_tables.values().next().unwrap();
                let and = (0..1usize << n)
                    .map(|mask| mask == (1 << n) - 1)
                    .collect::<Vec<_>>();
                let or = (0..1usize << n).map(|mask| mask != 0).collect::<Vec<_>>();
                assert!(
                    actual == &and
                        || actual == &or
                        || (n == 1 && actual == &vec![true, false])
                        || (n == 2 && actual == &vec![false, true, true, false]),
                    "unexpected mixed leaf {}",
                    leaf.name
                );
            }
            for mask in 0..32 {
                let mut input = HashMap::new();
                bits(&mut input, "digit", 4, mask & 15);
                bits(&mut input, "blank", 1, mask >> 4);
                assert_eq!(
                    simulate(&expanded, &input),
                    simulate(&primitive, &input),
                    "{name} input={mask}"
                );
            }
        }
        let logical=LogicalDesign::from_verilog_source("module top(a,y,z); input [7:0] a; output y,z; wire p; assign p=a[0]|a[1]|a[2]|a[3]; assign y=p|a[4]|a[5]|a[6]|a[7]; assign z=p^a[7]; endmodule")?;
        let shared = logical.lower_to_routable_with_associative_cells(8)?;
        let expanded = logical.lower_to_routable_with_expanded_associative_cells(8)?;
        assert_eq!(
            shared.to_string(),
            expanded.to_string(),
            "equal cell count must keep shared implementation"
        );
        Ok(())
    }

    #[test]
    fn literal_associative_cells_preserve_decoder_and_single_exception_signatures(
    ) -> eyre::Result<()> {
        let sop = include_str!("../../../test/alu_display/bcd_to_7seg_sop.v");
        let digit = format!(
            "{}\n{}",
            sop,
            include_str!("../../../test/alu_display/bcd_digit_to_7seg.v")
        );
        for (name, source) in [("full", sop), ("digit", digit.as_str())] {
            let logical = LogicalDesign::from_verilog_source(source)?;
            let baseline = logical.lower_to_routable_with_expanded_associative_cells(8)?;
            let literal = logical.lower_to_routable_with_literal_associative_cells(8)?;
            eprintln!(
                "literal {name}: {:?} -> {:?} (instances, stages)",
                primitive_stats(&baseline),
                primitive_stats(&literal)
            );
            assert!(primitive_stats(&literal).0 < primitive_stats(&baseline).0);
            for leaf in literal
                .modules
                .iter()
                .filter(|module| matches!(module.body, RoutableModuleBody::Leaf { .. }))
            {
                let table = LogicGraph {
                    graph: crate::ir::graph_from_routable_leaf(leaf)?,
                }
                .truth_table()?;
                let n = table.input_names.len();
                assert!((1..=8).contains(&n));
                assert_eq!(table.output_tables.len(), 1);
                let actual = table.output_tables.values().next().unwrap();
                let true_rows = actual.iter().filter(|&&value| value).count();
                assert!(
                    true_rows == 1 || true_rows == actual.len() - 1,
                    "not a literal cube/clause: {}",
                    leaf.name
                );
            }
            let masks = [
                0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f, 0, 0, 0, 0, 0, 0,
            ];
            for mask in 0..if name == "full" { 32 } else { 16 } {
                let mut input = HashMap::new();
                bits(&mut input, "digit", 4, mask & 15);
                if name == "full" {
                    bits(&mut input, "blank", 1, mask >> 4);
                }
                let actual = simulate(&literal, &input);
                assert_eq!(actual, simulate(&baseline, &input));
                assert_eq!(
                    number(&actual, "seg", 7),
                    if mask & 16 == 0 { masks[mask & 15] } else { 0 }
                );
            }
        }
        Ok(())
    }

    #[test]
    fn literal_associative_cells_keep_not_operands_opaque_and_handle_opposite_literals(
    ) -> eyre::Result<()> {
        let source="module top(a,y,z,c); input [3:0] a; output y,z,c; wire p; assign p=a[0]&a[1]; assign y=(~p)&a[2]&a[3]; assign z=p|a[3]; assign c=(a[0]&a[1])&~a[0]; endmodule";
        let logical = LogicalDesign::from_verilog_source(source)?;
        let literal = logical.lower_to_routable_with_literal_associative_cells(8)?;
        for mask in 0..16 {
            let mut input = HashMap::new();
            bits(&mut input, "a", 4, mask);
            let actual = simulate(&literal, &input);
            let p = mask & 3 == 3;
            assert_eq!(actual["y"], !p && mask & 12 == 12);
            assert_eq!(actual["z"], p || mask & 8 != 0);
            assert!(!actual["c"]);
        }
        Ok(())
    }

    #[test]
    fn minimized_decoder_covers_preserve_all_values_and_reduce_routed_edges() -> eyre::Result<()> {
        let masks = [
            0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f, 0, 0, 0, 0, 0, 0,
        ];
        let old = include_str!("../../../test/alu_display/bcd_to_7seg_sop.v");
        let minimized = include_str!("../../../test/alu_display/bcd_to_7seg_minimized.v");
        let wrapper = include_str!("../../../test/alu_display/bcd_digit_to_7seg.v");
        for blank in [true, false] {
            let make = |source: &str| {
                let source = if blank {
                    source.to_owned()
                } else {
                    format!("{source}\n{wrapper}")
                };
                LogicalDesign::from_verilog_source(&source)?
                    .lower_to_routable_with_literal_associative_cells(8)
            };
            let baseline = make(old)?;
            let mapped = make(minimized)?;
            let stats = |design: &RoutableDesign| {
                let RoutableModuleBody::Composite { instances, nets } =
                    &design.module(&design.top).unwrap().body
                else {
                    panic!()
                };
                (
                    instances.len(),
                    nets.iter()
                        .flat_map(|net| &net.sinks)
                        .filter(|sink| matches!(sink, Endpoint::InstancePort { .. }))
                        .count(),
                    nets.iter().map(|net| net.sinks.len()).max().unwrap_or(0),
                )
            };
            let before = stats(&baseline);
            let after = stats(&mapped);
            eprintln!("minimized decoder blank={blank}: (cells, input edges, max fanout) {before:?} -> {after:?}");
            assert!(after.1 < before.1);
            for mask in 0..if blank { 32 } else { 16 } {
                let mut input = HashMap::new();
                bits(&mut input, "digit", 4, mask & 15);
                if blank {
                    bits(&mut input, "blank", 1, mask >> 4);
                }
                let actual = simulate(&mapped, &input);
                assert_eq!(actual, simulate(&baseline, &input));
                assert_eq!(
                    number(&actual, "seg", 7),
                    if mask & 16 == 0 { masks[mask & 15] } else { 0 }
                );
            }
            for leaf in mapped
                .modules
                .iter()
                .filter(|module| matches!(module.body, RoutableModuleBody::Leaf { .. }))
            {
                let table = LogicGraph {
                    graph: crate::ir::graph_from_routable_leaf(leaf)?,
                }
                .truth_table()?;
                assert!(table.input_names.len() <= 8);
                let values = table.output_tables.values().next().unwrap();
                let count = values.iter().filter(|&&v| v).count();
                assert!(count == 1 || count == values.len() - 1);
            }
        }
        Ok(())
    }

    #[test]
    fn explicit_shared_add_sub_fixtures_match_integer_oracle() -> eyre::Result<()> {
        for (width, source) in [
            (1, include_str!("../../../test/alu_display/add_sub1.v")),
            (2, include_str!("../../../test/alu_display/add_sub2.v")),
            (4, include_str!("../../../test/alu_display/add_sub4.v")),
        ] {
            let logical = LogicalDesign::from_verilog_source(source)?;
            let design = logical.lower_to_routable_with_associative_cells(8)?;
            let mask = (1usize << width) - 1;
            eprintln!("shared add/sub {width}bit: {:?}", primitive_stats(&design));
            for a in 0..=mask {
                for b in 0..=mask {
                    for subtract in 0..2 {
                        let mut input = HashMap::new();
                        bits(&mut input, "a", width, a);
                        bits(&mut input, "b", width, b);
                        bits(&mut input, "subtract", 1, subtract);
                        let output = simulate(&design, &input);
                        assert_eq!(
                            number(&output, "y", width),
                            if subtract != 0 {
                                a.wrapping_sub(b) & mask
                            } else {
                                (a + b) & mask
                            }
                        );
                        assert_eq!(
                            output["c"],
                            if subtract != 0 { a >= b } else { a + b > mask }
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn unsigned_arithmetic_select_concat_and_mux_are_bit_exact() -> eyre::Result<()> {
        let design = compile(
            "module arithmetic(a,b,s,sum,diff,ge,eq,y,rev); input [3:0] a,b; input s; output [4:0] sum; output [3:0] diff,y,rev; output ge,eq; assign sum={1'b0,a}+{1'b0,b}; assign diff=a-b; assign ge=a>=b; assign eq=a==b; assign y=s?a:b; assign rev={a[0],a[1],a[2],a[3]}; endmodule",
        )?;
        for a in 0..16 {
            for b in 0..16 {
                for s in 0..2 {
                    let mut input = HashMap::new();
                    bits(&mut input, "a", 4, a);
                    bits(&mut input, "b", 4, b);
                    bits(&mut input, "s", 1, s);
                    let output = simulate(&design, &input);
                    assert_eq!(number(&output, "sum", 5), a + b);
                    assert_eq!(number(&output, "diff", 4), a.wrapping_sub(b) & 15);
                    assert_eq!(number(&output, "ge", 1), usize::from(a >= b));
                    assert_eq!(number(&output, "eq", 1), usize::from(a == b));
                    assert_eq!(number(&output, "y", 4), if s == 1 { a } else { b });
                    assert_eq!(
                        number(&output, "rev", 4),
                        (a as u8).reverse_bits() as usize >> 4
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn nested_hierarchy_and_expression_bindings_preserve_bits() -> eyre::Result<()> {
        let design = compile(
            "module invert(a,y); input [1:0] a; output [1:0] y; assign y=~a; endmodule module inner(a,y); input [3:0] a; output [1:0] y; invert u(.a({a[0],a[3]}),.y(y)); endmodule module outer(a,y,z); input [3:0] a; output [1:0] y; output z; wire [3:0] intermediate; assign intermediate=a^4'b1010; inner u(.a(intermediate),.y(y)); assign z=y[1]^y[0]; endmodule",
        )?;
        for a in 0..16 {
            let mut input = HashMap::new();
            bits(&mut input, "a", 4, a);
            let output = simulate(&design, &input);
            let intermediate = a ^ 10;
            let expected = (!(((intermediate & 1) << 1) | (intermediate >> 3))) & 3;
            assert_eq!(number(&output, "y", 2), expected);
            assert_eq!(number(&output, "z", 1), (expected >> 1) ^ (expected & 1));
        }
        Ok(())
    }

    #[test]
    fn constant_and_pass_through_outputs_have_physical_drivers() -> eyre::Result<()> {
        let design = compile(
            "module wiring(a,y,z); input [1:0] a; output [3:0] y; output z; assign y={2'b10,a}; assign z=a[1]; endmodule",
        )?;
        for a in 0..4 {
            let mut input = HashMap::new();
            bits(&mut input, "a", 2, a);
            let output = simulate(&design, &input);
            assert_eq!(number(&output, "y", 4), 8 + a);
            assert_eq!(number(&output, "z", 1), a >> 1);
        }
        Ok(())
    }
}
