use redstone_compiler::ir::LogicalDesign;

#[test]
fn independent_child_outputs_allow_acyclic_cross_instance_wiring() -> eyre::Result<()> {
    // 첫 child의 low 출력은 b 입력에 의존하지 않는다. 전개하면 y=a, z=b다.
    let source = "module split(a,b,lo,hi); input [1:0] a,b; output [1:0] lo,hi; assign lo=a; assign hi=b; endmodule
      module top(a,b,y,z); input [1:0] a,b; output [1:0] y,z; wire [1:0] u,v;
      split first(.a(a),.b(v),.lo(u),.hi(z));
      split second(.a(b),.b(u),.lo(v),.hi(y)); endmodule";
    let design = LogicalDesign::from_verilog_source(source)?;
    let routable = design.lower_to_routable()?;
    routable.validate()?;
    for a in 0..4 {
        for b in 0..4 {
            let input = [a & 1 != 0, a & 2 != 0, b & 1 != 0, b & 2 != 0];
            assert_eq!(evaluate(&routable, &input), input);
        }
    }
    Ok(())
}

#[test]
fn scalar_port_names_cannot_alias_generated_bus_bit_names() -> eyre::Result<()> {
    let source =
        "module top(a,a_0,y); input [1:0] a; input a_0; output y; assign y=a[0]^a_0; endmodule";
    let design = LogicalDesign::from_verilog_source(source)?;
    let routable = design.lower_to_routable()?;
    routable.validate()?;
    for bits in 0..8 {
        let input = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
        assert_eq!(evaluate(&routable, &input), [input[0] ^ input[2]]);
    }
    Ok(())
}

// 실제 물리 배치 대신 생성된 Routable leaf 논리를 평가한다.
// 입력 목록은 선언 순서이며 버스 내부는 LSB부터다.
fn evaluate(design: &redstone_compiler::ir::RoutableDesign, input: &[bool]) -> Vec<bool> {
    use std::collections::HashMap;

    use redstone_compiler::ir::{
        Endpoint, RoutableModuleBody, RoutableNodeKind, RoutablePortDirection,
    };
    let top = design.module(&design.top).unwrap();
    let input_ports = top
        .ports
        .iter()
        .filter(|p| p.direction == RoutablePortDirection::Input)
        .collect::<Vec<_>>();
    assert_eq!(input_ports.len(), input.len());
    let mut values = input_ports
        .iter()
        .zip(input)
        .map(|(p, &v)| {
            (
                Endpoint::SelfPort {
                    port: p.name.clone(),
                },
                v,
            )
        })
        .collect::<HashMap<_, _>>();
    let RoutableModuleBody::Composite { instances, nets } = &top.body else {
        panic!("expected flat composite")
    };
    let mut remaining = instances.iter().collect::<Vec<_>>();
    while !remaining.is_empty() {
        for net in nets {
            if let Some(v) = values.get(&net.driver).copied() {
                for sink in &net.sinks {
                    values.insert(sink.clone(), v);
                }
            }
        }
        let before = remaining.len();
        remaining.retain(|instance| {
            let leaf = design.module(&instance.module).unwrap();
            let endpoint = |name: &str| Endpoint::InstancePort {
                instance: instance.name.clone(),
                port: name.to_owned(),
            };
            if leaf
                .ports
                .iter()
                .filter(|p| p.direction == RoutablePortDirection::Input)
                .any(|p| !values.contains_key(&endpoint(&p.name)))
            {
                return true;
            }
            let RoutableModuleBody::Leaf { nodes } = &leaf.body else {
                panic!("expected leaf")
            };
            let mut computed = HashMap::new();
            let mut unresolved = nodes.iter().collect::<Vec<_>>();
            while !unresolved.is_empty() {
                let previous = unresolved.len();
                unresolved.retain(|node| {
                    let args = node
                        .inputs
                        .iter()
                        .map(|id| computed.get(id).copied())
                        .collect::<Option<Vec<bool>>>();
                    let Some(args) = args else {
                        return true;
                    };
                    let result = match &node.kind {
                        RoutableNodeKind::Input { name } => values[&endpoint(name)],
                        RoutableNodeKind::Not => !args[0],
                        RoutableNodeKind::Or => args.iter().any(|v| *v),
                        RoutableNodeKind::Output { name } => {
                            values.insert(endpoint(name), args[0]);
                            args[0]
                        }
                        _ => panic!("unexpected sequential logic"),
                    };
                    computed.insert(node.id, result);
                    false
                });
                assert!(unresolved.len() < previous);
            }
            false
        });
        assert!(remaining.len() < before);
    }
    for net in nets {
        if let Some(v) = values.get(&net.driver).copied() {
            for sink in &net.sinks {
                values.insert(sink.clone(), v);
            }
        }
    }
    top.ports
        .iter()
        .filter(|p| p.direction == RoutablePortDirection::Output)
        .map(|p| {
            values[&Endpoint::SelfPort {
                port: p.name.clone(),
            }]
        })
        .collect()
}
