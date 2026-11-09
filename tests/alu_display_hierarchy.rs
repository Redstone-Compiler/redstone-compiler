use redstone_compiler::ir::{LogicalDesign, LogicalPortDirection, RoutablePortDirection};

#[test]
fn three_digit_display_compiles_through_typed_hierarchy() -> eyre::Result<()> {
    let source = [
        include_str!("../test/alu_display/bin8_to_bcd.v"),
        include_str!("../test/alu_display/bcd_to_7seg.v"),
        include_str!("../test/alu_display/display_u8.v"),
    ]
    .join("\n");
    let logical = LogicalDesign::from_verilog_source(&source)?;
    let top = logical.module("display_u8").unwrap();
    assert_eq!(top.instances.len(), 4);
    assert_eq!(
        top.ports
            .iter()
            .filter(|p| p.direction == LogicalPortDirection::Output)
            .count(),
        3
    );
    // 입력 상수도 폭을 가진 일반 셀로 남아 계층 인스턴스와 함께 처리된다.
    assert!(!top.cells.is_empty());
    let serialized = logical.to_string();
    let parsed: LogicalDesign = serialized.parse()?;
    let routable = parsed.lower_to_routable()?;
    routable.validate()?;
    let top = routable
        .modules
        .iter()
        .find(|m| m.name == routable.top)
        .unwrap();
    assert_eq!(
        top.ports
            .iter()
            .filter(|p| p.direction == RoutablePortDirection::Input)
            .count(),
        8
    );
    assert_eq!(
        top.ports
            .iter()
            .filter(|p| p.direction == RoutablePortDirection::Output)
            .count(),
        21
    );
    Ok(())
}
