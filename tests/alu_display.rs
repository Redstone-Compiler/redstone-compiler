use std::collections::HashMap;

use redstone_compiler::verilog::parser;
use redstone_compiler::verilog::rtl::{lower_rtl_module, RtlExpr, RtlModule, RtlSignalId};

#[path = "../test/alu_display/oracle.rs"]
mod oracle;

// RTL lowering의 결과를 평가하고 독립적인 산술/segment oracle과 비교한다.
// parser/상수 폭/조건식/출력 전체 할당을 실제 fixture로 검증한다.
fn evaluate(rtl: &RtlModule, expr: &RtlExpr, values: &HashMap<RtlSignalId, u128>) -> u128 {
    let result = match expr {
        RtlExpr::Signal(signal) => values[&signal.signal],
        RtlExpr::Const { value, .. } => *value,
        RtlExpr::GreaterEqual(left, right) => {
            u128::from(evaluate(rtl, left, values) >= evaluate(rtl, right, values))
        }
        RtlExpr::Sub(left, right) => {
            evaluate(rtl, left, values).wrapping_sub(evaluate(rtl, right, values))
        }
        RtlExpr::Equal(left, right) => {
            u128::from(evaluate(rtl, left, values) == evaluate(rtl, right, values))
        }
        RtlExpr::Mux {
            select,
            when_true,
            when_false,
        } => evaluate(
            rtl,
            if evaluate(rtl, select, values) != 0 {
                when_true
            } else {
                when_false
            },
            values,
        ),
        RtlExpr::Resize { value, width } => {
            let mask = if *width == 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            evaluate(rtl, value, values) & mask
        }
        RtlExpr::Not(value) => !evaluate(rtl, value, values),
        RtlExpr::And(left, right) => evaluate(rtl, left, values) & evaluate(rtl, right, values),
        RtlExpr::Or(left, right) => evaluate(rtl, left, values) | evaluate(rtl, right, values),
        RtlExpr::Xor(left, right) => evaluate(rtl, left, values) ^ evaluate(rtl, right, values),
        RtlExpr::Add(left, right) => {
            evaluate(rtl, left, values).wrapping_add(evaluate(rtl, right, values))
        }
        RtlExpr::Select { value, msb, lsb } => {
            (evaluate(rtl, value, values) >> lsb) & ((1u128 << (msb - lsb + 1)) - 1)
        }
        RtlExpr::Concat(parts) => parts.iter().fold(0, |value, part| {
            (value << rtl.expr_width(part).unwrap()) | evaluate(rtl, part, values)
        }),
    };
    let width = rtl.expr_width(expr).unwrap();
    result
        & if width == 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        }
}

#[test]
fn bcd_decoder_all_digits_and_blank_match_independent_segments() -> eyre::Result<()> {
    for source in [
        include_str!("../test/alu_display/bcd_to_7seg.v"),
        include_str!("../test/alu_display/bcd_to_7seg_sop.v"),
    ] {
        let modules = parser::parse_modules(source)?;
        let rtl = lower_rtl_module(&modules[0])?;
        let digit_signal = rtl.signal_ref("digit")?;
        let blank_signal = rtl.signal_ref("blank")?;
        let seg_signal = rtl.signal_ref("seg")?;
        assert_eq!(rtl.signal_width(digit_signal)?, 4);
        assert_eq!(rtl.signal_width(blank_signal)?, 1);
        assert_eq!(rtl.signal_width(seg_signal)?, 7);
        for blank in [false, true] {
            for digit in 0..16 {
                let mut values = HashMap::from([
                    (digit_signal.signal, digit as u128),
                    (blank_signal.signal, u128::from(blank)),
                ]);
                for assignment in &rtl.continuous_assigns {
                    let mask = (1u128 << rtl.signal_width(assignment.output)?) - 1;
                    values.insert(
                        assignment.output.signal,
                        evaluate(&rtl, &assignment.expr, &values) & mask,
                    );
                }
                assert_eq!(
                    values[&seg_signal.signal],
                    oracle::segment_mask(digit, blank) as u128,
                    "digit={digit}, blank={blank}"
                );
            }
        }
    }
    Ok(())
}

fn run(rtl: &RtlModule, inputs: &[(&str, u128)]) -> eyre::Result<HashMap<RtlSignalId, u128>> {
    let mut values = HashMap::new();
    for (name, value) in inputs {
        values.insert(rtl.signal_ref(name)?.signal, *value);
    }
    for assignment in &rtl.continuous_assigns {
        let width = rtl.signal_width(assignment.output)?;
        let mask = if width == 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        values.insert(
            assignment.output.signal,
            evaluate(rtl, &assignment.expr, &values) & mask,
        );
    }
    Ok(values)
}

#[test]
fn double_dabble_all_256_values_match_decimal_arithmetic() -> eyre::Result<()> {
    let modules = parser::parse_modules(include_str!("../test/alu_display/bin8_to_bcd.v"))?;
    let rtl = lower_rtl_module(&modules[0])?;
    let h = rtl.signal_ref("hundreds")?.signal;
    let t = rtl.signal_ref("tens")?.signal;
    let o = rtl.signal_ref("ones")?.signal;
    for value in 0..256 {
        let values = run(&rtl, &[("value", value)])?;
        assert_eq!(
            [values[&h], values[&t], values[&o]],
            [value / 100, (value / 10) % 10, value % 10],
            "value={value}"
        );
    }
    Ok(())
}

#[test]
fn alu_all_524288_operands_and_operations_match_arithmetic() -> eyre::Result<()> {
    let modules = parser::parse_modules(include_str!("../test/alu_display/alu8.v"))?;
    let rtl = lower_rtl_module(&modules[0])?;
    let outputs = ["y", "z", "n", "c", "v"].map(|name| rtl.signal_ref(name).unwrap().signal);
    for a in 0..256u128 {
        for b in 0..256u128 {
            for op in 0..8 {
                let signed_a = a as u8 as i8 as i16;
                let signed_b = b as u8 as i8 as i16;
                let (y, c, v) = match op {
                    0 => (
                        (a + b) & 255,
                        a + b > 255,
                        !(-128..=127).contains(&(signed_a + signed_b)),
                    ),
                    1 => (
                        a.wrapping_sub(b) & 255,
                        a >= b,
                        !(-128..=127).contains(&(signed_a - signed_b)),
                    ),
                    2 => (a & b, false, false),
                    3 => (a | b, false, false),
                    4 => (a ^ b, false, false),
                    5 => (a, false, false),
                    6 => ((a * 2) & 255, a >= 128, false),
                    7 => (a / 2, a % 2 != 0, false),
                    _ => unreachable!(),
                };
                let values = run(&rtl, &[("a", a), ("b", b), ("op", op)])?;
                assert_eq!(
                    outputs.map(|id| values[&id]),
                    [
                        y,
                        u128::from(y == 0),
                        u128::from(y >= 128),
                        u128::from(c),
                        u128::from(v)
                    ],
                    "a={a}, b={b}, op={op}"
                );
            }
        }
    }
    Ok(())
}
