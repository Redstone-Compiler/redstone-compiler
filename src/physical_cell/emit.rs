use std::fmt;

use super::{CellGlyph, PhysicalCellDocument};

impl fmt::Display for PhysicalCellDocument {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(output, "rcell 1;")?;
        writeln!(
            output,
            "cell \"{}\" size [{}, {}, {}] {{",
            self.name, self.size.0, self.size.1, self.size.2
        )?;
        let mut support = Vec::new();
        if self.auto_support.dust {
            support.push("dust");
        }
        if self.auto_support.repeater {
            support.push("repeater");
        }
        if !support.is_empty() {
            writeln!(output, "  auto-support {};", support.join(" "))?;
        }
        for (glyph, spec) in &self.glyphs {
            match spec {
                CellGlyph::Torch { support } => writeln!(
                    output,
                    "  glyph \"{glyph}\" = torch on {};",
                    support.as_str()
                )?,
                CellGlyph::Repeater { toward, delay } => writeln!(
                    output,
                    "  glyph \"{glyph}\" = repeater toward {} delay {delay};",
                    toward.as_str()
                )?,
            }
        }
        if !self.glyphs.is_empty() {
            writeln!(output)?;
        }
        for input in &self.inputs {
            writeln!(
                output,
                "  input \"{}\" at [{}, {}, {}] on {};",
                input.name,
                input.position.0,
                input.position.1,
                input.position.2,
                input.support.as_str()
            )?;
        }
        for cell_output in &self.outputs {
            writeln!(
                output,
                "  output \"{}\" at [{}, {}, {}];",
                cell_output.name,
                cell_output.position.0,
                cell_output.position.1,
                cell_output.position.2
            )?;
        }
        for probe in &self.probes {
            writeln!(
                output,
                "  probe \"{}\" at [{}, {}, {}];",
                probe.name, probe.position.0, probe.position.1, probe.position.2
            )?;
        }
        if !self.inputs.is_empty() || !self.outputs.is_empty() || !self.probes.is_empty() {
            writeln!(output)?;
        }
        for plane in &self.planes {
            writeln!(
                output,
                "  plane {} at {}={} {{",
                plane.axes.as_str(),
                plane.axes.fixed_axis(),
                plane.fixed
            )?;
            for (row, cells) in plane.rows.iter().rev() {
                writeln!(
                    output,
                    "    {}={} \"{}\";",
                    plane.axes.row_axis(),
                    row,
                    cells
                )?;
            }
            writeln!(output, "  }}")?;
            writeln!(output)?;
        }
        for expectation in &self.expectations {
            writeln!(
                output,
                "  expect \"{}\" = {};",
                expectation.output, expectation.expression
            )?;
        }
        writeln!(output, "}}")
    }
}
