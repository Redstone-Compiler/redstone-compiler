use std::collections::BTreeMap;
use std::str::FromStr;

use eyre::{bail, Context, ContextCompat};

use super::{
    AutoSupport, AxisDirection, CellExpectation, CellGlyph, CellInput, CellOutput, CellPlane,
    CellProbe, PhysicalCellDocument, PlaneAxes,
};
use crate::world::position::{DimSize, Position};

impl FromStr for PhysicalCellDocument {
    type Err = eyre::Report;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Parser::new(source).parse()
    }
}

struct Parser {
    lines: Vec<(usize, String)>,
    cursor: usize,
}

impl Parser {
    fn new(source: &str) -> Self {
        let lines = source
            .lines()
            .enumerate()
            .filter_map(|(index, line)| {
                let line = line.split_once("//").map_or(line, |(before, _)| before);
                let line = line.trim();
                (!line.is_empty()).then(|| (index + 1, line.to_owned()))
            })
            .collect();
        Self { lines, cursor: 0 }
    }

    fn parse(mut self) -> eyre::Result<PhysicalCellDocument> {
        self.expect_exact("rcell 1;")?;
        let (_, header) = self.next().context("expected cell declaration")?;
        let header = header
            .strip_prefix("cell ")
            .context("expected `cell \"name\" size [x, y, z] {`")?;
        let (name, rest) = take_quoted(header)?;
        let rest = rest
            .trim()
            .strip_prefix("size ")
            .context("expected cell size")?;
        let (size, rest) = take_position(rest)?;
        if rest.trim() != "{" {
            bail!("expected `{{` after cell size");
        }

        let mut document = PhysicalCellDocument {
            name,
            size: DimSize(size.0, size.1, size.2),
            auto_support: AutoSupport::default(),
            settled_start: false,
            glyphs: BTreeMap::new(),
            inputs: Vec::new(),
            probes: Vec::new(),
            outputs: Vec::new(),
            planes: Vec::new(),
            expectations: Vec::new(),
        };

        loop {
            let (line_number, line) = self.next().context("unterminated cell block")?;
            if line == "}" {
                break;
            }
            if line.starts_with("start ") {
                document.settled_start = match strip_semicolon(line.strip_prefix("start ").unwrap())
                    .wrap_err_with(|| format!("line {line_number}"))?
                    .trim()
                {
                    "settled" => true,
                    "lit" => false,
                    other => bail!("line {line_number}: unknown start state `{other}`"),
                };
            } else if line.starts_with("auto-support ") {
                parse_auto_support(&line, &mut document.auto_support)
                    .wrap_err_with(|| format!("line {line_number}"))?;
            } else if line.starts_with("glyph ") {
                let (glyph, spec) =
                    parse_glyph(&line).wrap_err_with(|| format!("line {line_number}"))?;
                if matches!(glyph, '.' | '#' | 'r') {
                    bail!("line {line_number}: `.`, `#`, and `r` are built-in glyphs");
                }
                if document.glyphs.insert(glyph, spec).is_some() {
                    bail!("line {line_number}: duplicate glyph `{glyph}`");
                }
            } else if line.starts_with("input ") {
                document
                    .inputs
                    .push(parse_input(&line).wrap_err_with(|| format!("line {line_number}"))?);
            } else if line.starts_with("output ") {
                document
                    .outputs
                    .push(parse_output(&line).wrap_err_with(|| format!("line {line_number}"))?);
            } else if line.starts_with("probe ") {
                document
                    .probes
                    .push(parse_probe(&line).wrap_err_with(|| format!("line {line_number}"))?);
            } else if line.starts_with("expect ") {
                document.expectations.push(
                    parse_expectation(&line).wrap_err_with(|| format!("line {line_number}"))?,
                );
            } else if line.starts_with("plane ") {
                document.planes.push(
                    self.parse_plane(line_number, &line)
                        .wrap_err_with(|| format!("line {line_number}"))?,
                );
            } else {
                bail!("line {line_number}: unknown cell statement `{line}`");
            }
        }

        if let Some((line, text)) = self.next() {
            bail!("line {line}: unexpected text after cell: `{text}`");
        }
        Ok(document)
    }

    fn parse_plane(&mut self, header_line: usize, header: &str) -> eyre::Result<CellPlane> {
        let header = header
            .strip_prefix("plane ")
            .unwrap()
            .strip_suffix(" {")
            .context("expected `plane yz at x=N {`")?;
        let (axes, rest) = header
            .split_once(" at ")
            .context("expected `at` in plane declaration")?;
        let axes = parse_plane_axes(axes)?;
        let (fixed_axis, fixed) = parse_axis_assignment(rest)?;
        if fixed_axis != axes.fixed_axis() {
            bail!(
                "plane {} must fix {}, not {}",
                axes.as_str(),
                axes.fixed_axis(),
                fixed_axis
            );
        }

        let mut rows = BTreeMap::new();
        loop {
            let (line_number, line) = self
                .next()
                .with_context(|| format!("unterminated plane from line {header_line}"))?;
            if line == "}" {
                break;
            }
            let line = strip_semicolon(&line)?;
            let (coordinate, cells) = line
                .split_once(char::is_whitespace)
                .context("expected `z=N \"cells\";`")?;
            let (axis, row) = parse_axis_assignment(coordinate)?;
            if axis != axes.row_axis() {
                bail!(
                    "line {line_number}: plane {} rows must use {}",
                    axes.as_str(),
                    axes.row_axis()
                );
            }
            let (cells, rest) = take_quoted(cells.trim())?;
            if !rest.trim().is_empty() {
                bail!("line {line_number}: unexpected text after plane row");
            }
            if rows.insert(row, cells).is_some() {
                bail!("line {line_number}: duplicate plane row {axis}={row}");
            }
        }
        Ok(CellPlane { axes, fixed, rows })
    }

    fn expect_exact(&mut self, expected: &str) -> eyre::Result<()> {
        let (line, actual) = self.next().context("unexpected end of file")?;
        if actual != expected {
            bail!("line {line}: expected `{expected}`, found `{actual}`");
        }
        Ok(())
    }

    fn next(&mut self) -> Option<(usize, String)> {
        let item = self.lines.get(self.cursor).cloned();
        self.cursor += usize::from(item.is_some());
        item
    }
}

fn parse_auto_support(line: &str, support: &mut AutoSupport) -> eyre::Result<()> {
    for name in strip_semicolon(line.strip_prefix("auto-support ").unwrap())?.split_whitespace() {
        match name {
            "dust" => support.dust = true,
            "repeater" => support.repeater = true,
            _ => bail!("unknown auto-support target `{name}`"),
        }
    }
    Ok(())
}

fn parse_glyph(line: &str) -> eyre::Result<(char, CellGlyph)> {
    let line = strip_semicolon(line.strip_prefix("glyph ").unwrap())?;
    let (glyph, rest) = take_quoted(line)?;
    let mut chars = glyph.chars();
    let glyph = chars.next().context("glyph cannot be empty")?;
    if chars.next().is_some() {
        bail!("glyph must contain exactly one character");
    }
    let rest = rest
        .trim()
        .strip_prefix('=')
        .context("expected `=` after glyph")?
        .trim();
    if let Some(rest) = rest.strip_prefix("torch on ") {
        return Ok((
            glyph,
            CellGlyph::Torch {
                support: parse_direction(rest.trim())?,
            },
        ));
    }
    if let Some(rest) = rest.strip_prefix("repeater toward ") {
        let mut tokens = rest.split_whitespace();
        let toward = parse_direction(tokens.next().context("missing repeater direction")?)?;
        let mut delay = 1;
        if let Some(keyword) = tokens.next() {
            if keyword != "delay" {
                bail!("expected `delay`, found `{keyword}`");
            }
            delay = tokens
                .next()
                .context("missing repeater delay")?
                .parse()
                .context("invalid repeater delay")?;
        }
        if tokens.next().is_some() {
            bail!("unexpected text in repeater glyph");
        }
        if !matches!(
            toward,
            AxisDirection::XNegative
                | AxisDirection::XPositive
                | AxisDirection::YNegative
                | AxisDirection::YPositive
        ) {
            bail!("repeaters must point along x or y");
        }
        if !(1..=4).contains(&delay) {
            bail!("repeater delay must be in 1..=4");
        }
        return Ok((glyph, CellGlyph::Repeater { toward, delay }));
    }
    bail!("unknown glyph definition `{rest}`")
}

fn parse_input(line: &str) -> eyre::Result<CellInput> {
    let line = strip_semicolon(line.strip_prefix("input ").unwrap())?;
    let (name, rest) = take_quoted(line)?;
    let rest = rest
        .trim()
        .strip_prefix("at ")
        .context("expected input position")?;
    let (position, rest) = take_position(rest)?;
    let support = parse_direction(
        rest.trim()
            .strip_prefix("on ")
            .context("expected input support direction")?,
    )?;
    Ok(CellInput {
        name,
        position,
        support,
    })
}

fn parse_output(line: &str) -> eyre::Result<CellOutput> {
    let line = strip_semicolon(line.strip_prefix("output ").unwrap())?;
    let (name, rest) = take_quoted(line)?;
    let rest = rest
        .trim()
        .strip_prefix("at ")
        .context("expected output position")?;
    let (position, rest) = take_position(rest)?;
    if !rest.trim().is_empty() {
        bail!("unexpected text after output position");
    }
    Ok(CellOutput { name, position })
}

fn parse_probe(line: &str) -> eyre::Result<CellProbe> {
    let line = strip_semicolon(line.strip_prefix("probe ").unwrap())?;
    let (name, rest) = take_quoted(line)?;
    let rest = rest
        .trim()
        .strip_prefix("at ")
        .context("expected probe position")?;
    let (position, rest) = take_position(rest)?;
    if !rest.trim().is_empty() {
        bail!("unexpected text after probe position");
    }
    Ok(CellProbe { name, position })
}

fn parse_expectation(line: &str) -> eyre::Result<CellExpectation> {
    let line = strip_semicolon(line.strip_prefix("expect ").unwrap())?;
    let (output, rest) = take_quoted(line)?;
    let expression = rest
        .trim()
        .strip_prefix('=')
        .context("expected `=` after expected output")?
        .trim();
    if expression.is_empty() {
        bail!("expected logic expression");
    }
    Ok(CellExpectation {
        output,
        expression: expression.to_owned(),
    })
}

fn parse_plane_axes(text: &str) -> eyre::Result<PlaneAxes> {
    match text {
        "xy" => Ok(PlaneAxes::Xy),
        "xz" => Ok(PlaneAxes::Xz),
        "yz" => Ok(PlaneAxes::Yz),
        _ => bail!("unknown plane axes `{text}`"),
    }
}

fn parse_direction(text: &str) -> eyre::Result<AxisDirection> {
    match text {
        "x-" => Ok(AxisDirection::XNegative),
        "x+" => Ok(AxisDirection::XPositive),
        "y-" => Ok(AxisDirection::YNegative),
        "y+" => Ok(AxisDirection::YPositive),
        "z-" => Ok(AxisDirection::ZNegative),
        "z+" => Ok(AxisDirection::ZPositive),
        _ => bail!("unknown axis direction `{text}`"),
    }
}

fn parse_axis_assignment(text: &str) -> eyre::Result<(char, usize)> {
    let (axis, value) = text
        .split_once('=')
        .context("expected axis coordinate such as `x=0`")?;
    let mut chars = axis.chars();
    let axis = chars.next().context("missing axis")?;
    if chars.next().is_some() || !matches!(axis, 'x' | 'y' | 'z') {
        bail!("invalid axis `{axis}`");
    }
    Ok((axis, value.parse().context("invalid axis coordinate")?))
}

fn take_position(text: &str) -> eyre::Result<(Position, &str)> {
    let start = text.find('[').context("expected `[`")?;
    if !text[..start].trim().is_empty() {
        bail!("unexpected text before position");
    }
    let end = text[start..].find(']').context("expected `]`")? + start;
    let values = text[start + 1..end]
        .split(',')
        .map(|value| value.trim().parse::<usize>().context("invalid position"))
        .collect::<eyre::Result<Vec<_>>>()?;
    if values.len() != 3 {
        bail!("position must contain exactly three coordinates");
    }
    Ok((Position(values[0], values[1], values[2]), &text[end + 1..]))
}

fn take_quoted(text: &str) -> eyre::Result<(String, &str)> {
    let text = text.trim_start();
    let rest = text.strip_prefix('"').context("expected quoted string")?;
    let end = rest.find('"').context("unterminated quoted string")?;
    Ok((rest[..end].to_owned(), &rest[end + 1..]))
}

fn strip_semicolon(text: &str) -> eyre::Result<&str> {
    text.trim()
        .strip_suffix(';')
        .context("expected trailing `;`")
}
