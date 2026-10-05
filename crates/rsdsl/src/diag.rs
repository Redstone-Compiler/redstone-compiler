//! Diagnostics with source spans.

use std::fmt;
use std::sync::Arc;

/// Byte range in a source file, plus the file it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub file: u32,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(file: u32, start: usize, end: usize) -> Self {
        Self {
            file,
            start: start as u32,
            end: end as u32,
        }
    }

    pub fn to(self, other: Span) -> Span {
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// Source files known to the compiler, for rendering diagnostics.
#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    files: Vec<(Arc<str>, Arc<str>)>,
}

impl SourceMap {
    pub fn add(&mut self, name: &str, text: &str) -> u32 {
        self.files.push((name.into(), text.into()));
        (self.files.len() - 1) as u32
    }

    pub fn text(&self, file: u32) -> &str {
        &self.files[file as usize].1
    }

    fn location(&self, span: Span) -> Option<(&str, usize, usize, &str)> {
        let (name, text) = self.files.get(span.file as usize)?;
        let start = (span.start as usize).min(text.len());
        let line_start = text[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = text[start..].find('\n').map_or(text.len(), |i| start + i);
        let line = text[..start].matches('\n').count() + 1;
        let column = text[line_start..start].chars().count() + 1;
        Some((name, line, column, &text[line_start..line_end]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: &'static str,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn error(code: &'static str, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Error,
            code,
            message: message.into(),
            span: Some(span),
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn error_nospan(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn warning(code: &'static str, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Warning,
            ..Self::error(code, message, span)
        }
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Renders in the style of rustc, pointing at the source line.
    pub fn render(&self, sources: &SourceMap) -> String {
        let kind = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        let mut out = format!("{kind}[{}]: {}\n", self.code, self.message);
        if let Some(span) = self.span {
            if let Some((name, line, column, text)) = sources.location(span) {
                let gutter = line.to_string().len();
                let width = ((span.end - span.start) as usize).max(1);
                let width = width.min(text.chars().count().saturating_sub(column - 1).max(1));
                out += &format!("{:gutter$}--> {name}:{line}:{column}\n", "");
                out += &format!("{:gutter$} |\n", "");
                out += &format!("{line} | {text}\n");
                out += &format!(
                    "{:gutter$} | {}{}\n",
                    "",
                    " ".repeat(column - 1),
                    "^".repeat(width)
                );
            }
        }
        for note in &self.notes {
            out += &format!("  = note: {note}\n");
        }
        if let Some(help) = &self.help {
            out += &format!("  = help: {help}\n");
        }
        out
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

/// A failed compilation: every diagnostic plus the sources to render them.
#[derive(Debug, Clone)]
pub struct Error {
    pub diagnostics: Vec<Diagnostic>,
    pub sources: SourceMap,
}

impl Error {
    pub fn render(&self) -> String {
        self.diagnostics
            .iter()
            .map(|diagnostic| diagnostic.render(&self.sources))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Diagnostic>;
