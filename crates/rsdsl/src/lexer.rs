//! Tokens of rsdsl v2 (docs/solver_dsl_grammar.md §2).

use crate::diag::{Diagnostic, Result, Span};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Str(String),
    /// A reserved word.
    Kw(&'static str),
    /// Punctuation or an operator.
    P(&'static str),
    /// A lone `_`.
    Hole,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

pub const KEYWORDS: &[&str] = &[
    "rsdsl",
    "model",
    "instance",
    "of",
    "include",
    "grid",
    "dirs",
    "enum",
    "subset",
    "domain",
    "fact",
    "param",
    "fn",
    "choice",
    "var",
    "over",
    "def",
    "relation",
    "int",
    "bool",
    "set",
    "rule",
    "minimize",
    "maximize",
    "require",
    "forall",
    "where",
    "if",
    "else",
    "let",
    "match",
    "for",
    "in",
    "is",
    "not",
    "and",
    "or",
    "xor",
    "true",
    "false",
    "none",
    "any",
    "all",
    "count",
    "exactly_one",
    "at_most_one",
];

// Longest first so that maximal munch works by scanning in order.
const PUNCT: &[&str] = &[
    "<->", "..=", "->", "=>", "==", "!=", "<=", ">=", ":=", "|=", "..", ";", ",", ":", "=", "(",
    ")", "[", "]", "{", "}", ".", "@", "?", "|", "<", ">", "+", "-", "*", "/", "%",
];

pub fn lex(file: u32, text: &str) -> Result<Vec<Token>> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if text[i..].starts_with("//") {
            i = text[i..].find('\n').map_or(bytes.len(), |n| i + n);
            continue;
        }
        if text[i..].starts_with("/*") {
            let Some(end) = text[i + 2..].find("*/") else {
                return Err(Diagnostic::error(
                    "E0001",
                    "unterminated block comment",
                    Span::new(file, i, i + 2),
                ));
            };
            i += end + 4;
            continue;
        }
        let start = i;
        if c.is_ascii_digit() {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let value = text[start..i].parse::<i64>().map_err(|_| {
                Diagnostic::error(
                    "E0002",
                    "integer literal is too large",
                    Span::new(file, start, i),
                )
            })?;
            tokens.push(Token {
                tok: Tok::Int(value),
                span: Span::new(file, start, i),
            });
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &text[start..i];
            let tok = if word == "_" {
                Tok::Hole
            } else if let Some(keyword) = KEYWORDS.iter().find(|k| **k == word) {
                Tok::Kw(keyword)
            } else {
                Tok::Ident(word.to_owned())
            };
            tokens.push(Token {
                tok,
                span: Span::new(file, start, i),
            });
            continue;
        }
        if c == b'"' {
            i += 1;
            let mut value = String::new();
            loop {
                let Some(ch) = text[i..].chars().next() else {
                    return Err(Diagnostic::error(
                        "E0003",
                        "unterminated string literal",
                        Span::new(file, start, i),
                    ));
                };
                match ch {
                    '"' => {
                        i += 1;
                        break;
                    }
                    '\n' => {
                        return Err(Diagnostic::error(
                            "E0003",
                            "string literal spans a line break",
                            Span::new(file, start, i),
                        ))
                    }
                    '\\' => {
                        let escaped = text[i + 1..].chars().next().unwrap_or(' ');
                        value.push(match escaped {
                            'n' => '\n',
                            't' => '\t',
                            '"' => '"',
                            '\\' => '\\',
                            other => {
                                return Err(Diagnostic::error(
                                    "E0004",
                                    format!("unknown escape `\\{other}`"),
                                    Span::new(file, i, i + 2),
                                ))
                            }
                        });
                        i += 1 + escaped.len_utf8();
                    }
                    other => {
                        value.push(other);
                        i += other.len_utf8();
                    }
                }
            }
            tokens.push(Token {
                tok: Tok::Str(value),
                span: Span::new(file, start, i),
            });
            continue;
        }
        let Some(punct) = PUNCT.iter().find(|p| text[i..].starts_with(**p)) else {
            let ch = text[i..].chars().next().unwrap();
            return Err(Diagnostic::error(
                "E0005",
                format!("unexpected character `{ch}`"),
                Span::new(file, i, i + ch.len_utf8()),
            ));
        };
        i += punct.len();
        tokens.push(Token {
            tok: Tok::P(punct),
            span: Span::new(file, start, i),
        });
    }
    tokens.push(Token {
        tok: Tok::Eof,
        span: Span::new(file, bytes.len(), bytes.len()),
    });
    Ok(tokens)
}
