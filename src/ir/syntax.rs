use eyre::ContextCompat;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Token {
    Word(String),
    String(String),
    Number(u128),
    Symbol(char),
}

pub(crate) fn tokenize(source: &str) -> eyre::Result<Vec<Token>> {
    let mut chars = source.char_indices().peekable();
    let mut tokens = Vec::new();
    while let Some((offset, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }
        if ch == '#' {
            for (_, next) in chars.by_ref() {
                if next == '\n' {
                    break;
                }
            }
            continue;
        }
        if "{}[],:.;<>=()".contains(ch) {
            tokens.push(Token::Symbol(ch));
            continue;
        }
        if ch == '"' {
            let mut value = String::new();
            let mut closed = false;
            while let Some((_, next)) = chars.next() {
                if next == '"' {
                    closed = true;
                    break;
                }
                if next != '\\' {
                    value.push(next);
                    continue;
                }
                let (_, escaped) = chars.next().context("unterminated string escape")?;
                value.push(match escaped {
                    '\\' => '\\',
                    '"' => '"',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => eyre::bail!("unsupported string escape `\\{other}` at byte {offset}"),
                });
            }
            if !closed {
                eyre::bail!("unterminated string at byte {offset}");
            }
            tokens.push(Token::String(value));
            continue;
        }
        if ch.is_ascii_digit() {
            let mut value = ch.to_string();
            while let Some((_, next)) = chars.peek() {
                if !next.is_ascii_digit() {
                    break;
                }
                value.push(*next);
                chars.next();
            }
            tokens.push(Token::Number(value.parse()?));
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let mut value = ch.to_string();
            while let Some((_, next)) = chars.peek() {
                if !next.is_ascii_alphanumeric() && *next != '_' && *next != '-' {
                    break;
                }
                value.push(*next);
                chars.next();
            }
            tokens.push(Token::Word(value));
            continue;
        }
        eyre::bail!("unexpected character `{ch}` at byte {offset}");
    }
    Ok(tokens)
}
