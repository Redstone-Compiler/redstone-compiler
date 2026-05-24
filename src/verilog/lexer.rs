#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Module,
    EndModule,
    Input,
    Output,
    Wire,
    Assign,
    Always,
    Ident(String),
    Number(usize),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    Eq,
    At,
    Not,
    And,
    Xor,
    Or,
}

pub fn lex(source: &str) -> eyre::Result<Vec<Token>> {
    let stripped = strip_comments(source);
    let chars = stripped.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() {
            index += 1;
            continue;
        }

        match ch {
            '(' => tokens.push(Token::LParen),
            ')' => tokens.push(Token::RParen),
            '[' => tokens.push(Token::LBracket),
            ']' => tokens.push(Token::RBracket),
            ',' => tokens.push(Token::Comma),
            ';' => tokens.push(Token::Semi),
            ':' => tokens.push(Token::Colon),
            '=' => tokens.push(Token::Eq),
            '@' => tokens.push(Token::At),
            '~' => tokens.push(Token::Not),
            '&' => tokens.push(Token::And),
            '^' => tokens.push(Token::Xor),
            '|' => tokens.push(Token::Or),
            ch if ch.is_ascii_digit() => {
                let start = index;
                index += 1;
                while index < chars.len() && chars[index].is_ascii_digit() {
                    index += 1;
                }
                let text = chars[start..index].iter().collect::<String>();
                tokens.push(Token::Number(text.parse()?));
                continue;
            }
            ch if is_ident_start(ch) => {
                let start = index;
                index += 1;
                while index < chars.len() && is_ident_continue(chars[index]) {
                    index += 1;
                }
                let text = chars[start..index].iter().collect::<String>();
                tokens.push(match text.as_str() {
                    "module" => Token::Module,
                    "endmodule" => Token::EndModule,
                    "input" => Token::Input,
                    "output" => Token::Output,
                    "wire" => Token::Wire,
                    "assign" => Token::Assign,
                    "always" => Token::Always,
                    _ => Token::Ident(text),
                });
                continue;
            }
            _ => eyre::bail!("unsupported Verilog character `{ch}` at byte-like index {index}"),
        }

        index += 1;
    }

    Ok(tokens)
}

fn strip_comments(source: &str) -> String {
    let mut result = String::new();
    let mut chars = source.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\n' {
                    result.push('\n');
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::{lex, Token};

    #[test]
    fn lexes_combinational_module_subset() -> eyre::Result<()> {
        let tokens = lex(
            r#"
            module half_adder(a, b, s, c);
              input a, b;
              output s, c;
              assign s = a ^ b;
              assign c = a & b;
            endmodule
            "#,
        )?;

        assert_eq!(
            tokens,
            vec![
                Token::Module,
                Token::Ident("half_adder".to_owned()),
                Token::LParen,
                Token::Ident("a".to_owned()),
                Token::Comma,
                Token::Ident("b".to_owned()),
                Token::Comma,
                Token::Ident("s".to_owned()),
                Token::Comma,
                Token::Ident("c".to_owned()),
                Token::RParen,
                Token::Semi,
                Token::Input,
                Token::Ident("a".to_owned()),
                Token::Comma,
                Token::Ident("b".to_owned()),
                Token::Semi,
                Token::Output,
                Token::Ident("s".to_owned()),
                Token::Comma,
                Token::Ident("c".to_owned()),
                Token::Semi,
                Token::Assign,
                Token::Ident("s".to_owned()),
                Token::Eq,
                Token::Ident("a".to_owned()),
                Token::Xor,
                Token::Ident("b".to_owned()),
                Token::Semi,
                Token::Assign,
                Token::Ident("c".to_owned()),
                Token::Eq,
                Token::Ident("a".to_owned()),
                Token::And,
                Token::Ident("b".to_owned()),
                Token::Semi,
                Token::EndModule,
            ]
        );

        Ok(())
    }
}
