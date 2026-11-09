#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Module,
    EndModule,
    Input,
    Output,
    Reg,
    Wire,
    Assign,
    Always,
    Posedge,
    Begin,
    End,
    If,
    Ident(String),
    Number(usize),
    SizedNumber { value: u128, width: usize },
    LBrace,
    RBrace,
    Question,
    Equal,
    GreaterEqual,
    Minus,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Dot,
    Comma,
    Semi,
    Colon,
    Eq,
    Le,
    At,
    Star,
    Not,
    And,
    Xor,
    Or,
    Plus,
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
            '.' => tokens.push(Token::Dot),
            ',' => tokens.push(Token::Comma),
            ';' => tokens.push(Token::Semi),
            ':' => tokens.push(Token::Colon),
            '=' if chars.get(index + 1) == Some(&'=') => {
                tokens.push(Token::Equal);
                index += 2;
                continue;
            }
            '=' => tokens.push(Token::Eq),
            '{' => tokens.push(Token::LBrace),
            '}' => tokens.push(Token::RBrace),
            '?' => tokens.push(Token::Question),
            '@' => tokens.push(Token::At),
            '~' => tokens.push(Token::Not),
            '&' => tokens.push(Token::And),
            '^' => tokens.push(Token::Xor),
            '|' => tokens.push(Token::Or),
            '+' => tokens.push(Token::Plus),
            '-' => tokens.push(Token::Minus),
            '>' if chars.get(index + 1) == Some(&'=') => {
                tokens.push(Token::GreaterEqual);
                index += 2;
                continue;
            }
            '*' => tokens.push(Token::Star),
            '<' if chars.get(index + 1) == Some(&'=') => {
                tokens.push(Token::Le);
                index += 2;
                continue;
            }
            ch if ch.is_ascii_digit() => {
                let start = index;
                index += 1;
                while index < chars.len() && chars[index].is_ascii_digit() {
                    index += 1;
                }
                let text = chars[start..index].iter().collect::<String>();
                if chars.get(index) == Some(&'\'') {
                    let width: usize = text.parse()?;
                    eyre::ensure!(
                        (1..=128).contains(&width),
                        "literal width must be in 1..=128"
                    );
                    index += 1;
                    let radix = match chars.get(index).map(char::to_ascii_lowercase) {
                        Some('b') => 2,
                        Some('h') => 16,
                        Some('d') => 10,
                        _ => eyre::bail!("expected unsigned sized binary, hex, or decimal literal"),
                    };
                    index += 1;
                    let start = index;
                    while index < chars.len()
                        && (chars[index].is_ascii_alphanumeric() || chars[index] == '_')
                    {
                        index += 1;
                    }
                    let digits = chars[start..index]
                        .iter()
                        .filter(|c| **c != '_')
                        .collect::<String>();
                    let value = u128::from_str_radix(&digits, radix)
                        .map_err(|_| eyre::eyre!("invalid two-state sized literal `{digits}`"))?;
                    let value = if width == 128 {
                        value
                    } else {
                        value & ((1u128 << width) - 1)
                    };
                    tokens.push(Token::SizedNumber { value, width });
                } else {
                    tokens.push(Token::Number(text.parse()?));
                }
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
                    "reg" => Token::Reg,
                    "wire" => Token::Wire,
                    "assign" => Token::Assign,
                    "always" => Token::Always,
                    "posedge" => Token::Posedge,
                    "begin" => Token::Begin,
                    "end" => Token::End,
                    "if" => Token::If,
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
    fn sized_literals_are_two_state_and_keep_declared_width() -> eyre::Result<()> {
        assert_eq!(
            lex("4'h1f 8'b1010_0101 7'd127")?,
            vec![
                Token::SizedNumber {
                    value: 15,
                    width: 4
                },
                Token::SizedNumber {
                    value: 165,
                    width: 8
                },
                Token::SizedNumber {
                    value: 127,
                    width: 7
                },
            ]
        );
        for bad in ["0'h0", "129'h1", "4'b10xz", "8'shff", "4'h"] {
            assert!(lex(bad).is_err(), "{bad}");
        }
        Ok(())
    }

    #[test]
    fn lexes_combinational_module_subset() -> eyre::Result<()> {
        let tokens = lex(r#"
            module half_adder(a, b, s, c);
              input a, b;
              output s, c;
              assign s = a ^ b;
              assign c = a & b;
            endmodule
            "#)?;

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
