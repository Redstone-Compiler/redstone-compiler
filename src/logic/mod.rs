#[derive(Debug, Copy, Clone)]
pub enum LogicType {
    Not,
    And,
    Or,
    Xor,
}

#[derive(Debug, Copy, Clone)]
pub struct Logic {
    logic_type: LogicType,
}
