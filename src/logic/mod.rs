#[derive(Debug, Clone, PartialEq)]
pub enum LogicType {
    Not,
    And,
    Or,
    Xor,
}

impl LogicType {
    pub fn name(&self) -> String {
        match self {
            LogicType::Not => "Not".to_owned(),
            LogicType::And => "And".to_owned(),
            LogicType::Or => "Or".to_owned(),
            LogicType::Xor => "Xor".to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Logic {
    pub logic_type: LogicType,
}
