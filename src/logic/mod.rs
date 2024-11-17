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

    pub fn is_not(&self) -> bool {
        matches!(self, LogicType::Not)
    }

    pub fn is_or(&self) -> bool {
        matches!(self, LogicType::Or)
    }
}

#[derive(Debug, Clone, derive_more::Deref)]
pub struct Logic {
    #[deref]
    pub logic_type: LogicType,
}
