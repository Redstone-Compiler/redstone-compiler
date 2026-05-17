#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SequentialType {
    RsLatch,
    DLatch,
    DFlipFlop,
}

impl SequentialType {
    pub fn name(&self) -> String {
        match self {
            SequentialType::RsLatch => "RsLatch".to_owned(),
            SequentialType::DLatch => "DLatch".to_owned(),
            SequentialType::DFlipFlop => "DFlipFlop".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SequentialPrimitive {
    pub sequential_type: SequentialType,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
}

impl SequentialPrimitive {
    pub fn new(
        sequential_type: SequentialType,
        input_ports: Vec<String>,
        output_ports: Vec<String>,
    ) -> Self {
        Self {
            sequential_type,
            input_ports,
            output_ports,
        }
    }

    pub fn name(&self) -> String {
        self.sequential_type.name()
    }
}
