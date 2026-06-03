use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::world::position::Position;
use crate::world::World3D;

const FORMAT: &str = "redstone-compiler.outputs.v1";

#[derive(Debug, Clone)]
pub struct PlacedWorld {
    pub world: World3D,
    pub inputs: Vec<OutputEndpoint>,
    pub outputs: Vec<OutputEndpoint>,
}

impl PlacedWorld {
    pub fn metadata(&self) -> OutputMetadata {
        OutputMetadata::new(self.outputs.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub format: String,
    pub outputs: Vec<OutputEndpoint>,
}

impl OutputMetadata {
    pub fn new(outputs: Vec<OutputEndpoint>) -> Self {
        Self {
            format: FORMAT.to_owned(),
            outputs,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> eyre::Result<Self> {
        let metadata = serde_json::from_str(&fs::read_to_string(path)?)?;
        Ok(metadata)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> eyre::Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputEndpoint {
    pub name: String,
    pub position: [usize; 3],
}

impl OutputEndpoint {
    pub fn new(name: String, position: Position) -> Self {
        Self {
            name,
            position: [position.0, position.1, position.2],
        }
    }

    pub fn position(&self) -> Position {
        Position(self.position[0], self.position[1], self.position[2])
    }
}
