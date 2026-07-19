use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::nbt::ToNBT;
use crate::snapshot::{emit_json, emit_nbt, SnapshotProduct, SnapshotProductInfo};
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

impl SnapshotProduct for PlacedWorld {
    fn emit_snapshot(&self, design_name: &str) -> eyre::Result<SnapshotProductInfo> {
        let nbt_name = format!("{}.nbt", safe_artifact_name(design_name));
        emit_nbt(&nbt_name, self.world.to_nbt())?;
        emit_json(
            "interface.json",
            json!({
                "format": "redstone-compiler.interface.v1",
                "inputs": self.inputs,
                "outputs": self.outputs,
            }),
        )?;
        Ok(SnapshotProductInfo {
            top_module: design_name.to_owned(),
            final_nbt: nbt_name,
            summary: json!({
                "world": {
                    "size": [self.world.size.0, self.world.size.1, self.world.size.2],
                    "non_air_blocks": self.world.iter_block().len(),
                },
                "interface": {
                    "inputs": self.inputs.len(),
                    "outputs": self.outputs.len(),
                }
            }),
        })
    }
}

fn safe_artifact_name(name: &str) -> String {
    let safe = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    if safe.is_empty() {
        "design".to_owned()
    } else {
        safe
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
