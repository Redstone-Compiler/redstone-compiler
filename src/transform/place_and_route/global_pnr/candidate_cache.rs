use std::collections::HashSet;
use std::path::Path;

use eyre::WrapErr;
use serde::{Deserialize, Serialize};

use super::candidate_parity_hash;
use super::ir::{LayoutCandidate, PhysicalPort, PhysicalPortDirection, PortConnection};
use crate::nbt::{NBTRoot, ToNBT};
use crate::world::position::Position;
use crate::world::World3D;

const CACHE_FORMAT: &str = "redstone-compiler.local-candidate-cache.v1";

#[derive(Serialize, Deserialize)]
struct CacheManifest {
    format: String,
    candidates: Vec<CacheCandidate>,
}

#[derive(Serialize, Deserialize)]
struct CacheCandidate {
    id: String,
    nbt: String,
    metadata: String,
}

#[derive(Serialize, Deserialize)]
struct CandidateMetadata {
    ports: Vec<PhysicalPortDto>,
    blocked_cells: Vec<[usize; 3]>,
}

#[derive(Serialize, Deserialize)]
struct PhysicalPortDto {
    name: String,
    direction: PhysicalPortDirection,
    position: [usize; 3],
    route_position: Option<[usize; 3]>,
    access_points: Vec<[usize; 3]>,
    connection: PortConnection,
}

pub(super) fn load(
    root: &Path,
    key: &str,
    module_name: &str,
) -> eyre::Result<Option<Vec<LayoutCandidate>>> {
    let directory = root.join(key);
    let index = directory.join("index.json");
    if !index.is_file() {
        return Ok(None);
    }
    let manifest: CacheManifest = serde_json::from_slice(
        &std::fs::read(&index).wrap_err_with(|| format!("read {}", index.display()))?,
    )?;
    if manifest.format != CACHE_FORMAT {
        return Ok(None);
    }

    let mut candidates = Vec::with_capacity(manifest.candidates.len());
    for entry in manifest.candidates {
        let metadata: CandidateMetadata =
            serde_json::from_slice(&std::fs::read(directory.join(&entry.metadata))?)?;
        let nbt = NBTRoot::from_nbt_bytes(&std::fs::read(directory.join(&entry.nbt))?)?;
        let world = World3D::from(&nbt.to_world());
        let ports = metadata
            .ports
            .into_iter()
            .map(|port| PhysicalPort {
                name: port.name,
                direction: port.direction,
                position: array_position(port.position),
                route_position: port.route_position.map(array_position),
                access_points: port.access_points.into_iter().map(array_position).collect(),
                connection: port.connection,
            })
            .collect();
        let mut candidate = LayoutCandidate::from_world(module_name.to_owned(), world, ports)
            .context("cached candidate contains no blocks")?;
        candidate.blocked_cells = metadata
            .blocked_cells
            .into_iter()
            .map(array_position)
            .collect::<HashSet<_>>();
        if candidate_parity_hash(&candidate) != entry.id {
            eyre::bail!("cached candidate `{}` failed its content hash", entry.id);
        }
        candidates.push(candidate);
    }
    Ok(Some(candidates))
}

pub(super) fn store(root: &Path, key: &str, candidates: &[LayoutCandidate]) -> eyre::Result<()> {
    let directory = root.join(key);
    std::fs::create_dir_all(&directory)?;
    let mut entries = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let id = candidate_parity_hash(candidate);
        let nbt = format!("candidate-{index:03}.nbt");
        let metadata = format!("candidate-{index:03}.json");
        std::fs::write(
            directory.join(&nbt),
            fastnbt::to_bytes(&candidate.world.to_nbt())?,
        )?;
        let mut blocked_cells = candidate
            .blocked_cells
            .iter()
            .copied()
            .map(position_array)
            .collect::<Vec<_>>();
        blocked_cells.sort();
        let value = CandidateMetadata {
            ports: candidate
                .ports
                .iter()
                .map(|port| PhysicalPortDto {
                    name: port.name.clone(),
                    direction: port.direction.clone(),
                    position: position_array(port.position),
                    route_position: port.route_position.map(position_array),
                    access_points: port
                        .access_points
                        .iter()
                        .copied()
                        .map(position_array)
                        .collect(),
                    connection: port.connection,
                })
                .collect(),
            blocked_cells,
        };
        std::fs::write(
            directory.join(&metadata),
            serde_json::to_vec_pretty(&value)?,
        )?;
        entries.push(CacheCandidate { id, nbt, metadata });
    }
    let manifest = CacheManifest {
        format: CACHE_FORMAT.to_owned(),
        candidates: entries,
    };
    std::fs::write(
        directory.join("index.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

fn position_array(position: Position) -> [usize; 3] {
    [position.0, position.1, position.2]
}

fn array_position(position: [usize; 3]) -> Position {
    Position(position[0], position[1], position[2])
}

#[cfg(test)]
mod tests {
    use eyre::ContextCompat;

    use super::*;
    use crate::world::block::{Block, BlockKind};
    use crate::world::position::DimSize;

    #[test]
    fn persistent_candidate_cache_round_trips_and_relabels() -> eyre::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "redstone-candidate-cache-test-{}",
            std::process::id()
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root)?;
        }
        let mut world = World3D::new(DimSize(2, 2, 2));
        world[Position(0, 0, 0)] = Block {
            kind: BlockKind::RedstoneBlock,
            ..Default::default()
        };
        let mut candidate = LayoutCandidate::from_world(
            "first".to_owned(),
            world,
            vec![PhysicalPort {
                name: "q".to_owned(),
                direction: PhysicalPortDirection::Output,
                position: Position(0, 0, 0),
                route_position: Some(Position(0, 0, 0)),
                access_points: vec![Position(0, 0, 0)],
                connection: PortConnection::Direct,
            }],
        )?;
        candidate.blocked_cells.insert(Position(1, 0, 0));

        store(&root, "shape", &[candidate.clone()])?;
        let loaded = load(&root, "shape", "second")?.context("cache miss")?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].module_name, "second");
        assert_eq!(
            candidate_parity_hash(&loaded[0]),
            candidate_parity_hash(&candidate)
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
