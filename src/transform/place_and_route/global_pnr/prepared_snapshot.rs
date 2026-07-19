use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use eyre::{ContextCompat, WrapErr};
use serde::{Deserialize, Serialize};

use super::ir::{LayoutCandidate, PhysicalPort, PhysicalPortDirection, PortConnection};
use super::{
    candidate_parity_hash, debug_parity_hash, PnrPreparationSummary, PnrPrepareConfig,
    PreparedCandidateSet, PreparedInstanceCandidateBinding, PreparedPnrBody, PreparedPnrDesign,
};
use crate::ir::{RoutableDocument, RoutableModuleBody};
use crate::nbt::{NBTRoot, ToNBT};
use crate::snapshot::{emit_json, emit_nbt as snapshot_emit_nbt};
use crate::transform::place_and_route::global_pnr::topology::ResolvedPnrTopology;
use crate::transform::place_and_route::global_pnr::{
    apply_routable_document, GlobalPnrConfig, ResolvedPhysicalIntent,
};
use crate::world::position::Position;
use crate::world::World3D;

const CANDIDATE_LIBRARY_FORMAT: &str = "redstone-compiler.candidate-library.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateLibraryManifest {
    format: String,
    prepare_config_fingerprint: String,
    summary: PnrPreparationSummary,
    candidate_sets: Vec<CandidateSetManifest>,
    instance_bindings: Vec<InstanceBindingManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateSetManifest {
    index: usize,
    candidates: Vec<CandidateManifestEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateManifestEntry {
    id: String,
    nbt: String,
    metadata: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InstanceBindingManifest {
    instance: String,
    candidate_set_index: usize,
    preferred_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateMetadata {
    format: String,
    id: String,
    module_name: String,
    ports: Vec<PhysicalPortDto>,
    blocked_cells: Vec<[usize; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PhysicalPortDto {
    name: String,
    direction: PhysicalPortDirection,
    position: [usize; 3],
    route_position: Option<[usize; 3]>,
    access_points: Vec<[usize; 3]>,
    connection: PortConnection,
}

pub(super) fn emit_candidate_library(prepared: &PreparedPnrDesign) -> eyre::Result<()> {
    if !crate::snapshot::is_active() {
        return Ok(());
    }

    let (candidate_sets, instance_bindings) = match &prepared.body {
        PreparedPnrBody::Leaf { candidates } => (
            vec![PreparedCandidateSet {
                candidates: candidates.clone(),
            }],
            Vec::new(),
        ),
        PreparedPnrBody::Composite {
            candidate_sets,
            instance_bindings,
        } => (candidate_sets.clone(), instance_bindings.clone()),
    };

    let mut set_manifests = Vec::new();
    for (set_index, set) in candidate_sets.iter().enumerate() {
        let mut candidate_entries = Vec::new();
        for (candidate_index, candidate) in set.candidates.iter().enumerate() {
            let id = candidate_parity_hash(candidate);
            let directory = format!("candidates/set-{set_index:03}");
            let nbt = format!("{directory}/candidate-{candidate_index:03}.nbt");
            let metadata = format!("{directory}/candidate-{candidate_index:03}.json");
            snapshot_emit_nbt(&nbt, candidate.world.to_nbt())?;

            let mut blocked_cells = candidate
                .blocked_cells
                .iter()
                .copied()
                .map(position_array)
                .collect::<Vec<_>>();
            blocked_cells.sort();
            emit_json(
                &metadata,
                CandidateMetadata {
                    format: "redstone-compiler.candidate.v1".to_owned(),
                    id: id.clone(),
                    module_name: candidate.module_name.clone(),
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
                },
            )?;
            candidate_entries.push(CandidateManifestEntry { id, nbt, metadata });
        }
        set_manifests.push(CandidateSetManifest {
            index: set_index,
            candidates: candidate_entries,
        });
    }

    emit_json(
        "candidates/index.json",
        CandidateLibraryManifest {
            format: CANDIDATE_LIBRARY_FORMAT.to_owned(),
            prepare_config_fingerprint: prepare_config_fingerprint(
                &prepared.prepare_config,
                &prepared.topology,
            ),
            summary: prepared.summary.clone(),
            candidate_sets: set_manifests,
            instance_bindings: instance_bindings
                .into_iter()
                .map(|binding| InstanceBindingManifest {
                    instance: binding.instance_name,
                    candidate_set_index: binding.candidate_set_index,
                    preferred_index: binding.preferred_index,
                })
                .collect(),
        },
    )
}

pub fn load_prepared_pnr_snapshot(
    path: impl AsRef<Path>,
    config: &PnrPrepareConfig,
) -> eyre::Result<PreparedPnrDesign> {
    let mut source = SnapshotSource::open(path.as_ref())?;
    let routable_source = String::from_utf8(source.read("ir/routable.rcir")?)
        .wrap_err("snapshot Routable IR is not UTF-8")?;
    let document: RoutableDocument = routable_source.parse()?;
    let routable = &document.design;
    let mut effective_config = config.clone();
    let mut embedded = GlobalPnrConfig::default();
    apply_routable_document(&document, &mut embedded)?;
    effective_config.candidate = embedded.candidate;
    let snapshot_pnr = document
        .design_bindings
        .get(&document.design.top)
        .and_then(|name| document.design_profiles.get(name))
        .cloned();
    let topology = ResolvedPnrTopology::from_routable(&routable)?;
    let top = routable
        .module(&routable.top)
        .with_context(|| format!("missing top Routable module `{}`", routable.top))?;
    let module_name = top.name.clone();
    let is_leaf = matches!(top.body, RoutableModuleBody::Leaf { .. });
    let snapshot_intent = source
        .read_optional("intent/resolved.json")?
        .map(|bytes| serde_json::from_slice::<ResolvedPhysicalIntent>(&bytes))
        .transpose()?;
    if let Some(intent) = &snapshot_intent {
        intent.validate(&topology)?;
    }
    let manifest: CandidateLibraryManifest =
        serde_json::from_slice(&source.read("candidates/index.json")?)?;
    if manifest.format != CANDIDATE_LIBRARY_FORMAT {
        eyre::bail!("unsupported candidate library format `{}`", manifest.format);
    }
    let expected_fingerprint = prepare_config_fingerprint(&effective_config, &topology);
    if manifest.prepare_config_fingerprint != expected_fingerprint {
        eyre::bail!(
            "prepared candidate configuration mismatch; expected {}, snapshot contains {}",
            expected_fingerprint,
            manifest.prepare_config_fingerprint
        );
    }

    let mut candidate_sets = Vec::new();
    for (expected_index, set) in manifest.candidate_sets.iter().enumerate() {
        if set.index != expected_index {
            eyre::bail!("candidate set indices are not canonical");
        }
        let mut candidates = Vec::new();
        for entry in &set.candidates {
            let metadata: CandidateMetadata =
                serde_json::from_slice(&source.read(&entry.metadata)?)?;
            if metadata.id != entry.id {
                eyre::bail!("candidate metadata ID does not match its manifest entry");
            }
            let nbt = NBTRoot::from_nbt_bytes(&source.read(&entry.nbt)?)?;
            let imported = nbt.to_world();
            let world = World3D::from(&imported);
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
            let mut candidate = LayoutCandidate::from_world(metadata.module_name, world, ports)?;
            candidate.blocked_cells = metadata
                .blocked_cells
                .into_iter()
                .map(array_position)
                .collect::<HashSet<_>>();
            let actual_id = candidate_parity_hash(&candidate);
            if actual_id != entry.id {
                eyre::bail!(
                    "candidate content mismatch: expected {}, loaded {}",
                    entry.id,
                    actual_id
                );
            }
            candidates.push(candidate);
        }
        candidate_sets.push(PreparedCandidateSet { candidates });
    }

    let instance_bindings = manifest
        .instance_bindings
        .into_iter()
        .map(|binding| {
            if binding.candidate_set_index >= candidate_sets.len() {
                eyre::bail!(
                    "instance `{}` references missing candidate set {}",
                    binding.instance,
                    binding.candidate_set_index
                );
            }
            Ok(PreparedInstanceCandidateBinding {
                instance_name: binding.instance,
                candidate_set_index: binding.candidate_set_index,
                preferred_index: binding.preferred_index,
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    let body = if is_leaf {
        let candidates = candidate_sets
            .into_iter()
            .next()
            .context("leaf snapshot has no candidate set")?
            .candidates;
        PreparedPnrBody::Leaf { candidates }
    } else {
        PreparedPnrBody::Composite {
            candidate_sets,
            instance_bindings,
        }
    };
    Ok(PreparedPnrDesign {
        module_name,
        topology,
        prepare_config: effective_config,
        body,
        summary: manifest.summary,
        snapshot_intent,
        snapshot_pnr,
    })
}

pub(super) fn prepare_config_fingerprint(
    config: &PnrPrepareConfig,
    topology: &ResolvedPnrTopology,
) -> String {
    let mut canonical = normalized_candidate_policies(config, topology)
        .into_iter()
        .map(|(_, policy)| {
            let input_positions = policy
                .input_constraints
                .input_positions()
                .map(|(name, positions)| (name.to_owned(), positions.to_vec()))
                .collect::<std::collections::BTreeMap<_, _>>();
            format!(
                "{:?}|{:?}",
                super::rcir::candidate_spec_from_policy(&policy),
                input_positions
            )
        })
        .collect::<Vec<_>>();
    canonical.sort();
    canonical.dedup();
    debug_parity_hash(&canonical)
}

pub(super) fn normalized_candidate_policies(
    config: &PnrPrepareConfig,
    topology: &ResolvedPnrTopology,
) -> std::collections::BTreeMap<
    String,
    crate::transform::place_and_route::global_pnr::candidate::UnitCandidateConfig,
> {
    topology
        .definitions
        .iter()
        .filter(|definition| definition.is_leaf)
        .map(|definition| {
            let valid_inputs = definition
                .ports
                .iter()
                .filter_map(|port| topology.port(*port))
                .filter(|port| port.direction == crate::ir::RoutablePortDirection::Input)
                .map(|port| port.name.as_str())
                .collect::<HashSet<_>>();
            let mut policy = config
                .candidate
                .effective_for_definition(&definition.display_name);
            policy.input_constraints = policy
                .input_constraints
                .input_positions()
                .filter(|(name, _)| valid_inputs.contains(name))
                .fold(
                    crate::transform::place_and_route::local_placer::LocalPlacerInputConstraints::default(),
                    |constraints, (name, positions)| {
                        constraints.with_input_positions(name, positions.iter().copied())
                    },
                );
            (definition.display_name.clone(), policy)
        })
        .collect()
}

fn position_array(position: Position) -> [usize; 3] {
    [position.0, position.1, position.2]
}

fn array_position(position: [usize; 3]) -> Position {
    Position(position[0], position[1], position[2])
}

enum SnapshotSource {
    Directory(PathBuf),
    Archive(zip::ZipArchive<File>),
}

impl SnapshotSource {
    fn open(path: &Path) -> eyre::Result<Self> {
        if path.is_dir() {
            return Ok(Self::Directory(path.to_owned()));
        }
        Ok(Self::Archive(zip::ZipArchive::new(File::open(path)?)?))
    }

    fn read(&mut self, relative_path: &str) -> eyre::Result<Vec<u8>> {
        match self {
            Self::Directory(root) => Ok(std::fs::read(root.join(relative_path))?),
            Self::Archive(archive) => {
                let mut entry = archive
                    .by_name(relative_path)
                    .wrap_err_with(|| format!("snapshot archive is missing `{relative_path}`"))?;
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                Ok(bytes)
            }
        }
    }

    fn read_optional(&mut self, relative_path: &str) -> eyre::Result<Option<Vec<u8>>> {
        match self {
            Self::Directory(root) => {
                let path = root.join(relative_path);
                if !path.is_file() {
                    return Ok(None);
                }
                Ok(Some(std::fs::read(path)?))
            }
            Self::Archive(archive) => match archive.by_name(relative_path) {
                Ok(mut entry) => {
                    let mut bytes = Vec::new();
                    entry.read_to_end(&mut bytes)?;
                    Ok(Some(bytes))
                }
                Err(zip::result::ZipError::FileNotFound) => Ok(None),
                Err(error) => Err(error.into()),
            },
        }
    }
}
