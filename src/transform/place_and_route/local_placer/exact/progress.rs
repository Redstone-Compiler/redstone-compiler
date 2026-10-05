//! Frames of construction and compaction, for watching a layout being built.
//!
//! Inside a compilation snapshot every accepted step is always recorded as a
//! frame of the placer's sequence (`snapshot::emit_frame`), so the `.rsnap`
//! plays the run back in the viewer. `ConstructionConfig::progress` and
//! `CompactionConfig::progress` also write the frames to a plain directory:
//! `NNN-<label>.nbt` (the settled world), `NNN-<label>.outputs.json` (its
//! inputs and outputs), and `frames.json` (`?frames=<path to frames.json>`).

use std::path::Path;

use serde_json::{json, Value};

use super::ExactPlacement;

/// Records one frame of `sequence` into the active snapshot, if any, and into
/// `directory`, if given; a failure is logged and otherwise ignored, so
/// watching never breaks a run.
pub(super) fn record_frame(
    directory: Option<&Path>,
    sequence: &str,
    label: &str,
    placement: &ExactPlacement,
) {
    if crate::snapshot::is_active() {
        let size = placement.rcell.size;
        crate::snapshot::emit_frame(
            sequence,
            crate::snapshot::SnapshotFrame {
                label: label.to_owned(),
                nbt: crate::nbt::NBTRoot::from(&placement.placed.world),
                interface: Some(placement.rcell.interface_json()),
                details: json!({
                    "blocks": placement.block_count,
                    "size": [size.0, size.1, size.2],
                }),
            },
        );
    }
    if let Some(directory) = directory {
        if let Err(error) = try_record_frame(directory, label, placement) {
            tracing::warn!(%error, directory = %directory.display(), "could not record a frame");
        }
    }
}

fn try_record_frame(directory: &Path, label: &str, placement: &ExactPlacement) -> eyre::Result<()> {
    std::fs::create_dir_all(directory)?;
    let index_path = directory.join("frames.json");
    let mut frames = match std::fs::read_to_string(&index_path) {
        Ok(text) => serde_json::from_str::<Vec<Value>>(&text)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let slug = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let stem = format!("{:03}-{slug}", frames.len());
    crate::nbt::NBTRoot::from(&placement.placed.world).save(directory.join(format!("{stem}.nbt")));
    std::fs::write(
        directory.join(format!("{stem}.outputs.json")),
        serde_json::to_string_pretty(&placement.rcell.interface_json())?,
    )?;
    let size = placement.rcell.size;
    frames.push(json!({
        "file": format!("{stem}.nbt"),
        "outputs": format!("{stem}.outputs.json"),
        "label": label,
        "blocks": placement.block_count,
        "size": [size.0, size.1, size.2],
    }));
    std::fs::write(index_path, serde_json::to_string_pretty(&frames)?)?;
    Ok(())
}
