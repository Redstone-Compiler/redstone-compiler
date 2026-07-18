use std::cell::Cell;
use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use serde::Serialize;
use serde_json::{json, Value};

use crate::nbt::NBTRoot;

const SNAPSHOT_FORMAT: &str = "redstone-compiler.snapshot.v1";

type RunId = u64;

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);
static SNAPSHOT_HUB: OnceLock<Mutex<HashMap<RunId, Sender<WriterMessage>>>> = OnceLock::new();

thread_local! {
    static CURRENT_RUN: Cell<Option<RunId>> = const { Cell::new(None) };
}

#[derive(Clone, Debug)]
pub struct SnapshotOptions {
    pub output_dir: PathBuf,
    pub design_name: String,
    pub source_path: Option<PathBuf>,
    pub source_text: Option<(String, String)>,
}

impl SnapshotOptions {
    pub fn new(output_dir: impl Into<PathBuf>, design_name: impl Into<String>) -> Self {
        Self {
            output_dir: output_dir.into(),
            design_name: design_name.into(),
            source_path: None,
            source_text: None,
        }
    }

    pub fn with_source(mut self, source_path: impl Into<PathBuf>) -> Self {
        self.source_path = Some(source_path.into());
        self
    }

    pub fn with_source_text(
        mut self,
        file_name: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        self.source_text = Some((file_name.into(), source.into()));
        self
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SnapshotEvent {
    Stage {
        module: String,
        step: usize,
        total: usize,
        name: String,
    },
    CandidateSummary {
        instances: usize,
        unique: usize,
        reused: usize,
        candidates: usize,
        elapsed_ms: u64,
    },
    RoutingProgress {
        attempt: usize,
        total_attempts: usize,
        best_routes: usize,
        routing_failures: usize,
        contract_failures: usize,
        elapsed_ms: u64,
    },
    RoutingSummary {
        selected_attempt: usize,
        routes: usize,
        routing_failures: usize,
        assembly_failures: usize,
        contract_failures: usize,
        verifier_failures: usize,
        elapsed_ms: u64,
    },
    PlacementCost {
        volume: usize,
        xy_footprint: usize,
        height: usize,
        wire_length: usize,
        vertical_distance: usize,
        congestion: usize,
        weighted_total: usize,
    },
}

#[derive(Clone, Debug)]
pub struct SnapshotProductInfo {
    pub top_module: String,
    pub final_nbt: String,
    pub summary: Value,
}

pub trait SnapshotProduct {
    fn emit_snapshot(&self, design_name: &str) -> eyre::Result<SnapshotProductInfo>;
}

pub fn compile_with_snapshot<T>(
    options: SnapshotOptions,
    compile: impl FnOnce() -> eyre::Result<T>,
) -> eyre::Result<T>
where
    T: SnapshotProduct,
{
    let session = SnapshotSession::start(&options.output_dir)?;
    let _scope = SnapshotScope::enter(session.run_id);
    let started = Instant::now();

    if let Some(source_path) = options.source_path.as_deref() {
        let file_name = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("source.v");
        emit_bytes(PathBuf::from(file_name), fs::read(source_path)?)?;
    }
    if let Some((file_name, source)) = options.source_text {
        emit_bytes(PathBuf::from(file_name), source.into_bytes())?;
    }

    match compile() {
        Ok(product) => match product.emit_snapshot(&options.design_name) {
            Ok(info) => {
                session.finish(FinishMessage {
                    status: "success",
                    top_module: Some(info.top_module),
                    final_nbt: Some(info.final_nbt),
                    elapsed_ms: duration_ms(started.elapsed()),
                    error: None,
                    product: info.summary,
                })?;
                Ok(product)
            }
            Err(error) => {
                let message = format!("failed to export compilation snapshot: {error:#}");
                session.finish(FinishMessage {
                    status: "failed",
                    top_module: None,
                    final_nbt: None,
                    elapsed_ms: duration_ms(started.elapsed()),
                    error: Some(message.clone()),
                    product: Value::Null,
                })?;
                Err(eyre::eyre!(message))
            }
        },
        Err(error) => {
            let message = format!("{error:#}");
            session.finish(FinishMessage {
                status: "failed",
                top_module: None,
                final_nbt: None,
                elapsed_ms: duration_ms(started.elapsed()),
                error: Some(message),
                product: Value::Null,
            })?;
            Err(error)
        }
    }
}

pub fn record(event: SnapshotEvent) {
    let _ = send(WriterMessage::Event(event));
}

pub fn emit_nbt(path: impl Into<PathBuf>, nbt: NBTRoot) -> eyre::Result<()> {
    let path = checked_relative_path(path.into())?;
    send(WriterMessage::Nbt { path, nbt })
}

pub fn emit_json(path: impl Into<PathBuf>, value: impl Serialize) -> eyre::Result<()> {
    let path = checked_relative_path(path.into())?;
    send(WriterMessage::Json {
        path,
        value: serde_json::to_value(value)?,
    })
}

pub fn is_active() -> bool {
    CURRENT_RUN.get().is_some()
}

#[derive(Clone, Copy, Debug)]
pub struct SnapshotToken {
    run_id: Option<RunId>,
}

impl SnapshotToken {
    pub fn in_scope<T>(self, operation: impl FnOnce() -> T) -> T {
        let Some(run_id) = self.run_id else {
            return operation();
        };
        let _scope = SnapshotScope::enter(run_id);
        operation()
    }
}

pub fn capture() -> SnapshotToken {
    SnapshotToken {
        run_id: CURRENT_RUN.get(),
    }
}

fn emit_bytes(path: impl Into<PathBuf>, bytes: Vec<u8>) -> eyre::Result<()> {
    let path = checked_relative_path(path.into())?;
    send(WriterMessage::Bytes { path, bytes })
}

fn send(message: WriterMessage) -> eyre::Result<()> {
    let Some(run_id) = CURRENT_RUN.get() else {
        return Ok(());
    };
    let sender = hub()
        .lock()
        .map_err(|_| eyre::eyre!("snapshot hub lock is poisoned"))?
        .get(&run_id)
        .cloned();
    let Some(sender) = sender else {
        return Ok(());
    };
    sender
        .send(message)
        .map_err(|_| eyre::eyre!("snapshot writer stopped unexpectedly"))
}

fn hub() -> &'static Mutex<HashMap<RunId, Sender<WriterMessage>>> {
    SNAPSHOT_HUB.get_or_init(|| Mutex::new(HashMap::new()))
}

struct SnapshotScope {
    previous: Option<RunId>,
}

impl SnapshotScope {
    fn enter(run_id: RunId) -> Self {
        let previous = CURRENT_RUN.replace(Some(run_id));
        Self { previous }
    }
}

impl Drop for SnapshotScope {
    fn drop(&mut self) {
        CURRENT_RUN.set(self.previous);
    }
}

struct SnapshotSession {
    run_id: RunId,
    sender: Option<Sender<WriterMessage>>,
    writer: Option<JoinHandle<eyre::Result<()>>>,
}

impl SnapshotSession {
    fn start(output_dir: &Path) -> eyre::Result<Self> {
        fs::create_dir_all(output_dir)?;
        for generated_index in ["manifest.json", "summary.json"] {
            let path = output_dir.join(generated_index);
            if path.is_file() {
                fs::remove_file(path)?;
            }
        }
        let run_id = NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::channel();
        hub()
            .lock()
            .map_err(|_| eyre::eyre!("snapshot hub lock is poisoned"))?
            .insert(run_id, sender.clone());
        let output_dir = output_dir.to_owned();
        let writer = thread::Builder::new()
            .name(format!("snapshot-writer-{run_id}"))
            .spawn(move || writer_loop(&output_dir, receiver))?;
        Ok(Self {
            run_id,
            sender: Some(sender),
            writer: Some(writer),
        })
    }

    fn finish(mut self, finish: FinishMessage) -> eyre::Result<()> {
        self.finalize(finish)
    }

    fn finalize(&mut self, finish: FinishMessage) -> eyre::Result<()> {
        hub()
            .lock()
            .map_err(|_| eyre::eyre!("snapshot hub lock is poisoned"))?
            .remove(&self.run_id);
        self.sender
            .take()
            .ok_or_else(|| eyre::eyre!("snapshot session was already finalized"))?
            .send(WriterMessage::Finish(finish))
            .map_err(|_| eyre::eyre!("snapshot writer stopped before finalization"))?;
        self.writer
            .take()
            .ok_or_else(|| eyre::eyre!("snapshot writer was already joined"))?
            .join()
            .map_err(|_| eyre::eyre!("snapshot writer panicked"))?
    }
}

impl Drop for SnapshotSession {
    fn drop(&mut self) {
        if self.writer.is_none() {
            return;
        }
        let _ = self.finalize(FinishMessage {
            status: "aborted",
            top_module: None,
            final_nbt: None,
            elapsed_ms: 0,
            error: Some("snapshot scope exited before normal finalization".to_owned()),
            product: Value::Null,
        });
    }
}

enum WriterMessage {
    Event(SnapshotEvent),
    Nbt { path: PathBuf, nbt: NBTRoot },
    Json { path: PathBuf, value: Value },
    Bytes { path: PathBuf, bytes: Vec<u8> },
    Finish(FinishMessage),
}

struct FinishMessage {
    status: &'static str,
    top_module: Option<String>,
    final_nbt: Option<String>,
    elapsed_ms: u64,
    error: Option<String>,
    product: Value,
}

#[derive(Serialize)]
struct ArtifactEntry {
    path: String,
    kind: &'static str,
}

fn writer_loop(output_dir: &Path, receiver: Receiver<WriterMessage>) -> eyre::Result<()> {
    let mut events = Vec::new();
    let mut artifacts = Vec::new();
    while let Ok(message) = receiver.recv() {
        match message {
            WriterMessage::Event(event) => events.push(event),
            WriterMessage::Nbt { path, nbt } => {
                write_file(output_dir, &path, &nbt.to_gzip_bytes()?)?;
                artifacts.push(artifact_entry(&path, "nbt"));
            }
            WriterMessage::Json { path, value } => {
                write_file(output_dir, &path, &serde_json::to_vec_pretty(&value)?)?;
                artifacts.push(artifact_entry(&path, "json"));
            }
            WriterMessage::Bytes { path, bytes } => {
                write_file(output_dir, &path, &bytes)?;
                artifacts.push(artifact_entry(&path, "source"));
            }
            WriterMessage::Finish(finish) => {
                let summary = json!({
                    "format": SNAPSHOT_FORMAT,
                    "status": finish.status,
                    "top_module": finish.top_module,
                    "final_nbt": finish.final_nbt,
                    "elapsed_ms": finish.elapsed_ms,
                    "error": finish.error,
                    "result": finish.product,
                    "events": events,
                });
                let summary_path = PathBuf::from("summary.json");
                write_file(
                    output_dir,
                    &summary_path,
                    &serde_json::to_vec_pretty(&summary)?,
                )?;
                artifacts.push(artifact_entry(&summary_path, "summary"));
                let manifest = json!({
                    "format": SNAPSHOT_FORMAT,
                    "status": finish.status,
                    "top_module": finish.top_module,
                    "final_nbt": finish.final_nbt,
                    "artifacts": artifacts,
                });
                write_file(
                    output_dir,
                    Path::new("manifest.json"),
                    &serde_json::to_vec_pretty(&manifest)?,
                )?;
                return Ok(());
            }
        }
    }
    Err(eyre::eyre!("snapshot session ended without finalization"))
}

fn write_file(output_dir: &Path, relative: &Path, bytes: &[u8]) -> eyre::Result<()> {
    let path = output_dir.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn artifact_entry(path: &Path, kind: &'static str) -> ArtifactEntry {
    ArtifactEntry {
        path: path.to_string_lossy().replace('\\', "/"),
        kind,
    }
}

fn checked_relative_path(path: PathBuf) -> eyre::Result<PathBuf> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        eyre::bail!("snapshot artifact path must be a safe relative path: {path:?}");
    }
    Ok(path)
}

pub fn duration_ms(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyProduct;

    impl SnapshotProduct for DummyProduct {
        fn emit_snapshot(&self, design_name: &str) -> eyre::Result<SnapshotProductInfo> {
            emit_json("product.json", json!({ "name": design_name }))?;
            Ok(SnapshotProductInfo {
                top_module: design_name.to_owned(),
                final_nbt: "dummy.nbt".to_owned(),
                summary: json!({ "dummy": true }),
            })
        }
    }

    #[test]
    fn rejects_artifact_paths_outside_snapshot() {
        assert!(checked_relative_path(PathBuf::from("../outside.json")).is_err());
        assert!(checked_relative_path(PathBuf::from("inside/data.json")).is_ok());
    }

    #[test]
    fn compile_scope_writes_events_product_and_manifest() -> eyre::Result<()> {
        let output = unique_test_directory("success");
        let product = compile_with_snapshot(
            SnapshotOptions::new(&output, "dummy"),
            || -> eyre::Result<DummyProduct> {
                record(SnapshotEvent::Stage {
                    module: "dummy".to_owned(),
                    step: 1,
                    total: 1,
                    name: "compile".to_owned(),
                });
                Ok(DummyProduct)
            },
        )?;
        let _ = product;

        let summary: Value = serde_json::from_slice(&fs::read(output.join("summary.json"))?)?;
        assert_eq!(summary["status"], "success");
        assert_eq!(summary["events"].as_array().map(Vec::len), Some(1));
        assert!(output.join("manifest.json").is_file());
        assert!(output.join("product.json").is_file());
        fs::remove_dir_all(output)?;
        Ok(())
    }

    #[test]
    fn failed_compile_still_finalizes_snapshot() -> eyre::Result<()> {
        let output = unique_test_directory("failure");
        let result = compile_with_snapshot(
            SnapshotOptions::new(&output, "dummy"),
            || -> eyre::Result<DummyProduct> { eyre::bail!("expected failure") },
        );
        assert!(result.is_err());

        let summary: Value = serde_json::from_slice(&fs::read(output.join("summary.json"))?)?;
        assert_eq!(summary["status"], "failed");
        assert!(summary["error"]
            .as_str()
            .is_some_and(|error| error.contains("expected failure")));
        fs::remove_dir_all(output)?;
        Ok(())
    }

    fn unique_test_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "redstone-compiler-snapshot-{label}-{}-{}",
            std::process::id(),
            NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }
}
