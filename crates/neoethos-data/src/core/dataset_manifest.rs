//! Verified immutable Vortex generation publication.
//!
//! `data.vortex.complete` is a small versioned pointer/manifest. Data files are
//! content-addressed and never replaced in place, so every crash leaves either
//! the previous verified generation or the fully verified new generation.

use crate::core::dataset_candidate_lease::DatasetCandidateLease;
use crate::core::dataset_generation_lease::{
    DatasetGenerationLease, remove_lock_file, try_acquire_exclusive,
};
use crate::core::vortex_io::{
    VortexWriteStats, atomic_replace_file, read_vortex_i64_projection_range, read_vortex_row_count,
    sync_parent_directory, write_vortex_chunks,
};
use anyhow::{Context, Result, bail};
use neoethos_dataset_contracts::{BarTimestampConvention, CanonicalDatasetIdentity};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::Error as _};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use vortex_array::ArrayRef;

const MANIFEST_SCHEMA: &str = "neoethos.dataset-manifest.v1";
const MANIFEST_VERSION: u16 = 1;
const MANIFEST_FILE: &str = "data.vortex.complete";
const BINDING_DOMAIN: &[u8] = b"neoethos.dataset-manifest-binding.v1\0";
const SELECTED_GENERATION_SCHEMA: &str = "neoethos.selected-dataset-generation.v1";
const SELECTED_GENERATION_VERSION: u16 = 1;
const DATASET_SERIES_RECEIPT_SCHEMA: &str = "neoethos.canonical-dataset-series-receipt.v1";
const DATASET_SERIES_RECEIPT_VERSION: u16 = 1;
static PUBLICATION_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerProvenanceEnvelopeV1 {
    schema_id: String,
    canonical_payload: Vec<u8>,
    payload_sha256: String,
}

impl ProducerProvenanceEnvelopeV1 {
    pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024;
    pub const MAX_SCHEMA_ID_BYTES: usize = 96;

    pub fn new(schema_id: impl Into<String>, canonical_payload: Vec<u8>) -> Result<Self> {
        let envelope = Self {
            schema_id: schema_id.into(),
            payload_sha256: sha256_bytes(&canonical_payload),
            canonical_payload,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    pub fn canonical_payload(&self) -> &[u8] {
        &self.canonical_payload
    }

    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema_id(&self.schema_id)?;
        if self.canonical_payload.len() > Self::MAX_PAYLOAD_BYTES {
            bail!(
                "producer provenance payload is {} bytes; maximum is {}",
                self.canonical_payload.len(),
                Self::MAX_PAYLOAD_BYTES
            );
        }
        validate_sha256_hex("producer payload", &self.payload_sha256)?;
        let actual = sha256_bytes(&self.canonical_payload);
        if actual != self.payload_sha256 {
            bail!(
                "producer provenance payload hash mismatch: expected {}, got {}",
                self.payload_sha256,
                actual
            );
        }
        Ok(())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).context("serialize producer provenance envelope")
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let envelope: Self =
            serde_json::from_slice(bytes).context("decode producer provenance envelope")?;
        envelope.validate()?;
        Ok(envelope)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetTimestampRange {
    start_ms: i64,
    end_ms: i64,
}

impl DatasetTimestampRange {
    pub fn new(start_ms: i64, end_ms: i64) -> Result<Self> {
        if start_ms > end_ms {
            bail!("dataset timestamp range is descending: {start_ms}..{end_ms}");
        }
        Ok(Self { start_ms, end_ms })
    }

    pub const fn start_ms(self) -> i64 {
        self.start_ms
    }

    pub const fn end_ms(self) -> i64 {
        self.end_ms
    }
}

pub struct PublishRequest<'a> {
    pub configured_root: &'a Path,
    pub identity: &'a CanonicalDatasetIdentity,
    pub expected_generation: Option<&'a str>,
    pub timestamp_range: DatasetTimestampRange,
    pub provenance: &'a ProducerProvenanceEnvelopeV1,
    pub chunks: Vec<ArrayRef>,
}

pub struct PublishMetadataRequest<'a> {
    pub configured_root: &'a Path,
    pub identity: &'a CanonicalDatasetIdentity,
    pub expected_generation: Option<&'a str>,
    pub provenance: &'a ProducerProvenanceEnvelopeV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateWriteOutcome {
    pub write_stats: VortexWriteStats,
    pub timestamp_range: DatasetTimestampRange,
}

#[derive(Clone, Debug)]
pub struct PublishResult {
    manifest: DatasetManifestV1,
    previous_generation: Option<String>,
    durable_commit_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicationConflict {
    expected_generation: Option<String>,
    current_generation: Option<String>,
}

impl PublicationConflict {
    pub fn expected_generation(&self) -> Option<&str> {
        self.expected_generation.as_deref()
    }

    pub fn current_generation(&self) -> Option<&str> {
        self.current_generation.as_deref()
    }
}

impl fmt::Display for PublicationConflict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "generation conflict: expected {:?}, current is {:?}",
            self.expected_generation, self.current_generation
        )
    }
}

impl std::error::Error for PublicationConflict {}

impl PublishResult {
    pub fn generation(&self) -> &str {
        self.manifest.generation_id()
    }

    pub fn previous_generation(&self) -> Option<&str> {
        self.previous_generation.as_deref()
    }

    pub fn durable_commit_id(&self) -> &str {
        &self.durable_commit_id
    }

    pub const fn row_count(&self) -> u64 {
        self.manifest.row_count()
    }

    pub const fn manifest(&self) -> &DatasetManifestV1 {
        &self.manifest
    }
}

#[derive(Clone, Debug)]
pub struct DatasetManifestV1 {
    wire: ManifestWireV1,
    dataset_root: PathBuf,
    identity: CanonicalDatasetIdentity,
}

impl DatasetManifestV1 {
    pub const fn schema_id(&self) -> &'static str {
        MANIFEST_SCHEMA
    }

    pub fn generation_id(&self) -> &str {
        &self.wire.generation_id
    }

    pub const fn row_count(&self) -> u64 {
        self.wire.row_count
    }

    pub fn vortex_sha256(&self) -> &str {
        &self.wire.vortex_sha256
    }

    pub fn generation_path(&self) -> PathBuf {
        self.dataset_root.join(&self.wire.generation_id)
    }

    pub fn identity(&self) -> &CanonicalDatasetIdentity {
        &self.identity
    }

    pub fn timestamp_range(&self) -> DatasetTimestampRange {
        self.wire.timestamp_range
    }

    pub fn provenance(&self) -> &ProducerProvenanceEnvelopeV1 {
        &self.wire.producer_provenance
    }

    /// Canonical hash that binds every manifest field except the hash field
    /// itself. This is the stable manifest identity carried into downstream
    /// feature-artifact provenance.
    pub fn manifest_binding_sha256(&self) -> &str {
        &self.wire.manifest_binding_sha256
    }
}

/// Exact immutable dataset generation selected at an API or scheduling
/// boundary.
///
/// All fields are private and every construction/deserialization path validates
/// the generation component and canonical lowercase manifest hash. This value
/// is a receipt, not a request to load whatever generation happens to be
/// current later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedDatasetGenerationV1 {
    identity: CanonicalDatasetIdentity,
    generation_id: String,
    manifest_binding_sha256: String,
}

impl SelectedDatasetGenerationV1 {
    pub fn new(
        identity: CanonicalDatasetIdentity,
        generation_id: impl Into<String>,
        manifest_binding_sha256: impl Into<String>,
    ) -> Result<Self> {
        let receipt = Self {
            identity,
            generation_id: generation_id.into(),
            manifest_binding_sha256: manifest_binding_sha256.into(),
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn from_manifest(manifest: &DatasetManifestV1) -> Result<Self> {
        Self::new(
            manifest.identity().clone(),
            manifest.generation_id(),
            manifest.manifest_binding_sha256(),
        )
    }

    pub const fn identity(&self) -> &CanonicalDatasetIdentity {
        &self.identity
    }

    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    pub fn manifest_binding_sha256(&self) -> &str {
        &self.manifest_binding_sha256
    }

    pub fn validate(&self) -> Result<()> {
        let identity_path = self.identity.to_path_component();
        let decoded = CanonicalDatasetIdentity::from_path_component(&identity_path)
            .context("validate selected dataset identity")?;
        if decoded != self.identity {
            bail!("selected dataset identity does not round-trip exactly");
        }
        validate_generation_id(&self.generation_id)?;
        validate_sha256_hex("selected manifest binding", &self.manifest_binding_sha256)?;
        Ok(())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).context("serialize selected dataset generation receipt")
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("decode selected dataset generation receipt")
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectedDatasetGenerationWireV1 {
    schema: String,
    version: u16,
    dataset_identity: String,
    generation_id: String,
    manifest_binding_sha256: String,
}

impl Serialize for SelectedDatasetGenerationV1 {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(S::Error::custom)?;
        SelectedDatasetGenerationWireV1 {
            schema: SELECTED_GENERATION_SCHEMA.to_owned(),
            version: SELECTED_GENERATION_VERSION,
            dataset_identity: self.identity.to_path_component(),
            generation_id: self.generation_id.clone(),
            manifest_binding_sha256: self.manifest_binding_sha256.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SelectedDatasetGenerationV1 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SelectedDatasetGenerationWireV1::deserialize(deserializer)?;
        if wire.schema != SELECTED_GENERATION_SCHEMA || wire.version != SELECTED_GENERATION_VERSION
        {
            return Err(D::Error::custom(format!(
                "unsupported selected dataset generation schema/version {:?}/{}",
                wire.schema, wire.version
            )));
        }
        let identity = CanonicalDatasetIdentity::from_path_component(&wire.dataset_identity)
            .map_err(D::Error::custom)?;
        Self::new(identity, wire.generation_id, wire.manifest_binding_sha256)
            .map_err(D::Error::custom)
    }
}

/// Canonically ordered set of independently persisted timeframe generations
/// that belong to one exact source/account/symbol series.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalDatasetSeriesReceiptV1 {
    anchor: SelectedDatasetGenerationV1,
    direct_timeframes: Vec<SelectedDatasetGenerationV1>,
}

impl CanonicalDatasetSeriesReceiptV1 {
    pub fn new(
        anchor: SelectedDatasetGenerationV1,
        mut direct_timeframes: Vec<SelectedDatasetGenerationV1>,
    ) -> Result<Self> {
        direct_timeframes.sort_by_key(|receipt| receipt.identity().timeframe());
        let receipt = Self {
            anchor,
            direct_timeframes,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub const fn anchor(&self) -> &SelectedDatasetGenerationV1 {
        &self.anchor
    }

    pub fn direct_timeframes(&self) -> &[SelectedDatasetGenerationV1] {
        &self.direct_timeframes
    }

    pub fn validate(&self) -> Result<()> {
        self.anchor.validate()?;
        if self.direct_timeframes.is_empty() {
            bail!("canonical dataset series receipt has no direct timeframe generations");
        }

        let anchor_identity = self.anchor.identity();
        let mut seen = HashSet::with_capacity(self.direct_timeframes.len());
        let mut previous = None;
        let mut contains_exact_anchor = false;
        for direct in &self.direct_timeframes {
            direct.validate()?;
            let identity = direct.identity();
            if identity.scope() != anchor_identity.scope()
                || identity.symbol_name() != anchor_identity.symbol_name()
                || identity.bar_timestamp_convention() != anchor_identity.bar_timestamp_convention()
            {
                bail!(
                    "direct timeframe {} belongs to a different source/account series than {}",
                    identity.timeframe(),
                    anchor_identity.timeframe()
                );
            }
            if !seen.insert(identity.timeframe()) {
                bail!(
                    "canonical dataset series receipt repeats direct timeframe {}",
                    identity.timeframe()
                );
            }
            if previous.is_some_and(|timeframe| timeframe >= identity.timeframe()) {
                bail!("canonical dataset series receipt is not in timeframe order");
            }
            previous = Some(identity.timeframe());
            if identity == anchor_identity {
                if direct != &self.anchor {
                    bail!(
                        "anchor direct timeframe generation/binding differs from the selected anchor receipt"
                    );
                }
                contains_exact_anchor = true;
            }
        }
        if !contains_exact_anchor {
            bail!("canonical dataset series receipt does not contain its exact anchor generation");
        }
        Ok(())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).context("serialize canonical dataset series receipt")
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("decode canonical dataset series receipt")
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalDatasetSeriesReceiptWireV1 {
    schema: String,
    version: u16,
    anchor: SelectedDatasetGenerationV1,
    direct_timeframes: Vec<SelectedDatasetGenerationV1>,
}

impl Serialize for CanonicalDatasetSeriesReceiptV1 {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(S::Error::custom)?;
        CanonicalDatasetSeriesReceiptWireV1 {
            schema: DATASET_SERIES_RECEIPT_SCHEMA.to_owned(),
            version: DATASET_SERIES_RECEIPT_VERSION,
            anchor: self.anchor.clone(),
            direct_timeframes: self.direct_timeframes.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CanonicalDatasetSeriesReceiptV1 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CanonicalDatasetSeriesReceiptWireV1::deserialize(deserializer)?;
        if wire.schema != DATASET_SERIES_RECEIPT_SCHEMA
            || wire.version != DATASET_SERIES_RECEIPT_VERSION
        {
            return Err(D::Error::custom(format!(
                "unsupported canonical dataset series receipt schema/version {:?}/{}",
                wire.schema, wire.version
            )));
        }
        let original_order = wire.direct_timeframes.clone();
        let receipt = Self::new(wire.anchor, wire.direct_timeframes).map_err(D::Error::custom)?;
        if receipt.direct_timeframes != original_order {
            return Err(D::Error::custom(
                "canonical dataset series receipt direct timeframes are not canonically ordered",
            ));
        }
        Ok(receipt)
    }
}

/// Typed fail-closed signal that an exact generation receipt no longer equals
/// the verified current manifest. No caller may reinterpret this as permission
/// to load the new current generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactDatasetGenerationConflict {
    expected_generation: String,
    expected_manifest_binding_sha256: String,
    current_generation: Option<String>,
    current_manifest_binding_sha256: Option<String>,
}

impl ExactDatasetGenerationConflict {
    fn new(expected: &SelectedDatasetGenerationV1, current: Option<&DatasetManifestV1>) -> Self {
        Self {
            expected_generation: expected.generation_id().to_owned(),
            expected_manifest_binding_sha256: expected.manifest_binding_sha256().to_owned(),
            current_generation: current.map(|manifest| manifest.generation_id().to_owned()),
            current_manifest_binding_sha256: current
                .map(|manifest| manifest.manifest_binding_sha256().to_owned()),
        }
    }

    pub fn expected_generation(&self) -> &str {
        &self.expected_generation
    }

    pub fn expected_manifest_binding_sha256(&self) -> &str {
        &self.expected_manifest_binding_sha256
    }

    pub fn current_generation(&self) -> Option<&str> {
        self.current_generation.as_deref()
    }

    pub fn current_manifest_binding_sha256(&self) -> Option<&str> {
        self.current_manifest_binding_sha256.as_deref()
    }
}

impl fmt::Display for ExactDatasetGenerationConflict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "stale exact dataset generation receipt: expected generation {} / manifest {}, current is {:?} / {:?}",
            self.expected_generation,
            self.expected_manifest_binding_sha256,
            self.current_generation,
            self.current_manifest_binding_sha256
        )
    }
}

impl std::error::Error for ExactDatasetGenerationConflict {}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWireV1 {
    schema: String,
    version: u16,
    generation_id: String,
    previous_generation: Option<String>,
    durable_commit_id: String,
    row_count: u64,
    timestamp_range: DatasetTimestampRange,
    vortex_sha256: String,
    dataset_identity_path: String,
    bar_timestamp_convention: String,
    published_unix_ms: u64,
    producer_provenance: ProducerProvenanceEnvelopeV1,
    manifest_binding_sha256: String,
}

pub fn canonical_dataset_root(
    configured_root: impl AsRef<Path>,
    identity: &CanonicalDatasetIdentity,
) -> Result<PathBuf> {
    let configured_root = configured_root.as_ref();
    fs::create_dir_all(configured_root).with_context(|| {
        format!(
            "failed to create canonical dataset root {}",
            configured_root.display()
        )
    })?;
    let resolved_root = fs::canonicalize(configured_root).with_context(|| {
        format!(
            "failed to resolve canonical dataset root {}",
            configured_root.display()
        )
    })?;
    let component = identity.to_path_component();
    if component.is_empty()
        || component == "."
        || component == ".."
        || component.contains(['/', '\\', ':'])
    {
        bail!("canonical dataset identity produced an unsafe path component");
    }
    let dataset_root = configured_root.join(&component);
    if dataset_root.exists() {
        reject_link_or_reparse(&dataset_root)?;
        let resolved_dataset_root = fs::canonicalize(&dataset_root).with_context(|| {
            format!("failed to resolve dataset root {}", dataset_root.display())
        })?;
        if !resolved_dataset_root.starts_with(&resolved_root) {
            bail!(
                "dataset root {} escapes configured root {}",
                resolved_dataset_root.display(),
                resolved_root.display()
            );
        }
        let decoded = CanonicalDatasetIdentity::from_path_component(
            dataset_root
                .file_name()
                .and_then(|name| name.to_str())
                .context("dataset root component is not UTF-8")?,
        )
        .context("decode canonical dataset root component")?;
        if &decoded != identity {
            bail!("dataset root identity does not round-trip exactly");
        }
    }
    Ok(dataset_root)
}

pub fn publish_vortex_generation(request: PublishRequest<'_>) -> Result<PublishResult> {
    let PublishRequest {
        configured_root,
        identity,
        expected_generation,
        timestamp_range,
        provenance,
        chunks,
    } = request;
    if chunks.is_empty() {
        bail!("cannot publish zero Vortex chunks");
    }
    publish_vortex_generation_streaming(
        PublishMetadataRequest {
            configured_root,
            identity,
            expected_generation,
            provenance,
        },
        move |candidate_path| {
            let write_stats = write_vortex_chunks(candidate_path, chunks)?;
            Ok(CandidateWriteOutcome {
                write_stats,
                timestamp_range,
            })
        },
    )
}

/// Publish a candidate produced by a bounded streaming writer.
pub fn publish_vortex_generation_streaming(
    request: PublishMetadataRequest<'_>,
    write_candidate: impl FnOnce(&Path) -> Result<CandidateWriteOutcome>,
) -> Result<PublishResult> {
    publish_vortex_generation_streaming_with_install_hook(
        request,
        None,
        write_candidate,
        |_, _, _| Ok(()),
    )
}

/// Publish only if the exact selected generation receipt is still current at
/// the pointer-update linearization lock.
///
/// A generation id alone is insufficient because byte-identical Vortex data
/// can be republished with different manifest provenance. The selected
/// manifest binding is therefore compared under the same lock as the normal
/// expected-generation CAS.
pub fn publish_vortex_generation_streaming_exact(
    request: PublishMetadataRequest<'_>,
    selected: &SelectedDatasetGenerationV1,
    write_candidate: impl FnOnce(&Path) -> Result<CandidateWriteOutcome>,
) -> Result<PublishResult> {
    selected.validate()?;
    if request.identity != selected.identity() {
        bail!(
            "exact publication identity {} does not match selected identity {}",
            request.identity.to_path_component(),
            selected.identity().to_path_component()
        );
    }
    if request.expected_generation != Some(selected.generation_id()) {
        bail!(
            "exact publication expected generation {:?} does not match selected generation {}",
            request.expected_generation,
            selected.generation_id()
        );
    }
    publish_vortex_generation_streaming_with_install_hook(
        request,
        Some(selected),
        write_candidate,
        |_, _, _| Ok(()),
    )
}

fn publish_vortex_generation_streaming_with_install_hook(
    request: PublishMetadataRequest<'_>,
    exact_selection: Option<&SelectedDatasetGenerationV1>,
    write_candidate: impl FnOnce(&Path) -> Result<CandidateWriteOutcome>,
    after_generation_install: impl FnOnce(&Path, &Path, &str) -> Result<()>,
) -> Result<PublishResult> {
    request.provenance.validate()?;
    let dataset_root = canonical_dataset_root(request.configured_root, request.identity)?;
    let dataset_root_existed = dataset_root.exists();
    fs::create_dir_all(&dataset_root)
        .with_context(|| format!("failed to create dataset root {}", dataset_root.display()))?;
    let dataset_root_cleanup =
        NewDatasetRootCleanup::new((!dataset_root_existed).then(|| dataset_root.clone()));
    reject_link_or_reparse(&dataset_root)?;

    let nonce = publication_nonce();
    let candidate_path = dataset_root.join(format!("candidate-{nonce}.vortex"));
    let candidate_lease = DatasetCandidateLease::acquire(&candidate_path)?;
    let _candidate_cleanup =
        CandidatePublicationCleanup::new(candidate_path.clone(), candidate_lease);
    let outcome = write_candidate(&candidate_path)?;
    let write_stats = outcome.write_stats;
    if write_stats.row_count == 0 {
        bail!("refusing to publish an empty Vortex generation");
    }
    let vortex_sha256 = sha256_file(&candidate_path)?;
    let generation_id = format!("g1-{vortex_sha256}.vortex");
    validate_generation_id(&generation_id)?;
    let generation_path = dataset_root.join(&generation_id);

    // The expected-generation comparison and immutable-generation installation
    // share one linearization lock. Installing before this lock is unsafe: a
    // second publisher can deduplicate the same bytes, publish them, and then
    // have the losing installer unlink the now-current generation while
    // cleaning up its CAS conflict.
    let root_lock_path = dataset_root.join(".publication.lock");
    let root_lock = open_lock_file(&root_lock_path)?;
    root_lock
        .lock()
        .with_context(|| format!("failed to lock dataset root {}", dataset_root.display()))?;
    let current = read_current_manifest_at(&dataset_root, request.identity, false)?;
    let current_generation = current.as_ref().map(DatasetManifestV1::generation_id);
    if let Some(selected) = exact_selection
        && (current_generation != Some(selected.generation_id())
            || current
                .as_ref()
                .map(DatasetManifestV1::manifest_binding_sha256)
                != Some(selected.manifest_binding_sha256()))
    {
        let conflict = ExactDatasetGenerationConflict::new(selected, current.as_ref());
        root_lock
            .unlock()
            .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;
        return Err(conflict.into());
    } else if exact_selection.is_none() && current_generation != request.expected_generation {
        root_lock
            .unlock()
            .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;
        return Err(PublicationConflict {
            expected_generation: request.expected_generation.map(str::to_owned),
            current_generation: current_generation.map(str::to_owned),
        }
        .into());
    }

    let installed_new_generation = if generation_path.exists() {
        reject_link_or_reparse(&generation_path)?;
        let existing_hash = sha256_file(&generation_path)?;
        if existing_hash != vortex_sha256 {
            bail!(
                "content-addressed generation collision at {}",
                generation_path.display()
            );
        }
        fs::remove_file(&candidate_path).with_context(|| {
            format!(
                "failed to remove duplicate candidate {}",
                candidate_path.display()
            )
        })?;
        false
    } else {
        atomic_replace_file(&candidate_path, &generation_path).with_context(|| {
            format!(
                "failed to install immutable generation {} -> {}",
                candidate_path.display(),
                generation_path.display()
            )
        })?;
        sync_parent_directory(&generation_path)?;
        true
    };
    let installed_generation_cleanup =
        InstalledGenerationCleanup::new(installed_new_generation.then(|| generation_path.clone()));
    after_generation_install(&root_lock_path, &generation_path, &vortex_sha256)?;
    verify_generation(
        &generation_path,
        &vortex_sha256,
        write_stats.row_count,
        outcome.timestamp_range,
    )?;

    let durable_commit_id = format!("c1-{}", publication_nonce());
    let mut wire = ManifestWireV1 {
        schema: MANIFEST_SCHEMA.to_owned(),
        version: MANIFEST_VERSION,
        generation_id: generation_id.clone(),
        previous_generation: current_generation.map(str::to_owned),
        durable_commit_id: durable_commit_id.clone(),
        row_count: write_stats.row_count,
        timestamp_range: outcome.timestamp_range,
        vortex_sha256,
        dataset_identity_path: request.identity.to_path_component(),
        bar_timestamp_convention: BarTimestampConvention::BarOpen.as_str().to_owned(),
        published_unix_ms: unix_time_ms()?,
        producer_provenance: request.provenance.clone(),
        manifest_binding_sha256: String::new(),
    };
    wire.manifest_binding_sha256 = manifest_binding_sha256(&wire);
    let bytes = serde_json::to_vec(&wire).context("serialize dataset manifest")?;
    write_pointer_atomically(&dataset_root, &bytes, move || {
        // The pointer rename is the publication linearization point. Disarm
        // generation cleanup immediately after that rename, before directory
        // fsync can return an ambiguous durability error: an observable
        // pointer must never reference a generation that error cleanup unlinks.
        installed_generation_cleanup.commit();
    })?;
    let manifest = read_current_manifest_at(&dataset_root, request.identity, true)?
        .context("published dataset has no reopened canonical manifest")?;
    if manifest.generation_id() != generation_id
        || manifest.row_count() != write_stats.row_count
        || manifest.manifest_binding_sha256() != wire.manifest_binding_sha256
    {
        bail!("reopened canonical manifest disagrees with the published generation");
    }
    root_lock
        .unlock()
        .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;

    let result = PublishResult {
        manifest,
        previous_generation: wire.previous_generation,
        durable_commit_id,
    };
    dataset_root_cleanup.commit();
    Ok(result)
}

pub fn read_current_manifest(
    configured_root: impl AsRef<Path>,
    identity: &CanonicalDatasetIdentity,
) -> Result<DatasetManifestV1> {
    let dataset_root = canonical_dataset_root(configured_root, identity)?;
    read_current_manifest_at(&dataset_root, identity, true)?
        .context("canonical dataset has no versioned completion manifest")
}

/// Read and validate the versioned manifest and its contained generation path
/// without hashing or decoding the generation bytes.
///
/// This is intentionally an inventory-only operation. Callers must preserve
/// that distinction in their API/UI and use [`read_current_manifest`] or
/// [`open_current_dataset_generation`] before consuming any market data.
pub fn read_current_manifest_metadata(
    configured_root: impl AsRef<Path>,
    identity: &CanonicalDatasetIdentity,
) -> Result<DatasetManifestV1> {
    let dataset_root = canonical_dataset_root(configured_root, identity)?;
    read_current_manifest_at(&dataset_root, identity, false)?
        .context("canonical dataset has no versioned completion manifest")
}

pub fn open_current_generation(
    configured_root: impl AsRef<Path>,
    identity: &CanonicalDatasetIdentity,
) -> Result<DatasetGenerationLease> {
    Ok(open_current_dataset_generation(configured_root, identity)?.1)
}

/// Resolve the current manifest and acquire its generation reader pin under
/// the same dataset-root lock. Returning these separately from two calls would
/// permit a publication between them and could bind bytes from generation N+1
/// to the manifest for generation N.
pub fn open_current_dataset_generation(
    configured_root: impl AsRef<Path>,
    identity: &CanonicalDatasetIdentity,
) -> Result<(DatasetManifestV1, DatasetGenerationLease)> {
    let dataset_root = canonical_dataset_root(configured_root, identity)?;
    let root_lock_path = dataset_root.join(".publication.lock");
    let root_lock = open_lock_file(&root_lock_path)?;
    root_lock
        .lock()
        .with_context(|| format!("failed to lock dataset root {}", dataset_root.display()))?;
    let manifest = read_current_manifest_at(&dataset_root, identity, true)?
        .context("canonical dataset has no versioned completion manifest")?;
    let lease = DatasetGenerationLease::acquire(
        &dataset_root,
        manifest.generation_id(),
        manifest.generation_path(),
        manifest.vortex_sha256().to_owned(),
    )?;
    root_lock
        .unlock()
        .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;
    Ok((manifest, lease))
}

/// Fully verify and pin exactly the selected current manifest under the
/// publication lock.
///
/// A pointer that has already advanced is a typed conflict. Once the lease is
/// acquired, a later publication may advance the pointer but cannot invalidate
/// the immutable bytes held by the returned reader pin.
pub fn open_exact_dataset_generation(
    configured_root: impl AsRef<Path>,
    selected: &SelectedDatasetGenerationV1,
) -> Result<(DatasetManifestV1, DatasetGenerationLease)> {
    selected.validate()?;
    let dataset_root = canonical_dataset_root(configured_root, selected.identity())?;
    if !dataset_root.exists() {
        return Err(ExactDatasetGenerationConflict::new(selected, None).into());
    }

    let root_lock_path = dataset_root.join(".publication.lock");
    let root_lock = open_lock_file(&root_lock_path)?;
    root_lock
        .lock()
        .with_context(|| format!("failed to lock dataset root {}", dataset_root.display()))?;
    let manifest = read_current_manifest_at(&dataset_root, selected.identity(), true)?;
    let Some(manifest) = manifest else {
        root_lock
            .unlock()
            .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;
        return Err(ExactDatasetGenerationConflict::new(selected, None).into());
    };

    if manifest.generation_id() != selected.generation_id()
        || manifest.manifest_binding_sha256() != selected.manifest_binding_sha256()
    {
        let conflict = ExactDatasetGenerationConflict::new(selected, Some(&manifest));
        root_lock
            .unlock()
            .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;
        return Err(conflict.into());
    }

    let lease = DatasetGenerationLease::acquire(
        &dataset_root,
        manifest.generation_id(),
        manifest.generation_path(),
        manifest.vortex_sha256().to_owned(),
    )?;
    root_lock
        .unlock()
        .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;
    Ok((manifest, lease))
}

pub fn collect_unreferenced_generations(
    configured_root: impl AsRef<Path>,
    identity: &CanonicalDatasetIdentity,
) -> Result<Vec<PathBuf>> {
    let dataset_root = canonical_dataset_root(configured_root, identity)?;
    let root_lock_path = dataset_root.join(".publication.lock");
    let root_lock = open_lock_file(&root_lock_path)?;
    root_lock
        .lock()
        .with_context(|| format!("failed to lock dataset root {}", dataset_root.display()))?;
    let current = read_current_manifest_at(&dataset_root, identity, true)?;
    let mut protected = HashSet::new();
    if let Some(manifest) = &current {
        protected.insert(manifest.wire.generation_id.clone());
        if let Some(previous) = &manifest.wire.previous_generation {
            protected.insert(previous.clone());
        }
    }
    let mut removed = Vec::new();
    for entry in fs::read_dir(&dataset_root)
        .with_context(|| format!("failed to enumerate {}", dataset_root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !is_generation_id(name) || protected.contains(name) {
            continue;
        }
        reject_link_or_reparse(&path)?;
        let expected_hash = name
            .strip_prefix("g1-")
            .and_then(|value| value.strip_suffix(".vortex"))
            .context("generation hash missing from canonical name")?;
        if sha256_file(&path)? != expected_hash {
            tracing::error!(
                target: "neoethos_data::dataset_manifest",
                path = %path.display(),
                "unreferenced generation hash does not match its name; refusing destructive GC"
            );
            continue;
        }
        let Some(exclusive) = try_acquire_exclusive(&dataset_root, name)? else {
            continue;
        };
        fs::remove_file(&path)
            .with_context(|| format!("failed to remove generation {}", path.display()))?;
        let lock_path = exclusive.lock_path.clone();
        exclusive.unlock()?;
        remove_lock_file(&lock_path)?;
        removed.push(path);
    }
    if !removed.is_empty() {
        sync_parent_directory(&dataset_root.join(MANIFEST_FILE))?;
    }
    root_lock
        .unlock()
        .with_context(|| format!("failed to unlock dataset root {}", dataset_root.display()))?;
    Ok(removed)
}

fn read_current_manifest_at(
    dataset_root: &Path,
    expected_identity: &CanonicalDatasetIdentity,
    verify_data: bool,
) -> Result<Option<DatasetManifestV1>> {
    let pointer = dataset_root.join(MANIFEST_FILE);
    if !pointer.exists() {
        return Ok(None);
    }
    reject_link_or_reparse(&pointer)?;
    let bytes = fs::read(&pointer)
        .with_context(|| format!("failed to read dataset manifest {}", pointer.display()))?;
    if bytes.is_empty() {
        bail!(
            "legacy empty completion marker at {}; run the explicit legacy migration",
            pointer.display()
        );
    }
    let wire: ManifestWireV1 =
        serde_json::from_slice(&bytes).context("decode versioned dataset manifest")?;
    validate_manifest_wire(&wire, expected_identity)?;
    let generation_path = dataset_root.join(&wire.generation_id);
    reject_link_or_reparse(&generation_path)?;
    let manifest = DatasetManifestV1 {
        identity: CanonicalDatasetIdentity::from_path_component(&wire.dataset_identity_path)
            .context("decode dataset identity from manifest")?,
        wire,
        dataset_root: dataset_root.to_path_buf(),
    };
    if verify_data {
        verify_generation(
            &generation_path,
            manifest.vortex_sha256(),
            manifest.row_count(),
            manifest.timestamp_range(),
        )?;
    }
    Ok(Some(manifest))
}

fn validate_manifest_wire(
    wire: &ManifestWireV1,
    expected_identity: &CanonicalDatasetIdentity,
) -> Result<()> {
    if wire.schema != MANIFEST_SCHEMA || wire.version != MANIFEST_VERSION {
        bail!(
            "unsupported dataset manifest schema/version {:?}/{}",
            wire.schema,
            wire.version
        );
    }
    validate_generation_id(&wire.generation_id)?;
    if let Some(previous) = &wire.previous_generation {
        validate_generation_id(previous)?;
    }
    validate_safe_component("durable commit id", &wire.durable_commit_id)?;
    if wire.row_count == 0 {
        bail!("dataset manifest row count must be positive");
    }
    DatasetTimestampRange::new(wire.timestamp_range.start_ms, wire.timestamp_range.end_ms)?;
    validate_sha256_hex("Vortex generation", &wire.vortex_sha256)?;
    let identity = CanonicalDatasetIdentity::from_path_component(&wire.dataset_identity_path)
        .context("decode dataset manifest identity")?;
    if &identity != expected_identity {
        bail!("dataset manifest identity does not match the requested canonical identity");
    }
    if wire.bar_timestamp_convention != BarTimestampConvention::BarOpen.as_str()
        || !identity.bar_timestamp_convention().is_canonical_bar_open()
    {
        bail!("canonical dataset manifest requires explicit bar_open timestamps");
    }
    wire.producer_provenance.validate()?;
    validate_sha256_hex("manifest binding", &wire.manifest_binding_sha256)?;
    let actual_binding = manifest_binding_sha256(wire);
    if actual_binding != wire.manifest_binding_sha256 {
        bail!(
            "dataset manifest binding mismatch: expected {}, got {}",
            wire.manifest_binding_sha256,
            actual_binding
        );
    }
    let expected_generation = format!("g1-{}.vortex", wire.vortex_sha256);
    if wire.generation_id != expected_generation {
        bail!("generation id does not bind the declared Vortex hash");
    }
    Ok(())
}

fn verify_generation(
    path: &Path,
    expected_sha256: &str,
    expected_rows: u64,
    expected_range: DatasetTimestampRange,
) -> Result<()> {
    reject_link_or_reparse(path)?;
    let actual_sha256 = sha256_file(path)?;
    if actual_sha256 != expected_sha256 {
        bail!(
            "Vortex generation hash mismatch for {}: expected {}, got {}",
            path.display(),
            expected_sha256,
            actual_sha256
        );
    }
    let actual_rows = read_vortex_row_count(path)
        .with_context(|| format!("failed to read Vortex footer {}", path.display()))?;
    if actual_rows != expected_rows {
        bail!(
            "Vortex generation row-count mismatch for {}: expected {}, got {}",
            path.display(),
            expected_rows,
            actual_rows
        );
    }
    if expected_rows == 0 {
        bail!("verified Vortex generation is empty");
    }
    let first = read_vortex_i64_projection_range(path, "timestamp", 0..1)?;
    let last_start = expected_rows - 1;
    let last = read_vortex_i64_projection_range(path, "timestamp", last_start..expected_rows)?;
    let first = first
        .first()
        .copied()
        .context("first timestamp projection is empty")?;
    let last = last
        .first()
        .copied()
        .context("last timestamp projection is empty")?;
    if first != expected_range.start_ms || last != expected_range.end_ms {
        bail!(
            "Vortex generation timestamp range mismatch for {}: expected {}..{}, got {}..{}",
            path.display(),
            expected_range.start_ms,
            expected_range.end_ms,
            first,
            last
        );
    }
    Ok(())
}

fn write_pointer_atomically(
    dataset_root: &Path,
    bytes: &[u8],
    after_swap: impl FnOnce(),
) -> Result<()> {
    write_pointer_atomically_with_sync(dataset_root, bytes, after_swap, sync_parent_directory)
}

fn write_pointer_atomically_with_sync(
    dataset_root: &Path,
    bytes: &[u8],
    after_swap: impl FnOnce(),
    sync_parent: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let pointer = dataset_root.join(MANIFEST_FILE);
    let staged = dataset_root.join(format!("{MANIFEST_FILE}.tmp-{}", publication_nonce()));
    let mut cleanup = StagedFileCleanup::new(staged.clone());
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)
        .with_context(|| format!("failed to create staged manifest {}", staged.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("failed to write staged manifest {}", staged.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush staged manifest {}", staged.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to fsync staged manifest {}", staged.display()))?;
    drop(file);
    atomic_replace_file(&staged, &pointer)?;
    cleanup.committed = true;
    after_swap();
    sync_parent(&pointer)?;
    Ok(())
}

fn manifest_binding_sha256(wire: &ManifestWireV1) -> String {
    let mut bytes = Vec::with_capacity(512);
    bytes.extend_from_slice(BINDING_DOMAIN);
    push_binding_text(&mut bytes, &wire.schema);
    bytes.extend_from_slice(&wire.version.to_be_bytes());
    push_binding_text(&mut bytes, &wire.generation_id);
    match &wire.previous_generation {
        Some(previous) => {
            bytes.push(1);
            push_binding_text(&mut bytes, previous);
        }
        None => bytes.push(0),
    }
    push_binding_text(&mut bytes, &wire.durable_commit_id);
    bytes.extend_from_slice(&wire.row_count.to_be_bytes());
    bytes.extend_from_slice(&wire.timestamp_range.start_ms.to_be_bytes());
    bytes.extend_from_slice(&wire.timestamp_range.end_ms.to_be_bytes());
    push_binding_text(&mut bytes, &wire.vortex_sha256);
    push_binding_text(&mut bytes, &wire.dataset_identity_path);
    push_binding_text(&mut bytes, &wire.bar_timestamp_convention);
    bytes.extend_from_slice(&wire.published_unix_ms.to_be_bytes());
    push_binding_text(&mut bytes, wire.producer_provenance.schema_id());
    push_binding_text(&mut bytes, wire.producer_provenance.payload_sha256());
    sha256_bytes(&bytes)
}

fn push_binding_text(bytes: &mut Vec<u8>, value: &str) {
    let length = u32::try_from(value.len()).expect("bounded manifest text fits in u32");
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open {} for SHA-256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn validate_sha256_hex(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} SHA-256 is not 64 lowercase hex characters");
    }
    Ok(())
}

fn validate_schema_id(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > ProducerProvenanceEnvelopeV1::MAX_SCHEMA_ID_BYTES {
        bail!("producer schema id has an invalid length");
    }
    let segments: Vec<&str> = value.split('.').collect();
    if segments.len() < 3
        || segments.iter().any(|segment| {
            segment.is_empty()
                || !segment
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
                || !segment
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_lowercase)
                || !segment
                    .as_bytes()
                    .last()
                    .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
        || !segments.last().is_some_and(|segment| {
            segment.strip_prefix('v').is_some_and(|version| {
                !version.is_empty() && version.bytes().all(|b| b.is_ascii_digit())
            })
        })
    {
        bail!("invalid namespaced producer schema id {value:?}");
    }
    Ok(())
}

fn validate_generation_id(value: &str) -> Result<()> {
    if !is_generation_id(value) {
        bail!("unsafe or noncanonical generation id {value:?}");
    }
    Ok(())
}

fn is_generation_id(value: &str) -> bool {
    value
        .strip_prefix("g1-")
        .and_then(|hash| hash.strip_suffix(".vortex"))
        .is_some_and(|hash| {
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

fn validate_safe_component(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains(['/', '\\', ':'])
        || Path::new(value).is_absolute()
    {
        bail!("unsafe {label} {value:?}");
    }
    Ok(())
}

fn open_lock_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open lock file {}", path.display()))
}

fn remove_candidate_lease_file(candidate_path: &Path) -> Result<()> {
    let file_name = candidate_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("candidate path has no UTF-8 file name")?;
    let lock_path = candidate_path.with_file_name(format!("{file_name}.lease"));
    match fs::remove_file(&lock_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to remove candidate lease {}", lock_path.display())),
    }
}

fn publication_nonce() -> String {
    format!(
        "{}-{}-{}",
        std::process::id(),
        unix_time_ms().unwrap_or_default(),
        PUBLICATION_NONCE.fetch_add(1, Ordering::Relaxed)
    )
}

fn unix_time_ms() -> Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis();
    u64::try_from(millis).context("Unix timestamp milliseconds overflow u64")
}

fn reject_link_or_reparse(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect path {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!(
            "refusing symlink at canonical dataset path {}",
            path.display()
        );
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!(
                "refusing reparse point at canonical dataset path {}",
                path.display()
            );
        }
    }
    Ok(())
}

struct StagedFileCleanup {
    path: PathBuf,
    committed: bool,
}

struct CandidatePublicationCleanup {
    candidate_path: PathBuf,
    lease: Option<DatasetCandidateLease>,
}

struct NewDatasetRootCleanup {
    path: Option<PathBuf>,
}

struct InstalledGenerationCleanup {
    path: Option<PathBuf>,
}

impl InstalledGenerationCleanup {
    fn new(path: Option<PathBuf>) -> Self {
        Self { path }
    }

    fn commit(mut self) {
        self.path = None;
    }
}

impl Drop for InstalledGenerationCleanup {
    fn drop(&mut self) {
        let Some(path) = self.path.take() else {
            return;
        };
        match fs::remove_file(&path) {
            Ok(()) => {
                if let Err(error) = sync_parent_directory(&path) {
                    tracing::warn!(
                        target: "neoethos_data::dataset_manifest",
                        path = %path.display(),
                        error = %error,
                        "failed to sync dataset root after removing unpublished generation"
                    );
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                target: "neoethos_data::dataset_manifest",
                path = %path.display(),
                error = %error,
                "failed to remove unpublished immutable generation"
            ),
        }
    }
}

impl NewDatasetRootCleanup {
    fn new(path: Option<PathBuf>) -> Self {
        Self { path }
    }

    fn commit(mut self) {
        self.path = None;
    }
}

impl Drop for NewDatasetRootCleanup {
    fn drop(&mut self) {
        let Some(path) = self.path.take() else {
            return;
        };
        match fs::remove_dir(&path) {
            Ok(()) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
                ) => {}
            Err(error) => tracing::warn!(
                target: "neoethos_data::dataset_manifest",
                path = %path.display(),
                error = %error,
                "failed to remove newly-created empty dataset root after publication error"
            ),
        }
    }
}

impl CandidatePublicationCleanup {
    fn new(candidate_path: PathBuf, lease: DatasetCandidateLease) -> Self {
        Self {
            candidate_path,
            lease: Some(lease),
        }
    }
}

impl Drop for CandidatePublicationCleanup {
    fn drop(&mut self) {
        drop(self.lease.take());
        match fs::remove_file(&self.candidate_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                target: "neoethos_data::dataset_manifest",
                path = %self.candidate_path.display(),
                error = %error,
                "failed to remove unpublished Vortex candidate"
            ),
        }
        if let Err(error) = remove_candidate_lease_file(&self.candidate_path) {
            tracing::warn!(
                target: "neoethos_data::dataset_manifest",
                path = %self.candidate_path.display(),
                error = %error,
                "failed to remove Vortex candidate lease file"
            );
        }
    }
}

impl StagedFileCleanup {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Ohlcv, ohlcv_to_vortex_chunks};

    const SHA256_BOUNDED_STACK_CHILD: &str = "NEOETHOS_SHA256_BOUNDED_STACK_CHILD";

    #[test]
    fn sha256_file_child_uses_windows_main_thread_sized_stack() {
        if std::env::var_os(SHA256_BOUNDED_STACK_CHILD).is_none() {
            return;
        }

        let temp = tempfile::tempdir().expect("temporary SHA-256 fixture root");
        let path = temp.path().join("candidate.vortex");
        fs::write(&path, b"canonical trendbar candidate").expect("write SHA-256 fixture");
        std::thread::Builder::new()
            .name("sha256-bounded-stack".to_owned())
            .stack_size(1024 * 1024)
            .spawn(move || sha256_file(&path))
            .expect("spawn bounded-stack SHA-256 worker")
            .join()
            .expect("bounded-stack SHA-256 worker must not overflow")
            .expect("hash fixture on bounded stack");
    }

    #[test]
    fn sha256_file_does_not_require_a_megabyte_stack_frame() {
        let status = std::process::Command::new(
            std::env::current_exe().expect("current neoethos-data unit-test executable"),
        )
        .arg("--exact")
        .arg(
            "core::dataset_manifest::tests::sha256_file_child_uses_windows_main_thread_sized_stack",
        )
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(SHA256_BOUNDED_STACK_CHILD, "1")
        .status()
        .expect("spawn bounded-stack SHA-256 regression child");
        assert!(
            status.success(),
            "SHA-256 child overflowed or failed on a Windows main-thread-sized stack: {status}"
        );
    }

    #[test]
    fn post_swap_directory_sync_error_keeps_the_visible_generation() {
        let root = tempfile::tempdir().expect("temporary dataset root");
        let generation = root.path().join("g1-test.vortex");
        fs::write(&generation, b"immutable generation").expect("generation fixture");
        let cleanup = InstalledGenerationCleanup::new(Some(generation.clone()));

        let error = write_pointer_atomically_with_sync(
            root.path(),
            br#"{"generation":"g1-test.vortex"}"#,
            move || cleanup.commit(),
            |_| bail!("injected directory fsync failure after pointer swap"),
        )
        .expect_err("post-swap fsync fault must remain observable");

        assert!(format!("{error:#}").contains("injected directory fsync failure"));
        assert!(
            root.path().join(MANIFEST_FILE).is_file(),
            "the pointer rename already linearized and must remain visible"
        );
        assert!(
            generation.is_file(),
            "an error after pointer swap must never unlink its referenced generation"
        );
    }

    #[test]
    fn generation_install_phase_is_inside_the_publication_lock() {
        let root = tempfile::tempdir().expect("temporary dataset root");
        let identity = CanonicalDatasetIdentity::external(
            "publication-lock-fixture",
            "EURUSD",
            neoethos_dataset_contracts::CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        let provenance = ProducerProvenanceEnvelopeV1::new(
            "neoethos.publication-lock-fixture.v1",
            b"lock-order".to_vec(),
        )
        .expect("provenance");
        let timestamps = vec![1_700_000_040_000, 1_700_000_100_000];
        let data = Ohlcv {
            timestamp: Some(timestamps.clone()),
            open: vec![1.0, 1.1],
            high: vec![1.2, 1.3],
            low: vec![0.9, 1.0],
            close: vec![1.1, 1.2],
            volume: Some(vec![1.0, 2.0]),
        };

        publish_vortex_generation_streaming_with_install_hook(
            PublishMetadataRequest {
                configured_root: root.path(),
                identity: &identity,
                expected_generation: None,
                provenance: &provenance,
            },
            None,
            |candidate_path| {
                let write_stats =
                    write_vortex_chunks(candidate_path, ohlcv_to_vortex_chunks(&data, 1)?)?;
                Ok(CandidateWriteOutcome {
                    write_stats,
                    timestamp_range: DatasetTimestampRange::new(timestamps[0], timestamps[1])?,
                })
            },
            |lock_path, generation_path, generation_hash| {
                assert!(generation_path.is_file());
                let expected_file_name = format!("g1-{generation_hash}.vortex");
                assert_eq!(
                    generation_path.file_name().and_then(|name| name.to_str()),
                    Some(expected_file_name.as_str())
                );
                let probe = open_lock_file(lock_path)?;
                match probe.try_lock() {
                    Ok(()) => {
                        probe.unlock()?;
                        bail!("generation became visible before acquiring the publication lock")
                    }
                    Err(std::fs::TryLockError::WouldBlock) => Ok(()),
                    Err(std::fs::TryLockError::Error(error)) => {
                        Err(error).context("probing publication lock ownership")
                    }
                }
            },
        )
        .expect("publication whose install is protected by the root lock");
    }
}

impl Drop for StagedFileCleanup {
    fn drop(&mut self) {
        if !self.committed {
            match fs::remove_file(&self.path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => tracing::warn!(
                    target: "neoethos_data::dataset_manifest",
                    path = %self.path.display(),
                    error = %error,
                    "failed to remove staged manifest"
                ),
            }
        }
    }
}
