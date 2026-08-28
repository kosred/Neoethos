use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::contracts::{
    BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1, BrokerFinancialTruthBindingV1,
    BrokerFinancialTruthBundleManifestV1, BrokerFinancialTruthBundleReceiptV1,
    BrokerFinancialTruthContractErrorV1, ImmutableVortexArtifactV1, max_manifest_bytes,
    sha256_bytes, sha256_file,
};
use crate::contracts_v2::{
    BrokerFinancialTruthBundleManifestV2, BrokerFinancialTruthBundleReceiptV2,
};

static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerFinancialTruthStoreErrorCodeV1 {
    ContractInvalid,
    UnsafeFilesystemEntry,
    ReceiptInvalid,
    ManifestMissing,
    ManifestDigestMismatch,
    ManifestInvalid,
    BindingMismatch,
    ArtifactSetMismatch,
    ArtifactLengthMismatch,
    ArtifactDigestMismatch,
    SourceMismatch,
    PublishConflict,
    Io,
}

#[derive(Debug)]
pub struct BrokerFinancialTruthStoreErrorV1 {
    code: BrokerFinancialTruthStoreErrorCodeV1,
    detail: String,
}

impl BrokerFinancialTruthStoreErrorV1 {
    fn new(code: BrokerFinancialTruthStoreErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> BrokerFinancialTruthStoreErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for BrokerFinancialTruthStoreErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "broker financial truth store: {}", self.detail)
    }
}

impl Error for BrokerFinancialTruthStoreErrorV1 {}

impl From<BrokerFinancialTruthContractErrorV1> for BrokerFinancialTruthStoreErrorV1 {
    fn from(error: BrokerFinancialTruthContractErrorV1) -> Self {
        Self::new(
            BrokerFinancialTruthStoreErrorCodeV1::ContractInvalid,
            error.to_string(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerFinancialTruthArtifactSourceV1 {
    relative_path: String,
    source_path: PathBuf,
}

impl BrokerFinancialTruthArtifactSourceV1 {
    pub fn new(
        relative_path: impl Into<String>,
        source_path: impl Into<PathBuf>,
    ) -> Result<Self, BrokerFinancialTruthStoreErrorV1> {
        let source = Self {
            relative_path: relative_path.into(),
            source_path: source_path.into(),
        };
        validate_source_basename(&source.relative_path)?;
        Ok(source)
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
}

/// An integrity-checked immutable bundle reopen.
///
/// This is deliberately not a financial-truth capability. Chunk 1 proves only
/// exact receipt, file-set, length and digest integrity. A later semantic
/// validator must decode the Vortex schemas, compare every decoded row with its
/// retained raw cTrader envelope, verify synchronization/reconciliation, and
/// only then create a run-scoped capability.
#[derive(Clone, Debug)]
pub struct VerifiedImmutableBrokerFinancialTruthBundleV1 {
    root: PathBuf,
    receipt: BrokerFinancialTruthBundleReceiptV1,
    manifest: BrokerFinancialTruthBundleManifestV1,
}

/// Integrity-checked reopen of an additive V2 bundle. Semantic validation and
/// capability construction remain a separate fail-closed step.
#[derive(Clone, Debug)]
pub struct VerifiedImmutableBrokerFinancialTruthBundleV2 {
    root: PathBuf,
    receipt: BrokerFinancialTruthBundleReceiptV2,
    manifest: BrokerFinancialTruthBundleManifestV2,
}

impl VerifiedImmutableBrokerFinancialTruthBundleV2 {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn receipt(&self) -> &BrokerFinancialTruthBundleReceiptV2 {
        &self.receipt
    }

    pub const fn manifest(&self) -> &BrokerFinancialTruthBundleManifestV2 {
        &self.manifest
    }

    pub fn artifact_path(&self, artifact: &ImmutableVortexArtifactV1) -> PathBuf {
        self.root.join(artifact.relative_path())
    }
}

impl VerifiedImmutableBrokerFinancialTruthBundleV1 {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn receipt(&self) -> &BrokerFinancialTruthBundleReceiptV1 {
        &self.receipt
    }

    pub const fn manifest(&self) -> &BrokerFinancialTruthBundleManifestV1 {
        &self.manifest
    }

    pub fn artifact_path(&self, artifact: &ImmutableVortexArtifactV1) -> PathBuf {
        self.root.join(artifact.relative_path())
    }
}

/// Content-addressed broker-evidence store with no mutable `current` pointer.
#[derive(Clone, Debug)]
pub struct BrokerFinancialTruthBundleStoreV1 {
    root: PathBuf,
}

impl BrokerFinancialTruthBundleStoreV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn bundle_path(&self, receipt: &BrokerFinancialTruthBundleReceiptV1) -> PathBuf {
        self.root.join(receipt.bundle_id())
    }

    pub fn bundle_path_v2(&self, receipt: &BrokerFinancialTruthBundleReceiptV2) -> PathBuf {
        self.root.join(receipt.bundle_id())
    }

    pub fn publish(
        &self,
        manifest: &BrokerFinancialTruthBundleManifestV1,
        sources: &[BrokerFinancialTruthArtifactSourceV1],
    ) -> Result<BrokerFinancialTruthBundleReceiptV1, BrokerFinancialTruthStoreErrorV1> {
        let manifest_bytes = manifest.canonical_json_bytes()?;
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        let receipt = BrokerFinancialTruthBundleReceiptV1::from_manifest_sha256(manifest_sha256)?;
        let artifacts = manifest.artifacts();
        let source_map = validate_sources(&artifacts, sources)?;
        self.ensure_safe_store_root()?;

        let final_root = self.bundle_path(&receipt);
        match fs::symlink_metadata(&final_root) {
            Ok(_) => {
                self.open_exact(&receipt, manifest.binding())?;
                return Ok(receipt);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(io_error(
                    format!("cannot inspect bundle path {}", final_root.display()),
                    error,
                ));
            }
        }

        let staging_root = self.create_staging_directory()?;
        let publish_result = (|| {
            for artifact in &artifacts {
                let source = source_map.get(artifact.relative_path()).ok_or_else(|| {
                    BrokerFinancialTruthStoreErrorV1::new(
                        BrokerFinancialTruthStoreErrorCodeV1::SourceMismatch,
                        format!("no source supplied for {}", artifact.relative_path()),
                    )
                })?;
                let destination = staging_root.join(artifact.relative_path());
                copy_exact_artifact(source, &destination, artifact)?;
            }
            write_new_file(
                &staging_root.join(BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1),
                &manifest_bytes,
            )?;
            fs::rename(&staging_root, &final_root).map_err(|error| {
                io_error(
                    format!(
                        "cannot atomically publish {} as {}",
                        staging_root.display(),
                        final_root.display()
                    ),
                    error,
                )
            })?;
            self.open_exact(&receipt, manifest.binding())?;
            Ok(receipt.clone())
        })();

        if publish_result.is_err() {
            cleanup_exact_staging_directory(&self.root, &staging_root);
        }
        publish_result
    }

    pub fn open_exact(
        &self,
        receipt: &BrokerFinancialTruthBundleReceiptV1,
        expected_binding: &BrokerFinancialTruthBindingV1,
    ) -> Result<VerifiedImmutableBrokerFinancialTruthBundleV1, BrokerFinancialTruthStoreErrorV1>
    {
        receipt.validate().map_err(|error| {
            BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::ReceiptInvalid,
                error.to_string(),
            )
        })?;
        self.ensure_safe_store_root()?;
        let bundle_root = self.bundle_path(receipt);
        ensure_regular_directory(&bundle_root)?;

        let manifest_path = bundle_root.join(BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1);
        let manifest_bytes = read_bounded_regular_file(&manifest_path, max_manifest_bytes())
            .map_err(|error| {
                if error.code == BrokerFinancialTruthStoreErrorCodeV1::Io {
                    BrokerFinancialTruthStoreErrorV1::new(
                        BrokerFinancialTruthStoreErrorCodeV1::ManifestMissing,
                        error.detail,
                    )
                } else {
                    error
                }
            })?;
        let actual_manifest_sha256 = sha256_bytes(&manifest_bytes);
        if actual_manifest_sha256 != receipt.manifest_sha256() {
            return Err(BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::ManifestDigestMismatch,
                format!(
                    "manifest digest {} does not match exact receipt {}",
                    actual_manifest_sha256,
                    receipt.manifest_sha256()
                ),
            ));
        }
        let manifest = BrokerFinancialTruthBundleManifestV1::from_json_bytes(&manifest_bytes)
            .map_err(|error| {
                BrokerFinancialTruthStoreErrorV1::new(
                    BrokerFinancialTruthStoreErrorCodeV1::ManifestInvalid,
                    error.to_string(),
                )
            })?;
        if manifest.binding() != expected_binding {
            return Err(BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::BindingMismatch,
                "stored broker evidence binding differs from the exact requested dataset/search/window/assets",
            ));
        }

        let artifacts = manifest.artifacts();
        validate_exact_file_set(&bundle_root, &artifacts)?;
        for artifact in artifacts {
            validate_published_artifact(&bundle_root.join(artifact.relative_path()), artifact)?;
        }

        Ok(VerifiedImmutableBrokerFinancialTruthBundleV1 {
            root: bundle_root,
            receipt: receipt.clone(),
            manifest,
        })
    }

    pub fn publish_v2(
        &self,
        manifest: &BrokerFinancialTruthBundleManifestV2,
        sources: &[BrokerFinancialTruthArtifactSourceV1],
    ) -> Result<BrokerFinancialTruthBundleReceiptV2, BrokerFinancialTruthStoreErrorV1> {
        let manifest_bytes = manifest.canonical_json_bytes()?;
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        let receipt = BrokerFinancialTruthBundleReceiptV2::from_manifest_sha256(manifest_sha256)?;
        let artifacts = manifest.artifacts();
        let source_map = validate_sources(&artifacts, sources)?;
        self.ensure_safe_store_root()?;

        let final_root = self.bundle_path_v2(&receipt);
        match fs::symlink_metadata(&final_root) {
            Ok(_) => {
                self.open_exact_v2(&receipt, manifest.binding())?;
                return Ok(receipt);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(io_error(
                    format!("cannot inspect V2 bundle path {}", final_root.display()),
                    error,
                ));
            }
        }

        let staging_root = self.create_staging_directory()?;
        let publish_result = (|| {
            for artifact in &artifacts {
                let source = source_map.get(artifact.relative_path()).ok_or_else(|| {
                    BrokerFinancialTruthStoreErrorV1::new(
                        BrokerFinancialTruthStoreErrorCodeV1::SourceMismatch,
                        format!("no V2 source supplied for {}", artifact.relative_path()),
                    )
                })?;
                let destination = staging_root.join(artifact.relative_path());
                copy_exact_artifact(source, &destination, artifact)?;
            }
            write_new_file(
                &staging_root.join(BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1),
                &manifest_bytes,
            )?;
            fs::rename(&staging_root, &final_root).map_err(|error| {
                io_error(
                    format!(
                        "cannot atomically publish V2 {} as {}",
                        staging_root.display(),
                        final_root.display()
                    ),
                    error,
                )
            })?;
            self.open_exact_v2(&receipt, manifest.binding())?;
            Ok(receipt.clone())
        })();

        if publish_result.is_err() {
            cleanup_exact_staging_directory(&self.root, &staging_root);
        }
        publish_result
    }

    pub fn open_exact_v2(
        &self,
        receipt: &BrokerFinancialTruthBundleReceiptV2,
        expected_binding: &BrokerFinancialTruthBindingV1,
    ) -> Result<VerifiedImmutableBrokerFinancialTruthBundleV2, BrokerFinancialTruthStoreErrorV1>
    {
        receipt.validate().map_err(|error| {
            BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::ReceiptInvalid,
                error.to_string(),
            )
        })?;
        self.ensure_safe_store_root()?;
        let bundle_root = self.bundle_path_v2(receipt);
        ensure_regular_directory(&bundle_root)?;

        let manifest_path = bundle_root.join(BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1);
        let manifest_bytes = read_bounded_regular_file(&manifest_path, max_manifest_bytes())
            .map_err(|error| {
                if error.code == BrokerFinancialTruthStoreErrorCodeV1::Io {
                    BrokerFinancialTruthStoreErrorV1::new(
                        BrokerFinancialTruthStoreErrorCodeV1::ManifestMissing,
                        error.detail,
                    )
                } else {
                    error
                }
            })?;
        let actual_manifest_sha256 = sha256_bytes(&manifest_bytes);
        if actual_manifest_sha256 != receipt.manifest_sha256() {
            return Err(BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::ManifestDigestMismatch,
                format!(
                    "V2 manifest digest {} does not match exact receipt {}",
                    actual_manifest_sha256,
                    receipt.manifest_sha256()
                ),
            ));
        }
        let manifest = BrokerFinancialTruthBundleManifestV2::from_json_bytes(&manifest_bytes)
            .map_err(|error| {
                BrokerFinancialTruthStoreErrorV1::new(
                    BrokerFinancialTruthStoreErrorCodeV1::ManifestInvalid,
                    error.to_string(),
                )
            })?;
        if manifest.binding() != expected_binding {
            return Err(BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::BindingMismatch,
                "stored V2 broker evidence binding differs from the exact requested dataset/search/window/assets",
            ));
        }

        let artifacts = manifest.artifacts();
        validate_exact_file_set(&bundle_root, &artifacts)?;
        for artifact in artifacts {
            validate_published_artifact(&bundle_root.join(artifact.relative_path()), artifact)?;
        }

        Ok(VerifiedImmutableBrokerFinancialTruthBundleV2 {
            root: bundle_root,
            receipt: receipt.clone(),
            manifest,
        })
    }

    fn ensure_safe_store_root(&self) -> Result<(), BrokerFinancialTruthStoreErrorV1> {
        match fs::symlink_metadata(&self.root) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(BrokerFinancialTruthStoreErrorV1::new(
                        BrokerFinancialTruthStoreErrorCodeV1::UnsafeFilesystemEntry,
                        format!(
                            "broker truth store root {} is not a regular directory",
                            self.root.display()
                        ),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).map_err(|error| {
                    io_error(
                        format!(
                            "cannot create broker truth store root {}",
                            self.root.display()
                        ),
                        error,
                    )
                })?;
                let metadata = fs::symlink_metadata(&self.root).map_err(|error| {
                    io_error(
                        format!(
                            "cannot inspect created broker truth store root {}",
                            self.root.display()
                        ),
                        error,
                    )
                })?;
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(BrokerFinancialTruthStoreErrorV1::new(
                        BrokerFinancialTruthStoreErrorCodeV1::UnsafeFilesystemEntry,
                        "created broker truth store root is not a regular directory",
                    ));
                }
            }
            Err(error) => {
                return Err(io_error(
                    format!(
                        "cannot inspect broker truth store root {}",
                        self.root.display()
                    ),
                    error,
                ));
            }
        }
        Ok(())
    }

    fn create_staging_directory(&self) -> Result<PathBuf, BrokerFinancialTruthStoreErrorV1> {
        for _ in 0..32 {
            let clock = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| {
                    BrokerFinancialTruthStoreErrorV1::new(
                        BrokerFinancialTruthStoreErrorCodeV1::Io,
                        format!("system clock is before Unix epoch: {error}"),
                    )
                })?
                .as_nanos();
            let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = self.root.join(format!(
                ".bft1-staging-{}-{clock}-{nonce}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(io_error(
                        format!("cannot create staging directory {}", path.display()),
                        error,
                    ));
                }
            }
        }
        Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::PublishConflict,
            "cannot allocate a unique broker truth staging directory",
        ))
    }
}

fn validate_sources<'a>(
    artifacts: &[&ImmutableVortexArtifactV1],
    sources: &'a [BrokerFinancialTruthArtifactSourceV1],
) -> Result<HashMap<&'a str, &'a Path>, BrokerFinancialTruthStoreErrorV1> {
    let expected: HashSet<&str> = artifacts
        .iter()
        .map(|artifact| artifact.relative_path())
        .collect();
    let mut source_map = HashMap::new();
    for source in sources {
        validate_source_basename(&source.relative_path)?;
        if source_map
            .insert(source.relative_path.as_str(), source.source_path.as_path())
            .is_some()
        {
            return Err(BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::SourceMismatch,
                format!("duplicate source mapping for {}", source.relative_path),
            ));
        }
    }
    let received: HashSet<&str> = source_map.keys().copied().collect();
    if expected != received {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::SourceMismatch,
            "artifact source paths are not an exact set match for the manifest",
        ));
    }
    Ok(source_map)
}

fn validate_source_basename(value: &str) -> Result<(), BrokerFinancialTruthStoreErrorV1> {
    let path = Path::new(value);
    if path.file_name().and_then(|name| name.to_str()) != Some(value) || !value.ends_with(".vortex")
    {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::SourceMismatch,
            format!("source mapping {value:?} is not one .vortex basename"),
        ));
    }
    Ok(())
}

fn copy_exact_artifact(
    source: &Path,
    destination: &Path,
    expected: &ImmutableVortexArtifactV1,
) -> Result<(), BrokerFinancialTruthStoreErrorV1> {
    let source_metadata = fs::symlink_metadata(source).map_err(|error| {
        io_error(
            format!("cannot inspect artifact source {}", source.display()),
            error,
        )
    })?;
    if source_metadata.file_type().is_symlink() || !source_metadata.is_file() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!("artifact source {} is not a regular file", source.display()),
        ));
    }
    if source_metadata.len() != expected.byte_len() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::SourceMismatch,
            format!("source length changed for {}", expected.relative_path()),
        ));
    }
    let source_digest = sha256_file(source).map_err(contract_io_to_store)?;
    if source_digest != expected.sha256() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::SourceMismatch,
            format!("source digest changed for {}", expected.relative_path()),
        ));
    }

    let mut input = File::open(source)
        .map_err(|error| io_error(format!("cannot open source {}", source.display()), error))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| {
            io_error(
                format!("cannot create artifact {}", destination.display()),
                error,
            )
        })?;
    std::io::copy(&mut input, &mut output).map_err(|error| {
        io_error(
            format!(
                "cannot copy exact artifact {} to {}",
                source.display(),
                destination.display()
            ),
            error,
        )
    })?;
    output.sync_all().map_err(|error| {
        io_error(
            format!("cannot fsync artifact {}", destination.display()),
            error,
        )
    })?;
    validate_published_artifact(destination, expected)
}

fn write_new_file(
    destination: &Path,
    bytes: &[u8],
) -> Result<(), BrokerFinancialTruthStoreErrorV1> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| {
            io_error(
                format!("cannot create immutable file {}", destination.display()),
                error,
            )
        })?;
    output.write_all(bytes).map_err(|error| {
        io_error(
            format!("cannot write immutable file {}", destination.display()),
            error,
        )
    })?;
    output.sync_all().map_err(|error| {
        io_error(
            format!("cannot fsync immutable file {}", destination.display()),
            error,
        )
    })
}

fn validate_exact_file_set(
    bundle_root: &Path,
    artifacts: &[&ImmutableVortexArtifactV1],
) -> Result<(), BrokerFinancialTruthStoreErrorV1> {
    let mut expected: HashSet<String> = artifacts
        .iter()
        .map(|artifact| artifact.relative_path().to_owned())
        .collect();
    expected.insert(BROKER_FINANCIAL_TRUTH_MANIFEST_FILE_V1.to_owned());
    let mut received = HashSet::new();
    for entry in fs::read_dir(bundle_root).map_err(|error| {
        io_error(
            format!("cannot enumerate bundle {}", bundle_root.display()),
            error,
        )
    })? {
        let entry = entry.map_err(|error| {
            io_error(
                format!("cannot enumerate bundle {}", bundle_root.display()),
                error,
            )
        })?;
        let file_type = entry.file_type().map_err(|error| {
            io_error(
                format!("cannot inspect bundle entry {}", entry.path().display()),
                error,
            )
        })?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::UnsafeFilesystemEntry,
                format!(
                    "bundle entry {} is not a regular file",
                    entry.path().display()
                ),
            ));
        }
        let name = entry.file_name().into_string().map_err(|_| {
            BrokerFinancialTruthStoreErrorV1::new(
                BrokerFinancialTruthStoreErrorCodeV1::ArtifactSetMismatch,
                "bundle contains a non-UTF-8 file name",
            )
        })?;
        received.insert(name);
    }
    if expected != received {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::ArtifactSetMismatch,
            "bundle file set differs from the exact manifest",
        ));
    }
    Ok(())
}

fn validate_published_artifact(
    path: &Path,
    expected: &ImmutableVortexArtifactV1,
) -> Result<(), BrokerFinancialTruthStoreErrorV1> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| io_error(format!("cannot inspect artifact {}", path.display()), error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!("artifact {} is not a regular file", path.display()),
        ));
    }
    if metadata.len() != expected.byte_len() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::ArtifactLengthMismatch,
            format!(
                "artifact {} has length {}, expected {}",
                expected.relative_path(),
                metadata.len(),
                expected.byte_len()
            ),
        ));
    }
    let digest = sha256_file(path).map_err(contract_io_to_store)?;
    if digest != expected.sha256() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::ArtifactDigestMismatch,
            format!(
                "artifact {} has digest {}, expected {}",
                expected.relative_path(),
                digest,
                expected.sha256()
            ),
        ));
    }
    Ok(())
}

fn read_bounded_regular_file(
    path: &Path,
    maximum_bytes: u64,
) -> Result<Vec<u8>, BrokerFinancialTruthStoreErrorV1> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| io_error(format!("cannot inspect file {}", path.display()), error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!("{} is not a regular file", path.display()),
        ));
    }
    if metadata.len() > maximum_bytes {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::ManifestInvalid,
            format!("{} exceeds {maximum_bytes} bytes", path.display()),
        ));
    }
    let mut file = File::open(path)
        .map_err(|error| io_error(format!("cannot open file {}", path.display()), error))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| io_error(format!("cannot read file {}", path.display()), error))?;
    Ok(bytes)
}

fn ensure_regular_directory(path: &Path) -> Result<(), BrokerFinancialTruthStoreErrorV1> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        io_error(
            format!("cannot inspect bundle directory {}", path.display()),
            error,
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(BrokerFinancialTruthStoreErrorV1::new(
            BrokerFinancialTruthStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!("bundle path {} is not a regular directory", path.display()),
        ));
    }
    Ok(())
}

fn cleanup_exact_staging_directory(store_root: &Path, staging_root: &Path) {
    let Some(name) = staging_root.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    if staging_root.parent() == Some(store_root) && name.starts_with(".bft1-staging-") {
        let _ = fs::remove_dir_all(staging_root);
    }
}

fn contract_io_to_store(
    error: BrokerFinancialTruthContractErrorV1,
) -> BrokerFinancialTruthStoreErrorV1 {
    BrokerFinancialTruthStoreErrorV1::new(
        BrokerFinancialTruthStoreErrorCodeV1::Io,
        error.to_string(),
    )
}

fn io_error(context: impl Into<String>, error: std::io::Error) -> BrokerFinancialTruthStoreErrorV1 {
    BrokerFinancialTruthStoreErrorV1::new(
        BrokerFinancialTruthStoreErrorCodeV1::Io,
        format!("{}: {error}", context.into()),
    )
}
