use super::{
    MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1, MAX_PROMOTION_CANDIDATE_MODEL_FILE_COUNT_V1,
    MAX_PROMOTION_CANDIDATE_MODEL_TREE_BYTES_V1, PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1,
    PromotionCandidateModelArtifactV1, PromotionCandidateTrainingHandoffV1,
    PromotionCandidateTrainingManifestV1, PromotionCandidateTrainingRefusalCodeV1,
    PromotionCandidateTrainingRefusalV1, PromotionCandidateTrainingTerminalV1, refusal_v1,
};
use crate::TrainingRunSummary;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const EVIDENCE_SCHEMA_V1: &str = "neoethos.promotion-candidate-training-evidence.v1";
const EVIDENCE_VERSION_V1: u16 = 1;
const MODEL_TREE_HASH_DOMAIN_V1: &[u8] = b"neoethos.promotion-candidate-model-artifact-tree.v1\0";
const CANDIDATE_TREE_HASH_DOMAIN_V1: &[u8] = b"neoethos.promotion-candidate-installed-tree.v1\0";
const MAX_TREE_DEPTH_V1: usize = 32;
const MAX_CANONICAL_RELATIVE_PATH_BYTES_V1: usize = 4096;
const MAX_EVIDENCE_BYTES_V1: usize = MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 + 1024 * 1024;
const STAGING_PREFIX_V1: &str = ".promotion-candidate.tmp-v1-";

static NEXT_STAGING_V1: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize)]
struct TrainingEvidenceWriteV1<'a> {
    schema: &'static str,
    version: u16,
    handoff_identity_sha256: &'a str,
    handoff: &'a PromotionCandidateTrainingHandoffV1,
    planned_models: &'a [String],
    completed_models: &'a [String],
    failed_model_count: usize,
    model_artifacts: &'a [PromotionCandidateModelArtifactV1],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingEvidenceReadV1 {
    schema: String,
    version: u16,
    handoff_identity_sha256: String,
    handoff: PromotionCandidateTrainingHandoffV1,
    planned_models: Vec<String>,
    completed_models: Vec<String>,
    failed_model_count: usize,
    model_artifacts: Vec<PromotionCandidateModelArtifactV1>,
}

struct ScannedFileV1 {
    relative: String,
    path: PathBuf,
    expected_len: u64,
}

struct TreeScanV1 {
    files: Vec<ScannedFileV1>,
    directories: Vec<PathBuf>,
}

struct TreeHashV1 {
    sha256: String,
    file_count: u64,
    total_bytes: u64,
}

pub(super) fn create_staging_root_v1(
    candidate_root: &Path,
    data_root: &Path,
) -> Result<PathBuf, PromotionCandidateTrainingRefusalV1> {
    fs::create_dir_all(candidate_root).map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::CandidateRootInvalid,
            format!(
                "create candidate root {}: {error}",
                candidate_root.display()
            ),
        )
    })?;
    validate_physical_directory_v1(candidate_root, "candidate root")?;
    validate_physical_directory_v1(data_root, "exact dataset root")?;
    let candidate = fs::canonicalize(candidate_root).map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::CandidateRootInvalid,
            format!("canonicalize candidate root: {error}"),
        )
    })?;
    let data = fs::canonicalize(data_root).map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::CandidateRootInvalid,
            format!("canonicalize exact dataset root: {error}"),
        )
    })?;
    if candidate == data || candidate.starts_with(&data) || data.starts_with(&candidate) {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::CandidateRootInvalid,
            "candidate model root and exact dataset root must be separate, non-nested roots",
        ));
    }
    for _ in 0..1024 {
        let sequence = NEXT_STAGING_V1.fetch_add(1, Ordering::Relaxed);
        let staging = candidate_root.join(format!(
            "{STAGING_PREFIX_V1}{}-{sequence}",
            std::process::id()
        ));
        match fs::create_dir(&staging) {
            Ok(()) => {
                sync_directory_v1(candidate_root).map_err(|error| {
                    refusal_v1(
                        PromotionCandidateTrainingRefusalCodeV1::StagingFailed,
                        format!("sync candidate root after staging create: {error}"),
                    )
                })?;
                return Ok(staging);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::StagingFailed,
                    format!("create unique candidate staging directory: {error}"),
                ));
            }
        }
    }
    Err(refusal_v1(
        PromotionCandidateTrainingRefusalCodeV1::StagingFailed,
        "could not allocate a unique candidate staging directory",
    ))
}

pub(super) fn cleanup_staging_root_v1(candidate_root: &Path, staging: &Path) {
    if staging.parent() != Some(candidate_root)
        || !staging
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.starts_with(STAGING_PREFIX_V1))
    {
        return;
    }
    let Ok(metadata) = fs::symlink_metadata(staging) else {
        return;
    };
    if metadata.file_type().is_dir()
        && !metadata.file_type().is_symlink()
        && !metadata_is_reparse_point_v1(&metadata)
    {
        let _ = fs::remove_dir_all(staging);
        let _ = sync_directory_v1(candidate_root);
    }
}

pub(crate) fn install_promotion_candidate_model_tree_v1(
    candidate_root: &Path,
    staging: &Path,
    handoff: PromotionCandidateTrainingHandoffV1,
    summary: &TrainingRunSummary,
) -> PromotionCandidateTrainingTerminalV1 {
    match install_inner_v1(candidate_root, staging, handoff, summary) {
        Ok(terminal) => terminal,
        Err(error) => {
            cleanup_staging_root_v1(candidate_root, staging);
            PromotionCandidateTrainingTerminalV1::Refused(error)
        }
    }
}

fn install_inner_v1(
    candidate_root: &Path,
    staging: &Path,
    handoff: PromotionCandidateTrainingHandoffV1,
    summary: &TrainingRunSummary,
) -> Result<PromotionCandidateTrainingTerminalV1, PromotionCandidateTrainingRefusalV1> {
    validate_physical_directory_v1(candidate_root, "candidate root")?;
    validate_owned_staging_v1(candidate_root, staging)?;
    handoff.validate()?;
    if !summary.failed_models.is_empty() {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::ModelTrainingFailed,
            format!(
                "{} configured model(s) failed: {}",
                summary.failed_models.len(),
                summary
                    .failed_models
                    .iter()
                    .map(|failure| failure.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    let planned = handoff.training_config().planned_models();
    if summary.planned_models != planned || summary.completed_models != planned {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::ModelInventoryIncomplete,
            format!(
                "trained inventory planned={:?}, completed={:?}, expected={planned:?}",
                summary.planned_models, summary.completed_models
            ),
        ));
    }

    let symbol = handoff.canonical_series().anchor().identity().symbol_name();
    let model_artifacts =
        model_artifacts_v1(staging, symbol, handoff.base_timeframe().as_str(), planned)?;
    let handoff_identity_sha256 = handoff.identity_sha256()?;
    let evidence = TrainingEvidenceWriteV1 {
        schema: EVIDENCE_SCHEMA_V1,
        version: EVIDENCE_VERSION_V1,
        handoff_identity_sha256: &handoff_identity_sha256,
        handoff: &handoff,
        planned_models: &summary.planned_models,
        completed_models: &summary.completed_models,
        failed_model_count: 0,
        model_artifacts: &model_artifacts,
    };
    let evidence_bytes = serde_json::to_vec(&evidence).map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::EvidenceEncodingFailed,
            format!("encode exact candidate training evidence: {error}"),
        )
    })?;
    if evidence_bytes.len() > MAX_EVIDENCE_BYTES_V1 {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::EvidenceEncodingFailed,
            format!(
                "candidate training evidence is {} bytes",
                evidence_bytes.len()
            ),
        ));
    }
    let evidence_path = staging.join(PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1);
    let mut evidence_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&evidence_path)
        .map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::EvidenceEncodingFailed,
                format!("create candidate training evidence: {error}"),
            )
        })?;
    evidence_file.write_all(&evidence_bytes).map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::EvidenceEncodingFailed,
            format!("write candidate training evidence: {error}"),
        )
    })?;
    evidence_file.sync_all().map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::EvidenceEncodingFailed,
            format!("sync candidate training evidence: {error}"),
        )
    })?;
    drop(evidence_file);
    sync_tree_v1(staging)?;

    let candidate_tree = hash_tree_v1(staging, CANDIDATE_TREE_HASH_DOMAIN_V1)?;
    let evidence_sha256 = plain_sha256_v1(&evidence_bytes);
    let manifest = PromotionCandidateTrainingManifestV1::new_v1(
        handoff_identity_sha256,
        handoff.locked_portfolio().identity_sha256().to_owned(),
        handoff.training_config().runtime_config_sha256().to_owned(),
        handoff.training_config().model_config_sha256().to_owned(),
        candidate_tree.sha256,
        evidence_sha256,
        model_artifacts,
        candidate_tree.file_count,
        candidate_tree.total_bytes,
    );
    manifest.validate_shape_v1()?;
    let destination = candidate_root.join(manifest.candidate_relative_dir());
    match atomic_install_no_replace_v1(staging, &destination) {
        Ok(()) => {
            sync_directory_v1(candidate_root).map_err(|error| {
                refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::AtomicInstallFailed,
                    format!("sync candidate root after atomic install: {error}"),
                )
            })?;
            verify_installed_manifest_v1(candidate_root, &manifest)?;
            Ok(PromotionCandidateTrainingTerminalV1::Installed(manifest))
        }
        Err(error) if is_already_exists_v1(&error) => {
            let verification = verify_installed_manifest_v1(candidate_root, &manifest);
            cleanup_staging_root_v1(candidate_root, staging);
            verification.map_err(|verification_error| {
                refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::CandidateIdentityCollision,
                    format!(
                        "content-addressed candidate winner differs from staged bytes: {verification_error}"
                    ),
                )
            })?;
            Ok(PromotionCandidateTrainingTerminalV1::ExistingIdentical(
                manifest,
            ))
        }
        Err(error) if error.kind() == io::ErrorKind::Unsupported => Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::AtomicInstallUnsupported,
            format!("atomic candidate install is unsupported: {error}"),
        )),
        Err(error) => Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::AtomicInstallFailed,
            format!("atomic candidate install failed: {error}"),
        )),
    }
}

pub(super) fn verify_installed_manifest_v1(
    candidate_root: &Path,
    manifest: &PromotionCandidateTrainingManifestV1,
) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    validate_physical_directory_v1(candidate_root, "candidate root")?;
    manifest.validate_shape_v1()?;
    let candidate = candidate_root.join(manifest.candidate_relative_dir());
    validate_physical_directory_v1(&candidate, "installed candidate")?;
    let tree = hash_tree_v1(&candidate, CANDIDATE_TREE_HASH_DOMAIN_V1)?;
    if tree.sha256 != manifest.candidate_tree_sha256
        || tree.file_count != manifest.total_file_count
        || tree.total_bytes != manifest.total_bytes
    {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
            "installed candidate tree hash/count/bytes changed",
        ));
    }
    let evidence_bytes = read_bounded_file_v1(
        &candidate.join(PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1),
        MAX_EVIDENCE_BYTES_V1,
    )?;
    if plain_sha256_v1(&evidence_bytes) != manifest.evidence_sha256 {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
            "installed candidate evidence bytes changed",
        ));
    }
    let evidence = decode_evidence_v1(&evidence_bytes)?;
    validate_evidence_against_manifest_v1(&evidence, manifest)?;
    for expected in &manifest.model_artifacts {
        let model_root = candidate.join(parse_relative_path_v1(&expected.relative_dir)?);
        let actual = hash_tree_v1(&model_root, MODEL_TREE_HASH_DOMAIN_V1)?;
        if actual.sha256 != expected.tree_sha256
            || actual.file_count != expected.file_count
            || actual.total_bytes != expected.total_bytes
        {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
                format!("installed model tree {} changed", expected.model_name),
            ));
        }
    }
    Ok(())
}

pub(super) fn reopen_installed_handoff_v1(
    candidate_root: &Path,
    manifest: &PromotionCandidateTrainingManifestV1,
) -> Result<PromotionCandidateTrainingHandoffV1, PromotionCandidateTrainingRefusalV1> {
    verify_installed_manifest_v1(candidate_root, manifest)?;
    let path = candidate_root
        .join(manifest.candidate_relative_dir())
        .join(PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1);
    let bytes = read_bounded_file_v1(&path, MAX_EVIDENCE_BYTES_V1)?;
    let evidence = decode_evidence_v1(&bytes)?;
    validate_evidence_against_manifest_v1(&evidence, manifest)?;
    Ok(evidence.handoff)
}

fn validate_evidence_against_manifest_v1(
    evidence: &TrainingEvidenceReadV1,
    manifest: &PromotionCandidateTrainingManifestV1,
) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    evidence.handoff.validate()?;
    if evidence.schema != EVIDENCE_SCHEMA_V1
        || evidence.version != EVIDENCE_VERSION_V1
        || evidence.failed_model_count != 0
        || evidence.handoff.identity_sha256()? != evidence.handoff_identity_sha256
        || evidence.handoff_identity_sha256 != manifest.handoff_identity_sha256
        || evidence.handoff.locked_portfolio().identity_sha256()
            != manifest.locked_portfolio_identity_sha256
        || evidence.handoff.training_config().runtime_config_sha256()
            != manifest.runtime_config_sha256
        || evidence.handoff.training_config().model_config_sha256() != manifest.model_config_sha256
        || evidence.planned_models != evidence.completed_models
        || evidence.planned_models
            != evidence
                .model_artifacts
                .iter()
                .map(|artifact| artifact.model_name.clone())
                .collect::<Vec<_>>()
        || evidence.model_artifacts != manifest.model_artifacts
    {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
            "installed training evidence disagrees with its manifest or handoff",
        ));
    }
    Ok(())
}

fn decode_evidence_v1(
    bytes: &[u8],
) -> Result<TrainingEvidenceReadV1, PromotionCandidateTrainingRefusalV1> {
    serde_json::from_slice(bytes).map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
            format!("decode installed candidate evidence: {error}"),
        )
    })
}

fn model_artifacts_v1(
    staging: &Path,
    symbol: &str,
    base_timeframe: &str,
    models: &[String],
) -> Result<Vec<PromotionCandidateModelArtifactV1>, PromotionCandidateTrainingRefusalV1> {
    let mut artifacts = Vec::with_capacity(models.len());
    for model_name in models {
        let relative = PathBuf::from(symbol).join(base_timeframe).join(model_name);
        let relative_dir = canonical_relative_path_v1(&relative)?;
        let tree = hash_tree_v1(&staging.join(&relative), MODEL_TREE_HASH_DOMAIN_V1)?;
        artifacts.push(PromotionCandidateModelArtifactV1 {
            model_name: model_name.clone(),
            relative_dir,
            tree_sha256: tree.sha256,
            file_count: tree.file_count,
            total_bytes: tree.total_bytes,
        });
    }
    Ok(artifacts)
}

fn hash_tree_v1(
    root: &Path,
    domain: &[u8],
) -> Result<TreeHashV1, PromotionCandidateTrainingRefusalV1> {
    let scan = scan_tree_v1(root)?;
    if scan.files.is_empty() {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::ArtifactTreeInvalid,
            format!("artifact tree {} is empty", root.display()),
        ));
    }
    let mut tree = Sha256::new();
    tree.update(domain);
    let mut total_bytes = 0_u64;
    for entry in &scan.files {
        let before = physical_file_metadata_v1(&entry.path)?;
        if before.len() != entry.expected_len {
            return Err(tree_changed_v1(
                "artifact file length changed before hashing",
            ));
        }
        let mut file = File::open(&entry.path).map_err(|error| {
            tree_error_v1(format!(
                "open artifact file {}: {error}",
                entry.path.display()
            ))
        })?;
        let mut file_hash = Sha256::new();
        let mut observed_len = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|error| {
                tree_error_v1(format!(
                    "read artifact file {}: {error}",
                    entry.path.display()
                ))
            })?;
            if read == 0 {
                break;
            }
            observed_len = observed_len.checked_add(read as u64).ok_or_else(|| {
                refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::ArtifactTreeLimitExceeded,
                    "artifact file length overflow",
                )
            })?;
            file_hash.update(&buffer[..read]);
        }
        let after = physical_file_metadata_v1(&entry.path)?;
        if observed_len != entry.expected_len || after.len() != entry.expected_len {
            return Err(tree_changed_v1("artifact file changed while hashing"));
        }
        total_bytes = total_bytes.checked_add(observed_len).ok_or_else(|| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::ArtifactTreeLimitExceeded,
                "artifact tree byte count overflow",
            )
        })?;
        if total_bytes > MAX_PROMOTION_CANDIDATE_MODEL_TREE_BYTES_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::ArtifactTreeLimitExceeded,
                "artifact tree exceeds the 64 GiB bound",
            ));
        }
        tree.update((entry.relative.len() as u64).to_le_bytes());
        tree.update(entry.relative.as_bytes());
        tree.update(observed_len.to_le_bytes());
        tree.update(file_hash.finalize());
    }
    Ok(TreeHashV1 {
        sha256: format!("{:x}", tree.finalize()),
        file_count: scan.files.len() as u64,
        total_bytes,
    })
}

fn scan_tree_v1(root: &Path) -> Result<TreeScanV1, PromotionCandidateTrainingRefusalV1> {
    validate_physical_directory_v1(root, "artifact root")?;
    let mut files = Vec::new();
    let mut directories = vec![root.to_path_buf()];
    collect_tree_v1(root, root, 0, &mut files, &mut directories)?;
    files.sort_by(|left, right| left.relative.cmp(&right.relative));
    Ok(TreeScanV1 { files, directories })
}

fn collect_tree_v1(
    root: &Path,
    current: &Path,
    depth: usize,
    files: &mut Vec<ScannedFileV1>,
    directories: &mut Vec<PathBuf>,
) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    if depth > MAX_TREE_DEPTH_V1 {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::ArtifactTreeLimitExceeded,
            "artifact tree exceeds the depth bound",
        ));
    }
    let mut entries = fs::read_dir(current)
        .map_err(|error| tree_error_v1(format!("read artifact directory: {error}")))?
        .collect::<io::Result<Vec<_>>>()
        .map_err(|error| tree_error_v1(format!("enumerate artifact directory: {error}")))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        if files.len() + directories.len() >= MAX_PROMOTION_CANDIDATE_MODEL_FILE_COUNT_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::ArtifactTreeLimitExceeded,
                "artifact tree exceeds the entry-count bound",
            ));
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| tree_error_v1(format!("inspect artifact entry: {error}")))?;
        if metadata.file_type().is_symlink() || metadata_is_reparse_point_v1(&metadata) {
            return Err(tree_error_v1(
                "artifact tree contains a symlink or reparse point",
            ));
        }
        if metadata.file_type().is_dir() {
            directories.push(path.clone());
            collect_tree_v1(root, &path, depth + 1, files, directories)?;
        } else if metadata.file_type().is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| tree_error_v1("artifact entry escaped its root"))?;
            files.push(ScannedFileV1 {
                relative: canonical_relative_path_v1(relative)?,
                path,
                expected_len: metadata.len(),
            });
        } else {
            return Err(tree_error_v1("artifact tree contains a non-file entry"));
        }
    }
    Ok(())
}

fn sync_tree_v1(root: &Path) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    let mut scan = scan_tree_v1(root)?;
    for entry in &scan.files {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&entry.path)
            .map_err(|error| tree_error_v1(format!("open artifact for sync: {error}")))?;
        file.sync_all()
            .map_err(|error| tree_error_v1(format!("sync artifact file: {error}")))?;
    }
    scan.directories.reverse();
    for directory in scan.directories {
        sync_directory_v1(&directory)
            .map_err(|error| tree_error_v1(format!("sync artifact directory: {error}")))?;
    }
    Ok(())
}

fn read_bounded_file_v1(
    path: &Path,
    maximum: usize,
) -> Result<Vec<u8>, PromotionCandidateTrainingRefusalV1> {
    let metadata = physical_file_metadata_v1(path)?;
    if metadata.len() > maximum as u64 {
        return Err(tree_changed_v1("installed evidence exceeds its byte bound"));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| tree_error_v1(format!("read installed evidence: {error}")))?;
    if bytes.len() > maximum || bytes.len() as u64 != metadata.len() {
        return Err(tree_changed_v1(
            "installed evidence changed while reopening",
        ));
    }
    Ok(bytes)
}

fn validate_owned_staging_v1(
    candidate_root: &Path,
    staging: &Path,
) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    if staging.parent() != Some(candidate_root)
        || !staging
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.starts_with(STAGING_PREFIX_V1))
    {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::StagingFailed,
            "candidate staging is not one owned direct child of the candidate root",
        ));
    }
    validate_physical_directory_v1(staging, "candidate staging")
}

fn validate_physical_directory_v1(
    path: &Path,
    label: &str,
) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::CandidateRootInvalid,
            format!("inspect {label} {}: {error}", path.display()),
        )
    })?;
    if !metadata.file_type().is_dir()
        || metadata.file_type().is_symlink()
        || metadata_is_reparse_point_v1(&metadata)
    {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::CandidateRootInvalid,
            format!("{label} is not one physical directory"),
        ));
    }
    Ok(())
}

fn physical_file_metadata_v1(
    path: &Path,
) -> Result<fs::Metadata, PromotionCandidateTrainingRefusalV1> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| tree_error_v1(format!("inspect artifact file: {error}")))?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata_is_reparse_point_v1(&metadata)
    {
        return Err(tree_error_v1("artifact entry is not one physical file"));
    }
    Ok(metadata)
}

fn canonical_relative_path_v1(path: &Path) -> Result<String, PromotionCandidateTrainingRefusalV1> {
    let mut parts = Vec::new();
    for component in path.components() {
        let Component::Normal(value) = component else {
            return Err(tree_error_v1("artifact relative path is not canonical"));
        };
        let value = value
            .to_str()
            .ok_or_else(|| tree_error_v1("artifact relative path is not UTF-8"))?;
        if value.is_empty()
            || value == "."
            || value == ".."
            || value.chars().any(char::is_control)
            || value.contains(['/', '\\'])
        {
            return Err(tree_error_v1("artifact path contains an unsafe component"));
        }
        parts.push(value);
    }
    let relative = parts.join("/");
    if relative.is_empty() || relative.len() > MAX_CANONICAL_RELATIVE_PATH_BYTES_V1 {
        return Err(tree_error_v1("artifact relative path is empty or too long"));
    }
    Ok(relative)
}

fn parse_relative_path_v1(value: &str) -> Result<PathBuf, PromotionCandidateTrainingRefusalV1> {
    let path = value
        .split('/')
        .fold(PathBuf::new(), |path, part| path.join(part));
    if canonical_relative_path_v1(&path)? != value {
        return Err(tree_changed_v1(
            "manifest contains a non-canonical relative path",
        ));
    }
    Ok(path)
}

fn plain_sha256_v1(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn tree_error_v1(detail: impl Into<String>) -> PromotionCandidateTrainingRefusalV1 {
    refusal_v1(
        PromotionCandidateTrainingRefusalCodeV1::ArtifactTreeInvalid,
        detail,
    )
}

fn tree_changed_v1(detail: impl Into<String>) -> PromotionCandidateTrainingRefusalV1 {
    refusal_v1(
        PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
        detail,
    )
}

fn is_already_exists_v1(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::AlreadyExists
        || matches!(error.raw_os_error(), Some(17 | 39 | 80 | 183))
}

#[cfg(target_os = "linux")]
fn atomic_install_no_replace_v1(source: &Path, destination: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let source = CString::new(source.as_os_str().as_bytes())?;
    let destination = CString::new(destination.as_os_str().as_bytes())?;
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(windows)]
fn atomic_install_no_replace_v1(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }
    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let result = unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), 0x0000_0008) };
    if result != 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn atomic_install_no_replace_v1(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no reviewed directory no-replace primitive exists for this platform",
    ))
}

#[cfg(not(windows))]
fn sync_directory_v1(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(windows)]
fn sync_directory_v1(path: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    type Handle = *mut std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *mut std::ffi::c_void,
            disposition: u32,
            flags: u32,
            template: Handle,
        ) -> Handle;
        fn FlushFileBuffers(handle: Handle) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            0x8000_0000,
            0x0000_0001 | 0x0000_0002 | 0x0000_0004,
            std::ptr::null_mut(),
            3,
            0x0200_0000,
            std::ptr::null_mut(),
        )
    };
    if handle == (-1_isize as Handle) {
        return Err(io::Error::last_os_error());
    }
    let result = unsafe { FlushFileBuffers(handle) };
    let error = (result == 0).then(io::Error::last_os_error);
    unsafe {
        CloseHandle(handle);
    }
    match error {
        Some(error) if error.raw_os_error() == Some(5) => Ok(()),
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(windows)]
fn metadata_is_reparse_point_v1(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    metadata.file_attributes() & 0x400 != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse_point_v1(_metadata: &fs::Metadata) -> bool {
    false
}
