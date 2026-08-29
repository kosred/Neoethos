use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::artifact_io::{fnv1a64, stable_json_hash, write_json_atomic};
use crate::data_selection::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1,
};
use crate::discovery::{
    DiscoveryPerKindEvidenceHashes, DiscoveryResult, PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
    PromotionSummaryAuthorityPayloadV3, build_promotion_summary_envelope,
};
use crate::genetic::Gene;
use crate::validation::{
    CANONICAL_BACKTEST_ARTIFACT_KIND, CanonicalBacktestArtifactFile,
    FORWARD_TEST_VALIDATION_ARTIFACT_KIND, ForwardTestValidationArtifactFile,
    PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND, PropFirmRiskValidationArtifactFile,
    ValidationStrategyIdentityV2, WALKFORWARD_VALIDATION_ARTIFACT_KIND,
    WalkforwardValidationArtifactFile,
};

pub const DISCOVERY_VALIDATION_SNAPSHOT_POINTER_SCHEMA_VERSION: u32 = 1;
pub const DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_SCHEMA_VERSION: u32 = 2;
pub const DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_KIND: &str =
    "neoethos.search.validation-snapshot-manifest.v2";

const CURRENT_FILE_NAME: &str = "CURRENT.json";
const GENERATIONS_DIR_NAME: &str = "generations";
const MANIFEST_FILE_NAME: &str = "manifest.json";
const PROMOTION_SUMMARY_FILE_NAME: &str = "promotion_summary.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryValidationSnapshotPointerV1 {
    schema_version: u32,
    generation_id: String,
    manifest_hash: String,
}

impl DiscoveryValidationSnapshotPointerV1 {
    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    pub fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == DISCOVERY_VALIDATION_SNAPSHOT_POINTER_SCHEMA_VERSION,
            "unsupported validation snapshot CURRENT schema version {}; expected {}",
            self.schema_version,
            DISCOVERY_VALIDATION_SNAPSHOT_POINTER_SCHEMA_VERSION
        );
        validate_generation_id(&self.generation_id)?;
        validate_fnv64_hash("validation snapshot manifest hash", &self.manifest_hash)?;
        ensure!(
            self.generation_id == generation_id_for_hash(&self.manifest_hash)?,
            "validation snapshot generation id `{}` does not match manifest hash `{}`",
            self.generation_id,
            self.manifest_hash
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryValidationSnapshotMemberV1 {
    artifact_kind: String,
    relative_path: String,
    content_hash: String,
    byte_hash: String,
}

impl DiscoveryValidationSnapshotMemberV1 {
    pub fn artifact_kind(&self) -> &str {
        &self.artifact_kind
    }

    pub fn relative_path(&self) -> &Path {
        Path::new(&self.relative_path)
    }

    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }

    pub fn byte_hash(&self) -> &str {
        &self.byte_hash
    }

    fn validate(&self, expected_kind: &str, expected_path: &str) -> Result<()> {
        ensure!(
            self.artifact_kind == expected_kind,
            "snapshot member kind `{}` does not match expected `{expected_kind}`",
            self.artifact_kind
        );
        validate_snapshot_relative_path(Path::new(&self.relative_path))?;
        ensure!(
            self.relative_path == expected_path,
            "snapshot member path `{}` does not match expected `{expected_path}`",
            self.relative_path
        );
        validate_fnv64_hash("snapshot member content hash", &self.content_hash)?;
        validate_fnv64_hash("snapshot member byte hash", &self.byte_hash)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryValidationSnapshotStrategyV1 {
    strategy_identity: ValidationStrategyIdentityV2,
    gene: Gene,
    canonical_backtest: DiscoveryValidationSnapshotMemberV1,
    walkforward: DiscoveryValidationSnapshotMemberV1,
    forward_test: DiscoveryValidationSnapshotMemberV1,
    prop_firm: DiscoveryValidationSnapshotMemberV1,
}

impl DiscoveryValidationSnapshotStrategyV1 {
    pub fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        &self.strategy_identity
    }

    pub fn gene(&self) -> &Gene {
        &self.gene
    }

    pub fn members(&self) -> [&DiscoveryValidationSnapshotMemberV1; 4] {
        [
            &self.canonical_backtest,
            &self.walkforward,
            &self.forward_test,
            &self.prop_firm,
        ]
    }

    fn validate(&self) -> Result<()> {
        self.strategy_identity.validate_against(&self.gene)?;
        let leaf = strategy_file_leaf(self.strategy_identity.exact_gene_hash())?;
        self.canonical_backtest.validate(
            CANONICAL_BACKTEST_ARTIFACT_KIND,
            &format!("canonical/{leaf}.json"),
        )?;
        self.walkforward.validate(
            WALKFORWARD_VALIDATION_ARTIFACT_KIND,
            &format!("walkforward/{leaf}.json"),
        )?;
        self.forward_test.validate(
            FORWARD_TEST_VALIDATION_ARTIFACT_KIND,
            &format!("forward_test/{leaf}.json"),
        )?;
        self.prop_firm.validate(
            PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND,
            &format!("prop_firm/{leaf}.json"),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryValidationSnapshotManifestV2 {
    schema_version: u32,
    holdout_scope: CanonicalSearchArtifactScopeV2,
    strategies: Vec<DiscoveryValidationSnapshotStrategyV1>,
    promotion_summary: DiscoveryValidationSnapshotMemberV1,
}

impl DiscoveryValidationSnapshotManifestV2 {
    pub fn holdout_scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.holdout_scope
    }

    pub fn strategies(&self) -> &[DiscoveryValidationSnapshotStrategyV1] {
        &self.strategies
    }

    pub fn promotion_summary(&self) -> &DiscoveryValidationSnapshotMemberV1 {
        &self.promotion_summary
    }

    pub fn all_members(&self) -> Vec<&DiscoveryValidationSnapshotMemberV1> {
        let mut members = Vec::with_capacity(self.strategies.len() * 4 + 1);
        for strategy in &self.strategies {
            members.extend(strategy.members());
        }
        members.push(&self.promotion_summary);
        members
    }

    fn validate(&self, selection_scope: &CanonicalSearchArtifactScopeV2) -> Result<()> {
        ensure!(
            self.schema_version == DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            "unsupported validation snapshot manifest schema version {}; expected {}",
            self.schema_version,
            DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        );
        validate_scope_pair(selection_scope, &self.holdout_scope)?;
        ensure!(
            !self.strategies.is_empty(),
            "validation snapshot manifest has no final strategies"
        );
        let mut previous: Option<(&str, &str)> = None;
        let mut strategy_ids = HashSet::new();
        let mut exact_hashes = HashSet::new();
        let mut paths = BTreeSet::new();
        for strategy in &self.strategies {
            strategy.validate()?;
            let identity = strategy.strategy_identity();
            if let Some((previous_hash, previous_id)) = previous {
                ensure!(
                    (previous_hash, previous_id)
                        < (identity.exact_gene_hash(), identity.strategy_id()),
                    "validation snapshot strategies are not strictly sorted by exact identity"
                );
            }
            previous = Some((identity.exact_gene_hash(), identity.strategy_id()));
            ensure!(
                strategy_ids.insert(identity.strategy_id()),
                "validation snapshot contains duplicate strategy_id `{}`",
                identity.strategy_id()
            );
            ensure!(
                exact_hashes.insert(identity.exact_gene_hash()),
                "validation snapshot contains duplicate exact gene hash `{}`",
                identity.exact_gene_hash()
            );
            for member in strategy.members() {
                ensure!(
                    paths.insert(member.relative_path().to_path_buf()),
                    "validation snapshot contains duplicate member path {}",
                    member.relative_path().display()
                );
            }
        }
        self.promotion_summary.validate(
            PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
            PROMOTION_SUMMARY_FILE_NAME,
        )?;
        ensure!(
            paths.insert(self.promotion_summary.relative_path().to_path_buf()),
            "validation snapshot contains duplicate promotion-summary path"
        );
        Ok(())
    }
}

pub type DiscoveryValidationSnapshotManifestEnvelopeV2 =
    CanonicalSearchArtifactEnvelopeV2<DiscoveryValidationSnapshotManifestV2>;

#[derive(Debug, Clone)]
pub struct ValidatedDiscoveryValidationSnapshotV2 {
    pointer: DiscoveryValidationSnapshotPointerV1,
    manifest: DiscoveryValidationSnapshotManifestEnvelopeV2,
    generation_dir: PathBuf,
}

impl ValidatedDiscoveryValidationSnapshotV2 {
    pub fn pointer(&self) -> &DiscoveryValidationSnapshotPointerV1 {
        &self.pointer
    }

    pub fn manifest(&self) -> &DiscoveryValidationSnapshotManifestEnvelopeV2 {
        &self.manifest
    }

    pub fn generation_dir(&self) -> &Path {
        &self.generation_dir
    }

    pub fn validate_against(&self, result: &DiscoveryResult) -> Result<()> {
        let expected = build_snapshot_plan(result)?;
        ensure!(
            self.pointer == expected.pointer,
            "loaded validation snapshot pointer does not match the exact DiscoveryResult"
        );
        ensure!(
            self.manifest == expected.manifest,
            "loaded validation snapshot manifest does not match the exact DiscoveryResult"
        );
        Ok(())
    }
}

#[derive(Debug)]
struct SnapshotPlan {
    pointer: DiscoveryValidationSnapshotPointerV1,
    manifest: DiscoveryValidationSnapshotManifestEnvelopeV2,
    files: BTreeMap<String, Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapshotFault {
    None,
    AfterFirstMemberWrite,
    BeforeCurrentSwap,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValidationSnapshotTestFault {
    AfterFirstMemberWrite,
    BeforeCurrentSwap,
}

pub fn save_discovery_validation_snapshot(
    root: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<DiscoveryValidationSnapshotPointerV1> {
    save_snapshot_with_fault(root.as_ref(), result, SnapshotFault::None)
}

#[cfg(test)]
pub(crate) fn save_discovery_validation_snapshot_with_test_fault(
    root: impl AsRef<Path>,
    result: &DiscoveryResult,
    fault: ValidationSnapshotTestFault,
) -> Result<DiscoveryValidationSnapshotPointerV1> {
    let fault = match fault {
        ValidationSnapshotTestFault::AfterFirstMemberWrite => SnapshotFault::AfterFirstMemberWrite,
        ValidationSnapshotTestFault::BeforeCurrentSwap => SnapshotFault::BeforeCurrentSwap,
    };
    save_snapshot_with_fault(root.as_ref(), result, fault)
}

fn save_snapshot_with_fault(
    root: &Path,
    result: &DiscoveryResult,
    fault: SnapshotFault,
) -> Result<DiscoveryValidationSnapshotPointerV1> {
    let plan = build_snapshot_plan(result)?;
    prepare_snapshot_root(root)?;
    let generations_dir = root.join(GENERATIONS_DIR_NAME);
    let generation_dir = generations_dir.join(plan.pointer.generation_id());

    if generation_dir.exists() {
        verify_existing_generation(&generation_dir, &plan)?;
    } else {
        let staging_dir = create_unique_staging_dir(&generations_dir)?;
        let write_result = (|| -> Result<()> {
            let mut wrote_first_member = false;
            for (relative, bytes) in &plan.files {
                let path = staging_dir.join(relative);
                write_snapshot_bytes(&path, bytes)?;
                if relative != MANIFEST_FILE_NAME && !wrote_first_member {
                    wrote_first_member = true;
                    if fault == SnapshotFault::AfterFirstMemberWrite {
                        bail!(
                            "injected validation snapshot failure after the first staged member write"
                        );
                    }
                }
            }
            verify_generation_tree(&staging_dir, &plan)?;
            match fs::rename(&staging_dir, &generation_dir) {
                Ok(()) => verify_generation_tree(&generation_dir, &plan)?,
                Err(rename_error) if generation_dir.exists() => {
                    verify_existing_generation(&generation_dir, &plan).with_context(|| {
                        format!(
                            "content-address race after rename failed ({rename_error}); existing generation is not exact"
                        )
                    })?;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "commit staged validation snapshot {} to immutable generation {}",
                            staging_dir.display(),
                            generation_dir.display()
                        )
                    });
                }
            }
            Ok(())
        })();
        write_result?;
    }

    if fault == SnapshotFault::BeforeCurrentSwap {
        bail!("injected validation snapshot failure before CURRENT swap");
    }
    plan.pointer.validate()?;
    let current_path = root.join(CURRENT_FILE_NAME);
    match fs::symlink_metadata(&current_path) {
        Ok(_) => {
            ensure_regular_file_without_symlink(&current_path, "validation snapshot CURRENT")?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "read validation snapshot CURRENT metadata at {}",
                    current_path.display()
                )
            });
        }
    }
    write_json_atomic(&current_path, &plan.pointer)
        .context("atomically commit validation snapshot CURRENT pointer")?;
    Ok(plan.pointer)
}

pub fn load_discovery_validation_snapshot(
    root: impl AsRef<Path>,
) -> Result<ValidatedDiscoveryValidationSnapshotV2> {
    let root = root.as_ref();
    ensure_existing_directory_without_symlink(root, "validation snapshot root")?;
    let generations_dir = root.join(GENERATIONS_DIR_NAME);
    ensure_existing_directory_without_symlink(
        &generations_dir,
        "validation snapshot generations directory",
    )?;
    let current_path = root.join(CURRENT_FILE_NAME);
    ensure_regular_file_without_symlink(&current_path, "validation snapshot CURRENT")?;
    let pointer_bytes = fs::read(&current_path).with_context(|| {
        format!(
            "read validation snapshot CURRENT {}",
            current_path.display()
        )
    })?;
    let pointer: DiscoveryValidationSnapshotPointerV1 = serde_json::from_slice(&pointer_bytes)
        .map_err(|error| {
            anyhow!(
                "parse strict validation snapshot CURRENT {}: {error}",
                current_path.display()
            )
        })?;
    pointer.validate()?;

    let generation_dir = generations_dir.join(pointer.generation_id());
    ensure_existing_directory_without_symlink(&generation_dir, "validation snapshot generation")?;
    let manifest_path =
        resolve_snapshot_member_without_symlinks(&generation_dir, Path::new(MANIFEST_FILE_NAME))?;
    let manifest_bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "read validation snapshot manifest {}",
            manifest_path.display()
        )
    })?;
    let manifest: DiscoveryValidationSnapshotManifestEnvelopeV2 =
        CanonicalSearchArtifactEnvelopeV2::from_json_bytes(&manifest_bytes)
            .map_err(anyhow::Error::new)?;
    let canonical_manifest_bytes = pretty_json_bytes(&manifest)?;
    ensure!(
        manifest_bytes == canonical_manifest_bytes,
        "validation snapshot manifest bytes are not the canonical committed encoding"
    );
    ensure!(
        manifest.artifact_kind() == DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_KIND,
        "wrong validation snapshot manifest kind `{}`",
        manifest.artifact_kind()
    );
    manifest.payload().validate(manifest.scope())?;
    let manifest_value = serde_json::to_value(&manifest).context("serialize snapshot manifest")?;
    let manifest_hash = stable_json_hash(&manifest_value)?;
    ensure!(
        manifest_hash == pointer.manifest_hash,
        "validation snapshot manifest hash `{manifest_hash}` does not match CURRENT `{}`",
        pointer.manifest_hash
    );

    validate_loaded_generation(&generation_dir, &manifest)?;
    Ok(ValidatedDiscoveryValidationSnapshotV2 {
        pointer,
        manifest,
        generation_dir,
    })
}

fn build_snapshot_plan(result: &DiscoveryResult) -> Result<SnapshotPlan> {
    result.validate_complete_promotion_evidence()?;
    let selection_scope = result.selection_scope()?.clone();
    let holdout_scope = result
        .holdout_scope()?
        .ok_or_else(|| anyhow!("validation snapshot requires the exact stored holdout scope"))?
        .clone();
    validate_scope_pair(&selection_scope, &holdout_scope)?;

    let mut genes = result
        .portfolio
        .iter()
        .map(|gene| Ok((ValidationStrategyIdentityV2::from_gene(gene)?, gene)))
        .collect::<Result<Vec<_>>>()?;
    genes.sort_by(|(left_identity, _), (right_identity, _)| {
        left_identity
            .exact_gene_hash()
            .cmp(right_identity.exact_gene_hash())
            .then_with(|| {
                left_identity
                    .strategy_id()
                    .cmp(right_identity.strategy_id())
            })
    });

    let mut files = BTreeMap::new();
    let mut strategies = Vec::with_capacity(genes.len());
    for (identity, gene) in genes {
        let exact_hash = identity.exact_gene_hash();
        let leaf = strategy_file_leaf(exact_hash)?;
        let canonical = find_exact_artifact(
            &result.canonical_backtest_artifacts,
            exact_hash,
            "canonical backtest",
            |artifact| artifact.strategy_identity(),
        )?;
        let walkforward = find_exact_artifact(
            &result.walkforward_validation_artifacts,
            exact_hash,
            "walkforward",
            |artifact| artifact.strategy_identity(),
        )?;
        let forward_test = find_exact_artifact(
            &result.forward_test_validation_artifacts,
            exact_hash,
            "forward test",
            |artifact| artifact.strategy_identity(),
        )?;
        let prop_firm = find_exact_artifact(
            &result.prop_firm_validation_artifacts,
            exact_hash,
            "prop firm",
            |artifact| artifact.strategy_identity(),
        )?;

        canonical.validate_against(&selection_scope, &result.search_config_hash, gene)?;
        walkforward.validate_against(&selection_scope, &result.search_config_hash, gene)?;
        forward_test.validate_against(&holdout_scope, &result.search_config_hash, gene)?;
        prop_firm.validate_against(&holdout_scope, &result.search_config_hash, gene)?;

        let canonical_member = plan_member(
            CANONICAL_BACKTEST_ARTIFACT_KIND,
            format!("canonical/{leaf}.json"),
            canonical,
            &mut files,
        )?;
        let walkforward_member = plan_member(
            WALKFORWARD_VALIDATION_ARTIFACT_KIND,
            format!("walkforward/{leaf}.json"),
            walkforward,
            &mut files,
        )?;
        let forward_member = plan_member(
            FORWARD_TEST_VALIDATION_ARTIFACT_KIND,
            format!("forward_test/{leaf}.json"),
            forward_test,
            &mut files,
        )?;
        let prop_member = plan_member(
            PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND,
            format!("prop_firm/{leaf}.json"),
            prop_firm,
            &mut files,
        )?;
        strategies.push(DiscoveryValidationSnapshotStrategyV1 {
            strategy_identity: identity,
            gene: gene.clone(),
            canonical_backtest: canonical_member,
            walkforward: walkforward_member,
            forward_test: forward_member,
            prop_firm: prop_member,
        });
    }

    let promotion_summary = build_promotion_summary_envelope(result)?;
    let promotion_member = plan_member(
        PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
        PROMOTION_SUMMARY_FILE_NAME.to_owned(),
        &promotion_summary,
        &mut files,
    )?;
    let payload = DiscoveryValidationSnapshotManifestV2 {
        schema_version: DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        holdout_scope,
        strategies,
        promotion_summary: promotion_member,
    };
    let manifest = CanonicalSearchArtifactEnvelopeV2::new(
        DISCOVERY_VALIDATION_SNAPSHOT_MANIFEST_KIND,
        selection_scope,
        result.search_config_hash.clone(),
        payload,
    )
    .map_err(anyhow::Error::new)?;
    manifest.payload().validate(manifest.scope())?;
    let manifest_value = serde_json::to_value(&manifest).context("serialize snapshot manifest")?;
    let manifest_hash = stable_json_hash(&manifest_value)?;
    let pointer = DiscoveryValidationSnapshotPointerV1 {
        schema_version: DISCOVERY_VALIDATION_SNAPSHOT_POINTER_SCHEMA_VERSION,
        generation_id: generation_id_for_hash(&manifest_hash)?,
        manifest_hash,
    };
    pointer.validate()?;
    files.insert(MANIFEST_FILE_NAME.to_owned(), pretty_json_bytes(&manifest)?);
    Ok(SnapshotPlan {
        pointer,
        manifest,
        files,
    })
}

fn find_exact_artifact<'a, T, F>(
    artifacts: &'a [T],
    exact_hash: &str,
    label: &str,
    identity: F,
) -> Result<&'a T>
where
    F: Fn(&T) -> &ValidationStrategyIdentityV2,
{
    let matches = artifacts
        .iter()
        .filter(|artifact| identity(artifact).exact_gene_hash() == exact_hash)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "validation snapshot requires exactly one {label} artifact for {exact_hash}; found {}",
        matches.len()
    );
    Ok(matches[0])
}

fn plan_member<T: Serialize + ?Sized>(
    artifact_kind: &str,
    relative_path: String,
    value: &T,
    files: &mut BTreeMap<String, Vec<u8>>,
) -> Result<DiscoveryValidationSnapshotMemberV1> {
    validate_snapshot_relative_path(Path::new(&relative_path))?;
    let bytes = pretty_json_bytes(value)?;
    let member = DiscoveryValidationSnapshotMemberV1 {
        artifact_kind: artifact_kind.to_owned(),
        relative_path: relative_path.clone(),
        content_hash: stable_json_hash(value)?,
        byte_hash: hash_bytes(&bytes),
    };
    ensure!(
        files.insert(relative_path.clone(), bytes).is_none(),
        "duplicate planned validation snapshot path `{relative_path}`"
    );
    Ok(member)
}

fn validate_loaded_generation(
    generation_dir: &Path,
    manifest: &DiscoveryValidationSnapshotManifestEnvelopeV2,
) -> Result<()> {
    let expected_paths = manifest
        .payload()
        .all_members()
        .into_iter()
        .map(|member| member.relative_path().to_path_buf())
        .chain(std::iter::once(PathBuf::from(MANIFEST_FILE_NAME)))
        .collect::<BTreeSet<_>>();
    let actual_paths = collect_generation_files(generation_dir)?;
    let actual_set = actual_paths.keys().cloned().collect::<BTreeSet<_>>();
    if let Some(missing) = expected_paths.difference(&actual_set).next() {
        bail!(
            "validation snapshot generation is missing member {}",
            missing.display()
        );
    }
    if let Some(extra) = actual_set.difference(&expected_paths).next() {
        bail!(
            "validation snapshot generation contains extra member {}",
            extra.display()
        );
    }

    let mut canonical = Vec::new();
    let mut walkforward = Vec::new();
    let mut forward_test = Vec::new();
    let mut prop_firm = Vec::new();
    for strategy in manifest.payload().strategies() {
        let members = strategy.members();
        let canonical_artifact: CanonicalBacktestArtifactFile = load_typed_member(
            generation_dir,
            members[0],
            CanonicalBacktestArtifactFile::from_json_bytes,
        )?;
        canonical_artifact.validate_against(
            manifest.scope(),
            manifest.search_config_hash(),
            strategy.gene(),
        )?;
        canonical.push(canonical_artifact);

        let walkforward_artifact: WalkforwardValidationArtifactFile = load_typed_member(
            generation_dir,
            members[1],
            WalkforwardValidationArtifactFile::from_json_bytes,
        )?;
        walkforward_artifact.validate_against(
            manifest.scope(),
            manifest.search_config_hash(),
            strategy.gene(),
        )?;
        walkforward.push(walkforward_artifact);

        let forward_artifact: ForwardTestValidationArtifactFile = load_typed_member(
            generation_dir,
            members[2],
            ForwardTestValidationArtifactFile::from_json_bytes,
        )?;
        forward_artifact.validate_against(
            manifest.payload().holdout_scope(),
            manifest.search_config_hash(),
            strategy.gene(),
        )?;
        forward_test.push(forward_artifact);

        let prop_artifact: PropFirmRiskValidationArtifactFile = load_typed_member(
            generation_dir,
            members[3],
            PropFirmRiskValidationArtifactFile::from_json_bytes,
        )?;
        prop_artifact.validate_against(
            manifest.payload().holdout_scope(),
            manifest.search_config_hash(),
            strategy.gene(),
        )?;
        prop_firm.push(prop_artifact);
    }

    let summary_member = manifest.payload().promotion_summary();
    let summary_bytes = read_member_bytes(generation_dir, summary_member)?;
    let summary: CanonicalSearchArtifactEnvelopeV2<PromotionSummaryAuthorityPayloadV3> =
        CanonicalSearchArtifactEnvelopeV2::from_json_bytes(&summary_bytes)
            .map_err(anyhow::Error::new)?;
    ensure!(
        pretty_json_bytes(&summary)? == summary_bytes,
        "promotion-summary snapshot member bytes are not canonical"
    );
    summary
        .validate_against(
            PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
            manifest.search_config_hash(),
            manifest.scope().receipt(),
            manifest.scope().evaluated_window(),
        )
        .map_err(anyhow::Error::new)?;
    ensure!(
        summary.payload().holdout_scope() == manifest.payload().holdout_scope(),
        "promotion summary holdout scope differs from snapshot manifest"
    );
    summary.payload().validate_shape()?;
    validate_promotion_summary_members(manifest, summary.payload())?;

    let hashes = DiscoveryPerKindEvidenceHashes {
        canonical_backtest: Some(hash_sorted_artifacts(&canonical)?),
        walkforward: Some(hash_sorted_artifacts(&walkforward)?),
        forward_test: Some(hash_sorted_artifacts(&forward_test)?),
        prop_firm: Some(hash_sorted_artifacts(&prop_firm)?),
        live_execution_simulation: None,
    };
    ensure!(
        summary.payload().validation_evidence_hashes() == &hashes,
        "promotion summary aggregate evidence hashes differ from snapshot members"
    );
    Ok(())
}

fn validate_promotion_summary_members(
    manifest: &DiscoveryValidationSnapshotManifestEnvelopeV2,
    summary: &PromotionSummaryAuthorityPayloadV3,
) -> Result<()> {
    ensure!(
        summary.strategy_evidence().len() == manifest.payload().strategies().len(),
        "promotion summary strategy count differs from snapshot manifest"
    );
    for (summary_entry, manifest_entry) in summary
        .strategy_evidence()
        .iter()
        .zip(manifest.payload().strategies())
    {
        ensure!(
            summary_entry.strategy_identity() == manifest_entry.strategy_identity(),
            "promotion summary exact strategy identity differs from snapshot manifest"
        );
        let members = manifest_entry.members();
        ensure!(
            summary_entry.canonical_backtest_hash() == members[0].content_hash()
                && summary_entry.walkforward_hash() == members[1].content_hash()
                && summary_entry.forward_test_hash() == members[2].content_hash()
                && summary_entry.prop_firm_hash() == members[3].content_hash(),
            "promotion summary per-strategy hashes differ from snapshot members for {}",
            manifest_entry.strategy_identity().exact_gene_hash()
        );
    }
    Ok(())
}

fn hash_sorted_artifacts<T: Serialize>(artifacts: &[T]) -> Result<String> {
    stable_json_hash(artifacts)
}

fn load_typed_member<T, F>(
    generation_dir: &Path,
    member: &DiscoveryValidationSnapshotMemberV1,
    parse: F,
) -> Result<T>
where
    T: Serialize,
    F: Fn(&[u8]) -> Result<T>,
{
    let bytes = read_member_bytes(generation_dir, member)?;
    let value = parse(&bytes)?;
    ensure!(
        pretty_json_bytes(&value)? == bytes,
        "snapshot member {} bytes are not canonical",
        member.relative_path().display()
    );
    ensure!(
        stable_json_hash(&value)? == member.content_hash,
        "snapshot member {} canonical content hash mismatch",
        member.relative_path().display()
    );
    Ok(value)
}

fn read_member_bytes(
    generation_dir: &Path,
    member: &DiscoveryValidationSnapshotMemberV1,
) -> Result<Vec<u8>> {
    let path = resolve_snapshot_member_without_symlinks(generation_dir, member.relative_path())?;
    let bytes = fs::read(&path)
        .with_context(|| format!("read validation snapshot member {}", path.display()))?;
    ensure!(
        hash_bytes(&bytes) == member.byte_hash,
        "snapshot member {} byte hash mismatch",
        member.relative_path().display()
    );
    Ok(bytes)
}

fn prepare_snapshot_root(root: &Path) -> Result<()> {
    if root.exists() {
        ensure_existing_directory_without_symlink(root, "validation snapshot root")?;
    } else {
        fs::create_dir_all(root)
            .with_context(|| format!("create validation snapshot root {}", root.display()))?;
        ensure_existing_directory_without_symlink(root, "validation snapshot root")?;
    }
    let generations = root.join(GENERATIONS_DIR_NAME);
    if generations.exists() {
        ensure_existing_directory_without_symlink(
            &generations,
            "validation snapshot generations directory",
        )?;
    } else {
        fs::create_dir(&generations).with_context(|| {
            format!(
                "create validation snapshot generations directory {}",
                generations.display()
            )
        })?;
    }
    Ok(())
}

fn create_unique_staging_dir(generations_dir: &Path) -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    for _ in 0..32 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before unix epoch")?
            .as_nanos();
        let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            generations_dir.join(format!(".staging-{}-{nanos}-{suffix}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("create snapshot staging dir {}", path.display()));
            }
        }
    }
    bail!("could not allocate a unique validation snapshot staging directory")
}

fn write_snapshot_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("snapshot member {} has no parent", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create staged snapshot member dir {}", parent.display()))?;
    neoethos_core::storage::json::write_bytes_atomic(path, bytes)
        .with_context(|| format!("write staged snapshot member {}", path.display()))
}

fn verify_existing_generation(generation_dir: &Path, plan: &SnapshotPlan) -> Result<()> {
    verify_generation_tree(generation_dir, plan).with_context(|| {
        format!(
            "existing immutable generation {} differs from the exact planned bytes; refusing overwrite",
            generation_dir.display()
        )
    })
}

fn verify_generation_tree(generation_dir: &Path, plan: &SnapshotPlan) -> Result<()> {
    ensure_existing_directory_without_symlink(generation_dir, "validation snapshot generation")?;
    let actual = collect_generation_files(generation_dir)?;
    let expected_paths = plan
        .files
        .keys()
        .map(PathBuf::from)
        .collect::<BTreeSet<_>>();
    let actual_paths = actual.keys().cloned().collect::<BTreeSet<_>>();
    if let Some(missing) = expected_paths.difference(&actual_paths).next() {
        bail!("generation is missing member {}", missing.display());
    }
    if let Some(extra) = actual_paths.difference(&expected_paths).next() {
        bail!("generation contains extra member {}", extra.display());
    }
    for (relative, expected_bytes) in &plan.files {
        let actual_bytes = actual
            .get(Path::new(relative))
            .ok_or_else(|| anyhow!("generation is missing member {relative}"))?;
        ensure!(
            actual_bytes == expected_bytes,
            "generation member `{relative}` is not byte-identical"
        );
    }
    Ok(())
}

fn collect_generation_files(root: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>> {
    fn visit(root: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) -> Result<()> {
        ensure_existing_directory_without_symlink(dir, "snapshot generation directory")?;
        for entry in fs::read_dir(dir)
            .with_context(|| format!("enumerate snapshot generation dir {}", dir.display()))?
        {
            let entry = entry.with_context(|| format!("read entry below {}", dir.display()))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("read snapshot member metadata {}", path.display()))?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "validation snapshot member traverses symlink {}",
                path.display()
            );
            if metadata.is_dir() {
                visit(root, &path, out)?;
            } else {
                ensure!(
                    metadata.is_file(),
                    "validation snapshot contains non-file member {}",
                    path.display()
                );
                let relative = path
                    .strip_prefix(root)
                    .with_context(|| format!("member escaped generation root: {}", path.display()))?
                    .to_path_buf();
                ensure!(
                    relative
                        .components()
                        .all(|component| matches!(component, Component::Normal(_))),
                    "snapshot filesystem member contains a non-normal component: {}",
                    relative.display()
                );
                let bytes = fs::read(&path)
                    .with_context(|| format!("read snapshot member {}", path.display()))?;
                ensure!(
                    out.insert(relative.clone(), bytes).is_none(),
                    "duplicate snapshot member path {}",
                    relative.display()
                );
            }
        }
        Ok(())
    }

    let mut files = BTreeMap::new();
    visit(root, root, &mut files)?;
    Ok(files)
}

pub(crate) fn validate_snapshot_relative_path(path: &Path) -> Result<()> {
    let raw = path
        .to_str()
        .ok_or_else(|| anyhow!("snapshot member path is not UTF-8"))?;
    ensure!(!raw.is_empty(), "snapshot member path is empty");
    ensure!(
        !raw.contains('\\') && !raw.contains(':'),
        "snapshot member path `{raw}` is not portable/safe"
    );
    ensure!(
        !raw.starts_with('/') && !raw.starts_with("//"),
        "snapshot member path `{raw}` is absolute"
    );
    for segment in raw.split('/') {
        ensure!(
            !segment.is_empty() && segment != "." && segment != "..",
            "snapshot member path `{raw}` contains an unsafe component"
        );
        ensure!(
            !segment
                .chars()
                .any(|character| matches!(character, '<' | '>' | '"' | '|' | '?' | '*')),
            "snapshot member path `{raw}` contains a forbidden character"
        );
    }
    ensure!(
        !path.is_absolute(),
        "snapshot member path `{raw}` is absolute"
    );
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "snapshot member path `{raw}` contains a non-normal component"
    );
    Ok(())
}

pub(crate) fn resolve_snapshot_member_without_symlinks(
    generation_dir: &Path,
    relative_path: &Path,
) -> Result<PathBuf> {
    validate_snapshot_relative_path(relative_path)?;
    ensure_existing_directory_without_symlink(generation_dir, "validation snapshot generation")?;
    let mut cursor = generation_dir.to_path_buf();
    let component_count = relative_path.components().count();
    for (index, component) in relative_path.components().enumerate() {
        let Component::Normal(component) = component else {
            bail!("snapshot member path contains a non-normal component");
        };
        cursor.push(component);
        let metadata = fs::symlink_metadata(&cursor)
            .with_context(|| format!("read snapshot path metadata {}", cursor.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "validation snapshot member traverses symlink {}",
            cursor.display()
        );
        if index + 1 < component_count {
            ensure!(
                metadata.is_dir(),
                "snapshot member parent is not a directory: {}",
                cursor.display()
            );
        } else {
            ensure!(
                metadata.is_file(),
                "snapshot member is not a regular file: {}",
                cursor.display()
            );
        }
    }
    Ok(cursor)
}

fn ensure_existing_directory_without_symlink(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("read {label} metadata at {}", path.display()))?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "{label} may not be a symlink: {}",
        path.display()
    );
    ensure!(
        metadata.is_dir(),
        "{label} is not a directory: {}",
        path.display()
    );
    Ok(())
}

fn ensure_regular_file_without_symlink(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("read {label} metadata at {}", path.display()))?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "{label} may not be a symlink: {}",
        path.display()
    );
    ensure!(
        metadata.is_file(),
        "{label} is not a file: {}",
        path.display()
    );
    Ok(())
}

fn validate_scope_pair(
    selection_scope: &CanonicalSearchArtifactScopeV2,
    holdout_scope: &CanonicalSearchArtifactScopeV2,
) -> Result<()> {
    selection_scope.validate().map_err(anyhow::Error::new)?;
    holdout_scope.validate().map_err(anyhow::Error::new)?;
    ensure!(
        selection_scope.evaluated_window().role() == CanonicalSearchWindowRoleV1::InSample,
        "validation snapshot selection scope must have InSample role"
    );
    ensure!(
        holdout_scope.evaluated_window().role() == CanonicalSearchWindowRoleV1::Holdout,
        "validation snapshot holdout scope must have Holdout role"
    );
    ensure!(
        selection_scope.receipt() == holdout_scope.receipt(),
        "validation snapshot selection and holdout scopes have different receipts"
    );
    let anchor_id = selection_scope.receipt().anchor_dataset_identity();
    let anchor_bindings = selection_scope
        .receipt()
        .source_bindings()
        .iter()
        .filter(|binding| binding.dataset_identity() == anchor_id)
        .collect::<Vec<_>>();
    ensure!(
        anchor_bindings.len() == 1,
        "validation snapshot receipt must contain exactly one anchor binding"
    );
    let first_segment = anchor_bindings[0]
        .segments()
        .first()
        .ok_or_else(|| anyhow!("validation snapshot anchor binding has no first segment"))?;
    let last_segment = anchor_bindings[0]
        .segments()
        .last()
        .ok_or_else(|| anyhow!("validation snapshot anchor binding has no last segment"))?;
    ensure!(
        selection_scope.evaluated_window().row_start() == first_segment.row_start()
            && selection_scope.evaluated_window().row_end()
                == holdout_scope.evaluated_window().row_start()
            && holdout_scope.evaluated_window().row_end() == last_segment.row_end()
            && selection_scope.evaluated_window().timestamp_start_ms()
                == first_segment.timestamp_start_ms()
            && holdout_scope.evaluated_window().timestamp_end_ms()
                == last_segment.timestamp_end_ms(),
        "validation snapshot scopes are not the exact full receipt prefix/suffix"
    );
    ensure!(
        selection_scope.evaluated_window().timestamp_end_ms()
            < holdout_scope.evaluated_window().timestamp_start_ms(),
        "validation snapshot selection/holdout timestamps are not strictly ordered"
    );
    Ok(())
}

fn pretty_json_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).context("serialize snapshot JSON")?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("fnv64:{:016x}", fnv1a64(bytes))
}

fn validate_fnv64_hash(label: &str, value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("fnv64:") else {
        bail!("{label} must use fnv64:<16 lowercase hex>");
    };
    ensure!(
        hex.len() == 16
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} must use fnv64:<16 lowercase hex>"
    );
    Ok(())
}

fn generation_id_for_hash(hash: &str) -> Result<String> {
    validate_fnv64_hash("validation snapshot manifest hash", hash)?;
    Ok(hash.replacen(':', "-", 1))
}

fn validate_generation_id(value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("fnv64-") else {
        bail!("validation snapshot generation id must use fnv64-<16 lowercase hex>");
    };
    ensure!(
        hex.len() == 16
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "validation snapshot generation id must use fnv64-<16 lowercase hex>"
    );
    validate_snapshot_relative_path(Path::new(value))
}

fn strategy_file_leaf(exact_gene_hash: &str) -> Result<String> {
    validate_fnv64_hash("snapshot exact gene hash", exact_gene_hash)?;
    Ok(exact_gene_hash.replacen(':', "-", 1))
}
