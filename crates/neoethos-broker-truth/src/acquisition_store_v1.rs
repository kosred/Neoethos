//! Content-addressed storage for acquisition authority and bundle-link evidence.
//!
//! Publication and exact reopen prove only immutable file integrity and exact
//! cross-links. They deliberately create no semantic or financial authority.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::acquisition_v1::{
    BrokerTruthAcquisitionArtifactV1, BrokerTruthAcquisitionAuthorityManifestV1,
    BrokerTruthAcquisitionPromotionEligibilityV1, BrokerTruthAcquisitionSemanticStatusV1,
};
use crate::contracts::{
    BrokerFinancialTruthBindingV1, BrokerFinancialTruthContractErrorCodeV1,
    BrokerFinancialTruthContractErrorV1, max_manifest_bytes, sha256_bytes, sha256_file,
    validate_sha256_hex,
};
use crate::contracts_v2::BrokerFinancialTruthBundleReceiptV2;
use crate::store::BrokerFinancialTruthBundleStoreV1;

pub const BROKER_TRUTH_ACQUISITION_LINK_SCHEMA_VERSION_V1: u16 = 1;
pub const BROKER_TRUTH_ACQUISITION_AUTHORITY_ID_PREFIX_V1: &str = "bfta1-";
pub const BROKER_TRUTH_ACQUISITION_LINK_ID_PREFIX_V1: &str = "bftl1-";
pub const BROKER_TRUTH_ACQUISITION_AUTHORITY_MANIFEST_FILE_V1: &str =
    "broker-truth-acquisition-authority.manifest.json";
pub const BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1: &str =
    "broker-truth-acquisition-link.manifest.json";

const ACQUISITION_LINK_HASH_DOMAIN_V1: &[u8] = b"neoethos.broker-truth-acquisition-link.v1\0";
const MAX_SOURCE_NAME_BYTES_V1: usize = 160;
static ACQUISITION_STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerTruthAcquisitionStoreErrorCodeV1 {
    ContractInvalid,
    UnsafeFilesystemEntry,
    ReceiptInvalid,
    ManifestMissing,
    ManifestDigestMismatch,
    ManifestInvalid,
    ArtifactSetMismatch,
    ArtifactLengthMismatch,
    ArtifactDigestMismatch,
    SourceMismatch,
    ReferencedAuthorityInvalid,
    ReferencedBrokerBundleInvalid,
    PublishConflict,
    Io,
}

#[derive(Debug)]
pub struct BrokerTruthAcquisitionStoreErrorV1 {
    code: BrokerTruthAcquisitionStoreErrorCodeV1,
    detail: String,
}

impl BrokerTruthAcquisitionStoreErrorV1 {
    fn new(code: BrokerTruthAcquisitionStoreErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> BrokerTruthAcquisitionStoreErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for BrokerTruthAcquisitionStoreErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "broker truth acquisition store: {}", self.detail)
    }
}

impl Error for BrokerTruthAcquisitionStoreErrorV1 {}

impl From<BrokerFinancialTruthContractErrorV1> for BrokerTruthAcquisitionStoreErrorV1 {
    fn from(error: BrokerFinancialTruthContractErrorV1) -> Self {
        Self::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ContractInvalid,
            error.to_string(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerTruthAcquisitionArtifactSourceV1 {
    relative_path: String,
    source_path: PathBuf,
}

impl BrokerTruthAcquisitionArtifactSourceV1 {
    pub fn new(
        relative_path: impl Into<String>,
        source_path: impl Into<PathBuf>,
    ) -> Result<Self, BrokerTruthAcquisitionStoreErrorV1> {
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerTruthAcquisitionAuthorityReceiptV1 {
    authority_id: String,
    manifest_sha256: String,
}

impl BrokerTruthAcquisitionAuthorityReceiptV1 {
    fn from_manifest_sha256(
        manifest_sha256: String,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let receipt = Self {
            authority_id: format!(
                "{BROKER_TRUTH_ACQUISITION_AUTHORITY_ID_PREFIX_V1}{manifest_sha256}"
            ),
            manifest_sha256,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        ensure_bounded_contract_bytes(bytes, "acquisition authority receipt")?;
        let receipt: Self = serde_json::from_slice(bytes).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot decode acquisition authority receipt: {error}"),
            )
        })?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot encode acquisition authority receipt: {error}"),
            )
        })
    }

    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_sha256_hex(
            "acquisition authority manifest SHA-256",
            &self.manifest_sha256,
        )?;
        if self.authority_id
            != format!(
                "{BROKER_TRUTH_ACQUISITION_AUTHORITY_ID_PREFIX_V1}{}",
                self.manifest_sha256
            )
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                "authority id does not equal bfta1- plus the exact manifest SHA-256",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerTruthAcquisitionLinkManifestV1 {
    schema_version: u16,
    semantic_status: BrokerTruthAcquisitionSemanticStatusV1,
    promotion_eligibility: BrokerTruthAcquisitionPromotionEligibilityV1,
    authority_receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
    broker_truth_receipt: BrokerFinancialTruthBundleReceiptV2,
    binding: BrokerFinancialTruthBindingV1,
}

impl BrokerTruthAcquisitionLinkManifestV1 {
    pub fn new(
        authority_receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
        broker_truth_receipt: BrokerFinancialTruthBundleReceiptV2,
        binding: BrokerFinancialTruthBindingV1,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let manifest = Self {
            schema_version: BROKER_TRUTH_ACQUISITION_LINK_SCHEMA_VERSION_V1,
            semantic_status: BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly,
            promotion_eligibility:
                BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible,
            authority_receipt,
            broker_truth_receipt,
            binding,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn semantic_status(&self) -> BrokerTruthAcquisitionSemanticStatusV1 {
        self.semantic_status
    }

    pub const fn promotion_eligibility(&self) -> BrokerTruthAcquisitionPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub const fn authority_receipt(&self) -> &BrokerTruthAcquisitionAuthorityReceiptV1 {
        &self.authority_receipt
    }

    pub const fn broker_truth_receipt(&self) -> &BrokerFinancialTruthBundleReceiptV2 {
        &self.broker_truth_receipt
    }

    pub const fn binding(&self) -> &BrokerFinancialTruthBindingV1 {
        &self.binding
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot encode acquisition link manifest: {error}"),
            )
        })
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        ensure_bounded_contract_bytes(bytes, "acquisition link manifest")?;
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                format!("cannot decode acquisition link manifest: {error}"),
            )
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn identity_sha256(&self) -> Result<String, BrokerFinancialTruthContractErrorV1> {
        let canonical = self.canonical_json_bytes()?;
        let mut hasher = Sha256::new();
        hasher.update(ACQUISITION_LINK_HASH_DOMAIN_V1);
        hasher.update(canonical);
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        if self.schema_version != BROKER_TRUTH_ACQUISITION_LINK_SCHEMA_VERSION_V1 {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::UnsupportedSchemaVersion,
                format!(
                    "unsupported acquisition link schema {}; expected {}",
                    self.schema_version, BROKER_TRUTH_ACQUISITION_LINK_SCHEMA_VERSION_V1
                ),
            ));
        }
        if self.semantic_status != BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
            || self.promotion_eligibility
                != BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
                "acquisition link must remain unvalidated and not promotion eligible",
            ));
        }
        self.authority_receipt.validate()?;
        self.broker_truth_receipt.validate()?;
        self.binding.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrokerTruthAcquisitionLinkReceiptV1 {
    link_id: String,
    manifest_sha256: String,
}

impl BrokerTruthAcquisitionLinkReceiptV1 {
    fn from_manifest_sha256(
        manifest_sha256: String,
    ) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        let receipt = Self {
            link_id: format!("{BROKER_TRUTH_ACQUISITION_LINK_ID_PREFIX_V1}{manifest_sha256}"),
            manifest_sha256,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BrokerFinancialTruthContractErrorV1> {
        ensure_bounded_contract_bytes(bytes, "acquisition link receipt")?;
        let receipt: Self = serde_json::from_slice(bytes).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot decode acquisition link receipt: {error}"),
            )
        })?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, BrokerFinancialTruthContractErrorV1> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                format!("cannot encode acquisition link receipt: {error}"),
            )
        })
    }

    pub fn link_id(&self) -> &str {
        &self.link_id
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    fn validate(&self) -> Result<(), BrokerFinancialTruthContractErrorV1> {
        validate_sha256_hex("acquisition link manifest SHA-256", &self.manifest_sha256)?;
        if self.link_id
            != format!(
                "{BROKER_TRUTH_ACQUISITION_LINK_ID_PREFIX_V1}{}",
                self.manifest_sha256
            )
        {
            return Err(contract_error(
                BrokerFinancialTruthContractErrorCodeV1::InvalidReceipt,
                "link id does not equal bftl1- plus the exact manifest SHA-256",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedImmutableBrokerTruthAcquisitionAuthorityV1 {
    root: PathBuf,
    receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
    manifest: BrokerTruthAcquisitionAuthorityManifestV1,
}

impl VerifiedImmutableBrokerTruthAcquisitionAuthorityV1 {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn receipt(&self) -> &BrokerTruthAcquisitionAuthorityReceiptV1 {
        &self.receipt
    }

    pub const fn manifest(&self) -> &BrokerTruthAcquisitionAuthorityManifestV1 {
        &self.manifest
    }

    pub fn artifact_path(&self, artifact: &BrokerTruthAcquisitionArtifactV1) -> PathBuf {
        self.root.join(artifact.relative_path())
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedImmutableBrokerTruthAcquisitionLinkV1 {
    root: PathBuf,
    receipt: BrokerTruthAcquisitionLinkReceiptV1,
    manifest: BrokerTruthAcquisitionLinkManifestV1,
}

impl VerifiedImmutableBrokerTruthAcquisitionLinkV1 {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn receipt(&self) -> &BrokerTruthAcquisitionLinkReceiptV1 {
        &self.receipt
    }

    pub const fn manifest(&self) -> &BrokerTruthAcquisitionLinkManifestV1 {
        &self.manifest
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root
            .join(BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1)
    }
}

#[derive(Clone, Debug)]
pub struct BrokerTruthAcquisitionStoreV1 {
    root: PathBuf,
}

impl BrokerTruthAcquisitionStoreV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn authority_path(&self, receipt: &BrokerTruthAcquisitionAuthorityReceiptV1) -> PathBuf {
        self.root.join(receipt.authority_id())
    }

    pub fn link_path(&self, receipt: &BrokerTruthAcquisitionLinkReceiptV1) -> PathBuf {
        self.root.join(receipt.link_id())
    }

    pub fn publish_authority(
        &self,
        manifest: &BrokerTruthAcquisitionAuthorityManifestV1,
        sources: &[BrokerTruthAcquisitionArtifactSourceV1],
    ) -> Result<BrokerTruthAcquisitionAuthorityReceiptV1, BrokerTruthAcquisitionStoreErrorV1> {
        let manifest_bytes = manifest.canonical_json_bytes()?;
        let receipt = BrokerTruthAcquisitionAuthorityReceiptV1::from_manifest_sha256(
            sha256_bytes(&manifest_bytes),
        )?;
        let source_map = validate_sources(manifest.artifacts(), sources)?;
        for artifact in manifest.artifacts() {
            let source = source_map.get(artifact.relative_path()).ok_or_else(|| {
                BrokerTruthAcquisitionStoreErrorV1::new(
                    BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch,
                    format!("no source supplied for {}", artifact.relative_path()),
                )
            })?;
            validate_source_artifact(source, artifact)?;
        }
        self.ensure_safe_store_root()?;

        let final_root = self.authority_path(&receipt);
        match fs::symlink_metadata(&final_root) {
            Ok(_) => {
                self.open_authority(&receipt)?;
                return Ok(receipt);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(io_error(
                    format!("cannot inspect authority path {}", final_root.display()),
                    error,
                ));
            }
        }

        let staging_root = self.create_staging_directory("bfta1")?;
        let publish_result = (|| {
            for artifact in manifest.artifacts() {
                let source = source_map.get(artifact.relative_path()).ok_or_else(|| {
                    BrokerTruthAcquisitionStoreErrorV1::new(
                        BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch,
                        format!("no source supplied for {}", artifact.relative_path()),
                    )
                })?;
                copy_exact_artifact(
                    source,
                    &staging_root.join(artifact.relative_path()),
                    artifact,
                )?;
            }
            write_new_file(
                &staging_root.join(BROKER_TRUTH_ACQUISITION_AUTHORITY_MANIFEST_FILE_V1),
                &manifest_bytes,
            )?;
            rename_publication(&staging_root, &final_root, "acquisition authority")?;
            self.open_authority(&receipt)?;
            Ok(receipt.clone())
        })();
        if publish_result.is_err() {
            cleanup_staging_directory(&self.root, &staging_root, ".bfta1-staging-");
        }
        publish_result
    }

    pub fn open_authority(
        &self,
        receipt: &BrokerTruthAcquisitionAuthorityReceiptV1,
    ) -> Result<
        VerifiedImmutableBrokerTruthAcquisitionAuthorityV1,
        BrokerTruthAcquisitionStoreErrorV1,
    > {
        receipt.validate().map_err(receipt_error)?;
        self.ensure_safe_store_root()?;
        let authority_root = self.authority_path(receipt);
        ensure_regular_directory(&authority_root, "authority")?;
        let manifest_path =
            authority_root.join(BROKER_TRUTH_ACQUISITION_AUTHORITY_MANIFEST_FILE_V1);
        let manifest_bytes = read_manifest(&manifest_path)?;
        let actual_manifest_sha256 = sha256_bytes(&manifest_bytes);
        if actual_manifest_sha256 != receipt.manifest_sha256() {
            return Err(BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::ManifestDigestMismatch,
                format!(
                    "authority manifest digest {actual_manifest_sha256} does not match exact receipt {}",
                    receipt.manifest_sha256()
                ),
            ));
        }
        let manifest = BrokerTruthAcquisitionAuthorityManifestV1::from_json_bytes(&manifest_bytes)
            .map_err(manifest_error)?;
        validate_exact_file_set(
            &authority_root,
            manifest
                .artifacts()
                .iter()
                .map(BrokerTruthAcquisitionArtifactV1::relative_path),
            BROKER_TRUTH_ACQUISITION_AUTHORITY_MANIFEST_FILE_V1,
        )?;
        for artifact in manifest.artifacts() {
            validate_published_artifact(&authority_root.join(artifact.relative_path()), artifact)?;
        }
        Ok(VerifiedImmutableBrokerTruthAcquisitionAuthorityV1 {
            root: authority_root,
            receipt: receipt.clone(),
            manifest,
        })
    }

    pub fn publish_link(
        &self,
        authority_receipt: &BrokerTruthAcquisitionAuthorityReceiptV1,
        broker_truth_receipt: &BrokerFinancialTruthBundleReceiptV2,
        expected_binding: &BrokerFinancialTruthBindingV1,
    ) -> Result<BrokerTruthAcquisitionLinkReceiptV1, BrokerTruthAcquisitionStoreErrorV1> {
        self.reopen_link_targets(authority_receipt, broker_truth_receipt, expected_binding)?;
        let manifest = BrokerTruthAcquisitionLinkManifestV1::new(
            authority_receipt.clone(),
            broker_truth_receipt.clone(),
            expected_binding.clone(),
        )?;
        let manifest_bytes = manifest.canonical_json_bytes()?;
        let receipt = BrokerTruthAcquisitionLinkReceiptV1::from_manifest_sha256(sha256_bytes(
            &manifest_bytes,
        ))?;
        self.ensure_safe_store_root()?;

        let final_root = self.link_path(&receipt);
        match fs::symlink_metadata(&final_root) {
            Ok(_) => {
                self.open_link(&receipt)?;
                return Ok(receipt);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(io_error(
                    format!("cannot inspect link path {}", final_root.display()),
                    error,
                ));
            }
        }

        let staging_root = self.create_staging_directory("bftl1")?;
        let publish_result = (|| {
            write_new_file(
                &staging_root.join(BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1),
                &manifest_bytes,
            )?;
            rename_publication(&staging_root, &final_root, "acquisition link")?;
            self.open_link(&receipt)?;
            Ok(receipt.clone())
        })();
        if publish_result.is_err() {
            cleanup_staging_directory(&self.root, &staging_root, ".bftl1-staging-");
        }
        publish_result
    }

    pub fn open_link(
        &self,
        receipt: &BrokerTruthAcquisitionLinkReceiptV1,
    ) -> Result<VerifiedImmutableBrokerTruthAcquisitionLinkV1, BrokerTruthAcquisitionStoreErrorV1>
    {
        receipt.validate().map_err(receipt_error)?;
        self.ensure_safe_store_root()?;
        let link_root = self.link_path(receipt);
        ensure_regular_directory(&link_root, "link")?;
        let manifest_path = link_root.join(BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1);
        let manifest_bytes = read_manifest(&manifest_path)?;
        let actual_manifest_sha256 = sha256_bytes(&manifest_bytes);
        if actual_manifest_sha256 != receipt.manifest_sha256() {
            return Err(BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::ManifestDigestMismatch,
                format!(
                    "link manifest digest {actual_manifest_sha256} does not match exact receipt {}",
                    receipt.manifest_sha256()
                ),
            ));
        }
        let manifest = BrokerTruthAcquisitionLinkManifestV1::from_json_bytes(&manifest_bytes)
            .map_err(manifest_error)?;
        validate_exact_file_set(
            &link_root,
            std::iter::empty::<&str>(),
            BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1,
        )?;
        self.reopen_link_targets(
            manifest.authority_receipt(),
            manifest.broker_truth_receipt(),
            manifest.binding(),
        )?;
        Ok(VerifiedImmutableBrokerTruthAcquisitionLinkV1 {
            root: link_root,
            receipt: receipt.clone(),
            manifest,
        })
    }

    fn reopen_link_targets(
        &self,
        authority_receipt: &BrokerTruthAcquisitionAuthorityReceiptV1,
        broker_truth_receipt: &BrokerFinancialTruthBundleReceiptV2,
        expected_binding: &BrokerFinancialTruthBindingV1,
    ) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
        let authority = self.open_authority(authority_receipt).map_err(|error| {
            BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::ReferencedAuthorityInvalid,
                format!("referenced authority cannot be reopened exactly: {error}"),
            )
        })?;
        validate_authority_binding(authority.manifest(), expected_binding)?;
        BrokerFinancialTruthBundleStoreV1::new(self.root.clone())
            .open_exact_v2(broker_truth_receipt, expected_binding)
            .map_err(|error| {
                BrokerTruthAcquisitionStoreErrorV1::new(
                    BrokerTruthAcquisitionStoreErrorCodeV1::ReferencedBrokerBundleInvalid,
                    format!("referenced BFT2 bundle cannot be reopened exactly: {error}"),
                )
            })?;
        Ok(())
    }

    fn ensure_safe_store_root(&self) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
        match fs::symlink_metadata(&self.root) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(BrokerTruthAcquisitionStoreErrorV1::new(
                        BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry,
                        format!(
                            "acquisition store root {} is not a regular directory",
                            self.root.display()
                        ),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).map_err(|error| {
                    io_error(
                        format!(
                            "cannot create acquisition store root {}",
                            self.root.display()
                        ),
                        error,
                    )
                })?;
                let metadata = fs::symlink_metadata(&self.root).map_err(|error| {
                    io_error(
                        format!(
                            "cannot inspect created acquisition store root {}",
                            self.root.display()
                        ),
                        error,
                    )
                })?;
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(BrokerTruthAcquisitionStoreErrorV1::new(
                        BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry,
                        "created acquisition store root is not a regular directory",
                    ));
                }
            }
            Err(error) => {
                return Err(io_error(
                    format!(
                        "cannot inspect acquisition store root {}",
                        self.root.display()
                    ),
                    error,
                ));
            }
        }
        Ok(())
    }

    fn create_staging_directory(
        &self,
        kind: &str,
    ) -> Result<PathBuf, BrokerTruthAcquisitionStoreErrorV1> {
        for _ in 0..32 {
            let clock = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| {
                    BrokerTruthAcquisitionStoreErrorV1::new(
                        BrokerTruthAcquisitionStoreErrorCodeV1::Io,
                        format!("system clock is before Unix epoch: {error}"),
                    )
                })?
                .as_nanos();
            let nonce = ACQUISITION_STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = self.root.join(format!(
                ".{kind}-staging-{}-{clock}-{nonce}",
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
        Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::PublishConflict,
            "cannot allocate a unique acquisition staging directory",
        ))
    }
}

fn validate_authority_binding(
    authority: &BrokerTruthAcquisitionAuthorityManifestV1,
    expected_binding: &BrokerFinancialTruthBindingV1,
) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    let CanonicalDatasetScope::CTrader {
        account_id,
        symbol_id,
        ..
    } = expected_binding.canonical_dataset_identity().scope()
    else {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ReferencedAuthorityInvalid,
            "link binding is not an exact cTrader dataset identity",
        ));
    };
    if authority.canonical_search_input_receipt_sha256()
        != expected_binding.canonical_search_input_receipt_sha256()
        || authority.reviewed_synchronizations().iter().any(|binding| {
            binding.account_id() != *account_id
                || binding.symbol_id() != *symbol_id
                || binding.window() != expected_binding.evaluated_window()
        })
    {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ReferencedAuthorityInvalid,
            "acquisition authority differs from the exact receipt/account/symbol/window binding",
        ));
    }
    Ok(())
}

fn validate_sources<'a>(
    artifacts: &[BrokerTruthAcquisitionArtifactV1],
    sources: &'a [BrokerTruthAcquisitionArtifactSourceV1],
) -> Result<HashMap<&'a str, &'a Path>, BrokerTruthAcquisitionStoreErrorV1> {
    let expected = artifacts
        .iter()
        .map(BrokerTruthAcquisitionArtifactV1::relative_path)
        .collect::<HashSet<_>>();
    let mut source_map = HashMap::new();
    for source in sources {
        validate_source_basename(source.relative_path())?;
        if source_map
            .insert(source.relative_path(), source.source_path())
            .is_some()
        {
            return Err(BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch,
                format!("duplicate source mapping for {}", source.relative_path()),
            ));
        }
    }
    if expected != source_map.keys().copied().collect::<HashSet<_>>() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch,
            "artifact sources are not an exact set match for the authority manifest",
        ));
    }
    Ok(source_map)
}

fn validate_source_basename(value: &str) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    let path = Path::new(value);
    let mut components = path.components();
    let exactly_one_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if value.is_empty()
        || value.len() > MAX_SOURCE_NAME_BYTES_V1
        || !exactly_one_normal_component
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        })
    {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch,
            format!("source mapping {value:?} is not one safe lowercase basename"),
        ));
    }
    Ok(())
}

fn copy_exact_artifact(
    source: &Path,
    destination: &Path,
    expected: &BrokerTruthAcquisitionArtifactV1,
) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    validate_source_artifact(source, expected)?;

    let mut input = File::open(source).map_err(|error| {
        io_error(
            format!("cannot open authority source {}", source.display()),
            error,
        )
    })?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| {
            io_error(
                format!("cannot create authority artifact {}", destination.display()),
                error,
            )
        })?;
    std::io::copy(&mut input, &mut output).map_err(|error| {
        io_error(
            format!(
                "cannot copy exact authority source {} to {}",
                source.display(),
                destination.display()
            ),
            error,
        )
    })?;
    output.sync_all().map_err(|error| {
        io_error(
            format!("cannot fsync authority artifact {}", destination.display()),
            error,
        )
    })?;
    validate_published_artifact(destination, expected)
}

fn validate_source_artifact(
    source: &Path,
    expected: &BrokerTruthAcquisitionArtifactV1,
) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        io_error(
            format!("cannot inspect authority source {}", source.display()),
            error,
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!(
                "authority source {} is not a regular file",
                source.display()
            ),
        ));
    }
    if metadata.len() != expected.byte_len() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch,
            format!("source length changed for {}", expected.relative_path()),
        ));
    }
    let source_digest = sha256_file(source).map_err(contract_io_to_store)?;
    if source_digest != expected.sha256() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch,
            format!("source digest changed for {}", expected.relative_path()),
        ));
    }
    Ok(())
}

fn validate_published_artifact(
    path: &Path,
    expected: &BrokerTruthAcquisitionArtifactV1,
) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        io_error(
            format!("cannot inspect authority artifact {}", path.display()),
            error,
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!(
                "authority artifact {} is not a regular file",
                path.display()
            ),
        ));
    }
    if metadata.len() != expected.byte_len() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ArtifactLengthMismatch,
            format!(
                "authority artifact {} has length {}, expected {}",
                expected.relative_path(),
                metadata.len(),
                expected.byte_len()
            ),
        ));
    }
    let digest = sha256_file(path).map_err(contract_io_to_store)?;
    if digest != expected.sha256() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ArtifactDigestMismatch,
            format!(
                "authority artifact {} has digest {digest}, expected {}",
                expected.relative_path(),
                expected.sha256()
            ),
        ));
    }
    Ok(())
}

fn validate_exact_file_set<'a>(
    root: &Path,
    artifacts: impl Iterator<Item = &'a str>,
    manifest_name: &str,
) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    let mut expected = artifacts.map(str::to_owned).collect::<HashSet<_>>();
    expected.insert(manifest_name.to_owned());
    let mut received = HashSet::new();
    for entry in fs::read_dir(root).map_err(|error| {
        io_error(
            format!("cannot enumerate acquisition package {}", root.display()),
            error,
        )
    })? {
        let entry = entry.map_err(|error| {
            io_error(
                format!("cannot enumerate acquisition package {}", root.display()),
                error,
            )
        })?;
        let file_type = entry.file_type().map_err(|error| {
            io_error(
                format!("cannot inspect package entry {}", entry.path().display()),
                error,
            )
        })?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry,
                format!(
                    "package entry {} is not a regular file",
                    entry.path().display()
                ),
            ));
        }
        let name = entry.file_name().into_string().map_err(|_| {
            BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::ArtifactSetMismatch,
                "acquisition package contains a non-UTF-8 file name",
            )
        })?;
        received.insert(name);
    }
    if expected != received {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ArtifactSetMismatch,
            "acquisition package file set differs from its exact manifest",
        ));
    }
    Ok(())
}

fn ensure_regular_directory(
    path: &Path,
    label: &str,
) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::ManifestMissing,
                format!("{label} package {} does not exist", path.display()),
            )
        } else {
            io_error(
                format!("cannot inspect {label} package {}", path.display()),
                error,
            )
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!(
                "{label} package {} is not a regular directory",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn read_manifest(path: &Path) -> Result<Vec<u8>, BrokerTruthAcquisitionStoreErrorV1> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            BrokerTruthAcquisitionStoreErrorV1::new(
                BrokerTruthAcquisitionStoreErrorCodeV1::ManifestMissing,
                format!("acquisition manifest {} does not exist", path.display()),
            )
        } else {
            io_error(
                format!("cannot inspect acquisition manifest {}", path.display()),
                error,
            )
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry,
            format!(
                "acquisition manifest {} is not a regular file",
                path.display()
            ),
        ));
    }
    if metadata.len() > max_manifest_bytes() {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ManifestInvalid,
            format!(
                "acquisition manifest {} exceeds the size limit",
                path.display()
            ),
        ));
    }
    let file = File::open(path).map_err(|error| {
        io_error(
            format!("cannot open acquisition manifest {}", path.display()),
            error,
        )
    })?;
    let maximum_bytes = max_manifest_bytes();
    let mut bounded = file.take(maximum_bytes + 1);
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    bounded.read_to_end(&mut bytes).map_err(|error| {
        io_error(
            format!("cannot read acquisition manifest {}", path.display()),
            error,
        )
    })?;
    if bytes.len() as u64 > maximum_bytes {
        return Err(BrokerTruthAcquisitionStoreErrorV1::new(
            BrokerTruthAcquisitionStoreErrorCodeV1::ManifestInvalid,
            format!(
                "acquisition manifest {} exceeds the size limit",
                path.display()
            ),
        ));
    }
    Ok(bytes)
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            io_error(
                format!("cannot create immutable file {}", path.display()),
                error,
            )
        })?;
    file.write_all(bytes).map_err(|error| {
        io_error(
            format!("cannot write immutable file {}", path.display()),
            error,
        )
    })?;
    file.sync_all().map_err(|error| {
        io_error(
            format!("cannot fsync immutable file {}", path.display()),
            error,
        )
    })
}

fn rename_publication(
    staging_root: &Path,
    final_root: &Path,
    label: &str,
) -> Result<(), BrokerTruthAcquisitionStoreErrorV1> {
    fs::rename(staging_root, final_root).map_err(|error| {
        let code = if error.kind() == std::io::ErrorKind::AlreadyExists {
            BrokerTruthAcquisitionStoreErrorCodeV1::PublishConflict
        } else {
            BrokerTruthAcquisitionStoreErrorCodeV1::Io
        };
        BrokerTruthAcquisitionStoreErrorV1::new(
            code,
            format!(
                "cannot atomically publish {label} {} as {}: {error}",
                staging_root.display(),
                final_root.display()
            ),
        )
    })
}

fn cleanup_staging_directory(root: &Path, staging_root: &Path, prefix: &str) {
    let Some(name) = staging_root.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    if staging_root.parent() == Some(root) && name.starts_with(prefix) {
        let _ = fs::remove_dir_all(staging_root);
    }
}

fn ensure_bounded_contract_bytes(
    bytes: &[u8],
    label: &str,
) -> Result<(), BrokerFinancialTruthContractErrorV1> {
    if bytes.len() as u64 > max_manifest_bytes() {
        return Err(contract_error(
            BrokerFinancialTruthContractErrorCodeV1::InvalidManifest,
            format!("{label} exceeds the size limit"),
        ));
    }
    Ok(())
}

fn receipt_error(error: BrokerFinancialTruthContractErrorV1) -> BrokerTruthAcquisitionStoreErrorV1 {
    BrokerTruthAcquisitionStoreErrorV1::new(
        BrokerTruthAcquisitionStoreErrorCodeV1::ReceiptInvalid,
        error.to_string(),
    )
}

fn manifest_error(
    error: BrokerFinancialTruthContractErrorV1,
) -> BrokerTruthAcquisitionStoreErrorV1 {
    BrokerTruthAcquisitionStoreErrorV1::new(
        BrokerTruthAcquisitionStoreErrorCodeV1::ManifestInvalid,
        error.to_string(),
    )
}

fn contract_io_to_store(
    error: BrokerFinancialTruthContractErrorV1,
) -> BrokerTruthAcquisitionStoreErrorV1 {
    BrokerTruthAcquisitionStoreErrorV1::new(
        BrokerTruthAcquisitionStoreErrorCodeV1::Io,
        error.to_string(),
    )
}

fn contract_error(
    code: BrokerFinancialTruthContractErrorCodeV1,
    detail: impl Into<String>,
) -> BrokerFinancialTruthContractErrorV1 {
    BrokerFinancialTruthContractErrorV1::new(code, detail)
}

fn io_error(
    context: impl Into<String>,
    error: std::io::Error,
) -> BrokerTruthAcquisitionStoreErrorV1 {
    BrokerTruthAcquisitionStoreErrorV1::new(
        BrokerTruthAcquisitionStoreErrorCodeV1::Io,
        format!("{}: {error}", context.into()),
    )
}
