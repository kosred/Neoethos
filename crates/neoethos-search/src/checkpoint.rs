pub use crate::artifact_io::stable_json_hash;
use crate::artifact_io::{read_json, write_json_atomic};
use crate::genetic::strategy_gene::{Gene, SearchResult};
use anyhow::{Result, bail};
use neoethos_core::contracts::{TemporalFeatureContract, TemporalScopeHashes};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub const SEARCH_CHECKPOINT_ARTIFACT_KIND: &str = "search_checkpoint_artifact";
pub const PORTFOLIO_SELECTION_ARTIFACT_KIND: &str = "portfolio_selection_artifact";
const CHECKPOINT_SCHEMA_VERSION: u32 = 2;
const PORTFOLIO_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchCheckpointScope {
    pub config_hash: String,
    pub dataset_hash: String,
    pub search_space_hash: String,
    pub temporal_scope: TemporalScopeHashes,
}

impl SearchCheckpointScope {
    pub fn new(
        config_hash: impl Into<String>,
        dataset_hash: impl Into<String>,
        search_space_hash: impl Into<String>,
        temporal_contract: &TemporalFeatureContract,
    ) -> Self {
        Self {
            config_hash: config_hash.into(),
            dataset_hash: dataset_hash.into(),
            search_space_hash: search_space_hash.into(),
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        }
    }

    pub fn from_parts<T: Serialize, U: Serialize, V: Serialize>(
        config: &T,
        dataset: &U,
        search_space: &V,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self::new(
            stable_json_hash(config)?,
            stable_json_hash(dataset)?,
            stable_json_hash(search_space)?,
            temporal_contract,
        ))
    }

    fn validate_resume(&self, expected: &Self) -> Result<()> {
        if self.config_hash != expected.config_hash {
            bail!(
                "Search checkpoint from a previous run can't be resumed — config hash mismatch \
                 (stored={} expected={}). \
                 Delete cache/search/<symbol>_<tf>.checkpoint to start fresh.",
                self.config_hash,
                expected.config_hash
            );
        }
        if self.dataset_hash != expected.dataset_hash {
            bail!(
                "Search checkpoint from a previous run can't be resumed — dataset hash mismatch \
                 (stored={} expected={}). \
                 Delete cache/search/<symbol>_<tf>.checkpoint to start fresh.",
                self.dataset_hash,
                expected.dataset_hash
            );
        }
        if self.search_space_hash != expected.search_space_hash {
            bail!(
                "Search checkpoint from a previous run can't be resumed — search-space hash mismatch \
                 (stored={} expected={}). \
                 Delete cache/search/<symbol>_<tf>.checkpoint to start fresh.",
                self.search_space_hash,
                expected.search_space_hash
            );
        }
        self.validate_resume_field(
            "temporal contract",
            &self.temporal_scope.temporal_contract_hash,
            &expected.temporal_scope.temporal_contract_hash,
        )?;
        self.validate_resume_field(
            "timestamp policy",
            &self.temporal_scope.timestamp_policy_hash,
            &expected.temporal_scope.timestamp_policy_hash,
        )?;
        self.validate_resume_field(
            "feature-availability policy",
            &self.temporal_scope.feature_availability_policy_hash,
            &expected.temporal_scope.feature_availability_policy_hash,
        )?;
        self.validate_resume_field(
            "label policy",
            &self.temporal_scope.label_policy_hash,
            &expected.temporal_scope.label_policy_hash,
        )?;
        Ok(())
    }

    fn validate_resume_field(&self, label: &str, stored: &str, expected: &str) -> Result<()> {
        if stored != expected {
            bail!(
                "Search checkpoint from a previous run can't be resumed — {label} hash mismatch \
                 (stored={} expected={}). \
                 Delete cache/search/<symbol>_<tf>.checkpoint to start fresh.",
                stored,
                expected
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeterministicSeedChain {
    pub root_seed: u64,
    pub generation_seed: u64,
    pub candidate_seed: u64,
}

impl DeterministicSeedChain {
    pub fn new(root_seed: u64) -> Self {
        Self {
            root_seed,
            generation_seed: derive_seed(root_seed, b"generation", 0),
            candidate_seed: derive_seed(root_seed, b"candidate", 0),
        }
    }

    pub fn for_generation(&self, generation: usize) -> Self {
        Self {
            root_seed: self.root_seed,
            generation_seed: derive_seed(self.root_seed, b"generation", generation as u64),
            candidate_seed: derive_seed(self.root_seed, b"candidate", generation as u64),
        }
    }

    pub fn candidate_seed(&self, generation: usize, candidate_index: usize) -> u64 {
        derive_seed(
            derive_seed(self.root_seed, b"candidate", generation as u64),
            b"candidate-index",
            candidate_index as u64,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvaluatedCandidateRecord {
    pub candidate_hash: String,
    pub generation: usize,
    pub candidate_index: usize,
    pub seed: u64,
}

impl EvaluatedCandidateRecord {
    pub fn from_gene(
        gene: &Gene,
        generation: usize,
        candidate_index: usize,
        seed_chain: &DeterministicSeedChain,
    ) -> Result<Self> {
        Ok(Self {
            candidate_hash: stable_json_hash(gene)?,
            generation,
            candidate_index,
            seed: seed_chain.candidate_seed(generation, candidate_index),
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvaluatedCandidateLedger {
    pub records: Vec<EvaluatedCandidateRecord>,
}

impl EvaluatedCandidateLedger {
    pub fn insert(&mut self, record: EvaluatedCandidateRecord) -> bool {
        if self.contains_hash(&record.candidate_hash) {
            return false;
        }
        self.records.push(record);
        true
    }

    pub fn contains_hash(&self, candidate_hash: &str) -> bool {
        self.records
            .iter()
            .any(|record| record.candidate_hash == candidate_hash)
    }

    pub fn candidate_hashes(&self) -> HashSet<&str> {
        self.records
            .iter()
            .map(|record| record.candidate_hash.as_str())
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCheckpointArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: SearchCheckpointScope,
    pub completed_generations: usize,
    pub seed_chain: DeterministicSeedChain,
    pub ledger: EvaluatedCandidateLedger,
    pub genes: Vec<Gene>,
    pub metrics: Vec<[f64; 11]>,
}

impl SearchCheckpointArtifactFile {
    pub fn new(
        scope: SearchCheckpointScope,
        completed_generations: usize,
        seed_chain: DeterministicSeedChain,
        ledger: EvaluatedCandidateLedger,
        result: SearchResult,
    ) -> Self {
        Self {
            artifact_kind: SEARCH_CHECKPOINT_ARTIFACT_KIND.to_string(),
            artifact_schema_version: CHECKPOINT_SCHEMA_VERSION,
            scope,
            completed_generations,
            seed_chain,
            ledger,
            genes: result.genes,
            metrics: result.metrics,
        }
    }

    pub fn search_result(&self) -> SearchResult {
        SearchResult {
            genes: self.genes.clone(),
            metrics: self.metrics.clone(),
        }
    }

    pub fn validate_for_resume(&self, expected_scope: &SearchCheckpointScope) -> Result<()> {
        if self.artifact_kind != SEARCH_CHECKPOINT_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be resumed as a search checkpoint",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != CHECKPOINT_SCHEMA_VERSION {
            bail!(
                "unsupported checkpoint schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_resume(expected_scope)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSelectionArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub source_checkpoint_hash: String,
    pub source_scope: SearchCheckpointScope,
    pub selected_genes: Vec<Gene>,
    pub metrics: Vec<[f64; 11]>,
}

impl PortfolioSelectionArtifactFile {
    pub fn new(
        source_checkpoint: &SearchCheckpointArtifactFile,
        selected_genes: Vec<Gene>,
        metrics: Vec<[f64; 11]>,
    ) -> Result<Self> {
        Ok(Self {
            artifact_kind: PORTFOLIO_SELECTION_ARTIFACT_KIND.to_string(),
            artifact_schema_version: PORTFOLIO_SCHEMA_VERSION,
            source_checkpoint_hash: stable_json_hash(source_checkpoint)?,
            source_scope: source_checkpoint.scope.clone(),
            selected_genes,
            metrics,
        })
    }
}

pub fn write_checkpoint_atomic(
    path: impl AsRef<Path>,
    checkpoint: &SearchCheckpointArtifactFile,
) -> Result<()> {
    write_json_atomic(path, checkpoint)
}

pub fn read_checkpoint_for_resume(
    path: impl AsRef<Path>,
    expected_scope: &SearchCheckpointScope,
) -> Result<SearchCheckpointArtifactFile> {
    let checkpoint: SearchCheckpointArtifactFile = read_json(path, "checkpoint")?;
    checkpoint.validate_for_resume(expected_scope)?;
    Ok(checkpoint)
}

pub fn write_portfolio_artifact_atomic(
    path: impl AsRef<Path>,
    portfolio: &PortfolioSelectionArtifactFile,
) -> Result<()> {
    write_json_atomic(path, portfolio)
}

fn derive_seed(root_seed: u64, label: &[u8], value: u64) -> u64 {
    let mut bytes = Vec::with_capacity(16 + label.len());
    bytes.extend_from_slice(&root_seed.to_le_bytes());
    bytes.extend_from_slice(label);
    bytes.extend_from_slice(&value.to_le_bytes());
    crate::artifact_io::fnv1a64(&bytes)
}

impl Hash for EvaluatedCandidateRecord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.candidate_hash.hash(state);
        self.generation.hash(state);
        self.candidate_index.hash(state);
        self.seed.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact_io::temporary_path;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporal_contract(label_policy_hash: &str) -> TemporalFeatureContract {
        TemporalFeatureContract::strict_live(
            "UTC",
            "alignment-policy-v1",
            label_policy_hash,
            "walk-forward-policy-v1",
            "live-readiness-policy-v1",
        )
        .expect("strict temporal contract should be valid")
    }

    fn sample_gene(weight: f32) -> Gene {
        Gene {
            indices: vec![0, 2],
            weights: vec![weight, -0.25],
            long_threshold: 0.3,
            short_threshold: -0.2,
            strategy_id: format!("sample-{weight}"),
            ..Gene::default()
        }
    }

    fn sample_checkpoint(scope: SearchCheckpointScope) -> SearchCheckpointArtifactFile {
        let gene = sample_gene(0.5);
        let seed_chain = DeterministicSeedChain::new(42);
        let mut ledger = EvaluatedCandidateLedger::default();
        ledger.insert(EvaluatedCandidateRecord::from_gene(&gene, 0, 0, &seed_chain).unwrap());
        SearchCheckpointArtifactFile::new(
            scope,
            1,
            seed_chain,
            ledger,
            SearchResult {
                genes: vec![gene],
                metrics: vec![[1.0; 11]],
            },
        )
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("neoethos-search-{name}-{unique}.json"))
    }

    #[test]
    fn checkpoint_write_is_atomic_and_resume_validates_scope() {
        let scope = SearchCheckpointScope::new(
            "cfg-a",
            "data-a",
            "space-a",
            &temporal_contract("label-policy-v1"),
        );
        let path = temp_path("checkpoint");
        let checkpoint = sample_checkpoint(scope.clone());

        write_checkpoint_atomic(&path, &checkpoint).expect("atomic checkpoint write");
        assert!(!temporary_path(&path).exists());

        let loaded = read_checkpoint_for_resume(&path, &scope).expect("resume scope should match");
        assert_eq!(loaded.artifact_kind, SEARCH_CHECKPOINT_ARTIFACT_KIND);
        assert_eq!(loaded.completed_generations, 1);
        assert_eq!(loaded.search_result().genes.len(), 1);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn checkpoint_resume_rejects_changed_hashes_and_wrong_artifact_kind() {
        let scope = SearchCheckpointScope::new(
            "cfg-a",
            "data-a",
            "space-a",
            &temporal_contract("label-policy-v1"),
        );
        let path = temp_path("mismatch");
        let mut checkpoint = sample_checkpoint(scope.clone());
        checkpoint.artifact_kind = PORTFOLIO_SELECTION_ARTIFACT_KIND.to_string();
        write_checkpoint_atomic(&path, &checkpoint).expect("write wrong-kind payload");
        let err =
            read_checkpoint_for_resume(&path, &scope).expect_err("wrong kind must not resume");
        assert!(err.to_string().contains("cannot be resumed"));

        checkpoint.artifact_kind = SEARCH_CHECKPOINT_ARTIFACT_KIND.to_string();
        write_checkpoint_atomic(&path, &checkpoint).expect("write checkpoint payload");
        let expected = SearchCheckpointScope::new(
            "cfg-b",
            "data-a",
            "space-a",
            &temporal_contract("label-policy-v1"),
        );
        let err = read_checkpoint_for_resume(&path, &expected)
            .expect_err("changed config must not resume");
        assert!(err.to_string().contains("config hash mismatch"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn checkpoint_resume_rejects_temporal_policy_drift() {
        let contract = temporal_contract("label-policy-v1");
        let scope = SearchCheckpointScope::new("cfg-a", "data-a", "space-a", &contract);
        let path = temp_path("temporal-mismatch");
        let checkpoint = sample_checkpoint(scope.clone());
        write_checkpoint_atomic(&path, &checkpoint).expect("write temporal checkpoint payload");

        let changed_label_scope = SearchCheckpointScope::new(
            "cfg-a",
            "data-a",
            "space-a",
            &temporal_contract("label-policy-v2"),
        );
        let err = read_checkpoint_for_resume(&path, &changed_label_scope)
            .expect_err("changed temporal contract must not resume");
        assert!(err.to_string().contains("temporal contract hash mismatch"));

        let mut changed_timestamp_scope = scope.clone();
        changed_timestamp_scope.temporal_scope.timestamp_policy_hash =
            "different-timestamp-policy".to_string();
        let err = read_checkpoint_for_resume(&path, &changed_timestamp_scope)
            .expect_err("changed timestamp policy must not resume");
        assert!(err.to_string().contains("timestamp policy hash mismatch"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn seed_chain_and_candidate_ledger_are_deterministic_and_unique() {
        let seed_chain = DeterministicSeedChain::new(7).for_generation(3);
        assert_eq!(
            seed_chain.candidate_seed(3, 2),
            seed_chain.candidate_seed(3, 2)
        );
        assert_ne!(
            seed_chain.candidate_seed(3, 2),
            seed_chain.candidate_seed(3, 3)
        );

        let record =
            EvaluatedCandidateRecord::from_gene(&sample_gene(0.1), 3, 2, &seed_chain).unwrap();
        let duplicate = record.clone();
        let mut ledger = EvaluatedCandidateLedger::default();
        assert!(ledger.insert(record));
        assert!(!ledger.insert(duplicate));
        assert_eq!(ledger.candidate_hashes().len(), 1);
    }

    #[test]
    fn portfolio_artifact_is_separate_from_checkpoint_artifact() {
        let checkpoint = sample_checkpoint(SearchCheckpointScope::new(
            "cfg",
            "data",
            "space",
            &temporal_contract("label-policy-v1"),
        ));
        let portfolio = PortfolioSelectionArtifactFile::new(
            &checkpoint,
            checkpoint.genes.clone(),
            checkpoint.metrics.clone(),
        )
        .expect("portfolio artifact");
        assert_eq!(checkpoint.artifact_kind, SEARCH_CHECKPOINT_ARTIFACT_KIND);
        assert_eq!(portfolio.artifact_kind, PORTFOLIO_SELECTION_ARTIFACT_KIND);
        assert_eq!(portfolio.source_scope, checkpoint.scope);
        assert_ne!(
            stable_json_hash(&checkpoint).unwrap(),
            stable_json_hash(&portfolio).unwrap()
        );
    }
}
