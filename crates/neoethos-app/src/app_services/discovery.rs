use crate::app_services::{
    ServiceEvent,
    execution_admission::AdmittedCpuLease,
    jobs::{
        CancellationFlag, JobEventLevel, JobKind, JobProgress, JobReport, JobSnapshot, JobState,
        push_recent_event,
    },
};
use crate::app_state::AppExecutionState;
use anyhow::{Context, Result};
use neoethos_broker_history as broker_history;
use neoethos_broker_history::canonical_research_costs::{
    ScreeningCostEnvelopeWireV2, build_screening_cost_envelope_v2, generation_sha256,
    read_bounded_regular_file, validate_broker_symbol_contract, validate_costs,
    validate_exact_file_settings,
};
use neoethos_core::{
    execution::BudgetedCpuScope,
    execution_budget::CpuPermitRequest,
    logging::{canonical_log_path, write_subsystem_record},
    sectioned_log::{SectionedRunRecord, SubsystemSection},
};
#[cfg(any(test, feature = "gpu-nvidia"))]
use neoethos_data::prepare_multitimeframe_features_with_control;
use neoethos_data::{
    CanonicalDatasetIdentity, CanonicalDatasetScope, CanonicalDatasetSeriesReceiptV1,
    CanonicalTimeframe, DatasetDiscovery, FeatureBuildCancelled, FeatureBuildControl,
    FeatureBuildOptions, FeatureBuildProgress, PinnedCanonicalSeriesV1,
    SelectedDatasetGenerationV1, SymbolDataset, discover_canonical_dataset_identities,
    pin_exact_canonical_series_v1, require_direct_timeframes,
};
// `DiscoveryValidationGates` is used by the sibling tests file
// (`discovery_tests.rs::success_snapshot_carries_candidate_and_portfolio_counters`),
// not by anything in this module. Importing it gated on `#[cfg(test)]`
// keeps the release build clean while staying visible to tests via
// `use super::*;`.
#[cfg(test)]
use neoethos_search::DiscoveryValidationGates;
use neoethos_search::data_selection::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchInputReceiptV2,
};
use neoethos_search::{
    DiscoveryConfig, DiscoveryProgress, DiscoveryResult, PromotionSummaryAuthorityPayloadV3,
    PropFirmRiskRules, ensure_non_empty_portfolio,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Original operator settings retained for a Discovery job. The selected
/// symbol/timeframes belong to the request, not to this file identity. This is
/// source ownership only; it cannot authorize financial evaluation or trading.
pub struct DiscoverySettingsSource {
    settings: neoethos_core::Settings,
    source_path: PathBuf,
    discovery_cache_root: PathBuf,
    exact_bytes: Vec<u8>,
    sha256: String,
}

impl std::fmt::Debug for DiscoverySettingsSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Settings may contain private operator data: never dump their values
        // or raw bytes when a job/request is formatted for diagnostics.
        formatter
            .debug_struct("DiscoverySettingsSource")
            .field("source_path", &self.source_path)
            .field("byte_count", &self.exact_bytes.len())
            .field("sha256", &self.sha256)
            .finish_non_exhaustive()
    }
}

impl DiscoverySettingsSource {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let exact_bytes = read_bounded_regular_file(path)?;
        let settings = neoethos_core::Settings::from_yaml(path)?;
        validate_exact_file_settings(&settings, path, &exact_bytes)?;
        // Configured relative paths are CWD-relative, not YAML-parent-relative.
        // Capture once without requiring the cache to exist or changing Settings.
        let discovery_cache_root = std::path::absolute(&settings.system.cache_dir)
            .context("resolve configured Discovery cache root")?
            .join("discovery");
        Ok(Self {
            settings,
            source_path: std::fs::canonicalize(path)?,
            discovery_cache_root,
            sha256: format!("{:x}", Sha256::digest(&exact_bytes)),
            exact_bytes,
        })
    }

    pub(crate) fn settings(&self) -> &neoethos_core::Settings {
        &self.settings
    }

    fn discovery_cache_root(&self) -> PathBuf {
        self.discovery_cache_root.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedDiscoveryGenerationOverrideV1 {
    Exact(usize),
    Floor(usize),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TypedDiscoveryOverridesV1 {
    population: Option<usize>,
    generation_policy: Option<TypedDiscoveryGenerationOverrideV1>,
    max_indicators: Option<usize>,
    max_rows: Option<usize>,
    target_candidates: Option<usize>,
    portfolio_size: Option<usize>,
}

impl TypedDiscoveryOverridesV1 {
    pub(crate) fn checked_new(
        population: Option<usize>,
        generation_policy: Option<TypedDiscoveryGenerationOverrideV1>,
        max_indicators: Option<usize>,
        max_rows: Option<usize>,
        target_candidates: Option<usize>,
        portfolio_size: Option<usize>,
    ) -> Result<Self, &'static str> {
        let values = [population, max_indicators, max_rows, portfolio_size];
        if values.into_iter().flatten().any(|value| value == 0) {
            return Err("typed Discovery overrides must be nonzero when supplied");
        }
        // Unlike dimensions above, target_candidates=0 explicitly removes a
        // saved cap and validates every candidate returned by the GA.
        if generation_policy.is_some_and(|policy| match policy {
            TypedDiscoveryGenerationOverrideV1::Exact(value)
            | TypedDiscoveryGenerationOverrideV1::Floor(value) => value == 0,
        }) {
            return Err("typed Discovery generation override must be nonzero");
        }
        Ok(Self {
            population,
            generation_policy,
            max_indicators,
            max_rows,
            target_candidates,
            portfolio_size,
        })
    }

    pub(crate) fn apply(&self, config: &mut DiscoveryConfig) {
        if let Some(population) = self.population {
            config.population = population;
        }
        if let Some(policy) = self.generation_policy {
            config.generations = match policy {
                TypedDiscoveryGenerationOverrideV1::Exact(value) => value,
                TypedDiscoveryGenerationOverrideV1::Floor(value) => config.generations.max(value),
            };
        }
        if let Some(max_indicators) = self.max_indicators {
            config.max_indicators = max_indicators;
        }
        if let Some(max_rows) = self.max_rows {
            config.max_rows = max_rows;
        }
        if let Some(target_candidates) = self.target_candidates {
            config.candidate_count = target_candidates;
        }
        if let Some(portfolio_size) = self.portfolio_size {
            config.portfolio_size = portfolio_size;
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryRequest {
    pub data_root: PathBuf,
    /// Exact original file and resolved values. Later source checks may refuse
    /// changed bytes, but never replace this snapshot with newer Settings.
    pub settings_source: Arc<DiscoverySettingsSource>,
    /// Exact manifests plus reader leases, with no decoded OHLCV values. The
    /// selected prepared CPU/native factory consumes this pin after device
    /// admission, so the worker cannot follow a newer `current` pointer.
    pub pinned_input: Arc<PinnedDiscoveryInput>,
    pub higher_tfs: Vec<String>,
    /// Financial configuration is absent until the worker has a real feature
    /// receipt and has sealed its source-bound research costs.
    pub config: Option<DiscoveryConfig>,
    pub(crate) overrides: TypedDiscoveryOverridesV1,
    /// Prop-firm rule set applied to the OOS prop-firm validation pass.
    /// Defaults to `PropFirmRiskRules::default()` (FTMO-style) when the
    /// caller does not need to override per-challenge thresholds.
    pub prop_firm_rules: PropFirmRiskRules,
}

impl DiscoveryRequest {
    pub fn symbol(&self) -> &str {
        self.dataset_identity().symbol_name()
    }

    pub fn base_tf(&self) -> &'static str {
        self.dataset_identity().timeframe().as_str()
    }

    pub fn dataset_identity(&self) -> &CanonicalDatasetIdentity {
        self.pinned_input.receipt().anchor().identity()
    }

    pub fn validate(&self) -> Result<()> {
        if self.data_root.as_os_str().is_empty() {
            anyhow::bail!("discovery request data root must not be empty");
        }
        let higher = self.canonical_higher_timeframes()?;
        self.pinned_input.validate(&higher)?;
        if let Some(config) = &self.config {
            self.validate_resolved_config(config)?;
        }
        Ok(())
    }

    fn validate_resolved_config(&self, config: &DiscoveryConfig) -> Result<()> {
        let higher = self.canonical_higher_timeframes()?;
        anyhow::ensure!(
            config.evaluation_symbol == self.symbol(),
            "discovery config symbol differs from the pinned dataset; resolve costs for the selected symbol before launch"
        );
        anyhow::ensure!(
            config.timeframe_label == self.base_tf(),
            "discovery config timeframe differs from the pinned dataset; resolve mode filters for the selected timeframe before launch"
        );
        anyhow::ensure!(
            config.higher_timeframes == higher,
            "discovery config higher timeframes differ from the pinned feature selection"
        );
        Ok(())
    }

    fn canonical_higher_timeframes(&self) -> Result<Vec<String>> {
        canonical_higher_timeframes(self.dataset_identity().timeframe(), &self.higher_tfs)
    }

    fn execution_config(&self) -> Result<DiscoveryConfig> {
        let config = self.config.as_ref().context(
            "Discovery financial configuration has not been sealed against the actual feature receipt",
        )?;
        self.validate_resolved_config(config)?;
        // Resolution applies the mode once; consumers must not rescale it.
        Ok(config.clone())
    }

    fn resolve_research_config(
        &self,
        contract: &neoethos_search::CanonicalTrendbarResearchExecutionContractV3,
    ) -> Result<DiscoveryConfig> {
        let mut run_settings = self.settings_source.settings().clone();
        run_settings.system.symbol = self.symbol().to_owned();
        run_settings.system.base_timeframe = self.base_tf().to_owned();
        run_settings.system.higher_timeframes = self.canonical_higher_timeframes()?;
        let mut config = DiscoveryConfig::try_from_settings_for_canonical_trendbar_research(
            &run_settings,
            contract,
        )?;
        self.overrides.apply(&mut config);
        self.validate_resolved_config(&config)?;
        Ok(config)
    }

    fn feature_build_options(&self, base_row_count: usize) -> Result<FeatureBuildOptions> {
        // Use Search's exact outer split, before any feature work. Fitting
        // normalization on the full frame would consume held-out observations.
        let training_rows =
            neoethos_search::canonical_discovery_normalization_training_rows(base_row_count)?;
        Ok(FeatureBuildOptions {
            higher_tfs: self.canonical_higher_timeframes()?,
            prefix_base_features: self
                .settings_source
                .settings()
                .system
                .multi_resolution_prefix_base,
            normalization_training_rows: Some(training_rows),
            // A structure absent throughout training cannot be normalized or
            // searched. Reuse the existing provenance-recorded projection;
            // never fit it on holdout rows or abort all other usable features.
            drop_columns_without_normalization_training_support: true,
            ..FeatureBuildOptions::default()
        })
    }
}

/// One immutable, directly downloaded/imported timeframe set held for the
/// complete discovery lifetime. Construction is private to the exact pinning
/// functions below so callers cannot pair a receipt with unrelated values.
#[derive(Debug)]
pub struct PinnedDiscoveryInput {
    receipt: CanonicalDatasetSeriesReceiptV1,
    pinned_series: Mutex<Option<PinnedCanonicalSeriesV1>>,
}

impl PinnedDiscoveryInput {
    pub const fn receipt(&self) -> &CanonicalDatasetSeriesReceiptV1 {
        &self.receipt
    }

    fn validate(&self, higher_tfs: &[String]) -> Result<()> {
        self.receipt.validate()?;
        let anchor = self.receipt.anchor().identity();
        let mut required = vec![anchor.timeframe()];
        for label in higher_tfs {
            let timeframe = label
                .parse::<CanonicalTimeframe>()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            if !required.contains(&timeframe) {
                required.push(timeframe);
            }
        }
        let received = self
            .receipt
            .direct_timeframes()
            .iter()
            .map(|selected| selected.identity().timeframe())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            required.len() == received.len()
                && required
                    .iter()
                    .all(|timeframe| received.contains(timeframe)),
            "pinned discovery receipt does not exactly match the requested direct timeframe set"
        );
        Ok(())
    }

    fn take_pinned_series_v1(&self) -> Result<PinnedCanonicalSeriesV1> {
        self.pinned_series
            .lock()
            .map_err(|_| anyhow::anyhow!("pinned canonical series lock is poisoned"))?
            .take()
            .context("pinned canonical series was already consumed by another factory")
    }
}

#[derive(Debug)]
pub struct DirectTimeframeAcquisitionRequired {
    missing: Vec<CanonicalDatasetIdentity>,
}

impl std::fmt::Display for DirectTimeframeAcquisitionRequired {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let missing = self
            .missing
            .iter()
            .map(CanonicalDatasetIdentity::to_path_component)
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            formatter,
            "direct timeframe acquisition required before discovery: [{missing}]"
        )
    }
}

impl std::error::Error for DirectTimeframeAcquisitionRequired {}

/// Resolve metadata for the operator-selected anchor and every explicitly
/// requested higher timeframe, then pin exact manifests and reader leases
/// without decoding values. Missing data is an acquisition request, never an
/// instruction for the discovery worker to mutate its own inputs.
pub fn pin_discovery_input(
    root: &std::path::Path,
    anchor: SelectedDatasetGenerationV1,
    higher_tfs: &[String],
) -> Result<PinnedDiscoveryInput> {
    anchor.validate()?;
    anyhow::ensure!(!root.as_os_str().is_empty(), "discovery data root is empty");
    anyhow::ensure!(
        root.is_dir(),
        "discovery data root is not a directory: {}",
        root.display()
    );

    let base = anchor.identity().timeframe();
    let higher = canonical_higher_timeframes(base, higher_tfs)?;
    let mut required = vec![base];
    required.extend(higher.iter().map(|label| {
        label
            .parse::<CanonicalTimeframe>()
            .expect("canonical higher timeframe was already parsed")
    }));

    let inventory = DatasetDiscovery::scan_metadata(root)?;
    let mut selected = Vec::with_capacity(required.len());
    let mut missing = Vec::new();
    for timeframe in required {
        if timeframe == base {
            selected.push(anchor.clone());
            continue;
        }
        let identity = identity_for_timeframe(anchor.identity(), timeframe)?;
        let identity_path = identity.to_path_component();
        let matches = inventory
            .entries
            .iter()
            .filter(|entry| entry.dataset_identity == identity_path)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => {
                let expected_root = root.join(&identity_path);
                if expected_root.exists() {
                    let diagnostics = inventory
                        .skipped
                        .iter()
                        .filter(|skipped| skipped.path.starts_with(&expected_root))
                        .map(|skipped| {
                            format!("{}: {}", skipped.reason.category(), skipped.reason.detail())
                        })
                        .collect::<Vec<_>>();
                    anyhow::bail!(
                        "direct timeframe dataset {identity_path} exists but is not a verified canonical generation: [{}]",
                        diagnostics.join("; ")
                    );
                }
                missing.push(identity);
            }
            [entry] => selected.push(SelectedDatasetGenerationV1::new(
                identity,
                entry.generation.clone(),
                entry.manifest_binding_sha256.clone(),
            )?),
            _ => anyhow::bail!(
                "canonical inventory contains duplicate entries for exact identity {identity_path}"
            ),
        }
    }
    if !missing.is_empty() {
        return Err(DirectTimeframeAcquisitionRequired { missing }.into());
    }

    let receipt = CanonicalDatasetSeriesReceiptV1::new(anchor, selected.clone())?;
    let pinned_series = pin_exact_canonical_series_v1(root, receipt.clone())?;
    let pinned = PinnedDiscoveryInput {
        receipt,
        pinned_series: Mutex::new(Some(pinned_series)),
    };
    pinned.validate(&higher)?;
    Ok(pinned)
}

/// Non-interactive callers first resolve exactly one canonical identity, then
/// snapshot its current metadata into a typed receipt and enter the same exact
/// generation pinning path as the HTTP API.
pub fn pin_current_discovery_input(
    root: &std::path::Path,
    identity: &CanonicalDatasetIdentity,
    higher_tfs: &[String],
) -> Result<PinnedDiscoveryInput> {
    let manifest =
        neoethos_data::core::dataset_manifest::read_current_manifest_metadata(root, identity)?;
    pin_discovery_input(
        root,
        SelectedDatasetGenerationV1::from_manifest(&manifest)?,
        higher_tfs,
    )
}

fn canonical_higher_timeframes(
    base: CanonicalTimeframe,
    higher_tfs: &[String],
) -> Result<Vec<String>> {
    let mut parsed = Vec::with_capacity(higher_tfs.len());
    for raw in higher_tfs {
        let label = raw.trim().to_uppercase();
        let timeframe = label
            .parse::<CanonicalTimeframe>()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            timeframe > base,
            "higher timeframe {timeframe} must be strictly above base {base}"
        );
        anyhow::ensure!(
            !parsed.contains(&timeframe),
            "duplicate higher timeframe {timeframe}"
        );
        parsed.push(timeframe);
    }
    Ok(parsed
        .into_iter()
        .map(|timeframe| timeframe.as_str().to_owned())
        .collect())
}

/// Background jobs do not have an interactive dataset picker. They may reuse
/// a symbol/timeframe hint only when it resolves to exactly one canonical
/// source/account series. Zero or multiple matches fail closed and print every
/// candidate identity so the operator can make the selection explicit.
pub fn resolve_unique_background_dataset_identity(
    root: &std::path::Path,
    symbol: &str,
    base_tf: &str,
) -> Result<CanonicalDatasetIdentity> {
    let identities = discover_canonical_dataset_identities(root, symbol)?;
    select_unique_background_identity(identities, symbol, base_tf)
}

fn select_unique_background_identity(
    identities: Vec<CanonicalDatasetIdentity>,
    symbol: &str,
    base_tf: &str,
) -> Result<CanonicalDatasetIdentity> {
    let timeframe = base_tf
        .trim()
        .to_uppercase()
        .parse::<CanonicalTimeframe>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut all_candidates = identities
        .iter()
        .map(CanonicalDatasetIdentity::to_path_component)
        .collect::<Vec<_>>();
    all_candidates.sort();
    let mut matches = identities
        .into_iter()
        .filter(|identity| {
            identity.symbol_name().eq_ignore_ascii_case(symbol) && identity.timeframe() == timeframe
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(CanonicalDatasetIdentity::to_path_component);
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => anyhow::bail!(
            "background discovery found no exact canonical identity for {} {}; candidates=[{}]",
            symbol.trim().to_uppercase(),
            timeframe,
            all_candidates.join(", ")
        ),
        count => anyhow::bail!(
            "background discovery found {count} canonical identities for {} {}; explicit dataset identity required; candidates=[{}]",
            symbol.trim().to_uppercase(),
            timeframe,
            matches
                .iter()
                .map(CanonicalDatasetIdentity::to_path_component)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn identity_for_timeframe(
    selected: &CanonicalDatasetIdentity,
    timeframe: CanonicalTimeframe,
) -> Result<CanonicalDatasetIdentity> {
    let convention = selected.bar_timestamp_convention();
    match selected.scope() {
        CanonicalDatasetScope::External { source_namespace } => CanonicalDatasetIdentity::external(
            source_namespace.clone(),
            selected.symbol_name(),
            timeframe,
            convention,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string())),
        CanonicalDatasetScope::CTrader {
            environment,
            server,
            account_id,
            symbol_id,
        } => CanonicalDatasetIdentity::ctrader(
            *environment,
            server.clone(),
            *account_id,
            *symbol_id,
            selected.symbol_name(),
            timeframe,
            convention,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string())),
    }
}

fn required_direct_timeframes(request: &DiscoveryRequest) -> Result<Vec<CanonicalTimeframe>> {
    let mut required = Vec::new();
    let mut push = |timeframe: CanonicalTimeframe| {
        if !required.contains(&timeframe) {
            required.push(timeframe);
        }
    };
    push(request.dataset_identity().timeframe());
    for label in &request.higher_tfs {
        push(
            label
                .trim()
                .to_uppercase()
                .parse::<CanonicalTimeframe>()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
        );
    }
    Ok(required)
}

fn validate_direct_timeframe_artifacts(
    dataset: &SymbolDataset,
    selected: &CanonicalDatasetIdentity,
    required: &[CanonicalTimeframe],
) -> Result<()> {
    require_direct_timeframes(dataset, selected, required)
}

#[cfg(test)]
fn prepare_cpu_discovery_features(
    request: &DiscoveryRequest,
    dataset: SymbolDataset,
    required_direct: &[CanonicalTimeframe],
) -> Result<neoethos_search::data_selection::CanonicalSearchInput> {
    prepare_cpu_discovery_features_with_control(
        request,
        dataset,
        required_direct,
        &FeatureBuildControl::default(),
    )
}

#[cfg(any(test, feature = "gpu-nvidia"))]
fn prepare_cpu_discovery_features_with_control(
    request: &DiscoveryRequest,
    dataset: SymbolDataset,
    required_direct: &[CanonicalTimeframe],
    control: &FeatureBuildControl,
) -> Result<neoethos_search::data_selection::CanonicalSearchInput> {
    control.checkpoint()?;
    validate_direct_timeframe_artifacts(&dataset, request.dataset_identity(), required_direct)?;
    let base_rows = dataset
        .timeframe(request.base_tf())
        .context("validated Discovery base timeframe is missing")?
        .len();
    let options = request.feature_build_options(base_rows)?;
    let features = prepare_multitimeframe_features_with_control(
        &dataset,
        request.base_tf(),
        &options,
        control,
    )?;
    control.checkpoint()?;
    let base_frame = dataset.into_canonical_frame(request.base_tf())?;
    control.report(
        "feature_receipt",
        "hashing the complete prepared input",
        0,
        1,
    )?;
    neoethos_search::data_selection::CanonicalSearchInput::from_prepared_canonical_frame_with_control(
        request.dataset_identity().clone(),
        base_frame,
        features,
        control,
    )
    .map_err(anyhow::Error::new)
}

struct PreparedDiscoveryScreeningCosts {
    selected_series: CanonicalDatasetSeriesReceiptV1,
    settings_source_sha256: String,
    envelope: ScreeningCostEnvelopeWireV2,
    pip_value_per_lot: f64,
    // Retain the exact D1 generations while the worker owns their cost basis.
    _basis_pins: Vec<PinnedDiscoveryInput>,
}

// One retained source generation and cost basis, never a union of feature cubes.
// Each batch gets local column indices and its own exact replay recipe.
struct CpuDiscoveryWorkingSet {
    #[cfg(not(feature = "gpu-nvidia"))]
    dataset: SymbolDataset,
    #[cfg(not(feature = "gpu-nvidia"))]
    costs: Arc<PreparedDiscoveryScreeningCosts>,
    search: neoethos_search::discovery::StreamingSearch,
    #[cfg(not(feature = "gpu-nvidia"))]
    selected_cursor: usize,
}

#[cfg(not(feature = "gpu-nvidia"))]
fn prepare_cpu_discovery_batch_with_control(
    request: &DiscoveryRequest,
    dataset: &SymbolDataset,
    required_direct: &[CanonicalTimeframe],
    batch: Arc<neoethos_data::core::hpc_ta::SweepBatch>,
    control: &FeatureBuildControl,
) -> Result<neoethos_search::data_selection::CanonicalSearchInput> {
    control.checkpoint()?;
    validate_direct_timeframe_artifacts(dataset, request.dataset_identity(), required_direct)?;
    let base_rows = dataset
        .timeframe(request.base_tf())
        .context("validated Discovery base timeframe is missing")?
        .len();
    let options = request.feature_build_options(base_rows)?;
    let features = neoethos_data::prepare_multitimeframe_features_batch_with_options_and_control(
        dataset,
        request.base_tf(),
        &options,
        Some(batch),
        control,
    )?;
    control.checkpoint()?;
    // Retain the original raw dataset for the next batch. Only this batch's
    // base OHLCV is copied; completed feature cubes are not retained by the job.
    let base_frame = dataset.canonical_frame(request.base_tf())?;
    control.report(
        "feature_receipt",
        "hashing the complete prepared batch",
        0,
        1,
    )?;
    neoethos_search::data_selection::CanonicalSearchInput::from_prepared_canonical_frame_with_control(
        request.dataset_identity().clone(), base_frame, features, control,
    ).map_err(anyhow::Error::new)
}

#[derive(Default)]
struct DiscoveryWorkingSetProgress {
    active: u64,
    completed: u64,
    completed_entries: u64,
    total_entries: u64,
    handoffs: u64,
    publication_failures: u64,
    single_handoff: Option<String>,
}

impl DiscoveryWorkingSetProgress {
    fn apply(&self, report: &mut JobReport, seed: u64) {
        for (name, value) in [
            ("working_set_batch", self.active),
            ("working_set_completed_batches", self.completed),
            ("working_set_completed_entries", self.completed_entries),
            ("working_set_total_entries", self.total_entries),
            ("working_set_saved_results", self.completed),
            ("working_set_training_handoffs", self.handoffs),
            (
                "working_set_publication_failures",
                self.publication_failures,
            ),
        ] {
            upsert_counter(&mut report.counters, name, value);
        }
        report
            .highlights
            .retain(|(name, _)| name != "working_set_seed");
        report
            .highlights
            .push(("working_set_seed".into(), seed.to_string()));
    }

    fn summary(&self, ending: &str) -> String {
        format!(
            "{ending}: {} completed batches / {} selection entries; {} saved research reports, {} final-evaluation handoffs, {} publication failures. Each batch retains the configured population/generations; broker/account/risk admission remains separate",
            self.completed,
            self.completed_entries,
            self.completed,
            self.handoffs,
            self.publication_failures
        )
    }
}

#[derive(Default)]
struct DiscoveryDeadline {
    started: bool,
    expired: Arc<std::sync::atomic::AtomicBool>,
    timer: Option<tokio::task::JoinHandle<()>>,
}

impl DiscoveryDeadline {
    fn start_once(&mut self, hours: f64, cancel: CancellationFlag) -> Result<()> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        anyhow::ensure!(
            hours.is_finite() && hours >= 0.0,
            "invalid overall Discovery time budget"
        );
        if hours > 0.0 {
            let duration = std::time::Duration::try_from_secs_f64(hours * 3600.0)
                .context("overall Discovery time budget is out of range")?;
            let expired = Arc::clone(&self.expired);
            self.timer = Some(tokio::spawn(async move {
                tokio::time::sleep(duration).await;
                if !cancel.is_requested() {
                    expired.store(true, std::sync::atomic::Ordering::SeqCst);
                    cancel.request();
                }
            }));
        }
        Ok(())
    }

    fn cancelled(&self, snapshot: JobSnapshot, operator_message: &str) -> JobSnapshot {
        let expired = self.expired.load(std::sync::atomic::Ordering::SeqCst);
        let message = if expired {
            "overall Discovery time budget exhausted; incomplete batch stopped; previously saved batches retained"
        } else {
            operator_message
        };
        let mut snapshot = cancelled_snapshot_from(snapshot, message);
        if expired {
            snapshot.progress.stage = "overall_time_budget_exhausted".into();
        }
        snapshot
    }
}

impl Drop for DiscoveryDeadline {
    fn drop(&mut self) {
        if let Some(timer) = self.timer.take() {
            timer.abort();
        }
    }
}

fn pin_discovery_cost_basis(
    request: &DiscoveryRequest,
    identity: &CanonicalDatasetIdentity,
) -> Result<PinnedDiscoveryInput> {
    if let Some(selected) = request
        .pinned_input
        .receipt()
        .direct_timeframes()
        .iter()
        .find(|selected| selected.identity() == identity)
    {
        pin_discovery_input(&request.data_root, selected.clone(), &[])
    } else {
        pin_current_discovery_input(&request.data_root, identity, &[])
    }
}

fn prepare_discovery_screening_costs(
    request: &DiscoveryRequest,
    cancel: &CancellationFlag,
) -> Result<PreparedDiscoveryScreeningCosts> {
    let selection_bytes = serde_json::to_vec(request.pinned_input.receipt())?;
    let source_key = format!("{:x}", Sha256::digest(&selection_bytes));
    let output_root = request
        .settings_source
        .discovery_cache_root()
        .join("sources")
        .join(source_key);
    prepare_discovery_screening_costs_with(request, cancel, &output_root, |capture| {
        if let Some(receipt) =
            broker_history::symbol_contract_cli::reopen_cached_research_symbol_contract_v1(capture)?
        {
            tracing::warn!(path = %receipt.full_symbol_path().display(),
                "Discovery reuses captured broker metadata for offline screening; metadata is not refreshed and this is not live financial authority");
            return Ok(receipt);
        }
        broker_history::symbol_contract_cli::capture_exact_production_broker_symbol_contract_v1(
            capture,
        )
    })
}

fn prepare_discovery_screening_costs_with<F>(
    request: &DiscoveryRequest,
    cancel: &CancellationFlag,
    output_root: &Path,
    capture_symbol: F,
) -> Result<PreparedDiscoveryScreeningCosts>
where
    F: FnOnce(
        &broker_history::symbol_contract_cli::PreparedExactBrokerSymbolContractCaptureV1,
    ) -> Result<broker_history::ExactBrokerSymbolContractReceiptV1>,
{
    anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
    let source = &request.settings_source;
    validate_exact_file_settings(source.settings(), &source.source_path, &source.exact_bytes)?;
    let CanonicalDatasetScope::CTrader {
        environment,
        server,
        account_id,
        symbol_id,
    } = request.dataset_identity().scope()
    else {
        anyhow::bail!(
            "Discovery screening costs require an exact broker-bound dataset; an imported price file alone has no broker cost source"
        );
    };
    let symbol = request.symbol();
    anyhow::ensure!(
        symbol.len() == 6 && symbol.bytes().all(|byte| byte.is_ascii_uppercase()),
        "Discovery screening cost conversion requires an explicit six-letter currency symbol"
    );
    let quote_currency = &symbol[3..];
    let account_currency = source.settings().system.account_currency.trim();
    anyhow::ensure!(
        account_currency.len() == 3
            && account_currency
                .bytes()
                .all(|byte| byte.is_ascii_uppercase()),
        "Discovery account currency is not canonical"
    );
    let mut identities = vec![identity_for_timeframe(
        request.dataset_identity(),
        CanonicalTimeframe::D1,
    )?];
    if quote_currency != account_currency {
        let mut conversion = Vec::new();
        for name in [
            format!("{quote_currency}{account_currency}"),
            format!("{account_currency}{quote_currency}"),
        ] {
            for identity in discover_canonical_dataset_identities(&request.data_root, &name)? {
                if identity.timeframe() == CanonicalTimeframe::D1
                    && matches!(identity.scope(), CanonicalDatasetScope::CTrader {
                        environment: candidate_environment, server: candidate_server,
                        account_id: candidate_account, ..
                    } if candidate_environment == environment && candidate_server == server && candidate_account == account_id)
                {
                    conversion.push(identity);
                }
            }
        }
        anyhow::ensure!(
            conversion.len() == 1,
            "Discovery requires exactly one direct same-account D1 conversion series from {quote_currency} to {account_currency}; found {}",
            conversion.len()
        );
        if !identities.contains(&conversion[0]) {
            identities.push(conversion.remove(0));
        }
    }
    identities.sort_by(|left, right| left.symbol_name().cmp(right.symbol_name()));
    let mut pins = Vec::with_capacity(identities.len());
    let mut symbols = Vec::with_capacity(identities.len());
    let mut cells = Vec::with_capacity(identities.len());
    let mut requested_window = None;
    for identity in identities {
        anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
        let pin = pin_discovery_cost_basis(request, &identity).with_context(|| {
            format!(
                "direct D1 cost basis acquisition required for {}",
                identity.symbol_name()
            )
        })?;
        let selected = pin.receipt().anchor().clone();
        let (manifest, _lease) =
            neoethos_data::open_exact_dataset_generation(&request.data_root, &selected)?;
        let provenance =
            broker_history::bootstrap_writer::CTraderTrendbarProvenanceV1::from_envelope(
                manifest.provenance(),
            )?;
        let window = provenance.requested_range_ms();
        anyhow::ensure!(
            provenance.dataset_identity() == &identity
                && provenance.row_count() == manifest.row_count(),
            "Discovery cost basis provenance differs from its exact selected generation"
        );
        if let Some(expected) = requested_window {
            anyhow::ensure!(
                window == expected,
                "Discovery D1 cost/conversion sources require the same exact acquisition window"
            );
        } else {
            requested_window = Some(window);
        }
        let CanonicalDatasetScope::CTrader { symbol_id, .. } = identity.scope() else {
            unreachable!()
        };
        symbols.push(broker_history::CanonicalTrendbarSymbolV1::new(
            *symbol_id,
            identity.symbol_name(),
        )?);
        cells.push(broker_history::CanonicalTrendbarAcquisitionCellV1::new(
            selected,
        )?);
        pins.push(pin);
    }
    let (from_ms, to_ms) = requested_window.context("Discovery has no direct cost basis")?;
    let plan = broker_history::CanonicalTrendbarAcquisitionPlanV1::new(
        *environment,
        server,
        *account_id,
        from_ms,
        to_ms,
        symbols,
        vec![CanonicalTimeframe::D1],
    )?;
    let broker_environment = match environment {
        neoethos_data::CTraderEnvironment::Demo => broker_history::BrokerEnvironment::Demo,
        neoethos_data::CTraderEnvironment::Live => broker_history::BrokerEnvironment::Live,
    };
    let binding = broker_history::ExactBrokerSymbolContractBindingV1::new(
        broker_environment,
        *account_id,
        *symbol_id,
        symbol,
    )?;
    anyhow::ensure!(
        binding.server() == server,
        "Discovery selected broker server differs from the metadata capture endpoint"
    );
    // Adopt only already-published exact D1 generations. This does not fetch,
    // mutate or replace price history, nor turn current metadata into historical truth.
    let store = broker_history::CanonicalTrendbarAcquisitionStoreV1::new(output_root);
    let plan_receipt = store.publish_plan(&plan)?;
    let checkpoint = store.publish_checkpoint(&request.data_root, &plan_receipt, None, cells)?;
    let matrix_receipt = store.publish_matrix(&request.data_root, &plan_receipt, &checkpoint)?;
    anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
    let capture =
        broker_history::symbol_contract_cli::PreparedExactBrokerSymbolContractCaptureV1::new(
            binding.clone(),
            output_root.to_path_buf(),
        )?;
    let broker_receipt = capture_symbol(&capture)?;
    anyhow::ensure!(
        broker_receipt.binding() == &binding,
        "Discovery metadata capture returned another account or symbol"
    );
    anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
    let envelope = build_screening_cost_envelope_v2(
        source.settings(),
        &request.data_root,
        &store,
        &plan_receipt,
        &matrix_receipt,
        symbol,
        CanonicalTimeframe::D1,
        broker_receipt.full_symbol_path(),
        &source.source_path,
    )?;
    let broker_bytes = read_bounded_regular_file(broker_receipt.full_symbol_path())?;
    anyhow::ensure!(
        format!("{:x}", Sha256::digest(&broker_bytes)) == broker_receipt.full_symbol_sha256(),
        "Discovery broker symbol source changed after capture"
    );
    let broker_facts = validate_broker_symbol_contract(&broker_bytes, &envelope, symbol, &plan)?;
    let matrix = store.open_matrix(&request.data_root, &plan_receipt, &matrix_receipt)?;
    let pip_value_per_lot = validate_costs(
        &envelope,
        symbol,
        source.settings(),
        &plan,
        &matrix,
        &request.data_root,
        broker_facts,
    )?;
    Ok(PreparedDiscoveryScreeningCosts {
        selected_series: request.pinned_input.receipt().clone(),
        settings_source_sha256: source.sha256.clone(),
        envelope,
        pip_value_per_lot,
        _basis_pins: pins,
    })
}

fn seal_discovery_research_contract(
    request: &DiscoveryRequest,
    receipt: neoethos_search::CanonicalSearchInputReceiptV2,
    costs: &PreparedDiscoveryScreeningCosts,
) -> Result<neoethos_search::CanonicalTrendbarResearchExecutionContractV3> {
    anyhow::ensure!(
        receipt.validate()? == *request.dataset_identity(),
        "Discovery feature receipt differs from the requested exact dataset identity"
    );
    anyhow::ensure!(
        receipt.source_bindings().len() == request.pinned_input.receipt().direct_timeframes().len(),
        "Discovery feature receipt does not cover the complete pinned timeframe set"
    );
    for binding in receipt.source_bindings() {
        let selected = request
            .pinned_input
            .receipt()
            .direct_timeframes()
            .iter()
            .find(|selected| selected.identity().to_path_component() == binding.dataset_identity())
            .context("Discovery feature receipt contains an unselected source")?;
        anyhow::ensure!(
            binding.generation_id() == selected.generation_id()
                && binding.manifest_sha256() == selected.manifest_binding_sha256()
                && binding.vortex_sha256() == generation_sha256(selected)?,
            "Discovery feature receipt differs from its pinned generation/manifest/bytes"
        );
    }
    anyhow::ensure!(
        costs.selected_series == *request.pinned_input.receipt()
            && costs.settings_source_sha256 == request.settings_source.sha256,
        "Discovery screening costs belong to another pinned series or Settings source"
    );
    let envelope = &costs.envelope;
    let bytes = serde_json::to_vec(envelope)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let contract = neoethos_search::CanonicalTrendbarResearchExecutionContractV3::new(
        receipt.clone(),
        neoethos_search::CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: &envelope.symbol,
            account_currency: &envelope.account_currency,
            assumption_source_id: &envelope.assumption_source_id,
            assumption_source_sha256: &hash,
            pip_size: envelope.pip_size,
            pip_value_per_lot: costs.pip_value_per_lot,
            full_spread_pips_assumption: envelope.full_spread_pips_assumption,
            slippage_pips_per_fill_assumption: envelope.slippage_pips_per_fill_assumption,
            commission_account_per_lot_per_fill_assumption: envelope
                .commission_account_per_lot_per_fill_assumption,
            swap_long_pips_per_day: envelope.swap_long_pips_per_day,
            swap_short_pips_per_day: envelope.swap_short_pips_per_day,
            pnl_conversion_fee_rate: envelope.pnl_conversion_fee_rate,
        },
    )?;
    contract.validate_against_receipt(&receipt)?;
    Ok(contract)
}

fn waiting_for_worker_message(context: &str, elapsed_secs: u64, cancel_requested: bool) -> String {
    let state = if cancel_requested {
        "Stop requested; waiting for the worker to return"
    } else {
        "awaiting engine progress"
    };
    format!(
        "{context} · {state} — {}m {:02}s elapsed",
        elapsed_secs / 60,
        elapsed_secs % 60
    )
}

fn apply_feature_preparation_progress(snapshot: &mut JobSnapshot, progress: &FeatureBuildProgress) {
    snapshot.progress = JobProgress {
        percent: None,
        stage: format!("preparing_{}", progress.stage),
        message: format!(
            "{} · {} · {} · {}/{} stage units",
            progress.timeframe, progress.stage, progress.item, progress.completed, progress.total
        ),
    };
    upsert_counter(
        &mut snapshot.report.counters,
        "preparation_stage_completed",
        progress.completed as u64,
    );
    upsert_counter(
        &mut snapshot.report.counters,
        "preparation_stage_total",
        progress.total as u64,
    );
}

#[derive(Debug, Clone)]
pub struct DiscoveryJobHandle {
    pub snapshot: JobSnapshot,
    pub cancel: CancellationFlag,
}

impl DiscoveryJobHandle {
    pub fn new() -> Self {
        Self {
            snapshot: JobSnapshot::new(JobKind::Discovery),
            cancel: CancellationFlag::new(),
        }
    }
}

fn unprepared_validation_candidate_target(settings: &neoethos_core::Settings) -> usize {
    // Zero means every candidate returned by the GA, whose eventual pool size
    // is not known while queued. Positive values are explicit operator caps.
    settings.models.prop_search_val_candidates
}

fn requested_discovery_counters(request: &DiscoveryRequest) -> Vec<(String, u64)> {
    // Until financial preparation finishes these are requested dimensions only,
    // not a usable DiscoveryConfig or a claim that Search has started.
    let (candidates, portfolio, generations, population, max_rows) =
        if let Some(config) = &request.config {
            (
                config.candidate_count,
                config.portfolio_size,
                config.generations,
                config.population,
                config.max_rows,
            )
        } else {
            let models = &request.settings_source.settings().models;
            let overrides = &request.overrides;
            let requested_candidates =
                unprepared_validation_candidate_target(request.settings_source.settings());
            let generations = models.prop_search_generations.max(1);
            let generations = match overrides.generation_policy {
                Some(TypedDiscoveryGenerationOverrideV1::Exact(value)) => value,
                Some(TypedDiscoveryGenerationOverrideV1::Floor(value)) => generations.max(value),
                None => generations,
            };
            (
                overrides.target_candidates.unwrap_or(requested_candidates),
                overrides
                    .portfolio_size
                    .unwrap_or(models.prop_search_portfolio_size.max(1)),
                generations,
                overrides
                    .population
                    .unwrap_or(models.prop_search_population.max(10)),
                overrides.max_rows.unwrap_or(models.prop_search_max_rows),
            )
        };
    let mut counters = vec![
        ("target_candidates".to_owned(), candidates as u64),
        ("target_portfolio".to_owned(), portfolio as u64),
        ("generations".to_owned(), generations as u64),
        ("population".to_owned(), population as u64),
        (
            "planned_ga_evaluations".to_owned(),
            population.saturating_mul(generations) as u64,
        ),
    ];
    if max_rows > 0 {
        counters.push(("max_rows".to_owned(), max_rows as u64));
    }
    counters
}

fn requested_discovery_highlights(request: &DiscoveryRequest) -> Vec<(String, String)> {
    let mut highlights = vec![
        ("symbol".to_string(), request.symbol().to_owned()),
        ("base_tf".to_string(), request.base_tf().to_owned()),
        (
            "settings_source_sha256".to_owned(),
            request.settings_source.sha256.clone(),
        ),
        (
            "dataset_identity".to_string(),
            request.dataset_identity().to_path_component(),
        ),
        (
            "dataset_generation".to_string(),
            request
                .pinned_input
                .receipt()
                .anchor()
                .generation_id()
                .to_owned(),
        ),
        (
            "dataset_manifest_binding_sha256".to_string(),
            request
                .pinned_input
                .receipt()
                .anchor()
                .manifest_binding_sha256()
                .to_owned(),
        ),
        (
            "direct_dataset_generations".to_string(),
            request
                .pinned_input
                .receipt()
                .direct_timeframes()
                .iter()
                .map(|selected| {
                    format!(
                        "{}:{}:{}",
                        selected.identity().timeframe(),
                        selected.generation_id(),
                        selected.manifest_binding_sha256()
                    )
                })
                .collect::<Vec<_>>()
                .join(","),
        ),
        (
            "higher_tfs".to_string(),
            if request.higher_tfs.is_empty() {
                "-".to_string()
            } else {
                request.higher_tfs.join(", ")
            },
        ),
    ];
    let models = &request.settings_source.settings().models;
    let max_hours = request
        .config
        .as_ref()
        .map_or(models.prop_search_max_hours, |config| config.max_hours);
    if max_hours > 0.0 {
        highlights.push(("time_budget".to_string(), format!("{max_hours:.2}h")));
    }
    if request
        .config
        .as_ref()
        .map_or(models.prop_search_use_opportunistic, |config| {
            config.filtering.use_opportunistic_candidates
        })
    {
        highlights.push((
            "quality_lane".to_string(),
            "strict+opportunistic".to_string(),
        ));
    }
    highlights.push(("artifact_class".to_owned(), "ResearchOnly".to_owned()));
    highlights.push((
        "promotion_eligibility".to_owned(),
        "NotPromotionEligible".to_owned(),
    ));
    highlights
}

fn upsert_counter(counters: &mut Vec<(String, u64)>, name: &str, value: u64) {
    if let Some((_, existing)) = counters.iter_mut().find(|(key, _)| key == name) {
        *existing = value;
    } else {
        counters.push((name.to_string(), value));
    }
}

fn push_recent_entry(entries: &[String], entry: impl Into<String>) -> Vec<String> {
    let mut next = entries.to_vec();
    next.push(entry.into());
    if next.len() > 12 {
        next.drain(0..(next.len() - 12));
    }
    next
}

fn apply_backend_discovery_event(snapshot: &mut JobSnapshot, event: &DiscoveryProgress) {
    match event {
        DiscoveryProgress::CandidateCensusUpdated { census } => {
            for (key, value) in census.counters() {
                upsert_counter(&mut snapshot.report.counters, key, value as u64);
            }
        }
        DiscoveryProgress::SearchStarted {
            population,
            generations,
            max_indicators,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.78),
                stage: "search_started".to_string(),
                message: format!(
                    "genetic search started with population={} and generations={}",
                    population, generations
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "population",
                *population as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "generations",
                *generations as u64,
            );
            // SearchStarted carries the receipt-resolved population, which may
            // differ from the queued settings. This remains an evaluation-slot
            // plan, not completed work or a count of unique candidate genomes.
            upsert_counter(
                &mut snapshot.report.counters,
                "planned_ga_evaluations",
                population.saturating_mul(*generations) as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "max_indicators",
                *max_indicators as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "search started with population={} generations={} max_indicators={}",
                    population, generations, max_indicators
                ),
            );
        }
        DiscoveryProgress::GenerationCompleted {
            generation,
            total_generations,
            best_fitness,
            stagnant_generations,
            archived_profitable,
        } => {
            let ratio = if *total_generations == 0 {
                0.0
            } else {
                *generation as f32 / *total_generations as f32
            };
            snapshot.progress = JobProgress {
                percent: Some((0.8 + 0.1 * ratio).clamp(0.8, 0.9)),
                stage: "search_generations".to_string(),
                message: format!(
                    "generation {}/{} complete (best fitness {:.2})",
                    generation, total_generations, best_fitness
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "generation",
                *generation as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "archived_profitable",
                *archived_profitable as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "stagnant_generations",
                *stagnant_generations as u64,
            );
            snapshot.report.entries = push_recent_entry(
                &snapshot.report.entries,
                format!(
                    "generation | {}/{} | best_fitness={:.2} | archived={}",
                    generation, total_generations, best_fitness, archived_profitable
                ),
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "generation {}/{} completed with best fitness {:.2}",
                    generation, total_generations, best_fitness
                ),
            );
        }
        DiscoveryProgress::CandidatesRanked {
            candidate_count,
            truncated_to,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.91),
                stage: "ranking_candidates".to_string(),
                message: format!(
                    "ranked {} candidates and kept top {}",
                    candidate_count, truncated_to
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "candidates",
                *truncated_to as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "ranked {} candidates and truncated to {}",
                    candidate_count, truncated_to
                ),
            );
        }
        DiscoveryProgress::CandidatesFiltered {
            passed_filters,
            evaluated_candidates,
            min_trades_required,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.94),
                stage: "filtering_candidates".to_string(),
                message: format!(
                    "{} of {} candidates passed filters",
                    passed_filters, evaluated_candidates
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "filtered_candidates",
                *passed_filters as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "min_trades_required",
                *min_trades_required as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "{} of {} candidates passed filters (min trades {})",
                    passed_filters, evaluated_candidates, min_trades_required
                ),
            );
        }
        DiscoveryProgress::QualityScreened {
            strict_passed,
            opportunistic_passed,
            evaluated_candidates,
            logged_trade_sets,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.955),
                stage: "quality_screen".to_string(),
                message: format!(
                    "quality screen kept {} strict and {} opportunistic candidates",
                    strict_passed, opportunistic_passed
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "quality_screened",
                (*strict_passed + *opportunistic_passed) as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "quality_evaluated",
                *evaluated_candidates as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "opportunistic_candidates",
                *opportunistic_passed as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "trade_logs",
                *logged_trade_sets as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "quality screen kept {} strict + {} opportunistic out of {} candidates",
                    strict_passed, opportunistic_passed, evaluated_candidates
                ),
            );
        }
        DiscoveryProgress::PortfolioSelected {
            portfolio_size,
            rejected_by_correlation,
            target_portfolio,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.97),
                stage: "portfolio_construction".to_string(),
                message: format!(
                    "portfolio selection accepted {} of target {}",
                    portfolio_size, target_portfolio
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "portfolio",
                *portfolio_size as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "rejected_by_correlation",
                *rejected_by_correlation as u64,
            );
            snapshot.report.entries = push_recent_entry(
                &snapshot.report.entries,
                format!(
                    "portfolio | accepted={} | rejected_by_correlation={} | target={}",
                    portfolio_size, rejected_by_correlation, target_portfolio
                ),
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "portfolio selection accepted {} and rejected {} by correlation",
                    portfolio_size, rejected_by_correlation
                ),
            );
        }
        DiscoveryProgress::StageAdvanced { stage, detail } => {
            // Boundary markers for the long, otherwise-silent post-GA blocks
            // (2026-07-20: a healthy run frozen at "quality_screen 95.5%" for
            // hours was mistaken for a hang and killed). Percent is a fixed,
            // monotonic per-stage map inside the 0.945–0.99 tail window.
            let percent = match *stage {
                "quality_screen" => 0.945,
                "selecting_portfolio" => 0.96,
                "candidate_walkforward" => 0.965,
                "robustness_filters" => 0.975,
                "validation_gates" => 0.985,
                // The holdout replay runs in the WRAPPER, after the inner
                // cycle's Completed (0.99) — same value avoids a visible
                // percent regression while the message switches to the tail.
                "holdout_forward_test" => 0.99,
                _ => 0.95,
            };
            snapshot.progress = JobProgress {
                percent: Some(percent),
                stage: (*stage).to_string(),
                message: detail.clone(),
            };
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!("stage advanced: {stage} — {detail}"),
            );
        }
        DiscoveryProgress::Completed {
            candidate_count,
            filtered_count,
            portfolio_size,
        } => {
            snapshot.progress = JobProgress {
                percent: Some(0.99),
                stage: "finalizing_discovery".to_string(),
                message: format!(
                    "discovery finalized with {} portfolio strategies",
                    portfolio_size
                ),
            };
            upsert_counter(
                &mut snapshot.report.counters,
                "candidates",
                *candidate_count as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "filtered_candidates",
                *filtered_count as u64,
            );
            upsert_counter(
                &mut snapshot.report.counters,
                "portfolio",
                *portfolio_size as u64,
            );
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "discovery finalized with {} candidates, {} filtered, {} portfolio",
                    candidate_count, filtered_count, portfolio_size
                ),
            );
        }
    }

    snapshot.report.log_path = Some(canonical_log_path().display().to_string());
}

pub fn completed_snapshot(mut snapshot: JobSnapshot, result: &DiscoveryResult) -> JobSnapshot {
    let candidates = result.candidates.len() as u64;
    let portfolio = result.portfolio.len() as u64;
    let not_selected = candidates.saturating_sub(portfolio);
    let quality_by_strategy = result
        .quality_metrics
        .iter()
        .map(|metrics| (metrics.strategy_id.as_str(), metrics))
        .collect::<std::collections::HashMap<_, _>>();
    let best_gene = result.portfolio.iter().max_by(|left, right| {
        left.fitness
            .partial_cmp(&right.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut highlights = vec![
        ("accepted".to_string(), portfolio.to_string()),
        ("not_selected".to_string(), not_selected.to_string()),
    ];
    if !result.quality_metrics.is_empty() {
        let strict_count = result
            .quality_metrics
            .iter()
            .filter(|metrics| metrics.has_edge)
            .count();
        highlights.push((
            "quality_scored".to_string(),
            result.quality_metrics.len().to_string(),
        ));
        highlights.push(("quality_edge".to_string(), strict_count.to_string()));
    }
    if !result.logged_trades.is_empty() {
        highlights.push((
            "trade_logs".to_string(),
            result.logged_trades.len().to_string(),
        ));
    }
    if let Some(best_quality) = result.quality_metrics.iter().max_by(|left, right| {
        left.quality_score
            .partial_cmp(&right.quality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        highlights.push((
            "best_quality".to_string(),
            format!("{:.1}", best_quality.quality_score),
        ));
        highlights.push((
            "best_quality_strategy".to_string(),
            best_quality.strategy_id.clone(),
        ));
    }
    if let Some(best) = best_gene {
        highlights.push(("best_strategy".to_string(), best.strategy_id.clone()));
        highlights.push((
            "best_sharpe".to_string(),
            format!("{:.2}", best.sharpe_ratio),
        ));
        highlights.push(("best_win_rate".to_string(), format!("{:.2}", best.win_rate)));
        // Surface the best gene's max-drawdown (fraction of equity)
        // so `--validation-mode` can record the selected gene's risk per TF
        // without re-reading the on-disk portfolio JSON. Additive
        // highlight — no existing reader keys off the highlights list
        // length, and the UI ignores unknown keys.
        highlights.push((
            "best_max_dd".to_string(),
            format!("{:.4}", best.max_drawdown),
        ));
    }
    // #211: surface the BEST Sharpe across the forward-test (OOS) tail
    // artifacts so `--validation-mode` can record both in-sample and
    // out-of-sample top-Sharpe per TF. `best_sharpe` above is in-sample
    // (stage-1) and is by construction what the GA optimized against —
    // it always looks inflated. The forward-test artifact is the
    // held-out 20% tail that the discovery cycle never trained on. The best
    // score across many tested genes is selection-sensitive, not an unbiased
    // performance estimate for a subsequently selected portfolio.
    //
    // Empty `forward_test_validation_artifacts` (e.g. when the tail
    // window was too short or `compute_discovery_forward_test_artifacts`
    // failed) → no highlight emitted. The validation reader treats the
    // absence as `None` and falls back to in-sample reporting.
    if let Some(best_oos) = result
        .forward_test_validation_artifacts
        .iter()
        .map(|artifact| artifact.summary().metrics.sharpe)
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
        highlights.push(("best_oos_sharpe".to_string(), format!("{:.4}", best_oos)));
    }
    let entries = result
        .portfolio
        .iter()
        .take(3)
        .map(|gene| {
            if let Some(metrics) = quality_by_strategy.get(gene.strategy_id.as_str()) {
                format!(
                    "{} | fitness={:.2} | quality={:.1} | monthly_win={:.2} | trades/mo={:.1} | edge={}",
                    gene.strategy_id,
                    gene.fitness,
                    metrics.quality_score,
                    metrics.monthly_win_rate,
                    metrics.trades_per_month,
                    metrics.has_edge
                )
            } else {
                format!(
                    "{} | fitness={:.2} | sharpe={:.2} | win_rate={:.2} | trades={}",
                    gene.strategy_id,
                    gene.fitness,
                    gene.sharpe_ratio,
                    gene.win_rate,
                    gene.trades_count
                )
            }
        })
        .collect();

    let mut counters = std::mem::take(&mut snapshot.report.counters);
    counters.retain(|(key, _)| key != "rejected");
    for (key, value) in [
        ("candidates", candidates),
        ("portfolio", portfolio),
        ("not_selected", not_selected),
        ("quality_scored", result.quality_metrics.len() as u64),
        ("trade_logs", result.logged_trades.len() as u64),
    ] {
        upsert_counter(&mut counters, key, value);
    }
    if let Some(census) = result
        .funnel_profile
        .as_ref()
        .and_then(|f| f.candidate_census.as_ref())
    {
        for (key, value) in census.counters() {
            upsert_counter(&mut counters, key, value as u64);
        }
    }
    snapshot.state = JobState::Succeeded;
    snapshot.report = JobReport {
        counters,
        highlights,
        entries,
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Info,
            format!(
                "completed discovery with {portfolio} portfolio strategies out of {candidates} candidates"
            ),
        ),
        summary: format!(
            "discovery completed with {} portfolio strategies out of {} candidates",
            portfolio, candidates
        ),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    snapshot
}

#[cfg(test)]
pub fn failed_snapshot(kind: JobKind, err: anyhow::Error) -> JobSnapshot {
    failed_snapshot_from(JobSnapshot::new(kind), err)
}

fn failed_snapshot_from(mut snapshot: JobSnapshot, err: anyhow::Error) -> JobSnapshot {
    let message = format!("{err:#}");
    snapshot.state = JobState::Failed;
    snapshot.report = JobReport {
        errors: vec![message.clone()],
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Error,
            format!("discovery failed: {message}"),
        ),
        summary: message,
        log_path: Some(canonical_log_path().display().to_string()),
        ..snapshot.report
    };
    snapshot
}

#[cfg(test)]
pub fn cancelled_snapshot(kind: JobKind, message: impl Into<String>) -> JobSnapshot {
    cancelled_snapshot_from(JobSnapshot::new(kind), message)
}

fn cancelled_snapshot_from(mut snapshot: JobSnapshot, message: impl Into<String>) -> JobSnapshot {
    let message = message.into();
    snapshot.state = JobState::Cancelled;
    snapshot.report = JobReport {
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Warning,
            format!("discovery cancelled: {message}"),
        ),
        summary: message,
        log_path: Some(canonical_log_path().display().to_string()),
        ..snapshot.report
    };
    snapshot
}

/// Wait asynchronously for this process's configured CPU width. A temporarily
/// busy broker is not a reason to shrink the search or start unaccounted work.
async fn admit_discovery_cpu_stage(
    execution: &AppExecutionState,
    cancel: &CancellationFlag,
    tx: &mpsc::Sender<ServiceEvent>,
    snapshot: &mut JobSnapshot,
    context: &str,
) -> Result<AdmittedCpuLease> {
    anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
    let width = execution.admission_snapshot().cpu.installed_limit;
    snapshot.progress = JobProgress {
        percent: None,
        stage: "waiting_for_cpu".to_owned(),
        message: format!(
            "{context}: waiting for {} CPU workers from the shared application budget",
            width.get()
        ),
    };
    upsert_counter(
        &mut snapshot.report.counters,
        "cpu_workers_requested",
        width.get() as u64,
    );
    upsert_counter(&mut snapshot.report.counters, "cpu_workers_reserved", 0);
    try_send_progress_event(tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

    let pending = execution
        .admission_client()
        .submit(CpuPermitRequest::local(width))?;
    // Keep one pinned future across timer ticks. Dropping it on Stop removes
    // the queued request; do not recreate or leak a request on every poll.
    let admission = pending.wait();
    tokio::pin!(admission);
    let mut cancellation_poll = tokio::time::interval(std::time::Duration::from_millis(25));
    loop {
        anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
        tokio::select! {
            admitted = &mut admission => {
                let admitted = admitted?;
                // If Stop raced with the grant, dropping it returns all capacity.
                anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
                upsert_counter(&mut snapshot.report.counters, "cpu_workers_reserved", admitted.width().get() as u64);
                return Ok(admitted);
            }
            _ = cancellation_poll.tick() => {}
        }
    }
}

/// The blocking task owns both the lease and its coordinator until work has
/// returned (including panic unwinding). Nested Rayon work uses this same pool.
fn spawn_discovery_cpu_stage<R, Work>(
    execution: Arc<AppExecutionState>,
    lease: AdmittedCpuLease,
    cancel: CancellationFlag,
    work: Work,
) -> tokio::task::JoinHandle<Result<R>>
where
    R: Send + 'static,
    Work: FnOnce(&BudgetedCpuScope<'_>) -> Result<R> + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        lease.execute_with_scope(execution.executor(), |scope| {
            scope.require_current_pool()?;
            // A queued spawn_blocking task may only start after Stop was pressed.
            anyhow::ensure!(!cancel.is_requested(), "__DISCOVERY_CANCELLED__");
            work(scope)
        })?
    })
}

pub fn start_discovery_job(
    mut request: DiscoveryRequest,
    execution: Arc<AppExecutionState>,
    tx: mpsc::Sender<ServiceEvent>,
) -> Result<DiscoveryJobHandle> {
    request.validate()?;
    request.higher_tfs = request.canonical_higher_timeframes()?;
    let requested_counters = requested_discovery_counters(&request);
    let requested_count = |key: &str| {
        requested_counters
            .iter()
            .find(|(name, _)| name == key)
            .map_or(0, |(_, value)| *value)
    };

    let handle = DiscoveryJobHandle::new();
    let cancel = handle.cancel.clone();
    let mut snapshot = handle.snapshot.clone();
    snapshot.state = JobState::Running;
    snapshot.progress = JobProgress {
        percent: Some(0.05),
        stage: "using_pinned_data".to_string(),
        message: format!(
            "using pinned exact dataset generation {} @ {}",
            request.pinned_input.receipt().anchor().generation_id(),
            request.dataset_identity().to_path_component()
        ),
    };
    snapshot.report = JobReport {
        counters: requested_discovery_counters(&request),
        highlights: requested_discovery_highlights(&request),
        events: push_recent_event(
            &snapshot.report.events,
            JobEventLevel::Info,
            format!(
                "planned discovery for {} {} with population={}, generations={}, candidate_count={} (0=all GA-returned candidates), portfolio_size={}",
                request.symbol(),
                request.base_tf(),
                requested_count("population"),
                requested_count("generations"),
                requested_count("target_candidates"),
                requested_count("target_portfolio")
            ),
        ),
        summary: format!(
            "using pinned discovery dataset for {} on {}",
            request.symbol(),
            request.base_tf()
        ),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };
    try_send_progress_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));
    log_discovery_event(
        "ui_discovery_job",
        "STARTED",
        format!(
            "starting discovery for {} ({})",
            request.symbol(),
            request.dataset_identity().to_path_component()
        ),
    );

    tokio::spawn(async move {
        if cancel.is_requested() {
            let cancelled = cancelled_snapshot_from(
                snapshot,
                "operator cancelled discovery before feature preparation",
            );
            send_terminal_snapshot(&tx, &cancelled).await;
            log_discovery_event(
                "ui_discovery_job",
                "CANCELLED",
                cancelled.report.summary.clone(),
            );
            return;
        }

        let required_direct = match required_direct_timeframes(&request) {
            Ok(required) => required,
            Err(err) => {
                let failed = failed_snapshot_from(snapshot, err);
                send_terminal_snapshot(&tx, &failed).await;
                log_discovery_event("ui_discovery_job", "FAILED", failed.report.summary.clone());
                return;
            }
        };
        let selected_timeframe_count = request.pinned_input.receipt().direct_timeframes().len();
        let seed = rand::random::<u64>();
        let mut working_set: Option<CpuDiscoveryWorkingSet> = None;
        let mut working_progress = DiscoveryWorkingSetProgress::default();
        let mut deadline = DiscoveryDeadline::default();
        let max_hours = request.config.as_ref().map_or(
            request
                .settings_source
                .settings()
                .models
                .prop_search_max_hours
                .max(0.0),
            |config| config.max_hours,
        );

        loop {
            working_progress.active = working_progress.completed + 1;
            snapshot.progress = JobProgress {
                percent: Some(0.35),
                stage: "preparing_inputs".to_string(),
                message: format!(
                    "preparing exact cost sources and multi-timeframe features for {}",
                    request.symbol()
                ),
            };
            snapshot.report = JobReport {
                counters: requested_discovery_counters(&request)
                    .into_iter()
                    .chain(std::iter::once((
                        "selected_timeframes".to_string(),
                        selected_timeframe_count as u64,
                    )))
                    .collect(),
                highlights: requested_discovery_highlights(&request),
                events: push_recent_event(
                    &snapshot.report.events,
                    JobEventLevel::Info,
                    format!(
                        "selected {} exact timeframe generation(s) for {}",
                        selected_timeframe_count,
                        request.symbol()
                    ),
                ),
                summary: format!(
                    "selected {} exact timeframe generations for {}",
                    selected_timeframe_count,
                    request.symbol()
                ),
                log_path: Some(canonical_log_path().display().to_string()),
                ..JobReport::default()
            };
            working_progress.apply(&mut snapshot.report, seed);
            try_send_progress_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

            if cancel.is_requested() {
                let cancelled = deadline.cancelled(
                    snapshot,
                    "operator cancelled discovery after pinned-data validation",
                );
                send_terminal_snapshot(&tx, &cancelled).await;
                log_discovery_event(
                    "ui_discovery_job",
                    "CANCELLED",
                    cancelled.report.summary.clone(),
                );
                return;
            }

            let feature_cpu = match admit_discovery_cpu_stage(
                &execution,
                &cancel,
                &tx,
                &mut snapshot,
                "Discovery input preparation",
            )
            .await
            {
                Ok(lease) => lease,
                Err(error) => {
                    let (terminal, status) = if cancel.is_requested() {
                        (
                        deadline.cancelled(
                            snapshot,
                            "operator cancelled discovery while waiting for input-preparation CPU",
                        ),
                        "CANCELLED",
                    )
                    } else {
                        (failed_snapshot_from(snapshot, error), "FAILED")
                    };
                    send_terminal_snapshot(&tx, &terminal).await;
                    log_discovery_event("ui_discovery_job", status, terminal.report.summary);
                    return;
                }
            };
            // Start once on actual CPU admission, not while the initial job is queued.
            // Subsequent feature/search/admission/publication time shares this deadline.
            if let Err(error) = deadline.start_once(max_hours, cancel.clone()) {
                let failed = failed_snapshot_from(snapshot, error);
                send_terminal_snapshot(&tx, &failed).await;
                return;
            }

            // Per-run latest progress is bounded; CPU workers never wait for the
            // UI queue. Timers can expose elapsed time, never fabricated completion.
            snapshot.progress = JobProgress {
                percent: None,
                stage: "preparing_inputs".to_string(),
                message: format!(
                    "preparing exact research inputs for {} ({})",
                    request.symbol(),
                    request.base_tf()
                ),
            };
            try_send_progress_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

            let feature_request = request.clone();
            let feature_required_direct = required_direct.clone();
            #[cfg(not(feature = "gpu-nvidia"))]
            let previous_working_set = working_set.take();
            let feature_input = Arc::clone(&request.pinned_input);
            let feature_cancel = cancel.clone();
            let feature_progress = Arc::new(Mutex::new(None));
            let feature_progress_writer = Arc::clone(&feature_progress);
            let feature_control =
                FeatureBuildControl::new(cancel.cancel_arc()).with_observer(move |progress| {
                    if let Ok(mut latest) = feature_progress_writer.lock() {
                        *latest = Some((progress, std::time::Instant::now()));
                    }
                });
            let feature_handle = spawn_discovery_cpu_stage(
                Arc::clone(&execution),
                feature_cpu,
                cancel.clone(),
                move |_scope| {
                    let (prepared_input, screening_costs, next_working_set) = (|| {
                        #[cfg(feature = "gpu-nvidia")]
                        {
                            let mut screening_costs = None;
                            let prepared =
                                neoethos_search::prepare_canonical_discovery_run_input_v3(
                                    |no_physical_gpu_admission| {
                                        // A physical-GPU refusal must happen before any host
                                        // feature allocation or broker metadata request.
                                        screening_costs = Some(prepare_discovery_screening_costs(
                                            &feature_request,
                                            &feature_cancel,
                                        )?);
                                        let pinned_series =
                                            feature_input.take_pinned_series_v1()?;
                                        let dataset = pinned_series
                                            .into_cpu_dataset_after_no_physical_gpu_v1(
                                                &no_physical_gpu_admission,
                                            )?;
                                        let input = prepare_cpu_discovery_features_with_control(
                                            &feature_request,
                                            dataset,
                                            &feature_required_direct,
                                            &feature_control,
                                        )?;
                                        Ok((input, no_physical_gpu_admission))
                                    },
                                    || {
                                        anyhow::bail!(
                                            "full native Discovery workspace-plan sealing is not integrated; refusing host feature materialization on a physical GPU"
                                        )
                                    },
                                    |_admitted_native_run| {
                                        anyhow::bail!(
                                            "resident native Data materialization is unreachable until the complete workspace plan is sealed"
                                        )
                                    },
                                )?;
                            Ok::<_, anyhow::Error>((
                            prepared,
                            Arc::new(screening_costs.context(
                                "prepared CPU Discovery omitted its source-bound screening costs",
                            )?),
                            None::<CpuDiscoveryWorkingSet>,
                        ))
                        }
                        #[cfg(not(feature = "gpu-nvidia"))]
                        {
                            let mut run = match previous_working_set {
                                Some(run) => run,
                                None => {
                                    let costs = Arc::new(prepare_discovery_screening_costs(
                                        &feature_request,
                                        &feature_cancel,
                                    )?);
                                    let pinned_series = feature_input.take_pinned_series_v1()?;
                                    let dataset = pinned_series
                                        .into_cpu_dataset_without_native_adapter_v1()?;
                                    let rows = dataset
                                        .frames
                                        .values()
                                        .map(|frame| frame.len())
                                        .max()
                                        .unwrap_or(0);
                                    let search =
                                        neoethos_search::discovery::StreamingSearch::new_seeded(
                                            rows, seed,
                                        );
                                    CpuDiscoveryWorkingSet {
                                        dataset,
                                        costs,
                                        search,
                                        selected_cursor: 0,
                                    }
                                }
                            };
                            let batch = run.search.next_batch().context(
                            "CPU memory admission affords no Discovery working set; no fixed-prefix fallback was run",
                        )?;
                            run.selected_cursor = batch.cursor;
                            let input = prepare_cpu_discovery_batch_with_control(
                                &feature_request,
                                &run.dataset,
                                &feature_required_direct,
                                batch,
                                &feature_control,
                            )?;
                            let costs = Arc::clone(&run.costs);
                            Ok::<_, anyhow::Error>((input, costs, Some(run)))
                        }
                    })(
                    )?;
                    anyhow::ensure!(!feature_cancel.is_requested(), "__DISCOVERY_CANCELLED__");
                    feature_control.report(
                        "feature_receipt",
                        "binding the prepared receipt to research configuration",
                        0,
                        1,
                    )?;
                    // The CPU producer already hashed the full feature frame.
                    // Retrieve that original receipt; the Search consumer still
                    // checks current values against it before financial evaluation.
                    #[cfg(feature = "gpu-nvidia")]
                    let receipt = prepared_input.cpu_receipt_v2().cloned().context(
                        "canonical research requires the actual prepared CPU feature receipt",
                    )?;
                    #[cfg(not(feature = "gpu-nvidia"))]
                    let receipt = prepared_input.receipt()?;
                    feature_control.checkpoint()?;
                    // Neither an HTTP selector nor a dummy frame can mint this:
                    // seal only after the actual producer has returned its values.
                    let research_contract = seal_discovery_research_contract(
                        &feature_request,
                        receipt,
                        &screening_costs,
                    )?;
                    let config = feature_request.resolve_research_config(&research_contract)?;
                    feature_control.report(
                        "feature_receipt",
                        "sealed exact research inputs",
                        1,
                        1,
                    )?;
                    Ok((
                        prepared_input,
                        screening_costs,
                        research_contract,
                        config,
                        next_working_set,
                    ))
                },
            );

            // A running spawn_blocking closure cannot be aborted. Keep its owner
            // alive and report a pending Stop honestly until it returns; do not
            // mislabel timer activity as feature-engine progress.
            let hb_tx = tx.clone();
            let hb_snapshot = snapshot.clone();
            let hb_feature_progress = Arc::clone(&feature_progress);
            let hb_cancel = cancel.clone();
            let hb_started = std::time::Instant::now();
            let heartbeat = tokio::spawn(async move {
                let mut ticker = tokio::time::interval(std::time::Duration::from_secs(2));
                ticker.tick().await; // consume the immediate first tick
                loop {
                    ticker.tick().await;
                    let mut hb_snapshot = hb_snapshot.clone();
                    let secs = hb_started.elapsed().as_secs();
                    if let Ok(latest) = hb_feature_progress.lock() {
                        if let Some((progress, observed_at)) = latest.as_ref() {
                            apply_feature_preparation_progress(&mut hb_snapshot, progress);
                            hb_snapshot.progress.message.push_str(&format!(
                                " · last work event {}s ago",
                                observed_at.elapsed().as_secs()
                            ));
                        }
                    }
                    if hb_cancel.is_requested() {
                        hb_snapshot.progress.message =
                            waiting_for_worker_message(&hb_snapshot.progress.message, secs, true);
                    }
                    try_send_progress_event(
                        &hb_tx,
                        ServiceEvent::DiscoveryUpdated(hb_snapshot.clone()),
                    );
                }
            });

            let feature_build = feature_handle.await;
            heartbeat.abort();
            // No stale Running event may race with the final terminal snapshot.
            let _ = heartbeat.await;
            if let Ok(latest) = feature_progress.lock() {
                if let Some((progress, _)) = latest.as_ref() {
                    apply_feature_preparation_progress(&mut snapshot, progress);
                }
            }

            let (prepared_input, screening_costs, research_contract, config, next_working_set) =
                match feature_build {
                    Ok(Ok(prepared_input)) => prepared_input,
                    Ok(Err(err))
                        if cancel.is_requested()
                            && (FeatureBuildCancelled::matches(&err)
                                || err.chain().any(|cause| {
                                    cause.to_string().contains("__DISCOVERY_CANCELLED__")
                                })) =>
                    {
                        let cancelled = deadline
                            .cancelled(snapshot, "operator cancelled Discovery input preparation");
                        send_terminal_snapshot(&tx, &cancelled).await;
                        log_discovery_event(
                            "ui_discovery_job",
                            "CANCELLED",
                            cancelled.report.summary.clone(),
                        );
                        return;
                    }
                    Ok(Err(err)) => {
                        let failed = failed_snapshot_from(snapshot, err);
                        send_terminal_snapshot(&tx, &failed).await;
                        log_discovery_event(
                            "ui_discovery_job",
                            "FAILED",
                            failed.report.summary.clone(),
                        );
                        return;
                    }
                    Err(err) => {
                        let failed = failed_snapshot_from(
                            snapshot,
                            anyhow::anyhow!("feature preparation join error: {err}"),
                        );
                        send_terminal_snapshot(&tx, &failed).await;
                        log_discovery_event(
                            "ui_discovery_job",
                            "FAILED",
                            failed.report.summary.clone(),
                        );
                        return;
                    }
                };
            working_set = next_working_set;
            if let Some(run) = &working_set {
                working_progress.total_entries = run.search.space_len() as u64;
            }
            if cancel.is_requested() {
                let cancelled = deadline.cancelled(
                    snapshot,
                    "operator cancelled Discovery after input preparation",
                );
                send_terminal_snapshot(&tx, &cancelled).await;
                log_discovery_event(
                    "ui_discovery_job",
                    "CANCELLED",
                    cancelled.report.summary.clone(),
                );
                return;
            }
            request.config = Some(config);
            #[cfg(feature = "gpu-nvidia")]
            let prepared_shape = prepared_input.shape();
            #[cfg(not(feature = "gpu-nvidia"))]
            let prepared_shape = Ok((
                prepared_input.features().n_samples(),
                prepared_input.features().n_features(),
            ));
            let (feature_rows, feature_columns) = match prepared_shape {
                Ok(shape) => shape,
                Err(err) => {
                    let failed = failed_snapshot_from(snapshot, err);
                    send_terminal_snapshot(&tx, &failed).await;
                    log_discovery_event(
                        "ui_discovery_job",
                        "FAILED",
                        failed.report.summary.clone(),
                    );
                    return;
                }
            };

            snapshot.report = JobReport {
                counters: requested_discovery_counters(&request)
                    .into_iter()
                    .chain([
                        ("feature_rows".to_string(), feature_rows as u64),
                        ("feature_columns".to_string(), feature_columns as u64),
                    ])
                    .collect(),
                highlights: requested_discovery_highlights(&request),
                events: push_recent_event(
                    &snapshot.report.events,
                    JobEventLevel::Info,
                    format!(
                        "prepared feature frame {}x{} for {}",
                        feature_rows,
                        feature_columns,
                        request.symbol()
                    ),
                ),
                summary: format!(
                    "prepared {} rows x {} columns for discovery",
                    feature_rows, feature_columns
                ),
                log_path: Some(canonical_log_path().display().to_string()),
                ..JobReport::default()
            };
            working_progress.apply(&mut snapshot.report, seed);
            if cancel.is_requested() {
                let cancelled = deadline.cancelled(
                    snapshot,
                    "operator cancelled discovery before portfolio construction",
                );
                send_terminal_snapshot(&tx, &cancelled).await;
                log_discovery_event(
                    "ui_discovery_job",
                    "CANCELLED",
                    cancelled.report.summary.clone(),
                );
                return;
            }

            let search_cpu = match admit_discovery_cpu_stage(
                &execution,
                &cancel,
                &tx,
                &mut snapshot,
                "Discovery strategy search",
            )
            .await
            {
                Ok(lease) => lease,
                Err(error) => {
                    let (terminal, status) = if cancel.is_requested() {
                        (
                            deadline.cancelled(
                                snapshot,
                                "operator cancelled discovery while waiting for search CPU",
                            ),
                            "CANCELLED",
                        )
                    } else {
                        (failed_snapshot_from(snapshot, error), "FAILED")
                    };
                    send_terminal_snapshot(&tx, &terminal).await;
                    log_discovery_event("ui_discovery_job", status, terminal.report.summary);
                    return;
                }
            };
            snapshot.progress = JobProgress {
                percent: None,
                stage: "running_discovery".to_string(),
                message: format!("evaluating strategy candidates for {}", request.symbol()),
            };
            try_send_progress_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));

            let live_snapshot = Arc::new(Mutex::new(snapshot.clone()));
            let search_request = request.clone();
            #[cfg(not(feature = "gpu-nvidia"))]
            let batch_cursor = working_set
                .as_ref()
                .expect("prepared CPU working set")
                .selected_cursor;
            let tx_progress = tx.clone();
            let live_snapshot_for_progress = Arc::clone(&live_snapshot);
            // Report time since the last actual engine event without claiming
            // that a silent stage is healthy. This note is transient, so genuine
            // progress events retain their own message and measured counters.
            let last_event_at = Arc::new(Mutex::new(std::time::Instant::now()));
            let last_event_for_progress = Arc::clone(&last_event_at);
            #[cfg(not(feature = "gpu-nvidia"))]
            let input_control = {
                let receipt_snapshot = Arc::clone(&live_snapshot);
                let receipt_last_event = Arc::clone(&last_event_at);
                let receipt_tx = tx.clone();
                FeatureBuildControl::new(cancel.cancel_arc()).with_observer(move |event| {
                    if let Ok(mut last) = receipt_last_event.lock() {
                        *last = std::time::Instant::now();
                    }
                    if let Ok(mut snapshot) = receipt_snapshot.lock() {
                        apply_feature_preparation_progress(&mut snapshot, &event);
                        try_send_progress_event(
                            &receipt_tx,
                            ServiceEvent::DiscoveryUpdated(snapshot.clone()),
                        );
                    }
                })
            };
            let hb_live_snapshot = Arc::clone(&live_snapshot);
            let hb_tx = tx.clone();
            let hb_cancel = cancel.clone();
            let stale_heartbeat = tokio::spawn(async move {
                let mut ticker = tokio::time::interval(std::time::Duration::from_secs(20));
                ticker.tick().await; // consume the immediate first tick
                loop {
                    ticker.tick().await;
                    let silent_secs = last_event_at
                        .lock()
                        .map(|t| t.elapsed().as_secs())
                        .unwrap_or(0);
                    if silent_secs < 45 {
                        continue;
                    }
                    let Ok(current) = hb_live_snapshot.lock().map(|s| s.clone()) else {
                        continue;
                    };
                    let mut beat = current;
                    beat.progress.message = waiting_for_worker_message(
                        &beat.progress.message,
                        silent_secs,
                        hb_cancel.is_requested(),
                    );
                    try_send_progress_event(&hb_tx, ServiceEvent::DiscoveryUpdated(beat));
                }
            });
            // Install the cancel flag the GA polls EACH GENERATION (discovery is
            // single-instance, so a process-global flag is safe). This makes Stop
            // interrupt the search mid-run instead of only at coarse phase boundaries.
            // Cleared right after the blocking search returns.
            neoethos_search::set_search_cancel(Some(cancel.cancel_arc()));
            let cancel_arc_for_closure = cancel.cancel_arc();
            let search_result = spawn_discovery_cpu_stage(
            Arc::clone(&execution), search_cpu, cancel.clone(), move |_scope| {
            let resolved_config = search_request.execution_config()?;
            let progress = move |event| {
                if let Ok(mut last) = last_event_for_progress.lock() {
                    *last = std::time::Instant::now();
                }
                if let Ok(mut snapshot) = live_snapshot_for_progress.lock() {
                    apply_backend_discovery_event(&mut snapshot, &event);
                    try_send_progress_event(
                        &tx_progress,
                        ServiceEvent::DiscoveryUpdated(snapshot.clone()),
                    );
                }
            };
            // Use the existing explicitly research-only Search route. Current
            // broker metadata and final D1 conversion prices remain screening
            // assumptions, never quote-validated historical execution truth.
            #[cfg(feature = "gpu-nvidia")]
            let research_result =
                neoethos_search::run_prepared_canonical_trendbar_research_with_holdout_and_progress_v3(
                    prepared_input,
                    &resolved_config,
                    &research_contract,
                    search_request.prop_firm_rules,
                    progress,
                )?;
            #[cfg(not(feature = "gpu-nvidia"))]
            let research_result = {
                let run_input = prepared_input.as_run_input_with_control(&input_control)
                    .map_err(anyhow::Error::new)?;
                let (result, _) = neoethos_search::discovery::with_streaming_batch_context(batch_cursor, ||
                neoethos_search::run_canonical_trendbar_research_discovery_with_holdout_and_progress(
                    &run_input,
                    &resolved_config,
                    &research_contract,
                    search_request.prop_firm_rules,
                    progress,
                ));
                result?
            };

            // A stopped run must not publish partial results.
            if cancel_arc_for_closure.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("__DISCOVERY_CANCELLED__");
            }
            let research_root = search_request.settings_source.discovery_cache_root().join("research");
            // Persist the entire research envelope first, including the funnel
            // for an empty portfolio. A non-empty result with complete
            // canonical bar-OOS evidence then emits a schema-v5 portfolio for
            // bar-based execution. Broker/account/risk admission remains
            // separate; demo fills are optional, not a profit certificate.
            let output_path = save_discovery_research_result(
                &research_root, &research_result, &screening_costs,
            )?;
            let result = research_result.discovery_result();
            // An empty batch is a saved negative result, not termination of the
            // remaining vocabulary. A non-empty but unpublishable portfolio
            // remains a diagnostic and cannot mint a training handoff.
            let publication = if result.portfolio.is_empty() { None } else { Some((|| {
            ensure_non_empty_portfolio(
                result,
                &format!("{} {} (research artifact: {})",
                    search_request.symbol(), search_request.base_tf(), output_path.display()),
            )?;
            let live_portfolio_path = output_path.with_extension("live_portfolio.json");
            neoethos_search::save_live_portfolio_json(&live_portfolio_path, result)
                .with_context(|| {
                    format!(
                        "publish OOS-validated execution portfolio {}",
                        live_portfolio_path.display()
                    )
                })?;
            let portfolio = neoethos_search::load_live_portfolio_json(&live_portfolio_path)?;
            let training_settings = crate::app_services::training::handoff::settings_for_series(
                search_request.settings_source.settings(), search_request.pinned_input.receipt(),
            );
            let handoff = neoethos_models::PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(
                search_request.pinned_input.receipt().clone(), research_contract,
                &portfolio, &training_settings,
            )?;
            let training_handoff = crate::app_services::training::handoff::publish(
                &search_request.data_root, &handoff,
            )?;
            Ok::<_, anyhow::Error>((live_portfolio_path, training_handoff))
            })()) };
            Ok::<_, anyhow::Error>((result.clone(), output_path, publication))
        })
        .await;
            stale_heartbeat.abort();
            // Join the cancelled sender before publishing terminal state, just as
            // input preparation does, so a late Running snapshot cannot overwrite it.
            let _ = stale_heartbeat.await;
            // Clear the GA cancel flag now the blocking search has returned.
            neoethos_search::set_search_cancel(None);

            let (result, research_artifact_path, publication) = match search_result {
                Ok(Ok(result)) => result,
                // Operator Stop mid-search: a clean CANCELLED, not a failure.
                Ok(Err(err))
                    if cancel.is_requested()
                        && (FeatureBuildCancelled::matches(&err)
                            || err.chain().any(|cause| {
                                cause.to_string().contains("__DISCOVERY_CANCELLED__")
                            })) =>
                {
                    let base_snapshot = live_snapshot
                        .lock()
                        .map(|snapshot| snapshot.clone())
                        .unwrap_or(snapshot);
                    let cancelled = deadline.cancelled(
                        base_snapshot,
                        "operator cancelled discovery during the search",
                    );
                    send_terminal_snapshot(&tx, &cancelled).await;
                    log_discovery_event(
                        "ui_discovery_job",
                        "CANCELLED",
                        cancelled.report.summary.clone(),
                    );
                    return;
                }
                Ok(Err(err)) => {
                    let base_snapshot = live_snapshot
                        .lock()
                        .map(|snapshot| snapshot.clone())
                        .unwrap_or(snapshot);
                    let failed = failed_snapshot_from(base_snapshot, err);
                    send_terminal_snapshot(&tx, &failed).await;
                    log_discovery_event(
                        "ui_discovery_job",
                        "FAILED",
                        failed.report.summary.clone(),
                    );
                    return;
                }
                Err(err) => {
                    let base_snapshot = live_snapshot
                        .lock()
                        .map(|snapshot| snapshot.clone())
                        .unwrap_or(snapshot);
                    let failed = failed_snapshot_from(
                        base_snapshot,
                        anyhow::anyhow!("discovery join error: {err}"),
                    );
                    send_terminal_snapshot(&tx, &failed).await;
                    log_discovery_event(
                        "ui_discovery_job",
                        "FAILED",
                        failed.report.summary.clone(),
                    );
                    return;
                }
            };

            let base_snapshot = live_snapshot
                .lock()
                .map(|snapshot| snapshot.clone())
                .unwrap_or(snapshot);
            snapshot = completed_snapshot(base_snapshot, &result);
            working_progress.completed += 1;
            working_progress.completed_entries = working_set
                .as_ref()
                .map_or(0, |run| run.search.cursor() as u64);
            snapshot.report.highlights.extend([
                ("artifact_class".to_owned(), "ResearchOnly".to_owned()),
                (
                    "promotion_eligibility".to_owned(),
                    "NotPromotionEligible".to_owned(),
                ),
                (
                    "research_artifact".to_owned(),
                    research_artifact_path.display().to_string(),
                ),
            ]);
            match publication {
                Some(Ok((path, identity))) => {
                    working_progress.handoffs += 1;
                    working_progress.single_handoff = if working_progress.handoffs == 1 {
                        Some(identity.clone())
                    } else {
                        None
                    };
                    snapshot.report.highlights.extend([
                        ("execution_portfolio".into(), path.display().to_string()),
                        ("last_batch_training_handoff".into(), identity),
                    ]);
                }
                Some(Err(error)) => {
                    working_progress.publication_failures += 1;
                    let message = format!(
                        "batch {} research saved at {}; portfolio/handoff refused: {error:#}",
                        working_progress.active,
                        research_artifact_path.display()
                    );
                    snapshot.report.warnings.push(message.clone());
                    log_discovery_event("ui_discovery_job", "PUBLICATION_REFUSED", message);
                }
                None => {}
            }
            working_progress.apply(&mut snapshot.report, seed);
            log_discovery_event(
                "ui_discovery_batch",
                "SAVED",
                format!(
                    "seed={seed} batch={} completed_entries={} total_entries={} research={} handoffs={} publication_failures={} last_handoff={}",
                    working_progress.active,
                    working_progress.completed_entries,
                    working_progress.total_entries,
                    research_artifact_path.display(),
                    working_progress.handoffs,
                    working_progress.publication_failures,
                    snapshot
                        .report
                        .highlights
                        .iter()
                        .find(|(key, _)| key == "last_batch_training_handoff")
                        .map_or("none", |(_, value)| value.as_str()),
                ),
            );
            if cancel.is_requested() {
                let cancelled = deadline.cancelled(
                    snapshot,
                    "operator cancelled Discovery; already saved batches retained",
                );
                send_terminal_snapshot(&tx, &cancelled).await;
                return;
            }
            let exhausted = working_set
                .as_ref()
                .is_none_or(|run| run.search.cursor() >= run.search.space_len());
            if exhausted {
                if working_progress.publication_failures > 0 {
                    snapshot.state = JobState::Degraded;
                }
                if let Some(identity) = &working_progress.single_handoff {
                    snapshot
                        .report
                        .highlights
                        .push(("training_handoff".into(), identity.clone()));
                }
                snapshot.report.summary =
                    working_progress.summary("Discovery selection sweep complete");
                snapshot.progress.message = snapshot.report.summary.clone();
                send_terminal_snapshot(&tx, &snapshot).await;
                log_discovery_event(
                    "ui_discovery_job",
                    "COMPLETE",
                    snapshot.report.summary.clone(),
                );
                return;
            }
            snapshot.state = JobState::Running;
            snapshot.progress = JobProgress {
                percent: None,
                stage: "between_working_sets".into(),
                message: working_progress.summary("Continuing the next disjoint working set"),
            };
            snapshot.report.summary = snapshot.progress.message.clone();
            try_send_progress_event(&tx, ServiceEvent::DiscoveryUpdated(snapshot.clone()));
        }
    });

    Ok(handle)
}

/// On-disk contract between Discovery output and Training input.
/// Retained for strict readers of previously produced target artifacts.
/// The research-only `start_discovery_job` does not write this hand-off.
/// Filename: `<data_root>/discovery_targets/<symbol>_<base_tf>_model_targets.json`.
///
/// Version 3 is the fail-closed promotion evidence hand-off. It carries the
/// exact search receipt/config identity plus a typed copy of the canonical
/// promotion-summary envelope written from the same [`DiscoveryResult`]. The
/// currently embedded summary is diagnostic v3 evidence and cannot mint a
/// live-copy permit; search-core must first produce exact composite v3 scope.
/// Symbol/timeframe labels alone never authorize a copy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelTargetsFile {
    /// Bump this whenever the schema changes incompatibly. Readers
    /// that see a version they don't recognise should refuse the
    /// file (NOT silently fall back).
    pub schema_version: u32,
    pub symbol: String,
    pub base_tf: String,
    pub higher_tfs: Vec<String>,
    /// ISO-8601 UTC at the moment the file was written.
    pub discovered_at_utc: String,
    pub search_input_receipt: CanonicalSearchInputReceiptV2,
    pub search_input_receipt_sha256: String,
    pub search_config_hash: String,
    pub promotion_summary_authority: StoredPromotionSummaryAuthorityV3,
    pub portfolio: Vec<ModelTargetEntry>,
}

/// Exact typed authority copied from the canonical promotion-summary sidecar.
/// The loader reloads the canonical file and requires whole-envelope equality;
/// this embedded value is not a reconstruction from request labels.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredPromotionSummaryAuthorityV3 {
    pub canonical_file_name: String,
    pub envelope: CanonicalSearchArtifactEnvelopeV2<PromotionSummaryAuthorityPayloadV3>,
}

/// One accepted strategy from the portfolio.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelTargetEntry {
    pub strategy_id: String,
    pub fitness: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub trades_count: u64,
    /// **F-330**: peak-to-trough drawdown as a PERCENTAGE. The GA's
    /// `Gene::max_drawdown` is a fraction (0.25 = 25%); we ×100 at
    /// write time so the promotion gate + UI speak percentages
    /// consistently. Version 3 rejects a missing value; legacy targets are
    /// intentionally not promotion authorities.
    pub max_drawdown_pct: f64,
    /// **F-330**: gross profit / gross loss. Required by strict v3.
    pub profit_factor: f64,
}

/// Current `ModelTargetsFile::schema_version`. Bump when the schema
/// changes; the reader on the Training side asserts on this.
pub const MODEL_TARGETS_SCHEMA_VERSION: u32 = 3;

/// `discovery_targets/<symbol>_<base_tf>_model_targets.json` path
/// resolver for strict readers of existing target artifacts. Research-only
/// Discovery does not publish a model-target hand-off at this path.
pub fn model_targets_path_for(
    data_root: &std::path::Path,
    symbol: &str,
    base_tf: &str,
) -> std::path::PathBuf {
    data_root
        .join("discovery_targets")
        .join(format!("{symbol}_{base_tf}_model_targets.json"))
}

/// Canonical promotion authority stored beside the v3 target hand-off. Keeping
/// both under `data_root` avoids ambient-CWD lookup and makes the bound file
/// name deterministic for the strict reader.
pub fn promotion_summary_path_for(
    data_root: &std::path::Path,
    symbol: &str,
    base_tf: &str,
) -> std::path::PathBuf {
    data_root
        .join("discovery_targets")
        .join(format!("{symbol}_{base_tf}_promotion_summary.json"))
}

/// Preserve the research classification and its exact cost source together.
/// These files are diagnostics, not the legacy portfolio/model-target inputs.
fn save_discovery_research_result(
    output_root: &Path,
    result: &neoethos_search::CanonicalTrendbarResearchDiscoveryResultV3,
    costs: &PreparedDiscoveryScreeningCosts,
) -> Result<PathBuf> {
    result.validate()?;
    let source_bytes = serde_json::to_vec(&costs.envelope)?;
    anyhow::ensure!(
        format!("{:x}", Sha256::digest(&source_bytes))
            == result.execution_contract().assumption_source_sha256(),
        "research output cost source differs from the sealed execution contract"
    );
    let path = output_root.join(format!(
        "{}.research.json",
        result.evidence_identity_sha256()
    ));
    // Publish the source first; a visible result must have its matching source.
    neoethos_core::storage::json::write_bytes_atomic(
        path.with_extension("costs.json"),
        &source_bytes,
    )?;
    neoethos_core::storage::json::write_json_atomic_compact(&path, result)?;
    Ok(path)
}

/// Intermediate snapshots are replaceable: never stall a CPU worker because
/// its async UI consumer is briefly behind. Terminal snapshots use the awaited
/// publisher below and must not pass through this best-effort path.
fn try_send_progress_event(tx: &mpsc::Sender<ServiceEvent>, event: ServiceEvent) {
    match tx.try_send(event) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            tracing::debug!("Discovery progress snapshot skipped: event queue is full");
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            tracing::warn!("Discovery progress receiver is closed");
        }
    }
}

/// Preserve the actual final outcome under queue pressure. This waits
/// cooperatively for the existing consumer; it neither blocks a runtime thread
/// nor starts detached delivery work. A disconnected consumer ends the wait.
async fn send_terminal_snapshot(tx: &mpsc::Sender<ServiceEvent>, snapshot: &JobSnapshot) {
    debug_assert!(!matches!(
        snapshot.state,
        JobState::Queued | JobState::Running
    ));
    if let Err(error) = tx
        .send(ServiceEvent::DiscoveryUpdated(snapshot.clone()))
        .await
    {
        tracing::error!(
            job_id = ?snapshot.id,
            state = ?snapshot.state,
            %error,
            "Discovery terminal snapshot could not be delivered: receiver is closed"
        );
    }
}

fn log_discovery_event(operation: &str, status: &str, message: String) {
    if let Err(err) = write_subsystem_record(
        SubsystemSection::Discovery,
        discovery_record(operation, status, message),
    ) {
        tracing::error!("Failed to write DISCOVERY section log: {}", err);
    }
}

fn discovery_record(operation: &str, status: &str, message: String) -> SectionedRunRecord {
    let now = system_time_string();
    SectionedRunRecord {
        run_id: format!("discovery-{}-{}", operation, now.replace(':', "-")),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: SubsystemSection::Discovery,
        operation: operation.to_string(),
        status: status.to_string(),
        symbol: None,
        timeframe: None,
        error_code: None,
        message,
        body: String::new(),
    }
}

fn system_time_string() -> String {
    // F-282 fix (2026-05-25): never panic on pre-1970 clock skew.
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(now) => format!("{}.{:09}Z", now.as_secs(), now.subsec_nanos()),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::discovery",
                error = %err,
                "system clock is before UNIX epoch; falling back to sentinel"
            );
            "pre-1970.000000000Z".to_string()
        }
    }
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;

#[cfg(test)]
mod funnel_counter_tests {
    use super::*;
    use neoethos_search::funnel_profile::DiscoveryCandidateCensus;

    #[test]
    fn search_start_updates_the_resolved_plan_without_inventing_completed_counts() {
        for (population, planned) in [(200, 200_000), (3_206, 3_206_000)] {
            let mut snapshot = JobSnapshot::new(JobKind::Discovery);
            snapshot.report.counters = vec![
                ("target_candidates".into(), 200),
                ("target_portfolio".into(), 4),
                ("generations".into(), 1_000),
                ("population".into(), 200),
                ("planned_ga_evaluations".into(), 200_000),
            ];
            let event = DiscoveryProgress::SearchStarted {
                population,
                generations: 1_000,
                max_indicators: 12,
            };
            let expected = vec![
                ("target_candidates".into(), 200),
                ("target_portfolio".into(), 4),
                ("generations".into(), 1_000),
                ("population".into(), population as u64),
                ("planned_ga_evaluations".into(), planned),
                ("max_indicators".into(), 12),
            ];
            // Re-delivery updates the same counters rather than appending
            // ambiguous duplicates. No observed/unique count exists yet.
            for _ in 0..2 {
                apply_backend_discovery_event(&mut snapshot, &event);
                assert_eq!(snapshot.report.counters, expected);
            }
        }
    }

    #[test]
    fn partial_generation_progress_does_not_turn_the_plan_into_completed_evaluations() {
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Running;
        apply_backend_discovery_event(
            &mut snapshot,
            &DiscoveryProgress::SearchStarted {
                population: 3_206,
                generations: 1_000,
                max_indicators: 12,
            },
        );
        apply_backend_discovery_event(
            &mut snapshot,
            &DiscoveryProgress::GenerationCompleted {
                generation: 2,
                total_generations: 1_000,
                best_fitness: 0.25,
                stagnant_generations: 1,
                archived_profitable: 7,
            },
        );
        let expected = vec![
            ("population".into(), 3_206),
            ("generations".into(), 1_000),
            ("planned_ga_evaluations".into(), 3_206_000),
            ("max_indicators".into(), 12),
            ("generation".into(), 2),
            ("archived_profitable".into(), 7),
            ("stagnant_generations".into(), 1),
        ];
        // Exact keys matter: repeated survivors make P*G unsuitable as a
        // unique-candidate count, and this event provides no evaluation total.
        assert_eq!(snapshot.report.counters, expected);
        let cancelled = cancelled_snapshot_from(snapshot, "fixture stopped after generation 2");
        assert_eq!(cancelled.state, JobState::Cancelled);
        assert_eq!(cancelled.report.counters, expected);
    }

    #[test]
    fn queued_validation_target_preserves_all_candidates_and_explicit_caps() {
        let mut settings = neoethos_core::Settings::default();
        for (population, generations, portfolio) in [(3, 1, 2), (200, 1000, 4), (3206, 1000, 4)] {
            settings.models.prop_search_population = population;
            settings.models.prop_search_generations = generations;
            settings.models.prop_search_portfolio_size = portfolio;
            for configured_cap in [0, 1, 7, usize::MAX] {
                settings.models.prop_search_val_candidates = configured_cap;
                assert_eq!(
                    unprepared_validation_candidate_target(&settings),
                    configured_cap,
                    "queued validation must not infer archive coverage from population or portfolio size",
                );
            }
        }
    }

    #[test]
    fn census_updates_preserve_phase_and_distinguish_cap_from_wf_failure() {
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Running;
        snapshot.progress = JobProgress {
            percent: Some(0.965),
            stage: "candidate_walkforward".into(),
            message: "active".into(),
        };
        snapshot.report.counters.push(("population".into(), 200));
        let original_progress = snapshot.progress.clone();
        let census = DiscoveryCandidateCensus {
            ga_returned_candidates: 10_000,
            validation_candidate_limit: 200,
            validation_candidates_admitted: 200,
            validation_candidates_capped: 9_800,
            quality_evaluated: 120,
            walkforward_tested: 20,
            walkforward_passed: 6,
            walkforward_failed: 14,
            walkforward_not_tested: 180,
            ..Default::default()
        };
        apply_backend_discovery_event(
            &mut snapshot,
            &DiscoveryProgress::CandidateCensusUpdated {
                census: census.clone(),
            },
        );
        assert_eq!(snapshot.progress, original_progress);
        assert!(
            snapshot
                .report
                .counters
                .contains(&("population".into(), 200))
        );
        for (key, value) in census.counters() {
            assert!(
                snapshot
                    .report
                    .counters
                    .contains(&(key.into(), value as u64))
            );
        }
        assert!(
            !snapshot
                .report
                .counters
                .iter()
                .any(|(key, _)| key == "rejected")
        );
        apply_backend_discovery_event(
            &mut snapshot,
            &DiscoveryProgress::CandidateCensusUpdated { census },
        );
        assert_eq!(
            snapshot
                .report
                .counters
                .iter()
                .filter(|(key, _)| key == "walkforward_tested")
                .count(),
            1
        );
    }

    #[test]
    fn ranked_pool_does_not_masquerade_as_admitted_candidates() {
        for admitted in [200, 10_000] {
            let mut snapshot = JobSnapshot::new(JobKind::Discovery);
            apply_backend_discovery_event(
                &mut snapshot,
                &DiscoveryProgress::CandidatesRanked {
                    candidate_count: 10_000,
                    truncated_to: admitted,
                },
            );
            assert_eq!(
                snapshot.report.counters,
                vec![("candidates".into(), admitted as u64)],
                "ranking reports retained candidates, not a duplicate skipped count",
            );
            assert!(snapshot.progress.message.contains("10000"));
            assert!(snapshot.progress.message.contains(&admitted.to_string()));
            let capped = 10_000 - admitted;
            apply_backend_discovery_event(
                &mut snapshot,
                &DiscoveryProgress::CandidateCensusUpdated {
                    census: DiscoveryCandidateCensus {
                        ga_returned_candidates: 10_000,
                        validation_candidate_limit: if capped == 0 { 0 } else { admitted },
                        validation_candidates_admitted: admitted,
                        validation_candidates_capped: capped,
                        walkforward_not_tested: admitted,
                        ..Default::default()
                    },
                },
            );
            for (name, expected) in [
                ("candidates", admitted),
                ("validation_candidates_admitted", admitted),
                ("validation_candidates_capped", capped),
            ] {
                assert!(
                    snapshot
                        .report
                        .counters
                        .contains(&(name.into(), expected as u64))
                );
            }
            assert!(
                !snapshot
                    .report
                    .counters
                    .iter()
                    .any(|(name, _)| name == "truncated_candidates")
            );
        }
    }

    #[test]
    fn failed_and_cancelled_snapshots_keep_completed_quality_and_robustness_counts() {
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Running;
        let census = DiscoveryCandidateCensus {
            quality_evaluated: 5,
            walkforward_tested: 4,
            walkforward_passed: 4,
            robustness_removed: Some(1),
            portfolio_selected: 3,
            ..Default::default()
        };
        apply_backend_discovery_event(
            &mut snapshot,
            &DiscoveryProgress::CandidateCensusUpdated { census },
        );
        let expected = snapshot.report.counters.clone();
        let failed = failed_snapshot_from(
            snapshot.clone(),
            anyhow::anyhow!("fixture artifact failure"),
        );
        let cancelled = cancelled_snapshot_from(snapshot, "fixture cancellation");
        assert_eq!(failed.state, JobState::Failed);
        assert_eq!(cancelled.state, JobState::Cancelled);
        for terminal in [failed, cancelled] {
            assert_eq!(terminal.report.counters, expected);
            assert!(
                terminal
                    .report
                    .counters
                    .contains(&("quality_evaluated".into(), 5))
            );
            assert!(
                terminal
                    .report
                    .counters
                    .contains(&("robustness_removed".into(), 1))
            );
            assert!(
                terminal
                    .report
                    .counters
                    .contains(&("portfolio_selected".into(), 3))
            );
            assert!(
                terminal
                    .report
                    .counters
                    .contains(&("walkforward_failed".into(), 0))
            );
        }
    }
}

#[cfg(test)]
#[path = "discovery_preparation_tests.rs"]
mod preparation_tests;

#[cfg(test)]
mod cache_root_tests {
    use super::*;

    #[test]
    fn discovery_paths_use_the_captured_operator_cache_root() {
        let mut settings = neoethos_core::Settings::default();
        settings.system.cache_dir = PathBuf::from("operator-selected-cache");
        let cache_root = std::path::absolute(&settings.system.cache_dir)
            .unwrap()
            .join("discovery");
        let source = DiscoverySettingsSource {
            settings: settings.clone(),
            source_path: PathBuf::from("fixture.yaml"),
            discovery_cache_root: cache_root.clone(),
            exact_bytes: Vec::new(),
            sha256: "fixture-not-financial-authority".to_owned(),
        };
        settings.system.cache_dir = PathBuf::from("later-unrelated-settings");
        for area in ["sources", "research"] {
            assert_eq!(
                source.discovery_cache_root().join(area),
                cache_root.join(area)
            );
        }
    }
}
