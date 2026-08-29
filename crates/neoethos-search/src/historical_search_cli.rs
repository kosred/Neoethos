//! Strict, model-free CLI adapter for receipt-bound historical research.
//!
//! This module deliberately does not read current dataset pointers, broker
//! financial settings, model settings, or process-global search overrides.
//! Every source generation and every feature semantic identity must reproduce
//! the operator-supplied canonical receipt before a candidate is generated.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_core::execution_budget::{
    CpuPermitRequest, InstalledExecutionBudget, WorkerLimit, detected_request_with_parent,
    install_process_budget, installed_process_budget, parse_parent_cpu_assignment,
};
use neoethos_data::{
    CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe, FeatureFrame,
    SelectedDatasetGenerationV1,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::data_selection::{CanonicalSearchInputReceiptV2, CanonicalSearchRunInputV2};
use crate::genetic::{Gene, SmcSearchConfig, new_random_gene};
use crate::historical_research::{
    HISTORICAL_CANDIDATE_RANKING_POLICY_ID, HistoricalCandidateDistanceSourceV1,
    HistoricalCandidateFailurePolicyV1, HistoricalCandidateScanRequestV2,
    HistoricalCandidateScanResultV2, HistoricalResearchAccountingV1,
    HistoricalResearchArtifactClassV1, HistoricalResearchBackendV1,
    HistoricalResearchPromotionEligibilityV1, historical_candidate_signal_identity_sha256,
    scan_historical_candidates_v2,
};
use crate::historical_search_receipt_prep::{
    ExactSelectedFeatureInput, build_exact_selected_feature_input,
};

const HISTORICAL_SEARCH_CLI_SCHEMA_VERSION: u16 = 2;
const CANDIDATE_GENERATOR_POLICY_ID: &str =
    "neoethos.historical-search-cli.deterministic-random-gene.v1";
const DISTANCE_POLICY_ID: &str = "canonical.reference-ohlc.true-range-causal-carry-forward.v1";
const MIN_VALID_CANDIDATE_ROWS: usize = 32;
const MAX_GENERATION_ATTEMPTS_PER_CANDIDATE: usize = 512;
static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct HistoricalSearchArgs {
    expected_input_receipt: PathBuf,
    root: PathBuf,
    output: PathBuf,
    operator_seed: u64,
    candidate_count: usize,
    max_indicators: usize,
    stop_multiple: f64,
    target_multiple: f64,
    cpu_threads: Option<WorkerLimit>,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct HistoricalSearchArtifactV2 {
    schema_version: u16,
    input_receipt_sha256: String,
    input_receipt: CanonicalSearchInputReceiptV2,
    candidate_generation: HistoricalCandidateGenerationV1,
    search: HistoricalSearchResultV2,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct HistoricalSearchResultV2 {
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    backend: HistoricalResearchBackendV1,
    accounting: HistoricalResearchAccountingV1,
    #[serde(flatten)]
    scan: HistoricalCandidateScanResultV2,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct HistoricalCandidateGenerationV1 {
    policy_id: &'static str,
    operator_seed: u64,
    derived_seed: u64,
    candidate_count: usize,
    max_indicators: usize,
    eligible_feature_count: usize,
    minimum_valid_rows: usize,
    distance_policy_id: &'static str,
    signal_rules: Vec<HistoricalCandidateSignalRuleV1>,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct HistoricalCandidateSignalRuleV1 {
    input_ordinal: u64,
    candidate_identity_sha256: String,
    feature_indices: Vec<usize>,
    weights: Vec<f64>,
    long_threshold: f64,
    short_threshold: f64,
}

/// Install the sole process-capacity authority allowed by strict historical
/// search: automatic host detection, optionally narrowed by a validated parent
/// `--cpu-threads` assignment. This deliberately has no `Settings` input.
pub fn install_historical_search_process_budget(process_args: &[String]) -> Result<()> {
    let parent = parse_parent_cpu_assignment(process_args)?;
    install_process_budget(detected_request_with_parent(parent))?;
    Ok(())
}

/// Run one explicit ordered CpuOnly historical candidate scan.
pub fn run(args: &[String]) -> Result<()> {
    let args = HistoricalSearchArgs::parse(args)?;
    let installed = installed_process_budget()
        .context("historical search requires the immutable process CPU budget to be installed")?;
    args.validate_cpu_assignment(installed)?;

    // This read and strict decode intentionally precede every dataset open or
    // feature computation. A path typo can never turn into a current-generation
    // fallback or an expensive partial run.
    let receipt_bytes = fs::read(&args.expected_input_receipt).with_context(|| {
        format!(
            "read --expected-input-receipt {} before feature computation",
            args.expected_input_receipt.display()
        )
    })?;
    let receipt = CanonicalSearchInputReceiptV2::from_json_bytes(&receipt_bytes)
        .context("validate --expected-input-receipt before feature computation")?;
    let receipt_sha256 = receipt
        .identity_sha256()
        .context("hash exact input receipt before feature computation")?;
    let anchor = receipt
        .validate()
        .context("validate exact input receipt anchor before feature computation")?;

    let width = installed.resolved().effective_worker_limit;
    let broker = installed.broker().clone();
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), width);

    let load_lease = broker
        .acquire(CpuPermitRequest::local(width))
        .context("acquire the process CPU budget for exact Vortex load and feature build")?;
    let loaded = executor
        .execute(load_lease.into_transfer(), || {
            load_and_rebuild_exact_input(&args.root, &receipt, &anchor)
        })
        .map_err(|error| anyhow::anyhow!("budgeted exact feature execution failed: {error}"))??;

    let rebuilt_receipt =
        CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, loaded.features())
            .context("rebuild canonical receipt from exact direct-generation features")?;
    ensure!(
        rebuilt_receipt == receipt,
        "rebuilt direct-generation feature receipt does not exactly match --expected-input-receipt"
    );
    let run_input =
        CanonicalSearchRunInputV2::new(receipt.clone(), loaded.features(), loaded.base_frame())
            .context("bind exact receipt, features, and base generation for historical research")?;

    let eligible_features = eligible_feature_indices(run_input.features())?;
    ensure!(
        !eligible_features.is_empty(),
        "exact feature frame has no varying feature with at least {MIN_VALID_CANDIDATE_ROWS} valid finite rows"
    );
    let derived_seed = derive_candidate_seed(
        &receipt_sha256,
        args.operator_seed,
        args.candidate_count,
        args.max_indicators,
    );
    let candidates = generate_candidates(
        run_input.features(),
        &eligible_features,
        derived_seed,
        args.candidate_count,
        args.max_indicators,
    )?;
    let distance = causal_true_range_distance(run_input.ohlcv())?;

    let scan_lease = broker
        .acquire(CpuPermitRequest::local(width))
        .context("acquire the process CPU budget for historical candidate scan")?;
    let search = scan_historical_candidates_v2(
        HistoricalCandidateScanRequestV2 {
            input: &run_input,
            backend: HistoricalResearchBackendV1::CpuOnly,
            candidates: &candidates,
            failure_policy: HistoricalCandidateFailurePolicyV1::FailEntireScan,
            distance_source: HistoricalCandidateDistanceSourceV1 {
                receipt_sha256: &receipt_sha256,
                semantic_id: DISTANCE_POLICY_ID,
                values: &distance,
            },
            stop_multiple: args.stop_multiple,
            target_multiple: args.target_multiple,
        },
        &executor,
        scan_lease.into_transfer(),
    )
    .context("run budgeted CpuOnly historical candidate scan")?;
    let signal_rules = candidates
        .iter()
        .enumerate()
        .map(|(input_ordinal, candidate)| {
            Ok(HistoricalCandidateSignalRuleV1 {
                input_ordinal: u64::try_from(input_ordinal)
                    .context("candidate input ordinal does not fit u64")?,
                candidate_identity_sha256: historical_candidate_signal_identity_sha256(candidate)
                    .map_err(|error| anyhow::anyhow!(error))?,
                feature_indices: candidate.indices.clone(),
                weights: candidate.weights.clone(),
                long_threshold: candidate.long_threshold,
                short_threshold: candidate.short_threshold,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let artifact = HistoricalSearchArtifactV2 {
        schema_version: HISTORICAL_SEARCH_CLI_SCHEMA_VERSION,
        input_receipt_sha256: receipt_sha256.clone(),
        input_receipt: receipt,
        candidate_generation: HistoricalCandidateGenerationV1 {
            policy_id: CANDIDATE_GENERATOR_POLICY_ID,
            operator_seed: args.operator_seed,
            derived_seed,
            candidate_count: candidates.len(),
            max_indicators: args.max_indicators,
            eligible_feature_count: eligible_features.len(),
            minimum_valid_rows: MIN_VALID_CANDIDATE_ROWS,
            distance_policy_id: DISTANCE_POLICY_ID,
            signal_rules,
        },
        search: HistoricalSearchResultV2 {
            artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
            promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
            backend: HistoricalResearchBackendV1::CpuOnly,
            accounting: HistoricalResearchAccountingV1::GrossReferenceR,
            scan: search,
        },
    };
    let artifact_bytes = serde_json::to_vec(&artifact).context("serialize historical-search v2")?;
    atomic_write_create_new(&args.output, &artifact_bytes)?;

    println!("receipt_sha256={receipt_sha256}");
    println!(
        "search_identity_sha256={}",
        artifact.search.scan.search_identity_sha256()
    );
    println!(
        "ranking_policy_id={}",
        HISTORICAL_CANDIDATE_RANKING_POLICY_ID
    );
    println!("artifact_class=ResearchOnly");
    println!("promotion_eligibility=NotPromotionEligible");
    println!(
        "accounting={:?}",
        HistoricalResearchAccountingV1::GrossReferenceR
    );
    println!("output={}", args.output.display());
    Ok(())
}

impl HistoricalSearchArgs {
    fn parse(args: &[String]) -> Result<Self> {
        let mut receipt = None;
        let mut root = None;
        let mut output = None;
        let mut seed = None;
        let mut candidates = None;
        let mut max_indicators = None;
        let mut stop_multiple = None;
        let mut target_multiple = None;
        let mut cpu_threads = None;
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            if let Some(raw) = flag.strip_prefix("--cpu-threads=") {
                ensure!(
                    cpu_threads.is_none(),
                    "--cpu-threads may be supplied only once"
                );
                cpu_threads = Some(parse_positive_worker_limit(raw, "--cpu-threads")?);
                index += 1;
                continue;
            }
            let value = args.get(index + 1).with_context(|| {
                format!("historical search flag {flag} requires exactly one value")
            })?;
            match flag {
                "--expected-input-receipt" => set_once_path(&mut receipt, flag, value)?,
                "--root" => set_once_path(&mut root, flag, value)?,
                "--out" => set_once_path(&mut output, flag, value)?,
                "--seed" => set_once(&mut seed, flag, parse_u64(value, flag)?)?,
                "--candidates" => {
                    set_once(&mut candidates, flag, parse_positive_usize(value, flag)?)?
                }
                "--max-indicators" => set_once(
                    &mut max_indicators,
                    flag,
                    parse_positive_usize(value, flag)?,
                )?,
                "--stop-multiple" => {
                    set_once(&mut stop_multiple, flag, parse_positive_f64(value, flag)?)?
                }
                "--target-multiple" => {
                    set_once(&mut target_multiple, flag, parse_positive_f64(value, flag)?)?
                }
                "--cpu-threads" => {
                    ensure!(
                        cpu_threads.is_none(),
                        "--cpu-threads may be supplied only once"
                    );
                    cpu_threads = Some(parse_positive_worker_limit(value, flag)?);
                }
                _ => bail!("unknown historical search flag `{flag}`"),
            }
            index += 2;
        }

        Ok(Self {
            expected_input_receipt: required(receipt, "--expected-input-receipt")?,
            root: required(root, "--root")?,
            output: required(output, "--out")?,
            operator_seed: required(seed, "--seed")?,
            candidate_count: required(candidates, "--candidates")?,
            max_indicators: required(max_indicators, "--max-indicators")?,
            stop_multiple: required(stop_multiple, "--stop-multiple")?,
            target_multiple: required(target_multiple, "--target-multiple")?,
            cpu_threads,
        })
    }

    fn validate_cpu_assignment(&self, installed: &InstalledExecutionBudget) -> Result<()> {
        match (self.cpu_threads, installed.resolved().parent_limit) {
            (Some(received), Some(installed_parent)) => ensure!(
                received == installed_parent.limit,
                "--cpu-threads={} does not match installed parent assignment {}",
                received.get(),
                installed_parent.limit.get()
            ),
            (Some(received), None) => bail!(
                "--cpu-threads={} was not installed at process startup",
                received.get()
            ),
            (None, _) => {}
        }
        Ok(())
    }
}

fn load_and_rebuild_exact_input(
    root: &Path,
    expected: &CanonicalSearchInputReceiptV2,
    anchor: &CanonicalDatasetIdentity,
) -> Result<ExactSelectedFeatureInput> {
    let mut selections = BTreeMap::<CanonicalTimeframe, SelectedDatasetGenerationV1>::new();
    for binding in expected.source_bindings() {
        let identity = CanonicalDatasetIdentity::from_path_component(binding.dataset_identity())
            .with_context(|| {
                format!(
                    "decode receipt source identity {}",
                    binding.dataset_identity()
                )
            })?;
        ensure!(
            identity.scope() == anchor.scope()
                && identity.symbol_name() == anchor.symbol_name()
                && identity.bar_timestamp_convention() == anchor.bar_timestamp_convention(),
            "receipt source {} is outside the exact anchor series {}",
            identity.to_path_component(),
            anchor.to_path_component()
        );
        let selected = SelectedDatasetGenerationV1::new(
            identity.clone(),
            binding.generation_id(),
            binding.manifest_sha256(),
        )?;
        if let Some(previous) = selections.insert(identity.timeframe(), selected.clone()) {
            ensure!(
                previous == selected,
                "receipt contains conflicting exact generations for direct timeframe {}",
                identity.timeframe()
            );
        }
    }
    let selected_anchor = selections
        .get(&anchor.timeframe())
        .context("receipt has no selected generation for its anchor timeframe")?
        .clone();
    ensure!(
        selected_anchor.identity() == anchor,
        "receipt anchor timeframe resolves to a different canonical identity"
    );
    let selected =
        CanonicalDatasetSeriesReceiptV1::new(selected_anchor, selections.into_values().collect())
            .context("validate exact receipt-selected direct timeframe set")?;
    let loaded = build_exact_selected_feature_input(root, &selected)
        .context("rebuild features through the shared exact selected-frame recipe")?;
    expected
        .validate_against(anchor, loaded.features())
        .context("recomputed direct-generation features disagree with exact receipt")?;
    Ok(loaded)
}

fn eligible_feature_indices(features: &FeatureFrame) -> Result<Vec<usize>> {
    let mut eligible = Vec::new();
    for index in 0..features.n_features() {
        let column = features
            .feature_column(index)
            .with_context(|| format!("materialize feature column {index} for candidate census"))?;
        let mut valid_count = 0_usize;
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for (&value, validity) in column.values.iter().zip(&column.validity) {
            if validity.is_valid() {
                ensure!(
                    value.is_finite(),
                    "feature column {index} has a valid non-finite value"
                );
                valid_count += 1;
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
        }
        if valid_count >= MIN_VALID_CANDIDATE_ROWS && minimum < maximum {
            eligible.push(index);
        }
    }
    Ok(eligible)
}

fn generate_candidates(
    features: &FeatureFrame,
    eligible: &[usize],
    seed: u64,
    candidate_count: usize,
    max_indicators: usize,
) -> Result<Vec<Gene>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let disabled_smc = SmcSearchConfig {
        force_ratio: 0.0,
        min_flags: 0,
        p_ob: 0.0,
        p_fvg: 0.0,
        p_liq: 0.0,
        p_premium: 0.0,
        p_inducement: 0.0,
        p_mtf: 0.0,
        p_bos: 0.0,
        p_choch: 0.0,
        p_eqh: 0.0,
        p_eql: 0.0,
        p_displacement: 0.0,
    };
    let max_indicators = max_indicators.min(eligible.len());
    ensure!(
        max_indicators > 0,
        "no eligible feature can enter a candidate"
    );
    let max_attempts = candidate_count
        .checked_mul(MAX_GENERATION_ATTEMPTS_PER_CANDIDATE)
        .context("candidate generation attempt bound overflow")?;
    let mut identities = BTreeSet::new();
    let mut candidates = Vec::with_capacity(candidate_count);
    for _ in 0..max_attempts {
        if candidates.len() == candidate_count {
            break;
        }
        let mut candidate =
            new_random_gene(eligible.len(), max_indicators, 0, &disabled_smc, &mut rng);
        for index in &mut candidate.indices {
            *index = eligible[*index];
        }
        disable_structural_flags(&mut candidate);
        if candidate_valid_intersection(features, &candidate)? < MIN_VALID_CANDIDATE_ROWS {
            continue;
        }
        let identity = historical_candidate_signal_identity_sha256(&candidate)
            .map_err(|error| anyhow::anyhow!(error))?;
        if identities.insert(identity) {
            candidates.push(candidate);
        }
    }
    ensure!(
        candidates.len() == candidate_count,
        "generated only {} unique valid candidates after {max_attempts} attempts; requested {candidate_count}",
        candidates.len()
    );
    Ok(candidates)
}

fn disable_structural_flags(candidate: &mut Gene) {
    candidate.use_ob = false;
    candidate.use_fvg = false;
    candidate.use_liq_sweep = false;
    candidate.mtf_confirmation = false;
    candidate.use_premium_discount = false;
    candidate.use_inducement = false;
    candidate.use_bos = false;
    candidate.use_choch = false;
    candidate.use_eqh = false;
    candidate.use_eql = false;
    candidate.use_displacement = false;
    candidate.stop_vol_mult = 0.0;
}

fn candidate_valid_intersection(features: &FeatureFrame, candidate: &Gene) -> Result<usize> {
    let columns = candidate
        .indices
        .iter()
        .map(|&index| features.feature_column(index))
        .collect::<Result<Vec<_>>>()?;
    Ok((0..features.n_samples())
        .filter(|&row| {
            columns
                .iter()
                .all(|column| column.validity[row].is_valid() && column.values[row].is_finite())
        })
        .count())
}

fn causal_true_range_distance(ohlcv: &neoethos_data::Ohlcv) -> Result<Vec<f64>> {
    let rows = ohlcv.len();
    ensure!(
        rows >= 2,
        "historical research requires at least two OHLCV rows"
    );
    ensure!(
        ohlcv.open.len() == rows
            && ohlcv.high.len() == rows
            && ohlcv.low.len() == rows
            && ohlcv
                .timestamp
                .as_ref()
                .is_some_and(|values| values.len() == rows),
        "historical research OHLCV columns disagree"
    );
    let mut output = Vec::with_capacity(rows);
    let mut last_positive = None;
    for row in 0..rows {
        let open = ohlcv.open[row];
        let high = ohlcv.high[row];
        let low = ohlcv.low[row];
        let close = ohlcv.close[row];
        ensure!(
            [open, high, low, close]
                .into_iter()
                .all(|value| value.is_finite() && value > 0.0),
            "OHLC row {row} is not finite and positive"
        );
        ensure!(
            high >= open.max(close) && low <= open.min(close) && high >= low,
            "OHLC row {row} violates candle bounds"
        );
        let previous_close = if row == 0 {
            close
        } else {
            ohlcv.close[row - 1]
        };
        let observed = (high - low)
            .max((high - previous_close).abs())
            .max((low - previous_close).abs());
        let distance = if observed.is_finite() && observed > 0.0 {
            last_positive = Some(observed);
            observed
        } else if let Some(previous) = last_positive {
            previous
        } else {
            positive_ulp(close)?
        };
        output.push(distance);
    }
    Ok(output)
}

fn positive_ulp(value: f64) -> Result<f64> {
    ensure!(
        value.is_finite() && value > 0.0,
        "ULP anchor is not positive finite"
    );
    let next = f64::from_bits(
        value
            .to_bits()
            .checked_add(1)
            .context("positive f64 ULP overflow")?,
    );
    let ulp = next - value;
    ensure!(ulp.is_finite() && ulp > 0.0, "positive f64 ULP is invalid");
    Ok(ulp)
}

fn derive_candidate_seed(
    receipt_sha256: &str,
    operator_seed: u64,
    candidate_count: usize,
    max_indicators: usize,
) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.historical-search-cli.candidate-seed.v1\0");
    hasher.update(receipt_sha256.as_bytes());
    hasher.update(CANDIDATE_GENERATOR_POLICY_ID.as_bytes());
    hasher.update(operator_seed.to_le_bytes());
    hasher.update((candidate_count as u64).to_le_bytes());
    hasher.update((max_indicators as u64).to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

fn atomic_write_create_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    ensure!(
        parent.is_dir(),
        "output parent {} does not exist",
        parent.display()
    );
    ensure!(
        !path.exists(),
        "refusing to overwrite existing output {}",
        path.display()
    );
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("output path has no UTF-8 file name")?;
    let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.neoethos-{}-{sequence}.tmp",
        std::process::id()
    ));
    let mut cleanup = TemporaryOutput::new(temporary.clone());
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("create temporary output {}", temporary.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write temporary output {}", temporary.display()))?;
    file.sync_all()
        .with_context(|| format!("sync temporary output {}", temporary.display()))?;
    drop(file);
    fs::hard_link(&temporary, path).with_context(|| {
        format!(
            "atomically install create-new output {} from {}",
            path.display(),
            temporary.display()
        )
    })?;
    fs::remove_file(&temporary)
        .with_context(|| format!("remove linked temporary output {}", temporary.display()))?;
    cleanup.disarm();
    Ok(())
}

struct TemporaryOutput {
    path: PathBuf,
    armed: bool,
}

impl TemporaryOutput {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if self.armed
            && let Err(error) = fs::remove_file(&self.path)
        {
            eprintln!(
                "ERROR failed to clean historical-search temporary output {}: {error}",
                self.path.display()
            );
        }
    }
}

fn set_once<T>(slot: &mut Option<T>, flag: &str, value: T) -> Result<()> {
    ensure!(slot.is_none(), "{flag} may be supplied only once");
    *slot = Some(value);
    Ok(())
}

fn set_once_path(slot: &mut Option<PathBuf>, flag: &str, value: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{flag} path is empty");
    set_once(slot, flag, PathBuf::from(value))
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T> {
    value.with_context(|| format!("historical search requires {flag}"))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .with_context(|| format!("{flag} expects a u64, got `{value}`"))
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize> {
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("{flag} expects a positive integer, got `{value}`"))?;
    ensure!(parsed > 0, "{flag} must be greater than zero");
    Ok(parsed)
}

fn parse_positive_worker_limit(value: &str, flag: &str) -> Result<WorkerLimit> {
    WorkerLimit::new(parse_positive_usize(value, flag)?)
        .map_err(|error| anyhow::anyhow!("{flag}: {error}"))
}

fn parse_positive_f64(value: &str, flag: &str) -> Result<f64> {
    let parsed = value
        .parse::<f64>()
        .with_context(|| format!("{flag} expects a finite positive number, got `{value}`"))?;
    ensure!(
        parsed.is_finite() && parsed > 0.0,
        "{flag} must be finite and positive"
    );
    Ok(parsed)
}
