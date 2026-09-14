//! Receipt-bound canonical-trendbar cost, contract, and training commands.

use std::collections::BTreeSet;
use std::fs;
#[cfg(feature = "gpu-nvidia-full")]
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
#[cfg(feature = "gpu-nvidia")]
use neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit;
use neoethos_broker_history::canonical_research_costs::{
    ScreeningCostEnvelopeWireV2, build_screening_cost_envelope_v2,
    ensure_unique_selected_timeframe, ensure_unique_series, generation_sha256,
    read_bounded_regular_file, validate_broker_symbol_contract, validate_costs,
    validate_settings_source,
};
use neoethos_broker_history::{
    CanonicalTrendbarAcquisitionStoreV1, CanonicalTrendbarMatrixReceiptV1,
    CanonicalTrendbarPlanReceiptV1,
};
use neoethos_data::{CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe};
#[cfg(feature = "gpu-nvidia")]
use neoethos_data::{FeatureBuildOptions, load_exact_canonical_timeframe};
#[cfg(feature = "gpu-nvidia-full")]
use neoethos_search::historical_research::{
    HistoricalResearchArtifactClassV1, HistoricalResearchPromotionEligibilityV1,
};
#[cfg(feature = "gpu-nvidia-full")]
use serde::Deserialize;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[cfg(feature = "gpu-nvidia-full")]
const CANONICAL_TRAIN_SCHEMA_V1: &str = "neoethos.canonical-trendbar-training.v1";
#[cfg(feature = "gpu-nvidia-full")]
const CANONICAL_TRAIN_RECEIPT_SCHEMA_V1: &str = "neoethos.canonical-trendbar-training-receipt.v1";
#[cfg(feature = "gpu-nvidia")]
const MAX_CONTRACT_ARTIFACT_BYTES: u64 =
    neoethos_search::MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 as u64;
#[cfg(feature = "gpu-nvidia-full")]
const MAX_TRAINING_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const CANONICAL_TRAIN_REQUIRED_FLAGS: [&str; 14] = [
    "--authority-root",
    "--data-root",
    "--plan-sha256",
    "--matrix-sha256",
    "--symbol",
    "--base-timeframe",
    "--input-receipt",
    "--cost-assumptions",
    "--broker-symbol-contract",
    "--settings-source",
    "--models-dir",
    "--oos-from-ms",
    "--out",
    "--receipt-out",
];
const COST_BUILD_REQUIRED_FLAGS: [&str; 9] = [
    "--authority-root",
    "--data-root",
    "--plan-sha256",
    "--matrix-sha256",
    "--symbol",
    "--basis-timeframe",
    "--broker-symbol-contract",
    "--settings-source",
    "--out",
];
const CONTRACT_BUILD_REQUIRED_FLAGS: [&str; 11] = [
    "--authority-root",
    "--data-root",
    "--plan-sha256",
    "--matrix-sha256",
    "--symbol",
    "--base-timeframe",
    "--cost-assumptions",
    "--broker-symbol-contract",
    "--settings-source",
    "--contract-out",
    "--receipt-out",
];

#[cfg(feature = "gpu-nvidia-full")]
#[derive(Debug, Serialize)]
struct CanonicalTrainingArtifactWireV1 {
    schema: &'static str,
    version: u16,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    authorization_issued: bool,
    symbol: String,
    base_timeframe: String,
    plan_sha256: String,
    matrix_sha256: String,
    canonical_series: CanonicalDatasetSeriesReceiptV1,
    input_receipt_sha256: String,
    input_receipt_file_sha256: String,
    input_receipt_exact_utf8: String,
    research_contract_sha256: String,
    research_contract: neoethos_search::CanonicalTrendbarResearchExecutionContractV3,
    resolved_settings: neoethos_core::Settings,
    cost_assumption_file_sha256: String,
    cost_assumption_exact_utf8: String,
    settings_source_file_sha256: String,
    settings_source_exact_utf8: String,
    broker_symbol_contract_file_sha256: String,
    broker_symbol_contract_exact_utf8: String,
    cost_assumptions: ScreeningCostEnvelopeWireV2,
    training_oos_from_ms: i64,
    planned_models: Vec<String>,
    completed_models: Vec<String>,
    failed_models: Vec<TrainingFailureWireV1>,
    training_label_round_trip_cost_pips: f64,
    model_artifacts: Vec<ModelArtifactEvidenceWireV1>,
}

#[cfg(feature = "gpu-nvidia-full")]
#[derive(Debug, Serialize)]
struct TrainingFailureWireV1 {
    name: String,
    error: String,
}

#[cfg(feature = "gpu-nvidia-full")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ModelArtifactEvidenceWireV1 {
    model_name: String,
    relative_dir: String,
    tree_sha256: String,
    file_count: u64,
    total_bytes: u64,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalCostBuildOutcomeV1<'a> {
    schema: &'static str,
    version: u16,
    cost_assumption_sha256: &'a str,
    path: &'a Path,
}

#[cfg(feature = "gpu-nvidia")]
#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalContractBuildOutcomeV1<'a> {
    schema: &'static str,
    version: u16,
    base_row_count: usize,
    oos_from_ms: i64,
    contract_sha256: &'a str,
    receipt_sha256: &'a str,
    contract_path: &'a Path,
    receipt_path: &'a Path,
}

#[cfg(feature = "gpu-nvidia")]
pub fn build_contract(args: &[String], settings: &neoethos_core::Settings) -> Result<()> {
    validate_contract_build_args(args)?;
    let authority_root = required_path(args, "--authority-root")?;
    let data_root = required_path(args, "--data-root")?;
    let plan_sha256 = required(args, "--plan-sha256")?;
    let matrix_sha256 = required(args, "--matrix-sha256")?;
    let symbol = required(args, "--symbol")?;
    let base_timeframe = required(args, "--base-timeframe")?
        .parse::<CanonicalTimeframe>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let cost_assumption_path = required_path(args, "--cost-assumptions")?;
    let broker_symbol_contract_path = required_path(args, "--broker-symbol-contract")?;
    let settings_source_path = required_path(args, "--settings-source")?;
    let contract_out = required_path(args, "--contract-out")?;
    let receipt_out = required_path(args, "--receipt-out")?;
    ensure_distinct_output_targets(
        &contract_out,
        &receipt_out,
        &[
            &cost_assumption_path,
            &broker_symbol_contract_path,
            &settings_source_path,
        ],
    )?;

    let plan_receipt = CanonicalTrendbarPlanReceiptV1::from_sha256(plan_sha256)?;
    let matrix_receipt = CanonicalTrendbarMatrixReceiptV1::from_sha256(matrix_sha256)?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority_root);
    let plan = store.open_plan(&plan_receipt)?;
    let matrix = store.open_matrix(&data_root, &plan_receipt, &matrix_receipt)?;
    let series = ensure_unique_series(&matrix, &symbol)?;
    series.validate()?;
    let selected_base = ensure_unique_selected_timeframe(series, base_timeframe)?;
    let exact_base = load_exact_canonical_timeframe(&data_root, selected_base)
        .context("open exact canonical base timeframe for discovery split")?;
    let normalization_training_rows =
        neoethos_search::canonical_discovery_normalization_training_rows(exact_base.ohlcv().len())?;

    let cost_assumption_bytes = read_bounded_regular_file(&cost_assumption_path)?;
    let costs: ScreeningCostEnvelopeWireV2 = serde_json::from_slice(&cost_assumption_bytes)
        .context("decode canonical screening-cost envelope V2")?;
    let settings_source_bytes = read_bounded_regular_file(&settings_source_path)?;
    validate_settings_source(
        settings,
        &settings_source_path,
        &settings_source_bytes,
        &costs,
    )?;
    let broker_symbol_contract_bytes = read_bounded_regular_file(&broker_symbol_contract_path)?;
    let broker_cost_facts =
        validate_broker_symbol_contract(&broker_symbol_contract_bytes, &costs, &symbol, &plan)?;
    let pip_value_per_lot = validate_costs(
        &costs,
        &symbol,
        settings,
        &plan,
        &matrix,
        &data_root,
        broker_cost_facts,
    )?;

    let mut feature_options = canonical_feature_options(settings, base_timeframe)?;
    feature_options.normalization_training_rows = Some(normalization_training_rows);
    let search_input = neoethos_search::data_selection::CanonicalSearchInput::from_exact_series_receipt_gpu_exact_parity_cpu_reference_v3(
        &data_root,
        series,
        base_timeframe,
        &feature_options,
    )
    .context(
        "build exact resident-GPU-V3-subset CPU reference input for standalone research contract",
    )?;
    let base_ohlcv = search_input.base_frame().ohlcv();
    let base_row_count = base_ohlcv.len();
    ensure!(
        base_row_count == exact_base.ohlcv().len(),
        "exact canonical base timeframe row count changed during contract construction"
    );
    let normalization_training_rows =
        neoethos_search::canonical_discovery_normalization_training_rows(base_row_count)?;
    ensure!(
        feature_options.normalization_training_rows.as_ref() == Some(&normalization_training_rows),
        "canonical feature normalization split changed during contract construction"
    );
    let oos_from_ms = base_ohlcv
        .timestamp
        .as_ref()
        .and_then(|timestamps| timestamps.get(normalization_training_rows.end))
        .copied()
        .context("exact canonical base timeframe has no timestamp at the discovery OOS split")?;
    let receipt = search_input.receipt()?;
    validate_input_receipt_against_series(&receipt, series, base_timeframe)?;
    let assumption_source_sha256 = format!("{:x}", Sha256::digest(&cost_assumption_bytes));
    let contract = neoethos_search::CanonicalTrendbarResearchExecutionContractV3::new(
        receipt.clone(),
        neoethos_search::CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: &costs.symbol,
            account_currency: &costs.account_currency,
            assumption_source_id: &costs.assumption_source_id,
            assumption_source_sha256: &assumption_source_sha256,
            pip_size: costs.pip_size,
            pip_value_per_lot,
            full_spread_pips_assumption: costs.full_spread_pips_assumption,
            slippage_pips_per_fill_assumption: costs.slippage_pips_per_fill_assumption,
            commission_account_per_lot_per_fill_assumption: costs
                .commission_account_per_lot_per_fill_assumption,
            swap_long_pips_per_day: costs.swap_long_pips_per_day,
            swap_short_pips_per_day: costs.swap_short_pips_per_day,
            pnl_conversion_fee_rate: costs.pnl_conversion_fee_rate,
        },
    )?;
    contract.validate_against_receipt(&receipt)?;

    neoethos_core::storage::json::write_json_atomic(&receipt_out, &receipt)
        .context("publish standalone canonical search-input receipt")?;
    neoethos_core::storage::json::write_json_atomic(&contract_out, &contract)
        .context("publish standalone canonical research contract")?;
    let contract_bytes = read_regular_file_with_limit(&contract_out, MAX_CONTRACT_ARTIFACT_BYTES)?;
    let receipt_bytes = read_regular_file_with_limit(&receipt_out, MAX_CONTRACT_ARTIFACT_BYTES)?;
    let reopened_contract: neoethos_search::CanonicalTrendbarResearchExecutionContractV3 =
        serde_json::from_slice(&contract_bytes).context("reopen standalone research contract")?;
    let reopened_receipt =
        neoethos_search::CanonicalSearchInputReceiptV2::from_json_bytes(&receipt_bytes)
            .context("reopen standalone canonical search-input receipt")?;
    ensure!(
        reopened_contract == contract && reopened_receipt == receipt,
        "standalone canonical contract or receipt did not reopen exactly"
    );
    reopened_contract.validate_against_receipt(&reopened_receipt)?;
    let contract_sha256 = format!("{:x}", Sha256::digest(&contract_bytes));
    let receipt_sha256 = format!("{:x}", Sha256::digest(&receipt_bytes));
    println!(
        "{}",
        serde_json::to_string(&CanonicalContractBuildOutcomeV1 {
            schema: "neoethos.canonical-contract-build-outcome.v1",
            version: 1,
            base_row_count,
            oos_from_ms,
            contract_sha256: &contract_sha256,
            receipt_sha256: &receipt_sha256,
            contract_path: &contract_out,
            receipt_path: &receipt_out,
        })?
    );
    Ok(())
}

#[cfg(not(feature = "gpu-nvidia"))]
pub fn build_contract(args: &[String], _settings: &neoethos_core::Settings) -> Result<()> {
    validate_contract_build_args(args)?;
    anyhow::bail!(
        "canonical-contract-build requires the gpu-nvidia feature so it can seal the \
         exact resident GPU V3 Classic subset; the ordinary full Classic CPU graph is \
         not a substitute"
    )
}

pub fn build_cost_assumptions(args: &[String], settings: &neoethos_core::Settings) -> Result<()> {
    validate_cost_build_args(args)?;
    let authority_root = required_path(args, "--authority-root")?;
    let data_root = required_path(args, "--data-root")?;
    let plan_sha256 = required(args, "--plan-sha256")?;
    let matrix_sha256 = required(args, "--matrix-sha256")?;
    let symbol = required(args, "--symbol")?;
    let basis_timeframe = required(args, "--basis-timeframe")?
        .parse::<CanonicalTimeframe>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    ensure!(
        basis_timeframe == CanonicalTimeframe::D1,
        "canonical cost evidence requires the explicit direct D1 basis"
    );
    let broker_symbol_contract_path = required_path(args, "--broker-symbol-contract")?;
    let settings_source_path = required_path(args, "--settings-source")?;
    let out = required_path(args, "--out")?;
    ensure_distinct_cost_output(&out, &[&broker_symbol_contract_path, &settings_source_path])?;

    let plan_receipt = CanonicalTrendbarPlanReceiptV1::from_sha256(plan_sha256)?;
    let matrix_receipt = CanonicalTrendbarMatrixReceiptV1::from_sha256(matrix_sha256)?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority_root);
    let costs = build_screening_cost_envelope_v2(
        settings,
        &data_root,
        &store,
        &plan_receipt,
        &matrix_receipt,
        &symbol,
        basis_timeframe,
        &broker_symbol_contract_path,
        &settings_source_path,
    )?;

    neoethos_core::storage::json::write_json_atomic(&out, &costs)
        .context("publish canonical screening-cost envelope")?;
    let published_bytes = read_bounded_regular_file(&out)?;
    let reopened: ScreeningCostEnvelopeWireV2 = serde_json::from_slice(&published_bytes)
        .context("reopen canonical screening-cost envelope")?;
    ensure!(
        reopened == costs,
        "published canonical cost assumptions do not reopen exactly"
    );
    let cost_assumption_sha256 = format!("{:x}", Sha256::digest(&published_bytes));
    println!(
        "{}",
        serde_json::to_string(&CanonicalCostBuildOutcomeV1 {
            schema: "neoethos.canonical-cost-build-outcome.v1",
            version: 1,
            cost_assumption_sha256: &cost_assumption_sha256,
            path: &out,
        })?
    );
    Ok(())
}

/// Seal the exact, already-built CPU feature frame for one canonical-trendbar
/// research run.
///
/// This is deliberately separate from `canonical-contract-build`: that command
/// authors the resident-GPU-V3 parity subset, while this helper binds the
/// ordinary adaptive CPU vocabulary that the caller has already materialized.
/// Both routes require the same exact acquisition plan, matrix, broker symbol,
/// settings bytes, and D1-derived cost envelope. The returned contract remains
/// `ResearchOnly` and `NotPromotionEligible`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn seal_cpu_research_contract_for_input(
    settings: &neoethos_core::Settings,
    authority_root: &Path,
    data_root: &Path,
    plan_sha256: &str,
    matrix_sha256: &str,
    symbol: &str,
    base_timeframe: CanonicalTimeframe,
    cost_assumption_path: &Path,
    broker_symbol_contract_path: &Path,
    settings_source_path: &Path,
    receipt: neoethos_search::CanonicalSearchInputReceiptV2,
) -> Result<neoethos_search::CanonicalTrendbarResearchExecutionContractV3> {
    let plan_receipt = CanonicalTrendbarPlanReceiptV1::from_sha256(plan_sha256.to_owned())?;
    let matrix_receipt = CanonicalTrendbarMatrixReceiptV1::from_sha256(matrix_sha256.to_owned())?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority_root);
    let plan = store.open_plan(&plan_receipt)?;
    let matrix = store.open_matrix(data_root, &plan_receipt, &matrix_receipt)?;
    let series = ensure_unique_series(&matrix, symbol)?;
    validate_input_receipt_against_series(&receipt, series, base_timeframe)?;

    let cost_assumption_bytes = read_bounded_regular_file(cost_assumption_path)?;
    let costs: ScreeningCostEnvelopeWireV2 = serde_json::from_slice(&cost_assumption_bytes)
        .context("decode canonical screening-cost envelope V2")?;
    let settings_source_bytes = read_bounded_regular_file(settings_source_path)?;
    validate_settings_source(
        settings,
        settings_source_path,
        &settings_source_bytes,
        &costs,
    )?;
    let broker_symbol_contract_bytes = read_bounded_regular_file(broker_symbol_contract_path)?;
    let broker_cost_facts =
        validate_broker_symbol_contract(&broker_symbol_contract_bytes, &costs, symbol, &plan)?;
    let pip_value_per_lot = validate_costs(
        &costs,
        symbol,
        settings,
        &plan,
        &matrix,
        data_root,
        broker_cost_facts,
    )?;
    ensure!(
        read_bounded_regular_file(cost_assumption_path)? == cost_assumption_bytes,
        "cost assumptions changed while the CPU research contract was sealed"
    );
    ensure!(
        read_bounded_regular_file(broker_symbol_contract_path)? == broker_symbol_contract_bytes,
        "broker symbol contract changed while the CPU research contract was sealed"
    );

    let assumption_source_sha256 = format!("{:x}", Sha256::digest(&cost_assumption_bytes));
    let contract = neoethos_search::CanonicalTrendbarResearchExecutionContractV3::new(
        receipt.clone(),
        neoethos_search::CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: &costs.symbol,
            account_currency: &costs.account_currency,
            assumption_source_id: &costs.assumption_source_id,
            assumption_source_sha256: &assumption_source_sha256,
            pip_size: costs.pip_size,
            pip_value_per_lot,
            full_spread_pips_assumption: costs.full_spread_pips_assumption,
            slippage_pips_per_fill_assumption: costs.slippage_pips_per_fill_assumption,
            commission_account_per_lot_per_fill_assumption: costs
                .commission_account_per_lot_per_fill_assumption,
            swap_long_pips_per_day: costs.swap_long_pips_per_day,
            swap_short_pips_per_day: costs.swap_short_pips_per_day,
            pnl_conversion_fee_rate: costs.pnl_conversion_fee_rate,
        },
    )?;
    contract.validate_against_receipt(&receipt)?;
    Ok(contract)
}

pub(crate) fn ensure_cpu_research_output_target(
    out: &Path,
    protected_inputs: &[&Path],
) -> Result<()> {
    ensure_distinct_cost_output(out, protected_inputs)
}

#[cfg(feature = "gpu-nvidia-full")]
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CanonicalTrainingReceiptWireV1 {
    schema: String,
    version: u16,
    artifact_sha256: String,
}

#[cfg(feature = "gpu-nvidia-full")]
pub fn train_receipt_bound(args: &[String], settings: &neoethos_core::Settings) -> Result<()> {
    validate_canonical_train_args(args)?;
    ensure!(
        cfg!(feature = "gpu-nvidia-full"),
        "canonical-train requires the complete NVIDIA CUDA feature; rebuild neoethos-cli with --features gpu-nvidia-full"
    );
    let authority_root = required_path(args, "--authority-root")?;
    let data_root = required_path(args, "--data-root")?;
    let plan_sha256 = required(args, "--plan-sha256")?;
    let matrix_sha256 = required(args, "--matrix-sha256")?;
    let symbol = required(args, "--symbol")?;
    let base_timeframe = required(args, "--base-timeframe")?
        .parse::<CanonicalTimeframe>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let input_receipt_path = required_path(args, "--input-receipt")?;
    let cost_assumption_path = required_path(args, "--cost-assumptions")?;
    let broker_symbol_contract_path = required_path(args, "--broker-symbol-contract")?;
    let settings_source_path = required_path(args, "--settings-source")?;
    let models_dir = required_path(args, "--models-dir")?;
    let training_oos_from_ms = required(args, "--oos-from-ms")?
        .parse::<i64>()
        .context("--oos-from-ms must be one i64 Unix-millisecond timestamp")?;
    ensure!(
        training_oos_from_ms > 0,
        "--oos-from-ms must be a positive Unix-millisecond timestamp"
    );
    let out = required_path(args, "--out")?;
    let receipt_out = required_path(args, "--receipt-out")?;
    ensure_distinct_output_targets(
        &out,
        &receipt_out,
        &[
            &input_receipt_path,
            &cost_assumption_path,
            &broker_symbol_contract_path,
            &settings_source_path,
        ],
    )?;

    let plan_receipt = CanonicalTrendbarPlanReceiptV1::from_sha256(plan_sha256.clone())?;
    let matrix_receipt = CanonicalTrendbarMatrixReceiptV1::from_sha256(matrix_sha256.clone())?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority_root);
    let plan = store.open_plan(&plan_receipt)?;
    let matrix = store.open_matrix(&data_root, &plan_receipt, &matrix_receipt)?;
    let series = ensure_unique_series(&matrix, &symbol)?;
    series.validate()?;

    let selected_base = ensure_unique_selected_timeframe(series, base_timeframe)?;
    let exact_base = load_exact_canonical_timeframe(&data_root, selected_base)
        .context("open exact canonical base timeframe for training OOS boundary")?;
    let exact_training_rows =
        neoethos_search::canonical_discovery_normalization_training_rows(exact_base.ohlcv().len())?;
    let exact_training_oos_from_ms = exact_base
        .ohlcv()
        .timestamp
        .as_ref()
        .and_then(|timestamps| timestamps.get(exact_training_rows.end))
        .copied()
        .context("exact canonical base timeframe has no timestamp at the training OOS split")?;
    ensure!(
        training_oos_from_ms == exact_training_oos_from_ms,
        "--oos-from-ms does not equal the deterministic OOS boundary of the exact canonical base generation"
    );

    let input_receipt_bytes = read_bounded_regular_file(&input_receipt_path)?;
    let input_receipt =
        neoethos_search::CanonicalSearchInputReceiptV2::from_json_bytes(&input_receipt_bytes)
            .context("decode and validate exact canonical-search input receipt")?;
    validate_input_receipt_against_series(&input_receipt, series, base_timeframe)?;

    let cost_assumption_bytes = read_bounded_regular_file(&cost_assumption_path)?;
    let costs: ScreeningCostEnvelopeWireV2 = serde_json::from_slice(&cost_assumption_bytes)
        .context("decode canonical screening-cost envelope V2")?;
    let settings_source_bytes = read_bounded_regular_file(&settings_source_path)?;
    validate_settings_source(
        settings,
        &settings_source_path,
        &settings_source_bytes,
        &costs,
    )?;
    let broker_symbol_contract_bytes = read_bounded_regular_file(&broker_symbol_contract_path)?;
    let broker_cost_facts =
        validate_broker_symbol_contract(&broker_symbol_contract_bytes, &costs, &symbol, &plan)?;
    let pip_value_per_lot = validate_costs(
        &costs,
        &symbol,
        settings,
        &plan,
        &matrix,
        &data_root,
        broker_cost_facts,
    )?;
    let cost_assumption_file_sha256 = format!("{:x}", Sha256::digest(&cost_assumption_bytes));
    let contract = neoethos_search::CanonicalTrendbarResearchExecutionContractV3::new(
        input_receipt.clone(),
        neoethos_search::CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: &costs.symbol,
            account_currency: &costs.account_currency,
            assumption_source_id: &costs.assumption_source_id,
            assumption_source_sha256: &cost_assumption_file_sha256,
            pip_size: costs.pip_size,
            pip_value_per_lot,
            full_spread_pips_assumption: costs.full_spread_pips_assumption,
            slippage_pips_per_fill_assumption: costs.slippage_pips_per_fill_assumption,
            commission_account_per_lot_per_fill_assumption: costs
                .commission_account_per_lot_per_fill_assumption,
            swap_long_pips_per_day: costs.swap_long_pips_per_day,
            swap_short_pips_per_day: costs.swap_short_pips_per_day,
            pnl_conversion_fee_rate: costs.pnl_conversion_fee_rate,
        },
    )?;
    contract.validate_against_receipt(&input_receipt)?;
    let training_label_round_trip_cost_pips = contract.screening_round_trip_cost_pips();
    ensure!(
        training_label_round_trip_cost_pips.is_finite()
            && training_label_round_trip_cost_pips >= 0.0,
        "canonical training label screening costs are not finite and non-negative"
    );

    let orchestrator =
        neoethos_models::TrainingOrchestrator::new(settings.clone(), models_dir.clone())
            .with_data_root(&data_root)
            .with_oos_lock_from_ms(training_oos_from_ms);
    let preflight_planned_models = orchestrator.preflight_configured_nvidia_training()?;
    ensure!(
        !preflight_planned_models.is_empty(),
        "canonical receipt-bound training preflight produced an empty model plan"
    );
    let mut artifact = CanonicalTrainingArtifactWireV1 {
        schema: CANONICAL_TRAIN_SCHEMA_V1,
        version: 1,
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        authorization_issued: false,
        symbol: symbol.clone(),
        base_timeframe: base_timeframe.as_str().to_owned(),
        plan_sha256,
        matrix_sha256,
        canonical_series: series.clone(),
        input_receipt_sha256: input_receipt.identity_sha256()?,
        input_receipt_file_sha256: format!("{:x}", Sha256::digest(&input_receipt_bytes)),
        input_receipt_exact_utf8: String::from_utf8(input_receipt_bytes)
            .context("canonical-search input receipt is not UTF-8 JSON")?,
        research_contract_sha256: contract.identity_sha256()?,
        research_contract: contract.clone(),
        resolved_settings: settings.clone(),
        cost_assumption_file_sha256,
        cost_assumption_exact_utf8: String::from_utf8(cost_assumption_bytes)
            .context("cost-assumption evidence is not UTF-8 JSON")?,
        settings_source_file_sha256: format!("{:x}", Sha256::digest(&settings_source_bytes)),
        settings_source_exact_utf8: String::from_utf8(settings_source_bytes)
            .context("settings evidence is not UTF-8 YAML")?,
        broker_symbol_contract_file_sha256: format!(
            "{:x}",
            Sha256::digest(&broker_symbol_contract_bytes)
        ),
        broker_symbol_contract_exact_utf8: String::from_utf8(broker_symbol_contract_bytes)
            .context("broker symbol evidence is not UTF-8 JSON")?,
        cost_assumptions: costs,
        training_oos_from_ms,
        planned_models: preflight_planned_models.clone(),
        completed_models: Vec::new(),
        failed_models: Vec::new(),
        training_label_round_trip_cost_pips,
        model_artifacts: Vec::new(),
    };

    let installed = neoethos_core::execution_budget::installed_process_budget()
        .context("canonical training requires the installed process CPU budget")?;
    let lease =
        installed
            .broker()
            .acquire(neoethos_core::execution_budget::CpuPermitRequest::local(
                installed.resolved().effective_worker_limit,
            ))?;
    let training = match orchestrator.train_canonical_series_receipt_with_progress(
        series,
        base_timeframe,
        &input_receipt,
        &contract,
        &lease,
        |progress| tracing::info!(target: "neoethos_cli::canonical_train", ?progress),
    ) {
        Ok(training) => training,
        Err(error) => {
            artifact.failed_models.push(TrainingFailureWireV1 {
                name: "__training_pipeline__".to_owned(),
                error: error.to_string(),
            });
            let artifact_sha256 = publish_canonical_training_artifact(
                &out,
                &receipt_out,
                &models_dir,
                &symbol,
                base_timeframe,
                &artifact,
            )?;
            anyhow::bail!(
                "canonical receipt-bound training failed; exact evidence was written to {} with SHA-256 {}: {}",
                out.display(),
                artifact_sha256,
                error
            );
        }
    };
    ensure!(
        training.planned_models == preflight_planned_models,
        "canonical training plan drifted: preflight={:?}, execution={:?}",
        preflight_planned_models,
        training.planned_models
    );
    artifact.planned_models = training.planned_models;
    artifact.completed_models = training.completed_models;
    artifact.failed_models = training
        .failed_models
        .into_iter()
        .map(|failure| TrainingFailureWireV1 {
            name: failure.name,
            error: failure.error,
        })
        .collect();
    artifact.model_artifacts = model_artifact_evidence(
        &models_dir,
        &symbol,
        base_timeframe,
        &artifact.completed_models,
    )?;
    let artifact_sha256 = publish_canonical_training_artifact(
        &out,
        &receipt_out,
        &models_dir,
        &symbol,
        base_timeframe,
        &artifact,
    )?;
    ensure!(
        artifact.failed_models.is_empty(),
        "canonical receipt-bound training completed with {} failed model jobs; exact evidence was written to {}",
        artifact.failed_models.len(),
        out.display()
    );

    println!("canonical_training_status=complete");
    println!("artifact_class=ResearchOnly");
    println!("promotion_eligibility=NotPromotionEligible");
    println!("authorization_issued=false");
    println!("completed_model_count={}", artifact.completed_models.len());
    println!("artifact_sha256={artifact_sha256}");
    println!("evidence_path={}", out.display());
    println!("receipt_path={}", receipt_out.display());
    Ok(())
}

#[cfg(not(feature = "gpu-nvidia-full"))]
pub fn train_receipt_bound(args: &[String], _settings: &neoethos_core::Settings) -> Result<()> {
    validate_canonical_train_args(args)?;
    anyhow::bail!(
        "canonical-train requires the complete NVIDIA CUDA feature; rebuild neoethos-cli with --features gpu-nvidia-full"
    )
}

#[cfg(feature = "gpu-nvidia-full")]
fn publish_canonical_training_artifact(
    out: &Path,
    receipt_out: &Path,
    models_dir: &Path,
    symbol: &str,
    base_timeframe: CanonicalTimeframe,
    artifact: &CanonicalTrainingArtifactWireV1,
) -> Result<String> {
    validate_model_artifact_evidence_unchanged(
        models_dir,
        symbol,
        base_timeframe,
        &artifact.model_artifacts,
    )?;
    neoethos_core::storage::json::write_json_atomic(out, artifact)?;
    validate_model_artifact_evidence_unchanged(
        models_dir,
        symbol,
        base_timeframe,
        &artifact.model_artifacts,
    )?;
    let artifact_bytes = read_regular_file_with_limit(out, MAX_TRAINING_ARTIFACT_BYTES)?;
    let artifact_sha256 = format!("{:x}", Sha256::digest(&artifact_bytes));
    let receipt = CanonicalTrainingReceiptWireV1 {
        schema: CANONICAL_TRAIN_RECEIPT_SCHEMA_V1.to_owned(),
        version: 1,
        artifact_sha256: artifact_sha256.clone(),
    };
    neoethos_core::storage::json::write_json_atomic(receipt_out, &receipt)?;
    let reopened_receipt: CanonicalTrainingReceiptWireV1 =
        serde_json::from_slice(&read_bounded_regular_file(receipt_out)?)
            .context("reopen canonical training receipt")?;
    ensure!(
        reopened_receipt == receipt
            && format!("{:x}", Sha256::digest(&artifact_bytes)) == reopened_receipt.artifact_sha256,
        "canonical training receipt did not reopen against the exact artifact bytes"
    );
    Ok(artifact_sha256)
}

fn validate_input_receipt_against_series(
    receipt: &neoethos_search::CanonicalSearchInputReceiptV2,
    series: &CanonicalDatasetSeriesReceiptV1,
    base_timeframe: CanonicalTimeframe,
) -> Result<()> {
    series.validate()?;
    let anchor = receipt
        .validate()
        .context("validate canonical training receipt anchor")?;
    let selected_base = ensure_unique_selected_timeframe(series, base_timeframe)?;
    ensure!(
        &anchor == selected_base.identity(),
        "canonical training input receipt anchor does not match the exact selected base generation identity"
    );
    ensure!(
        series.anchor().identity().symbol_name() == anchor.symbol_name(),
        "canonical training matrix series symbol does not match the input receipt anchor"
    );

    let mut bound_timeframes = BTreeSet::new();
    for binding in receipt.source_bindings() {
        let identity = neoethos_data::CanonicalDatasetIdentity::from_path_component(
            binding.dataset_identity(),
        )
        .with_context(|| {
            format!(
                "decode canonical training source identity {}",
                binding.dataset_identity()
            )
        })?;
        let matches = series
            .direct_timeframes()
            .iter()
            .filter(|selected| selected.identity() == &identity)
            .collect::<Vec<_>>();
        ensure!(
            matches.len() == 1,
            "canonical training receipt source {} does not resolve to exactly one selected matrix generation",
            identity.to_path_component()
        );
        let selected = matches[0];
        ensure!(
            binding.generation_id() == selected.generation_id()
                && binding.manifest_sha256() == selected.manifest_binding_sha256()
                && binding.vortex_sha256() == generation_sha256(selected)?,
            "canonical training receipt source {} disagrees with its selected generation, manifest binding, or Vortex bytes",
            identity.to_path_component()
        );
        ensure!(
            bound_timeframes.insert(identity.timeframe()),
            "canonical training receipt repeats direct timeframe {}",
            identity.timeframe()
        );
    }
    ensure!(
        bound_timeframes.contains(&base_timeframe),
        "canonical training receipt does not bind its selected base timeframe {base_timeframe}"
    );
    Ok(())
}

#[cfg(feature = "gpu-nvidia")]
fn canonical_feature_options(
    settings: &neoethos_core::Settings,
    base_timeframe: CanonicalTimeframe,
) -> Result<FeatureBuildOptions> {
    let configured = settings
        .system
        .resolve_higher_timeframes(base_timeframe.as_str());
    let mut selected = BTreeSet::new();
    for value in &configured {
        let timeframe = value.parse::<CanonicalTimeframe>().with_context(|| {
            format!("configured feature timeframe {value} is not broker-canonical")
        })?;
        if timeframe != base_timeframe {
            selected.insert(timeframe);
        }
    }
    Ok(FeatureBuildOptions {
        higher_tfs: selected
            .into_iter()
            .map(|timeframe| timeframe.as_str().to_owned())
            .collect(),
        prefix_base_features: settings.system.multi_resolution_prefix_base,
        ..FeatureBuildOptions::default()
    })
}

#[cfg(feature = "gpu-nvidia-full")]
fn model_artifact_evidence(
    models_dir: &Path,
    symbol: &str,
    base_timeframe: CanonicalTimeframe,
    completed_models: &[String],
) -> Result<Vec<ModelArtifactEvidenceWireV1>> {
    let mut seen = BTreeSet::new();
    let mut evidence = Vec::with_capacity(completed_models.len());
    for model_name in completed_models {
        ensure_safe_path_component(model_name, "completed model name")?;
        ensure!(
            seen.insert(model_name.as_str()),
            "completed model inventory repeats {model_name}"
        );
        let relative_dir = PathBuf::from(symbol)
            .join(base_timeframe.as_str())
            .join(model_name);
        let artifact_dir = models_dir.join(&relative_dir);
        let (tree_sha256, file_count, total_bytes) = hash_model_artifact_tree(&artifact_dir)?;
        evidence.push(ModelArtifactEvidenceWireV1 {
            model_name: model_name.clone(),
            relative_dir: canonical_relative_tree_path(&relative_dir)?,
            tree_sha256,
            file_count,
            total_bytes,
        });
    }
    Ok(evidence)
}

#[cfg(feature = "gpu-nvidia-full")]
fn validate_model_artifact_evidence_unchanged(
    models_dir: &Path,
    symbol: &str,
    base_timeframe: CanonicalTimeframe,
    expected: &[ModelArtifactEvidenceWireV1],
) -> Result<()> {
    let completed_models = expected
        .iter()
        .map(|entry| entry.model_name.clone())
        .collect::<Vec<_>>();
    let reopened = model_artifact_evidence(models_dir, symbol, base_timeframe, &completed_models)?;
    ensure!(
        reopened == expected,
        "completed model artifacts changed while final evidence was published"
    );
    Ok(())
}

#[cfg(feature = "gpu-nvidia-full")]
fn hash_model_artifact_tree(root: &Path) -> Result<(String, u64, u64)> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("inspect completed model artifact {}", root.display()))?;
    ensure!(
        metadata.file_type().is_dir()
            && !metadata.file_type().is_symlink()
            && !metadata_is_reparse_point(&metadata),
        "completed model artifact root is not one physical directory"
    );

    let mut files = Vec::new();
    collect_model_artifact_files(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    ensure!(
        !files.is_empty(),
        "completed model artifact directory is empty"
    );

    let mut tree = Sha256::new();
    tree.update(b"neoethos.canonical-training-model-artifact-tree.v1\0");
    let mut total_bytes = 0_u64;
    for (relative, path, expected_len) in &files {
        let before = fs::symlink_metadata(path)
            .with_context(|| format!("inspect model artifact file {}", path.display()))?;
        ensure!(
            before.file_type().is_file()
                && !before.file_type().is_symlink()
                && !metadata_is_reparse_point(&before)
                && before.len() == *expected_len,
            "model artifact file identity changed before hashing"
        );
        let mut file = fs::File::open(path)
            .with_context(|| format!("open model artifact file {}", path.display()))?;
        let mut file_hash = Sha256::new();
        let mut observed_len = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .with_context(|| format!("hash model artifact file {}", path.display()))?;
            if read == 0 {
                break;
            }
            observed_len = observed_len
                .checked_add(read as u64)
                .context("model artifact file byte count overflow")?;
            file_hash.update(&buffer[..read]);
        }
        let after = fs::symlink_metadata(path)
            .with_context(|| format!("reinspect model artifact file {}", path.display()))?;
        ensure!(
            after.file_type().is_file()
                && !after.file_type().is_symlink()
                && !metadata_is_reparse_point(&after)
                && observed_len == *expected_len
                && after.len() == *expected_len,
            "model artifact file changed while hashing"
        );
        total_bytes = total_bytes
            .checked_add(observed_len)
            .context("model artifact tree byte count overflow")?;
        let relative_bytes = relative.as_bytes();
        tree.update((relative_bytes.len() as u64).to_le_bytes());
        tree.update(relative_bytes);
        tree.update(observed_len.to_le_bytes());
        tree.update(file_hash.finalize());
    }
    Ok((
        format!("{:x}", tree.finalize()),
        files.len() as u64,
        total_bytes,
    ))
}

#[cfg(feature = "gpu-nvidia-full")]
fn collect_model_artifact_files(
    root: &Path,
    current: &Path,
    files: &mut Vec<(String, PathBuf, u64)>,
) -> Result<()> {
    let current_metadata = fs::symlink_metadata(current)
        .with_context(|| format!("inspect model artifact path {}", current.display()))?;
    ensure!(
        current_metadata.file_type().is_dir()
            && !current_metadata.file_type().is_symlink()
            && !metadata_is_reparse_point(&current_metadata),
        "model artifact tree contains a non-physical directory"
    );
    let mut entries = fs::read_dir(current)
        .with_context(|| format!("read model artifact directory {}", current.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("inspect model artifact entry {}", path.display()))?;
        ensure!(
            !metadata.file_type().is_symlink() && !metadata_is_reparse_point(&metadata),
            "model artifact tree contains a symlink or reparse point"
        );
        if metadata.file_type().is_dir() {
            collect_model_artifact_files(root, &path, files)?;
        } else {
            ensure!(
                metadata.file_type().is_file(),
                "model artifact tree contains a non-file entry"
            );
            let relative = path
                .strip_prefix(root)
                .context("model artifact entry escaped its root")?;
            files.push((
                canonical_relative_tree_path(relative)?,
                path,
                metadata.len(),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "gpu-nvidia-full")]
fn canonical_relative_tree_path(path: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        let std::path::Component::Normal(value) = component else {
            anyhow::bail!("model artifact relative path is not canonical");
        };
        let value = value
            .to_str()
            .context("model artifact relative path is not UTF-8")?;
        ensure_safe_path_component(value, "model artifact path component")?;
        parts.push(value);
    }
    ensure!(!parts.is_empty(), "model artifact relative path is empty");
    Ok(parts.join("/"))
}

#[cfg(feature = "gpu-nvidia-full")]
fn ensure_safe_path_component(value: &str, label: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value != "."
            && value != ".."
            && !value.chars().any(char::is_control)
            && !value.contains('/')
            && !value.contains('\\'),
        "{label} is not one safe path component"
    );
    Ok(())
}

#[cfg(all(windows, feature = "gpu-nvidia-full"))]
fn metadata_is_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    metadata.file_attributes() & 0x400 != 0
}

#[cfg(all(not(windows), feature = "gpu-nvidia-full"))]
fn metadata_is_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

#[cfg(feature = "gpu-nvidia")]
fn ensure_distinct_output_targets(
    artifact_out: &Path,
    receipt_out: &Path,
    protected_inputs: &[&Path],
) -> Result<()> {
    let artifact_target = output_target_key(artifact_out)?;
    let receipt_target = output_target_key(receipt_out)?;
    ensure!(
        artifact_target != receipt_target,
        "artifact and receipt outputs resolve to the same target"
    );
    for input in protected_inputs {
        let input_target = output_target_key(input)?;
        ensure!(
            artifact_target != input_target && receipt_target != input_target,
            "canonical artifact output aliases one of its exact input files"
        );
    }
    Ok(())
}

fn output_target_key(path: &Path) -> Result<String> {
    let file_name = path
        .file_name()
        .context("canonical artifact path has no final component")?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let resolved_parent = fs::canonicalize(parent).with_context(|| {
        format!(
            "resolve canonical artifact path parent {}",
            parent.display()
        )
    })?;
    let target = resolved_parent
        .join(file_name)
        .to_string_lossy()
        .to_string();
    if cfg!(windows) {
        Ok(target.to_ascii_lowercase())
    } else {
        Ok(target)
    }
}

fn required(args: &[String], flag: &str) -> Result<String> {
    let values = args
        .windows(2)
        .filter(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .collect::<Vec<_>>();
    ensure!(values.len() == 1, "{flag} must be supplied exactly once");
    ensure!(!values[0].trim().is_empty(), "{flag} must not be empty");
    Ok(values[0].clone())
}

fn validate_cost_build_args(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == COST_BUILD_REQUIRED_FLAGS.len() * 2,
        "canonical-cost-build requires exactly {} flag/value pairs",
        COST_BUILD_REQUIRED_FLAGS.len()
    );
    let mut seen = BTreeSet::new();
    for pair in args.chunks_exact(2) {
        let flag = pair[0].as_str();
        let value = pair[1].as_str();
        ensure!(
            COST_BUILD_REQUIRED_FLAGS.contains(&flag),
            "canonical-cost-build received unknown argument {flag}"
        );
        ensure!(
            seen.insert(flag),
            "canonical-cost-build argument {flag} was supplied more than once"
        );
        ensure!(
            !value.trim().is_empty() && !value.starts_with("--"),
            "canonical-cost-build argument {flag} has no value"
        );
    }
    ensure!(
        seen.len() == COST_BUILD_REQUIRED_FLAGS.len(),
        "canonical-cost-build omitted a required argument"
    );
    Ok(())
}

fn validate_contract_build_args(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == CONTRACT_BUILD_REQUIRED_FLAGS.len() * 2,
        "canonical-contract-build requires exactly {} flag/value pairs",
        CONTRACT_BUILD_REQUIRED_FLAGS.len()
    );
    let mut seen = BTreeSet::new();
    for pair in args.chunks_exact(2) {
        let flag = pair[0].as_str();
        let value = pair[1].as_str();
        ensure!(
            CONTRACT_BUILD_REQUIRED_FLAGS.contains(&flag),
            "canonical-contract-build received unknown argument {flag}"
        );
        ensure!(
            seen.insert(flag),
            "canonical-contract-build argument {flag} was supplied more than once"
        );
        ensure!(
            !value.trim().is_empty() && !value.starts_with("--"),
            "canonical-contract-build argument {flag} has no value"
        );
    }
    ensure!(
        seen.len() == CONTRACT_BUILD_REQUIRED_FLAGS.len(),
        "canonical-contract-build omitted a required argument"
    );
    Ok(())
}

fn validate_canonical_train_args(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == CANONICAL_TRAIN_REQUIRED_FLAGS.len() * 2,
        "canonical-train requires exactly {} flag/value pairs",
        CANONICAL_TRAIN_REQUIRED_FLAGS.len()
    );
    let mut seen = BTreeSet::new();
    for pair in args.chunks_exact(2) {
        let flag = pair[0].as_str();
        let value = pair[1].as_str();
        ensure!(
            CANONICAL_TRAIN_REQUIRED_FLAGS.contains(&flag),
            "canonical-train received unknown argument {flag}"
        );
        ensure!(
            seen.insert(flag),
            "canonical-train argument {flag} was supplied more than once"
        );
        ensure!(
            !value.trim().is_empty() && !value.starts_with("--"),
            "canonical-train argument {flag} has no value"
        );
    }
    ensure!(
        seen.len() == CANONICAL_TRAIN_REQUIRED_FLAGS.len(),
        "canonical-train omitted a required argument"
    );
    Ok(())
}

fn required_path(args: &[String], flag: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(required(args, flag)?))
}

fn ensure_distinct_cost_output(out: &Path, protected_inputs: &[&Path]) -> Result<()> {
    let output_target = output_target_key(out)?;
    for input in protected_inputs {
        ensure!(
            output_target != output_target_key(input)?,
            "canonical cost output aliases one of its exact input files"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_cost_build_args() -> Vec<String> {
        COST_BUILD_REQUIRED_FLAGS
            .iter()
            .flat_map(|flag| [(*flag).to_owned(), "value".to_owned()])
            .collect()
    }

    fn exact_canonical_train_args() -> Vec<String> {
        CANONICAL_TRAIN_REQUIRED_FLAGS
            .iter()
            .flat_map(|flag| [(*flag).to_owned(), "value".to_owned()])
            .collect()
    }

    #[test]
    fn exact_cost_build_arguments_reject_unknown_duplicate_and_unpaired_inputs() {
        let args = exact_cost_build_args();
        validate_cost_build_args(&args).expect("screening cost-build argument set");

        let mut unknown = args.clone();
        unknown[0] = "--unknown".to_owned();
        assert!(validate_cost_build_args(&unknown).is_err());

        let mut duplicate = args.clone();
        duplicate[2] = duplicate[0].clone();
        assert!(validate_cost_build_args(&duplicate).is_err());

        let mut unpaired = args;
        unpaired.pop();
        assert!(validate_cost_build_args(&unpaired).is_err());
    }

    #[test]
    fn exact_canonical_train_arguments_reject_unknown_duplicate_and_unpaired_inputs() {
        let args = exact_canonical_train_args();
        validate_canonical_train_args(&args).expect("exact canonical-train argument set");

        let mut unknown = args.clone();
        unknown[0] = "--unknown".to_owned();
        assert!(validate_canonical_train_args(&unknown).is_err());

        let mut duplicate = args.clone();
        duplicate[2] = duplicate[0].clone();
        assert!(validate_canonical_train_args(&duplicate).is_err());

        let mut unpaired = args;
        unpaired.pop();
        assert!(validate_canonical_train_args(&unpaired).is_err());
    }
}
