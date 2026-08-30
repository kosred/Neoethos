//! One matrix-bound canonical-trendbar screening-research + training run.

use std::collections::BTreeSet;
use std::fs;
#[cfg(feature = "gpu-nvidia-full")]
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use neoethos_broker_history::{
    CanonicalTrendbarAcquisitionStoreV1, CanonicalTrendbarMatrixReceiptV1,
    CanonicalTrendbarMatrixV1, CanonicalTrendbarPlanReceiptV1,
};
use neoethos_data::{
    CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe, FeatureBuildOptions,
    SelectedDatasetGenerationV1, load_exact_canonical_timeframe,
};
#[cfg(feature = "gpu-nvidia-full")]
use neoethos_data::{pin_exact_canonical_series_v1, prepare_multitimeframe_features_with_options};
#[cfg(feature = "gpu-nvidia-full")]
use neoethos_search::historical_research::{
    HistoricalResearchArtifactClassV1, HistoricalResearchPromotionEligibilityV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCREENING_COST_SCHEMA_V2: &str = "neoethos.canonical-trendbar-screening-cost-envelope.v2";
#[cfg(feature = "gpu-nvidia-full")]
const FULL_RUN_SCHEMA_V1: &str = "neoethos.canonical-trendbar-full-run.v1";
#[cfg(feature = "gpu-nvidia-full")]
const FULL_RUN_RECEIPT_SCHEMA_V1: &str = "neoethos.canonical-trendbar-full-run-receipt.v1";
#[cfg(feature = "gpu-nvidia-full")]
const CANONICAL_TRAIN_SCHEMA_V1: &str = "neoethos.canonical-trendbar-training.v1";
#[cfg(feature = "gpu-nvidia-full")]
const CANONICAL_TRAIN_RECEIPT_SCHEMA_V1: &str = "neoethos.canonical-trendbar-training-receipt.v1";
const MAX_COST_ASSUMPTION_BYTES: u64 = 64 * 1024;
const MAX_CONTRACT_ARTIFACT_BYTES: u64 =
    neoethos_search::MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 as u64;
#[cfg(feature = "gpu-nvidia-full")]
const MAX_FULL_RUN_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const FULL_RUN_REQUIRED_FLAGS: [&str; 12] = [
    "--authority-root",
    "--data-root",
    "--plan-sha256",
    "--matrix-sha256",
    "--symbol",
    "--base-timeframe",
    "--cost-assumptions",
    "--broker-symbol-contract",
    "--settings-source",
    "--models-dir",
    "--out",
    "--receipt-out",
];
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ScreeningCostEnvelopeWireV2 {
    schema: String,
    version: u16,
    assumption_source_id: String,
    source_environment: String,
    source_server: String,
    source_account_id: i64,
    source_components: Vec<CostSourceComponentWireV1>,
    symbol: String,
    account_currency: String,
    pip_size: f64,
    pip_value_quote_per_lot: f64,
    pip_value_conversion: PipValueConversionWireV1,
    commission_symbol_price_basis: CommissionSymbolPriceBasisWireV1,
    full_spread_pips_assumption: f64,
    slippage_pips_per_fill_assumption: f64,
    commission_account_per_lot_per_fill_assumption: f64,
    swap_long_pips_per_day: f64,
    swap_short_pips_per_day: f64,
    pnl_conversion_fee_rate: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CostSourceComponentWireV1 {
    role: String,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PipValueConversionWireV1 {
    symbol: String,
    timeframe: String,
    operation: PipValueConversionOperationV1,
    timestamp_ms: i64,
    close: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CommissionSymbolPriceBasisWireV1 {
    symbol: String,
    timeframe: String,
    timestamp_ms: i64,
    close: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PipValueConversionOperationV1 {
    Identity,
    Multiply,
    Divide,
}

#[cfg(feature = "gpu-nvidia-full")]
#[derive(Debug, Serialize)]
struct CanonicalResearchDiscoveryArtifactV1 {
    schema: &'static str,
    version: u16,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    authorization_issued: bool,
    plan_sha256: String,
    matrix_sha256: String,
    research_contract_sha256: String,
    discovery_evidence_sha256: String,
    resolved_settings: neoethos_core::Settings,
    cost_assumption_exact_utf8: String,
    settings_source_exact_utf8: String,
    broker_symbol_contract_exact_utf8: String,
    cost_assumptions: ScreeningCostEnvelopeWireV2,
    research_contract: neoethos_search::CanonicalTrendbarResearchExecutionContractV3,
    discovery_result: neoethos_search::DiscoveryResult,
    training_oos_from_ms: i64,
    planned_models: Vec<String>,
    completed_models: Vec<String>,
    failed_models: Vec<TrainingFailureWireV1>,
    training_label_round_trip_cost_pips: f64,
    model_artifacts: Vec<ModelArtifactEvidenceWireV1>,
}

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

#[derive(Clone, Copy, Debug)]
struct BrokerSymbolCostFactsV1 {
    pip_position: i32,
    lot_size_cents: i64,
    commission_type: i64,
    precise_trading_commission_rate: i64,
}

#[derive(Clone, Copy, Debug)]
struct ExactBrokerSymbolCostInputsV1 {
    facts: BrokerSymbolCostFactsV1,
    swap_long_pips_per_day: f64,
    swap_short_pips_per_day: f64,
    pnl_conversion_fee_rate: f64,
}

#[derive(Clone, Debug)]
struct FinalDirectBasisV1 {
    symbol: String,
    timeframe: CanonicalTimeframe,
    timestamp_ms: i64,
    close: f64,
    generation_sha256: String,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalCostBuildOutcomeV1<'a> {
    schema: &'static str,
    version: u16,
    cost_assumption_sha256: &'a str,
    path: &'a Path,
}

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
    let search_input =
        neoethos_search::data_selection::CanonicalSearchInput::from_exact_series_receipt(
            &data_root,
            series,
            base_timeframe,
            &feature_options,
        )
        .context("build exact canonical search input for standalone research contract")?;
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
    let plan = store.open_plan(&plan_receipt)?;
    let matrix = store.open_matrix(&data_root, &plan_receipt, &matrix_receipt)?;
    ensure_unique_series(&matrix, &symbol)?;

    let settings_source_bytes = read_bounded_regular_file(&settings_source_path)?;
    let broker_symbol_contract_bytes = read_bounded_regular_file(&broker_symbol_contract_path)?;
    let broker =
        parse_exact_broker_symbol_cost_inputs(&broker_symbol_contract_bytes, &symbol, &plan)?;
    let commission_basis = load_final_direct_basis(&matrix, &data_root, &symbol, basis_timeframe)?;
    let account_currency = settings.system.account_currency.trim();
    ensure!(
        account_currency.len() == 3
            && account_currency
                .bytes()
                .all(|byte| byte.is_ascii_uppercase()),
        "settings account currency is not one canonical uppercase currency code"
    );
    let quote_currency = exact_forex_quote_currency(&symbol)?;
    let (pip_value_conversion, conversion_basis) = resolve_exact_conversion_basis(
        &matrix,
        &data_root,
        &symbol,
        basis_timeframe,
        quote_currency,
        account_currency,
    )?;

    let pip_size = 10.0_f64.powi(-broker.facts.pip_position);
    let pip_value_quote_per_lot = (broker.facts.lot_size_cents as f64 / 100.0) * pip_size;
    ensure!(
        pip_size.is_finite()
            && pip_size > 0.0
            && pip_value_quote_per_lot.is_finite()
            && pip_value_quote_per_lot > 0.0,
        "broker pip value basis is not finite and positive"
    );
    let commission_account_per_lot_per_fill_assumption =
        derive_commission_account_per_lot_per_fill_assumption(
            broker.facts,
            commission_basis.close,
            quote_currency,
            account_currency,
            &pip_value_conversion,
            conversion_basis.close,
        )?;
    ensure_no_session_spread_curve(settings)?;

    let full_spread_pips_assumption = require_non_negative_screening_assumption(
        "risk.backtest_spread_pips",
        settings.risk.backtest_spread_pips,
    )?;
    let slippage_pips_per_fill_assumption = require_non_negative_screening_assumption(
        "risk.slippage_pips",
        settings.risk.slippage_pips,
    )?;
    let costs = ScreeningCostEnvelopeWireV2 {
        schema: SCREENING_COST_SCHEMA_V2.to_owned(),
        version: 2,
        assumption_source_id: "neoethos.canonical-d1-screening-cost-assumptions.v2".to_owned(),
        source_environment: plan.environment().as_str().to_owned(),
        source_server: plan.server().to_owned(),
        source_account_id: plan.account_id(),
        source_components: vec![
            exact_source_component("broker_symbol_contract", &broker_symbol_contract_bytes),
            exact_source_component("settings", &settings_source_bytes),
            CostSourceComponentWireV1 {
                role: "pip_value_basis".to_owned(),
                sha256: conversion_basis.generation_sha256.clone(),
            },
            CostSourceComponentWireV1 {
                role: "commission_symbol_price_basis".to_owned(),
                sha256: commission_basis.generation_sha256.clone(),
            },
        ],
        symbol: symbol.clone(),
        account_currency: account_currency.to_owned(),
        pip_size,
        pip_value_quote_per_lot,
        pip_value_conversion,
        commission_symbol_price_basis: CommissionSymbolPriceBasisWireV1 {
            symbol: commission_basis.symbol.clone(),
            timeframe: commission_basis.timeframe.as_str().to_owned(),
            timestamp_ms: commission_basis.timestamp_ms,
            close: commission_basis.close,
        },
        full_spread_pips_assumption,
        slippage_pips_per_fill_assumption,
        commission_account_per_lot_per_fill_assumption,
        swap_long_pips_per_day: broker.swap_long_pips_per_day,
        swap_short_pips_per_day: broker.swap_short_pips_per_day,
        pnl_conversion_fee_rate: broker.pnl_conversion_fee_rate,
    };

    validate_settings_source(
        settings,
        &settings_source_path,
        &settings_source_bytes,
        &costs,
    )?;
    let validated_broker =
        validate_broker_symbol_contract(&broker_symbol_contract_bytes, &costs, &symbol, &plan)?;
    ensure!(
        validated_broker.pip_position == broker.facts.pip_position
            && validated_broker.lot_size_cents == broker.facts.lot_size_cents
            && validated_broker.commission_type == broker.facts.commission_type
            && validated_broker.precise_trading_commission_rate
                == broker.facts.precise_trading_commission_rate,
        "broker symbol cost inputs changed during cost construction"
    );
    validate_costs(
        &costs,
        &symbol,
        settings,
        &plan,
        &matrix,
        &data_root,
        validated_broker,
    )?;
    ensure!(
        read_bounded_regular_file(&broker_symbol_contract_path)? == broker_symbol_contract_bytes,
        "broker symbol contract changed while cost evidence was constructed"
    );

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

#[cfg(feature = "gpu-nvidia-full")]
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CanonicalFullRunReceiptV1 {
    schema: String,
    version: u16,
    artifact_sha256: String,
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
pub fn run(args: &[String], settings: &neoethos_core::Settings) -> Result<()> {
    validate_exact_args(args)?;
    ensure!(
        cfg!(feature = "gpu-nvidia-full"),
        "canonical-full-run requires the complete NVIDIA CUDA feature; rebuild neoethos-cli with --features gpu-nvidia-full"
    );
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
    let models_dir = required_path(args, "--models-dir")?;
    let out = required_path(args, "--out")?;
    let receipt_out = required_path(args, "--receipt-out")?;
    ensure_distinct_output_targets(
        &out,
        &receipt_out,
        &[
            &cost_assumption_path,
            &broker_symbol_contract_path,
            &settings_source_path,
        ],
    )?;
    let orchestrator =
        neoethos_models::TrainingOrchestrator::new(settings.clone(), models_dir.clone())
            .with_data_root(&data_root);
    let preflight_planned_models = orchestrator.preflight_full_nvidia_cuda_training()?;
    ensure!(
        !preflight_planned_models.is_empty(),
        "canonical full run training preflight produced an empty model plan"
    );

    let plan_receipt = CanonicalTrendbarPlanReceiptV1::from_sha256(plan_sha256.clone())?;
    let matrix_receipt = CanonicalTrendbarMatrixReceiptV1::from_sha256(matrix_sha256.clone())?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority_root);
    let plan = store.open_plan(&plan_receipt)?;
    let matrix = store.open_matrix(&data_root, &plan_receipt, &matrix_receipt)?;
    let series = ensure_unique_series(&matrix, &symbol)?;
    let pinned_series = pin_exact_canonical_series_v1(&data_root, series.clone())?;

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
    let pinned_series = std::cell::RefCell::new(Some(pinned_series));
    let prepared_input = neoethos_search::prepare_canonical_discovery_run_input_v3(
        |no_physical_gpu_admission| {
            let pinned_series = pinned_series
                .borrow_mut()
                .take()
                .context("canonical full-run pin was already consumed")?;
            let dataset = pinned_series
                .into_cpu_dataset_after_no_physical_gpu_v1(&no_physical_gpu_admission)?;
            let base_frame = dataset.canonical_frame(base_timeframe.as_str())?;
            let normalization_training_rows =
                neoethos_search::canonical_discovery_normalization_training_rows(
                    base_frame.ohlcv().len(),
                )?;
            let mut feature_options = canonical_feature_options(settings, base_timeframe)?;
            feature_options.normalization_training_rows = Some(normalization_training_rows);
            let features = prepare_multitimeframe_features_with_options(
                &dataset,
                base_timeframe.as_str(),
                &feature_options,
            )?;
            let search_input = neoethos_search::data_selection::CanonicalSearchInput::from_prepared_canonical_frame(
                base_frame.artifact().identity().clone(),
                base_frame,
                features,
            )?;
            Ok((search_input, no_physical_gpu_admission))
        },
        || {
            anyhow::bail!(
                "canonical full run cannot seal the complete native Discovery workspace yet; refusing host feature materialization on a physical GPU"
            )
        },
        |_admitted_native_run| {
            let _pinned_series = pinned_series
                .borrow_mut()
                .take()
                .context("canonical full-run pin was already consumed")?;
            anyhow::bail!(
                "canonical full run native Data materialization is unreachable before workspace sealing"
            )
        },
    )?;

    let assumption_source_sha256 = format!("{:x}", Sha256::digest(&cost_assumption_bytes));
    let contract = neoethos_search::CanonicalTrendbarResearchExecutionContractV3::new(
        prepared_input
            .cpu_receipt_v2()
            .context("canonical full run requires a CPU receipt until the native Discovery workspace is sealed")?
            .clone(),
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
    let training_label_round_trip_cost_pips = contract.screening_round_trip_cost_pips();
    ensure!(
        training_label_round_trip_cost_pips.is_finite()
            && training_label_round_trip_cost_pips >= 0.0,
        "canonical training label screening costs are not finite and non-negative"
    );
    let config =
        neoethos_search::DiscoveryConfig::try_from_settings_for_canonical_trendbar_research(
            settings, &contract,
        )?;
    let prepared_research =
        neoethos_search::run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3(
            prepared_input,
            &config,
            &contract,
            neoethos_search::PropFirmRiskRules::default(),
            |progress| tracing::info!(target: "neoethos_cli::canonical_full_run", ?progress),
        )?;
    let (research, training_input) = prepared_research.into_parts();
    research.validate()?;
    neoethos_search::ensure_non_empty_portfolio(
        research.discovery_result(),
        &format!("canonical research {symbol} {base_timeframe}"),
    )?;

    let result = research.discovery_result();
    let holdout_scope = result
        .holdout_scope
        .as_ref()
        .context("canonical full run requires a holdout scope before model training")?;
    let training_oos_from_ms = holdout_scope.evaluated_window().timestamp_start_ms();
    let mut artifact = CanonicalResearchDiscoveryArtifactV1 {
        schema: FULL_RUN_SCHEMA_V1,
        version: 1,
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        authorization_issued: false,
        plan_sha256,
        matrix_sha256,
        research_contract_sha256: contract.identity_sha256()?,
        discovery_evidence_sha256: research.evidence_identity_sha256().to_owned(),
        resolved_settings: settings.clone(),
        cost_assumption_exact_utf8: String::from_utf8(cost_assumption_bytes.clone())
            .context("cost-assumption evidence is not UTF-8 JSON")?,
        settings_source_exact_utf8: String::from_utf8(settings_source_bytes)
            .context("settings evidence is not UTF-8 YAML")?,
        broker_symbol_contract_exact_utf8: String::from_utf8(broker_symbol_contract_bytes)
            .context("broker symbol evidence is not UTF-8 JSON")?,
        cost_assumptions: costs.clone(),
        research_contract: research.execution_contract().clone(),
        discovery_result: result.clone(),
        training_oos_from_ms,
        planned_models: preflight_planned_models.clone(),
        completed_models: Vec::new(),
        failed_models: Vec::new(),
        training_label_round_trip_cost_pips,
        model_artifacts: Vec::new(),
    };
    drop(research);

    let installed = neoethos_core::execution_budget::installed_process_budget()
        .context("canonical full run requires the installed process CPU budget")?;
    let lease =
        installed
            .broker()
            .acquire(neoethos_core::execution_budget::CpuPermitRequest::local(
                installed.resolved().effective_worker_limit,
            ))?;
    let orchestrator = orchestrator.with_oos_lock_from_ms(training_oos_from_ms);
    let training = match orchestrator.train_canonical_series_with_progress(
        series,
        base_timeframe,
        training_input,
        &contract,
        &lease,
        |progress| tracing::info!(target: "neoethos_cli::canonical_full_run", ?progress),
    ) {
        Ok(training) => training,
        Err(error) => {
            artifact.failed_models.push(TrainingFailureWireV1 {
                name: "__training_pipeline__".to_owned(),
                error: error.to_string(),
            });
            let artifact_sha256 = publish_full_run_artifact(
                &out,
                &receipt_out,
                &models_dir,
                &symbol,
                base_timeframe,
                &artifact,
            )?;
            anyhow::bail!(
                "canonical full run completed search but the training pipeline failed; exact evidence was written to {} with SHA-256 {}: {}",
                out.display(),
                artifact_sha256,
                error
            );
        }
    };
    ensure!(
        training.planned_models == preflight_planned_models,
        "canonical training plan drifted after search: preflight={:?}, execution={:?}",
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
    let artifact_sha256 = publish_full_run_artifact(
        &out,
        &receipt_out,
        &models_dir,
        &symbol,
        base_timeframe,
        &artifact,
    )?;
    ensure!(
        artifact.failed_models.is_empty(),
        "canonical full run completed search but {} model training jobs failed; exact evidence was written to {}",
        artifact.failed_models.len(),
        out.display()
    );

    println!("canonical_full_run_status=complete");
    println!("artifact_class=ResearchOnly");
    println!("promotion_eligibility=NotPromotionEligible");
    println!("authorization_issued=false");
    println!(
        "portfolio_count={}",
        artifact.discovery_result.portfolio.len()
    );
    println!("completed_model_count={}", artifact.completed_models.len());
    println!("artifact_sha256={artifact_sha256}");
    println!("evidence_path={}", out.display());
    println!("receipt_path={}", receipt_out.display());
    Ok(())
}

#[cfg(not(feature = "gpu-nvidia-full"))]
pub fn run(args: &[String], _settings: &neoethos_core::Settings) -> Result<()> {
    validate_exact_args(args)?;
    anyhow::bail!(
        "canonical-full-run requires the complete NVIDIA CUDA feature; rebuild neoethos-cli with --features gpu-nvidia-full"
    )
}

#[cfg(feature = "gpu-nvidia-full")]
fn publish_full_run_artifact(
    out: &Path,
    receipt_out: &Path,
    models_dir: &Path,
    symbol: &str,
    base_timeframe: CanonicalTimeframe,
    artifact: &CanonicalResearchDiscoveryArtifactV1,
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
    let artifact_bytes = read_regular_file_with_limit(out, MAX_FULL_RUN_ARTIFACT_BYTES)?;
    let artifact_sha256 = format!("{:x}", Sha256::digest(&artifact_bytes));
    let receipt = CanonicalFullRunReceiptV1 {
        schema: FULL_RUN_RECEIPT_SCHEMA_V1.to_owned(),
        version: 1,
        artifact_sha256: artifact_sha256.clone(),
    };
    neoethos_core::storage::json::write_json_atomic(receipt_out, &receipt)?;
    let reopened_receipt: CanonicalFullRunReceiptV1 =
        serde_json::from_slice(&read_bounded_regular_file(receipt_out)?)
            .context("reopen canonical full-run receipt")?;
    ensure!(
        reopened_receipt == receipt
            && format!("{:x}", Sha256::digest(&artifact_bytes)) == reopened_receipt.artifact_sha256,
        "canonical full-run receipt did not reopen against the exact artifact bytes"
    );
    Ok(artifact_sha256)
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
    let artifact_bytes = read_regular_file_with_limit(out, MAX_FULL_RUN_ARTIFACT_BYTES)?;
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

fn ensure_unique_series<'a>(
    matrix: &'a CanonicalTrendbarMatrixV1,
    symbol: &str,
) -> Result<&'a CanonicalDatasetSeriesReceiptV1> {
    let matches = matrix
        .series()
        .iter()
        .filter(|series| series.anchor().identity().symbol_name() == symbol)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "matrix must contain exactly one canonical series for {symbol}; found {}",
        matches.len()
    );
    Ok(matches[0])
}

fn ensure_unique_selected_timeframe(
    series: &CanonicalDatasetSeriesReceiptV1,
    timeframe: CanonicalTimeframe,
) -> Result<&SelectedDatasetGenerationV1> {
    let matches = series
        .direct_timeframes()
        .iter()
        .filter(|selected| selected.identity().timeframe() == timeframe)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "canonical series must contain exactly one direct {timeframe} generation; found {}",
        matches.len()
    );
    Ok(matches[0])
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

fn load_final_direct_basis(
    matrix: &CanonicalTrendbarMatrixV1,
    data_root: &Path,
    symbol: &str,
    timeframe: CanonicalTimeframe,
) -> Result<FinalDirectBasisV1> {
    let series = ensure_unique_series(matrix, symbol)?;
    let selected = ensure_unique_selected_timeframe(series, timeframe)?;
    let frame = load_exact_canonical_timeframe(data_root, selected).with_context(|| {
        format!("open exact final direct canonical {symbol} {timeframe} generation")
    })?;
    let timestamp_ms = frame
        .ohlcv()
        .timestamp
        .as_ref()
        .and_then(|values| values.last().copied())
        .with_context(|| format!("direct canonical {symbol} {timeframe} has no final timestamp"))?;
    let close = frame
        .ohlcv()
        .close
        .last()
        .copied()
        .with_context(|| format!("direct canonical {symbol} {timeframe} has no final close"))?;
    ensure!(
        close.is_finite() && close > 0.0,
        "direct canonical {symbol} {timeframe} final close is not finite and positive"
    );
    Ok(FinalDirectBasisV1 {
        symbol: symbol.to_owned(),
        timeframe,
        timestamp_ms,
        close,
        generation_sha256: generation_sha256(selected)?.to_owned(),
    })
}

fn resolve_exact_conversion_basis(
    matrix: &CanonicalTrendbarMatrixV1,
    data_root: &Path,
    selected_symbol: &str,
    timeframe: CanonicalTimeframe,
    source_currency: &str,
    account_currency: &str,
) -> Result<(PipValueConversionWireV1, FinalDirectBasisV1)> {
    let account_currency = account_currency.trim();
    if source_currency == account_currency {
        let basis = load_final_direct_basis(matrix, data_root, selected_symbol, timeframe)?;
        return Ok((
            PipValueConversionWireV1 {
                symbol: basis.symbol.clone(),
                timeframe: basis.timeframe.as_str().to_owned(),
                operation: PipValueConversionOperationV1::Identity,
                timestamp_ms: basis.timestamp_ms,
                close: basis.close,
            },
            basis,
        ));
    }

    let direct = format!("{source_currency}{account_currency}");
    let inverse = format!("{account_currency}{source_currency}");
    let matches = matrix
        .series()
        .iter()
        .filter_map(|series| {
            let symbol = series.anchor().identity().symbol_name();
            if symbol == direct {
                Some((symbol, PipValueConversionOperationV1::Multiply))
            } else if symbol == inverse {
                Some((symbol, PipValueConversionOperationV1::Divide))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "matrix must contain exactly one direct conversion series from {source_currency} to {account_currency}"
    );
    let (basis_symbol, operation) = matches[0];
    let basis = load_final_direct_basis(matrix, data_root, basis_symbol, timeframe)?;
    let conversion = PipValueConversionWireV1 {
        symbol: basis.symbol.clone(),
        timeframe: basis.timeframe.as_str().to_owned(),
        operation,
        timestamp_ms: basis.timestamp_ms,
        close: basis.close,
    };
    validate_conversion_route(source_currency, account_currency, &conversion)?;
    Ok((conversion, basis))
}

fn exact_source_component(role: &str, exact_bytes: &[u8]) -> CostSourceComponentWireV1 {
    CostSourceComponentWireV1 {
        role: role.to_owned(),
        sha256: format!("{:x}", Sha256::digest(exact_bytes)),
    }
}

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

fn validate_costs(
    costs: &ScreeningCostEnvelopeWireV2,
    symbol: &str,
    settings: &neoethos_core::Settings,
    plan: &neoethos_broker_history::CanonicalTrendbarAcquisitionPlanV1,
    matrix: &CanonicalTrendbarMatrixV1,
    data_root: &Path,
    broker_cost_facts: BrokerSymbolCostFactsV1,
) -> Result<f64> {
    ensure!(
        costs.schema == SCREENING_COST_SCHEMA_V2 && costs.version == 2,
        "unsupported canonical screening-cost envelope schema/version"
    );
    ensure!(costs.symbol == symbol, "cost assumptions symbol mismatch");
    ensure!(
        costs.source_environment == plan.environment().as_str(),
        "cost assumptions broker environment does not match the acquisition plan"
    );
    ensure!(
        costs.source_server == plan.server(),
        "cost assumptions broker server does not match the acquisition plan"
    );
    ensure!(
        costs.source_account_id == plan.account_id(),
        "cost assumptions broker account does not match the acquisition plan"
    );
    ensure!(
        !costs.assumption_source_id.trim().is_empty()
            && costs.assumption_source_id.len() <= 255
            && !costs.assumption_source_id.chars().any(char::is_control),
        "cost assumptions source id is not one bounded identity"
    );
    ensure!(
        costs.assumption_source_id == "neoethos.canonical-d1-screening-cost-assumptions.v2",
        "unsupported canonical screening-cost assumption source"
    );
    let mut roles = BTreeSet::new();
    for component in &costs.source_components {
        ensure!(
            !component.role.trim().is_empty()
                && component.role.len() <= 64
                && component
                    .role
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
            "cost source component role is not canonical"
        );
        ensure!(
            roles.insert(component.role.as_str()),
            "cost source component role {} is duplicated",
            component.role
        );
        ensure_canonical_sha256(&component.sha256, &component.role)?;
    }
    for required_role in [
        "broker_symbol_contract",
        "settings",
        "pip_value_basis",
        "commission_symbol_price_basis",
    ] {
        ensure!(
            roles.contains(required_role),
            "cost assumptions omit required source component {required_role}"
        );
    }
    ensure!(
        roles.len() == 4,
        "cost assumptions contain unsupported source-component roles"
    );
    ensure!(
        settings.system.symbol.trim() == symbol,
        "config symbol {} does not match selected matrix symbol {symbol}",
        settings.system.symbol
    );
    ensure!(
        costs.account_currency == settings.system.account_currency.trim(),
        "cost assumptions account currency does not match config"
    );
    ensure_no_session_spread_curve(settings)?;
    let expected_full_spread = require_non_negative_screening_assumption(
        "risk.backtest_spread_pips",
        settings.risk.backtest_spread_pips,
    )?;
    ensure!(
        costs.full_spread_pips_assumption.to_bits() == expected_full_spread.to_bits(),
        "screening full-spread assumption does not match config backtest spread"
    );
    let expected_slippage_per_fill = require_non_negative_screening_assumption(
        "risk.slippage_pips",
        settings.risk.slippage_pips,
    )?;
    ensure!(
        costs.slippage_pips_per_fill_assumption.to_bits() == expected_slippage_per_fill.to_bits(),
        "screening per-fill slippage assumption does not match config"
    );
    let configured_commission_per_lot = require_non_negative_screening_assumption(
        "risk.commission_per_lot",
        settings.risk.commission_per_lot,
    )?;
    let configured_commission_per_fill = configured_commission_per_lot
        * if settings.risk.commission_per_lot_is_per_side {
            1.0
        } else {
            0.5
        };
    let pip_size = 10.0_f64.powi(-broker_cost_facts.pip_position);
    let expected_pip_value_quote_per_lot =
        (broker_cost_facts.lot_size_cents as f64 / 100.0) * pip_size;
    ensure!(
        expected_pip_value_quote_per_lot.is_finite()
            && expected_pip_value_quote_per_lot > 0.0
            && costs.pip_value_quote_per_lot.to_bits()
                == expected_pip_value_quote_per_lot.to_bits(),
        "quote-currency pip value does not match broker lotSize and pipPosition"
    );
    ensure!(
        costs.pip_value_conversion.close.is_finite() && costs.pip_value_conversion.close > 0.0,
        "pip-value conversion close must be finite and positive"
    );
    let conversion_timeframe = costs
        .pip_value_conversion
        .timeframe
        .parse::<CanonicalTimeframe>()
        .context("pip-value conversion timeframe is not broker-canonical")?;
    let last_close = load_exact_basis_close(
        matrix,
        data_root,
        &costs.pip_value_conversion.symbol,
        conversion_timeframe,
        costs.pip_value_conversion.timestamp_ms,
        costs.pip_value_conversion.close,
        costs,
        "pip_value_basis",
    )?;
    let quote_currency = exact_forex_quote_currency(symbol)?;
    validate_conversion_route(
        quote_currency,
        &costs.account_currency,
        &costs.pip_value_conversion,
    )?;
    let pip_value_per_lot = apply_conversion(
        costs.pip_value_quote_per_lot,
        costs.pip_value_conversion.operation,
        last_close,
    );
    ensure!(
        pip_value_per_lot.is_finite() && pip_value_per_lot > 0.0,
        "account-currency pip value is not finite and positive"
    );

    ensure!(
        costs.commission_symbol_price_basis.symbol == symbol,
        "commission price basis symbol does not match the selected symbol"
    );
    let commission_timeframe = costs
        .commission_symbol_price_basis
        .timeframe
        .parse::<CanonicalTimeframe>()
        .context("commission price-basis timeframe is not broker-canonical")?;
    let symbol_price = load_exact_basis_close(
        matrix,
        data_root,
        &costs.commission_symbol_price_basis.symbol,
        commission_timeframe,
        costs.commission_symbol_price_basis.timestamp_ms,
        costs.commission_symbol_price_basis.close,
        costs,
        "commission_symbol_price_basis",
    )?;
    let expected_commission_per_fill = derive_commission_account_per_lot_per_fill_assumption(
        broker_cost_facts,
        symbol_price,
        quote_currency,
        &costs.account_currency,
        &costs.pip_value_conversion,
        last_close,
    )?;
    ensure!(
        costs
            .commission_account_per_lot_per_fill_assumption
            .to_bits()
            == expected_commission_per_fill.to_bits(),
        "screening per-fill commission assumption does not match the broker rate and canonical D1 bases"
    );
    if configured_commission_per_fill.to_bits() != expected_commission_per_fill.to_bits() {
        tracing::warn!(
            target: "neoethos_cli::canonical_full_run",
            configured_commission_account_per_lot_per_fill = configured_commission_per_fill,
            screening_commission_account_per_lot_per_fill_assumption = expected_commission_per_fill,
            "canonical screening uses the receipt-bound D1 commission assumption instead of the config commission"
        );
    }
    Ok(pip_value_per_lot)
}

fn require_non_negative_screening_assumption(label: &str, value: f64) -> Result<f64> {
    ensure!(
        value.is_finite() && value >= 0.0,
        "{label} must be finite and non-negative"
    );
    Ok(value)
}

fn ensure_no_session_spread_curve(settings: &neoethos_core::Settings) -> Result<()> {
    ensure!(
        settings.risk.backtest_spread_pips_asian.is_none()
            && settings.risk.backtest_spread_pips_overlap.is_none()
            && settings.risk.backtest_spread_pips_late_ny.is_none(),
        "canonical research scalar spread cannot represent a configured session spread curve"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_exact_basis_close(
    matrix: &CanonicalTrendbarMatrixV1,
    data_root: &Path,
    symbol: &str,
    timeframe: CanonicalTimeframe,
    expected_timestamp_ms: i64,
    expected_close: f64,
    costs: &ScreeningCostEnvelopeWireV2,
    source_role: &str,
) -> Result<f64> {
    ensure!(
        expected_close.is_finite() && expected_close > 0.0,
        "{source_role} close must be finite and positive"
    );
    let series = ensure_unique_series(matrix, symbol)?;
    let selected = ensure_unique_selected_timeframe(series, timeframe)?;
    let basis_sha256 = costs
        .source_components
        .iter()
        .find(|component| component.role == source_role)
        .map(|component| component.sha256.as_str())
        .with_context(|| format!("{source_role} source component disappeared after validation"))?;
    ensure!(
        generation_sha256(selected)? == basis_sha256,
        "{source_role} SHA-256 does not match its selected direct generation"
    );
    let frame = load_exact_canonical_timeframe(data_root, selected)
        .with_context(|| format!("open exact direct canonical {source_role} generation"))?;
    let timestamps = frame
        .ohlcv()
        .timestamp
        .as_ref()
        .with_context(|| format!("{source_role} generation has no timestamps"))?;
    let last_timestamp = timestamps
        .last()
        .copied()
        .with_context(|| format!("{source_role} generation is empty"))?;
    let last_close = frame
        .ohlcv()
        .close
        .last()
        .copied()
        .with_context(|| format!("{source_role} generation has no close"))?;
    ensure!(
        expected_timestamp_ms == last_timestamp && expected_close.to_bits() == last_close.to_bits(),
        "{source_role} must bind the exact final direct canonical close"
    );
    Ok(last_close)
}

fn exact_forex_quote_currency(symbol: &str) -> Result<&str> {
    ensure!(
        symbol.len() == 6 && symbol.bytes().all(|byte| byte.is_ascii_uppercase()),
        "canonical scalar research currently requires one six-letter uppercase FX symbol"
    );
    Ok(&symbol[3..])
}

fn validate_conversion_route(
    source_currency: &str,
    account_currency: &str,
    conversion: &PipValueConversionWireV1,
) -> Result<()> {
    let account_currency = account_currency.trim();
    ensure!(
        source_currency.len() == 3
            && account_currency.len() == 3
            && account_currency
                .bytes()
                .all(|byte| byte.is_ascii_uppercase()),
        "conversion currencies are not canonical ISO-style codes"
    );
    if source_currency == account_currency {
        ensure!(
            matches!(
                conversion.operation,
                PipValueConversionOperationV1::Identity
            ),
            "same-currency pip/commission conversion must use identity"
        );
        return Ok(());
    }

    let direct = format!("{source_currency}{account_currency}");
    let inverse = format!("{account_currency}{source_currency}");
    let valid = (conversion.symbol == direct
        && matches!(
            conversion.operation,
            PipValueConversionOperationV1::Multiply
        ))
        || (conversion.symbol == inverse
            && matches!(conversion.operation, PipValueConversionOperationV1::Divide));
    ensure!(
        valid,
        "conversion symbol/operation does not map {source_currency} into {account_currency}"
    );
    Ok(())
}

fn apply_conversion(
    amount: f64,
    operation: PipValueConversionOperationV1,
    conversion_close: f64,
) -> f64 {
    match operation {
        PipValueConversionOperationV1::Identity => amount,
        PipValueConversionOperationV1::Multiply => amount * conversion_close,
        PipValueConversionOperationV1::Divide => amount / conversion_close,
    }
}

fn derive_commission_account_per_lot_per_fill_assumption(
    broker: BrokerSymbolCostFactsV1,
    symbol_close: f64,
    quote_currency: &str,
    account_currency: &str,
    conversion: &PipValueConversionWireV1,
    conversion_close: f64,
) -> Result<f64> {
    let rate_divisor = if broker.commission_type == 3 {
        1.0e5
    } else {
        1.0e8
    };
    let rate = broker.precise_trading_commission_rate as f64 / rate_divisor;
    ensure!(
        rate.is_finite() && rate >= 0.0,
        "broker precise commission rate is not finite and non-negative"
    );
    let contract_units = broker.lot_size_cents as f64 / 100.0;
    let (one_side, commission_currency) = match broker.commission_type {
        1 => {
            ensure!(
                quote_currency == "USD",
                "USD-per-million commission requires an explicit quote-to-USD basis for non-USD quotes"
            );
            (rate * (contract_units * symbol_close) / 1_000_000.0, "USD")
        }
        2 => (rate, "USD"),
        3 => (
            (rate / 100.0) * contract_units * symbol_close,
            quote_currency,
        ),
        4 => (rate, quote_currency),
        value => anyhow::bail!("unsupported broker commissionType {value}"),
    };
    validate_conversion_route(commission_currency, account_currency, conversion)?;
    let one_side_account = apply_conversion(one_side, conversion.operation, conversion_close);
    ensure!(
        one_side_account.is_finite() && one_side_account >= 0.0,
        "derived per-fill commission assumption is not finite and non-negative"
    );
    Ok(one_side_account)
}

fn validate_settings_source(
    settings: &neoethos_core::Settings,
    supplied_path: &Path,
    exact_bytes: &[u8],
    costs: &ScreeningCostEnvelopeWireV2,
) -> Result<()> {
    ensure!(
        settings.provenance().source() == neoethos_core::config::ConfigSource::EnvConfigFile,
        "canonical full run requires Settings loaded through explicit CONFIG_FILE"
    );
    let loaded_path = settings
        .provenance()
        .path()
        .context("explicit CONFIG_FILE settings provenance has no path")?;
    let loaded_path = fs::canonicalize(loaded_path)
        .with_context(|| format!("resolve loaded config path {}", loaded_path.display()))?;
    let supplied_path = fs::canonicalize(supplied_path)
        .with_context(|| format!("resolve supplied config path {}", supplied_path.display()))?;
    ensure!(
        loaded_path == supplied_path,
        "--settings-source is not the exact file that produced Settings"
    );
    validate_source_component_bytes(costs, "settings", exact_bytes)?;
    let reloaded = neoethos_core::Settings::from_yaml(&supplied_path)
        .context("reload the exact settings source through the sealed config parser")?;
    let after_reload = read_bounded_regular_file(&supplied_path)?;
    ensure!(
        exact_bytes == after_reload,
        "settings source changed while its exact resolved values were validated"
    );
    ensure!(
        serde_json::to_value(settings)? == serde_json::to_value(&reloaded)?,
        "settings source does not resolve to the exact Settings used by this process"
    );
    Ok(())
}

fn validate_broker_symbol_contract(
    exact_bytes: &[u8],
    costs: &ScreeningCostEnvelopeWireV2,
    symbol: &str,
    plan: &neoethos_broker_history::CanonicalTrendbarAcquisitionPlanV1,
) -> Result<BrokerSymbolCostFactsV1> {
    validate_source_component_bytes(costs, "broker_symbol_contract", exact_bytes)?;
    let inputs = parse_exact_broker_symbol_cost_inputs(exact_bytes, symbol, plan)?;
    let expected_pip_size = 10.0_f64.powi(-inputs.facts.pip_position);
    ensure!(
        costs.pip_size.to_bits() == expected_pip_size.to_bits(),
        "cost pip size does not match broker pipPosition"
    );
    ensure!(
        costs.swap_long_pips_per_day.to_bits() == inputs.swap_long_pips_per_day.to_bits()
            && costs.swap_short_pips_per_day.to_bits() == inputs.swap_short_pips_per_day.to_bits(),
        "cost swap values do not match the exact broker symbol contract"
    );
    ensure!(
        costs.pnl_conversion_fee_rate.to_bits() == inputs.pnl_conversion_fee_rate.to_bits(),
        "cost PnL conversion fee does not match the exact broker symbol contract"
    );
    Ok(inputs.facts)
}

fn parse_exact_broker_symbol_cost_inputs(
    exact_bytes: &[u8],
    symbol: &str,
    plan: &neoethos_broker_history::CanonicalTrendbarAcquisitionPlanV1,
) -> Result<ExactBrokerSymbolCostInputsV1> {
    let document: serde_json::Value =
        serde_json::from_slice(exact_bytes).context("decode exact broker symbol contract")?;
    ensure!(
        document.get("payloadType").and_then(|value| value.as_i64()) == Some(2117),
        "broker symbol contract is not ProtoOASymbolByIdRes payloadType 2117"
    );
    let payload = document
        .get("payload")
        .and_then(|value| value.as_object())
        .context("broker symbol contract has no payload object")?;
    ensure!(
        payload
            .get("ctidTraderAccountId")
            .and_then(|value| value.as_i64())
            == Some(plan.account_id()),
        "broker symbol contract account does not match the acquisition plan"
    );
    let planned = plan
        .symbols()
        .iter()
        .filter(|candidate| candidate.symbol_name() == symbol)
        .collect::<Vec<_>>();
    ensure!(
        planned.len() == 1,
        "acquisition plan must contain exactly one symbol identity for {symbol}"
    );
    let broker_symbols = payload
        .get("symbol")
        .and_then(|value| value.as_array())
        .context("broker symbol contract has no symbol array")?;
    ensure!(
        broker_symbols.len() == 1,
        "broker symbol contract must contain exactly one full symbol"
    );
    let broker_symbol = broker_symbols[0]
        .as_object()
        .context("broker symbol contract entry is not an object")?;
    ensure!(
        broker_symbol
            .get("symbolId")
            .and_then(|value| value.as_i64())
            == Some(planned[0].symbol_id()),
        "broker symbol contract symbol id does not match the acquisition plan"
    );
    let pip_position = broker_symbol
        .get("pipPosition")
        .and_then(|value| value.as_i64())
        .context("broker symbol contract omits pipPosition")?;
    ensure!(
        (0..=15).contains(&pip_position),
        "broker symbol pipPosition is outside the supported exact range"
    );
    let lot_size_cents = broker_symbol
        .get("lotSize")
        .and_then(|value| value.as_i64())
        .context("broker symbol contract omits lotSize")?;
    ensure!(
        lot_size_cents > 0 && lot_size_cents <= 9_000_000_000_000_000,
        "broker lotSize is outside the exact positive f64 integer range"
    );
    let commission_type = broker_symbol
        .get("commissionType")
        .and_then(|value| value.as_i64())
        .context("broker symbol contract omits commissionType")?;
    ensure!(
        (1..=4).contains(&commission_type),
        "broker commissionType is outside the supported enum range"
    );
    let precise_trading_commission_rate = broker_symbol
        .get("preciseTradingCommissionRate")
        .and_then(|value| value.as_i64())
        .context("broker symbol contract omits preciseTradingCommissionRate")?;
    ensure!(
        precise_trading_commission_rate >= 0,
        "broker preciseTradingCommissionRate is negative"
    );
    let precise_min_commission = broker_symbol
        .get("preciseMinCommission")
        .and_then(|value| value.as_i64())
        .unwrap_or(0);
    let legacy_min_commission = broker_symbol
        .get("minCommission")
        .and_then(|value| value.as_i64())
        .unwrap_or(0);
    ensure!(
        precise_min_commission == 0 && legacy_min_commission == 0,
        "canonical scalar research does not yet model a non-zero broker minimum commission"
    );
    ensure!(
        broker_symbol
            .get("swapCalculationType")
            .and_then(|value| value.as_i64())
            .unwrap_or(0)
            == 0,
        "broker symbol swap is not denominated in pips"
    );
    let swap_long = broker_symbol
        .get("swapLong")
        .and_then(|value| value.as_f64())
        .context("broker symbol contract omits swapLong")?;
    let swap_short = broker_symbol
        .get("swapShort")
        .and_then(|value| value.as_f64())
        .context("broker symbol contract omits swapShort")?;
    ensure!(
        swap_long.is_finite() && swap_short.is_finite(),
        "broker symbol swap values are not finite"
    );
    let raw_pnl_fee = broker_symbol
        .get("pnlConversionFeeRate")
        .and_then(|value| value.as_i64())
        .context("broker symbol contract omits pnlConversionFeeRate")?;
    ensure!(
        (0..10_000).contains(&raw_pnl_fee),
        "broker PnL conversion fee is outside the supported exact range"
    );
    let pnl_fee = raw_pnl_fee as f64 / 10_000.0;
    Ok(ExactBrokerSymbolCostInputsV1 {
        facts: BrokerSymbolCostFactsV1 {
            pip_position: pip_position as i32,
            lot_size_cents,
            commission_type,
            precise_trading_commission_rate,
        },
        swap_long_pips_per_day: swap_long,
        swap_short_pips_per_day: swap_short,
        pnl_conversion_fee_rate: pnl_fee,
    })
}

fn validate_source_component_bytes(
    costs: &ScreeningCostEnvelopeWireV2,
    role: &str,
    exact_bytes: &[u8],
) -> Result<()> {
    let matching = costs
        .source_components
        .iter()
        .filter(|component| component.role == role)
        .collect::<Vec<_>>();
    ensure!(
        matching.len() == 1,
        "cost assumptions must contain exactly one {role} source component"
    );
    let actual = format!("{:x}", Sha256::digest(exact_bytes));
    ensure!(
        matching[0].sha256 == actual,
        "exact {role} bytes do not match their declared SHA-256"
    );
    Ok(())
}

fn generation_sha256(selected: &SelectedDatasetGenerationV1) -> Result<&str> {
    selected
        .generation_id()
        .strip_prefix("g1-")
        .and_then(|value| value.strip_suffix(".vortex"))
        .filter(|value| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        .context("selected direct generation id is not canonical g1 SHA-256 Vortex")
}

fn ensure_canonical_sha256(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "cost source component {label} is not canonical lowercase SHA-256"
    );
    Ok(())
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

fn read_bounded_regular_file(path: &Path) -> Result<Vec<u8>> {
    read_regular_file_with_limit(path, MAX_COST_ASSUMPTION_BYTES)
}

fn read_regular_file_with_limit(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect exact input {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "input is not a regular file"
    );
    ensure!(
        !metadata.file_type().is_symlink(),
        "input must not be a symlink"
    );
    ensure!(
        metadata.len() <= max_bytes,
        "input exceeds its exact byte bound"
    );
    fs::read(path).with_context(|| format!("read exact input {}", path.display()))
}

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
            "full-run output aliases one of its exact input files"
        );
    }
    Ok(())
}

fn output_target_key(path: &Path) -> Result<String> {
    let file_name = path
        .file_name()
        .context("full-run file path has no final component")?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let resolved_parent = fs::canonicalize(parent)
        .with_context(|| format!("resolve full-run path parent {}", parent.display()))?;
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

fn validate_exact_args(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == FULL_RUN_REQUIRED_FLAGS.len() * 2,
        "canonical-full-run requires exactly {} flag/value pairs",
        FULL_RUN_REQUIRED_FLAGS.len()
    );
    let mut seen = BTreeSet::new();
    for pair in args.chunks_exact(2) {
        let flag = pair[0].as_str();
        let value = pair[1].as_str();
        ensure!(
            FULL_RUN_REQUIRED_FLAGS.contains(&flag),
            "canonical-full-run received unknown argument {flag}"
        );
        ensure!(
            seen.insert(flag),
            "canonical-full-run argument {flag} was supplied more than once"
        );
        ensure!(
            !value.trim().is_empty() && !value.starts_with("--"),
            "canonical-full-run argument {flag} has no value"
        );
    }
    ensure!(
        seen.len() == FULL_RUN_REQUIRED_FLAGS.len(),
        "canonical-full-run omitted a required argument"
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

    fn exact_args() -> Vec<String> {
        FULL_RUN_REQUIRED_FLAGS
            .iter()
            .flat_map(|flag| [(*flag).to_owned(), "value".to_owned()])
            .collect()
    }

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
    fn exact_full_run_arguments_reject_unknown_duplicate_and_unpaired_inputs() {
        let args = exact_args();
        validate_exact_args(&args).expect("exact argument set");

        let mut unknown = args.clone();
        unknown[0] = "--unknown".to_owned();
        assert!(validate_exact_args(&unknown).is_err());

        let mut duplicate = args.clone();
        duplicate[2] = duplicate[0].clone();
        assert!(validate_exact_args(&duplicate).is_err());

        let mut unpaired = args;
        unpaired.pop();
        assert!(validate_exact_args(&unpaired).is_err());
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

    #[test]
    fn screening_cost_envelope_v2_rejects_legacy_v1_wire() {
        let legacy = serde_json::json!({
            "schema": "neoethos.canonical-trendbar-research-cost-assumptions.v1",
            "version": 1,
            "spread_pips": 2.0,
            "round_trip_commission_per_trade": 14.0
        });
        assert!(serde_json::from_value::<ScreeningCostEnvelopeWireV2>(legacy).is_err());
    }

    #[test]
    fn canonical_screening_costs_refuse_any_session_spread_curve_without_a_quote_gate() {
        let mut settings = neoethos_core::Settings::default();
        ensure_no_session_spread_curve(&settings).expect("flat scalar spread");
        settings.risk.backtest_spread_pips_asian = Some(1.0);
        assert!(ensure_no_session_spread_curve(&settings).is_err());
    }

    fn inverse_usd_to_gbp() -> PipValueConversionWireV1 {
        PipValueConversionWireV1 {
            symbol: "GBPUSD".to_owned(),
            timeframe: "D1".to_owned(),
            operation: PipValueConversionOperationV1::Divide,
            timestamp_ms: 0,
            close: 1.25,
        }
    }

    #[test]
    fn screening_per_fill_commission_uses_broker_rate_notional_and_account_conversion() {
        let broker = BrokerSymbolCostFactsV1 {
            pip_position: 4,
            lot_size_cents: 10_000_000,
            commission_type: 1,
            precise_trading_commission_rate: 4_500_000_000,
        };
        let conversion = inverse_usd_to_gbp();
        let actual = derive_commission_account_per_lot_per_fill_assumption(
            broker,
            1.2,
            "USD",
            "GBP",
            &conversion,
            conversion.close,
        )
        .expect("screening per-fill commission assumption");
        let expected: f64 = (45.0 * (100_000.0 * 1.2) / 1_000_000.0) / 1.25;
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn conversion_route_refuses_wrong_direction_or_operator() {
        let mut conversion = inverse_usd_to_gbp();
        validate_conversion_route("USD", "GBP", &conversion).expect("GBPUSD divide");
        conversion.operation = PipValueConversionOperationV1::Multiply;
        assert!(validate_conversion_route("USD", "GBP", &conversion).is_err());
    }
}
