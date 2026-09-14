//! Shared exact-source screening costs for the desktop and headless research producers.
//!
//! Moved from the CLI: all calculations and source/identity checks have one owner.
//! The values remain declared screening assumptions, not historical execution
//! evidence or an authorization for trading or promotion.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use neoethos_core::research_conversion_fee::{
    ResearchPnlConversionFeePolicyV1, effective_conversion_fee_rate_v1,
};
use neoethos_data::{
    CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe, SelectedDatasetGenerationV1,
    load_exact_canonical_timeframe,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::CanonicalTrendbarMatrixV1;

pub const SCREENING_COST_SCHEMA_V2: &str = "neoethos.canonical-trendbar-screening-cost-envelope.v2";
const MAX_COST_ASSUMPTION_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScreeningCostEnvelopeWireV2 {
    pub schema: String,
    pub version: u16,
    pub assumption_source_id: String,
    pub source_environment: String,
    pub source_server: String,
    pub source_account_id: i64,
    pub source_components: Vec<CostSourceComponentWireV1>,
    pub symbol: String,
    pub account_currency: String,
    pub pip_size: f64,
    pub pip_value_quote_per_lot: f64,
    pub pip_value_conversion: PipValueConversionWireV1,
    pub commission_symbol_price_basis: CommissionSymbolPriceBasisWireV1,
    pub full_spread_pips_assumption: f64,
    pub slippage_pips_per_fill_assumption: f64,
    pub commission_account_per_lot_per_fill_assumption: f64,
    pub swap_long_pips_per_day: f64,
    pub swap_short_pips_per_day: f64,
    pub pnl_conversion_fee_rate: f64,
    /// Mandatory nested policy V1: legacy V2 wire without this identity is refused.
    pub pnl_conversion_fee_policy: ResearchPnlConversionFeePolicyV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CostSourceComponentWireV1 {
    pub role: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PipValueConversionWireV1 {
    pub symbol: String,
    pub timeframe: String,
    pub operation: PipValueConversionOperationV1,
    pub timestamp_ms: i64,
    pub close: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CommissionSymbolPriceBasisWireV1 {
    pub symbol: String,
    pub timeframe: String,
    pub timestamp_ms: i64,
    pub close: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PipValueConversionOperationV1 {
    Identity,
    Multiply,
    Divide,
}

#[derive(Clone, Copy, Debug)]
pub struct BrokerSymbolCostFactsV1 {
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

/// Build declared screening assumptions from one exact plan/matrix and file-bound Settings.
///
/// Both frontends must use this producer instead of duplicating the financial
/// calculations. No output is published here and no execution or promotion
/// authority is produced. Current broker fields and final D1 prices remain
/// screening assumptions, not historical cost-policy evidence.
#[allow(clippy::too_many_arguments)]
pub fn build_screening_cost_envelope_v2(
    settings: &neoethos_core::Settings,
    data_root: &Path,
    acquisition_store: &crate::CanonicalTrendbarAcquisitionStoreV1,
    plan_receipt: &crate::CanonicalTrendbarPlanReceiptV1,
    matrix_receipt: &crate::CanonicalTrendbarMatrixReceiptV1,
    symbol: &str,
    basis_timeframe: CanonicalTimeframe,
    broker_symbol_contract_path: &Path,
    settings_source_path: &Path,
) -> Result<ScreeningCostEnvelopeWireV2> {
    ensure!(
        basis_timeframe == CanonicalTimeframe::D1,
        "canonical cost evidence requires the explicit direct D1 basis"
    );
    // Opening through the acquisition store proves the plan/matrix binding;
    // a caller cannot substitute a matrix from another acquisition.
    let plan = &acquisition_store.open_plan(plan_receipt)?;
    let matrix = &acquisition_store.open_matrix(data_root, plan_receipt, matrix_receipt)?;
    ensure_unique_series(matrix, symbol)?;

    let settings_source_bytes = read_bounded_regular_file(settings_source_path)?;
    let broker_symbol_contract_bytes = read_bounded_regular_file(broker_symbol_contract_path)?;
    let broker =
        parse_exact_broker_symbol_cost_inputs(&broker_symbol_contract_bytes, symbol, plan)?;
    let commission_basis = load_final_direct_basis(matrix, data_root, symbol, basis_timeframe)?;
    let account_currency = settings.system.account_currency.trim();
    ensure!(
        account_currency.len() == 3
            && account_currency
                .bytes()
                .all(|byte| byte.is_ascii_uppercase()),
        "settings account currency is not one canonical uppercase currency code"
    );
    let quote_currency = exact_forex_quote_currency(symbol)?;
    let (pip_value_conversion, conversion_basis) = resolve_exact_conversion_basis(
        matrix,
        data_root,
        symbol,
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
        symbol: symbol.to_owned(),
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
        pnl_conversion_fee_rate: effective_conversion_fee_rate_v1(
            broker.pnl_conversion_fee_rate,
            quote_currency,
            account_currency,
        )
        .map_err(anyhow::Error::msg)?,
        pnl_conversion_fee_policy:
            ResearchPnlConversionFeePolicyV1::AbsoluteRealizedPriceGrossDebitV1,
    };

    validate_settings_source(
        settings,
        settings_source_path,
        &settings_source_bytes,
        &costs,
    )?;
    let validated_broker =
        validate_broker_symbol_contract(&broker_symbol_contract_bytes, &costs, symbol, plan)?;
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
        symbol,
        settings,
        plan,
        matrix,
        data_root,
        validated_broker,
    )?;
    ensure!(
        read_bounded_regular_file(broker_symbol_contract_path)? == broker_symbol_contract_bytes,
        "broker symbol contract changed while cost evidence was constructed"
    );

    Ok(costs)
}

pub fn ensure_unique_series<'a>(
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

pub fn ensure_unique_selected_timeframe(
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

pub fn validate_costs(
    costs: &ScreeningCostEnvelopeWireV2,
    symbol: &str,
    settings: &neoethos_core::Settings,
    plan: &crate::CanonicalTrendbarAcquisitionPlanV1,
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
    // The explicit matrix/broker selection owns the run symbol. The file may
    // legitimately retain a different UI default; its complete original values
    // and bytes are still checked by validate_settings_source above the builder.
    ensure_unique_series(matrix, symbol)?;
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
            target: "neoethos_cli::canonical_evidence",
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

pub fn validate_settings_source(
    settings: &neoethos_core::Settings,
    supplied_path: &Path,
    exact_bytes: &[u8],
    costs: &ScreeningCostEnvelopeWireV2,
) -> Result<()> {
    validate_source_component_bytes(costs, "settings", exact_bytes)?;
    validate_exact_file_settings(settings, supplied_path, exact_bytes)
}

/// Check only the exact file-to-Settings binding. This is not cost admission:
/// consumers must still validate their broker inputs and screening envelope.
/// The desktop uses the same check while taking ownership of its original
/// source, before applying the separately selected symbol/timeframes.
pub fn validate_exact_file_settings(
    settings: &neoethos_core::Settings,
    supplied_path: &Path,
    exact_bytes: &[u8],
) -> Result<()> {
    ensure!(
        matches!(
            settings.provenance().source(),
            neoethos_core::config::ConfigSource::EnvConfigFile
                | neoethos_core::config::ConfigSource::ExplicitPath
        ),
        "canonical research requires file-bound Settings loaded through CONFIG_FILE or an explicit path"
    );
    let loaded_path = settings
        .provenance()
        .path()
        .context("file-bound Settings provenance has no path")?;
    let loaded_path = fs::canonicalize(loaded_path)
        .with_context(|| format!("resolve loaded config path {}", loaded_path.display()))?;
    let supplied_path = fs::canonicalize(supplied_path)
        .with_context(|| format!("resolve supplied config path {}", supplied_path.display()))?;
    ensure!(
        loaded_path == supplied_path,
        "supplied settings source is not the exact file that produced Settings"
    );
    // Parse the owned bytes that were hashed, not a second file read. Otherwise
    // a file that changes and changes back can bind one document to the values
    // parsed from another. Settings' Deserialize implementation is the same
    // sealed parser used by its explicit-path loader.
    let decoded: neoethos_core::Settings = serde_yaml_ng::from_slice(exact_bytes)
        .context("decode the captured Settings bytes through the sealed config parser")?;
    let after_reload = read_bounded_regular_file(&supplied_path)?;
    ensure!(
        exact_bytes == after_reload,
        "settings source changed while its exact resolved values were validated"
    );
    ensure!(
        serde_json::to_value(settings)? == serde_json::to_value(&decoded)?,
        "settings source does not resolve to the exact Settings used by this process"
    );
    Ok(())
}

pub fn validate_broker_symbol_contract(
    exact_bytes: &[u8],
    costs: &ScreeningCostEnvelopeWireV2,
    symbol: &str,
    plan: &crate::CanonicalTrendbarAcquisitionPlanV1,
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
    let effective_fee = effective_conversion_fee_rate_v1(
        inputs.pnl_conversion_fee_rate,
        exact_forex_quote_currency(symbol)?,
        &costs.account_currency,
    )
    .map_err(anyhow::Error::msg)?;
    ensure!(
        costs.pnl_conversion_fee_rate.to_bits() == effective_fee.to_bits(),
        "cost PnL conversion fee does not match the broker rate and quote/account applicability"
    );
    Ok(inputs.facts)
}

fn parse_exact_broker_symbol_cost_inputs(
    exact_bytes: &[u8],
    symbol: &str,
    plan: &crate::CanonicalTrendbarAcquisitionPlanV1,
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

pub fn generation_sha256(selected: &SelectedDatasetGenerationV1) -> Result<&str> {
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

pub fn read_bounded_regular_file(path: &Path) -> Result<Vec<u8>> {
    read_regular_file_with_limit(path, MAX_COST_ASSUMPTION_BYTES)
}

pub fn read_regular_file_with_limit(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
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
    let file =
        fs::File::open(path).with_context(|| format!("open exact input {}", path.display()))?;
    ensure!(
        file.metadata()?.is_file(),
        "opened input is not a regular file"
    );
    read_bounded_contents(file, max_bytes)
        .with_context(|| format!("read exact input {}", path.display()))
}

fn read_bounded_contents(reader: impl Read, max_bytes: u64) -> Result<Vec<u8>> {
    let limit = max_bytes
        .checked_add(1)
        .context("input byte bound must leave room for overflow detection")?;
    let mut bytes = Vec::new();
    reader.take(limit).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= max_bytes,
        "input exceeds its exact byte bound"
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_reader_stops_if_contents_grow_past_the_metadata_limit() {
        let mut growing_input = std::io::Cursor::new(vec![b'x'; 1024]);
        let error = read_bounded_contents(&mut growing_input, 4)
            .expect_err("a prior metadata check must not allow an unbounded read");
        assert!(error.to_string().contains("exact byte bound"));
        assert_eq!(growing_input.position(), 5);
        assert_eq!(read_bounded_contents(&b"abcd"[..], 4).unwrap(), b"abcd");
        assert!(read_bounded_contents(&b""[..], 0).unwrap().is_empty());
    }

    #[test]
    fn bounded_file_reader_refuses_oversize_and_non_regular_sources() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("source");
        fs::write(&path, b"abcd").unwrap();
        assert_eq!(read_regular_file_with_limit(&path, 4).unwrap(), b"abcd");
        assert!(read_regular_file_with_limit(&path, 3).is_err());
        assert!(read_regular_file_with_limit(directory.path(), 4).is_err());
    }

    fn settings_identity_fixture(exact_bytes: &[u8]) -> ScreeningCostEnvelopeWireV2 {
        ScreeningCostEnvelopeWireV2 {
            schema: SCREENING_COST_SCHEMA_V2.to_owned(),
            version: 2,
            assumption_source_id: "neoethos.canonical-d1-screening-cost-assumptions.v2".to_owned(),
            source_environment: "demo".to_owned(),
            source_server: "demo.ctraderapi.com".to_owned(),
            source_account_id: 1,
            // This fixture exercises only the Settings identity validator. It
            // is intentionally not sufficient for full cost-source admission.
            source_components: vec![exact_source_component("settings", exact_bytes)],
            symbol: "EURUSD".to_owned(),
            account_currency: "USD".to_owned(),
            pip_size: 0.0001,
            pip_value_quote_per_lot: 10.0,
            pip_value_conversion: PipValueConversionWireV1 {
                symbol: "EURUSD".to_owned(),
                timeframe: "D1".to_owned(),
                operation: PipValueConversionOperationV1::Identity,
                timestamp_ms: 1,
                close: 1.0,
            },
            commission_symbol_price_basis: CommissionSymbolPriceBasisWireV1 {
                symbol: "EURUSD".to_owned(),
                timeframe: "D1".to_owned(),
                timestamp_ms: 1,
                close: 1.0,
            },
            full_spread_pips_assumption: 1.5,
            slippage_pips_per_fill_assumption: 0.5,
            commission_account_per_lot_per_fill_assumption: 7.0,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            pnl_conversion_fee_policy:
                ResearchPnlConversionFeePolicyV1::AbsoluteRealizedPriceGrossDebitV1,
        }
    }

    #[test]
    fn explicit_desktop_settings_keep_exact_file_and_value_binding() {
        let directory = tempfile::tempdir().expect("temporary explicit Settings source");
        let path = directory.path().join("settings.yaml");
        let bytes = b"system:\n  symbol: EURUSD\n";
        fs::write(&path, bytes).expect("write test-owned Settings source");
        let settings = neoethos_core::Settings::from_yaml(&path).expect("desktop Settings loader");
        assert_eq!(
            settings.provenance().source(),
            neoethos_core::config::ConfigSource::ExplicitPath
        );
        validate_settings_source(&settings, &path, bytes, &settings_identity_fixture(bytes))
            .expect("exact explicit desktop Settings are a valid file-bound research source");
    }

    #[test]
    fn settings_identity_rejects_defaults_and_in_memory_values() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.yaml");
        let bytes = b"system:\n  symbol: EURUSD\n";
        fs::write(&path, bytes).unwrap();
        let explicit = neoethos_core::Settings::from_yaml(&path).unwrap();
        let in_memory: neoethos_core::Settings =
            serde_json::from_value(serde_json::to_value(&explicit).unwrap()).unwrap();
        for settings in [neoethos_core::Settings::default(), in_memory] {
            let error = validate_settings_source(
                &settings,
                &path,
                bytes,
                &settings_identity_fixture(bytes),
            )
            .expect_err("values alone cannot prove a file-bound settings source");
            assert!(error.to_string().contains("file-bound Settings"));
        }
    }

    #[test]
    fn settings_identity_rejects_another_file_even_with_identical_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let loaded_path = directory.path().join("loaded.yaml");
        let supplied_path = directory.path().join("other.yaml");
        let bytes = b"system:\n  symbol: EURUSD\n";
        fs::write(&loaded_path, bytes).unwrap();
        fs::write(&supplied_path, bytes).unwrap();
        let settings = neoethos_core::Settings::from_yaml(&loaded_path).unwrap();
        let error = validate_settings_source(
            &settings,
            &supplied_path,
            bytes,
            &settings_identity_fixture(bytes),
        )
        .expect_err("identical values in another file do not establish source identity");
        assert!(error.to_string().contains("not the exact file"));
    }

    #[test]
    fn settings_identity_rejects_post_load_value_changes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.yaml");
        let bytes = b"system:\n  symbol: EURUSD\n";
        fs::write(&path, bytes).unwrap();
        let mut settings = neoethos_core::Settings::from_yaml(&path).unwrap();
        settings.system.symbol = "GBPUSD".to_owned();
        let error =
            validate_settings_source(&settings, &path, bytes, &settings_identity_fixture(bytes))
                .expect_err("file provenance must not authorize edited in-memory values");
        assert!(
            error
                .to_string()
                .contains("exact Settings used by this process")
        );
    }

    #[test]
    fn settings_identity_rejects_file_changes_with_old_or_updated_hashes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.yaml");
        let original = b"system:\n  symbol: EURUSD\n";
        let changed = b"system:\n  symbol: GBPUSD\n";
        fs::write(&path, original).unwrap();
        let settings = neoethos_core::Settings::from_yaml(&path).unwrap();
        fs::write(&path, changed).unwrap();
        let stale_error = validate_settings_source(
            &settings,
            &path,
            original,
            &settings_identity_fixture(original),
        )
        .expect_err("a changed source cannot retain its earlier identity");
        assert!(stale_error.to_string().contains("settings source changed"));
        let updated_error = validate_settings_source(
            &settings,
            &path,
            changed,
            &settings_identity_fixture(changed),
        )
        .expect_err("rehashing a replacement file cannot authorize the original resolved settings");
        assert!(
            updated_error
                .to_string()
                .contains("exact Settings used by this process")
        );
    }

    #[test]
    fn settings_identity_requires_one_matching_source_hash() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.yaml");
        let bytes = b"system:\n  symbol: EURUSD\n";
        fs::write(&path, bytes).unwrap();
        let settings = neoethos_core::Settings::from_yaml(&path).unwrap();
        let mut missing = settings_identity_fixture(bytes);
        missing.source_components.clear();
        let mut duplicate = settings_identity_fixture(bytes);
        duplicate
            .source_components
            .push(exact_source_component("settings", bytes));
        let mut wrong_hash = settings_identity_fixture(bytes);
        wrong_hash.source_components[0].sha256 = "0".repeat(64);
        for costs in [missing, duplicate, wrong_hash] {
            validate_settings_source(&settings, &path, bytes, &costs).expect_err(
                "settings identity requires exactly one source component with the exact hash",
            );
        }
    }

    #[test]
    fn captured_settings_bytes_go_through_the_sealed_parser() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.yaml");
        fs::write(&path, b"system:\n  symbol: EURUSD\n").unwrap();
        let settings = neoethos_core::Settings::from_yaml(&path).unwrap();
        let invalid_captured_bytes =
            b"system:\n  symbol: EURUSD\n  definitely_unknown_cost_source_key: 1\n";
        let error = validate_settings_source(
            &settings,
            &path,
            invalid_captured_bytes,
            &settings_identity_fixture(invalid_captured_bytes),
        )
        .expect_err("captured bytes must not be replaced by parsing the current file");
        assert!(error.to_string().contains("captured Settings bytes"));
        assert!(format!("{error:#}").contains("definitely_unknown_cost_source_key"));
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
