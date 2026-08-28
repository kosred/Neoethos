//! GET /indicators?datasetIdentity=&expectedGeneration=&indicator=&period=&limit=
//!
//! Compute a single technical indicator from one exact, fully verified and
//! pinned local Vortex generation. The canonical identity determines symbol,
//! source/account and timeframe; the generation receipt prevents silent drift
//! after inventory selection. This route never contacts the broker.
//!
//! Wire shape — single-output indicator:
//! ```json
//! { "symbol":"EURUSD","timeframe":"M1","indicator":"sma","period":20,
//!   "candleCount":200,
//!   "lines":[{"name":"sma","values":[1.0823,1.0824,…]}] }
//! ```
//! Multi-output (Bollinger Bands, MACD, Stochastic) decomposes into
//! several entries in `lines`, named `<id>_line0`, `<id>_line1`, etc.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{CanonicalDatasetIdentity, IndicatorLine, compute_single_indicator};

use super::chart::ExactDatasetReceipt;
use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

/// Top-10 indicators we surface on the Chart screen. Adding a new
/// one here also requires the upstream id to appear in
/// `crates/neoethos-data/src/core/all_indicators.rs::ALL_INDICATORS`.
/// Order matters: it drives the order they show up in the UI
/// dropdown.
pub const ALLOWED_INDICATORS: &[&str] = &[
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "stoch",
    "adx",
    "vwap",
];

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndicatorQuery {
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    #[serde(
        default,
        deserialize_with = "super::chart::deserialize_optional_dataset_identity"
    )]
    pub dataset_identity: Option<CanonicalDatasetIdentity>,
    pub expected_generation: Option<String>,
    pub indicator: Option<String>,
    /// Optional period for indicators that take one
    /// (sma/ema/rsi/atr/adx). Library default when missing.
    pub period: Option<f64>,
    /// Bollinger Bands standard-deviation multiplier. Library
    /// default when missing.
    pub std_dev: Option<f64>,
    /// MACD specifics — caller can omit any of these to use
    /// library defaults (12/26/9).
    pub fast: Option<f64>,
    pub slow: Option<f64>,
    pub signal: Option<f64>,
    /// Stochastic specifics — library defaults are 14/3/3.
    pub k_period: Option<f64>,
    pub k_slow: Option<f64>,
    pub d_period: Option<f64>,
    /// How many trailing candles to return. Mirrors `/chart`.
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndicatorDto {
    pub symbol: String,
    pub timeframe: String,
    pub indicator: String,
    pub candle_count: usize,
    /// One per output series — multi-output indicators decompose.
    pub lines: Vec<IndicatorLineDto>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndicatorLineDto {
    pub name: String,
    pub values: Vec<f64>,
}

const DEFAULT_LIMIT: usize = 200;
const MAX_LIMIT: usize = 2000;

pub async fn indicators(
    State(state): State<AppApiState>,
    Query(q): Query<IndicatorQuery>,
) -> Response {
    let receipt = match ExactDatasetReceipt::from_optional(
        q.dataset_identity,
        q.expected_generation,
    ) {
        Ok(Some(receipt)) => receipt,
        Ok(None) => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Indicators require an exact local datasetIdentity and expectedGeneration receipt.",
                &anyhow::anyhow!("no exact dataset receipt was provided"),
            );
        }
        Err(error) => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Indicator dataset selection is incomplete or invalid.",
                &error,
            );
        }
    };
    if let Some(asserted) = q
        .symbol
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if !asserted.eq_ignore_ascii_case(receipt.identity().symbol_name()) {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Indicator symbol assertion disagrees with the exact dataset identity.",
                &anyhow::anyhow!(
                    "symbol assertion {asserted:?} does not match {:?}",
                    receipt.identity().symbol_name()
                ),
            );
        }
    }
    if let Some(asserted) = q
        .timeframe
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if !asserted.eq_ignore_ascii_case(receipt.identity().timeframe().as_str()) {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Indicator timeframe assertion disagrees with the exact dataset identity.",
                &anyhow::anyhow!(
                    "timeframe assertion {asserted:?} does not match {:?}",
                    receipt.identity().timeframe().as_str()
                ),
            );
        }
    }
    let symbol = receipt.identity().symbol_name().to_owned();
    let timeframe = receipt.identity().timeframe().as_str().to_owned();
    let indicator = q
        .indicator
        .unwrap_or_else(|| "sma".to_string())
        .trim()
        .to_ascii_lowercase();
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

    if !ALLOWED_INDICATORS.contains(&indicator.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "indicator '{indicator}' is not in the allowed list; valid: {}",
                    ALLOWED_INDICATORS.join(", ")
                ),
            })),
        )
            .into_response();
    }

    // Translate the per-query params into a generic key→f64 map.
    // The few keys we honour cover the top-10 indicators the UI
    // surfaces; library defaults fill in the rest.
    let mut params: HashMap<String, f64> = HashMap::new();
    if let Some(p) = q.period {
        params.insert("period".to_string(), p);
    }
    if let Some(s) = q.std_dev {
        params.insert("std_dev".to_string(), s);
    }
    if let Some(f) = q.fast {
        params.insert("fast".to_string(), f);
    }
    if let Some(s) = q.slow {
        params.insert("slow".to_string(), s);
    }
    if let Some(s) = q.signal {
        params.insert("signal".to_string(), s);
    }
    if let Some(k) = q.k_period {
        params.insert("k_period".to_string(), k);
    }
    if let Some(s) = q.k_slow {
        params.insert("k_slow".to_string(), s);
    }
    if let Some(d) = q.d_period {
        params.insert("d_period".to_string(), d);
    }

    let indicator_clone = indicator.clone();
    let config_path = state.config_path().to_path_buf();
    let result = tokio::task::spawn_blocking(move || {
        load_and_compute(&config_path, receipt, indicator_clone, params, limit)
    })
    .await;

    match result {
        Ok(Ok((candle_count, lines))) => Json(IndicatorDto {
            symbol,
            timeframe,
            indicator,
            candle_count,
            lines: lines
                .into_iter()
                .map(|l| IndicatorLineDto {
                    name: l.name,
                    values: l.values,
                })
                .collect(),
        })
        .into_response(),
        Ok(Err(err)) => actionable_error(
            StatusCode::CONFLICT,
            "Could not fully verify the exact dataset generation and compute this indicator.",
            &err,
        ),
        Err(join_err) => internal_panic("Computing the indicator", join_err),
    }
}

fn load_and_compute(
    config_path: &std::path::Path,
    receipt: ExactDatasetReceipt,
    indicator: String,
    params: HashMap<String, f64>,
    limit: usize,
) -> anyhow::Result<(usize, Vec<IndicatorLine>)> {
    let settings = Settings::from_yaml(config_path)
        .map_err(|error| anyhow::anyhow!("{} not loadable: {error}", config_path.display()))?;
    let frame = super::chart::load_exact_current_frame(&settings.system.data_dir, &receipt)?;
    let ohlcv = frame.ohlcv();

    // Compute on the full series, then trim to the trailing `limit`
    // candles to match `/chart` semantics — trimming after compute avoids
    // edge effects at the window start (indicators need warm-up bars).
    let lines_full = compute_single_indicator(ohlcv, &indicator, &params)?;
    let total = ohlcv.len();
    let start = total.saturating_sub(limit);
    let trimmed: Vec<IndicatorLine> = lines_full
        .into_iter()
        .map(|l| IndicatorLine {
            name: l.name,
            values: l.values.into_iter().skip(start).collect(),
        })
        .collect();
    let returned_count = trimmed.first().map(|l| l.values.len()).unwrap_or(0);
    Ok((returned_count, trimmed))
}
