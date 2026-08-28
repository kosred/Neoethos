//! `/journal/trades` + `/journal/stats` + `/journal/analytics` — the
//! trade-journal read API.
//!
//! Serves the JSONL closed-trade store ([`crate::app_services::journal_store`]),
//! the computed professional stats ([`crate::app_services::journal_stats`]) and
//! the per-trade breakdown ([`crate::app_services::journal_analytics`] —
//! pips, R, MFE/MAE and the by-hour/by-symbol slices). The data dir is resolved from
//! the live `config.yaml` — the same source the rest of the server uses,
//! so it honours the operator's `--config` / user-data path.
//!
//! Defensive: a config-load failure returns `500` with a clear,
//! actionable message; an empty/missing journal returns an empty list /
//! all-zero stats (never a panic).

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{CanonicalDatasetIdentity, CanonicalOhlcvFrame};
use serde::Deserialize;

use super::chart::ExactDatasetReceipt;
use super::state::AppApiState;
use crate::app_services::{journal_analytics, journal_stats, journal_store};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JournalQuery {
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    /// Max closed trades to return (most-recent first). Default 500.
    pub limit: Option<usize>,
    /// Optional instrument filter, case-insensitive. Absent = every symbol,
    /// which is the behaviour every caller had before this existed.
    ///
    /// Applied AFTER [`heal_row`], because 97 of the operator's stored rows
    /// carry placeholder symbols (`#1`, `#5`, `#41`) written before the broker
    /// catalog populated — filtering on the raw value would silently drop them.
    pub symbol: Option<String>,
    /// Optional exact market-data identity for excursion analytics. This is a
    /// pair with `expected_generation`; partial pairs are rejected.
    #[serde(
        default,
        deserialize_with = "super::chart::deserialize_optional_dataset_identity"
    )]
    pub dataset_identity: Option<CanonicalDatasetIdentity>,
    pub expected_generation: Option<String>,
}

/// Keep only rows for `symbol` when one was asked for. Returns how many rows
/// were dropped, so the caller can say so rather than serving a short list
/// that looks like a short history.
fn apply_symbol_filter(rows: &mut Vec<journal_store::ClosedTrade>, symbol: Option<&str>) -> usize {
    let Some(wanted) = symbol.map(str::trim).filter(|s| !s.is_empty()) else {
        return 0;
    };
    let before = rows.len();
    rows.retain(|r| r.symbol.eq_ignore_ascii_case(wanted));
    before - rows.len()
}

/// Resolve the data dir from the live `config.yaml`, or a 500 Response
/// with a clear message.
fn resolve_data_dir() -> Result<std::path::PathBuf, Response> {
    let path = super::state::current_config_path();
    Settings::from_yaml(&path)
        .map(|s| s.system.data_dir)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("config.yaml not loadable ({}): {e}", path.display()),
                    "code": "config_load_failed",
                })),
            )
                .into_response()
        })
}

/// Heal a served row IN PLACE (display only — the JSONL on disk is never
/// rewritten):
///   - `#<id>` / `sym#<id>` placeholder symbols resolve through the live
///     symbol catalog (records written before the catalog populated).
///   - v1 rows stored the CLOSING deal's side (opposite of the position) and
///     raw base-unit volume; flip the side and convert units → lots.
fn heal_row(t: &mut journal_store::ClosedTrade, names: &std::collections::HashMap<i64, String>) {
    if let Some(id) = t
        .symbol
        .strip_prefix("sym#")
        .or_else(|| t.symbol.strip_prefix('#'))
        .and_then(|s| s.parse::<i64>().ok())
    {
        if let Some(name) = names.get(&id) {
            t.symbol = name.clone();
        }
    }
    if t.schema_version < 2 {
        t.side = match t.side.trim().to_ascii_uppercase().as_str() {
            "BUY" => "SELL".to_string(),
            "SELL" => "BUY".to_string(),
            _ => t.side.clone(),
        };
        if let Some(m) = neoethos_core::symbol_metadata::resolve(&t.symbol)
            .filter(|m| m.contract_size.is_finite() && m.contract_size > 0.0)
        {
            t.lots /= m.contract_size;
        }
        // Serve with v2 semantics so the UI treats every row uniformly.
        t.schema_version = 2;
    }
}

/// `GET /journal/trades?fromMs&toMs&limit&symbol` — closed trades of the ACTIVE
/// account only (automatic — same account selection as the execution path),
/// most-recent first. Legacy rows written before per-account scoping carry no
/// account id (unattributable mixed history) and are hidden once an active
/// account is known; the JSONL on disk keeps everything.
pub async fn trades(State(state): State<AppApiState>, Query(q): Query<JournalQuery>) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let names = state.symbol_catalog_snapshot().await;
    let mut rows = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    if let Some(active) = journal_store::active_account_id() {
        rows.retain(|r| r.account_id.as_deref() == Some(active.as_str()));
    }
    for row in &mut rows {
        heal_row(row, &names);
    }
    apply_symbol_filter(&mut rows, q.symbol.as_deref());
    rows.reverse(); // most-recent first
    rows.truncate(q.limit.unwrap_or(500));
    Json(rows).into_response()
}

/// `GET /journal/stats?fromMs&toMs&symbol` — computed performance stats, scoped
/// to the ACTIVE account (automatic — mirrors `/journal/trades`).
///
/// **`symbol` narrows the TRADE-derived figures only.** An `EquitySample` is an
/// account-level row with no symbol on it, so max drawdown, recovery factor and
/// Sharpe still describe the whole account even when one instrument is asked
/// for. That asymmetry is the same one that made the demo forward-test gate
/// measure the wrong drawdown (#197); it is logged here rather than hidden.
pub async fn stats(State(state): State<AppApiState>, Query(q): Query<JournalQuery>) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let mut trades = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    let mut equity = journal_store::query_equity(&data_dir, q.from_ms, q.to_ms);
    if let Some(active) = journal_store::active_account_id() {
        trades.retain(|r| r.account_id.as_deref() == Some(active.as_str()));
        equity.retain(|e| e.account_id.as_deref() == Some(active.as_str()));
    }
    if q.symbol.is_some() {
        // Placeholder symbols must resolve before they can be filtered on.
        let names = state.symbol_catalog_snapshot().await;
        for row in &mut trades {
            heal_row(row, &names);
        }
        apply_symbol_filter(&mut trades, q.symbol.as_deref());
        // NOTE, and it is the same fact `live_gate` had to be repaired for
        // (#197): an `EquitySample` is an ACCOUNT-level row. It cannot be
        // filtered to one instrument, so with `symbol` set the equity-derived
        // figures below — max drawdown, recovery factor, Sharpe — still
        // describe the whole account. Only the trade-derived figures narrow.
        // `journal_stats::max_drawdown_from_trade_pnl` is the symbol-scoped
        // drawdown when that is what is wanted.
        tracing::debug!(
            target: "neoethos_app::journal",
            symbol = %q.symbol.as_deref().unwrap_or(""),
            trades_in_scope = trades.len(),
            equity_samples = equity.len(),
            "journal stats: trades narrowed to one symbol; equity-derived \
             metrics remain account-wide (an equity sample has no symbol)"
        );
    }
    Json(journal_stats::compute_stats(&trades, &equity)).into_response()
}

/// One fully verified and pinned immutable price generation used for excursion
/// analytics. A different symbol/source/account/timeframe is never guessed.
struct StorePrices {
    frame: CanonicalOhlcvFrame,
}

impl StorePrices {
    fn open(root: &std::path::Path, receipt: &ExactDatasetReceipt) -> anyhow::Result<Self> {
        Ok(Self {
            frame: super::chart::load_exact_current_frame(root, receipt)?,
        })
    }
}

impl journal_analytics::PriceWindow for StorePrices {
    fn window(&self, symbol: &str, from_ms: i64, to_ms: i64) -> Option<(Vec<f64>, Vec<f64>)> {
        if !self
            .frame
            .artifact()
            .identity()
            .symbol_name()
            .eq_ignore_ascii_case(symbol)
        {
            return None;
        }
        let ohlcv = self.frame.ohlcv();
        let stamps = ohlcv.timestamp.as_ref()?;
        let mut highs = Vec::new();
        let mut lows = Vec::new();
        for (index, ts) in stamps.iter().enumerate() {
            if *ts >= from_ms && *ts <= to_ms {
                highs.push(ohlcv.high[index]);
                lows.push(ohlcv.low[index]);
            }
        }
        (!highs.is_empty()).then_some((highs, lows))
    }

    fn pip_size(&self, symbol: &str) -> Option<f64> {
        neoethos_core::symbol_metadata::resolve(symbol)
            .map(|meta| meta.pip_size)
            .filter(|pip| pip.is_finite() && *pip > 0.0)
    }
}

/// `GET /journal/analytics?fromMs&toMs&symbol` — the same closed trades, in
/// units that compare and sliced the ways that locate a problem.
///
/// `/journal/stats` reports the totals. This reports where they came from: per
/// trade in pips and R with the excursion recovered from the price series, and
/// grouped by symbol, hour, weekday and direction. A total says the account
/// lost money; these say which hour, which symbol, and whether the profit was
/// ever there to begin with.
///
/// Response shape (camelCase), for a client rendering it:
/// * `trades[]` — one [`journal_analytics::DerivedTrade`] per closed trade:
///   `pips`, `rMultiple`, `mfePips`, `maePips`, `captureRatio`,
///   `durationHours`, `entryHourUtc`, `entryWeekday`, plus `riskPerLot` and
///   `riskBasis` (`"symbol"` / `"all_symbols"` / `"none"`) so an R is never
///   shown without its denominator.
/// * `bySymbol[]`, `byHourUtc[]`, `byWeekday[]`, `bySide[]` — `BucketSummary`:
///   `bucket`, `trades`, `wins`, `winRatePct`, `netProfit`, `expectancy`,
///   `netPips`.
/// * `avgMfePips`, `avgCaptureRatio`, `inactiveHoursUtc[]`.
/// * `coverage` — how many rows each derived figure could be computed for, and
///   why the rest could not: `tradesTotal`, `withPips`, `withExcursion`,
///   `withRMultiple`, `withDuration`, `missingEntryTime`,
///   `missingPriceSeries`, `riskPerLotBySymbol`, `riskPerLotAllSymbols`,
///   `symbolsUsingFallbackRisk[]`, `minLossesForSymbolRisk`. **Render it.** A
///   blank cell with no reason is how a missing price store gets read as "the
///   trade never went anywhere".
///
/// Every optional field is `null` when it could not be derived, never `0`.
pub async fn analytics(
    State(state): State<AppApiState>,
    Query(q): Query<JournalQuery>,
) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let receipt: Option<ExactDatasetReceipt> =
        match ExactDatasetReceipt::from_optional(q.dataset_identity, q.expected_generation) {
            Ok(receipt) => receipt,
            Err(error) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": format!("invalid journal price dataset receipt: {error}"),
                        "code": "invalid_dataset_receipt",
                    })),
                )
                    .into_response();
            }
        };
    if let (Some(receipt), Some(asserted)) = (
        receipt.as_ref(),
        q.symbol
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty()),
    ) {
        if !asserted.eq_ignore_ascii_case(receipt.identity().symbol_name()) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "journal symbol assertion {asserted:?} disagrees with exact dataset identity symbol {:?}",
                        receipt.identity().symbol_name()
                    ),
                    "code": "dataset_identity_mismatch",
                })),
            )
                .into_response();
        }
    }

    let names = state.symbol_catalog_snapshot().await;
    let mut trades = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    if let Some(active) = journal_store::active_account_id() {
        trades.retain(|r| r.account_id.as_deref() == Some(active.as_str()));
    }
    // Same healing `/journal/trades` applies. Without it the per-symbol
    // breakdown splits one instrument across a name and a placeholder id — 97
    // of the operator's 238 trades are stored as `#1`, `#5`, `#41` from before
    // the catalog populated, so the grouping this endpoint exists for would be
    // wrong in exactly the rows that matter most.
    for row in &mut trades {
        heal_row(row, &names);
    }
    let dropped_other_symbols = apply_symbol_filter(&mut trades, q.symbol.as_deref());
    if dropped_other_symbols > 0 {
        tracing::debug!(
            target: "neoethos_app::journal",
            symbol = %q.symbol.as_deref().unwrap_or(""),
            dropped_other_symbols,
            kept = trades.len(),
            "journal analytics: narrowed to one instrument"
        );
    }
    // Absence of a receipt is an explicit no-price-data mode: optional
    // excursion fields stay unavailable and the analytics layer records that
    // absence. When a receipt is supplied, open/verify/pin it before deriving a
    // single high/low value; generation drift is a loud conflict.
    let prices = match receipt {
        Some(receipt) => {
            let root = data_dir.clone();
            match tokio::task::spawn_blocking(move || StorePrices::open(&root, &receipt)).await {
                Ok(Ok(prices)) => Some(prices),
                Ok(Err(error)) => {
                    return (
                        StatusCode::CONFLICT,
                        Json(serde_json::json!({
                            "error": format!(
                                "journal price generation could not be fully verified and pinned: {error}"
                            ),
                            "code": "dataset_generation_conflict",
                        })),
                    )
                        .into_response();
                }
                Err(join_error) => {
                    return super::errors::internal_panic(
                        "Loading exact journal price generation",
                        join_error,
                    );
                }
            }
        }
        None => None,
    };
    let result = match prices.as_ref() {
        Some(prices) => journal_analytics::analyse(&trades, Some(prices)),
        None => journal_analytics::analyse(&trades, None),
    };
    Json(result).into_response()
}
