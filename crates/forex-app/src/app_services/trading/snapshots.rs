//! Pure snapshot/builder helpers + chart utilities + bootstrap runner +
//! discovered-accounts sync.
//!
//! Everything in this module is a free function that produces a
//! `MarketChartSnapshot` / `ExecutionSurfaceSnapshot` / `JobSnapshot` from
//! the inputs it receives. No mutation of `TradingSession` state lives
//! here — this is the "view materialization" layer.
//!
//! PRESERVED FIX (do not change without auditor sign-off):
//! - Batch 10 (canonical timeframes): `supported_ctrader_chart_timeframes`
//!   returns `forex_core::CANONICAL_TIMEFRAMES` only. cTrader's API also
//!   accepts M2/M4/M10 but we deliberately drop them so every UI selector,
//!   training pipeline, and resample step agrees on the same 11 canonical
//!   timeframes.

use crate::app_services::broker_config::BrokerAccountTarget;
use crate::app_services::ctrader_auth::CTraderDiscoveredAccount;
use crate::app_services::ctrader_bootstrap::{
    bootstrap_from_ctrader_history, plan_bootstrap_chunks,
};
use crate::app_services::ctrader_data::{CTraderChartHistoryRequest, HistoricalBar};
use crate::app_services::ctrader_messages::{
    SUPPORTED_CTRADER_ORDER_TRIGGER_METHODS, SUPPORTED_CTRADER_ORDER_TYPES,
    SUPPORTED_CTRADER_TIME_IN_FORCE, SUPPORTED_CTRADER_TRADE_SIDES,
};
use crate::app_services::jobs::{JobEventLevel, JobKind, JobSnapshot, JobState, push_recent_event};
use crate::app_state::AppState;
use anyhow::Context;
use forex_data::Ohlcv;
use std::time::{SystemTime, UNIX_EPOCH};

use super::diagnostics::{
    append_ctrader_order_builder_diagnostics, format_ctrader_deal_line, format_ctrader_history_row,
    format_ctrader_pending_order_line, format_ctrader_position_line,
};
use super::{
    AppExecutionRuntimeSnapshot, ChartCandle, ChartOverlay, ConnectionSnapshot, ExecutionAction,
    ExecutionSelectionOption, ExecutionSurfaceSnapshot, ExecutionTicketSnapshot,
    MarketChartSnapshot, SUPPORTED_TRADING_ADAPTERS, TradingPanelMode,
};

pub(super) const MAX_CHART_CANDLES: usize = 96;

pub(super) fn supported_ctrader_chart_timeframes() -> Vec<String> {
    // The chart panel exposes only the canonical timeframes; cTrader
    // also accepts M2/M4/M10 but we deliberately drop them so every UI
    // selector, training pipeline, and resample step agrees.
    forex_core::CANONICAL_TIMEFRAMES
        .iter()
        .map(|tf| (*tf).to_string())
        .collect()
}

pub(super) fn chart_history_window_ms(timeframe: &str) -> Option<i64> {
    // Reuse the canonical timeframe→minutes parser so we don't drift
    // from forex-data's resample logic.
    let minutes = forex_data::parse_timeframe_to_minutes(timeframe).ok()?;
    Some(minutes * 60_000 * (MAX_CHART_CANDLES as i64 + 24))
}

fn ohlcv_from_historical_bars(bars: &[HistoricalBar]) -> Ohlcv {
    Ohlcv {
        timestamp: Some(bars.iter().map(|bar| bar.timestamp_ms).collect()),
        open: bars.iter().map(|bar| bar.open).collect(),
        high: bars.iter().map(|bar| bar.high).collect(),
        low: bars.iter().map(|bar| bar.low).collect(),
        close: bars.iter().map(|bar| bar.close).collect(),
        volume: Some(
            bars.iter()
                .map(|bar| bar.volume.unwrap_or_default() as f64)
                .collect(),
        ),
    }
}

pub fn build_market_chart_snapshot_from_historical_bars(
    symbol: &str,
    timeframe: &str,
    available_timeframes: Vec<String>,
    bars: &[HistoricalBar],
    overlays: Vec<ChartOverlay>,
    warnings: Vec<String>,
) -> MarketChartSnapshot {
    let ohlcv = ohlcv_from_historical_bars(bars);
    build_market_chart_snapshot_from_ohlcv(
        symbol,
        timeframe,
        available_timeframes,
        &ohlcv,
        overlays,
        warnings,
    )
}

pub fn build_market_chart_snapshot_from_ohlcv(
    symbol: &str,
    timeframe: &str,
    available_timeframes: Vec<String>,
    ohlcv: &Ohlcv,
    overlays: Vec<ChartOverlay>,
    warnings: Vec<String>,
) -> MarketChartSnapshot {
    let start = ohlcv.len().saturating_sub(MAX_CHART_CANDLES);
    let timestamps = ohlcv.timestamp.as_deref();
    let volumes = ohlcv.volume.as_deref();
    let candles: Vec<ChartCandle> = (start..ohlcv.len())
        .map(|idx| ChartCandle {
            timestamp: timestamps.and_then(|ts| ts.get(idx)).copied(),
            open: ohlcv.open[idx],
            high: ohlcv.high[idx],
            low: ohlcv.low[idx],
            close: ohlcv.close[idx],
            volume: volumes.and_then(|v| v.get(idx)).copied().unwrap_or(0.0),
        })
        .collect();

    let (price_min, price_max) = if candles.is_empty() {
        (0.0, 0.0)
    } else {
        candles
            .iter()
            .fold((f64::MAX, f64::MIN), |(min_v, max_v), candle| {
                (min_v.min(candle.low), max_v.max(candle.high))
            })
    };

    let latest_close = candles
        .last()
        .map(|candle| candle.close)
        .unwrap_or_default();
    let headline = if candles.is_empty() {
        format!("No candles loaded for {symbol} {timeframe}")
    } else {
        format!(
            "{} candles · latest close {:.5} · range {:.5}-{:.5}",
            candles.len(),
            latest_close,
            price_min,
            price_max
        )
    };

    let price_change_pct = if candles.len() >= 2 {
        let first_open = candles.first().map(|c| c.open).unwrap_or(0.0);
        let last_close = candles.last().map(|c| c.close).unwrap_or(0.0);
        if first_open > 0.0 {
            Some((last_close - first_open) / first_open * 100.0)
        } else {
            None
        }
    } else {
        None
    };

    MarketChartSnapshot {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        available_timeframes,
        candles,
        overlays,
        price_min,
        price_max,
        bid: None,
        ask: None,
        price_change_pct,
        headline,
        overlay_status: "Trade overlays will appear here once execution events are available."
            .to_string(),
        warnings,
    }
}

pub fn build_execution_surface_snapshot_with_runtime(
    state: &AppState,
    connection: &ConnectionSnapshot,
    runtime: Option<&AppExecutionRuntimeSnapshot>,
    mut runtime_warnings: Vec<String>,
) -> ExecutionSurfaceSnapshot {
    let action_reason = match connection.mode {
        TradingPanelMode::LocalOnly => {
            Some("Local mode disables live order submission.".to_string())
        }
        TradingPanelMode::Disconnected => {
            Some("Connect a broker adapter before sending live orders.".to_string())
        }
        TradingPanelMode::Connected => None,
    };
    let action_enabled = action_reason.is_none() && connection.supports_live_orders;
    let mut warnings: Vec<String> = action_reason
        .clone()
        .into_iter()
        .chain((!connection.connected && connection.requires_local_terminal).then(|| {
            "The configured adapter requires a local terminal runtime that is not currently connected.".to_string()
        }))
        .collect();
    warnings.append(&mut runtime_warnings);
    let mut diagnostics = vec![
        format!("Adapter: {}", connection.adapter_name),
        format!("Integration: {}", connection.integration_mode),
        format!(
            "Market data capability: {}",
            if connection.supports_market_data {
                "available"
            } else {
                "unavailable"
            }
        ),
        format!(
            "Live order capability: {}",
            if connection.supports_live_orders {
                "available when connected"
            } else {
                "unavailable"
            }
        ),
    ];
    if !connection.terminal_info.trim().is_empty() {
        diagnostics.push(format!("Terminal: {}", connection.terminal_info));
    }
    if connection.adapter_name == "cTrader" {
        diagnostics.push(format!(
            "Supported trade sides: {}",
            SUPPORTED_CTRADER_TRADE_SIDES
                .iter()
                .map(|side| side.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        diagnostics.push(format!(
            "Supported order types: {}",
            SUPPORTED_CTRADER_ORDER_TYPES
                .iter()
                .map(|order_type| order_type.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        diagnostics.push(format!(
            "Supported time-in-force: {}",
            SUPPORTED_CTRADER_TIME_IN_FORCE
                .iter()
                .map(|tif| tif.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        diagnostics.push(format!(
            "Supported trigger methods: {}",
            SUPPORTED_CTRADER_ORDER_TRIGGER_METHODS
                .iter()
                .map(|trigger| trigger.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let (
        positions,
        pending_orders,
        bot_timeline,
        history_rows,
        position_choices,
        pending_order_choices,
    ) = if let Some(runtime) = runtime {
        match runtime {
            AppExecutionRuntimeSnapshot::CTrader(runtime) => {
                diagnostics.push(format!("Trader balance: {:.2}", runtime.trader.balance));
                diagnostics.push(format!("Trader account id: {}", runtime.trader.account_id));
                if let Some(leverage) = runtime.trader.leverage {
                    diagnostics.push(format!("Leverage: {:.2}x", leverage));
                }
                if let Some(account_type) = &runtime.trader.account_type {
                    diagnostics.push(format!("Account type: {account_type}"));
                }
                if let Some(broker_name) = &runtime.trader.broker_name {
                    diagnostics.push(format!("Broker: {broker_name}"));
                }
                diagnostics.push(format!(
                    "Open positions: {}",
                    runtime.reconcile.positions.len()
                ));
                diagnostics.push(format!(
                    "Pending orders: {}",
                    runtime.reconcile.pending_orders.len()
                ));
                diagnostics.push(format!("Recent fills: {}", runtime.recent_deals.len()));
                append_ctrader_order_builder_diagnostics(&mut diagnostics, runtime);
                (
                    runtime
                        .reconcile
                        .positions
                        .iter()
                        .map(format_ctrader_position_line)
                        .collect(),
                    runtime
                        .reconcile
                        .pending_orders
                        .iter()
                        .map(format_ctrader_pending_order_line)
                        .collect(),
                    runtime
                        .recent_deals
                        .iter()
                        .map(format_ctrader_deal_line)
                        .collect(),
                    runtime
                        .recent_deals
                        .iter()
                        .map(format_ctrader_history_row)
                        .collect(),
                    runtime
                        .reconcile
                        .positions
                        .iter()
                        .map(|position| ExecutionSelectionOption {
                            id: position.position_id,
                            label: format_ctrader_position_line(position),
                        })
                        .collect(),
                    runtime
                        .reconcile
                        .pending_orders
                        .iter()
                        .map(|order| ExecutionSelectionOption {
                            id: order.order_id,
                            label: format_ctrader_pending_order_line(order),
                        })
                        .collect(),
                )
            }
        }
    } else {
        diagnostics.push("Live execution runtime info is currently being managed via the central broker background loop.".to_string());
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };

    ExecutionSurfaceSnapshot {
        symbol: state.selected_pair.clone(),
        adapter_name: connection.adapter_name.clone(),
        integration_mode: connection.integration_mode.clone(),
        connection_status: connection.status_text.clone(),
        supported_adapters: SUPPORTED_TRADING_ADAPTERS
            .iter()
            .map(|kind| kind.as_str().to_string())
            .collect(),
        primary_actions: vec![
            ExecutionAction {
                label: "Buy".to_string(),
                enabled: action_enabled,
                reason: action_reason.clone(),
            },
            ExecutionAction {
                label: "Sell".to_string(),
                enabled: action_enabled,
                reason: action_reason,
            },
        ],
        warnings,
        diagnostics,
        positions,
        pending_orders,
        bot_timeline,
        history_rows,
        journal_rows: Vec::new(),
        selected_position_id: state.order_ticket.selected_position_id,
        selected_order_id: state.order_ticket.selected_order_id,
        position_choices,
        pending_order_choices,
        ticket: ExecutionTicketSnapshot {
            lot_size: state.order_ticket.lot_size,
            slippage_in_points: state.order_ticket.slippage_in_points,
            comment: state.order_ticket.comment.clone(),
            label: state.order_ticket.label.clone(),
            max_lot_size: state.risk.max_lot_size,
        },
    }
}

pub(super) fn preferred_chart_timeframe(
    available_timeframes: &[String],
    requested: &str,
) -> String {
    if available_timeframes.iter().any(|tf| tf == requested) {
        return requested.to_string();
    }

    for preferred in ["M1", "M5", "M15", "H1"] {
        if available_timeframes.iter().any(|tf| tf == preferred) {
            return preferred.to_string();
        }
    }

    available_timeframes
        .first()
        .cloned()
        .unwrap_or_else(|| requested.to_string())
}

pub(super) fn sync_ctrader_discovered_accounts_into_targets(
    existing_targets: &[BrokerAccountTarget],
    discovered_accounts: &[CTraderDiscoveredAccount],
) -> Vec<BrokerAccountTarget> {
    discovered_accounts
        .iter()
        .map(|account| {
            if let Some(existing) = existing_targets
                .iter()
                .find(|target| target.account_id == account.account_id)
            {
                BrokerAccountTarget {
                    account_id: existing.account_id.clone(),
                    label: existing.label.clone(),
                    enabled_for_execution: existing.enabled_for_execution,
                }
            } else {
                BrokerAccountTarget {
                    account_id: account.account_id.clone(),
                    label: if !account.account_name.trim().is_empty() {
                        account.account_name.clone()
                    } else if !account.broker_title.trim().is_empty() {
                        account.broker_title.clone()
                    } else {
                        format!("cTrader Account {}", account.account_id)
                    },
                    enabled_for_execution: false,
                }
            }
        })
        .collect()
}

pub(super) fn sync_discovered_accounts_with_targets(
    discovered_accounts: &[CTraderDiscoveredAccount],
    targets: &[BrokerAccountTarget],
) -> Vec<CTraderDiscoveredAccount> {
    discovered_accounts
        .iter()
        .map(|account| {
            let enabled = targets
                .iter()
                .find(|target| target.account_id == account.account_id)
                .map(|target| target.enabled_for_execution)
                .unwrap_or(false);
            let mut synced = account.clone();
            synced.enabled_for_execution = enabled;
            synced
        })
        .collect()
}

pub(super) fn run_ctrader_bootstrap_batch_with_context(
    context: &super::CTraderBootstrapContext,
    data_root: &std::path::Path,
    symbols: &[String],
    timeframes: &[String],
    years: u32,
) -> anyhow::Result<JobSnapshot> {
    if symbols.is_empty() || timeframes.is_empty() {
        return Err(anyhow::anyhow!(
            "bootstrap requires at least one symbol and one timeframe"
        ));
    }

    let mut snapshot = JobSnapshot::new(JobKind::Bootstrap);
    snapshot.state = JobState::Running;
    snapshot.progress.stage = "bootstrap_planning".to_string();
    snapshot.progress.message = format!(
        "Preparing {} symbols across {} timeframes",
        symbols.len(),
        timeframes.len()
    );
    snapshot
        .report
        .counters
        .push(("requested_symbols".to_string(), symbols.len() as u64));
    snapshot
        .report
        .counters
        .push(("requested_timeframes".to_string(), timeframes.len() as u64));
    snapshot
        .report
        .counters
        .push(("requested_years".to_string(), years as u64));

    let total_requests = (symbols.len() * timeframes.len()) as u64;
    let mut completed = 0_u64;
    let mut successes = 0_u64;
    let mut degraded = 0_u64;
    let mut failures = 0_u64;
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
        .as_millis() as i64;

    for symbol in symbols {
        for timeframe in timeframes {
            let planned_chunks =
                plan_bootstrap_chunks(now_ms, timeframe, years).with_context(|| {
                    format!(
                        "failed to plan cTrader bootstrap chunks for {} {} over {} years",
                        symbol, timeframe, years
                    )
                })?;
            snapshot.progress.stage = "bootstrap_fetch".to_string();
            snapshot.progress.message = format!("Bootstrapping {symbol} {timeframe}");
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "bootstrap started for {symbol} {timeframe} with {} planned chunks",
                    planned_chunks.len()
                ),
            );

            let request = CTraderChartHistoryRequest {
                client_id: context.client_id.clone(),
                client_secret: context.client_secret.clone(),
                access_token: context.access_token.clone(),
                environment: context.environment,
                account_id: context.account_id.clone(),
                symbol_name: symbol.clone(),
                timeframe: timeframe.clone(),
                from_timestamp_ms: now_ms,
                to_timestamp_ms: now_ms,
                count: None,
            };
            let outcome = bootstrap_from_ctrader_history(data_root, &request, now_ms, years);
            completed += 1;
            snapshot.progress.percent = Some(completed as f32 / total_requests as f32);

            match outcome {
                Ok(outcome) => {
                    let missing_segments = outcome.coverage.missing_segments.len();
                    if outcome.coverage.fully_covered {
                        successes += 1;
                    } else {
                        degraded += 1;
                        snapshot.report.warnings.push(format!(
                            "{} {} has {} uncovered segments after bootstrap",
                            symbol, timeframe, missing_segments
                        ));
                    }
                    snapshot.report.entries.push(format!(
                        "{} {} | planned_chunks={} | bars_written={} | covered_segments={} | missing_segments={}",
                        symbol,
                        timeframe,
                        planned_chunks.len(),
                        outcome.bars_written,
                        outcome.coverage.covered_segments.len(),
                        missing_segments
                    ));
                    snapshot.report.events = push_recent_event(
                        &snapshot.report.events,
                        if outcome.coverage.fully_covered {
                            JobEventLevel::Info
                        } else {
                            JobEventLevel::Warning
                        },
                        format!(
                            "bootstrap finished for {symbol} {timeframe} with {} covered segments",
                            outcome.coverage.covered_segments.len()
                        ),
                    );
                }
                Err(err) => {
                    failures += 1;
                    snapshot
                        .report
                        .errors
                        .push(format!("{symbol} {timeframe}: {err}"));
                    snapshot.report.events = push_recent_event(
                        &snapshot.report.events,
                        JobEventLevel::Error,
                        format!("bootstrap failed for {symbol} {timeframe}: {err}"),
                    );
                }
            }
        }
    }

    snapshot
        .report
        .counters
        .push(("completed_requests".to_string(), completed));
    snapshot
        .report
        .counters
        .push(("succeeded_requests".to_string(), successes));
    snapshot
        .report
        .counters
        .push(("degraded_requests".to_string(), degraded));
    snapshot
        .report
        .counters
        .push(("failed_requests".to_string(), failures));
    snapshot.report.highlights.push((
        "requests".to_string(),
        format!("{}/{} completed", completed, total_requests),
    ));
    snapshot.report.summary = format!(
        "Bootstrap finished: {} succeeded, {} degraded, {} failed",
        successes, degraded, failures
    );
    snapshot.report.log_path = Some("logs/forex-ai.log".to_string());
    snapshot.state = if failures == total_requests {
        JobState::Failed
    } else if failures > 0 || degraded > 0 {
        JobState::Degraded
    } else {
        JobState::Succeeded
    };
    snapshot.progress.stage = "bootstrap_complete".to_string();
    snapshot.progress.message = snapshot.report.summary.clone();
    Ok(snapshot)
}
