//! cTrader market-data: chart history fetch, live spot/trendbar streaming,
//! and the timeframe surface for the chart panel.
//!
//! Carved out of `trading/mod.rs` (Batch 5 follow-up). This module owns:
//! - The session-facing `market_chart_snapshot` entry point used by the UI.
//! - cTrader chart-history requests (`build_ctrader_chart_history_request`,
//!   `load_ctrader_market_chart_snapshot`).
//! - Live spot / trendbar subscription (`build_ctrader_live_chart_update_request`,
//!   `load_ctrader_live_chart_update_cached`).
//! - The "which discovered account drives the chart panel" picker
//!   (`selected_ctrader_chart_account_id`).
//!
//! The canonical timeframe list (`supported_ctrader_chart_timeframes`) and the
//! `MarketChartCacheKey` / `CachedMarketSnapshot` / `CachedCTraderLiveSpotUpdate`
//! caches still live in `mod.rs` / `snapshots.rs` so the public re-exports
//! and field-level borrow patterns are unchanged.
//!
//! PRESERVED FIXES (do not change without auditor sign-off):
//! - `supported_ctrader_chart_timeframes()` returns
//!   `forex_core::CANONICAL_TIMEFRAMES` only, owned by `snapshots.rs`. Per
//!   `docs/audits/research/ctrader_api_reference.md` §4 the cTrader
//!   `ProtoOATrendbarPeriod` enum has no `H2` value, so any caller asking
//!   for H2 must resample from H1 — the chart-panel timeframe set we
//!   advertise here MUST stay aligned with that fact.
//! - The chart-history request is timestamp-bounded by
//!   `chart_history_window_ms(timeframe)` (defined in `snapshots.rs`) so an
//!   unknown timeframe is rejected at the request-build boundary rather
//!   than silently falling back to a bogus window.

use super::{
    AppState, CTraderChartHistoryRequest, CTraderLiveChartUpdate, CTraderLiveChartUpdateRequest,
    CTraderLiveSpotCacheKey, CachedCTraderLiveSpotUpdate, CachedMarketSnapshot, DataSource,
    MAX_CHART_CANDLES, MarketChartCacheKey, MarketChartSnapshot, TradingAdapterKind,
    TradingSession, build_market_chart_snapshot_from_historical_bars,
    build_market_chart_snapshot_from_ohlcv, chart_history_window_ms, discover_timeframes,
    load_chart_history, load_symbol_timeframe, merge_live_spot_update_into_bars,
    preferred_chart_timeframe, supported_ctrader_chart_timeframes,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

impl TradingSession {
    pub fn market_chart_snapshot(
        &mut self,
        state: &AppState,
        tx: Option<&tokio::sync::mpsc::Sender<super::ServiceEvent>>,
    ) -> MarketChartSnapshot {
        let adapter_kind = self.active_adapter_kind();
        let available_timeframes = if matches!(state.data_source, DataSource::Local) {
            discover_timeframes(&state.runtime.data_dir, &state.selected_pair).unwrap_or_default()
        } else if matches!(adapter_kind, TradingAdapterKind::CTrader) {
            supported_ctrader_chart_timeframes()
        } else {
            discover_timeframes(&state.runtime.data_dir, &state.selected_pair).unwrap_or_default()
        };
        let timeframe =
            preferred_chart_timeframe(&available_timeframes, state.chart_timeframe.as_str());
        let ctrader_environment = matches!(state.data_source, DataSource::CTrader)
            .then_some(adapter_kind)
            .filter(|kind| matches!(kind, TradingAdapterKind::CTrader))
            .map(|_| self.selected_ctrader_environment());
        let ctrader_account_id = matches!(state.data_source, DataSource::CTrader)
            .then_some(adapter_kind)
            .filter(|kind| matches!(kind, TradingAdapterKind::CTrader))
            .and_then(|_| self.selected_ctrader_chart_account_id());
        let cache_key = MarketChartCacheKey {
            data_root: state.runtime.data_dir.clone(),
            data_source: state.data_source,
            adapter_kind,
            symbol: state.selected_pair.clone(),
            timeframe: timeframe.clone(),
            ctrader_environment,
            ctrader_account_id,
        };

        if let Some(cache) = &self.market_chart_cache {
            let is_live_ctrader_chart = matches!(state.data_source, DataSource::CTrader)
                && matches!(adapter_kind, TradingAdapterKind::CTrader)
                && self.connected;
            if cache.key == cache_key
                && (!is_live_ctrader_chart || cache.refreshed_at.elapsed() < Duration::from_secs(1))
            {
                return cache.snapshot.clone();
            }
        }

        let overlay_status = self.overlay_status(state);
        let resolved_timeframes = if available_timeframes.is_empty() {
            vec![timeframe.clone()]
        } else {
            available_timeframes.clone()
        };
        let snapshot = match (state.data_source, adapter_kind) {
            (DataSource::CTrader, TradingAdapterKind::CTrader) => {
                // Promote any completed background fetch that is waiting.
                if let Some(pending) = self.pending_ctrader_chart.take() {
                    if pending.symbol == state.selected_pair && pending.timeframe == timeframe {
                        self.market_chart_cache = Some(CachedMarketSnapshot {
                            key: cache_key,
                            refreshed_at: Instant::now(),
                            snapshot: pending.clone(),
                        });
                        return pending;
                    }
                    // Symbol/timeframe changed while the fetch was in flight —
                    // discard the stale result; a fresh fetch will be started below.
                }

                // Kick off a background fetch if one is not already running and
                // a sender is available. Return the stale cache (or a placeholder)
                // immediately so the render thread is never blocked on I/O.
                if let Some(tx) = tx {
                    self.start_ctrader_chart_fetch(
                        &state.selected_pair,
                        &timeframe,
                        resolved_timeframes.clone(),
                        overlay_status.clone(),
                        tx.clone(),
                    );
                }
                // Return whatever the cache holds right now (may be stale /
                // empty). When the background thread finishes it sends
                // ServiceEvent::ChartDataUpdated, which updates the cache and
                // triggers an egui repaint.
                if let Some(cache) = &self.market_chart_cache {
                    return cache.snapshot.clone();
                }
                MarketChartSnapshot::empty_for(
                    &state.selected_pair,
                    &timeframe,
                    resolved_timeframes,
                    if self.chart_fetch_is_running() {
                        "Loading cTrader chart data…".to_string()
                    } else {
                        format!("No cTrader data for {} {}", state.selected_pair, timeframe)
                    },
                    overlay_status,
                    Vec::new(),
                )
            }
            _ => match load_symbol_timeframe(
                &state.runtime.data_dir,
                &state.selected_pair,
                &timeframe,
            ) {
                Ok(ohlcv) => {
                    let mut snap = build_market_chart_snapshot_from_ohlcv(
                        &state.selected_pair,
                        &timeframe,
                        resolved_timeframes,
                        &ohlcv,
                        Vec::new(),
                        Vec::new(),
                    )
                    .with_overlay_status(overlay_status);
                    // Audit gap #11: populate the chart overlays from
                    // the bot-decision ring buffer instead of leaving
                    // `Vec::new()`. The candle list is already built
                    // so the mapping just needs the timestamp lookup.
                    snap.overlays =
                        self.bot_decisions_to_overlays(&state.selected_pair, &snap.candles);
                    snap
                }
                Err(err) => MarketChartSnapshot::empty_for(
                    &state.selected_pair,
                    &timeframe,
                    if available_timeframes.is_empty() {
                        vec![timeframe.clone()]
                    } else {
                        available_timeframes.clone()
                    },
                    format!(
                        "No market data loaded for {} {}",
                        state.selected_pair, timeframe
                    ),
                    overlay_status,
                    vec![format!(
                        "Failed to load {} market data for {}: {}",
                        timeframe, state.selected_pair, err
                    )],
                ),
            },
        };

        self.market_chart_cache = Some(CachedMarketSnapshot {
            key: cache_key,
            refreshed_at: Instant::now(),
            snapshot: snapshot.clone(),
        });
        snapshot
    }

    /// Called from `main.rs::process_messages` when
    /// `ServiceEvent::ChartDataUpdated` arrives. Stores the completed
    /// snapshot so the next render-thread call to `market_chart_snapshot`
    /// can promote it into the regular cache without any I/O.
    pub fn apply_chart_data_update(&mut self, snapshot: MarketChartSnapshot) {
        self.pending_ctrader_chart = Some(snapshot);
    }

    pub(super) fn load_ctrader_market_chart_snapshot(
        &mut self,
        symbol: &str,
        timeframe: &str,
        available_timeframes: Vec<String>,
        overlay_status: String,
    ) -> MarketChartSnapshot {
        match self.build_ctrader_chart_history_request(symbol, timeframe) {
            Ok(request) => match load_chart_history(&request) {
                Ok(history) => {
                    let mut warnings = Vec::new();
                    let live_update = if self.connected {
                        match self.build_ctrader_live_chart_update_request(
                            &request,
                            history.symbol.symbol_id,
                            history.symbol.digits,
                        ) {
                            Ok(live_request) => {
                                match self.load_ctrader_live_chart_update_cached(&live_request) {
                                    Ok(update) => Some(update),
                                    Err(err) => {
                                        warnings.push(format!(
                                            "Failed to load cTrader live {} update for {}: {}",
                                            timeframe, symbol, err
                                        ));
                                        None
                                    }
                                }
                            }
                            Err(err) => {
                                warnings.push(format!(
                                    "cTrader live {} update is unavailable for {}: {}",
                                    timeframe, symbol, err
                                ));
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let bars =
                        merge_live_spot_update_into_bars(&history.bars, live_update.as_ref());

                    let mut snapshot = build_market_chart_snapshot_from_historical_bars(
                        &history.symbol.symbol_name,
                        timeframe,
                        available_timeframes,
                        &bars,
                        Vec::new(),
                        warnings,
                    )
                    .with_overlay_status(overlay_status);
                    // Audit gap #11: populate overlays from the
                    // session-held bot decision buffer once the candle
                    // list exists.
                    snapshot.overlays = self
                        .bot_decisions_to_overlays(&history.symbol.symbol_name, &snapshot.candles);
                    if let Some(update) = live_update {
                        snapshot.bid = update.bid;
                        snapshot.ask = update.ask;
                        let quote_line = match (update.bid, update.ask) {
                            (Some(bid), Some(ask)) => {
                                format!(" · bid {:.5} ask {:.5}", bid, ask)
                            }
                            (Some(bid), None) => format!(" · bid {:.5}", bid),
                            (None, Some(ask)) => format!(" · ask {:.5}", ask),
                            (None, None) => String::new(),
                        };
                        if !quote_line.is_empty() {
                            snapshot.headline.push_str(&quote_line);
                        }
                    }
                    snapshot
                }
                Err(err) => MarketChartSnapshot::empty_for(
                    symbol,
                    timeframe,
                    available_timeframes,
                    format!("No cTrader market data loaded for {} {}", symbol, timeframe),
                    overlay_status,
                    vec![format!(
                        "Failed to load cTrader {} market data for {}: {}",
                        timeframe, symbol, err
                    )],
                ),
            },
            Err(err) => MarketChartSnapshot::empty_for(
                symbol,
                timeframe,
                available_timeframes,
                format!("No cTrader market data loaded for {} {}", symbol, timeframe),
                overlay_status,
                vec![format!(
                    "cTrader chart history is unavailable for {} {}: {}",
                    symbol, timeframe, err
                )],
            ),
        }
    }

    /// Latest mid-market price for `symbol_id` from the cached cTrader spot
    /// stream, or `None` if no fresh quote is available.
    ///
    /// Used by the risk gate (Note) as the entry-price
    /// fallback for Market orders that carry no `limit_price`/`stop_price`.
    /// Refusing to size such an order without a quote is the safe behavior;
    /// the previous gate silently bypassed the risk-per-trade check.
    ///
    /// The cache holds only the most recent update, keyed on
    /// `(env, account, symbol, timeframe)`. We accept a symbol-id-only match
    /// because the price for a given symbol is identical across timeframes —
    /// the live spot stream is independent of the trendbar timeframe the
    /// chart panel happens to be showing.
    pub(super) fn ctrader_live_mid_price_for_symbol(&self, symbol_id: i64) -> Option<f64> {
        let cache = self.ctrader_live_spot_cache.as_ref()?;
        if cache.key.symbol_id != symbol_id {
            return None;
        }
        // Treat anything older than 30 s as stale — a market quote that old
        // is not safe as an entry-price estimate for sizing an order.
        if cache.refreshed_at.elapsed() > Duration::from_secs(30) {
            return None;
        }
        match (cache.update.bid, cache.update.ask) {
            (Some(bid), Some(ask)) if bid.is_finite() && ask.is_finite() && bid > 0.0 && ask > 0.0 => {
                Some((bid + ask) / 2.0)
            }
            // Fall back to the side we have if only one is present. A
            // half-quote is still better than no entry estimate at all,
            // since the gate uses `(entry - sl).abs()` symmetrically.
            (Some(bid), None) if bid.is_finite() && bid > 0.0 => Some(bid),
            (None, Some(ask)) if ask.is_finite() && ask > 0.0 => Some(ask),
            _ => None,
        }
    }

    pub(super) fn load_ctrader_live_chart_update_cached(
        &mut self,
        request: &CTraderLiveChartUpdateRequest,
    ) -> anyhow::Result<CTraderLiveChartUpdate> {
        let cache_key = CTraderLiveSpotCacheKey {
            environment: request.environment,
            account_id: request.account_id.clone(),
            symbol_id: request.symbol_id,
            timeframe: request.timeframe.clone(),
        };

        if let Some(cache) = &self.ctrader_live_spot_cache
            && cache.key == cache_key
            && cache.refreshed_at.elapsed() < Duration::from_secs(1)
        {
            return Ok(cache.update.clone());
        }

        let update = self
            .ctrader_live_streaming_backend
            .load_live_chart_update(request)?;
        self.ctrader_live_spot_cache = Some(CachedCTraderLiveSpotUpdate {
            key: cache_key,
            refreshed_at: Instant::now(),
            update: update.clone(),
        });
        Ok(update)
    }

    pub(super) fn build_ctrader_chart_history_request(
        &mut self,
        symbol: &str,
        timeframe: &str,
    ) -> anyhow::Result<CTraderChartHistoryRequest> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader chart history requires configured client_id and client_secret"
            ));
        }

        let access_token = self
            .ensure_fresh_ctrader_token_bundle(
                "cTrader chart history requires a stored token bundle",
            )?
            .access_token;

        let account_id = self.selected_ctrader_chart_account_id().ok_or_else(|| {
            anyhow::anyhow!("cTrader chart history requires at least one discovered account")
        })?;

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
            .as_millis() as i64;
        let window_ms = chart_history_window_ms(timeframe)
            .ok_or_else(|| anyhow::anyhow!("unsupported cTrader chart timeframe {}", timeframe))?;

        Ok(CTraderChartHistoryRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            symbol_name: symbol.to_string(),
            timeframe: timeframe.to_string(),
            from_timestamp_ms: now_ms.saturating_sub(window_ms),
            to_timestamp_ms: now_ms,
            count: Some((MAX_CHART_CANDLES + 24) as u32),
        })
    }

    pub(super) fn build_ctrader_live_chart_update_request(
        &self,
        history_request: &CTraderChartHistoryRequest,
        symbol_id: i64,
        digits: i32,
    ) -> anyhow::Result<CTraderLiveChartUpdateRequest> {
        if history_request.account_id.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader live chart update requires a discovered account"
            ));
        }

        Ok(CTraderLiveChartUpdateRequest {
            client_id: history_request.client_id.clone(),
            client_secret: history_request.client_secret.clone(),
            access_token: history_request.access_token.clone(),
            environment: history_request.environment,
            account_id: history_request.account_id.clone(),
            symbol_id,
            digits,
            timeframe: history_request.timeframe.clone(),
            subscribe_to_spot_timestamp: true,
        })
    }

    pub(super) fn selected_ctrader_chart_account_id(&self) -> Option<String> {
        self.broker_settings
            .ctrader
            .accounts
            .iter()
            .find(|account| account.enabled_for_execution)
            .or_else(|| self.broker_settings.ctrader.accounts.first())
            .map(|account| account.account_id.clone())
    }
}
