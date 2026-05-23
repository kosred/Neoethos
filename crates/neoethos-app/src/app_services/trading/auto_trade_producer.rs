//! Live-inference auto-trade producer.
//!
//! This is the **producer half** of the AI auto-trade pipeline whose
//! consumer ([`super::auto_trade::TradingSession::dispatch_auto_trade_signal`])
//! already runs the §7.1 gate chain. The producer subscribes to a
//! live bar source (cTrader streaming in production; an in-memory
//! fake in tests), maintains a rolling window of recent bars, runs
//! a [`ModelPredictor`] on each freshly-closed bar, and pushes the
//! resulting [`super::auto_trade::AutoTradeSignal`] back to the
//! [`TradingSession`] via a `Sender<AutoTradeSignal>` channel that
//! the main UI thread drains and feeds to the dispatcher.
//!
//! ## Architecture
//!
//! ```text
//!   live broker (cTrader)
//!         │
//!   ┌─────▼──────────────┐
//!   │ LiveBarSource      │  trait — `poll_latest_bar`
//!   │  └ CTraderLive…    │  prod impl: polls
//!   │                    │      CTraderLiveStreamingBackend
//!   └─────┬──────────────┘
//!         │ HistoricalBar (newly closed)
//!   ┌─────▼──────────────┐
//!   │ LiveInferenceProducer │
//!   │  ├ rolling window  │  Vec<HistoricalBar>, capacity-bounded
//!   │  ├ ModelPredictor  │  trait — `predict(&[bars])`
//!   │  │   └ Ensemble…   │  prod impl: lands in Phase D1.2 —
//!   │  │                 │    full 33-base-expert ensemble +
//!   │  │                 │    MetaDecisionStack stack
//!   │  └ cancel flag     │  Arc<AtomicBool> for graceful shutdown
//!   └─────┬──────────────┘
//!         │ AutoTradeSignal
//!   ┌─────▼──────────────┐
//!   │ Sender<AutoTrade…> │  passed in at construction
//!   └─────┬──────────────┘
//!         │
//!   TradingSession.dispatch_auto_trade_signal
//! ```
//!
//! ## Determinism + testability
//!
//! Both [`LiveBarSource`] and [`ModelPredictor`] are trait-objects so
//! tests can wire deterministic fakes without spinning up a broker
//! connection or a trained model. The orchestrator
//! [`LiveInferenceProducer::run`] returns a [`JoinHandle`] that
//! drives the loop in a named OS thread; callers stop the producer
//! by flipping the shared `Arc<AtomicBool>` cancel flag returned by
//! [`LiveInferenceProducer::cancel_flag`] and joining the handle.
//!
//! ## Concrete predictor — landing in Phase D1.2
//!
//! This module ships the producer **FRAMEWORK ONLY**. No concrete
//! [`ModelPredictor`] implementation ships in production code today.
//!
//! Why: an earlier draft of this commit shipped a
//! `MovingAverageCrossPredictor` (SMA-fast × SMA-slow). That was
//! REJECTED by the operator on the 2026-05-17 directive grounds:
//! hardcoded textbook indicators applied to live forex without
//! cost-aware backtest validation are near-certain ruin in seconds.
//! The bot's job is to DISCOVER strategies through the 33-model
//! training stack (see `neoethos_models::runtime::capabilities::KNOWN_MODEL_NAMES`
//! and `crates/neoethos-models/src/training_orchestrator.rs`); it
//! must NOT trade on a hand-picked indicator until the ensemble has
//! produced + validated a signal source.
//!
//! Phase D1.2 lands the `EnsemblePredictor` in `neoethos-models` that:
//!   1. Loads all enabled base experts (15-26 depending on config)
//!      from their saved artifact dirs.
//!   2. On each `predict` call, runs `predict_proba` on every base
//!      expert.
//!   3. Stacks their outputs into the meta-feature column layout
//!      the `meta_blender` was trained on.
//!   4. Feeds that to `MetaDecisionStack::predict_runtime`.
//!   5. Returns the calibrated, conformally-gated final
//!      `RuntimePrediction` → mapped to [`PredictionOutput`].
//!
//! Until D1.2 lands, [`super::TradingSession::start_auto_trade_producer`]
//! will refuse to start without an explicit predictor — the
//! operator cannot accidentally enable auto-trade without a real
//! model behind it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use super::auto_trade::{AutoTradeSide, AutoTradeSignal};
use super::{CTraderLiveChartUpdateRequest, CTraderLiveStreamingBackend, HistoricalBar};

// ---------------------------------------------------------------------------
// Public traits
// ---------------------------------------------------------------------------

/// One bar from the live broker. The producer polls the source on a
/// fixed cadence; on each poll the implementation MUST return:
/// - `Ok(Some(bar))` for a freshly-closed bar that's newer than any
///   previously emitted bar, OR
/// - `Ok(None)` when the broker has nothing new (the producer naps
///   for its poll interval before trying again), OR
/// - `Err(...)` for a transport/auth failure — the producer logs the
///   error and continues polling.
///
/// Implementations are responsible for de-duplicating against the
/// last-emitted timestamp; the producer treats a duplicate
/// timestamp as a no-op rather than a re-prediction trigger.
pub trait LiveBarSource: Send + Sync {
    /// Poll for the next freshly-closed bar. Non-blocking.
    fn poll_latest_bar(&self) -> Result<Option<HistoricalBar>>;
}

/// Decision output of the predictor.
///
/// Strategy choice is per-implementation; the framework only cares
/// that the predictor maps a rolling window of bars to a (side,
/// confidence) pair. Implementations that have nothing useful to
/// say MUST return `AutoTradeSide::Flat` rather than picking a
/// random direction — the dispatcher's `is_actionable` gate filters
/// Flat signals before they ever reach the order path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionOutput {
    pub side: AutoTradeSide,
    /// Confidence in `[0.0, 1.0]`. Compared against
    /// [`super::auto_trade::AUTO_TRADE_MIN_CONFIDENCE`] by the
    /// dispatcher's gate chain. Implementations should return a
    /// HONEST estimate of their signal quality — overstating
    /// confidence pollutes the operator's downstream decision
    /// metrics.
    pub confidence: f64,
}

/// Maps a rolling window of recent bars to a [`PredictionOutput`].
/// Called by the producer once per freshly-closed bar.
///
/// The `bars` slice is in **chronological order** (oldest first,
/// newest last). The newest bar's timestamp is what the producer
/// will stamp on the emitted signal.
pub trait ModelPredictor: Send + Sync {
    /// Run inference on the rolling window. Errors are logged and
    /// treated as `Flat` by the orchestrator — they do not stop the
    /// producer.
    fn predict(&self, bars: &[HistoricalBar]) -> Result<PredictionOutput>;

    /// Minimum number of bars the predictor needs in the rolling
    /// window before it can emit anything non-Flat. The producer
    /// short-circuits to Flat without calling `predict` when the
    /// window is shorter than this. Default 1 (predictor handles
    /// its own warmup); strategies with longer lookbacks override.
    fn warmup_bars(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tunables for the live-inference loop. Defaults are sensible for
/// a scalping cadence (the operator-directive 2026-05-17 framing).
#[derive(Debug, Clone)]
pub struct LiveInferenceProducerConfig {
    /// Symbol the producer is bound to. Stamped on every emitted
    /// signal; the dispatcher's gate 2 rejects signals whose symbol
    /// doesn't match `AppState.selected_pair`.
    pub symbol: String,
    /// Source-poll interval. Default 1 s — fast enough to catch a
    /// new minute-bar within a second of close on most brokers; slow
    /// enough to avoid hot-looping when the source returns None.
    pub poll_interval: Duration,
    /// Maximum number of bars kept in the rolling window. The
    /// predictor sees the most-recent `rolling_window` bars; older
    /// bars are dropped FIFO. Default 512.
    pub rolling_window: usize,
    /// Human-readable label written onto each `AutoTradeSignal`.
    /// Defaults to "AI" — operator UIs render this on the chart
    /// overlay marker.
    pub signal_label_prefix: String,
}

impl LiveInferenceProducerConfig {
    /// Build a config for a symbol with the defaults named above.
    pub fn for_symbol(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            poll_interval: Duration::from_millis(1_000),
            rolling_window: 512,
            signal_label_prefix: "AI".to_string(),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.symbol.trim().is_empty() {
            bail!("LiveInferenceProducerConfig.symbol must not be empty");
        }
        if self.poll_interval.is_zero() {
            bail!("LiveInferenceProducerConfig.poll_interval must be positive");
        }
        if self.rolling_window == 0 {
            bail!("LiveInferenceProducerConfig.rolling_window must be >= 1");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Live-inference producer. Owns the rolling window state, the
/// predictor, and the bar source; runs as a background OS thread.
pub struct LiveInferenceProducer {
    config: LiveInferenceProducerConfig,
    bar_source: Arc<dyn LiveBarSource>,
    predictor: Arc<dyn ModelPredictor>,
    signal_sender: Sender<AutoTradeSignal>,
    cancel: Arc<AtomicBool>,
}

impl LiveInferenceProducer {
    /// Build a producer. Validates the config. Construction does
    /// NOT spawn the thread — call [`Self::spawn`] for that.
    pub fn new(
        config: LiveInferenceProducerConfig,
        bar_source: Arc<dyn LiveBarSource>,
        predictor: Arc<dyn ModelPredictor>,
        signal_sender: Sender<AutoTradeSignal>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            bar_source,
            predictor,
            signal_sender,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Shared cancel handle. Flip to `true` to ask the running
    /// thread to wind down at its next poll boundary.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    /// Spawn the producer on a named OS thread. Returns the join
    /// handle the caller stores for shutdown.
    ///
    /// The thread runs until either:
    /// - the cancel flag flips to `true`, OR
    /// - the signal channel is closed by the consumer (the dispatch
    ///   side dropped its `Receiver`).
    pub fn spawn(self) -> Result<JoinHandle<ProducerOutcome>> {
        let name = format!("auto-trade-producer-{}", self.config.symbol);
        std::thread::Builder::new()
            .name(name)
            .spawn(move || self.run_loop())
            .map_err(|e| anyhow::anyhow!("failed to spawn auto-trade producer thread: {e}"))
    }

    /// Run the polling loop synchronously on the calling thread.
    /// Useful for tests; production uses [`Self::spawn`].
    pub fn run_loop(self) -> ProducerOutcome {
        let mut rolling: Vec<HistoricalBar> = Vec::with_capacity(self.config.rolling_window);
        let mut last_emit_ts: Option<i64> = None;
        let mut consecutive_errors: u32 = 0;
        const MAX_CONSECUTIVE_ERRORS: u32 = 16;
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return ProducerOutcome::Cancelled;
            }
            let started = Instant::now();
            match self.bar_source.poll_latest_bar() {
                Ok(Some(bar)) => {
                    consecutive_errors = 0;
                    // De-duplicate: the source may return the same
                    // timestamp twice if the broker hasn't ticked.
                    let is_new = match rolling.last() {
                        Some(prev) => bar.timestamp_ms > prev.timestamp_ms,
                        None => true,
                    };
                    if !is_new {
                        // Same or older timestamp — replace the open
                        // bar (broker emitted an update for the
                        // still-forming bar) but DO NOT trigger a
                        // re-prediction.
                        if let Some(prev) = rolling.last_mut()
                            && prev.timestamp_ms == bar.timestamp_ms
                        {
                            *prev = bar;
                        }
                    } else {
                        rolling.push(bar);
                        if rolling.len() > self.config.rolling_window {
                            // FIFO eviction.
                            let drop = rolling.len() - self.config.rolling_window;
                            rolling.drain(..drop);
                        }
                        // Predict only on a fresh close, never on
                        // an updated open bar.
                        if let Some(signal) = self.run_predict(&rolling, last_emit_ts) {
                            last_emit_ts = Some(signal.timestamp_ms);
                            if self.signal_sender.send(signal).is_err() {
                                tracing::info!(
                                    target: "neoethos_app::auto_trade::producer",
                                    "signal channel closed by consumer — stopping producer"
                                );
                                return ProducerOutcome::ConsumerHungUp;
                            }
                        }
                    }
                }
                Ok(None) => {
                    consecutive_errors = 0;
                }
                Err(err) => {
                    consecutive_errors = consecutive_errors.saturating_add(1);
                    tracing::warn!(
                        target: "neoethos_app::auto_trade::producer",
                        error = %err,
                        consecutive_errors,
                        "auto-trade producer poll error"
                    );
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            target: "neoethos_app::auto_trade::producer",
                            consecutive_errors,
                            "too many consecutive poll errors — stopping producer"
                        );
                        return ProducerOutcome::FailedAfterRetries(err.to_string());
                    }
                }
            }
            // Sleep the remainder of the poll interval to avoid hot
            // looping when the source returns None.
            let elapsed = started.elapsed();
            if elapsed < self.config.poll_interval {
                std::thread::sleep(self.config.poll_interval - elapsed);
            }
        }
    }

    /// Run the predictor on the rolling window and build an outbound
    /// signal. Returns `None` when either the window is shorter than
    /// the predictor's warmup, the predictor returns `Flat`, or the
    /// newest bar's timestamp is not strictly newer than the last
    /// emitted one (defensive — the run_loop already checks this).
    fn run_predict(
        &self,
        rolling: &[HistoricalBar],
        last_emit_ts: Option<i64>,
    ) -> Option<AutoTradeSignal> {
        let newest_ts = rolling.last()?.timestamp_ms;
        if let Some(prev) = last_emit_ts
            && newest_ts <= prev
        {
            return None;
        }
        if rolling.len() < self.predictor.warmup_bars() {
            return None;
        }
        let outcome = match self.predictor.predict(rolling) {
            Ok(p) => p,
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::auto_trade::producer",
                    error = %err,
                    "predictor returned error; treating as Flat"
                );
                return None;
            }
        };
        if matches!(outcome.side, AutoTradeSide::Flat) {
            return None;
        }
        // Clamp confidence + cap label length for chart overlay
        // sanity — the AutoTradeSignal type doesn't enforce this.
        let confidence = outcome.confidence.clamp(0.0, 1.0);
        Some(AutoTradeSignal {
            symbol: self.config.symbol.clone(),
            side: outcome.side,
            confidence,
            label: format!(
                "{} {:?} · {:.2}",
                self.config.signal_label_prefix, outcome.side, confidence
            ),
            timestamp_ms: newest_ts,
            // #127: stamp the signal with provenance so
            // explain_recent_trades can narrate why each trade fired.
            // We use the producer's `signal_label_prefix` as the
            // strategy_id (it identifies which strategy/source the
            // operator wired up — e.g. "ema_cross_v3") and a stable
            // ensemble label derived from the symbol the producer is
            // bound to. The feature_snapshot stays empty for now; a
            // richer PredictionOutput that exposes the last feature
            // row is a follow-up — capturing it here would require
            // threading the DataFrame back from ModelPredictor::predict
            // which we don't want to do in the same pass that wires
            // the explain tool.
            strategy_id: Some(self.config.signal_label_prefix.clone()),
            model_id: Some(format!("ensemble:{}", self.config.symbol)),
            feature_snapshot: std::collections::HashMap::new(),
        })
    }
}

/// Terminal state of the producer loop. Returned by
/// [`LiveInferenceProducer::run_loop`] and the join handle from
/// [`LiveInferenceProducer::spawn`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProducerOutcome {
    /// Cancel flag flipped — graceful shutdown.
    Cancelled,
    /// The signal `Sender` was dropped by the consumer.
    ConsumerHungUp,
    /// The bar source returned errors on enough consecutive polls
    /// that the producer gave up. The string is the last error.
    FailedAfterRetries(String),
}

// ---------------------------------------------------------------------------
// REMOVED: MovingAverageCrossPredictor
// ---------------------------------------------------------------------------
//
// A previous draft of this module shipped a `MovingAverageCrossPredictor`
// (SMA fast × SMA slow crossover with ATR-normalised confidence) and
// presented it as the "production predictor that ships today". That
// was wrong and the operator-directive 2026-05-17 rejected it: a
// hardcoded textbook indicator applied to live forex without cost-
// aware backtest validation is near-certain ruin in seconds. The
// bot must DISCOVER strategies through the 33-model training stack;
// it must NOT trade until that stack has produced + validated a
// signal source.
//
// The struct + its tests have been deleted intentionally. The
// `ModelPredictor` trait stays as the integration boundary; the
// production implementation lands in Phase D1.2 as the
// `EnsemblePredictor` that orchestrates all 33 base experts +
// the `MetaDecisionStack` (see `crates/neoethos-models/`).

// ---------------------------------------------------------------------------
// Production bar source: cTrader streaming adapter
// ---------------------------------------------------------------------------

/// [`LiveBarSource`] backed by a [`CTraderLiveStreamingBackend`].
///
/// Each `poll_latest_bar` calls the backend's `load_live_chart_update`
/// once, extracts `latest_trendbar` (if present), and de-duplicates
/// against the last-seen timestamp held in the adapter's internal
/// mutex. Returns `Ok(None)` when:
/// - the broker had no trendbar in the response (only a tick update),
///   OR
/// - the returned bar's timestamp is `<=` the last-seen one
///   (broker emitted an update for the still-open bar — only fully
///   closed bars matter for the inference loop).
///
/// The auth/credential plumbing lives ENTIRELY on the
/// `CTraderLiveChartUpdateRequest` the caller hands in: the request
/// carries the embedded developer client_id + client_secret and the
/// operator's access_token. The adapter never touches credentials
/// directly. Construction is just a config object; the live broker
/// session is opened lazily on the first poll.
pub struct CTraderLiveBarSource {
    backend: Arc<dyn CTraderLiveStreamingBackend>,
    request: CTraderLiveChartUpdateRequest,
    last_seen_ts: Mutex<Option<i64>>,
}

impl CTraderLiveBarSource {
    /// Build a new adapter. The backend trait object is shared via
    /// `Arc` so the TradingSession's existing
    /// `ctrader_live_streaming_backend` field can be cloned in.
    pub fn new(
        backend: Arc<dyn CTraderLiveStreamingBackend>,
        request: CTraderLiveChartUpdateRequest,
    ) -> Self {
        Self {
            backend,
            request,
            last_seen_ts: Mutex::new(None),
        }
    }
}

impl LiveBarSource for CTraderLiveBarSource {
    fn poll_latest_bar(&self) -> Result<Option<HistoricalBar>> {
        let update = self.backend.load_live_chart_update(&self.request)?;
        let Some(bar) = update.latest_trendbar else {
            return Ok(None);
        };
        let mut last = self
            .last_seen_ts
            .lock()
            .map_err(|_| anyhow::anyhow!("CTraderLiveBarSource last_seen_ts mutex poisoned"))?;
        if let Some(prev) = *last
            && bar.timestamp_ms <= prev
        {
            return Ok(None);
        }
        *last = Some(bar.timestamp_ms);
        Ok(Some(bar))
    }
}

// ---------------------------------------------------------------------------
// TradingSession-side handle
// ---------------------------------------------------------------------------

/// Running producer state held by the [`super::TradingSession`].
///
/// The session creates this in `start_auto_trade_producer`, drains
/// signals from `signal_rx` in `drain_auto_trade_signals`, and tears
/// it down in `stop_auto_trade_producer`. The cancel flag is shared
/// with the producer thread so a clean shutdown is a single atomic
/// store + a `join`.
pub struct AutoTradeProducerHandle {
    pub(super) cancel: Arc<AtomicBool>,
    pub(super) handle: Option<JoinHandle<ProducerOutcome>>,
    pub(super) signal_rx: std::sync::mpsc::Receiver<AutoTradeSignal>,
    pub(super) symbol: String,
}

impl AutoTradeProducerHandle {
    /// Symbol the producer was started for. Used by the chrome
    /// status pill and by `drain_auto_trade_signals` to log the
    /// active symbol on every dispatch.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::mpsc;

    // --- helpers --------------------------------------------------

    fn bar(timestamp_ms: i64, open: f64, high: f64, low: f64, close: f64) -> HistoricalBar {
        HistoricalBar {
            timestamp_ms,
            open,
            high,
            low,
            close,
            volume: Some(1),
        }
    }

    /// Bar source backed by a Vec — each `poll_latest_bar` advances
    /// the cursor and returns the next bar, then `Ok(None)`.
    struct CursorBarSource {
        bars: Mutex<std::vec::IntoIter<HistoricalBar>>,
    }

    impl CursorBarSource {
        fn new(bars: Vec<HistoricalBar>) -> Self {
            Self {
                bars: Mutex::new(bars.into_iter()),
            }
        }
    }

    impl LiveBarSource for CursorBarSource {
        fn poll_latest_bar(&self) -> Result<Option<HistoricalBar>> {
            let mut iter = self.bars.lock().expect("CursorBarSource lock");
            Ok(iter.next())
        }
    }

    /// Predictor that always returns the same (side, confidence).
    struct ConstantPredictor {
        side: AutoTradeSide,
        confidence: f64,
    }

    impl ModelPredictor for ConstantPredictor {
        fn predict(&self, _bars: &[HistoricalBar]) -> Result<PredictionOutput> {
            Ok(PredictionOutput {
                side: self.side,
                confidence: self.confidence,
            })
        }
    }

    /// Predictor that always errors. Used to drive the error-tolerance
    /// test — the producer must NOT crash on a predictor error; it
    /// just logs and skips that bar.
    struct ErroringPredictor;

    impl ModelPredictor for ErroringPredictor {
        fn predict(&self, _bars: &[HistoricalBar]) -> Result<PredictionOutput> {
            bail!("synthetic predictor error")
        }
    }

    // --- config ---------------------------------------------------

    #[test]
    fn config_validate_rejects_empty_symbol() {
        let mut cfg = LiveInferenceProducerConfig::for_symbol("EURUSD");
        cfg.symbol = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_zero_poll_interval() {
        let mut cfg = LiveInferenceProducerConfig::for_symbol("EURUSD");
        cfg.poll_interval = Duration::ZERO;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_zero_rolling_window() {
        let mut cfg = LiveInferenceProducerConfig::for_symbol("EURUSD");
        cfg.rolling_window = 0;
        assert!(cfg.validate().is_err());
    }

    // --- producer orchestrator ------------------------------------

    fn producer_with_predictor(
        bars: Vec<HistoricalBar>,
        predictor: Arc<dyn ModelPredictor>,
    ) -> (LiveInferenceProducer, mpsc::Receiver<AutoTradeSignal>) {
        let (tx, rx) = mpsc::channel();
        let mut cfg = LiveInferenceProducerConfig::for_symbol("EURUSD");
        cfg.poll_interval = Duration::from_millis(1); // fast tests
        let source: Arc<dyn LiveBarSource> = Arc::new(CursorBarSource::new(bars));
        let producer = LiveInferenceProducer::new(cfg, source, predictor, tx).expect("producer");
        (producer, rx)
    }

    #[test]
    fn producer_emits_one_signal_per_new_bar_when_predictor_is_active() {
        // 3 new-timestamp bars + constant Buy predictor — exactly 3
        // signals must arrive.
        let bars = vec![
            bar(1_000, 1.0, 1.0, 1.0, 1.0),
            bar(2_000, 1.0, 1.0, 1.0, 1.0),
            bar(3_000, 1.0, 1.0, 1.0, 1.0),
        ];
        let predictor: Arc<dyn ModelPredictor> = Arc::new(ConstantPredictor {
            side: AutoTradeSide::Buy,
            confidence: 0.9,
        });
        let (producer, rx) = producer_with_predictor(bars, predictor);
        let cancel = producer.cancel_flag();
        let handle = std::thread::spawn(move || producer.run_loop());
        let mut received = Vec::new();
        for _ in 0..3 {
            received.push(
                rx.recv_timeout(Duration::from_secs(2))
                    .expect("signal within 2s"),
            );
        }
        cancel.store(true, Ordering::Relaxed);
        let outcome = handle.join().expect("join");
        assert!(matches!(outcome, ProducerOutcome::Cancelled));
        assert_eq!(received.len(), 3);
        for sig in &received {
            assert_eq!(sig.side, AutoTradeSide::Buy);
            assert!((sig.confidence - 0.9).abs() < 1e-9);
            assert_eq!(sig.symbol, "EURUSD");
        }
        assert_eq!(received[0].timestamp_ms, 1_000);
        assert_eq!(received[1].timestamp_ms, 2_000);
        assert_eq!(received[2].timestamp_ms, 3_000);
    }

    #[test]
    fn producer_skips_emit_on_flat_predictor_output() {
        let bars = vec![
            bar(1_000, 1.0, 1.0, 1.0, 1.0),
            bar(2_000, 1.0, 1.0, 1.0, 1.0),
        ];
        let predictor: Arc<dyn ModelPredictor> = Arc::new(ConstantPredictor {
            side: AutoTradeSide::Flat,
            confidence: 0.9,
        });
        let (producer, rx) = producer_with_predictor(bars, predictor);
        let cancel = producer.cancel_flag();
        let handle = std::thread::spawn(move || producer.run_loop());
        // Give the producer time to drain the cursor.
        std::thread::sleep(Duration::from_millis(50));
        cancel.store(true, Ordering::Relaxed);
        handle.join().expect("join");
        assert!(
            rx.try_recv().is_err(),
            "no signal must arrive when predictor returns Flat"
        );
    }

    #[test]
    fn producer_deduplicates_same_timestamp() {
        // Two bars at the same timestamp — the second updates the
        // first but does NOT trigger a fresh prediction. Combined
        // with a constant Buy predictor that would otherwise emit
        // on every new bar, only ONE signal should be emitted.
        let bars = vec![
            bar(5_000, 1.0, 1.0, 1.0, 1.0),
            bar(5_000, 1.0, 1.1, 0.9, 1.05), // same timestamp, updated OHLC
        ];
        let predictor: Arc<dyn ModelPredictor> = Arc::new(ConstantPredictor {
            side: AutoTradeSide::Buy,
            confidence: 0.7,
        });
        let (producer, rx) = producer_with_predictor(bars, predictor);
        let cancel = producer.cancel_flag();
        let handle = std::thread::spawn(move || producer.run_loop());
        let first = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("at least one signal");
        std::thread::sleep(Duration::from_millis(50));
        cancel.store(true, Ordering::Relaxed);
        handle.join().expect("join");
        assert_eq!(first.timestamp_ms, 5_000);
        assert!(
            rx.try_recv().is_err(),
            "second poll at same timestamp must NOT trigger a second signal"
        );
    }

    #[test]
    fn producer_tolerates_predictor_errors_without_crashing() {
        // ErroringPredictor returns Err on every call. The producer
        // must log and continue rather than panicking; we observe
        // by exhausting the source + receiving zero signals + the
        // outcome being Cancelled (not a panic during join).
        let bars = vec![bar(1_000, 1.0, 1.0, 1.0, 1.0)];
        let predictor: Arc<dyn ModelPredictor> = Arc::new(ErroringPredictor);
        let (producer, rx) = producer_with_predictor(bars, predictor);
        let cancel = producer.cancel_flag();
        let handle = std::thread::spawn(move || producer.run_loop());
        std::thread::sleep(Duration::from_millis(50));
        cancel.store(true, Ordering::Relaxed);
        let outcome = handle.join().expect("join must not panic");
        assert!(matches!(outcome, ProducerOutcome::Cancelled));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn producer_stops_when_consumer_closes_channel() {
        let bars = vec![
            bar(1_000, 1.0, 1.0, 1.0, 1.0),
            bar(2_000, 1.0, 1.0, 1.0, 1.0),
        ];
        let predictor: Arc<dyn ModelPredictor> = Arc::new(ConstantPredictor {
            side: AutoTradeSide::Buy,
            confidence: 0.9,
        });
        let (producer, rx) = producer_with_predictor(bars, predictor);
        let handle = std::thread::spawn(move || producer.run_loop());
        // Drop the receiver — the next send must fail and the
        // producer must wind down gracefully.
        drop(rx);
        let outcome = handle.join().expect("join");
        assert!(
            matches!(outcome, ProducerOutcome::ConsumerHungUp),
            "expected ConsumerHungUp, got {outcome:?}"
        );
    }

    #[test]
    fn producer_label_includes_symbol_side_and_confidence() {
        let bars = vec![bar(1_000, 1.0, 1.0, 1.0, 1.0)];
        let predictor: Arc<dyn ModelPredictor> = Arc::new(ConstantPredictor {
            side: AutoTradeSide::Sell,
            confidence: 0.74,
        });
        let (producer, rx) = producer_with_predictor(bars, predictor);
        let cancel = producer.cancel_flag();
        let handle = std::thread::spawn(move || producer.run_loop());
        let sig = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("first signal");
        cancel.store(true, Ordering::Relaxed);
        handle.join().expect("join");
        assert_eq!(sig.symbol, "EURUSD");
        assert_eq!(sig.side, AutoTradeSide::Sell);
        assert!(sig.label.contains("AI"));
        assert!(sig.label.contains("Sell"));
        assert!(sig.label.contains("0.74"));
    }
}
