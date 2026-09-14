#[cfg(feature = "gpu")]
use crate::cubecl_eval::{cuda_eval_backtest_kernel_enabled, cuda_eval_signal_kernel_enabled};
use crate::quality::Trade;
use ndarray::ArrayView2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// `use std::env` removed 2026-08-10 with the last production env read in this
// file.
use std::sync::OnceLock;

#[cfg(test)]
#[path = "cpu_execution_ledger_tests.rs"]
mod cpu_execution_ledger_tests;

#[cfg(test)]
#[path = "eval_diagnostic_tests.rs"]
mod eval_diagnostic_tests;

pub type SmcRow = [i8; 11];

#[path = "netted_bar_research.rs"]
mod netted_bar_research;
pub use netted_bar_research::{
    BarReplayLotGridV1, NettedBarDecisionTapeV1, NettedBarResearchResultV1,
    PreparedNettedCanonicalBarResearchV1, TerminalBarPositionV1,
};

pub struct PopulationEvalInputs<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub indicators: ArrayView2<'a, f64>,
    pub gene_offsets: &'a [i32],
    pub gene_indices: &'a [i32],
    pub gene_weights: &'a [f64],
    pub long_thr: &'a [f64],
    pub short_thr: &'a [f64],
    pub month_idx: &'a [i64],
    pub day_idx: &'a [i64],
    pub timestamps: &'a [i64],
    pub sl_pips: &'a [f64],
    pub tp_pips: &'a [f64],
    /// Per-gene adaptive volatility multiplier (`Gene.stop_vol_mult`). Empty or
    /// `0.0` entries mean the gene uses its fixed `sl_pips`/`tp_pips`; `> 0` pairs
    /// with `settings.adaptive_base_pips` to scale the per-entry stop by
    /// volatility. Pass `&[]` where adaptive stops don't apply (GPU validation).
    pub stop_vol_mult: &'a [f64],
    pub smc_data: &'a [SmcRow],
    pub gene_smc_flags: &'a [SmcRow],
    pub gate_threshold: f64,
    pub weights: &'a [f64; 11],
    pub settings: &'a BacktestSettings,
}

fn require_historical_evaluation_authority() -> anyhow::Result<()> {
    crate::historical_evaluation_authority::require_historical_evaluation_authority_v1().map(|_| ())
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let (mean, std) = neoethos_core::utils::mean_std(values);
    if !mean.is_finite() || !std.is_finite() {
        return (0.0, 0.0);
    }
    (mean, std)
}

/// Versioned hard-reject marker for a completed-month return whose account-money
/// numerator or month-start-equity denominator is not a valid finite quantity.
/// `ga_fitness` already rejects non-finite Sharpe, so this marker must survive
/// metric assembly instead of being folded into the generic zero scrub.
pub(crate) const INVALID_MONTHLY_RETURN_SHARPE_V1: f64 = f64::NEG_INFINITY;

/// Sharpe over completed-month percentage returns, not raw account-currency PnL.
///
/// The period return is `month_pnl / that_month_start_equity`. Any shape,
/// denominator, input, intermediate, or final arithmetic failure is a candidate
/// rejection, represented by [`INVALID_MONTHLY_RETURN_SHARPE_V1`]. Empty or
/// single-period valid samples retain the existing zero-Sharpe behavior.
pub(crate) fn completed_month_return_sharpe_v1(
    monthly_pnls: &[f64],
    month_start_equities: &[f64],
) -> f64 {
    if monthly_pnls.len() != month_start_equities.len() {
        return INVALID_MONTHLY_RETURN_SHARPE_V1;
    }

    let mut sum = 0.0;
    for (&monthly_pnl, &start_equity) in monthly_pnls.iter().zip(month_start_equities) {
        if !monthly_pnl.is_finite() || !start_equity.is_finite() || start_equity <= 0.0 {
            return INVALID_MONTHLY_RETURN_SHARPE_V1;
        }
        let period_return = monthly_pnl / start_equity;
        if !period_return.is_finite() {
            return INVALID_MONTHLY_RETURN_SHARPE_V1;
        }
        sum += period_return;
        if !sum.is_finite() {
            return INVALID_MONTHLY_RETURN_SHARPE_V1;
        }
    }

    if monthly_pnls.len() < 2 {
        return 0.0;
    }
    let mean = sum / monthly_pnls.len() as f64;
    if !mean.is_finite() {
        return INVALID_MONTHLY_RETURN_SHARPE_V1;
    }

    let mut squared_deviation_sum = 0.0;
    for (&monthly_pnl, &start_equity) in monthly_pnls.iter().zip(month_start_equities) {
        let period_return = monthly_pnl / start_equity;
        let deviation = period_return - mean;
        let squared_deviation = deviation * deviation;
        if !squared_deviation.is_finite() {
            return INVALID_MONTHLY_RETURN_SHARPE_V1;
        }
        squared_deviation_sum += squared_deviation;
        if !squared_deviation_sum.is_finite() {
            return INVALID_MONTHLY_RETURN_SHARPE_V1;
        }
    }

    let stddev = (squared_deviation_sum / (monthly_pnls.len() - 1) as f64).sqrt();
    if !stddev.is_finite() {
        return INVALID_MONTHLY_RETURN_SHARPE_V1;
    }
    if stddev <= 0.0 {
        return 0.0;
    }

    let sharpe = (mean / stddev) * 3.4641;
    if sharpe.is_finite() {
        sharpe
    } else {
        INVALID_MONTHLY_RETURN_SHARPE_V1
    }
}

#[inline]
fn sanitize_sharpe_v1(sharpe: f64) -> f64 {
    if sharpe == INVALID_MONTHLY_RETURN_SHARPE_V1 || sharpe.is_finite() {
        sharpe
    } else {
        0.0
    }
}

fn report_non_finite_candidate_metrics_v1(
    trade_count: usize,
    non_finite_metrics: &[&str],
    sharpe: f64,
) {
    if non_finite_metrics == ["sharpe"] && sharpe == INVALID_MONTHLY_RETURN_SHARPE_V1 {
        tracing::warn!(
            target: "neoethos_search::eval",
            trade_count,
            non_finite_metrics = ?non_finite_metrics,
            reason = "invalid_completed_month_return",
            "candidate rejected: completed-month percentage returns have invalid input shape, month-start equity, or non-finite arithmetic; preserving the negative-infinity Sharpe sentinel"
        );
    } else {
        tracing::warn!(
            target: "neoethos_search::eval",
            trade_count,
            non_finite_metrics = ?non_finite_metrics,
            reason = "non_finite_evaluation_metrics",
            "candidate produced non-finite evaluation metrics; inspect the named metrics and their inputs before attributing a financial-data or arithmetic cause"
        );
    }
}

#[cfg(test)]
mod monthly_return_sharpe_authority_v1_tests {
    use super::{
        INVALID_MONTHLY_RETURN_SHARPE_V1, completed_month_return_sharpe_v1, sanitize_sharpe_v1,
    };

    #[test]
    fn equal_money_pnl_with_different_start_equity_is_not_a_zero_return_sharpe() {
        let sharpe = completed_month_return_sharpe_v1(&[1_000.0, 1_000.0], &[100_000.0, 200_000.0]);
        assert!(sharpe.is_finite() && sharpe > 0.0);
    }

    #[test]
    fn invalid_denominator_is_a_fitness_rejection_sentinel_not_zero() {
        for invalid_start in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let sharpe = completed_month_return_sharpe_v1(&[100.0], &[invalid_start]);
            assert_eq!(sharpe, INVALID_MONTHLY_RETURN_SHARPE_V1);
            assert_eq!(sanitize_sharpe_v1(sharpe), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn valid_empty_or_single_period_keeps_the_existing_zero_sharpe_scope() {
        assert_eq!(completed_month_return_sharpe_v1(&[], &[]), 0.0);
        assert_eq!(completed_month_return_sharpe_v1(&[100.0], &[10_000.0]), 0.0);
    }
}

/// Per-session spread overrides. Values are spread in pips for each
/// liquidity window. When attached to `BacktestSettings`, the simulator
/// resolves the spread per bar from the bar's UTC hour-of-day instead
/// of using the scalar `spread_pips`. `None` → fall back to
/// `BacktestSettings::spread_pips` for backwards compatibility.
///
/// Buckets are intentionally coarse:
/// - `asian_pips`: 22:00-07:00 UTC (Tokyo, lower liquidity, wider spread)
/// - `overlap_pips`: 07:00-16:00 UTC (London + London/NY overlap, peak
///    liquidity, tightest spread)
/// - `late_ny_pips`: 16:00-22:00 UTC (NY tail, medium spread)
///
/// Real broker data is finer-grained but the 3-bucket approximation
/// already cuts the live-vs-backtest gap meaningfully because the
/// London/NY-overlap spread is typically 30-50% of the Asian spread.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SessionSpreadProfile {
    pub asian_pips: f64,
    pub overlap_pips: f64,
    pub late_ny_pips: f64,
}

/// Names of the three UTC session buckets, in `bucket_index` order.
pub const SESSION_BUCKET_NAMES: [&str; 3] = ["asian_22_07", "overlap_07_16", "late_ny_16_22"];

impl SessionSpreadProfile {
    /// Which of the three UTC buckets a timestamp falls in: `0` Asian (22–07),
    /// `1` London/NY overlap (07–16), `2` late NY (16–22).
    ///
    /// The SINGLE definition of the boundaries. The per-session trade census in
    /// `discovery.rs` calls this rather than re-deriving the hours, so the
    /// census can never report against different buckets than the cost model
    /// charges.
    #[inline]
    pub fn bucket_index(timestamp_ms: i64) -> usize {
        let hour = utc_hour_of_day(timestamp_ms);
        if (7..16).contains(&hour) {
            1
        } else if (16..22).contains(&hour) {
            2
        } else {
            0
        }
    }

    /// Resolve the bucket spread (pips) for a UTC unix-millisecond timestamp.
    pub(crate) fn spread_pips_at(self, timestamp_ms: i64) -> f64 {
        match Self::bucket_index(timestamp_ms) {
            1 => self.overlap_pips,
            2 => self.late_ny_pips,
            _ => self.asian_pips,
        }
    }
}

#[inline]
fn utc_hour_of_day(timestamp_ms: i64) -> u32 {
    let secs = timestamp_ms.div_euclid(1_000);
    let hour = secs.div_euclid(3_600).rem_euclid(24);
    hour as u32
}

#[derive(Debug, Clone)]
pub struct BacktestSettings {
    /// Immutable account balance for this evaluation run. Standalone legacy
    /// callers may omit it and use the configured runtime default.
    pub initial_equity_override: Option<f64>,
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub max_hold_bars: usize,
    /// Legacy strategy/config field retained for compatibility. The CPU
    /// bracket-only evaluators have no discretionary signal exit to delay:
    /// SL, TP, an active trail and the maximum holding deadline always apply.
    pub min_hold_bars: usize,
    pub max_trades_per_day: usize,
    pub gap_threshold_ms: i64,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    /// Pips of profit the trail must lock once it engages, measured from the
    /// entry.
    ///
    /// The other two knobs are multiples of the gene's stop distance, so the
    /// same pair locks a different amount for every gene: with `trigger -
    /// distance = 0.1`, a 20-pip stop protects 2 pips and a 10-pip stop only 1.
    /// An account is not risked in multiples of R, so the floor is absolute —
    /// once the trail is active the stop never sits closer to entry than this.
    pub trailing_min_lock_pips: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub kill_zones_enabled: bool,
    /// Optional session-aware spread override. When `Some`, `spread_pips`
    /// is ignored and the simulator looks up the per-bar spread from
    /// the bar's UTC timestamp. Requires bar timestamps to be present;
    /// falls back to `spread_pips` when timestamps are empty or zero.
    pub session_spread_profile: Option<SessionSpreadProfile>,

    /// **Phase C (2026-05-28)** — broker-supplied overnight SWAP and
    /// cross-currency conversion fee. Flow:
    ///   - `SymbolMetadata.daily_swap_{long,short}_pips` (cTrader
    ///     `ProtoOASymbol::swap_long/short` when calc-type is `PIPS`)
    ///   - copied into `MarketCostProfile` by
    ///     `genetic::strategy_gene::infer_market_cost_profile`
    ///   - copied here by the BacktestSettings constructor.
    ///
    /// Semantics: at each trade exit, the eval kernel subtracts
    ///   `swap_{long|short}_pips × overnight_days × pip_value_per_lot`
    /// from the trade PnL. `overnight_days` = count of UTC midnight
    /// crossings between entry and exit timestamps; 0 means the
    /// trade was day-traded (no swap charge).
    ///
    /// Defaults to `0.0` (no charge) when the broker hasn't supplied
    /// the value. This matches the pre-Phase-C silent behaviour but
    /// emits a warn in `infer_market_cost_profile` to surface the
    /// missing-broker-data path.
    pub swap_long_pips_per_day: f64,
    pub swap_short_pips_per_day: f64,
    /// Effective research conversion-fee fraction (0.005 = 0.5%). Zero when
    /// quote and account currencies match. The explicit V1 research policy
    /// debits abs(realized price gross) * rate, excluding commission and swap;
    /// it does not estimate a broker's exact settlement rounding.
    pub pnl_conversion_fee_rate: f64,

    // ── Risk-based, confidence-scaled position sizing (Phase 1, 2026-06-05) ──
    //
    // When `risk_based_sizing` is true AND the per-bar confidence slice
    // passed to `fast_evaluate_strategy_core` is non-empty, the simulator
    // sizes each position at entry so that a full stop-loss loss is
    // approximately `risk_pct × equity_at_entry`, where
    //   risk_pct = risk_per_trade_min
    //            + (risk_per_trade_max - risk_per_trade_min)
    //              * min(conf / high_quality_confidence, 1.0)
    // and `conf` is the clamped [0,1] confidence at the entry signal bar.
    // The resulting `pos_lots` is captured at entry and multiplies EVERY
    // realized PnL, cost, float-PnL, and carry/fee for that trade — so the
    // sizing compounds with current equity. When `risk_based_sizing` is
    // false OR no confidence slice is supplied, `pos_lots` is forced to
    // 1.0, reproducing the legacy fixed-1-lot behaviour exactly.
    /// Enable risk-based, confidence-scaled position sizing on the CPU
    /// backtest path. Default `true`. GPU path is unchanged (Phase 2).
    pub risk_based_sizing: bool,
    /// Lower bound of the per-trade risk fraction (e.g. 0.005 = 0.5%).
    pub risk_per_trade_min: f64,
    /// Upper bound of the per-trade risk fraction (e.g. 0.03 = 3%),
    /// reached at confidence >= `high_quality_confidence`.
    pub risk_per_trade_max: f64,
    /// Confidence at/above which a trade is sized at `risk_per_trade_max`.
    pub high_quality_confidence: f64,

    /// **Adaptive stops (2026-07-23)** — volatility-scaled per-entry SL/TP.
    /// `adaptive_base_pips` is the per-BAR base stop distance in pips (the
    /// dataset's vol/tail distance at vol_mult = 1), shared across the whole
    /// population as a single `Arc`. When it is `Some` AND `adaptive_vol_mult >
    /// 0`, a trade opened at bar i takes `sl = adaptive_vol_mult * base[i]` and
    /// applies the gene's evolved `tp_pips / sl_pips` reward:risk ratio. The
    /// `adaptive_rr` scalar is only a legacy fallback for an invalid stored pair.
    /// Thus the stop scales with volatility at entry (tight in calm markets,
    /// wider when choppy) without collapsing every gene to the same exit;
    /// risk-based sizing uses the same per-entry SL so a
    /// wider stop sizes a smaller position at constant risk. `adaptive_vol_mult`
    /// is set PER GENE (the gene's searchable `stop_vol_mult`) so the base series
    /// is computed once per combo and each gene just scales it — no per-gene
    /// allocation on the hot path. When base is `None` or the mult is `0` the
    /// scalar `sl_pips`/`tp_pips` fixed path runs, byte-for-byte as before.
    pub adaptive_base_pips: Option<std::sync::Arc<[f64]>>,
    pub adaptive_vol_mult: f64,
    pub adaptive_rr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BacktestMetrics {
    pub net_profit: f64,
    pub sharpe: f64,
    pub peak_equity: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub monthly_target_hit_rate: f64,
    pub trade_count: usize,
    pub consistency: f64,
    pub max_daily_drawdown: f64,
}

/// **Index-7 slot = `monthly_target_hit_rate`** (scoring_version 3, 2026-06-06).
///
/// History, so the name is never mistaken for "spare room": under F-001 (2026-05-25)
/// this slot was *reserved* and pinned to 0.0 — an earlier revision used it for
/// `average_trade_pnl`, which was dropped while the `[f64; 11]` shape was kept so the
/// GPU kernel's per-gene output stride (11 floats/gene) stayed intact.
///
/// It is no longer reserved. The RAW output of `fast_evaluate_strategy_core` carries
/// `monthly_target_hit_rate` (fraction of months hitting the operator's >=4% bar) in
/// slot 7 — the consistency signal [`crate::scoring::ga_fitness`] optimises toward, and
/// its dominant term. The GA fitness reads the raw eval array directly (see
/// `genetic::evolution_math::apply_metrics`).
///
/// BOTH producers write it: the CPU eval (`fast_evaluate_strategy_core`, this file) and
/// the GPU lane, which is ENABLED (`PHASE1_GPU_SIZING_PORTED = true`, this file) —
/// cubecl (`cubecl_eval.rs`), prototype B (`prototype_b_population.cu` → `values[7]`)
/// and prototype C (`prototype_c_engine/device.rs` → `metric_base + 7`). Any new
/// producer MUST write this slot or the dominant reward silently reads 0.0.
///
/// [`BacktestMetrics`] models and persists this field explicitly, and its array
/// conversions preserve all 11 slots. Code that hand-rolls a `[f64; 11]` to feed
/// `ga_fitness` must still set slot 7 to the hit-rate (0.0 disables the dominant
/// consistency reward).
pub const BACKTEST_METRICS_MONTHLY_TARGET_HIT_RATE_INDEX: usize = 7;

impl BacktestMetrics {
    /// Index of `monthly_target_hit_rate` in the array form. See
    /// [`BACKTEST_METRICS_MONTHLY_TARGET_HIT_RATE_INDEX`] for history and producers.
    pub const MONTHLY_TARGET_HIT_RATE_INDEX: usize = BACKTEST_METRICS_MONTHLY_TARGET_HIT_RATE_INDEX;

    pub fn from_metric_array(metrics: [f64; 11]) -> Self {
        Self {
            net_profit: metrics[0],
            sharpe: metrics[1],
            peak_equity: metrics[2],
            max_drawdown: metrics[3],
            win_rate: metrics[4],
            profit_factor: metrics[5],
            expectancy: metrics[6],
            monthly_target_hit_rate: metrics[Self::MONTHLY_TARGET_HIT_RATE_INDEX],
            trade_count: if metrics[8].is_finite() && metrics[8] > 0.0 {
                metrics[8].round() as usize
            } else {
                0
            },
            consistency: metrics[9],
            max_daily_drawdown: metrics[10],
        }
    }

    pub fn to_metric_array(self) -> [f64; 11] {
        [
            self.net_profit,
            self.sharpe,
            self.peak_equity,
            self.max_drawdown,
            self.win_rate,
            self.profit_factor,
            self.expectancy,
            self.monthly_target_hit_rate,
            self.trade_count as f64,
            self.consistency,
            self.max_daily_drawdown,
        ]
    }
}

impl From<[f64; 11]> for BacktestMetrics {
    fn from(metrics: [f64; 11]) -> Self {
        Self::from_metric_array(metrics)
    }
}

impl From<BacktestMetrics> for [f64; 11] {
    fn from(metrics: BacktestMetrics) -> Self {
        metrics.to_metric_array()
    }
}

impl Default for BacktestSettings {
    fn default() -> Self {
        // GROUP C remediation (operator directive 2026-05-25): the
        // previous code called `infer_market_cost_profile("", "", ...)`
        // which silently fell back to EURUSD/USD. We now emit NaN
        // sentinels. `Default` is now structural/test scaffolding only:
        // production financial entry points fail at the broker-truth boundary
        // before any caller can turn these fields into trades or metrics.
        Self {
            initial_equity_override: None,
            sl_pips: 20.0,
            tp_pips: 40.0,
            max_hold_bars: 0,
            min_hold_bars: 0,
            max_trades_per_day: 0,
            // Four days. Long enough to sit through an FX weekend, which is
            // about two and a half and is a normal thing to hold through; short
            // enough to catch a hole in the data.
            //
            // This was 0 — detection off — so a position open before a hole was
            // carried across it as if the market had not moved, and its stop
            // and target were then tested against prices from the far side.
            // That mattered before anything was dropped (any missing history
            // does it) and matters more now that non-positive bars are removed
            // on read: December 2014 is missing from every series in this
            // store, twelve days of it on H1.
            gap_threshold_ms: 4 * 24 * 60 * 60 * 1000,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            trailing_min_lock_pips: 2.0,
            pip_value: f64::NAN,
            spread_pips: f64::NAN,
            commission_per_trade: f64::NAN,
            pip_value_per_lot: f64::NAN,
            kill_zones_enabled: false,
            session_spread_profile: None,
            // **Phase C (2026-05-28)**: swap + conversion-fee default
            // to 0.0 (no charge). NaN-sentinel pattern from the cost
            // fields above is NOT applied here because (a) it would
            // collapse every backtest that doesn't have broker swap
            // data into NaN fitness — a regression for symbols with
            // no overnight exposure — and (b) the swap term is a
            // CHARGE: 0.0 produces a conservative (rosy) PnL, which
            // the existing F-029 LAST-RESORT warn in
            // `infer_market_cost_profile` already flags. When broker
            // data exists, `for_symbol(...)` overrides these to the
            // real values.
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            // Risk-based sizing defaults (Phase 1). `risk_based_sizing`
            // is ON by default but only takes effect when a non-empty
            // confidence slice is supplied to the evaluator; callers that
            // pass `&[]` (legacy fixed-1-lot) are unaffected.
            risk_based_sizing: true,
            risk_per_trade_min: 0.005,
            risk_per_trade_max: 0.03,
            high_quality_confidence: 0.65,
            // Adaptive stops OFF by default (mult 0) → the scalar sl_pips/tp_pips path.
            adaptive_base_pips: None,
            adaptive_vol_mult: 0.0,
            adaptive_rr: 2.0,
        }
    }
}

/// Typed replacement for the legacy `NEOETHOS_BOT_BACKTEST_*` env vars that
/// previously changed canonical backtest math (`initial_equity`,
/// `month_capacity`) on every metric evaluation. The struct is the single
/// place these values live; production callers install them once via
/// [`install_backtest_runtime_overrides`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct BacktestRuntimeOverrides {
    /// Starting equity used for canonical backtest PnL accounting. Must be
    /// strictly positive.
    pub initial_equity: f64,
    /// Maximum number of monthly PnL buckets retained for consistency math.
    /// Must be non-zero.
    pub month_capacity: usize,
    /// Legacy Rayon width cap retained in the sealed runtime record. Production
    /// startup folds it into the immutable process execution budget; it never
    /// creates or mutates Rayon's process-global pool.
    ///
    /// **F-695 closure (2026-05-25 — F-CORE3)**: the retired environment
    /// controls were first consolidated here. The lease-bound executor is now
    /// the only authority that turns the resolved cap into worker threads.
    pub rayon_threads: Option<usize>,
}

impl Default for BacktestRuntimeOverrides {
    fn default() -> Self {
        Self {
            initial_equity: 100_000.0,
            month_capacity: 240,
            rayon_threads: None,
        }
    }
}

impl BacktestRuntimeOverrides {
    // `from_env()` DELETED 2026-08-10. It read
    // `NEOETHOS_BOT_BACKTEST_INITIAL_EQUITY`,
    // `NEOETHOS_BOT_BACKTEST_MAX_MONTH_BUCKETS`, `NEOETHOS_BOT_RUST_THREADS`
    // and `RAYON_NUM_THREADS` — all four now typed on
    // `models.backtest_runtime`. Two of them changed BACKTEST ARITHMETIC
    // (`initial_equity` feeds the absolute 0..100 lot clamp; `month_capacity`
    // sizes metric slot 7, weighted 0.45 in the prop-firm objective), which is
    // the last kind of value that should have been settable by an export.
    //
    // ⚠ `RAYON_NUM_THREADS` is still read by `neoethos-models`
    // (`tree_models/config.rs:119`, `cpu_threads_hint`). It is retired HERE,
    // not workspace-wide; the asymmetry is recorded in the retired-env table.

    /// Config-driven constructor (was the `NEOETHOS_BOT_BACKTEST_*` env
    /// vars). Numeric fields are validated (equity > 0, capacity > 0,
    /// threads > 0) exactly like the env reader. A
    /// `backtest_from_settings_default_matches_env_default` test guarantees
    /// a fresh `Settings` reproduces [`Self::default`].
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.backtest_runtime;
        let d = Self::default();
        Self {
            initial_equity: if c.initial_equity.is_finite() && c.initial_equity > 0.0 {
                c.initial_equity
            } else {
                d.initial_equity
            },
            month_capacity: if c.month_capacity > 0 {
                c.month_capacity
            } else {
                d.month_capacity
            },
            rayon_threads: c.rayon_threads.filter(|v| *v > 0),
        }
    }
}

static BACKTEST_RUNTIME_OVERRIDES: OnceLock<BacktestRuntimeOverrides> = OnceLock::new();

/// Install process-wide backtest runtime overrides. Returns `Err(existing)`
/// if overrides were already installed earlier (the first install wins).
pub fn install_backtest_runtime_overrides(
    overrides: BacktestRuntimeOverrides,
) -> Result<(), BacktestRuntimeOverrides> {
    BACKTEST_RUNTIME_OVERRIDES.set(overrides)
}

/// Config-driven install — reads the backtest knobs from the single
/// `Settings` instead of the environment. Idempotent.
///
/// This is also where the retired-environment report fires: every production
/// binary reaches this function through
/// `install_search_runtime_overrides_from_settings` at startup, so it is the
/// one place guaranteed to run once, with `Settings` in hand, before any
/// evaluation happens.
pub fn install_backtest_runtime_overrides_from_settings(s: &neoethos_core::Settings) {
    crate::execution_profile::report_retired_env_vars();
    let resolved = BacktestRuntimeOverrides::from_settings(s);
    report_equity_denominator_disagreement(s, resolved.initial_equity);
    let _ = BACKTEST_RUNTIME_OVERRIDES.set(resolved);
}

/// Announce a distinct standalone default without falsely describing it as the
/// Discovery account: Discovery snapshots `risk.initial_balance` into its own
/// immutable evaluation settings and uses it for sizing and validation alike.
fn report_equity_denominator_disagreement(s: &neoethos_core::Settings, initial_equity: f64) {
    let account = s.risk.initial_balance;
    if !account.is_finite() || account <= 0.0 {
        tracing::error!(
            target: "neoethos_search::cost_model",
            configured_account_balance = account,
            search_initial_equity = initial_equity,
            "risk.initial_balance is not usable; Discovery must reject this account balance"
        );
        return;
    }
    if (account - initial_equity).abs() <= f64::EPSILON * account.abs().max(1.0) {
        return;
    }
    tracing::info!(
        target: "neoethos_search::cost_model",
        account_balance = account,
        search_initial_equity = initial_equity,
        ratio = initial_equity / account,
        "standalone backtest default differs from the account; Discovery snapshots \
         risk.initial_balance for both Search sizing and account-risk validation"
    );
}

/// Returns the currently installed backtest runtime overrides, or the
/// deterministic defaults when no install has happened.
pub fn current_backtest_runtime_overrides() -> BacktestRuntimeOverrides {
    BACKTEST_RUNTIME_OVERRIDES
        .get()
        .copied()
        .unwrap_or_default()
}

impl BacktestSettings {
    pub fn initial_equity(&self) -> f64 {
        self.initial_equity_override
            .unwrap_or_else(|| current_backtest_runtime_overrides().initial_equity)
    }

    pub fn month_capacity(&self) -> usize {
        current_backtest_runtime_overrides().month_capacity
    }
}

/// Announce, once, that `month_capacity` is shorter than the frame and that
/// metric slot 7 is therefore being computed on a prefix of the history.
///
/// 2026-08-10. Before this, the overflow was dropped at two sites
/// (`simulate_trades_core`'s month roll and the slot-7 loop's `.min()`) and
/// reported at neither, so a truncated consistency score was indistinguishable
/// from an honest one. This does not change any number — it makes the number
/// legible. The refusal belongs in the config validator, which is routed to
/// `config.rs`; a per-gene evaluator cannot refuse without aborting the run.
fn report_month_capacity_overflow(month_capacity: usize, month_ptr: i64) {
    let months_seen = (month_ptr.max(0) as usize).saturating_add(1);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let coverage = if months_seen > 0 {
            (month_capacity as f64 / months_seen as f64) * 100.0
        } else {
            100.0
        };
        tracing::error!(
            target: "neoethos_search::eval",
            month_capacity,
            months_in_frame_so_far = months_seen,
            coverage_pct = format!("{coverage:.1}"),
            "models.backtest_runtime.month_capacity ({month_capacity}) is SHORTER than the \
             months this frame spans ({months_seen}). Months past the cap are dropped, so \
             metric slot 7 (monthly_target_hit_rate — weighted 0.45, the dominant term of \
             the prop-firm objective) is scored on only {coverage:.1}% of the history and \
             still returns a plausible number. Raise month_capacity to at least the months \
             in your dataset."
        );
    });
}

/// Apply signed carry and the versioned research conversion-fee debit.
/// `pnl_after_costs` retains the existing spread/commission arithmetic.
/// `price_gross` includes executable-side spread but excludes commission/swap.
/// `in_pos` is +1 for long, −1 for short. `entry_ts_ms` / `exit_ts_ms`
/// are millisecond timestamps; pass 0 when timestamps are unavailable
/// and the swap charge should be skipped (back-compat with pre-Phase-C
/// callers that don't carry timestamps).
///
/// Math:
///   overnight_days = max(exit_ts − entry_ts, 0) / 86_400_000  (fractional)
///   swap_pips_per_day = swap_long if long else swap_short
///     ↑ broker sign convention: positive = credit, negative = charge
///   pnl_with_carry = pnl_after_costs + swap_pips_per_day × overnight_days
///                      × pip_value_per_lot
///   net_pnl = pnl_with_carry - abs(price_gross) × effective_conversion_fee_rate
///
/// With both swap fields = 0.0 and conversion fee = 0.0 this is the
/// identity, matching the pre-Phase-C kernel exactly.
#[inline]
fn apply_carry_and_fee(
    pnl_after_costs: f64,
    price_gross: f64,
    in_pos: i8,
    entry_ts_ms: i64,
    exit_ts_ms: i64,
    settings: &BacktestSettings,
) -> f64 {
    let overnight_days = if exit_ts_ms > entry_ts_ms && entry_ts_ms > 0 {
        (exit_ts_ms - entry_ts_ms) as f64 / 86_400_000.0
    } else {
        0.0
    };
    let swap_pips_per_day = if in_pos == 1 {
        settings.swap_long_pips_per_day
    } else {
        settings.swap_short_pips_per_day
    };
    let swap_credit = swap_pips_per_day * overnight_days * settings.pip_value_per_lot;
    let pnl_with_carry = pnl_after_costs + swap_credit;
    debit_research_conversion_fee(
        pnl_with_carry,
        price_gross,
        settings.pnl_conversion_fee_rate,
    )
}

#[inline]
fn debit_research_conversion_fee(pnl_with_carry: f64, price_gross: f64, rate: f64) -> f64 {
    use neoethos_core::research_conversion_fee::{
        ResearchPnlConversionFeePolicyV1, validate_conversion_fee_rate_v1,
    };
    validate_conversion_fee_rate_v1(rate).expect("invalid research conversion fee rate");
    if rate == 0.0 {
        // Do not subtract even +0: preserve every bit of the fee-zero path.
        return pnl_with_carry;
    }
    let fee = ResearchPnlConversionFeePolicyV1::AbsoluteRealizedPriceGrossDebitV1
        .debit(price_gross, rate)
        .expect("invalid research conversion fee gross");
    let net = pnl_with_carry - fee;
    assert!(
        net.is_finite(),
        "research conversion fee produced non-finite net PnL"
    );
    net
}

/// Both PnL inputs are already scaled by entry-captured lots. Scale only the
/// per-lot swap here; never multiply the conversion fee by lots a second time.
#[inline]
fn apply_carry_and_fee_scaled(
    pnl_after_costs_scaled: f64,
    price_gross_scaled: f64,
    pos_lots: f64,
    in_pos: i8,
    entry_ts_ms: i64,
    exit_ts_ms: i64,
    settings: &BacktestSettings,
) -> f64 {
    if pos_lots == 1.0 {
        // Exact legacy path — no extra arithmetic, no rounding drift.
        return apply_carry_and_fee(
            pnl_after_costs_scaled,
            price_gross_scaled,
            in_pos,
            entry_ts_ms,
            exit_ts_ms,
            settings,
        );
    }
    let overnight_days = if exit_ts_ms > entry_ts_ms && entry_ts_ms > 0 {
        (exit_ts_ms - entry_ts_ms) as f64 / 86_400_000.0
    } else {
        0.0
    };
    let swap_pips_per_day = if in_pos == 1 {
        settings.swap_long_pips_per_day
    } else {
        settings.swap_short_pips_per_day
    };
    // Swap term scales with size (it is a per-lot cash flow).
    let swap_credit = swap_pips_per_day * overnight_days * settings.pip_value_per_lot * pos_lots;
    let pnl_with_carry = pnl_after_costs_scaled + swap_credit;
    debit_research_conversion_fee(
        pnl_with_carry,
        price_gross_scaled,
        settings.pnl_conversion_fee_rate,
    )
}

/// Per-ENTRY `(sl_pips, tp_pips)` for the position opening at bar `i`. When the
/// gene's `adaptive_vol_mult > 0` and a base vol-distance series is present, the
/// stop scales with volatility at `i` (`sl = mult * base[i]`) while the target
/// keeps the gene's evolved `tp_pips / sl_pips` ratio;
/// otherwise the scalar fixed `sl_pips`/`tp_pips`. An unavailable cell in an
/// enabled adaptive series means no entry, not permission to trade fixed pips.
#[inline]
fn entry_sl_tp_pips(settings: &BacktestSettings, i: usize) -> Option<(f64, f64)> {
    crate::stop_target::resolve_entry_stop_target_pips(
        settings.sl_pips,
        settings.tp_pips,
        settings.adaptive_vol_mult,
        settings
            .adaptive_base_pips
            .as_ref()
            .and_then(|base| base.get(i))
            .copied(),
        settings.adaptive_base_pips.is_some(),
        settings.adaptive_rr,
    )
}

/// Risk-based, confidence-scaled lot size for a single trade entry.
///
/// Returns the constant `pos_lots` multiplier applied to every PnL / cost /
/// float / carry term for the trade. With `risk_based_sizing == false` or an
/// empty `confidences` slice the caller forces `pos_lots = 1.0` (legacy
/// fixed-1-lot) — this function is only consulted on the risk-based path.
///
/// Math (see `BacktestSettings` risk-sizing fields):
///   conf     = confidence at the entry signal bar, clamped [0,1]
///   risk_pct = risk_min + (risk_max - risk_min)
///              * min(conf / high_quality_confidence, 1.0)
///   eff_sl   = sl_pips                           // actual, positive entry stop
///   pos_lots = if equity > 0 {
///                  (risk_pct * equity) / (eff_sl * pip_value_per_lot)
///              } else { 0.0 }
///   pos_lots = pos_lots.clamp(0.0, 100.0)         // sane leverage backstop
///
/// Net effect: a full-SL loss ≈ `risk_pct × equity`, a TP win ≈
/// `risk_pct × equity × (tp/sl)`.
#[inline]
fn risk_based_pos_lots(
    conf: f64,
    equity: f64,
    eff_sl_pips: f64,
    settings: &BacktestSettings,
) -> f64 {
    let conf = conf.clamp(0.0, 1.0);
    let risk_min = settings.risk_per_trade_min;
    let risk_max = settings.risk_per_trade_max;
    // Guard the confidence normaliser against a zero/negative/non-finite
    // high_quality_confidence so we never divide by ~0.
    let hq = settings.high_quality_confidence;
    let conf_scale = if hq.is_finite() && hq > 0.0 {
        (conf / hq).min(1.0)
    } else {
        // Degenerate config: treat any signal as max-quality.
        1.0
    };
    let risk_pct = risk_min + (risk_max - risk_min) * conf_scale;
    // Size against the actual ENTRY stop, including legitimate sub-pip stops.
    // The lot cap below limits exposure; changing the risk denominator to a
    // fictitious one-pip stop makes the reported risk disagree with the trade.
    let pip_value_per_lot = settings.pip_value_per_lot;
    let denom = eff_sl_pips * pip_value_per_lot;
    let pos_lots = if equity.is_finite()
        && equity > 0.0
        && eff_sl_pips.is_finite()
        && eff_sl_pips > 0.0
        && pip_value_per_lot.is_finite()
        && pip_value_per_lot > 0.0
        && denom.is_finite()
        && denom > 0.0
    {
        (risk_pct * equity) / denom
    } else {
        0.0
    };
    if pos_lots.is_finite() {
        pos_lots.clamp(0.0, 100.0)
    } else {
        0.0
    }
}

/// One protective-exit policy for CPU fitness and the detailed trade ledger.
/// Existing protection is checked before this bar can ratchet the trail, and
/// SL wins an ambiguous OHLC bar that crosses both SL and TP. A minimum hold
/// must never suppress protection or extend the maximum holding deadline.
#[inline]
#[allow(clippy::too_many_arguments)]
fn protective_exit_price(
    side: i8,
    entry_px: f64,
    sl_pips: f64,
    tp_pips: f64,
    trail_px: &mut f64,
    hi: f64,
    lo: f64,
    close: f64,
    bars_held: usize,
    pip: f64,
    settings: &BacktestSettings,
) -> Option<f64> {
    if side == 1 {
        let mut sl = entry_px - (sl_pips * pip);
        let tp = entry_px + (tp_pips * pip);
        // Only PRIOR bars may have moved the stop protecting this bar.
        // Zero is the existing unset sentinel for both position directions.
        if settings.trailing_enabled && *trail_px > 0.0 && *trail_px > sl {
            sl = *trail_px;
        }
        if lo <= sl {
            return Some(sl);
        }
        if hi >= tp {
            return Some(tp);
        }
        if settings.trailing_enabled {
            let mv = hi - entry_px;
            if mv >= (settings.trailing_be_trigger_r * sl_pips * pip) {
                let locked = entry_px + settings.trailing_min_lock_pips * pip;
                let candidate =
                    (hi - (settings.trailing_atr_multiplier * sl_pips * pip)).max(locked);
                if *trail_px == 0.0 || candidate > *trail_px {
                    *trail_px = candidate;
                }
            }
        }
    } else {
        let mut sl = entry_px + (sl_pips * pip);
        let tp = entry_px - (tp_pips * pip);
        if settings.trailing_enabled && *trail_px > 0.0 && *trail_px < sl {
            sl = *trail_px;
        }
        if hi >= sl {
            return Some(sl);
        }
        if lo <= tp {
            return Some(tp);
        }
        if settings.trailing_enabled {
            let mv = entry_px - lo;
            if mv >= (settings.trailing_be_trigger_r * sl_pips * pip) {
                let locked = entry_px - settings.trailing_min_lock_pips * pip;
                let candidate =
                    (lo + (settings.trailing_atr_multiplier * sl_pips * pip)).min(locked);
                if *trail_px == 0.0 || candidate < *trail_px {
                    *trail_px = candidate;
                }
            }
        }
    }
    // Timeout is a close-of-bar exit, after the intrabar protective barriers.
    (settings.max_hold_bars > 0 && bars_held >= settings.max_hold_bars).then_some(close)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fast_evaluate_strategy_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    confidences: &[f64],
    month_idx: &[i64],
    day_idx: &[i64],
    timestamps: &[i64],
    settings: &BacktestSettings,
) -> [f64; 11] {
    evaluate_strategy_with_ledger_core::<false>(
        close,
        high,
        low,
        signals,
        confidences,
        month_idx,
        day_idx,
        timestamps,
        settings,
        &mut Vec::new(),
        None,
    )
}

/// The CPU fitness and detailed ledger share the same position lifecycle.
/// `RECORD_TRADES = false` removes ledger/excursion recording at compile time;
/// population search does not allocate a trade history for each candidate.
#[allow(clippy::too_many_arguments)]
fn evaluate_strategy_with_ledger_core<const RECORD_TRADES: bool>(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    confidences: &[f64],
    month_idx: &[i64],
    day_idx: &[i64],
    timestamps: &[i64],
    settings: &BacktestSettings,
    trades: &mut Vec<Trade>,
    mut netted: Option<&mut netted_bar_research::NettedBarCoreState<'_>>,
) -> [f64; 11] {
    // Reports itself on first call — see `eval_telemetry`.
    neoethos_core::research_conversion_fee::validate_conversion_fee_rate_v1(
        settings.pnl_conversion_fee_rate,
    )
    .expect("invalid research conversion fee rate");
    struct TelemetryGuard(&'static str, usize, std::time::Instant);
    impl Drop for TelemetryGuard {
        fn drop(&mut self) {
            crate::eval_telemetry::record(self.0, self.1, self.2.elapsed());
        }
    }
    let _telemetry = TelemetryGuard(
        if RECORD_TRADES {
            "eval::simulate_trades_core"
        } else {
            "eval::fast_evaluate_strategy_core"
        },
        1,
        std::time::Instant::now(),
    );
    let n = close.len();
    if n == 0 {
        return [0.0; 11];
    }

    // Risk-based sizing is active only when explicitly enabled AND a
    // per-bar confidence slice is supplied. Otherwise `pos_lots` stays
    // 1.0 for every trade — exact legacy fixed-1-lot behaviour, which
    // keeps existing callers (and the `&[]` callers below) unchanged.
    let use_risk_sizing = settings.risk_based_sizing && !confidences.is_empty();
    // Captured at each entry; constant for the life of an open position.
    let mut pos_lots: f64 = 1.0;
    // Per-entry SL/TP in pips (adaptive-per-entry when enabled, else the scalar
    // `sl_pips`/`tp_pips`). Captured at entry, held for the trade's life. Init to
    // the scalar so an open-before-first-entry read (impossible — only used while
    // in_pos) is still well-defined.
    let mut pos_sl_pips: f64 = settings.sl_pips;
    let mut pos_tp_pips: f64 = settings.tp_pips;

    let initial_equity = settings.initial_equity();
    let month_capacity = netted
        .as_ref()
        .map(|state| state.month_capacity)
        .unwrap_or_else(|| settings.month_capacity());

    let mut equity = initial_equity;
    let mut peak_equity = initial_equity;
    let mut max_dd = 0.0;
    let mut trade_count = 0usize;
    let mut wins = 0usize;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;

    let mut last_month = -1i64;
    let mut current_month_pnl = 0.0;
    let mut monthly_pnls = vec![0.0; month_capacity];
    let mut month_ptr = -1i64;
    // Parallel to `monthly_pnls`: equity at the START of each completed month, so we
    // can compute each month's RETURN % (pnl / month-start-equity) for the
    // monthly_target_hit_rate metric (slot 7). Compounding makes total net a
    // poor consistency signal; per-month return % is scale-invariant.
    let mut month_start_equities = vec![initial_equity; month_capacity];
    let mut current_month_start_equity = initial_equity;

    let mut last_day = -1i64;
    let mut day_peak = equity;
    let mut day_low = equity;
    let mut max_daily_dd = 0.0;
    let mut day_trade_count = 0usize;

    let mut in_pos = 0i8;
    let mut entry_px = 0.0;
    let mut entry_idx = -1i64;
    let mut trail_px = 0.0;
    let mut mfe_money = 0.0_f64;
    let mut mae_money = 0.0_f64;

    let pip = if settings.pip_value.abs() < 1e-12 {
        1e-12
    } else {
        settings.pip_value
    };
    let scalar_half_spread_px = settings.spread_pips * 0.5 * pip;
    let scalar_half_spread_cost = settings.spread_pips * 0.5 * settings.pip_value_per_lot;

    let use_timestamps = !timestamps.is_empty() && timestamps.len() == n;
    let session_profile = settings.session_spread_profile.filter(|_| use_timestamps);

    for i in 1..n {
        let timestamp = timestamps.get(i).copied().unwrap_or(0);
        let (session_close, session_block_entry) =
            weekend_session_policy(timestamp, settings.kill_zones_enabled && use_timestamps);
        // Per-bar spread cost. When `session_spread_profile` is unset
        // these collapse to the loop-invariant scalar, which the
        // optimiser is free to hoist; the explicit per-bar form keeps
        // the code uniform whether the profile is on or off.
        let (half_spread_px, half_spread_cost) = match session_profile {
            Some(profile) if timestamp > 0 => {
                let s = profile.spread_pips_at(timestamps[i]);
                (s * 0.5 * pip, s * 0.5 * settings.pip_value_per_lot)
            }
            _ => (scalar_half_spread_px, scalar_half_spread_cost),
        };
        let m_val = *month_idx.get(i).unwrap_or(&last_month);
        if m_val != last_month {
            if last_month != -1 {
                month_ptr += 1;
                if month_ptr < month_capacity as i64 {
                    monthly_pnls[month_ptr as usize] = current_month_pnl;
                    month_start_equities[month_ptr as usize] = current_month_start_equity;
                } else {
                    // OVERFLOW. Every month past the capacity is DROPPED, and
                    // it used to be dropped in silence — here and again in the
                    // slot-7 loop. That matters more than a truncated array
                    // usually would: `monthly_pnls` produces
                    // `monthly_target_hit_rate`, metric slot 7, which the
                    // prop-firm objective multiplies by 0.45 (`named.rs:161`)
                    // — the dominant term. A capacity below the months the
                    // frame spans therefore scores every gene on a PREFIX of
                    // its history and returns a number that looks fine.
                    //
                    // The UI advertises `min: Some(12)` for this field while
                    // calling it a RAM cap, so the documented minimum scores a
                    // ten-year dataset on its first twelve months.
                    //
                    // Reported once per process (this is a per-gene hot path),
                    // at ERROR, with both numbers.
                    report_month_capacity_overflow(month_capacity, month_ptr);
                }
            }
            current_month_pnl = 0.0;
            current_month_start_equity = equity; // equity carried in = start of the new month
            last_month = m_val;
        }

        let d_val = *day_idx.get(i).unwrap_or(&last_day);
        if d_val != last_day {
            if last_day != -1 {
                finalize_daily_drawdown_segment(day_peak, day_low, &mut max_daily_dd);
            }
            last_day = d_val;
            day_peak = equity;
            day_low = equity;
            day_trade_count = 0;
        }

        if RECORD_TRADES && in_pos != 0 {
            // Retain the existing bar-resolution excursion convention. This is
            // not tick-resolved evidence about the ordering of intrabar prices.
            let (favorable, adverse) = if in_pos == 1 {
                (high[i] - entry_px, entry_px - low[i])
            } else {
                (entry_px - low[i], high[i] - entry_px)
            };
            mfe_money = mfe_money.max(favorable / pip * settings.pip_value_per_lot * pos_lots);
            mae_money = mae_money.max(adverse / pip * settings.pip_value_per_lot * pos_lots);
        }

        // Gap detection: force-exit open position when market gap exceeds threshold
        if in_pos != 0 && use_timestamps && settings.gap_threshold_ms > 0 {
            let ts_prev = timestamps[i - 1];
            let ts_curr = timestamps[i];
            if ts_curr > ts_prev && (ts_curr - ts_prev) >= settings.gap_threshold_ms {
                // Force exit at current close (proxy for gap open price).
                // Risk-based sizing: scale the price-derived PnL and the
                // commission+spread cost by the entry-captured `pos_lots`.
                let pnl = if in_pos == 1 {
                    (close[i] - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - close[i]) / pip * settings.pip_value_per_lot
                };
                let price_gross = pnl * pos_lots - half_spread_cost * pos_lots;
                let pnl =
                    pnl * pos_lots - (settings.commission_per_trade + half_spread_cost) * pos_lots;
                // Preserve old cost/swap arithmetic when fee is zero, but carry
                // the commission-free price gross separately for a nonzero fee.
                let entry_ts_ms = if use_timestamps && entry_idx >= 0 {
                    timestamps.get(entry_idx as usize).copied().unwrap_or(0)
                } else {
                    0
                };
                let exit_ts_ms = if use_timestamps {
                    timestamps.get(i).copied().unwrap_or(0)
                } else {
                    0
                };
                let pnl = apply_carry_and_fee_scaled(
                    pnl,
                    price_gross,
                    pos_lots,
                    in_pos,
                    entry_ts_ms,
                    exit_ts_ms,
                    settings,
                );
                if RECORD_TRADES {
                    record_closed_cpu_trade(
                        trades,
                        entry_ts_ms,
                        exit_ts_ms,
                        pnl,
                        initial_equity,
                        pos_sl_pips * settings.pip_value_per_lot * pos_lots,
                        mfe_money,
                        mae_money,
                    );
                }
                equity += pnl;
                current_month_pnl += pnl;
                trade_count += 1;
                if pnl > 0.0 {
                    wins += 1;
                    gross_profit += pnl;
                } else {
                    gross_loss += pnl.abs();
                }
                in_pos = 0;
                if equity > peak_equity {
                    peak_equity = equity;
                }
                if equity > day_peak {
                    finalize_daily_drawdown_segment(day_peak, day_low, &mut max_daily_dd);
                    day_peak = equity;
                    day_low = equity;
                } else if equity < day_low {
                    day_low = equity;
                }
                let current_dd = if peak_equity > 0.0 {
                    (peak_equity - equity) / peak_equity
                } else {
                    0.0
                };
                if current_dd > max_dd {
                    max_dd = current_dd;
                }
                // This close consumed the bar. The pre-gap signal must not
                // reopen a position at the same close after forced liquidation.
                continue;
            }
        }

        if in_pos != 0 {
            let lo = low[i];
            let hi = high[i];
            // Float (unrealized) PnL drives intrabar DD/peak. Scale by the
            // entry-captured `pos_lots` so the drawdown the GA sees matches
            // the realized-PnL sizing (a 3%-risk trade floats 3× the DD of
            // a 1%-risk trade at the same price excursion).
            let worst_float_pnl = pos_lots
                * if in_pos == 1 {
                    (lo - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - hi) / pip * settings.pip_value_per_lot
                };
            let best_float_pnl = pos_lots
                * if in_pos == 1 {
                    (hi - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - lo) / pip * settings.pip_value_per_lot
                };
            if (equity + best_float_pnl) > peak_equity {
                peak_equity = equity + best_float_pnl;
            }
            if (equity + best_float_pnl) > day_peak {
                finalize_daily_drawdown_segment(day_peak, day_low, &mut max_daily_dd);
                day_peak = equity + best_float_pnl;
                // A new same-day peak starts a new causal drawdown segment.
                // Preserve this bar's worst excursion (the canonical
                // best-before-worst intrabar convention) but discard troughs
                // that occurred before the new peak.
                day_low = equity + worst_float_pnl;
            } else if (equity + worst_float_pnl) < day_low {
                day_low = equity + worst_float_pnl;
            }

            let current_dd = if peak_equity > 0.0 {
                (peak_equity - (equity + worst_float_pnl)) / peak_equity
            } else {
                0.0
            };
            if current_dd > max_dd {
                max_dd = current_dd;
            }

            let exit_px = protective_exit_price(
                in_pos,
                entry_px,
                pos_sl_pips,
                pos_tp_pips,
                &mut trail_px,
                hi,
                lo,
                close[i],
                i - entry_idx as usize,
                pip,
                settings,
            )
            .or_else(|| session_close.then_some(close[i]));
            if let Some(exit_px) = exit_px {
                let pnl = if in_pos == 1 {
                    (exit_px - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - exit_px) / pip * settings.pip_value_per_lot
                };
                // Risk-based sizing: the price-derived `pnl` (set in the
                // shared protective-exit policy above) and the commission +
                // half-spread cost both scale by the entry-captured
                // `pos_lots`. (Half-spread was already paid at entry via the
                // adjusted entry_px; this is the exit-side half + commission.)
                let price_gross = pnl * pos_lots - half_spread_cost * pos_lots;
                let pnl =
                    pnl * pos_lots - (settings.commission_per_trade + half_spread_cost) * pos_lots;
                // Phase C.2: apply broker swap + conversion fee (size-aware).
                let entry_ts_ms = if use_timestamps && entry_idx >= 0 {
                    timestamps.get(entry_idx as usize).copied().unwrap_or(0)
                } else {
                    0
                };
                let exit_ts_ms = if use_timestamps {
                    timestamps.get(i).copied().unwrap_or(0)
                } else {
                    0
                };
                let pnl = apply_carry_and_fee_scaled(
                    pnl,
                    price_gross,
                    pos_lots,
                    in_pos,
                    entry_ts_ms,
                    exit_ts_ms,
                    settings,
                );
                if RECORD_TRADES {
                    record_closed_cpu_trade(
                        trades,
                        entry_ts_ms,
                        exit_ts_ms,
                        pnl,
                        initial_equity,
                        pos_sl_pips * settings.pip_value_per_lot * pos_lots,
                        mfe_money,
                        mae_money,
                    );
                }
                equity += pnl;
                current_month_pnl += pnl;
                trade_count += 1;
                if pnl > 0.0 {
                    wins += 1;
                    gross_profit += pnl;
                } else {
                    gross_loss += pnl.abs();
                }
                in_pos = 0;
                if equity > peak_equity {
                    peak_equity = equity;
                }
                if equity > day_peak {
                    finalize_daily_drawdown_segment(day_peak, day_low, &mut max_daily_dd);
                    day_peak = equity;
                    day_low = equity;
                } else if equity < day_low {
                    day_low = equity;
                }

                let current_dd = if peak_equity > 0.0 {
                    (peak_equity - equity) / peak_equity
                } else {
                    0.0
                };
                if current_dd > max_dd {
                    max_dd = current_dd;
                }
            }
        } else {
            // Causal entry: act on the signal observed at the PRIOR bar's
            // close, fill at the CURRENT bar's close. Previously the code
            // read `signals[i]` and immediately filled at `close[i]` — but
            // the signal itself is computed from bar i's close/high/low, so
            // the trade was peeking at the very bar it was supposed to
            // execute on. This 1-bar shift removes that intra-bar look-ahead.
            let s = signals[i - 1];
            if s != 0 && !session_block_entry {
                // max_trades_per_day gate
                if settings.max_trades_per_day > 0 && day_trade_count >= settings.max_trades_per_day
                {
                    continue;
                }
                let (entry_sl, entry_tp, size_multiplier) = if let Some(state) = netted.as_ref() {
                    (
                        state.tape.sl_pips[i - 1],
                        state.tape.tp_pips[i - 1],
                        state.tape.ml_multipliers[i - 1],
                    )
                } else {
                    let Some((sl, tp)) = entry_sl_tp_pips(settings, i - 1) else {
                        continue;
                    };
                    (sl, tp, 1.0)
                };
                if size_multiplier == 0.0 {
                    continue;
                }
                let mut entry_lots = if use_risk_sizing {
                    let conf = confidences.get(i - 1).copied().unwrap_or(1.0);
                    risk_based_pos_lots(conf, equity, entry_sl, settings)
                } else {
                    1.0
                };
                if let Some(state) = netted.as_deref_mut() {
                    // ML shrinks the already confidence-sized risk ONCE. A
                    // broker-quantity refusal must occur before lane occupancy.
                    entry_lots *= size_multiplier;
                    if entry_lots <= 0.0 {
                        continue;
                    }
                    if let Some(grid) = state.lot_grid {
                        match grid.normalize_down(entry_lots) {
                            Some(lots) => entry_lots = lots,
                            None => {
                                state.below_min_entries += 1;
                                continue;
                            }
                        }
                    }
                }
                if entry_lots <= 0.0 {
                    continue;
                }
                in_pos = s;
                // Bug #1 fix: half-spread applied at entry (entry_px offset), half at exit
                entry_px = close[i] + (s as f64) * half_spread_px;
                entry_idx = i as i64;
                trail_px = 0.0;
                mfe_money = 0.0;
                mae_money = 0.0;
                day_trade_count += 1;

                // Risk-based, confidence-scaled position sizing (Phase 1).
                // Confidence is read at the signal bar (i-1), matching the
                // causal 1-bar entry shift (signal observed at i-1, filled
                // at i). `pos_lots` is captured here and stays constant for
                // the life of this trade; it multiplies every realized PnL,
                // cost, float-PnL and carry/fee below. When sizing is off
                // (or no confidence slice) `pos_lots` is forced to 1.0 =
                // exact legacy fixed-1-lot behaviour.
                // Adaptive stops: capture the ENTRY SL/TP from the SIGNAL bar
                // (i-1) — the same causally-available just-closed bar the signal
                // and confidence come from, matching live's "signal + bracket from
                // one closed bar". Held for the trade's life. The fixed path
                // returns the scalar sl_pips/tp_pips (byte-identical).
                pos_sl_pips = entry_sl;
                pos_tp_pips = entry_tp;
                pos_lots = entry_lots;
            }
        }
    }

    if let Some(state) = netted.as_deref_mut() {
        state.ending_realized_balance = equity;
        if in_pos != 0 {
            let last = n - 1;
            let gross = f64::from(in_pos) * (close[last] - entry_px) / pip
                * settings.pip_value_per_lot
                * pos_lots;
            // A mark is not a fill. The canonical core charges its complete
            // round-trip commission on closure; report pending costs separately.
            state.terminal_open = Some(TerminalBarPositionV1 {
                direction: in_pos,
                entry_bar_index: entry_idx as usize,
                entry_timestamp_ms: timestamps.get(entry_idx as usize).copied().unwrap_or(0),
                modeled_entry_price: entry_px,
                lots: pos_lots,
                stop_pips: pos_sl_pips,
                target_pips: pos_tp_pips,
                active_trailing_stop_price: (trail_px > 0.0).then_some(trail_px),
                mark_timestamp_ms: timestamps.get(last).copied().unwrap_or(0),
                mark_close_price: close[last],
                gross_unrealized_account: gross,
                pending_round_trip_commission_account: settings.commission_per_trade * pos_lots,
                marked_equity_before_pending_costs: equity + gross,
            });
        }
    }

    // The boundary block finalizes only completed days. Include the current
    // final day so a terminal same-day peak-to-trough move is never dropped.
    if last_day != -1 {
        finalize_daily_drawdown_segment(day_peak, day_low, &mut max_daily_dd);
    }

    let net_profit = equity - initial_equity;
    let win_rate = if trade_count > 0 {
        wins as f64 / trade_count as f64
    } else {
        0.0
    };
    let pf = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        10.0
    } else {
        0.0
    };
    let expectancy = if trade_count > 0 {
        net_profit / trade_count as f64
    } else {
        0.0
    };

    let (sharpe, avg_m, std_m, completed_month_count) = if month_ptr >= 0 {
        let limit = month_ptr.min(month_capacity.saturating_sub(1) as i64) as usize;
        let sharpe = completed_month_return_sharpe_v1(
            &monthly_pnls[..=limit],
            &month_start_equities[..=limit],
        );
        let (avg_m, std_m) = mean_std(&monthly_pnls[..=limit]);
        (sharpe, avg_m, std_m, limit + 1)
    } else {
        (0.0, 0.0, 0.0, 0)
    };

    // Custom/versioned consistency stays on its historical raw-money inputs.
    let consistency = if std_m > 0.0 {
        (avg_m / std_m).clamp(0.0, 1.0)
    } else if avg_m > 0.0 && completed_month_count < 2 {
        1.0
    } else {
        0.0
    };

    // monthly_target_hit_rate (slot 7, scoring_version 3, 2026-06-06):
    // the fraction of COMPLETE months whose return >= MONTHLY_RETURN_TARGET of that
    // month's STARTING equity. This is the CONSISTENT-monthly-return signal the GA
    // now optimises toward (ga_fitness reads metrics[7]) — it matches the prop-firm
    // window-consistency gate, unlike total net (compounding makes it lumpy) or
    // `consistency`/`sharpe` (= monthly mean/std, which a few big months inflate).
    // 0.04 = the operator's >=4%/month bar. Months with no trades count as misses
    // (a strategy that sits out a month did NOT hit the bar) — same spirit as the gate.
    // GPU PARITY: the GPU lane is ENABLED (`PHASE1_GPU_SIZING_PORTED = true`, this
    // file) and every device producer fills slot 7 with this same rate — cubecl
    // (`cubecl_eval.rs`), prototype B (`prototype_b_population.cu` values[7]),
    // prototype C (`prototype_c_engine/device.rs` metric_base + 7). A producer that
    // omits it scores GPU-evaluated genes with monthly_hit = 0, i.e. with the
    // dominant fitness term switched off.
    const MONTHLY_RETURN_TARGET: f64 = 0.04;
    let monthly_target_hit_rate = if month_ptr >= 0 {
        // SECOND silent drop site (2026-08-10): this `.min()` is what actually
        // truncates the rate to the first `month_capacity` months. The
        // announcement is made at the write site above (once per process, with
        // both numbers and the coverage fraction) rather than here, because
        // this loop runs per gene and the write site already knows the frame
        // ran past the array.
        let limit = month_ptr.min(month_capacity.saturating_sub(1) as i64) as usize;
        let mut hit = 0usize;
        let mut counted = 0usize;
        for idx in 0..=limit {
            let base = month_start_equities[idx];
            if base > 0.0 {
                counted += 1;
                if monthly_pnls[idx] / base >= MONTHLY_RETURN_TARGET {
                    hit += 1;
                }
            }
        }
        if counted > 0 {
            hit as f64 / counted as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Final NaN/inf scrub. A single non-finite slot would poison sorting in
    // the GA (any comparison with NaN returns Equal via partial_cmp fallback).
    //
    // Report the metric mask without guessing its cause. In particular, the
    // known negative-infinity Sharpe marker rejects invalid completed-month
    // returns even when every financial input and every other metric is finite.
    // The existing scrub, sentinel and downstream fitness guards are unchanged.
    let inputs = [
        ("net_profit", net_profit),
        ("sharpe", sharpe),
        ("peak_equity", peak_equity),
        ("max_dd", max_dd),
        ("win_rate", win_rate),
        ("pf", pf),
        ("expectancy", expectancy),
        ("consistency", consistency),
        ("max_daily_dd", max_daily_dd),
    ];
    let nan_names: Vec<&str> = inputs
        .iter()
        .filter(|(_, v)| !v.is_finite())
        .map(|(name, _)| *name)
        .collect();
    if !nan_names.is_empty() {
        if let Some(state) = netted.as_deref_mut() {
            state.nonfinite_metrics = true;
        }
        report_non_finite_candidate_metrics_v1(trade_count, &nan_names, sharpe);
    }
    let sanitize = |v: f64| if v.is_finite() { v } else { 0.0 };
    [
        sanitize(net_profit),
        sanitize_sharpe_v1(sharpe),
        sanitize(peak_equity),
        sanitize(max_dd),
        sanitize(win_rate),
        sanitize(pf),
        sanitize(expectancy),
        sanitize(monthly_target_hit_rate), // slot 7: the consistent-monthly-return signal (scoring_version 3)
        trade_count as f64,
        sanitize(consistency),
        sanitize(max_daily_dd),
    ]
}

fn finalize_daily_drawdown_segment(day_peak: f64, day_low: f64, max_daily_dd: &mut f64) {
    if day_peak > 0.0 {
        let drawdown = (day_peak - day_low) / day_peak;
        if drawdown > *max_daily_dd {
            *max_daily_dd = drawdown;
        }
    }
}

/// Explicit fixed-lot historical ledger for per-lot economic/timing screens.
/// Account-risk validation must use [`simulate_trades_with_confidence_core`]
/// instead: discarding confidence changes the sized strategy being evaluated.
pub(crate) fn simulate_trades_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    timestamps: &[i64],
    signals: &[i8],
    settings: &BacktestSettings,
) -> Vec<Trade> {
    let n = close
        .len()
        .min(high.len())
        .min(low.len())
        .min(timestamps.len())
        .min(signals.len());
    let days = timestamps[..n]
        .iter()
        .map(|ts| ts.div_euclid(86_400_000))
        .collect::<Vec<_>>();
    let mut trades = Vec::new();
    evaluate_strategy_with_ledger_core::<true>(
        &close[..n],
        &high[..n],
        &low[..n],
        &signals[..n],
        &[],
        &[],
        &days,
        &timestamps[..n],
        settings,
        &mut trades,
        None,
    );
    trades
}

/// Validate the confidence lane before an account-risk replay. An omitted or
/// short lane must never silently turn risk sizing into one-lot or max-risk
/// sizing. Zero confidence is a valid threshold crossing at the minimum risk.
pub(crate) fn validate_sizing_confidences(
    bars: usize,
    confidences: &[f64],
    settings: &BacktestSettings,
) -> anyhow::Result<()> {
    neoethos_core::research_conversion_fee::validate_conversion_fee_rate_v1(
        settings.pnl_conversion_fee_rate,
    )
    .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        settings.initial_equity().is_finite() && settings.initial_equity() > 0.0,
        "account-risk replay initial equity must be finite and positive"
    );
    if (!confidences.is_empty() || settings.risk_based_sizing) && confidences.len() != bars {
        anyhow::bail!(
            "account-risk replay confidence length {} must match {bars} bars (risk_based_sizing={})",
            confidences.len(),
            settings.risk_based_sizing
        );
    }
    if confidences
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        anyhow::bail!("account-risk replay confidences must be finite fractions in [0, 1]");
    }
    if settings.risk_based_sizing
        && (!settings.risk_per_trade_min.is_finite()
            || !settings.risk_per_trade_max.is_finite()
            || settings.risk_per_trade_min < 0.0
            || settings.risk_per_trade_max < settings.risk_per_trade_min
            || settings.risk_per_trade_max > 1.0
            || !settings.high_quality_confidence.is_finite()
            || !(0.0..=1.0).contains(&settings.high_quality_confidence)
            || settings.high_quality_confidence == 0.0)
    {
        anyhow::bail!(
            "account-risk replay requires ordered finite risk fractions and a positive confidence normalizer"
        );
    }
    Ok(())
}

/// One replay returns both the metrics and the actual entry-sized trades.
/// Consumers cannot accidentally combine risk-sized metrics with a fixed-lot
/// ledger, and walk-forward does not have to simulate every window twice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_strategy_with_confidence_and_ledger_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    confidences: &[f64],
    month_idx: &[i64],
    day_idx: &[i64],
    timestamps: &[i64],
    settings: &BacktestSettings,
) -> anyhow::Result<([f64; 11], Vec<Trade>)> {
    let bars = close.len();
    if high.len() != bars
        || low.len() != bars
        || signals.len() != bars
        || timestamps.len() != bars
        || day_idx.len() != bars
        || (!month_idx.is_empty() && month_idx.len() != bars)
    {
        anyhow::bail!(
            "account-risk replay length mismatch across OHLC, signals, calendar and timestamp arrays"
        );
    }
    validate_sizing_confidences(bars, confidences, settings)?;
    let mut trades = Vec::new();
    let metrics = evaluate_strategy_with_ledger_core::<true>(
        close,
        high,
        low,
        signals,
        confidences,
        month_idx,
        day_idx,
        timestamps,
        settings,
        &mut trades,
        None,
    );
    Ok((metrics, trades))
}

/// Account-risk ledger using the same prior-bar confidence, captured entry
/// lots, compounding, stops and costs as CPU population fitness. Fixed-lot
/// settings are supported explicitly; an enabled risk policy requires the
/// complete confidence series rather than an implicit one-lot fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trades_with_confidence_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    timestamps: &[i64],
    signals: &[i8],
    confidences: &[f64],
    settings: &BacktestSettings,
) -> anyhow::Result<Vec<Trade>> {
    let days = timestamps
        .iter()
        .map(|ts| ts.div_euclid(86_400_000))
        .collect::<Vec<_>>();
    evaluate_strategy_with_confidence_and_ledger_core(
        close,
        high,
        low,
        signals,
        confidences,
        &[],
        &days,
        timestamps,
        settings,
    )
    .map(|(_, trades)| trades)
}

/// Configured weekend policy, not a universal assumption about FX sessions.
/// Intrabar SL/TP/trailing barriers always precede a close-of-bar session exit.
fn weekend_session_policy(timestamp_ms: i64, enabled: bool) -> (bool, bool) {
    if !enabled || timestamp_ms <= 0 {
        return (false, false);
    }
    let minutes = timestamp_ms.div_euclid(60_000).rem_euclid(1_440);
    let weekday = (timestamp_ms.div_euclid(86_400_000) + 4).rem_euclid(7);
    let friday = weekday == 5 && minutes >= 20 * 60;
    let monday = weekday == 1 && minutes < 30;
    (friday, friday || monday)
}

#[allow(clippy::too_many_arguments)]
fn record_closed_cpu_trade(
    trades: &mut Vec<Trade>,
    entry_time: i64,
    exit_time: i64,
    pnl: f64,
    initial_balance: f64,
    initial_risk: f64,
    mfe: f64,
    mae: f64,
) {
    trades.push(Trade {
        entry_time,
        exit_time: Some(exit_time),
        pnl,
        pnl_pct: Some(pnl / initial_balance),
        duration_hours: (exit_time >= entry_time)
            .then_some((exit_time - entry_time) as f64 / 3_600_000.0),
        mfe,
        mae,
        r_multiple: pnl / initial_risk.max(1e-9),
    });
}

/// Public broker-real trade simulation boundary. The raw OHLC/cost simulator
/// stays crate-private so external callers cannot skip the typed capability.
pub fn simulate_trades_broker_real(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    timestamps: &[i64],
    signals: &[i8],
    settings: &BacktestSettings,
) -> anyhow::Result<Vec<Trade>> {
    require_historical_evaluation_authority()?;
    Ok(simulate_trades_core(
        close, high, low, timestamps, signals, settings,
    ))
}

/// Synthesize the per-gene SMC-gated signals plus a per-bar confidence in
/// `[0,1]` used by the risk-based position sizer. (The CPU population
/// evaluator's single signal+confidence source.) Confidence is `0.0` where
/// the signal is `0`; otherwise it
/// measures how far the combined indicator score sits past the crossed
/// threshold, normalised by the long/short threshold gap:
///   gap    = (long_threshold - short_threshold).abs().max(1e-6)
///   long:  margin = combined[i] - long_threshold
///   short: margin = short_threshold - combined[i]
///   conf   = (margin / gap).clamp(0.0, 1.0)
///
/// Confidence is computed from the RAW threshold crossing (pre-SMC-gate),
/// and emitted only for bars that survive SMC gating (i.e. where the final
/// signal is non-zero), so it aligns exactly with the signals slice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn synthesize_signals_and_confidence_cpu(
    indicators: ArrayView2<'_, f64>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    weights: &[f64; 11],
    gene_index: usize,
    n_samples: usize,
) -> (Vec<i8>, Vec<f64>) {
    let mut combined = vec![0.0_f64; n_samples];
    let start = gene_offsets[gene_index] as usize;
    let end = gene_offsets[gene_index + 1] as usize;
    for i in start..end {
        let idx = gene_indices[i] as usize;
        let w = gene_weights[i];
        if idx < indicators.nrows() {
            let row = indicators.row(idx);
            for (j, &v) in row.iter().enumerate() {
                combined[j] += w * v;
            }
        }
    }

    let mut signals = vec![0i8; n_samples];
    let mut confidences = vec![0.0_f64; n_samples];
    let lt = long_thr[gene_index];
    let st = short_thr[gene_index];
    // Threshold gap normaliser for confidence; guard against a zero/inverted
    // gap so the division is always finite.
    let gap = (lt - st).abs().max(1e-6);
    let flags = gene_smc_flags[gene_index];
    let active_sum: f64 = flags
        .iter()
        .enumerate()
        .map(|(i, &f)| if f != 0 { weights[i] } else { 0.0 })
        .sum();
    // Hard bypass — see `signals_for_gene_full` in search_engine.rs.
    // Lets the GA's evaluation path also skip SMC gating when set.
    //
    // F-CORE3 closure (2026-05-25): previously read `std::env::var`
    // inline on EVERY gene during per-gene signal synthesis (i.e.
    // population × generations env reads per discovery run). Now
    // resolved through the typed `SmcGateOverrides::disable_gate`
    // boundary so the env is hit at most once per process.
    // Perf: read ONLY the bool — cloning the whole overrides struct here
    // heap-allocated a String per gene (see `smc_gate_disabled`).
    let smc_bypass = crate::genetic::smc_gate_disabled();
    let active_sum = if smc_bypass { 0.0 } else { active_sum };
    let gate = gate_threshold.min(active_sum);

    for i in 0..n_samples {
        let v = combined[i];
        let sig = if v >= lt {
            1
        } else if v <= st {
            -1
        } else {
            0
        };
        if sig == 0 {
            continue;
        }

        // Confidence of the raw threshold crossing (pre-gate). Only stored
        // for bars whose final (post-SMC-gate) signal survives.
        let margin = if sig == 1 { v - lt } else { st - v };
        let conf = (margin / gap).clamp(0.0, 1.0);

        if active_sum > 0.0 {
            let mut score = 0.0f64;
            let smc = smc_data[i];
            for j in 0..11 {
                if flags[j] != 0 {
                    if j == 5 {
                        if smc[j] == 1 {
                            score += weights[j];
                        }
                    } else if smc[j] == sig {
                        score += weights[j];
                    }
                }
            }
            if score >= gate {
                signals[i] = sig;
                confidences[i] = conf;
            }
        } else {
            signals[i] = sig;
            confidences[i] = conf;
        }
    }

    (signals, confidences)
}

/// The largest population worth submitting to [`validation_backtest_population`]
/// in ONE call, derived from the card's free memory — or `None` when no card
/// will take the work (no device, kernels disabled, or the build has no native
/// engine).
///
/// This is the batching/search separation made callable. A SUBMISSION size is
/// free to change: genes are independent, chunk boundaries do not appear in any
/// per-gene metric, and the evaluator re-checks and splits internally anyway.
/// A POPULATION (the GA's) is not free to change: it decides which candidates
/// exist at all. Callers of this function are only ever choosing a submission
/// size; the GA population must come from config
/// (`models.prop_search_population` / `prop_search_population_auto`), where
/// raising it is a deliberate, logged, selection-changing act.
#[cfg(feature = "gpu")]
pub fn gpu_submission_ceiling(bars: usize, feature_count: usize) -> Option<usize> {
    // Same gate as the dispatch path below: when the gate is closed every
    // population runs on the CPU, where chunk size only bounds host memory —
    // report None so callers keep their conservative constants.
    if !cuda_eval_signal_kernel_enabled() || !cuda_eval_backtest_kernel_enabled() {
        return None;
    }
    #[cfg(feature = "gpu-b-adapter")]
    {
        // This pre-run API has no sealed device route, so it cannot guess
        // ordinal zero or let configuration choose one. Exact same-ordinal
        // allocation rebatching occurs after the run probe seals its route.
        let _ = (bars, feature_count);
        None
    }
    #[cfg(not(feature = "gpu-b-adapter"))]
    {
        // The CubeCL lane has no published fits arithmetic; a wrong guess here
        // would be worse than the callers' existing constants.
        let _ = (bars, feature_count);
        None
    }
}

/// Standalone Prototype-B adapter build: query Prototype B directly without
/// compiling or consulting the generic CubeCL runtime. Adapter-only builds link
/// the honest no-device stub; `gpu-b-native` replaces it with the CUDA backend.
#[cfg(all(feature = "gpu-b-adapter", not(feature = "gpu")))]
pub fn gpu_submission_ceiling(bars: usize, feature_count: usize) -> Option<usize> {
    // A no-argument sizing query cannot select a CUDA ordinal. The exact
    // run-bound route owns device sizing and deterministic same-card rebatching.
    let _ = (bars, feature_count);
    None
}

/// Non-GPU build: there is no card, so there is no ceiling — callers keep their
/// CPU-sized constants and the build compiles the same call sites unchanged.
#[cfg(not(any(feature = "gpu", feature = "gpu-b-adapter")))]
pub fn gpu_submission_ceiling(_bars: usize, _feature_count: usize) -> Option<usize> {
    None
}

/// Materialise the optional `stop_vol_mult` contract into a full-length slice.
///
/// The field is documented as optional: an empty slice means every gene uses
/// its fixed stops. The CPU walk honours that with `.get(g).unwrap_or(0.0)`,
/// while the CUDA dispatch slices it per batch. Both lanes must see identical
/// input, so normalise once at every entry point.
pub(crate) fn normalized_stop_vol_mult(stop_vol_mult: &[f64], n_genes: usize) -> Option<Vec<f64>> {
    stop_vol_mult.is_empty().then(|| vec![0.0; n_genes])
}

#[cfg(any(test, feature = "gpu-b-adapter"))]
fn require_exact_native_population_rows(
    outcome: Result<Vec<[f64; 11]>, String>,
    expected: usize,
    lane: &'static str,
) -> Result<Vec<[f64; 11]>, String> {
    match outcome {
        Ok(rows) if rows.len() == expected => Ok(rows),
        Ok(rows) => Err(format!(
            "Prototype B native {lane} returned {} metric rows for {expected} candidates; \
             refusing CPU substitution",
            rows.len()
        )),
        Err(error) => Err(format!(
            "Prototype B native {lane} device evaluation failed: {error}; \
             refusing CPU substitution"
        )),
    }
}

#[cfg(test)]
mod native_population_outcome_contract_tests {
    use super::require_exact_native_population_rows;

    #[test]
    fn prototype_b_native_production_contract_accepts_only_the_exact_shape() {
        let rows = vec![[1.0_f64; 11], [2.0_f64; 11]];
        let accepted = require_exact_native_population_rows(Ok(rows.clone()), 2, "population_eval")
            .expect("the exact native result shape must be accepted");
        assert_eq!(accepted, rows);

        let error =
            require_exact_native_population_rows(Ok(vec![[0.0_f64; 11]]), 2, "population_eval")
                .expect_err("a short native result must fail loud");
        assert!(error.contains("population_eval"));
        assert!(error.contains("returned 1 metric rows for 2 candidates"));
        assert!(error.contains("refusing CPU substitution"));
    }

    #[test]
    fn prototype_b_native_production_contract_propagates_device_errors() {
        let error = require_exact_native_population_rows(
            Err("CUDA launch failed".to_string()),
            4,
            "validation_eval",
        )
        .expect_err("a native device error must fail loud");
        assert!(error.contains("validation_eval"));
        assert!(error.contains("CUDA launch failed"));
        assert!(error.contains("refusing CPU substitution"));
    }
}

/// Production route for the native CUDA engine.
///
/// `gpu-b-adapter` type-checks this control flow against the honest no-device
/// stub; `gpu-b-native` selects the linked CUDA implementation. When a card is
/// present and the resolved backend did not explicitly select the CPU, the
/// canonical population caller enters this function before the generic CubeCL
/// lane, a Rayon pool, or a CPU work list. A native error or wrong result shape
/// is an error; only an absent card or an explicit CPU selection returns `None`.
#[cfg(feature = "gpu-b-adapter")]
fn evaluate_population_b_when_available(
    inputs: &PopulationEvalInputs<'_>,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
    lane: &'static str,
) -> Result<Option<Vec<[f64; 11]>>, String> {
    let expected = inputs.long_thr.len();
    if expected == 0 {
        return Ok(Some(Vec::new()));
    }

    if evidence.require_cpu_route_receipt_v1().is_ok() {
        return Ok(None);
    }
    evidence
        .require_exact_cuda_device_ordinal_v1()
        .map_err(|error| error.to_string())?;

    let stop_vol_mult_fallback =
        normalized_stop_vol_mult(inputs.stop_vol_mult, inputs.long_thr.len());
    let stop_vol_mult = stop_vol_mult_fallback
        .as_deref()
        .unwrap_or(inputs.stop_vol_mult);
    let started = std::time::Instant::now();
    let outcome = crate::gpu_native::prototype_b_population_eval::try_evaluate_population_b(
        evidence,
        inputs.gene_offsets,
        inputs.gene_indices,
        inputs.gene_weights,
        inputs.long_thr,
        inputs.short_thr,
        inputs.sl_pips,
        inputs.tp_pips,
        stop_vol_mult,
        inputs.gene_smc_flags,
        inputs.gate_threshold,
        inputs.weights,
    )
    .map_err(|error| format!("{error:#}"));
    let rows = require_exact_native_population_rows(outcome, expected, lane)?;
    crate::eval_telemetry::record_device(
        lane,
        crate::eval_telemetry::Device::Gpu,
        started.elapsed(),
    );
    Ok(Some(rows))
}

pub fn evaluate_population_core(
    inputs: PopulationEvalInputs<'_>,
) -> Result<Vec<[f64; 11]>, String> {
    require_historical_evaluation_authority().map_err(|error| error.to_string())?;
    evaluate_population_core_unchecked(inputs, None)
}

fn evaluate_population_core_unchecked(
    inputs: PopulationEvalInputs<'_>,
    evidence: Option<&crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>>,
) -> Result<Vec<[f64; 11]>, String> {
    // Reports itself on first call, so "never used" is visible rather
    // than inferred. See `eval_telemetry`.
    let _telemetry_started = std::time::Instant::now();
    let _telemetry_items = inputs.long_thr.len();
    struct TelemetryGuard(&'static str, usize, std::time::Instant);
    impl Drop for TelemetryGuard {
        fn drop(&mut self) {
            crate::eval_telemetry::record(self.0, self.1, self.2.elapsed());
        }
    }
    let _telemetry = TelemetryGuard(
        "eval::evaluate_population_core",
        _telemetry_items,
        _telemetry_started,
    );
    let evidence = evidence.ok_or_else(|| {
        "population evaluation requires a run-bound sealed device route; refusing detached execution"
            .to_string()
    })?;
    let cpu_route = evidence.require_cpu_route_receipt_v1().is_ok();
    #[cfg(feature = "gpu-b-adapter")]
    if let Some(rows) = evaluate_population_b_when_available(&inputs, evidence, "population_eval")?
    {
        return Ok(rows);
    }
    if !cpu_route {
        return Err(
            "sealed native route did not execute the native CUDA engine; refusing CPU substitution"
                .to_string(),
        );
    }
    let PopulationEvalInputs {
        close,
        high,
        low,
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        month_idx,
        day_idx,
        timestamps,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        weights,
        settings,
    } = inputs;
    let n_genes = long_thr.len();
    let n_samples = close.len();
    let stop_vol_mult_fallback = normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_mult_fallback.as_deref().unwrap_or(stop_vol_mult);

    // Per-gene CPU evaluation (signal synthesis + SL/TP backtest), reachable
    // only under the sealed exact-zero-device CPU receipt.
    let eval_gene_cpu = |g: usize| -> [f64; 11] {
        let (signals, confidences) = synthesize_signals_and_confidence_cpu(
            indicators,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            smc_data,
            gene_smc_flags,
            gate_threshold,
            weights,
            g,
            n_samples,
        );
        let mut gene_settings = settings.clone();
        gene_settings.sl_pips = sl_pips[g];
        gene_settings.tp_pips = tp_pips[g];
        // Per-gene adaptive stop multiplier (0.0 / empty slice => fixed path).
        // Pairs with the shared `adaptive_base_pips` on the cloned settings.
        gene_settings.adaptive_vol_mult = stop_vol_mult.get(g).copied().unwrap_or(0.0);
        // Risk-based sizing uses the per-bar confidence; with
        // `risk_based_sizing == false` the slice is ignored (legacy).
        fast_evaluate_strategy_core(
            close,
            high,
            low,
            &signals,
            &confidences,
            month_idx,
            day_idx,
            timestamps,
            &gene_settings,
        )
    };

    // The only legal CPU branch is the exact-zero-device route sealed before
    // this evaluation. Backend configuration cannot grant or block it.
    evidence
        .require_cpu_route_receipt_v1()
        .map_err(|error| error.to_string())?;
    let cpu_started = std::time::Instant::now();
    let results: Vec<[f64; 11]> = (0..n_genes).into_par_iter().map(&eval_gene_cpu).collect();
    evidence
        .record_successful_population(
            crate::engine_identity::PopulationEvalEngine::Cpu,
            n_genes,
            results.len(),
        )
        .map_err(|error| error.to_string())?;
    crate::eval_telemetry::record_device(
        "population_eval",
        crate::eval_telemetry::Device::Cpu,
        cpu_started.elapsed(),
    );
    Ok(results)
}

/// Non-production mathematical reference used only by unit-level parity tests.
/// Production callers cannot compile this symbol and must pass the broker-real
/// capability gate in [`evaluate_population_core`].
#[cfg(test)]
pub(crate) fn evaluate_population_core_test_oracle(
    inputs: PopulationEvalInputs<'_>,
) -> Result<Vec<[f64; 11]>, String> {
    Ok(validation_backtest_population_cpu(inputs))
}

/// Strict validation-tail population route. A sealed CUDA receipt dispatches
/// the whole population to native Prototype B; a sealed exact-zero-device
/// receipt uses the canonical CPU evaluator. Device errors, panics, and result
/// shape mismatches fail closed and never authorize CPU substitution.
#[cfg(feature = "gpu")]
fn validation_backtest_population_inner(
    inputs: PopulationEvalInputs<'_>,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> Vec<[f64; 11]> {
    // Reports itself on first call, so "never used" is visible rather
    // than inferred. See `eval_telemetry`.
    let _telemetry_started = std::time::Instant::now();
    let _telemetry_items = inputs.long_thr.len();
    struct TelemetryGuard(&'static str, usize, std::time::Instant);
    impl Drop for TelemetryGuard {
        fn drop(&mut self) {
            crate::eval_telemetry::record(self.0, self.1, self.2.elapsed());
        }
    }
    let _telemetry = TelemetryGuard(
        "eval::validation_backtest_population",
        _telemetry_items,
        _telemetry_started,
    );
    let cpu_route = evidence.require_cpu_route_receipt_v1().is_ok();
    #[cfg(feature = "gpu-b-adapter")]
    if !cpu_route {
        match evaluate_population_b_when_available(&inputs, evidence, "validation_eval") {
            Ok(Some(rows)) => return rows,
            Ok(None) => panic!("sealed native validation route resolved as CPU"),
            Err(error) => panic!("strict native validation failed: {error}"),
        }
    }
    #[cfg(not(feature = "gpu-b-adapter"))]
    if !cpu_route {
        panic!("sealed native validation route requires the compiled native CUDA adapter");
    }
    let PopulationEvalInputs {
        close,
        high,
        low,
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        month_idx,
        day_idx,
        timestamps,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        weights,
        settings,
    } = inputs;
    let n_genes = long_thr.len();
    let n_samples = close.len();
    let stop_vol_mult_fallback = normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_mult_fallback.as_deref().unwrap_or(stop_vol_mult);
    if n_genes == 0 {
        return Vec::new();
    }

    // Canonical per-gene CPU route, shared mathematically with the GA. It is a
    // local closure because it borrows the input slices.
    let eval_gene_cpu = |g: usize| -> [f64; 11] {
        let (signals, confidences) = synthesize_signals_and_confidence_cpu(
            indicators,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            smc_data,
            gene_smc_flags,
            gate_threshold,
            weights,
            g,
            n_samples,
        );
        let mut gene_settings = settings.clone();
        gene_settings.sl_pips = sl_pips[g];
        gene_settings.tp_pips = tp_pips[g];
        // Per-gene adaptive stop multiplier (0.0 / empty slice => fixed path).
        // Pairs with the shared `adaptive_base_pips` on the cloned settings.
        gene_settings.adaptive_vol_mult = stop_vol_mult.get(g).copied().unwrap_or(0.0);
        fast_evaluate_strategy_core(
            close,
            high,
            low,
            &signals,
            &confidences,
            month_idx,
            day_idx,
            timestamps,
            &gene_settings,
        )
    };

    // Canonical CPU execution exists only under the sealed exact-zero-device
    // receipt checked above; it is not a recovery action after native failure.
    evidence
        .require_cpu_route_receipt_v1()
        .unwrap_or_else(|error| panic!("CPU validation route is not authorized: {error}"));
    let cpu_started = std::time::Instant::now();
    let out: Vec<[f64; 11]> = (0..n_genes).into_par_iter().map(&eval_gene_cpu).collect();
    evidence
        .record_successful_population(
            crate::engine_identity::PopulationEvalEngine::Cpu,
            n_genes,
            out.len(),
        )
        .unwrap_or_else(|error| panic!("record exact CPU validation output: {error}"));
    crate::eval_telemetry::record_device(
        "validation_eval",
        crate::eval_telemetry::Device::Cpu,
        cpu_started.elapsed(),
    );
    out
}

#[cfg(feature = "gpu")]
pub(crate) fn validation_backtest_population_with_evidence(
    inputs: PopulationEvalInputs<'_>,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> Vec<[f64; 11]> {
    validation_backtest_population_inner(inputs, evidence)
}

/// Pure-CPU population evaluation — the canonical semantic reference for the
/// whole validation tail, available in EVERY build (with or without `gpu`).
///
/// It is the in-process mathematical reference the GPU benchmark harness checks
/// for parity. Production dispatch must validate the sealed route before entry;
/// against. There is exactly one CPU population implementation, so the reference
/// can never drift from what actually runs.
pub(crate) fn validation_backtest_population_cpu(
    inputs: PopulationEvalInputs<'_>,
) -> Vec<[f64; 11]> {
    validation_backtest_population_cpu_core::<false>(inputs)
        .expect("metrics-only population retains the existing infallible contract")
        .into_iter()
        .map(|(metrics, _)| metrics)
        .collect()
}

/// Same CPU population simulator with its actual account-risk ledger retained.
/// The receipt is checked before execution; a native route cannot enter here.
pub(crate) fn validation_backtest_population_with_ledger_cpu(
    inputs: PopulationEvalInputs<'_>,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> anyhow::Result<(Vec<[f64; 11]>, Vec<Vec<Trade>>)> {
    evidence.require_cpu_route_receipt_v1()?;
    let expected = inputs.long_thr.len();
    let started = std::time::Instant::now();
    let rows = validation_backtest_population_cpu_core::<true>(inputs)?;
    evidence.record_successful_population(
        crate::engine_identity::PopulationEvalEngine::Cpu,
        expected,
        rows.len(),
    )?;
    crate::eval_telemetry::record_device(
        "validation_eval",
        crate::eval_telemetry::Device::Cpu,
        started.elapsed(),
    );
    Ok(rows.into_iter().unzip())
}

fn validation_backtest_population_cpu_core<const RECORD_TRADES: bool>(
    inputs: PopulationEvalInputs<'_>,
) -> anyhow::Result<Vec<([f64; 11], Vec<Trade>)>> {
    // Reports itself on first call, so "never used" is visible rather
    // than inferred. See `eval_telemetry`.
    let _telemetry_started = std::time::Instant::now();
    let _telemetry_items = inputs.long_thr.len();
    struct TelemetryGuard(&'static str, usize, std::time::Instant);
    impl Drop for TelemetryGuard {
        fn drop(&mut self) {
            crate::eval_telemetry::record(self.0, self.1, self.2.elapsed());
        }
    }
    let _telemetry = TelemetryGuard(
        "eval::validation_backtest_population_cpu",
        _telemetry_items,
        _telemetry_started,
    );
    let PopulationEvalInputs {
        close,
        high,
        low,
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        month_idx,
        day_idx,
        timestamps,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        weights,
        settings,
    } = inputs;
    let n_genes = long_thr.len();
    let n_samples = close.len();
    let stop_vol_mult_fallback = normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_mult_fallback.as_deref().unwrap_or(stop_vol_mult);
    if n_genes == 0 {
        return Ok(Vec::new());
    }
    let eval_gene_cpu = |g: usize| -> anyhow::Result<([f64; 11], Vec<Trade>)> {
        let (signals, confidences) = synthesize_signals_and_confidence_cpu(
            indicators,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            smc_data,
            gene_smc_flags,
            gate_threshold,
            weights,
            g,
            n_samples,
        );
        let mut gene_settings = settings.clone();
        gene_settings.sl_pips = sl_pips[g];
        gene_settings.tp_pips = tp_pips[g];
        // Per-gene adaptive stop multiplier (0.0 / empty slice => fixed path).
        // Pairs with the shared `adaptive_base_pips` on the cloned settings.
        gene_settings.adaptive_vol_mult = stop_vol_mult.get(g).copied().unwrap_or(0.0);
        if RECORD_TRADES {
            evaluate_strategy_with_confidence_and_ledger_core(
                close,
                high,
                low,
                &signals,
                &confidences,
                month_idx,
                day_idx,
                timestamps,
                &gene_settings,
            )
        } else {
            Ok((
                fast_evaluate_strategy_core(
                    close,
                    high,
                    low,
                    &signals,
                    &confidences,
                    month_idx,
                    day_idx,
                    timestamps,
                    &gene_settings,
                ),
                Vec::new(),
            ))
        }
    };
    (0..n_genes).into_par_iter().map(&eval_gene_cpu).collect()
}

/// Standalone native validation route. A present card is evaluated by the same
/// direct Prototype-B function as the canonical population path. `Ok(None)` is
/// reserved for a card-less host or an explicit CPU selection; an attempted
/// device evaluation can only return rows or terminate with the error below.
#[cfg(all(feature = "gpu-b-adapter", not(feature = "gpu")))]
fn validation_backtest_population_native_only_with_evidence(
    inputs: PopulationEvalInputs<'_>,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> Vec<[f64; 11]> {
    let expected = inputs.long_thr.len();
    match evaluate_population_b_when_available(&inputs, evidence, "validation_eval") {
        Ok(Some(rows)) => rows,
        Ok(None) => {
            evidence
                .require_cpu_route_receipt_v1()
                .unwrap_or_else(|error| panic!("CPU validation route is not authorized: {error}"));
            let rows = validation_backtest_population_cpu(inputs);
            evidence
                .record_successful_population(
                    crate::engine_identity::PopulationEvalEngine::Cpu,
                    expected,
                    rows.len(),
                )
                .unwrap_or_else(|error| panic!("record native-only CPU validation: {error}"));
            rows
        }
        Err(error) => panic!("Prototype B native validation failed: {error}"),
    }
}

#[cfg(all(feature = "gpu-b-adapter", not(feature = "gpu")))]
pub(crate) fn validation_backtest_population_with_evidence(
    inputs: PopulationEvalInputs<'_>,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> Vec<[f64; 11]> {
    validation_backtest_population_native_only_with_evidence(inputs, evidence)
}

#[cfg(not(any(feature = "gpu", feature = "gpu-b-adapter")))]
pub(crate) fn validation_backtest_population_with_evidence(
    inputs: PopulationEvalInputs<'_>,
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> Vec<[f64; 11]> {
    let expected = inputs.long_thr.len();
    evidence
        .require_cpu_route_receipt_v1()
        .unwrap_or_else(|error| panic!("CPU validation route is not authorized: {error}"));
    let rows = validation_backtest_population_cpu(inputs);
    evidence
        .record_successful_population(
            crate::engine_identity::PopulationEvalEngine::Cpu,
            expected,
            rows.len(),
        )
        .unwrap_or_else(|error| panic!("record exact CPU validation: {error}"));
    rows
}

// ── Scenarios: one launch, many treatments ───────────────────────────────────

/// The CPU mirror of a scenario work list.
///
/// This is the canonical implementation for an explicitly sealed CPU route and
/// the mathematical oracle for CUDA parity. It reproduces each scenario exactly:
///
///   * the gene named by `base_candidate_id`, sliced out of the shared CSR
///     arrays into a one-gene batch and run through the SAME
///     `synthesize_signals_and_confidence_cpu` the whole engine uses;
///   * for `SCENARIO_PERTURB`, the gene as the device would perturb it, through
///     the generator both lanes share (`scenario::perturbed_gene`);
///   * for a cost override, the spread and commission the descriptor carries,
///     converted by the SAME division the device performs.
///
/// It refuses what it cannot reproduce rather than approximating it — see
/// [`crate::gpu_native::scenario::cpu_mirror_unsupported`]. A sub-window is the
/// only such case today and nothing builds one.
///
/// Note the deliberate asymmetry with the spread override: the device's override
/// replaces the whole per-bar resolution, so the mirror clears
/// `session_spread_profile` as well as setting `spread_pips`. Leaving the
/// profile in place would make the CPU charge a per-hour spread where the device
/// charged a flat one, creating a CPU/CUDA divergence.
pub fn validation_backtest_scenarios_cpu(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
) -> anyhow::Result<Vec<[f64; 11]>> {
    require_historical_evaluation_authority()?;
    validation_backtest_scenarios_cpu_unchecked(inputs, scenarios)
}

fn validation_backtest_scenarios_cpu_unchecked(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
) -> anyhow::Result<Vec<[f64; 11]>> {
    use crate::gpu_native::scenario;

    let PopulationEvalInputs {
        close,
        high,
        low,
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        month_idx,
        day_idx,
        timestamps,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        weights,
        settings,
    } = inputs;
    let n_genes = long_thr.len();
    let n_samples = close.len();
    if scenarios.is_empty() {
        return Ok(Vec::new());
    }
    let stop_vol_mult_fallback = normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_mult_fallback.as_deref().unwrap_or(stop_vol_mult);

    scenarios
        .par_iter()
        .map(|descriptor| -> anyhow::Result<[f64; 11]> {
            if let Some(reason) = scenario::cpu_mirror_unsupported(descriptor, n_samples) {
                anyhow::bail!("the CPU scenario mirror cannot reproduce this launch: {reason}");
            }
            let gene = descriptor.base_candidate_id as usize;
            if gene >= n_genes {
                anyhow::bail!(
                    "scenario {} names gene {gene} outside the population of {n_genes}",
                    descriptor.scenario_id
                );
            }
            let start = gene_offsets[gene] as usize;
            let end = gene_offsets[gene + 1] as usize;

            // A one-gene batch, so the canonical synth runs unchanged. Building
            // it is what lets a perturbed gene go through exactly the same code
            // path as an unperturbed one — the alternative, a second synth that
            // takes a factor, is a second implementation of the thing parity is
            // measured against.
            let perturbed = if descriptor.scenario_type == scenario::SCENARIO_PERTURB {
                Some(scenario::perturbed_gene(
                    descriptor.rng_counter,
                    long_thr[gene],
                    short_thr[gene],
                    &gene_weights[start..end],
                    sl_pips[gene],
                    tp_pips[gene],
                ))
            } else {
                None
            };
            let one_offsets = [0_i32, (end - start) as i32];
            let one_flags = [gene_smc_flags[gene]];
            let one_long = [perturbed
                .as_ref()
                .map_or(long_thr[gene], |p| p.long_threshold)];
            let one_short = [perturbed
                .as_ref()
                .map_or(short_thr[gene], |p| p.short_threshold)];
            let one_weights: &[f64] = perturbed
                .as_ref()
                .map_or(&gene_weights[start..end], |p| p.weights.as_slice());

            let (signals, confidences) = synthesize_signals_and_confidence_cpu(
                indicators,
                &one_offsets,
                &gene_indices[start..end],
                one_weights,
                &one_long,
                &one_short,
                smc_data,
                &one_flags,
                gate_threshold,
                weights,
                0,
                n_samples,
            );

            let mut gene_settings = settings.clone();
            gene_settings.sl_pips = perturbed.as_ref().map_or(sl_pips[gene], |p| p.sl_pips);
            gene_settings.tp_pips = perturbed.as_ref().map_or(tp_pips[gene], |p| p.tp_pips);
            gene_settings.adaptive_vol_mult = stop_vol_mult.get(gene).copied().unwrap_or(0.0);
            if descriptor.spread_ticks != scenario::NO_TICK_OVERRIDE {
                gene_settings.spread_pips =
                    f64::from(descriptor.spread_ticks) / scenario::TICKS_PER_PIP;
                // The device override bypasses the per-hour resolution entirely.
                gene_settings.session_spread_profile = None;
            }
            if descriptor.commission_micros != scenario::NO_MICRO_OVERRIDE {
                gene_settings.commission_per_trade =
                    descriptor.commission_micros as f64 / scenario::MICROS_PER_UNIT;
            }

            Ok(fast_evaluate_strategy_core(
                close,
                high,
                low,
                &signals,
                &confidences,
                month_idx,
                day_idx,
                timestamps,
                &gene_settings,
            ))
        })
        .collect()
}

/// Non-production scenario mirror for formula/parity unit tests. This keeps
/// mathematical coverage without exposing an unchecked financial entry point
/// in release builds.
#[cfg(test)]
pub(crate) fn validation_backtest_scenarios_cpu_test_oracle(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
) -> anyhow::Result<Vec<[f64; 11]>> {
    validation_backtest_scenarios_cpu_unchecked(inputs, scenarios)
}

/// Evaluate a scenario work list — ONE launch for the whole quality screen.
///
/// The population twin sends one full-series scenario per gene, so a screen
/// wanting three treatments of the same genes needed three launches and a
/// Monte-Carlo screen wanting 100 needed the genes cloned 100 times. This takes
/// the work list directly: 174 genes and 17 574 scenarios in one submission.
///
/// Routing is strict: a sealed CUDA receipt uses native Prototype B and fails
/// closed on any device error; a sealed exact-zero-device receipt uses the CPU
/// mirror above. No runtime failure changes the selected engine.
#[cfg(feature = "gpu")]
fn validation_backtest_scenarios_inner(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> anyhow::Result<Vec<[f64; 11]>> {
    require_historical_evaluation_authority()?;
    let cpu_route = evidence.require_cpu_route_receipt_v1().is_ok();
    let n_scenarios = scenarios.len();
    if n_scenarios == 0 {
        return Ok(Vec::new());
    }
    #[cfg(feature = "gpu-b-adapter")]
    if !cpu_route {
        evidence.require_exact_cuda_device_ordinal_v1()?;
        let started = std::time::Instant::now();
        let rows = crate::gpu_native::prototype_b_population_eval::try_evaluate_scenarios_b(
            evidence,
            inputs.gene_offsets,
            inputs.gene_indices,
            inputs.gene_weights,
            inputs.long_thr,
            inputs.short_thr,
            inputs.sl_pips,
            inputs.tp_pips,
            inputs.stop_vol_mult,
            inputs.gene_smc_flags,
            inputs.gate_threshold,
            inputs.weights,
            scenarios,
        )?;
        if rows.len() != n_scenarios {
            anyhow::bail!(
                "strict native scenario evaluation returned {} rows for {n_scenarios}; refusing CPU substitution",
                rows.len()
            );
        }
        crate::eval_telemetry::record_device(
            "validation_eval",
            crate::eval_telemetry::Device::Gpu,
            started.elapsed(),
        );
        return Ok(rows);
    }
    #[cfg(not(feature = "gpu-b-adapter"))]
    if !cpu_route {
        anyhow::bail!("sealed native scenario route requires the compiled native CUDA adapter");
    }
    evidence.require_cpu_route_receipt_v1()?;
    let cpu_started = std::time::Instant::now();
    let out = validation_backtest_scenarios_cpu(inputs, scenarios);
    if let Ok(rows) = &out {
        evidence
            .record_successful_population(
                crate::engine_identity::PopulationEvalEngine::Cpu,
                n_scenarios,
                rows.len(),
            )
            .map_err(anyhow::Error::new)?;
    }
    crate::eval_telemetry::record_device(
        "validation_eval",
        crate::eval_telemetry::Device::Cpu,
        cpu_started.elapsed(),
    );
    out
}

#[cfg(feature = "gpu")]
pub(crate) fn validation_backtest_scenarios_with_evidence(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> anyhow::Result<Vec<[f64; 11]>> {
    validation_backtest_scenarios_inner(inputs, scenarios, evidence)
}

/// Standalone Prototype-B adapter route without the generic CubeCL `gpu`
/// feature. The adapter-only build reaches the honest no-device stub; a native
/// build reaches the same sealed scenario evaluator used by the aggregate CUDA
/// feature. Once a device attempt begins, failure is returned rather than
/// silently recomputed on the CPU.
#[cfg(all(feature = "gpu-b-adapter", not(feature = "gpu")))]
fn validation_backtest_scenarios_native_only(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> anyhow::Result<Vec<[f64; 11]>> {
    require_historical_evaluation_authority()?;
    let expected = scenarios.len();
    if expected == 0 {
        return Ok(Vec::new());
    }

    if evidence.require_cpu_route_receipt_v1().is_ok() {
        let started = std::time::Instant::now();
        let rows = validation_backtest_scenarios_cpu_unchecked(inputs, scenarios)?;
        evidence
            .record_successful_population(
                crate::engine_identity::PopulationEvalEngine::Cpu,
                expected,
                rows.len(),
            )
            .map_err(anyhow::Error::new)?;
        crate::eval_telemetry::record_device(
            "validation_eval",
            crate::eval_telemetry::Device::Cpu,
            started.elapsed(),
        );
        return Ok(rows);
    }

    evidence.require_exact_cuda_device_ordinal_v1()?;
    let started = std::time::Instant::now();
    let rows = crate::gpu_native::prototype_b_population_eval::try_evaluate_scenarios_b(
        evidence,
        inputs.gene_offsets,
        inputs.gene_indices,
        inputs.gene_weights,
        inputs.long_thr,
        inputs.short_thr,
        inputs.sl_pips,
        inputs.tp_pips,
        inputs.stop_vol_mult,
        inputs.gene_smc_flags,
        inputs.gate_threshold,
        inputs.weights,
        scenarios,
    )?;
    if rows.len() != expected {
        anyhow::bail!(
            "strict native scenario evaluation returned {} rows for {expected}; refusing CPU substitution",
            rows.len()
        );
    }
    crate::eval_telemetry::record_device(
        "validation_eval",
        crate::eval_telemetry::Device::Gpu,
        started.elapsed(),
    );
    Ok(rows)
}

#[cfg(all(feature = "gpu-b-adapter", not(feature = "gpu")))]
pub(crate) fn validation_backtest_scenarios_with_evidence(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> anyhow::Result<Vec<[f64; 11]>> {
    validation_backtest_scenarios_native_only(inputs, scenarios, evidence)
}

#[cfg(not(any(feature = "gpu", feature = "gpu-b-adapter")))]
pub(crate) fn validation_backtest_scenarios_with_evidence(
    inputs: PopulationEvalInputs<'_>,
    scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1<'_>,
) -> anyhow::Result<Vec<[f64; 11]>> {
    let expected = scenarios.len();
    evidence.require_cpu_route_receipt_v1()?;
    let rows = validation_backtest_scenarios_cpu(inputs, scenarios)?;
    evidence
        .record_successful_population(
            crate::engine_identity::PopulationEvalEngine::Cpu,
            expected,
            rows.len(),
        )
        .map_err(anyhow::Error::new)?;
    Ok(rows)
}

#[cfg(test)]
mod scenario_mirror_tests {
    use super::*;
    use crate::gpu_native::scenario;

    /// A tiny but real population: two genes over 400 bars of a synthetic
    /// series, evaluated through the same CPU engine production uses.
    fn fixture() -> (
        Vec<f64>,
        ndarray::Array2<f64>,
        Vec<SmcRow>,
        Vec<i64>,
        BacktestSettings,
    ) {
        let bars = 400;
        let close: Vec<f64> = (0..bars)
            .map(|i| 1.1000 + ((i as f64) * 0.37).sin() * 0.0040)
            .collect();
        let mut indicators = ndarray::Array2::<f64>::zeros((2, bars));
        for i in 0..bars {
            indicators[(0, i)] = ((i as f64) * 0.11).sin();
            indicators[(1, i)] = ((i as f64) * 0.07).cos();
        }
        let smc = vec![[0_i8; 11]; bars];
        let months = vec![0_i64; bars];
        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.spread_pips = 1.0;
        settings.commission_per_trade = 2.0;
        settings.pip_value_per_lot = 10.0;
        settings.risk_based_sizing = false;
        settings.sl_pips = 20.0;
        settings.tp_pips = 40.0;
        (close, indicators, smc, months, settings)
    }

    fn inputs<'a>(
        close: &'a [f64],
        indicators: &'a ndarray::Array2<f64>,
        smc: &'a [SmcRow],
        months: &'a [i64],
        settings: &'a BacktestSettings,
        offsets: &'a [i32],
        idx: &'a [i32],
        w: &'a [f64],
        lt: &'a [f64],
        st: &'a [f64],
        sl: &'a [f64],
        tp: &'a [f64],
        flags: &'a [SmcRow],
        smc_weights: &'a [f64; 11],
    ) -> PopulationEvalInputs<'a> {
        PopulationEvalInputs {
            close,
            high: close,
            low: close,
            indicators: indicators.view(),
            gene_offsets: offsets,
            gene_indices: idx,
            gene_weights: w,
            long_thr: lt,
            short_thr: st,
            month_idx: months,
            day_idx: months,
            timestamps: &[],
            sl_pips: sl,
            tp_pips: tp,
            stop_vol_mult: &[],
            smc_data: smc,
            gene_smc_flags: flags,
            gate_threshold: 0.0,
            weights: smc_weights,
            settings,
        }
    }

    /// THE PARITY FLOOR, stated as a test rather than as a claim.
    ///
    /// A work list of nothing but `base_scenario`, one per gene, must produce
    /// exactly what the population path produces. If this ever diverges, every
    /// number the scenario lane reports is suspect, because this is the case the
    /// 147 GPU parity fixtures also exercise.
    #[test]
    fn a_base_work_list_equals_the_population_path_exactly() {
        let (close, indicators, smc, months, settings) = fixture();
        let offsets = [0_i32, 1, 2];
        let idx = [0_i32, 1];
        let w = [1.0_f64, 1.0];
        let lt = [0.5_f64, 0.4];
        let st = [-0.5_f64, -0.4];
        let sl = [20.0_f64, 25.0];
        let tp = [40.0_f64, 50.0];
        let flags = [[0_i8; 11]; 2];
        let smc_weights = [0.0_f64; 11];

        let population = validation_backtest_population_cpu(inputs(
            &close,
            &indicators,
            &smc,
            &months,
            &settings,
            &offsets,
            &idx,
            &w,
            &lt,
            &st,
            &sl,
            &tp,
            &flags,
            &smc_weights,
        ));
        let work_list: Vec<_> = (0..2u64)
            .map(|g| scenario::base_scenario(g, g, close.len()))
            .collect();
        let scenarios = validation_backtest_scenarios_cpu_test_oracle(
            inputs(
                &close,
                &indicators,
                &smc,
                &months,
                &settings,
                &offsets,
                &idx,
                &w,
                &lt,
                &st,
                &sl,
                &tp,
                &flags,
                &smc_weights,
            ),
            &work_list,
        )
        .expect("a base work list is always reproducible on the CPU");

        assert_eq!(scenarios.len(), population.len());
        for (gene, (a, b)) in population.iter().zip(scenarios.iter()).enumerate() {
            assert_eq!(
                a, b,
                "gene {gene} differs between the population path and a base work list"
            );
        }
    }

    /// A cost scenario must charge what the descriptor says, and the descriptor
    /// must be able to say it exactly.
    #[test]
    fn a_cost_scenario_charges_the_descriptor_not_the_settings() {
        let (close, indicators, smc, months, settings) = fixture();
        let offsets = [0_i32, 1];
        let idx = [0_i32];
        let w = [1.0_f64];
        let lt = [0.3_f64];
        let st = [-0.3_f64];
        let sl = [20.0_f64];
        let tp = [40.0_f64];
        let flags = [[0_i8; 11]; 1];
        let smc_weights = [0.0_f64; 11];

        let base = scenario::base_scenario(0, 0, close.len());
        // 4 pips and $9/lot against the fixture's 1 pip and $2/lot.
        let dear = scenario::cost_scenario(
            0,
            1,
            close.len(),
            scenario::spread_ticks_exact(4.0),
            scenario::commission_micros_exact(9.0),
        );
        let rows = validation_backtest_scenarios_cpu_test_oracle(
            inputs(
                &close,
                &indicators,
                &smc,
                &months,
                &settings,
                &offsets,
                &idx,
                &w,
                &lt,
                &st,
                &sl,
                &tp,
                &flags,
                &smc_weights,
            ),
            &[base, dear],
        )
        .expect("cost scenarios are reproducible on the CPU");

        assert_eq!(rows.len(), 2);
        // The strategy trades, so paying four times the spread and 4.5x the
        // commission has to show up as strictly less money.
        assert!(rows[0][8] > 0.0, "the fixture must actually trade");
        assert!(
            rows[1][0] < rows[0][0],
            "a dearer scenario reported {} against the cheap one's {}",
            rows[1][0],
            rows[0][0]
        );
    }

    /// The CPU mirror must evaluate the PERTURBED gene, not the base gene.
    ///
    /// This is the property that makes it an independent parity oracle rather
    /// than 100 copies of the unperturbed result.
    #[test]
    fn a_perturbed_scenario_is_not_the_unperturbed_gene() {
        let (close, indicators, smc, months, settings) = fixture();
        let offsets = [0_i32, 2];
        let idx = [0_i32, 1];
        let w = [1.0_f64, 0.5];
        let lt = [0.3_f64];
        let st = [-0.3_f64];
        let sl = [20.0_f64];
        let tp = [40.0_f64];
        let flags = [[0_i8; 11]; 1];
        let smc_weights = [0.0_f64; 11];

        let mut work_list = vec![scenario::base_scenario(0, 0, close.len())];
        for run in 0..8u64 {
            work_list.push(scenario::perturb_scenario(
                0,
                1 + run,
                close.len(),
                7717 ^ run,
            ));
        }
        let rows = validation_backtest_scenarios_cpu_test_oracle(
            inputs(
                &close,
                &indicators,
                &smc,
                &months,
                &settings,
                &offsets,
                &idx,
                &w,
                &lt,
                &st,
                &sl,
                &tp,
                &flags,
                &smc_weights,
            ),
            &work_list,
        )
        .expect("device-perturbation scenarios are reproducible on the CPU");

        assert_eq!(rows.len(), 9);
        let distinct = rows[1..].iter().filter(|row| row[0] != rows[0][0]).count();
        assert!(
            distinct >= 4,
            "only {distinct} of 8 perturbations moved net profit — the mirror is \
             probably evaluating the unperturbed gene"
        );
        // And it is deterministic: the same counters give the same numbers.
        let again = validation_backtest_scenarios_cpu_test_oracle(
            inputs(
                &close,
                &indicators,
                &smc,
                &months,
                &settings,
                &offsets,
                &idx,
                &w,
                &lt,
                &st,
                &sl,
                &tp,
                &flags,
                &smc_weights,
            ),
            &work_list,
        )
        .unwrap();
        assert_eq!(rows, again);
    }

    /// What the mirror cannot reproduce, it refuses.
    #[test]
    fn the_mirror_refuses_a_window_it_cannot_walk() {
        let (close, indicators, smc, months, settings) = fixture();
        let offsets = [0_i32, 1];
        let idx = [0_i32];
        let w = [1.0_f64];
        let lt = [0.3_f64];
        let st = [-0.3_f64];
        let sl = [20.0_f64];
        let tp = [40.0_f64];
        let flags = [[0_i8; 11]; 1];
        let smc_weights = [0.0_f64; 11];

        let mut windowed = scenario::base_scenario(0, 0, close.len());
        windowed.window_offset = 50;
        windowed.window_len = 100;
        let error = validation_backtest_scenarios_cpu_test_oracle(
            inputs(
                &close,
                &indicators,
                &smc,
                &months,
                &settings,
                &offsets,
                &idx,
                &w,
                &lt,
                &st,
                &sl,
                &tp,
                &flags,
                &smc_weights,
            ),
            &[windowed],
        )
        .expect_err("a sub-window must be refused, not silently widened");
        assert!(format!("{error}").contains("whole series"), "{error}");
    }
}

#[cfg(test)]
mod overrides_tests {
    use super::*;

    #[test]
    fn backtest_runtime_overrides_defaults_match_legacy_env_defaults() {
        let defaults = BacktestRuntimeOverrides::default();
        assert!((defaults.initial_equity - 100_000.0).abs() < 1e-9);
        assert_eq!(defaults.month_capacity, 240);
    }

    #[test]
    fn backtest_from_settings_default_matches_env_default() {
        // Behavior-preservation gate (config-consolidation S2d): a fresh
        // `Settings` reproduces the engine backtest defaults exactly.
        let s = neoethos_core::Settings::default();
        assert_eq!(
            BacktestRuntimeOverrides::from_settings(&s),
            BacktestRuntimeOverrides::default()
        );
    }

    #[test]
    fn backtest_settings_methods_use_typed_overrides() {
        // Without a process-wide install the BacktestSettings accessors must
        // return the audited defaults rather than reading the environment
        // directly each call.
        let settings = BacktestSettings::default();
        assert!((settings.initial_equity() - 100_000.0).abs() < 1e-9);
        assert_eq!(settings.month_capacity(), 240);
    }

    #[test]
    fn session_spread_profile_buckets_by_utc_hour() {
        let profile = SessionSpreadProfile {
            asian_pips: 1.8,
            overlap_pips: 0.5,
            late_ny_pips: 1.0,
        };
        // 02:00 UTC → Asian bucket
        let asian = profile.spread_pips_at(2 * 3_600_000);
        // 09:00 UTC → London/NY overlap
        let overlap = profile.spread_pips_at(9 * 3_600_000);
        // 18:00 UTC → late NY
        let late_ny = profile.spread_pips_at(18 * 3_600_000);
        // 23:30 UTC → Asian (wraps around midnight)
        let pre_asian = profile.spread_pips_at(23 * 3_600_000 + 30 * 60_000);

        assert!((asian - 1.8).abs() < 1e-9);
        assert!((overlap - 0.5).abs() < 1e-9);
        assert!((late_ny - 1.0).abs() < 1e-9);
        assert!((pre_asian - 1.8).abs() < 1e-9);
    }

    #[test]
    fn current_backtest_runtime_overrides_falls_back_to_defaults() {
        // Without a process-wide install, the current-overrides accessor
        // must surface the audited defaults rather than panicking or
        // reading the environment.
        let observed = current_backtest_runtime_overrides();
        // We cannot assume the OnceLock is unset (other tests in the same
        // process may have installed it), but the returned value must at
        // least be one of the legal configurations: either the documented
        // defaults or whatever was installed earlier.
        assert!(observed.initial_equity.is_finite() && observed.initial_equity > 0.0);
        assert!(observed.month_capacity > 0);
    }

    #[test]
    fn daily_drawdown_tracks_intraday_peak_and_finalizes_at_day_boundary() {
        let close = [100.0, 100.0, 10_100.0, 100.0, -4_900.0, -4_900.0];
        let high = [100.0, 100.0, 10_100.0, 100.0, 100.0, -4_900.0];
        let low = [100.0, 100.0, 10_100.0, 100.0, -4_900.0, -4_900.0];
        // First long: +10k target on bar 2. Second long: -5k stop on bar 4.
        let signals = [1_i8, 0, 1, 0, 0, 0];
        let months = [0_i64; 6];
        let days = [0_i64, 0, 0, 0, 0, 1];
        let mut settings = BacktestSettings::default();
        settings.sl_pips = 5_000.0;
        settings.tp_pips = 10_000.0;
        settings.pip_value = 1.0;
        settings.pip_value_per_lot = 1.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.risk_based_sizing = false;

        let metrics = fast_evaluate_strategy_core(
            &close,
            &high,
            &low,
            &signals,
            &[],
            &months,
            &days,
            &[],
            &settings,
        );
        let expected = 5_000.0 / 110_000.0;
        assert_eq!(metrics[8], 2.0);
        assert!(
            (metrics[10] - expected).abs() < 1.0e-12,
            "daily DD must be (110k-105k)/110k={expected}, got {}",
            metrics[10]
        );
    }

    #[test]
    fn daily_drawdown_preserves_a_trough_before_a_later_same_day_peak() {
        let close = [100.0, 100.0, -9_900.0, 100.0, 20_100.0, 20_100.0];
        let high = [100.0, 100.0, 100.0, 100.0, 20_100.0, 20_100.0];
        let low = [100.0, 100.0, -9_900.0, 100.0, 20_100.0, 20_100.0];
        // Realized equity path: 100k -> 90k -> 110k -> 110k, all on day zero.
        let signals = [1_i8, 0, 1, 0, 0, 0];
        let months = [0_i64; 6];
        let days = [0_i64, 0, 0, 0, 0, 1];
        let mut settings = BacktestSettings::default();
        settings.sl_pips = 10_000.0;
        settings.tp_pips = 20_000.0;
        settings.pip_value = 1.0;
        settings.pip_value_per_lot = 1.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.risk_based_sizing = false;

        let metrics = fast_evaluate_strategy_core(
            &close,
            &high,
            &low,
            &signals,
            &[],
            &months,
            &days,
            &[],
            &settings,
        );

        assert_eq!(metrics[8], 2.0);
        assert!(
            (metrics[10] - 0.10).abs() < 1.0e-12,
            "the 100k -> 90k segment must remain a 10% daily drawdown after the 110k peak, got {}",
            metrics[10]
        );
    }

    // ─── Phase C.2 carry-cost + conversion-fee helper ────────────────
    //
    // These tests pin the math used by every trade-close branch of the
    // CPU evaluator. Both close branches call the size-aware helper, so a
    // regression here would corrupt every backtest's PnL.

    fn settings_with_carry(
        swap_long: f64,
        swap_short: f64,
        conv_fee: f64,
        pip_value_per_lot: f64,
    ) -> BacktestSettings {
        let mut s = BacktestSettings::default();
        s.swap_long_pips_per_day = swap_long;
        s.swap_short_pips_per_day = swap_short;
        s.pnl_conversion_fee_rate = conv_fee;
        s.pip_value_per_lot = pip_value_per_lot;
        s
    }

    #[test]
    fn carry_fee_zero_zero_is_identity() {
        let s = settings_with_carry(0.0, 0.0, 0.0, 10.0);
        // Day-trade (entry == exit): no swap, no fee → gross.
        assert!((apply_carry_and_fee(123.45, 123.45, 1, 0, 0, &s) - 123.45).abs() < 1e-9);
        // Long trade held 5 days, zero swap & fee → still gross.
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 5 * 86_400_000;
        assert!((apply_carry_and_fee(123.45, 123.45, 1, entry, exit, &s) - 123.45).abs() < 1e-9);
    }

    #[test]
    fn carry_fee_negative_swap_reduces_pnl_for_long() {
        // EURUSD-style: swap_long = −2.445 pips/day, pip_value_per_lot = $10.
        // Long held 5.0 days → carry = −2.445 × 5 × 10 = −$122.25.
        // Gross $200 → net $77.75. No fee.
        let s = settings_with_carry(-2.445, -0.105, 0.0, 10.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 5 * 86_400_000;
        let net = apply_carry_and_fee(200.0, 200.0, 1, entry, exit, &s);
        assert!((net - 77.75).abs() < 1e-6, "expected ~77.75, got {net}");
    }

    #[test]
    fn carry_fee_positive_swap_credits_short() {
        // XTIUSD-style: swap_short = +0.4375 pips/day, pip_value_per_lot = $1.
        // Short held 4.0 days → carry = +0.4375 × 4 × 1 = +$1.75 credit.
        let s = settings_with_carry(-0.5, 0.4375, 0.0, 1.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 4 * 86_400_000;
        let net = apply_carry_and_fee(10.0, 10.0, -1, entry, exit, &s);
        assert!((net - 11.75).abs() < 1e-6, "expected ~11.75, got {net}");
    }

    #[test]
    fn carry_fee_fractional_days() {
        // 12 hours = 0.5 days. swap = −1.0, pip_value = 10 → carry = −5.0.
        let s = settings_with_carry(-1.0, -1.0, 0.0, 10.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 12 * 3_600_000;
        let net = apply_carry_and_fee(50.0, 50.0, 1, entry, exit, &s);
        assert!((net - 45.0).abs() < 1e-6, "expected ~45.0, got {net}");
    }

    #[test]
    fn carry_fee_conversion_debits_price_gross() {
        // No swap/commission, fee = 0.005. Gross $100 → net $99.50.
        let s = settings_with_carry(0.0, 0.0, 0.005, 10.0);
        let net = apply_carry_and_fee(100.0, 100.0, 1, 0, 0, &s);
        assert!((net - 99.5).abs() < 1e-6, "expected 99.5, got {net}");
    }

    #[test]
    fn carry_fee_handles_missing_timestamps_as_day_trade() {
        // entry_ts = 0 means "no timestamp data": skip swap entirely.
        let s = settings_with_carry(-100.0, -100.0, 0.0, 10.0);
        let net = apply_carry_and_fee(50.0, 50.0, 1, 0, 1_700_000_000_000, &s);
        assert!(
            (net - 50.0).abs() < 1e-9,
            "expected 50.0 (no swap), got {net}"
        );
    }

    #[test]
    fn carry_fee_rejects_inverted_timestamps() {
        // exit < entry: no negative time, no swap charge.
        let s = settings_with_carry(-1.0, -1.0, 0.0, 10.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry - 86_400_000;
        let net = apply_carry_and_fee(50.0, 50.0, 1, entry, exit, &s);
        assert!((net - 50.0).abs() < 1e-9);
    }

    #[test]
    fn carry_fee_rejects_out_of_range_conversion_fee() {
        for rate in [1.0, -0.1, f64::NAN, f64::INFINITY] {
            let s = settings_with_carry(0.0, 0.0, rate, 10.0);
            assert!(
                std::panic::catch_unwind(|| apply_carry_and_fee(100.0, 100.0, 1, 0, 0, &s))
                    .is_err()
            );
            assert!(validate_sizing_confidences(0, &[], &s).is_err());
        }
    }

    #[test]
    fn research_fee_never_rebates_losses_or_discounts_commission_and_swap() {
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 86_400_000;
        for gross in [100.0_f64, -100.0, 0.0] {
            for swap in [-2.0_f64, 2.0] {
                for lots in [0.25_f64, 1.0, 2.5] {
                    let s = settings_with_carry(swap, -swap, 0.005, 10.0);
                    for side in [1_i8, -1] {
                        let costed = (gross - 7.0) * lots;
                        let carry = if side == 1 { swap } else { -swap } * 10.0 * lots;
                        let expected = costed + carry - gross.abs() * lots * 0.005;
                        let actual = apply_carry_and_fee_scaled(
                            costed,
                            gross * lots,
                            lots,
                            side,
                            entry,
                            exit,
                            &s,
                        );
                        assert_eq!(actual.to_bits(), expected.to_bits());
                        assert!(actual <= costed + carry);
                    }
                }
            }
        }
    }

    #[test]
    fn zero_fee_preserves_legacy_cost_and_swap_bits_for_all_lot_paths() {
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 123_456_789;
        for rate in [0.0, -0.0] {
            for lots in [0.17, 1.0, 3.71] {
                for price in [137.123456789_f64, -137.123456789, -0.0] {
                    let s = settings_with_carry(-2.445, 0.4375, rate, 9.173);
                    let costed = price * lots - (7.123 + 0.9173) * lots;
                    let days = (exit - entry) as f64 / 86_400_000.0;
                    let carry = if lots == 1.0 {
                        -2.445 * days * 9.173
                    } else {
                        -2.445 * days * 9.173 * lots
                    };
                    let expected = costed + carry;
                    // Fee zero must not read/reconstruct the extra fee basis.
                    let actual =
                        apply_carry_and_fee_scaled(costed, f64::NAN, lots, 1, entry, exit, &s);
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn full_cpu_close_branches_charge_fee_on_price_gross_and_match_ledger() {
        let entry = 1_700_000_000_000_i64;
        for gap in [false, true] {
            for gross_move in [100.0_f64, -100.0, 0.0] {
                for side in [1_i8, -1] {
                    let times = [entry, entry + 1_000, entry + 86_401_000, entry + 86_402_000];
                    let close = [
                        1_000.0,
                        1_000.0,
                        1_000.0 + gross_move * f64::from(side),
                        1_000.0,
                    ];
                    let settings = BacktestSettings {
                        pip_value: 1.0,
                        pip_value_per_lot: 1.0,
                        spread_pips: 2.0,
                        commission_per_trade: 7.0,
                        sl_pips: 10_000.0,
                        tp_pips: 10_000.0,
                        max_hold_bars: 1,
                        gap_threshold_ms: if gap { 10_000 } else { 0 },
                        risk_based_sizing: false,
                        kill_zones_enabled: false,
                        swap_long_pips_per_day: -2.0,
                        swap_short_pips_per_day: 3.0,
                        pnl_conversion_fee_rate: 0.005,
                        ..BacktestSettings::default()
                    };
                    let signals = [side, 0, 0, 0];
                    let months = [0; 4];
                    let days = times.map(|t| t / 86_400_000);
                    let (metrics, trades) = evaluate_strategy_with_confidence_and_ledger_core(
                        &close,
                        &close,
                        &close,
                        &signals,
                        &[],
                        &months,
                        &days,
                        &times,
                        &settings,
                    )
                    .unwrap();
                    let fast = fast_evaluate_strategy_core(
                        &close,
                        &close,
                        &close,
                        &signals,
                        &[],
                        &months,
                        &days,
                        &times,
                        &settings,
                    );
                    let price_gross = gross_move - 2.0; // both modeled half-spreads, no commission
                    let expected = price_gross - 7.0 + if side == 1 { -2.0 } else { 3.0 }
                        - price_gross.abs() * 0.005;
                    assert_eq!(trades.len(), 1);
                    assert_eq!(trades[0].exit_time, Some(times[2]));
                    assert!(
                        (trades[0].pnl - expected).abs() < 1e-10,
                        "gap={gap}, side={side}, move={gross_move}"
                    );
                    assert_eq!(metrics.map(f64::to_bits), fast.map(f64::to_bits));
                    // Net profit is the realized balance delta, not the direct
                    // trade field: adding to initial equity can round first.
                    let initial_equity = settings.initial_equity();
                    let realized_delta = (initial_equity + trades[0].pnl) - initial_equity;
                    assert_eq!(metrics[0].to_bits(), realized_delta.to_bits());
                    assert!((metrics[0] - expected).abs() < 1e-10);
                }
            }
        }
    }

    // ─── Risk-based, confidence-scaled sizing (Phase 1) ──────────────────
    //
    // A cost-free fixture: one long entry that hits the stop-loss exactly,
    // no spread / commission / swap / conversion fee, so the realized loss
    // is purely the SL move × the entry-captured pos_lots.

    /// Build a clean backtest fixture for the sizing tests. The single long
    /// trade enters at bar 1 (signal observed at bar 0) and is stopped out at
    /// bar 2 because `low[2]` dives well below the stop. With zero costs the
    /// only realized PnL is the SL loss × pos_lots.
    ///
    /// Returns the metrics array from `fast_evaluate_strategy_core`. The
    /// caller picks `sl_pips`, `risk_based_sizing`, and the risk bounds.
    fn run_single_sl_trade(
        sl_pips: f64,
        risk_based_sizing: bool,
        risk_min: f64,
        risk_max: f64,
        confidences: &[f64],
    ) -> [f64; 11] {
        let pip = 0.0001_f64;
        let pip_value_per_lot = 10.0_f64;
        // Entry fills at close[1] = 1.0000. Stop sits sl_pips below.
        // low[2] is forced far below the deepest stop we test (sl=40) so the
        // SL always triggers at bar 2 regardless of sl_pips.
        let close = vec![1.0000_f64, 1.0000, 0.9900, 0.9900];
        let high = vec![1.0001_f64, 1.0001, 1.0001, 1.0001];
        // low[2] = 0.9900 → 100 pips below entry, well past any tested SL,
        // and below TP-side too (this is a long, so low only matters for SL).
        let low = vec![0.9999_f64, 0.9999, 0.9900, 0.9900];
        // Signal at index 0 → entry at bar 1; flat afterwards.
        let signals = vec![1_i8, 0, 0, 0];
        let months = vec![0_i64; 4];
        let days = vec![0_i64; 4];

        let mut settings = BacktestSettings::default();
        settings.sl_pips = sl_pips;
        settings.tp_pips = 10_000.0; // never hit
        settings.max_hold_bars = 0; // no max-hold exit
        settings.min_hold_bars = 0;
        settings.pip_value = pip;
        settings.pip_value_per_lot = pip_value_per_lot;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = risk_based_sizing;
        settings.risk_per_trade_min = risk_min;
        settings.risk_per_trade_max = risk_max;
        settings.high_quality_confidence = 0.65;

        fast_evaluate_strategy_core(
            &close,
            &high,
            &low,
            &signals,
            confidences,
            &months,
            &days,
            &[],
            &settings,
        )
    }

    #[test]
    fn risk_sizing_full_sl_loses_risk_pct() {
        // Force risk_pct = 1% by pinning min == max. Confidence is full.
        let risk = 0.01_f64;
        let conf = vec![1.0_f64; 4];
        let initial_equity = BacktestSettings::default().initial_equity();
        let expected_loss = -risk * initial_equity; // -1% of entry equity

        // Two DIFFERENT stop distances must yield the SAME % loss, proving
        // the loss is risk-driven and INDEPENDENT of sl_pips.
        for sl_pips in [20.0_f64, 40.0_f64] {
            let m = run_single_sl_trade(sl_pips, true, risk, risk, &conf);
            let net_profit = m[0];
            let trade_count = m[8];
            assert_eq!(
                trade_count, 1.0,
                "expected exactly one trade (sl={sl_pips})"
            );
            assert!(
                (net_profit - expected_loss).abs() < 1e-6,
                "sl={sl_pips}: full-SL loss should be {expected_loss} (1% of {initial_equity}), got {net_profit}"
            );
        }
    }

    #[test]
    fn risk_sizing_disabled_is_legacy() {
        // risk_based_sizing = false → fixed 1 lot. The realized loss must be
        // exactly sl_pips × pip_value_per_lot (the legacy fixed-1-lot path),
        // and must SCALE with sl_pips (unlike the risk-based path).
        let pip_value_per_lot = 10.0_f64;
        let conf = vec![1.0_f64; 4]; // ignored when sizing is disabled
        for sl_pips in [20.0_f64, 40.0_f64] {
            let m = run_single_sl_trade(sl_pips, false, 0.01, 0.01, &conf);
            let net_profit = m[0];
            let expected = -sl_pips * pip_value_per_lot; // fixed 1 lot
            assert_eq!(m[8], 1.0, "expected exactly one trade (sl={sl_pips})");
            assert!(
                (net_profit - expected).abs() < 1e-9,
                "sl={sl_pips}: legacy fixed-1-lot loss should be {expected}, got {net_profit}"
            );
        }

        // Also assert that an EMPTY confidence slice forces legacy behaviour
        // even when risk_based_sizing is true.
        let m = run_single_sl_trade(20.0, true, 0.01, 0.01, &[]);
        assert!(
            (m[0] - (-20.0 * pip_value_per_lot)).abs() < 1e-9,
            "empty confidence slice must force fixed-1-lot, got {}",
            m[0]
        );
    }

    #[test]
    fn risk_sizing_subpip_stops_use_the_actual_stop_not_a_one_pip_floor() {
        let risk = 0.0001;
        let expected_loss = -risk * BacktestSettings::default().initial_equity();
        for stop in [0.25, 0.5, 1.0, 20.0] {
            let metrics = run_single_sl_trade(stop, true, risk, risk, &[1.0; 4]);
            assert_eq!(metrics[8], 1.0);
            assert!(
                (metrics[0] - expected_loss).abs() < 1e-6,
                "stop={stop}: expected {expected_loss}, got {}",
                metrics[0]
            );
        }
    }

    #[test]
    fn risk_sizing_invalid_or_zero_exposure_is_not_a_trade() {
        let settings = BacktestSettings {
            pip_value_per_lot: 10.0,
            ..BacktestSettings::default()
        };
        for stop in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(risk_based_pos_lots(1.0, 100_000.0, stop, &settings), 0.0);
        }
        let metrics = run_single_sl_trade(20.0, true, 0.0, 0.0, &[1.0; 4]);
        assert_eq!(
            metrics[8], 0.0,
            "zero exposure must not consume a trade slot"
        );
    }

    #[test]
    fn adaptive_unavailable_cells_skip_entries_without_consuming_daily_quota() {
        let close = [1.0; 7];
        let high = [1.0; 7];
        let mut low = [1.0; 7];
        low[6] = 0.99;
        let signals = [1_i8; 7];
        let times: Vec<i64> = (0..7).map(|i| 1_700_000_000_000 + i * 300_000).collect();
        let days = [times[0] / 86_400_000; 7];
        let settings = BacktestSettings {
            sl_pips: 20.0,
            tp_pips: 40.0,
            pip_value: 0.0001,
            pip_value_per_lot: 10.0,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            kill_zones_enabled: false,
            risk_based_sizing: false,
            max_trades_per_day: 1,
            adaptive_vol_mult: 1.0,
            adaptive_base_pips: Some(
                vec![f64::NAN, 0.0, -1.0, f64::INFINITY, 20.0, f64::NAN, 20.0].into(),
            ),
            ..BacktestSettings::default()
        };
        let metrics = fast_evaluate_strategy_core(
            &close,
            &high,
            &low,
            &signals,
            &[],
            &[0; 7],
            &days,
            &times,
            &settings,
        );
        let trades = simulate_trades_core(&close, &high, &low, &times, &signals, &settings);
        assert_eq!(metrics[8], 1.0);
        assert_eq!(trades.len(), 1);
        assert_eq!(
            trades[0].entry_time, times[5],
            "only signal bar 4 has an available stop"
        );
        assert_eq!(trades[0].exit_time, Some(times[6]));
        assert!((trades[0].pnl + 200.0).abs() < 1e-8);
        assert!((trades[0].r_multiple + 1.0).abs() < 1e-10);
        // The logger converts the fill price back to pips; the fast evaluator
        // books the stop directly. Allow floating-point roundoff, not a change
        // to the loss, entry time, or trade count checked above.
        let pnl_tolerance =
            64.0 * f64::EPSILON * metrics[0].abs().max(trades[0].pnl.abs()).max(1.0);
        assert!((metrics[0] - trades[0].pnl).abs() <= pnl_tolerance);
    }

    #[test]
    fn adaptive_entry_cannot_borrow_the_next_bars_stop_or_fixed_fallback() {
        let mut settings = BacktestSettings::default();
        settings.adaptive_vol_mult = 1.0;
        settings.adaptive_base_pips = Some(vec![f64::NAN, 20.0].into());
        assert_eq!(entry_sl_tp_pips(&settings, 0), None);
        assert_eq!(entry_sl_tp_pips(&settings, 1), Some((20.0, 40.0)));
        assert_eq!(entry_sl_tp_pips(&settings, 2), None);
        settings.adaptive_vol_mult = 0.0;
        assert_eq!(entry_sl_tp_pips(&settings, 0), Some((20.0, 40.0)));
        settings.adaptive_vol_mult = 1.0;
        settings.adaptive_base_pips = None;
        assert_eq!(entry_sl_tp_pips(&settings, 0), Some((20.0, 40.0)));
    }

    #[test]
    fn adaptive_stops_constant_series_equals_scalar_and_varying_changes_outcome() {
        // Long entry at bar 1 (signal at bar 0). Price drifts down so a 15-pip
        // stop is hit at bar 3 but a very wide stop is not.
        let close = vec![1.0000_f64, 1.0000, 0.9990, 0.9980, 0.9975];
        let high = vec![1.0002_f64, 1.0002, 1.0002, 1.0002, 1.0002];
        let low = vec![0.9998_f64, 0.9998, 0.9989, 0.9979, 0.9974];
        let signals = vec![1_i8, 0, 0, 0, 0];
        let months = vec![0_i64; 5];
        let days = vec![0_i64; 5];
        let n = close.len();

        let mut base = BacktestSettings::default();
        base.sl_pips = 15.0;
        base.tp_pips = 30.0;
        base.max_hold_bars = 0;
        base.min_hold_bars = 0;
        base.pip_value = 0.0001;
        base.pip_value_per_lot = 10.0;
        base.spread_pips = 0.0;
        base.commission_per_trade = 0.0;
        base.swap_long_pips_per_day = 0.0;
        base.swap_short_pips_per_day = 0.0;
        base.pnl_conversion_fee_rate = 0.0;
        base.kill_zones_enabled = false;
        base.risk_based_sizing = false; // fixed 1 lot → deterministic PnL

        let run = |s: &BacktestSettings| {
            fast_evaluate_strategy_core(&close, &high, &low, &signals, &[], &months, &days, &[], s)
        };

        // (1) fixed scalar path.
        let fixed = run(&base);
        assert_eq!(
            fixed[8], 1.0,
            "the 15-pip scalar stop should produce one trade"
        );

        // (2) an adaptive series IDENTICAL to the scalar must be byte-identical.
        let mut same = base.clone();
        // base stop 15p, mult 1, gene rr 30/15 = 2 → scalar path exactly.
        // Deliberately poison the legacy scalar: valid genes must not read it.
        same.adaptive_base_pips = Some(vec![15.0_f64; n].into());
        same.adaptive_vol_mult = 1.0;
        same.adaptive_rr = 9.0;
        let same_m = run(&same);
        for k in 0..11 {
            assert_eq!(
                fixed[k].to_bits(),
                same_m[k].to_bits(),
                "constant adaptive series must be byte-identical to the scalar path (slot {k})"
            );
        }

        // (3) a much WIDER stop captured at the entry (signal) bar 0 → the stop
        // that fired for the tight scalar no longer fires → different outcome.
        let mut wide = base.clone();
        let mut base_series = vec![15.0_f64; n];
        base_series[0] = 500.0; // entry captures the signal bar (i-1 = 0)
        wide.adaptive_base_pips = Some(base_series.into());
        wide.adaptive_vol_mult = 1.0;
        wide.adaptive_rr = 2.0;
        let wide_m = run(&wide);
        assert!(
            wide_m[8].to_bits() != fixed[8].to_bits() || wide_m[0].to_bits() != fixed[0].to_bits(),
            "a 500-pip adaptive stop must change the trade outcome vs the 15-pip scalar"
        );
    }

    #[test]
    fn trailing_stop_moves_to_break_even_and_saves_the_trade() {
        // Long entry at bar 1 (signal at bar 0), entry = 1.0000, sl = 20 pips
        // (1R = 0.0020). Bar 2 runs +1R (high 1.0021) → the trail ratchets the
        // stop up to ~entry. Bar 3 reverses through the ORIGINAL stop
        // (low 0.9975, below 0.9980). With trailing ON the position exits at the
        // ratcheted ~break-even stop; with trailing OFF it runs to the full
        // 20-pip loss. This test pins the MECHANISM, not the policy: since
        // 2026-08-09 discovery runs with the trail OFF by default
        // (`models.exit_policy.trailing_enabled`), because leaving it on capped
        // the realised payoff near 1.0 and made the 2.0 payoff floor unreachable.
        // The mechanism must still be correct for the operator who turns it on.
        let close = vec![1.0000_f64, 1.0000, 1.0010, 0.9975, 0.9975];
        let high = vec![1.0001_f64, 1.0001, 1.0021, 1.0000, 0.9976];
        let low = vec![0.9999_f64, 1.0000, 1.0000, 0.9975, 0.9974];
        let signals = vec![1_i8, 0, 0, 0, 0];
        let months = vec![0_i64; 5];
        let days = vec![0_i64; 5];

        let mut base = BacktestSettings::default();
        base.sl_pips = 20.0;
        base.tp_pips = 40.0; // far — never hit here
        base.max_hold_bars = 0;
        base.min_hold_bars = 0;
        base.pip_value = 0.0001;
        base.pip_value_per_lot = 10.0;
        base.spread_pips = 0.0;
        base.commission_per_trade = 0.0;
        base.swap_long_pips_per_day = 0.0;
        base.swap_short_pips_per_day = 0.0;
        base.pnl_conversion_fee_rate = 0.0;
        base.kill_zones_enabled = false;
        base.risk_based_sizing = false; // fixed 1 lot → deterministic PnL
        base.trailing_be_trigger_r = 1.0;
        base.trailing_atr_multiplier = 1.0;

        let run = |s: &BacktestSettings| {
            fast_evaluate_strategy_core(&close, &high, &low, &signals, &[], &months, &days, &[], s)
        };

        let mut off = base.clone();
        off.trailing_enabled = false;
        let m_off = run(&off);
        let mut on = base.clone();
        on.trailing_enabled = true;
        let m_on = run(&on);

        // Both close exactly one trade on bar 3.
        assert_eq!(m_off[8], 1.0, "trailing OFF should close one trade");
        assert_eq!(m_on[8], 1.0, "trailing ON should close one trade");
        // Trailing OFF runs to the full 20-pip stop: -20 * pip_value_per_lot.
        assert!(
            (m_off[0] - (-20.0 * 10.0)).abs() < 1e-6,
            "trailing OFF should take the full 20-pip loss, got {}",
            m_off[0]
        );
        // Trailing ON exits at ~break-even → dramatically better than the full
        // loss (the whole point of the break-even trail).
        assert!(
            m_on[0] > m_off[0] + 150.0,
            "trailing ON (break-even) should save most of the 20-pip loss: on={}, off={}",
            m_on[0],
            m_off[0]
        );
        assert!(
            m_on[0] >= -1.0 * 10.0,
            "trailing ON should exit at ~break-even (>= -1 pip), got {}",
            m_on[0]
        );
    }
}

#[cfg(test)]
mod protective_exit_tests {
    use super::*;

    fn protected_settings() -> BacktestSettings {
        BacktestSettings {
            sl_pips: 20.0,
            tp_pips: 40.0,
            min_hold_bars: 10,
            max_hold_bars: 0,
            pip_value: 0.0001,
            pip_value_per_lot: 10.0,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            kill_zones_enabled: false,
            risk_based_sizing: false,
            gap_threshold_ms: 0,
            trailing_enabled: false,
            ..BacktestSettings::default()
        }
    }

    fn assert_protected_exit(
        close: &[f64],
        high: &[f64],
        low: &[f64],
        signals: &[i8],
        settings: &BacktestSettings,
        exit_bar: usize,
        expected_pnl: f64,
    ) {
        let timestamps: Vec<i64> = (0..close.len())
            .map(|i| 1_700_000_000_000 + i as i64 * 300_000)
            .collect();
        let months = vec![0; close.len()];
        let days: Vec<i64> = timestamps.iter().map(|ts| ts / 86_400_000).collect();
        let run = |settings: &BacktestSettings| {
            (
                fast_evaluate_strategy_core(
                    close,
                    high,
                    low,
                    signals,
                    &[],
                    &months,
                    &days,
                    &timestamps,
                    settings,
                ),
                simulate_trades_core(close, high, low, &timestamps, signals, settings),
            )
        };
        let (metrics, trades) = run(settings);
        assert_eq!(metrics[8], 1.0, "fast evaluator lost the protected exit");
        assert_eq!(trades.len(), 1, "trade logger lost the protected exit");
        assert_eq!(trades[0].entry_time, timestamps[1]);
        assert_eq!(trades[0].exit_time, Some(timestamps[exit_bar]));
        for pnl in [metrics[0], trades[0].pnl] {
            assert!(
                (pnl - expected_pnl).abs() < 1e-8,
                "expected independently calculated PnL {expected_pnl}, got {pnl}"
            );
        }

        // A minimum-hold preference must not change any protective exit or
        // deadline. Compare with the established zero-minimum behavior too,
        // but keep the independent price/time/money assertions above.
        let mut no_minimum = settings.clone();
        no_minimum.min_hold_bars = 0;
        let (baseline_metrics, baseline_trades) = run(&no_minimum);
        assert_eq!(
            metrics.map(f64::to_bits),
            baseline_metrics.map(f64::to_bits)
        );
        assert_eq!(trades[0].exit_time, baseline_trades[0].exit_time);
        assert_eq!(trades[0].pnl.to_bits(), baseline_trades[0].pnl.to_bits());
    }

    #[test]
    fn minimum_hold_never_delays_long_or_short_stop_and_target() {
        for side in [1_i8, -1] {
            for target in [false, true] {
                let close = [1.0; 4];
                let mut high = [1.0001; 4];
                let mut low = [0.9999; 4];
                match (side, target) {
                    (1, false) => low[2] = 0.997,
                    (1, true) => high[2] = 1.006,
                    (-1, false) => high[2] = 1.003,
                    (-1, true) => low[2] = 0.994,
                    _ => unreachable!(),
                }
                assert_protected_exit(
                    &close,
                    &high,
                    &low,
                    &[side, 0, 0, 0],
                    &protected_settings(),
                    2,
                    if target { 400.0 } else { -200.0 },
                );
            }
        }
    }

    #[test]
    fn minimum_hold_never_delays_a_trail_and_the_trail_stays_causal() {
        for side in [1_i8, -1] {
            let (close, high, low) = if side == 1 {
                (
                    [1.0, 1.0, 1.002, 1.0, 1.0],
                    [1.0001, 1.0001, 1.003, 1.002, 1.0001],
                    [0.9999, 0.9999, 1.0, 0.999, 0.9999],
                )
            } else {
                (
                    [1.0, 1.0, 0.998, 1.0, 1.0],
                    [1.0001, 1.0001, 1.0, 1.001, 1.0001],
                    [0.9999, 0.9999, 0.997, 0.998, 0.9999],
                )
            };
            let settings = BacktestSettings {
                tp_pips: 100.0,
                trailing_enabled: true,
                trailing_be_trigger_r: 1.0,
                trailing_atr_multiplier: 0.5,
                trailing_min_lock_pips: 2.0,
                ..protected_settings()
            };
            // Bar 2 establishes a +20-pip trail. Its own opposite extreme is
            // already beyond that NEW level, which must protect only bar 3.
            assert_protected_exit(
                &close,
                &high,
                &low,
                &[side, 0, 0, 0, 0],
                &settings,
                3,
                200.0,
            );
        }
    }

    #[test]
    fn minimum_hold_cannot_extend_the_maximum_holding_deadline() {
        let close = [1.0, 1.0, 1.0, 1.001, 1.001];
        let high = close.map(|value| value + 0.0001);
        let low = close.map(|value| value - 0.0001);
        for side in [1_i8, -1] {
            let settings = BacktestSettings {
                max_hold_bars: 2,
                ..protected_settings()
            };
            assert_protected_exit(
                &close,
                &high,
                &low,
                &[side, 0, 0, 0, 0],
                &settings,
                3,
                side as f64 * 100.0,
            );
        }
    }

    #[test]
    fn protection_wins_when_stop_target_and_timeout_share_a_bar() {
        let settings = BacktestSettings {
            max_hold_bars: 2,
            ..protected_settings()
        };
        for side in [1_i8, -1] {
            assert_protected_exit(
                &[1.0, 1.0, 1.0, 1.001],
                &[1.0001, 1.0001, 1.0001, 1.006],
                &[0.9999, 0.9999, 0.9999, 0.994],
                &[side, 0, 0, 0],
                &settings,
                3,
                -200.0,
            );
        }
    }

    #[test]
    fn opposite_signals_do_not_replace_the_bracket_only_exit_policy() {
        let settings = BacktestSettings {
            max_hold_bars: 3,
            ..protected_settings()
        };
        assert_protected_exit(
            &[1.0, 1.0, 1.0, 1.0, 1.001, 1.001],
            &[1.0001, 1.0001, 1.0001, 1.0001, 1.0011, 1.0011],
            &[0.9999, 0.9999, 0.9999, 0.9999, 1.0009, 1.0009],
            &[1, -1, 1, -1, 1, 0],
            &settings,
            4,
            100.0,
        );
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_cpu_parity_tests {
    //! Adversarial correctness gate for the GPU evaluator. The cubecl kernels
    //! (`crate::cubecl_eval`) were ported cubecl 0.9 → 0.10 and had NEVER
    //! compiled or run before, so this asserts the GPU population eval
    //! reproduces the CPU reference (the path the shipped binary runs) on a
    //! deterministic scenario. SMC gating is disabled (all-zero flags + zero
    //! gate) so signals are pure indicator-threshold crossings — CPU and GPU
    //! must agree, hence the metrics match within the declared numeric tolerance. Skips
    //! cleanly when no GPU device is present.
    use super::*;
    use ndarray::Array2;

    fn real_cuda_search_test_enabled(test: &str) -> bool {
        if std::env::var("NEOETHOS_RUN_CUDA_SEARCH_TESTS").as_deref() == Ok("1") {
            true
        } else {
            eprintln!("SKIPPED {test} — set NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 on a real GPU host");
            false
        }
    }

    #[test]
    fn gpu_population_eval_matches_cpu() {
        if !real_cuda_search_test_enabled("gpu_population_eval_matches_cpu") {
            return;
        }
        let n_samples = 800usize;
        let n_features = 6usize;
        let n_genes = 4usize;

        // Deterministic price wave large enough to trigger SL/TP exits.
        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + ((i as f64) * 0.02).sin() * 0.01)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0008).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0008).collect();

        // [features × samples], values well clear of the ±0.3 thresholds.
        let indicators = Array2::from_shape_fn((n_features, n_samples), |(f, i)| {
            (((i + f * 11) as f64) * 0.05).sin() * 0.8
        });

        // CSR genes: each sums 2 features (weight 1.0).
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 4];
        let gene_weights: Vec<f64> = vec![1.0; 8];
        let long_thr: Vec<f64> = vec![0.3; n_genes];
        let short_thr: Vec<f64> = vec![-0.3; n_genes];
        let sl_pips: Vec<f64> = vec![25.0; n_genes];
        let tp_pips: Vec<f64> = vec![50.0; n_genes];

        // SMC gating OFF: zero flags + zero gate → signals pass through ungated.
        let smc_data: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let gene_smc_flags: Vec<SmcRow> = vec![[0i8; 11]; n_genes];
        let smc_weights = [0.0f64; 11];
        let gate_threshold = 0.0f64;

        // 1-minute bars; fine month/day buckets so slot-7 monthly_target_hit_rate
        // is non-trivial (800 bars → 8 months, ~27 days; crosses real boundaries).
        let timestamps: Vec<i64> = (0..n_samples as i64).map(|i| i * 60_000).collect();
        let month_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 100).collect();
        let day_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 30).collect();

        // Finite cost model + explicit risk-sizing knobs (default settings are
        // all-NaN for pip_value/spread/commission → would zero pos_lots and pass
        // vacuously). Mirrors run_single_sl_trade's setup.
        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = true;
        settings.risk_per_trade_min = 0.005;
        settings.risk_per_trade_max = 0.03;
        settings.high_quality_confidence = 0.65;

        // CPU reference — the path the shipped binary actually runs.
        let cpu: Vec<[f64; 11]> = (0..n_genes)
            .map(|g| {
                let (signals, conf) = synthesize_signals_and_confidence_cpu(
                    indicators.view(),
                    &gene_offsets,
                    &gene_indices,
                    &gene_weights,
                    &long_thr,
                    &short_thr,
                    &smc_data,
                    &gene_smc_flags,
                    gate_threshold,
                    &smc_weights,
                    g,
                    n_samples,
                );
                let mut s = settings.clone();
                s.sl_pips = sl_pips[g];
                s.tp_pips = tp_pips[g];
                // Phase 2: real per-bar confidence feeds the CPU risk-based
                // sizing; the GPU kernel recomputes confidence on-device, so this
                // asserts both lanes agree on sizing AND slot-7.
                fast_evaluate_strategy_core(
                    &close,
                    &high,
                    &low,
                    &signals,
                    &conf,
                    &month_idx,
                    &day_idx,
                    &timestamps,
                    &s,
                )
            })
            .collect();

        // Direct real-device CubeCL path; the explicit gate above makes errors fatal.
        let gpu = match crate::cubecl_eval::try_evaluate_population_cuda(
            &close,
            &high,
            &low,
            indicators.view(),
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &month_idx,
            &day_idx,
            &timestamps,
            &sl_pips,
            &tp_pips,
            &[],
            &smc_data,
            &gene_smc_flags,
            gate_threshold,
            &smc_weights,
            &settings,
            None,
        ) {
            Ok(g) => g,
            Err(e) => panic!("GPU population real-device evaluation failed: {e}"),
        };

        assert_eq!(gpu.len(), n_genes, "gpu returned wrong gene count");

        // Metric layout (see try_evaluate_population_cuda): index 7 is
        // monthly_target_hit_rate; index 8 is the integer trade count.
        for g in 0..n_genes {
            let (ct, gt) = (cpu[g][8], gpu[g][8]);
            assert!(
                (ct - gt).abs() <= 1.0,
                "gene {g} trade-count mismatch: cpu={ct} gpu={gt} (GPU kernel logic bug)"
            );
            for m in [0usize, 1, 2, 3, 4, 5, 6, 7, 9, 10] {
                let (c, v) = (cpu[g][m], gpu[g][m]);
                // Both lanes are f64: tolerate accumulation rounding, catch
                // gross logic divergence.
                let tol = 1e-2 * c.abs().max(1.0) + 1e-3;
                assert!(
                    (c - v).abs() <= tol,
                    "gene {g} metric[{m}] mismatch: cpu={c} gpu={v} tol={tol}"
                );
            }
        }

        // This module is the direct CubeCL CUDA parity gate. Aggregate routing
        // to native Prototype B has its own real-device parity module and source
        // contract; calling the production dispatcher here would test that
        // separate engine and would also cross the fail-closed broker-financial-
        // truth boundary before reaching either GPU kernel.
    }

    /// PARITY GATE (2026-06-09) for the GPU FTMO prop-firm observables emitted by
    /// `backtest_population_kernel` into the new `ftmo_out` array. The GPU computes
    /// the 6 FTMO observables on-device; this asserts they match the CPU ground
    /// truth = `simulate_trades_core` → `compute_prop_firm_risk_summary`
    /// (validation.rs:746) within f64 accumulation tolerance. A subtle bug here would
    /// corrupt which strategies the prop-firm gate keeps, so this is the safety net.
    /// Runs only when the explicit real-device test gate is set; once selected,
    /// every device error fails loud.
    #[test]
    fn gpu_cpu_prop_firm_ftmo_matches() {
        if !real_cuda_search_test_enabled("gpu_cpu_prop_firm_ftmo_matches") {
            return;
        }
        use crate::validation::{
            PropFirmRiskInput, PropFirmRiskRules, compute_prop_firm_risk_summary,
        };

        // ~2400 bars at 1h spacing → spans ~100 calendar days, several trades/day on
        // some days, dry stretches on others → exercises every FTMO observable
        // (multi-day buckets, positive + negative days, end-of-day DD, trade-day count).
        let n_samples = 2400usize;
        let n_features = 6usize;
        let n_genes = 6usize;

        // Deterministic price wave large enough to trigger SL/TP exits regularly.
        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + ((i as f64) * 0.05).sin() * 0.012)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0010).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0010).collect();

        // [features × samples], values well clear of the ±0.3 thresholds.
        let indicators = Array2::from_shape_fn((n_features, n_samples), |(f, i)| {
            (((i + f * 7) as f64) * 0.06).sin() * 0.85
        });

        // CSR genes: each sums 2 features (weight 1.0).
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8, 10, 12];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0];
        let gene_weights: Vec<f64> = vec![1.0; 12];
        let long_thr: Vec<f64> = vec![0.3; n_genes];
        let short_thr: Vec<f64> = vec![-0.3; n_genes];
        let sl_pips: Vec<f64> = vec![20.0; n_genes];
        let tp_pips: Vec<f64> = vec![40.0; n_genes];

        // SMC gating OFF: zero flags + zero gate → signals pass through ungated.
        let smc_data: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let gene_smc_flags: Vec<SmcRow> = vec![[0i8; 11]; n_genes];
        let smc_weights = [0.0f64; 11];
        let gate_threshold = 0.0f64;

        // 1-hour bars; day_idx = timestamp/86_400_000 (the SAME key both the CPU
        // bucketing in compute_prop_firm_risk_summary and the GPU kernel use).
        let timestamps: Vec<i64> = (0..n_samples as i64).map(|i| i * 3_600_000).collect();
        let month_idx: Vec<i64> = timestamps.iter().map(|ts| ts / (30 * 86_400_000)).collect();
        let day_idx: Vec<i64> = timestamps.iter().map(|ts| ts / 86_400_000).collect();

        // Finite cost model + explicit risk-sizing knobs (default settings are
        // all-NaN for pip_value/spread/commission).
        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        // The PRODUCTION prop-firm gate (discovery.rs compute_prop_firm_pass_rate)
        // calls simulate_trades_core WITHOUT a confidences slice, so risk-based
        // sizing is inert there (use_risk_sizing = risk_based_sizing && !conf.is_empty()
        // == false) → fixed-1-lot. To match that exact behavior, the GPU FTMO path
        // is exercised here with risk_based_sizing = false (pos_lots forced to 1.0
        // in the kernel), so CPU and GPU run the identical fixed-1-lot backtest.
        settings.risk_based_sizing = false;
        settings.risk_per_trade_min = 0.005;
        settings.risk_per_trade_max = 0.03;
        settings.high_quality_confidence = 0.65;

        let initial_balance = settings.initial_equity();

        // CPU ground truth: per gene, simulate trades then aggregate FTMO observables
        // via the SAME function the production prop-firm gate uses.
        let cpu: Vec<[f64; 6]> = (0..n_genes)
            .map(|g| {
                let (signals, _conf) = synthesize_signals_and_confidence_cpu(
                    indicators.view(),
                    &gene_offsets,
                    &gene_indices,
                    &gene_weights,
                    &long_thr,
                    &short_thr,
                    &smc_data,
                    &gene_smc_flags,
                    gate_threshold,
                    &smc_weights,
                    g,
                    n_samples,
                );
                let mut s = settings.clone();
                s.sl_pips = sl_pips[g];
                s.tp_pips = tp_pips[g];
                let trades = simulate_trades_core(&close, &high, &low, &timestamps, &signals, &s);
                let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
                    trades: &trades,
                    initial_balance,
                    rules: PropFirmRiskRules::default(),
                });
                [
                    summary.net_return_pct,
                    summary.max_daily_loss_pct_observed,
                    summary.max_overall_drawdown_pct_observed,
                    summary.largest_profit_share_observed,
                    summary.max_trades_per_day_observed as f64,
                    summary.trading_days_observed as f64,
                ]
            })
            .collect();

        // Direct real-device CubeCL path; the explicit gate above makes errors fatal.
        let gpu = match crate::cubecl_eval::try_evaluate_ftmo_population_cuda(
            &close,
            &high,
            &low,
            indicators.view(),
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &month_idx,
            &day_idx,
            &timestamps,
            &sl_pips,
            &tp_pips,
            &[],
            &smc_data,
            &gene_smc_flags,
            gate_threshold,
            &smc_weights,
            &settings,
            None,
        ) {
            Ok(g) => g,
            Err(e) => panic!("GPU FTMO real-device evaluation failed: {e}"),
        };

        assert_eq!(gpu.len(), n_genes, "gpu returned wrong gene count");

        for g in 0..n_genes {
            let c = cpu[g];
            let v = gpu[g];
            // [0] net_return_pct, [1] max_daily_loss_pct, [2] max_overall_drawdown_pct,
            // [3] largest_profit_share — f64 accumulation: abs tol ~1e-3.
            for m in [0usize, 1, 2, 3] {
                let (cm, vm) = (c[m], v[m] as f64);
                let tol = 1e-3 * cm.abs().max(1.0) + 1e-3;
                assert!(
                    (cm - vm).abs() <= tol,
                    "gene {g} FTMO[{m}] mismatch: cpu={cm} gpu={vm} tol={tol}"
                );
            }
            // [4] max_trades_per_day, [5] trading_days — integer-valued, exact.
            for m in [4usize, 5] {
                let (cm, vm) = (c[m], v[m] as f64);
                assert!(
                    (cm - vm).abs() <= 0.5,
                    "gene {g} FTMO[{m}] (integer) mismatch: cpu={cm} gpu={vm}"
                );
            }
        }
    }

    /// GPU↔CPU parity for ADAPTIVE per-entry stops. Same synthetic combo as the
    /// fixed test, but each gene carries `stop_vol_mult > 0` and the settings a
    /// per-bar base vol series, so BOTH lanes scale the stop by volatility at each
    /// entry (`sl = mult × base[signal_bar]`) while retaining each gene's own
    /// `tp/sl` ratio. Proves the ported
    /// kernel's per-entry capture matches the CPU `entry_sl_tp_pips` — the gate
    /// before the adaptive→CPU guard is lifted. Skips cleanly with no GPU.
    #[test]
    fn gpu_population_eval_matches_cpu_adaptive_stops() {
        if !real_cuda_search_test_enabled("gpu_population_eval_matches_cpu_adaptive_stops") {
            return;
        }
        let n_samples = 800usize;
        let n_features = 6usize;
        let n_genes = 4usize;

        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + ((i as f64) * 0.02).sin() * 0.01)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0008).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0008).collect();
        let indicators = Array2::from_shape_fn((n_features, n_samples), |(f, i)| {
            (((i + f * 11) as f64) * 0.05).sin() * 0.8
        });
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 4];
        let gene_weights: Vec<f64> = vec![1.0; 8];
        let long_thr: Vec<f64> = vec![0.3; n_genes];
        let short_thr: Vec<f64> = vec![-0.3; n_genes];
        // Absolute pips are replaced by volatility scaling, but their ratios
        // remain active and deliberately differ across the population.
        let sl_pips: Vec<f64> = vec![25.0, 20.0, 40.0, 10.0];
        let tp_pips: Vec<f64> = vec![37.5, 40.0, 100.0, 30.0];
        // Per-gene adaptive multipliers (all > 0 ⇒ every gene runs adaptive).
        let stop_vol_mult: Vec<f64> = vec![1.2, 2.0, 0.8, 1.5];
        let smc_data: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let gene_smc_flags: Vec<SmcRow> = vec![[0i8; 11]; n_genes];
        let smc_weights = [0.0f64; 11];
        let gate_threshold = 0.0f64;
        let timestamps: Vec<i64> = (0..n_samples as i64).map(|i| i * 60_000).collect();
        let month_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 100).collect();
        let day_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 30).collect();

        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = true;
        settings.risk_per_trade_min = 0.005;
        settings.risk_per_trade_max = 0.03;
        settings.high_quality_confidence = 0.65;
        // Shared per-bar base vol series (the exact production builder).
        let base =
            crate::stop_target::adaptive_base_pips_series(&high, &low, &close, settings.pip_value)
                .expect("base vol series builds on 800 bars");
        assert_eq!(
            base.len(),
            n_samples,
            "base series must align with n_samples"
        );
        settings.adaptive_base_pips = Some(base.into());
        // Deliberately unlike every gene ratio: this is legacy fallback only.
        settings.adaptive_rr = 9.0;

        // CPU reference — each gene runs adaptive via its own multiplier.
        let cpu: Vec<[f64; 11]> = (0..n_genes)
            .map(|g| {
                let (signals, conf) = synthesize_signals_and_confidence_cpu(
                    indicators.view(),
                    &gene_offsets,
                    &gene_indices,
                    &gene_weights,
                    &long_thr,
                    &short_thr,
                    &smc_data,
                    &gene_smc_flags,
                    gate_threshold,
                    &smc_weights,
                    g,
                    n_samples,
                );
                let mut s = settings.clone();
                s.sl_pips = sl_pips[g];
                s.tp_pips = tp_pips[g];
                s.adaptive_vol_mult = stop_vol_mult[g];
                fast_evaluate_strategy_core(
                    &close,
                    &high,
                    &low,
                    &signals,
                    &conf,
                    &month_idx,
                    &day_idx,
                    &timestamps,
                    &s,
                )
            })
            .collect();

        // GPU path — the ported kernel, fed the REAL multiplier + base series
        // (calls the device path directly, bypassing the adaptive→CPU guard).
        let gpu = match crate::cubecl_eval::try_evaluate_population_cuda(
            &close,
            &high,
            &low,
            indicators.view(),
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &month_idx,
            &day_idx,
            &timestamps,
            &sl_pips,
            &tp_pips,
            &stop_vol_mult,
            &smc_data,
            &gene_smc_flags,
            gate_threshold,
            &smc_weights,
            &settings,
            None,
        ) {
            Ok(g) => g,
            Err(e) => panic!("adaptive GPU real-device evaluation failed: {e}"),
        };
        assert_eq!(gpu.len(), n_genes, "gpu returned wrong gene count");
        // At least one gene must trade, else the parity assertion is vacuous.
        assert!(
            cpu.iter().any(|m| m[8] > 0.0),
            "expected the adaptive combo to open trades"
        );
        for g in 0..n_genes {
            let (ct, gt) = (cpu[g][8], gpu[g][8]);
            assert!(
                (ct - gt).abs() <= 1.0,
                "adaptive gene {g} trade-count mismatch: cpu={ct} gpu={gt} (kernel adaptive bug)"
            );
            for m in [0usize, 1, 2, 3, 4, 5, 6, 7, 9, 10] {
                let (c, v) = (cpu[g][m], gpu[g][m]);
                let tol = 1e-2 * c.abs().max(1.0) + 1e-3;
                assert!(
                    (c - v).abs() <= tol,
                    "adaptive gene {g} metric[{m}] mismatch: cpu={c} gpu={v} tol={tol}"
                );
            }
        }
    }

    /// AREA 1 (2026-06-09) regression lock for the raised per-buffer cap. The
    /// 800-row case above fits one window/one batch, so it never exercised the
    /// windowing/batching machinery the cap-raise relies on as its safety net.
    /// This drives ~2M samples with a DELIBERATELY tiny per-buffer cap through
    /// the test-only cap seam so signal synthesis splits into MANY windows AND the backtest
    /// splits into MANY gene-batches — and asserts the concatenated GPU result
    /// still matches the whole-series CPU reference within the SAME tolerance. This
    /// is the lock proving the cap change did not break windowing. The explicit
    /// real-device gate skips card-less runs; once selected, errors fail loud.
    ///
    /// Note: the env-var read by `gpu_buffer_elem_cap` is process-global, but
    /// windowing/batching are EXACT by construction (each window splits on
    /// independent samples, each batch on independent genes, both concatenated in
    /// order), so even if this races the sibling test's window sizing the result
    /// is still correct — only the split COUNT differs, never the math.
    #[test]
    fn gpu_population_eval_matches_cpu_heavy_rows() {
        if !real_cuda_search_test_enabled("gpu_population_eval_matches_cpu_heavy_rows") {
            return;
        }
        // 600k bars × 8 genes: large enough that an 8MB buffer cap forces SEVERAL
        // sample-windows in the signal synth (600k×8B = 4.8MB/gene > the 8MB/4=2M-elem
        // window) AND several gene-batches in the backtest (8MB/4/600k ≈ 3 genes per
        // batch → 8 genes = 3 batches). 600k keeps each test allocation bounded;
        // windowing exists precisely to avoid one oversized device allocation.
        let n_samples = 600_000usize;
        let n_features = 6usize;
        let n_genes = 8usize;

        // Price: a flat close with a WIDE per-bar high/low band (±300 pips, far
        // beyond SL=25 / TP=50) so EVERY trade resolves on its first bar by a huge
        // margin. This is deliberate: with a grazing price path, tiny arithmetic
        // f64-on-CPU sub-pip rounding flips TP-hit-vs-SL-hit for thousands of
        // borderline trades over 2M bars → divergent win-rate/profit-factor that has
        // nothing to do with windowing. Overshooting both levels by 250+ pips makes
        // the per-trade OUTCOME precision-agnostic, leaving windowing/batching as the
        // only GPU-vs-CPU variable.
        let close: Vec<f64> = vec![1.10; n_samples];
        let high: Vec<f64> = close.iter().map(|c| c + 0.0300).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0300).collect();

        // SLOW SQUARE wave (period ~5000 bars), each feature decisively ±0.8 — NOT a
        // smooth sine. This is deliberate: the 800-bar test keeps values "well clear
        // of the ±0.3 thresholds" so CPU and GPU agree on the SIGN at every
        // bar; a sine grazes the threshold and, over 2M bars, tiny differences flip
        // thousands of boundary signals → wildly different trade SETS (not just a
        // count drift), which would mask the thing we're actually testing. With a
        // square wave each gene's 2-feature sum is ±1.6 (5× the threshold) except on
        // the rare single-bar transitions, so the SIGNAL is identical on both lanes
        // and the ONLY variable left is the windowing/batching the cap-raise relies
        // on. Per-feature phase offset gives the genes distinct (but still clear)
        // signals.
        let period = 5_000i64;
        let indicators = Array2::from_shape_fn((n_features, n_samples), |(f, i)| {
            let phase = (i as i64 + (f as i64) * 911) % period;
            if phase < period / 2 { 0.8f64 } else { -0.8f64 }
        });

        // 8 CSR genes, each summing 2 distinct features (unit weight). Feature pairs
        // are rotated so genes get different (but still threshold-clear) signals.
        let mut gene_offsets: Vec<i32> = vec![0];
        let mut gene_indices: Vec<i32> = Vec::new();
        for g in 0..n_genes {
            let a = (g % n_features) as i32;
            let b = ((g + 2) % n_features) as i32;
            gene_indices.push(a);
            gene_indices.push(b);
            gene_offsets.push(gene_indices.len() as i32);
        }
        let gene_weights: Vec<f64> = vec![1.0; gene_indices.len()];
        let long_thr: Vec<f64> = vec![0.3; n_genes];
        let short_thr: Vec<f64> = vec![-0.3; n_genes];
        let sl_pips: Vec<f64> = vec![25.0; n_genes];
        let tp_pips: Vec<f64> = vec![50.0; n_genes];

        let smc_data: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let gene_smc_flags: Vec<SmcRow> = vec![[0i8; 11]; n_genes];
        let smc_weights = [0.0f64; 11];
        let gate_threshold = 0.0f64;

        // 1-minute bars; ~7 months / ~417 days so the monthly/slot-7 buckets are
        // exercised across many month boundaries.
        let timestamps: Vec<i64> = (0..n_samples as i64).map(|i| i * 60_000).collect();
        let month_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 86_400).collect();
        let day_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 1_440).collect();

        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = true;
        settings.risk_per_trade_min = 0.005;
        settings.risk_per_trade_max = 0.03;
        settings.high_quality_confidence = 0.65;

        // CPU reference — whole series in one pass (no windowing).
        let cpu: Vec<[f64; 11]> = (0..n_genes)
            .map(|g| {
                let (signals, conf) = synthesize_signals_and_confidence_cpu(
                    indicators.view(),
                    &gene_offsets,
                    &gene_indices,
                    &gene_weights,
                    &long_thr,
                    &short_thr,
                    &smc_data,
                    &gene_smc_flags,
                    gate_threshold,
                    &smc_weights,
                    g,
                    n_samples,
                );
                let mut s = settings.clone();
                s.sl_pips = sl_pips[g];
                s.tp_pips = tp_pips[g];
                fast_evaluate_strategy_core(
                    &close,
                    &high,
                    &low,
                    &signals,
                    &conf,
                    &month_idx,
                    &day_idx,
                    &timestamps,
                    &s,
                )
            })
            .collect();

        // Run the GPU eval under an explicit per-buffer cap (MB).
        //
        // 2026-08-10: this drove the cap through
        // `NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB`, which production no longer
        // reads. The split-invariance assertion below is the point of the test
        // and had to survive, so the cap now goes through an explicitly
        // `#[cfg(test)]` seam — reachable from a test, not from a shell.
        let run_gpu_with_cap = |cap_mb: u64| -> anyhow::Result<Vec<[f64; 11]>> {
            use std::sync::atomic::Ordering;
            let prev = crate::cubecl_eval::TEST_GPU_BUFFER_CAP_MB.load(Ordering::Relaxed);
            crate::cubecl_eval::TEST_GPU_BUFFER_CAP_MB.store(cap_mb, Ordering::Relaxed);
            let out = crate::cubecl_eval::try_evaluate_population_cuda(
                &close,
                &high,
                &low,
                indicators.view(),
                &gene_offsets,
                &gene_indices,
                &gene_weights,
                &long_thr,
                &short_thr,
                &month_idx,
                &day_idx,
                &timestamps,
                &sl_pips,
                &tp_pips,
                &[],
                &smc_data,
                &gene_smc_flags,
                gate_threshold,
                &smc_weights,
                &settings,
                None,
            );
            crate::cubecl_eval::TEST_GPU_BUFFER_CAP_MB.store(prev, Ordering::Relaxed);
            out
        };

        // Two DIFFERENT small caps, both forcing MANY windows / gene-batches over
        // the 2M-row buffers but with DIFFERENT split granularities. Both stay tiny
        // so they never approach the device's memory limit; this is exactly what
        // windowing exists to guarantee.
        let gpu_a = run_gpu_with_cap(8)
            .unwrap_or_else(|e| panic!("heavy-row GPU real-device evaluation failed: {e}"));
        let gpu_b =
            run_gpu_with_cap(16).expect("second-granularity GPU eval after the first succeeded");

        assert_eq!(gpu_a.len(), n_genes, "gpu(cap=8) wrong gene count");
        assert_eq!(gpu_b.len(), n_genes, "gpu(cap=16) wrong gene count");

        // ── PRIMARY LOCK: windowing/batching is EXACT (split-invariant) ───────
        // Both runs are f64-on-GPU but split the 2M rows into DIFFERENT numbers of
        // windows + gene-batches. If the split is numerically faithful they must be
        // BIT-IDENTICAL. This isolates the cap-raise's windowing from all precision
        // noise — a gather/concatenation bug shows up here as an exact-equality
        // failure regardless of the CPU reference.
        let gpu_multi = &gpu_a;
        for g in 0..n_genes {
            for m in 0..11 {
                let (a, b) = (gpu_a[g][m], gpu_b[g][m]);
                assert!(
                    (a - b).abs() <= 1e-9 * a.abs().max(1.0),
                    "heavy gene {g} metric[{m}]: GPU cap=8 ({a}) != cap=16 ({b}) \
                     (windowing/batching is NOT split-invariant — the cap-raise corrupted results)"
                );
            }
        }

        // ── SECONDARY: GPU multi-window matches the CPU reference ─────────────
        // The square-wave signal is clear of the ±0.3 threshold and every trade
        // overshoots SL/TP by 250+ pips, so rounding cannot flip a signal or a
        // trade outcome → the GPU result matches CPU within the EXISTING tolerance.
        for g in 0..n_genes {
            let (ct, gt) = (cpu[g][8], gpu_multi[g][8]);
            assert!(
                (ct - gt).abs() <= 1.0,
                "heavy gene {g} trade-count mismatch: cpu={ct} gpu={gt}"
            );
            for m in [0usize, 1, 2, 3, 4, 5, 6, 7, 9, 10] {
                let (c, v) = (cpu[g][m], gpu_multi[g][m]);
                let tol = 1e-2 * c.abs().max(1.0) + 1e-3;
                assert!(
                    (c - v).abs() <= tol,
                    "heavy gene {g} metric[{m}] mismatch: cpu={c} gpu={v} tol={tol}"
                );
            }
        }
    }

    /// AREA 2 / Stage A (2026-06-09) — Monte-Carlo batched-population parity.
    ///
    /// The discovery quality screen used to run, per surviving candidate, a
    /// SERIAL loop of `mc_runs` perturbations: each perturbed gene → SMC-gated
    /// signal → `simulate_trades_core` (fixed-1-lot) → count `pnl_sum > 0`. Stage A
    /// replaces that with ONE batched `validation_backtest_population` launch over
    /// the `mc_runs` perturbed genes and counts `metrics[run][0] > 0.0`. The two
    /// must report the SAME profitable-run COUNT.
    ///
    /// This test pins BOTH halves of that equivalence:
    ///  1. RNG determinism — the perturbed genes are built with a `ChaCha8Rng`
    ///     seeded per `(combo, candidate, run)` (exactly as the discovery loop now
    ///     does), so the batched run is reproducible and CPU==GPU on the same seeds.
    ///  2. Pass-test equivalence — `metrics[0] > 0.0` (net_profit, fixed-1-lot)
    ///     equals `simulate_trades_core(...).iter().map(|t| t.pnl).sum() > 0.0`
    ///     because with `risk_based_sizing == false` net_profit IS the fixed-1-lot
    ///     trade-pnl sum.
    ///
    /// The only consumed signal is the SIGN of net_profit, so the COUNT is asserted
    /// EXACT-equal with a ±1 tolerance for the sole edge case (a run whose net sits
    /// within floating-point epsilon of zero); any near-zero sign flip is logged. The
    /// per-metric tolerance band of the sibling tests is NOT loosened. The explicit
    /// real-device gate skips card-less runs; once selected, the test calls CubeCL
    /// directly and permits no CPU substitution.
    #[test]
    fn gpu_montecarlo_batch_matches_cpu() {
        if !real_cuda_search_test_enabled("gpu_montecarlo_batch_matches_cpu") {
            return;
        }
        use rand::Rng;
        use rand::SeedableRng;

        let n_samples = 800usize;
        let n_features = 6usize;
        let mc_runs = 64usize;

        // Same deterministic price wave + indicators as the population parity test,
        // so SL/TP exits actually fire across the perturbed thresholds.
        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + ((i as f64) * 0.02).sin() * 0.01)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0008).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0008).collect();
        let indicators = Array2::from_shape_fn((n_features, n_samples), |(f, i)| {
            (((i + f * 11) as f64) * 0.05).sin() * 0.8
        });

        let timestamps: Vec<i64> = (0..n_samples as i64).map(|i| i * 60_000).collect();
        let month_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 100).collect();
        let day_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 30).collect();

        // SMC gating OFF (zero flags + zero gate) so signals are pure
        // indicator-threshold crossings — keeps the test's signal path identical
        // CPU↔GPU, isolating the MC batching as the thing under test.
        let smc_data: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let smc_weights = [0.0f64; 11];
        let gate_threshold = 0.0f64;

        // Fixed-1-lot cost model (mirrors `discovery_backtest_settings` after the
        // helper forces `risk_based_sizing = false`).
        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = false; // fixed-1-lot, matching the helper

        // A "base gene": sums features 0+1 (weight 1.0), modest thresholds, finite
        // SL/TP. Each MC run perturbs a clone of this, exactly like the discovery
        // loop perturbs `gene`.
        let base_long_thr = 0.30f64;
        let base_short_thr = -0.30f64;
        let base_weights = [1.0f64, 1.0f64];
        let base_indices = [0i32, 1i32];
        let base_sl = 25.0f64;
        let base_tp = 50.0f64;

        // Deterministic per-(combo, candidate, run) perturbation — IDENTICAL draw
        // order to `finalize_candidates_with_progress`'s MC loop
        // (long_threshold → short_threshold → each weight → sl? → tp?).
        let combo_seed: u64 = 0x1234_5678_9abc_def0;
        let candidate_idx: u64 = 7;

        // Per-run perturbed gene parameters (CSR-flat for the population call).
        let mut gene_offsets: Vec<i32> = Vec::with_capacity(mc_runs + 1);
        let mut gene_indices: Vec<i32> = Vec::with_capacity(mc_runs * 2);
        let mut gene_weights: Vec<f64> = Vec::with_capacity(mc_runs * 2);
        let mut long_thr: Vec<f64> = Vec::with_capacity(mc_runs);
        let mut short_thr: Vec<f64> = Vec::with_capacity(mc_runs);
        let mut sl_pips: Vec<f64> = Vec::with_capacity(mc_runs);
        let mut tp_pips: Vec<f64> = Vec::with_capacity(mc_runs);
        gene_offsets.push(0);
        for run_idx in 0..mc_runs as u64 {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
                combo_seed ^ ((candidate_idx) << 20) ^ run_idx,
            );
            let lt = base_long_thr * (1.0 + rng.random_range(-0.15f64..=0.15));
            let st = base_short_thr * (1.0 + rng.random_range(-0.15f64..=0.15));
            let w0 = base_weights[0] * (1.0 + rng.random_range(-0.20f64..=0.20));
            let w1 = base_weights[1] * (1.0 + rng.random_range(-0.20f64..=0.20));
            // sl/tp are finite>0, so the conditional draws ALWAYS happen (same as
            // the discovery loop, whose base gene has finite SL/TP).
            let sl = base_sl * (1.0 + rng.random_range(-0.25f64..=0.25));
            let tp = base_tp * (1.0 + rng.random_range(-0.25f64..=0.25));
            long_thr.push(lt);
            short_thr.push(st);
            gene_indices.extend_from_slice(&base_indices);
            gene_weights.push(w0);
            gene_weights.push(w1);
            gene_offsets.push(gene_indices.len() as i32);
            sl_pips.push(sl);
            tp_pips.push(tp);
        }
        let gene_smc_flags: Vec<SmcRow> = vec![[0i8; 11]; mc_runs];

        // ── SERIAL CPU REFERENCE — the exact old MC path ────────────────────────
        // Per run: synthesize the SMC-gated signal (the same synth the helper uses
        // internally), then `simulate_trades_core` (fixed-1-lot) and count pnl>0.
        let mut cpu_profitable = 0usize;
        let mut cpu_net: Vec<f64> = Vec::with_capacity(mc_runs);
        for run in 0..mc_runs {
            let (signals, _conf) = synthesize_signals_and_confidence_cpu(
                indicators.view(),
                &gene_offsets,
                &gene_indices,
                &gene_weights,
                &long_thr,
                &short_thr,
                &smc_data,
                &gene_smc_flags,
                gate_threshold,
                &smc_weights,
                run,
                n_samples,
            );
            let mut s = settings.clone();
            s.sl_pips = sl_pips[run];
            s.tp_pips = tp_pips[run];
            let trades = simulate_trades_core(&close, &high, &low, &timestamps, &signals, &s);
            let net: f64 = trades.iter().map(|t| t.pnl).sum();
            cpu_net.push(net);
            if net > 0.0 {
                cpu_profitable += 1;
            }
        }

        // ── BATCHED POPULATION — direct CubeCL, no CPU fallback ───────────────
        let metrics = crate::cubecl_eval::try_evaluate_population_cuda(
            &close,
            &high,
            &low,
            indicators.view(),
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &month_idx,
            &day_idx,
            &timestamps,
            &sl_pips,
            &tp_pips,
            &[],
            &smc_data,
            &gene_smc_flags,
            gate_threshold,
            &smc_weights,
            &settings,
            None,
        )
        .unwrap_or_else(|error| panic!("GPU Monte-Carlo real-device evaluation failed: {error}"));
        assert_eq!(
            metrics.len(),
            mc_runs,
            "batched MC returned the wrong run count"
        );
        let batch_profitable = metrics.iter().filter(|m| m[0] > 0.0).count();

        // Log + tolerate near-zero sign flips (the floating-point edge case the ±1
        // tolerance covers). A flip far from zero is a real divergence and fails.
        let mut near_zero_flips = 0usize;
        for run in 0..mc_runs {
            let (cpu_sign, gpu_sign) = (cpu_net[run] > 0.0, metrics[run][0] > 0.0);
            if cpu_sign != gpu_sign {
                let cn = cpu_net[run];
                let gn = metrics[run][0];
                // "Near zero" = within a few cents of break-even relative to the
                // run's magnitude; anything larger is a genuine logic divergence.
                let scale = cn.abs().max(gn.abs()).max(1.0);
                let near_zero = cn.abs() <= 1e-2 * scale && gn.abs() <= 1e-2 * scale;
                eprintln!(
                    "MC run {run}: sign flip cpu_net={cn} gpu_net={gn} (near_zero={near_zero})"
                );
                assert!(
                    near_zero,
                    "MC run {run}: profitable-sign flip FAR from zero (cpu_net={cn} gpu_net={gn}) \
                     — GPU batched MC diverges from the serial CPU reference"
                );
                near_zero_flips += 1;
            }
        }

        assert!(
            (batch_profitable as i64 - cpu_profitable as i64).abs() <= 1,
            "MC profitable-run COUNT mismatch: cpu={cpu_profitable} batched={batch_profitable} \
             (near-zero sign flips={near_zero_flips}) — only ±1 is tolerated (single break-even run)"
        );
    }

    /// AREA 2 / Stage B (2026-06-09) — THE CPCV gate parity test.
    ///
    /// A CPCV fold is a NON-CONTIGUOUS gathered index set. The serial CPU gate
    /// (`discovery::evaluate_cpcv_gate`) gathers the per-bar arrays HOST-SIDE into
    /// fresh contiguous Vecs and runs `fast_evaluate_strategy_core` on them; the
    /// GPU path (`validation_genes_population_gathered` →
    /// `validation_backtest_population`) feeds the SAME host-gathered contiguous
    /// buffer to the population kernel, which re-synthesizes signals/confidence
    /// pointwise from the gathered indicators + gathered FULL-SERIES SMC. This test
    /// is the GATE that proves the gather→contiguous-buffer→kernel path is bit-
    /// faithful: a gather-indexing bug (wrong column, off-by-one, SMC recomputed on
    /// the gathered slice rather than gathered from the full series) would silently
    /// corrupt the in-sample CPCV phi, promoting bad strategies to live.
    ///
    /// The fixture deliberately mirrors CPCV reality:
    ///  - a non-contiguous fold (every-3rd-bar stride PLUS a disjoint tail block,
    ///    like two CPCV test groups),
    ///  - REAL per-bar confidence with `risk_based_sizing == true` (CPCV inherits
    ///    `discovery_backtest_settings`, which keeps risk sizing on — unlike the MC
    ///    path's forced fixed-1-lot),
    ///  - ACTIVE SMC gating (non-zero flags + a non-trivial full-series SMC pattern)
    ///    so the gather of SMC rows is exercised, not bypassed.
    ///
    /// CPU reference: synthesize each gene's signals/confidence on the FULL series
    /// (with the full-series SMC), GATHER them at `absolute_idx`, then
    /// `fast_evaluate_strategy_core` on the gathered Vecs (timestamps = &[],
    /// exactly as the gate does). GPU: gather the indicators + full-series SMC at
    /// `absolute_idx` and run the population kernel. Assert per-gene metric parity
    /// within the EXISTING tolerance (`1e-2*|c|.max(1)+1e-3`, trade-count ±1) — NOT
    /// loosened. The explicit real-device gate skips card-less runs; once
    /// selected, the direct CubeCL call permits no CPU substitution.
    #[test]
    fn gpu_cpcv_gathered_fold_matches_cpu() {
        if !real_cuda_search_test_enabled("gpu_cpcv_gathered_fold_matches_cpu") {
            return;
        }
        let n_samples = 1_200usize;
        let n_features = 6usize;
        let n_genes = 6usize;

        // Deterministic price wave large enough to trigger SL/TP exits across the
        // gathered (non-contiguous) bars.
        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + ((i as f64) * 0.02).sin() * 0.01)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0008).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0008).collect();

        // [features × samples], values well clear of the ±0.3 thresholds so the
        // SIGN is precision-agnostic and the gather is the only variable.
        let indicators = Array2::from_shape_fn((n_features, n_samples), |(f, i)| {
            (((i + f * 11) as f64) * 0.05).sin() * 0.8
        });

        // CSR genes: each sums 2 features (weight 1.0).
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8, 10, 12];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0];
        let gene_weights: Vec<f64> = vec![1.0; 12];
        let long_thr: Vec<f64> = vec![0.3; n_genes];
        let short_thr: Vec<f64> = vec![-0.3; n_genes];
        let sl_pips: Vec<f64> = vec![25.0; n_genes];
        let tp_pips: Vec<f64> = vec![50.0; n_genes];

        // ACTIVE SMC gating. A non-trivial FULL-SERIES SMC pattern (alternating
        // ±1 with period-7 zeros) so the gathered SMC rows actually influence the
        // gate — this is the path most likely to corrupt under a gather bug.
        let full_smc: Vec<SmcRow> = (0..n_samples)
            .map(|i| {
                let dir = if (i / 5) % 2 == 0 { 1i8 } else { -1i8 };
                let mut row = [0i8; 11];
                if i % 7 != 0 {
                    // Populate a few channels (trend/bos/premium-ish) with the
                    // direction; leave others zero so the score is partial.
                    row[3] = dir; // mtf/trend channel
                    row[6] = dir; // bos
                    row[4] = dir; // premium
                }
                row
            })
            .collect();
        // Enable a subset of flags per gene (rotating) with a modest gate so SOME
        // bars pass and SOME are suppressed — exercising the gate both ways.
        let gene_smc_flags: Vec<SmcRow> = (0..n_genes)
            .map(|g| {
                let mut row = [0i8; 11];
                row[3] = 1;
                if g % 2 == 0 {
                    row[6] = 1;
                }
                if g % 3 == 0 {
                    row[4] = 1;
                }
                row
            })
            .collect();
        let smc_weights = [0.0f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let gate_threshold = 1.0f64;

        // Full-series month/day buckets (gathered per fold below).
        let full_month_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 150).collect();
        let full_day_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 30).collect();

        // Finite cost model + REAL risk-based sizing (CPCV keeps it on).
        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = true; // CPCV uses real confidence
        settings.risk_per_trade_min = 0.005;
        settings.risk_per_trade_max = 0.03;
        settings.high_quality_confidence = 0.65;

        // NON-CONTIGUOUS gathered fold: every-3rd bar over the first ~60% PLUS a
        // disjoint tail block — exactly the shape a CPCV split (two disjoint test
        // groups) produces.
        let mut absolute_idx: Vec<usize> = (0..(n_samples * 6 / 10)).step_by(3).collect();
        absolute_idx.extend((n_samples * 8 / 10)..(n_samples * 9 / 10));
        let fold_n = absolute_idx.len();
        assert!(fold_n > 50, "fold must be non-trivial");

        // Gather the per-bar arrays HOST-SIDE (exactly as evaluate_cpcv_gate does).
        let g_close: Vec<f64> = absolute_idx.iter().map(|&i| close[i]).collect();
        let g_high: Vec<f64> = absolute_idx.iter().map(|&i| high[i]).collect();
        let g_low: Vec<f64> = absolute_idx.iter().map(|&i| low[i]).collect();
        let g_month: Vec<i64> = absolute_idx.iter().map(|&i| full_month_idx[i]).collect();
        let g_day: Vec<i64> = absolute_idx.iter().map(|&i| full_day_idx[i]).collect();

        // ── CPU REFERENCE — the exact serial CPCV path ──────────────────────────
        // Synthesize each gene's signals/confidence on the FULL series with the
        // FULL-SERIES SMC, GATHER them at absolute_idx, then backtest the gathered
        // Vecs (timestamps = &[], risk-based sizing on).
        let cpu: Vec<[f64; 11]> = (0..n_genes)
            .map(|g| {
                let (full_signals, full_conf) = synthesize_signals_and_confidence_cpu(
                    indicators.view(),
                    &gene_offsets,
                    &gene_indices,
                    &gene_weights,
                    &long_thr,
                    &short_thr,
                    &full_smc,
                    &gene_smc_flags,
                    gate_threshold,
                    &smc_weights,
                    g,
                    n_samples,
                );
                let g_sig: Vec<i8> = absolute_idx.iter().map(|&i| full_signals[i]).collect();
                let g_conf: Vec<f64> = absolute_idx.iter().map(|&i| full_conf[i]).collect();
                let mut s = settings.clone();
                s.sl_pips = sl_pips[g];
                s.tp_pips = tp_pips[g];
                fast_evaluate_strategy_core(
                    &g_close,
                    &g_high,
                    &g_low,
                    &g_sig,
                    &g_conf,
                    &g_month,
                    &g_day,
                    &[],
                    &s,
                )
            })
            .collect();

        // ── GPU PATH — gather indicators + full-series SMC at absolute_idx, then
        // run the population kernel directly with no CPU substitution ───────────
        // Build the gathered indicators matrix [features × fold_n].
        let mut g_ind = Array2::<f64>::zeros((n_features, fold_n));
        for f in 0..n_features {
            for (k, &abs) in absolute_idx.iter().enumerate() {
                g_ind[(f, k)] = indicators[(f, abs)];
            }
        }
        let g_smc: Vec<SmcRow> = absolute_idx.iter().map(|&i| full_smc[i]).collect();

        // Direct CubeCL execution: this test has already passed the
        // NEOETHOS_RUN_CUDA_SEARCH_TESTS gate, so any error is fatal and there
        // is no CPU/native-engine substitution.
        let gpu = crate::cubecl_eval::try_evaluate_population_cuda(
            &g_close,
            &g_high,
            &g_low,
            g_ind.view(),
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &g_month,
            &g_day,
            &[],
            &sl_pips,
            &tp_pips,
            &[],
            &g_smc,
            &gene_smc_flags,
            gate_threshold,
            &smc_weights,
            &settings,
            None,
        )
        .unwrap_or_else(|e| panic!("GPU CPCV real-device evaluation failed: {e:#}"));
        assert_eq!(gpu.len(), n_genes, "gpu CPCV returned wrong gene count");

        for g in 0..n_genes {
            let (ct, gt) = (cpu[g][8], gpu[g][8]);
            assert!(
                (ct - gt).abs() <= 1.0,
                "CPCV gene {g} trade-count mismatch: cpu={ct} gpu={gt} \
                 (gather indexing bug?)"
            );
            for m in [0usize, 1, 2, 3, 4, 5, 6, 7, 9, 10] {
                let (c, v) = (cpu[g][m], gpu[g][m]);
                let tol = 1e-2 * c.abs().max(1.0) + 1e-3;
                assert!(
                    (c - v).abs() <= tol,
                    "CPCV gene {g} metric[{m}] mismatch: cpu={c} gpu={v} tol={tol} \
                     (gathered host-buffer not bit-faithful to the kernel)"
                );
            }
        }
    }

    /// AREA 2 / Stage C (2026-06-09) — GPU-routed **walk-forward** split parity.
    ///
    /// The walk-forward population path
    /// ([`crate::validation::embargoed_walkforward_population`]) replaces the
    /// per-gene `embargoed_walkforward_backtest` with, per CONTIGUOUS split window
    /// `[test_start..end]`, ONE population launch over all survivor genes
    /// (`validation_genes_population_window` → `validation_backtest_population`),
    /// keeping the risk diagnostics on the CPU. This test is the easiest parity
    /// case: the test slice is contiguous (no gather), and the launch is fixed-1-lot
    /// (`risk_based_sizing == false`, matching the single-gene WF's `&[]` confidence
    /// at validation.rs:1129-1130).
    ///
    /// It asserts BOTH halves of the WF parity claim:
    ///  1. the kernel's RE-SYNTHESIZED signal on `indicators[test_start..end]`
    ///     equals the PRECOMPUTED full-series signal sliced `[test_start..end]`
    ///     (the pointwise-synth-on-a-contiguous-slice insight), and
    ///  2. the population metrics[0/3/4/8] match the per-gene CPU
    ///     `fast_evaluate_strategy_core` reference (fixed-1-lot, `&[]` confidence)
    ///     within the EXISTING tolerance (`1e-2*|c|.max(1)+1e-3`, trade-count ±1) —
    ///     NOT loosened.
    ///
    /// ACTIVE SMC gating is used so the SMC slice actually influences the gate.
    /// The modest fixture (1 200 rows × 6 genes) keeps device allocations small.
    /// The explicit real-device gate skips card-less runs; once selected, the
    /// direct CubeCL call permits no CPU substitution.
    #[test]
    fn gpu_walkforward_split_matches_cpu() {
        if !real_cuda_search_test_enabled("gpu_walkforward_split_matches_cpu") {
            return;
        }
        let n_samples = 1_200usize;
        let n_features = 6usize;
        let n_genes = 6usize;

        // Deterministic price wave large enough to trigger SL/TP exits.
        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + ((i as f64) * 0.02).sin() * 0.01)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0008).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0008).collect();

        // [features × samples], values well clear of the ±0.3 thresholds.
        let indicators = Array2::from_shape_fn((n_features, n_samples), |(f, i)| {
            (((i + f * 11) as f64) * 0.05).sin() * 0.8
        });

        // CSR genes: each sums 2 features (weight 1.0).
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8, 10, 12];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0];
        let gene_weights: Vec<f64> = vec![1.0; 12];
        let long_thr: Vec<f64> = vec![0.3; n_genes];
        let short_thr: Vec<f64> = vec![-0.3; n_genes];
        let sl_pips: Vec<f64> = vec![25.0; n_genes];
        let tp_pips: Vec<f64> = vec![50.0; n_genes];

        // ACTIVE SMC gating with a non-trivial full-series SMC pattern, so the
        // contiguous SMC slice influences the gate (not bypassed).
        let full_smc: Vec<SmcRow> = (0..n_samples)
            .map(|i| {
                let dir = if (i / 5) % 2 == 0 { 1i8 } else { -1i8 };
                let mut row = [0i8; 11];
                if i % 7 != 0 {
                    row[3] = dir; // mtf/trend channel
                    row[6] = dir; // bos
                    row[4] = dir; // premium
                }
                row
            })
            .collect();
        let gene_smc_flags: Vec<SmcRow> = (0..n_genes)
            .map(|g| {
                let mut row = [0i8; 11];
                row[3] = 1;
                if g % 2 == 0 {
                    row[6] = 1;
                }
                if g % 3 == 0 {
                    row[4] = 1;
                }
                row
            })
            .collect();
        let smc_weights = [0.0f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let gate_threshold = 1.0f64;

        // Full-series calendar buckets (sliced contiguously per window).
        let full_month_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 150).collect();
        let full_day_idx: Vec<i64> = (0..n_samples as i64).map(|i| i / 30).collect();
        // 1-minute bars for the timestamp slice (full-length ⇒ passed through).
        let full_timestamps: Vec<i64> = (0..n_samples as i64).map(|i| i * 60_000).collect();

        // Finite cost model. FIXED-1-LOT: risk_based_sizing OFF — the walk-forward's
        // legacy fixed-1-lot (the single-gene path passes `&[]` confidence so
        // pos_lots stays 1.0 regardless; the population pack FORCES this flag off).
        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = false; // fixed-1-lot, matching WF `&[]` confidence

        // Reproduce one walk-forward split's contiguous test window EXACTLY as
        // `embargoed_walkforward_backtest` computes it (train_ratio=0.70, the same
        // window/train/embargo arithmetic), for n_splits=4.
        let n_splits = 4usize;
        let train_ratio = 0.70f64;
        let embargo_bars = 5usize;
        let window = (n_samples / n_splits).max(1);
        // Pick split index 1 (a mid-series window) and derive its test slice.
        let split_i = 1usize;
        let start = split_i * window;
        let end = ((split_i + 1) * window).min(n_samples);
        let train_end = start + ((window as f64) * train_ratio) as usize;
        let test_start = train_end + embargo_bars;
        assert!(
            test_start < end && (train_end - start) >= 40 && (end - test_start) >= 40,
            "fixture must yield a qualifying split window"
        );
        let win_len = end - test_start;
        assert!(win_len > 80, "window must clear the 80-bar floor");

        // ── Precompute full-series signals per gene, then SLICE [test_start..end].
        // This is what the single-gene WF path feeds (it slices the precomputed
        // `signals`). The population path RE-SYNTHESIZES from the sliced indicators;
        // both must agree because synth is pointwise.
        let full_signals_per_gene: Vec<Vec<i8>> = (0..n_genes)
            .map(|g| {
                synthesize_signals_and_confidence_cpu(
                    indicators.view(),
                    &gene_offsets,
                    &gene_indices,
                    &gene_weights,
                    &long_thr,
                    &short_thr,
                    &full_smc,
                    &gene_smc_flags,
                    gate_threshold,
                    &smc_weights,
                    g,
                    n_samples,
                )
                .0
            })
            .collect();

        // ── CPU REFERENCE — the single-gene WF slice eval: precomputed signals
        // sliced contiguously + `&[]` confidence (fixed-1-lot) on the window. ──────
        let slice_close = &close[test_start..end];
        let slice_high = &high[test_start..end];
        let slice_low = &low[test_start..end];
        let slice_month = &full_month_idx[test_start..end];
        let slice_day = &full_day_idx[test_start..end];
        let slice_ts = &full_timestamps[test_start..end];
        let cpu: Vec<[f64; 11]> = (0..n_genes)
            .map(|g| {
                let slice_sig = &full_signals_per_gene[g][test_start..end];
                let mut s = settings.clone();
                s.sl_pips = sl_pips[g];
                s.tp_pips = tp_pips[g];
                fast_evaluate_strategy_core(
                    slice_close,
                    slice_high,
                    slice_low,
                    slice_sig,
                    // Phase 1 walk-forward: legacy fixed-1-lot `&[]` confidence.
                    &[],
                    slice_month,
                    slice_day,
                    slice_ts,
                    &s,
                )
            })
            .collect();

        // ── GPU PATH — contiguous slice of the indicators/SMC; the kernel
        // re-synthesizes signals on the slice and backtests on CUDA. ───────────
        let win_ind = indicators
            .slice(ndarray::s![.., test_start..end])
            .to_owned();
        let win_smc: Vec<SmcRow> = full_smc[test_start..end].to_vec();

        // Half (1): the re-synthesized window signals must equal the precomputed
        // full-series signals sliced — the core WF parity insight. (CPU synth, so
        // this is device-independent and always runs.)
        for g in 0..n_genes {
            let (resynth, _conf) = synthesize_signals_and_confidence_cpu(
                win_ind.view(),
                &gene_offsets,
                &gene_indices,
                &gene_weights,
                &long_thr,
                &short_thr,
                &win_smc,
                &gene_smc_flags,
                gate_threshold,
                &smc_weights,
                g,
                win_len,
            );
            let sliced = &full_signals_per_gene[g][test_start..end];
            assert_eq!(
                resynth.as_slice(),
                sliced,
                "gene {g}: re-synth on the contiguous window != precomputed-signal slice \
                 (walk-forward window-synth not bit-faithful)"
            );
        }

        // Direct CubeCL execution; no skip/fallback is permitted after the
        // NEOETHOS_RUN_CUDA_SEARCH_TESTS gate.
        let gpu = crate::cubecl_eval::try_evaluate_population_cuda(
            slice_close,
            slice_high,
            slice_low,
            win_ind.view(),
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            slice_month,
            slice_day,
            slice_ts,
            &sl_pips,
            &tp_pips,
            &[],
            &win_smc,
            &gene_smc_flags,
            gate_threshold,
            &smc_weights,
            &settings,
            None,
        )
        .unwrap_or_else(|e| panic!("GPU walk-forward real-device evaluation failed: {e:#}"));
        assert_eq!(
            gpu.len(),
            n_genes,
            "gpu walk-forward returned wrong gene count"
        );

        // The WF path consumes metric slots 0 (net_profit), 3 (max_dd), 4
        // (win_rate), 8 (trade_count); slot 10 (max_daily_dd) feeds the risk
        // diagnostics. Assert all of them (incl. the others, for safety) within
        // the EXISTING tolerance.
        for g in 0..n_genes {
            let (ct, gt) = (cpu[g][8], gpu[g][8]);
            assert!(
                (ct - gt).abs() <= 1.0,
                "WF gene {g} trade-count mismatch: cpu={ct} gpu={gt} (window-synth bug?)"
            );
            for m in [0usize, 1, 2, 3, 4, 5, 6, 7, 9, 10] {
                let (c, v) = (cpu[g][m], gpu[g][m]);
                let tol = 1e-2 * c.abs().max(1.0) + 1e-3;
                assert!(
                    (c - v).abs() <= tol,
                    "WF gene {g} metric[{m}] mismatch: cpu={c} gpu={v} tol={tol} \
                     (contiguous window not bit-faithful to the kernel)"
                );
            }
        }
    }
}

#[cfg(test)]
mod cubecl_trailing_parity_tests {
    //! Direct CUDA CubeCL trailing-stop parity gate.
    //!
    //! The fixture first proves that trailing and its `max(raw, locked)` ratchet
    //! materially change the result, then compares the CUDA CubeCL kernel with
    //! the CPU oracle. Native Prototype B has a separate real-device parity
    //! module below, so the two CUDA engines remain independently covered.
    use super::*;

    struct TrailingFixture {
        close: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        indicators: ndarray::Array2<f64>,
        gene_offsets: Vec<i32>,
        gene_indices: Vec<i32>,
        gene_weights: Vec<f64>,
        long_thr: Vec<f64>,
        short_thr: Vec<f64>,
        sl_pips: Vec<f64>,
        tp_pips: Vec<f64>,
        stop_vol_mult: Vec<f64>,
        month_idx: Vec<i64>,
        day_idx: Vec<i64>,
        timestamps: Vec<i64>,
        smc_data: Vec<SmcRow>,
        gene_smc_flags: Vec<SmcRow>,
        settings: BacktestSettings,
    }

    /// One workload, built once, so the sensitivity check below and the parity
    /// check are provably about the same trades.
    ///
    /// The price series carries sustained runs (a 90-pip slow component) because
    /// the production trail only reaches its interesting branch on a long one:
    /// with `be_trigger_r = 1.0` it arms at +1R = +20 pips, and
    /// `raw = high - 1.0 * 20 pips` only exceeds `locked = entry + 2 pips` once
    /// the trade is more than 22 pips in profit. A gentle series would leave
    /// `candidate == locked` on every bar — which is exactly the value the
    /// miscompiled else-branch produced, i.e. a fixture that cannot fail.
    fn trailing_fixture(trailing_enabled: bool) -> TrailingFixture {
        let n_samples = 1_200usize;
        let n_genes = 4usize;
        let close: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64;
                1.1000 + (t / 220.0).sin() * 0.0090 + (t / 37.0).sin() * 0.0016
            })
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0006).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0006).collect();
        let indicators = ndarray::Array2::from_shape_fn((4, n_samples), |(f, i)| {
            let t = i as f64;
            (t / (18.0 + 11.0 * f as f64)).sin()
        });

        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = false;
        // The production trailing triple, verbatim: `EvaluationConfig::for_symbol`
        // sets enabled/be_trigger_r/atr_multiplier, and neither settings builder
        // touches `trailing_min_lock_pips`, so it stays at the eval default 2.0.
        settings.trailing_enabled = trailing_enabled;
        settings.trailing_atr_multiplier = 1.0;
        settings.trailing_be_trigger_r = 1.0;
        settings.trailing_min_lock_pips = 2.0;

        TrailingFixture {
            close,
            high,
            low,
            indicators,
            gene_offsets: vec![0, 2, 4, 6, 8],
            gene_indices: vec![0, 1, 1, 2, 2, 3, 3, 0],
            gene_weights: vec![1.0; 8],
            long_thr: vec![0.25; n_genes],
            short_thr: vec![-0.25; n_genes],
            sl_pips: vec![20.0; n_genes],
            tp_pips: vec![60.0; n_genes],
            stop_vol_mult: vec![0.0; n_genes],
            month_idx: (0..n_samples).map(|i| (i / 200) as i64).collect(),
            day_idx: (0..n_samples).map(|i| (i / 24) as i64).collect(),
            timestamps: (0..n_samples)
                .map(|i| 1_600_000_000_000 + (i as i64) * 3_600_000)
                .collect(),
            smc_data: vec![[0i8; 11]; n_samples],
            gene_smc_flags: vec![[0i8; 11]; n_genes],
            settings,
        }
    }

    impl TrailingFixture {
        /// The other half of what production actually runs.
        ///
        /// Genes carry a `stop_vol_mult` and `resolve_adaptive_stops`
        /// (genetic/search_engine.rs:1361-1386) installs the shared per-bar base
        /// series whenever ANY gene's multiplier is positive — which is the
        /// default since adaptive volatility-scaled stops were turned on. So the
        /// live configuration is trailing AND adaptive, and neither existing
        /// fixture is it: `gpu_population_eval_matches_cpu_adaptive_stops`
        /// (eval.rs:3295) runs adaptive with the trail off, and the plain
        /// fixture above runs the trail with `stop_vol_mult` all zero.
        ///
        /// The combination is not merely the sum of the two. Under adaptive
        /// stops `sl_distance` is derived from the base series at the entry bar
        /// instead of the gene's scalar `sl_pips`, and the trail is built out of
        /// `sl_distance` three times over — the arming test
        /// (`mv >= be_trigger_r * sl_distance`), the ratchet
        /// (`hi - atr_multiplier * sl_distance`) and the stop it replaces. A
        /// disagreement about the adaptive distance therefore reaches the exit
        /// price through the trail, on a path no fixture walked.
        fn with_adaptive_stops(mut self) -> Self {
            let base = crate::stop_target::adaptive_base_pips_series(
                &self.high,
                &self.low,
                &self.close,
                self.settings.pip_value,
            )
            .expect("adaptive base vol series builds on 1 200 bars");
            assert_eq!(
                base.len(),
                self.close.len(),
                "base series must align with the price series"
            );
            self.settings.adaptive_base_pips = Some(base.into());
            // The production builder, not a hand-picked number.
            self.settings.adaptive_rr = crate::stop_target::adaptive_stops_rr();
            // All > 0, so every gene runs adaptive — same shape as the existing
            // adaptive fixture's per-gene spread.
            self.stop_vol_mult = vec![1.2, 2.0, 0.8, 1.5];
            self
        }

        fn inputs(&self) -> PopulationEvalInputs<'_> {
            PopulationEvalInputs {
                close: &self.close,
                high: &self.high,
                low: &self.low,
                indicators: self.indicators.view(),
                gene_offsets: &self.gene_offsets,
                gene_indices: &self.gene_indices,
                gene_weights: &self.gene_weights,
                long_thr: &self.long_thr,
                short_thr: &self.short_thr,
                month_idx: &self.month_idx,
                day_idx: &self.day_idx,
                timestamps: &self.timestamps,
                sl_pips: &self.sl_pips,
                tp_pips: &self.tp_pips,
                stop_vol_mult: &self.stop_vol_mult,
                smc_data: &self.smc_data,
                gene_smc_flags: &self.gene_smc_flags,
                gate_threshold: 0.0,
                weights: &[1.0f64; 11],
                settings: &self.settings,
            }
        }
    }

    /// Proves the fixture can fail — no GPU required.
    ///
    /// A parity fixture whose trail never engages passes whether or not the
    /// kernel implements one, which is precisely the hole the whole suite had.
    /// This asserts on the CPU alone that turning the production trail on
    /// changes the answer, so the parity test below cannot quietly decay into
    /// comparing two identical no-ops if the series or the thresholds are ever
    /// retuned.
    #[test]
    fn trailing_fixture_actually_changes_the_result() {
        let off = trailing_fixture(false);
        let on = trailing_fixture(true);
        let m_off = validation_backtest_population_cpu(off.inputs());
        let m_on = validation_backtest_population_cpu(on.inputs());
        assert_eq!(m_off.len(), m_on.len());

        let traded: f64 = m_off.iter().map(|m| m[8]).sum();
        assert!(
            traded > 0.0,
            "fixture took no trades at all — it tests nothing"
        );

        let differs = |a: &[[f64; 11]], b: &[[f64; 11]]| {
            a.iter()
                .zip(b.iter())
                .any(|(x, y)| (x[0] - y[0]).abs() > 1e-9 || (x[8] - y[8]).abs() > 0.5)
        };

        assert!(
            differs(&m_off, &m_on),
            "the production trailing triple (enabled / atr 1.0 / trigger 1.0 / \
             2.0 min-lock) left every gene's net profit and trade count \
             unchanged — this fixture cannot detect a missing trail"
        );

        // Stronger: prove the `max(raw, locked)` branch itself fires.
        //
        // A fixture where the trail arms but `raw` never exceeds `locked` would
        // pass the check above and still be blind to cubecl#1375, because the
        // miscompiled kernel returns `locked` — the same value the correct one
        // would. Reproduce that state on the CPU: an ATR multiplier large enough
        // that `raw = high - mult * sl_distance` can never rise above
        // `locked = entry + min_lock` pins `candidate` to `locked` on every
        // armed bar, which is exactly what the else-branch-always-wins bug
        // produced. If production (multiplier 1.0) is indistinguishable from
        // that, the parity test below cannot see the defect it exists for.
        let mut pinned = trailing_fixture(true);
        pinned.settings.trailing_atr_multiplier = 50.0;
        let m_pinned = validation_backtest_population_cpu(pinned.inputs());
        assert!(
            differs(&m_on, &m_pinned),
            "production settings produced the same result as a trail pinned to \
             the min-lock floor — `raw` never exceeds `locked` in this fixture, \
             so it cannot distinguish a correct ATR ratchet from cubecl#1375's \
             unconditional else branch"
        );

        // Same two proofs for the adaptive variant, because it is a different
        // trail: `sl_distance` comes from the base vol series, so the arming
        // threshold and the ratchet distance are per-entry values the fixed-stop
        // case never produces. A blind adaptive fixture would be the same defect
        // one layer down.
        let ad_off = trailing_fixture(false).with_adaptive_stops();
        let ad_on = trailing_fixture(true).with_adaptive_stops();
        let m_ad_off = validation_backtest_population_cpu(ad_off.inputs());
        let m_ad_on = validation_backtest_population_cpu(ad_on.inputs());
        let ad_traded: f64 = m_ad_off.iter().map(|m| m[8]).sum();
        assert!(
            ad_traded > 0.0,
            "adaptive variant took no trades at all — it tests nothing"
        );
        assert!(
            differs(&m_ad_off, &m_ad_on),
            "with adaptive stops the production trail changed nothing — the \
             adaptive parity case cannot detect a missing trail"
        );
        let mut ad_pinned = trailing_fixture(true).with_adaptive_stops();
        ad_pinned.settings.trailing_atr_multiplier = 50.0;
        let m_ad_pinned = validation_backtest_population_cpu(ad_pinned.inputs());
        assert!(
            differs(&m_ad_on, &m_ad_pinned),
            "adaptive + production trail is indistinguishable from a trail \
             pinned to the min-lock floor — the adaptive case cannot see \
             cubecl#1375's unconditional else branch"
        );
    }

    /// CubeCL kernel vs CPU with the trail ON.
    ///
    /// Named with a `gpu_` prefix because GPU CI filters on it. The test calls
    /// the exclusively-CubeCL entrypoint directly, so aggregate CUDA builds
    /// exercise this kernel even when native Prototype B is also compiled.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_cubecl_trailing_stop_matches_cpu() {
        // This is a real-device gate, not a generic feature-build unit test.
        // The explicit test-only switch prevents card-less `cargo test --features
        // gpu-*` jobs from claiming device evidence. Once selected, every device
        // error is fatal; Prototype B being available is irrelevant because this
        // test calls the exclusively-CubeCL entrypoint directly.
        if std::env::var("NEOETHOS_RUN_CUDA_SEARCH_TESTS").as_deref() != Ok("1") {
            eprintln!(
                "SKIPPED gpu_cubecl_trailing_stop_matches_cpu — set \
                 NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 on a real GPU host"
            );
            return;
        }

        // Both shapes production runs: the gene's scalar stop, and the adaptive
        // per-bar stop the trail is built out of.
        for (label, fx) in [
            ("fixed stops", trailing_fixture(true)),
            (
                "adaptive stops",
                trailing_fixture(true).with_adaptive_stops(),
            ),
        ] {
            let gpu = match crate::cubecl_eval::try_evaluate_population_cuda(
                &fx.close,
                &fx.high,
                &fx.low,
                fx.indicators.view(),
                &fx.gene_offsets,
                &fx.gene_indices,
                &fx.gene_weights,
                &fx.long_thr,
                &fx.short_thr,
                &fx.month_idx,
                &fx.day_idx,
                &fx.timestamps,
                &fx.sl_pips,
                &fx.tp_pips,
                &fx.stop_vol_mult,
                &fx.smc_data,
                &fx.gene_smc_flags,
                0.0,
                &[1.0f64; 11],
                &fx.settings,
                None,
            ) {
                Ok(rows) => rows,
                Err(e) => panic!("CubeCL trailing real-device evaluation failed ({label}): {e}"),
            };

            let cpu = validation_backtest_population_cpu(fx.inputs());
            assert_eq!(gpu.len(), cpu.len(), "gene count mismatch ({label})");

            // Compares the money, not just the exits. A trail closes at a level
            // that moved: reconstructing the exit from the original stop gives
            // the right bar and the wrong profit, and only net profit catches
            // that.
            for (gene, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
                // Trade count is the sharpest signal for a trail that does not
                // ratchet: a stop stuck at the min-lock floor survives bars the
                // moved stop would have closed on, so the counts separate long
                // before the money does.
                //
                // The fixture is deliberately small (1 200 smooth bars,
                // 20/60-pip barriers, zero spread) so no comparison sits near a
                // tie; any trade-count mismatch is an arithmetic defect.
                assert!(
                    (g[8] - c[8]).abs() <= 0.5,
                    "{label} gene {gene} trade count: gpu {} vs cpu {} — the kernel and \
                     the CPU disagree about when a trailing stop fires",
                    g[8],
                    c[8]
                );
                for slot in [0usize, 1, 2, 3, 4, 5, 6, 9, 10] {
                    let (a, b) = (g[slot], c[slot]);
                    if !a.is_finite() && !b.is_finite() {
                        continue;
                    }
                    let tol = 1e-2 * b.abs().max(1.0) + 1e-3;
                    assert!(
                        (a - b).abs() <= tol,
                        "{label} gene {gene} slot {slot}: gpu {a} vs cpu {b} (tol {tol}) \
                         — the CubeCL kernel and the CPU disagree about a trailing stop"
                    );
                }
            }
        }
    }
}

#[cfg(all(test, feature = "gpu-b-adapter"))]
mod trailing_parity_tests {
    use super::*;

    fn exact_test_parent(
        close: &[f64],
        high: &[f64],
        low: &[f64],
        indicators: ndarray::ArrayView2<'_, f64>,
        timestamps: &[i64],
    ) -> (
        crate::data_selection::CanonicalSearchArtifactScopeV2,
        neoethos_data::FeatureFrame,
        neoethos_data::Ohlcv,
    ) {
        let columns = (0..indicators.nrows())
            .map(|feature| {
                neoethos_data::FeatureColumnF64::new(
                    format!("native_parity_{feature}"),
                    indicators.row(feature).to_vec(),
                    vec![neoethos_data::FeatureCellValidity::Valid; indicators.ncols()],
                )
                .unwrap()
            })
            .collect();
        let features = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            timestamps.to_vec(),
            columns,
        )
        .unwrap();
        let ohlcv = neoethos_data::Ohlcv {
            timestamp: Some(timestamps.to_vec()),
            open: close.to_vec(),
            high: high.to_vec(),
            low: low.to_vec(),
            close: close.to_vec(),
            volume: None,
        };
        let anchor = features.provenance().bindings()[0]
            .dataset_identity()
            .clone();
        let receipt = crate::data_selection::CanonicalSearchInputReceiptV2::from_feature_frame(
            &anchor, &features,
        )
        .unwrap();
        let scope = crate::data_selection::CanonicalSearchArtifactScopeV2::for_entire_receipt(
            crate::data_selection::CanonicalSearchWindowRoleV1::DiscoveryInput,
            receipt,
        )
        .unwrap();
        (scope, features, ohlcv)
    }

    fn real_native_cuda_search_test_enabled(test: &str) -> bool {
        if std::env::var("NEOETHOS_RUN_CUDA_SEARCH_TESTS").as_deref() == Ok("1") {
            true
        } else {
            eprintln!("SKIPPED {test} — set NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 on a real CUDA host");
            false
        }
    }

    /// The case that had no test at all.
    ///
    /// Every parity fixture ran with `trailing_enabled: false`, so the kernel's
    /// complete absence of a trailing stop — while the CPU engine applies one by
    /// default — passed every check for as long as the suite existed. This is
    /// the same workload with it on.
    ///
    /// It compares the money, not just the exits: a trailing stop closes at a
    /// level that moved, and rebuilding the exit price from the original stop
    /// gives the right bar and the wrong profit, which only a P&L comparison
    /// catches.
    #[test]
    fn gpu_matches_cpu_with_a_trailing_stop() {
        if !real_native_cuda_search_test_enabled("gpu_matches_cpu_with_a_trailing_stop") {
            return;
        }
        let n_samples = 1_200usize;
        let n_genes = 4usize;
        // A series with sustained runs, so the trail actually arms and ratchets
        // rather than every trade dying on the initial stop.
        let close: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64;
                1.1000 + (t / 220.0).sin() * 0.0090 + (t / 37.0).sin() * 0.0016
            })
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0006).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0006).collect();
        let indicators = ndarray::Array2::from_shape_fn((4, n_samples), |(f, i)| {
            let t = i as f64;
            (t / (18.0 + 11.0 * f as f64)).sin()
        });
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let gene_weights: Vec<f64> = vec![1.0; 8];
        let long_thr: Vec<f64> = vec![0.25; n_genes];
        let short_thr: Vec<f64> = vec![-0.25; n_genes];
        let sl_pips: Vec<f64> = vec![20.0; n_genes];
        let tp_pips: Vec<f64> = vec![60.0; n_genes];
        let stop_vol_mult: Vec<f64> = vec![0.0; n_genes];
        let months: Vec<i64> = (0..n_samples).map(|i| (i / 200) as i64).collect();
        let days: Vec<i64> = (0..n_samples).map(|i| (i / 24) as i64).collect();
        let timestamps: Vec<i64> = (0..n_samples)
            .map(|i| 1_600_000_000_000 + (i as i64) * 3_600_000)
            .collect();
        let smc: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let gene_smc: Vec<SmcRow> = vec![[0i8; 11]; n_genes];

        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = false;
        // The operator's own configuration, which is what production runs.
        settings.trailing_enabled = true;
        settings.trailing_atr_multiplier = 0.4;
        settings.trailing_be_trigger_r = 0.1;
        settings.trailing_min_lock_pips = 2.0;

        let (scope, features, ohlcv) =
            exact_test_parent(&close, &high, &low, indicators.view(), &timestamps);
        let admission = crate::acquire_strict_discovery_device_admission_v1().unwrap();
        let run = crate::population_execution_evidence_v1::begin_exact_population_execution_run_v1(
            admission, &scope, &features, &ohlcv,
        )
        .unwrap();
        let evidence = run
            .seal_evaluation(
                &settings,
                crate::exact_resident_dataset_authority_v1::ExactResidentDatasetViewRequestV1::Full,
            )
            .unwrap();
        let gpu = crate::gpu_native::prototype_b_population_eval::try_evaluate_population_b(
            &evidence,
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &sl_pips,
            &tp_pips,
            &stop_vol_mult,
            &gene_smc,
            0.0,
            &[1.0f64; 11],
        )
        .unwrap_or_else(|err| panic!("Prototype B real-device parity failed: {err}"));

        let cpu = validation_backtest_population_cpu(PopulationEvalInputs {
            close: &close,
            high: &high,
            low: &low,
            indicators: indicators.view(),
            gene_offsets: &gene_offsets,
            gene_indices: &gene_indices,
            gene_weights: &gene_weights,
            long_thr: &long_thr,
            short_thr: &short_thr,
            month_idx: &months,
            day_idx: &days,
            timestamps: &timestamps,
            sl_pips: &sl_pips,
            tp_pips: &tp_pips,
            stop_vol_mult: &stop_vol_mult,
            smc_data: &smc,
            gene_smc_flags: &gene_smc,
            gate_threshold: 0.0,
            weights: &[1.0f64; 11],
            settings: &settings,
        });

        assert_eq!(gpu.len(), cpu.len());
        for (gene, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            for slot in 0..11 {
                let (a, b) = (g[slot], c[slot]);
                if !a.is_finite() && !b.is_finite() {
                    continue;
                }
                assert!(
                    (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "gene {gene} slot {slot}: gpu {a} vs cpu {b} — the kernel and the \
                     CPU disagree about a trailing stop"
                );
            }
        }
    }

    /// Splits the failure in two: plumbing, or hour boundaries.
    ///
    /// All three buckets hold the same non-zero value, so the hour lookup
    /// cannot matter — every bar resolves to 2.0 whichever branch it takes.
    /// If this passes and the varied-bucket test fails, the bug is in the hour
    /// arithmetic. If this fails too, the value is not reaching the kernel at
    /// all and the buckets are a red herring.
    #[test]
    fn uniform_buckets_are_a_scalar_by_another_name() {
        if !real_native_cuda_search_test_enabled("uniform_buckets_are_a_scalar_by_another_name") {
            return;
        }
        let n_samples = 1_200usize;
        let n_genes = 4usize;
        let close: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64;
                1.1000 + (t / 220.0).sin() * 0.0090 + (t / 37.0).sin() * 0.0016
            })
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0006).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0006).collect();
        let indicators = ndarray::Array2::from_shape_fn((4, n_samples), |(f, i)| {
            let t = i as f64;
            (t / (18.0 + 11.0 * f as f64)).sin()
        });
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let gene_weights: Vec<f64> = vec![1.0; 8];
        let long_thr: Vec<f64> = vec![0.25; n_genes];
        let short_thr: Vec<f64> = vec![-0.25; n_genes];
        let sl_pips: Vec<f64> = vec![20.0; n_genes];
        let tp_pips: Vec<f64> = vec![60.0; n_genes];
        let stop_vol_mult: Vec<f64> = vec![0.0; n_genes];
        let months: Vec<i64> = (0..n_samples).map(|i| (i / 200) as i64).collect();
        let days: Vec<i64> = (0..n_samples).map(|i| (i / 24) as i64).collect();
        let timestamps: Vec<i64> = (0..n_samples)
            .map(|i| 1_600_000_000_000 + (i as i64) * 3_600_000)
            .collect();
        let smc: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let gene_smc: Vec<SmcRow> = vec![[0i8; 11]; n_genes];

        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = false;
        settings.trailing_enabled = false;
        settings.session_spread_profile = Some(SessionSpreadProfile {
            asian_pips: 2.0,
            overlap_pips: 2.0,
            late_ny_pips: 2.0,
        });

        let (scope, features, ohlcv) =
            exact_test_parent(&close, &high, &low, indicators.view(), &timestamps);
        let admission = crate::acquire_strict_discovery_device_admission_v1().unwrap();
        let run = crate::population_execution_evidence_v1::begin_exact_population_execution_run_v1(
            admission, &scope, &features, &ohlcv,
        )
        .unwrap();
        let evidence = run
            .seal_evaluation(
                &settings,
                crate::exact_resident_dataset_authority_v1::ExactResidentDatasetViewRequestV1::Full,
            )
            .unwrap();
        let gpu = crate::gpu_native::prototype_b_population_eval::try_evaluate_population_b(
            &evidence,
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &sl_pips,
            &tp_pips,
            &stop_vol_mult,
            &gene_smc,
            0.0,
            &[1.0f64; 11],
        )
        .unwrap_or_else(|err| {
            panic!("Prototype B uniform-bucket real-device parity failed: {err}")
        });

        let cpu = validation_backtest_population_cpu(PopulationEvalInputs {
            close: &close,
            high: &high,
            low: &low,
            indicators: indicators.view(),
            gene_offsets: &gene_offsets,
            gene_indices: &gene_indices,
            gene_weights: &gene_weights,
            long_thr: &long_thr,
            short_thr: &short_thr,
            month_idx: &months,
            day_idx: &days,
            timestamps: &timestamps,
            sl_pips: &sl_pips,
            tp_pips: &tp_pips,
            stop_vol_mult: &stop_vol_mult,
            smc_data: &smc,
            gene_smc_flags: &gene_smc,
            gate_threshold: 0.0,
            weights: &[1.0f64; 11],
            settings: &settings,
        });

        for (gene, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            eprintln!(
                "gene {gene}: gpu net {:.4} trades {:.0} | cpu net {:.4} trades {:.0}",
                g[0], g[8], c[0], c[8]
            );
        }
        for (gene, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g[0] - c[0]).abs() <= 1e-6 * c[0].abs().max(1.0),
                "gene {gene}: uniform buckets still disagree — the value is not reaching                  the kernel, so the hour arithmetic is not the cause"
            );
        }
    }

    /// The case every fixture leaves off, exactly like the trailing stop did.
    ///
    /// `SessionSpreadProfile` has existed on the CPU since the type was
    /// written, and the device settings had no field to receive it — so a run
    /// with a profile would have priced trades one way on the CPU and another
    /// on the card, with nothing to notice. Every parity fixture sets
    /// `session_spread_profile: None`, which is precisely the value that hides
    /// it.
    ///
    /// The buckets here are deliberately far apart (0.6 / 3.5 / 1.4) and the
    /// series spans four full days at hourly bars, so every bucket is entered
    /// many times and a kernel that charged one flat number cannot match.
    #[test]
    fn gpu_matches_cpu_with_a_session_spread_profile() {
        if !real_native_cuda_search_test_enabled("gpu_matches_cpu_with_a_session_spread_profile") {
            return;
        }
        let n_samples = 1_200usize;
        let n_genes = 4usize;
        let close: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64;
                1.1000 + (t / 220.0).sin() * 0.0090 + (t / 37.0).sin() * 0.0016
            })
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0006).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0006).collect();
        let indicators = ndarray::Array2::from_shape_fn((4, n_samples), |(f, i)| {
            let t = i as f64;
            (t / (18.0 + 11.0 * f as f64)).sin()
        });
        let gene_offsets: Vec<i32> = vec![0, 2, 4, 6, 8];
        let gene_indices: Vec<i32> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let gene_weights: Vec<f64> = vec![1.0; 8];
        let long_thr: Vec<f64> = vec![0.25; n_genes];
        let short_thr: Vec<f64> = vec![-0.25; n_genes];
        let sl_pips: Vec<f64> = vec![20.0; n_genes];
        let tp_pips: Vec<f64> = vec![60.0; n_genes];
        let stop_vol_mult: Vec<f64> = vec![0.0; n_genes];
        let months: Vec<i64> = (0..n_samples).map(|i| (i / 200) as i64).collect();
        let days: Vec<i64> = (0..n_samples).map(|i| (i / 24) as i64).collect();
        // Hourly bars from a UTC midnight, so the hour-of-day walks all three
        // buckets repeatedly rather than sitting in one.
        let timestamps: Vec<i64> = (0..n_samples)
            .map(|i| 1_600_000_000_000 + (i as i64) * 3_600_000)
            .collect();
        let smc: Vec<SmcRow> = vec![[0i8; 11]; n_samples];
        let gene_smc: Vec<SmcRow> = vec![[0i8; 11]; n_genes];

        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        // Deliberately NOT the mean of the buckets: if the kernel ignored the
        // profile and charged this, the P&L would differ and the test fails.
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = false;
        settings.trailing_enabled = false;
        settings.session_spread_profile = Some(SessionSpreadProfile {
            asian_pips: 3.5,
            overlap_pips: 0.6,
            late_ny_pips: 1.4,
        });

        let (scope, features, ohlcv) =
            exact_test_parent(&close, &high, &low, indicators.view(), &timestamps);
        let admission = crate::acquire_strict_discovery_device_admission_v1().unwrap();
        let run = crate::population_execution_evidence_v1::begin_exact_population_execution_run_v1(
            admission, &scope, &features, &ohlcv,
        )
        .unwrap();
        let evidence = run
            .seal_evaluation(
                &settings,
                crate::exact_resident_dataset_authority_v1::ExactResidentDatasetViewRequestV1::Full,
            )
            .unwrap();
        let gpu = crate::gpu_native::prototype_b_population_eval::try_evaluate_population_b(
            &evidence,
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &sl_pips,
            &tp_pips,
            &stop_vol_mult,
            &gene_smc,
            0.0,
            &[1.0f64; 11],
        )
        .unwrap_or_else(|err| {
            panic!("Prototype B session-spread real-device parity failed: {err}")
        });

        let cpu = validation_backtest_population_cpu(PopulationEvalInputs {
            close: &close,
            high: &high,
            low: &low,
            indicators: indicators.view(),
            gene_offsets: &gene_offsets,
            gene_indices: &gene_indices,
            gene_weights: &gene_weights,
            long_thr: &long_thr,
            short_thr: &short_thr,
            month_idx: &months,
            day_idx: &days,
            timestamps: &timestamps,
            sl_pips: &sl_pips,
            tp_pips: &tp_pips,
            stop_vol_mult: &stop_vol_mult,
            smc_data: &smc,
            gene_smc_flags: &gene_smc,
            gate_threshold: 0.0,
            weights: &[1.0f64; 11],
            settings: &settings,
        });

        // Guard the guard: with a flat 0.0 scalar and these buckets, a kernel
        // that ignored the profile would report a different net profit. If the
        // fixture ever stops trading, this catches it before the parity loop
        // passes vacuously.
        assert!(
            cpu.iter().any(|row| row[8] > 0.0),
            "fixture produced no trades — the parity assertion below would be empty"
        );

        assert_eq!(gpu.len(), cpu.len());
        for (gene, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            for slot in 0..11 {
                let (a, b) = (g[slot], c[slot]);
                if !a.is_finite() && !b.is_finite() {
                    continue;
                }
                assert!(
                    (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "gene {gene} slot {slot}: gpu {a} vs cpu {b} — the kernel and the                      CPU disagree about the session spread"
                );
            }
        }
    }
}

#[cfg(test)]
mod gap_threshold_tests {
    use super::*;

    /// A weekend is normal to hold through; a hole in the data is not.
    ///
    /// The default was 0, which switches detection off entirely, so a position
    /// open before a missing stretch was carried across it and then tested
    /// against prices from the far side. December 2014 is absent from every
    /// series in the operator's store — twelve days of it on H1 — so this is
    /// not hypothetical.
    #[test]
    fn the_default_sits_between_a_weekend_and_a_hole() {
        let day = 24 * 60 * 60 * 1000i64;
        let threshold = BacktestSettings::default().gap_threshold_ms;
        assert!(threshold > 0, "detection must not be off by default");

        // An FX weekend is about two and a half days, Friday close to Sunday
        // open. Flagging those would close every position every week.
        assert!(
            threshold > 5 * day / 2,
            "a weekend would be treated as a gap: {threshold} ms"
        );

        // The December 2014 hole is twelve days on H1 and far more on D1.
        assert!(
            threshold < 12 * day,
            "the known hole would not be caught: {threshold} ms"
        );
    }
}
