#[cfg(feature = "gpu")]
use crate::cubecl_eval::{
    cuda_eval_backtest_kernel_enabled, cuda_eval_signal_kernel_enabled,
    try_evaluate_population_cuda, try_generate_signal_rows_cuda,
};
use crate::quality::Trade;
use ndarray::ArrayView2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::{Once, OnceLock};

pub type SmcRow = [i8; 11];

pub struct PopulationEvalInputs<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub indicators: ArrayView2<'a, f32>,
    pub gene_offsets: &'a [i32],
    pub gene_indices: &'a [i32],
    pub gene_weights: &'a [f32],
    pub long_thr: &'a [f32],
    pub short_thr: &'a [f32],
    pub month_idx: &'a [i64],
    pub day_idx: &'a [i64],
    pub timestamps: &'a [i64],
    pub sl_pips: &'a [f64],
    pub tp_pips: &'a [f64],
    pub smc_data: &'a [SmcRow],
    pub gene_smc_flags: &'a [SmcRow],
    pub gate_threshold: f32,
    pub weights: &'a [f32; 11],
    pub settings: &'a BacktestSettings,
}

static RAYON_INIT: Once = Once::new();

fn init_rayon() {
    RAYON_INIT.call_once(|| {
        // F-695 closure (2026-05-25 — F-CORE3): resolved through the
        // typed `BacktestRuntimeOverrides::rayon_threads` boundary so
        // the env vars (`FOREX_BOT_RUST_THREADS` /
        // `RAYON_NUM_THREADS`) are read once at process startup.
        let threads = current_backtest_runtime_overrides().rayon_threads;
        if let Some(n) = threads {
            // `build_global` errors if the global pool was already built
            // (e.g. another crate touched rayon first); that's expected
            // and harmless for the rest of the run.
            if let Err(err) = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
            {
                tracing::debug!(
                    target: "neoethos_search::eval",
                    requested_threads = n,
                    error = %err,
                    "rayon global pool already initialised; thread count not overridden"
                );
            }
        }
    });
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let (mean, std) = neoethos_core::utils::mean_std(values);
    if !mean.is_finite() || !std.is_finite() {
        return (0.0, 0.0);
    }
    (mean, std)
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionSpreadProfile {
    pub asian_pips: f64,
    pub overlap_pips: f64,
    pub late_ny_pips: f64,
}

impl SessionSpreadProfile {
    /// Resolve the bucket spread (pips) for a UTC unix-millisecond timestamp.
    pub fn spread_pips_at(self, timestamp_ms: i64) -> f64 {
        let hour = utc_hour_of_day(timestamp_ms);
        if (7..16).contains(&hour) {
            self.overlap_pips
        } else if (16..22).contains(&hour) {
            self.late_ny_pips
        } else {
            self.asian_pips
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
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub max_hold_bars: usize,
    pub min_hold_bars: usize,
    pub max_trades_per_day: usize,
    pub gap_threshold_ms: i64,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
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
    /// **Phase C (2026-05-28)** — `pnl_net = pnl_gross × (1 −
    /// pnl_conversion_fee_rate)` applied once per closed trade.
    /// Fraction (0.005 = 0.5 %), default 0.0.
    pub pnl_conversion_fee_rate: f64,
}

impl BacktestSettings {
    /// Resolve the spread in pips for a single bar. Uses the typed
    /// session profile when set, else the scalar `spread_pips`.
    #[inline]
    pub fn spread_pips_for_bar(&self, timestamp_ms: i64) -> f64 {
        match self.session_spread_profile {
            Some(profile) if timestamp_ms > 0 => profile.spread_pips_at(timestamp_ms),
            _ => self.spread_pips,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub net_profit: f64,
    pub sharpe: f64,
    pub peak_equity: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub trade_count: usize,
    pub consistency: f64,
    pub max_daily_drawdown: f64,
}

/// **F-001 documentation (2026-05-25)** — index-7 slot is *deliberately
/// reserved* and MUST be passed through as 0.0 on the round-trip.
///
/// History: an earlier revision used index 7 for `average_trade_pnl`
/// (computed inline in the kernel). That metric was dropped from the
/// struct but the `[f64; 11]` array shape was kept to avoid breaking
/// the GPU kernel's per-gene output stride (which writes 11 floats per
/// gene). Any caller that hand-rolls a `[f64; 11]` MUST write `0.0` at
/// index 7 or the round-trip via [`BacktestMetrics::from_metric_array`]
/// will silently misinterpret subsequent fields.
///
/// Callers should prefer the named-field constructor + `to_metric_array`
/// rather than hand-rolling the array. The constant below documents the
/// reserved slot for grep-discoverability.
pub const BACKTEST_METRICS_RESERVED_INDEX_7: usize = 7;

impl BacktestMetrics {
    /// Index of the deliberately-reserved slot in the array form. See
    /// [`BACKTEST_METRICS_RESERVED_INDEX_7`] for history.
    pub const RESERVED_INDEX_7: usize = BACKTEST_METRICS_RESERVED_INDEX_7;

    pub fn from_metric_array(metrics: [f64; 11]) -> Self {
        // metrics[7] is the reserved-slot (F-001 / 2026-05-25 doc fix).
        // Old kernel revision used it for average_trade_pnl; that field
        // was dropped but the array width was kept to preserve the GPU
        // output stride. We do not read index 7 here.
        Self {
            net_profit: metrics[0],
            sharpe: metrics[1],
            peak_equity: metrics[2],
            max_drawdown: metrics[3],
            win_rate: metrics[4],
            profit_factor: metrics[5],
            expectancy: metrics[6],
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
        // Index 7 is the reserved slot (F-001 — see struct-level doc).
        // Always 0.0 — DO NOT repurpose without updating every caller
        // that hand-rolls a [f64; 11] expecting this slot to stay zero.
        [
            self.net_profit,
            self.sharpe,
            self.peak_equity,
            self.max_drawdown,
            self.win_rate,
            self.profit_factor,
            self.expectancy,
            0.0, // F-001: reserved index 7 — see BACKTEST_METRICS_RESERVED_INDEX_7
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
        // sentinels so any caller that uses Default::default() WITHOUT
        // then binding a real symbol via `for_symbol(...)` will be
        // caught by the downstream NaN-fitness guard. Production
        // backtests MUST construct via `for_symbol(...)` — see
        // [`BacktestSettings::for_symbol`].
        Self {
            sl_pips: 20.0,
            tp_pips: 40.0,
            max_hold_bars: 0,
            min_hold_bars: 0,
            max_trades_per_day: 0,
            gap_threshold_ms: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
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
        }
    }
}

impl BacktestSettings {
    /// **F-003 fix** (2026-05-25 — operator directive: kill the EURUSD-
    /// fallback path).
    ///
    /// Real-data backtest entry point. Resolves the per-symbol cost
    /// profile via [`crate::genetic::strategy_gene::infer_market_cost_profile`]
    /// and populates `pip_value`, `pip_value_per_lot`, `spread_pips`,
    /// `commission_per_trade` from it. Non-cost knobs (sl_pips, tp_pips,
    /// trailing, etc.) inherit from `Default::default()` — callers can
    /// override post-construction with struct-update syntax.
    ///
    /// **Mirrors** [`crate::genetic::strategy_gene::EvaluationConfig::for_symbol`]
    /// — the audit identified the latter as the template for this method.
    /// The two together kill the F-002 / F-012 / F-025 / F-033 / F-050
    /// EURUSD-leak chain (audit GROUP C extension).
    ///
    /// ## Behaviour on empty / missing inputs
    ///
    /// `infer_market_cost_profile` returns NaN sentinels for empty
    /// symbol / account_currency. Callers that fail to supply those will
    /// get a `BacktestSettings` with NaN cost fields, which the
    /// downstream NaN-fitness guard (see audit GROUP C remediation
    /// 2026-05-25) catches loudly. No silent EURUSD fallback.
    pub fn for_symbol(
        symbol: &str,
        account_currency: &str,
        price_hint: Option<f64>,
        spread_pips_override: Option<f64>,
        commission_override: Option<f64>,
    ) -> Self {
        let profile = crate::genetic::strategy_gene::infer_market_cost_profile(
            symbol,
            account_currency,
            price_hint,
            spread_pips_override,
            commission_override,
        );
        Self {
            pip_value: profile.pip_value,
            pip_value_per_lot: profile.pip_value_per_lot,
            spread_pips: profile.spread_pips,
            commission_per_trade: profile.commission_per_trade,
            // **Phase C (2026-05-28)** — propagate broker-supplied
            // swap & conversion fee. `infer_market_cost_profile`
            // returns 0.0 when the broker hasn't provided these on
            // `SymbolMetadata`, so the production behaviour is
            // "no charge if no broker data" (conservative-rosy);
            // populated values yield the real broker-aligned cost.
            swap_long_pips_per_day: profile.swap_long_pips_per_day,
            swap_short_pips_per_day: profile.swap_short_pips_per_day,
            pnl_conversion_fee_rate: profile.pnl_conversion_fee_rate,
            ..Self::default()
        }
    }
}

/// Typed replacement for the legacy `FOREX_BOT_BACKTEST_*` env vars that
/// previously changed canonical backtest math (`initial_equity`,
/// `month_capacity`) on every metric evaluation. The struct is the single
/// place these values live; production callers install them once via
/// [`install_backtest_runtime_overrides`] (or
/// [`install_backtest_runtime_overrides_from_env`] for backward compat).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BacktestRuntimeOverrides {
    /// Starting equity used for canonical backtest PnL accounting. Must be
    /// strictly positive.
    pub initial_equity: f64,
    /// Maximum number of monthly PnL buckets retained for consistency math.
    /// Must be non-zero.
    pub month_capacity: usize,
    /// Explicit rayon thread-pool size override. `None` → use rayon's
    /// default (one worker per logical core). `Some(n)` pins the global
    /// pool to `n` threads.
    ///
    /// **F-695 closure (2026-05-25 — F-CORE3)**: previously read inline
    /// inside `init_rayon` via `env::var("FOREX_BOT_RUST_THREADS")` +
    /// `env::var("RAYON_NUM_THREADS")`. Now consolidated to this typed
    /// boundary so the env is read once at process startup through
    /// `BacktestRuntimeOverrides::from_env`.
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
    /// One-shot read of the legacy `FOREX_BOT_BACKTEST_*` env vars. This is
    /// the only place the backtest evaluator consults the environment for
    /// these knobs.
    pub fn from_env() -> Self {
        let mut overrides = Self::default();
        if let Some(value) = env::var("FOREX_BOT_BACKTEST_INITIAL_EQUITY")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            overrides.initial_equity = value;
        }
        if let Some(value) = env::var("FOREX_BOT_BACKTEST_MAX_MONTH_BUCKETS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            overrides.month_capacity = value;
        }
        // F-695 closure (2026-05-25 — F-CORE3): rayon thread count. The
        // primary env var matches the audit-recommended NeoEthos naming;
        // the rayon-stdlib `RAYON_NUM_THREADS` is honoured as a fallback
        // so existing deployment scripts keep working unchanged.
        if let Some(value) = env::var("FOREX_BOT_RUST_THREADS")
            .ok()
            .or_else(|| env::var("RAYON_NUM_THREADS").ok())
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
        {
            overrides.rayon_threads = Some(value);
        }
        overrides
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

/// Convenience wrapper that resolves the legacy `FOREX_BOT_BACKTEST_*` env
/// vars once and installs them. Idempotent: subsequent calls are ignored.
pub fn install_backtest_runtime_overrides_from_env() {
    // DOCUMENTED-DEFAULT: OnceLock::set returning Err just means the first
    // installer won (the public API documents idempotency). Nothing to log.
    let _ = BACKTEST_RUNTIME_OVERRIDES.set(BacktestRuntimeOverrides::from_env());
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
        current_backtest_runtime_overrides().initial_equity
    }

    pub fn month_capacity(&self) -> usize {
        current_backtest_runtime_overrides().month_capacity
    }
}

/// **Phase C.2 (2026-05-28)** — apply broker-supplied carry costs to a
/// closed-trade gross PnL.
///
/// `gross_pnl` is the price-derived PnL after commission + half-spread.
/// `in_pos` is +1 for long, −1 for short. `entry_ts_ms` / `exit_ts_ms`
/// are millisecond timestamps; pass 0 when timestamps are unavailable
/// and the swap charge should be skipped (back-compat with pre-Phase-C
/// callers that don't carry timestamps).
///
/// Math:
///   overnight_days = max(exit_ts − entry_ts, 0) / 86_400_000  (fractional)
///   swap_pips_per_day = swap_long if long else swap_short
///     ↑ broker sign convention: positive = credit, negative = charge
///   pnl_with_carry = gross_pnl + swap_pips_per_day × overnight_days
///                      × pip_value_per_lot
///   net_pnl = pnl_with_carry × (1 − pnl_conversion_fee_rate)
///
/// With both swap fields = 0.0 and conversion fee = 0.0 this is the
/// identity, matching the pre-Phase-C kernel exactly.
#[inline]
fn apply_carry_and_fee(
    gross_pnl: f64,
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
    let pnl_with_carry = gross_pnl + swap_credit;
    let conv_fee = settings.pnl_conversion_fee_rate;
    if conv_fee.is_finite() && conv_fee > 0.0 && conv_fee < 1.0 {
        pnl_with_carry * (1.0 - conv_fee)
    } else {
        pnl_with_carry
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fast_evaluate_strategy_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    month_idx: &[i64],
    day_idx: &[i64],
    timestamps: &[i64],
    settings: &BacktestSettings,
) -> [f64; 11] {
    let n = close.len();
    if n == 0 {
        return [0.0; 11];
    }

    let initial_equity = settings.initial_equity();
    let month_capacity = settings.month_capacity();

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

    let mut last_day = -1i64;
    let mut day_peak = equity;
    let mut day_low = equity;
    let mut max_daily_dd = 0.0;
    let mut day_trade_count = 0usize;

    let mut in_pos = 0i8;
    let mut entry_px = 0.0;
    let mut entry_idx = -1i64;
    let mut trail_px = 0.0;

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
        // Per-bar spread cost. When `session_spread_profile` is unset
        // these collapse to the loop-invariant scalar, which the
        // optimiser is free to hoist; the explicit per-bar form keeps
        // the code uniform whether the profile is on or off.
        let (half_spread_px, half_spread_cost) = match session_profile {
            Some(profile) => {
                let s = profile.spread_pips_at(timestamps[i]);
                (s * 0.5 * pip, s * 0.5 * settings.pip_value_per_lot)
            }
            None => (scalar_half_spread_px, scalar_half_spread_cost),
        };
        let m_val = *month_idx.get(i).unwrap_or(&last_month);
        if m_val != last_month {
            if last_month != -1 {
                month_ptr += 1;
                if month_ptr < month_capacity as i64 {
                    monthly_pnls[month_ptr as usize] = current_month_pnl;
                }
            }
            current_month_pnl = 0.0;
            last_month = m_val;
        }

        let d_val = *day_idx.get(i).unwrap_or(&last_day);
        if d_val != last_day {
            if last_day != -1 && day_peak > 0.0 {
                let dd = (day_peak - day_low) / day_peak;
                if dd > max_daily_dd {
                    max_daily_dd = dd;
                }
            }
            last_day = d_val;
            day_peak = equity;
            day_low = equity;
            day_trade_count = 0;
        }

        // Gap detection: force-exit open position when market gap exceeds threshold
        if in_pos != 0 && use_timestamps && settings.gap_threshold_ms > 0 {
            let ts_prev = timestamps[i - 1];
            let ts_curr = timestamps[i];
            if ts_curr > ts_prev && (ts_curr - ts_prev) >= settings.gap_threshold_ms {
                // Force exit at current close (proxy for gap open price)
                let pnl = if in_pos == 1 {
                    (close[i] - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - close[i]) / pip * settings.pip_value_per_lot
                };
                let pnl = pnl - settings.commission_per_trade - half_spread_cost;
                // Phase C.2: apply broker swap + conversion fee.
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
                let pnl = apply_carry_and_fee(pnl, in_pos, entry_ts_ms, exit_ts_ms, settings);
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
                if equity < day_low {
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
        }

        if in_pos != 0 {
            let lo = low[i];
            let hi = high[i];
            let worst_float_pnl = if in_pos == 1 {
                (lo - entry_px) / pip * settings.pip_value_per_lot
            } else {
                (entry_px - hi) / pip * settings.pip_value_per_lot
            };
            if (equity + worst_float_pnl) < day_low {
                day_low = equity + worst_float_pnl;
            }

            let best_float_pnl = if in_pos == 1 {
                (hi - entry_px) / pip * settings.pip_value_per_lot
            } else {
                (entry_px - lo) / pip * settings.pip_value_per_lot
            };
            if (equity + best_float_pnl) > peak_equity {
                peak_equity = equity + best_float_pnl;
            }

            let current_dd = if peak_equity > 0.0 {
                (peak_equity - (equity + worst_float_pnl)) / peak_equity
            } else {
                0.0
            };
            if current_dd > max_dd {
                max_dd = current_dd;
            }

            let mut pnl = 0.0;
            let mut exit = false;

            // Minimum holding period: skip exit checks until min_hold_bars elapsed
            let bars_held = i as i64 - entry_idx;
            let past_min_hold =
                settings.min_hold_bars == 0 || bars_held >= settings.min_hold_bars as i64;

            if past_min_hold {
                if in_pos == 1 {
                    let mut sl = entry_px - (settings.sl_pips * pip);
                    let tp = entry_px + (settings.tp_pips * pip);
                    if settings.trailing_enabled {
                        let mv = hi - entry_px;
                        if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                            let candidate =
                                hi - (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                            if trail_px == 0.0 || candidate > trail_px {
                                trail_px = candidate;
                            }
                            if trail_px > sl {
                                sl = trail_px;
                            }
                        }
                    }
                    if lo <= sl {
                        pnl = (sl - entry_px) / pip * settings.pip_value_per_lot;
                        exit = true;
                    } else if hi >= tp {
                        pnl = (tp - entry_px) / pip * settings.pip_value_per_lot;
                        exit = true;
                    }
                } else {
                    let mut sl = entry_px + (settings.sl_pips * pip);
                    let tp = entry_px - (settings.tp_pips * pip);
                    // L2: trailing stop on a short position only activates once
                    // the price has moved at least `trailing_be_trigger_r * sl_pips`
                    // in the trader's favour. Until then `trail_px` stays at 0.0
                    // and the original `entry_px - sl_pips` stop holds.
                    if settings.trailing_enabled {
                        let mv = entry_px - lo;
                        if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                            let candidate =
                                lo + (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                            if trail_px == 0.0 || candidate < trail_px {
                                trail_px = candidate;
                            }
                            if trail_px < sl {
                                sl = trail_px;
                            }
                        }
                    }
                    if hi >= sl {
                        pnl = (entry_px - sl) / pip * settings.pip_value_per_lot;
                        exit = true;
                    } else if lo <= tp {
                        pnl = (entry_px - tp) / pip * settings.pip_value_per_lot;
                        exit = true;
                    }
                }

                if !exit && settings.max_hold_bars > 0 && bars_held >= settings.max_hold_bars as i64
                {
                    pnl = if in_pos == 1 {
                        (close[i] - entry_px) / pip * settings.pip_value_per_lot
                    } else {
                        (entry_px - close[i]) / pip * settings.pip_value_per_lot
                    };
                    exit = true;
                }
            }

            if exit {
                // Half-spread on exit + commission (half-spread was already paid at entry via adjusted entry_px)
                pnl -= settings.commission_per_trade + half_spread_cost;
                // Phase C.2: apply broker swap + conversion fee.
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
                let pnl = apply_carry_and_fee(pnl, in_pos, entry_ts_ms, exit_ts_ms, settings);
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
                if equity < day_low {
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
            if s != 0 {
                // max_trades_per_day gate
                if settings.max_trades_per_day > 0 && day_trade_count >= settings.max_trades_per_day
                {
                    continue;
                }
                in_pos = s;
                // Bug #1 fix: half-spread applied at entry (entry_px offset), half at exit
                entry_px = close[i] + (s as f64) * half_spread_px;
                entry_idx = i as i64;
                trail_px = 0.0;
                day_trade_count += 1;
            }
        }
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

    let mut month_returns = Vec::new();
    if month_ptr >= 0 {
        let limit = month_ptr.min(month_capacity.saturating_sub(1) as i64) as usize;
        month_returns.extend_from_slice(&monthly_pnls[..=limit]);
    }
    let (avg_m, std_m) = mean_std(&month_returns);

    // Annualize Sharpe using monthly returns: sqrt(12)
    let sharpe = if std_m > 0.0 {
        (avg_m / std_m) * 3.4641
    } else {
        0.0
    };
    let consistency = if std_m > 0.0 {
        (avg_m / std_m).clamp(0.0, 1.0)
    } else if avg_m > 0.0 && month_returns.len() < 2 {
        1.0
    } else {
        0.0
    };

    // Final NaN/inf scrub. A single non-finite slot would poison sorting in
    // the GA (any comparison with NaN returns Equal via partial_cmp fallback).
    let sanitize = |v: f64| if v.is_finite() { v } else { 0.0 };
    [
        sanitize(net_profit),
        sanitize(sharpe),
        sanitize(peak_equity),
        sanitize(max_dd),
        sanitize(win_rate),
        sanitize(pf),
        sanitize(expectancy),
        0.0,
        trade_count as f64,
        sanitize(consistency),
        sanitize(max_daily_dd),
    ]
}

pub fn simulate_trades_core(
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
    if n == 0 {
        return Vec::new();
    }

    let initial_balance = settings.initial_equity();
    let pip = if settings.pip_value.abs() < 1e-12 {
        1e-12
    } else {
        settings.pip_value
    };
    let scalar_half_spread_px = settings.spread_pips * 0.5 * pip;
    let scalar_half_spread_cost = settings.spread_pips * 0.5 * settings.pip_value_per_lot;
    let session_profile = settings.session_spread_profile;

    let mut trades = Vec::new();
    let mut in_pos = 0i8;
    let mut entry_px = 0.0;
    let mut entry_idx = 0usize;
    let mut trail_px = 0.0;
    let mut last_day_key = -1i64;
    let mut day_trade_count = 0usize;

    for i in 1..n {
        // DOCUMENTED-DEFAULT: `n` above is the min length of `timestamps`
        // and the price slices, so `get(i)` is guaranteed Some(_). The
        // `unwrap_or_default()` is defence-in-depth only.
        let ts = timestamps.get(i).copied().unwrap_or_default();

        let (half_spread_px, half_spread_cost) = match session_profile {
            Some(profile) if ts > 0 => {
                let s = profile.spread_pips_at(ts);
                (s * 0.5 * pip, s * 0.5 * settings.pip_value_per_lot)
            }
            _ => (scalar_half_spread_px, scalar_half_spread_cost),
        };

        // Day rollover for max_trades_per_day tracking
        let day_key = if ts > 0 { ts / 86_400_000 } else { -1 };
        if day_key != last_day_key {
            last_day_key = day_key;
            day_trade_count = 0;
        }

        if in_pos != 0 {
            // Gap detection: force-exit on large market gap
            if settings.gap_threshold_ms > 0 && i > 0 {
                let ts_prev = timestamps[i - 1];
                if ts > ts_prev && (ts - ts_prev) >= settings.gap_threshold_ms {
                    let pnl = if in_pos == 1 {
                        (close[i] - entry_px) / pip * settings.pip_value_per_lot
                    } else {
                        (entry_px - close[i]) / pip * settings.pip_value_per_lot
                    };
                    let pnl = pnl - settings.commission_per_trade - half_spread_cost;
                    let entry_time = timestamps.get(entry_idx).copied().unwrap_or_default();
                    let exit_time = ts;
                    // Phase C.2: apply broker swap + conversion fee.
                    let pnl = apply_carry_and_fee(pnl, in_pos, entry_time, exit_time, settings);
                    let duration_hours = if exit_time >= entry_time {
                        Some((exit_time - entry_time) as f64 / 3_600_000.0)
                    } else {
                        None
                    };
                    trades.push(Trade {
                        entry_time,
                        exit_time: Some(exit_time),
                        pnl,
                        pnl_pct: Some(pnl / initial_balance),
                        duration_hours,
                    });
                    in_pos = 0;
                    continue;
                }
            }

            let lo = low[i];
            let hi = high[i];
            let mut pnl = 0.0;
            let mut exit = false;

            // Session-Aware Trading: force exit before weekend
            if ts > 0 && settings.kill_zones_enabled {
                let sec_in_day = (ts / 1000) % 86400;
                let hour = sec_in_day / 3600;
                let days_since_epoch = ts / 86_400_000;
                let weekday = (days_since_epoch + 4) % 7; // 0=Sun, 1=Mon, 5=Fri

                if weekday == 5 && hour >= 20 {
                    exit = true;
                    pnl = if in_pos == 1 {
                        (close[i] - entry_px) / pip * settings.pip_value_per_lot
                    } else {
                        (entry_px - close[i]) / pip * settings.pip_value_per_lot
                    };
                }
            }

            let bars_held = i as i64 - entry_idx as i64;
            let past_min_hold =
                settings.min_hold_bars == 0 || bars_held >= settings.min_hold_bars as i64;

            if in_pos == 1 && !exit && past_min_hold {
                let mut sl = entry_px - (settings.sl_pips * pip);
                let tp = entry_px + (settings.tp_pips * pip);
                if settings.trailing_enabled {
                    let mv = hi - entry_px;
                    if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                        let candidate =
                            hi - (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                        if trail_px == 0.0 || candidate > trail_px {
                            trail_px = candidate;
                        }
                        if trail_px > sl {
                            sl = trail_px;
                        }
                    }
                }
                if lo <= sl {
                    pnl = (sl - entry_px) / pip * settings.pip_value_per_lot;
                    exit = true;
                } else if hi >= tp {
                    pnl = (tp - entry_px) / pip * settings.pip_value_per_lot;
                    exit = true;
                }
            } else if in_pos == -1 && !exit && past_min_hold {
                let mut sl = entry_px + (settings.sl_pips * pip);
                let tp = entry_px - (settings.tp_pips * pip);
                if settings.trailing_enabled {
                    let mv = entry_px - lo;
                    if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                        let candidate =
                            lo + (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                        if trail_px == 0.0 || candidate < trail_px {
                            trail_px = candidate;
                        }
                        if trail_px < sl {
                            sl = trail_px;
                        }
                    }
                }
                if hi >= sl {
                    pnl = (entry_px - sl) / pip * settings.pip_value_per_lot;
                    exit = true;
                } else if lo <= tp {
                    pnl = (entry_px - tp) / pip * settings.pip_value_per_lot;
                    exit = true;
                }
            }

            if !exit
                && past_min_hold
                && settings.max_hold_bars > 0
                && (i - entry_idx) >= settings.max_hold_bars
            {
                pnl = if in_pos == 1 {
                    (close[i] - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - close[i]) / pip * settings.pip_value_per_lot
                };
                exit = true;
            }

            if exit {
                pnl -= settings.commission_per_trade + half_spread_cost;
                let entry_time = timestamps.get(entry_idx).copied().unwrap_or_default();
                let exit_time = timestamps.get(i).copied().unwrap_or(entry_time);
                // Phase C.2: apply broker swap + conversion fee.
                let pnl = apply_carry_and_fee(pnl, in_pos, entry_time, exit_time, settings);
                let duration_hours = if exit_time >= entry_time {
                    Some((exit_time - entry_time) as f64 / 3_600_000.0)
                } else {
                    None
                };
                trades.push(Trade {
                    entry_time,
                    exit_time: Some(exit_time),
                    pnl,
                    pnl_pct: Some(pnl / initial_balance),
                    duration_hours,
                });
                in_pos = 0;
            }
        } else if signals[i - 1] != 0 {
            // Causal: act on the PRIOR bar's signal at THIS bar's close.
            // Same intra-bar look-ahead fix as `fast_evaluate_strategy_core`.
            // Kill zones: block entries
            let mut block_entry = false;
            if ts > 0 && settings.kill_zones_enabled {
                let sec_in_day = (ts / 1000) % 86400;
                let hour = sec_in_day / 3600;
                let min = (sec_in_day % 3600) / 60;
                let days_since_epoch = ts / 86_400_000;
                let weekday = (days_since_epoch + 4) % 7;

                let is_friday_kill = weekday == 5 && hour >= 20;
                let is_monday_kill = weekday == 1 && hour == 0 && min < 30;
                if is_friday_kill || is_monday_kill {
                    block_entry = true;
                }
            }

            // max_trades_per_day gate
            if settings.max_trades_per_day > 0 && day_trade_count >= settings.max_trades_per_day {
                block_entry = true;
            }

            if !block_entry {
                let s = signals[i - 1];
                in_pos = s;
                // Bug #1 fix: half-spread at entry
                entry_px = close[i] + (s as f64) * half_spread_px;
                entry_idx = i;
                trail_px = 0.0;
                day_trade_count += 1;
            }
        }
    }

    trades
}

#[allow(clippy::too_many_arguments)]
fn synthesize_signals_cpu(
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    weights: &[f32; 11],
    gene_index: usize,
    n_samples: usize,
) -> Vec<i8> {
    let mut combined = vec![0.0_f32; n_samples];
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
    let lt = long_thr[gene_index];
    let st = short_thr[gene_index];
    let flags = gene_smc_flags[gene_index];
    let active_sum: f32 = flags
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
    let smc_bypass = crate::genetic::current_genetic_search_runtime_overrides()
        .smc_gate
        .disable_gate;
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

        if active_sum > 0.0 {
            let mut score = 0.0f32;
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
            }
        } else {
            signals[i] = sig;
        }
    }

    signals
}

pub fn evaluate_population_core(
    inputs: PopulationEvalInputs<'_>,
) -> Result<Vec<[f64; 11]>, String> {
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
        smc_data,
        gene_smc_flags,
        gate_threshold,
        weights,
        settings,
    } = inputs;
    init_rayon();
    let n_genes = long_thr.len();
    let n_samples = close.len();

    #[cfg(feature = "gpu")]
    if cuda_eval_signal_kernel_enabled() && cuda_eval_backtest_kernel_enabled() {
        match try_evaluate_population_cuda(
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
            smc_data,
            gene_smc_flags,
            gate_threshold,
            weights,
            settings,
        ) {
            Ok(results) => return Ok(results),
            Err(err) => {
                tracing::warn!(
                    "full cuda evaluator unavailable, falling back to partial gpu/cpu evaluation: {err}"
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    let gpu_signal_rows = if cuda_eval_signal_kernel_enabled() {
        match try_generate_signal_rows_cuda(
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
        ) {
            Ok(rows) => Some(rows),
            Err(err) => {
                tracing::warn!(
                    "cuda evaluator signal kernel unavailable, falling back to cpu evaluator synthesis: {err}"
                );
                None
            }
        }
    } else {
        None
    };

    let results: Vec<[f64; 11]> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            #[cfg(feature = "gpu")]
            let signals = if let Some(signal_rows) = gpu_signal_rows.as_ref() {
                signal_rows[g].clone()
            } else {
                synthesize_signals_cpu(
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
                )
            };

            #[cfg(not(feature = "gpu"))]
            let signals = synthesize_signals_cpu(
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
            fast_evaluate_strategy_core(
                close,
                high,
                low,
                &signals,
                month_idx,
                day_idx,
                timestamps,
                &gene_settings,
            )
        })
        .collect();

    Ok(results)
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
    fn backtest_settings_spread_for_bar_uses_profile_when_present() {
        let mut settings = BacktestSettings::default();
        settings.spread_pips = 99.0;
        // Without a profile, every bar uses the scalar.
        assert!((settings.spread_pips_for_bar(0) - 99.0).abs() < 1e-9);
        assert!((settings.spread_pips_for_bar(9 * 3_600_000) - 99.0).abs() < 1e-9);

        settings.session_spread_profile = Some(SessionSpreadProfile {
            asian_pips: 2.0,
            overlap_pips: 0.5,
            late_ny_pips: 1.5,
        });
        // With a profile, 09:00 UTC resolves to the overlap bucket.
        assert!((settings.spread_pips_for_bar(9 * 3_600_000) - 0.5).abs() < 1e-9);
        // Zero timestamp falls back to the scalar (no real-time signal).
        assert!((settings.spread_pips_for_bar(0) - 99.0).abs() < 1e-9);
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

    // ─── Phase C.2 carry-cost + conversion-fee helper ────────────────
    //
    // These tests pin the math used by every trade-close branch of the
    // CPU evaluator. All four sites call `apply_carry_and_fee` so a
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
        assert!((apply_carry_and_fee(123.45, 1, 0, 0, &s) - 123.45).abs() < 1e-9);
        // Long trade held 5 days, zero swap & fee → still gross.
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 5 * 86_400_000;
        assert!((apply_carry_and_fee(123.45, 1, entry, exit, &s) - 123.45).abs() < 1e-9);
    }

    #[test]
    fn carry_fee_negative_swap_reduces_pnl_for_long() {
        // EURUSD-style: swap_long = −2.445 pips/day, pip_value_per_lot = $10.
        // Long held 5.0 days → carry = −2.445 × 5 × 10 = −$122.25.
        // Gross $200 → net $77.75. No fee.
        let s = settings_with_carry(-2.445, -0.105, 0.0, 10.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 5 * 86_400_000;
        let net = apply_carry_and_fee(200.0, 1, entry, exit, &s);
        assert!((net - 77.75).abs() < 1e-6, "expected ~77.75, got {net}");
    }

    #[test]
    fn carry_fee_positive_swap_credits_short() {
        // XTIUSD-style: swap_short = +0.4375 pips/day, pip_value_per_lot = $1.
        // Short held 4.0 days → carry = +0.4375 × 4 × 1 = +$1.75 credit.
        let s = settings_with_carry(-0.5, 0.4375, 0.0, 1.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 4 * 86_400_000;
        let net = apply_carry_and_fee(10.0, -1, entry, exit, &s);
        assert!((net - 11.75).abs() < 1e-6, "expected ~11.75, got {net}");
    }

    #[test]
    fn carry_fee_fractional_days() {
        // 12 hours = 0.5 days. swap = −1.0, pip_value = 10 → carry = −5.0.
        let s = settings_with_carry(-1.0, -1.0, 0.0, 10.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry + 12 * 3_600_000;
        let net = apply_carry_and_fee(50.0, 1, entry, exit, &s);
        assert!((net - 45.0).abs() < 1e-6, "expected ~45.0, got {net}");
    }

    #[test]
    fn carry_fee_conversion_scales_after_swap() {
        // Conversion fee 0.5% applied AFTER swap.
        // No swap, fee = 0.005. Gross $100 → net $99.50.
        let s = settings_with_carry(0.0, 0.0, 0.005, 10.0);
        let net = apply_carry_and_fee(100.0, 1, 0, 0, &s);
        assert!((net - 99.5).abs() < 1e-6, "expected 99.5, got {net}");
    }

    #[test]
    fn carry_fee_handles_missing_timestamps_as_day_trade() {
        // entry_ts = 0 means "no timestamp data": skip swap entirely.
        let s = settings_with_carry(-100.0, -100.0, 0.0, 10.0);
        let net = apply_carry_and_fee(50.0, 1, 0, 1_700_000_000_000, &s);
        assert!((net - 50.0).abs() < 1e-9, "expected 50.0 (no swap), got {net}");
    }

    #[test]
    fn carry_fee_rejects_inverted_timestamps() {
        // exit < entry: no negative time, no swap charge.
        let s = settings_with_carry(-1.0, -1.0, 0.0, 10.0);
        let entry = 1_700_000_000_000_i64;
        let exit = entry - 86_400_000;
        let net = apply_carry_and_fee(50.0, 1, entry, exit, &s);
        assert!((net - 50.0).abs() < 1e-9);
    }

    #[test]
    fn carry_fee_rejects_out_of_range_conversion_fee() {
        // fee = 1.0 would wipe out PnL — reject and skip.
        let s = settings_with_carry(0.0, 0.0, 1.0, 10.0);
        assert!((apply_carry_and_fee(100.0, 1, 0, 0, &s) - 100.0).abs() < 1e-9);
        // Negative fee also rejected.
        let s = settings_with_carry(0.0, 0.0, -0.1, 10.0);
        assert!((apply_carry_and_fee(100.0, 1, 0, 0, &s) - 100.0).abs() < 1e-9);
    }
}
