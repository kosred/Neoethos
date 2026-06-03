use crate::artifact_io::{stable_json_hash, write_json_atomic};
use crate::eval::{BacktestMetrics, fast_evaluate_strategy_core, simulate_trades_core};
use crate::genetic::strategy_gene::EvaluationConfig;
use crate::genetic::{
    Gene, evolve_search_with_progress_and_limits, month_day_indices, signals_for_gene_full,
};
use crate::quality::{StrategyMetrics, StrategyQualityAnalyzer, Trade};
use crate::validation::{
    CanonicalBacktestArtifactFile, CanonicalBacktestScope, CombinatorialPurgedCV, ForwardTestInput,
    ForwardTestValidationArtifactFile, ForwardTestValidationScope, PropFirmRiskInput,
    PropFirmRiskRules, PropFirmRiskValidationArtifactFile, PropFirmRiskValidationScope,
    WalkforwardBacktestInput, WalkforwardSummary, WalkforwardValidationArtifactFile,
    WalkforwardValidationScope, compute_forward_test_summary, compute_prop_firm_risk_summary,
    embargoed_walkforward_backtest, write_canonical_backtest_artifact_atomic,
    write_forward_test_validation_artifact_atomic, write_prop_firm_risk_validation_artifact_atomic,
    write_walkforward_validation_artifact_atomic,
};
use anyhow::{Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use neoethos_core::contracts::{
    DeterminismPolicy, LiveValidationEvidence, TemporalFeatureContract, ValidationEvidenceManifest,
};
use neoethos_core::domain::prop_firm::PropFirmConstraints;
use neoethos_data::{FeatureFrame, Ohlcv};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Typed runtime knobs that previously lived only in `NEOETHOS_BOT_*` env vars.
///
/// These values change *production* discovery semantics (which features are
/// kept, how much data the stage-1 funnel sees, what counts as in-sample for
/// the prefilter), so they belong in typed config rather than ambient env
/// state. These are configured via `models.discovery_runtime` (typed config)
/// and resolved by [`DiscoveryRuntimeOverrides::from_settings`]; the legacy
/// [`DiscoveryRuntimeOverrides::from_env`] reader is retained for reference
/// only — the discovery cycle no longer reads the environment for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage1Window {
    /// Slice from the most recent rows. Captures the latest regime but is
    /// catastrophic if the caller passed full data including the held-out
    /// OOS tail — stage 1 then trains directly on OOS rows. Use only when
    /// the caller has already split in-sample / out-of-sample.
    MostRecent,
    /// Slice from the earliest rows. Maximally distant from any held-out
    /// tail, so it is OOS-safe even if the caller forgot to split. Default.
    Earliest,
}

impl Stage1Window {
    fn from_env_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "most_recent" | "recent" | "tail" => Some(Self::MostRecent),
            "earliest" | "head" | "oldest" => Some(Self::Earliest),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscoveryRuntimeOverrides {
    /// Maximum number of features to keep after the in-sample correlation
    /// prefilter. `0` disables the prefilter entirely.
    pub prefilter_top_k: usize,
    /// Fraction of rows treated as in-sample when ranking features. Must be
    /// strictly positive and at most `1.0`.
    pub prefilter_insample_frac: f64,
    /// Fraction of rows fed to the multi-stage funnel's first stage.
    /// Clamped to `[0.01, 1.0]` at use time.
    pub funnel_stage1_pct: f64,
    /// Where in the input window to slice the stage-1 fast-evaluation
    /// rows. Defaults to [`Stage1Window::Earliest`] for OOS safety.
    pub stage1_window: Stage1Window,
    /// **F-096 fix (2026-05-25)** — minimum historical-data window
    /// in years that the discovery pipeline requires before it agrees
    /// to run. Default `10` per operator real-data directive
    /// 2026-05-24. Setting to `0` skips the check (test fixtures /
    /// demo replays). The pre-flight check lives in
    /// [`ensure_sufficient_history`] and runs at the top of
    /// `run_discovery_cycle_with_progress`.
    pub min_history_years: u32,
}

impl Default for DiscoveryRuntimeOverrides {
    fn default() -> Self {
        Self {
            prefilter_top_k: 50,
            prefilter_insample_frac: 0.70,
            funnel_stage1_pct: 0.25,
            stage1_window: Stage1Window::Earliest,
            // **2026-05-26 operator directive (Κωνσταντίνος)**: the design
            // intent was always "use 80/20 of WHATEVER data we have", not
            // "require absolute 10y before running". The 80/20 train/val
            // split is enforced downstream by `prop_search_val_years` (last
            // N years as validation) which already adapts to any window
            // length. Setting the absolute-minimum gate to 0 by default
            // means short windows (5y M5, 3y crypto, etc.) run through
            // the same pipeline and the operator gets a *result* (even if
            // empty portfolio because the strategies overfit) rather than
            // a hard "Failed: insufficient history" preflight stop. Operators
            // who want the strict 10y gate back can set
            // `NEOETHOS_BOT_MIN_HISTORY_YEARS=10` via env.
            //
            // F-096 history (2026-05-24, now superseded): the previous
            // default was 10 because synthetic-data leaks into discovery
            // had produced misleading results. With Vortex now refusing
            // synthetic fallbacks (#221) the leak risk is gone, so the
            // 10y floor is no longer needed.
            min_history_years: 0,
        }
    }
}

impl DiscoveryRuntimeOverrides {
    /// One-shot read of the legacy `NEOETHOS_BOT_*` env vars. This is the only
    /// place in `neoethos-search` that consults the environment for these
    /// knobs; production callers should prefer constructing the struct from
    /// typed config.
    pub fn from_env() -> Self {
        let mut overrides = Self::default();
        if let Some(top_k) = std::env::var("NEOETHOS_BOT_PREFILTER_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            overrides.prefilter_top_k = top_k;
        }
        if let Some(insample) = std::env::var("NEOETHOS_BOT_PREFILTER_INSAMPLE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
        {
            overrides.prefilter_insample_frac = insample;
        }
        if let Some(stage1) = std::env::var("NEOETHOS_BOT_FUNNEL_STAGE1_PCT")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
        {
            overrides.funnel_stage1_pct = stage1.clamp(0.01, 1.0);
        }
        if let Some(window) = std::env::var("NEOETHOS_BOT_FUNNEL_STAGE1_WINDOW")
            .ok()
            .and_then(|v| Stage1Window::from_env_str(&v))
        {
            overrides.stage1_window = window;
        }
        // F-096: minimum-history-years env override. 0 disables the
        // check (for test runners and `--allow-short-history` operator
        // flag). Production deployments leave it at the default 10y.
        if let Some(years) = std::env::var("NEOETHOS_BOT_MIN_HISTORY_YEARS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            overrides.min_history_years = years;
        }
        overrides
    }

    /// Config-driven replacement for [`from_env`]: reads the discovery
    /// runtime knobs from `models.discovery_runtime` instead of the
    /// `NEOETHOS_BOT_*` environment. The validation mirrors `from_env`
    /// (out-of-range values fall back to the default) so config defaults
    /// reproduce the env-absent behaviour exactly.
    pub fn from_settings(settings: &neoethos_core::Settings) -> Self {
        let cfg = &settings.models.discovery_runtime;
        let mut overrides = Self::default();
        overrides.prefilter_top_k = cfg.prefilter_top_k;
        if cfg.prefilter_insample_frac.is_finite()
            && cfg.prefilter_insample_frac > 0.0
            && cfg.prefilter_insample_frac <= 1.0
        {
            overrides.prefilter_insample_frac = cfg.prefilter_insample_frac;
        }
        if cfg.funnel_stage1_pct.is_finite() {
            overrides.funnel_stage1_pct = cfg.funnel_stage1_pct.clamp(0.01, 1.0);
        }
        if let Some(window) = Stage1Window::from_env_str(&cfg.stage1_window) {
            overrides.stage1_window = window;
        }
        overrides.min_history_years = cfg.min_history_years;
        overrides
    }

    fn resolved_funnel_stage1_pct(&self) -> f64 {
        if self.funnel_stage1_pct.is_finite() {
            self.funnel_stage1_pct.clamp(0.01, 1.0)
        } else {
            0.25
        }
    }

    fn resolved_prefilter_insample_frac(&self) -> f64 {
        if self.prefilter_insample_frac.is_finite()
            && self.prefilter_insample_frac > 0.0
            && self.prefilter_insample_frac <= 1.0
        {
            self.prefilter_insample_frac
        } else {
            0.70
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub timeframe_label: String,
    pub evaluation_symbol: String,
    pub evaluation_account_currency: String,
    pub evaluation_spread_pips: f64,
    pub evaluation_commission_per_trade: f64,
    pub population: usize,
    pub generations: usize,
    pub max_indicators: usize,
    pub candidate_count: usize,
    pub portfolio_size: usize,
    pub max_rows: usize,
    pub max_rows_by_timeframe: HashMap<String, usize>,
    pub max_hours: f64,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub cpcv_max_rows: usize,
    pub filtering: crate::genetic::FilteringConfig,
    /// Starting account balance used for PnL%, DD%, and regime loss limits.
    pub initial_balance: f64,
    /// Reject a gene if any regime-specific PnL drops below
    /// `-initial_balance * max_regime_loss_pct / 100`.
    pub max_regime_loss_pct: f64,
    /// Higher timeframes to include in multitimeframe feature preparation.
    pub higher_timeframes: Vec<String>,
    /// Typed replacements for the legacy `NEOETHOS_BOT_PREFILTER_*` /
    /// `NEOETHOS_BOT_FUNNEL_STAGE1_PCT` env vars.
    pub runtime_overrides: DiscoveryRuntimeOverrides,
    /// When `Some`, the discovery pipeline replaces its full-history
    /// walkforward consistency gate with a "passes prop-firm rules on
    /// N random 30-day windows ≥ pass_rate" gate. Populated from
    /// the FTMO baseline (+ the `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_*` overrides
    /// that `derive_prop_firm_gate` still reads — Stage B tail) when
    /// `apply_mode_overrides` runs in PropFirm mode. `None` keeps the
    /// production behavior unchanged.
    pub prop_firm_gate: Option<PropFirmGateOverrides>,
    /// 2026-05-26 operator directive (dual-mode product): Monte-Carlo
    /// perturbation runs per surviving candidate. Previously hardcoded 100.
    pub mc_runs: u32,
    /// Minimum profitable MC runs required (out of `mc_runs`). Previously
    /// hardcoded 70 (i.e. 70% threshold).
    pub mc_min_profitable: u32,
    /// Spread (pips) used in the sensitivity test. Previously hardcoded 2.0.
    pub sensitivity_spread_pips: f64,
    /// Commission per lot used in the sensitivity test. Previously
    /// hardcoded $7/lot.
    pub sensitivity_commission_per_lot: f64,
    /// Opt-in adaptive coarse-threshold ladder (config-driven replacement
    /// for the `NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS` env flag). Read by
    /// `run_discovery_cycle` before gene initialisation. Default `false`
    /// reproduces the env-absent behaviour.
    pub adaptive_thresholds: bool,
    /// Discovery search regime (config-driven via `models.discovery_mode`).
    /// `PropFirm` (default) applies permissive filter floors + the FTMO
    /// window-pass gate; `Strict` keeps the full `FilteringConfig` floors.
    /// Replaces the env-only `resolve_discovery_mode()` that read
    /// `NEOETHOS_BOT_DISCOVERY_MODE` / `_PERMISSIVE`. Consumed by
    /// `apply_mode_overrides`.
    pub mode: DiscoveryMode,
    /// Prop-firm window-pass gate parameters (config-driven via
    /// `models.discovery_runtime.prop_firm_gate`). Consumed by
    /// `derive_prop_firm_gate` when `apply_mode_overrides` runs in PropFirm
    /// mode. Replaces the `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_*` env overrides.
    pub prop_firm_gate_params: neoethos_core::config::PropFirmGateConfig,
    /// Risky-Mode capital-multiplication goal (config-driven via `system.risky_*`).
    /// When `mode == Risky` these PRESSURE the candidate ranking: each strategy
    /// is scored by how well it could compound from `risky_start_balance` to
    /// `risky_target_balance` within `risky_horizon_days` at safe (half-Kelly)
    /// sizing of its own measured edge — so the search surfaces strategies that
    /// can actually hit the operator's goal in time. Ignored in Strict/PropFirm.
    pub risky_start_balance: f64,
    pub risky_target_balance: f64,
    pub risky_horizon_days: f64,
}

/// Configuration for the prop-firm window-pass gate.
#[derive(Debug, Clone)]
pub struct PropFirmGateOverrides {
    pub rules: PropFirmRiskRules,
    pub n_windows: usize,
    pub window_days: usize,
    pub pass_rate: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            timeframe_label: "M1".to_string(),
            // GROUP C remediation (operator directive 2026-05-25):
            // empty + NaN sentinels so a DiscoveryConfig that was
            // constructed via Default::default() (rather than via
            // `for_symbol(...)` or explicit field assignment) does
            // NOT silently backtest against EURUSD/USD. Production
            // callers MUST set these explicitly before run.
            evaluation_symbol: String::new(),
            evaluation_account_currency: String::new(),
            evaluation_spread_pips: f64::NAN,
            evaluation_commission_per_trade: f64::NAN,
            population: 1000,
            generations: 10,
            max_indicators: 5,
            candidate_count: 5000,
            portfolio_size: 2000,
            max_rows: 0,
            max_rows_by_timeframe: HashMap::new(),
            max_hours: 0.0,
            corr_threshold: 0.85,
            min_trades_per_day: 0.2,
            walkforward_splits: 20,
            embargo_minutes: 120,
            enable_cpcv: true,
            cpcv_n_splits: 5,
            cpcv_n_test_groups: 2,
            cpcv_embargo_pct: 0.01,
            cpcv_purge_pct: 0.02,
            cpcv_min_phi: 0.80,
            cpcv_max_rows: 0,
            filtering: crate::genetic::FilteringConfig::default(),
            initial_balance: 100_000.0,
            max_regime_loss_pct: 3.0,
            higher_timeframes: Vec::new(),
            runtime_overrides: DiscoveryRuntimeOverrides::default(),
            prop_firm_gate: None,
            // 2026-05-26 operator directive (dual-mode product): defaults
            // reproduce the previous hardcoded behavior; from_settings
            // overrides from typed config.
            mc_runs: 100,
            mc_min_profitable: 70,
            sensitivity_spread_pips: 2.0,
            sensitivity_commission_per_lot: 7.0,
            adaptive_thresholds: false,
            // Env-absent default reproduces the retired
            // resolve_discovery_mode() fallback (PropFirm).
            mode: DiscoveryMode::PropFirm,
            prop_firm_gate_params: neoethos_core::config::PropFirmGateConfig::default(),
            // Risky-Mode goal defaults (mirror SystemConfig): 100 -> 50,000 in
            // 180 days. Ignored unless mode == Risky.
            risky_start_balance: 100.0,
            risky_target_balance: 50000.0,
            risky_horizon_days: 180.0,
        }
    }
}

impl DiscoveryConfig {
    pub fn from_settings(settings: &neoethos_core::Settings) -> Self {
        let model_settings = &settings.models;
        let filtering = crate::genetic::FilteringConfig {
            min_trades: model_settings.prop_min_trades.max(1) as f64,
            anomaly_guard: true,
            min_positive_months: model_settings.prop_search_val_min_positive_months,
            min_trades_per_month: model_settings.prop_search_val_min_trades_per_month as f64,
            min_monthly_return_pct: model_settings.prop_search_val_min_monthly_profit_pct / 100.0,
            log_trades: model_settings.prop_search_val_log_trades,
            trade_log_max: model_settings.prop_search_val_trade_log_max.max(1),
            opportunistic_enabled: model_settings.prop_search_opportunistic_enabled,
            use_opportunistic_candidates: model_settings.prop_search_use_opportunistic,
            opportunistic_min_positive_months: model_settings
                .prop_search_opportunistic_min_positive_months,
            opportunistic_min_trades_per_month: model_settings
                .prop_search_opportunistic_min_trades_per_month
                as f64,
            opportunistic_min_trade_return_pct: model_settings
                .prop_search_opportunistic_min_trade_return_pct,
            opportunistic_max_dd: model_settings.prop_search_opportunistic_max_dd.max(0.0),
            ..Default::default()
        };

        // P2 fix: `0` now means "no artificial cap — use population *
        // generations". Previously `0` silently became `population` which
        // capped the archive way below what the heavy reject funnel needs.
        let candidate_count = if model_settings.prop_search_val_candidates == 0 {
            model_settings
                .prop_search_population
                .saturating_mul(model_settings.prop_search_generations.max(1))
                .max(model_settings.prop_search_population.max(50))
        } else {
            model_settings.prop_search_val_candidates.max(1)
        };

        Self {
            timeframe_label: settings.system.base_timeframe.clone(),
            evaluation_symbol: settings.system.symbol.clone(),
            // F-304 fix (2026-05-28): SystemConfig.account_currency is
            // the typed channel for operator/broker-supplied account
            // currency, populated from one of:
            //  - `config.yaml` `system.account_currency`
            //  - cTrader trader profile (bridge writes back at startup)
            //  - `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` env override
            // Empty propagates downstream so the cost-model NaN guard
            // can reject runs that haven't bound a real currency. The
            // previous F-007 fix used `String::new()` here unconditionally,
            // making *every* `from_settings` call fall into the NaN trap
            // even when the operator had set the value — root cause #304.
            evaluation_account_currency: settings.system.account_currency.clone(),
            evaluation_spread_pips: settings.risk.backtest_spread_pips.max(0.0),
            evaluation_commission_per_trade: settings.risk.commission_per_lot.max(0.0),
            population: model_settings.prop_search_population.max(10),
            generations: model_settings.prop_search_generations.max(1),
            // P2 fix: `0` now means "use ALL available enabled features"
            // (sentinel value `usize::MAX` so downstream `min(n_features)`
            // collapses to the actual feature count). Previously
            // silently became 5, which limited search to a tiny subset.
            max_indicators: if model_settings.prop_search_max_indicators == 0 {
                usize::MAX
            } else {
                model_settings.prop_search_max_indicators.max(1)
            },
            candidate_count,
            portfolio_size: model_settings.prop_search_portfolio_size.max(1),
            max_rows: model_settings.prop_search_max_rows,
            max_rows_by_timeframe: model_settings.prop_search_max_rows_by_tf.clone(),
            max_hours: model_settings.prop_search_max_hours.max(0.0),
            // 2026-05-26 operator directive (dual-mode product): wired from
            // Settings.models.prop_search_corr_threshold. Defaults to 0.85
            // (the previous hardcoded value) when the config key is absent.
            corr_threshold: model_settings.prop_search_corr_threshold.clamp(0.0, 1.0),
            min_trades_per_day: model_settings.prop_search_val_min_trades_per_day.max(0.2),
            walkforward_splits: model_settings.walkforward_splits.max(2),
            embargo_minutes: model_settings.embargo_minutes,
            enable_cpcv: model_settings.enable_cpcv,
            cpcv_n_splits: model_settings.cpcv_n_splits.max(2),
            cpcv_n_test_groups: model_settings.cpcv_n_test_groups.max(1),
            cpcv_embargo_pct: model_settings.cpcv_embargo_pct.max(0.0),
            cpcv_purge_pct: model_settings.cpcv_purge_pct.max(0.0),
            cpcv_min_phi: model_settings.cpcv_min_phi.max(0.0),
            cpcv_max_rows: model_settings.cpcv_max_rows,
            filtering,
            initial_balance: settings.risk.initial_balance.max(1.0),
            max_regime_loss_pct: 3.0,
            higher_timeframes: settings.system.higher_timeframes.clone(),
            runtime_overrides: DiscoveryRuntimeOverrides::from_settings(settings),
            prop_firm_gate: None,
            // 2026-05-26 operator directive (dual-mode product): Settings is
            // now the single source of truth for these knobs. The corr_threshold
            // assignment a few lines above stays as 0.85 fallback — it gets
            // overwritten here so the operator's config wins.
            mc_runs: model_settings.prop_search_mc_runs.max(1),
            mc_min_profitable: model_settings
                .prop_search_mc_min_profitable
                .min(model_settings.prop_search_mc_runs.max(1)),
            sensitivity_spread_pips: model_settings
                .prop_search_sensitivity_spread_pips
                .max(0.0),
            sensitivity_commission_per_lot: model_settings
                .prop_search_sensitivity_commission_per_lot
                .max(0.0),
            adaptive_thresholds: model_settings.discovery_runtime.adaptive_thresholds,
            mode: resolve_discovery_mode(
                &settings.system.trading_mode,
                &model_settings.discovery_mode,
            ),
            prop_firm_gate_params: model_settings.discovery_runtime.prop_firm_gate.clone(),
            risky_start_balance: settings.system.risky_start_balance_usd,
            risky_target_balance: settings.system.risky_target_balance_usd,
            risky_horizon_days: settings.system.risky_horizon_days as f64,
        }
    }

    /// Resolve runtime knobs. The system prefers self-tuning over
    /// hand-rolled env vars: if the caller does not opt out via
    /// `NEOETHOS_BOT_DISCOVERY_MODE=strict`, discovery enters its
    /// "smart prop-firm" mode automatically — permissive filters,
    /// FTMO-rule scoring on N random 60-day windows, ranking-based
    /// portfolio selection (no thresholds to tune), window count
    /// auto-derived from dataset length.
    ///
    /// Env vars are still honored as overrides for the rare cases
    /// where the operator wants to lock in a specific value, but the
    /// happy-path call needs none of them.
    pub fn apply_mode_overrides(mut self) -> Self {
        // Config-consolidation (2026-06-03): the mode comes from `self.mode`
        // (set by `from_settings` from `models.discovery_mode`) and the
        // discovery runtime knobs from `self.runtime_overrides` (set by
        // `from_settings` from `models.discovery_runtime`) — neither is read
        // from the environment any more. This applies the mode-dependent
        // overrides: PropFirm permissive filter floors, TF-scaled
        // trade-frequency floors, and the FTMO window-pass gate. (The FTMO
        // *rule parameters* are still derived inside `derive_prop_firm_gate`
        // — that env read is the Stage B tail.)
        let mode = self.mode;

        if matches!(mode, DiscoveryMode::PropFirm) {
            // Permissive filter floor — the GA's output is judged by the
            // prop-firm window-pass score, not by these legacy thresholds.
            self.filtering.max_dd = 0.50;
            self.filtering.min_profit = 0.0;
            self.filtering.min_trades = 1.0;
            self.filtering.min_sharpe = -10.0;
            self.filtering.min_win_rate = 0.0;
            self.filtering.min_profit_factor = 0.0;
            self.filtering.anomaly_guard = false;
            self.cpcv_min_phi = 0.0;
            // Lowered from 0.02 (~30 trades over 1500 days) to 0.001
            // (~1.5 trades over 1500 days) — the previous floor was
            // killing every gene whose `long_threshold` was just shy
            // of triggering frequently, and the prop-firm window-pass
            // gate downstream already filters out genuinely useless
            // strategies on its own.
            // Permissive PropFirm trade-frequency floor — this was the
            // env-absent default of the retired
            // NEOETHOS_BOT_DISCOVERY_MIN_TRADES_PER_DAY override; the
            // window-pass gate downstream filters genuinely useless genes.
            self.min_trades_per_day = 0.001;

            // F-305 fix (2026-05-28): scale `min_trades_per_month` by TF
            // bar density. The operator's `config.yaml` sets the value
            // for M1/M5/M15 (typically 15 trades/month). On D1 with ~21
            // bars/month, 15 trades requires trading 70%+ of bars —
            // mathematically forced over-trading. Empty portfolios on
            // D1/H4 weren't a strategy problem; they were a config
            // problem masking a strategy.
            //
            // Scale factors picked to keep daily trade frequency
            // approximately stable across TFs:
            //   M1/M3/M5/M15: 1.0× operator value  (intra-day strategies)
            //   M30:          0.67× (15 → 10/month)
            //   H1:           0.40× (15 → 6/month)
            //   H4:           0.20× (15 → 3/month)
            //   D1:           0.13× (15 → 2/month — ~1 trade/two weeks)
            //   W1/MN1:       0.03× (essentially "any trade qualifies")
            //
            // Risky/Strict modes keep the operator's exact value — those
            // are scenario-specific runs where the operator explicitly
            // wants to over- or under-shoot.
            let scale = min_trades_per_month_scale_for_tf(&self.timeframe_label);
            if self.filtering.min_trades_per_month > 0.0 && scale < 1.0 {
                let base = self.filtering.min_trades_per_month;
                self.filtering.min_trades_per_month = (base * scale).max(0.5);
                tracing::info!(
                    target: "neoethos_search::discovery",
                    tf = %self.timeframe_label,
                    base = base,
                    scale = scale,
                    scaled = self.filtering.min_trades_per_month,
                    "F-305: scaled min_trades_per_month for PropFirm mode on higher TF"
                );
            }
            if self.filtering.opportunistic_min_trades_per_month > 0.0 && scale < 1.0 {
                self.filtering.opportunistic_min_trades_per_month =
                    (self.filtering.opportunistic_min_trades_per_month * scale).max(0.5);
            }

            self.prop_firm_gate = Some(self.derive_prop_firm_gate());
        }

        if matches!(mode, DiscoveryMode::Risky) {
            // Risky / capital-multiplication mode: KEEP the aggressive,
            // high-drawdown strategies that the strict / prop-firm floors would
            // reject, but impose NO FTMO window-pass gate — we are not passing a
            // challenge, we are compounding a small balance toward a large
            // target. Deep drawdown is acceptable; the growth-tilted ranking
            // (see `calculate_income_score`) prefers the fastest compounders.
            // Floors stay loose-but-sane so genuinely broken genes (negative
            // edge, never-trading) still drop out.
            self.filtering.max_dd = 0.60;
            self.filtering.min_profit = 0.0;
            self.filtering.min_trades = 1.0;
            self.filtering.min_sharpe = -5.0;
            self.filtering.min_win_rate = 0.0;
            self.filtering.min_profit_factor = 0.0;
            self.filtering.anomaly_guard = false;
            self.cpcv_min_phi = 0.0;
            self.min_trades_per_day = 0.001;
            // No TF-scaling of trade-frequency floors and NO prop_firm_gate:
            // Risky is judged purely on growth, not challenge-passing.
        }
        self
    }

    fn derive_prop_firm_gate(&self) -> PropFirmGateOverrides {
        // FTMO baseline; the operator overrides individual fields via
        // `models.discovery_runtime.prop_firm_gate`, but a `None`/default
        // keeps the standard challenge rule so the happy-path config needs
        // nothing. (Config-driven replacement for the
        // `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_*` env overrides.)
        let cfg = &self.prop_firm_gate_params;
        let mut rules = PropFirmRiskRules::default();
        rules.min_profit_target_pct =
            PropFirmConstraints::FTMO_STANDARD.challenge_profit_target_pct as f64;
        rules.require_profit_target = true;
        if let Some(v) = cfg.max_daily_loss_pct {
            rules.max_daily_loss_pct = v;
        }
        if let Some(v) = cfg.max_overall_drawdown_pct {
            rules.max_overall_drawdown_pct = v;
        }
        if let Some(v) = cfg.profit_target_pct {
            rules.min_profit_target_pct = v;
            rules.require_profit_target = v > 0.0;
        }
        if let Some(v) = cfg.min_trading_days {
            rules.min_trading_days = v;
        }
        // 60 days = the longest standard prop-firm phase (FTMO Phase 2);
        // a strategy that passes a 60-day window with a 10% target also
        // passes the easier Phase 1 rules at 30 days, so a single
        // measurement covers both.
        let window_days = cfg.window_days.max(1);
        // n_windows is auto-tuned later from dataset length when this stays
        // at its sentinel value (0).
        let n_windows = cfg.n_windows;
        // No hard pass-rate threshold by default — the gate ranks
        // candidates and lets the corr-diversification step pick the
        // top survivors. A non-zero config value still acts as a floor.
        let pass_rate = cfg.pass_rate.clamp(0.0, 1.0);
        PropFirmGateOverrides {
            rules,
            n_windows,
            window_days,
            pass_rate,
        }
    }

    pub fn evaluation_config(&self, price_hint: Option<f64>) -> EvaluationConfig {
        EvaluationConfig::for_symbol(
            &self.evaluation_symbol,
            &self.evaluation_account_currency,
            price_hint,
            Some(self.evaluation_spread_pips),
            Some(self.evaluation_commission_per_trade),
        )
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    pub portfolio: Vec<Gene>,
    pub candidates: Vec<Gene>,
    pub quality_metrics: Vec<StrategyMetrics>,
    pub logged_trades: Vec<LoggedStrategyTrades>,
    /// Feature names as they existed *after* prefiltering inside discovery.
    /// Gene indices refer to columns in this list, not the caller's original names.
    pub effective_feature_names: Vec<String>,
    pub validation_gates: DiscoveryValidationGates,
    pub canonical_backtest_artifacts: Vec<CanonicalBacktestArtifactFile>,
    pub walkforward_validation_artifacts: Vec<WalkforwardValidationArtifactFile>,
    /// Forward-test artifacts produced by replaying the portfolio on a
    /// held-out tail. Empty until the caller invokes
    /// [`compute_discovery_forward_test_artifacts`] with a tail dataset.
    pub forward_test_validation_artifacts: Vec<ForwardTestValidationArtifactFile>,
    /// Prop-firm risk validation artifacts produced by replaying the
    /// portfolio on a held-out tail and applying typed
    /// [`PropFirmRiskRules`]. Empty until the caller invokes
    /// [`compute_discovery_prop_firm_artifacts`] with a tail dataset and
    /// a rule set.
    pub prop_firm_validation_artifacts: Vec<PropFirmRiskValidationArtifactFile>,
    /// 2026-05-26 operator directive (dual-mode product): 16-stage rejection
    /// funnel. Captures count_in / count_out / top_reasons at every filter
    /// boundary so an empty portfolio is debuggable without re-running the
    /// pipeline. Saved as `<symbol>_<tf>_funnel.json` next to the portfolio
    /// JSON by the caller (see `save_portfolio_json` + `funnel_profile`).
    /// `None` only when something panicked early enough that we couldn't
    /// even open the funnel — production callers should treat that as a
    /// bug, not a normal case.
    pub funnel_profile: Option<crate::funnel_profile::FunnelProfile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoggedStrategyTrades {
    pub strategy_id: String,
    pub opportunistic: bool,
    pub trades: Vec<Trade>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryFilterProfile {
    pub max_dd: f64,
    pub min_profit: f64,
    pub min_trades: f64,
    pub min_sharpe: f64,
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
    pub min_positive_months: usize,
    pub min_trades_per_month: f64,
    pub min_monthly_return_pct: f64,
    pub opportunistic_enabled: bool,
    pub opportunistic_min_positive_months: usize,
    pub opportunistic_min_trades_per_month: f64,
    pub opportunistic_min_trade_return_pct: f64,
    pub opportunistic_max_dd: f64,
    pub log_trades: bool,
    pub trade_log_max: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryValidationGates {
    pub walkforward_passed: bool,
    pub cpcv_passed: bool,
    pub canonical_backtest_artifacts: usize,
    pub walkforward_validation_artifacts: usize,
    pub cpcv_fold_count: usize,
    pub cpcv_profitable_fold_ratio: f64,
    pub temporal_contract_hash: Option<String>,
    /// Set when the prop-firm window-pass gate
    /// (`NEOETHOS_BOT_DISCOVERY_PROP_FIRM_GATE=1`) replaces the walkforward
    /// + CPCV consistency gates. Each portfolio member has already passed
    /// FTMO-style rules on at least `pass_rate` of N random 30-day
    /// windows from the dataset; this is what an actual prop-firm
    /// challenge measures, so the much stricter "every walkforward
    /// split must be profitable" requirement is bypassed here.
    pub prop_firm_window_passed: bool,
    pub prop_firm_window_pass_rate: f64,
    pub prop_firm_window_count: usize,
}

impl DiscoveryValidationGates {
    pub fn pending() -> Self {
        Self {
            walkforward_passed: false,
            cpcv_passed: false,
            canonical_backtest_artifacts: 0,
            walkforward_validation_artifacts: 0,
            cpcv_fold_count: 0,
            cpcv_profitable_fold_ratio: 0.0,
            temporal_contract_hash: None,
            prop_firm_window_passed: false,
            prop_firm_window_pass_rate: 0.0,
            prop_firm_window_count: 0,
        }
    }

    pub fn is_portfolio_export_ready(&self) -> bool {
        // Prop-firm window mode is the canonical export path when
        // active — it measures exactly what a challenge measures, so
        // the older walkforward-consistency / CPCV gates do not apply.
        self.prop_firm_window_passed || (self.walkforward_passed && self.cpcv_passed)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryRunProfile {
    pub timeframe_label: String,
    pub population: usize,
    pub generations: usize,
    pub max_indicators: usize,
    pub candidate_count_target: usize,
    pub portfolio_size_target: usize,
    pub max_rows: usize,
    pub max_runtime_hours: f64,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub filters: DiscoveryFilterProfile,
    pub candidates_observed: usize,
    pub portfolio_observed: usize,
    pub quality_metrics_observed: usize,
    pub logged_trade_sets: usize,
    pub walkforward_passed: bool,
    pub cpcv_passed: bool,
    pub canonical_backtest_artifacts_observed: usize,
    pub walkforward_validation_artifacts_observed: usize,
    pub forward_test_validation_artifacts_observed: usize,
    pub prop_firm_validation_artifacts_observed: usize,
    pub cpcv_fold_count: usize,
    pub cpcv_profitable_fold_ratio: f64,
    pub validation_temporal_contract_hash: Option<String>,
    pub prefilter_top_k: usize,
    pub prefilter_insample_frac: f64,
    pub funnel_stage1_pct: f64,
    /// Per-kind validation-evidence hashes ready for the typed
    /// [`neoethos_core::contracts::ValidationEvidenceManifest`]. `None`
    /// per field indicates that artifact kind was not produced for
    /// this run.
    pub validation_evidence_hashes: DiscoveryPerKindEvidenceHashes,
    pub validation_evidence_complete: bool,
    pub validation_evidence_missing_kinds: Vec<String>,
    /// Resolved determinism policy under which the genetic search ran.
    /// `Deterministic { seed }` means the run is reproducible; the two
    /// non-deterministic variants surface in the persisted profile so
    /// `LivePromotionGate::PromotionRejectedDeterminism` failures can
    /// be diagnosed without re-running.
    pub determinism_policy: DeterminismPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryProgress {
    SearchStarted {
        population: usize,
        generations: usize,
        max_indicators: usize,
    },
    GenerationCompleted {
        generation: usize,
        total_generations: usize,
        best_fitness: f64,
        stagnant_generations: usize,
        archived_profitable: usize,
    },
    CandidatesRanked {
        candidate_count: usize,
        truncated_to: usize,
    },
    CandidatesFiltered {
        passed_filters: usize,
        evaluated_candidates: usize,
        min_trades_required: usize,
    },
    QualityScreened {
        strict_passed: usize,
        opportunistic_passed: usize,
        evaluated_candidates: usize,
        logged_trade_sets: usize,
    },
    PortfolioSelected {
        portfolio_size: usize,
        rejected_by_correlation: usize,
        target_portfolio: usize,
    },
    Completed {
        candidate_count: usize,
        filtered_count: usize,
        portfolio_size: usize,
    },
}

pub fn ensure_non_empty_portfolio(result: &DiscoveryResult, context: &str) -> Result<()> {
    if !result.portfolio.is_empty() {
        return Ok(());
    }
    // F-343 (#14): an empty portfolio is the most common — and most
    // confusing — discovery outcome. Instead of a generic "produced an
    // empty portfolio", turn the rejection funnel into an actionable
    // diagnosis: which stage threw everything away, the reasons it gave,
    // and a concrete remedy the operator can act on.
    let diagnosis = result
        .funnel_profile
        .as_ref()
        .map(describe_empty_portfolio_funnel)
        .unwrap_or_else(|| {
            format!(
                "{} candidates were generated but none survived filtering \
                 (no funnel profile was captured — this is a bug; check the logs).",
                result.candidates.len()
            )
        });
    anyhow::bail!("Discovery produced no strategies for {context}. {diagnosis}");
}

/// Turn a rejection [`FunnelProfile`] into a one-paragraph, operator-
/// actionable explanation of WHY the portfolio is empty: the bottleneck
/// stage, the reasons it rejected things, and a concrete remedy.
fn describe_empty_portfolio_funnel(funnel: &crate::funnel_profile::FunnelProfile) -> String {
    // Prefer the funnel's own bottleneck; fall back to the stage that
    // rejected the most among stages that actually received input.
    let bottleneck = if !funnel.bottleneck_stage.is_empty() {
        funnel
            .stages
            .iter()
            .find(|s| s.name == funnel.bottleneck_stage)
    } else {
        None
    }
    .or_else(|| {
        funnel
            .stages
            .iter()
            .filter(|s| s.count_in > 0)
            .max_by_key(|s| s.rejected)
    });

    let Some(stage) = bottleneck else {
        return "The search produced nothing at all — no candidate strategies were \
                generated. Try a longer history window or more generations."
            .to_string();
    };

    let reasons = if stage.top_reasons.is_empty() {
        String::new()
    } else {
        let joined = stage
            .top_reasons
            .iter()
            .take(3)
            .map(|(reason, n)| format!("{reason}×{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" Top reasons: {joined}.")
    };

    format!(
        "Bottleneck: stage '{}' let {} of {} through (rejected {}).{} Hint: {}",
        stage.name,
        stage.count_out,
        stage.count_in,
        stage.rejected,
        reasons,
        remedy_for_stage(&stage.name),
    )
}

/// Map a canonical funnel stage name to a concrete remedy. Stage names
/// are the 16 defined in [`crate::funnel_profile`].
fn remedy_for_stage(stage: &str) -> &'static str {
    match stage {
        "data_loaded" | "rows_after_trimming" => {
            "not enough history — fetch more bars (Settings → Data) or pick a higher timeframe."
        }
        "features_built" | "features_after_prefilter" => {
            "the feature prefilter removed everything — widen the indicator set or check the \
             imported data for gaps."
        }
        "stage1_candidates_generated" | "profitable_archive_size" => {
            "the genetic search found no profitable seeds — raise population / generations, or \
             allow more indicators per strategy."
        }
        "full_is_evaluated" | "passed_base_filter" => {
            "every candidate failed the base filter — relax max-drawdown / min-profit in the \
             discovery filters."
        }
        "nonzero_signals" => {
            "strategies generated zero trades — relax entry thresholds or verify indicator \
             warm-up has enough bars."
        }
        "passed_min_trades" => {
            "candidates traded too rarely — lower the min-trades requirement or use a longer \
             window."
        }
        "passed_quality" => {
            "the quality screen rejected all survivors — lower the min Sharpe / win-rate / \
             profit-factor, or enable opportunistic mode."
        }
        "passed_prop_firm_window" => {
            "nothing passed the prop-firm window gate — loosen the FTMO rule set, or switch off \
             the prop-firm gate if you're not targeting a challenge."
        }
        "passed_correlation" => {
            "survivors were too correlated with each other — raise the correlation threshold to \
             admit more of them."
        }
        "passed_walkforward" => {
            "strategies didn't hold up out-of-sample (walk-forward) — widen the search or reduce \
             the number of walk-forward splits."
        }
        "passed_cpcv" => {
            "strategies failed CPCV cross-validation — lower the CPCV min-phi tolerance or disable \
             CPCV for this run."
        }
        "export_ready" => {
            "candidates passed every gate but failed final export-readiness — check the \
             validation-gate configuration."
        }
        _ => "review the saved funnel JSON (cache/discovery/<symbol>_<tf>.json) for the full \
              stage-by-stage breakdown.",
    }
}

fn row_cap_for_config(config: &DiscoveryConfig) -> usize {
    let tf_cap = config
        .max_rows_by_timeframe
        .get(&config.timeframe_label)
        .copied()
        .unwrap_or(0);
    match (config.max_rows, tf_cap) {
        (0, 0) => 0,
        (0, tf) => tf,
        (global, 0) => global,
        (global, tf) => global.min(tf),
    }
}

fn trim_recent_history(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<(FeatureFrame, Ohlcv, Option<usize>)> {
    let frame_rows = features.data.nrows();
    let ohlcv_rows = ohlcv.close.len();
    let available_rows = frame_rows.min(ohlcv_rows);
    if available_rows == 0 {
        anyhow::bail!(
            "Cannot run discovery on empty history for {} {} — \
             import at least the minimum bars (run `neoethos-cli import`) then retry.",
            config.evaluation_symbol, config.timeframe_label
        );
    }

    let mut start_idx = 0usize;
    let row_cap = row_cap_for_config(config);
    if row_cap > 0 && row_cap < available_rows {
        start_idx = available_rows - row_cap;
    }

    let trimmed_rows = available_rows.saturating_sub(start_idx);
    let row_budget_applied = if start_idx > 0 {
        Some(trimmed_rows)
    } else {
        None
    };

    let trimmed_features = FeatureFrame {
        timestamps: features.timestamps[start_idx..available_rows].to_vec(),
        names: features.names.clone(),
        data: features
            .data
            .slice(ndarray::s![start_idx..available_rows, ..])
            .to_owned(),
    };
    let trimmed_ohlcv = slice_ohlcv(ohlcv, start_idx, available_rows);
    Ok((trimmed_features, trimmed_ohlcv, row_budget_applied))
}

fn slice_ohlcv(ohlcv: &Ohlcv, start_idx: usize, end_idx: usize) -> Ohlcv {
    neoethos_data::slice_ohlcv(ohlcv, start_idx, end_idx, None)
}

fn quality_analyzer_for_config(config: &DiscoveryConfig) -> StrategyQualityAnalyzer {
    StrategyQualityAnalyzer {
        min_sharpe: config.filtering.min_sharpe.max(0.0),
        min_sortino: config.filtering.min_sharpe.max(0.0),
        min_calmar: 0.0,
        min_profit_factor: config.filtering.min_profit_factor.max(0.0),
        min_win_rate: config.filtering.min_win_rate.clamp(0.0, 1.0),
        min_trades: config.filtering.min_trades.max(0.0) as usize,
        max_dd_acceptable: config.filtering.max_dd.max(0.0),
        min_monthly_return_pct: config.filtering.min_monthly_return_pct.max(0.0),
        edge_significance_pvalue: 0.05,
        // 2026-05-26 operator directive (dual-mode product): the Settings
        // path (FilteringConfig::min_trades_per_month from
        // prop_search_val_min_trades_per_month) is the canonical source.
        // Setting `Some(...)` here makes the analyzer ignore the env-driven
        // QualityRuntimeOverrides default for this run — exactly one
        // threshold drives the monthly consistency gate.
        min_trades_per_month: Some(config.filtering.min_trades_per_month.max(0.0) as usize),
    }
}

fn discovery_backtest_settings(
    config: &DiscoveryConfig,
    gene: &Gene,
    price_hint: Option<f64>,
) -> crate::eval::BacktestSettings {
    let evaluation = config.evaluation_config(price_hint);
    crate::eval::BacktestSettings {
        sl_pips: if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
            gene.sl_pips
        } else {
            20.0
        },
        tp_pips: if gene.tp_pips.is_finite() && gene.tp_pips > 0.0 {
            gene.tp_pips
        } else {
            40.0
        },
        max_hold_bars: evaluation.max_hold_bars,
        trailing_enabled: evaluation.trailing_enabled,
        trailing_atr_multiplier: evaluation.trailing_atr_multiplier,
        trailing_be_trigger_r: evaluation.trailing_be_trigger_r,
        pip_value: evaluation.pip_value,
        spread_pips: evaluation.spread_pips,
        commission_per_trade: evaluation.commission_per_trade,
        pip_value_per_lot: evaluation.pip_value_per_lot,
        kill_zones_enabled: true,
        ..crate::eval::BacktestSettings::default()
    }
}

/// F-305 (2026-05-28): scale `min_trades_per_month` proportionally to
/// timeframe bar density so the operator's `config.yaml` value
/// (typically 15 trades/month, tuned for M1/M5/M15 intra-day flow)
/// doesn't mechanically reject every D1/H4 candidate that trades at
/// a sensible-for-the-TF cadence.
///
/// Bar count per calendar month roughly:
///   M1:    ~30_240   (24 × 60 × 21 trading days)
///   M5:    ~6_048
///   M15:   ~2_016
///   M30:   ~1_008
///   H1:    ~504
///   H4:    ~126
///   D1:    ~21
///   W1:    ~4.3
///   MN1:   ~1
///
/// For the operator's default 15 trades/month on M1/M5/M15, that's
/// ~0.05% of bars — completely reasonable. On D1 with only 21 bars,
/// 15 trades means trading 70%+ of bars (mechanically impossible for
/// any signal with non-trivial selectivity). The scale below targets
/// roughly "~5-10% of bars must trade" on the longer TFs.
fn min_trades_per_month_scale_for_tf(tf: &str) -> f64 {
    match tf.to_ascii_uppercase().as_str() {
        // Intra-day TFs keep operator's value as-is — they have
        // thousands of bars per month, 15-50 trades is a small
        // fraction of total bar count.
        "M1" | "M3" | "M5" | "M15" => 1.0,
        // Half-hour: still plenty of bars (~1000/month), small relax
        "M30" => 0.67,
        // Hourly: ~500 bars/month, 6 trades = ~1.2% of bars
        "H1" => 0.40,
        // 4h: ~126 bars/month, 3 trades = ~2.4% of bars
        "H4" => 0.20,
        // Daily: ~21 bars/month, 2 trades = ~10% of bars (one swing
        // trade every ~2 weeks is realistic for prop-firm passing)
        "D1" => 0.13,
        // Weekly/monthly: very long-horizon, ANY signal qualifies
        "W1" => 0.04,
        "MN1" => 0.02,
        // Unknown TF: be conservative, keep operator's value
        _ => 1.0,
    }
}

fn passes_strict_quality(metrics: &StrategyMetrics, cfg: &crate::genetic::FilteringConfig) -> bool {
    if cfg.min_positive_months > 0 && metrics.positive_months < cfg.min_positive_months {
        return false;
    }
    if cfg.min_trades_per_month > 0.0 && metrics.trades_per_month < cfg.min_trades_per_month {
        return false;
    }
    if cfg.min_monthly_return_pct > 0.0
        && metrics.avg_monthly_return_pct < cfg.min_monthly_return_pct
    {
        return false;
    }
    true
}

fn passes_opportunistic_quality(
    metrics: &StrategyMetrics,
    cfg: &crate::genetic::FilteringConfig,
) -> bool {
    if !cfg.opportunistic_enabled || !cfg.use_opportunistic_candidates {
        return false;
    }
    if cfg.opportunistic_min_positive_months > 0
        && metrics.positive_months < cfg.opportunistic_min_positive_months
    {
        return false;
    }
    if cfg.opportunistic_min_trades_per_month > 0.0
        && metrics.trades_per_month < cfg.opportunistic_min_trades_per_month
    {
        return false;
    }
    let avg_trade_return_pct = metrics.avg_win_pct.abs() * 100.0;
    if cfg.opportunistic_min_trade_return_pct > 0.0
        && avg_trade_return_pct < cfg.opportunistic_min_trade_return_pct
    {
        return false;
    }
    if cfg.opportunistic_max_dd > 0.0 && metrics.max_drawdown_pct > cfg.opportunistic_max_dd {
        return false;
    }
    true
}

#[derive(Debug, Serialize)]
struct DiscoveryDatasetFingerprint<'a> {
    row_count: usize,
    first_timestamp: Option<i64>,
    last_timestamp: Option<i64>,
    feature_names: &'a [String],
    close_rows: usize,
    first_close: Option<f64>,
    last_close: Option<f64>,
}

#[derive(Debug, Serialize)]
struct DiscoveryTemporalPolicy<'a> {
    timeframe_label: &'a str,
    higher_timeframes: &'a [String],
    feature_names: &'a [String],
}

#[derive(Debug, Serialize)]
struct DiscoveryWalkforwardPolicy {
    train_ratio: f64,
    walkforward_splits: usize,
    embargo_minutes: usize,
    enable_cpcv: bool,
    cpcv_n_splits: usize,
    cpcv_n_test_groups: usize,
    cpcv_embargo_pct: f64,
    cpcv_purge_pct: f64,
    cpcv_min_phi: f64,
}

#[derive(Debug, Serialize)]
struct DiscoveryLiveReadinessPolicy {
    portfolio_size_target: usize,
    max_regime_loss_pct: f64,
    filtering: crate::genetic::FilteringConfig,
}

#[derive(Debug, Serialize)]
struct DiscoveryBacktestPolicy {
    symbol: String,
    account_currency: String,
    timeframe_label: String,
    sl_pips: f64,
    tp_pips: f64,
    max_hold_bars: usize,
    min_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    kill_zones_enabled: bool,
}

fn discovery_temporal_contract(
    config: &DiscoveryConfig,
    feature_names: &[String],
) -> Result<TemporalFeatureContract> {
    let feature_policy_hash = stable_json_hash(&DiscoveryTemporalPolicy {
        timeframe_label: &config.timeframe_label,
        higher_timeframes: &config.higher_timeframes,
        feature_names,
    })?;
    let label_policy_hash = stable_json_hash(&(
        "strategy-search-signal-v1",
        "prior-bar-signal-next-bar-fill",
        &config.timeframe_label,
    ))?;
    let walk_forward_policy_hash = stable_json_hash(&DiscoveryWalkforwardPolicy {
        train_ratio: 0.70,
        walkforward_splits: config.walkforward_splits,
        embargo_minutes: config.embargo_minutes,
        enable_cpcv: config.enable_cpcv,
        cpcv_n_splits: config.cpcv_n_splits,
        cpcv_n_test_groups: config.cpcv_n_test_groups,
        cpcv_embargo_pct: config.cpcv_embargo_pct,
        cpcv_purge_pct: config.cpcv_purge_pct,
        cpcv_min_phi: config.cpcv_min_phi,
    })?;
    let live_readiness_policy_hash = stable_json_hash(&DiscoveryLiveReadinessPolicy {
        portfolio_size_target: config.portfolio_size,
        max_regime_loss_pct: config.max_regime_loss_pct,
        filtering: config.filtering,
    })?;

    Ok(TemporalFeatureContract::strict_live(
        "UTC",
        feature_policy_hash,
        label_policy_hash,
        walk_forward_policy_hash,
        live_readiness_policy_hash,
    )?)
}

fn validation_row_count(features: &FeatureFrame, ohlcv: &Ohlcv) -> Result<usize> {
    let n = features.data.nrows();
    if n == 0
        || features.timestamps.len() != n
        || ohlcv.close.len() != n
        || ohlcv.high.len() != n
        || ohlcv.low.len() != n
    {
        anyhow::bail!(
            "discovery validation requires aligned non-empty features/OHLCV rows (features={}, timestamps={}, close={}, high={}, low={})",
            n,
            features.timestamps.len(),
            ohlcv.close.len(),
            ohlcv.high.len(),
            ohlcv.low.len()
        );
    }
    Ok(n)
}

fn discovery_dataset_hash(features: &FeatureFrame, ohlcv: &Ohlcv) -> Result<String> {
    stable_json_hash(&DiscoveryDatasetFingerprint {
        row_count: features.data.nrows(),
        first_timestamp: features.timestamps.first().copied(),
        last_timestamp: features.timestamps.last().copied(),
        feature_names: &features.names,
        close_rows: ohlcv.close.len(),
        first_close: ohlcv.close.first().copied(),
        last_close: ohlcv.close.last().copied(),
    })
}

fn discovery_backtest_policy_hash(
    config: &DiscoveryConfig,
    gene: &Gene,
    settings: &crate::eval::BacktestSettings,
) -> Result<String> {
    stable_json_hash(&DiscoveryBacktestPolicy {
        symbol: config.evaluation_symbol.clone(),
        account_currency: config.evaluation_account_currency.clone(),
        timeframe_label: config.timeframe_label.clone(),
        sl_pips: settings.sl_pips,
        tp_pips: settings.tp_pips,
        max_hold_bars: settings.max_hold_bars,
        min_hold_bars: settings.min_hold_bars,
        trailing_enabled: settings.trailing_enabled,
        trailing_atr_multiplier: settings.trailing_atr_multiplier,
        trailing_be_trigger_r: settings.trailing_be_trigger_r,
        pip_value: settings.pip_value,
        spread_pips: settings.spread_pips,
        commission_per_trade: settings.commission_per_trade,
        pip_value_per_lot: settings.pip_value_per_lot,
        kill_zones_enabled: settings.kill_zones_enabled,
    })
    .with_context(|| format!("hashing backtest policy for {}", gene.strategy_id))
}

fn embargo_bars_from_timestamps(timestamps: &[i64], embargo_minutes: usize) -> usize {
    if embargo_minutes == 0 || timestamps.len() < 2 {
        return 0;
    }
    let step_ms = timestamps
        .windows(2)
        .filter_map(|window| {
            let step = window[1].saturating_sub(window[0]);
            (step > 0).then_some(step)
        })
        .min()
        .unwrap_or(60_000);
    let embargo_ms = (embargo_minutes as i64).saturating_mul(60_000);
    ((embargo_ms + step_ms - 1) / step_ms).max(0) as usize
}

fn walkforward_summary_passed(summary: &WalkforwardSummary) -> bool {
    summary.walk_forward_splits > 0
        && summary.avg_pnl > 0.0
        && !summary.any_daily_loss_breach
        && !summary.any_consistency_violation
        && !summary.any_trade_limit_violation
        && summary.all_min_trading_days_ok
}

fn evaluate_cpcv_gate(
    portfolio: &[Gene],
    portfolio_signals: &[Vec<i8>],
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    months: &[i64],
    days: &[i64],
) -> Result<(bool, usize, f64)> {
    if portfolio.is_empty() {
        return Ok((false, 0, 0.0));
    }
    // **F-018 documentation (2026-05-25)** — when CPCV is operator-
    // disabled via `enable_cpcv = false`, this gate returns
    // `(true, 0, 1.0)` so the discovery cycle continues. The original
    // audit flagged this as "passes without running CPCV" — which is
    // CORRECT: a disabled gate cannot fail. The fold_count of `0`
    // surfaces in the run-profile so operators see "CPCV: disabled
    // (0 folds)". Production prop-firm runs MUST keep CPCV enabled
    // — the disable flag is only honoured for test fixtures /
    // research-mode quick checks. Tracked by the upstream Settings-
    // exposed `discovery.enable_cpcv` knob in `config.yaml`.
    if !config.enable_cpcv {
        tracing::warn!(
            target: "neoethos_search::discovery",
            "CPCV gate is DISABLED via config.enable_cpcv=false — \
             portfolio promoted without out-of-sample validation. \
             For prop-firm production runs, set enable_cpcv=true."
        );
        return Ok((true, 0, 1.0));
    }

    let n = ohlcv.close.len();
    let capped_n = if config.cpcv_max_rows > 0 {
        config.cpcv_max_rows.min(n)
    } else {
        n
    };
    let offset = n.saturating_sub(capped_n);
    let cv = CombinatorialPurgedCV::new(
        config.cpcv_n_splits,
        config.cpcv_n_test_groups,
        config.cpcv_embargo_pct,
        config.cpcv_purge_pct,
    );
    let splits = cv.split(capped_n);
    if splits.is_empty() {
        return Ok((false, 0, 0.0));
    }

    let mut fold_count = 0usize;
    let mut profitable_folds = 0usize;
    for (gene, signals) in portfolio.iter().zip(portfolio_signals) {
        let settings = discovery_backtest_settings(config, gene, ohlcv.close.last().copied());
        for (_, test_idx) in &splits {
            if test_idx.is_empty() {
                continue;
            }
            let absolute_idx: Vec<usize> = test_idx.iter().map(|idx| offset + *idx).collect();
            let close: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.close[*idx]).collect();
            let high: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.high[*idx]).collect();
            let low: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.low[*idx]).collect();
            let sig: Vec<i8> = absolute_idx.iter().map(|idx| signals[*idx]).collect();
            let fold_months: Vec<i64> = absolute_idx.iter().map(|idx| months[*idx]).collect();
            let fold_days: Vec<i64> = absolute_idx.iter().map(|idx| days[*idx]).collect();
            let metrics = BacktestMetrics::from_metric_array(fast_evaluate_strategy_core(
                &close,
                &high,
                &low,
                &sig,
                &fold_months,
                &fold_days,
                &[],
                &settings,
            ));
            fold_count += 1;
            let drawdown_ok =
                config.filtering.max_dd <= 0.0 || metrics.max_drawdown <= config.filtering.max_dd;
            if metrics.trade_count > 0 && metrics.net_profit > 0.0 && drawdown_ok {
                profitable_folds += 1;
            }
        }
    }

    if fold_count == 0 {
        return Ok((false, 0, 0.0));
    }
    let ratio = profitable_folds as f64 / fold_count as f64;
    Ok((
        ratio >= config.cpcv_min_phi.clamp(0.0, 1.0),
        fold_count,
        ratio,
    ))
}

fn build_discovery_validation_artifacts(
    portfolio: &[Gene],
    portfolio_signals: &[Vec<i8>],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<(
    DiscoveryValidationGates,
    Vec<CanonicalBacktestArtifactFile>,
    Vec<WalkforwardValidationArtifactFile>,
)> {
    if portfolio.is_empty() {
        return Ok((DiscoveryValidationGates::pending(), Vec::new(), Vec::new()));
    }
    let n = validation_row_count(features, ohlcv)?;
    if portfolio_signals.iter().any(|signals| signals.len() != n) {
        let mismatched = portfolio_signals
            .iter()
            .enumerate()
            .find(|(_, s)| s.len() != n)
            .map(|(i, s)| format!("signals[{}].len()={}", i, s.len()))
            .unwrap_or_default();
        anyhow::bail!(
            "Internal bug: discovery validation requires portfolio signals aligned to feature rows \
             (expected {} rows, {}). Please report this with config.yaml and the discovery log.",
            n, mismatched
        );
    }

    let temporal_contract = discovery_temporal_contract(config, &features.names)?;
    let temporal_contract_hash = temporal_contract.temporal_contract_hash();
    let dataset_hash = discovery_dataset_hash(features, ohlcv)?;
    let (months, days) = month_day_indices(&features.timestamps);
    let timestamps = &features.timestamps[..n];
    let embargo_bars = embargo_bars_from_timestamps(timestamps, config.embargo_minutes);

    let mut canonical_backtest_artifacts = Vec::with_capacity(portfolio.len());
    let mut walkforward_validation_artifacts = Vec::with_capacity(portfolio.len());
    let mut walkforward_passed = true;

    for (gene, signals) in portfolio.iter().zip(portfolio_signals) {
        let settings = discovery_backtest_settings(config, gene, ohlcv.close.last().copied());
        let strategy_hash = stable_json_hash(gene)?;
        let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
        let metrics = BacktestMetrics::from_metric_array(fast_evaluate_strategy_core(
            &ohlcv.close,
            &ohlcv.high,
            &ohlcv.low,
            signals,
            &months,
            &days,
            timestamps,
            &settings,
        ));
        canonical_backtest_artifacts.push(CanonicalBacktestArtifactFile::new(
            CanonicalBacktestScope::new(
                dataset_hash.clone(),
                evaluation_config_hash.clone(),
                strategy_hash.clone(),
                &temporal_contract,
            ),
            metrics,
        ));

        let walkforward_summary = embargoed_walkforward_backtest(WalkforwardBacktestInput {
            close: &ohlcv.close,
            high: &ohlcv.high,
            low: &ohlcv.low,
            signals,
            months: &months,
            days: &days,
            timestamps,
            train_ratio: 0.70,
            n_splits: config.walkforward_splits.max(1),
            embargo_bars,
            settings: &settings,
            max_daily_loss_pct: config.max_regime_loss_pct,
            max_daily_profit_pct: 0.0,
            min_trading_days: 0,
            max_trades_per_day: 0,
            initial_balance: config.initial_balance,
        })?;
        walkforward_passed &= walkforward_summary_passed(&walkforward_summary);
        walkforward_validation_artifacts.push(WalkforwardValidationArtifactFile::new(
            WalkforwardValidationScope::for_strategy(
                dataset_hash.clone(),
                evaluation_config_hash,
                strategy_hash,
                &temporal_contract,
            ),
            walkforward_summary,
        ));
    }

    let (cpcv_passed, cpcv_fold_count, cpcv_profitable_fold_ratio) =
        evaluate_cpcv_gate(portfolio, portfolio_signals, ohlcv, config, &months, &days)?;

    let validation_gates = DiscoveryValidationGates {
        walkforward_passed,
        cpcv_passed,
        canonical_backtest_artifacts: canonical_backtest_artifacts.len(),
        walkforward_validation_artifacts: walkforward_validation_artifacts.len(),
        cpcv_fold_count,
        cpcv_profitable_fold_ratio,
        temporal_contract_hash: Some(temporal_contract_hash),
        prop_firm_window_passed: false,
        prop_firm_window_pass_rate: 0.0,
        prop_firm_window_count: 0,
    };

    Ok((
        validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
    ))
}

/// Replay each portfolio gene on a held-out tail window and produce one
/// [`ForwardTestValidationArtifactFile`] per strategy. The caller passes
/// the *raw* tail (with the same `feature_names` ordering it had before
/// discovery) and `effective_feature_names` produced by discovery; the
/// helper aligns the tail's columns to the post-prefilter set so the
/// gene indices match.
///
/// Returns `Err` when any name in `effective_feature_names` is missing
/// from the tail's columns — this indicates the tail comes from a
/// different feature pipeline than the discovery run that produced the
/// portfolio, and a forward-test on it would be meaningless.
pub fn compute_discovery_forward_test_artifacts(
    portfolio: &[Gene],
    effective_feature_names: &[String],
    tail_features: &FeatureFrame,
    tail_ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<Vec<ForwardTestValidationArtifactFile>> {
    if portfolio.is_empty() {
        return Ok(Vec::new());
    }

    // Project the tail's columns onto the post-prefilter set used by the
    // portfolio. When the tail already matches, this is a cheap clone of
    // the underlying ndarray; when it does not, we slice column-by-column.
    let tail_features = if tail_features.names == effective_feature_names {
        std::borrow::Cow::Borrowed(tail_features)
    } else {
        let mut keep_indices = Vec::with_capacity(effective_feature_names.len());
        for name in effective_feature_names {
            let idx = tail_features
                .names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "forward-test tail is missing feature '{}' from the discovery effective \
                         feature set; the tail must come from the same feature pipeline as the \
                         in-sample discovery run",
                        name
                    )
                })?;
            keep_indices.push(idx);
        }
        let n_rows = tail_features.data.nrows();
        let mut projected = ndarray::Array2::<f32>::zeros((n_rows, keep_indices.len()));
        for (new_idx, &orig_idx) in keep_indices.iter().enumerate() {
            projected
                .column_mut(new_idx)
                .assign(&tail_features.data.column(orig_idx));
        }
        std::borrow::Cow::Owned(FeatureFrame {
            timestamps: tail_features.timestamps.clone(),
            names: effective_feature_names.to_vec(),
            data: projected,
        })
    };
    let tail_features = tail_features.as_ref();

    let n = validation_row_count(tail_features, tail_ohlcv)?;
    if n == 0 {
        anyhow::bail!("forward-test tail must contain at least one bar");
    }

    let temporal_contract = discovery_temporal_contract(config, &tail_features.names)?;
    let tail_dataset_hash = discovery_dataset_hash(tail_features, tail_ohlcv)?;
    let (months, days) = month_day_indices(&tail_features.timestamps);
    let timestamps = &tail_features.timestamps[..n];

    // **F-315 (2026-05-29) — partial diagnostic, full fix deferred to F-315b**.
    //
    // The stage-1 GA in `search_engine.rs:818` anneals the SMC gate
    // from `gate_start` (default 0.75) down to `gate_end` (default
    // 0.35) across generations. The forward-test pass below
    // re-evaluates each survivor with the STATIC
    // `runtime.smc_weights.gate_threshold` from
    // `current_strategy_evaluation_runtime_overrides()` (default 0.75
    // per `runtime_overrides.rs:417`). Candidates that survived the
    // last generation under e.g. 0.35 may fail forward-test with 0.75
    // — the asymmetry the F-315 ticket flagged. The proper fix is to
    // forward the stage-1 final gate value through `SearchResult` and
    // override the forward-test runtime gate to match; that touches
    // 7 SearchResult construction sites in search_engine.rs plus the
    // discovery + forward-test plumbing. **F-315b** tracks the
    // architectural follow-up. For this ticket, we emit a warn at
    // the top of `current_strategy_evaluation_runtime_overrides()`
    // when the operator's static gate exceeds 0.5 (i.e. clearly above
    // the GA's typical `gate_end` of 0.35) so the asymmetry surfaces
    // in the discovery log instead of staying invisible.
    {
        use crate::genetic::runtime_overrides::current_strategy_evaluation_runtime_overrides;
        let static_gate = current_strategy_evaluation_runtime_overrides()
            .smc_weights
            .gate_threshold;
        if static_gate > 0.5 {
            tracing::warn!(
                target: "neoethos_search::discovery",
                forward_test_smc_gate = static_gate,
                "F-315 mismatch: forward-test SMC gate ({static_gate:.2}) is well above the GA stage-1 typical end (0.35). Survivors of the final generation may fail forward-test for a threshold they never passed in stage-1. Lower the runtime override (`smc_weights.gate_threshold`) toward 0.35 to align, or wait for F-315b to plumb the GA's final gate through SearchResult."
            );
        }
    }

    let mut artifacts = Vec::with_capacity(portfolio.len());
    for gene in portfolio {
        let settings = discovery_backtest_settings(config, gene, tail_ohlcv.close.last().copied());
        let strategy_hash = stable_json_hash(gene)?;
        let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
        let evaluation_config = config.evaluation_config(tail_ohlcv.close.last().copied());
        let signals = signals_for_gene_full(tail_features, tail_ohlcv, gene, &evaluation_config);
        if signals.len() != n {
            anyhow::bail!(
                "forward-test signals length {} does not match validation row count {}",
                signals.len(),
                n
            );
        }
        let summary = compute_forward_test_summary(ForwardTestInput {
            close: &tail_ohlcv.close[..n],
            high: &tail_ohlcv.high[..n],
            low: &tail_ohlcv.low[..n],
            signals: &signals[..n],
            months: &months[..n],
            days: &days[..n],
            timestamps,
            settings: &settings,
        })?;
        artifacts.push(ForwardTestValidationArtifactFile::new(
            ForwardTestValidationScope::new(
                tail_dataset_hash.clone(),
                evaluation_config_hash,
                strategy_hash,
                &temporal_contract,
            ),
            summary,
        ));
    }
    Ok(artifacts)
}

/// Replay each portfolio gene on a held-out tail window, simulate trades
/// under the canonical backtest core, and aggregate them through
/// [`compute_prop_firm_risk_summary`] to produce one
/// [`PropFirmRiskValidationArtifactFile`] per strategy. The signature
/// mirrors [`compute_discovery_forward_test_artifacts`]: the caller
/// passes the tail with its original `feature_names` ordering, and the
/// helper aligns it to `effective_feature_names` before running the
/// simulation.
///
/// Returns `Err` when the tail is missing any effective feature, when
/// the tail is empty, or when the simulator produces a signal vector of
/// the wrong length — each path indicates the tail comes from a
/// different feature pipeline than the discovery run that produced the
/// portfolio.
pub fn compute_discovery_prop_firm_artifacts(
    portfolio: &[Gene],
    effective_feature_names: &[String],
    tail_features: &FeatureFrame,
    tail_ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    rules: PropFirmRiskRules,
) -> Result<Vec<PropFirmRiskValidationArtifactFile>> {
    if portfolio.is_empty() {
        return Ok(Vec::new());
    }

    let tail_features = if tail_features.names == effective_feature_names {
        std::borrow::Cow::Borrowed(tail_features)
    } else {
        let mut keep_indices = Vec::with_capacity(effective_feature_names.len());
        for name in effective_feature_names {
            let idx = tail_features
                .names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "prop-firm tail is missing feature '{}' from the discovery effective \
                         feature set; the tail must come from the same feature pipeline as the \
                         in-sample discovery run",
                        name
                    )
                })?;
            keep_indices.push(idx);
        }
        let n_rows = tail_features.data.nrows();
        let mut projected = ndarray::Array2::<f32>::zeros((n_rows, keep_indices.len()));
        for (new_idx, &orig_idx) in keep_indices.iter().enumerate() {
            projected
                .column_mut(new_idx)
                .assign(&tail_features.data.column(orig_idx));
        }
        std::borrow::Cow::Owned(FeatureFrame {
            timestamps: tail_features.timestamps.clone(),
            names: effective_feature_names.to_vec(),
            data: projected,
        })
    };
    let tail_features = tail_features.as_ref();

    let n = validation_row_count(tail_features, tail_ohlcv)?;
    if n == 0 {
        anyhow::bail!("prop-firm tail must contain at least one bar");
    }
    let temporal_contract = discovery_temporal_contract(config, &tail_features.names)?;
    let tail_dataset_hash = discovery_dataset_hash(tail_features, tail_ohlcv)?;
    let timestamps = &tail_features.timestamps[..n];

    let mut artifacts = Vec::with_capacity(portfolio.len());
    for gene in portfolio {
        let settings = discovery_backtest_settings(config, gene, tail_ohlcv.close.last().copied());
        let strategy_hash = stable_json_hash(gene)?;
        let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
        let evaluation_config = config.evaluation_config(tail_ohlcv.close.last().copied());
        let signals = signals_for_gene_full(tail_features, tail_ohlcv, gene, &evaluation_config);
        if signals.len() != n {
            anyhow::bail!(
                "prop-firm signals length {} does not match validation row count {}",
                signals.len(),
                n
            );
        }
        let trades = simulate_trades_core(
            &tail_ohlcv.close[..n],
            &tail_ohlcv.high[..n],
            &tail_ohlcv.low[..n],
            timestamps,
            &signals[..n],
            &settings,
        );
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &trades,
            initial_balance: config.initial_balance,
            rules,
        });
        let scope = PropFirmRiskValidationScope::new(
            tail_dataset_hash.clone(),
            evaluation_config_hash,
            strategy_hash,
            &rules,
            &temporal_contract,
        )?;
        artifacts.push(PropFirmRiskValidationArtifactFile::new(scope, summary));
    }
    Ok(artifacts)
}

#[derive(Debug, Serialize)]
struct GeneExport<'a> {
    strategy_id: &'a str,
    indicators: Vec<&'a str>,
    indices: Vec<usize>,
    weights: Vec<f32>,
    long_threshold: f32,
    short_threshold: f32,
    fitness: f64,
    sharpe_ratio: f64,
    win_rate: f64,
    tp_pips: f64,
    sl_pips: f64,
}

/// **F-096 fix (2026-05-25)** — minimum-history pre-flight check.
///
/// Operator real-data directive 2026-05-24: discovery / training /
/// validation MUST refuse to run when fewer than ~10 years of bars
/// are available per symbol. The exact bar count threshold is
/// timeframe-dependent (10 years × bars-per-year for the given TF),
/// so we approximate by `min_bars = years × bars_per_year(tf)` with
/// a conservative 220 trading days/year × 24 hours/day for M1, etc.
///
/// Returns `Ok(())` when the OHLCV has enough rows; returns
/// `Err(anyhow!(...))` with the symbol name + actual coverage + the
/// remediation path (user-imported OR auto-fetch from cTrader) when
/// it doesn't. The caller (CLI, server, wizard) decides whether to
/// auto-fetch and re-run, or bail to the operator.
///
/// `min_history_years` defaults to **0** (use whatever data exists, ratio-
/// split via `prop_search_val_years` downstream — see operator directive
/// 2026-05-26 in `DiscoveryRuntimeOverrides::default`). Set to a positive
/// integer either via `NEOETHOS_BOT_MIN_HISTORY_YEARS` env or explicitly in
/// the override struct to re-instate a hard floor.
pub fn ensure_sufficient_history(
    ohlcv: &Ohlcv,
    symbol: &str,
    timeframe: &str,
    min_history_years: u32,
) -> Result<()> {
    if min_history_years == 0 {
        // Caller explicitly opted out (test / demo path).
        return Ok(());
    }
    let bars_per_year = approx_bars_per_year(timeframe);
    let required_bars = (min_history_years as usize).saturating_mul(bars_per_year);
    let actual_bars = ohlcv.close.len();
    if actual_bars < required_bars {
        anyhow::bail!(
            "Insufficient history for {symbol} {timeframe}: have {actual_bars} bars, \
             need at least {required_bars} (≈ {min_history_years} years × {bars_per_year} \
             bars/yr). Remediation: (1) Settings → Data → 'Download history from broker' \
             with a ~{min_history_years}-year window for {symbol} {timeframe}, then re-run \
             Discovery; OR (2) relax the floor via the NEOETHOS_BOT_MIN_HISTORY_YEARS \
             environment variable — set it to 0 to run on whatever data exists \
             (accepts the over-fitting risk). Operator policy 2026-05-24: refuse \
             synthetic / insufficient data."
        );
    }
    Ok(())
}

/// Approximate bars-per-year for a canonical timeframe label. Uses a
/// conservative 220 trading-day year (FX market). Returns 0 for
/// unknown timeframes — the caller's `saturating_mul` will then make
/// `required_bars = 0` so the check effectively skips for non-canonical
/// inputs (which should already have been rejected upstream).
pub fn approx_bars_per_year(tf: &str) -> usize {
    // 220 trading days × hours × bars-per-hour, conservatively. The
    // FX market is 24/5 but we use 220 days × 24 hours instead of
    // 252 × 24 to leave headroom for holiday gaps. For weekly /
    // monthly timeframes we count calendar weeks / months.
    match tf.trim().to_ascii_uppercase().as_str() {
        "M1" => 220 * 24 * 60,
        "M3" => 220 * 24 * 20,
        "M5" => 220 * 24 * 12,
        "M15" => 220 * 24 * 4,
        "M30" => 220 * 24 * 2,
        "H1" => 220 * 24,
        "H4" => 220 * 6,
        "H12" => 220 * 2,
        "D1" => 220,
        "W1" => 52,
        "MN1" => 12,
        _ => 0,
    }
}

pub fn run_discovery_cycle(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<DiscoveryResult> {
    run_discovery_cycle_with_progress(features, ohlcv, config, |_| {})
}

pub fn run_discovery_cycle_with_progress<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    // F-304 fix (2026-05-28): pre-flight bail. The cost-model NaN
    // guard at `strategy_gene::infer_market_cost_profile` returns
    // empty-string + NaN-sentinel values when `evaluation_symbol` or
    // `evaluation_account_currency` are blank. Those NaN values then
    // propagate through `pip = settings.pip_value` (only near-zero is
    // checked, not NaN) → spread_pips * pip = NaN entry_px, no trades
    // open, sanitizer scrubs metrics to 0.0, GA sees a zero-trade
    // candidate. Operator gets "no trades found" with no explanation.
    //
    // Bail loud here BEFORE the FunnelProfile/GA spin up so the error
    // message points at the right config field instead of a downstream
    // silent-failure metric.
    if config.evaluation_symbol.trim().is_empty() {
        anyhow::bail!(
            "run_discovery_cycle: DiscoveryConfig.evaluation_symbol is empty. \
             Set it explicitly before calling — the cost-model NaN guard \
             would otherwise produce zero-trade candidates with no clear \
             failure signal. Bind the symbol via DiscoveryConfig::from_settings() \
             then `config.evaluation_symbol = symbol.to_string()` if it differs \
             from settings.system.symbol."
        );
    }
    if config.evaluation_account_currency.trim().is_empty() {
        anyhow::bail!(
            "run_discovery_cycle: DiscoveryConfig.evaluation_account_currency \
             is empty. Set `system.account_currency` in config.yaml (or via \
             the cTrader trader-profile bridge when the broker session is \
             alive), or pass the env var NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY. \
             Empty currency causes the cost model to return NaN spread/pip \
             values that the sanitizer scrubs to 0.0 — every GA candidate \
             ends up with 0 trades and the operator sees no diagnostic."
        );
    }
    if !config.evaluation_spread_pips.is_finite() {
        anyhow::bail!(
            "run_discovery_cycle: DiscoveryConfig.evaluation_spread_pips is \
             non-finite ({}). Set settings.risk.backtest_spread_pips in \
             config.yaml (typical: 0.5–2.0 for FX, 2.5–8.0 for indices/\
             commodities; live spread varies — pick a backtest-conservative \
             value).",
            config.evaluation_spread_pips
        );
    }
    if !config.evaluation_commission_per_trade.is_finite() {
        anyhow::bail!(
            "run_discovery_cycle: DiscoveryConfig.evaluation_commission_per_trade \
             is non-finite ({}). Set settings.risk.commission_per_lot in \
             config.yaml. (D.2e wire-up now derives this from the broker's \
             commission_type+rate when SymbolMetadata is populated — but \
             the default-NaN sentinel still needs a real number for fully \
             standalone runs without a broker session.)",
            config.evaluation_commission_per_trade
        );
    }

    // 2026-05-26 operator directive (dual-mode product): instrument the
    // 16-stage rejection funnel before any pipeline work so a panic /
    // preflight failure still leaves a partially-populated funnel for the
    // operator to read. The funnel travels through the pipeline as a
    // borrowed mutable handle and is moved into the final DiscoveryResult.
    let mut funnel = crate::funnel_profile::FunnelProfile::new(
        if config.evaluation_symbol.is_empty() {
            "unknown_symbol".to_string()
        } else {
            config.evaluation_symbol.clone()
        },
        config.timeframe_label.clone(),
    );
    // Mode is determined by whether the prop-firm gate is configured. The
    // canonical paths are: PropFirm (config.prop_firm_gate.is_some()) and
    // Risky (gate absent — Strict / Risky modes fall here). Distinguishing
    // Strict vs Risky requires inspecting filtering thresholds; that nuance
    // lives in the report itself so the operator can tell modes apart.
    let mode_label = match config.mode {
        DiscoveryMode::Strict => "Strict",
        DiscoveryMode::PropFirm => "PropFirm",
        DiscoveryMode::Risky => "Risky",
    };
    funnel.set_mode(mode_label);

    // F-277 (2026-05-28): adaptive threshold ladder. The hardcoded
    // ladder in `evolution_math::random_coarse_threshold` is calibrated
    // for z-score-normalised features with unit-ish variance, but real
    // datasets vary widely in magnitude (XAGUSD M1 vs EURUSD D1 differ
    // by ~10×). When the operator opts in via
    // `models.discovery_runtime.adaptive_thresholds`, derive a per-dataset
    // ladder from the actual feature cube — gene init then picks
    // thresholds at percentile points of the dataset's own signal
    // magnitude distribution.
    //
    // The OnceLock semantics mean only the FIRST discovery run in a
    // process installs the ladder; subsequent runs on different
    // symbols would inherit the first symbol's ladder. The operator
    // should disable the feature for production multi-symbol sweeps
    // until F-277b adds per-symbol installation (deferred).
    if config.adaptive_thresholds {
        if let Some(ladder) =
            crate::genetic::derive_adaptive_threshold_ladder_from_features(&features.data)
        {
            match crate::genetic::install_adaptive_threshold_ladder(ladder) {
                Ok(_) => tracing::info!(
                    target: "neoethos_search::discovery",
                    p10 = ladder[0],
                    p25 = ladder[1],
                    p50 = ladder[2],
                    p75 = ladder[3],
                    p90 = ladder[4],
                    p99 = ladder[5],
                    "F-277: installed adaptive threshold ladder from feature cube"
                ),
                Err(existing) => tracing::warn!(
                    target: "neoethos_search::discovery",
                    new = ?ladder,
                    existing = ?existing,
                    "F-277: adaptive ladder already installed (first-run wins). \
                     Restart the process for a different symbol's ladder."
                ),
            }
        } else {
            tracing::warn!(
                target: "neoethos_search::discovery",
                "F-277: adaptive ladder derivation returned None (degenerate \
                 feature cube: empty or zero-variance). Falling back to static ladder."
            );
        }
    }

    // F-096 pre-flight: refuse to run with insufficient history per
    // operator's real-data directive 2026-05-24. The minimum-years
    // threshold lives on `DiscoveryRuntimeOverrides` (operator-tunable
    // via `Settings`); when zero, the check is skipped — used by test
    // fixtures + replay paths that have intentionally-small windows.
    if let Err(err) = ensure_sufficient_history(
        ohlcv,
        &config.evaluation_symbol,
        &config.timeframe_label,
        config.runtime_overrides.min_history_years,
    ) {
        funnel.finalize("preflight_failed");
        return Err(err);
    }

    let n_input_rows = ohlcv.close.len();
    funnel.record_stage("data_loaded", n_input_rows, n_input_rows);

    let (mut features, ohlcv, _) = trim_recent_history(features, ohlcv, config)?;
    let n_after_trim = ohlcv.close.len();
    funnel.record_stage("rows_after_trimming", n_input_rows, n_after_trim);
    funnel.record_stage("features_built", 0, features.data.ncols());

    // Feature Pre-filtering (Idea #3)
    let prefilter_top_k = config.runtime_overrides.prefilter_top_k;
    let prefilter_insample_frac = config.runtime_overrides.resolved_prefilter_insample_frac();

    let n_features_before_prefilter = features.names.len();
    if prefilter_top_k > 0 && features.names.len() > prefilter_top_k {
        features = prefilter_features(&features, &ohlcv, prefilter_top_k, prefilter_insample_frac);
    }
    funnel.record_stage(
        "features_after_prefilter",
        n_features_before_prefilter,
        features.names.len(),
    );
    // Capture names after prefilter — gene indices refer to this list.
    let effective_feature_names = features.names.clone();

    // Multi-stage Funnel: Stage 1 (Fast Evaluation)
    let stage1_pct = config.runtime_overrides.resolved_funnel_stage1_pct();
    let stage1_window = config.runtime_overrides.stage1_window;

    let total_rows = ohlcv.close.len();
    let stage1_len = ((total_rows as f64 * stage1_pct) as usize).min(total_rows);
    let (stage1_start, stage1_end) = match stage1_window {
        Stage1Window::MostRecent => (total_rows.saturating_sub(stage1_len), total_rows),
        Stage1Window::Earliest => (0, stage1_len),
    };
    tracing::info!(
        target: "neoethos_search::funnel",
        window = ?stage1_window,
        stage1_pct,
        stage1_rows = stage1_len,
        total_rows,
        "stage 1 fast-evaluation slice"
    );
    let ohlcv_stage1 = slice_ohlcv(&ohlcv, stage1_start, stage1_end);
    let features_stage1 = FeatureFrame {
        timestamps: features.timestamps[stage1_start..stage1_end].to_vec(),
        names: features.names.clone(),
        data: features
            .data
            .slice(ndarray::s![stage1_start..stage1_end, ..])
            .to_owned(),
    };
    progress_fn(DiscoveryProgress::SearchStarted {
        population: config.population,
        generations: config.generations,
        max_indicators: config.max_indicators,
    });
    let max_runtime = if config.max_hours > 0.0 {
        Some(std::time::Duration::from_secs_f64(
            config.max_hours * 3600.0,
        ))
    } else {
        None
    };
    let search = evolve_search_with_progress_and_limits(
        &features_stage1,
        &ohlcv_stage1,
        config.population,
        config.generations,
        config.max_indicators,
        max_runtime,
        Some(config.evaluation_config(ohlcv_stage1.close.last().copied())),
        |generation, total_generations, best_fitness, stagnant_generations, archived_profitable| {
            progress_fn(DiscoveryProgress::GenerationCompleted {
                generation,
                total_generations,
                best_fitness,
                stagnant_generations,
                archived_profitable,
            });
        },
    )?;

    let stage1_count = search.genes.len();
    funnel.record_stage("stage1_candidates_generated", 0, stage1_count);
    // The archive that survived the GA is what we hand the IS evaluator. The
    // genes themselves carry a `fitness` field reflecting the stage-1
    // evaluation, so "profitable" here means fitness > 0.0. The GA already
    // applies its own profitable-archive filter (`apply_metrics` archives
    // only nonnegative-fitness genes), so this stage is informational —
    // count_in == count_out unless the GA archive logic changes.
    let profitable_count = search.genes.iter().filter(|g| g.fitness > 0.0).count();
    funnel.record_stage("profitable_archive_size", stage1_count, profitable_count);

    finalize_candidates_with_progress(
        search.genes,
        &features,
        &ohlcv,
        config,
        effective_feature_names,
        &mut funnel,
        progress_fn,
    )
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..x.len() {
        let a = x[i];
        let b = y[i];
        sum_x += a;
        sum_y += b;
        sum_xy += a * b;
        sum_x2 += a * a;
        sum_y2 += b * b;
    }

    let num = n * sum_xy - sum_x * sum_y;
    let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    if den == 0.0 || !den.is_finite() {
        0.0
    } else {
        num / den
    }
}

fn prefilter_features(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    top_k: usize,
    insample_frac: f64,
) -> FeatureFrame {
    let n_rows = features.data.nrows();
    let n_cols = features.data.ncols();
    if n_rows < 2 || n_cols <= top_k {
        return features.clone();
    }

    // BUGFIX (data snooping): the prefilter ranks indicators by correlation
    // with 1-bar FORWARD returns. Computing this over the full dataset means
    // the "best" indicators are chosen with full knowledge of the OOS bars
    // they will later be evaluated on, inflating in-sample metrics.
    // Restrict the ranking to an IN-SAMPLE prefix so the final 30% of bars
    // (which the GA/walk-forward later treats as held-out) cannot leak into
    // the feature-selection step. The fraction is supplied by the caller
    // through `DiscoveryRuntimeOverrides::prefilter_insample_frac`.
    let train_end = ((n_rows as f64) * insample_frac).floor() as usize;
    let train_end = train_end.clamp(2, n_rows.saturating_sub(1)).max(2);

    // Calculate 1-bar forward returns ONLY for the in-sample window.
    // Returns past `train_end-1` are ignored (treated as zeros) so the
    // correlation reflects in-sample behaviour only.
    let mut returns = vec![0.0f32; n_rows];
    for (i, ret_slot) in returns
        .iter_mut()
        .enumerate()
        .take(train_end.saturating_sub(1))
    {
        let denom = ohlcv.close[i];
        if denom.abs() > 1e-12 {
            *ret_slot = ((ohlcv.close[i + 1] - denom) / denom) as f32;
        }
    }

    let mut correlations = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        let name = &features.names[col_idx];
        if name.starts_with("regime_") {
            // Force keep regime columns by giving them infinite correlation
            correlations.push((col_idx, f32::INFINITY));
        } else {
            // Restrict the column slice to the in-sample window so the
            // Pearson correlation only sees in-sample co-movement.
            let col = features.data.column(col_idx);
            let col_full: Vec<f32> = col.iter().copied().collect();
            let col_train = &col_full[..train_end.saturating_sub(1)];
            let ret_train = &returns[..train_end.saturating_sub(1)];
            let corr = pearson_correlation(col_train, ret_train);
            correlations.push((col_idx, corr.abs()));
        }
    }

    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate how many to actually keep: top_k + any regime columns
    let regime_count = features
        .names
        .iter()
        .filter(|n| n.starts_with("regime_"))
        .count();
    let actual_top_k = (top_k + regime_count).min(n_cols);

    let mut keep_indices: Vec<usize> = correlations
        .iter()
        .take(actual_top_k)
        .map(|(idx, _)| *idx)
        .collect();
    keep_indices.sort(); // Maintain original order

    let mut new_names = Vec::with_capacity(actual_top_k);
    let mut new_data = ndarray::Array2::zeros((n_rows, actual_top_k));

    for (new_col_idx, &orig_col_idx) in keep_indices.iter().enumerate() {
        new_names.push(features.names[orig_col_idx].clone());
        new_data
            .column_mut(new_col_idx)
            .assign(&features.data.column(orig_col_idx));
    }

    FeatureFrame {
        timestamps: features.timestamps.clone(),
        names: new_names,
        data: new_data,
    }
}

fn validate_regime_robustness(
    trades: &[crate::quality::Trade],
    features: &FeatureFrame,
    initial_balance: f64,
    max_regime_loss_pct: f64,
) -> bool {
    let trend_idx = features
        .names
        .iter()
        .position(|n| n == "regime_trend_strength");
    let vol_idx = features.names.iter().position(|n| n == "regime_vol_state");

    // **2026-05-25 unwrap audit**: collapsed the early-return guard +
    // two `.unwrap()` calls into a single `let-else` destructure. Same
    // behaviour, no panic-shaped expression remains.
    let (Some(t_idx), Some(v_idx)) = (trend_idx, vol_idx) else {
        return true;
    };

    let mut trend_pnl = 0.0;
    let mut range_pnl = 0.0;
    let mut high_vol_pnl = 0.0;
    let mut low_vol_pnl = 0.0;

    let mut last_idx = 0;
    let t_len = features.timestamps.len();

    for trade in trades {
        let ts = trade.entry_time;
        while last_idx < t_len && features.timestamps[last_idx] < ts {
            last_idx += 1;
        }
        let idx = if last_idx < t_len {
            last_idx
        } else {
            t_len.saturating_sub(1)
        };
        if idx >= features.data.nrows() {
            continue;
        }

        let trend_str = features.data[(idx, t_idx)];
        let vol_state = features.data[(idx, v_idx)];

        if trend_str > 0.25 {
            trend_pnl += trade.pnl;
        } else if trend_str < 0.15 {
            range_pnl += trade.pnl;
        }

        if vol_state > 0.5 {
            high_vol_pnl += trade.pnl;
        } else if vol_state < -0.5 {
            low_vol_pnl += trade.pnl;
        }
    }

    let limit = -(initial_balance * max_regime_loss_pct / 100.0);

    if trend_pnl < limit || range_pnl < limit || high_vol_pnl < limit || low_vol_pnl < limit {
        return false;
    }

    true
}

/// Discovery search modes. The default is `PropFirm`; `Strict` is opted into
/// via `models.discovery_mode = "strict"` in config (mapped by
/// `discovery_mode_from_config`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryMode {
    /// Production-grade strict pipeline (legacy walkforward + CPCV +
    /// MC-perturbation gates). Use only when looking for unicorn
    /// strategies that survive every consistency test in the codebase.
    Strict,
    /// Self-tuning prop-firm passing mode. Default. Permissive filter
    /// floors + FTMO window-pass scoring + ranking-based portfolio
    /// selection. Designed to deliver portfolios that can pass an
    /// actual prop-firm challenge in 60 days per phase.
    PropFirm,
    /// Aggressive capital-multiplication mode (the user-facing "Risky"
    /// trading mode). High-risk-tolerant filter floors, a growth-tilted
    /// candidate ranking (fitness-dominated, NO drawdown tax) and NO
    /// prop-firm window-pass gate. Optimises for the fastest compounding of a
    /// small balance toward a large target, accepting deep drawdown and a high
    /// ruin probability by design.
    Risky,
}

/// Map the config `models.discovery_mode` string to a [`DiscoveryMode`].
/// `"strict"` / `"legacy"` → `Strict`; anything else (including the default
/// `"prop_firm"`) → `PropFirm`. Config-driven replacement for the env-only
/// `resolve_discovery_mode` that read `NEOETHOS_BOT_DISCOVERY_MODE` and the
/// legacy `NEOETHOS_BOT_DISCOVERY_PERMISSIVE` back-compat toggle. The
/// permissive-toggle path is retired with the env var — operators select the
/// regime through `config.yaml` / the UI now.
fn discovery_mode_from_config(value: &str) -> DiscoveryMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "strict" | "legacy" => DiscoveryMode::Strict,
        _ => DiscoveryMode::PropFirm,
    }
}

/// Resolve the active [`DiscoveryMode`] from the operator's top-level
/// `system.trading_mode` (the user-facing master switch) and the advanced
/// `models.discovery_mode` escape hatch.
///
/// Precedence:
///  1. An explicit `models.discovery_mode = "strict"` / `"legacy"` forces the
///     strict unicorn-hunting pipeline regardless of trading mode (power user).
///  2. Otherwise `system.trading_mode` decides: `"risky"` (or `"growth"`) →
///     [`DiscoveryMode::Risky`]; anything else (incl. the `"prop_firm"`
///     default) → [`DiscoveryMode::PropFirm`].
fn resolve_discovery_mode(trading_mode: &str, discovery_mode: &str) -> DiscoveryMode {
    if matches!(
        discovery_mode_from_config(discovery_mode),
        DiscoveryMode::Strict
    ) {
        return DiscoveryMode::Strict;
    }
    match trading_mode.trim().to_ascii_lowercase().as_str() {
        "risky" | "growth" => DiscoveryMode::Risky,
        _ => DiscoveryMode::PropFirm,
    }
}

/// Pick a window count that scales with how many full window-spans the
/// dataset can offer. Lots of history → more samples; bare minimum data
/// → at least a few samples so the score is meaningful.
fn auto_tune_n_windows(timestamps: &[i64], window_days: usize) -> usize {
    if timestamps.is_empty() || window_days == 0 {
        return 50;
    }
    let span_ms = (timestamps[timestamps.len() - 1] - timestamps[0]).max(0);
    let window_ms = (window_days as i64) * 86_400_000;
    if window_ms == 0 {
        return 50;
    }
    let full_spans = (span_ms / window_ms).max(0) as usize;
    // Sample ~3× as many windows as the dataset contains non-overlapping
    // spans (overlap is fine — we want resolution along the timeline)
    // but cap so we don't spend the whole budget here.
    (full_spans * 3).clamp(20, 200)
}

/// Sample roughly evenly-spaced 30-day (configurable) windows from the
/// dataset history; for each window simulate trades and check the strategy
/// against `compute_prop_firm_risk_summary`. Return the fraction of
/// windows whose `all_rules_passed` flag is true.
///
/// This measures what an actual prop-firm challenge measures (one
/// 30-day window, FTMO rules) — much more directly relevant than the
/// "every walkforward split must be profitable" gate.
fn compute_prop_firm_pass_rate(
    gene: &Gene,
    signals: &[i8],
    ohlcv: &Ohlcv,
    timestamps: &[i64],
    config: &DiscoveryConfig,
    overrides: &PropFirmGateOverrides,
) -> (f64, usize) {
    let n = signals
        .len()
        .min(timestamps.len())
        .min(ohlcv.close.len())
        .min(ohlcv.high.len())
        .min(ohlcv.low.len());
    if n == 0 || overrides.window_days == 0 || overrides.n_windows == 0 {
        return (0.0, 0);
    }
    let window_ms: i64 = (overrides.window_days as i64) * 86_400_000;
    let first_ts = timestamps[0];
    let last_ts = timestamps[n - 1];
    if last_ts - first_ts < window_ms {
        return (0.0, 0);
    }
    let max_start_ts = last_ts - window_ms;
    let span = (max_start_ts - first_ts).max(1) as f64;
    let n_windows = overrides.n_windows.max(1);
    let stride = if n_windows == 1 {
        0.0
    } else {
        span / (n_windows as f64 - 1.0)
    };

    let settings = discovery_backtest_settings(config, gene, ohlcv.close.last().copied());
    let initial_balance = config.initial_balance.max(1.0);

    let mut passes = 0usize;
    let mut counted = 0usize;
    for i in 0..n_windows {
        let start_ts = if n_windows == 1 {
            first_ts
        } else {
            first_ts + stride.mul_add(i as f64, 0.0) as i64
        };
        let end_ts = start_ts + window_ms;
        let start_idx = timestamps.partition_point(|&t| t < start_ts);
        let end_idx = timestamps.partition_point(|&t| t < end_ts).min(n);
        if end_idx <= start_idx + 1 {
            continue;
        }
        let close = &ohlcv.close[start_idx..end_idx];
        let high = &ohlcv.high[start_idx..end_idx];
        let low = &ohlcv.low[start_idx..end_idx];
        let ts = &timestamps[start_idx..end_idx];
        let sig = &signals[start_idx..end_idx];
        let trades = simulate_trades_core(close, high, low, ts, sig, &settings);
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &trades,
            initial_balance,
            rules: overrides.rules,
        });
        if summary.all_rules_passed {
            passes += 1;
        }
        counted += 1;
    }
    if counted == 0 {
        return (0.0, 0);
    }
    (passes as f64 / counted as f64, counted)
}

fn finalize_candidates_with_progress<F>(
    candidates: Vec<Gene>,
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_feature_names: Vec<String>,
    funnel: &mut crate::funnel_profile::FunnelProfile,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    // Diagnostic: summarise the feature frame so we can tell whether the
    // GA's empty-portfolio outcome is downstream filtering vs the upstream
    // features being broken (NaN-saturated, all-zero, wrong magnitude).
    {
        let total = features.data.len();
        let mut nan = 0usize;
        let mut zero = 0usize;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum_abs = 0.0_f64;
        let mut finite_count = 0usize;
        for &v in features.data.iter() {
            if v.is_nan() {
                nan += 1;
            } else if v == 0.0 {
                zero += 1;
                finite_count += 1;
            } else if v.is_finite() {
                finite_count += 1;
                sum_abs += v.abs() as f64;
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
        }
        let mean_abs = if finite_count > 0 {
            sum_abs / finite_count as f64
        } else {
            0.0
        };
        tracing::info!(
            target: "neoethos_search::funnel",
            rows = features.data.nrows(),
            cols = features.data.ncols(),
            nan_frac = nan as f64 / total.max(1) as f64,
            zero_frac = zero as f64 / total.max(1) as f64,
            min_finite = if min_v.is_finite() { min_v as f64 } else { 0.0 },
            max_finite = if max_v.is_finite() { max_v as f64 } else { 0.0 },
            mean_abs_finite = mean_abs,
            "feature frame summary"
        );

        // F-310 (2026-05-28): per-column variance check on the trailing
        // window. The NaN+zero counters above can't see "frozen
        // constant" columns — F-308 was about higher-TF forward-fill
        // staling, and the resulting column values are FINITE NON-ZERO
        // but all identical. Indicators on a constant input emit a
        // constant; GA on a constant signal produces zero-trade
        // candidates. This sub-diagnostic walks each column over the
        // last `min(rows, 1000)` rows and counts columns whose
        // (max−min) is essentially zero. A high count is the
        // unambiguous signal that the alignment / data pipeline broke.
        let trailing = features.data.nrows().min(1000);
        if trailing > 1 {
            let n_cols = features.data.ncols();
            let mut zero_var_cols = 0usize;
            let mut named_examples: Vec<String> = Vec::new();
            let start_row = features.data.nrows() - trailing;
            for c in 0..n_cols {
                let mut col_min = f32::INFINITY;
                let mut col_max = f32::NEG_INFINITY;
                let mut finite_seen = 0usize;
                for r in start_row..features.data.nrows() {
                    let v = features.data[(r, c)];
                    if v.is_finite() {
                        finite_seen += 1;
                        if v < col_min {
                            col_min = v;
                        }
                        if v > col_max {
                            col_max = v;
                        }
                    }
                }
                // Zero-variance only if we saw enough finite values AND
                // the span is below epsilon. Skip mostly-NaN columns —
                // those are already counted in `nan_frac`.
                if finite_seen >= (trailing * 7 / 10)
                    && col_min.is_finite()
                    && col_max.is_finite()
                    && (col_max - col_min).abs() < 1e-9
                {
                    zero_var_cols += 1;
                    if named_examples.len() < 5
                        && c < features.names.len()
                    {
                        named_examples.push(features.names[c].clone());
                    }
                }
            }
            if zero_var_cols > 0 {
                tracing::warn!(
                    target: "neoethos_search::funnel",
                    zero_var_cols,
                    total_cols = n_cols,
                    trailing_rows = trailing,
                    examples = ?named_examples,
                    "F-310: zero-variance feature columns detected over trailing window. \
                     Most-likely cause: stale higher-TF data being forward-filled into \
                     base bars (F-308 / F-309 scope). Operator action: re-bootstrap \
                     the affected higher timeframe."
                );
            }
        }
    }
    // Sort by an income-focused ranking score to find reliably profitable ones
    let mut ranked_candidates: Vec<(usize, Gene)> = candidates.into_iter().enumerate().collect();

    // Ranking score. PropFirm / Strict use the income-focused blend
    // (consistency, win-rate, drawdown-safety, profit-factor). Risky /
    // capital-multiplication uses a growth-tilted score: fitness-dominated
    // (fitness is the GA's own growth objective) with NO drawdown tax, so the
    // fastest compounder wins even on a deep equity curve.
    let risky_ranking = matches!(config.mode, DiscoveryMode::Risky);
    // Target-aware Risky ranking precompute: the required TOTAL log-growth to
    // get from the operator's start balance to their target, and the dataset
    // span in days (to scale each gene's trade cadence to the horizon). This is
    // the "pressure on the search" — the goal flows into selection.
    let required_log_growth = if risky_ranking && config.risky_start_balance > 0.0 {
        (config.risky_target_balance / config.risky_start_balance)
            .max(1.0)
            .ln()
    } else {
        0.0
    };
    let span_days = if features.timestamps.len() >= 2 {
        ((features.timestamps[features.timestamps.len() - 1] - features.timestamps[0]).max(0)
            as f64)
            / 86_400_000.0
    } else {
        0.0
    };
    let calculate_income_score = |gene: &Gene| -> f64 {
        if risky_ranking {
            // Per-trade edge from the gene's OWN measured stats.
            let p = gene.win_rate.clamp(0.0, 1.0);
            let pf = gene.profit_factor.max(0.0);
            // Kelly fraction f* = p·(pf−1)/pf (0 when no edge); half-Kelly,
            // capped at 25% so a single loss never wipes the bankroll.
            let f_star = if pf > 1.0 && p > 0.0 {
                p * (pf - 1.0) / pf
            } else {
                0.0
            };
            let f = (f_star * 0.5).clamp(0.0, 0.25);
            // Reward-to-risk implied by (pf, p): avg_win / avg_loss.
            let rr = if p > 0.0 && p < 1.0 {
                pf * (1.0 - p) / p
            } else {
                0.0
            };
            // Expected per-trade log-growth at f (the Kelly growth rate).
            let g_trade = if f > 0.0 && rr > 0.0 {
                p * (1.0 + rr * f).ln() + (1.0 - p) * (1.0 - f).ln()
            } else {
                0.0
            };
            // Trades this gene would fire over the horizon (scale its backtest
            // cadence to the horizon length).
            let trades_in_horizon = if span_days > 0.0 {
                gene.trades_count as f64 / span_days * config.risky_horizon_days
            } else {
                0.0
            };
            let achievable = g_trade * trades_in_horizon;
            // Score by how close to (or past) the required growth, capped so a
            // high-variance overshoot does not win on luck; a mild fitness tilt
            // breaks ties toward robust genes.
            let ratio = if required_log_growth > 0.0 {
                (achievable / required_log_growth).max(0.0)
            } else {
                achievable.max(0.0)
            };
            ratio.min(3.0) * (0.7 + 0.3 * gene.fitness.max(0.0).min(1.0))
        } else {
            let pf_capped = gene.profit_factor.min(3.0) / 3.0; // Normalized 0-1
            let safety = (1.0 - gene.max_drawdown / 0.07).clamp(0.0, 1.0);
            let consistency_score = gene.consistency; // 0-1
            let win_rate_score = gene.win_rate; // 0-1

            let multiplier = (consistency_score * 0.4)
                + (win_rate_score * 0.3)
                + (safety * 0.2)
                + (pf_capped * 0.1);

            // Bonus for high consistency (proxy for 10/12+ positive months)
            let bonus = if consistency_score > 0.8 { 2.0 } else { 1.0 };

            gene.fitness * multiplier * bonus
        }
    };

    ranked_candidates.sort_by(|(idx_a, a), (idx_b, b)| {
        let score_a = calculate_income_score(a);
        let score_b = calculate_income_score(b);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.consistency
                    .partial_cmp(&a.consistency)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.strategy_id.cmp(&b.strategy_id))
            .then_with(|| idx_a.cmp(idx_b))
    });
    let max_candidates =
        candidate_truncation_limit(config.candidate_count, ranked_candidates.len());
    ranked_candidates.truncate(max_candidates);
    let ranked_candidate_genes: Vec<Gene> = ranked_candidates
        .iter()
        .map(|(_, gene)| gene.clone())
        .collect();
    progress_fn(DiscoveryProgress::CandidatesRanked {
        candidate_count: ranked_candidates.len(),
        truncated_to: max_candidates,
    });

    let min_trades = min_trades_required(
        &features.timestamps,
        config.min_trades_per_day,
        features.data.nrows(),
    );
    let ranked_total = ranked_candidates.len();
    // 2026-05-26 operator directive: now that GA produced candidates,
    // record the "full IS eval" stage in the funnel — this is the gene
    // count fed into the post-search filter ladder.
    funnel.record_stage("full_is_evaluated", ranked_total, ranked_total);

    // Diagnostic counter #1: `passes_filter` survivors. In permissive
    // / prop-firm mode this gate is trivially open, so a low number
    // here would be a strong signal that the filter floor still has
    // a hidden constraint we missed.
    // 2026-05-26: also bucket WHY each gene failed `passes_filter` so the
    // funnel JSON tells the operator which threshold (DD / win-rate / PF)
    // killed most candidates.
    let mut reject_dd = 0usize;
    let mut reject_win_rate = 0usize;
    let mut reject_profit_factor = 0usize;
    let mut reject_fitness = 0usize;
    let mut reject_other = 0usize;
    let prefiltered: Vec<(usize, Gene)> = ranked_candidates
        .iter()
        .filter(|(_, g)| {
            let ok = g.passes_filter(&config.filtering);
            if !ok {
                // Cheap heuristic: pick the FIRST violated threshold so the
                // counts roughly partition the rejections. Not every Gene
                // populates every metric, so the buckets are a guide rather
                // than an audit trail.
                if !g.max_drawdown.is_nan() && g.max_drawdown > config.filtering.max_dd {
                    reject_dd += 1;
                } else if !g.win_rate.is_nan() && g.win_rate < config.filtering.min_win_rate {
                    reject_win_rate += 1;
                } else if !g.profit_factor.is_nan()
                    && g.profit_factor < config.filtering.min_profit_factor
                {
                    reject_profit_factor += 1;
                } else if !g.fitness.is_nan() && g.fitness < config.filtering.min_sharpe {
                    reject_fitness += 1;
                } else {
                    reject_other += 1;
                }
            }
            ok
        })
        .map(|(idx, g)| (*idx, g.clone()))
        .collect();
    let post_passes_filter = prefiltered.len();
    funnel.record_stage("passed_base_filter", ranked_total, post_passes_filter);
    if reject_dd > 0 {
        funnel.add_reject_reason("passed_base_filter", "max_dd_exceeded", reject_dd);
    }
    if reject_win_rate > 0 {
        funnel.add_reject_reason("passed_base_filter", "win_rate_too_low", reject_win_rate);
    }
    if reject_profit_factor > 0 {
        funnel.add_reject_reason(
            "passed_base_filter",
            "profit_factor_too_low",
            reject_profit_factor,
        );
    }
    if reject_fitness > 0 {
        funnel.add_reject_reason("passed_base_filter", "fitness_too_low", reject_fitness);
    }
    if reject_other > 0 {
        funnel.add_reject_reason("passed_base_filter", "other_threshold", reject_other);
    }

    // Item 6: use the SMC-gated signal path so the post-search "min_trades"
    // filter sees the SAME trade count the evaluator scored. The previous
    // `signals_for_gene` ignored gene SMC flags; some candidates passed the
    // search archive (with their SMC-gated trade count) but were then pruned
    // here because the un-gated count was higher than min_trades.
    let eval_config_for_signals = config.evaluation_config(ohlcv.close.last().copied());

    // Diagnostic counter #2: how many genes generated ANY non-zero
    // signal at all? A gene with `long_threshold > max possible
    // combined signal` never fires. We track this separately from
    // the min_trades gate so we can tell "no signal" from "too few
    // trades".
    let nonzero_signal_count = std::sync::atomic::AtomicUsize::new(0);
    let signals_with_idx: Vec<(usize, Gene, Vec<i8>)> = prefiltered
        .into_par_iter()
        .filter_map(|(candidate_idx, gene)| {
            let sig = signals_for_gene_full(features, ohlcv, &gene, &eval_config_for_signals);
            let trade_count = sig.iter().filter(|v| **v != 0).count() as f64;
            if trade_count > 0.0 {
                nonzero_signal_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if trade_count >= min_trades as f64 {
                Some((candidate_idx, gene, sig))
            } else {
                None
            }
        })
        .collect();
    let post_min_trades = signals_with_idx.len();
    let post_nonzero_signal = nonzero_signal_count.load(std::sync::atomic::Ordering::Relaxed);
    // 2026-05-26: record "any signal at all" + "passed min-trades" as separate
    // stages so the funnel can tell "SMC gate killed everything" (the common
    // empty-portfolio root cause) apart from "had signals but too few".
    funnel.record_stage("nonzero_signals", post_passes_filter, post_nonzero_signal);
    let zero_signal_rejects = post_passes_filter.saturating_sub(post_nonzero_signal);
    if zero_signal_rejects > 0 {
        funnel.add_reject_reason(
            "nonzero_signals",
            "zero_signals_after_smc_gate",
            zero_signal_rejects,
        );
    }
    funnel.record_stage("passed_min_trades", post_nonzero_signal, post_min_trades);
    let mut filtered: Vec<(usize, Gene)> = Vec::with_capacity(signals_with_idx.len());
    let mut signals_map: Vec<Vec<i8>> = Vec::with_capacity(signals_with_idx.len());
    for (idx, gene, sig) in signals_with_idx {
        filtered.push((idx, gene));
        signals_map.push(sig);
    }
    progress_fn(DiscoveryProgress::CandidatesFiltered {
        passed_filters: filtered.len(),
        evaluated_candidates: ranked_candidates.len(),
        min_trades_required: min_trades,
    });

    let filtered_count = filtered.len();
    let mut quality_metrics = Vec::new();
    let mut logged_trades = Vec::new();
    if Gene::requires_quality_screen(&config.filtering) {
        type QualityCandidate = (usize, Gene, Vec<i8>, StrategyMetrics, bool, Vec<Trade>);
        let analyzer = quality_analyzer_for_config(config);
        let initial_balance = config.initial_balance;

        // Outer-parallel quality screen: each candidate runs simulate_trades +
        // 100 MC perturbations + spread sensitivity independently. Previously
        // the outer loop was serial and only the 100-run MC was parallel,
        // which under-utilised cores when the candidate set was large. Move
        // parallelism to the outer level and keep the MC loop serial — this
        // avoids rayon nested-parallel oversubscription and gives ~Ncores×
        // throughput on the per-candidate work.
        let pairs: Vec<((usize, Gene), Vec<i8>)> = filtered.into_iter().zip(signals_map).collect();
        let screened: Vec<Option<QualityCandidate>> = pairs
            .into_par_iter()
            .map(|((candidate_idx, gene), sig)| {
                let trades = crate::eval::simulate_trades_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &features.timestamps,
                    &sig,
                    &discovery_backtest_settings(config, &gene, ohlcv.close.last().copied()),
                );
                let metrics =
                    analyzer.analyze_strategy(&gene.strategy_id, &trades, initial_balance);
                let strict_quality = passes_strict_quality(&metrics, &config.filtering);
                let opportunistic_quality =
                    !strict_quality && passes_opportunistic_quality(&metrics, &config.filtering);

                if !(strict_quality || opportunistic_quality) {
                    return None;
                }

                // Regime-Aware Validation (Idea #3.2)
                let regime_robust = validate_regime_robustness(
                    &trades,
                    features,
                    config.initial_balance,
                    config.max_regime_loss_pct,
                );
                if !regime_robust {
                    return None;
                }

                // Monte Carlo Parameter Perturbation Test.
                // Serial here because we are already inside a par_iter on
                // candidates — nesting rayon would oversubscribe cores.
                // 2026-05-26 operator directive (dual-mode product): runs +
                // min_profitable threshold sourced from typed Settings,
                // previously hardcoded 100/70.
                let mc_runs = config.mc_runs as usize;
                let mut profitable_runs = 0usize;
                let mut rng = rand::rng();
                use rand::Rng;
                for _ in 0..mc_runs {
                    let mut perturbed = gene.clone();
                    perturbed.long_threshold *= 1.0 + rng.random_range(-0.15..=0.15);
                    perturbed.short_threshold *= 1.0 + rng.random_range(-0.15..=0.15);
                    for w in &mut perturbed.weights {
                        *w *= 1.0 + rng.random_range(-0.20..=0.20);
                    }
                    if perturbed.sl_pips.is_finite() && perturbed.sl_pips > 0.0 {
                        perturbed.sl_pips *= 1.0 + rng.random_range(-0.25..=0.25);
                    }
                    if perturbed.tp_pips.is_finite() && perturbed.tp_pips > 0.0 {
                        perturbed.tp_pips *= 1.0 + rng.random_range(-0.25..=0.25);
                    }
                    // Item 6: SMC-gated signal so the MC perturbation reward is
                    // measured against the same execution rule the search used.
                    let p_sig = crate::genetic::signals_for_gene_full(
                        features,
                        ohlcv,
                        &perturbed,
                        &eval_config_for_signals,
                    );
                    let p_trades = crate::eval::simulate_trades_core(
                        &ohlcv.close,
                        &ohlcv.high,
                        &ohlcv.low,
                        &features.timestamps,
                        &p_sig,
                        &discovery_backtest_settings(
                            config,
                            &perturbed,
                            ohlcv.close.last().copied(),
                        ),
                    );
                    let pnl: f64 = p_trades.iter().map(|t| t.pnl).sum();
                    if pnl > 0.0 {
                        profitable_runs += 1;
                    }
                }

                if (profitable_runs as u32) < config.mc_min_profitable {
                    return None;
                }

                // Spread/Slippage Sensitivity Test — wired from Settings
                // 2026-05-26 (dual-mode product).
                let mut sensitive_settings =
                    discovery_backtest_settings(config, &gene, ohlcv.close.last().copied());
                sensitive_settings.spread_pips = config.sensitivity_spread_pips;
                sensitive_settings.commission_per_trade = config.sensitivity_commission_per_lot;
                let sens_trades = crate::eval::simulate_trades_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &features.timestamps,
                    &sig,
                    &sensitive_settings,
                );
                let sens_pnl: f64 = sens_trades.iter().map(|t| t.pnl).sum();
                if sens_pnl < 0.0 {
                    return None;
                }

                Some((
                    candidate_idx,
                    gene,
                    sig,
                    metrics,
                    opportunistic_quality,
                    trades,
                ))
            })
            .collect();

        let mut strict_passed: Vec<QualityCandidate> = Vec::new();
        let mut opportunistic_passed = 0usize;
        for entry in screened.into_iter().flatten() {
            if entry.4 {
                opportunistic_passed += 1;
            }
            quality_metrics.push(entry.3.clone());
            strict_passed.push(entry);
        }

        strict_passed.sort_by(|a, b| {
            let lane_a = if a.4 { 0_u8 } else { 1_u8 };
            let lane_b = if b.4 { 0_u8 } else { 1_u8 };
            lane_b
                .cmp(&lane_a)
                .then_with(|| {
                    b.3.quality_score
                        .partial_cmp(&a.3.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.1.fitness
                        .partial_cmp(&a.1.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.1.strategy_id.cmp(&b.1.strategy_id))
                .then_with(|| a.0.cmp(&b.0))
        });

        if config.filtering.log_trades {
            logged_trades = strict_passed
                .iter()
                .filter(|entry| !entry.5.is_empty())
                .take(config.filtering.trade_log_max)
                .map(|entry| LoggedStrategyTrades {
                    strategy_id: entry.1.strategy_id.clone(),
                    opportunistic: entry.4,
                    trades: entry.5.clone(),
                })
                .collect();
        }
        let logged_trade_sets = logged_trades.len();

        progress_fn(DiscoveryProgress::QualityScreened {
            strict_passed: strict_passed.len().saturating_sub(opportunistic_passed),
            opportunistic_passed,
            evaluated_candidates: filtered_count,
            logged_trade_sets,
        });

        let mut screened_genes = Vec::with_capacity(strict_passed.len());
        let mut screened_signals = Vec::with_capacity(strict_passed.len());
        for (candidate_idx, gene, sig, _, _, _) in strict_passed {
            screened_genes.push((candidate_idx, gene));
            screened_signals.push(sig);
        }
        filtered = screened_genes;
        signals_map = screened_signals;
    }
    // 2026-05-26: quality screen (MC perturbation + spread sensitivity +
    // regime robustness) collapses into a single funnel stage. The
    // sub-rejection reasons (MC <70/100, sensitivity loss, regime fail)
    // are visible via the tracing logs but not separately counted here —
    // adding per-reason buckets would require threading atomics into the
    // par_iter closure above. If operators report this stage is the
    // bottleneck, follow-up adds those atomics.
    funnel.record_stage("passed_quality", post_min_trades, filtered.len());

    // Prop-firm window-pass gate. Default behavior in `PropFirm` mode.
    // For each surviving candidate, simulate trades on N 60-day windows
    // sampled across history and check FTMO rules on each. Candidates
    // are then SORTED by pass-rate descending — no hard threshold to
    // tune. The downstream corr-diversification step takes the best
    // prop-firm-grade candidates first. A non-zero `pf.pass_rate` env
    // override still acts as a hard floor for operators who want it.
    let pre_prop_firm = filtered.len();
    let mut prop_firm_pass_rates: Vec<f64> = Vec::new();
    if let Some(mut pf) = config.prop_firm_gate.clone() {
        // Auto-tune the window count if the operator left it at the
        // sentinel value (0). Scales with available history.
        if pf.n_windows == 0 {
            pf.n_windows = auto_tune_n_windows(&features.timestamps, pf.window_days);
        }
        let candidates_in: Vec<((usize, Gene), Vec<i8>)> =
            filtered.into_iter().zip(signals_map.into_iter()).collect();
        let timestamps_owned = features.timestamps.clone();
        let candidates_in_count = candidates_in.len();
        let pf_pass_rate_floor = pf.pass_rate;
        let scored_all: Vec<(((usize, Gene), Vec<i8>), f64, usize)> = candidates_in
            .into_par_iter()
            .map(|(pair, sig)| {
                let (rate, counted) = compute_prop_firm_pass_rate(
                    &pair.1,
                    &sig,
                    ohlcv,
                    &timestamps_owned,
                    config,
                    &pf,
                );
                ((pair, sig), rate, counted)
            })
            .collect();
        // Diagnostic: bucket what the gate did to each candidate.
        let mut dbg_counted_zero = 0usize;
        let mut dbg_below_pass_rate = 0usize;
        let mut dbg_counted_sum = 0usize;
        let mut dbg_max_rate: f64 = 0.0;
        for (_, rate, counted) in &scored_all {
            dbg_counted_sum += *counted;
            if *counted == 0 {
                dbg_counted_zero += 1;
            } else if *rate < pf_pass_rate_floor {
                dbg_below_pass_rate += 1;
            }
            if *rate > dbg_max_rate {
                dbg_max_rate = *rate;
            }
        }
        let avg_counted = if candidates_in_count > 0 {
            dbg_counted_sum as f64 / candidates_in_count as f64
        } else {
            0.0
        };
        let ts_first = timestamps_owned.first().copied().unwrap_or(0);
        let ts_last = timestamps_owned.last().copied().unwrap_or(0);
        let ts_span = ts_last - ts_first;
        let window_ms_eff = (pf.window_days as i64) * 86_400_000;
        tracing::info!(
            target: "neoethos_search::prop_firm_dbg",
            candidates_in = candidates_in_count,
            rejected_counted_zero = dbg_counted_zero,
            rejected_below_pass_rate = dbg_below_pass_rate,
            avg_counted,
            max_rate = dbg_max_rate,
            pass_rate_floor = pf_pass_rate_floor,
            ts_first,
            ts_last,
            ts_span,
            window_ms_eff,
            timestamps_len = timestamps_owned.len(),
            "prop-firm gate breakdown — why candidates were rejected"
        );
        let mut scored: Vec<(((usize, Gene), Vec<i8>), f64, usize)> = scored_all
            .into_iter()
            .filter(|(_, rate, counted)| *counted > 0 && *rate >= pf.pass_rate)
            .collect();
        // Sort by pass-rate descending; ties broken by gene fitness.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.0.0
                        .1
                        .fitness
                        .partial_cmp(&a.0.0.1.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let mut next_filtered: Vec<(usize, Gene)> = Vec::with_capacity(scored.len());
        let mut next_signals: Vec<Vec<i8>> = Vec::with_capacity(scored.len());
        for ((pair, sig), rate, _) in scored {
            next_filtered.push(pair);
            next_signals.push(sig);
            prop_firm_pass_rates.push(rate);
        }
        let best_rate = prop_firm_pass_rates.first().copied().unwrap_or(0.0);
        tracing::info!(
            target: "neoethos_search::prop_firm",
            survivors = next_filtered.len(),
            best_pass_rate = best_rate,
            window_days = pf.window_days,
            n_windows = pf.n_windows,
            profit_target_pct = pf.rules.min_profit_target_pct,
            max_daily_loss_pct = pf.rules.max_daily_loss_pct,
            max_overall_drawdown_pct = pf.rules.max_overall_drawdown_pct,
            "prop-firm window-pass gate applied"
        );
        // 2026-05-26: record the prop-firm-window stage with its two top
        // reject reasons (counted_zero = the window-pass simulation produced
        // zero windows for this gene, e.g. dataset too short or all windows
        // crashed; below_pass_rate = some windows ran but pass-rate < floor).
        funnel.record_stage("passed_prop_firm_window", pre_prop_firm, next_filtered.len());
        if dbg_counted_zero > 0 {
            funnel.add_reject_reason(
                "passed_prop_firm_window",
                "counted_zero",
                dbg_counted_zero,
            );
        }
        if dbg_below_pass_rate > 0 {
            funnel.add_reject_reason(
                "passed_prop_firm_window",
                "below_pass_rate",
                dbg_below_pass_rate,
            );
        }
        filtered = next_filtered;
        signals_map = next_signals;
    } else {
        // No prop-firm gate (Risky mode / Strict mode): the stage is a
        // passthrough so the funnel doesn't show a phantom rejection.
        funnel.record_stage("passed_prop_firm_window", pre_prop_firm, pre_prop_firm);
    }

    let mut portfolio = Vec::new();
    let mut portfolio_signals: Vec<Vec<i8>> = Vec::new();
    let mut rejected_by_correlation = 0usize;
    let mut portfolio_pass_rates: Vec<f64> = Vec::new();
    for (idx, ((_, gene), sig)) in filtered.into_iter().zip(signals_map).enumerate() {
        if portfolio.len() >= config.portfolio_size {
            break;
        }
        let mut ok = true;
        for existing in &portfolio_signals {
            let pearson = pearson_corr_i8(&sig, existing);
            // DS-2: also check Spearman to catch non-linear dependencies
            let spearman = spearman_corr_i8(&sig, existing);
            // Reject if EITHER correlation exceeds threshold
            if pearson.abs() >= config.corr_threshold || spearman.abs() >= config.corr_threshold {
                ok = false;
                rejected_by_correlation += 1;
                break;
            }
        }
        if ok {
            portfolio_signals.push(sig);
            portfolio.push(gene);
            if let Some(rate) = prop_firm_pass_rates.get(idx) {
                portfolio_pass_rates.push(*rate);
            }
        }
    }
    progress_fn(DiscoveryProgress::PortfolioSelected {
        portfolio_size: portfolio.len(),
        rejected_by_correlation,
        target_portfolio: config.portfolio_size,
    });
    // Diagnostic summary: one line per (symbol, TF) work-unit showing
    // how many candidates survived each gate. Without this, an empty
    // portfolio just says "empty" — with it, you can pinpoint which
    // gate is rejecting everything.
    let post_prop_firm = if config.prop_firm_gate.is_some() {
        // After the gate ran, `filtered` was replaced with the
        // surviving set — its length is `prop_firm_pass_rates.len()`
        // (we pushed one rate per survivor).
        prop_firm_pass_rates.len()
    } else {
        pre_prop_firm
    };
    // 2026-05-26: correlation pruning is the last stage before walkforward.
    // Input = post_prop_firm count; output = portfolio.len().
    funnel.record_stage("passed_correlation", post_prop_firm, portfolio.len());
    if rejected_by_correlation > 0 {
        funnel.add_reject_reason(
            "passed_correlation",
            "pearson_or_spearman_above_threshold",
            rejected_by_correlation,
        );
    }
    tracing::info!(
        target: "neoethos_search::funnel",
        ranked = ranked_total,
        post_passes_filter,
        post_nonzero_signal,
        post_min_trades,
        min_trades_required = min_trades,
        pre_prop_firm,
        post_prop_firm,
        rejected_by_correlation,
        portfolio_size = portfolio.len(),
        "candidate funnel — how many genes survived each gate"
    );
    let (mut validation_gates, canonical_backtest_artifacts, walkforward_validation_artifacts) =
        build_discovery_validation_artifacts(
            &portfolio,
            &portfolio_signals,
            features,
            ohlcv,
            config,
        )?;
    if let Some(pf) = config.prop_firm_gate.as_ref() {
        validation_gates.prop_firm_window_passed = !portfolio.is_empty();
        validation_gates.prop_firm_window_count = pf.n_windows;
        validation_gates.prop_firm_window_pass_rate = if portfolio_pass_rates.is_empty() {
            0.0
        } else {
            portfolio_pass_rates.iter().sum::<f64>() / portfolio_pass_rates.len() as f64
        };
    }
    // 2026-05-26: walkforward + CPCV stages — Strict mode runs these as gates,
    // PropFirm mode uses them as informational. Either way the funnel records
    // pass/fail so the operator can see whether a non-empty portfolio later
    // got dropped at the walkforward stage. The validation_gates bool fields
    // are the canonical pass/fail signal.
    let portfolio_size = portfolio.len();
    let walkforward_pass = if validation_gates.walkforward_passed {
        portfolio_size
    } else {
        0
    };
    funnel.record_stage("passed_walkforward", portfolio_size, walkforward_pass);
    let cpcv_pass = if validation_gates.cpcv_passed {
        walkforward_pass
    } else {
        0
    };
    funnel.record_stage("passed_cpcv", walkforward_pass, cpcv_pass);
    // For PropFirm mode the canonical export-ready signal is
    // `prop_firm_window_passed`; for Strict mode it's both walkforward + cpcv
    // passed. `is_portfolio_export_ready()` handles both — so the final stage
    // count is the portfolio size when ready, else 0.
    let export_ready = if validation_gates.is_portfolio_export_ready() {
        portfolio_size
    } else {
        0
    };
    funnel.record_stage("export_ready", portfolio_size, export_ready);

    progress_fn(DiscoveryProgress::Completed {
        candidate_count: ranked_candidate_genes.len(),
        filtered_count,
        portfolio_size: portfolio.len(),
    });

    // 2026-05-26: finalize funnel with outcome label. The caller saves the
    // file next to the portfolio JSON — that's where the file lives in the
    // production layout.
    let outcome = if portfolio.is_empty() {
        "no_candidates"
    } else if export_ready > 0 {
        "exported"
    } else {
        "failed"
    };
    funnel.finalize(outcome);

    Ok(DiscoveryResult {
        portfolio,
        candidates: ranked_candidate_genes,
        quality_metrics,
        logged_trades,
        effective_feature_names,
        validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: Some(funnel.clone()),
    })
}

fn candidate_truncation_limit(requested: usize, available: usize) -> usize {
    if available == 0 {
        0
    } else if requested == 0 {
        available
    } else {
        requested.min(available)
    }
}

fn min_trades_required(timestamps: &[i64], min_trades_per_day: f64, n_rows: usize) -> usize {
    if timestamps.is_empty() {
        let days = (n_rows as f64 / 1440.0).max(1.0);
        return (days * min_trades_per_day).ceil() as usize;
    }
    let mut days = HashSet::new();
    for ts in timestamps {
        if let Some(dt) = Utc.timestamp_millis_opt(*ts).single()
            && dt.weekday().num_days_from_monday() < 5
        {
            let key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
            days.insert(key);
        }
    }
    let day_count = days.len().max(1) as f64;
    (day_count * min_trades_per_day).ceil() as usize
}

/// DS-2: Spearman rank correlation for i8 signals.
/// For discrete values (-1, 0, 1), ranks ties by mean rank. Detects monotonic (non-linear) dependency.
fn spearman_corr_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    // For i8 with only 3 distinct values, compute rank as fractional rank
    // mean_rank(v) = (first_idx + last_idx) / 2 over sorted positions
    let rank_of = |vals: &[i8], v: i8| -> f64 {
        let count = vals[..n].iter().filter(|&&x| x == v).count() as f64;
        let before = vals[..n].iter().filter(|&&x| x < v).count() as f64;
        before + (count + 1.0) / 2.0
    };
    let ranks_a: Vec<f64> = a[..n].iter().map(|&v| rank_of(&a[..n], v)).collect();
    let ranks_b: Vec<f64> = b[..n].iter().map(|&v| rank_of(&b[..n], v)).collect();
    let mean_a: f64 = ranks_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = ranks_b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut denom_a = 0.0_f64;
    let mut denom_b = 0.0_f64;
    for i in 0..n {
        let da = ranks_a[i] - mean_a;
        let db = ranks_b[i] - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a <= 1e-12 || denom_b <= 1e-12 {
        return 0.0;
    }
    num / (denom_a.sqrt() * denom_b.sqrt())
}

fn pearson_corr_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..n {
        sum_a += a[i] as f64;
        sum_b += b[i] as f64;
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;
    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a <= 1e-12 || denom_b <= 1e-12 {
        return 0.0;
    }
    num / (denom_a.sqrt() * denom_b.sqrt())
}

pub fn ensure_portfolio_export_ready(result: &DiscoveryResult) -> Result<()> {
    if result.validation_gates.is_portfolio_export_ready() {
        return Ok(());
    }
    anyhow::bail!(
        "Portfolio export requires passing validation gates (walkforward_passed={} cpcv_passed={}). \
         Lower the walk-forward splits or disable CPCV in config.yaml and re-run.",
        result.validation_gates.walkforward_passed,
        result.validation_gates.cpcv_passed
    );
}

fn build_portfolio_exports<'a>(
    portfolio: &'a [Gene],
    feature_names: &'a [String],
) -> Vec<GeneExport<'a>> {
    let mut exports = Vec::new();
    for gene in portfolio {
        let mut names = Vec::new();
        for idx in &gene.indices {
            if let Some(name) = feature_names.get(*idx) {
                names.push(name.as_str());
            }
        }
        exports.push(GeneExport {
            strategy_id: &gene.strategy_id,
            indicators: names,
            indices: gene.indices.clone(),
            weights: gene.weights.clone(),
            long_threshold: gene.long_threshold,
            short_threshold: gene.short_threshold,
            fitness: gene.fitness,
            sharpe_ratio: gene.sharpe_ratio,
            win_rate: gene.win_rate,
            tp_pips: gene.tp_pips,
            sl_pips: gene.sl_pips,
        });
    }
    exports
}

pub fn save_portfolio_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    ensure_portfolio_export_ready(result)?;
    let exports = build_portfolio_exports(&result.portfolio, &result.effective_feature_names);
    write_json_atomic(path, &exports)
}

pub fn save_quality_report_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    write_json_atomic(path, &result.quality_metrics)
}

/// 2026-05-26 operator directive (dual-mode product): save the 16-stage
/// rejection funnel as `<portfolio_stem>_funnel.json` next to the portfolio
/// JSON. The funnel is the operator's debug artifact for "why did the
/// portfolio come out empty?" — without it the answer is "look at the logs",
/// which doesn't survive across runs. No-op if the result has no funnel
/// (only the case if the GA panicked before the FunnelProfile was created).
pub fn save_funnel_json(
    portfolio_json_path: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<()> {
    let path = portfolio_json_path.as_ref();
    if let Some(ref funnel) = result.funnel_profile {
        funnel
            .save_next_to(path)
            .with_context(|| format!("saving funnel JSON next to {}", path.display()))?;
    }
    Ok(())
}

pub fn save_trade_log_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    write_json_atomic(path, &result.logged_trades)
}

fn artifact_filename_for_strategy_hash(strategy_hash: &str, fallback_index: usize) -> String {
    let cleaned: String = strategy_hash
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            _ => '_',
        })
        .collect();
    if cleaned.is_empty() {
        format!("strategy_{fallback_index:04}.json")
    } else {
        format!("{cleaned}.json")
    }
}

pub fn save_canonical_backtest_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.canonical_backtest_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create canonical backtest dir {}", dir.display()))?;
    for (idx, artifact) in result.canonical_backtest_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(&artifact.scope.strategy_hash, idx);
        write_canonical_backtest_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.canonical_backtest_artifacts.len())
}

pub fn save_walkforward_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.walkforward_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create walk-forward validation dir {}", dir.display()))?;
    for (idx, artifact) in result.walkforward_validation_artifacts.iter().enumerate() {
        let strategy_hash = artifact
            .scope
            .strategy_hash
            .as_deref()
            .unwrap_or("portfolio");
        let file_name = artifact_filename_for_strategy_hash(strategy_hash, idx);
        write_walkforward_validation_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.walkforward_validation_artifacts.len())
}

pub fn save_forward_test_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.forward_test_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create forward-test validation dir {}", dir.display()))?;
    for (idx, artifact) in result.forward_test_validation_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(&artifact.scope.strategy_hash, idx);
        write_forward_test_validation_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.forward_test_validation_artifacts.len())
}

/// Persist a focused promotion-readiness summary at `path` derived
/// from the discovery result. The summary is the same per-kind
/// evidence + missing-kinds + producer-side-completeness payload that
/// already lives on `DiscoveryRunProfile` (Phase 49), but written to
/// its own file so operators / UI scrapers can poll it without
/// parsing the full profile JSON.
pub fn save_promotion_summary_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    #[derive(Serialize)]
    struct PromotionSummary<'a> {
        validation_evidence_hashes: &'a DiscoveryPerKindEvidenceHashes,
        validation_evidence_complete: bool,
        validation_evidence_missing_kinds: Vec<&'static str>,
        producer_side_complete: bool,
        check_summary: Vec<(&'static str, &'static str)>,
        determinism_policy: neoethos_core::contracts::DeterminismPolicy,
    }
    let hashes = discovery_per_kind_evidence_hashes(result)?;
    let summary = PromotionSummary {
        producer_side_complete: hashes.all_producer_kinds_present(),
        check_summary: hashes.check_summary(),
        validation_evidence_complete: hashes.all_present(),
        validation_evidence_missing_kinds: hashes.missing_kinds(),
        validation_evidence_hashes: &hashes,
        determinism_policy: crate::genetic::current_determinism_policy(),
    };
    write_json_atomic(path, &summary)
}

pub fn save_prop_firm_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.prop_firm_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create prop-firm validation dir {}", dir.display()))?;
    for (idx, artifact) in result.prop_firm_validation_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(&artifact.scope.strategy_hash, idx);
        write_prop_firm_risk_validation_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.prop_firm_validation_artifacts.len())
}

/// Translate a [`DiscoveryResult`] into a typed
/// [`neoethos_core::contracts::LiveValidationEvidence`] record so a live
/// bridge can call `LiveExecutionContract::validate_evidence` without
/// re-deriving any pass/fail logic itself. The mapping is:
///
/// - `walkforward_passed` / `cpcv_passed` come straight from
///   `result.validation_gates`.
/// - `forward_test_passed` is `Some(true)` only when the result carries
///   at least one forward-test artifact AND every artifact reports a
///   non-zero trade count with strictly positive net profit.
///   `Some(false)` is returned when artifacts exist but at least one
///   fails the rule, and `None` when no artifact was produced (the live
///   bridge will treat that as missing evidence if it requires the
///   gate).
/// - `prop_firm_passed` aggregates the per-strategy
///   [`PropFirmRiskValidationArtifactFile::summary.all_rules_passed`]
///   flags: `Some(true)` when every persisted prop-firm artifact passes,
///   `Some(false)` when at least one fails, and `None` when no
///   prop-firm artifact was produced (the live bridge will treat that
///   as missing evidence whenever the gate is required).
/// - `live_sim_runtime_model_hash` stays `None` until a live-execution
///   simulator is wired into the discovery pipeline.
pub fn live_validation_evidence_from_discovery(result: &DiscoveryResult) -> LiveValidationEvidence {
    let forward_test_passed = if result.forward_test_validation_artifacts.is_empty() {
        None
    } else {
        let all_pass = result
            .forward_test_validation_artifacts
            .iter()
            .all(|artifact| {
                artifact.summary.metrics.trade_count > 0
                    && artifact.summary.metrics.net_profit > 0.0
            });
        Some(all_pass)
    };
    let prop_firm_passed = if result.prop_firm_validation_artifacts.is_empty() {
        None
    } else {
        let all_pass = result
            .prop_firm_validation_artifacts
            .iter()
            .all(|artifact| artifact.summary.all_rules_passed);
        Some(all_pass)
    };
    LiveValidationEvidence {
        walkforward_passed: result.validation_gates.walkforward_passed,
        cpcv_passed: result.validation_gates.cpcv_passed,
        forward_test_passed,
        prop_firm_passed,
        live_sim_runtime_model_hash: None,
    }
}

/// Build a [`ValidationEvidenceManifest`] from the persisted discovery
/// artifacts. The helper computes one stable hash per artifact kind by
/// hashing the full vector of per-strategy artifacts; an empty vector
/// produces an empty hash, which causes
/// [`ValidationEvidenceManifest::validate`] to surface a typed
/// `MissingValidationEvidence` error naming the missing kind.
///
/// Today this always returns an error for the
/// `live_execution_simulation_hash` kind because `DiscoveryResult` does
/// not yet carry live-sim artifacts (the simulator is still deferred).
/// Callers that want a partial manifest for diagnostic display should
/// use the per-kind helpers below; callers that need a fully-validated
/// manifest must wait until the live-execution simulator lands.
pub fn discovery_validation_evidence_manifest(
    result: &DiscoveryResult,
) -> Result<ValidationEvidenceManifest> {
    let canonical = hash_validation_artifacts(&result.canonical_backtest_artifacts)?;
    let walkforward = hash_validation_artifacts(&result.walkforward_validation_artifacts)?;
    let forward_test = hash_validation_artifacts(&result.forward_test_validation_artifacts)?;
    let prop_firm = hash_validation_artifacts(&result.prop_firm_validation_artifacts)?;
    // Live-execution simulation artifacts are not yet emitted by the
    // discovery pipeline — propagate as the empty string so the
    // manifest's `validate()` rejects with the typed
    // `MissingValidationEvidence("live_execution_simulation_hash")`
    // variant rather than silently filling a placeholder.
    let live_sim = String::new();
    ValidationEvidenceManifest::new(canonical, walkforward, forward_test, live_sim, prop_firm)
        .map_err(|err| anyhow::anyhow!(err.to_string()))
}

/// Build a [`ValidationEvidenceManifest`] without enforcing the
/// always-missing `live_execution_simulation_hash` gate. Producer-side
/// kinds that are missing still return an error — the relaxation only
/// covers the simulator hash that is structurally absent until the
/// simulator lands. Operators / UI layers can use this for diagnostic
/// display ("which producer-side kinds shipped?") without tripping on
/// the structural live-sim absence.
pub fn discovery_validation_evidence_manifest_excluding_live_sim(
    result: &DiscoveryResult,
) -> Result<ValidationEvidenceManifest> {
    let canonical = hash_validation_artifacts(&result.canonical_backtest_artifacts)?;
    let walkforward = hash_validation_artifacts(&result.walkforward_validation_artifacts)?;
    let forward_test = hash_validation_artifacts(&result.forward_test_validation_artifacts)?;
    let prop_firm = hash_validation_artifacts(&result.prop_firm_validation_artifacts)?;
    let live_sim = "deferred:live_execution_simulator_not_wired".to_string();
    ValidationEvidenceManifest::new(canonical, walkforward, forward_test, live_sim, prop_firm)
        .map_err(|err| anyhow::anyhow!(err.to_string()))
}

/// Per-kind helper that returns `Some(hash)` when the artifact vector
/// is non-empty and `None` otherwise. Operator/UI layers can use this
/// to build a diagnostic view ("forward-test artifact present, live-sim
/// missing") without forcing a full manifest validation.
pub fn discovery_per_kind_evidence_hashes(
    result: &DiscoveryResult,
) -> Result<DiscoveryPerKindEvidenceHashes> {
    Ok(DiscoveryPerKindEvidenceHashes {
        canonical_backtest: optional_hash_validation_artifacts(
            &result.canonical_backtest_artifacts,
        )?,
        walkforward: optional_hash_validation_artifacts(&result.walkforward_validation_artifacts)?,
        forward_test: optional_hash_validation_artifacts(
            &result.forward_test_validation_artifacts,
        )?,
        prop_firm: optional_hash_validation_artifacts(&result.prop_firm_validation_artifacts)?,
        live_execution_simulation: None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveryPerKindEvidenceHashes {
    pub canonical_backtest: Option<String>,
    pub walkforward: Option<String>,
    pub forward_test: Option<String>,
    pub prop_firm: Option<String>,
    pub live_execution_simulation: Option<String>,
}

impl DiscoveryPerKindEvidenceHashes {
    /// Returns `true` only when every kind has a non-empty hash. The
    /// live-execution simulation hash is part of this check, so the
    /// summary will currently always return `false` until a simulator
    /// produces evidence.
    pub fn all_present(&self) -> bool {
        self.canonical_backtest.is_some()
            && self.walkforward.is_some()
            && self.forward_test.is_some()
            && self.prop_firm.is_some()
            && self.live_execution_simulation.is_some()
    }

    /// Returns `true` when every producer-side kind (canonical,
    /// walkforward, forward-test, prop-firm) is present, ignoring the
    /// always-missing `live_execution_simulation` hash. Operators that
    /// want to gauge producer-side completeness without waiting for the
    /// simulator can use this instead of `all_present()`.
    pub fn all_producer_kinds_present(&self) -> bool {
        self.canonical_backtest.is_some()
            && self.walkforward.is_some()
            && self.forward_test.is_some()
            && self.prop_firm.is_some()
    }

    /// Returns one `(kind_name, status)` tuple per validation kind,
    /// where `status` is `"present"` or `"missing"`. Render directly
    /// in operator-facing log lines / UI tables without re-deriving
    /// per-kind logic.
    pub fn check_summary(&self) -> Vec<(&'static str, &'static str)> {
        let label = |opt: &Option<String>| if opt.is_some() { "present" } else { "missing" };
        vec![
            ("canonical_backtest", label(&self.canonical_backtest)),
            ("walkforward", label(&self.walkforward)),
            ("forward_test", label(&self.forward_test)),
            ("prop_firm", label(&self.prop_firm)),
            (
                "live_execution_simulation",
                label(&self.live_execution_simulation),
            ),
        ]
    }

    /// Returns the list of kinds that have no hash on this profile.
    /// Operators / UI layers can render this directly without parsing
    /// `MissingValidationEvidence` strings.
    pub fn missing_kinds(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.canonical_backtest.is_none() {
            missing.push("canonical_backtest");
        }
        if self.walkforward.is_none() {
            missing.push("walkforward");
        }
        if self.forward_test.is_none() {
            missing.push("forward_test");
        }
        if self.prop_firm.is_none() {
            missing.push("prop_firm");
        }
        if self.live_execution_simulation.is_none() {
            missing.push("live_execution_simulation");
        }
        missing
    }
}

fn hash_validation_artifacts<T: Serialize>(artifacts: &[T]) -> Result<String> {
    if artifacts.is_empty() {
        Ok(String::new())
    } else {
        stable_json_hash(artifacts)
    }
}

fn optional_hash_validation_artifacts<T: Serialize>(artifacts: &[T]) -> Result<Option<String>> {
    if artifacts.is_empty() {
        Ok(None)
    } else {
        stable_json_hash(artifacts).map(Some)
    }
}

pub fn build_discovery_profile(
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> DiscoveryRunProfile {
    let validation_evidence_hashes =
        discovery_per_kind_evidence_hashes(result).unwrap_or_else(|_| {
            DiscoveryPerKindEvidenceHashes {
                canonical_backtest: None,
                walkforward: None,
                forward_test: None,
                prop_firm: None,
                live_execution_simulation: None,
            }
        });
    DiscoveryRunProfile {
        timeframe_label: config.timeframe_label.clone(),
        population: config.population,
        generations: config.generations,
        max_indicators: config.max_indicators,
        candidate_count_target: config.candidate_count,
        portfolio_size_target: config.portfolio_size,
        max_rows: row_cap_for_config(config),
        max_runtime_hours: config.max_hours,
        corr_threshold: config.corr_threshold,
        min_trades_per_day: config.min_trades_per_day,
        walkforward_splits: config.walkforward_splits,
        embargo_minutes: config.embargo_minutes,
        enable_cpcv: config.enable_cpcv,
        cpcv_n_splits: config.cpcv_n_splits,
        cpcv_n_test_groups: config.cpcv_n_test_groups,
        cpcv_embargo_pct: config.cpcv_embargo_pct,
        cpcv_purge_pct: config.cpcv_purge_pct,
        cpcv_min_phi: config.cpcv_min_phi,
        filters: DiscoveryFilterProfile {
            max_dd: config.filtering.max_dd,
            min_profit: config.filtering.min_profit,
            min_trades: config.filtering.min_trades,
            min_sharpe: config.filtering.min_sharpe,
            min_win_rate: config.filtering.min_win_rate,
            min_profit_factor: config.filtering.min_profit_factor,
            min_positive_months: config.filtering.min_positive_months,
            min_trades_per_month: config.filtering.min_trades_per_month,
            min_monthly_return_pct: config.filtering.min_monthly_return_pct,
            opportunistic_enabled: config.filtering.use_opportunistic_candidates
                && config.filtering.opportunistic_enabled,
            opportunistic_min_positive_months: config.filtering.opportunistic_min_positive_months,
            opportunistic_min_trades_per_month: config.filtering.opportunistic_min_trades_per_month,
            opportunistic_min_trade_return_pct: config.filtering.opportunistic_min_trade_return_pct,
            opportunistic_max_dd: config.filtering.opportunistic_max_dd,
            log_trades: config.filtering.log_trades,
            trade_log_max: config.filtering.trade_log_max,
        },
        candidates_observed: result.candidates.len(),
        portfolio_observed: result.portfolio.len(),
        quality_metrics_observed: result.quality_metrics.len(),
        logged_trade_sets: result.logged_trades.len(),
        walkforward_passed: result.validation_gates.walkforward_passed,
        cpcv_passed: result.validation_gates.cpcv_passed,
        canonical_backtest_artifacts_observed: result.validation_gates.canonical_backtest_artifacts,
        walkforward_validation_artifacts_observed: result
            .validation_gates
            .walkforward_validation_artifacts,
        forward_test_validation_artifacts_observed: result.forward_test_validation_artifacts.len(),
        prop_firm_validation_artifacts_observed: result.prop_firm_validation_artifacts.len(),
        cpcv_fold_count: result.validation_gates.cpcv_fold_count,
        cpcv_profitable_fold_ratio: result.validation_gates.cpcv_profitable_fold_ratio,
        validation_temporal_contract_hash: result.validation_gates.temporal_contract_hash.clone(),
        prefilter_top_k: config.runtime_overrides.prefilter_top_k,
        prefilter_insample_frac: config.runtime_overrides.resolved_prefilter_insample_frac(),
        funnel_stage1_pct: config.runtime_overrides.resolved_funnel_stage1_pct(),
        validation_evidence_hashes: validation_evidence_hashes.clone(),
        validation_evidence_complete: validation_evidence_hashes.all_present(),
        validation_evidence_missing_kinds: validation_evidence_hashes
            .missing_kinds()
            .into_iter()
            .map(str::to_string)
            .collect(),
        determinism_policy: crate::genetic::current_determinism_policy(),
    }
}

pub fn save_discovery_profile_json(
    path: impl AsRef<Path>,
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> Result<()> {
    write_json_atomic(path, &build_discovery_profile(config, result))
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;
