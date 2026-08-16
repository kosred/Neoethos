use crate::artifact_io::{stable_json_hash, write_json_atomic};
use crate::eval::{BacktestMetrics, fast_evaluate_strategy_core, simulate_trades_core};
use crate::genetic::strategy_gene::EvaluationConfig;
use crate::genetic::{
    Gene, SmcGateArrays, build_smc_arrays, evolve_search_with_progress_and_limits,
    month_day_indices, signals_and_confidence_for_gene_full, signals_for_gene_full,
    signals_for_gene_full_with_smc, validation_genes_population_gathered,
};
use crate::quality::{StrategyMetrics, StrategyQualityAnalyzer, Trade};
use crate::validation::{
    CanonicalBacktestArtifactFile, CanonicalBacktestScope, CombinatorialPurgedCV, ForwardTestInput,
    ForwardTestValidationArtifactFile, ForwardTestValidationScope, PropFirmRiskInput,
    PropFirmRiskRules, PropFirmRiskValidationArtifactFile, PropFirmRiskValidationScope,
    WalkforwardSummary, WalkforwardValidationArtifactFile, WalkforwardValidationScope,
    compute_forward_test_summary, compute_prop_firm_risk_summary,
    write_canonical_backtest_artifact_atomic, write_forward_test_validation_artifact_atomic,
    write_prop_firm_risk_validation_artifact_atomic, write_walkforward_validation_artifact_atomic,
};
use anyhow::{Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use neoethos_core::contracts::{
    DeterminismPolicy, LiveValidationEvidence, TemporalFeatureContract, ValidationEvidenceManifest,
};
use neoethos_data::{FeatureFrame, Ohlcv};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Typed runtime knobs that previously lived only in `NEOETHOS_BOT_*` env vars.
///
/// These values change *production* discovery semantics (which features are
/// kept, how much data the stage-1 funnel sees, what counts as in-sample for
/// the prefilter), so they belong in typed config rather than ambient env
/// state. These are configured via `models.discovery_runtime` (typed config)
/// and resolved by [`DiscoveryRuntimeOverrides::from_settings`], which is the
/// ONLY constructor that reads operator input.
///
/// 2026-08-10: the legacy `from_env()` reader was deleted. It carried six
/// `NEOETHOS_BOT_*` names — `PREFILTER_TOP_K`, `PREFILTER_INSAMPLE`,
/// `PREFILTER_MIN_PER_TF`, `FUNNEL_STAGE1_PCT`, `FUNNEL_STAGE1_WINDOW`,
/// `MIN_HISTORY_YEARS` — had zero production callers, and was "retained for
/// reference": a second, invisible way to set the same knobs. `prefilter_top_k`
/// is the exact key `shipped_config_matches_defaults.rs` exists to protect, and
/// an env var that silently lowers it to 50 collapses the base feature set from
/// 217 columns to roughly 64 with the SMC, session and footprint families dying
/// first. One config, no env.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
    /// Parse the `models.discovery_runtime.stage1_window` config string. An
    /// unrecognised value returns `None` and the caller keeps the default —
    /// which it says out loud rather than substituting in silence.
    fn from_config_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "most_recent" | "recent" | "tail" => Some(Self::MostRecent),
            "earliest" | "head" | "oldest" => Some(Self::Earliest),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DiscoveryRuntimeOverrides {
    /// FLOOR on the number of features kept after the in-sample correlation
    /// prefilter. `0` disables the prefilter entirely.
    ///
    /// Since 2026-08-10 the effective pool is derived from GA capacity by
    /// [`resolve_prefilter_top_k`] and this value is its lower bound. See that
    /// function for why a constant against a hardware-sized cube was the same
    /// defect class as sizing memory from a user parameter, and for the numbers
    /// that refuse the three obvious alternatives.
    pub prefilter_top_k: usize,
    /// Fraction of rows treated as in-sample when ranking features. Must be
    /// strictly positive and at most `1.0`.
    pub prefilter_insample_frac: f64,
    /// Minimum number of features to force-keep from EACH present higher
    /// timeframe group during the prefilter, on top of the global
    /// `prefilter_top_k`. The correlation ranking is against the BASE
    /// timeframe's 1-bar forward return, against which a near-constant
    /// higher-TF indicator scores ~0 — so without this quota the global
    /// top-K discards every multi-TF feature and the GA's multi-TF seed
    /// templates find no `H1_`/`H4_`/… prefixes. `0` = legacy behaviour.
    pub prefilter_min_per_timeframe: usize,
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
            // 240, matching config.yaml AND
            // `neoethos_core::config::DiscoveryRuntimeConfig::default()`. The
            // three had drifted — code 50 / root yaml 240 / desktop yaml 50 —
            // so a run's indicator pool was five times smaller or larger
            // depending on whether a config file could be read, and no
            // artifact recorded which branch ran. The other two sites were
            // fixed by the indicator-vocabulary workflow and are pinned by
            // `crates/neoethos-core/tests/shipped_config_matches_defaults.rs`;
            // this is the third, applied from
            // `docs/pending-edits-forbidden-territory.md` §2.
            prefilter_top_k: 240,
            prefilter_insample_frac: 0.80,
            prefilter_min_per_timeframe: 6,
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
            // who want the strict 10y gate back set
            // `models.discovery_runtime.min_history_years: 10` in config.
            // (Before 2026-08-10 this comment named an env var; that reader is
            // deleted — there is one place to set this and it is the config.)
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
    /// The ONE constructor that reads operator input: `models.discovery_runtime`.
    ///
    /// There is no env reader. An out-of-range value keeps the default, and
    /// says so by name with both numbers — a knob that quietly reverts is
    /// indistinguishable from a knob that was honoured.
    pub(crate) fn from_settings(settings: &neoethos_core::Settings) -> Self {
        let cfg = &settings.models.discovery_runtime;
        let mut overrides = Self::default();
        overrides.prefilter_top_k = cfg.prefilter_top_k;
        if cfg.prefilter_insample_frac.is_finite()
            && cfg.prefilter_insample_frac > 0.0
            && cfg.prefilter_insample_frac <= 1.0
        {
            overrides.prefilter_insample_frac = cfg.prefilter_insample_frac;
        } else {
            tracing::warn!(
                target: "neoethos_search::config_resolution",
                key = "models.discovery_runtime.prefilter_insample_frac",
                configured = cfg.prefilter_insample_frac,
                effective = overrides.prefilter_insample_frac,
                "configured value is not a fraction in (0, 1] — the DEFAULT is in force, \
                 not your number"
            );
        }
        overrides.prefilter_min_per_timeframe = cfg.prefilter_min_per_timeframe;
        if cfg.funnel_stage1_pct.is_finite() {
            let clamped = cfg.funnel_stage1_pct.clamp(0.01, 1.0);
            if (clamped - cfg.funnel_stage1_pct).abs() > f64::EPSILON {
                tracing::warn!(
                    target: "neoethos_search::config_resolution",
                    key = "models.discovery_runtime.funnel_stage1_pct",
                    configured = cfg.funnel_stage1_pct,
                    effective = clamped,
                    "configured value is outside [0.01, 1.0] and was CLAMPED — stage 1 sees \
                     a different slice of the data than you asked for"
                );
            }
            overrides.funnel_stage1_pct = clamped;
        } else {
            tracing::warn!(
                target: "neoethos_search::config_resolution",
                key = "models.discovery_runtime.funnel_stage1_pct",
                configured = cfg.funnel_stage1_pct,
                effective = overrides.funnel_stage1_pct,
                "configured value is non-finite — the DEFAULT is in force"
            );
        }
        match Stage1Window::from_config_str(&cfg.stage1_window) {
            Some(window) => overrides.stage1_window = window,
            None => tracing::warn!(
                target: "neoethos_search::config_resolution",
                key = "models.discovery_runtime.stage1_window",
                configured = %cfg.stage1_window,
                effective = ?overrides.stage1_window,
                "unrecognised stage1_window — accepted values are \
                 most_recent|recent|tail and earliest|head|oldest. The DEFAULT is in force"
            ),
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

    pub(crate) fn resolved_prefilter_insample_frac(&self) -> f64 {
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

/// Name the winner of every knob that exists twice in this config, with both
/// values, once per run.
///
/// Called from [`DiscoveryConfig::from_settings`]. It changes no behaviour — it
/// removes the ability for a duplicate to be edited invisibly. The pairs here
/// are the ones whose deciding read lives in `neoethos-search`; the shape is
/// deliberately copied from `session_spread_pips()` above, which the 2026-08-09
/// knob pass names as the honest pattern every other twin should look like.
///
/// The TRAILING pair is gone from this function (2026-08-10, audit #206) —
/// not silenced, RESOLVED: the `risk.trailing_*` four were deleted from
/// `RiskConfig`, so `models.exit_policy.*` is now the only place the trail can
/// be set and there is no second value to name. A store that still carries the
/// old keys is told so by name, with the rename, by `RETIRED_KEYS` in
/// `neoethos-core/src/config.rs`.
fn resolve_and_log_duplicate_knobs(settings: &neoethos_core::Settings) {
    // ── COST 💰 ───────────────────────────────────────────────────────────
    //
    // `risk.*` DECIDES, unconditionally. `models.eval_runtime.spread_pips` /
    // `.commission_per_trade` reach nothing in a discovery run.
    //
    // `DiscoveryConfig::from_settings` computes `evaluation_spread_pips` and
    // `evaluation_commission_per_trade` from `risk.*` and passes them as the
    // EXPLICIT per-call override into `EvaluationConfig::for_symbol` →
    // `infer_market_cost_profile`, which is step (1) of a four-step chain whose
    // step (2) is the eval_runtime pair. Step (1) is filled on every discovery
    // run and `run_discovery_cycle` refuses a non-finite override, so step (2)
    // is unreachable from here. This matters because the Settings screen renders
    // the eval_runtime pair as `cost.spread_pips` / `cost.commission_per_trade`
    // WITH tuning presets: the surface the operator is offered is the one that
    // loses.
    let eval_cost = &settings.models.eval_runtime;
    if let Some(shadow_spread) = eval_cost.spread_pips {
        tracing::warn!(
            target: "neoethos_search::config_resolution",
            winner = "risk.backtest_spread_pips + risk.slippage_pips",
            loser = "models.eval_runtime.spread_pips",
            effective_spread_pips = settings.risk.backtest_spread_pips.max(0.0)
                + settings.risk.slippage_pips.max(0.0),
            ignored_spread_pips = shadow_spread,
            "SPREAD IS SET TWICE. Discovery charges the risk.* number; \
             models.eval_runtime.spread_pips (what the Settings screen calls \
             cost.spread_pips) is ignored on every discovery run."
        );
    }
    if let Some(shadow_commission) = eval_cost.commission_per_trade {
        tracing::warn!(
            target: "neoethos_search::config_resolution",
            winner = "risk.commission_per_lot (broker metadata first)",
            loser = "models.eval_runtime.commission_per_trade",
            ignored_commission_per_trade = shadow_commission,
            "COMMISSION IS SET TWICE. Discovery charges the broker-authoritative or \
             risk.* number; models.eval_runtime.commission_per_trade (what the Settings \
             screen calls cost.commission_per_trade) is ignored on every discovery run."
        );
    }

    // ── SYMBOL / ACCOUNT CURRENCY 💰 ──────────────────────────────────────
    //
    // `system.*` DECIDES whenever non-empty, and `from_settings` above reads
    // ONLY `system.*`. Two `symbol:` keys ~1300 lines apart in the same file is
    // how a run ends up measuring the wrong instrument's pip value.
    if let Some(shadow_symbol) = eval_cost.symbol.as_deref().map(str::trim) {
        if !shadow_symbol.is_empty() && !shadow_symbol.eq_ignore_ascii_case(&settings.system.symbol)
        {
            tracing::warn!(
                target: "neoethos_search::config_resolution",
                winner = "system.symbol",
                loser = "models.eval_runtime.symbol",
                effective_symbol = %settings.system.symbol,
                ignored_symbol = %shadow_symbol,
                "SYMBOL IS SET TWICE AND THE TWO DISAGREE. Discovery evaluates \
                 system.symbol."
            );
        }
    }
    if let Some(shadow_ccy) = eval_cost.account_currency.as_deref().map(str::trim) {
        if !shadow_ccy.is_empty()
            && !shadow_ccy.eq_ignore_ascii_case(&settings.system.account_currency)
        {
            tracing::warn!(
                target: "neoethos_search::config_resolution",
                winner = "system.account_currency",
                loser = "models.eval_runtime.account_currency",
                effective_account_currency = %settings.system.account_currency,
                ignored_account_currency = %shadow_ccy,
                "ACCOUNT CURRENCY IS SET TWICE AND THE TWO DISAGREE. Discovery converts \
                 pip value into system.account_currency; a wrong currency silently \
                 rescales every result."
            );
        }
    }
}

/// Print each admission/export gate's EFFECTIVE value beside the Rust `Default`
/// it came from, once per run.
///
/// Why the Default is printed and not the file: the four config surfaces
/// (`Default`, repo `config.yaml`, the desktop seed, and
/// `%LOCALAPPDATA%\neoethos\config.yaml`) disagree on these keys, and a run has
/// no way to know which file it was handed — that resolution is logged at load
/// time by `Settings::load`. What a run CAN say is "this gate is off and the
/// Default says on", which is exactly the class of surprise §3 of the
/// 2026-08-09 knob pass describes: a gate the operator deliberately disarmed
/// silently re-arming after a reinstall, or an install that lost a key keeping
/// a gate disarmed with no diff to explain why exports stopped.
///
/// This function changes NOTHING. It does not turn a gate on. Turning any of
/// these on changes what the search admits, and that is the operator's call.
fn log_gate_states(settings: &neoethos_core::Settings) {
    let d = neoethos_core::Settings::default();
    let m = &settings.models;
    let dm = &d.models;

    macro_rules! gate_bool {
        ($key:literal, $eff:expr, $def:expr, $what:literal) => {{
            let eff: bool = $eff;
            let def: bool = $def;
            if eff == def {
                tracing::info!(
                    target: "neoethos_search::gate_state",
                    key = $key,
                    effective = eff,
                    rust_default = def,
                    what_it_gates = $what,
                    "gate state (agrees with its Rust default)"
                );
            } else {
                tracing::warn!(
                    target: "neoethos_search::gate_state",
                    key = $key,
                    effective = eff,
                    rust_default = def,
                    what_it_gates = $what,
                    "GATE DIFFERS FROM ITS RUST DEFAULT. If this was a deliberate \
                     decision it is holding; if a config file lost or gained this key, \
                     this run just changed what it admits with no diff to explain it."
                );
            }
        }};
    }

    gate_bool!(
        "models.require_walkforward_for_export",
        m.require_walkforward_for_export,
        dm.require_walkforward_for_export,
        "hard out-of-sample export gate: false lets a portfolio export without \
         clearing walk-forward."
    );
    gate_bool!(
        "models.enable_cpcv",
        m.enable_cpcv,
        dm.enable_cpcv,
        "the SEARCH's combinatorial-purged-CV admission gate (not the training-side \
         models.ml_cpcv_enabled): false promotes a portfolio with no purged OOS \
         validation at all."
    );
    gate_bool!(
        "models.ml_cpcv_enabled",
        m.ml_cpcv_enabled,
        dm.ml_cpcv_enabled,
        "the TRAINING-side CPCV, a different gate that shares the letters. Disarming \
         the wrong one of the two admits candidates that never passed purged CV."
    );
    gate_bool!(
        "models.regime_router_enabled",
        m.regime_router_enabled,
        dm.regime_router_enabled,
        "per-regime routing of candidates."
    );
    gate_bool!(
        "models.l1_feature_selection_enabled",
        m.l1_feature_selection_enabled,
        dm.l1_feature_selection_enabled,
        "L1 feature selection: off means the full feature set reaches the model."
    );
    gate_bool!(
        "models.l1_feature_selection_per_regime",
        m.l1_feature_selection_per_regime,
        dm.l1_feature_selection_per_regime,
        "per-regime L1 feature selection."
    );
    gate_bool!(
        "system.multi_resolution_enabled",
        settings.system.multi_resolution_enabled,
        d.system.multi_resolution_enabled,
        "multi-timeframe resolution — the seed config calls this 'the pre-GA wall that \
         stopped combos completing on laptop AND VPS'."
    );
    gate_bool!(
        "risk.challenge_mode",
        settings.risk.challenge_mode,
        d.risk.challenge_mode,
        "prop-firm challenge mode. UNWIRED: domain::risk::RiskManager has no production \
         constructor, so this arms nothing today — it is retained as recorded intent."
    );
    gate_bool!(
        "risk.max_trades_per_day_enabled",
        settings.risk.max_trades_per_day_enabled,
        d.risk.max_trades_per_day_enabled,
        "the daily entry cap. Arming it live without arming it in the search means the \
         backtest that selected your strategies took entries live will refuse."
    );

    // ── the two money floors, printed as numbers 💰 ──────────────────────────
    if (m.prop_search_min_payoff_ratio - dm.prop_search_min_payoff_ratio).abs() > f64::EPSILON {
        tracing::warn!(
            target: "neoethos_search::gate_state",
            key = "models.prop_search_min_payoff_ratio",
            effective = m.prop_search_min_payoff_ratio,
            rust_default = dm.prop_search_min_payoff_ratio,
            "PAYOFF FLOOR DIFFERS FROM ITS RUST DEFAULT. 0.0 means the quality screen's \
             payoff criterion is OFF (it is guarded by `> 0.0`), leaving the screen as \
             net-expectancy plus the trade-count floors."
        );
    } else {
        tracing::info!(
            target: "neoethos_search::gate_state",
            key = "models.prop_search_min_payoff_ratio",
            effective = m.prop_search_min_payoff_ratio,
            "payoff floor in force"
        );
    }
    if (m.prop_firm_min_pass_rate - dm.prop_firm_min_pass_rate).abs() > f64::EPSILON {
        tracing::warn!(
            target: "neoethos_search::gate_state",
            key = "models.prop_firm_min_pass_rate",
            effective = m.prop_firm_min_pass_rate,
            rust_default = dm.prop_firm_min_pass_rate,
            "PROP-FIRM PASS-RATE FLOOR DIFFERS FROM ITS RUST DEFAULT. 0.0 = RANKING ONLY: \
             the window gate runs, ranks, and rejects nothing."
        );
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub timeframe_label: String,
    pub evaluation_symbol: String,
    pub evaluation_account_currency: String,
    pub evaluation_spread_pips: f64,
    /// ROUND-TRIP commission per lot, in account currency.
    ///
    /// The evaluators subtract this exactly once per closed trade, so a
    /// per-side broker quote must already have been doubled before it lands
    /// here — `from_settings` does that through
    /// [`crate::genetic::strategy_gene::round_trip_commission_per_lot`], gated
    /// on `risk.commission_per_lot_is_per_side`.
    pub evaluation_commission_per_trade: f64,
    /// Session-aware spread curve in pips, `[asian, overlap, late_ny]`,
    /// slippage already folded in — or `None` for a flat spread at every hour.
    ///
    /// The per-bar lookup has existed on both the CPU path (`eval.rs:843`) and
    /// the CUDA kernel (`prototype_b_population.cu:47`) for months and was
    /// populated ONLY under `#[cfg(test)]`: every production construction site
    /// left `session_spread_profile: None`. So the London open and 03:00 Tokyo
    /// were charged the same spread, and a strategy that only trades the Asian
    /// session was measured at a cost it would never get. `None` here keeps
    /// exactly that behaviour and the run says so out loud; `Some` turns the
    /// curve on for CPU and card alike with no kernel change.
    pub session_spread_pips: Option<[f64; 3]>,
    /// Round-trip cost band in pips, `(optimistic, pessimistic)`, that every
    /// reported result is measured against. See `RiskConfig::cost_band_pips`.
    pub cost_band_pips: Option<(f64, f64)>,
    /// Broker overnight financing, pips/night, from the symbol's metadata
    /// (`daily_swap_long_pips` / `daily_swap_short_pips`). Decision D
    /// (2026-08-09): a zero-swap backtest silently overstates every held
    /// position's edge — the search was buying carry it will never earn live.
    /// 0.0 only when the symbol has no swap metadata (logged loudly).
    pub swap_long_pips_per_day: f64,
    pub swap_short_pips_per_day: f64,
    /// Weekend kill zones — force-close before the weekend close and block
    /// Friday-late / Monday-open entries (`eval.rs:1537`, `:1654`).
    ///
    /// WIRED 2026-08-10 (audit #75/#217). This was the literal `true` in
    /// [`discovery_backtest_settings`], sitting between two fields that read
    /// `config.`. Live read `risk.kill_zones_enabled`
    /// (`live_trading.rs:732-735`) and the search read nothing, so the knob was
    /// ONE-SIDED: setting it to `false` could only make live hold through
    /// weekend gaps that no backtest in the artifact history had ever held
    /// through. It could never make live match a validated backtest, because no
    /// backtest could be run with kill zones off.
    ///
    /// Both sides now read the same `risk.kill_zones_enabled` (default `true`,
    /// `config.rs:671`), so the shipped behaviour is unchanged and the two sides
    /// can no longer disagree. Turning it OFF re-scores against a different
    /// simulator, and that is visible rather than silent: the value is part of
    /// the backtest policy hash (`DiscoveryBacktestPolicy::kill_zones_enabled`)
    /// and of the run profile, so artifacts produced on either side of the
    /// switch are distinguishable after the fact.
    pub kill_zones_enabled: bool,
    pub population: usize,
    /// When `true`, `run_search` raises the GA population to the card's fits
    /// ceiling (bounded to 16 384, never below `population`) and logs the
    /// resolved value as a selection-changing decision. `false` keeps
    /// `population` exactly. From `models.prop_search_population_auto`.
    pub population_auto: bool,
    pub generations: usize,
    pub max_indicators: usize,
    pub candidate_count: usize,
    pub portfolio_size: usize,
    pub max_rows: usize,
    pub max_rows_by_timeframe: HashMap<String, usize>,
    pub max_hours: f64,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
    pub target_profile: TargetProfile,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub cpcv_max_rows: usize,
    /// PBO ceiling: export is blocked when the measured Probability of
    /// Backtest Overfitting exceeds this. 0.5 = "the in-sample champion must
    /// beat the out-of-sample median more often than a coin flip". `<= 0`
    /// disables the gate (research/test fixtures only).
    pub max_pbo: f64,
    pub filtering: crate::genetic::FilteringConfig,
    /// Starting account balance used for PnL%, DD%, and regime loss limits.
    pub initial_balance: f64,
    /// Per-trade risk band the backtest sizes positions with, as balance
    /// fractions. A trade is sized so a full stop-loss costs
    /// `min + (max - min) * confidence` of equity at entry.
    ///
    /// These come from the operator's `risk.min_risk_per_trade` /
    /// `risk.max_risk_per_trade`. Before 2026-07-21 the discovery backtest
    /// silently used `BacktestSettings::default()` (0.5%..3%) no matter what
    /// the config said, so raising the risk knob changed live sizing but NOT
    /// the search — even though the Discovery pre-flight told the operator it
    /// applied to "this search". Risky mode in particular could never actually
    /// search at the aggressive size it exists for.
    pub risk_per_trade_min: f64,
    pub risk_per_trade_max: f64,
    /// Per-mode overrides of the band above, resolved by
    /// [`Self::apply_mode_overrides`]. `None` = inherit the shared band.
    /// Risky and Prop-firm are different products; one shared sizing knob
    /// silently carried 30%-compounding risk into a challenge search (where
    /// the firm's daily rule makes it unpassable) and vice versa.
    pub risky_risk_band: Option<(f64, f64)>,
    pub prop_firm_risk_band: Option<(f64, f64)>,
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
    /// Commission per lot used in the sensitivity test — a ROUND-TRIP charge,
    /// like [`Self::evaluation_commission_per_trade`], because the stress pass
    /// subtracts it exactly once per closed trade.
    ///
    /// `from_settings` puts `models.prop_search_sensitivity_commission_per_lot`
    /// through the same `round_trip_commission_per_lot` conversion as the
    /// baseline (gated on `risk.commission_per_lot_is_per_side`) and then
    /// clamps it UP to the baseline: a stress scenario may cost more than the
    /// run it stresses, never less.
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
    /// agent 2026-06-05 overfitting fix: when `true` (default), PropFirm-mode
    /// export-readiness ALSO requires the walk-forward gate to pass — not just
    /// the prop-firm window gate. Previously walk-forward was informational in
    /// PropFirm mode, so overfit strategies that failed out-of-sample still
    /// exported. Wired from `models.require_walkforward_for_export`. When
    /// `false`, behaviour is identical to before (window gate only).
    pub require_walkforward_for_export: bool,
    /// agent 2026-06-05 overfitting fix: hard floor for the prop-firm
    /// window-pass rate, combined (max) with `prop_firm_gate.pass_rate`. Wired
    /// from `models.prop_firm_min_pass_rate` (default 0.65). A value of 0.0
    /// reproduces the old ranking-only behaviour.
    pub prop_firm_min_pass_rate: f64,
    /// Search-memory + weekly-refresh ledger (2026-06-06): when `true`, this run
    /// loads the prior per-symbol/TF ledger and seeds the GA's seen-signature
    /// memory before search, then writes an updated ledger after finalize. When
    /// `false`, behaviour is byte-identical to a build without the feature.
    /// Wired from `models.discovery_ledger.enabled`.
    pub discovery_ledger_enabled: bool,
    /// Directory the discovery ledger JSON files live in. Wired from
    /// `models.discovery_ledger.cache_dir`.
    pub discovery_ledger_cache_dir: String,
    /// How many top archive (non-portfolio) genes to also record in the ledger.
    /// Wired from `models.discovery_ledger.archive_top_n`.
    pub discovery_ledger_archive_top_n: usize,
}

/// Configuration for the prop-firm window-pass gate.
#[derive(Debug, Clone, Serialize)]
pub struct PropFirmGateOverrides {
    pub rules: PropFirmRiskRules,
    pub n_windows: usize,
    pub window_days: usize,
    pub pass_rate: f64,
}

/// Resolve one trading mode's per-trade risk band from its config pair.
///
/// A band counts as SET only when a positive, finite max is given; the min
/// defaults to 0 so sizing scales up from zero with signal confidence — the
/// same shape as the shared band. Values are ordered and clamped to [0, 100%]
/// so a mis-typed entry can never invert the band or exceed the account.
/// `None` means "inherit the shared `risk.min/max_risk_per_trade`".
fn resolve_mode_risk_band(min: Option<f64>, max: Option<f64>) -> Option<(f64, f64)> {
    let max = max.filter(|m| m.is_finite() && *m > 0.0)?;
    let min = min.filter(|m| m.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0);
    Some((min, max.clamp(min, 1.0)))
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
            // Flat spread at every hour — the behaviour every production run
            // has had since the profile type was written. `from_settings`
            // populates this when the operator has measured a curve.
            session_spread_pips: None,
            // The same 1.6–2.4 band `RiskConfig::default()` resolves to.
            //
            // This was `None` on the reasoning that `default()` is only a test
            // fixture. It is not: `engines_control` falls back to `default()`
            // whenever config.yaml cannot be read, so that reasoning shipped a
            // production path on which every candidate came back
            // `cost_band_unmeasured` and nothing said why. An unmeasured band
            // is not neutral — it is the absence of the only evidence that
            // separates a result from a result-at-the-optimistic-edge.
            cost_band_pips: Some((1.6, 2.4)),
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            // Same value `RiskConfig::default()` ships (`config.rs:671`), so a
            // config-less fallback searches under the same weekend policy the
            // live loop applies. See the field's doc for why this is one knob
            // and not two.
            kill_zones_enabled: true,
            population: 1000,
            population_auto: false,
            generations: 10,
            max_indicators: 5,
            candidate_count: 5000,
            portfolio_size: 2000,
            max_rows: 0,
            max_rows_by_timeframe: HashMap::new(),
            max_hours: 0.0,
            corr_threshold: 0.85,
            min_trades_per_day: 0.2,
            // Decision A default (2026-08-09): the 2RR payoff floor is the
            // operator's intent, so the config-less fallback must embody it too
            // — a run that lands here because config.yaml failed to load must
            // NOT silently drop the floor to 0. Kept in lockstep with
            // `models.prop_search_min_payoff_ratio`'s default (divergence test).
            //
            // The expectancy fields are left at their `Default` (0.0 / 0.0),
            // which for `min_net_expectancy_per_trade` means "strictly positive
            // required" — the floor is unconditional and cannot be configured
            // away, here or anywhere.
            target_profile: TargetProfile {
                min_payoff_ratio: 2.0,
                ..TargetProfile::default()
            },
            walkforward_splits: 20,
            embargo_minutes: 120,
            enable_cpcv: true,
            cpcv_n_splits: 5,
            cpcv_n_test_groups: 2,
            cpcv_embargo_pct: 0.01,
            cpcv_purge_pct: 0.02,
            cpcv_min_phi: 0.80,
            cpcv_max_rows: 0,
            max_pbo: 0.5,
            filtering: crate::genetic::FilteringConfig::default(),
            initial_balance: 100_000.0,
            // Historical BacktestSettings defaults, kept so a bare
            // DiscoveryConfig::default() behaves exactly as before.
            risk_per_trade_min: 0.005,
            risk_per_trade_max: 0.03,
            // Decision default (2026-08-09): the Risky 30% ceiling is operator
            // intent, so the config-less fallback carries the same band as
            // `from_settings` derives from `risk.risky_max_risk_per_trade`
            // (min inherits 0.0). Kept in lockstep (divergence test).
            risky_risk_band: Some((0.0, 0.30)),
            prop_firm_risk_band: None,
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
            // ROUND TRIP, not per side (2026-08-10, same change that put
            // `from_settings` through `round_trip_commission_per_lot`). The
            // field is subtracted ONCE per closed trade, so the per-side 7.0
            // that stood here was half a stress test: the "higher commission"
            // pass charged less than the baseline it was stressing and every
            // candidate cleared it. 14.0 is the shipped
            // `risk.commission_per_lot: 7.0` per side taken both ways, which is
            // exactly what `from_settings(&Settings::default())` resolves to —
            // the two production constructors are pinned together by
            // `discovery_config_default_vs_from_settings_divergence_does_not_grow`.
            sensitivity_commission_per_lot: 14.0,
            // Matches `DiscoveryRuntimeConfig::default()`, which moved to `true`
            // in the same batch. The gene threshold ladder's own comment says it
            // is "calibrated for z-score-normalised features"; leaving this
            // `false` on the config-load-failure path meant the fallback run
            // searched a different objective from the configured one and said
            // nothing about it.
            adaptive_thresholds: true,
            // Env-absent default reproduces the retired
            // resolve_discovery_mode() fallback (PropFirm).
            mode: DiscoveryMode::PropFirm,
            prop_firm_gate_params: neoethos_core::config::PropFirmGateConfig::default(),
            // Risky-Mode goal defaults (mirror SystemConfig): 100 -> 50,000 in
            // 180 days. Ignored unless mode == Risky.
            risky_start_balance: 100.0,
            risky_target_balance: 50000.0,
            risky_horizon_days: 180.0,
            // walk-forward export gate stays ON (robustness). prop-firm pass-rate floor
            // RE-CALIBRATED 0.65→0.40 (2026-06-06): with the per-window target now at the
            // operator's bar (8%/60d = 4%/month), 0.40 means "hits >=4%/month in >=40% of
            // all 60-day windows" — a genuine, persistent edge, while the live models lift
            // the rest (discovery=edge, models=grow). 0.65 demanded near-always-prop-firm-
            // grade consistency, which cut every gene. (from_settings overrides from typed
            // config; these defaults match ModelsConfig::default.)
            require_walkforward_for_export: true,
            prop_firm_min_pass_rate: 0.40,
            // Search-memory ledger defaults mirror DiscoveryLedgerConfig::default
            // (enabled, cache/search, top-20 archive). from_settings overrides
            // from typed config.
            discovery_ledger_enabled: true,
            discovery_ledger_cache_dir: "cache/search".to_string(),
            discovery_ledger_archive_top_n: 20,
        }
    }
}

impl DiscoveryConfig {
    /// Production settings adapter. Financial fields are unreachable until
    /// the exact broker replay capability is installed; callers must not use
    /// `from_settings` as a fallback after this refusal.
    pub fn try_from_settings(settings: &neoethos_core::Settings) -> anyhow::Result<Self> {
        neoethos_core::current_broker_financial_truth_capability_v1()
            .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
            .map_err(anyhow::Error::new)?;
        Ok(Self::from_settings(settings))
    }

    pub(crate) fn from_settings(settings: &neoethos_core::Settings) -> Self {
        // The cost model converts a pip value into the account currency, which
        // for a cross pair needs the bridging pair's price. This is the one
        // place every discovery path passes through holding the full Settings,
        // so it is where the store gets pointed at. A CLI `--root` / `--data-path`
        // override reinstalls it later and wins.
        crate::fx_rates::set_store_root(settings.system.data_dir.clone());
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

        // Decision D (2026-08-09): charge the broker's REAL costs. Every held
        // position pays overnight financing, and the broker charges its own
        // per-lot commission; a zero-swap / config-flat backtest overstates
        // edge, and the search then buys carry and volume it will never earn
        // live. Resolve both once here from the symbol's broker-authoritative
        // metadata so the CPU and GPU kernels charge identical numbers. Swap is
        // signed as the broker stores it (negative = the account pays).
        let symbol = settings.system.symbol.clone();
        let meta = neoethos_core::symbol_metadata::global_table().lookup(&symbol);
        let (swap_long, swap_short) = match meta {
            Some(m) => (
                m.daily_swap_long_pips.unwrap_or(0.0),
                m.daily_swap_short_pips.unwrap_or(0.0),
            ),
            None => (0.0, 0.0),
        };
        let config_commission = settings.risk.commission_per_lot.max(0.0);
        let quoted_commission = meta
            .and_then(|m| m.commission_per_lot)
            .filter(|c| *c > 0.0)
            .unwrap_or(config_commission);
        // PER SIDE -> ROUND TRIP (2026-08-09). Both sources above are broker
        // quotes, and a broker quotes per side; every evaluator here subtracts
        // `commission_per_trade` exactly ONCE per closed trade. So the number
        // has to be doubled somewhere, and this is one of the only two places
        // that do it (the other is `infer_market_cost_profile`, which never
        // sees a value that has already been through here — discovery passes
        // this as the explicit `commission_override`). At the shipped 7.0 the
        // charge goes from $7 to $14 per lot per closed trade: about 1.4 pips
        // on a EURUSD standard lot instead of 0.7. That is not an improvement
        // to the strategies, it is the removal of a subsidy the search was
        // selecting on.
        let commission_is_per_side = settings.risk.commission_per_lot_is_per_side;
        let resolved_commission = crate::genetic::strategy_gene::round_trip_commission_per_lot(
            quoted_commission,
            commission_is_per_side,
        );
        tracing::info!(
            target: "neoethos_search::cost_model",
            symbol = %symbol,
            quoted_commission_per_lot = quoted_commission,
            commission_is_per_side,
            round_trip_commission_per_lot = resolved_commission,
            "commission resolved to a ROUND TRIP charge — the evaluators subtract \
             it once per closed trade"
        );

        // The session-spread curve. `Err` is a partial / malformed curve and is
        // refused rather than repaired: a cost model configured for two of the
        // three UTC buckets charges an unchosen number for a third of every
        // trading day. `Ok(None)` is the shipped state and gets a WARN naming
        // what it costs, because the curve existing-but-never-populated is the
        // exact defect this field was added to end.
        let session_spread_pips: Option<[f64; 3]> = match settings.risk.session_spread_pips() {
            Ok(Some(curve)) => {
                let slip = settings.risk.slippage_pips.max(0.0);
                // Slippage rides on each bucket exactly as it rides on the flat
                // `evaluation_spread_pips` below, so the two paths charge the
                // same thing when the curve is uniform.
                let with_slip = [curve[0] + slip, curve[1] + slip, curve[2] + slip];
                tracing::info!(
                    target: "neoethos_search::cost_model",
                    symbol = %symbol,
                    asian_pips = with_slip[0],
                    overlap_pips = with_slip[1],
                    late_ny_pips = with_slip[2],
                    slippage_pips = slip,
                    "session spread curve ACTIVE — spread is now resolved per bar from its \
                     UTC hour on the CPU path and in the CUDA kernel alike"
                );
                Some(with_slip)
            }
            Ok(None) => {
                tracing::warn!(
                    target: "neoethos_search::cost_model",
                    symbol = %symbol,
                    flat_spread_pips = settings.risk.backtest_spread_pips.max(0.0)
                        + settings.risk.slippage_pips.max(0.0),
                    "no session spread curve configured — a FLAT spread is charged at 03:00 \
                     Tokyo and at the London open alike. The per-bar lookup exists on both the \
                     CPU path and the CUDA kernel and is simply unpopulated. Measure your \
                     broker's per-hour spread and set risk.backtest_spread_pips_{{asian,\
                     overlap,late_ny}}. Until then, any result that depends on WHEN it trades \
                     is measured at the wrong cost."
                );
                None
            }
            Err(reason) => {
                // Not a panic and not a silent flat fall-back: the run continues
                // on the flat spread, but the operator is told their curve was
                // rejected and why, in the same words the config doc uses.
                tracing::error!(
                    target: "neoethos_search::cost_model",
                    symbol = %symbol,
                    reason = %reason,
                    "session spread curve REFUSED — falling back to the flat spread. Fix the \
                     three risk.backtest_spread_pips_* keys or remove all three."
                );
                None
            }
        };

        // The cost band every reported result is measured against. `None` means
        // the operator's band is unusable (inverted, negative or non-finite) —
        // reported as such rather than silently collapsed to a point estimate.
        let cost_band_pips = settings.risk.cost_band_pips();
        match cost_band_pips {
            Some((lo, hi)) => tracing::info!(
                target: "neoethos_search::cost_model",
                optimistic_pips = lo,
                pessimistic_pips = hi,
                "cost band ACTIVE — every survivor is re-measured at BOTH edges and one that \
                 clears only the optimistic edge is flagged, not reported as a result"
            ),
            None => tracing::warn!(
                target: "neoethos_search::cost_model",
                optimistic_pips = settings.risk.cost_band_optimistic_pips,
                pessimistic_pips = settings.risk.cost_band_pessimistic_pips,
                "cost band is unusable (non-finite, negative, or optimistic > pessimistic) — \
                 results will carry a single cost point, which nobody can check"
            ),
        }

        if meta.is_none() || (swap_long == 0.0 && swap_short == 0.0) {
            tracing::warn!(
                target: "neoethos_search::discovery",
                symbol = %symbol,
                has_metadata = meta.is_some(),
                swap_long,
                swap_short,
                resolved_commission,
                config_commission,
                "Decision D: swap resolved to ZERO — held positions pay no \
                 overnight financing in the backtest. Reconcile the broker symbol \
                 table (data/symbol_metadata.json) so carry is charged honestly."
            );
        } else {
            tracing::info!(
                target: "neoethos_search::discovery",
                symbol = %symbol,
                swap_long,
                swap_short,
                resolved_commission,
                "Decision D: charging broker-authoritative swap + commission"
            );
        }

        // ── DUPLICATE-KNOB RESOLUTION, SAID OUT LOUD (2026-08-10) ────────────
        //
        // Three knobs in this config exist TWICE under different section
        // names. In every case one copy decides and the other reaches nothing,
        // and until now nothing said which. An operator editing the losing copy
        // saw a saved value, a green config, and no change in behaviour — the
        // failure wearing the costume of a choice.
        //
        // This block does not change which copy wins. It names the winner, the
        // loser and both values, once per run, before a bar is read. The
        // deletion of the losing fields is a separate, config-side change; a
        // key that has been telling the truth in the log for a run or two is
        // safe to remove, a key removed while it still looked live is not.
        resolve_and_log_duplicate_knobs(settings);

        // ── GATE STATE, AND WHERE IT CAME FROM ───────────────────────────────
        //
        // Every gate below is a safety check the code implements and a shipped
        // config can switch off. §3 of the 2026-08-09 knob pass calls this "the
        // most consequential class in the report": a lost key silently re-arms
        // a gate the operator deliberately disarmed, or keeps one disarmed that
        // the Rust Default says should be on, and no config diff explains why
        // exports stopped or started.
        //
        // The line below is the record. It prints the EFFECTIVE value and the
        // Rust Default beside it, so "these two differ" is visible in the log
        // of the run itself rather than derivable only by diffing four files.
        log_gate_states(settings);

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
            // Honest-costs fix (2026-07-02): `risk.slippage_pips` existed (and
            // the live order-cost helper charged it) but the DISCOVERY
            // evaluator ignored it — strategies were validated against costs
            // the live fills never see. Fold slippage into the effective
            // spread HERE (single resolution point) so the CPU and GPU
            // kernels charge it identically with zero kernel changes.
            evaluation_spread_pips: settings.risk.backtest_spread_pips.max(0.0)
                + settings.risk.slippage_pips.max(0.0),
            evaluation_commission_per_trade: resolved_commission,
            session_spread_pips,
            cost_band_pips,
            swap_long_pips_per_day: swap_long,
            swap_short_pips_per_day: swap_short,
            // #75/#217: the SAME field the live loop reads
            // (`live_trading.rs:732-735`). One knob, both sides.
            kill_zones_enabled: settings.risk.kill_zones_enabled,
            population: model_settings.prop_search_population.max(10),
            population_auto: model_settings.prop_search_population_auto,
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
            target_profile: TargetProfile {
                // `.max(0.0)` is deliberate and load-bearing: a negative floor
                // configured here would admit money-losers by arithmetic. The
                // floor may be raised above zero, never below it.
                min_net_expectancy_per_trade: model_settings
                    .prop_search_min_net_expectancy_per_trade
                    .max(0.0),
                min_expectancy_t_stat: model_settings.prop_search_min_expectancy_t_stat.max(0.0),
                min_win_rate: model_settings.prop_search_min_win_rate.clamp(0.0, 1.0),
                min_payoff_ratio: model_settings.prop_search_min_payoff_ratio.max(0.0),
                max_in_market: model_settings.prop_search_max_in_market.max(0.0),
            },
            walkforward_splits: model_settings.walkforward_splits.max(2),
            embargo_minutes: model_settings.embargo_minutes,
            enable_cpcv: model_settings.enable_cpcv,
            cpcv_n_splits: model_settings.cpcv_n_splits.max(2),
            cpcv_n_test_groups: model_settings.cpcv_n_test_groups.max(1),
            cpcv_embargo_pct: model_settings.cpcv_embargo_pct.max(0.0),
            cpcv_purge_pct: model_settings.cpcv_purge_pct.max(0.0),
            cpcv_min_phi: model_settings.cpcv_min_phi.max(0.0),
            cpcv_max_rows: model_settings.cpcv_max_rows,
            // PBO gate default 0.5 — the honest ceiling; not yet a Settings
            // knob (deliberate: loosening it should require editing code or
            // raw YAML, not one careless click).
            max_pbo: 0.5,
            filtering,
            initial_balance: settings.risk.initial_balance.max(1.0),
            // The operator's own risk band now reaches the search. Clamped to
            // a sane [0, 100%] and ordered so a mis-set min can never exceed
            // max (which would size every trade at the floor).
            risk_per_trade_min: settings.risk.min_risk_per_trade.clamp(0.0, 1.0),
            risk_per_trade_max: settings
                .risk
                .max_risk_per_trade
                .clamp(settings.risk.min_risk_per_trade.clamp(0.0, 1.0), 1.0),
            risky_risk_band: resolve_mode_risk_band(
                settings.risk.risky_min_risk_per_trade,
                settings.risk.risky_max_risk_per_trade,
            ),
            prop_firm_risk_band: resolve_mode_risk_band(
                settings.risk.prop_firm_min_risk_per_trade,
                settings.risk.prop_firm_max_risk_per_trade,
            ),
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
            sensitivity_spread_pips: model_settings.prop_search_sensitivity_spread_pips.max(0.0),
            // SAME PER-SIDE→ROUND-TRIP CONVERSION AS THE BASELINE (2026-08-10).
            //
            // This number is assigned straight into `settings.commission_per_trade`
            // for the stress pass (`discovery.rs` sensitivity arm) and into the
            // scenario descriptor, and BOTH charge it exactly once per closed
            // trade — the same contract as `evaluation_commission_per_trade`. It
            // was the one commission input that never went through
            // `round_trip_commission_per_lot`, so at the shipped defaults
            // (`risk.commission_per_lot: 7.0` per side → 14.0 round trip, and
            // `prop_search_sensitivity_commission_per_lot: 7.0` charged as-is)
            // the "higher commission" stress test charged HALF the baseline. A
            // stress scenario that is cheaper than the run it stresses passes
            // everything, which is worse than not running it.
            //
            // The `.max(baseline)` is not a repair of a bad number, it is the
            // definition of the pass: a sensitivity test is the baseline cost or
            // worse, never better. It is logged when it binds.
            sensitivity_commission_per_lot: {
                let quoted = model_settings
                    .prop_search_sensitivity_commission_per_lot
                    .max(0.0);
                let round_trip = crate::genetic::strategy_gene::round_trip_commission_per_lot(
                    quoted,
                    commission_is_per_side,
                );
                if round_trip < resolved_commission {
                    tracing::warn!(
                        target: "neoethos_search::cost_model",
                        sensitivity_quoted_per_lot = quoted,
                        sensitivity_round_trip = round_trip,
                        baseline_round_trip = resolved_commission,
                        "models.prop_search_sensitivity_commission_per_lot is BELOW the \
                         baseline commission — raising it to the baseline so the stress \
                         pass cannot be cheaper than the run it stresses"
                    );
                }
                round_trip.max(resolved_commission)
            },
            adaptive_thresholds: model_settings.discovery_runtime.adaptive_thresholds,
            mode: resolve_discovery_mode(
                &settings.system.trading_mode,
                &model_settings.discovery_mode,
            ),
            prop_firm_gate_params: model_settings.discovery_runtime.prop_firm_gate.clone(),
            risky_start_balance: settings.system.risky_start_balance_usd,
            risky_target_balance: settings.system.risky_target_balance_usd,
            risky_horizon_days: settings.system.risky_horizon_days as f64,
            // agent 2026-06-05 overfitting fix: walk-forward export gate +
            // prop-firm pass-rate floor, both from typed config (Settings.models).
            require_walkforward_for_export: model_settings.require_walkforward_for_export,
            prop_firm_min_pass_rate: model_settings.prop_firm_min_pass_rate.clamp(0.0, 1.0),
            // Search-memory + weekly-refresh ledger (2026-06-06): wired from
            // models.discovery_ledger so discovery.rs can read it.
            discovery_ledger_enabled: model_settings.discovery_ledger.enabled,
            discovery_ledger_cache_dir: model_settings.discovery_ledger.cache_dir.clone(),
            discovery_ledger_archive_top_n: model_settings.discovery_ledger.archive_top_n,
        }
        // The mode's own floors, which production never applied.
        //
        // `apply_mode_overrides` was called from tests and nowhere else, so
        // every real run used the struct defaults — max_dd 0.15, min_win_rate
        // 0.50, min_profit_factor 1.20 — no matter what `trading_mode` said.
        // Risky's 0.60 cap and PropFirm's 0.50 existed and never took effect.
        //
        // It shows up as a search that finds nothing: 2 211 candidates ranked,
        // 1 713 of them rejected for exceeding a 15 % drawdown cap that the
        // selected mode had raised. Choosing a mode has to change what the mode
        // says it changes.
        .apply_mode_overrides()
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
        // MODE-SCOPED SIZING (2026-07-21). Risky and Prop-firm are two
        // different products sharing one engine: one compounds aggressively,
        // the other must survive a challenge whose daily-loss rule is a few
        // percent. They used to share ONE risk band, so switching
        // `system.trading_mode` silently carried the other mode's sizing —
        // a 30% risky band made every prop-firm candidate break the daily rule
        // on its first loss, and the search returned nothing with no
        // explanation. Each mode now takes its own band when one is set;
        // `None` inherits the shared `risk.min/max_risk_per_trade` exactly as
        // before, so existing configs are untouched.
        let mode_band = match self.mode {
            DiscoveryMode::Risky => self.risky_risk_band,
            DiscoveryMode::PropFirm => self.prop_firm_risk_band,
            _ => None,
        };
        if let Some((min, max)) = mode_band {
            self.risk_per_trade_min = min;
            self.risk_per_trade_max = max;
        }
        tracing::info!(
            target: "neoethos_search::discovery",
            mode = ?self.mode,
            risk_per_trade_min = self.risk_per_trade_min,
            risk_per_trade_max = self.risk_per_trade_max,
            mode_scoped = mode_band.is_some(),
            "resolved per-trade risk band for this search"
        );
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
            // Activity is NOT a quality floor, so it is not loosened with the
            // others. Compounding a small balance to a large one needs a certain
            // number of winning trades — around 25 at 1.3× each — and a strategy
            // that trades twice a decade cannot deliver them however good each
            // trade is. This used to be pinned to 0.001 here, which silently
            // discarded `models.prop_search_val_min_trades_per_day` in the one
            // mode the operator actually runs: the knob existed, was set, and did
            // nothing. The permissive value now applies only when the operator
            // nothing. The value the operator set now survives into risky mode;
            // its upstream `.max(0.2)` already keeps a never-trading gene out, so
            // no local floor is needed here.
            //
            // Logged unconditionally, because an activity floor that is silently
            // rewritten is exactly the class of bug this line used to be.
            tracing::info!(
                target: "neoethos_search::discovery",
                min_trades_per_day = format!("{:.3}", self.min_trades_per_day),
                "risky mode: keeping the operator's activity floor"
            );
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
        // 2026-06-06 RE-CALIBRATED to the operator's actual bar (after validation showed
        // ALL genes cut here). The window check requires hitting `min_profit_target_pct`
        // per 60-day window. The full FTMO target is 10%/60d (~5%/month) — but the
        // operator's product bar is **>=4% net per MONTH** = ~8% per 60-day window. The
        // earlier 10% demanded MORE than the stated bar, so a steady +4%/month strategy
        // (+8%/window) failed EVERY window. We now require the operator's bar directly:
        // 8%/60-day window. Architecture: discovery finds the EDGE (consistent >=4%/month,
        // low DD); the live models grow the account. Config `profit_target_pct` still
        // overrides (e.g. set 0.10 to restore the full FTMO challenge target).
        // (FTMO_STANDARD.challenge_profit_target_pct = 0.10 remains the reference constant.)
        const DISCOVERY_MONTHLY_BAR_PER_60D_WINDOW: f64 = 0.08; // = operator's >=4%/month over a 60-day window
        rules.min_profit_target_pct = DISCOVERY_MONTHLY_BAR_PER_60D_WINDOW;
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

    /// Checked public boundary for callers outside `neoethos-search`.
    /// Financial configuration cannot be resolved until exact broker evidence
    /// is installed; the crate-private builder remains only for gated internal
    /// paths and formula tests during this disabled phase.
    pub fn try_evaluation_config(
        &self,
        price_hint: Option<f64>,
    ) -> anyhow::Result<EvaluationConfig> {
        neoethos_core::current_broker_financial_truth_capability_v1()
            .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
            .map_err(anyhow::Error::new)?;
        Ok(self.evaluation_config(price_hint))
    }

    pub(crate) fn evaluation_config(&self, price_hint: Option<f64>) -> EvaluationConfig {
        let mut cfg = EvaluationConfig::for_symbol(
            &self.evaluation_symbol,
            &self.evaluation_account_currency,
            price_hint,
            Some(self.evaluation_spread_pips),
            Some(self.evaluation_commission_per_trade),
        );
        // scoring_version 5: Risky discovery evolves under the Kelly
        // log-growth objective — the SAME math its post-GA ranking
        // (`calculate_income_score`) scores with, so the population the
        // ranking sees was actually searched FOR growth. PropFirm/Strict
        // keep the v4 consistency landscape untouched.
        cfg.growth_objective = matches!(self.mode, DiscoveryMode::Risky);
        cfg
    }

    pub(crate) fn evaluation_config_with_smc_gate(
        &self,
        price_hint: Option<f64>,
        effective_smc_gate_threshold: f32,
    ) -> EvaluationConfig {
        let mut cfg = self.evaluation_config(price_hint);
        cfg.smc_gate_threshold = effective_smc_gate_threshold;
        cfg
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// The cost-band verdict for every candidate that SURVIVED the quality
    /// screen, as `(strategy_id, verdict)` — audit #71.
    ///
    /// The band was measured at both edges and counted run-level since
    /// 2026-08-09, and then DROPPED at this boundary: the export loop bound it
    /// `_cost_band` and threw it away, so a gene profitable only at the
    /// optimistic 1.6-pip edge reached `live_portfolio.json` indistinguishable
    /// from one profitable across the whole band. The census answered "how many"
    /// and nothing answered "which", which is the question an operator looking
    /// at a deployed strategy is actually asking.
    ///
    /// EMPTY IS NOT "ALL CLEAR". It means the quality screen did not run, or ran
    /// with no band configured. A reader that treats an absent entry as
    /// `SurvivesBand` re-creates the defect; the verdict for a gene with no
    /// entry is [`CostBandVerdict::Unmeasured`], which is what
    /// [`cost_band_for_strategy`] returns.
    pub cost_band_by_strategy: Vec<(String, CostBandVerdict)>,
    pub portfolio: Vec<Gene>,
    pub candidates: Vec<Gene>,
    pub quality_metrics: Vec<StrategyMetrics>,
    pub logged_trades: Vec<LoggedStrategyTrades>,
    /// Feature names as they existed *after* prefiltering inside discovery.
    /// Gene indices refer to columns in this list, not the caller's original names.
    pub effective_feature_names: Vec<String>,
    /// Final annealed SMC gate used by the GA and every post-search replay.
    pub effective_smc_gate_threshold: f32,
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

impl DiscoveryResult {
    /// What the cost band said about this strategy — audit #71.
    ///
    /// A strategy with no entry is [`CostBandVerdict::Unmeasured`], never
    /// `SurvivesBand`: "we did not measure" and "it passed" are different
    /// answers and only one of them supports a claim.
    pub fn cost_band_for_strategy(&self, strategy_id: &str) -> CostBandVerdict {
        self.cost_band_by_strategy
            .iter()
            .find(|(id, _)| id == strategy_id)
            .map(|(_, verdict)| *verdict)
            .unwrap_or(CostBandVerdict::Unmeasured)
    }
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
    /// SLICE 5 (2026-08-08): the three `FilteringConfig` fields the profile
    /// silently dropped before. `opportunistic_enabled` above remains the
    /// legacy MERGED flag (`use_opportunistic_candidates && opportunistic_enabled`)
    /// for consumers that already read it; the two raw flags are now also
    /// recorded so the merge itself is auditable.
    pub use_opportunistic_candidates_raw: bool,
    pub opportunistic_enabled_raw: bool,
    pub anomaly_guard: bool,
    pub elite_mode: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryValidationGates {
    pub walkforward_passed: bool,
    pub cpcv_passed: bool,
    pub canonical_backtest_artifacts: usize,
    pub walkforward_validation_artifacts: usize,
    pub cpcv_fold_count: usize,
    pub cpcv_profitable_fold_ratio: f64,
    /// Probability of Backtest Overfitting (CSCV over the CPCV splits, López
    /// de Prado): the fraction of splits where the IN-SAMPLE champion of the
    /// candidate set ranked at-or-below the median OUT-of-sample. `None` when
    /// not computable (too few candidates or the gate is disabled).
    pub pbo: Option<f64>,
    /// False only when PBO was computed AND exceeded `config.max_pbo` —
    /// a portfolio whose selection process looks like luck is not exportable.
    pub pbo_passed: bool,
    /// How many candidate strategies fed the PBO estimate.
    pub pbo_candidates: usize,
    /// Honesty counter: how many candidates the whole run RANKED before any
    /// gate — the selection pressure the survivors' metrics were bought with.
    pub trials_tested: usize,
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
    /// NEVER-ZERO (2026-06-09, operator non-negotiable): set when the strict
    /// funnel rejected EVERY candidate and discovery promoted the best-found
    /// genes as a best-effort portfolio instead of dying empty. These genes did
    /// NOT pass the prop bar — `prop_firm_window_passed`/`walkforward_passed`/
    /// `cpcv_passed` are all forced false so `is_portfolio_export_ready()` stays
    /// honest. Downstream consumers (the autonomous trader) MUST treat a
    /// fallback portfolio cautiously (e.g. demo-only / heavily down-sized).
    #[serde(default)]
    pub fallback_mode: bool,
    /// Which strict stage was the bottleneck that emptied the portfolio (e.g.
    /// `"passed_prop_firm_window"`). Empty unless `fallback_mode`.
    #[serde(default)]
    pub fallback_reason: String,
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
            pbo: None,
            pbo_passed: true, // pending gates fail on wf/cpcv; PBO only blocks when measured
            pbo_candidates: 0,
            trials_tested: 0,
            temporal_contract_hash: None,
            prop_firm_window_passed: false,
            prop_firm_window_pass_rate: 0.0,
            prop_firm_window_count: 0,
            fallback_mode: false,
            fallback_reason: String::new(),
        }
    }

    pub fn is_portfolio_export_ready(&self) -> bool {
        // MANDATORY out-of-sample validation (operator directive 2026-06-30):
        // a strategy is export-ready ONLY if it passed BOTH out-of-sample gates
        // — walkforward AND CPCV. The prop-firm window is an ADDITIONAL
        // requirement for prop-firm runs (folded into `walkforward_passed` via
        // the mode-aware criterion), never a bypass. This closes the hole where
        // a strategy that FAILED walkforward (e.g. AUDUSD: 20 live trades, all
        // losing) was still exported because it cleared the prop-firm window.
        //
        // 2026-07-02: plus the PBO gate — when the Probability of Backtest
        // Overfitting was measured and exceeded the configured ceiling, the
        // selection process is statistically indistinguishable from luck and
        // nothing gets exported, no matter how good the survivors look.
        self.walkforward_passed && self.cpcv_passed && self.pbo_passed
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryRunProfile {
    pub timeframe_label: String,
    pub population: usize,
    /// Whether `run_search` was allowed to raise `population` to the card's
    /// fits ceiling. Profiled because it is selection-changing: two runs with
    /// the same `population` but different `population_auto` can search
    /// different candidate counts. From `models.prop_search_population_auto`.
    pub population_auto: bool,
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
    pub prefilter_min_per_timeframe: usize,
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
    /// Which engine(s) actually evaluated the population in this run.
    ///
    /// Recovered from f910c0f2 during the 2026-08-10 cherry-pick: the strict-mode
    /// fix is only half a fix without it. The CubeCL f64 lane is ~0.19% off at
    /// 200,000 bars and the f32 lane is 54% off there, because rounding flips
    /// stop/target comparisons and the run takes 129-430 more trades. Two runs on
    /// different engines therefore ranked different strategies, and nothing in the
    /// artifact said which had run. Empty means no population evaluation was
    /// recorded in this process (a profile built from a fixture).
    pub population_eval_engines: Vec<crate::engine_identity::PopulationEvalEngine>,
    pub determinism_policy: DeterminismPolicy,
    // ── SLICE 5 (2026-08-08): the config fields the profile silently ──────
    // dropped before. `build_discovery_profile` now destructures
    // `DiscoveryConfig` WITHOUT `..`, so adding a config field that skips
    // this profile is a compile error, not a silent omission.
    /// Cost basis the whole search was evaluated on. A profile without the
    /// symbol/spread/commission cannot be compared across runs.
    pub evaluation_symbol: String,
    pub evaluation_account_currency: String,
    pub evaluation_spread_pips: f64,
    /// ROUND-TRIP commission per lot actually charged. Two runs that differ
    /// only in `risk.commission_per_lot_is_per_side` produce different money,
    /// so the resolved number — not the quote — belongs in the profile.
    pub evaluation_commission_per_trade: f64,
    /// Session spread curve `[asian, overlap, late_ny]` in pips (slippage
    /// folded in), or `None` for a flat spread at every hour. `None` is not a
    /// neutral value: it means the run could not distinguish a strategy that
    /// only trades the London open from one that only trades Tokyo.
    pub session_spread_pips: Option<[f64; 3]>,
    /// Round-trip cost band `(optimistic, pessimistic)` in pips that this run's
    /// survivors were re-measured against.
    pub cost_band_pips: Option<(f64, f64)>,
    /// Broker overnight financing charged in the backtest (Decision D). Part of
    /// the cost basis: two runs at different swap are not comparable.
    pub swap_long_pips_per_day: f64,
    pub swap_short_pips_per_day: f64,
    /// Weekend kill zones as this run resolved them (`risk.kill_zones_enabled`).
    /// Recorded from 2026-08-10: it decides whether a Friday-evening position
    /// was force-closed and whether Monday-open entries were blocked, so two
    /// runs that differ on it are not the same experiment.
    pub kill_zones_enabled: bool,
    /// Discovery regime (Strict / PropFirm / Risky) — changes filter floors,
    /// ranking, and gates. Was NOT recorded before slice 5.
    pub mode: DiscoveryMode,
    /// Operator strategy-shape preference applied during candidate ranking.
    pub target_profile: TargetProfile,
    pub max_pbo: f64,
    pub cpcv_max_rows: usize,
    /// Resolved prop-firm window gate (None = gate off), and the raw config
    /// params it was derived from.
    pub prop_firm_gate: Option<PropFirmGateOverrides>,
    pub prop_firm_gate_params: neoethos_core::config::PropFirmGateConfig,
    pub require_walkforward_for_export: bool,
    pub prop_firm_min_pass_rate: f64,
    /// Sizing basis of the backtests the selection ran on.
    pub initial_balance: f64,
    pub risk_per_trade_min: f64,
    pub risk_per_trade_max: f64,
    pub risky_risk_band: Option<(f64, f64)>,
    pub prop_firm_risk_band: Option<(f64, f64)>,
    pub max_regime_loss_pct: f64,
    /// Robustness screens (Monte-Carlo / sensitivity) parameters.
    pub mc_runs: u32,
    pub mc_min_profitable: u32,
    pub sensitivity_spread_pips: f64,
    pub sensitivity_commission_per_lot: f64,
    /// Search-space shaping.
    pub adaptive_thresholds: bool,
    pub higher_timeframes: Vec<String>,
    /// Sorted (BTreeMap) so two identical runs serialize byte-identically —
    /// a HashMap here would make the profile JSON itself non-reproducible.
    pub max_rows_by_timeframe: std::collections::BTreeMap<String, usize>,
    pub stage1_window: Stage1Window,
    pub min_history_years: u32,
    /// Risky-mode compounding goal (ignored unless `mode == Risky`).
    pub risky_start_balance: f64,
    pub risky_target_balance: f64,
    pub risky_horizon_days: f64,
    /// Discovery ledger — CROSS-RUN search memory. When enabled, this run's
    /// candidate generation was seeded by prior runs' seen-signatures, so an
    /// identical-config re-run may legitimately explore differently unless
    /// the ledger dir is cleared or pinned.
    pub discovery_ledger_enabled: bool,
    pub discovery_ledger_cache_dir: String,
    pub discovery_ledger_archive_top_n: usize,
    /// Ambient process-wide execution state (seed/selection policy, cost +
    /// SMC overrides, threads, adaptive stops, seen-memory, GPU lane) —
    /// captured through the same accessors the engine reads. See
    /// [`crate::execution_profile::ExecutionEnvironmentProfile`].
    pub execution: crate::execution_profile::ExecutionEnvironmentProfile,
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
    /// Coarse boundary marker for the long, otherwise-silent post-GA stages
    /// (quality screen, portfolio selection, validation gates, robustness
    /// filters, holdout replay). Purely informational: on dense timeframes
    /// these blocks run for HOURS with no other event, and the UI would
    /// otherwise freeze on the last milestone — which operators read as a
    /// hang (observed live 2026-07-20: a healthy EURCAD M3 run was killed at
    /// 95.5% because "it looked stuck").
    StageAdvanced { stage: &'static str, detail: String },
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
        _ => {
            "review the saved funnel JSON (cache/discovery/<symbol>_<tf>.json) for the full \
              stage-by-stage breakdown."
        }
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
    let frame_rows = features.n_samples();
    let ohlcv_rows = ohlcv.close.len();
    let available_rows = frame_rows.min(ohlcv_rows);
    if available_rows == 0 {
        anyhow::bail!(
            "Cannot run discovery on empty history for {} {} — \
             import at least the minimum bars (run `neoethos-cli import`) then retry.",
            config.evaluation_symbol,
            config.timeframe_label
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

    let trimmed_features = if start_idx == 0 && available_rows == frame_rows {
        // No trim — pass the (possibly mmap-backed) frame through untouched so
        // the full multi-resolution feature matrix is NEVER materialised into
        // RAM. This is the hot path: discovery runs with `max_rows = 0`.
        features.clone()
    } else {
        FeatureFrame {
            timestamps: features.timestamps[start_idx..available_rows].to_vec(),
            names: features.names.clone(),
            data: neoethos_data::FeatureData::InMemory(
                features.sample_window(start_idx, available_rows),
            ),
        }
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

/// RAW settings builder — the shared cost/exit template every discovery stage
/// starts from. **Do not call this directly from a new site.** It leaves the
/// adaptive-stop fields at their defaults (`adaptive_vol_mult = 0`), which is
/// the fixed-stop regime — while GA scoring evaluates adaptive genes with
/// `sl = stop_vol_mult × base[i]`. A site that calls this raw builder and then
/// backtests with the result is screening a DIFFERENT strategy from the one
/// that was scored (measured on one signal: fixed 13p ⇒ 30 331 trades,
/// adaptive ×1.75 ⇒ 1 727 — a 17.6× divergence). Route through
/// [`GeneEvalSettingsResolver`] (serial per-gene evaluation) or
/// [`PopulationTemplateResolver`] (templates for the population helpers, which
/// resolve adaptive stops themselves). The
/// `discovery_backtest_settings_has_no_callers_outside_the_resolvers` test
/// counts the call sites of this function and fails on any new one.
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
        // All four exit-geometry fields together. `trailing_min_lock_pips` used
        // to be omitted here and silently inherited `BacktestSettings::default()`
        // (2.0) — a fourth number of the same policy arriving by a different
        // route, which is how policies drift apart one field at a time.
        trailing_enabled: evaluation.trailing_enabled,
        trailing_atr_multiplier: evaluation.trailing_atr_multiplier,
        trailing_be_trigger_r: evaluation.trailing_be_trigger_r,
        trailing_min_lock_pips: evaluation.trailing_min_lock_pips,
        pip_value: evaluation.pip_value,
        spread_pips: evaluation.spread_pips,
        commission_per_trade: evaluation.commission_per_trade,
        // THE WIRE (2026-08-09). `SessionSpreadProfile` has existed since the
        // type was written; the active CPU evaluator resolves it per bar and
        // `prototype_b_population.cu` mirrors that resolution on the card. Every production
        // construction site left it `None` — the only `Some(..)` in the tree
        // were under `#[cfg(test)]` — so the curve was dead code and a flat
        // spread was charged at every hour of the day. This is the single point
        // all discovery settings flow through, so setting it here turns the
        // curve on for the GA, the quality screen, walk-forward and CPCV at
        // once, with no kernel change.
        //
        // `None` reproduces the old behaviour exactly; `from_settings` has
        // already WARNed in that case.
        session_spread_profile: config.session_spread_pips.map(|curve| {
            crate::eval::SessionSpreadProfile {
                asian_pips: curve[0],
                overlap_pips: curve[1],
                late_ny_pips: curve[2],
            }
        }),
        pip_value_per_lot: evaluation.pip_value_per_lot,
        // Decision D: charge overnight financing (the engine applies it in both
        // the CPU path and the CUDA kernel; it was silently 0 here before).
        swap_long_pips_per_day: config.swap_long_pips_per_day,
        swap_short_pips_per_day: config.swap_short_pips_per_day,
        // #75/#217 (2026-08-10). This was the literal `true`, between two
        // fields that read `config.`. It is now the one knob both sides read.
        kill_zones_enabled: config.kill_zones_enabled,
        risk_per_trade_min: config.risk_per_trade_min,
        risk_per_trade_max: config.risk_per_trade_max,
        ..crate::eval::BacktestSettings::default()
    }
}

/// Settings template source for the POPULATION evaluation helpers
/// (`validation_genes_population`, `validation_genes_population_gathered`,
/// `validation_genes_population_window` via `WalkforwardPopulationGenePack`).
///
/// Those helpers take a gene-independent template and resolve BOTH the
/// per-gene SL/TP arrays AND the adaptive stop regime (per-gene
/// `stop_vol_mult` + a base vol series computed on exactly the slice they
/// evaluate) themselves — so the template deliberately carries NO adaptive
/// fields. Handing this template to a serial evaluator
/// (`simulate_trades_core` / `fast_evaluate_strategy_core`) would run fixed
/// stops on an adaptive gene; use [`GeneEvalSettingsResolver`] for that.
pub(crate) struct PopulationTemplateResolver<'c> {
    config: &'c DiscoveryConfig,
    price_hint: Option<f64>,
}

impl<'c> PopulationTemplateResolver<'c> {
    pub(crate) fn new(config: &'c DiscoveryConfig, price_hint: Option<f64>) -> Self {
        Self { config, price_hint }
    }

    /// Gene-independent template (the gene argument only supplies the SL/TP
    /// scalars the population helpers overwrite per gene anyway).
    pub(crate) fn template(&self, gene: &Gene) -> crate::eval::BacktestSettings {
        discovery_backtest_settings(self.config, gene, self.price_hint)
    }
}

/// THE single source of per-gene `BacktestSettings` for every SERIAL
/// (single-gene) evaluation in discovery — the quality screen's base
/// backtest, the canonical backtest artifacts, the forward-test and
/// prop-firm tails, the prop-firm window gate, faithful OOS, the
/// permutation/plateau robustness filters, and the walk-forward risk
/// diagnostics' per-gene settings.
///
/// It exists because of a measured divergence: GA scoring (and the
/// population validation helpers) evaluate adaptive genes with
/// `sl = stop_vol_mult × base[i]` (volatility-scaled per entry), while 9 of
/// the 13 former `discovery_backtest_settings` call sites left
/// `adaptive_vol_mult = 0` and ran the gene's unused FIXED pips — including
/// the quality screen, so a candidate was screened as a different strategy
/// from the one that was scored and the one that will trade (17.6× trade
/// count on one measured signal).
///
/// Construction is per evaluation SLICE: `high`/`low`/`close` MUST be exactly
/// the arrays the produced settings will be backtested against, because the
/// adaptive base series is per-bar and indexed into them. This matches the
/// established convention of the population paths (`resolve_adaptive_stops`,
/// `validation_genes_population_window`) and of live trading: the base is
/// computed on the data the evaluation actually sees.
pub(crate) struct GeneEvalSettingsResolver<'c> {
    config: &'c DiscoveryConfig,
    price_hint: Option<f64>,
    adaptive_pip: f64,
    adaptive_rr: f64,
    /// Shared per-bar base stop distance (pips) for THIS resolver's slice.
    /// `None` when no gene is adaptive or the slice is too short for the
    /// estimator (fixed-pip fallback, logged) — same policy as
    /// `resolve_adaptive_stops`.
    base: Option<std::sync::Arc<[f64]>>,
}

impl<'c> GeneEvalSettingsResolver<'c> {
    /// Build the resolver for ONE evaluation slice. Computes the shared
    /// adaptive base series once (gene-independent) when any gene in `genes`
    /// is adaptive; fail-loud on every base-series error except the benign
    /// too-short slice.
    pub(crate) fn for_slice<'g>(
        config: &'c DiscoveryConfig,
        genes: impl IntoIterator<Item = &'g Gene>,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> anyhow::Result<Self> {
        let price_hint = close.last().copied();
        let evaluation = config.evaluation_config(price_hint);
        let adaptive_pip =
            crate::genetic::adaptive_pip_size(evaluation.pip_value, &evaluation.symbol);
        let any_adaptive = genes
            .into_iter()
            .any(|g| g.stop_vol_mult.is_finite() && g.stop_vol_mult > 0.0);
        let base = if any_adaptive {
            match crate::stop_target::adaptive_base_pips_series(high, low, close, adaptive_pip) {
                Ok(base) => Some(std::sync::Arc::from(base)),
                Err(e @ crate::stop_target::StopDistanceError::TooShort { .. }) => {
                    tracing::debug!(
                        target: "neoethos_search::adaptive_stops",
                        bars = close.len(), error = %e,
                        "adaptive base series unavailable on this slice — fixed pips"
                    );
                    None
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "adaptive stop base series failed on {} bars: {e}",
                        close.len()
                    ));
                }
            }
        } else {
            None
        };
        Ok(Self {
            config,
            price_hint,
            adaptive_pip,
            adaptive_rr: crate::stop_target::adaptive_stops_rr(),
            base,
        })
    }

    /// Per-gene settings for a serial evaluation on this resolver's slice —
    /// the SAME stop regime GA scoring scored the gene under: the gene's
    /// `stop_vol_mult` scaling the shared base series, or its fixed pips when
    /// it is not adaptive (or the slice was too short for a base).
    pub(crate) fn settings_for_gene(&self, gene: &Gene) -> crate::eval::BacktestSettings {
        let mut settings = discovery_backtest_settings(self.config, gene, self.price_hint);
        if gene.stop_vol_mult.is_finite() && gene.stop_vol_mult > 0.0 {
            settings.adaptive_vol_mult = gene.stop_vol_mult;
            settings.adaptive_base_pips = self.base.clone();
            settings.adaptive_rr = self.adaptive_rr;
        }
        settings
    }

    /// Base series recomputed on a SUB-window of bars, with this resolver's
    /// pip. For stages that evaluate window slices (the prop-firm window
    /// gate): the base must be computed on exactly the slice being simulated,
    /// both for index alignment and to match the population walk-forward
    /// convention (`validation_genes_population_window` recomputes per
    /// window). Returns `Ok(None)` when the window is too short (fixed-pip
    /// fallback) and an error for every other base-series failure.
    pub(crate) fn base_for_window(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> anyhow::Result<Option<std::sync::Arc<[f64]>>> {
        match crate::stop_target::adaptive_base_pips_series(high, low, close, self.adaptive_pip) {
            Ok(base) => Ok(Some(std::sync::Arc::from(base))),
            Err(e @ crate::stop_target::StopDistanceError::TooShort { .. }) => {
                tracing::debug!(
                    target: "neoethos_search::adaptive_stops",
                    bars = close.len(), error = %e,
                    "window too short for an adaptive base — fixed pips"
                );
                Ok(None)
            }
            Err(e) => Err(anyhow::anyhow!(
                "adaptive stop base series failed on a {}-bar window: {e}",
                close.len()
            )),
        }
    }
}

/// Faithful out-of-sample result for ONE gene: its in-sample (discovery) metrics
/// + its REAL out-of-sample metrics (same engine, gene's own SL/TP + risk sizing)
/// + Walk-Forward Efficiency (Pardo) = OOS/IS retention.
#[derive(Debug, Clone)]
pub struct GeneOosResult {
    pub strategy_id: String,
    pub n_indicators: usize,
    pub n_smc: usize,
    pub is_profit_factor: f64,
    pub is_sharpe: f64,
    pub is_max_drawdown: f64,
    pub is_trades: usize,
    pub oos: crate::eval::BacktestMetrics,
    pub oos_monthly_hit_rate: f64,
    pub wfe_sharpe: f64,
    pub wfe_pf: f64,
}

/// FAITHFUL forward/OOS test (research-backed methodology). For each gene in a
/// `*.live_portfolio.json`, runs the gene's REAL strategy (its indicators+SMC
/// signals, its own SL/TP, risk-based confidence-scaled sizing, full costs) via
/// the SAME discovery backtest engine on the holdout window — NOT the Phase-1
/// trader stub. Features are computed on the FULL series (warm, no cold-start
/// contamination) then the evaluation is sliced to `[oos_start_ts, end)`, so the
/// holdout features are bit-identical to what discovery saw. Compares to the
/// gene's in-sample discovery metrics to get Walk-Forward Efficiency.
pub fn faithful_oos_eval(
    config: &DiscoveryConfig,
    data_dir: &std::path::Path,
    portfolio_path: &std::path::Path,
    oos_start_ts_ms: i64,
) -> anyhow::Result<Vec<GeneOosResult>> {
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;
    let _scope = crate::eval_telemetry::CallerScope::enter("faithful_oos");
    let artifact = crate::load_live_portfolio_json(portfolio_path)?;
    if artifact.genes.is_empty() {
        anyhow::bail!("portfolio {} has no genes", portfolio_path.display());
    }
    let symbol = artifact.symbol.clone();
    let base_tf = artifact.base_tf.clone();
    let base_ohlcv = neoethos_data::load_symbol_timeframe(data_dir, &symbol, &base_tf)?;
    if base_ohlcv.is_empty() {
        anyhow::bail!("no base bars for {symbol} {base_tf}");
    }
    // Rebuild the SAME multi-TF cube discovery used (warm over the FULL series),
    // then project onto the genes' effective feature set (fail-loud on drift).
    let dataset = neoethos_data::load_symbol_dataset(data_dir, &symbol)?;
    let higher_refs: Vec<&str> = artifact.higher_tfs.iter().map(|s| s.as_str()).collect();
    let raw_features =
        neoethos_data::prepare_multitimeframe_features(&dataset, &base_tf, &higher_refs)?;
    let features =
        crate::project_features_to_effective(&raw_features, &artifact.effective_feature_names)?;
    if features.n_samples() != base_ohlcv.len() {
        anyhow::bail!(
            "feature/bar length mismatch {symbol} {base_tf}: {} vs {}",
            features.n_samples(),
            base_ohlcv.len()
        );
    }
    let timestamps = features.timestamps.clone();
    let (months, days) = month_day_indices(&timestamps);
    // OOS window = first bar whose timestamp >= the cutoff (warm features behind it).
    let oos_start = timestamps
        .iter()
        .position(|&t| t >= oos_start_ts_ms)
        .unwrap_or(timestamps.len());
    if oos_start >= timestamps.len().saturating_sub(2) {
        anyhow::bail!(
            "no OOS bars after {oos_start_ts_ms} for {symbol} {base_tf} (last ts {})",
            timestamps.last().copied().unwrap_or(0)
        );
    }
    let eval_config = config.evaluation_config(base_ohlcv.close.last().copied());
    const MONTHLY_RETURN_TARGET_IDX: usize = 7; // slot-7 monthly_target_hit_rate

    // ONE resolver, constructed over the OOS slice the backtest below actually
    // runs on — adaptive genes are replayed under the SAME stop regime they
    // were scored under (base series indexed to `[oos_start..]`).
    let oos_resolver = GeneEvalSettingsResolver::for_slice(
        config,
        artifact.genes.iter(),
        &base_ohlcv.high[oos_start..],
        &base_ohlcv.low[oos_start..],
        &base_ohlcv.close[oos_start..],
    )?;
    let mut out = Vec::with_capacity(artifact.genes.len());
    for gene in &artifact.genes {
        let settings = oos_resolver.settings_for_gene(gene);
        let (signals, confidences) =
            signals_and_confidence_for_gene_full(&features, &base_ohlcv, gene, &eval_config);
        // Slice everything to the OOS window (features already warm).
        let close = &base_ohlcv.close[oos_start..];
        let high = &base_ohlcv.high[oos_start..];
        let low = &base_ohlcv.low[oos_start..];
        let sig = &signals[oos_start..];
        let conf = &confidences[oos_start..];
        let mo = &months[oos_start..];
        let dy = &days[oos_start..];
        let ts = &timestamps[oos_start..];
        let raw = fast_evaluate_strategy_core(close, high, low, sig, conf, mo, dy, ts, &settings);
        let oos_monthly_hit_rate = raw.get(MONTHLY_RETURN_TARGET_IDX).copied().unwrap_or(0.0);
        let oos = crate::eval::BacktestMetrics::from_metric_array(raw);
        let is_sharpe = gene.sharpe_ratio;
        let is_pf = gene.profit_factor;
        out.push(GeneOosResult {
            strategy_id: gene.strategy_id.clone(),
            n_indicators: gene.indices.len(),
            n_smc: [
                gene.use_ob,
                gene.use_fvg,
                gene.use_liq_sweep,
                gene.mtf_confirmation,
                gene.use_premium_discount,
                gene.use_inducement,
                gene.use_bos,
                gene.use_choch,
                gene.use_eqh,
                gene.use_eql,
                gene.use_displacement,
            ]
            .iter()
            .filter(|b| **b)
            .count(),
            is_profit_factor: is_pf,
            is_sharpe,
            is_max_drawdown: gene.max_drawdown,
            is_trades: gene.trades_count,
            oos_monthly_hit_rate,
            wfe_sharpe: if is_sharpe.abs() > 1e-9 {
                oos.sharpe / is_sharpe
            } else {
                0.0
            },
            wfe_pf: if is_pf.abs() > 1e-9 {
                oos.profit_factor / is_pf
            } else {
                0.0
            },
            oos,
        });
    }
    Ok(out)
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

/// A drawdown of 100 % means the account reached zero.
///
/// Past that the simulation is describing a state that cannot exist: a real
/// account is closed out, not carried into negative equity to keep compounding.
/// A 2026-07-29 AUDUSD H4 candidate reported `maxDD 403.1%` with an equity
/// curve minimum of -30 596 EUR and was still scored as EXCELLENT on profit
/// factor — 4 917 trades on an account that had been wiped several times over.
///
/// This is not a threshold to tune alongside `max_dd`; it is the boundary of
/// what the numbers can mean, so it is checked separately and unconditionally.
/// Anything at or beyond total loss is rejected whatever else it scores.
/// What a candidate must be to survive: one correctness bound, then the
/// operator's shape preferences.
///
/// THE CORRECTNESS BOUND — cost-charged net expectancy per trade. This is not a
/// preference and it is not optional. `min_net_expectancy_per_trade` is the only
/// field on this struct for which `0.0` does NOT mean "no preference": it means
/// "must be strictly greater than zero". There is no configuration in which a
/// candidate that loses money on the average trade is admitted.
///
/// WHY IT HAD TO BE ADDED — the proof, with the measured numbers.
///
/// Until 2026-08-09 this struct held shape preferences only, and with
/// `prop_search_min_win_rate` and `prop_search_max_in_market` both defaulting to
/// `0.0`, `accepts` reduced to exactly one comparison:
/// `payoff_ratio >= min_payoff_ratio`. That single comparison gated EVERY
/// survival path in the quality screen — both the strict and the opportunistic
/// branch require `profile_ok` (see the screen, below in this file). So the
/// payoff ratio alone decided who lived.
///
/// A payoff ratio cannot do that job, because it says nothing about money. It is
/// `avg_win / avg_loss`: a description of the SHAPE of the win/loss split, blind
/// to how often each occurs and blind to what the broker charges. Measured on
/// real EURUSD bars while sweeping the trailing-stop geometry:
///
///   trail multiplier 1.0 → payoff 0.91, expectancy -4.15 pips/trade
///   trail multiplier 3.0 → payoff 2.53, expectancy -4.18 pips/trade
///
/// The payoff ratio moved by a factor of 2.8. The money did not move at all. On
/// a driftless price, exit geometry REDISTRIBUTES the (win-rate, payoff) split
/// and their product stays pinned at minus the cost. A 2.0 payoff floor accepts
/// the second row and rejects the first, and the second row empties the account
/// 0.7 % faster than the first.
///
/// That is the reward hack this gate exists to close, and it is not hypothetical:
/// the same commit that made the trailing stop searchable would have handed the
/// GA a free way to clear a 2.0 payoff floor by widening the trail. Making the
/// trail searchable WITHOUT this gate is strictly worse than changing neither.
///
/// The payoff floor remains, as a secondary filter. It expresses a real operator
/// preference — a 2:1 system survives a losing run differently from a 0.6:1
/// system at the same expectancy — and it can only narrow what the expectancy
/// gate already admitted. It can never admit anything on its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TargetProfile {
    /// PRIMARY. Lowest acceptable cost-charged net expectancy per trade, in
    /// account currency. `0.0` means "must be strictly positive", NOT "no
    /// preference" — see the type doc.
    pub min_net_expectancy_per_trade: f64,
    /// PRIMARY. How many standard errors above zero that expectancy must sit.
    /// `0.0` requires only the sign.
    ///
    /// This bounds SAMPLING noise on one candidate's own trades. It does not
    /// bound selection bias across the thousands of candidates the GA tried —
    /// only DSR/PBO over the per-trial return series can do that. Those series
    /// are now persisted (`trial_returns.rs`); nothing reads them yet.
    ///
    /// SHIPPED AT 0.0, deliberately and on the record. `TargetProfile::evaluate`
    /// guards this check with `> 0.0`, so at the shipped value
    /// `net_expectancy_stderr` and `net_expectancy_t_stat` are computed on every
    /// candidate and consulted on none, and the objective reduces to
    /// `profit_per_trade > 0` — in-sample net profit over the screen window.
    /// What currently carries the load against a lucky sample is
    /// `prop_search_val_min_trades_per_month` (15 strict, 10 opportunistic): a
    /// trade-COUNT floor, which over a ten-year window demands ~1200 trades and
    /// so does defeat the two-lucky-trades case. It is not a noise bound.
    ///
    /// It ships off because the 2026-08-09 diagnostic run must first establish
    /// what the ten rejection counters look like with no new gate binding; a
    /// significance floor introduced in the same run would confound that
    /// baseline. `t >= 2.0` is the value to set once the baseline is read — at
    /// 1200+ trades it costs almost nothing to clear if there is a real edge.
    pub min_expectancy_t_stat: f64,
    /// Lowest acceptable win rate, as a fraction. `0.0` = no preference.
    pub min_win_rate: f64,
    /// SECONDARY. Lowest acceptable average-win over average-loss. `0.0` = no
    /// preference.
    ///
    /// Stated separately from the win rate because `profit_factor` folds the two
    /// together: 30 % of trades at 5:1 and 70 % at 0.6:1 both give about 2.1, and
    /// they are completely different systems to hold through a losing run.
    /// Never sufficient on its own — payoff 2.53 at expectancy -4.18 pips is a
    /// gate-passing money-loser.
    pub min_payoff_ratio: f64,
    /// Most of the span a candidate may spend holding a position. `0.0` = no
    /// preference.
    ///
    /// A strategy in the market almost always is not selecting entries, and its
    /// win rate converges on the market's base rate however the entry rule is
    /// written.
    pub max_in_market: f64,
}

/// Why a candidate was refused by [`TargetProfile::accepts`].
///
/// Named, one variant per criterion, because "rejected" with no reason is what
/// the quality screen used to report: a single `rejected_base_quality` counter
/// standing in for at least eight independent gates, from which no run could
/// ever say WHY it found nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetProfileRejection {
    /// The average trade loses money after costs. The one unconditional refusal.
    NegativeNetExpectancy,
    /// The expectancy is positive but inside its own sampling noise.
    ExpectancyNotSignificant,
    TooFewWinners,
    PayoffTooLow,
    TooMuchTimeInMarket,
}

impl TargetProfileRejection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NegativeNetExpectancy => "net_expectancy",
            Self::ExpectancyNotSignificant => "expectancy_significance",
            Self::TooFewWinners => "win_rate",
            Self::PayoffTooLow => "payoff_ratio",
            Self::TooMuchTimeInMarket => "in_market",
        }
    }
}

impl TargetProfile {
    /// `Ok(())` when the candidate may survive, or the FIRST criterion it failed.
    ///
    /// Order matters and is deliberate: the money question is asked first, so a
    /// rejection census reads "most candidates lose money" rather than "most
    /// candidates have the wrong shape" when both are true.
    pub fn evaluate(&self, metrics: &StrategyMetrics) -> Result<(), TargetProfileRejection> {
        // UNCONDITIONAL. Note the strict `>`, and note that it is NOT guarded by
        // `if self.min_net_expectancy_per_trade > 0.0`. Every other criterion on
        // this struct is opt-in; this one is the floor under all of them. A
        // candidate with zero trades reports 0.0 here and is refused, which is
        // also correct: nothing traded is not an edge.
        if !(metrics.profit_per_trade > self.min_net_expectancy_per_trade) {
            return Err(TargetProfileRejection::NegativeNetExpectancy);
        }
        if self.min_expectancy_t_stat > 0.0
            && metrics.net_expectancy_t_stat < self.min_expectancy_t_stat
        {
            return Err(TargetProfileRejection::ExpectancyNotSignificant);
        }
        if self.min_win_rate > 0.0 && metrics.win_rate < self.min_win_rate {
            return Err(TargetProfileRejection::TooFewWinners);
        }
        // SECONDARY, and only ever subtractive: by the time control reaches this
        // line the candidate has already proven it makes money after costs.
        if self.min_payoff_ratio > 0.0 && metrics.payoff_ratio < self.min_payoff_ratio {
            return Err(TargetProfileRejection::PayoffTooLow);
        }
        // Exposure rejects only when it was measurable. A candidate whose trades
        // carry no exit times reports 0.0, and reading that as "never in the
        // market" would admit exactly the ones this is meant to catch.
        if self.max_in_market > 0.0
            && metrics.in_market_pct > 0.0
            && metrics.in_market_pct > self.max_in_market
        {
            return Err(TargetProfileRejection::TooMuchTimeInMarket);
        }
        Ok(())
    }

    /// Whether `metrics` may survive. Never vacuously true — see
    /// [`Self::evaluate`] and the type doc.
    pub fn accepts(&self, metrics: &StrategyMetrics) -> bool {
        self.evaluate(metrics).is_ok()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// THE EARLY-REJECT PREDICATE, AND THE BATCH LEDGER THAT MAKES IT HONEST
//
// WHAT IT IS FOR, WITH THE NUMBER THAT JUSTIFIES IT.
//
// On the run at `docs/measurements/3090-47260276/card-run-valid.log`
// (2026-08-09, EURUSD M5, 843,456 rows) the quality screen cost 88,971 ms —
// **50.4% of the run's wall time** — and took 174 candidates in and **0** out.
// All 174 died on `rejected_base_quality`: 0 on regime, 0 on Monte-Carlo, 0 on
// spread sensitivity. The base-quality criteria ARE the expectancy / payoff /
// win-rate floors on `TargetProfile`, and every one of those floors can be
// evaluated against fields the GA population already carries, for free, before
// the screen starts. The best gene the whole run produced had profit factor
// 0.92 and net EUR -50,682 over 136 months: profit factor below 1 means
// expectancy is negative BY CONSTRUCTION.
//
// So half a run's wall time was spent proving something the population's own
// numbers already said.
//
// THE BIAS IS EXPLICIT AND IT IS TOWARD PASSING.
//
// A FALSE REJECT is invisible and permanent — the batch is gone, no artifact
// records what it would have found, and nothing in the run says a survivor was
// discarded. A FALSE ACCEPT costs time and nothing else. So this predicate is
// built to be WRONG IN ONE DIRECTION ONLY. It rejects only when all three of
// these hold at once, and passes the batch through on any doubt whatsoever:
//
//   1. enough of the population carries measured metrics at all
//      ([`EARLY_REJECT_MIN_MEASURED`]) — a thin sample is uncertainty, not
//      evidence. This is the leg that answers the run1-baseline observation
//      that the GA archive was 0/200 for four generations and 289/527 by
//      generation 527: a predicate that fires on "the archive looks empty" is a
//      false-reject generator;
//   2. NOT ONE candidate has a gross edge (`profit_factor >= 1.0`). This is the
//      certainty leg and it is arithmetic, not judgement: `profit_factor < 1`
//      means gross losses exceeded gross wins, so `net_profit < 0`, so
//      `expectancy = net_profit / trades < 0`, so
//      `TargetProfile::evaluate` refuses it unconditionally at its first line;
//   3. the best candidate's cost-charged expectancy is below the operator's
//      CONFIGURED floor by a stated margin.
//
// THE THRESHOLD IS READ, NEVER INVENTED. "Below target" means below
// `TargetProfile::min_net_expectancy_per_trade`, which is
// `models.prop_search_min_net_expectancy_per_trade` — the same field the
// quality screen's own primary gate reads, in the same units (account currency
// per trade; `Gene::expectancy` is `net_profit / trade_count` from
// `eval.rs`, booked after spread, commission and swap). The margin below is NOT
// a second threshold on the objective: it only ever makes the predicate more
// permissive than the configured floor, so it can never reject something the
// operator's own target would have admitted.
// ═════════════════════════════════════════════════════════════════════════════

/// Fewest candidates that must carry measured metrics before the predicate is
/// allowed to reject anything.
///
/// Not a tuning knob for the objective — a sample-size floor for the DECISION.
/// Below it the predicate reports `Uncertain` and passes.
pub const EARLY_REJECT_MIN_MEASURED: usize = 32;

/// How far below the configured floor the best candidate must sit before the
/// batch is abandoned, as a fraction of the population's own expectancy SCALE
/// (mean absolute expectancy over the measured candidates).
///
/// Scale-relative rather than absolute because expectancy is in account
/// currency and a run on a 100 EUR balance and a run on a 100,000 EUR balance
/// have different natural magnitudes; a fixed cushion would be meaningless on
/// one of them. The margin exists because the GA scores on its own evaluation
/// window while the quality screen re-simulates, so the two numbers are not the
/// same measurement — the cushion is the price of that difference, paid in the
/// safe direction.
pub const EARLY_REJECT_MARGIN_FRACTION: f64 = 0.25;

/// Why a batch was abandoned, or why it was not.
///
/// Every variant is COUNTED and NAMED. A batch abandoned with no record is the
/// silent drop this codebase has spent the day closing, one level up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchRejectReason {
    /// Not one candidate in the batch made money gross, and the best
    /// cost-charged expectancy is below the configured floor by the stated
    /// margin. The only reason that rejects.
    NoCandidateClearsExpectancyFloor,
}

impl BatchRejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoCandidateClearsExpectancyFloor => "no_candidate_clears_expectancy_floor",
        }
    }
}

/// Why a batch was KEPT. Named too, because "we did not reject it" and "we
/// could not tell" are different facts and only one of them is evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAcceptReason {
    /// A candidate cleared the floor (or has a gross edge). Real evidence.
    CandidateClearsFloor,
    /// Too few candidates carried measured metrics to decide. Passed on
    /// uncertainty — the deliberate bias.
    UncertainTooFewMeasured,
    /// The best candidate is below the floor but not by the margin. Passed on
    /// uncertainty.
    UncertainWithinMargin,
    /// No candidate carried metrics at all (an empty or unevaluated
    /// population). Passed on uncertainty.
    UncertainNoMetrics,
}

impl BatchAcceptReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CandidateClearsFloor => "candidate_clears_floor",
            Self::UncertainTooFewMeasured => "uncertain_too_few_measured",
            Self::UncertainWithinMargin => "uncertain_within_margin",
            Self::UncertainNoMetrics => "uncertain_no_metrics",
        }
    }
}

/// The predicate's answer, with every number that produced it. Nothing here is
/// derived later or re-read from elsewhere — a verdict that cannot be
/// reconstructed from its own fields is not auditable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchVerdict {
    pub rejected: Option<BatchRejectReason>,
    pub accepted: Option<BatchAcceptReason>,
    /// Candidates in the population the predicate saw.
    pub population: usize,
    /// Candidates that carried usable measured metrics.
    pub measured: usize,
    /// Best cost-charged expectancy per trade, account currency.
    pub best_expectancy: f64,
    /// Best profit factor across the measured candidates.
    pub best_profit_factor: f64,
    /// Payoff ratio implied by the best-expectancy candidate's `(pf, win_rate)`
    /// as `pf * (1 - p) / p` — the same expression the Risky ranking already
    /// computes. Reported, never used to reject: the payoff floor is SECONDARY
    /// and, per `TargetProfile`'s own doc, can only ever narrow what the
    /// expectancy gate already admitted.
    pub best_payoff_ratio: f64,
    /// Trades behind `best_expectancy`.
    pub best_trades: usize,
    /// The configured floor this was judged against.
    pub floor: f64,
    /// The cushion granted below the floor.
    pub margin: f64,
}

impl BatchVerdict {
    pub fn is_reject(&self) -> bool {
        self.rejected.is_some()
    }

    pub fn reason(&self) -> &'static str {
        match (self.rejected, self.accepted) {
            (Some(r), _) => r.as_str(),
            (None, Some(a)) => a.as_str(),
            // Unreachable by construction — `evaluate_batch_early_reject`
            // always sets exactly one. Named rather than `unreachable!()`
            // because a panic on a reporting path is never the right trade.
            (None, None) => "unclassified",
        }
    }
}

/// THE PREDICATE. Cheap: `O(population)` field reads over metrics the GA's own
/// population evaluation already produced, so it cannot call the stages it
/// exists to skip. Honest: every input it used is on the returned verdict.
///
/// Reads `profile.min_net_expectancy_per_trade` — the operator's configured
/// objective — and never a threshold of its own. Note the strict `>` in
/// `TargetProfile::evaluate`: a floor of `0.0` means "must be strictly greater
/// than zero", so this predicate compares the same way.
pub fn evaluate_batch_early_reject(genes: &[Gene], profile: &TargetProfile) -> BatchVerdict {
    let floor = profile.min_net_expectancy_per_trade;
    let mut measured = 0usize;
    let mut best_expectancy = f64::NEG_INFINITY;
    let mut best_profit_factor = f64::NEG_INFINITY;
    let mut best_payoff_ratio = 0.0f64;
    let mut best_trades = 0usize;
    let mut abs_sum = 0.0f64;
    for gene in genes {
        // A gene that never traded has `expectancy = 0.0` by construction in
        // `eval.rs`, which is not a measurement of anything. Counting it would
        // let a batch of non-traders look like a batch of break-even
        // strategies — and at a floor of 0.0 that is the difference between
        // "uncertain" and "rejected".
        if gene.trades_count == 0 || !gene.expectancy.is_finite() {
            continue;
        }
        measured += 1;
        abs_sum += gene.expectancy.abs();
        if gene.profit_factor.is_finite() && gene.profit_factor > best_profit_factor {
            best_profit_factor = gene.profit_factor;
        }
        if gene.expectancy > best_expectancy {
            best_expectancy = gene.expectancy;
            best_trades = gene.trades_count;
            let p = gene.win_rate.clamp(0.0, 1.0);
            let pf = gene.profit_factor.max(0.0);
            best_payoff_ratio = if p > 0.0 && p < 1.0 {
                pf * (1.0 - p) / p
            } else {
                0.0
            };
        }
    }

    let scale = if measured > 0 {
        abs_sum / measured as f64
    } else {
        0.0
    };
    let margin = EARLY_REJECT_MARGIN_FRACTION * scale;

    let mut verdict = BatchVerdict {
        rejected: None,
        accepted: None,
        population: genes.len(),
        measured,
        best_expectancy: if measured > 0 { best_expectancy } else { 0.0 },
        best_profit_factor: if measured > 0 { best_profit_factor } else { 0.0 },
        best_payoff_ratio,
        best_trades,
        floor,
        margin,
    };

    // ── Leg 1: is there anything to decide on? ──────────────────────────────
    if measured == 0 {
        verdict.accepted = Some(BatchAcceptReason::UncertainNoMetrics);
        return verdict;
    }
    if measured < EARLY_REJECT_MIN_MEASURED {
        verdict.accepted = Some(BatchAcceptReason::UncertainTooFewMeasured);
        return verdict;
    }
    // ── Leg 2: the certainty leg. Any gross edge anywhere ⇒ pass. ───────────
    // `profit_factor >= 1.0` on even one candidate means at least one
    // candidate did not lose money gross, which is exactly the case the
    // predicate must never touch.
    if best_profit_factor >= 1.0 {
        verdict.accepted = Some(BatchAcceptReason::CandidateClearsFloor);
        return verdict;
    }
    // ── Leg 3: below the CONFIGURED floor, by the margin. ───────────────────
    if best_expectancy >= floor - margin {
        verdict.accepted = Some(BatchAcceptReason::UncertainWithinMargin);
        return verdict;
    }
    verdict.rejected = Some(BatchRejectReason::NoCandidateClearsExpectancyFloor);
    verdict
}

/// Per-batch accounting, shaped after `neoethos_data::core::indicator_ledger`
/// — reason, count, named examples, one census line — because that is the shape
/// this codebase already trusts. It is a NEW ledger rather than that one:
/// `IndicatorLedger` never crosses the crate boundary (see the census comment
/// in `run_discovery_cycle_with_progress`, which says in its own words that
/// from the search crate "only presence is observable").
#[derive(Debug, Clone, Default)]
pub struct BatchRejectionLedger {
    pub batches_seen: usize,
    pub batches_rejected: usize,
    pub rejected_no_candidate_clears_floor: usize,
    pub accepted_candidate_clears_floor: usize,
    pub accepted_uncertain_no_metrics: usize,
    pub accepted_uncertain_too_few_measured: usize,
    pub accepted_uncertain_within_margin: usize,
    /// `(cursor, reason, best_expectancy, best_payoff_ratio, best_trades)` for
    /// the first rejections, so the census names batches rather than counting
    /// them anonymously.
    pub rejected_examples: Vec<(usize, &'static str, f64, f64, usize)>,
}

impl BatchRejectionLedger {
    const MAX_EXAMPLES: usize = 24;

    pub fn record(&mut self, cursor: usize, verdict: &BatchVerdict) {
        self.batches_seen += 1;
        match (verdict.rejected, verdict.accepted) {
            (Some(BatchRejectReason::NoCandidateClearsExpectancyFloor), _) => {
                self.batches_rejected += 1;
                self.rejected_no_candidate_clears_floor += 1;
                if self.rejected_examples.len() < Self::MAX_EXAMPLES {
                    self.rejected_examples.push((
                        cursor,
                        verdict.reason(),
                        verdict.best_expectancy,
                        verdict.best_payoff_ratio,
                        verdict.best_trades,
                    ));
                }
            }
            (None, Some(BatchAcceptReason::CandidateClearsFloor)) => {
                self.accepted_candidate_clears_floor += 1
            }
            (None, Some(BatchAcceptReason::UncertainNoMetrics)) => {
                self.accepted_uncertain_no_metrics += 1
            }
            (None, Some(BatchAcceptReason::UncertainTooFewMeasured)) => {
                self.accepted_uncertain_too_few_measured += 1
            }
            (None, Some(BatchAcceptReason::UncertainWithinMargin)) => {
                self.accepted_uncertain_within_margin += 1
            }
            (None, None) => {}
        }
    }

    /// The run-end tally. Printed whenever any batch was seen, including when
    /// none were rejected — "we streamed 40 batches and rejected none" is a
    /// result, and a census that only appears on rejection cannot say it.
    pub fn log_summary(&self, stage: &str) {
        if self.batches_seen == 0 {
            return;
        }
        tracing::info!(
            target: "neoethos_search::batch_ledger",
            stage,
            batches_seen = self.batches_seen,
            batches_rejected = self.batches_rejected,
            rejected_no_candidate_clears_floor = self.rejected_no_candidate_clears_floor,
            accepted_candidate_clears_floor = self.accepted_candidate_clears_floor,
            accepted_uncertain_no_metrics = self.accepted_uncertain_no_metrics,
            accepted_uncertain_too_few_measured = self.accepted_uncertain_too_few_measured,
            accepted_uncertain_within_margin = self.accepted_uncertain_within_margin,
            rejected_examples = ?self.rejected_examples,
            "streaming batch census — every abandoned batch, by cursor and reason. The three \
             `uncertain_*` buckets are batches the predicate PASSED because it could not tell; \
             they are the cost of never being able to discard a survivor."
        );
    }
}

/// The process-wide batch ledger.
///
/// A run is a sequence of batches and the tally belongs to the RUN, not to one
/// cycle — but the cycle is where the predicate fires. So the cycle records
/// here and the loop (or anything else that wants it) reads one census.
static BATCH_LEDGER: std::sync::Mutex<Option<BatchRejectionLedger>> =
    std::sync::Mutex::new(None);

fn with_batch_ledger<T>(f: impl FnOnce(&mut BatchRejectionLedger) -> T) -> T {
    let mut guard = BATCH_LEDGER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(guard.get_or_insert_with(BatchRejectionLedger::default))
}

/// Record one batch verdict in the process ledger.
pub fn record_batch_verdict(cursor: usize, verdict: &BatchVerdict) {
    with_batch_ledger(|ledger| ledger.record(cursor, verdict));
}

/// Snapshot of the process ledger.
pub fn batch_rejection_ledger() -> BatchRejectionLedger {
    with_batch_ledger(|ledger| ledger.clone())
}

/// Print the run-end batch census.
pub fn log_batch_rejection_summary(stage: &str) {
    with_batch_ledger(|ledger| ledger.log_summary(stage));
}

/// Reset the process ledger. For tests and for a caller that runs several
/// independent streaming searches in one process.
pub fn reset_batch_rejection_ledger() {
    with_batch_ledger(|ledger| *ledger = BatchRejectionLedger::default());
}

// ═════════════════════════════════════════════════════════════════════════════
// THE SWAP LOOP
//
// WHERE IT LIVES, AND WHY NOT IN `run_discovery_cycle_with_progress`.
//
// `run_discovery_cycle_with_progress` takes `features: &FeatureFrame` and never
// rebuilds it: the working set is chosen ENTIRELY outside this function, in the
// orchestrator that calls `prepare_multitimeframe_features` and then the cycle.
// So the loop goes AROUND that pair, which is why it is expressed here as a
// driver over two closures rather than as surgery inside the cycle — and why
// the parity test is trivial to write: with one batch covering the whole space
// and the predicate never firing, the loop performs exactly one build and one
// cycle, which is today's path.
//
// A DECISION THAT HAD TO BE MADE BEFORE THE LOOP COULD BE WRITTEN, and which
// the design doc never raises: **survivors from different batches cannot share
// a portfolio.** Gene indices address `DiscoveryResult::effective_feature_names`
// — the cube THAT batch built — portfolio selection correlates candidates
// against each other, and export projects by name. A portfolio assembled from
// batch 3 and batch 11 would reference columns no single cube ever held.
//
// DECIDED: **cross-batch portfolios are forbidden.** Each batch produces its
// own `DiscoveryResult`, with its own `effective_feature_names`, and the loop
// returns them as a list. It never merges them. The alternative — rebuilding a
// union cube over the surviving batches' columns — is cheap (survivors are few)
// and is a legitimate follow-up, but it changes what a portfolio MEANS and so
// is not something to do silently inside a loop.
// ═════════════════════════════════════════════════════════════════════════════

/// The cursor of the streaming working set in force, or `0` when none is
/// installed (the non-streaming path — one implicit batch at cursor 0).
///
/// Read rather than passed so the cycle's signature does not have to change for
/// callers that do not stream; the working set is installed by
/// `neoethos_data::with_extended_sweep_working_set` around the feature build
/// and is a pure function of `(cursor, batch_columns)`.
pub fn streaming_sweep_cursor() -> usize {
    neoethos_data::core::hpc_ta::current_extended_sweep_working_set()
        .map(|batch| batch.cursor)
        .unwrap_or(0)
}

/// A streaming search: a cursor through the (indicator, period) space, a
/// hardware-derived batch width, and the census of what it abandoned.
pub struct StreamingSearch {
    cursor: usize,
    batch_columns: usize,
    space_len: usize,
    budget_rows: usize,
    batches_started: usize,
}

impl StreamingSearch {
    /// Size the working set from the machine, against the run's WIDEST frame.
    ///
    /// `budget_rows` is the base timeframe's bar count — the same number
    /// `compute_hpc_feature_frame_sized` is given, for the same reason: the
    /// batch must not be a function of which timeframe is being built, or the
    /// per-TF cube widths diverge and the cube cannot be assembled.
    pub fn new(budget_rows: usize) -> Self {
        let batch_columns = neoethos_data::core::hpc_ta::streaming_batch_columns(budget_rows);
        let space_len = neoethos_data::core::hpc_ta::extended_sweep_space_len();
        tracing::info!(
            target: "neoethos_search::streaming",
            budget_rows,
            batch_columns,
            resident_columns =
                neoethos_data::core::hpc_ta::planned_resident_columns(budget_rows),
            space_len,
            "streaming working set sized from FREE RAM and the widest frame — never from a \
             config constant. batch_columns of 0 means this machine cannot afford any \
             streaming extension at all."
        );
        Self {
            cursor: 0,
            batch_columns,
            space_len,
            budget_rows,
            batches_started: 0,
        }
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn batch_columns(&self) -> usize {
        self.batch_columns
    }

    pub fn space_len(&self) -> usize {
        self.space_len
    }

    pub fn budget_rows(&self) -> usize {
        self.budget_rows
    }

    pub fn batches_started(&self) -> usize {
        self.batches_started
    }

    /// The next working set, or `None` when the space is exhausted or the
    /// machine affords no extension.
    ///
    /// NEVER WRAPS. Running off the end returns `None` so the caller decides
    /// what a second pass means; a silent wrap would re-explore parameter
    /// regions the run already rejected and report them as new.
    pub fn next_batch(
        &mut self,
    ) -> Option<std::sync::Arc<neoethos_data::core::hpc_ta::SweepBatch>> {
        if self.batch_columns == 0 || self.cursor >= self.space_len {
            return None;
        }
        let batch =
            neoethos_data::core::hpc_ta::extended_sweep_batch(self.cursor, self.batch_columns);
        if batch.is_empty() {
            return None;
        }
        self.cursor = batch.next_cursor;
        self.batches_started += 1;
        Some(std::sync::Arc::new(batch))
    }

    /// Drive the loop: build a cube per batch, run one discovery cycle on it,
    /// keep the results that produced a portfolio.
    ///
    /// The predicate is NOT applied here — it fires inside
    /// `run_discovery_cycle_with_progress`, before the quality screen, which is
    /// the only place early enough to skip the 50.4%. This loop reads the
    /// process ledger to know whether the batch it just ran was abandoned.
    ///
    /// `max_batches` of `0` means "until the space is exhausted".
    pub fn run<B, C>(
        &mut self,
        max_batches: usize,
        mut build_features: B,
        mut run_cycle: C,
    ) -> Result<Vec<DiscoveryResult>>
    where
        B: FnMut(&std::sync::Arc<neoethos_data::core::hpc_ta::SweepBatch>) -> Result<FeatureFrame>,
        C: FnMut(&FeatureFrame) -> Result<DiscoveryResult>,
    {
        let mut survivors: Vec<DiscoveryResult> = Vec::new();
        let mut ran = 0usize;
        while max_batches == 0 || ran < max_batches {
            let Some(batch) = self.next_batch() else {
                break;
            };
            let cursor = batch.cursor;
            let before = batch_rejection_ledger().batches_rejected;
            let features = build_features(&batch)?;
            let result = run_cycle(&features)?;
            let after = batch_rejection_ledger().batches_rejected;
            let abandoned = after > before;
            tracing::info!(
                target: "neoethos_search::streaming",
                cursor,
                next_cursor = batch.next_cursor,
                space_len = batch.space_len,
                batch_pairs = batch.pairs.len(),
                batch_columns = batch.planned_columns,
                abandoned,
                portfolio = result.portfolio.len(),
                "streaming batch complete"
            );
            if !abandoned && !result.portfolio.is_empty() {
                // Kept WHOLE, never merged with another batch's portfolio —
                // gene indices address this result's own
                // `effective_feature_names`. See the module note above.
                survivors.push(result);
            }
            ran += 1;
        }
        log_batch_rejection_summary("streaming_search");
        Ok(survivors)
    }
}

fn survived_the_backtest(metrics: &StrategyMetrics) -> bool {
    // `max_drawdown_pct` is a fraction despite the name — `(peak - equity) / peak`
    // in quality.rs — so total loss is 1.0, and the 403.1 % in that log line is
    // stored as 4.031. Reading it as a percentage would let every ruined
    // candidate through.
    metrics.max_drawdown_pct < 1.0
}

fn passes_strict_quality(metrics: &StrategyMetrics, cfg: &crate::genetic::FilteringConfig) -> bool {
    if !survived_the_backtest(metrics) {
        return false;
    }
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
    if !survived_the_backtest(metrics) {
        return false;
    }
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
    /// Alignment-semantics version baked into the policy hash. Bumped to
    /// "closed-htf-bar-only-v2" for audit D02 (2026-07-13): higher-TF
    /// features now become available at bar CLOSE (stamp + period), not at
    /// the containing bucket's open stamp. Cubes/artifacts built under the
    /// old lookahead alignment hash differently and cannot silently mix
    /// with post-D02 evidence.
    mtf_alignment: &'static str,
}

/// See [`DiscoveryTemporalPolicy::mtf_alignment`].
const MTF_ALIGNMENT_POLICY_VERSION: &str = "closed-htf-bar-only-v2";

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
    /// The FOURTH field of the same exit policy. Added 2026-08-09: it was
    /// omitted while its three siblings were hashed, so two artifacts produced
    /// under different min-lock values hashed identically and were
    /// indistinguishable afterwards — reproducing the exact defect class the
    /// field was introduced to close (it used to travel by a separate route and
    /// silently inherit `BacktestSettings::default() = 2.0`). Adding it changes
    /// the policy hash and therefore invalidates cached artifact identity across
    /// this change, which is correct: artifacts on either side were produced
    /// under a policy the hash could not tell apart.
    trailing_min_lock_pips: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    kill_zones_enabled: bool,
    /// Stop regime (2026-08-08): an adaptive gene's artifact is produced under
    /// `sl = adaptive_vol_mult × base[i]`, a DIFFERENT policy from the fixed
    /// `sl_pips`/`tp_pips` above — the hash must not conflate the two. The
    /// base series itself is deterministic from the dataset (covered by the
    /// dataset hash).
    adaptive_vol_mult: f64,
    adaptive_rr: f64,
}

fn discovery_temporal_contract(
    config: &DiscoveryConfig,
    feature_names: &[String],
) -> Result<TemporalFeatureContract> {
    let feature_policy_hash = stable_json_hash(&DiscoveryTemporalPolicy {
        timeframe_label: &config.timeframe_label,
        higher_timeframes: &config.higher_timeframes,
        feature_names,
        mtf_alignment: MTF_ALIGNMENT_POLICY_VERSION,
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
    let n = features.n_samples();
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
        row_count: features.n_samples(),
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
        trailing_min_lock_pips: settings.trailing_min_lock_pips,
        pip_value: settings.pip_value,
        spread_pips: settings.spread_pips,
        commission_per_trade: settings.commission_per_trade,
        pip_value_per_lot: settings.pip_value_per_lot,
        kill_zones_enabled: settings.kill_zones_enabled,
        adaptive_vol_mult: settings.adaptive_vol_mult,
        adaptive_rr: settings.adaptive_rr,
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

fn walkforward_summary_passed(summary: &WalkforwardSummary, mode: DiscoveryMode) -> bool {
    if summary.walk_forward_splits == 0 {
        return false;
    }
    if matches!(mode, DiscoveryMode::Risky) {
        // Risky = fast capital multiplication, drawdown-agnostic. The prop-firm
        // per-window rules (daily-loss / consistency / trade-limit / min trading
        // days) are FTMO constraints — irrelevant here and brutal enough to
        // reject every aggressive compounder (one bad regime window kills it).
        // The robustness bar that actually matters for risky is GENERALISATION:
        // positive AVERAGE out-of-sample PnL AND a MAJORITY of walk-forward folds
        // individually profitable. Walk-forward still RUNS + is recorded; this is
        // just the risky-appropriate pass bar.
        let positive_folds = summary.splits.iter().filter(|s| s.pnl > 0.0).count();
        let positive_frac = if summary.splits.is_empty() {
            0.0
        } else {
            positive_folds as f64 / summary.splits.len() as f64
        };
        return summary.avg_pnl > 0.0 && positive_frac >= 0.60;
    }
    // PropFirm / Strict: demand full prop-firm robustness across EVERY window.
    summary.avg_pnl > 0.0
        && !summary.any_daily_loss_breach
        && !summary.any_consistency_violation
        && !summary.any_trade_limit_violation
        && summary.all_min_trading_days_ok
}

fn evaluate_cpcv_gate(
    portfolio: &[Gene],
    // AREA 2 / Stage B (2026-06-09): the GPU population path re-synthesizes each
    // gene's signals on-device from the GATHERED indicators + GATHERED full-series
    // SMC (pointwise, so it reproduces the full-series signals at `absolute_idx`),
    // so the precomputed `portfolio_signals` are no longer gathered here. Kept in
    // the signature for the caller's alignment sanity check below.
    portfolio_signals: &[Vec<i8>],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f32,
    months: &[i64],
    days: &[i64],
    pbo_candidates: &[Gene],
) -> Result<(bool, usize, f64, Option<f64>, bool)> {
    if portfolio.is_empty() {
        return Ok((false, 0, 0.0, None, true));
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
        return Ok((true, 0, 1.0, None, true));
    }

    let n = ohlcv.close.len();
    let capped_n = if config.cpcv_max_rows > 0 {
        config.cpcv_max_rows.min(n)
    } else {
        n
    };
    let offset = n.saturating_sub(capped_n);
    // ── COVERAGE, REPORTED (2026-08-10) 💰 ──────────────────────────────────
    //
    // This gate validates the TAIL ONLY — `offset = n - capped_n` — and until
    // now it returned pass/fail with no coverage figure anywhere. 200 000 rows
    // against 1.05 M bars is 19% of history: the out-of-sample gate that
    // promotes a strategy toward real money reported a clean pass on a fifth of
    // the record, and nothing in the run said which fifth.
    //
    // Nothing is refused here. `cpcv_max_rows` is a memory bound, and refusing
    // a run because the operator's box is small is the wrong trade. What
    // changes is that the number is now in the log next to the verdict, so a
    // "CPCV passed" can be read together with what it passed on.
    let coverage_fraction = if n > 0 {
        capped_n as f64 / n as f64
    } else {
        0.0
    };
    if coverage_fraction < 0.50 {
        tracing::warn!(
            target: "neoethos_search::discovery",
            cpcv_max_rows = config.cpcv_max_rows,
            rows_available = n,
            rows_validated = capped_n,
            first_validated_row = offset,
            coverage_fraction,
            "CPCV COVERAGE IS BELOW HALF THE LOADED HISTORY. The gate validates the \
             most recent rows only, so a pass here is a statement about the tail of the \
             dataset and not about the record. Raise models.cpcv_max_rows (0 = every \
             row) if the box has the memory."
        );
    } else {
        tracing::info!(
            target: "neoethos_search::discovery",
            cpcv_max_rows = config.cpcv_max_rows,
            rows_available = n,
            rows_validated = capped_n,
            first_validated_row = offset,
            coverage_fraction,
            "CPCV coverage (tail-anchored: the gate validates the most recent \
             rows_validated bars)"
        );
    }
    let cv = CombinatorialPurgedCV::new(
        config.cpcv_n_splits,
        config.cpcv_n_test_groups,
        config.cpcv_embargo_pct,
        config.cpcv_purge_pct,
    );
    let splits = cv.split(capped_n);
    if splits.is_empty() {
        return Ok((false, 0, 0.0, None, true));
    }

    // Alignment sanity check: the GPU path re-synthesizes signals on-device from
    // the gathered full-series indicators/SMC, which is only equivalent to the
    // precomputed `portfolio_signals` when those were computed on the SAME full
    // series. A length mismatch means the caller built signals on a different
    // window — fail loud rather than silently validating against a stale series.
    if portfolio_signals.len() != portfolio.len() {
        anyhow::bail!(
            "CPCV gate: {} portfolio signals for {} genes — internal bug",
            portfolio_signals.len(),
            portfolio.len()
        );
    }
    if let Some((i, s)) = portfolio_signals
        .iter()
        .enumerate()
        .find(|(_, s)| s.len() != ohlcv.close.len())
    {
        anyhow::bail!(
            "CPCV gate: signals[{}].len()={} != full series len {} — signals must be \
             full-series aligned for the gathered GPU re-synthesis to match",
            i,
            s.len(),
            ohlcv.close.len()
        );
    }

    let mut fold_count = 0usize;
    let mut profitable_folds = 0usize;
    let eval_config = config
        .evaluation_config_with_smc_gate(ohlcv.close.last().copied(), effective_smc_gate_threshold);

    // AREA 2 / Stage B (2026-06-09) — GPU-route the CPCV gate.
    //
    // TRANSPOSE: was a nested loop over genes × folds, each (gene, fold) gathering
    // a non-contiguous index set and running a SINGLE-gene
    // `fast_evaluate_strategy_core`. Now batches ACROSS GENES PER FOLD: for each
    // fold we gather the per-sample arrays ONCE and fire ONE
    // `validation_genes_population_gathered` launch over the WHOLE portfolio
    // (GPU-try, CPU-fallback). portfolio×folds backtests → folds launches.
    //
    // PARITY: the gather happens HOST-SIDE (exactly as the old serial loop did),
    // so the population kernel consumes the SAME contiguous re-indexed buffer the
    // CPU built — byte-identical input, no kernel change. SMC is GATHERED from the
    // FULL-SERIES arrays at `absolute_idx` (NOT recomputed on the gathered slice,
    // which would break the cross-bar SMC lookback); see
    // `validation_genes_population_gathered`. Confidence/sizing match the serial
    // path: `discovery_backtest_settings` keeps `risk_based_sizing == true`, the
    // gene's REAL per-bar confidence is recomputed on-device pointwise, and
    // `timestamps = &[]` is honoured exactly as before. The fold-pass test below is
    // the EXACT condition the serial loop used (via `BacktestMetrics`, so trade-
    // count rounding is identical).
    //
    // Settings template: every field of `discovery_backtest_settings` except the
    // per-gene `sl_pips`/`tp_pips` is gene-INDEPENDENT (sourced from
    // `evaluation_config`), and the helper re-resolves per-gene SL/TP with the same
    // 20/40 fallback, so one template + the helper's per-gene SL/TP arrays
    // reproduce the per-gene settings the serial loop built.
    let settings_template = if let Some(gene) = portfolio.first() {
        PopulationTemplateResolver::new(config, ohlcv.close.last().copied()).template(gene)
    } else {
        return Ok((false, 0, 0.0, None, true));
    };

    // Full-series indicators + SMC, computed ONCE and gathered per fold. The SMC
    // arrays carry cross-bar lookback, so they MUST be derived on the full
    // contiguous series and then gathered — never recomputed on a gathered slice.
    let full_indicators = features.as_indicators_view();
    let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
        build_smc_arrays(features, ohlcv);
    let full_n = ohlcv.close.len();
    let mut full_smc: Vec<crate::eval::SmcRow> = Vec::with_capacity(full_n);
    for i in 0..full_n {
        full_smc.push([
            ob[i], fvg[i], liq[i], trend[i], prem[i], ind[i], bos[i], choch[i], eqh[i], eql[i],
            disp[i],
        ]);
    }

    for (_, test_idx) in &splits {
        if test_idx.is_empty() {
            continue;
        }
        let absolute_idx: Vec<usize> = test_idx.iter().map(|idx| offset + *idx).collect();
        let close: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.close[*idx]).collect();
        let high: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.high[*idx]).collect();
        let low: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.low[*idx]).collect();
        let fold_months: Vec<i64> = absolute_idx.iter().map(|idx| months[*idx]).collect();
        let fold_days: Vec<i64> = absolute_idx.iter().map(|idx| days[*idx]).collect();

        // ONE GPU launch over the whole portfolio on this gathered fold. Serialize
        // the device launch behind GPU_LAUNCH_LOCK so the (possible) outer
        // parallelism never spins up N GPU clients → VRAM × N → OOM. The
        // CPU-fallback inside the helper still parallelises across genes.
        let metrics_per_gene = {
            #[cfg(feature = "gpu")]
            let _gpu_guard = GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            validation_genes_population_gathered(
                full_indicators,
                &full_smc,
                portfolio,
                &eval_config,
                &settings_template,
                &absolute_idx,
                &close,
                &high,
                &low,
                &fold_months,
                &fold_days,
            )?
        };
        if metrics_per_gene.len() != portfolio.len() {
            anyhow::bail!(
                "CPCV fold eval returned {} metric rows for {} genes — internal bug",
                metrics_per_gene.len(),
                portfolio.len()
            );
        }

        for m in metrics_per_gene {
            // EXACT fold-pass criteria of the original serial loop — routed through
            // `BacktestMetrics` so net_profit/max_drawdown/trade_count (incl. the
            // trade-count rounding) are read identically.
            let metrics = BacktestMetrics::from_metric_array(m);
            fold_count += 1;
            let drawdown_ok =
                config.filtering.max_dd <= 0.0 || metrics.max_drawdown <= config.filtering.max_dd;
            if metrics.trade_count > 0 && metrics.net_profit > 0.0 && drawdown_ok {
                profitable_folds += 1;
            }
        }
    }

    if fold_count == 0 {
        return Ok((false, 0, 0.0, None, true));
    }
    let ratio = profitable_folds as f64 / fold_count as f64;

    // ── PBO — Probability of Backtest Overfitting (CSCV, López de Prado) ────
    // For each CPCV split: crown the IN-SAMPLE champion of the candidate pool
    // on the TRAIN side, then ask where that champion ranks OUT-of-sample on
    // the TEST side. PBO = fraction of splits where the champion lands at or
    // below the OOS median. High PBO ⇒ "the selection process is picking
    // luck" — the survivors' metrics were bought with trials, not edge.
    // Reuses the exact fold gather + population evaluator of the gate above,
    // so IS/OOS are measured with the same engine (costs, sizing, SMC).
    let mut pbo: Option<f64> = None;
    let mut pbo_passed = true;
    if config.max_pbo > 0.0 && pbo_candidates.len() >= 8 {
        let cands: Vec<Gene> = pbo_candidates.iter().take(64).cloned().collect();
        let eval_pool = |idx: &[usize]| -> Result<Vec<f64>> {
            let absolute_idx: Vec<usize> = idx.iter().map(|i| offset + *i).collect();
            let close: Vec<f64> = absolute_idx.iter().map(|i| ohlcv.close[*i]).collect();
            let high: Vec<f64> = absolute_idx.iter().map(|i| ohlcv.high[*i]).collect();
            let low: Vec<f64> = absolute_idx.iter().map(|i| ohlcv.low[*i]).collect();
            let m: Vec<i64> = absolute_idx.iter().map(|i| months[*i]).collect();
            let d: Vec<i64> = absolute_idx.iter().map(|i| days[*i]).collect();
            let metrics = {
                #[cfg(feature = "gpu")]
                let _gpu_guard = GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                validation_genes_population_gathered(
                    full_indicators,
                    &full_smc,
                    &cands,
                    &eval_config,
                    &settings_template,
                    &absolute_idx,
                    &close,
                    &high,
                    &low,
                    &m,
                    &d,
                )?
            };
            Ok(metrics
                .into_iter()
                .map(|arr| BacktestMetrics::from_metric_array(arr).net_profit)
                .collect())
        };

        let mut splits_evaluated = 0usize;
        let mut champion_below_median = 0usize;
        for (train_idx, test_idx) in &splits {
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }
            let is_perf = eval_pool(train_idx)?;
            let oos_perf = eval_pool(test_idx)?;
            if is_perf.len() != cands.len() || oos_perf.len() != cands.len() {
                anyhow::bail!("PBO: evaluator returned wrong candidate count — internal bug");
            }
            let Some(champion) = is_perf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            else {
                continue;
            };
            let mut oos_sorted = oos_perf.clone();
            oos_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Lower-middle median; ties count AGAINST the champion — the
            // conservative direction (slightly overestimates PBO).
            let median = oos_sorted[(oos_sorted.len() - 1) / 2];
            splits_evaluated += 1;
            if oos_perf[champion] <= median {
                champion_below_median += 1;
            }
        }
        if splits_evaluated > 0 {
            let p = champion_below_median as f64 / splits_evaluated as f64;
            pbo = Some(p);
            pbo_passed = p <= config.max_pbo;
            tracing::info!(
                target: "neoethos_search::discovery",
                pbo = format!("{p:.2}"),
                max_pbo = config.max_pbo,
                candidates = cands.len(),
                splits = splits_evaluated,
                passed = pbo_passed,
                "PBO gate — probability the in-sample champion is luck"
            );
            if !pbo_passed {
                tracing::warn!(
                    target: "neoethos_search::discovery",
                    "PBO {p:.2} exceeds the {:.2} ceiling — the selection looks like \
                     overfitting; export will be BLOCKED for this unit",
                    config.max_pbo
                );
            }
        }
    } else if config.max_pbo > 0.0 {
        tracing::info!(
            target: "neoethos_search::discovery",
            candidates = pbo_candidates.len(),
            "PBO not computed — needs ≥8 candidates in the selection pool \
             (gate does not block)"
        );
    }

    Ok((
        ratio >= config.cpcv_min_phi.clamp(0.0, 1.0),
        fold_count,
        ratio,
        pbo,
        pbo_passed,
    ))
}

fn build_discovery_validation_artifacts(
    portfolio: &[Gene],
    portfolio_signals: &[Vec<i8>],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f32,
    pbo_candidates: &[Gene],
    trials_tested: usize,
) -> Result<(
    DiscoveryValidationGates,
    Vec<CanonicalBacktestArtifactFile>,
    Vec<WalkforwardValidationArtifactFile>,
    Vec<bool>,
)> {
    let _scope = crate::eval_telemetry::CallerScope::enter("validation_artifacts");
    if portfolio.is_empty() {
        return Ok((
            DiscoveryValidationGates::pending(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
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
            n,
            mismatched
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
    // Per-gene walk-forward pass flags (aligned to `portfolio` order). The caller
    // uses this in Risky mode to FILTER the exported portfolio down to the genes
    // that individually clear the risky walk-forward bar (selection pressure)
    // instead of rejecting the whole portfolio when any single gene fails.
    let mut per_gene_wf: Vec<bool> = Vec::with_capacity(portfolio.len());

    // ── AREA 2 / Stage C (2026-06-09): GPU-routed POPULATION walk-forward ─────
    //
    // The single-gene `embargoed_walkforward_backtest` ran `n_genes × n_splits`
    // tiny CPU backtests. Transpose it: build the full-series indicators + SMC
    // ONCE, build the window-independent gene pack ONCE, then per qualifying
    // split do ONE GPU population launch (`validation_genes_population_window`)
    // over all survivor genes — `n_splits` launches instead of `n_genes ×
    // n_splits` CPU backtests. The kernel emits only the metric half; the risk
    // diagnostics stay on the CPU inside `embargoed_walkforward_population`,
    // which builds a byte-identical `WalkforwardSummary` per gene.
    //
    // The GPU metrics half is gene-independent except SL/TP (handled by the
    // pack's per-gene SL/TP arrays with the SAME finite-positive-else-20/40
    // fallback `discovery_backtest_settings` applies), so one template built
    // from `portfolio[0]` + the pack reproduces every gene's WF metrics.
    let wf_settings_template =
        PopulationTemplateResolver::new(config, ohlcv.close.last().copied())
            .template(&portfolio[0]);
    // ONE resolver over the full series: per-gene settings for the CPU
    // risk-diagnostic half (its SL/TP + adaptive mult drive
    // `simulate_trades_core`'s exits) and for the canonical full-series
    // backtest below. The walk-forward diagnostics re-base the adaptive series
    // per split window (see `embargoed_walkforward_population`), so what
    // matters here is that the gene's `stop_vol_mult` and reward:risk are
    // carried — the same regime the GPU metrics half runs.
    let wf_resolver = GeneEvalSettingsResolver::for_slice(
        config,
        portfolio.iter(),
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
    )?;
    let wf_gene_settings: Vec<crate::eval::BacktestSettings> = portfolio
        .iter()
        .map(|gene| wf_resolver.settings_for_gene(gene))
        .collect();
    let wf_eval_config = config
        .evaluation_config_with_smc_gate(ohlcv.close.last().copied(), effective_smc_gate_threshold);
    let wf_full_indicators = features.as_indicators_view();
    let (wob, wfvg, wliq, wtrend, wprem, wind, wbos, wchoch, weqh, weql, wdisp) =
        build_smc_arrays(features, ohlcv);
    let wf_full_n = ohlcv.close.len();
    let mut wf_full_smc: Vec<crate::eval::SmcRow> = Vec::with_capacity(wf_full_n);
    for i in 0..wf_full_n {
        wf_full_smc.push([
            wob[i], wfvg[i], wliq[i], wtrend[i], wprem[i], wind[i], wbos[i], wchoch[i], weqh[i],
            weql[i], wdisp[i],
        ]);
    }
    let wf_gene_pack = crate::genetic::WalkforwardPopulationGenePack::new(
        portfolio,
        &wf_eval_config,
        &wf_settings_template,
    );

    let walkforward_summaries = crate::validation::embargoed_walkforward_population(
        crate::validation::WalkforwardPopulationInput {
            close: &ohlcv.close,
            high: &ohlcv.high,
            low: &ohlcv.low,
            months: &months,
            days: &days,
            timestamps,
            train_ratio: 0.70,
            n_splits: config.walkforward_splits.max(1),
            embargo_bars,
            gene_settings: &wf_gene_settings,
            // THE pip the GPU metrics half scales its window base with — taken
            // from the pack itself (not re-resolved), so the CPU
            // risk-diagnostic half CANNOT run a different stop than the
            // metrics beside it.
            adaptive_pip: wf_gene_pack.adaptive_pip(),
            max_daily_loss_pct: config.max_regime_loss_pct,
            max_daily_profit_pct: 0.0,
            min_trading_days: 0,
            max_trades_per_day: 0,
            initial_balance: config.initial_balance,
        },
        portfolio_signals,
        |test_start, end| {
            // ONE GPU population launch over the whole portfolio on this
            // contiguous split window. Serialize the device launch behind
            // GPU_LAUNCH_LOCK so any outer parallelism never spins up N GPU
            // clients → VRAM × N → OOM. The CPU fallback inside the helper still
            // parallelises across genes.
            #[cfg(feature = "gpu")]
            let _gpu_guard = GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            crate::genetic::validation_genes_population_window(
                &wf_gene_pack,
                wf_full_indicators,
                &wf_full_smc,
                &ohlcv.close,
                &ohlcv.high,
                &ohlcv.low,
                &months,
                &days,
                timestamps,
                test_start,
                end,
            )
            // Metrics only for now: the device writes a trade list but nothing
            // reads it back yet, so the diagnostics still simulate this window
            // a second time on the CPU. Supplying `trades` here is what turns
            // that off — see `WindowEvaluation`.
            .map(crate::validation::WindowEvaluation::from)
        },
    )?;
    if walkforward_summaries.len() != portfolio.len() {
        anyhow::bail!(
            "walk-forward population returned {} summaries for {} genes — internal bug",
            walkforward_summaries.len(),
            portfolio.len()
        );
    }

    for ((gene, signals), walkforward_summary) in portfolio
        .iter()
        .zip(portfolio_signals)
        .zip(walkforward_summaries)
    {
        let settings = wf_resolver.settings_for_gene(gene);
        let strategy_hash = stable_json_hash(gene)?;
        let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
        // Regenerate per-bar confidence for risk-based, confidence-scaled
        // sizing. We reuse the precomputed `signals` for the signal vector
        // (identity-preserving) and only take the fresh confidence slice —
        // both are produced from the SAME gene + evaluation config, so they
        // are aligned by construction.
        let (_regen_signals, confidences) =
            signals_and_confidence_for_gene_full(features, ohlcv, gene, &wf_eval_config);
        let metrics = BacktestMetrics::from_metric_array(fast_evaluate_strategy_core(
            &ohlcv.close,
            &ohlcv.high,
            &ohlcv.low,
            signals,
            &confidences,
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

        let gene_wf_passed = walkforward_summary_passed(&walkforward_summary, config.mode);
        walkforward_passed &= gene_wf_passed;
        per_gene_wf.push(gene_wf_passed);
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

    let (cpcv_passed, cpcv_fold_count, cpcv_profitable_fold_ratio, pbo, pbo_passed) =
        evaluate_cpcv_gate(
            portfolio,
            portfolio_signals,
            features,
            ohlcv,
            config,
            effective_smc_gate_threshold,
            &months,
            &days,
            pbo_candidates,
        )?;

    let validation_gates = DiscoveryValidationGates {
        walkforward_passed,
        cpcv_passed,
        canonical_backtest_artifacts: canonical_backtest_artifacts.len(),
        walkforward_validation_artifacts: walkforward_validation_artifacts.len(),
        cpcv_fold_count,
        cpcv_profitable_fold_ratio,
        pbo,
        pbo_passed,
        pbo_candidates: pbo_candidates.len().min(64),
        trials_tested,
        temporal_contract_hash: Some(temporal_contract_hash),
        prop_firm_window_passed: false,
        prop_firm_window_pass_rate: 0.0,
        prop_firm_window_count: 0,
        fallback_mode: false,
        fallback_reason: String::new(),
    };

    Ok((
        validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
        per_gene_wf,
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
    let effective_smc_gate_threshold = config
        .evaluation_config(tail_ohlcv.close.last().copied())
        .smc_gate_threshold;
    compute_discovery_forward_test_artifacts_with_smc_gate(
        portfolio,
        effective_feature_names,
        tail_features,
        tail_ohlcv,
        config,
        effective_smc_gate_threshold,
    )
}

pub fn compute_discovery_forward_test_artifacts_with_smc_gate(
    portfolio: &[Gene],
    effective_feature_names: &[String],
    tail_features: &FeatureFrame,
    tail_ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f32,
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
        let n_rows = tail_features.n_samples();
        let mut projected = ndarray::Array2::<f32>::zeros((n_rows, keep_indices.len()));
        for (new_idx, &orig_idx) in keep_indices.iter().enumerate() {
            projected
                .column_mut(new_idx)
                .assign(&tail_features.feature_column(orig_idx));
        }
        std::borrow::Cow::Owned(FeatureFrame {
            timestamps: tail_features.timestamps.clone(),
            names: effective_feature_names.to_vec(),
            data: neoethos_data::FeatureData::InMemory(projected),
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

    // Each portfolio gene's forward-test replay is fully independent, so run
    // them across the rayon pool instead of one-at-a-time. `par_iter()` on a
    // slice is an INDEXED parallel iterator, so `collect::<Result<Vec<_>>>()`
    // preserves gene order and short-circuits on the first error exactly like
    // the serial `?` did — byte-identical artifacts, but no single-core stall
    // on this silent validation-tail stage.
    //
    // ONE resolver over the tail slice the replay runs on: adaptive genes are
    // forward-tested under the stop regime they were scored under.
    let tail_resolver = GeneEvalSettingsResolver::for_slice(
        config,
        portfolio.iter(),
        &tail_ohlcv.high[..n],
        &tail_ohlcv.low[..n],
        &tail_ohlcv.close[..n],
    )?;
    let artifacts = portfolio
        .par_iter()
        .map(|gene| -> Result<ForwardTestValidationArtifactFile> {
            let settings = tail_resolver.settings_for_gene(gene);
            let strategy_hash = stable_json_hash(gene)?;
            let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
            let evaluation_config = config.evaluation_config_with_smc_gate(
                tail_ohlcv.close.last().copied(),
                effective_smc_gate_threshold,
            );
            let signals =
                signals_for_gene_full(tail_features, tail_ohlcv, gene, &evaluation_config);
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
            Ok(ForwardTestValidationArtifactFile::new(
                ForwardTestValidationScope::new(
                    tail_dataset_hash.clone(),
                    evaluation_config_hash,
                    strategy_hash,
                    &temporal_contract,
                ),
                summary,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
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
    let effective_smc_gate_threshold = config
        .evaluation_config(tail_ohlcv.close.last().copied())
        .smc_gate_threshold;
    compute_discovery_prop_firm_artifacts_with_smc_gate(
        portfolio,
        effective_feature_names,
        tail_features,
        tail_ohlcv,
        config,
        effective_smc_gate_threshold,
        rules,
    )
}

pub fn compute_discovery_prop_firm_artifacts_with_smc_gate(
    portfolio: &[Gene],
    effective_feature_names: &[String],
    tail_features: &FeatureFrame,
    tail_ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f32,
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
        let n_rows = tail_features.n_samples();
        let mut projected = ndarray::Array2::<f32>::zeros((n_rows, keep_indices.len()));
        for (new_idx, &orig_idx) in keep_indices.iter().enumerate() {
            projected
                .column_mut(new_idx)
                .assign(&tail_features.feature_column(orig_idx));
        }
        std::borrow::Cow::Owned(FeatureFrame {
            timestamps: tail_features.timestamps.clone(),
            names: effective_feature_names.to_vec(),
            data: neoethos_data::FeatureData::InMemory(projected),
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

    // Same independence as the forward-test tail above: replay each gene on the
    // held-out window in parallel. Order-preserving indexed collect keeps the
    // artifact order and first-error semantics identical to the serial loop —
    // the prop-firm risk numbers are unchanged, the machine just stops idling
    // on 1 core through this stage.
    //
    // ONE resolver over the tail slice — the prop-firm verdict is measured on
    // the SAME stop regime the gene was scored under, not its unused fixed pips.
    let tail_resolver = GeneEvalSettingsResolver::for_slice(
        config,
        portfolio.iter(),
        &tail_ohlcv.high[..n],
        &tail_ohlcv.low[..n],
        &tail_ohlcv.close[..n],
    )?;
    let artifacts = portfolio
        .par_iter()
        .map(|gene| -> Result<PropFirmRiskValidationArtifactFile> {
            let settings = tail_resolver.settings_for_gene(gene);
            let strategy_hash = stable_json_hash(gene)?;
            let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
            let evaluation_config = config.evaluation_config_with_smc_gate(
                tail_ohlcv.close.last().copied(),
                effective_smc_gate_threshold,
            );
            let signals =
                signals_for_gene_full(tail_features, tail_ohlcv, gene, &evaluation_config);
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
            Ok(PropFirmRiskValidationArtifactFile::new(scope, summary))
        })
        .collect::<Result<Vec<_>>>()?;
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
/// 2026-05-26 in `DiscoveryRuntimeOverrides::default`). Set
/// `models.discovery_runtime.min_history_years` to a positive integer to
/// re-instate a hard floor. There is no env reader for it in this crate as of
/// 2026-08-10.
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
             Discovery; OR (2) relax the floor by setting \
             `models.discovery_runtime.min_history_years` in config — 0 runs on \
             whatever data exists (accepts the over-fitting risk). Operator policy \
             2026-05-24: refuse synthetic / insufficient data."
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

/// Which of the TEN independent base-quality criteria rejected a candidate.
///
/// (Eight when this split was written; the net-expectancy objective added two
/// more `TargetProfile` criteria in the same batch, and this enum follows it
/// rather than duplicating it.)
///
/// MEASUREMENT SLICE (2026-08-09). `rejected_base_quality` used to be one
/// counter standing in for at least eight independent gates
/// (`TargetProfile::accepts` is five, `passes_strict_quality` is four counting
/// the total-loss guard, and the opportunistic lane's enable switch is a ninth
/// way to die that no metric explains). A run could therefore report "174
/// screened, 0 survived" without anyone being able to say WHICH condition did
/// it — and the answer, in that run, was a single one: the payoff floor, which
/// was arithmetically unreachable under the run's own exit geometry.
///
/// Order is the ATTRIBUTION order, not the evaluation order of the original
/// code: a candidate is charged to the FIRST variant it fails, so the counters
/// partition the rejects exactly (they sum to `base_quality`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaseQualityReject {
    /// `max_drawdown_pct >= 1.0` — the account reached zero. Checked first
    /// because past total loss the other numbers describe a state that cannot
    /// exist.
    AccountWiped,
    /// `target_profile.min_net_expectancy_per_trade` — the average trade loses
    /// money after costs. THE primary criterion, and the only unconditional one.
    ProfileNetExpectancy,
    /// `target_profile.min_expectancy_t_stat` — the expectancy is positive but
    /// inside its own sampling noise.
    ProfileExpectancySignificance,
    /// `target_profile.min_win_rate`.
    ProfileWinRate,
    /// `target_profile.min_payoff_ratio` — the gate that decided the 0-of-174
    /// run all by itself.
    ProfilePayoffRatio,
    /// `target_profile.max_in_market`.
    ProfileInMarket,
    /// Would have cleared the OPPORTUNISTIC bar's metric floors, but the lane
    /// is switched off (`opportunistic_enabled` / `use_opportunistic_candidates`).
    /// Killed by a config switch, not by a measurement — which is a completely
    /// different thing to know.
    OpportunisticLaneClosed,
    /// `filtering.min_positive_months` (and the opportunistic lane did not
    /// rescue it).
    PositiveMonths,
    /// `filtering.min_trades_per_month` (ditto).
    TradesPerMonth,
    /// `filtering.min_monthly_return_pct` (ditto).
    MonthlyReturn,
}

impl BaseQualityReject {
    fn label(self) -> &'static str {
        match self {
            Self::AccountWiped => "base_quality.account_wiped",
            Self::ProfileNetExpectancy => "base_quality.profile_net_expectancy",
            Self::ProfileExpectancySignificance => "base_quality.profile_expectancy_significance",
            Self::ProfileWinRate => "base_quality.profile_win_rate",
            Self::ProfilePayoffRatio => "base_quality.profile_payoff_ratio",
            Self::ProfileInMarket => "base_quality.profile_in_market",
            Self::OpportunisticLaneClosed => "base_quality.opportunistic_lane_closed",
            Self::PositiveMonths => "base_quality.positive_months",
            Self::TradesPerMonth => "base_quality.trades_per_month",
            Self::MonthlyReturn => "base_quality.monthly_return",
        }
    }
}

/// Attribute a base-quality rejection to exactly one criterion.
///
/// `None` = the candidate PASSED the base-quality stage; the bool is
/// `opportunistic_quality` (it passed on the opportunistic lane rather than the
/// strict one), preserving the caller's existing lane bookkeeping.
///
/// Pure, so the attribution is testable without a run. It reproduces the
/// original control flow exactly — `profile_ok && (strict || opportunistic)` —
/// and only adds a reason to the `false` branch.
fn classify_base_quality(
    metrics: &StrategyMetrics,
    profile: &TargetProfile,
    cfg: &crate::genetic::FilteringConfig,
) -> Result<bool, BaseQualityReject> {
    // Total loss first: it is a boundary of meaning, not a threshold.
    if !survived_the_backtest(metrics) {
        return Err(BaseQualityReject::AccountWiped);
    }
    // DELEGATED, never re-implemented. A previous revision of this function
    // spelled the profile's criteria out inline; when the net-expectancy gate
    // was added to `TargetProfile::evaluate` the copy here did not learn about
    // it, so the quality screen would have kept admitting money-losers with a
    // high payoff ratio — the exact reward hack the expectancy gate exists to
    // close. One implementation, one place, mapped here to a counter.
    if let Err(rejection) = profile.evaluate(metrics) {
        return Err(match rejection {
            TargetProfileRejection::NegativeNetExpectancy => {
                BaseQualityReject::ProfileNetExpectancy
            }
            TargetProfileRejection::ExpectancyNotSignificant => {
                BaseQualityReject::ProfileExpectancySignificance
            }
            TargetProfileRejection::TooFewWinners => BaseQualityReject::ProfileWinRate,
            TargetProfileRejection::PayoffTooLow => BaseQualityReject::ProfilePayoffRatio,
            TargetProfileRejection::TooMuchTimeInMarket => BaseQualityReject::ProfileInMarket,
        });
    }

    if passes_strict_quality(metrics, cfg) {
        return Ok(false);
    }
    if passes_opportunistic_quality(metrics, cfg) {
        return Ok(true);
    }

    // Strict said no and the opportunistic lane did not rescue it. Ask whether
    // the lane REFUSED it or was simply closed: "N candidates were killed by a
    // switch" and "N candidates missed a metric" call for opposite decisions.
    let lane_closed = !cfg.opportunistic_enabled || !cfg.use_opportunistic_candidates;
    if lane_closed {
        let mut open = *cfg;
        open.opportunistic_enabled = true;
        open.use_opportunistic_candidates = true;
        if passes_opportunistic_quality(metrics, &open) {
            return Err(BaseQualityReject::OpportunisticLaneClosed);
        }
    }

    if cfg.min_positive_months > 0 && metrics.positive_months < cfg.min_positive_months {
        return Err(BaseQualityReject::PositiveMonths);
    }
    if cfg.min_trades_per_month > 0.0 && metrics.trades_per_month < cfg.min_trades_per_month {
        return Err(BaseQualityReject::TradesPerMonth);
    }
    // The strict-only criterion, tested EXPLICITLY rather than assumed.
    //
    // The forward mapping is compiler-guarded (the counter match over
    // `BaseQualityReject` is exhaustive); the inverse mapping was not. An
    // unguarded `Err(MonthlyReturn)` here books whatever reaches this line as a
    // monthly-return failure, so adding a fourth criterion to
    // `passes_strict_quality` would silently misattribute those rejections — and
    // the run-end sum self-check would still balance, because the total and the
    // buckets both increment. In a slice whose whole purpose is attribution
    // integrity, that was the one place attribution could go wrong quietly.
    if cfg.min_monthly_return_pct > 0.0
        && metrics.avg_monthly_return_pct < cfg.min_monthly_return_pct
    {
        return Err(BaseQualityReject::MonthlyReturn);
    }
    tracing::error!(
        target: "neoethos_search::funnel",
        avg_monthly_return_pct = metrics.avg_monthly_return_pct,
        positive_months = metrics.positive_months,
        trades_per_month = metrics.trades_per_month,
        "base-quality attribution FELL THROUGH: strict said no, the opportunistic lane did \
         not rescue it, and none of the three named criteria fired. `passes_strict_quality` \
         has grown a criterion this function does not know about. Counting it as \
         base_quality.monthly_return so the totals still balance, but the attribution for \
         these candidates is WRONG and must not be read."
    );
    Err(BaseQualityReject::MonthlyReturn)
}

/// Which gate inside the quality screen rejected how many candidates.
///
/// The screen is four independent tests chained with `&&`, and it is routinely
/// the funnel's bottleneck, so a single collapsed "rejected 7 792" number is
/// not actionable: widening the Monte-Carlo floor and widening the regime check
/// are different decisions with different risks, and only the split says which
/// one is even relevant.
///
/// 2026-08-09: `base_quality` is now itself split ten ways (see
/// [`BaseQualityReject`]) and the Monte-Carlo/sensitivity EVALUATION errors are
/// no longer conflated — they used to share one counter, so an infrastructure
/// failure in the sensitivity launch was reported as a Monte-Carlo error.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct QualityScreenRejects {
    /// Failed both the strict and the opportunistic metric bars. Equal to the
    /// sum of the ten fields below it, by construction.
    base_quality: usize,
    bq_account_wiped: usize,
    bq_profile_net_expectancy: usize,
    bq_profile_expectancy_significance: usize,
    bq_profile_win_rate: usize,
    bq_profile_payoff_ratio: usize,
    bq_profile_in_market: usize,
    bq_opportunistic_lane_closed: usize,
    bq_positive_months: usize,
    bq_trades_per_month: usize,
    bq_monthly_return: usize,
    /// Lost more than `max_regime_loss_pct` in some market regime.
    regime: usize,
    /// The batched Monte-Carlo evaluation itself failed (a real bug, not a
    /// verdict on the candidate).
    mc_error: usize,
    /// The SENSITIVITY launch failed. Was folded into `mc_error` until
    /// 2026-08-09, which made a broken sensitivity launch look like a
    /// Monte-Carlo problem.
    sensitivity_error: usize,
    /// Fewer than `mc_min_profitable` of `mc_runs` perturbations stayed
    /// profitable.
    mc_floor: usize,
    /// Subset of `mc_floor` that came within 10 runs of the floor.
    mc_near_miss: usize,
    /// Went unprofitable once the stress spread/commission were applied.
    sensitivity: usize,
}

/// Run-level tally of [`CostBandVerdict`] across every screened candidate.
///
/// Read `optimistic_edge_only` before reading the survivor count. Those
/// candidates cleared every configured gate and are still not results: they are
/// profitable only at the cheap end of a cost the operator cannot pin down to a
/// tenth of a pip.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CostBandCensus {
    pub survives: usize,
    pub optimistic_edge_only: usize,
    pub fails: usize,
    pub unmeasured: usize,
    pub not_discriminating: usize,
}

impl CostBandCensus {
    pub fn total(&self) -> usize {
        self.survives
            + self.optimistic_edge_only
            + self.fails
            + self.unmeasured
            + self.not_discriminating
    }
}

/// Can the configured band tell anything apart from the baseline it is measured
/// against?
///
/// The band edges are charged as a TOTAL round-trip cost, replacing spread and
/// commission both. Cost is monotone: a candidate that cleared the baseline at
/// cost `c` clears any cheaper cost by construction. So if the PESSIMISTIC edge
/// is at or below the run's own charged cost, every survivor is guaranteed
/// `SurvivesBand` and the census reads clean on every run — which is worse than
/// no census, because a reader takes it as evidence.
///
/// MEASURED at the shipped configuration (2026-08-09 review): baseline is
/// `backtest_spread 1.5 + slippage 0.5 + commission 14 USD/lot ÷ 10 USD/pip`
/// = 3.4 pips, against band edges 1.6 / 2.4. Both edges are CHEAPER than the run
/// the candidate already survived.
pub fn cost_band_discriminates(band: Option<(f64, f64)>, baseline_cost_pips: f64) -> bool {
    match band {
        Some((_, pessimistic)) => {
            pessimistic.is_finite() && baseline_cost_pips.is_finite() && pessimistic > baseline_cost_pips
        }
        None => false,
    }
}

/// What a candidate did across the round-trip cost band.
///
/// The band exists because a backtest result is a function of the cost you
/// charged it, and nobody knows their all-in cost to a tenth of a pip. A single
/// cost point cannot be checked by a reader; two edges can.
///
/// `OptimisticEdgeOnly` is the finding this type exists to make unmissable:
/// profitable at the cheap end of the band and not at the expensive end. It is
/// NOT a result, and it must not be reported as one.
/// `Serialize`/`Deserialize` added 2026-08-10 (#71) so the verdict can be
/// written into `live_portfolio.json` beside the genes it judges. The wire
/// spelling is exactly [`CostBandVerdict::label`], so a log line and an
/// artifact can be grepped with the same string.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostBandVerdict {
    /// No band configured, or both launches failed. The candidate carries NO
    /// cost-robustness evidence — which is different from carrying good news.
    #[default]
    #[serde(rename = "cost_band_unmeasured")]
    Unmeasured,
    /// The band cannot discriminate: its pessimistic edge is at or below the
    /// cost the run already charged, so passing it is arithmetic, not evidence.
    /// Counted separately and never as good news. See
    /// [`cost_band_discriminates`].
    #[serde(rename = "cost_band_not_discriminating")]
    NotDiscriminating,
    /// Profitable at BOTH edges. The only verdict that supports a claim.
    #[serde(rename = "cost_band_survives")]
    SurvivesBand,
    /// Profitable at the optimistic edge, not at the pessimistic one.
    #[serde(rename = "cost_band_optimistic_edge_only")]
    OptimisticEdgeOnly,
    /// Unprofitable at both edges.
    #[serde(rename = "cost_band_fails")]
    FailsBand,
}

impl CostBandVerdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Unmeasured => "cost_band_unmeasured",
            Self::NotDiscriminating => "cost_band_not_discriminating",
            Self::SurvivesBand => "cost_band_survives",
            Self::OptimisticEdgeOnly => "cost_band_optimistic_edge_only",
            Self::FailsBand => "cost_band_fails",
        }
    }

    /// Classify one candidate from the two edge net-profits. `None` on either
    /// edge means that edge was not measured, and an unmeasured edge cannot be
    /// counted as passed.
    pub fn from_edges(optimistic: Option<f64>, pessimistic: Option<f64>) -> Self {
        match (optimistic, pessimistic) {
            (Some(lo), Some(hi)) => {
                let lo_ok = lo.is_finite() && lo > 0.0;
                let hi_ok = hi.is_finite() && hi > 0.0;
                match (lo_ok, hi_ok) {
                    (true, true) => Self::SurvivesBand,
                    (true, false) => Self::OptimisticEdgeOnly,
                    // Unprofitable cheap but profitable expensive is not a
                    // coherent outcome for a monotone cost; it means the two
                    // launches disagree about something other than cost. Treat
                    // it as a failure rather than inventing a pass.
                    (false, _) => Self::FailsBand,
                }
            }
            _ => Self::Unmeasured,
        }
    }
}

impl QualityScreenRejects {
    fn total(&self) -> usize {
        self.base_quality
            + self.regime
            + self.mc_error
            + self.sensitivity_error
            + self.mc_floor
            + self.sensitivity
    }

    /// The ten base-quality criteria, named. Used for both the run-end log
    /// and the persisted funnel, so the two can never disagree.
    fn base_quality_breakdown(&self) -> [(&'static str, usize); 10] {
        [
            (
                BaseQualityReject::AccountWiped.label(),
                self.bq_account_wiped,
            ),
            (
                BaseQualityReject::ProfileNetExpectancy.label(),
                self.bq_profile_net_expectancy,
            ),
            (
                BaseQualityReject::ProfileExpectancySignificance.label(),
                self.bq_profile_expectancy_significance,
            ),
            (
                BaseQualityReject::ProfileWinRate.label(),
                self.bq_profile_win_rate,
            ),
            (
                BaseQualityReject::ProfilePayoffRatio.label(),
                self.bq_profile_payoff_ratio,
            ),
            (
                BaseQualityReject::ProfileInMarket.label(),
                self.bq_profile_in_market,
            ),
            (
                BaseQualityReject::OpportunisticLaneClosed.label(),
                self.bq_opportunistic_lane_closed,
            ),
            (
                BaseQualityReject::PositiveMonths.label(),
                self.bq_positive_months,
            ),
            (
                BaseQualityReject::TradesPerMonth.label(),
                self.bq_trades_per_month,
            ),
            (
                BaseQualityReject::MonthlyReturn.label(),
                self.bq_monthly_return,
            ),
        ]
    }
}

pub fn run_discovery_cycle(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<DiscoveryResult> {
    run_discovery_cycle_with_progress(features, ohlcv, config, |_| {})
}

/// Fraction of the dataset withheld from discovery as the honest
/// out-of-sample tail. Matches the 80/20 split the desktop app has always
/// applied; audit B02/B03 (2026-07-13) found the CLI and the batch
/// orchestrator called [`run_discovery_cycle`] with the FULL series — the
/// GA and candidate selection saw every bar, so no window was ever truly
/// out-of-sample on those paths.
pub const DEFAULT_OOS_HOLDOUT_FRACTION: f64 = 0.2;

/// [`run_discovery_cycle`] behind the outer OOS holdout split — the single
/// source of truth for "discovery never sees the tail" (audit B02/B03).
///
/// Splits the series once: discovery (GA + candidate selection + all
/// in-sample gates) runs on the FIRST 80% only; the last 20% is withheld
/// and used exclusively to compute forward-test + prop-firm artifacts for
/// the selected portfolio. Every production caller (desktop app, CLI,
/// batch orchestrator) must go through this wrapper; calling
/// [`run_discovery_cycle`] directly is only correct for tests or callers
/// that manage their own holdout.
pub fn run_discovery_cycle_with_holdout(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
) -> Result<DiscoveryResult> {
    run_discovery_cycle_with_holdout_and_progress(features, ohlcv, config, prop_firm_rules, |_| {})
}

/// See [`run_discovery_cycle_with_holdout`]; this variant forwards discovery
/// progress events to `progress_fn` (same contract as
/// [`run_discovery_cycle_with_progress`]).
pub fn run_discovery_cycle_with_holdout_and_progress<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;

    crate::gpu_native::capability::gpu_pipeline_preflight(
        crate::backend::current_evaluation_backend(),
        &crate::gpu_native::capability::GpuCapabilityManifest::stage1_baseline(),
        &crate::gpu_native::capability::PipelineStage::FULL_DISCOVERY,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    // Align both inputs to the same row count before splitting so the
    // in-sample and tail windows stay index-consistent even when the
    // caller's OHLCV and feature cube disagree by a few rows.
    let n_rows = ohlcv.close.len().min(features.n_samples());
    anyhow::ensure!(
        n_rows > 0,
        "run_discovery_cycle_with_holdout: empty dataset (ohlcv rows = {}, feature rows = {})",
        ohlcv.close.len(),
        features.n_samples()
    );
    let is_end = ((n_rows as f64) * (1.0 - DEFAULT_OOS_HOLDOUT_FRACTION)).floor() as usize;
    anyhow::ensure!(
        is_end >= 64,
        "run_discovery_cycle_with_holdout: dataset too short for an OOS holdout split \
         ({n_rows} rows total, {is_end} in-sample). Provide more history — discovery \
         without a held-out tail cannot produce trustworthy out-of-sample evidence."
    );

    let is_ohlcv = slice_ohlcv(ohlcv, 0, is_end);
    // Never-OOM fix (2026-07-18): `row_window` is a zero-copy VIEW for
    // mmap-backed cubes. The old `sample_window` materialized 80% of the
    // disk cube in RAM before the GA even started — on EURUSD M5 that was a
    // surprise ~5.8 GB allocation plus a full read of the 7.3 GB store,
    // which froze the operator's machine mid-discovery. In-memory cubes
    // still copy (they already fit in RAM by construction).
    let is_features = features.row_window(0, is_end);
    tracing::info!(
        target: "neoethos_search::discovery",
        total_rows = n_rows,
        in_sample_rows = is_end,
        holdout_rows = n_rows - is_end,
        holdout_fraction = DEFAULT_OOS_HOLDOUT_FRACTION,
        "outer OOS holdout: discovery sees only the first {is_end} rows; the tail is \
         withheld for forward-test + prop-firm evidence"
    );

    let mut result =
        run_discovery_cycle_with_progress(&is_features, &is_ohlcv, config, &mut progress_fn)?;

    // Operator Stop mid-search: the cycle returned early with a partial
    // result the caller will discard — don't burn time forward-testing it.
    if crate::genetic::search_engine::search_cancel_requested() {
        return Ok(result);
    }
    if result.portfolio.is_empty() || is_end >= n_rows {
        return Ok(result);
    }

    progress_fn(DiscoveryProgress::StageAdvanced {
        stage: "holdout_forward_test",
        detail: format!(
            "replaying {} strategies on the held-out {}-row tail (forward-test + \
             prop-firm evidence) — silent but active",
            result.portfolio.len(),
            n_rows - is_end
        ),
    });

    let tail_ohlcv = slice_ohlcv(ohlcv, is_end, n_rows);
    // Same never-OOM treatment for the held-out tail: a view, not a copy.
    // The forward-test projection then copies only the (small) effective
    // feature columns it actually needs.
    let tail_features = features.row_window(is_end, n_rows);

    match compute_discovery_forward_test_artifacts_with_smc_gate(
        &result.portfolio,
        &result.effective_feature_names,
        &tail_features,
        &tail_ohlcv,
        config,
        result.effective_smc_gate_threshold,
    ) {
        Ok(artifacts) => result.forward_test_validation_artifacts = artifacts,
        Err(err) => tracing::warn!(
            target: "neoethos_search::discovery",
            error = %err,
            "forward-test artifact computation on the held-out tail failed \
             (non-fatal — portfolio export proceeds without forward-test evidence, \
             and the live-portfolio OOS gate will keep all members)"
        ),
    }
    match compute_discovery_prop_firm_artifacts_with_smc_gate(
        &result.portfolio,
        &result.effective_feature_names,
        &tail_features,
        &tail_ohlcv,
        config,
        result.effective_smc_gate_threshold,
        prop_firm_rules,
    ) {
        Ok(artifacts) => result.prop_firm_validation_artifacts = artifacts,
        Err(err) => tracing::warn!(
            target: "neoethos_search::discovery",
            error = %err,
            "prop-firm artifact computation on the held-out tail failed \
             (non-fatal — portfolio export proceeds without prop-firm evidence)"
        ),
    }
    Ok(result)
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
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;

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

    // Never-OOM auto-tune (2026-06-08): probe host RAM + GPU VRAM ONCE and
    // install memory budgets sized to the detected hardware, so peak memory
    // tracks the box and NOT the requested population/generations. Idempotent
    // (OnceLock) and override-respecting (explicit NEOETHOS_BOT_SEARCH_* env
    // wins). The average user gets a hardware-fit config with zero tuning; huge
    // population/gene requests stream through in chunks instead of OOMing.
    // Gated on the `gpu` feature: cubecl_eval (and the budgets it installs) only
    // exists in GPU builds; a CPU-only build has no GPU eval to tune.
    #[cfg(feature = "gpu")]
    crate::cubecl_eval::auto_tune_memory_budgets();

    // Auto-enable the fused VRAM-resident eval (signals stay on the GPU, no host
    // round-trip) IFF it proves byte-identical to the windowed path on THIS
    // machine's card — resolved + logged up-front so the ~sub-second probe runs
    // before the GA loop, not lazily mid-generation. There is NO operator override
    // (`NEOETHOS_GPU_FUSED_EVAL` was deleted 2026-08-10): the decision is
    // auto-detected — OFF when native prototype B owns population eval, OFF on an
    // integrated GPU, otherwise decided by the byte-parity probe. This is the
    // biggest win on dense timeframes (M1/M5), where the signal matrix is largest.
    #[cfg(feature = "gpu")]
    crate::cubecl_eval::ensure_fused_eval_decided();

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
    // F-277 + audit D06: install THIS run's adaptive ladder, or clear back to
    // the static one, so no previous symbol's ladder leaks into this run (the
    // batch orchestrator runs many symbols in one process). Runs are
    // sequential (discovery is single-instance), so a per-run replace is safe.
    if config.adaptive_thresholds {
        if let Some(ladder) =
            crate::genetic::derive_adaptive_threshold_ladder_from_features(&features)
        {
            crate::genetic::install_adaptive_threshold_ladder(ladder);
            tracing::info!(
                target: "neoethos_search::discovery",
                p10 = ladder[0],
                p25 = ladder[1],
                p50 = ladder[2],
                p75 = ladder[3],
                p90 = ladder[4],
                p99 = ladder[5],
                "F-277: installed adaptive threshold ladder from this run's feature cube"
            );
        } else {
            crate::genetic::clear_adaptive_threshold_ladder();
            tracing::warn!(
                target: "neoethos_search::discovery",
                "F-277: adaptive ladder derivation returned None (degenerate \
                 feature cube: empty or zero-variance). Falling back to static ladder."
            );
        }
    } else {
        // Adaptive off for this run — ensure a prior run's ladder is gone.
        crate::genetic::clear_adaptive_threshold_ladder();
    }

    // ── Gene stop/target band scale (2026-08-09) ──────────────────────────
    //
    // The GA drew every stop from `[6, 20]` pips and every target from
    // `[12, 45]` pips — M5 numbers, hardcoded in `evolution_math.rs`. On H1
    // (ATR ≈ 12 pips) the whole stop band sits inside one bar's range; on H4
    // (ATR ≈ 30 pips) it sits below it. Every "search a higher timeframe"
    // suggestion was therefore void: the higher timeframes could not be
    // expressed by any gene the search was able to write.
    //
    // Install THIS dataset's median ATR as the band's unit, or clear back to the
    // absolute band, exactly like the threshold ladder above and for the same
    // audit-D06 reason — the batch orchestrator runs many (symbol, timeframe)
    // combos in one process and a leaked M5 scale is worse than no scale.
    //
    // Stated plainly so nobody sells this as an edge: widening the band has ZERO
    // prior expected value in money. Measured across the exit-geometry sweep,
    // expectancy stayed at -4.15 pips per trade while payoff moved 0.91 → 2.53.
    // This changes which shapes are REACHABLE. The expectancy gate decides which
    // of them survive.
    {
        let bounds_cfg = crate::genetic::current_gene_stop_bounds_overrides();
        let evaluation = config.evaluation_config(ohlcv.close.last().copied());
        let pip = crate::genetic::adaptive_pip_size(evaluation.pip_value, &evaluation.symbol);
        let atr = if bounds_cfg.atr_scaled {
            crate::stop_target::median_atr_pips(&ohlcv.high, &ohlcv.low, &ohlcv.close, pip, 14)
        } else {
            None
        };
        match atr {
            Some(atr_pips) => {
                crate::genetic::install_gene_stop_atr_scale(atr_pips);
                let resolved = crate::genetic::current_gene_stop_bounds();
                tracing::info!(
                    target: "neoethos_search::discovery",
                    timeframe = %config.timeframe_label,
                    atr_pips = atr_pips,
                    sl_min_pips = resolved.sl_min_pips,
                    sl_max_pips = resolved.sl_max_pips,
                    tp_min_pips = resolved.tp_min_pips,
                    tp_max_pips = resolved.tp_max_pips,
                    rr_min = resolved.rr_min,
                    rr_max = resolved.rr_max,
                    "gene stop/target band scaled to this dataset's median ATR"
                );
            }
            None => {
                crate::genetic::clear_gene_stop_atr_scale();
                let resolved = crate::genetic::current_gene_stop_bounds();
                tracing::warn!(
                    target: "neoethos_search::discovery",
                    timeframe = %config.timeframe_label,
                    atr_scaled_requested = bounds_cfg.atr_scaled,
                    sl_min_pips = resolved.sl_min_pips,
                    sl_max_pips = resolved.sl_max_pips,
                    tp_min_pips = resolved.tp_min_pips,
                    tp_max_pips = resolved.tp_max_pips,
                    "gene stop/target band is the ABSOLUTE pip band — no ATR scale for this \
                     dataset (too few bars, or a constant series, or atr_scaled disabled). On \
                     anything above M5 this band is far tighter than one bar's range"
                );
            }
        }
    }

    // ── CONFIG-IDENTITY GATE (2026-08-09) ─────────────────────────────────
    //
    // Refuse to start a run whose configured payoff floor cannot be reached
    // under this run's own resolved trailing settings, stop/target band and
    // charged cost. It must come AFTER the ATR band above (which decides
    // `sl_min`/`tp_max` for this dataset) and BEFORE anything is searched.
    //
    // WHAT THIS REFUSES, stated explicitly: exactly one class of run — the one
    // whose outcome was arithmetically fixed before a bar was read. The
    // 2026-08-09 review established that "174 candidates screened, 0 survived"
    // was such a run: `target_profile.accepts()` gates EVERY survival path in
    // the quality screen, both `min_win_rate` and `max_in_market` defaulted to
    // 0.0, so the profile reduced to `payoff_ratio >= 2.0` — against a realised
    // payoff the exit geometry pinned near 1.0. It permits nothing new.
    //
    // WHAT IT DOES NOT BUY: nothing, in money. It converts an unfalsifiable
    // multi-hour "the market said no" into an immediate, arithmetic
    // "the configuration said no".
    {
        let pip_value_per_lot = config
            .evaluation_config(ohlcv.close.last().copied())
            .pip_value_per_lot;
        let inputs = crate::run_identity::payoff_inputs_for_config(config, pip_value_per_lot);
        match crate::run_identity::assert_payoff_floor_reachable(
            config.target_profile.min_payoff_ratio,
            &inputs,
        ) {
            Ok(ceiling) => {
                tracing::info!(
                    target: "neoethos_search::run_identity",
                    payoff_floor = config.target_profile.min_payoff_ratio,
                    enforced_ceiling = ceiling.enforced_ceiling,
                    arithmetic_ceiling = ceiling.arithmetic_ceiling,
                    initializer_ceiling = ceiling.initializer_ceiling,
                    binding = ceiling.binding.label(),
                    sl_min_pips = inputs.sl_min_pips,
                    tp_max_pips = inputs.tp_max_pips,
                    cost_pips_round_trip = inputs.cost_pips_round_trip,
                    trailing_enabled = inputs.trailing_enabled,
                    required_win_rate = ceiling.required_win_rate_at_floor,
                    breakeven_win_rate = ceiling.breakeven_win_rate_at_ceiling,
                    zero_edge_base_rate = ceiling.zero_edge_base_rate,
                    edge_points_required = ceiling.edge_points_required_to_break_even(),
                    "config-identity gate passed — the configured payoff floor is reachable \
                     under this run's own settings"
                );
                // Stamp the resolved config into the LOG as well as the ledger,
                // so a run with the ledger disabled still names what it
                // searched under. Same function, same hash — a log line and a
                // ledger entry from one run are comparable by `config_hash`.
                let normalize_features =
                    neoethos_data::current_data_runtime_overrides().normalize_features;
                match crate::run_identity::stamp_resolved_config(
                    config,
                    &inputs,
                    ceiling,
                    pip_value_per_lot,
                    normalize_features,
                ) {
                    Ok(stamp) => {
                        let stamp_json = serde_json::to_string(&stamp)
                            .unwrap_or_else(|e| format!("<unserializable: {e}>"));
                        tracing::info!(
                            target: "neoethos_search::run_identity",
                            config_hash = %stamp.config_hash,
                            stamp = %stamp_json,
                            "resolved-config stamp — the decision-critical values this run \
                             resolved. Two runs with the same config_hash are the same \
                             experiment."
                        );
                    }
                    Err(err) => tracing::warn!(
                        target: "neoethos_search::run_identity",
                        error = %err,
                        "could not stamp the resolved config — this run will not be \
                         attributable to a configuration after the fact"
                    ),
                }
            }
            Err(err) => {
                funnel.finalize("preflight_failed_payoff_floor_unreachable");
                return Err(err);
            }
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
    funnel.record_stage("features_built", 0, features.n_features());

    // ── INDICATOR-BUILD CENSUS (2026-08-09) ───────────────────────────────
    //
    // How many of the declared indicator ids actually produced columns in the
    // cube this run is about to search. Logged ONCE, on the PRE-prefilter frame
    // — after the prefilter the number would measure the prefilter, not the
    // build.
    //
    // WHAT THIS HOOK CAN AND CANNOT SEE. `neoethos-data` builds an
    // `IndicatorLedger` inside `core::hpc_ta` that records EVERY discarded
    // column with a typed `DropReason` (kernel panic, unknown indicator, short
    // series, degenerate, duplicate, over budget). That ledger is
    // `log_summary`'d inside the builder and then DROPPED — it is not returned
    // with the `FeatureFrame`, so from here we can only observe presence, not
    // cause. THE HOOK STILL NEEDED: `prepare_multitimeframe_features*` should
    // return the `IndicatorLedger` (or stash it on the `FeatureFrame`) so the
    // discovery run can record per-reason drop counts in its own artifacts
    // instead of leaving them in a log line nobody correlates with a result.
    // That change belongs to whoever owns `neoethos-data`.
    //
    // What IS reachable and is used here:
    //   * `ALL_INDICATORS` — the declared vocabulary;
    //   * longest-id-wins attribution of each column to a declared id, so
    //     `ema_21` is charged to `ema` and never double-charged to a longer id
    //     that also prefixes it;
    //   * `unknown_feature_names` — the registry gate that has existed with
    //     ZERO callers repo-wide. It is now called.
    {
        use neoethos_data::core::all_indicators::ALL_INDICATORS;

        let declared: Vec<&'static str> = ALL_INDICATORS.to_vec();
        let mut producing: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        let mut unattributed = 0usize;
        for name in &features.names {
            // Strip the higher-timeframe prefix so `H4_rsi_21` is charged to
            // `rsi` and counted once, not treated as its own vocabulary.
            // The `len()` guard is not decorative: `timeframe_group("H4")`
            // returns `Some("H4")` for a column with no suffix at all, and an
            // unguarded `&name[tf.len() + 1..]` would panic mid-run on it.
            let bare = match timeframe_group(name) {
                Some(tf) if name.len() > tf.len() => &name[tf.len() + 1..],
                _ => name.as_str(),
            };
            // Longest declared id that is `bare` or a `bare` prefix ending on a
            // `_` boundary. Longest-wins so `rolling_z_score_trend_zscore` is
            // not attributed to a shorter id that happens to prefix it.
            let mut best: Option<&'static str> = None;
            for id in declared.iter().copied() {
                let matches = bare == id
                    || (bare.len() > id.len()
                        && bare.starts_with(id)
                        && bare.as_bytes()[id.len()] == b'_');
                if matches && best.map(|b| id.len() > b.len()).unwrap_or(true) {
                    best = Some(id);
                }
            }
            match best {
                Some(id) => {
                    producing.insert(id);
                }
                // SMC / session / regime / quant columns are not in
                // ALL_INDICATORS and legitimately land here. Counted, never
                // silently ignored, so a sudden jump is visible.
                None => unattributed += 1,
            }
        }
        let non_producing: Vec<&'static str> = declared
            .iter()
            .copied()
            .filter(|id| !producing.contains(id))
            .collect();
        let unregistered =
            neoethos_data::core::feature_registry::unknown_feature_names(&features.names);
        tracing::info!(
            target: "neoethos_search::indicator_census",
            declared_ids = declared.len(),
            producing_ids = producing.len(),
            non_producing_ids = non_producing.len(),
            columns_total = features.names.len(),
            columns_not_attributable_to_a_declared_id = unattributed,
            columns_unregistered = unregistered.len(),
            non_producing_sample = ?non_producing.iter().take(24).collect::<Vec<_>>(),
            unregistered_sample = ?unregistered.iter().take(12).collect::<Vec<_>>(),
            "indicator-build census — declared ids vs ids that produced a column in this run's \
             cube. Per-DROP-REASON attribution needs the hpc_ta IndicatorLedger to be returned \
             to the caller; from here only presence is observable."
        );
    }

    // Feature Pre-filtering (Idea #3)
    // The "indicator pool" (prefilter_top_k) can never meaningfully exceed the
    // number of features that actually exist — the total indicators + SMC +
    // regime columns. A configured pool above that is silently the whole
    // universe; clamp it to the real ceiling HERE (the authoritative point
    // where the true count is known) and log loudly so the operator sees the
    // effective cap instead of a meaningless number. This is what enforces the
    // "pool ≤ total indicators + SMC" rule the UI can only hint at.
    let configured_top_k = config.runtime_overrides.prefilter_top_k;
    let available_features = features.names.len();
    let prefilter_top_k = resolve_prefilter_top_k(
        configured_top_k,
        available_features,
        config.population,
        config.max_indicators,
    );
    let prefilter_insample_frac = config.runtime_overrides.resolved_prefilter_insample_frac();

    let prefilter_min_per_tf = config.runtime_overrides.prefilter_min_per_timeframe;
    let n_features_before_prefilter = features.names.len();
    if prefilter_top_k > 0 && features.names.len() > prefilter_top_k {
        // The prefilter's inputs, assembled HERE so the function has no ambient
        // state. Two of them are new and both are behaviour changes:
        //
        //  * the TARGET is now the triple-barrier / first-passage label the
        //    objective actually scores, not the 1-bar forward return, and
        //  * the ranking is refit INSIDE each CPCV fold's purged train set
        //    instead of once on a leading prefix.
        let evaluation = config.evaluation_config(ohlcv.close.last().copied());
        let pip = if evaluation.pip_value.is_finite() && evaluation.pip_value > 0.0 {
            evaluation.pip_value
        } else {
            // The cost-model guard upstream has already bailed on a non-finite
            // spread, so this is only reachable for an exotic with no pip size.
            // Charging zero cost into the label would make it claim trades pay
            // for free, so refuse the cost term instead and say so.
            tracing::warn!(
                target: "neoethos_search::discovery",
                pip_value = evaluation.pip_value,
                "prefilter label: no usable pip size — the first-passage barriers carry NO \
                 cost, so the label is optimistic about which trades would have paid"
            );
            0.0
        };
        // Round-trip cost in PRICE units: the full spread (slippage already
        // folded in by `from_settings`) plus the round-trip commission
        // converted from account currency into pips via the per-lot pip value.
        let commission_pips = if evaluation.pip_value_per_lot.is_finite()
            && evaluation.pip_value_per_lot > 0.0
            && evaluation.commission_per_trade.is_finite()
        {
            evaluation.commission_per_trade / evaluation.pip_value_per_lot
        } else {
            0.0
        };
        let round_trip_cost_px = (evaluation.spread_pips.max(0.0) + commission_pips.max(0.0)) * pip;
        // The gene stop band this run installed, converted from pips back into
        // ATR multiples so the labeller (which sizes barriers off its own
        // rolling ATR in price units) speaks the same language. Falls back to
        // the pre-2026-08-09 literals, loudly, when no ATR scale is in force —
        // the absolute pip band cannot be turned into an ATR multiple.
        let (label_sl_atr_mult, label_rr) = {
            let bounds = crate::genetic::current_gene_stop_bounds();
            let mid_rr = 0.5 * (bounds.rr_min + bounds.rr_max);
            match bounds.atr_pips {
                Some(atr_pips) if atr_pips.is_finite() && atr_pips > 0.0 => {
                    let mid_sl_pips = 0.5 * (bounds.sl_min_pips + bounds.sl_max_pips);
                    let mult = mid_sl_pips / atr_pips;
                    if mult.is_finite() && mult > 0.0 && mid_rr.is_finite() && mid_rr > 0.0 {
                        (mult, mid_rr)
                    } else {
                        (1.0, 2.0)
                    }
                }
                _ => {
                    tracing::warn!(
                        target: "neoethos_search::prefilter",
                        sl_min_pips = bounds.sl_min_pips,
                        sl_max_pips = bounds.sl_max_pips,
                        rr_min = bounds.rr_min,
                        rr_max = bounds.rr_max,
                        "no ATR scale installed for this dataset, so the prefilter label falls \
                         back to the literal (1.0 ATR, rr 2.0) geometry. That is the bottom \
                         corner of the searchable band and the ranking it produces is not \
                         representative of what the GA will explore."
                    );
                    (1.0, 2.0)
                }
            }
        };
        let spec = PrefilterSpec {
            top_k: prefilter_top_k,
            insample_frac: prefilter_insample_frac,
            min_per_tf: prefilter_min_per_tf,
            max_hold_bars: if evaluation.max_hold_bars > 0 {
                evaluation.max_hold_bars
            } else {
                // A gene with no vertical barrier still cannot be graded over
                // an unbounded horizon by a labeller, so the label uses the
                // configured triple-barrier horizon. Stated, not silent.
                35
            },
            atr_period: 14,
            // THE LABEL'S GEOMETRY, read from the band the GA will actually
            // search rather than pinned to a literal. Corrected 2026-08-09: the
            // hardcoded (1.0 ATR, rr 2.0) was the very bottom corner of a band
            // that change #3 made searchable (sl 1–4 ATR, rr 1.5–4.0), so a
            // feature that predicts first passage for a 3-ATR stop at rr 3.5 was
            // ranked as if it did not.
            //
            // The MIDPOINT of the band, not a sweep across it: scoring at k
            // points multiplies the fold×column correlation work by k, and with
            // two direction labels and up to 8 refit folds that is already the
            // expensive part of the gate. The midpoint is a strictly better
            // single representative than a corner; a full band sweep, keeping
            // the worst point the way the fold rule keeps the worst fold, is the
            // follow-up and it costs k× the correlation pass.
            sl_atr_mult: label_sl_atr_mult,
            rr: label_rr,
            round_trip_cost_px,
            cpcv: config.enable_cpcv.then_some((
                config.cpcv_n_splits,
                config.cpcv_n_test_groups,
                config.cpcv_embargo_pct,
                config.cpcv_purge_pct,
                config.cpcv_max_rows,
            )),
        };
        let (filtered_frame, census) = prefilter_features(&features, &ohlcv, &spec);
        features = filtered_frame;

        let total_labels =
            census.label_up + census.label_down + census.label_vertical + census.label_ambiguous;
        tracing::info!(
            target: "neoethos_search::prefilter",
            columns_considered = census.columns_considered,
            columns_kept = census.columns_kept,
            regime_forced = census.regime_forced,
            columns_with_nonfinite_rows = census.columns_with_nonfinite_rows,
            columns_unrankable = census.columns_unrankable,
            refit_folds_used = census.refit_folds_used,
            refit_folds_available = census.refit_folds_available,
            mean_fold_instability = census.mean_fold_instability,
            label_sl_atr_mult = spec.sl_atr_mult,
            label_rr = spec.rr,
            label_up = census.label_up,
            label_down = census.label_down,
            label_vertical = census.label_vertical,
            label_ambiguous = census.label_ambiguous,
            label_short_win = census.label_short_win,
            label_short_loss = census.label_short_loss,
            label_vertical_short = census.label_vertical_short,
            label_ambiguous_short = census.label_ambiguous_short,
            label_undefined = census.label_undefined,
            label_fell_back_to_forward_return = census.label_fell_back_to_forward_return,
            label_decided_pct = if total_labels > 0 {
                100.0 * (census.label_up + census.label_down) as f64 / total_labels as f64
            } else {
                0.0
            },
            round_trip_cost_px = spec.round_trip_cost_px,
            "prefilter — target is the triple-barrier label in BOTH directions, geometry read \
             from this run's gene stop band, ranking refit inside every fold, correlation \
             two-pass f64. Ranking MOVED relative to any earlier run."
        );
        if census.columns_unrankable > 0 {
            tracing::warn!(
                target: "neoethos_search::prefilter",
                count = census.columns_unrankable,
                sample = ?census.unrankable_sample,
                "columns EXCLUDED as unrankable — fewer than the minimum pairwise-complete \
                 rows, or no variance, in at least one fold. They are named and dropped, not \
                 scored 0.0 and left to win a tie-break, which is what the old f32 code did."
            );
        }
        if census.columns_with_nonfinite_rows > 0 {
            tracing::warn!(
                target: "neoethos_search::prefilter",
                count = census.columns_with_nonfinite_rows,
                "columns carried non-finite rows; the correlation used the pairwise-complete \
                 rows only. Before 2026-08-09 a single NaN scored the whole column exactly 0.0, \
                 which is how every H1/H4/D1 column lost its rank to a stable-sort tie-break."
            );
        }
        if neoethos_data::current_data_runtime_overrides().normalize_features
            && census.columns_with_nonfinite_rows == 0
        {
            tracing::info!(
                target: "neoethos_search::prefilter",
                "normalize_features is ON, so non-finite cells were already turned into 0.0 \
                 upstream and this pass legitimately sees none. The higher-timeframe alignment \
                 gap is therefore NOT visible here — it shows up as a constant leading block \
                 instead. Read the indicator ledger for that evidence, not this counter."
            );
        }
        // Named reject buckets so the persisted funnel answers "which columns
        // left, and why" without needing the run's logs. Zero-count reasons are
        // not recorded — an absent bucket means the cause did not fire.
        for (reason, count) in [
            (
                "prefilter_unrankable_correlation",
                census.columns_unrankable,
            ),
            (
                "prefilter_below_worst_fold_top_k",
                census
                    .columns_considered
                    .saturating_sub(census.columns_kept)
                    .saturating_sub(census.columns_unrankable),
            ),
        ] {
            if count > 0 {
                funnel.add_reject_reason("features_after_prefilter", reason, count);
            }
        }
    }
    funnel.record_stage(
        "features_after_prefilter",
        n_features_before_prefilter,
        features.names.len(),
    );
    // Capture names after prefilter — gene indices refer to this list.
    let effective_feature_names = features.names.clone();

    // Diagnostic (2026-06-08): surface the per-timeframe coverage of the
    // prefiltered cube so a "multi-TF features never reached the GA"
    // regression is visible at a glance instead of hiding behind a flat
    // `cols=N`. base = unprefixed/regime; each higher TF shows its survivor
    // count. A higher TF reading 0 here means the warm-start can't use it.
    {
        let mut by_tf: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for name in &effective_feature_names {
            let key = timeframe_group(name)
                .map(|g| g.to_string())
                .unwrap_or_else(|| "base".to_string());
            *by_tf.entry(key).or_insert(0) += 1;
        }
        tracing::info!(
            target: "neoethos_search::discovery",
            total = effective_feature_names.len(),
            coverage = ?by_tf,
            "prefilter timeframe coverage (base = unprefixed/regime)"
        );
    }

    // Search-memory + weekly-refresh (2026-06-06): BEFORE the GA starts, seed the
    // seen-signature memory with the hashes recorded in the prior run's ledger so
    // the engine SKIPS re-discovering strategies it already found — each weekly
    // run then ADDS new diverse strategies to a growing library. Config-gated:
    // when `discovery_ledger_enabled` is false this whole block is skipped and
    // behaviour is byte-identical to a build without the feature.
    //
    // The GA builds its OWN `SeenSignatureMemory::from_env()` (search_engine.rs,
    // not modified here). We seed into a memory built from the same env/config so
    // — when an on-disk seen-file is configured via
    // `models.seen_signature_runtime.file_path` — the seeded hashes are persisted
    // and the engine's `from_env()` reads them at construction. When no seen-file
    // is configured (the in-memory default) we still run the seed step but log
    // that cross-run dedup needs a seen-file path to reach the engine.
    if config.discovery_ledger_enabled {
        if let Some(prior) = crate::discovery_ledger::load_prior_ledger(
            &config.discovery_ledger_cache_dir,
            &config.evaluation_symbol,
            &config.timeframe_label,
        ) {
            let mut seen = crate::genetic::SeenSignatureMemory::from_env();
            let seen_has_file = seen.file_path.is_some();
            let prior_total = prior.portfolio.len() + prior.archive.len();
            let inserted = crate::discovery_ledger::seed_seen_from_ledger(&prior, &mut seen);
            seen.flush();
            if seen_has_file {
                tracing::info!(
                    target: "neoethos_search::discovery_ledger",
                    symbol = %config.evaluation_symbol,
                    tf = %config.timeframe_label,
                    prior_total,
                    seeded = inserted,
                    "seeded GA seen-set from prior discovery ledger (persisted to seen-file)"
                );
            } else {
                tracing::warn!(
                    target: "neoethos_search::discovery_ledger",
                    symbol = %config.evaluation_symbol,
                    tf = %config.timeframe_label,
                    prior_total,
                    seeded = inserted,
                    "loaded prior discovery ledger but no on-disk seen-file is configured \
                     (models.seen_signature_runtime.file_path) — the seeded hashes will NOT \
                     reach the engine's fresh in-memory seen-set. Set a file_path for true \
                     cross-run dedup."
                );
            }
        }
    }

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
        data: neoethos_data::FeatureData::InMemory(
            features.sample_window(stage1_start, stage1_end),
        ),
    };
    // ── The search-more knob, resolved here and nowhere else ────────────────
    //
    // Population is NOT a batching parameter: a bigger one creates different
    // candidates and selects different survivors. So it may only grow when the
    // operator asked (`population_auto`), it is logged with both the configured
    // and the resolved value, and it never shrinks below what was configured.
    // The 16 384 bound is where the measured throughput curve flattens
    // (843 M cand-bars/s at 16 384 vs 966 M at 131 072) and roughly the card's
    // own fits ceiling at H1 bar counts — beyond it a generation splits into
    // multiple launches for ~14 % more rate while multiplying the downstream
    // validation funnel's work linearly.
    let ga_population = if config.population_auto {
        match crate::eval::gpu_submission_ceiling(
            ohlcv_stage1.close.len(),
            features_stage1.n_features(),
        ) {
            Some(fits) => {
                let resolved = fits.min(16_384).max(config.population);
                tracing::warn!(
                    target: "neoethos_search::funnel",
                    configured = config.population,
                    card_fits = fits,
                    resolved,
                    "population_auto is ON — GA population sized from the card. \
                     This SEARCHES MORE than the configured population: different \
                     candidates, different survivors, different exports."
                );
                resolved
            }
            None => {
                tracing::warn!(
                    target: "neoethos_search::funnel",
                    configured = config.population,
                    "population_auto is ON but no card ceiling is readable \
                     (no CUDA device, kernels disabled, or CPU build) — \
                     keeping the configured population"
                );
                config.population
            }
        }
    } else {
        config.population
    };
    // Everything downstream of this point must see the RESOLVED population, or
    // the artifacts this run writes from `config` (the search ledger at the
    // end of this function records `config.population`) would describe a
    // different search than the one that ran — the exact defect class this
    // campaign exists to remove. `candidate_count` deliberately stays as
    // configured: auto widens the SEARCH, not the validation funnel's budget.
    // (The run profile written by callers still holds the caller's config;
    // when auto engages, the log line above and the ledger are the record.)
    let resolved_config_storage;
    let config: &DiscoveryConfig = if ga_population != config.population {
        resolved_config_storage = DiscoveryConfig {
            population: ga_population,
            ..config.clone()
        };
        &resolved_config_storage
    } else {
        config
    };
    progress_fn(DiscoveryProgress::SearchStarted {
        population: ga_population,
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
        ga_population,
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

    let effective_smc_gate_threshold = search.effective_smc_gate_threshold;
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

    let result = finalize_candidates_with_progress(
        search.genes,
        &features,
        &ohlcv,
        config,
        effective_smc_gate_threshold,
        effective_feature_names,
        &mut funnel,
        progress_fn,
    )?;

    // Search-memory + weekly-refresh (2026-06-06): AFTER finalize, on the
    // SUCCESS path, write this run's ledger (portfolio + top archive genes, each
    // with its canonical gene-signature hash) so the NEXT run can seed from it.
    // Config-gated; non-fatal (a ledger write failure must not fail an otherwise
    // successful discovery). Timestamp uses the same chrono::Utc clock the crate
    // stamps its other artifacts with — passed in so the ledger module stays pure.
    if config.discovery_ledger_enabled {
        let timestamp_ms = Utc::now().timestamp_millis();
        if let Err(err) = crate::discovery_ledger::save_discovery_ledger(
            &config.discovery_ledger_cache_dir,
            &config.evaluation_symbol,
            &config.timeframe_label,
            &result,
            config,
            timestamp_ms,
        ) {
            tracing::warn!(
                target: "neoethos_search::discovery_ledger",
                symbol = %config.evaluation_symbol,
                tf = %config.timeframe_label,
                error = %err,
                "save_discovery_ledger failed (non-fatal — discovery result is unaffected)"
            );
        } else {
            tracing::info!(
                target: "neoethos_search::discovery_ledger",
                symbol = %config.evaluation_symbol,
                tf = %config.timeframe_label,
                portfolio = result.portfolio.len(),
                "wrote discovery ledger for next-run search-memory seeding"
            );
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// The prefilter statistic (rewritten 2026-08-09)
//
// The function that used to live here was the textbook single-pass covariance,
// in f32, with `n` as f32:
//
//     num = n*Sxy - Sx*Sy
//     den = sqrt( (n*Sx2 - Sx*Sx) * (n*Sy2 - Sy*Sy) )
//
// `n*Sx2 - Sx*Sx` subtracts two nearly equal large numbers, and it cancels
// catastrophically exactly when the mean is large relative to the spread — which
// is EVERY level and distance feature in this cube: `ema_*`, `sma_*`, `vwap_*`,
// `session_*_dist`, `smc_fib_*`, `quant_pivot_dist`. Measured at the real row
// count on a price-scale column: |r| = 0.000070 in f32 versus 0.000289 in f64, a
// factor of 0.24 — and the RANK moved, which is the only thing this number is
// used for.
//
// It is replaced by `neoethos_data::core::stats_f64::pearson_pairwise_f32`:
// two-pass, mean-centred, accumulated in f64, and pairwise-complete so a single
// NaN no longer scores an entire column exactly 0.0 (the old `!den.is_finite()`
// guard did that, which is indistinguishable from "genuinely uncorrelated" — and
// every higher-timeframe column carries a NaN prefix by construction).
//
// THIS CHANGES FEATURE RANKING, AND THEREFORE WHAT THE SEARCH EXPLORES. That is
// the point, not a side effect. No artifact produced before this is comparable
// to one produced after.
// ─────────────────────────────────────────────────────────────────────────────

/// How many CPCV folds the prefilter refits inside.
///
/// The gate's own fold count is `C(n_splits, n_test_groups)` — 28 at the shipped
/// 8/2 — and the refit is `folds × columns × train_rows` of two-pass f64 work.
/// Eight evenly-spaced folds is enough to see whether a feature's correlation is
/// stable across the series; the run LOGS how many of how many it used, so the
/// subsample is a stated fact rather than a hidden one.
const PREFILTER_MAX_REFIT_FOLDS: usize = 8;

/// Everything `prefilter_features` needs, gathered at the one call site so the
/// function has no ambient inputs.
#[derive(Debug, Clone)]
pub(crate) struct PrefilterSpec {
    pub top_k: usize,
    /// Fallback in-sample prefix fraction, used only when no CPCV fold
    /// structure is available (CPCV disabled, or too few rows to split).
    pub insample_frac: f64,
    pub min_per_tf: usize,
    /// Vertical barrier, in bars.
    pub max_hold_bars: usize,
    /// ATR lookback used to size the horizontal barriers.
    pub atr_period: usize,
    /// Stop distance = `sl_atr_mult × ATR`.
    pub sl_atr_mult: f64,
    /// Take distance = `rr × stop distance`.
    pub rr: f64,
    /// Round-trip cost in PRICE units, charged into both barriers so the label
    /// says "would this trade have paid" rather than "did price move".
    pub round_trip_cost_px: f64,
    /// CPCV fold geometry to refit inside: `(n_splits, n_test_groups,
    /// embargo_pct, purge_pct, max_rows)`. `None` = fit once on the prefix,
    /// which is the contaminating behaviour this replaces.
    pub cpcv: Option<(usize, usize, f64, f64, usize)>,
}

/// Counted outcomes of one prefilter pass. Nothing on this path is discarded
/// without a name and a number.
#[derive(Debug, Default, Clone)]
pub(crate) struct PrefilterCensus {
    pub columns_considered: usize,
    pub columns_kept: usize,
    /// Columns force-kept because they are `regime_*`.
    pub regime_forced: usize,
    /// Columns whose ranking slice contained non-finite rows. Those rows were
    /// EXCLUDED pairwise rather than zero-filled, and this is how many columns
    /// were affected. Before 2026-08-09 each of these scored exactly 0.0.
    pub columns_with_nonfinite_rows: usize,
    /// Columns excluded because no fold produced a rankable correlation (too
    /// few pairwise-complete rows, or zero variance). NOT scored 0.0 and left
    /// to compete — named and dropped.
    pub columns_unrankable: usize,
    /// A few unrankable column names, for the log line.
    pub unrankable_sample: Vec<String>,
    /// Folds the refit actually used, and how many the gate will use.
    pub refit_folds_used: usize,
    pub refit_folds_available: usize,
    /// Mean across kept columns of `max_fold|r| - min_fold|r|`. A large value
    /// means the ranking a single up-front fit produced was an artifact of
    /// which rows it happened to see.
    pub mean_fold_instability: f64,
    /// Triple-barrier label census, LONG direction.
    pub label_up: usize,
    pub label_down: usize,
    pub label_vertical: usize,
    pub label_ambiguous: usize,
    /// Same, SHORT direction. Kept separate rather than summed: the two labels
    /// are different questions and a reader must be able to see that one of them
    /// decided far more often than the other, which is exactly what a 2:1
    /// asymmetric barrier pair produces.
    pub label_short_win: usize,
    pub label_short_loss: usize,
    pub label_vertical_short: usize,
    pub label_ambiguous_short: usize,
    /// Bars with no usable entry price or ATR. Counted once (both directions
    /// share the guard), so `label_undefined` plus each direction's four buckets
    /// covers every bar exactly once.
    pub label_undefined: usize,
    /// The first-passage label was degenerate (almost nothing reached either
    /// barrier inside the horizon), so the ranking fell back to the 1-bar
    /// forward return. A ranking produced this way is the OLD target and must
    /// not be read as evidence about the objective's label.
    pub label_fell_back_to_forward_return: bool,
}

/// Below this many DECIDED first-passage labels (upper or lower touched), the
/// label carries no information and every column would come back unrankable —
/// which would empty the feature pool. That is a fixture or a mis-specified
/// barrier, not a market fact, so the prefilter says so and falls back rather
/// than silently deleting the cube.
const MIN_DECIDED_FIRST_PASSAGE_LABELS: usize = 100;

/// The two first-passage label series, one per trade direction.
///
/// The GA emits both `+1` and `-1` signals, so a single long-only label ranks
/// features by what predicts a decline and calls it a target. Each column is
/// scored against BOTH and keeps whichever direction it predicts better.
struct FirstPassageLabels {
    long: Vec<f32>,
    short: Vec<f32>,
}

/// Rolling mean true range in f64, for sizing the label's horizontal barriers.
///
/// f64 throughout: the OHLC arrays are f64 and a barrier distance derived in f32
/// on a 1.08-level instrument loses the digits that distinguish a 6-pip stop
/// from a 7-pip one.
fn rolling_atr_f64(ohlcv: &Ohlcv, period: usize) -> Vec<f64> {
    let n = ohlcv.close.len();
    let period = period.max(1);
    let mut tr = vec![0.0f64; n];
    for i in 0..n {
        let hi = ohlcv.high[i];
        let lo = ohlcv.low[i];
        let prev_close = if i > 0 { ohlcv.close[i - 1] } else { ohlcv.close[i] };
        if !hi.is_finite() || !lo.is_finite() || !prev_close.is_finite() {
            tr[i] = f64::NAN;
            continue;
        }
        tr[i] = (hi - lo)
            .max((hi - prev_close).abs())
            .max((lo - prev_close).abs());
    }
    // Simple trailing mean over the finite entries in the window. A window with
    // no finite true range yields NaN, which the labeller treats as "no label"
    // and COUNTS — it does not silently become a zero-width barrier.
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let start = (i + 1).saturating_sub(period);
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for value in tr.iter().take(i + 1).skip(start) {
            if value.is_finite() {
                sum += *value;
                count += 1;
            }
        }
        if count > 0 {
            out[i] = sum / count as f64;
        }
    }
    out
}

/// Triple-barrier / first-passage label — the thing the objective actually
/// scores.
///
/// The prefilter used to rank features by their correlation with the **1-bar
/// forward return**, which is not what any gene is graded on. A gene opens a
/// position with a stop, a target and a maximum hold, and is graded on whether
/// the target was reached before the stop. Those two quantities can rank
/// features in opposite orders: a slow trend feature has near-zero 1-bar
/// correlation by construction (it barely moves between adjacent base bars) and
/// can still be the best predictor of which barrier gets hit first over 35 bars.
/// That mismatch is why the per-timeframe force-keep quota had to be invented in
/// the first place — it was papering over a target that asked the wrong
/// question.
///
/// Returned label at bar `i`:
/// * `+1` upper barrier touched first,
/// * `-1` lower barrier touched first,
/// * `0` neither touched inside the horizon (vertical barrier), OR both touched
///   inside the SAME bar.
///
/// The both-in-one-bar case is genuinely undecidable at bar resolution and is
/// labelled 0 and COUNTED as `ambiguous`. It is not resolved by guessing from
/// the close — that would invent intrabar information and the label would then
/// encode the guess rather than the market.
///
/// The round-trip cost is charged into BOTH barriers, so the label answers
/// "would this trade have paid" and not "did price move".
fn first_passage_labels(
    ohlcv: &Ohlcv,
    spec: &PrefilterSpec,
) -> (FirstPassageLabels, PrefilterCensus) {
    let n = ohlcv.close.len();
    let mut long_labels = vec![f32::NAN; n];
    let mut short_labels = vec![f32::NAN; n];
    let mut census = PrefilterCensus::default();
    if n < 2 {
        census.label_undefined = n;
        return (
            FirstPassageLabels {
                long: long_labels,
                short: short_labels,
            },
            census,
        );
    }
    let atr = rolling_atr_f64(ohlcv, spec.atr_period);
    let hold = spec.max_hold_bars.max(1);
    let sl_mult = if spec.sl_atr_mult.is_finite() && spec.sl_atr_mult > 0.0 {
        spec.sl_atr_mult
    } else {
        1.0
    };
    let rr = if spec.rr.is_finite() && spec.rr > 0.0 {
        spec.rr
    } else {
        2.0
    };
    let cost = if spec.round_trip_cost_px.is_finite() && spec.round_trip_cost_px >= 0.0 {
        spec.round_trip_cost_px
    } else {
        0.0
    };

    for i in 0..n {
        let entry = ohlcv.close[i];
        let a = atr[i];
        if !entry.is_finite() || !a.is_finite() || a <= 0.0 || i + 1 >= n {
            census.label_undefined += 1;
            continue;
        }
        let stop_distance = sl_mult * a;
        let take_distance = rr * stop_distance;
        // NOT symmetric, and the comment that used to sit here said it was.
        // Corrected 2026-08-09 on two counts.
        //
        // 1. TWO LABELS, one per direction. The old single label put the take
        //    profit `rr × stop` above and the stop `stop` below, which is a LONG
        //    trade's geometry. At the configured rr = 2 the loss barrier is half
        //    as far as the win barrier, so on a driftless walk P(-1) is roughly
        //    twice P(+1) and a `-1` means only "price fell one stop" — the SHORT
        //    trade's take profit was never modelled at all, while the GA emits
        //    both +1 and -1 signals. Columns were being ranked by what predicts
        //    a one-ATR decline. Now the short trade gets its own mirrored barrier
        //    pair and each column is scored on whichever direction it predicts.
        //
        // 2. THE COST SIGN on the losing side was inverted. For the net loss to
        //    equal the stop distance the exit must be at `entry - stop + cost`
        //    (the trade gives back the stop AND pays the round trip). The old
        //    `entry - stop - cost` sat further away, making a loss rarer than the
        //    cost model implies. Both of a long's barriers therefore shift UP by
        //    the cost, and both of a short's shift DOWN — that is what "charge
        //    the round trip to the trade" actually looks like.
        let long_tp = entry + take_distance + cost;
        let long_sl = entry - stop_distance + cost;
        let short_tp = entry - take_distance - cost;
        let short_sl = entry + stop_distance - cost;
        let horizon_end = (i + hold).min(n - 1);

        let mut long_label = 0.0f32;
        let mut short_label = 0.0f32;
        let mut long_decided = false;
        let mut short_decided = false;
        for f in (i + 1)..=horizon_end {
            let hi = ohlcv.high[f];
            let lo = ohlcv.low[f];
            let hi_ok = hi.is_finite();
            let lo_ok = lo.is_finite();
            if !long_decided {
                match (hi_ok && hi >= long_tp, lo_ok && lo <= long_sl) {
                    // Both barriers inside one bar is genuinely undecidable at
                    // bar resolution: labelled 0 and COUNTED, never guessed at
                    // from the close.
                    (true, true) => {
                        census.label_ambiguous += 1;
                        long_decided = true;
                    }
                    (true, false) => {
                        long_label = 1.0;
                        census.label_up += 1;
                        long_decided = true;
                    }
                    (false, true) => {
                        long_label = -1.0;
                        census.label_down += 1;
                        long_decided = true;
                    }
                    (false, false) => {}
                }
            }
            if !short_decided {
                match (lo_ok && lo <= short_tp, hi_ok && hi >= short_sl) {
                    (true, true) => {
                        census.label_ambiguous_short += 1;
                        short_decided = true;
                    }
                    (true, false) => {
                        short_label = 1.0;
                        census.label_short_win += 1;
                        short_decided = true;
                    }
                    (false, true) => {
                        short_label = -1.0;
                        census.label_short_loss += 1;
                        short_decided = true;
                    }
                    (false, false) => {}
                }
            }
            if long_decided && short_decided {
                break;
            }
        }
        if !long_decided {
            census.label_vertical += 1;
        }
        if !short_decided {
            census.label_vertical_short += 1;
        }
        long_labels[i] = long_label;
        short_labels[i] = short_label;
    }
    (
        FirstPassageLabels {
            long: long_labels,
            short: short_labels,
        },
        census,
    )
}

/// The row index sets the prefilter refits inside.
///
/// With CPCV configured this is the gate's OWN fold train sets (purged and
/// embargoed), subsampled to [`PREFILTER_MAX_REFIT_FOLDS`]. Without it, one set:
/// the leading `insample_frac` prefix — the old behaviour, kept only so a
/// CPCV-disabled fixture still runs, and it is the contaminating one.
fn prefilter_fit_windows(n_rows: usize, spec: &PrefilterSpec) -> (Vec<Vec<usize>>, usize) {
    if let Some((n_splits, n_test_groups, embargo_pct, purge_pct, max_rows)) = spec.cpcv {
        let capped = if max_rows > 0 {
            max_rows.min(n_rows)
        } else {
            n_rows
        };
        let offset = n_rows.saturating_sub(capped);
        let cv = CombinatorialPurgedCV::new(n_splits, n_test_groups, embargo_pct, purge_pct);
        let splits = cv.split(capped);
        let available = splits.len();
        if available > 0 {
            // Evenly spaced subsample so the kept folds span the whole
            // combination space rather than clustering at one end.
            let step = available.div_ceil(PREFILTER_MAX_REFIT_FOLDS).max(1);
            let windows: Vec<Vec<usize>> = splits
                .into_iter()
                .step_by(step)
                .take(PREFILTER_MAX_REFIT_FOLDS)
                .map(|(train, _test)| train.into_iter().map(|i| i + offset).collect())
                .filter(|w: &Vec<usize>| !w.is_empty())
                .collect();
            if !windows.is_empty() {
                return (windows, available);
            }
        }
    }
    let train_end = ((n_rows as f64) * spec.insample_frac).floor() as usize;
    let train_end = train_end.clamp(2, n_rows.saturating_sub(1)).max(2);
    (vec![(0..train_end.saturating_sub(1)).collect()], 0)
}

fn prefilter_features(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    spec: &PrefilterSpec,
) -> (FeatureFrame, PrefilterCensus) {
    let n_rows = features.n_samples();
    let n_cols = features.n_features();
    if n_rows < 2 || n_cols <= spec.top_k {
        let census = PrefilterCensus {
            columns_considered: n_cols,
            columns_kept: n_cols,
            ..PrefilterCensus::default()
        };
        return (features.clone(), census);
    }

    // THE TARGET (2026-08-09). Was the 1-bar forward return; is now the
    // triple-barrier label the objective scores. See `first_passage_labels`.
    let (label_set, mut census) = first_passage_labels(ohlcv, spec);
    let mut labels = label_set.long;
    let mut short_labels = Some(label_set.short);
    census.columns_considered = n_cols;

    // Degenerate-label guard. If nearly nothing reached a barrier the label is
    // constant, every correlation comes back `degenerate`, every column is
    // unrankable, and the feature pool would empty out — an outcome produced by
    // the labeller, not by the market. Fall back to the old 1-bar forward
    // return, and make the fallback impossible to miss: a ranking produced this
    // way is NOT evidence about the objective's label.
    // Both directions must be degenerate before falling back — one side can be
    // starved by an asymmetric barrier pair while the other decides plenty.
    let decided_long = census.label_up + census.label_down;
    let decided_short = census.label_short_win + census.label_short_loss;
    if decided_long.max(decided_short) < MIN_DECIDED_FIRST_PASSAGE_LABELS {
        tracing::error!(
            target: "neoethos_search::prefilter",
            label_up = census.label_up,
            label_down = census.label_down,
            label_vertical = census.label_vertical,
            label_ambiguous = census.label_ambiguous,
            minimum = MIN_DECIDED_FIRST_PASSAGE_LABELS,
            atr_period = spec.atr_period,
            sl_atr_mult = spec.sl_atr_mult,
            rr = spec.rr,
            max_hold_bars = spec.max_hold_bars,
            "first-passage label is degenerate — almost no bar reached either barrier inside \
             the horizon. Ranking falls back to the 1-bar FORWARD RETURN (the pre-2026-08-09 \
             target). Check the barrier geometry against this timeframe's ATR before reading \
             anything into this run's feature selection."
        );
        census.label_fell_back_to_forward_return = true;
        let n = ohlcv.close.len();
        let mut returns = vec![f32::NAN; n];
        for i in 0..n.saturating_sub(1) {
            let denom = ohlcv.close[i];
            if denom.abs() > 1e-12 {
                returns[i] = ((ohlcv.close[i + 1] - denom) / denom) as f32;
            }
        }
        labels = returns;
        // The forward return has no direction pair; scoring against a stale
        // short label would silently mix two targets.
        short_labels = None;
    }
    let short_labels = short_labels;

    // THE FIT WINDOWS (2026-08-09). Was one leading prefix, computed ONCE
    // up front — which means every CPCV "out of sample" number in every prior
    // run was contaminated: the features the folds were scored on had been
    // chosen with the folds' own test bars visible. Now the ranking is refit
    // inside each fold's purged, embargoed TRAIN set and a column is scored by
    // its WORST fold.
    //
    // Worst-fold, not mean: the question a prefilter should answer is "would
    // this column have been selected whatever slice of history we looked at",
    // and a mean lets one spectacular fold carry a column that six folds
    // reject. This is deliberately conservative and it is a behaviour change.
    //
    // WHAT REMAINS, stated here and not only in a report, because the comment is
    // what the next reader finds: this does NOT make CPCV clean. ONE global
    // top-K is selected from the worst-across-folds scores, and every fold's
    // test group is another fold's train rows, so the union of the fit windows
    // covers essentially the whole series and the selected feature set is still
    // a function of nearly every row. What went away is the WORST form — a
    // single fit on a leading prefix that overlaps every fold's test group.
    // `mean_fold_instability` is the residual measure. Removing the rest means
    // re-running the GA per fold: 28× the cost and a pipeline restructure.
    //
    // Compounding it: with `normalize_features: true` the robust z-score fits
    // its median/MAD on the leading `NORM_FIT_FRACTION` of rows
    // (neoethos-data `normalization.rs`), which overlaps most CPCV test groups,
    // so the values handed to the fold-wise correlation are already scaled using
    // statistics fitted on those test rows.
    //
    // The purge does hold: 2% of at most 200k rows is up to 4000 bars against a
    // 35-bar label horizon, so the label's own forward window cannot leak across
    // a fold boundary.
    let (windows, folds_available) = prefilter_fit_windows(n_rows, spec);
    census.refit_folds_used = windows.len();
    census.refit_folds_available = folds_available;

    struct ColumnScore {
        idx: usize,
        score: f64,
        instability: f64,
        had_nonfinite: bool,
        rankable: bool,
    }

    let scored: Vec<ColumnScore> = (0..n_cols)
        .into_par_iter()
        .map(|col_idx| {
            let name = &features.names[col_idx];
            if is_prefilter_state_column(name) {
                // Force-keep. `regime_` was always here; `smc_`, `session_` and
                // `fp_` joined it 2026-08-10 — see PREFILTER_STATE_FAMILIES for
                // the argument and for why repairing the correlation function
                // made it urgent. These are the GA's context and event
                // channels; they are not selected on a univariate correlation
                // with a directional label, because they do not have one.
                return ColumnScore {
                    idx: col_idx,
                    score: f64::INFINITY,
                    instability: 0.0,
                    had_nonfinite: false,
                    rankable: true,
                };
            }
            let col: Vec<f32> = features.feature_column(col_idx).iter().copied().collect();
            let mut worst = f64::INFINITY;
            let mut best = 0.0f64;
            let mut had_nonfinite = false;
            let mut rankable_in_all = true;
            for window in &windows {
                let mut xs: Vec<f32> = Vec::with_capacity(window.len());
                let mut ys: Vec<f32> = Vec::with_capacity(window.len());
                let mut ys_short: Vec<f32> = Vec::with_capacity(window.len());
                for &row in window {
                    // The label series is bar-indexed and the feature cube is
                    // row-indexed; they are the same length in production, but a
                    // caller that hands over mismatched lengths must lose the
                    // extra rows rather than index out of bounds. The two lives
                    // in the same guard so neither can be forgotten.
                    if row >= n_rows || row >= labels.len() {
                        continue;
                    }
                    xs.push(col[row]);
                    ys.push(labels[row]);
                    if let Some(short) = short_labels.as_ref() {
                        ys_short.push(short.get(row).copied().unwrap_or(f32::NAN));
                    }
                }
                let outcome = neoethos_data::core::stats_f64::pearson_pairwise_f32(&xs, &ys);
                if outcome.skipped > 0 {
                    had_nonfinite = true;
                }
                // A column is scored on the direction it predicts BETTER. The
                // GA trades both ways, so a feature that only calls declines is
                // as useful as one that only calls advances — and ranking on the
                // long label alone silently preferred the latter.
                let mut a = if outcome.is_rankable() {
                    Some(outcome.abs())
                } else {
                    None
                };
                if short_labels.is_some() {
                    let short_outcome =
                        neoethos_data::core::stats_f64::pearson_pairwise_f32(&xs, &ys_short);
                    if short_outcome.skipped > 0 {
                        had_nonfinite = true;
                    }
                    if short_outcome.is_rankable() {
                        let s = short_outcome.abs();
                        a = Some(a.map_or(s, |l: f64| l.max(s)));
                    }
                }
                // Unrankable in BOTH directions is what excludes a column — one
                // direction being degenerate is not enough to drop it.
                let Some(a) = a else {
                    rankable_in_all = false;
                    break;
                };
                worst = worst.min(a);
                best = best.max(a);
            }
            if !rankable_in_all || !worst.is_finite() {
                return ColumnScore {
                    idx: col_idx,
                    score: f64::NEG_INFINITY,
                    instability: 0.0,
                    had_nonfinite,
                    rankable: false,
                };
            }
            ColumnScore {
                idx: col_idx,
                score: worst,
                instability: best - worst,
                had_nonfinite,
                rankable: true,
            }
        })
        .collect();

    let mut correlations: Vec<(usize, f64)> = Vec::with_capacity(n_cols);
    let mut instability_sum = 0.0f64;
    let mut instability_count = 0usize;
    for entry in &scored {
        if entry.had_nonfinite {
            census.columns_with_nonfinite_rows += 1;
        }
        if !entry.rankable {
            census.columns_unrankable += 1;
            if census.unrankable_sample.len() < 12 {
                census
                    .unrankable_sample
                    .push(features.names[entry.idx].clone());
            }
            // NOT pushed into `correlations`: an unrankable column is excluded
            // by name, never scored 0.0 and left to lose a tie-break.
            continue;
        }
        if entry.score.is_finite() {
            instability_sum += entry.instability;
            instability_count += 1;
        }
        correlations.push((entry.idx, entry.score));
    }
    census.mean_fold_instability = if instability_count > 0 {
        instability_sum / instability_count as f64
    } else {
        0.0
    };
    // Named `regime_forced` for artifact compatibility; it now counts every
    // force-kept STATE column (regime_ + smc_ + session_ + fp_ on the base
    // timeframe). The per-family split is in the log line below so a reader can
    // see which families the number is made of.
    census.regime_forced = features
        .names
        .iter()
        .filter(|n| is_prefilter_state_column(n))
        .count();
    {
        let mut per_family: Vec<(&str, usize)> = Vec::new();
        for family in PREFILTER_STATE_FAMILIES {
            per_family.push((
                family,
                features
                    .names
                    .iter()
                    .filter(|n| n.starts_with(family))
                    .count(),
            ));
        }
        tracing::info!(
            target: "neoethos_search::prefilter",
            state_forced_total = census.regime_forced,
            per_family = ?per_family,
            "state-family columns force-kept ADDITIVELY (they do not consume the operator's \
             top_k budget). BEHAVIOUR CHANGE 2026-08-10: smc_/session_/fp_ joined regime_ here, \
             so correlation ranking may no longer evict them and every SMC gate binds to a real \
             smc_ column rather than to whatever substring survived."
        );
    }

    correlations.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Keep top_k + the regime columns (which occupy the INFINITY slots at the
    // head of the sort, so they do not consume the operator's budget).
    let actual_top_k = (spec.top_k + census.regime_forced).min(n_cols);

    let mut keep_indices: Vec<usize> = correlations
        .iter()
        .take(actual_top_k)
        .map(|(idx, _)| *idx)
        .collect();

    // Per-higher-timeframe quota (2026-06-08), retained. Its original
    // justification — "a higher-TF indicator's 1-bar-forward correlation is ~0
    // by construction" — is weaker now that the target is a 35-bar first-passage
    // label rather than a 1-bar return, so this quota should ADD less than it
    // used to. It is kept because the quota also guarantees the multi-TF seed
    // templates resolve, and because removing two things at once makes neither
    // measurable. If the per-TF coverage log shows the quota is no longer
    // binding, that is the evidence to drop it.
    {
        let mut kept: std::collections::HashSet<usize> = keep_indices.iter().copied().collect();
        if spec.min_per_tf > 0 {
            let mut per_group: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for &idx in &keep_indices {
                if let Some(group) = timeframe_group(&features.names[idx]) {
                    *per_group.entry(group).or_insert(0) += 1;
                }
            }
            for &(idx, _) in &correlations {
                let Some(group) = timeframe_group(&features.names[idx]) else {
                    continue;
                };
                let count = per_group.entry(group).or_insert(0);
                if *count >= spec.min_per_tf {
                    continue;
                }
                if kept.insert(idx) {
                    *count += 1;
                }
            }
        }
        // Force-keep the EXACT features the multi-TF seed templates reference,
        // resolved by the templates' own role logic against the full
        // pre-prefilter names — single source of truth, no duplicated family
        // list.
        //
        // MOVED OUT OF THE `min_per_tf > 0` BLOCK (2026-08-10). It used to sit
        // inside it, so setting `prefilter_min_per_timeframe: 0` — a knob about
        // per-timeframe quotas — ALSO disabled the warm-start force-keep, and
        // the GA's seed templates then referenced columns the prefilter had
        // dropped. Two unrelated decisions on one flag, with nothing saying so.
        // BEHAVIOUR CHANGE: at `min_per_tf = 0` the template columns are now
        // kept; at any positive value nothing changes.
        for idx in crate::genetic::seed_templates::template_feature_indices(&features.names) {
            kept.insert(idx);
        }
        keep_indices = kept.into_iter().collect();
    }

    keep_indices.sort(); // Maintain original order
    keep_indices.dedup();
    let n_keep = keep_indices.len();
    census.columns_kept = n_keep;

    let mut new_names = Vec::with_capacity(n_keep);
    let mut new_data = ndarray::Array2::zeros((n_rows, n_keep));

    for (new_col_idx, &orig_col_idx) in keep_indices.iter().enumerate() {
        new_names.push(features.names[orig_col_idx].clone());
        new_data
            .column_mut(new_col_idx)
            .assign(&features.feature_column(orig_col_idx));
    }

    (
        FeatureFrame {
            timestamps: features.timestamps.clone(),
            names: new_names,
            data: neoethos_data::FeatureData::InMemory(new_data),
        },
        census,
    )
}

/// Genes that must be expected to touch a given column before that column
/// earns a place in the GA's alphabet.
///
/// CALIBRATED, NOT CHOSEN. At the historical operating point — 265 columns kept
/// (`docs/measurements/3090-47260276/card-run-valid.log`, 651 in / 265 out),
/// population 4,096, `max_indicators` 5 so `E[indices per gene] = 3` — the
/// expected number of genes touching any given column is
/// `4096 * 3 / 265 = 46.4`. That is the coverage the search has actually been
/// operating at, and it is the quantity to hold fixed.
pub const PREFILTER_COVERAGE_GENES_PER_COLUMN: f64 = 46.0;

/// How many features the prefilter keeps.
///
/// ## The defect this closes
///
/// `prefilter_top_k` was a CONSTANT 240 applied to the whole assembled
/// multi-timeframe cube. It was set when the cube was 217 columns per timeframe.
/// The cube is no longer that: per-TF width is now bounded by
/// `VocabularyBudget`, i.e. by FREE RAM and the frame length. So the FRACTION of
/// the vocabulary the GA can see became a function of the hardware —
/// 240/1,736 = 13.8% at the old vocabulary, 240/4,920 ≈ 4.9% at what this box
/// affords on the real M5 frame, 240/32,768 = 0.7% on a box that reaches the
/// 4,096-column ceiling. That is the same defect class as sizing memory from a
/// user parameter, one level up.
///
/// ## Why the obvious answers are all wrong, with the numbers
///
/// * **A fraction of the cube width.** 40% of the cube at the hard ceiling is
///   5,056 columns, which at population 4,096 is `4096*3/5056 = 2.4` expected
///   genes per column. The search gets WORSE on the bigger box.
/// * **Derive it from free RAM.** `top_k` is not a memory quantity. 1,000 kept
///   columns cost 4.2 GB at the M5 store's 1,054,320 rows, so a 512 GB box
///   would keep the entire cube — affordable, and therefore fatal. This is the
///   one place where the never-OOM idiom is the wrong answer.
/// * **Drop the cap and let the early-reject predicate do the work.** At the
///   full 12,639-column cube the expected genes per column is 0.97: the median
///   column is never sampled by the initial population at all. The predicate
///   would then abandon batches whose useful column the GA never looked at —
///   a FALSE REJECT, which the predicate is explicitly forbidden to be capable
///   of. `top_k` bounds the GA's index space; the predicate bounds wasted
///   downstream stages. They are orthogonal and neither replaces the other.
///
/// ## What it is instead
///
/// Derived from GA CAPACITY — the alphabet the population can actually cover —
/// and floored by the operator's configured value:
///
/// ```text
/// E[indices per gene] = (1 + max_indicators) / 2      # new_random_gene samples 1..=max
/// derived  = population * E / PREFILTER_COVERAGE_GENES_PER_COLUMN
/// top_k    = clamp(derived, configured, cube_width)
/// ```
///
/// At the shipped GPU population (4,096) that is `4096*3/46 = 267`. At the
/// shipped CPU population it is far below 240 and the operator's configured
/// value wins. So the number does NOT grow when the box grows or when the
/// timeframe list grows — because the alphabet the GA can cover does not grow
/// either.
///
/// `configured == 0` still means "no prefilter", unchanged.
///
/// Conservative in the safe direction with `population_auto`: that flag lets
/// `run_search` raise the population toward the card's ceiling, and this reads
/// the CONFIGURED population, so the derived `top_k` is a lower bound —
/// i.e. more coverage per column than the calibration point, never less.
pub fn resolve_prefilter_top_k(
    configured: usize,
    cube_width: usize,
    population: usize,
    max_indicators: usize,
) -> usize {
    if configured == 0 {
        return 0;
    }
    let expected_indices = (1.0 + max_indicators.max(1) as f64) / 2.0;
    let derived = (population as f64 * expected_indices / PREFILTER_COVERAGE_GENES_PER_COLUMN)
        .round()
        .max(0.0) as usize;
    let effective = derived.max(configured).min(cube_width.max(1));
    tracing::info!(
        target: "neoethos_search::prefilter",
        configured,
        derived,
        effective,
        cube_width,
        population,
        max_indicators,
        expected_indices_per_gene = expected_indices,
        coverage_genes_per_column = PREFILTER_COVERAGE_GENES_PER_COLUMN,
        expected_genes_per_kept_column = if effective > 0 {
            population as f64 * expected_indices / effective as f64
        } else {
            0.0
        },
        "indicator pool sized from GA CAPACITY (population x expected indices per gene / \
         coverage), floored by the configured value and capped by the cube width — never a \
         fraction of the cube and never derived from free RAM. See resolve_prefilter_top_k."
    );
    effective
}

/// Feature families whose ranking criterion the prefilter cannot evaluate.
///
/// The prefilter ranks on univariate correlation with a first-passage label,
/// and the code has always conceded that criterion is wrong for state-like
/// columns — `regime_` was exempted with `f64::INFINITY` for exactly that
/// reason. THE EXEMPTION WAS GRANTED TO ONE FAMILY AND THE ARGUMENT COVERS
/// FOUR. An order block, a session marker and a footprint imbalance are states
/// and events, not directional predictors; they matter in combination, which is
/// what the GA evaluates and what a univariate rank cannot see.
///
/// This became urgent rather than tidy when `pearson_correlation` was repaired.
/// Under the broken function every column scored exactly 0.0, ties broke by
/// column index, and `smc_` columns occupy indices 0-45 — so they always swept
/// the top-K. The repair removed the tie-break that was silently guaranteeing
/// their survival. Nothing else changed; the exposure is new.
///
/// BASE TIMEFRAME ONLY, by construction: higher-TF columns carry a `H1_`/`H4_`
/// prefix (see `timeframe_group`), so `starts_with` matches the base block and
/// not its ten resamplings. Same as `regime_` has always behaved.
const PREFILTER_STATE_FAMILIES: [&str; 4] = ["regime_", "smc_", "session_", "fp_"];

/// Whether this column is force-kept by the prefilter regardless of its
/// correlation rank.
fn is_prefilter_state_column(name: &str) -> bool {
    PREFILTER_STATE_FAMILIES
        .iter()
        .any(|family| name.starts_with(family))
}

/// Identify the higher-timeframe prefix group of a multi-TF feature name.
///
/// Multi-resolution features are emitted as `"{TF}_{indicator}"` (e.g.
/// `"H1_rsi_14"`, `"M15_ema_20"`) by
/// `prepare_multitimeframe_features_with_options`. Base-TF features are
/// unprefixed and `regime_*` columns are handled separately, so this returns
/// `None` for both. A timeframe label is one or two leading uppercase letters
/// (`M`, `H`, `D`, `W`, or `MN`) followed by digits, terminated by `_` — which
/// distinguishes it from lowercase/longer base indicator heads (`rsi`, `macd`,
/// `ema`, `bb`) without needing the live higher-TF list.
fn timeframe_group(name: &str) -> Option<&str> {
    let head = name.split('_').next()?;
    // Canonical TF labels are 2-3 chars: M1/H1/D1/W1 (2) or M15/MN1 (3).
    if head.len() < 2 || head.len() > 3 {
        return None;
    }
    let digits = if let Some(rest) = head.strip_prefix("MN") {
        rest
    } else if head.starts_with(['M', 'H', 'D', 'W']) {
        &head[1..]
    } else {
        return None;
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(head)
}

fn validate_regime_robustness(
    trades: &[crate::quality::Trade],
    features: &FeatureFrame,
    initial_balance: f64,
    max_regime_loss_pct: f64,
) -> bool {
    let _scope = crate::eval_telemetry::CallerScope::enter("regime_robustness");
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
        if idx >= features.n_samples() {
            continue;
        }

        let trend_str = features.feature_at(idx, t_idx);
        let vol_state = features.feature_at(idx, v_idx);

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
/// `"strict"` / `"legacy"` → `Some(Strict)`; anything else (including the
/// shipped `"prop_firm"`) → `None`, meaning "this key decided nothing" — the
/// caller then resolves from `system.trading_mode` and says so.
/// Config-driven replacement for the env-only
/// `resolve_discovery_mode` that read `NEOETHOS_BOT_DISCOVERY_MODE` and the
/// legacy `NEOETHOS_BOT_DISCOVERY_PERMISSIVE` back-compat toggle. The
/// permissive-toggle path is retired with the env var — operators select the
/// regime through `config.yaml` / the UI now.
///
/// RESTRICTED, NOT MERGED (2026-08-10). `models.discovery_mode` accepts exactly
/// two values — `strict` and `legacy`, both meaning [`DiscoveryMode::Strict`].
/// It is NOT a duplicate of `system.trading_mode` and must not be merged into
/// it: it reaches `Strict`, which `trading_mode` structurally cannot express.
/// Any other value is a NO-OP that falls through to `system.trading_mode`, and
/// that fall-through is now named in the log instead of happening in silence.
/// The values a UI or TUI may offer for this key are therefore `strict` and
/// `legacy` only; the regime (risky vs prop-firm) is chosen with
/// `system.trading_mode`.
fn discovery_mode_from_config(value: &str) -> Option<DiscoveryMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "strict" | "legacy" => Some(DiscoveryMode::Strict),
        _ => None,
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
    if let Some(forced) = discovery_mode_from_config(discovery_mode) {
        tracing::info!(
            target: "neoethos_search::config_resolution",
            winner = "models.discovery_mode",
            models_discovery_mode = %discovery_mode,
            system_trading_mode = %trading_mode,
            resolved_mode = ?forced,
            "discovery regime forced to Strict by models.discovery_mode — \
             system.trading_mode is not consulted"
        );
        return forced;
    }
    let resolved = match trading_mode.trim().to_ascii_lowercase().as_str() {
        "risky" | "growth" => DiscoveryMode::Risky,
        _ => DiscoveryMode::PropFirm,
    };
    // The fall-through, said out loud. `models.discovery_mode: risky` and
    // `: prop_firm` are both NO-OPS here — the engine maps neither — and the
    // CLI TUI has been offering exactly those two while rejecting `legacy`, the
    // one value the engine honours. An operator who set `discovery_mode` and
    // watched nothing change was reading a knob that does nothing at that value.
    let value = discovery_mode.trim();
    if !value.is_empty() {
        tracing::warn!(
            target: "neoethos_search::config_resolution",
            key = "models.discovery_mode",
            configured = %value,
            winner = "system.trading_mode",
            system_trading_mode = %trading_mode,
            resolved_mode = ?resolved,
            "models.discovery_mode IS NOT SET TO A RECOGNISED VALUE and decided nothing. \
             It accepts only 'strict' or 'legacy' (both = Strict). The regime was decided \
             by system.trading_mode. To pick risky vs prop-firm, set system.trading_mode."
        );
    } else {
        tracing::info!(
            target: "neoethos_search::config_resolution",
            winner = "system.trading_mode",
            system_trading_mode = %trading_mode,
            resolved_mode = ?resolved,
            "discovery regime resolved from system.trading_mode"
        );
    }
    resolved
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

/// A sampled prop-firm challenge window: `[start_idx, end_idx)` plus the
/// window-local adaptive base series (`None` ⇒ fixed pips on this window).
type PropFirmWindow = (usize, usize, Option<std::sync::Arc<[f64]>>);

/// Plan the gate's evenly-spaced windows ONCE — the geometry and the
/// window-local adaptive bases are gene-INDEPENDENT, so they are computed here
/// and shared across every candidate instead of once per candidate. The base
/// is computed on exactly the window slice being simulated (index alignment +
/// the same convention as `validation_genes_population_window`).
fn plan_prop_firm_windows(
    ohlcv: &Ohlcv,
    timestamps: &[i64],
    overrides: &PropFirmGateOverrides,
    resolver: &GeneEvalSettingsResolver<'_>,
    any_adaptive: bool,
) -> Result<Vec<PropFirmWindow>> {
    let n = timestamps
        .len()
        .min(ohlcv.close.len())
        .min(ohlcv.high.len())
        .min(ohlcv.low.len());
    if n == 0 || overrides.window_days == 0 || overrides.n_windows == 0 {
        return Ok(Vec::new());
    }
    let window_ms: i64 = (overrides.window_days as i64) * 86_400_000;
    let first_ts = timestamps[0];
    let last_ts = timestamps[n - 1];
    if last_ts - first_ts < window_ms {
        return Ok(Vec::new());
    }
    let max_start_ts = last_ts - window_ms;
    let span = (max_start_ts - first_ts).max(1) as f64;
    let n_windows = overrides.n_windows.max(1);
    let stride = if n_windows == 1 {
        0.0
    } else {
        span / (n_windows as f64 - 1.0)
    };
    let mut windows = Vec::with_capacity(n_windows);
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
        let base = if any_adaptive {
            resolver.base_for_window(
                &ohlcv.high[start_idx..end_idx],
                &ohlcv.low[start_idx..end_idx],
                &ohlcv.close[start_idx..end_idx],
            )?
        } else {
            None
        };
        windows.push((start_idx, end_idx, base));
    }
    Ok(windows)
}

/// Simulate the candidate on the pre-planned prop-firm windows and check each
/// against `compute_prop_firm_risk_summary`. Returns the fraction of windows
/// whose `all_rules_passed` flag is true.
///
/// This measures what an actual prop-firm challenge measures (one
/// 30-day window, FTMO rules) — much more directly relevant than the
/// "every walkforward split must be profitable" gate. The settings come from
/// the ONE resolver, so an adaptive candidate is challenged under the SAME
/// volatility-scaled stop it was scored under — with the base re-derived per
/// window, as live trading derives it from its own recent buffer.
fn compute_prop_firm_pass_rate(
    gene: &Gene,
    signals: &[i8],
    ohlcv: &Ohlcv,
    timestamps: &[i64],
    config: &DiscoveryConfig,
    overrides: &PropFirmGateOverrides,
    resolver: &GeneEvalSettingsResolver<'_>,
    windows: &[PropFirmWindow],
) -> (f64, usize) {
    if windows.is_empty() {
        return (0.0, 0);
    }
    let mut settings = resolver.settings_for_gene(gene);
    let initial_balance = config.initial_balance.max(1.0);

    let mut passes = 0usize;
    let mut counted = 0usize;
    for (start_idx, end_idx, window_base) in windows {
        let (start_idx, end_idx) = (*start_idx, *end_idx);
        if end_idx > signals.len() {
            // Defensive: a signal vector shorter than the planned series would
            // mis-align the window — skip rather than read out of range.
            continue;
        }
        if settings.adaptive_vol_mult > 0.0 {
            // The base series is indexed per bar of the simulated slice, so
            // each window uses ITS OWN base (planned above); `None` here means
            // the window was too short for the estimator ⇒ fixed-pip fallback,
            // the same policy every other adaptive path applies.
            settings.adaptive_base_pips = window_base.clone();
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

/// AREA 2 / Stage A (2026-06-09) — serializes GPU launches across the
/// quality-screen's candidate `rayon::par_iter`. The Monte-Carlo screen now
/// fires ONE batched GPU population launch per candidate (mc_runs perturbed
/// genes), but that launch happens from inside the outer candidate par_iter:
/// without this lock, N rayon threads would each build a cubecl GPU client
/// concurrently → VRAM × N → OOM on a single device. Holding this lock around
/// the launch lets candidates SHARE one client one-at-a-time (the GPU is one
/// device anyway); the CPU-bound screen work (regime robustness, spread
/// sensitivity, metrics analysis) still parallelizes freely across threads.
///
/// On the non-GPU build `validation_genes_population` is pure CPU (rayon
/// internally), so the lock would needlessly serialize CPU work — it is only
/// taken under `cfg(feature = "gpu")`.
#[cfg(feature = "gpu")]
static GPU_LAUNCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Synthesise each prefiltered candidate's full-series signal vector and keep
/// the ones that fire often enough, returning
/// `(survivors, candidates_that_fired_at_all)`.
///
/// The second value is diagnostic counter #2: how many genes generated ANY
/// non-zero signal? A gene whose `long_threshold` exceeds the largest possible
/// combined signal never fires. It is tracked separately from the `min_trades`
/// gate so the funnel can tell "the SMC gate killed everything" — the common
/// empty-portfolio root cause — apart from "fired, but too rarely".
///
/// `min_trades` is compared against BARS THAT FIRE, not against executed
/// trades; that has always been this gate's meaning and the funnel's
/// `passed_min_trades` count is calibrated on it.
///
/// The SMC gate arrays are gene-independent — `build_smc_arrays` takes no
/// `Gene`, and `features`/`ohlcv` are fixed for the whole screen — so they are
/// built once here instead of once per candidate. On a full series that
/// rebuild dominated the stage: eleven fresh full-series arrays plus the
/// `derive_smc_arrays` scan, repeated for every candidate, all producing
/// byte-identical output. Each survivor's signal vector is unchanged.
fn screen_candidates_by_signal_count(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    prefiltered: Vec<(usize, Gene)>,
    eval_config: &EvaluationConfig,
    min_trades: usize,
) -> (Vec<(usize, Gene, Vec<i8>)>, usize) {
    // An empty pool pays nothing. `build_smc_arrays` scans every bar of the
    // series (~90 f64 ops each) before it knows there is no gene to gate, and
    // an empty prefilter is the normal outcome of a run that found nothing —
    // which is most M3 runs today.
    if prefiltered.is_empty() {
        return (Vec::new(), 0);
    }
    let smc = SmcGateArrays::build(features, ohlcv);
    let nonzero_signal_count = std::sync::atomic::AtomicUsize::new(0);
    let survivors: Vec<(usize, Gene, Vec<i8>)> = prefiltered
        .into_par_iter()
        .filter_map(|(candidate_idx, gene)| {
            let sig = signals_for_gene_full_with_smc(features, &gene, eval_config, &smc);
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
    (
        survivors,
        nonzero_signal_count.load(std::sync::atomic::Ordering::Relaxed),
    )
}

fn finalize_candidates_with_progress<F>(
    candidates: Vec<Gene>,
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f32,
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
        let total = features.n_values();
        let mut nan = 0usize;
        let mut zero = 0usize;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum_abs = 0.0_f64;
        let mut finite_count = 0usize;
        for v in features.iter_values() {
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
            rows = features.n_samples(),
            cols = features.n_features(),
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
        let trailing = features.n_samples().min(1000);
        if trailing > 1 {
            let n_cols = features.n_features();
            let mut zero_var_cols = 0usize;
            let mut named_examples: Vec<String> = Vec::new();
            let start_row = features.n_samples() - trailing;
            for c in 0..n_cols {
                let mut col_min = f32::INFINITY;
                let mut col_max = f32::NEG_INFINITY;
                let mut finite_seen = 0usize;
                for r in start_row..features.n_samples() {
                    let v = features.feature_at(r, c);
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
                    if named_examples.len() < 5 && c < features.names.len() {
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
            // capped at the Risky risk ceiling (30% per operator decision
            // 2026-08-09) so the growth projection matches what the sim is
            // allowed to bet. Half-Kelly is deliberately conservative vs full
            // Kelly: it keeps the projected growth ruin-aware, so a strategy
            // whose full-Kelly size would be ruinous does not score as if it
            // compounds cleanly.
            let f_star = if pf > 1.0 && p > 0.0 {
                p * (pf - 1.0) / pf
            } else {
                0.0
            };
            let f = (f_star * 0.5).clamp(0.0, 0.30);
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

    // ── THE EARLY-REJECT PREDICATE ─────────────────────────────────────────
    //
    // Here, and not one line later. This is the last point at which nothing
    // expensive has happened yet: signal generation for every candidate, the
    // quality screen (50.4% of the cited run's wall time), the prop-firm window
    // gate, the walk-forward and CPCV all lie AFTER it. The predicate itself is
    // `O(population)` field reads over metrics the GA already produced.
    //
    // BIASED TOWARD PASSING, deliberately and irreversibly: see
    // `evaluate_batch_early_reject`. A false reject is invisible and permanent;
    // a false accept only costs time.
    let batch_verdict = evaluate_batch_early_reject(&ranked_candidate_genes, &config.target_profile);
    record_batch_verdict(streaming_sweep_cursor(), &batch_verdict);
    if batch_verdict.is_reject() {
        tracing::warn!(
            target: "neoethos_search::batch_ledger",
            cursor = streaming_sweep_cursor(),
            reason = batch_verdict.reason(),
            population = batch_verdict.population,
            measured = batch_verdict.measured,
            best_expectancy = batch_verdict.best_expectancy,
            best_profit_factor = batch_verdict.best_profit_factor,
            best_payoff_ratio = batch_verdict.best_payoff_ratio,
            best_trades = batch_verdict.best_trades,
            expectancy_floor = batch_verdict.floor,
            margin = batch_verdict.margin,
            "BATCH ABANDONED before the quality screen — not one candidate made money gross \
             and the best cost-charged expectancy is below the configured floor by the stated \
             margin. The quality screen, the prop-firm gate, the walk-forward and OOS \
             validation are SKIPPED for this batch. The floor is \
             models.prop_search_min_net_expectancy_per_trade; the margin only ever makes this \
             decision more permissive than that floor."
        );
    } else {
        tracing::info!(
            target: "neoethos_search::batch_ledger",
            cursor = streaming_sweep_cursor(),
            reason = batch_verdict.reason(),
            population = batch_verdict.population,
            measured = batch_verdict.measured,
            best_expectancy = batch_verdict.best_expectancy,
            best_profit_factor = batch_verdict.best_profit_factor,
            best_payoff_ratio = batch_verdict.best_payoff_ratio,
            expectancy_floor = batch_verdict.floor,
            margin = batch_verdict.margin,
            "batch kept — the early-reject predicate did not fire"
        );
    }

    let min_trades = min_trades_required(
        &features.timestamps,
        config.min_trades_per_day,
        features.n_samples(),
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
    // AN ABANDONED BATCH STOPS HERE. Emptying the ladder's input is how the
    // rejection is enforced: signal generation, the quality screen, the
    // prop-firm window gate, correlation pruning, the walk-forward and OOS
    // validation all iterate over this list, so an empty one costs each of them
    // nothing and the cycle returns an honest empty portfolio. `ranked_candidate_genes`
    // is NOT emptied — the batch's own evidence stays in the artifact, which is
    // what lets a reader check the predicate's decision after the fact.
    let prefiltered: Vec<(usize, Gene)> = if batch_verdict.is_reject() {
        Vec::new()
    } else {
    ranked_candidates
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
        .collect()
    };
    let post_passes_filter = prefiltered.len();
    funnel.record_stage("passed_base_filter", ranked_total, post_passes_filter);
    // The batch rejection is recorded on THIS stage rather than on a stage of
    // its own: `FunnelProfile::record_stage` silently no-ops on a name that is
    // not in the declared stage list (`funnel_profile.rs`), so inventing
    // `early_reject_batch` here would have written the rejection to nowhere —
    // a silent drop in the very accounting that exists to prevent one. The
    // authoritative census is the batch ledger
    // (`log_batch_rejection_summary`); this line is so the PERSISTED funnel
    // also says why every candidate vanished at this step.
    if batch_verdict.is_reject() {
        funnel.add_reject_reason(
            "passed_base_filter",
            format!("early_reject_batch.{}", batch_verdict.reason()),
            ranked_total,
        );
    }
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
    // Said out loud, not only written to the funnel file.
    //
    // A run that rejects every candidate reports `post_passes_filter=0` and
    // stops, and the reasons sit in a JSON the probe never writes. Which floor
    // did it is the whole question — "all 49 344 exceeded the drawdown cap" and
    // "all 49 344 had too few trades" call for opposite responses, and telling
    // them apart should not need a second run.
    // Fires on "almost none", not only on "none".
    //
    // A measured M3 run had ranked=22 486 -> post_passes_filter=2, and this
    // stayed silent because two is not zero. Two survivors out of 22 486 is the
    // same diagnosis as none and needs the same answer — which floor did it —
    // and the run then reported an empty portfolio with nothing said about why.
    if post_passes_filter * 100 <= ranked_total && ranked_total > 0 {
        tracing::warn!(
            target: "neoethos_search::funnel",
            ranked = ranked_total,
            max_dd_exceeded = reject_dd,
            win_rate_too_low = reject_win_rate,
            profit_factor_too_low = reject_profit_factor,
            fitness_too_low = reject_fitness,
            other_threshold = reject_other,
            max_dd_floor = config.filtering.max_dd,
            min_win_rate_floor = config.filtering.min_win_rate,
            min_profit_factor_floor = config.filtering.min_profit_factor,
            "every candidate failed the base filter — this is which floor did it"
        );
    }

    // Item 6: use the SMC-gated signal path so the post-search "min_trades"
    // filter sees the SAME trade count the evaluator scored. The previous
    // `signals_for_gene` ignored gene SMC flags; some candidates passed the
    // search archive (with their SMC-gated trade count) but were then pruned
    // here because the un-gated count was higher than min_trades.
    let eval_config_for_signals = config
        .evaluation_config_with_smc_gate(ohlcv.close.last().copied(), effective_smc_gate_threshold);

    let (signals_with_idx, post_nonzero_signal) = screen_candidates_by_signal_count(
        features,
        ohlcv,
        prefiltered,
        &eval_config_for_signals,
        min_trades,
    );
    let post_min_trades = signals_with_idx.len();
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

    // ── PBO candidate snapshot (2026-07-02) ────────────────────────────────
    // The Probability-of-Backtest-Overfitting estimate needs the SELECTION
    // POOL, not just the final portfolio: it asks "when I crown an in-sample
    // champion among these candidates, does that champion also perform
    // out-of-sample?". Take the top-by-fitness base-filter survivors here —
    // the richest honest pool before the strict gates shrink it. Capped at 64
    // (rank statistics saturate well before that; keeps the extra CPCV-side
    // evaluations bounded).
    let mut pbo_candidates: Vec<Gene> = filtered.iter().map(|(_, g)| g.clone()).collect();
    pbo_candidates.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pbo_candidates.truncate(64);
    // ── NEVER-ZERO best-effort snapshot (2026-06-09, operator non-negotiable) ──
    // Capture the top-by-fitness base-filtered survivors WITH their signals here,
    // at the richest point before the strict quality / prop-firm / correlation
    // gates can empty the portfolio. If those gates reject EVERY candidate, we
    // promote this set (correlation-pruned, honestly labeled "did not pass the
    // prop bar") so a hard combo (e.g. AUDUSD M3) emits its best-found genes
    // instead of dying with zero output. Cloning only the top N keeps it cheap.
    const FALLBACK_PORTFOLIO_MAX: usize = 8;
    let best_effort_fallback: Vec<((usize, Gene), Vec<i8>)> = {
        let mut order: Vec<usize> = (0..filtered.len()).collect();
        order.sort_by(|&a, &b| {
            filtered[b]
                .1
                .fitness
                .partial_cmp(&filtered[a].1.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order
            .into_iter()
            .take(FALLBACK_PORTFOLIO_MAX)
            .map(|i| (filtered[i].clone(), signals_map[i].clone()))
            .collect()
    };
    progress_fn(DiscoveryProgress::CandidatesFiltered {
        passed_filters: filtered.len(),
        evaluated_candidates: ranked_candidates.len(),
        min_trades_required: min_trades,
    });

    let filtered_count = filtered.len();
    let mut quality_metrics = Vec::new();
    // Filled by the quality screen below; stays all-zero when the screen is
    // skipped, so the funnel never reports invented rejections.
    let mut quality_rejects = QualityScreenRejects::default();
    // Same contract: all-zero when the screen is skipped, so an absent band is
    // never reported as a band that everything passed.
    let mut cost_band_census = CostBandCensus::default();
    // Per-strategy companion to the census (audit #71): the census answers "how
    // many", this answers "which". Same contract — EMPTY when the screen is
    // skipped, and an absent entry reads as `Unmeasured`, never as a pass.
    let mut cost_band_by_strategy: Vec<(String, CostBandVerdict)> = Vec::new();
    let mut logged_trades = Vec::new();
    if Gene::requires_quality_screen(&config.filtering) {
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "quality_screen",
            detail: format!(
                "screening {filtered_count} candidates (full backtest + Monte-Carlo \
                 perturbations each) — silent but active"
            ),
        });
        /// `.6` is the COST-BAND VERDICT — see [`CostBandVerdict`]. It rides on
        /// the survivor so the report cannot lose it between the screen and the
        /// export, which is how "we measured the band" becomes "we mentioned
        /// the band once in a log".
        type QualityCandidate = (
            usize,
            Gene,
            Vec<i8>,
            StrategyMetrics,
            bool,
            Vec<Trade>,
            CostBandVerdict,
        );
        let analyzer = quality_analyzer_for_config(config);
        let initial_balance = config.initial_balance;

        // AREA 2 / Stage A (2026-06-09): deterministic per-combo seed for the
        // Monte-Carlo perturbation RNG. Derived ONLY from combo-stable material
        // (symbol + timeframe label + sample count) so the seed is identical on
        // every run of the same combo+window and reproduces CPU↔GPU bit-for-bit.
        // It is XOR-combined per (candidate_idx, run_idx) inside the loop so each
        // (candidate, run) draws an independent-but-reproducible perturbation.
        let combo_seed: u64 = {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
            let mut mix = |bytes: &[u8]| {
                for &b in bytes {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01B3);
                }
            };
            mix(config.evaluation_symbol.as_bytes());
            mix(config.timeframe_label.as_bytes());
            mix(&(ohlcv.close.len() as u64).to_le_bytes());
            h
        };

        // Outer-parallel quality screen: each candidate runs simulate_trades +
        // 100 MC perturbations + spread sensitivity independently. Previously
        // the outer loop was serial and only the 100-run MC was parallel,
        // which under-utilised cores when the candidate set was large. Move
        // parallelism to the outer level and keep the MC loop serial — this
        // avoids rayon nested-parallel oversubscription and gives ~Ncores×
        // throughput on the per-candidate work. AREA 2 / Stage A: the inner MC
        // loop now builds `mc_runs` perturbed genes deterministically and fires
        // ONE batched GPU population launch (CPU fallback) per candidate via
        // `validation_genes_population`, replacing the per-run serial
        // `signals_for_gene_full` + `simulate_trades_core`.
        // Per-reason rejection counters for the quality screen. This stage is
        // routinely the funnel's bottleneck — a real AUDUSD H4 run took 7 793
        // candidates to 1 here — and until now it reported a single collapsed
        // number, so there was no way to tell whether the survivors were being
        // killed by the base metrics, the regime check, the Monte-Carlo
        // perturbation floor or the spread sensitivity test. Answering "which
        // gate costs us the candidates" is the prerequisite for any decision
        // about widening one, and it cannot be answered by staring at a total.
        //
        // These are the atomics the 2026-05-26 note above said a follow-up
        // would need. They are pure instrumentation: every counter sits next to
        // a `return None` that already existed, so the surviving set is
        // unchanged.
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        let rejected_base_quality = AtomicUsize::new(0);
        // MEASUREMENT SLICE (2026-08-09): the eight criteria the single
        // `rejected_base_quality` counter used to collapse. Indexed by
        // `BaseQualityReject` in `base_quality_index` order below, so adding a
        // variant without adding a counter is a compile error at the match.
        let bq_account_wiped = AtomicUsize::new(0);
        let bq_profile_net_expectancy = AtomicUsize::new(0);
        let bq_profile_expectancy_significance = AtomicUsize::new(0);
        let bq_profile_win_rate = AtomicUsize::new(0);
        let bq_profile_payoff_ratio = AtomicUsize::new(0);
        let bq_profile_in_market = AtomicUsize::new(0);
        let bq_opportunistic_lane_closed = AtomicUsize::new(0);
        let bq_positive_months = AtomicUsize::new(0);
        let bq_trades_per_month = AtomicUsize::new(0);
        let bq_monthly_return = AtomicUsize::new(0);
        let rejected_regime = AtomicUsize::new(0);
        let rejected_mc_error = AtomicUsize::new(0);
        // Split from `rejected_mc_error` (2026-08-09): a failed sensitivity
        // launch is an infrastructure failure in a different subsystem and must
        // not be reported as a Monte-Carlo problem.
        let rejected_sensitivity_error = AtomicUsize::new(0);
        let rejected_mc_floor = AtomicUsize::new(0);
        let rejected_sensitivity = AtomicUsize::new(0);
        // Cost-band census. These do NOT reject: they classify what the run is
        // entitled to claim. `optimistic_only` is the one that matters — a
        // candidate profitable at 1.6 pips and not at 2.4 is not a result, and
        // without this counter it is indistinguishable from one that is.
        let cost_band_survived = AtomicUsize::new(0);
        let cost_band_optimistic_only = AtomicUsize::new(0);
        let cost_band_failed = AtomicUsize::new(0);
        let cost_band_unmeasured = AtomicUsize::new(0);
        let cost_band_not_discriminating = AtomicUsize::new(0);
        // PER-SESSION TRADE CENSUS. While `session_spread_pips` is unset the run
        // charges a FLAT spread at 03:00 Tokyo and at the London open alike, so
        // a gene that concentrates its entries in the Asian session is priced on
        // a subsidy. The curve is wired end to end and simply unpopulated; until
        // it is measured, the size of that exposure should be a NUMBER in the
        // log rather than a caveat in a report. Counted over every screened
        // candidate, before any gate — the honest denominator. Pnl is summed in
        // cents so it can live in an atomic.
        let session_trade_counts = [
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        ];
        let session_pnl_cents = [
            std::sync::atomic::AtomicI64::new(0),
            std::sync::atomic::AtomicI64::new(0),
            std::sync::atomic::AtomicI64::new(0),
        ];
        // Monte-Carlo pass counts of the candidates the floor rejected, so the
        // floor can be judged against the distribution it is cutting rather
        // than in the abstract: "7 000 rejects that scored 68/100" and "7 000
        // that scored 4/100" call for opposite decisions.
        let mc_near_miss = AtomicUsize::new(0);

        let pairs: Vec<((usize, Gene), Vec<i8>)> = filtered.into_iter().zip(signals_map).collect();

        // ONE resolver for every serial backtest in this screen (built over
        // the full series the screen simulates) and ONE template source for
        // the population launches (which resolve adaptive stops themselves).
        // This is THE fix for the screen's measured 17.6x divergence: the
        // base-quality backtest below previously ran adaptive genes on their
        // unused fixed pips while GA scoring ran them volatility-scaled.
        let screen_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            pairs.iter().map(|((_, gene), _)| gene),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        let screen_templates = PopulationTemplateResolver::new(config, ohlcv.close.last().copied());

        // The bar-derived half of validation host prep, built ONCE for the whole
        // screen.
        //
        // This was rebuilt on every call: the transposed indicator matrix, the
        // month/day indices and eleven lookback-heavy SMC series, over the full
        // history, seven times. None of it depends on which genes are being
        // evaluated. `validation_genes_population`'s own in-tree measurement
        // reads "eighteen of these calls take 413.6 s of a 452.4 s run — 23 s
        // each — while the device stage timing inside one adds up to 0.30 s";
        // this is a large part of the 22.7 s nobody could account for.
        let screen_prep = crate::genetic::search_engine::ValidationPrep::build(features, ohlcv)?;

        // ── Monte-Carlo perturbations, batched ────────────────────────────
        //
        // The screen below used to call the evaluator once per candidate: a
        // real AUDUSD H4 run made 7 793 separate launches of 100 perturbed
        // genes each. Thousands of small launches waste a card however fast the
        // kernel is — the fixed per-call cost dominates when each call carries
        // so little work.
        //
        // The perturbations are seeded per (combo, candidate, run), so batching
        // reproduces every candidate's result exactly; only the number of
        // launches changes. Chunked by candidate so peak host memory is a
        // function of the chunk rather than the population — cloning 7 793 x
        // 100 genes at once would be a gigabyte-scale allocation for nothing.
        //
        // `None` marks a candidate whose batch failed to evaluate: a real bug,
        // reported as such below, never silently counted as "zero profitable".
        // Verbatim, NOT `.max(1)`-ed: `mc_runs == 0` is a degenerate config
        // whose behaviour (zero perturbation runs per candidate) must not be
        // changed by a batching edit.
        let mc_runs = config.mc_runs as usize;
        // ── ONE WORK LIST, ONE LAUNCH ─────────────────────────────────────
        //
        // This screen used to be SEVEN launches over the same bars: six chunks
        // of Monte-Carlo perturbations plus a sensitivity pass. Each chunk
        // cloned its own genes, re-transposed the whole indicator matrix,
        // rebuilt the month/day indices and re-derived eleven lookback-heavy SMC
        // series — on 843 456 bars — to evaluate genes that had changed and bars
        // that had not.
        //
        // It is one array and one submission now, because the descriptor carries
        // what used to force a separate launch:
        //
        //   * a Monte-Carlo run is a scenario naming its perturbed gene (host
        //     lane) or its perturbation counter (device lane);
        //   * a sensitivity run is a scenario naming the SAME gene with its own
        //     spread and commission — no second settings struct, no second
        //     launch;
        //   * and the bar-derived prep is built ONCE, above, for all of it.
        //
        // The DEVICE chunking is gone rather than resized. It existed to stay
        // under the card's scenario ceiling, and that is the evaluator's own
        // business: it queries free VRAM, sizes the launch and splits the
        // DESCRIPTOR array itself, so a caller guessing a chunk size can only
        // get it wrong. Whatever `screen_chunk` is below, the per-candidate
        // results are identical — genes and scenarios are independent.
        //
        // What remains is a HOST-memory bound, and it is a different quantity
        // for a different reason. See `MAX_STAGED_CLONES`.
        // REPORTED, NOT USED. This queries free VRAM through `submission_ceiling`
        // and the screen sizes nothing from it — the evaluator asks the card
        // itself and splits the descriptor array. It is logged so an operator can
        // compare what the card would host against what the launch actually
        // asked for; a caller that SIZED from it could only get it wrong, which
        // is what the six-chunk device loop was.
        let gpu_ceiling =
            crate::eval::gpu_submission_ceiling(ohlcv.close.len(), features.n_features());
        let bars = ohlcv.close.len();
        let candidates = pairs.len();
        let device_mc = crate::gpu_native::scenario::device_monte_carlo();

        // How many perturbed gene CLONES may exist in RAM at once.
        //
        // The host Monte-Carlo lane materialises one `Gene` per (candidate, run)
        // — ~1.3 KB measured — and `candidates * mc_runs` is a pure function of
        // USER PARAMETERS. A 7 793-candidate screen at 100 runs is 779 300 clones,
        // about a gigabyte, and the never-OOM invariant is explicit that peak
        // memory must follow the hardware and never the parameters. Staging the
        // whole screen at once would have made it follow `mc_runs`.
        //
        // So the screen walks candidates in chunks sized by this budget. Note
        // what that is NOT: it is not the old six-chunk device loop. Each chunk
        // is still ONE launch covering its Monte-Carlo AND its cost scenarios,
        // and the bar-derived prep is built once for all of them. At the measured
        // 174 candidates x 100 runs this is a single chunk.
        //
        // The device Monte-Carlo lane removes this entirely — no clone exists
        // there, the counter is in the descriptor — which is the second thing it
        // buys after the launch count.
        const MAX_STAGED_CLONES: usize = 131_072;
        // The clamp FLOOR defeated the budget when `mc_runs` exceeded it.
        //
        // `MAX_STAGED_CLONES / mc_runs` is 0 for `mc_runs > 131 072`, and
        // `.clamp(1, ..)` lifted that to 1 — so one chunk staged `mc_runs`
        // clones and peak host memory became exactly f(mc_runs), which is the
        // invariant the comment above invokes by name. Refuse instead: the
        // number is the operator's and the machine's limit is not, so this is a
        // configuration to fix rather than a memory to gamble.
        if !device_mc && mc_runs > MAX_STAGED_CLONES {
            anyhow::bail!(
                "mc_runs = {mc_runs} exceeds the host staging budget of {MAX_STAGED_CLONES} \
                 perturbed gene clones (~1.3 KB each). The host Monte-Carlo lane materialises \
                 one clone per (candidate, run), so this would make peak host memory a function \
                 of a user parameter. Lower mc_runs, or turn on the device Monte-Carlo lane, \
                 where the counter travels in the descriptor and no clone exists at all."
            );
        }
        let screen_chunk = if device_mc || mc_runs == 0 {
            candidates.max(1)
        } else {
            (MAX_STAGED_CLONES / mc_runs).clamp(1, candidates.max(1))
        };

        // Can the sensitivity costs be carried in a descriptor EXACTLY?
        //
        // The fields are integers, so the answer is "only if they round-trip
        // through the same division the device performs". When they do not, this
        // does NOT round them — a spread quietly moved by 0.4 % is a screen
        // reporting that strategies survive costs they never paid. It falls back
        // to a second launch carrying the exact f64 in the settings struct,
        // which is what the code did before scenarios existed, and says so.
        let sensitivity_spread =
            crate::gpu_native::scenario::spread_ticks_exact(config.sensitivity_spread_pips);
        let sensitivity_commission = crate::gpu_native::scenario::commission_micros_exact(
            config.sensitivity_commission_per_lot,
        );
        let fuse_costs = sensitivity_spread.is_some() && sensitivity_commission.is_some();
        if !fuse_costs {
            tracing::warn!(
                target: "neoethos_search::discovery",
                spread_pips = config.sensitivity_spread_pips,
                commission_per_lot = config.sensitivity_commission_per_lot,
                "sensitivity costs cannot be carried in a scenario descriptor without \
                 changing them — running the sensitivity pass as its own launch with the \
                 exact values rather than quantising what the screen measures"
            );
        }

        tracing::info!(
            target: "neoethos_search::discovery",
            candidates,
            mc_runs,
            device_monte_carlo = device_mc,
            screen_chunk,
            launches = candidates.div_ceil(screen_chunk.max(1)),
            scenarios_per_chunk = screen_chunk * (mc_runs + usize::from(fuse_costs)),
            fused_cost_pass = fuse_costs,
            // For comparison only — nothing here is sized from it.
            gpu_ceiling_reported = gpu_ceiling.unwrap_or(0),
            "quality screen — ONE work list per chunk, Monte-Carlo and costs together; \
             the evaluator sizes and splits it against free VRAM, and per-candidate \
             results are split-invariant"
        );

        let mut mc_profitable_runs: Vec<Option<usize>> = Vec::with_capacity(candidates);
        let mut fused_sensitivity: Option<Vec<Option<f64>>> =
            fuse_costs.then(|| Vec::with_capacity(candidates));

        for chunk in pairs.chunks(screen_chunk) {
            let chunk_len = chunk.len();

            // ── Genes ─────────────────────────────────────────────────────
            //
            // The base candidates first, at indices 0..chunk_len, because every
            // cost scenario and (in the device lane) every perturbation names one
            // of them. The host lane appends the perturbed clones after them.
            let mut screen_genes: Vec<Gene> =
                chunk.iter().map(|((_, gene), _)| gene.clone()).collect();
            let clone_base = screen_genes.len();
            if !device_mc && mc_runs > 0 {
                // THE DEFAULT AND THE REFERENCE. ChaCha8, host-side, in the exact
                // draw order the serial screen used, seeded per (combo,
                // candidate, run) — see `host_monte_carlo_perturbation`, which
                // both this and the pinning test call so the pin covers the code
                // rather than a copy of it.
                //
                // `map` over an indexed parallel iterator collects in order, so
                // the array stays candidate-major with runs ascending, which is
                // what the descriptor indices below rely on. The RNG is seeded
                // per (combo, candidate, run) and never shared, so parallel
                // construction is bit-identical to the serial one — and the seed
                // uses the candidate's ORIGINAL index, not its position in this
                // chunk, so chunking cannot change a single draw.
                let clones: Vec<Gene> = chunk
                    .par_iter()
                    .map(|((candidate_idx, gene), _)| {
                        (0..mc_runs as u64)
                            .map(|run_idx| {
                                host_monte_carlo_perturbation(
                                    gene,
                                    combo_seed,
                                    *candidate_idx,
                                    run_idx,
                                )
                            })
                            .collect::<Vec<Gene>>()
                    })
                    .reduce(Vec::new, |mut acc, mut part| {
                        acc.append(&mut part);
                        acc
                    });
                screen_genes.extend(clones);
            }

            // ── The work list ─────────────────────────────────────────────
            //
            // Monte-Carlo scenarios first, candidate-major with runs ascending,
            // then the cost scenarios — ONE array, ONE call. `scenario_id` is
            // the position, so the evaluator's positional check against the
            // returned rows is self-describing: a permuted result names the
            // position it should have been at.
            let mut work: Vec<neoethos_gpu_contracts::device::ScenarioDescriptor> =
                Vec::with_capacity(chunk_len * (mc_runs + 1));
            for (position, ((candidate_idx, _), _)) in chunk.iter().enumerate() {
                for run in 0..mc_runs as u64 {
                    let id = work.len() as u64;
                    work.push(if device_mc {
                        // The gene is the unperturbed candidate; the counter is
                        // what makes this run different. No clone exists on the
                        // host at all — that is what the device lane buys.
                        crate::gpu_native::scenario::perturb_scenario(
                            position as u64,
                            id,
                            bars,
                            combo_seed ^ ((*candidate_idx as u64) << 20) ^ run,
                        )
                    } else {
                        // The perturbation is already in the gene, so this is an
                        // ordinary full-series evaluation of clone
                        // `clone_base + position * mc_runs + run`.
                        crate::gpu_native::scenario::base_scenario(
                            (clone_base + position * mc_runs + run as usize) as u64,
                            id,
                            bars,
                        )
                    });
                }
            }
            let mc_total = work.len();
            if fuse_costs {
                for position in 0..chunk_len {
                    let id = work.len() as u64;
                    work.push(crate::gpu_native::scenario::cost_scenario(
                        position as u64,
                        id,
                        bars,
                        sensitivity_spread,
                        sensitivity_commission,
                    ));
                }
            }

            // ── The launch ────────────────────────────────────────────────
            //
            // Cost/pip configuration is shared: the template helper takes a gene
            // only for the price hint, and each gene's own SL/TP and
            // adaptive-stop regime are re-resolved inside the prep.
            //
            // GPU_LAUNCH_LOCK COVERS THE DEVICE CALL AND NOTHING ELSE. It exists
            // so a rayon `par_iter` cannot create one ~16 GB session per worker;
            // holding it across the gene pack and the adaptive-stop resolution —
            // which is where it used to sit, inside
            // `validation_genes_population` — serialises CPU work that no other
            // thread's device access could conflict with.
            let fused = if work.is_empty() {
                Ok(Vec::new())
            } else {
                let screen_settings = screen_templates.template(&chunk[0].0.1);
                match crate::genetic::search_engine::prepare_validation_population(
                    ohlcv,
                    &screen_genes,
                    &eval_config_for_signals,
                    &screen_settings,
                ) {
                    Ok(prepared) => {
                        #[cfg(feature = "gpu")]
                        let _gpu_guard =
                            GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                        crate::genetic::search_engine::validation_genes_scenarios(
                            features,
                            ohlcv,
                            &screen_prep,
                            &prepared,
                            &work,
                        )
                    }
                    Err(error) => Err(error),
                }
            };

            // ── Demultiplex ───────────────────────────────────────────────
            //
            // `None` marks a candidate whose evaluation failed: a real bug,
            // reported as such below, never silently counted as "zero
            // profitable".
            match fused {
                Ok(rows) if rows.len() == work.len() => {
                    for candidate in 0..chunk_len {
                        mc_profitable_runs.push(if mc_runs == 0 {
                            // A degenerate config asks for no perturbation runs;
                            // zero profitable out of zero is the honest answer
                            // and not a failure.
                            Some(0)
                        } else {
                            let start = candidate * mc_runs;
                            Some(
                                rows[start..start + mc_runs]
                                    .iter()
                                    .filter(|m| m[0] > 0.0)
                                    .count(),
                            )
                        });
                    }
                    if let Some(sensitivity) = fused_sensitivity.as_mut() {
                        for candidate in 0..chunk_len {
                            sensitivity.push(Some(rows[mc_total + candidate][0]));
                        }
                    }
                }
                Ok(rows) => {
                    tracing::warn!(
                        target: "neoethos_search::discovery",
                        expected = work.len(),
                        returned = rows.len(),
                        candidates = chunk_len,
                        "quality-screen launch returned the wrong number of rows — rejecting its candidates"
                    );
                    mc_profitable_runs.extend(std::iter::repeat_n(None, chunk_len));
                    if let Some(sensitivity) = fused_sensitivity.as_mut() {
                        sensitivity.extend(std::iter::repeat_n(None, chunk_len));
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        target: "neoethos_search::discovery",
                        error = %error,
                        candidates = chunk_len,
                        scenarios = work.len(),
                        "quality-screen launch failed — rejecting its candidates"
                    );
                    mc_profitable_runs.extend(std::iter::repeat_n(None, chunk_len));
                    if let Some(sensitivity) = fused_sensitivity.as_mut() {
                        sensitivity.extend(std::iter::repeat_n(None, chunk_len));
                    }
                }
            }
        }

        // ── Spread/slippage sensitivity ───────────────────────────────────
        //
        // The stress test is the same backtest over the same bars with a wider
        // spread and a higher commission, and its verdict is a single number:
        // does net profit survive? That is metric slot 0.
        //
        // Normally it is already answered — the cost scenarios rode along in the
        // launch above and cost one extra thread each. This arm exists only for
        // the case the descriptor cannot carry the configured costs EXACTLY, and
        // its whole point is that the screen keeps measuring the operator's
        // actual numbers rather than the nearest millipip: one extra launch, the
        // exact f64 in the settings struct, loudly logged where it was decided.
        let sensitivity_net_profit: Vec<Option<f64>> = match fused_sensitivity {
            Some(values) => values,
            None => {
                let mut settings = screen_templates.template(&pairs[0].0.1);
                settings.spread_pips = config.sensitivity_spread_pips;
                settings.commission_per_trade = config.sensitivity_commission_per_lot;
                // A flat sensitivity spread must BYPASS the per-hour resolution,
                // exactly as the fused path does.
                //
                // The device's `spread_ticks` override replaces the whole
                // per-bar lookup, and the CPU mirror clears the profile for the
                // same reason. This arm used to set only the scalar while leaving
                // the profile active, so every real bar still used one of the
                // three original buckets.
                // With a profile configured the sensitivity test therefore ran at
                // the ORIGINAL spread and reported that every strategy survives a
                // cost it was never charged.
                //
                // Which arm runs is decided by whether the operator's spread
                // round-trips through millipips, so a fourth decimal place
                // silently changed what the screen measured.
                settings.session_spread_profile = None;
                // Only the BASE candidates — the perturbed clones are not part
                // of this test — so this is one gene per candidate and one
                // full-series scenario each. No `mc_runs` multiplier, so the
                // staging is bounded by the candidate count alone and needs no
                // chunking of its own; the evaluator splits the descriptor array
                // against free VRAM as usual.
                let base_genes: Vec<Gene> =
                    pairs.iter().map(|((_, gene), _)| gene.clone()).collect();
                let base_work: Vec<neoethos_gpu_contracts::device::ScenarioDescriptor> = (0
                    ..candidates as u64)
                    .map(|candidate| {
                        crate::gpu_native::scenario::base_scenario(candidate, candidate, bars)
                    })
                    .collect();
                let evaluated = match crate::genetic::search_engine::prepare_validation_population(
                    ohlcv,
                    &base_genes,
                    &eval_config_for_signals,
                    &settings,
                ) {
                    Ok(prepared) => {
                        #[cfg(feature = "gpu")]
                        let _gpu_guard =
                            GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                        crate::genetic::search_engine::validation_genes_scenarios(
                            features,
                            ohlcv,
                            &screen_prep,
                            &prepared,
                            &base_work,
                        )
                    }
                    Err(error) => Err(error),
                };
                match evaluated {
                    Ok(metrics) if metrics.len() == candidates => {
                        metrics.iter().map(|m| Some(m[0])).collect()
                    }
                    Ok(metrics) => {
                        tracing::warn!(
                            target: "neoethos_search::discovery",
                            expected = candidates,
                            returned = metrics.len(),
                            "sensitivity launch returned the wrong number of rows — rejecting every candidate"
                        );
                        vec![None; candidates]
                    }
                    Err(error) => {
                        tracing::warn!(
                            target: "neoethos_search::discovery",
                            error = %error,
                            candidates,
                            "sensitivity launch failed — rejecting every candidate"
                        );
                        vec![None; candidates]
                    }
                }
            }
        };
        // ── THE COST BAND ────────────────────────────────────────────────────
        //
        // A backtest result is a function of the cost you charged it. Nobody
        // knows their all-in round-trip cost to better than a few tenths of a
        // pip: spread moves by hour and by news, commission is quoted per side,
        // and slippage is not a constant. Reporting ONE number invites the
        // reader to believe it, which is how a run that only clears at the
        // optimistic end gets read as a result.
        //
        // So every survivor is re-measured at BOTH edges of the operator's band
        // (`risk.cost_band_{optimistic,pessimistic}_pips`, default 1.6 / 2.4)
        // and a candidate that is profitable at the optimistic edge but not at
        // the pessimistic one is FLAGGED. The flag does not reject it — the
        // existing sensitivity gate still decides that — because the band's job
        // is to make the fragility visible, not to move a threshold nobody
        // agreed to move.
        //
        // The band is a TOTAL round-trip cost, so it is charged entirely as
        // spread with commission zeroed; charging both would double-count.
        // Session profile cleared for the same reason the sensitivity arm
        // clears it: a flat stress cost must bypass the per-hour lookup or the
        // pass measures the original spread and reports that everything
        // survives a cost it was never charged.
        //
        // ZERO expected value in money. It changes no strategy. It changes what
        // a reader is allowed to conclude.
        //
        // AND IT MUST BE ABLE TO DISCRIMINATE. Added 2026-08-09 after the review
        // showed the shipped band is arithmetically incapable of failing anyone:
        // the edges REPLACE the whole cost, cost is monotone, and both shipped
        // edges (1.6 / 2.4) sit BELOW the run's own charged cost (spread 1.5 +
        // slippage 0.5 + doubled commission ~1.4 = ~3.4 pips). Every survivor
        // would come back `SurvivesBand` on every run. Rather than spend two
        // population launches producing a guaranteed answer and a census that
        // reads as evidence, the band is SKIPPED and every candidate is marked
        // `NotDiscriminating` with the two numbers printed.
        let baseline_cost_pips = crate::run_identity::cost_pips_round_trip(
            config.evaluation_spread_pips,
            config.evaluation_commission_per_trade,
            eval_config_for_signals.pip_value_per_lot,
        );
        let band_discriminates = cost_band_discriminates(config.cost_band_pips, baseline_cost_pips);
        if config.cost_band_pips.is_some() && !band_discriminates {
            let (lo, hi) = config.cost_band_pips.unwrap_or((f64::NAN, f64::NAN));
            tracing::error!(
                target: "neoethos_search::cost_model",
                optimistic_pips = lo,
                pessimistic_pips = hi,
                baseline_cost_pips,
                spread_pips = config.evaluation_spread_pips,
                commission_per_trade = config.evaluation_commission_per_trade,
                pip_value_per_lot = eval_config_for_signals.pip_value_per_lot,
                "COST BAND CANNOT DISCRIMINATE: its pessimistic edge ({hi:.2} pips) is at or \
                 below the cost this run already charged ({baseline_cost_pips:.2} pips round \
                 trip). Cost is monotone, so every candidate that cleared the screen clears \
                 both edges BY CONSTRUCTION and the census would read clean on every run. The \
                 band is SKIPPED and every survivor is marked cost_band_not_discriminating. \
                 Fix: raise risk.cost_band_pessimistic_pips above {baseline_cost_pips:.2}, or \
                 lower the charged cost. This is a defect in the measuring instrument, not a \
                 result about any strategy."
            );
        }
        let cost_band_edges = config.cost_band_pips.filter(|_| band_discriminates);
        let mut cost_band_optimistic: Vec<Option<f64>> = vec![None; candidates];
        let mut cost_band_pessimistic: Vec<Option<f64>> = vec![None; candidates];
        if let Some((optimistic_pips, pessimistic_pips)) = cost_band_edges.filter(|_| candidates > 0)
        {
            let base_genes: Vec<Gene> = pairs.iter().map(|((_, gene), _)| gene.clone()).collect();
            let base_work: Vec<neoethos_gpu_contracts::device::ScenarioDescriptor> = (0..candidates
                as u64)
                .map(|candidate| {
                    crate::gpu_native::scenario::base_scenario(candidate, candidate, bars)
                })
                .collect();
            let evaluate_at_total_cost = |total_pips: f64| -> Vec<Option<f64>> {
                let mut settings = screen_templates.template(&pairs[0].0.1);
                settings.spread_pips = total_pips;
                settings.commission_per_trade = 0.0;
                settings.session_spread_profile = None;
                let evaluated = match crate::genetic::search_engine::prepare_validation_population(
                    ohlcv,
                    &base_genes,
                    &eval_config_for_signals,
                    &settings,
                ) {
                    Ok(prepared) => {
                        #[cfg(feature = "gpu")]
                        let _gpu_guard =
                            GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                        crate::genetic::search_engine::validation_genes_scenarios(
                            features,
                            ohlcv,
                            &screen_prep,
                            &prepared,
                            &base_work,
                        )
                    }
                    Err(error) => Err(error),
                };
                match evaluated {
                    Ok(metrics) if metrics.len() == candidates => {
                        metrics.iter().map(|m| Some(m[0])).collect()
                    }
                    Ok(metrics) => {
                        tracing::warn!(
                            target: "neoethos_search::cost_model",
                            expected = candidates,
                            returned = metrics.len(),
                            total_pips,
                            "cost-band launch returned the wrong number of rows — this edge is \
                             UNMEASURED for every candidate, so no candidate can be reported \
                             as having survived it"
                        );
                        vec![None; candidates]
                    }
                    Err(error) => {
                        tracing::warn!(
                            target: "neoethos_search::cost_model",
                            error = %error,
                            total_pips,
                            "cost-band launch failed — this edge is UNMEASURED for every \
                             candidate"
                        );
                        vec![None; candidates]
                    }
                }
            };
            cost_band_optimistic = evaluate_at_total_cost(optimistic_pips);
            cost_band_pessimistic = evaluate_at_total_cost(pessimistic_pips);
        }

        // THE PERIOD GRID for the per-trial return matrix, derived once from the
        // span the screen actually simulates. Every trial gets the same columns,
        // so the result is a rectangular (trials × periods) matrix — the shape
        // CSCV/PBO and DSR require. See `trial_returns.rs` for the format and
        // the byte arithmetic.
        let trial_period_keys = crate::trial_returns::month_keys_spanning(
            features.timestamps.first().copied().unwrap_or(0),
            features.timestamps.last().copied().unwrap_or(0),
        );
        if trial_period_keys.is_empty() {
            tracing::warn!(
                target: "neoethos_search::trial_returns",
                first_ts = features.timestamps.first().copied().unwrap_or(0),
                last_ts = features.timestamps.last().copied().unwrap_or(0),
                "no usable period grid for this run — the per-trial return series will be \
                 EMPTY and DSR/PBO stay uncomputable. This is a timestamp problem, not a \
                 strategy result."
            );
        }

        // ── THE TRIAL-RETURNS WRITER, opened BEFORE the screen ────────────
        //
        // Review finding (2026-08-09): the first cut collected every row into
        // RAM and wrote once after the whole parallel screen — Monte-Carlo and
        // sensitivity launches included — had finished. This project's record
        // has exit-137 kills and multi-hour runs that ended with no artifact, so
        // the one file that makes a result falsifiable was being lost in exactly
        // the failure mode that happens. It is now appended chunk by chunk, with
        // the header patched after every flush, so a kill leaves a shorter but
        // valid matrix. Non-fatal either way: a failed write must not lose a
        // discovery result, but it is reported, never swallowed.
        let trial_config_hash = crate::run_identity::config_hash_for(
            config,
            eval_config_for_signals.pip_value_per_lot,
            neoethos_data::current_data_runtime_overrides().normalize_features,
        );
        let mut trial_writer = if config.discovery_ledger_enabled && !trial_period_keys.is_empty() {
            match crate::trial_returns::TrialReturnsWriter::open(
                &config.discovery_ledger_cache_dir,
                &config.evaluation_symbol,
                &config.timeframe_label,
                trial_period_keys.clone(),
                initial_balance,
                candidates,
                trial_config_hash,
            ) {
                Ok(w) => Some(w),
                Err(err) => {
                    tracing::warn!(
                        target: "neoethos_search::trial_returns",
                        error = %err,
                        "could not OPEN the per-trial return series for writing — DSR and PBO \
                         are NOT computable for this run and its result is not falsifiable"
                    );
                    None
                }
            }
        } else {
            if !config.discovery_ledger_enabled {
                tracing::warn!(
                    target: "neoethos_search::trial_returns",
                    trials = candidates,
                    "discovery ledger disabled — the per-trial return series will be computed \
                     and then DISCARDED. DSR and PBO are not computable for this run."
                );
            }
            None
        };

        // Chunked so the writer has something to flush before the run ends. The
        // chunk is an I/O cadence, not a parallelism limit: each chunk is still
        // evaluated across every core, and `position` is the SAME global index
        // the batched Monte-Carlo / sensitivity / cost-band vectors are keyed by.
        const TRIAL_FLUSH_CHUNK: usize = 512;
        let mut screened: Vec<Option<QualityCandidate>> = Vec::with_capacity(candidates);
        let mut trial_rows_total = 0usize;
        let mut pairs_iter = pairs.into_iter();
        let mut chunk_base = 0usize;
        loop {
            let chunk: Vec<((usize, Gene), Vec<i8>)> =
                pairs_iter.by_ref().take(TRIAL_FLUSH_CHUNK).collect();
            if chunk.is_empty() {
                break;
            }
            let chunk_len = chunk.len();
            let base = chunk_base;
            let screened_rows: Vec<(
                Option<QualityCandidate>,
                crate::trial_returns::TrialReturnRow,
            )> = chunk
                .into_par_iter()
                .enumerate()
                .map(|(local_position, ((candidate_idx, gene), sig))| {
                let position = base + local_position;
                let trades = crate::eval::simulate_trades_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &features.timestamps,
                    &sig,
                    &screen_resolver.settings_for_gene(&gene),
                );
                let metrics =
                    analyzer.analyze_strategy(&gene.strategy_id, &trades, initial_balance);

                // Per-session exposure, over EVERY screened candidate. Same
                // bucket boundaries the cost model charges by construction —
                // `SessionSpreadProfile::bucket_index` is the one definition.
                for t in &trades {
                    if t.entry_time <= 0 {
                        continue;
                    }
                    let b = crate::eval::SessionSpreadProfile::bucket_index(t.entry_time);
                    session_trade_counts[b].fetch_add(1, AtomicOrdering::Relaxed);
                    if t.pnl.is_finite() {
                        session_pnl_cents[b]
                            .fetch_add((t.pnl * 100.0).round() as i64, AtomicOrdering::Relaxed);
                    }
                }

                // EVERY trial's per-period return series, captured BEFORE any
                // gate — that is the whole point. A matrix built only from
                // survivors is the selected sample, which is exactly what PBO
                // exists to detect and therefore cannot be computed from.
                let (returns, trades_outside_grid) = crate::trial_returns::period_returns(
                    &trades,
                    &trial_period_keys,
                    initial_balance,
                );
                let trial_row = crate::trial_returns::TrialReturnRow {
                    candidate_index: candidate_idx,
                    strategy_id: gene.strategy_id.clone(),
                    returns,
                    trades_outside_grid,
                };

                let verdict =
                    classify_base_quality(&metrics, &config.target_profile, &config.filtering);
                let opportunistic_quality = match verdict {
                    Ok(opportunistic) => opportunistic,
                    Err(reason) => {
                        rejected_base_quality.fetch_add(1, AtomicOrdering::Relaxed);
                        // One counter per criterion. The match is exhaustive, so
                        // a new `BaseQualityReject` variant cannot be added
                        // without deciding where it is counted.
                        let counter = match reason {
                            BaseQualityReject::AccountWiped => &bq_account_wiped,
                            BaseQualityReject::ProfileNetExpectancy => {
                                &bq_profile_net_expectancy
                            }
                            BaseQualityReject::ProfileExpectancySignificance => {
                                &bq_profile_expectancy_significance
                            }
                            BaseQualityReject::ProfileWinRate => &bq_profile_win_rate,
                            BaseQualityReject::ProfilePayoffRatio => &bq_profile_payoff_ratio,
                            BaseQualityReject::ProfileInMarket => &bq_profile_in_market,
                            BaseQualityReject::OpportunisticLaneClosed => {
                                &bq_opportunistic_lane_closed
                            }
                            BaseQualityReject::PositiveMonths => &bq_positive_months,
                            BaseQualityReject::TradesPerMonth => &bq_trades_per_month,
                            BaseQualityReject::MonthlyReturn => &bq_monthly_return,
                        };
                        counter.fetch_add(1, AtomicOrdering::Relaxed);
                        return (None, trial_row);
                    }
                };

                // Regime-Aware Validation (Idea #3.2)
                let regime_robust = validate_regime_robustness(
                    &trades,
                    features,
                    config.initial_balance,
                    config.max_regime_loss_pct,
                );
                if !regime_robust {
                    rejected_regime.fetch_add(1, AtomicOrdering::Relaxed);
                    return (None, trial_row);
                }

                // Monte Carlo Parameter Perturbation Test.
                // 2026-05-26 operator directive (dual-mode product): runs +
                // min_profitable threshold sourced from typed Settings,
                // previously hardcoded 100/70.
                //
                // AREA 2 / Stage A (2026-06-09): GPU-routed. The serial
                // per-run `signals_for_gene_full` + `simulate_trades_core` is
                // replaced by ONE batched population launch over `mc_runs`
                // perturbed gene clones via `validation_genes_population`
                // (GPU-try, CPU-fallback). The perturbations are applied with a
                // DETERMINISTIC ChaCha8 RNG seeded per (combo, candidate, run),
                // in the EXACT same draw order the serial loop used
                // (long_threshold → short_threshold → each weight → sl_pips? →
                // tp_pips?), so the batched run reproduces the old serial run
                // bit-for-bit and is reproducible CPU↔GPU. The pass test
                // `metrics[run][0] > 0.0` (net_profit) is the trade-pnl sum
                // (fixed-1-lot, `risk_based_sizing == false`), semantically
                // identical to the old `p_trades.iter().map(|t| t.pnl).sum() > 0.0`.
                let Some(profitable_runs) = mc_profitable_runs[position] else {
                    rejected_mc_error.fetch_add(1, AtomicOrdering::Relaxed);
                    return (None, trial_row);
                };

                if (profitable_runs as u32) < config.mc_min_profitable {
                    rejected_mc_floor.fetch_add(1, AtomicOrdering::Relaxed);
                    // Within 10 points of the floor: the candidate is robust on
                    // most perturbations and lost on a minority, which is a very
                    // different signal from one that collapses outright.
                    if profitable_runs as u32 + 10 >= config.mc_min_profitable {
                        mc_near_miss.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    return (None, trial_row);
                }

                // Spread/Slippage Sensitivity Test — wired from Settings
                // 2026-05-26 (dual-mode product).
                let Some(sens_pnl) = sensitivity_net_profit[position] else {
                    // Split from `rejected_mc_error` (2026-08-09): this is the
                    // SENSITIVITY launch failing, not the Monte-Carlo one.
                    rejected_sensitivity_error.fetch_add(1, AtomicOrdering::Relaxed);
                    return (None, trial_row);
                };
                if sens_pnl < 0.0 {
                    rejected_sensitivity.fetch_add(1, AtomicOrdering::Relaxed);
                    return (None, trial_row);
                }

                // THE COST BAND. Deliberately AFTER every gate: it classifies,
                // it does not reject. A candidate that only clears the cheap end
                // of the band is still a survivor of the screen the operator
                // configured — it is just not a result.
                //
                // HOW FAR THE VERDICT TRAVELS, corrected again 2026-08-10 (#71):
                // it rides on the survivor through this function, is counted
                // run-level in `CostBandCensus` and on the funnel's
                // `passed_quality` stage, AND is now carried per strategy out of
                // the export loop on `DiscoveryResult::cost_band_by_strategy`
                // and into `live_portfolio.json` as `cost_band`. Until today the
                // export loop bound it `_cost_band` and dropped it, so a reader
                // of the one artifact a live run consumes could not tell an
                // optimistic-edge-only gene from one robust across the band.
                let cost_band = if config.cost_band_pips.is_some() && !band_discriminates {
                    CostBandVerdict::NotDiscriminating
                } else {
                    CostBandVerdict::from_edges(
                        cost_band_optimistic[position],
                        cost_band_pessimistic[position],
                    )
                };
                match cost_band {
                    CostBandVerdict::NotDiscriminating => {
                        cost_band_not_discriminating.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    CostBandVerdict::OptimisticEdgeOnly => {
                        cost_band_optimistic_only.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    CostBandVerdict::FailsBand => {
                        cost_band_failed.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    CostBandVerdict::Unmeasured => {
                        cost_band_unmeasured.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    CostBandVerdict::SurvivesBand => {
                        cost_band_survived.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                }

                (
                    Some((
                        candidate_idx,
                        gene,
                        sig,
                        metrics,
                        opportunistic_quality,
                        trades,
                        cost_band,
                    )),
                    trial_row,
                )
            })
            .collect();

            // Split the chunk's output: the survivors go on down the funnel, the
            // return series go to disk NOW. Every screened candidate contributed
            // a row, gate or no gate.
            let mut chunk_rows: Vec<crate::trial_returns::TrialReturnRow> =
                Vec::with_capacity(screened_rows.len());
            for (candidate, row) in screened_rows {
                screened.push(candidate);
                chunk_rows.push(row);
            }
            trial_rows_total += chunk_rows.len();
            if let Some(writer) = trial_writer.as_mut() {
                if let Err(err) = writer.append(&chunk_rows) {
                    tracing::warn!(
                        target: "neoethos_search::trial_returns",
                        error = %err,
                        rows = chunk_rows.len(),
                        "FAILED to append a chunk of the per-trial return series — the matrix \
                         is now SHORT by this chunk and any DSR/PBO computed from it is over a \
                         different trial set than the one that ran"
                    );
                }
            }
            chunk_base += chunk_len;
        }

        // ── CLOSE THE PER-TRIAL RETURN SERIES ─────────────────────────────
        //
        // Not the winner's summary — every trial. Without this matrix the
        // Deflated Sharpe Ratio and the Probability of Backtest Overfitting are
        // UNCOMPUTABLE, and no result this project produces is falsifiable.
        // Size, format and the disk-headroom-derived cap are documented in
        // `trial_returns.rs`.
        if let Some(writer) = trial_writer.take() {
            match writer.finish(Utc::now().timestamp_millis()) {
                Ok(manifest) => tracing::info!(
                    target: "neoethos_search::trial_returns",
                    trials_offered = manifest.trials_offered,
                    trials_written = manifest.trials_written,
                    trials_dropped = manifest.trials_dropped,
                    retention = %manifest.retention_rule,
                    periods = manifest.period_count,
                    bytes_written = manifest.bytes_written,
                    bytes_budget = manifest.bytes_budget,
                    disk_available_bytes = manifest.disk_available_bytes,
                    budget_source = %manifest.budget_source,
                    trades_outside_grid = manifest.trades_outside_grid,
                    trades_outside_grid_offered = manifest.trades_outside_grid_offered,
                    config_hash = ?manifest.config_hash,
                    file = %manifest.binary_file,
                    "persisted every trial's per-period return series. NOTE: nothing in this \
                     workspace READS this matrix yet — DSR and PBO are now computable, they are \
                     not yet computed"
                ),
                Err(err) => tracing::warn!(
                    target: "neoethos_search::trial_returns",
                    error = %err,
                    trials = trial_rows_total,
                    "FAILED to close the per-trial return series — DSR and PBO are NOT \
                     computable for this run and its result is not falsifiable"
                ),
            }
        }

        quality_rejects = QualityScreenRejects {
            base_quality: rejected_base_quality.load(AtomicOrdering::Relaxed),
            bq_account_wiped: bq_account_wiped.load(AtomicOrdering::Relaxed),
            bq_profile_net_expectancy: bq_profile_net_expectancy.load(AtomicOrdering::Relaxed),
            bq_profile_expectancy_significance: bq_profile_expectancy_significance
                .load(AtomicOrdering::Relaxed),
            bq_profile_win_rate: bq_profile_win_rate.load(AtomicOrdering::Relaxed),
            bq_profile_payoff_ratio: bq_profile_payoff_ratio.load(AtomicOrdering::Relaxed),
            bq_profile_in_market: bq_profile_in_market.load(AtomicOrdering::Relaxed),
            bq_opportunistic_lane_closed: bq_opportunistic_lane_closed
                .load(AtomicOrdering::Relaxed),
            bq_positive_months: bq_positive_months.load(AtomicOrdering::Relaxed),
            bq_trades_per_month: bq_trades_per_month.load(AtomicOrdering::Relaxed),
            bq_monthly_return: bq_monthly_return.load(AtomicOrdering::Relaxed),
            regime: rejected_regime.load(AtomicOrdering::Relaxed),
            mc_error: rejected_mc_error.load(AtomicOrdering::Relaxed),
            sensitivity_error: rejected_sensitivity_error.load(AtomicOrdering::Relaxed),
            mc_floor: rejected_mc_floor.load(AtomicOrdering::Relaxed),
            mc_near_miss: mc_near_miss.load(AtomicOrdering::Relaxed),
            sensitivity: rejected_sensitivity.load(AtomicOrdering::Relaxed),
        };
        // Arithmetic self-check: the ten criteria must partition the
        // base-quality rejects exactly. If they ever disagree the breakdown is
        // lying, which is worse than not having one — say so loudly rather than
        // publish a number that does not add up.
        let bq_sum: usize = quality_rejects
            .base_quality_breakdown()
            .iter()
            .map(|(_, n)| *n)
            .sum();
        if bq_sum != quality_rejects.base_quality {
            tracing::error!(
                target: "neoethos_search::funnel",
                base_quality_total = quality_rejects.base_quality,
                criteria_sum = bq_sum,
                "the per-criterion base-quality counters do not sum to the total — the \
                 attribution in `classify_base_quality` has a hole"
            );
        }
        tracing::info!(
            target: "neoethos_search::funnel",
            rejected_base_quality = quality_rejects.base_quality,
            rejected_regime = quality_rejects.regime,
            rejected_monte_carlo = quality_rejects.mc_floor,
            monte_carlo_near_miss = quality_rejects.mc_near_miss,
            monte_carlo_floor = config.mc_min_profitable,
            monte_carlo_runs = config.mc_runs,
            rejected_monte_carlo_error = quality_rejects.mc_error,
            rejected_sensitivity_error = quality_rejects.sensitivity_error,
            rejected_spread_sensitivity = quality_rejects.sensitivity,
            "quality screen — which gate rejected the candidates"
        );
        // The ten named criteria, at run end. THIS is the line that says
        // whether "0 survived" was a market verdict or a configuration one: a
        // run in which `base_quality.profile_payoff_ratio` equals the whole
        // candidate count did not measure the market at all. Conversely, a run
        // in which `profile_net_expectancy` is the whole count DID measure the
        // market, and the market said no.
        tracing::info!(
            target: "neoethos_search::funnel",
            account_wiped = quality_rejects.bq_account_wiped,
            profile_net_expectancy = quality_rejects.bq_profile_net_expectancy,
            profile_expectancy_significance =
                quality_rejects.bq_profile_expectancy_significance,
            profile_win_rate = quality_rejects.bq_profile_win_rate,
            profile_payoff_ratio = quality_rejects.bq_profile_payoff_ratio,
            profile_in_market = quality_rejects.bq_profile_in_market,
            opportunistic_lane_closed = quality_rejects.bq_opportunistic_lane_closed,
            positive_months = quality_rejects.bq_positive_months,
            trades_per_month = quality_rejects.bq_trades_per_month,
            monthly_return = quality_rejects.bq_monthly_return,
            net_expectancy_floor = config.target_profile.min_net_expectancy_per_trade,
            expectancy_t_stat_floor = config.target_profile.min_expectancy_t_stat,
            payoff_floor = config.target_profile.min_payoff_ratio,
            min_win_rate_floor = config.target_profile.min_win_rate,
            max_in_market_floor = config.target_profile.max_in_market,
            opportunistic_enabled = config.filtering.opportunistic_enabled,
            use_opportunistic = config.filtering.use_opportunistic_candidates,
            "base-quality screen — which of the TEN criteria rejected the candidates"
        );
        // PER-SESSION EXPOSURE, at run end. When the curve is unset this says
        // how much of the screen's activity — and how much of its money — was
        // priced at a spread nobody measured for that hour.
        {
            let counts: [usize; 3] =
                std::array::from_fn(|i| session_trade_counts[i].load(AtomicOrdering::Relaxed));
            let pnl: [f64; 3] = std::array::from_fn(|i| {
                session_pnl_cents[i].load(AtomicOrdering::Relaxed) as f64 / 100.0
            });
            let total: usize = counts.iter().sum();
            let share = |i: usize| {
                if total > 0 {
                    100.0 * counts[i] as f64 / total as f64
                } else {
                    0.0
                }
            };
            if config.session_spread_pips.is_none() {
                tracing::warn!(
                    target: "neoethos_search::cost_model",
                    asian_trades = counts[0],
                    overlap_trades = counts[1],
                    late_ny_trades = counts[2],
                    asian_pct = share(0),
                    overlap_pct = share(1),
                    late_ny_pct = share(2),
                    asian_pnl = pnl[0],
                    overlap_pnl = pnl[1],
                    late_ny_pnl = pnl[2],
                    flat_spread_pips = config.evaluation_spread_pips,
                    "PER-SESSION EXPOSURE at an UNPRICED spread. Every one of these trades was \
                     charged the same flat spread regardless of the hour. The Asian share is \
                     the part of this run's result that depends on a cost nobody measured. \
                     Fix: average the hourly means already recorded in spread_stats.json over \
                     22-07 / 07-16 / 16-22 UTC into risk.backtest_spread_pips_{{asian,overlap,\
                     late_ny}}."
                );
            } else {
                tracing::info!(
                    target: "neoethos_search::cost_model",
                    asian_trades = counts[0],
                    overlap_trades = counts[1],
                    late_ny_trades = counts[2],
                    asian_pnl = pnl[0],
                    overlap_pnl = pnl[1],
                    late_ny_pnl = pnl[2],
                    "per-session exposure, priced from the configured curve"
                );
            }
        }
        // THE COST BAND, at run end. Read `optimistic_edge_only` before reading
        // any survivor count: those candidates cleared every gate and are still
        // not results. A run whose survivors are mostly in that bucket has found
        // strategies that live inside the uncertainty of its own cost estimate.
        cost_band_census = CostBandCensus {
            survives: cost_band_survived.load(AtomicOrdering::Relaxed),
            optimistic_edge_only: cost_band_optimistic_only.load(AtomicOrdering::Relaxed),
            fails: cost_band_failed.load(AtomicOrdering::Relaxed),
            unmeasured: cost_band_unmeasured.load(AtomicOrdering::Relaxed),
            not_discriminating: cost_band_not_discriminating.load(AtomicOrdering::Relaxed),
        };
        tracing::info!(
            target: "neoethos_search::cost_model",
            baseline_cost_pips,
            band_discriminates,
            not_discriminating = cost_band_census.not_discriminating,
            band_optimistic_pips = cost_band_edges.map(|(lo, _)| lo).unwrap_or(f64::NAN),
            band_pessimistic_pips = cost_band_edges.map(|(_, hi)| hi).unwrap_or(f64::NAN),
            survives_band = cost_band_census.survives,
            optimistic_edge_only = cost_band_census.optimistic_edge_only,
            fails_band = cost_band_census.fails,
            unmeasured = cost_band_census.unmeasured,
            "cost band — every screened candidate re-measured at BOTH edges. \
             `optimistic_edge_only` counts candidates that are profitable ONLY at the cheap \
             end of the cost estimate: those are not results."
        );

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
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "selecting_portfolio",
            detail: "ranking survivors + prop-firm gate + correlation pruning — \
                     silent but active"
                .to_string(),
        });

        let mut screened_genes = Vec::with_capacity(strict_passed.len());
        let mut screened_signals = Vec::with_capacity(strict_passed.len());
        // AUDIT #71 CLOSED HERE (2026-08-10). This loop used to bind the verdict
        // `_cost_band` and drop it, which is where the band stopped travelling:
        // it was measured at both edges and counted run-level, and then the only
        // artifact a live run reads could not say WHICH genes were
        // optimistic-edge-only. The verdict now rides out on
        // `DiscoveryResult::cost_band_by_strategy`, keyed by `strategy_id` —
        // the same key `logged_trades` uses, so no positional assumption is
        // made about a portfolio that is re-ranked and correlation-pruned
        // downstream.
        for (candidate_idx, gene, sig, _, _, _, cost_band) in strict_passed {
            cost_band_by_strategy.push((gene.strategy_id.clone(), cost_band));
            screened_genes.push((candidate_idx, gene));
            screened_signals.push(sig);
        }
        filtered = screened_genes;
        signals_map = screened_signals;
    }
    // The quality screen collapses into a single funnel stage; the per-gate
    // breakdown below is what makes the persisted funnel answer "which test cost
    // us the candidates" without needing the run's logs.
    funnel.record_stage("passed_quality", post_min_trades, filtered.len());
    // Only non-zero reasons are recorded; a skipped screen therefore adds
    // nothing.
    //
    // HOW TO SUM THIS LIST, because three different kinds of entry share it and
    // a naive sum is roughly twice the rejections plus the survivor count:
    //   * entries with NO dot and no prefix are the independent gates. THEY are
    //     the ones that sum to `count_in - count_out`.
    //   * `total.base_quality` is the base-quality AGGREGATE, and the ten
    //     `base_quality.*` entries are its breakdown. Adding either to the gate
    //     sum double-counts; adding both triple-counts.
    //   * `cost_band_*` entries are a CLASSIFICATION of the survivors, not
    //     rejections. Nothing was rejected for them.
    // The prefixes carry that distinction so a reader does not have to know it.
    if quality_rejects.total() > 0 {
        for (reason, count) in [
            ("total.base_quality", quality_rejects.base_quality),
            ("regime_robustness", quality_rejects.regime),
            ("monte_carlo_perturbation", quality_rejects.mc_floor),
            ("monte_carlo_eval_error", quality_rejects.mc_error),
            ("sensitivity_eval_error", quality_rejects.sensitivity_error),
            ("spread_slippage_sensitivity", quality_rejects.sensitivity),
        ] {
            if count > 0 {
                funnel.add_reject_reason("passed_quality", reason, count);
            }
        }
        for (reason, count) in quality_rejects.base_quality_breakdown() {
            if count > 0 {
                funnel.add_reject_reason("passed_quality", reason, count);
            }
        }
    }
    // The cost band is NOT a reject reason — nothing was rejected for it — but it
    // belongs in the persisted funnel next to the rejects, because a reader who
    // has the survivor count and not this classification will over-read the
    // survivor count. Recorded whenever the band was evaluated at all.
    if cost_band_census.total() > 0 {
        for (reason, count) in [
            (
                CostBandVerdict::OptimisticEdgeOnly.label(),
                cost_band_census.optimistic_edge_only,
            ),
            (CostBandVerdict::FailsBand.label(), cost_band_census.fails),
            (
                CostBandVerdict::Unmeasured.label(),
                cost_band_census.unmeasured,
            ),
            (
                CostBandVerdict::NotDiscriminating.label(),
                cost_band_census.not_discriminating,
            ),
            (
                CostBandVerdict::SurvivesBand.label(),
                cost_band_census.survives,
            ),
        ] {
            if count > 0 {
                funnel.add_reject_reason("passed_quality", reason, count);
            }
        }
    }

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
        // agent 2026-06-05 overfitting fix: enforce a hard pass-rate floor
        // ON TOP of the gate's own `pass_rate`. The effective floor is the max
        // of the two, so a candidate must clear FTMO-style rules on at least
        // that share of the random windows. Raising `pf.pass_rate` here means
        // BOTH the diagnostic bucket below and the survival filter
        // (`*rate >= pf.pass_rate`) use the floored threshold consistently.
        //
        // TWO NAMES, ONE DECISION (2026-08-10). `models.prop_firm_min_pass_rate`
        // and `models.discovery_runtime.prop_firm_gate.pass_rate` are collapsed
        // here by `.max()`, and until now no line said so. A silent `.max()`
        // means RAISING EITHER RAISES THE EFFECTIVE FLOOR — so an operator who
        // lowered one of them has not lowered the setting, and the 2026-06-06
        // mandate written into both shipped YAMLs names only the first, which
        // means raising the second silently overrides that disarm.
        //
        // The safer (higher) number wins — that is the existing behaviour and it
        // is the correct one — but the disagreement is now stated with both
        // numbers. One of the two fields is scheduled for deletion; until it
        // goes, this log is the operator's only way to see which one bound.
        let gate_pass_rate = pf.pass_rate;
        let floor_pass_rate = config.prop_firm_min_pass_rate;
        pf.pass_rate = gate_pass_rate.max(floor_pass_rate);
        if (gate_pass_rate - floor_pass_rate).abs() > f64::EPSILON {
            tracing::warn!(
                target: "neoethos_search::config_resolution",
                key_a = "models.discovery_runtime.prop_firm_gate.pass_rate",
                value_a = gate_pass_rate,
                key_b = "models.prop_firm_min_pass_rate",
                value_b = floor_pass_rate,
                effective = pf.pass_rate,
                winner = if gate_pass_rate >= floor_pass_rate {
                    "models.discovery_runtime.prop_firm_gate.pass_rate"
                } else {
                    "models.prop_firm_min_pass_rate"
                },
                "PROP-FIRM PASS RATE IS SET TWICE and the two disagree — the SAFER \
                 (higher) value binds. Lowering only one of them does not lower the \
                 gate."
            );
        } else {
            tracing::info!(
                target: "neoethos_search::config_resolution",
                effective_prop_firm_pass_rate = pf.pass_rate,
                "prop-firm window pass-rate floor (both config keys agree)"
            );
        }
        let candidates_in: Vec<((usize, Gene), Vec<i8>)> =
            filtered.into_iter().zip(signals_map.into_iter()).collect();
        let timestamps_owned = features.timestamps.clone();
        let candidates_in_count = candidates_in.len();
        let pf_pass_rate_floor = pf.pass_rate;
        // ONE resolver + ONE window plan for the whole gate: the window
        // geometry and the window-local adaptive bases are gene-independent,
        // so they are computed once and shared across candidates.
        let pf_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            candidates_in.iter().map(|((_, gene), _)| gene),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        let pf_any_adaptive = candidates_in
            .iter()
            .any(|((_, g), _)| g.stop_vol_mult.is_finite() && g.stop_vol_mult > 0.0);
        let pf_windows =
            plan_prop_firm_windows(ohlcv, &timestamps_owned, &pf, &pf_resolver, pf_any_adaptive)?;
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
                    &pf_resolver,
                    &pf_windows,
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
        funnel.record_stage(
            "passed_prop_firm_window",
            pre_prop_firm,
            next_filtered.len(),
        );
        if dbg_counted_zero > 0 {
            funnel.add_reject_reason("passed_prop_firm_window", "counted_zero", dbg_counted_zero);
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
    // Where the time actually went, printed next to where the candidates went.
    // The two together answer both halves of "why did this take ten hours and
    // produce nothing" without a profiler or a rerun.
    crate::eval_telemetry::log_summary("discovery");
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
    // ── NEVER-ZERO rescue (2026-06-09, operator non-negotiable) ─────────────
    // If the strict funnel (quality + prop-firm + correlation) rejected EVERY
    // candidate, promote the best-found base-filtered genes instead of dying
    // empty. They are correlation-pruned like a real portfolio and their metrics
    // recomputed so portfolio/quality_metrics/signals stay consistent — but the
    // heavy CPCV/walk-forward validation is SKIPPED (running it on genes that
    // already failed the bar would just burn the validation tail). They are
    // emitted honestly flagged `fallback_mode` and forced not-export-ready.
    let mut fallback_mode = false;
    if !portfolio.is_empty() {
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "validation_gates",
            detail: format!(
                "walk-forward + CPCV + PBO + canonical backtests on {} strategies — \
                 the LONGEST silent stage on dense timeframes (can run for hours; \
                 do not stop the run)",
                portfolio.len()
            ),
        });
    }
    let (
        mut validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
        mut per_gene_wf,
    ) = if portfolio.is_empty() && !best_effort_fallback.is_empty() {
        fallback_mode = true;
        let fallback_reason = funnel
            .stages
            .iter()
            .filter(|s| s.count_in > 0)
            .max_by_key(|s| s.rejected)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "strict_gates".to_string());
        let analyzer = quality_analyzer_for_config(config);
        // Even the honesty-flagged fallback genes are DESCRIBED with the stop
        // regime they were scored under — their exported metrics must not come
        // from a strategy they never were.
        let fallback_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            best_effort_fallback.iter().map(|((_, gene), _)| gene),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        for ((_, gene), sig) in best_effort_fallback {
            if portfolio.len() >= FALLBACK_PORTFOLIO_MAX {
                break;
            }
            let mut ok = true;
            for existing in &portfolio_signals {
                if pearson_corr_i8(&sig, existing).abs() >= config.corr_threshold {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let trades = crate::eval::simulate_trades_core(
                &ohlcv.close,
                &ohlcv.high,
                &ohlcv.low,
                &features.timestamps,
                &sig,
                &fallback_resolver.settings_for_gene(&gene),
            );
            quality_metrics.push(analyzer.analyze_strategy(
                &gene.strategy_id,
                &trades,
                config.initial_balance,
            ));
            portfolio_signals.push(sig);
            portfolio.push(gene);
        }
        funnel.record_stage("fallback_best_effort", 0, portfolio.len());
        tracing::warn!(
            target: "neoethos_search::discovery",
            promoted = portfolio.len(),
            reason = %fallback_reason,
            "NEVER-ZERO: the strict funnel emptied the portfolio — promoting the \
             best-found genes (did NOT pass the prop bar) so the run is not empty. \
             Flagged fallback_mode + not-export-ready."
        );
        let mut gates = DiscoveryValidationGates::pending();
        gates.fallback_mode = true;
        gates.fallback_reason = fallback_reason;
        (gates, Vec::new(), Vec::new(), Vec::new())
    } else {
        build_discovery_validation_artifacts(
            &portfolio,
            &portfolio_signals,
            features,
            ohlcv,
            config,
            effective_smc_gate_threshold,
            &pbo_candidates,
            ranked_total,
        )?
    };

    // ── Robustness filters (2026-07-02): permutation + plateau, parallel ────
    // Two per-gene tests on the final portfolio, rayon-parallel across genes,
    // bounded to the most recent ROBUST_WINDOW bars (cheap even on M1):
    //
    // #10 SIGNAL-PERMUTATION (Masters-style): shuffle the gene's signal
    //     sequence — same exposure frequency, destroyed timing. If the REAL
    //     net doesn't beat ≥95% of shuffles, the timing carries no information
    //     (the profit was exposure/luck) → drop the gene.
    // #11 PLATEAU: thresholds perturbed ±15% and re-backtested — a robust edge
    //     sits on a performance PLATEAU (variants keep ≥30% of the real net);
    //     an overfit one falls off a cliff → drop the gene.
    //
    // NEVER-ZERO: applied only when at least ONE gene survives; if they would
    // empty the portfolio, keep it and warn loudly (OOS/PBO/demo gates remain
    // the final authority).
    if !fallback_mode && !portfolio.is_empty() && portfolio_signals.len() == portfolio.len() {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        use rayon::prelude::*;

        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "robustness_filters",
            detail: format!(
                "permutation + plateau tests on {} genes — silent but active",
                portfolio.len()
            ),
        });

        const ROBUST_WINDOW: usize = 150_000;
        const N_PERM: usize = 50;
        const PERM_P_MAX: f64 = 0.05; // real must beat ≥95% of shuffles
        const PLATEAU_MIN_RATIO: f64 = 0.30;

        let n_all = ohlcv.close.len();
        let w0 = n_all.saturating_sub(ROBUST_WINDOW);
        let ts_all: &[i64] = ohlcv.timestamp.as_deref().unwrap_or(&[]);
        let ts_win: &[i64] = if ts_all.len() == n_all {
            &ts_all[w0..]
        } else {
            &[]
        };
        let eval_cfg_rb = config.evaluation_config_with_smc_gate(
            ohlcv.close.last().copied(),
            effective_smc_gate_threshold,
        );
        // ONE resolver over the trailing robustness window — `net_of` below
        // simulates `[w0..]` slices, so the adaptive base is built on exactly
        // that slice and each gene is permutation/plateau-tested under the
        // stop regime it was scored under.
        let robust_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            portfolio.iter(),
            &ohlcv.high[w0..],
            &ohlcv.low[w0..],
            &ohlcv.close[w0..],
        )?;

        let verdicts: Vec<(bool, String)> = portfolio
            .par_iter()
            .enumerate()
            .map(|(gi, gene)| {
                let settings = robust_resolver.settings_for_gene(gene);
                let sig_full = &portfolio_signals[gi];
                if sig_full.len() != n_all {
                    return (true, "skipped (signal length mismatch)".to_string());
                }
                let sig_win = &sig_full[w0..];
                let net_of = |sigs: &[i8]| -> f64 {
                    simulate_trades_core(
                        &ohlcv.close[w0..],
                        &ohlcv.high[w0..],
                        &ohlcv.low[w0..],
                        ts_win,
                        sigs,
                        &settings,
                    )
                    .iter()
                    .map(|t| t.pnl)
                    .sum()
                };
                let real_net = net_of(sig_win);
                let signal_bars = sig_win.iter().filter(|s| **s != 0).count();
                if real_net <= 0.0 || signal_bars < 30 {
                    // Too little recent evidence to test against — pass through;
                    // the OOS/PBO gates already judged the full history.
                    return (true, "skipped (thin recent window)".to_string());
                }

                // #10 permutation p-value — deterministic seed per gene.
                let seed =
                    0x4E45_4F45_5448_4F53u64 ^ (gi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut beats = 0usize;
                let mut shuffled: Vec<i8> = sig_win.to_vec();
                for _ in 0..N_PERM {
                    shuffled.shuffle(&mut rng);
                    if net_of(&shuffled) >= real_net {
                        beats += 1;
                    }
                }
                let p_value = beats as f64 / N_PERM as f64;
                if p_value >= PERM_P_MAX {
                    return (
                        false,
                        format!(
                            "permutation FAIL (p={p_value:.2}: random timing matches the real net)"
                        ),
                    );
                }

                // #11 plateau: ±15% threshold perturbations must keep ≥30% net.
                for factor in [0.85_f32, 1.15] {
                    let mut variant = gene.clone();
                    variant.long_threshold *= factor;
                    variant.short_threshold *= factor;
                    let sig_v = signals_for_gene_full(features, ohlcv, &variant, &eval_cfg_rb);
                    if sig_v.len() != n_all {
                        continue;
                    }
                    let net_v = net_of(&sig_v[w0..]);
                    if net_v < PLATEAU_MIN_RATIO * real_net {
                        return (
                            false,
                            format!(
                                "plateau FAIL (thresholds ×{factor:.2} → net {net_v:.0} \
                                 vs real {real_net:.0} — cliff, not plateau)"
                            ),
                        );
                    }
                }
                (true, format!("robust (p={p_value:.2}, plateau ok)"))
            })
            .collect();

        for (gi, (kept, why)) in verdicts.iter().enumerate() {
            tracing::info!(
                target: "neoethos_search::discovery",
                gene = %portfolio[gi].strategy_id,
                kept, reason = %why,
                "robustness filter verdict"
            );
        }
        let keep: Vec<bool> = verdicts.into_iter().map(|(k, _)| k).collect();
        if keep.iter().any(|k| *k) && !keep.iter().all(|k| *k) {
            let before = portfolio.len();
            let mut i = 0usize;
            portfolio.retain(|_| {
                let k = keep[i];
                i += 1;
                k
            });
            let mut i = 0usize;
            portfolio_signals.retain(|_| {
                let k = keep[i];
                i += 1;
                k
            });
            if per_gene_wf.len() == keep.len() {
                let mut i = 0usize;
                per_gene_wf.retain(|_| {
                    let k = keep[i];
                    i += 1;
                    k
                });
            }
            tracing::info!(
                target: "neoethos_search::discovery",
                kept = portfolio.len(),
                dropped = before - portfolio.len(),
                "robustness filters: exporting only the permutation+plateau survivors"
            );
        } else if !keep.iter().any(|k| *k) {
            tracing::warn!(
                target: "neoethos_search::discovery",
                "robustness filters would drop EVERY portfolio gene — keeping the \
                 portfolio (never-zero) but treat these exports with suspicion"
            );
        }
    }

    // Risky-mode walk-forward FILTER (operator 2026-06-28). The portfolio-level
    // gate was all-or-nothing: ONE marginal gene made `walkforward_passed=false`
    // → the whole portfolio was rejected → 0 exports across the sweep. Instead,
    // keep only the genes that individually clear the risky walk-forward bar, so
    // walk-forward acts as SELECTION pressure and we export the robust SUBSET
    // (never the overfit ones). Only `portfolio` is filtered — `quality_metrics`
    // is the full screened-candidate record (a superset, searched by id
    // downstream), not positionally aligned to `portfolio`. The clean final OOS
    // read remains the (upcoming) sealed lockbox + the live demo-forward gate.
    if matches!(config.mode, DiscoveryMode::Risky)
        && !fallback_mode
        && per_gene_wf.len() == portfolio.len()
        && per_gene_wf.iter().any(|&p| p)
        && !per_gene_wf.iter().all(|&p| p)
    {
        let keep = per_gene_wf.clone();
        let before = portfolio.len();
        let mut i = 0usize;
        portfolio.retain(|_| {
            let k = keep[i];
            i += 1;
            k
        });
        // Surviving genes each passed walk-forward → the portfolio now does too.
        validation_gates.walkforward_passed = !portfolio.is_empty();
        tracing::info!(
            target: "neoethos_search::discovery",
            kept = portfolio.len(),
            dropped = before - portfolio.len(),
            "risky walk-forward filter: exported only the WF-passing gene subset"
        );
    }

    if let Some(pf) = config.prop_firm_gate.as_ref() {
        // agent 2026-06-05 overfitting fix: the prop-firm window gate alone let
        // in-sample-overfit portfolios export (walk-forward was informational).
        // When `require_walkforward_for_export` is set (default), the portfolio
        // must ALSO clear the walk-forward gate to be window-passed — so
        // `is_portfolio_export_ready()` (which keys off `prop_firm_window_passed`)
        // now demands genuine out-of-sample robustness. When the flag is false
        // the AND collapses to the previous `!portfolio.is_empty()` behaviour.
        let window_passed = !portfolio.is_empty();
        validation_gates.prop_firm_window_passed = if config.require_walkforward_for_export {
            window_passed && validation_gates.walkforward_passed
        } else {
            window_passed
        };
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
    if fallback_mode {
        // Honest: best-effort fallback genes did NOT pass the prop bar, so they
        // must never read as export-ready downstream (the autonomous trader keys
        // off `is_portfolio_export_ready()` / `prop_firm_window_passed`).
        validation_gates.prop_firm_window_passed = false;
    }
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
    let outcome = if fallback_mode {
        "fallback_best_effort"
    } else if portfolio.is_empty() {
        "no_candidates"
    } else if export_ready > 0 {
        "exported"
    } else {
        "failed"
    };
    funnel.finalize(outcome);

    // GPU-vs-CPU proof on the same surface as the goal report: what fraction of
    // population-eval WALL time ran on the card, and how many times it fell back
    // to the CPU while a card was present. Prints even when empty, so a real GPU
    // run and a silent-CPU run can never again produce identical end-of-run
    // output — the exact indistinguishability that hid the starved card.
    crate::eval_telemetry::device_summary();

    // The batch census, cumulative across every cycle this process has run.
    // Printed at the END OF EVERY CYCLE rather than only by the streaming loop,
    // so a rejection can never be lost by a caller that drives the cycle
    // directly. On a non-streaming run it prints one line for one batch, which
    // is the honest description of what a non-streaming run is.
    log_batch_rejection_summary("discovery_cycle");

    // Honest goal projection (Risky only): "reach the target, when, at what
    // risk?" from the selected portfolio's REAL per-trade R-multiples. Logged
    // here, before the result is moved, while config and the trades coexist.
    log_goal_report(config, &portfolio, &quality_metrics, &logged_trades);

    Ok(DiscoveryResult {
        cost_band_by_strategy,
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

        effective_smc_gate_threshold,
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
/// Midrank of every possible `i8` value, from ONE pass over the slice.
///
/// A signal is `i8`, so it has at most 256 distinct values (in practice 3:
/// −1/0/+1) and an element's midrank depends ONLY on its value — never on
/// its position. So a single 256-bucket histogram yields every rank:
/// `rank(v) = (#elements < v) + (count(v) + 1) / 2` — the same tie-corrected
/// midrank formula as before, just computed once per value instead of once
/// per element.
///
/// Perf (2026-07-20): the previous implementation rescanned the WHOLE slice
/// twice for every element, making Spearman O(n²). On a 1.36M-bar M3 signal
/// that is ~3.7e12 element comparisons per array — ~30 minutes per gene PAIR
/// single-threaded, so the portfolio's correlation pruning (O(genes²) pairs)
/// could not finish in days. It presented as a discovery run frozen at
/// "quality_screen 95.5%" burning exactly one core. This is O(n).
fn i8_midranks(vals: &[i8]) -> [f64; 256] {
    let mut counts = [0usize; 256];
    for &v in vals {
        counts[(v as i16 + 128) as usize] += 1;
    }
    let mut ranks = [0.0_f64; 256];
    let mut before = 0usize;
    for (bucket, &count) in counts.iter().enumerate() {
        if count > 0 {
            ranks[bucket] = before as f64 + (count as f64 + 1.0) / 2.0;
            before += count;
        }
    }
    ranks
}

#[inline]
fn midrank_of(ranks: &[f64; 256], v: i8) -> f64 {
    ranks[(v as i16 + 128) as usize]
}

fn spearman_corr_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ranks_a = i8_midranks(a);
    let ranks_b = i8_midranks(b);
    // Means over the SAME element order as before, so the floating-point
    // result is identical to the old per-element implementation.
    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    for i in 0..n {
        sum_a += midrank_of(&ranks_a, a[i]);
        sum_b += midrank_of(&ranks_b, b[i]);
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;
    let mut num = 0.0_f64;
    let mut denom_a = 0.0_f64;
    let mut denom_b = 0.0_f64;
    for i in 0..n {
        let da = midrank_of(&ranks_a, a[i]) - mean_a;
        let db = midrank_of(&ranks_b, b[i]) - mean_b;
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

/// Unicode sparkline of an equity curve (operator 2026-06-06): see the shape
/// (start → trough → end) at a glance in the log, not just numbers.
fn equity_sparkline(curve: &[f64], width: usize) -> String {
    if curve.len() < 2 {
        return String::new();
    }
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let lo = curve.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = curve.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (hi - lo).max(1e-9);
    let n = curve.len();
    let cols = width.max(1).min(n);
    let mut s = String::with_capacity(cols);
    for c in 0..cols {
        let idx = (c * (n - 1)) / cols;
        let v = curve[idx];
        let b = (((v - lo) / range) * (BLOCKS.len() - 1) as f64).round() as usize;
        s.push(BLOCKS[b.min(BLOCKS.len() - 1)]);
    }
    s
}

pub fn save_quality_report_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    // Operator observability (2026-06-06): surface the rich per-candidate metrics
    // that were previously written ONLY to <stem>.quality.json (invisible at
    // runtime — "we don't see how many trades / how often each strategy does").
    // Flags likely-overfit candidates (in-sample Sharpe > 3.0) at a glance — the
    // operator's "Sharpe 3 = wrong / overfit" rule.
    if !result.quality_metrics.is_empty() {
        // Which of these rows actually made it out. `quality_metrics` is the
        // full screened-candidate record — a superset of the portfolio — so
        // without this the question "what do the ones that survived earn?"
        // cannot be answered from the log at all: the exported rows and the
        // rejected ones are printed identically.
        let exported: std::collections::HashSet<&str> = result
            .portfolio
            .iter()
            .map(|gene| gene.strategy_id.as_str())
            .collect();
        tracing::info!(
            target: "neoethos_search::discovery",
            count = result.quality_metrics.len(),
            exported = exported.len(),
            "CANDIDATE METRICS — id | trades(/mo,/day) | hold | WR | PF | Sharpe | maxDD | verdict"
        );
        for q in &result.quality_metrics {
            let flag = if q.sharpe_ratio > 3.0 {
                " [!] OVERFIT?"
            } else {
                ""
            };
            let export_tag = if exported.contains(q.strategy_id.as_str()) {
                "[EXPORTED] "
            } else {
                ""
            };
            tracing::info!(
                target: "neoethos_search::discovery",
                "  {}{} | {} trades ({:.1}/mo, {:.2}/day) | {:.1}h hold | WR {:.0}% | PF {:.2} | Sharpe {:.2}{} | maxDD {:.1}% | {}",
                export_tag,
                q.strategy_id,
                q.total_trades,
                q.trades_per_month,
                q.trades_per_month / 21.0,
                q.avg_trade_duration_hours,
                q.win_rate * 100.0,
                q.profit_factor,
                q.sharpe_ratio,
                flag,
                q.max_drawdown_pct * 100.0,
                q.recommendation,
            );
            // Pro money-view (2026-06-06): "how much € in how long, with what curve" —
            // ratios alone (Sharpe 7) hide that a strategy made ~5% over 9 months (useless).
            let curve_min = q.equity_curve.iter().cloned().fold(f64::INFINITY, f64::min);
            tracing::info!(
                target: "neoethos_search::discovery",
                "      money: EUR {:.0} -> {:.0} (net {:+.0}, {:.1} months, {:.2}%/mo) | recovery {:.2} | curve min EUR {:.0} | maxDD EUR {:.0}",
                q.initial_capital,
                q.final_balance,
                q.net_profit,
                q.period_days / 30.44,
                if q.period_days > 0.0 { q.total_return_pct * 100.0 / (q.period_days / 30.44) } else { 0.0 },
                q.recovery_factor,
                if curve_min.is_finite() { curve_min } else { q.initial_capital },
                q.max_drawdown_money,
            );
            if q.equity_curve.len() >= 2 {
                tracing::info!(
                    target: "neoethos_search::discovery",
                    "      curve: {}",
                    equity_sparkline(&q.equity_curve, 50)
                );
            }
            tracing::info!(
                target: "neoethos_search::discovery",
                "      per-trade: MFE EUR {:.0} | MAE EUR {:.0} | avg R {:+.2} | MFE-capture {:.0}%",
                q.avg_mfe,
                q.avg_mae,
                q.avg_r_multiple,
                q.mfe_capture_ratio * 100.0,
            );
        }
        log_exported_money_summary(result, &exported);
    }
    write_json_atomic(path, &result.quality_metrics)
}

/// What the strategies that survived validation actually earn.
///
/// The per-candidate rows answer this one strategy at a time, across a set that
/// is mostly rejects — so the question needed reading dozens of lines while
/// knowing which ids were exported. This answers it directly, over the exported
/// subset only.
///
/// The figures are deliberately NOT summed into a single euro total. Each
/// strategy is analysed alone on the full starting balance, so adding thirty
/// net-profit figures would describe thirty separate accounts rather than one
/// portfolio, overstating the result by roughly the portfolio size. Returns are
/// therefore reported as a distribution, and only trade frequency is aggregated,
/// because the strategies really do trade in parallel on one account.
/// Honest goal projection for a Risky run — the "no fake results" output.
///
/// Monte-Carlos the SELECTED portfolio's real, cost-charged per-trade
/// R-multiples (Decision D) across a risk sweep and logs P(reach target),
/// P(ruin), median/mean terminal, median time-to-target, and the risk that
/// maximises P(reach). No-op for non-Risky modes, where the target/horizon are
/// meaningless. See [`crate::goal_report`].
fn log_goal_report(
    config: &DiscoveryConfig,
    portfolio: &[Gene],
    quality_metrics: &[StrategyMetrics],
    logged_trades: &[LoggedStrategyTrades],
) {
    if !matches!(config.mode, DiscoveryMode::Risky) {
        return;
    }
    let ids: std::collections::HashSet<&str> =
        portfolio.iter().map(|g| g.strategy_id.as_str()).collect();
    // Pool the exported strategies' real per-trade R-multiples (size-independent,
    // net of the broker costs Decision D charges).
    let r_multiples: Vec<f64> = logged_trades
        .iter()
        .filter(|lt| ids.contains(lt.strategy_id.as_str()))
        .flat_map(|lt| lt.trades.iter().map(|t| t.r_multiple))
        .filter(|r| r.is_finite())
        .collect();
    // Combined cadence: the strategies hold positions at the same time on the
    // one account, so their per-day trade rates add.
    let trades_per_day: f64 = quality_metrics
        .iter()
        .filter(|q| ids.contains(q.strategy_id.as_str()))
        .map(|q| q.trades_per_month / 21.0)
        .sum();
    if r_multiples.is_empty() || trades_per_day <= 0.0 {
        tracing::info!(
            target: "neoethos_search::discovery",
            "GOAL REPORT — skipped: the Risky portfolio produced no usable trades to project."
        );
        return;
    }
    let report = crate::goal_report::build_report(
        &r_multiples,
        config.risky_start_balance,
        config.risky_target_balance,
        config.risky_horizon_days,
        trades_per_day,
        crate::goal_report::DEFAULT_RISK_LEVELS,
        // Fixed seed: the projection is reproducible for the same portfolio.
        0x00C0_FFEE_u64,
    );
    for line in report.render().lines() {
        tracing::info!(target: "neoethos_search::discovery", "{line}");
    }
}

fn log_exported_money_summary(
    result: &DiscoveryResult,
    exported: &std::collections::HashSet<&str>,
) {
    let survivors: Vec<&StrategyMetrics> = result
        .quality_metrics
        .iter()
        .filter(|q| exported.contains(q.strategy_id.as_str()))
        .collect();
    if survivors.is_empty() {
        // An empty portfolio is already reported by the funnel, so this is not
        // worth a warning — but saying it beats printing a table of zeros that
        // reads like a measurement.
        tracing::info!(
            target: "neoethos_search::discovery",
            "EXPORTED MONEY VIEW — nothing was exported, so there is nothing to earn"
        );
        return;
    }

    let mut returns_pct: Vec<f64> = survivors
        .iter()
        .map(|q| q.total_return_pct * 100.0)
        .collect();
    returns_pct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_return = returns_pct[returns_pct.len() / 2];
    let profitable = returns_pct.iter().filter(|r| **r > 0.0).count();
    let n = survivors.len() as f64;
    let mean = |f: &dyn Fn(&StrategyMetrics) -> f64| -> f64 {
        survivors.iter().map(|q| f(q)).sum::<f64>() / n
    };

    // Additive across strategies, unlike the euro figures: they hold positions
    // at the same time on the one account.
    let trades_per_day: f64 = survivors.iter().map(|q| q.trades_per_month / 21.0).sum();
    let worst_dd = survivors
        .iter()
        .map(|q| q.max_drawdown_pct)
        .fold(0.0_f64, f64::max);
    let months = mean(&|q| q.period_days) / 30.44;

    tracing::info!(
        target: "neoethos_search::discovery",
        "EXPORTED MONEY VIEW — {} strategies survived validation, {} profitable, over {:.1} months",
        survivors.len(),
        profitable,
        months,
    );
    tracing::info!(
        target: "neoethos_search::discovery",
        "  per strategy on EUR {:.0} alone: return {:+.1}% worst / {:+.1}% median / {:+.1}% best \
         (median net EUR {:+.0}) — NOT additive, one account splits capital across all {}",
        mean(&|q| q.initial_capital),
        returns_pct[0],
        median_return,
        returns_pct[returns_pct.len() - 1],
        mean(&|q| q.net_profit),
        survivors.len(),
    );
    tracing::info!(
        target: "neoethos_search::discovery",
        "  portfolio activity: {:.2} trades/day combined | mean hold {:.1}h | worst maxDD {:.1}%",
        trades_per_day,
        mean(&|q| q.avg_trade_duration_hours),
        worst_dd * 100.0,
    );
    // The "money that disappears": the trade reached this much profit and gave
    // most of it back. Averaged over survivors it says whether the exits, not
    // the entries, are where the money is being left behind.
    tracing::info!(
        target: "neoethos_search::discovery",
        "  exit quality: mean MFE EUR {:.0} per trade, {:.0}% captured | mean R {:+.2}",
        mean(&|q| q.avg_mfe),
        mean(&|q| q.mfe_capture_ratio) * 100.0,
        mean(&|q| q.avg_r_multiple),
    );
    // One trade a day is the operator's stated floor for risky mode: below it
    // the account cannot compound often enough to reach the target, however
    // good each individual trade is.
    if trades_per_day < 1.0 {
        tracing::warn!(
            target: "neoethos_search::discovery",
            trades_per_day = format!("{trades_per_day:.2}"),
            "the exported portfolio trades less than once a day — too few \
             compounding events for the risky-mode target"
        );
    }
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
        /// Held-out out-of-sample verdict — the single most decision-relevant
        /// promotion signal. Surfaced here (per-portfolio) so operators AND the
        /// model-training corpus can weight each strategy by its honest OOS
        /// reliability instead of in-sample metrics. `forward_test_passed` /
        /// `prop_firm_passed` are `None` when no held-out artifact was produced
        /// (e.g. the CLI full-data path or too short a tail), `Some(false)` when
        /// the strategy lost money / breached prop rules on the unseen tail.
        out_of_sample: OutOfSampleVerdict,
    }
    #[derive(Serialize)]
    struct OutOfSampleVerdict {
        forward_test_passed: Option<bool>,
        prop_firm_passed: Option<bool>,
        walkforward_passed: bool,
        cpcv_passed: bool,
    }
    let hashes = discovery_per_kind_evidence_hashes(result)?;
    let evidence = live_validation_evidence_from_discovery(result);
    let summary = PromotionSummary {
        producer_side_complete: hashes.all_producer_kinds_present(),
        check_summary: hashes.check_summary(),
        validation_evidence_complete: hashes.all_present(),
        validation_evidence_missing_kinds: hashes.missing_kinds(),
        validation_evidence_hashes: &hashes,
        determinism_policy: crate::genetic::current_determinism_policy(),
        out_of_sample: OutOfSampleVerdict {
            forward_test_passed: evidence.forward_test_passed,
            prop_firm_passed: evidence.prop_firm_passed,
            walkforward_passed: evidence.walkforward_passed,
            cpcv_passed: evidence.cpcv_passed,
        },
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
    let resolved_max_rows = row_cap_for_config(config);
    // SLICE 5 COMPLETENESS GATE (2026-08-08): destructure the config WITHOUT
    // `..`. Every field that can change what the search selects must appear
    // in the profile; a new `DiscoveryConfig` field therefore FAILS TO
    // COMPILE here until someone decides where it is recorded (or explicitly
    // binds it to `_name` with a written justification). Sixteen months of
    // "two runs differ and nobody can say why" is the cost of the old `..`.
    let DiscoveryConfig {
        timeframe_label,
        evaluation_symbol,
        evaluation_account_currency,
        evaluation_spread_pips,
        evaluation_commission_per_trade,
        session_spread_pips,
        cost_band_pips,
        swap_long_pips_per_day,
        swap_short_pips_per_day,
        kill_zones_enabled,
        population,
        generations,
        max_indicators,
        candidate_count,
        portfolio_size,
        // Raw knob; the profile records the RESOLVED row cap
        // (`row_cap_for_config`, which folds in `max_rows_by_timeframe`)
        // as `max_rows`, plus the per-timeframe table itself below.
        max_rows: _,
        max_rows_by_timeframe,
        max_hours,
        corr_threshold,
        min_trades_per_day,
        target_profile,
        walkforward_splits,
        embargo_minutes,
        enable_cpcv,
        cpcv_n_splits,
        cpcv_n_test_groups,
        cpcv_embargo_pct,
        cpcv_purge_pct,
        cpcv_min_phi,
        cpcv_max_rows,
        max_pbo,
        filtering,
        initial_balance,
        risk_per_trade_min,
        risk_per_trade_max,
        risky_risk_band,
        prop_firm_risk_band,
        max_regime_loss_pct,
        higher_timeframes,
        runtime_overrides,
        prop_firm_gate,
        mc_runs,
        mc_min_profitable,
        sensitivity_spread_pips,
        sensitivity_commission_per_lot,
        adaptive_thresholds,
        mode,
        prop_firm_gate_params,
        risky_start_balance,
        risky_target_balance,
        risky_horizon_days,
        require_walkforward_for_export,
        prop_firm_min_pass_rate,
        discovery_ledger_enabled,
        discovery_ledger_cache_dir,
        discovery_ledger_archive_top_n,
        population_auto,
    } = config;
    // Same completeness gate for the filter floors: `FilteringConfig` grew
    // `anomaly_guard` / `elite_mode` without the profile noticing — never
    // again.
    let crate::genetic::FilteringConfig {
        max_dd,
        min_profit,
        min_trades,
        min_sharpe,
        min_win_rate,
        min_profit_factor,
        min_positive_months,
        min_trades_per_month,
        min_monthly_return_pct,
        log_trades,
        trade_log_max,
        opportunistic_enabled,
        use_opportunistic_candidates,
        opportunistic_min_positive_months,
        opportunistic_min_trades_per_month,
        opportunistic_min_trade_return_pct,
        opportunistic_max_dd,
        anomaly_guard,
        elite_mode,
    } = filtering;
    // And for the runtime overrides: `stage1_window` + `min_history_years`
    // were silently absent from the profile before slice 5.
    let DiscoveryRuntimeOverrides {
        prefilter_top_k,
        prefilter_insample_frac: _, // recorded resolved below
        prefilter_min_per_timeframe,
        funnel_stage1_pct: _, // recorded resolved below
        stage1_window,
        min_history_years,
    } = runtime_overrides;
    DiscoveryRunProfile {
        population_eval_engines: crate::engine_identity::observed_population_engines(),
        timeframe_label: timeframe_label.clone(),
        population: *population,
        population_auto: *population_auto,
        generations: *generations,
        max_indicators: *max_indicators,
        candidate_count_target: *candidate_count,
        portfolio_size_target: *portfolio_size,
        max_rows: resolved_max_rows,
        max_runtime_hours: *max_hours,
        corr_threshold: *corr_threshold,
        min_trades_per_day: *min_trades_per_day,
        walkforward_splits: *walkforward_splits,
        embargo_minutes: *embargo_minutes,
        enable_cpcv: *enable_cpcv,
        cpcv_n_splits: *cpcv_n_splits,
        cpcv_n_test_groups: *cpcv_n_test_groups,
        cpcv_embargo_pct: *cpcv_embargo_pct,
        cpcv_purge_pct: *cpcv_purge_pct,
        cpcv_min_phi: *cpcv_min_phi,
        filters: DiscoveryFilterProfile {
            max_dd: *max_dd,
            min_profit: *min_profit,
            min_trades: *min_trades,
            min_sharpe: *min_sharpe,
            min_win_rate: *min_win_rate,
            min_profit_factor: *min_profit_factor,
            min_positive_months: *min_positive_months,
            min_trades_per_month: *min_trades_per_month,
            min_monthly_return_pct: *min_monthly_return_pct,
            opportunistic_enabled: *use_opportunistic_candidates && *opportunistic_enabled,
            opportunistic_min_positive_months: *opportunistic_min_positive_months,
            opportunistic_min_trades_per_month: *opportunistic_min_trades_per_month,
            opportunistic_min_trade_return_pct: *opportunistic_min_trade_return_pct,
            opportunistic_max_dd: *opportunistic_max_dd,
            log_trades: *log_trades,
            trade_log_max: *trade_log_max,
            use_opportunistic_candidates_raw: *use_opportunistic_candidates,
            opportunistic_enabled_raw: *opportunistic_enabled,
            anomaly_guard: *anomaly_guard,
            elite_mode: *elite_mode,
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
        prefilter_top_k: *prefilter_top_k,
        prefilter_insample_frac: runtime_overrides.resolved_prefilter_insample_frac(),
        prefilter_min_per_timeframe: *prefilter_min_per_timeframe,
        funnel_stage1_pct: runtime_overrides.resolved_funnel_stage1_pct(),
        validation_evidence_hashes: validation_evidence_hashes.clone(),
        validation_evidence_complete: validation_evidence_hashes.all_present(),
        validation_evidence_missing_kinds: validation_evidence_hashes
            .missing_kinds()
            .into_iter()
            .map(str::to_string)
            .collect(),
        determinism_policy: crate::genetic::current_determinism_policy(),
        evaluation_symbol: evaluation_symbol.clone(),
        evaluation_account_currency: evaluation_account_currency.clone(),
        evaluation_spread_pips: *evaluation_spread_pips,
        evaluation_commission_per_trade: *evaluation_commission_per_trade,
        session_spread_pips: *session_spread_pips,
        cost_band_pips: *cost_band_pips,
        swap_long_pips_per_day: *swap_long_pips_per_day,
        swap_short_pips_per_day: *swap_short_pips_per_day,
        kill_zones_enabled: *kill_zones_enabled,
        mode: *mode,
        target_profile: *target_profile,
        max_pbo: *max_pbo,
        cpcv_max_rows: *cpcv_max_rows,
        prop_firm_gate: prop_firm_gate.clone(),
        prop_firm_gate_params: prop_firm_gate_params.clone(),
        require_walkforward_for_export: *require_walkforward_for_export,
        prop_firm_min_pass_rate: *prop_firm_min_pass_rate,
        initial_balance: *initial_balance,
        risk_per_trade_min: *risk_per_trade_min,
        risk_per_trade_max: *risk_per_trade_max,
        risky_risk_band: *risky_risk_band,
        prop_firm_risk_band: *prop_firm_risk_band,
        max_regime_loss_pct: *max_regime_loss_pct,
        mc_runs: *mc_runs,
        mc_min_profitable: *mc_min_profitable,
        sensitivity_spread_pips: *sensitivity_spread_pips,
        sensitivity_commission_per_lot: *sensitivity_commission_per_lot,
        adaptive_thresholds: *adaptive_thresholds,
        higher_timeframes: higher_timeframes.clone(),
        max_rows_by_timeframe: max_rows_by_timeframe
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
        stage1_window: *stage1_window,
        min_history_years: *min_history_years,
        risky_start_balance: *risky_start_balance,
        risky_target_balance: *risky_target_balance,
        risky_horizon_days: *risky_horizon_days,
        discovery_ledger_enabled: *discovery_ledger_enabled,
        discovery_ledger_cache_dir: discovery_ledger_cache_dir.clone(),
        discovery_ledger_archive_top_n: *discovery_ledger_archive_top_n,
        execution: crate::execution_profile::ExecutionEnvironmentProfile::capture(),
    }
}

pub fn save_discovery_profile_json(
    path: impl AsRef<Path>,
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> Result<()> {
    write_json_atomic(path, &build_discovery_profile(config, result))
}

/// THE Monte-Carlo perturbation the quality screen measures. Host lane, ChaCha8.
///
/// Extracted from the screen so it can be PINNED. Turning the device
/// perturbation on changes which generator draws the numbers — the device cannot
/// walk a ChaCha8 stream, see `gpu_native::scenario` — and the only defence
/// against that difference being made silently is a test that fails when this
/// function's output moves. A test that re-implemented the loop would pin a
/// copy, not the code, so the screen and the test call the same function.
///
/// The draw ORDER is the contract: long_threshold, short_threshold, each weight
/// ascending, then sl_pips and tp_pips — each only if finite and positive, so a
/// gene with no fixed stop does not acquire one by being multiplied. Reordering
/// these, or drawing for a skipped stop, changes every subsequent number.
fn host_monte_carlo_perturbation(
    gene: &Gene,
    combo_seed: u64,
    candidate_idx: usize,
    run_idx: u64,
) -> Gene {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
        combo_seed ^ ((candidate_idx as u64) << 20) ^ run_idx,
    );
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
    perturbed
}

#[cfg(test)]
mod monte_carlo_reference_tests {
    use super::*;

    fn seed_gene() -> Gene {
        Gene {
            weights: vec![1.0, -0.5, 0.25],
            long_threshold: 0.60,
            short_threshold: -0.40,
            sl_pips: 20.0,
            tp_pips: 40.0,
            ..Gene::default()
        }
    }

    /// THE REFERENCE, PINNED TO EXACT BITS.
    ///
    /// The Monte-Carlo screen is the gate 7 792 of 7 793 candidates die at in a
    /// measured run, so what it measures IS the search. Two things could change
    /// it without anyone noticing: a `rand` upgrade that alters
    /// `random_range`'s rejection sampling, and turning the device perturbation
    /// on, which uses a counter-based generator that cannot reproduce ChaCha8
    /// and is not trying to.
    ///
    /// Neither is forbidden. Both must be DELIBERATE, and this is what makes
    /// them so: the numbers below were produced by this function and any change
    /// to what the default screen measures now arrives as a failing test with
    /// the old and new values printed side by side.
    #[test]
    fn the_host_monte_carlo_draw_order_is_pinned() {
        let gene = seed_gene();
        let perturbed = host_monte_carlo_perturbation(&gene, 0xC0FFEE, 3, 7);

        // Determinism first: same inputs, same gene, every time and from any
        // thread. This is what makes parallel construction of the clone array
        // bit-identical to the serial construction it replaced.
        assert_eq!(
            perturbed,
            host_monte_carlo_perturbation(&gene, 0xC0FFEE, 3, 7)
        );

        // Then the exact values. Written as bit patterns so a printed decimal
        // that happens to round the same cannot pass.
        assert_eq!(
            perturbed.long_threshold.to_bits(),
            0x3F16_D12E,
            "long_threshold moved: {} (the reference is 0.5891293)",
            perturbed.long_threshold
        );
        assert_eq!(
            perturbed.short_threshold.to_bits(),
            0xBEB3_50F1,
            "short_threshold moved: {} (the reference is -0.3502269)",
            perturbed.short_threshold
        );
        assert_eq!(
            perturbed
                .weights
                .iter()
                .map(|w| w.to_bits())
                .collect::<Vec<_>>(),
            vec![0x3F54_84CF_u32, 0xBEEA_F19D, 0x3E6F_AFC8],
            "the weight draws moved: {:?} (the reference is \
             [0.8301515, -0.4588746, 0.23406899])",
            perturbed.weights
        );
        assert_eq!(
            perturbed.sl_pips.to_bits(),
            0x4032_82A3_D25D_27A8,
            "sl_pips moved: {} (the reference is 18.510312221281907)",
            perturbed.sl_pips
        );
        assert_eq!(
            perturbed.tp_pips.to_bits(),
            0x4041_AC00_D7A2_0C94,
            "tp_pips moved: {} (the reference is 35.34377570545726)",
            perturbed.tp_pips
        );
    }

    /// Each (candidate, run) must be its OWN perturbation, or 100 Monte-Carlo
    /// runs are one run counted 100 times and the screen's pass rate is a
    /// constant.
    #[test]
    fn every_candidate_and_run_draws_its_own_perturbation() {
        let gene = seed_gene();
        let mut seen = std::collections::HashSet::new();
        for candidate in 0..8usize {
            for run in 0..8u64 {
                let p = host_monte_carlo_perturbation(&gene, 0xC0FFEE, candidate, run);
                assert!(
                    seen.insert(p.long_threshold.to_bits()),
                    "candidate {candidate} run {run} reused a draw"
                );
            }
        }
    }

    /// A gene with no fixed stop must not acquire one, and the guard must match
    /// the device mirror's exactly — both lanes skip the draw rather than
    /// multiplying a zero or a NaN.
    #[test]
    fn an_unset_stop_stays_unset_on_both_lanes() {
        let mut gene = seed_gene();
        gene.sl_pips = 0.0;
        gene.tp_pips = f64::NAN;
        let host = host_monte_carlo_perturbation(&gene, 1, 0, 0);
        assert_eq!(host.sl_pips, 0.0);
        assert!(host.tp_pips.is_nan());

        let device = crate::gpu_native::scenario::perturbed_gene(
            1,
            gene.long_threshold,
            gene.short_threshold,
            &gene.weights,
            gene.sl_pips,
            gene.tp_pips,
        );
        assert_eq!(device.sl_pips, 0.0);
        assert!(device.tp_pips.is_nan());
    }

    /// The two lanes measure the same DISTRIBUTION and not the same DRAWS, and
    /// that is stated here rather than left to be discovered.
    ///
    /// If this ever starts failing because the two agree, something has quietly
    /// made the device reproduce ChaCha8 — which would be excellent news and
    /// must still be verified rather than assumed.
    #[test]
    fn the_device_lane_is_a_different_sequence_and_says_so() {
        let gene = seed_gene();
        let host = host_monte_carlo_perturbation(&gene, 0xC0FFEE, 3, 7);
        let counter = 0xC0FFEE_u64 ^ ((3_u64) << 20) ^ 7;
        let device = crate::gpu_native::scenario::perturbed_gene(
            counter,
            gene.long_threshold,
            gene.short_threshold,
            &gene.weights,
            gene.sl_pips,
            gene.tp_pips,
        );
        assert_ne!(
            host.long_threshold, device.long_threshold,
            "the two Monte-Carlo lanes are not expected to agree draw for draw"
        );

        // But both must stay inside the SAME amplitude, because that is the
        // property the screen actually depends on: a 15 % threshold band, a
        // 20 % weight band and a 25 % stop band.
        for (value, base, amplitude) in [
            (
                f64::from(device.long_threshold),
                f64::from(gene.long_threshold),
                0.15,
            ),
            (device.sl_pips, gene.sl_pips, 0.25),
            (device.tp_pips, gene.tp_pips, 0.25),
        ] {
            let ratio = value / base;
            assert!(
                ratio >= 1.0 - amplitude && ratio <= 1.0 + amplitude,
                "device perturbation {value} is {ratio}x its base, outside +/-{amplitude}"
            );
        }

        // And the host lane must be inside the same bands, which is what makes
        // "same distribution" a checkable claim rather than a hope.
        let host_ratio = f64::from(host.long_threshold) / f64::from(gene.long_threshold);
        assert!(host_ratio >= 0.85 && host_ratio <= 1.15);
    }
}

#[cfg(test)]
mod streaming_and_predicate_tests {
    use super::*;

    fn gene_with(expectancy: f64, profit_factor: f64, win_rate: f64, trades: usize) -> Gene {
        Gene {
            expectancy,
            profit_factor,
            win_rate,
            trades_count: trades,
            ..Gene::default()
        }
    }

    /// The floor the predicate reads is the one the quality screen reads. At
    /// the shipped `0.0`, both mean "strictly greater than zero".
    fn shipped_profile() -> TargetProfile {
        TargetProfile {
            min_net_expectancy_per_trade: 0.0,
            min_expectancy_t_stat: 0.0,
            min_win_rate: 0.0,
            min_payoff_ratio: 2.0,
            max_in_market: 0.0,
        }
    }

    // ── PARITY, FIRST ──────────────────────────────────────────────────────

    /// THE PARITY CASE for the loop: one batch whose working set is the whole
    /// vocabulary, with the predicate never firing, must perform EXACTLY one
    /// feature build and one discovery cycle — which is today's path.
    ///
    /// The column-level half of this parity claim is
    /// `neoethos_data::core::hpc_ta::streaming_advance_tests::
    /// whole_space_batch_is_byte_identical_to_the_non_streaming_plan`, which
    /// asserts on the (id, period) LIST rather than on a width, because the
    /// extension emits `<id>_<period>` into the same namespace as the base pass
    /// and a duplicate NAME is a hard error there.
    #[test]
    fn whole_space_single_batch_runs_exactly_one_cycle() {
        let space_len = neoethos_data::core::hpc_ta::extended_sweep_space_len();
        let mut search = StreamingSearch {
            cursor: 0,
            batch_columns: usize::MAX,
            space_len,
            budget_rows: 1_000,
            batches_started: 0,
        };
        let first = search.next_batch().expect("a whole-space batch");
        assert!(first.covers_whole_space());
        assert!(first.exhausted);
        assert!(
            search.next_batch().is_none(),
            "the cursor must not wrap — a second batch would re-explore the same space"
        );
        assert_eq!(search.batches_started(), 1);
    }

    /// A batch width of zero means "this machine affords no streaming
    /// extension". It must produce NO batches rather than an endless stream of
    /// empty ones.
    #[test]
    fn a_machine_that_affords_nothing_streams_nothing() {
        let mut search = StreamingSearch {
            cursor: 0,
            batch_columns: 0,
            space_len: neoethos_data::core::hpc_ta::extended_sweep_space_len(),
            budget_rows: 1_000,
            batches_started: 0,
        };
        assert!(search.next_batch().is_none());
    }

    // ── THE PREDICATE: it must be incapable of rejecting a survivor ────────

    /// The measured case. On `card-run-valid.log` the best of 174 candidates
    /// had profit factor 0.92 and net EUR -50,682 — expectancy negative by
    /// construction. The predicate rejects, and the run's own numbers prove it
    /// could not have discarded a survivor: `portfolio_size = 0`.
    #[test]
    fn the_measured_174_of_174_batch_is_rejected() {
        let genes: Vec<Gene> = (0..200)
            .map(|i| gene_with(-40.0 - i as f64, 0.92, 0.49, 300))
            .collect();
        let verdict = evaluate_batch_early_reject(&genes, &shipped_profile());
        assert!(verdict.is_reject());
        assert_eq!(
            verdict.reason(),
            BatchRejectReason::NoCandidateClearsExpectancyFloor.as_str()
        );
        assert_eq!(verdict.measured, 200);
    }

    /// ONE candidate with a gross edge is enough to save the whole batch, even
    /// when every other candidate is catastrophic. This is the certainty leg:
    /// `profit_factor >= 1.0` means the candidate did not lose money gross, and
    /// the predicate is not allowed to have an opinion about it.
    #[test]
    fn one_candidate_with_a_gross_edge_saves_the_batch() {
        let mut genes: Vec<Gene> = (0..200).map(|_| gene_with(-80.0, 0.5, 0.30, 400)).collect();
        genes.push(gene_with(0.01, 1.0, 0.51, 400));
        let verdict = evaluate_batch_early_reject(&genes, &shipped_profile());
        assert!(!verdict.is_reject());
        assert_eq!(verdict.reason(), BatchAcceptReason::CandidateClearsFloor.as_str());
    }

    /// A thin sample is uncertainty, not evidence. This is the leg that answers
    /// the observed GA archive going 0/200 at generation 4 and 289/527 at
    /// generation 527: a predicate that fires on "the archive looks empty" is a
    /// false-reject generator.
    #[test]
    fn a_thin_population_passes_rather_than_rejecting() {
        let genes: Vec<Gene> = (0..EARLY_REJECT_MIN_MEASURED - 1)
            .map(|_| gene_with(-500.0, 0.2, 0.10, 90))
            .collect();
        let verdict = evaluate_batch_early_reject(&genes, &shipped_profile());
        assert!(!verdict.is_reject());
        assert_eq!(
            verdict.reason(),
            BatchAcceptReason::UncertainTooFewMeasured.as_str()
        );
    }

    /// Genes that never traded carry `expectancy = 0.0` by construction. They
    /// are not measurements and must not be counted — at a floor of 0.0 that is
    /// exactly the difference between "uncertain" and "rejected".
    #[test]
    fn genes_that_never_traded_are_not_evidence() {
        let genes: Vec<Gene> = (0..500).map(|_| gene_with(0.0, 0.0, 0.0, 0)).collect();
        let verdict = evaluate_batch_early_reject(&genes, &shipped_profile());
        assert!(!verdict.is_reject());
        assert_eq!(verdict.reason(), BatchAcceptReason::UncertainNoMetrics.as_str());
        assert_eq!(verdict.measured, 0);
    }

    /// An empty population is uncertainty, never rejection.
    #[test]
    fn an_empty_population_passes() {
        let verdict = evaluate_batch_early_reject(&[], &shipped_profile());
        assert!(!verdict.is_reject());
    }

    /// The margin only ever makes the predicate MORE permissive than the
    /// configured floor. A batch sitting just under the floor is passed.
    #[test]
    fn the_margin_can_only_widen_what_is_accepted() {
        // Scale = mean |expectancy| = 100, margin = 25. best = -10 > 0 - 25.
        let mut genes: Vec<Gene> = (0..100).map(|_| gene_with(-100.0, 0.8, 0.4, 300)).collect();
        genes[0] = gene_with(-10.0, 0.9, 0.45, 300);
        let verdict = evaluate_batch_early_reject(&genes, &shipped_profile());
        assert!(!verdict.is_reject());
        assert_eq!(
            verdict.reason(),
            BatchAcceptReason::UncertainWithinMargin.as_str()
        );
    }

    /// The predicate reads the OPERATOR'S floor. Raising it in config must move
    /// the decision, and lowering it must too — the threshold is never a
    /// literal in this file.
    #[test]
    fn the_floor_comes_from_config_not_from_the_predicate() {
        let genes: Vec<Gene> = (0..100).map(|_| gene_with(-100.0, 0.8, 0.4, 300)).collect();
        let mut lenient = shipped_profile();
        lenient.min_net_expectancy_per_trade = -1_000.0;
        assert!(!evaluate_batch_early_reject(&genes, &lenient).is_reject());
        let strict = shipped_profile();
        assert!(evaluate_batch_early_reject(&genes, &strict).is_reject());
    }

    /// Every batch the ledger sees is counted, and a rejection is NAMED with
    /// its cursor.
    #[test]
    fn the_ledger_names_what_it_abandoned() {
        let mut ledger = BatchRejectionLedger::default();
        let genes: Vec<Gene> = (0..100).map(|_| gene_with(-100.0, 0.8, 0.4, 300)).collect();
        let reject = evaluate_batch_early_reject(&genes, &shipped_profile());
        ledger.record(864, &reject);
        let keep = evaluate_batch_early_reject(&[], &shipped_profile());
        ledger.record(1728, &keep);
        assert_eq!(ledger.batches_seen, 2);
        assert_eq!(ledger.batches_rejected, 1);
        assert_eq!(ledger.rejected_examples.len(), 1);
        assert_eq!(ledger.rejected_examples[0].0, 864);
        assert_eq!(ledger.accepted_uncertain_no_metrics, 1);
    }

    // ── prefilter_top_k ────────────────────────────────────────────────────

    /// At the shipped configuration the derived value does not bind, so the
    /// effective pool is exactly the 240 it has always been. The change is
    /// therefore inert until the population is large enough for the alphabet to
    /// support more.
    #[test]
    fn the_shipped_population_still_gets_the_configured_240() {
        assert_eq!(resolve_prefilter_top_k(240, 1_795, 1_000, 5), 240);
        assert_eq!(resolve_prefilter_top_k(240, 1_795, 100, 5), 240);
    }

    /// At the GPU population the derivation binds and reproduces the
    /// historical operating point (265 kept) to within rounding.
    #[test]
    fn the_gpu_population_derives_the_historical_coverage() {
        let k = resolve_prefilter_top_k(240, 12_639, 4_096, 5);
        assert_eq!(k, 267);
        let coverage = 4_096.0 * 3.0 / k as f64;
        assert!(
            (44.0..48.0).contains(&coverage),
            "expected ~46 genes per column, got {coverage}"
        );
    }

    /// The number must NOT grow with the cube. A bigger box and a longer
    /// timeframe list do not enlarge the alphabet the GA can cover.
    #[test]
    fn a_wider_cube_does_not_widen_the_pool() {
        let narrow = resolve_prefilter_top_k(240, 651, 4_096, 5);
        let wide = resolve_prefilter_top_k(240, 46_343, 4_096, 5);
        assert_eq!(narrow, wide);
    }

    /// The cube width is still a ceiling — a pool wider than the cube is
    /// meaningless.
    #[test]
    fn the_cube_width_remains_a_hard_ceiling() {
        assert_eq!(resolve_prefilter_top_k(240, 90, 4_096, 5), 90);
    }

    /// `0` still disables the prefilter entirely. Unchanged semantics.
    #[test]
    fn zero_still_means_no_prefilter() {
        assert_eq!(resolve_prefilter_top_k(0, 1_795, 4_096, 5), 0);
    }

    /// The four state families are force-kept; a classic indicator column is
    /// not, and a higher-timeframe copy of a state column is not (it carries a
    /// TF prefix, so it is ranked like any other multi-TF column and protected
    /// by the per-TF quota instead).
    #[test]
    fn only_base_timeframe_state_columns_are_force_kept() {
        assert!(is_prefilter_state_column("regime_vol_state"));
        assert!(is_prefilter_state_column("smc_ob"));
        assert!(is_prefilter_state_column("session_london_open"));
        assert!(is_prefilter_state_column("fp_delta"));
        assert!(!is_prefilter_state_column("rsi_14"));
        assert!(!is_prefilter_state_column("H1_smc_ob"));
    }
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;
