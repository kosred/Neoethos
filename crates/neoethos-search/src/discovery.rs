use crate::artifact_io::{stable_json_hash, write_json_atomic};
use crate::data_selection::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeV2,
    CanonicalSearchEvaluatedWindowV1, CanonicalSearchInputReceiptV2, CanonicalSearchRunInputV2,
    CanonicalSearchWindowRoleV1,
};
use crate::eval::{
    BacktestMetrics, fast_evaluate_strategy_core, simulate_trades_core,
    simulate_trades_with_confidence_core,
};
use crate::genetic::search_engine::signals_and_confidence_for_gene_full_with_smc;
use crate::genetic::strategy_gene::EvaluationConfig;
use crate::genetic::{
    Gene, SmcGateArrays, build_smc_arrays, evolve_search_with_progress_and_limits_exact,
    month_day_indices, signals_and_confidence_for_gene_full, signals_for_gene_full,
    signals_for_gene_full_with_smc,
};
use crate::quality::{StrategyMetrics, StrategyQualityAnalyzer, Trade};
use crate::validation::{
    CanonicalBacktestArtifactFile, CombinatorialPurgedCV, ForwardTestInput,
    ForwardTestValidationArtifactFile, PropFirmRiskInput, PropFirmRiskRules,
    PropFirmRiskValidationArtifactFile, ValidationStrategyIdentityV2, WalkforwardSummary,
    WalkforwardValidationArtifactFile, compute_forward_test_summary,
    compute_prop_firm_risk_summary, write_canonical_backtest_artifact_atomic,
    write_forward_test_validation_artifact_atomic, write_prop_firm_risk_validation_artifact_atomic,
    write_walkforward_validation_artifact_atomic,
};
use anyhow::{Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use neoethos_core::contracts::{
    DeterminismPolicy, LiveValidationEvidence, TemporalFeatureContract, ValidationEvidenceManifest,
};
use neoethos_data::{FeatureFrame, Ohlcv};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;

const PERMUTATION_MONTE_CARLO_P_VALUE_SEMANTICS_V1: &str =
    "neoethos.permutation-monte-carlo-p-value.v1";
const PERMUTATION_MONTE_CARLO_P_VALUE_PRIMARY_SOURCE_V1: &str =
    "https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf";

/// Exact Monte Carlo permutation p-value authority for the robustness gate.
///
/// Phipson and Smyth prove that an independently sampled permutation test must
/// use `(b + 1) / (m + 1)`: `b / m` can report the impossible p-value zero and
/// biases the finite-resample decision. Invalid counts fail closed rather than
/// silently manufacturing a gate value.
fn permutation_monte_carlo_p_value_v1(beats: usize, permutations: usize) -> Result<f64> {
    if permutations == 0 || beats > permutations {
        anyhow::bail!(
            "invalid counts for {} (primary authority {}): beats={beats}, permutations={permutations}",
            PERMUTATION_MONTE_CARLO_P_VALUE_SEMANTICS_V1,
            PERMUTATION_MONTE_CARLO_P_VALUE_PRIMARY_SOURCE_V1,
        );
    }
    let numerator = beats.checked_add(1).ok_or_else(|| {
        anyhow::anyhow!(
            "numerator overflow for {}",
            PERMUTATION_MONTE_CARLO_P_VALUE_SEMANTICS_V1
        )
    })?;
    let denominator = permutations.checked_add(1).ok_or_else(|| {
        anyhow::anyhow!(
            "denominator overflow for {}",
            PERMUTATION_MONTE_CARLO_P_VALUE_SEMANTICS_V1
        )
    })?;
    Ok(numerator as f64 / denominator as f64)
}

/// Typed runtime knobs that previously lived only in `NEOETHOS_BOT_*` env vars.
///
/// These values change *production* discovery semantics (which features are
/// kept, how much data the stage-1 funnel sees, what counts as in-sample for
/// the prefilter), so they belong in typed config rather than ambient env
/// state. These are configured via `models.discovery_runtime` (typed config)
/// and resolved by `DiscoveryRuntimeOverrides::from_settings`, which is the
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
            // The outer discovery wrapper has already removed the untouched
            // holdout. Ranking on only 80% of that admissible selection prefix
            // is not a second out-of-sample check; it merely discards another
            // fifth of the evidence available to choose the vocabulary.
            prefilter_insample_frac: 1.0,
            prefilter_min_per_timeframe: 6,
            funnel_stage1_pct: 0.25,
            stage1_window: Stage1Window::Earliest,
            // **2026-05-26 operator directive (Κωνσταντίνος)**: the design
            // intent was always "use 80/20 of WHATEVER data we have", not
            // "require absolute 10y before running". The 80/20 train/val
            // split is enforced downstream by
            // `run_discovery_cycle_with_holdout` using the canonical outer-OOS
            // fraction, which adapts to any window length. Setting the
            // absolute-minimum gate to 0 by default
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

    pub(crate) fn resolved_funnel_stage1_pct(&self) -> f64 {
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
            winner = "risk.backtest_spread_pips + 2 * risk.slippage_pips_per_fill",
            loser = "models.eval_runtime.spread_pips",
            effective_spread_pips = settings.risk.backtest_spread_pips.max(0.0)
                + 2.0 * settings.risk.slippage_pips.max(0.0),
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
        "prop-firm challenge mode. The app live-trading service consumes this through \
         domain::risk::RiskManager; this discovery hook reports the configured gate only."
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
    /// `crate::genetic::strategy_gene::round_trip_commission_per_lot`, gated
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
    /// Fractional fee applied once when realised PnL is converted into the
    /// account currency. This is resolved with the rest of the broker cost
    /// basis and frozen into the run identity; evaluators must not re-read
    /// process-global symbol metadata after the run has been constructed.
    pub pnl_conversion_fee_rate: f64,
    /// Weekend kill zones — force-close before the weekend close and block
    /// Friday-late / Monday-open entries (`eval.rs:1537`, `:1654`).
    ///
    /// WIRED 2026-08-10 (audit #75/#217). This was the literal `true` in
    /// `discovery_backtest_settings`, sitting between two fields that read
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
    /// simulator, and that is visible rather than silent: the resolved value is
    /// part of the canonical search-config hash and of the run profile, so
    /// artifacts produced on either side of the switch remain distinguishable.
    pub kill_zones_enabled: bool,
    pub population: usize,
    /// When `true`, `run_search` raises the GA population to the card's fits
    /// ceiling (bounded to 16 384, never below `population`) and logs the
    /// resolved value as a selection-changing decision. `false` keeps
    /// `population` exactly. From `models.prop_search_population_auto`.
    pub population_auto: bool,
    pub generations: usize,
    pub max_indicators: usize,
    /// Post-GA coverage limit. Zero admits every returned candidate; RAM limits
    /// concurrent replay workers, not the number of strategies eligible for WF.
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
    /// Confidence at which sizing reaches `risk_per_trade_max`, from
    /// `risk.high_quality_confidence`. This is the third input to the same
    /// confidence-scaled sizing formula as the two fields above.
    pub high_quality_confidence: f64,
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

const DEFAULT_HIGH_QUALITY_CONFIDENCE: f64 = 0.65;

/// Exact risk inputs that a Discovery run resolves before evaluating a gene.
///
/// The desktop uses this same resolver for its pre-flight controls, so the
/// numbers shown to the operator cannot drift from the values copied into
/// [`DiscoveryConfig`] and then into CPU/CUDA evaluation settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedDiscoveryRiskProfile {
    pub shared_band: (f64, f64),
    pub risky_band_override: Option<(f64, f64)>,
    pub prop_firm_band_override: Option<(f64, f64)>,
    pub high_quality_confidence: f64,
}

impl ResolvedDiscoveryRiskProfile {
    pub fn risky_band(self) -> (f64, f64) {
        self.risky_band_override.unwrap_or(self.shared_band)
    }

    pub fn prop_firm_band(self) -> (f64, f64) {
        self.prop_firm_band_override.unwrap_or(self.shared_band)
    }
}

/// Refuse values that the sizing implementation would otherwise interpret as
/// "every signal has maximum quality" and therefore size at maximum risk.
fn resolve_high_quality_confidence(configured: f64) -> f64 {
    if configured.is_finite() && configured > 0.0 && configured <= 1.0 {
        return configured;
    }
    tracing::warn!(
        target: "neoethos_search::discovery",
        configured,
        used = DEFAULT_HIGH_QUALITY_CONFIDENCE,
        "risk.high_quality_confidence must be in (0, 1]; refusing the configured value because \
         the sizing path would otherwise treat every signal as maximum quality"
    );
    DEFAULT_HIGH_QUALITY_CONFIDENCE
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

/// Resolve every search-time sizing input from the operator's settings.
/// This function is intentionally independent of broker-financial authority:
/// it describes the proposed run, but grants no right to execute one.
pub fn resolve_discovery_risk_profile(
    settings: &neoethos_core::Settings,
) -> ResolvedDiscoveryRiskProfile {
    let shared_min = settings.risk.min_risk_per_trade.clamp(0.0, 1.0);
    let shared_max = settings.risk.max_risk_per_trade.clamp(shared_min, 1.0);

    ResolvedDiscoveryRiskProfile {
        shared_band: (shared_min, shared_max),
        risky_band_override: resolve_mode_risk_band(
            settings.risk.risky_min_risk_per_trade,
            settings.risk.risky_max_risk_per_trade,
        ),
        prop_firm_band_override: resolve_mode_risk_band(
            settings.risk.prop_firm_min_risk_per_trade,
            settings.risk.prop_firm_max_risk_per_trade,
        ),
        high_quality_confidence: resolve_high_quality_confidence(
            settings.risk.high_quality_confidence,
        ),
    }
}

/// Default product objective for one 60-day Prop-firm screening window:
/// approximately 4% net per month. A configured target still overrides it.
pub const DEFAULT_PROP_FIRM_DISCOVERY_WINDOW_TARGET: f64 = 0.08;

/// Resolve the configurable Prop-firm search gate without starting Discovery.
/// The search driver and desktop pre-flight both call this function; the
/// promotion and live gates remain separate authorities.
pub fn resolve_prop_firm_discovery_gate(
    config: &neoethos_core::config::PropFirmGateConfig,
) -> PropFirmGateOverrides {
    let mut rules = PropFirmRiskRules::default();
    rules.min_profit_target_pct = DEFAULT_PROP_FIRM_DISCOVERY_WINDOW_TARGET;
    rules.require_profit_target = true;
    if let Some(value) = config.max_daily_loss_pct {
        rules.max_daily_loss_pct = value;
    }
    if let Some(value) = config.max_overall_drawdown_pct {
        rules.max_overall_drawdown_pct = value;
    }
    if let Some(value) = config.profit_target_pct {
        rules.min_profit_target_pct = value;
        rules.require_profit_target = value > 0.0;
    }
    if let Some(value) = config.min_trading_days {
        rules.min_trading_days = value;
    }

    PropFirmGateOverrides {
        rules,
        n_windows: config.n_windows,
        window_days: config.window_days.max(1),
        pass_rate: config.pass_rate.clamp(0.0, 1.0),
    }
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
            pnl_conversion_fee_rate: 0.0,
            // Same value `RiskConfig::default()` ships (`config.rs:671`), so a
            // config-less fallback searches under the same weekend policy the
            // live loop applies. See the field's doc for why this is one knob
            // and not two.
            kill_zones_enabled: true,
            population: 1000,
            population_auto: true,
            generations: 10,
            max_indicators: 5,
            // Post-GA replay/validation cap. This is deliberately distinct
            // from the GA population and generation count: 200 x 1,000
            // expands the evolutionary search, it must not silently turn into
            // 1,000 full-history finalist replays when Settings has not been
            // loaded.
            candidate_count: 0,
            portfolio_size: 4,
            max_rows: 0,
            max_rows_by_timeframe: HashMap::new(),
            max_hours: 0.0,
            corr_threshold: 0.85,
            min_trades_per_day: 0.0,
            // Payoff shape and trade cadence belong in the search objective,
            // not in a config-less validity gate. Positive net expectancy after
            // actual costs remains unconditional in `TargetProfile::evaluate`.
            target_profile: TargetProfile {
                min_payoff_ratio: 0.0,
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
            // Keep the config-less fallback identical to
            // `DiscoveryConfig::from_settings(&Settings::default())`. A failed
            // YAML load must not widen PropFirm risk from 1% to 3%.
            risk_per_trade_min: 0.0,
            risk_per_trade_max: 0.01,
            high_quality_confidence: DEFAULT_HIGH_QUALITY_CONFIDENCE,
            // Decision default (2026-08-09): the Risky 30% ceiling is operator
            // intent, so the config-less fallback carries the same band as
            // `from_settings` derives from `risk.risky_max_risk_per_trade`
            // (min inherits 0.0). Kept in lockstep (divergence test).
            risky_risk_band: Some((0.0, 0.30)),
            prop_firm_risk_band: Some((0.0, 0.01)),
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

fn screening_spread_and_slippage_pips(spread_pips: f64, slippage_pips_per_fill: f64) -> f64 {
    // A full quoted width is paid once per round trip; adverse slippage is
    // charged at BOTH fills. Executable-side Bid/Ask replay is separate.
    spread_pips + 2.0 * slippage_pips_per_fill.max(0.0)
}

impl DiscoveryConfig {
    /// Bind research capital to its account currency at every metric consumer.
    /// This validates units and capital, not broker execution authority.
    pub fn initial_account_balance(&self) -> anyhow::Result<neoethos_broker_truth::AccountMoneyV1> {
        let balance = neoethos_broker_truth::AccountMoneyV1::new(
            self.evaluation_account_currency.clone(),
            self.initial_balance,
        )?;
        anyhow::ensure!(
            balance.amount() > 0.0,
            "initial account balance must be positive"
        );
        Ok(balance)
    }

    /// Production settings adapter. Financial fields are unreachable until
    /// the exact broker replay capability is installed; callers must not use
    /// `from_settings` as a fallback after this refusal.
    pub fn try_from_settings(settings: &neoethos_core::Settings) -> anyhow::Result<Self> {
        neoethos_core::current_broker_financial_truth_capability_v1()
            .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
            .map_err(anyhow::Error::new)?;
        Ok(Self::from_settings(settings))
    }

    /// Resolve the ordinary search knobs from settings while binding every
    /// financial input to an explicit canonical-trendbar research contract.
    /// This path never grants live or promotion authority.
    pub fn try_from_settings_for_canonical_trendbar_research(
        settings: &neoethos_core::Settings,
        contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    ) -> anyhow::Result<Self> {
        contract.validate()?;
        anyhow::ensure!(
            settings.system.symbol.trim() == contract.symbol(),
            "research contract symbol {} does not match settings symbol {}",
            contract.symbol(),
            settings.system.symbol
        );
        anyhow::ensure!(
            settings.system.account_currency.trim() == contract.account_currency(),
            "research contract account currency {} does not match settings account currency {}",
            contract.account_currency(),
            settings.system.account_currency
        );
        // Ordinary settings resolution still runs unchanged, but its temporary
        // metadata/config costs are not the costs this research route evaluates.
        // Report only the final sealed values after applying the contract.
        let mut config = Self::from_settings_with_cost_diagnostics(settings, false);
        apply_research_contract_to_discovery_config(&mut config, contract);
        tracing::info!(
            target: "neoethos_search::cost_model",
            classification = "research_only",
            assumption_source_id = contract.assumption_source_id(),
            assumption_source_sha256 = contract.assumption_source_sha256(),
            symbol = contract.symbol(),
            account_currency = contract.account_currency(),
            full_spread_pips = contract.screening_costs().full_spread_pips_assumption(),
            entry_slippage_pips = contract.screening_costs().slippage_pips_per_fill_assumption(),
            exit_slippage_pips = contract.screening_costs().slippage_pips_per_fill_assumption(),
            total_spread_and_slippage_round_trip_pips = config.evaluation_spread_pips,
            commission_account_per_lot_per_fill = contract.screening_costs().commission_account_per_lot_per_fill_assumption(),
            commission_account_per_lot_round_trip = config.evaluation_commission_per_trade,
            swap_long_pips_per_day = config.swap_long_pips_per_day,
            swap_short_pips_per_day = config.swap_short_pips_per_day,
            sensitivity_commission_account_per_lot_round_trip = config.sensitivity_commission_per_lot,
            "resolved sealed canonical research cost assumptions; not broker/live financial authority"
        );
        Ok(config)
    }

    pub(crate) fn from_settings(settings: &neoethos_core::Settings) -> Self {
        Self::from_settings_with_cost_diagnostics(settings, true)
    }

    fn from_settings_with_cost_diagnostics(
        settings: &neoethos_core::Settings,
        log_unsealed_financials: bool,
    ) -> Self {
        // Cross-currency market data cannot be selected from Settings alone:
        // `data_dir + symbol` does not identify a source/account. The caller
        // installs `fx_rates::set_store_selection` only after choosing and
        // verifying the exact CanonicalDatasetIdentity for this run.
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

        // Keep zero as "all returned candidates" until the actual GA/archive
        // result exists. Population/generation settings are evaluation slots,
        // not a count of unique strategies or a post-GA admission ceiling.
        let candidate_count = model_settings.prop_search_val_candidates;

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
        let pnl_conversion_fee_rate = meta
            .and_then(|m| m.pnl_conversion_fee_rate)
            .filter(|rate| rate.is_finite() && *rate >= 0.0 && *rate < 1.0)
            .unwrap_or(0.0);
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
        if log_unsealed_financials {
            tracing::info!(
                target: "neoethos_search::cost_model",
                symbol = %symbol,
                quoted_commission_per_lot = quoted_commission,
                commission_is_per_side,
                round_trip_commission_per_lot = resolved_commission,
                "commission resolved to a ROUND TRIP charge — the evaluators subtract \
                 it once per closed trade"
            );
        }

        // The session-spread curve. `Err` can be an authority refusal or a
        // partial / malformed curve: a cost model configured for two of the
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
                let with_slip =
                    curve.map(|spread| screening_spread_and_slippage_pips(spread, slip));
                if log_unsealed_financials {
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
                }
                Some(with_slip)
            }
            Ok(None) => {
                if log_unsealed_financials {
                    tracing::warn!(
                        target: "neoethos_search::cost_model",
                        symbol = %symbol,
                        flat_spread_pips = screening_spread_and_slippage_pips(
                            settings.risk.backtest_spread_pips.max(0.0),
                            settings.risk.slippage_pips,
                        ),
                        "no session spread curve configured — a FLAT spread is charged at 03:00 \
                         Tokyo and at the London open alike. The per-bar lookup exists on both the \
                         CPU path and the CUDA kernel and is simply unpopulated. Measure your \
                         broker's per-hour spread and set risk.backtest_spread_pips_{{asian,\
                         overlap,late_ny}}. Until then, any result that depends on WHEN it trades \
                         is measured at the wrong cost."
                    );
                }
                None
            }
            Err(reason) => {
                // Preserve the refusal and the existing settings resolution.
                // Ordinary resolution reports it; the explicit research route
                // reports its sealed scalar costs after replacing this value.
                if log_unsealed_financials {
                    tracing::error!(
                        target: "neoethos_search::cost_model",
                        symbol = %symbol,
                        reason = %reason,
                        "session spread curve REFUSED — falling back to the flat spread. Fix the \
                         three risk.backtest_spread_pips_* keys or remove all three."
                    );
                }
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

        if log_unsealed_financials && (meta.is_none() || (swap_long == 0.0 && swap_short == 0.0)) {
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
        } else if log_unsealed_financials {
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

        let risk_profile = resolve_discovery_risk_profile(settings);

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
            // the live fills never see. This scalar is a full round-trip
            // screening envelope: full spread once plus adverse slippage on
            // BOTH entry and exit fills. The later Bid/Ask replay is separate;
            // executable-side quote prices already contain spread and must not
            // charge this scalar again.
            evaluation_spread_pips: screening_spread_and_slippage_pips(
                settings.risk.backtest_spread_pips.max(0.0),
                settings.risk.slippage_pips,
            ),
            evaluation_commission_per_trade: resolved_commission,
            session_spread_pips,
            cost_band_pips,
            swap_long_pips_per_day: swap_long,
            swap_short_pips_per_day: swap_short,
            pnl_conversion_fee_rate,
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
            // Activity is an operator preference, not a mathematical validity
            // condition. `0.0` explicitly disables it; positive values remain
            // available for a run whose economic target requires a cadence.
            min_trades_per_day: model_settings.prop_search_val_min_trades_per_day.max(0.0),
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
            // One resolver feeds both this execution config and the desktop
            // pre-flight. The UI therefore shows the same clamped, ordered
            // band that Generation 0 and every later validation stage receive.
            risk_per_trade_min: risk_profile.shared_band.0,
            risk_per_trade_max: risk_profile.shared_band.1,
            high_quality_confidence: risk_profile.high_quality_confidence,
            risky_risk_band: risk_profile.risky_band_override,
            prop_firm_risk_band: risk_profile.prop_firm_band_override,
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
                if log_unsealed_financials && round_trip < resolved_commission {
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
            // Activity is a configurable delivery preference, not evidence of
            // profitability. Keep the operator's value exactly; `0.0` disables
            // the cadence gate, while the unconditional positive-net-expectancy
            // check and the total-trade sanity check still reject a gene that
            // never trades. Growth fitness already rewards profitable cadence.
            //
            // Logged unconditionally, because an activity floor that is silently
            // rewritten is exactly the class of bug this line used to be.
            tracing::info!(
                target: "neoethos_search::discovery",
                min_trades_per_day = format!("{:.3}", self.min_trades_per_day),
                "risky mode: using configured activity floor (0 disables it)"
            );
            // No TF-scaling of trade-frequency floors and NO prop_firm_gate:
            // Risky is judged purely on growth, not challenge-passing.
        }
        self
    }

    fn derive_prop_firm_gate(&self) -> PropFirmGateOverrides {
        resolve_prop_firm_discovery_gate(&self.prop_firm_gate_params)
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
        let research_contract =
            crate::historical_evaluation_authority::active_research_contract_v1();
        let mut cfg = if research_contract.as_ref().is_some_and(|contract| {
            self.evaluation_symbol == contract.symbol()
                && self.evaluation_account_currency == contract.account_currency()
        }) {
            // The immutable contract below supplies every monetary field. Do
            // not resolve unrelated ambient broker/FX costs only to overwrite
            // them; keep the same non-monetary defaults as `for_symbol`.
            EvaluationConfig {
                symbol: self.evaluation_symbol.clone(),
                account_currency: self.evaluation_account_currency.clone(),
                ..EvaluationConfig::default()
            }
        } else {
            #[cfg(test)]
            cost_consistency_tests::LEGACY_COST_RESOLUTIONS
                .with(|count| count.set(count.get() + 1));
            EvaluationConfig::for_symbol(
                &self.evaluation_symbol,
                &self.evaluation_account_currency,
                price_hint,
                Some(self.evaluation_spread_pips),
                Some(self.evaluation_commission_per_trade),
            )
        };
        // Generation 0 must use the same execution and sizing policy as every
        // post-GA validation stage. Previously these values were only copied by
        // `discovery_backtest_settings`, so the search silently used
        // `BacktestSettings::default()` while the funnel used operator config.
        cfg.kill_zones_enabled = self.kill_zones_enabled;
        cfg.session_spread_pips = self.session_spread_pips;
        cfg.risk_per_trade_min = self.risk_per_trade_min;
        cfg.risk_per_trade_max = self.risk_per_trade_max;
        cfg.high_quality_confidence = self.high_quality_confidence;
        cfg.initial_equity = self.initial_balance;
        // These values were already captured in `DiscoveryConfig`, but
        // `EvaluationConfig::for_symbol` re-read the mutable global metadata
        // table and could therefore evaluate a different cost basis under the
        // same search-config hash. Freeze the run-scoped values here.
        cfg.swap_long_pips_per_day = self.swap_long_pips_per_day;
        cfg.swap_short_pips_per_day = self.swap_short_pips_per_day;
        cfg.pnl_conversion_fee_rate = self.pnl_conversion_fee_rate;
        // Risky uses the same run-bound realized-balance goal-pace objective
        // inside the GA and after replay. PropFirm/Strict remain unchanged.
        cfg.growth_objective = matches!(self.mode, DiscoveryMode::Risky);
        cfg.growth_goal = cfg
            .growth_objective
            .then_some(crate::scoring::RiskyGrowthGoal {
                start_balance: self.risky_start_balance,
                target_balance: self.risky_target_balance,
                horizon_days: self.risky_horizon_days,
            });
        if let Some(contract) = research_contract {
            // V3 seals a scalar cost envelope, not an ambient session curve.
            // Clear it even on identity mismatch so it cannot mask a refusal.
            cfg.session_spread_pips = None;
            if self.evaluation_symbol == contract.symbol()
                && self.evaluation_account_currency == contract.account_currency()
            {
                cfg.pip_value = contract.pip_size();
                cfg.pip_value_per_lot = contract.pip_value_per_lot();
                cfg.spread_pips = contract.screening_spread_and_slippage_round_trip_pips();
                cfg.commission_per_trade = contract.round_trip_commission_account_per_lot();
                cfg.swap_long_pips_per_day = contract.swap_long_pips_per_day();
                cfg.swap_short_pips_per_day = contract.swap_short_pips_per_day();
                cfg.pnl_conversion_fee_rate = contract.pnl_conversion_fee_rate();
            } else {
                tracing::error!(
                    target: "neoethos_search::canonical_research",
                    config_symbol = %self.evaluation_symbol,
                    contract_symbol = %contract.symbol(),
                    config_account_currency = %self.evaluation_account_currency,
                    contract_account_currency = %contract.account_currency(),
                    "active canonical-trendbar research contract does not match discovery config"
                );
                cfg.pip_value = f64::NAN;
                cfg.pip_value_per_lot = f64::NAN;
                cfg.spread_pips = f64::NAN;
                cfg.commission_per_trade = f64::NAN;
                cfg.swap_long_pips_per_day = f64::NAN;
                cfg.swap_short_pips_per_day = f64::NAN;
                cfg.pnl_conversion_fee_rate = f64::NAN;
            }
        }
        cfg
    }

    pub(crate) fn evaluation_config_with_smc_gate(
        &self,
        price_hint: Option<f64>,
        effective_smc_gate_threshold: f64,
    ) -> EvaluationConfig {
        let mut cfg = self.evaluation_config(price_hint);
        cfg.smc_gate_threshold = effective_smc_gate_threshold;
        cfg
    }
}

pub(crate) fn apply_research_contract_to_discovery_config(
    config: &mut DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
) {
    config.evaluation_symbol = contract.symbol().to_owned();
    config.evaluation_account_currency = contract.account_currency().to_owned();
    config.evaluation_spread_pips = contract.screening_spread_and_slippage_round_trip_pips();
    // The contract does not seal session costs from the surrounding Settings.
    config.session_spread_pips = None;
    config.evaluation_commission_per_trade = contract.round_trip_commission_account_per_lot();
    config.sensitivity_commission_per_lot = config
        .sensitivity_commission_per_lot
        .max(contract.round_trip_commission_account_per_lot());
    config.swap_long_pips_per_day = contract.swap_long_pips_per_day();
    config.swap_short_pips_per_day = contract.swap_short_pips_per_day();
    config.pnl_conversion_fee_rate = contract.pnl_conversion_fee_rate();
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryResult {
    /// Immutable identity/generation/manifest/Vortex/feature-plan proof for
    /// every value this run consumed. Persisted outputs must carry this exact
    /// receipt rather than reconstructing a dataset from symbol/timeframe.
    pub search_input_receipt: CanonicalSearchInputReceiptV2,
    /// Exact window evaluated by the selection pipeline. This is
    /// `DiscoveryInput` for a holdout-free run and `InSample` for a split run.
    pub selection_scope: CanonicalSearchArtifactScopeV2,
    /// Post-search strategy selection and sizing window. Absent on legacy
    /// two-way diagnostics, which cannot authorize a new final-tested export.
    pub calibration_scope: Option<CanonicalSearchArtifactScopeV2>,
    /// Reserved final suffix on three-way runs. Never used for selecting genes
    /// or fitting their position sizes. Legacy diagnostics retain their old tail.
    pub holdout_scope: Option<CanonicalSearchArtifactScopeV2>,
    /// Exact resolved search configuration identity shared by the ledger,
    /// trial-return matrix, and every result artifact. Never recompute this from
    /// ambient settings after the run.
    pub search_config_hash: String,
    /// Exact run-level classification across every candidate whose cost band
    /// was measured. This is carried separately from the per-strategy verdicts
    /// so downstream screens can prove both the population totals and the
    /// identity of each surviving strategy without reconstructing either from
    /// logs or a truncated funnel profile.
    pub cost_band_census: CostBandCensus,
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
    /// [`DiscoveryResult::cost_band_for_strategy`] returns.
    pub cost_band_by_strategy: Vec<(String, CostBandVerdict)>,
    pub portfolio: Vec<Gene>,
    pub candidates: Vec<Gene>,
    /// Scalar rows for all quality survivors. To bound archive-sized payloads,
    /// full per-trade equity curves are retained for the final portfolio only;
    /// an empty curve on another row means not materialized, not flat equity.
    pub quality_metrics: Vec<StrategyMetrics>,
    pub logged_trades: Vec<LoggedStrategyTrades>,
    /// Feature names as they existed *after* prefiltering inside discovery.
    /// Gene indices refer to columns in this list, not the caller's original names.
    pub effective_feature_names: Vec<String>,
    /// Final annealed SMC gate used by the GA and every post-search replay.
    pub effective_smc_gate_threshold: f64,
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

trait ExactDiscoveryValidationArtifact: Serialize {
    fn strategy_identity(&self) -> &ValidationStrategyIdentityV2;

    fn validate_exact(
        &self,
        scope: &CanonicalSearchArtifactScopeV2,
        search_config_hash: &str,
        gene: &Gene,
    ) -> Result<()>;
}

impl ExactDiscoveryValidationArtifact for CanonicalBacktestArtifactFile {
    fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        CanonicalBacktestArtifactFile::strategy_identity(self)
    }

    fn validate_exact(
        &self,
        scope: &CanonicalSearchArtifactScopeV2,
        search_config_hash: &str,
        gene: &Gene,
    ) -> Result<()> {
        CanonicalBacktestArtifactFile::validate_against(self, scope, search_config_hash, gene)
    }
}

impl ExactDiscoveryValidationArtifact for WalkforwardValidationArtifactFile {
    fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        WalkforwardValidationArtifactFile::strategy_identity(self)
    }

    fn validate_exact(
        &self,
        scope: &CanonicalSearchArtifactScopeV2,
        search_config_hash: &str,
        gene: &Gene,
    ) -> Result<()> {
        WalkforwardValidationArtifactFile::validate_against(self, scope, search_config_hash, gene)
    }
}

impl ExactDiscoveryValidationArtifact for ForwardTestValidationArtifactFile {
    fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        ForwardTestValidationArtifactFile::strategy_identity(self)
    }

    fn validate_exact(
        &self,
        scope: &CanonicalSearchArtifactScopeV2,
        search_config_hash: &str,
        gene: &Gene,
    ) -> Result<()> {
        ForwardTestValidationArtifactFile::validate_against(self, scope, search_config_hash, gene)
    }
}

impl ExactDiscoveryValidationArtifact for PropFirmRiskValidationArtifactFile {
    fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        PropFirmRiskValidationArtifactFile::strategy_identity(self)
    }

    fn validate_exact(
        &self,
        scope: &CanonicalSearchArtifactScopeV2,
        search_config_hash: &str,
        gene: &Gene,
    ) -> Result<()> {
        PropFirmRiskValidationArtifactFile::validate_against(self, scope, search_config_hash, gene)
    }
}

fn final_strategy_index<'a>(portfolio: &'a [Gene]) -> Result<HashMap<String, &'a Gene>> {
    let mut by_hash = HashMap::with_capacity(portfolio.len());
    let mut strategy_ids = HashSet::with_capacity(portfolio.len());
    for gene in portfolio {
        let identity = ValidationStrategyIdentityV2::from_gene(gene)?;
        anyhow::ensure!(
            strategy_ids.insert(identity.strategy_id().to_owned()),
            "final portfolio contains duplicate strategy_id `{}`",
            identity.strategy_id()
        );
        anyhow::ensure!(
            by_hash
                .insert(identity.exact_gene_hash().to_owned(), gene)
                .is_none(),
            "final portfolio contains a duplicate exact strategy identity `{}`",
            identity.exact_gene_hash()
        );
    }
    Ok(by_hash)
}

fn validate_exact_artifact_set<T: ExactDiscoveryValidationArtifact>(
    kind: &str,
    artifacts: &[T],
    portfolio: &[Gene],
    scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
    required: bool,
) -> Result<()> {
    if artifacts.is_empty() {
        anyhow::ensure!(
            !required || portfolio.is_empty(),
            "{kind} validation evidence is missing for the final portfolio"
        );
        return Ok(());
    }

    let expected = final_strategy_index(portfolio)?;
    let mut observed = HashSet::with_capacity(artifacts.len());
    for artifact in artifacts {
        let identity = artifact.strategy_identity();
        let Some(gene) = expected.get(identity.exact_gene_hash()).copied() else {
            anyhow::bail!(
                "{kind} validation evidence contains extra strategy `{}` ({}) outside the final portfolio",
                identity.strategy_id(),
                identity.exact_gene_hash()
            );
        };
        anyhow::ensure!(
            observed.insert(identity.exact_gene_hash().to_owned()),
            "{kind} validation evidence contains duplicate strategy `{}` ({})",
            identity.strategy_id(),
            identity.exact_gene_hash()
        );
        artifact
            .validate_exact(scope, search_config_hash, gene)
            .with_context(|| {
                format!(
                    "validating {kind} evidence for final strategy `{}`",
                    gene.strategy_id
                )
            })?;
    }
    if let Some((missing_hash, missing_gene)) = expected
        .iter()
        .find(|(hash, _)| !observed.contains(hash.as_str()))
    {
        anyhow::bail!(
            "{kind} validation evidence is missing final strategy `{}` ({missing_hash})",
            missing_gene.strategy_id
        );
    }
    Ok(())
}

fn retain_exact_artifacts_for_final_portfolio<T: ExactDiscoveryValidationArtifact>(
    kind: &str,
    portfolio: &[Gene],
    artifacts: &mut Vec<T>,
) -> Result<()> {
    let expected = final_strategy_index(portfolio)?;
    artifacts
        .retain(|artifact| expected.contains_key(artifact.strategy_identity().exact_gene_hash()));

    let mut observed = HashSet::with_capacity(artifacts.len());
    for artifact in artifacts.iter() {
        let identity = artifact.strategy_identity();
        let gene = expected
            .get(identity.exact_gene_hash())
            .copied()
            .expect("retain removed identities outside the final portfolio");
        identity.validate_against(gene)?;
        anyhow::ensure!(
            observed.insert(identity.exact_gene_hash().to_owned()),
            "{kind} validation evidence contains duplicate final strategy `{}`",
            identity.strategy_id()
        );
    }
    if let Some((missing_hash, missing_gene)) = expected
        .iter()
        .find(|(hash, _)| !observed.contains(hash.as_str()))
    {
        anyhow::bail!(
            "{kind} validation evidence is missing final strategy `{}` ({missing_hash}) after pruning",
            missing_gene.strategy_id
        );
    }
    Ok(())
}

fn retain_selection_validation_artifacts_for_final_portfolio(
    portfolio: &[Gene],
    canonical_backtest_artifacts: &mut Vec<CanonicalBacktestArtifactFile>,
    walkforward_validation_artifacts: &mut Vec<WalkforwardValidationArtifactFile>,
) -> Result<()> {
    if canonical_backtest_artifacts.is_empty() && walkforward_validation_artifacts.is_empty() {
        return Ok(());
    }
    retain_exact_artifacts_for_final_portfolio(
        "canonical_backtest",
        portfolio,
        canonical_backtest_artifacts,
    )?;
    retain_exact_artifacts_for_final_portfolio(
        "walkforward",
        portfolio,
        walkforward_validation_artifacts,
    )
}

impl DiscoveryResult {
    pub fn search_input_receipt_sha256(&self) -> Result<String> {
        self.search_input_receipt
            .identity_sha256()
            .map_err(anyhow::Error::new)
    }

    /// Refuse any public-struct literal whose stored scopes do not exactly bind
    /// to `search_input_receipt` or do not form one supported full/split shape.
    pub fn validate_evaluated_scopes(&self) -> Result<()> {
        self.selection_scope
            .validate_against_receipt(&self.search_input_receipt)
            .map_err(anyhow::Error::new)?;
        if let Some(holdout) = &self.holdout_scope {
            holdout
                .validate_against_receipt(&self.search_input_receipt)
                .map_err(anyhow::Error::new)?;
        }
        if let Some(calibration) = &self.calibration_scope {
            calibration
                .validate_against_receipt(&self.search_input_receipt)
                .map_err(anyhow::Error::new)?;
            anyhow::ensure!(
                calibration.evaluated_window().role()
                    == CanonicalSearchWindowRoleV1::SelectionValidation
                    && self.holdout_scope.is_some(),
                "calibration requires the selection_validation role and a separate final holdout"
            );
        }

        let anchor_id = self.search_input_receipt.anchor_dataset_identity();
        let anchor_bindings = self
            .search_input_receipt
            .source_bindings()
            .iter()
            .filter(|binding| binding.dataset_identity() == anchor_id)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            anchor_bindings.len() == 1,
            "discovery result scopes require exactly one receipt anchor binding; found {}",
            anchor_bindings.len()
        );
        let segments = anchor_bindings[0].segments();
        anyhow::ensure!(
            !segments.is_empty(),
            "discovery result receipt anchor has no segments"
        );
        anyhow::ensure!(
            segments
                .windows(2)
                .all(|adjacent| adjacent[0].row_end() == adjacent[1].row_start()),
            "discovery result scopes cannot cover disjoint receipt anchor segments"
        );
        let first = segments.first().expect("segments checked non-empty");
        let last = segments.last().expect("segments checked non-empty");
        let selected = self.selection_scope.evaluated_window();

        match (selected.role(), self.holdout_scope.as_ref()) {
            (CanonicalSearchWindowRoleV1::DiscoveryInput, None) => {
                anyhow::ensure!(
                    self.calibration_scope.is_none(),
                    "full-input diagnostics cannot carry calibration"
                );
                anyhow::ensure!(
                    selected.row_start() == first.row_start()
                        && selected.row_end() == last.row_end()
                        && selected.timestamp_start_ms() == first.timestamp_start_ms()
                        && selected.timestamp_end_ms() == last.timestamp_end_ms(),
                    "holdout-free DiscoveryInput scope must exactly cover the receipt anchor"
                );
            }
            (CanonicalSearchWindowRoleV1::InSample, Some(holdout)) => {
                let held_out = holdout.evaluated_window();
                anyhow::ensure!(
                    held_out.role() == CanonicalSearchWindowRoleV1::Holdout,
                    "split discovery result evidence scope must have the holdout role"
                );
                anyhow::ensure!(
                    selected.row_start() == first.row_start()
                        && selected.timestamp_start_ms() == first.timestamp_start_ms(),
                    "split discovery result selection must start at the receipt anchor"
                );
                anyhow::ensure!(
                    held_out.row_end() == last.row_end()
                        && held_out.timestamp_end_ms() == last.timestamp_end_ms(),
                    "split discovery result holdout must end at the receipt anchor"
                );
                let next = self
                    .calibration_scope
                    .as_ref()
                    .map(CanonicalSearchArtifactScopeV2::evaluated_window)
                    .unwrap_or(held_out);
                anyhow::ensure!(
                    selected.row_end() == next.row_start()
                        && selected.timestamp_end_ms() < next.timestamp_start_ms(),
                    "split discovery result selection/evidence rows must be contiguous and timestamps ordered"
                );
                if let Some(calibration) = &self.calibration_scope {
                    let calibrated = calibration.evaluated_window();
                    anyhow::ensure!(
                        calibrated.row_end() == held_out.row_start()
                            && calibrated.timestamp_end_ms() < held_out.timestamp_start_ms(),
                        "calibration/final holdout rows must be contiguous and timestamps ordered"
                    );
                }
            }
            (CanonicalSearchWindowRoleV1::DiscoveryInput, Some(_)) => anyhow::bail!(
                "holdout-free DiscoveryInput scope cannot carry a holdout evidence scope"
            ),
            (CanonicalSearchWindowRoleV1::InSample, None) => anyhow::bail!(
                "in-sample discovery result scope is missing its holdout evidence scope"
            ),
            (role, _) => anyhow::bail!(
                "unsupported discovery result selection role {role:?}; expected discovery_input or in_sample"
            ),
        }
        Ok(())
    }

    pub fn selection_scope(&self) -> Result<&CanonicalSearchArtifactScopeV2> {
        self.validate_evaluated_scopes()?;
        Ok(&self.selection_scope)
    }

    pub fn holdout_scope(&self) -> Result<Option<&CanonicalSearchArtifactScopeV2>> {
        self.validate_evaluated_scopes()?;
        Ok(self.holdout_scope.as_ref())
    }

    pub fn calibration_scope(&self) -> Result<Option<&CanonicalSearchArtifactScopeV2>> {
        self.validate_evaluated_scopes()?;
        Ok(self.calibration_scope.as_ref())
    }

    fn validate_validation_evidence_sets(&self, require_complete: bool) -> Result<()> {
        self.validate_evaluated_scopes()?;
        if require_complete {
            anyhow::ensure!(
                !self.portfolio.is_empty(),
                "promotion validation evidence is missing a final strategy portfolio"
            );
        }
        validate_exact_artifact_set(
            "canonical_backtest",
            &self.canonical_backtest_artifacts,
            &self.portfolio,
            &self.selection_scope,
            &self.search_config_hash,
            require_complete,
        )?;
        validate_exact_artifact_set(
            "walkforward",
            &self.walkforward_validation_artifacts,
            &self.portfolio,
            &self.selection_scope,
            &self.search_config_hash,
            require_complete,
        )?;

        match self
            .calibration_scope
            .as_ref()
            .or(self.holdout_scope.as_ref())
        {
            Some(holdout_scope) => {
                validate_exact_artifact_set(
                    "forward_test",
                    &self.forward_test_validation_artifacts,
                    &self.portfolio,
                    holdout_scope,
                    &self.search_config_hash,
                    require_complete,
                )?;
                validate_exact_artifact_set(
                    "prop_firm",
                    &self.prop_firm_validation_artifacts,
                    &self.portfolio,
                    holdout_scope,
                    &self.search_config_hash,
                    require_complete,
                )?;
            }
            None => {
                anyhow::ensure!(
                    self.forward_test_validation_artifacts.is_empty()
                        && self.prop_firm_validation_artifacts.is_empty(),
                    "holdout validation evidence cannot exist without an exact stored holdout scope"
                );
                anyhow::ensure!(
                    !require_complete,
                    "promotion validation evidence is missing the exact holdout scope"
                );
            }
        }
        Ok(())
    }

    pub fn validate_complete_promotion_evidence(&self) -> Result<()> {
        self.validate_validation_evidence_sets(true)?;
        crate::quote_validated_outer_holdout_v1::require_quote_validated_outer_holdout_v1(None)
            .map(|_| ())
            .map_err(anyhow::Error::new)
    }

    /// Complete numerical selection evidence and configured gates. On new
    /// three-way runs this validates CALIBRATION, not a completed final test.
    /// A reserved final scope is not a final verdict, live admission or exact
    /// quote/fill certification. It only permits freezing a research candidate.
    pub fn validate_complete_selection_evidence(&self) -> Result<()> {
        self.validate_validation_evidence_sets(true)?;
        anyhow::ensure!(
            self.validation_gates.is_portfolio_export_ready(),
            "candidate selection export requires passed walk-forward, CPCV and PBO gates"
        );
        Ok(())
    }

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
    /// artifact said which had run. Empty means the full run-scoped receipt
    /// below is absent; it must never be filled from a process-global
    /// observation.
    pub population_eval_engines: Vec<crate::engine_identity::PopulationEvalEngine>,
    /// Full immutable authority behind `population_eval_engines`: exact
    /// canonical scope, successful population count, ordered engines, and
    /// domain-separated receipt identity. `None` is fail-closed for profiles
    /// built from fixtures or legacy results.
    pub population_execution_run_receipt_v2:
        Option<crate::population_execution_run_receipt_v2::ExactPopulationExecutionRunReceiptV2>,
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
    /// Realised-PnL account-currency conversion fee used by this run.
    pub pnl_conversion_fee_rate: f64,
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
    /// Resolved confidence normaliser used by the sizing formula. It belongs in
    /// the run profile because changing it changes every confidence-sized trade.
    pub high_quality_confidence: f64,
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
    CandidateCensusUpdated {
        census: crate::funnel_profile::DiscoveryCandidateCensus,
    },
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
    let base_timestamps = ohlcv
        .timestamp
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("base OHLCV has no canonical timestamps"))?;
    anyhow::ensure!(
        features.timestamps.as_slice() == base_timestamps,
        "feature/base timestamp mismatch; refusing row-position trimming"
    );
    anyhow::ensure!(
        features.timestamps.len() == frame_rows
            && base_timestamps.len() == frame_rows
            && ohlcv_rows == frame_rows
            && ohlcv.open.len() == frame_rows
            && ohlcv.high.len() == frame_rows
            && ohlcv.low.len() == frame_rows
            && ohlcv
                .volume
                .as_ref()
                .is_none_or(|volume| volume.len() == frame_rows),
        "discovery trimming requires exact feature/OHLCV row and column alignment"
    );
    let available_rows = frame_rows;
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
        features.row_window(start_idx, available_rows)?
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
        // Use the resolved evaluation policy: a sealed scalar research cost
        // envelope must not be overridden by the ambient Discovery curve.
        session_spread_profile: evaluation.session_spread_pips.map(|curve| {
            crate::eval::SessionSpreadProfile {
                asian_pips: curve[0],
                overlap_pips: curve[1],
                late_ny_pips: curve[2],
            }
        }),
        pip_value_per_lot: evaluation.pip_value_per_lot,
        // Decision D: charge overnight financing (the engine applies it in both
        // the CPU path and the CUDA kernel; it was silently 0 here before).
        swap_long_pips_per_day: evaluation.swap_long_pips_per_day,
        swap_short_pips_per_day: evaluation.swap_short_pips_per_day,
        pnl_conversion_fee_rate: evaluation.pnl_conversion_fee_rate,
        // #75/#217 (2026-08-10). This was the literal `true`, between two
        // fields that read `config.`. It is now the one knob both sides read.
        kill_zones_enabled: config.kill_zones_enabled,
        risk_per_trade_min: config.risk_per_trade_min,
        risk_per_trade_max: config.risk_per_trade_max,
        high_quality_confidence: config.high_quality_confidence,
        initial_equity_override: Some(config.initial_balance),
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
    let input = artifact.load_exact_search_input(data_dir)?;
    crate::fx_rates::set_store_selection(data_dir.to_path_buf(), input.anchor_identity().clone())?;
    let base_ohlcv = input.base_frame().ohlcv();
    if base_ohlcv.is_empty() {
        anyhow::bail!("no base bars for {symbol} {base_tf}");
    }
    // Rebuild the SAME multi-TF cube discovery used (warm over the FULL series),
    // then project onto the genes' effective feature set (fail-loud on drift).
    let features =
        crate::project_features_to_effective(input.features(), &artifact.effective_feature_names)?;
    if features.n_samples() != base_ohlcv.len() {
        anyhow::bail!(
            "feature/bar length mismatch {symbol} {base_tf}: {} vs {}",
            features.n_samples(),
            base_ohlcv.len()
        );
    }
    anyhow::ensure!(
        base_ohlcv.timestamp.as_deref() == Some(features.timestamps.as_slice()),
        "feature/base timestamp mismatch {symbol} {base_tf}; refusing row-position alignment"
    );
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
            signals_and_confidence_for_gene_full(&features, base_ohlcv, gene, &eval_config)?;
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
    use neoethos_core::CanonicalTimeframe as T;

    let Ok(timeframe) = tf.to_ascii_uppercase().parse::<T>() else {
        return 1.0;
    };
    match timeframe {
        // Intra-day TFs keep operator's value as-is — they have
        // thousands of bars per month, 15-50 trades is a small
        // fraction of total bar count.
        T::M1 | T::M2 | T::M3 | T::M4 | T::M5 | T::M10 | T::M15 => 1.0,
        // Half-hour: still plenty of bars (~1000/month), small relax
        T::M30 => 0.67,
        // Hourly: ~500 bars/month, 6 trades = ~1.2% of bars
        T::H1 => 0.40,
        // 4h: ~126 bars/month, 3 trades = ~2.4% of bars
        T::H4 => 0.20,
        // Preserve the existing H12 fallback exactly until the independent
        // promotion-gate policy review supplies an evidenced H12 value.
        T::H12 => 1.0,
        // Daily: ~21 bars/month, 2 trades = ~10% of bars (one swing
        // trade every ~2 weeks is realistic for prop-firm passing)
        T::D1 => 0.13,
        // Weekly/monthly: very long-horizon, ANY signal qualifies
        T::W1 => 0.04,
        T::MN1 => 0.02,
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
        best_profit_factor: if measured > 0 {
            best_profit_factor
        } else {
            0.0
        },
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
static BATCH_LEDGER: std::sync::Mutex<Option<BatchRejectionLedger>> = std::sync::Mutex::new(None);

thread_local! {
    /// Per synchronous discovery invocation. Unlike the former Data-layer
    /// process-global working set, this cannot be overwritten by a different
    /// batch running on another worker thread.
    static ACTIVE_STREAMING_BATCH: std::cell::RefCell<(Option<usize>, Option<BatchVerdict>)> =
        const { std::cell::RefCell::new((None, None)) };
}

/// Run one discovery cycle with an exact batch cursor and capture the verdict
/// produced by that cycle. The previous context is restored on success and on
/// unwind, so nested callers and concurrent worker threads remain independent.
pub fn with_streaming_batch_context<T>(
    cursor: usize,
    run: impl FnOnce() -> T,
) -> (T, Option<BatchVerdict>) {
    let previous = ACTIVE_STREAMING_BATCH.with(|context| context.replace((Some(cursor), None)));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
    let current = ACTIVE_STREAMING_BATCH.with(|context| context.replace(previous));
    match outcome {
        Ok(value) => (value, current.1),
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn with_batch_ledger<T>(f: impl FnOnce(&mut BatchRejectionLedger) -> T) -> T {
    let mut guard = BATCH_LEDGER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(guard.get_or_insert_with(BatchRejectionLedger::default))
}

/// Record one batch verdict in the process ledger.
pub fn record_batch_verdict(cursor: usize, verdict: &BatchVerdict) {
    ACTIVE_STREAMING_BATCH.with(|context| {
        let mut context = context.borrow_mut();
        if context.0.is_some() {
            context.1 = Some(*verdict);
        }
    });
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

/// The cursor of the synchronous discovery cycle in force, or `0` for a direct
/// non-streaming caller. Orchestration scopes it around the cycle itself (not
/// merely around feature construction), so logs and verdict evidence retain the
/// actual cursor and concurrent batches cannot overwrite one another.
pub fn streaming_sweep_cursor() -> usize {
    ACTIVE_STREAMING_BATCH.with(|context| context.borrow().0.unwrap_or(0))
}

/// A streaming search: a cursor through the (indicator, period) space, a
/// hardware-derived batch width, and the census of what it abandoned.
pub struct StreamingSearch {
    cursor: usize,
    batch_columns: usize,
    space_len: usize,
    budget_rows: usize,
    replace_base_vocabulary: bool,
    batches_started: usize,
    selection_seed: Option<u64>,
}

impl StreamingSearch {
    /// Size the working set from the machine, against the run's WIDEST frame.
    ///
    /// `budget_rows` is the base timeframe's bar count — the same number
    /// `compute_hpc_feature_frame_sized` is given, for the same reason: the
    /// batch must not be a function of which timeframe is being built, or the
    /// per-TF cube widths diverge and the cube cannot be assembled.
    pub fn new(budget_rows: usize) -> Self {
        Self::with_selection_seed(budget_rows, None)
    }

    /// A fresh job visits a deterministic permutation, without revisiting entries.
    /// Persist each returned batch in the feature recipe; the seed is not replay authority.
    pub fn new_seeded(budget_rows: usize, seed: u64) -> Self {
        Self::with_selection_seed(budget_rows, Some(seed))
    }

    fn with_selection_seed(budget_rows: usize, selection_seed: Option<u64>) -> Self {
        let sizing = neoethos_data::core::hpc_ta::streaming_working_set_sizing(budget_rows);
        let batch_columns = sizing.batch_columns;
        let space_len = sizing.space_len;
        tracing::info!(
            target: "neoethos_search::streaming",
            budget_rows,
            batch_columns,
            available_bytes = sizing.available_bytes,
            max_columns = sizing.max_columns,
            resident_columns = sizing.resident_columns,
            replace_base_vocabulary = sizing.replace_base_vocabulary,
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
            replace_base_vocabulary: sizing.replace_base_vocabulary,
            batches_started: 0,
            selection_seed,
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
        let batch = match self.selection_seed {
            Some(seed) => neoethos_data::core::hpc_ta::search_working_set_batch_seeded(
                self.cursor,
                self.batch_columns,
                self.replace_base_vocabulary,
                seed,
            ),
            None => neoethos_data::core::hpc_ta::search_working_set_batch(
                self.cursor,
                self.batch_columns,
                self.replace_base_vocabulary,
            ),
        };
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
            let features = build_features(&batch)?;
            let (result, verdict) = with_streaming_batch_context(cursor, || run_cycle(&features));
            let result = result?;
            let abandoned = verdict.is_some_and(|verdict| verdict.is_reject());
            tracing::info!(
                target: "neoethos_search::streaming",
                cursor,
                next_cursor = batch.next_cursor,
                space_len = batch.space_len,
                batch_pairs = batch.pairs.len(),
                batch_base_indicators = batch.base_indicator_ids.len(),
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

fn validate_holdout_values_against_scope(
    holdout_scope: &CanonicalSearchArtifactScopeV2,
    timestamps: &[i64],
    row_count: usize,
) -> Result<()> {
    holdout_scope.validate().map_err(anyhow::Error::new)?;
    let window = holdout_scope.evaluated_window();
    anyhow::ensure!(
        matches!(
            window.role(),
            CanonicalSearchWindowRoleV1::Holdout | CanonicalSearchWindowRoleV1::SelectionValidation
        ),
        "post-search validation requires the exact stored calibration or final holdout scope"
    );
    anyhow::ensure!(
        row_count > 0 && timestamps.len() == row_count,
        "held-out validation values require aligned non-empty timestamps"
    );
    let expected_rows = window
        .row_end()
        .checked_sub(window.row_start())
        .and_then(|rows| usize::try_from(rows).ok())
        .ok_or_else(|| anyhow::anyhow!("held-out validation scope row range is invalid"))?;
    anyhow::ensure!(
        expected_rows == row_count,
        "held-out validation values contain {row_count} rows but the stored holdout scope contains {expected_rows}"
    );
    anyhow::ensure!(
        timestamps.first().copied() == Some(window.timestamp_start_ms())
            && timestamps.last().copied() == Some(window.timestamp_end_ms()),
        "held-out validation timestamps do not exactly match the stored holdout scope boundaries"
    );
    Ok(())
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
    walkforward_rejection_reasons(summary, mode).is_empty()
}

fn walkforward_rejection_reasons(
    summary: &WalkforwardSummary,
    mode: DiscoveryMode,
) -> Vec<&'static str> {
    if summary.walk_forward_splits == 0 || summary.splits.is_empty() {
        return vec!["no_walkforward_folds"];
    }
    if summary.walk_forward_splits != summary.splits.len() {
        return vec!["walkforward_split_count_mismatch"];
    }
    if !summary.avg_pnl.is_finite() || summary.splits.iter().any(|s| !s.pnl.is_finite()) {
        return vec!["nonfinite_walkforward_pnl"];
    }
    let mut reasons = Vec::new();
    if summary.avg_pnl <= 0.0 {
        reasons.push("nonpositive_average_pnl");
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
        if positive_frac < 0.60 {
            reasons.push("positive_fold_fraction_below_60_percent");
        }
        return reasons;
    }
    // PropFirm / Strict: demand full prop-firm robustness across EVERY window.
    if summary.any_daily_loss_breach {
        reasons.push("daily_loss_breach");
    }
    if summary.any_consistency_violation {
        reasons.push("consistency_violation");
    }
    if summary.any_trade_limit_violation {
        reasons.push("trade_limit_violation");
    }
    if !summary.all_min_trading_days_ok {
        reasons.push("minimum_trading_days_failed");
    }
    reasons
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
    effective_smc_gate_threshold: f64,
    months: &[i64],
    days: &[i64],
    pbo_candidates: &[Gene],
    population_execution_run: &crate::population_execution_evidence_v1::ExactPopulationExecutionRunV1<'_>,
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
    let full_indicators = features.to_dense_samples_major()?.values.reversed_axes();
    let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
        build_smc_arrays(features, ohlcv)?;
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
            crate::genetic::search_engine::validation_genes_population_gathered_exact(
                full_indicators.view(),
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
                population_execution_run,
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
                crate::genetic::search_engine::validation_genes_population_gathered_exact(
                    full_indicators.view(),
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
                    population_execution_run,
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

struct WalkforwardSelectionCandidate {
    // Stable within this finalization call, unlike the human-readable strategy_id.
    candidate_idx: usize,
    gene: Gene,
    signals: Vec<i8>,
    prop_firm_pass_rate: Option<f64>,
}

/// Journaling is diagnostic, never candidate admission. When enabled, every
/// final selection comes first; remaining slots retain the existing quality
/// order. A cap smaller than the selected set (including zero) cannot omit a
/// selected ledger. Disabling logging still means no journals at all.
fn plan_diagnostic_candidates<'a>(
    portfolio_indices: &[usize],
    portfolio: &'a [Gene],
    ranked_candidates: &'a [(usize, Gene)],
    ranked_quality: &[(usize, bool)],
    enabled: bool,
    cap: usize,
) -> Result<Vec<(usize, &'a Gene, bool)>> {
    if !enabled {
        return Ok(Vec::new());
    }
    anyhow::ensure!(
        portfolio_indices.len() == portfolio.len(),
        "diagnostic portfolio candidate identities are not aligned"
    );
    let by_index: HashMap<_, _> = ranked_candidates
        .iter()
        .map(|(i, gene)| (*i, gene))
        .collect();
    let lanes: HashMap<_, _> = ranked_quality.iter().copied().collect();
    anyhow::ensure!(
        by_index.len() == ranked_candidates.len() && lanes.len() == ranked_quality.len(),
        "diagnostic candidate indices are duplicated"
    );
    let mut seen = HashSet::new();
    let mut planned = Vec::new();
    for (&index, gene) in portfolio_indices.iter().zip(portfolio) {
        anyhow::ensure!(
            seen.insert(index),
            "duplicate selected diagnostic candidate {index}"
        );
        let original = by_index.get(&index).ok_or_else(|| {
            anyhow::anyhow!(
                "selected diagnostic candidate {index} is absent from the ranked population"
            )
        })?;
        ValidationStrategyIdentityV2::from_gene(original)?.validate_against(gene)?;
        // A legacy best-effort fallback can have failed the quality screen;
        // its ledger is not an opportunistic-quality pass.
        planned.push((index, gene, lanes.get(&index).copied().unwrap_or(false)));
    }
    let limit = cap.max(planned.len());
    for &(index, opportunistic) in ranked_quality {
        if planned.len() >= limit {
            break;
        }
        if seen.insert(index) {
            let gene = by_index.get(&index).ok_or_else(|| {
                anyhow::anyhow!(
                    "diagnostic quality candidate {index} is absent from the ranked population"
                )
            })?;
            planned.push((index, *gene, opportunistic));
        }
    }
    Ok(planned)
}

fn replay_diagnostic_candidates(
    candidates: Vec<(usize, &Gene, bool)>,
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    evaluation: &EvaluationConfig,
    smc: &SmcGateArrays,
    selected_signals: &HashMap<usize, &[i8]>,
) -> Result<Vec<(usize, LoggedStrategyTrades)>> {
    let resolver = GeneEvalSettingsResolver::for_slice(
        config,
        candidates.iter().map(|(_, gene, _)| *gene),
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
    )?;
    crate::post_ga::map_bounded(
        candidates,
        features.n_samples(),
        |(index, gene, opportunistic)| {
            let signals = match selected_signals.get(&index) {
                Some(signals) => Cow::Borrowed(*signals),
                None => Cow::Owned(signals_for_gene_full_with_smc(
                    features, gene, evaluation, smc,
                )?),
            };
            let confidences =
                account_sizing_confidences(features, gene, evaluation, smc, &signals)?;
            let trades = simulate_trades_with_confidence_core(
                &ohlcv.close,
                &ohlcv.high,
                &ohlcv.low,
                &features.timestamps,
                &signals,
                &confidences,
                &resolver.settings_for_gene(gene),
            )?;
            Ok((
                index,
                LoggedStrategyTrades {
                    strategy_id: gene.strategy_id.clone(),
                    opportunistic,
                    trades,
                },
            ))
        },
    )
}

fn candidate_equity_curve(
    candidate_idx: usize,
    initial_balance: f64,
    logged_candidate_indices: &[usize],
    logged_trades: &[LoggedStrategyTrades],
    replay: impl FnOnce() -> Result<Vec<Trade>>,
) -> Result<Vec<f64>> {
    anyhow::ensure!(
        logged_candidate_indices.len() == logged_trades.len(),
        "logged trade candidate identities are not aligned"
    );
    let mut matching = logged_candidate_indices
        .iter()
        .zip(logged_trades)
        .filter(|(idx, _)| **idx == candidate_idx);
    if let Some((_, logged)) = matching.next() {
        anyhow::ensure!(
            matching.next().is_none(),
            "duplicate logged trade candidate identity {candidate_idx}"
        );
        Ok(crate::post_ga::trade_equity_curve(
            initial_balance,
            &logged.trades,
        ))
    } else {
        let trades = replay()?;
        Ok(crate::post_ga::trade_equity_curve(initial_balance, &trades))
    }
}

fn restore_candidate_quality_curve(
    candidate_idx: usize,
    curve: &[f64],
    quality_candidate_indices: &[usize],
    quality_metrics: &mut [StrategyMetrics],
) -> Result<()> {
    anyhow::ensure!(
        quality_candidate_indices.len() == quality_metrics.len(),
        "quality candidate identities are not aligned"
    );
    let mut found = false;
    // Validate every matching row before modifying any, including a duplicate
    // report row added by the explicitly flagged best-effort fallback.
    for (idx, metrics) in quality_candidate_indices.iter().zip(quality_metrics.iter()) {
        if *idx == candidate_idx {
            found = true;
            if metrics.total_trades > 0 && metrics.equity_curve.is_empty() {
                anyhow::ensure!(
                    metrics.total_trades.checked_add(1) == Some(curve.len()),
                    "restored equity curve does not match the completed quality replay for candidate {candidate_idx}"
                );
            }
        }
    }
    anyhow::ensure!(
        found,
        "quality candidate identity {candidate_idx} is absent"
    );
    for (idx, metrics) in quality_candidate_indices.iter().zip(quality_metrics) {
        if *idx == candidate_idx && metrics.total_trades > 0 && metrics.equity_curve.is_empty() {
            metrics.equity_curve = curve.to_vec();
        }
    }
    Ok(())
}

#[cfg(test)]
fn evaluate_walkforward_batches(
    candidate_count: usize,
    batch_width: usize,
    mut evaluate: impl FnMut(std::ops::Range<usize>) -> Result<Vec<WalkforwardSummary>>,
) -> Result<Vec<WalkforwardSummary>> {
    anyhow::ensure!(batch_width > 0, "walk-forward batch width must be positive");
    let mut summaries = Vec::with_capacity(candidate_count);
    consume_walkforward_batches(
        candidate_count,
        |_| Ok(batch_width),
        &mut evaluate,
        |_, batch| {
            summaries.extend(batch);
            Ok(())
        },
    )?;
    Ok(summaries)
}

/// Recheck admission after the previous wave has been consumed. The sink owns
/// each completed wave, so the broad candidate route need not retain any of its
/// daily-return curves. An error never publishes a partial/misaligned wave.
fn consume_walkforward_batches(
    candidate_count: usize,
    mut batch_width: impl FnMut(usize) -> Result<usize>,
    mut evaluate: impl FnMut(std::ops::Range<usize>) -> Result<Vec<WalkforwardSummary>>,
    mut completed_batch: impl FnMut(std::ops::Range<usize>, Vec<WalkforwardSummary>) -> Result<()>,
) -> Result<()> {
    let mut start = 0;
    while start < candidate_count {
        let width = batch_width(candidate_count - start)?;
        anyhow::ensure!(width > 0, "walk-forward batch width must be positive");
        let end = start.saturating_add(width).min(candidate_count);
        let range = start..end;
        let batch = evaluate(range.clone())?;
        anyhow::ensure!(
            batch.len() == end - start,
            "walk-forward batch summary count mismatch"
        );
        completed_batch(range, batch)?;
        start = end;
    }
    Ok(())
}

/// Keep candidate values together through both gates. WF is evaluated for the
/// entire ranked input before portfolio capacity can stop correlation work.
#[cfg(test)]
fn select_walkforward_diverse_candidates(
    candidates: Vec<WalkforwardSelectionCandidate>,
    summaries: &[WalkforwardSummary],
    mode: DiscoveryMode,
    portfolio_size: usize,
    corr_threshold: f64,
    census: &mut crate::funnel_profile::DiscoveryCandidateCensus,
) -> Result<Vec<WalkforwardSelectionCandidate>> {
    let verdicts = summaries
        .iter()
        .map(|summary| WalkforwardVerdict {
            tested: summary.walk_forward_splits > 0,
            passed: walkforward_summary_passed(summary, mode),
        })
        .collect::<Vec<_>>();
    select_walkforward_diverse_candidates_with_signals(
        candidates,
        &verdicts,
        None,
        portfolio_size,
        corr_threshold,
        census,
        |_| Ok(Vec::new()),
    )
}

fn select_walkforward_diverse_candidates_with_signals(
    candidates: Vec<WalkforwardSelectionCandidate>,
    summaries: &[WalkforwardVerdict],
    profitable_calibration_genes: Option<&HashSet<String>>,
    portfolio_size: usize,
    corr_threshold: f64,
    census: &mut crate::funnel_profile::DiscoveryCandidateCensus,
    mut load_signals: impl FnMut(&Gene) -> Result<Vec<i8>>,
) -> Result<Vec<WalkforwardSelectionCandidate>> {
    anyhow::ensure!(
        candidates.len() == summaries.len(),
        "candidate/WF summary count mismatch"
    );
    census.walkforward_tested = summaries.iter().filter(|s| s.tested).count();
    census.walkforward_passed = summaries.iter().filter(|s| s.passed).count();
    census.walkforward_failed = census
        .walkforward_tested
        .saturating_sub(census.walkforward_passed);
    census.walkforward_not_tested = census
        .validation_candidates_admitted
        .saturating_sub(census.walkforward_tested);
    let mut selected: Vec<WalkforwardSelectionCandidate> = Vec::new();
    for (mut candidate, summary) in candidates.into_iter().zip(summaries) {
        if !summary.passed {
            continue;
        }
        if let Some(profitable) = profitable_calibration_genes {
            if !profitable.contains(&stable_json_hash(&candidate.gene)?) {
                continue;
            }
        }
        if selected.len() >= portfolio_size {
            census.portfolio_capacity_not_selected += 1;
            continue;
        }
        if candidate.signals.is_empty() {
            candidate.signals = load_signals(&candidate.gene)?;
        }
        census.correlation_tested += 1;
        let rankable = portfolio_signal_is_correlation_rankable_v1(&candidate.signals);
        let diverse = rankable
            && selected.iter().all(|existing| {
                matches!(
                    pairwise_portfolio_correlation_decision_v1(
                        &candidate.signals,
                        &existing.signals,
                        corr_threshold
                    ),
                    PortfolioCorrelationDecisionV1::Accept
                )
            });
        if diverse {
            selected.push(candidate);
        } else {
            census.rejected_by_correlation += 1;
        }
    }
    census.portfolio_selected = selected.len();
    Ok(selected)
}

/// The parallel chunk has joined before this is called. Publish completed
/// full-IS replays even if a later operation in that chunk returned an error.
fn publish_completed_quality_chunk<T>(
    joined_chunk: Result<T>,
    completed_replays: &std::sync::atomic::AtomicUsize,
    census: &mut crate::funnel_profile::DiscoveryCandidateCensus,
    progress: &mut impl FnMut(DiscoveryProgress),
) -> Result<T> {
    census.quality_evaluated = completed_replays.load(std::sync::atomic::Ordering::Relaxed);
    progress(DiscoveryProgress::CandidateCensusUpdated {
        census: census.clone(),
    });
    joined_chunk
}

/// Describe actual retained membership, without calling skipped or retained-all
/// robustness verdicts a pass. Diagnostic fallback genes are not selected ones.
fn publish_portfolio_after_robustness(
    retained_count: usize,
    fallback_mode: bool,
    census: &mut crate::funnel_profile::DiscoveryCandidateCensus,
    funnel: &mut crate::funnel_profile::FunnelProfile,
    progress: &mut impl FnMut(DiscoveryProgress),
) {
    let before = census.portfolio_selected;
    let retained = if fallback_mode { 0 } else { retained_count };
    let removed = before.saturating_sub(retained);
    census.robustness_removed = Some(removed);
    census.portfolio_selected = retained;
    funnel.record_stage("portfolio_after_robustness", before, retained);
    if removed > 0 {
        funnel.add_reject_reason("portfolio_after_robustness", "robustness_removed", removed);
    }
    funnel.candidate_census = Some(census.clone());
    progress(DiscoveryProgress::CandidateCensusUpdated {
        census: census.clone(),
    });
}

/// Retain detailed summaries only for the final-artifact caller. The broad
/// candidate queue uses `discovery_walkforward_verdicts` instead.
fn discovery_walkforward_summaries<F>(
    portfolio: &[Gene],
    portfolio_signals: &[Vec<i8>],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
    population_execution_run: &crate::population_execution_evidence_v1::ExactPopulationExecutionRunV1<'_>,
    mut completed_batch: F,
) -> Result<Vec<WalkforwardSummary>>
where
    F: FnMut(&[WalkforwardSummary]),
{
    let mut summaries = Vec::with_capacity(portfolio.len());
    discovery_walkforward_batches(
        portfolio,
        Some(portfolio_signals),
        features,
        ohlcv,
        config,
        effective_smc_gate_threshold,
        population_execution_run,
        |_, batch| {
            completed_batch(&batch);
            summaries.extend(batch);
            Ok(())
        },
    )?;
    Ok(summaries)
}

/// This is a projection of the completed mode-specific WF result, not a new
/// fitness or selection rule. No curves/signals are retained in this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WalkforwardVerdict {
    tested: bool,
    passed: bool,
}

impl WalkforwardVerdict {
    fn from_summary(summary: &WalkforwardSummary, mode: DiscoveryMode) -> Self {
        Self {
            tested: summary.walk_forward_splits > 0,
            passed: walkforward_summary_passed(summary, mode),
        }
    }
}

fn walkforward_selection_trial(
    candidate_archive_index: usize,
    gene: &Gene,
    summary: &WalkforwardSummary,
    mode: DiscoveryMode,
) -> Result<crate::funnel_profile::WalkforwardSelectionTrial> {
    use crate::funnel_profile::{WalkforwardFoldDiagnostic, WalkforwardSelectionTrial};
    let verdict = WalkforwardVerdict::from_summary(summary, mode);
    Ok(WalkforwardSelectionTrial {
        candidate_archive_index,
        strategy_identity: ValidationStrategyIdentityV2::from_gene(gene)?,
        tested: verdict.tested,
        passed: verdict.passed,
        reported_splits: summary.walk_forward_splits,
        avg_pnl: summary.avg_pnl.is_finite().then_some(summary.avg_pnl),
        positive_folds: summary.splits.iter().filter(|s| s.pnl > 0.0).count(),
        trading_folds: summary.splits.iter().filter(|s| s.trades > 0).count(),
        rejection_reasons: walkforward_rejection_reasons(summary, mode)
            .into_iter()
            .map(str::to_owned)
            .collect(),
        folds: summary
            .splits
            .iter()
            .map(|s| WalkforwardFoldDiagnostic {
                split: s.split,
                trades: s.trades,
                pnl: s.pnl.is_finite().then_some(s.pnl),
                daily_loss_breach: s.daily_loss_breach,
                consistency_violation: s.consistency_violation,
                trade_limit_violation: s.trade_limit_violation,
                min_trading_days_ok: s.min_trading_days_ok,
            })
            .collect(),
    })
}

fn discovery_walkforward_verdicts<F>(
    portfolio: &[Gene],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
    population_execution_run: &crate::population_execution_evidence_v1::ExactPopulationExecutionRunV1<'_>,
    mut completed_batch: F,
) -> Result<Vec<WalkforwardVerdict>>
where
    F: FnMut(std::ops::Range<usize>, &[WalkforwardSummary], &[WalkforwardVerdict]) -> Result<()>,
{
    let mut verdicts = Vec::with_capacity(portfolio.len());
    discovery_walkforward_batches(
        portfolio,
        None,
        features,
        ohlcv,
        config,
        effective_smc_gate_threshold,
        population_execution_run,
        |range, summaries| {
            let batch = summaries
                .iter()
                .map(|summary| WalkforwardVerdict::from_summary(summary, config.mode))
                .collect::<Vec<_>>();
            completed_batch(range, &summaries, &batch)?;
            verdicts.extend(batch);
            Ok(())
        },
    )?;
    Ok(verdicts)
}

/// Share the full feature matrix, SMC and adaptive bases once. Signals and
/// confidence arrays exist only within each headroom-admitted wave, except for
/// the final-artifact caller's already retained signal tapes. The shared matrix
/// allocation has its own Data admission and precedes candidate-wave admission;
/// these measured-headroom checks are not OS memory reservations.
fn discovery_walkforward_batches<F>(
    portfolio: &[Gene],
    portfolio_signals: Option<&[Vec<i8>]>,
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
    population_execution_run: &crate::population_execution_evidence_v1::ExactPopulationExecutionRunV1<'_>,
    completed_batch: F,
) -> Result<()>
where
    F: FnMut(std::ops::Range<usize>, Vec<WalkforwardSummary>) -> Result<()>,
{
    if portfolio.is_empty() {
        return Ok(());
    }
    let n = validation_row_count(features, ohlcv)?;
    if let Some(signals) = portfolio_signals {
        anyhow::ensure!(
            signals.len() == portfolio.len() && signals.iter().all(|signals| signals.len() == n),
            "walk-forward candidates/signals are not exactly aligned"
        );
    }
    // Stop before disk-backed full-history materialization, not only before
    // the first candidate wave after that shared preparation has completed.
    crate::post_ga::check_cancel()?;
    let parallel_cpu_windows = match population_execution_run
        .population_auto_sizing_primitives_v1()?
        .route
    {
        crate::PopulationAutoSizingRouteV1::CpuExplicitResearch { .. }
        | crate::PopulationAutoSizingRouteV1::CpuNoCompatibleGpu { .. } => true,
        crate::PopulationAutoSizingRouteV1::NativeCuda { .. } => false,
    };
    let (months, days) = month_day_indices(&features.timestamps);
    let timestamps = &features.timestamps[..n];
    let embargo_bars = embargo_bars_from_timestamps(timestamps, config.embargo_minutes);
    let wf_full_indicators = features.to_dense_samples_major()?.values.reversed_axes();
    let wf_smc = SmcGateArrays::build(features, ohlcv)?;
    // Adaptive bases are also shared across chunks, never recalculated once
    // per candidate batch on the same historical window.
    let wf_resolver = GeneEvalSettingsResolver::for_slice(
        config,
        portfolio.iter(),
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
    )?;
    let wf_eval_config = config
        .evaluation_config_with_smc_gate(ohlcv.close.last().copied(), effective_smc_gate_threshold);
    consume_walkforward_batches(
        portfolio.len(),
        |remaining| {
            crate::post_ga::check_cancel()?;
            crate::post_ga::post_ga_batch_width(n, remaining)
        },
        |range| {
            let portfolio = &portfolio[range.clone()];
            if crate::genetic::search_engine::search_cancel_requested() {
                anyhow::bail!(
                    "__DISCOVERY_CANCELLED__ discovery cancelled during candidate walk-forward validation"
                );
            }
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
            let wf_gene_settings: Vec<crate::eval::BacktestSettings> = portfolio
                .iter()
                .map(|gene| wf_resolver.settings_for_gene(gene))
                .collect();
            let (wave_signals, wf_confidences): (std::borrow::Cow<'_, [Vec<i8>]>, Vec<Vec<f64>>) =
                if let Some(signals) = portfolio_signals {
                    let signals = &signals[range];
                    let confidences = portfolio
                        .par_iter()
                        .zip(signals.par_iter())
                        .map(|(gene, signals)| {
                            account_sizing_confidences(
                                features,
                                gene,
                                &wf_eval_config,
                                &wf_smc,
                                signals,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    (std::borrow::Cow::Borrowed(signals), confidences)
                } else {
                    let paired = portfolio
                        .par_iter()
                        .map(|gene| {
                            let pair = signals_and_confidence_for_gene_full_with_smc(
                                features,
                                gene,
                                &wf_eval_config,
                                &wf_smc,
                            )?;
                            anyhow::ensure!(
                                pair.0.len() == n && pair.1.len() == n,
                                "walk-forward signals/confidences for '{}' are not exactly aligned",
                                gene.strategy_id
                            );
                            Ok(pair)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (signals, confidences) = paired.into_iter().unzip();
                    (std::borrow::Cow::Owned(signals), confidences)
                };
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
                    confidences_per_gene: &wf_confidences,
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
                wave_signals.as_ref(),
                parallel_cpu_windows,
                |test_start, end| {
                    // ONE GPU population launch over the whole portfolio on this
                    // contiguous split window. Serialize the device launch behind
                    // GPU_LAUNCH_LOCK so any outer parallelism never spins up N GPU
                    // clients → VRAM × N → OOM. A compiled GPU feature must not
                    // serialize the separately sealed CPU route on that device lock.
                    #[cfg(feature = "gpu")]
                    let _gpu_guard = (!parallel_cpu_windows)
                        .then(|| GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner()));
                    crate::genetic::search_engine::validation_genes_population_window_exact(
                        &wf_gene_pack,
                        wf_full_indicators.view(),
                        wf_smc.rows(),
                        &ohlcv.close,
                        &ohlcv.high,
                        &ohlcv.low,
                        &months,
                        &days,
                        timestamps,
                        test_start,
                        end,
                        population_execution_run,
                    )
                    // The exact CPU provider returns metrics and the actual risk-sized
                    // ledger together, so diagnostics consume that one simulation.
                },
            )?;
            anyhow::ensure!(
                walkforward_summaries.len() == portfolio.len(),
                "walk-forward returned {} summaries for {} candidates",
                walkforward_summaries.len(),
                portfolio.len()
            );
            Ok(walkforward_summaries)
        },
        completed_batch,
    )
}

#[cfg(test)]
mod walkforward_wave_tests {
    use super::*;
    use crate::validation::WalkforwardSplitResult;
    use std::cell::Cell;

    #[test]
    fn precancelled_walkforward_never_materializes_vortex_features() -> Result<()> {
        use neoethos_data::core::feature_run_lease::FeatureRunLease;
        use neoethos_data::core::vortex_feature_store::{
            VortexFeatureStore, VortexFeatureStoreOptions,
        };
        use std::sync::{Arc, atomic::AtomicBool};

        const CHILD: &str = "NEOETHOS_TEST_WALKFORWARD_PRE_CANCEL_CHILD";
        const TEST: &str = "discovery::walkforward_wave_tests::precancelled_walkforward_never_materializes_vortex_features";
        const COMPLETED: &str = "walkforward-pre-cancel-no-materialization-pass";
        if std::env::var_os(CHILD).as_deref() != Some(std::ffi::OsStr::new("1")) {
            // SEARCH_CANCEL is process-global; never change it in the parent
            // harness while unrelated Search tests can still be executing.
            let temp = tempfile::tempdir()?;
            let stdout_path = temp.path().join("child.stdout");
            let stderr_path = temp.path().join("child.stderr");
            let mut child = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .stdout(std::fs::File::create(&stdout_path)?)
                .stderr(std::fs::File::create(&stderr_path)?)
                .spawn()?;
            let started = std::time::Instant::now();
            let status = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break status,
                    Ok(None) if started.elapsed() < std::time::Duration::from_secs(60) => {
                        std::thread::sleep(std::time::Duration::from_millis(25));
                    }
                    outcome => {
                        let _ = child.kill();
                        let _ = child.wait();
                        let stdout = std::fs::read_to_string(&stdout_path)?;
                        let stderr = std::fs::read_to_string(&stderr_path)?;
                        anyhow::bail!(
                            "walk-forward cancellation child did not finish: {outcome:?}\n{stdout}\n{stderr}"
                        );
                    }
                }
            };
            let stdout = std::fs::read_to_string(stdout_path)?;
            let stderr = std::fs::read_to_string(stderr_path)?;
            print!("{stdout}");
            eprint!("{stderr}");
            anyhow::ensure!(
                status.success()
                    && stdout.contains("test result: ok. 1 passed; 0 failed;")
                    && stdout.lines().any(|line| line.contains(COMPLETED)),
                "walk-forward cancellation child did not execute its complete regression: {status}"
            );
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let raw = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let columns = raw.project_columns(
            &(0..raw.n_features()).collect::<Vec<_>>(),
            0..raw.n_samples(),
        )?;
        let store = VortexFeatureStore::create(
            Arc::new(FeatureRunLease::create(temp.path(), "walkforward-cancel")?),
            &raw.timestamps,
            &columns.columns,
            VortexFeatureStoreOptions {
                chunk_rows: 8,
                decoded_cache_bytes: 0,
            },
        )?;
        let features = FeatureFrame::from_vortex(
            raw.timestamps.clone(),
            Arc::clone(&store),
            raw.plan().clone(),
            raw.provenance().clone(),
        )?;
        let anchor = features.provenance().bindings()[0].dataset_identity();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &features)?;
        let input =
            CanonicalSearchRunInputV2::new_for_test_values(receipt.clone(), &features, &ohlcv)?;
        let scope = CanonicalSearchArtifactScopeV2::from_run_input(
            CanonicalSearchWindowRoleV1::DiscoveryInput,
            &input,
        )?;
        let assumption_hash = "a".repeat(64);
        let contract = crate::CanonicalTrendbarResearchExecutionContractV3::new(
            receipt,
            crate::CanonicalTrendbarResearchCostAssumptionsV2 {
                symbol: "EURUSD",
                account_currency: "USD",
                assumption_source_id: "neoethos.test.walkforward-cancellation.v1",
                assumption_source_sha256: &assumption_hash,
                pip_size: 0.0001,
                pip_value_per_lot: 10.0,
                full_spread_pips_assumption: 1.2,
                slippage_pips_per_fill_assumption: 0.1,
                commission_account_per_lot_per_fill_assumption: 3.5,
                swap_long_pips_per_day: -0.2,
                swap_short_pips_per_day: -0.1,
                pnl_conversion_fee_rate: 0.0,
            },
        )?;
        let admission =
            crate::SealedStrictDiscoveryDeviceAdmissionV1::from_explicit_canonical_cpu_research_v1(
                &contract,
            )?;
        let run = crate::population_execution_evidence_v1::begin_exact_population_execution_run_v1(
            admission, &scope, &features, &ohlcv,
        )?;

        // Seal real values first, then make only this owned shard unavailable.
        // With caching disabled, missing payload is an independent negative
        // control: removing the early Stop check must expose this I/O error.
        let payload = store.path().canonicalize()?;
        anyhow::ensure!(payload.starts_with(temp.path().canonicalize()?));
        std::fs::remove_file(&payload)?;
        assert_eq!(store.cache_stats().resident_bytes, 0);
        let misses_before_failure = store.cache_stats().misses;
        let io_error = features.to_dense_samples_major().unwrap_err();
        assert!(!io_error.to_string().contains("__DISCOVERY_CANCELLED__"));
        assert!(store.cache_stats().misses > misses_before_failure);
        let misses_before = store.cache_stats().misses;
        crate::genetic::search_engine::set_search_cancel(Some(Arc::new(AtomicBool::new(true))));
        let completed = Cell::new(0);
        let result = discovery_walkforward_batches(
            &[Gene::default()],
            None,
            &features,
            &ohlcv,
            &DiscoveryConfig::default(),
            0.0,
            &run,
            |range, _| {
                completed.set(range.end);
                Ok(())
            },
        );
        crate::genetic::search_engine::set_search_cancel(None);
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("__DISCOVERY_CANCELLED__"),
            "{error:#}"
        );
        assert_eq!(completed.get(), 0);
        assert_eq!(store.cache_stats().misses, misses_before);
        println!("{COMPLETED}");
        Ok(())
    }

    fn summary(pnls: &[f64], daily_loss_breach: bool) -> WalkforwardSummary {
        WalkforwardSummary {
            walk_forward_splits: pnls.len(),
            avg_pnl: if pnls.is_empty() {
                0.0
            } else {
                pnls.iter().sum::<f64>() / pnls.len() as f64
            },
            avg_win_rate: 0.5,
            avg_max_dd: 1.0,
            avg_max_consec_losses: 1.0,
            avg_daily_min_dd: -1.0,
            avg_max_daily_loss: 1.0,
            any_daily_loss_breach: daily_loss_breach,
            any_consistency_violation: false,
            any_trade_limit_violation: false,
            all_min_trading_days_ok: true,
            splits: pnls
                .iter()
                .enumerate()
                .map(|(split, &pnl)| WalkforwardSplitResult {
                    split,
                    trades: 2,
                    pnl,
                    win_rate: 0.5,
                    max_dd: 1.0,
                    max_consec_losses: 1,
                    daily_min_dd: -1.0,
                    max_daily_loss: 1.0,
                    daily_loss_breach,
                    consistency_violation: false,
                    trade_limit_violation: false,
                    min_trading_days_ok: true,
                    daily_returns: vec![pnl; 256],
                    max_daily_dd_pct: 1.0,
                    prop_compliant: !daily_loss_breach,
                })
                .collect(),
        }
    }

    #[test]
    fn lightweight_verdict_preserves_no_test_and_mode_specific_math() {
        let no_splits = summary(&[], false);
        let risky_winner = summary(&[30.0, 20.0, -10.0], true);
        let minority_positive = summary(&[100.0, -10.0, -10.0], false);
        let loss = summary(&[-30.0, 10.0, 10.0], false);
        assert_eq!(
            WalkforwardVerdict::from_summary(&no_splits, DiscoveryMode::Risky),
            WalkforwardVerdict {
                tested: false,
                passed: false
            }
        );
        assert_eq!(
            WalkforwardVerdict::from_summary(&risky_winner, DiscoveryMode::Risky),
            WalkforwardVerdict {
                tested: true,
                passed: true
            }
        );
        assert_eq!(
            WalkforwardVerdict::from_summary(&risky_winner, DiscoveryMode::PropFirm),
            WalkforwardVerdict {
                tested: true,
                passed: false
            }
        );
        assert_eq!(
            WalkforwardVerdict::from_summary(&minority_positive, DiscoveryMode::Risky),
            WalkforwardVerdict {
                tested: true,
                passed: false
            }
        );
        assert_eq!(
            WalkforwardVerdict::from_summary(&loss, DiscoveryMode::Risky),
            WalkforwardVerdict {
                tested: true,
                passed: false
            }
        );
    }

    #[test]
    fn walkforward_diagnostics_preserve_failed_folds_without_return_tapes() -> Result<()> {
        let gene = Gene {
            strategy_id: "failed-wf-candidate".to_owned(),
            ..Gene::default()
        };
        let mut sparse = summary(&[50.0, 0.0, 0.0, -10.0, 0.0], false);
        for split in &mut sparse.splits {
            if split.pnl == 0.0 {
                split.trades = 0;
            }
        }
        let trial = walkforward_selection_trial(7, &gene, &sparse, DiscoveryMode::Risky)?;
        assert!(trial.tested);
        assert!(!trial.passed);
        assert_eq!(trial.avg_pnl, Some(8.0));
        assert_eq!(trial.positive_folds, 1);
        assert_eq!(trial.trading_folds, 2);
        assert_eq!(trial.folds.iter().map(|s| s.trades).sum::<usize>(), 4);
        assert_eq!(
            trial.rejection_reasons,
            ["positive_fold_fraction_below_60_percent"]
        );
        let wire = serde_json::to_string(&trial)?;
        assert!(!wire.contains("daily_returns"));
        let restored: crate::funnel_profile::WalkforwardSelectionTrial =
            serde_json::from_str(&wire)?;
        assert_eq!(restored.candidate_archive_index, 7);
        restored.strategy_identity.validate_against(&gene)?;
        assert_eq!(restored.folds.len(), 5);
        assert_eq!(restored.folds[3].pnl, Some(-10.0));
        let changed = Gene {
            long_threshold: gene.long_threshold + 0.1,
            ..gene
        };
        assert!(
            restored
                .strategy_identity
                .validate_against(&changed)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn walkforward_rejects_invalid_profit_and_incomplete_fold_payloads() -> Result<()> {
        let gene = Gene {
            strategy_id: "invalid-wf".to_owned(),
            ..Gene::default()
        };
        let mut invalid = summary(&[10.0, 10.0, 10.0, 10.0, f64::NAN], false);
        invalid.avg_pnl = 10.0;
        // Four positive folds and a positive average previously admitted this
        // malformed result despite the fifth fold having no finite PnL.
        for mode in [
            DiscoveryMode::Risky,
            DiscoveryMode::PropFirm,
            DiscoveryMode::Strict,
        ] {
            assert!(!walkforward_summary_passed(&invalid, mode));
        }
        let trial = walkforward_selection_trial(0, &gene, &invalid, DiscoveryMode::Risky)?;
        assert_eq!(trial.rejection_reasons, ["nonfinite_walkforward_pnl"]);
        assert_eq!(trial.folds[4].pnl, None);
        assert!(serde_json::to_string(&trial)?.contains("\"pnl\":null"));
        let mut incomplete = summary(&[10.0], false);
        incomplete.walk_forward_splits = 20;
        assert_eq!(
            walkforward_rejection_reasons(&incomplete, DiscoveryMode::Risky),
            ["walkforward_split_count_mismatch"]
        );
        incomplete.splits.clear();
        assert!(!walkforward_summary_passed(
            &incomplete,
            DiscoveryMode::Strict
        ));
        Ok(())
    }

    #[test]
    fn waves_release_detailed_payloads_before_readmitting_and_keep_order() -> Result<()> {
        let completed = Cell::new(0);
        let mut widths = [3, 1, 2].into_iter();
        let mut verdicts = Vec::new();
        consume_walkforward_batches(
            6,
            |remaining| {
                assert_eq!(remaining, 6 - completed.get());
                Ok(widths.next().expect("exactly three waves"))
            },
            |range| {
                assert_eq!(range.start, completed.get());
                Ok(range
                    .map(|idx| summary(&[idx as f64 - 2.0], false))
                    .collect())
            },
            |range, batch| {
                assert_eq!(batch.len(), range.len());
                assert!(batch.iter().all(|s| s.splits[0].daily_returns.len() == 256));
                verdicts.extend(
                    batch
                        .into_iter()
                        .map(|s| WalkforwardVerdict::from_summary(&s, DiscoveryMode::Risky)),
                );
                // The owned detailed summaries have been consumed/dropped here;
                // admission of the next wave observes only the compact output.
                completed.set(range.end);
                Ok(())
            },
        )?;
        assert_eq!(completed.get(), 6);
        assert_eq!(
            verdicts.iter().map(|v| v.passed).collect::<Vec<_>>(),
            [false, false, false, true, true, true]
        );
        assert!(verdicts.iter().all(|v| v.tested));
        assert!(widths.next().is_none());
        Ok(())
    }

    #[test]
    fn final_artifact_collection_preserves_all_curves_independent_of_wave_width() -> Result<()> {
        let expected = (0..5)
            .map(|idx| summary(&[idx as f64, -1.0, 3.0], false))
            .collect::<Vec<_>>();
        for width in [1, 2, 8] {
            let retained = evaluate_walkforward_batches(expected.len(), width, |range| {
                Ok(expected[range].to_vec())
            })?;
            assert_eq!(retained, expected);
        }
        Ok(())
    }

    #[test]
    fn failed_readmission_or_wave_does_not_invent_completed_verdicts() {
        let published = Cell::new(0);
        let result = consume_walkforward_batches(
            3,
            |_| {
                if published.get() == 0 {
                    Ok(2)
                } else {
                    anyhow::bail!("no full-history worker fits")
                }
            },
            |range| Ok(range.map(|_| summary(&[1.0], false)).collect()),
            |range, _| {
                published.set(range.end);
                Ok(())
            },
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no full-history worker fits")
        );
        assert_eq!(published.get(), 2);

        for (width, count) in [(0, 0), (2, 1)] {
            let result = consume_walkforward_batches(
                3,
                |_| Ok(width),
                |_| Ok((0..count).map(|_| summary(&[1.0], false)).collect()),
                |_, _| {
                    panic!("invalid wave must not publish");
                },
            );
            assert!(result.is_err());
        }
    }
}

fn build_discovery_validation_artifacts(
    portfolio: &[Gene],
    portfolio_signals: &[Vec<i8>],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    selection_scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
    pbo_candidates: &[Gene],
    trials_tested: usize,
    population_execution_run: &crate::population_execution_evidence_v1::ExactPopulationExecutionRunV1<'_>,
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
    if portfolio_signals.len() != portfolio.len()
        || portfolio_signals.iter().any(|signals| signals.len() != n)
    {
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
    let (months, days) = month_day_indices(&features.timestamps);
    let timestamps = &features.timestamps[..n];

    let mut canonical_backtest_artifacts = Vec::with_capacity(portfolio.len());
    let mut walkforward_validation_artifacts = Vec::with_capacity(portfolio.len());
    let mut walkforward_passed = true;
    // Per-gene flags remain aligned to the exact final portfolio. Candidate
    // selection already applied the same mode-specific predicate before the
    // portfolio-size cut; this replay verifies final evidence independently.
    let mut per_gene_wf: Vec<bool> = Vec::with_capacity(portfolio.len());

    let walkforward_summaries = discovery_walkforward_summaries(
        portfolio,
        portfolio_signals,
        features,
        ohlcv,
        config,
        effective_smc_gate_threshold,
        population_execution_run,
        |_| {},
    )?;
    let wf_eval_config = config
        .evaluation_config_with_smc_gate(ohlcv.close.last().copied(), effective_smc_gate_threshold);
    let wf_smc = SmcGateArrays::build(features, ohlcv)?;
    let wf_resolver = GeneEvalSettingsResolver::for_slice(
        config,
        portfolio.iter(),
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
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
        let confidences =
            account_sizing_confidences(features, gene, &wf_eval_config, &wf_smc, signals)?;
        let settings = wf_resolver.settings_for_gene(gene);
        // Reuse the exact signal-checked confidence supplied to walk-forward.
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
            selection_scope.clone(),
            search_config_hash,
            gene,
            metrics,
        )?);

        let gene_wf_passed = walkforward_summary_passed(&walkforward_summary, config.mode);
        walkforward_passed &= gene_wf_passed;
        per_gene_wf.push(gene_wf_passed);
        walkforward_validation_artifacts.push(WalkforwardValidationArtifactFile::new(
            selection_scope.clone(),
            search_config_hash,
            gene,
            walkforward_summary,
        )?);
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
            population_execution_run,
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

/// Evaluate the complete WF-passed research pool before active-portfolio
/// capacity. Only compact summaries survive the admitted parallel waves; signal
/// and evaluator working arrays die with each worker. The later final holdout
/// is deliberately not an input to this function.
fn evaluate_selection_calibration_cohort(
    candidates: &[Gene],
    candidate_archive: &[Gene],
    effective_feature_names: &[String],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
    sealed_policy: Option<&crate::live_portfolio::LiveTradingPolicyV1>,
) -> Result<crate::funnel_profile::SelectionCalibrationCohort> {
    use crate::funnel_profile::SelectionCalibrationCohort;
    crate::post_ga::check_cancel()?;
    anyhow::ensure!(
        scope.evaluated_window().role() == CanonicalSearchWindowRoleV1::SelectionValidation,
        "research pool calibration requires SelectionValidation, never the final holdout"
    );
    let rows = validation_row_count(features, ohlcv)?;
    validate_holdout_values_against_scope(scope, &features.timestamps, rows)?;
    anyhow::ensure!(
        ohlcv.open.len() == rows
            && ohlcv.timestamp.as_deref() == Some(features.timestamps.as_slice()),
        "selection calibration requires complete aligned OHLCV timestamps"
    );
    let scope_ref = crate::data_selection::CanonicalSearchArtifactScopeRefV1::from_scope(scope)
        .map_err(anyhow::Error::new)?;
    let mut archive_positions = HashMap::with_capacity(candidate_archive.len());
    for (index, gene) in candidate_archive.iter().enumerate() {
        anyhow::ensure!(
            archive_positions
                .insert(gene.strategy_id.as_str(), index)
                .is_none(),
            "calibration candidate archive contains duplicate strategy IDs"
        );
    }
    // Keep exact identities/indexes, not an unlinked list of display names.
    let jobs = candidates
        .iter()
        .map(|gene| {
            let index = *archive_positions
                .get(gene.strategy_id.as_str())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "calibration gene {} is absent from its archive",
                        gene.strategy_id
                    )
                })?;
            let identity = ValidationStrategyIdentityV2::from_gene(gene)?;
            identity.validate_against(&candidate_archive[index])?;
            Ok((gene, index, identity))
        })
        .collect::<Result<Vec<_>>>()?;
    if jobs.is_empty() {
        return Ok(SelectionCalibrationCohort {
            scope: scope_ref,
            search_config_hash: search_config_hash.to_owned(),
            trials: Vec::new(),
        });
    }
    crate::post_ga::post_ga_batch_width(rows, jobs.len())?;
    let projected =
        crate::live_portfolio::project_features_to_effective(features, effective_feature_names)?;
    let (evaluation, bypass) = if let Some(policy) = sealed_policy {
        let mut evaluation = policy.sealed_evaluation_config()?;
        evaluation.smc_gate_threshold = effective_smc_gate_threshold;
        (evaluation, policy.sealed_smc_gate_disabled()?)
    } else {
        (
            config.evaluation_config_with_smc_gate(
                ohlcv.close.last().copied(),
                effective_smc_gate_threshold,
            ),
            crate::genetic::smc_gate_disabled(),
        )
    };
    anyhow::ensure!(
        evaluation.smc_gate_threshold.is_finite() && evaluation.smc_gate_threshold >= 0.0,
        "selection calibration requires a finite non-negative final SMC gate"
    );
    let smc = SmcGateArrays::build(&projected, ohlcv)?;
    let settings = GeneEvalSettingsResolver::for_slice(
        config,
        candidates.iter(),
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
    )?;
    let (months, days) = month_day_indices(&features.timestamps);
    let trials =
        crate::post_ga::map_bounded(jobs, rows, |(gene, index, identity)| {
            crate::post_ga::check_cancel()?;
            let (signals, confidences) =
            crate::genetic::search_engine::signals_and_confidence_for_gene_full_with_smc_policy(
                &projected, gene, &evaluation, &smc, bypass,
            )?;
            let summary = compute_forward_test_summary(ForwardTestInput {
                close: &ohlcv.close,
                high: &ohlcv.high,
                low: &ohlcv.low,
                signals: &signals,
                confidences: &confidences,
                months: &months,
                days: &days,
                timestamps: &features.timestamps,
                settings: &settings.settings_for_gene(gene),
            })?;
            Ok(selection_calibration_trial(index, identity, summary))
        })?;
    crate::post_ga::check_cancel()?;
    Ok(SelectionCalibrationCohort {
        scope: scope_ref,
        search_config_hash: search_config_hash.to_owned(),
        trials,
    })
}

fn selection_calibration_trial(
    candidate_archive_index: usize,
    strategy_identity: ValidationStrategyIdentityV2,
    summary: crate::validation::ForwardTestSummary,
) -> crate::funnel_profile::SelectionCalibrationTrial {
    let invalid_slots = summary
        .metrics
        .to_metric_array()
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_finite())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let nonfinite = !invalid_slots.is_empty() || !summary.span_days.is_finite();
    let rejection_reason = if nonfinite {
        Some(format!(
            "invalid calibration metrics: canonical slots {invalid_slots:?}; non-finite span_days={}",
            !summary.span_days.is_finite()
        ))
    } else {
        crate::live_portfolio::LiveSizingEvidenceV1::validate_calibration_metrics(
            strategy_identity.strategy_id(),
            &summary.metrics,
        )
        .err()
        .map(|error| error.to_string())
    };
    crate::funnel_profile::SelectionCalibrationTrial {
        candidate_archive_index,
        strategy_identity,
        summary: (!nonfinite).then_some(summary),
        profitable_for_selection: rejection_reason.is_none(),
        rejection_reason,
    }
}

fn selected_calibration_artifacts(
    portfolio: &[Gene],
    scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
    cohort: &crate::funnel_profile::SelectionCalibrationCohort,
) -> Result<Vec<ForwardTestValidationArtifactFile>> {
    anyhow::ensure!(
        scope.evaluated_window().role() == CanonicalSearchWindowRoleV1::SelectionValidation,
        "selected calibration evidence cannot be relabeled as final holdout"
    );
    anyhow::ensure!(
        cohort.search_config_hash == search_config_hash
            && cohort.scope
                == crate::data_selection::CanonicalSearchArtifactScopeRefV1::from_scope(scope)
                    .map_err(anyhow::Error::new)?,
        "selected calibration summaries belong to another configuration/window"
    );
    let measured: HashMap<_, _> = cohort
        .trials
        .iter()
        .map(|trial| (trial.strategy_identity.exact_gene_hash(), trial))
        .collect();
    anyhow::ensure!(
        measured.len() == cohort.trials.len(),
        "duplicate calibration identities"
    );
    portfolio
        .iter()
        .map(|gene| {
            let hash = stable_json_hash(gene)?;
            let trial = measured.get(hash.as_str()).ok_or_else(|| {
                anyhow::anyhow!(
                    "selected gene {} has no measured calibration",
                    gene.strategy_id
                )
            })?;
            trial.strategy_identity.validate_against(gene)?;
            anyhow::ensure!(
                trial.profitable_for_selection,
                "selected gene failed calibration"
            );
            let summary = trial.summary.as_ref().ok_or_else(|| {
                anyhow::anyhow!("selected gene has no finite calibration summary")
            })?;
            crate::live_portfolio::LiveSizingEvidenceV1::validate_calibration_metrics(
                &gene.strategy_id,
                &summary.metrics,
            )?;
            ForwardTestValidationArtifactFile::new(
                scope.clone(),
                search_config_hash,
                gene,
                summary.clone(),
            )
        })
        .collect()
}

/// One post-lock preparation shared by the numerical holdout consumers and
/// the quote-replay provider. It is not a receipt or financial authority:
/// signals come from canonical bars; quote fills and economics stay separate.
struct PreparedLockedHoldoutResearch<'a> {
    portfolio: &'a [Gene],
    ohlcv: &'a Ohlcv,
    timestamps: &'a [i64],
    holdout_scope: &'a CanonicalSearchArtifactScopeV2,
    search_config_hash: &'a str,
    evaluation: EvaluationConfig,
    settings: GeneEvalSettingsResolver<'a>,
    ordered_signals: Vec<Vec<i8>>,
    ordered_confidences: Vec<Vec<f64>>,
}

impl<'a> PreparedLockedHoldoutResearch<'a> {
    fn new(
        portfolio: &'a [Gene],
        effective_feature_names: &[String],
        features: &'a FeatureFrame,
        ohlcv: &'a Ohlcv,
        holdout_scope: &'a CanonicalSearchArtifactScopeV2,
        search_config_hash: &'a str,
        config: &'a DiscoveryConfig,
        effective_smc_gate_threshold: f64,
    ) -> Result<Self> {
        Self::new_with_policy(
            portfolio,
            effective_feature_names,
            features,
            ohlcv,
            holdout_scope,
            search_config_hash,
            config,
            effective_smc_gate_threshold,
            None,
        )
    }

    fn new_with_policy(
        portfolio: &'a [Gene],
        effective_feature_names: &[String],
        features: &'a FeatureFrame,
        ohlcv: &'a Ohlcv,
        holdout_scope: &'a CanonicalSearchArtifactScopeV2,
        search_config_hash: &'a str,
        config: &'a DiscoveryConfig,
        effective_smc_gate_threshold: f64,
        sealed_policy: Option<&crate::live_portfolio::LiveTradingPolicyV1>,
    ) -> Result<Self> {
        let n = validation_row_count(features, ohlcv)?;
        validate_holdout_values_against_scope(holdout_scope, &features.timestamps, n)?;
        let projected = if features.names == effective_feature_names {
            Cow::Borrowed(features)
        } else {
            let keep = effective_feature_names
                .iter()
                .map(|name| {
                    features.names.iter().position(|candidate| candidate == name)
                        .ok_or_else(|| anyhow::anyhow!(
                            "holdout tail is missing feature '{}' from the discovery effective feature set",
                            name
                        ))
                })
                .collect::<Result<Vec<_>>>()?;
            Cow::Owned(features.select_columns(&keep)?)
        };
        let (evaluation, bypass) = if let Some(policy) = sealed_policy {
            let mut evaluation = policy.sealed_evaluation_config()?;
            evaluation.smc_gate_threshold = effective_smc_gate_threshold;
            (evaluation, policy.sealed_smc_gate_disabled()?)
        } else {
            (
                config.evaluation_config_with_smc_gate(
                    ohlcv.close.last().copied(),
                    effective_smc_gate_threshold,
                ),
                crate::genetic::smc_gate_disabled(),
            )
        };
        let (ordered_signals, ordered_confidences) =
            locked_holdout_signals_and_confidences_with_policy(
                portfolio,
                &projected,
                ohlcv,
                &evaluation,
                bypass,
            )?;
        let settings = GeneEvalSettingsResolver::for_slice(
            config,
            portfolio.iter(),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        Ok(Self {
            portfolio,
            ohlcv,
            timestamps: &features.timestamps,
            holdout_scope,
            search_config_hash,
            evaluation,
            settings,
            ordered_signals,
            ordered_confidences,
        })
    }

    fn forward_test_artifacts(&self) -> Result<Vec<ForwardTestValidationArtifactFile>> {
        let (months, days) = month_day_indices(self.timestamps);
        self.portfolio
            .par_iter()
            .zip(self.ordered_signals.par_iter())
            .zip(self.ordered_confidences.par_iter())
            .map(|((gene, signals), confidences)| {
                let settings = self.settings.settings_for_gene(gene);
                let summary = compute_forward_test_summary(ForwardTestInput {
                    close: &self.ohlcv.close,
                    high: &self.ohlcv.high,
                    low: &self.ohlcv.low,
                    signals,
                    confidences,
                    months: &months,
                    days: &days,
                    timestamps: self.timestamps,
                    settings: &settings,
                })?;
                ForwardTestValidationArtifactFile::new(
                    self.holdout_scope.clone(),
                    self.search_config_hash,
                    gene,
                    summary,
                )
            })
            .collect()
    }

    fn prop_firm_artifacts(
        &self,
        rules: PropFirmRiskRules,
    ) -> Result<Vec<PropFirmRiskValidationArtifactFile>> {
        self.portfolio
            .par_iter()
            .zip(self.ordered_signals.par_iter())
            .zip(self.ordered_confidences.par_iter())
            .map(|((gene, signals), confidences)| {
                let settings = self.settings.settings_for_gene(gene);
                let trades = simulate_trades_with_confidence_core(
                    &self.ohlcv.close,
                    &self.ohlcv.high,
                    &self.ohlcv.low,
                    self.timestamps,
                    signals,
                    confidences,
                    &settings,
                )?;
                let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
                    trades: &trades,
                    initial_balance: self.settings.config.initial_balance,
                    rules,
                });
                PropFirmRiskValidationArtifactFile::new(
                    self.holdout_scope.clone(),
                    self.search_config_hash,
                    gene,
                    summary,
                )
            })
            .collect()
    }
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
    holdout_scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
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
        holdout_scope,
        search_config_hash,
        config,
        effective_smc_gate_threshold,
    )
}

pub fn compute_discovery_forward_test_artifacts_with_smc_gate(
    portfolio: &[Gene],
    effective_feature_names: &[String],
    tail_features: &FeatureFrame,
    tail_ohlcv: &Ohlcv,
    holdout_scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
) -> Result<Vec<ForwardTestValidationArtifactFile>> {
    if portfolio.is_empty() {
        return Ok(Vec::new());
    }
    PreparedLockedHoldoutResearch::new(
        portfolio,
        effective_feature_names,
        tail_features,
        tail_ohlcv,
        holdout_scope,
        search_config_hash,
        config,
        effective_smc_gate_threshold,
    )?
    .forward_test_artifacts()
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
    holdout_scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
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
        holdout_scope,
        search_config_hash,
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
    holdout_scope: &CanonicalSearchArtifactScopeV2,
    search_config_hash: &str,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
    rules: PropFirmRiskRules,
) -> Result<Vec<PropFirmRiskValidationArtifactFile>> {
    if portfolio.is_empty() {
        return Ok(Vec::new());
    }
    PreparedLockedHoldoutResearch::new(
        portfolio,
        effective_feature_names,
        tail_features,
        tail_ohlcv,
        holdout_scope,
        search_config_hash,
        config,
        effective_smc_gate_threshold,
    )?
    .prop_firm_artifacts(rules)
}

#[derive(Debug, Serialize)]
struct GeneExport<'a> {
    strategy_id: &'a str,
    indicators: Vec<&'a str>,
    indices: Vec<usize>,
    weights: Vec<f64>,
    long_threshold: f64,
    short_threshold: f64,
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
/// it doesn't. The caller (CLI or server) decides whether to
/// auto-fetch and re-run, or bail to the operator.
///
/// `min_history_years` defaults to **0** (use whatever verified data exists;
/// `run_discovery_cycle_with_holdout` applies the ratio-based canonical outer
/// OOS split downstream — see the 2026-05-26 operator directive in
/// `DiscoveryRuntimeOverrides::default`). Set
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
    use neoethos_core::CanonicalTimeframe as T;

    let Ok(timeframe) = tf.trim().to_ascii_uppercase().parse::<T>() else {
        return 0;
    };
    match timeframe {
        T::D1 => 220,
        T::W1 => 52,
        T::MN1 => 12,
        fixed => {
            const TRADING_YEAR_MS: i64 = 220 * 24 * 60 * 60 * 1_000;
            let duration_ms = fixed
                .fixed_duration_ms()
                .expect("minute/hour canonical timeframe has a fixed duration");
            usize::try_from(TRADING_YEAR_MS / duration_ms)
                .expect("fixed timeframe annual bar count fits usize")
        }
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
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
            pessimistic.is_finite()
                && baseline_cost_pips.is_finite()
                && pessimistic > baseline_cost_pips
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
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
) -> Result<DiscoveryResult> {
    run_discovery_cycle_with_progress(input, config, |_| {})
}

/// Total fraction withheld from the GA/training prefix. Half is post-search
/// selection/sizing calibration; the remainder is a separate reserved final
/// test. Keeping 0.2 preserves the established first-80% search history.
pub const DEFAULT_OOS_HOLDOUT_FRACTION: f64 = 0.2;

/// Exact first-80% rows permitted to fit feature normalization. Calibration
/// and final evaluation reuse that frozen fit and never extend its training rows.
pub fn canonical_discovery_normalization_training_rows(
    row_count: usize,
) -> Result<std::ops::Range<usize>> {
    anyhow::ensure!(row_count > 0, "canonical discovery input is empty");
    let split_at = ((row_count as f64) * (1.0 - DEFAULT_OOS_HOLDOUT_FRACTION)).floor() as usize;
    anyhow::ensure!(split_at > 0, "in-sample selection window is empty");
    anyhow::ensure!(
        split_at < row_count,
        "holdout suffix is empty or split {split_at} exceeds {row_count} parent rows"
    );
    anyhow::ensure!(
        split_at >= 64,
        "in-sample selection must contain at least 64 rows; got {split_at}"
    );
    Ok(0..split_at)
}

/// One receipt-bound discovery window. Construction is private so values and
/// scope can only be derived together from one already-validated parent input.
#[derive(Debug)]
struct ScopedDiscoveryInput<'a> {
    scope: CanonicalSearchArtifactScopeV2,
    features: Cow<'a, FeatureFrame>,
    ohlcv: Cow<'a, Ohlcv>,
}

impl<'a> ScopedDiscoveryInput<'a> {
    fn entire(input: &'a CanonicalSearchRunInputV2<'_>) -> Result<Self> {
        let window = Self {
            scope: CanonicalSearchArtifactScopeV2::from_run_input(
                CanonicalSearchWindowRoleV1::DiscoveryInput,
                input,
            )
            .map_err(anyhow::Error::new)?,
            features: Cow::Borrowed(input.features()),
            ohlcv: Cow::Borrowed(input.ohlcv()),
        };
        window.validate_against_parent(input, 0..input.ohlcv().len())?;
        Ok(window)
    }

    fn owned_range(
        input: &'a CanonicalSearchRunInputV2<'_>,
        role: CanonicalSearchWindowRoleV1,
        range: std::ops::Range<usize>,
    ) -> Result<Self> {
        anyhow::ensure!(
            range.start < range.end && range.end <= input.ohlcv().len(),
            "discovery value window {}..{} is empty or exceeds {} parent rows",
            range.start,
            range.end,
            input.ohlcv().len()
        );
        let window = Self {
            scope: CanonicalSearchArtifactScopeV2::from_run_input_range(role, input, range.clone())
                .map_err(anyhow::Error::new)?,
            features: Cow::Owned(input.features().row_window(range.start, range.end)?),
            ohlcv: Cow::Owned(slice_ohlcv(input.ohlcv(), range.start, range.end)),
        };
        window.validate_against_parent(input, range)?;
        Ok(window)
    }

    fn validate_against_parent(
        &self,
        input: &CanonicalSearchRunInputV2<'_>,
        range: std::ops::Range<usize>,
    ) -> Result<()> {
        self.scope
            .validate_against_receipt(input.receipt())
            .map_err(anyhow::Error::new)?;
        let expected_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            self.scope.evaluated_window().role(),
            input,
            range.clone(),
        )
        .map_err(anyhow::Error::new)?;
        anyhow::ensure!(
            self.scope == expected_scope,
            "discovery values do not carry the exact parent receipt/role/row/timestamp scope"
        );

        let expected_timestamps = &input.features().timestamps[range.clone()];
        anyhow::ensure!(
            self.features.n_samples() == range.len()
                && self.ohlcv.len() == range.len()
                && self.features.timestamps.as_slice() == expected_timestamps
                && self.ohlcv.timestamp.as_deref() == Some(expected_timestamps),
            "discovery value window rows/timestamps do not match its exact parent range"
        );
        anyhow::ensure!(
            self.features.names == input.features().names,
            "discovery value window changed the parent feature schema"
        );
        self.features
            .ensure_same_artifact(input.features())
            .context("discovery value window changed the parent feature artifact")?;

        let expected_ohlcv = input.ohlcv();
        anyhow::ensure!(
            exact_f64_slice(&self.ohlcv.open, &expected_ohlcv.open[range.clone()])
                && exact_f64_slice(&self.ohlcv.high, &expected_ohlcv.high[range.clone()])
                && exact_f64_slice(&self.ohlcv.low, &expected_ohlcv.low[range.clone()])
                && exact_f64_slice(&self.ohlcv.close, &expected_ohlcv.close[range.clone()])
                && exact_optional_f64_slice(
                    self.ohlcv.volume.as_deref(),
                    expected_ohlcv
                        .volume
                        .as_deref()
                        .map(|volume| &volume[range.clone()]),
                ),
            "discovery OHLCV values do not exactly match the scoped parent range"
        );
        Ok(())
    }

    fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.scope
    }

    fn features(&self) -> &FeatureFrame {
        self.features.as_ref()
    }

    fn ohlcv(&self) -> &Ohlcv {
        self.ohlcv.as_ref()
    }
}

fn exact_f64_slice(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
}

fn exact_optional_f64_slice(actual: Option<&[f64]>, expected: Option<&[f64]>) -> bool {
    match (actual, expected) {
        (Some(actual), Some(expected)) => exact_f64_slice(actual, expected),
        (None, None) => true,
        _ => false,
    }
}

/// Full or holdout-split discovery authority derived from one canonical input.
/// The selection and evidence windows cannot be supplied independently.
#[derive(Debug)]
struct CanonicalDiscoveryRunInputs<'a> {
    selection: ScopedDiscoveryInput<'a>,
    calibration: Option<ScopedDiscoveryInput<'a>>,
    holdout: Option<ScopedDiscoveryInput<'a>>,
}

impl<'a> CanonicalDiscoveryRunInputs<'a> {
    fn entire(input: &'a CanonicalSearchRunInputV2<'_>) -> Result<Self> {
        let inputs = Self {
            selection: ScopedDiscoveryInput::entire(input)?,
            calibration: None,
            holdout: None,
        };
        validate_discovery_scope_pair(input, inputs.selection.scope(), None, None)?;
        Ok(inputs)
    }

    fn with_holdout(input: &'a CanonicalSearchRunInputV2<'_>) -> Result<Self> {
        let row_count = input.ohlcv().len();
        let training_rows = canonical_discovery_normalization_training_rows(row_count)?;
        Self::with_holdout_at(input, training_rows.end)
    }

    fn with_holdout_at(input: &'a CanonicalSearchRunInputV2<'_>, split_at: usize) -> Result<Self> {
        let row_count = input.ohlcv().len();
        anyhow::ensure!(row_count > 0, "canonical discovery input is empty");
        anyhow::ensure!(split_at > 0, "in-sample selection window is empty");
        anyhow::ensure!(
            split_at < row_count,
            "holdout suffix is empty or split {split_at} exceeds {row_count} parent rows"
        );
        anyhow::ensure!(
            split_at >= 64,
            "in-sample selection must contain at least 64 rows; got {split_at}"
        );
        let final_start = split_at
            .checked_add((row_count - split_at) / 2)
            .ok_or_else(|| anyhow::anyhow!("final holdout boundary overflow"))?;
        anyhow::ensure!(
            final_start > split_at && final_start < row_count,
            "selection calibration and final holdout must each contain actual rows"
        );

        let inputs = Self {
            selection: ScopedDiscoveryInput::owned_range(
                input,
                CanonicalSearchWindowRoleV1::InSample,
                0..split_at,
            )?,
            calibration: Some(ScopedDiscoveryInput::owned_range(
                input,
                CanonicalSearchWindowRoleV1::SelectionValidation,
                split_at..final_start,
            )?),
            holdout: Some(ScopedDiscoveryInput::owned_range(
                input,
                CanonicalSearchWindowRoleV1::Holdout,
                final_start..row_count,
            )?),
        };
        validate_discovery_scope_pair(
            input,
            inputs.selection.scope(),
            inputs.calibration.as_ref().map(ScopedDiscoveryInput::scope),
            inputs.holdout.as_ref().map(ScopedDiscoveryInput::scope),
        )?;
        Ok(inputs)
    }

    fn selection(&self) -> &ScopedDiscoveryInput<'_> {
        &self.selection
    }

    fn holdout(&self) -> Option<&ScopedDiscoveryInput<'_>> {
        self.holdout.as_ref()
    }

    fn calibration(&self) -> Option<&ScopedDiscoveryInput<'_>> {
        self.calibration.as_ref()
    }
}

fn validate_discovery_scope_pair(
    input: &CanonicalSearchRunInputV2<'_>,
    selection: &CanonicalSearchArtifactScopeV2,
    calibration: Option<&CanonicalSearchArtifactScopeV2>,
    holdout: Option<&CanonicalSearchArtifactScopeV2>,
) -> Result<()> {
    selection
        .validate_against_receipt(input.receipt())
        .map_err(anyhow::Error::new)?;
    let full_scope = CanonicalSearchArtifactScopeV2::from_run_input(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        input,
    )
    .map_err(anyhow::Error::new)?;
    let full = full_scope.evaluated_window();
    let selected = selection.evaluated_window();
    validate_normalization_training_scope(input.receipt(), selected)?;

    let Some(holdout) = holdout else {
        anyhow::ensure!(
            calibration.is_none(),
            "calibration requires a separate final holdout"
        );
        anyhow::ensure!(
            selected.role() == CanonicalSearchWindowRoleV1::DiscoveryInput,
            "holdout-free selection scope has the wrong role"
        );
        anyhow::ensure!(
            selection == &full_scope,
            "holdout-free selection scope must exactly cover the full canonical input"
        );
        return Ok(());
    };

    holdout
        .validate_against_receipt(input.receipt())
        .map_err(anyhow::Error::new)?;
    let held_out = holdout.evaluated_window();
    anyhow::ensure!(
        selected.role() == CanonicalSearchWindowRoleV1::InSample,
        "holdout split selection scope has the wrong role; expected in_sample"
    );
    anyhow::ensure!(
        held_out.role() == CanonicalSearchWindowRoleV1::Holdout,
        "holdout split evidence scope has the wrong role; expected holdout"
    );
    anyhow::ensure!(
        selected.row_start() == full.row_start()
            && selected.timestamp_start_ms() == full.timestamp_start_ms(),
        "holdout split selection must start at the exact canonical input boundary"
    );
    anyhow::ensure!(
        held_out.row_end() == full.row_end()
            && held_out.timestamp_end_ms() == full.timestamp_end_ms(),
        "holdout split evidence must end at the exact canonical input boundary"
    );
    let calibration = calibration.ok_or_else(|| {
        anyhow::anyhow!("a new split run requires calibration separate from final holdout")
    })?;
    calibration
        .validate_against_receipt(input.receipt())
        .map_err(anyhow::Error::new)?;
    let calibrated = calibration.evaluated_window();
    anyhow::ensure!(
        calibrated.role() == CanonicalSearchWindowRoleV1::SelectionValidation
            && selected.row_end() == calibrated.row_start()
            && calibrated.row_end() == held_out.row_start(),
        "selection, calibration and final holdout must be contiguous without a gap or overlap"
    );

    let split_at = selected
        .row_end()
        .checked_sub(full.row_start())
        .and_then(|rows| usize::try_from(rows).ok())
        .ok_or_else(|| anyhow::anyhow!("holdout split row boundary cannot map into parent rows"))?;
    anyhow::ensure!(
        split_at > 0 && split_at < input.ohlcv().len(),
        "holdout split must contain non-empty selection and evidence windows"
    );
    let expected_selection = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::InSample,
        input,
        0..split_at,
    )
    .map_err(anyhow::Error::new)?;
    let expected_holdout = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::Holdout,
        input,
        usize::try_from(held_out.row_start() - full.row_start())?..input.ohlcv().len(),
    )
    .map_err(anyhow::Error::new)?;
    anyhow::ensure!(
        selection == &expected_selection
            && holdout == &expected_holdout
            && calibration
                == &CanonicalSearchArtifactScopeV2::from_run_input_range(
                    CanonicalSearchWindowRoleV1::SelectionValidation,
                    input,
                    split_at..usize::try_from(held_out.row_start() - full.row_start())?
                )
                .map_err(anyhow::Error::new)?,
        "selection, calibration and final holdout scopes do not exactly match the parent rows/timestamps"
    );
    Ok(())
}

/// A feature view retains the original producer's fitted-row coordinates.
/// Map those coordinates through the exact contiguous anchor before deciding
/// whether the frozen fit was learned only from permitted selection history.
/// This runs before discovery work, not merely when a survivor is exported.
pub(crate) fn validate_normalization_training_scope(
    receipt: &CanonicalSearchInputReceiptV2,
    selected: &CanonicalSearchEvaluatedWindowV1,
) -> Result<()> {
    let Some(fitted) = receipt.normalization_fitted_state() else {
        return Ok(());
    };
    let anchors = receipt
        .source_bindings()
        .iter()
        .filter(|binding| binding.dataset_identity() == receipt.anchor_dataset_identity())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        anchors.len() == 1,
        "normalization training scope requires exactly one receipt anchor binding"
    );
    let segments = anchors[0].segments();
    let first = segments.first().ok_or_else(|| {
        anyhow::anyhow!("normalization training scope has no anchor source segments")
    })?;
    anyhow::ensure!(
        segments
            .windows(2)
            .all(|pair| pair[0].row_end() == pair[1].row_start()),
        "normalization training scope cannot map disjoint anchor source segments"
    );
    let training_rows = fitted.training_rows()?;
    let absolute_fit_end = first
        .row_start()
        .checked_add(u64::try_from(training_rows.end)?)
        .ok_or_else(|| anyhow::anyhow!("normalization training row boundary overflow"))?;
    anyhow::ensure!(
        absolute_fit_end <= selected.row_end(),
        "normalization fit extends beyond selection training rows into held-out data"
    );
    Ok(())
}

/// [`run_discovery_cycle`] behind the outer OOS holdout split — the single
/// source of truth for "discovery never sees the tail" (audit B02/B03).
///
/// Discovery (GA + in-sample selection/gates) runs on the FIRST 80% only.
/// The remaining actual rows form separate calibration and final windows:
/// forward-test/prop-firm diagnostics guide selection and sizing on calibration;
/// the final tail remains reserved until the candidate is locked. Every production caller (desktop app, CLI,
/// batch orchestrator) must go through this wrapper; calling
/// [`run_discovery_cycle`] directly is only correct for tests or callers
/// that manage their own holdout.
pub fn run_discovery_cycle_with_holdout(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
) -> Result<DiscoveryResult> {
    run_discovery_cycle_with_holdout_and_progress(input, config, prop_firm_rules, |_| {})
}

/// See [`run_discovery_cycle_with_holdout`]; this variant forwards discovery
/// progress events to `progress_fn` (same contract as
/// [`run_discovery_cycle_with_progress`]).
pub fn run_discovery_cycle_with_holdout_and_progress<F>(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;
    run_discovery_cycle_with_holdout_and_progress_authorized(
        input,
        config,
        prop_firm_rules,
        None,
        None,
        None,
        progress_fn,
    )
    .map(|(result, _)| result)
}

/// CPU half of the prepared V3 entrypoint. The caller already consumed the
/// one physical-inventory admission and converted its zero-physical-GPU proof
/// into the exact run-owned Search route, so this path must not probe again.
#[cfg(feature = "gpu-cuda")]
pub(crate) fn run_discovery_cycle_with_prepared_cpu_admission_v3<F>(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    strict_device_admission: crate::SealedStrictDiscoveryDeviceAdmissionV1,
    progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;
    run_discovery_cycle_with_holdout_and_progress_authorized(
        input,
        config,
        prop_firm_rules,
        None,
        Some(strict_device_admission),
        None,
        progress_fn,
    )
    .map(|(result, _)| result)
}

/// Discovery result plus the only versioned quote-validation evidence that may
/// be produced after the final portfolio and outer holdout are locked.
///
/// The evidence remains research-only and is not a promotion permit.
#[derive(Debug)]
pub struct QuoteValidatedDiscoveryResultV1 {
    result: DiscoveryResult,
    quote_validated_outer_holdout: crate::QuoteValidatedOuterHoldoutResearchEvidenceV1,
}

impl QuoteValidatedDiscoveryResultV1 {
    pub fn result(&self) -> &DiscoveryResult {
        &self.result
    }

    pub fn quote_validated_outer_holdout(
        &self,
    ) -> &crate::QuoteValidatedOuterHoldoutResearchEvidenceV1 {
        &self.quote_validated_outer_holdout
    }

    pub fn into_parts(
        self,
    ) -> (
        DiscoveryResult,
        crate::QuoteValidatedOuterHoldoutResearchEvidenceV1,
    ) {
        (self.result, self.quote_validated_outer_holdout)
    }

    pub fn validate_complete_promotion_evidence(&self) -> Result<()> {
        self.result.validate_validation_evidence_sets(true)?;
        crate::quote_validated_outer_holdout_v1::require_quote_validated_outer_holdout_v1(Some(
            &self.quote_validated_outer_holdout,
        ))
        .map(|_| ())
        .map_err(anyhow::Error::new)
    }
}

type LockedOuterHoldoutReplayProviderV1<'a> = dyn FnMut(
        &DiscoveryResult,
        &[Vec<i8>],
        &CanonicalSearchArtifactScopeV2,
    ) -> Result<crate::LockedPortfolioOuterHoldoutReplaySetV1>
    + 'a;

type LockedQuoteReplayObserverV2<'a> =
    dyn FnMut(&DiscoveryResult, &PreparedLockedHoldoutResearch<'_>) -> Result<()> + 'a;

fn account_sizing_confidences(
    features: &FeatureFrame,
    gene: &Gene,
    evaluation: &EvaluationConfig,
    smc: &SmcGateArrays,
    expected_signals: &[i8],
) -> Result<Vec<f64>> {
    let (signals, confidences) =
        signals_and_confidence_for_gene_full_with_smc(features, gene, evaluation, smc)?;
    anyhow::ensure!(
        signals == expected_signals && confidences.len() == expected_signals.len(),
        "account-risk confidence for '{}' does not match its exact SMC-gated signal series",
        gene.strategy_id
    );
    Ok(confidences)
}

#[cfg(test)]
fn locked_holdout_signals(
    portfolio: &[Gene],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    evaluation: &EvaluationConfig,
) -> Result<Vec<Vec<i8>>> {
    locked_holdout_signals_and_confidences(portfolio, features, ohlcv, evaluation)
        .map(|(signals, _)| signals)
}

#[cfg(test)]
fn locked_holdout_signals_and_confidences(
    portfolio: &[Gene],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    evaluation: &EvaluationConfig,
) -> Result<(Vec<Vec<i8>>, Vec<Vec<f64>>)> {
    locked_holdout_signals_and_confidences_with_policy(
        portfolio,
        features,
        ohlcv,
        evaluation,
        crate::genetic::smc_gate_disabled(),
    )
}

/// Pure signal values from an explicit archived signal policy. The bypass is
/// captured once with Search authority, not re-read from mutable process state
/// inside the parallel gene loop. This function grants no trading authority.
pub fn locked_holdout_signals_and_confidences_with_policy(
    portfolio: &[Gene],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    evaluation: &EvaluationConfig,
    smc_gate_disabled: bool,
) -> Result<(Vec<Vec<i8>>, Vec<Vec<f64>>)> {
    if portfolio.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let n = validation_row_count(features, ohlcv)?;
    anyhow::ensure!(
        ohlcv.open.len() == n && ohlcv.timestamp.as_deref() == Some(features.timestamps.as_slice()),
        "locked holdout signals require the same complete OHLCV rows and timestamps as the feature frame"
    );
    anyhow::ensure!(
        evaluation.smc_gate_threshold.is_finite() && evaluation.smc_gate_threshold >= 0.0,
        "locked holdout signals require a finite non-negative final SMC gate"
    );
    // Build the exact frame/bar SMC inputs once, then share them across the
    // indexed gene loop. The raw indicator-only function omits these gates.
    let smc = SmcGateArrays::build(features, ohlcv)?;
    let generated = portfolio
        .par_iter()
        .map(|gene| {
            let (signals, confidences) =
                crate::genetic::search_engine::signals_and_confidence_for_gene_full_with_smc_policy(
                    features, gene, evaluation, &smc, smc_gate_disabled,
                )?;
            anyhow::ensure!(
                signals.len() == n && confidences.len() == n,
                "locked holdout signals for '{}' contain {} rows instead of {n}",
                gene.strategy_id,
                signals.len()
            );
            Ok((signals, confidences))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(generated.into_iter().unzip())
}

/// Run canonical-trendbar discovery, then ask for sealed quote replay exactly
/// once after the portfolio is final and all early-return conditions have
/// passed. The provider cannot influence GA, CPCV, features, or selection.
pub fn run_discovery_cycle_with_quote_validated_outer_holdout_and_progress<F, P>(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    mut replay_provider: P,
    progress_fn: F,
) -> Result<QuoteValidatedDiscoveryResultV1>
where
    F: FnMut(DiscoveryProgress),
    P: FnMut(
        &DiscoveryResult,
        &[Vec<i8>],
        &CanonicalSearchArtifactScopeV2,
    ) -> Result<crate::LockedPortfolioOuterHoldoutReplaySetV1>,
{
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;
    let (result, quote_validated_outer_holdout) =
        run_discovery_cycle_with_holdout_and_progress_authorized(
            input,
            config,
            prop_firm_rules,
            Some(&mut replay_provider),
            None,
            None,
            progress_fn,
        )?;
    let quote_validated_outer_holdout =
        crate::quote_validated_outer_holdout_v1::require_quote_validated_outer_holdout_v1(
            quote_validated_outer_holdout.as_ref(),
        )
        .map_err(anyhow::Error::new)?
        .clone();
    Ok(QuoteValidatedDiscoveryResultV1 {
        result,
        quote_validated_outer_holdout,
    })
}

/// Full receipt-bound CUDA/CPU discovery for canonical-trendbar research.
/// The returned envelope is structurally research-only and cannot authorize
/// live execution or promotion.
pub fn run_canonical_trendbar_research_discovery_with_holdout_and_progress<F>(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3>
where
    F: FnMut(DiscoveryProgress),
{
    contract.validate_against_input(input)?;
    let strict_device_admission =
        crate::SealedStrictDiscoveryDeviceAdmissionV1::from_explicit_canonical_cpu_research_v1(
            contract,
        )
        .map_err(anyhow::Error::new)?;
    let mut research_config = config.clone();
    apply_research_contract_to_discovery_config(&mut research_config, contract);
    let _research_execution =
        crate::canonical_trendbar_research::install_canonical_trendbar_research_execution_v3(
            contract,
        )?;
    let result = run_discovery_cycle_with_holdout_and_progress_authorized(
        input,
        &research_config,
        prop_firm_rules,
        None,
        Some(strict_device_admission),
        None,
        progress_fn,
    )?
    .0;
    crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3::new(
        contract.clone(),
        result,
    )
}

/// Explicit canonical CPU research plus the prelocked quote/economics
/// continuation. The provider runs only after the same final full-SMC signal
/// preparation used by the existing holdout consumers. It receives no mutable
/// portfolio and cannot participate in GA/CPCV selection. Cancellation or an
/// empty portfolio returns no quote evidence; neither is called a successful
/// validation. Actual captured review and execution economics stay explicit.
pub fn run_canonical_trendbar_research_with_quote_holdout_v3<F, P>(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    contract: &crate::CanonicalTrendbarResearchExecutionContractV3,
    prop_firm_rules: PropFirmRiskRules,
    mut replay_provider: P,
    progress_fn: F,
) -> Result<(
    crate::CanonicalTrendbarResearchDiscoveryResultV3,
    Option<crate::QuoteValidatedOuterHoldoutResearchEvidenceV3>,
)>
where
    F: FnMut(DiscoveryProgress),
    P: FnMut(
        &crate::LockedCanonicalSignalPlanV3<'_>,
    ) -> Result<crate::LockedPortfolioOuterHoldoutReplaySetV3>,
{
    contract.validate_against_input(input)?;
    let admission =
        crate::SealedStrictDiscoveryDeviceAdmissionV1::from_explicit_canonical_cpu_research_v1(
            contract,
        )
        .map_err(anyhow::Error::new)?;
    let mut research_config = config.clone();
    apply_research_contract_to_discovery_config(&mut research_config, contract);
    let _research_execution =
        crate::canonical_trendbar_research::install_canonical_trendbar_research_execution_v3(
            contract,
        )?;
    let mut quote_evidence = None;
    let mut observe = |result: &DiscoveryResult,
                       prepared: &PreparedLockedHoldoutResearch<'_>|
     -> Result<()> {
        // Bind the artifact actually shipped to the model handoff. Its existing
        // export filter can remove genes; the original prepared array is not an
        // interchangeable portfolio identity.
        let artifact = crate::LivePortfolioArtifact::from_discovery(
            result
                .search_input_receipt
                .normalization_fitted_state()
                .is_some(),
            result,
        )?;
        let full_portfolio_hash = crate::canonical_locked_portfolio_identity_sha256_v1(&artifact)?;
        let mut signals = Vec::with_capacity(artifact.genes.len());
        let mut confidences = Vec::with_capacity(artifact.genes.len());
        for gene in &artifact.genes {
            let index = prepared
                .portfolio
                .iter()
                .position(|candidate| candidate.strategy_id == gene.strategy_id)
                .ok_or_else(|| anyhow::anyhow!("shipped gene is absent from prelocked signals"))?;
            ValidationStrategyIdentityV2::from_gene(gene)?
                .validate_against(&prepared.portfolio[index])?;
            signals.push(prepared.ordered_signals[index].clone());
            confidences.push(prepared.ordered_confidences[index].clone());
        }
        let multipliers = vec![vec![1.0; prepared.timestamps.len()]; artifact.genes.len()];
        let evaluation = artifact.live_trading_policy.sealed_evaluation_config()?;
        let locked = crate::LockedCanonicalSignalPlanV3::new(
            &artifact.genes,
            &full_portfolio_hash,
            &signals,
            &confidences,
            &multipliers,
            prepared.ohlcv,
            prepared.holdout_scope,
            prepared.search_config_hash,
            crate::CanonicalSignalExitPolicyV2::from_evaluation(&evaluation),
            crate::CanonicalSignalAccountRiskPolicyV3::from_evaluation(&evaluation)?,
            artifact
                .live_trading_policy
                .sealed_adaptive_stops_policy()?,
        )?;
        let replay = replay_provider(&locked)?;
        quote_evidence = Some(crate::evaluate_locked_portfolio_outer_holdout_v3(
            &locked, replay,
        )?);
        Ok(())
    };
    let (result, _) = run_discovery_cycle_with_holdout_and_progress_authorized(
        input,
        &research_config,
        prop_firm_rules,
        None,
        Some(admission),
        Some(&mut observe),
        progress_fn,
    )?;
    Ok((
        crate::CanonicalTrendbarResearchDiscoveryResultV3::new(contract.clone(), result)?,
        quote_evidence,
    ))
}

fn run_discovery_cycle_with_holdout_and_progress_authorized<F>(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    mut quote_validated_outer_holdout: Option<&mut LockedOuterHoldoutReplayProviderV1<'_>>,
    strict_device_admission: Option<crate::SealedStrictDiscoveryDeviceAdmissionV1>,
    mut prelocked_quote_replay: Option<&mut LockedQuoteReplayObserverV2<'_>>,
    mut progress_fn: F,
) -> Result<(
    DiscoveryResult,
    Option<crate::QuoteValidatedOuterHoldoutResearchEvidenceV1>,
)>
where
    F: FnMut(DiscoveryProgress),
{
    #[cfg(feature = "gpu")]
    let _cubecl_population_residency = crate::cubecl_eval::cubecl_residency_scope();
    let inputs = CanonicalDiscoveryRunInputs::with_holdout(input)?;
    let selection = inputs.selection();
    let holdout = inputs
        .holdout()
        .expect("with_holdout always constructs a held-out evidence suffix");
    let calibration = inputs
        .calibration()
        .expect("with_holdout always constructs a separate selection calibration window");
    let n_rows = input.ohlcv().len();
    let is_end = selection.ohlcv().len();
    let explicit_cpu_research = strict_device_admission.as_ref().is_some_and(
        crate::SealedStrictDiscoveryDeviceAdmissionV1::is_explicit_canonical_cpu_research_v1,
    );
    let gpu_manifest = crate::gpu_native::capability::GpuCapabilityManifest::stage1_baseline();
    let host_feature_preparation = gpu_manifest
        .capability(crate::gpu_native::capability::PipelineStage::FeaturePreparation)
        .ok_or_else(|| anyhow::anyhow!("GPU capability manifest omitted feature preparation"))?;
    anyhow::ensure!(
        host_feature_preparation.capability
            == crate::gpu_native::capability::StageGpuCapability::CpuOnly,
        "feature-preparation capability changed without a reviewed resident-Data continuation"
    );
    // Discovery is still a mixed CPU/GPU pipeline. Preflight only the stage
    // that this entrypoint is about to require from the exact device route;
    // the separate full-Discovery permit remains fail-closed over all stages.
    // In particular, do not label host feature preparation or GA orchestration
    // as resident GPU work merely because population evaluation is native.
    if !explicit_cpu_research {
        crate::gpu_native::capability::gpu_pipeline_preflight(
            crate::backend::current_evaluation_backend(),
            &gpu_manifest,
            &[crate::gpu_native::capability::PipelineStage::PopulationEvaluation],
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    let population_evaluation = if explicit_cpu_research {
        crate::gpu_native::capability::StageGpuCapability::CpuOnly
    } else {
        crate::gpu_native::capability::StageGpuCapability::StrictGpu
    };
    tracing::warn!(
        target: "neoethos_search::discovery",
        feature_preparation = ?host_feature_preparation.capability,
        feature_preparation_detail = host_feature_preparation.detail,
        population_evaluation = ?population_evaluation,
        "stage-scoped Discovery admission: population evaluation must use its exact native/typed CPU route; feature preparation remains explicitly host-side"
    );

    tracing::info!(
        target: "neoethos_search::discovery",
        total_rows = n_rows,
        in_sample_rows = is_end,
        calibration_rows = calibration.ohlcv().len(),
        final_holdout_rows = holdout.ohlcv().len(),
        holdout_fraction = DEFAULT_OOS_HOLDOUT_FRACTION,
        "discovery sees only the first {is_end} rows; calibration selects/sizes survivors, and the separate final tail remains reserved"
    );

    let mut result = if let Some(strict_device_admission) = strict_device_admission {
        run_discovery_cycle_values_with_progress(
            strict_device_admission,
            &inputs,
            config,
            &mut progress_fn,
        )?
    } else {
        run_discovery_cycle_values_with_real_device_admission(&inputs, config, &mut progress_fn)?
    };

    // Operator Stop mid-search: the cycle returned early with a partial
    // result the caller will discard — don't burn time forward-testing it.
    if crate::genetic::search_engine::search_cancel_requested() {
        return Ok((result, None));
    }
    if result.portfolio.is_empty() {
        return Ok((result, None));
    }

    progress_fn(DiscoveryProgress::StageAdvanced {
        stage: "holdout_forward_test",
        detail: format!(
            "preparing SMC-gated signals for {} strategies on the {}-row selection calibration window; final holdout remains reserved",
            result.portfolio.len(),
            calibration.ohlcv().len()
        ),
    });
    let holdout_scope = result
        .holdout_scope()?
        .ok_or_else(|| anyhow::anyhow!("split discovery result lost its holdout scope"))?
        .clone();
    let calibration_scope = result
        .calibration_scope()?
        .ok_or_else(|| anyhow::anyhow!("split discovery result lost its calibration scope"))?
        .clone();
    let sealed_policy = result
        .funnel_profile
        .as_ref()
        .and_then(|funnel| funnel.live_trading_policy_v1())
        .ok_or_else(|| {
            anyhow::anyhow!("production holdout lost its sealed Search signal policy")
        })?;
    let prepared = PreparedLockedHoldoutResearch::new_with_policy(
        &result.portfolio,
        &result.effective_feature_names,
        calibration.features(),
        calibration.ohlcv(),
        &calibration_scope,
        &result.search_config_hash,
        config,
        result.effective_smc_gate_threshold,
        Some(sealed_policy),
    );
    if let Ok(prepared) = &prepared {
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "holdout_forward_test",
            detail: format!(
                "prepared {} ordered strategy signal vectors; running forward-test and prop-firm checks in the shared CPU pool",
                prepared.ordered_signals.len()
            ),
        });
        // Both consumers are independent CPU calculations on immutable inputs.
        // Nested gene loops use the same pool, not additional thread pools.
        let measured_forward = if result
            .funnel_profile
            .as_ref()
            .is_some_and(|funnel| funnel.selection_calibration_cohort.is_some())
        {
            Some(std::mem::take(
                &mut result.forward_test_validation_artifacts,
            ))
        } else {
            None
        };
        let (forward, prop_firm) = rayon::join(
            || {
                measured_forward
                    .map(Ok)
                    .unwrap_or_else(|| prepared.forward_test_artifacts())
            },
            || prepared.prop_firm_artifacts(prop_firm_rules),
        );
        match forward {
            Ok(artifacts) => result.forward_test_validation_artifacts = artifacts,
            Err(err) => tracing::warn!(
                target: "neoethos_search::discovery",
                error = %err,
                "forward-test artifact computation on the held-out tail failed \
                 (the research result remains inspectable, but live/promotion authorities \
                 will fail closed because exact forward-test evidence is missing)"
            ),
        }
        match prop_firm {
            Ok(artifacts) => result.prop_firm_validation_artifacts = artifacts,
            Err(err) => tracing::warn!(
                target: "neoethos_search::discovery",
                error = %err,
                "prop-firm artifact computation on the held-out tail failed \
                 (the research result remains inspectable, but live/promotion authorities \
                 will fail closed because exact prop-firm evidence is missing)"
            ),
        }
    } else if let Err(err) = &prepared {
        tracing::warn!(
            target: "neoethos_search::discovery",
            error = %err,
            "held-out signal preparation failed; the research result remains inspectable, \
             but neither numerical holdout artifacts nor quote replay can be produced"
        );
    }
    if crate::genetic::search_engine::search_cancel_requested() {
        return Ok((result, None));
    }
    // Release calibration vectors before preparing final signals. Plain search
    // leaves the final numerical test reserved until the models are locked.
    drop(prepared);
    let final_prepared = if prelocked_quote_replay.is_some()
        || quote_validated_outer_holdout.is_some()
    {
        let sealed_policy = result
            .funnel_profile
            .as_ref()
            .and_then(|funnel| funnel.live_trading_policy_v1())
            .ok_or_else(|| anyhow::anyhow!("final replay lost its sealed Search signal policy"))?;
        Some(PreparedLockedHoldoutResearch::new_with_policy(
            &result.portfolio,
            &result.effective_feature_names,
            holdout.features(),
            holdout.ohlcv(),
            &holdout_scope,
            &result.search_config_hash,
            config,
            result.effective_smc_gate_threshold,
            Some(sealed_policy),
        )?)
    } else {
        None
    };
    if let Some(replay) = prelocked_quote_replay.as_mut() {
        let prepared = final_prepared
            .as_ref()
            .expect("requested final quote preparation");
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "holdout_quote_replay",
            detail: "replaying the prelocked strategy/exit policy against independently reviewed Bid/Ask quotes".to_owned(),
        });
        replay(&result, prepared)?;
    }
    let quote_validated_outer_holdout = if let Some(replay_provider) =
        quote_validated_outer_holdout.as_mut()
    {
        anyhow::ensure!(
            !result.portfolio.iter().any(|gene| gene.stop_vol_mult > 0.0),
            "quote-validated outer-holdout V1 refuses adaptive stops because its ordered fixed-risk binding cannot represent per-bar stop distances"
        );
        let prepared = final_prepared
            .as_ref()
            .expect("requested final quote preparation");
        let evaluation = &prepared.evaluation;
        anyhow::ensure!(
            evaluation.pip_value_per_lot.is_finite() && evaluation.pip_value_per_lot > 0.0,
            "quote-validated outer-holdout metrics require an exact positive pip value per lot"
        );
        let ordered_signals = &prepared.ordered_signals;
        let replay_set: crate::LockedPortfolioOuterHoldoutReplaySetV1 =
            replay_provider(&result, ordered_signals, &holdout_scope)?;
        Some(
            crate::evaluate_locked_portfolio_outer_holdout_v1(
                &result.portfolio,
                ordered_signals,
                &result.search_config_hash,
                &holdout_scope,
                config.initial_account_balance()?,
                evaluation.pip_value_per_lot,
                replay_set,
            )
            .map_err(anyhow::Error::new)?,
        )
    } else {
        None
    };
    Ok((result, quote_validated_outer_holdout))
}

pub fn run_discovery_cycle_with_progress<F>(
    input: &CanonicalSearchRunInputV2<'_>,
    config: &DiscoveryConfig,
    progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    #[cfg(feature = "gpu")]
    let _cubecl_population_residency = crate::cubecl_eval::cubecl_residency_scope();
    let inputs = CanonicalDiscoveryRunInputs::entire(input)?;
    run_discovery_cycle_values_with_real_device_admission(&inputs, config, progress_fn)
}

fn run_discovery_cycle_values_with_real_device_admission<F>(
    inputs: &CanonicalDiscoveryRunInputs<'_>,
    config: &DiscoveryConfig,
    progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    let strict_device_admission =
        crate::acquire_strict_discovery_device_admission_v1().map_err(anyhow::Error::new)?;
    run_discovery_cycle_values_with_progress(strict_device_admission, inputs, config, progress_fn)
}

fn run_discovery_cycle_values_with_progress<F>(
    strict_device_admission: crate::SealedStrictDiscoveryDeviceAdmissionV1,
    inputs: &CanonicalDiscoveryRunInputs<'_>,
    config: &DiscoveryConfig,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    let selection = inputs.selection();
    let features = selection.features();
    let ohlcv = selection.ohlcv();
    let selection_scope = selection.scope();
    let search_input_receipt = selection.scope().receipt();
    crate::historical_evaluation_authority::require_historical_evaluation_authority_v1()?;

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
    if !config.pnl_conversion_fee_rate.is_finite()
        || !(0.0..1.0).contains(&config.pnl_conversion_fee_rate)
    {
        anyhow::bail!(
            "run_discovery_cycle: DiscoveryConfig.pnl_conversion_fee_rate must be finite and in \
             [0, 1), got {}. Bind it from the exact broker financial contract before search.",
            config.pnl_conversion_fee_rate
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
            crate::genetic::derive_adaptive_threshold_ladder_from_features(&features)?
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
                // ledger entry from one run are comparable by the same legacy
                // resolved-config subset hash. Full S3b search authority also
                // requires the sizing receipt and exact stage-1 projection.
                let normalize_features = features.normalization_fitted_state().is_some();
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
                            "resolved-config stamp — the legacy decision subset this run \
                             resolved. Equal subset hashes are not an exhaustive experiment \
                             identity; use the strict search authority plus sizing receipt."
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
            let bare = match crate::prefilter_schema_v1::timeframe_group_v1(name) {
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
        //  * the cheap ranking is fitted once inside the already-isolated
        //    selection window. CPCV belongs to the post-GA finalist stage.
        let financial_geometry =
            resolve_prefilter_financial_geometry_v1(config, ohlcv.close.last().copied());
        let spec = PrefilterSpec {
            top_k: prefilter_top_k,
            insample_frac: prefilter_insample_frac,
            min_per_tf: prefilter_min_per_tf,
            max_hold_bars: financial_geometry.max_hold_bars,
            atr_period: 14,
            // THE LABEL'S GEOMETRY, read from the band the GA will actually
            // search rather than pinned to a literal. Corrected 2026-08-09: the
            // hardcoded (1.0 ATR, rr 2.0) was the very bottom corner of a band
            // that change #3 made searchable (sl 1–4 ATR, rr 1.5–4.0), so a
            // feature that predicts first passage for a 3-ATR stop at rr 3.5 was
            // ranked as if it did not.
            //
            // The MIDPOINT of the band, not a sweep across it: scoring at k
            // points multiplies the column-correlation work by k. The midpoint
            // is a strictly better
            // single representative than a corner; a full band sweep, keeping
            // the worst point the way the fold rule keeps the worst fold, is the
            // follow-up and it costs k× the correlation pass.
            sl_atr_mult: financial_geometry.stop_atr_multiplier,
            rr: financial_geometry.reward_risk_ratio,
            round_trip_cost_px: financial_geometry.round_trip_cost_price,
        };
        let (filtered_frame, census) = prefilter_features(&features, &ohlcv, &spec)?;
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
             from this run's gene stop band, ranking fitted once on the label-safe selection \
             window, correlation two-pass f64. CPCV remains a post-GA finalist gate."
        );
        if census.columns_unrankable > 0 {
            tracing::warn!(
                target: "neoethos_search::prefilter",
                count = census.columns_unrankable,
                sample = ?census.unrankable_sample,
                 "columns EXCLUDED as unrankable — fewer than the minimum pairwise-complete \
                  rows, or no variance, in the selection window. They are named and dropped, not \
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
        if features.normalization_fitted_state().is_some()
            && census.columns_with_nonfinite_rows == 0
        {
            tracing::info!(
                target: "neoethos_search::prefilter",
                "this feature frame carries a fitted normalizer; typed invalidity is preserved \
                 and excluded pairwise, never replaced with valid zeros. The ranked selection \
                 window contained no excluded invalid or non-finite cells."
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
                "prefilter_below_selection_window_top_k",
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

    // The resident execution parent is the exact post-trim, post-prefilter
    // dataset consumed by every population lane in this run. The public
    // selection scope still names the caller's full selected input; derive a
    // narrower internal scope for the suffix retained by `max_rows` so the
    // engine receipt never over-claims rows it did not evaluate.
    let selected_window = selection_scope.evaluated_window();
    let selected_rows = selected_window
        .row_end()
        .checked_sub(selected_window.row_start())
        .and_then(|rows| usize::try_from(rows).ok());
    anyhow::ensure!(
        selected_rows == Some(n_input_rows),
        "canonical selection scope row count does not match the discovery input"
    );
    let trimmed_prefix = n_input_rows
        .checked_sub(n_after_trim)
        .ok_or_else(|| anyhow::anyhow!("trimmed discovery row count exceeds its input"))?;
    let resident_row_start = selected_window
        .row_start()
        .checked_add(trimmed_prefix as u64)
        .ok_or_else(|| anyhow::anyhow!("resident population row-start overflow"))?;
    let resident_window = CanonicalSearchEvaluatedWindowV1::new(
        selected_window.role(),
        resident_row_start,
        selected_window.row_end(),
        *features
            .timestamps
            .first()
            .ok_or_else(|| anyhow::anyhow!("resident population input is empty"))?,
        *features
            .timestamps
            .last()
            .ok_or_else(|| anyhow::anyhow!("resident population input is empty"))?,
    )?;
    let resident_scope =
        CanonicalSearchArtifactScopeV2::new(search_input_receipt.clone(), resident_window)?;
    let population_execution_run =
        crate::population_execution_evidence_v1::begin_exact_population_execution_run_v1(
            strict_device_admission,
            &resident_scope,
            &features,
            &ohlcv,
        )
        .map_err(anyhow::Error::new)?;

    // Diagnostic (2026-06-08): surface the per-timeframe coverage of the
    // prefiltered cube so a "multi-TF features never reached the GA"
    // regression is visible at a glance instead of hiding behind a flat
    // `cols=N`. base = unprefixed/regime; each higher TF shows its survivor
    // count. A higher TF reading 0 here means the warm-start can't use it.
    {
        let mut by_tf: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for name in &effective_feature_names {
            let key = crate::prefilter_schema_v1::timeframe_group_v1(name)
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

    // Multi-stage Funnel: Stage 1 (Fast Evaluation)
    let stage1_pct = config.runtime_overrides.resolved_funnel_stage1_pct();
    let stage1_window = config.runtime_overrides.stage1_window;

    let total_rows = ohlcv.close.len();
    let stage1_len = ((total_rows as f64 * stage1_pct) as usize).min(total_rows);
    let (stage1_start, stage1_end) = match stage1_window {
        Stage1Window::MostRecent => (total_rows.saturating_sub(stage1_len), total_rows),
        Stage1Window::Earliest => (0, stage1_len),
    };
    funnel.stage1_evaluation_window = Some(crate::funnel_profile::Stage1EvaluationWindow {
        selection_rows: total_rows,
        start_row: stage1_start,
        end_row_exclusive: stage1_end,
        first_timestamp_ms: features.timestamps.get(stage1_start).copied(),
        last_timestamp_ms: stage1_end
            .checked_sub(1)
            .filter(|&last| last >= stage1_start)
            .and_then(|last| features.timestamps.get(last).copied()),
    });
    tracing::info!(
        target: "neoethos_search::funnel",
        window = ?stage1_window,
        stage1_pct,
        stage1_rows = stage1_len,
        total_rows,
        "stage 1 fast-evaluation slice"
    );
    let ohlcv_stage1 = slice_ohlcv(&ohlcv, stage1_start, stage1_end);
    let features_stage1 = features.row_window(stage1_start, stage1_end)?;
    // Borrow the already-admitted run's immutable route facts. This performs
    // no device operation and therefore cannot select a different ordinal or
    // observe a post-parent free-memory value. Full resident-parent rows size
    // fixed VRAM; the exact Stage1 range above sizes evaluation time.
    let sizing_primitives = population_execution_run
        .population_auto_sizing_primitives_v1()
        .map_err(anyhow::Error::new)?;
    let stage1_sizing_window =
        crate::population_auto_sizing_receipt_v1::seal_population_auto_stage1_window_v1(
            &sizing_primitives.parent_dataset_identity_sha256,
            "selection_stage1",
            stage1_start,
            stage1_end,
        )
        .map_err(anyhow::Error::new)?;
    let migration_enabled_for_run = crate::genetic::migration_enabled();
    let month_capacity = crate::eval::current_backtest_runtime_overrides().month_capacity;
    let stage1_evaluation_config = config.evaluation_config(ohlcv_stage1.close.last().copied());
    let exact_stage1_view =
        crate::exact_resident_dataset_authority_v1::ExactResidentDatasetViewRequestV1::ContiguousRange {
            start: stage1_start,
            end: stage1_end,
        };
    // CPU auto-sizing is a measurement of this exact timeframe view, after the
    // Stage-1 feature cache exists and inside the already-installed Rayon pool.
    // The returned cache is retained for the real search so calibration does
    // not turn into a second transpose/SMC build. Native CUDA keeps using its
    // admission-bound planner and therefore returns no host calibration.
    let prepared_cpu_population_auto = if config.population_auto {
        crate::genetic::search_engine::prepare_exact_cpu_population_auto_v1(
            &features_stage1,
            &ohlcv_stage1,
            config.population,
            config.max_indicators,
            month_capacity,
            &stage1_evaluation_config,
            &population_execution_run,
            exact_stage1_view,
            &sizing_primitives.route,
        )?
    } else {
        None
    };
    let cpu_plan = prepared_cpu_population_auto
        .as_ref()
        .map(|prepared| prepared.cpu_plan.clone());
    let population_auto_sizing_receipt =
        crate::population_auto_sizing_receipt_v1::seal_population_auto_sizing_receipt_v1(
            crate::population_auto_sizing_receipt_v1::PopulationAutoSizingRequestV1 {
                population_auto: config.population_auto,
                configured_population: config.population,
                resident_parent_rows: sizing_primitives.resident_parent_rows,
                evaluation_rows: ohlcv_stage1.close.len(),
                feature_count: sizing_primitives.feature_count,
                month_capacity,
                requested_max_indicators: config.max_indicators,
                migration_enabled: migration_enabled_for_run,
                parent_canonical_scope_identity_sha256: sizing_primitives
                    .parent_canonical_scope_identity_sha256,
                parent_dataset_identity_sha256: sizing_primitives.parent_dataset_identity_sha256,
                stage1_window: stage1_sizing_window,
                route: sizing_primitives.route,
                cpu_plan,
            },
        )
        .map_err(anyhow::Error::new)?;
    let search_authority = crate::run_identity::build_population_auto_search_authority_v1(
        config,
        &population_auto_sizing_receipt,
        stage1_evaluation_config.pip_value_per_lot,
        features.normalization_fitted_state().is_some(),
    )?;
    // Freeze the position-management contract at the same authority boundary
    // as the search hash. The live artifact later copies this exact value; it
    // must never reconstruct exits from whatever config happens to be current
    // after a multi-hour discovery run.
    let resolved_adaptive_stops_policy =
        crate::stop_target::ResolvedAdaptiveStopsPolicyV1::capture_current()?;
    funnel.attach_live_trading_policy_v1(
        crate::live_portfolio::LiveTradingPolicyV1::from_search_authority(
            &search_authority,
            &stage1_evaluation_config,
            crate::genetic::smc_gate_disabled(),
            &resolved_adaptive_stops_policy,
        )?,
    )?;
    let ga_population = population_auto_sizing_receipt.resolved_population();
    if ga_population != config.population {
        tracing::warn!(
            target: "neoethos_search::funnel",
            configured = config.population,
            resolved = ga_population,
            term_cap = population_auto_sizing_receipt.term_cap(),
            reason = population_auto_sizing_receipt.resolution_reason(),
            sizing_receipt_sha256 = population_auto_sizing_receipt.identity_sha256(),
            "population_auto widened the exact receipt-governed search"
        );
    } else {
        tracing::info!(
            target: "neoethos_search::funnel",
            configured = config.population,
            resolved = ga_population,
            population_auto = config.population_auto,
            reason = population_auto_sizing_receipt.resolution_reason(),
            sizing_receipt_sha256 = population_auto_sizing_receipt.identity_sha256(),
            "exact population sizing receipt kept the configured search population"
        );
    }
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

    // One immutable configuration identity for every persistent search-state
    // edge. This is resolved only after population auto-sizing, so the prior
    // ledger, trial-return matrix, and new ledger all name the search that
    // actually runs rather than the caller's unresolved request.
    let search_state_config_hash = search_authority.search_config_hash().to_owned();

    // Search-memory + weekly-refresh: seed only from state addressed by this
    // exact receipt and resolved config. Corruption, a legacy unbound ledger, or
    // any embedded identity mismatch is fatal rather than treated as absence.
    // Clear a one-shot hand-off left by any earlier aborted discovery on this
    // worker before binding the current exact receipt/config ledger.
    crate::genetic::evolution_math::clear_staged_seen_signature_hashes_on_this_thread();
    if config.discovery_ledger_enabled {
        if let Some(prior) = crate::discovery_ledger::load_prior_ledger(
            &config.discovery_ledger_cache_dir,
            &config.evaluation_symbol,
            &config.timeframe_label,
            search_input_receipt,
            &search_state_config_hash,
            &search_authority.resolved_config_stamp().config_hash,
        )? {
            let mut seen = crate::genetic::SeenSignatureMemory::current();
            let seen_has_file = seen.file_path.is_some();
            let prior_total = prior.portfolio.len() + prior.archive.len();
            let inserted = crate::discovery_ledger::seed_seen_from_ledger(&prior, &mut seen);
            seen.flush();
            let handoff = if seen_has_file {
                "configured seen-signature file"
            } else {
                "one-shot thread-local engine hand-off"
            };
            tracing::info!(
                target: "neoethos_search::discovery_ledger",
                symbol = %config.evaluation_symbol,
                tf = %config.timeframe_label,
                receipt_sha256 = %prior.search_input_receipt_sha256,
                config_hash = %prior.config_hash,
                prior_total,
                seeded = inserted,
                handoff,
                "seeded GA seen-set from exact receipt/config discovery ledger"
            );
        }
    }
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
    let search = evolve_search_with_progress_and_limits_exact(
        &features_stage1,
        &ohlcv_stage1,
        ga_population,
        config.generations,
        config.max_indicators,
        max_runtime,
        Some(stage1_evaluation_config.clone()),
        &population_execution_run,
        exact_stage1_view,
        &search_authority,
        prepared_cpu_population_auto.map(|prepared| prepared.eval_cache),
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

    let mut result = finalize_candidates_with_progress(
        search.genes,
        search.metrics,
        &stage1_evaluation_config,
        &features_stage1.timestamps,
        &features,
        &ohlcv,
        search_input_receipt,
        selection_scope,
        inputs.calibration().map(ScopedDiscoveryInput::scope),
        inputs.calibration(),
        inputs.holdout().map(ScopedDiscoveryInput::scope),
        &search_state_config_hash,
        config,
        effective_smc_gate_threshold,
        effective_feature_names,
        &population_execution_run,
        &mut funnel,
        progress_fn,
    )?;
    let population_execution_run_receipt_v2 = population_execution_run
        .finish()
        .map_err(anyhow::Error::new)?;
    tracing::info!(
        target: "neoethos_search::engine",
        receipt_sha256 = %population_execution_run_receipt_v2.identity_sha256(),
        engines = ?population_execution_run_receipt_v2.engines(),
        successful_populations = population_execution_run_receipt_v2
            .engine_receipt_v1()
            .successful_population_count(),
        "closed the run-scoped exact population-engine receipt"
    );
    result
        .funnel_profile
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("discovery result lost its run-scoped funnel carrier"))?
        .attach_population_execution_run_receipt_v2(population_execution_run_receipt_v2)
        .map_err(anyhow::Error::msg)?;

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
            search_input_receipt,
            &search_authority,
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

/// Everything `prefilter_features` needs, gathered at the one call site so the
/// function has no ambient inputs.
#[derive(Debug, Clone)]
pub(crate) struct PrefilterSpec {
    pub top_k: usize,
    /// Fraction of the already-isolated selection window used for the cheap
    /// feature ranking. The label horizon is removed from its right edge.
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
}

/// One numerical authority for the CPU and resident-CUDA prefilter label
/// geometry. The GPU path consumes these resolved scalars; it never rebuilds
/// cost conversion or stop-band midpoints from a second formula.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ResolvedPrefilterFinancialGeometryV1 {
    pub(crate) max_hold_bars: usize,
    pub(crate) stop_atr_multiplier: f64,
    pub(crate) reward_risk_ratio: f64,
    pub(crate) round_trip_cost_price: f64,
}

pub(crate) fn resolve_prefilter_financial_geometry_v1(
    config: &DiscoveryConfig,
    price_hint: Option<f64>,
) -> ResolvedPrefilterFinancialGeometryV1 {
    let evaluation = config.evaluation_config(price_hint);
    resolve_prefilter_financial_geometry_from_evaluation_v2(&evaluation)
}

/// Resolve the prefilter barriers from an already-sealed evaluation authority.
/// The canonical native route obtains this value from its explicit financial
/// contract before CUDA admission, so rebuilding it from ambient settings here
/// would create a second, conflicting cost authority.
pub(crate) fn resolve_prefilter_financial_geometry_from_evaluation_v2(
    evaluation: &crate::genetic::EvaluationConfig,
) -> ResolvedPrefilterFinancialGeometryV1 {
    let pip = if evaluation.pip_value.is_finite() && evaluation.pip_value > 0.0 {
        evaluation.pip_value
    } else {
        // The cost-model guard upstream has already rejected a non-finite
        // spread. Preserve the historical zero-cost fallback for an exotic
        // with no usable pip size, but state it at the shared authority.
        tracing::warn!(
            target: "neoethos_search::discovery",
            pip_value = evaluation.pip_value,
            "prefilter label: no usable pip size — the first-passage barriers carry NO \
             cost, so the label is optimistic about which trades would have paid"
        );
        0.0
    };
    let commission_pips = if evaluation.pip_value_per_lot.is_finite()
        && evaluation.pip_value_per_lot > 0.0
        && evaluation.commission_per_trade.is_finite()
    {
        evaluation.commission_per_trade / evaluation.pip_value_per_lot
    } else {
        0.0
    };
    let round_trip_cost_price = (evaluation.spread_pips.max(0.0) + commission_pips.max(0.0)) * pip;
    let bounds = crate::genetic::current_gene_stop_bounds();
    let midpoint_rr = 0.5 * (bounds.rr_min + bounds.rr_max);
    let (stop_atr_multiplier, reward_risk_ratio) = match bounds.atr_pips {
        Some(atr_pips) if atr_pips.is_finite() && atr_pips > 0.0 => {
            let midpoint_stop_pips = 0.5 * (bounds.sl_min_pips + bounds.sl_max_pips);
            let multiplier = midpoint_stop_pips / atr_pips;
            if multiplier.is_finite()
                && multiplier > 0.0
                && midpoint_rr.is_finite()
                && midpoint_rr > 0.0
            {
                (multiplier, midpoint_rr)
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
    };
    ResolvedPrefilterFinancialGeometryV1 {
        max_hold_bars: if evaluation.max_hold_bars > 0 {
            evaluation.max_hold_bars
        } else {
            35
        },
        stop_atr_multiplier,
        reward_risk_ratio,
        round_trip_cost_price,
    }
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
    /// Columns excluded because the selection window produced no rankable correlation (too
    /// few pairwise-complete rows, or zero variance). NOT scored 0.0 and left
    /// to compete — named and dropped.
    pub columns_unrankable: usize,
    /// A few unrankable column names, for the log line.
    pub unrankable_sample: Vec<String>,
    /// Kept for run-log compatibility. Prefilter now uses exactly one fit
    /// window; CPCV is reported by the post-GA validation census instead.
    pub refit_folds_used: usize,
    pub refit_folds_available: usize,
    /// Kept for run-log compatibility; always zero for a single fit window.
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
    long: Vec<f64>,
    short: Vec<f64>,
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
        let prev_close = if i > 0 {
            ohlcv.close[i - 1]
        } else {
            ohlcv.close[i]
        };
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
    let mut long_labels = vec![f64::NAN; n];
    let mut short_labels = vec![f64::NAN; n];
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

        let mut long_label = 0.0f64;
        let mut short_label = 0.0f64;
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

/// The one row-index set used by the cheap pre-GA feature ranking.
///
/// The caller has already removed the untouched outer holdout. The additional
/// leading fraction is operator-configurable, and the final label horizon is
/// excluded so a first-passage label cannot read beyond the fitted window.
/// CPCV deliberately does not run here: applying it to every raw column is both
/// expensive and not nested-clean; the full CPCV/PBO gate remains post-GA.
fn prefilter_fit_windows(n_rows: usize, spec: &PrefilterSpec) -> (Vec<Vec<usize>>, usize) {
    let requested_end = ((n_rows as f64) * spec.insample_frac).floor() as usize;
    let fit_end = requested_end.min(n_rows);
    let label_safe_end = fit_end.saturating_sub(spec.max_hold_bars.max(1));
    (vec![(0..label_safe_end).collect()], 0)
}

fn prefilter_features(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    spec: &PrefilterSpec,
) -> Result<(FeatureFrame, PrefilterCensus)> {
    let n_rows = features.n_samples();
    let n_cols = features.n_features();
    if n_rows < 2 || n_cols <= spec.top_k {
        let census = PrefilterCensus {
            columns_considered: n_cols,
            columns_kept: n_cols,
            ..PrefilterCensus::default()
        };
        return Ok((features.clone(), census));
    }
    let sealed_schema =
        crate::prefilter_schema_v1::seal_prefilter_column_classification_v1(&features.names)
            .ok_or_else(|| {
                anyhow::anyhow!("cannot seal an empty or overflowing prefilter schema")
            })?;
    if sealed_schema.ordered_feature_schema_sha256() == [0; 32]
        || sealed_schema.column_classification_content_sha256() == [0; 32]
        || sealed_schema
            .timeframe_group_ids()
            .iter()
            .copied()
            .max()
            .is_some_and(|group| u64::from(group) > sealed_schema.timeframe_group_count())
        || sealed_schema
            .column_class_flags()
            .iter()
            .zip(sealed_schema.template_force_keep_flags())
            .any(|(class, force_keep)| {
                (*class & crate::prefilter_schema_v1::COLUMN_CLASS_TEMPLATE_V1 != 0)
                    != (*force_keep != 0)
            })
    {
        anyhow::bail!("prefilter schema classification is internally inconsistent");
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
        let mut returns = vec![f64::NAN; n];
        for i in 0..n.saturating_sub(1) {
            let denom = ohlcv.close[i];
            if denom.abs() > 1e-12 {
                returns[i] = (ohlcv.close[i + 1] - denom) / denom;
            }
        }
        labels = returns;
        // The forward return has no direction pair; scoring against a stale
        // short label would silently mix two targets.
        short_labels = None;
    }
    let short_labels = short_labels;

    // Cheap selection happens once, before the GA, on the selection window only.
    // Its right edge is embargoed by `max_hold_bars`, so the forward-looking
    // first-passage label cannot cross that fit boundary. Running CPCV here used
    // to multiply this full column scan by up to eight while still failing the
    // requirement for a truly nested feature-selection estimate. CPCV/PBO stays
    // in the post-GA validation path, where it evaluates actual finalists.
    let (windows, folds_available) = prefilter_fit_windows(n_rows, spec);
    census.refit_folds_used = windows.len();
    census.refit_folds_available = folds_available;
    let usable_selection_rows = windows.first().map_or(0, Vec::len);
    if usable_selection_rows < 3 {
        // A short fixture or a hold horizon longer than the available
        // selection prefix contains no rankable pair. Deleting every ordinary
        // feature in that situation would manufacture a narrow search from an
        // absence of evidence. Preserve the frame and let the normal search /
        // validation path report that the dataset is too short.
        census.columns_kept = n_cols;
        tracing::warn!(
            target: "neoethos_search::prefilter",
            n_rows,
            requested_insample_fraction = spec.insample_frac,
            max_hold_bars = spec.max_hold_bars,
            usable_selection_rows,
            columns_kept = n_cols,
            "prefilter selection window is too short to rank features; preserving the complete \
             frame instead of excluding columns on an empty label-safe sample"
        );
        return Ok((features.clone(), census));
    }

    struct ColumnScore {
        idx: usize,
        score: f64,
        instability: f64,
        had_nonfinite: bool,
        rankable: bool,
    }

    let score_column = |col_idx: usize, col: &neoethos_data::FeatureColumnF64| -> ColumnScore {
        if sealed_schema.column_class_flags()[col_idx]
            & crate::prefilter_schema_v1::COLUMN_CLASS_STATE_V1
            != 0
        {
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
        let mut worst = f64::INFINITY;
        let mut best = 0.0f64;
        let mut had_nonfinite = false;
        let mut rankable_in_all = true;
        for window in &windows {
            let mut xs: Vec<f64> = Vec::with_capacity(window.len());
            let mut ys: Vec<f64> = Vec::with_capacity(window.len());
            let mut ys_short: Vec<f64> = Vec::with_capacity(window.len());
            for &row in window {
                // The label series is bar-indexed and the feature cube is
                // row-indexed; they are the same length in production, but a
                // caller that hands over mismatched lengths must lose the
                // extra rows rather than index out of bounds. The two lives
                // in the same guard so neither can be forgotten.
                if row >= n_rows || row >= labels.len() {
                    continue;
                }
                xs.push(if col.validity[row].is_valid() {
                    col.values[row]
                } else {
                    f64::NAN
                });
                ys.push(labels[row]);
                if let Some(short) = short_labels.as_ref() {
                    ys_short.push(short.get(row).copied().unwrap_or(f64::NAN));
                }
            }
            let outcome = neoethos_data::core::stats_f64::pearson_pairwise(&xs, &ys);
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
                    neoethos_data::core::stats_f64::pearson_pairwise(&xs, &ys_short);
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
    };

    // Bound I/O and residency from the actual frame size and live allocation
    // headroom, then keep up to the configured Rayon width busy. The old path issued
    // 779 serial one-column Vortex projections before the parallel arithmetic
    // began and retained the complete decoded cube in RAM. Each wave below has
    // at most `concurrent_batches` live projections; every projection contains
    // multiple physical columns and is dropped immediately after scoring.
    let projection_plan =
        neoethos_data::adaptive_feature_projection_plan(features, rayon::current_num_threads())?;
    tracing::info!(
        target: "neoethos_search::prefilter",
        rows = n_rows,
        columns = n_cols,
        columns_per_projection = projection_plan.columns_per_batch,
        concurrent_projections = projection_plan.concurrent_batches,
        projection_budget_bytes = projection_plan.budget_bytes,
        "prefilter scoring uses adaptive bounded parallel Vortex projections"
    );
    let columns_per_batch = projection_plan.columns_per_batch;
    let wave_columns = columns_per_batch * projection_plan.concurrent_batches;
    let column_indices = (0..n_cols).collect::<Vec<_>>();
    let mut scored = Vec::with_capacity(n_cols);
    for wave in column_indices.chunks(wave_columns) {
        let wave_scored = wave
            .par_chunks(columns_per_batch)
            .map(|indices| -> Result<Vec<ColumnScore>> {
                let projection = features.project_columns(indices, 0..n_rows)?;
                anyhow::ensure!(
                    projection.timestamps.as_slice() == features.timestamps.as_slice(),
                    "prefilter projection timestamps changed"
                );
                anyhow::ensure!(
                    projection.columns.len() == indices.len(),
                    "prefilter projection returned {} columns for {} indices",
                    projection.columns.len(),
                    indices.len()
                );
                indices
                    .iter()
                    .copied()
                    .zip(&projection.columns)
                    .map(|(column_index, column)| {
                        anyhow::ensure!(
                            column.name == features.names[column_index],
                            "prefilter column {column_index} materialized as `{}` instead of `{}`",
                            column.name,
                            features.names[column_index]
                        );
                        Ok(score_column(column_index, column))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        for batch in wave_scored {
            scored.extend(batch);
        }
    }
    anyhow::ensure!(
        scored.len() == n_cols,
        "prefilter scored {} columns for a {n_cols}-column frame",
        scored.len()
    );

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
    census.regime_forced = sealed_schema
        .column_class_flags()
        .iter()
        .filter(|flags| **flags & crate::prefilter_schema_v1::COLUMN_CLASS_STATE_V1 != 0)
        .count();
    {
        let mut per_family: Vec<(&str, usize)> = Vec::new();
        for family in crate::prefilter_schema_v1::PREFILTER_STATE_FAMILIES_V1 {
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
            let mut per_group: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::new();
            for &idx in &keep_indices {
                let group = sealed_schema.timeframe_group_ids()[idx];
                if group != 0 {
                    *per_group.entry(group).or_insert(0) += 1;
                }
            }
            for &(idx, _) in &correlations {
                let group = sealed_schema.timeframe_group_ids()[idx];
                if group == 0 {
                    continue;
                }
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
        for (idx, force_keep) in sealed_schema
            .template_force_keep_flags()
            .iter()
            .copied()
            .enumerate()
        {
            if force_keep != 0 {
                kept.insert(idx);
            }
        }
        keep_indices = kept.into_iter().collect();
    }

    keep_indices.sort(); // Maintain original order
    keep_indices.dedup();
    let n_keep = keep_indices.len();
    census.columns_kept = n_keep;

    Ok((features.select_columns(&keep_indices)?, census))
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

    // Project each regime column once. Re-reading a Vortex projection for
    // every trade dominated this validation stage and mixed storage I/O into
    // the hot arithmetic loop.
    let trend_column = match features.feature_column(t_idx) {
        Ok(column) => column,
        Err(error) => {
            tracing::error!(?error, "failed to read trend-regime feature column");
            return false;
        }
    };
    let volatility_column = match features.feature_column(v_idx) {
        Ok(column) => column,
        Err(error) => {
            tracing::error!(?error, "failed to read volatility-regime feature column");
            return false;
        }
    };

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

        if !trend_column.validity[idx].is_valid() || !volatility_column.validity[idx].is_valid() {
            // A trade whose regime inputs are undefined cannot prove regime
            // robustness. Reject the candidate instead of treating undefined
            // as zero or silently omitting the trade.
            return false;
        }
        let trend_str = trend_column.values[idx];
        let vol_state = volatility_column.values[idx];

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
    confidences: &[f64],
    ohlcv: &Ohlcv,
    timestamps: &[i64],
    config: &DiscoveryConfig,
    overrides: &PropFirmGateOverrides,
    resolver: &GeneEvalSettingsResolver<'_>,
    windows: &[PropFirmWindow],
) -> Result<(f64, usize)> {
    let rows = ohlcv.close.len();
    anyhow::ensure!(
        signals.len() == rows
            && confidences.len() == rows
            && timestamps.len() == rows
            && ohlcv.high.len() == rows
            && ohlcv.low.len() == rows,
        "prop-firm windows require exactly aligned OHLC, timestamps, signals and confidence"
    );
    if windows.is_empty() {
        return Ok((0.0, 0));
    }
    let mut settings = resolver.settings_for_gene(gene);
    let initial_balance = config.initial_account_balance()?.amount();

    let mut passes = 0usize;
    let mut counted = 0usize;
    for (start_idx, end_idx, window_base) in windows {
        let (start_idx, end_idx) = (*start_idx, *end_idx);
        anyhow::ensure!(
            start_idx < end_idx && end_idx <= rows,
            "prop-firm window {start_idx}..{end_idx} exceeds its exact {rows}-row series"
        );
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
        let conf = &confidences[start_idx..end_idx];
        let trades =
            simulate_trades_with_confidence_core(close, high, low, ts, sig, conf, &settings)?;
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
        return Ok((0.0, 0));
    }
    Ok((passes as f64 / counted as f64, counted))
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
) -> Result<(Vec<(usize, Gene)>, usize)> {
    // An empty pool pays nothing. `build_smc_arrays` scans every bar of the
    // series (~90 f64 ops each) before it knows there is no gene to gate, and
    // an empty prefilter is the normal outcome of a run that found nothing —
    // which is most M3 runs today.
    if prefiltered.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let smc = SmcGateArrays::build(features, ohlcv)?;
    let nonzero_signal_count = std::sync::atomic::AtomicUsize::new(0);
    let survivors = crate::post_ga::map_bounded(
        prefiltered,
        features.n_samples(),
        |(candidate_idx, gene)| -> Result<Option<(usize, Gene)>> {
            let sig = signals_for_gene_full_with_smc(features, &gene, eval_config, &smc)?;
            let trade_count = sig.iter().filter(|v| **v != 0).count() as f64;
            if trade_count > 0.0 {
                nonzero_signal_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if trade_count >= min_trades as f64 {
                // Directions are deterministic from this frozen gene/input.
                // Retaining every full tape here made automatic coverage a
                // RAM limit disguised as a candidate-count limit.
                Ok(Some((candidate_idx, gene)))
            } else {
                Ok(None)
            }
        },
    )?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    Ok((
        survivors,
        nonzero_signal_count.load(std::sync::atomic::Ordering::Relaxed),
    ))
}

fn finalize_candidates_with_progress<F>(
    candidates: Vec<Gene>,
    candidate_metrics: Vec<[f64; 11]>,
    stage1_evaluation: &EvaluationConfig,
    stage1_timestamps: &[i64],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    search_input_receipt: &CanonicalSearchInputReceiptV2,
    selection_scope: &CanonicalSearchArtifactScopeV2,
    calibration_scope: Option<&CanonicalSearchArtifactScopeV2>,
    calibration_input: Option<&ScopedDiscoveryInput<'_>>,
    holdout_scope: Option<&CanonicalSearchArtifactScopeV2>,
    search_state_config_hash: &str,
    config: &DiscoveryConfig,
    effective_smc_gate_threshold: f64,
    effective_feature_names: Vec<String>,
    population_execution_run: &crate::population_execution_evidence_v1::ExactPopulationExecutionRunV1<'_>,
    funnel: &mut crate::funnel_profile::FunnelProfile,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    // Diagnostic: summarise the feature frame so we can tell whether the GA's
    // empty-portfolio outcome is downstream filtering or broken upstream
    // features. Each column is projected exactly ONCE and columns are processed
    // in parallel. The previous implementation projected every column again for
    // the trailing variance pass, serialising a second Vortex read/decompression
    // of the complete working set after the expensive search had already run.
    {
        #[derive(Debug)]
        struct ColumnDiagnostic {
            invalid_or_non_finite: usize,
            zero: usize,
            finite: usize,
            sum_abs: f64,
            min: f64,
            max: f64,
            trailing_zero_variance: bool,
        }

        let total = features.n_values();
        let n_cols = features.n_features();
        let trailing = features.n_samples().min(1000);
        let trailing_start = features.n_samples().saturating_sub(trailing);
        let per_column = (0..n_cols)
            .into_par_iter()
            .map(|column_index| -> Result<ColumnDiagnostic> {
                let column = features.feature_column(column_index)?;
                let mut diagnostic = ColumnDiagnostic {
                    invalid_or_non_finite: 0,
                    zero: 0,
                    finite: 0,
                    sum_abs: 0.0,
                    min: f64::INFINITY,
                    max: f64::NEG_INFINITY,
                    trailing_zero_variance: false,
                };
                let mut trailing_min = f64::INFINITY;
                let mut trailing_max = f64::NEG_INFINITY;
                let mut trailing_finite = 0usize;

                for (row, (value, validity)) in
                    column.values.iter().zip(&column.validity).enumerate()
                {
                    // Validity and IEEE finiteness are independent. Treat a
                    // validity-marked NaN/Inf as broken data rather than letting
                    // it poison `sum_abs` and silently disappear from min/max.
                    if !validity.is_valid() || !value.is_finite() {
                        diagnostic.invalid_or_non_finite += 1;
                        continue;
                    }
                    diagnostic.finite += 1;
                    diagnostic.sum_abs += value.abs();
                    diagnostic.min = diagnostic.min.min(*value);
                    diagnostic.max = diagnostic.max.max(*value);
                    if *value == 0.0 {
                        diagnostic.zero += 1;
                    }

                    if trailing > 1 && row >= trailing_start {
                        trailing_finite += 1;
                        trailing_min = trailing_min.min(*value);
                        trailing_max = trailing_max.max(*value);
                    }
                }

                diagnostic.trailing_zero_variance = trailing > 1
                    && trailing_finite >= (trailing * 7 / 10)
                    && trailing_min.is_finite()
                    && trailing_max.is_finite()
                    && (trailing_max - trailing_min).abs() < 1e-9;
                Ok(diagnostic)
            })
            .collect::<Result<Vec<_>>>()?;

        // Aggregate in stable column order so the floating-point diagnostic is
        // reproducible even though projection/decompression ran in parallel.
        let mut invalid_or_non_finite = 0usize;
        let mut zero = 0usize;
        let mut finite_count = 0usize;
        let mut sum_abs = 0.0_f64;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        let mut zero_var_cols = 0usize;
        let mut named_examples: Vec<String> = Vec::new();
        for (column_index, diagnostic) in per_column.iter().enumerate() {
            invalid_or_non_finite += diagnostic.invalid_or_non_finite;
            zero += diagnostic.zero;
            finite_count += diagnostic.finite;
            sum_abs += diagnostic.sum_abs;
            min_v = min_v.min(diagnostic.min);
            max_v = max_v.max(diagnostic.max);
            if diagnostic.trailing_zero_variance {
                zero_var_cols += 1;
                if named_examples.len() < 5 {
                    if let Some(name) = features.names.get(column_index) {
                        named_examples.push(name.clone());
                    }
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
            cols = n_cols,
            diagnostic_workers = rayon::current_num_threads(),
            nan_frac = invalid_or_non_finite as f64 / total.max(1) as f64,
            zero_frac = zero as f64 / total.max(1) as f64,
            min_finite = if min_v.is_finite() { min_v } else { 0.0 },
            max_finite = if max_v.is_finite() { max_v } else { 0.0 },
            mean_abs_finite = mean_abs,
            "feature frame summary"
        );

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
    // Every initial score uses its paired measured GA metrics and exactly the
    // GA slice's timestamps. Never divide Stage1 evidence by the full IS span.
    let risky_ranking = matches!(config.mode, DiscoveryMode::Risky);
    let growth_goal = if risky_ranking {
        Some(stage1_evaluation.growth_goal.ok_or_else(|| {
            anyhow::anyhow!("Risky candidate ranking lost its configured growth goal")
        })?)
    } else {
        None
    };
    let mut ranked_candidates = rank_candidates_on_matching_window(
        candidates,
        candidate_metrics,
        stage1_evaluation.initial_equity,
        stage1_timestamps,
        growth_goal,
    )?;
    let ga_returned_candidates = ranked_candidates.len();
    let max_candidates =
        candidate_truncation_limit(config.candidate_count, ranked_candidates.len());
    ranked_candidates.truncate(max_candidates);
    let renamed_candidate_ids = crate::post_ga::disambiguate_candidate_ids(&mut ranked_candidates);
    if renamed_candidate_ids > 0 {
        tracing::info!(
            renamed_candidate_ids,
            "disambiguated colliding candidate display IDs before quality and validation artifacts; ranked order and genomes unchanged"
        );
    }
    let ranked_candidate_genes: Vec<Gene> = ranked_candidates
        .iter()
        .map(|(_, gene)| gene.clone())
        .collect();
    progress_fn(DiscoveryProgress::CandidatesRanked {
        candidate_count: ga_returned_candidates,
        truncated_to: max_candidates,
    });
    let mut candidate_census = crate::funnel_profile::DiscoveryCandidateCensus {
        ga_returned_candidates,
        validation_candidate_limit: config.candidate_count,
        validation_candidates_admitted: max_candidates,
        validation_candidates_capped: ga_returned_candidates.saturating_sub(max_candidates),
        walkforward_not_tested: max_candidates,
        ..Default::default()
    };
    funnel.record_stage(
        "validation_candidates_admitted",
        ga_returned_candidates,
        max_candidates,
    );
    if candidate_census.validation_candidates_capped > 0 {
        funnel.add_reject_reason(
            "validation_candidates_admitted",
            "validation_candidates_capped",
            candidate_census.validation_candidates_capped,
        );
    }
    progress_fn(DiscoveryProgress::CandidateCensusUpdated {
        census: candidate_census.clone(),
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
    let batch_verdict =
        evaluate_batch_early_reject(&ranked_candidate_genes, &config.target_profile);
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

    let (mut filtered, post_nonzero_signal) = screen_candidates_by_signal_count(
        features,
        ohlcv,
        prefiltered,
        &eval_config_for_signals,
        min_trades,
    )?;
    let post_min_trades = filtered.len();
    // Account-money replays share one exact SMC preparation. Confidence is
    // regenerated only inside each admitted parallel candidate, not retained as
    // an eight-byte-per-cell matrix for the entire potentially large archive.
    let account_smc = (!filtered.is_empty())
        .then(|| SmcGateArrays::build(features, ohlcv))
        .transpose()?;
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
    // Capture the top-by-fitness base-filtered survivors here,
    // at the richest point before the strict quality / prop-firm / correlation
    // gates can empty the portfolio. If those gates reject EVERY candidate, we
    // promote this set (correlation-pruned, honestly labeled "did not pass the
    // prop bar") so a hard combo (e.g. AUDUSD M3) emits its best-found genes
    // instead of dying with zero output. Cloning only the top N keeps it cheap.
    const FALLBACK_PORTFOLIO_MAX: usize = 8;
    let best_effort_fallback: Vec<(usize, Gene)> = {
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
            .map(|i| filtered[i].clone())
            .collect()
    };
    progress_fn(DiscoveryProgress::CandidatesFiltered {
        passed_filters: filtered.len(),
        evaluated_candidates: ranked_candidates.len(),
        min_trades_required: min_trades,
    });

    let filtered_count = filtered.len();
    let mut quality_metrics = Vec::new();
    let mut quality_candidate_indices = Vec::new();
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
    let mut logged_candidate_indices = Vec::new();
    let mut ranked_diagnostic_candidates = Vec::new();
    // Profit validity is not an observability switch. `log_trades` used to be
    // one of the conditions that decided whether this entire cost-aware replay
    // ran, so turning logging off could also turn off the positive-expectancy
    // decision. Run the base quality screen whenever candidates exist; logging
    // controls only whether trade details are retained.
    if !filtered.is_empty() {
        let candidate_deep_robustness = !matches!(config.mode, DiscoveryMode::Risky);
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "quality_screen",
            detail: if candidate_deep_robustness {
                format!(
                    "cost-aware replay of {filtered_count} post-GA finalists + candidate-level \
                     robustness; final portfolio validation follows"
                )
            } else {
                format!(
                    "cost-aware replay of {filtered_count} post-GA finalists; Risky mode omits \
                     redundant candidate-level regime/parameter/sensitivity stress, while the \
                     selected portfolio still receives permutation, parameter-plateau, \
                     walk-forward and CPCV/PBO validation"
                )
            },
        });
        /// The COST-BAND VERDICT — see [`CostBandVerdict`] — rides on
        /// the survivor so the report cannot lose it between the screen and the
        /// export, which is how "we measured the band" becomes "we mentioned
        /// the band once in a log".
        struct QualityCandidate {
            candidate_idx: usize,
            gene: Gene,
            metrics: StrategyMetrics,
            ranking_score: f64,
            opportunistic: bool,
            cost_band: CostBandVerdict,
        }
        let analyzer = quality_analyzer_for_config(config);
        let initial_balance = config.initial_balance;
        let quality_start_ms = features.timestamps.first().copied().ok_or_else(|| {
            anyhow::anyhow!("quality replay requires an evaluation start timestamp")
        })?;
        let quality_end_ms = features.timestamps.last().copied().ok_or_else(|| {
            anyhow::anyhow!("quality replay requires an evaluation end timestamp")
        })?;
        let (quality_months, quality_days) = month_day_indices(&features.timestamps);

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

        let pairs = filtered;

        // ONE resolver for every serial backtest in this screen (built over
        // the full series the screen simulates) and ONE template source for
        // the population launches (which resolve adaptive stops themselves).
        // This is THE fix for the screen's measured 17.6x divergence: the
        // base-quality backtest below previously ran adaptive genes on their
        // unused fixed pips while GA scoring ran them volatility-scaled.
        let screen_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            pairs.iter().map(|(_, gene)| gene),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        let screen_templates = PopulationTemplateResolver::new(config, ohlcv.close.last().copied());

        // Decide whether any scenario launch below can add information BEFORE
        // paying for its full-history transposed feature/SMC preparation. In
        // Risky mode the candidate-level robustness battery is intentionally
        // omitted, and a cost band at or below the already charged cost is
        // monotone and therefore incapable of discriminating. That common path
        // needs only the direct cost-aware replay and must not spend ~23 seconds
        // constructing an input that no evaluator will read.
        let baseline_cost_pips = crate::run_identity::cost_pips_round_trip(
            config.evaluation_spread_pips,
            config.evaluation_commission_per_trade,
            eval_config_for_signals.pip_value_per_lot,
        );
        let band_discriminates = cost_band_discriminates(config.cost_band_pips, baseline_cost_pips);

        // The bar-derived half of validation host prep, built ONCE for the whole
        // screen when at least one scenario launch will actually consume it.
        //
        // This was rebuilt on every call: the transposed indicator matrix, the
        // month/day indices and eleven lookback-heavy SMC series, over the full
        // history, seven times. None of it depends on which genes are being
        // evaluated. `validation_genes_population`'s own in-tree measurement
        // reads "eighteen of these calls take 413.6 s of a 452.4 s run — 23 s
        // each — while the device stage timing inside one adds up to 0.30 s";
        // this is a large part of the 22.7 s nobody could account for.
        let screen_prep = (candidate_deep_robustness || band_discriminates)
            .then(|| crate::genetic::search_engine::ValidationPrep::build(features, ohlcv))
            .transpose()?;

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
        let mc_runs = if candidate_deep_robustness {
            config.mc_runs as usize
        } else {
            0
        };
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
        let fuse_costs = candidate_deep_robustness
            && sensitivity_spread.is_some()
            && sensitivity_commission.is_some();
        if candidate_deep_robustness && !fuse_costs {
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
            candidate_deep_robustness,
            device_monte_carlo = device_mc,
            screen_chunk,
            launches = candidates.div_ceil(screen_chunk.max(1)),
            scenarios_per_chunk = screen_chunk * (mc_runs + usize::from(fuse_costs)),
            fused_cost_pass = fuse_costs,
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
            let mut screen_genes: Vec<Gene> = chunk.iter().map(|(_, gene)| gene.clone()).collect();
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
                    .map(|(candidate_idx, gene)| {
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
            for (position, (candidate_idx, _)) in chunk.iter().enumerate() {
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
                let screen_settings = screen_templates.template(&chunk[0].1);
                match crate::genetic::search_engine::prepare_validation_population(
                    ohlcv,
                    &screen_genes,
                    &eval_config_for_signals,
                    &screen_settings,
                ) {
                    Ok(prepared) => {
                        #[cfg(feature = "gpu")]
                        let _gpu_guard = GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                        crate::genetic::search_engine::validation_genes_scenarios_exact(
                            features,
                            ohlcv,
                            screen_prep.as_ref().expect(
                                "candidate robustness work requires validation preparation",
                            ),
                            &prepared,
                            &work,
                            population_execution_run,
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
        let sensitivity_net_profit: Vec<Option<f64>> = if !candidate_deep_robustness {
            vec![None; candidates]
        } else {
            match fused_sensitivity {
                Some(values) => values,
                None => {
                    let mut settings = screen_templates.template(&pairs[0].1);
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
                        pairs.iter().map(|(_, gene)| gene.clone()).collect();
                    let base_work: Vec<neoethos_gpu_contracts::device::ScenarioDescriptor> = (0
                        ..candidates as u64)
                        .map(|candidate| {
                            crate::gpu_native::scenario::base_scenario(candidate, candidate, bars)
                        })
                        .collect();
                    let evaluated =
                        match crate::genetic::search_engine::prepare_validation_population(
                            ohlcv,
                            &base_genes,
                            &eval_config_for_signals,
                            &settings,
                        ) {
                            Ok(prepared) => {
                                #[cfg(feature = "gpu")]
                                let _gpu_guard =
                                    GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                                crate::genetic::search_engine::validation_genes_scenarios_exact(
                                    features,
                                    ohlcv,
                                    screen_prep.as_ref().expect(
                                        "candidate sensitivity work requires validation preparation",
                                    ),
                                    &prepared,
                                    &base_work,
                                    population_execution_run,
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
        if let Some((optimistic_pips, pessimistic_pips)) =
            cost_band_edges.filter(|_| candidates > 0)
        {
            let base_genes: Vec<Gene> = pairs.iter().map(|(_, gene)| gene.clone()).collect();
            let base_work: Vec<neoethos_gpu_contracts::device::ScenarioDescriptor> = (0
                ..candidates as u64)
                .map(|candidate| {
                    crate::gpu_native::scenario::base_scenario(candidate, candidate, bars)
                })
                .collect();
            let evaluate_at_total_cost = |total_pips: f64| -> Vec<Option<f64>> {
                let mut settings = screen_templates.template(&pairs[0].1);
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
                        let _gpu_guard = GPU_LAUNCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                        crate::genetic::search_engine::validation_genes_scenarios_exact(
                            features,
                            ohlcv,
                            screen_prep.as_ref().expect(
                                "a discriminating cost band requires validation preparation",
                            ),
                            &prepared,
                            &base_work,
                            population_execution_run,
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
        let mut trial_writer = if config.discovery_ledger_enabled && !trial_period_keys.is_empty() {
            match crate::trial_returns::TrialReturnsWriter::open(
                &config.discovery_ledger_cache_dir,
                &config.evaluation_symbol,
                &config.timeframe_label,
                trial_period_keys.clone(),
                initial_balance,
                candidates,
                search_input_receipt,
                search_state_config_hash,
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

        // Refresh admission between parallel waves; the full TF (not the GA's
        // smaller screening window) determines transient worker memory. Global
        // positions still index exactly the same precomputed scenario results.
        let completed_quality_replays = AtomicUsize::new(0);
        let mut screened: Vec<Option<QualityCandidate>> = Vec::with_capacity(candidates);
        let mut trial_rows_total = 0usize;
        let mut pairs_iter = pairs.into_iter();
        let mut chunk_base = 0usize;
        loop {
            crate::post_ga::check_cancel()?;
            let width = crate::post_ga::post_ga_batch_width(bars, candidates - chunk_base)?;
            let chunk: Vec<(usize, Gene)> = pairs_iter.by_ref().take(width).collect();
            if chunk.is_empty() {
                break;
            }
            let chunk_len = chunk.len();
            let base = chunk_base;
            let screened_rows: Result<
                Vec<(
                    Option<QualityCandidate>,
                    crate::trial_returns::TrialReturnRow,
                )>,
            > = chunk
                .into_par_iter()
                .enumerate()
                .map(|(local_position, (candidate_idx, gene))| -> Result<_> {
                    let position = base + local_position;
                    let sig = signals_for_gene_full_with_smc(
                        features,
                        &gene,
                        &eval_config_for_signals,
                        account_smc
                            .as_ref()
                            .expect("quality candidates have shared SMC"),
                    )?;
                    let confidences = account_sizing_confidences(
                        features,
                        &gene,
                        &eval_config_for_signals,
                        account_smc
                            .as_ref()
                            .expect("quality candidates have shared SMC"),
                        &sig,
                    )?;
                    let (account_metrics, trades) =
                        crate::eval::evaluate_strategy_with_confidence_and_ledger_core(
                            &ohlcv.close,
                            &ohlcv.high,
                            &ohlcv.low,
                            &sig,
                            &confidences,
                            &quality_months,
                            &quality_days,
                            &features.timestamps,
                            &screen_resolver.settings_for_gene(&gene),
                        )?;
                    completed_quality_replays.fetch_add(1, AtomicOrdering::Relaxed);
                    let mut metrics = analyzer.analyze_strategy_with_evaluation(
                        &gene.strategy_id,
                        &trades,
                        initial_balance,
                        quality_start_ms,
                        quality_end_ms,
                        &account_metrics,
                    )?;

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
                    )?;
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
                            return Ok((None, trial_row));
                        }
                    };

                    // Candidate-level regime, parameter-MC and spread-sensitivity
                    // gates are retained for Strict / PropFirm mode. Risky
                    // discovery makes the cost-aware profitability decision here
                    // and omits that redundant pre-portfolio battery. Its selected
                    // portfolio still receives the independent work implemented
                    // below: permutation, parameter plateau, walk-forward and
                    // CPCV/PBO. This is deliberately narrower than claiming that
                    // every omitted candidate-level test is repeated later.
                    if candidate_deep_robustness {
                        let regime_robust = validate_regime_robustness(
                            &trades,
                            features,
                            config.initial_balance,
                            config.max_regime_loss_pct,
                        );
                        if !regime_robust {
                            rejected_regime.fetch_add(1, AtomicOrdering::Relaxed);
                            return Ok((None, trial_row));
                        }
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
                    if candidate_deep_robustness {
                        let Some(profitable_runs) = mc_profitable_runs[position] else {
                            rejected_mc_error.fetch_add(1, AtomicOrdering::Relaxed);
                            return Ok((None, trial_row));
                        };

                        if (profitable_runs as u32) < config.mc_min_profitable {
                            rejected_mc_floor.fetch_add(1, AtomicOrdering::Relaxed);
                            // Within 10 points of the floor: the candidate is robust on
                            // most perturbations and lost on a minority, which is a very
                            // different signal from one that collapses outright.
                            if profitable_runs as u32 + 10 >= config.mc_min_profitable {
                                mc_near_miss.fetch_add(1, AtomicOrdering::Relaxed);
                            }
                            return Ok((None, trial_row));
                        }

                        // Spread/Slippage Sensitivity Test — wired from Settings
                        // 2026-05-26 (dual-mode product).
                        let Some(sens_pnl) = sensitivity_net_profit[position] else {
                            // Split from `rejected_mc_error` (2026-08-09): this is the
                            // SENSITIVITY launch failing, not the Monte-Carlo one.
                            rejected_sensitivity_error.fetch_add(1, AtomicOrdering::Relaxed);
                            return Ok((None, trial_row));
                        };
                        if sens_pnl < 0.0 {
                            rejected_sensitivity.fetch_add(1, AtomicOrdering::Relaxed);
                            return Ok((None, trial_row));
                        }
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

                    // Scalar metrics stay exact. Full trade/equity tapes are
                    // materialized again only for selected/report consumers.
                    metrics.equity_curve = Vec::new();
                    let ranking_score = full_window_candidate_ranking_score(
                        &account_metrics,
                        metrics.quality_score,
                        initial_balance,
                        &features.timestamps,
                        growth_goal,
                    );
                    Ok((
                        Some(QualityCandidate {
                            candidate_idx,
                            gene,
                            metrics,
                            ranking_score,
                            opportunistic: opportunistic_quality,
                            cost_band,
                        }),
                        trial_row,
                    ))
                })
                .collect::<Result<Vec<_>>>();
            let screened_rows = publish_completed_quality_chunk(
                screened_rows,
                &completed_quality_replays,
                &mut candidate_census,
                &mut progress_fn,
            )?;

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
            configured_monte_carlo_runs = config.mc_runs,
            candidate_deep_robustness,
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
            if entry.opportunistic {
                opportunistic_passed += 1;
            }
            // Keep the existing all-survivor report order; only the heavy curve
            // has been released, not its scalar quality/accounting evidence.
            quality_metrics.push(entry.metrics.clone());
            quality_candidate_indices.push(entry.candidate_idx);
            strict_passed.push(entry);
        }

        strict_passed.sort_by(|a, b| {
            let lane_a = if a.opportunistic { 0_u8 } else { 1_u8 };
            let lane_b = if b.opportunistic { 0_u8 } else { 1_u8 };
            lane_b
                .cmp(&lane_a)
                .then_with(|| {
                    b.ranking_score
                        .partial_cmp(&a.ranking_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.gene
                        .fitness
                        .partial_cmp(&a.gene.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.gene.strategy_id.cmp(&b.gene.strategy_id))
                .then_with(|| a.candidate_idx.cmp(&b.candidate_idx))
        });

        if config.filtering.log_trades {
            // Retain only candidate order/lane metadata here. Final membership
            // is not known until WF, calibration, correlation and robustness;
            // spending the journal cap now can omit every eventual winner.
            ranked_diagnostic_candidates = strict_passed
                .iter()
                .filter(|entry| entry.metrics.total_trades > 0)
                .map(|entry| (entry.candidate_idx, entry.opportunistic))
                .collect();
        }

        progress_fn(DiscoveryProgress::QualityScreened {
            strict_passed: strict_passed.len().saturating_sub(opportunistic_passed),
            opportunistic_passed,
            evaluated_candidates: filtered_count,
            // No journals have been materialized yet, only their ranking.
            logged_trade_sets: 0,
        });
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "selecting_portfolio",
            detail: "ranking survivors + prop-firm gate + correlation pruning — \
                     silent but active"
                .to_string(),
        });

        let mut screened_genes = Vec::with_capacity(strict_passed.len());
        // AUDIT #71 CLOSED HERE (2026-08-10). This loop used to bind the verdict
        // `_cost_band` and drop it, which is where the band stopped travelling:
        // it was measured at both edges and counted run-level, and then the only
        // artifact a live run reads could not say WHICH genes were
        // optimistic-edge-only. The verdict now rides out on
        // `DiscoveryResult::cost_band_by_strategy`, keyed by `strategy_id` —
        // the same key `logged_trades` uses, so no positional assumption is
        // made about a portfolio that is re-ranked and correlation-pruned
        // downstream.
        for entry in strict_passed {
            cost_band_by_strategy.push((entry.gene.strategy_id.clone(), entry.cost_band));
            screened_genes.push((entry.candidate_idx, entry.gene));
        }
        filtered = screened_genes;
    }
    // The quality screen collapses into a single funnel stage; the per-gate
    // breakdown below is what makes the persisted funnel answer "which test cost
    // us the candidates" without needing the run's logs.
    funnel.record_stage(
        "full_is_evaluated",
        post_min_trades,
        candidate_census.quality_evaluated,
    );
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
    let mut resolved_prop_firm_window_count = 0;
    if let Some(mut pf) = config.prop_firm_gate.clone() {
        // Auto-tune the window count if the operator left it at the
        // sentinel value (0). Scales with available history.
        if pf.n_windows == 0 {
            pf.n_windows = auto_tune_n_windows(&features.timestamps, pf.window_days);
        }
        resolved_prop_firm_window_count = pf.n_windows;
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
        let candidates_in = filtered;
        let timestamps_owned = features.timestamps.clone();
        let candidates_in_count = candidates_in.len();
        let pf_pass_rate_floor = pf.pass_rate;
        // ONE resolver + ONE window plan for the whole gate: the window
        // geometry and the window-local adaptive bases are gene-independent,
        // so they are computed once and shared across candidates.
        let pf_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            candidates_in.iter().map(|(_, gene)| gene),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        let pf_any_adaptive = candidates_in
            .iter()
            .any(|(_, g)| g.stop_vol_mult.is_finite() && g.stop_vol_mult > 0.0);
        let pf_windows =
            plan_prop_firm_windows(ohlcv, &timestamps_owned, &pf, &pf_resolver, pf_any_adaptive)?;
        let scored_all =
            crate::post_ga::map_bounded(candidates_in, features.n_samples(), |pair| {
                let sig = signals_for_gene_full_with_smc(
                    features,
                    &pair.1,
                    &eval_config_for_signals,
                    account_smc
                        .as_ref()
                        .expect("screened candidates have shared SMC"),
                )?;
                let confidences = account_sizing_confidences(
                    features,
                    &pair.1,
                    &eval_config_for_signals,
                    account_smc
                        .as_ref()
                        .expect("screened candidates have shared SMC"),
                    &sig,
                )?;
                let (rate, counted) = compute_prop_firm_pass_rate(
                    &pair.1,
                    &sig,
                    &confidences,
                    ohlcv,
                    &timestamps_owned,
                    config,
                    &pf,
                    &pf_resolver,
                    &pf_windows,
                )?;
                Ok((pair, rate, counted))
            })?;
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
        let mut scored: Vec<((usize, Gene), f64, usize)> = scored_all
            .into_iter()
            .filter(|(_, rate, counted)| *counted > 0 && *rate >= pf.pass_rate)
            .collect();
        // Sort by pass-rate descending; ties broken by gene fitness.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.0.1
                        .fitness
                        .partial_cmp(&a.0.1.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let mut next_filtered: Vec<(usize, Gene)> = Vec::with_capacity(scored.len());
        for (pair, rate, _) in scored {
            next_filtered.push(pair);
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
    } else {
        // No prop-firm gate (Risky mode / Strict mode): the stage is a
        // passthrough so the funnel doesn't show a phantom rejection.
        funnel.record_stage("passed_prop_firm_window", pre_prop_firm, pre_prop_firm);
    }

    let post_prop_firm = filtered.len();
    anyhow::ensure!(
        config.prop_firm_gate.is_none() || prop_firm_pass_rates.len() == filtered.len(),
        "post-quality genes/prop-firm pass rates are not aligned"
    );
    let wf_candidates = filtered
        .iter()
        .map(|(_, gene)| gene.clone())
        .collect::<Vec<_>>();
    progress_fn(DiscoveryProgress::StageAdvanced {
        stage: "candidate_walkforward",
        detail: format!(
            "walk-forward on all {} quality/window survivors before portfolio capacity {}",
            post_prop_firm, config.portfolio_size
        ),
    });
    let mut wf_cohort = crate::funnel_profile::WalkforwardSelectionCohort {
        scope: crate::data_selection::CanonicalSearchArtifactScopeRefV1::from_scope(
            selection_scope,
        )
        .map_err(anyhow::Error::new)?,
        search_config_hash: search_state_config_hash.to_owned(),
        trials: Vec::with_capacity(wf_candidates.len()),
    };
    let candidate_wf = discovery_walkforward_verdicts(
        &wf_candidates,
        features,
        ohlcv,
        config,
        effective_smc_gate_threshold,
        population_execution_run,
        |range, detailed, summaries| {
            for (gene, summary) in wf_candidates[range].iter().zip(detailed) {
                let archive_index = ranked_candidate_genes
                    .iter()
                    .position(|archived| archived.strategy_id == gene.strategy_id)
                    .ok_or_else(|| anyhow::anyhow!("WF candidate is absent from its archive"))?;
                let trial = walkforward_selection_trial(archive_index, gene, summary, config.mode)?;
                trial
                    .strategy_identity
                    .validate_against(&ranked_candidate_genes[archive_index])?;
                wf_cohort.trials.push(trial);
            }
            candidate_census.walkforward_tested += summaries.iter().filter(|s| s.tested).count();
            candidate_census.walkforward_passed += summaries.iter().filter(|s| s.passed).count();
            candidate_census.walkforward_failed = candidate_census
                .walkforward_tested
                .saturating_sub(candidate_census.walkforward_passed);
            candidate_census.walkforward_not_tested = candidate_census
                .validation_candidates_admitted
                .saturating_sub(candidate_census.walkforward_tested);
            progress_fn(DiscoveryProgress::CandidateCensusUpdated {
                census: candidate_census.clone(),
            });
            Ok(())
        },
    )?;
    funnel.record_stage(
        "passed_walkforward",
        candidate_census.walkforward_tested,
        candidate_census.walkforward_passed,
    );
    let mut wf_reasons = std::collections::BTreeMap::<&str, usize>::new();
    for trial in &wf_cohort.trials {
        for reason in &trial.rejection_reasons {
            *wf_reasons.entry(reason).or_default() += 1;
        }
    }
    for (reason, count) in wf_reasons {
        funnel.add_reject_reason("passed_walkforward", reason, count);
    }
    funnel.walkforward_selection_cohort = Some(wf_cohort);
    let profitable_calibration_genes = if let Some(calibration) = calibration_input {
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "holdout_forward_test",
            detail: format!(
                "selection calibration for all {} internal-WF survivors before active portfolio capacity {}; final holdout remains reserved",
                candidate_census.walkforward_passed, config.portfolio_size,
            ),
        });
        let calibration_candidates = wf_candidates
            .iter()
            .zip(&candidate_wf)
            .filter(|(_, verdict)| verdict.passed)
            .map(|(gene, _)| gene.clone())
            .collect::<Vec<_>>();
        let policy = funnel.live_trading_policy_v1().ok_or_else(|| {
            anyhow::anyhow!("selection calibration lost its sealed Search policy")
        })?;
        let cohort = evaluate_selection_calibration_cohort(
            &calibration_candidates,
            &ranked_candidate_genes,
            &effective_feature_names,
            calibration.features(),
            calibration.ohlcv(),
            calibration.scope(),
            search_state_config_hash,
            config,
            effective_smc_gate_threshold,
            Some(policy),
        )?;
        let profitable = cohort
            .trials
            .iter()
            .filter(|trial| trial.profitable_for_selection)
            .map(|trial| trial.strategy_identity.exact_gene_hash().to_owned())
            .collect::<HashSet<_>>();
        funnel.record_stage(
            "passed_selection_calibration",
            cohort.trials.len(),
            profitable.len(),
        );
        if profitable.len() < cohort.trials.len() {
            funnel.add_reject_reason(
                "passed_selection_calibration",
                "calibration_sizing_gate_failed",
                cohort.trials.len() - profitable.len(),
            );
        }
        tracing::info!(
            target: "neoethos_search::discovery",
            calibration_tested = cohort.trials.len(), calibration_profitable = profitable.len(),
            active_portfolio_capacity = config.portfolio_size,
            "completed the selection-used research cohort before active portfolio selection; this is not an untouched final test"
        );
        funnel.selection_calibration_cohort = Some(cohort);
        Some(profitable)
    } else {
        None
    };
    drop(wf_candidates);
    let selection_candidates = filtered
        .into_iter()
        .enumerate()
        .map(
            |(idx, (candidate_idx, gene))| WalkforwardSelectionCandidate {
                candidate_idx,
                gene,
                signals: Vec::new(),
                prop_firm_pass_rate: prop_firm_pass_rates.get(idx).copied(),
            },
        )
        .collect();
    let selected = select_walkforward_diverse_candidates_with_signals(
        selection_candidates,
        &candidate_wf,
        profitable_calibration_genes.as_ref(),
        config.portfolio_size,
        config.corr_threshold,
        &mut candidate_census,
        |gene| {
            crate::post_ga::check_cancel()?;
            crate::post_ga::post_ga_batch_width(features.n_samples(), 1)?;
            signals_for_gene_full_with_smc(
                features,
                gene,
                &eval_config_for_signals,
                account_smc
                    .as_ref()
                    .expect("selected candidates have shared SMC"),
            )
        },
    )?;
    let mut portfolio = Vec::with_capacity(selected.len());
    let mut portfolio_candidate_indices = Vec::with_capacity(selected.len());
    let mut portfolio_signals = Vec::with_capacity(selected.len());
    let mut portfolio_pass_rates = Vec::with_capacity(selected.len());
    for candidate in selected {
        portfolio_candidate_indices.push(candidate.candidate_idx);
        portfolio.push(candidate.gene);
        portfolio_signals.push(candidate.signals);
        if let Some(rate) = candidate.prop_firm_pass_rate {
            portfolio_pass_rates.push(rate);
        }
    }
    let rejected_by_correlation = candidate_census.rejected_by_correlation;
    progress_fn(DiscoveryProgress::PortfolioSelected {
        portfolio_size: portfolio.len(),
        rejected_by_correlation,
        target_portfolio: config.portfolio_size,
    });
    funnel.record_stage(
        "passed_correlation",
        candidate_census.correlation_tested,
        portfolio.len(),
    );
    if rejected_by_correlation > 0 {
        funnel.add_reject_reason(
            "passed_correlation",
            "undefined_or_pearson_or_spearman_above_threshold",
            rejected_by_correlation,
        );
    }
    funnel.record_stage(
        "portfolio_selected",
        profitable_calibration_genes
            .as_ref()
            .map_or(candidate_census.walkforward_passed, HashSet::len)
            .saturating_sub(rejected_by_correlation),
        portfolio.len(),
    );
    if candidate_census.portfolio_capacity_not_selected > 0 {
        funnel.add_reject_reason(
            "portfolio_selected",
            "portfolio_capacity_not_selected",
            candidate_census.portfolio_capacity_not_selected,
        );
    }
    progress_fn(DiscoveryProgress::CandidateCensusUpdated {
        census: candidate_census.clone(),
    });
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
    // Legacy holdout-free diagnostic rescue only. A three-way run keeps failed
    // calibration candidates in its research archive, never in active portfolio.
    // ── NEVER-ZERO rescue (2026-06-09) ─────────────────────────────────────
    // If the strict funnel (quality + prop-firm + correlation) rejected EVERY
    // candidate, promote the best-found base-filtered genes instead of dying
    // empty. They are correlation-pruned like a real portfolio and their metrics
    // recomputed so portfolio/quality_metrics/signals stay consistent — but the
    // heavy CPCV/walk-forward validation is SKIPPED (running it on genes that
    // already failed the bar would just burn the validation tail). They are
    // emitted honestly flagged `fallback_mode` and forced not-export-ready.
    let mut fallback_mode = false;
    let (
        mut validation_gates,
        mut canonical_backtest_artifacts,
        mut walkforward_validation_artifacts,
        _,
    ) = if portfolio.is_empty() && calibration_input.is_none() && !best_effort_fallback.is_empty() {
        fallback_mode = true;
        let fallback_reason = funnel
            .stages
            .iter()
            .filter(|s| s.count_in > 0)
            .max_by_key(|s| s.rejected)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "strict_gates".to_string());
        let analyzer = quality_analyzer_for_config(config);
        let quality_start_ms = features.timestamps.first().copied().ok_or_else(|| {
            anyhow::anyhow!("fallback replay requires an evaluation start timestamp")
        })?;
        let quality_end_ms = features.timestamps.last().copied().ok_or_else(|| {
            anyhow::anyhow!("fallback replay requires an evaluation end timestamp")
        })?;
        let (quality_months, quality_days) = month_day_indices(&features.timestamps);
        // Even the honesty-flagged fallback genes are DESCRIBED with the stop
        // regime they were scored under — their exported metrics must not come
        // from a strategy they never were.
        let fallback_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            best_effort_fallback.iter().map(|(_, gene)| gene),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        for (candidate_idx, gene) in best_effort_fallback {
            if portfolio.len() >= FALLBACK_PORTFOLIO_MAX {
                break;
            }
            crate::post_ga::check_cancel()?;
            crate::post_ga::post_ga_batch_width(features.n_samples(), 1)?;
            let sig = signals_for_gene_full_with_smc(
                features,
                &gene,
                &eval_config_for_signals,
                account_smc
                    .as_ref()
                    .expect("fallback candidates have shared SMC"),
            )?;
            if !portfolio_signal_is_correlation_rankable_v1(&sig) {
                continue;
            }
            let mut ok = true;
            for existing in &portfolio_signals {
                if !matches!(
                    pairwise_portfolio_correlation_decision_v1(
                        &sig,
                        existing,
                        config.corr_threshold,
                    ),
                    PortfolioCorrelationDecisionV1::Accept
                ) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let confidences = account_sizing_confidences(
                features,
                &gene,
                &eval_config_for_signals,
                account_smc
                    .as_ref()
                    .expect("fallback candidates have shared SMC"),
                &sig,
            )?;
            let (account_metrics, trades) =
                crate::eval::evaluate_strategy_with_confidence_and_ledger_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &sig,
                    &confidences,
                    &quality_months,
                    &quality_days,
                    &features.timestamps,
                    &fallback_resolver.settings_for_gene(&gene),
                )?;
            quality_metrics.push(analyzer.analyze_strategy_with_evaluation(
                &gene.strategy_id,
                &trades,
                config.initial_balance,
                quality_start_ms,
                quality_end_ms,
                &account_metrics,
            )?);
            quality_candidate_indices.push(candidate_idx);
            portfolio_candidate_indices.push(candidate_idx);
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
        (
            DiscoveryValidationGates::pending(),
            Vec::new(),
            Vec::new(),
            Vec::<bool>::new(),
        )
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
            .map(|(gi, gene)| -> Result<(bool, String)> {
                let settings = robust_resolver.settings_for_gene(gene);
                let sig_full = &portfolio_signals[gi];
                if sig_full.len() != n_all {
                    return Ok((true, "skipped (signal length mismatch)".to_string()));
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
                    return Ok((true, "skipped (thin recent window)".to_string()));
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
                let p_value = permutation_monte_carlo_p_value_v1(beats, N_PERM)?;
                if p_value >= PERM_P_MAX {
                    return Ok((
                        false,
                        format!(
                            "permutation FAIL (p={p_value:.2}: random timing matches the real net)"
                        ),
                    ));
                }

                // #11 plateau: ±15% threshold perturbations must keep ≥30% net.
                for factor in [0.85_f64, 1.15] {
                    let mut variant = gene.clone();
                    variant.long_threshold *= factor;
                    variant.short_threshold *= factor;
                    let sig_v = signals_for_gene_full(features, ohlcv, &variant, &eval_cfg_rb)?;
                    if sig_v.len() != n_all {
                        continue;
                    }
                    let net_v = net_of(&sig_v[w0..]);
                    if net_v < PLATEAU_MIN_RATIO * real_net {
                        return Ok((
                            false,
                            format!(
                                "plateau FAIL (thresholds ×{factor:.2} → net {net_v:.0} \
                                 vs real {real_net:.0} — cliff, not plateau)"
                            ),
                        ));
                    }
                }
                Ok((true, format!("robust (p={p_value:.2}, plateau ok)")))
            })
            .collect::<Result<Vec<_>>>()?;

        anyhow::ensure!(
            verdicts.len() == portfolio.len()
                && portfolio_candidate_indices.len() == portfolio.len()
                && portfolio_signals.len() == portfolio.len()
                && (portfolio_pass_rates.is_empty()
                    || portfolio_pass_rates.len() == portfolio.len()),
            "robustness portfolio candidate metadata is not aligned"
        );
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
            portfolio_candidate_indices.retain(|_| {
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
            if !portfolio_pass_rates.is_empty() {
                let mut i = 0usize;
                portfolio_pass_rates.retain(|_| {
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

    // Publish the completed membership transition before final artifact work,
    // so a later CPCV/serialization error cannot leave a stale selected count.
    publish_portfolio_after_robustness(
        portfolio.len(),
        fallback_mode,
        &mut candidate_census,
        funnel,
        &mut progress_fn,
    );

    // Membership is now fixed by quality, mode-aware WF, correlation and the
    // robustness screen. Aggregate CPCV/PBO and persisted per-gene artifacts
    // must describe exactly these genes, not the pre-filter portfolio.
    if !fallback_mode {
        progress_fn(DiscoveryProgress::StageAdvanced {
            stage: "validation_gates",
            detail: format!(
                "final canonical/WF/CPCV/PBO evidence for {} selected strategies",
                portfolio.len()
            ),
        });
        (
            validation_gates,
            canonical_backtest_artifacts,
            walkforward_validation_artifacts,
            _,
        ) = build_discovery_validation_artifacts(
            &portfolio,
            &portfolio_signals,
            features,
            ohlcv,
            selection_scope,
            search_state_config_hash,
            config,
            effective_smc_gate_threshold,
            &pbo_candidates,
            ga_returned_candidates,
            population_execution_run,
        )?;
    }

    if config.prop_firm_gate.is_some() {
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
        validation_gates.prop_firm_window_count = resolved_prop_firm_window_count;
        validation_gates.prop_firm_window_pass_rate = if portfolio_pass_rates.is_empty() {
            0.0
        } else {
            portfolio_pass_rates.iter().sum::<f64>() / portfolio_pass_rates.len() as f64
        };
    }
    // Candidate WF coverage was recorded before portfolio selection. Do not
    // overwrite it with the final portfolio size, or count skipped fallback
    // validation as failed WF. Final aggregate gates remain authoritative.
    if fallback_mode {
        // Honest: best-effort fallback genes did NOT pass the prop bar, so they
        // must never read as export-ready downstream (the autonomous trader keys
        // off `is_portfolio_export_ready()` / `prop_firm_window_passed`).
        validation_gates.prop_firm_window_passed = false;
    }
    let journal_plan = plan_diagnostic_candidates(
        &portfolio_candidate_indices,
        &portfolio,
        &ranked_candidates,
        &ranked_diagnostic_candidates,
        config.filtering.log_trades,
        config.filtering.trade_log_max,
    )?;
    if !journal_plan.is_empty() {
        anyhow::ensure!(
            portfolio_candidate_indices.len() == portfolio_signals.len(),
            "diagnostic portfolio signals are not aligned"
        );
        let selected_signals = portfolio_candidate_indices
            .iter()
            .copied()
            .zip(portfolio_signals.iter().map(Vec::as_slice))
            .collect();
        let indexed_logs = replay_diagnostic_candidates(
            journal_plan,
            features,
            ohlcv,
            config,
            &eval_config_for_signals,
            account_smc
                .as_ref()
                .expect("diagnostic candidates have shared SMC"),
            &selected_signals,
        )?;
        (logged_candidate_indices, logged_trades) = indexed_logs.into_iter().unzip();
        tracing::info!(
            selected_strategies = portfolio.len(),
            configured_diagnostic_set_cap = config.filtering.trade_log_max,
            logged_trade_sets = logged_trades.len(),
            "complete in-sample diagnostic journals: final selections first, then quality-ranked extras; selected coverage takes precedence over the set cap"
        );
    }
    let portfolio_size = portfolio.len();
    let walkforward_pass = if validation_gates.walkforward_passed {
        portfolio_size
    } else {
        0
    };
    let cpcv_pass = if validation_gates.cpcv_passed {
        walkforward_pass
    } else {
        0
    };
    funnel.record_stage("passed_cpcv", walkforward_pass, cpcv_pass);
    // Every mode requires the final WF, CPCV and measured PBO gates.
    let export_ready = if validation_gates.is_portfolio_export_ready() {
        portfolio_size
    } else {
        0
    };
    funnel.record_stage("export_ready", portfolio_size, export_ready);
    funnel.candidate_census = Some(candidate_census.clone());
    progress_fn(DiscoveryProgress::CandidateCensusUpdated {
        census: candidate_census,
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

    // Restore full per-trade curves only after membership is fixed. The broad
    // quality report keeps every scalar row, but never an archive-sized matrix
    // of equity tapes. Reuse exact logged ledgers when available; otherwise
    // replay the same frozen gene/signals/settings in RAM-admitted waves.
    anyhow::ensure!(
        portfolio_candidate_indices.len() == portfolio.len()
            && portfolio_signals.len() == portfolio.len()
            && quality_candidate_indices.len() == quality_metrics.len()
            && logged_candidate_indices.len() == logged_trades.len(),
        "final equity curve candidate metadata is not aligned"
    );
    let curve_candidates: Vec<_> = portfolio
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            let candidate_idx = portfolio_candidate_indices[*idx];
            quality_candidate_indices
                .iter()
                .zip(&quality_metrics)
                .any(|(quality_idx, metrics)| {
                    *quality_idx == candidate_idx
                        && metrics.total_trades > 0
                        && metrics.equity_curve.is_empty()
                })
        })
        .collect();
    if !curve_candidates.is_empty() {
        let curve_resolver = GeneEvalSettingsResolver::for_slice(
            config,
            curve_candidates.iter().map(|(_, gene)| *gene),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        let curves =
            crate::post_ga::map_bounded(curve_candidates, features.n_samples(), |(idx, gene)| {
                let candidate_idx = portfolio_candidate_indices[idx];
                let curve = candidate_equity_curve(
                    candidate_idx,
                    config.initial_balance,
                    &logged_candidate_indices,
                    &logged_trades,
                    || {
                        let sig = &portfolio_signals[idx];
                        let confidences = account_sizing_confidences(
                            features,
                            gene,
                            &eval_config_for_signals,
                            account_smc
                                .as_ref()
                                .expect("final portfolio has shared SMC"),
                            sig,
                        )?;
                        simulate_trades_with_confidence_core(
                            &ohlcv.close,
                            &ohlcv.high,
                            &ohlcv.low,
                            &features.timestamps,
                            sig,
                            &confidences,
                            &curve_resolver.settings_for_gene(gene),
                        )
                    },
                )?;
                Ok((candidate_idx, curve))
            })?;
        for (candidate_idx, curve) in curves {
            restore_candidate_quality_curve(
                candidate_idx,
                &curve,
                &quality_candidate_indices,
                &mut quality_metrics,
            )?;
        }
    }

    // Conditional bootstrap scenario (Risky only), not a replay of the netted
    // account or future-success evidence. Config and realized trades coexist here.
    log_goal_report(config, &portfolio, &quality_metrics, &logged_trades);

    retain_selection_validation_artifacts_for_final_portfolio(
        &portfolio,
        &mut canonical_backtest_artifacts,
        &mut walkforward_validation_artifacts,
    )?;
    validation_gates.canonical_backtest_artifacts = canonical_backtest_artifacts.len();
    validation_gates.walkforward_validation_artifacts = walkforward_validation_artifacts.len();

    let forward_test_validation_artifacts = match (
        calibration_scope,
        funnel.selection_calibration_cohort.as_ref(),
    ) {
        (Some(scope), Some(cohort)) => {
            selected_calibration_artifacts(&portfolio, scope, search_state_config_hash, cohort)?
        }
        _ => Vec::new(),
    };
    let result = DiscoveryResult {
        search_input_receipt: search_input_receipt.clone(),
        selection_scope: selection_scope.clone(),
        calibration_scope: calibration_scope.cloned(),
        holdout_scope: holdout_scope.cloned(),
        search_config_hash: search_state_config_hash.to_string(),
        cost_band_census,
        cost_band_by_strategy,
        portfolio,
        candidates: ranked_candidate_genes,
        quality_metrics,
        logged_trades,
        effective_feature_names,
        validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
        forward_test_validation_artifacts,
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: Some(funnel.clone()),

        effective_smc_gate_threshold,
    };
    result.validate_evaluated_scopes()?;
    progress_fn(DiscoveryProgress::Completed {
        candidate_count: result.candidates.len(),
        filtered_count,
        portfolio_size: result.portfolio.len(),
    });
    Ok(result)
}

fn ranking_window_span_days(timestamps: &[i64]) -> f64 {
    match (timestamps.first(), timestamps.last()) {
        (Some(first), Some(last)) if last > first => (*last as f64 - *first as f64) / 86_400_000.0,
        _ => 0.0,
    }
}

fn full_window_candidate_ranking_score(
    canonical_metrics: &[f64; 11],
    quality_score: f64,
    initial_equity: f64,
    timestamps: &[i64],
    goal: Option<crate::scoring::RiskyGrowthGoal>,
) -> f64 {
    goal.map_or(quality_score, |goal| {
        crate::scoring::ga_fitness_goal(
            canonical_metrics,
            initial_equity,
            ranking_window_span_days(timestamps),
            goal,
        )
    })
}

fn rank_candidates_on_matching_window(
    candidates: Vec<Gene>,
    metrics: Vec<[f64; 11]>,
    initial_equity: f64,
    timestamps: &[i64],
    goal: Option<crate::scoring::RiskyGrowthGoal>,
) -> Result<Vec<(usize, Gene)>> {
    anyhow::ensure!(
        candidates.len() == metrics.len(),
        "GA candidate/metric alignment mismatch"
    );
    let span_days = ranking_window_span_days(timestamps);
    let mut ranked = candidates
        .into_iter()
        .zip(metrics)
        .enumerate()
        .map(|(index, (gene, metrics))| {
            let score = if let Some(goal) = goal {
                crate::scoring::ga_fitness_goal(&metrics, initial_equity, span_days, goal)
            } else {
                let pf_capped = gene.profit_factor.min(3.0) / 3.0;
                let safety = (1.0 - gene.max_drawdown / 0.07).clamp(0.0, 1.0);
                let multiplier =
                    gene.consistency * 0.4 + gene.win_rate * 0.3 + safety * 0.2 + pf_capped * 0.1;
                let bonus = if gene.consistency > 0.8 { 2.0 } else { 1.0 };
                gene.fitness * multiplier * bonus
            };
            (index, gene, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|(idx_a, a, score_a), (idx_b, b, score_b)| {
        score_b
            .partial_cmp(score_a)
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
    Ok(ranked
        .into_iter()
        .map(|(index, gene, _)| (index, gene))
        .collect())
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

/// Canonical decision authority `neoethos.portfolio-correlation-authority.v1`.
///
/// Pearson formula and undefined/near-constant boundaries:
/// https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
/// Spearman constant-input boundary:
/// https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
/// Tie handling (average ranks, then Pearson correlation of the ranks):
/// https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/rankcorr.htm
///
/// SciPy warns rather than rejects a near-constant input. V1 deliberately
/// fails closed at that published numerical-instability boundary because this
/// correlation is a decision-critical portfolio gate, not a descriptive
/// statistic.
const SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1: f64 = 1.0e-13;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CorrelationUndefinedV1 {
    LengthMismatch,
    InsufficientPairedObservations,
    ConstantInput,
    NearConstantInput,
    NonFiniteResult,
    InvalidThreshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortfolioCorrelationDecisionV1 {
    Accept,
    RejectUndefined(CorrelationUndefinedV1),
    RejectThreshold,
}

fn validate_paired_correlation_shape_v1(
    a: &[i8],
    b: &[i8],
) -> Result<usize, CorrelationUndefinedV1> {
    if a.len() != b.len() {
        return Err(CorrelationUndefinedV1::LengthMismatch);
    }
    if a.len() < 2 {
        return Err(CorrelationUndefinedV1::InsufficientPairedObservations);
    }
    Ok(a.len())
}

fn classify_centered_correlation_input_v1(
    mean: f64,
    centered_sum_squares: f64,
) -> Result<(), CorrelationUndefinedV1> {
    if !mean.is_finite() || !centered_sum_squares.is_finite() || centered_sum_squares < 0.0 {
        return Err(CorrelationUndefinedV1::NonFiniteResult);
    }
    if centered_sum_squares == 0.0 {
        return Err(CorrelationUndefinedV1::ConstantInput);
    }
    let centered_norm = centered_sum_squares.sqrt();
    if !centered_norm.is_finite() {
        return Err(CorrelationUndefinedV1::NonFiniteResult);
    }
    if centered_norm < SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1 * mean.abs() {
        return Err(CorrelationUndefinedV1::NearConstantInput);
    }
    Ok(())
}

fn finish_correlation_v1(
    numerator: f64,
    centered_sum_squares_a: f64,
    centered_sum_squares_b: f64,
) -> Result<f64, CorrelationUndefinedV1> {
    if !numerator.is_finite() {
        return Err(CorrelationUndefinedV1::NonFiniteResult);
    }
    let denominator = centered_sum_squares_a.sqrt() * centered_sum_squares_b.sqrt();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(CorrelationUndefinedV1::NonFiniteResult);
    }
    let correlation = numerator / denominator;
    if !correlation.is_finite() {
        return Err(CorrelationUndefinedV1::NonFiniteResult);
    }
    Ok(correlation)
}

fn portfolio_signal_is_correlation_rankable_v1(signal: &[i8]) -> bool {
    pearson_corr_i8(signal, signal).is_ok() && spearman_corr_i8(signal, signal).is_ok()
}

fn pairwise_portfolio_correlation_decision_v1(
    a: &[i8],
    b: &[i8],
    threshold: f64,
) -> PortfolioCorrelationDecisionV1 {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return PortfolioCorrelationDecisionV1::RejectUndefined(
            CorrelationUndefinedV1::InvalidThreshold,
        );
    }
    let pearson = match pearson_corr_i8(a, b) {
        Ok(value) => value,
        Err(reason) => return PortfolioCorrelationDecisionV1::RejectUndefined(reason),
    };
    let spearman = match spearman_corr_i8(a, b) {
        Ok(value) => value,
        Err(reason) => return PortfolioCorrelationDecisionV1::RejectUndefined(reason),
    };
    if pearson.abs() >= threshold || spearman.abs() >= threshold {
        PortfolioCorrelationDecisionV1::RejectThreshold
    } else {
        PortfolioCorrelationDecisionV1::Accept
    }
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

fn spearman_corr_i8(a: &[i8], b: &[i8]) -> Result<f64, CorrelationUndefinedV1> {
    let n = validate_paired_correlation_shape_v1(a, b)?;
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
    classify_centered_correlation_input_v1(mean_a, denom_a)?;
    classify_centered_correlation_input_v1(mean_b, denom_b)?;
    finish_correlation_v1(num, denom_a, denom_b)
}

fn pearson_corr_i8(a: &[i8], b: &[i8]) -> Result<f64, CorrelationUndefinedV1> {
    let n = validate_paired_correlation_shape_v1(a, b)?;
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
    classify_centered_correlation_input_v1(mean_a, denom_a)?;
    classify_centered_correlation_input_v1(mean_b, denom_b)?;
    finish_correlation_v1(num, denom_a, denom_b)
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
    result.validate_evaluated_scopes()?;
    ensure_portfolio_export_ready(result)?;
    let exports = build_portfolio_exports(&result.portfolio, &result.effective_feature_names);
    let envelope = CanonicalSearchArtifactEnvelopeV2::new(
        "neoethos.search-portfolio.v1",
        result.selection_scope()?.clone(),
        result.search_config_hash.clone(),
        exports,
    )
    .map_err(anyhow::Error::new)?;
    write_json_atomic(path, &envelope)
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
            let curve_min = q
                .equity_curve
                .iter()
                .copied()
                .reduce(f64::min)
                .filter(|value| value.is_finite())
                .map(|value| format!("{value:.0}"))
                .unwrap_or_else(|| "not materialized".to_string());
            tracing::info!(
                target: "neoethos_search::discovery",
                "      money: EUR {:.0} -> {:.0} (net {:+.0}, {:.1} months, {:.2}%/mo) | recovery {:.2} | curve min EUR {} | maxDD EUR {:.0}",
                q.initial_capital,
                q.final_balance,
                q.net_profit,
                q.period_days / 30.44,
                if q.period_days > 0.0 { q.total_return_pct * 100.0 / (q.period_days / 30.44) } else { 0.0 },
                q.recovery_factor,
                curve_min,
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

/// Conditional IID bootstrap of the selected strategies' realized R-multiples.
/// Cadence uses their full evaluated calendar exposure, matching the goal's
/// calendar deadline. Summing independently replayed trade rates does not
/// reproduce portfolio netting, confidence sizing or reference-capital limits;
/// the rendered report states those limitations. No-op outside Risky mode.
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
    // A trading-weekday rate multiplied by a calendar horizon invents extra
    // trades over weekends/inactive dates. Use each complete replay interval,
    // and do not turn missing/invalid exposure into a zero or one-day estimate.
    let trades_per_day = quality_metrics
        .iter()
        .filter(|q| ids.contains(q.strategy_id.as_str()))
        .try_fold(0.0_f64, |total, q| {
            let rate = crate::goal_report::calendar_trades_per_day(q.total_trades, q.period_days)?;
            let combined = total + rate;
            combined.is_finite().then_some(combined)
        });
    let Some(trades_per_day) = trades_per_day else {
        tracing::warn!(
            target: "neoethos_search::discovery",
            "GOAL REPORT — skipped: selected strategies have invalid or unavailable calendar exposure."
        );
        return;
    };
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
    result.validate_validation_evidence_sets(false)?;
    if result.canonical_backtest_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create canonical backtest dir {}", dir.display()))?;
    for (idx, artifact) in result.canonical_backtest_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(
            artifact.strategy_identity().exact_gene_hash(),
            idx,
        );
        write_canonical_backtest_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.canonical_backtest_artifacts.len())
}

pub fn save_walkforward_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    result.validate_validation_evidence_sets(false)?;
    if result.walkforward_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create walk-forward validation dir {}", dir.display()))?;
    for (idx, artifact) in result.walkforward_validation_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(
            artifact.strategy_identity().exact_gene_hash(),
            idx,
        );
        write_walkforward_validation_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.walkforward_validation_artifacts.len())
}

pub fn save_forward_test_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    result.validate_validation_evidence_sets(false)?;
    if result.forward_test_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create forward-test validation dir {}", dir.display()))?;
    for (idx, artifact) in result.forward_test_validation_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(
            artifact.strategy_identity().exact_gene_hash(),
            idx,
        );
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
pub const PROMOTION_SUMMARY_ARTIFACT_KIND_V3: &str = "neoethos.search-promotion-summary.v3";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionStrategyEvidenceV2 {
    strategy_identity: ValidationStrategyIdentityV2,
    canonical_backtest_hash: String,
    walkforward_hash: String,
    forward_test_hash: String,
    prop_firm_hash: String,
}

impl PromotionStrategyEvidenceV2 {
    pub fn strategy_identity(&self) -> &ValidationStrategyIdentityV2 {
        &self.strategy_identity
    }

    pub fn canonical_backtest_hash(&self) -> &str {
        &self.canonical_backtest_hash
    }

    pub fn walkforward_hash(&self) -> &str {
        &self.walkforward_hash
    }

    pub fn forward_test_hash(&self) -> &str {
        &self.forward_test_hash
    }

    pub fn prop_firm_hash(&self) -> &str {
        &self.prop_firm_hash
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionOutOfSampleVerdictV2 {
    forward_test_passed: bool,
    prop_firm_passed: bool,
    walkforward_passed: bool,
    cpcv_passed: bool,
}

impl PromotionOutOfSampleVerdictV2 {
    pub fn forward_test_passed(&self) -> bool {
        self.forward_test_passed
    }

    pub fn prop_firm_passed(&self) -> bool {
        self.prop_firm_passed
    }

    pub fn walkforward_passed(&self) -> bool {
        self.walkforward_passed
    }

    pub fn cpcv_passed(&self) -> bool {
        self.cpcv_passed
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionSummaryAuthorityPayloadV3 {
    schema_version: u32,
    holdout_scope: CanonicalSearchArtifactScopeV2,
    validation_evidence_hashes: DiscoveryPerKindEvidenceHashes,
    strategy_evidence: Vec<PromotionStrategyEvidenceV2>,
    determinism_policy: DeterminismPolicy,
    out_of_sample: PromotionOutOfSampleVerdictV2,
}

impl PromotionSummaryAuthorityPayloadV3 {
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn holdout_scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.holdout_scope
    }

    pub fn validation_evidence_hashes(&self) -> &DiscoveryPerKindEvidenceHashes {
        &self.validation_evidence_hashes
    }

    pub fn strategy_evidence(&self) -> &[PromotionStrategyEvidenceV2] {
        &self.strategy_evidence
    }

    pub fn determinism_policy(&self) -> &DeterminismPolicy {
        &self.determinism_policy
    }

    pub fn out_of_sample(&self) -> &PromotionOutOfSampleVerdictV2 {
        &self.out_of_sample
    }

    pub fn validate_shape(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == 3,
            "unsupported promotion-summary payload schema version {}; expected 3",
            self.schema_version
        );
        self.holdout_scope.validate().map_err(anyhow::Error::new)?;
        anyhow::ensure!(
            self.holdout_scope.evaluated_window().role() == CanonicalSearchWindowRoleV1::Holdout,
            "promotion summary requires an exact Holdout scope"
        );
        anyhow::ensure!(
            self.validation_evidence_hashes.all_producer_kinds_present(),
            "promotion summary is missing a producer-side validation hash"
        );
        anyhow::ensure!(
            self.validation_evidence_hashes
                .live_execution_simulation
                .is_none(),
            "promotion summary v3 does not accept an unbound live-simulation hash"
        );
        for (label, hash) in [
            (
                "canonical backtest aggregate hash",
                self.validation_evidence_hashes
                    .canonical_backtest
                    .as_deref(),
            ),
            (
                "walkforward aggregate hash",
                self.validation_evidence_hashes.walkforward.as_deref(),
            ),
            (
                "forward-test aggregate hash",
                self.validation_evidence_hashes.forward_test.as_deref(),
            ),
            (
                "prop-firm aggregate hash",
                self.validation_evidence_hashes.prop_firm.as_deref(),
            ),
        ] {
            crate::validation::validate_fnv64_hash(
                label,
                hash.expect("all producer hashes checked present"),
            )?;
        }
        anyhow::ensure!(
            !self.strategy_evidence.is_empty(),
            "promotion summary contains no final strategies"
        );
        let mut previous: Option<(&str, &str)> = None;
        let mut strategy_ids = HashSet::with_capacity(self.strategy_evidence.len());
        let mut exact_gene_hashes = HashSet::with_capacity(self.strategy_evidence.len());
        for evidence in &self.strategy_evidence {
            evidence.strategy_identity.validate()?;
            if let Some((previous_hash, previous_id)) = previous {
                anyhow::ensure!(
                    (previous_hash, previous_id)
                        < (
                            evidence.strategy_identity.exact_gene_hash(),
                            evidence.strategy_identity.strategy_id(),
                        ),
                    "promotion summary strategies are not strictly sorted by exact identity"
                );
            }
            previous = Some((
                evidence.strategy_identity.exact_gene_hash(),
                evidence.strategy_identity.strategy_id(),
            ));
            anyhow::ensure!(
                strategy_ids.insert(evidence.strategy_identity.strategy_id()),
                "promotion summary contains duplicate strategy_id `{}`",
                evidence.strategy_identity.strategy_id()
            );
            anyhow::ensure!(
                exact_gene_hashes.insert(evidence.strategy_identity.exact_gene_hash()),
                "promotion summary contains duplicate exact gene hash `{}`",
                evidence.strategy_identity.exact_gene_hash()
            );
            for (label, hash) in [
                (
                    "canonical backtest strategy hash",
                    &evidence.canonical_backtest_hash,
                ),
                ("walkforward strategy hash", &evidence.walkforward_hash),
                ("forward-test strategy hash", &evidence.forward_test_hash),
                ("prop-firm strategy hash", &evidence.prop_firm_hash),
            ] {
                crate::validation::validate_fnv64_hash(label, hash)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn build_promotion_summary_envelope(
    result: &DiscoveryResult,
) -> Result<CanonicalSearchArtifactEnvelopeV2<PromotionSummaryAuthorityPayloadV3>> {
    result.validate_complete_promotion_evidence()?;
    let hashes = discovery_per_kind_evidence_hashes(result)?;
    let evidence = live_validation_evidence_from_discovery(result)?;
    let holdout_scope = result
        .holdout_scope()?
        .ok_or_else(|| {
            anyhow::anyhow!("promotion summary requires the exact stored holdout scope")
        })?
        .clone();
    let mut strategy_evidence = result
        .portfolio
        .iter()
        .map(|gene| -> Result<PromotionStrategyEvidenceV2> {
            let identity = ValidationStrategyIdentityV2::from_gene(gene)?;
            let exact_hash = identity.exact_gene_hash();
            let canonical = result
                .canonical_backtest_artifacts
                .iter()
                .find(|artifact| artifact.strategy_identity().exact_gene_hash() == exact_hash)
                .expect("complete validation checked canonical strategy coverage");
            let walkforward = result
                .walkforward_validation_artifacts
                .iter()
                .find(|artifact| artifact.strategy_identity().exact_gene_hash() == exact_hash)
                .expect("complete validation checked walkforward strategy coverage");
            let forward_test = result
                .forward_test_validation_artifacts
                .iter()
                .find(|artifact| artifact.strategy_identity().exact_gene_hash() == exact_hash)
                .expect("complete validation checked forward-test strategy coverage");
            let prop_firm = result
                .prop_firm_validation_artifacts
                .iter()
                .find(|artifact| artifact.strategy_identity().exact_gene_hash() == exact_hash)
                .expect("complete validation checked prop-firm strategy coverage");
            Ok(PromotionStrategyEvidenceV2 {
                strategy_identity: identity,
                canonical_backtest_hash: stable_json_hash(canonical)?,
                walkforward_hash: stable_json_hash(walkforward)?,
                forward_test_hash: stable_json_hash(forward_test)?,
                prop_firm_hash: stable_json_hash(prop_firm)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    strategy_evidence.sort_by(|left, right| {
        left.strategy_identity
            .exact_gene_hash()
            .cmp(right.strategy_identity.exact_gene_hash())
            .then_with(|| {
                left.strategy_identity
                    .strategy_id()
                    .cmp(right.strategy_identity.strategy_id())
            })
    });
    let summary = PromotionSummaryAuthorityPayloadV3 {
        schema_version: 3,
        holdout_scope,
        validation_evidence_hashes: hashes,
        strategy_evidence,
        determinism_policy: crate::genetic::current_determinism_policy(),
        out_of_sample: PromotionOutOfSampleVerdictV2 {
            forward_test_passed: evidence
                .forward_test_passed
                .expect("complete validation requires forward-test evidence"),
            prop_firm_passed: evidence
                .prop_firm_passed
                .expect("complete validation requires prop-firm evidence"),
            walkforward_passed: evidence.walkforward_passed,
            cpcv_passed: evidence.cpcv_passed,
        },
    };
    let envelope = CanonicalSearchArtifactEnvelopeV2::new(
        PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
        result.selection_scope()?.clone(),
        result.search_config_hash.clone(),
        summary,
    )
    .map_err(anyhow::Error::new)?;
    envelope.payload().validate_shape()?;
    Ok(envelope)
}

pub fn save_promotion_summary_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    let envelope = build_promotion_summary_envelope(result)?;
    write_json_atomic(path, &envelope)
}

pub fn save_prop_firm_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    result.validate_validation_evidence_sets(false)?;
    if result.prop_firm_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create prop-firm validation dir {}", dir.display()))?;
    for (idx, artifact) in result.prop_firm_validation_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(
            artifact.strategy_identity().exact_gene_hash(),
            idx,
        );
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
///   [`PropFirmRiskValidationArtifactFile`] values' `summary().all_rules_passed`
///   flags: `Some(true)` when every persisted prop-firm artifact passes,
///   `Some(false)` when at least one fails, and `None` when no
///   prop-firm artifact was produced (the live bridge will treat that
///   as missing evidence whenever the gate is required).
/// - `live_sim_runtime_model_hash` stays `None` until a live-execution
///   simulator is wired into the discovery pipeline.
pub fn live_validation_evidence_from_discovery(
    result: &DiscoveryResult,
) -> Result<LiveValidationEvidence> {
    result.validate_complete_promotion_evidence()?;
    let forward_test_passed = if result.forward_test_validation_artifacts.is_empty() {
        None
    } else {
        let all_pass = result
            .forward_test_validation_artifacts
            .iter()
            .all(|artifact| {
                artifact.summary().metrics.trade_count > 0
                    && artifact.summary().metrics.net_profit > 0.0
            });
        Some(all_pass)
    };
    let prop_firm_passed = if result.prop_firm_validation_artifacts.is_empty() {
        None
    } else {
        let all_pass = result
            .prop_firm_validation_artifacts
            .iter()
            .all(|artifact| artifact.summary().all_rules_passed);
        Some(all_pass)
    };
    Ok(LiveValidationEvidence {
        walkforward_passed: result.validation_gates.walkforward_passed,
        cpcv_passed: result.validation_gates.cpcv_passed,
        forward_test_passed,
        prop_firm_passed,
        live_sim_runtime_model_hash: None,
    })
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
    result.validate_validation_evidence_sets(false)?;
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
    result.validate_validation_evidence_sets(false)?;
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
    result.validate_validation_evidence_sets(false)?;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

fn hash_validation_artifacts<T: ExactDiscoveryValidationArtifact>(
    artifacts: &[T],
) -> Result<String> {
    if artifacts.is_empty() {
        Ok(String::new())
    } else {
        let mut ordered = artifacts.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| {
            left.strategy_identity()
                .exact_gene_hash()
                .cmp(right.strategy_identity().exact_gene_hash())
                .then_with(|| {
                    left.strategy_identity()
                        .strategy_id()
                        .cmp(right.strategy_identity().strategy_id())
                })
        });
        stable_json_hash(&ordered)
    }
}

fn optional_hash_validation_artifacts<T: ExactDiscoveryValidationArtifact>(
    artifacts: &[T],
) -> Result<Option<String>> {
    if artifacts.is_empty() {
        Ok(None)
    } else {
        hash_validation_artifacts(artifacts).map(Some)
    }
}

pub fn build_discovery_profile(
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> DiscoveryRunProfile {
    let population_execution_run_receipt_v2 = result
        .funnel_profile
        .as_ref()
        .and_then(|funnel| funnel.population_execution_run_receipt_v2());
    let population_eval_engines = population_execution_run_receipt_v2
        .map(|receipt| receipt.engine_receipt_v1().engines().to_vec())
        .unwrap_or_default();
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
        pnl_conversion_fee_rate,
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
        high_quality_confidence,
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
        population_eval_engines,
        population_execution_run_receipt_v2: population_execution_run_receipt_v2.cloned(),
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
        pnl_conversion_fee_rate: *pnl_conversion_fee_rate,
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
        high_quality_confidence: *high_quality_confidence,
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
    result.validate_validation_evidence_sets(false)?;
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
            0x3FE0_CF96_9566_5F44,
            "long_threshold moved: {} (the f64 reference is 0.525340358540213)",
            perturbed.long_threshold
        );
        assert_eq!(
            perturbed.short_threshold.to_bits(),
            0xBFD8_0552_2CA6_FA9A,
            "short_threshold moved: {} (the f64 reference is -0.37532476769039536)",
            perturbed.short_threshold
        );
        assert_eq!(
            perturbed
                .weights
                .iter()
                .map(|w| w.to_bits())
                .collect::<Vec<_>>(),
            vec![
                0x3FF2_A8BD_ADEE_FAFC,
                0xBFE0_ABE4_4E42_3F21,
                0x3FD2_E869_E7AE_8290,
            ],
            "the weight draws moved: {:?} (the f64 reference is \
             [1.166196517398645, -0.5209828880778994, 0.2954354059711841])",
            perturbed.weights
        );
        assert_eq!(
            perturbed.sl_pips.to_bits(),
            0x4030_BE74_D5AA_A3D4,
            "sl_pips moved: {} (the f64 reference is 16.743970255050797)",
            perturbed.sl_pips
        );
        assert_eq!(
            perturbed.tp_pips.to_bits(),
            0x4048_A846_152B_D173,
            "tp_pips moved: {} (the f64 reference is 49.31463875426825)",
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
        let space_len =
            neoethos_data::core::hpc_ta::search_working_set_batch(0, usize::MAX, true).space_len;
        let mut search = StreamingSearch {
            cursor: 0,
            batch_columns: usize::MAX,
            space_len,
            budget_rows: 1_000,
            replace_base_vocabulary: true,
            batches_started: 0,
            selection_seed: None,
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
            space_len: neoethos_data::core::hpc_ta::search_working_set_batch(0, usize::MAX, true)
                .space_len,
            budget_rows: 1_000,
            replace_base_vocabulary: true,
            batches_started: 0,
            selection_seed: None,
        };
        assert!(search.next_batch().is_none());
    }

    #[test]
    fn seeded_streaming_planner_advances_the_exact_selected_batches_without_wrapping() {
        let seed = 79;
        let first = neoethos_data::search_working_set_batch_seeded(0, 2, true, seed);
        let mut search = StreamingSearch {
            cursor: 0,
            batch_columns: 2,
            space_len: first.space_len,
            budget_rows: 100,
            replace_base_vocabulary: true,
            batches_started: 0,
            selection_seed: Some(seed),
        };
        assert_eq!(search.next_batch().as_deref(), Some(&first));
        let second =
            neoethos_data::search_working_set_batch_seeded(first.next_cursor, 2, true, seed);
        assert_eq!(search.next_batch().as_deref(), Some(&second));
        assert_eq!(search.cursor(), second.next_cursor);
        assert_eq!(search.batches_started(), 2);
        search.cursor = search.space_len();
        assert!(search.next_batch().is_none());
    }

    #[test]
    fn second_batch_context_is_exact_and_restores_nested_and_outer_callers() {
        let first = neoethos_data::search_working_set_batch_seeded(0, 2, true, 79);
        let second = neoethos_data::search_working_set_batch_seeded(first.next_cursor, 2, true, 79);
        assert!(second.cursor > 0);
        let (observed, _) = with_streaming_batch_context(second.cursor, || {
            assert_eq!(streaming_sweep_cursor(), second.cursor);
            let (nested, _) = with_streaming_batch_context(37, streaming_sweep_cursor);
            assert_eq!(nested, 37);
            assert_eq!(streaming_sweep_cursor(), second.cursor);
            streaming_sweep_cursor()
        });
        assert_eq!(observed, second.cursor);
        assert_eq!(streaming_sweep_cursor(), 0);
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
        assert_eq!(
            verdict.reason(),
            BatchAcceptReason::CandidateClearsFloor.as_str()
        );
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
        assert_eq!(
            verdict.reason(),
            BatchAcceptReason::UncertainNoMetrics.as_str()
        );
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
        assert!(crate::prefilter_schema_v1::is_prefilter_state_column_v1(
            "regime_vol_state"
        ));
        assert!(crate::prefilter_schema_v1::is_prefilter_state_column_v1(
            "smc_ob"
        ));
        assert!(crate::prefilter_schema_v1::is_prefilter_state_column_v1(
            "session_london_open"
        ));
        assert!(crate::prefilter_schema_v1::is_prefilter_state_column_v1(
            "fp_delta"
        ));
        assert!(!crate::prefilter_schema_v1::is_prefilter_state_column_v1(
            "rsi_14"
        ));
        assert!(!crate::prefilter_schema_v1::is_prefilter_state_column_v1(
            "H1_smc_ob"
        ));
    }
}

#[cfg(test)]
mod account_policy_and_normalization_scope_tests {
    use super::*;

    #[test]
    fn configured_initial_equity_reaches_search_and_every_backtest_template() {
        for initial_balance in [250.0, 10_000.0, 73_421.125] {
            let config = DiscoveryConfig {
                initial_balance,
                ..DiscoveryConfig::default()
            };
            let evaluation = config.evaluation_config(Some(1.1));
            let gene = Gene::default();
            let direct = GeneEvalSettingsResolver::for_slice(
                &config,
                std::iter::once(&gene),
                &[1.1],
                &[1.1],
                &[1.1],
            )
            .expect("one-bar equity fixture has no invalid adaptive stop data")
            .settings_for_gene(&gene);
            let population = PopulationTemplateResolver::new(&config, Some(1.1)).template(&gene);
            assert_eq!(
                evaluation.initial_equity.to_bits(),
                initial_balance.to_bits()
            );
            for settings in [direct, population] {
                assert_eq!(settings.initial_equity_override, Some(initial_balance));
                assert_eq!(
                    settings.initial_equity().to_bits(),
                    initial_balance.to_bits()
                );
            }
        }
    }

    #[test]
    fn holdout_preflight_rejects_a_full_data_fit_before_any_search_runs() -> Result<()> {
        let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let rows = ohlcv.close.len();
        let training_rows = canonical_discovery_normalization_training_rows(rows)?;
        let timestamps = ohlcv.timestamp.clone().expect("fixture timestamps");
        let columns = vec![neoethos_data::FeatureColumnF64::new(
            "signal",
            (0..rows).map(|row| row as f64 + 1.0).collect(),
            vec![neoethos_data::FeatureCellValidity::Valid; rows],
        )?];
        for (fit_rows, permitted) in [(training_rows.clone(), true), (0..rows, false)] {
            let features =
                neoethos_data::test_fixtures::ctrader_test_normalized_feature_frame_from_columns(
                    timestamps.clone(),
                    columns.clone(),
                    neoethos_data::FeatureBuildOptions {
                        normalization_training_rows: Some(fit_rows.clone()),
                        ..Default::default()
                    },
                )?;
            let anchor = features.provenance().bindings()[0].dataset_identity();
            let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &features)?;
            let input = CanonicalSearchRunInputV2::new_for_test_values(receipt, &features, &ohlcv)?;
            // A full-window research selection may use its full-window fit;
            // splitting off an unseen holdout is a different authority.
            CanonicalDiscoveryRunInputs::entire(&input)?;
            let split = CanonicalDiscoveryRunInputs::with_holdout(&input);
            if permitted {
                let split = split?;
                assert_eq!(
                    split
                        .selection()
                        .features()
                        .normalization_fitted_state()
                        .expect("selection retains the original fit")
                        .training_rows()?,
                    training_rows
                );
                assert_eq!(
                    split
                        .holdout()
                        .expect("held-out suffix")
                        .features()
                        .normalization_fitted_state()
                        .expect("holdout reuses the fit")
                        .training_rows()?,
                    training_rows
                );
            } else {
                let error = split.expect_err("a full-data fit must not be called holdout-safe");
                assert!(error.to_string().contains("beyond selection training rows"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "discovery_cost_tests.rs"]
mod cost_consistency_tests;

#[cfg(test)]
mod candidate_validation_order_tests {
    use super::*;
    use crate::funnel_profile::DiscoveryCandidateCensus;
    use crate::validation::WalkforwardSplitResult;

    fn summary(pnl: f64) -> WalkforwardSummary {
        WalkforwardSummary {
            walk_forward_splits: 1,
            avg_pnl: pnl,
            avg_win_rate: if pnl > 0.0 { 1.0 } else { 0.0 },
            avg_max_dd: 0.0,
            avg_max_consec_losses: 0.0,
            avg_daily_min_dd: 0.0,
            avg_max_daily_loss: 0.0,
            any_daily_loss_breach: false,
            any_consistency_violation: false,
            any_trade_limit_violation: false,
            all_min_trading_days_ok: true,
            splits: vec![WalkforwardSplitResult {
                split: 0,
                trades: 1,
                pnl,
                win_rate: if pnl > 0.0 { 1.0 } else { 0.0 },
                max_dd: 0.0,
                max_consec_losses: 0,
                daily_min_dd: 0.0,
                max_daily_loss: 0.0,
                daily_loss_breach: false,
                consistency_violation: false,
                trade_limit_violation: false,
                min_trading_days_ok: true,
                daily_returns: vec![pnl / 10_000.0],
                max_daily_dd_pct: 0.0,
                prop_compliant: true,
            }],
        }
    }

    fn candidate(index: usize) -> WalkforwardSelectionCandidate {
        WalkforwardSelectionCandidate {
            candidate_idx: index,
            gene: Gene {
                strategy_id: format!("candidate-{index}"),
                indices: vec![0],
                weights: vec![1.0],
                fitness: 100.0 - index as f64,
                ..Gene::default()
            },
            signals: (0..100)
                .map(|row| if (row >> (index % 6)) & 1 == 0 { 0 } else { 1 })
                .collect(),
            prop_firm_pass_rate: Some(0.80 + index as f64 / 100.0),
        }
    }

    fn collision_trades(pnls: [f64; 2]) -> Vec<Trade> {
        pnls.into_iter()
            .enumerate()
            .map(|(i, pnl)| Trade {
                entry_time: 1_735_689_600_000 + i as i64 * 86_400_000,
                exit_time: Some(1_735_693_200_000 + i as i64 * 86_400_000),
                pnl,
                pnl_pct: Some(pnl / 100.0),
                ..Trade::default()
            })
            .collect()
    }

    fn collision_metrics(pnls: [f64; 2]) -> StrategyMetrics {
        let mut metrics = StrategyQualityAnalyzer::default().analyze_strategy(
            "same-display-id",
            &collision_trades(pnls),
            100.0,
        );
        assert_eq!(metrics.total_trades, 2);
        metrics.equity_curve.clear();
        metrics
    }

    #[test]
    fn diagnostic_journals_prioritize_final_selections_beyond_the_first_fifty() -> Result<()> {
        let ranked = (0..60).map(|i| (i, candidate(i).gene)).collect::<Vec<_>>();
        let before = serde_json::to_vec(&ranked)?;
        let quality = (0..60).map(|i| (i, i % 2 == 1)).collect::<Vec<_>>();
        let indices = [55, 58, 52];
        let selected = indices
            .iter()
            .map(|&i| ranked[i].1.clone())
            .collect::<Vec<_>>();
        let plan = plan_diagnostic_candidates(&indices, &selected, &ranked, &quality, true, 50)?;
        let planned = plan.iter().map(|(i, _, _)| *i).collect::<Vec<_>>();
        assert_eq!(planned.len(), 50);
        assert_eq!(&planned[..3], &indices);
        assert_eq!(&planned[3..], &(0..47).collect::<Vec<_>>());
        assert_eq!(planned.iter().collect::<HashSet<_>>().len(), planned.len());
        assert_eq!(
            plan[0].2, true,
            "selected opportunistic lane remains attached"
        );
        for (index, gene, _) in plan {
            assert_eq!(
                serde_json::to_vec(gene)?,
                serde_json::to_vec(&ranked[index].1)?
            );
        }
        for cap in [0, 1, 2, 3] {
            let plan =
                plan_diagnostic_candidates(&indices, &selected, &ranked, &quality, true, cap)?;
            assert_eq!(plan.iter().map(|(i, _, _)| *i).collect::<Vec<_>>(), indices);
        }
        assert!(
            plan_diagnostic_candidates(&indices, &selected, &ranked, &quality, false, 50)?
                .is_empty()
        );
        assert_eq!(
            serde_json::to_vec(&ranked)?,
            before,
            "journaling must not change the search population"
        );
        Ok(())
    }

    #[test]
    fn diagnostic_journals_reject_rebound_or_ambiguous_selected_candidate_indices() -> Result<()> {
        let ranked = (0..3).map(|i| (i, candidate(i).gene)).collect::<Vec<_>>();
        let quality = vec![(0, false), (1, true), (2, false)];
        let selected = vec![ranked[2].1.clone()];
        assert!(plan_diagnostic_candidates(&[], &selected, &ranked, &quality, true, 50).is_err());
        assert!(plan_diagnostic_candidates(&[9], &selected, &ranked, &quality, true, 50).is_err());
        let mut rebound = selected.clone();
        rebound[0].weights[0] = 2.0;
        assert!(plan_diagnostic_candidates(&[2], &rebound, &ranked, &quality, true, 50).is_err());
        assert!(
            plan_diagnostic_candidates(
                &[2, 2],
                &[selected[0].clone(), selected[0].clone()],
                &ranked,
                &quality,
                true,
                50
            )
            .is_err()
        );
        assert!(
            plan_diagnostic_candidates(
                &[2],
                &selected,
                &ranked,
                &[(0, false), (0, true)],
                true,
                50
            )
            .is_err()
        );
        // Equal display IDs must never redirect a candidate to another genome.
        let mut colliding = ranked.clone();
        colliding[0].1.strategy_id = colliding[2].1.strategy_id.clone();
        colliding[0].1.weights[0] = 9.0;
        let plan = plan_diagnostic_candidates(&[2], &selected, &colliding, &quality, true, 1)?;
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].0, 2);
        assert_eq!(plan[0].1.weights, vec![1.0]);
        Ok(())
    }

    #[test]
    fn diagnostic_journals_replay_complete_selected_is_ledgers_and_reuse_them_for_curves()
    -> Result<()> {
        // A bounded captured-data accounting regression, not a GA/OOS result.
        // Keep the calibration/final suffix outside both reference and journal.
        let full_features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let full_ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
            full_features.provenance().bindings()[0].dataset_identity(),
            &full_features,
        )?;
        let input =
            CanonicalSearchRunInputV2::new_for_test_values(receipt, &full_features, &full_ohlcv)?;
        let windows = CanonicalDiscoveryRunInputs::with_holdout(&input)?;
        let selection = windows.selection();
        let features = selection.features();
        let ohlcv = selection.ohlcv();
        assert_eq!(features.n_samples(), 80);
        assert_eq!(windows.calibration().unwrap().features().n_samples(), 10);
        assert_eq!(windows.holdout().unwrap().features().n_samples(), 10);
        let config = DiscoveryConfig {
            evaluation_symbol: "EURUSD".to_string(),
            evaluation_account_currency: "USD".to_string(),
            evaluation_spread_pips: 0.0,
            evaluation_commission_per_trade: 0.0,
            initial_balance: 10_000.0,
            kill_zones_enabled: false,
            ..DiscoveryConfig::default()
        };
        let ranked = (0..60)
            .map(|i| {
                let mut gene = candidate(i).gene;
                gene.long_threshold = 0.0;
                gene.short_threshold = 0.0;
                gene.sl_pips = 20.0;
                gene.tp_pips = 1.0;
                (i, gene)
            })
            .collect::<Vec<_>>();
        let indices = [55, 58];
        let selected = indices
            .iter()
            .map(|&i| ranked[i].1.clone())
            .collect::<Vec<_>>();
        let quality = (0..60).map(|i| (i, false)).collect::<Vec<_>>();
        let evaluation = config.evaluation_config_with_smc_gate(ohlcv.close.last().copied(), 0.0);
        let smc = SmcGateArrays::build(features, ohlcv)?;
        let signals = selected
            .iter()
            .map(|gene| signals_for_gene_full_with_smc(features, gene, &evaluation, &smc))
            .collect::<Result<Vec<_>>>()?;
        let cached = indices
            .iter()
            .copied()
            .zip(signals.iter().map(Vec::as_slice))
            .collect();
        // Capacity 1 must retain BOTH final selections and no optional extras.
        let plan = plan_diagnostic_candidates(&indices, &selected, &ranked, &quality, true, 1)?;
        let logs = replay_diagnostic_candidates(
            plan,
            features,
            ohlcv,
            &config,
            &evaluation,
            &smc,
            &cached,
        )?;
        assert_eq!(logs.iter().map(|(i, _)| *i).collect::<Vec<_>>(), indices);
        let resolver = GeneEvalSettingsResolver::for_slice(
            &config,
            selected.iter(),
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
        )?;
        let (months, days) = month_day_indices(&features.timestamps);
        let mut metrics = Vec::new();
        for (position, (_, log)) in logs.iter().enumerate() {
            let gene = &selected[position];
            let confidences =
                account_sizing_confidences(features, gene, &evaluation, &smc, &signals[position])?;
            let (account_metrics, reference_trades) =
                crate::eval::evaluate_strategy_with_confidence_and_ledger_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &signals[position],
                    &confidences,
                    &months,
                    &days,
                    &features.timestamps,
                    &resolver.settings_for_gene(gene),
                )?;
            let mut measured = quality_analyzer_for_config(&config)
                .analyze_strategy_with_evaluation(
                    &gene.strategy_id,
                    &reference_trades,
                    config.initial_balance,
                    features.timestamps[0],
                    *features.timestamps.last().unwrap(),
                    &account_metrics,
                )?;
            assert!(
                measured.total_trades > 1,
                "the set cap must not truncate individual trades"
            );
            assert_eq!(log.strategy_id, gene.strategy_id);
            assert_eq!(log.trades.len(), measured.total_trades);
            assert_eq!(
                serde_json::to_vec(&log.trades)?,
                serde_json::to_vec(&reference_trades)?
            );
            let net: f64 = log.trades.iter().map(|trade| trade.pnl).sum();
            assert!((net - measured.net_profit).abs() <= 1e-9 * net.abs().max(1.0));
            assert!((net / config.initial_balance - measured.total_return_pct).abs() <= 1e-9);
            for trade in &log.trades {
                assert!(trade.entry_time >= features.timestamps[0]);
                let exit = trade.exit_time.expect("journal must contain closed trades");
                assert!(exit >= trade.entry_time && exit <= *features.timestamps.last().unwrap());
                assert_eq!(trade.pnl_pct, Some(trade.pnl / config.initial_balance));
            }
            measured.equity_curve.clear();
            metrics.push(measured);
        }
        let (logged_indices, journals): (Vec<_>, Vec<_>) = logs.into_iter().unzip();
        for index in indices {
            let curve = candidate_equity_curve(
                index,
                config.initial_balance,
                &logged_indices,
                &journals,
                || anyhow::bail!("selected journal must be reused, not replayed again"),
            )?;
            restore_candidate_quality_curve(index, &curve, &indices, &mut metrics)?;
        }
        assert!(
            metrics
                .iter()
                .all(|m| m.equity_curve.len() == m.total_trades + 1)
        );
        Ok(())
    }

    #[test]
    fn selected_curves_use_original_indices_not_colliding_display_ids() -> Result<()> {
        let quality_indices = [10, 20, 30];
        let mut metrics = vec![
            collision_metrics([50.0, -10.0]),
            collision_metrics([7.0, -2.0]),
            collision_metrics([-20.0, 5.0]),
        ];
        let original_scalars: Vec<_> = metrics
            .iter()
            .map(|m| {
                (
                    m.net_profit,
                    m.final_balance,
                    m.max_drawdown_money,
                    m.total_trades,
                )
            })
            .collect();
        // Logs have a different order, and the first selected candidate has no
        // retained log. All three rows have the same display ID and trade count.
        let log_indices = [20, 10];
        let logs: Vec<_> = [[7.0, -2.0], [50.0, -10.0]]
            .into_iter()
            .map(|pnls| LoggedStrategyTrades {
                strategy_id: "same-display-id".to_string(),
                opportunistic: false,
                trades: collision_trades(pnls),
            })
            .collect();
        let mut replayed = Vec::new();
        // These are the same per-candidate lookup/restore operations invoked
        // inside/after the production RAM-admitted map_bounded replay waves.
        for candidate_idx in [30, 10] {
            let curve = candidate_equity_curve(candidate_idx, 100.0, &log_indices, &logs, || {
                replayed.push(candidate_idx);
                assert_eq!(
                    candidate_idx, 30,
                    "only the missing exact-index log may replay"
                );
                Ok(collision_trades([-20.0, 5.0]))
            })?;
            restore_candidate_quality_curve(candidate_idx, &curve, &quality_indices, &mut metrics)?;
        }
        assert_eq!(replayed, vec![30]);
        assert_eq!(metrics[0].equity_curve, vec![100.0, 150.0, 140.0]);
        assert!(
            metrics[1].equity_curve.is_empty(),
            "nonselected equal-ID row stays scalar-only"
        );
        assert_eq!(metrics[2].equity_curve, vec![100.0, 80.0, 85.0]);
        assert_eq!(
            metrics
                .iter()
                .map(|m| (
                    m.net_profit,
                    m.final_balance,
                    m.max_drawdown_money,
                    m.total_trades
                ))
                .collect::<Vec<_>>(),
            original_scalars,
        );
        Ok(())
    }

    #[test]
    fn curve_identity_shape_errors_fail_before_replay_or_partial_assignment() {
        let logs = vec![LoggedStrategyTrades {
            strategy_id: "same-display-id".to_string(),
            opportunistic: false,
            trades: collision_trades([50.0, -10.0]),
        }];
        let mut replayed = false;
        assert!(
            candidate_equity_curve(10, 100.0, &[], &logs, || {
                replayed = true;
                Ok(Vec::new())
            })
            .is_err()
        );
        assert!(!replayed);
        assert!(
            candidate_equity_curve(
                10,
                100.0,
                &[10, 10],
                &[logs[0].clone(), logs[0].clone()],
                || { panic!("duplicate identity must not replay") }
            )
            .is_err()
        );

        let mut metrics = vec![
            collision_metrics([50.0, -10.0]),
            collision_metrics([7.0, -2.0]),
        ];
        let curve = [100.0, 150.0, 140.0];
        assert!(restore_candidate_quality_curve(10, &curve, &[10], &mut metrics).is_err());
        assert!(restore_candidate_quality_curve(99, &curve, &[10, 20], &mut metrics).is_err());
        // A later mismatched duplicate cannot leave the earlier row modified.
        metrics[1].total_trades = 3;
        assert!(restore_candidate_quality_curve(10, &curve, &[10, 10], &mut metrics).is_err());
        assert!(metrics.iter().all(|m| m.equity_curve.is_empty()));
    }

    #[test]
    fn fallback_duplicate_quality_row_restores_only_the_same_original_candidate() -> Result<()> {
        let mut metrics = vec![
            collision_metrics([50.0, -10.0]),
            collision_metrics([7.0, -2.0]),
            collision_metrics([50.0, -10.0]),
        ];
        restore_candidate_quality_curve(10, &[100.0, 150.0, 140.0], &[10, 20, 10], &mut metrics)?;
        assert_eq!(metrics[0].equity_curve, vec![100.0, 150.0, 140.0]);
        assert_eq!(metrics[2].equity_curve, metrics[0].equity_curve);
        assert!(metrics[1].equity_curve.is_empty());
        Ok(())
    }

    #[test]
    fn completed_quality_replays_are_published_before_a_later_chunk_error() -> Result<()> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let completed = AtomicUsize::new(0);
        let mut census = DiscoveryCandidateCensus {
            validation_candidates_admitted: 8,
            walkforward_not_tested: 8,
            ..Default::default()
        };
        let mut observed = Vec::new();
        let mut progress = |event| {
            if let DiscoveryProgress::CandidateCensusUpdated { census } = event {
                observed.push(census);
            }
        };
        // Exercise the real joined-chunk publication seam without running extra
        // backtests. A completed replay and its later processing are separate.
        let first: Result<Vec<_>> = (0..4)
            .into_par_iter()
            .map(|i| {
                completed.fetch_add(1, Ordering::Relaxed);
                Ok(i)
            })
            .collect();
        let rows = publish_completed_quality_chunk(first, &completed, &mut census, &mut progress)?;
        assert_eq!(rows, vec![0, 1, 2, 3]);
        assert_eq!(census.quality_evaluated, 4);

        let later_processing_error: Result<Vec<()>> = (0..1)
            .into_par_iter()
            .map(|_| {
                completed.fetch_add(1, Ordering::Relaxed);
                anyhow::bail!("fixture return-grid failure after successful replay")
            })
            .collect();
        let error = publish_completed_quality_chunk(
            later_processing_error,
            &completed,
            &mut census,
            &mut progress,
        )
        .expect_err("later processing must still fail");
        assert!(error.to_string().contains("after successful replay"));
        assert_eq!(census.quality_evaluated, 5);

        let replay_error: Result<Vec<()>> = (0..1)
            .into_par_iter()
            .map(|_| anyhow::bail!("fixture replay failed before completion"))
            .collect();
        assert!(
            publish_completed_quality_chunk(replay_error, &completed, &mut census, &mut progress)
                .is_err()
        );
        assert_eq!(census.quality_evaluated, 5);
        assert_eq!(
            observed
                .iter()
                .map(|c| c.quality_evaluated)
                .collect::<Vec<_>>(),
            vec![4, 5, 5]
        );
        assert!(observed.iter().all(|c| {
            c.walkforward_failed == 0 && c.walkforward_tested == 0 && c.walkforward_not_tested == 8
        }));
        Ok(())
    }

    #[test]
    fn robustness_removals_reconcile_before_final_artifact_failure() -> Result<()> {
        let mut census = DiscoveryCandidateCensus {
            validation_candidates_admitted: 6,
            ..Default::default()
        };
        let mut selected = select_walkforward_diverse_candidates(
            (0..6).map(candidate).collect(),
            &vec![summary(10.0); 6],
            DiscoveryMode::Risky,
            4,
            0.90,
            &mut census,
        )?;
        assert_eq!(selected.len(), 4);
        assert_eq!(census.robustness_removed, None);
        assert!(
            !census
                .counters()
                .iter()
                .any(|(key, _)| *key == "robustness_removed")
        );
        selected.remove(1);
        let mut funnel = crate::funnel_profile::FunnelProfile::new("EURUSD", "M5");
        let mut published = None;
        let mut progress = |event| {
            if let DiscoveryProgress::CandidateCensusUpdated { census } = event {
                published = Some(census);
            }
        };
        let result: Result<()> = (|| {
            publish_portfolio_after_robustness(
                selected.len(),
                false,
                &mut census,
                &mut funnel,
                &mut progress,
            );
            anyhow::bail!("fixture final artifact failure")
        })();
        assert!(result.is_err());
        assert_eq!(published, Some(census.clone()));
        assert_eq!(funnel.candidate_census, Some(census.clone()));
        assert_eq!(
            (census.robustness_removed, census.portfolio_selected),
            (Some(1), 3)
        );
        assert_eq!(census.walkforward_failed, 0);
        assert_eq!(census.rejected_by_correlation, 0);
        assert_eq!(census.portfolio_capacity_not_selected, 2);
        assert_eq!(
            census.walkforward_passed,
            census.rejected_by_correlation
                + census.portfolio_capacity_not_selected
                + census
                    .robustness_removed
                    .expect("completed membership transition")
                + census.portfolio_selected
        );
        assert_eq!(
            census.correlation_tested,
            census.rejected_by_correlation
                + census
                    .robustness_removed
                    .expect("completed membership transition")
                + census.portfolio_selected
        );
        let stage = funnel
            .stages
            .iter()
            .find(|s| s.name == "portfolio_after_robustness")
            .expect("explicit retained-membership stage");
        assert_eq!((stage.count_in, stage.count_out, stage.rejected), (4, 3, 1));
        assert_eq!(stage.top_reasons, vec![("robustness_removed".into(), 1)]);
        Ok(())
    }

    #[test]
    fn retained_all_and_fallback_are_not_reported_as_robustness_passes() {
        // The census observes membership only. Both a skipped screen and the
        // existing all-fail/retain-all policy keep four, without proving a pass.
        for (case, before, retained_count, fallback_mode, selected_count) in [
            ("skipped", 4, 4, false, 4),
            ("all kept", 4, 4, false, 4),
            ("all failed but retained", 4, 4, false, 4),
            ("diagnostic fallback", 0, 8, true, 0),
            ("empty portfolio", 0, 0, false, 0),
        ] {
            let mut census = DiscoveryCandidateCensus {
                portfolio_selected: before,
                walkforward_failed: 2,
                ..Default::default()
            };
            let mut funnel = crate::funnel_profile::FunnelProfile::new("EURUSD", "M5");
            publish_portfolio_after_robustness(
                retained_count,
                fallback_mode,
                &mut census,
                &mut funnel,
                &mut |_| {},
            );
            assert_eq!(census.robustness_removed, Some(0), "{case}");
            assert_eq!(census.portfolio_selected, selected_count, "{case}");
            assert_eq!(census.walkforward_failed, 2, "{case}");
            let stage = funnel
                .stages
                .iter()
                .find(|s| s.name == "portfolio_after_robustness")
                .unwrap();
            assert_eq!(stage.count_in, selected_count, "{case}");
            assert_eq!(stage.count_out, selected_count, "{case}");
            assert_eq!(stage.rejected, 0, "{case}");
            assert!(stage.top_reasons.is_empty(), "{case}");
            assert!(!funnel.stages.iter().any(|s| s.name == "passed_robustness"));
        }
    }

    #[test]
    fn all_candidates_receive_wf_before_capacity_and_lower_rank_backfills() -> Result<()> {
        // This exercises the production batching and mode-aware selection
        // seams, not a copied predicate or a source-text ordering assertion.
        for mode in [
            DiscoveryMode::Risky,
            DiscoveryMode::PropFirm,
            DiscoveryMode::Strict,
        ] {
            let mut reference = None;
            for width in [1, 2, 10] {
                let mut evaluated = Vec::new();
                let summaries = evaluate_walkforward_batches(8, width, |range| {
                    assert!(range.len() <= width);
                    evaluated.extend(range.clone());
                    Ok(range
                        .map(|i| summary(if i < 4 { -1.0 } else { 10.0 }))
                        .collect())
                })?;
                assert_eq!(evaluated, (0..8).collect::<Vec<_>>());
                let mut census = DiscoveryCandidateCensus {
                    validation_candidates_admitted: 8,
                    ..Default::default()
                };
                let selected = select_walkforward_diverse_candidates(
                    (0..8).map(candidate).collect(),
                    &summaries,
                    mode,
                    4,
                    0.90,
                    &mut census,
                )?;
                let ids = selected
                    .iter()
                    .map(|c| c.gene.strategy_id.clone())
                    .collect::<Vec<_>>();
                assert_eq!(
                    ids,
                    (4..8).map(|i| format!("candidate-{i}")).collect::<Vec<_>>()
                );
                assert_eq!(
                    (
                        census.walkforward_tested,
                        census.walkforward_passed,
                        census.walkforward_failed,
                        census.walkforward_not_tested
                    ),
                    (8, 4, 4, 0)
                );
                assert_eq!(
                    (
                        census.correlation_tested,
                        census.portfolio_capacity_not_selected
                    ),
                    (4, 0)
                );
                for (i, actual) in (4..8).zip(&selected) {
                    let expected = candidate(i);
                    assert_eq!(actual.candidate_idx, expected.candidate_idx);
                    assert_eq!(actual.gene, expected.gene);
                    assert_eq!(actual.signals, expected.signals);
                    assert_eq!(actual.prop_firm_pass_rate, expected.prop_firm_pass_rate);
                }
                if let Some(reference) = &reference {
                    assert_eq!(&ids, reference);
                } else {
                    reference = Some(ids);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn calibration_precedes_capacity_and_keeps_the_profitable_research_reserve() -> Result<()> {
        let all = (0..12).map(candidate).collect::<Vec<_>>();
        let verdicts = vec![
            WalkforwardVerdict {
                tested: true,
                passed: true
            };
            all.len()
        ];
        // All twelve passed internal WF. The first four fail the separate
        // selection calibration; eight remain eligible, not only four.
        let profitable = all
            .iter()
            .skip(4)
            .map(|item| stable_json_hash(&item.gene))
            .collect::<Result<HashSet<_>>>()?;
        let mut census = DiscoveryCandidateCensus {
            validation_candidates_admitted: all.len(),
            ..Default::default()
        };
        let selected = select_walkforward_diverse_candidates_with_signals(
            all,
            &verdicts,
            Some(&profitable),
            4,
            0.90,
            &mut census,
            |_| anyhow::bail!("fixture already has exact selected signals"),
        )?;
        assert_eq!(
            selected
                .iter()
                .map(|item| item.candidate_idx)
                .collect::<Vec<_>>(),
            vec![4, 5, 6, 7]
        );
        assert_eq!(
            profitable.len(),
            8,
            "active capacity must not shrink the positive research pool"
        );
        assert_eq!(
            (
                census.walkforward_tested,
                census.walkforward_passed,
                census.walkforward_failed
            ),
            (12, 12, 0)
        );
        assert_eq!(
            (
                census.correlation_tested,
                census.portfolio_capacity_not_selected,
                census.portfolio_selected
            ),
            (4, 4, 4)
        );
        let mut empty_census = DiscoveryCandidateCensus::default();
        assert!(
            select_walkforward_diverse_candidates_with_signals(
                (0..12).map(candidate).collect(),
                &verdicts,
                Some(&HashSet::new()),
                4,
                0.90,
                &mut empty_census,
                |_| anyhow::bail!("failed calibration must not reach correlation"),
            )?
            .is_empty()
        );
        assert_eq!(empty_census.walkforward_failed, 0);
        assert_eq!(empty_census.correlation_tested, 0);
        Ok(())
    }

    #[test]
    fn goal_ranking_uses_paired_ga_metrics_then_refreshed_full_window_evidence() -> Result<()> {
        let goal = crate::scoring::RiskyGrowthGoal {
            start_balance: 100.0,
            target_balance: 50_000.0,
            horizon_days: 180.0,
        };
        let mut genes = vec![candidate(0).gene, candidate(1).gene];
        // Deliberately misleading legacy summaries must not supply net/pace.
        genes[0].expectancy = 10_000.0;
        genes[0].trades_count = 10_000;
        genes[1].expectancy = -10_000.0;
        let first = [
            100.0, 1.0, 10_100.0, 0.01, 0.6, 1.5, 10.0, 0.5, 10.0, 0.8, 0.005,
        ];
        let second = [
            200.0, 1.0, 10_200.0, 0.01, 0.6, 1.5, 20.0, 0.5, 10.0, 0.8, 0.005,
        ];
        let day = 86_400_000_i64;
        let earlier = [1_700_000_000_000, 1_700_000_000_000 + 10 * day];
        let recent = [1_710_000_000_000, 1_710_000_000_000 + 10 * day];
        for timestamps in [&earlier, &recent] {
            let ranked = rank_candidates_on_matching_window(
                genes.clone(),
                vec![first, second],
                10_000.0,
                timestamps,
                Some(goal),
            )?;
            assert_eq!(
                ranked.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
                vec![1, 0]
            );
            assert_eq!(ranked[0].1, genes[1]);
            let actual = full_window_candidate_ranking_score(
                &second,
                999.0,
                10_000.0,
                timestamps,
                Some(goal),
            );
            assert_eq!(
                actual.to_bits(),
                crate::scoring::ga_fitness_goal(&second, 10_000.0, 10.0, goal).to_bits()
            );
        }
        assert!(
            rank_candidates_on_matching_window(genes, vec![first], 10_000.0, &earlier, Some(goal))
                .is_err()
        );
        let full_span = [earlier[0], earlier[0] + 40 * day];
        let mut full_a = first;
        full_a[0] = 800.0;
        let mut full_b = second;
        full_b[0] = 50.0;
        assert!(
            full_window_candidate_ranking_score(&full_a, 1.0, 10_000.0, &full_span, Some(goal))
                > full_window_candidate_ranking_score(
                    &full_b,
                    100.0,
                    10_000.0,
                    &full_span,
                    Some(goal)
                )
        );
        assert_eq!(
            full_window_candidate_ranking_score(&full_a, 37.0, 10_000.0, &full_span, None),
            37.0
        );
        assert_ne!(
            full_window_candidate_ranking_score(&second, 0.0, 10_000.0, &earlier, Some(goal)),
            full_window_candidate_ranking_score(&second, 0.0, 10_000.0, &full_span, Some(goal))
        );
        Ok(())
    }

    #[test]
    fn broad_wf_census_does_not_materialize_failed_or_capacity_excluded_signals() -> Result<()> {
        let count = 503;
        let candidates = (0..count)
            .map(|idx| {
                let mut entry = candidate(idx);
                entry.signals = Vec::new();
                entry
            })
            .collect();
        let verdicts = (0..count)
            .map(|idx| WalkforwardVerdict {
                tested: true,
                passed: idx >= 250,
            })
            .collect::<Vec<_>>();
        let mut census = DiscoveryCandidateCensus {
            validation_candidates_admitted: count,
            ..Default::default()
        };
        let mut loaded = Vec::new();
        let selected = select_walkforward_diverse_candidates_with_signals(
            candidates,
            &verdicts,
            None,
            4,
            0.90,
            &mut census,
            |gene| {
                let idx = gene
                    .strategy_id
                    .strip_prefix("candidate-")
                    .unwrap()
                    .parse::<usize>()?;
                loaded.push(idx);
                Ok(candidate(idx).signals)
            },
        )?;
        assert_eq!(loaded, vec![250, 251, 252, 253]);
        assert_eq!(selected.len(), 4);
        assert_eq!(census.walkforward_tested, 503);
        assert_eq!(census.walkforward_passed, 253);
        assert_eq!(census.walkforward_failed, 250);
        assert_eq!(census.walkforward_not_tested, 0);
        assert_eq!(census.portfolio_capacity_not_selected, 249);
        assert_eq!(census.correlation_tested, 4);
        for (idx, selected) in (250..254).zip(selected) {
            assert_eq!(selected.candidate_idx, idx);
            assert_eq!(selected.gene, candidate(idx).gene);
            assert_eq!(selected.signals, candidate(idx).signals);
            assert_eq!(
                selected.prop_firm_pass_rate,
                candidate(idx).prop_firm_pass_rate
            );
        }
        Ok(())
    }

    #[test]
    fn correlation_rejection_backfills_and_capacity_is_not_a_failed_wf() -> Result<()> {
        let mut candidates = (0..5).map(candidate).collect::<Vec<_>>();
        candidates[1].signals = candidates[0].signals.clone();
        let mut census = DiscoveryCandidateCensus {
            validation_candidates_admitted: 5,
            ..Default::default()
        };
        let selected = select_walkforward_diverse_candidates(
            candidates,
            &vec![summary(10.0); 5],
            DiscoveryMode::Risky,
            2,
            0.90,
            &mut census,
        )?;
        assert_eq!(
            selected
                .iter()
                .map(|c| c.gene.strategy_id.as_str())
                .collect::<Vec<_>>(),
            ["candidate-0", "candidate-2"]
        );
        assert_eq!(
            (census.walkforward_passed, census.walkforward_failed),
            (5, 0)
        );
        assert_eq!(
            (
                census.correlation_tested,
                census.rejected_by_correlation,
                census.portfolio_capacity_not_selected
            ),
            (3, 1, 2)
        );
        Ok(())
    }

    #[test]
    fn no_wf_folds_is_not_tested_and_zero_survivors_never_pass() -> Result<()> {
        let mut untested = summary(10.0);
        untested.walk_forward_splits = 0;
        untested.splits.clear();
        let mut census = DiscoveryCandidateCensus {
            validation_candidates_admitted: 4,
            ..Default::default()
        };
        let selected = select_walkforward_diverse_candidates(
            vec![candidate(0), candidate(1)],
            &[untested, summary(-1.0)],
            DiscoveryMode::Risky,
            4,
            0.90,
            &mut census,
        )?;
        assert!(selected.is_empty());
        assert_eq!(
            (
                census.walkforward_tested,
                census.walkforward_passed,
                census.walkforward_failed,
                census.walkforward_not_tested
            ),
            (1, 0, 1, 3)
        );
        assert!(!DiscoveryValidationGates::pending().is_portfolio_export_ready());
        assert!(evaluate_walkforward_batches(3, 2, |_| Ok(vec![summary(1.0)])).is_err());
        Ok(())
    }

    #[test]
    fn mode_specific_wf_risk_constraints_remain_distinct() -> Result<()> {
        let mut breach = summary(10.0);
        breach.any_daily_loss_breach = true;
        assert!(walkforward_summary_passed(&breach, DiscoveryMode::Risky));
        assert!(!walkforward_summary_passed(
            &breach,
            DiscoveryMode::PropFirm
        ));
        assert!(!walkforward_summary_passed(&breach, DiscoveryMode::Strict));
        Ok(())
    }

    #[test]
    fn selected_evidence_is_bound_to_exact_backfilled_genes() -> Result<()> {
        let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
            features.provenance().bindings()[0].dataset_identity(),
            &features,
        )?;
        let input = CanonicalSearchRunInputV2::new_for_test_values(receipt, &features, &ohlcv)?;
        let scope = CanonicalSearchArtifactScopeV2::from_run_input(
            CanonicalSearchWindowRoleV1::DiscoveryInput,
            &input,
        )?;
        let summaries = (0..6)
            .map(|i| summary(if i < 4 { -1.0 } else { 1.0 }))
            .collect::<Vec<_>>();
        let mut census = DiscoveryCandidateCensus {
            validation_candidates_admitted: 6,
            ..Default::default()
        };
        let selected = select_walkforward_diverse_candidates(
            (0..6).map(candidate).collect(),
            &summaries,
            DiscoveryMode::Risky,
            4,
            0.9,
            &mut census,
        )?;
        let genes = selected.into_iter().map(|c| c.gene).collect::<Vec<_>>();
        let hash = "fnv64:0123456789abcdef";
        let mut canonical = (0..6)
            .map(|i| {
                CanonicalBacktestArtifactFile::new(
                    scope.clone(),
                    hash,
                    &candidate(i).gene,
                    BacktestMetrics::from_metric_array([0.0; 11]),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut wf = (0..6)
            .map(|i| {
                WalkforwardValidationArtifactFile::new(
                    scope.clone(),
                    hash,
                    &candidate(i).gene,
                    summaries[i].clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        retain_selection_validation_artifacts_for_final_portfolio(&genes, &mut canonical, &mut wf)?;
        assert_eq!((canonical.len(), wf.len()), (2, 2));
        validate_exact_artifact_set("canonical", &canonical, &genes, &scope, hash, true)?;
        validate_exact_artifact_set("walkforward", &wf, &genes, &scope, hash, true)?;
        assert!(
            wf.iter()
                .all(|a| walkforward_summary_passed(a.summary(), DiscoveryMode::Risky))
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "discovery_holdout_signal_tests.rs"]
mod holdout_signal_tests;

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;
