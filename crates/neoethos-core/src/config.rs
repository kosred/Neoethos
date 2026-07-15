// Core configuration structures for Forex trading system
// Project configuration loader.

use crate::contracts::CANONICAL_TIMEFRAMES;
use crate::domain::prop_firm::{PropFirmConstraints, PropFirmPreset, PropFirmRuntimeDefaults};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serialize a `HashMap` in SORTED key order (audit M06/M07 follow-up).
/// HashMap iteration order is randomized per process, so `Settings::save`
/// reshuffled these config maps on every write — dirtying config.yaml in git
/// with no real change. Sorting on serialize makes two saves of equivalent
/// settings byte-identical. Lookup semantics (the public API) are unchanged.
fn serialize_sorted_map<S, V>(map: &HashMap<String, V>, ser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    V: Serialize,
{
    let sorted: std::collections::BTreeMap<&String, &V> = map.iter().collect();
    serde::Serialize::serialize(&sorted, ser)
}

/// Sorted serialization for a nested map (both levels ordered).
fn serialize_sorted_nested_map<S>(
    map: &HashMap<String, HashMap<String, String>>,
    ser: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let sorted: std::collections::BTreeMap<&String, std::collections::BTreeMap<&String, &String>> =
        map.iter().map(|(k, v)| (k, v.iter().collect())).collect();
    serde::Serialize::serialize(&sorted, ser)
}
use std::path::PathBuf;

/// Public, no-API-key financial NEWS RSS feeds for the AI news desk
/// (`GET /news/feed`). Verified reachable 2026-06-30 (HTTP 200 + XML). The
/// economic *calendar* is separate (`news_calendar_source`) — ForexFactory's
/// ffcal XML is a calendar format, not RSS, so it does NOT belong here. Used
/// both as the default and as the runtime fallback when a user's configured
/// feeds are all unreachable, so a stale config never leaves the desk blank.
pub fn default_news_rss_feeds() -> Vec<String> {
    vec![
        "https://www.investing.com/rss/news.rss".to_string(),
        "https://www.fxstreet.com/rss/news".to_string(),
        "https://www.cnbc.com/id/100003114/device/rss/rss.html".to_string(),
    ]
}

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemConfig {
    pub symbol: String,
    /// Market Watch live-tick subscription set (F-338). Empty → the spot
    /// streamer falls back to `DEFAULT_STREAMED_SYMBOLS` (the 8 majors).
    /// The operator edits this from Market Watch; it takes effect on the
    /// next backend start.
    #[serde(default)]
    pub watchlist: Vec<String>,
    /// Operator's account currency for cost-model FX conversions
    /// (commission, swap, pnl_conversion_fee → account ccy) and the
    /// risk-gate sizing math.
    ///
    /// Population paths:
    ///  1. Manual via `config.yaml` `system.account_currency: "USD"`.
    ///  2. Auto from the cTrader trader profile when the broker session
    ///     is alive — the `/account/snapshot` bridge resolves
    ///     `ProtoOATrader.depositAssetId` → currency name via the
    ///     asset table and writes it back here (Phase D follow-up).
    ///
    /// (The legacy `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` env fallback was
    /// retired in v0.4.36 — config is the single source.)
    ///
    /// **Empty string (`""`) is the deliberate fail-loud default**
    /// matching the `symbol` field's policy — `DiscoveryConfig::
    /// from_settings()` propagates it to `evaluation_account_currency`
    /// and the cost-model NaN-sentinel guard then rejects backtests
    /// rather than silently lying about commission/swap values.
    /// Operators must populate before running discovery.
    #[serde(default)]
    pub account_currency: String,
    pub data_dir: PathBuf,
    /// UI language for the Flutter front-end: `"en"` (default) or `"el"`
    /// (Greek). Persisted here so the choice survives restarts and travels
    /// with config.yaml — the app's single source of truth — rather than a
    /// separate Flutter store. The backend does not consume it; it is surfaced
    /// via GET / POST `/settings` for the Settings language picker.
    pub ui_locale: String,
    /// Top-level **trading mode** — the single master switch the operator picks
    /// in the Risk screen. Two mutually-exclusive values:
    ///   - `"risky"`     → aggressive capital multiplication (small balance →
    ///     large target, ASAP). Drives discovery into `DiscoveryMode::Risky`
    ///     (high-risk filter floors, growth-tilted ranking, no prop-firm gate).
    ///   - `"prop_firm"` → safety / stability: pass prop-firm challenges and
    ///     bank a steady monthly return. Drives `DiscoveryMode::PropFirm` (FTMO
    ///     window-pass gate) and the active `risk.preset` constraints.
    /// Search/discovery + risk framing orient around this one choice. An
    /// explicit `models.discovery_mode = "strict"` is a power-user escape hatch
    /// that overrides the discovery side only. Default `"prop_firm"`.
    pub trading_mode: String,
    /// Risky-Mode goal — capital multiplication. The operator sets where to
    /// start, where to reach, and by when; in Risky mode these PRESSURE the
    /// strategy search to surface portfolios that can compound from start to
    /// target within the horizon (see `DiscoveryConfig::risky_*` + the
    /// target-aware candidate ranking). Sizing is a fraction of the *current*
    /// balance, so risk compounds with the bankroll. Defaults 100 -> 50,000 in
    /// 180 days (~6 months — beyond that it is closer to normal trading, per
    /// the operator's Risky-vs-normal distinction). Fully operator-editable.
    pub risky_start_balance_usd: f64,
    pub risky_target_balance_usd: f64,
    pub risky_horizon_days: u32,
    /// When auto-cull permanently retires a live strategy, automatically
    /// queue a fresh Discovery run on the same symbol + base timeframe to
    /// refill the gap (the retired strategy itself can never come back —
    /// its fingerprint stays blacklisted). The Symbiotic-GP retraining-trigger
    /// loop (2026-07-02). Default ON; toggle in Settings.
    pub auto_rediscover_on_cull: bool,
    pub multi_resolution_enabled: bool,
    pub multi_resolution_timeframes: Vec<String>,
    pub multi_resolution_prefix_base: bool,
    pub indices_path: String,
    pub use_online_indices: bool,
    pub base_timeframe: String,
    pub use_volume_features: bool,
    pub higher_timeframes: Vec<String>,
    pub required_timeframes: Vec<String>,
    pub enable_level2: bool,
    pub level2_depth_levels: usize,
    /// Broker time zone used for prop-firm calendar-day boundaries (e.g.
    /// daily-DD reset). Most cTrader prop firms run on EET ("Europe/Athens",
    /// UTC+2/+3); some run pure UTC. When set, the trading runtime computes
    /// `day_id` against this offset instead of the local clock. Empty string
    /// falls back to UTC. M12 in the audit.
    #[serde(default)]
    pub broker_timezone: String,
    pub poll_interval_seconds: u64,
    pub metrics_db_path: PathBuf,
    pub cache_dir: PathBuf,
    pub n_jobs: usize,
    pub enable_gpu_preference: String,
    // agent 2026-06-05 overfitting fix: removed three dead `discovery_*` fields
    // (`discovery_auto_cap` / `discovery_max_rows` / `discovery_stream`). They
    // were never read anywhere in the workspace — the REAL discovery row cap is
    // `models.prop_search_max_rows` (→ DiscoveryConfig.max_rows, discovery.rs).
    // SystemConfig does NOT derive `#[serde(deny_unknown_fields)]`, so any stale
    // copies of these keys still in a user's config.yaml are ignored, not errors.
    pub enable_gpu: bool,
    pub num_gpus: usize,
    pub device: String,
    pub evo_multiproc_per_gpu: bool,
    pub cache_training_frames: bool,
    pub training_cache_max_bytes: usize,
    pub max_training_rows_per_tf: usize,
    pub downcast_training_float32: bool,
    pub vortex_memory_map: bool,
    pub smc_freshness_limit: usize,
    pub smc_atr_displacement: f64,
    pub smc_max_levels: usize,
    pub smc_use_cuda: bool,
    /// Hardware / accelerator runtime knobs (config-driven replacement for the
    /// env vars read by `HardwareRuntimeOverrides::from_env`). See
    /// [`HardwareConfig`].
    #[serde(default)]
    pub hardware: HardwareConfig,
}

/// Hardware / accelerator runtime knobs — config-driven replacement for the env
/// vars read by [`crate::system::HardwareRuntimeOverrides::from_env`]:
/// `NEOETHOS_BOT_CPU_BUDGET`, `NEOETHOS_BOT_TRAIN_PRECISION` (+ the legacy
/// `FOREX_TRAIN_PRECISION` remnant), `NEOETHOS_BOT_{CUDA,ROCM,WGPU}_PRECISIONS`,
/// `NEOETHOS_BOT_WGPU_DEVICES`. All-`None`/empty defaults reproduce the
/// historical env-absent behaviour.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HardwareConfig {
    /// CPU thread budget for model training; `None` = auto (cores-based).
    pub cpu_budget: Option<usize>,
    /// Forced training precision; `None` = auto per accelerator.
    pub training_precision: Option<crate::system::TrainingPrecision>,
    /// Per-backend precision ladders; `None` = engine defaults.
    pub cuda_precisions: Option<Vec<crate::system::TrainingPrecision>>,
    pub rocm_precisions: Option<Vec<crate::system::TrainingPrecision>>,
    pub wgpu_precisions: Option<Vec<crate::system::TrainingPrecision>>,
    /// Explicit Vulkan/WGPU device names; empty = auto-enumerate.
    pub wgpu_device_names: Vec<String>,
}

impl Default for SystemConfig {
    fn default() -> Self {
        let n_jobs = std::thread::available_parallelism()
            .map(|n| (n.get() - 1).max(1))
            .unwrap_or(1);

        Self {
            // F-129 fix (2026-05-25): the previous defaults hardcoded
            // `symbol = "EURUSD"` + `symbols = vec!["EURUSD"]`. Both
            // are synthetic-data violations per the operator's
            // real-data directive 2026-05-24. Empty defaults force
            // the loader / caller to populate from real `config.yaml`
            // (which is the production path — `SystemConfig::default()`
            // is only the seed for serde defaults). Any production code
            // that runs against the all-empty default will hit the
            // downstream guard that rejects empty-symbol orders
            // (see `risk_gate::prop_firm_pre_trade_check` Batch B Pass 3).
            symbol: String::new(),
            watchlist: Vec::new(),
            // F-304 fix (2026-05-28): empty default forces operator/
            // broker-session population, matching `symbol`. The
            // cost-model NaN-sentinel guard rejects empty values
            // downstream, so a bare-install run fails LOUD instead of
            // silently producing zero-trade GA results from a NaN pip
            // value.
            account_currency: String::new(),
            data_dir: PathBuf::from("data"),
            ui_locale: "en".to_string(),
            trading_mode: "prop_firm".to_string(),
            risky_start_balance_usd: 100.0,
            risky_target_balance_usd: 50000.0,
            risky_horizon_days: 180,
            auto_rediscover_on_cull: true,
            multi_resolution_enabled: true,
            multi_resolution_timeframes: CANONICAL_TIMEFRAMES
                .iter()
                .map(|tf| (*tf).to_string())
                .collect(),
            multi_resolution_prefix_base: false,
            indices_path: String::new(),
            use_online_indices: false,
            base_timeframe: "M1".to_string(),
            use_volume_features: true,
            higher_timeframes: CANONICAL_TIMEFRAMES
                .iter()
                .map(|tf| (*tf).to_string())
                .collect(),
            required_timeframes: CANONICAL_TIMEFRAMES
                .iter()
                .map(|tf| (*tf).to_string())
                .collect(),
            enable_level2: false,
            level2_depth_levels: 10,
            broker_timezone: String::new(), // empty = fall back to UTC
            poll_interval_seconds: 60,
            metrics_db_path: PathBuf::from("metrics.sqlite"),
            cache_dir: PathBuf::from("cache"),
            n_jobs,
            enable_gpu_preference: "auto".to_string(),
            // agent 2026-06-05 overfitting fix: dead `discovery_*` fields removed
            // (see struct decl). The real row cap is `models.prop_search_max_rows`.
            enable_gpu: false,
            num_gpus: 0,
            device: "cpu".to_string(),
            evo_multiproc_per_gpu: true,
            cache_training_frames: false,
            training_cache_max_bytes: 2_000_000_000,
            max_training_rows_per_tf: 0,
            downcast_training_float32: true,
            vortex_memory_map: true,
            smc_freshness_limit: 0,
            smc_atr_displacement: 0.0,
            smc_max_levels: 0,
            smc_use_cuda: false,
            hardware: HardwareConfig::default(),
        }
    }
}

impl SystemConfig {
    /// Resolve the effective **base timeframe** from config.
    ///
    /// THE single source of truth shared by BOTH the CLI (`default_base_tf`)
    /// and the app server (`/engines/*/start`) so the two never diverge.
    /// Operator mandate (2026-06-04): the bot must behave identically whether
    /// driven from the UI or the CLI — no difference anywhere.
    pub fn resolve_base_timeframe(&self) -> String {
        self.base_timeframe.trim().to_string()
    }

    /// Resolve the effective **symbol** from config (shared by CLI + server).
    pub fn resolve_symbol(&self) -> String {
        self.symbol.trim().to_string()
    }

    /// Resolve the effective **higher timeframes** for an already-resolved
    /// `base`, honouring `multi_resolution_enabled` / `multi_resolution_timeframes`
    /// / `higher_timeframes` exactly. SHARED by CLI + server.
    ///
    /// - When multi-resolution is on and a non-empty explicit list is set, that
    ///   list wins (minus any entry equal to `base`).
    /// - Otherwise the configured `higher_timeframes` are filtered to those
    ///   strictly *above* `base` in canonical order (never a lower/equal TF).
    ///
    /// The filter is relative to the **effective** `base` passed in (which may be
    /// a CLI `--base` / payload override), not necessarily `self.base_timeframe`
    /// — so an overridden base always gets the correct top-down ladder above it.
    pub fn resolve_higher_timeframes(&self, base: &str) -> Vec<String> {
        let base_trim = base.trim();
        if self.multi_resolution_enabled && !self.multi_resolution_timeframes.is_empty() {
            self.multi_resolution_timeframes
                .iter()
                .map(|tf| tf.trim().to_string())
                .filter(|tf| !tf.is_empty() && !tf.eq_ignore_ascii_case(base_trim))
                .collect()
        } else {
            let above = crate::contracts::canonical_higher_timeframes(base_trim);
            self.higher_timeframes
                .iter()
                .map(|tf| tf.trim().to_string())
                .filter(|tf| !tf.is_empty() && above.iter().any(|a| a.eq_ignore_ascii_case(tf)))
                .collect()
        }
    }
}

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    /// Named prop-firm preset that seeds every other field in this
    /// struct. The runtime is firm-agnostic; this field just selects
    /// which lookup table populates the numeric thresholds at default
    /// construction. Operators can override any field below — preset
    /// values are seeds, not locks. Setting `preset: none` disables
    /// the external-challenge gate without touching the other fields.
    #[serde(default)]
    pub preset: PropFirmPreset,
    pub initial_balance: f64,
    pub monthly_profit_target_pct: f64,
    pub min_risk_per_trade: f64,
    pub max_risk_per_trade: f64,
    pub risk_per_trade: f64,
    /// Portfolio-level cap on TOTAL concurrent risk across all running live
    /// engines, as a balance fraction (e.g. 0.05 = at most ~5% of the account
    /// at risk across every open autopilot position at once). Each engine
    /// budgets its entry against `cap − (open positions × risk_per_trade)`,
    /// sizing down or skipping when the budget is spent. `0.0` disables the
    /// cap (per-engine sizing only — the pre-2026-07 behavior).
    pub max_portfolio_risk: f64,
    pub daily_drawdown_limit: f64,
    pub total_drawdown_limit: f64,
    pub min_risk_reward: f64,
    pub max_lot_size: f64,
    pub require_stop_loss: bool,
    pub challenge_mode: bool,
    pub challenge_phase: String,
    pub prop_firm_rules: bool,
    pub kill_zones_enabled: bool,
    pub max_trades_per_day: usize,
    pub recovery_mode_enabled: bool,
    pub feature_drift_threshold: f64,
    pub high_quality_confidence: f64,
    pub atr_period: usize,
    pub atr_stop_multiplier: f64,
    pub triple_barrier_max_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub slippage_pips: f64,
    pub commission_per_lot: f64,
    pub backtest_spread_pips: f64,
    pub conformal_enabled: bool,
    pub conformal_alpha: f64,
    pub conformal_abstain_min_set_size: usize,
    pub meta_label_tp_pips: Option<f64>,
    pub meta_label_sl_pips: Option<f64>,
    pub meta_label_max_hold_bars: usize,
    pub meta_label_min_dist: f64,
    pub meta_label_fixed_sl: f64,
    pub meta_label_fixed_tp: f64,
    pub vol_ensemble_weights_trend: Option<HashMap<String, f64>>,
    pub vol_ensemble_weights_range: Option<HashMap<String, f64>>,
    pub vol_ensemble_weights_neutral: Option<HashMap<String, f64>>,
    pub vol_horizon_bars: usize,
}

impl Default for RiskConfig {
    fn default() -> Self {
        // Config is the single source: `config.yaml`'s `risk.preset` key drives
        // the preset (serde fills it post-construction). The legacy
        // `NEOETHOS_PROP_FIRM_PRESET` env override was retired in v0.4.36 —
        // headless deployments set `risk.preset` in config.yaml instead. The
        // default (`PropFirmPreset::default()`) is unchanged from the prior
        // env-absent behaviour, so existing config.yaml / default users are
        // unaffected; only env-only deployments must move the preset to YAML.
        let preset = PropFirmPreset::default();
        let constraints = PropFirmConstraints::for_preset(preset);
        let runtime = PropFirmRuntimeDefaults::for_preset(preset);
        Self {
            preset,
            // Account starting balance is broker-specific. Operators
            // override this via `config.yaml`'s `risk.initial_balance`.
            initial_balance: 10_000.0,
            // Monthly profit floor (operator directive 2026-05-14)
            // tracks the active preset's published target.
            monthly_profit_target_pct: constraints.min_monthly_net_profit_pct as f64,
            min_risk_per_trade: 0.0,
            max_risk_per_trade: 0.030,
            risk_per_trade: 0.030,
            // Portfolio-level concurrent-risk cap: 0 = disabled (per-engine
            // sizing only). Opt-in via config/Advanced — never a silent
            // sizing change for existing users.
            max_portfolio_risk: 0.0,
            // Internal early stop sits 20% below the firm's published
            // daily-loss ceiling so a guard-rail trips before a real
            // breach. Operators override in YAML if their firm gives
            // tighter / looser tolerance.
            daily_drawdown_limit: runtime.daily_dd_stop_trading_pct,
            // Internal trailing total cap at 70% of the firm's
            // overall-drawdown ceiling for the same buffer reason.
            total_drawdown_limit: (constraints.max_overall_drawdown_pct as f64) * 0.7,
            min_risk_reward: 2.0,
            max_lot_size: runtime.max_lot_size,
            require_stop_loss: true,
            challenge_mode: false,
            challenge_phase: "phase_1".to_string(),
            // Disable the prop-firm gate entirely when the operator
            // selected `preset: none` — they're trading their own
            // money; we still respect per-trade risk limits but skip
            // the challenge accounting.
            prop_firm_rules: preset != PropFirmPreset::None,
            kill_zones_enabled: true,
            // Cap is preset-driven. FTMO defaults to 15; The5%ers is
            // tighter; "own money" raises it. Operators can override
            // via YAML when their style demands a different cap.
            max_trades_per_day: runtime.max_trades_per_day,
            recovery_mode_enabled: true,
            feature_drift_threshold: 0.30,
            high_quality_confidence: 0.65,
            atr_period: 14,
            atr_stop_multiplier: 1.5,
            triple_barrier_max_bars: 35,
            trailing_enabled: true,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            slippage_pips: 0.5,
            commission_per_lot: 7.0,
            backtest_spread_pips: 1.5,
            conformal_enabled: true,
            conformal_alpha: 0.10,
            conformal_abstain_min_set_size: 3,
            meta_label_tp_pips: None,
            meta_label_sl_pips: None,
            meta_label_max_hold_bars: 100,
            meta_label_min_dist: 0.0005,
            meta_label_fixed_sl: 0.0020,
            meta_label_fixed_tp: 0.0040,
            vol_ensemble_weights_trend: None,
            vol_ensemble_weights_range: None,
            vol_ensemble_weights_neutral: None,
            vol_horizon_bars: 5,
        }
    }
}

/// Models and training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub ml_models: Vec<String>,
    pub use_rl_agent: bool,
    pub use_sac_agent: bool,
    pub use_rllib_agent: bool,
    pub rllib_num_workers: usize,
    pub auto_enable_rllib: bool,
    pub use_neuroevolution: bool,
    pub rl_population_size: usize,
    pub rl_timesteps: usize,
    pub rl_eval_episodes: usize,
    pub rl_network_arch: Vec<usize>,
    pub rl_parallel_envs: usize,
    pub rl_state_bins: usize,
    pub rl_state_encoding: String,
    pub rl_update_interval: usize,
    pub rl_update_freq: usize,
    pub rl_learning_rate: f64,
    pub rl_gamma: f64,
    pub rl_epsilon_start: f64,
    pub rl_epsilon_end: f64,
    pub rl_epsilon_decay: f64,
    pub rl_buffer_capacity: usize,
    pub rl_reward_horizon: usize,
    pub rl_episode_len: usize,
    pub rl_train_seconds: u64,
    pub exit_agent_hidden_dim: usize,
    pub exit_agent_gamma: f64,
    pub exit_agent_epsilon: f64,
    pub exit_agent_epsilon_min: f64,
    pub exit_agent_epsilon_decay: f64,
    pub exit_agent_memory_capacity: usize,
    pub exit_agent_reward_horizon: usize,
    pub exit_agent_warmup_steps: usize,
    pub evo_train_seconds: u64,
    pub evo_hidden_size: usize,
    pub evo_population: usize,
    pub evo_islands: usize,
    pub evo_sigma: f64,
    pub prop_search_enabled: bool,
    pub prop_search_population: usize,
    pub prop_search_generations: usize,
    pub prop_search_max_hours: f64,
    pub prop_search_max_rows: usize,
    #[serde(serialize_with = "serialize_sorted_map")]
    pub prop_search_max_rows_by_tf: HashMap<String, usize>,
    pub prop_search_portfolio_size: usize,
    pub prop_search_max_indicators: usize,
    pub prop_search_checkpoint: PathBuf,
    pub prop_search_device: String,
    pub prop_search_train_years: usize,
    pub prop_search_val_years: usize,
    pub prop_search_val_candidates: usize,
    pub prop_search_val_min_positive_months: usize,
    pub prop_search_val_min_trades_per_month: usize,
    pub prop_search_val_min_trades_per_day: f64,
    pub prop_search_val_min_monthly_profit_pct: f64,
    pub prop_search_val_log_trades: bool,
    pub prop_search_val_trade_log_max: usize,
    pub prop_search_async: bool,
    pub prop_search_async_wait: bool,
    pub tree_device_preference: String,
    /// ML overfit-reduction (v0.5 ML-integration Stage 1). When `true`
    /// (default), the gradient boosters (xgboost/lightgbm/catboost + variants)
    /// train with regularized, bar-scaled defaults (shallower trees, column +
    /// row subsampling, L1/L2, leaf-size floors, bar-scaled tree counts) instead
    /// of the legacy full-depth / full-data / no-shrinkage defaults that
    /// memorize thin-TF (D1/W1/MN) targets. Set `false` to restore the legacy
    /// unregularized defaults for a controlled before/after OOS comparison.
    /// `#[serde(default)]` on `ModelsConfig` makes a missing key fall back to
    /// the `Default` impl below (= `true`).
    pub regularized_model_defaults: bool,
    /// ML overfit-reduction: minimum per-(symbol,TF) bar count below which the
    /// heavy gradient boosters are forced onto a shrunk preset (shallow depth,
    /// few trees, strong L2) and per-bar HPO is disabled (a thin holdout cannot
    /// select 5+ hyperparameters). Default 4000 — D1 (~2700 bars) and coarser
    /// TFs fall below it. Below an absolute floor (800) an even tinier preset is
    /// used. Set to 0 to disable the gate entirely.
    pub heavy_booster_min_bars: usize,
    /// ML overfit-reduction: when `true` (default) ML hyperparameter selection
    /// uses CombinatorialPurgedCV (purge+embargo, 15 paths) scored by
    /// mean-minus-stdev of the objective across folds — penalizing params that
    /// only generalize to one lucky window — instead of a single time-series
    /// holdout. Gated to `bars >= heavy_booster_min_bars` and `trials > 1` to
    /// bound the 15×-fold fit cost; below that it falls back to the single
    /// holdout. Set `false` to restore the single-holdout HPO.
    pub ml_cpcv_enabled: bool,
    pub prop_search_parent_selection: String,
    pub prop_search_survivor_selection: String,
    pub prop_search_survivor_fraction: f64,
    pub prop_search_immigrant_fraction: f64,
    pub prop_search_selection_temperature: f64,
    pub prop_search_tournament_size: usize,
    pub prop_search_opportunistic_enabled: bool,
    pub prop_search_opportunistic_min_positive_months: usize,
    pub prop_search_opportunistic_min_trades_per_month: usize,
    pub prop_search_opportunistic_min_trade_return_pct: f64,
    pub prop_search_opportunistic_max_dd: f64,
    pub prop_search_use_opportunistic: bool,
    /// 2026-05-26 operator directive (dual-mode product): correlation
    /// threshold for portfolio diversification (Pearson + Spearman both
    /// checked). Strategies with |correlation| ≥ this value against any
    /// portfolio member are rejected. Previously hardcoded 0.85 in
    /// `discovery.rs` — surfaced here so the operator can tune dedup
    /// aggressiveness from config / Settings UI without rebuilding.
    pub prop_search_corr_threshold: f64,
    /// Monte-Carlo perturbation runs per surviving candidate. The MC test
    /// re-evaluates each gene with random ±15-25% noise on thresholds,
    /// weights, and SL/TP and requires a configurable minimum to be
    /// profitable. Previously hardcoded 100 in discovery.rs.
    pub prop_search_mc_runs: u32,
    /// Minimum number of profitable MC runs required for a candidate to
    /// survive (out of `prop_search_mc_runs`). Previously hardcoded 70/100
    /// in discovery.rs (i.e. 70% threshold).
    pub prop_search_mc_min_profitable: u32,
    /// Spread (in pips) used in the sensitivity test — re-runs the
    /// candidate's backtest with a wider spread to verify the strategy
    /// stays profitable under degraded execution. Previously hardcoded
    /// 2.0 in discovery.rs.
    pub prop_search_sensitivity_spread_pips: f64,
    /// Commission per lot used in the sensitivity test. Previously
    /// hardcoded $7/lot in discovery.rs.
    pub prop_search_sensitivity_commission_per_lot: f64,
    pub train_batch_size: usize,
    pub inference_batch_size: usize,
    pub enable_transformer_expert: bool,
    pub transformer_heads: usize,
    pub transformer_layers: usize,
    pub transformer_hidden_dim: usize,
    pub transformer_dropout: f64,
    pub transformer_seq_len: usize,
    pub transformer_train_seconds: u64,
    pub nbeats_train_seconds: u64,
    pub tide_train_seconds: u64,
    pub tabnet_train_seconds: u64,
    pub kan_train_seconds: u64,
    pub mlp_train_seconds: u64,
    pub num_transformers: usize,
    pub swarm_memory_limit_mb: f64,
    pub swarm_horizon: usize,
    pub swarm_frequency: String,
    pub swarm_strategy: String,
    pub swarm_online_learning: bool,
    pub swarm_interpretability_needed: bool,
    pub swarm_latency_ms: usize,
    pub hpo_backend: String,
    pub hpo_trials: usize,
    #[serde(serialize_with = "serialize_sorted_map")]
    pub hpo_trials_by_model: HashMap<String, usize>,
    pub hpo_max_rows: usize,
    #[serde(serialize_with = "serialize_sorted_map")]
    pub max_epochs_by_model: HashMap<String, usize>,
    pub ray_tune_max_concurrency: usize,
    pub export_onnx: bool,
    pub calibration_enabled: bool,
    pub calibration_method: String,
    pub calibration_min_rows: usize,
    /// LIVE ML gate (Stage 3 blend in the live autopilot). When true, the
    /// live loop loads the symbol's soft-voting ensemble once at engine
    /// start and, on every closed bar, scales the per-trade risk by the
    /// ensemble's agreement × regime gate × anomaly scale (MlScale mode:
    /// the genes ALWAYS pick the direction; ML can only SHRINK size or
    /// skip a bar on a hard regime/anomaly collapse — never flip, never
    /// manufacture a trade). Default FALSE: live sizing must never change
    /// silently; the operator flips this knowingly. Fail-soft: any
    /// ensemble error on a bar falls back to gene-only sizing, loudly.
    pub live_ml_gate: bool,
    #[serde(serialize_with = "serialize_sorted_nested_map")]
    pub model_param_overrides: HashMap<String, HashMap<String, String>>,
    pub regime_router_enabled: bool,
    pub regime_router_min_models: usize,
    pub regime_trend_models: Vec<String>,
    pub regime_range_models: Vec<String>,
    pub regime_neutral_models: Vec<String>,
    pub l1_feature_selection_enabled: bool,
    pub l1_feature_selection_per_regime: bool,
    pub l1_feature_selection_min_features: usize,
    pub l1_feature_selection_max_features: usize,
    pub l1_feature_selection_sample_limit: usize,
    pub l1_feature_selection_c: f64,
    pub filter_to_base_signal: bool,
    pub global_max_rows: usize,
    pub global_max_rows_per_symbol: usize,
    pub symbol_hash_buckets: usize,
    pub global_train_ratio: f64,
    pub train_holdout_pct: f64,
    pub label_use_triple_barrier: bool,
    pub label_horizon_bars: usize,
    pub label_neutral_band_atr_fraction: f64,
    pub label_stop_atr_multiplier: f64,
    pub label_take_profit_rr: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    /// Discovery search regime: `"prop_firm"` (default — permissive
    /// quality floors so the prop-firm gauntlet does the heavy lifting)
    /// or `"strict"` (full FilteringConfig floors). Was the env-only
    /// `NEOETHOS_BOT_DISCOVERY_MODE`; now a first-class config knob the
    /// operator sets from the UI / TUI — never the environment.
    pub discovery_mode: String,
    /// agent 2026-06-05 overfitting fix: when `true` (default), a discovered
    /// portfolio is only export-ready in PropFirm mode if it ALSO passes the
    /// walk-forward gate (not just the prop-firm window gate). Previously the
    /// walk-forward result was purely informational in PropFirm mode, so
    /// overfit strategies (in-sample Sharpe 3-11 / PF up to 62) that failed
    /// out-of-sample still exported. Set `false` to restore the old behaviour
    /// (prop-firm-window gate only). `#[serde(default)]` on `ModelsConfig`
    /// makes a missing key fall back to the `Default` impl below (= `true`).
    pub require_walkforward_for_export: bool,
    /// Hard floor for the prop-firm window-pass rate, applied on top of
    /// `discovery_runtime.prop_firm_gate.pass_rate` (effective floor = max of the
    /// two). RE-CALIBRATED 2026-06-06 from 0.65 → **0.40** when the per-window
    /// profit target was set to the operator's bar (8%/60-day window = >=4%/month,
    /// in `derive_prop_firm_gate`). 0.40 = a candidate must hit >=4%/month in at
    /// least 40% of the random 60-day windows to survive — a genuine persistent
    /// edge, with the live models lifting the rest (discovery=edge, models=grow).
    /// The base-filter max-DD + walk-forward export gate still reject blow-ups /
    /// overfit. Raise toward 0.65 for stricter selection; lower for more candidates.
    pub prop_firm_min_pass_rate: f64,
    /// Genetic-search runtime knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_*` search env vars). See [`SearchRuntimeConfig`].
    pub search_runtime: SearchRuntimeConfig,
    /// Discovery-pipeline runtime knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_PREFILTER_*` / `NEOETHOS_BOT_FUNNEL_STAGE1_*` /
    /// `NEOETHOS_BOT_MIN_HISTORY_YEARS` / `NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS`
    /// env vars). See [`DiscoveryRuntimeConfig`].
    pub discovery_runtime: DiscoveryRuntimeConfig,
    /// Strategy-evaluation runtime knobs (config-driven replacement for
    /// the `NEOETHOS_BOT_PROP_*` cost + SMC-weight env vars). See
    /// [`EvalRuntimeConfig`].
    pub eval_runtime: EvalRuntimeConfig,
    /// Strategy-quality scoring knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_PROP_*` monthly-quality env vars). See
    /// [`QualityRuntimeConfig`].
    pub quality_runtime: QualityRuntimeConfig,
    /// Backtest-evaluation runtime knobs (config-driven replacement for
    /// the `NEOETHOS_BOT_BACKTEST_*` env vars). See [`BacktestRuntimeConfig`].
    pub backtest_runtime: BacktestRuntimeConfig,
    /// Seen-signature dedup-memory knobs (config-driven replacement for
    /// the `NEOETHOS_BOT_PROP_SEEN_*` env vars). See
    /// [`SeenSignatureRuntimeConfig`].
    pub seen_signature_runtime: SeenSignatureRuntimeConfig,
    /// Search-memory + weekly-refresh ledger knobs (2026-06-06): persist what
    /// each discovery run found and seed the next run's seen-set so weekly runs
    /// add NEW strategies instead of re-discovering old ones. See
    /// [`DiscoveryLedgerConfig`].
    pub discovery_ledger: DiscoveryLedgerConfig,
    /// SMC search-injection knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_PROP_SMC_*` env vars). See [`SmcSearchRuntimeConfig`].
    pub smc_search_runtime: SmcSearchRuntimeConfig,
    /// Data-layer behavior knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_NORMALIZE_FEATURES` / `..._REBUILD_STALE_HIGHER_TFS`
    /// env vars). See [`DataRuntimeConfig`].
    pub data_runtime: DataRuntimeConfig,
    /// Tree-model training knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_EARLY_STOP_*` env vars). See [`TreeRuntimeConfig`].
    pub tree_runtime: TreeRuntimeConfig,
    pub prop_metric_weight: f64,
    pub prop_accuracy_weight: f64,
    pub prop_min_trades: usize,
    pub prop_conf_threshold: f64,
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub cpcv_max_rows: usize,
    pub enable_ddp: bool,
    pub enable_fsdp: bool,
    pub ddp_world_size: usize,
    pub transformer_d_model: usize,
    pub transformer_n_heads: usize,
    pub transformer_n_layers: usize,
    pub nf_hidden_dim: usize,
    pub tide_hidden_dim: usize,
    pub nbeats_hidden_dim: usize,
    pub kan_hidden_dim: usize,
    pub kan_grid_size: usize,
    pub tabnet_hidden_dim: usize,
    pub phase5_filter_meta_blender: bool,
    pub phase5_core_models: Vec<String>,
}

/// Genetic-search runtime knobs — the config-driven replacement for the
/// `NEOETHOS_BOT_*` genetic-search env vars (RNG seed, novelty weighting,
/// tournament / archive sizing, SMC-gate curve, archive scoring, selection
/// policy). Mirrors `neoethos_search::genetic::GeneticSearchRuntimeOverrides`,
/// which the search crate now builds via `from_settings(&Settings)` so the
/// operator sets these from config / UI / TUI — never the environment.
///
/// Defaults here MUST match that override struct's `Default`; a
/// `from_settings(&Settings::default()) == default()` unit test in
/// `neoethos-search` enforces it. Empty strings on the policy / archive-mode
/// fields mean "use the engine default" (so the config default need not
/// duplicate the parser vocabulary).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SearchRuntimeConfig {
    pub seed: Option<u64>,
    pub novelty_weight: f64,
    pub stagnation_patience: usize,
    pub tournament_size_override: Option<usize>,
    pub archive_cap_override: Option<usize>,
    pub seen_retry_attempts: usize,
    pub smc_gate_start: f32,
    pub smc_gate_end: f32,
    pub smc_gate_curve: f32,
    pub smc_gate_stagnation_step: f32,
    pub disable_smc_gate: bool,
    pub archive_mode: String,
    pub archive_min_net: f64,
    pub archive_min_pf: f64,
    pub archive_min_sharpe: f64,
    pub parent_selection: String,
    pub survivor_selection: String,
    pub immigrant_ratio: f64,
    pub survivor_fraction: f64,
    pub selection_temperature: f64,
    /// Generations of no meaningful improvement before the GA hard
    /// early-stops THIS combo and returns its archive, freeing the
    /// wall-clock budget for the next symbol×timeframe. `0` disables the
    /// early-stop (run to the time / generation cap as before). This is a
    /// SEPARATE, larger threshold than `stagnation_patience`: the soft
    /// diversity kick (gate relaxation + immigrants + hypermutation) is
    /// attempted first; the hard stop fires only if the search is STILL
    /// flat after `convergence_patience` generations.
    pub convergence_patience: usize,
    /// Minimum increase in top fitness counted as "improvement" when
    /// tracking stagnation; a generation gaining less than this is
    /// stagnant. Replaces the legacy hard-coded `1e-12`.
    pub min_improvement: f64,
    /// Wall-clock floor for the convergence early-stop, as a fraction of
    /// the per-combo time budget (`prop_search_max_hours`). The early-stop
    /// (see `convergence_patience`) may fire ONLY after this fraction of
    /// the budget has elapsed. This makes the early-stop throughput-robust:
    /// generation rate varies ~300× across timeframes, so a pure
    /// generation count (e.g. 250 gens ≈ 1 s on a fast TF, ≈ 21 min on M1)
    /// would otherwise kill fast timeframes before they ever search. `0.5`
    /// = every combo gets at least half its budget; `0` = no floor (pure
    /// generation count, NOT recommended); `1.0` = effectively disables the
    /// early-stop (only the time cap stops the combo).
    pub convergence_min_elapsed_fraction: f64,
}

impl Default for SearchRuntimeConfig {
    fn default() -> Self {
        Self {
            seed: None,
            novelty_weight: 0.0,
            stagnation_patience: 2,
            tournament_size_override: None,
            archive_cap_override: None,
            seen_retry_attempts: 16,
            smc_gate_start: 0.75,
            smc_gate_end: 0.35,
            smc_gate_curve: 1.0,
            smc_gate_stagnation_step: 0.03,
            disable_smc_gate: false,
            archive_mode: String::new(),
            archive_min_net: 0.0,
            archive_min_pf: 1.0,
            archive_min_sharpe: 0.0,
            parent_selection: String::new(),
            survivor_selection: String::new(),
            immigrant_ratio: 0.25,
            survivor_fraction: 0.10,
            selection_temperature: 0.75,
            convergence_patience: 250,
            min_improvement: 1e-12,
            convergence_min_elapsed_fraction: 0.5,
        }
    }
}

/// Discovery-pipeline runtime knobs — the config-driven replacement for the
/// legacy `NEOETHOS_BOT_PREFILTER_TOP_K`, `NEOETHOS_BOT_PREFILTER_INSAMPLE`,
/// `NEOETHOS_BOT_FUNNEL_STAGE1_PCT`, `NEOETHOS_BOT_FUNNEL_STAGE1_WINDOW`,
/// `NEOETHOS_BOT_MIN_HISTORY_YEARS`, and `NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS`
/// env vars. Consumed by `DiscoveryConfig::from_settings` (via
/// `DiscoveryRuntimeOverrides::from_settings`) — the operator sets these from
/// the UI / TUI, never the environment. Defaults reproduce the previous
/// env-absent behaviour exactly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DiscoveryRuntimeConfig {
    /// Max features kept after the in-sample correlation prefilter; `0`
    /// disables the prefilter. (was `NEOETHOS_BOT_PREFILTER_TOP_K`)
    pub prefilter_top_k: usize,
    /// Fraction of rows treated as in-sample when ranking features; must be
    /// in `(0, 1]`. (was `NEOETHOS_BOT_PREFILTER_INSAMPLE`)
    pub prefilter_insample_frac: f64,
    /// Minimum number of features to force-keep from EACH present higher
    /// timeframe group during the prefilter, in addition to the global
    /// `prefilter_top_k`. The prefilter ranks by correlation with the BASE
    /// timeframe's 1-bar forward return; a slow higher-TF indicator is
    /// near-constant across many base bars so that correlation is ~0 by
    /// construction, and the global top-K therefore discards EVERY multi-TF
    /// feature — wasting the entire multi-resolution cube and starving the
    /// GA's multi-TF seed templates. This quota guarantees each higher TF
    /// (`H1_`, `H4_`, `M15_`, …) reaches the GA. `0` reproduces the legacy
    /// base-only behaviour. (new 2026-06-08)
    pub prefilter_min_per_timeframe: usize,
    /// Fraction of rows fed to the multi-stage funnel's first stage; clamped
    /// to `[0.01, 1.0]`. (was `NEOETHOS_BOT_FUNNEL_STAGE1_PCT`)
    pub funnel_stage1_pct: f64,
    /// Where to slice the stage-1 fast-eval rows: `"earliest"` (default,
    /// OOS-safe), `"latest"`, or `"random"`. (was
    /// `NEOETHOS_BOT_FUNNEL_STAGE1_WINDOW`)
    pub stage1_window: String,
    /// Minimum historical-data window (years) discovery requires before it
    /// runs; `0` skips the pre-flight check. (was
    /// `NEOETHOS_BOT_MIN_HISTORY_YEARS`)
    pub min_history_years: u32,
    /// Opt-in: derive a per-dataset adaptive coarse-threshold ladder from the
    /// feature cube. Experimental — the install is process-global (OnceLock),
    /// so leave off for multi-symbol sweeps until per-symbol install lands
    /// (F-277b). (was `NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS`)
    pub adaptive_thresholds: bool,
    /// Prop-firm window-pass gate parameters (FTMO baseline + overrides).
    /// See [`PropFirmGateConfig`]. (was the
    /// `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_*` env overrides)
    pub prop_firm_gate: PropFirmGateConfig,
}

impl Default for DiscoveryRuntimeConfig {
    fn default() -> Self {
        Self {
            prefilter_top_k: 50,
            prefilter_insample_frac: 0.80,
            prefilter_min_per_timeframe: 6,
            funnel_stage1_pct: 0.25,
            stage1_window: "earliest".to_string(),
            min_history_years: 0,
            adaptive_thresholds: false,
            prop_firm_gate: PropFirmGateConfig::default(),
        }
    }
}

/// Prop-firm window-pass gate parameters — the config-driven replacement for
/// the `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_*` env overrides read by
/// `derive_prop_firm_gate`. The `Option` rule fields default to `None`,
/// meaning "use the FTMO baseline" (`PropFirmRiskRules::default` /
/// `FTMO_STANDARD`) — exactly reproducing the env-absent behaviour; set a
/// value to override that specific rule (e.g. to target a non-FTMO firm's
/// challenge from the UI / TUI).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PropFirmGateConfig {
    /// Max daily-loss fraction (e.g. `0.05` = 5%). `None` = FTMO baseline.
    /// (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_MAX_DAILY_LOSS_PCT`)
    pub max_daily_loss_pct: Option<f64>,
    /// Max overall-drawdown fraction (e.g. `0.10` = 10%). `None` = FTMO
    /// baseline. (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_MAX_DD_PCT`)
    pub max_overall_drawdown_pct: Option<f64>,
    /// Challenge profit-target fraction (e.g. `0.10` = 10%); `0` disables the
    /// target requirement. `None` = `FTMO_STANDARD` target. (was
    /// `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_PROFIT_TARGET_PCT`)
    pub profit_target_pct: Option<f64>,
    /// Minimum trading days the strategy must be active. `None` = FTMO
    /// baseline. (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_MIN_TRADING_DAYS`)
    pub min_trading_days: Option<usize>,
    /// Length (days) of each random evaluation window. Default `60` (the
    /// longest standard prop-firm phase). (was
    /// `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_WINDOW_DAYS`)
    pub window_days: usize,
    /// Number of random windows to score. `0` = auto-tune from dataset
    /// length. (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_N_WINDOWS`)
    pub n_windows: usize,
    /// Hard pass-rate floor in `[0, 1]`. `0` = ranking-only (no hard
    /// threshold). (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_PASS_RATE`)
    pub pass_rate: f64,
}

impl Default for PropFirmGateConfig {
    fn default() -> Self {
        Self {
            max_daily_loss_pct: None,
            max_overall_drawdown_pct: None,
            profit_target_pct: None,
            min_trading_days: None,
            window_days: 60,
            n_windows: 0,
            pass_rate: 0.0,
        }
    }
}

/// Strategy-evaluation runtime knobs — the config-driven replacement for
/// the `NEOETHOS_BOT_PROP_*` cost-profile + SMC-weight env vars (symbol /
/// currency / pip-value / spread / commission overrides used by
/// `infer_market_cost_profile`, and the 12 SMC indicator weights +
/// gate threshold used by `EvaluationConfig::default`). Mirrors
/// `neoethos_search::genetic::StrategyEvaluationRuntimeOverrides`; the
/// search crate builds it via `from_settings(&Settings)`. Defaults MUST
/// match that struct's `Default` (a `from_settings(&Settings::default())
/// == default()` test enforces it). `None` cost fields mean "no
/// override" (production callers pass explicit values).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EvalRuntimeConfig {
    pub symbol: Option<String>,
    pub account_currency: Option<String>,
    pub pip_value: Option<f64>,
    pub quote_to_account_rate: Option<f64>,
    pub pip_value_per_lot: Option<f64>,
    pub spread_pips: Option<f64>,
    pub commission_per_trade: Option<f64>,
    pub reject_pip_fallback: bool,
    pub smc_gate_threshold: f32,
    pub smc_w_ob: f32,
    pub smc_w_fvg: f32,
    pub smc_w_liq: f32,
    pub smc_w_mtf: f32,
    pub smc_w_premium: f32,
    pub smc_w_inducement: f32,
    pub smc_w_bos: f32,
    pub smc_w_choch: f32,
    pub smc_w_eqh: f32,
    pub smc_w_eql: f32,
    pub smc_w_displacement: f32,
}

impl Default for EvalRuntimeConfig {
    fn default() -> Self {
        Self {
            symbol: None,
            account_currency: None,
            pip_value: None,
            quote_to_account_rate: None,
            pip_value_per_lot: None,
            spread_pips: None,
            commission_per_trade: None,
            reject_pip_fallback: false,
            smc_gate_threshold: 0.75,
            smc_w_ob: 1.0,
            smc_w_fvg: 1.0,
            smc_w_liq: 1.0,
            smc_w_mtf: 1.0,
            smc_w_premium: 1.0,
            smc_w_inducement: 1.0,
            smc_w_bos: 1.0,
            smc_w_choch: 1.0,
            smc_w_eqh: 1.0,
            smc_w_eql: 1.0,
            smc_w_displacement: 1.0,
        }
    }
}

/// Strategy-quality scoring knobs — config-driven replacement for the
/// `NEOETHOS_BOT_PROP_MIN_TRADES_PER_MONTH` /
/// `NEOETHOS_BOT_TRADING_DAYS_PER_MONTH` env vars. Mirrors
/// `neoethos_search::quality::QualityRuntimeOverrides`; a
/// `from_settings(&Settings::default()) == default()` test enforces the
/// matching defaults.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct QualityRuntimeConfig {
    /// Minimum trades a calendar month needs to count toward monthly
    /// win-rate / avg-return scoring.
    pub min_trades_per_month: usize,
    /// Trading days per month used to convert observed trading days into
    /// a months-traded estimate.
    pub trading_days_per_month: f64,
}

impl Default for QualityRuntimeConfig {
    fn default() -> Self {
        Self {
            min_trades_per_month: 4,
            trading_days_per_month: 21.0,
        }
    }
}

/// Backtest-evaluation runtime knobs — config-driven replacement for the
/// `NEOETHOS_BOT_BACKTEST_*` + `NEOETHOS_BOT_RUST_THREADS` env vars.
/// Mirrors `neoethos_search::eval::BacktestRuntimeOverrides`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BacktestRuntimeConfig {
    /// Starting equity for canonical backtest PnL accounting (> 0).
    pub initial_equity: f64,
    /// Max monthly PnL buckets retained for consistency math (> 0).
    pub month_capacity: usize,
    /// Explicit rayon thread-pool size. `None` → one worker per logical
    /// core (rayon default).
    pub rayon_threads: Option<usize>,
}

impl Default for BacktestRuntimeConfig {
    fn default() -> Self {
        Self {
            initial_equity: 100_000.0,
            month_capacity: 240,
            rayon_threads: None,
        }
    }
}

/// Seen-signature memory knobs — config-driven replacement for the
/// `NEOETHOS_BOT_PROP_SEEN_*` env vars (dedup-memory flush cadence,
/// load/entry caps, and on-disk path). Mirrors
/// `neoethos_search::genetic::SeenSignatureMemoryRuntimeOverrides`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SeenSignatureRuntimeConfig {
    pub flush_every: usize,
    pub load_max: usize,
    /// `0` → unbounded (`usize::MAX`); otherwise the entry cap.
    pub max_entries: usize,
    /// Optional on-disk seen-signature file. Empty / unset → in-memory only.
    pub file_path: Option<String>,
}

impl Default for SeenSignatureRuntimeConfig {
    fn default() -> Self {
        Self {
            flush_every: 4096,
            load_max: 3_000_000,
            max_entries: 3_000_000,
            file_path: None,
        }
    }
}

/// Search-memory + weekly-refresh knobs (2026-06-06). When `enabled`, each
/// discovery run reads a per-symbol/TF on-disk **ledger** of previously found
/// strategies (indicator + SMC-flag combos + fitness) and seeds the GA's
/// seen-signature memory with their hashes so the next run AVOIDS
/// re-discovering them — every weekly run ADDS new diverse strategies to a
/// growing library. Mirrors the nested-config pattern of
/// [`DiscoveryRuntimeConfig`]; consumed via
/// `neoethos_search::DiscoveryConfig::from_settings`.
///
/// Cross-run dedup of the seeded hashes only takes effect for the GA when an
/// on-disk seen-signature file is configured (`seen_signature_runtime.file_path`):
/// the genetic engine builds its own `SeenSignatureMemory::from_env()` and reads
/// previously-persisted hashes from that file. When `file_path` is unset
/// (in-memory only, the default), the ledger is still recorded + the seed step
/// runs, but the seeded hashes are not visible to the engine's fresh in-memory
/// set — set a `file_path` to get true cross-run dedup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DiscoveryLedgerConfig {
    /// Master switch. When `false`, discovery behaves byte-identically to a
    /// build without this feature (no ledger read, no seed, no ledger write).
    pub enabled: bool,
    /// Directory the per-symbol/TF ledger JSON files live in. Relative paths
    /// resolve against the process CWD (same convention as `cache/features`).
    pub cache_dir: String,
    /// How many top archive (non-portfolio) genes to also record per run, so
    /// the seen-set grows beyond just the promoted portfolio.
    pub archive_top_n: usize,
    /// Promotion policy for `discovery-promote-weekly`. `"additive"` (the
    /// default + only implemented policy) merges new genes by hash and keeps
    /// existing ones; unknown values fall back to additive.
    pub promotion_policy: String,
}

impl Default for DiscoveryLedgerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_dir: "cache/search".to_string(),
            archive_top_n: 20,
            promotion_policy: "additive".to_string(),
        }
    }
}

/// SMC (smart-money-concept) search-injection knobs — config-driven
/// replacement for the `NEOETHOS_BOT_PROP_SMC_*` env vars (the per-flag
/// enable probabilities, the force-ratio + min-flags that seed each GA
/// generation with SMC-aware genes, and the master `force_enabled`
/// toggle). Mirrors `neoethos_search::genetic::SmcSearchConfig`
/// (probabilities are clamped to `[0,1]`; `force_enabled = false` zeroes
/// `force_ratio` + `min_flags`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SmcSearchRuntimeConfig {
    pub force_ratio: f64,
    pub min_flags: usize,
    /// Master toggle — `false` disables SMC forcing (zeroes force_ratio +
    /// min_flags). Was `NEOETHOS_BOT_PROP_SMC_FORCE_ENABLED`.
    pub force_enabled: bool,
    pub p_ob: f64,
    pub p_fvg: f64,
    pub p_liq: f64,
    pub p_premium: f64,
    pub p_inducement: f64,
    pub p_mtf: f64,
    pub p_bos: f64,
    pub p_choch: f64,
    pub p_eqh: f64,
    pub p_eql: f64,
    pub p_displacement: f64,
}

impl Default for SmcSearchRuntimeConfig {
    fn default() -> Self {
        Self {
            force_ratio: 0.30,
            min_flags: 1,
            force_enabled: true,
            p_ob: 0.50,
            p_fvg: 0.50,
            p_liq: 0.50,
            p_premium: 0.50,
            p_inducement: 0.50,
            p_mtf: 0.85,
            p_bos: 0.50,
            p_choch: 0.50,
            p_eqh: 0.50,
            p_eql: 0.50,
            p_displacement: 0.50,
        }
    }
}

/// Data-layer behavior knobs — config-driven replacement for the
/// `NEOETHOS_BOT_NORMALIZE_FEATURES` / `NEOETHOS_BOT_REBUILD_STALE_HIGHER_TFS`
/// env vars. Both default OFF (opt-in). Consumed by the data crate via
/// `neoethos_data::install_data_runtime_overrides(...)` at startup.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DataRuntimeConfig {
    /// Per-column robust z-score normalization of the feature matrix
    /// before the GA search (was `NEOETHOS_BOT_NORMALIZE_FEATURES`).
    /// OFF by default — enabling without re-calibrating GA thresholds
    /// changes discovery for symbols that currently work.
    pub normalize_features: bool,
    /// Auto-rebuild a present-but-stale higher timeframe from the base
    /// instead of NaN-ing the stale tail (was
    /// `NEOETHOS_BOT_REBUILD_STALE_HIGHER_TFS`). OFF by default.
    pub rebuild_stale_higher_tfs: bool,
}

impl Default for DataRuntimeConfig {
    fn default() -> Self {
        Self {
            normalize_features: false,
            rebuild_stale_higher_tfs: false,
        }
    }
}

/// Tree-model (LightGBM / XGBoost / CatBoost) device + training knobs —
/// config-driven replacement for the `NEOETHOS_BOT_TREE_DEVICE` / `_GPU_ONLY`
/// / `_EARLY_STOP_*` env vars and the `FOREX_GPU_COUNT` rebrand remnant.
/// Platform-standard GPU-selection knobs (`CUDA_VISIBLE_DEVICES`, …) are NOT
/// app config and stay honored. (The cross-cutting `cpu_threads` budget — read
/// in core/search/models, so it needs a single system-level knob — is a
/// separate follow-up.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TreeRuntimeConfig {
    /// Device preference for tree-model training: `"auto"` | `"cpu"` |
    /// `"gpu"` | `"cuda"` | `"cuda:N"`. `""` is treated as `"auto"`. Was
    /// `NEOETHOS_BOT_TREE_DEVICE` (the per-model `_{MODEL}_DEVICE` overrides
    /// are folded into this single global knob).
    pub device: String,
    /// Require GPU for tree training — no silent CPU fallback. Was
    /// `NEOETHOS_BOT_GPU_ONLY`.
    pub gpu_only: bool,
    /// Explicit GPU count; `None` = auto-detect (the standard
    /// `*_VISIBLE_DEVICES` vars, then `nvidia-smi` / `rocm`). Was the
    /// `FOREX_GPU_COUNT` rebrand remnant.
    pub gpu_count: Option<usize>,
    /// Early-stop patience override for tree-model training; `None` (the
    /// default) = use each model's built-in default. Was
    /// `NEOETHOS_BOT_EARLY_STOP_PATIENCE`.
    pub early_stop_patience: Option<usize>,
    /// Early-stop min-delta override; `None` = use the model's default.
    /// Was `NEOETHOS_BOT_EARLY_STOP_MIN_DELTA`.
    pub early_stop_min_delta: Option<f64>,
}

impl Default for TreeRuntimeConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            gpu_only: false,
            gpu_count: None,
            early_stop_patience: None,
            early_stop_min_delta: None,
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        let mut hpo_trials_by_model = HashMap::new();
        for (model, trials) in [
            ("lightgbm", 8),
            ("xgboost", 8),
            ("xgboost_rf", 6),
            ("xgboost_dart", 6),
            ("catboost", 8),
            ("catboost_alt", 6),
            ("mlp", 6),
            ("tabnet", 6),
            ("nbeats", 6),
            ("tide", 6),
            ("kan", 6),
            ("transformer", 6),
        ] {
            hpo_trials_by_model.insert(model.to_string(), trials);
        }

        Self {
            ml_models: vec![
                "lightgbm",
                "xgboost",
                "xgboost_rf",
                "xgboost_dart",
                "catboost",
                "catboost_alt",
                "sklears_tree",
                "mlp",
                "elasticnet",
                // Plain L2 logistic regression — trainable since day one but
                // absent from every request list, so it never existed on any
                // install (operator audit 2026-07-11). Cheap linear voter.
                "logistic",
                "bayes_logit",
                "online_pa",
                "online_hoeffding",
                "swarm_forecaster",
                "isolation_forest",
                "neat",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            use_rl_agent: true,
            use_sac_agent: true,
            use_rllib_agent: false,
            rllib_num_workers: 0,
            auto_enable_rllib: true,
            use_neuroevolution: true,
            rl_population_size: 5,
            rl_timesteps: 10_000_000,
            rl_eval_episodes: 15,
            rl_network_arch: vec![4096, 4096, 4096, 2048, 1024],
            rl_parallel_envs: 1,
            rl_state_bins: 255,
            rl_state_encoding: "normalized".to_string(),
            rl_update_interval: 0,
            rl_update_freq: 0,
            rl_learning_rate: 1e-3,
            rl_gamma: 0.99,
            rl_epsilon_start: 1.0,
            rl_epsilon_end: 0.02,
            rl_epsilon_decay: 0.995,
            rl_buffer_capacity: 0,
            rl_reward_horizon: 0,
            rl_episode_len: 0,
            rl_train_seconds: 3600,
            exit_agent_hidden_dim: 64,
            exit_agent_gamma: 0.99,
            exit_agent_epsilon: 0.20,
            exit_agent_epsilon_min: 0.05,
            exit_agent_epsilon_decay: 0.999,
            exit_agent_memory_capacity: 10_000,
            exit_agent_reward_horizon: 0,
            exit_agent_warmup_steps: 0,
            evo_train_seconds: 3600,
            evo_hidden_size: 64,
            evo_population: 32,
            evo_islands: 4,
            evo_sigma: 0.25,
            prop_search_enabled: false,
            prop_search_population: 100,
            prop_search_generations: 50,
            prop_search_max_hours: 0.5, // 2026-06-05: sane default (was 8.0=absurd 8h/combo); config-overridable (VPS budget run uses 0.25)
            prop_search_max_rows: 0,
            prop_search_max_rows_by_tf: HashMap::new(),
            prop_search_portfolio_size: 3000,
            prop_search_max_indicators: 12,
            prop_search_checkpoint: PathBuf::from("models/strategy_evo_checkpoint.json"),
            prop_search_device: "cpu".to_string(),
            prop_search_train_years: 0,
            prop_search_val_years: 0,
            prop_search_val_candidates: 0,
            prop_search_val_min_positive_months: 0,
            prop_search_val_min_trades_per_month: 0,
            prop_search_val_min_trades_per_day: 0.0,
            prop_search_val_min_monthly_profit_pct: 0.0,
            prop_search_val_log_trades: false,
            prop_search_val_trade_log_max: 20,
            prop_search_async: false,
            prop_search_async_wait: false,
            tree_device_preference: "auto".to_string(),
            regularized_model_defaults: true,
            heavy_booster_min_bars: 4000,
            ml_cpcv_enabled: true,
            prop_search_parent_selection: "rank".to_string(),
            prop_search_survivor_selection: "rank".to_string(),
            prop_search_survivor_fraction: 0.10,
            prop_search_immigrant_fraction: 0.18,
            prop_search_selection_temperature: 0.75,
            prop_search_tournament_size: 0,
            prop_search_opportunistic_enabled: true,
            prop_search_opportunistic_min_positive_months: 3,
            prop_search_opportunistic_min_trades_per_month: 10,
            prop_search_opportunistic_min_trade_return_pct: 4.0,
            prop_search_opportunistic_max_dd: 0.025,
            prop_search_use_opportunistic: true,
            // 2026-05-26 operator directive (dual-mode product): the 5 knobs
            // below were previously hardcoded in discovery.rs. Surfaced here
            // so the dual-mode product can tune them without rebuilds. The
            // defaults reproduce the previous hardcoded behavior.
            prop_search_corr_threshold: 0.85,
            prop_search_mc_runs: 100,
            prop_search_mc_min_profitable: 70,
            prop_search_sensitivity_spread_pips: 2.0,
            prop_search_sensitivity_commission_per_lot: 7.0,
            train_batch_size: 32,
            inference_batch_size: 32,
            enable_transformer_expert: true,
            transformer_heads: 8,
            transformer_layers: 4,
            transformer_hidden_dim: 256,
            transformer_dropout: 0.20,
            transformer_seq_len: 64,
            transformer_train_seconds: 3600,
            nbeats_train_seconds: 3600,
            tide_train_seconds: 3600,
            tabnet_train_seconds: 3600,
            kan_train_seconds: 3600,
            mlp_train_seconds: 3600,
            num_transformers: 2,
            swarm_memory_limit_mb: 256.0,
            swarm_horizon: 0,
            swarm_frequency: "H".to_string(),
            swarm_strategy: "bayesian".to_string(),
            swarm_online_learning: true,
            swarm_interpretability_needed: true,
            swarm_latency_ms: 0,
            hpo_backend: "ax".to_string(),
            hpo_trials: 8,
            hpo_trials_by_model,
            hpo_max_rows: 1_000_000,
            max_epochs_by_model: HashMap::new(),
            ray_tune_max_concurrency: 1,
            export_onnx: false,
            calibration_enabled: true,
            calibration_method: "platt".to_string(),
            calibration_min_rows: 300,
            live_ml_gate: false,
            model_param_overrides: HashMap::new(),
            regime_router_enabled: false,
            regime_router_min_models: 2,
            regime_trend_models: vec![
                "transformer",
                "patchtst",
                "timesnet",
                "nbeats",
                "nbeatsx_nf",
                "tide",
                "tide_nf",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            regime_range_models: vec![
                "tabnet",
                "lightgbm",
                "xgboost",
                "xgboost_rf",
                "xgboost_dart",
                "catboost",
                "catboost_alt",
                "elasticnet",
                "bayes_logit",
                "online_pa",
                "online_hoeffding",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            regime_neutral_models: Vec::new(),
            l1_feature_selection_enabled: false,
            l1_feature_selection_per_regime: false,
            l1_feature_selection_min_features: 20,
            l1_feature_selection_max_features: 256,
            l1_feature_selection_sample_limit: 200_000,
            l1_feature_selection_c: 0.20,
            filter_to_base_signal: true,
            global_max_rows: 0,
            global_max_rows_per_symbol: 0,
            symbol_hash_buckets: 32,
            global_train_ratio: 0.8,
            train_holdout_pct: 0.2,
            label_use_triple_barrier: true,
            label_horizon_bars: 0,
            label_neutral_band_atr_fraction: 0.25,
            label_stop_atr_multiplier: 0.0,
            label_take_profit_rr: 0.0,
            walkforward_splits: 10, // 2026-06-05: robust OOS default (was 20, slow); config-overridable
            embargo_minutes: 120,
            discovery_mode: "prop_firm".to_string(),
            // walk-forward export gate ON (robustness). prop-firm pass-rate floor
            // RE-CALIBRATED 0.65→0.40 (2026-06-06) to match the operator's >=4%/month
            // bar now used as the per-window target — see derive_prop_firm_gate +
            // config.yaml prop_firm_min_pass_rate. (see field docs above.)
            require_walkforward_for_export: true,
            prop_firm_min_pass_rate: 0.40,
            search_runtime: SearchRuntimeConfig::default(),
            discovery_runtime: DiscoveryRuntimeConfig::default(),
            eval_runtime: EvalRuntimeConfig::default(),
            quality_runtime: QualityRuntimeConfig::default(),
            backtest_runtime: BacktestRuntimeConfig::default(),
            seen_signature_runtime: SeenSignatureRuntimeConfig::default(),
            discovery_ledger: DiscoveryLedgerConfig::default(),
            smc_search_runtime: SmcSearchRuntimeConfig::default(),
            data_runtime: DataRuntimeConfig::default(),
            tree_runtime: TreeRuntimeConfig::default(),
            prop_metric_weight: 1.0,
            prop_accuracy_weight: 0.1,
            prop_min_trades: 0,
            prop_conf_threshold: 0.55,
            enable_cpcv: true,
            cpcv_n_splits: 5,
            cpcv_n_test_groups: 2,
            cpcv_embargo_pct: 0.01,
            cpcv_purge_pct: 0.02,
            cpcv_min_phi: 0.80,
            cpcv_max_rows: 200000, // 2026-06-05: cap informational CPCV (was 0=full=heavy on full-data); config-overridable
            enable_ddp: false,
            enable_fsdp: false,
            ddp_world_size: 1,
            transformer_d_model: 256,
            transformer_n_heads: 8,
            transformer_n_layers: 4,
            nf_hidden_dim: 256,
            tide_hidden_dim: 256,
            nbeats_hidden_dim: 256,
            kan_hidden_dim: 256,
            kan_grid_size: 9,
            tabnet_hidden_dim: 64,
            phase5_filter_meta_blender: true,
            phase5_core_models: vec!["transformer", "nbeats", "tide", "tabnet", "kan"]
                .into_iter()
                .map(String::from)
                .collect(),
        }
    }
}

/// How the trading gate should treat high-impact news events.
///
/// Until #117 the only option was auto-pause; the runtime would block
/// new orders inside the kill window. Operators with directional
/// strategies (event-driven, breakout-on-news, news-fade) need the
/// opposite — explicit opt-in to trade through events. This enum
/// makes the choice an operator-driven setting instead of a baked
/// policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NewsTradingMode {
    /// Block new orders inside `news_kill_window_min` of any
    /// high-impact event. Default — the safe choice.
    #[default]
    BlockOnNews,
    /// Allow orders through the kill window. The UI shows a banner
    /// while a high-impact event is imminent so the operator knows
    /// what they're flying into.
    AllowAlways,
    /// Don't block, but surface a prominent warning in the UI when
    /// inside the kill window. Suited to operators who want a head's-
    /// up but don't want the gate to override their judgment.
    WarnOnly,
}

impl NewsTradingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BlockOnNews => "block_on_news",
            Self::AllowAlways => "allow_always",
            Self::WarnOnly => "warn_only",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::BlockOnNews => "Pause during news (safe default)",
            Self::AllowAlways => "Play through news (event-driven strategies)",
            Self::WarnOnly => "Warn only — don't block",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "block_on_news" | "block" | "pause" => Some(Self::BlockOnNews),
            "allow_always" | "allow" | "play" => Some(Self::AllowAlways),
            "warn_only" | "warn" => Some(Self::WarnOnly),
            _ => None,
        }
    }
}

/// News and LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NewsConfig {
    /// How the trading gate handles incoming high-impact news.
    /// Operator-controlled; default `block_on_news` preserves the
    /// pre-#117 safe behaviour. See [`NewsTradingMode`].
    #[serde(default)]
    pub news_trading_mode: NewsTradingMode,
    pub news_decay_minutes: usize,
    pub news_kill_window_min: usize,
    pub news_confidence_threshold: f64,
    pub news_calendar_enabled: bool,
    pub news_calendar_source: String,
    pub news_lookahead_minutes: usize,
    pub news_trade_on_event: bool,
    pub news_trade_confidence_threshold: f64,
    pub news_event_risk_pct: f64,
    pub enable_news: bool,
    pub news_sources: Vec<String>,
    pub rss_feeds: Vec<String>,
    pub enable_llm_helper: bool,
    pub llm_helper_enabled: bool,
    pub llm_sentiment_positive_threshold: f64,
    pub llm_sentiment_negative_threshold: f64,
    pub news_backfill_enabled: bool,
    pub news_backfill_days: usize,
    pub news_local_glob: String,
    pub perplexity_enabled: bool,
    pub perplexity_api_key_env: String,
    pub perplexity_model: String,
    pub perplexity_num_results: usize,
    pub perplexity_timeframe_hours: usize,
    pub strategist_enabled: bool,
    pub strategist_interval_minutes: usize,
    pub auto_rescore_enabled: bool,
    pub auto_rescore_days: usize,
    pub auto_rescore_max_events: usize,
    pub auto_rescore_only_missing: bool,
}

impl Default for NewsConfig {
    fn default() -> Self {
        Self {
            news_trading_mode: NewsTradingMode::default(),
            news_decay_minutes: 120,
            news_kill_window_min: 30,
            news_confidence_threshold: 0.65,
            news_calendar_enabled: true,
            news_calendar_source: "forexfactory".to_string(),
            news_lookahead_minutes: 60,
            news_trade_on_event: false,
            news_trade_confidence_threshold: 0.90,
            news_event_risk_pct: 0.001,
            enable_news: true,
            news_sources: vec!["rss".to_string()],
            // Public, no-API-key financial NEWS feeds for the AI news
            // desk (GET /news/feed). Operator-editable in Settings → News.
            // NB: the economic *calendar* lives in `news_calendar_source`
            // (ForexFactory's ffcal XML is a custom calendar format, not
            // RSS), so it intentionally does NOT belong in this list.
            // Verified reachable 2026-06-30 (200 + XML). The old defaults
            // (dailyfx, forexlive) now 403/redirect; ForexFactory's ffcal is a
            // calendar, not RSS (see `news_calendar_source`). Reused as the
            // runtime fallback when a user's configured feeds all fail.
            rss_feeds: default_news_rss_feeds(),
            enable_llm_helper: true,
            llm_helper_enabled: true,
            llm_sentiment_positive_threshold: 0.2,
            llm_sentiment_negative_threshold: -0.2,
            news_backfill_enabled: true,
            news_backfill_days: 30,
            news_local_glob: String::new(),
            perplexity_enabled: true,
            perplexity_api_key_env: "PPLX_API_KEY".to_string(),
            perplexity_model: "sonar".to_string(),
            perplexity_num_results: 10,
            perplexity_timeframe_hours: 24,
            strategist_enabled: false,
            strategist_interval_minutes: 30,
            auto_rescore_enabled: false,
            auto_rescore_days: 30,
            auto_rescore_max_events: 200,
            auto_rescore_only_missing: true,
        }
    }
}

/// App / server / trading-runtime knobs — config-driven replacement for the
/// `neoethos-app` env_overrides registry (HTTP server bind, cTrader
/// connection retry/backoff/timeout, partial-fill acceptance, chart-merge
/// quote side, PnL audit / circuit-breaker thresholds). The app installs
/// these into its `env_overrides` cache at startup so the trading layer reads
/// the single config instead of `std::env`. Clamping is applied by the
/// getters (same bounds the env readers used).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AppRuntimeConfig {
    /// HTTP server bind address `host:port` (default `127.0.0.1:7423`).
    pub server_bind: String,
    /// cTrader execution read-timeout (seconds); 0 disables. Clamped [0,3600].
    pub ctrader_read_timeout_secs: u64,
    /// cTrader execution attempts (initial + retries). Clamped [1,5].
    pub ctrader_max_attempts: u32,
    /// cTrader retry backoff base (ms). Clamped [10,2000].
    pub ctrader_backoff_base_ms: u64,
    /// Accept partial fills as final (default false).
    pub ctrader_allow_partial_fill: bool,
    /// cTrader streaming poll attempts. Clamped [1,5].
    pub ctrader_stream_max_attempts: u32,
    /// cTrader streaming backoff base (ms). Clamped [10,2000].
    pub ctrader_stream_backoff_base_ms: u64,
    /// Chart-merge quote side (`mid`/`bid`/`ask`); empty → caller default.
    pub chart_merge_side: String,
    /// PnL audit drift threshold (fraction of notional). Clamped [1e-5,0.05].
    pub pnl_audit_drift_fraction: f64,
    /// PnL circuit-breaker threshold (fraction). Clamped [1e-4,0.20].
    pub pnl_circuit_breaker_fraction: f64,
}

impl Default for AppRuntimeConfig {
    fn default() -> Self {
        Self {
            server_bind: "127.0.0.1:7423".to_string(),
            ctrader_read_timeout_secs: 30,
            ctrader_max_attempts: 3,
            ctrader_backoff_base_ms: 200,
            ctrader_allow_partial_fill: false,
            ctrader_stream_max_attempts: 3,
            ctrader_stream_backoff_base_ms: 200,
            chart_merge_side: String::new(),
            pnl_audit_drift_fraction: 0.001,
            pnl_circuit_breaker_fraction: 0.01,
        }
    }
}

/// Main settings structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub system: SystemConfig,
    pub risk: RiskConfig,
    pub models: ModelsConfig,
    pub news: NewsConfig,
    /// App / server / trading-runtime knobs (config-driven replacement for
    /// the `neoethos-app` env_overrides registry). See [`AppRuntimeConfig`].
    pub app_runtime: AppRuntimeConfig,
    pub secrets_file: PathBuf,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            risk: RiskConfig::default(),
            models: ModelsConfig::default(),
            news: NewsConfig::default(),
            app_runtime: AppRuntimeConfig::default(),
            secrets_file: PathBuf::from("keys.txt"),
        }
    }
}

/// Canonical user-data path for the operator's editable `config.yaml`.
///
/// **F-311 (2026-05-29) — single source of truth**. Historically four
/// separate call sites (`neoethos-core::Settings::load`, the
/// `neoethos-app` server routes, `neoethos-cli` argument parsing, the
/// `neoethos-models::registry`) each rolled their own resolution: some
/// honoured `$CONFIG_FILE`, some used a relative literal, some
/// hard-coded the install dir. That shadow made F-310 (supervisor
/// seeding the wrong file) extremely hard to diagnose because two
/// readers saw two different states for the same logical config.
///
/// Going forward every read should resolve via this helper:
///
/// * **Windows**: `%LOCALAPPDATA%\neoethos\config.yaml`
///   (`C:\Users\<u>\AppData\Local\neoethos\config.yaml`).
/// * **Linux**: `$XDG_DATA_HOME/neoethos/config.yaml` or
///   `~/.local/share/neoethos/config.yaml`.
/// * **macOS**: `~/Library/Application Support/neoethos/config.yaml`.
///
/// On startup the F-310 supervisor seeds this path from the bundle's
/// read-only config when the user file is missing; subsequent edits
/// (Settings → App tab, F-312 raw YAML editor, `/settings` POST) write
/// back to the same path. Tests that need a synthetic path can still
/// supply one via the `CONFIG_FILE` env var — `Settings::load` checks
/// that first.
pub fn user_config_path() -> PathBuf {
    // Explicit override (NEOETHOS_USER_DATA_DIR) wins on every platform, so the
    // desktop shell / power users can point ALL config + data readers at one
    // chosen root (e.g. a project dir) — keeping every resolver consistent.
    if let Some(dir) = crate::env_overrides::user_data_dir_override() {
        return PathBuf::from(dir).join("config.yaml");
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local).join("neoethos").join("config.yaml");
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("neoethos")
                .join("config.yaml");
        }
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            return PathBuf::from(xdg).join("neoethos").join("config.yaml");
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("neoethos")
                .join("config.yaml");
        }
    }
    PathBuf::from("config.yaml")
}

impl Settings {
    /// Load settings from YAML config file
    pub fn from_yaml(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let settings: Settings = serde_yaml_ng::from_str(&content)?;
        settings.validate_safety_bounds();
        Ok(settings)
    }

    /// Load settings from the canonical user-data config path.
    ///
    /// **F-311 (2026-05-29)**: this used to fall back to the literal
    /// string `"config.yaml"` (so it resolved against the process cwd).
    /// That broke whenever the operator ran the installed app from
    /// `Start Menu` — cwd is `C:\Windows\System32` and the bundle's
    /// read-only `config.yaml` lived elsewhere. The 4-location shadow
    /// (CLI / server / models / TUI each rolling their own path) made
    /// debugging F-310 painful. The unified resolution order is now:
    ///   1. `$CONFIG_FILE` env var (CI / test overrides)
    ///   2. `user_config_path()` — `%LOCALAPPDATA%\neoethos\config.yaml`
    ///      on Windows, `$XDG_DATA_HOME/neoethos/config.yaml` on Linux,
    ///      `~/Library/Application Support/neoethos/config.yaml` on
    ///      macOS — the same place the F-310 supervisor seeds.
    ///   3. Last-resort fallback to literal `"config.yaml"` (relative)
    ///      so cargo-test invocations in the workspace root still work.
    pub fn load() -> anyhow::Result<Self> {
        let config_file = std::env::var("CONFIG_FILE")
            .map(PathBuf::from)
            .ok()
            .unwrap_or_else(|| {
                let user = user_config_path();
                if user.exists() {
                    user
                } else {
                    PathBuf::from("config.yaml")
                }
            });
        Self::from_yaml(&config_file)
    }

    /// Sanity-check loaded RiskConfig values against prop-firm-safe bounds.
    ///
    /// We can't reject the load — config consumers expect a non-fatal load —
    /// but a mistyped `risk_per_trade: 50` (meaning 50% instead of 0.5%) needs
    /// to be screamed about, otherwise the bot silently sizes 100× too big.
    /// All checks emit `tracing::error` with the field, the loaded value,
    /// and a recommended sane value. M9 in the audit.
    fn validate_safety_bounds(&self) {
        let risk = &self.risk;
        // risk_per_trade should be a fraction (0.0 — 0.05 typical, 0.10 max).
        // A YAML value > 1.0 means the user typed a percentage (e.g. 1.5 for
        // 1.5%) — we recover by interpreting it as percent and warning.
        if risk.risk_per_trade > 1.0 {
            tracing::error!(
                target: "neoethos_core::config",
                risk_per_trade = risk.risk_per_trade,
                "RiskConfig.risk_per_trade > 1.0 — looks like a percentage typo. \
                 0.005 means 0.5%, NOT 0.5 = 50%. Halt or fix the config."
            );
        } else if risk.risk_per_trade > 0.05 {
            tracing::warn!(
                target: "neoethos_core::config",
                risk_per_trade = risk.risk_per_trade,
                "RiskConfig.risk_per_trade > 5% per trade — uncommonly aggressive for a prop firm"
            );
        }
        if risk.daily_drawdown_limit <= 0.0 || risk.daily_drawdown_limit > 0.20 {
            tracing::error!(
                target: "neoethos_core::config",
                daily_drawdown_limit = risk.daily_drawdown_limit,
                "RiskConfig.daily_drawdown_limit must be in (0, 0.20]; typical prop firms set 0.04-0.05"
            );
        }
        if risk.total_drawdown_limit <= risk.daily_drawdown_limit {
            tracing::error!(
                target: "neoethos_core::config",
                total = risk.total_drawdown_limit,
                daily = risk.daily_drawdown_limit,
                "RiskConfig.total_drawdown_limit should exceed daily_drawdown_limit"
            );
        }
        if risk.total_drawdown_limit > 0.30 {
            tracing::error!(
                target: "neoethos_core::config",
                total_drawdown_limit = risk.total_drawdown_limit,
                "RiskConfig.total_drawdown_limit > 30% — exceeds every published prop-firm rule"
            );
        }
    }

    fn parse_csv_list(value: &str) -> Vec<String> {
        value
            .split(',')
            .map(|entry| entry.trim().to_string())
            .filter(|entry| !entry.is_empty())
            .collect()
    }

    fn apply_overrides_from_lookup<F>(&mut self, mut lookup: F)
    where
        F: FnMut(&str) -> Option<String>,
    {
        if let Some(symbol) = lookup("NEOETHOS_BOT_SYMBOL") {
            self.system.symbol = symbol;
        }

        let data_root = lookup("NEOETHOS_BOT_DATA_ROOT").or_else(|| lookup("NEOETHOS_BOT_DATA_DIR"));
        if let Some(data_root) = data_root {
            self.system.data_dir = PathBuf::from(data_root);
        }

        if let Some(base_tf) = lookup("NEOETHOS_BOT_BASE_TIMEFRAME") {
            self.system.base_timeframe = base_tf;
        }

        if let Some(higher_tfs) = lookup("NEOETHOS_BOT_HIGHER_TFS") {
            let parsed = Self::parse_csv_list(&higher_tfs);
            if !parsed.is_empty() {
                self.system.higher_timeframes = parsed;
            }
        }

        if let Some(device) = lookup("NEOETHOS_BOT_DEVICE") {
            self.system.device = device;
        }

        if let Some(preference) = lookup("NEOETHOS_BOT_ENABLE_GPU_PREFERENCE") {
            self.system.enable_gpu_preference = preference;
        }

        if let Some(tree_device) = lookup("NEOETHOS_BOT_TREE_DEVICE") {
            self.models.tree_device_preference = tree_device;
        }

        if let Some(model_names) = lookup("NEOETHOS_BOT_ML_MODELS") {
            let parsed = Self::parse_csv_list(&model_names);
            if !parsed.is_empty() {
                self.models.ml_models = parsed;
            }
        }

        if let Some(num_transformers) =
            lookup("NEOETHOS_BOT_NUM_TRANSFORMERS").and_then(|value| value.parse::<usize>().ok())
        {
            self.models.num_transformers = num_transformers.max(1);
        }

        if let Some(model_names) = lookup("NEOETHOS_BOT_PHASE5_CORE_MODELS") {
            let parsed = Self::parse_csv_list(&model_names);
            if !parsed.is_empty() {
                self.models.phase5_core_models = parsed;
            }
        }

        if let Some(model_names) = lookup("NEOETHOS_BOT_REGIME_TREND_MODELS") {
            let parsed = Self::parse_csv_list(&model_names);
            if !parsed.is_empty() {
                self.models.regime_trend_models = parsed;
            }
        }

        if let Some(model_names) = lookup("NEOETHOS_BOT_REGIME_RANGE_MODELS") {
            let parsed = Self::parse_csv_list(&model_names);
            if !parsed.is_empty() {
                self.models.regime_range_models = parsed;
            }
        }

        if let Some(model_names) = lookup("NEOETHOS_BOT_REGIME_NEUTRAL_MODELS") {
            let parsed = Self::parse_csv_list(&model_names);
            if !parsed.is_empty() {
                self.models.regime_neutral_models = parsed;
            }
        }

        if let Some(enabled) = lookup("NEOETHOS_BOT_PHASE5_FILTER_META_BLENDER") {
            self.models.phase5_filter_meta_blender = matches!(
                enabled.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }

        if let Some(enabled) = lookup("NEOETHOS_BOT_REGIME_ROUTER_ENABLED") {
            self.models.regime_router_enabled = matches!(
                enabled.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }

        if let Some(min_models) = lookup("NEOETHOS_BOT_REGIME_ROUTER_MIN_MODELS")
            .and_then(|value| value.parse::<usize>().ok())
        {
            self.models.regime_router_min_models = min_models.max(1);
        }

        if let Some(method) = lookup("NEOETHOS_BOT_CALIBRATION_METHOD") {
            self.models.calibration_method = method;
        }

        if let Some(min_rows) =
            lookup("NEOETHOS_BOT_CALIBRATION_MIN_ROWS").and_then(|value| value.parse::<usize>().ok())
        {
            self.models.calibration_min_rows = min_rows.max(1);
        }

        if let Some(holdout_pct) =
            lookup("NEOETHOS_BOT_TRAIN_HOLDOUT_PCT").and_then(|value| value.parse::<f64>().ok())
        {
            self.models.train_holdout_pct = holdout_pct;
        }

        if let Some(label_horizon) =
            lookup("NEOETHOS_BOT_LABEL_HORIZON_BARS").and_then(|value| value.parse::<usize>().ok())
        {
            self.models.label_horizon_bars = label_horizon;
        }

        if let Some(meta_hold) = lookup("NEOETHOS_BOT_META_LABEL_MAX_HOLD_BARS")
            .and_then(|value| value.parse::<usize>().ok())
        {
            self.risk.meta_label_max_hold_bars = meta_hold.max(1);
        }

        if let Some(conf_threshold) =
            lookup("NEOETHOS_BOT_PROP_CONF_THRESHOLD").and_then(|value| value.parse::<f64>().ok())
        {
            self.models.prop_conf_threshold = conf_threshold;
        }

        if let Some(use_rllib_agent) = lookup("NEOETHOS_BOT_USE_RLLIB_AGENT") {
            self.models.use_rllib_agent = matches!(
                use_rllib_agent.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }

        if let Some(rllib_workers) =
            lookup("NEOETHOS_BOT_RLLIB_NUM_WORKERS").and_then(|value| value.parse::<usize>().ok())
        {
            self.models.rllib_num_workers = rllib_workers;
        }

        if let Some(auto_enable_rllib) = lookup("NEOETHOS_BOT_AUTO_ENABLE_RLLIB") {
            self.models.auto_enable_rllib = matches!(
                auto_enable_rllib.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }

        if let Some(prop_search_device) = lookup("NEOETHOS_BOT_PROP_SEARCH_DEVICE") {
            self.models.prop_search_device = prop_search_device;
        }

        if let Some(prop_search_async) = lookup("NEOETHOS_BOT_PROP_SEARCH_ASYNC") {
            self.models.prop_search_async = matches!(
                prop_search_async.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }

        if let Some(prop_search_async_wait) = lookup("NEOETHOS_BOT_PROP_SEARCH_ASYNC_WAIT") {
            self.models.prop_search_async_wait = matches!(
                prop_search_async_wait.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
    }

    /// Load settings with environment variable overrides
    pub fn load_with_env() -> anyhow::Result<Self> {
        let mut settings = Self::load()?;
        settings.apply_overrides_from_lookup(|key| std::env::var(key).ok());
        Ok(settings)
    }

    /// Save settings to YAML file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let yaml = serde_yaml_ng::to_string(self)?;
        // Audit M07: atomic write (temp + fsync + rename) so a crash mid-write
        // can never leave a truncated config.yaml — a corrupt config is the
        // known "app won't open" root cause. The previous std::fs::write was
        // non-atomic.
        crate::storage::json::write_bytes_atomic(path, yaml.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn config_maps_serialize_in_sorted_deterministic_order() {
        // M06/M07 follow-up: HashMap config fields must serialize in sorted
        // key order so Settings::save doesn't reshuffle config.yaml on every
        // write. Insert keys out of order and confirm two serializations
        // match and the keys come out sorted.
        let mut s = Settings::default();
        for (k, v) in [("H4", 3usize), ("M1", 1), ("D1", 5), ("M5", 2)] {
            s.models.hpo_trials_by_model.insert(k.to_string(), v);
            s.models.prop_search_max_rows_by_tf.insert(k.to_string(), v);
        }
        s.models
            .model_param_overrides
            .insert("zeta".to_string(), HashMap::from([("b".to_string(), "1".to_string())]));
        s.models
            .model_param_overrides
            .insert("alpha".to_string(), HashMap::from([("a".to_string(), "0".to_string())]));

        let a = serde_yaml_ng::to_string(&s).unwrap();
        let b = serde_yaml_ng::to_string(&s).unwrap();
        assert_eq!(a, b, "two serializations must be byte-identical");

        // hpo_trials_by_model keys appear in sorted order.
        let positions: Vec<usize> = ["D1", "H4", "M1", "M5"]
            .iter()
            .map(|k| {
                a.find(&format!("{k}:"))
                    .unwrap_or_else(|| panic!("key {k} present"))
            })
            .collect();
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "hpo_trials_by_model keys must be sorted"
        );
        // Nested override keys sorted too.
        assert!(a.find("alpha:").unwrap() < a.find("zeta:").unwrap());
    }

    #[test]
    fn test_default_settings() {
        // F-303 (2026-05-28): updated post-F-129 — the previous
        // assertion `settings.system.symbol == "EURUSD"` was a stale
        // hardcoded-default check from before the synthetic-data
        // cleanup. `SystemConfig::default()` now returns empty for
        // both `symbol` and `account_currency` (F-304), forcing the
        // operator's `config.yaml` to populate them. The pre-flight
        // bail in `run_discovery_cycle` catches the omission with an
        // actionable error.
        let settings = Settings::default();
        assert_eq!(settings.system.symbol, "", "default symbol must be empty per F-129");
        assert_eq!(
            settings.system.account_currency, "",
            "default account_currency must be empty per F-304"
        );
        assert_eq!(settings.risk.initial_balance, 10_000.0);
        assert!(!settings.models.ml_models.is_empty());
    }

    // ─── UI↔CLI parity: the shared timeframe/symbol resolvers ───────────────
    // These lock the behaviour of `SystemConfig::resolve_*`, the SINGLE source
    // of truth that BOTH `neoethos-cli` and the app server call. If this drifts,
    // the two entry points would search differently from the same config —
    // exactly the divergence the 2026-06-04 parity pass removed.

    #[test]
    fn resolve_higher_timeframes_default_config_multi_resolution() {
        // Default: multi_resolution_enabled = true and the multi-res list is the
        // FULL canonical set → "every configured TF except the effective base".
        let sys = SystemConfig::default();

        let m1 = sys.resolve_higher_timeframes("M1");
        assert_eq!(
            m1,
            vec!["M3", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"],
            "M1 base → all canonical above M1"
        );
        assert!(!m1.iter().any(|tf| tf == "M1"), "base itself is excluded");

        // base=H1: multi-resolution keeps LOWER TFs (M1..M30) as extra context
        // too — only the base is dropped. The Flutter UI cannot replicate this,
        // which is precisely why an untouched UI sends no override and lets this
        // resolver decide (parity with the CLI).
        let h1 = sys.resolve_higher_timeframes("H1");
        assert!(h1.contains(&"M5".to_string()), "lower TFs retained under multi-res");
        assert!(h1.contains(&"H4".to_string()), "higher TFs retained");
        assert!(!h1.iter().any(|tf| tf == "H1"), "base itself is excluded");
        assert_eq!(h1.len(), 10, "all 11 canonical minus the base");

        // Effective-base relativity: an overridden base trims itself out even
        // when it differs from `self.base_timeframe`.
        assert!(!sys.resolve_higher_timeframes("H4").iter().any(|tf| tf == "H4"));
    }

    #[test]
    fn resolve_higher_timeframes_multi_resolution_off_filters_strictly_above() {
        // multi_resolution OFF → higher_timeframes filtered to strictly-above
        // the base in canonical order (never a lower/equal TF).
        let mut sys = SystemConfig::default();
        sys.multi_resolution_enabled = false;
        assert_eq!(
            sys.resolve_higher_timeframes("H1"),
            vec!["H4", "H12", "D1", "W1", "MN1"],
            "H1 base, multi-res off → only canonical TFs strictly above H1"
        );

        // An operator exclusion in higher_timeframes is honoured, and entries
        // not strictly above the base are dropped (D1/H4 kept, M5 below M1? no —
        // M5 is above M1, so it stays; M1-equal would be dropped).
        sys.higher_timeframes = vec!["H4".to_string(), "D1".to_string(), "M5".to_string()];
        assert_eq!(
            sys.resolve_higher_timeframes("H1"),
            vec!["H4", "D1"],
            "restricted higher_timeframes respected; M5 (below H1) excluded"
        );
    }

    #[test]
    fn resolve_base_and_symbol_trim_preserve_config_value() {
        let mut sys = SystemConfig::default();
        sys.base_timeframe = "  H4 ".to_string();
        sys.symbol = " EURUSD ".to_string();
        assert_eq!(sys.resolve_base_timeframe(), "H4");
        assert_eq!(sys.resolve_symbol(), "EURUSD");
    }

    #[test]
    fn test_serialize_deserialize() {
        let settings = Settings::default();
        let yaml = serde_yaml_ng::to_string(&settings).unwrap();
        let deserialized: Settings = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(deserialized.system.symbol, settings.system.symbol);
    }

    #[test]
    fn runtime_overrides_apply_to_dispatch_and_label_settings() {
        let mut settings = Settings::default();
        let overrides = HashMap::from([
            (
                "NEOETHOS_BOT_ENABLE_GPU_PREFERENCE".to_string(),
                "gpu".to_string(),
            ),
            ("NEOETHOS_BOT_TREE_DEVICE".to_string(), "cuda".to_string()),
            ("NEOETHOS_BOT_NUM_TRANSFORMERS".to_string(), "4".to_string()),
            (
                "NEOETHOS_BOT_ML_MODELS".to_string(),
                "lightgbm, xgboost , neat".to_string(),
            ),
            (
                "NEOETHOS_BOT_PHASE5_CORE_MODELS".to_string(),
                "transformer, tabnet".to_string(),
            ),
            (
                "NEOETHOS_BOT_PHASE5_FILTER_META_BLENDER".to_string(),
                "false".to_string(),
            ),
            (
                "NEOETHOS_BOT_REGIME_ROUTER_ENABLED".to_string(),
                "true".to_string(),
            ),
            (
                "NEOETHOS_BOT_REGIME_ROUTER_MIN_MODELS".to_string(),
                "3".to_string(),
            ),
            (
                "NEOETHOS_BOT_CALIBRATION_METHOD".to_string(),
                "temperature".to_string(),
            ),
            (
                "NEOETHOS_BOT_CALIBRATION_MIN_ROWS".to_string(),
                "512".to_string(),
            ),
            ("NEOETHOS_BOT_TRAIN_HOLDOUT_PCT".to_string(), "0.3".to_string()),
            ("NEOETHOS_BOT_LABEL_HORIZON_BARS".to_string(), "24".to_string()),
            (
                "NEOETHOS_BOT_META_LABEL_MAX_HOLD_BARS".to_string(),
                "144".to_string(),
            ),
            (
                "NEOETHOS_BOT_PROP_CONF_THRESHOLD".to_string(),
                "0.72".to_string(),
            ),
            ("NEOETHOS_BOT_USE_RLLIB_AGENT".to_string(), "1".to_string()),
            ("NEOETHOS_BOT_RLLIB_NUM_WORKERS".to_string(), "6".to_string()),
            ("NEOETHOS_BOT_AUTO_ENABLE_RLLIB".to_string(), "off".to_string()),
            (
                "NEOETHOS_BOT_PROP_SEARCH_DEVICE".to_string(),
                "cuda:0".to_string(),
            ),
            (
                "NEOETHOS_BOT_PROP_SEARCH_ASYNC".to_string(),
                "true".to_string(),
            ),
            (
                "NEOETHOS_BOT_PROP_SEARCH_ASYNC_WAIT".to_string(),
                "true".to_string(),
            ),
        ]);

        settings.apply_overrides_from_lookup(|key| overrides.get(key).cloned());

        assert_eq!(settings.system.enable_gpu_preference, "gpu");
        assert_eq!(settings.models.tree_device_preference, "cuda");
        assert_eq!(settings.models.num_transformers, 4);
        assert_eq!(
            settings.models.ml_models,
            vec![
                "lightgbm".to_string(),
                "xgboost".to_string(),
                "neat".to_string(),
            ]
        );
        assert_eq!(
            settings.models.phase5_core_models,
            vec!["transformer".to_string(), "tabnet".to_string()]
        );
        assert!(!settings.models.phase5_filter_meta_blender);
        assert!(settings.models.regime_router_enabled);
        assert_eq!(settings.models.regime_router_min_models, 3);
        assert_eq!(settings.models.calibration_method, "temperature");
        assert_eq!(settings.models.calibration_min_rows, 512);
        assert_eq!(settings.models.train_holdout_pct, 0.3);
        assert_eq!(settings.models.label_horizon_bars, 24);
        assert_eq!(settings.risk.meta_label_max_hold_bars, 144);
        assert_eq!(settings.models.prop_conf_threshold, 0.72);
        assert!(settings.models.use_rllib_agent);
        assert_eq!(settings.models.rllib_num_workers, 6);
        assert!(!settings.models.auto_enable_rllib);
        assert_eq!(settings.models.prop_search_device, "cuda:0");
        assert!(settings.models.prop_search_async);
        assert!(settings.models.prop_search_async_wait);
    }
}
