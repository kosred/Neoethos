// Core configuration structures for Forex trading system
// Project configuration loader.

use crate::contracts::CANONICAL_TIMEFRAMES;
use crate::domain::prop_firm::{PropFirmConstraints, PropFirmPreset, PropFirmRuntimeDefaults};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemConfig {
    pub symbol: String,
    pub symbols: Vec<String>,
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
    pub history_years: usize,
    pub trading_session_start: String,
    pub trading_session_end: String,
    pub session_timezone: String,
    /// Broker time zone used for prop-firm calendar-day boundaries (e.g.
    /// daily-DD reset). Most cTrader prop firms run on EET ("Europe/Athens",
    /// UTC+2/+3); some run pure UTC. When set, the trading runtime computes
    /// `day_id` against this offset instead of the local clock. Empty string
    /// falls back to `session_timezone` (default "UTC"). M12 in the audit.
    #[serde(default)]
    pub broker_timezone: String,
    pub broker_backend: String,
    pub poll_interval_seconds: u64,
    pub metrics_logging_enabled: bool,
    pub metrics_db_path: PathBuf,
    pub risk_ledger_enabled: bool,
    pub risk_ledger_max_events: usize,
    pub strategy_ledger_path: PathBuf,
    pub enable_dashboard: bool,
    pub cache_enabled: bool,
    pub cache_dir: PathBuf,
    pub cache_max_age_minutes: u64,
    pub deep_purge_mode: String,
    pub deep_purge_on_train: bool,
    pub n_jobs: usize,
    pub enable_gpu_preference: String,
    pub discovery_auto_cap: bool,
    pub discovery_max_rows: usize,
    pub discovery_stream: bool,
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
            symbols: Vec::new(),
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
            history_years: 10,
            trading_session_start: "00:05".to_string(),
            trading_session_end: "23:55".to_string(),
            session_timezone: "UTC".to_string(),
            broker_timezone: String::new(), // empty = fall back to session_timezone
            broker_backend: "ctrader".to_string(),
            poll_interval_seconds: 60,
            metrics_logging_enabled: true,
            metrics_db_path: PathBuf::from("metrics.sqlite"),
            risk_ledger_enabled: true,
            risk_ledger_max_events: 1000,
            strategy_ledger_path: PathBuf::from("strategy_ledger.sqlite"),
            enable_dashboard: true,
            cache_enabled: true,
            cache_dir: PathBuf::from("cache"),
            cache_max_age_minutes: 60,
            deep_purge_mode: "off".to_string(),
            deep_purge_on_train: true,
            n_jobs,
            enable_gpu_preference: "auto".to_string(),
            discovery_auto_cap: true,
            discovery_max_rows: 0,
            discovery_stream: false,
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
    pub daily_drawdown_limit: f64,
    pub total_drawdown_limit: f64,
    pub min_risk_reward: f64,
    pub spread_guard_multiplier: f64,
    pub slippage_guard_multiplier: f64,
    pub max_lot_size: f64,
    pub require_stop_loss: bool,
    pub challenge_mode: bool,
    pub challenge_phase: String,
    pub prop_firm_rules: bool,
    pub max_daily_risk_pct: f64,
    pub base_risk_per_trade: f64,
    pub daily_risk_budget: f64,
    pub consistency_tracking: bool,
    pub min_confidence_threshold: f64,
    pub kill_zones_enabled: bool,
    pub enhanced_features: bool,
    pub uncertainty_quantification: bool,
    pub max_trades_per_day: usize,
    pub daily_profit_stop_pct: f64,
    pub recovery_mode_enabled: bool,
    pub feature_drift_threshold: f64,
    pub high_quality_confidence: f64,
    pub high_quality_risk_pct: f64,
    pub high_quality_rr: f64,
    pub atr_period: usize,
    pub atr_stop_multiplier: f64,
    pub triple_barrier_max_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub kelly_lambda: f64,
    pub slippage_pips: f64,
    pub commission_per_lot: f64,
    pub backtest_spread_pips: f64,
    pub cost_penalty_r: f64,
    pub gate_trade_prob: f64,
    pub daily_hard_stop_pct: f64,
    pub conformal_enabled: bool,
    pub conformal_alpha: f64,
    pub conformal_abstain_min_set_size: usize,
    pub volatility_stop_sigma: f64,
    pub volatility_lookback: usize,
    pub meta_label_tp_pips: Option<f64>,
    pub meta_label_sl_pips: Option<f64>,
    pub meta_label_max_hold_bars: usize,
    pub meta_label_min_prob_threshold: f64,
    pub meta_label_min_dist: f64,
    pub meta_label_fixed_sl: f64,
    pub meta_label_fixed_tp: f64,
    pub stop_target_mode: String,
    pub vol_estimator: String,
    pub vol_ensemble_weights: HashMap<String, f64>,
    pub vol_ensemble_weights_trend: Option<HashMap<String, f64>>,
    pub vol_ensemble_weights_range: Option<HashMap<String, f64>>,
    pub vol_ensemble_weights_neutral: Option<HashMap<String, f64>>,
    pub vol_window: usize,
    pub ewma_lambda: f64,
    pub ewma_lambda_by_timeframe: HashMap<String, f64>,
    pub vol_horizon_bars: usize,
    pub tail_window: usize,
    pub tail_alpha: f64,
    pub tail_step: usize,
    pub tail_max_bars: usize,
    pub stop_k_vol: f64,
    pub stop_k_tail: f64,
    pub rr_trend: f64,
    pub rr_range: f64,
    pub rr_neutral: f64,
    pub regime_adx_trend: f64,
    pub regime_adx_range: f64,
    pub hurst_window: usize,
    pub hurst_trend: f64,
    pub hurst_range: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        let mut vol_ensemble_weights = HashMap::new();
        vol_ensemble_weights.insert("yang_zhang".to_string(), 1.0);
        vol_ensemble_weights.insert("garman_klass".to_string(), 1.0);
        vol_ensemble_weights.insert("rogers_satchell".to_string(), 1.0);
        vol_ensemble_weights.insert("parkinson".to_string(), 1.0);

        // EWMA lambdas keyed by canonical timeframe. The list MUST match
        // `CANONICAL_TIMEFRAMES` exactly — every supported timeframe gets
        // its own decay coefficient, and unknown timeframes must fall
        // back to a default rather than be hard-coded here.
        let mut ewma_lambda_by_timeframe = HashMap::new();
        for (tf, lambda) in [
            ("M1", 0.90),
            ("M3", 0.91),
            ("M5", 0.92),
            ("M15", 0.94),
            ("M30", 0.95),
            ("H1", 0.96),
            ("H4", 0.97),
            ("H12", 0.98),
            ("D1", 0.985),
            ("W1", 0.99),
            ("MN1", 0.995),
        ] {
            debug_assert!(
                CANONICAL_TIMEFRAMES.contains(&tf),
                "ewma_lambda_by_timeframe key {} must be a canonical timeframe",
                tf
            );
            ewma_lambda_by_timeframe.insert(tf.to_string(), lambda);
        }

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
            // Internal early stop sits 20% below the firm's published
            // daily-loss ceiling so a guard-rail trips before a real
            // breach. Operators override in YAML if their firm gives
            // tighter / looser tolerance.
            daily_drawdown_limit: runtime.daily_dd_stop_trading_pct,
            // Internal trailing total cap at 70% of the firm's
            // overall-drawdown ceiling for the same buffer reason.
            total_drawdown_limit: (constraints.max_overall_drawdown_pct as f64) * 0.7,
            min_risk_reward: 2.0,
            spread_guard_multiplier: 2.5,
            slippage_guard_multiplier: 2.0,
            max_lot_size: runtime.max_lot_size,
            require_stop_loss: true,
            challenge_mode: false,
            challenge_phase: "phase_1".to_string(),
            // Disable the prop-firm gate entirely when the operator
            // selected `preset: none` — they're trading their own
            // money; we still respect per-trade risk limits but skip
            // the challenge accounting.
            prop_firm_rules: preset != PropFirmPreset::None,
            max_daily_risk_pct: runtime.daily_dd_stop_trading_pct,
            base_risk_per_trade: 0.03,
            daily_risk_budget: runtime.daily_dd_stop_trading_pct,
            consistency_tracking: true,
            min_confidence_threshold: 0.55,
            kill_zones_enabled: true,
            enhanced_features: true,
            uncertainty_quantification: true,
            // Cap is preset-driven. FTMO defaults to 15; The5%ers is
            // tighter; "own money" raises it. Operators can override
            // via YAML when their style demands a different cap.
            max_trades_per_day: runtime.max_trades_per_day,
            daily_profit_stop_pct: runtime.daily_profit_lock_pct,
            recovery_mode_enabled: true,
            feature_drift_threshold: 0.30,
            high_quality_confidence: 0.65,
            high_quality_risk_pct: 0.030,
            high_quality_rr: 2.0,
            atr_period: 14,
            atr_stop_multiplier: 1.5,
            triple_barrier_max_bars: 35,
            trailing_enabled: true,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            kelly_lambda: 1.0,
            slippage_pips: 0.5,
            commission_per_lot: 7.0,
            backtest_spread_pips: 1.5,
            cost_penalty_r: 0.0,
            gate_trade_prob: 0.55,
            daily_hard_stop_pct: runtime.daily_dd_stop_trading_pct,
            conformal_enabled: true,
            conformal_alpha: 0.10,
            conformal_abstain_min_set_size: 3,
            volatility_stop_sigma: 0.02,
            volatility_lookback: 50,
            meta_label_tp_pips: None,
            meta_label_sl_pips: None,
            meta_label_max_hold_bars: 100,
            meta_label_min_prob_threshold: 0.55,
            meta_label_min_dist: 0.0005,
            meta_label_fixed_sl: 0.0020,
            meta_label_fixed_tp: 0.0040,
            stop_target_mode: "blend".to_string(),
            vol_estimator: "ensemble".to_string(),
            vol_ensemble_weights,
            vol_ensemble_weights_trend: None,
            vol_ensemble_weights_range: None,
            vol_ensemble_weights_neutral: None,
            vol_window: 50,
            ewma_lambda: 0.94,
            ewma_lambda_by_timeframe,
            vol_horizon_bars: 5,
            tail_window: 100,
            tail_alpha: 0.975,
            tail_step: 5,
            tail_max_bars: 300_000,
            stop_k_vol: 1.0,
            stop_k_tail: 1.25,
            rr_trend: 2.5,
            rr_range: 1.5,
            rr_neutral: 2.0,
            regime_adx_trend: 25.0,
            regime_adx_range: 20.0,
            hurst_window: 100,
            hurst_trend: 0.55,
            hurst_range: 0.45,
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
    pub hpo_trials_by_model: HashMap<String, usize>,
    pub hpo_max_rows: usize,
    pub max_epochs_by_model: HashMap<String, usize>,
    pub ray_tune_max_concurrency: usize,
    pub export_onnx: bool,
    pub calibration_enabled: bool,
    pub calibration_method: String,
    pub calibration_min_rows: usize,
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
            prefilter_insample_frac: 0.70,
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
            prop_search_max_hours: 8.0,
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
            walkforward_splits: 20,
            embargo_minutes: 120,
            discovery_mode: "prop_firm".to_string(),
            search_runtime: SearchRuntimeConfig::default(),
            discovery_runtime: DiscoveryRuntimeConfig::default(),
            eval_runtime: EvalRuntimeConfig::default(),
            quality_runtime: QualityRuntimeConfig::default(),
            backtest_runtime: BacktestRuntimeConfig::default(),
            seen_signature_runtime: SeenSignatureRuntimeConfig::default(),
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
            cpcv_max_rows: 0,
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
            rss_feeds: vec![
                "https://www.dailyfx.com/feeds/market-news".to_string(),
                "https://www.forexlive.com/feed/news".to_string(),
                "https://feeds.marketwatch.com/marketwatch/topstories/".to_string(),
                "https://feeds.marketwatch.com/marketwatch/marketpulse/".to_string(),
            ],
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
        std::fs::write(path, yaml)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
