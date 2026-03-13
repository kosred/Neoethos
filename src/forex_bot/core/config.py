import os
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

ALL_TIMEFRAMES: tuple[str, ...] = (
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M10",
    "M12",
    "M15",
    "M20",
    "M30",
    "H1",
    "H2",
    "H3",
    "H4",
    "H6",
    "H8",
    "H12",
    "D1",
    "W1",
    "MN1",
)


def _default_all_timeframes() -> list[str]:
    # Return a fresh list for each settings instance.
    return list(ALL_TIMEFRAMES)


_RUNTIME_PROFILE_ALIASES: dict[str, str] = {
    "rust": "rust_32gb",
    "rust32": "rust_32gb",
    "rust_32": "rust_32gb",
    "rust_32gb": "rust_32gb",
    "rust-safe": "rust_32gb",
    "rust_safe": "rust_32gb",
    "rustmax": "rust_max",
    "rust_max": "rust_max",
    "rust-fast": "rust_max",
    "rust_fast": "rust_max",
}

_RUNTIME_PROFILE_PRESETS: dict[str, dict[str, str]] = {
    # Safe defaults for 32GB-class machines.
    "rust_32gb": {
        "FOREX_BOT_PANDAS_FREE": "1",
        "FOREX_BOT_RUST_ONLY": "1",
        "FOREX_BOT_TREE_BACKEND": "rust_strict",
        "FOREX_BOT_TREE_RUST_FALLBACK": "0",
        "FOREX_BOT_RUST_FEATURES": "1",
        "FOREX_BOT_FEATURES_BACKEND": "rust_strict",
        "FOREX_BOT_DATA_BACKEND": "rust_strict",
        "FOREX_BOT_FRAME_IO_BACKEND": "polars",
        "FOREX_BOT_USE_ALL_TIMEFRAMES": "1",
        "FOREX_BOT_PARALLEL_FEATURES": "auto",
        "FOREX_BOT_FEATURE_WORKER_GB": "6",
        "FOREX_BOT_RUST_FEATURE_PROFILE": "core",
        "FOREX_BOT_RUST_HTF_FEATURE_PROFILE": "compact",
        "FOREX_BOT_RUST_MAX_FEATURES": "96",
        "FOREX_BOT_RUST_MAX_HTF_FEATURES": "12",
        "FOREX_BOT_PROP_SEARCH_TRAIN_YEARS": "10",
        "FOREX_BOT_PROP_HOLDOUT_YEARS": "3",
        "FOREX_BOT_PROP_HOLDOUT_REQUIRED": "1",
        "FOREX_BOT_PROP_HOLDOUT_FRACTION": "0",
        "FOREX_BOT_PROP_ELITE_FILTER": "1",
        "FOREX_BOT_PROP_REQUIRE_FORWARD_PASS": "1",
        "FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS": "6",
        "FOREX_BOT_PROP_HOLDOUT_MAX_DD": "0.03",
        "FOREX_BOT_PROP_REQUIRE_ALL_TFS": "1",
        "FOREX_BOT_PROP_KEEP_MIN_SHARPE": "1.20",
        "FOREX_BOT_PROP_KEEP_MIN_WIN_RATE": "0.52",
        "FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR": "1.30",
        "FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT": "0.01",
        "FOREX_BOT_PROP_SEARCH_MAX_ROWS": "0",
        "FOREX_BOT_RUST_BASE_SIGNAL_MODE": "auto",
        "FOREX_BOT_BASE_SIGNAL_ALLOW_PY_MIXER": "0",
        "FOREX_BOT_BASE_SIGNAL_ALLOW_CLASSIC_FALLBACK": "0",
        "FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK": "0",
        "FOREX_BOT_DISCOVERY_RUST_ONLY": "1",
        "FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK": "0",
        "FOREX_BOT_TALIB_ALLOW_PY_FALLBACK": "0",
        "FOREX_BOT_PROP_PY_FALLBACK": "0",
        "FOREX_BOT_PROP_ALLOW_PY_RESCORING": "0",
        "FOREX_BOT_PROP_ALLOW_PY_EXPANSION": "0",
        "FOREX_BOT_STOP_TARGET_ALLOW_PY_FALLBACK": "0",
    },
    # More aggressive profile for larger RAM/CPU machines.
    "rust_max": {
        "FOREX_BOT_PANDAS_FREE": "1",
        "FOREX_BOT_RUST_ONLY": "1",
        "FOREX_BOT_TREE_BACKEND": "rust_strict",
        "FOREX_BOT_TREE_RUST_FALLBACK": "0",
        "FOREX_BOT_RUST_FEATURES": "1",
        "FOREX_BOT_FEATURES_BACKEND": "rust_strict",
        "FOREX_BOT_DATA_BACKEND": "rust_strict",
        "FOREX_BOT_FRAME_IO_BACKEND": "polars",
        "FOREX_BOT_USE_ALL_FEATURES": "1",
        "FOREX_BOT_USE_ALL_TIMEFRAMES": "1",
        "FOREX_BOT_PARALLEL_FEATURES": "auto",
        "FOREX_BOT_FEATURE_WORKER_GB": "8",
        "FOREX_BOT_RUST_FEATURE_PROFILE": "full",
        "FOREX_BOT_RUST_HTF_FEATURE_PROFILE": "full",
        "FOREX_BOT_RUST_MAX_FEATURES": "0",
        "FOREX_BOT_RUST_MAX_HTF_FEATURES": "0",
        "FOREX_BOT_PROP_SEARCH_TRAIN_YEARS": "10",
        "FOREX_BOT_PROP_HOLDOUT_YEARS": "3",
        "FOREX_BOT_PROP_HOLDOUT_REQUIRED": "1",
        "FOREX_BOT_PROP_HOLDOUT_FRACTION": "0",
        "FOREX_BOT_PROP_ELITE_FILTER": "1",
        "FOREX_BOT_PROP_REQUIRE_FORWARD_PASS": "1",
        "FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS": "6",
        "FOREX_BOT_PROP_HOLDOUT_MAX_DD": "0.03",
        "FOREX_BOT_PROP_REQUIRE_ALL_TFS": "1",
        "FOREX_BOT_PROP_KEEP_MIN_SHARPE": "1.20",
        "FOREX_BOT_PROP_KEEP_MIN_WIN_RATE": "0.52",
        "FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR": "1.30",
        "FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT": "0.01",
        "FOREX_BOT_PROP_SEARCH_MAX_ROWS": "0",
        "FOREX_BOT_RUST_BASE_SIGNAL_MODE": "auto",
        "FOREX_BOT_BASE_SIGNAL_ALLOW_PY_MIXER": "0",
        "FOREX_BOT_BASE_SIGNAL_ALLOW_CLASSIC_FALLBACK": "0",
        "FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK": "0",
        "FOREX_BOT_DISCOVERY_RUST_ONLY": "1",
        "FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK": "0",
        "FOREX_BOT_TALIB_ALLOW_PY_FALLBACK": "0",
        "FOREX_BOT_PROP_PY_FALLBACK": "0",
        "FOREX_BOT_PROP_ALLOW_PY_RESCORING": "0",
        "FOREX_BOT_PROP_ALLOW_PY_EXPANSION": "0",
        "FOREX_BOT_STOP_TARGET_ALLOW_PY_FALLBACK": "0",
    },
}


def apply_runtime_profile_defaults(profile: str | None = None) -> str:
    """
    Apply grouped runtime defaults so users can set one profile instead of many env vars.

    Resolution order:
    1) explicit `profile` argument
    2) FOREX_BOT_RUNTIME_PROFILE
    3) FOREX_BOT_PROFILE (legacy alias)
    """
    raw = profile
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get("FOREX_BOT_RUNTIME_PROFILE")
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get("FOREX_BOT_PROFILE")

    mode = str(raw or "").strip().lower().replace("-", "_")
    mode = _RUNTIME_PROFILE_ALIASES.get(mode, mode)
    if mode in {"", "off", "none", "default", "auto"}:
        return ""

    preset = _RUNTIME_PROFILE_PRESETS.get(mode)
    if not preset:
        return ""

    for key, value in preset.items():
        os.environ.setdefault(key, value)

    # CPU/thread defaults tuned per profile so Rust workers don't collapse to single-core.
    cpu_total = max(1, os.cpu_count() or 1)
    if mode == "rust_32gb":
        cpu_budget = max(1, min(cpu_total - 1, 8))
    else:
        cpu_budget = max(1, cpu_total - 1)
    os.environ.setdefault("FOREX_BOT_DISCOVERY_CPU_BUDGET", str(cpu_budget))
    os.environ.setdefault("FOREX_BOT_PROP_SEARCH_WORKERS", str(cpu_budget))
    os.environ.setdefault("RAYON_NUM_THREADS", str(cpu_budget))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_budget))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_budget))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_budget))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_budget))
    os.environ.setdefault("FOREX_BOT_RUNTIME_PROFILE", mode)
    return mode


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: object, field_name: str) -> tuple[object, str, bool]:
        return super().get_field_value(field, field_name)

    def __call__(self) -> dict[str, object]:
        config_file = os.environ.get("CONFIG_FILE", "config.yaml")
        if not os.path.exists(config_file):
            return {}
        with open(config_file) as f:
            return yaml.safe_load(f) or {}

class SystemConfig(BaseModel):
    symbol: str = "EURUSD"
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    data_dir: str = "data"
    indices_path: str = ""
    use_online_indices: bool = False
    mt5_dxy_symbol: str = "USDX"
    mt5_eur_symbol: str = "EXY"
    base_timeframe: str = "M1"
    multi_resolution_enabled: bool = True
    multi_resolution_timeframes: list[str] = Field(default_factory=_default_all_timeframes)
    multi_resolution_prefix_base: bool = False
    use_volume_features: bool = True
    higher_timeframes: list[str] = Field(default_factory=_default_all_timeframes)
    required_timeframes: list[str] = Field(default_factory=_default_all_timeframes)
    enable_level2: bool = False
    level2_depth_levels: int = 10
    history_years: int = 10
    trading_session_start: str = "00:05"
    trading_session_end: str = "23:55"
    session_timezone: str = "UTC"
    broker_backend: str = "mt5_local"
    mt5_required: bool = Field(default_factory=lambda: os.name == "nt")
    mt5_terminal_path: str = ""
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_timeout_seconds: int = 30
    mt5_timezone_offset_hours: int = 0
    poll_interval_seconds: int = 60
    metrics_logging_enabled: bool = True
    metrics_db_path: str = "metrics.sqlite"
    risk_ledger_enabled: bool = True
    risk_ledger_max_events: int = 1000
    strategy_ledger_path: str = "strategy_ledger.sqlite"
    enable_dashboard: bool = True
    cache_enabled: bool = True
    cache_dir: str = "cache"
    cache_max_age_minutes: int = 60
    deep_purge_mode: str = "off"
    deep_purge_on_train: bool = True
    n_jobs: int = 0
    enable_gpu_preference: str = "auto"
    discovery_auto_cap: bool = True
    discovery_max_rows: int = 0
    discovery_stream: bool = False
    enable_gpu: bool = False
    num_gpus: int = 0
    device: str = "cpu"
    evo_multiproc_per_gpu: bool = True
    cache_training_frames: bool = False
    training_cache_max_bytes: int = 2_000_000_000
    max_training_rows_per_tf: int = 0
    downcast_training_float32: bool = True
    parquet_memory_map: bool = True
    smc_freshness_limit: int = 0
    smc_atr_displacement: float = 0.0
    smc_max_levels: int = 0
    smc_use_cuda: bool = False

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, v: int) -> int:
        cpu_total = os.cpu_count() or 1
        if v <= 0:
            return max(1, cpu_total - 1)
        return max(1, min(v, cpu_total))

    @field_validator("poll_interval_seconds")
    @classmethod
    def validate_poll(cls, v: int) -> int:
        return max(1, v)

class RiskConfig(BaseModel):
    initial_balance: float = 10_000.0
    monthly_profit_target_pct: float = 0.04
    min_risk_per_trade: float = 0.0
    max_risk_per_trade: float = 0.030
    risk_per_trade: float = 0.030
    daily_drawdown_limit: float = 0.04
    total_drawdown_limit: float = 0.07
    min_risk_reward: float = 2.0
    min_edge_cost_multiple: float = 3.0
    spread_guard_multiplier: float = 2.5
    slippage_guard_multiplier: float = 2.0
    challenge_mode: bool = False
    challenge_phase: str = "phase_1"
    challenge_target_return_pct: float = 0.10
    challenge_target_trading_days: int = 44
    challenge_progress_tolerance_pct: float = 0.01
    challenge_progress_boost_mult: float = 1.08
    challenge_progress_reduce_mult: float = 0.85
    drawdown_soft_brake_1_fraction: float = 0.50
    drawdown_soft_brake_2_fraction: float = 0.75
    drawdown_soft_brake_1_mult: float = 0.60
    drawdown_soft_brake_2_mult: float = 0.35
    drawdown_pre_stop_fraction: float = 0.90
    prop_firm_rules: bool = True
    max_daily_risk_pct: float = 0.04
    base_risk_per_trade: float = 0.03
    daily_risk_budget: float = 0.040
    consistency_tracking: bool = True
    min_confidence_threshold: float = 0.55
    dynamic_confidence_enabled: bool = True
    dynamic_confidence_vol_sensitivity: float = 0.15
    dynamic_confidence_min: float = 0.50
    dynamic_confidence_max: float = 0.90
    max_ensemble_disagreement: float = 0.20
    kill_zones_enabled: bool = True
    enhanced_features: bool = True
    uncertainty_quantification: bool = True
    max_trades_per_day: int = 8
    max_trades_per_session: int = 3
    correlation_filter_enabled: bool = True
    max_correlated_positions: int = 1
    daily_profit_stop_pct: float = 0.0
    recovery_mode_enabled: bool = True
    feature_drift_threshold: float = 0.30
    high_quality_confidence: float = 0.65
    high_quality_risk_pct: float = 0.030
    high_quality_rr: float = 2.0
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    chandelier_enabled: bool = True
    chandelier_period: int = 22
    chandelier_atr_multiplier: float = 3.0
    triple_barrier_max_bars: int = 35
    trailing_enabled: bool = True
    trailing_atr_multiplier: float = 1.0
    trailing_be_trigger_r: float = 1.0
    time_stop_bars: int = 8
    partial_take_profit_enabled: bool = True
    partial_tp_r_levels: str = "1.0,2.0,3.0"
    partial_tp_size_fracs: str = "0.5,0.25,0.25"
    partial_tp_min_total_lot: float = 0.03
    entry_patience_enabled: bool = True
    entry_patience_bars: int = 3
    entry_patience_pullback_atr: float = 0.20
    volatility_targeting_enabled: bool = True
    volatility_target: float = 0.0015
    volatility_target_min_scale: float = 0.35
    volatility_target_max_scale: float = 1.30
    regime_transition_size_multiplier: float = 0.5
    block_night_session: bool = True
    night_block_start_utc: int = 0
    night_block_end_utc: int = 6
    night_min_volatility: float = 0.0008
    session_london_confidence_threshold: float = 0.55
    session_asia_confidence_threshold: float = 0.70
    session_newyork_confidence_threshold: float = 0.58
    kelly_lambda: float = 1.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    backtest_spread_pips: float = 1.5
    cost_penalty_r: float = 0.0
    gate_trade_prob: float = 0.55
    daily_hard_stop_pct: float = 0.04
    conformal_enabled: bool = True
    conformal_alpha: float = 0.10
    conformal_abstain_min_set_size: int = 3
    volatility_stop_sigma: float = 0.02
    volatility_lookback: int = 50
    meta_label_tp_pips: float | None = None
    meta_label_sl_pips: float | None = None
    meta_label_max_hold_bars: int = 100
    meta_label_min_prob_threshold: float = 0.55
    meta_label_min_dist: float = 0.0005
    meta_label_fixed_sl: float = 0.0020
    meta_label_fixed_tp: float = 0.0040
    stop_target_mode: str = "blend"
    structure_lookback_bars: int = 120
    structure_swing_window: int = 2
    structure_min_atr_mult: float = 0.8
    structure_max_atr_mult: float = 4.0
    vol_estimator: str = "ensemble"
    vol_ensemble_weights: dict[str, float] = Field(default_factory=lambda: {"yang_zhang": 1.0,"garman_klass": 1.0,"rogers_satchell": 1.0,"parkinson": 1.0})
    vol_ensemble_weights_trend: dict[str, float] | None = None
    vol_ensemble_weights_range: dict[str, float] | None = None
    vol_ensemble_weights_neutral: dict[str, float] | None = None
    vol_window: int = 50
    ewma_lambda: float = 0.94
    ewma_lambda_by_timeframe: dict[str, float] = Field(
        default_factory=lambda: {
            "M1": 0.90,
            "M2": 0.905,
            "M3": 0.91,
            "M4": 0.915,
            "M5": 0.92,
            "M6": 0.925,
            "M10": 0.935,
            "M12": 0.938,
            "M15": 0.94,
            "M20": 0.945,
            "M30": 0.95,
            "H1": 0.96,
            "H2": 0.965,
            "H3": 0.968,
            "H4": 0.97,
            "H6": 0.974,
            "H8": 0.977,
            "H12": 0.98,
            "D1": 0.985,
            "W1": 0.99,
            "MN1": 0.995,
        }
    )
    vol_horizon_bars: int = 5
    tail_window: int = 100
    tail_alpha: float = 0.975
    tail_step: int = 5
    tail_max_bars: int = 300_000
    stop_k_vol: float = 1.0
    stop_k_tail: float = 1.25
    rr_trend: float = 2.5
    rr_range: float = 1.5
    rr_neutral: float = 2.0
    regime_adx_trend: float = 25.0
    regime_adx_range: float = 20.0
    hurst_window: int = 100
    hurst_trend: float = 0.55
    hurst_range: float = 0.45

    @field_validator("risk_per_trade", "max_risk_per_trade", "daily_drawdown_limit", "total_drawdown_limit")
    @classmethod
    def validate_risk_pct(cls, v: float) -> float:
        return max(0.0, min(v, 0.10))

    @field_validator("min_risk_reward")
    @classmethod
    def validate_rr(cls, v: float) -> float:
        return max(1.5, v)

class ModelsConfig(BaseModel):
    ml_models: list[str] = Field(
        default_factory=lambda: [
            "lightgbm",
            "xgboost",
            "xgboost_rf",
            "xgboost_dart",
            "catboost",
            "catboost_alt",
            "mlp",
            "elasticnet",
            "bayes_logit",
            "online_pa",
            "online_hoeffding",
        ]
    )
    use_rl_agent: bool = False
    use_sac_agent: bool = False
    use_rllib_agent: bool = False
    train_all_registered_models: bool = False
    ensure_linear_anchors: bool = True
    online_learners_enabled: bool = True
    rllib_num_workers: int = 0
    auto_enable_rllib: bool = True
    use_neuroevolution: bool = False
    rl_population_size: int = 5
    rl_timesteps: int = 10000000
    rl_eval_episodes: int = 15
    rl_network_arch: list[int] = Field(default_factory=lambda: [4096, 4096, 4096, 2048, 1024])
    rl_parallel_envs: int = 1
    rl_train_seconds: int = 3600
    evo_train_seconds: int = 3600
    evo_hidden_size: int = 64
    evo_population: int = 32
    evo_islands: int = 4
    prop_search_enabled: bool = False
    prop_search_population: int = 100
    prop_search_generations: int = 20
    prop_search_max_hours: float = 2.0
    prop_search_max_rows: int = 0
    prop_search_max_rows_by_tf: dict[str, int] = Field(default_factory=dict)
    prop_search_portfolio_size: int = 4
    # 0 = allow full indicator set (no hard cap)
    prop_search_max_indicators: int = 0
    prop_search_checkpoint: str = "models/strategy_evo_checkpoint.json"
    prop_search_device: str = "cpu"
    evogp_enabled: bool = True
    evogp_population: int = 4096
    evogp_generations: int = 80
    evogp_eval_candidates: int = 3000
    prop_search_train_years: int = 0
    prop_search_val_years: int = 0
    prop_search_val_candidates: int = 0
    prop_search_val_min_positive_months: int = 0
    prop_search_val_min_trades_per_month: int = 0
    prop_search_val_min_trades_per_day: float = 0.0
    prop_search_val_min_monthly_profit_pct: float = 0.0
    prop_search_val_log_trades: bool = False
    prop_search_val_trade_log_max: int = 20
    prop_search_holdout_fraction: float = 0.20
    prop_search_holdout_min_rows: int = 8000
    prop_search_holdout_min_sharpe: float = 1.20
    prop_search_holdout_min_win_rate: float = 0.52
    prop_search_holdout_min_profit_factor: float = 1.30
    prop_search_holdout_min_trades: int = 20
    prop_search_holdout_years: float = 0.0
    prop_search_holdout_min_truth_probability: float = 0.0
    prop_search_holdout_required: bool = False
    prop_search_async: bool = False
    prop_search_async_wait: bool = False
    tree_device_preference: str = "auto"
    prop_search_opportunistic_enabled: bool = True
    prop_search_opportunistic_min_positive_months: int = 3
    prop_search_opportunistic_min_trades_per_month: int = 10
    prop_search_opportunistic_min_trade_return_pct: float = 4.0
    prop_search_opportunistic_max_dd: float = 0.025
    prop_search_use_opportunistic: bool = True
    train_batch_size: int = 32
    inference_batch_size: int = 32
    enable_transformer_expert: bool = True
    transformer_heads: int = 16
    transformer_layers: int = 12
    transformer_hidden_dim: int = 1200
    transformer_dropout: float = 0.20
    transformer_seq_len: int = 64
    transformer_train_seconds: int = 3600
    nbeats_train_seconds: int = 3600
    tide_train_seconds: int = 3600
    tabnet_train_seconds: int = 3600
    kan_train_seconds: int = 3600
    mlp_train_seconds: int = 3600
    num_transformers: int = 2
    hpo_backend: str = "ax"
    hpo_trials: int = 8
    hpo_trials_by_model: dict[str, int] = Field(
        default_factory=lambda: {
            "lightgbm": 8,
            "xgboost": 8,
            "xgboost_rf": 6,
            "xgboost_dart": 6,
            "catboost": 8,
            "catboost_alt": 6,
            "mlp": 6,
            "tabnet": 6,
            "nbeats": 6,
            "tide": 6,
            "kan": 6,
            "transformer": 6,
        }
    )
    hpo_max_rows: int = 1_000_000
    max_epochs_by_model: dict[str, int] = Field(default_factory=dict)
    ray_tune_max_concurrency: int = 1
    export_onnx: bool = False
    filter_to_base_signal: bool = True
    global_max_rows: int = 0
    global_max_rows_per_symbol: int = 0
    symbol_hash_buckets: int = 32
    global_train_ratio: float = 0.8
    train_holdout_pct: float = 0.2
    walkforward_splits: int = 20
    embargo_minutes: int = 120
    prop_metric_weight: float = 1.0
    prop_accuracy_weight: float = 0.1
    prop_min_trades: int = 0
    prop_conf_threshold: float = 0.55
    enable_cpcv: bool = True
    cpcv_n_splits: int = 5
    cpcv_n_test_groups: int = 2
    cpcv_embargo_pct: float = 0.01
    cpcv_purge_pct: float = 0.02
    cpcv_min_phi: float = 0.80
    cpcv_max_rows: int = 0
    enable_ddp: bool = False
    enable_fsdp: bool = False
    ddp_world_size: int = 1
    transformer_d_model: int = 256
    transformer_n_heads: int = 8
    transformer_n_layers: int = 4
    nf_hidden_dim: int = 256
    tide_hidden_dim: int = 256
    nbeats_hidden_dim: int = 256
    kan_hidden_dim: int = 256
    tabnet_hidden_dim: int = 64
    phase5_filter_meta_blender: bool = True
    phase5_core_models: list[str] = Field(default_factory=lambda: ["transformer", "nbeats", "tide", "tabnet", "kan"])
    phase5_diversity_filter: bool = True
    phase5_model_corr_max: float = 0.85
    adversarial_validation_enabled: bool = True
    adversarial_validation_max_rows: int = 200_000
    adversarial_validation_alert_acc: float = 0.55
    calibration_enabled: bool = True
    calibration_method: str = "platt"
    calibration_min_rows: int = 300
    lightgbm_monotone_constraints_enabled: bool = True
    lightgbm_monotone_constraints: dict[str, int] = Field(default_factory=dict)
    regime_router_enabled: bool = True
    regime_router_min_models: int = 2
    regime_trend_models: list[str] = Field(
        default_factory=lambda: ["transformer", "patchtst", "timesnet", "nbeats", "nbeatsx_nf", "tide", "tide_nf"]
    )
    regime_range_models: list[str] = Field(
        default_factory=lambda: [
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
    )
    regime_neutral_models: list[str] = Field(default_factory=list)
    l1_feature_selection_enabled: bool = True
    l1_feature_selection_per_regime: bool = True
    l1_feature_selection_min_features: int = 20
    l1_feature_selection_max_features: int = 256
    l1_feature_selection_sample_limit: int = 200_000
    l1_feature_selection_c: float = 0.20

    @field_validator("train_batch_size", "inference_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        return max(1, v)

class NewsConfig(BaseModel):
    news_decay_minutes: int = 120
    news_kill_window_min: int = 5
    news_confidence_threshold: float = 0.65
    news_lookahead_minutes: int = 60
    news_pre_event_seconds: int = 300
    news_post_event_seconds: int = 300
    news_trade_on_event: bool = False
    news_trade_confidence_threshold: float = 0.90
    enable_news: bool = True
    enable_llm_helper: bool = True
    llm_helper_enabled: bool = True
    llm_sentiment_positive_threshold: float = 0.2
    llm_sentiment_negative_threshold: float = -0.2
    news_backfill_enabled: bool = True
    news_backfill_days: int = 30
    news_local_glob: str = ""
    openai_model: str = "gpt-5-nano-2025-08-07"
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_max_tokens: int = 256
    openai_max_events_per_fetch: int = 50
    openai_news_enabled: bool = True
    perplexity_enabled: bool = True
    perplexity_api_key_env: str = "PPLX_API_KEY"
    perplexity_model: str = "sonar"
    perplexity_num_results: int = 10
    perplexity_timeframe_hours: int = 24
    strategist_enabled: bool = False
    strategist_interval_minutes: int = 30
    auto_rescore_enabled: bool = False
    auto_rescore_days: int = 30
    auto_rescore_max_events: int = 200
    auto_rescore_only_missing: bool = True

class Settings(BaseSettings):
    system: SystemConfig = Field(default_factory=SystemConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    secrets_file: str = "keys.txt"
    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=None, extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, YamlConfigSettingsSource(settings_cls), env_settings, file_secret_settings)

# Apply preset env defaults before constructing the singleton settings object.
apply_runtime_profile_defaults()
settings = Settings()
