import os

from forex_bot.core import config as cfg
from forex_bot.core.config import apply_runtime_profile_defaults


def _clear_profile_env(monkeypatch):
    for preset in cfg._RUNTIME_PROFILE_PRESETS.values():
        for key in preset:
            monkeypatch.delenv(key, raising=False)
    for key in (
        "FOREX_BOT_DISCOVERY_CPU_BUDGET",
        "FOREX_BOT_PROP_SEARCH_WORKERS",
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "NUMEXPR_NUM_THREADS",
        "FOREX_BOT_PROP_ELITE_FILTER",
        "FOREX_BOT_PROP_REQUIRE_FORWARD_PASS",
        "FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS",
        "FOREX_BOT_PROP_HOLDOUT_MAX_DD",
        "FOREX_BOT_PROP_REQUIRE_ALL_TFS",
        "FOREX_BOT_PROP_KEEP_MIN_SHARPE",
        "FOREX_BOT_PROP_KEEP_MIN_WIN_RATE",
        "FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR",
        "FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("FOREX_BOT_PROFILE", raising=False)


def test_runtime_profile_sets_rust_defaults(monkeypatch):
    _clear_profile_env(monkeypatch)

    resolved = apply_runtime_profile_defaults("rust_32gb")

    assert resolved == "rust_32gb"
    assert os.environ.get("FOREX_BOT_RUNTIME_PROFILE") == "rust_32gb"
    assert os.environ.get("FOREX_BOT_RUST_ONLY") == "1"
    assert os.environ.get("FOREX_BOT_TREE_BACKEND") == "rust_strict"
    assert os.environ.get("FOREX_BOT_TREE_RUST_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_FEATURES_BACKEND") == "rust_strict"
    assert os.environ.get("FOREX_BOT_FRAME_IO_BACKEND") is None
    assert os.environ.get("FOREX_BOT_USE_ALL_TIMEFRAMES") == "1"
    assert os.environ.get("FOREX_BOT_PROP_SEARCH_TRAIN_YEARS") == "10"
    assert os.environ.get("FOREX_BOT_PROP_HOLDOUT_YEARS") == "3"
    assert os.environ.get("FOREX_BOT_PROP_HOLDOUT_REQUIRED") == "1"
    assert os.environ.get("FOREX_BOT_PROP_ELITE_FILTER") == "1"
    assert os.environ.get("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS") == "1"
    assert os.environ.get("FOREX_BOT_PROP_REQUIRE_ALL_TFS") == "1"
    assert os.environ.get("FOREX_BOT_BASE_SIGNAL_ALLOW_PY_MIXER") == "0"
    assert os.environ.get("FOREX_BOT_BASE_SIGNAL_ALLOW_CLASSIC_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_TALIB_ALLOW_PY_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_PROP_PY_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_PROP_ALLOW_PY_RESCORING") == "0"
    assert os.environ.get("FOREX_BOT_PROP_ALLOW_PY_EXPANSION") == "0"
    assert os.environ.get("FOREX_BOT_STOP_TARGET_ALLOW_PY_FALLBACK") == "0"
    assert int(os.environ.get("FOREX_BOT_DISCOVERY_CPU_BUDGET", "0") or 0) >= 1
    assert int(os.environ.get("FOREX_BOT_PROP_SEARCH_WORKERS", "0") or 0) >= 1
    assert int(os.environ.get("RAYON_NUM_THREADS", "0") or 0) >= 1
    assert int(os.environ.get("NUMEXPR_MAX_THREADS", "0") or 0) >= 1
    assert os.environ.get("NUMEXPR_MAX_THREADS") == os.environ.get("NUMEXPR_NUM_THREADS")


def test_runtime_profile_does_not_override_explicit_env(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("FOREX_BOT_RUST_MAX_FEATURES", "77")
    monkeypatch.setenv("RAYON_NUM_THREADS", "3")
    monkeypatch.setenv("NUMEXPR_MAX_THREADS", "9")
    resolved = apply_runtime_profile_defaults("rust_32gb")
    assert resolved == "rust_32gb"
    assert os.environ.get("FOREX_BOT_RUST_MAX_FEATURES") == "77"
    assert os.environ.get("RAYON_NUM_THREADS") == "3"
    assert os.environ.get("NUMEXPR_MAX_THREADS") == "9"


def test_runtime_profile_rust_32gb_does_not_cap_large_cpu_hosts(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setattr(cfg.os, "cpu_count", lambda: 32)

    resolved = apply_runtime_profile_defaults("rust_32gb")

    assert resolved == "rust_32gb"
    assert os.environ.get("FOREX_BOT_DISCOVERY_CPU_BUDGET") == "31"
    assert os.environ.get("FOREX_BOT_PROP_SEARCH_WORKERS") == "31"
    assert os.environ.get("RAYON_NUM_THREADS") == "31"


def test_runtime_profile_rust_max_enables_full_feature_search(monkeypatch):
    _clear_profile_env(monkeypatch)
    resolved = apply_runtime_profile_defaults("rust_max")
    assert resolved == "rust_max"
    assert os.environ.get("FOREX_BOT_USE_ALL_FEATURES") == "1"
    assert os.environ.get("FOREX_BOT_USE_ALL_TIMEFRAMES") == "1"
    assert os.environ.get("FOREX_BOT_RUST_ONLY") == "1"
    assert os.environ.get("FOREX_BOT_FEATURES_BACKEND") == "rust_strict"
    assert os.environ.get("FOREX_BOT_RUST_FEATURE_PROFILE") == "full"
    assert os.environ.get("FOREX_BOT_RUST_HTF_FEATURE_PROFILE") == "full"
    assert os.environ.get("FOREX_BOT_RUST_MAX_FEATURES") == "0"
    assert os.environ.get("FOREX_BOT_RUST_MAX_HTF_FEATURES") == "0"
    assert os.environ.get("FOREX_BOT_PROP_SEARCH_TRAIN_YEARS") == "10"
    assert os.environ.get("FOREX_BOT_PROP_HOLDOUT_YEARS") == "3"
    assert os.environ.get("FOREX_BOT_PROP_ELITE_FILTER") == "1"
    assert os.environ.get("FOREX_BOT_BASE_SIGNAL_ALLOW_PY_MIXER") == "0"
    assert os.environ.get("FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK") == "0"
    assert os.environ.get("FOREX_BOT_PROP_ALLOW_PY_RESCORING") == "0"
    assert os.environ.get("FOREX_BOT_PROP_ALLOW_PY_EXPANSION") == "0"
    assert os.environ.get("FOREX_BOT_STOP_TARGET_ALLOW_PY_FALLBACK") == "0"
