from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

from forex_bot.core.config import Settings
from forex_bot.features.pipeline import FeatureEngineer


def _make_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.linspace(1.0, 1.2, len(index), dtype=np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.0005
    low = np.minimum(open_, close) - 0.0005
    volume = np.full(len(index), 100.0, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            # Neutral values so legacy RSI/MACD path would produce all-zero signals.
            "rsi": np.full(len(index), 50.0, dtype=np.float64),
            "macd_hist": np.zeros(len(index), dtype=np.float64),
        },
        index=index,
    )


def test_base_signal_prefers_discovered_talib_knowledge(tmp_path, monkeypatch):
    settings = Settings()
    settings.system.cache_dir = str(tmp_path / "cache")
    settings.models.prop_search_checkpoint = str(tmp_path / "models" / "strategy_evo_checkpoint.json")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    knowledge_path = cache_dir / "talib_knowledge_EURUSD.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "symbol": "EURUSD",
                "timeframe": "M1",
                "best_genes": [
                    {
                        "strategy_id": "test_gene",
                        "indicators": ["RSI"],
                        "weights": {"RSI": 1.0},
                        "long_threshold": 0.3,
                        "short_threshold": -0.3,
                        "fitness": 1.5,
                        "trades": 30,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    n = 12
    discovered = np.zeros(n, dtype=np.int8)
    discovered[::3] = 1
    discovered[2::3] = -1

    def _bulk(*_args, **_kwargs):
        return discovered.reshape(-1, 1)

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))
    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery")
    monkeypatch.setenv("FOREX_BOT_PROP_SYMBOL_STRICT", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_STRICT_FILTER", "0")

    fe = FeatureEngineer(settings)
    idx = pd.date_range("2025-01-01", periods=12, freq="min", tz="UTC")
    df = _make_ohlcv(idx)

    out = fe._compute_base_signal(df, symbol="EURUSD")
    got = out["base_signal"].to_numpy(dtype=np.int8)

    np.testing.assert_array_equal(got, discovered)


def test_base_signal_discovery_first_without_artifact_returns_neutral(monkeypatch):
    settings = Settings()
    settings.system.cache_dir = "cache_missing_for_test"
    settings.models.prop_search_checkpoint = "models/missing_checkpoint.json"

    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery_first")
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_STRICT_FILTER", "0")

    fe = FeatureEngineer(settings)
    idx = pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.1, 1.1, 1.1],
            "low": [0.9, 0.9, 0.9],
            "close": [1.0, 1.0, 1.0],
            "volume": [100.0, 100.0, 100.0],
            "rsi": [25.0, 75.0, 50.0],
            "macd_hist": [0.5, -0.5, 0.0],
        },
        index=idx,
    )

    out = fe._compute_base_signal(df, symbol="EURUSD")
    np.testing.assert_array_equal(out["base_signal"].to_numpy(dtype=np.int8), np.zeros(3, dtype=np.int8))


def test_base_signal_discovery_first_defaults_to_no_classic_fallback(monkeypatch):
    settings = Settings()
    settings.system.cache_dir = "cache_missing_for_test"
    settings.models.prop_search_checkpoint = "models/missing_checkpoint.json"

    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery_first")
    monkeypatch.delenv("FOREX_BOT_BASE_SIGNAL_ALLOW_CLASSIC_FALLBACK", raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_STRICT_FILTER", "0")

    fe = FeatureEngineer(settings)
    idx = pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.1, 1.1, 1.1],
            "low": [0.9, 0.9, 0.9],
            "close": [1.0, 1.0, 1.0],
            "volume": [100.0, 100.0, 100.0],
            "rsi": [25.0, 75.0, 50.0],
            "macd_hist": [0.5, -0.5, 0.0],
        },
        index=idx,
    )

    out = fe._compute_base_signal(df, symbol="EURUSD")
    np.testing.assert_array_equal(out["base_signal"].to_numpy(dtype=np.int8), np.zeros(3, dtype=np.int8))


def test_base_signal_discovery_uses_rust_bulk_when_talib_unavailable(monkeypatch):
    settings = Settings()
    settings.system.cache_dir = "cache_missing_for_test"
    settings.models.prop_search_checkpoint = "models/missing_checkpoint.json"

    from forex_bot.features import talib_mixer

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", False, raising=False)
    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery")
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_STRICT_FILTER", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.0")

    n = 12
    discovered = np.zeros(n, dtype=np.int8)
    discovered[::3] = 1
    discovered[2::3] = -1
    calls = {"bulk": 0}

    def _bulk(*_args, **_kwargs):
        calls["bulk"] += 1
        return discovered.reshape(-1, 1)

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))

    fe = FeatureEngineer(settings)
    gene = SimpleNamespace(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        long_threshold=0.3,
        short_threshold=-0.3,
        fitness=2.0,
    )
    monkeypatch.setattr(fe, "_load_discovered_base_signal_genes", lambda _symbol, max_genes=100: [gene])

    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    df = _make_ohlcv(idx)
    out = fe._compute_base_signal(df, symbol="EURUSD")
    got = out["base_signal"].to_numpy(dtype=np.int8)

    assert calls["bulk"] > 0
    np.testing.assert_array_equal(got, discovered)


def test_base_signal_discovery_strict_rust_skips_python_talib_fallback(monkeypatch):
    settings = Settings()
    settings.system.cache_dir = "cache_missing_for_test"
    settings.models.prop_search_checkpoint = "models/missing_checkpoint.json"

    from forex_bot.features import talib_mixer

    class _DummyMixer:
        def __init__(self, *, device: str = "cpu", use_volume_features: bool = False) -> None:
            self.available_indicators = ["RSI"]

        @staticmethod
        def bulk_calculate_indicators(df: pd.DataFrame, population) -> dict[str, pd.Series]:
            return {}

        @staticmethod
        def compute_signals(df: pd.DataFrame, gene, *, cache=None) -> pd.Series:
            arr = np.zeros(len(df), dtype=np.float64)
            arr[::2] = 1.0
            return pd.Series(arr, index=df.index)

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(talib_mixer, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery")
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.0")

    fe = FeatureEngineer(settings)
    monkeypatch.setattr(
        fe,
        "_compute_discovered_base_signal_ohlcv_numpy",
        lambda **_kwargs: np.zeros(10, dtype=np.int8),
    )
    gene = SimpleNamespace(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        long_threshold=0.3,
        short_threshold=-0.3,
        fitness=2.0,
    )
    monkeypatch.setattr(fe, "_load_discovered_base_signal_genes", lambda _symbol, max_genes=100: [gene])

    idx = pd.date_range("2025-01-01", periods=10, freq="min", tz="UTC")
    df = _make_ohlcv(idx)
    out = fe._compute_base_signal(df, symbol="EURUSD")
    got = out["base_signal"].to_numpy(dtype=np.int8)

    # In strict Rust mode we should keep Rust result (zeros) and not fallback to Python TA-Lib mixer.
    np.testing.assert_array_equal(got, np.zeros(10, dtype=np.int8))

