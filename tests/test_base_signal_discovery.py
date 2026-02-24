from __future__ import annotations

import json

import numpy as np
import pandas as pd

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
            arr[::3] = 1.0
            arr[2::3] = -1.0
            return pd.Series(arr, index=df.index)

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(talib_mixer, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery")
    monkeypatch.setenv("FOREX_BOT_PROP_SYMBOL_STRICT", "1")

    fe = FeatureEngineer(settings)
    idx = pd.date_range("2025-01-01", periods=12, freq="min", tz="UTC")
    df = _make_ohlcv(idx)

    out = fe._compute_base_signal(df, symbol="EURUSD")
    got = out["base_signal"].to_numpy(dtype=np.int8)

    expected = np.zeros(len(idx), dtype=np.int8)
    expected[::3] = 1
    expected[2::3] = -1
    np.testing.assert_array_equal(got, expected)


def test_base_signal_discovery_first_falls_back_to_legacy_when_no_artifact(monkeypatch):
    settings = Settings()
    settings.system.cache_dir = "cache_missing_for_test"
    settings.models.prop_search_checkpoint = "models/missing_checkpoint.json"

    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery_first")

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
    np.testing.assert_array_equal(out["base_signal"].to_numpy(dtype=np.int8), np.array([1, -1, 0], dtype=np.int8))
