from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from forex_bot.strategy import evo_prop


def test_run_evo_search_rust_path_handles_datetime_index_view_ndarray(tmp_path, monkeypatch):
    """
    Regression test for pandas DatetimeIndex conversion in Rust prop-search path.

    Older code assumed `idx.view("int64")` always had `.to_numpy()`, which is false
    on some pandas versions where it is already an ndarray.
    """
    monkeypatch.chdir(tmp_path)

    # Force Rust path and stub the binding call.
    monkeypatch.setattr(evo_prop, "_RUST_SEARCH", True, raising=False)
    monkeypatch.setattr(evo_prop, "ALL_INDICATORS", ["RSI"], raising=False)

    class _DummyBindings:
        @staticmethod
        def search_evolve_ohlcv(
            open_,
            high,
            low,
            close,
            ts,
            volume,
            population,
            generations,
            max_indicators,
            use_gpu,
        ):
            # Assert timestamp conversion succeeded and produced unix ms int64.
            assert ts is not None
            assert isinstance(ts, np.ndarray)
            assert ts.dtype == np.int64
            assert ts.ndim == 1
            return {
                "feature_names": ["RSI"],
                "genes": [
                    {
                        "indices": [0],
                        "weights": [1.0],
                        "fitness": 1.25,
                        "trades": 25,
                        "max_dd_pct": 0.02,
                        "long_threshold": 0.66,
                        "short_threshold": -0.66,
                        "combination_method": "weighted_vote",
                    }
                ],
            }

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)

    # If Rust fast path breaks and falls back, this should trip loudly.
    class _FailMixer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Should not fall back to Python TALibStrategyMixer path.")

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _FailMixer, raising=False)

    idx = pd.date_range("2025-01-01", periods=128, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.5, len(idx), dtype=np.float64),
            "high": np.linspace(1.1, 1.6, len(idx), dtype=np.float64),
            "low": np.linspace(0.9, 1.4, len(idx), dtype=np.float64),
            "close": np.linspace(1.0, 1.5, len(idx), dtype=np.float64),
            "volume": np.ones(len(idx), dtype=np.float64),
        },
        index=idx,
    )
    df.attrs["symbol"] = "EURUSD"
    df.attrs["timeframe"] = "M1"

    settings = SimpleNamespace(
        risk=SimpleNamespace(total_drawdown_limit=0.07),
        models=SimpleNamespace(prop_search_portfolio_size=50, prop_search_max_indicators=4),
    )

    checkpoint = tmp_path / "strategy_evo_checkpoint.json"
    evo_prop.run_evo_search(
        df=df,
        settings=settings,
        population=10,
        generations=2,
        checkpoint=str(checkpoint),
        max_hours=0.1,
        actual_balance=10_000.0,
    )

    assert checkpoint.exists()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["symbol"] == "EURUSD"
    assert payload["timeframe"] == "M1"
    assert len(payload.get("best_genes", [])) >= 1
