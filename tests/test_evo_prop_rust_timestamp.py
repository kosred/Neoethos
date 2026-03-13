from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

from forex_bot.strategy import evo_prop


def _make_df(rows: int = 64) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "high": np.linspace(1.01, 1.11, len(idx), dtype=np.float64),
            "low": np.linspace(0.99, 1.09, len(idx), dtype=np.float64),
            "close": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "volume": np.ones(len(idx), dtype=np.float64),
        },
        index=idx,
    )
    df.attrs["symbol"] = "EURUSD"
    df.attrs["timeframe"] = "M1"
    return df


def _settings():
    return SimpleNamespace(
        risk=SimpleNamespace(total_drawdown_limit=0.07),
        models=SimpleNamespace(prop_search_portfolio_size=10, prop_search_max_indicators=2),
    )


def test_run_evo_search_rust_path_handles_datetime_index_view_ndarray(tmp_path, monkeypatch):
    """
    Regression test for pandas DatetimeIndex conversion in Rust prop-search path.

    Older code assumed `idx.view("int64")` always had `.to_numpy()`, which is false
    on some pandas versions where it is already an ndarray.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_REQUIRED", "0")

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


def test_run_evo_search_rust_only_skips_python_fallback_when_backend_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_REQUIRED", "0")
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setattr(evo_prop, "_RUST_SEARCH", False, raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_GPU_SEARCH", False, raising=False)
    monkeypatch.setattr(evo_prop, "_fb", None, raising=False)

    class _FailMixer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Python TALibStrategyMixer fallback should be disabled in rust-only mode.")

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _FailMixer, raising=False)

    idx = pd.date_range("2025-01-01", periods=64, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "high": np.linspace(1.01, 1.11, len(idx), dtype=np.float64),
            "low": np.linspace(0.99, 1.09, len(idx), dtype=np.float64),
            "close": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "volume": np.ones(len(idx), dtype=np.float64),
        },
        index=idx,
    )
    df.attrs["symbol"] = "EURUSD"
    df.attrs["timeframe"] = "M1"

    settings = SimpleNamespace(
        risk=SimpleNamespace(total_drawdown_limit=0.07),
        models=SimpleNamespace(prop_search_portfolio_size=20, prop_search_max_indicators=4),
    )

    checkpoint = tmp_path / "strategy_evo_checkpoint.json"
    evo_prop.run_evo_search(
        df=df,
        settings=settings,
        population=8,
        generations=2,
        checkpoint=str(checkpoint),
        max_hours=0.1,
        actual_balance=10_000.0,
    )

    assert not checkpoint.exists()


def test_rust_only_skips_rescoring_and_expansion(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setattr(evo_prop, "_RUST_SEARCH", True, raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_GPU_SEARCH", False, raising=False)

    class _DummyBindings:
        @staticmethod
        def search_evolve_ohlcv(open_, high, low, close, ts, volume, population, generations, max_indicators, include_raw):
            assert ts is not None
            return {
                "feature_names": ["RSI"],
                "genes": [
                    {
                        "indices": [0],
                        "weights": [1.0],
                        "fitness": 1.0,
                        "trades": 10,
                        "max_dd_pct": 0.02,
                        "long_threshold": 0.6,
                        "short_threshold": -0.6,
                        "combination_method": "weighted_vote",
                    }
                ],
            }

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)

    class _FailMixer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Rescoring should be skipped in rust-only mode.")

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _FailMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_expand_threshold_variants", lambda **_: (_ for _ in ()).throw(AssertionError("Expansion should be skipped")), raising=False)

    idx = pd.date_range("2025-01-01", periods=64, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "high": np.linspace(1.01, 1.11, len(idx), dtype=np.float64),
            "low": np.linspace(0.99, 1.09, len(idx), dtype=np.float64),
            "close": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "volume": np.ones(len(idx), dtype=np.float64),
        },
        index=idx,
    )
    df.attrs["symbol"] = "EURUSD"
    df.attrs["timeframe"] = "M1"

    settings = SimpleNamespace(
        risk=SimpleNamespace(total_drawdown_limit=0.07),
        models=SimpleNamespace(prop_search_portfolio_size=10, prop_search_max_indicators=2),
    )

    checkpoint = tmp_path / "strategy_evo_checkpoint.json"
    evo_prop.run_evo_search(
        df=df,
        settings=settings,
        population=4,
        generations=1,
        checkpoint=str(checkpoint),
        max_hours=0.05,
        actual_balance=5_000.0,
    )

    assert checkpoint.exists()


def test_non_rust_default_skips_python_rescore_and_expansion(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_PY_FALLBACK", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_REQUIRED", "0")
    monkeypatch.delenv("FOREX_BOT_PROP_ALLOW_PY_RESCORING", raising=False)
    monkeypatch.delenv("FOREX_BOT_PROP_ALLOW_PY_EXPANSION", raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_SEARCH", True, raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_GPU_SEARCH", False, raising=False)

    class _DummyBindings:
        @staticmethod
        def search_evolve_ohlcv(open_, high, low, close, ts, volume, population, generations, max_indicators, include_raw):
            return {
                "feature_names": ["RSI"],
                "genes": [
                    {
                        "indices": [0],
                        "weights": [1.0],
                        "fitness": 1.0,
                        "trades": 10,
                        "max_dd_pct": 0.02,
                        "long_threshold": 0.6,
                        "short_threshold": -0.6,
                        "combination_method": "weighted_vote",
                    }
                ],
            }

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    calls = {"mixer_init": 0, "expand": 0}

    class _CountingMixer:
        def __init__(self, *args, **kwargs):
            calls["mixer_init"] += 1
            self.available_indicators = []

    def _expand(**kwargs):
        calls["expand"] += 1
        return list(kwargs.get("genes") or [])

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _CountingMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_expand_threshold_variants", _expand, raising=False)

    checkpoint = tmp_path / "strategy_evo_checkpoint.json"
    evo_prop.run_evo_search(
        df=_make_df(64),
        settings=_settings(),
        population=4,
        generations=1,
        checkpoint=str(checkpoint),
        max_hours=0.05,
        actual_balance=5_000.0,
    )

    assert checkpoint.exists()
    # Python journal/eval fallbacks are disabled by default in rust-first mode.
    assert calls["mixer_init"] == 0
    assert calls["expand"] == 0


def test_non_rust_opt_in_still_skips_python_rescore_and_expansion(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_PY_FALLBACK", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_REQUIRED", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_ALLOW_PY_RESCORING", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_ALLOW_PY_EXPANSION", "1")
    monkeypatch.setattr(evo_prop, "_RUST_SEARCH", True, raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_GPU_SEARCH", False, raising=False)

    class _DummyBindings:
        @staticmethod
        def search_evolve_ohlcv(open_, high, low, close, ts, volume, population, generations, max_indicators, include_raw):
            return {
                "feature_names": ["RSI"],
                "genes": [
                    {
                        "indices": [0],
                        "weights": [1.0],
                        "fitness": 1.0,
                        "trades": 10,
                        "max_dd_pct": 0.02,
                        "long_threshold": 0.6,
                        "short_threshold": -0.6,
                        "combination_method": "weighted_vote",
                    }
                ],
            }

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    calls = {"mixer_init": 0, "expand": 0}

    class _CountingMixer:
        def __init__(self, *args, **kwargs):
            calls["mixer_init"] += 1
            self.available_indicators = []

    def _expand(**kwargs):
        calls["expand"] += 1
        return list(kwargs.get("genes") or [])

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _CountingMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_expand_threshold_variants", _expand, raising=False)

    checkpoint = tmp_path / "strategy_evo_checkpoint.json"
    evo_prop.run_evo_search(
        df=_make_df(64),
        settings=_settings(),
        population=4,
        generations=1,
        checkpoint=str(checkpoint),
        max_hours=0.05,
        actual_balance=5_000.0,
    )

    assert checkpoint.exists()
    # rust_ga path does not execute EvoGP GPU rescoring; with Python journal fallback
    # disabled we should not initialize mixer here.
    assert calls["mixer_init"] == 0
    assert calls["expand"] == 0


def test_evo_prop_safe_indices_uses_rust_binding(monkeypatch):
    calls = {"derive": 0}

    class _DummyBindings:
        @staticmethod
        def derive_time_index_arrays(idx_ns):
            calls["derive"] += 1
            n = int(np.asarray(idx_ns, dtype=np.int64).shape[0])
            return (
                np.arange(n, dtype=np.int64),
                np.full(n, 66, dtype=np.int64),
                np.full(n, 9902, dtype=np.int64),
            )

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    month_idx, day_idx = evo_prop._safe_indices(
        np.array(
            [
                1_704_067_200_000_000_000,
                1_704_067_260_000_000_000,
                1_704_067_320_000_000_000,
            ],
            dtype=np.int64,
        ),
        3,
    )

    assert calls["derive"] == 1
    np.testing.assert_array_equal(month_idx, np.array([66, 66, 66], dtype=np.int64))
    np.testing.assert_array_equal(day_idx, np.array([9902, 9902, 9902], dtype=np.int64))


def test_py_fallback_disabled_skips_python_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_PY_FALLBACK", "0")
    monkeypatch.setattr(evo_prop, "_RUST_SEARCH", False, raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_GPU_SEARCH", False, raising=False)
    monkeypatch.setattr(evo_prop, "_fb", None, raising=False)

    class _FailMixer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Python fallback should be disabled when FOREX_BOT_PROP_PY_FALLBACK!=1.")

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _FailMixer, raising=False)

    idx = pd.date_range("2025-01-01", periods=64, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "high": np.linspace(1.01, 1.11, len(idx), dtype=np.float64),
            "low": np.linspace(0.99, 1.09, len(idx), dtype=np.float64),
            "close": np.linspace(1.0, 1.1, len(idx), dtype=np.float64),
            "volume": np.ones(len(idx), dtype=np.float64),
        },
        index=idx,
    )
    df.attrs["symbol"] = "EURUSD"
    df.attrs["timeframe"] = "M1"

    settings = SimpleNamespace(
        risk=SimpleNamespace(total_drawdown_limit=0.07),
        models=SimpleNamespace(prop_search_portfolio_size=10, prop_search_max_indicators=2),
    )

    checkpoint = tmp_path / "strategy_evo_checkpoint.json"
    evo_prop.run_evo_search(
        df=df,
        settings=settings,
        population=4,
        generations=1,
        checkpoint=str(checkpoint),
        max_hours=0.05,
        actual_balance=5_000.0,
    )

    assert not checkpoint.exists()

