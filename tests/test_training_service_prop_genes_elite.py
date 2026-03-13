from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import numpy as np
from forex_bot.domain.events import PreparedDataset
from tests._compat_pd import pd

from forex_bot.features import talib_mixer


def _training_service_cls():
    # Isolate this unit test from unrelated import-time breakages in data loader modules.
    data_pkg = sys.modules.get("forex_bot.data")
    if data_pkg is None:
        data_pkg = types.ModuleType("forex_bot.data")
        data_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["forex_bot.data"] = data_pkg
    news_pkg = sys.modules.get("forex_bot.data.news")
    if news_pkg is None:
        news_pkg = types.ModuleType("forex_bot.data.news")
        news_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["forex_bot.data.news"] = news_pkg
    loader_mod = types.ModuleType("forex_bot.data.loader")

    class DataLoader:  # pragma: no cover - import shim
        pass

    loader_mod.DataLoader = DataLoader
    sys.modules["forex_bot.data.loader"] = loader_mod
    client_mod = types.ModuleType("forex_bot.data.news.client")
    client_mod.get_sentiment_analyzer = lambda: None
    sys.modules["forex_bot.data.news.client"] = client_mod

    mod = importlib.import_module("forex_bot.execution.training_service")
    return mod.TrainingService


def _make_service(paths: list[Path]):
    TrainingService = _training_service_cls()
    svc = TrainingService.__new__(TrainingService)
    settings = SimpleNamespace(
        system=SimpleNamespace(
            use_volume_features=False,
            base_timeframe="M1",
            required_timeframes=["M5"],
            higher_timeframes=["M5"],
            multi_resolution_timeframes=["M1", "M5"],
        ),
        risk=SimpleNamespace(total_drawdown_limit=0.07),
        models=SimpleNamespace(prop_search_holdout_min_truth_probability=0.70, prop_search_max_indicators=0),
    )

    def _model_copy():
        out = SimpleNamespace(
            system=SimpleNamespace(**vars(settings.system)),
            risk=SimpleNamespace(**vars(settings.risk)),
            models=SimpleNamespace(**vars(settings.models)),
        )
        out.model_copy = _model_copy
        return out

    settings.model_copy = _model_copy
    svc.settings = settings
    svc._prop_gene_artifact_paths = lambda _symbol: list(paths)
    return svc


class _ArrayFrame:
    def __init__(self, data: dict[str, np.ndarray], index: np.ndarray, attrs: dict[str, str] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self.index = np.asarray(index).reshape(-1)
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return len(self.index) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def __setitem__(self, key: str, value) -> None:
        col = str(key)
        arr = np.asarray(value).reshape(-1)
        n = len(self)
        if arr.size != n:
            if arr.size <= 0:
                arr = np.zeros(n, dtype=np.float32)
            elif arr.size > n:
                arr = arr[:n]
            else:
                arr = np.concatenate([arr, np.full(n - arr.size, arr[-1], dtype=arr.dtype)])
        self._data[col] = arr
        if col not in self.columns:
            self.columns.append(col)

    def copy(self, deep: bool = False) -> "_ArrayFrame":
        _ = deep
        return _ArrayFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            dict(self.attrs),
        )


def _write_artifact(path: Path, *, symbol: str, timeframe: str, strategy_id: str, forward: bool) -> None:
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "best_genes": [
            {
                "strategy_id": strategy_id,
                "indicators": ["RSI"],
                "weights": {"RSI": 1.0},
                "long_threshold": 0.3,
                "short_threshold": -0.3,
                "fitness": 2.0,
                "trades": 60.0,
                "max_dd_pct": 0.02,
                "holdout_months": 8.0,
                "holdout_max_dd_pct": 0.02,
                "truth_probability": 0.80,
                "forward_test_passed": forward,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_prop_best_genes_elite_requires_forward_pass(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / "genes_m1.json"
    payload = {
        "symbol": "EURUSD",
        "timeframe": "M1",
        "best_genes": [
            {
                "strategy_id": "pass",
                "indicators": ["RSI"],
                "weights": {"RSI": 1.0},
                "long_threshold": 0.3,
                "short_threshold": -0.3,
                "fitness": 2.0,
                "trades": 60.0,
                "max_dd_pct": 0.02,
                "holdout_months": 8.0,
                "holdout_max_dd_pct": 0.02,
                "truth_probability": 0.80,
                "forward_test_passed": True,
            },
            {
                "strategy_id": "fail",
                "indicators": ["RSI"],
                "weights": {"RSI": 1.0},
                "long_threshold": 0.3,
                "short_threshold": -0.3,
                "fitness": 3.0,
                "trades": 80.0,
                "max_dd_pct": 0.01,
                "holdout_months": 9.0,
                "holdout_max_dd_pct": 0.01,
                "truth_probability": 0.90,
                "forward_test_passed": False,
            },
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    svc = _make_service([p])

    class _DummyMixer:
        def __init__(self, *, device: str = "cpu", use_volume_features: bool = False) -> None:
            self.available_indicators = ["RSI"]

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(talib_mixer, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_STRICT_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.7")

    genes = svc._load_prop_best_genes("EURUSD", max_genes=8)
    assert len(genes) == 1
    assert str(getattr(genes[0], "strategy_id", "")) == "pass"


def test_load_prop_best_genes_enforces_timeframe_coverage(monkeypatch, tmp_path: Path) -> None:
    p1 = tmp_path / "genes_m1.json"
    p2 = tmp_path / "genes_m5.json"
    _write_artifact(p1, symbol="EURUSD", timeframe="M1", strategy_id="m1_gene", forward=True)
    _write_artifact(p2, symbol="EURUSD", timeframe="M5", strategy_id="m5_gene", forward=True)
    svc = _make_service([p1, p2])

    class _DummyMixer:
        def __init__(self, *, device: str = "cpu", use_volume_features: bool = False) -> None:
            self.available_indicators = ["RSI"]

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(talib_mixer, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_STRICT_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS_STRICT", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.7")

    genes = svc._load_prop_best_genes("EURUSD", max_genes=1)
    gene_tfs = {str(getattr(g, "source_timeframe", "") or "").upper() for g in genes}
    assert {"M1", "M5"}.issubset(gene_tfs)


def test_apply_prop_discovered_base_signal_updates_numpy_dataset(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / "genes_m1.json"
    _write_artifact(p, symbol="EURUSD", timeframe="M1", strategy_id="m1_gene", forward=True)
    svc = _make_service([p])

    class _DummyMixer:
        def __init__(self, *, device: str = "cpu", use_volume_features: bool = False) -> None:
            self.available_indicators = ["RSI"]

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(talib_mixer, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_STRICT_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.7")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.0")
    monkeypatch.setattr(
        svc,
        "_load_prop_best_genes",
        lambda symbol, max_genes: [
            SimpleNamespace(
                indicators=["RSI"],
                weights={"RSI": 1.0},
                long_threshold=0.3,
                short_threshold=-0.3,
                fitness=1.0,
            )
        ],
    )

    n = 12
    discovered = np.zeros(n, dtype=np.int8)
    discovered[::3] = 1
    discovered[2::3] = -1
    svc.feature_engineer = SimpleNamespace(
        _compute_discovered_base_signal_ohlcv_numpy=lambda **_kwargs: discovered
    )

    X = np.zeros((n, 2), dtype=np.float32)
    ds = PreparedDataset(
        X=X,
        y=np.zeros(n, dtype=np.int8),
        index=np.arange(n, dtype=np.int64),
        feature_names=["f0", "f1"],
        metadata=None,
        labels=np.zeros(n, dtype=np.int8),
    )
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    source = pd.DataFrame(
        {
            "open": np.linspace(1.1, 1.2, n),
            "high": np.linspace(1.1, 1.2, n) + 0.0005,
            "low": np.linspace(1.1, 1.2, n) - 0.0005,
            "close": np.linspace(1.1, 1.2, n),
        },
        index=idx,
    )

    out = svc._apply_prop_discovered_base_signal(ds, symbol="EURUSD", source_df=source)
    assert isinstance(out.X, np.ndarray)
    assert "base_signal" in out.feature_names
    base_idx = out.feature_names.index("base_signal")
    got = np.asarray(out.X[:, base_idx], dtype=np.int8)
    np.testing.assert_array_equal(got, discovered)


def test_apply_prop_discovered_base_signal_dataframe_uses_rust_when_talib_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    p = tmp_path / "genes_m1.json"
    _write_artifact(p, symbol="EURUSD", timeframe="M1", strategy_id="m1_gene", forward=True)
    svc = _make_service([p])

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", False, raising=False)
    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=lambda *args, **kwargs: None))
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_STRICT_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.7")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.0")

    n = 10
    discovered = np.zeros(n, dtype=np.int8)
    discovered[::2] = 1
    svc.feature_engineer = SimpleNamespace(
        _compute_discovered_base_signal_ohlcv_numpy=lambda **_kwargs: discovered
    )

    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    x_df = pd.DataFrame({"f0": np.arange(n, dtype=np.float32)}, index=idx)
    ds = PreparedDataset(
        X=x_df,
        y=np.zeros(n, dtype=np.int8),
        index=idx,
        feature_names=list(x_df.columns),
        metadata=None,
        labels=np.zeros(n, dtype=np.int8),
    )
    source = pd.DataFrame(
        {
            "open": np.linspace(1.1, 1.2, n),
            "high": np.linspace(1.1, 1.2, n) + 0.0005,
            "low": np.linspace(1.1, 1.2, n) - 0.0005,
            "close": np.linspace(1.1, 1.2, n),
        },
        index=idx,
    )

    out = svc._apply_prop_discovered_base_signal(ds, symbol="EURUSD", source_df=source)
    assert hasattr(out.X, "columns")
    assert "base_signal" in out.X.columns
    assert "prop_signal_score" in out.X.columns
    got = out.X["base_signal"].to_numpy(dtype=np.int8, copy=False)
    np.testing.assert_array_equal(got, discovered)


def test_apply_prop_discovered_base_signal_dataframe_uses_rust_bulk_when_primary_rust_missing(
    monkeypatch, tmp_path: Path
) -> None:
    p = tmp_path / "genes_m1.json"
    _write_artifact(p, symbol="EURUSD", timeframe="M1", strategy_id="m1_gene", forward=True)
    svc = _make_service([p])

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", False, raising=False)
    calls: dict[str, object] = {"bulk": 0, "kwargs": None}

    def _bulk(open_arr, high_arr, low_arr, close_arr, **kwargs):
        calls["bulk"] = int(calls["bulk"]) + 1
        calls["kwargs"] = kwargs
        n = int(len(close_arr))
        return np.ones((n, 1), dtype=np.int8)

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_STRICT_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.7")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.0")
    monkeypatch.setenv("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "11")

    svc.feature_engineer = SimpleNamespace(
        _compute_discovered_base_signal_ohlcv_numpy=lambda **_kwargs: None
    )

    n = 9
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    x_df = pd.DataFrame({"f0": np.arange(n, dtype=np.float32)}, index=idx)
    ds = PreparedDataset(
        X=x_df,
        y=np.zeros(n, dtype=np.int8),
        index=idx,
        feature_names=list(x_df.columns),
        metadata=None,
        labels=np.zeros(n, dtype=np.int8),
    )
    source = pd.DataFrame(
        {
            "open": np.linspace(1.1, 1.2, n),
            "high": np.linspace(1.1, 1.2, n) + 0.0005,
            "low": np.linspace(1.1, 1.2, n) - 0.0005,
            "close": np.linspace(1.1, 1.2, n),
        },
        index=idx,
    )

    out = svc._apply_prop_discovered_base_signal(ds, symbol="EURUSD", source_df=source)
    assert hasattr(out.X, "columns")
    assert int(calls["bulk"]) == 1
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert int(kwargs.get("causal_min_bars", 0)) == 11
    assert kwargs.get("timestamps") is not None
    got = out.X["base_signal"].to_numpy(dtype=np.int8, copy=False)
    np.testing.assert_array_equal(got, np.ones(n, dtype=np.int8))
    scores = out.X["prop_signal_score"].to_numpy(dtype=np.float32, copy=False)
    np.testing.assert_allclose(scores, np.ones(n, dtype=np.float32), rtol=0.0, atol=1e-6)


def test_apply_prop_discovered_base_signal_numpy_uses_rust_bulk_when_primary_rust_missing(
    monkeypatch, tmp_path: Path
) -> None:
    p = tmp_path / "genes_m1.json"
    _write_artifact(p, symbol="EURUSD", timeframe="M1", strategy_id="m1_gene", forward=True)
    svc = _make_service([p])

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", False, raising=False)
    calls = {"bulk": 0}

    def _bulk(open_arr, high_arr, low_arr, close_arr, **kwargs):
        calls["bulk"] += 1
        n = int(len(close_arr))
        return np.ones((1, n), dtype=np.int8)

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_STRICT_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.7")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.0")

    svc.feature_engineer = SimpleNamespace(
        _compute_discovered_base_signal_ohlcv_numpy=lambda **_kwargs: None
    )

    n = 10
    X = np.zeros((n, 2), dtype=np.float32)
    ds = PreparedDataset(
        X=X,
        y=np.zeros(n, dtype=np.int8),
        index=np.arange(n, dtype=np.int64),
        feature_names=["f0", "f1"],
        metadata=None,
        labels=np.zeros(n, dtype=np.int8),
    )
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    source = pd.DataFrame(
        {
            "open": np.linspace(1.1, 1.2, n),
            "high": np.linspace(1.1, 1.2, n) + 0.0005,
            "low": np.linspace(1.1, 1.2, n) - 0.0005,
            "close": np.linspace(1.1, 1.2, n),
        },
        index=idx,
    )

    out = svc._apply_prop_discovered_base_signal(ds, symbol="EURUSD", source_df=source)
    assert isinstance(out.X, np.ndarray)
    assert calls["bulk"] == 1
    assert "base_signal" in out.feature_names
    base_idx = out.feature_names.index("base_signal")
    got = np.asarray(out.X[:, base_idx], dtype=np.int8)
    np.testing.assert_array_equal(got, np.ones(n, dtype=np.int8))


def test_apply_prop_discovered_base_signal_fallback_disabled_skips_python_mixer(
    monkeypatch, tmp_path: Path
) -> None:
    p = tmp_path / "genes_m1.json"
    _write_artifact(p, symbol="EURUSD", timeframe="M1", strategy_id="m1_gene", forward=True)
    svc = _make_service([p])

    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_RUST_ONLY", "1")
    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=lambda *args, **kwargs: None))
    monkeypatch.setenv("FOREX_BOT_PROP_ELITE_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_STRICT_FILTER", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_FORWARD_PASS", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_REQUIRE_ALL_TFS", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_GENES", "0")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_HOLDOUT_MONTHS", "6")
    monkeypatch.setenv("FOREX_BOT_PROP_HOLDOUT_MAX_DD", "0.03")
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.7")
    monkeypatch.setenv("FOREX_BOT_PROP_BASE_SIGNAL_MIN_COVERAGE", "0.0")
    calls: dict[str, int] = {"mixer_init": 0}

    class _FailMixer:
        def __init__(self, *args, **kwargs):
            calls["mixer_init"] += 1
            raise AssertionError("Python TALib mixer path should be disabled.")

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(talib_mixer, "TALibStrategyMixer", _FailMixer, raising=False)

    svc.feature_engineer = SimpleNamespace(
        _compute_discovered_base_signal_ohlcv_numpy=lambda **_kwargs: None
    )

    n = 12
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    x_df = pd.DataFrame({"f0": np.arange(n, dtype=np.float32)}, index=idx)
    ds = PreparedDataset(
        X=x_df,
        y=np.zeros(n, dtype=np.int8),
        index=idx,
        feature_names=list(x_df.columns),
        metadata=None,
        labels=np.zeros(n, dtype=np.int8),
    )
    source = pd.DataFrame(
        {
            "open": np.linspace(1.1, 1.2, n),
            "high": np.linspace(1.1, 1.2, n) + 0.0005,
            "low": np.linspace(1.1, 1.2, n) - 0.0005,
            "close": np.linspace(1.1, 1.2, n),
        },
        index=idx,
    )

    out = svc._apply_prop_discovered_base_signal(ds, symbol="EURUSD", source_df=source)
    assert out is ds
    assert calls["mixer_init"] == 0
    assert "base_signal" not in out.X.columns


def test_build_discovery_frames_handles_numpy_base_dataset(monkeypatch, tmp_path: Path) -> None:
    svc = _make_service([])
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_FULL_TF_FEATURES", "0")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "0")

    n = 16
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    base = pd.DataFrame(
        {
            "open": np.linspace(1.10, 1.20, n),
            "high": np.linspace(1.10, 1.20, n) + 0.0005,
            "low": np.linspace(1.10, 1.20, n) - 0.0005,
            "close": np.linspace(1.10, 1.20, n),
        },
        index=idx,
    )
    htf = base.iloc[::5].copy()
    htf.attrs["timeframe"] = "M5"
    frames = {"M1": base, "M5": htf}

    x = np.column_stack([np.linspace(0.0, 1.0, n, dtype=np.float32)]).astype(np.float32)
    base_ds = PreparedDataset(
        X=x,
        y=np.zeros(n, dtype=np.int8),
        index=idx.view("int64"),
        feature_names=["f0"],
        metadata=None,
        labels=np.zeros(n, dtype=np.int8),
    )

    out, tfs = svc._build_discovery_frames_for_tensor(frames, None, "EURUSD", base_dataset=base_ds)
    assert "M1" in out
    assert "M5" in out
    assert "M1" in tfs
    assert "f0" in out["M1"].columns
    for col in ("open", "high", "low", "close"):
        assert col in out["M1"].columns
        assert col in out["M5"].columns
    np.testing.assert_allclose(out["M1"]["f0"].to_numpy(dtype=np.float32), x.reshape(-1))


def test_build_discovery_frames_prefers_rust_feature_path(monkeypatch, tmp_path: Path) -> None:
    svc = _make_service([])
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_FULL_TF_FEATURES", "1")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_RUST_FEATURES", "1")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "0")

    n1 = 12
    n5 = 6
    idx1 = pd.date_range("2025-01-01", periods=n1, freq="min", tz="UTC")
    idx5 = pd.date_range("2025-01-01", periods=n5, freq="5min", tz="UTC")
    frames = {
        "M1": pd.DataFrame(
            {
                "open": np.linspace(1.10, 1.20, n1),
                "high": np.linspace(1.10, 1.20, n1) + 0.0005,
                "low": np.linspace(1.10, 1.20, n1) - 0.0005,
                "close": np.linspace(1.10, 1.20, n1),
            },
            index=idx1,
        ),
        "M5": pd.DataFrame(
            {
                "open": np.linspace(1.10, 1.20, n5),
                "high": np.linspace(1.10, 1.20, n5) + 0.0005,
                "low": np.linspace(1.10, 1.20, n5) - 0.0005,
                "close": np.linspace(1.10, 1.20, n5),
            },
            index=idx5,
        ),
    }

    mod = importlib.import_module("forex_bot.execution.training_service")
    calls: list[tuple[str, bool]] = []

    class _DummyFE:
        def __init__(self, settings) -> None:
            self._tf = str(getattr(settings.system, "base_timeframe", "M1") or "M1").upper()

        def prepare(self, frames_arg, *, news_features=None, symbol=None):
            calls.append((self._tf, bool(frames_arg)))
            src = frames[self._tf]
            arr = np.full((len(src), 1), 1.0 if self._tf == "M1" else 5.0, dtype=np.float32)
            y = np.zeros(len(src), dtype=np.int8)
            return PreparedDataset(
                X=arr,
                y=y,
                index=src.index.view("int64"),
                feature_names=[f"{self._tf}_f"],
                metadata=None,
                labels=y,
            )

    monkeypatch.setattr(mod, "FeatureEngineer", _DummyFE)

    out, _ = svc._build_discovery_frames_for_tensor(frames, None, "EURUSD", base_dataset=None)
    assert calls, "FeatureEngineer.prepare should be called for per-TF generation"
    assert all((not used_frames) for _tf, used_frames in calls)
    assert "M1" in out and "M5" in out
    assert "M1_f" in out["M1"].columns
    assert "M5_f" in out["M5"].columns


def test_inject_discovery_mixer_signals_uses_rust_backend_when_available(monkeypatch, tmp_path: Path) -> None:
    svc = _make_service([])
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "1")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_MIXER_STRATEGIES", "3")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_MIXER_MAX_INDICATORS", "2")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_MIXER_SEED", "7")
    monkeypatch.setenv("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "8")

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", False, raising=False)
    calls: dict[str, int] = {"bulk": 0}

    def _bulk(open_arr, high_arr, low_arr, close_arr, **kwargs):
        calls["bulk"] += 1
        n = int(len(close_arr))
        sets = kwargs.get("indicator_sets") or []
        m = int(len(sets))
        out = np.zeros((n, m), dtype=np.int8)
        if m > 0:
            out[:, 0] = 1
        if m > 1:
            out[::2, 1] = -1
        if m > 2:
            out[1::2, 2] = 1
        return out

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))

    n1 = 10
    n5 = 5
    idx1 = pd.date_range("2025-01-01", periods=n1, freq="min", tz="UTC")
    idx5 = pd.date_range("2025-01-01", periods=n5, freq="5min", tz="UTC")
    frames = {
        "M1": pd.DataFrame(
            {
                "open": np.linspace(1.10, 1.20, n1),
                "high": np.linspace(1.10, 1.20, n1) + 0.0005,
                "low": np.linspace(1.10, 1.20, n1) - 0.0005,
                "close": np.linspace(1.10, 1.20, n1),
            },
            index=idx1,
        ),
        "M5": pd.DataFrame(
            {
                "open": np.linspace(1.10, 1.20, n5),
                "high": np.linspace(1.10, 1.20, n5) + 0.0005,
                "low": np.linspace(1.10, 1.20, n5) - 0.0005,
                "close": np.linspace(1.10, 1.20, n5),
            },
            index=idx5,
        ),
    }

    out = svc._inject_discovery_mixer_signals(frames, base_tf="M1", per_tf=False)
    assert calls["bulk"] >= 1
    assert "M1" in out and "M5" in out
    assert "tmx_sig_0" in out["M1"].columns
    assert "tmx_sig_1" in out["M1"].columns
    assert "tmx_sig_2" in out["M1"].columns
    assert "tmx_sig_0" in out["M5"].columns


def test_inject_discovery_mixer_signals_py_fallback_disabled_skips_python_mixer(monkeypatch, tmp_path: Path) -> None:
    svc = _make_service([])
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "1")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_RUST_ONLY", "1")
    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace())

    calls: dict[str, int] = {"mixer_init": 0}

    class _FailMixer:
        def __init__(self, *args, **kwargs):
            calls["mixer_init"] += 1
            raise AssertionError("Python TA-Lib mixer should be skipped when fallback is disabled.")

    monkeypatch.setattr(talib_mixer, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(talib_mixer, "TALibStrategyMixer", _FailMixer, raising=False)

    n1 = 8
    n5 = 4
    idx1 = pd.date_range("2025-01-01", periods=n1, freq="min", tz="UTC")
    idx5 = pd.date_range("2025-01-01", periods=n5, freq="5min", tz="UTC")
    frames = {
        "M1": pd.DataFrame(
            {
                "open": np.linspace(1.10, 1.20, n1),
                "high": np.linspace(1.10, 1.20, n1) + 0.0005,
                "low": np.linspace(1.10, 1.20, n1) - 0.0005,
                "close": np.linspace(1.10, 1.20, n1),
            },
            index=idx1,
        ),
        "M5": pd.DataFrame(
            {
                "open": np.linspace(1.10, 1.20, n5),
                "high": np.linspace(1.10, 1.20, n5) + 0.0005,
                "low": np.linspace(1.10, 1.20, n5) - 0.0005,
                "close": np.linspace(1.10, 1.20, n5),
            },
            index=idx5,
        ),
    }

    out = svc._inject_discovery_mixer_signals(frames, base_tf="M1", per_tf=False)
    assert calls["mixer_init"] == 0
    assert set(out.keys()) == {"M1", "M5"}
    assert "tmx_sig_0" not in out["M1"].columns
    assert "tmx_sig_0" not in out["M5"].columns


def test_prepared_dataset_to_frame_pandas_free_uses_fallback_frame(monkeypatch) -> None:
    svc = _make_service([])

    n = 12
    idx = np.datetime64("2025-01-01T00:00:00") + np.arange(n, dtype=np.int64) * np.timedelta64(1, "m")
    fallback = _ArrayFrame(
        {
            "open": np.linspace(1.10, 1.20, n, dtype=np.float64),
            "high": np.linspace(1.10, 1.20, n, dtype=np.float64) + 0.0005,
            "low": np.linspace(1.10, 1.20, n, dtype=np.float64) - 0.0005,
            "close": np.linspace(1.10, 1.20, n, dtype=np.float64),
        },
        idx,
    )

    x = np.column_stack(
        [
            np.linspace(0.0, 1.0, n, dtype=np.float32),
            np.linspace(1.0, 2.0, n, dtype=np.float32),
        ]
    )
    ds = PreparedDataset(
        X=x,
        y=np.zeros(n, dtype=np.int8),
        index=idx.astype("datetime64[ns]").astype(np.int64),
        feature_names=["f0", "f1"],
        metadata=None,
        labels=np.zeros(n, dtype=np.int8),
    )

    out = svc._prepared_dataset_to_frame(ds, fallback_frame=fallback)
    assert out is not None
    cols = {str(c) for c in out.columns}
    assert {"open", "high", "low", "close", "f0", "f1"}.issubset(cols)
    np.testing.assert_allclose(np.asarray(out["f0"], dtype=np.float32), x[:, 0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["close"], dtype=np.float64), np.asarray(fallback["close"], dtype=np.float64))


def test_build_discovery_frames_non_dataframe_pandas_free(monkeypatch) -> None:
    svc = _make_service([])
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_FULL_TF_FEATURES", "0")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "0")

    n1 = 12
    n5 = 6
    idx1 = np.datetime64("2025-01-01T00:00:00") + np.arange(n1, dtype=np.int64) * np.timedelta64(1, "m")
    idx5 = np.datetime64("2025-01-01T00:00:00") + np.arange(n5, dtype=np.int64) * np.timedelta64(5, "m")
    frames = {
        "M1": _ArrayFrame(
            {
                "open": np.linspace(1.10, 1.20, n1),
                "high": np.linspace(1.10, 1.20, n1) + 0.0005,
                "low": np.linspace(1.10, 1.20, n1) - 0.0005,
                "close": np.linspace(1.10, 1.20, n1),
            },
            idx1,
            attrs={"symbol": "EURUSD", "timeframe": "M1"},
        ),
        "M5": _ArrayFrame(
            {
                "open": np.linspace(1.10, 1.20, n5),
                "high": np.linspace(1.10, 1.20, n5) + 0.0005,
                "low": np.linspace(1.10, 1.20, n5) - 0.0005,
                "close": np.linspace(1.10, 1.20, n5),
            },
            idx5,
            attrs={"symbol": "EURUSD", "timeframe": "M5"},
        ),
    }

    x = np.column_stack([np.linspace(0.0, 1.0, n1, dtype=np.float32)]).astype(np.float32)
    base_ds = PreparedDataset(
        X=x,
        y=np.zeros(n1, dtype=np.int8),
        index=idx1.astype("datetime64[ns]").astype(np.int64),
        feature_names=["f0"],
        metadata=None,
        labels=np.zeros(n1, dtype=np.int8),
    )

    out, tfs = svc._build_discovery_frames_for_tensor(frames, None, "EURUSD", base_dataset=base_ds)
    assert "M1" in out and "M5" in out
    assert "M1" in tfs
    assert "f0" in {str(c) for c in out["M1"].columns}
    for col in ("open", "high", "low", "close"):
        assert col in {str(c) for c in out["M1"].columns}
        assert col in {str(c) for c in out["M5"].columns}

