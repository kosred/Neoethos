import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests._compat_pd import pd

from forex_bot.domain.events import PreparedDataset
from forex_bot.training.evaluation_service import EvaluationService


class _ArrayFrame:
    def __init__(self, data, index, attrs=None):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def _settings() -> SimpleNamespace:
    models = SimpleNamespace(
        walkforward_splits=4,
        enable_cpcv=True,
        cpcv_n_splits=5,
        cpcv_n_test_groups=1,
        cpcv_embargo_pct=0.01,
        cpcv_purge_pct=0.01,
        cpcv_max_rows=10_000,
    )
    return SimpleNamespace(models=models)


def test_walkforward_accepts_numpy_dataset(tmp_path: Path) -> None:
    n = 400
    x = np.random.default_rng(42).normal(size=(n, 8)).astype(np.float32)
    y = np.random.default_rng(7).integers(0, 3, size=n, dtype=np.int8)
    ds = PreparedDataset(X=x, y=y, index=np.arange(n), feature_names=[f"f{i}" for i in range(x.shape[1])])

    svc = EvaluationService(_settings(), tmp_path)

    def _ensemble(chunk_x):
        rows = int(len(chunk_x))
        out = np.zeros((rows, 3), dtype=np.float32)
        out[:, 1] = 1.0
        return out

    metrics = svc.run_walkforward(ds, models={}, ensemble_func=_ensemble, start_index=50)
    assert int(metrics.get("walkforward_splits", 0)) > 0
    assert "avg_accuracy" in metrics


def test_walkforward_accepts_frame_like_dataset(tmp_path: Path) -> None:
    n = 420
    x = _ArrayFrame(
        {
            "f0": np.random.default_rng(42).normal(size=n).astype(np.float32),
            "f1": np.random.default_rng(43).normal(size=n).astype(np.float32),
            "f2": np.random.default_rng(44).normal(size=n).astype(np.float32),
        },
        index=np.arange(n, dtype=np.int64),
    )
    y = np.random.default_rng(7).integers(0, 3, size=n, dtype=np.int8)
    ds = PreparedDataset(X=x, y=y, index=np.arange(n), feature_names=["f0", "f1", "f2"])

    svc = EvaluationService(_settings(), tmp_path)

    def _ensemble(chunk_x):
        rows = int(len(chunk_x))
        out = np.zeros((rows, 3), dtype=np.float32)
        out[:, 1] = 1.0
        return out

    metrics = svc.run_walkforward(ds, models={}, ensemble_func=_ensemble, start_index=50)
    assert int(metrics.get("walkforward_splits", 0)) > 0
    assert "avg_accuracy" in metrics


def test_cpcv_runs_in_pandas_free_mode_without_pandas(tmp_path: Path) -> None:
    prev = os.environ.get("FOREX_BOT_PANDAS_FREE")
    prev_block = os.environ.get("FOREX_BOT_PANDAS_BLOCK")
    os.environ["FOREX_BOT_PANDAS_FREE"] = "1"
    os.environ["FOREX_BOT_PANDAS_BLOCK"] = "1"
    try:
        x = np.random.default_rng(1).normal(size=(200, 4)).astype(np.float32)
        y = np.random.default_rng(2).integers(0, 3, size=200, dtype=np.int8)
        close = np.linspace(1.0, 1.01, num=len(x), dtype=np.float64)
        ds = PreparedDataset(
            X=x,
            y=y,
            index=np.arange(len(x)),
            feature_names=[f"f{i}" for i in range(x.shape[1])],
            metadata={"close": close},
        )
        svc = EvaluationService(_settings(), tmp_path)
        out = svc.run_cpcv(ds, models={})
        assert isinstance(out, dict)
        assert int(out.get("n_splits", 0)) > 0
        assert int(out.get("n_combinations", 0)) > 0
    finally:
        if prev is None:
            os.environ.pop("FOREX_BOT_PANDAS_FREE", None)
        else:
            os.environ["FOREX_BOT_PANDAS_FREE"] = prev
        if prev_block is None:
            os.environ.pop("FOREX_BOT_PANDAS_BLOCK", None)
        else:
            os.environ["FOREX_BOT_PANDAS_BLOCK"] = prev_block


def test_cpcv_runs_with_frame_like_x_and_metadata(tmp_path: Path) -> None:
    n = 240
    x = _ArrayFrame(
        {
            "f0": np.random.default_rng(1).normal(size=n).astype(np.float32),
            "f1": np.random.default_rng(2).normal(size=n).astype(np.float32),
            "f2": np.random.default_rng(3).normal(size=n).astype(np.float32),
            "f3": np.random.default_rng(4).normal(size=n).astype(np.float32),
        },
        index=np.arange(n, dtype=np.int64),
    )
    y = np.random.default_rng(5).integers(0, 3, size=n, dtype=np.int8)
    close = np.linspace(1.0, 1.01, num=n, dtype=np.float64)
    meta = _ArrayFrame({"Close": close}, index=np.arange(n, dtype=np.int64), attrs={"symbol": "EURUSD"})
    ds = PreparedDataset(
        X=x,
        y=y,
        index=np.arange(n),
        feature_names=["f0", "f1", "f2", "f3"],
        metadata=meta,
    )
    svc = EvaluationService(_settings(), tmp_path)
    out = svc.run_cpcv(ds, models={})
    assert isinstance(out, dict)
    assert int(out.get("n_splits", 0)) > 0
    assert int(out.get("n_combinations", 0)) > 0


def test_cpcv_prefers_prop_backtest_when_ohlc_metadata_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    n = 240
    x = np.random.default_rng(9).normal(size=(n, 4)).astype(np.float32)
    y = np.random.default_rng(10).integers(0, 3, size=n, dtype=np.int8)
    close = np.linspace(1.0, 1.02, num=n, dtype=np.float64)
    ds = PreparedDataset(
        X=x,
        y=y,
        index=np.arange(n),
        feature_names=[f"f{i}" for i in range(x.shape[1])],
        metadata={
            "close": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "index": np.arange(n, dtype=np.int64),
            "symbol": "EURUSD",
        },
    )
    svc = EvaluationService(_settings(), tmp_path)

    calls = {"prop": 0, "quick": 0}

    def _fake_prop(frame, signals):
        calls["prop"] += 1
        n_sig = int(len(np.asarray(signals).reshape(-1)))
        return {
            "net_profit": 5.0,
            "pnl_score": 5.0,
            "win_rate": 0.55,
            "sharpe": 1.25,
            "trades": float(n_sig),
        }

    def _fake_quick(frame, signals):  # noqa: ARG001
        calls["quick"] += 1
        return {"pnl_score": -1.0, "win_rate": 0.0}

    monkeypatch.setattr("forex_bot.training.evaluation_service.prop_backtest", _fake_prop)
    monkeypatch.setattr("forex_bot.training.evaluation_service.quick_backtest", _fake_quick)

    out = svc.run_cpcv(ds, models={})
    assert isinstance(out, dict)
    assert int(out.get("n_splits", 0)) > 0
    assert calls["prop"] > 0
    assert calls["quick"] == 0
    assert float(out.get("avg_pnl", 0.0)) == pytest.approx(5.0)


def test_extract_eval_frame_normalizes_object_datetime_index_to_ns(tmp_path: Path) -> None:
    n = 4
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    expected = np.asarray(
        [int(ts.value) if hasattr(ts, "value") else int(np.datetime64(ts, "ns").astype(np.int64)) for ts in list(idx)],
        dtype=np.int64,
    )
    meta = _ArrayFrame(
        {
            "close": np.linspace(1.0, 1.01, num=n, dtype=np.float64),
            "high": np.linspace(1.0002, 1.0102, num=n, dtype=np.float64),
            "low": np.linspace(0.9998, 1.0098, num=n, dtype=np.float64),
        },
        index=np.asarray(list(idx), dtype=object),
        attrs={"symbol": "EURUSD"},
    )

    svc = EvaluationService(_settings(), tmp_path)
    frame = svc._extract_eval_frame(meta, n)

    assert isinstance(frame, dict)
    assert "index" in frame
    assert np.asarray(frame["index"], dtype=np.int64).dtype == np.int64
    np.testing.assert_array_equal(np.asarray(frame["index"], dtype=np.int64), expected)
