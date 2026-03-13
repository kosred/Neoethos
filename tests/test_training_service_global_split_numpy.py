from __future__ import annotations

import numpy as np
import pytest

from forex_bot.domain.events import PreparedDataset
from forex_bot.execution.training_service import TrainingService
from tests._compat_pd import pd


def test_global_split_accepts_numpy_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOREX_BOT_GLOBAL_EVAL_FROM", raising=False)
    monkeypatch.setenv("FOREX_BOT_GLOBAL_EVAL_YEARS", "0")

    idx = pd.date_range("2018-01-01", periods=1200, freq="h", tz="UTC")
    x = pd.DataFrame(
        {
            "f0": np.linspace(0.0, 1.0, num=len(idx), dtype=np.float32),
            "f1": np.linspace(1.0, 2.0, num=len(idx), dtype=np.float32),
        },
        index=idx,
    )
    y = np.random.default_rng(42).integers(0, 3, size=len(idx), dtype=np.int8)
    ds = PreparedDataset(X=x, y=y, index=x.index, feature_names=list(x.columns))

    svc = object.__new__(TrainingService)
    train_parts, eval_map, meta = svc._split_global_train_eval(
        [("EURUSD", ds)],
        train_ratio=0.8,
        embargo_bars=8,
        min_train_rows=100,
        min_eval_rows=50,
    )

    assert train_parts
    assert "EURUSD" in eval_map
    assert meta.get("cutoff_mode") == "ratio"

    _sym, train_ds = train_parts[0]
    eval_ds = eval_map["EURUSD"]
    assert isinstance(train_ds.y, np.ndarray)
    assert isinstance(eval_ds.y, np.ndarray)
    assert len(train_ds.y) == len(train_ds.X)
    assert len(eval_ds.y) == len(eval_ds.X)


def test_global_split_accepts_series_labels_and_returns_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOREX_BOT_GLOBAL_EVAL_FROM", raising=False)
    monkeypatch.setenv("FOREX_BOT_GLOBAL_EVAL_YEARS", "0")

    idx = pd.date_range("2019-01-01", periods=1200, freq="h", tz="UTC")
    x = pd.DataFrame(
        {
            "f0": np.linspace(0.0, 1.0, num=len(idx), dtype=np.float32),
            "f1": np.linspace(1.0, 2.0, num=len(idx), dtype=np.float32),
        },
        index=idx,
    )
    y = pd.Series(np.random.default_rng(7).integers(0, 3, size=len(idx), dtype=np.int8), index=idx, dtype=np.int8)
    ds = PreparedDataset(X=x, y=y, index=x.index, feature_names=list(x.columns))

    svc = object.__new__(TrainingService)
    train_parts, eval_map, _meta = svc._split_global_train_eval(
        [("EURUSD", ds)],
        train_ratio=0.8,
        embargo_bars=8,
        min_train_rows=100,
        min_eval_rows=50,
    )

    assert train_parts
    assert "EURUSD" in eval_map
    _sym, train_ds = train_parts[0]
    eval_ds = eval_map["EURUSD"]
    assert isinstance(train_ds.y, np.ndarray)
    assert isinstance(eval_ds.y, np.ndarray)
    assert train_ds.y.dtype == np.int8
    assert eval_ds.y.dtype == np.int8
    assert len(train_ds.y) == len(train_ds.X)
    assert len(eval_ds.y) == len(eval_ds.X)

