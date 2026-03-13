from __future__ import annotations

import os

import numpy as np
import pytest

from forex_bot.training import cpcv as cpcv_mod
from forex_bot.training.cpcv import CombinatorialPurgedCV, cpcv_backtest


class _DummyClassifier:
    def fit(self, x, y, sample_weight=None):  # noqa: ARG002
        return None

    def predict_proba(self, x):
        n = int(len(x))
        out = np.zeros((n, 3), dtype=np.float64)
        out[:, 1] = 1.0
        return out


def _model_factory():
    return _DummyClassifier()


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


def test_cpcv_score_accepts_numpy_without_pandas() -> None:
    prev = os.environ.get("FOREX_BOT_PANDAS_BLOCK")
    os.environ["FOREX_BOT_PANDAS_BLOCK"] = "1"
    try:
        x = np.random.default_rng(11).normal(size=(240, 6)).astype(np.float32)
        y = np.random.default_rng(12).integers(0, 3, size=240, dtype=np.int8)
        w = np.ones(240, dtype=np.float32)

        cv = CombinatorialPurgedCV(n_splits=5, n_test_groups=1, embargo_pct=0.01, purge_pct=0.01)
        result = cv.score(x=x, y=y, model_factory=_model_factory, sample_weights=w, n_jobs=1)

        assert result.n_combinations > 0
        assert len(result.scores) > 0
        assert float(result.phi) >= 0.0
    finally:
        if prev is None:
            os.environ.pop("FOREX_BOT_PANDAS_BLOCK", None)
        else:
            os.environ["FOREX_BOT_PANDAS_BLOCK"] = prev


def test_cpcv_backtest_accepts_numpy_and_dict_metadata_without_pandas() -> None:
    prev = os.environ.get("FOREX_BOT_PANDAS_BLOCK")
    os.environ["FOREX_BOT_PANDAS_BLOCK"] = "1"
    try:
        n = 260
        x = np.random.default_rng(21).normal(size=(n, 4)).astype(np.float32)
        y = np.random.default_rng(22).integers(0, 3, size=n, dtype=np.int8)
        close = np.linspace(1.0, 1.01, num=n, dtype=np.float64)
        meta = {
            "close": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "index": np.arange(n, dtype=np.int64),
        }

        out = cpcv_backtest(
            x=x,
            y=y,
            metadata=meta,
            model_factory=_model_factory,
            n_splits=5,
            n_test_groups=1,
            embargo_pct=0.01,
            purge_pct=0.01,
            n_jobs=1,
        )
        assert isinstance(out, dict)
        assert int(out.get("n_combinations", 0)) > 0
        assert int(out.get("n_splits", 0)) > 0
    finally:
        if prev is None:
            os.environ.pop("FOREX_BOT_PANDAS_BLOCK", None)
        else:
            os.environ["FOREX_BOT_PANDAS_BLOCK"] = prev


def test_cpcv_score_accepts_frame_like_without_pandas() -> None:
    prev = os.environ.get("FOREX_BOT_PANDAS_BLOCK")
    os.environ["FOREX_BOT_PANDAS_BLOCK"] = "1"
    try:
        n = 220
        close = np.linspace(1.0, 1.02, num=n, dtype=np.float64)
        x = _ArrayFrame(
            {
                "close": close,
                "high": close + 0.0002,
                "low": close - 0.0002,
                "feat_1": np.sin(np.linspace(0.0, 3.14, num=n)),
            },
            index=np.arange(n, dtype=np.int64),
            attrs={"symbol": "EURUSD"},
        )
        y = np.random.default_rng(31).integers(0, 3, size=n, dtype=np.int8)
        w = np.ones(n, dtype=np.float32)

        cv = CombinatorialPurgedCV(n_splits=5, n_test_groups=1, embargo_pct=0.01, purge_pct=0.01)
        result = cv.score(x=x, y=y, model_factory=_model_factory, sample_weights=w, n_jobs=1)
        assert result.n_combinations > 0
        assert len(result.scores) > 0
    finally:
        if prev is None:
            os.environ.pop("FOREX_BOT_PANDAS_BLOCK", None)
        else:
            os.environ["FOREX_BOT_PANDAS_BLOCK"] = prev


def test_cpcv_backtest_maps_prop_backtest_net_profit_to_avg_pnl(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 220
    x = np.random.default_rng(41).normal(size=(n, 5)).astype(np.float32)
    y = np.random.default_rng(42).integers(0, 3, size=n, dtype=np.int8)
    close = np.linspace(1.0, 1.01, num=n, dtype=np.float64)
    metadata = {
        "close": close,
        "high": close + 0.0002,
        "low": close - 0.0002,
        "index": np.arange(n, dtype=np.int64),
        "symbol": "EURUSD",
    }

    def _fake_prop_backtest(frame, signals):  # noqa: ARG001
        return {
            "net_profit": 2.5,
            "win_rate": 0.6,
            "sharpe": 1.1,
            "trades": 10.0,
        }

    monkeypatch.setattr("forex_bot.training.evaluation.prop_backtest", _fake_prop_backtest)
    out = cpcv_backtest(
        x=x,
        y=y,
        metadata=metadata,
        model_factory=_model_factory,
        n_splits=5,
        n_test_groups=1,
        embargo_pct=0.01,
        purge_pct=0.01,
        n_jobs=1,
    )
    assert isinstance(out, dict)
    assert int(out.get("n_splits", 0)) > 0
    assert float(out.get("avg_pnl", 0.0)) == pytest.approx(2.5)


def test_cpcv_month_day_indices_uses_rust_binding_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = type(
        "_Fake",
        (),
        {
            "derive_time_index_arrays": staticmethod(
                lambda index_ns: (
                    np.asarray(index_ns, dtype=np.int64) // 1_000_000,
                    np.array([111, 112, 113], dtype=np.int64),
                    np.array([211, 212, 213], dtype=np.int64),
                )
            )
        },
    )()
    monkeypatch.setattr(cpcv_mod, "_fb", fake, raising=False)
    idx = np.array([1_700_000_000_000_000_000, 1_700_000_060_000_000_000, 1_700_000_120_000_000_000], dtype=np.int64)
    month_idx, day_idx = cpcv_mod._month_day_indices(idx)
    np.testing.assert_array_equal(month_idx, np.array([111, 112, 113], dtype=np.int64))
    np.testing.assert_array_equal(day_idx, np.array([211, 212, 213], dtype=np.int64))
