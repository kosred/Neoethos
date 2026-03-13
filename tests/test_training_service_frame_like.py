from __future__ import annotations

import numpy as np
import pytest

from forex_bot.domain.events import PreparedDataset
from forex_bot.execution.training_service import TrainingService


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

    def __setitem__(self, key, value) -> None:
        self._data[str(key)] = np.asarray(value).reshape(-1)
        if str(key) not in self.columns:
            self.columns.append(str(key))

    def copy(self, deep: bool = False):
        _ = deep
        return _ArrayFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            dict(self.attrs),
        )


def test_align_global_feature_space_accepts_frame_like() -> None:
    prefer_numpy = True
    n = 800
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    ds1 = PreparedDataset(
        X=_ArrayFrame(
            {
                "a": np.random.default_rng(1).normal(size=n).astype(np.float32),
                "b": np.random.default_rng(2).normal(size=n).astype(np.float32),
            },
            index=idx,
        ),
        y=np.random.default_rng(3).integers(0, 3, size=n, dtype=np.int8),
        index=idx,
        feature_names=["a", "b"],
        metadata=_ArrayFrame({"close": np.linspace(1.0, 1.1, num=n, dtype=np.float32)}, index=idx),
        labels=None,
    )
    ds2 = PreparedDataset(
        X=_ArrayFrame(
            {
                "b": np.random.default_rng(4).normal(size=n).astype(np.float32),
                "c": np.random.default_rng(5).normal(size=n).astype(np.float32),
            },
            index=idx,
        ),
        y=np.random.default_rng(6).integers(0, 3, size=n, dtype=np.int8),
        index=idx,
        feature_names=["b", "c"],
        metadata=_ArrayFrame({"close": np.linspace(1.2, 1.3, num=n, dtype=np.float32)}, index=idx),
        labels=None,
    )

    svc = object.__new__(TrainingService)
    cols, aligned = svc._align_global_feature_space([("EURUSD", ds1), ("GBPUSD", ds2)], prefer_numpy=prefer_numpy)

    assert cols == ["a", "b", "c"]
    assert len(aligned) == 2
    for _sym, ds in aligned:
        assert isinstance(ds.X, np.ndarray)
        assert ds.X.shape == (n, 3)
        assert isinstance(ds.y, np.ndarray)
        assert ds.y.shape == (n,)
        assert isinstance(ds.index, np.ndarray)
        assert ds.index.shape == (n,)
        if prefer_numpy:
            assert ds.metadata is not None


def test_split_global_train_eval_accepts_frame_like(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOREX_BOT_GLOBAL_EVAL_FROM", raising=False)
    monkeypatch.setenv("FOREX_BOT_GLOBAL_EVAL_YEARS", "0")

    n = 1200
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    x = _ArrayFrame(
        {
            "f0": np.linspace(0.0, 1.0, num=n, dtype=np.float32),
            "f1": np.linspace(1.0, 2.0, num=n, dtype=np.float32),
        },
        index=idx,
    )
    y = np.random.default_rng(42).integers(0, 3, size=n, dtype=np.int8)
    meta = _ArrayFrame({"close": np.linspace(1.0, 1.1, num=n, dtype=np.float32)}, index=idx)
    ds = PreparedDataset(X=x, y=y, index=idx, feature_names=["f0", "f1"], metadata=meta, labels=y)

    svc = object.__new__(TrainingService)
    train_parts, eval_map, split_meta = svc._split_global_train_eval(
        [("EURUSD", ds)],
        train_ratio=0.8,
        embargo_bars=8,
        min_train_rows=100,
        min_eval_rows=50,
    )

    assert train_parts
    assert "EURUSD" in eval_map
    assert split_meta.get("cutoff_mode") == "ratio"
    _sym, train_ds = train_parts[0]
    eval_ds = eval_map["EURUSD"]
    assert len(train_ds.X) == len(train_ds.y)
    assert len(eval_ds.X) == len(eval_ds.y)
    assert hasattr(train_ds.X, "columns")
    assert hasattr(eval_ds.X, "columns")
    assert train_ds.metadata is not None
    assert eval_ds.metadata is not None


def test_tail_dataset_accepts_frame_like() -> None:
    n = 10
    idx = np.arange(n, dtype=np.int64)
    x = _ArrayFrame({"f0": np.arange(n, dtype=np.float32), "f1": np.arange(n, dtype=np.float32) + 10.0}, index=idx)
    y = np.arange(n, dtype=np.int8)
    meta = _ArrayFrame({"close": np.linspace(1.0, 1.1, num=n, dtype=np.float32)}, index=idx)
    ds = PreparedDataset(X=x, y=y, index=idx, feature_names=["f0", "f1"], metadata=meta, labels=y)

    out = TrainingService._tail_dataset(ds, 3)
    assert len(out.X) == 3
    assert len(out.y) == 3
    assert out.metadata is not None
    assert len(out.metadata) == 3


def test_merge_symbol_shards_accepts_frame_like_numpy_output() -> None:
    idx1 = np.array([3, 1, 2], dtype=np.int64) * 60_000_000_000
    idx2 = np.array([2, 4], dtype=np.int64) * 60_000_000_000
    ds1 = PreparedDataset(
        X=_ArrayFrame({"a": np.array([3.0, 1.0, 2.0], dtype=np.float32)}, index=idx1),
        y=np.array([1, 0, 1], dtype=np.int8),
        index=idx1,
        feature_names=["a"],
        metadata=None,
        labels=None,
    )
    ds2 = PreparedDataset(
        X=_ArrayFrame(
            {"a": np.array([20.0, 40.0], dtype=np.float32), "b": np.array([200.0, 400.0], dtype=np.float32)},
            index=idx2,
        ),
        y=np.array([2, 2], dtype=np.int8),
        index=idx2,
        feature_names=["a", "b"],
        metadata=None,
        labels=None,
    )

    svc = object.__new__(TrainingService)
    merged = svc._merge_symbol_shards("EURUSD", [ds1, ds2], prefer_numpy=True)

    assert merged is not None
    assert isinstance(merged.X, np.ndarray)
    assert isinstance(merged.y, np.ndarray)
    assert merged.X.shape == (4, 2)
    assert merged.y.shape == (4,)
    np.testing.assert_allclose(merged.X[:, 0], np.array([1.0, 2.0, 3.0, 40.0], dtype=np.float32))
    np.testing.assert_allclose(merged.X[:, 1], np.array([0.0, 0.0, 0.0, 400.0], dtype=np.float32))


def test_numpy_dataset_returns_uses_frame_like_metadata_close() -> None:
    n = 6
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    ds = PreparedDataset(
        X=np.zeros((n, 1), dtype=np.float32),
        y=np.zeros(n, dtype=np.int8),
        index=idx,
        feature_names=["f0"],
        metadata=_ArrayFrame({"close": close}, index=idx),
        labels=None,
    )

    svc = object.__new__(TrainingService)
    out = svc._numpy_dataset_returns(ds)

    assert out is not None
    idx_out, ret = out
    assert isinstance(idx_out, np.ndarray)
    assert idx_out.shape == (n,)
    np.testing.assert_allclose(ret, np.array([0.0, 1.0, 0.5, 1.0 / 3.0, 0.25, 0.2], dtype=np.float32), rtol=1e-5)
