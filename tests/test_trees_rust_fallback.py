from __future__ import annotations

import numpy as np
import pytest
from tests._compat_pd import pd

from forex_bot.models import trees as tmod
from forex_bot.models import trees_rust as tr


class _FailingRustModel:
    def __init__(self, idx: int = 1, params: dict | None = None) -> None:
        self.idx = idx
        self.params = params or {}

    def fit(self, *_args, **_kwargs) -> None:
        raise RuntimeError("rust fit failed")

    def predict_proba(self, *_args, **_kwargs):
        raise RuntimeError("rust predict failed")

    def save(self, *_args, **_kwargs) -> None:
        raise RuntimeError("rust save failed")

    def load(self, *_args, **_kwargs) -> None:
        raise RuntimeError("rust load failed")


class _RustArrayModel:
    def __init__(self, idx: int = 1, params: dict | None = None) -> None:
        self.idx = idx
        self.params = params or {}
        self.fit_x: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None

    def fit(self, x, y) -> None:
        self.fit_x = np.asarray(x)
        self.fit_y = np.asarray(y)

    def predict_proba(self, x):
        x_arr = np.asarray(x)
        return np.tile(np.array([0.3, 0.4, 0.3], dtype=np.float32), (len(x_arr), 1))

    def save(self, _path: str) -> None:
        return None

    def load(self, _path: str) -> None:
        return None


def test_rust_tree_raises_when_rust_fit_fails(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.setenv("FOREX_BOT_TREE_RUST_FALLBACK", "1")
    monkeypatch.setattr(tr, "_fb", object(), raising=False)

    class _TestExpert(tr._RustTreeBase):
        _model_cls = _FailingRustModel

    x = np.array([[0.1, 1.0], [0.2, 1.1], [0.3, 1.2]], dtype=np.float32)
    y = np.array([1, 0, -1], dtype=np.int8)

    model = _TestExpert(params={"n_estimators": 10}, idx=2)
    with pytest.raises(RuntimeError, match="training failed"):
        model.fit(x, y)
    with pytest.raises(RuntimeError, match="prediction failed"):
        model.predict_proba(x)
    with pytest.raises(RuntimeError, match="save failed"):
        model.save("dummy.bin")
    with pytest.raises(RuntimeError, match="load failed"):
        model.load("dummy.bin")


def test_rust_tree_accepts_numpy_arrays_without_pandas(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.setenv("FOREX_BOT_TREE_RUST_FALLBACK", "0")
    monkeypatch.setattr(tr, "_fb", object(), raising=False)

    class _TestExpert(tr._RustTreeBase):
        _model_cls = _RustArrayModel

    x = np.array([[0.1, 1.0], [0.2, 1.1], [0.3, 1.2]], dtype=np.float32)
    y = np.array([1, 0, -1], dtype=np.int8)

    model = _TestExpert(params={"n_estimators": 10}, idx=2)
    model.fit(x, y)
    out = model.predict_proba(x)

    assert model._model is not None
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3, dtype=np.float32), atol=1e-6)

def test_tree_sort_by_datetime_index_prefers_rust_sorted_order(monkeypatch):
    calls = {"sort": 0}

    class _DummyBindings:
        @staticmethod
        def sorted_index_order(idx_ns):
            calls["sort"] += 1
            assert np.asarray(idx_ns, dtype=np.int64).shape[0] == 3
            return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(tmod, "_fb", _DummyBindings(), raising=False)
    idx = pd.to_datetime(
        [
            "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
        ]
    )
    x = pd.DataFrame({"f1": [3.0, 1.0, 2.0]}, index=idx)
    y = np.array([30, 10, 20], dtype=np.int64)

    x_sorted, y_sorted = tmod._sort_by_datetime_index(x, y)

    assert calls["sort"] == 1
    np.testing.assert_allclose(x_sorted["f1"].to_numpy(dtype=np.float64), np.array([1.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(y_sorted, dtype=np.int64), np.array([10, 20, 30], dtype=np.int64))


def test_tree_sort_by_object_datetime_index_without_binding_normalizes_ns(monkeypatch):
    monkeypatch.setattr(tmod, "_fb", None, raising=False)

    class _ArrayFrame:
        def __init__(self, data: dict[str, np.ndarray], index: np.ndarray) -> None:
            self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
            self.columns = list(self._data.keys())
            self.index = np.asarray(index).reshape(-1)

        def __getitem__(self, key: str) -> np.ndarray:
            return self._data[str(key)]

    idx = pd.to_datetime(
        [
            "2024-01-03T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
        ]
    )
    frame = _ArrayFrame(
        {"f1": np.array([3.0, 1.0, 2.0], dtype=np.float64)},
        np.asarray(list(idx), dtype=object),
    )
    y = np.array([30, 10, 20], dtype=np.int64)

    x_sorted, y_sorted = tmod._sort_by_datetime_index(frame, y)

    np.testing.assert_allclose(np.asarray(x_sorted["f1"], dtype=np.float64), np.array([1.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_array_equal(
        np.asarray(x_sorted["index"]),
        pd.to_datetime(
            [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ]
        ).to_numpy(),
    )
    np.testing.assert_array_equal(np.asarray(y_sorted, dtype=np.int64), np.array([10, 20, 30], dtype=np.int64))
