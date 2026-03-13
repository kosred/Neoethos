import numpy as np
from tests._compat_pd import pd

from forex_bot.training import ensemble as emod
from forex_bot.training.ensemble import MetaBlender


class _ArrayFrame:
    def __init__(self, data, index):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self.index = np.asarray(index).reshape(-1)

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


class _StubProbaModel:
    def __init__(self, probs: np.ndarray, classes: list[int]) -> None:
        self._probs = np.asarray(probs, dtype=float)
        self.classes_ = np.asarray(classes, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        n = len(X)
        base = self._probs.reshape(1, -1) if self._probs.ndim == 1 else self._probs
        if len(base) == n:
            return base
        return np.repeat(base[:1], n, axis=0)


class _CaptureFitModel:
    def __init__(self) -> None:
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.classes_ = np.asarray([0, 1], dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        self.fit_X = np.asarray(X, dtype=float)
        self.fit_y = np.asarray(y, dtype=int)
        uniq = np.unique(self.fit_y)
        self.classes_ = uniq.astype(int, copy=False) if uniq.size else np.asarray([0, 1], dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.zeros(len(X), dtype=int)


class _CapturePredictProbaModel:
    def __init__(self) -> None:
        self.classes_ = np.asarray([0, 1, -1], dtype=int)
        self.last_X: np.ndarray | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        self.last_X = np.asarray(X, dtype=np.float32)
        n = int(self.last_X.shape[0])
        return np.tile(np.array([[0.2, 0.7, 0.1]], dtype=float), (n, 1))


def test_meta_blender_reorders_to_neutral_buy_sell():
    blender = MetaBlender()
    blender.model = _StubProbaModel(probs=np.array([0.10, 0.20, 0.70]), classes=[-1, 0, 1])  # [-1,0,1] order
    blender.feature_columns = pd.Index(["f1"])

    out = blender.predict_proba(pd.DataFrame({"f1": [1.0, 2.0]}))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], [0.20, 0.70, 0.10], rtol=0, atol=1e-12)


def test_meta_blender_pads_missing_class():
    blender = MetaBlender()
    blender.model = _StubProbaModel(probs=np.array([0.25, 0.75]), classes=[-1, 1])  # sell,buy only
    blender.feature_columns = pd.Index(["f1"])

    out = blender.predict_proba(pd.DataFrame({"f1": [1.0]}))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], [0.0, 0.75, 0.25], rtol=0, atol=1e-12)


def test_meta_blender_predict_proba_accepts_ndarray_input():
    blender = MetaBlender()
    blender.model = _StubProbaModel(probs=np.array([0.40, 0.50, 0.10]), classes=[0, 1, -1])
    blender.feature_columns = ["f1", "f2"]

    out = blender.predict_proba(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], [0.40, 0.50, 0.10], rtol=0, atol=1e-12)


def test_meta_blender_predict_proba_aligns_named_numpy_payload():
    blender = MetaBlender()
    blender.model = _StubProbaModel(probs=np.array([0.33, 0.34, 0.33]), classes=[0, 1, -1])
    blender.feature_columns = ["left", "right", "missing"]

    payload = {
        "X": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        "feature_names": ["right", "left"],
    }
    out = blender.predict_proba(payload)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], [0.33, 0.34, 0.33], rtol=0, atol=1e-12)


def test_meta_blender_fit_sorts_by_index_for_numpy_payload():
    blender = MetaBlender()
    capture = _CaptureFitModel()
    blender.model = capture
    payload = {
        "X": np.array([[3.0], [1.0], [2.0]], dtype=np.float32),
        "y": np.array([1, 0, 1], dtype=np.int8),
        "feature_names": ["f1"],
        "index": np.array([3, 1, 2], dtype=np.int64),
    }

    blender.fit(payload, val_ratio=0.34)
    assert capture.fit_X is not None
    np.testing.assert_allclose(capture.fit_X[:, 0], np.array([1.0, 2.0]), rtol=0, atol=1e-12)


def test_meta_blender_fit_fallback_sort_uses_normalized_index_helper(monkeypatch):
    blender = MetaBlender()
    capture = _CaptureFitModel()
    blender.model = capture
    calls = {"index": 0}

    monkeypatch.setattr(emod, "_fb", None, raising=False)

    def _index_to_ns_int64(index):
        calls["index"] += 1
        assert len(np.asarray(index, dtype=object).reshape(-1)) == 3
        return np.array([3, 1, 2], dtype=np.int64)

    monkeypatch.setattr(emod, "_index_to_ns_int64", _index_to_ns_int64, raising=False)
    payload = {
        "X": np.array([[3.0], [1.0], [2.0]], dtype=np.float32),
        "y": np.array([1, 0, 1], dtype=np.int8),
        "feature_names": ["f1"],
        "index": np.array([object(), object(), object()], dtype=object),
    }

    blender.fit(payload, val_ratio=0.34)
    assert calls["index"] == 1
    assert capture.fit_X is not None
    np.testing.assert_allclose(capture.fit_X[:, 0], np.array([1.0, 2.0]), rtol=0, atol=1e-12)


def test_meta_blender_fit_fallback_prefers_rust_sorted_index_order(monkeypatch):
    blender = MetaBlender()
    capture = _CaptureFitModel()
    blender.model = capture
    calls = {"sort": 0}

    monkeypatch.setattr(emod, "_fb", None, raising=False)

    def _rust_sorted_index_order(index):
        calls["sort"] += 1
        assert len(np.asarray(index, dtype=object).reshape(-1)) == 3
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(emod, "_rust_sorted_index_order", _rust_sorted_index_order, raising=False)
    monkeypatch.setattr(emod, "_index_to_ns_int64", lambda index: np.array([3, 1, 2], dtype=np.int64), raising=False)
    payload = {
        "X": np.array([[3.0], [1.0], [2.0]], dtype=np.float32),
        "y": np.array([1, 0, 1], dtype=np.int8),
        "feature_names": ["f1"],
        "index": np.array([object(), object(), object()], dtype=object),
    }

    blender.fit(payload, val_ratio=0.34)
    assert calls["sort"] == 1
    assert capture.fit_X is not None
    np.testing.assert_allclose(capture.fit_X[:, 0], np.array([1.0, 2.0]), rtol=0, atol=1e-12)


def test_meta_blender_sorted_time_order_returns_none_when_monotonic() -> None:
    out = emod._sorted_time_order(np.array([1, 2, 3], dtype=np.int64), 3)
    assert out is None


def test_meta_blender_fit_uses_rust_sort_for_numpy_payload(monkeypatch):
    blender = MetaBlender()
    capture = _CaptureFitModel()
    blender.model = capture
    calls = {"sort": 0}

    def _sort(x, y, idx):
        calls["sort"] += 1
        return (
            np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            np.array([0, 1, 1], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
        )

    monkeypatch.setattr(emod, "_fb", type("_Dummy", (), {"sort_rows_with_labels_by_index": staticmethod(_sort)})(), raising=False)
    payload = {
        "X": np.array([[3.0], [1.0], [2.0]], dtype=np.float32),
        "y": np.array([1, 0, 1], dtype=np.int8),
        "feature_names": ["f1"],
        "index": np.array([3, 1, 2], dtype=np.int64),
    }

    blender.fit(payload, val_ratio=0.34)
    assert calls["sort"] == 1
    assert capture.fit_X is not None
    np.testing.assert_allclose(capture.fit_X[:, 0], np.array([1.0, 2.0]), rtol=0, atol=1e-12)


def test_meta_blender_fit_accepts_frame_like_without_iloc():
    blender = MetaBlender()
    capture = _CaptureFitModel()
    blender.model = capture
    frame = _ArrayFrame(
        {
            "f1": np.array([3.0, 1.0, 2.0], dtype=np.float32),
            "symbol": np.array(["EURUSD", "GBPUSD", "EURUSD"], dtype=object),
            "label": np.array([1, 0, 1], dtype=np.int8),
        },
        index=np.array([3, 1, 2], dtype=np.int64),
    )

    blender.fit(frame, val_ratio=0.34)
    assert capture.fit_X is not None
    np.testing.assert_allclose(capture.fit_X[:, 0], np.array([1.0, 2.0]), rtol=0, atol=1e-12)
    assert blender.feature_columns is not None
    assert "sym_EURUSD" in blender.feature_columns
    assert "sym_GBPUSD" in blender.feature_columns


def test_meta_blender_fit_uses_rust_sort_for_frame_like_without_iloc(monkeypatch):
    blender = MetaBlender()
    capture = _CaptureFitModel()
    blender.model = capture
    calls = {"sort": 0}

    def _sort(x, y, idx):
        calls["sort"] += 1
        return (
            np.array(
                [
                    [1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            np.array([0, 1, 1], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
        )

    monkeypatch.setattr(emod, "_fb", type("_Dummy", (), {"sort_rows_with_labels_by_index": staticmethod(_sort)})(), raising=False)
    frame = _ArrayFrame(
        {
            "f1": np.array([3.0, 1.0, 2.0], dtype=np.float32),
            "symbol": np.array(["EURUSD", "GBPUSD", "EURUSD"], dtype=object),
            "label": np.array([1, 0, 1], dtype=np.int8),
        },
        index=np.array([3, 1, 2], dtype=np.int64),
    )

    blender.fit(frame, val_ratio=0.34)
    assert calls["sort"] == 1
    assert capture.fit_X is not None
    np.testing.assert_allclose(capture.fit_X[:, 0], np.array([1.0, 2.0]), rtol=0, atol=1e-12)


def test_meta_blender_predict_proba_maps_symbol_from_frame_like_without_iloc():
    blender = MetaBlender()
    capture = _CapturePredictProbaModel()
    blender.model = capture
    blender.feature_columns = ["f1", "sym_EURUSD"]
    frame = _ArrayFrame(
        {
            "f1": np.array([1.0, 2.0], dtype=np.float32),
            "symbol": np.array(["EURUSD", "GBPUSD"], dtype=object),
        },
        index=np.array([0, 1], dtype=np.int64),
    )

    out = blender.predict_proba(frame)
    assert out.shape == (2, 3)
    assert capture.last_X is not None
    np.testing.assert_allclose(capture.last_X[:, 1], np.array([1.0, 0.0], dtype=np.float32), rtol=0, atol=1e-12)

