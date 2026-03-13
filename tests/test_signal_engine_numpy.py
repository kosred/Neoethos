from __future__ import annotations

import numpy as np

from forex_bot.core.config import Settings
from forex_bot.domain.events import PreparedDataset
from forex_bot.features import engine as engine_mod
from forex_bot.features.engine import SignalEngine


class _CaptureModel:
    def __init__(self, *, buy_prob: float = 0.7) -> None:
        self.buy_prob = float(buy_prob)
        self.last_X: np.ndarray | None = None

    def predict_proba(self, X):  # noqa: ANN001,N803
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.last_X = arr
        n = int(arr.shape[0])
        out = np.zeros((n, 3), dtype=np.float64)
        out[:, 0] = 1.0 - self.buy_prob
        out[:, 1] = self.buy_prob
        out[:, 2] = 0.0
        return out


class _CaptureMetaBlender:
    def __init__(self, *, buy_prob: float = 0.8) -> None:
        self.buy_prob = float(buy_prob)
        self.last_payload = None

    def predict_proba(self, payload):  # noqa: ANN001,N803
        self.last_payload = payload
        if isinstance(payload, dict) and "X" in payload:
            n = int(np.asarray(payload["X"]).shape[0])
        else:
            n = len(payload)
        out = np.zeros((n, 3), dtype=np.float64)
        out[:, 0] = 1.0 - self.buy_prob
        out[:, 1] = self.buy_prob
        out[:, 2] = 0.0
        return out


class _ArrayFrame:
    def __init__(self, data, index):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def test_generate_ensemble_signals_numpy_path_with_pandas_block(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    settings = Settings()
    engine = SignalEngine(settings)
    model = _CaptureModel()
    engine.models = {"m1": model}
    engine.selected_features = ["adx", "missing_col", "close"]

    X = np.array(
        [
            [1.1000, 18.0, 0.0010],
            [1.1010, 23.0, 0.0011],
            [1.1020, 31.0, 0.0012],
        ],
        dtype=np.float32,
    )
    dataset = PreparedDataset(
        X=X,
        y=np.zeros(len(X), dtype=np.int8),
        index=np.array([10, 11, 12], dtype=np.int64),
        feature_names=["close", "adx", "atr14"],
        metadata=None,
        labels=None,
    )

    result = engine.generate_ensemble_signals(dataset)

    assert isinstance(result.signals, np.ndarray)
    assert result.signals.shape == (3,)
    assert int(result.signal) == 1
    assert model.last_X is not None
    assert model.last_X.shape == (3, 3)
    np.testing.assert_allclose(model.last_X[:, 0], X[:, 1], rtol=0, atol=1e-8)
    np.testing.assert_allclose(model.last_X[:, 1], np.zeros(3, dtype=np.float32), rtol=0, atol=1e-8)
    np.testing.assert_allclose(model.last_X[:, 2], X[:, 0], rtol=0, atol=1e-8)
    assert float(result.meta_features.get("market_volatility", 0.0)) > 0.0
    assert str(result.meta_features.get("regime_bucket")) == "trend"


def test_generate_ensemble_signals_frame_like_path(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    settings = Settings()
    engine = SignalEngine(settings)
    model = _CaptureModel()
    engine.models = {"m1": model}
    engine.selected_features = ["adx", "missing_col", "close"]

    close = np.array([1.1000, 1.1010, 1.1020], dtype=np.float32)
    adx = np.array([18.0, 23.0, 31.0], dtype=np.float32)
    atr = np.array([0.0010, 0.0011, 0.0012], dtype=np.float32)
    X = _ArrayFrame(
        {"close": close, "adx": adx, "atr14": atr},
        index=np.array([20, 21, 22], dtype=np.int64),
    )
    dataset = PreparedDataset(
        X=X,
        y=np.zeros(len(X), dtype=np.int8),
        index=np.array([20, 21, 22], dtype=np.int64),
        feature_names=[],
        metadata=None,
        labels=None,
    )

    result = engine.generate_ensemble_signals(dataset)

    assert isinstance(result.signals, np.ndarray)
    assert result.signals.shape == (3,)
    assert int(result.signal) == 1
    assert model.last_X is not None
    assert model.last_X.shape == (3, 3)
    np.testing.assert_allclose(model.last_X[:, 0], adx, rtol=0, atol=1e-8)
    np.testing.assert_allclose(model.last_X[:, 1], np.zeros(3, dtype=np.float32), rtol=0, atol=1e-8)
    np.testing.assert_allclose(model.last_X[:, 2], close, rtol=0, atol=1e-8)
    assert float(result.meta_features.get("market_volatility", 0.0)) > 0.0
    assert str(result.meta_features.get("regime_bucket")) == "trend"


def test_generate_ensemble_signals_empty_numpy_returns_zero_array() -> None:
    settings = Settings()
    engine = SignalEngine(settings)
    dataset = PreparedDataset(
        X=np.zeros((0, 0), dtype=np.float32),
        y=np.zeros(0, dtype=np.int8),
        index=np.zeros(0, dtype=np.int64),
        feature_names=[],
        metadata=None,
        labels=None,
    )

    result = engine.generate_ensemble_signals(dataset)

    assert isinstance(result.signals, np.ndarray)
    assert result.signals.shape == (0,)
    assert int(result.signal) == 0
    np.testing.assert_allclose(result.probs, np.zeros(3, dtype=float), rtol=0, atol=0.0)


def test_generate_ensemble_signals_meta_blender_uses_numpy_payload_dict() -> None:
    settings = Settings()
    engine = SignalEngine(settings)
    model = _CaptureModel(buy_prob=0.65)
    blender = _CaptureMetaBlender(buy_prob=0.9)
    engine.models = {"m1": model}
    engine.meta_blender = blender

    X = np.array(
        [
            [1.1000, 18.0, 0.0010],
            [1.1010, 23.0, 0.0011],
            [1.1020, 31.0, 0.0012],
        ],
        dtype=np.float32,
    )
    dataset = PreparedDataset(
        X=X,
        y=np.zeros(len(X), dtype=np.int8),
        index=np.array([10, 11, 12], dtype=np.int64),
        feature_names=["close", "adx", "atr14"],
        metadata=None,
        labels=None,
    )

    result = engine.generate_ensemble_signals(dataset)

    assert int(result.signal) == 1
    assert blender.last_payload is not None
    assert isinstance(blender.last_payload, dict)
    assert "X" in blender.last_payload and "feature_names" in blender.last_payload
    x_meta = np.asarray(blender.last_payload["X"], dtype=np.float32)
    assert x_meta.shape == (3, 1)
    assert blender.last_payload["feature_names"] == ["m1_buy"]


def test_generate_ensemble_signals_uses_rust_align_feature_matrix_when_available(monkeypatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_align_feature_matrix(src_matrix, src_col_idx, dst_col_idx, dst_width):
        calls["count"] += 1
        out = np.zeros((int(np.asarray(src_matrix).shape[0]), int(dst_width)), dtype=np.float32)
        out[:, 0] = 7.0
        return out

    fake = type("_Fake", (), {"align_feature_matrix": staticmethod(_fake_align_feature_matrix)})()
    monkeypatch.setattr(engine_mod, "_fb", fake, raising=False)

    settings = Settings()
    engine = SignalEngine(settings)
    model = _CaptureModel()
    engine.models = {"m1": model}
    engine.selected_features = ["adx"]

    X = np.array(
        [
            [1.1000, 18.0, 0.0010],
            [1.1010, 23.0, 0.0011],
            [1.1020, 31.0, 0.0012],
        ],
        dtype=np.float32,
    )
    dataset = PreparedDataset(
        X=X,
        y=np.zeros(len(X), dtype=np.int8),
        index=np.array([10, 11, 12], dtype=np.int64),
        feature_names=["close", "adx", "atr14"],
        metadata=None,
        labels=None,
    )

    _ = engine.generate_ensemble_signals(dataset)
    assert calls["count"] >= 1
    assert model.last_X is not None
    assert model.last_X.shape == (3, 1)
    np.testing.assert_allclose(model.last_X[:, 0], np.full(3, 7.0, dtype=np.float32), rtol=0, atol=1e-8)
