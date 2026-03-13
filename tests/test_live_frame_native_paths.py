from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from forex_bot.features.engine import SignalEngine


class _ArrayColumn:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self, dtype=None, copy: bool = False):
        arr = np.asarray(self._values, dtype=dtype) if dtype is not None else np.asarray(self._values)
        return np.array(arr, copy=True) if copy else arr


class _Frame:
    def __init__(self, data, index=None, columns=None):
        src = dict(data)
        ordered_cols = list(columns) if columns is not None else list(src.keys())
        self._data = {str(col): np.asarray(src[str(col)]).reshape(-1) for col in ordered_cols}
        self.columns = list(ordered_cols)
        n_rows = len(next(iter(self._data.values()))) if self._data else 0
        self.index = np.asarray(index if index is not None else np.arange(n_rows))

    def __len__(self) -> int:
        return int(self.index.size)

    def __getitem__(self, key):
        return _ArrayColumn(self._data[str(key)])

    def to_numpy(self, dtype=None, copy: bool = False):
        if not self.columns:
            arr = np.zeros((len(self), 0), dtype=np.float32)
        else:
            arr = np.column_stack([self._data[str(col)] for col in self.columns])
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        return np.array(arr, copy=True) if copy else arr


def _make_engine() -> SignalEngine:
    engine = object.__new__(SignalEngine)
    engine.settings = SimpleNamespace(
        risk=SimpleNamespace(
            regime_adx_trend=25.0,
            regime_adx_range=20.0,
            volatility_target=0.0015,
        ),
        models=SimpleNamespace(
            l1_feature_selection_per_regime=True,
            regime_router_enabled=True,
            regime_trend_models=[],
            regime_range_models=[],
            regime_neutral_models=[],
            regime_router_min_models=2,
        ),
    )
    engine.selected_features = ["adx", "atr14"]
    engine.selected_features_by_regime = {}
    engine.models = {}
    return engine


def test_signal_engine_accepts_frame_native_inputs_without_pandas():
    engine = _make_engine()
    frame = _Frame(
        {
            "close": [100.0, 102.0],
            "atr14": [1.0, 2.0],
            "adx": [18.0, 31.0],
        },
        index=[10, 11],
    )

    aligned, names = engine._align_selected_features(frame)

    assert hasattr(aligned, "columns")
    assert hasattr(aligned, "to_numpy")
    assert aligned.columns == ["adx", "atr14"]
    assert names == ["adx", "atr14"]
    assert np.allclose(aligned.to_numpy(dtype=np.float32), np.asarray([[18.0, 1.0], [31.0, 2.0]], dtype=np.float32))

    vol = engine._latest_market_volatility(frame, -1)
    assert vol == np.float64(2.0 / 102.0)
    assert engine._infer_regime_bucket(frame, -1) == "trend"
