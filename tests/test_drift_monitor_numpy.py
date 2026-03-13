import numpy as np

from forex_bot.execution import drift_monitor as dm
from forex_bot.execution.drift_monitor import ConceptDriftMonitor


class _ArrayColumn:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self, dtype=None, copy: bool = False):
        arr = np.asarray(self._values, dtype=dtype) if dtype is not None else np.asarray(self._values)
        return np.array(arr, copy=True) if copy else arr


class _ArrayColumnNoCopyKw:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self):
        return np.asarray(self._values)


class _ArrayFrame:
    def __init__(self, data):
        self._data = {str(k): np.asarray(v) for k, v in data.items()}
        self.columns = list(self._data.keys())

    def __len__(self):
        if not self._data:
            return 0
        return int(next(iter(self._data.values())).shape[0])

    def __getitem__(self, key):
        return self._data[str(key)]


class _FrameWithColumns:
    def __init__(self, data):
        self._data = {str(k): np.asarray(v) for k, v in data.items()}
        self.columns = list(self._data.keys())

    def __len__(self):
        if not self._data:
            return 0
        return int(next(iter(self._data.values())).shape[0])

    def __getitem__(self, key):
        return _ArrayColumn(self._data[str(key)])


class _FrameWithColumnsNoCopyKw:
    def __init__(self, data):
        self._data = {str(k): np.asarray(v) for k, v in data.items()}
        self.columns = list(self._data.keys())

    def __len__(self):
        if not self._data:
            return 0
        return int(next(iter(self._data.values())).shape[0])

    def __getitem__(self, key):
        return _ArrayColumnNoCopyKw(self._data[str(key)])


def test_check_feature_drift_accepts_frame_like_without_iloc():
    monitor = ConceptDriftMonitor()
    monitor.alpha = 0.01
    monitor.feature_stats = {
        "rsi_14": {"mean": 50.0, "std": 5.0, "initialized": True},
    }

    current = _ArrayFrame({"rsi_14": np.array([49.0, 80.0], dtype=np.float32)})
    assert monitor.check_feature_drift(current, threshold=0.0) is True


def test_frame_like_latest_scalar_accepts_column_wrapper_without_take():
    frame = _FrameWithColumns({"adx": np.array([10.0, 30.0], dtype=np.float32)})

    out = dm._frame_like_latest_scalar(frame, "adx")

    assert out == 30.0


def test_frame_like_latest_scalar_accepts_to_numpy_without_copy_kw():
    frame = _FrameWithColumnsNoCopyKw({"adx": np.array([10.0, 30.0], dtype=np.float32)})

    out = dm._frame_like_latest_scalar(frame, "adx")

    assert out == 30.0


def test_initialize_feature_monitor_accepts_to_numpy_without_copy_kw():
    monitor = ConceptDriftMonitor()
    frame = _FrameWithColumnsNoCopyKw({"adx": np.array([10.0] * 12, dtype=np.float32)})

    monitor.initialize_feature_monitor(frame, "EURUSD")

    assert "adx" in getattr(monitor, "feature_stats", {})
