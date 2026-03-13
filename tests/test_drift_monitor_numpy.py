import numpy as np

from forex_bot.execution.drift_monitor import ConceptDriftMonitor


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


def test_check_feature_drift_accepts_frame_like_without_iloc():
    monitor = ConceptDriftMonitor()
    monitor.alpha = 0.01
    monitor.feature_stats = {
        "rsi_14": {"mean": 50.0, "std": 5.0, "initialized": True},
    }

    current = _ArrayFrame({"rsi_14": np.array([49.0, 80.0], dtype=np.float32)})
    assert monitor.check_feature_drift(current, threshold=0.0) is True

