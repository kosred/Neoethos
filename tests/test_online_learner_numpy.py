from __future__ import annotations

import numpy as np

from forex_bot.training.online_learner import OnlineLearner


class _ArrayFrame:
    def __init__(self, data, index):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self.index = np.asarray(index).reshape(-1)

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def test_is_repeat_mistake_accepts_numpy_features() -> None:
    learner = OnlineLearner("models")
    learner.loss_archive = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

    x_now = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert learner.is_repeat_mistake(x_now, threshold=0.80) is True


def test_is_repeat_mistake_skips_mismatched_feature_width() -> None:
    learner = OnlineLearner("models")
    learner.loss_archive = [np.array([1.0, 2.0], dtype=np.float32)]

    x_now = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert learner.is_repeat_mistake(x_now, threshold=0.10) is False


def test_add_sample_numpy_triggers_update_hook() -> None:
    learner = OnlineLearner("models", min_samples_for_update=2, update_frequency=2)
    calls = {"update": 0}

    def _stub_update() -> bool:
        calls["update"] += 1
        return True

    learner.update_models = _stub_update  # type: ignore[assignment]

    learner.add_sample(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), np.array([-1], dtype=np.int8))
    learner.add_sample(np.array([[3.0, 2.0, 1.0]], dtype=np.float32), np.array([1], dtype=np.int8))

    assert calls["update"] == 1
    assert len(learner.loss_archive) == 1
    assert isinstance(learner.buffer[0][0], np.ndarray)
    assert isinstance(learner.buffer[0][1], np.ndarray)


def test_is_repeat_mistake_accepts_frame_like_features() -> None:
    learner = OnlineLearner("models")
    learner.loss_archive = [np.array([1.0, 2.0], dtype=np.float32)]
    frame = _ArrayFrame({"f0": np.array([1.0]), "f1": np.array([2.0])}, index=np.array([0], dtype=np.int64))
    assert learner.is_repeat_mistake(frame, threshold=0.80) is True


def test_add_sample_accepts_frame_like_x_and_y() -> None:
    learner = OnlineLearner("models", min_samples_for_update=10, update_frequency=10)
    x = _ArrayFrame({"f0": np.array([1.0]), "f1": np.array([2.0])}, index=np.array([0], dtype=np.int64))
    y = _ArrayFrame({"label": np.array([-1], dtype=np.int8)}, index=np.array([0], dtype=np.int64))

    learner.add_sample(x, y)

    assert len(learner.buffer) == 1
    x_buf, y_buf, _w = learner.buffer[0]
    assert isinstance(x_buf, np.ndarray)
    assert isinstance(y_buf, np.ndarray)
    assert x_buf.shape == (1, 2)
    assert y_buf.shape == (1,)
    assert len(learner.loss_archive) == 1
