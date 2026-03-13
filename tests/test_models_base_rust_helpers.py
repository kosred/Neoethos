from __future__ import annotations

import numpy as np

from forex_bot.models import base as base_mod


def test_compute_class_weights_prefers_rust_binding(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_balanced(labels):
        calls["count"] += 1
        return np.array([0, 2], dtype=np.int64), np.array([1.5, 3.5], dtype=np.float64)

    fake = type("_Fake", (), {"balanced_class_weights": staticmethod(_fake_balanced)})()
    monkeypatch.setattr(base_mod, "_fb", fake, raising=False)

    out = base_mod.compute_class_weights(np.array([0, 0, 2], dtype=np.int64))

    assert out == {0: 1.5, 2: 3.5}
    assert calls["count"] == 1


def test_compute_sample_weights_prefers_rust_binding(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_sample_weights(labels):
        calls["count"] += 1
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    fake = type("_Fake", (), {"sample_weights_from_labels": staticmethod(_fake_sample_weights)})()
    monkeypatch.setattr(base_mod, "_fb", fake, raising=False)

    out = base_mod.compute_sample_weights(np.array([0, 1, 2], dtype=np.int64))

    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float32), rtol=0.0, atol=0.0)
    assert calls["count"] == 1
