from __future__ import annotations

import numpy as np

from forex_bot.training.calibration import ProbabilityCalibrator, SKLEARN_AVAILABLE
from forex_bot.training.conformal import ConformalClassifierGate


def test_probability_calibrator_shapes_and_normalization():
    if not SKLEARN_AVAILABLE:
        return
    rng = np.random.default_rng(42)
    raw = rng.random((600, 3))
    probs = raw / raw.sum(axis=1, keepdims=True)
    y = np.argmax(probs, axis=1).astype(int)
    # Add some noise so calibration isn't degenerate.
    flip = rng.choice(len(y), size=80, replace=False)
    y[flip] = rng.integers(0, 3, size=len(flip))

    cal = ProbabilityCalibrator(method="platt")
    assert cal.fit(probs, y)
    out = cal.predict_proba(probs[:128])
    assert out.shape == (128, 3)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(axis=1), np.ones(128), atol=1e-6, rtol=0.0)


def test_conformal_gate_fit_and_abstain_logic():
    rng = np.random.default_rng(7)
    raw = rng.random((500, 3))
    probs = raw / raw.sum(axis=1, keepdims=True)
    y = np.argmax(probs, axis=1).astype(int)

    gate = ConformalClassifierGate(alpha=0.10)
    assert type(gate).__module__.startswith("forex_bindings")
    assert gate.fit(probs, y)
    assert gate.fitted
    assert 0.0 <= gate.qhat <= 1.0
    assert gate.n_calib == 500

    # Force a permissive threshold to validate abstention behavior explicitly.
    gate.qhat = 1.0
    abstain, set_size = gate.should_abstain(np.array([1 / 3, 1 / 3, 1 / 3], dtype=float), min_set_size=3)
    assert abstain
    assert set_size == 3
