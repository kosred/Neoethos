from __future__ import annotations

import numpy as np
import pytest

import forex_bindings
from forex_bot.training import probability_utils as prob_utils
from forex_bot.training.evaluation import pad_probs as evaluation_pad_probs
from forex_bot.training.ensemble import pad_probs as ensemble_pad_probs


def test_forex_bindings_exposes_probability_padding_helper() -> None:
    assert hasattr(forex_bindings, "pad_probs_neutral_buy_sell")


def test_forex_bindings_exposes_threshold_signal_helper() -> None:
    assert hasattr(forex_bindings, "threshold_signals_and_accuracy")


def test_probability_padding_supports_explicit_class_mapping() -> None:
    raw = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
        ],
        dtype=np.float64,
    )
    classes = [1, 0, -1]
    expected = np.array(
        [
            [0.2, 0.7, 0.1],
            [0.6, 0.1, 0.3],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(evaluation_pad_probs(raw, classes=classes), expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(ensemble_pad_probs(raw, classes=classes), expected, atol=1e-12, rtol=0.0)


def test_probability_padding_supports_binary_and_single_column_shapes() -> None:
    binary = np.array([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)
    single = np.array([[0.9], [0.1]], dtype=np.float64)

    expected_binary = np.array([[0.8, 0.2, 0.0], [0.25, 0.75, 0.0]], dtype=np.float64)
    expected_single = np.array([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]], dtype=np.float64)

    np.testing.assert_allclose(evaluation_pad_probs(binary), expected_binary, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(ensemble_pad_probs(binary), expected_binary, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(evaluation_pad_probs(single), expected_single, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(ensemble_pad_probs(single), expected_single, atol=1e-12, rtol=0.0)


def test_threshold_signals_and_accuracy_prefers_rust_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_thresholded(probs, conf_threshold, y_true=None):
        calls["count"] += 1
        return np.array([1, 0, -1], dtype=np.int8), 0.75

    fake = type("_Fake", (), {"threshold_signals_and_accuracy": staticmethod(_fake_thresholded)})()
    monkeypatch.setattr(prob_utils, "_fb", fake, raising=False)

    probs = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    y_true = np.array([1, 0, -1], dtype=np.int64)

    signals, accuracy = prob_utils.threshold_signals_and_accuracy(
        probs,
        conf_threshold=0.66,
        y_true=y_true,
    )

    np.testing.assert_array_equal(signals, np.array([1, 0, -1], dtype=np.int8))
    assert accuracy == 0.75
    assert calls["count"] == 1


def test_binding_threshold_signals_and_accuracy_fails_closed_on_nan_rows() -> None:
    probs = np.array(
        [
            [0.0, np.nan, 0.9],
            [0.0, 0.8, 0.1],
            [0.0, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    y_true = np.array([0, 1, -1], dtype=np.int64)

    signals, accuracy = forex_bindings.threshold_signals_and_accuracy(
        probs,
        0.66,
        y_true,
    )

    np.testing.assert_array_equal(np.asarray(signals, dtype=np.int8), np.array([0, 1, -1], dtype=np.int8))
    assert float(accuracy) == 1.0


def test_probability_utils_threshold_signals_and_accuracy_fails_closed_on_nan_rows() -> None:
    probs = np.array(
        [
            [0.0, np.nan, 0.9],
            [0.0, 0.8, 0.1],
            [0.0, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    y_true = np.array([0, 1, -1], dtype=np.int64)

    signals, accuracy = prob_utils.threshold_signals_and_accuracy(
        probs,
        conf_threshold=0.66,
        y_true=y_true,
    )

    np.testing.assert_array_equal(signals, np.array([0, 1, -1], dtype=np.int8))
    assert accuracy == 1.0

