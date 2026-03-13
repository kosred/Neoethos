from __future__ import annotations

import numpy as np

from forex_bot.models.label_utils import (
    margins_to_probs,
    probs_to_three_class,
    remap_labels_neutral_buy_sell,
    remap_labels_sell_neutral_buy,
)
from forex_bot.models import label_utils as label_mod


def test_label_remap_neutral_buy_sell():
    y = np.array([-1, 0, 1, 2, 9], dtype=int)
    out = remap_labels_neutral_buy_sell(y)
    np.testing.assert_array_equal(out, np.array([2, 0, 1, 2, 2], dtype=int))


def test_label_remap_sell_neutral_buy():
    y = np.array([-1, 0, 1], dtype=int)
    out = remap_labels_sell_neutral_buy(y)
    np.testing.assert_array_equal(out, np.array([0, 1, 2], dtype=int))


def test_probs_to_three_class_with_custom_mapping():
    probs = np.array([[0.2, 0.5, 0.3]], dtype=float)
    classes = [0, 1, 2]  # sell, neutral, buy
    out = probs_to_three_class(probs, classes, class_to_output={0: 2, 1: 0, 2: 1})
    np.testing.assert_allclose(out, np.array([[0.5, 0.3, 0.2]], dtype=float), atol=1e-12, rtol=0.0)


def test_label_utils_prefers_rust_bindings_when_available(monkeypatch):
    calls = {"nbs": 0, "snb": 0, "margins": 0}

    def _fake_nbs(y):
        calls["nbs"] += 1
        return np.array([9, 8, 7], dtype=np.int64)

    def _fake_snb(y):
        calls["snb"] += 1
        return np.array([3, 4, 5], dtype=np.int64)

    def _fake_margins(decision):
        calls["margins"] += 1
        arr = np.asarray(decision, dtype=np.float64).reshape(-1)
        out = np.zeros((arr.size, 3), dtype=np.float64)
        out[:, 1] = 1.0
        return out

    fake = type(
        "_Fake",
        (),
        {
            "remap_labels_neutral_buy_sell": staticmethod(_fake_nbs),
            "remap_labels_sell_neutral_buy": staticmethod(_fake_snb),
            "margins_to_probs": staticmethod(_fake_margins),
        },
    )()
    monkeypatch.setattr(label_mod, "_fb", fake, raising=False)

    np.testing.assert_array_equal(label_mod.remap_labels_neutral_buy_sell(np.array([-1, 0, 1])), np.array([9, 8, 7]))
    np.testing.assert_array_equal(label_mod.remap_labels_sell_neutral_buy(np.array([-1, 0, 1])), np.array([3, 4, 5]))
    np.testing.assert_allclose(margins_to_probs(np.array([0.1, -0.2], dtype=np.float64)), np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
    assert calls == {"nbs": 1, "snb": 1, "margins": 1}
