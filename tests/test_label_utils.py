from __future__ import annotations

import numpy as np

from forex_bot.models.label_utils import (
    probs_to_three_class,
    remap_labels_neutral_buy_sell,
    remap_labels_sell_neutral_buy,
)


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
