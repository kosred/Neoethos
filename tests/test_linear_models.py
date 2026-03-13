from __future__ import annotations

import warnings

import numpy as np

from forex_bot.models.linear import OnlinePassiveAggressiveExpert


def test_online_pa_fit_emits_no_passive_aggressive_deprecation_warning() -> None:
    expert = OnlinePassiveAggressiveExpert(C=0.7, max_iter=32)
    x = np.array(
        [
            [0.1, 1.0],
            [0.2, 0.8],
            [-0.1, -0.9],
            [-0.2, -1.1],
            [0.0, 0.2],
            [0.3, 1.1],
        ],
        dtype=np.float32,
    )
    y = np.array([1, 1, 2, 2, 0, 1], dtype=np.int8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expert.fit(x, y)

    assert not any(
        "PassiveAggressiveClassifier is deprecated" in str(w.message)
        for w in caught
    )
