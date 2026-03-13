from __future__ import annotations

import numpy as np

from forex_bot.models import evaluation_helpers as eval_helpers


class _Series:
    def __init__(self, values):
        self._values = np.asarray(values)

    def __len__(self):
        return int(self._values.size)

    def to_numpy(self):
        return np.asarray(self._values)


class _Frame:
    def __init__(self, close):
        self._close = _Series(close)

    def __getitem__(self, key):
        if key != "close":
            raise KeyError(key)
        return self._close

    def __len__(self):
        return len(self._close.to_numpy())


def test_simple_backtest_prefers_rust_metrics(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_quick(close, signals):
        calls["count"] += 1
        return 0.5, 2.0, 0.25, 4

    fake = type("_Fake", (), {"quick_backtest_metrics": staticmethod(_fake_quick)})()
    monkeypatch.setattr(eval_helpers, "_fb", fake, raising=False)

    out = eval_helpers.simple_backtest(_Frame([1.0, 1.1, 1.2, 1.3]), _Series([1, 0, -1, 1]))

    assert out == {"accuracy": 0.5, "pnl_score": 2.0, "win_rate": 0.25, "trades": 4}
    assert calls["count"] == 1


def test_probs_to_signals_prefers_rust_binding(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_probs_to_signals(probs):
        calls["count"] += 1
        return np.array([0, 1, -1], dtype=np.int8)

    fake = type("_Fake", (), {"probs_to_signals": staticmethod(_fake_probs_to_signals)})()
    monkeypatch.setattr(eval_helpers, "_fb", fake, raising=False)

    out = eval_helpers.probs_to_signals(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=np.float64))

    np.testing.assert_array_equal(out, np.array([0, 1, -1], dtype=int))
    assert calls["count"] == 1
