from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from forex_bot.strategy import fast_backtest as fb


def _sample_inputs(n: int = 8) -> dict[str, np.ndarray]:
    close = np.linspace(1.0, 1.1, n, dtype=np.float64)
    high = close + 0.001
    low = close - 0.001
    signals = np.where(np.arange(n) % 3 == 0, 1, np.where(np.arange(n) % 3 == 1, -1, 0)).astype(np.int8)
    month = np.arange(n, dtype=np.int64)
    day = np.arange(n, dtype=np.int64)
    return {
        "close_prices": close,
        "high_prices": high,
        "low_prices": low,
        "signals": signals,
        "month_indices": month,
        "day_indices": day,
    }


def test_fast_backtest_prefers_bindings_in_auto_mode(monkeypatch):
    calls = {"bindings": 0, "core": 0}

    def b_fast(*args):
        calls["bindings"] += 1
        return np.arange(11, dtype=np.float64)

    def b_batch(*args):
        calls["bindings"] += 1
        return np.zeros((2, 11), dtype=np.float64)

    def c_fast(*args):
        calls["core"] += 1
        return np.full(11, 9.0, dtype=np.float64)

    def c_batch(*args):
        calls["core"] += 1
        return np.full((2, 11), 9.0, dtype=np.float64)

    monkeypatch.setattr(
        fb,
        "_forex_bindings",
        SimpleNamespace(fast_evaluate_strategy=b_fast, batch_evaluate_strategies=b_batch),
        raising=False,
    )
    monkeypatch.setattr(
        fb,
        "_forex_core",
        SimpleNamespace(fast_evaluate_strategy=c_fast, batch_evaluate_strategies=c_batch),
        raising=False,
    )
    monkeypatch.setenv("FOREX_BOT_BACKTEST_BACKEND", "auto")

    out = fb.fast_evaluate_strategy(**_sample_inputs(), sl_pips=20.0, tp_pips=40.0)
    assert out.shape == (11,)
    assert calls["bindings"] == 1
    assert calls["core"] == 0


def test_batch_backtest_falls_back_to_core_when_bindings_fail(monkeypatch):
    calls = {"core": 0}

    def b_fast(*args):
        return np.zeros(11, dtype=np.float64)

    def b_batch(*args):
        raise RuntimeError("bindings failed")

    def c_fast(*args):
        return np.zeros(11, dtype=np.float64)

    def c_batch(*args):
        calls["core"] += 1
        return np.full((2, 11), 7.0, dtype=np.float64)

    monkeypatch.setattr(
        fb,
        "_forex_bindings",
        SimpleNamespace(fast_evaluate_strategy=b_fast, batch_evaluate_strategies=b_batch),
        raising=False,
    )
    monkeypatch.setattr(
        fb,
        "_forex_core",
        SimpleNamespace(fast_evaluate_strategy=c_fast, batch_evaluate_strategies=c_batch),
        raising=False,
    )
    monkeypatch.setenv("FOREX_BOT_BACKTEST_BACKEND", "auto")

    d = _sample_inputs()
    sig2 = np.vstack([d["signals"], -d["signals"]]).astype(np.int8, copy=False)
    out = fb.batch_evaluate_strategies(
        close_prices=d["close_prices"],
        high_prices=d["high_prices"],
        low_prices=d["low_prices"],
        signals=sig2,
        month_indices=d["month_indices"],
        day_indices=d["day_indices"],
        sl_pips=np.array([20.0, 25.0], dtype=np.float64),
        tp_pips=np.array([40.0, 50.0], dtype=np.float64),
    )

    assert out.shape == (2, 11)
    assert calls["core"] == 1
    np.testing.assert_allclose(out, np.full((2, 11), 7.0, dtype=np.float64))


def test_batch_backtest_forced_bindings_mode_raises(monkeypatch):
    def b_fast(*args):
        return np.zeros(11, dtype=np.float64)

    def b_batch(*args):
        raise RuntimeError("bindings hard failure")

    monkeypatch.setattr(
        fb,
        "_forex_bindings",
        SimpleNamespace(fast_evaluate_strategy=b_fast, batch_evaluate_strategies=b_batch),
        raising=False,
    )
    monkeypatch.setattr(fb, "_forex_core", None, raising=False)
    monkeypatch.setenv("FOREX_BOT_BACKTEST_BACKEND", "bindings")

    d = _sample_inputs()
    sig2 = np.vstack([d["signals"], -d["signals"]]).astype(np.int8, copy=False)
    with pytest.raises(RuntimeError):
        fb.batch_evaluate_strategies(
            close_prices=d["close_prices"],
            high_prices=d["high_prices"],
            low_prices=d["low_prices"],
            signals=sig2,
            month_indices=d["month_indices"],
            day_indices=d["day_indices"],
            sl_pips=np.array([20.0, 25.0], dtype=np.float64),
            tp_pips=np.array([40.0, 50.0], dtype=np.float64),
        )
