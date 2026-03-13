from __future__ import annotations

import numpy as np
import pytest
from tests._compat_pd import pd

from forex_bot.training import evaluation as eval_mod


def test_quick_backtest_accepts_numpy_payload_with_no_pandas() -> None:
    close = np.array([1.0000, 1.0002, 0.9998, 1.0001, 1.0004], dtype=np.float64)
    signals = np.array([1, 0, -1, 1, 0], dtype=np.int8)
    payload = {"close": close}

    out = eval_mod.quick_backtest(payload, signals)
    assert isinstance(out, dict)
    assert "accuracy" in out and "pnl_score" in out and "win_rate" in out
    assert np.isfinite(float(out["accuracy"]))
    assert np.isfinite(float(out["pnl_score"]))
    assert np.isfinite(float(out["win_rate"]))


def test_quick_backtest_prefers_rust_metrics_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_quick(close, signals):
        calls["count"] += 1
        return 0.75, 3.0, 0.5, 2

    fake = type("_Fake", (), {"quick_backtest_metrics": staticmethod(_fake_quick)})()
    monkeypatch.setattr(eval_mod, "_fb", fake, raising=False)

    close = np.array([1.0, 1.1, 1.2], dtype=np.float64)
    signals = np.array([1, 0, -1], dtype=np.int8)
    out = eval_mod.quick_backtest({"close": close}, signals)

    assert out == {"accuracy": 0.75, "pnl_score": 3.0, "win_rate": 0.5}
    assert calls["count"] == 1


def test_probs_to_signals_prefers_rust_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_probs_to_signals(probs):
        calls["count"] += 1
        return np.array([1, -1], dtype=np.int8)

    fake = type("_Fake", (), {"probs_to_signals": staticmethod(_fake_probs_to_signals)})()
    monkeypatch.setattr(eval_mod, "_fb", fake, raising=False)

    probs = np.array([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]], dtype=np.float64)
    out = eval_mod.probs_to_signals(probs)

    np.testing.assert_array_equal(out, np.array([1, -1], dtype=int))
    assert calls["count"] == 1


def test_prop_backtest_accepts_numpy_payload_and_ns_index(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 8
    close = np.linspace(1.0, 1.004, num=n, dtype=np.float64)
    high = close + 0.0003
    low = close - 0.0003
    signals = np.array([1, 0, -1, 1, 0, -1, 1, 0], dtype=np.int8)
    idx_ns = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000

    captured: dict[str, np.ndarray] = {}

    def _fake_fast(**kwargs):
        captured["month_indices"] = np.asarray(kwargs["month_indices"], dtype=np.int64)
        captured["day_indices"] = np.asarray(kwargs["day_indices"], dtype=np.int64)
        captured["signals"] = np.asarray(kwargs["signals"], dtype=np.int8)
        return np.arange(11, dtype=np.float64)

    monkeypatch.setattr(eval_mod, "fast_evaluate_strategy", _fake_fast)
    monkeypatch.setattr(eval_mod, "infer_pip_metrics", lambda _symbol: (0.0001, 10.0))

    payload = {
        "close": close,
        "high": high,
        "low": low,
        "index": idx_ns,
        "symbol": "EURUSD",
    }
    out = eval_mod.prop_backtest(payload, signals)

    assert out["net_profit"] == 0.0
    assert out["pnl_score"] == 0.0
    assert out["sharpe"] == 1.0
    assert out["max_dd"] == 3.0
    assert out["daily_dd"] == 10.0
    assert "month_indices" in captured and "day_indices" in captured and "signals" in captured
    assert captured["month_indices"].shape[0] == n
    assert captured["day_indices"].shape[0] == n
    assert captured["signals"].shape[0] == n


def test_prop_backtest_uses_rust_time_index_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 5
    close = np.linspace(1.0, 1.001, num=n, dtype=np.float64)
    high = close + 0.0002
    low = close - 0.0002
    signals = np.array([1, 0, -1, 1, 0], dtype=np.int8)
    idx_ns = np.array(
        [
            1_704_067_200_000_000_000,
            1_704_153_600_000_000_000,
            1_704_240_000_000_000_000,
            1_704_326_400_000_000_000,
            1_704_412_800_000_000_000,
        ],
        dtype=np.int64,
    )

    captured: dict[str, np.ndarray] = {}

    def _fake_fast(**kwargs):
        captured["month_indices"] = np.asarray(kwargs["month_indices"], dtype=np.int64)
        captured["day_indices"] = np.asarray(kwargs["day_indices"], dtype=np.int64)
        return np.arange(11, dtype=np.float64)

    fake = type(
        "_Fake",
        (),
        {
            "derive_time_index_arrays": staticmethod(
                lambda index_ns: (
                    np.asarray(index_ns, dtype=np.int64) // 1_000_000,
                    np.full(np.asarray(index_ns).shape[0], 999, dtype=np.int64),
                    np.full(np.asarray(index_ns).shape[0], 888, dtype=np.int64),
                )
            )
        },
    )()
    monkeypatch.setattr(eval_mod, "_fb", fake, raising=False)
    monkeypatch.setattr(eval_mod, "fast_evaluate_strategy", _fake_fast)
    monkeypatch.setattr(eval_mod, "infer_pip_metrics", lambda _symbol: (0.0001, 10.0))

    payload = {
        "close": close,
        "high": high,
        "low": low,
        "index": idx_ns,
        "symbol": "EURUSD",
    }
    _ = eval_mod.prop_backtest(payload, signals)

    np.testing.assert_array_equal(captured["month_indices"], np.full(n, 999, dtype=np.int64))
    np.testing.assert_array_equal(captured["day_indices"], np.full(n, 888, dtype=np.int64))


def test_extract_index_normalizes_object_datetime_index_to_ns() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    payload = {"index": np.asarray(list(idx), dtype=object)}
    expected = np.asarray(
        [int(ts.value) if hasattr(ts, "value") else int(np.datetime64(ts, "ns").astype(np.int64)) for ts in list(idx)],
        dtype=np.int64,
    )

    out = eval_mod._extract_index(payload, 4)

    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, expected)
