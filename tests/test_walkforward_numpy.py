from __future__ import annotations

import numpy as np
from tests._compat_pd import pd

from forex_bot.training import walkforward as wf
from forex_bot.training.walkforward import embargoed_walkforward_backtest


def test_walkforward_backtest_runs_with_numpy_and_pandas_block(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    monkeypatch.setenv("FOREX_BOT_WALKFORWARD_EMBARGO", "30")

    n = 1200
    close = np.linspace(1.0, 1.2, n, dtype=np.float64)
    df = close.reshape(-1, 1)
    signals = np.where(np.arange(n) % 7 == 0, 1, 0).astype(np.int8, copy=False)

    metrics = embargoed_walkforward_backtest(
        df=df,
        signals=signals,
        train_ratio=0.7,
        n_splits=5,
        embargo_minutes=60,
        timeframe_minutes=5,
        use_gpu=False,
    )

    assert int(metrics.get("walk_forward_splits", 0)) >= 1
    assert "avg_pnl" in metrics
    assert "splits" in metrics


def test_extract_index_ns_normalizes_object_datetime_index() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    frame = type("_Frame", (), {"index": np.asarray(list(idx), dtype=object)})()
    expected = np.asarray([int(ts.value) if hasattr(ts, "value") else int(np.datetime64(ts, "ns").astype(np.int64)) for ts in list(idx)], dtype=np.int64)

    out = wf._extract_index_ns(frame, 3)

    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, expected)


def test_extract_day_keys_uses_rust_binding_when_available(monkeypatch) -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    frame = type("_Frame", (), {"index": idx})()
    calls = {"derive": 0}

    class _DummyBindings:
        @staticmethod
        def derive_time_index_arrays(index_ns):
            calls["derive"] += 1
            n = int(np.asarray(index_ns, dtype=np.int64).shape[0])
            return (
                np.asarray(index_ns, dtype=np.int64) // 1_000_000,
                np.full(n, 77, dtype=np.int64),
                np.full(n, 20250101, dtype=np.int64),
            )

    monkeypatch.setattr(wf, "_fb", _DummyBindings(), raising=False)
    out = wf._extract_day_keys(frame, 3)

    assert calls["derive"] == 1
    np.testing.assert_array_equal(out, np.full(3, 20250101, dtype=np.int64))
