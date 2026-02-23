from __future__ import annotations

import numpy as np
import pandas as pd

from forex_bot.core.config import Settings
from forex_bot.strategy.stop_target import infer_stop_target_pips


def _make_ohlc(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    base = np.linspace(1.05, 1.15, n)
    wave = 0.004 * np.sin(np.linspace(0.0, 18.0 * np.pi, n))
    noise = rng.normal(0.0, 0.0006, n)
    close = base + wave + noise
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0008, 0.0002, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_stop_target_modes_return_finite_values():
    df = _make_ohlc()
    settings = Settings()
    pip_size = 0.0001

    settings.risk.stop_target_mode = "atr"
    atr = infer_stop_target_pips(df, settings=settings, pip_size=pip_size, signal=1)
    assert atr is not None
    assert all(np.isfinite(np.asarray(atr, dtype=float)))
    assert atr[0] > 0.0 and atr[1] > 0.0 and atr[2] > 0.1

    settings.risk.stop_target_mode = "structure"
    struct = infer_stop_target_pips(df, settings=settings, pip_size=pip_size, signal=1)
    assert struct is not None
    assert all(np.isfinite(np.asarray(struct, dtype=float)))
    assert struct[0] > 0.0 and struct[1] > 0.0 and struct[2] > 0.1

    settings.risk.stop_target_mode = "blend"
    blend = infer_stop_target_pips(df, settings=settings, pip_size=pip_size, signal=1)
    assert blend is not None
    assert all(np.isfinite(np.asarray(blend, dtype=float)))
    assert blend[0] > 0.0 and blend[1] > 0.0 and blend[2] > 0.1

    lo = min(float(atr[0]), float(struct[0]))
    hi = max(float(atr[0]), float(struct[0]))
    assert lo <= float(blend[0]) <= hi

