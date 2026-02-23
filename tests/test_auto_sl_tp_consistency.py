from __future__ import annotations

import numpy as np

from forex_bot.core.config import Settings
from forex_bot.strategy.fast_backtest import infer_sl_tp_pips_auto


def _ohlc(n: int = 360) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 14.0 * np.pi, n)
    base = 1.10 + (0.015 * np.sin(x)) + np.linspace(0.0, 0.02, n)
    close = base.astype(np.float64)
    open_ = np.r_[close[0], close[:-1]].astype(np.float64)
    spread = 0.0007 + 0.0002 * np.cos(x * 0.5)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return open_, high.astype(np.float64), low.astype(np.float64), close


def test_infer_sl_tp_auto_uses_shared_stop_target_engine_modes() -> None:
    open_, high, low, close = _ohlc()
    settings = Settings()

    settings.risk.stop_target_mode = "atr"
    atr_out = infer_sl_tp_pips_auto(
        open_prices=open_,
        high_prices=high,
        low_prices=low,
        close_prices=close,
        atr_values=None,
        pip_size=0.0001,
        atr_mult=float(settings.risk.atr_stop_multiplier),
        min_rr=float(settings.risk.min_risk_reward),
        min_dist=float(settings.risk.meta_label_min_dist),
        settings=settings,
    )
    assert atr_out is not None
    assert atr_out[0] > 0.0 and atr_out[1] > 0.0

    settings.risk.stop_target_mode = "structure"
    struct_out = infer_sl_tp_pips_auto(
        open_prices=open_,
        high_prices=high,
        low_prices=low,
        close_prices=close,
        atr_values=None,
        pip_size=0.0001,
        atr_mult=float(settings.risk.atr_stop_multiplier),
        min_rr=float(settings.risk.min_risk_reward),
        min_dist=float(settings.risk.meta_label_min_dist),
        settings=settings,
    )
    assert struct_out is not None
    assert struct_out[0] > 0.0 and struct_out[1] > 0.0
    assert abs(float(struct_out[0]) - float(atr_out[0])) > 1e-9 or abs(float(struct_out[1]) - float(atr_out[1])) > 1e-9


def test_infer_sl_tp_auto_without_settings_still_returns_valid_values() -> None:
    open_, high, low, close = _ohlc()
    out = infer_sl_tp_pips_auto(
        open_prices=open_,
        high_prices=high,
        low_prices=low,
        close_prices=close,
        atr_values=None,
        pip_size=0.0001,
        atr_mult=1.5,
        min_rr=2.0,
        min_dist=0.0,
        settings=None,
    )
    assert out is not None
    assert out[0] > 0.0 and out[1] > 0.0

