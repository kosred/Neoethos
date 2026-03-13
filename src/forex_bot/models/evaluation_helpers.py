from __future__ import annotations

from typing import Any

import numpy as np

try:
    import forex_bindings as _fb
except Exception:  # pragma: no cover
    _fb = None


def probs_to_signals(probs: np.ndarray) -> np.ndarray:
    if _fb is not None and hasattr(_fb, "probs_to_signals"):
        try:
            out = _fb.probs_to_signals(np.asarray(probs, dtype=np.float64))
            return np.asarray(out, dtype=int).reshape(-1)
        except Exception:
            pass
    sig_idx = probs.argmax(axis=1)
    signals = np.zeros(len(probs), dtype=int)
    signals[sig_idx == 1] = 1
    signals[sig_idx == 2] = -1
    return signals


def simple_backtest(df: Any, signals: Any) -> dict[str, Any]:
    if df is None or signals is None or len(df) == 0 or len(signals) != len(df):
        return {}
    close = df["close"].to_numpy()
    sig_arr = signals.to_numpy() if hasattr(signals, "to_numpy") else np.asarray(signals)
    if _fb is not None and hasattr(_fb, "quick_backtest_metrics"):
        try:
            acc, pnl_score, win_rate, trades = _fb.quick_backtest_metrics(
                np.asarray(close, dtype=np.float64),
                np.asarray(sig_arr, dtype=np.int8).reshape(-1),
            )
            return {
                "accuracy": float(acc),
                "pnl_score": float(pnl_score),
                "win_rate": float(win_rate),
                "trades": int(trades),
            }
        except Exception:
            pass
    future = np.roll(close, -1)
    ret = future - close
    pnl = []
    for s, r in zip(np.asarray(sig_arr).reshape(-1), ret, strict=False):
        if s == 0:
            pnl.append(0.0)
        elif s == 1:
            pnl.append(1.0 if r > 0 else -1.0)
        else:
            pnl.append(1.0 if r < 0 else -1.0)
    pnl = np.array(pnl[:-1])
    return {
        "pnl_score": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0.0,
        "trades": int(np.count_nonzero(signals)),
    }

