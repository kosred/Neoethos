from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from tests._compat_pd import pd

from forex_bot.core.config import Settings
from forex_bot.strategy import stop_target as stop_target_mod
from forex_bot.strategy.stop_target import infer_stop_target_pips, infer_stop_target_pips_ohlcv


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


def test_stop_target_modes_return_finite_values(monkeypatch):
    monkeypatch.setattr(
        stop_target_mod,
        "_fb",
        SimpleNamespace(infer_stop_target_pips_ohlcv=lambda *_args, **_kwargs: (16.0, 40.0, 2.5)),
        raising=False,
    )
    monkeypatch.setattr(stop_target_mod, "_RUST_STOP_FULL_SUPPORT", True, raising=False)
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


def test_stop_target_ohlcv_array_api_returns_values(monkeypatch):
    monkeypatch.setattr(
        stop_target_mod,
        "_fb",
        SimpleNamespace(infer_stop_target_pips_ohlcv=lambda *_args, **_kwargs: (14.0, 28.0, 2.0)),
        raising=False,
    )
    monkeypatch.setattr(stop_target_mod, "_RUST_STOP_FULL_SUPPORT", True, raising=False)
    df = _make_ohlc()
    settings = Settings()
    settings.risk.stop_target_mode = "blend"
    pip_size = 0.0001

    out = infer_stop_target_pips_ohlcv(
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
        settings=settings,
        pip_size=pip_size,
        signal=1,
    )
    assert out is not None
    assert all(np.isfinite(np.asarray(out, dtype=float)))
    assert float(out[0]) > 0.0 and float(out[1]) > 0.0 and float(out[2]) > 0.1


def test_stop_target_rust_only_blocks_python_fallback_when_backend_unavailable(monkeypatch):
    df = _make_ohlc()
    settings = Settings()
    settings.risk.stop_target_mode = "blend"
    pip_size = 0.0001

    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("FOREX_BOT_STOP_TARGET_ALLOW_PY_FALLBACK", raising=False)
    monkeypatch.setattr(stop_target_mod, "_fb", None, raising=False)
    monkeypatch.setattr(stop_target_mod, "_RUST_STOP_FULL_SUPPORT", None, raising=False)

    out = infer_stop_target_pips_ohlcv(
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
        settings=settings,
        pip_size=pip_size,
        signal=1,
    )
    assert out is None


def test_stop_target_rust_only_opt_in_still_blocks_without_rust_backend(monkeypatch):
    df = _make_ohlc()
    settings = Settings()
    settings.risk.stop_target_mode = "blend"
    pip_size = 0.0001

    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_STOP_TARGET_ALLOW_PY_FALLBACK", "1")
    monkeypatch.setattr(stop_target_mod, "_fb", None, raising=False)
    monkeypatch.setattr(stop_target_mod, "_RUST_STOP_FULL_SUPPORT", None, raising=False)

    out = infer_stop_target_pips_ohlcv(
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
        settings=settings,
        pip_size=pip_size,
        signal=1,
    )
    assert out is None


