from __future__ import annotations

import logging
import os
from typing import Iterable, Tuple
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

_FOREX_BINDINGS_ERROR: Exception | None = None
try:
    import forex_bindings as _forex_bindings
except Exception as exc:  # pragma: no cover - depends on native build
    _forex_bindings = None
    _FOREX_BINDINGS_ERROR = exc

_FOREX_CORE_ERROR: Exception | None = None
try:
    import forex_core as _forex_core
except Exception as exc:  # pragma: no cover - depends on native build
    _forex_core = None
    _FOREX_CORE_ERROR = exc

def _bindings_backtest_available() -> bool:
    return bool(
        _forex_bindings is not None
        and hasattr(_forex_bindings, "fast_evaluate_strategy")
        and hasattr(_forex_bindings, "batch_evaluate_strategies")
    )

def _bindings_pip_metrics_available() -> bool:
    return bool(
        _forex_bindings is not None
        and hasattr(_forex_bindings, "infer_pip_metrics")
    )


def _strict_rust_mode_enabled() -> bool:
    rust_only = str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower()
    if rust_only in {"1", "true", "yes", "on"}:
        return True
    profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
    if profile.startswith("rust"):
        return True
    backend = str(os.environ.get("FOREX_BOT_BACKTEST_BACKEND", "auto") or "auto").strip().lower()
    return backend in {"rust", "bindings", "forex_bindings", "core", "forex_core"}

def _core_backtest_available() -> bool:
    return bool(
        _forex_core is not None
        and hasattr(_forex_core, "fast_evaluate_strategy")
        and hasattr(_forex_core, "batch_evaluate_strategies")
    )


def _backtest_backend_mode() -> str:
    raw = str(os.environ.get("FOREX_BOT_BACKTEST_BACKEND", "auto") or "auto").strip().lower()
    if raw in {"bindings", "binding", "forex_bindings", "rust_bindings"}:
        return "bindings"
    if raw in {"core", "forex_core", "legacy"}:
        return "core"
    if raw in {"python", "py"}:
        logger.warning("Python backtest backend is deprecated; using Rust backend auto selection.")
        return "auto"
    return "auto"


def _select_backtest_backend() -> str | None:
    mode = _backtest_backend_mode()
    if mode == "bindings":
        if _bindings_backtest_available():
            return "bindings"
        return None
    if mode == "core":
        if _core_backtest_available():
            return "core"
        return None
    if _bindings_backtest_available():
        return "bindings"
    if _core_backtest_available():
        return "core"
    return None


def _require_rust() -> None:
    backend = _select_backtest_backend()
    if backend is not None:
        return
    raise ImportError(
        "No Rust backtest backend available (forex_bindings/forex_core)."
    ) from (_FOREX_BINDINGS_ERROR or _FOREX_CORE_ERROR)


def _as_contig(arr: Iterable, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out, dtype=dtype)
    return out


def infer_pip_metrics(
    symbol: str,
    *,
    price: float | None = None,
    account_currency: str = "USD",
    reference_prices: dict[str, float] | None = None,
) -> Tuple[float, float]:
    if not _bindings_pip_metrics_available():
        raise RuntimeError("Rust infer_pip_metrics backend unavailable.")
    try:
        pip_size_rs, pip_value_rs = _forex_bindings.infer_pip_metrics(
            symbol,
            price=price,
            account_currency=account_currency,
            reference_prices=reference_prices,
        )
        pip_size_f = float(pip_size_rs)
        pip_value_f = float(pip_value_rs)
        if not (np.isfinite(pip_size_f) and pip_size_f > 0.0):
            raise RuntimeError(f"Rust infer_pip_metrics returned invalid pip_size: {pip_size_f}")
        if not (np.isfinite(pip_value_f) and pip_value_f > 0.0):
            raise RuntimeError(f"Rust infer_pip_metrics returned invalid pip_value: {pip_value_f}")
        return pip_size_f, pip_value_f
    except Exception as exc:
        raise RuntimeError("Rust infer_pip_metrics failed.") from exc


def infer_sl_tp_pips_auto(
    *,
    open_prices: Iterable,
    high_prices: Iterable,
    low_prices: Iterable,
    close_prices: Iterable,
    atr_values: Iterable | None,
    pip_size: float,
    atr_mult: float,
    min_rr: float,
    min_dist: float,
    settings: object | None = None,
) -> Tuple[float, float] | None:
    try:
        from .stop_target import infer_stop_target_pips_ohlcv
    except Exception:
        infer_stop_target_pips_ohlcv = None  # type: ignore[assignment]

    open_arr = _as_contig(open_prices, np.float64)
    close_arr = _as_contig(close_prices, np.float64)
    high_arr = _as_contig(high_prices, np.float64)
    low_arr = _as_contig(low_prices, np.float64)
    if open_arr.size != close_arr.size:
        open_arr = close_arr
    _ = atr_values

    if close_arr.size < 2:
        return None

    if infer_stop_target_pips_ohlcv is not None and pip_size > 0.0:
        settings_obj = settings
        if settings_obj is None or not hasattr(settings_obj, "risk"):
            risk = SimpleNamespace(
                atr_stop_multiplier=float(max(0.1, atr_mult)),
                min_risk_reward=float(max(1.5, min_rr)),
                meta_label_min_dist=float(max(0.0, min_dist)),
                stop_target_mode=str(os.environ.get("FOREX_BOT_STOP_TARGET_MODE", "blend") or "blend"),
            )
            settings_obj = SimpleNamespace(risk=risk)
        try:
            out = infer_stop_target_pips_ohlcv(
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                settings=settings_obj,
                pip_size=float(pip_size),
                signal=None,
            )
            if out is not None:
                sl_pips, tp_pips, _rr = out
                if np.isfinite(sl_pips) and np.isfinite(tp_pips) and sl_pips > 0.0 and tp_pips > 0.0:
                    return float(sl_pips), float(tp_pips)
        except Exception:
            # Fallback to ATR-only path below.
            pass

    return None


def fast_evaluate_strategy(
    *,
    close_prices: Iterable,
    high_prices: Iterable,
    low_prices: Iterable,
    signals: Iterable,
    month_indices: Iterable,
    day_indices: Iterable,
    sl_pips: float,
    tp_pips: float,
    max_hold_bars: int = 0,
    trailing_enabled: bool = False,
    trailing_atr_multiplier: float = 1.0,
    trailing_be_trigger_r: float = 1.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission_per_trade: float = 0.0,
    pip_value_per_lot: float = 10.0,
) -> np.ndarray:
    _require_rust()

    close_arr = _as_contig(close_prices, np.float64)
    high_arr = _as_contig(high_prices, np.float64)
    low_arr = _as_contig(low_prices, np.float64)
    sig_arr = _as_contig(signals, np.int8)
    month_arr = _as_contig(month_indices, np.int64)
    day_arr = _as_contig(day_indices, np.int64)

    backend = _select_backtest_backend()
    if backend is None:
        raise RuntimeError("Rust backtest backend selection failed for fast_evaluate_strategy().")

    args = (
        close_arr,
        high_arr,
        low_arr,
        sig_arr,
        month_arr,
        day_arr,
        float(sl_pips),
        float(tp_pips),
        int(max_hold_bars),
        bool(trailing_enabled),
        float(trailing_atr_multiplier),
        float(trailing_be_trigger_r),
        float(pip_value),
        float(spread_pips),
        float(commission_per_trade),
        float(pip_value_per_lot),
    )

    mode = _backtest_backend_mode()
    if backend == "bindings" and _forex_bindings is not None:
        try:
            out = _forex_bindings.fast_evaluate_strategy(*args)
            return np.asarray(out, dtype=np.float64)
        except Exception as exc:
            if mode == "bindings" or not _core_backtest_available():
                raise
            logger.warning("forex_bindings backtest failed; falling back to forex_core: %s", exc)

    if _forex_core is None:
        raise RuntimeError("forex_core backend is unavailable for fast_evaluate_strategy().")

    out = _forex_core.fast_evaluate_strategy(*args)
    return np.asarray(out, dtype=np.float64)


def batch_evaluate_strategies(
    close_prices: Iterable,
    high_prices: Iterable,
    low_prices: Iterable,
    signals: Iterable,
    month_indices: Iterable,
    day_indices: Iterable,
    sl_pips: Iterable,
    tp_pips: Iterable,
    max_hold_bars: int = 0,
    trailing_enabled: bool = False,
    trailing_atr_multiplier: float = 1.0,
    trailing_be_trigger_r: float = 1.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission_per_trade: float = 0.0,
    pip_value_per_lot: float = 10.0,
) -> np.ndarray:
    _require_rust()

    close_arr = _as_contig(close_prices, np.float64)
    high_arr = _as_contig(high_prices, np.float64)
    low_arr = _as_contig(low_prices, np.float64)
    sig_arr = _as_contig(signals, np.int8)
    month_arr = _as_contig(month_indices, np.int64)
    day_arr = _as_contig(day_indices, np.int64)
    sl_arr = _as_contig(sl_pips, np.float64)
    tp_arr = _as_contig(tp_pips, np.float64)

    backend = _select_backtest_backend()
    if backend is None:
        raise RuntimeError("Rust backtest backend selection failed for batch_evaluate_strategies().")

    args = (
        close_arr,
        high_arr,
        low_arr,
        sig_arr,
        month_arr,
        day_arr,
        sl_arr,
        tp_arr,
        int(max_hold_bars),
        bool(trailing_enabled),
        float(trailing_atr_multiplier),
        float(trailing_be_trigger_r),
        float(pip_value),
        float(spread_pips),
        float(commission_per_trade),
        float(pip_value_per_lot),
    )

    mode = _backtest_backend_mode()
    if backend == "bindings" and _forex_bindings is not None:
        try:
            out = _forex_bindings.batch_evaluate_strategies(*args)
            return np.asarray(out, dtype=np.float64)
        except Exception as exc:
            if mode == "bindings" or not _core_backtest_available():
                raise
            logger.warning("forex_bindings batch backtest failed; falling back to forex_core: %s", exc)

    if _forex_core is None:
        raise RuntimeError("forex_core backend is unavailable for batch_evaluate_strategies().")

    out = _forex_core.batch_evaluate_strategies(*args)
    return np.asarray(out, dtype=np.float64)

