from __future__ import annotations

import logging
import os
from typing import Iterable, Mapping, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
        return "python"
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
    if mode == "python":
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
    if str(os.environ.get("FOREX_BOT_ALLOW_PY_BACKTEST", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    raise ImportError(
        "No Rust backtest backend available (forex_bindings/forex_core). Build Rust extensions or set "
        "FOREX_BOT_ALLOW_PY_BACKTEST=1 to allow slow Python fallback."
    ) from (_FOREX_BINDINGS_ERROR or _FOREX_CORE_ERROR)


def _as_contig(arr: Iterable, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out, dtype=dtype)
    return out


def _norm_symbol(symbol: str) -> str:
    raw = "".join(ch for ch in str(symbol or "").upper() if ch.isalpha())
    if len(raw) >= 6:
        return raw[:6]
    return raw


def _split_symbol(symbol: str) -> tuple[str, str] | None:
    sym = _norm_symbol(symbol)
    if len(sym) == 6 and sym.isalpha():
        return sym[:3], sym[3:]
    return None


def _symbol_kind(symbol: str, parts: tuple[str, str] | None) -> str:
    if parts is not None:
        base, quote = parts
        if base in {"XAU", "XAG"}:
            return "metal"
        if base in {"BTC", "ETH", "LTC"}:
            return "crypto"
        if len(base) == 3 and len(quote) == 3:
            return "fx"
    sym = _norm_symbol(symbol)
    if "BTC" in sym or "ETH" in sym or "LTC" in sym:
        return "crypto"
    if sym.startswith("XAU") or sym.startswith("XAG"):
        return "metal"
    return "other"


def _pip_size(symbol: str, parts: tuple[str, str] | None) -> float:
    kind = _symbol_kind(symbol, parts)
    if kind == "metal":
        return 0.01
    if kind == "crypto":
        return 1.0
    if kind == "fx" and parts is not None and parts[1] == "JPY":
        return 0.01
    if kind == "fx":
        return 0.0001
    return 0.0001


def _contract_size(symbol: str, parts: tuple[str, str] | None) -> float:
    kind = _symbol_kind(symbol, parts)
    if kind == "metal" and parts is not None:
        base = parts[0]
        if base == "XAU":
            return 100.0
        if base == "XAG":
            return 5000.0
    if kind == "crypto":
        return 1.0
    if kind == "fx":
        return 100000.0
    return 1.0


def _norm_price_map(reference_prices: Mapping[str, float] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not reference_prices:
        return out
    for key, value in reference_prices.items():
        try:
            price = float(value)
        except Exception:
            continue
        if not np.isfinite(price) or price <= 0.0:
            continue
        pair = _norm_symbol(str(key))
        if len(pair) == 6:
            out[pair] = price
    return out


def _quote_to_account_rate(
    *,
    base: str,
    quote: str,
    account: str,
    price: float | None,
    reference_prices: Mapping[str, float] | None,
) -> float | None:
    acc = str(account or "USD").upper()
    if quote == acc:
        return 1.0

    if price is not None:
        px = float(price)
        if np.isfinite(px) and px > 0.0 and base == acc:
            return 1.0 / px
    else:
        px = np.nan

    refs = _norm_price_map(reference_prices)

    direct = refs.get(f"{quote}{acc}")
    if direct is not None and direct > 0.0:
        return float(direct)

    inverse = refs.get(f"{acc}{quote}")
    if inverse is not None and inverse > 0.0:
        return 1.0 / float(inverse)

    if np.isfinite(px) and px > 0.0:
        base_to_acc = refs.get(f"{base}{acc}")
        if base_to_acc is not None and base_to_acc > 0.0:
            return float(base_to_acc) / px
        acc_to_base = refs.get(f"{acc}{base}")
        if acc_to_base is not None and acc_to_base > 0.0:
            return 1.0 / (float(acc_to_base) * px)
        # Last-resort approximation: treat account currency as base.
        return 1.0 / px

    return None


def infer_pip_metrics(
    symbol: str,
    *,
    price: float | None = None,
    account_currency: str = "USD",
    reference_prices: Mapping[str, float] | None = None,
) -> Tuple[float, float]:
    parts = _split_symbol(symbol)
    pip_size = _pip_size(symbol, parts)
    lot_size = _contract_size(symbol, parts)
    pip_value_quote = pip_size * lot_size

    pip_value = pip_value_quote
    if parts is not None:
        base, quote = parts
        rate = _quote_to_account_rate(
            base=base,
            quote=quote,
            account=account_currency,
            price=price,
            reference_prices=reference_prices,
        )
        if rate is not None and np.isfinite(rate) and rate > 0.0:
            pip_value = pip_value_quote * rate

    if not np.isfinite(pip_value) or pip_value <= 0.0:
        pip_value = max(1e-6, float(pip_value_quote))

    return float(pip_size), float(pip_value)


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
        from .stop_target import infer_stop_target_pips
    except Exception:
        infer_stop_target_pips = None  # type: ignore[assignment]

    open_arr = _as_contig(open_prices, np.float64)
    close_arr = _as_contig(close_prices, np.float64)
    high_arr = _as_contig(high_prices, np.float64)
    low_arr = _as_contig(low_prices, np.float64)
    if open_arr.size != close_arr.size:
        open_arr = close_arr
    if atr_values is not None:
        atr_arr = _as_contig(atr_values, np.float64)
    else:
        atr_arr = None

    if close_arr.size < 2:
        return None

    if infer_stop_target_pips is not None and pip_size > 0.0:
        settings_obj = settings
        if settings_obj is None or not hasattr(settings_obj, "risk"):
            risk = SimpleNamespace(
                atr_stop_multiplier=float(max(0.1, atr_mult)),
                min_risk_reward=float(max(0.1, min_rr)),
                meta_label_min_dist=float(max(0.0, min_dist)),
                stop_target_mode=str(os.environ.get("FOREX_BOT_STOP_TARGET_MODE", "blend") or "blend"),
            )
            settings_obj = SimpleNamespace(risk=risk)
        try:
            df = {
                "open": open_arr,
                "high": high_arr,
                "low": low_arr,
                "close": close_arr,
            }
            if atr_arr is not None and atr_arr.size == close_arr.size:
                df["atr"] = atr_arr
            out = infer_stop_target_pips(
                pd.DataFrame(df),
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

    if atr_arr is None or atr_arr.size == 0:
        prev_close = np.roll(close_arr, 1)
        tr = np.maximum(high_arr - low_arr, np.maximum(np.abs(high_arr - prev_close), np.abs(low_arr - prev_close)))
        atr_period = 14
        try:
            if settings is not None and hasattr(settings, "risk"):
                atr_period = int(getattr(settings.risk, "atr_period", 14) or 14)
        except Exception:
            atr_period = 14
        window = max(2, atr_period)
        if tr.size < window:
            atr = float(np.nanmean(tr))
        else:
            atr = float(np.nanmean(tr[-window:]))
    else:
        atr = float(atr_arr[-1])

    if not np.isfinite(atr) or atr <= 0.0:
        return None

    atr_mult = float(max(0.1, atr_mult))
    min_rr = float(max(0.1, min_rr))
    min_dist = float(max(0.0, min_dist))
    sl_dist = max(atr * atr_mult, min_dist)
    if sl_dist <= 0.0:
        return None
    sl_pips = float(sl_dist / max(pip_size, 1e-9))
    tp_pips = float(sl_pips * min_rr)
    return sl_pips, tp_pips


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
        # Optional fallback path (very slow); return zeros if explicitly allowed.
        return np.zeros(11, dtype=np.float64)

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
        return np.zeros(11, dtype=np.float64)

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
        return np.zeros((0, 11), dtype=np.float64)

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
        return np.zeros((0, 11), dtype=np.float64)

    out = _forex_core.batch_evaluate_strategies(*args)
    return np.asarray(out, dtype=np.float64)
