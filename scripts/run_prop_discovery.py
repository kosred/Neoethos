#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forex_bot.core.config import Settings
from forex_bot.data.loader import DataLoader
from forex_bot.features.talib_mixer import ALL_INDICATORS
from forex_bot.strategy.evo_prop import run_evo_search
from forex_bot.strategy.fast_backtest import infer_pip_metrics


def _parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


def _discover_symbols(data_dir: str) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        return []
    out: list[str] = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("symbol="):
            out.append(path.name.split("=", 1)[1].upper())
    return sorted(set(out))


def _rows_for_days(timeframe: str, days: int) -> int:
    tf = str(timeframe or "").upper()
    bars_per_day = {
        "M1": 1440.0,
        "M3": 480.0,
        "M5": 288.0,
        "M15": 96.0,
        "M30": 48.0,
        "H1": 24.0,
        "H2": 12.0,
        "H4": 6.0,
        "D1": 1.0,
        "W1": 1.0 / 7.0,
        "MN1": 1.0 / 30.0,
    }.get(tf, 24.0)
    # Small buffer avoids edge-cases from missing bars/holidays.
    return int(max(1, math.ceil(days * bars_per_day * 1.1)))


class _ScriptFrame:
    def __init__(self, data: dict[str, np.ndarray], index: np.ndarray, attrs: dict[str, object] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return len(self.index) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def copy(self) -> _ScriptFrame:
        return _ScriptFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            attrs=dict(self.attrs),
        )


def _index_to_ns_int64(index: object) -> np.ndarray | None:
    if index is None:
        return None
    try:
        if hasattr(index, "asi8"):
            arr = np.asarray(index.asi8, dtype=np.int64).reshape(-1)
            return arr if arr.size > 0 else np.zeros(0, dtype=np.int64)
    except Exception:
        pass
    try:
        arr = np.asarray(index).reshape(-1)
    except Exception:
        return None
    if arr.size <= 0:
        return np.zeros(0, dtype=np.int64)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
    except Exception:
        pass
    out = np.zeros(arr.size, dtype=np.int64)
    for i, value in enumerate(arr.tolist()):
        try:
            ns = getattr(value, "value", None)
            if ns is not None:
                out[i] = int(ns)
            else:
                out[i] = int(np.datetime64(value, "ns").astype(np.int64))
        except Exception:
            try:
                out[i] = int(value)
            except Exception:
                out[i] = 0
    return out


def _slice_frame_mask(df: object, mask: np.ndarray) -> object:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if hasattr(df, "loc"):
        try:
            out = df.loc[m]  # type: ignore[index]
            return out.copy() if hasattr(out, "copy") else out
        except Exception:
            pass
    if hasattr(df, "take"):
        try:
            rows = np.flatnonzero(m).astype(np.int64, copy=False)
            out = df.take(rows)  # type: ignore[attr-defined]
            return out.copy() if hasattr(out, "copy") else out
        except Exception:
            pass
    idx = getattr(df, "index", None)
    cols = getattr(df, "columns", None)
    if idx is None or cols is None:
        return df
    idx_arr = np.asarray(idx).reshape(-1)
    idx_n = min(int(idx_arr.size), int(m.size))
    out_idx = idx_arr[:idx_n][m[:idx_n]]
    data: dict[str, np.ndarray] = {}
    for col in list(cols):
        try:
            values = np.asarray(df[col]).reshape(-1)  # type: ignore[index]
        except Exception:
            continue
        n = min(int(values.size), int(m.size))
        data[str(col)] = values[:n][m[:n]]
    attrs = getattr(df, "attrs", None)
    return _ScriptFrame(data, out_idx, attrs=(dict(attrs) if isinstance(attrs, dict) else None))


def _with_lookback(df: object, days: int) -> object:
    if days <= 0 or df is None or df.empty:
        return df
    idx_ns = _index_to_ns_int64(getattr(df, "index", None))
    n_rows = int(len(df))
    if idx_ns is None or idx_ns.size != n_rows:
        return df
    nat = np.iinfo(np.int64).min
    valid = idx_ns != nat
    if not np.any(valid):
        return df
    cutoff_ns = int(np.max(idx_ns[valid])) - int(days) * 86_400 * 1_000_000_000
    mask = valid & (idx_ns >= cutoff_ns)
    if mask.shape[0] != n_rows or np.all(mask):
        return df
    return _slice_frame_mask(df, mask)


def _history_span_days_months(df: object) -> tuple[float, float]:
    if df is None or df.empty:
        return 0.0, 0.0
    idx_ns = _index_to_ns_int64(getattr(df, "index", None))
    if idx_ns is None or idx_ns.size < 2:
        return 0.0, 0.0
    nat = np.iinfo(np.int64).min
    valid = idx_ns[idx_ns != nat]
    if valid.size < 2:
        span_days = 0.0
    else:
        span_days = float((int(np.max(valid)) - int(np.min(valid))) / 86_400_000_000_000.0)
    span_days = max(0.0, span_days)
    span_months = (span_days / 30.4375) if span_days > 0.0 else 0.0
    return float(span_days), float(span_months)


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _parse_number_list(raw: str | None, cast: type[float] | type[int], default: list[float] | list[int]) -> list[float] | list[int]:
    if raw is None:
        return list(default)
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return list(default)
    out: list[float] | list[int] = []
    for token in parts:
        try:
            out.append(cast(token))
        except Exception:
            continue
    if not out:
        return list(default)
    return out


def _hyperband_stages(
    *,
    base_population: int,
    base_generations: int,
    base_hours: float,
    pop_mults: list[float],
    gen_mults: list[float],
    hour_mults: list[float],
    promote_min: list[int],
) -> list[dict[str, float | int | str]]:
    stages_n = max(1, min(len(pop_mults), len(gen_mults), len(hour_mults)))
    out: list[dict[str, float | int | str]] = []
    for idx in range(stages_n):
        pm = float(max(0.05, pop_mults[idx]))
        gm = float(max(0.05, gen_mults[idx]))
        hm = float(max(0.01, hour_mults[idx]))
        promote = 0
        if idx < len(promote_min):
            promote = int(max(0, promote_min[idx]))
        out.append(
            {
                "suffix": f"hb{idx}",
                "population": int(max(2, round(base_population * pm))),
                "generations": int(max(1, round(base_generations * gm))),
                "max_hours": float(max(0.01, base_hours * hm)),
                "promote_min": int(promote),
            }
        )
    if out:
        out[-1]["promote_min"] = 0
    return out


def _apply_challenge_discovery_defaults(settings: Settings, args: argparse.Namespace) -> None:
    settings.risk.challenge_mode = True
    os.environ.setdefault("FOREX_BOT_CHALLENGE_MODE", "1")
    os.environ.setdefault("FOREX_BOT_CHALLENGE_TARGET_TRADING_DAYS", "44")
    os.environ.setdefault("FOREX_BOT_PROP_ANOMALY_GUARD", "1")
    os.environ.setdefault("FOREX_BOT_PROP_KEEP_PROFIT_METRIC", "net_profit")
    os.environ.setdefault("FOREX_BOT_PROP_KEEP_MIN_TRADES", str(int(max(1.0, float(args.challenge_min_trades)))))
    os.environ.setdefault(
        "FOREX_BOT_PROP_KEEP_MIN_TRADES_PER_MONTH",
        f"{float(args.challenge_min_trades_per_month):.6g}",
    )
    os.environ.setdefault(
        "FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT",
        f"{float(args.challenge_min_monthly_profit_pct):.6g}",
    )
    os.environ.setdefault(
        "FOREX_BOT_PROP_KEEP_MIN_SHARPE",
        f"{float(args.challenge_min_sharpe):.6g}",
    )
    os.environ.setdefault(
        "FOREX_BOT_PROP_KEEP_MIN_WIN_RATE",
        f"{float(args.challenge_min_win_rate):.6g}",
    )
    os.environ.setdefault(
        "FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR",
        f"{float(args.challenge_min_profit_factor):.6g}",
    )
    os.environ.setdefault(
        "FOREX_BOT_PROP_INITIAL_BALANCE",
        f"{float(getattr(settings.risk, 'initial_balance', 100000.0) or 100000.0):.6g}",
    )
    os.environ.setdefault("FOREX_BOT_PROP_ANOMALY_MAX_PROFIT_PER_TRADE", "800.0")
    os.environ.setdefault("FOREX_BOT_PROP_ANOMALY_LOW_DD_MIN_TRADES", "120.0")
    os.environ.setdefault("FOREX_BOT_PROP_ANOMALY_LOW_DD_MAX_DD", "0.0008")
    os.environ.setdefault("FOREX_BOT_PROP_SMC_FORCE_ENABLED", "1")
    os.environ.setdefault("FOREX_BOT_PROP_SMC_FORCE_RATIO", "0.75")
    os.environ.setdefault("FOREX_BOT_PROP_HOLDOUT_FRACTION", "0.20")
    os.environ.setdefault("FOREX_BOT_PROP_HOLDOUT_YEARS", "3.0")
    os.environ.setdefault("FOREX_BOT_PROP_HOLDOUT_REQUIRED", "1")
    os.environ.setdefault("FOREX_BOT_PROP_FORWARD_TEST_REQUIRED", "1")
    os.environ.setdefault("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.70")
    os.environ.setdefault("FOREX_BOT_MIN_TRUTH_PROBABILITY", "0.70")
    holdout_min_sharpe = max(1.20, float(args.challenge_min_sharpe))
    holdout_min_wr = max(0.52, float(args.challenge_min_win_rate))
    holdout_min_pf = max(1.30, float(args.challenge_min_profit_factor))
    holdout_min_trades = int(max(15.0, float(args.challenge_min_trades) * 0.7))
    os.environ.setdefault("FOREX_BOT_PROP_HOLDOUT_MIN_SHARPE", f"{holdout_min_sharpe:.6g}")
    os.environ.setdefault("FOREX_BOT_PROP_HOLDOUT_MIN_WIN_RATE", f"{holdout_min_wr:.6g}")
    os.environ.setdefault("FOREX_BOT_PROP_HOLDOUT_MIN_PROFIT_FACTOR", f"{holdout_min_pf:.6g}")
    os.environ.setdefault("FOREX_BOT_PROP_HOLDOUT_MIN_TRADES", str(holdout_min_trades))
    os.environ.setdefault("FOREX_BOT_DISCOVERY_MIN_SHARPE", f"{holdout_min_sharpe:.6g}")
    os.environ.setdefault("FOREX_BOT_DISCOVERY_MIN_WIN_RATE", f"{holdout_min_wr:.6g}")
    os.environ.setdefault("FOREX_BOT_DISCOVERY_MIN_PROFIT_FACTOR", f"{holdout_min_pf:.6g}")


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


async def _cached_last_close(
    loader: DataLoader,
    *,
    symbol: str,
    timeframe: str,
    cache: dict[tuple[str, str], float | None],
) -> float | None:
    key = (_norm_symbol(symbol), str(timeframe or "").upper())
    if key in cache:
        return cache[key]
    value: float | None = None
    try:
        # Fast-path for bootstrapping: avoid full resample fan-out unless needed.
        frames = loader._load_frames(symbol, allow_resample=False)
        if not isinstance(frames, dict) or not frames:
            frames = await loader.get_training_data(symbol)
        if isinstance(frames, dict):
            tf_u = str(timeframe or "").upper()
            df = frames.get(tf_u) or frames.get(str(timeframe or "")) or frames.get(tf_u.lower())
            if df is not None and not df.empty and "close" in df.columns:
                px = _safe_float(df["close"].iloc[-1], 0.0)
                if math.isfinite(px) and px > 0.0:
                    value = float(px)
    except Exception:
        value = None
    cache[key] = value
    return value


async def _resolve_reference_prices(
    loader: DataLoader,
    *,
    symbol: str,
    timeframe: str,
    symbol_universe: set[str],
    close_cache: dict[tuple[str, str], float | None],
) -> dict[str, float]:
    parts = _split_symbol(symbol)
    if parts is None:
        return {}
    base, quote = parts
    candidates = {
        f"{quote}USD",
        f"USD{quote}",
        f"{base}USD",
        f"USD{base}",
    }
    refs: dict[str, float] = {}
    norm_self = _norm_symbol(symbol)
    for cand in candidates:
        if cand == norm_self:
            continue
        if symbol_universe and cand not in symbol_universe:
            continue
        px = await _cached_last_close(
            loader,
            symbol=cand,
            timeframe=timeframe,
            cache=close_cache,
        )
        if px is not None and math.isfinite(px) and px > 0.0:
            refs[cand] = float(px)
    return refs


async def _history_span_for_symbol_tf(
    loader: DataLoader,
    *,
    symbol: str,
    timeframe: str,
    cache: dict[tuple[str, str], tuple[float, float]],
) -> tuple[float, float]:
    key = (_norm_symbol(symbol), str(timeframe or "").upper())
    if key in cache:
        return cache[key]
    days = 0.0
    months = 0.0
    try:
        # Fast-path for bootstrapping: avoid full resample fan-out unless needed.
        frames = loader._load_frames(symbol, allow_resample=False)
        if not isinstance(frames, dict) or not frames:
            frames = await loader.get_training_data(symbol)
        if isinstance(frames, dict):
            tf_u = str(timeframe or "").upper()
            df = frames.get(tf_u) or frames.get(str(timeframe or "")) or frames.get(tf_u.lower())
            if df is not None and not df.empty:
                days, months = _history_span_days_months(df)
    except Exception:
        days = 0.0
        months = 0.0
    cache[key] = (float(days), float(months))
    return cache[key]


def _checkpoint_symbol_tf(path: Path) -> tuple[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    symbol = str(payload.get("symbol", "") or "").upper().strip()
    timeframe = str(payload.get("timeframe", payload.get("tf", "")) or "").upper().strip()
    return symbol, timeframe


def _apply_env(overrides: dict[str, str]) -> dict[str, str | None]:
    prev: dict[str, str | None] = {}
    for key, value in overrides.items():
        prev[key] = os.environ.get(key)
        os.environ[key] = str(value)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for key, value in prev.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _safe_tag(text: str) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "default"


def _default_state_db_path(
    checkpoint: Path,
    *,
    profit_key: str,
    threshold: float,
    min_trades: float,
    min_trades_per_month: float = 0.0,
    min_monthly_profit_pct: float = 0.0,
    min_sharpe: float = 0.0,
    min_win_rate: float = 0.0,
    min_profit_factor: float = 0.0,
    max_dd: float | None,
) -> Path:
    dd_tag = "inf" if max_dd is None else f"{max_dd:.6g}"
    stem = _safe_tag(checkpoint.stem)
    key_tag = _safe_tag(profit_key)
    thr_tag = _safe_tag(f"{threshold:.6g}")
    tr_tag = _safe_tag(f"{min_trades:.6g}")
    tpm_tag = _safe_tag(f"{float(min_trades_per_month):.6g}")
    mpp_tag = _safe_tag(f"{float(min_monthly_profit_pct):.6g}")
    sh_tag = _safe_tag(f"{float(min_sharpe):.6g}")
    wr_tag = _safe_tag(f"{float(min_win_rate):.6g}")
    pf_tag = _safe_tag(f"{float(min_profit_factor):.6g}")
    dd_safe = _safe_tag(dd_tag)
    holdout_frac = _safe_tag(str(os.environ.get("FOREX_BOT_PROP_HOLDOUT_FRACTION", "0") or "0"))
    holdout_years = _safe_tag(str(os.environ.get("FOREX_BOT_PROP_HOLDOUT_YEARS", "0") or "0"))
    holdout_from = _safe_tag(str(os.environ.get("FOREX_BOT_PROP_HOLDOUT_FROM", "") or ""))
    holdout_req = _safe_tag(str(os.environ.get("FOREX_BOT_PROP_HOLDOUT_REQUIRED", "0") or "0"))
    forward_req = _safe_tag(str(os.environ.get("FOREX_BOT_PROP_FORWARD_TEST_REQUIRED", "0") or "0"))
    min_truth = _safe_tag(
        str(
            os.environ.get(
                "FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY",
                os.environ.get("FOREX_BOT_MIN_TRUTH_PROBABILITY", "0"),
            )
            or "0"
        )
    )
    return (
        Path("cache")
        / (
            f"prop_discovery_state_{stem}_{key_tag}_{thr_tag}_{tr_tag}_{tpm_tag}_{mpp_tag}_{sh_tag}_{wr_tag}_{pf_tag}_"
            f"{dd_safe}_{holdout_frac}_{holdout_years}_{holdout_from}_{holdout_req}_{forward_req}_{min_truth}.sqlite"
        )
    )


def _seen_file_path(seen_dir: str, symbol: str, timeframe: str) -> Path:
    base = Path(str(seen_dir or "cache/prop_seen_hashes")).expanduser()
    return base / f"{_safe_tag(symbol)}_{_safe_tag(timeframe)}.bin"


def _latest_round_index(base_ckpt: Path, symbol: str, timeframe: str) -> int:
    parent = base_ckpt.parent
    if not parent.exists():
        return 0
    prefix = f"{base_ckpt.stem}_{symbol}_{timeframe}_r"
    suffix = str(base_ckpt.suffix or "")
    rgx = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    latest = 0
    for path in parent.glob(f"{prefix}*{suffix}"):
        m = rgx.match(path.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            idx = 0
        if idx > latest:
            latest = idx
    return int(latest)


def _strategy_key(raw: dict) -> str:
    indicators = tuple(str(x).strip().upper() for x in (raw.get("indicators") or []))
    weights = raw.get("weights") if isinstance(raw.get("weights"), dict) else {}
    items = tuple(sorted((str(k).upper(), round(_safe_float(v), 6)) for k, v in weights.items()))
    lt = round(_safe_float(raw.get("long_threshold", 0.66)), 6)
    st = round(_safe_float(raw.get("short_threshold", -0.66)), 6)
    tp = round(_safe_float(raw.get("tp_pips", 40.0)), 3)
    sl = round(_safe_float(raw.get("sl_pips", 20.0)), 3)
    return f"sig:{indicators}|w:{items}|lt:{lt}|st:{st}|tp:{tp}|sl:{sl}"


class DiscoveryStateStore:
    def __init__(
        self,
        path: Path,
        *,
        profit_key: str,
        threshold: float,
        min_trades: float,
        max_dd: float | None,
    ) -> None:
        self.path = path
        self.profit_key = str(profit_key or "net_profit")
        self.threshold = float(threshold)
        self.min_trades = float(min_trades)
        self.max_dd = float(max_dd) if max_dd is not None else None
        self.min_trades_per_month = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_TRADES_PER_MONTH", 0.0), 0.0)
        self.min_monthly_profit_pct = _safe_float(
            os.environ.get("FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT", 0.0),
            0.0,
        )
        self.min_sharpe = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_SHARPE", 0.0), 0.0)
        self.min_win_rate = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_WIN_RATE", 0.0), 0.0)
        self.min_profit_factor = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR", 0.0), 0.0)
        self.min_truth_probability = _safe_float(
            os.environ.get(
                "FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY",
                os.environ.get("FOREX_BOT_MIN_TRUTH_PROBABILITY", 0.0),
            ),
            0.0,
        )
        if self.min_truth_probability > 1.0:
            self.min_truth_probability *= 0.01
        self.min_truth_probability = max(0.0, min(1.0, self.min_truth_probability))
        self.forward_test_required = str(os.environ.get("FOREX_BOT_PROP_FORWARD_TEST_REQUIRED", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.initial_balance = max(
            1e-9,
            _safe_float(os.environ.get("FOREX_BOT_PROP_INITIAL_BALANCE", 100000.0), 100000.0),
        )
        self.anomaly_guard = str(os.environ.get("FOREX_BOT_PROP_ANOMALY_GUARD", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.anomaly_min_trades = _safe_float(os.environ.get("FOREX_BOT_PROP_ANOMALY_MIN_TRADES", 120.0), 120.0)
        self.anomaly_max_dd = _safe_float(os.environ.get("FOREX_BOT_PROP_ANOMALY_MAX_DD", 0.0025), 0.0025)
        self.anomaly_min_win_rate = _safe_float(os.environ.get("FOREX_BOT_PROP_ANOMALY_MIN_WIN_RATE", 0.92), 0.92)
        self.anomaly_min_pf = _safe_float(os.environ.get("FOREX_BOT_PROP_ANOMALY_MIN_PF", 12.0), 12.0)
        self.anomaly_min_profit = _safe_float(os.environ.get("FOREX_BOT_PROP_ANOMALY_MIN_PROFIT", 200000.0), 200000.0)
        self.anomaly_max_profit_per_trade = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_MAX_PROFIT_PER_TRADE", 2000.0),
            2000.0,
        )
        self.anomaly_ultra_min_trades = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_ULTRA_MIN_TRADES", 50.0),
            50.0,
        )
        self.anomaly_ultra_max_dd = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_ULTRA_MAX_DD", 0.001),
            0.001,
        )
        self.anomaly_ultra_min_profit = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_ULTRA_MIN_PROFIT", 150000.0),
            150000.0,
        )
        self.anomaly_ultra_min_ppt = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_ULTRA_MIN_PPT", 1000.0),
            1000.0,
        )
        self.anomaly_low_dd_min_trades = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_LOW_DD_MIN_TRADES", 80.0),
            80.0,
        )
        self.anomaly_low_dd_max_dd = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_LOW_DD_MAX_DD", 0.001),
            0.001,
        )
        self.anomaly_low_dd_min_profit = _safe_float(
            os.environ.get("FOREX_BOT_PROP_ANOMALY_LOW_DD_MIN_PROFIT", 50000.0),
            50000.0,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self._create_schema()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                signature TEXT PRIMARY KEY,
                profitable INTEGER NOT NULL DEFAULT 0,
                metric REAL NOT NULL DEFAULT 0.0,
                trades REAL NOT NULL DEFAULT 0.0,
                max_dd REAL NOT NULL DEFAULT 0.0,
                symbol TEXT NOT NULL DEFAULT '',
                timeframe TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_strategies_profitable ON strategies(profitable)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                local_total INTEGER NOT NULL,
                local_profitable INTEGER NOT NULL,
                processed_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _checkpoint_stamp(self, path: Path) -> tuple[int, int] | None:
        if not path.exists():
            return None
        try:
            st = path.stat()
            return int(st.st_mtime_ns), int(st.st_size)
        except Exception:
            return None

    def cached_checkpoint_counts(self, path: Path) -> tuple[int, int] | None:
        stamp = self._checkpoint_stamp(path)
        if stamp is None:
            return None
        row = self.conn.execute(
            "SELECT local_profitable, local_total, mtime_ns, size_bytes FROM checkpoints WHERE path=?",
            (str(path),),
        ).fetchone()
        if row is None:
            return None
        local_profitable, local_total, mtime_ns, size_bytes = row
        if int(mtime_ns) != int(stamp[0]) or int(size_bytes) != int(stamp[1]):
            return None
        return int(local_profitable), int(local_total)

    def _is_profitable(self, raw: dict, *, history_months: float | None = None) -> tuple[bool, float, float, float]:
        pnl = _safe_float(raw.get(self.profit_key, 0.0), 0.0)
        trades = _safe_float(raw.get("trades", raw.get("trades_count", raw.get("trade_count", 0.0))), 0.0)
        dd = _safe_float(
            raw.get("max_dd_pct", raw.get("max_drawdown", raw.get("max_dd", raw.get("drawdown", 0.0)))),
            0.0,
        )
        truth_probability = _safe_float(raw.get("truth_probability", raw.get("truth_prob", 0.0)), 0.0)
        if truth_probability > 1.0:
            truth_probability *= 0.01
        forward_passed = bool(raw.get("forward_test_passed", False))
        sharpe = _safe_float(raw.get("sharpe_ratio", raw.get("sharpe", 0.0)), 0.0)
        win_rate = _safe_float(raw.get("win_rate", raw.get("win_pct", 0.0)), 0.0)
        profit_factor = _safe_float(raw.get("profit_factor", raw.get("pf", 0.0)), 0.0)
        if self._is_anomalous(raw, pnl=pnl, trades=trades, dd=dd):
            return False, pnl, trades, dd
        if self.forward_test_required and not forward_passed:
            return False, pnl, trades, dd
        if self.min_truth_probability > 0.0 and truth_probability < self.min_truth_probability:
            return False, pnl, trades, dd
        if self.max_dd is not None and dd > float(self.max_dd):
            return False, pnl, trades, dd
        if self.min_sharpe > 0.0 and sharpe < self.min_sharpe:
            return False, pnl, trades, dd
        if self.min_win_rate > 0.0 and win_rate < self.min_win_rate:
            return False, pnl, trades, dd
        if self.min_profit_factor > 0.0 and profit_factor < self.min_profit_factor:
            return False, pnl, trades, dd
        hm = float(history_months) if history_months is not None else 0.0
        if (self.min_trades_per_month > 0.0 or self.min_monthly_profit_pct > 0.0) and hm <= 0.0:
            # Activity/monthly-return filters need a known history span; otherwise reject as unverifiable.
            return False, pnl, trades, dd
        if self.min_trades_per_month > 0.0 and hm > 0.0:
            if (trades / hm) < self.min_trades_per_month:
                return False, pnl, trades, dd
        if self.min_monthly_profit_pct > 0.0 and hm > 0.0:
            monthly_profit_pct = pnl / (self.initial_balance * hm)
            if monthly_profit_pct < self.min_monthly_profit_pct:
                return False, pnl, trades, dd
        return bool(pnl > self.threshold and trades >= self.min_trades), pnl, trades, dd

    def _is_anomalous(self, raw: dict, *, pnl: float, trades: float, dd: float) -> bool:
        if not self.anomaly_guard:
            return False
        win_rate = _safe_float(raw.get("win_rate", 0.0), 0.0)
        profit_factor = _safe_float(raw.get("profit_factor", 0.0), 0.0)
        ppt = (pnl / trades) if trades > 0 else 0.0
        suspicious_combo = (
            trades >= self.anomaly_min_trades
            and dd <= self.anomaly_max_dd
            and win_rate >= self.anomaly_min_win_rate
            and profit_factor >= self.anomaly_min_pf
            and pnl >= self.anomaly_min_profit
        )
        suspicious_ppt = (
            trades >= max(40.0, self.anomaly_min_trades * 0.5)
            and dd <= max(0.01, self.anomaly_max_dd * 2.0)
            and ppt >= self.anomaly_max_profit_per_trade
        )
        suspicious_ultra = (
            trades >= self.anomaly_ultra_min_trades
            and dd <= self.anomaly_ultra_max_dd
            and pnl >= self.anomaly_ultra_min_profit
            and ppt >= self.anomaly_ultra_min_ppt
        )
        suspicious_low_dd = (
            trades >= self.anomaly_low_dd_min_trades
            and dd <= self.anomaly_low_dd_max_dd
            and pnl >= self.anomaly_low_dd_min_profit
        )
        return bool(suspicious_combo or suspicious_ppt or suspicious_ultra or suspicious_low_dd)

    def ingest_checkpoint(
        self,
        path: Path,
        *,
        history_months: float | None = None,
        history_days: float | None = None,
    ) -> tuple[int, int, bool]:
        cached = self.cached_checkpoint_counts(path)
        if cached is not None:
            return int(cached[0]), int(cached[1]), True
        if not path.exists():
            return 0, 0, False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0, 0, False
        genes = payload.get("best_genes")
        if not isinstance(genes, list):
            return 0, 0, False
        payload_months = _safe_float(payload.get("history_months", 0.0), 0.0)
        payload_days = _safe_float(payload.get("history_days", 0.0), 0.0)
        hm = float(history_months) if history_months is not None else 0.0
        hd = float(history_days) if history_days is not None else 0.0
        if hm <= 0.0 and payload_months > 0.0:
            hm = float(payload_months)
        if hd <= 0.0 and payload_days > 0.0:
            hd = float(payload_days)
        if hm <= 0.0 and hd > 0.0:
            hm = float(hd / 30.4375)

        symbol = str(payload.get("symbol", "") or "").upper().strip()
        timeframe = str(payload.get("timeframe", payload.get("tf", "")) or "").upper().strip()
        processed_at = datetime.now(timezone.utc).isoformat()
        local_total = int(len(genes))
        local_profitable = 0
        local_seen: set[str] = set()

        stamp = self._checkpoint_stamp(path)
        if stamp is None:
            return 0, 0, False

        with self.conn:
            for raw in genes:
                if not isinstance(raw, dict):
                    continue
                sig = _strategy_key(raw)
                is_profitable, pnl, trades, dd = self._is_profitable(raw, history_months=hm if hm > 0.0 else None)
                if is_profitable and sig not in local_seen:
                    local_seen.add(sig)
                    local_profitable += 1
                self.conn.execute(
                    """
                    INSERT INTO strategies (
                        signature, profitable, metric, trades, max_dd, symbol, timeframe, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(signature) DO UPDATE SET
                        profitable = CASE
                            WHEN excluded.profitable > strategies.profitable THEN excluded.profitable
                            ELSE strategies.profitable
                        END,
                        metric = CASE
                            WHEN excluded.metric > strategies.metric THEN excluded.metric
                            ELSE strategies.metric
                        END,
                        trades = CASE
                            WHEN excluded.trades > strategies.trades THEN excluded.trades
                            ELSE strategies.trades
                        END,
                        max_dd = CASE
                            WHEN strategies.max_dd <= 0 THEN excluded.max_dd
                            WHEN excluded.max_dd <= 0 THEN strategies.max_dd
                            WHEN excluded.max_dd < strategies.max_dd THEN excluded.max_dd
                            ELSE strategies.max_dd
                        END,
                        symbol = CASE
                            WHEN strategies.symbol = '' THEN excluded.symbol
                            ELSE strategies.symbol
                        END,
                        timeframe = CASE
                            WHEN strategies.timeframe = '' THEN excluded.timeframe
                            ELSE strategies.timeframe
                        END,
                        updated_at = excluded.updated_at
                    """,
                    (
                        sig,
                        1 if is_profitable else 0,
                        float(pnl),
                        float(trades),
                        float(dd),
                        symbol,
                        timeframe,
                        processed_at,
                    ),
                )
            self.conn.execute(
                """
                INSERT INTO checkpoints (
                    path, mtime_ns, size_bytes, local_total, local_profitable, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    mtime_ns = excluded.mtime_ns,
                    size_bytes = excluded.size_bytes,
                    local_total = excluded.local_total,
                    local_profitable = excluded.local_profitable,
                    processed_at = excluded.processed_at
                """,
                (
                    str(path),
                    int(stamp[0]),
                    int(stamp[1]),
                    int(local_total),
                    int(local_profitable),
                    processed_at,
                ),
            )
        return int(local_profitable), int(local_total), False

    def stats(self) -> tuple[int, int]:
        profitable = self.conn.execute("SELECT COUNT(*) FROM strategies WHERE profitable=1").fetchone()
        total_rows = self.conn.execute("SELECT COALESCE(SUM(local_total), 0) FROM checkpoints").fetchone()
        return int((profitable or [0])[0] or 0), int((total_rows or [0])[0] or 0)

    def stats_for_symbol(self, symbol: str) -> tuple[int, int]:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return self.stats()
        profitable = self.conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE profitable=1 AND symbol=?",
            (sym,),
        ).fetchone()
        total_rows = self.conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE symbol=?",
            (sym,),
        ).fetchone()
        return int((profitable or [0])[0] or 0), int((total_rows or [0])[0] or 0)


def _scan_profitable(
    checkpoints: list[Path],
    *,
    profit_key: str,
    threshold: float,
    min_trades: float,
    max_dd: float | None = None,
) -> tuple[int, int]:
    seen: set[str] = set()
    profitable = 0
    total = 0
    min_truth_probability = _safe_float(
        os.environ.get(
            "FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY",
            os.environ.get("FOREX_BOT_MIN_TRUTH_PROBABILITY", 0.0),
        ),
        0.0,
    )
    if min_truth_probability > 1.0:
        min_truth_probability *= 0.01
    min_truth_probability = max(0.0, min(1.0, min_truth_probability))
    forward_required = str(os.environ.get("FOREX_BOT_PROP_FORWARD_TEST_REQUIRED", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    for path in checkpoints:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        genes = payload.get("best_genes")
        if not isinstance(genes, list):
            continue
        total += len(genes)
        for raw in genes:
            if not isinstance(raw, dict):
                continue
            if forward_required and not bool(raw.get("forward_test_passed", False)):
                continue
            if min_truth_probability > 0.0:
                truth_probability = _safe_float(raw.get("truth_probability", raw.get("truth_prob", 0.0)), 0.0)
                if truth_probability > 1.0:
                    truth_probability *= 0.01
                if truth_probability < min_truth_probability:
                    continue
            pnl = _safe_float(raw.get(profit_key, 0.0), 0.0)
            trades = _safe_float(raw.get("trades", raw.get("trades_count", raw.get("trade_count", 0.0))), 0.0)
            if max_dd is not None:
                dd = _safe_float(
                    raw.get("max_dd_pct", raw.get("max_drawdown", raw.get("max_dd", raw.get("drawdown", 0.0)))),
                    0.0,
                )
                if dd > max_dd:
                    continue
            if pnl <= threshold or trades < min_trades:
                continue
            key = _strategy_key(raw)
            if key in seen:
                continue
            seen.add(key)
            profitable += 1
    return profitable, total


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    loader = DataLoader(settings)

    symbols = _parse_csv(args.symbols)
    if not symbols:
        symbols = _discover_symbols(settings.system.data_dir)
    if not symbols:
        print("No symbols found. Use --symbols or check data directory.", file=sys.stderr)
        return 2

    timeframes = _parse_csv(args.timeframes)
    if not timeframes:
        base_tf = str(getattr(settings.system, "base_timeframe", "M1") or "M1").upper()
        timeframes = [base_tf]
    # Restrict loader timeframe universe to the explicit CLI set to avoid
    # loading/resampling unrelated frames from broad config defaults.
    settings.system.base_timeframe = str(timeframes[0]).upper()
    settings.system.multi_resolution_enabled = True
    settings.system.multi_resolution_timeframes = [str(tf).upper() for tf in timeframes]
    settings.system.required_timeframes = []
    settings.system.higher_timeframes = []

    base_population = int(args.population)
    base_generations = int(args.generations)
    if base_population > 0:
        settings.models.prop_search_population = base_population
    if base_generations > 0:
        settings.models.prop_search_generations = base_generations
    if float(args.max_hours) > 0:
        settings.models.prop_search_max_hours = float(args.max_hours)
    if str(args.device).strip():
        settings.models.prop_search_device = str(args.device).strip().lower()
    if str(args.checkpoint).strip():
        settings.models.prop_search_checkpoint = str(args.checkpoint).strip()
    if int(args.max_indicators) >= 0:
        settings.models.prop_search_max_indicators = int(args.max_indicators)
    os.environ.setdefault(
        "FOREX_BOT_PROP_INITIAL_BALANCE",
        f"{float(getattr(settings.risk, 'initial_balance', 100000.0) or 100000.0):.6g}",
    )
    if float(args.holdout_years) > 0.0:
        os.environ["FOREX_BOT_PROP_HOLDOUT_YEARS"] = f"{float(args.holdout_years):.6g}"
    holdout_from_raw = str(args.holdout_from or "").strip()
    if holdout_from_raw:
        os.environ["FOREX_BOT_PROP_HOLDOUT_FROM"] = holdout_from_raw
    if float(args.min_truth_probability) > 0.0:
        tp = float(args.min_truth_probability)
        if tp > 1.0:
            tp *= 0.01
        tp = max(0.0, min(1.0, tp))
        os.environ["FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY"] = f"{tp:.6g}"
        os.environ["FOREX_BOT_MIN_TRUTH_PROBABILITY"] = f"{tp:.6g}"
    if int(args.forward_test_required) in (0, 1):
        os.environ["FOREX_BOT_PROP_FORWARD_TEST_REQUIRED"] = str(int(args.forward_test_required))
    os.environ["FOREX_BOT_PROP_JOURNAL_TOP_K"] = str(max(0, int(args.journal_top_k)))
    os.environ["FOREX_BOT_PROP_SWAP_LONG_PER_DAY"] = f"{float(args.swap_long_per_day):.6g}"
    os.environ["FOREX_BOT_PROP_SWAP_SHORT_PER_DAY"] = f"{float(args.swap_short_per_day):.6g}"
    challenge_mode = bool(int(args.challenge_mode) > 0)
    if challenge_mode:
        _apply_challenge_discovery_defaults(settings, args)
        if int(args.adaptive_retries) <= 0:
            args.adaptive_retries = 2
        if float(args.max_hours) <= 0:
            settings.models.prop_search_max_hours = max(0.05, float(settings.models.prop_search_max_hours))
        print(
            f"[CHLG] enabled: min_trades={float(args.challenge_min_trades):g} "
            f"min_tpm={float(args.challenge_min_trades_per_month):g} "
            f"min_monthly_pct={float(args.challenge_min_monthly_profit_pct):g} "
            f"min_sharpe={float(args.challenge_min_sharpe):g} "
            f"min_win_rate={float(args.challenge_min_win_rate):g} "
            f"min_pf={float(args.challenge_min_profit_factor):g} "
            f"holdout={os.environ.get('FOREX_BOT_PROP_HOLDOUT_FRACTION','0')} "
            f"holdout_years={os.environ.get('FOREX_BOT_PROP_HOLDOUT_YEARS','0')} "
            f"holdout_from={os.environ.get('FOREX_BOT_PROP_HOLDOUT_FROM','')} "
            f"min_truth={os.environ.get('FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY', os.environ.get('FOREX_BOT_MIN_TRUTH_PROBABILITY','0'))} "
            f"journal_top_k={os.environ.get('FOREX_BOT_PROP_JOURNAL_TOP_K','0')} "
            f"hb_pop={args.hb_stage_pop_mults} hb_gen={args.hb_stage_gen_mults} hb_hours={args.hb_stage_hour_mults}"
        )
    if int(args.adaptive_retries) > 0:
        # Exploration defaults: keep memory bounded while increasing strategy diversity.
        os.environ.setdefault("FOREX_BOT_PROP_ARCHIVE_MODE", "active")
        os.environ.setdefault("FOREX_BOT_PROP_RANDOM_IMMIGRANTS", "0.35")
        os.environ.setdefault("FOREX_BOT_PROP_STAGNATION_GENS", "2")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_GATE", "0.75")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_FORCE_ENABLED", "1")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_FORCE_RATIO", "0.70")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_MIN_FLAGS", "1")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_ENABLE_P", "0.55")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_GATE_START", "0.75")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_GATE_END", "0.35")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_GATE_CURVE", "1.0")
        os.environ.setdefault("FOREX_BOT_PROP_SMC_GATE_STAGNATION_STEP", "0.03")
    max_indicator_universe = len(ALL_INDICATORS)

    lookback_days = int(args.lookback_days)
    default_rows = int(args.max_rows)
    target_profitable = int(args.target_profitable)
    target_profitable_per_symbol = int(args.target_profitable_per_symbol)
    repeat_until_target = bool(
        int(args.repeat_until_target) > 0 and (target_profitable > 0 or target_profitable_per_symbol > 0)
    )

    print(
        "Discovery-only run: symbols=%s timeframes=%s rounds=%s pop=%s gen=%s max_hours=%.3f lookback_days=%s max_rows=%s repeat_until_target=%s target_total=%s target_per_symbol=%s"
        % (
            symbols,
            timeframes,
            int(args.rounds),
            int(settings.models.prop_search_population),
            int(settings.models.prop_search_generations),
            float(settings.models.prop_search_max_hours),
            lookback_days,
            default_rows,
            int(repeat_until_target),
            int(target_profitable),
            int(target_profitable_per_symbol),
        )
    )

    symbol_universe = {_norm_symbol(s) for s in symbols if _norm_symbol(s)}
    close_cache: dict[tuple[str, str], float | None] = {}
    history_span_cache: dict[tuple[str, str], tuple[float, float]] = {}

    ran = 0
    resumed = 0
    skipped = 0
    written: list[Path] = []
    min_profit = float(args.profit_threshold)
    min_profit_trades = float(args.profit_min_trades)
    if challenge_mode:
        min_profit_trades = max(min_profit_trades, float(args.challenge_min_trades))
    min_trades_per_month_filter = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_TRADES_PER_MONTH", 0.0), 0.0)
    min_monthly_profit_pct_filter = _safe_float(
        os.environ.get("FOREX_BOT_PROP_KEEP_MIN_MONTHLY_PROFIT_PCT", 0.0),
        0.0,
    )
    min_sharpe_filter = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_SHARPE", 0.0), 0.0)
    min_win_rate_filter = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_WIN_RATE", 0.0), 0.0)
    min_profit_factor_filter = _safe_float(os.environ.get("FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR", 0.0), 0.0)
    profit_key = str(args.profit_key or "net_profit").strip()
    max_dd_filter = None
    if str(args.max_dd).strip() != "":
        max_dd_filter = float(args.max_dd)

    base_ckpt = Path(str(settings.models.prop_search_checkpoint or "models/strategy_evo_checkpoint.json"))
    state_db_raw = str(args.state_db or "").strip()
    if state_db_raw:
        state_db_path = Path(state_db_raw)
    else:
        state_db_path = _default_state_db_path(
            base_ckpt,
            profit_key=profit_key,
            threshold=min_profit,
            min_trades=min_profit_trades,
            min_trades_per_month=min_trades_per_month_filter,
            min_monthly_profit_pct=min_monthly_profit_pct_filter,
            min_sharpe=min_sharpe_filter,
            min_win_rate=min_win_rate_filter,
            min_profit_factor=min_profit_factor_filter,
            max_dd=max_dd_filter,
        )
    if int(args.state_reset) > 0 and state_db_path.exists():
        try:
            state_db_path.unlink()
        except Exception:
            pass
    state = DiscoveryStateStore(
        state_db_path,
        profit_key=profit_key,
        threshold=min_profit,
        min_trades=min_profit_trades,
        max_dd=max_dd_filter,
    )

    skip_bootstrap = False
    if int(args.bootstrap_existing) > 0 and int(args.state_reset) <= 0:
        existing_profitable, existing_total = state.stats()
        if existing_total > 0:
            skip_bootstrap = True
            print(
                f"[BOOT] skip: using existing state_db={state_db_path} "
                f"unique_profitable={existing_profitable} total_strategies={existing_total}"
            )

    if int(args.bootstrap_existing) > 0 and not skip_bootstrap:
        candidates: list[Path] = []
        seen_ckpt: set[str] = set()
        if base_ckpt.exists():
            candidates.append(base_ckpt)
            seen_ckpt.add(str(base_ckpt.resolve()))
        if base_ckpt.parent.exists():
            for p in sorted(base_ckpt.parent.glob(f"{base_ckpt.stem}_*{base_ckpt.suffix}")):
                key = str(p.resolve())
                if key in seen_ckpt:
                    continue
                seen_ckpt.add(key)
                candidates.append(p)
        boot_ok = 0
        for p in candidates:
            bsym, btf = _checkpoint_symbol_tf(p)
            hd = 0.0
            hm = 0.0
            if bsym and btf:
                hd, hm = await _history_span_for_symbol_tf(
                    loader,
                    symbol=bsym,
                    timeframe=btf,
                    cache=history_span_cache,
                )
            _lp, lt, _cached = state.ingest_checkpoint(p, history_months=hm if hm > 0.0 else None, history_days=hd if hd > 0.0 else None)
            if lt > 0:
                boot_ok += 1
        p_now, t_now = state.stats()
        print(
            f"[BOOT] state_db={state_db_path} checkpoints={boot_ok} "
            f"unique_profitable={p_now} total_strategies={t_now}"
        )

    def _symbol_profitable(symbol: str) -> int:
        if target_profitable_per_symbol <= 0:
            return 0
        p, _ = state.stats_for_symbol(symbol)
        return int(p)

    def _all_symbols_target_hit() -> bool:
        if target_profitable_per_symbol <= 0:
            return False
        for s in symbols:
            if _symbol_profitable(s) < target_profitable_per_symbol:
                return False
        return True

    try:
        for sym in symbols:
            if target_profitable > 0:
                current_profitable, _ = state.stats()
                if current_profitable >= target_profitable:
                    break
            if _all_symbols_target_hit():
                break
            if target_profitable_per_symbol > 0:
                sym_prof0 = _symbol_profitable(sym)
                if sym_prof0 >= target_profitable_per_symbol:
                    print(
                        f"[SKIP] {sym}: already reached target profitable "
                        f"{sym_prof0}/{target_profitable_per_symbol}"
                    )
                    continue
            settings.system.symbol = sym
            frames = await loader.get_training_data(sym)
            if not isinstance(frames, dict) or not frames:
                print(f"[SKIP] {sym}: no history/frames")
                skipped += 1
                continue

            for tf in timeframes:
                if target_profitable > 0:
                    current_profitable, _ = state.stats()
                    if current_profitable >= target_profitable:
                        break
                if target_profitable_per_symbol > 0:
                    sym_prof0 = _symbol_profitable(sym)
                    if sym_prof0 >= target_profitable_per_symbol:
                        break
                df = frames.get(tf)
                if df is None or df.empty:
                    print(f"[SKIP] {sym} {tf}: empty frame")
                    skipped += 1
                    continue

                df = _with_lookback(df, lookback_days)
                rows_cap = default_rows
                if rows_cap <= 0 and lookback_days > 0:
                    rows_cap = _rows_for_days(tf, lookback_days)
                if rows_cap > 0 and len(df) > rows_cap:
                    df = df.tail(rows_cap)

                if len(df) < int(args.min_rows):
                    print(f"[SKIP] {sym} {tf}: only {len(df)} rows (< min_rows={int(args.min_rows)})")
                    skipped += 1
                    continue

                try:
                    df = df.copy()
                    df.attrs["symbol"] = sym
                    df.attrs["timeframe"] = tf
                    df.attrs["tf"] = tf
                except Exception:
                    pass
                history_days, history_months = _history_span_days_months(df)
                try:
                    last_close = _safe_float(df["close"].iloc[-1], 0.0)
                    if math.isfinite(last_close) and last_close > 0.0:
                        close_cache[(_norm_symbol(sym), str(tf).upper())] = float(last_close)
                except Exception:
                    pass

                pip_size = 0.0001
                pip_value_per_lot = 10.0
                pip_reference_prices: dict[str, float] = {}
                try:
                    last_close = _safe_float(df["close"].iloc[-1], 0.0)
                    if not (math.isfinite(last_close) and last_close > 0.0):
                        last_close = None
                    pip_reference_prices = await _resolve_reference_prices(
                        loader,
                        symbol=sym,
                        timeframe=tf,
                        symbol_universe=symbol_universe,
                        close_cache=close_cache,
                    )
                    pip_size, pip_value_per_lot = infer_pip_metrics(
                        sym,
                        price=last_close,
                        account_currency="USD",
                        reference_prices=pip_reference_prices,
                    )
                except Exception:
                    pip_size = 0.0001
                    pip_value_per_lot = 10.0
                    pip_reference_prices = {}

                try:
                    df.attrs["pip_size"] = float(pip_size)
                    df.attrs["pip_value_per_lot"] = float(pip_value_per_lot)
                    if pip_reference_prices:
                        df.attrs["pip_reference_prices"] = dict(pip_reference_prices)
                    if history_days > 0.0:
                        df.attrs["history_days"] = float(history_days)
                    if history_months > 0.0:
                        df.attrs["history_months"] = float(history_months)
                except Exception:
                    pass

                use_round_suffix = bool(int(args.rounds) > 1 or repeat_until_target)
                round_no = 1
                if repeat_until_target and int(args.resume_existing) > 0 and use_round_suffix:
                    round_no = max(1, _latest_round_index(base_ckpt, sym, tf) + 1)
                    if round_no > 1:
                        print(
                            f"[SEEK] {sym} {tf}: resume from round={round_no} "
                            f"(latest existing={round_no-1})"
                        )

                while True:
                    if target_profitable > 0:
                        current_profitable, _ = state.stats()
                        if current_profitable >= target_profitable:
                            break
                    if target_profitable_per_symbol > 0:
                        sym_prof_now = _symbol_profitable(sym)
                        if sym_prof_now >= target_profitable_per_symbol:
                            break
                    if not repeat_until_target and round_no > max(1, int(args.rounds)):
                        break

                    ckpt = Path(str(settings.models.prop_search_checkpoint or "models/strategy_evo_checkpoint.json"))
                    if use_round_suffix:
                        ckpt = ckpt.with_name(f"{ckpt.stem}_{sym}_{tf}_r{round_no}{ckpt.suffix}")
                    else:
                        ckpt = ckpt.with_name(f"{ckpt.stem}_{sym}_{tf}{ckpt.suffix}")

                    run_pop = int(settings.models.prop_search_population)
                    run_gen = int(settings.models.prop_search_generations)
                    if str(tf).upper() == "M1":
                        if int(args.m1_population) > 0:
                            run_pop = int(args.m1_population)
                        if int(args.m1_generations) > 0:
                            run_gen = int(args.m1_generations)
                    hb_stages: list[dict[str, float | int | str]] = []
                    if challenge_mode:
                        hb_pop_mults = [float(x) for x in _parse_number_list(args.hb_stage_pop_mults, float, [0.35, 0.70, 1.00])]
                        hb_gen_mults = [float(x) for x in _parse_number_list(args.hb_stage_gen_mults, float, [0.50, 0.75, 1.00])]
                        hb_hour_mults = [float(x) for x in _parse_number_list(args.hb_stage_hour_mults, float, [0.15, 0.40, 1.00])]
                        hb_promote = [int(x) for x in _parse_number_list(args.hb_promote_min, int, [1, 1, 0])]
                        hb_stages = _hyperband_stages(
                            base_population=run_pop,
                            base_generations=run_gen,
                            base_hours=float(settings.models.prop_search_max_hours),
                            pop_mults=hb_pop_mults,
                            gen_mults=hb_gen_mults,
                            hour_mults=hb_hour_mults,
                            promote_min=hb_promote,
                        )
                    seen_file = _seen_file_path(str(args.seen_dir), sym, tf)
                    seen_env_base: dict[str, str] = {
                        "FOREX_BOT_PROP_SEEN_FILE": str(seen_file),
                        "FOREX_BOT_PROP_SEEN_FLUSH_EVERY": str(int(args.seen_flush_every)),
                        "FOREX_BOT_PROP_SEEN_LOAD_MAX": str(int(args.seen_load_max)),
                        "FOREX_BOT_PROP_PIP_VALUE": f"{float(pip_size):.12g}",
                        "FOREX_BOT_PROP_PIP_VALUE_PER_LOT": f"{float(pip_value_per_lot):.12g}",
                    }
                    local_passed = 0
                    local_total = 0
                    resumed_main = False
                    if (not challenge_mode) and int(args.resume_existing) > 0 and ckpt.exists():
                        lp, lt, _cached = state.ingest_checkpoint(
                            ckpt,
                            history_months=history_months if history_months > 0.0 else None,
                            history_days=history_days if history_days > 0.0 else None,
                        )
                        if lt > 0:
                            resumed_main = True
                            local_passed = int(lp)
                            local_total = int(lt)
                            resumed += 1
                            written.append(ckpt)
                            print(
                                f"[RESM] {sym} {tf} r{round_no}: using existing checkpoint={ckpt} "
                                f"accepted={local_passed} total={local_total}"
                            )
                        else:
                            print(
                                f"[RESM] {sym} {tf} r{round_no}: checkpoint exists but invalid/empty, rerunning ({ckpt})"
                            )
                    if not resumed_main:
                        if challenge_mode and hb_stages:
                            for stage_idx, stage in enumerate(hb_stages):
                                stage_suffix = str(stage.get("suffix", f"hb{stage_idx}"))
                                stage_pop = int(stage.get("population", run_pop))
                                stage_gen = int(stage.get("generations", run_gen))
                                stage_hours = float(stage.get("max_hours", float(settings.models.prop_search_max_hours)))
                                stage_promote_min = int(stage.get("promote_min", 0))
                                stage_ckpt = ckpt.with_name(f"{ckpt.stem}_{stage_suffix}{ckpt.suffix}")

                                stage_passed = 0
                                stage_total = 0
                                resumed_stage = False
                                if int(args.resume_existing) > 0 and stage_ckpt.exists():
                                    slp, slt, _cached = state.ingest_checkpoint(
                                        stage_ckpt,
                                        history_months=history_months if history_months > 0.0 else None,
                                        history_days=history_days if history_days > 0.0 else None,
                                    )
                                    if slt > 0:
                                        resumed_stage = True
                                        stage_passed = int(slp)
                                        stage_total = int(slt)
                                        resumed += 1
                                        written.append(stage_ckpt)
                                        print(
                                            f"[HBND] {sym} {tf} r{round_no} stage={stage_suffix}: resumed "
                                            f"accepted={stage_passed} total={stage_total} checkpoint={stage_ckpt}"
                                        )
                                if not resumed_stage:
                                    print(
                                        f"[HBND] {sym} {tf} r{round_no} stage={stage_suffix}: rows={len(df):,} "
                                        f"pop={stage_pop} gen={stage_gen} max_hours={stage_hours:.4f} "
                                        f"pip={pip_size:.6g} pip_val_lot={pip_value_per_lot:.6g} ckpt={stage_ckpt}"
                                    )
                                    env_prev = _apply_env(seen_env_base)
                                    try:
                                        run_evo_search(
                                            df=df,
                                            settings=settings,
                                            population=stage_pop,
                                            generations=stage_gen,
                                            checkpoint=str(stage_ckpt),
                                            max_hours=float(stage_hours),
                                            actual_balance=float(
                                                getattr(settings.risk, "initial_balance", 10_000.0) or 10_000.0
                                            ),
                                            max_workers=int(args.workers) if int(args.workers) > 0 else None,
                                        )
                                    finally:
                                        _restore_env(env_prev)
                                    ran += 1
                                    written.append(stage_ckpt)
                                    slp, slt, _cached = state.ingest_checkpoint(
                                        stage_ckpt,
                                        history_months=history_months if history_months > 0.0 else None,
                                        history_days=history_days if history_days > 0.0 else None,
                                    )
                                    stage_passed = int(slp)
                                    stage_total = int(slt)
                                local_passed = int(stage_passed)
                                local_total = int(stage_total)
                                if stage_promote_min > 0 and stage_passed < stage_promote_min:
                                    print(
                                        f"[HBND] {sym} {tf} r{round_no} stage={stage_suffix}: "
                                        f"pruned (accepted={stage_passed} < promote_min={stage_promote_min})"
                                    )
                                    break
                        else:
                            print(
                                f"[RUN ] {sym} {tf} r{round_no}: rows={len(df):,} pop={run_pop} gen={run_gen} "
                                f"pip={pip_size:.6g} pip_val_lot={pip_value_per_lot:.6g} "
                                f"seen={seen_file} checkpoint={ckpt}"
                            )
                            env_prev = _apply_env(seen_env_base)
                            try:
                                run_evo_search(
                                    df=df,
                                    settings=settings,
                                    population=run_pop,
                                    generations=run_gen,
                                    checkpoint=str(ckpt),
                                    max_hours=float(settings.models.prop_search_max_hours),
                                    actual_balance=float(getattr(settings.risk, "initial_balance", 10_000.0) or 10_000.0),
                                    max_workers=int(args.workers) if int(args.workers) > 0 else None,
                                )
                            finally:
                                _restore_env(env_prev)
                            ran += 1
                            written.append(ckpt)
                            lp, lt, _cached = state.ingest_checkpoint(
                                ckpt,
                                history_months=history_months if history_months > 0.0 else None,
                                history_days=history_days if history_days > 0.0 else None,
                            )
                            local_passed = int(lp)
                            local_total = int(lt)
                    if local_passed <= 0 and int(args.adaptive_retries) > 0:
                        base_smc_gate = _safe_float(os.environ.get("FOREX_BOT_PROP_SMC_GATE", "0.75"), 0.75)
                        base_immigrants = _safe_float(
                            os.environ.get("FOREX_BOT_PROP_RANDOM_IMMIGRANTS", "0.35"),
                            0.35,
                        )
                        env_max_ind = _safe_int(os.environ.get("FOREX_BOT_PROP_SEARCH_MAX_INDICATORS", "0"), 0)
                        cfg_max_ind = _safe_int(getattr(settings.models, "prop_search_max_indicators", 0), 0)
                        base_max_ind = env_max_ind if env_max_ind > 0 else cfg_max_ind
                        if base_max_ind <= 0 and max_indicator_universe > 0:
                            base_max_ind = max(2, min(max_indicator_universe, 16))
                        for retry_idx in range(int(args.adaptive_retries)):
                            retry_ckpt = ckpt.with_name(f"{ckpt.stem}_ax{retry_idx+1}{ckpt.suffix}")
                            retry_pop = max(
                                run_pop + 1,
                                int(math.ceil(run_pop * (float(args.adaptive_pop_mult) ** float(retry_idx + 1)))),
                            )
                            retry_gen = max(
                                run_gen + 1,
                                run_gen + int(args.adaptive_gen_step) * int(retry_idx + 1),
                            )
                            retry_pop = min(retry_pop, int(args.adaptive_max_pop))
                            retry_gen = min(retry_gen, int(args.adaptive_max_gen))
                            retry_step = retry_idx + 1
                            retry_smc_gate = max(
                                float(args.adaptive_min_smc_gate),
                                base_smc_gate - float(args.adaptive_smc_step) * float(retry_step),
                            )
                            retry_immigrants = min(0.9, base_immigrants + 0.1 * float(retry_step))
                            retry_include_raw = retry_step >= int(args.adaptive_include_raw_after)
                            retry_max_ind = 0
                            if int(args.adaptive_indicator_step) > 0 and max_indicator_universe > 0:
                                retry_max_ind = min(
                                    max_indicator_universe,
                                    base_max_ind + int(args.adaptive_indicator_step) * int(retry_step),
                                )
                            env_overrides: dict[str, str] = {
                                "FOREX_BOT_PROP_SMC_GATE": f"{retry_smc_gate:.4f}",
                                "FOREX_BOT_PROP_RANDOM_IMMIGRANTS": f"{retry_immigrants:.3f}",
                                "FOREX_BOT_PROP_INCLUDE_RAW_FEATURES": "1" if retry_include_raw else "0",
                            }
                            env_overrides.update(seen_env_base)
                            if retry_max_ind > 0:
                                env_overrides["FOREX_BOT_PROP_SEARCH_MAX_INDICATORS"] = str(int(retry_max_ind))
                            print(
                                f"[ADPT] {sym} {tf} r{round_no} retry={retry_idx+1}: "
                                f"pop={retry_pop} gen={retry_gen} smc_gate={retry_smc_gate:.3f} "
                                f"immigrants={retry_immigrants:.2f} include_raw={int(retry_include_raw)} "
                                f"max_ind={retry_max_ind if retry_max_ind > 0 else 'base'} checkpoint={retry_ckpt}"
                            )
                            retry_passed = 0
                            retry_total = 0
                            resumed_retry = False
                            if int(args.resume_existing) > 0 and retry_ckpt.exists():
                                rlp, rlt, _cached = state.ingest_checkpoint(
                                    retry_ckpt,
                                    history_months=history_months if history_months > 0.0 else None,
                                    history_days=history_days if history_days > 0.0 else None,
                                )
                                if rlt > 0:
                                    resumed_retry = True
                                    retry_passed = int(rlp)
                                    retry_total = int(rlt)
                                    resumed += 1
                                    written.append(retry_ckpt)
                                    print(
                                        f"[RESM] {sym} {tf} r{round_no} retry={retry_idx+1}: "
                                        f"using existing checkpoint={retry_ckpt} accepted={retry_passed} total={retry_total}"
                                    )
                            if not resumed_retry:
                                env_prev = _apply_env(env_overrides)
                                try:
                                    run_evo_search(
                                        df=df,
                                        settings=settings,
                                        population=retry_pop,
                                        generations=retry_gen,
                                        checkpoint=str(retry_ckpt),
                                        max_hours=float(settings.models.prop_search_max_hours),
                                        actual_balance=float(getattr(settings.risk, "initial_balance", 10_000.0) or 10_000.0),
                                        max_workers=int(args.workers) if int(args.workers) > 0 else None,
                                    )
                                finally:
                                    _restore_env(env_prev)
                                ran += 1
                                written.append(retry_ckpt)
                                rlp, rlt, _cached = state.ingest_checkpoint(
                                    retry_ckpt,
                                    history_months=history_months if history_months > 0.0 else None,
                                    history_days=history_days if history_days > 0.0 else None,
                                )
                                retry_passed = int(rlp)
                                retry_total = int(rlt)
                            print(
                                f"[ADPT] {sym} {tf} r{round_no} retry={retry_idx+1} "
                                f"accepted={retry_passed}"
                            )
                            if retry_passed > 0:
                                break

                    profitable_now, total_now = state.stats()
                    sym_prof_now = _symbol_profitable(sym) if target_profitable_per_symbol > 0 else 0
                    print(
                        f"[STAT] unique profitable={profitable_now} threshold>{min_profit:g} key={profit_key} "
                        f"min_trades>={min_profit_trades:g} min_tpm>={min_trades_per_month_filter:g} "
                        f"min_monthly_pct>={min_monthly_profit_pct_filter:g} min_sharpe>={min_sharpe_filter:g} "
                        f"min_win_rate>={min_win_rate_filter:g} min_pf>={min_profit_factor_filter:g} "
                        f"max_dd<={max_dd_filter if max_dd_filter is not None else 'inf'} "
                        f"symbol={sym} sym_profitable={sym_prof_now}/{target_profitable_per_symbol if target_profitable_per_symbol > 0 else 0} "
                        f"total_strategies={total_now}"
                    )
                    if target_profitable > 0 and profitable_now >= target_profitable:
                        break
                    if target_profitable_per_symbol > 0 and sym_prof_now >= target_profitable_per_symbol:
                        print(
                            f"[DONE] {sym}: reached target profitable "
                            f"{sym_prof_now}/{target_profitable_per_symbol}"
                        )
                        break
                    round_no += 1
    finally:
        profitable_final, total_final = state.stats()
        state.close()
    print(
        f"Done. completed={ran} resumed={resumed} skipped={skipped} "
        f"unique_profitable={profitable_final} total_strategies={total_final} state_db={state_db_path}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prop discovery only (no model training).")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Default: auto-discover from data dir.")
    parser.add_argument("--timeframes", default="M1", help="Comma-separated timeframes (e.g. M1,M5,H1).")
    parser.add_argument("--lookback-days", type=int, default=30, help="Use only latest N days per timeframe.")
    parser.add_argument("--max-rows", type=int, default=0, help="Hard cap rows per timeframe (0=auto-from-lookback).")
    parser.add_argument("--min-rows", type=int, default=500, help="Skip timeframe if rows below this.")
    parser.add_argument("--population", type=int, default=100, help="Prop search population.")
    parser.add_argument("--generations", type=int, default=20, help="Prop search generations.")
    parser.add_argument("--m1-population", type=int, default=0, help="Override population only for M1 (0=disabled).")
    parser.add_argument("--m1-generations", type=int, default=0, help="Override generations only for M1 (0=disabled).")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat each symbol/timeframe this many times.")
    parser.add_argument(
        "--challenge-mode",
        type=int,
        default=0,
        help="Enable prop-challenge preset: stricter anti-overfit filters + Hyperband early-stop stages.",
    )
    parser.add_argument(
        "--challenge-min-trades",
        type=float,
        default=30.0,
        help="Minimum trades required for challenge-mode profitable counting/filtering.",
    )
    parser.add_argument(
        "--challenge-min-trades-per-month",
        type=float,
        default=10.0,
        help="Challenge-mode floor for strategy activity (trades/month over history).",
    )
    parser.add_argument(
        "--challenge-min-monthly-profit-pct",
        type=float,
        default=0.015,
        help="Challenge-mode floor for estimated monthly return (net_profit/(initial_balance*months)).",
    )
    parser.add_argument(
        "--challenge-min-sharpe",
        type=float,
        default=1.2,
        help="Challenge-mode floor for Sharpe ratio.",
    )
    parser.add_argument(
        "--challenge-min-win-rate",
        type=float,
        default=0.52,
        help="Challenge-mode floor for win rate (0-1).",
    )
    parser.add_argument(
        "--challenge-min-profit-factor",
        type=float,
        default=1.3,
        help="Challenge-mode floor for profit factor.",
    )
    parser.add_argument(
        "--holdout-years",
        type=float,
        default=0.0,
        help="Use strict calendar holdout on latest N years (0 disables year-based holdout).",
    )
    parser.add_argument(
        "--holdout-from",
        default="",
        help="Use strict forward holdout starting from this UTC date/time (e.g. 2025-08-01).",
    )
    parser.add_argument(
        "--min-truth-probability",
        type=float,
        default=0.0,
        help="Require truth_probability >= value for profitable counting/selection (0-1 or 0-100).",
    )
    parser.add_argument(
        "--forward-test-required",
        type=int,
        default=-1,
        help="Require forward_test_passed for profitable counting (1=yes, 0=no, -1 keep env/default).",
    )
    parser.add_argument(
        "--journal-top-k",
        type=int,
        default=10,
        help="Compute detailed per-trade/per-month journals for top K selected strategies (0 disables).",
    )
    parser.add_argument(
        "--swap-long-per-day",
        type=float,
        default=0.0,
        help="Estimated long swap cost per day (account currency, 1 lot) for journal metrics.",
    )
    parser.add_argument(
        "--swap-short-per-day",
        type=float,
        default=0.0,
        help="Estimated short swap cost per day (account currency, 1 lot) for journal metrics.",
    )
    parser.add_argument(
        "--hb-stage-pop-mults",
        default="0.35,0.70,1.00",
        help="Comma-separated population multipliers per Hyperband stage.",
    )
    parser.add_argument(
        "--hb-stage-gen-mults",
        default="0.50,0.75,1.00",
        help="Comma-separated generation multipliers per Hyperband stage.",
    )
    parser.add_argument(
        "--hb-stage-hour-mults",
        default="0.15,0.40,1.00",
        help="Comma-separated max-hours multipliers per Hyperband stage.",
    )
    parser.add_argument(
        "--hb-promote-min",
        default="1,1,0",
        help="Min accepted strategies needed to promote between Hyperband stages.",
    )
    parser.add_argument(
        "--repeat-until-target",
        type=int,
        default=1,
        help=(
            "When target-profitable or target-profitable-per-symbol is set, "
            "keep generating new rounds until target reached (1=yes, 0=no)."
        ),
    )
    parser.add_argument("--max-hours", type=float, default=1.0, help="Per symbol/timeframe time limit.")
    parser.add_argument("--max-indicators", type=int, default=0, help="0 = full indicator set.")
    parser.add_argument("--adaptive-retries", type=int, default=2, help="Extra exploration retries if a run yields 0 accepted.")
    parser.add_argument("--adaptive-pop-mult", type=float, default=1.8, help="Population multiplier per adaptive retry.")
    parser.add_argument("--adaptive-gen-step", type=int, default=1, help="Generations added per adaptive retry.")
    parser.add_argument("--adaptive-max-pop", type=int, default=5000, help="Hard cap population in adaptive retries.")
    parser.add_argument("--adaptive-max-gen", type=int, default=14, help="Hard cap generations in adaptive retries.")
    parser.add_argument(
        "--adaptive-smc-step",
        type=float,
        default=0.1,
        help="SMC gate decrement per retry (lower = less strict SMC gating).",
    )
    parser.add_argument(
        "--adaptive-min-smc-gate",
        type=float,
        default=0.45,
        help="Floor for adaptive SMC gate.",
    )
    parser.add_argument(
        "--adaptive-indicator-step",
        type=int,
        default=8,
        help="Increase max indicators per retry (0 disables).",
    )
    parser.add_argument(
        "--adaptive-include-raw-after",
        type=int,
        default=1,
        help="Enable raw OHLCV-derived features starting from this retry index.",
    )
    parser.add_argument(
        "--seen-dir",
        default="cache/prop_seen_hashes",
        help="Directory for persistent Rust seen-signature files (per symbol/timeframe).",
    )
    parser.add_argument(
        "--seen-flush-every",
        type=int,
        default=4096,
        help="Flush new seen signatures to disk every N inserts.",
    )
    parser.add_argument(
        "--seen-load-max",
        type=int,
        default=3000000,
        help="Max signatures loaded from seen file into RAM at startup (0 = unlimited).",
    )
    parser.add_argument("--workers", type=int, default=0, help="CPU workers hint for prop search (0=auto).")
    parser.add_argument("--target-profitable", type=int, default=0, help="Stop early when unique profitable count reaches this.")
    parser.add_argument(
        "--target-profitable-per-symbol",
        type=int,
        default=0,
        help="Stop only after each symbol reaches this many unique profitable strategies (0=disabled).",
    )
    parser.add_argument("--profit-key", default="net_profit", help="Gene field to use for profitability count.")
    parser.add_argument("--profit-threshold", type=float, default=0.0, help="Profitable if key is > threshold.")
    parser.add_argument("--profit-min-trades", type=float, default=1.0, help="Require at least this many trades.")
    parser.add_argument("--max-dd", default="", help="Optional max drawdown filter for counted strategies (e.g. 0.07).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "auto"], help="Prop search device.")
    parser.add_argument(
        "--checkpoint",
        default="models/strategy_evo_checkpoint.json",
        help="Base checkpoint file; script appends _<SYMBOL>_<TF>.",
    )
    parser.add_argument(
        "--state-db",
        default="",
        help="SQLite file for persistent discovery memory; default derived from checkpoint + filters.",
    )
    parser.add_argument(
        "--state-reset",
        type=int,
        default=0,
        help="Delete existing state DB before run (1=yes, 0=no).",
    )
    parser.add_argument(
        "--resume-existing",
        type=int,
        default=1,
        help="Reuse existing checkpoint files to resume after interruption (1=yes, 0=no).",
    )
    parser.add_argument(
        "--bootstrap-existing",
        type=int,
        default=1,
        help="Index existing checkpoint files into state DB before run (1=yes, 0=no).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
