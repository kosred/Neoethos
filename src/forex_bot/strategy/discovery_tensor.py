from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import forex_bindings as _fb  # type: ignore

    _RUST_DISCOVERY = hasattr(_fb, "search_discovery_ohlcv")
    _RUST_TALIB_POP = hasattr(_fb, "evaluate_population_talib_ohlcv")
except Exception:
    _fb = None  # type: ignore
    _RUST_DISCOVERY = False
    _RUST_TALIB_POP = False


@dataclass(slots=True)
class DiscoveryGene:
    indicators: list[str]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    combination_method: str = "weighted_vote"
    long_threshold: float = 0.66
    short_threshold: float = -0.66
    weights: dict[str, float] = field(default_factory=dict)
    preferred_regime: str = "any"
    strategy_id: str = ""
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_dd_pct: float = 0.0
    trades: float = 0.0
    use_ob: bool = False
    use_fvg: bool = False
    use_liq_sweep: bool = False
    mtf_confirmation: bool = False
    use_premium_discount: bool = False
    use_inducement: bool = False
    use_bos: bool = False
    use_choch: bool = False
    use_eqh: bool = False
    use_eql: bool = False
    use_displacement: bool = False
    tp_pips: float = 40.0
    sl_pips: float = 20.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0


def _is_datetime_index(idx: Any) -> bool:
    if idx is None:
        return False
    if hasattr(idx, "year") and hasattr(idx, "month") and hasattr(idx, "day"):
        return True
    with np.errstate(all="ignore"):
        try:
            arr = np.asarray(idx).reshape(-1)
        except Exception:
            return False
    if arr.size <= 0:
        return False
    return bool(np.issubdtype(arr.dtype, np.datetime64))


def _index_to_ns_int64(idx: Any) -> np.ndarray:
    if idx is None:
        return np.zeros(0, dtype=np.int64)
    try:
        if hasattr(idx, "asi8"):
            return np.asarray(idx.asi8, dtype=np.int64).reshape(-1)
    except Exception:
        pass
    with np.errstate(all="ignore"):
        arr = np.asarray(idx).reshape(-1)
    if arr.size <= 0:
        return np.zeros(0, dtype=np.int64)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        if hasattr(idx, "view"):
            viewed = idx.view("int64")
            if hasattr(viewed, "to_numpy"):
                return np.asarray(viewed.to_numpy(dtype=np.int64, copy=False), dtype=np.int64).reshape(-1)
            return np.asarray(viewed, dtype=np.int64).reshape(-1)
        if arr.dtype.kind == "O":
            out = np.zeros(arr.size, dtype=np.int64)
            for i, value in enumerate(arr.tolist()):
                try:
                    ns = getattr(value, "value", None)
                    if ns is not None:
                        out[i] = int(ns)
                    else:
                        out[i] = int(np.datetime64(value, "ns").astype(np.int64))
                except Exception:
                    out[i] = 0
            return out
    except Exception:
        pass
    try:
        return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
    except Exception:
        return np.zeros(arr.size, dtype=np.int64)


def _rust_time_index_arrays(idx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "derive_time_index_arrays"):
        return None
    ns = _index_to_ns_int64(idx)
    if ns.size <= 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    try:
        unix_ms, month_idx, day_idx = _fb.derive_time_index_arrays(np.asarray(ns, dtype=np.int64))
    except Exception:
        return None
    return (
        np.asarray(unix_ms, dtype=np.int64).reshape(-1),
        np.asarray(month_idx, dtype=np.int64).reshape(-1),
        np.asarray(day_idx, dtype=np.int64).reshape(-1),
    )


def _safe_indices(idx: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    rust = _rust_time_index_arrays(idx)
    if rust is not None:
        _unix_ms, month_idx, day_idx = rust
        return month_idx[:n], day_idx[:n]
    if _is_datetime_index(idx):
        if hasattr(idx, "year") and hasattr(idx, "month") and hasattr(idx, "day"):
            month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(dtype=np.int64)
            day_idx = (
                idx.year.astype(np.int32) * 10000 + idx.month.astype(np.int32) * 100 + idx.day.astype(np.int32)
            ).to_numpy(dtype=np.int64)
            return month_idx[:n], day_idx[:n]
        with np.errstate(all="ignore"):
            arr = np.asarray(idx).reshape(-1)
        if arr.size > 0 and np.issubdtype(arr.dtype, np.datetime64):
            month_idx = arr.astype("datetime64[M]").astype(np.int64)
            day_idx = arr.astype("datetime64[D]").astype(np.int64)
            return month_idx[:n], day_idx[:n]
    ns = _index_to_ns_int64(idx)
    if ns.size > 0:
        vmax = int(np.max(np.abs(ns))) if ns.size > 0 else 0
        if vmax > 10**14:
            dt = np.asarray(ns, dtype=np.int64).astype("datetime64[ns]")
            month_idx = dt.astype("datetime64[M]").astype(np.int64)
            day_idx = dt.astype("datetime64[D]").astype(np.int64)
            return month_idx[:n], day_idx[:n]
    seq = np.arange(n, dtype=np.int64)
    return seq, seq


def _datetime_index_to_unix_ms(idx: Any) -> np.ndarray:
    rust = _rust_time_index_arrays(idx)
    if rust is not None:
        unix_ms, _month_idx, _day_idx = rust
        return unix_ms
    ns = _index_to_ns_int64(idx)
    if ns.size <= 0:
        return np.zeros(0, dtype=np.int64)
    return (np.asarray(ns, dtype=np.int64) // 1_000_000).astype(np.int64, copy=False)


def _frame_empty(frame: Any) -> bool:
    if frame is None:
        return True
    try:
        return bool(frame.empty)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return int(len(frame)) <= 0
    except Exception:
        return True


def _frame_len(frame: Any) -> int:
    try:
        return int(len(frame))
    except Exception:
        return 0


def _frame_copy(frame: Any) -> Any:
    if frame is None:
        return None
    try:
        return frame.copy()
    except Exception:
        return frame


def _frame_tail(frame: Any, n: int) -> Any:
    if frame is None:
        return None
    try:
        return frame.tail(int(n))
    except Exception:
        return frame


def _frame_attr(frame: Any, key: str, default: Any = None) -> Any:
    attrs = getattr(frame, "attrs", None)
    if isinstance(attrs, dict):
        return attrs.get(key, default)
    return default


def _frame_columns(frame: Any) -> list[str]:
    cols = getattr(frame, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _resolve_column_name(frame: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    if not target:
        return None
    for col in _frame_columns(frame):
        if str(col).strip().lower() == target:
            return col
    return None


def _to_numpy_1d(values: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        with np.errstate(all="ignore"):
            arr = values.to_numpy(dtype=dtype, copy=False)  # type: ignore[call-arg]
    else:
        arr = np.asarray(values, dtype=dtype)
    return np.asarray(arr, dtype=dtype).reshape(-1)


def _fit_len(values: Any, n: int, *, fill: float = 0.0, dtype: Any = np.float64) -> np.ndarray:
    out = _to_numpy_1d(values, dtype=dtype)
    target = max(0, int(n))
    if out.size == target:
        return out
    if out.size <= 0:
        return np.full(target, float(fill), dtype=dtype)
    if out.size > target:
        return out[:target]
    pad = np.full(target - out.size, float(out[-1]), dtype=dtype)
    return np.concatenate([out, pad])


def _frame_column_numpy(
    frame: Any,
    name: str,
    *,
    n_rows: int,
    dtype: Any = np.float64,
    default: Any | None = None,
) -> np.ndarray:
    col = _resolve_column_name(frame, name)
    if col is None:
        if default is None:
            raise KeyError(name)
        return _fit_len(default, n_rows, fill=0.0, dtype=dtype)
    values = frame[col]  # type: ignore[index]
    return _fit_len(values, n_rows, fill=0.0, dtype=dtype)


def _gene_to_dict(gene: Any) -> dict[str, Any]:
    return {
        "indicators": list(gene.indicators),
        "params": gene.params,
        "combination_method": gene.combination_method,
        "long_threshold": float(gene.long_threshold),
        "short_threshold": float(gene.short_threshold),
        "weights": gene.weights,
        "preferred_regime": gene.preferred_regime,
        "strategy_id": gene.strategy_id,
        "fitness": float(getattr(gene, "fitness", 0.0)),
        "sharpe_ratio": float(getattr(gene, "sharpe_ratio", 0.0)),
        "win_rate": float(getattr(gene, "win_rate", 0.0)),
        "net_profit": float(getattr(gene, "net_profit", 0.0)),
        "profit_factor": float(getattr(gene, "profit_factor", 0.0)),
        "expectancy": float(getattr(gene, "expectancy", 0.0)),
        "max_dd_pct": float(getattr(gene, "max_dd_pct", 0.0)),
        "max_drawdown": float(getattr(gene, "max_dd_pct", 0.0)),
        "trades": float(getattr(gene, "trades", 0.0)),
        "trades_count": float(getattr(gene, "trades", 0.0)),
        "use_ob": bool(getattr(gene, "use_ob", False)),
        "use_fvg": bool(getattr(gene, "use_fvg", False)),
        "use_liq_sweep": bool(getattr(gene, "use_liq_sweep", False)),
        "mtf_confirmation": bool(getattr(gene, "mtf_confirmation", False)),
        "use_premium_discount": bool(getattr(gene, "use_premium_discount", False)),
        "use_inducement": bool(getattr(gene, "use_inducement", False)),
        "use_bos": bool(getattr(gene, "use_bos", False)),
        "use_choch": bool(getattr(gene, "use_choch", False)),
        "use_eqh": bool(getattr(gene, "use_eqh", False)),
        "use_eql": bool(getattr(gene, "use_eql", False)),
        "use_displacement": bool(getattr(gene, "use_displacement", False)),
        "tp_pips": float(getattr(gene, "tp_pips", 40.0)),
        "sl_pips": float(getattr(gene, "sl_pips", 20.0)),
    }


def _feature_to_indicator(name: str, available: set[str]) -> str | None:
    if not name:
        return None
    raw = str(name).strip()
    if raw.lower().startswith("ta_"):
        raw = raw[3:]
    cand = raw.upper()
    if cand.startswith("SMC_"):
        return cand
    if not available:
        return cand.split("_")[0] if "_" in cand else cand
    if cand in available:
        return cand
    base = cand.split("_")[0]
    if base in available:
        return base
    return None


def _convert_rust_gene(gene: dict[str, Any], feature_names: list[str], available: set[str]) -> DiscoveryGene | None:
    indices = gene.get("indices") or []
    weights = gene.get("weights") or []
    indicators: list[str] = []
    weight_map: dict[str, float] = {}
    params: dict[str, dict[str, Any]] = {}

    for idx, w in zip(indices, weights):
        try:
            i = int(idx)
        except Exception:
            continue
        if i < 0 or i >= len(feature_names):
            continue
        ind = _feature_to_indicator(feature_names[i], available)
        if not ind:
            continue
        indicators.append(ind)
        weight_map[ind] = float(weight_map.get(ind, 0.0) + float(w))
        params.setdefault(ind, {})

    if not indicators:
        return None

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    max_dd_pct = _to_float(
        gene.get(
            "max_dd_pct",
            gene.get("max_drawdown", gene.get("max_dd", gene.get("drawdown", 0.0))),
        ),
        0.0,
    )
    trades = _to_float(gene.get("trades", gene.get("trades_count", gene.get("trade_count", 0.0))), 0.0)
    net_profit = _to_float(gene.get("net_profit", 0.0), 0.0)
    profit_factor = _to_float(gene.get("profit_factor", 0.0), 0.0)
    expectancy = _to_float(gene.get("expectancy", 0.0), 0.0)

    return DiscoveryGene(
        indicators=indicators,
        params=params,
        weights=weight_map,
        long_threshold=float(gene.get("long_threshold", 0.66)),
        short_threshold=float(gene.get("short_threshold", -0.66)),
        combination_method=str(gene.get("combination_method", "weighted_vote")),
        preferred_regime=str(gene.get("preferred_regime", "any")),
        strategy_id=str(gene.get("strategy_id", "")),
        fitness=float(gene.get("fitness", 0.0)),
        sharpe_ratio=_to_float(gene.get("sharpe_ratio", 0.0), 0.0),
        win_rate=_to_float(gene.get("win_rate", 0.0), 0.0),
        max_dd_pct=max_dd_pct,
        trades=trades,
        net_profit=net_profit,
        profit_factor=profit_factor,
        expectancy=expectancy,
        use_ob=bool(gene.get("use_ob", False)),
        use_fvg=bool(gene.get("use_fvg", False)),
        use_liq_sweep=bool(gene.get("use_liq_sweep", False)),
        mtf_confirmation=bool(gene.get("mtf_confirmation", False)),
        use_premium_discount=bool(gene.get("use_premium_discount", False)),
        use_inducement=bool(gene.get("use_inducement", False)),
        use_bos=bool(gene.get("use_bos", False)),
        use_choch=bool(gene.get("use_choch", False)),
        use_eqh=bool(gene.get("use_eqh", False)),
        use_eql=bool(gene.get("use_eql", False)),
        use_displacement=bool(gene.get("use_displacement", False)),
        tp_pips=float(gene.get("tp_pips", 40.0)),
        sl_pips=float(gene.get("sl_pips", 20.0)),
    )


def _gene_key(gene: Any) -> str:
    sid = str(getattr(gene, "strategy_id", "") or "").strip()
    if sid:
        return f"id:{sid}"
    return (
        f"sig:{tuple(gene.indicators)}|{gene.combination_method}|"
        f"{float(gene.long_threshold):.6f}|{float(gene.short_threshold):.6f}"
    )


def _dedupe_ranked(genes: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for gene in sorted(
        genes,
        key=lambda g: (
            float(getattr(g, "fitness", 0.0) or 0.0),
            float(getattr(g, "sharpe_ratio", 0.0) or 0.0),
            float(getattr(g, "win_rate", 0.0) or 0.0),
        ),
        reverse=True,
    ):
        key = _gene_key(gene)
        if key in seen:
            continue
        seen.add(key)
        out.append(gene)
    return out


def _strategy_keep_limits(settings: Any | None, default_cap: int) -> tuple[float, float, float, int, int]:
    try:
        risk_dd = getattr(settings.risk, "total_drawdown_limit", 0.07) if settings is not None else 0.07
        keep_max_dd = float(
            os.environ.get(
                "FOREX_BOT_DISCOVERY_KEEP_MAX_DD",
                risk_dd,
            )
            or 0.07
        )
    except Exception:
        keep_max_dd = 0.07
    keep_max_dd = float(min(1.0, max(0.0, keep_max_dd)))

    try:
        keep_min_profit = float(os.environ.get("FOREX_BOT_DISCOVERY_KEEP_MIN_PROFIT", "0.0") or 0.0)
    except Exception:
        keep_min_profit = 0.0

    try:
        keep_min_trades = float(os.environ.get("FOREX_BOT_DISCOVERY_KEEP_MIN_TRADES", "1.0") or 1.0)
    except Exception:
        keep_min_trades = 1.0
    keep_min_trades = float(max(0.0, keep_min_trades))

    try:
        keep_min_count = int(os.environ.get("FOREX_BOT_DISCOVERY_KEEP_MIN_COUNT", "100") or 100)
    except Exception:
        keep_min_count = 100
    keep_min_count = max(0, keep_min_count)

    fallback_cap = max(1, int(default_cap or 1))
    try:
        keep_cap = int(os.environ.get("FOREX_BOT_DISCOVERY_PORTFOLIO", str(fallback_cap)) or fallback_cap)
    except Exception:
        keep_cap = fallback_cap
    keep_cap = max(1, keep_cap)
    if keep_min_count > keep_cap:
        keep_min_count = keep_cap

    return keep_max_dd, keep_min_profit, keep_min_trades, keep_min_count, keep_cap


def _strategy_quality_limits(settings: Any | None) -> tuple[float, float, float]:
    try:
        default_sharpe = float(
            getattr(getattr(settings, "models", None), "prop_search_holdout_min_sharpe", 1.0) or 1.0
        )
    except Exception:
        default_sharpe = 1.0
    try:
        default_pf = float(
            getattr(getattr(settings, "models", None), "prop_search_holdout_min_profit_factor", 1.20) or 1.20
        )
    except Exception:
        default_pf = 1.20

    try:
        min_sharpe = float(os.environ.get("FOREX_BOT_DISCOVERY_MIN_SHARPE", str(default_sharpe)) or default_sharpe)
    except Exception:
        min_sharpe = default_sharpe
    try:
        min_profit_factor = float(os.environ.get("FOREX_BOT_DISCOVERY_MIN_PROFIT_FACTOR", str(default_pf)) or default_pf)
    except Exception:
        min_profit_factor = default_pf
    try:
        min_win_rate = float(os.environ.get("FOREX_BOT_DISCOVERY_MIN_WIN_RATE", "0.0") or 0.0)
    except Exception:
        min_win_rate = 0.0

    min_sharpe = float(max(-10.0, min(10.0, min_sharpe)))
    min_profit_factor = float(max(0.0, min(10.0, min_profit_factor)))
    min_win_rate = float(max(0.0, min(1.0, min_win_rate)))
    return min_sharpe, min_profit_factor, min_win_rate


def _passes_quality(
    gene: Any,
    *,
    min_sharpe: float,
    min_profit_factor: float,
    min_win_rate: float,
) -> bool:
    try:
        sharpe = float(getattr(gene, "sharpe_ratio", 0.0) or 0.0)
    except Exception:
        sharpe = 0.0
    try:
        profit_factor = float(getattr(gene, "profit_factor", 0.0) or 0.0)
    except Exception:
        profit_factor = 0.0
    try:
        win_rate = float(getattr(gene, "win_rate", 0.0) or 0.0)
    except Exception:
        win_rate = 0.0
    if win_rate > 1.0:
        win_rate *= 0.01

    return (
        sharpe >= float(min_sharpe)
        and profit_factor >= float(min_profit_factor)
        and win_rate >= float(min_win_rate)
    )


def _profit_value(gene: Any) -> float:
    metric = str(os.environ.get("FOREX_BOT_DISCOVERY_KEEP_PROFIT_METRIC", "fitness") or "fitness").strip().lower()
    if metric in {"net", "net_profit", "pnl"}:
        try:
            return float(getattr(gene, "net_profit", 0.0) or 0.0)
        except Exception:
            return 0.0
    try:
        return float(getattr(gene, "fitness", 0.0) or 0.0)
    except Exception:
        return 0.0


def _select_ranked(
    candidates: list[Any],
    *,
    filtered: list[Any],
    min_keep: int,
    cap: int,
) -> tuple[list[Any], int, int]:
    ranked_all = _dedupe_ranked(candidates)
    ranked_filtered = _dedupe_ranked(filtered) if filtered else []
    selected = list(ranked_filtered)
    if min_keep > 0 and len(selected) < min_keep:
        seen = {_gene_key(g) for g in selected}
        for gene in ranked_all:
            key = _gene_key(gene)
            if key in seen:
                continue
            selected.append(gene)
            seen.add(key)
            if len(selected) >= min_keep:
                break
    if not selected:
        selected = ranked_all
    if cap > 0:
        selected = selected[:cap]
    return selected, len(ranked_filtered), len(ranked_all)


class TensorDiscoveryEngine:
    def __init__(
        self,
        *,
        device: str = "cpu",
        n_experts: int = 100,
        timeframes: list[str] | None = None,
        max_rows: int = 0,
        stream_mode: bool = False,
        auto_cap: bool = True,
        settings: Any | None = None,
    ) -> None:
        self.device = device
        self.n_experts = int(n_experts or 0)
        self.timeframes = timeframes or []
        self.max_rows = int(max_rows or 0)
        self.stream_mode = bool(stream_mode)
        self.auto_cap = bool(auto_cap)
        self.settings = settings
        self._last_payload: dict[str, Any] | None = None

    def run_unsupervised_search(
        self,
        frames: dict[str, Any],
        *,
        iterations: int = 1000,
        news_features: Any | None = None,
    ) -> None:
        if frames is None or len(frames) == 0:
            return
        base_tf = self.timeframes[0] if self.timeframes else next(iter(frames.keys()))
        base_df = frames.get(base_tf)
        if _frame_empty(base_df):
            return
        df = _frame_copy(base_df)
        if self.max_rows > 0 and _frame_len(df) > self.max_rows:
            df = _frame_tail(df, self.max_rows)
        n_rows = _frame_len(df)
        if n_rows < 50:
            return
        iter_budget = max(1, int(iterations or 1))

        def _env_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or str(raw).strip() == "":
                return int(default)
            try:
                return int(raw)
            except Exception:
                return int(default)

        if _RUST_DISCOVERY and _fb is not None:
            try:
                ts = None
                idx = getattr(df, "index", None)
                if _is_datetime_index(idx):
                    ts = _datetime_index_to_unix_ms(idx)
                close = _frame_column_numpy(df, "close", n_rows=n_rows, dtype=np.float64)
                high = _frame_column_numpy(df, "high", n_rows=n_rows, dtype=np.float64)
                low = _frame_column_numpy(df, "low", n_rows=n_rows, dtype=np.float64)
                open_ = _frame_column_numpy(df, "open", n_rows=n_rows, dtype=np.float64, default=close)
                volume = None
                if _resolve_column_name(df, "volume") is not None:
                    volume = _frame_column_numpy(df, "volume", n_rows=n_rows, dtype=np.float64)

                default_pop = max(8, min(100, iter_budget))
                default_gens = max(1, min(5, (iter_budget + 19) // 20))
                default_candidates = max(10, min(200, default_pop * 2))
                default_portfolio = 3000
                if self.settings is not None:
                    try:
                        default_portfolio = int(
                            getattr(self.settings.models, "prop_search_portfolio_size", default_portfolio)
                            or default_portfolio
                        )
                    except Exception:
                        default_portfolio = 3000
                default_portfolio = max(1, default_portfolio)
                default_candidates = max(default_candidates, min(10_000, max(10, default_portfolio * 4)))

                keep_max_dd, keep_min_profit, keep_min_trades, keep_min_count, keep_cap = _strategy_keep_limits(
                    self.settings,
                    default_cap=default_portfolio,
                )
                min_sharpe, min_profit_factor, min_win_rate = _strategy_quality_limits(self.settings)
                rust_pop = max(4, _env_int("FOREX_BOT_DISCOVERY_POP", default_pop))
                rust_gens = max(1, _env_int("FOREX_BOT_DISCOVERY_GENS", default_gens))
                rust_max_ind = max(2, _env_int("FOREX_BOT_DISCOVERY_MAX_INDICATORS", 12))
                rust_candidates = max(10, _env_int("FOREX_BOT_DISCOVERY_CANDIDATES", default_candidates))
                rust_portfolio = max(1, _env_int("FOREX_BOT_DISCOVERY_PORTFOLIO", default_portfolio))
                rust_corr = float(os.environ.get("FOREX_BOT_DISCOVERY_CORR", "0.7") or 0.7)
                rust_min_trades_day = float(os.environ.get("FOREX_BOT_DISCOVERY_MIN_TRADES", "1.0") or 1.0)
                try:
                    result = _fb.search_discovery_ohlcv(
                        open_,
                        high,
                        low,
                        close,
                        ts,
                        volume,
                        rust_pop,
                        rust_gens,
                        rust_max_ind,
                        rust_candidates,
                        rust_portfolio,
                        rust_corr,
                        rust_min_trades_day,
                        True,
                        keep_max_dd,
                        keep_min_profit,
                        keep_min_trades,
                        keep_min_count,
                        keep_cap,
                    )
                except TypeError:
                    # Backward compatibility for older bindings without Rust-side ranking arguments.
                    result = _fb.search_discovery_ohlcv(
                        open_,
                        high,
                        low,
                        close,
                        ts,
                        volume,
                        rust_pop,
                        rust_gens,
                        rust_max_ind,
                        rust_candidates,
                        rust_portfolio,
                        rust_corr,
                        rust_min_trades_day,
                        True,
                    )

                feature_names = list(result.get("feature_names") or [])
                portfolio = list(result.get("portfolio") or [])
                # Keep Rust path independent from Python TA-Lib imports.
                available: set[str] = set()
                best: list[Any] = []
                for g in portfolio:
                    if not isinstance(g, dict):
                        continue
                    gene = _convert_rust_gene(g, feature_names, available)
                    if gene:
                        best.append(gene)
                if not best:
                    raise RuntimeError("Rust discovery produced no usable genes")

                rust_ranked = bool(result.get("rust_ranked", False))
                if rust_ranked:
                    base_filtered = list(best)
                    quality_filtered = [
                        g
                        for g in base_filtered
                        if _passes_quality(
                            g,
                            min_sharpe=min_sharpe,
                            min_profit_factor=min_profit_factor,
                            min_win_rate=min_win_rate,
                        )
                    ]
                    selected, strict_kept, ranked_total = _select_ranked(
                        best,
                        filtered=quality_filtered,
                        min_keep=keep_min_count,
                        cap=keep_cap,
                    )
                else:
                    base_filtered = [
                        g
                        for g in best
                        if _profit_value(g) > keep_min_profit
                        and float(getattr(g, "max_dd_pct", 0.0) or 0.0) <= keep_max_dd
                        and float(getattr(g, "trades", 0.0) or 0.0) >= keep_min_trades
                    ]
                    quality_filtered = [
                        g
                        for g in base_filtered
                        if _passes_quality(
                            g,
                            min_sharpe=min_sharpe,
                            min_profit_factor=min_profit_factor,
                            min_win_rate=min_win_rate,
                        )
                    ]
                    selected, strict_kept, ranked_total = _select_ranked(
                        best,
                        filtered=quality_filtered,
                        min_keep=keep_min_count,
                        cap=keep_cap,
                    )
                symbol = str(_frame_attr(df, "symbol", "") or "")

                payload = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "timeframe": str(_frame_attr(df, "timeframe", _frame_attr(df, "tf", "")) or ""),
                    "best_genes": [_gene_to_dict(g) for g in selected],
                }
                out_dir = Path("cache")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "talib_knowledge.json"
                if symbol:
                    safe = "".join(c for c in symbol if c.isalnum() or c in ("-", "_"))
                    out_path = out_dir / f"talib_knowledge_{safe}.json"
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                self._last_payload = payload
                logger.info(
                    "Discovery (Rust): kept %s/%s genes (strict=%s, min_keep=%s) "
                    "(profit>%.3f, max_dd<=%.3f, trades>=%.0f, sharpe>=%.2f, pf>=%.2f, win>=%.2f). Wrote %s",
                    len(selected),
                    ranked_total,
                    strict_kept,
                    keep_min_count,
                    keep_min_profit,
                    keep_max_dd,
                    keep_min_trades,
                    min_sharpe,
                    min_profit_factor,
                    min_win_rate,
                    out_path,
                )
                return
            except Exception as exc:
                logger.error("Rust discovery failed; skipping Python fallback: %s", exc, exc_info=True)
                return
        else:
            logger.warning("Rust discovery backend unavailable; skipping discovery.")
            return

    def save_experts(self, path: str) -> None:
        """
        Backward-compatible artifact writer expected by TrainingService.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._last_payload or {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_genes": [],
        }
        try:
            import torch

            torch.save(payload, out_path)
        except Exception:
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Discovery: saved experts artifact %s", out_path)


__all__ = ["TensorDiscoveryEngine"]

